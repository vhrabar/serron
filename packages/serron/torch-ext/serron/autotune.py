"""
Autotuning for algorithm selection.

Times the algorithms against each other on this GPU to find where the separable path overtakes the direct kernels,
saves the result, and loads it back the first time a CUDA op runs.
"""

from __future__ import annotations

import itertools
import os
import threading
import warnings
from collections.abc import Sequence

import torch

from . import _tuning_cache
from ._algorithms import device_fingerprint, feasible_algorithms, set_separable_min_k
from ._cmake_ops import ops
from ._timing import time_ms
from .enums import Algorithm

__all__ = [
    "DEFAULT_DTYPES",
    "DEFAULT_KS",
    "HYSTERESIS",
    "apply_thresholds",
    "calibrate",
    "crossover",
    "ensure_loaded",
    "sweep_separable_threshold",
    "threshold_key",
]

HYSTERESIS = 0.05
"""A candidate replaces the heuristic only when it is at least this fraction faster."""

DEFAULT_KS = (3, 5, 7, 9, 11, 13, 15, 21, 31, 63)
"""Window lengths a sweep measures."""

DEFAULT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
"""Element types :func:`calibrate` measures unless told otherwise."""

_REFERENCE_SHAPE = (1, 3, 512, 512)
_BORDER = 1  # replicate


def threshold_key(dtype: torch.dtype, *, backward: bool) -> str:
    """
    Key of one threshold in a saved table.

    :param dtype: element type of the input.
    :param backward: whether the threshold is for the backward pass.
    :returns: for example ``"float32:fwd"``.
    """
    return f"{str(dtype).removeprefix('torch.')}:{'bwd' if backward else 'fwd'}"


def crossover(
    ks: Sequence[int],
    separable_ms: Sequence[float],
    direct_ms: Sequence[float],
    margin: float = HYSTERESIS,
) -> int | None:
    """
    Window length from which the separable path should be used.

    Separable wins at a length when it is at least ``margin`` faster than the best direct kernel there. The answer is
    the smallest length whose win holds for every longer one, so a noisy win in the middle of a range that is
    otherwise lost does not move the threshold down.

    :param ks: measured window lengths, in ascending order.
    :param separable_ms: separable time at each length.
    :param direct_ms: best direct time at each length.
    :param margin: fraction by which separable must be faster to count as a win.
    :returns: the window length, or ``None`` if separable does not win at the longest one measured.
    :raises ValueError: if the sequences differ in length or ``ks`` is not ascending.
    """
    if not len(ks) == len(separable_ms) == len(direct_ms):
        raise ValueError("ks, separable_ms and direct_ms must have the same length")
    if any(a >= b for a, b in itertools.pairwise(ks)):
        raise ValueError("ks must be strictly ascending")

    lasting: int | None = None
    for k, sep, direct in zip(reversed(ks), reversed(separable_ms), reversed(direct_ms), strict=True):
        if sep * (1 + margin) > direct:
            break
        lasting = k
    return lasting


def sweep_separable_threshold(
    dtype: torch.dtype,
    *,
    backward: bool,
    device: torch.device | int | None = None,
    ks: Sequence[int] = DEFAULT_KS,
    shape: tuple[int, int, int, int] = _REFERENCE_SHAPE,
    reps: int = 20,
) -> int:
    """
    Measure where the separable path overtakes the direct kernels on one GPU, for one dtype and pass.

    Every call runs on scratch tensors, so nothing a caller owns is written. The forward pass is timed through the
    erosion op and the backward pass through its gradient op alone, since the two have different crossovers.

    :param dtype: element type to measure.
    :param backward: whether to measure the backward pass instead of the forward.
    :param device: CUDA device, the current one when ``None``.
    :param ks: window lengths to measure, ascending. Lengths too long for the separable kernels to stage are skipped.
    :param shape: ``(N, C, H, W)`` of the input.
    :param reps: timed repetitions per measurement.
    :returns: the window length from which separable wins. When it never wins in the range, one past the longest
    length measured.
    :raises RuntimeError: if called while a CUDA graph is being captured, or if no length could be measured.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("serron: cannot tune while a CUDA graph is being captured")

    with torch.cuda.device(device):
        x = torch.randn(shape, dtype=dtype, device="cuda")
        grad = torch.randn_like(x)

        measured: list[int] = []
        separable_ms: list[float] = []
        direct_ms: list[float] = []
        for k in ks:
            kernel = torch.zeros(shape[1], k, k, dtype=dtype, device="cuda")  # flat, so separable is eligible
            candidates = feasible_algorithms(x, kernel, flat=True, backward=backward)
            direct = [algo for algo in candidates if algo is not Algorithm.SEPARABLE]
            if Algorithm.SEPARABLE not in candidates or not direct:
                continue

            def run(algo: Algorithm, kernel: torch.Tensor = kernel) -> object:
                if backward:
                    return ops.erode_backward(grad, x, kernel, _BORDER, True, True, int(algo))
                return ops.erode(x, kernel, _BORDER, True, int(algo))

            measured.append(k)
            separable_ms.append(time_ms(lambda run=run: run(Algorithm.SEPARABLE), reps))
            direct_ms.append(min(time_ms(lambda run=run, algo=algo: run(algo), reps) for algo in direct))

    if not measured:
        raise RuntimeError(f"serron: no window length in {tuple(ks)} could be measured for {dtype}")
    found = crossover(measured, separable_ms, direct_ms)
    return found if found is not None else measured[-1] + 1


def apply_thresholds(thresholds: dict[str, int], device: torch.device | int | None = None) -> None:
    """
    Make a saved threshold table the ones the heuristic uses in this process.

    :param thresholds: mapping from :func:`threshold_key` to window length. Keys naming a dtype this torch does not
    have are ignored, so a table from a newer version still loads.
    :param device: CUDA device the thresholds are for, the current one when ``None``.
    """
    with torch.cuda.device(device):
        for key, min_k in thresholds.items():
            dtype_name, _, pass_name = key.partition(":")
            dtype = getattr(torch, dtype_name, None)
            if not isinstance(dtype, torch.dtype) or pass_name not in {"fwd", "bwd"}:
                continue
            set_separable_min_k(torch.empty(0, dtype=dtype, device="cuda"), min_k, backward=pass_name == "bwd")


def calibrate(
    device: torch.device | int | None = None,
    *,
    dtypes: Sequence[torch.dtype] = DEFAULT_DTYPES,
    force: bool = False,
    ks: Sequence[int] = DEFAULT_KS,
    shape: tuple[int, int, int, int] = _REFERENCE_SHAPE,
    reps: int = 20,
) -> dict[str, int]:
    """
    Measure where the separable path overtakes the direct kernels on a GPU, save the result and use it.

    The thresholds apply to this process straight away and are saved under ``$SERRON_CACHE_DIR`` (default
    ``~/.cache/serron``), keyed by GPU, CUDA version and serron version, so later processes load them on their first
    CUDA op without measuring again. Entries already saved for the same key are reused, and only the missing ones are
    measured, unless ``force`` is set. ``SERRON_SEPARABLE_MIN_K`` still overrides every threshold.

    :param device: CUDA device, the current one when ``None``.
    :param dtypes: element types to calibrate. Each is measured forward and backward.
    :param force: measure everything again and overwrite what was saved.
    :param ks: window lengths each sweep measures, ascending.
    :param shape: ``(N, C, H, W)`` of the input each sweep times.
    :param reps: timed repetitions per measurement.
    :returns: the thresholds in force for ``dtypes``, from :func:`threshold_key` to window length.
    :raises RuntimeError: if CUDA is not available, or a CUDA graph is being captured and something needs measuring.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("serron: calibrate needs a CUDA device")
    index = torch.device("cuda", torch.cuda.current_device()).index if device is None else torch.device(device).index
    fingerprint = device_fingerprint(index)

    wanted = {threshold_key(dtype, backward=bw): (dtype, bw) for dtype in dtypes for bw in (False, True)}
    saved: dict[str, int] = {} if force else (_tuning_cache.load_thresholds(fingerprint) or {})
    measured: dict[str, int] = {
        key: sweep_separable_threshold(dtype, backward=bw, device=index, ks=ks, shape=shape, reps=reps)
        for key, (dtype, bw) in wanted.items()
        if key not in saved
    }

    if measured:
        try:
            _tuning_cache.store_thresholds(fingerprint, measured)
        except OSError as exc:
            warnings.warn(f"serron: could not save the calibration, it applies to this process only: {exc}", stacklevel=2)

    table = {key: measured.get(key, saved.get(key)) for key in wanted}
    result = {key: value for key, value in table.items() if value is not None}
    apply_thresholds(result, index)
    with _load_lock:
        _loaded.add(index)
    return result


# Devices whose saved thresholds have been looked up, so the hot path is one set lookup.
_loaded: set[int] = set()
_load_lock = threading.Lock()


def _lazy_loading_enabled() -> bool:
    return os.environ.get("SERRON_AUTOTUNE", "1") != "0"


def ensure_loaded(device: torch.device) -> None:
    """
    Load the thresholds saved for ``device``, once per process.

    Called before a CUDA op runs. It looks at the cache the first time a device is seen, applies what it finds, and
    afterwards costs one set lookup. It never raises: a cache that cannot be used leaves the defaults in force. It
    does nothing under CUDA graph capture, and tries again at the next call outside it.

    ``SERRON_AUTOTUNE=0`` turns the lookup off.

    :param device: device of the op's input. Non-CUDA devices are ignored.
    """
    if device.type != "cuda" or device.index in _loaded:
        return
    with _load_lock:
        index = device.index
        if index is None or index in _loaded or torch.cuda.is_current_stream_capturing():
            return
        _loaded.add(index)
        if not _lazy_loading_enabled():
            return
        try:
            thresholds = _tuning_cache.load_thresholds(device_fingerprint(index))
            if thresholds:
                apply_thresholds(thresholds, index)
        except Exception as exc:
            warnings.warn(f"serron: could not load the saved calibration, using the defaults: {exc}", stacklevel=2)


def _forget_loaded() -> None:
    """Make every device look unseen again. For tests."""
    with _load_lock:
        _loaded.clear()
