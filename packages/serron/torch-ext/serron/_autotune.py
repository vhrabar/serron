"""
Autotuning for algorithm selection.

Two levels. :func:`calibrate` finds, per GPU, dtype and pass, the window length from which the separable path beats the
direct kernels. :func:`autotune` goes further and times every feasible algorithm on specific problems, remembering the
winner per problem. Both save their results and load them back the first time a CUDA op runs.
"""

from __future__ import annotations

import itertools
import math
import os
import threading
import warnings
from collections.abc import Callable, Sequence

import torch

from . import _seeds, _tuning_cache
from ._algorithms import device_fingerprint, feasible_algorithms, heuristic_algorithm, set_separable_min_k
from ._cmake_ops import ops
from ._timing import time_ms
from .enums import Algorithm

__all__ = [
    "DEFAULT_DTYPES",
    "DEFAULT_KS",
    "DEFAULT_SHAPE_KS",
    "HYSTERESIS",
    "LAUNCH_BOUND_ELEMENTS",
    "apply_thresholds",
    "autotune",
    "calibrate",
    "crossover",
    "ensure_loaded",
    "select_algorithm",
    "shape_key",
    "sweep_separable_threshold",
    "threshold_key",
    "tune_problem",
]

HYSTERESIS = 0.05
"""A candidate replaces the heuristic only when it is at least this fraction faster."""

DEFAULT_KS = (3, 5, 7, 9, 11, 13, 15, 21, 31, 63)
"""Window lengths a sweep measures."""

DEFAULT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
"""Element types :func:`calibrate` measures unless told otherwise."""

DEFAULT_SHAPE_KS = (3, 5, 7, 11, 15, 21, 31)
"""Square window lengths :func:`autotune` measures unless told otherwise."""

LAUNCH_BOUND_ELEMENTS = 1 << 16
"""
Largest ``N*C*H*W`` that is not worth tuning, so ``1x1x256x256`` is left alone. Up to it a call is bound by launch
latency, every algorithm costs about the same, and a measurement would be noise.
"""

_SHAPE_PREFIX = "shape:"

_REFERENCE_SHAPE = (1, 3, 512, 512)
_BORDER = 1  # replicate


def _runner(x: torch.Tensor, grad: torch.Tensor, kernel: torch.Tensor, backward: bool) -> Callable[[Algorithm], object]:
    """A function that runs the forward, or the backward, once with a forced algorithm, on scratch inputs."""

    def run(algo: Algorithm) -> object:
        if backward:
            return ops.erode_backward(grad, x, kernel, _BORDER, True, True, int(algo))
        return ops.erode(x, kernel, _BORDER, True, int(algo))

    return run


SCREEN_REPS = 2
"""Repetitions of the quick first pass in :func:`_time_candidates`."""

SCREEN_SLOWDOWN = 2.0
"""A candidate this many times slower than the best in the quick pass is not timed further."""


def _time_candidates(run: Callable[[Algorithm], object], candidates: Sequence[Algorithm], reps: int) -> dict[Algorithm, float]:
    """
    Median time of ``run`` for each candidate.

    Most of the cost of tuning is spent on candidates that are far from winning, so every candidate is first timed
    for :data:`SCREEN_REPS` repetitions, and only those within :data:`SCREEN_SLOWDOWN` of the best get the full
    ``reps``. Nothing is ruled out in advance, only measured out. A candidate dropped after the quick pass keeps its
    quick time, which is a fair stand-in because it is far from the best anyway.

    :param run: runs one candidate once.
    :param candidates: algorithms to time.
    :param reps: repetitions for the candidates that survive the quick pass.
    :returns: a time for every candidate.
    """
    if reps <= SCREEN_REPS or len(candidates) < 2:
        return {algo: time_ms(lambda algo=algo: run(algo), reps) for algo in candidates}

    times = {algo: time_ms(lambda algo=algo: run(algo), SCREEN_REPS) for algo in candidates}
    best = min(times.values())
    for algo in candidates:
        if times[algo] <= best * SCREEN_SLOWDOWN:
            times[algo] = time_ms(lambda algo=algo: run(algo), reps, warmup=0)
    return times


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def threshold_key(dtype: torch.dtype, *, backward: bool) -> str:
    """
    Key of one threshold in a saved table.

    :param dtype: element type of the input.
    :param backward: whether the threshold is for the backward pass.
    :returns: for example ``"float32:fwd"``.
    """
    return f"{_dtype_name(dtype)}:{'bwd' if backward else 'fwd'}"


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

            times = _time_candidates(_runner(x, grad, kernel, backward), candidates, reps)
            measured.append(k)
            separable_ms.append(times[Algorithm.SEPARABLE])
            direct_ms.append(min(times[algo] for algo in direct))

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
    ensure_loaded(torch.device("cuda", index))  # seeds for the dtypes not measured below stay in force

    wanted = {threshold_key(dtype, backward=bw): (dtype, bw) for dtype in dtypes for bw in (False, True)}
    stored = _tuning_cache.load_thresholds(fingerprint) or {}
    saved: dict[str, int] = {} if force else {k: v for k, v in stored.items() if not k.startswith(_SHAPE_PREFIX)}
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
        _shape_table.update({f"{index}|{k}": v for k, v in stored.items() if k.startswith(_SHAPE_PREFIX)})
    return result


# Devices whose saved results have been looked up, so the hot path is one set lookup.
_loaded: set[int] = set()
_load_lock = threading.Lock()

# Winning algorithm per tuned problem, from ``"<device>|<shape_key>"`` to an ``Algorithm`` value; 0 means the heuristic
# was already as fast within the margin.
_shape_table: dict[str, int] = {}
_lazy = False
_tune_lock = threading.Lock()


def _mode() -> str:
    """``SERRON_AUTOTUNE``: ``"0"`` ignores saved results, ``"1"`` also tunes unseen problems, anything else only loads."""
    return os.environ.get("SERRON_AUTOTUNE", "")


def ensure_loaded(device: torch.device) -> None:
    """
    Load the thresholds and tuned problems known for ``device``, once per process.

    What is known is the shipped seed for the device's architecture (see :mod:`serron._seeds`), overlaid with whatever
    was saved on this machine.

    Called before a CUDA op runs. It looks at the cache the first time a device is seen, applies what it finds, and
    afterwards costs one set lookup. It never raises: a cache that cannot be used leaves the defaults in force. It
    does nothing under CUDA graph capture, and tries again at the next call outside it.

    ``SERRON_AUTOTUNE=0`` turns the lookup off, shipped seeds included. ``SERRON_AUTOTUNE=1`` also lets :func:`select_algorithm` tune
    problems it has not seen.

    :param device: device of the op's input. Non-CUDA devices are ignored.
    """
    if device.type != "cuda" or device.index in _loaded:
        return
    global _lazy
    with _load_lock:
        index = device.index
        if index is None or index in _loaded or torch.cuda.is_current_stream_capturing():
            return
        _loaded.add(index)
        mode = _mode()
        _lazy = mode == "1"
        if mode == "0":
            return
        try:
            fingerprint = device_fingerprint(index)
            # A seed is the starting point for an architecture; whatever this machine saved overrides it key by key.
            seed = _seeds.seed_thresholds(fingerprint["capability"]) or {}
            known = {**seed, **(_tuning_cache.load_thresholds(fingerprint) or {})}
            if known:
                _adopt(known, index)
        except Exception as exc:
            warnings.warn(f"serron: could not load the saved calibration, using the defaults: {exc}", stacklevel=2)


def _adopt(saved: dict[str, int], index: int) -> None:
    """Take saved results into use: thresholds go to the compiled heuristic, per-problem winners to the table."""
    shapes = {k: v for k, v in saved.items() if k.startswith(_SHAPE_PREFIX)}
    apply_thresholds({k: v for k, v in saved.items() if k not in shapes}, index)
    _shape_table.update({f"{index}|{k}": v for k, v in shapes.items()})


def _forget_loaded() -> None:
    """Make every device look unseen again and drop what was loaded or tuned. For tests."""
    global _lazy
    with _load_lock:
        _loaded.clear()
        _shape_table.clear()
        _lazy = False


# --------------------------------------------------------------------------- #
# Per-problem tuning
# --------------------------------------------------------------------------- #


def shape_key(dtype: torch.dtype, *, backward: bool, flat: bool, k_h: int, k_w: int, numel: int) -> str:
    """
    Key of one tuned problem in a saved table.

    The input is identified by the size class of ``N*C*H*W`` (its bit length), not its exact shape, so one tuning
    serves every shape within a factor of two of it.

    :param dtype: element type of the input.
    :param backward: whether the problem is the backward pass.
    :param flat: whether the structuring element is flat.
    :param k_h: structuring-element height.
    :param k_w: structuring-element width.
    :param numel: ``N*C*H*W`` of the input.
    :returns: for example ``"shape:float32:fwd:flat:5x5:p20"``.
    """
    return (
        f"{_SHAPE_PREFIX}{_dtype_name(dtype)}:{'bwd' if backward else 'fwd'}:{'flat' if flat else 'dense'}"
        f":{k_h}x{k_w}:p{max(numel, 1).bit_length()}"
    )


def tune_problem(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    k_h: int,
    k_w: int,
    *,
    flat: bool,
    backward: bool,
    device: torch.device | int | None = None,
    reps: int = 20,
) -> Algorithm:
    """
    Time every algorithm that can run a problem and pick the one to use.

    All calls run on scratch tensors, so nothing a caller owns is written. The winner replaces the library's own
    choice only if it is at least :data:`HYSTERESIS` faster than that choice, so a near tie keeps the heuristic and
    noise does not flip between algorithms.

    :param shape: ``(N, C, H, W)`` of the input.
    :param dtype: element type.
    :param k_h: structuring-element height.
    :param k_w: structuring-element width.
    :param flat: whether the structuring element is flat.
    :param backward: whether to tune the backward pass, timed through its gradient op alone.
    :param device: CUDA device, the current one when ``None``.
    :param reps: timed repetitions per candidate.
    :returns: the algorithm to force, or :attr:`Algorithm.AUTO` when the heuristic should stay.
    :raises RuntimeError: if called while a CUDA graph is being captured.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("serron: cannot tune while a CUDA graph is being captured")

    with torch.cuda.device(device):
        x = torch.randn(shape, dtype=dtype, device="cuda")
        grad = torch.randn_like(x)
        kernel = torch.zeros(shape[1], k_h, k_w, dtype=dtype, device="cuda")
        if not flat:
            kernel = torch.randn_like(kernel)

        candidates = feasible_algorithms(x, kernel, flat=flat, backward=backward)
        if len(candidates) < 2:
            return Algorithm.AUTO
        chosen = heuristic_algorithm(x, kernel, flat=flat, backward=backward)
        times = _time_candidates(_runner(x, grad, kernel, backward), candidates, reps)

    best = min(times, key=lambda algo: times[algo])
    if best is not chosen and times[best] * (1 + HYSTERESIS) <= times[chosen]:
        return best
    return Algorithm.AUTO


def autotune(
    shapes: Sequence[tuple[int, int, int, int]],
    *,
    ks: Sequence[int] = DEFAULT_SHAPE_KS,
    dtypes: Sequence[torch.dtype] = (torch.float32,),
    flat: Sequence[bool] = (True, False),
    device: torch.device | int | None = None,
    reps: int = 20,
    force: bool = False,
) -> dict[str, int]:
    """
    Tune the algorithm for given input shapes ahead of time, so the first call with each does not pay for it.

    Every combination of shape, square window length, dtype and structuring-element kind is timed, forward and
    backward, on scratch tensors. The winners apply to this process at once and are saved next to the
    :func:`calibrate` thresholds, so later processes load them on their first CUDA op. A winner is only used where it
    beats the heuristic by :data:`HYSTERESIS`. Shapes of up to :data:`LAUNCH_BOUND_ELEMENTS` elements are skipped:
    they are bound by launch latency and left to the heuristic.

    :param shapes: ``(N, C, H, W)`` inputs to tune for. A tuning also serves shapes within a factor of two in ``N*C*H*W``.
    :param ks: square window lengths to tune.
    :param dtypes: element types to tune.
    :param flat: structuring-element kinds to tune: ``True`` for flat, ``False`` for non-flat.
    :param device: CUDA device, the current one when ``None``.
    :param reps: timed repetitions per candidate.
    :param force: tune again even where a result is already known.
    :returns: for each problem tuned or already known, from :func:`shape_key` to the :class:`Algorithm` value that is
    used (``0`` when the heuristic stays).
    :raises RuntimeError: if CUDA is not available, or a CUDA graph is being captured and something needs tuning.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("serron: autotune needs a CUDA device")
    index = torch.device("cuda", torch.cuda.current_device()).index if device is None else torch.device(device).index
    if index is None:
        raise RuntimeError("serron: autotune needs a CUDA device index")
    ensure_loaded(torch.device("cuda", index))

    results: dict[str, int] = {}
    fresh: dict[str, int] = {}
    for shape, dtype, k, is_flat, backward in itertools.product(shapes, dtypes, ks, flat, (False, True)):
        numel = math.prod(shape)
        if numel <= LAUNCH_BOUND_ELEMENTS:
            continue
        key = shape_key(dtype, backward=backward, flat=is_flat, k_h=k, k_w=k, numel=numel)
        known = _shape_table.get(f"{index}|{key}")
        if known is not None and not force:
            results[key] = known
            continue
        winner = tune_problem(shape, dtype, k, k, flat=is_flat, backward=backward, device=index, reps=reps)
        results[key] = fresh[key] = int(winner)

    if fresh:
        _remember(index, fresh)
    return results


def _remember(index: int, fresh: dict[str, int]) -> None:
    """Use ``fresh`` in this process and save it, warning rather than failing when it cannot be saved."""
    with _load_lock:
        _shape_table.update({f"{index}|{k}": v for k, v in fresh.items()})
    try:
        _tuning_cache.store_thresholds(device_fingerprint(index), fresh)
    except OSError as exc:
        warnings.warn(f"serron: could not save the tuning, it applies to this process only: {exc}", stacklevel=3)


def select_algorithm(input_: torch.Tensor, kernel: torch.Tensor, flat: bool, *, backward: bool) -> int:
    """
    The algorithm to force for a call, from what :func:`autotune` and lazy tuning have learned.

    Called by the autograd functions before a CUDA op. When nothing has been tuned and lazy tuning is off it is a
    truthiness check on two globals and returns at once.

    With ``SERRON_AUTOTUNE=1`` a problem not seen before is tuned here, on the calling thread, the first time it
    occurs: that call pays for timing every candidate on scratch tensors. Nothing is tuned during CUDA graph
    capture, for sizes up to :data:`LAUNCH_BOUND_ELEMENTS`, or if tuning fails; those use the heuristic.

    :param input_: input of the call.
    :param kernel: structuring element of the call.
    :param flat: whether ``kernel`` is flat.
    :param backward: whether the call is the backward pass.
    :returns: an :class:`Algorithm` value, ``0`` to let the library choose.
    """
    if not (_shape_table or _lazy) or not input_.is_cuda or input_.numel() <= LAUNCH_BOUND_ELEMENTS:
        return 0
    index = input_.device.index
    if index is None:
        return 0
    key = shape_key(
        input_.dtype,
        backward=backward,
        flat=flat,
        k_h=kernel.shape[-2],
        k_w=kernel.shape[-1],
        numel=input_.numel(),
    )
    full = f"{index}|{key}"
    known = _shape_table.get(full)
    if known is not None or not _lazy:
        return known or 0
    return _tune_lazily(full, key, index, input_, kernel, flat, backward)


def _tune_lazily(full: str, key: str, index: int, input_: torch.Tensor, kernel: torch.Tensor, flat: bool, backward: bool) -> int:
    if torch.cuda.is_current_stream_capturing():
        return 0
    with _tune_lock:
        known = _shape_table.get(full)
        if known is not None:
            return known
        try:
            winner = int(
                tune_problem(
                    (input_.shape[0], input_.shape[1], input_.shape[2], input_.shape[3]),
                    input_.dtype,
                    kernel.shape[-2],
                    kernel.shape[-1],
                    flat=flat,
                    backward=backward,
                    device=index,
                )
            )
        except Exception as exc:
            warnings.warn(f"serron: tuning failed, using the heuristic for this problem: {exc}", stacklevel=2)
            _shape_table[full] = 0
            return 0
        _remember(index, {key: winner})
        return winner
