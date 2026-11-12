"""
Autotuning for algorithm selection.

Times the algorithms against each other on this GPU to find where the separable path overtakes the direct kernels.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch

from ._algorithms import feasible_algorithms, set_separable_min_k
from ._cmake_ops import ops
from ._timing import time_ms
from .enums import Algorithm

__all__ = [
    "DEFAULT_KS",
    "HYSTERESIS",
    "apply_thresholds",
    "crossover",
    "sweep_separable_threshold",
    "threshold_key",
]

HYSTERESIS = 0.05
"""A candidate replaces the heuristic only when it is at least this fraction faster."""

DEFAULT_KS = (3, 5, 7, 9, 11, 13, 15, 21, 31, 63)
"""Window lengths a sweep measures."""

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
