"""
Cross-library speed comparison for grayscale erosion and dilation.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

import serron
from benchmarks._table import markdown_table
from serron import BorderMode

try:
    import kornia.morphology as km
except ImportError:
    km = None  # type: ignore[assignment]

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cnd
except ImportError:
    cp = None
    cnd = None

# (batch, channels, height, width), kernel size
CASES = [
    (1, 1, 512, 512, 3),
    (1, 1, 512, 512, 15),
    (8, 3, 512, 512, 7),
    (8, 32, 256, 256, 5),
    (1, 1, 512, 512, 31),
    (1, 1, 512, 512, 63),
    (8, 3, 512, 512, 31),
    (8, 3, 512, 512, 63),
    (8, 3, 512, 512, 127),
]
_OPS = ("erosion", "dilation")
_LIBS = ("torch", "scipy", "kornia", "cupy")
_COLUMNS = ("op", "shape", "k", "serron (ms)", *(c for lib in _LIBS for c in (f"{lib} (ms)", f"vs {lib}")))
_ALIGNS = "ll" + "r" * (len(_COLUMNS) - 2)
_ITERS = 30
_WARMUP = 5


def _time_gpu(fn: Callable[[], object], warmup: int = _WARMUP, iters: int = _ITERS) -> float:
    """Time a GPU callable, syncing around the timed region.

    :param fn: the callable to time.
    :param warmup: untimed calls before measuring.
    :param iters: timed calls.
    :returns: mean seconds per call.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def _time_cpu(fn: Callable[[], object], warmup: int = 1, iters: int = 3) -> float:
    """Time a CPU callable. Fewer iterations, since the CPU path is the slow one.

    :param fn: the callable to time.
    :param warmup: untimed calls before measuring.
    :param iters: timed calls.
    :returns: mean seconds per call.
    """
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters


def _measure(fn: Callable[[], object], timer: Callable[[Callable[[], object]], float] = _time_gpu) -> float | str:
    """Time a callable, reporting an allocation failure instead of raising.

    :param fn: the callable to time.
    :param timer: the timing helper to use.
    :returns: mean seconds per call, or ``"OOM"`` if the library ran out of memory.
    """
    try:
        return timer(fn)
    except (torch.cuda.OutOfMemoryError, MemoryError):  # cupy's OOM subclasses MemoryError
        torch.cuda.empty_cache()
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
        return "OOM"


def _lib_cells(t_lib: float | str, t_serron: float) -> list[str]:
    """Format one library's time and serron's speedup over it.

    :param t_lib: the library's mean seconds per call, or a status such as ``"OOM"``.
    :param t_serron: serron's mean seconds per call, for the ratio.
    :returns: the two cells, time then speedup.
    """
    if isinstance(t_lib, str):
        return [t_lib, "-"]
    return [f"{t_lib * 1e3:.3f}", f"{t_lib / t_serron:.2f}x"]


def _run_case(op: str, n: int, c: int, h: int, w: int, k: int) -> list[str]:
    """Time one op/shape across every library and format the result as table cells.

    :param op: ``"erosion"`` or ``"dilation"``.
    :param n: batch size.
    :param c: channel count.
    :param h: image height.
    :param w: image width.
    :param k: structuring-element side length.
    :returns: one cell per column, in :data:`_COLUMNS` order.
    """
    import numpy as np
    from scipy import ndimage

    x = torch.randn(n, c, h, w, device="cuda", dtype=torch.float32)
    se_flat = torch.zeros(c, k, k, device="cuda", dtype=torch.float32)
    pad = k // 2

    if op == "erosion":

        def serron_op() -> torch.Tensor:
            return serron.erosion(x, se_flat, border=BorderMode.CONSTANT)

        def torch_op() -> torch.Tensor:
            return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=pad)

        scipy_filter = ndimage.minimum_filter
        cupy_filter = cnd.minimum_filter if cnd is not None else None
        cval = float(np.inf)
        kornia_op = km.erosion if km is not None else None
    else:

        def serron_op() -> torch.Tensor:
            return serron.dilation(x, se_flat, border=BorderMode.CONSTANT)

        def torch_op() -> torch.Tensor:
            return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)

        scipy_filter = ndimage.maximum_filter
        cupy_filter = cnd.maximum_filter if cnd is not None else None
        cval = float(-np.inf)
        kornia_op = km.dilation if km is not None else None

    torch.testing.assert_close(serron_op(), torch_op())  # same result before we race them
    t_serron = _time_gpu(serron_op)
    times: dict[str, float | str] = {"torch": _measure(torch_op)}

    x_np = x.cpu().numpy()

    def scipy_run() -> object:
        return np.stack([scipy_filter(img, size=k, mode="constant", cval=cval) for img in x_np.reshape(-1, h, w)])

    times["scipy"] = _measure(scipy_run, _time_cpu)

    if kornia_op is not None:
        se_k = torch.ones(k, k, device="cuda")
        times["kornia"] = _measure(lambda: kornia_op(x, se_k))
    else:
        times["kornia"] = "n/a"

    if cupy_filter is not None:
        x_cp = cp.from_dlpack(x)  # zero-copy view of the same GPU buffer
        # size 1 on N and C keeps this a per-image 2-D filter, like every other path here
        times["cupy"] = _measure(lambda: cupy_filter(x_cp, size=(1, 1, k, k), mode="constant", cval=cval))
    else:
        times["cupy"] = "n/a"

    cells = [op, f"{n}x{c}x{h}x{w}", str(k), f"{t_serron * 1e3:.3f}"]
    for lib in _LIBS:
        cells += _lib_cells(times[lib], t_serron)
    return cells


def main() -> None:
    """Print the comparison table for every op and case, as markdown."""
    if not torch.cuda.is_available():
        print("benchmarks require a CUDA device")
        return
    cases = [(op, *case) for op in _OPS for case in CASES]
    rows = []
    for i, (op, n, c, h, w, k) in enumerate(cases, start=1):
        # progress on stderr, so stdout stays a pasteable markdown table
        print(f"[{i}/{len(cases)}] {op} {n}x{c}x{h}x{w} k={k}", file=sys.stderr, flush=True)
        rows.append(_run_case(op, n, c, h, w, k))
    print(f"\nFlat SE, float32 on {torch.cuda.get_device_name(0)} (scipy on CPU).")
    print("`vs <lib>` is that library's time divided by serron's: >1x means serron is faster.\n")
    print(markdown_table(_COLUMNS, rows, _ALIGNS))


if __name__ == "__main__":
    main()
