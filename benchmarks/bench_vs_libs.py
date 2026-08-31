"""
Cross-library speed comparison for grayscale erosion and dilation.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

import serron
from serron import BorderMode

try:
    import kornia.morphology as km
except ImportError:
    km = None  # type: ignore[assignment]

# (batch, channels, height, width), kernel size
CASES = [
    (1, 1, 512, 512, 3),
    (1, 1, 512, 512, 15),
    (8, 3, 512, 512, 7),
    (8, 32, 256, 256, 5),
]
_OPS = ("erosion", "dilation")
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


def _run_case(op: str, n: int, c: int, h: int, w: int, k: int) -> str:
    """Time one op/shape across every library and format the result as a table row.

    :param op: ``"erosion"`` or ``"dilation"``.
    :param n: batch size.
    :param c: channel count.
    :param h: image height.
    :param w: image width.
    :param k: structuring-element side length.
    :returns: the formatted row.
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
        scipy_cval = float(np.inf)
        kornia_op = km.erosion if km is not None else None
    else:

        def serron_op() -> torch.Tensor:
            return serron.dilatation(x, se_flat, border=BorderMode.CONSTANT)

        def torch_op() -> torch.Tensor:
            return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)

        scipy_filter = ndimage.maximum_filter
        scipy_cval = float(-np.inf)
        kornia_op = km.dilation if km is not None else None

    torch.testing.assert_close(serron_op(), torch_op())  # same result before we race them
    t_serron = _time_gpu(serron_op)
    t_torch = _time_gpu(torch_op)

    x_np = x.cpu().numpy()

    def scipy_run() -> object:
        return np.stack([scipy_filter(img, size=k, mode="constant", cval=scipy_cval) for img in x_np.reshape(-1, h, w)])

    t_scipy = _time_cpu(scipy_run)

    if kornia_op is not None:
        se_k = torch.ones(k, k, device="cuda")

        def kornia_run() -> torch.Tensor:
            return kornia_op(x, se_k)

        kornia_col = f"{_time_gpu(kornia_run) * 1e3:>13.3f}"
    else:
        kornia_col = f"{'n/a':>13}"

    case = f"{op[:3]}:{n}x{c}x{h}x{w}_k{k}"
    return f"{case:<22}{t_serron * 1e3:>13.3f}{t_torch * 1e3:>12.3f}{t_scipy * 1e3:>12.3f}{kornia_col}"


def main() -> None:
    """Print the comparison table for every op and case."""
    if not torch.cuda.is_available():
        print("benchmarks require a CUDA device")
        return
    print(f"\nflat SE, float32 on {torch.cuda.get_device_name(0)} (scipy on CPU)\n")
    header = f"{'case':<22}{'serron (ms)':>13}{'torch (ms)':>12}{'scipy (ms)':>12}{'kornia (ms)':>13}"
    print(header)
    print("-" * len(header))
    for op in _OPS:
        for n, c, h, w, k in CASES:
            print(_run_case(op, n, c, h, w, k))


if __name__ == "__main__":
    main()
