"""
Throughput benchmarks for the morphology kernels, driven by ``pytest-benchmark``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import serron
from benchmarks.conftest import CUDA_AVAILABLE, cuda_sync, requires_cuda
from serron import BorderMode

pytestmark = requires_cuda

DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

# (batch, channels, height, width)
SHAPES = [
    (1, 1, 256, 256),
    (8, 3, 512, 512),
    (16, 3, 1024, 1024),
]
KERNEL_SIZES = [3, 7, 15]
OPS = ["erosion", "dilation", "opening", "closing", "gradient", "top_hat", "black_hat"]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SE_KINDS = ["flat", "grayscale"]
LAYERS = {
    "Erosion2d": serron.Erosion2d,
    "Dilation2d": serron.Dilation2d,
    "Opening2d": serron.Opening2d,
    "Closing2d": serron.Closing2d,
}

_shape_id = lambda s: "x".join(map(str, s))  # noqa: E731


def _make(
    shape: tuple[int, int, int, int],
    ksize: int,
    *,
    se_kind: str = "flat",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a random input and a matching structuring element on ``DEVICE``.

    :param shape: input shape, ``(batch, channels, height, width)``.
    :param ksize: side length of the square structuring element.
    :param se_kind: ``"flat"`` for an all-zero SE, ``"grayscale"`` for small
        random taps.
    :param dtype: dtype shared by both tensors.
    :returns: the ``(input, kernel)`` pair.
    """
    n, c, h, w = shape
    x = torch.randn(n, c, h, w, device=DEVICE, dtype=dtype)
    if se_kind == "flat":
        kernel = torch.zeros(c, ksize, ksize, device=DEVICE, dtype=dtype)
    else:
        kernel = torch.randn(c, ksize, ksize, device=DEVICE, dtype=dtype).mul_(0.1)
    return x, kernel


def _warmup(fn: Any) -> None:
    fn()
    cuda_sync()


@pytest.mark.parametrize("ksize", KERNEL_SIZES, ids=lambda k: f"k{k}")
@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
@pytest.mark.parametrize("op", OPS)
def test_forward_throughput(benchmark: Any, op: str, shape: tuple[int, int, int, int], ksize: int) -> None:
    x, kernel = _make(shape, ksize)
    func = getattr(serron.functional, op)

    def run() -> Any:
        out = func(x, kernel)
        cuda_sync()
        return out

    _warmup(run)
    benchmark(run)


@pytest.mark.parametrize("shape", SHAPES[:2], ids=_shape_id)
@pytest.mark.parametrize("op", OPS)
def test_forward_backward_throughput(benchmark: Any, op: str, shape: tuple[int, int, int, int]) -> None:
    x, kernel = _make(shape, ksize=7)
    x = x.requires_grad_(True)
    kernel = kernel.requires_grad_(True)
    func = getattr(serron.functional, op)

    def run() -> Any:
        loss = func(x, kernel).sum()
        grads = torch.autograd.grad(loss, (x, kernel))
        cuda_sync()
        return grads

    _warmup(run)
    benchmark(run)


@pytest.mark.parametrize("dtype_name", list(DTYPES))
@pytest.mark.parametrize("op", ["erosion", "dilation"])
def test_dtype_forward_throughput(benchmark: Any, op: str, dtype_name: str) -> None:
    x, kernel = _make((8, 3, 512, 512), 7, dtype=DTYPES[dtype_name])
    func = getattr(serron.functional, op)

    def run() -> Any:
        out = func(x, kernel)
        cuda_sync()
        return out

    _warmup(run)
    benchmark(run)


@pytest.mark.parametrize("se_kind", SE_KINDS)
@pytest.mark.parametrize("op", ["erosion", "dilation"])
def test_se_kind_forward_throughput(benchmark: Any, op: str, se_kind: str) -> None:
    x, kernel = _make((8, 3, 512, 512), 15, se_kind=se_kind)
    func = getattr(serron.functional, op)

    def run() -> Any:
        out = func(x, kernel)
        cuda_sync()
        return out

    _warmup(run)
    benchmark(run)


@pytest.mark.parametrize("border", list(BorderMode), ids=lambda b: b.name)
def test_border_forward_throughput(benchmark: Any, border: BorderMode) -> None:
    x, kernel = _make((8, 3, 512, 512), 7)

    def run() -> Any:
        out = serron.functional.erosion(x, kernel, border=border)
        cuda_sync()
        return out

    _warmup(run)
    benchmark(run)


@pytest.mark.parametrize("name", list(LAYERS))
def test_layer_training_step_throughput(benchmark: Any, name: str) -> None:
    layer = LAYERS[name](channels=3, kernel_size=7).to(DEVICE)
    x = torch.randn(8, 3, 512, 512, device=DEVICE)
    target = torch.zeros_like(x)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)

    def run() -> None:
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(layer(x), target)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        cuda_sync()

    _warmup(run)
    benchmark(run)
