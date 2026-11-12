"""
Forcing the algorithm a compiled op runs through its ``algo`` argument.
"""

from __future__ import annotations

import pytest
import torch

import serron
from serron.autograd import _DilateFunction, _ErodeFunction, ops
from serron.enums import Algorithm
from tests.conftest import flat_se, make_image, requires_cuda

_OPS = ["erode", "dilate"]
_DIRECT = [Algorithm.ELEMENT, Algorithm.TILED]
_BORDER = 0
_SEPARABLE_K = 21


def _forward(op: str, x: torch.Tensor, kernel: torch.Tensor, algo: Algorithm) -> torch.Tensor:
    return getattr(ops, op)(x, kernel, _BORDER, True, int(algo))


@requires_cuda
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("k", [3, _SEPARABLE_K])
def test_forced_algorithms_are_bitwise_equal_forward(op: str, k: int, rng: torch.Generator) -> None:
    """Min and max are exact, so every algorithm that can run the problem returns the same bits."""
    x = make_image(rng, (2, 3, 40, 37), device=torch.device("cuda"))
    kernel = flat_se(x.shape[1], k, device=x.device, shared=False)

    expected = _forward(op, x, kernel, Algorithm.AUTO)
    for algo in (*_DIRECT, Algorithm.SEPARABLE):
        assert torch.equal(_forward(op, x, kernel, algo), expected), algo.name


@requires_cuda
@pytest.mark.parametrize("op", _OPS)
def test_forced_algorithms_agree_on_gradients(op: str, rng: torch.Generator) -> None:
    function = _ErodeFunction if op == "erode" else _DilateFunction
    x = make_image(rng, (2, 3, 40, 37), device=torch.device("cuda"))
    kernel = flat_se(x.shape[1], _SEPARABLE_K, device=x.device, shared=False)

    def grads(algo: Algorithm) -> tuple[torch.Tensor, torch.Tensor]:
        xi = x.clone().requires_grad_()
        ki = kernel.clone().requires_grad_()
        function.apply(xi, ki, _BORDER, int(algo)).sum().backward()
        assert xi.grad is not None
        assert ki.grad is not None
        return xi.grad, ki.grad

    expected_x, expected_k = grads(Algorithm.AUTO)
    for algo in (*_DIRECT, Algorithm.SEPARABLE):
        grad_x, grad_k = grads(algo)
        # The argmax tie-break is shared by every algorithm, so these are exact, not merely close.
        assert torch.equal(grad_x, expected_x), algo.name
        assert torch.equal(grad_k, expected_k), algo.name


@requires_cuda
def test_forced_separable_rejects_non_flat_se(rng: torch.Generator) -> None:
    x = make_image(rng, (1, 1, 16, 16), device=torch.device("cuda"))
    kernel = torch.randn(5, 5, device=x.device, dtype=x.dtype)

    with pytest.raises(RuntimeError, match="flat"):
        ops.erode(x, kernel, _BORDER, False, int(Algorithm.SEPARABLE))


@requires_cuda
def test_forced_tiled_rejects_window_over_shared_memory() -> None:
    x = torch.zeros(1, 1, 8, 8, device="cuda")
    kernel = torch.zeros(1001, 1001, device="cuda")

    with pytest.raises(RuntimeError, match="tiled"):
        ops.erode(x, kernel, _BORDER, True, int(Algorithm.TILED))


def test_invalid_algo_is_rejected(device: torch.device) -> None:
    x = torch.zeros(1, 1, 8, 8, device=device)
    kernel = torch.zeros(3, 3, device=device)

    with pytest.raises(RuntimeError, match="invalid algo"):
        ops.erode(x, kernel, _BORDER, True, 99)


def test_cpu_only_takes_auto() -> None:
    x = torch.zeros(1, 1, 8, 8)
    kernel = torch.zeros(3, 3)

    assert torch.equal(ops.erode(x, kernel, _BORDER, True, int(Algorithm.AUTO)), serron.functional.erosion(x, kernel))
    with pytest.raises(RuntimeError, match="algo must be auto"):
        ops.erode(x, kernel, _BORDER, True, int(Algorithm.TILED))
