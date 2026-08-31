"""
Backward-pass and gradient-check tests.

"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import serron
from serron import BorderMode
from tests.conftest import (
    ALL_OPS,
    BORDERS,
    PRIMITIVES,
    make_image,
    make_se,
    reference,
    requires_cuda,
)


def _gradcheck(func: Callable[..., torch.Tensor], inputs: tuple[torch.Tensor, ...]) -> bool:
    return bool(torch.autograd.gradcheck(func, inputs, atol=1e-4, rtol=1e-3, fast_mode=True, nondet_tol=1e-5))


def _inputs(rng: torch.Generator, shape: tuple[int, int, int, int] = (1, 2, 5, 5)) -> tuple[torch.Tensor, torch.Tensor]:
    x = make_image(rng, shape)
    kernel = make_se(rng, shape[1])
    return x, kernel


@pytest.mark.parametrize("op", ALL_OPS)
def test_backward_runs_and_shapes_match(op: str, rng: torch.Generator) -> None:
    x, kernel = _inputs(rng, (1, 2, 6, 7))
    x = x.detach().requires_grad_(True)
    kernel = kernel.detach().requires_grad_(True)

    getattr(serron.functional, op)(x, kernel).sum().backward()

    assert x.grad is not None and x.grad.shape == x.shape
    assert kernel.grad is not None and kernel.grad.shape == kernel.shape


@pytest.mark.parametrize("wrt", ["input", "kernel"])
@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize("op", ALL_OPS)
def test_gradcheck(op: str, border: BorderMode, wrt: str, rng: torch.Generator) -> None:
    x, kernel = _inputs(rng)
    func = getattr(serron.functional, op)
    if wrt == "input":
        x = x.detach().requires_grad_(True)
        assert _gradcheck(lambda t: func(t, kernel, border=border), (x,))
    else:
        kernel = kernel.detach().requires_grad_(True)
        assert _gradcheck(lambda k: func(x, k, border=border), (kernel,))


@pytest.mark.parametrize("wrt", ["input", "kernel"])
@pytest.mark.parametrize("op", PRIMITIVES)
def test_gradcheck_shared_se(op: str, wrt: str, rng: torch.Generator) -> None:
    x = make_image(rng, (1, 2, 5, 5))
    kernel = make_se(rng, 2, shared=True)
    func = getattr(serron.functional, op)
    if wrt == "input":
        x = x.detach().requires_grad_(True)
        assert _gradcheck(lambda t: func(t, kernel), (x,))
    else:
        kernel = kernel.detach().requires_grad_(True)
        assert _gradcheck(lambda k: func(x, k), (kernel,))


@pytest.mark.parametrize("wrt", ["input", "kernel"])
@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize("op", ALL_OPS)
def test_backward_matches_reference_grad(op: str, border: BorderMode, wrt: str, rng: torch.Generator) -> None:
    """The kernel's analytic gradient should match autograd run through the reference."""
    x, kernel = _inputs(rng, (1, 2, 6, 7))

    xs = x.detach().requires_grad_(True)
    ks = kernel.detach().requires_grad_(True)
    getattr(serron.functional, op)(xs, ks, border=border).sum().backward()

    xr = x.detach().requires_grad_(True)
    kr = kernel.detach().requires_grad_(True)
    reference(op, xr, kr, border).sum().backward()  # type: ignore[no-untyped-call]

    got, want = (xs.grad, xr.grad) if wrt == "input" else (ks.grad, kr.grad)
    assert got is not None and want is not None
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("op", PRIMITIVES)
def test_zero_grad_output_gives_zero_grad(op: str, rng: torch.Generator) -> None:
    x, kernel = _inputs(rng)
    x = x.detach().requires_grad_(True)
    kernel = kernel.detach().requires_grad_(True)
    out = getattr(serron.functional, op)(x, kernel)
    out.backward(torch.zeros_like(out))
    assert x.grad is not None and kernel.grad is not None
    assert torch.count_nonzero(x.grad) == 0
    assert torch.count_nonzero(kernel.grad) == 0


@pytest.mark.parametrize("op", PRIMITIVES)
def test_grad_input_is_a_permutation_of_grad_output(op: str, rng: torch.Generator) -> None:
    """Every output pixel sends its upstream gradient to exactly one input pixel,
    so the two sums have to match: sum(grad_input) == sum(grad_output)."""
    x, kernel = _inputs(rng, (1, 2, 8, 8))
    x = x.detach().requires_grad_(True)
    go = make_image(rng, (1, 2, 8, 8))
    getattr(serron.functional, op)(x, kernel).backward(go)
    assert x.grad is not None
    torch.testing.assert_close(x.grad.sum(), go.sum())


@requires_cuda
@pytest.mark.parametrize("wrt", ["input", "kernel"])
@pytest.mark.parametrize("op", ALL_OPS)
def test_cpu_cuda_backward_parity(op: str, wrt: str, rng: torch.Generator) -> None:
    x = make_image(rng, (1, 2, 6, 7), device="cpu")
    kernel = make_se(rng, 2, device="cpu")
    go = make_image(rng, (1, 2, 6, 7), device="cpu")

    def grads(dev: str) -> torch.Tensor:
        xt = x.to(dev).detach().requires_grad_(True)
        kt = kernel.to(dev).detach().requires_grad_(True)
        getattr(serron.functional, op)(xt, kt).backward(go.to(dev))
        grad = xt.grad if wrt == "input" else kt.grad
        assert grad is not None
        return grad.cpu()

    torch.testing.assert_close(grads("cuda"), grads("cpu"))
