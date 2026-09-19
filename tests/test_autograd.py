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


@pytest.mark.parametrize("op", PRIMITIVES)
@pytest.mark.parametrize("ksize", [3, 21], ids=["direct", "separable"])
def test_grad_input_unaffected_by_se_requiring_grad(op: str, ksize: int, device: torch.device, rng: torch.Generator) -> None:
    """The backward skips the grad_kernel scatter for a fixed SE; grad_input must not move."""
    base = make_image(rng, (1, 2, 12, 14), device=device)
    kernel = torch.zeros(ksize, ksize, dtype=base.dtype, device=device)

    fixed_x = base.clone().requires_grad_(True)
    getattr(serron, op)(fixed_x, kernel).sum().backward()

    learned_x = base.clone().requires_grad_(True)
    learned_k = kernel.clone().requires_grad_(True)
    getattr(serron, op)(learned_x, learned_k).sum().backward()

    assert torch.equal(fixed_x.grad, learned_x.grad)
    assert learned_k.grad is not None


@pytest.mark.parametrize("op", PRIMITIVES)
def test_no_gradient_requested_returns_nothing(op: str, device: torch.device, rng: torch.Generator) -> None:
    """With neither input requiring grad the op produces no graph at all."""
    x = make_image(rng, (1, 2, 8, 9), device=device)
    kernel = torch.zeros(3, 3, dtype=x.dtype, device=device)

    out = getattr(serron, op)(x, kernel)

    assert out.grad_fn is None
    assert not out.requires_grad


@pytest.mark.parametrize("op", PRIMITIVES)
@pytest.mark.parametrize("ksize", [3, 21], ids=["direct", "separable"])
def test_tie_break_keeps_the_lowest_offset(op: str, ksize: int, device: torch.device) -> None:
    """On a constant image every tap in the window ties, so the winner is decided purely by
    the tie-break rule. The kernels keep the lowest ``(di, dj)``, which puts the gradient at
    each window's top-left corner
    """
    shape = (1, 1, 9, 10)
    anchor = ksize // 2
    x = torch.full(shape, 2.5, dtype=torch.float64, device=device, requires_grad=True)
    kernel = torch.zeros(ksize, ksize, dtype=torch.float64, device=device)

    getattr(serron, op)(x, kernel, border=BorderMode.REPLICATE).sum().backward()

    expected = torch.zeros(shape, dtype=torch.float64, device=device)
    for h in range(shape[2]):
        for w in range(shape[3]):
            ih = min(max(h - anchor, 0), shape[2] - 1)
            iw = min(max(w - anchor, 0), shape[3] - 1)
            expected[:, :, ih, iw] += 1.0

    assert torch.equal(x.grad, expected)


@requires_cuda
@pytest.mark.parametrize("op", PRIMITIVES)
@pytest.mark.parametrize("ksize", [3, 21], ids=["direct", "separable"])
def test_tie_break_matches_across_devices(op: str, ksize: int, rng: torch.Generator) -> None:
    """A coarsely quantised image ties often; both backends must resolve those the same way."""
    quantised = (make_image(rng, (1, 2, 16, 18), device="cpu") * 2).round()

    grads = []
    for device in ("cpu", "cuda"):
        x = quantised.clone().to(device).requires_grad_(True)
        kernel = torch.zeros(ksize, ksize, dtype=x.dtype, device=device)
        getattr(serron, op)(x, kernel).sum().backward()
        grads.append(x.grad.cpu())

    assert torch.equal(grads[0], grads[1])
