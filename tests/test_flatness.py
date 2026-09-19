"""
Tests for the cached structuring-element flatness that selects the separable path.
"""

from __future__ import annotations

import pytest
import torch

import serron
from serron import _flatness
from serron import structuring_element as se
from tests.conftest import PRIMITIVES, make_image, reference, requires_cuda

# past the default SERRON_SEPARABLE_MIN_K, so a flat SE here really does take the fast path
SEPARABLE_K = 21


def _grayscale_like(kernel: torch.Tensor) -> torch.Tensor:
    """A strongly non-flat pattern shaped like ``kernel``, identical on every device."""
    ramp = torch.arange(kernel.numel(), dtype=kernel.dtype, device=kernel.device)
    return (ramp * 0.01).reshape(kernel.shape)


@pytest.mark.parametrize("op", PRIMITIVES)
def test_in_place_edit_reroutes(op: str, device: torch.device, rng: torch.Generator) -> None:
    """An SE mutated in place is re-measured rather than answered from the stale entry."""
    x = make_image(rng, (1, 2, 16, 18), device=device)
    kernel = torch.zeros(SEPARABLE_K, SEPARABLE_K, dtype=x.dtype, device=device)

    # first use caches "flat" and takes the separable path
    torch.testing.assert_close(getattr(serron, op)(x, kernel), reference(op, x, kernel))

    with torch.no_grad():
        kernel.copy_(_grayscale_like(kernel))

    torch.testing.assert_close(getattr(serron, op)(x, kernel), reference(op, x, kernel))


@pytest.mark.parametrize("op", PRIMITIVES)
def test_learnable_se_after_optimiser_step(op: str, device: torch.device, rng: torch.Generator) -> None:
    """A layer's weight starts all-zero (flat) and stops being flat once it is trained."""
    layer_cls = {"erosion": serron.Erosion2d, "dilation": serron.Dilation2d}[op]
    layer = layer_cls(channels=2, kernel_size=SEPARABLE_K).to(device=device, dtype=torch.float64)
    opt = torch.optim.SGD(layer.parameters(), lr=0.5)
    x = make_image(rng, (1, 2, 16, 18), device=device)

    for _ in range(2):
        opt.zero_grad()
        layer(x).sum().backward()
        opt.step()
        torch.testing.assert_close(layer(x), reference(op, x, layer.weight.detach()))


@pytest.mark.parametrize("op", PRIMITIVES)
def test_load_state_dict_reroutes(op: str, device: torch.device, rng: torch.Generator) -> None:
    """Loading a trained SE over a freshly built (flat) one must not keep the flat answer."""
    layer_cls = {"erosion": serron.Erosion2d, "dilation": serron.Dilation2d}[op]
    layer = layer_cls(channels=2, kernel_size=SEPARABLE_K).to(device=device, dtype=torch.float64)
    x = make_image(rng, (1, 2, 16, 18), device=device)
    torch.testing.assert_close(layer(x), reference(op, x, layer.weight.detach()))

    trained = layer_cls(channels=2, kernel_size=SEPARABLE_K).to(device=device, dtype=torch.float64)
    with torch.no_grad():
        trained.weight.copy_(_grayscale_like(trained.weight))
    layer.load_state_dict(trained.state_dict())

    torch.testing.assert_close(layer(x), reference(op, x, layer.weight.detach()))


def test_builders_seed_the_cache(device: torch.device) -> None:
    """``square`` is a flat rectangle; the masked shapes carry ``-inf`` and are not."""
    assert _flatness.is_flat(se.square(5, device=device)) is True
    for shaped in (se.cross(5, device=device), se.disk(3, device=device), se.diamond(3, device=device)):
        assert _flatness.is_flat(shaped) is False


def test_seeded_answers_never_overclaim(device: torch.device) -> None:
    """A seeded ``True`` has to survive an actual measurement; ``False`` may be conservative."""
    for builder in (se.square(5, device=device), se.cross(5, device=device), se.disk(3, device=device)):
        if _flatness.is_flat(builder):
            assert bool(builder.eq(0).all().item()), "seeded flat but is not flat"


@requires_cuda
def test_operators_are_cuda_graph_capturable() -> None:
    """The cached answer is what keeps the device readback out of the captured region."""
    x = torch.randn(1, 2, 64, 64, device="cuda")
    kernel = se.square(SEPARABLE_K, device="cuda")
    for _ in range(3):
        serron.dilation(x, kernel)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = serron.dilation(x, kernel)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(out, reference("dilation", x, kernel))
