"""
Tests for the learnable ``nn.Module`` morphology layers.
"""

from __future__ import annotations

import pytest
import torch

import serron
from serron import BorderMode
from serron.modules import _MorphologyNd
from tests.conftest import BORDERS, PREFERRED_DEVICE, make_image, make_se, reference, requires_cuda

LAYERS = {
    serron.Erosion2d: "erosion",
    serron.Dilation2d: "dilatation",
    serron.Opening2d: "opening",
    serron.Closing2d: "closing",
}


@pytest.mark.parametrize("cls", list(LAYERS))
def test_state_dict_roundtrip(cls: type[_MorphologyNd]) -> None:
    layer = cls(3, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))

    clone = cls(3, 3)
    clone.load_state_dict(layer.state_dict())
    assert torch.equal(clone.weight, layer.weight)


@pytest.mark.parametrize("cls", list(LAYERS))
def test_weight_is_trainable_parameter(cls: type[_MorphologyNd]) -> None:
    layer = cls(2, 3)
    assert any(p is layer.weight for p in layer.parameters())
    assert sum(p.numel() for p in layer.parameters() if p.requires_grad) == 2 * 3 * 3


@pytest.mark.parametrize("cls", list(LAYERS))
def test_reset_parameters_rezeros(cls: type[_MorphologyNd]) -> None:
    layer = cls(2, 3)
    with torch.no_grad():
        layer.weight.add_(1.5)
    layer.reset_parameters()
    assert torch.all(layer.weight == 0)


@pytest.mark.parametrize("cls, op", list(LAYERS.items()))
@pytest.mark.parametrize("border", BORDERS)
def test_forward_matches_functional(cls: type[_MorphologyNd], op: str, border: BorderMode, rng: torch.Generator) -> None:
    channels, ksize = 2, 3
    layer = cls(channels, ksize, border=border, dtype=torch.float64, device=PREFERRED_DEVICE)
    with torch.no_grad():
        layer.weight.copy_(make_se(rng, channels, ksize))

    x = make_image(rng, (1, channels, 8, 8))
    torch.testing.assert_close(layer(x), reference(op, x, layer.weight.detach(), border))


@pytest.mark.parametrize("cls", list(LAYERS))
def test_gradient_flows_into_weight_and_input(cls: type[_MorphologyNd], rng: torch.Generator) -> None:
    layer = cls(2, 3, dtype=torch.float64, device=PREFERRED_DEVICE)
    with torch.no_grad():
        layer.weight.copy_(make_se(rng, 2))
    x = make_image(rng, (1, 2, 8, 8)).requires_grad_(True)

    layer(x).pow(2).sum().backward()

    assert layer.weight.grad is not None and layer.weight.grad.shape == layer.weight.shape
    assert x.grad is not None and x.grad.shape == x.shape


@pytest.mark.parametrize("cls", list(LAYERS))
def test_single_optimization_step(cls: type[_MorphologyNd], rng: torch.Generator) -> None:
    """One SGD step should push a gradient into the SE and actually move it."""
    channels, ksize = 1, 3
    layer = cls(channels, ksize, dtype=torch.float64, device=PREFERRED_DEVICE)
    with torch.no_grad():
        layer.weight.copy_(make_se(rng, channels, ksize))
    before = layer.weight.detach().clone()

    x = make_image(rng, (1, channels, 8, 8))
    target = torch.zeros_like(layer(x))

    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    opt.zero_grad()
    torch.nn.functional.mse_loss(layer(x), target).backward()  # type: ignore[no-untyped-call]
    assert layer.weight.grad is not None
    opt.step()

    assert not torch.equal(before, layer.weight)


@requires_cuda
@pytest.mark.parametrize("cls", list(LAYERS))
def test_to_device_moves_weight_and_runs(cls: type[_MorphologyNd]) -> None:
    layer = cls(2, 3)
    layer.cuda()
    assert layer.weight.device.type == "cuda"
    y = layer(torch.randn(1, 2, 8, 8, device="cuda"))
    assert y.device.type == "cuda"
    layer.cpu()
    assert layer(torch.randn(1, 2, 8, 8)).device.type == "cpu"
