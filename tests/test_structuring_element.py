"""
Structuring-element builder tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

import serron
from serron import BorderMode
from serron import structuring_element as se
from tests.conftest import DEVICES, PREFERRED_DEVICE, requires_cuda


def _members(el: torch.Tensor) -> torch.Tensor:
    """Which taps are 'on'.

    :param el: a flat structuring element.
    :returns: boolean mask, ``True`` where ``el`` is ``0.0`` and ``False`` where
        it is ``-inf``.
    """
    return el == 0.0


def _is_flat(el: torch.Tensor) -> bool:
    """Check that a structuring element only uses the two flat values.

    :param el: the tensor to inspect.
    :returns: ``True`` if every entry is either ``0.0`` (on) or ``-inf`` (off).
    """
    return bool(torch.all((el == 0.0) | (el == float("-inf"))))


@pytest.mark.parametrize("size", [1, 3, 5])
def test_square_shape_and_full(size: int) -> None:
    el = se.square(size)
    assert el.shape == (size, size)
    assert el.dtype == torch.float32
    # A square covers the whole window, so every pixel is a member.
    assert torch.all(el == 0.0)


@pytest.mark.parametrize("size", [3, 5])
def test_cross_pattern_and_symmetry(size: int) -> None:
    el = se.cross(size)
    assert el.shape == (size, size)
    assert _is_flat(el)
    assert torch.equal(el, el.flip(0))
    assert torch.equal(el, el.flip(1))

    c = size // 2
    expected = torch.full((size, size), float("-inf"))
    expected[c, :] = 0.0
    expected[:, c] = 0.0
    assert torch.equal(el, expected)


def test_cross_is_not_square() -> None:
    """A cross has to drop its corners -- otherwise it's just a square."""
    el = se.cross(3)
    assert el[0, 0] == float("-inf")
    assert not torch.equal(el, se.square(3))


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_disk_shape_symmetry_and_membership(radius: int) -> None:
    el = se.disk(radius)
    n = 2 * radius + 1
    assert el.shape == (n, n)
    assert _is_flat(el)
    assert torch.equal(el, el.flip(0))
    assert torch.equal(el, el.flip(1))

    members = _members(el)
    for i in range(n):
        for j in range(n):
            dy, dx = i - radius, j - radius
            assert members[i, j].item() == (dx * dx + dy * dy <= radius * radius)
    # The bounding-box corners fall outside the disk.
    assert el[0, 0] == float("-inf")


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_diamond_shape_and_membership(radius: int) -> None:
    el = se.diamond(radius)
    n = 2 * radius + 1
    assert el.shape == (n, n)
    assert _is_flat(el)
    assert torch.equal(el, el.flip(0))
    assert torch.equal(el, el.flip(1))

    members = _members(el)
    for i in range(n):
        for j in range(n):
            dy, dx = i - radius, j - radius
            assert members[i, j].item() == (abs(dx) + abs(dy) <= radius)


def test_from_tensor_roundtrip() -> None:
    weights = torch.arange(9, dtype=torch.float64).reshape(3, 3)
    el = se.from_tensor(weights)
    assert el.shape == weights.shape
    assert el.dtype == torch.float32
    torch.testing.assert_close(el, weights.to(torch.float32))


def test_from_tensor_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        se.from_tensor(torch.zeros(3, 3, 3))


@pytest.mark.parametrize(
    "builder, arg",
    [(se.square, 0), (se.cross, 0), (se.disk, -1), (se.diamond, -1)],
)
def test_builders_reject_bad_size(builder: Callable[[int], torch.Tensor], arg: int) -> None:
    with pytest.raises(ValueError):
        builder(arg)


@pytest.mark.parametrize("device", DEVICES)
def test_builders_honor_device(device: str) -> None:
    assert se.square(3, device=device).device.type == device
    assert se.cross(3, device=device).device.type == device
    assert se.disk(2, device=device).device.type == device
    assert se.diamond(2, device=device).device.type == device


@requires_cuda
def test_flat_se_behaves_like_masked_min_max() -> None:
    """Under a flat SE, erosion/dilation should equal a plain min/max over the footprint.

    This runs the ``-inf`` exclusion end to end through the CUDA kernels: the
    off-shape pixels have to fall out of both the erosion min and the dilation
    max.
    """
    x = torch.randn(1, 1, 12, 12, dtype=torch.float64, device=PREFERRED_DEVICE)
    kernel = se.disk(2, device=PREFERRED_DEVICE).to(torch.float64).unsqueeze(0)  # -> (C=1, kh, kw)
    members = _members(kernel[0])

    ero = serron.functional.erosion(x, kernel, border=BorderMode.REPLICATE)
    dil = serron.functional.dilation(x, kernel, border=BorderMode.REPLICATE)

    # Reference: masked min/max over the disk footprint, REPLICATE padding.
    xp = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="replicate")
    patches = xp.unfold(2, 5, 1).unfold(3, 5, 1)  # (1, 1, 12, 12, 5, 5)
    m = members.view(1, 1, 1, 1, 5, 5)
    ref_ero = patches.masked_fill(~m, math.inf).amin(dim=(-1, -2))
    ref_dil = patches.masked_fill(~m, -math.inf).amax(dim=(-1, -2))

    torch.testing.assert_close(ero, ref_ero)
    torch.testing.assert_close(dil, ref_dil)
