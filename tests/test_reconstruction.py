"""
Geodesic reconstruction: parity with the pure-PyTorch reference, the algebraic
invariants of the fixed point, and input validation.
"""

from __future__ import annotations

import pytest
import torch

import serron
from serron import BorderMode
from serron import structuring_element as se
from tests.conftest import (
    DEVICES,
    PREFERRED_DEVICE,
    flat_se,
    make_image,
    reference_reconstruction,
)

VARIANTS = ["reconstruction_by_dilation", "reconstruction_by_erosion"]


def _call(variant: str, marker: torch.Tensor, mask: torch.Tensor, kernel: torch.Tensor, **kwargs: object) -> torch.Tensor:
    fn = getattr(serron, variant)
    result: torch.Tensor = fn(marker, mask, kernel, **kwargs)
    return result


def _seed(variant: str, mask: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Draw a marker on the correct side of ``mask`` for ``variant``."""
    offset = torch.rand(mask.shape, generator=rng, dtype=mask.dtype).to(mask.device)
    if variant == "reconstruction_by_dilation":
        return mask - 4.0 * offset
    return mask + 4.0 * offset


def _kernels(channels: int, device: torch.device | str) -> dict[str, torch.Tensor]:
    return {
        "square3": se.square(3, dtype=torch.float64, device=device),
        "cross3": se.cross(3, dtype=torch.float64, device=device),
        "diamond2": se.diamond(2, dtype=torch.float64, device=device),
        "per_channel": flat_se(channels, 3, device=device),
    }


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("kernel_name", ["square3", "cross3", "diamond2", "per_channel"])
def test_matches_reference(variant: str, kernel_name: str, rng: torch.Generator) -> None:
    mask = make_image(rng, (2, 3, 9, 11))
    marker = _seed(variant, mask, rng)
    kernel = _kernels(mask.shape[1], mask.device)[kernel_name]

    got = _call(variant, marker, mask, kernel)
    want = reference_reconstruction(marker, mask, kernel, dilate=variant == "reconstruction_by_dilation")
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("border", [BorderMode.CONSTANT, BorderMode.REPLICATE, BorderMode.REFLECT])
def test_matches_reference_per_border(variant: str, border: BorderMode, rng: torch.Generator) -> None:
    mask = make_image(rng, (1, 2, 7, 8))
    marker = _seed(variant, mask, rng)
    kernel = se.square(3, dtype=torch.float64, device=mask.device)

    got = _call(variant, marker, mask, kernel, border=border)
    want = reference_reconstruction(marker, mask, kernel, border, dilate=variant == "reconstruction_by_dilation")
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("check_every", [1, 3, 16, 1024])
def test_check_every_does_not_change_the_result(variant: str, check_every: int, rng: torch.Generator) -> None:
    """The convergence test is an optimisation; the fixed point it lands on is not."""
    mask = make_image(rng, (1, 1, 12, 12))
    marker = _seed(variant, mask, rng)
    kernel = se.cross(3, dtype=torch.float64, device=mask.device)

    baseline = _call(variant, marker, mask, kernel, check_every=1)
    torch.testing.assert_close(_call(variant, marker, mask, kernel, check_every=check_every), baseline)


@pytest.mark.skipif(len(DEVICES) < 2, reason="needs both a CPU and a CUDA device")
@pytest.mark.parametrize("variant", VARIANTS)
def test_cpu_and_cuda_agree(variant: str, rng: torch.Generator) -> None:
    mask = make_image(rng, (2, 2, 10, 13), device="cpu")
    marker = _seed(variant, mask, rng)
    kernel = se.disk(2, dtype=torch.float64, device="cpu")

    on_cpu = _call(variant, marker, mask, kernel)
    on_cuda = _call(variant, marker.cuda(), mask.cuda(), kernel.cuda())
    torch.testing.assert_close(on_cuda.cpu(), on_cpu)


# --------------------------------------------------------------------------- #
# Fixed-point invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", VARIANTS)
def test_result_is_a_fixed_point(variant: str, rng: torch.Generator) -> None:
    """One more geodesic step off the result must leave it where it is."""
    mask = make_image(rng, (1, 1, 11, 11))
    marker = _seed(variant, mask, rng)
    kernel = se.cross(3, dtype=torch.float64, device=mask.device)

    out = _call(variant, marker, mask, kernel)
    if variant == "reconstruction_by_dilation":
        stepped = torch.minimum(serron.dilation(out, kernel, border=BorderMode.CONSTANT), mask)
    else:
        stepped = torch.maximum(serron.erosion(out, kernel, border=BorderMode.CONSTANT), mask)
    torch.testing.assert_close(stepped, out)


@pytest.mark.parametrize("variant", VARIANTS)
def test_idempotent(variant: str, rng: torch.Generator) -> None:
    mask = make_image(rng, (1, 2, 9, 9))
    marker = _seed(variant, mask, rng)
    kernel = se.square(3, dtype=torch.float64, device=mask.device)

    once = _call(variant, marker, mask, kernel)
    torch.testing.assert_close(_call(variant, once, mask, kernel), once)


def test_dilation_stays_between_marker_and_mask(rng: torch.Generator) -> None:
    mask = make_image(rng, (1, 2, 9, 9))
    marker = _seed("reconstruction_by_dilation", mask, rng)
    kernel = se.square(3, dtype=torch.float64, device=mask.device)

    out = serron.reconstruction_by_dilation(marker, mask, kernel)
    assert bool((out >= torch.minimum(marker, mask)).all())
    assert bool((out <= mask).all())


def test_erosion_stays_between_mask_and_marker(rng: torch.Generator) -> None:
    mask = make_image(rng, (1, 2, 9, 9))
    marker = _seed("reconstruction_by_erosion", mask, rng)
    kernel = se.square(3, dtype=torch.float64, device=mask.device)

    out = serron.reconstruction_by_erosion(marker, mask, kernel)
    assert bool((out <= torch.maximum(marker, mask)).all())
    assert bool((out >= mask).all())


@pytest.mark.parametrize("variant", VARIANTS)
def test_mask_as_marker_returns_mask(variant: str, rng: torch.Generator) -> None:
    mask = make_image(rng, (1, 1, 8, 8))
    kernel = se.square(3, dtype=torch.float64, device=mask.device)
    torch.testing.assert_close(_call(variant, mask.clone(), mask, kernel), mask)


@pytest.mark.parametrize("variant", VARIANTS)
def test_marker_is_clipped_into_the_mask(variant: str, rng: torch.Generator) -> None:
    """A marker on the wrong side of the mask is clipped, not rejected."""
    mask = make_image(rng, (1, 1, 8, 8))
    kernel = se.square(3, dtype=torch.float64, device=mask.device)
    marker = _seed(variant, mask, rng)
    clip = torch.minimum if variant == "reconstruction_by_dilation" else torch.maximum

    out_of_range = marker + (4.0 if variant == "reconstruction_by_dilation" else -4.0)
    torch.testing.assert_close(
        _call(variant, out_of_range, mask, kernel),
        _call(variant, clip(out_of_range, mask), mask, kernel),
    )


def test_duality(rng: torch.Generator) -> None:
    """Negating both inputs turns one variant into the other, for a symmetric SE."""
    mask = make_image(rng, (1, 1, 9, 9))
    marker = _seed("reconstruction_by_dilation", mask, rng)
    kernel = se.square(3, dtype=torch.float64, device=mask.device)

    by_dilation = serron.reconstruction_by_dilation(marker, mask, kernel)
    by_erosion = serron.reconstruction_by_erosion(-marker, -mask, kernel)
    torch.testing.assert_close(by_erosion, -by_dilation)


# --------------------------------------------------------------------------- #
# Propagation over long geodesic paths
# --------------------------------------------------------------------------- #


def _corridor(device: torch.device | str) -> torch.Tensor:
    """A serpentine of open rows joined at alternating ends, in a 13x13 frame.

    Its geodesic length is far longer than its diameter, so a reconstruction has
    to keep iterating well past the point a blocked convergence test first looks.
    """
    grid = torch.zeros(13, 13, dtype=torch.float64, device=device)
    for row in range(0, 13, 2):
        grid[row, :] = 1.0
        if row + 1 < 13:
            grid[row + 1, 12 if (row // 2) % 2 == 0 else 0] = 1.0
    return grid.reshape(1, 1, 13, 13)


@pytest.mark.parametrize("check_every", [1, 16, 256])
def test_floods_a_long_corridor(check_every: int) -> None:
    mask = _corridor(PREFERRED_DEVICE)
    marker = torch.zeros_like(mask)
    marker[0, 0, 0, 0] = 1.0

    out = serron.reconstruction_by_dilation(
        marker, mask, se.cross(3, dtype=torch.float64, device=mask.device), check_every=check_every
    )
    torch.testing.assert_close(out, mask)


def test_walls_block_propagation() -> None:
    """A seed outside the corridor reaches nothing but its own cell."""
    mask = _corridor(PREFERRED_DEVICE)
    marker = torch.zeros_like(mask)
    marker[0, 0, 1, 6] = 1.0  # a closed cell between two open rows

    out = serron.reconstruction_by_dilation(marker, mask, se.cross(3, dtype=torch.float64, device=mask.device))
    torch.testing.assert_close(out, torch.zeros_like(mask))


def test_flood_is_bounded_by_the_mask_profile() -> None:
    """A peak spreads only as high as the mask allows along the way."""
    mask = torch.tensor([[[[1.0, 2.0, 3.0], [0.0, 0.0, 4.0], [9.0, 0.0, 5.0]]]], dtype=torch.float64, device=PREFERRED_DEVICE)
    marker = torch.zeros_like(mask)
    marker[0, 0, 0, 0] = 1.0

    out = serron.reconstruction_by_dilation(marker, mask, se.cross(3, dtype=torch.float64, device=mask.device))
    want = torch.tensor([[[[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]], dtype=torch.float64, device=PREFERRED_DEVICE)
    torch.testing.assert_close(out, want)


# --------------------------------------------------------------------------- #
# dtype and validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", VARIANTS)
def test_promotes_to_the_common_dtype(variant: str, rng: torch.Generator) -> None:
    mask = make_image(rng, (1, 1, 6, 6), dtype=torch.float32)
    marker = _seed(variant, mask, rng)
    kernel = se.square(3, dtype=torch.float64, device=mask.device)

    assert _call(variant, marker, mask, kernel).dtype == torch.float64


@pytest.mark.parametrize("variant", VARIANTS)
def test_rejects_shape_mismatch(variant: str, rng: torch.Generator) -> None:
    mask = make_image(rng, (1, 1, 6, 6))
    marker = make_image(rng, (1, 1, 6, 7))
    with pytest.raises(ValueError, match="share a shape"):
        _call(variant, marker, mask, se.square(3, dtype=torch.float64, device=mask.device))


@pytest.mark.parametrize("variant", VARIANTS)
def test_rejects_inputs_wanting_a_gradient(variant: str, rng: torch.Generator) -> None:
    mask = make_image(rng, (1, 1, 6, 6))
    marker = _seed(variant, mask, rng).requires_grad_(True)
    with pytest.raises(RuntimeError, match="no backward pass"):
        _call(variant, marker, mask, se.square(3, dtype=torch.float64, device=mask.device))


@pytest.mark.parametrize("variant", VARIANTS)
def test_runs_under_no_grad_despite_requires_grad(variant: str, rng: torch.Generator) -> None:
    mask = make_image(rng, (1, 1, 6, 6))
    marker = _seed(variant, mask, rng).requires_grad_(True)
    with torch.no_grad():
        out = _call(variant, marker, mask, se.square(3, dtype=torch.float64, device=mask.device))
    assert not out.requires_grad


@pytest.mark.parametrize("variant", VARIANTS)
def test_rejects_negative_anchor_tap(variant: str, rng: torch.Generator) -> None:
    mask = make_image(rng, (1, 1, 6, 6))
    marker = _seed(variant, mask, rng)
    kernel = torch.zeros(3, 3, dtype=torch.float64, device=mask.device)
    kernel[1, 1] = -0.5
    with pytest.raises(ValueError, match="anchor tap"):
        _call(variant, marker, mask, kernel)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("kwargs", [{"check_every": 0}, {"check_every": -1}, {"max_iter": 0}, {"max_iter": -3}])
def test_rejects_out_of_range_budgets(variant: str, kwargs: dict[str, int], rng: torch.Generator) -> None:
    mask = make_image(rng, (1, 1, 6, 6))
    marker = _seed(variant, mask, rng)
    with pytest.raises(ValueError, match="must be >="):
        _call(variant, marker, mask, se.square(3, dtype=torch.float64, device=mask.device), **kwargs)


def test_raises_when_the_budget_runs_out() -> None:
    mask = _corridor(PREFERRED_DEVICE)
    marker = torch.zeros_like(mask)
    marker[0, 0, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="no fixed point"):
        serron.reconstruction_by_dilation(marker, mask, se.cross(3, dtype=torch.float64, device=mask.device), max_iter=3)


def test_budget_that_covers_convergence_is_fine() -> None:
    mask = _corridor(PREFERRED_DEVICE)
    marker = torch.zeros_like(mask)
    marker[0, 0, 0, 0] = 1.0
    out = serron.reconstruction_by_dilation(marker, mask, se.cross(3, dtype=torch.float64, device=mask.device), max_iter=4096)
    torch.testing.assert_close(out, mask)
