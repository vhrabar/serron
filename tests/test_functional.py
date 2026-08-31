"""
Forward-correctness tests for the stateless morphological ops.
"""

from __future__ import annotations

import pytest
import torch

import serron
from serron import BorderMode
from serron import structuring_element as se
from tests.conftest import (
    ALL_OPS,
    BORDERS,
    PREFERRED_DEVICE,
    flat_se,
    make_image,
    make_se,
    reference,
    reference_dilation,
    reference_erosion,
    requires_cuda,
)


def _close(got: torch.Tensor, want: torch.Tensor) -> None:
    torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-9)


# --------------------------------------------------------------------------- #
# Reference parity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("op", ALL_OPS)
@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize("shared", [False, True], ids=["per_channel_se", "shared_se"])
def test_forward_matches_reference(op: str, border: BorderMode, shared: bool, rng: torch.Generator) -> None:
    x = make_image(rng)
    kernel = make_se(rng, channels=x.shape[1], shared=shared)

    out = getattr(serron.functional, op)(x, kernel, border=border)
    expected = reference(op, x, kernel, border)

    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("op", ALL_OPS)
def test_output_shape_preserved(op: str, rng: torch.Generator) -> None:
    x = make_image(rng)
    out = getattr(serron.functional, op)(x, make_se(rng, x.shape[1]))
    assert out.shape == x.shape


@pytest.mark.parametrize("ksize", [2, 4], ids=["k2", "k4"])
@pytest.mark.parametrize("op", ["erosion", "dilatation"])
def test_even_kernel_matches_reference(op: str, ksize: int, rng: torch.Generator) -> None:
    """Even-sized structuring elements anchor at ``k // 2``, and the reference
    anchors them the same way."""
    x = make_image(rng)
    kernel = make_se(rng, x.shape[1], size=ksize)
    out = getattr(serron.functional, op)(x, kernel)
    expected = reference(op, x, kernel)
    assert out.shape == x.shape
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("op", ["erosion", "dilatation"])
def test_kernel_larger_than_image(op: str, rng: torch.Generator) -> None:
    x = make_image(rng, (1, 1, 3, 3))
    kernel = flat_se(1, 5)
    out = getattr(serron.functional, op)(x, kernel, border=BorderMode.CONSTANT)
    torch.testing.assert_close(out, reference(op, x, kernel, BorderMode.CONSTANT))


@pytest.mark.parametrize("op", ALL_OPS)
def test_noncontiguous_input(op: str, rng: torch.Generator) -> None:
    base = make_image(rng, (2, 3, 9, 22))
    x = base[..., ::2]  # strided view, not contiguous; shape (2, 3, 9, 11)
    assert not x.is_contiguous()
    kernel = make_se(rng, x.shape[1])
    got = getattr(serron.functional, op)(x, kernel)
    expected = getattr(serron.functional, op)(x.contiguous(), kernel)
    torch.testing.assert_close(got, expected)


# --------------------------------------------------------------------------- #
# dtype coverage
# --------------------------------------------------------------------------- #


def test_float32_matches_reference(rng: torch.Generator) -> None:
    x = make_image(rng, dtype=torch.float32)
    kernel = make_se(rng, x.shape[1], dtype=torch.float32)
    for op in ALL_OPS:
        out = getattr(serron.functional, op)(x, kernel)
        torch.testing.assert_close(out, reference(op, x, kernel).float(), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("op", ["erosion", "dilatation"])
def test_low_precision_runs_and_tracks_fp32(op: str, dtype: torch.dtype, rng: torch.Generator) -> None:
    x = make_image(rng, (2, 3, 16, 16), dtype=dtype)
    kernel = flat_se(3, 3, dtype=dtype)
    out = getattr(serron.functional, op)(x, kernel)

    assert out.dtype == dtype
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    ref = getattr(serron.functional, op)(x.float(), kernel.float())
    tol = 3e-3 if dtype is torch.float16 else 2e-2
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)


# --------------------------------------------------------------------------- #
# Degenerate shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape",
    [(2, 0, 9, 11), (0, 3, 9, 11), (1, 1, 0, 5), (1, 1, 5, 0)],
    ids=["no_channels", "no_batch", "no_height", "no_width"],
)
def test_empty_tensors(shape: tuple[int, int, int, int], rng: torch.Generator) -> None:
    x = make_image(rng, shape)
    kernel = flat_se(shape[1], 3)
    for op in ("erosion", "dilatation", "opening", "closing"):
        out = getattr(serron.functional, op)(x, kernel)
        assert out.shape == x.shape


@pytest.mark.parametrize("op", ["erosion", "dilatation"])
def test_identity_1x1_kernel(op: str, rng: torch.Generator) -> None:
    """A 1x1 flat SE only ever sees the centre tap, so the image comes back untouched."""
    x = make_image(rng)
    out = getattr(serron.functional, op)(x, flat_se(x.shape[1], 1))
    _close(out, x)


def test_zero_kernel_is_translation_invariant() -> None:
    """A flat image is a fixed point of every op when the SE is flat."""
    x = torch.full((1, 1, 8, 8), 0.42, dtype=torch.float64, device=PREFERRED_DEVICE)
    kernel = flat_se(1, 3)
    for op in ALL_OPS:
        got = getattr(serron.functional, op)(x, kernel)
        want = x if op in ("erosion", "dilatation", "opening", "closing") else torch.zeros_like(x)
        _close(got, want)


# --------------------------------------------------------------------------- #
# Morphological algebra
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("border", BORDERS)
def test_erosion_dilation_duality(border: BorderMode, rng: torch.Generator) -> None:
    """erosion(x, k) == -dilatation(-x, k) for the additive (correlation) form."""
    x = make_image(rng)
    kernel = make_se(rng, x.shape[1])
    ero = serron.functional.erosion(x, kernel, border=border)
    dil = serron.functional.dilatation(-x, kernel, border=border)
    _close(ero, -dil)


@pytest.mark.parametrize("border", BORDERS)
def test_extensivity_with_flat_se(border: BorderMode, rng: torch.Generator) -> None:
    """Flat SE (origin included): erosion <= x <= dilation and opening <= x <= closing."""
    x = make_image(rng)
    kernel = flat_se(x.shape[1])
    fn = serron.functional
    assert (fn.erosion(x, kernel, border=border) <= x + 1e-12).all()
    assert (fn.dilatation(x, kernel, border=border) >= x - 1e-12).all()
    assert (fn.opening(x, kernel, border=border) <= x + 1e-12).all()
    assert (fn.closing(x, kernel, border=border) >= x - 1e-12).all()


@pytest.mark.parametrize("op", ["opening", "closing"])
def test_opening_closing_are_idempotent(op: str, rng: torch.Generator) -> None:
    x = make_image(rng)
    kernel = flat_se(x.shape[1])
    once = getattr(serron.functional, op)(x, kernel)
    twice = getattr(serron.functional, op)(once, kernel)
    _close(twice, once)


@pytest.mark.parametrize("op", ["erosion", "dilatation", "opening", "closing"])
def test_monotonic_increasing(op: str, rng: torch.Generator) -> None:
    """All four operators are increasing: x <= y  =>  op(x) <= op(y)."""
    x = make_image(rng)
    y = x + make_image(rng).abs()
    kernel = make_se(rng, x.shape[1])
    ox = getattr(serron.functional, op)(x, kernel)
    oy = getattr(serron.functional, op)(y, kernel)
    assert (ox <= oy + 1e-12).all()


def test_gradient_identities(rng: torch.Generator) -> None:
    x = make_image(rng)
    kernel = make_se(rng, x.shape[1])
    fn = serron.functional
    _close(fn.gradient(x, kernel), fn.dilatation(x, kernel) - fn.erosion(x, kernel))
    _close(fn.top_hat(x, kernel), x - fn.opening(x, kernel))
    _close(fn.black_hat(x, kernel), fn.closing(x, kernel) - x)
    assert (fn.gradient(x, kernel) >= -1e-12).all()
    assert (fn.top_hat(x, flat_se(x.shape[1])) >= -1e-12).all()
    assert (fn.black_hat(x, flat_se(x.shape[1])) >= -1e-12).all()


# --------------------------------------------------------------------------- #
# Structuring-element builders, end to end
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", ["square", "cross", "disk", "diamond"])
def test_builder_se_matches_reference(shape: str, rng: torch.Generator) -> None:
    x = make_image(rng, (1, 1, 16, 16))
    builder = {"square": se.square(5), "cross": se.cross(5), "disk": se.disk(2), "diamond": se.diamond(2)}[shape]
    kernel = builder.to(x).unsqueeze(0)  # (1, kh, kw), float64
    for op in ("erosion", "dilatation"):
        out = getattr(serron.functional, op)(x, kernel)
        torch.testing.assert_close(out, reference(op, x, kernel))


# --------------------------------------------------------------------------- #
# Cross-device parity
# --------------------------------------------------------------------------- #


@requires_cuda
@pytest.mark.parametrize("op", ALL_OPS)
@pytest.mark.parametrize("border", BORDERS)
def test_cpu_cuda_forward_parity(op: str, border: BorderMode, rng: torch.Generator) -> None:
    x = make_image(rng, device="cpu")
    kernel = make_se(rng, x.shape[1], device="cpu")
    out_cpu = getattr(serron.functional, op)(x, kernel, border=border)
    out_cuda = getattr(serron.functional, op)(x.cuda(), kernel.cuda(), border=border)
    torch.testing.assert_close(out_cuda.cpu(), out_cpu)


# --------------------------------------------------------------------------- #
# Self-check on the reference itself
# --------------------------------------------------------------------------- #


def test_reference_primitives_are_consistent(rng: torch.Generator) -> None:
    x = torch.randn(1, 2, 6, 6, generator=rng, dtype=torch.float64)
    kernel = torch.zeros(2, 3, 3, dtype=torch.float64)
    ero = reference_erosion(x, kernel)
    dil = reference_dilation(x, kernel)
    assert ero.shape == x.shape
    assert torch.all(ero <= dil + 1e-9)
    torch.testing.assert_close(reference("gradient", x, kernel), dil - ero)
    torch.testing.assert_close(reference("opening", x, kernel), reference_dilation(ero, kernel))
