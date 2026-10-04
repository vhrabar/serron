"""
Autocast behaviour of ``serron::erode`` / ``serron::dilate``.
"""

from __future__ import annotations

import pytest
import torch

import serron
from serron.enums import BorderMode
from serron.functional import BORDER_TO_INT
from tests.conftest import flat_se, make_image

REPLICATE = BORDER_TO_INT[BorderMode.REPLICATE]

PRIMITIVES = ["erosion", "dilation"]


@pytest.fixture
def rng() -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    return g


def low_precision_dtype(device: torch.device) -> torch.dtype:
    """The dtype autocast uses on ``device``: float16 on CUDA, bfloat16 on CPU."""
    return torch.get_autocast_dtype(device.type)


def raw_op(name: str, input_: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Call the registered operator directly, bypassing the autograd wrapper's dtype matching."""
    op = getattr(torch.ops.serron, "erode" if name == "erosion" else "dilate")
    result: torch.Tensor = op(input_, kernel, REPLICATE, None)
    return result


def public_op(name: str, input_: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    result: torch.Tensor = getattr(serron, name)(input_, kernel)
    return result


# --------------------------------------------------------------------------- #
# Execution dtype
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("op", PRIMITIVES)
def test_all_fp32_is_not_downcast(op: str, device: torch.device, rng: torch.Generator) -> None:
    """
    float32 in, float32 out, bit-for-bit the same as outside autocast.

    ``promote_type`` prefers float32 over the lower-precision dtype, so autocast is a no-op here. This is the
    behaviour a fixed ``cast_inputs`` would change.
    """
    x = make_image(rng, (1, 1, 12, 12), dtype=torch.float32, device=device)
    k = flat_se(1, shared=True, dtype=torch.float32, device=device)

    with torch.autocast(device_type=device.type, dtype=low_precision_dtype(device)):
        under = public_op(op, x, k)
    outside = public_op(op, x, k)

    assert under.dtype == torch.float32
    assert torch.equal(under, outside)


@pytest.mark.parametrize("op", PRIMITIVES)
def test_low_precision_inputs_stay_low(op: str, device: torch.device, rng: torch.Generator) -> None:
    """Both arguments already low precision: the op runs there, and the result is unchanged by autocast."""
    low = low_precision_dtype(device)
    x = make_image(rng, (1, 1, 12, 12), dtype=low, device=device)
    k = flat_se(1, shared=True, dtype=low, device=device)

    with torch.autocast(device_type=device.type, dtype=low):
        under = public_op(op, x, k)
    outside = public_op(op, x, k)

    assert under.dtype == low
    assert torch.equal(under, outside)


@pytest.mark.parametrize("op", PRIMITIVES)
@pytest.mark.parametrize("low_arg", ["input", "kernel"])
def test_mixed_precision_promotes_to_fp32(op: str, low_arg: str, device: torch.device, rng: torch.Generator) -> None:
    """
    One argument low precision and one float32 promotes the whole op to float32, not down to the low dtype.

    This is also what proves the autocast kernel is registered at all: the operator itself requires a shared
    dtype, so the same call outside autocast raises. Tests that only check matching dtypes would still pass with
    the registration removed.
    """
    low = low_precision_dtype(device)
    x_dtype = low if low_arg == "input" else torch.float32
    k_dtype = low if low_arg == "kernel" else torch.float32

    x = make_image(rng, (1, 1, 12, 12), dtype=x_dtype, device=device)
    k = flat_se(1, shared=True, dtype=k_dtype, device=device)

    with torch.autocast(device_type=device.type, dtype=low):
        assert raw_op(op, x, k).dtype == torch.float32

    with pytest.raises(RuntimeError, match="share a dtype"):
        raw_op(op, x, k)


@pytest.mark.parametrize("op", PRIMITIVES)
def test_double_is_left_alone(op: str, device: torch.device, rng: torch.Generator) -> None:
    """``promote_type`` ignores float64 arguments, and float64 is never cast down."""
    x = make_image(rng, (1, 1, 12, 12), dtype=torch.float64, device=device)
    k = flat_se(1, shared=True, dtype=torch.float64, device=device)

    with torch.autocast(device_type=device.type, dtype=low_precision_dtype(device)):
        under = public_op(op, x, k)
    outside = public_op(op, x, k)

    assert under.dtype == torch.float64
    assert torch.equal(under, outside)


# --------------------------------------------------------------------------- #
# The two call paths differ
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("op", PRIMITIVES)
def test_public_api_matches_dtypes_before_the_op_sees_them(op: str, device: torch.device, rng: torch.Generator) -> None:
    """
    ``_match_dtype`` promotes float32 and float64 to float64 before dispatch, so the public API accepts a pair
    that the operator itself rejects.
    """
    x = make_image(rng, (1, 1, 12, 12), dtype=torch.float32, device=device)
    k = flat_se(1, shared=True, dtype=torch.float64, device=device)

    with torch.autocast(device_type=device.type, dtype=low_precision_dtype(device)):
        assert public_op(op, x, k).dtype == torch.float64

        with pytest.raises(RuntimeError, match="share a dtype"):
            raw_op(op, x, k)


@pytest.mark.parametrize("op", PRIMITIVES)
def test_autocast_dtype_unlike_the_input_dtype_is_rejected(op: str, device: torch.device, rng: torch.Generator) -> None:
    """
    Low-precision inputs under a *different* low-precision autocast dtype are an error, because
    ``at::autocast::prioritize`` has no rule for the pair.
    """
    low = low_precision_dtype(device)
    other = torch.bfloat16 if low is torch.float16 else torch.float16

    x = make_image(rng, (1, 1, 12, 12), dtype=low, device=device)
    k = flat_se(1, shared=True, dtype=low, device=device)

    with torch.autocast(device_type=device.type, dtype=other), pytest.raises(RuntimeError, match="ScalarType"):
        raw_op(op, x, k)


# --------------------------------------------------------------------------- #
# Backward
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("op", PRIMITIVES)
def test_backward_under_autocast_keeps_the_leaf_dtype(op: str, device: torch.device, rng: torch.Generator) -> None:
    """
    The backward operators carry no autocast registration, so gradients come back in the dtype the forward ran
    at — float32 here, since both leaves are float32.
    """
    x = make_image(rng, (1, 1, 12, 12), dtype=torch.float32, device=device).requires_grad_()
    k = make_image(rng, (3, 3), dtype=torch.float32, device=device).requires_grad_()

    with torch.autocast(device_type=device.type, dtype=low_precision_dtype(device)):
        out = public_op(op, x, k)
    out.sum().backward()

    assert out.dtype == torch.float32
    assert x.grad is not None and x.grad.dtype == torch.float32
    assert k.grad is not None and k.grad.dtype == torch.float32
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("op", PRIMITIVES)
def test_backward_under_autocast_with_low_precision_leaves(op: str, device: torch.device, rng: torch.Generator) -> None:
    """With low-precision leaves the forward runs low, and the gradients follow."""
    low = low_precision_dtype(device)
    x = make_image(rng, (1, 1, 12, 12), dtype=low, device=device).requires_grad_()
    k = make_image(rng, (3, 3), dtype=low, device=device).requires_grad_()

    with torch.autocast(device_type=device.type, dtype=low):
        out = public_op(op, x, k)
    out.sum().backward()

    assert out.dtype == low
    assert x.grad is not None and x.grad.dtype == low
    assert k.grad is not None and k.grad.dtype == low
