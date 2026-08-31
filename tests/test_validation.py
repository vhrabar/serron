"""
Input-validation tests.
"""

from __future__ import annotations

import pytest
import torch

import serron
from tests.conftest import PREFERRED_DEVICE, flat_se, make_image


@pytest.fixture
def rng() -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(1)
    return g


def test_rejects_non_4d_input() -> None:
    with pytest.raises(RuntimeError, match="4-D"):
        serron.erosion(torch.randn(5, 5, device=PREFERRED_DEVICE), flat_se(1)[0])


def test_rejects_bad_kernel_rank(rng: torch.Generator) -> None:
    x = make_image(rng, (1, 1, 6, 6))
    with pytest.raises(RuntimeError, match="must be 2-D"):
        serron.erosion(x, torch.zeros(3, dtype=torch.float64, device=PREFERRED_DEVICE))


def test_promotes_dtype_mismatch(rng: torch.Generator) -> None:
    x = make_image(rng, (1, 1, 6, 6), dtype=torch.float32)
    out = serron.erosion(x, flat_se(1, dtype=torch.float64))
    assert out.dtype == torch.float64


def test_raw_op_rejects_dtype_mismatch(rng: torch.Generator) -> None:
    x = make_image(rng, (1, 1, 6, 6), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="share a dtype"):
        torch.ops.serron.erode(x, flat_se(1, dtype=torch.float64), 0)


def test_rejects_kernel_channel_mismatch(rng: torch.Generator) -> None:
    x = make_image(rng, (1, 3, 6, 6))
    with pytest.raises(RuntimeError, match="channel"):
        serron.erosion(x, flat_se(2))  # 2-channel SE against a 3-channel image


def test_rejects_nonpositive_kernel(rng: torch.Generator) -> None:
    x = make_image(rng, (1, 1, 6, 6))
    with pytest.raises(RuntimeError, match="positive"):
        serron.erosion(x, torch.zeros(0, 3, dtype=torch.float64, device=PREFERRED_DEVICE))


@pytest.mark.parametrize("border", [-1, 3, 99])
def test_raw_op_rejects_out_of_range_border(border: int, rng: torch.Generator) -> None:
    x = make_image(rng, (1, 1, 6, 6))
    with pytest.raises(RuntimeError, match="border"):
        torch.ops.serron.erode(x, flat_se(1), border)


def test_raw_backward_rejects_grad_output_shape_mismatch(rng: torch.Generator) -> None:
    x = make_image(rng, (1, 1, 6, 6))
    grad_output = make_image(rng, (1, 1, 4, 4))
    with pytest.raises(RuntimeError, match="grad_output"):
        torch.ops.serron.erode_backward(grad_output, x, flat_se(1), 1)


@pytest.mark.parametrize("op", ["erosion", "dilation", "opening", "closing", "gradient", "top_hat", "black_hat"])
def test_all_ops_share_the_same_guards(op: str, rng: torch.Generator) -> None:
    with pytest.raises(RuntimeError):
        getattr(serron.functional, op)(torch.randn(5, 5, device=PREFERRED_DEVICE), flat_se(1)[0])
