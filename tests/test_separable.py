"""
Parity tests for the flat-SE separable (van Herk / Gil-Werman) path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import serron
from serron import BorderMode
from tests._separable_worker import run_cases
from tests.conftest import BORDERS, PRIMITIVES, flat_se, make_image, reference

# force direct path
_FORCE_DIRECT = "1000000000"

SEPARABLE_K = 21
BELOW_K = 3


@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize("op", PRIMITIVES)
@pytest.mark.parametrize("shared", [False, True], ids=["per_channel_se", "shared_se"])
def test_forward_matches_reference(op: str, border: BorderMode, shared: bool, device: torch.device, rng: torch.Generator) -> None:
    x = make_image(rng, (2, 3, 24, 27), device=device)
    kernel = flat_se(x.shape[1], SEPARABLE_K, device=device, shared=shared)

    out = getattr(serron.functional, op)(x, kernel, border=border)

    torch.testing.assert_close(out, reference(op, x, kernel, border))


@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize("op", PRIMITIVES)
@pytest.mark.parametrize("ksize", [(SEPARABLE_K, BELOW_K), (BELOW_K, SEPARABLE_K), (SEPARABLE_K, SEPARABLE_K - 1)])
def test_forward_non_square_se(
    op: str, ksize: tuple[int, int], border: BorderMode, device: torch.device, rng: torch.Generator
) -> None:
    """The row pass takes kW and the column pass kH; swapping them would survive a
    square kernel but not these."""
    x = make_image(rng, (1, 2, 17, 23), device=device)
    kernel = torch.zeros(x.shape[1], *ksize, dtype=x.dtype, device=device)

    out = getattr(serron.functional, op)(x, kernel, border=border)

    torch.testing.assert_close(out, reference(op, x, kernel, border))


@pytest.mark.parametrize("border", [BorderMode.REPLICATE, BorderMode.CONSTANT])
@pytest.mark.parametrize("op", PRIMITIVES)
def test_forward_window_wider_than_image(op: str, border: BorderMode, device: torch.device, rng: torch.Generator) -> None:
    """A window longer than the line makes the halo wrap the image several times.

    REFLECT is covered by :func:`test_matches_direct_path` instead: the reference pads
    with ``F.pad``, which refuses a reflect pad wider than the image."""
    x = make_image(rng, (1, 1, 5, 7), device=device)
    kernel = flat_se(1, SEPARABLE_K, device=device)

    out = getattr(serron.functional, op)(x, kernel, border=border)

    torch.testing.assert_close(out, reference(op, x, kernel, border))


@pytest.mark.parametrize("op", PRIMITIVES)
def test_forward_float32_matches_reference(op: str, device: torch.device, rng: torch.Generator) -> None:
    x = make_image(rng, (1, 2, 16, 20), dtype=torch.float32, device=device)
    kernel = flat_se(x.shape[1], SEPARABLE_K, dtype=torch.float32, device=device)

    out = getattr(serron.functional, op)(x, kernel)

    torch.testing.assert_close(out, reference(op, x, kernel))


def _reference_grads(
    op: str, border: BorderMode, wrt: str, device: torch.device, rng: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient from the compiled op, and from autograd through the reference."""
    x = make_image(rng, (1, 2, 12, 14), device=device)
    kernel = flat_se(x.shape[1], SEPARABLE_K, device=device)
    go = make_image(rng, (1, 2, 12, 14), device=device)

    def grads(fn: object) -> torch.Tensor:
        xt = x.detach().requires_grad_(True)
        kt = kernel.detach().requires_grad_(True)
        fn(xt, kt).backward(go)  # type: ignore[operator]
        grad = xt.grad if wrt == "input" else kt.grad
        assert grad is not None
        return grad

    return (
        grads(lambda t, k: getattr(serron.functional, op)(t, k, border=border)),
        grads(lambda t, k: reference(op, t, k, border)),
    )


@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize("op", PRIMITIVES)
def test_backward_matches_reference_input_grad(op: str, border: BorderMode, device: torch.device, rng: torch.Generator) -> None:
    got, expected = _reference_grads(op, border, "input", device, rng)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("op", PRIMITIVES)
def test_backward_matches_reference_se_grad(op: str, device: torch.device, rng: torch.Generator) -> None:
    """CONSTANT only: REFLECT and REPLICATE feed the same input pixel to several taps of
    the window, and with a flat SE those taps tie, so which SE cell takes the gradient is
    genuinely ambiguous. The input pixel it lands on is not, and that is what
    :func:`test_backward_matches_reference_input_grad` covers."""
    got, expected = _reference_grads(op, BorderMode.CONSTANT, "kernel", device, rng)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("op", PRIMITIVES)
def test_grad_input_is_a_permutation_of_grad_output(op: str, device: torch.device, rng: torch.Generator) -> None:
    x = make_image(rng, (1, 2, 12, 14), device=device).detach().requires_grad_(True)
    kernel = flat_se(2, SEPARABLE_K, device=device)
    go = make_image(rng, (1, 2, 12, 14), device=device)

    getattr(serron.functional, op)(x, kernel).backward(go)

    assert x.grad is not None
    torch.testing.assert_close(x.grad.sum(), go.sum())


def test_matches_direct_path(device: torch.device, tmp_path: Path) -> None:
    """The separable path has to agree with the direct 2-D path exactly: same values and
    the same winning taps. The direct run happens in its own process, since the path
    selector reads its env var once at startup."""
    direct_path = tmp_path / "direct.pt"
    subprocess.run(
        [sys.executable, "-m", "tests._separable_worker", "--device", str(device), "--out", str(direct_path)],
        env={**os.environ, "SERRON_SEPARABLE_MIN_K": _FORCE_DIRECT},
        capture_output=True,
        text=True,
        check=True,
    )

    direct = torch.load(direct_path, weights_only=True)
    separable = run_cases(device)

    assert separable.keys() == direct.keys()
    for key, got in separable.items():
        # The CUDA scatter accumulates with atomicAdd, so a gradient sum depends on the
        # order the blocks land in; a different winning tap would still be far outside
        # the default tolerance.
        def annotate(message: str, k: str = key) -> str:
            return f"{k}: {message}"

        if device.type == "cpu" or key.endswith(".forward"):
            torch.testing.assert_close(got, direct[key], rtol=0.0, atol=0.0, msg=annotate)
        else:
            torch.testing.assert_close(got, direct[key], msg=annotate)
