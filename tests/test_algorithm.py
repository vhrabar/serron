"""
Forcing the algorithm a compiled op runs through its ``algo`` argument.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

import serron
from serron._algorithms import (
    device_fingerprint,
    feasible_algorithms,
    get_separable_min_k,
    heuristic_algorithm,
    set_separable_min_k,
)
from serron._timing import time_ms
from serron.autograd import _DilateFunction, _ErodeFunction, ops
from serron.enums import Algorithm
from tests.conftest import flat_se, make_image, requires_cuda

_OPS = ["erode", "dilate"]
_DIRECT = [Algorithm.ELEMENT, Algorithm.TILED]
_BORDER = 0
_SEPARABLE_K = 21


def _forward(op: str, x: torch.Tensor, kernel: torch.Tensor, algo: Algorithm) -> torch.Tensor:
    return getattr(ops, op)(x, kernel, _BORDER, True, int(algo))


@requires_cuda
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("k", [3, _SEPARABLE_K])
def test_forced_algorithms_are_bitwise_equal_forward(op: str, k: int, rng: torch.Generator) -> None:
    """Min and max are exact, so every algorithm that can run the problem returns the same bits."""
    x = make_image(rng, (2, 3, 40, 37), device=torch.device("cuda"))
    kernel = flat_se(x.shape[1], k, device=x.device, shared=False)

    expected = _forward(op, x, kernel, Algorithm.AUTO)
    for algo in (*_DIRECT, Algorithm.SEPARABLE):
        assert torch.equal(_forward(op, x, kernel, algo), expected), algo.name


@requires_cuda
@pytest.mark.parametrize("op", _OPS)
def test_forced_algorithms_agree_on_gradients(op: str, rng: torch.Generator) -> None:
    function = _ErodeFunction if op == "erode" else _DilateFunction
    x = make_image(rng, (2, 3, 40, 37), device=torch.device("cuda"))
    kernel = flat_se(x.shape[1], _SEPARABLE_K, device=x.device, shared=False)

    def grads(algo: Algorithm) -> tuple[torch.Tensor, torch.Tensor]:
        xi = x.clone().requires_grad_()
        ki = kernel.clone().requires_grad_()
        function.apply(xi, ki, _BORDER, int(algo)).sum().backward()
        assert xi.grad is not None
        assert ki.grad is not None
        return xi.grad, ki.grad

    expected_x, expected_k = grads(Algorithm.AUTO)
    for algo in (*_DIRECT, Algorithm.SEPARABLE):
        grad_x, grad_k = grads(algo)
        # The argmax tie-break is shared by every algorithm, so these are exact, not merely close.
        assert torch.equal(grad_x, expected_x), algo.name
        assert torch.equal(grad_k, expected_k), algo.name


@requires_cuda
def test_forced_separable_rejects_non_flat_se(rng: torch.Generator) -> None:
    x = make_image(rng, (1, 1, 16, 16), device=torch.device("cuda"))
    kernel = torch.randn(5, 5, device=x.device, dtype=x.dtype)

    with pytest.raises(RuntimeError, match="flat"):
        ops.erode(x, kernel, _BORDER, False, int(Algorithm.SEPARABLE))


@requires_cuda
def test_forced_tiled_rejects_window_over_shared_memory() -> None:
    x = torch.zeros(1, 1, 8, 8, device="cuda")
    kernel = torch.zeros(1001, 1001, device="cuda")

    with pytest.raises(RuntimeError, match="tiled"):
        ops.erode(x, kernel, _BORDER, True, int(Algorithm.TILED))


def test_invalid_algo_is_rejected(device: torch.device) -> None:
    x = torch.zeros(1, 1, 8, 8, device=device)
    kernel = torch.zeros(3, 3, device=device)

    with pytest.raises(RuntimeError, match="invalid algo"):
        ops.erode(x, kernel, _BORDER, True, 99)


def test_cpu_only_takes_auto() -> None:
    x = torch.zeros(1, 1, 8, 8)
    kernel = torch.zeros(3, 3)

    assert torch.equal(ops.erode(x, kernel, _BORDER, True, int(Algorithm.AUTO)), serron.functional.erosion(x, kernel))
    with pytest.raises(RuntimeError, match="algo must be auto"):
        ops.erode(x, kernel, _BORDER, True, int(Algorithm.TILED))


# --------------------------------------------------------------------------- #
# Selection queries
# --------------------------------------------------------------------------- #


@requires_cuda
def test_feasible_algorithms_for_flat_se() -> None:
    x = torch.zeros(1, 3, 32, 32, device="cuda")
    kernel = torch.zeros(3, 5, 5, device="cuda")

    assert feasible_algorithms(x, kernel) == (Algorithm.ELEMENT, Algorithm.TILED, Algorithm.SEPARABLE)
    assert feasible_algorithms(x, kernel, backward=True) == (Algorithm.ELEMENT, Algorithm.TILED, Algorithm.SEPARABLE)


@requires_cuda
def test_feasible_algorithms_drop_separable_for_non_flat_se() -> None:
    x = torch.zeros(1, 3, 32, 32, device="cuda")
    kernel = torch.ones(3, 5, 5, device="cuda")

    assert feasible_algorithms(x, kernel) == (Algorithm.ELEMENT, Algorithm.TILED)


@requires_cuda
def test_feasible_algorithms_drop_tiled_for_window_over_shared_memory() -> None:
    x = torch.zeros(1, 1, 8, 8, device="cuda")
    kernel = torch.zeros(1, 1001, 1001, device="cuda")

    assert Algorithm.TILED not in feasible_algorithms(x, kernel)
    assert Algorithm.ELEMENT in feasible_algorithms(x, kernel)


@requires_cuda
@pytest.mark.parametrize("backward", [False, True])
def test_every_feasible_algorithm_runs(backward: bool, rng: torch.Generator) -> None:
    x = make_image(rng, (1, 2, 48, 48), device=torch.device("cuda"))
    kernel = flat_se(x.shape[1], 9, device=x.device, shared=False)

    for algo in feasible_algorithms(x, kernel, backward=backward):
        ops.erode(x, kernel, _BORDER, True, int(algo))


@requires_cuda
def test_heuristic_follows_window_length_and_flatness() -> None:
    x = torch.zeros(1, 3, 64, 64, device="cuda")

    assert heuristic_algorithm(x, torch.zeros(3, 3, 3, device="cuda")) == Algorithm.TILED
    assert heuristic_algorithm(x, torch.zeros(3, _SEPARABLE_K, _SEPARABLE_K, device="cuda")) == Algorithm.SEPARABLE
    assert heuristic_algorithm(x, torch.ones(3, _SEPARABLE_K, _SEPARABLE_K, device="cuda")) == Algorithm.TILED


@requires_cuda
def test_separable_threshold_is_per_dtype_and_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("SERRON_SEPARABLE_MIN_K"):
        pytest.skip("SERRON_SEPARABLE_MIN_K overrides every threshold")
    x = torch.zeros(1, 3, 64, 64, device="cuda")
    half = x.half()
    kernel = torch.zeros(3, 5, 5, device="cuda")
    default = get_separable_min_k(x)

    try:
        assert set_separable_min_k(x, 5) == 5
        assert heuristic_algorithm(x, kernel) == Algorithm.SEPARABLE
        assert heuristic_algorithm(x, kernel, backward=True) == Algorithm.TILED, "the backward has its own threshold"
        assert get_separable_min_k(half) == default, "other dtypes are untouched"

        set_separable_min_k(x, 0)
        assert get_separable_min_k(x) == default
        assert heuristic_algorithm(x, kernel) == Algorithm.TILED
    finally:
        set_separable_min_k(x, 0)


@requires_cuda
def test_environment_override_beats_the_table() -> None:
    code = (
        "import torch\n"
        "from serron._algorithms import get_separable_min_k, set_separable_min_k\n"
        "x = torch.zeros(1, 1, 8, 8, device='cuda')\n"
        "set_separable_min_k(x, 5)\n"
        "print(get_separable_min_k(x))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "SERRON_SEPARABLE_MIN_K": "7"},
    )

    assert out.stdout.strip().splitlines()[-1] == "7"


@requires_cuda
def test_device_fingerprint_names_what_a_result_depends_on() -> None:
    fingerprint = device_fingerprint()

    assert set(fingerprint) == {"name", "capability", "sm_count", "max_optin_smem", "cuda", "serron"}
    assert fingerprint["sm_count"] > 0
    assert fingerprint["max_optin_smem"] >= 48 * 1024


@requires_cuda
def test_time_ms_is_positive_and_rejects_no_reps() -> None:
    x = torch.randn(256, 256, device="cuda")

    assert time_ms(lambda: x @ x, reps=3) > 0
    with pytest.raises(ValueError, match="reps"):
        time_ms(lambda: x @ x, reps=0)
