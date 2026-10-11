"""
The CUDA ops reach the GPU through the driver API, which works on the calling thread's current context. These run
the ops on a thread whose current context has been cleared, which torch does not guard against.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from collections.abc import Callable

import pytest
import torch

import serron
from tests.conftest import requires_cuda

pytestmark = requires_cuda


def _driver() -> ctypes.CDLL:
    name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    try:
        return ctypes.CDLL(name)
    except OSError as error:
        pytest.skip(f"cannot open the CUDA driver: {error}")


def _current_context(cu: ctypes.CDLL) -> int | None:
    ctx = ctypes.c_void_p()
    assert cu.cuCtxGetCurrent(ctypes.byref(ctx)) == 0
    return ctx.value


def _run_without_context(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, int | None]:
    """Run ``fn`` on a new thread after clearing that thread's current context; returns its result and the
    context current on the thread afterwards."""
    cu = _driver()
    result: dict[str, object] = {}

    def body() -> None:
        try:
            assert cu.cuCtxSetCurrent(None) == 0
            assert _current_context(cu) is None
            out = fn()
            torch.cuda.synchronize()
            result["out"] = out
            result["after"] = _current_context(cu)
        except BaseException as error:  # re-raised on the test thread
            result["error"] = error

    thread = threading.Thread(target=body)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result["out"], result["after"]  # type: ignore[return-value]


@pytest.mark.parametrize("ksize", [3, 21], ids=["direct", "separable"])
@pytest.mark.parametrize("op", ["erosion", "dilation"])
def test_forward_without_current_context(op: str, ksize: int) -> None:
    x = torch.randn(2, 3, 33, 40, device="cuda")
    kernel = torch.zeros(ksize, ksize, device="cuda")
    expected = getattr(serron.functional, op)(x, kernel)

    out, _ = _run_without_context(lambda: getattr(serron.functional, op)(x, kernel))

    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("ksize", [3, 21], ids=["direct", "separable"])
def test_backward_without_current_context(ksize: int) -> None:
    x = torch.randn(1, 2, 24, 30, device="cuda", dtype=torch.float64)
    kernel = torch.zeros(ksize, ksize, device="cuda", dtype=torch.float64)

    def grad() -> torch.Tensor:
        xg = x.clone().requires_grad_(True)
        serron.functional.erosion(xg, kernel).sum().backward()
        assert xg.grad is not None
        return xg.grad

    expected = grad()
    out, _ = _run_without_context(grad)

    torch.testing.assert_close(out, expected)
