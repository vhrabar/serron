"""
The ROCm library loads into any torch and registers the GPU ops, with no ROCm runtime and no GPU involved.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from serron import _cmake_ops

_BASE = _cmake_ops._find_library("_C")
_ROCM = _cmake_ops._find_library("_C_rocm")

pytestmark = pytest.mark.skipif(_BASE is None or _ROCM is None, reason="this build has no _C_rocm")

_OPS = (
    "erode",
    "dilate",
    "erode_backward",
    "dilate_backward",
    "feasible_algos",
    "heuristic_algo",
    "set_separable_min_k",
    "get_separable_min_k",
)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter, which has loaded none of serron's libraries."""
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=300)


def test_the_rocm_library_registers_every_gpu_op() -> None:
    result = _run(
        f"""
        import torch

        torch.ops.load_library({_BASE!r})
        ops = {_OPS!r}

        def gpu_kernels() -> list[str]:
            return [op for op in ops if any(line.startswith("CUDA:") for line in torch._C._dispatch_dump("serron::" + op).splitlines())]

        assert gpu_kernels() == [], "the base library must not register a GPU kernel"
        torch.ops.load_library({_ROCM!r})
        missing = sorted(set(ops) - set(gpu_kernels()))
        assert not missing, f"no CUDA-key kernel for {{missing}}"
        """
    )

    assert result.returncode == 0, result.stderr


def test_loading_the_rocm_library_does_not_map_a_hip_runtime() -> None:
    """Nothing links the HIP runtime, so loading adds none: it is looked up when a GPU op first runs.

    A ROCm torch maps its own runtime at import, which is why this compares before and after.
    """
    result = _run(
        f"""
        import torch

        def hip_mappings() -> set[str]:
            return {{line.split()[-1] for line in open("/proc/self/maps") if "libamdhip64" in line or "libhiprtc" in line}}

        torch.ops.load_library({_BASE!r})
        before = hip_mappings()
        torch.ops.load_library({_ROCM!r})
        added = hip_mappings() - before
        assert not added, sorted(added)
        """
    )

    assert result.returncode == 0, result.stderr
