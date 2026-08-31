"""
Fixtures for the benchmark suite.
"""

from __future__ import annotations

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

collect_ignore = ["bench_shmem.py", "bench_vs_libs.py"]

requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="benchmarks require a CUDA device")


def cuda_sync() -> None:
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
