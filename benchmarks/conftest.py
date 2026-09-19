"""
Fixtures for the benchmark suite.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from benchmarks._table import markdown_table

CUDA_AVAILABLE = torch.cuda.is_available()

_COLUMNS = ("test", "params", "min (ms)", "mean (ms)", "median (ms)", "stddev (ms)", "rounds", "ops/s")
_ALIGNS = "llrrrrrr"

collect_ignore = ["bench_shmem.py", "bench_vs_libs.py"]

requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="benchmarks require a CUDA device")


def cuda_sync() -> None:
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()


def _benchmark_row(bench: Any) -> list[str]:
    """Format one ``pytest-benchmark`` result as table cells.

    :param bench: a benchmark's metadata, as collected by ``pytest-benchmark``.
    :returns: one cell per column, in :data:`_COLUMNS` order.
    """
    name, _, params = bench.name.partition("[")
    stats = bench.stats
    return [
        name,
        params.rstrip("]"),
        f"{stats.min * 1e3:.3f}",
        f"{stats.mean * 1e3:.3f}",
        f"{stats.median * 1e3:.3f}",
        f"{stats.stddev * 1e3:.3f}",
        str(stats.rounds),
        f"{stats.ops:,.1f}",
    ]


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter: Any) -> None:
    """Print the benchmark results as a markdown table, below pytest-benchmark's own table.

    :param terminalreporter: pytest's terminal reporter, used for the device name and output.
    """
    session = getattr(terminalreporter.config, "_benchmarksession", None)
    if session is None:
        return
    rows = [_benchmark_row(b) for b in session.benchmarks if not b.has_error]
    if not rows:
        return
    device = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"
    terminalreporter.write_line("")
    terminalreporter.write_line(f"Morphology kernels on {device}.")
    terminalreporter.write_line("")
    for line in markdown_table(_COLUMNS, rows, _ALIGNS).splitlines():
        terminalreporter.write_line(line)
