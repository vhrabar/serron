"""
Timing of CUDA work, shared by the autotuner and the benchmarks.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

import torch


def time_ms(fn: Callable[[], object], reps: int = 30, warmup: int = 1) -> float:
    """
    Median GPU time of ``fn`` in milliseconds.

    Each repetition is bracketed by CUDA events on the current stream and waited on, so a repetition is timed on its
    own and the median is not skewed by queued work from the one before.

    :param fn: callable that enqueues the work to time on the current CUDA stream.
    :param reps: timed repetitions.
    :param warmup: untimed calls made first, which pay for kernel loading and shared-memory setup.
    :returns: the median over ``reps`` repetitions.
    :raises ValueError: if ``reps`` is not positive.
    """
    if reps < 1:
        raise ValueError(f"reps must be positive, got {reps}")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)
