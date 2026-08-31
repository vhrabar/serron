"""
Shared-memory vs global-memory kernel comparison for the morphology ops.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# (batch, channels, height, width), kernel size
CASES = [
    (1, 1, 512, 512, 3),
    (1, 1, 512, 512, 15),
    (8, 3, 512, 512, 7),
    (16, 3, 1024, 1024, 7),
    (8, 32, 256, 256, 5),
]
_ITERS = 50
_WARMUP = 5


def _run_worker() -> None:
    """Child process: time every case and print the results as JSON on stdout."""
    import time

    import torch

    import serron

    results = {}
    for n, c, h, w, k in CASES:
        x = torch.randn(n, c, h, w, device="cuda", dtype=torch.float32)
        se = torch.zeros(c, k, k, device="cuda", dtype=torch.float32)
        for _ in range(_WARMUP):
            serron.erosion(x, se)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(_ITERS):
            serron.erosion(x, se)
        torch.cuda.synchronize()
        results[f"{n}x{c}x{h}x{w}_k{k}"] = (time.perf_counter() - start) / _ITERS
    print(json.dumps(results))


def _time_variant(force_global: bool) -> dict[str, float]:
    """Run the worker in a fresh subprocess and hand back its timings.

    :param force_global: if ``True``, pin the global-memory kernel via
        ``SERRON_FORCE_GLOBAL``; otherwise let the kernel choose.
    :returns: mean seconds per iteration, keyed by case name.
    """
    env = dict(os.environ, SERRON_FORCE_GLOBAL="1" if force_global else "0")
    out = subprocess.run(
        [sys.executable, "-m", "benchmarks.bench_shmem", "--worker"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    result: dict[str, float] = json.loads(out.stdout.strip().splitlines()[-1])
    return result


def main() -> None:
    """Time both kernel variants and print a per-case speedup table."""
    import torch

    if not torch.cuda.is_available():
        print("benchmarks require a CUDA device")
        return

    shared = _time_variant(force_global=False)
    glob = _time_variant(force_global=True)

    dev = torch.cuda.get_device_name(0)
    print(f"\nerosion forward, float32, {_ITERS} iters on {dev}\n")
    header = f"{'case':<22}{'global (ms)':>14}{'shared (ms)':>14}{'speedup':>10}"
    print(header)
    print("-" * len(header))
    for case in shared:
        g = glob[case] * 1e3
        s = shared[case] * 1e3
        print(f"{case:<22}{g:>14.3f}{s:>14.3f}{g / s:>9.2f}x")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _run_worker()
    else:
        main()
