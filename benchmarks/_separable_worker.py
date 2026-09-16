"""
Single-configuration timing worker for ``sweep_separable_threshold.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

import serron
from serron import BorderMode

_BORDERS = {"reflect": BorderMode.REFLECT, "replicate": BorderMode.REPLICATE, "constant": BorderMode.CONSTANT}


def _time_ms(fn, reps: int) -> float:
    """Median wall time in ms over ``reps`` CUDA-timed repetitions, after a warmup call."""
    fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True, help="N,C,H,W")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--op", choices=["erosion", "dilation"], required=True)
    parser.add_argument("--border", choices=list(_BORDERS), default="replicate")
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()

    n, c, h, w = (int(x) for x in args.shape.split(","))
    device = torch.device("cuda")
    func = getattr(serron.functional, args.op)
    border = _BORDERS[args.border]

    x = torch.randn(n, c, h, w, device=device)
    kernel = torch.zeros(c, args.k, args.k, device=device)  # flat -> separable-eligible

    def fwd() -> torch.Tensor:
        return func(x, kernel, border=border)

    fwd_ms = _time_ms(fwd, args.reps)

    x_g = x.clone().requires_grad_(True)
    kernel_g = kernel.clone().requires_grad_(True)

    def fwd_bwd() -> None:
        out = func(x_g, kernel_g, border=border)
        out.sum().backward()
        x_g.grad = None
        kernel_g.grad = None

    fwd_bwd_ms = _time_ms(fwd_bwd, args.reps)

    print(json.dumps({"fwd_ms": fwd_ms, "fwd_bwd_ms": fwd_bwd_ms}))


if __name__ == "__main__":
    main()
