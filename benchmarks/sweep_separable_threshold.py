"""
Crossover sweep behind the ``SERRON_SEPARABLE_MIN_K`` defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from benchmarks._table import markdown_table

_COLUMNS = ("k", "separable fwd (ms)", "direct fwd (ms)", "fwd", "separable f+b (ms)", "direct f+b (ms)", "f+b")
_ALIGNS = "rrrrrrr"

# direct ovveride
_NEVER = 1 << 20


def _measure(shape: str, k: int, op: str, border: str, reps: int, min_k: int) -> dict[str, float]:
    """Time one configuration in a fresh process with ``SERRON_SEPARABLE_MIN_K`` set to ``min_k``."""
    out = subprocess.run(
        [sys.executable, "-m", "benchmarks._separable_worker", "--shape", shape, "--k", str(k),
         "--op", op, "--border", border, "--reps", str(reps)],
        capture_output=True, text=True, check=True,
        env={**os.environ, "SERRON_SEPARABLE_MIN_K": str(min_k)},
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _first_lasting_win(ks: list[int], wins: list[bool]) -> int | None:
    """Smallest k whose win holds for every larger k too.
    """
    lasting = None
    for k, won in zip(reversed(ks), reversed(wins), strict=True):
        if not won:
            break
        lasting = k
    return lasting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="1,3,512,512", help="N,C,H,W")
    parser.add_argument("--ks", default="3,5,7,9,11,13,15,21,31,63", help="comma-separated window lengths")
    parser.add_argument("--op", choices=["erosion", "dilation"], default="erosion")
    parser.add_argument("--border", choices=["reflect", "replicate", "constant"], default="replicate")
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    rows, fwd_wins, both_wins = [], [], []

    for k in ks:
        sep = _measure(args.shape, k, args.op, args.border, args.reps, min_k=k)
        direct = _measure(args.shape, k, args.op, args.border, args.reps, min_k=_NEVER)

        fwd_wins.append(sep["fwd_ms"] < direct["fwd_ms"])
        both_wins.append(sep["fwd_bwd_ms"] < direct["fwd_bwd_ms"])

        rows.append([
            str(k),
            f"{sep['fwd_ms']:.3f}", f"{direct['fwd_ms']:.3f}", f"{direct['fwd_ms'] / sep['fwd_ms']:.2f}x",
            f"{sep['fwd_bwd_ms']:.3f}", f"{direct['fwd_bwd_ms']:.3f}",
            f"{direct['fwd_bwd_ms'] / sep['fwd_bwd_ms']:.2f}x",
        ])

    import torch

    print(f"\n{args.op}, {args.shape}, {args.border} border, on {torch.cuda.get_device_name(0)}.\n")
    print(markdown_table(_COLUMNS, rows, _ALIGNS))
    crossover_fwd = _first_lasting_win(ks, fwd_wins)
    crossover_both = _first_lasting_win(ks, both_wins)
    print(f"\nSeparable path wins from k={crossover_fwd} (forward), k={crossover_both} (forward+backward) upwards.")
    print("SERRON_SEPARABLE_MIN_K should be the forward+backward figure.")


if __name__ == "__main__":
    main()
