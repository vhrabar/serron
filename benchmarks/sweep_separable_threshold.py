"""
Crossover sweep behind the ``SERRON_SEPARABLE_MIN_K`` defaults.
"""

from __future__ import annotations

import argparse

import torch

from benchmarks._table import markdown_table
from serron import BorderMode
from serron._algorithms import feasible_algorithms
from serron._timing import time_ms
from serron.autograd import _DilateFunction, _ErodeFunction
from serron.enums import BORDER_TO_INT, Algorithm

_COLUMNS = ("k", "separable fwd (ms)", "direct fwd (ms)", "fwd", "separable f+b (ms)", "direct f+b (ms)", "f+b")
_ALIGNS = "rrrrrrr"

_BORDERS = {"reflect": BorderMode.REFLECT, "replicate": BorderMode.REPLICATE, "constant": BorderMode.CONSTANT}
_FUNCTIONS = {"erosion": _ErodeFunction, "dilation": _DilateFunction}


def _measure(x: torch.Tensor, kernel: torch.Tensor, op: str, border: int, algo: Algorithm, reps: int) -> dict[str, float]:
    """Time the forward and the forward+backward of ``op`` forced to ``algo``."""
    function = _FUNCTIONS[op]

    def fwd() -> torch.Tensor:
        out: torch.Tensor = function.apply(x, kernel, border, int(algo))
        return out

    x_g = x.clone().requires_grad_(True)
    kernel_g = kernel.clone().requires_grad_(True)

    def fwd_bwd() -> None:
        function.apply(x_g, kernel_g, border, int(algo)).sum().backward()
        x_g.grad = None
        kernel_g.grad = None

    return {"fwd_ms": time_ms(fwd, reps), "fwd_bwd_ms": time_ms(fwd_bwd, reps)}


def _first_lasting_win(ks: list[int], wins: list[bool]) -> int | None:
    """Smallest k whose win holds for every larger k too."""
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

    ks = [int(v) for v in args.ks.split(",")]
    rows: list[list[str]] = []
    measured: list[int] = []
    fwd_wins: list[bool] = []
    both_wins: list[bool] = []

    n, c, h, w = (int(v) for v in args.shape.split(","))
    x = torch.randn(n, c, h, w, device="cuda")
    border = BORDER_TO_INT[_BORDERS[args.border]]

    for k in ks:
        kernel = torch.zeros(c, k, k, device="cuda")  # flat -> separable-eligible
        # A window too long for either pass to stage cannot be compared.
        candidates = feasible_algorithms(x, kernel, flat=True, backward=True)
        if Algorithm.SEPARABLE not in candidates:
            continue
        direct_algo = Algorithm.TILED if Algorithm.TILED in candidates else Algorithm.ELEMENT

        sep = _measure(x, kernel, args.op, border, Algorithm.SEPARABLE, args.reps)
        direct = _measure(x, kernel, args.op, border, direct_algo, args.reps)

        measured.append(k)
        fwd_wins.append(sep["fwd_ms"] < direct["fwd_ms"])
        both_wins.append(sep["fwd_bwd_ms"] < direct["fwd_bwd_ms"])

        rows.append(
            [
                str(k),
                f"{sep['fwd_ms']:.3f}",
                f"{direct['fwd_ms']:.3f}",
                f"{direct['fwd_ms'] / sep['fwd_ms']:.2f}x",
                f"{sep['fwd_bwd_ms']:.3f}",
                f"{direct['fwd_bwd_ms']:.3f}",
                f"{direct['fwd_bwd_ms'] / sep['fwd_bwd_ms']:.2f}x",
            ]
        )

    print(f"\n{args.op}, {args.shape}, {args.border} border, on {torch.cuda.get_device_name(0)}.\n")
    print(markdown_table(_COLUMNS, rows, _ALIGNS))
    crossover_fwd = _first_lasting_win(measured, fwd_wins)
    crossover_both = _first_lasting_win(measured, both_wins)
    print(f"\nSeparable path wins from k={crossover_fwd} (forward), k={crossover_both} (forward+backward) upwards.")
    print("The forward+backward figure is the one for SERRON_SEPARABLE_MIN_K.")


if __name__ == "__main__":
    main()
