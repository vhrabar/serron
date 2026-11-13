"""
Measure this GPU and record its separable thresholds in the seed file the library ships.

Run it on real hardware, once per architecture, and commit the result::

    python -m benchmarks.make_seed_table            # measure and write the entry for this GPU's compute capability
    python -m benchmarks.make_seed_table --check    # measure and report whether the shipped entry still holds

The seed file is keyed by compute capability, so one GPU stands for its architecture. Entries for other
architectures are left alone. Nothing here reads or writes the per-user calibration cache.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import torch

from serron import __version__
from serron._algorithms import device_fingerprint
from serron._autotune import DEFAULT_DTYPES, DEFAULT_KS, sweep_separable_threshold, threshold_key
from serron._seeds import build_entry, dumps, merge_entry, parse_seeds

_SHAPE = (1, 3, 512, 512)
_FILE = "_seeds.json"


def _default_output() -> Path:
    return Path(str(resources.files("serron").joinpath(_FILE)))


def measure(
    device: int, dtypes: Sequence[torch.dtype], ks: Sequence[int], shape: tuple[int, int, int, int], reps: int
) -> dict[str, int]:
    """
    Sweep every dtype, forward and backward.

    :returns: from threshold key to window length.
    :raises SystemExit: if a result is the one-past-the-end value that means separable never won, which is a limit of
    the sweep and not a measurement to ship.
    """
    never = max(ks) + 1
    found: dict[str, int] = {}
    for dtype in dtypes:
        for backward in (False, True):
            key = threshold_key(dtype, backward=backward)
            found[key] = sweep_separable_threshold(dtype, backward=backward, device=device, ks=ks, shape=shape, reps=reps)
            print(f"  {key:16} {found[key]}", flush=True)
            if found[key] == never:
                raise SystemExit(f"{key}: separable never won up to k={max(ks)}; widen the ladder with --ks")
    return found


def _report_drift(shipped: dict[str, int], measured: dict[str, int]) -> bool:
    """Print how a fresh measurement differs from the shipped entry, and say whether it does."""
    differs = False
    for key in sorted(shipped.keys() | measured.keys()):
        old, new = shipped.get(key), measured.get(key)
        mark = "" if old == new else "   <-- differs"
        differs |= old != new
        print(f"  {key:16} shipped {old!s:>5}  measured {new!s:>5}{mark}")
    return differs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare with the shipped entry instead of writing")
    parser.add_argument("--output", type=Path, default=None, help="seed file to read and write (default: the packaged one)")
    parser.add_argument("--device", type=int, default=None, help="CUDA device index (default: the current one)")
    parser.add_argument("--reps", type=int, default=40, help="timed repetitions per measurement")
    parser.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)), help="comma-separated window lengths")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("make_seed_table needs a CUDA device")
    device = torch.cuda.current_device() if args.device is None else args.device
    ks = [int(k) for k in args.ks.split(",")]
    output: Path = args.output or _default_output()

    fingerprint = device_fingerprint(device)
    capability = fingerprint["capability"]
    print(f"{fingerprint['name']} (compute capability {capability}), serron {__version__}, reps={args.reps}")
    measured = measure(device, DEFAULT_DTYPES, ks, _SHAPE, args.reps)

    current = _load(output) if output.exists() else {}
    if args.check:
        shipped = parse_seeds(current).get(capability) if current else None
        if shipped is None:
            print(f"No shipped entry for compute capability {capability}.")
            return 1
        return 1 if _report_drift(shipped, measured) else 0

    entry = build_entry(
        fingerprint["name"],
        measured,
        {
            "serron": __version__,
            "torch": torch.__version__,
            "cuda": fingerprint["cuda"],
            "date": datetime.date.today().isoformat(),
            "shape": list(_SHAPE),
            "ks": ks,
            "reps": args.reps,
        },
    )
    merged = merge_entry(current, capability, entry)
    output.write_text(dumps(merged), encoding="utf-8")
    print(f"Wrote compute capability {capability} to {output}")
    return 0


def _load(path: Path) -> dict[str, Any]:
    """The decoded seed file at ``path``."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


if __name__ == "__main__":
    sys.exit(main())
