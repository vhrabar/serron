"""
Separable thresholds shipped with the library, for GPUs someone has already calibrated.

``_seeds.json`` maps a compute capability to the thresholds :func:`serron.calibrate` found on a GPU with it. They
are the starting point on that architecture, so a fresh install picks a sensible algorithm before anyone calibrates.
A calibration saved on the machine beats a seed, and ``SERRON_SEPARABLE_MIN_K`` beats both.

The file is regenerated on real hardware with ``python -m benchmarks.make_seed_table``.
"""

from __future__ import annotations

import functools
import json
import re
import warnings
from importlib import resources
from typing import Any

__all__ = [
    "SEED_FORMAT_VERSION",
    "build_entry",
    "dumps",
    "load_seeds",
    "merge_entry",
    "parse_seeds",
    "seed_thresholds",
]

SEED_FORMAT_VERSION = 1
_FILE = "_seeds.json"

_CAPABILITY = re.compile(r"^\d+\.\d+$")
_KEY = re.compile(r"^[a-z0-9]+:(fwd|bwd)$")


def parse_seeds(raw: object) -> dict[str, dict[str, int]]:
    """
    Check the contents of a seed file and pull out its thresholds.

    :param raw: the decoded JSON.
    :returns: from compute capability (``"12.0"``) to that architecture's thresholds, keyed as
    :func:`serron._autotune.threshold_key` writes them.
    :raises ValueError: if anything is malformed. The message says what and where.
    """
    if not isinstance(raw, dict):
        raise ValueError("the seed file must hold a JSON object")
    if raw.get("version") != SEED_FORMAT_VERSION:
        raise ValueError(f"unsupported seed file version {raw.get('version')!r}, expected {SEED_FORMAT_VERSION}")
    devices = raw.get("devices")
    if not isinstance(devices, dict):
        raise ValueError("'devices' must be an object")

    seeds: dict[str, dict[str, int]] = {}
    for capability, entry in devices.items():
        if not _CAPABILITY.match(capability):
            raise ValueError(f"{capability!r} is not a compute capability such as '12.0'")
        if not isinstance(entry, dict) or not isinstance(entry.get("thresholds"), dict) or not entry["thresholds"]:
            raise ValueError(f"devices.{capability}.thresholds must be a non-empty object")
        for key, value in entry["thresholds"].items():
            if not _KEY.match(key):
                raise ValueError(f"devices.{capability}: {key!r} is not a threshold key such as 'float32:fwd'")
            if type(value) is not int or value < 1:
                raise ValueError(f"devices.{capability}.{key} must be a positive integer, got {value!r}")
        seeds[capability] = dict(entry["thresholds"])
    return seeds


@functools.cache
def load_seeds() -> dict[str, dict[str, int]]:
    """
    The shipped seeds.

    A file that cannot be read or fails validation warns once and yields no seeds, so a damaged install loses its
    head start rather than its ops.

    :returns: from compute capability to thresholds. Treat it as read-only.
    """
    try:
        return parse_seeds(json.loads(resources.files(__package__).joinpath(_FILE).read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        warnings.warn(f"serron: ignoring the shipped calibration seeds: {exc}", stacklevel=2)
        return {}


def seed_thresholds(capability: str) -> dict[str, int] | None:
    """
    Seed thresholds for a compute capability.

    :param capability: for example ``"12.0"``.
    :returns: a copy of the thresholds, or ``None`` if nothing has been calibrated on that architecture.
    """
    found = load_seeds().get(capability)
    return None if found is None else dict(found)


def build_entry(measured_on: str, thresholds: dict[str, int], measured_with: dict[str, Any]) -> dict[str, Any]:
    """
    One architecture's entry for the seed file.

    :param measured_on: name of the GPU the numbers came from, which tells a reader what they describe.
    :param thresholds: from threshold key to window length.
    :param measured_with: how it was measured (library and torch versions, sweep settings, date).
    :returns: the entry to hand to :func:`merge_entry`.
    """
    return {"measured_on": measured_on, "measured_with": measured_with, "thresholds": dict(sorted(thresholds.items()))}


def merge_entry(data: dict[str, Any], capability: str, entry: dict[str, Any]) -> dict[str, Any]:
    """
    Put ``entry`` under ``capability``, replacing what was there and keeping every other architecture.

    :param data: the decoded seed file, or an empty dict for a new one.
    :param capability: compute capability the entry is for.
    :param entry: from :func:`build_entry`.
    :returns: a new seed file's contents. ``data`` is not modified.
    :raises ValueError: if the result would not pass :func:`parse_seeds`.
    """
    merged = {
        "version": SEED_FORMAT_VERSION,
        "devices": {**data.get("devices", {}), capability: entry},
    }
    parse_seeds(merged)
    return merged


def dumps(data: dict[str, Any]) -> str:
    """The seed file as text: sorted keys, two-space indent, a final newline, so a regeneration diffs cleanly."""
    return json.dumps(data, indent=2, sort_keys=True) + "\n"
