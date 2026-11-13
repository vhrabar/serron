"""
On-disk cache of tuning results, one file per GPU and library version.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path

from ._algorithms import DeviceFingerprint

__all__ = ["cache_dir", "cache_path", "load_thresholds", "store_thresholds"]

_FORMAT_VERSION = 1


def cache_dir() -> Path:
    """
    Directory the cache lives in.

    :returns: ``$SERRON_CACHE_DIR`` when set and non-empty, otherwise ``~/.cache/serron``.
    """
    override = os.environ.get("SERRON_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / "serron"


def cache_path(fingerprint: DeviceFingerprint) -> Path:
    """
    File holding the results for ``fingerprint``.

    The name is a hash of every fingerprint field, so a different GPU, driver-side opt-in limit, CUDA version or
    serron version reads a different file instead of another one's results.

    :param fingerprint: what the results depend on, from :func:`serron._algorithms.device_fingerprint`.
    :returns: path of the file, which may not exist.
    """
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    return cache_dir() / f"calibration-{digest}.json"


def _parse(raw: str, fingerprint: DeviceFingerprint) -> dict[str, int] | None:
    """Thresholds in the file contents ``raw``, or ``None`` if they are malformed or for another fingerprint."""
    try:
        data = json.loads(raw)
        thresholds = data["thresholds"]
        valid = (
            data["version"] == _FORMAT_VERSION
            and data["fingerprint"] == dict(fingerprint)
            and isinstance(thresholds, dict)
            and all(isinstance(k, str) and type(v) is int and v >= 0 for k, v in thresholds.items())
        )
    except (ValueError, KeyError, TypeError):
        return None
    return dict(thresholds) if valid else None


def _read(path: Path, fingerprint: DeviceFingerprint) -> dict[str, int] | None:
    """Thresholds stored in ``path`` for ``fingerprint``, or ``None`` if it is missing, unreadable or foreign."""
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        warnings.warn(f"serron: ignoring unreadable tuning cache {path}: {exc}", stacklevel=3)
        return None

    thresholds = _parse(raw, fingerprint)
    if thresholds is None:
        warnings.warn(f"serron: ignoring corrupt or mismatched tuning cache {path}", stacklevel=3)
    return thresholds


def load_thresholds(fingerprint: DeviceFingerprint) -> dict[str, int] | None:
    """
    Separable thresholds saved for ``fingerprint``.

    A missing file is the normal cold start and returns ``None`` silently. A file that is unreadable, corrupt or
    written for another fingerprint also returns ``None``, with a warning, so tuning falls back to the heuristic
    rather than failing.

    :param fingerprint: what the results depend on.
    :returns: mapping from threshold key to window length, or ``None``.
    """
    return _read(cache_path(fingerprint), fingerprint)


def store_thresholds(fingerprint: DeviceFingerprint, thresholds: dict[str, int]) -> Path:
    """
    Merge ``thresholds`` into the file for ``fingerprint``.

    The file is replaced atomically, so a reader or a crash never sees a half-written file. Entries already on disk
    that ``thresholds`` does not mention are kept, so two processes tuning different dtypes do not discard each
    other's results. Two writers racing on the same entry leave one of the two values, both complete.

    :param fingerprint: what the results depend on.
    :param thresholds: mapping from threshold key to window length.
    :returns: path written.
    :raises OSError: if the directory cannot be created or written.
    """
    path = cache_path(fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)

    merged = {**(_read(path, fingerprint) or {}), **thresholds}
    payload = json.dumps(
        {"version": _FORMAT_VERSION, "fingerprint": dict(fingerprint), "thresholds": merged}, indent=2, sort_keys=True
    )

    # Same directory as the target: rename is only atomic within one filesystem.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return path
