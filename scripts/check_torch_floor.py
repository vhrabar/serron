"""
Check that every place stating the minimum supported torch agrees on one major.minor.
"""

from __future__ import annotations

import pathlib
import re
import sys

import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG_PYPROJECT = ROOT / "packages/serron/pyproject.toml"
ROOT_PYPROJECT = ROOT / "pyproject.toml"
CMAKELISTS = ROOT / "packages/serron/CMakeLists.txt"

FLOOR_GROUPS = ("cpu", "cu132")


def torch_spec(deps: list[str]) -> str:
    return next(d for d in deps if re.match(r"torch\b", d))


def match_version(pattern: str, text: str) -> str | None:
    found = re.search(pattern, text)
    return found.group(1) if found else None


def main() -> int:
    pkg = tomllib.loads(PKG_PYPROJECT.read_text())
    root = tomllib.loads(ROOT_PYPROJECT.read_text())

    floors: dict[str, str | None] = {
        "packages/serron/pyproject.toml [project].dependencies": match_version(
            r">=\s*(\d+\.\d+)", torch_spec(pkg["project"]["dependencies"])
        ),
        "packages/serron/CMakeLists.txt SERRON_TORCH_TARGET": match_version(
            r'set\(SERRON_TORCH_TARGET\s+"(\d+\.\d+)"', CMAKELISTS.read_text()
        ),
        "pyproject.toml [project].dependencies": match_version(r">=\s*(\d+\.\d+)", torch_spec(root["project"]["dependencies"])),
    }
    for group in FLOOR_GROUPS:
        floors[f"pyproject.toml [dependency-groups].{group}"] = match_version(
            r"==\s*(\d+\.\d+)\.\*", torch_spec(root["dependency-groups"][group])
        )

    for where, version in floors.items():
        print(f"{version or 'not found':>10}  {where}")

    if None in floors.values() or len(set(floors.values())) != 1:
        print("torch floor is not the same everywhere; raise it in all of the places above together", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
