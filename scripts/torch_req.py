"""
Print the pip requirement for one end of the supported torch range.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

import tomllib

PKG_DIR = pathlib.Path(os.environ.get("PKG_DIR", "packages/serron"))


def main() -> int:
    metadata = tomllib.loads((PKG_DIR / "pyproject.toml").read_text())
    spec = next(d for d in metadata["project"]["dependencies"] if d.startswith("torch"))

    if os.environ.get("TORCH_AXIS") == "floor":
        floor = re.search(r">=\s*(\d+\.\d+)", spec)
        if floor is None:
            print(f"no lower bound in torch requirement {spec!r}", file=sys.stderr)
            return 1
        spec = f"torch=={floor.group(1)}.*"

    print(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
