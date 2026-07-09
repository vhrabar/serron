"""
Op-namespace loader for the local CMake / PyPI build.
"""

from __future__ import annotations

import glob
import os

import torch

_NAMESPACE = "serron"


def _load_c_extension() -> None:
    pkg_dir = os.path.dirname(__file__)
    for pattern in ("_C*.so", "_C*.pyd", "_C*.dll"):
        matches = glob.glob(os.path.join(pkg_dir, pattern))
        if matches:
            torch.ops.load_library(matches[0])  # type: ignore[no-untyped-call]
            return
    raise ImportError(
        f"Could not find the compiled Serron '_C' extension in {pkg_dir}. Reinstall the package so the CUDA kernels are built."
    )


_load_c_extension()

ops = getattr(torch.ops, _NAMESPACE)


def add_op_namespace_prefix(op_name: str) -> str:
    return f"{_NAMESPACE}::{op_name}"
