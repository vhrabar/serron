"""
Autograd wrapper and nn.Module around the compiled serron CUDA kernels.
"""

from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):
    from ._ops import add_op_namespace_prefix, ops  # type: ignore[import-not-found]

__all__ = ["add_op_namespace_prefix", "ops"]
