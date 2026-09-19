"""
Flatness of a structuring element, remembered per tensor.
"""

from __future__ import annotations

import torch
from torch.utils.weak import WeakTensorKeyDictionary

# caching mapping
_cache: WeakTensorKeyDictionary = WeakTensorKeyDictionary()


def is_flat(kernel: torch.Tensor) -> bool:
    """
    Whether ``kernel`` is flat and staus is unknown

    :param kernel: structuring-element tensor.
    :returns: ``True`` when every entry is ``0``.
    """
    cached: tuple[int, bool] | None = _cache.get(kernel)  # type: ignore[no-untyped-call]
    version: int = kernel._version
    if cached is not None and cached[0] == version:
        return cached[1]

    flat = bool(kernel.eq(0).all().item())
    _cache[kernel] = (version, flat)
    return flat


def mark(kernel: torch.Tensor, flat: bool) -> torch.Tensor:
    """
    Record AKAOT answer for ``kernel``, so the first use does not measure it.

    :param kernel: structuring-element tensor.
    :param flat: whether every entry is ``0``.
    :returns: ``kernel``, to allow use in a return statement.
    """
    _cache[kernel] = (kernel._version, flat)
    return kernel
