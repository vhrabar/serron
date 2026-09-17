"""
Structuring-element builders.
"""

from __future__ import annotations

import torch

from . import _flatness

_OFF = float("-inf")


def _checked_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Reject a dtype the morphology kernels cannot dispatch on.

    :param dtype: requested element type.
    :return: ``dtype`` unchanged.
    :raises ValueError: if ``dtype`` is not a floating type.
    """
    if not dtype.is_floating_point:
        raise ValueError(f"structuring elements must use a floating dtype, got {dtype}")
    return dtype


def _flat(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Turn a boolean membership ``mask`` into a flat additive SE (0 / -inf).
    :param mask: input mask
    :param dtype: element type of the result
    :return: flat additive SE
    """
    # Marked not-flat without inspecting the mask: a shape that happens to fill its whole
    # box is still correct on the 2-D path, only slower.
    return _flatness.mark(torch.where(mask, 0.0, _OFF).to(dtype), False)


def _centered_coords(size: int, device: torch.device | str | None) -> torch.Tensor:
    """
    1-D coordinate axis centered on 0
    :param size: size of the cord axis
    :param device: torch device
    :return: cord axis of size ``sieze`` centered on 0
    """
    return torch.arange(size, device=device) - (size - 1) // 2


def square(size: int, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Square  SE of shape ``(size, size)``.
    :param size: SE size
    :param dtype: element type; match the input's to keep an operation in that precision
    :param device: device to create the SE on
    :return: SE tensor
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    return _flatness.mark(torch.zeros(size, size, dtype=_checked_dtype(dtype), device=device), True)


def cross(size: int, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Cross-shaped SE of shape ``(size, size)``.
    :param size: SE size (should be odd for a symmetric cross)
    :param dtype: element type; match the input's to keep an operation in that precision
    :param device: device to create the SE on
    :return: SE tensor
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    coords = _centered_coords(size, device)
    dy, dx = torch.meshgrid(coords, coords, indexing="ij")
    return _flat((dy == 0) | (dx == 0), _checked_dtype(dtype))


def disk(radius: int, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Disk-shaped SE of shape ``(2*radius+1, 2*radius+1)``.
    :param radius: SE radius
    :param dtype: element type; match the input's to keep an operation in that precision
    :param device: device to create the SE on
    :return: SE tensor
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    coords = _centered_coords(2 * radius + 1, device)
    dy, dx = torch.meshgrid(coords, coords, indexing="ij")
    return _flat(dx * dx + dy * dy <= radius * radius, _checked_dtype(dtype))


def diamond(radius: int, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Diamond SE of shape ``(2*radius+1, 2*radius+1)``.
    :param radius: SE radius
    :param dtype: element type; match the input's to keep an operation in that precision
    :param device: device to create the SE on
    :return: SE tensor
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    coords = _centered_coords(2 * radius + 1, device)
    dy, dx = torch.meshgrid(coords, coords, indexing="ij")
    return _flat(dx.abs() + dy.abs() <= radius, _checked_dtype(dtype))


def from_tensor(weights: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Wrap an arbitrary 2-D tensor as a grayscale SE.
    :param weights: 2-D tensor of weights
    :param dtype: element type; match the input's to keep an operation in that precision
    :return: SE tensor
    """
    if weights.dim() != 2:
        raise ValueError(f"expected a 2-D (kh, kw) tensor, got shape {tuple(weights.shape)}")
    return weights.to(_checked_dtype(dtype))
