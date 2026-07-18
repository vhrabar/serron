"""
Structuring-element builders.
"""

from __future__ import annotations

import torch

_OFF = float("-inf")


def _flat(mask: torch.Tensor) -> torch.Tensor:
    """
    Turn a boolean membership ``mask`` into a flat additive SE (0 / -inf).
    :param mask: input mask
    :return: flat additive SE
    """
    return torch.where(mask, 0.0, _OFF).to(torch.float32)


def _centered_coords(size: int, device: torch.device | str | None) -> torch.Tensor:
    """
    1-D coordinate axis centered on 0
    :param size: size of the cord axis
    :param device: torch device
    :return: cord axis of size ``sieze`` centered on 0
    """
    return torch.arange(size, device=device) - (size - 1) // 2


def square(size: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Square  SE of shape ``(size, size)``.
    :param size: SE size
    :param device: device to create the SE on
    :return: SE tensor
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    return torch.zeros(size, size, dtype=torch.float32, device=device)


def cross(size: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Cross-shaped SE of shape ``(size, size)``.
    :param size: SE size (should be odd for a symmetric cross)
    :param device: device to create the SE on
    :return: SE tensor
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    coords = _centered_coords(size, device)
    dy, dx = torch.meshgrid(coords, coords, indexing="ij")
    return _flat((dy == 0) | (dx == 0))


def disk(radius: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Disk-shaped SE of shape ``(2*radius+1, 2*radius+1)``.
    :param radius: SE radius
    :param device: device to create the SE on
    :return: SE tensor
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    coords = _centered_coords(2 * radius + 1, device)
    dy, dx = torch.meshgrid(coords, coords, indexing="ij")
    return _flat(dx * dx + dy * dy <= radius * radius)


def diamond(radius: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Diamond SE of shape ``(2*radius+1, 2*radius+1)``.
    :param radius: SE radius
    :param device: device to create the SE on
    :return: SE tensor
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    coords = _centered_coords(2 * radius + 1, device)
    dy, dx = torch.meshgrid(coords, coords, indexing="ij")
    return _flat(dx.abs() + dy.abs() <= radius)


def from_tensor(weights: torch.Tensor) -> torch.Tensor:
    """
    Wrap an arbitrary 2-D tensor as a grayscale SE.
    :param weights: 2-D tensor of weights
    :return: SE tensor
    """
    if weights.dim() != 2:
        raise ValueError(f"expected a 2-D (kh, kw) tensor, got shape {tuple(weights.shape)}")
    return weights.to(torch.float32)
