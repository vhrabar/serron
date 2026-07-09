"""
Structuring-element builders.
"""

from __future__ import annotations

import torch


def square(size: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Square (full) SE of shape ``(size, size)``.
    :param size: SE size
    :param device: device to create the SE on
    :return: SE tensor
    """
    raise NotImplementedError


def cross(size: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Cross/plus-shaped SE of shape ``(size, size)``.
    :param size: SE size
    :param device: device to create the SE on
    :return: SE tensor"""
    raise NotImplementedError


def disk(radius: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Disk SE of shape ``(2*radius+1, 2*radius+1)``.
    :param radius: SE radius
    :param device: device to create the SE on
    :return: SE tensor
    """
    raise NotImplementedError


def diamond(radius: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Diamond SE of shape ``(2*radius+1, 2*radius+1)``.
    :param radius: SE radius
    :param device: device to create the SE on
    :return: SE tensor
    """
    raise NotImplementedError


def from_tensor(weights: torch.Tensor) -> torch.Tensor:
    """
    Wrap an arbitrary 2-D tensor as a grayscale SE.
    :param weights: 2-D tensor of weights
    :return: SE tensor
    """
    raise NotImplementedError
