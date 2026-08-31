"""
Learnable morphology layers (``torch.nn.Module``).
"""

from __future__ import annotations

import torch
from torch import nn

from . import functional as F
from .enums import BorderMode


class _MorphologyNd(nn.Module):
    """Shared base: owns the learnable SE parameter."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        border: BorderMode = BorderMode.REPLICATE,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.border = border
        self.weight = nn.Parameter(torch.empty(channels, kernel_size, kernel_size, dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Erosion2d(_MorphologyNd):
    """Learnable 2-D erosion layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.erosion(x, self.weight, border=self.border)


class Dilation2d(_MorphologyNd):
    """Learnable 2-D dilation layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dilation(x, self.weight, border=self.border)


class Opening2d(_MorphologyNd):
    """Learnable 2-D opening layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.opening(x, self.weight, border=self.border)


class Closing2d(_MorphologyNd):
    """Learnable 2-D closing layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.closing(x, self.weight, border=self.border)
