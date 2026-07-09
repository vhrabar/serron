"""
Serron: CUDA-accelerated mathematical morphology for PyTorch.
"""

from __future__ import annotations

from . import functional, structuring_element
from .enums import BorderMode, Operation
from .functional import (
    black_hat,
    closing,
    dilatation,
    erosion,
    gradient,
    opening,
    top_hat,
)
from .modules import Closing2d, Dilation2d, Erosion2d, Opening2d

__all__ = [
    "BorderMode",
    "Closing2d",
    "Dilation2d",
    "Erosion2d",
    "Opening2d",
    "Operation",
    "black_hat",
    "closing",
    "dilatation",
    "erosion",
    "functional",
    "gradient",
    "opening",
    "structuring_element",
    "top_hat",
]
