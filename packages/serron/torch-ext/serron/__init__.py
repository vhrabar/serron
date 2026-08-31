"""
Serron: CUDA-accelerated mathematical morphology for PyTorch.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import functional, structuring_element
from .enums import BorderMode, Operation
from .functional import (
    black_hat,
    closing,
    dilation,
    erosion,
    gradient,
    opening,
    top_hat,
)
from .modules import Closing2d, Dilation2d, Erosion2d, Opening2d

try:
    __version__ = _version("serron")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "BorderMode",
    "Closing2d",
    "Dilation2d",
    "Erosion2d",
    "Opening2d",
    "Operation",
    "__version__",
    "black_hat",
    "closing",
    "dilation",
    "erosion",
    "functional",
    "gradient",
    "opening",
    "structuring_element",
    "top_hat",
]
