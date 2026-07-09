"""
Enumerations shared across the morphology ops and layers.
"""

from __future__ import annotations

from enum import Enum


class Operation(Enum):
    """
    Morphological operations
    """

    ERODE = "erode"
    DILATE = "dilate"
    OPEN = "open"
    CLOSE = "close"
    GRADIENT = "gradient"
    TOP_HAT = "top_hat"
    BLACK_HAT = "black_hat"


class BorderMode(Enum):
    """
    OuB neighborhood border mode.
    """

    REFLECT = "reflect"
    REPLICATE = "replicate"
    CONSTANT = "constant"
