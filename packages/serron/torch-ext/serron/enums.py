"""
Enumerations shared across the morphology ops and layers.
"""

from __future__ import annotations

from enum import Enum, IntEnum


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


BORDER_TO_INT: dict[BorderMode, int] = {
    BorderMode.REFLECT: 0,
    BorderMode.REPLICATE: 1,
    BorderMode.CONSTANT: 2,
}
"""Encoding of :class:`BorderMode` the compiled ops take as their ``border`` argument."""


class Algorithm(IntEnum):
    """
    Algorithm a compiled op runs, as the ``algo`` argument encodes it.

    ``AUTO`` lets the library choose. The others force one; a CUDA op raises if the forced algorithm cannot run on
    the problem, and a CPU op only takes ``AUTO``.
    """

    AUTO = 0
    ELEMENT = 1
    TILED = 2
    SEPARABLE = 3
