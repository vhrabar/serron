"""
Stateless morphological operations backed by the CUDA kernels.
"""

from __future__ import annotations

import torch

from .autograd import _DilateFunction, _ErodeFunction
from .enums import BorderMode

_BORDER_TO_INT: dict[BorderMode, int] = {
    BorderMode.REFLECT: 0,
    BorderMode.REPLICATE: 1,
    BorderMode.CONSTANT: 2,
}


def erosion(input_: torch.Tensor, kernel: torch.Tensor, *, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """
    Grayscale erosion (sliding minimum) of ``input`` by ``kernel``
    :param input_: input tensor
    :param kernel: kernel tensor
    :param border: border mode
    :return: eroded tensor
    """
    result: torch.Tensor = _ErodeFunction.apply(input_, kernel, _BORDER_TO_INT[border])  # type: ignore[no-untyped-call]
    return result


def dilatation(input_: torch.Tensor, kernel: torch.Tensor, *, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """
    Grayscale dilation (sliding maximum) of ``input`` by ``kernel``
    :param input_: input tensor
    :param kernel: kernel tensor
    :param border: border mode
    :return: dilated tensor
    """
    result: torch.Tensor = _DilateFunction.apply(input_, kernel, _BORDER_TO_INT[border])  # type: ignore[no-untyped-call]
    return result


def opening(input_: torch.Tensor, kernel: torch.Tensor, *, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """
    Opening: erosion followed by dilation.
    :param input_: input tensor
    :param kernel: kernel tensor
    :param border: border mode
    :return: top-hat tensor
    """
    return dilatation(erosion(input_, kernel, border=border), kernel, border=border)


def closing(input_: torch.Tensor, kernel: torch.Tensor, *, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """
    "Closing: dilation followed by erosion.
    :param input_: input tensor
    :param kernel: kernel tensor
    :param border: border mode
    :return: top-hat tensor
    """
    return erosion(dilatation(input_, kernel, border=border), kernel, border=border)


def gradient(input_: torch.Tensor, kernel: torch.Tensor, *, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """
    Morphological gradient: ``dilate(input) - erode(input)``.
    :param input_: input tensor
    :param kernel: kernel tensor
    :param border: border mode
    :return: top-hat tensor
    """
    return dilatation(input_, kernel, border=border) - erosion(input_, kernel, border=border)


def top_hat(input_: torch.Tensor, kernel: torch.Tensor, *, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """
    White top-hat: ``input - open(input)``
    :param input_: input tensor
    :param kernel: kernel tensor
    :param border: border mode
    :return: top-hat tensor
    """
    return input_ - opening(input_, kernel, border=border)


def black_hat(input_: torch.Tensor, kernel: torch.Tensor, *, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """
    Black top-hat: ``close(input) - input``
    :param input_: input tensor
    :param kernel: kernel tensor
    :param border: border mode
    :return: top-hat tensor
    """
    return closing(input_, kernel, border=border) - input_
