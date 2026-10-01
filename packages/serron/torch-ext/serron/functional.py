"""
Stateless morphological operations backed by the CUDA kernels.
"""

from __future__ import annotations

import torch

from . import _flatness
from ._cmake_ops import ops
from .autograd import _DilateFunction, _ErodeFunction
from .enums import BORDER_TO_INT, BorderMode


def erosion(input_: torch.Tensor, kernel: torch.Tensor, *, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """
    Grayscale erosion (sliding minimum) of ``input`` by ``kernel``
    :param input_: input tensor
    :param kernel: kernel tensor
    :param border: border mode
    :return: eroded tensor
    """
    result: torch.Tensor = _ErodeFunction.apply(input_, kernel, BORDER_TO_INT[border])
    return result


def dilation(input_: torch.Tensor, kernel: torch.Tensor, *, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """
    Grayscale dilation (sliding maximum) of ``input`` by ``kernel``
    :param input_: input tensor
    :param kernel: kernel tensor
    :param border: border mode
    :return: dilated tensor
    """
    result: torch.Tensor = _DilateFunction.apply(input_, kernel, BORDER_TO_INT[border])
    return result


def opening(input_: torch.Tensor, kernel: torch.Tensor, *, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """
    Opening: erosion followed by dilation.
    :param input_: input tensor
    :param kernel: kernel tensor
    :param border: border mode
    :return: top-hat tensor
    """
    return dilation(erosion(input_, kernel, border=border), kernel, border=border)


def closing(input_: torch.Tensor, kernel: torch.Tensor, *, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """
    "Closing: dilation followed by erosion.
    :param input_: input tensor
    :param kernel: kernel tensor
    :param border: border mode
    :return: top-hat tensor
    """
    return erosion(dilation(input_, kernel, border=border), kernel, border=border)


def gradient(input_: torch.Tensor, kernel: torch.Tensor, *, border: BorderMode = BorderMode.REPLICATE) -> torch.Tensor:
    """
    Morphological gradient: ``dilate(input) - erode(input)``.
    :param input_: input tensor
    :param kernel: kernel tensor
    :param border: border mode
    :return: top-hat tensor
    """
    return dilation(input_, kernel, border=border) - erosion(input_, kernel, border=border)


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


_DEFAULT_CHECK_EVERY = 16


def _align_inputs(
    marker: torch.Tensor, mask: torch.Tensor, kernel: torch.Tensor, name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """
    Check the three tensors a reconstruction runs on and bring them to one dtype.

    :param marker: propagation seed, shape ``(N, C, H, W)``.
    :param mask: tensor constraining the propagation, same shape as ``marker``.
    :param kernel: structuring element, ``(kH, kW)`` or ``(C, kH, kW)``.
    :param name: qualified caller name used to prefix diagnostics.
    :return: ``(marker, mask, kernel)`` sharing a dtype, plus whether ``kernel`` is flat.
    :raises ValueError: if the shapes or devices disagree.
    :raises RuntimeError: if autograd is live and one of the inputs wants a gradient.
    """
    if marker.shape != mask.shape:
        raise ValueError(f"{name}: marker and mask must share a shape, got {tuple(marker.shape)} and {tuple(mask.shape)}")
    if marker.device != mask.device or marker.device != kernel.device:
        raise ValueError(
            f"{name}: marker, mask and kernel must be on one device, got {marker.device}, {mask.device} and {kernel.device}"
        )
    if torch.is_grad_enabled() and (marker.requires_grad or mask.requires_grad or kernel.requires_grad):
        raise RuntimeError(f"{name}: reconstruction has no backward pass yet; run it under torch.no_grad() or detach its inputs")

    compute_dtype = torch.promote_types(torch.promote_types(marker.dtype, mask.dtype), kernel.dtype)
    # Read flatness off the caller's kernel, so a cast does not throw away a cached answer.
    flat = _flatness.is_flat(kernel)
    return marker.to(compute_dtype), mask.to(compute_dtype), _flatness.mark(kernel.to(compute_dtype), flat), flat


def _require_nonnegative_anchor(kernel: torch.Tensor, name: str) -> None:
    """
    Reject a structuring element that would let a geodesic step move the wrong way.

    :param kernel: structuring element, ``(kH, kW)`` or ``(C, kH, kW)``.
    :param name: qualified caller name used to prefix diagnostics.
    :raises ValueError: if any channel's anchor tap is negative or NaN.
    """
    anchor_h, anchor_w = kernel.shape[-2] // 2, kernel.shape[-1] // 2
    anchor = kernel[..., anchor_h, anchor_w]
    if not bool((anchor >= 0).all().item()):
        raise ValueError(
            f"{name}: the structuring element's anchor tap kernel[..., {anchor_h}, {anchor_w}] must be >= 0, "
            f"otherwise the geodesic iteration is not monotone and need not converge"
        )


def _reconstruct(
    marker: torch.Tensor,
    mask: torch.Tensor,
    kernel: torch.Tensor,
    border: BorderMode,
    *,
    dilate: bool,
    max_iter: int | None,
    check_every: int,
    name: str,
) -> torch.Tensor:
    """
    Iterate a geodesic step to its fixed point.

    :param marker: propagation seed, shape ``(N, C, H, W)``.
    :param mask: tensor constraining the propagation, same shape as ``marker``.
    :param kernel: structuring element, ``(kH, kW)`` or ``(C, kH, kW)``.
    :param border: boundary mode for out-of-image reads.
    :param dilate: ``True`` to reconstruct by dilation, ``False`` by erosion.
    :param max_iter: iteration budget, or ``None`` to run until the fixed point.
    :param check_every: how many steps to run between convergence tests.
    :param name: qualified caller name used to prefix diagnostics.
    :return: the reconstruction, same shape as ``marker`` and dtype promoted across the inputs.
    :raises ValueError: if the inputs disagree, or ``max_iter`` / ``check_every`` are out of range.
    :raises RuntimeError: if ``max_iter`` runs out before the fixed point.
    """
    if check_every < 1:
        raise ValueError(f"{name}: check_every must be >= 1, got {check_every}")
    if max_iter is not None and max_iter < 1:
        raise ValueError(f"{name}: max_iter must be >= 1 or None, got {max_iter}")

    marker, mask, kernel, flat = _align_inputs(marker, mask, kernel, name)
    if not flat:
        _require_nonnegative_anchor(kernel, name)

    border_int = BORDER_TO_INT[border]
    step = ops.dilate if dilate else ops.erode
    clip = torch.minimum if dilate else torch.maximum

    with torch.no_grad():
        current = clip(marker, mask)
        snapshot = current
        iterations = 0
        while max_iter is None or iterations < max_iter:
            current = clip(step(current, kernel, border_int, flat), mask)
            iterations += 1
            if iterations % check_every == 0:
                if torch.equal(current, snapshot):
                    return current
                snapshot = current

        if torch.equal(current, snapshot):
            return current

    raise RuntimeError(
        f"{name}: no fixed point after {max_iter} iterations; raise max_iter, or pass finite inputs "
        f"(a NaN never compares equal, so the convergence test can never fire)"
    )


def reconstruction_by_dilation(
    marker: torch.Tensor,
    mask: torch.Tensor,
    kernel: torch.Tensor,
    *,
    border: BorderMode = BorderMode.CONSTANT,
    max_iter: int | None = None,
    check_every: int = _DEFAULT_CHECK_EVERY,
) -> torch.Tensor:
    """
    Geodesic reconstruction by dilation: the fixed point of ``min(dilation(y), mask)`` from ``y = marker``.

    Values propagate out of ``marker`` along the structuring element, held down by ``mask`` at every
    step, so a seeded peak floods only as far as ``mask`` allows. ``marker`` is clipped to
    ``min(marker, mask)`` first, which is the precondition the definition asks for.

    This operation is not differentiable yet and raises if handed an input that wants a gradient.

    :param marker: propagation seed, shape ``(N, C, H, W)``.
    :param mask: upper bound on the propagation, same shape as ``marker``.
    :param kernel: structuring element, ``(kH, kW)`` or ``(C, kH, kW)``, anchor tap >= 0.
    :param border: boundary mode for out-of-image reads.
    :param max_iter: iteration budget, or ``None`` to run until the fixed point.
    :param check_every: how many steps to run between convergence tests; larger trades a few wasted
        steps for fewer device-to-host reads.
    :return: the reconstruction, same shape as ``marker`` and dtype promoted across the inputs.
    :raises ValueError: if the inputs disagree, or ``max_iter`` / ``check_every`` are out of range.
    :raises RuntimeError: if an input wants a gradient, or ``max_iter`` runs out before the fixed point.
    """
    return _reconstruct(
        marker,
        mask,
        kernel,
        border,
        dilate=True,
        max_iter=max_iter,
        check_every=check_every,
        name="serron.reconstruction_by_dilation",
    )


def reconstruction_by_erosion(
    marker: torch.Tensor,
    mask: torch.Tensor,
    kernel: torch.Tensor,
    *,
    border: BorderMode = BorderMode.CONSTANT,
    max_iter: int | None = None,
    check_every: int = _DEFAULT_CHECK_EVERY,
) -> torch.Tensor:
    """
    Geodesic reconstruction by erosion: the fixed point of ``max(erosion(y), mask)`` from ``y = marker``.

    The dual of :func:`reconstruction_by_dilation`: ``marker`` sits above ``mask``, is clipped to
    ``max(marker, mask)`` first, and the iteration descends until it settles.

    This operation is not differentiable yet and raises if handed an input that wants a gradient.

    :param marker: propagation seed, shape ``(N, C, H, W)``.
    :param mask: lower bound on the propagation, same shape as ``marker``.
    :param kernel: structuring element, ``(kH, kW)`` or ``(C, kH, kW)``, anchor tap >= 0.
    :param border: boundary mode for out-of-image reads.
    :param max_iter: iteration budget, or ``None`` to run until the fixed point.
    :param check_every: how many steps to run between convergence tests; larger trades a few wasted
        steps for fewer device-to-host reads.
    :return: the reconstruction, same shape as ``marker`` and dtype promoted across the inputs.
    :raises ValueError: if the inputs disagree, or ``max_iter`` / ``check_every`` are out of range.
    :raises RuntimeError: if an input wants a gradient, or ``max_iter`` runs out before the fixed point.
    """
    return _reconstruct(
        marker,
        mask,
        kernel,
        border,
        dilate=False,
        max_iter=max_iter,
        check_every=check_every,
        name="serron.reconstruction_by_erosion",
    )
