"""
Autograd wrapper around the compiled serron CUDA kernels.
"""

from __future__ import annotations

import torch

from ._cmake_ops import add_op_namespace_prefix, ops

__all__ = ["_DilateFunction", "_ErodeFunction", "add_op_namespace_prefix", "ops"]


def _match_dtype(input_: torch.Tensor, kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cast ``input_`` and ``kernel`` to their common floating dtype.


    :param input_: image tensor.
    :param kernel: structuring-element tensor.
    :returns: ``(input_, kernel)`` sharing ``torch.promote_types(input_.dtype, kernel.dtype)``.
    """
    compute_dtype = torch.promote_types(input_.dtype, kernel.dtype)
    return input_.to(compute_dtype), kernel.to(compute_dtype)


class _ErodeFunction(torch.autograd.Function):
    """Autograd binding for ``serron::erode`` / ``serron::erode_backward``."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input_: torch.Tensor, kernel: torch.Tensor, border: int
    ) -> torch.Tensor:
        input_, kernel = _match_dtype(input_, kernel)
        ctx.save_for_backward(input_, kernel)
        ctx.border = border  # type: ignore[attr-defined]
        result: torch.Tensor = ops.erode(input_, kernel, border)
        return result

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        input_, kernel = ctx.saved_tensors  # type: ignore[attr-defined]
        grad_output = grad_output.to(input_.dtype).contiguous()
        grad_input, grad_kernel = ops.erode_backward(grad_output, input_, kernel, ctx.border)  # type: ignore[attr-defined]
        return grad_input, grad_kernel, None


class _DilateFunction(torch.autograd.Function):
    """Autograd binding for ``serron::dilate`` / ``serron::dilate_backward``."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input_: torch.Tensor, kernel: torch.Tensor, border: int
    ) -> torch.Tensor:
        input_, kernel = _match_dtype(input_, kernel)
        ctx.save_for_backward(input_, kernel)
        ctx.border = border  # type: ignore[attr-defined]
        result: torch.Tensor = ops.dilate(input_, kernel, border)
        return result

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        input_, kernel = ctx.saved_tensors  # type: ignore[attr-defined]
        grad_output = grad_output.to(input_.dtype).contiguous()
        grad_input, grad_kernel = ops.dilate_backward(grad_output, input_, kernel, ctx.border)  # type: ignore[attr-defined]
        return grad_input, grad_kernel, None
