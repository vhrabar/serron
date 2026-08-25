"""
Autograd wrapper around the compiled serron CUDA kernels.
"""

from __future__ import annotations

import torch

from ._cmake_ops import add_op_namespace_prefix, ops

__all__ = ["add_op_namespace_prefix", "ops", "_ErodeFunction", "_DilateFunction"]


class _ErodeFunction(torch.autograd.Function):
    """Autograd binding for ``serron::erode`` / ``serron::erode_backward``."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, input_: torch.Tensor, kernel: torch.Tensor, border: int) -> torch.Tensor:
        ctx.save_for_backward(input_, kernel)
        ctx.border = border
        result: torch.Tensor = ops.erode(input_, kernel, border)
        return result

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        input_, kernel = ctx.saved_tensors
        grad_input, grad_kernel = ops.erode_backward(grad_output.contiguous(), input_, kernel, ctx.border)
        return grad_input, grad_kernel, None


class _DilateFunction(torch.autograd.Function):
    """Autograd binding for ``serron::dilate`` / ``serron::dilate_backward``."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, input_: torch.Tensor, kernel: torch.Tensor, border: int) -> torch.Tensor:
        ctx.save_for_backward(input_, kernel)
        ctx.border = border
        result: torch.Tensor = ops.dilate(input_, kernel, border)
        return result

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        input_, kernel = ctx.saved_tensors
        grad_input, grad_kernel = ops.dilate_backward(grad_output.contiguous(), input_, kernel, ctx.border)
        return grad_input, grad_kernel, None
