#include "ops.h"

#include <cuda/morphology/enums.cuh>
#include <cuda/morphology/ops_policy.cuh>
#include <cuda/utils/boundaries.cuh>
#include <cuda/utils/declarations.cuh>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <tuple>

namespace serron {

namespace {

/**
 * Shared host-side backward for @ref erode_backward / @ref dilate_backward.
 *
 * Both primitives select a single tap per output window (argmin for erosion, argmax for dilation) and the backward
 * scatters @p grad_output to the selected input pixel and structuring-element cell. Recomputing the selection here
 * keeps the tap identical to the forward reduction (@ref ErodeOp / @ref DilateOp in ops_policy.cuh).
 *
 *
 * @param grad_output  Upstream gradient, shape (N, C, H, W).
 * @param input        Forward input, shape (N, C, H, W).
 * @param kernel       Forward structuring element, shape (kH, kW) or (C, kH, kW).
 * @param border       Boundary mode (@ref BorderMode encoding).
 * @param op           Operation to differentiate (@ref MorphOp).
 * @param name         Qualified caller name used to prefix diagnostics.
 * @return             Pair (grad_input, grad_kernel) matching @p input and @p kernel.
 */
std::tuple<at::Tensor, at::Tensor> morphology_backward_impl(const at::Tensor& grad_output, const at::Tensor& input,
                                                            const at::Tensor& kernel, int64_t border, MorphOp op,
                                                            const char* name) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, name, ": backward is not implemented yet");
}

} // namespace

std::tuple<at::Tensor, at::Tensor> erode_backward(const at::Tensor& grad_output, const at::Tensor& input,
                                                  const at::Tensor& kernel, const int64_t border) {
    return morphology_backward_impl(grad_output, input, kernel, border, MorphOp::kErode, "serron::erode_backward");
}

std::tuple<at::Tensor, at::Tensor> dilate_backward(const at::Tensor& grad_output, const at::Tensor& input,
                                                   const at::Tensor& kernel, const int64_t border) {
    return morphology_backward_impl(grad_output, input, kernel, border, MorphOp::kDilate, "serron::dilate_backward");
}

} // namespace serron