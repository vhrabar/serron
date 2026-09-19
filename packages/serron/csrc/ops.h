#ifndef SERRON_OPS_H
#define SERRON_OPS_H

#include <torch/torch.h>

#include <tuple>

namespace serron {

/**
 * Grayscale erosion (sliding minimum) of @p input by structuring element @p kernel.
 *
 * @param input   CUDA tensor of shape (N, C, H, W), floating dtype.
 * @param kernel  CUDA structuring element of shape (kH, kW) or (C, kH, kW).
 * @param border  Boundary mode (serron.enums.BorderMode encoding) at the image edges.
 * @return        Eroded tensor, same shape and dtype as @p input.
 */
at::Tensor erode(const at::Tensor& input, const at::Tensor& kernel, int64_t border);

/**
 * Grayscale dilation (sliding maximum) of @p input by structuring element @p kernel.
 *
 * @param input   CUDA tensor of shape (N, C, H, W), floating dtype.
 * @param kernel  CUDA structuring element of shape (kH, kW) or (C, kH, kW).
 * @param border  Boundary mode (serron.enums.BorderMode encoding) at the image edges.
 * @return        Dilated tensor, same shape and dtype as @p input.
 */
at::Tensor dilate(const at::Tensor& input, const at::Tensor& kernel, int64_t border);

/**
 * Grayscale erosion (sliding minimum) of @p input by structuring element @p kernel.
 *
 * @param input   CPU tensor of shape (N, C, H, W), floating dtype.
 * @param kernel  CPU structuring element of shape (kH, kW) or (C, kH, kW).
 * @param border  Boundary mode (serron.enums.BorderMode encoding) at the image edges.
 * @return        Eroded tensor, same shape and dtype as @p input.
 */
at::Tensor erode_cpu(const at::Tensor& input, const at::Tensor& kernel, int64_t border);

/**
 * Grayscale dilation (sliding maximum) of @p input by structuring element @p kernel.
 *
 * @param input   CPU tensor of shape (N, C, H, W), floating dtype.
 * @param kernel  CPU structuring element of shape (kH, kW) or (C, kH, kW).
 * @param border  Boundary mode (serron.enums.BorderMode encoding) at the image edges.
 * @return        Dilated tensor, same shape and dtype as @p input.
 */
at::Tensor dilate_cpu(const at::Tensor& input, const at::Tensor& kernel, int64_t border);

/**
 * Backward pass of grayscale erosion.
 *
 * Erosion selects, per output window, the tap minimising @c sample-se; the backward scatters the upstream gradient to
 * that selected input pixel and (with a sign flip) to the selected structuring-element cell.
 *
 * @param grad_output  Upstream gradient, same shape as the erosion output (N, C, H, W).
 * @param input        The forward input, shape (N, C, H, W).
 * @param kernel       The forward structuring element, shape (kH, kW) or (C, kH, kW).
 * @param border       Boundary mode (serron.enums.BorderMode encoding) used in the forward pass.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 */
std::tuple<at::Tensor, at::Tensor> erode_backward(const at::Tensor& grad_output, const at::Tensor& input,
                                                  const at::Tensor& kernel, int64_t border);

/**
 * Backward pass of grayscale dilation.
 *
 * Dilation selects, per output window, the tap maximising @c sample+se; the backward scatters the upstream gradient to
 * that selected input pixel and to the selected structuring-element cell.
 *
 * @param grad_output  Upstream gradient, same shape as the dilation output (N, C, H, W).
 * @param input        The forward input, shape (N, C, H, W).
 * @param kernel       The forward structuring element, shape (kH, kW) or (C, kH, kW).
 * @param border       Boundary mode (serron.enums.BorderMode encoding) used in the forward pass.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 */
std::tuple<at::Tensor, at::Tensor> dilate_backward(const at::Tensor& grad_output, const at::Tensor& input,
                                                   const at::Tensor& kernel, int64_t border);

/**
 * Backward pass of grayscale erosion (CPU), mirroring @ref erode_backward.
 *
 * @param grad_output  Upstream gradient, CPU tensor of shape (N, C, H, W), same dtype as @p input.
 * @param input        The forward input, shape (N, C, H, W).
 * @param kernel       The forward structuring element, shape (kH, kW) or (C, kH, kW).
 * @param border       Boundary mode (serron.enums.BorderMode encoding) used in the forward pass.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 */
std::tuple<at::Tensor, at::Tensor> erode_backward_cpu(const at::Tensor& grad_output, const at::Tensor& input,
                                                      const at::Tensor& kernel, int64_t border);

/**
 * Backward pass of grayscale dilation (CPU), mirroring @ref dilate_backward.
 *
 * @param grad_output  Upstream gradient, CPU tensor of shape (N, C, H, W), same dtype as @p input.
 * @param input        The forward input, shape (N, C, H, W).
 * @param kernel       The forward structuring element, shape (kH, kW) or (C, kH, kW).
 * @param border       Boundary mode (serron.enums.BorderMode encoding) used in the forward pass.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 */
std::tuple<at::Tensor, at::Tensor> dilate_backward_cpu(const at::Tensor& grad_output, const at::Tensor& input,
                                                       const at::Tensor& kernel, int64_t border);

} // namespace serron

#endif // SERRON_OPS_H