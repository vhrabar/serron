#ifndef SERRON_OPS_H
#define SERRON_OPS_H

#include <torch/torch.h>

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

} // namespace serron

#endif // SERRON_OPS_H