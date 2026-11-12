#ifndef SERRON_OPS_H
#define SERRON_OPS_H

#include <morph_algorithm.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>
#include <tuple>

namespace serron {

/// The operators work on stable-ABI tensors
using torch::stable::Tensor;

/**
 * Grayscale erosion (sliding minimum) of @p input by structuring element @p kernel.
 *
 * @param input   CUDA tensor of shape (N, C, H, W), floating dtype.
 * @param kernel  CUDA structuring element of shape (kH, kW) or (C, kH, kW).
 * @param border  Boundary mode (serron.enums.BorderMode encoding) at the image edges.
 * @param flat    Whether @p kernel is flat (all-zero). Pass @c std::nullopt to have the operator check,
 * which reads back from the device and synchronises.
 * @param algo    Algorithm to run (serron.enums.Algorithm encoding); 0 lets the library choose. The CPU
 * operators only take 0.
 * @return        Eroded tensor, same shape and dtype as @p input.
 */
Tensor erode(const Tensor& input, const Tensor& kernel, int64_t border, const std::optional<bool>& flat = std::nullopt,
             int64_t algo = static_cast<int64_t>(Algorithm::kAuto));

/**
 * Grayscale dilation (sliding maximum) of @p input by structuring element @p kernel.
 *
 * @param input   CUDA tensor of shape (N, C, H, W), floating dtype.
 * @param kernel  CUDA structuring element of shape (kH, kW) or (C, kH, kW).
 * @param border  Boundary mode (serron.enums.BorderMode encoding) at the image edges.
 * @param flat    Whether @p kernel is flat (all-zero). Pass @c std::nullopt to have the operator check,
 * which reads back from the device and synchronises.
 * @param algo    Algorithm to run (serron.enums.Algorithm encoding); 0 lets the library choose. The CPU
 * operators only take 0.
 * @return        Dilated tensor, same shape and dtype as @p input.
 */
Tensor dilate(const Tensor& input, const Tensor& kernel, int64_t border, const std::optional<bool>& flat = std::nullopt,
              int64_t algo = static_cast<int64_t>(Algorithm::kAuto));

/**
 * Grayscale erosion (sliding minimum) of @p input by structuring element @p kernel.
 *
 * @param input   CPU tensor of shape (N, C, H, W), floating dtype.
 * @param kernel  CPU structuring element of shape (kH, kW) or (C, kH, kW).
 * @param border  Boundary mode (serron.enums.BorderMode encoding) at the image edges.
 * @param flat    Whether @p kernel is flat (all-zero). Pass @c std::nullopt to have the operator check,
 * which reads back from the device and synchronises.
 * @param algo    Algorithm to run (serron.enums.Algorithm encoding); 0 lets the library choose. The CPU
 * operators only take 0.
 * @return        Eroded tensor, same shape and dtype as @p input.
 */
Tensor erode_cpu(const Tensor& input, const Tensor& kernel, int64_t border,
                 const std::optional<bool>& flat = std::nullopt, int64_t algo = static_cast<int64_t>(Algorithm::kAuto));

/**
 * Grayscale dilation (sliding maximum) of @p input by structuring element @p kernel.
 *
 * @param input   CPU tensor of shape (N, C, H, W), floating dtype.
 * @param kernel  CPU structuring element of shape (kH, kW) or (C, kH, kW).
 * @param border  Boundary mode (serron.enums.BorderMode encoding) at the image edges.
 * @param flat    Whether @p kernel is flat (all-zero). Pass @c std::nullopt to have the operator check,
 * which reads back from the device and synchronises.
 * @param algo    Algorithm to run (serron.enums.Algorithm encoding); 0 lets the library choose. The CPU
 * operators only take 0.
 * @return        Dilated tensor, same shape and dtype as @p input.
 */
Tensor dilate_cpu(const Tensor& input, const Tensor& kernel, int64_t border,
                  const std::optional<bool>& flat = std::nullopt,
                  int64_t algo = static_cast<int64_t>(Algorithm::kAuto));

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
 * @param flat    Whether @p kernel is flat (all-zero). Pass @c std::nullopt to have the operator check,
 * which reads back from the device and synchronises.
 * @param need_kernel_grad  Whether grad_kernel is wanted; when false its scatter is skipped and the
 * returned tensor stays zero.
 * @param algo          Algorithm to run (serron.enums.Algorithm encoding); 0 lets the library choose. The CPU
 * operators only take 0.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 */
std::tuple<Tensor, Tensor> erode_backward(const Tensor& grad_output, const Tensor& input, const Tensor& kernel,
                                          int64_t border, const std::optional<bool>& flat = std::nullopt,
                                          bool need_kernel_grad = true,
                                          int64_t algo = static_cast<int64_t>(Algorithm::kAuto));

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
 * @param flat    Whether @p kernel is flat (all-zero). Pass @c std::nullopt to have the operator check,
 * which reads back from the device and synchronises.
 * @param need_kernel_grad  Whether grad_kernel is wanted; when false its scatter is skipped and the
 * returned tensor stays zero.
 * @param algo          Algorithm to run (serron.enums.Algorithm encoding); 0 lets the library choose. The CPU
 * operators only take 0.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 */
std::tuple<Tensor, Tensor> dilate_backward(const Tensor& grad_output, const Tensor& input, const Tensor& kernel,
                                           int64_t border, const std::optional<bool>& flat = std::nullopt,
                                           bool need_kernel_grad = true,
                                           int64_t algo = static_cast<int64_t>(Algorithm::kAuto));

/**
 * Backward pass of grayscale erosion (CPU), mirroring @ref erode_backward.
 *
 * @param grad_output  Upstream gradient, CPU tensor of shape (N, C, H, W), same dtype as @p input.
 * @param input        The forward input, shape (N, C, H, W).
 * @param kernel       The forward structuring element, shape (kH, kW) or (C, kH, kW).
 * @param border       Boundary mode (serron.enums.BorderMode encoding) used in the forward pass.
 * @param flat    Whether @p kernel is flat (all-zero). Pass @c std::nullopt to have the operator check,
 * which reads back from the device and synchronises.
 * @param need_kernel_grad  Whether grad_kernel is wanted; when false its scatter is skipped and the
 * returned tensor stays zero.
 * @param algo          Algorithm to run (serron.enums.Algorithm encoding); 0 lets the library choose. The CPU
 * operators only take 0.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 */
std::tuple<Tensor, Tensor> erode_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& kernel,
                                              int64_t border, const std::optional<bool>& flat = std::nullopt,
                                              bool need_kernel_grad = true,
                                              int64_t algo = static_cast<int64_t>(Algorithm::kAuto));

/**
 * Backward pass of grayscale dilation (CPU), mirroring @ref dilate_backward.
 *
 * @param grad_output  Upstream gradient, CPU tensor of shape (N, C, H, W), same dtype as @p input.
 * @param input        The forward input, shape (N, C, H, W).
 * @param kernel       The forward structuring element, shape (kH, kW) or (C, kH, kW).
 * @param border       Boundary mode (serron.enums.BorderMode encoding) used in the forward pass.
 * @param flat    Whether @p kernel is flat (all-zero). Pass @c std::nullopt to have the operator check,
 * which reads back from the device and synchronises.
 * @param need_kernel_grad  Whether grad_kernel is wanted; when false its scatter is skipped and the
 * returned tensor stays zero.
 * @param algo          Algorithm to run (serron.enums.Algorithm encoding); 0 lets the library choose. The CPU
 * operators only take 0.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 */
std::tuple<Tensor, Tensor> dilate_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& kernel,
                                               int64_t border, const std::optional<bool>& flat = std::nullopt,
                                               bool need_kernel_grad = true,
                                               int64_t algo = static_cast<int64_t>(Algorithm::kAuto));

/**
 * Bit mask of the algorithms (@ref Algorithm) that can run a call, with bit i set for algorithm i. Auto is never set.
 *
 * @param input     CUDA tensor of the input to the call; its device and dtype are what count.
 * @param kernel    CUDA structuring element of the call.
 * @param flat      Whether @p kernel is flat. Pass @c std::nullopt to have the operator check, which reads back
 * from the device and synchronises.
 * @param backward  Whether the call is the backward pass.
 */
int64_t feasible_algos(const Tensor& input, const Tensor& kernel, const std::optional<bool>& flat, bool backward);

/**
 * The algorithm (@ref Algorithm) the library picks for a call when asked for auto. Arguments as in @ref
 * feasible_algos.
 */
int64_t heuristic_algo(const Tensor& input, const Tensor& kernel, const std::optional<bool>& flat, bool backward);

/**
 * Sets the window length from which the separable path is preferred for the device and dtype of @p like.
 *
 * @param like      CUDA tensor with the device and dtype to set it for.
 * @param backward  Whether it is for the backward pass.
 * @param min_k     New window length, or <= 0 to restore the default.
 * @return          The threshold in effect afterwards, which SERRON_SEPARABLE_MIN_K may still override.
 */
int64_t set_separable_min_k_op(const Tensor& like, bool backward, int64_t min_k);

/// The window length from which the separable path is preferred for the device and dtype of @p like.
int64_t get_separable_min_k_op(const Tensor& like, bool backward);

} // namespace serron

#endif // SERRON_OPS_H