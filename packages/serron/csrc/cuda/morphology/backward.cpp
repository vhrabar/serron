#include "ops.h"

#include <cuda/morphology/arg_tap.h>
#include <cuda/morphology/entry_points.h>
#include <cuda/morphology/enums.h>
#include <cuda/morphology/kernel_tables.h>
#include <cuda/morphology/separable.h>
#include <cuda/utils/border_mode.h>
#include <cuda/utils/declarations.h>
#include <cuda/utils/launch.h>
#include <cuda/utils/smem.h>

#include <cuda_runtime.h>

#include <backward_fatbin.h>
#include <compat/acc_type.h>
#include <compat/check.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <tuple>

namespace serron {

namespace {

/// The backward kernels' fatbin, embedded at build time and loaded on first use.
cudaLibrary_t backward_library() {
    static const cudaLibrary_t library = load_kernel_library(serron_backward_fatbin, "backward");
    return library;
}

/// Backward entry points for @p op on @c scalar_t, looked up by name on first use; specialised per dtype below.
template <typename scalar_t>
const BackwardKernels<scalar_t>& backward_kernels(MorphOp op);

#define SERRON_BACKWARD_TABLE_ROW(Op, op, scalar_t, suffix)                                                            \
    BackwardKernels<scalar_t>{SERRON_BACKWARD_KERNELS(SERRON_GET_ENTRY_POINT, op, scalar_t, suffix)},
#define SERRON_BACKWARD_TABLE(scalar_t, suffix)                                                                        \
    template <>                                                                                                        \
    const BackwardKernels<scalar_t>& backward_kernels<scalar_t>(const MorphOp op) {                                    \
        static const std::array<BackwardKernels<scalar_t>, 2> table = [] {                                             \
            const cudaLibrary_t library = backward_library();                                                          \
            return std::array<BackwardKernels<scalar_t>, 2>{                                                           \
                SERRON_CUDA_FOR_EACH_OP(SERRON_BACKWARD_TABLE_ROW, scalar_t, suffix)};                                 \
        }();                                                                                                           \
        return table[static_cast<int>(op)];                                                                            \
    }
SERRON_CUDA_FOR_EACH_DTYPE(SERRON_BACKWARD_TABLE)

/**
 * Launches the separable backward: the row pass into @p row_best_val / @p row_best_dj,
 * then the column pass scattering into @p grad_input / @p grad_kernel. O(kH+kW) per
 * output element versus @ref morphology_backward_kernel's O(kH*kW).
 */
template <typename scalar_t>
bool launch_morphology_backward_separable(const BackwardKernels<scalar_t>& kernels, const scalar_t* input,
                                          const scalar_t* grad_output, scalar_t* grad_input, scalar_t* grad_kernel,
                                          scalar_t* row_best_val, int32_t* row_best_dj, int64_t N, int64_t C, int64_t H,
                                          int64_t W, int64_t kH, int64_t kW, int64_t kernel_channel_stride,
                                          BorderMode border, bool need_kernel_grad, cudaStream_t stream) {
    using acc_t = acc_type<scalar_t, true>;
    constexpr size_t kTap = sizeof(ArgTap<acc_t>);

    const dim3 block(LINE_TILE);
    const size_t budget = smem_budget();

    const int64_t chunks_row = line_chunks_per_block(kW, W, kTap, budget);
    const int64_t chunks_col = line_chunks_per_block(kH, H, kTap, budget);
    const size_t smem_row = line_smem_bytes(chunks_row, kW, kTap);
    const size_t smem_col = line_smem_bytes(chunks_col, kH, kTap);

    if (!configure_kernel_smem(kernels.row_argreduce, smem_row) ||
        !configure_kernel_smem(kernels.col_argreduce, smem_col)) {
        return false;
    }

    const int64_t out_row = chunks_row * kW - kW + 1;
    const dim3 grid_row(static_cast<unsigned int>((W + out_row - 1) / out_row), static_cast<unsigned int>(H),
                        static_cast<unsigned int>(N * C));
    launch_kernel(kernels.row_argreduce, grid_row, block, smem_row, stream, input, row_best_val, row_best_dj, H, W, kW,
                  chunks_row, border);

    const int64_t out_col = chunks_col * kH - kH + 1;
    const dim3 grid_col(static_cast<unsigned int>((H + out_col - 1) / out_col), static_cast<unsigned int>(W),
                        static_cast<unsigned int>(N * C));
    launch_kernel(kernels.col_argreduce, grid_col, block, smem_col, stream, row_best_val, row_best_dj, grad_output,
                  grad_input, grad_kernel, C, H, W, kH, kW, chunks_col, kernel_channel_stride, border,
                  need_kernel_grad);
    return true;
}

/**
 * Launch the morphology backward pass on @p stream. Uses the separable row+column
 * argreduce when @p use_separable was set by the caller (flat, axis-separable SE
 * at/above @ref separable_min_k); otherwise recomputes the winning tap directly, as
 * @ref morphology_backward_kernel.
 */
template <typename scalar_t>
void launch_morphology_backward(const BackwardKernels<scalar_t>& kernels, const scalar_t* grad_output,
                                const scalar_t* input, const scalar_t* kernel, scalar_t* grad_input,
                                scalar_t* grad_kernel, scalar_t* row_best_val, int32_t* row_best_dj, bool use_separable,
                                int64_t N, int64_t C, int64_t H, int64_t W, int64_t kH, int64_t kW,
                                int64_t kernel_channel_stride, BorderMode border, bool need_kernel_grad,
                                cudaStream_t stream) {
    using acc_t = acc_type<scalar_t, true>;

    // One scan chunk is the smallest tile a line pass can stage. If even that overflows the
    // budget, the kernels below take over no matter how separable the SE is.
    const size_t budget = smem_budget();
    if (use_separable && line_smem_bytes(1, std::max(kH, kW), sizeof(ArgTap<acc_t>)) <= budget &&
        launch_morphology_backward_separable<scalar_t>(kernels, input, grad_output, grad_input, grad_kernel,
                                                       row_best_val, row_best_dj, N, C, H, W, kH, kW,
                                                       kernel_channel_stride, border, need_kernel_grad, stream)) {
        return;
    }

    const int64_t tile_w = TILE_X + kW - 1;
    const int64_t tile_h = TILE_Y + kH - 1;
    const size_t smem = (static_cast<size_t>(tile_h * tile_w) + static_cast<size_t>(kH * kW)) * sizeof(scalar_t);

    // Tile if it fits, element-wise otherwise
    if (smem <= budget && configure_kernel_smem(kernels.tiled, smem)) {
        constexpr dim3 block(TILE_X, TILE_Y);
        const dim3 grid(static_cast<unsigned int>((W + TILE_X - 1) / TILE_X),
                        static_cast<unsigned int>((H + TILE_Y - 1) / TILE_Y), static_cast<unsigned int>(N * C));
        launch_kernel(kernels.tiled, grid, block, smem, stream, grad_output, input, kernel, grad_input, grad_kernel, C,
                      H, W, kH, kW, kernel_channel_stride, border, need_kernel_grad);
    } else {
        const int64_t total = N * C * H * W;
        const auto blocks = static_cast<unsigned int>((total + THREADS - 1) / THREADS);
        launch_kernel(kernels.element, blocks, THREADS, 0, stream, grad_output, input, kernel, grad_input, grad_kernel,
                      N, C, H, W, kH, kW, kernel_channel_stride, border, need_kernel_grad);
    }
}

/**
 * Shared host-side backward behind @ref erode_backward / @ref dilate_backward
 *
 * @param grad_output  Upstream gradient, CUDA tensor of shape (N, C, H, W), same dtype as @p input.
 * @param input        Forward input, CUDA tensor of shape (N, C, H, W), floating dtype.
 * @param kernel       Forward structuring element, CUDA tensor of shape (kH, kW) or (C, kH, kW); same dtype as @p
 * input.
 * @param border       Boundary mode (@ref BorderMode encoding) used in the forward pass.
 * @param op           Operation to differentiate (@ref MorphOp).
 * @param name         Qualified caller name used to prefix diagnostics.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 * @throws std::runtime_error  if the tensors are not on CUDA, have the wrong rank or dtype, the channel counts
 * disagree, or @p border is out of range.
 */
std::tuple<Tensor, Tensor> morphology_backward_impl(const Tensor& grad_output, const Tensor& input,
                                                    const Tensor& kernel, int64_t border,
                                                    const std::optional<bool>& flat, bool need_kernel_grad, MorphOp op,
                                                    const char* name) {
    SERRON_CHECK(grad_output.is_cuda(), name, ": grad_output must be a CUDA tensor");
    SERRON_CHECK(input.is_cuda(), name, ": input must be a CUDA tensor");
    SERRON_CHECK(kernel.is_cuda(), name, ": kernel must be a CUDA tensor");
    SERRON_CHECK(input.dim() == 4, name, ": input must be 4-D (N, C, H, W), got ", input.dim(), "-D");
    SERRON_CHECK(grad_output.dim() == 4, name, ": grad_output must be 4-D (N, C, H, W), got ", grad_output.dim(), "-D");
    SERRON_CHECK(kernel.dim() == 2 || kernel.dim() == 3, name, ": kernel must be 2-D (kH, kW) or 3-D (C, kH, kW), got ",
                 kernel.dim(), "-D");
    SERRON_CHECK(input.scalar_type() == kernel.scalar_type(), name, ": input and kernel must share a dtype");
    SERRON_CHECK(grad_output.scalar_type() == input.scalar_type(), name, ": grad_output and input must share a dtype");
    SERRON_CHECK(border >= kReflect && border <= kConstant, name, ": invalid border mode ", border);

    const Tensor grad_output_c = torch::stable::contiguous(grad_output);
    const Tensor input_c = torch::stable::contiguous(input);
    const Tensor kernel_c = torch::stable::contiguous(kernel);

    const int64_t N = input_c.size(0);
    const int64_t C = input_c.size(1);
    const int64_t H = input_c.size(2);
    const int64_t W = input_c.size(3);

    int64_t kH = 0;
    int64_t kW = 0;
    int64_t kernel_channel_stride = 0;
    if (kernel_c.dim() == 3) {
        SERRON_CHECK(kernel_c.size(0) == C, name, ": kernel channel dim (", kernel_c.size(0),
                     ") must match input channels (", C, ")");
        kH = kernel_c.size(1);
        kW = kernel_c.size(2);
        kernel_channel_stride = kH * kW;
    } else {
        kH = kernel_c.size(0);
        kW = kernel_c.size(1);
        kernel_channel_stride = 0;
    }
    SERRON_CHECK(kH > 0 && kW > 0, name, ": kernel spatial dims must be positive");
    SERRON_CHECK(grad_output_c.sizes() == input_c.sizes(), name, ": grad_output shape must match input");

    Tensor grad_input = torch::stable::new_zeros(input_c, input_c.sizes());
    Tensor grad_kernel = torch::stable::new_zeros(kernel_c, kernel_c.sizes());
    if (input_c.numel() == 0)
        return {grad_input, grad_kernel};

    const bool use_separable = use_separable_path(resolve_flat(flat, kernel_c), kH, kW);
    Tensor row_best_val;
    Tensor row_best_dj;
    if (use_separable) {
        row_best_val = torch::stable::empty_like(input_c);
        row_best_dj = torch::stable::new_empty(input_c, input_c.sizes(), torch::headeronly::ScalarType::Int);
    }

    const torch::stable::accelerator::DeviceGuard device_guard(input_c.get_device_index());
    const auto stream = static_cast<cudaStream_t>(
        torch::stable::accelerator::getCurrentStream(input_c.get_device_index()).nativeHandle());
    const auto border_mode = static_cast<BorderMode>(border);

    THO_DISPATCH_V2(input_c.scalar_type(), "serron_morphology_backward", AT_WRAP([&] {
                        scalar_t* row_val_ptr = use_separable ? row_best_val.mutable_data_ptr<scalar_t>() : nullptr;
                        int32_t* row_dj_ptr = use_separable ? row_best_dj.mutable_data_ptr<int32_t>() : nullptr;
                        launch_morphology_backward<scalar_t>(
                            backward_kernels<scalar_t>(op), grad_output_c.const_data_ptr<scalar_t>(),
                            input_c.const_data_ptr<scalar_t>(), kernel_c.const_data_ptr<scalar_t>(),
                            grad_input.mutable_data_ptr<scalar_t>(), grad_kernel.mutable_data_ptr<scalar_t>(),
                            row_val_ptr, row_dj_ptr, use_separable, N, C, H, W, kH, kW, kernel_channel_stride,
                            border_mode, need_kernel_grad, stream);
                    }),
                    torch::headeronly::ScalarType::Float, torch::headeronly::ScalarType::Double,
                    torch::headeronly::ScalarType::Half, torch::headeronly::ScalarType::BFloat16);

    return {grad_input, grad_kernel};
}

} // namespace

std::tuple<Tensor, Tensor> erode_backward(const Tensor& grad_output, const Tensor& input, const Tensor& kernel,
                                          const int64_t border, const std::optional<bool>& flat,
                                          const bool need_kernel_grad) {
    return morphology_backward_impl(grad_output, input, kernel, border, flat, need_kernel_grad, MorphOp::kErode,
                                    "serron::erode_backward");
}

std::tuple<Tensor, Tensor> dilate_backward(const Tensor& grad_output, const Tensor& input, const Tensor& kernel,
                                           const int64_t border, const std::optional<bool>& flat,
                                           const bool need_kernel_grad) {
    return morphology_backward_impl(grad_output, input, kernel, border, flat, need_kernel_grad, MorphOp::kDilate,
                                    "serron::dilate_backward");
}

} // namespace serron
