#include "ops.h"

#include <cuda/morphology/algorithm.h>
#include <cuda/morphology/arg_tap.h>
#include <cuda/morphology/entry_points.h>
#include <cuda/morphology/enums.h>
#include <cuda/morphology/kernel_tables.h>
#include <cuda/morphology/separable.h>
#include <cuda/morphology/thresholds.h>
#include <cuda/utils/border_mode.h>
#include <cuda/utils/declarations.h>
#include <cuda/utils/driver.h>
#include <cuda/utils/launch.h>
#include <cuda/utils/smem.h>

#include <cuda.h>

#include <backward_fatbin.h>
#include <compat/acc_type.h>
#include <compat/check.h>
#include <morph_algorithm.h>
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
CUlibrary backward_library() {
    static const CUlibrary library = load_kernel_library(serron_backward_fatbin, "backward");
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
            const CUlibrary library = backward_library();                                                              \
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
                                          BorderMode border, bool need_kernel_grad, CUstream stream) {
    constexpr size_t kTap = backward_line_elem_size<scalar_t>();

    const Dim3 block(LINE_TILE);
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
    const Dim3 grid_row(static_cast<unsigned int>((W + out_row - 1) / out_row), static_cast<unsigned int>(H),
                        static_cast<unsigned int>(N * C));
    launch_kernel(kernels.row_argreduce, grid_row, block, smem_row, stream, input, row_best_val, row_best_dj, H, W, kW,
                  chunks_row, border);

    const int64_t out_col = chunks_col * kH - kH + 1;
    const Dim3 grid_col(static_cast<unsigned int>((H + out_col - 1) / out_col), static_cast<unsigned int>(W),
                        static_cast<unsigned int>(N * C));
    launch_kernel(kernels.col_argreduce, grid_col, block, smem_col, stream, row_best_val, row_best_dj, grad_output,
                  grad_input, grad_kernel, C, H, W, kH, kW, chunks_col, kernel_channel_stride, border,
                  need_kernel_grad);
    return true;
}

/**
 * Launch the morphology backward pass on @p stream with @p algorithm, which must already be resolved (see @ref
 * resolve_algorithm) and cannot be @ref Algorithm::kAuto. Separable runs the row+column argreduce; tiled and
 * element-wise recompute the winning tap directly, as @ref morphology_backward_kernel.
 *
 * @param may_fall_back  Whether a launch whose shared-memory setup the driver rejects may fall back to the next
 * algorithm down. Only the library's own choice may; a forced algorithm that cannot launch throws.
 */
template <typename scalar_t>
void launch_morphology_backward(const BackwardKernels<scalar_t>& kernels, const scalar_t* grad_output,
                                const scalar_t* input, const scalar_t* kernel, scalar_t* grad_input,
                                scalar_t* grad_kernel, scalar_t* row_best_val, int32_t* row_best_dj,
                                Algorithm algorithm, bool may_fall_back, int64_t N, int64_t C, int64_t H, int64_t W,
                                int64_t kH, int64_t kW, int64_t kernel_channel_stride, BorderMode border,
                                bool need_kernel_grad, CUstream stream) {
    const size_t budget = smem_budget();

    if (algorithm == Algorithm::kSeparable) {
        if (launch_morphology_backward_separable<scalar_t>(kernels, input, grad_output, grad_input, grad_kernel,
                                                           row_best_val, row_best_dj, N, C, H, W, kH, kW,
                                                           kernel_channel_stride, border, need_kernel_grad, stream)) {
            return;
        }
        SERRON_CHECK(may_fall_back, "serron: could not configure shared memory for the separable kernels");
        algorithm = tiled_feasible(kH, kW, sizeof(scalar_t), budget) ? Algorithm::kTiled : Algorithm::kElement;
    }

    if (algorithm == Algorithm::kTiled) {
        const size_t smem = tiled_smem_bytes(kH, kW, sizeof(scalar_t));
        if (configure_kernel_smem(kernels.tiled, smem)) {
            constexpr Dim3 block(TILE_X, TILE_Y);
            const Dim3 grid(static_cast<unsigned int>((W + TILE_X - 1) / TILE_X),
                            static_cast<unsigned int>((H + TILE_Y - 1) / TILE_Y), static_cast<unsigned int>(N * C));
            launch_kernel(kernels.tiled, grid, block, smem, stream, grad_output, input, kernel, grad_input, grad_kernel,
                          C, H, W, kH, kW, kernel_channel_stride, border, need_kernel_grad);
            return;
        }
        SERRON_CHECK(may_fall_back, "serron: could not configure shared memory for the tiled kernels");
    }

    const int64_t total = N * C * H * W;
    const auto blocks = static_cast<unsigned int>((total + THREADS - 1) / THREADS);
    launch_kernel(kernels.element, blocks, THREADS, 0, stream, grad_output, input, kernel, grad_input, grad_kernel, N,
                  C, H, W, kH, kW, kernel_channel_stride, border, need_kernel_grad);
}

/**
 * Shared host-side backward behind @ref erode_backward / @ref dilate_backward
 *
 * @param grad_output  Upstream gradient, CUDA tensor of shape (N, C, H, W), same dtype as @p input.
 * @param input        Forward input, CUDA tensor of shape (N, C, H, W), floating dtype.
 * @param kernel       Forward structuring element, CUDA tensor of shape (kH, kW) or (C, kH, kW); same dtype as @p
 * input.
 * @param border       Boundary mode (@ref BorderMode encoding) used in the forward pass.
 * @param algo         Algorithm to run (@ref Algorithm encoding); auto lets the library choose.
 * @param op           Operation to differentiate (@ref MorphOp).
 * @param name         Qualified caller name used to prefix diagnostics.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 * @throws std::runtime_error  if the tensors are not on CUDA, have the wrong rank or dtype, the channel counts
 * disagree, or @p border is out of range.
 */
std::tuple<Tensor, Tensor> morphology_backward_impl(const Tensor& grad_output, const Tensor& input,
                                                    const Tensor& kernel, int64_t border,
                                                    const std::optional<bool>& flat, bool need_kernel_grad,
                                                    int64_t algo, MorphOp op, const char* name) {
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
    const Algorithm requested = parse_algorithm(algo, name);

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

    // A forced tiled or element-wise launch never reads the flatness, so it skips the read-back from the device.
    const bool is_flat = algorithm_needs_flatness(requested) && resolve_flat(flat, kernel_c);

    const torch::stable::accelerator::DeviceGuard device_guard(input_c.get_device_index());
    const CurrentContextGuard context_guard(input_c.get_device_index());
    const auto stream =
        static_cast<CUstream>(torch::stable::accelerator::getCurrentStream(input_c.get_device_index()).nativeHandle());
    const auto border_mode = static_cast<BorderMode>(border);

    THO_DISPATCH_V2(input_c.scalar_type(), "serron_morphology_backward", AT_WRAP([&] {
                        const AlgorithmProblem problem = make_problem<scalar_t>(
                            kH, kW, is_flat, smem_budget(),
                            separable_min_k(input_c.get_device_index(), input_c.scalar_type(), true), true);
                        const Algorithm algorithm = resolve_algorithm(requested, problem, name);
                        const bool separable = algorithm == Algorithm::kSeparable;
                        Tensor row_best_val;
                        Tensor row_best_dj;
                        if (separable) {
                            row_best_val = torch::stable::empty_like(input_c);
                            row_best_dj =
                                torch::stable::new_empty(input_c, input_c.sizes(), torch::headeronly::ScalarType::Int);
                        }
                        scalar_t* row_val_ptr = separable ? row_best_val.mutable_data_ptr<scalar_t>() : nullptr;
                        int32_t* row_dj_ptr = separable ? row_best_dj.mutable_data_ptr<int32_t>() : nullptr;
                        launch_morphology_backward<scalar_t>(
                            backward_kernels<scalar_t>(op), grad_output_c.const_data_ptr<scalar_t>(),
                            input_c.const_data_ptr<scalar_t>(), kernel_c.const_data_ptr<scalar_t>(),
                            grad_input.mutable_data_ptr<scalar_t>(), grad_kernel.mutable_data_ptr<scalar_t>(),
                            row_val_ptr, row_dj_ptr, algorithm, requested == Algorithm::kAuto, N, C, H, W, kH, kW,
                            kernel_channel_stride, border_mode, need_kernel_grad, stream);
                    }),
                    torch::headeronly::ScalarType::Float, torch::headeronly::ScalarType::Double,
                    torch::headeronly::ScalarType::Half, torch::headeronly::ScalarType::BFloat16);

    return {grad_input, grad_kernel};
}

} // namespace

std::tuple<Tensor, Tensor> erode_backward(const Tensor& grad_output, const Tensor& input, const Tensor& kernel,
                                          const int64_t border, const std::optional<bool>& flat,
                                          const bool need_kernel_grad, const int64_t algo) {
    return morphology_backward_impl(grad_output, input, kernel, border, flat, need_kernel_grad, algo, MorphOp::kErode,
                                    "serron::erode_backward");
}

std::tuple<Tensor, Tensor> dilate_backward(const Tensor& grad_output, const Tensor& input, const Tensor& kernel,
                                           const int64_t border, const std::optional<bool>& flat,
                                           const bool need_kernel_grad, const int64_t algo) {
    return morphology_backward_impl(grad_output, input, kernel, border, flat, need_kernel_grad, algo, MorphOp::kDilate,
                                    "serron::dilate_backward");
}

} // namespace serron
