#include "ops.h"

// Generated from forward_kernels.cu by cmake/embed_fatbin.cmake.
#include <cuda/morphology/entry_points.h>
#include <cuda/morphology/enums.h>
#include <cuda/morphology/kernel_tables.h>
#include <cuda/morphology/separable.h>
#include <cuda/utils/border_mode.h>
#include <cuda/utils/declarations.h>
#include <cuda/utils/launch.h>
#include <cuda/utils/smem.h>

#include <cuda_runtime.h>

#include <compat/check.h>
#include <forward_fatbin.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>
#include <array>
#include <cstdint>

namespace serron {

namespace {

/// The forward kernels' fatbin, embedded at build time and loaded on first use.
cudaLibrary_t forward_library() {
    static const cudaLibrary_t library = load_kernel_library(serron_forward_fatbin, "forward");
    return library;
}

/// Forward entry points for @p op on @c scalar_t, looked up by name on first use; specialised per dtype below.
template <typename scalar_t>
const ForwardKernels<scalar_t>& forward_kernels(MorphOp op);

#define SERRON_FORWARD_TABLE_ROW(Op, op, scalar_t, suffix)                                                             \
    ForwardKernels<scalar_t>{SERRON_FORWARD_KERNELS(SERRON_GET_ENTRY_POINT, op, scalar_t, suffix)},
#define SERRON_FORWARD_TABLE(scalar_t, suffix)                                                                         \
    template <>                                                                                                        \
    const ForwardKernels<scalar_t>& forward_kernels<scalar_t>(const MorphOp op) {                                      \
        static const std::array<ForwardKernels<scalar_t>, 2> table = [] {                                              \
            const cudaLibrary_t library = forward_library();                                                           \
            return std::array<ForwardKernels<scalar_t>, 2>{                                                            \
                SERRON_CUDA_FOR_EACH_OP(SERRON_FORWARD_TABLE_ROW, scalar_t, suffix)};                                  \
        }();                                                                                                           \
        return table[static_cast<int>(op)];                                                                            \
    }
SERRON_CUDA_FOR_EACH_DTYPE(SERRON_FORWARD_TABLE)

/**
 * Chains the row pass (@p input -> @p scratch) into the column pass (@p scratch ->
 * @p output). Requires the structuring element to be flat and axis-separable
 *
 * @param kernels  Entry points for the (op, dtype) being launched.
 * @param input    Input image, contiguous (N, C, H, W).
 * @param scratch  Intermediate buffer, same shape/dtype as @p input; holds the row-pass result.
 * @param output   Output image, contiguous (N, C, H, W).
 * @param N        Batch size.
 * @param C        Channel count.
 * @param H        Input/output height.
 * @param W        Input/output width.
 * @param kH       Structuring-element height (window length for the column pass).
 * @param kW       Structuring-element width (window length for the row pass).
 * @param border   Boundary mode (@ref BorderMode) for out-of-image reads.
 * @param stream   CUDA stream both passes are launched on.
 */
template <typename scalar_t>
bool launch_morphology_separable(const ForwardKernels<scalar_t>& kernels, const scalar_t* input, scalar_t* scratch,
                                 scalar_t* output, int64_t N, int64_t C, int64_t H, int64_t W, int64_t kH, int64_t kW,
                                 BorderMode border, cudaStream_t stream) {
    const dim3 block(LINE_TILE);
    const size_t budget = smem_budget();

    const int64_t chunks_row = line_chunks_per_block(kW, W, sizeof(scalar_t), budget);
    const int64_t chunks_col = line_chunks_per_block(kH, H, sizeof(scalar_t), budget);
    const size_t smem_row = line_smem_bytes(chunks_row, kW, sizeof(scalar_t));
    const size_t smem_col = line_smem_bytes(chunks_col, kH, sizeof(scalar_t));

    if (!configure_kernel_smem(kernels.line_row, smem_row) || !configure_kernel_smem(kernels.line_col, smem_col)) {
        return false;
    }

    const int64_t out_row = chunks_row * kW - kW + 1;
    const dim3 grid_row(static_cast<unsigned int>((W + out_row - 1) / out_row), static_cast<unsigned int>(H),
                        static_cast<unsigned int>(N * C));
    launch_kernel(kernels.line_row, grid_row, block, smem_row, stream, input, scratch, N, C, H, W, kW, chunks_row,
                  border);

    const int64_t out_col = chunks_col * kH - kH + 1;
    const dim3 grid_col(static_cast<unsigned int>((H + out_col - 1) / out_col), static_cast<unsigned int>(W),
                        static_cast<unsigned int>(N * C));
    launch_kernel(kernels.line_col, grid_col, block, smem_col, stream, scratch, output, N, C, H, W, kH, chunks_col,
                  border);
    return true;
}

/**
 * Launch the morphology forward pass on @p stream.
 *
 */
template <typename scalar_t>
void launch_morphology(const ForwardKernels<scalar_t>& kernels, const scalar_t* input, const scalar_t* kernel,
                       scalar_t* output, scalar_t* scratch, bool use_separable, int64_t N, int64_t C, int64_t H,
                       int64_t W, int64_t kH, int64_t kW, int64_t kernel_channel_stride, BorderMode border,
                       cudaStream_t stream) {
    const size_t budget = smem_budget();

    if (use_separable && line_smem_bytes(1, std::max(kH, kW), sizeof(scalar_t)) <= budget &&
        launch_morphology_separable<scalar_t>(kernels, input, scratch, output, N, C, H, W, kH, kW, border, stream)) {
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
        launch_kernel(kernels.tiled, grid, block, smem, stream, input, kernel, output, C, H, W, kH, kW,
                      kernel_channel_stride, border);
    } else {
        const int64_t total = N * C * H * W;
        const auto blocks = static_cast<unsigned int>((total + THREADS - 1) / THREADS);
        launch_kernel(kernels.element, blocks, THREADS, 0, stream, input, kernel, output, N, C, H, W, kH, kW,
                      kernel_channel_stride, border);
    }
}

/**
 * Shared host-side implementation behind @ref erode and @ref dilate: validates the inputs, normalises the
 * structuring-element layout, then dispatches on dtype and @p op to @ref launch_morphology.
 *
 * @param input         CUDA tensor of shape (N, C, H, W), floating dtype.
 * @param kernel        CUDA structuring element of shape (kH, kW), shared across channels, or (C, kH, kW) for a
 * per-channel element; same dtype as @p input.
 * @param border        Boundary mode (@ref BorderMode encoding) applied at the image edges.
 * @param op            Operation to apply (@ref MorphOp).
 * @param name          Qualified caller name ("serron::erode") used to prefix diagnostics.
 * @return              Result tensor, same shape and dtype as @p input.
 * @throws std::runtime_error   if the tensors are not on CUDA, have the wrong rank or dtype, the channel counts
 * disagree, or @p border is out of range.
 */
Tensor morphology_impl(const Tensor& input, const Tensor& kernel, int64_t border, const std::optional<bool>& flat,
                       MorphOp op, const char* name) {
    SERRON_CHECK(input.is_cuda(), name, ": input must be a CUDA tensor");
    SERRON_CHECK(kernel.is_cuda(), name, ": kernel must be a CUDA tensor");
    SERRON_CHECK(input.dim() == 4, name, ": input must be 4-D (N, C, H, W), got ", input.dim(), "-D");
    SERRON_CHECK(kernel.dim() == 2 || kernel.dim() == 3, name, ": kernel must be 2-D (kH, kW) or 3-D (C, kH, kW), got ",
                 kernel.dim(), "-D");
    SERRON_CHECK(input.scalar_type() == kernel.scalar_type(), name, ": input and kernel must share a dtype");
    SERRON_CHECK(border >= kReflect && border <= kConstant, name, ": invalid border mode ", border);

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

    Tensor output = torch::stable::empty_like(input_c);
    if (output.numel() == 0)
        return output;

    const bool use_separable = use_separable_path(resolve_flat(flat, kernel_c), kH, kW);
    Tensor scratch;
    if (use_separable) {
        scratch = torch::stable::empty_like(input_c);
    }

    const torch::stable::accelerator::DeviceGuard device_guard(input_c.get_device_index());
    const auto stream = static_cast<cudaStream_t>(
        torch::stable::accelerator::getCurrentStream(input_c.get_device_index()).nativeHandle());
    const auto border_mode = static_cast<BorderMode>(border);

    THO_DISPATCH_V2(input_c.scalar_type(), "serron_morphology", AT_WRAP([&] {
                        scalar_t* scratch_ptr = use_separable ? scratch.mutable_data_ptr<scalar_t>() : nullptr;
                        launch_morphology<scalar_t>(forward_kernels<scalar_t>(op), input_c.const_data_ptr<scalar_t>(),
                                                    kernel_c.const_data_ptr<scalar_t>(),
                                                    output.mutable_data_ptr<scalar_t>(), scratch_ptr, use_separable, N,
                                                    C, H, W, kH, kW, kernel_channel_stride, border_mode, stream);
                    }),
                    torch::headeronly::ScalarType::Float, torch::headeronly::ScalarType::Double,
                    torch::headeronly::ScalarType::Half, torch::headeronly::ScalarType::BFloat16);

    return output;
}

} // namespace

/**
 * Grayscale erosion (sliding minimum) of @p input by structuring element @p kernel.
 *
 * @param input         CUDA tensor of shape (N, C, H, W), floating dtype.
 * @param kernel        CUDA structuring element of shape (kH, kW), shared across channels, or (C, kH, kW) for a
 * per-channel element; same dtype as @p input.
 * @param border        Boundary mode (@ref BorderMode) applied at the image edges.
 * @return              Eroded tensor, same shape and dtype as @p input.
 * @throws std::runtime_error   if the tensors are not on CUDA, have the wrong rank or dtype, the channel counts
 * disagree, or @p border is out of range.
 */
Tensor erode(const Tensor& input, const Tensor& kernel, const int64_t border, const std::optional<bool>& flat) {
    return morphology_impl(input, kernel, border, flat, MorphOp::kErode, "serron::erode");
}

/**
 * Grayscale dilation (sliding maximum) of @p input by structuring element @p kernel.
 *
 * @param input         CUDA tensor of shape (N, C, H, W), floating dtype.
 * @param kernel        CUDA structuring element of shape (kH, kW), shared across channels, or (C, kH, kW) for a
 * per-channel element; same dtype as @p input.
 * @param border        Boundary mode (@ref BorderMode) applied at the image edges.
 * @return              Dilated tensor, same shape and dtype as @p input.
 * @throws std::runtime_error   if the tensors are not on CUDA, have the wrong rank or dtype, the channel counts
 * disagree, or @p border is out of range.
 */
Tensor dilate(const Tensor& input, const Tensor& kernel, const int64_t border, const std::optional<bool>& flat) {
    return morphology_impl(input, kernel, border, flat, MorphOp::kDilate, "serron::dilate");
}

} // namespace serron
