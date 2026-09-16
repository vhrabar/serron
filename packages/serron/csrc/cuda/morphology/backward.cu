#include "ops.h"

#include <cuda/morphology/enums.cuh>
#include <cuda/morphology/ops_policy.cuh>
#include <cuda/morphology/separable.cuh>
#include <cuda/utils/boundaries.cuh>
#include <cuda/utils/declarations.cuh>

#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <tuple>

namespace serron {

namespace {

/**
 * Grayscale morphology backward kernel, one thread per output element.
 *
 *
 * @tparam scalar_t              Element type of the tensors; the reduction accumulates in at::acc_type<scalar_t>.
 * @tparam Op                    Operation policy (@ref ErodeOp or @ref DilateOp).
 * @param grad_output            Upstream gradient, contiguous (N, C, H, W).
 * @param input                  Forward input, contiguous (N, C, H, W).
 * @param kernel                 Forward structuring element, contiguous (kH, kW) or (C, kH, kW).
 * @param grad_input             Gradient w.r.t. @p input, pre-zeroed (N, C, H, W); scattered into.
 * @param grad_kernel            Gradient w.r.t. @p kernel, pre-zeroed, same shape as @p kernel; scattered into.
 * @param N                      Batch size.
 * @param C                      Channel count.
 * @param H                      Input/output height.
 * @param W                      Input/output width.
 * @param kH                     Structuring-element height.
 * @param kW                     Structuring-element width.
 * @param kernel_channel_stride  Per-channel stride into @p kernel / @p grad_kernel (kH*kW), or 0 for a shared SE.
 * @param border                 Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename Op>
__global__ void morphology_backward_kernel(const scalar_t* __restrict__ grad_output, const scalar_t* __restrict__ input,
                                           const scalar_t* __restrict__ kernel, scalar_t* __restrict__ grad_input,
                                           scalar_t* __restrict__ grad_kernel, const int64_t N, const int64_t C,
                                           const int64_t H, const int64_t W, const int64_t kH, const int64_t kW,
                                           const int64_t kernel_channel_stride, const BorderMode border) {
    using acc_t = at::acc_type<scalar_t, true>;

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W)
        return;

    const int64_t w = idx % W;
    const int64_t h = (idx / W) % H;
    const int64_t c = (idx / (W * H)) % C;
    const int64_t n = idx / (W * H * C);

    const int64_t anchor_h = kH / 2;
    const int64_t anchor_w = kW / 2;

    const scalar_t* input_nc = input + (n * C + c) * H * W;
    const scalar_t* kernel_c = kernel + c * kernel_channel_stride;
    scalar_t* grad_input_nc = grad_input + (n * C + c) * H * W;
    scalar_t* grad_kernel_c = grad_kernel + c * kernel_channel_stride;

    const auto neutral = Op::template neutral<acc_t>();

    // Recompute the winning tap: identical tap / reduce / tie-break as the FWD
    acc_t best = neutral;
    int64_t best_k = -1;  // di*kW + dj -> SE cell
    int64_t best_in = -1; // ih*W + iw  -> border-resolved input pixel
    for (int64_t di = 0; di < kH; ++di) {
        int64_t ih = h + di - anchor_h;
        const bool ih_valid = resolve_coord(ih, H, border);
        for (int64_t dj = 0; dj < kW; ++dj) {
            acc_t val = neutral;
            int64_t in_off = -1;
            if (ih_valid) {
                if (int64_t iw = w + dj - anchor_w; resolve_coord(iw, W, border)) {
                    in_off = ih * W + iw;
                    val = Op::tap(static_cast<acc_t>(input_nc[in_off]), static_cast<acc_t>(kernel_c[di * kW + dj]));
                }
                // OoB -> neutral
            }
            const acc_t merged = Op::reduce(best, val);
            if (merged != best) {
                best = merged;
                best_k = di * kW + dj;
                best_in = in_off;
            }
        }
    }

    // The centre tap (di=anchor_h, dj=anchor_w) always resolves in-image, so a winner is guaranteed.
    const acc_t go = static_cast<acc_t>(grad_output[idx]);
    gpuAtomicAdd(&grad_input_nc[best_in], static_cast<scalar_t>(go));
    gpuAtomicAdd(&grad_kernel_c[best_k], static_cast<scalar_t>(Op::template se_grad_sign<acc_t>() * go));
}

/**
 * Row pass of the separable backward: for each (line = real input row @c h, output
 * column @c w), finds the winning tap over @c dj in [0, kW).
 *
 * @tparam scalar_t     Element type; the reduction accumulates in at::acc_type<scalar_t>.
 * @tparam Op           Operation policy (@ref ErodeOp or @ref DilateOp).
 * @param input         Forward input, contiguous (N, C, H, W).
 * @param row_best_val  Row-champion value per (n, c, h, w), contiguous (N, C, H, W).
 * @param row_best_dj   Row-champion @c dj offset per (n, c, h, w), contiguous (N, C, H, W).
 * @param NC            N * C.
 * @param H             Input height.
 * @param W             Input/output width.
 * @param kW            Structuring-element width.
 * @param border        Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename Op>
__global__ void morphology_row_argreduce_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ row_best_val,
                                                int32_t* __restrict__ row_best_dj, const int64_t NC, const int64_t H,
                                                const int64_t W, const int64_t kW, const BorderMode border) {
    using acc_t = at::acc_type<scalar_t, true>;

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= NC * H * W)
        return;

    const int64_t w = idx % W;
    const int64_t nc_h = idx / W; // == nc * H + h, since input is contiguous (..., H, W)
    const int64_t anchor_w = kW / 2;

    const scalar_t* input_line = input + nc_h * W;

    const auto neutral = Op::template neutral<acc_t>();
    acc_t best = neutral;
    int32_t best_dj = 0;
    for (int64_t dj = 0; dj < kW; ++dj) {
        if (int64_t iw = w + dj - anchor_w; resolve_coord(iw, W, border)) {
            const acc_t val = static_cast<acc_t>(input_line[iw]); // flat SE -> tap(sample, 0) == sample
            if (const acc_t merged = Op::reduce(best, val); merged != best) {
                best = merged;
                best_dj = static_cast<int32_t>(dj);
            }
        }
    }
    // The centre tap (dj=anchor_w) always resolves in-image
    row_best_val[idx] = static_cast<scalar_t>(best);
    row_best_dj[idx] = best_dj;
}

/**
 * Column pass of the separable backward: combines each output pixel's row champions
 * over @c di in [0, kH), with the same strict-improvement tie-break, then scatters the
 * upstream gradient.
 *
 * @tparam scalar_t              Element type; the reduction accumulates in at::acc_type<scalar_t>.
 * @tparam Op                    Operation policy (@ref ErodeOp or @ref DilateOp).
 * @param row_best_val           Row-pass output, contiguous (N, C, H, W).
 * @param row_best_dj            Row-pass output, contiguous (N, C, H, W).
 * @param grad_output            Upstream gradient, contiguous (N, C, H, W).
 * @param grad_input             Gradient w.r.t. the forward input, pre-zeroed (N, C, H, W); scattered into.
 * @param grad_kernel            Gradient w.r.t. the forward SE, pre-zeroed, (kH, kW) or (C, kH, kW); scattered into.
 * @param N                      Batch size.
 * @param C                      Channel count.
 * @param H                      Input/output height.
 * @param W                      Input/output width.
 * @param kH                     Structuring-element height.
 * @param kW                     Structuring-element width.
 * @param kernel_channel_stride  Per-channel stride into @p grad_kernel (kH*kW), or 0 for a shared SE.
 * @param border                 Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename Op>
__global__ void
morphology_col_argreduce_kernel(const scalar_t* __restrict__ row_best_val, const int32_t* __restrict__ row_best_dj,
                                const scalar_t* __restrict__ grad_output, scalar_t* __restrict__ grad_input,
                                scalar_t* __restrict__ grad_kernel, const int64_t N, const int64_t C, const int64_t H,
                                const int64_t W, const int64_t kH, const int64_t kW,
                                const int64_t kernel_channel_stride, const BorderMode border) {
    using acc_t = at::acc_type<scalar_t, true>;

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W)
        return;

    const int64_t w = idx % W;
    const int64_t h = (idx / W) % H;
    const int64_t c = (idx / (W * H)) % C;
    const int64_t nc = idx / (W * H);

    const int64_t anchor_h = kH / 2;
    const int64_t anchor_w = kW / 2;

    const scalar_t* row_val_nc = row_best_val + nc * H * W;
    const int32_t* row_dj_nc = row_best_dj + nc * H * W;
    scalar_t* grad_input_nc = grad_input + nc * H * W;
    scalar_t* grad_kernel_c = grad_kernel + c * kernel_channel_stride;

    const auto neutral = Op::template neutral<acc_t>();
    acc_t best = neutral;
    int64_t best_di = 0;
    int64_t best_ih = -1;
    for (int64_t di = 0; di < kH; ++di) {
        if (int64_t ih = h + di - anchor_h; resolve_coord(ih, H, border)) {
            const acc_t val = static_cast<acc_t>(row_val_nc[ih * W + w]);
            if (const acc_t merged = Op::reduce(best, val); merged != best) {
                best = merged;
                best_di = di;
                best_ih = ih;
            }
        }
    }

    // The centre tap (di=anchor_h, dj=anchor_w) always resolves in-image, so best_ih is set.
    const int32_t dj = row_dj_nc[best_ih * W + w];
    int64_t iw = w + dj - anchor_w;
    resolve_coord(iw, W, border);
    const int64_t best_k = best_di * kW + dj;

    const acc_t go = static_cast<acc_t>(grad_output[idx]);
    gpuAtomicAdd(&grad_input_nc[best_ih * W + iw], static_cast<scalar_t>(go));
    gpuAtomicAdd(&grad_kernel_c[best_k], static_cast<scalar_t>(Op::template se_grad_sign<acc_t>() * go));
}

/**
 * Launches the separable backward: the row pass into @p row_best_val / @p row_best_dj,
 * then the column pass scattering into @p grad_input / @p grad_kernel. O(kH+kW) per
 * output element versus @ref morphology_backward_kernel's O(kH*kW).
 */
template <typename scalar_t, typename Op>
void launch_morphology_backward_separable(const scalar_t* input, const scalar_t* grad_output, scalar_t* grad_input,
                                          scalar_t* grad_kernel, scalar_t* row_best_val, int32_t* row_best_dj,
                                          int64_t N, int64_t C, int64_t H, int64_t W, int64_t kH, int64_t kW,
                                          int64_t kernel_channel_stride, BorderMode border, cudaStream_t stream) {
    const int64_t total = N * C * H * W;
    const auto blocks = static_cast<unsigned int>((total + THREADS - 1) / THREADS);

    morphology_row_argreduce_kernel<scalar_t, Op>
        <<<blocks, THREADS, 0, stream>>>(input, row_best_val, row_best_dj, N * C, H, W, kW, border);
    morphology_col_argreduce_kernel<scalar_t, Op>
        <<<blocks, THREADS, 0, stream>>>(row_best_val, row_best_dj, grad_output, grad_input, grad_kernel, N, C, H, W,
                                         kH, kW, kernel_channel_stride, border);
}

/**
 * Launch the morphology backward pass on @p stream. Uses the separable row+column
 * argreduce when @p use_separable was set by the caller (flat, axis-separable SE
 * at/above @ref separable_min_k); otherwise recomputes the winning tap directly, as
 * @ref morphology_backward_kernel.
 */
template <typename scalar_t, typename Op>
void launch_morphology_backward(const scalar_t* grad_output, const scalar_t* input, const scalar_t* kernel,
                                scalar_t* grad_input, scalar_t* grad_kernel, scalar_t* row_best_val,
                                int32_t* row_best_dj, bool use_separable, int64_t N, int64_t C, int64_t H, int64_t W,
                                int64_t kH, int64_t kW, int64_t kernel_channel_stride, BorderMode border,
                                cudaStream_t stream) {
    if (use_separable) {
        launch_morphology_backward_separable<scalar_t, Op>(input, grad_output, grad_input, grad_kernel, row_best_val,
                                                           row_best_dj, N, C, H, W, kH, kW, kernel_channel_stride,
                                                           border, stream);
        return;
    }

    const int64_t total = N * C * H * W;
    const auto blocks = static_cast<unsigned int>((total + THREADS - 1) / THREADS);
    morphology_backward_kernel<scalar_t, Op><<<blocks, THREADS, 0, stream>>>(
        grad_output, input, kernel, grad_input, grad_kernel, N, C, H, W, kH, kW, kernel_channel_stride, border);
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
 * @throws c10::Error  if the tensors are not on CUDA, have the wrong rank or dtype, the channel counts disagree, or @p
 * border is out of range.
 */
std::tuple<at::Tensor, at::Tensor> morphology_backward_impl(const at::Tensor& grad_output, const at::Tensor& input,
                                                            const at::Tensor& kernel, int64_t border, MorphOp op,
                                                            const char* name) {
    TORCH_CHECK(grad_output.is_cuda(), name, ": grad_output must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), name, ": input must be a CUDA tensor");
    TORCH_CHECK(kernel.is_cuda(), name, ": kernel must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, name, ": input must be 4-D (N, C, H, W), got ", input.dim(), "-D");
    TORCH_CHECK(grad_output.dim() == 4, name, ": grad_output must be 4-D (N, C, H, W), got ", grad_output.dim(), "-D");
    TORCH_CHECK(kernel.dim() == 2 || kernel.dim() == 3, name, ": kernel must be 2-D (kH, kW) or 3-D (C, kH, kW), got ",
                kernel.dim(), "-D");
    TORCH_CHECK(input.scalar_type() == kernel.scalar_type(), name, ": input and kernel must share a dtype");
    TORCH_CHECK(grad_output.scalar_type() == input.scalar_type(), name, ": grad_output and input must share a dtype");
    TORCH_CHECK(border >= kReflect && border <= kConstant, name, ": invalid border mode ", border);

    const at::Tensor grad_output_c = grad_output.contiguous();
    const at::Tensor input_c = input.contiguous();
    const at::Tensor kernel_c = kernel.contiguous();

    const int64_t N = input_c.size(0);
    const int64_t C = input_c.size(1);
    const int64_t H = input_c.size(2);
    const int64_t W = input_c.size(3);

    int64_t kH = 0;
    int64_t kW = 0;
    int64_t kernel_channel_stride = 0;
    if (kernel_c.dim() == 3) {
        TORCH_CHECK(kernel_c.size(0) == C, name, ": kernel channel dim (", kernel_c.size(0),
                    ") must match input channels (", C, ")");
        kH = kernel_c.size(1);
        kW = kernel_c.size(2);
        kernel_channel_stride = kH * kW;
    } else {
        kH = kernel_c.size(0);
        kW = kernel_c.size(1);
        kernel_channel_stride = 0;
    }
    TORCH_CHECK(kH > 0 && kW > 0, name, ": kernel spatial dims must be positive");
    TORCH_CHECK(grad_output_c.sizes() == input_c.sizes(), name, ": grad_output shape must match input");

    at::Tensor grad_input = at::zeros_like(input_c);
    at::Tensor grad_kernel = at::zeros_like(kernel_c);
    if (input_c.numel() == 0)
        return {grad_input, grad_kernel};

    const bool use_separable = use_separable_path(kernel_c, kH, kW);
    at::Tensor row_best_val;
    at::Tensor row_best_dj;
    if (use_separable) {
        row_best_val = at::empty_like(input_c);
        row_best_dj = at::empty(input_c.sizes(), input_c.options().dtype(at::kInt));
    }

    const c10::cuda::CUDAGuard device_guard(input_c.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto border_mode = static_cast<BorderMode>(border);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input_c.scalar_type(), "serron_morphology_backward", [&] {
            const scalar_t* grad_output_ptr = grad_output_c.data_ptr<scalar_t>();
            const scalar_t* input_ptr = input_c.data_ptr<scalar_t>();
            const scalar_t* kernel_ptr = kernel_c.data_ptr<scalar_t>();
            auto* grad_input_ptr = grad_input.data_ptr<scalar_t>();
            auto* grad_kernel_ptr = grad_kernel.data_ptr<scalar_t>();
            scalar_t* row_val_ptr = use_separable ? row_best_val.data_ptr<scalar_t>() : nullptr;
            int32_t* row_dj_ptr = use_separable ? row_best_dj.data_ptr<int32_t>() : nullptr;
            switch (op) {
            case MorphOp::kErode:
                launch_morphology_backward<scalar_t, ErodeOp>(
                    grad_output_ptr, input_ptr, kernel_ptr, grad_input_ptr, grad_kernel_ptr, row_val_ptr, row_dj_ptr,
                    use_separable, N, C, H, W, kH, kW, kernel_channel_stride, border_mode, stream);
                break;
            case MorphOp::kDilate:
                launch_morphology_backward<scalar_t, DilateOp>(
                    grad_output_ptr, input_ptr, kernel_ptr, grad_input_ptr, grad_kernel_ptr, row_val_ptr, row_dj_ptr,
                    use_separable, N, C, H, W, kH, kW, kernel_channel_stride, border_mode, stream);
                break;
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_input, grad_kernel};
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
