#include "ops.h"

#include <cuda/utils/boundaries.cuh>
#include <cuda/utils/declarations.cuh>

#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cmath>

namespace serron {

namespace {

/**
 * Grayscale erosion kernel, one thread per output element.
 *
 * Computes out(n,c,i,j) = min_{(di,dj)} [ in(n,c,i+di-aH,j+dj-aW) - se(c,di,dj) ],
 * where the anchor (aH,aW) is the kernel centre and out-of-image samples are
 * resolved by @p border
 *
 * @tparam scalar_t              Element type of @p input / @p kernel / @p output; the reduction accumulates in
 * at::acc_type<scalar_t>.
 * @param input                  Input image, contiguous (N, C, H, W).
 * @param kernel                 Structuring element, contiguous (kH, kW) or (C, kH, kW).
 * @param output                 Output image, contiguous (N, C, H, W); written in full.
 * @param N                      Batch size.
 * @param C                      Channel count.
 * @param H                      Input/output height.
 * @param W                      Input/output width.
 * @param kH                     Structuring-element height.
 * @param kW                     Structuring-element width.
 * @param kernel_channel_stride  Per-channel stride into @p kernel (kH*kW), or 0 when the structuring element is shared
 * across channels.
 * @param border                 Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t>
__global__ void erode_kernel(const scalar_t* __restrict__ input, const scalar_t* __restrict__ kernel,
                             scalar_t* __restrict__ output, const int64_t N, const int64_t C, const int64_t H,
                             const int64_t W, const int64_t kH, const int64_t kW, const int64_t kernel_channel_stride,
                             const int border) {
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

    // Neutral element for a min-reduction;
    const auto kInf = static_cast<acc_t>(INFINITY);
    acc_t acc = kInf;

    for (int64_t di = 0; di < kH; ++di) {
        int64_t ih = h + di - anchor_h;
        for (int64_t dj = 0; dj < kW; ++dj) {
            acc_t val = kInf;
            if (resolve_coord(ih, H, border)) {
                if (int64_t iw = w + dj - anchor_w; resolve_coord(iw, W, border)) {
                    val = static_cast<acc_t>(input_nc[ih * W + iw]) - static_cast<acc_t>(kernel_c[di * kW + dj]);
                }
                // OuB -> inf
            }
            acc = val < acc ? val : acc;
        }
    }

    output[idx] = static_cast<scalar_t>(acc);
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
 * @throws c10::Error   if the tensors are not on CUDA, have the wrong rank or dtype, the channel counts disagree, or @p
 * border is out of range.
 */
at::Tensor erode(const at::Tensor& input, const at::Tensor& kernel, int64_t border) {
    TORCH_CHECK(input.is_cuda(), "serron::erode: input must be a CUDA tensor");
    TORCH_CHECK(kernel.is_cuda(), "serron::erode: kernel must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "serron::erode: input must be 4-D (N, C, H, W), got ", input.dim(), "-D");
    TORCH_CHECK(kernel.dim() == 2 || kernel.dim() == 3,
                "serron::erode: kernel must be 2-D (kH, kW) or 3-D (C, kH, kW), got ", kernel.dim(), "-D");
    TORCH_CHECK(input.scalar_type() == kernel.scalar_type(), "serron::erode: input and kernel must share a dtype");
    TORCH_CHECK(border >= 0 && border <= 2, "serron::erode: invalid border mode ", border);

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
        TORCH_CHECK(kernel_c.size(0) == C, "serron::erode: kernel channel dim (", kernel_c.size(0),
                    ") must match input channels (", C, ")");
        kH = kernel_c.size(1);
        kW = kernel_c.size(2);
        kernel_channel_stride = kH * kW;
    } else {
        kH = kernel_c.size(0);
        kW = kernel_c.size(1);
        kernel_channel_stride = 0;
    }
    TORCH_CHECK(kH > 0 && kW > 0, "serron::erode: kernel spatial dims must be positive");

    at::Tensor output = at::empty_like(input_c);
    const int64_t total = output.numel();
    if (total == 0)
        return output;

    const c10::cuda::CUDAGuard device_guard(input_c.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const auto blocks = static_cast<unsigned int>((total + THREADS - 1) / THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input_c.scalar_type(), "serron_erode", [&] {
            erode_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                input_c.data_ptr<scalar_t>(), kernel_c.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), N, C, H, W,
                kH, kW, kernel_channel_stride, static_cast<int>(border));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

at::Tensor dilate(const at::Tensor& input, const at::Tensor& kernel, int64_t border) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "serron::dilate is not implemented yet");
}

} // namespace serron