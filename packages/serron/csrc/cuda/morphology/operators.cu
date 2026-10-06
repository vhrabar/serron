#include "ops.h"

#include <cuda/morphology/enums.cuh>
#include <cuda/morphology/ops_policy.cuh>
#include <cuda/morphology/scan.cuh>
#include <cuda/morphology/separable.cuh>
#include <cuda/utils/boundaries.cuh>
#include <cuda/utils/declarations.cuh>
#include <cuda/utils/smem.cuh>

#include <cuda_runtime.h>

#include <compat/acc_type.h>
#include <compat/check.h>
#include <compat/cuda_launch_check.cuh>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cmath>
#include <cstdlib>

namespace serron {

namespace {

/**
 * Grayscale morphology kernel, one thread per output element.
 *
 * @ref ErodeOp:   out = min_{(di,dj)} [ in(n,c,i+di-aH,j+dj-aW) - se(c,di,dj) ]
 * @ref DilateOp:  out = max_{(di,dj)} [ in(n,c,i+di-aH,j+dj-aW) + se(c,di,dj) ]
 *
 *
 * @tparam scalar_t              Element type of @p input / @p kernel / @p output; the reduction accumulates in
 * acc_type<scalar_t>.
 * @tparam Op                    Operation policy (@ref ErodeOp or @ref DilateOp).
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
template <typename scalar_t, typename Op>
__global__ void morphology_kernel(const scalar_t* __restrict__ input, const scalar_t* __restrict__ kernel,
                                  scalar_t* __restrict__ output, const int64_t N, const int64_t C, const int64_t H,
                                  const int64_t W, const int64_t kH, const int64_t kW,
                                  const int64_t kernel_channel_stride, const BorderMode border) {
    using acc_t = acc_type<scalar_t, true>;

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

    const auto neutral = Op::template neutral<acc_t>();
    acc_t acc = neutral;

    for (int64_t di = 0; di < kH; ++di) {
        int64_t ih = h + di - anchor_h;

        const bool ih_valid = resolve_coord(ih, H, border);
        const int64_t ih_off = ih * W;
        const scalar_t* kernel_row = kernel_c + di * kW;
        for (int64_t dj = 0; dj < kW; ++dj) {
            acc_t val = neutral;
            if (ih_valid) {
                if (int64_t iw = w + dj - anchor_w; resolve_coord(iw, W, border)) {
                    val = Op::tap(static_cast<acc_t>(input_nc[ih_off + iw]), static_cast<acc_t>(kernel_row[dj]));
                }
                // OuB -> neutral element (+/- inf)
            }
            acc = Op::reduce(acc, val);
        }
    }

    output[idx] = static_cast<scalar_t>(acc);
}

/**
 * SMEM tiled grayscale morphology kernel (one thread per output pixel).
 *
 * Each block cooperatively stages its output tile plus the surrounding halo of the
 * input, a (TILE_Y + kH - 1) x (TILE_X + kW - 1) region, into shared memory, then
 * the structuring element (kH x kW) after it. Border handling is baked into the halo
 * load (out-of-image cells receive the neutral element.
 *
 * Launch config: block = (TILE_X, TILE_Y); grid = (ceil(W/TILE_X), ceil(H/TILE_Y), N*C);
 * dynamic shared memory = (tile_h*tile_w + kH*kW) * sizeof(scalar_t) bytes.
 *
 * @tparam scalar_t              Element type; the reduction accumulates in acc_type<scalar_t>.
 * @tparam Op                    Operation policy (@ref ErodeOp or @ref DilateOp).
 * @param input                  Input image, contiguous (N, C, H, W).
 * @param kernel                 Structuring element, contiguous (kH, kW) or (C, kH, kW).
 * @param output                 Output image, contiguous (N, C, H, W).
 * @param C                      Channel count (to recover the SE channel from blockIdx.z).
 * @param H                      Input/output height.
 * @param W                      Input/output width.
 * @param kH                     Structuring-element height.
 * @param kW                     Structuring-element width.
 * @param kernel_channel_stride  Per-channel stride into @p kernel (kH*kW), or 0 when shared across channels.
 * @param border                 Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename Op>
__global__ void morphology_tiled_kernel(const scalar_t* __restrict__ input, const scalar_t* __restrict__ kernel,
                                        scalar_t* __restrict__ output, const int64_t C, const int64_t H,
                                        const int64_t W, const int64_t kH, const int64_t kW,
                                        const int64_t kernel_channel_stride, const BorderMode border) {
    using acc_t = acc_type<scalar_t, true>;

    const int64_t anchor_h = kH / 2;
    const int64_t anchor_w = kW / 2;
    const int64_t tile_w = TILE_X + kW - 1;
    const int64_t tile_h = TILE_Y + kH - 1;

    extern __shared__ __align__(sizeof(double)) unsigned char smem_raw[];
    auto* s_input = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* s_kernel = s_input + tile_h * tile_w;

    const int64_t nc = blockIdx.z;
    const int64_t c = nc % C;
    const scalar_t* input_nc = input + nc * H * W;
    const scalar_t* kernel_c = kernel + c * kernel_channel_stride;

    const int64_t out_x0 = static_cast<int64_t>(blockIdx.x) * TILE_X;
    const int64_t out_y0 = static_cast<int64_t>(blockIdx.y) * TILE_Y;

    const auto neutral = static_cast<scalar_t>(Op::template neutral<acc_t>());

    // Cooperative halo load: shared cell (ty,tx) maps to global (out_y0+ty-aH, out_x0+tx-aW).
    for (int64_t ty = threadIdx.y; ty < tile_h; ty += TILE_Y) {
        int64_t gy = out_y0 + ty - anchor_h;
        const bool gy_valid = resolve_coord(gy, H, border);
        for (int64_t tx = threadIdx.x; tx < tile_w; tx += TILE_X) {
            scalar_t v = neutral;
            if (gy_valid) {
                if (int64_t gx = out_x0 + tx - anchor_w; resolve_coord(gx, W, border)) {
                    v = input_nc[gy * W + gx];
                }
            }
            s_input[ty * tile_w + tx] = v;
        }
    }

    // Cooperative SE load
    for (int64_t k = threadIdx.y * TILE_X + threadIdx.x; k < kH * kW; k += TILE_X * TILE_Y) {
        s_kernel[k] = kernel_c[k];
    }

    __syncthreads();

    const int64_t out_x = out_x0 + threadIdx.x;
    const int64_t out_y = out_y0 + threadIdx.y;
    if (out_x >= W || out_y >= H)
        return;

    acc_t acc = Op::template neutral<acc_t>();
    for (int64_t di = 0; di < kH; ++di) {
        const scalar_t* s_row = s_input + (threadIdx.y + di) * tile_w + threadIdx.x;
        const scalar_t* k_row = s_kernel + di * kW;
        for (int64_t dj = 0; dj < kW; ++dj) {
            const auto val = Op::tap(static_cast<acc_t>(s_row[dj]), static_cast<acc_t>(k_row[dj]));
            acc = Op::reduce(acc, val);
        }
    }

    output[nc * H * W + out_y * W + out_x] = static_cast<scalar_t>(acc);
}

/// Which axis a separable line pass reduces over.
enum class LineAxis : int { kRow = 0, kCol = 1 };

/**
 * Separable line-reduction kernel (van Herk / Gil-Werman)
 *
 * @tparam scalar_t  Element type; the reduction accumulates in acc_type<scalar_t>.
 * @tparam Op        Operation policy (@ref ErodeOp or @ref DilateOp); only @c neutral and
 * @c reduce are used.
 * @tparam Axis      @ref LineAxis::kRow reduces along W (contiguous); @ref LineAxis::kCol
 * along H (stride W).
 * @param input      Input image, contiguous (N, C, H, W).
 * @param output     Output image, contiguous (N, C, H, W); written in full.
 * @param N          Batch size.
 * @param C          Channel count.
 * @param H          Input/output height.
 * @param W          Input/output width.
 * @param k          Window length along @p Axis (kW for kRow, kH for kCol).
 * @param chunks     Scan windows of @p k samples per block (@ref line_chunks_per_block).
 * @param border     Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename Op, LineAxis Axis>
__global__ void morphology_line_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
                                       const int64_t N, const int64_t C, const int64_t H, const int64_t W,
                                       const int64_t k, const int64_t chunks, const BorderMode border) {
    using acc_t = acc_type<scalar_t, true>;

    const int64_t line_len = (Axis == LineAxis::kRow) ? W : H;
    const int64_t anchor = k / 2;
    const int64_t tile_len = chunks * k;
    const int64_t out_count = tile_len - k + 1;

    extern __shared__ __align__(sizeof(double)) unsigned char smem_raw[];
    auto* s_forward = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* s_backward = s_forward + tile_len;

    const int64_t line = blockIdx.y;
    const int64_t nc = blockIdx.z;
    const scalar_t* input_nc = input + nc * H * W;
    scalar_t* output_nc = output + nc * H * W;

    const int64_t tile0 = static_cast<int64_t>(blockIdx.x) * out_count;
    const auto neutral = static_cast<scalar_t>(Op::template neutral<acc_t>());

    // Consecutive threads take consecutive samples
    for (int64_t t = threadIdx.x; t < tile_len; t += LINE_TILE) {
        int64_t pos = tile0 + t - anchor;
        const bool valid = resolve_coord(pos, line_len, border);
        scalar_t v = neutral;
        if (valid) {
            if constexpr (Axis == LineAxis::kRow) {
                v = input_nc[line * W + pos];
            } else {
                v = input_nc[pos * W + line];
            }
        }
        s_forward[t] = v;
        s_backward[t] = v;
    }
    __syncthreads();

    // each scan reads its s own chuck
    if (k < WARP_SIZE) {
        for (int64_t chunk = threadIdx.x; chunk < chunks; chunk += LINE_TILE) {
            scan_chunk_serial<Op, scalar_t, true>(s_forward + chunk * k, k);
            scan_chunk_serial<Op, scalar_t, false>(s_backward + chunk * k, k);
        }
    } else {
        const unsigned lane = threadIdx.x % WARP_SIZE;
        for (int64_t chunk = threadIdx.x / WARP_SIZE; chunk < chunks; chunk += LINE_WARPS) {
            scan_chunk_warp<Op, scalar_t, true>(s_forward + chunk * k, k, lane);
            scan_chunk_warp<Op, scalar_t, false>(s_backward + chunk * k, k, lane);
        }
    }
    __syncthreads();

    for (int64_t i = threadIdx.x; i < out_count; i += LINE_TILE) {
        const int64_t out_pos = tile0 + i;
        if (out_pos >= line_len)
            break;

        const acc_t acc = Op::reduce(static_cast<acc_t>(s_backward[i]), static_cast<acc_t>(s_forward[i + k - 1]));
        if constexpr (Axis == LineAxis::kRow) {
            output_nc[line * W + out_pos] = static_cast<scalar_t>(acc);
        } else {
            output_nc[out_pos * W + line] = static_cast<scalar_t>(acc);
        }
    }
}

/**
 * Chains the row pass (@p input -> @p scratch) into the column pass (@p scratch ->
 * @p output). Requires the structuring element to be flat and axis-separable
 *
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
template <typename scalar_t, typename Op>
bool launch_morphology_separable(const scalar_t* input, scalar_t* scratch, scalar_t* output, int64_t N, int64_t C,
                                 int64_t H, int64_t W, int64_t kH, int64_t kW, BorderMode border, cudaStream_t stream) {
    const dim3 block(LINE_TILE);
    const size_t budget = smem_budget();

    const int64_t chunks_row = line_chunks_per_block(kW, W, sizeof(scalar_t), budget);
    const int64_t chunks_col = line_chunks_per_block(kH, H, sizeof(scalar_t), budget);
    const size_t smem_row = line_smem_bytes(chunks_row, kW, sizeof(scalar_t));
    const size_t smem_col = line_smem_bytes(chunks_col, kH, sizeof(scalar_t));

    if (!configure_kernel_smem(morphology_line_kernel<scalar_t, Op, LineAxis::kRow>, smem_row) ||
        !configure_kernel_smem(morphology_line_kernel<scalar_t, Op, LineAxis::kCol>, smem_col)) {
        return false;
    }

    const int64_t out_row = chunks_row * kW - kW + 1;
    const dim3 grid_row(static_cast<unsigned int>((W + out_row - 1) / out_row), static_cast<unsigned int>(H),
                        static_cast<unsigned int>(N * C));
    morphology_line_kernel<scalar_t, Op, LineAxis::kRow>
        <<<grid_row, block, smem_row, stream>>>(input, scratch, N, C, H, W, kW, chunks_row, border);

    const int64_t out_col = chunks_col * kH - kH + 1;
    const dim3 grid_col(static_cast<unsigned int>((H + out_col - 1) / out_col), static_cast<unsigned int>(W),
                        static_cast<unsigned int>(N * C));
    morphology_line_kernel<scalar_t, Op, LineAxis::kCol>
        <<<grid_col, block, smem_col, stream>>>(scratch, output, N, C, H, W, kH, chunks_col, border);
    return true;
}

/**
 * Launch the morphology forward pass on @p stream.
 *
 */
template <typename scalar_t, typename Op>
void launch_morphology(const scalar_t* input, const scalar_t* kernel, scalar_t* output, scalar_t* scratch,
                       bool use_separable, int64_t N, int64_t C, int64_t H, int64_t W, int64_t kH, int64_t kW,
                       int64_t kernel_channel_stride, BorderMode border, cudaStream_t stream) {
    const size_t budget = smem_budget();

    if (use_separable && line_smem_bytes(1, std::max(kH, kW), sizeof(scalar_t)) <= budget &&
        launch_morphology_separable<scalar_t, Op>(input, scratch, output, N, C, H, W, kH, kW, border, stream)) {
        return;
    }

    const int64_t tile_w = TILE_X + kW - 1;
    const int64_t tile_h = TILE_Y + kH - 1;
    const size_t smem = (static_cast<size_t>(tile_h * tile_w) + static_cast<size_t>(kH * kW)) * sizeof(scalar_t);

    // Tile if it fits, element-wise otherwise
    if (smem <= budget && configure_kernel_smem(morphology_tiled_kernel<scalar_t, Op>, smem)) {
        constexpr dim3 block(TILE_X, TILE_Y);
        const dim3 grid(static_cast<unsigned int>((W + TILE_X - 1) / TILE_X),
                        static_cast<unsigned int>((H + TILE_Y - 1) / TILE_Y), static_cast<unsigned int>(N * C));
        morphology_tiled_kernel<scalar_t, Op>
            <<<grid, block, smem, stream>>>(input, kernel, output, C, H, W, kH, kW, kernel_channel_stride, border);
    } else {
        const int64_t total = N * C * H * W;
        const auto blocks = static_cast<unsigned int>((total + THREADS - 1) / THREADS);
        morphology_kernel<scalar_t, Op>
            <<<blocks, THREADS, 0, stream>>>(input, kernel, output, N, C, H, W, kH, kW, kernel_channel_stride, border);
    }
}

/**
 * Launches the dtype-specialised kernel selected by @p op.
 *
 * Hoisted out of the dispatch macro deliberately: THO_DISPATCH_V2's body cannot contain an unprotected comma, and
 * neither the one in @c launch_morphology<scalar_t, ErodeOp> nor the ones in a @c <<<>>> launch configuration
 * survive macro argument splitting.
 */
template <typename scalar_t>
void launch_morphology_for(const MorphOp op, const scalar_t* input, const scalar_t* kernel, scalar_t* output,
                           scalar_t* scratch, const bool use_separable, const int64_t N, const int64_t C,
                           const int64_t H, const int64_t W, const int64_t kH, const int64_t kW,
                           const int64_t kernel_channel_stride, const BorderMode border, cudaStream_t stream) {
    switch (op) {
    case MorphOp::kErode:
        launch_morphology<scalar_t, ErodeOp>(input, kernel, output, scratch, use_separable, N, C, H, W, kH, kW,
                                             kernel_channel_stride, border, stream);
        break;
    case MorphOp::kDilate:
        launch_morphology<scalar_t, DilateOp>(input, kernel, output, scratch, use_separable, N, C, H, W, kH, kW,
                                              kernel_channel_stride, border, stream);
        break;
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
                        launch_morphology_for<scalar_t>(op, input_c.const_data_ptr<scalar_t>(),
                                                        kernel_c.const_data_ptr<scalar_t>(),
                                                        output.mutable_data_ptr<scalar_t>(), scratch_ptr, use_separable,
                                                        N, C, H, W, kH, kW, kernel_channel_stride, border_mode, stream);
                    }),
                    torch::headeronly::ScalarType::Float, torch::headeronly::ScalarType::Double,
                    torch::headeronly::ScalarType::Half, torch::headeronly::ScalarType::BFloat16);
    SERRON_CUDA_KERNEL_LAUNCH_CHECK();

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
