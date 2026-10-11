#include <cuda/morphology/entry_points.h>
#include <cuda/morphology/ops_policy.cuh>
#include <cuda/morphology/scan.cuh>
#include <cuda/utils/boundaries.cuh>
#include <cuda/utils/declarations.h>

#include <cuda_runtime.h>

#include <compat/acc_type.h>

#include <cstdint>
#include <type_traits>

// Device code only: the forward kernels and their named entry points. The host side that picks and launches them is
// operators.cpp.

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
__device__ __forceinline__ void
morphology_kernel(const scalar_t* __restrict__ input, const scalar_t* __restrict__ kernel,
                  scalar_t* __restrict__ output, const int64_t N, const int64_t C, const int64_t H, const int64_t W,
                  const int64_t kH, const int64_t kW, const int64_t kernel_channel_stride, const BorderMode border) {
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
__device__ __forceinline__ void morphology_tiled_kernel(const scalar_t* __restrict__ input,
                                                        const scalar_t* __restrict__ kernel,
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
__device__ __forceinline__ void morphology_line_kernel(const scalar_t* __restrict__ input,
                                                       scalar_t* __restrict__ output, const int64_t N, const int64_t C,
                                                       const int64_t H, const int64_t W, const int64_t k,
                                                       const int64_t chunks, const BorderMode border) {
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

} // namespace

/// Forward entry points for one (op, dtype): element-wise, tiled, and the row / column line passes.
#define SERRON_FORWARD_ENTRY_POINTS(Op, op, scalar_t, suffix)                                                          \
    extern "C" __global__ void serron_morphology_##op##_##suffix(                                                      \
        const scalar_t* __restrict__ input, const scalar_t* __restrict__ kernel, scalar_t* __restrict__ output,        \
        const int64_t N, const int64_t C, const int64_t H, const int64_t W, const int64_t kH, const int64_t kW,        \
        const int64_t kernel_channel_stride, const BorderMode border) {                                                \
        morphology_kernel<scalar_t, Op>(input, kernel, output, N, C, H, W, kH, kW, kernel_channel_stride, border);     \
    }                                                                                                                  \
    extern "C" __global__ void serron_morphology_tiled_##op##_##suffix(                                                \
        const scalar_t* __restrict__ input, const scalar_t* __restrict__ kernel, scalar_t* __restrict__ output,        \
        const int64_t C, const int64_t H, const int64_t W, const int64_t kH, const int64_t kW,                         \
        const int64_t kernel_channel_stride, const BorderMode border) {                                                \
        morphology_tiled_kernel<scalar_t, Op>(input, kernel, output, C, H, W, kH, kW, kernel_channel_stride, border);  \
    }                                                                                                                  \
    extern "C" __global__ void serron_morphology_line_row_##op##_##suffix(                                             \
        const scalar_t* __restrict__ input, scalar_t* __restrict__ output, const int64_t N, const int64_t C,           \
        const int64_t H, const int64_t W, const int64_t k, const int64_t chunks, const BorderMode border) {            \
        morphology_line_kernel<scalar_t, Op, LineAxis::kRow>(input, output, N, C, H, W, k, chunks, border);            \
    }                                                                                                                  \
    extern "C" __global__ void serron_morphology_line_col_##op##_##suffix(                                             \
        const scalar_t* __restrict__ input, scalar_t* __restrict__ output, const int64_t N, const int64_t C,           \
        const int64_t H, const int64_t W, const int64_t k, const int64_t chunks, const BorderMode border) {            \
        morphology_line_kernel<scalar_t, Op, LineAxis::kCol>(input, output, N, C, H, W, k, chunks, border);            \
    }

#define SERRON_FORWARD_ENTRY_POINTS_FOR_DTYPE(scalar_t, suffix)                                                        \
    SERRON_CUDA_FOR_EACH_OP(SERRON_FORWARD_ENTRY_POINTS, scalar_t, suffix)
SERRON_CUDA_FOR_EACH_DTYPE(SERRON_FORWARD_ENTRY_POINTS_FOR_DTYPE)

#define SERRON_CHECK_FORWARD_ENTRY_POINTS(Op, op, scalar_t, suffix)                                                    \
    SERRON_FORWARD_KERNELS(SERRON_CHECK_ENTRY_POINT_TYPE, op, scalar_t, suffix)
#define SERRON_CHECK_FORWARD_ENTRY_POINTS_FOR_DTYPE(scalar_t, suffix)                                                  \
    SERRON_CUDA_FOR_EACH_OP(SERRON_CHECK_FORWARD_ENTRY_POINTS, scalar_t, suffix)
SERRON_CUDA_FOR_EACH_DTYPE(SERRON_CHECK_FORWARD_ENTRY_POINTS_FOR_DTYPE)

} // namespace serron
