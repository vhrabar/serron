#include "ops.h"

#include <cuda/morphology/enums.cuh>
#include <cuda/morphology/ops_policy.cuh>
#include <cuda/morphology/scan.cuh>
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
                                           const int64_t kernel_channel_stride, const BorderMode border,
                                           const bool need_kernel_grad) {
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
    if (need_kernel_grad) {
        gpuAtomicAdd(&grad_kernel_c[best_k], static_cast<scalar_t>(Op::template se_grad_sign<acc_t>() * go));
    }
}

/**
 * Tiled grayscale morphology backward kernel, one thread per output element.
 *
 * The block stages its halo tile and the SE in shared memory, so an input sample is read once
 * per block instead of once per window it lands in. The tap / reduce / tie-break are the ones
 * the winner was chosen with in the first place, so the gradient goes back to that same tap.
 *
 * block = (TILE_X, TILE_Y); grid = (ceil(W/TILE_X), ceil(H/TILE_Y), N*C);
 * smem = (tile_h*tile_w + kH*kW) * sizeof(scalar_t).
 *
 * @tparam scalar_t              Element type; the reduction accumulates in at::acc_type<scalar_t>.
 * @tparam Op                    Operation policy (@ref ErodeOp or @ref DilateOp).
 * @param grad_output            Upstream gradient, contiguous (N, C, H, W).
 * @param input                  Forward input, contiguous (N, C, H, W).
 * @param kernel                 Structuring element, contiguous (kH, kW) or (C, kH, kW).
 * @param grad_input             Gradient w.r.t. the forward input, pre-zeroed; scattered into.
 * @param grad_kernel            Gradient w.r.t. the forward SE, pre-zeroed; scattered into.
 * @param C                      Channel count.
 * @param H                      Input/output height.
 * @param W                      Input/output width.
 * @param kH                     Structuring-element height.
 * @param kW                     Structuring-element width.
 * @param kernel_channel_stride  Per-channel stride into @p kernel / @p grad_kernel, or 0 for a shared SE.
 * @param border                 Boundary mode (@ref BorderMode) for out-of-image reads.
 * @param need_kernel_grad       Whether grad_kernel is wanted; when false its scatter is skipped.
 */
template <typename scalar_t, typename Op>
__global__ void
morphology_backward_tiled_kernel(const scalar_t* __restrict__ grad_output, const scalar_t* __restrict__ input,
                                 const scalar_t* __restrict__ kernel, scalar_t* __restrict__ grad_input,
                                 scalar_t* __restrict__ grad_kernel, const int64_t C, const int64_t H, const int64_t W,
                                 const int64_t kH, const int64_t kW, const int64_t kernel_channel_stride,
                                 const BorderMode border, const bool need_kernel_grad) {
    using acc_t = at::acc_type<scalar_t, true>;

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

    const auto neutral = Op::template neutral<acc_t>();

    // Cooperative halo load: shared (ty,tx) -> global (out_y0+ty-aH, out_x0+tx-aW)
    for (int64_t ty = threadIdx.y; ty < tile_h; ty += TILE_Y) {
        int64_t gy = out_y0 + ty - anchor_h;
        const bool gy_valid = resolve_coord(gy, H, border);
        for (int64_t tx = threadIdx.x; tx < tile_w; tx += TILE_X) {
            scalar_t v = static_cast<scalar_t>(neutral);
            if (gy_valid) {
                if (int64_t gx = out_x0 + tx - anchor_w; resolve_coord(gx, W, border)) {
                    v = input_nc[gy * W + gx];
                }
            }
            s_input[ty * tile_w + tx] = v;
        }
    }

    for (int64_t t = threadIdx.y * TILE_X + threadIdx.x; t < kH * kW; t += TILE_X * TILE_Y) {
        s_kernel[t] = kernel_c[t];
    }

    __syncthreads();

    const int64_t w = out_x0 + threadIdx.x;
    const int64_t h = out_y0 + threadIdx.y;
    if (w >= W || h >= H)
        return;

    // Recompute the winning tap: identical tap / reduce / tie-break as the FWD
    acc_t best = neutral;
    int64_t best_k = -1;
    int64_t best_di = 0;
    int64_t best_dj = 0;
    for (int64_t di = 0; di < kH; ++di) {
        const scalar_t* s_row = s_input + (threadIdx.y + di) * tile_w + threadIdx.x;
        const scalar_t* k_row = s_kernel + di * kW;
        for (int64_t dj = 0; dj < kW; ++dj) {
            const acc_t val = Op::tap(static_cast<acc_t>(s_row[dj]), static_cast<acc_t>(k_row[dj]));
            if (const acc_t merged = Op::reduce(best, val); merged != best) {
                best = merged;
                best_k = di * kW + dj;
                best_di = di;
                best_dj = dj;
            }
        }
    }

    // An out-of-image tap sits at neutral in the tile, so it only wins when every tap does.
    // The centre tap always resolves in-image, so a finite input never gets that far.
    if (best_k < 0)
        return;
    int64_t ih = h + best_di - anchor_h;
    int64_t iw = w + best_dj - anchor_w;
    if (!resolve_coord(ih, H, border) || !resolve_coord(iw, W, border))
        return;

    const acc_t go = static_cast<acc_t>(grad_output[nc * H * W + h * W + w]);
    gpuAtomicAdd(&grad_input[nc * H * W + ih * W + iw], static_cast<scalar_t>(go));
    if (need_kernel_grad) {
        scalar_t* grad_kernel_c = grad_kernel + c * kernel_channel_stride;
        gpuAtomicAdd(&grad_kernel_c[best_k], static_cast<scalar_t>(Op::template se_grad_sign<acc_t>() * go));
    }
}

/**
 * Stages one line plus its @c k-1 halo into both scan buffers, tagging each sample with the
 * tile-local position it came from.
 *
 * Both passes walk a line the same way and only its layout differs, hence base + stride:
 * the row pass reads along W with stride 1, the column pass along H with stride W.
 *
 * @tparam scalar_t    Stored element type.
 * @tparam acc_t       Accumulate type the scans run in.
 * @param s_forward    Forward-scan buffer of @p tile_len taps; written in full.
 * @param s_backward   Backward-scan buffer of @p tile_len taps; written in full.
 * @param base         First sample of the line.
 * @param stride       Distance between consecutive samples of the line.
 * @param line_len     Samples in the line.
 * @param tile0        Line position the tile's first output covers.
 * @param tile_len     Samples staged, @c chunks*k.
 * @param anchor       Window anchor, @c k/2.
 * @param neutral      Value standing in for an out-of-image sample.
 * @param border       Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename acc_t>
__device__ __forceinline__ void stage_arg_tile(ArgTap<acc_t>* s_forward, ArgTap<acc_t>* s_backward,
                                               const scalar_t* base, const int64_t stride, const int64_t line_len,
                                               const int64_t tile0, const int64_t tile_len, const int64_t anchor,
                                               const acc_t neutral, const BorderMode border) {
    for (int64_t t = threadIdx.x; t < tile_len; t += LINE_TILE) {
        int64_t pos = tile0 + t - anchor;
        ArgTap<acc_t> tap{};
        tap.val = resolve_coord(pos, line_len, border) ? static_cast<acc_t>(base[pos * stride]) : neutral;
        tap.idx = static_cast<cuda::std::int32_t>(t);
        s_forward[t] = tap;
        s_backward[t] = tap;
    }
}

/**
 * Runs both chunk scans over a staged tile, splitting on window length: a warp per chunk once
 * a chunk is at least a warp wide, one thread per chunk below that.
 *
 * @tparam Op          Operation policy (@ref ErodeOp or @ref DilateOp).
 * @tparam acc_t       Accumulate type.
 * @param s_forward    Forward-scan buffer; scanned in place.
 * @param s_backward   Backward-scan buffer; scanned in place.
 * @param k            Window length.
 * @param chunks       Chunks in the tile.
 */
template <typename Op, typename acc_t>
__device__ __forceinline__ void run_arg_scans(ArgTap<acc_t>* s_forward, ArgTap<acc_t>* s_backward, const int64_t k,
                                              const int64_t chunks) {
    if (k < WARP_SIZE) {
        for (int64_t chunk = threadIdx.x; chunk < chunks; chunk += LINE_TILE) {
            arg_scan_chunk_serial<Op, acc_t, true>(s_forward + chunk * k, k);
            arg_scan_chunk_serial<Op, acc_t, false>(s_backward + chunk * k, k);
        }
    } else {
        const unsigned lane = threadIdx.x % WARP_SIZE;
        for (int64_t chunk = threadIdx.x / WARP_SIZE; chunk < chunks; chunk += LINE_WARPS) {
            arg_scan_chunk_warp<Op, acc_t, true>(s_forward + chunk * k, k, lane);
            arg_scan_chunk_warp<Op, acc_t, false>(s_backward + chunk * k, k, lane);
        }
    }
}

/**
 * Row pass of the separable backward (van Herk / Gil-Werman): per (input row @c h, output
 * column @c w), the winning tap over @c dj in [0, kW)
 *
 * block = (LINE_TILE); grid = (ceil(W/out_count), H, N*C);
 * smem = @ref line_smem_bytes over @c sizeof(ArgTap<acc_t>).
 *
 * @tparam scalar_t     Element type; the reduction accumulates in at::acc_type<scalar_t>.
 * @tparam Op           Operation policy (@ref ErodeOp or @ref DilateOp).
 * @param input         Forward input, contiguous (N, C, H, W).
 * @param row_best_val  Row-champion value per (n, c, h, w), contiguous (N, C, H, W).
 * @param row_best_dj   Row-champion @c dj offset per (n, c, h, w), contiguous (N, C, H, W).
 * @param H             Input height.
 * @param W             Input/output width.
 * @param kW            Structuring-element width.
 * @param chunks        Scan windows of @p kW samples per block (@ref line_chunks_per_block).
 * @param border        Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename Op>
__global__ void morphology_row_argreduce_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ row_best_val,
                                                cuda::std::int32_t* __restrict__ row_best_dj, const int64_t H,
                                                const int64_t W, const int64_t kW, const int64_t chunks,
                                                const BorderMode border) {
    using acc_t = at::acc_type<scalar_t, true>;

    const int64_t tile_len = chunks * kW;
    const int64_t out_count = tile_len - kW + 1;

    extern __shared__ __align__(sizeof(double)) unsigned char smem_raw[];
    auto* s_forward = reinterpret_cast<ArgTap<acc_t>*>(smem_raw);
    ArgTap<acc_t>* s_backward = s_forward + tile_len;

    const int64_t h = blockIdx.y;
    const int64_t nc = blockIdx.z;
    const int64_t plane = nc * H * W;
    const int64_t tile0 = static_cast<int64_t>(blockIdx.x) * out_count;

    stage_arg_tile<scalar_t, acc_t>(s_forward, s_backward, input + plane + h * W, 1, W, tile0, tile_len, kW / 2,
                                    Op::template neutral<acc_t>(), border);
    __syncthreads();
    run_arg_scans<Op, acc_t>(s_forward, s_backward, kW, chunks);
    __syncthreads();

    for (int64_t i = threadIdx.x; i < out_count; i += LINE_TILE) {
        const int64_t w = tile0 + i;
        if (w >= W)
            break;

        const ArgTap<acc_t> tap = arg_combine<Op>(s_backward[i], s_forward[i + kW - 1]);
        row_best_val[plane + h * W + w] = static_cast<scalar_t>(tap.val);
        row_best_dj[plane + h * W + w] = static_cast<cuda::std::int32_t>(tap.idx - i);
    }
}

/**
 * Column pass of the separable backward (van Herk / Gil-Werman): reduces each output pixel's
 * row champions over @c di in [0, kH), same lowest-offset tie-break, then scatters the
 * upstream gradient.
 *
 * block = (LINE_TILE); grid = (ceil(H/out_count), W, N*C);
 * smem = @ref line_smem_bytes over @c sizeof(ArgTap<acc_t>).
 *
 * @tparam scalar_t              Element type; the reduction accumulates in at::acc_type<scalar_t>.
 * @tparam Op                    Operation policy (@ref ErodeOp or @ref DilateOp).
 * @param row_best_val           Row-pass output, contiguous (N, C, H, W).
 * @param row_best_dj            Row-pass output, contiguous (N, C, H, W).
 * @param grad_output            Upstream gradient, contiguous (N, C, H, W).
 * @param grad_input             Gradient w.r.t. the forward input, pre-zeroed (N, C, H, W); scattered into.
 * @param grad_kernel            Gradient w.r.t. the forward SE, pre-zeroed, (kH, kW) or (C, kH, kW); scattered into.
 * @param C                      Channel count.
 * @param H                      Input/output height.
 * @param W                      Input/output width.
 * @param kH                     Structuring-element height.
 * @param kW                     Structuring-element width.
 * @param chunks                 Scan windows of @p kH samples per block (@ref line_chunks_per_block).
 * @param kernel_channel_stride  Per-channel stride into @p grad_kernel (kH*kW), or 0 for a shared SE.
 * @param border                 Boundary mode (@ref BorderMode) for out-of-image reads.
 * @param need_kernel_grad       Whether grad_kernel is wanted; when false its scatter is skipped.
 */
template <typename scalar_t, typename Op>
__global__ void morphology_col_argreduce_kernel(
    const scalar_t* __restrict__ row_best_val, const cuda::std::int32_t* __restrict__ row_best_dj,
    const scalar_t* __restrict__ grad_output, scalar_t* __restrict__ grad_input, scalar_t* __restrict__ grad_kernel,
    const int64_t C, const int64_t H, const int64_t W, const int64_t kH, const int64_t kW, const int64_t chunks,
    const int64_t kernel_channel_stride, const BorderMode border, const bool need_kernel_grad) {
    using acc_t = at::acc_type<scalar_t, true>;

    const int64_t tile_len = chunks * kH;
    const int64_t out_count = tile_len - kH + 1;
    const int64_t anchor_h = kH / 2;
    const int64_t anchor_w = kW / 2;

    extern __shared__ __align__(sizeof(double)) unsigned char smem_raw[];
    auto* s_forward = reinterpret_cast<ArgTap<acc_t>*>(smem_raw);
    ArgTap<acc_t>* s_backward = s_forward + tile_len;

    const int64_t w = blockIdx.y;
    const int64_t nc = blockIdx.z;
    const int64_t c = nc % C;
    const int64_t plane = nc * H * W;
    const int64_t tile0 = static_cast<int64_t>(blockIdx.x) * out_count;

    stage_arg_tile<scalar_t, acc_t>(s_forward, s_backward, row_best_val + plane + w, W, H, tile0, tile_len, anchor_h,
                                    Op::template neutral<acc_t>(), border);
    __syncthreads();
    run_arg_scans<Op, acc_t>(s_forward, s_backward, kH, chunks);
    __syncthreads();

    scalar_t* grad_kernel_c = grad_kernel + c * kernel_channel_stride;

    for (int64_t i = threadIdx.x; i < out_count; i += LINE_TILE) {
        const int64_t h = tile0 + i;
        if (h >= H)
            break;

        const ArgTap<acc_t> tap = arg_combine<Op>(s_backward[i], s_forward[i + kH - 1]);
        const int64_t best_di = tap.idx - i;

        // The centre tap (di=anchor_h, dj=anchor_w) always resolves in-image, so a finite input
        // always has a winner. This only catches an all-neutral input, which would otherwise
        // scatter outside grad_input.
        int64_t best_ih = h + best_di - anchor_h;
        if (!resolve_coord(best_ih, H, border))
            continue;

        const cuda::std::int32_t dj = row_best_dj[plane + best_ih * W + w];
        int64_t iw = w + dj - anchor_w;
        resolve_coord(iw, W, border);

        const acc_t go = static_cast<acc_t>(grad_output[plane + h * W + w]);
        gpuAtomicAdd(&grad_input[plane + best_ih * W + iw], static_cast<scalar_t>(go));
        if (need_kernel_grad) {
            gpuAtomicAdd(&grad_kernel_c[best_di * kW + dj],
                         static_cast<scalar_t>(Op::template se_grad_sign<acc_t>() * go));
        }
    }
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
                                          int64_t kernel_channel_stride, BorderMode border, bool need_kernel_grad,
                                          cudaStream_t stream) {
    using acc_t = at::acc_type<scalar_t, true>;
    constexpr size_t kTap = sizeof(ArgTap<acc_t>);

    const dim3 block(LINE_TILE);
    const size_t budget = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;

    const int64_t chunks_row = line_chunks_per_block(kW, W, kTap, budget);
    const int64_t out_row = chunks_row * kW - kW + 1;
    const dim3 grid_row(static_cast<unsigned int>((W + out_row - 1) / out_row), static_cast<unsigned int>(H),
                        static_cast<unsigned int>(N * C));
    morphology_row_argreduce_kernel<scalar_t, Op><<<grid_row, block, line_smem_bytes(chunks_row, kW, kTap), stream>>>(
        input, row_best_val, row_best_dj, H, W, kW, chunks_row, border);

    const int64_t chunks_col = line_chunks_per_block(kH, H, kTap, budget);
    const int64_t out_col = chunks_col * kH - kH + 1;
    const dim3 grid_col(static_cast<unsigned int>((H + out_col - 1) / out_col), static_cast<unsigned int>(W),
                        static_cast<unsigned int>(N * C));
    morphology_col_argreduce_kernel<scalar_t, Op><<<grid_col, block, line_smem_bytes(chunks_col, kH, kTap), stream>>>(
        row_best_val, row_best_dj, grad_output, grad_input, grad_kernel, C, H, W, kH, kW, chunks_col,
        kernel_channel_stride, border, need_kernel_grad);
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
                                bool need_kernel_grad, cudaStream_t stream) {
    using acc_t = at::acc_type<scalar_t, true>;

    // One scan chunk is the smallest tile a line pass can stage. If even that overflows the
    // budget, the kernels below take over no matter how separable the SE is.
    const size_t budget = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
    if (use_separable && line_smem_bytes(1, std::max(kH, kW), sizeof(ArgTap<acc_t>)) <= budget) {
        launch_morphology_backward_separable<scalar_t, Op>(input, grad_output, grad_input, grad_kernel, row_best_val,
                                                           row_best_dj, N, C, H, W, kH, kW, kernel_channel_stride,
                                                           border, need_kernel_grad, stream);
        return;
    }

    const int64_t tile_w = TILE_X + kW - 1;
    const int64_t tile_h = TILE_Y + kH - 1;
    const size_t smem = (static_cast<size_t>(tile_h * tile_w) + static_cast<size_t>(kH * kW)) * sizeof(scalar_t);

    // Tile if it fits, element-wise otherwise
    if (smem <= budget) {
        constexpr dim3 block(TILE_X, TILE_Y);
        const dim3 grid(static_cast<unsigned int>((W + TILE_X - 1) / TILE_X),
                        static_cast<unsigned int>((H + TILE_Y - 1) / TILE_Y), static_cast<unsigned int>(N * C));
        morphology_backward_tiled_kernel<scalar_t, Op>
            <<<grid, block, smem, stream>>>(grad_output, input, kernel, grad_input, grad_kernel, C, H, W, kH, kW,
                                            kernel_channel_stride, border, need_kernel_grad);
    } else {
        const int64_t total = N * C * H * W;
        const auto blocks = static_cast<unsigned int>((total + THREADS - 1) / THREADS);
        morphology_backward_kernel<scalar_t, Op>
            <<<blocks, THREADS, 0, stream>>>(grad_output, input, kernel, grad_input, grad_kernel, N, C, H, W, kH, kW,
                                             kernel_channel_stride, border, need_kernel_grad);
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
 * @throws c10::Error  if the tensors are not on CUDA, have the wrong rank or dtype, the channel counts disagree, or @p
 * border is out of range.
 */
std::tuple<at::Tensor, at::Tensor> morphology_backward_impl(const at::Tensor& grad_output, const at::Tensor& input,
                                                            const at::Tensor& kernel, int64_t border,
                                                            const std::optional<bool>& flat, bool need_kernel_grad,
                                                            MorphOp op, const char* name) {
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

    const bool use_separable = use_separable_path(resolve_flat(flat, kernel_c), kH, kW);
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
                    use_separable, N, C, H, W, kH, kW, kernel_channel_stride, border_mode, need_kernel_grad, stream);
                break;
            case MorphOp::kDilate:
                launch_morphology_backward<scalar_t, DilateOp>(
                    grad_output_ptr, input_ptr, kernel_ptr, grad_input_ptr, grad_kernel_ptr, row_val_ptr, row_dj_ptr,
                    use_separable, N, C, H, W, kH, kW, kernel_channel_stride, border_mode, need_kernel_grad, stream);
                break;
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_input, grad_kernel};
}

} // namespace

std::tuple<at::Tensor, at::Tensor> erode_backward(const at::Tensor& grad_output, const at::Tensor& input,
                                                  const at::Tensor& kernel, const int64_t border,
                                                  const std::optional<bool>& flat, const bool need_kernel_grad) {
    return morphology_backward_impl(grad_output, input, kernel, border, flat, need_kernel_grad, MorphOp::kErode,
                                    "serron::erode_backward");
}

std::tuple<at::Tensor, at::Tensor> dilate_backward(const at::Tensor& grad_output, const at::Tensor& input,
                                                   const at::Tensor& kernel, const int64_t border,
                                                   const std::optional<bool>& flat, const bool need_kernel_grad) {
    return morphology_backward_impl(grad_output, input, kernel, border, flat, need_kernel_grad, MorphOp::kDilate,
                                    "serron::dilate_backward");
}

} // namespace serron
