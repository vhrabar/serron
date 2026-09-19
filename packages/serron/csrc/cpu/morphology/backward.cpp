#include "ops.h"

#include <cpu/morphology/enums.h>
#include <cpu/morphology/ops_policy.h>
#include <cpu/morphology/separable.h>
#include <cpu/utils/boundaries.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace serron {

namespace {

/**
 * Grayscale morphology backward CPU kernel, host-side mirror of the CUDA
 * @c morphology_backward_kernel.
 *
 * Grayscale morphology selects exactly one @c (input, se) tap per output window
 * (argmin for erosion, argmax for dilation), so the backward is a scatter of the
 * upstream gradient to that selected location. The winning tap is recomputed with
 * the same @c tap / @c reduce / tie-break as the forward pass so the input-grad
 * and SE-grad target the same tap. Each output element is independent, so the
 * search runs under @c at::parallel_for.
 *
 * @tparam scalar_t              Element type of the tensors; the reduction accumulates in at::acc_type<scalar_t>.
 * @tparam Op                    Operation policy (@ref ErodeOp or @ref DilateOp).
 * @param input                  Forward input, contiguous (N, C, H, W).
 * @param kernel                 Forward structuring element, contiguous (kH, kW) or (C, kH, kW).
 * @param best_in                Per output element, the flat offset of the winning input pixel; written in full.
 * @param best_k                 Per output element, the flat offset of the winning SE cell; written in full.
 * @param N, C, H, W             Batch / channel / spatial extents.
 * @param kH, kW                 Structuring-element spatial extents.
 * @param kernel_channel_stride  Per-channel stride into @p kernel / @c grad_kernel (kH*kW), or 0 for a shared SE.
 * @param border                 Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename Op>
void morphology_backward_winners(const scalar_t* input, const scalar_t* kernel, int64_t* best_in, int64_t* best_k,
                                 const int64_t N, const int64_t C, const int64_t H, const int64_t W, const int64_t kH,
                                 const int64_t kW, const int64_t kernel_channel_stride, const BorderMode border) {
    using acc_t = at::acc_type<scalar_t, false>;

    const int64_t anchor_h = kH / 2;
    const int64_t anchor_w = kW / 2;
    const int64_t total = N * C * H * W;
    const auto neutral = Op::template neutral<acc_t>();

    at::parallel_for(0, total, at::internal::GRAIN_SIZE, [&](const int64_t begin, const int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            const int64_t w = idx % W;
            const int64_t h = (idx / W) % H;
            const int64_t c = (idx / (W * H)) % C;
            const int64_t n = idx / (W * H * C);

            const scalar_t* input_nc = input + (n * C + c) * H * W;
            const scalar_t* kernel_c = kernel + c * kernel_channel_stride;

            // Recompute the winning tap: identical tap / reduce / tie-break as the FWD.
            acc_t best = neutral;
            int64_t best_k_local = -1;  // di*kW + dj -> SE cell
            int64_t best_in_local = -1; // ih*W + iw  -> border-resolved input pixel
            for (int64_t di = 0; di < kH; ++di) {
                int64_t ih = h + di - anchor_h;
                const bool ih_valid = resolve_coord(ih, H, border);
                for (int64_t dj = 0; dj < kW; ++dj) {
                    acc_t val = neutral;
                    int64_t in_off = -1;
                    if (ih_valid) {
                        if (int64_t iw = w + dj - anchor_w; resolve_coord(iw, W, border)) {
                            in_off = ih * W + iw;
                            val = Op::tap(static_cast<acc_t>(input_nc[in_off]),
                                          static_cast<acc_t>(kernel_c[di * kW + dj]));
                        }
                        // OoB -> neutral
                    }
                    const acc_t merged = Op::reduce(best, val);
                    if (merged != best) {
                        best = merged;
                        best_k_local = di * kW + dj;
                        best_in_local = in_off;
                    }
                }
            }

            // The centre tap (di=anchor_h, dj=anchor_w) always resolves in-image, so a winner is guaranteed.
            best_in[idx] = (n * C + c) * H * W + best_in_local;
            best_k[idx] = c * kernel_channel_stride + best_k_local;
        }
    });
}

/**
 * A candidate tap
 *
 * @tparam acc_t  Accumulate type of the value being reduced.
 */
template <typename acc_t>
struct ArgTap {
    acc_t val;
    int64_t idx;
};

/**
 * Combines two candidates, @p lo holding the lower sample offsets and @p hi the higher.
 *
 */
template <typename Op, typename acc_t>
inline ArgTap<acc_t> arg_combine(const ArgTap<acc_t>& lo, const ArgTap<acc_t>& hi) {
    return Op::reduce(lo.val, hi.val) != lo.val ? hi : lo;
}

/**
 * One separable argreduce pass (van Herk / Gil-Werman) over every line of one axis.
 *
 * The scans carry @ref ArgTap, so each output yields the offset that won it as well as the
 * value. Every output window straddles exactly one chunk boundary, so one @ref arg_combine of
 * the backward scan at its first sample and the forward scan at its last settles it, whatever
 * @p k is.
 *
 * @tparam scalar_t  Element type; the reduction accumulates in at::acc_type<scalar_t, false>.
 * @tparam Op        Operation policy (@ref ErodeOp or @ref DilateOp).
 * @param read       Returns the sample at (line, position) for an in-image position.
 * @param write      Receives (line, position, winning value, winning offset).
 * @param lines      Lines along the axis.
 * @param line_len   Samples per line.
 * @param k          Window length along the axis.
 * @param border     Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename Op, typename Read, typename Write>
void argreduce_lines(Read read, Write write, const int64_t lines, const int64_t line_len, const int64_t k,
                     const BorderMode border) {
    using acc_t = at::acc_type<scalar_t, false>;

    const int64_t anchor = k / 2;
    const int64_t span = line_len + k - 1;         // positions [-anchor, line_len + k - 1 - anchor)
    const int64_t padded = (span + k - 1) / k * k; // whole chunks of k -> both scans reset in step
    const auto neutral = Op::template neutral<acc_t>();
    const int64_t grain = std::max<int64_t>(1, at::internal::GRAIN_SIZE / std::max<int64_t>(line_len, 1));

    at::parallel_for(0, lines, grain, [&](const int64_t begin, const int64_t end) {
        std::vector<ArgTap<acc_t>> samples(static_cast<size_t>(padded));
        std::vector<ArgTap<acc_t>> prefix(static_cast<size_t>(padded));
        std::vector<ArgTap<acc_t>> suffix(static_cast<size_t>(padded));

        for (int64_t line = begin; line < end; ++line) {
            for (int64_t t = 0; t < padded; ++t) {
                int64_t pos = t - anchor;
                const bool inside = t < span && resolve_coord(pos, line_len, border);
                samples[t] = {inside ? static_cast<acc_t>(read(line, pos)) : neutral, t};
            }

            for (int64_t t = 0; t < padded; ++t) {
                prefix[t] = (t % k == 0) ? samples[t] : arg_combine<Op>(prefix[t - 1], samples[t]);
            }
            for (int64_t t = padded - 1; t >= 0; --t) {
                suffix[t] = (t % k == k - 1) ? samples[t] : arg_combine<Op>(samples[t], suffix[t + 1]);
            }

            for (int64_t out_pos = 0; out_pos < line_len; ++out_pos) {
                const ArgTap<acc_t> tap = arg_combine<Op>(suffix[out_pos], prefix[out_pos + k - 1]);
                write(line, out_pos, tap.val, tap.idx - out_pos);
            }
        }
    });
}

/**
 * Separable winner search: a row pass over W, then a column pass over H, each O(1) per output
 * in the window length. Needs a flat, axis-separable SE.
 *
 * @tparam scalar_t              Element type.
 * @tparam Op                    Operation policy (@ref ErodeOp or @ref DilateOp).
 * @param input                  Forward input, contiguous (N, C, H, W).
 * @param best_in                Receives the winning input offset per output element.
 * @param best_k                 Receives the winning SE offset per output element.
 * @param N, C, H, W             Batch / channel / spatial extents.
 * @param kH                     Structuring-element height.
 * @param kW                     Structuring-element width.
 * @param kernel_channel_stride  Per-channel stride into the SE (kH*kW), or 0 for a shared SE.
 * @param border                 Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename Op>
void morphology_backward_winners_separable(const scalar_t* input, int64_t* best_in, int64_t* best_k, const int64_t N,
                                           const int64_t C, const int64_t H, const int64_t W, const int64_t kH,
                                           const int64_t kW, const int64_t kernel_channel_stride,
                                           const BorderMode border) {
    using acc_t = at::acc_type<scalar_t, false>;

    const int64_t anchor_h = kH / 2;
    const int64_t anchor_w = kW / 2;
    const int64_t total = N * C * H * W;

    std::vector<scalar_t> row_best_val(static_cast<size_t>(total));
    std::vector<int32_t> row_best_dj(static_cast<size_t>(total));

    // Row pass: line == nc * H + h, since the input is contiguous (..., H, W)
    argreduce_lines<scalar_t, Op>([&](const int64_t line, const int64_t pos) { return input[line * W + pos]; },
                                  [&](const int64_t line, const int64_t w, const acc_t val, const int64_t dj) {
                                      row_best_val[line * W + w] = static_cast<scalar_t>(val);
                                      row_best_dj[line * W + w] = static_cast<int32_t>(dj);
                                  },
                                  N * C * H, W, kW, border);

    // Column pass: line == nc * W + w, walking H with stride W
    argreduce_lines<scalar_t, Op>(
        [&](const int64_t line, const int64_t pos) {
            const int64_t nc = line / W;
            const int64_t w = line % W;
            return row_best_val[nc * H * W + pos * W + w];
        },
        [&](const int64_t line, const int64_t h, const acc_t, const int64_t di) {
            const int64_t nc = line / W;
            const int64_t w = line % W;
            const int64_t c = nc % C;

            int64_t ih = h + di - anchor_h;
            resolve_coord(ih, H, border);
            const int32_t dj = row_best_dj[nc * H * W + ih * W + w];
            int64_t iw = w + dj - anchor_w;
            resolve_coord(iw, W, border);

            const int64_t idx = nc * H * W + h * W + w;
            best_in[idx] = nc * H * W + ih * W + iw;
            best_k[idx] = c * kernel_channel_stride + di * kW + dj;
        },
        N * C * W, H, kH, border);
}

/**
 * Replays the upstream gradient onto the winning taps.
 *
 * @tparam scalar_t     Element type of the tensors; the scatter accumulates in at::acc_type<scalar_t>.
 * @tparam Op           Operation policy (@ref ErodeOp or @ref DilateOp).
 * @param grad_output   Upstream gradient, contiguous (N, C, H, W).
 * @param grad_input    Gradient w.r.t. the forward input, pre-zeroed (N, C, H, W); scattered into.
 * @param grad_kernel   Gradient w.r.t. the forward SE, pre-zeroed, same shape as the SE; scattered into.
 * @param best_in       Winning input pixel per output element.
 * @param best_k        Winning SE cell per output element.
 * @param total         Output element count (N*C*H*W).
 */
template <typename scalar_t, typename Op>
void scatter_backward(const scalar_t* grad_output, scalar_t* grad_input, scalar_t* grad_kernel, const int64_t* best_in,
                      const int64_t* best_k, const int64_t total, const bool need_kernel_grad) {
    using acc_t = at::acc_type<scalar_t, false>;

    const auto se_grad_sign = Op::template se_grad_sign<acc_t>();
    for (int64_t idx = 0; idx < total; ++idx) {
        const auto go = static_cast<acc_t>(grad_output[idx]);
        grad_input[best_in[idx]] += static_cast<scalar_t>(go);
        if (need_kernel_grad) {
            grad_kernel[best_k[idx]] += static_cast<scalar_t>(se_grad_sign * go);
        }
    }
}

/**
 * Run the morphology backward pass: the separable row+column argreduce when @p use_separable
 * was set by the caller (flat, axis-separable SE at/above @ref separable_min_k), otherwise the
 * direct 2-D search, as @ref morphology_backward_winners. Either way the winners are scattered
 * by @ref scatter_backward.
 */
template <typename scalar_t, typename Op>
void morphology_backward_cpu(const scalar_t* grad_output, const scalar_t* input, const scalar_t* kernel,
                             scalar_t* grad_input, scalar_t* grad_kernel, const bool use_separable, const int64_t N,
                             const int64_t C, const int64_t H, const int64_t W, const int64_t kH, const int64_t kW,
                             const int64_t kernel_channel_stride, const BorderMode border,
                             const bool need_kernel_grad) {
    const int64_t total = N * C * H * W;

    // Winning tap per output element, as flat offsets into grad_input / grad_kernel.
    std::vector<int64_t> best_in(static_cast<size_t>(total));
    std::vector<int64_t> best_k(static_cast<size_t>(total));

    if (use_separable) {
        morphology_backward_winners_separable<scalar_t, Op>(input, best_in.data(), best_k.data(), N, C, H, W, kH, kW,
                                                            kernel_channel_stride, border);
    } else {
        morphology_backward_winners<scalar_t, Op>(input, kernel, best_in.data(), best_k.data(), N, C, H, W, kH, kW,
                                                  kernel_channel_stride, border);
    }

    scatter_backward<scalar_t, Op>(grad_output, grad_input, grad_kernel, best_in.data(), best_k.data(), total,
                                   need_kernel_grad);
}

/**
 * Shared host-side backward behind @ref erode_backward_cpu / @ref dilate_backward_cpu.
 *
 * CPU mirror of the CUDA @c morphology_backward_impl: validates the inputs, normalises the structuring-element layout,
 * then dispatches on dtype and @p op to @ref morphology_backward_cpu.
 *
 * @param grad_output  Upstream gradient, CPU tensor of shape (N, C, H, W), same dtype as @p input.
 * @param input        Forward input, CPU tensor of shape (N, C, H, W), floating dtype.
 * @param kernel       Forward structuring element, CPU tensor of shape (kH, kW) or (C, kH, kW); same dtype as @p input.
 * @param border       Boundary mode (@ref BorderMode encoding) used in the forward pass.
 * @param op           Operation to differentiate (@ref MorphOp).
 * @param name         Qualified caller name used to prefix diagnostics.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 * @throws c10::Error  if the tensors are not on CPU, have the wrong rank or dtype, the channel counts disagree, or @p
 * border is out of range.
 */
std::tuple<at::Tensor, at::Tensor> morphology_backward_impl(const at::Tensor& grad_output, const at::Tensor& input,
                                                            const at::Tensor& kernel, int64_t border,
                                                            const std::optional<bool>& flat, bool need_kernel_grad,
                                                            MorphOp op, const char* name) {
    TORCH_CHECK(grad_output.is_cpu(), name, ": grad_output must be a CPU tensor");
    TORCH_CHECK(input.is_cpu(), name, ": input must be a CPU tensor");
    TORCH_CHECK(kernel.is_cpu(), name, ": kernel must be a CPU tensor");
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
    const auto border_mode = static_cast<BorderMode>(border);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input_c.scalar_type(), "serron_morphology_backward_cpu", [&] {
            const scalar_t* grad_output_ptr = grad_output_c.data_ptr<scalar_t>();
            const scalar_t* input_ptr = input_c.data_ptr<scalar_t>();
            const scalar_t* kernel_ptr = kernel_c.data_ptr<scalar_t>();
            auto* grad_input_ptr = grad_input.data_ptr<scalar_t>();
            auto* grad_kernel_ptr = grad_kernel.data_ptr<scalar_t>();
            switch (op) {
            case MorphOp::kErode:
                morphology_backward_cpu<scalar_t, ErodeOp>(grad_output_ptr, input_ptr, kernel_ptr, grad_input_ptr,
                                                           grad_kernel_ptr, use_separable, N, C, H, W, kH, kW,
                                                           kernel_channel_stride, border_mode, need_kernel_grad);
                break;
            case MorphOp::kDilate:
                morphology_backward_cpu<scalar_t, DilateOp>(grad_output_ptr, input_ptr, kernel_ptr, grad_input_ptr,
                                                            grad_kernel_ptr, use_separable, N, C, H, W, kH, kW,
                                                            kernel_channel_stride, border_mode, need_kernel_grad);
                break;
            }
        });

    return {grad_input, grad_kernel};
}

} // namespace

/**
 * Backward pass of grayscale erosion (CPU).
 *
 * @param grad_output  Upstream gradient, CPU tensor of shape (N, C, H, W), same dtype as @p input.
 * @param input        Forward input, CPU tensor of shape (N, C, H, W), floating dtype.
 * @param kernel       Forward structuring element, CPU tensor of shape (kH, kW) or (C, kH, kW); same dtype as @p input.
 * @param border       Boundary mode (@ref BorderMode) used in the forward pass.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 */
std::tuple<at::Tensor, at::Tensor> erode_backward_cpu(const at::Tensor& grad_output, const at::Tensor& input,
                                                      const at::Tensor& kernel, const int64_t border,
                                                      const std::optional<bool>& flat, const bool need_kernel_grad) {
    return morphology_backward_impl(grad_output, input, kernel, border, flat, need_kernel_grad, MorphOp::kErode,
                                    "serron::erode_backward");
}

/**
 * Backward pass of grayscale dilation (CPU).
 *
 * @param grad_output  Upstream gradient, CPU tensor of shape (N, C, H, W), same dtype as @p input.
 * @param input        Forward input, CPU tensor of shape (N, C, H, W), floating dtype.
 * @param kernel       Forward structuring element, CPU tensor of shape (kH, kW) or (C, kH, kW); same dtype as @p input.
 * @param border       Boundary mode (@ref BorderMode) used in the forward pass.
 * @return             Pair (grad_input, grad_kernel) matching the shapes of @p input and @p kernel.
 */
std::tuple<at::Tensor, at::Tensor> dilate_backward_cpu(const at::Tensor& grad_output, const at::Tensor& input,
                                                       const at::Tensor& kernel, const int64_t border,
                                                       const std::optional<bool>& flat, const bool need_kernel_grad) {
    return morphology_backward_impl(grad_output, input, kernel, border, flat, need_kernel_grad, MorphOp::kDilate,
                                    "serron::dilate_backward");
}

} // namespace serron
