#include "ops.h"

#include <cpu/morphology/enums.h>
#include <cpu/morphology/ops_policy.h>
#include <cpu/morphology/separable.h>
#include <cpu/utils/boundaries.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace serron {

namespace {

/**
 * Grayscale morphology CPU kernel, one iteration per output element, parallelised over the flattened output with
 * @c at::parallel_for. Each output is independent, so no synchronisation is needed.
 *
 * @tparam scalar_t              Element type of @p input / @p kernel / @p output; the reduction accumulates in
 * at::acc_type<scalar_t, false>.
 * @tparam Op                    Operation policy (@ref ErodeOp or @ref DilateOp).
 * @param input                  Input image, contiguous (N, C, H, W).
 * @param kernel                 Structuring element, contiguous (kH, kW) or (C, kH, kW).
 * @param output                 Output image, contiguous (N, C, H, W); written in full.
 * @param N, C, H, W             Batch / channel / spatial extents.
 * @param kH, kW                 Structuring-element spatial extents.
 * @param kernel_channel_stride  Per-channel stride into @p kernel (kH*kW), or 0 when shared across channels.
 * @param border                 Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename Op>
void morphology_cpu_kernel(const scalar_t* input, const scalar_t* kernel, scalar_t* output, const int64_t N,
                           const int64_t C, const int64_t H, const int64_t W, const int64_t kH, const int64_t kW,
                           const int64_t kernel_channel_stride, const BorderMode border) {
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
                            val =
                                Op::tap(static_cast<acc_t>(input_nc[ih_off + iw]), static_cast<acc_t>(kernel_row[dj]));
                        }
                        // OoB -> neutral element (+/- inf)
                    }
                    acc = Op::reduce(acc, val);
                }
            }
            output[idx] = static_cast<scalar_t>(acc);
        }
    });
}

/// Which axis a separable line pass reduces over.
enum class LineAxis : int { kRow = 0, kCol = 1 };

/**
 * Separable line-reduction pass (van Herk / Gil-Werman): every line of the chosen axis is
 * materialised with its @c k-1 halo resolved per @p border, then scanned twice -- forward
 * within blocks of @p k, backward within the same blocks. Each output window then falls out
 * of a single pairwise reduce of the two scans, so the cost per output is independent of the
 * window length. Requires a flat structuring element, whose taps leave the samples unchanged.
 *
 * @tparam scalar_t  Element type; the reduction accumulates in at::acc_type<scalar_t, false>.
 * @tparam Op        Operation policy (@ref ErodeOp or @ref DilateOp); only @c neutral and @c reduce are used.
 * @tparam Axis      @ref LineAxis::kRow reduces along W (contiguous); @ref LineAxis::kCol along H (stride W).
 * @param input      Input image, contiguous (N, C, H, W).
 * @param output     Output image, contiguous (N, C, H, W); written in full.
 * @param N, C, H, W Batch / channel / spatial extents.
 * @param k          Window length along @p Axis (kW for kRow, kH for kCol).
 * @param border     Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename Op, LineAxis Axis>
void morphology_line_cpu_kernel(const scalar_t* input, scalar_t* output, const int64_t N, const int64_t C,
                                const int64_t H, const int64_t W, const int64_t k, const BorderMode border) {
    using acc_t = at::acc_type<scalar_t, false>;

    const int64_t line_len = (Axis == LineAxis::kRow) ? W : H;
    const int64_t lines = (Axis == LineAxis::kRow) ? H : W;
    const int64_t stride = (Axis == LineAxis::kRow) ? 1 : W;
    const int64_t anchor = k / 2;
    const int64_t span = line_len + k - 1;
    const int64_t padded = (span + k - 1) / k * k;
    const auto neutral = Op::template neutral<acc_t>();
    const int64_t grain = std::max<int64_t>(1, at::internal::GRAIN_SIZE / std::max<int64_t>(line_len, 1));

    at::parallel_for(0, N * C * lines, grain, [&](const int64_t begin, const int64_t end) {
        // [span, padded) is never written, so it keeps the neutral element these start out with.
        std::vector<acc_t> samples(static_cast<size_t>(padded), neutral);
        std::vector<acc_t> prefix(static_cast<size_t>(padded), neutral);
        std::vector<acc_t> suffix(static_cast<size_t>(padded), neutral);

        for (int64_t idx = begin; idx < end; ++idx) {
            const int64_t line = idx % lines;
            const int64_t nc = idx / lines;
            const scalar_t* input_nc = input + nc * H * W;
            scalar_t* output_nc = output + nc * H * W;
            const int64_t line_off = (Axis == LineAxis::kRow) ? line * W : line;

            for (int64_t t = 0; t < span; ++t) {
                int64_t pos = t - anchor;
                samples[t] = resolve_coord(pos, line_len, border)
                                 ? static_cast<acc_t>(input_nc[line_off + pos * stride])
                                 : neutral;
            }

            for (int64_t t = 0; t < padded; ++t) {
                prefix[t] = (t % k == 0) ? samples[t] : Op::reduce(prefix[t - 1], samples[t]);
            }
            for (int64_t t = padded - 1; t >= 0; --t) {
                suffix[t] = (t % k == k - 1) ? samples[t] : Op::reduce(suffix[t + 1], samples[t]);
            }

            for (int64_t out_pos = 0; out_pos < line_len; ++out_pos) {
                const acc_t acc = Op::reduce(suffix[out_pos], prefix[out_pos + k - 1]);
                output_nc[line_off + out_pos * stride] = static_cast<scalar_t>(acc);
            }
        }
    });
}

/**
 * Chains the row pass (@p input -> @p scratch) into the column pass (@p scratch -> @p output).
 * Requires the structuring element to be flat and axis-separable.
 *
 * @param input    Input image, contiguous (N, C, H, W).
 * @param scratch  Intermediate buffer, same shape/dtype as @p input; holds the row-pass result.
 * @param output   Output image, contiguous (N, C, H, W).
 * @param N, C, H, W  Batch / channel / spatial extents.
 * @param kH       Structuring-element height (window length for the column pass).
 * @param kW       Structuring-element width (window length for the row pass).
 * @param border   Boundary mode (@ref BorderMode) for out-of-image reads.
 */
template <typename scalar_t, typename Op>
void morphology_separable_cpu(const scalar_t* input, scalar_t* scratch, scalar_t* output, const int64_t N,
                              const int64_t C, const int64_t H, const int64_t W, const int64_t kH, const int64_t kW,
                              const BorderMode border) {
    morphology_line_cpu_kernel<scalar_t, Op, LineAxis::kRow>(input, scratch, N, C, H, W, kW, border);
    morphology_line_cpu_kernel<scalar_t, Op, LineAxis::kCol>(scratch, output, N, C, H, W, kH, border);
}

/**
 * Run the morphology forward pass: the separable row+column reduction when @p use_separable was
 * set by the caller (flat, axis-separable SE at/above @ref separable_min_k), otherwise the direct
 * 2-D window, as @ref morphology_cpu_kernel.
 */
template <typename scalar_t, typename Op>
void morphology_cpu(const scalar_t* input, const scalar_t* kernel, scalar_t* output, scalar_t* scratch,
                    const bool use_separable, const int64_t N, const int64_t C, const int64_t H, const int64_t W,
                    const int64_t kH, const int64_t kW, const int64_t kernel_channel_stride, const BorderMode border) {
    if (use_separable) {
        morphology_separable_cpu<scalar_t, Op>(input, scratch, output, N, C, H, W, kH, kW, border);
        return;
    }
    morphology_cpu_kernel<scalar_t, Op>(input, kernel, output, N, C, H, W, kH, kW, kernel_channel_stride, border);
}

/**
 * Shared host-side implementation behind @ref erode_cpu / @ref dilate_cpu: validates the inputs, normalises the
 * structuring-element layout, then dispatches on dtype and @p op to @ref morphology_cpu_kernel.
 *
 * @param input   CPU tensor of shape (N, C, H, W), floating dtype.
 * @param kernel  CPU structuring element of shape (kH, kW) or (C, kH, kW); same dtype as @p input.
 * @param border  Boundary mode (@ref BorderMode encoding) applied at the image edges.
 * @param op      Operation to apply (@ref MorphOp).
 * @param name    Qualified caller name used to prefix diagnostics.
 * @return        Result tensor, same shape and dtype as @p input.
 * @throws c10::Error  if the tensors are not on CPU, have the wrong rank or dtype, the channel counts disagree, or @p
 * border is out of range.
 */
at::Tensor morphology_impl(const at::Tensor& input, const at::Tensor& kernel, int64_t border,
                           const std::optional<bool>& flat, MorphOp op, const char* name) {
    TORCH_CHECK(input.is_cpu(), name, ": input must be a CPU tensor");
    TORCH_CHECK(kernel.is_cpu(), name, ": kernel must be a CPU tensor");
    TORCH_CHECK(input.dim() == 4, name, ": input must be 4-D (N, C, H, W), got ", input.dim(), "-D");
    TORCH_CHECK(kernel.dim() == 2 || kernel.dim() == 3, name, ": kernel must be 2-D (kH, kW) or 3-D (C, kH, kW), got ",
                kernel.dim(), "-D");
    TORCH_CHECK(input.scalar_type() == kernel.scalar_type(), name, ": input and kernel must share a dtype");
    TORCH_CHECK(border >= kReflect && border <= kConstant, name, ": invalid border mode ", border);

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

    at::Tensor output = at::empty_like(input_c);
    if (output.numel() == 0)
        return output;

    const bool use_separable = use_separable_path(resolve_flat(flat, kernel_c), kH, kW);
    at::Tensor scratch;
    if (use_separable) {
        scratch = at::empty_like(input_c);
    }

    const auto border_mode = static_cast<BorderMode>(border);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input_c.scalar_type(), "serron_morphology_cpu", [&] {
            scalar_t* scratch_ptr = use_separable ? scratch.data_ptr<scalar_t>() : nullptr;
            switch (op) {
            case MorphOp::kErode:
                morphology_cpu<scalar_t, ErodeOp>(input_c.data_ptr<scalar_t>(), kernel_c.data_ptr<scalar_t>(),
                                                  output.data_ptr<scalar_t>(), scratch_ptr, use_separable, N, C, H, W,
                                                  kH, kW, kernel_channel_stride, border_mode);
                break;
            case MorphOp::kDilate:
                morphology_cpu<scalar_t, DilateOp>(input_c.data_ptr<scalar_t>(), kernel_c.data_ptr<scalar_t>(),
                                                   output.data_ptr<scalar_t>(), scratch_ptr, use_separable, N, C, H, W,
                                                   kH, kW, kernel_channel_stride, border_mode);
                break;
            }
        });

    return output;
}

} // namespace

/**
 * Grayscale erosion (sliding minimum) of @p input by structuring element @p kernel.
 *
 * @param input         CPU tensor of shape (N, C, H, W), floating dtype.
 * @param kernel        CPU structuring element of shape (kH, kW), shared across channels, or (C, kH, kW) for a
 * per-channel element; same dtype as @p input.
 * @param border        Boundary mode (@ref BorderMode) applied at the image edges.
 * @return              Eroded tensor, same shape and dtype as @p input.
 */
at::Tensor erode_cpu(const at::Tensor& input, const at::Tensor& kernel, const int64_t border,
                     const std::optional<bool>& flat) {
    return morphology_impl(input, kernel, border, flat, MorphOp::kErode, "serron::erode");
}

/**
 * Grayscale dilation (sliding maximum) of @p input by structuring element @p kernel.
 *
 * @param input         CPU tensor of shape (N, C, H, W), floating dtype.
 * @param kernel        CPU structuring element of shape (kH, kW), shared across channels, or (C, kH, kW) for a
 * per-channel element; same dtype as @p input.
 * @param border        Boundary mode (@ref BorderMode) applied at the image edges.
 * @return              Dilated tensor, same shape and dtype as @p input.
 */
at::Tensor dilate_cpu(const at::Tensor& input, const at::Tensor& kernel, const int64_t border,
                      const std::optional<bool>& flat) {
    return morphology_impl(input, kernel, border, flat, MorphOp::kDilate, "serron::dilate");
}

} // namespace serron
