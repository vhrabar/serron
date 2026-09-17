#ifndef SERRON_MORPHOLOGY_SEPARABLE_CUH
#define SERRON_MORPHOLOGY_SEPARABLE_CUH

#include <cuda/utils/declarations.cuh>

#include <ATen/core/Tensor.h>

#include <algorithm>
#include <cstdlib>

namespace serron {

/**
 * Window length at which the separable r+c path is preferred over the tiled/GMEM 2-D kernels, for a flat,
 * axis-separable structuring element.
 */
inline int64_t separable_min_k() {
    static const int64_t value = [] {
        if (const char* env = std::getenv("SERRON_SEPARABLE_MIN_K")) {
            char* end = nullptr;
            const long parsed = std::strtol(env, &end, 10);
            if (end != env && *end == '\0' && parsed > 0) {
                return static_cast<int64_t>(parsed);
            }
        }
        return static_cast<int64_t>(20);
    }();
    return value;
}

/**
 * Shared memory one van Herk / Gil-Werman line block needs
 */
inline size_t line_smem_bytes(const int64_t chunks, const int64_t k, const size_t elem_size) {
    return 2 * static_cast<size_t>(chunks * k) * elem_size;
}

/**
 * How many van Herk / Gil-Werman chunks of @p k samples one line block scans.
 *

 *
 * @return Chunks per block, or 0 when a single chunk already overflows @p smem_budget.
 */
inline int64_t line_chunks_per_block(const int64_t k, const int64_t line_len, const size_t elem_size,
                                     const size_t smem_budget) {
    const int64_t by_smem = static_cast<int64_t>(smem_budget / line_smem_bytes(1, k, elem_size));
    const int64_t by_line = (line_len + 2 * k - 2) / k;
    const int64_t by_outputs = (LINE_TILE + 2 * k - 2) / k;
    const int64_t by_scan = k < WARP_SIZE ? LINE_TILE : LINE_WARPS;
    return std::min({by_smem, by_line, std::max(by_scan, by_outputs)});
}

/**
 * True when @p kernel_c is a flat SE
 */
inline bool se_is_flat(const at::Tensor& kernel_c) {
    return kernel_c.eq(0).all().item<bool>();
}

/**
 * True when (@p kH, @p kW) and @p kernel_c justify the separable r+c path over the tiled/GMEM 2-D kernels, per @ref
 * separable_min_k.
 */
inline bool use_separable_path(const at::Tensor& kernel_c, int64_t kH, int64_t kW) {
    return std::max(kH, kW) >= separable_min_k() && se_is_flat(kernel_c);
}

} // namespace serron

#endif // SERRON_MORPHOLOGY_SEPARABLE_CUH
