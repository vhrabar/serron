#ifndef SERRON_CPU_MORPHOLOGY_SEPARABLE_H
#define SERRON_CPU_MORPHOLOGY_SEPARABLE_H

#include <ATen/core/Tensor.h>

#include <algorithm>
#include <cstdlib>

namespace serron {

/**
 * Window length at which the separable r+c path is preferred over the direct 2-D kernel, for a flat, axis-separable
 * structuring element.
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
 * True when @p kernel_c is a flat SE
 */
inline bool se_is_flat(const at::Tensor& kernel_c) {
    return kernel_c.eq(0).all().item<bool>();
}

/**
 * True when (@p kH, @p kW) and @p kernel_c justify the separable r+c path over the direct 2-D kernel, per @ref
 * separable_min_k.
 */
inline bool use_separable_path(const at::Tensor& kernel_c, int64_t kH, int64_t kW) {
    return std::max(kH, kW) >= separable_min_k() && se_is_flat(kernel_c);
}

} // namespace serron

#endif // SERRON_CPU_MORPHOLOGY_SEPARABLE_H
