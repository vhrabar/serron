#ifndef SERRON_CPU_MORPHOLOGY_SEPARABLE_H
#define SERRON_CPU_MORPHOLOGY_SEPARABLE_H

#include <ATen/core/Tensor.h>

#include <algorithm>
#include <cstdlib>
#include <optional>

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
        return static_cast<int64_t>(5);
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
 * Flatness of @p kernel_c, taking the caller's answer when it has one.
 *
 * @ref se_is_flat reads a device-side reduction back to the host, which synchronises. The
 * Python layer caches flatness per structuring element and passes it down, so @p flat is
 * normally set and no synchronisation happens; @c std::nullopt keeps the direct check for
 * callers that go at the operator without it.
 *
 * @param flat      Caller's answer, or @c std::nullopt to check @p kernel_c directly.
 * @param kernel_c  Contiguous structuring element.
 */
inline bool resolve_flat(const std::optional<bool>& flat, const at::Tensor& kernel_c) {
    return flat.has_value() ? *flat : se_is_flat(kernel_c);
}

/**
 * True when @p is_flat and (@p kH, @p kW) justify the separable r+c path over the direct 2-D kernel, per @ref
 * separable_min_k.
 */
inline bool use_separable_path(const bool is_flat, const int64_t kH, const int64_t kW) {
    return is_flat && std::max(kH, kW) >= separable_min_k();
}

} // namespace serron

#endif // SERRON_CPU_MORPHOLOGY_SEPARABLE_H
