#ifndef SERRON_CPU_MORPHOLOGY_SEPARABLE_H
#define SERRON_CPU_MORPHOLOGY_SEPARABLE_H

#include <compat/flatness.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>
#include <cstdlib>
#include <optional>

namespace serron {

using torch::stable::Tensor;

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
 * True when @p kernel_c is a flat SE.
 *
 * Scans the tensor's own memory, which is already host memory here.
 *
 * @param kernel_c  Contiguous structuring element on the CPU.
 */
inline bool se_is_flat(const Tensor& kernel_c) {
    bool flat = false;
    THO_DISPATCH_V2(kernel_c.scalar_type(), "serron_se_is_flat_cpu",
                    AT_WRAP([&] { flat = all_zero(kernel_c.const_data_ptr<scalar_t>(), kernel_c.numel()); }),
                    torch::headeronly::ScalarType::Float, torch::headeronly::ScalarType::Double,
                    torch::headeronly::ScalarType::Half, torch::headeronly::ScalarType::BFloat16);
    return flat;
}

/**
 * Flatness of @p kernel_c, taking the caller's answer when it has one.
 *
 * The Python layer caches flatness per structuring element and passes it down, so @p flat is normally set and
 * nothing is read; @c std::nullopt keeps the direct check for callers that go at the operator without it.
 *
 * @param flat      Caller's answer, or @c std::nullopt to check @p kernel_c directly.
 * @param kernel_c  Contiguous structuring element.
 */
inline bool resolve_flat(const std::optional<bool>& flat, const Tensor& kernel_c) {
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
