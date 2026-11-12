#ifndef SERRON_CUDA_MORPHOLOGY_THRESHOLDS_H
#define SERRON_CUDA_MORPHOLOGY_THRESHOLDS_H

#include <cuda/morphology/separable.h>

#include <torch/headeronly/core/ScalarType.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace serron {

/// Window length from which the separable path is preferred when nothing else says otherwise.
constexpr int64_t kDefaultSeparableMinK = 11;

namespace detail {

/// Key of a threshold: device index, dtype, and whether it is for the backward pass.
using ThresholdKey = std::tuple<int, int, bool>;

struct ThresholdTable {
    std::mutex mutex;
    std::map<ThresholdKey, int64_t> values;
};

inline ThresholdTable& threshold_table() {
    static ThresholdTable table;
    return table;
}

} // namespace detail

/**
 * Window length at which the separable path takes over from the tiled/element-wise kernels, for a flat,
 * axis-separable structuring element.
 *
 * The order of precedence is the SERRON_SEPARABLE_MIN_K environment variable, then a threshold set with
 * @ref set_separable_min_k for this device, dtype and pass, then @ref kDefaultSeparableMinK.
 *
 * @param device    CUDA device index.
 * @param dtype     Element type of the input.
 * @param backward  Whether the threshold is for the backward pass.
 */
inline int64_t separable_min_k(const int device, const torch::headeronly::ScalarType dtype, const bool backward) {
    if (const int64_t forced = separable_min_k_override(); forced > 0) {
        return forced;
    }
    auto& table = detail::threshold_table();
    const std::scoped_lock lock(table.mutex);
    const auto it = table.values.find({device, static_cast<int>(dtype), backward});
    return it == table.values.end() ? kDefaultSeparableMinK : it->second;
}

/**
 * Sets the separable threshold for one device, dtype and pass.
 *
 * @param min_k  New window length, or a value <= 0 to go back to @ref kDefaultSeparableMinK.
 */
inline void set_separable_min_k(const int device, const torch::headeronly::ScalarType dtype, const bool backward,
                                const int64_t min_k) {
    auto& table = detail::threshold_table();
    const std::scoped_lock lock(table.mutex);
    const detail::ThresholdKey key{device, static_cast<int>(dtype), backward};
    if (min_k > 0) {
        table.values[key] = min_k;
    } else {
        table.values.erase(key);
    }
}

} // namespace serron

#endif // SERRON_CUDA_MORPHOLOGY_THRESHOLDS_H
