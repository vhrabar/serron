#ifndef SERRON_COMPAT_FLATNESS_H
#define SERRON_COMPAT_FLATNESS_H

#include <cstdint>

namespace serron {

/**
 * True when all @p n elements at @p data compare equal to zero.
 *
 * @param data  Start of a contiguous run of elements, on the host.
 * @param n     Elements to read.
 */
template <typename T>
bool all_zero(const T* data, const int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        if (static_cast<double>(data[i]) != 0.0) {
            return false;
        }
    }
    return true;
}

} // namespace serron

#endif // SERRON_COMPAT_FLATNESS_H
