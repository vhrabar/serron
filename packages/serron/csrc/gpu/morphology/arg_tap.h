#ifndef SERRON_CUDA_MORPHOLOGY_ARG_TAP_H
#define SERRON_CUDA_MORPHOLOGY_ARG_TAP_H

#include <cstdint>

namespace serron {

/**
 * A candidate tap
 *
 * The host sizes the backward line passes' shared memory by it, so it lives outside the device-only scan.cuh.
 *
 * @tparam acc_t  Accumulate type of the value being reduced.
 */
template <typename acc_t>
struct ArgTap {
    acc_t val;
    std::int32_t idx;
};

} // namespace serron

#endif // SERRON_CUDA_MORPHOLOGY_ARG_TAP_H
