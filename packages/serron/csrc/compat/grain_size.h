#ifndef SERRON_COMPAT_GRAIN_SIZE_H
#define SERRON_COMPAT_GRAIN_SIZE_H

#include <cstdint>

namespace serron {

/**
 * Smallest number of iterations worth handing to a single thread in a parallel_for.
 *
 * Stands in for @c at::internal::GRAIN_SIZE, which lives in ATen/TensorIterator.h. The value is the one torch
 * uses, so the work split is unchanged.
 */
constexpr int64_t kGrainSize = 32768;

} // namespace serron

#endif // SERRON_COMPAT_GRAIN_SIZE_H
