#ifndef SERRON_COMPAT_ACC_TYPE_H
#define SERRON_COMPAT_ACC_TYPE_H

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

namespace serron {

/**
 * Type intermediate results accumulate in, for a stored element type @p T.
 *
 * Stands in for @c at::acc_type, which lives in ATen/AccumulateType.h. Only the dtypes the operators dispatch
 * over are covered: float, double, Half and BFloat16. The values are taken from at::acc_type's table, so
 * accumulation precision is unchanged.
 *
 * @tparam T       Stored element type.
 * @tparam IsCuda  Accumulate for a device kernel, where double costs throughput, rather than for the host.
 */
template <typename T, bool IsCuda>
struct AccumulateType {
    using type = T;
};

/// Reduced-precision floats accumulate in float, on the host and the device alike.
template <bool IsCuda>
struct AccumulateType<torch::headeronly::Half, IsCuda> {
    using type = float;
};

template <bool IsCuda>
struct AccumulateType<torch::headeronly::BFloat16, IsCuda> {
    using type = float;
};

/// float widens to double on the host, but stays float on the device.
template <>
struct AccumulateType<float, false> {
    using type = double;
};

template <>
struct AccumulateType<float, true> {
    using type = float;
};

// double accumulates in double on both, which the primary template already gives.

template <typename T, bool IsCuda>
using acc_type = typename AccumulateType<T, IsCuda>::type;

} // namespace serron

#endif // SERRON_COMPAT_ACC_TYPE_H
