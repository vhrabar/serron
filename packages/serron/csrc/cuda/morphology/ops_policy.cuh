#ifndef SERRON_MORPHOLOGY_OPS_POLICY_CUH
#define SERRON_MORPHOLOGY_OPS_POLICY_CUH

#include <cuda_runtime.h>

#include <cmath>

namespace serron {

/**
 * Operation policies for the morphology kernels.
 *
 * Shared by the forward reduction and the backward argmin/argmax recompute so
 * both select the same tap (identical @ref tap, @ref reduce, and tie-breaking).
 */
struct ErodeOp {
    template <typename acc_t>
    static __device__ __forceinline__ acc_t neutral() {
        return static_cast<acc_t>(INFINITY);
    }
    template <typename acc_t>
    static __device__ __forceinline__ acc_t tap(const acc_t sample, const acc_t se) {
        return sample - se;
    }
    template <typename acc_t>
    static __device__ __forceinline__ acc_t reduce(const acc_t acc, const acc_t val) {
        return val < acc ? val : acc;
    }
};

struct DilateOp {
    template <typename acc_t>
    static __device__ __forceinline__ acc_t neutral() {
        return static_cast<acc_t>(-INFINITY);
    }
    template <typename acc_t>
    static __device__ __forceinline__ acc_t tap(const acc_t sample, const acc_t se) {
        return sample + se;
    }
    template <typename acc_t>
    static __device__ __forceinline__ acc_t reduce(const acc_t acc, const acc_t val) {
        return val > acc ? val : acc;
    }
};

} // namespace serron

#endif // SERRON_MORPHOLOGY_OPS_POLICY_CUH