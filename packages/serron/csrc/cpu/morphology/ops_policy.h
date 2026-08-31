#ifndef SERRON_CPU_MORPHOLOGY_OPS_POLICY_H
#define SERRON_CPU_MORPHOLOGY_OPS_POLICY_H

#include <cmath>

namespace serron {

/**
 * Operation policies for the CPU morphology kernels.
 */
struct ErodeOp {
    template <typename acc_t>
    static acc_t neutral() {
        return static_cast<acc_t>(INFINITY);
    }
    template <typename acc_t>
    static acc_t tap(const acc_t sample, const acc_t se) {
        return sample - se;
    }
    template <typename acc_t>
    static acc_t reduce(const acc_t acc, const acc_t val) {
        return val < acc ? val : acc;
    }
    /// d(tap)/d(se): erosion's tap is @c sample-se
    template <typename acc_t>
    static acc_t se_grad_sign() {
        return static_cast<acc_t>(-1);
    }
};

struct DilateOp {
    template <typename acc_t>
    static acc_t neutral() {
        return static_cast<acc_t>(-INFINITY);
    }
    template <typename acc_t>
    static acc_t tap(const acc_t sample, const acc_t se) {
        return sample + se;
    }
    template <typename acc_t>
    static acc_t reduce(const acc_t acc, const acc_t val) {
        return val > acc ? val : acc;
    }
    /// d(tap)/d(se): dilation's tap is @c sample+se
    template <typename acc_t>
    static acc_t se_grad_sign() {
        return static_cast<acc_t>(1);
    }
};

} // namespace serron

#endif // SERRON_CPU_MORPHOLOGY_OPS_POLICY_H
