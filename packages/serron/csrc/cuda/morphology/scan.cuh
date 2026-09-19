#ifndef SERRON_MORPHOLOGY_SCAN_CUH
#define SERRON_MORPHOLOGY_SCAN_CUH

#include <cuda/std/cstdint>
#include <cuda/utils/declarations.cuh>

#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>

namespace serron {

/**
 * In-place scan of one @p k-sample chunk in shared memory -> single warp
 *
 * @tparam Op        Operation policy (@ref ErodeOp or @ref DilateOp).
 * @tparam scalar_t  Stored element type; lanes exchange @c at::acc_type<scalar_t>.
 * @tparam Forward   true scans from the chunk's first sample, false from its last.
 * @param chunk      Shared-memory base of the chunk; overwritten with the scan.
 * @param k          Samples in the chunk.
 * @param lane       Calling thread's lane within its warp.
 */
template <typename Op, typename scalar_t, bool Forward>
__device__ __forceinline__ void scan_chunk_warp(scalar_t* chunk, const int64_t k, const unsigned lane) {
    using acc_t = at::acc_type<scalar_t, true>;
    constexpr unsigned kAll = 0xffffffffu;

    const auto neutral = Op::template neutral<acc_t>();
    const int64_t widths = (k + WARP_SIZE - 1) / WARP_SIZE;
    acc_t carry = neutral;

    for (int64_t w = 0; w < widths; ++w) {
        const int64_t idx = (Forward ? w : widths - 1 - w) * WARP_SIZE + lane;
        const bool inside = idx < k;
        acc_t v = inside ? static_cast<acc_t>(chunk[idx]) : neutral;

        for (int d = 1; d < WARP_SIZE; d <<= 1) {
            const acc_t other = Forward ? __shfl_up_sync(kAll, v, d) : __shfl_down_sync(kAll, v, d);
            const bool consumes = Forward ? lane >= static_cast<unsigned>(d) : lane + d < WARP_SIZE;
            if (consumes) {
                v = Op::reduce(v, other);
            }
        }

        v = Op::reduce(v, carry);
        if (inside) {
            chunk[idx] = static_cast<scalar_t>(v);
        }
        carry = __shfl_sync(kAll, v, Forward ? WARP_SIZE - 1 : 0);
    }
}

/**
 * In-place scan of one @p k-sample chunk in shared memory -> single threaded
 *
 * @tparam Op        Operation policy (@ref ErodeOp or @ref DilateOp).
 * @tparam scalar_t  Stored element type; the scan accumulates in @c at::acc_type<scalar_t>.
 * @tparam Forward   true scans from the chunk's first sample, false from its last.
 * @param chunk      Shared-memory base of the chunk; overwritten with the scan.
 * @param k          Samples in the chunk.
 */
template <typename Op, typename scalar_t, bool Forward>
__device__ __forceinline__ void scan_chunk_serial(scalar_t* chunk, const int64_t k) {
    using acc_t = at::acc_type<scalar_t, true>;

    acc_t acc = static_cast<acc_t>(chunk[Forward ? 0 : k - 1]);
    for (int64_t i = 1; i < k; ++i) {
        const int64_t idx = Forward ? i : k - 1 - i;
        acc = Op::reduce(acc, static_cast<acc_t>(chunk[idx]));
        chunk[idx] = static_cast<scalar_t>(acc);
    }
}

/**
 * A candidate tap
 *
 * @tparam acc_t  Accumulate type of the value being reduced.
 */
template <typename acc_t>
struct ArgTap {
    acc_t val;
    cuda::std::int32_t idx;
};

/**
 * Combine two candidates
 *
 * @tparam Op     Operation policy (@ref ErodeOp or @ref DilateOp).
 * @tparam acc_t  Accumulate type.
 */
template <typename Op, typename acc_t>
__device__ __forceinline__ ArgTap<acc_t> arg_combine(const ArgTap<acc_t>& lo, const ArgTap<acc_t>& hi) {
    return Op::reduce(lo.val, hi.val) != lo.val ? hi : lo;
}

/**
 * In-place argreduce scan of one @p k-tap chunk in shared memory -> single warp
 *
 *
 * @tparam Op       Operation policy (@ref ErodeOp or @ref DilateOp).
 * @tparam acc_t    Accumulate type.
 * @tparam Forward  true scans from the chunk's first tap, false from its last.
 * @param chunk     Shared-memory base of the chunk; overwritten with the scan.
 * @param k         Taps in the chunk.
 * @param lane      Calling thread's lane within its warp.
 */
template <typename Op, typename acc_t, bool Forward>
__device__ __forceinline__ void arg_scan_chunk_warp(ArgTap<acc_t>* chunk, const int64_t k, const unsigned lane) {
    constexpr unsigned kAll = 0xffffffffu;

    const ArgTap<acc_t> neutral{Op::template neutral<acc_t>(), 0};
    const int64_t widths = (k + WARP_SIZE - 1) / WARP_SIZE;
    ArgTap<acc_t> carry = neutral;

    for (int64_t w = 0; w < widths; ++w) {
        const int64_t idx = (Forward ? w : widths - 1 - w) * WARP_SIZE + lane;
        const bool inside = idx < k;
        ArgTap<acc_t> v = inside ? chunk[idx] : neutral;

        for (int d = 1; d < WARP_SIZE; d <<= 1) {
            ArgTap<acc_t> other{};
            other.val = Forward ? __shfl_up_sync(kAll, v.val, d) : __shfl_down_sync(kAll, v.val, d);
            other.idx = Forward ? __shfl_up_sync(kAll, v.idx, d) : __shfl_down_sync(kAll, v.idx, d);

            const bool consumes = Forward ? lane >= static_cast<unsigned>(d) : lane + d < WARP_SIZE;
            if (consumes) {
                // Scanning forward the partner holds the lower offsets; scanning backward it holds the higher.
                v = Forward ? arg_combine<Op>(other, v) : arg_combine<Op>(v, other);
            }
        }

        v = Forward ? arg_combine<Op>(carry, v) : arg_combine<Op>(v, carry);
        if (inside) {
            chunk[idx] = v;
        }
        carry.val = __shfl_sync(kAll, v.val, Forward ? WARP_SIZE - 1 : 0);
        carry.idx = __shfl_sync(kAll, v.idx, Forward ? WARP_SIZE - 1 : 0);
    }
}

/**
 * In-place argreduce scan of one @p k-tap chunk -> single thraeded
 *
 *
 * @tparam Op       Operation policy (@ref ErodeOp or @ref DilateOp).
 * @tparam acc_t    Accumulate type.
 * @tparam Forward  true scans from the chunk's first tap, false from its last.
 * @param chunk     Shared-memory base of the chunk; overwritten with the scan.
 * @param k         Taps in the chunk.
 */
template <typename Op, typename acc_t, bool Forward>
__device__ __forceinline__ void arg_scan_chunk_serial(ArgTap<acc_t>* chunk, const int64_t k) {
    ArgTap<acc_t> acc = chunk[Forward ? 0 : k - 1];
    for (int64_t i = 1; i < k; ++i) {
        const int64_t idx = Forward ? i : k - 1 - i;
        acc = Forward ? arg_combine<Op>(acc, chunk[idx]) : arg_combine<Op>(chunk[idx], acc);
        chunk[idx] = acc;
    }
}

} // namespace serron

#endif // SERRON_MORPHOLOGY_SCAN_CUH
