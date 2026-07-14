#ifndef SERRON_BOUNDARIES_CUH
#define SERRON_BOUNDARIES_CUH

#include <cuda/std/cstdint>

#include <cuda_runtime.h>

#include <torch/enum.h>

/**
 * Boundary handling for out-of-image neighbourhood samples.
 */
enum BorderMode : int {
    kReflect = 0,   ///< mirror without repeating the edge sample (…cb|abcd|cb…)
    kReplicate = 1, ///< clamp to the edge sample (…aa|abcd|dd…)
    kConstant = 2,  ///< out-of-bounds resolved to the neutral element by the caller
};

/**
 * Map a possibly out-of-range coordinate back into [0, size) per @p border.
 *
 * @param coord  In/out coordinate.
 * @param size   Extent of the axis being indexed (must be > 0).
 * @param border Boundary mode (@ref BorderMode).
 * @return       true if @p coord now holds a valid in-range index; false for @ref kConstant when the coordinate lies
 * outside the image.
 */
__device__ __forceinline__ bool resolve_coord(cuda::std::int64_t& coord, const int64_t size, const int border) {
    if (coord >= 0 && coord < size)
        return true;

    switch (border) {
    case kReplicate:
        coord = coord < 0 ? 0 : size - 1;
        return true;
    case kReflect: {
        if (size == 1) {
            coord = 0;
            return true;
        }
        const int64_t period = 2 * size - 2;
        int64_t m = coord % period;
        if (m < 0) {
            m += period;
        }
        coord = m < size ? m : period - m;
        return true;
    }
    case kConstant:
    default:
        return false;
    }
}

#endif // SERRON_BOUNDARIES_CUH
