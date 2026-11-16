#ifndef SERRON_CUDA_SHUFFLE_CUH
#define SERRON_CUDA_SHUFFLE_CUH

#include <cuda/utils/declarations.h>
#include <cuda/utils/device_runtime.h>

namespace serron {

/// Value of @p v from the lane @p delta below the caller's; a lane with no such lane gets its own value back.
template <typename T>
__device__ __forceinline__ T shfl_up(const T v, const unsigned delta) {
    return __shfl_up_sync(0xffffffffu, v, delta, WARP_SIZE);
}

/// Value of @p v from the lane @p delta above the caller's; a lane with no such lane gets its own value back.
template <typename T>
__device__ __forceinline__ T shfl_down(const T v, const unsigned delta) {
    return __shfl_down_sync(0xffffffffu, v, delta, WARP_SIZE);
}

/// Value of @p v from lane @p src_lane of the caller's logical warp.
template <typename T>
__device__ __forceinline__ T shfl(const T v, const int src_lane) {
    return __shfl_sync(0xffffffffu, v, src_lane, WARP_SIZE);
}

} // namespace serron

#endif // SERRON_CUDA_SHUFFLE_CUH
