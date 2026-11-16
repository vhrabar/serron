#ifndef SERRON_GPU_SHUFFLE_CUH
#define SERRON_GPU_SHUFFLE_CUH

/**
 * Warp shuffles within a logical warp of @c WARP_SIZE lanes: @c shfl_up, @c shfl_down and @c shfl.
 *
 * The width is always passed. A CUDA warp is 32 lanes, but an AMD wave is 32 or 64 depending on the GPU, and a portable
 * code object cannot ask which, so the kernels work in logical warps of 32 and the width keeps the two halves of a
 * wave64 apart. Every lane of the logical warp must call these together.
 *
 * This header only picks the backend's implementation.
 */

#if defined(__HIP__)
#include <rocm/utils/shuffle.hip.h>
#else
#include <cuda/utils/shuffle.cuh>
#endif

#endif // SERRON_GPU_SHUFFLE_CUH
