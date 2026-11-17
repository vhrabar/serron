#ifndef SERRON_CUDA_DECLARATIONS_H
#define SERRON_CUDA_DECLARATIONS_H

/// Threads per block for the element-wise morphology launches.
constexpr int THREADS = 256;

/// Tiles for the element-wise morphology launches.
constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;

/// Threads per block for the separable morphology line passes.
constexpr int LINE_TILE = 256;

/**
 * Lanes per logical warp. This is the unit the separable scans work in, not the hardware width: an AMD wave can be 64
 * lanes, which the kernels treat as two logical warps (see gpu/utils/shuffle.cuh).
 */
constexpr int WARP_SIZE = 32;

/// Logical warps per separable line block
constexpr int LINE_WARPS = LINE_TILE / WARP_SIZE;

static_assert(LINE_TILE % WARP_SIZE == 0, "a line block must be a whole number of logical warps");

#endif // SERRON_CUDA_DECLARATIONS_H
