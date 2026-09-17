#ifndef SERRON_DECLARATIONS_CUH
#define SERRON_DECLARATIONS_CUH

/// Threads per block for the element-wise morphology launches.
constexpr int THREADS = 256;

/// Tiles for the element-wise morphology launches.
constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;

/// Threads per block for the separable morphology line passes.
constexpr int LINE_TILE = 256;

/// Lanes per warp
constexpr int WARP_SIZE = 32;

/// Warps per separable line block
constexpr int LINE_WARPS = LINE_TILE / WARP_SIZE;

#endif // SERRON_DECLARATIONS_CUH
