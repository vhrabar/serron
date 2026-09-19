#ifndef SERRON_DECLARATIONS_CUH
#define SERRON_DECLARATIONS_CUH

/// Threads per block for the element-wise morphology launches.
constexpr int THREADS = 256;

/// Tiles for the element-wise morphology launches.
constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;

#endif // SERRON_DECLARATIONS_CUH
