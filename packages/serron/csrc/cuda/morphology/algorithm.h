#ifndef SERRON_CUDA_MORPHOLOGY_ALGORITHM_H
#define SERRON_CUDA_MORPHOLOGY_ALGORITHM_H

#include <cuda/morphology/separable.h>
#include <cuda/utils/declarations.h>

#include <compat/check.h>
#include <morph_algorithm.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace serron {

/// Shared memory the tiled kernels need: the input tile with its halo, plus the structuring element.
inline size_t tiled_smem_bytes(const int64_t kH, const int64_t kW, const size_t elem_size) {
    const int64_t tile_w = TILE_X + kW - 1;
    const int64_t tile_h = TILE_Y + kH - 1;
    return (static_cast<size_t>(tile_h * tile_w) + static_cast<size_t>(kH * kW)) * elem_size;
}

/**
 * Whether the tiled kernels can run: the tile fits in @p smem_budget.
 */
inline bool tiled_feasible(const int64_t kH, const int64_t kW, const size_t elem_size, const size_t smem_budget) {
    return tiled_smem_bytes(kH, kW, elem_size) <= smem_budget;
}

/**
 * Whether the separable kernels can run: the SE is flat and one scan chunk of the longer window fits in
 * @p smem_budget.
 *
 * @param line_elem_size  Bytes per staged element of the line passes (the value for the forward, the value and
 * its tap index for the backward).
 */
inline bool separable_feasible(const bool is_flat, const int64_t kH, const int64_t kW, const size_t line_elem_size,
                               const size_t smem_budget) {
    return is_flat && line_smem_bytes(1, std::max(kH, kW), line_elem_size) <= smem_budget;
}

/**
 * The library's own choice for a problem: separable when it is feasible and the window is long enough, otherwise
 * tiled if the tile fits, otherwise element-wise.
 */
inline Algorithm heuristic_algorithm(const bool is_flat, const int64_t kH, const int64_t kW, const size_t elem_size,
                                     const size_t line_elem_size, const size_t smem_budget) {
    if (use_separable_path(is_flat, kH, kW) && separable_feasible(is_flat, kH, kW, line_elem_size, smem_budget)) {
        return Algorithm::kSeparable;
    }
    return tiled_feasible(kH, kW, elem_size, smem_budget) ? Algorithm::kTiled : Algorithm::kElement;
}

/**
 * Turns the caller's request into the algorithm to launch.
 *
 * @param requested       Algorithm asked for; @ref Algorithm::kAuto uses @ref heuristic_algorithm.
 * @param is_flat         Whether the SE is flat. Only read for auto and separable, so a caller that forces
 * another algorithm need not compute it.
 * @param name            Qualified caller name used to prefix diagnostics.
 * @throws std::runtime_error  if a forced algorithm cannot run on this problem.
 */
inline Algorithm resolve_algorithm(const Algorithm requested, const bool is_flat, const int64_t kH, const int64_t kW,
                                   const size_t elem_size, const size_t line_elem_size, const size_t smem_budget,
                                   const char* name) {
    switch (requested) {
    case Algorithm::kAuto:
        return heuristic_algorithm(is_flat, kH, kW, elem_size, line_elem_size, smem_budget);
    case Algorithm::kElement:
        return requested;
    case Algorithm::kTiled:
        SERRON_CHECK(tiled_feasible(kH, kW, elem_size, smem_budget), name, ": algo 'tiled' needs ",
                     tiled_smem_bytes(kH, kW, elem_size), " bytes of shared memory for a ", kH, "x", kW,
                     " window, over the ", smem_budget, " available");
        return requested;
    case Algorithm::kSeparable:
        SERRON_CHECK(is_flat, name, ": algo 'separable' needs a flat structuring element");
        SERRON_CHECK(separable_feasible(is_flat, kH, kW, line_elem_size, smem_budget), name,
                     ": algo 'separable' does not fit in shared memory for a ", kH, "x", kW, " window");
        return requested;
    }
    return Algorithm::kElement;
}

/// Whether @p requested needs the flatness of the SE to be resolved.
inline bool algorithm_needs_flatness(const Algorithm requested) {
    return requested == Algorithm::kAuto || requested == Algorithm::kSeparable;
}

} // namespace serron

#endif // SERRON_CUDA_MORPHOLOGY_ALGORITHM_H
