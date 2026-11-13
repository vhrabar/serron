#ifndef SERRON_CUDA_MORPHOLOGY_ALGORITHM_H
#define SERRON_CUDA_MORPHOLOGY_ALGORITHM_H

#include <cuda/morphology/arg_tap.h>
#include <cuda/morphology/separable.h>
#include <cuda/utils/declarations.h>

#include <compat/acc_type.h>
#include <compat/check.h>
#include <morph_algorithm.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace serron {

/// What decides which algorithms can run a morphology call and which one the library prefers.
struct AlgorithmProblem {
    int64_t kH = 0;              ///< Structuring-element height.
    int64_t kW = 0;              ///< Structuring-element width.
    bool is_flat = false;        ///< Whether the structuring element is flat (all-zero).
    size_t elem_size = 0;        ///< Bytes per element of the input.
    size_t line_elem_size = 0;   ///< Bytes per element the line passes stage: the value for the forward, the
                                 ///< value and its tap index for the backward.
    size_t smem_budget = 0;      ///< Dynamic shared memory one block can be given.
    int64_t separable_min_k = 0; ///< Window length from which separable is preferred.
};

/// Bytes per element the separable backward stages in shared memory: a value and the index of the tap that produced it.
template <typename scalar_t>
constexpr size_t backward_line_elem_size() {
    return sizeof(ArgTap<acc_type<scalar_t, true>>);
}

/**
 * Describes a call on @c scalar_t to the selection code below.
 *
 * Built by a function rather than inline because the braces of an initializer list would split the arguments of the
 * dispatch macros that surround its callers.
 *
 * @param backward  Whether the call is the backward pass, whose line passes stage a value and its tap index.
 */
template <typename scalar_t>
AlgorithmProblem make_problem(const int64_t kH, const int64_t kW, const bool is_flat, const size_t smem_budget,
                              const int64_t separable_min_k, const bool backward) {
    AlgorithmProblem problem;
    problem.kH = kH;
    problem.kW = kW;
    problem.is_flat = is_flat;
    problem.elem_size = sizeof(scalar_t);
    problem.line_elem_size = backward ? backward_line_elem_size<scalar_t>() : sizeof(scalar_t);
    problem.smem_budget = smem_budget;
    problem.separable_min_k = separable_min_k;
    return problem;
}

/// Shared memory the tiled kernels need: the input tile with its halo, plus the structuring element.
inline size_t tiled_smem_bytes(const int64_t kH, const int64_t kW, const size_t elem_size) {
    const int64_t tile_w = TILE_X + kW - 1;
    const int64_t tile_h = TILE_Y + kH - 1;
    return (static_cast<size_t>(tile_h * tile_w) + static_cast<size_t>(kH * kW)) * elem_size;
}

/// Whether the tiled kernels can run: the tile fits in the shared-memory budget.
inline bool tiled_feasible(const int64_t kH, const int64_t kW, const size_t elem_size, const size_t smem_budget) {
    return tiled_smem_bytes(kH, kW, elem_size) <= smem_budget;
}

/// Whether the tiled kernels can run on @p problem.
inline bool tiled_feasible(const AlgorithmProblem& problem) {
    return tiled_feasible(problem.kH, problem.kW, problem.elem_size, problem.smem_budget);
}

/**
 * Whether the separable kernels can run on @p problem: the SE is flat and one scan chunk of the longer window fits
 * in the shared-memory budget.
 */
inline bool separable_feasible(const AlgorithmProblem& problem) {
    return problem.is_flat &&
           line_smem_bytes(1, std::max(problem.kH, problem.kW), problem.line_elem_size) <= problem.smem_budget;
}

/// Whether @p algorithm can run on @p problem. Auto is not an algorithm and is never feasible.
inline bool is_feasible(const Algorithm algorithm, const AlgorithmProblem& problem) {
    switch (algorithm) {
    case Algorithm::kAuto:
        return false;
    case Algorithm::kElement:
        return true;
    case Algorithm::kTiled:
        return tiled_feasible(problem);
    case Algorithm::kSeparable:
        return separable_feasible(problem);
    }
    return false;
}

/**
 * The library's own choice for @p problem: separable when it is feasible and the window is long enough, otherwise
 * tiled if the tile fits, otherwise element-wise.
 */
inline Algorithm heuristic_algorithm(const AlgorithmProblem& problem) {
    if (std::max(problem.kH, problem.kW) >= problem.separable_min_k && separable_feasible(problem)) {
        return Algorithm::kSeparable;
    }
    return tiled_feasible(problem) ? Algorithm::kTiled : Algorithm::kElement;
}

/**
 * Turns the caller's request into the algorithm to launch.
 *
 * @param requested  Algorithm asked for; @ref Algorithm::kAuto uses @ref heuristic_algorithm.
 * @param problem    The call. Its @c is_flat is only read for auto and separable, so a caller that forces another
 * algorithm need not compute it.
 * @param name       Qualified caller name used to prefix diagnostics.
 * @throws std::runtime_error  if a forced algorithm cannot run on this problem.
 */
inline Algorithm resolve_algorithm(const Algorithm requested, const AlgorithmProblem& problem, const char* name) {
    switch (requested) {
    case Algorithm::kAuto:
        return heuristic_algorithm(problem);
    case Algorithm::kElement:
        return requested;
    case Algorithm::kTiled:
        SERRON_CHECK(tiled_feasible(problem), name, ": algo 'tiled' needs ",
                     tiled_smem_bytes(problem.kH, problem.kW, problem.elem_size), " bytes of shared memory for a ",
                     problem.kH, "x", problem.kW, " window, over the ", problem.smem_budget, " available");
        return requested;
    case Algorithm::kSeparable:
        SERRON_CHECK(problem.is_flat, name, ": algo 'separable' needs a flat structuring element");
        SERRON_CHECK(separable_feasible(problem), name, ": algo 'separable' does not fit in shared memory for a ",
                     problem.kH, "x", problem.kW, " window");
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
