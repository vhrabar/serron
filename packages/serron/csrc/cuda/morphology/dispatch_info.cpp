#include "ops.h"

#include <cuda/morphology/algorithm.h>
#include <cuda/morphology/separable.h>
#include <cuda/morphology/thresholds.h>
#include <cuda/utils/declarations.h>
#include <cuda/utils/smem.h>

#include <compat/check.h>
#include <morph_algorithm.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdint>

namespace serron {

namespace {

/**
 * Describes a call to the algorithm-selection code, validating the arguments the way the operators do.
 *
 * @param backward  Whether the call is the backward pass, which stages more per element in the line passes and has its
 * own separable threshold.
 */
AlgorithmProblem describe_problem(const Tensor& input, const Tensor& kernel, const std::optional<bool>& flat,
                                  const bool backward, const char* name) {
    SERRON_CHECK(input.is_cuda(), name, ": input must be a CUDA tensor");
    SERRON_CHECK(kernel.is_cuda(), name, ": kernel must be a CUDA tensor");
    SERRON_CHECK(kernel.dim() == 2 || kernel.dim() == 3, name, ": kernel must be 2-D (kH, kW) or 3-D (C, kH, kW), got ",
                 kernel.dim(), "-D");

    const Tensor kernel_c = torch::stable::contiguous(kernel);
    const int64_t kH = kernel_c.size(kernel_c.dim() - 2);
    const int64_t kW = kernel_c.size(kernel_c.dim() - 1);
    SERRON_CHECK(kH > 0 && kW > 0, name, ": kernel spatial dims must be positive");

    const torch::stable::accelerator::DeviceGuard device_guard(input.get_device_index());
    const bool is_flat = resolve_flat(flat, kernel_c);
    const size_t budget = smem_budget();
    const int64_t min_k = separable_min_k(input.get_device_index(), input.scalar_type(), backward);
    AlgorithmProblem problem;
    THO_DISPATCH_V2(input.scalar_type(), "serron_describe_problem",
                    AT_WRAP([&] { problem = make_problem<scalar_t>(kH, kW, is_flat, budget, min_k, backward); }),
                    torch::headeronly::ScalarType::Float, torch::headeronly::ScalarType::Double,
                    torch::headeronly::ScalarType::Half, torch::headeronly::ScalarType::BFloat16);
    return problem;
}

} // namespace

int64_t feasible_algos(const Tensor& input, const Tensor& kernel, const std::optional<bool>& flat,
                       const bool backward) {
    const AlgorithmProblem problem = describe_problem(input, kernel, flat, backward, "serron::feasible_algos");
    int64_t mask = 0;
    for (const Algorithm algorithm : {Algorithm::kElement, Algorithm::kTiled, Algorithm::kSeparable}) {
        if (is_feasible(algorithm, problem)) {
            mask |= int64_t{1} << static_cast<int>(algorithm);
        }
    }
    return mask;
}

int64_t heuristic_algo(const Tensor& input, const Tensor& kernel, const std::optional<bool>& flat,
                       const bool backward) {
    const AlgorithmProblem problem = describe_problem(input, kernel, flat, backward, "serron::heuristic_algo");
    return static_cast<int64_t>(heuristic_algorithm(problem));
}

int64_t set_separable_min_k_op(const Tensor& like, const bool backward, const int64_t min_k) {
    SERRON_CHECK(like.is_cuda(), "serron::set_separable_min_k: like must be a CUDA tensor");
    set_separable_min_k(like.get_device_index(), like.scalar_type(), backward, min_k);
    return separable_min_k(like.get_device_index(), like.scalar_type(), backward);
}

int64_t get_separable_min_k_op(const Tensor& like, const bool backward) {
    SERRON_CHECK(like.is_cuda(), "serron::get_separable_min_k: like must be a CUDA tensor");
    return separable_min_k(like.get_device_index(), like.scalar_type(), backward);
}

} // namespace serron
