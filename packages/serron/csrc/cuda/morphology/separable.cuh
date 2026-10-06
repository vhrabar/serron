#ifndef SERRON_MORPHOLOGY_SEPARABLE_CUH
#define SERRON_MORPHOLOGY_SEPARABLE_CUH

#include <cuda/utils/declarations.cuh>

#include <cuda_runtime.h>

#include <compat/check.h>
#include <compat/flatness.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <vector>

namespace serron {

using torch::stable::Tensor;

/**
 * Window length at which the separable r+c path is preferred over the tiled/GMEM 2-D kernels, for a flat,
 * axis-separable structuring element.
 */
inline int64_t separable_min_k() {
    static const int64_t value = [] {
        if (const char* env = std::getenv("SERRON_SEPARABLE_MIN_K")) {
            char* end = nullptr;
            const long parsed = std::strtol(env, &end, 10);
            if (end != env && *end == '\0' && parsed > 0) {
                return static_cast<int64_t>(parsed);
            }
        }
        return static_cast<int64_t>(11);
    }();
    return value;
}

/**
 * Shared memory one van Herk / Gil-Werman line block needs
 */
inline size_t line_smem_bytes(const int64_t chunks, const int64_t k, const size_t elem_size) {
    return 2 * static_cast<size_t>(chunks * k) * elem_size;
}

/**
 * How many van Herk / Gil-Werman chunks of @p k samples one line block scans.
 *

 *
 * @return Chunks per block, or 0 when a single chunk already overflows @p smem_budget.
 */
inline int64_t line_chunks_per_block(const int64_t k, const int64_t line_len, const size_t elem_size,
                                     const size_t smem_budget) {
    const int64_t by_smem = static_cast<int64_t>(smem_budget / line_smem_bytes(1, k, elem_size));
    const int64_t by_line = (line_len + 2 * k - 2) / k;
    const int64_t by_outputs = (LINE_TILE + 2 * k - 2) / k;
    const int64_t by_scan = k < WARP_SIZE ? LINE_TILE : LINE_WARPS;
    return std::min({by_smem, by_line, std::max(by_scan, by_outputs)});
}

/// Throws when @p status reports a failure, naming @p what.
inline void check_cuda(const cudaError_t status, const char* what) {
    SERRON_CHECK(status == cudaSuccess, "serron: ", what, " failed: ", cudaGetErrorString(status));
}

/**
 * True when @p kernel_c is a flat SE.
 *
 * Copies the structuring element to the host and scans it there. The copy is issued on the current stream and
 * waited on: torch's non-default streams are created non-blocking, so a null-stream copy would not be ordered
 * after whatever produced @p kernel_c.
 *
 * @param kernel_c  Contiguous structuring element on a CUDA device.
 */
inline bool se_is_flat(const Tensor& kernel_c) {
    const int64_t n = kernel_c.numel();
    if (n == 0) {
        return true;
    }

    std::vector<char> host(static_cast<size_t>(n) * static_cast<size_t>(kernel_c.element_size()));
    const auto stream = static_cast<cudaStream_t>(
        torch::stable::accelerator::getCurrentStream(kernel_c.get_device_index()).nativeHandle());
    check_cuda(cudaMemcpyAsync(host.data(), kernel_c.const_data_ptr(), host.size(), cudaMemcpyDeviceToHost, stream),
               "structuring-element copy");
    check_cuda(cudaStreamSynchronize(stream), "structuring-element copy sync");

    bool flat = false;
    THO_DISPATCH_V2(kernel_c.scalar_type(), "serron_se_is_flat_cuda",
                    AT_WRAP([&] { flat = all_zero(reinterpret_cast<const scalar_t*>(host.data()), n); }),
                    torch::headeronly::ScalarType::Float, torch::headeronly::ScalarType::Double,
                    torch::headeronly::ScalarType::Half, torch::headeronly::ScalarType::BFloat16);
    return flat;
}

/**
 * Flatness of @p kernel_c
 *
 * @param flat      Caller's answer, or @c std::nullopt to check @p kernel_c directly.
 * @param kernel_c  Contiguous structuring element.
 */
inline bool resolve_flat(const std::optional<bool>& flat, const Tensor& kernel_c) {
    return flat.has_value() ? *flat : se_is_flat(kernel_c);
}

/**
 * True when @p is_flat and (@p kH, @p kW) justify the separable r+c path over the tiled/GMEM 2-D kernels, per @ref
 * separable_min_k.
 */
inline bool use_separable_path(const bool is_flat, const int64_t kH, const int64_t kW) {
    return is_flat && std::max(kH, kW) >= separable_min_k();
}

} // namespace serron

#endif // SERRON_MORPHOLOGY_SEPARABLE_CUH
