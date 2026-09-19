#ifndef SERRON_SMEM_CUH
#define SERRON_SMEM_CUH

#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>

#include <cstddef>

namespace serron {

/// Dynamic shared memory a block gets bvy def
constexpr size_t DEFAULT_SMEM_PER_BLOCK = 48 * 1024;

/**
 * Dynamic shared memory one block can be given on the current device.
 *
 */
inline size_t smem_budget() {
    return at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
}

/**
 * Raises @p kernel's dynamic shared-memory ceiling to @p bytes.
 *
 *
 * @tparam KernelFn  Type of the @c __global__ function being configured.
 * @param kernel     Kernel whose ceiling is being raised.
 * @param bytes      Dynamic shared memory the launch will request.
 */
template <typename KernelFn>
bool configure_kernel_smem(KernelFn kernel, const size_t bytes) {
    if (bytes <= DEFAULT_SMEM_PER_BLOCK) {
        return true;
    }
    if (bytes > smem_budget()) {
        return false;
    }
    const cudaError_t status = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return true;
}

} // namespace serron

#endif // SERRON_SMEM_CUH
