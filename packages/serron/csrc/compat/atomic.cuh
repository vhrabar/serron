#ifndef SERRON_COMPAT_ATOMIC_CUH
#define SERRON_COMPAT_ATOMIC_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

namespace serron {

/**
 * Atomic add for the dtypes the operators dispatch over.
 *
 * Stands in for @c gpuAtomicAdd. ATen's version lives in @c ATen/cuda/Atomic.cuh, which is not part of the
 * stable ABI, and the header-only one in @c torch/headeronly/cuda/Atomic.h only exists from torch 2.14.
 *
 */
__device__ __forceinline__ void atomic_add(float* address, const float value) {
    atomicAdd(address, value);
}

__device__ __forceinline__ void atomic_add(double* address, const double value) {
    atomicAdd(address, value);
}

/// @c Half and @c __half share a size and layout, so only the pointer needs reinterpreting.
__device__ __forceinline__ void atomic_add(torch::headeronly::Half* address, const torch::headeronly::Half value) {
    atomicAdd(reinterpret_cast<__half*>(address), __float2half(static_cast<float>(value)));
}

/// @c BFloat16 and @c __nv_bfloat16 likewise.
__device__ __forceinline__ void atomic_add(torch::headeronly::BFloat16* address,
                                           const torch::headeronly::BFloat16 value) {
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(address), __float2bfloat16(static_cast<float>(value)));
}

} // namespace serron

#endif // SERRON_COMPAT_ATOMIC_CUH
