#ifndef SERRON_COMPAT_ATOMIC_CUH
#define SERRON_COMPAT_ATOMIC_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cstddef>

namespace serron {

/**
 * Atomic @c *address += val for the dtypes the backward kernels scatter into.
 *
 * Stands in for @c gpuAtomicAdd from torch/headeronly/cuda/Atomic.h, which first ships with torch 2.14; the
 * extension builds against 2.13. Covers the architectures in CMAKE_CUDA_ARCHITECTURES (sm_75 and newer): float,
 * double and Half use the native instruction, BFloat16 does from sm_80 and falls back to a compare-and-swap on the
 * enclosing 32-bit word below that.
 */
__device__ __forceinline__ void atomic_add(float* address, const float val) {
    atomicAdd(address, val);
}

__device__ __forceinline__ void atomic_add(double* address, const double val) {
    atomicAdd(address, val);
}

__device__ __forceinline__ void atomic_add(torch::headeronly::Half* address, const torch::headeronly::Half val) {
    atomicAdd(reinterpret_cast<__half*>(address), __ushort_as_half(val.x));
}

__device__ __forceinline__ void atomic_add(torch::headeronly::BFloat16* address,
                                           const torch::headeronly::BFloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    const bool high = (reinterpret_cast<size_t>(address) & 2) != 0;
    auto* word = reinterpret_cast<unsigned int*>(reinterpret_cast<char*>(address) - (high ? 2 : 0));
    unsigned int old = *word;
    unsigned int assumed = 0;
    do {
        assumed = old;
        torch::headeronly::BFloat16 sum;
        sum.x = static_cast<uint16_t>(high ? old >> 16 : old & 0xffffu);
        sum = sum + val;
        const unsigned int updated =
            high ? (old & 0xffffu) | (static_cast<unsigned int>(sum.x) << 16) : (old & 0xffff0000u) | sum.x;
        old = atomicCAS(word, assumed, updated);
    } while (assumed != old);
#else
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(address), __ushort_as_bfloat16(val.x));
#endif
}

} // namespace serron

#endif // SERRON_COMPAT_ATOMIC_CUH
