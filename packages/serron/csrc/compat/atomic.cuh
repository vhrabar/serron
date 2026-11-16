// SPDX-License-Identifier: BSD-3-Clause
// Portions derived from PyTorch (torch/headeronly/cuda/Atomic.h)
// Copyright (c) PyTorch contributors
// see THIRD_PARTY_NOTICES for full copyright and license text.
#ifndef SERRON_COMPAT_ATOMIC_CUH
#define SERRON_COMPAT_ATOMIC_CUH

#include <cuda/utils/device_runtime.h>

namespace serron {

/**
 * Atomic @c *address += val for the accumulators the backward kernels scatter into.
 *
 * Stands in for @c gpuAtomicAdd from torch/headeronly/cuda/Atomic.h, which first ships with torch 2.14; the
 * extension builds against 2.13. The gradient buffers hold @c acc_type of the element type, so only float and double
 * are scattered into: Half and BFloat16 accumulate in float and are converted once at the end. There is no 16-bit
 * overload.
 *
 * On AMD, @c atomicAdd on float compiles to a compare-and-swap loop unless the program is built with
 * @c -munsafe-fp-atomics, which emits the hardware instruction but is only correct on coarse-grained memory (what
 * torch allocates). It is left at the compiler default until the backward pass has been measured on an MI300X.
 */
__device__ __forceinline__ void atomic_add(float* address, const float val) {
    atomicAdd(address, val);
}

__device__ __forceinline__ void atomic_add(double* address, const double val) {
    atomicAdd(address, val);
}

} // namespace serron

#endif // SERRON_COMPAT_ATOMIC_CUH
