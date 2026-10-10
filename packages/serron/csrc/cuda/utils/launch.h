#ifndef SERRON_CUDA_LAUNCH_H
#define SERRON_CUDA_LAUNCH_H

#include <cuda_runtime.h>

#include <compat/check.h>

#include <cstddef>
#include <type_traits>

namespace serron {

/**
 * Launches @p kernel on @p stream and throws if the launch is rejected.
 *
 * Each argument is converted to the kernel's own parameter type before its address goes into the argument array, so
 * the launch copies exactly the bytes the kernel reads. Launch failures are reported asynchronously, so this only
 * catches configuration errors (bad grid or block size, too much shared memory); a fault inside the kernel surfaces
 * at the next synchronisation.
 *
 * @tparam Params  Parameter types of @p kernel.
 * @param kernel   Host-side handle of an @c extern "C" @c __global__ entry point.
 * @param grid     Grid dimensions.
 * @param block    Block dimensions.
 * @param smem     Dynamic shared memory per block, in bytes.
 * @param stream   Stream to launch on.
 * @param args     Kernel arguments, one per entry in @p Params.
 */
template <typename... Params>
void launch_kernel(void (*kernel)(Params...), const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream,
                   std::type_identity_t<Params>... args) {
    void* arg_ptrs[] = {static_cast<void*>(&args)...};
    const cudaError_t status =
        cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block, arg_ptrs, smem, stream);
    if (status != cudaSuccess) {
        // Clear the error so it does not resurface at torch's next cudaGetLastError.
        (void)cudaGetLastError();
        SERRON_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(status));
    }
}

} // namespace serron

#endif // SERRON_CUDA_LAUNCH_H
