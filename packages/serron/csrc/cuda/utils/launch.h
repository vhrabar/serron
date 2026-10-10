#ifndef SERRON_CUDA_LAUNCH_H
#define SERRON_CUDA_LAUNCH_H

#include <cuda_runtime.h>

#include <compat/check.h>

#include <cstddef>
#include <type_traits>

namespace serron {

/**
 * Handle of a kernel looked up in an embedded fatbin, tagged with the entry point's function type so a launch is
 * checked against its parameters at compile time.
 *
 * @tparam Fn  Function type of the entry point, one of the aliases in entry_points.h.
 */
template <typename Fn>
struct Kernel {
    cudaKernel_t handle = nullptr;
};

/**
 * Loads a fatbin embedded in the library. The result is context-independent: each device loads its own copy of the
 * code the first time a kernel from it is launched there.
 *
 * @param image  Start of the fatbin, aligned to 8 bytes.
 * @param what   Name used in the error message.
 * @throws std::runtime_error  if the runtime rejects the image, for example when it holds no code for this GPU.
 */
inline cudaLibrary_t load_kernel_library(const void* image, const char* what) {
    cudaLibrary_t library = nullptr;
    const cudaError_t status = cudaLibraryLoadData(&library, image, nullptr, nullptr, 0, nullptr, nullptr, 0);
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        SERRON_CHECK(false, "serron: loading the ", what, " kernels failed: ", cudaGetErrorString(status));
    }
    return library;
}

/**
 * Looks up the entry point @p name in @p library.
 *
 * @tparam Fn      Function type the entry point is declared with in entry_points.h.
 * @throws std::runtime_error  if the library has no kernel by that name.
 */
template <typename Fn>
Kernel<Fn> get_kernel(cudaLibrary_t library, const char* name) {
    Kernel<Fn> kernel;
    const cudaError_t status = cudaLibraryGetKernel(&kernel.handle, library, name);
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        SERRON_CHECK(false, "serron: kernel ", name, " not found: ", cudaGetErrorString(status));
    }
    return kernel;
}

/**
 * Launches @p kernel on @p stream and throws if the launch is rejected.
 *
 * Each argument is converted to the kernel's own parameter type before its address goes into the argument array, so
 * the launch copies exactly the bytes the kernel reads. Launch failures are reported asynchronously, so this only
 * catches configuration errors (bad grid or block size, too much shared memory); a fault inside the kernel surfaces
 * at the next synchronisation.
 *
 * @tparam Params  Parameter types of @p kernel.
 * @param kernel   Entry point to launch.
 * @param grid     Grid dimensions.
 * @param block    Block dimensions.
 * @param smem     Dynamic shared memory per block, in bytes.
 * @param stream   Stream to launch on.
 * @param args     Kernel arguments, one per entry in @p Params.
 */
template <typename... Params>
void launch_kernel(const Kernel<void(Params...)> kernel, const dim3 grid, const dim3 block, const size_t smem,
                   cudaStream_t stream, std::type_identity_t<Params>... args) {
    void* arg_ptrs[] = {static_cast<void*>(&args)...};
    const cudaError_t status =
        cudaLaunchKernel(reinterpret_cast<const void*>(kernel.handle), grid, block, arg_ptrs, smem, stream);
    if (status != cudaSuccess) {
        // Clear the error so it does not resurface at torch's next cudaGetLastError.
        (void)cudaGetLastError();
        SERRON_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(status));
    }
}

} // namespace serron

#endif // SERRON_CUDA_LAUNCH_H
