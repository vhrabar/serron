#ifndef SERRON_CUDA_LAUNCH_H
#define SERRON_CUDA_LAUNCH_H

#include <cuda/utils/driver.h>

#include <cuda.h>

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
    CUkernel handle = nullptr;
};

/// Grid or block dimensions of a launch.
struct Dim3 {
    unsigned int x = 1;
    unsigned int y = 1;
    unsigned int z = 1;

    constexpr Dim3(const unsigned int x_ = 1, const unsigned int y_ = 1, const unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

/**
 * Loads a fatbin embedded in the library. The result is context-independent: each device loads its own copy of the
 * code the first time a kernel from it is launched there.
 *
 * @param image  Start of the fatbin, aligned to 8 bytes.
 * @param what   Name used in the error message.
 * @throws std::runtime_error  if the driver rejects the image, for example when it holds no code this GPU or driver
 * can run.
 */
inline CUlibrary load_kernel_library(const void* image, const char* what) {
    const Driver& cu = driver();
    CUlibrary library = nullptr;
    const CUresult result = cu.cuLibraryLoadData(&library, image, nullptr, nullptr, 0, nullptr, nullptr, 0);
    if (result == CUDA_ERROR_NO_BINARY_FOR_GPU || result == CUDA_ERROR_UNSUPPORTED_PTX_VERSION) {
        SERRON_CHECK(false, "serron: the ", what, " kernels were built with CUDA ", CUDA_VERSION / 1000, ".",
                     (CUDA_VERSION % 1000) / 10, " and the driver (CUDA ", cu.version / 1000, ".",
                     (cu.version % 1000) / 10, ") cannot run them on this GPU. Update the NVIDIA driver.");
    }
    check_cu(result, "loading the kernels (cuLibraryLoadData)");
    return library;
}

/**
 * Looks up the entry point @p name in @p library.
 *
 * @tparam Fn  Function type the entry point is declared with in entry_points.h.
 * @throws std::runtime_error  if the library has no kernel by that name.
 */
template <typename Fn>
Kernel<Fn> get_kernel(CUlibrary library, const char* name) {
    Kernel<Fn> kernel;
    check_cu(driver().cuLibraryGetKernel(&kernel.handle, library, name), name);
    return kernel;
}

/**
 * Launches @p kernel on @p stream and throws if the launch is rejected.
 *
 * Each argument is converted to the kernel's own parameter type before its address goes into the argument array, so
 * the launch copies exactly the bytes the kernel reads. Launch failures are reported asynchronously, so this only
 * catches configuration errors (bad grid or block size, too much shared memory); a fault inside the kernel surfaces
 * at the next synchronisation. Runs in the calling thread's current context (see @ref CurrentContextGuard).
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
void launch_kernel(const Kernel<void(Params...)> kernel, const Dim3 grid, const Dim3 block, const size_t smem,
                   CUstream stream, std::type_identity_t<Params>... args) {
    void* arg_ptrs[] = {static_cast<void*>(&args)...};
    check_cu(driver().cuLaunchKernel(reinterpret_cast<CUfunction>(kernel.handle), grid.x, grid.y, grid.z, block.x,
                                     block.y, block.z, static_cast<unsigned int>(smem), stream, arg_ptrs, nullptr),
             "CUDA kernel launch");
}

} // namespace serron

#endif // SERRON_CUDA_LAUNCH_H
