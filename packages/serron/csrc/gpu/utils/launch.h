#ifndef SERRON_GPU_LAUNCH_H
#define SERRON_GPU_LAUNCH_H

#include <gpu/utils/backend.h>

#include <torch/csrc/stable/accelerator.h>

#include <type_traits>

namespace serron {

/**
 * Handle of a kernel looked up in an embedded image, tagged with the entry point's function type so a launch is checked
 * against its parameters at compile time.
 *
 * @tparam Fn  Function type of the entry point, one of the aliases in entry_points.h.
 */
template <typename Fn>
struct Kernel {
    KernelHandle handle = nullptr;
};

/**
 * Looks up the entry point @p name in @p library.
 *
 * @tparam Fn  Function type the entry point is declared with in entry_points.h.
 * @throws std::runtime_error  if the library has no kernel by that name.
 */
template <typename Fn>
Kernel<Fn> get_kernel(const Library library, const char* name) {
    return Kernel<Fn>{find_kernel(library, name)};
}

/**
 * Launches @p kernel on @p stream and throws if the launch is rejected.
 *
 * Each argument is converted to the kernel's own parameter type before its address goes into the argument array, so the
 * launch copies exactly the bytes the kernel reads. Launch failures are reported asynchronously, so this only catches
 * configuration errors (bad grid or block size, too much shared memory); a fault inside the kernel surfaces at the next
 * synchronisation.
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
                   const Stream stream, std::type_identity_t<Params>... args) {
    void* arg_ptrs[] = {static_cast<void*>(&args)...};
    launch_kernel_raw(kernel.handle, grid, block, smem, stream, arg_ptrs);
}

/// The stream torch is currently using on @p device_index.
inline Stream current_stream(const int device_index) {
    return static_cast<Stream>(torch::stable::accelerator::getCurrentStream(device_index).nativeHandle());
}

} // namespace serron

#endif // SERRON_GPU_LAUNCH_H
