#ifndef SERRON_GPU_BACKEND_H
#define SERRON_GPU_BACKEND_H

#include <cstddef>

/**
 * What the GPU host code needs from a backend, with no CUDA or HIP type in sight.
 *
 * The shared code launches kernels and queries the device through these names; the CUDA driver implements them in
 * csrc/cuda and the HIP runtime in csrc/rocm. Handles are opaque pointers (a CUstream or hipStream_t, a CUlibrary or
 * hipModule_t, a CUkernel or hipFunction_t), and the backend casts them back.
 */

namespace serron {

/// A stream, as the backend's native handle.
using Stream = void*;

/// A loaded set of kernels, as the backend's native handle.
using Library = void*;

/// One kernel in a @ref Library, as the backend's native handle.
using KernelHandle = void*;

/// Grid or block dimensions of a launch.
struct Dim3 {
    unsigned int x = 1;
    unsigned int y = 1;
    unsigned int z = 1;

    constexpr Dim3(const unsigned int x_ = 1, const unsigned int y_ = 1, const unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

/**
 * Makes @p device_index's primary context current on this thread for the guard's lifetime, where the backend has such a
 * thing.
 *
 * Torch does not guarantee that a thread which has not touched the GPU yet has a context. When the current one already
 * belongs to @p device_index the guard does nothing.
 */
class ContextGuard {
  public:
    explicit ContextGuard(int device_index);
    ~ContextGuard();

    ContextGuard(const ContextGuard&) = delete;
    ContextGuard& operator=(const ContextGuard&) = delete;

  private:
    void* previous_ = nullptr;
    bool restore_ = false;
};

/**
 * Loads a kernel image embedded in the library.
 *
 * @param image  Start of the image, aligned to 8 bytes.
 * @param what   Name used in the error message.
 * @throws std::runtime_error  if the runtime rejects the image, for example when it holds no code this GPU can run.
 */
Library load_kernel_library(const void* image, const char* what);

/**
 * Looks up the kernel called @p name in @p library.
 *
 * @throws std::runtime_error  if there is none.
 */
KernelHandle find_kernel(Library library, const char* name);

/**
 * Launches @p kernel on @p stream and throws if the launch is rejected.
 *
 * @param args  One pointer per kernel parameter, each to the value to pass.
 */
void launch_kernel_raw(KernelHandle kernel, Dim3 grid, Dim3 block, size_t smem, Stream stream, void** args);

/// Dynamic shared memory, in bytes, that one block can be given on the current device.
size_t smem_budget();

/**
 * Raises @p kernel's dynamic shared-memory ceiling on the current device to @p bytes.
 *
 * @return false if the runtime refuses.
 */
bool set_max_dynamic_smem(KernelHandle kernel, size_t bytes);

/// Copies @p bytes from device memory to @p host on @p stream, without waiting for the copy.
void copy_to_host_async(void* host, const void* device, size_t bytes, Stream stream, const char* what);

/// Waits for everything queued on @p stream.
void synchronize(Stream stream, const char* what);

} // namespace serron

#endif // SERRON_GPU_BACKEND_H
