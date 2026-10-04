#ifndef SERRON_COMPAT_CUDA_LAUNCH_CHECK_CUH
#define SERRON_COMPAT_CUDA_LAUNCH_CHECK_CUH

#include <cuda_runtime.h>

#include <torch/headeronly/util/Exception.h>

/**
 * Throws if the most recent kernel launch failed.
 *
 * Stands in for @c C10_CUDA_KERNEL_LAUNCH_CHECK, which lives in c10/cuda/CUDAException.h. Launch failures are
 * reported asynchronously, so this only catches configuration errors (bad grid or block size, too much shared
 * memory); a fault inside the kernel surfaces at the next synchronisation.
 */
#define SERRON_CUDA_KERNEL_LAUNCH_CHECK()                                                                            \
    do {                                                                                                             \
        const cudaError_t serron_launch_status_ = cudaGetLastError();                                                 \
        STD_TORCH_CHECK(serron_launch_status_ == cudaSuccess, "CUDA kernel launch failed: ",                          \
                        cudaGetErrorString(serron_launch_status_));                                                   \
    } while (0)

#endif // SERRON_COMPAT_CUDA_LAUNCH_CHECK_CUH
