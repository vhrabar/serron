#ifndef SERRON_GPU_DEVICE_RUNTIME_H
#define SERRON_GPU_DEVICE_RUNTIME_H

/**
 * The GPU runtime header for device code. This header only picks the backend's: the HIP one when compiled for AMD
 * (hipcc or hipRTC, which define @c __HIP__), the CUDA one under nvcc.
 */

#if defined(__HIP__)
#include <rocm/utils/device_runtime.hip.h>
#else
#include <cuda/utils/device_runtime.h>
#endif

#endif // SERRON_GPU_DEVICE_RUNTIME_H
