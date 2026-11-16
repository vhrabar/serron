#ifndef SERRON_CUDA_DEVICE_RUNTIME_H
#define SERRON_CUDA_DEVICE_RUNTIME_H

/**
 * The GPU runtime header for device code: the HIP one when compiled for AMD (hipcc or hipRTC, which define @c __HIP__),
 * the CUDA one under nvcc.
 */

#if defined(__HIP__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#endif // SERRON_CUDA_DEVICE_RUNTIME_H
