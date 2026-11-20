// SPDX-License-Identifier: BSD-3-Clause
// Portions derived from PyTorch (c10/cuda/driver_api.h)
// Copyright (c) PyTorch contributors
// see THIRD_PARTY_NOTICES for full copyright and license text.
#ifndef SERRON_CUDA_DRIVER_H
#define SERRON_CUDA_DRIVER_H

#include <cuda.h>

#include <compat/check.h>

/**
 * The CUDA driver API, resolved at runtime.
 *
 */

/**
 * Calls @c X(name) once per driver function serron uses.
 *
 * Each name goes through cuda.h's macros before it is used, so both the pointer type (@c decltype of the prototype)
 * and the symbol looked up follow the header's own mapping.
 */
#define SERRON_DRIVER_FUNCTIONS(X)                                                                                     \
    X(cuInit)                                                                                                          \
    X(cuDriverGetVersion)                                                                                              \
    X(cuGetErrorString)                                                                                                \
    X(cuDeviceGet)                                                                                                     \
    X(cuDeviceGetAttribute)                                                                                            \
    X(cuCtxGetCurrent)                                                                                                 \
    X(cuCtxSetCurrent)                                                                                                 \
    X(cuCtxGetDevice)                                                                                                  \
    X(cuDevicePrimaryCtxRetain)                                                                                        \
    X(cuLibraryLoadData)                                                                                               \
    X(cuLibraryGetKernel)                                                                                              \
    X(cuKernelSetAttribute)                                                                                            \
    X(cuLaunchKernel)                                                                                                  \
    X(cuMemcpyDtoHAsync)                                                                                               \
    X(cuStreamSynchronize)

namespace serron {

/// Oldest driver the loader accepts ( The cuLibrary API arrived in 12.0.)
constexpr int kMinDriverVersion = 12000;

/// Pointers to the driver functions in @ref SERRON_DRIVER_FUNCTIONS
struct Driver {
#define SERRON_DRIVER_MEMBER(name) decltype(&::name) name = nullptr;
    SERRON_DRIVER_FUNCTIONS(SERRON_DRIVER_MEMBER)
#undef SERRON_DRIVER_MEMBER
    int version = 0;
};

/**
 * The driver, opened and initialised on first use.
 *
 * @throws std::runtime_error  if the driver library is missing, lacks one of the functions, is older than
 * @ref kMinDriverVersion, or fails to initialise.
 */
const Driver& driver();

/**
 * Throws if @p result is not CUDA_SUCCESS, naming @p what and the driver's description of the error.
 */
void check_cu(CUresult result, const char* what);

/// Driver handle of device @p device_index.
CUdevice device_handle(int device_index);

} // namespace serron

#endif // SERRON_CUDA_DRIVER_H
