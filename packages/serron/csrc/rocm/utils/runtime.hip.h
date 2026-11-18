#ifndef SERRON_ROCM_RUNTIME_HIP_H
#define SERRON_ROCM_RUNTIME_HIP_H

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <gpu/utils/backend.h>

#include <hip/hip_runtime_api.h>

/**
 * Calls @c X(name) once per HIP function serron uses.
 *
 * The pointer type is the prototype's own (@c decltype), so a call is checked against the header it was built with.
 */
#define SERRON_HIP_FUNCTIONS(X)                                                                                        \
    X(hipInit)                                                                                                         \
    X(hipGetErrorString)                                                                                               \
    X(hipGetDevice)                                                                                                    \
    X(hipGetDeviceCount)                                                                                               \
    X(hipGetDevicePropertiesR0600)                                                                                     \
    X(hipRuntimeGetVersion)                                                                                            \
    X(hipSetDevice)                                                                                                    \
    X(hipDeviceGetAttribute)                                                                                           \
    X(hipModuleLoadData)                                                                                               \
    X(hipModuleGetFunction)                                                                                            \
    X(hipModuleLaunchKernel)                                                                                           \
    X(hipMemcpyDtoHAsync)                                                                                              \
    X(hipStreamSynchronize)

namespace serron::rocm {

/// Pointers to the HIP functions in @ref SERRON_HIP_FUNCTIONS.
struct Runtime {
#define SERRON_HIP_MEMBER(name) decltype(&::name) name = nullptr;
    SERRON_HIP_FUNCTIONS(SERRON_HIP_MEMBER)
#undef SERRON_HIP_MEMBER
};

/**
 * The runtime, opened and initialised on first use.
 *
 * @throws std::runtime_error  if libamdhip64 is not in the process and cannot be opened, lacks one of the functions, or
 * fails to initialise.
 */
const Runtime& runtime();

/// Throws if @p result is not hipSuccess, naming @p what and the runtime's description of the error.
void check_hip(hipError_t result, const char* what);

/**
 * Loads a code object or SPIR-V image from memory.
 *
 * @param image  Start of the image: a clang offload bundle or a code object.
 * @param what   Name used in the error message.
 * @throws std::runtime_error  if the runtime rejects the image.
 */
Library load_module(const void* image, const char* what);

} // namespace serron::rocm

#endif // SERRON_ROCM_RUNTIME_HIP_H
