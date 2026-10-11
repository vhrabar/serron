#ifndef SERRON_CUDA_SMEM_H
#define SERRON_CUDA_SMEM_H

#include <cuda/utils/driver.h>
#include <cuda/utils/launch.h>

#include <cuda.h>

#include <compat/check.h>
#include <torch/csrc/stable/accelerator.h>

#include <cstddef>

namespace serron {

/// Dynamic shared memory a block gets bvy def
constexpr size_t DEFAULT_SMEM_PER_BLOCK = 48 * 1024;

/**
 * Dynamic shared memory one block can be given on the current device.
 *
 */
inline size_t smem_budget() {
    const auto device = torch::stable::accelerator::getCurrentDeviceIndex();
    static constexpr int kMaxDevices = 64;
    static size_t cached[kMaxDevices] = {};
    SERRON_CHECK(device >= 0 && device < kMaxDevices, "serron: device index ", device, " out of range");
    if (cached[device] == 0) {
        int optin = 0;
        check_cu(driver().cuDeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                                               device_handle(static_cast<int>(device))),
                 "shared-memory attribute query");
        cached[device] = static_cast<size_t>(optin);
    }
    return cached[device];
}

/**
 * Raises @p kernel's dynamic shared-memory ceiling to @p bytes.
 *
 *
 * @tparam Fn      Function type of the entry point being configured.
 * @param kernel   Kernel whose ceiling is being raised.
 * @param bytes    Dynamic shared memory the launch will request.
 */
template <typename Fn>
bool configure_kernel_smem(const Kernel<Fn> kernel, const size_t bytes) {
    if (bytes <= DEFAULT_SMEM_PER_BLOCK) {
        return true;
    }
    if (bytes > smem_budget()) {
        return false;
    }
    // The ceiling is per device, so it is set for the current one on every launch that needs it.
    const auto device = device_handle(static_cast<int>(torch::stable::accelerator::getCurrentDeviceIndex()));
    return driver().cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(bytes),
                                         kernel.handle, device) == CUDA_SUCCESS;
}

} // namespace serron

#endif // SERRON_CUDA_SMEM_H
