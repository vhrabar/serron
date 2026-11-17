#ifndef SERRON_GPU_SMEM_H
#define SERRON_GPU_SMEM_H

#include <gpu/utils/backend.h>
#include <gpu/utils/launch.h>

#include <cstddef>

namespace serron {

/// Dynamic shared memory a block gets by default.
constexpr size_t DEFAULT_SMEM_PER_BLOCK = 48 * 1024;

/**
 * Raises @p kernel's dynamic shared-memory ceiling to @p bytes.
 *
 * @tparam Fn      Function type of the entry point being configured.
 * @param kernel   Kernel whose ceiling is being raised.
 * @param bytes    Dynamic shared memory the launch will request.
 * @return false if @p bytes is over what the device allows or the runtime refuses.
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
    return set_max_dynamic_smem(kernel.handle, bytes);
}

} // namespace serron

#endif // SERRON_GPU_SMEM_H
