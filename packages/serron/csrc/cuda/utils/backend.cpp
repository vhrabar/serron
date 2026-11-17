// The CUDA driver's side of gpu/utils/backend.h.

#include <cuda/utils/driver.h>
#include <gpu/utils/backend.h>

#include <cuda.h>

#include <compat/check.h>
#include <torch/csrc/stable/accelerator.h>

#include <array>
#include <mutex>

namespace serron {

namespace {

constexpr int kMaxDevices = 64;

} // namespace

ContextGuard::ContextGuard(const int device_index) {
    SERRON_CHECK(device_index >= 0 && device_index < kMaxDevices, "serron: device index ", device_index,
                 " out of range");
    const Driver& cu = driver();
    const CUdevice device = device_handle(device_index);

    CUcontext previous = nullptr;
    check_cu(cu.cuCtxGetCurrent(&previous), "cuCtxGetCurrent");
    previous_ = previous;
    if (previous != nullptr) {
        CUdevice current = 0;
        if (cu.cuCtxGetDevice(&current) == CUDA_SUCCESS && current == device) {
            return;
        }
    }

    static std::mutex mutex;
    static std::array<CUcontext, kMaxDevices> primary{};
    CUcontext context = nullptr;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        if (primary[device_index] == nullptr) {
            check_cu(cu.cuDevicePrimaryCtxRetain(&primary[device_index], device), "cuDevicePrimaryCtxRetain");
        }
        context = primary[device_index];
    }
    check_cu(cu.cuCtxSetCurrent(context), "cuCtxSetCurrent");
    restore_ = true;
}

ContextGuard::~ContextGuard() {
    if (restore_) {
        driver().cuCtxSetCurrent(static_cast<CUcontext>(previous_));
    }
}

Library load_kernel_library(const void* image, const char* what) {
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

KernelHandle find_kernel(const Library library, const char* name) {
    CUkernel kernel = nullptr;
    check_cu(driver().cuLibraryGetKernel(&kernel, static_cast<CUlibrary>(library), name), name);
    return kernel;
}

void launch_kernel_raw(const KernelHandle kernel, const Dim3 grid, const Dim3 block, const size_t smem,
                       const Stream stream, void** args) {
    check_cu(driver().cuLaunchKernel(reinterpret_cast<CUfunction>(kernel), grid.x, grid.y, grid.z, block.x, block.y,
                                     block.z, static_cast<unsigned int>(smem), static_cast<CUstream>(stream), args,
                                     nullptr),
             "CUDA kernel launch");
}

size_t smem_budget() {
    const auto device = torch::stable::accelerator::getCurrentDeviceIndex();
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

bool set_max_dynamic_smem(const KernelHandle kernel, const size_t bytes) {
    const auto device = device_handle(static_cast<int>(torch::stable::accelerator::getCurrentDeviceIndex()));
    return driver().cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(bytes),
                                         static_cast<CUkernel>(kernel), device) == CUDA_SUCCESS;
}

void copy_to_host_async(void* host, const void* device, const size_t bytes, const Stream stream, const char* what) {
    check_cu(
        driver().cuMemcpyDtoHAsync(host, reinterpret_cast<CUdeviceptr>(device), bytes, static_cast<CUstream>(stream)),
        what);
}

void synchronize(const Stream stream, const char* what) {
    check_cu(driver().cuStreamSynchronize(static_cast<CUstream>(stream)), what);
}

} // namespace serron
