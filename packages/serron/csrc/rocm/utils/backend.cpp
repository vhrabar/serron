// The HIP runtime's side of gpu/utils/backend.h.

#include <gpu/utils/backend.h>
#include <rocm/utils/runtime.hip.h>

#include <compat/check.h>

#include <cstdint>

namespace serron {

namespace {

constexpr int kMaxDevices = 64;

} // namespace

ContextGuard::ContextGuard(const int device_index) {
    SERRON_CHECK(device_index >= 0 && device_index < kMaxDevices, "serron: device index ", device_index,
                 " out of range");
    // HIP keeps a current device per thread and no separate context, so this only has to make the right device current.
    const rocm::Runtime& hip = rocm::runtime();
    int current = 0;
    rocm::check_hip(hip.hipGetDevice(&current), "hipGetDevice");
    previous_ = reinterpret_cast<void*>(static_cast<std::intptr_t>(current));
    if (current != device_index) {
        rocm::check_hip(hip.hipSetDevice(device_index), "hipSetDevice");
        restore_ = true;
    }
}

ContextGuard::~ContextGuard() {
    if (restore_) {
        static_cast<void>(rocm::runtime().hipSetDevice(static_cast<int>(reinterpret_cast<std::intptr_t>(previous_))));
    }
}

Library kernel_library(const KernelFile file) {
    // The images are built and embedded in a later change; until then there is nothing to load.
    SERRON_CHECK(false, "serron: this build has no ROCm kernel images (kernel file ", static_cast<int>(file), ")");
    return nullptr;
}

KernelHandle find_kernel(const Library library, const char* name) {
    hipFunction_t kernel = nullptr;
    rocm::check_hip(rocm::runtime().hipModuleGetFunction(&kernel, static_cast<hipModule_t>(library), name), name);
    return kernel;
}

void launch_kernel_raw(const KernelHandle kernel, const Dim3 grid, const Dim3 block, const size_t smem,
                       const Stream stream, void** args) {
    rocm::check_hip(rocm::runtime().hipModuleLaunchKernel(static_cast<hipFunction_t>(kernel), grid.x, grid.y, grid.z,
                                                          block.x, block.y, block.z, static_cast<unsigned int>(smem),
                                                          static_cast<hipStream_t>(stream), args, nullptr),
                    "HIP kernel launch");
}

size_t smem_budget() {
    const rocm::Runtime& hip = rocm::runtime();
    int device = 0;
    rocm::check_hip(hip.hipGetDevice(&device), "hipGetDevice");
    static size_t cached[kMaxDevices] = {};
    SERRON_CHECK(device >= 0 && device < kMaxDevices, "serron: device index ", device, " out of range");
    if (cached[device] == 0) {
        int bytes = 0;
        rocm::check_hip(hip.hipDeviceGetAttribute(&bytes, hipDeviceAttributeMaxSharedMemoryPerBlock, device),
                        "shared-memory attribute query");
        cached[device] = static_cast<size_t>(bytes);
    }
    return cached[device];
}

bool set_max_dynamic_smem(KernelHandle /*kernel*/, const size_t bytes) {
    // A launch can ask for dynamic shared memory up to the device limit with nothing to opt in to.
    return bytes <= smem_budget();
}

void copy_to_host_async(void* host, const void* device, const size_t bytes, const Stream stream, const char* what) {
    rocm::check_hip(
        rocm::runtime().hipMemcpyDtoHAsync(host, const_cast<void*>(device), bytes, static_cast<hipStream_t>(stream)),
        what);
}

void synchronize(const Stream stream, const char* what) {
    rocm::check_hip(rocm::runtime().hipStreamSynchronize(static_cast<hipStream_t>(stream)), what);
}

} // namespace serron
