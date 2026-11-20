// The HIP runtime's side of gpu/utils/backend.h.

#include <gpu/utils/backend.h>
#include <rocm/utils/images.h>
#include <rocm/utils/runtime.hip.h>

#include <compat/check.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <span>
#include <string>
#include <string_view>

namespace serron {

namespace {

constexpr int kMaxDevices = 64;

constexpr std::string_view kPortableTarget = "amdgcnspirv";

/// The architecture of device @p index without its feature flags
std::string device_architecture(const rocm::Runtime& hip, const int index) {
    hipDeviceProp_tR0600 properties{};
    rocm::check_hip(hip.hipGetDevicePropertiesR0600(&properties, index), "hipGetDeviceProperties");
    const std::string architecture(properties.gcnArchName);
    return architecture.substr(0, architecture.find(':'));
}

bool force_portable() {
    const char* value = std::getenv("SERRON_ROCM_FORCE_SPIRV");
    return value != nullptr && *value != '\0' && std::string_view(value) != "0";
}

/// "7.15.26333" for the runtime's version number, which is major * 10000000 + minor * 100000 + patch.
std::string runtime_version(const rocm::Runtime& hip) {
    int version = 0;
    if (hip.hipRuntimeGetVersion(&version) != hipSuccess) {
        return "unknown";
    }
    return std::to_string(version / 10000000) + "." + std::to_string(version / 100000 % 100) + "." +
           std::to_string(version % 100000);
}

/// What was decided for one kernel file, with the architecture the decision was made for.
struct Choice {
    const rocm::EmbeddedImage* image = nullptr;
    std::string architecture;
};

Choice choose_image(const KernelFile file) {
    const rocm::Runtime& hip = rocm::runtime();
    int count = 0;
    rocm::check_hip(hip.hipGetDeviceCount(&count), "hipGetDeviceCount");
    SERRON_CHECK(count > 0, "serron: no AMD GPU is visible");

    const std::string architecture = device_architecture(hip, 0);
    bool all_alike = true;
    for (int device = 1; device < count && all_alike; ++device) {
        all_alike = device_architecture(hip, device) == architecture;
    }

    const std::span<const rocm::EmbeddedImage> images = rocm::embedded_images(file);
    const rocm::EmbeddedImage* portable = nullptr;
    const rocm::EmbeddedImage* native = nullptr;
    for (const rocm::EmbeddedImage& image : images) {
        if (std::string_view(image.target) == kPortableTarget) {
            portable = &image;
        } else if (image.target == architecture) {
            native = &image;
        }
    }
    const bool use_native = native != nullptr && all_alike && !force_portable();
    const rocm::EmbeddedImage* image = use_native ? native : portable;
    SERRON_CHECK(image != nullptr, "serron: this build has no ROCm kernel image for ", architecture);
    return {image, architecture};
}

/// The choice for @p file, made on first use and kept.
const Choice& chosen(const KernelFile file) {
    static const Choice forward = choose_image(KernelFile::kForward);
    static const Choice backward = choose_image(KernelFile::kBackward);
    switch (file) {
    case KernelFile::kForward:
        return forward;
    case KernelFile::kBackward:
        return backward;
    }
    SERRON_CHECK(false, "serron: unknown kernel file ", static_cast<int>(file));
    return forward;
}

Library load_chosen(const KernelFile file) {
    const auto& [image, architecture] = chosen(file);
    const std::string what =
        std::string(file == KernelFile::kForward ? "forward" : "backward") + " (" + image->target + " image)";
    try {
        return rocm::load_module(image->data(), what.c_str());
    } catch (const std::exception& error) {
        SERRON_CHECK(false, error.what(), "; GPU architecture ", architecture, ", HIP runtime ",
                     runtime_version(rocm::runtime()), ". SERRON_ROCM_FORCE_SPIRV=1 loads the portable image instead.");
        throw;
    }
}

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
    switch (file) {
    case KernelFile::kForward: {
        static const Library library = load_chosen(KernelFile::kForward);
        return library;
    }
    case KernelFile::kBackward: {
        static const Library library = load_chosen(KernelFile::kBackward);
        return library;
    }
    }
    SERRON_CHECK(false, "serron: unknown kernel file ", static_cast<int>(file));
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

namespace rocm {

const char* selected_image_target(const KernelFile file) {
    return chosen(file).image->target;
}

} // namespace rocm

} // namespace serron
