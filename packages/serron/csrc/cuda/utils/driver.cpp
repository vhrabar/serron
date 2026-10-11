// SPDX-License-Identifier: BSD-3-Clause
// Portions derived from PyTorch (c10/cuda/driver_api.cpp)
// Copyright (c) PyTorch contributors
// see THIRD_PARTY_NOTICES for full copyright and license text.

#include <cuda/utils/driver.h>

#include <compat/check.h>

#include <array>
#include <mutex>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define SERRON_DRIVER_STR_(x) #x
/// Stringizes @p x after expanding it, so a cuda.h alias yields the symbol it maps to.
#define SERRON_DRIVER_STR(x) SERRON_DRIVER_STR_(x)

namespace serron {

namespace {

#if defined(_WIN32)
constexpr const char* kDriverLibrary = "nvcuda.dll";

void* open_driver(std::string& error) {
    HMODULE module = LoadLibraryExA(kDriverLibrary, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (module == nullptr) {
        error = "error " + std::to_string(GetLastError());
    }
    return reinterpret_cast<void*>(module);
}

void* find_symbol(void* library, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
}
#else
constexpr const char* kDriverLibrary = "libcuda.so.1";

void* open_driver(std::string& error) {
    // torch has already loaded the driver through its CUDA runtime, so this returns that same instance.
    void* library = dlopen(kDriverLibrary, RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        const char* message = dlerror();
        error = message != nullptr ? message : "unknown error";
    }
    return library;
}

void* find_symbol(void* library, const char* name) {
    return dlsym(library, name);
}
#endif

/// Opens the driver and resolves every function in SERRON_DRIVER_FUNCTIONS; throws on any failure.
Driver load_driver() {
    std::string error;
    void* library = open_driver(error);
    SERRON_CHECK(library != nullptr, "serron: could not load the CUDA driver (", kDriverLibrary, "): ", error,
                 ". Is an NVIDIA driver installed?");

    Driver loaded;
#define SERRON_RESOLVE(name)                                                                                           \
    loaded.name = reinterpret_cast<decltype(loaded.name)>(find_symbol(library, SERRON_DRIVER_STR(name)));              \
    SERRON_CHECK(loaded.name != nullptr, "serron: the CUDA driver has no ", SERRON_DRIVER_STR(name),                   \
                 "; the driver is too old (serron needs one supporting CUDA ", kMinDriverVersion / 1000, ".",          \
                 (kMinDriverVersion % 1000) / 10, " or newer)");
    SERRON_DRIVER_FUNCTIONS(SERRON_RESOLVE)
#undef SERRON_RESOLVE

    SERRON_CHECK(loaded.cuDriverGetVersion(&loaded.version) == CUDA_SUCCESS,
                 "serron: could not query the CUDA driver version");
    SERRON_CHECK(loaded.version >= kMinDriverVersion, "serron: the CUDA driver supports CUDA ", loaded.version / 1000,
                 ".", (loaded.version % 1000) / 10, ", but serron needs ", kMinDriverVersion / 1000, ".",
                 (kMinDriverVersion % 1000) / 10, " or newer. Update the NVIDIA driver.");

    const CUresult init = loaded.cuInit(0);
    if (init != CUDA_SUCCESS) {
        const char* message = nullptr;
        loaded.cuGetErrorString(init, &message);
        SERRON_CHECK(false, "serron: cuInit failed: ", message != nullptr ? message : "unknown error");
    }
    return loaded;
}

constexpr int kMaxDevices = 64;

} // namespace

const Driver& driver() {
    static std::mutex mutex;
    static Driver loaded;
    static bool ready = false;

    const std::lock_guard<std::mutex> lock(mutex);
    if (!ready) {
        loaded = load_driver();
        ready = true;
    }
    return loaded;
}

void check_cu(const CUresult result, const char* what) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char* message = nullptr;
    driver().cuGetErrorString(result, &message);
    SERRON_CHECK(false, "serron: ", what, " failed: ", message != nullptr ? message : "unknown error", " (CUresult ",
                 static_cast<int>(result), ")");
}

CUdevice device_handle(const int device_index) {
    CUdevice device = 0;
    check_cu(driver().cuDeviceGet(&device, device_index), "cuDeviceGet");
    return device;
}

CurrentContextGuard::CurrentContextGuard(const int device_index) {
    SERRON_CHECK(device_index >= 0 && device_index < kMaxDevices, "serron: device index ", device_index,
                 " out of range");
    const Driver& cu = driver();
    device_ = device_handle(device_index);

    check_cu(cu.cuCtxGetCurrent(&previous_), "cuCtxGetCurrent");
    if (previous_ != nullptr) {
        CUdevice current = 0;
        if (cu.cuCtxGetDevice(&current) == CUDA_SUCCESS && current == device_) {
            return;
        }
    }

    static std::mutex mutex;
    static std::array<CUcontext, kMaxDevices> primary{};
    CUcontext context = nullptr;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        if (primary[device_index] == nullptr) {
            check_cu(cu.cuDevicePrimaryCtxRetain(&primary[device_index], device_), "cuDevicePrimaryCtxRetain");
        }
        context = primary[device_index];
    }
    check_cu(cu.cuCtxSetCurrent(context), "cuCtxSetCurrent");
    restore_ = true;
}

CurrentContextGuard::~CurrentContextGuard() {
    if (restore_) {
        driver().cuCtxSetCurrent(previous_);
    }
}

} // namespace serron
