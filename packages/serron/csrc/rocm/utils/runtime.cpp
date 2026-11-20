#include <rocm/utils/runtime.hip.h>

#include <compat/check.h>
#include <dlfcn.h>

#include <fstream>
#include <mutex>
#include <string>

namespace serron::rocm {

namespace {

#define SERRON_HIP_STR_(x) #x
#define SERRON_HIP_STR(x) SERRON_HIP_STR_(x)

/// Names libamdhip64 goes by, newest first; ROCm releases bump the soname.
constexpr const char* kRuntimeNames[] = {"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so"};

/// Path of the libamdhip64 mapped into this process, or empty.
std::string mapped_runtime() {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        const auto name = line.find("libamdhip64");
        if (name != std::string::npos) {
            const auto path = line.rfind(' ', name);
            return line.substr(path == std::string::npos ? 0 : path + 1);
        }
    }
    return {};
}

/// The library torch loaded if there is one, else the first of the usual names the system can open.
void* open_runtime(std::string& error) {
    for (const char* name : kRuntimeNames) {
        if (void* loaded = dlopen(name, RTLD_NOW | RTLD_NOLOAD)) {
            return loaded;
        }
    }
    if (const std::string path = mapped_runtime(); !path.empty()) {
        if (void* loaded = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD)) {
            return loaded;
        }
    }
    for (const char* name : kRuntimeNames) {
        if (void* library = dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
            return library;
        }
    }
    const char* message = dlerror();
    error = message != nullptr ? message : "unknown error";
    return nullptr;
}

/// Opens the runtime and resolves every function in SERRON_HIP_FUNCTIONS; throws on any failure.
Runtime load_runtime() {
    std::string error;
    void* library = open_runtime(error);
    SERRON_CHECK(library != nullptr, "serron: could not load the HIP runtime (libamdhip64): ", error,
                 ". Is torch built for ROCm installed?");

    Runtime loaded;
#define SERRON_HIP_RESOLVE(name)                                                                                       \
    loaded.name = reinterpret_cast<decltype(loaded.name)>(dlsym(library, SERRON_HIP_STR(name)));                       \
    SERRON_CHECK(loaded.name != nullptr, "serron: the HIP runtime has no ", SERRON_HIP_STR(name),                      \
                 "; it is older than serron supports");
    SERRON_HIP_FUNCTIONS(SERRON_HIP_RESOLVE)
#undef SERRON_HIP_RESOLVE

    const hipError_t init = loaded.hipInit(0);
    SERRON_CHECK(init == hipSuccess, "serron: hipInit failed: ", loaded.hipGetErrorString(init));
    return loaded;
}

} // namespace

const Runtime& runtime() {
    static std::mutex mutex;
    static Runtime loaded;
    static bool ready = false;

    const std::lock_guard<std::mutex> lock(mutex);
    if (!ready) {
        loaded = load_runtime();
        ready = true;
    }
    return loaded;
}

void check_hip(const hipError_t result, const char* what) {
    if (result == hipSuccess) {
        return;
    }
    SERRON_CHECK(false, "serron: ", what, " failed: ", runtime().hipGetErrorString(result), " (hipError_t ",
                 static_cast<int>(result), ")");
}

Library load_module(const void* image, const char* what) {
    hipModule_t module = nullptr;
    const hipError_t result = runtime().hipModuleLoadData(&module, image);
    SERRON_CHECK(result == hipSuccess, "serron: could not load the ", what,
                 " kernels: ", runtime().hipGetErrorString(result), " (hipError_t ", static_cast<int>(result), ")");
    return module;
}

} // namespace serron::rocm
