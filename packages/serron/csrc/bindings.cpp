#include "ops.h"
#include "registration.h"

#include <ATen/autocast_mode.h>
#include <torch/library.h>

// Operator schemas, registered under the serron:: namespace.
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("erode(Tensor input, Tensor kernel, int border) -> Tensor");
    ops.def("dilate(Tensor input, Tensor kernel, int border) -> Tensor");
    ops.def("erode_backward(Tensor grad_output, Tensor input, Tensor kernel, int border) -> (Tensor, Tensor)");
    ops.def("dilate_backward(Tensor grad_output, Tensor input, Tensor kernel, int border) -> (Tensor, Tensor)");
}

// CPU implementation
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CPU, ops) {
    ops.impl("erode", &serron::erode_cpu);
    ops.impl("dilate", &serron::dilate_cpu);
    ops.impl("erode_backward", &serron::erode_backward_cpu);
    ops.impl("dilate_backward", &serron::dilate_backward_cpu);
}

// CUDA implementations
#ifdef SERRON_WITH_CUDA
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("erode", &serron::erode);
    ops.impl("dilate", &serron::dilate);
    ops.impl("erode_backward", &serron::erode_backward);
    ops.impl("dilate_backward", &serron::dilate_backward);
}
#endif

namespace {

at::ScalarType morphology_exec_type(const at::Tensor& input, const at::Tensor& kernel, c10::DeviceType device_type) {
    return at::autocast::promote_type(at::autocast::get_lower_precision_fp_from_device_type(device_type), device_type,
                                      input, kernel);
}

at::Tensor erode_autocast(const at::Tensor& input, const at::Tensor& kernel, int64_t border) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::autocast_dispatch_keyset);
    const c10::DeviceType device_type = input.device().type();
    const at::ScalarType exec_type = morphology_exec_type(input, kernel, device_type);
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("serron::erode", "")
                         .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t)>();
    return op.call(at::autocast::cached_cast(exec_type, input, device_type),
                   at::autocast::cached_cast(exec_type, kernel, device_type), border);
}

at::Tensor dilate_autocast(const at::Tensor& input, const at::Tensor& kernel, int64_t border) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::autocast_dispatch_keyset);
    const c10::DeviceType device_type = input.device().type();
    const at::ScalarType exec_type = morphology_exec_type(input, kernel, device_type);
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("serron::dilate", "")
                         .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t)>();
    return op.call(at::autocast::cached_cast(exec_type, input, device_type),
                   at::autocast::cached_cast(exec_type, kernel, device_type), border);
}

} // namespace

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Autocast, ops) {
    ops.impl("erode", &erode_autocast);
    ops.impl("dilate", &dilate_autocast);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, AutocastCPU, ops) {
    ops.impl("erode", &erode_autocast);
    ops.impl("dilate", &dilate_autocast);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)