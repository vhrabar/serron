#include "ops.h"
#include "registration.h"

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

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)