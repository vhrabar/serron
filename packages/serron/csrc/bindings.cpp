#include "ops.h"
#include "registration.h"

#include <torch/library.h>

// Operator schemas, registered under the serron:: namespace.
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("erode(Tensor input, Tensor kernel, int border) -> Tensor");
    ops.def("dilate(Tensor input, Tensor kernel, int border) -> Tensor");
}

// CUDA implementations.
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("erode", &serron::erode);
    ops.impl("dilate", &serron::dilate);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)