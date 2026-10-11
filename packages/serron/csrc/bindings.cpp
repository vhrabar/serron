#include "ops.h"
#include "registration.h"

#include <torch/csrc/stable/library.h>

// Operator schemas, registered under the serron:: namespace.
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("erode(Tensor input, Tensor kernel, int border, bool? flat=None) -> Tensor");
    ops.def("dilate(Tensor input, Tensor kernel, int border, bool? flat=None) -> Tensor");
    ops.def("erode_backward(Tensor grad_output, Tensor input, Tensor kernel, int border, bool? flat=None, "
            "bool need_kernel_grad=True) -> (Tensor, Tensor)");
    ops.def("dilate_backward(Tensor grad_output, Tensor input, Tensor kernel, int border, bool? flat=None, "
            "bool need_kernel_grad=True) -> (Tensor, Tensor)");
}

// CPU implementation
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CPU, ops) {
    ops.impl("erode", TORCH_BOX(&serron::erode_cpu));
    ops.impl("dilate", TORCH_BOX(&serron::dilate_cpu));
    ops.impl("erode_backward", TORCH_BOX(&serron::erode_backward_cpu));
    ops.impl("dilate_backward", TORCH_BOX(&serron::dilate_backward_cpu));
}

// CUDA implementations are registered by the separate _C_cuda library (cuda/bindings.cpp), so this
// library links only what the stable shim needs and loads on CPU-only torch builds.

// Autocast is registered from Python, in torch-ext/serron/_cmake_ops.py: the stable ABI has no equivalent of
// at::autocast, so torch.library.register_autocast stands in for the Autocast / AutocastCPU kernels that used to
// live here.

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
