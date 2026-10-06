#include "torch_binding.h"

#include "registration.h"

#include <torch/csrc/stable/library.h>

// Operator schemas. TORCH_EXTENSION_NAME carries the namespace: kernel-builder suffixes it with a per-build hash
STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("erode(Tensor input, Tensor kernel, int border, bool? flat=None) -> Tensor");
    ops.def("dilate(Tensor input, Tensor kernel, int border, bool? flat=None) -> Tensor");
    ops.def("erode_backward(Tensor grad_output, Tensor input, Tensor kernel, int border, bool? flat=None, "
            "bool need_kernel_grad=True) -> (Tensor, Tensor)");
    ops.def("dilate_backward(Tensor grad_output, Tensor input, Tensor kernel, int border, bool? flat=None, "
            "bool need_kernel_grad=True) -> (Tensor, Tensor)");
}

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("erode", TORCH_BOX(&serron::erode));
    ops.impl("dilate", TORCH_BOX(&serron::dilate));
    ops.impl("erode_backward", TORCH_BOX(&serron::erode_backward));
    ops.impl("dilate_backward", TORCH_BOX(&serron::dilate_backward));
}

#else

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CPU, ops) {
    ops.impl("erode", TORCH_BOX(&serron::erode_cpu));
    ops.impl("dilate", TORCH_BOX(&serron::dilate_cpu));
    ops.impl("erode_backward", TORCH_BOX(&serron::erode_backward_cpu));
    ops.impl("dilate_backward", TORCH_BOX(&serron::dilate_backward_cpu));
}

#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
