#include "ops.h"
#include "registration.h"

#include <torch/csrc/stable/library.h>

// CUDA implementations for the CMake / PyPI build, compiled into _C_cuda and loaded after _C, where the
// schemas live only on a CUDA-enabled torch.
// CUDA_KERNEL.
STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("erode", TORCH_BOX(&serron::erode));
    ops.impl("dilate", TORCH_BOX(&serron::dilate));
    ops.impl("erode_backward", TORCH_BOX(&serron::erode_backward));
    ops.impl("dilate_backward", TORCH_BOX(&serron::dilate_backward));
}
