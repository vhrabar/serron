#include "ops.h"
#include "registration.h"

#include <torch/csrc/stable/library.h>

// CUDA implementations, built into _C_cuda, which is loaded after _C (where the schemas live) only on a
// CUDA-enabled torch.
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("erode", TORCH_BOX(&serron::erode));
    ops.impl("dilate", TORCH_BOX(&serron::dilate));
    ops.impl("erode_backward", TORCH_BOX(&serron::erode_backward));
    ops.impl("dilate_backward", TORCH_BOX(&serron::dilate_backward));
}
