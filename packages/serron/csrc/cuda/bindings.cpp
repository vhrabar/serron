#include "ops.h"
#include "registration.h"

#include <torch/library.h>

// CUDA implementations,built into _C_cuda, which is loaded after _C (where the schemas live) only on aCUDA-enabled torch.
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("erode", &serron::erode);
    ops.impl("dilate", &serron::dilate);
    ops.impl("erode_backward", &serron::erode_backward);
    ops.impl("dilate_backward", &serron::dilate_backward);
}
