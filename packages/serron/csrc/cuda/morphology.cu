#include "ops.h"

#include <c10/util/Exception.h>


namespace serron {

at::Tensor erode(const at::Tensor& input, const at::Tensor& kernel, int64_t border) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "serron::erode is not implemented yet");
}

at::Tensor dilate(const at::Tensor& input, const at::Tensor& kernel, int64_t border) {

    TORCH_CHECK_NOT_IMPLEMENTED(false, "serron::dilate is not implemented yet");
}

} // namespace serron