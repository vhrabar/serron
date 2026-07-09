#ifndef SERRON_OPS_H
#define SERRON_OPS_H

#include <torch/torch.h>

namespace serron {

at::Tensor erode(const at::Tensor& input, const at::Tensor& kernel, int64_t border);
at::Tensor dilate(const at::Tensor& input, const at::Tensor& kernel, int64_t border);

} // namespace serron

#endif // SERRON_OPS_H