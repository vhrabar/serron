#ifndef SERRON_CUDA_MORPHOLOGY_ENTRY_POINTS_H
#define SERRON_CUDA_MORPHOLOGY_ENTRY_POINTS_H

#include <cuda/morphology/enums.h>
#include <cuda/utils/border_mode.h>

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cstdint>
#include <type_traits>

/**
 * Named kernel entry points, shared by the device code that defines them and the host code that launches them.
 *
 * Every kernel is reached through an @c extern "C" @c __global__ wrapper per (kernel, op, dtype), named
 * @c serron_<kernel>_<op>_<dtype>, so it can be looked up by a fixed symbol name instead of by template
 * instantiation. The kernel bodies are @c __device__ templates; the wrappers only forward to them.
 *
 * Plain C++: included by the host .cpp files as well as the device .cu files.
 */

/**
 * Calls @c X(scalar_t, suffix) once per dtype the CUDA kernels are built for. @c suffix ends the entry point's name.
 *
 * Must list the same dtypes as the @c THO_DISPATCH_V2 calls in operators.cpp and backward.cpp.
 */
#define SERRON_CUDA_FOR_EACH_DTYPE(X)                                                                                  \
    X(float, f32)                                                                                                      \
    X(double, f64)                                                                                                     \
    X(torch::headeronly::Half, f16)                                                                                    \
    X(torch::headeronly::BFloat16, bf16)

/**
 * Calls @c X(Op, op, scalar_t, suffix) once per morphology operation, for use inside a
 * @ref SERRON_CUDA_FOR_EACH_DTYPE callback. @c Op is the policy type, @c op the name part of the entry point.
 *
 * The order is the @ref MorphOp order, which the kernel tables index by.
 */
#define SERRON_CUDA_FOR_EACH_OP(X, scalar_t, suffix)                                                                   \
    X(ErodeOp, erode, scalar_t, suffix)                                                                                \
    X(DilateOp, dilate, scalar_t, suffix)

static_assert(static_cast<int>(MorphOp::kErode) == 0 && static_cast<int>(MorphOp::kDilate) == 1,
              "kernel tables are indexed by MorphOp in SERRON_CUDA_FOR_EACH_OP order");

/**
 * Calls @c X(member, name, Fn, scalar_t) once per forward kernel of one (op, dtype): the @ref ForwardKernels
 * member, the entry point's name, and its function type.
 */
#define SERRON_FORWARD_KERNELS(X, op, scalar_t, suffix)                                                                \
    X(element, serron_morphology_##op##_##suffix, ForwardElementFn, scalar_t)                                          \
    X(tiled, serron_morphology_tiled_##op##_##suffix, ForwardTiledFn, scalar_t)                                        \
    X(line_row, serron_morphology_line_row_##op##_##suffix, ForwardLineFn, scalar_t)                                   \
    X(line_col, serron_morphology_line_col_##op##_##suffix, ForwardLineFn, scalar_t)

/**
 * Calls @c X(member, name, Fn, scalar_t) once per backward kernel of one (op, dtype): the @ref BackwardKernels
 * member, the entry point's name, and its function type.
 */
#define SERRON_BACKWARD_KERNELS(X, op, scalar_t, suffix)                                                               \
    X(element, serron_morphology_backward_##op##_##suffix, BackwardElementFn, scalar_t)                                \
    X(tiled, serron_morphology_backward_tiled_##op##_##suffix, BackwardTiledFn, scalar_t)                              \
    X(row_argreduce, serron_morphology_row_argreduce_##op##_##suffix, BackwardRowArgreduceFn, scalar_t)                \
    X(col_argreduce, serron_morphology_col_argreduce_##op##_##suffix, BackwardColArgreduceFn, scalar_t)

/// Expands to the lookup of entry point @c name in @c library, for a kernel-table row.
#define SERRON_GET_ENTRY_POINT(member, name, Fn, scalar_t) get_kernel<Fn<scalar_t>>(library, #name),

/**
 * Expands to a @c static_assert that entry point @c name has function type @c Fn<scalar_t>.
 *
 * Skipped when nvcc runs on MSVC: there the type of an @c extern "C" @c __global__ function does not compare equal
 * to the plain alias even when the parameters match. The GCC and Clang builds check the same definitions.
 */
#if defined(_MSC_VER)
#define SERRON_CHECK_ENTRY_POINT_TYPE(member, name, Fn, scalar_t)
#else
#define SERRON_CHECK_ENTRY_POINT_TYPE(member, name, Fn, scalar_t)                                                      \
    static_assert(std::is_same_v<decltype(name), Fn<scalar_t>>, #name " does not match " #Fn);
#endif

namespace serron {

/// @c morphology_kernel: input, kernel, output, N, C, H, W, kH, kW, kernel_channel_stride, border.
template <typename scalar_t>
using ForwardElementFn = void(const scalar_t*, const scalar_t*, scalar_t*, int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t, BorderMode);

/// @c morphology_tiled_kernel: input, kernel, output, C, H, W, kH, kW, kernel_channel_stride, border.
template <typename scalar_t>
using ForwardTiledFn = void(const scalar_t*, const scalar_t*, scalar_t*, int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, BorderMode);

/// @c morphology_line_kernel, either axis: input, output, N, C, H, W, k, chunks, border.
template <typename scalar_t>
using ForwardLineFn = void(const scalar_t*, scalar_t*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                           BorderMode);

/**
 * @c morphology_backward_kernel: grad_output, input, kernel, grad_input, grad_kernel, N, C, H, W, kH, kW,
 * kernel_channel_stride, border, need_kernel_grad.
 */
template <typename scalar_t>
using BackwardElementFn = void(const scalar_t*, const scalar_t*, const scalar_t*, scalar_t*, scalar_t*, int64_t,
                               int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, BorderMode, bool);

/**
 * @c morphology_backward_tiled_kernel: grad_output, input, kernel, grad_input, grad_kernel, C, H, W, kH, kW,
 * kernel_channel_stride, border, need_kernel_grad.
 */
template <typename scalar_t>
using BackwardTiledFn = void(const scalar_t*, const scalar_t*, const scalar_t*, scalar_t*, scalar_t*, int64_t, int64_t,
                             int64_t, int64_t, int64_t, int64_t, BorderMode, bool);

/// @c morphology_row_argreduce_kernel: input, row_best_val, row_best_dj, H, W, kW, chunks, border.
template <typename scalar_t>
using BackwardRowArgreduceFn = void(const scalar_t*, scalar_t*, std::int32_t*, int64_t, int64_t, int64_t, int64_t,
                                    BorderMode);

/**
 * @c morphology_col_argreduce_kernel: row_best_val, row_best_dj, grad_output, grad_input, grad_kernel, C, H, W, kH,
 * kW, chunks, kernel_channel_stride, border, need_kernel_grad.
 */
template <typename scalar_t>
using BackwardColArgreduceFn = void(const scalar_t*, const std::int32_t*, const scalar_t*, scalar_t*, scalar_t*,
                                    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, BorderMode, bool);

} // namespace serron

#endif // SERRON_CUDA_MORPHOLOGY_ENTRY_POINTS_H
