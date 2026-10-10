#ifndef SERRON_MORPHOLOGY_ENTRY_POINTS_CUH
#define SERRON_MORPHOLOGY_ENTRY_POINTS_CUH

#include <cuda/morphology/enums.cuh>

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

/**
 * Lists for stamping out the named kernel entry points.
 *
 * Every kernel is reached through an @c extern "C" @c __global__ wrapper per (kernel, op, dtype), named
 * @c serron_<kernel>_<op>_<dtype>, so it can be looked up by a fixed symbol name instead of by template
 * instantiation. The kernel bodies are @c __device__ templates; the wrappers only forward to them.
 */

/**
 * Calls @c X(scalar_t, suffix) once per dtype the CUDA kernels are built for. @c suffix ends the entry point's name.
 *
 * Must list the same dtypes as the @c THO_DISPATCH_V2 calls in operators.cu and backward.cu.
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

#endif // SERRON_MORPHOLOGY_ENTRY_POINTS_CUH
