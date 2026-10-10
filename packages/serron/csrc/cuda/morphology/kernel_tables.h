#ifndef SERRON_CUDA_MORPHOLOGY_KERNEL_TABLES_H
#define SERRON_CUDA_MORPHOLOGY_KERNEL_TABLES_H

#include <cuda/morphology/entry_points.h>
#include <cuda/utils/launch.h>

/**
 * Host-side tables of the kernel handles for one (op, dtype), filled from the embedded fatbins by name.
 */

namespace serron {

/**
 * Forward entry points for one (op, dtype).
 *
 * @tparam scalar_t  Element type the entry points were stamped out for.
 */
template <typename scalar_t>
struct ForwardKernels {
    Kernel<ForwardElementFn<scalar_t>> element;
    Kernel<ForwardTiledFn<scalar_t>> tiled;
    Kernel<ForwardLineFn<scalar_t>> line_row;
    Kernel<ForwardLineFn<scalar_t>> line_col;
};

/**
 * Backward entry points for one (op, dtype).
 *
 * @tparam scalar_t  Element type the entry points were stamped out for.
 */
template <typename scalar_t>
struct BackwardKernels {
    Kernel<BackwardElementFn<scalar_t>> element;
    Kernel<BackwardTiledFn<scalar_t>> tiled;
    Kernel<BackwardRowArgreduceFn<scalar_t>> row_argreduce;
    Kernel<BackwardColArgreduceFn<scalar_t>> col_argreduce;
};

} // namespace serron

#endif // SERRON_CUDA_MORPHOLOGY_KERNEL_TABLES_H
