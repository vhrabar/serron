#ifndef SERRON_GPU_MORPHOLOGY_ENUMS_H
#define SERRON_GPU_MORPHOLOGY_ENUMS_H

/**
 * Morpho types
 */
enum class MorphOp : int {
    kErode = 0,  ///< min over (sample - se)
    kDilate = 1, ///< max over (sample + se)
};

#endif // SERRON_GPU_MORPHOLOGY_ENUMS_H
