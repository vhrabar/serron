#ifndef SERRON_MORPHOLOGY_ENUMS_CUH
#define SERRON_MORPHOLOGY_ENUMS_CUH

/**
 * Morpho types
 */
enum class MorphOp : int {
    kErode = 0,  ///< min over (sample - se)
    kDilate = 1, ///< max over (sample + se)
};

#endif // SERRON_MORPHOLOGY_ENUMS_CUH