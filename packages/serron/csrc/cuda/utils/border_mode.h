#ifndef SERRON_CUDA_BORDER_MODE_H
#define SERRON_CUDA_BORDER_MODE_H

/**
 * Boundary handling for out-of-image neighbourhood samples.
 */
enum BorderMode : int {
    kReflect = 0,   ///< mirror without repeating the edge sample (…cb|abcd|cb…)
    kReplicate = 1, ///< clamp to the edge sample (…aa|abcd|dd…)
    kConstant = 2,  ///< out-of-bounds resolved to the neutral element by the caller
};

#endif // SERRON_CUDA_BORDER_MODE_H
