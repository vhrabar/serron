#ifndef SERRON_MORPH_ALGORITHM_H
#define SERRON_MORPH_ALGORITHM_H

#include <compat/check.h>

#include <cstdint>

namespace serron {

/**
 * Algorithm a morphology op runs.
 *
 * The integer values are the encoding of the operators' @c algo argument, so they must match
 * serron.enums.Algorithm.
 */
enum class Algorithm : int {
    kAuto = 0,      ///< let the library choose
    kElement = 1,   ///< one thread per output element, reading global memory
    kTiled = 2,     ///< 2-D shared-memory tile
    kSeparable = 3, ///< van Herk / Gil-Werman row pass then column pass; flat, axis-separable SE only
};

/**
 * Decodes the operators' @c algo argument.
 *
 * @param value  Encoded algorithm.
 * @param name   Qualified caller name used to prefix the diagnostic.
 * @throws std::runtime_error  if @p value is not one of the @ref Algorithm values.
 */
inline Algorithm parse_algorithm(const int64_t value, const char* name) {
    SERRON_CHECK(value >= static_cast<int64_t>(Algorithm::kAuto) &&
                     value <= static_cast<int64_t>(Algorithm::kSeparable),
                 name, ": invalid algo ", value);
    return static_cast<Algorithm>(value);
}

/// Name of @p algorithm for diagnostics.
inline const char* algorithm_name(const Algorithm algorithm) {
    switch (algorithm) {
    case Algorithm::kAuto:
        return "auto";
    case Algorithm::kElement:
        return "element";
    case Algorithm::kTiled:
        return "tiled";
    case Algorithm::kSeparable:
        return "separable";
    }
    return "unknown";
}

} // namespace serron

#endif // SERRON_MORPH_ALGORITHM_H
