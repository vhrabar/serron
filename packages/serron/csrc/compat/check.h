#ifndef SERRON_COMPAT_CHECK_H
#define SERRON_COMPAT_CHECK_H

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace serron::detail {

/// Concatenates @p args with ostream formatting (TORCH_CHECK err-msg clone)
template <typename... Args>
std::string check_message(const Args&... args) {
    std::ostringstream oss;
    ((oss << args), ...);
    return std::move(oss).str();
}

} // namespace serron::detail

/**
 * Throws @c std::runtime_error carrying just the formatted message when @p cond is false.
 *
 * replacemnt for TORCH_CHECK
 */
#define SERRON_CHECK(cond, ...)                                                                                        \
    do {                                                                                                               \
        if (!(cond)) [[unlikely]] {                                                                                    \
            throw std::runtime_error(serron::detail::check_message(__VA_ARGS__));                                      \
        }                                                                                                              \
    } while (0)

#endif // SERRON_COMPAT_CHECK_H
