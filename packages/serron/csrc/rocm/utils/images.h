#ifndef SERRON_ROCM_IMAGES_H
#define SERRON_ROCM_IMAGES_H

#include <gpu/utils/backend.h>

#include <cstddef>
#include <span>

namespace serron::rocm {

/// One kernel image embedded in the library: a clang offload bundle for @c hipModuleLoadData.
struct EmbeddedImage {
    const char* target;
    const unsigned char* begin;
    const unsigned char* end;

    const void* data() const {
        return begin;
    }
    size_t size() const {
        return static_cast<size_t>(end - begin);
    }
};

/**
 * Every image built for @p file: one per architecture in SERRON_ROCM_ARCHS, then the portable one.
 *
 */
std::span<const EmbeddedImage> embedded_images(KernelFile file);

} // namespace serron::rocm

#endif // SERRON_ROCM_IMAGES_H
