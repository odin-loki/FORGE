#pragma once
// Read-only file mapping on top of forge::os. Empty files map to nullptr with size 0.
#include <forge/core/error.hpp>
#include <forge/os/os.hpp>
#include <cstddef>
#include <cstdint>
#include <string>

namespace forge {
inline const std::uint8_t* map_readonly(const std::string& path, std::size_t& size) {
    auto m = os::mapping::map_file(path, os::prot_read);
    if (!m) raise(errc::io_error, "cannot map {}: {}", path, m.err.what);
    size = m->size();
    return static_cast<const std::uint8_t*>(m->release());
}
inline void unmap_readonly(const std::uint8_t* p, std::size_t size) {
    if (!p) return;
#if defined(_WIN32)
    (void)size; UnmapViewOfFile(p);
#else
    ::munmap(const_cast<std::uint8_t*>(p), size);
#endif
}
} // namespace forge
