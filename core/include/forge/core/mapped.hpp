#pragma once
// Portable read-only file mapping (POSIX mmap / Win32 file mapping). Empty files map to nullptr with size 0.
#include <forge/core/error.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace forge {
inline const std::uint8_t* map_readonly(const std::string& path, std::size_t& size) {
#if defined(_WIN32)
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) raise(errc::io_error, "cannot open {}", path);
    LARGE_INTEGER sz{}; GetFileSizeEx(f, &sz); size = std::size_t(sz.QuadPart);
    if (!size) { CloseHandle(f); return nullptr; }
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(f);
    if (!m) raise(errc::io_error, "mmap failed for {}", path);
    auto* p = static_cast<const std::uint8_t*>(MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0));
    CloseHandle(m);
    if (!p) raise(errc::io_error, "mmap failed for {}", path);
    return p;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) raise(errc::io_error, "cannot open {}", path);
    struct stat st{}; ::fstat(fd, &st); size = std::size_t(st.st_size);
    if (!size) { ::close(fd); return nullptr; }
    void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) raise(errc::io_error, "mmap failed for {}", path);
    return static_cast<const std::uint8_t*>(p);
#endif
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
