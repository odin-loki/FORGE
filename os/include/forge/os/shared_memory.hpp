#pragma once
// Named shared memory between processes.
//   POSIX (Linux, macOS, iOS-within-app-group, FreeBSD): shm_open + mmap.  Windows: CreateFileMappingW
//   ("Local\\forge-<name>").  Android has no shm_open: a file under $TMPDIR (or the app cache dir) is mapped
//   MAP_SHARED, which other processes of the app / same uid can open by name.
#include <forge/os/detail.hpp>

namespace forge::os {
class shared_memory {
public:
    shared_memory() = default;
    shared_memory(shared_memory&& o) noexcept { *this = std::move(o); }
    shared_memory& operator=(shared_memory&& o) noexcept {
        if (this != &o) { close(); p_ = std::exchange(o.p_, nullptr); n_ = std::exchange(o.n_, 0); name_ = std::move(o.name_); owner_ = std::exchange(o.owner_, false);
#if defined(FORGE_OS_WINDOWS)
            h_ = std::exchange(o.h_, nullptr);
#endif
        }
        return *this;
    }
    ~shared_memory() { close(); }
    // Create (or truncate) a region of n bytes. The creator unlinks the name on close unless keep() is called.
    static result<shared_memory> create(const std::string& name, std::size_t n) { return open_(name, n, true); }
    // Open an existing region; n = 0 means "whatever size it has".
    static result<shared_memory> open(const std::string& name, std::size_t n = 0) { return open_(name, n, false); }
    std::uint8_t* data() const { return static_cast<std::uint8_t*>(p_); }
    std::size_t size() const { return n_; }
    const std::string& name() const { return name_; }
    void keep() { owner_ = false; }
    static status unlink(const std::string& name) {
#if defined(FORGE_OS_WINDOWS)
        (void)name; return {};   // named mappings vanish with their last handle
#elif defined(__ANDROID__)
        if (::unlink(path_(name).c_str())) return {{}, last_error("unlink")}; return {};
#else
        if (::shm_unlink(path_(name).c_str())) return {{}, last_error("shm_unlink")}; return {};
#endif
    }
    void close() {
        if (!p_) return;
#if defined(FORGE_OS_WINDOWS)
        UnmapViewOfFile(p_); if (h_) CloseHandle(h_); h_ = nullptr;
#else
        ::munmap(p_, n_); if (owner_) unlink(name_);
#endif
        p_ = nullptr; n_ = 0;
    }
private:
    static std::string path_(const std::string& name) {
#if defined(FORGE_OS_WINDOWS)
        return "Local\\forge-" + name;
#elif defined(__ANDROID__)
        return join(temp_dir(), "forge-shm-" + name);
#elif defined(__APPLE__)
        return "/fg-" + name.substr(0, 26);   // PSHMNAMLEN is 31
#else
        return "/forge-" + name;
#endif
    }
    static result<shared_memory> open_(const std::string& name, std::size_t n, bool create) {
        shared_memory s; s.name_ = name; s.owner_ = create;
        if (name.empty() || name.find('/') != std::string::npos || name.find('\\') != std::string::npos) return {std::move(s), {22, "shared_memory: bad name"}};
#if defined(FORGE_OS_WINDOWS)
        auto w = widen(path_(name));
        if (create) s.h_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, DWORD(std::uint64_t(n) >> 32), DWORD(n), w.c_str());
        else s.h_ = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, w.c_str());
        if (!s.h_) return {std::move(s), last_error("shared_memory " + name)};
        s.p_ = MapViewOfFile(s.h_, FILE_MAP_ALL_ACCESS, 0, 0, n);
        if (!s.p_) return {std::move(s), last_error("MapViewOfFile")};
        MEMORY_BASIC_INFORMATION mi{}; VirtualQuery(s.p_, &mi, sizeof mi); s.n_ = n ? n : mi.RegionSize;
        return {std::move(s), {}};
#else
        auto p = path_(name);
#if defined(__ANDROID__)
        int fd = ::open(p.c_str(), O_RDWR | O_CLOEXEC | (create ? O_CREAT | O_TRUNC : 0), 0600);
#else
        int fd = ::shm_open(p.c_str(), O_RDWR | (create ? O_CREAT | O_TRUNC : 0), 0600);
#endif
        if (fd < 0) { s.owner_ = false; return {std::move(s), last_error("shm_open " + name)}; }
        if (create && ::ftruncate(fd, off_t(n))) { auto e = last_error("ftruncate"); ::close(fd); return {std::move(s), e}; }
        if (!n) { struct ::stat st; if (::fstat(fd, &st) == 0) n = std::size_t(st.st_size); }
        if (!n) { ::close(fd); s.owner_ = false; return {std::move(s), {22, "shared_memory: empty region"}}; }
        void* m = ::mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); ::close(fd);
        if (m == MAP_FAILED) return {std::move(s), last_error("mmap shared")};
        s.p_ = m; s.n_ = n; return {std::move(s), {}};
#endif
    }
    void* p_ = nullptr; std::size_t n_ = 0; std::string name_; bool owner_ = false;
#if defined(FORGE_OS_WINDOWS)
    HANDLE h_ = nullptr;
#endif
};
} // namespace forge::os
