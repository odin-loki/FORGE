#pragma once
// forge::os::single_instance: let only one process per user hold a name ("one copy of this app").
//   POSIX (Linux, Android, macOS, iOS, FreeBSD): an exclusive, non-blocking flock() on
//   <dir>/forge-<uid>-<name>.lock, dir = $XDG_RUNTIME_DIR when set, else the temp directory (set_directory()
//   overrides it: Android apps pass their files dir). The kernel drops the lock when the holder exits or crashes,
//   so a stale file never blocks a start; the file itself is left in place (unlinking it would race a starter).
//   Windows: a named mutex Local\forge-instance-<name> (per logon session), released when the holder exits.
// acquire() returns the guard, or an error for which already_running(err) is true. The holder's pid is written
// into the lock file (POSIX) so the second instance can say who has it: holder_pid().
#include <forge/os/os.hpp>
#if defined(FORGE_OS_POSIX)
#include <sys/file.h>
#endif

namespace forge::os {
class single_instance {
public:
    single_instance() = default;
    single_instance(single_instance&& o) noexcept { swap_(o); }
    single_instance& operator=(single_instance&& o) noexcept { if (this != &o) { release(); swap_(o); } return *this; }
    single_instance(const single_instance&) = delete; single_instance& operator=(const single_instance&) = delete;
    ~single_instance() { release(); }

    // Names: 1-64 characters of [A-Za-z0-9._-].
    static bool valid_name(const std::string& n) {
        if (n.empty() || n.size() > 64) return false;
        for (unsigned char c : n) if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-')) return false;
        return true;
    }
    static inline constexpr int already_running_code = 0x0F0E6;
    static bool already_running(const error& e) { return e.code == already_running_code; }
    // POSIX: where lock files live ("" = the default). No effect on Windows.
    static void set_directory(std::string dir) { std::lock_guard<std::mutex> g(dir_mutex_()); dir_() = std::move(dir); }
    static std::string lock_path(const std::string& name) {
#if defined(FORGE_OS_POSIX)
        std::string d; { std::lock_guard<std::mutex> g(dir_mutex_()); d = dir_(); }
        if (d.empty()) { d = getenv("XDG_RUNTIME_DIR"); if (d.empty() || d[0] != '/') d = temp_dir(); }
        return d + "/forge-" + std::to_string(::getuid()) + "-" + name + ".lock";
#else
        return "Local\\forge-instance-" + name;
#endif
    }
    static result<single_instance> acquire(const std::string& name) {
        if (!valid_name(name)) return {{}, errno_error(EINVAL, "single_instance: invalid name '" + name + "'")};
        single_instance s; s.name_ = name;
#if defined(FORGE_OS_WINDOWS)
        HANDLE h = CreateMutexW(nullptr, FALSE, widen(lock_path(name)).c_str());
        if (!h) return {{}, last_error("CreateMutex " + name)};
        if (GetLastError() == ERROR_ALREADY_EXISTS) { CloseHandle(h); return {{}, error{already_running_code, "single_instance: '" + name + "' is already running"}}; }
        s.h_ = h;
#else
        std::string p = lock_path(name);
        int fd = ::open(p.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) return {{}, last_error("single_instance: open " + p)};
        int rc; do { rc = ::flock(fd, LOCK_EX | LOCK_NB); } while (rc != 0 && errno == EINTR);
        if (rc != 0) {
            int e = errno; ::close(fd);
            if (e == EWOULDBLOCK || e == EAGAIN) return {{}, error{already_running_code, "single_instance: '" + name + "' is already running"}};
            return {{}, errno_error(e, "single_instance: flock " + p + ": " + error_text(e))};
        }
        std::string pid = std::to_string(::getpid()) + "\n";
        if (::ftruncate(fd, 0) == 0) (void)!::pwrite(fd, pid.data(), pid.size(), 0);
        s.fd_ = fd;
#endif
        return {std::move(s), {}};
    }
    // The pid recorded by the current holder (POSIX); 0 when unknown, nobody holds it, or on Windows.
    static long holder_pid(const std::string& name) {
#if defined(FORGE_OS_POSIX)
        if (!valid_name(name)) return 0;
        auto f = read_file(lock_path(name)); if (!f) return 0;
        return std::strtol(f.value.c_str(), nullptr, 10);
#else
        (void)name; return 0;
#endif
    }
    bool held() const {
#if defined(FORGE_OS_WINDOWS)
        return h_ != nullptr;
#else
        return fd_ >= 0;
#endif
    }
    const std::string& name() const { return name_; }
    void release() {
#if defined(FORGE_OS_WINDOWS)
        if (h_) { CloseHandle(h_); h_ = nullptr; }
#else
        if (fd_ >= 0) { (void)!::ftruncate(fd_, 0); ::close(fd_); fd_ = -1; }   // closing drops the flock
#endif
    }
private:
    void swap_(single_instance& o) {
        std::swap(name_, o.name_);
#if defined(FORGE_OS_WINDOWS)
        std::swap(h_, o.h_);
#else
        std::swap(fd_, o.fd_);
#endif
    }
    static std::mutex& dir_mutex_() { static std::mutex m; return m; }
    static std::string& dir_() { static std::string d; return d; }
    std::string name_;
#if defined(FORGE_OS_WINDOWS)
    HANDLE h_ = nullptr;
#else
    int fd_ = -1;
#endif
};
} // namespace forge::os
