#pragma once
// forge::os: one OS-call API for Windows, Linux, Android, macOS, iOS and FreeBSD.
// Files and fs, mmap, processes, threads, sync, time, sockets and polling (epoll / kqueue / WSAPoll), dynamic
// libraries, environment, signals, memory, system and CPU info, paths, and entropy.
// Errors: calls return os::result<T> (value or a native error code with a message); nothing throws.
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define FORGE_OS_WINDOWS 1
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <bcrypt.h>
#include <direct.h>
#include <io.h>
#include <process.h>
#else
#define FORGE_OS_POSIX 1
#include <arpa/inet.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <mach-o/dyld.h>
#include <mach/mach_time.h>
#include <sys/event.h>
#include <sys/sysctl.h>
#define FORGE_OS_KQUEUE 1
#if TARGET_OS_IPHONE
#define FORGE_OS_IOS 1
#else
#define FORGE_OS_MACOS 1
#endif
#elif defined(__FreeBSD__)
#include <sys/event.h>
#include <sys/sysctl.h>
#define FORGE_OS_KQUEUE 1
#define FORGE_OS_FREEBSD 1
#include <pthread_np.h>
#elif defined(__linux__)
#include <sys/epoll.h>
#include <sys/syscall.h>
#define FORGE_OS_EPOLL 1
#if defined(__ANDROID__)
#define FORGE_OS_ANDROID 1
#else
#define FORGE_OS_LINUX 1
#include <sys/random.h>
#endif
#endif
#if !defined(FORGE_OS_IOS)
#include <spawn.h>
extern char** environ;
#endif
#endif

namespace forge::os {

// ---------------------------------------------------------------- errors
struct error { int code = 0; std::string what; explicit operator bool() const { return code != 0; } };
inline int last_errno() {
#if defined(FORGE_OS_WINDOWS)
    return int(GetLastError());
#else
    return errno;
#endif
}
inline std::string error_text(int code) {
#if defined(FORGE_OS_WINDOWS)
    char b[256] = {};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, DWORD(code), 0, b, sizeof b, nullptr);
    std::string s(b); while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back(); return s;
#else
    return std::strerror(code);
#endif
}
inline error last_error(std::string_view ctx) { int c = last_errno(); if (!c) c = -1; return {c, std::string(ctx) + ": " + error_text(c)}; }
template <class T> struct result {
    T value{}; error err;
    explicit operator bool() const { return !err; }
    T& operator*() { return value; }
    T* operator->() { return &value; }
};
struct ok_t {};
using status = result<ok_t>;

// ---------------------------------------------------------------- platform identity
enum class platform { windows, linux, android, macos, ios, freebsd, other };
constexpr platform current() {
#if defined(FORGE_OS_WINDOWS)
    return platform::windows;
#elif defined(FORGE_OS_ANDROID)
    return platform::android;
#elif defined(FORGE_OS_LINUX)
    return platform::linux;
#elif defined(FORGE_OS_IOS)
    return platform::ios;
#elif defined(FORGE_OS_MACOS)
    return platform::macos;
#elif defined(FORGE_OS_FREEBSD)
    return platform::freebsd;
#else
    return platform::other;
#endif
}
constexpr const char* platform_name(platform p = current()) {
    switch (p) { case platform::windows: return "windows"; case platform::linux: return "linux"; case platform::android: return "android";
                 case platform::macos: return "macos"; case platform::ios: return "ios"; case platform::freebsd: return "freebsd"; default: return "other"; }
}

#if defined(FORGE_OS_WINDOWS)
inline std::wstring widen(std::string_view s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(std::size_t(n), L'\0'); MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n); return w;
}
inline std::string narrow(std::wstring_view w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(std::size_t(n), '\0'); WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), s.data(), n, nullptr, nullptr); return s;
}
#endif

// ---------------------------------------------------------------- files
enum open_flags : unsigned { rd = 1, wr = 2, rdwr = 3, create = 4, truncate = 8, append = 16, excl = 32 };
class file {
public:
#if defined(FORGE_OS_WINDOWS)
    using native_t = HANDLE;
    static native_t bad() { return INVALID_HANDLE_VALUE; }
#else
    using native_t = int;
    static native_t bad() { return -1; }
#endif
    file() = default;
    explicit file(native_t h) : h_(h) {}
    file(file&& o) noexcept : h_(std::exchange(o.h_, bad())) {}
    file& operator=(file&& o) noexcept { if (this != &o) { close(); h_ = std::exchange(o.h_, bad()); } return *this; }
    ~file() { close(); }
    bool valid() const { return h_ != bad(); }
    native_t native() const { return h_; }
    void close() {
        if (!valid()) return;
#if defined(FORGE_OS_WINDOWS)
        CloseHandle(h_);
#else
        ::close(h_);
#endif
        h_ = bad();
    }
    static result<file> open(const std::string& path, unsigned flags = rd) {
#if defined(FORGE_OS_WINDOWS)
        DWORD acc = ((flags & rd) ? GENERIC_READ : 0) | ((flags & (wr | append)) ? GENERIC_WRITE : 0);
        DWORD disp = (flags & excl) ? CREATE_NEW : (flags & create) ? ((flags & truncate) ? CREATE_ALWAYS : OPEN_ALWAYS)
                   : (flags & truncate) ? TRUNCATE_EXISTING : OPEN_EXISTING;
        HANDLE h = CreateFileW(widen(path).c_str(), acc, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, disp, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return {{}, last_error("open " + path)};
        if (flags & append) { LARGE_INTEGER z{}; SetFilePointerEx(h, z, nullptr, FILE_END); }
        return {file(h), {}};
#else
        int f = ((flags & rdwr) == rdwr || ((flags & rd) && (flags & append))) ? O_RDWR : (flags & (wr | append)) ? O_WRONLY : O_RDONLY;
        if (flags & create) f |= O_CREAT; if (flags & truncate) f |= O_TRUNC; if (flags & append) f |= O_APPEND; if (flags & excl) f |= O_EXCL | O_CREAT;
        int h = ::open(path.c_str(), f | O_CLOEXEC, 0644);
        if (h < 0) return {{}, last_error("open " + path)};
        return {file(h), {}};
#endif
    }
    result<std::size_t> read(void* p, std::size_t n) {
#if defined(FORGE_OS_WINDOWS)
        DWORD got = 0; if (!ReadFile(h_, p, DWORD(n), &got, nullptr)) return {0, last_error("read")}; return {got, {}};
#else
        for (;;) { auto r = ::read(h_, p, n); if (r >= 0) return {std::size_t(r), {}}; if (errno != EINTR) return {0, last_error("read")}; }
#endif
    }
    result<std::size_t> write(const void* p, std::size_t n) {
#if defined(FORGE_OS_WINDOWS)
        DWORD put = 0; if (!WriteFile(h_, p, DWORD(n), &put, nullptr)) return {0, last_error("write")}; return {put, {}};
#else
        for (;;) { auto r = ::write(h_, p, n); if (r >= 0) return {std::size_t(r), {}}; if (errno != EINTR) return {0, last_error("write")}; }
#endif
    }
    result<std::size_t> pread(void* p, std::size_t n, std::uint64_t off) {
#if defined(FORGE_OS_WINDOWS)
        OVERLAPPED o{}; o.Offset = DWORD(off); o.OffsetHigh = DWORD(off >> 32); DWORD got = 0;
        if (!ReadFile(h_, p, DWORD(n), &got, &o)) { if (GetLastError() == ERROR_HANDLE_EOF) return {0, {}}; return {0, last_error("pread")}; }
        return {got, {}};
#else
        for (;;) { auto r = ::pread(h_, p, n, off_t(off)); if (r >= 0) return {std::size_t(r), {}}; if (errno != EINTR) return {0, last_error("pread")}; }
#endif
    }
    result<std::size_t> pwrite(const void* p, std::size_t n, std::uint64_t off) {
#if defined(FORGE_OS_WINDOWS)
        OVERLAPPED o{}; o.Offset = DWORD(off); o.OffsetHigh = DWORD(off >> 32); DWORD put = 0;
        if (!WriteFile(h_, p, DWORD(n), &put, &o)) return {0, last_error("pwrite")}; return {put, {}};
#else
        for (;;) { auto r = ::pwrite(h_, p, n, off_t(off)); if (r >= 0) return {std::size_t(r), {}}; if (errno != EINTR) return {0, last_error("pwrite")}; }
#endif
    }
    status write_all(const void* p, std::size_t n) {
        auto* b = static_cast<const std::uint8_t*>(p);
        while (n) { auto r = write(b, n); if (!r) return {{}, r.err}; if (!r.value) return {{}, {-1, "write: no progress"}}; b += r.value; n -= r.value; }
        return {};
    }
    result<std::uint64_t> size() const {
#if defined(FORGE_OS_WINDOWS)
        LARGE_INTEGER s{}; if (!GetFileSizeEx(h_, &s)) return {0, last_error("size")}; return {std::uint64_t(s.QuadPart), {}};
#else
        struct stat st{}; if (::fstat(h_, &st)) return {0, last_error("fstat")}; return {std::uint64_t(st.st_size), {}};
#endif
    }
    status resize(std::uint64_t n) {
#if defined(FORGE_OS_WINDOWS)
        FILE_END_OF_FILE_INFO i{}; i.EndOfFile.QuadPart = LONGLONG(n);
        if (!SetFileInformationByHandle(h_, FileEndOfFileInfo, &i, sizeof i)) return {{}, last_error("resize")}; return {};
#else
        if (::ftruncate(h_, off_t(n))) return {{}, last_error("ftruncate")}; return {};
#endif
    }
    status sync() {
#if defined(FORGE_OS_WINDOWS)
        if (!FlushFileBuffers(h_)) return {{}, last_error("fsync")}; return {};
#elif defined(__APPLE__)
        if (::fcntl(h_, F_FULLFSYNC) && ::fsync(h_)) return {{}, last_error("fsync")}; return {};
#else
        if (::fsync(h_)) return {{}, last_error("fsync")}; return {};
#endif
    }
private:
    native_t h_ = bad();
};

inline result<std::string> read_file(const std::string& path) {
    auto f = file::open(path, rd); if (!f) return {{}, f.err};
    std::string s; char buf[65536];
    for (;;) { auto r = f->read(buf, sizeof buf); if (!r) return {{}, r.err}; if (!r.value) break; s.append(buf, r.value); }
    return {std::move(s), {}};
}
inline status write_file(const std::string& path, std::string_view data) {
    auto f = file::open(path, wr | create | truncate); if (!f) return {{}, f.err};
    return f->write_all(data.data(), data.size());
}

// ---------------------------------------------------------------- filesystem
struct file_info { bool exists = false, is_dir = false, is_file = false, is_link = false; std::uint64_t size = 0; std::int64_t mtime_ns = 0; };
inline file_info stat(const std::string& path) {
    file_info i;
#if defined(FORGE_OS_WINDOWS)
    WIN32_FILE_ATTRIBUTE_DATA d{};
    if (!GetFileAttributesExW(widen(path).c_str(), GetFileExInfoStandard, &d)) return i;
    i.exists = true; i.is_dir = d.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY; i.is_file = !i.is_dir; i.is_link = d.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT;
    i.size = (std::uint64_t(d.nFileSizeHigh) << 32) | d.nFileSizeLow;
    std::uint64_t t = (std::uint64_t(d.ftLastWriteTime.dwHighDateTime) << 32) | d.ftLastWriteTime.dwLowDateTime;
    i.mtime_ns = (std::int64_t(t) - 116444736000000000LL) * 100;
#else
    struct stat st{}, lst{};
    if (::stat(path.c_str(), &st)) return i;
    i.exists = true; i.is_dir = S_ISDIR(st.st_mode); i.is_file = S_ISREG(st.st_mode); i.size = std::uint64_t(st.st_size);
    if (!::lstat(path.c_str(), &lst)) i.is_link = S_ISLNK(lst.st_mode);
#if defined(__APPLE__)
    i.mtime_ns = std::int64_t(st.st_mtimespec.tv_sec) * 1000000000 + st.st_mtimespec.tv_nsec;
#else
    i.mtime_ns = std::int64_t(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec;
#endif
#endif
    return i;
}
inline bool exists(const std::string& p) { return stat(p).exists; }
inline status mkdir(const std::string& p) {
#if defined(FORGE_OS_WINDOWS)
    if (!CreateDirectoryW(widen(p).c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) return {{}, last_error("mkdir " + p)};
#else
    if (::mkdir(p.c_str(), 0755) && errno != EEXIST) return {{}, last_error("mkdir " + p)};
#endif
    return {};
}
inline bool is_sep(char c) {
#if defined(FORGE_OS_WINDOWS)
    return c == '/' || c == '\\';
#else
    return c == '/';
#endif
}
inline status mkdirs(const std::string& p) {
    for (std::size_t i = 1; i <= p.size(); ++i)
        if (i == p.size() || is_sep(p[i])) {
            std::string sub = p.substr(0, i);
            if (sub.empty() || (sub.size() == 2 && sub[1] == ':')) continue;
            if (!stat(sub).is_dir) if (auto s = mkdir(sub); !s) return s;
        }
    return {};
}
inline status remove(const std::string& p) {
#if defined(FORGE_OS_WINDOWS)
    auto w = widen(p);
    if (stat(p).is_dir ? !RemoveDirectoryW(w.c_str()) : !DeleteFileW(w.c_str())) return {{}, last_error("remove " + p)};
#else
    if (::remove(p.c_str())) return {{}, last_error("remove " + p)};
#endif
    return {};
}
inline status rename(const std::string& from, const std::string& to) {   // atomic replace where the OS allows
#if defined(FORGE_OS_WINDOWS)
    if (!MoveFileExW(widen(from).c_str(), widen(to).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return {{}, last_error("rename " + from)};
#else
    if (::rename(from.c_str(), to.c_str())) return {{}, last_error("rename " + from)};
#endif
    return {};
}
struct dir_entry { std::string name; bool is_dir = false; };
inline result<std::vector<dir_entry>> list_dir(const std::string& p) {
    std::vector<dir_entry> out;
#if defined(FORGE_OS_WINDOWS)
    WIN32_FIND_DATAW d{}; HANDLE h = FindFirstFileW(widen(p + "\\*").c_str(), &d);
    if (h == INVALID_HANDLE_VALUE) return {{}, last_error("list " + p)};
    do { std::string n = narrow(d.cFileName); if (n != "." && n != "..") out.push_back({n, bool(d.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)}); } while (FindNextFileW(h, &d));
    FindClose(h);
#else
    DIR* d = ::opendir(p.c_str()); if (!d) return {{}, last_error("list " + p)};
    while (auto* e = ::readdir(d)) {
        std::string n = e->d_name; if (n == "." || n == "..") continue;
        bool dir = e->d_type == DT_DIR; if (e->d_type == DT_UNKNOWN) dir = stat(p + "/" + n).is_dir;
        out.push_back({n, dir});
    }
    ::closedir(d);
#endif
    return {std::move(out), {}};
}
inline status remove_all(const std::string& p) {
    auto i = stat(p); if (!i.exists) return {};
    if (i.is_dir && !i.is_link) { auto l = list_dir(p); if (!l) return {{}, l.err}; for (auto& e : l.value) if (auto s = remove_all(p + "/" + e.name); !s) return s; }
    return remove(p);
}

// ---------------------------------------------------------------- paths
inline std::string join(std::string_view a, std::string_view b) {
    if (a.empty()) return std::string(b); if (b.empty()) return std::string(a);
    std::string s(a); if (!is_sep(s.back())) s += '/'; s += b; return s;
}
inline std::string basename(std::string_view p) { std::size_t i = p.size(); while (i && !is_sep(p[i - 1])) --i; return std::string(p.substr(i)); }
inline std::string dirname(std::string_view p) { std::size_t i = p.size(); while (i && !is_sep(p[i - 1])) --i; if (!i) return "."; return std::string(p.substr(0, i > 1 ? i - 1 : 1)); }
inline std::string temp_dir() {
#if defined(FORGE_OS_WINDOWS)
    wchar_t b[MAX_PATH + 1]; DWORD n = GetTempPathW(MAX_PATH + 1, b); std::string s = narrow({b, n}); while (s.size() > 3 && is_sep(s.back())) s.pop_back(); return s;
#else
    for (const char* v : {"TMPDIR", "TMP", "TEMP"}) if (const char* t = std::getenv(v); t && *t) { std::string s = t; while (s.size() > 1 && s.back() == '/') s.pop_back(); return s; }
#if defined(FORGE_OS_ANDROID)
    return "/data/local/tmp";
#else
    return "/tmp";
#endif
#endif
}
inline std::string cwd() {
#if defined(FORGE_OS_WINDOWS)
    wchar_t b[32768]; DWORD n = GetCurrentDirectoryW(32768, b); return narrow({b, n});
#else
    char b[4096]; return ::getcwd(b, sizeof b) ? b : "";
#endif
}
inline status chdir(const std::string& p) {
#if defined(FORGE_OS_WINDOWS)
    if (!SetCurrentDirectoryW(widen(p).c_str())) return {{}, last_error("chdir")};
#else
    if (::chdir(p.c_str())) return {{}, last_error("chdir")};
#endif
    return {};
}
inline std::string exe_path() {
#if defined(FORGE_OS_WINDOWS)
    wchar_t b[32768]; DWORD n = GetModuleFileNameW(nullptr, b, 32768); return narrow({b, n});
#elif defined(__APPLE__)
    char b[4096]; std::uint32_t n = sizeof b; if (_NSGetExecutablePath(b, &n)) return ""; char r[4096]; return ::realpath(b, r) ? r : b;
#elif defined(FORGE_OS_FREEBSD)
    int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1}; char b[4096]; std::size_t n = sizeof b;
    return ::sysctl(mib, 4, b, &n, nullptr, 0) ? "" : b;
#else
    char b[4096]; auto n = ::readlink("/proc/self/exe", b, sizeof b - 1); return n > 0 ? std::string(b, std::size_t(n)) : "";
#endif
}
inline std::string absolute(const std::string& p) {
#if defined(FORGE_OS_WINDOWS)
    wchar_t b[32768]; DWORD n = GetFullPathNameW(widen(p).c_str(), 32768, b, nullptr); return n ? narrow({b, n}) : p;
#else
    char b[PATH_MAX]; if (::realpath(p.c_str(), b)) return b; return p.size() && p[0] == '/' ? p : join(cwd(), p);
#endif
}
inline std::string getenv(const std::string&, const std::string& = {});
inline std::string home_dir() {
#if defined(FORGE_OS_WINDOWS)
    return getenv("USERPROFILE");
#else
    return getenv("HOME");
#endif
}

// ---------------------------------------------------------------- memory mapping
enum prot : unsigned { prot_read = 1, prot_write = 2, prot_exec = 4 };
class mapping {
public:
    mapping() = default;
    mapping(mapping&& o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
    mapping& operator=(mapping&& o) noexcept { if (this != &o) { reset(); p_ = std::exchange(o.p_, nullptr); n_ = std::exchange(o.n_, 0); } return *this; }
    ~mapping() { reset(); }
    std::uint8_t* data() const { return static_cast<std::uint8_t*>(p_); }
    std::size_t size() const { return n_; }
    void* release() { n_ = 0; return std::exchange(p_, nullptr); }
    void reset() {
        if (!p_) return;
#if defined(FORGE_OS_WINDOWS)
        UnmapViewOfFile(p_);
#else
        ::munmap(p_, n_);
#endif
        p_ = nullptr; n_ = 0;
    }
    // Map a file (whole file if len == 0). Writable maps are shared (writes reach the file).
    static result<mapping> map_file(const std::string& path, unsigned pr = prot_read, std::uint64_t off = 0, std::size_t len = 0) {
        auto f = file::open(path, (pr & prot_write) ? rdwr : rd); if (!f) return {{}, f.err};
        auto sz = f->size(); if (!sz) return {{}, sz.err};
        if (!len) len = std::size_t(sz.value - off);
        mapping m; if (!len) return {std::move(m), {}};
#if defined(FORGE_OS_WINDOWS)
        HANDLE h = CreateFileMappingW(f->native(), nullptr, (pr & prot_write) ? PAGE_READWRITE : PAGE_READONLY, 0, 0, nullptr);
        if (!h) return {{}, last_error("CreateFileMapping " + path)};
        void* p = MapViewOfFile(h, (pr & prot_write) ? FILE_MAP_WRITE : FILE_MAP_READ, DWORD(off >> 32), DWORD(off), len);
        CloseHandle(h);
        if (!p) return {{}, last_error("MapViewOfFile " + path)};
#else
        int pp = ((pr & prot_read) ? PROT_READ : 0) | ((pr & prot_write) ? PROT_WRITE : 0);
        void* p = ::mmap(nullptr, len, pp, (pr & prot_write) ? MAP_SHARED : MAP_PRIVATE, f->native(), off_t(off));
        if (p == MAP_FAILED) return {{}, last_error("mmap " + path)};
#endif
        m.p_ = p; m.n_ = len; return {std::move(m), {}};
    }
    static result<mapping> anonymous(std::size_t len, unsigned pr = prot_read | prot_write) {
        mapping m;
#if defined(FORGE_OS_WINDOWS)
        (void)pr; HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, DWORD(std::uint64_t(len) >> 32), DWORD(len), nullptr);
        if (!h) return {{}, last_error("anonymous map")};
        void* p = MapViewOfFile(h, FILE_MAP_WRITE, 0, 0, len); CloseHandle(h);
        if (!p) return {{}, last_error("anonymous map")};
#else
        int pp = ((pr & prot_read) ? PROT_READ : 0) | ((pr & prot_write) ? PROT_WRITE : 0);
        void* p = ::mmap(nullptr, len, pp, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p == MAP_FAILED) return {{}, last_error("mmap anon")};
#endif
        m.p_ = p; m.n_ = len; return {std::move(m), {}};
    }
    status flush() {
#if defined(FORGE_OS_WINDOWS)
        if (p_ && !FlushViewOfFile(p_, n_)) return {{}, last_error("flush")};
#else
        if (p_ && ::msync(p_, n_, MS_SYNC)) return {{}, last_error("msync")};
#endif
        return {};
    }
private:
    void* p_ = nullptr; std::size_t n_ = 0;
};

// ---------------------------------------------------------------- memory
inline std::size_t page_size() {
#if defined(FORGE_OS_WINDOWS)
    SYSTEM_INFO s; GetSystemInfo(&s); return s.dwPageSize;
#else
    return std::size_t(::sysconf(_SC_PAGESIZE));
#endif
}
inline void* aligned_alloc(std::size_t n, std::size_t align) {
#if defined(FORGE_OS_WINDOWS)
    return _aligned_malloc(n, align);
#else
    void* p = nullptr; return ::posix_memalign(&p, align < sizeof(void*) ? sizeof(void*) : align, n) ? nullptr : p;
#endif
}
inline void aligned_free(void* p) {
#if defined(FORGE_OS_WINDOWS)
    _aligned_free(p);
#else
    std::free(p);
#endif
}
inline void* reserve(std::size_t n) {   // address space only; commit() before touching
#if defined(FORGE_OS_WINDOWS)
    return VirtualAlloc(nullptr, n, MEM_RESERVE, PAGE_NOACCESS);
#else
    void* p = ::mmap(nullptr, n, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0); return p == MAP_FAILED ? nullptr : p;
#endif
}
inline bool commit(void* p, std::size_t n) {
#if defined(FORGE_OS_WINDOWS)
    return VirtualAlloc(p, n, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
    return ::mprotect(p, n, PROT_READ | PROT_WRITE) == 0;
#endif
}
inline void release(void* p, std::size_t n) {
#if defined(FORGE_OS_WINDOWS)
    (void)n; VirtualFree(p, 0, MEM_RELEASE);
#else
    ::munmap(p, n);
#endif
}
inline bool protect(void* p, std::size_t n, unsigned pr) {
#if defined(FORGE_OS_WINDOWS)
    DWORD f = (pr & prot_exec) ? ((pr & prot_write) ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ) : (pr & prot_write) ? PAGE_READWRITE : (pr & prot_read) ? PAGE_READONLY : PAGE_NOACCESS;
    DWORD old; return VirtualProtect(p, n, f, &old) != 0;
#else
    int f = ((pr & prot_read) ? PROT_READ : 0) | ((pr & prot_write) ? PROT_WRITE : 0) | ((pr & prot_exec) ? PROT_EXEC : 0);
    return ::mprotect(p, n, f) == 0;
#endif
}
inline bool lock_memory(void* p, std::size_t n) {
#if defined(FORGE_OS_WINDOWS)
    return VirtualLock(p, n) != 0;
#else
    return ::mlock(p, n) == 0;
#endif
}
inline void unlock_memory(void* p, std::size_t n) {
#if defined(FORGE_OS_WINDOWS)
    VirtualUnlock(p, n);
#else
    ::munlock(p, n);
#endif
}
inline void advise_willneed(void* p, std::size_t n) {
#if defined(FORGE_OS_POSIX)
    ::madvise(p, n, MADV_WILLNEED);
#else
    WIN32_MEMORY_RANGE_ENTRY r{p, n}; PrefetchVirtualMemory(GetCurrentProcess(), 1, &r, 0);
#endif
}

// ---------------------------------------------------------------- dynamic libraries
class library {
public:
    library() = default;
    library(library&& o) noexcept : h_(std::exchange(o.h_, nullptr)) {}
    library& operator=(library&& o) noexcept { if (this != &o) { close(); h_ = std::exchange(o.h_, nullptr); } return *this; }
    ~library() { close(); }
    static result<library> open(const std::string& name) {   // empty name = the running program
        library l;
#if defined(FORGE_OS_WINDOWS)
        l.h_ = name.empty() ? static_cast<void*>(GetModuleHandleW(nullptr)) : static_cast<void*>(LoadLibraryW(widen(name).c_str()));
        if (!l.h_) return {{}, last_error("LoadLibrary " + name)};
#else
        l.h_ = ::dlopen(name.empty() ? nullptr : name.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!l.h_) { const char* e = ::dlerror(); return {{}, {-1, e ? e : ("dlopen " + name)}}; }
#endif
        l.self_ = name.empty(); return {std::move(l), {}};
    }
    void* symbol(const char* s) const {
        if (!h_) return nullptr;
#if defined(FORGE_OS_WINDOWS)
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(h_), s));
#else
        return ::dlsym(h_, s);
#endif
    }
    template <class F> F get(const char* s) const { return reinterpret_cast<F>(symbol(s)); }
    bool valid() const { return h_ != nullptr; }
    void* native() const { return h_; }
    void close() {
        if (!h_) return;
#if defined(FORGE_OS_WINDOWS)
        if (!self_) FreeLibrary(static_cast<HMODULE>(h_));
#else
        ::dlclose(h_);
#endif
        h_ = nullptr;
    }
    // Conventional file name: "foo" -> libfoo.so / libfoo.dylib / foo.dll
    static std::string decorate(std::string_view base) {
#if defined(FORGE_OS_WINDOWS)
        return std::string(base) + ".dll";
#elif defined(__APPLE__)
        return "lib" + std::string(base) + ".dylib";
#else
        return "lib" + std::string(base) + ".so";
#endif
    }
private:
    void* h_ = nullptr; bool self_ = false;
};

// ---------------------------------------------------------------- environment
inline std::string getenv(const std::string& k, const std::string& dflt) {
#if defined(FORGE_OS_WINDOWS)
    wchar_t b[32767]; DWORD n = GetEnvironmentVariableW(widen(k).c_str(), b, 32767); return n ? narrow({b, n}) : dflt;
#else
    const char* v = std::getenv(k.c_str()); return v ? v : dflt;
#endif
}
inline bool has_env(const std::string& k) {
#if defined(FORGE_OS_WINDOWS)
    return GetEnvironmentVariableW(widen(k).c_str(), nullptr, 0) != 0;
#else
    return std::getenv(k.c_str()) != nullptr;
#endif
}
inline bool setenv(const std::string& k, const std::string& v) {
#if defined(FORGE_OS_WINDOWS)
    return SetEnvironmentVariableW(widen(k).c_str(), widen(v).c_str()) && _putenv_s(k.c_str(), v.c_str()) == 0;
#else
    return ::setenv(k.c_str(), v.c_str(), 1) == 0;
#endif
}
inline bool unsetenv(const std::string& k) {
#if defined(FORGE_OS_WINDOWS)
    return SetEnvironmentVariableW(widen(k).c_str(), nullptr) && _putenv_s(k.c_str(), "") == 0;
#else
    return ::unsetenv(k.c_str()) == 0;
#endif
}

// ---------------------------------------------------------------- time
inline std::uint64_t monotonic_ns() {
#if defined(FORGE_OS_WINDOWS)
    static const std::uint64_t f = [] { LARGE_INTEGER q; QueryPerformanceFrequency(&q); return std::uint64_t(q.QuadPart); }();
    LARGE_INTEGER c; QueryPerformanceCounter(&c); std::uint64_t t = std::uint64_t(c.QuadPart);
    return t / f * 1000000000ull + t % f * 1000000000ull / f;
#elif defined(__APPLE__)
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    timespec ts; ::clock_gettime(CLOCK_MONOTONIC, &ts); return std::uint64_t(ts.tv_sec) * 1000000000ull + std::uint64_t(ts.tv_nsec);
#endif
}
inline std::int64_t wall_ns() {   // Unix epoch
#if defined(FORGE_OS_WINDOWS)
    FILETIME ft; GetSystemTimePreciseAsFileTime(&ft);
    return (std::int64_t((std::uint64_t(ft.dwHighDateTime) << 32) | ft.dwLowDateTime) - 116444736000000000LL) * 100;
#else
    timespec ts; ::clock_gettime(CLOCK_REALTIME, &ts); return std::int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
#endif
}
inline std::uint64_t cpu_time_ns() {   // this process, user + system
#if defined(FORGE_OS_WINDOWS)
    FILETIME c, e, k, u; GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    auto v = [](FILETIME f) { return (std::uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
    return (v(k) + v(u)) * 100;
#else
    timespec ts; ::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts); return std::uint64_t(ts.tv_sec) * 1000000000ull + std::uint64_t(ts.tv_nsec);
#endif
}
inline void sleep_ns(std::uint64_t ns) {
#if defined(FORGE_OS_WINDOWS)
    Sleep(DWORD((ns + 999999) / 1000000));
#else
    timespec ts{time_t(ns / 1000000000ull), long(ns % 1000000000ull)}; while (::nanosleep(&ts, &ts) && errno == EINTR) {}
#endif
}

// ---------------------------------------------------------------- threads
inline std::uint64_t thread_id() {
#if defined(FORGE_OS_WINDOWS)
    return GetCurrentThreadId();
#elif defined(__APPLE__)
    std::uint64_t t = 0; pthread_threadid_np(nullptr, &t); return t;
#elif defined(FORGE_OS_FREEBSD)
    return std::uint64_t(pthread_getthreadid_np());
#else
    return std::uint64_t(::gettid());
#endif
}
inline void yield() {
#if defined(FORGE_OS_WINDOWS)
    SwitchToThread();
#else
    ::sched_yield();
#endif
}
inline void set_thread_name(const std::string& n) {
#if defined(FORGE_OS_WINDOWS)
    using fn = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    if (auto f = reinterpret_cast<fn>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription")))) f(GetCurrentThread(), widen(n).c_str());
#elif defined(__APPLE__)
    pthread_setname_np(n.substr(0, 63).c_str());
#elif defined(FORGE_OS_FREEBSD)
    pthread_set_name_np(pthread_self(), n.c_str());
#else
    pthread_setname_np(pthread_self(), n.substr(0, 15).c_str());
#endif
}
enum class thread_priority { low, normal, high };
inline bool set_thread_priority(thread_priority p) {
#if defined(FORGE_OS_WINDOWS)
    return SetThreadPriority(GetCurrentThread(), p == thread_priority::low ? THREAD_PRIORITY_BELOW_NORMAL : p == thread_priority::high ? THREAD_PRIORITY_ABOVE_NORMAL : THREAD_PRIORITY_NORMAL) != 0;
#elif defined(__APPLE__)
    return pthread_set_qos_class_self_np(p == thread_priority::low ? QOS_CLASS_UTILITY : p == thread_priority::high ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_DEFAULT, 0) == 0;
#else
    // nice-level of the calling thread on Linux/Android (per-thread); raising priority usually needs privileges
    return ::setpriority(PRIO_PROCESS, 0, p == thread_priority::low ? 10 : p == thread_priority::high ? -5 : 0) == 0 || p == thread_priority::high;
#endif
}
inline bool pin_thread(unsigned cpu) {
#if defined(FORGE_OS_WINDOWS)
    return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu) != 0;
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); return ::sched_setaffinity(0, sizeof s, &s) == 0;
#else
    (void)cpu; return false;   // macOS/iOS have no hard affinity; FreeBSD: cpuset_setaffinity (not wired)
#endif
}
class thread {
public:
    thread() = default;
    explicit thread(std::function<void()> f, std::string name = {}) {
        auto* st = new start{std::move(f), std::move(name)};
#if defined(FORGE_OS_WINDOWS)
        h_ = reinterpret_cast<HANDLE>(_beginthreadex(nullptr, 0, &entry, st, 0, nullptr));
#else
        if (pthread_create(&t_, nullptr, &entry, st) == 0) live_ = true;
        else delete st;
#endif
    }
    thread(thread&& o) noexcept { *this = std::move(o); }
    thread& operator=(thread&& o) noexcept {
#if defined(FORGE_OS_WINDOWS)
        h_ = std::exchange(o.h_, nullptr);
#else
        t_ = o.t_; live_ = std::exchange(o.live_, false);
#endif
        return *this;
    }
    ~thread() { join(); }
    bool joinable() const {
#if defined(FORGE_OS_WINDOWS)
        return h_ != nullptr;
#else
        return live_;
#endif
    }
    void join() {
#if defined(FORGE_OS_WINDOWS)
        if (h_) { WaitForSingleObject(h_, INFINITE); CloseHandle(h_); h_ = nullptr; }
#else
        if (live_) { pthread_join(t_, nullptr); live_ = false; }
#endif
    }
private:
    struct start { std::function<void()> f; std::string name; };
#if defined(FORGE_OS_WINDOWS)
    static unsigned __stdcall entry(void* p) { auto* s = static_cast<start*>(p); if (!s->name.empty()) set_thread_name(s->name); s->f(); delete s; return 0; }
    HANDLE h_ = nullptr;
#else
    static void* entry(void* p) { auto* s = static_cast<start*>(p); if (!s->name.empty()) set_thread_name(s->name); s->f(); delete s; return nullptr; }
    pthread_t t_{}; bool live_ = false;
#endif
};

// ---------------------------------------------------------------- sync
class mutex {
public:
    mutex() {
#if defined(FORGE_OS_WINDOWS)
        InitializeSRWLock(&m_);
#else
        pthread_mutex_init(&m_, nullptr);
#endif
    }
    ~mutex() {
#if defined(FORGE_OS_POSIX)
        pthread_mutex_destroy(&m_);
#endif
    }
    mutex(const mutex&) = delete; mutex& operator=(const mutex&) = delete;
#if defined(FORGE_OS_WINDOWS)
    void lock() { AcquireSRWLockExclusive(&m_); }
    bool try_lock() { return TryAcquireSRWLockExclusive(&m_); }
    void unlock() { ReleaseSRWLockExclusive(&m_); }
    SRWLOCK m_;
#else
    void lock() { pthread_mutex_lock(&m_); }
    bool try_lock() { return pthread_mutex_trylock(&m_) == 0; }
    void unlock() { pthread_mutex_unlock(&m_); }
    pthread_mutex_t m_;
#endif
};
class condvar {
public:
    condvar() {
#if defined(FORGE_OS_WINDOWS)
        InitializeConditionVariable(&c_);
#elif defined(__APPLE__)
        pthread_cond_init(&c_, nullptr);
#else
        pthread_condattr_t a; pthread_condattr_init(&a); pthread_condattr_setclock(&a, CLOCK_MONOTONIC); pthread_cond_init(&c_, &a); pthread_condattr_destroy(&a);
#endif
    }
    ~condvar() {
#if defined(FORGE_OS_POSIX)
        pthread_cond_destroy(&c_);
#endif
    }
    condvar(const condvar&) = delete; condvar& operator=(const condvar&) = delete;
    void notify_one() {
#if defined(FORGE_OS_WINDOWS)
        WakeConditionVariable(&c_);
#else
        pthread_cond_signal(&c_);
#endif
    }
    void notify_all() {
#if defined(FORGE_OS_WINDOWS)
        WakeAllConditionVariable(&c_);
#else
        pthread_cond_broadcast(&c_);
#endif
    }
    void wait(mutex& m) {
#if defined(FORGE_OS_WINDOWS)
        SleepConditionVariableSRW(&c_, &m.m_, INFINITE, 0);
#else
        pthread_cond_wait(&c_, &m.m_);
#endif
    }
    // false on timeout
    bool wait_for(mutex& m, std::uint64_t ns) {
#if defined(FORGE_OS_WINDOWS)
        return SleepConditionVariableSRW(&c_, &m.m_, DWORD((ns + 999999) / 1000000), 0) != 0;
#elif defined(__APPLE__)
        timespec ts{time_t(ns / 1000000000ull), long(ns % 1000000000ull)};
        return pthread_cond_timedwait_relative_np(&c_, &m.m_, &ts) == 0;
#else
        timespec ts; ::clock_gettime(CLOCK_MONOTONIC, &ts);
        std::uint64_t t = std::uint64_t(ts.tv_sec) * 1000000000ull + std::uint64_t(ts.tv_nsec) + ns;
        ts.tv_sec = time_t(t / 1000000000ull); ts.tv_nsec = long(t % 1000000000ull);
        return pthread_cond_timedwait(&c_, &m.m_, &ts) == 0;
#endif
    }
private:
#if defined(FORGE_OS_WINDOWS)
    CONDITION_VARIABLE c_;
#else
    pthread_cond_t c_;
#endif
};
struct lock_guard { mutex& m; explicit lock_guard(mutex& x) : m(x) { m.lock(); } ~lock_guard() { m.unlock(); } };
class semaphore {
public:
    explicit semaphore(unsigned initial = 0) : n_(initial) {}
    void post(unsigned k = 1) { lock_guard g(m_); n_ += k; if (k == 1) c_.notify_one(); else c_.notify_all(); }
    void wait() { lock_guard g(m_); while (!n_) c_.wait(m_); --n_; }
    bool wait_for(std::uint64_t ns) {
        std::uint64_t end = monotonic_ns() + ns; lock_guard g(m_);
        while (!n_) { auto now = monotonic_ns(); if (now >= end || (!c_.wait_for(m_, end - now) && !n_ && monotonic_ns() >= end)) return false; }
        --n_; return true;
    }
private:
    mutex m_; condvar c_; unsigned n_;
};

// ---------------------------------------------------------------- processes
inline std::uint64_t pid() {
#if defined(FORGE_OS_WINDOWS)
    return GetCurrentProcessId();
#else
    return std::uint64_t(::getpid());
#endif
}
struct exec_result { int exit_code = -1; std::string out; };
// Run argv[0] with args, capture stdout (+ stderr when merge_stderr), wait. iOS cannot spawn processes.
inline result<exec_result> run(const std::vector<std::string>& argv, bool merge_stderr = false) {
    if (argv.empty()) return {{}, {-1, "run: empty argv"}};
#if defined(FORGE_OS_IOS)
    (void)merge_stderr; return {{}, {-1, "run: process spawning is not available on iOS"}};
#elif defined(FORGE_OS_WINDOWS)
    std::wstring cmd;
    for (auto& a : argv) {
        if (!cmd.empty()) cmd += L' ';
        std::wstring w = widen(a);
        if (!w.empty() && w.find_first_of(L" \t\"") == std::wstring::npos) { cmd += w; continue; }
        cmd += L'"'; std::size_t bs = 0;
        for (wchar_t c : w) { if (c == L'\\') { ++bs; continue; } if (c == L'"') cmd.append(bs * 2 + 1, L'\\'); else cmd.append(bs, L'\\'); bs = 0; cmd += c; }
        cmd.append(bs * 2, L'\\'); cmd += L'"';
    }
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE}; HANDLE r, w;
    if (!CreatePipe(&r, &w, &sa, 0)) return {{}, last_error("CreatePipe")};
    SetHandleInformation(r, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{}; si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = w; si.hStdError = merge_stderr ? w : GetStdHandle(STD_ERROR_HANDLE); si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(w);
    if (!ok) { auto e = last_error("CreateProcess " + argv[0]); CloseHandle(r); return {{}, e}; }
    exec_result res; char buf[4096]; DWORD n;
    while (ReadFile(r, buf, sizeof buf, &n, nullptr) && n) res.out.append(buf, n);
    CloseHandle(r); WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0; GetExitCodeProcess(pi.hProcess, &code); res.exit_code = int(code);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return {std::move(res), {}};
#else
    int fds[2]; if (::pipe(fds)) return {{}, last_error("pipe")};
    std::vector<char*> av; for (auto& a : argv) av.push_back(const_cast<char*>(a.c_str())); av.push_back(nullptr);
    pid_t p; int rc = 0;
#if defined(FORGE_OS_ANDROID) && __ANDROID_API__ < 28   // posix_spawn arrives in API 28
    p = ::fork();
    if (p == 0) { ::dup2(fds[1], 1); if (merge_stderr) ::dup2(fds[1], 2); ::close(fds[0]); ::close(fds[1]); ::execvp(av[0], av.data()); ::_exit(127); }
    if (p < 0) rc = errno;
    ::close(fds[1]);
#else
    posix_spawn_file_actions_t fa; posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
    if (merge_stderr) posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
    posix_spawn_file_actions_addclose(&fa, fds[0]); posix_spawn_file_actions_addclose(&fa, fds[1]);
    rc = ::posix_spawnp(&p, av[0], &fa, nullptr, av.data(), environ);
    posix_spawn_file_actions_destroy(&fa); ::close(fds[1]);
#endif
    if (rc) { ::close(fds[0]); return {{}, {rc, "spawn " + argv[0] + ": " + error_text(rc)}}; }
    exec_result res; char buf[4096];
    for (;;) { auto n = ::read(fds[0], buf, sizeof buf); if (n > 0) res.out.append(buf, std::size_t(n)); else if (n == 0 || errno != EINTR) break; }
    ::close(fds[0]);
    int st = 0; while (::waitpid(p, &st, 0) < 0 && errno == EINTR) {}
    res.exit_code = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
    return {std::move(res), {}};
#endif
}

// ---------------------------------------------------------------- signals
enum class sig { interrupt, terminate, hangup, pipe };
// Install a handler for Ctrl-C / termination. The handler runs in signal context (POSIX) or on a
// console-control thread (Windows): keep it to setting atomics.
inline bool on_signal(sig s, void (*handler)(int)) {
#if defined(FORGE_OS_WINDOWS)
    static void (*h_int)(int) = nullptr; static void (*h_term)(int) = nullptr;
    if (s == sig::interrupt) h_int = handler; else if (s == sig::terminate) h_term = handler; else return s == sig::pipe;
    return SetConsoleCtrlHandler([](DWORD t) -> BOOL {
        if (t == CTRL_C_EVENT && h_int) { h_int(2); return TRUE; }
        if ((t == CTRL_CLOSE_EVENT || t == CTRL_BREAK_EVENT || t == CTRL_SHUTDOWN_EVENT) && h_term) { h_term(15); return TRUE; }
        return FALSE; }, TRUE) != 0;
#else
    int n = s == sig::interrupt ? SIGINT : s == sig::terminate ? SIGTERM : s == sig::hangup ? SIGHUP : SIGPIPE;
    struct sigaction a{}; a.sa_handler = handler ? handler : SIG_DFL; sigemptyset(&a.sa_mask); a.sa_flags = SA_RESTART;
    return ::sigaction(n, &a, nullptr) == 0;
#endif
}
inline void ignore_sigpipe() {
#if defined(FORGE_OS_POSIX)
    ::signal(SIGPIPE, SIG_IGN);
#endif
}

// ---------------------------------------------------------------- system / CPU info
struct system_info {
    std::string os, arch, hostname;
    unsigned logical_cpus = 1;
    std::uint64_t total_ram = 0, avail_ram = 0;
    std::size_t page = 4096;
};
inline unsigned cpu_count() {
#if defined(FORGE_OS_WINDOWS)
    DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS); return n ? n : 1;
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
    cpu_set_t s; if (::sched_getaffinity(0, sizeof s, &s) == 0) { int n = CPU_COUNT(&s); if (n > 0) return unsigned(n); }
    long n = ::sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? unsigned(n) : 1;
#else
    long n = ::sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? unsigned(n) : 1;
#endif
}
inline std::uint64_t total_memory() {
#if defined(FORGE_OS_WINDOWS)
    MEMORYSTATUSEX m{}; m.dwLength = sizeof m; GlobalMemoryStatusEx(&m); return m.ullTotalPhys;
#elif defined(__APPLE__)
    std::uint64_t v = 0; std::size_t n = sizeof v; ::sysctlbyname("hw.memsize", &v, &n, nullptr, 0); return v;
#elif defined(FORGE_OS_FREEBSD)
    std::uint64_t v = 0; std::size_t n = sizeof v; ::sysctlbyname("hw.physmem", &v, &n, nullptr, 0); return v;
#else
    return std::uint64_t(::sysconf(_SC_PHYS_PAGES)) * std::uint64_t(::sysconf(_SC_PAGESIZE));
#endif
}
inline std::uint64_t available_memory() {
#if defined(FORGE_OS_WINDOWS)
    MEMORYSTATUSEX m{}; m.dwLength = sizeof m; GlobalMemoryStatusEx(&m); return m.ullAvailPhys;
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
    if (auto s = read_file("/proc/meminfo")) { auto p = s->find("MemAvailable:"); if (p != std::string::npos) return std::strtoull(s->c_str() + p + 13, nullptr, 10) * 1024; }
    return std::uint64_t(::sysconf(_SC_AVPHYS_PAGES)) * std::uint64_t(::sysconf(_SC_PAGESIZE));
#elif defined(__APPLE__)
    std::uint64_t v = 0; std::size_t n = sizeof v;   // best effort: page-ins free memory is a mach query; report total/4 floor
    ::sysctlbyname("hw.memsize", &v, &n, nullptr, 0); return v / 4;
#else
    unsigned v = 0; std::size_t n = sizeof v; ::sysctlbyname("vm.stats.vm.v_free_count", &v, &n, nullptr, 0); return std::uint64_t(v) * page_size();
#endif
}
inline std::string hostname() {
#if defined(FORGE_OS_WINDOWS)
    wchar_t b[256]; DWORD n = 256; return GetComputerNameExW(ComputerNameDnsHostname, b, &n) ? narrow({b, n}) : "";
#else
    char b[256] = {}; ::gethostname(b, sizeof b - 1); return b;
#endif
}
constexpr const char* arch_name() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__riscv) && __riscv_xlen == 64
    return "riscv64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__arm__)
    return "arm";
#else
    return "unknown";
#endif
}
inline system_info sysinfo() {
    return {platform_name(), arch_name(), hostname(), cpu_count(), total_memory(), available_memory(), page_size()};
}

// ---------------------------------------------------------------- entropy
inline bool entropy(void* p, std::size_t n) {
#if defined(FORGE_OS_WINDOWS)
    return BCryptGenRandom(nullptr, static_cast<PUCHAR>(p), static_cast<ULONG>(n), BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#elif defined(__APPLE__) || defined(FORGE_OS_ANDROID) || defined(FORGE_OS_FREEBSD)
    ::arc4random_buf(p, n); return true;
#else
    auto* b = static_cast<std::uint8_t*>(p);
    while (n) { auto r = ::getrandom(b, n, 0); if (r < 0 && errno == EINTR) continue; if (r <= 0) return false; b += r; n -= std::size_t(r); }
    return true;
#endif
}

// ---------------------------------------------------------------- sockets
#if defined(FORGE_OS_WINDOWS)
using sock_t = SOCKET;
inline const sock_t bad_sock = INVALID_SOCKET;
#else
using sock_t = int;
inline constexpr sock_t bad_sock = -1;
#endif
inline void net_init() {
#if defined(FORGE_OS_WINDOWS)
    static bool ok = [] { WSADATA d; return WSAStartup(MAKEWORD(2, 2), &d) == 0; }(); (void)ok;
#endif
}
inline int sock_errno() {
#if defined(FORGE_OS_WINDOWS)
    return WSAGetLastError();
#else
    return errno;
#endif
}
inline bool would_block(int e = sock_errno()) {
#if defined(FORGE_OS_WINDOWS)
    return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS;
#else
    return e == EAGAIN || e == EWOULDBLOCK || e == EINPROGRESS;
#endif
}
class socket {
public:
    socket() = default;
    explicit socket(sock_t s) : s_(s) {}
    socket(socket&& o) noexcept : s_(std::exchange(o.s_, bad_sock)) {}
    socket& operator=(socket&& o) noexcept { if (this != &o) { close(); s_ = std::exchange(o.s_, bad_sock); } return *this; }
    ~socket() { close(); }
    sock_t native() const { return s_; }
    sock_t release() { return std::exchange(s_, bad_sock); }
    bool valid() const { return s_ != bad_sock; }
    void close() {
        if (!valid()) return;
#if defined(FORGE_OS_WINDOWS)
        ::closesocket(s_);
#else
        ::close(s_);
#endif
        s_ = bad_sock;
    }
    bool set_nonblocking(bool on = true) {
#if defined(FORGE_OS_WINDOWS)
        u_long v = on; return ::ioctlsocket(s_, FIONBIO, &v) == 0;
#else
        int f = ::fcntl(s_, F_GETFL); return f >= 0 && ::fcntl(s_, F_SETFL, on ? (f | O_NONBLOCK) : (f & ~O_NONBLOCK)) == 0;
#endif
    }
    bool set_nodelay(bool on = true) { int v = on; return ::setsockopt(s_, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&v), sizeof v) == 0; }
    bool set_reuseaddr(bool on = true) { int v = on; return ::setsockopt(s_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&v), sizeof v) == 0; }
    bool set_timeout_ms(unsigned ms) {
#if defined(FORGE_OS_WINDOWS)
        DWORD v = ms;
#else
        timeval v{time_t(ms / 1000), suseconds_t((ms % 1000) * 1000)};
#endif
        return ::setsockopt(s_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&v), sizeof v) == 0 &&
               ::setsockopt(s_, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&v), sizeof v) == 0;
    }
    // >0 bytes, 0 orderly close, <0 error (check would_block())
    long send(const void* p, std::size_t n) {
#if defined(FORGE_OS_WINDOWS)
        return ::send(s_, static_cast<const char*>(p), int(n), 0);
#elif defined(MSG_NOSIGNAL)
        return long(::send(s_, p, n, MSG_NOSIGNAL));
#else
        return long(::send(s_, p, n, 0));   // Apple: SO_NOSIGPIPE set at creation
#endif
    }
    long recv(void* p, std::size_t n) {
#if defined(FORGE_OS_WINDOWS)
        return ::recv(s_, static_cast<char*>(p), int(n), 0);
#else
        for (;;) { auto r = ::recv(s_, p, n, 0); if (r >= 0 || errno != EINTR) return long(r); }
#endif
    }
    bool send_all(const void* p, std::size_t n) {
        auto* b = static_cast<const char*>(p);
        while (n) { long r = send(b, n); if (r <= 0) { if (r < 0 && would_block()) { wait_writable(1000); continue; } return false; } b += r; n -= std::size_t(r); }
        return true;
    }
    bool recv_all(void* p, std::size_t n) {
        auto* b = static_cast<char*>(p);
        while (n) { long r = recv(b, n); if (r <= 0) { if (r < 0 && would_block()) { wait_readable(1000); continue; } return false; } b += r; n -= std::size_t(r); }
        return true;
    }
    bool wait_readable(int ms) const { return poll1(POLLIN, ms); }
    bool wait_writable(int ms) const { return poll1(POLLOUT, ms); }
    int local_port() const {
        sockaddr_storage a{}; socklen_t l = sizeof a;
        if (::getsockname(s_, reinterpret_cast<sockaddr*>(&a), &l)) return -1;
        return a.ss_family == AF_INET6 ? ntohs(reinterpret_cast<sockaddr_in6*>(&a)->sin6_port) : ntohs(reinterpret_cast<sockaddr_in*>(&a)->sin_port);
    }
    static socket make(int family, int type) {
        net_init();
        sock_t s = ::socket(family, type, 0);
#if defined(__APPLE__)
        if (s != bad_sock) { int one = 1; ::setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one); }
#endif
#if defined(FORGE_OS_POSIX)
        if (s != bad_sock) ::fcntl(s, F_SETFD, FD_CLOEXEC);
#endif
        return socket(s);
    }
    // TCP listen on host:port (port 0 = ephemeral). host "" = all interfaces.
    static result<socket> listen(const std::string& host, int port, int backlog = 128) {
        auto ai = resolve(host.empty() ? nullptr : host.c_str(), port, true); if (!ai) return {{}, ai.err};
        for (auto* a = ai.value; a; a = a->ai_next) {
            socket s = make(a->ai_family, SOCK_STREAM); if (!s.valid()) continue;
            s.set_reuseaddr();
            if (::bind(s.s_, a->ai_addr, socklen_t(a->ai_addrlen)) == 0 && ::listen(s.s_, backlog) == 0) { ::freeaddrinfo(ai.value); return {std::move(s), {}}; }
        }
        ::freeaddrinfo(ai.value); return {{}, {sock_errno() ? sock_errno() : -1, "listen " + host + ":" + std::to_string(port) + " failed"}};
    }
    static result<socket> connect(const std::string& host, int port, int timeout_ms = 5000) {
        auto ai = resolve(host.c_str(), port, false); if (!ai) return {{}, ai.err};
        for (auto* a = ai.value; a; a = a->ai_next) {
            socket s = make(a->ai_family, SOCK_STREAM); if (!s.valid()) continue;
            s.set_nonblocking(true);
            int r = ::connect(s.s_, a->ai_addr, socklen_t(a->ai_addrlen));
            if (r != 0 && !(would_block() && s.wait_writable(timeout_ms) && s.so_error() == 0)) continue;
            s.set_nonblocking(false); s.set_nodelay();
            ::freeaddrinfo(ai.value); return {std::move(s), {}};
        }
        ::freeaddrinfo(ai.value); return {{}, {-1, "connect " + host + ":" + std::to_string(port) + " failed"}};
    }
    result<socket> accept() {
        for (;;) {
            sock_t c = ::accept(s_, nullptr, nullptr);
            if (c != bad_sock) {
#if defined(__APPLE__)
                int one = 1; ::setsockopt(c, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
                socket s(c); s.set_nodelay(); return {std::move(s), {}};
            }
#if defined(FORGE_OS_POSIX)
            if (errno == EINTR) continue;
#endif
            return {{}, {sock_errno(), "accept"}};
        }
    }
    int so_error() const { int e = 0; socklen_t l = sizeof e; ::getsockopt(s_, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&e), &l); return e; }
    void shutdown_both() {
#if defined(FORGE_OS_WINDOWS)
        ::shutdown(s_, SD_BOTH);
#else
        ::shutdown(s_, SHUT_RDWR);
#endif
    }
    void shutdown_write() {
#if defined(FORGE_OS_WINDOWS)
        ::shutdown(s_, SD_SEND);
#else
        ::shutdown(s_, SHUT_WR);
#endif
    }
    static result<addrinfo*> resolve(const char* host, int port, bool passive) {
        net_init();
        addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM; if (passive) hints.ai_flags = AI_PASSIVE;
        addrinfo* r = nullptr; std::string ps = std::to_string(port);
        if (int e = ::getaddrinfo(host, ps.c_str(), &hints, &r); e) return {nullptr, {e, std::string("resolve ") + (host ? host : "*") + ": " + ::gai_strerror(e)}};
        return {r, {}};
    }
private:
    bool poll1(short ev, int ms) const {
#if defined(FORGE_OS_WINDOWS)
        WSAPOLLFD p{s_, ev, 0}; return ::WSAPoll(&p, 1, ms) > 0;
#else
        pollfd p{s_, ev, 0}; for (;;) { int r = ::poll(&p, 1, ms); if (r >= 0) return r > 0; if (errno != EINTR) return false; }
#endif
    }
    sock_t s_ = bad_sock;
};

// ---------------------------------------------------------------- readiness polling
// epoll (Linux/Android), kqueue (macOS/iOS/FreeBSD), WSAPoll (Windows). Level-triggered.
enum poll_events : unsigned { readable = 1, writable = 2, hangup = 4 };
struct poll_event { sock_t fd; unsigned events; void* user; };
class poller {
public:
    poller() {
#if defined(FORGE_OS_EPOLL)
        q_ = ::epoll_create1(EPOLL_CLOEXEC);
#elif defined(FORGE_OS_KQUEUE)
        q_ = ::kqueue();
#endif
    }
    ~poller() {
        clear();
#if defined(FORGE_OS_EPOLL) || defined(FORGE_OS_KQUEUE)
        if (q_ >= 0) ::close(q_);
#endif
    }
    poller(const poller&) = delete; poller& operator=(const poller&) = delete;
    const char* backend() const {
#if defined(FORGE_OS_EPOLL)
        return "epoll";
#elif defined(FORGE_OS_KQUEUE)
        return "kqueue";
#elif defined(FORGE_OS_WINDOWS)
        return "wsapoll";
#else
        return "poll";
#endif
    }
    bool add(sock_t fd, unsigned ev, void* user = nullptr) { return ctl(fd, ev, user, 0); }
    bool modify(sock_t fd, unsigned ev, void* user = nullptr) { return ctl(fd, ev, user, 1); }
    bool remove(sock_t fd) { return ctl(fd, 0, nullptr, 2); }
    // Waits up to timeout_ms (-1 = forever). Returns the ready events.
    std::vector<poll_event> wait(int timeout_ms) {
        std::vector<poll_event> out;
#if defined(FORGE_OS_EPOLL)
        epoll_event ev[64]; int n = ::epoll_wait(q_, ev, 64, timeout_ms);
        for (int i = 0; i < n; ++i) {
            auto* r = static_cast<reg*>(ev[i].data.ptr);
            unsigned e = ((ev[i].events & EPOLLIN) ? readable : 0) | ((ev[i].events & EPOLLOUT) ? writable : 0) | ((ev[i].events & (EPOLLHUP | EPOLLERR | EPOLLRDHUP)) ? hangup : 0);
            out.push_back({r->fd, e, r->user});
        }
#elif defined(FORGE_OS_KQUEUE)
        struct kevent ev[64]; timespec ts{timeout_ms / 1000, (timeout_ms % 1000) * 1000000L};
        int n = ::kevent(q_, nullptr, 0, ev, 64, timeout_ms < 0 ? nullptr : &ts);
        for (int i = 0; i < n; ++i) {
            auto* r = static_cast<reg*>(ev[i].udata);
            unsigned e = (ev[i].filter == EVFILT_READ ? readable : 0) | (ev[i].filter == EVFILT_WRITE ? writable : 0) | ((ev[i].flags & (EV_EOF | EV_ERROR)) ? hangup : 0);
            bool merged = false;
            for (auto& o : out) if (o.fd == r->fd) { o.events |= e; merged = true; }
            if (!merged) out.push_back({r->fd, e, r->user});
        }
#else
        std::vector<
#if defined(FORGE_OS_WINDOWS)
            WSAPOLLFD
#else
            pollfd
#endif
            > p;
        for (auto& r : regs_) p.push_back({r->fd, short(((r->ev & readable) ? POLLIN : 0) | ((r->ev & writable) ? POLLOUT : 0)), 0});
        if (p.empty()) { sleep_ns(std::uint64_t(timeout_ms < 0 ? 1000 : timeout_ms) * 1000000ull); return out; }
#if defined(FORGE_OS_WINDOWS)
        int n = ::WSAPoll(p.data(), ULONG(p.size()), timeout_ms);
#else
        int n = ::poll(p.data(), nfds_t(p.size()), timeout_ms);
#endif
        for (std::size_t i = 0; n > 0 && i < p.size(); ++i)
            if (p[i].revents) {
                unsigned e = ((p[i].revents & POLLIN) ? readable : 0) | ((p[i].revents & POLLOUT) ? writable : 0) | ((p[i].revents & (POLLHUP | POLLERR)) ? hangup : 0);
                out.push_back({p[i].fd, e, regs_[i]->user});
            }
#endif
        return out;
    }
    std::size_t size() const { return regs_.size(); }
private:
    struct reg { sock_t fd; unsigned ev; void* user; };
    reg* find(sock_t fd) { for (auto& r : regs_) if (r->fd == fd) return r; return nullptr; }
    bool ctl(sock_t fd, unsigned ev, void* user, int op) {
        reg* r = find(fd);
        if (op == 0) { if (r) return false; regs_.push_back(new reg{fd, ev, user}); r = regs_.back(); }
        else if (!r) return false;
        bool ok = true;
#if defined(FORGE_OS_EPOLL)
        epoll_event e{}; e.events = ((ev & readable) ? EPOLLIN : 0) | ((ev & writable) ? EPOLLOUT : 0) | EPOLLRDHUP; e.data.ptr = r;
        ok = ::epoll_ctl(q_, op == 0 ? EPOLL_CTL_ADD : op == 1 ? EPOLL_CTL_MOD : EPOLL_CTL_DEL, fd, &e) == 0;
#elif defined(FORGE_OS_KQUEUE)
        struct kevent k[2]; int n = 0;
        unsigned want = op == 2 ? 0 : ev, had = op == 0 ? 0 : r->ev;
        if ((want ^ had) & readable) EV_SET(&k[n++], fd, EVFILT_READ, (want & readable) ? EV_ADD : EV_DELETE, 0, 0, r);
        if ((want ^ had) & writable) EV_SET(&k[n++], fd, EVFILT_WRITE, (want & writable) ? EV_ADD : EV_DELETE, 0, 0, r);
        if (op == 1 && !((want ^ had) & (readable | writable)) && want) { /* only user changed */ }
        ok = n == 0 || ::kevent(q_, k, n, nullptr, 0, nullptr) == 0;
#endif
        if (op == 2) { for (std::size_t i = 0; i < regs_.size(); ++i) if (regs_[i] == r) { delete r; regs_.erase(regs_.begin() + std::ptrdiff_t(i)); break; } }
        else { r->ev = ev; r->user = user; }
        if (!ok && op == 0) { delete regs_.back(); regs_.pop_back(); }
        return ok;
    }
#if defined(FORGE_OS_EPOLL) || defined(FORGE_OS_KQUEUE)
    int q_ = -1;
#endif
    std::vector<reg*> regs_;
public:
    poller(poller&&) = delete;
    void clear() { for (auto* r : regs_) delete r; regs_.clear(); }
};

} // namespace forge::os
