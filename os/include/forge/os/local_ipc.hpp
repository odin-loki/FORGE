#pragma once
// Local IPC streams: Unix-domain sockets (Linux/Android/macOS/iOS/FreeBSD; Linux & Android use the abstract
// namespace when the name starts with '@') and Windows named pipes (\\.\pipe\<name>).
// Names: a filesystem path (or "@abstract") on POSIX; a bare pipe name on Windows.  local_endpoint(name) builds
// a sensible default for both.
#include <forge/os/detail.hpp>
#if defined(FORGE_OS_POSIX)
#include <sys/un.h>
#endif

namespace forge::os {
inline std::string local_endpoint(const std::string& name) {
#if defined(FORGE_OS_WINDOWS)
    return name;
#else
    return join(temp_dir(), name + ".sock");
#endif
}
class local_stream {
public:
#if defined(FORGE_OS_WINDOWS)
    using native_t = HANDLE; static native_t bad() { return INVALID_HANDLE_VALUE; }
#else
    using native_t = int; static native_t bad() { return -1; }
#endif
    local_stream() = default;
    explicit local_stream(native_t h) : h_(h) {}
    local_stream(local_stream&& o) noexcept : h_(std::exchange(o.h_, bad())) {}
    local_stream& operator=(local_stream&& o) noexcept { if (this != &o) { close(); h_ = std::exchange(o.h_, bad()); } return *this; }
    ~local_stream() { close(); }
    bool valid() const { return h_ != bad(); }
    native_t native() const { return h_; }
    static result<local_stream> connect(const std::string& name, int timeout_ms = 2000) {
#if defined(FORGE_OS_WINDOWS)
        auto w = widen("\\\\.\\pipe\\" + name); std::uint64_t end = monotonic_ns() + std::uint64_t(timeout_ms) * 1000000ull;
        for (;;) {
            HANDLE h = CreateFileW(w.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
            if (h != INVALID_HANDLE_VALUE) return {local_stream(h), {}};
            if (GetLastError() != ERROR_PIPE_BUSY && GetLastError() != ERROR_FILE_NOT_FOUND) return {{}, last_error("connect pipe " + name)};
            if (monotonic_ns() > end) return {{}, last_error("connect pipe " + name)};
            if (GetLastError() == ERROR_PIPE_BUSY) WaitNamedPipeW(w.c_str(), 100); else Sleep(10);
        }
#else
        sockaddr_un a{}; socklen_t len = 0; if (auto e = addr_(name, a, len)) return {{}, e};
        int fd = ::socket(AF_UNIX, SOCK_STREAM, 0); if (fd < 0) return {{}, last_error("socket(AF_UNIX)")};
#if defined(__APPLE__) || defined(FORGE_OS_FREEBSD)
        int one = 1; ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
        ::fcntl(fd, F_SETFD, FD_CLOEXEC);
        std::uint64_t end = monotonic_ns() + std::uint64_t(timeout_ms) * 1000000ull;
        while (::connect(fd, reinterpret_cast<sockaddr*>(&a), len)) {
            if ((errno != ENOENT && errno != ECONNREFUSED && errno != EAGAIN) || monotonic_ns() > end) { auto e = last_error("connect " + name); ::close(fd); return {{}, e}; }
            sleep_ns(5000000);
        }
        return {local_stream(fd), {}};
#endif
    }
    result<std::size_t> read(void* p, std::size_t n) {
#if defined(FORGE_OS_WINDOWS)
        DWORD got = 0; if (!io_(false, p, n, got)) { if (GetLastError() == ERROR_BROKEN_PIPE) return {0, {}}; return {0, last_error("pipe read")}; }
        return {got, {}};
#else
        for (;;) { auto r = ::recv(h_, p, n, 0); if (r >= 0) return {std::size_t(r), {}}; if (errno != EINTR) return {0, last_error("recv")}; }
#endif
    }
    status write_all(const void* p, std::size_t n) {
        auto* b = static_cast<const char*>(p);
        while (n) {
#if defined(FORGE_OS_WINDOWS)
            DWORD w = 0; if (!io_(true, const_cast<char*>(b), n, w)) return {{}, last_error("pipe write")};
#else
#if defined(MSG_NOSIGNAL)
            auto w = ::send(h_, b, n, MSG_NOSIGNAL);
#else
            auto w = ::send(h_, b, n, 0);
#endif
            if (w < 0) { if (errno == EINTR) continue; return {{}, last_error("send")}; }
#endif
            b += w; n -= std::size_t(w);
        }
        return {};
    }
    status read_exact(void* p, std::size_t n) {
        auto* b = static_cast<char*>(p);
        while (n) { auto r = read(b, n); if (!r) return {{}, r.err}; if (!r.value) return {{}, {-1, "local_stream: eof"}}; b += r.value; n -= r.value; }
        return {};
    }
    void close() {
        if (!valid()) return;
#if defined(FORGE_OS_WINDOWS)
        FlushFileBuffers(h_); CloseHandle(h_);
#else
        ::close(h_);
#endif
        h_ = bad();
    }
#if defined(FORGE_OS_POSIX)
    static error addr_(const std::string& name, sockaddr_un& a, socklen_t& len) {
        a.sun_family = AF_UNIX; if (name.size() >= sizeof a.sun_path) return {36, "unix socket path too long: " + name};
        std::memcpy(a.sun_path, name.data(), name.size());
        len = socklen_t(offsetof(sockaddr_un, sun_path) + name.size() + (name[0] == '@' ? 0 : 1));
#if defined(__linux__)
        if (name[0] == '@') a.sun_path[0] = '\0';
#else
        if (name[0] == '@') return unsupported("abstract unix sockets");
#endif
        return {};
    }
#endif
private:
#if defined(FORGE_OS_WINDOWS)
    // Overlapped-safe blocking I/O: server ends are created FILE_FLAG_OVERLAPPED, clients are not; this works for both.
    bool io_(bool wr, void* p, std::size_t n, DWORD& got) {
        OVERLAPPED ov{}; ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        BOOL ok = wr ? WriteFile(h_, p, DWORD(n), nullptr, &ov) : ReadFile(h_, p, DWORD(n), nullptr, &ov);
        if (!ok && GetLastError() != ERROR_IO_PENDING) { DWORD e = GetLastError(); CloseHandle(ov.hEvent); SetLastError(e); return false; }
        ok = GetOverlappedResult(h_, &ov, &got, TRUE); DWORD e = GetLastError(); CloseHandle(ov.hEvent); SetLastError(e); return ok != 0;
    }
#endif
    native_t h_ = bad();
};

class local_listener {
public:
    local_listener() = default;
    local_listener(local_listener&& o) noexcept { *this = std::move(o); }
    local_listener& operator=(local_listener&& o) noexcept {
        if (this != &o) { close(); name_ = std::move(o.name_);
#if defined(FORGE_OS_WINDOWS)
            next_ = std::exchange(o.next_, INVALID_HANDLE_VALUE);
#else
            fd_ = std::exchange(o.fd_, -1);
#endif
        }
        return *this;
    }
    ~local_listener() { close(); }
    static result<local_listener> listen(const std::string& name) {
        local_listener l; l.name_ = name;
#if defined(FORGE_OS_WINDOWS)
        l.next_ = l.make_pipe_(true);
        if (l.next_ == INVALID_HANDLE_VALUE) return {std::move(l), last_error("CreateNamedPipe " + name)};
        return {std::move(l), {}};
#else
        sockaddr_un a{}; socklen_t len = 0; if (auto e = local_stream::addr_(name, a, len)) return {std::move(l), e};
        if (name[0] != '@') ::unlink(name.c_str());
        l.fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0); if (l.fd_ < 0) return {std::move(l), last_error("socket(AF_UNIX)")};
        ::fcntl(l.fd_, F_SETFD, FD_CLOEXEC);
        if (::bind(l.fd_, reinterpret_cast<sockaddr*>(&a), len) || ::listen(l.fd_, 64)) return {std::move(l), last_error("bind/listen " + name)};
        return {std::move(l), {}};
#endif
    }
    // Block up to timeout_ms (-1 forever) for one client.
    result<local_stream> accept(int timeout_ms = -1) {
#if defined(FORGE_OS_WINDOWS)
        OVERLAPPED ov{}; ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        BOOL ok = ConnectNamedPipe(next_, &ov); DWORD e = GetLastError();
        if (!ok && e == ERROR_IO_PENDING) {
            if (WaitForSingleObject(ov.hEvent, timeout_ms < 0 ? INFINITE : DWORD(timeout_ms)) != WAIT_OBJECT_0) { CancelIo(next_); CloseHandle(ov.hEvent); return {{}, {WAIT_TIMEOUT, "accept: timeout"}}; }
            DWORD x; ok = GetOverlappedResult(next_, &ov, &x, FALSE); e = ok ? 0 : GetLastError();
        }
        CloseHandle(ov.hEvent);
        if (!ok && e != ERROR_PIPE_CONNECTED) return {{}, {int(e), "ConnectNamedPipe: " + error_text(int(e))}};
        // The connected instance was opened overlapped; hand the client a blocking duplicate-free handle by reopening
        // a fresh listening instance and returning this one (ReadFile/WriteFile with nullptr OVERLAPPED on an
        // overlapped handle is fine for byte-mode pipes when no other I/O is outstanding).
        HANDLE c = std::exchange(next_, make_pipe_(false));
        return {local_stream(c), {}};
#else
        if (timeout_ms >= 0) { pollfd p{fd_, POLLIN, 0}; int r = ::poll(&p, 1, timeout_ms); if (r == 0) return {{}, {ETIMEDOUT, "accept: timeout"}}; if (r < 0) return {{}, last_error("poll")}; }
        int c = ::accept(fd_, nullptr, nullptr); if (c < 0) return {{}, last_error("accept")};
        ::fcntl(c, F_SETFD, FD_CLOEXEC);
#if defined(__APPLE__) || defined(FORGE_OS_FREEBSD)
        int one = 1; ::setsockopt(c, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
        return {local_stream(c), {}};
#endif
    }
    const std::string& name() const { return name_; }
    void close() {
#if defined(FORGE_OS_WINDOWS)
        if (next_ != INVALID_HANDLE_VALUE) { CloseHandle(next_); next_ = INVALID_HANDLE_VALUE; }
#else
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; if (!name_.empty() && name_[0] != '@') ::unlink(name_.c_str()); }
#endif
    }
private:
#if defined(FORGE_OS_WINDOWS)
    HANDLE make_pipe_(bool first) {
        return CreateNamedPipeW(widen("\\\\.\\pipe\\" + name_).c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | (first ? FILE_FLAG_FIRST_PIPE_INSTANCE : 0),
                                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, PIPE_UNLIMITED_INSTANCES, 65536, 65536, 0, nullptr);
    }
    HANDLE next_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
    std::string name_;
};
} // namespace forge::os
