#pragma once
// forge::os anonymous pipes with RAII ends and deadline reads, and named FIFOs (os::fifo). Guide: docs/os-wrapper.md.
#include <forge/os/os.hpp>

namespace forge::os {

#if defined(FORGE_OS_POSIX)
namespace detail {
// pipe() with both ends close-on-exec from the start, so a fork+exec on another thread never inherits them (a
// leaked write end keeps the reader from ever seeing EOF). pipe2(O_CLOEXEC) is atomic on Linux, Android and
// FreeBSD. Apple has no pipe2: pipe + FD_CLOEXEC run under fork_mutex(), which every FORGE fork
// (serialized_fork) takes, so no FORGE-spawned child can inherit the ends in between; a fork the embedding
// program makes without serialized_fork can still race there. A child that needs an end as 0/1/2 gets it with
// dup2, which clears the flag on the copy (see detail_proc::dup_to for the from == to case).
inline int pipe_cloexec(int fd[2]) noexcept {
#if defined(__APPLE__)
    std::lock_guard<std::mutex> lk(fork_mutex());
    if (::pipe(fd)) return -1;
    ::fcntl(fd[0], F_SETFD, FD_CLOEXEC); ::fcntl(fd[1], F_SETFD, FD_CLOEXEC);
    return 0;
#else
    return ::pipe2(fd, O_CLOEXEC);
#endif
}
// write() to a pipe or FIFO whose reader may be gone: EPIPE, not a SIGPIPE that kills the process. SIGPIPE is
// blocked in this thread for the call and a SIGPIPE it raised is taken back; macOS has no sigtimedwait, so there
// the descriptor is set F_SETNOSIGPIPE instead.
inline ssize_t write_no_sigpipe(int fd, const void* p, std::size_t n) {
#if defined(__APPLE__)
    (void)::fcntl(fd, F_SETNOSIGPIPE, 1);
    return ::write(fd, p, n);
#else
    sigset_t pipe_only, old, pending;
    sigemptyset(&pipe_only); sigaddset(&pipe_only, SIGPIPE);
    sigemptyset(&pending); (void)sigpending(&pending);
    const bool already = sigismember(&pending, SIGPIPE) == 1;   // someone else's: leave it pending
    pthread_sigmask(SIG_BLOCK, &pipe_only, &old);
    const ssize_t r = ::write(fd, p, n); const int e = errno;
    if (r < 0 && e == EPIPE && !already) { const timespec zero{0, 0}; while (sigtimedwait(&pipe_only, nullptr, &zero) < 0 && errno == EINTR) {} }
    pthread_sigmask(SIG_SETMASK, &old, nullptr);
    errno = e; return r;
#endif
}
}
#endif
#if defined(FORGE_OS_POSIX)
class pipe_end {
public:
    pipe_end() = default;
    explicit pipe_end(int fd) : fd_(fd) {}
    pipe_end(const pipe_end&) = delete; pipe_end& operator=(const pipe_end&) = delete;
    pipe_end(pipe_end&& o) noexcept : fd_(std::exchange(o.fd_, -1)) {}
    pipe_end& operator=(pipe_end&& o) noexcept { if (this != &o) { close(); fd_ = std::exchange(o.fd_, -1); } return *this; }
    ~pipe_end() { close(); }
    void close() { if (fd_ >= 0) { ::close(fd_); fd_ = -1; } }
    int native() const { return fd_; }
    explicit operator bool() const { return fd_ >= 0; }
    status write_all(std::string_view s) {
        std::size_t off = 0;
        while (off < s.size()) { auto n = detail::write_no_sigpipe(fd_, s.data() + off, s.size() - off); if (n < 0) { if (errno == EINTR) continue; return {{}, last_error("write")}; } off += std::size_t(n); }
        return {};
    }
    // Read what is available within timeout_ms (0 = block until data or EOF). Empty = EOF or timeout.
    std::string read_some(std::size_t max = 65536, int timeout_ms = 0) {
        if (fd_ < 0) return {};
        if (timeout_ms > 0) {
            struct pollfd f{fd_, POLLIN, 0}; std::uint64_t end = monotonic_ns() + std::uint64_t(timeout_ms) * 1000000ull;
            for (;;) {   // a signal (EINTR) is neither data nor EOF: wait out the rest of the timeout
                auto now = monotonic_ns(); int r = ::poll(&f, 1, now >= end ? 0 : int((end - now + 999999) / 1000000));
                if (r > 0) break; if (r == 0 || errno != EINTR) return {};
            }
        }
        std::string s(max, '\0'); ssize_t n;
        do n = ::read(fd_, s.data(), max); while (n < 0 && errno == EINTR);
        if (n <= 0) return {};
        s.resize(std::size_t(n)); return s;
    }
    std::string read_all() { std::string o, c; while (!(c = read_some()).empty()) o += c; return o; }
private:
    int fd_ = -1;
};
struct pipe_pair { pipe_end read, write; };
inline result<pipe_pair> make_pipe() {
    int f[2]; if (detail::pipe_cloexec(f)) return {{}, last_error("pipe")};
    pipe_pair p{pipe_end(f[0]), pipe_end(f[1])};
    return {std::move(p), {}};
}
#elif defined(FORGE_OS_WINDOWS)
// Windows: the read end is an overlapped named-pipe server (so read_some can time out); the write end a plain
// synchronous client. Both are non-inheritable; hand an end to a child through process.hpp's std redirection.
class pipe_end {
public:
    pipe_end() = default;
    explicit pipe_end(HANDLE h) : h_(h) {}
    pipe_end(const pipe_end&) = delete; pipe_end& operator=(const pipe_end&) = delete;
    pipe_end(pipe_end&& o) noexcept : h_(std::exchange(o.h_, INVALID_HANDLE_VALUE)) {}
    pipe_end& operator=(pipe_end&& o) noexcept { if (this != &o) { close(); h_ = std::exchange(o.h_, INVALID_HANDLE_VALUE); } return *this; }
    ~pipe_end() { close(); }
    void close() { if (h_ != INVALID_HANDLE_VALUE) { CloseHandle(h_); h_ = INVALID_HANDLE_VALUE; } }
    HANDLE native() const { return h_; }
    explicit operator bool() const { return h_ != INVALID_HANDLE_VALUE; }
    status write_all(std::string_view s) {
        std::size_t off = 0;
        while (off < s.size()) {
            DWORD n = 0; if (!io_(true, const_cast<char*>(s.data() + off), s.size() - off, n, -1)) return {{}, last_error("write")};
            off += n;
        }
        return {};
    }
    // Read what is available within timeout_ms (0 = block until data or EOF). Empty = EOF or timeout.
    std::string read_some(std::size_t max = 65536, int timeout_ms = 0) {
        if (h_ == INVALID_HANDLE_VALUE || !max) return {};
        std::string s(max, '\0'); DWORD n = 0;
        if (!io_(false, s.data(), max, n, timeout_ms > 0 ? timeout_ms : -1)) return {};
        s.resize(n); return s;
    }
    std::string read_all() { std::string o, c; while (!(c = read_some()).empty()) o += c; return o; }
private:
    // Overlapped I/O with an optional timeout; works on synchronous handles too (they just complete inline). On
    // timeout the read is cancelled and waited for, since the kernel completes into this stack frame; data that
    // arrived meanwhile is still returned.
    bool io_(bool wr, void* p, std::size_t len, DWORD& got, int timeout_ms) {
        got = 0;
        OVERLAPPED ov{}; ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr); if (!ov.hEvent) return false;
        DWORD n = DWORD(len < (std::size_t(1) << 30) ? len : (std::size_t(1) << 30));
        BOOL ok = wr ? WriteFile(h_, p, n, nullptr, &ov) : ReadFile(h_, p, n, nullptr, &ov);
        DWORD e = ok ? 0 : GetLastError();
        if (!ok && e == ERROR_IO_PENDING) {
            if (timeout_ms >= 0 && WaitForSingleObject(ov.hEvent, DWORD(timeout_ms)) != WAIT_OBJECT_0) CancelIoEx(h_, &ov);
            ok = GetOverlappedResult(h_, &ov, &got, TRUE); e = ok ? 0 : GetLastError();
        } else if (ok) GetOverlappedResult(h_, &ov, &got, TRUE);
        CloseHandle(ov.hEvent); SetLastError(e);
        return ok != 0;   // ERROR_BROKEN_PIPE (EOF), ERROR_OPERATION_ABORTED (timeout) and real errors all read as false
    }
    HANDLE h_ = INVALID_HANDLE_VALUE;
};
struct pipe_pair { pipe_end read, write; };
inline result<pipe_pair> make_pipe() {
    static std::atomic<unsigned> serial{0};
    std::uint64_t r = 0; entropy(&r, sizeof r);
    char name[96]; std::snprintf(name, sizeof name, "\\\\.\\pipe\\forge-anon-%lu-%u-%016llx", static_cast<unsigned long>(GetCurrentProcessId()),
                                 serial.fetch_add(1), static_cast<unsigned long long>(r));
    auto w = widen(name);
    HANDLE rd = CreateNamedPipeW(w.c_str(), PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
                                 PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1, 65536, 65536, 0, nullptr);
    if (rd == INVALID_HANDLE_VALUE) return {{}, last_error("CreateNamedPipe")};
    HANDLE wr = CreateFileW(w.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, nullptr);
    if (wr == INVALID_HANDLE_VALUE) { auto e = last_error("open pipe"); CloseHandle(rd); return {{}, e}; }
    pipe_pair p{pipe_end(rd), pipe_end(wr)};
    return {std::move(p), {}};
}
#endif

// ---------------------------------------------------------------- named FIFOs
// A named one-way byte pipe between unrelated processes.
//   POSIX: mkfifo(2). `name` is a path, or a bare name placed in temp_dir() as <name>.fifo. Reader and writer may
//   open in either order; each open waits for the other side (up to timeout_ms; < 0 = without limit).
//   Windows: a byte-mode named pipe \\.\pipe\<name> (one instance, local clients only). The reader owns it:
//   open_read creates it and waits for the writer; open_write waits for the pipe to appear. There is nothing on
//   disk, so create() and remove() only validate the name.
// The ends are os::pipe_end: read_some / read_all / write_all, and usable with os::reactor.
namespace fifo {
inline std::string path(const std::string& name) {
#if defined(FORGE_OS_WINDOWS)
    if (name.rfind("\\\\.\\pipe\\", 0) == 0) return name;
    return "\\\\.\\pipe\\" + name;
#else
    if (name.find('/') != std::string::npos) return name;
    return join(temp_dir(), name + ".fifo");
#endif
}
// Creates the FIFO (mode: POSIX permission bits). An existing FIFO of that name is fine; any other file is an error.
inline status create(const std::string& name, unsigned mode = 0600) {
    if (name.empty()) return {{}, errno_error(EINVAL, "fifo: empty name")};
#if defined(FORGE_OS_WINDOWS)
    (void)mode; return {};
#else
    const std::string p = path(name);
    if (::mkfifo(p.c_str(), mode_t(mode)) == 0) return {};
    if (errno == EEXIST) { struct stat st{}; if (::stat(p.c_str(), &st) == 0 && S_ISFIFO(st.st_mode)) return {}; return {{}, errno_error(EEXIST, "fifo " + p + ": exists and is not a FIFO")}; }
    return {{}, last_error("mkfifo " + p)};
#endif
}
inline status remove(const std::string& name) {
#if defined(FORGE_OS_WINDOWS)
    (void)name; return {};
#else
    const std::string p = path(name);
    if (::unlink(p.c_str()) != 0 && errno != ENOENT) return {{}, last_error("unlink " + p)};
    return {};
#endif
}
inline result<pipe_end> open_read(const std::string& name, int timeout_ms = -1) {
    const std::string p = path(name);
#if defined(FORGE_OS_WINDOWS)
    HANDLE h = CreateNamedPipeW(widen(p).c_str(), PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
                                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1, 65536, 65536, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return {{}, last_error("fifo " + p)};
    OVERLAPPED ov{}; ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!ov.hEvent) { auto e = last_error("CreateEvent"); CloseHandle(h); return {{}, e}; }
    BOOL ok = ConnectNamedPipe(h, &ov); DWORD e = ok ? 0 : GetLastError();
    if (!ok && e == ERROR_IO_PENDING) {
        if (WaitForSingleObject(ov.hEvent, timeout_ms < 0 ? INFINITE : DWORD(timeout_ms)) != WAIT_OBJECT_0) CancelIoEx(h, &ov);
        DWORD n = 0; ok = GetOverlappedResult(h, &ov, &n, TRUE); e = ok ? 0 : GetLastError();
    } else if (!ok && e == ERROR_PIPE_CONNECTED) { ok = TRUE; e = 0; }
    CloseHandle(ov.hEvent);
    if (!ok) { CloseHandle(h); if (e == ERROR_OPERATION_ABORTED) return {{}, errno_error(ETIMEDOUT, "fifo " + p + ": no writer")}; SetLastError(e); return {{}, last_error("fifo " + p)}; }
    return {pipe_end(h), {}};
#else
    if (timeout_ms < 0) {
        int fd; do fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC); while (fd < 0 && errno == EINTR);
        if (fd < 0) return {{}, last_error("open " + p)};
        return {pipe_end(fd), {}};
    }
    // Non-blocking open succeeds at once; then wait for a writer to show up (data, or its hang-up).
    int fd = ::open(p.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC); if (fd < 0) return {{}, last_error("open " + p)};
    const std::uint64_t end = monotonic_ns() + std::uint64_t(timeout_ms) * 1000000ull;
    for (;;) {
        struct pollfd f{fd, POLLIN, 0}; auto now = monotonic_ns();
        int r = ::poll(&f, 1, now >= end ? 0 : int((end - now + 999999ull) / 1000000ull));
        if (r > 0 && (f.revents & POLLIN)) break;
#if defined(__linux__)
        if (r > 0) break;   // hang-up: a writer came and left without data (Linux reports this only after a writer)
#else
        // BSD / macOS may report a hang-up while no writer has ever opened the FIFO: keep waiting for data.
        if (r > 0 && monotonic_ns() < end) { sleep_ns(5'000'000); continue; }
#endif
        if (r < 0 && errno == EINTR) continue;
        ::close(fd); return {{}, errno_error(ETIMEDOUT, "fifo " + p + ": no writer")};
    }
    ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL) & ~O_NONBLOCK);
    return {pipe_end(fd), {}};
#endif
}
inline result<pipe_end> open_write(const std::string& name, int timeout_ms = -1) {
    const std::string p = path(name);
    const std::uint64_t end = timeout_ms < 0 ? 0 : monotonic_ns() + std::uint64_t(timeout_ms) * 1000000ull;
#if defined(FORGE_OS_WINDOWS)
    const std::wstring w = widen(p);
    for (;;) {
        HANDLE h = CreateFileW(w.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, nullptr);
        if (h != INVALID_HANDLE_VALUE) return {pipe_end(h), {}};
        DWORD e = GetLastError();
        if (e != ERROR_FILE_NOT_FOUND && e != ERROR_PIPE_BUSY) return {{}, last_error("fifo " + p)};
        if (timeout_ms >= 0 && monotonic_ns() >= end) return {{}, errno_error(ETIMEDOUT, "fifo " + p + ": no reader")};
        if (e == ERROR_PIPE_BUSY) WaitNamedPipeW(w.c_str(), 50); else Sleep(5);
    }
#else
    if (timeout_ms < 0) {
        int fd; do fd = ::open(p.c_str(), O_WRONLY | O_CLOEXEC); while (fd < 0 && errno == EINTR);
        if (fd < 0) return {{}, last_error("open " + p)};
        return {pipe_end(fd), {}};
    }
    for (;;) {   // a non-blocking open for writing fails with ENXIO until a reader has the FIFO open
        int fd = ::open(p.c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC);
        if (fd >= 0) { ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL) & ~O_NONBLOCK); return {pipe_end(fd), {}}; }
        if (errno != ENXIO && errno != EINTR) return {{}, last_error("open " + p)};
        if (monotonic_ns() >= end) return {{}, errno_error(ETIMEDOUT, "fifo " + p + ": no reader")};
        sleep_ns(5'000'000);
    }
#endif
}
}

} // namespace forge::os
