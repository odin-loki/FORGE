#pragma once
// forge::os anonymous pipes with RAII ends and deadline reads. Guide: docs/os-wrapper.md.
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
}
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
        while (off < s.size()) { auto n = ::write(fd_, s.data() + off, s.size() - off); if (n < 0) { if (errno == EINTR) continue; return {{}, last_error("write")}; } off += std::size_t(n); }
        return {};
    }
    // Read what is available within timeout_ms (0 = block until data or EOF). Empty = EOF or timeout.
    std::string read_some(std::size_t max = 65536, int timeout_ms = 0) {
        if (fd_ < 0) return {};
        if (timeout_ms > 0) { struct pollfd f{fd_, POLLIN, 0}; if (::poll(&f, 1, timeout_ms) <= 0) return {}; }
        std::string s(max, '\0'); auto n = ::read(fd_, s.data(), max);
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
#endif

} // namespace forge::os
