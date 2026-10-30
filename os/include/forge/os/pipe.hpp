#pragma once
// forge::os anonymous pipes with RAII ends and deadline reads. Guide: docs/os-wrapper.md.
#include <forge/os/os.hpp>

namespace forge::os {

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
    int f[2]; if (::pipe(f)) return {{}, last_error("pipe")};
    ::fcntl(f[0], F_SETFD, FD_CLOEXEC); ::fcntl(f[1], F_SETFD, FD_CLOEXEC);
    pipe_pair p{pipe_end(f[0]), pipe_end(f[1])};
    return {std::move(p), {}};
}
#endif

} // namespace forge::os
