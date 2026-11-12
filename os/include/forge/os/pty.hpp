#pragma once
// forge::os pseudo-terminals: run a program that believes it talks to a terminal (line editing, colours,
// progress bars, password prompts, full-screen UIs) and drive it through a byte stream.
//   POSIX (Linux, Android, macOS, FreeBSD): posix_openpt / grantpt / unlockpt; the child gets a new session with
//       the slave side as its controlling terminal on stdin, stdout and stderr.
//   Windows 10 1809+: ConPTY (CreatePseudoConsole, looked up at run time); the output stream is UTF-8 text with
//       VT sequences, like a POSIX terminal's. Older Windows: unsupported.
//   iOS: no child processes: unsupported.
//
//   auto t = os::pty::spawn({"sh", "-c", "tty && stty size"}, {.cols = 100, .rows = 30});
//   while (!t->eof()) out += t->read(4096, 200);
//   int code = t->wait();
//
// The terminal echoes what you write() unless the program turns echo off, and translates "\n" to "\r\n" on
// output, as terminals do. Nothing throws.
#include <forge/os/os.hpp>
#include <forge/os/process.hpp>
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
#include <sys/ioctl.h>
#include <termios.h>
#endif

namespace forge::os {
struct pty_opts {
    int cols = 80, rows = 24;
    std::string cwd;                              // empty = inherit
    std::map<std::string, std::string> env_set;   // added / overridden variables (TERM is yours to set)
};
class pty {
public:
    pty() = default;
    pty(const pty&) = delete; pty& operator=(const pty&) = delete;
    pty(pty&& o) noexcept { swap_(o); }
    pty& operator=(pty&& o) noexcept { if (this != &o) { reset_(); swap_(o); } return *this; }
    ~pty() { reset_(); }
    // Whether this system has pseudo-terminals at all (Windows: ConPTY present).
    static bool available() {
#if defined(FORGE_OS_IOS)
        return false;
#elif defined(FORGE_OS_WINDOWS)
        return api_().create != nullptr;
#else
        return true;
#endif
    }
    // Start argv (argv[0] resolved through PATH) on a new pseudo-terminal of the given size.
    static result<pty> spawn(const std::vector<std::string>& argv, const pty_opts& o = {}) {
        pty t;
        if (argv.empty()) return {std::move(t), errno_error(EINVAL, "pty: empty argv")};
#if defined(FORGE_OS_IOS)
        (void)o; return {std::move(t), unsupported("pty (child processes on iOS)")};
#elif defined(FORGE_OS_WINDOWS)
        auto& a = api_();
        if (!a.create) return {std::move(t), unsupported("pty (ConPTY needs Windows 10 1809)")};
        spawn_spec sp; sp.argv = argv; sp.env_set = o.env_set; sp.cwd = o.cwd;
        std::string exe = find_executable(argv[0]);
        if (exe.empty()) return {std::move(t), {ERROR_FILE_NOT_FOUND, "pty " + argv[0] + ": not found"}};
        error cmd_err{};
        std::wstring cmd = detail_proc::win_cmdline(sp.argv, exe, &cmd_err), envb = detail_proc::win_env_block(sp);
        if (cmd.empty()) return {std::move(t), cmd_err};
        HANDLE in_r = nullptr, in_w = nullptr, out_r = nullptr, out_w = nullptr;
        if (!CreatePipe(&in_r, &in_w, nullptr, 0)) return {std::move(t), last_error("CreatePipe")};
        if (!CreatePipe(&out_r, &out_w, nullptr, 0)) { auto e = last_error("CreatePipe"); CloseHandle(in_r); CloseHandle(in_w); return {std::move(t), e}; }
        COORD size{}; size.X = SHORT(o.cols > 0 ? o.cols : 80); size.Y = SHORT(o.rows > 0 ? o.rows : 24);
        void* pc = nullptr;
        HRESULT hr = a.create(size, in_r, out_w, 0, &pc);
        CloseHandle(in_r); CloseHandle(out_w);   // the pseudoconsole holds its own references
        if (FAILED(hr)) { CloseHandle(in_w); CloseHandle(out_r); return {std::move(t), {int(hr), "CreatePseudoConsole failed"}}; }
        STARTUPINFOEXW sx{}; sx.StartupInfo.cb = sizeof sx;
        SIZE_T sz = 0; InitializeProcThreadAttributeList(nullptr, 1, 0, &sz);
        std::vector<char> buf(sz ? sz : 1);
        auto* al = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(buf.data());
        bool ok = InitializeProcThreadAttributeList(al, 1, 0, &sz) != 0;
        const bool init = ok;
        ok = ok && UpdateProcThreadAttribute(al, 0, 0x00020016 /*PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE*/, pc, sizeof(void*), nullptr, nullptr);
        sx.lpAttributeList = al;
        PROCESS_INFORMATION pi{};
        std::wstring wcwd = widen(o.cwd);
        if (ok) ok = CreateProcessW(widen(exe).c_str(), cmd.data(), nullptr, nullptr, FALSE, EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
                                    envb.data(), o.cwd.empty() ? nullptr : wcwd.c_str(), &sx.StartupInfo, &pi) != 0;
        error e{}; if (!ok) e = last_error("CreateProcess (pty) " + argv[0]);
        if (init) DeleteProcThreadAttributeList(al);
        if (!ok) { CloseHandle(out_r); a.close(pc); CloseHandle(in_w); return {std::move(t), e}; }
        CloseHandle(pi.hThread);
        t.pc_ = pc; t.in_ = in_w; t.out_ = out_r; t.proc_ = pi.hProcess; t.pid_ = pi.dwProcessId;
        return {std::move(t), {}};
#else
        std::string exe = find_executable(argv[0]);
        if (exe.empty()) return {std::move(t), {ENOENT, "pty " + argv[0] + ": command not found"}};
        int m = ::posix_openpt(O_RDWR | O_NOCTTY);
        if (m < 0) return {std::move(t), last_error("posix_openpt")};
        ::fcntl(m, F_SETFD, FD_CLOEXEC);
        auto fail = [&](error e) { ::close(m); return result<pty>{std::move(t), std::move(e)}; };
        if (::grantpt(m) != 0 || ::unlockpt(m) != 0) return fail(last_error("grantpt/unlockpt"));
        char slave[256] = {};
#if defined(__linux__)
        if (::ptsname_r(m, slave, sizeof slave) != 0) return fail(last_error("ptsname"));
#elif defined(__APPLE__) && defined(TIOCPTYGNAME)
        if (::ioctl(m, TIOCPTYGNAME, slave) != 0) return fail(last_error("ptsname"));
#else
        {   // ptsname() returns static storage: serialised with the lock every FORGE fork takes
            std::lock_guard<std::mutex> lk(fork_mutex());
            const char* n = ::ptsname(m);
            if (!n || std::strlen(n) >= sizeof slave) return fail(last_error("ptsname"));
            std::memcpy(slave, n, std::strlen(n) + 1);
        }
#endif
        struct winsize ws{}; ws.ws_col = static_cast<unsigned short>(o.cols > 0 ? o.cols : 80); ws.ws_row = static_cast<unsigned short>(o.rows > 0 ? o.rows : 24);
        ::ioctl(m, TIOCSWINSZ, &ws);
        std::vector<std::string> envs;
        for (char** e = environ; e && *e; ++e) { std::string s(*e); if (!o.env_set.count(s.substr(0, s.find('=')))) envs.push_back(std::move(s)); }
        for (auto& [k, v] : o.env_set) envs.push_back(k + "=" + v);
        std::vector<char*> av, ev;
        for (auto& a : argv) av.push_back(const_cast<char*>(a.c_str())); av.push_back(nullptr);
        for (auto& e : envs) ev.push_back(const_cast<char*>(e.c_str())); ev.push_back(nullptr);
        int px[2]; if (detail::pipe_cloexec(px)) return fail(last_error("pipe"));
        const char* wd = o.cwd.empty() ? nullptr : o.cwd.c_str();
        pid_t p = forge::os::serialized_fork();
        if (p < 0) { auto e = last_error("fork"); ::close(px[0]); ::close(px[1]); return fail(e); }
        if (p == 0) {   // async-signal-safe calls only
            ::setsid();
            int s = ::open(slave, O_RDWR);
            if (s < 0) { int e = errno; (void)!::write(px[1], &e, sizeof e); ::_exit(127); }
#if defined(TIOCSCTTY)
            (void)::ioctl(s, TIOCSCTTY, 0);   // BSD / macOS: opening is not enough to make it the controlling terminal
#endif
            ::dup2(s, 0); ::dup2(s, 1); ::dup2(s, 2); if (s > 2) ::close(s);
            ::signal(SIGPIPE, SIG_DFL);
            if (wd && ::chdir(wd) != 0) { int e = errno; (void)!::write(px[1], &e, sizeof e); ::_exit(127); }
            ::execve(exe.c_str(), av.data(), ev.data());
            int e = errno; (void)!::write(px[1], &e, sizeof e); ::_exit(127);
        }
        ::close(px[1]);
        int child_errno = 0;
        { ssize_t n; do n = ::read(px[0], &child_errno, sizeof child_errno); while (n < 0 && errno == EINTR); if (n != ssize_t(sizeof child_errno)) child_errno = 0; ::close(px[0]); }
        if (child_errno) { int st; while (::waitpid(p, &st, 0) < 0 && errno == EINTR) {} return fail({child_errno, "pty " + argv[0] + ": " + error_text(child_errno)}); }
        t.m_ = m; t.pid_ = p;
        return {std::move(t), {}};
#endif
    }
    std::uint64_t pid() const { return std::uint64_t(pid_); }
    bool valid() const { return pid_ > 0; }
    // The terminal's output (what the program printed, plus the echo of what was written). Waits up to timeout_ms
    // (< 0: until there is output or the program is gone). Empty on timeout and at the end; eof() tells which.
    std::string read(std::size_t max = 65536, int timeout_ms = -1) {
        if (!max || eof_) return {};
#if defined(FORGE_OS_IOS)
        (void)timeout_ms; return {};
#elif defined(FORGE_OS_WINDOWS)
        if (!out_) return {};
        const std::uint64_t end = timeout_ms < 0 ? 0 : monotonic_ns() + std::uint64_t(timeout_ms) * 1000000ull;
        std::uint64_t nap = 1000000ull; int after_exit = 0;
        for (;;) {
            DWORD avail = 0;
            if (!PeekNamedPipe(out_, nullptr, 0, nullptr, &avail, nullptr)) { eof_ = true; return {}; }
            if (avail) break;
            // The pipe stays open as long as the pseudoconsole exists: the end of output is "the program has
            // exited and the console host has had time (50 ms) to flush what was left".
            if (proc_ && WaitForSingleObject(proc_, 0) == WAIT_OBJECT_0 && ++after_exit > 10) { eof_ = true; return {}; }
            if (timeout_ms >= 0 && monotonic_ns() >= end) return {};
            sleep_ns(nap); nap = std::min<std::uint64_t>(nap * 2, 5000000ull);
        }
        std::string s(max, '\0'); DWORD n = 0;
        if (!ReadFile(out_, s.data(), DWORD(std::min<std::size_t>(max, 1u << 20)), &n, nullptr) || !n) { eof_ = true; return {}; }
        s.resize(n); return s;
#else
        if (m_ < 0) return {};
        const std::uint64_t end = timeout_ms < 0 ? 0 : monotonic_ns() + std::uint64_t(timeout_ms) * 1000000ull;
        for (;;) {
            int ms = -1;
            if (timeout_ms >= 0) { auto now = monotonic_ns(); ms = now >= end ? 0 : int((end - now + 999999ull) / 1000000ull); }
            struct pollfd f{m_, POLLIN, 0}; int r = ::poll(&f, 1, ms);
            if (r == 0) return {};
            if (r < 0) { if (errno == EINTR) continue; eof_ = true; return {}; }
            std::string s(max, '\0');
            ssize_t n = ::read(m_, s.data(), max);
            if (n > 0) { s.resize(std::size_t(n)); return s; }
            if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            eof_ = true; return {};   // 0, or EIO on Linux once every slave descriptor is closed
        }
#endif
    }
    // True once read() has seen the end: the program exited (or closed the terminal) and all output was read.
    bool eof() const { return eof_; }
    // Send input, as if typed.
    status write(std::string_view s) {
#if defined(FORGE_OS_IOS)
        (void)s; return {{}, unsupported("pty")};
#elif defined(FORGE_OS_WINDOWS)
        if (!in_) return {{}, {int(ERROR_INVALID_HANDLE), "pty: closed"}};
        for (std::size_t off = 0; off < s.size();) { DWORD n = 0; if (!WriteFile(in_, s.data() + off, DWORD(std::min<std::size_t>(s.size() - off, 1u << 20)), &n, nullptr)) return {{}, last_error("pty write")}; off += n; }
        return {};
#else
        if (m_ < 0) return {{}, errno_error(EBADF, "pty: closed")};
        for (std::size_t off = 0; off < s.size();) { auto n = ::write(m_, s.data() + off, s.size() - off); if (n < 0) { if (errno == EINTR) continue; return {{}, last_error("pty write")}; } off += std::size_t(n); }
        return {};
#endif
    }
    // Change the terminal size; the program is told (SIGWINCH / a console resize event).
    status resize(int cols, int rows) {
        if (cols <= 0 || rows <= 0) return {{}, errno_error(EINVAL, "pty: bad size")};
#if defined(FORGE_OS_IOS)
        return {{}, unsupported("pty")};
#elif defined(FORGE_OS_WINDOWS)
        if (!pc_) return {{}, {int(ERROR_INVALID_HANDLE), "pty: closed"}};
        COORD c{}; c.X = SHORT(cols); c.Y = SHORT(rows);
        HRESULT hr = api_().resize(pc_, c);
        if (FAILED(hr)) return {{}, {int(hr), "ResizePseudoConsole failed"}};
        return {};
#else
        if (m_ < 0) return {{}, errno_error(EBADF, "pty: closed")};
        struct winsize ws{}; ws.ws_col = static_cast<unsigned short>(cols); ws.ws_row = static_cast<unsigned short>(rows);
        if (::ioctl(m_, TIOCSWINSZ, &ws) != 0) return {{}, last_error("TIOCSWINSZ")};
        return {};
#endif
    }
    // Wait for the program to exit: its exit code (128 + signal on POSIX), or -1 on timeout (0 = no limit). Output
    // not yet read stays readable.
    int wait(std::uint64_t timeout_ms = 0) {
        if (done_) return code_;
#if defined(FORGE_OS_IOS)
        (void)timeout_ms; return -1;
#elif defined(FORGE_OS_WINDOWS)
        if (!proc_) return -1;
        if (WaitForSingleObject(proc_, timeout_ms ? DWORD(std::min<std::uint64_t>(timeout_ms, 0x7fffffffu)) : INFINITE) != WAIT_OBJECT_0) return -1;
        DWORD code = 0; GetExitCodeProcess(proc_, &code); done_ = true; code_ = int(code); return code_;
#else
        if (pid_ <= 0) return -1;
        const std::uint64_t end = timeout_ms ? monotonic_ns() + timeout_ms * 1000000ull : 0;
        int st = 0;
        for (;;) {
            pid_t w = ::waitpid(pid_, &st, end ? WNOHANG : 0);
            if (w == pid_) break;
            if (w < 0 && errno != EINTR) return -1;
            if (w == 0) { auto now = monotonic_ns(); if (now >= end) return -1; detail_proc::await_exit(pid_, end - now); }
        }
        done_ = true; code_ = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
        return code_;
#endif
    }
    // End the program: POSIX SIGKILL to its session's process group; Windows TerminateProcess (closing the
    // pseudoconsole then ends whatever else is attached to it).
    bool kill() {
        if (done_ || pid_ <= 0) return false;
#if defined(FORGE_OS_IOS)
        return false;
#elif defined(FORGE_OS_WINDOWS)
        return proc_ && TerminateProcess(proc_, 137) != 0;
#else
        return ::kill(-pid_, SIGKILL) == 0 || ::kill(pid_, SIGKILL) == 0;
#endif
    }
private:
    void swap_(pty& o) noexcept {
        std::swap(pid_, o.pid_); std::swap(done_, o.done_); std::swap(code_, o.code_); std::swap(eof_, o.eof_);
#if defined(FORGE_OS_WINDOWS)
        std::swap(pc_, o.pc_); std::swap(in_, o.in_); std::swap(out_, o.out_); std::swap(proc_, o.proc_);
#else
        std::swap(m_, o.m_);
#endif
    }
    void reset_() {
#if defined(FORGE_OS_WINDOWS)
        if (proc_ && !done_) { TerminateProcess(proc_, 137); WaitForSingleObject(proc_, 5000); }
        if (in_) CloseHandle(in_);
        // The read end goes first: ClosePseudoConsole waits for the console host, which (before Windows 11 24H2)
        // blocks on writing output nobody reads any more.
        if (out_) CloseHandle(out_);
        if (pc_) api_().close(pc_);
        if (proc_) CloseHandle(proc_);
        pc_ = nullptr; in_ = out_ = proc_ = nullptr;
#elif !defined(FORGE_OS_IOS)
        if (pid_ > 0 && !done_) { if (::kill(-pid_, SIGKILL) != 0) ::kill(pid_, SIGKILL); int st; while (::waitpid(pid_, &st, 0) < 0 && errno == EINTR) {} }
        if (m_ >= 0) ::close(m_);
        m_ = -1;
#endif
        pid_ = -1; done_ = false; code_ = -1; eof_ = false;
    }
#if defined(FORGE_OS_WINDOWS)
    struct api {
        HRESULT (WINAPI* create)(COORD, HANDLE, HANDLE, DWORD, void**) = nullptr;
        HRESULT (WINAPI* resize)(void*, COORD) = nullptr;
        void (WINAPI* close)(void*) = nullptr;
    };
    static const api& api_() {
        static const api a = [] {
            api x; HMODULE k = GetModuleHandleW(L"kernel32.dll"); if (!k) return x;
            x.create = reinterpret_cast<decltype(x.create)>(reinterpret_cast<void*>(GetProcAddress(k, "CreatePseudoConsole")));
            x.resize = reinterpret_cast<decltype(x.resize)>(reinterpret_cast<void*>(GetProcAddress(k, "ResizePseudoConsole")));
            x.close = reinterpret_cast<decltype(x.close)>(reinterpret_cast<void*>(GetProcAddress(k, "ClosePseudoConsole")));
            if (!x.create || !x.resize || !x.close) x = api{};
            return x;
        }();
        return a;
    }
    void* pc_ = nullptr; HANDLE in_ = nullptr, out_ = nullptr, proc_ = nullptr;
    long long pid_ = -1;
#else
    int m_ = -1;
    pid_t pid_ = -1;
#endif
    bool done_ = false, eof_ = false; int code_ = -1;
};
} // namespace forge::os
