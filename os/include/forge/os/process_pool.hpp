#pragma once
// forge::os::process_pool: a pool of worker *processes* with an API that mirrors forge::task::pool.
//
//   FORGE_PROCESS_TASK(square, [](std::string in) { int v = std::stoi(in); return std::to_string(v * v); });
//   int main(int argc, char** argv) {
//       if (forge::os::process_pool::run_worker_if_requested(argc, argv)) return 0;   // needed for spawn mode
//       forge::os::process_pool pp({.workers = 4});
//       auto f = pp.submit("square", "7");        // std::future<std::string>
//       pp.wait_all(); std::string r = f.get();   // "49"
//   }
//
// Tasks are named functions std::string(std::string) registered in every process image (so they work after
// exec, not only after fork). Payloads and results are opaque byte strings; serialise as you like.
//
// Modes:
//   fork  (POSIX default): workers are fork()ed children running the registered task loop. Linux, macOS,
//         FreeBSD, Android. Forking a multithreaded parent is only safe because the child touches nothing but
//         the registry, the pipes and the task function; tasks must not rely on locks held by other threads.
//   spawn (Windows only mode; optional on POSIX): the same binary is re-launched with
//         --forge-process-worker=<rfd>,<wfd> (POSIX fork+exec) or --forge-process-worker=pipe (Windows
//         CreateProcess with the worker's stdin/stdout as the IPC pipes). The program must call
//         run_worker_if_requested(argc, argv) first thing in main().
//   iOS: no fork/exec for apps. The constructor throws os_exception{os_errc::not_supported}.
//
// IPC: one pipe pair per worker. Frame = u32 body length, u8 type, u64 id, u32 name length, name, payload.
// Types: task (parent->worker), ok / err (worker->parent; err carries "<ScriptErrorType>\n<message>"),
// quit (parent->worker, graceful shutdown).
//
// Crash detection: if a worker's pipe hits EOF mid-task the parent reaps it and fails that task's future with
// os_exception: kind=signal + signo for a POSIX signal death (SIGSEGV, SIGSYS from a sandbox, SIGKILL ...) or a
// Windows NTSTATUS crash code (access violation=1, illegal instruction=2, int divide by zero=3, matching
// os::signal_name); kind=io for a plain non-zero exit. The worker is respawned; other tasks keep running.
// A task that throws in the worker fails its future with os_exception (kind from the worker's os_exception if
// any, else other) whose message is the worker's what().
//
// Sandbox: options.sandbox applies os::apply_sandbox in each worker right after it starts (before any task).
#include <forge/os/os.hpp>
#include <forge/os/oserror.hpp>
#include <forge/os/sandbox.hpp>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#if defined(FORGE_OS_WINDOWS)
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <pthread.h>
#if defined(FORGE_OS_MACOS)
#include <mach-o/dyld.h>
#elif defined(FORGE_OS_FREEBSD)
#include <sys/sysctl.h>
#endif
#endif

namespace forge::os {

using process_task_fn = std::function<std::string(std::string)>;

namespace detail::pp {
inline std::map<std::string, process_task_fn>& registry() { static std::map<std::string, process_task_fn> r; return r; }
inline constexpr const char* worker_flag = "--forge-process-worker=";
enum : std::uint8_t { msg_task = 1, msg_ok = 2, msg_err = 3, msg_quit = 4 };
struct msg { std::uint8_t type = 0; std::uint64_t id = 0; std::string name, payload; };

#if defined(FORGE_OS_WINDOWS)
using fd_t = HANDLE;
inline const fd_t bad_fd = INVALID_HANDLE_VALUE;
inline bool raw_write(fd_t h, const char* p, std::size_t n) {
    while (n) { DWORD put = 0; if (!WriteFile(h, p, DWORD(n > (1u << 30) ? (1u << 30) : n), &put, nullptr)) return false; p += put; n -= put; }
    return true;
}
inline bool raw_read(fd_t h, char* p, std::size_t n) {
    while (n) { DWORD got = 0; if (!ReadFile(h, p, DWORD(n > (1u << 30) ? (1u << 30) : n), &got, nullptr) || !got) return false; p += got; n -= got; }
    return true;
}
inline void close_fd(fd_t& h) { if (h != bad_fd) { CloseHandle(h); h = bad_fd; } }
#else
using fd_t = int;
inline const fd_t bad_fd = -1;
inline bool raw_write(fd_t f, const char* p, std::size_t n) {
    while (n) { auto r = ::write(f, p, n); if (r < 0) { if (errno == EINTR) continue; return false; } p += r; n -= std::size_t(r); }
    return true;
}
inline bool raw_read(fd_t f, char* p, std::size_t n) {
    while (n) { auto r = ::read(f, p, n); if (r < 0) { if (errno == EINTR) continue; return false; } if (r == 0) return false; p += r; n -= std::size_t(r); }
    return true;
}
inline void close_fd(fd_t& f) { if (f >= 0) { ::close(f); f = -1; } }
#endif

inline void put32(std::string& s, std::uint32_t v) { for (int i = 0; i < 4; ++i) s.push_back(char(v >> (8 * i))); }
inline void put64(std::string& s, std::uint64_t v) { for (int i = 0; i < 8; ++i) s.push_back(char(v >> (8 * i))); }
inline std::uint64_t get(const char* p, int n) { std::uint64_t v = 0; for (int i = 0; i < n; ++i) v |= std::uint64_t(std::uint8_t(p[i])) << (8 * i); return v; }

inline bool write_msg(fd_t f, const msg& m) {
    std::string body; body.push_back(char(m.type)); put64(body, m.id); put32(body, std::uint32_t(m.name.size()));
    body += m.name; body += m.payload;
    if (body.size() > 0xFFFFFFFFull) return false;
    std::string frame; put32(frame, std::uint32_t(body.size())); frame += body;
    return raw_write(f, frame.data(), frame.size());
}
inline std::optional<msg> read_msg(fd_t f) {
    char h[4]; if (!raw_read(f, h, 4)) return std::nullopt;
    std::uint32_t n = std::uint32_t(get(h, 4)); if (n < 13) return std::nullopt;
    std::string b(n, '\0'); if (!raw_read(f, b.data(), n)) return std::nullopt;
    msg m; m.type = std::uint8_t(b[0]); m.id = get(b.data() + 1, 8);
    std::uint32_t nl = std::uint32_t(get(b.data() + 9, 4)); if (13ull + nl > n) return std::nullopt;
    m.name = b.substr(13, nl); m.payload = b.substr(13 + nl);
    return m;
}

// The worker side: serve tasks until quit/EOF.
inline int serve(fd_t in, fd_t out, const sandbox_policy* sb) {
    if (sb) { if (auto st = apply_sandbox(*sb); st.err && !is_unsupported(st.err)) return 70; note_sandboxed(true); }
    for (;;) {
        auto m = read_msg(in);
        if (!m || m->type == msg_quit) return 0;
        msg r; r.id = m->id;
        auto it = registry().find(m->name);
        if (it == registry().end()) { r.type = msg_err; r.payload = std::string("ValueError\nunknown process task: ") + m->name; }
        else {
            try { r.payload = it->second(std::move(m->payload)); r.type = msg_ok; }
            catch (const os_exception& e) { r.type = msg_err; r.payload = std::string(e.script_type()) + "\n" + e.what(); }
            catch (const std::exception& e) { r.type = msg_err; r.payload = std::string("Exception\n") + e.what(); }
            catch (...) { r.type = msg_err; r.payload = "Exception\nunknown exception in process task"; }
        }
        if (!write_msg(out, r)) return 0;
    }
}

inline std::string self_exe() {
#if defined(FORGE_OS_WINDOWS)
    std::wstring w(32768, L'\0'); DWORD n = GetModuleFileNameW(nullptr, w.data(), DWORD(w.size())); w.resize(n);
    int k = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(std::size_t(k), '\0'); WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), s.data(), k, nullptr, nullptr); return s;
#elif defined(__linux__)
    char b[4096]; auto n = ::readlink("/proc/self/exe", b, sizeof b - 1); return n > 0 ? std::string(b, std::size_t(n)) : std::string();
#elif defined(FORGE_OS_MACOS)
    char b[4096]; std::uint32_t n = sizeof b; return _NSGetExecutablePath(b, &n) == 0 ? std::string(b) : std::string();
#elif defined(FORGE_OS_FREEBSD)
    int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1}; char b[4096]; std::size_t n = sizeof b;
    return ::sysctl(mib, 4, b, &n, nullptr, 0) == 0 ? std::string(b) : std::string();
#else
    return {};
#endif
}
} // namespace detail::pp

// Register a task under a name (call during static init in every image that may run as a worker).
inline bool register_process_task(const std::string& name, process_task_fn fn) { detail::pp::registry()[name] = std::move(fn); return true; }
#define FORGE_PROCESS_TASK(name, ...) \
    static const bool forge_process_task_reg_##name = ::forge::os::register_process_task(#name, __VA_ARGS__)

class process_pool {
public:
    enum class spawn_mode { fork, spawn };
    struct options {
        unsigned workers = 0;              // 0: FORGE_PROCESSES env, else hardware_concurrency
#if defined(FORGE_OS_WINDOWS)
        spawn_mode mode = spawn_mode::spawn;
#else
        spawn_mode mode = spawn_mode::fork;
#endif
        std::string exe{};                 // spawn mode: binary to launch (default: this executable)
        std::optional<sandbox_policy> sandbox{};
    };

    // Call first in main(). Returns true if this process was launched as a worker (and has finished serving).
    static bool run_worker_if_requested(int argc, char** argv, const std::optional<sandbox_policy>& sb = std::nullopt) {
        for (int i = 1; i < argc; ++i) {
            if (std::strncmp(argv[i], detail::pp::worker_flag, std::strlen(detail::pp::worker_flag)) != 0) continue;
            const char* spec = argv[i] + std::strlen(detail::pp::worker_flag);
            const sandbox_policy* p = sb ? &*sb : nullptr;
            sandbox_policy parsed;
#if defined(FORGE_OS_WINDOWS)
            (void)spec;
            HANDLE in = GetStdHandle(STD_INPUT_HANDLE), out = GetStdHandle(STD_OUTPUT_HANDLE);
            // Keep task stdout from corrupting the IPC stream: point the CRT stdout at stderr.
            SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE));
            if (const char* s = std::strchr(argv[i], ':')) { parsed = parse_sb(s + 1); if (!p) p = &parsed; }
            std::exit(detail::pp::serve(in, out, p));
#else
            int r = -1, w = -1; char sbs[8] = {0};
            if (std::sscanf(spec, "%d,%d:%7s", &r, &w, sbs) < 2) std::exit(64);
            if (sbs[0]) { parsed = parse_sb(sbs); if (!p) p = &parsed; }
            std::exit(detail::pp::serve(r, w, p));
#endif
        }
        return false;
    }

    process_pool() : process_pool(options{}) {}
    explicit process_pool(options o) : opt_(std::move(o)) {
#if defined(FORGE_OS_IOS)
        throw os_exception(os_errc::not_supported, unsupported_code, "process_pool: iOS apps cannot fork or exec worker processes");
#else
        if (opt_.mode == spawn_mode::spawn && opt_.exe.empty()) opt_.exe = detail::pp::self_exe();
#if defined(FORGE_OS_WINDOWS)
        if (opt_.mode == spawn_mode::fork) throw os_exception(os_errc::not_supported, unsupported_code, "process_pool: Windows has no fork; use spawn_mode::spawn");
#endif
        unsigned n = opt_.workers;
        if (!n) if (const char* e = std::getenv("FORGE_PROCESSES")) n = unsigned(std::strtoul(e, nullptr, 10));
        if (!n) n = std::max(1u, std::thread::hardware_concurrency());
        set_workers(n);
#endif
    }
    process_pool(const process_pool&) = delete;
    process_pool& operator=(const process_pool&) = delete;
    ~process_pool() { shutdown(); }

    unsigned size() const { std::lock_guard l(m_); return target_; }

    // Grow or shrink. Shrinking retires idle workers after their current task (graceful quit message).
    void set_workers(unsigned n) {
        if (!n) n = 1;
        std::lock_guard l(m_);
        if (stop_) return;
        target_ = n;
        while (live_ < target_) { ++live_; slots_.emplace_back(std::make_unique<slot>()); auto* s = slots_.back().get(); s->th = std::thread([this, s] { dispatch(s); }); }
        cv_.notify_all();
    }

    // Queue task `name` with `payload`; the future yields the worker's result string or throws os_exception.
    std::future<std::string> submit(const std::string& name, std::string payload) {
        auto pr = std::make_shared<std::promise<std::string>>();
        auto f = pr->get_future();
        {
            std::lock_guard l(m_);
            if (stop_) { pr->set_exception(std::make_exception_ptr(os_exception(os_errc::invalid_argument, 0, "process_pool: submit after shutdown"))); return f; }
            q_.push_back({next_id_++, name, std::move(payload), pr}); ++outstanding_;
        }
        cv_.notify_one();
        return f;
    }
    std::size_t pending() const { std::lock_guard l(m_); return q_.size(); }
    void wait_all() { std::unique_lock l(m_); idle_cv_.wait(l, [&] { return outstanding_ == 0; }); }

    // Graceful: finishes queued tasks, then sends quit to every worker and reaps it. Idempotent.
    void shutdown() {
        { std::lock_guard l(m_); if (stop_) return; }
        wait_all();
        { std::lock_guard l(m_); stop_ = true; }
        cv_.notify_all();
        for (auto& s : slots_) if (s->th.joinable()) s->th.join();
        slots_.clear();
    }

private:
    struct job { std::uint64_t id; std::string name, payload; std::shared_ptr<std::promise<std::string>> pr; };
    struct worker {
        detail::pp::fd_t to = detail::pp::bad_fd, from = detail::pp::bad_fd;
#if defined(FORGE_OS_WINDOWS)
        HANDLE proc = nullptr;
#else
        pid_t pid = -1;
#endif
    };
    struct slot { std::thread th; };

    static sandbox_policy parse_sb(const char* s) {
        sandbox_policy p; for (; *s; ++s) { if (*s == 'n') p.deny_network = true; if (*s == 'x') p.deny_exec = true; if (*s == 'w') p.deny_fs_write = true; } return p;
    }
    std::string sb_spec() const {
        if (!opt_.sandbox) return {};
        std::string s; if (opt_.sandbox->deny_network) s += 'n'; if (opt_.sandbox->deny_exec) s += 'x'; if (opt_.sandbox->deny_fs_write) s += 'w';
        return s.empty() ? s : ":" + s;
    }

    std::optional<worker> start(std::string& err) {
        worker w;
#if defined(FORGE_OS_WINDOWS)
        SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
        HANDLE c_in_r, c_in_w, c_out_r, c_out_w;
        if (!CreatePipe(&c_in_r, &c_in_w, &sa, 0)) { err = last_error("CreatePipe").what; return std::nullopt; }
        if (!CreatePipe(&c_out_r, &c_out_w, &sa, 0)) { err = last_error("CreatePipe").what; CloseHandle(c_in_r); CloseHandle(c_in_w); return std::nullopt; }
        SetHandleInformation(c_in_w, HANDLE_FLAG_INHERIT, 0); SetHandleInformation(c_out_r, HANDLE_FLAG_INHERIT, 0);
        std::string cmd = "\"" + opt_.exe + "\" " + detail::pp::worker_flag + "pipe" + sb_spec();
        std::wstring wc(cmd.size() + 1, L'\0'); int k = MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), int(cmd.size()), wc.data(), int(wc.size())); wc.resize(std::size_t(k));
        STARTUPINFOW si{}; si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = c_in_r; si.hStdOutput = c_out_w; si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        PROCESS_INFORMATION pi{};
        BOOL ok = CreateProcessW(nullptr, wc.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
        CloseHandle(c_in_r); CloseHandle(c_out_w);
        if (!ok) { err = last_error("CreateProcess " + opt_.exe).what; CloseHandle(c_in_w); CloseHandle(c_out_r); return std::nullopt; }
        CloseHandle(pi.hThread);
        w.to = c_in_w; w.from = c_out_r; w.proc = pi.hProcess;
        return w;
#else
        int a[2], b[2];   // a: parent->child, b: child->parent
        if (::pipe(a)) { err = last_error("pipe").what; return std::nullopt; }
        if (::pipe(b)) { err = last_error("pipe").what; ::close(a[0]); ::close(a[1]); return std::nullopt; }
        for (int f : {a[0], a[1], b[0], b[1]}) ::fcntl(f, F_SETFD, FD_CLOEXEC);
        std::string exe = opt_.exe, spec = std::string(detail::pp::worker_flag) + std::to_string(a[0]) + "," + std::to_string(b[1]) + sb_spec();
        pid_t pid = serialized_fork();   // one fork at a time (atfork handlers, e.g. Z3, livelock otherwise)
        if (pid < 0) { err = last_error("fork").what; for (int f : {a[0], a[1], b[0], b[1]}) ::close(f); return std::nullopt; }
        if (pid == 0) {
            ::close(a[1]); ::close(b[0]);
            ::signal(SIGPIPE, SIG_IGN);
            if (opt_.mode == spawn_mode::spawn) {
                ::fcntl(a[0], F_SETFD, 0); ::fcntl(b[1], F_SETFD, 0);
                char* av[] = {exe.data(), spec.data(), nullptr};
                ::execv(exe.c_str(), av);
                ::_exit(127);
            }
            ::_exit(detail::pp::serve(a[0], b[1], opt_.sandbox ? &*opt_.sandbox : nullptr));
        }
        ::close(a[0]); ::close(b[1]);
        w.to = a[1]; w.from = b[0]; w.pid = pid;
        return w;
#endif
    }

    // Reap a worker; returns the os_exception describing how it ended (kind ok if it exited 0).
    static os_exception reap(worker& w) {
        detail::pp::close_fd(w.to); detail::pp::close_fd(w.from);
#if defined(FORGE_OS_WINDOWS)
        WaitForSingleObject(w.proc, 5000);
        DWORD code = 0; GetExitCodeProcess(w.proc, &code); CloseHandle(w.proc); w.proc = nullptr;
        int sig = code == 0xC0000005u ? 1 : code == 0xC000001Du ? 2 : code == 0xC0000094u ? 3 : 0;
        if (sig) return os_exception(os_errc::signal, int(code), std::string("process_pool: worker crashed: ") + signal_name(sig), sig);
        if (code) return os_exception(os_errc::io, int(code), "process_pool: worker exited with code " + std::to_string(code));
        return os_exception(os_errc::ok, 0, "");
#else
        int st = 0; while (::waitpid(w.pid, &st, 0) < 0 && errno == EINTR) {}
        w.pid = -1;
        if (WIFSIGNALED(st)) {
            int s = WTERMSIG(st);
            return os_exception(os_errc::signal, s, std::string("process_pool: worker killed by ") + signal_name(s) + " (" + std::to_string(s) + ")", s);
        }
        int c = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
        if (c) return os_exception(os_errc::io, c, "process_pool: worker exited with code " + std::to_string(c) + (c == 127 ? " (exec failed)" : c == 70 ? " (sandbox could not be applied)" : ""));
        return os_exception(os_errc::ok, 0, "");
#endif
    }

    static os_errc kind_from_script_type(const std::string& t) {
        for (int k = 0; k <= int(os_errc::other); ++k) if (t == script_error_type(os_errc(k)) && t != "OSError") return os_errc(k);
        return t == "OSError" ? os_errc::io : os_errc::other;
    }

    void finish_one() { std::lock_guard l(m_); if (--outstanding_ == 0) idle_cv_.notify_all(); }

    void dispatch(slot*) {
#if !defined(FORGE_OS_WINDOWS)
        { sigset_t ss; sigemptyset(&ss); sigaddset(&ss, SIGPIPE); ::pthread_sigmask(SIG_BLOCK, &ss, nullptr); }   // dead worker -> EPIPE, not a parent kill
#endif
        std::optional<worker> w;
        for (;;) {
            job j;
            {
                std::unique_lock l(m_);
                cv_.wait(l, [&] { return stop_ || !q_.empty() || live_ > target_; });
                if (q_.empty() && (stop_ || live_ > target_)) { --live_; break; }
                j = std::move(q_.front()); q_.pop_front();
            }
            if (!w) {
                std::string err; w = start(err);
                if (!w) { j.pr->set_exception(std::make_exception_ptr(os_exception(os_errc::other, 0, "process_pool: cannot start worker: " + err))); finish_one(); continue; }
            }
            bool sent = detail::pp::write_msg(w->to, {detail::pp::msg_task, j.id, j.name, std::move(j.payload)});
            auto r = sent ? detail::pp::read_msg(w->from) : std::nullopt;
            if (!r || r->id != j.id) {   // crash / protocol failure: reap, report, respawn on next task
                auto e = reap(*w); w.reset();
                if (e.kind == os_errc::ok) e = os_exception(os_errc::broken_pipe, 0, "process_pool: worker closed its pipe mid-task");
                j.pr->set_exception(std::make_exception_ptr(e));
            } else if (r->type == detail::pp::msg_ok) {
                j.pr->set_value(std::move(r->payload));
            } else {
                auto nl = r->payload.find('\n');
                std::string t = r->payload.substr(0, nl), what = nl == std::string::npos ? std::string() : r->payload.substr(nl + 1);
                j.pr->set_exception(std::make_exception_ptr(os_exception(kind_from_script_type(t), 0, what)));
            }
            finish_one();
        }
        if (w) { detail::pp::write_msg(w->to, {detail::pp::msg_quit, 0, {}, {}}); (void)reap(*w); }
    }

    options opt_;
    mutable std::mutex m_;
    std::condition_variable cv_, idle_cv_;
    std::deque<job> q_;
    std::vector<std::unique_ptr<slot>> slots_;
    std::uint64_t next_id_ = 1;
    std::size_t outstanding_ = 0;
    unsigned target_ = 0, live_ = 0;
    bool stop_ = false;
};

} // namespace forge::os

namespace forge { using os::process_pool; }
