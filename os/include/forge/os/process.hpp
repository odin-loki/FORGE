#pragma once
// forge::os process control: spawn with argv/env/cwd, separate stdout/stderr capture, stdin data, timeouts,
// whole-tree kill (process group / Job Object) and resource limits applied at spawn. Guide: docs/os-wrapper.md.
// Like the rest of forge::os, calls return os::result<T>; nothing throws.
#include <forge/os/os.hpp>
#include <map>
#include <optional>
#include <span>
#if defined(FORGE_OS_POSIX)
#include <sys/time.h>
#endif

namespace forge::os {

struct proc_limits {
    std::uint64_t memory_bytes = 0;   // address-space cap (RLIMIT_AS / Job Object process memory); 0 = none
    std::uint64_t cpu_seconds = 0;    // RLIMIT_CPU / Job Object user time; 0 = none
    std::uint64_t open_files = 0;     // RLIMIT_NOFILE; 0 = inherit
    std::uint64_t file_size_bytes = 0;// RLIMIT_FSIZE (largest file the process may write); 0 = none. POSIX only
    bool no_core = false;             // RLIMIT_CORE = 0: no core dumps. POSIX only (Windows has no per-process core rlimit)
};

#if defined(FORGE_OS_POSIX)
// Apply `lim` to the calling process (POSIX; async-signal-safe, so it may run in a forked child before exec).
// A request above an existing hard limit is lowered to it (an unprivileged process cannot raise its hard limit,
// and setrlimit would otherwise fail and leave no limit at all); soft never exceeds hard. CPU: soft = cpu_seconds,
// hard = cpu_seconds + 1 (SIGXCPU first, then SIGKILL). Returns false if any setrlimit failed.
inline bool apply_limits(const proc_limits& lim) noexcept {
    bool ok = true;
    auto set = [&](int which, rlim_t soft, rlim_t hard) {
        struct rlimit cur{};
        if (::getrlimit(which, &cur) == 0 && cur.rlim_max != RLIM_INFINITY) {
            if (hard == RLIM_INFINITY || hard > cur.rlim_max) hard = cur.rlim_max;
        }
        if (soft > hard) soft = hard;
        struct rlimit r{soft, hard};
        if (::setrlimit(which, &r) != 0) ok = false;
    };
    if (lim.memory_bytes) set(RLIMIT_AS, rlim_t(lim.memory_bytes), rlim_t(lim.memory_bytes));
    if (lim.cpu_seconds) set(RLIMIT_CPU, rlim_t(lim.cpu_seconds), rlim_t(lim.cpu_seconds + 1));
    if (lim.open_files) set(RLIMIT_NOFILE, rlim_t(lim.open_files), rlim_t(lim.open_files));
    if (lim.file_size_bytes) set(RLIMIT_FSIZE, rlim_t(lim.file_size_bytes), rlim_t(lim.file_size_bytes));
    if (lim.no_core) set(RLIMIT_CORE, 0, 0);
    return ok;
}
#endif

struct spawn_spec {
    std::vector<std::string> argv;
    std::string cwd;                                   // empty = inherit
    std::map<std::string, std::string> env_set;        // added/overridden variables
    std::vector<std::string> env_unset;
    bool clear_env = false;                            // start from an empty environment
    std::optional<std::string> stdin_data;             // nullopt = /dev/null
    bool capture_stdout = true, capture_stderr = true; // false = inherit the parent's stream
    bool merge_stderr = false;                         // stderr goes into out
    std::uint64_t timeout_ms = 0;                      // 0 = wait forever; on expiry the process tree is killed
    std::uint64_t max_output = 64ull << 20;            // per stream; extra output is discarded, truncated = true
    proc_limits limits;
};

struct proc_result {
    int exit_code = -1;      // 128+signal when killed by a signal (POSIX convention)
    int term_signal = 0;
    bool timed_out = false, truncated = false;
    std::string out, err;
    std::uint64_t wall_ms = 0;
};

// Capability names this module contributes to os::caps()-style feature checks.
inline std::vector<std::string> process_caps() {
#if defined(FORGE_OS_IOS)
    return {};
#elif defined(FORGE_OS_WINDOWS)
    return {"spawn", "process.limits.memory", "process.limits.cpu", "process.tree_kill", "process.timeout"};
#else
    return {"spawn", "process.limits.memory", "process.limits.cpu", "process.limits.files", "process.tree_kill", "process.timeout", "signals"};
#endif
}

// Locate an executable on PATH (or return the name if it contains a separator and exists).
inline std::string find_executable(const std::string& name) {
    if (name.empty()) return {};
#if defined(FORGE_OS_WINDOWS)
    const char sep = ';';
    std::vector<std::string> exts = {"", ".exe", ".cmd", ".bat", ".com"};
    auto is_file = [](const std::string& p) { DWORD a = GetFileAttributesW(widen(p).c_str()); return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY); };
    if (name.find_first_of("/\\") != std::string::npos) { for (auto& e : exts) if (is_file(name + e)) return name + e; return {}; }
#else
    const char sep = ':';
    std::vector<std::string> exts = {""};
    auto is_file = [](const std::string& p) { struct stat st{}; return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) && ::access(p.c_str(), X_OK) == 0; };
    if (name.find('/') != std::string::npos) return is_file(name) ? name : std::string{};
#endif
    std::string path = getenv("PATH");
    std::size_t i = 0;
    while (i <= path.size()) {
        auto j = path.find(sep, i); if (j == std::string::npos) j = path.size();
        std::string d = path.substr(i, j - i); if (d.empty()) d = ".";
#if defined(FORGE_OS_WINDOWS)
        std::string base = d + "\\" + name;
#else
        std::string base = d + "/" + name;
#endif
        for (auto& e : exts) if (is_file(base + e)) return base + e;
        i = j + 1;
    }
    return {};
}

#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
namespace detail_proc {
inline void set_nonblock(int fd) { int fl = ::fcntl(fd, F_GETFL); ::fcntl(fd, F_SETFL, fl | O_NONBLOCK); }
inline void cloexec(int fd) { ::fcntl(fd, F_SETFD, FD_CLOEXEC); }
}
#endif

// Spawn, feed stdin, collect output, wait (with optional timeout). The child leads its own process group,
// so a timeout kills grandchildren too.
inline result<proc_result> spawn_wait(const spawn_spec& sp) {
    proc_result res;
    if (sp.argv.empty()) return {{}, {-1, "spawn: empty argv"}};
    auto t0 = monotonic_ns();
#if defined(FORGE_OS_IOS)
    return {{}, {unsupported_code, "spawn: process spawning is not available on iOS"}};
#elif defined(FORGE_OS_WINDOWS)
    std::string exe = find_executable(sp.argv[0]);
    if (exe.empty()) return {{}, {ERROR_FILE_NOT_FOUND, "spawn " + sp.argv[0] + ": not found"}};
    std::wstring cmd;
    bool is_batch = exe.size() > 4 && (exe.ends_with(".cmd") || exe.ends_with(".bat") || exe.ends_with(".CMD") || exe.ends_with(".BAT"));
    std::vector<std::string> av = sp.argv; av[0] = exe;
    if (is_batch) { av.insert(av.begin(), {"cmd.exe", "/d", "/c"}); exe = find_executable("cmd.exe"); }
    for (auto& a : av) {
        if (!cmd.empty()) cmd += L' ';
        std::wstring w = widen(a);
        if (!w.empty() && w.find_first_of(L" \t\"") == std::wstring::npos) { cmd += w; continue; }
        cmd += L'"'; std::size_t bs = 0;
        for (wchar_t c : w) { if (c == L'\\') { ++bs; continue; } if (c == L'"') cmd.append(bs * 2 + 1, L'\\'); else cmd.append(bs, L'\\'); bs = 0; cmd += c; }
        cmd.append(bs * 2, L'\\'); cmd += L'"';
    }
    // environment block
    std::map<std::wstring, std::wstring> envm;
    if (!sp.clear_env) {
        if (LPWCH blk = GetEnvironmentStringsW()) {
            for (LPWCH p = blk; *p; p += wcslen(p) + 1) { std::wstring e(p); auto eq = e.find(L'=', 1); if (eq != std::wstring::npos) envm[e.substr(0, eq)] = e.substr(eq + 1); }
            FreeEnvironmentStringsW(blk);
        }
    }
    for (auto& u : sp.env_unset) envm.erase(widen(u));
    for (auto& [k, v] : sp.env_set) envm[widen(k)] = widen(v);
    std::wstring envb; for (auto& [k, v] : envm) { envb += k; envb += L'='; envb += v; envb += L'\0'; } envb += L'\0';
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE in_r = nullptr, in_w = nullptr, out_r = nullptr, out_w = nullptr, err_r = nullptr, err_w = nullptr;
    auto closeh = [](HANDLE& h) { if (h) { CloseHandle(h); h = nullptr; } };
    if (!CreatePipe(&in_r, &in_w, &sa, 0)) return {{}, last_error("CreatePipe")};
    SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    if (sp.capture_stdout || sp.merge_stderr) { CreatePipe(&out_r, &out_w, &sa, 0); SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0); }
    if (sp.capture_stderr && !sp.merge_stderr) { CreatePipe(&err_r, &err_w, &sa, 0); SetHandleInformation(err_r, HANDLE_FLAG_INHERIT, 0); }
    STARTUPINFOW si{}; si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r;
    si.hStdOutput = out_w ? out_w : GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError = sp.merge_stderr ? out_w : (err_w ? err_w : GetStdHandle(STD_ERROR_HANDLE));
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (sp.limits.memory_bytes) { li.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY; li.ProcessMemoryLimit = SIZE_T(sp.limits.memory_bytes); }
        if (sp.limits.cpu_seconds) { li.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_TIME; li.BasicLimitInformation.PerProcessUserTimeLimit.QuadPart = LONGLONG(sp.limits.cpu_seconds) * 10000000LL; }
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof li);
    }
    PROCESS_INFORMATION pi{};
    std::wstring wcwd = widen(sp.cwd);
    BOOL ok = CreateProcessW(widen(exe).c_str(), cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED,
                             envb.data(), sp.cwd.empty() ? nullptr : wcwd.c_str(), &si, &pi);
    closeh(in_r); closeh(out_w); closeh(err_w);
    if (!ok) { auto e = last_error("CreateProcess " + sp.argv[0]); closeh(in_w); closeh(out_r); closeh(err_r); if (job) CloseHandle(job); return {{}, e}; }
    if (job) AssignProcessToJobObject(job, pi.hProcess);
    ResumeThread(pi.hThread);
    std::string in = sp.stdin_data.value_or("");
    struct rd { HANDLE h; std::string* s; std::uint64_t cap; bool trunc = false; };
    rd ro{out_r, &res.out, sp.max_output}, re{err_r, &res.err, sp.max_output};
    auto reader = [](void* p) -> unsigned { auto* r = static_cast<rd*>(p); char buf[4096]; DWORD n;
        while (r->h && ReadFile(r->h, buf, sizeof buf, &n, nullptr) && n) { if (r->s->size() + n <= r->cap) r->s->append(buf, n); else r->trunc = true; } return 0; };
    HANDLE th[2] = {nullptr, nullptr};
    if (out_r) th[0] = reinterpret_cast<HANDLE>(_beginthreadex(nullptr, 0, reader, &ro, 0, nullptr));
    if (err_r) th[1] = reinterpret_cast<HANDLE>(_beginthreadex(nullptr, 0, reader, &re, 0, nullptr));
    if (!in.empty()) { DWORD w = 0; std::size_t off = 0; while (off < in.size() && WriteFile(in_w, in.data() + off, DWORD(std::min<std::size_t>(in.size() - off, 1 << 20)), &w, nullptr)) off += w; }
    closeh(in_w);
    DWORD wr = WaitForSingleObject(pi.hProcess, sp.timeout_ms ? DWORD(sp.timeout_ms) : INFINITE);
    if (wr == WAIT_TIMEOUT) { res.timed_out = true; if (job) TerminateJobObject(job, 1); else TerminateProcess(pi.hProcess, 1); WaitForSingleObject(pi.hProcess, INFINITE); }
    for (auto t : th) if (t) { WaitForSingleObject(t, INFINITE); CloseHandle(t); }
    DWORD code = 0; GetExitCodeProcess(pi.hProcess, &code); res.exit_code = res.timed_out ? 124 : int(code);
    res.truncated = ro.trunc || re.trunc;
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread); closeh(out_r); closeh(err_r); if (job) CloseHandle(job);
    res.wall_ms = (monotonic_ns() - t0) / 1000000;
    return {std::move(res), {}};
#else
    std::string exe = find_executable(sp.argv[0]);
    if (exe.empty()) return {{}, {ENOENT, "spawn " + sp.argv[0] + ": command not found"}};
    // Everything the child needs is prepared before fork: after fork only async-signal-safe calls are made.
    std::vector<std::string> envs;
    if (!sp.clear_env) for (char** e = environ; e && *e; ++e) {
        std::string s(*e); auto eq = s.find('='); std::string k = s.substr(0, eq);
        bool drop = sp.env_set.count(k) > 0;
        for (auto& u : sp.env_unset) if (u == k) drop = true;
        if (!drop) envs.push_back(std::move(s));
    }
    for (auto& [k, v] : sp.env_set) envs.push_back(k + "=" + v);
    std::vector<char*> av, ev;
    for (auto& a : sp.argv) av.push_back(const_cast<char*>(a.c_str())); av.push_back(nullptr);
    for (auto& e : envs) ev.push_back(const_cast<char*>(e.c_str())); ev.push_back(nullptr);
    int pin[2] = {-1, -1}, pout[2] = {-1, -1}, perr[2] = {-1, -1}, pexec[2] = {-1, -1};
    auto cl = [](int& fd) { if (fd >= 0) { ::close(fd); fd = -1; } };
    if (::pipe(pin) || ::pipe(pexec)) return {{}, last_error("pipe")};
    if ((sp.capture_stdout || sp.merge_stderr) && ::pipe(pout)) return {{}, last_error("pipe")};
    if (sp.capture_stderr && !sp.merge_stderr && ::pipe(perr)) return {{}, last_error("pipe")};
    for (int fd : {pin[1], pout[0], perr[0], pexec[0], pexec[1]}) if (fd >= 0) detail_proc::cloexec(fd);
    int devnull = sp.stdin_data ? -1 : ::open("/dev/null", O_RDONLY | O_CLOEXEC);
    const char* cwd = sp.cwd.empty() ? nullptr : sp.cwd.c_str();
    const proc_limits lim = sp.limits;
    pid_t p = ::fork();
    if (p < 0) { auto e = last_error("fork"); for (int* f : {&pin[0], &pin[1], &pout[0], &pout[1], &perr[0], &perr[1], &pexec[0], &pexec[1]}) cl(*f); return {{}, e}; }
    if (p == 0) {
        ::setpgid(0, 0);
        ::signal(SIGPIPE, SIG_DFL);
        if (sp.stdin_data) ::dup2(pin[0], 0); else if (devnull >= 0) ::dup2(devnull, 0);
        if (pout[1] >= 0) ::dup2(pout[1], 1);
        if (sp.merge_stderr && pout[1] >= 0) ::dup2(pout[1], 2); else if (perr[1] >= 0) ::dup2(perr[1], 2);
        (void)apply_limits(lim);
        if (cwd && ::chdir(cwd) != 0) { int e = errno; (void)!::write(pexec[1], &e, sizeof e); ::_exit(127); }
        ::execve(exe.c_str(), av.data(), ev.data());
        int e = errno; (void)!::write(pexec[1], &e, sizeof e); ::_exit(127);
    }
    ::setpgid(p, p);   // also in the parent, to close the race
    cl(pin[0]); cl(pout[1]); cl(perr[1]); cl(pexec[1]); if (devnull >= 0) ::close(devnull);
    int child_errno = 0;
    { auto n = ::read(pexec[0], &child_errno, sizeof child_errno); if (n != sizeof child_errno) child_errno = 0; cl(pexec[0]); }
    if (child_errno) {
        cl(pin[1]); cl(pout[0]); cl(perr[0]); int st; ::waitpid(p, &st, 0);
        return {{}, {child_errno, "spawn " + sp.argv[0] + ": " + error_text(child_errno)}};
    }
    std::string in = sp.stdin_data.value_or("");
    std::size_t in_off = 0;
    if (in.empty()) cl(pin[1]); else detail_proc::set_nonblock(pin[1]);
    if (pout[0] >= 0) detail_proc::set_nonblock(pout[0]);
    if (perr[0] >= 0) detail_proc::set_nonblock(perr[0]);
    std::uint64_t deadline = sp.timeout_ms ? t0 + sp.timeout_ms * 1000000ull : 0;
    char buf[16384];
    auto drain = [&](int& fd, std::string& dst) {
        for (;;) { auto n = ::read(fd, buf, sizeof buf);
            if (n > 0) { if (dst.size() + std::size_t(n) <= sp.max_output) dst.append(buf, std::size_t(n)); else res.truncated = true; continue; }
            if (n == 0) { cl(fd); } else if (errno == EINTR) continue;
            break; }
    };
    while (pout[0] >= 0 || perr[0] >= 0 || pin[1] >= 0) {
        struct pollfd fds[3]; int nf = 0, io = -1, ie = -1, ii = -1;
        if (pout[0] >= 0) { io = nf; fds[nf++] = {pout[0], POLLIN, 0}; }
        if (perr[0] >= 0) { ie = nf; fds[nf++] = {perr[0], POLLIN, 0}; }
        if (pin[1] >= 0) { ii = nf; fds[nf++] = {pin[1], POLLOUT, 0}; }
        int to = -1;
        if (deadline) { auto now = monotonic_ns(); if (now >= deadline) { res.timed_out = true; break; } to = int(std::min<std::uint64_t>((deadline - now) / 1000000 + 1, 1000)); }
        int r = ::poll(fds, nfds_t(nf), to);
        if (r < 0) { if (errno == EINTR) continue; break; }
        if (io >= 0 && fds[io].revents) drain(pout[0], res.out);
        if (ie >= 0 && fds[ie].revents) drain(perr[0], res.err);
        if (ii >= 0 && fds[ii].revents) {
            if (fds[ii].revents & (POLLERR | POLLHUP)) cl(pin[1]);
            else { auto n = ::write(pin[1], in.data() + in_off, std::min<std::size_t>(in.size() - in_off, 65536));
                   if (n > 0) in_off += std::size_t(n); else if (n < 0 && errno != EAGAIN && errno != EINTR) cl(pin[1]);
                   if (in_off >= in.size()) cl(pin[1]); }
        }
    }
    if (res.timed_out) ::kill(-p, SIGKILL);
    cl(pin[1]); cl(pout[0]); cl(perr[0]);
    int st = 0;
    for (;;) {
        pid_t w = ::waitpid(p, &st, deadline && !res.timed_out ? WNOHANG : 0);
        if (w == p) break;
        if (w < 0 && errno != EINTR) break;
        if (w == 0) { if (monotonic_ns() >= deadline) { res.timed_out = true; ::kill(-p, SIGKILL); } else sleep_ns(1000000); }
    }
    if (WIFEXITED(st)) res.exit_code = WEXITSTATUS(st);
    else if (WIFSIGNALED(st)) { res.term_signal = WTERMSIG(st); res.exit_code = 128 + res.term_signal; }
    if (res.timed_out) res.exit_code = 124;
    res.wall_ms = (monotonic_ns() - t0) / 1000000;
    return {std::move(res), {}};
#endif
}

// Live process for streaming use: spawn without waiting; read/wait/kill later. RAII: the destructor kills
// and reaps a child that is still running, so a forgotten handle never leaves a zombie.
class child {
public:
    child() = default;
    child(const child&) = delete; child& operator=(const child&) = delete;
    child(child&& o) noexcept { swap(o); }
    child& operator=(child&& o) noexcept { if (this != &o) { reset(); swap(o); } return *this; }
    ~child() { reset(); }
    std::uint64_t pid() const { return std::uint64_t(pid_); }
    bool running() const { return pid_ > 0 && !done_; }
    // Start argv with stdout (and stderr merged when asked) readable through read().
    static result<child> start(const std::vector<std::string>& argv, bool merge_stderr = false, const std::string& cwd = {}) {
        child c;
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
        std::string exe = find_executable(argv.empty() ? std::string{} : argv[0]);
        if (exe.empty()) return {{}, {ENOENT, "spawn: command not found"}};
        std::vector<char*> av; for (auto& a : argv) av.push_back(const_cast<char*>(a.c_str())); av.push_back(nullptr);
        int po[2]; if (::pipe(po)) return {{}, last_error("pipe")};
        detail_proc::cloexec(po[0]);
        const char* wd = cwd.empty() ? nullptr : cwd.c_str();
        pid_t p = ::fork();
        if (p < 0) { ::close(po[0]); ::close(po[1]); return {{}, last_error("fork")}; }
        if (p == 0) { ::setpgid(0, 0); ::dup2(po[1], 1); if (merge_stderr) ::dup2(po[1], 2); ::close(po[1]);
                      if (wd && ::chdir(wd) != 0) ::_exit(127); ::execve(exe.c_str(), av.data(), environ); ::_exit(127); }
        ::setpgid(p, p); ::close(po[1]);
        c.pid_ = p; c.out_ = po[0];
        return {std::move(c), {}};
#else
        (void)argv; (void)merge_stderr; (void)cwd;
        return {{}, {unsupported_code, "child::start: not available on this platform yet (use spawn_wait)"}};
#endif
    }
    // Start from a full spawn_spec: argv, cwd, env_set/env_unset/clear_env, limits (memory/cpu/files) and
    // merge_stderr. stdout (plus stderr when merged) is readable through read(); stdin is /dev/null and an
    // unmerged stderr is inherited. An exec failure (e.g. command not found) is reported here, not at wait().
    static result<child> start(const spawn_spec& sp) {
        child c;
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
        if (sp.argv.empty()) return {{}, {EINVAL, "spawn: empty argv"}};
        std::string exe = find_executable(sp.argv[0]);
        if (exe.empty()) return {{}, {ENOENT, "spawn " + sp.argv[0] + ": command not found"}};
        std::vector<std::string> envs;
        if (!sp.clear_env) for (char** e = environ; e && *e; ++e) {
            std::string s(*e); auto eq = s.find('='); std::string k = s.substr(0, eq);
            bool drop = sp.env_set.count(k) > 0;
            for (auto& u : sp.env_unset) if (u == k) drop = true;
            if (!drop) envs.push_back(std::move(s));
        }
        for (auto& [k, v] : sp.env_set) envs.push_back(k + "=" + v);
        std::vector<char*> av, ev;
        for (auto& a : sp.argv) av.push_back(const_cast<char*>(a.c_str())); av.push_back(nullptr);
        for (auto& e : envs) ev.push_back(const_cast<char*>(e.c_str())); ev.push_back(nullptr);
        int po[2], px[2];
        if (::pipe(po)) return {{}, last_error("pipe")};
        if (::pipe(px)) { ::close(po[0]); ::close(po[1]); return {{}, last_error("pipe")}; }
        detail_proc::cloexec(po[0]); detail_proc::cloexec(px[0]); detail_proc::cloexec(px[1]);
        int devnull = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
        const char* wd = sp.cwd.empty() ? nullptr : sp.cwd.c_str();
        const proc_limits lim = sp.limits; const bool merge = sp.merge_stderr;
        pid_t p = ::fork();
        if (p < 0) { auto e = last_error("fork"); for (int fd : {po[0], po[1], px[0], px[1], devnull}) if (fd >= 0) ::close(fd); return {{}, e}; }
        if (p == 0) {
            ::setpgid(0, 0); ::signal(SIGPIPE, SIG_DFL);
            if (devnull >= 0) ::dup2(devnull, 0);
            ::dup2(po[1], 1); if (merge) ::dup2(po[1], 2); ::close(po[1]);
            (void)apply_limits(lim);
            if (wd && ::chdir(wd) != 0) { int e = errno; (void)!::write(px[1], &e, sizeof e); ::_exit(127); }
            ::execve(exe.c_str(), av.data(), ev.data());
            int e = errno; (void)!::write(px[1], &e, sizeof e); ::_exit(127);
        }
        ::setpgid(p, p); ::close(po[1]); ::close(px[1]); if (devnull >= 0) ::close(devnull);
        int child_errno = 0;
        { auto n = ::read(px[0], &child_errno, sizeof child_errno); if (n != sizeof child_errno) child_errno = 0; ::close(px[0]); }
        if (child_errno) { int st; while (::waitpid(p, &st, 0) < 0 && errno == EINTR) {} ::close(po[0]); errno = child_errno; return {{}, last_error("exec " + sp.argv[0])}; }
        c.pid_ = p; c.out_ = po[0];
        return {std::move(c), {}};
#else
        (void)sp;
        return {{}, {unsupported_code, "child::start: not available on this platform yet (use spawn_wait)"}};
#endif
    }
    // Non-blocking: the exit code (128+signal) if the child has exited, else -1.
    int try_wait() {
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
        if (done_) return code_;
        if (pid_ <= 0) return -1;
        int st = 0; pid_t w;
        do { w = ::waitpid(pid_, &st, WNOHANG); } while (w < 0 && errno == EINTR);
        if (w != pid_) return -1;
        done_ = true; code_ = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
        return code_;
#else
        return -1;
#endif
    }
    bool exited() const { return done_; }
    int exit_code() const { return done_ ? code_ : -1; }
    // Read up to `max` bytes of output; empty string at end of stream. Waits at most timeout_ms (0 = block).
    std::string read(std::size_t max = 65536, std::uint64_t timeout_ms = 0) {
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
        if (out_ < 0) return {};
        if (timeout_ms) { struct pollfd f{out_, POLLIN, 0}; if (::poll(&f, 1, int(timeout_ms)) <= 0) return {}; }
        std::string s(max, '\0');
        auto n = ::read(out_, s.data(), max);
        if (n <= 0) { ::close(out_); out_ = -1; return {}; }
        s.resize(std::size_t(n)); return s;
#else
        (void)max; (void)timeout_ms; return {};
#endif
    }
    // Signal the whole process group.
    bool kill(int sig = 9) {
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
        return pid_ > 0 && !done_ && ::kill(-pid_, sig) == 0;
#else
        (void)sig; return false;
#endif
    }
    // Wait for exit; returns the exit code (128+signal), or -1 on timeout.
    int wait(std::uint64_t timeout_ms = 0) {
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
        if (done_) return code_;
        if (pid_ <= 0) return -1;
        std::uint64_t end = timeout_ms ? monotonic_ns() + timeout_ms * 1000000ull : 0;
        int st = 0;
        for (;;) {
            pid_t w = ::waitpid(pid_, &st, end ? WNOHANG : 0);
            if (w == pid_) break;
            if (w < 0 && errno != EINTR) return -1;
            if (w == 0) { if (monotonic_ns() >= end) return -1; sleep_ns(1000000); }
        }
        done_ = true; code_ = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
        return code_;
#else
        (void)timeout_ms; return -1;
#endif
    }
private:
    void swap(child& o) noexcept { std::swap(pid_, o.pid_); std::swap(out_, o.out_); std::swap(done_, o.done_); std::swap(code_, o.code_); }
    void reset() {
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
        if (out_ >= 0) { ::close(out_); out_ = -1; }
        if (pid_ > 0 && !done_) { ::kill(-pid_, SIGKILL); int st; while (::waitpid(pid_, &st, 0) < 0 && errno == EINTR) {} }
#endif
        pid_ = -1; done_ = false; code_ = -1;
    }
#if defined(FORGE_OS_POSIX)
    pid_t pid_ = -1;
#else
    long long pid_ = -1;
#endif
    int out_ = -1; bool done_ = false; int code_ = -1;
};

// Wait for whichever of several children exits first. Returns its index into `cs` (that child is reaped,
// so exit_code() is valid), or nullopt when timeout_ms (0 = forever) passes with all still running, or when
// every child has already been reaped. Null entries and finished children are skipped. Polls with a short
// back-off (1 ms rising to 10 ms) so it is portable and never steals another waiter's child.
inline std::optional<std::size_t> wait_any(std::span<child* const> cs, std::uint64_t timeout_ms = 0) {
    std::uint64_t end = timeout_ms ? monotonic_ns() + timeout_ms * 1000000ull : 0;
    std::uint64_t nap = 1000000ull;
    for (;;) {
        bool any_live = false;
        for (std::size_t i = 0; i < cs.size(); ++i) {
            child* c = cs[i];
            if (!c || c->exited() || !c->running()) continue;
            any_live = true;
            if (c->try_wait() != -1) return i;
        }
        if (!any_live) return std::nullopt;
        if (end && monotonic_ns() >= end) return std::nullopt;
        sleep_ns(nap); nap = std::min<std::uint64_t>(nap * 2, 10000000ull);
    }
}
// Wait until every child has exited; false on timeout (the ones that finished are reaped either way).
inline bool wait_all(std::span<child* const> cs, std::uint64_t timeout_ms = 0) {
    std::uint64_t end = timeout_ms ? monotonic_ns() + timeout_ms * 1000000ull : 0;
    for (;;) {
        auto i = wait_any(cs, end ? std::max<std::uint64_t>(1, (end - std::min(end, monotonic_ns())) / 1000000ull) : 0);
        if (!i) break;
    }
    for (auto* c : cs) if (c && c->running()) return false;
    return true;
}

// One running process as seen by list_processes()/process_info().
struct process_entry { std::uint64_t pid = 0, ppid = 0, rss_bytes = 0; std::string name, cmdline; };

inline result<std::vector<process_entry>> list_processes() {
    std::vector<process_entry> v;
#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_ANDROID)
    auto ls = list_dir("/proc");
    if (!ls) return {{}, ls.err};
    long page = ::sysconf(_SC_PAGESIZE);
    for (auto& e : ls.value) {
        if (e.name.empty() || e.name.find_first_not_of("0123456789") != std::string::npos) continue;
        process_entry p; p.pid = std::stoull(e.name);
        auto st = read_file("/proc/" + e.name + "/stat");
        if (!st) continue;
        auto& s = st.value; auto l = s.find('('), r = s.rfind(')');
        if (l == std::string::npos || r == std::string::npos) continue;
        p.name = s.substr(l + 1, r - l - 1);
        std::vector<std::string> f; std::size_t i = r + 2;
        while (i < s.size()) { auto j = s.find(' ', i); if (j == std::string::npos) j = s.size(); f.push_back(s.substr(i, j - i)); i = j + 1; }
        if (f.size() > 21) { p.ppid = std::stoull(f[1]); p.rss_bytes = std::stoull(f[21]) * std::uint64_t(page); }
        if (auto c = read_file("/proc/" + e.name + "/cmdline")) { p.cmdline = c.value; for (auto& ch : p.cmdline) if (ch == '\0') ch = ' '; while (!p.cmdline.empty() && p.cmdline.back() == ' ') p.cmdline.pop_back(); }
        v.push_back(std::move(p));
    }
    return {std::move(v), {}};
#else
    return {{}, {unsupported_code, "list_processes: not implemented on this platform yet"}};
#endif
}

inline result<process_entry> process_info(std::uint64_t pid) {
    auto all = list_processes();
    if (!all) return {{}, all.err};
    for (auto& p : all.value) if (p.pid == pid) return {p, {}};
    return {{}, {-1, "process_info: no such process"}};
}

} // namespace forge::os
