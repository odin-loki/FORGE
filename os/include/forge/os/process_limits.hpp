#pragma once
// forge::os::limit_group: resource limits for a group of processes under one API. Guide: docs/os-wrapper.md.
//
//   Linux      cgroup v2, without systemd or D-Bus: a child cgroup is made next to (or, when the process's own
//              cgroup is an empty-able delegated subtree and `move_self` is set, below) the caller's cgroup; that
//              needs write access there (a delegated subtree: a systemd user session, a container, root).
//              memory.max / memory.high / cpu.max / cpu.weight / pids.max, membership through cgroup.procs,
//              usage from memory.current / memory.peak / cpu.stat / pids.current, cgroup.kill.
//   Windows    a job object: job memory, active-process limit, CPU rate hard cap or weight, kill on close.
//   FreeBSD    rctl(8) rules on each member process (`process:<pid>:memoryuse|pcpu|maxproc:deny=`; children
//              inherit them). Needs kern.racct.enable=1 and root. Limits are per process, not for the group.
//   Fallback   (Linux without a usable cgroup, FreeBSD without racct): RLIMIT_AS on each member (per process);
//              the group still tracks its members for kill() and usage(). backend() says "rlimit".
//   macOS, iOS, Android   no mechanism an application may use: create() returns unsupported when a limit is
//              asked for (spawn_spec::limits keeps its per-process rlimits and watchdog there).
//   enforced() reports which of the requested limits the backend really applies, and whether group-wide.
#include <forge/os/detail.hpp>
#include <forge/os/process.hpp>
#if defined(FORGE_OS_FREEBSD)
#include <sys/user.h>
extern "C" int rctl_add_rule(const char*, size_t, char*, size_t);      // <sys/rctl.h>, rctl_add_rule(2)
extern "C" int rctl_remove_rule(const char*, size_t, char*, size_t);
#endif

namespace forge::os {

struct process_limits {
    std::uint64_t memory_bytes = 0;        // hard cap (0 = none): beyond it allocations fail or the group is OOM-killed
    std::uint64_t memory_soft_bytes = 0;   // reclaim / throttle threshold (Linux cgroup memory.high only)
    double cpus = 0;                       // CPU cap in cores, e.g. 1.5 (0 = none)
    unsigned cpu_weight = 0;               // relative share, 1..10000 with 100 the default (0 = unchanged)
    std::uint64_t max_processes = 0;       // processes (Linux: tasks) alive in the group at once (0 = none)
    bool kill_on_close = true;             // closing the group kills what is left in it
    bool allow_fallback = true;            // false: fail unless a real group mechanism (cgroup, job, rctl) is available
    bool move_self = false;                // Linux: may move the calling process into a leaf cgroup to make room (see above)
};
enum limit_bits : unsigned { limit_memory = 1, limit_memory_soft = 2, limit_cpu = 4, limit_cpu_weight = 8, limit_processes = 16,
                             limit_group_wide = 32 /* the caps count the whole group, not each process */ };
struct group_usage { std::uint64_t memory_bytes = 0, memory_peak_bytes = 0, cpu_ns = 0, processes = 0; };

namespace detail::cg {
// Tests point this at a directory that stands in for the cgroup2 mount (files are then created as needed).
inline std::string& test_root() { static std::string s; return s; }
#if defined(__linux__)
inline int write_text(const std::string& path, const std::string& text, bool create) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC | (create ? O_CREAT | O_TRUNC : 0), 0644); if (fd < 0) return errno;
    int e = 0; if (::write(fd, text.data(), text.size()) < 0) e = errno;
    ::close(fd); return e;
}
inline std::string mount_point() {
    if (!test_root().empty()) return test_root();
    if (::access("/sys/fs/cgroup/cgroup.controllers", F_OK) == 0) return "/sys/fs/cgroup";
    if (::access("/sys/fs/cgroup/unified/cgroup.controllers", F_OK) == 0) return "/sys/fs/cgroup/unified";   // hybrid: no controllers there, usually
    return {};
}
inline std::string self_path() {   // "/user.slice/..." from the "0::" line
    if (!test_root().empty()) return "/self";
    std::string s = os::detail::slurp("/proc/self/cgroup"); std::size_t i = 0;
    while (i < s.size()) {
        std::size_t e = s.find('\n', i); if (e == std::string::npos) e = s.size();
        if (s.compare(i, 3, "0::") == 0) return s.substr(i + 3, e - i - 3);
        i = e + 1;
    }
    return {};
}
inline bool has_word(const std::string& text, const char* w) {
    std::size_t n = std::strlen(w), i = 0;
    while ((i = text.find(w, i)) != std::string::npos) {
        bool l = i == 0 || text[i - 1] == ' ' || text[i - 1] == '\n', r = i + n == text.size() || text[i + n] == ' ' || text[i + n] == '\n';
        if (l && r) return true;
        i += n;
    }
    return false;
}
#endif
}

class limit_group {
public:
    limit_group() = default;
    limit_group(const limit_group&) = delete; limit_group& operator=(const limit_group&) = delete;
    limit_group(limit_group&& o) noexcept { swap_(o); }
    limit_group& operator=(limit_group&& o) noexcept { if (this != &o) { close(); swap_(o); } return *this; }
    ~limit_group() { close(); }

    // A new, empty group carrying `lim`. `name` becomes part of the cgroup directory name (letters, digits, - _ .).
    static result<limit_group> create(const process_limits& lim, std::string_view name = "forge") {
        limit_group g; g.lim_ = lim;
        const bool any = lim.memory_bytes || lim.memory_soft_bytes || lim.cpus > 0 || lim.cpu_weight || lim.max_processes;
        (void)any;
#if defined(FORGE_OS_WINDOWS)
        (void)name;
        g.job_ = CreateJobObjectW(nullptr, nullptr); if (!g.job_) return {{}, last_error("CreateJobObject")};
        if (auto e = g.apply_job_()) { CloseHandle(g.job_); g.job_ = nullptr; return {{}, e}; }
        g.backend_ = "job"; g.open_ = true; return {std::move(g), {}};
#elif defined(__linux__) && !defined(__ANDROID__)
        std::string why;
        if (g.make_cgroup_(name, why)) { g.backend_ = "cgroup2"; g.open_ = true; return {std::move(g), {}}; }
        if (!lim.allow_fallback) return {{}, error{unsupported_code, "limit_group: no usable cgroup v2 subtree (" + why + ")"}};
        g.backend_ = "rlimit"; g.enforced_ = lim.memory_bytes ? unsigned(limit_memory) : 0u; g.why_ = why; g.open_ = true;
        g.att_.use_rlimits = true; g.att_.rlimits.memory_bytes = lim.memory_bytes;
        return {std::move(g), {}};
#elif defined(FORGE_OS_FREEBSD)
        (void)name;
        if (racct_enabled_()) { g.backend_ = "rctl"; g.enforced_ = (lim.memory_bytes ? limit_memory : 0u) | (lim.cpus > 0 ? limit_cpu : 0u) | (lim.max_processes ? limit_processes : 0u); g.open_ = true; return {std::move(g), {}}; }
        if (!lim.allow_fallback) return {{}, unsupported("limit_group: rctl (kern.racct.enable=0)")};
        g.backend_ = "rlimit"; g.enforced_ = lim.memory_bytes ? unsigned(limit_memory) : 0u; g.why_ = "kern.racct.enable=0 or not root"; g.open_ = true;
        g.att_.use_rlimits = true; g.att_.rlimits.memory_bytes = lim.memory_bytes;
        return {std::move(g), {}};
#else
        (void)name;
        if (any || !lim.allow_fallback) return {{}, unsupported("limit_group (no per-group limits an application may set)")};
#if defined(FORGE_OS_IOS)
        return {{}, unsupported("limit_group")};
#else
        g.backend_ = "rlimit"; g.open_ = true; return {std::move(g), {}};   // membership, kill() and usage() only
#endif
#endif
    }
    bool valid() const { return open_; }
    // "cgroup2", "job", "rctl" or "rlimit" (the per-process fallback).
    const char* backend() const { return backend_; }
    // limit_bits the backend applies; limit_group_wide when the caps count all members together.
    unsigned enforced() const { return enforced_; }
    // Why the fallback is in use ("" otherwise).
    const std::string& fallback_reason() const { return why_; }
    // Linux cgroup2: the group's directory ("" elsewhere).
    const std::string& path() const { return dir_; }

    // For spawn_spec::attach: the process starts inside the group (no window in which it runs outside it).
    // Backends that work on a running process (rctl; the fallback's bookkeeping) also need adopt(pid) after the
    // spawn: start() and run() below do both.
    const spawn_attach* attach() const { return open_ ? &att_ : nullptr; }
    // Move a running process into the group (its existing children stay where they are).
    status add(std::uint64_t pid) {
        if (!open_) return {{}, errno_error(EBADF, "limit_group: closed")};
#if defined(FORGE_OS_WINDOWS)
        HANDLE h = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, FALSE, DWORD(pid)); if (!h) return {{}, last_error("OpenProcess")};
        BOOL ok = AssignProcessToJobObject(job_, h); error e = ok ? error{} : last_error("AssignProcessToJobObject"); CloseHandle(h);
        if (!ok) return {{}, e};
        return {};
#else
#if defined(__linux__)
        if (procs_fd_ >= 0) {
            std::string s = std::to_string(pid) + "\n";
            if (::write(procs_fd_, s.data(), s.size()) < 0) return {{}, last_error("cgroup.procs")};
            return {};
        }
#endif
        return adopt(pid);
#endif
    }
    // Tell the group about a process that was spawned with attach(): rctl rules are added, the fallback applies
    // what it can (Linux: prlimit) and remembers the pid for kill() / usage(). A no-op for cgroups and jobs.
    status adopt(std::uint64_t pid) {
        if (!open_) return {{}, errno_error(EBADF, "limit_group: closed")};
#if defined(FORGE_OS_WINDOWS)
        (void)pid; return {};
#else
#if defined(__linux__)
        if (procs_fd_ >= 0) return {};
#if !defined(__ANDROID__)
        if (lim_.memory_bytes) { struct rlimit r{rlim_t(lim_.memory_bytes), rlim_t(lim_.memory_bytes)}; if (::prlimit(pid_t(pid), RLIMIT_AS, &r, nullptr) != 0 && errno != ESRCH) { auto e = last_error("prlimit"); pids_.push_back(pid); return {{}, e}; } }
#endif
#elif defined(FORGE_OS_FREEBSD)
        if (std::string_view(backend_) == "rctl") {
            auto rule = [&](const char* res, std::uint64_t v) -> error {
                std::string r = "process:" + std::to_string(pid) + ":" + res + ":deny=" + std::to_string(v); char out[8];
                if (rctl_add_rule(r.c_str(), r.size() + 1, out, sizeof out) != 0 && errno != ESRCH) return last_error("rctl_add_rule " + r);
                return {};
            };
            error e;
            if (lim_.memory_bytes) e = rule("memoryuse", lim_.memory_bytes);
            if (!e && lim_.cpus > 0) e = rule("pcpu", std::uint64_t(lim_.cpus * 100.0 + 0.5));
            if (!e && lim_.max_processes) e = rule("maxproc", lim_.max_processes);
            pids_.push_back(pid); if (e) return {{}, e}; return {};
        }
        if (lim_.memory_bytes) {
            int mib[5] = {CTL_KERN, KERN_PROC, KERN_PROC_RLIMIT, int(pid), RLIMIT_AS}; struct rlimit r{rlim_t(lim_.memory_bytes), rlim_t(lim_.memory_bytes)};
            if (::sysctl(mib, 5, nullptr, nullptr, &r, sizeof r) != 0 && errno != ESRCH) { auto e = last_error("kern.proc.rlimit"); pids_.push_back(pid); return {{}, e}; }
        }
#endif
        pids_.push_back(pid); return {};
#endif
    }
    // Spawn inside the group (see child::start / spawn_wait for the spec).
    result<child> start(spawn_spec sp) {
        sp.attach = attach(); auto c = child::start(sp);
        if (c) (void)adopt(c->pid());
        return c;
    }
    result<proc_result> run(spawn_spec sp) { sp.attach = attach(); return spawn_wait(sp); }

    // Replace the limits of a live group (the fallback only affects processes added later).
    status update(const process_limits& lim) {
        if (!open_) return {{}, errno_error(EBADF, "limit_group: closed")};
        lim_ = lim;
#if defined(FORGE_OS_WINDOWS)
        if (auto e = apply_job_()) return {{}, e};
#elif defined(__linux__)
        if (procs_fd_ >= 0) { if (auto e = write_limits_()) return {{}, e}; }
        else att_.rlimits.memory_bytes = lim.memory_bytes;
#else
        att_.rlimits.memory_bytes = lim.memory_bytes;
#endif
        return {};
    }
    result<group_usage> usage() const {
        group_usage u;
        if (!open_) return {u, errno_error(EBADF, "limit_group: closed")};
#if defined(FORGE_OS_WINDOWS)
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION ai{};
        if (!QueryInformationJobObject(job_, JobObjectBasicAccountingInformation, &ai, sizeof ai, nullptr)) return {u, last_error("QueryInformationJobObject")};
        u.cpu_ns = std::uint64_t(ai.TotalUserTime.QuadPart + ai.TotalKernelTime.QuadPart) * 100ull; u.processes = ai.ActiveProcesses;
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
        if (QueryInformationJobObject(job_, JobObjectExtendedLimitInformation, &li, sizeof li, nullptr)) u.memory_peak_bytes = li.PeakJobMemoryUsed;
        std::vector<ULONG_PTR> buf(2 + 1024); auto* list = reinterpret_cast<JOBOBJECT_BASIC_PROCESS_ID_LIST*>(buf.data());
        if (QueryInformationJobObject(job_, JobObjectBasicProcessIdList, list, DWORD(buf.size() * sizeof(ULONG_PTR)), nullptr))
            for (DWORD i = 0; i < list->NumberOfProcessIdsInList; ++i)
                if (HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, DWORD(list->ProcessIdList[i]))) {
                    PROCESS_MEMORY_COUNTERS_EX pm{}; pm.cb = sizeof pm;
                    if (K32GetProcessMemoryInfo(h, reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pm), sizeof pm)) u.memory_bytes += pm.PrivateUsage;
                    CloseHandle(h);
                }
        return {u, {}};
#else
#if defined(__linux__)
        if (procs_fd_ >= 0) {
            auto num = [&](const char* f) -> std::uint64_t { auto v = os::detail::slurp_ll(dir_ + "/" + f, 0); return v < 0 ? 0 : std::uint64_t(v); };
            u.memory_bytes = num("memory.current"); u.memory_peak_bytes = num("memory.peak"); u.processes = num("pids.current");
            std::string st = os::detail::slurp(dir_ + "/cpu.stat"); auto i = st.find("usage_usec ");
            if (i != std::string::npos) u.cpu_ns = std::strtoull(st.c_str() + i + 11, nullptr, 10) * 1000ull;
            if (!u.processes) { std::string p = os::detail::slurp(dir_ + "/cgroup.procs"); for (char c : p) u.processes += c == '\n'; }
            return {u, {}};
        }
#endif
        for (auto pid : pids_) {   // members we were told about (their descendants are not counted)
            if (::kill(pid_t(pid), 0) != 0 && errno == ESRCH) continue;
            ++u.processes;
#if defined(__linux__)
            long pages = 0, rss = 0;
            if (FILE* f = std::fopen(("/proc/" + std::to_string(pid) + "/statm").c_str(), "r")) { if (std::fscanf(f, "%ld %ld", &pages, &rss) == 2) u.memory_bytes += std::uint64_t(rss) * page_size(); std::fclose(f); }
#elif defined(FORGE_OS_FREEBSD)
            int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, int(pid)}; struct kinfo_proc kp{}; std::size_t n = sizeof kp;
            if (::sysctl(mib, 4, &kp, &n, nullptr, 0) == 0 && n == sizeof kp) {
                u.memory_bytes += std::uint64_t(kp.ki_rssize) * page_size();
                u.cpu_ns += std::uint64_t(kp.ki_runtime) * 1000ull;
            }
#endif
        }
        return {u, {}};
#endif
    }
    // Kill every process in the group (cgroup: all descendants; job: the whole job; otherwise the known members
    // and their process groups).
    status kill() {
        if (!open_) return {};
#if defined(FORGE_OS_WINDOWS)
        if (!TerminateJobObject(job_, 137)) return {{}, last_error("TerminateJobObject")};
        return {};
#else
#if defined(__linux__)
        if (procs_fd_ >= 0) {
            if (detail::cg::write_text(dir_ + "/cgroup.kill", "1", !detail::cg::test_root().empty()) == 0) return {};
            for (int round = 0; round < 50; ++round) {   // before Linux 5.14: signal until the list is empty
                std::string p = os::detail::slurp(dir_ + "/cgroup.procs"); if (os::detail::trim(p).empty()) return {};
                std::size_t i = 0;
                while (i < p.size()) { long v = std::strtol(p.c_str() + i, nullptr, 10); if (v > 1) ::kill(pid_t(v), SIGKILL); i = p.find('\n', i); if (i == std::string::npos) break; ++i; }
                sleep_ns(2'000'000);
            }
            return {{}, errno_error(EBUSY, "limit_group: processes keep appearing")};
        }
#endif
        for (auto pid : pids_) { if (pid > 1) { ::kill(pid_t(pid), SIGKILL); ::kill(-pid_t(pid), SIGKILL); } }
        return {};
#endif
    }
    // Leave the group: kill what is left (kill_on_close), then remove it. Safe to call twice.
    void close() {
        if (!open_) return;
        open_ = false;
#if defined(FORGE_OS_WINDOWS)
        if (job_) CloseHandle(job_);   // KILL_ON_JOB_CLOSE does the rest
        job_ = nullptr; att_.job = nullptr;
#else
#if defined(__linux__)
        if (procs_fd_ >= 0) {
            open_ = true; if (lim_.kill_on_close) (void)kill(); open_ = false;
            ::close(procs_fd_); procs_fd_ = -1; att_.cgroup_procs_fd = -1;
            if (detail::cg::test_root().empty()) { for (int i = 0; i < 100 && ::rmdir(dir_.c_str()) != 0 && errno == EBUSY; ++i) sleep_ns(2'000'000); }   // dying tasks leave asynchronously
            dir_.clear(); pids_.clear(); return;
        }
#endif
        if (lim_.kill_on_close) { open_ = true; (void)kill(); open_ = false; }
#if defined(FORGE_OS_FREEBSD)
        if (std::string_view(backend_) == "rctl") for (auto pid : pids_) { std::string r = "process:" + std::to_string(pid); char out[8]; rctl_remove_rule(r.c_str(), r.size() + 1, out, sizeof out); }
#endif
        pids_.clear();
#endif
    }

private:
    void swap_(limit_group& o) noexcept {
        std::swap(lim_, o.lim_); std::swap(open_, o.open_); std::swap(backend_, o.backend_); std::swap(enforced_, o.enforced_);
        std::swap(why_, o.why_); std::swap(dir_, o.dir_); std::swap(att_, o.att_); std::swap(pids_, o.pids_);
#if defined(FORGE_OS_WINDOWS)
        std::swap(job_, o.job_);
#else
        std::swap(procs_fd_, o.procs_fd_);
#endif
    }
#if defined(FORGE_OS_WINDOWS)
    error apply_job_() {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{}; auto& fl = li.BasicLimitInformation.LimitFlags;
        enforced_ = limit_group_wide;
        if (lim_.kill_on_close) fl |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (lim_.memory_bytes) { fl |= JOB_OBJECT_LIMIT_JOB_MEMORY; li.JobMemoryLimit = SIZE_T(lim_.memory_bytes); enforced_ |= limit_memory; }
        if (lim_.max_processes) { fl |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS; li.BasicLimitInformation.ActiveProcessLimit = DWORD(std::min<std::uint64_t>(lim_.max_processes, 0xFFFFFFFFull)); enforced_ |= limit_processes; }
        if (!SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &li, sizeof li)) return last_error("SetInformationJobObject");
        // CPU rate control (Windows 8+): a hard cap in 1/100 percent of all CPUs, or a weight 1..9.
        struct cpu_rate { DWORD flags; union { DWORD rate; DWORD weight; } u; } cr{};   // JOBOBJECT_CPU_RATE_CONTROL_INFORMATION
        if (lim_.cpus > 0) {
            double share = lim_.cpus / double(cpu_count() ? cpu_count() : 1); if (share > 1) share = 1;
            cr.flags = 0x1 /*ENABLE*/ | 0x4 /*HARD_CAP*/; cr.u.rate = DWORD(share * 10000.0 + 0.5); if (!cr.u.rate) cr.u.rate = 1;
        } else if (lim_.cpu_weight) {
            unsigned w = std::min<unsigned>(10000, lim_.cpu_weight); cr.flags = 0x1 /*ENABLE*/ | 0x2 /*WEIGHT_BASED*/;
            cr.u.weight = w <= 100 ? 1 + (w - 1) * 4 / 99 : 5 + (w - 100) * 4 / 9900;   // 1..9: 1 -> 1, 100 -> 5 (the default), 10000 -> 9
        }
        if (cr.flags) {
            if (SetInformationJobObject(job_, JOBOBJECTINFOCLASS(15) /*JobObjectCpuRateControlInformation*/, &cr, sizeof cr)) enforced_ |= lim_.cpus > 0 ? limit_cpu : limit_cpu_weight;
        } else { cr.flags = 0; SetInformationJobObject(job_, JOBOBJECTINFOCLASS(15), &cr, sizeof cr); }
        att_.job = job_;
        return {};
    }
    HANDLE job_ = nullptr;
#else
#if defined(__linux__)
    // Writes the limit files of dir_. An absent controller file is an error only for a limit that was asked for.
    error write_limits_() {
        const bool fake = !detail::cg::test_root().empty();
        auto put = [&](const char* file, const std::string& v, unsigned bit, bool wanted) -> error {
            int e = detail::cg::write_text(dir_ + "/" + file, v, fake);
            if (e == 0) { if (wanted) enforced_ |= bit; return {}; }
            if (!wanted) return {};
            return error{e, std::string("cgroup ") + file + ": " + error_text(e)};
        };
        enforced_ = limit_group_wide; error e;
        e = put("memory.max", lim_.memory_bytes ? std::to_string(lim_.memory_bytes) : "max", limit_memory, lim_.memory_bytes != 0); if (e) return e;
        e = put("memory.high", lim_.memory_soft_bytes ? std::to_string(lim_.memory_soft_bytes) : "max", limit_memory_soft, lim_.memory_soft_bytes != 0); if (e) return e;
        if (lim_.memory_bytes) (void)detail::cg::write_text(dir_ + "/memory.oom.group", "1", fake);   // an OOM takes the whole group, not one victim
        e = put("cpu.max", lim_.cpus > 0 ? std::to_string(std::max<long long>(1000, static_cast<long long>(lim_.cpus * 100000.0 + 0.5))) + " 100000" : "max 100000", limit_cpu, lim_.cpus > 0); if (e) return e;
        if (lim_.cpu_weight) { e = put("cpu.weight", std::to_string(std::min<unsigned>(10000, std::max<unsigned>(1, lim_.cpu_weight))), limit_cpu_weight, true); if (e) return e; }
        e = put("pids.max", lim_.max_processes ? std::to_string(lim_.max_processes) : "max", limit_processes, lim_.max_processes != 0);
        return e;
    }
    bool make_cgroup_(std::string_view name, std::string& why) {
        namespace cg = detail::cg;
        const std::string mnt = cg::mount_point(); if (mnt.empty()) { why = "no cgroup v2 mount"; return false; }
        const std::string self = cg::self_path(); if (self.empty()) { why = "the process is not in a cgroup v2 hierarchy"; return false; }
        const bool fake = !cg::test_root().empty();
        std::string clean; for (char c : name) if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.') clean += c;
        if (clean.empty()) clean = "forge";
        static std::atomic<unsigned> serial{0};
        const std::string leaf = clean + "-" + std::to_string(::getpid()) + "-" + std::to_string(serial.fetch_add(1));
        std::string need;
        if (lim_.memory_bytes || lim_.memory_soft_bytes) need += " +memory";
        if (lim_.cpus > 0 || lim_.cpu_weight) need += " +cpu";
        if (lim_.max_processes) need += " +pids";
        auto controllers_ok = [&](const std::string& parent) {
            std::string have = os::detail::slurp(parent + "/cgroup.subtree_control");
            bool ok = (need.find("memory") == std::string::npos || cg::has_word(have, "memory")) && (need.find("cpu") == std::string::npos || cg::has_word(have, "cpu")) &&
                      (need.find("pids") == std::string::npos || cg::has_word(have, "pids"));
            if (ok || need.empty()) return true;
            return cg::write_text(parent + "/cgroup.subtree_control", need.substr(1), false) == 0;   // works when the parent holds no processes
        };
        auto try_parent = [&](const std::string& parent) {
            if (::access(parent.c_str(), W_OK) != 0) { why = parent + ": not writable (not a delegated subtree)"; return false; }
            if (!controllers_ok(parent)) { why = parent + ": controllers (" + need.substr(1) + ") are not enabled for children"; return false; }
            std::string d = parent + "/" + leaf;
            if (::mkdir(d.c_str(), 0755) != 0) { why = d + ": " + error_text(errno); return false; }
            dir_ = d;
            if (auto e = write_limits_()) { why = e.what; ::rmdir(d.c_str()); dir_.clear(); return false; }
            procs_fd_ = ::open((d + "/cgroup.procs").c_str(), O_WRONLY | O_CLOEXEC | (fake ? O_CREAT | O_APPEND : 0), 0644);
            if (procs_fd_ < 0) { why = d + "/cgroup.procs: " + error_text(errno); ::rmdir(d.c_str()); dir_.clear(); return false; }
            att_.cgroup_procs_fd = procs_fd_; return true;
        };
        const std::string base = mnt + (self == "/" ? std::string() : self);
        // 1. A sibling of our own cgroup: the parent already distributes controllers (systemd slices, containers).
        if (self != "/") { std::string parent = dirname(base); if (parent.size() >= mnt.size() && try_parent(parent)) return true; }
        // 2. A child of our own cgroup. With processes inside, controllers cannot be enabled for children ("no
        //    internal processes"), unless everything in it is first moved to a leaf: only with move_self.
        std::string first = why;
        if (lim_.move_self && ::access(base.c_str(), W_OK) == 0) {
            std::string main = base + "/" + clean + "-main";
            if (::mkdir(main.c_str(), 0755) == 0 || errno == EEXIST) {
                std::string procs = os::detail::slurp(base + "/cgroup.procs"); std::size_t i = 0;
                while (i < procs.size()) { std::size_t e = procs.find('\n', i); if (e == std::string::npos) e = procs.size(); if (e > i) cg::write_text(main + "/cgroup.procs", procs.substr(i, e - i), fake); i = e + 1; }
            }
        }
        if (try_parent(base)) return true;
        if (!first.empty()) why = first + "; " + why;
        return false;
    }
#endif
#if defined(FORGE_OS_FREEBSD)
    static bool racct_enabled_() { int v = 0; std::size_t n = sizeof v; return ::sysctlbyname("kern.racct.enable", &v, &n, nullptr, 0) == 0 && v != 0 && ::geteuid() == 0; }
#endif
    int procs_fd_ = -1;
#endif
    process_limits lim_; bool open_ = false; const char* backend_ = ""; unsigned enforced_ = 0;
    std::string why_, dir_; spawn_attach att_; std::vector<std::uint64_t> pids_;
};

} // namespace forge::os
