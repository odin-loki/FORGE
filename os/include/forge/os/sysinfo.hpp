#pragma once
// forge::os system information beyond os::sysinfo(): OS release and kernel, uptime, load average, boot time,
// memory (incl. swap and the cgroup limit that containers actually enforce), per-CPU busy times, and process
// resource usage. Linux reads /proc and /sys; Apple/FreeBSD use sysctl; Windows uses the Win32 APIs.
// Guide: docs/os-wrapper.md.
#include <forge/os/os.hpp>
#include <fstream>
#include <sstream>
#if defined(FORGE_OS_POSIX)
#include <sys/time.h>
#include <sys/utsname.h>
#endif
#if defined(FORGE_OS_WINDOWS)
#include <psapi.h>   // K32GetProcessMemoryInfo lives in kernel32 (no psapi.lib)
#include <cmath>
#include <thread>
#endif
#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/processor_info.h>
#elif defined(__FreeBSD__)
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/user.h>
#include <vm/vm_param.h>   // struct xswdev
#endif

namespace forge::os {
struct os_release { std::string name, version, id, kernel, kernel_version, arch, pretty; };
inline os_release release_info() {
    os_release r;
#if defined(FORGE_OS_POSIX)
    utsname u{}; if (::uname(&u) == 0) { r.kernel = u.sysname; r.kernel_version = u.release; r.arch = u.machine; }
#endif
#if defined(__linux__)
    std::ifstream f("/etc/os-release"); std::string line;
    if (!f) f.open("/usr/lib/os-release");
    while (std::getline(f, line)) {
        auto eq = line.find('='); if (eq == std::string::npos) continue;
        auto k = line.substr(0, eq), v = line.substr(eq + 1);
        if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'')) v = v.substr(1, v.size() - 2);
        if (k == "NAME") r.name = v; else if (k == "VERSION_ID") r.version = v; else if (k == "ID") r.id = v; else if (k == "PRETTY_NAME") r.pretty = v;
    }
    if (r.name.empty()) r.name = "Linux";
#elif defined(__APPLE__)
    char b[64] = {}; std::size_t n = sizeof b;
    if (::sysctlbyname("kern.osproductversion", b, &n, nullptr, 0) == 0) r.version = b;
#if defined(FORGE_OS_IOS)
    r.name = "iOS"; r.id = "ios";
#else
    r.name = "macOS"; r.id = "macos";
#endif
#elif defined(FORGE_OS_FREEBSD)
    r.name = "FreeBSD"; r.id = "freebsd"; r.version = r.kernel_version;
#elif defined(FORGE_OS_WINDOWS)
    r.name = "Windows"; r.id = "windows"; r.kernel = "NT";
    using rtl_t = LONG(WINAPI*)(OSVERSIONINFOW*);
    if (auto m = GetModuleHandleW(L"ntdll.dll")) if (auto f = reinterpret_cast<rtl_t>(reinterpret_cast<void*>(GetProcAddress(m, "RtlGetVersion")))) {
        OSVERSIONINFOW v{}; v.dwOSVersionInfoSize = sizeof v;
        if (f(&v) == 0) {   // marketing version: NT 10.0 is Windows 10 below build 22000 and Windows 11 from it
            std::string nt = std::to_string(v.dwMajorVersion) + "." + std::to_string(v.dwMinorVersion); r.kernel_version = nt + "." + std::to_string(v.dwBuildNumber);
            r.version = v.dwMajorVersion == 10 ? (v.dwBuildNumber >= 22000 ? "11" : "10") : nt == "6.3" ? "8.1" : nt == "6.2" ? "8" : nt == "6.1" ? "7" : nt;
        }
    }
    SYSTEM_INFO si{}; GetNativeSystemInfo(&si);
    r.arch = si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64 ? "x86_64" : si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_ARM64 ? "aarch64" : "x86";
#endif
    if (r.pretty.empty()) r.pretty = r.name + (r.version.empty() ? "" : " " + r.version);
    return r;
}

inline result<double> uptime_seconds() {
#if defined(__linux__)
    std::ifstream f("/proc/uptime"); double u = 0; if (f >> u) return {u, {}};
    return {0, last_error("/proc/uptime")};
#elif defined(__APPLE__) || defined(FORGE_OS_FREEBSD)
    timeval tv{}; std::size_t n = sizeof tv; int mib[2] = {CTL_KERN, KERN_BOOTTIME};
    if (::sysctl(mib, 2, &tv, &n, nullptr, 0)) return {0, last_error("kern.boottime")};
    timeval now{}; ::gettimeofday(&now, nullptr);
    return {double(now.tv_sec - tv.tv_sec) + double(now.tv_usec - tv.tv_usec) / 1e6, {}};
#elif defined(FORGE_OS_WINDOWS)
    return {double(GetTickCount64()) / 1000.0, {}};
#else
    return {0, unsupported("uptime")};
#endif
}
struct load_avg { double one = 0, five = 0, fifteen = 0; };
#if defined(FORGE_OS_WINDOWS)
namespace detail {
// Windows keeps no run-queue average, so one is emulated like Linux's: every 5 s a background thread (started
// by the first load_average() call, never stopped) samples "running + waiting" = busy CPUs (GetSystemTimes over
// the tick x active processors) + "\System\Processor Queue Length" (PDH, pdh.dll loaded at run time; 0 if
// unavailable), and folds it into exponentially decaying 1/5/15-minute averages with factors exp(-5/60),
// exp(-5/300), exp(-5/900). The first call measures 100 ms and seeds all three averages with that instantaneous
// value, so early values follow the current load rather than starting from 0. Blocked I/O waits (Linux D state)
// are not counted. The sampler thread and its state are deliberately leaked (no join at exit).
// The few PDH declarations used, written out here: <pdh.h> and <lm.h> (users.hpp) both define HLOG.
struct pdh_ {
    using query_t = void*; using counter_t = void*;
    struct fmt_value { DWORD CStatus; union { LONG longValue; double doubleValue; LONGLONG largeValue; LPCWSTR WideStringValue; }; };
    static constexpr DWORD fmt_double = 0x00000200;   // PDH_FMT_DOUBLE
    using open_t = LONG(WINAPI*)(LPCWSTR, DWORD_PTR, query_t*);
    using add_t = LONG(WINAPI*)(query_t, LPCWSTR, DWORD_PTR, counter_t*);
    using collect_t = LONG(WINAPI*)(query_t);
    using value_t = LONG(WINAPI*)(counter_t, DWORD, LPDWORD, fmt_value*);
};
struct win_load_ {
    std::mutex mu; double avg[3] = {0, 0, 0};
    pdh_::collect_t collect = nullptr; pdh_::value_t value = nullptr;
    pdh_::query_t query = nullptr; pdh_::counter_t counter = nullptr;
    ULONGLONG idle = 0, total = 0; double ncpu = 1;
    static ULONGLONG ft_(const FILETIME& f) { return (ULONGLONG(f.dwHighDateTime) << 32) | f.dwLowDateTime; }
    static void times_(ULONGLONG& i, ULONGLONG& t) { FILETIME a{}, k{}, u{}; if (GetSystemTimes(&a, &k, &u)) { i = ft_(a); t = ft_(k) + ft_(u); } }   // kernel includes idle
    win_load_() {
        DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS); ncpu = n ? double(n) : 1.0;
        if (HMODULE h = LoadLibraryExW(L"pdh.dll", nullptr, 0x800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/)) {
            auto open = reinterpret_cast<pdh_::open_t>(reinterpret_cast<void*>(GetProcAddress(h, "PdhOpenQueryW")));
            auto add = reinterpret_cast<pdh_::add_t>(reinterpret_cast<void*>(GetProcAddress(h, "PdhAddEnglishCounterW")));
            collect = reinterpret_cast<decltype(collect)>(reinterpret_cast<void*>(GetProcAddress(h, "PdhCollectQueryData")));
            value = reinterpret_cast<decltype(value)>(reinterpret_cast<void*>(GetProcAddress(h, "PdhGetFormattedCounterValue")));
            if (!(open && add && collect && value && open(nullptr, 0, &query) == ERROR_SUCCESS && add(query, L"\\System\\Processor Queue Length", 0, &counter) == ERROR_SUCCESS)) counter = nullptr;
        }
        times_(idle, total);
    }
    double sample_() {   // busy CPUs since the previous sample + threads waiting for a CPU now
        ULONGLONG i = idle, t = total; times_(idle, total);
        double busy = total > t ? (1.0 - double(idle - i) / double(total - t)) * ncpu : 0.0;
        double q = 0;
        if (counter && collect(query) == ERROR_SUCCESS) { pdh_::fmt_value v{}; if (value(counter, pdh_::fmt_double, nullptr, &v) == ERROR_SUCCESS && v.CStatus == ERROR_SUCCESS) q = v.doubleValue; }
        double x = busy + q; return x > 0 ? x : 0.0;
    }
};
inline win_load_* win_load_state_() {
    static win_load_* st = [] {
        auto* p = new win_load_();
        Sleep(100); double first = p->sample_(); { std::lock_guard<std::mutex> g(p->mu); p->avg[0] = p->avg[1] = p->avg[2] = first; }
        std::thread([p] {
            const double e[3] = {std::exp(-5.0 / 60), std::exp(-5.0 / 300), std::exp(-5.0 / 900)};
            for (;;) {
                Sleep(5000); double x = p->sample_();
                std::lock_guard<std::mutex> g(p->mu); for (int k = 0; k < 3; ++k) p->avg[k] = p->avg[k] * e[k] + x * (1 - e[k]);
            }
        }).detach();
        return p;
    }();
    return st;
}
inline result<load_avg> win_load_average_() {
    auto* p = win_load_state_(); std::lock_guard<std::mutex> g(p->mu); return {{p->avg[0], p->avg[1], p->avg[2]}, {}};
}
} // namespace detail
#endif
inline result<load_avg> load_average() {
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_ANDROID)
    double l[3] = {}; if (::getloadavg(l, 3) != 3) return {{}, error{-1, "getloadavg failed"}};
    return {{l[0], l[1], l[2]}, {}};
#elif defined(FORGE_OS_ANDROID)
    std::ifstream f("/proc/loadavg"); load_avg a; if (f >> a.one >> a.five >> a.fifteen) return {a, {}};
    return {{}, last_error("/proc/loadavg")};
#elif defined(FORGE_OS_WINDOWS)
    return detail::win_load_average_();
#else
    return {{}, unsupported("load_average")};
#endif
}

#if defined(__linux__)
namespace detail {
// The calling process's cgroup directory: v2 ("0::/path" in /proc/self/cgroup) under /sys/fs/cgroup, or for v1
// the memory controller's ("N:memory:/path") under /sys/fs/cgroup/memory. Inside a container without a cgroup
// namespace the host path does not exist there; the mount root (the container's own cgroup) is used instead.
// "" when neither hierarchy is mounted.
inline std::string cgroup_dir_(bool v2, const std::string& self = "/proc/self/cgroup") {
    std::string root = v2 ? "/sys/fs/cgroup" : "/sys/fs/cgroup/memory", probe = v2 ? "/memory.max" : "/memory.limit_in_bytes";
    std::ifstream f(self); std::string line, rel; bool found = false;
    while (std::getline(f, line)) {
        auto a = line.find(':'), b = a == std::string::npos ? a : line.find(':', a + 1); if (b == std::string::npos) continue;
        auto ctl = line.substr(a + 1, b - a - 1); bool hit = v2 ? line.compare(0, a, "0") == 0 && ctl.empty() : ("," + ctl + ",").find(",memory,") != std::string::npos;
        if (hit) { rel = line.substr(b + 1); found = true; break; }
    }
    std::string d = root + (rel == "/" ? "" : rel);
    if (found && os::stat(d + probe).exists) return d;
    if (os::stat(root + probe).exists || (v2 && os::stat(root + "/cgroup.controllers").exists)) return root;
    return {};
}
inline std::string cgroup2_dir() { return cgroup_dir_(true); }
inline std::uint64_t cg_read_(const std::string& p) { std::ifstream g(p); std::string s; if (!(g >> s) || s == "max") return 0; return std::strtoull(s.c_str(), nullptr, 10); }
}
#endif
struct memory_info { std::uint64_t total = 0, available = 0, free = 0, swap_total = 0, swap_free = 0, cgroup_limit = 0, cgroup_usage = 0; };
// free: memory nobody uses (no cache); available: what can be had without swapping (free + reclaimable cache).
// cgroup_limit: the tightest memory.max on the path from this process's cgroup to the root (0 = none).
inline memory_info memory() {
    memory_info m; m.total = total_memory();
#if defined(__linux__)
    std::ifstream f("/proc/meminfo"); std::string k; std::uint64_t v; std::string unit;
    while (f >> k >> v) {
        std::getline(f, unit); v *= 1024;
        if (k == "MemAvailable:") m.available = v; else if (k == "MemFree:") m.free = v;
        else if (k == "SwapTotal:") m.swap_total = v; else if (k == "SwapFree:") m.swap_free = v;
    }
    if (auto d = detail::cgroup_dir_(true); !d.empty()) {
        m.cgroup_usage = detail::cg_read_(d + "/memory.current");
        for (std::string p = d; p.size() >= 14 && p.rfind("/sys/fs/cgroup", 0) == 0; p = p.substr(0, p.rfind('/'))) {   // walk up to the mount root
            auto l = detail::cg_read_(p + "/memory.max"); if (l && (!m.cgroup_limit || l < m.cgroup_limit)) m.cgroup_limit = l;
            if (p == "/sys/fs/cgroup") break;
        }
    } else if (auto d1 = detail::cgroup_dir_(false); !d1.empty()) {
        auto l = detail::cg_read_(d1 + "/memory.limit_in_bytes"); if (l && l < (1ull << 60)) m.cgroup_limit = l;
        m.cgroup_usage = detail::cg_read_(d1 + "/memory.usage_in_bytes");
    }
#elif defined(FORGE_OS_WINDOWS)
    MEMORYSTATUSEX s{}; s.dwLength = sizeof s;
    if (GlobalMemoryStatusEx(&s)) {   // the commit limit minus RAM approximates the page file; keep free <= total
        m.available = m.free = s.ullAvailPhys;
        m.swap_total = s.ullTotalPageFile > s.ullTotalPhys ? s.ullTotalPageFile - s.ullTotalPhys : 0;
        m.swap_free = std::min(m.swap_total, s.ullAvailPageFile > s.ullAvailPhys ? s.ullAvailPageFile - s.ullAvailPhys : 0);
    }
    // Free = zeroed + free page lists (SystemMemoryListInformation; may need SeProfileSingleProcessPrivilege),
    // else available minus the standby/system cache (K32GetPerformanceInfo).
    struct mem_lists { ULONG_PTR zero, free, modified, modified_nowrite, bad, standby[8], repurposed[8], modified_pagefile; } ml{};
    using nqsi = LONG(WINAPI*)(int, void*, ULONG, ULONG*);
    auto q = reinterpret_cast<nqsi>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation")));
    PERFORMANCE_INFORMATION pi{}; pi.cb = sizeof pi;
    if (q && q(80 /*SystemMemoryListInformation*/, &ml, sizeof ml, nullptr) >= 0 && (ml.zero || ml.free)) m.free = std::uint64_t(ml.zero + ml.free) * page_size();
    else if (K32GetPerformanceInfo(&pi, sizeof pi)) { std::uint64_t cache = std::uint64_t(pi.SystemCache) * pi.PageSize; m.free = m.available > cache ? m.available - cache : 0; }
#elif defined(__APPLE__)
    m.available = available_memory();
    vm_statistics64_data_t vs{}; mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vs), &n) == KERN_SUCCESS) m.free = std::uint64_t(vs.free_count) * page_size();
    else m.free = m.available;
    xsw_usage sw{}; std::size_t sn = sizeof sw;
    if (::sysctlbyname("vm.swapusage", &sw, &sn, nullptr, 0) == 0) { m.swap_total = sw.xsu_total; m.swap_free = sw.xsu_avail; }
#elif defined(FORGE_OS_FREEBSD)
    m.available = available_memory();
    u_int fc = 0; std::size_t fcn = sizeof fc;
    m.free = ::sysctlbyname("vm.stats.vm.v_free_count", &fc, &fcn, nullptr, 0) == 0 ? std::uint64_t(fc) * page_size() : m.available;
    // Swap without libkvm: vm.swap_info.<n> is one struct xswdev per swap device (block counts in pages).
    int mib[16]; std::size_t ml = 16 - 1;
    if (::sysctlnametomib("vm.swap_info", mib, &ml) == 0) {
        for (int i = 0;; ++i) {
            mib[ml] = i; xswdev x{}; std::size_t xn = sizeof x;
            if (::sysctl(mib, unsigned(ml + 1), &x, &xn, nullptr, 0)) break;   // ENOENT past the last device
            m.swap_total += std::uint64_t(x.xsw_nblks) * page_size(); m.swap_free += std::uint64_t(x.xsw_nblks - x.xsw_used) * page_size();
        }
    }
#else
    m.available = m.free = available_memory();
#endif
    return m;
}

// Cumulative busy/idle time per logical CPU (seconds since boot): /proc/stat, host_processor_info (Apple),
// kern.cp_times (FreeBSD), NtQuerySystemInformationEx (Windows). usage(a, b) gives the busy fraction between samples.
struct cpu_times { double user = 0, system = 0, idle = 0, other = 0; double total() const { return user + system + idle + other; } };
inline result<std::vector<cpu_times>> per_cpu_times() {
    std::vector<cpu_times> out;
#if defined(__linux__)
    std::ifstream f("/proc/stat"); std::string line; double hz = double(::sysconf(_SC_CLK_TCK)); if (hz <= 0) hz = 100;
    while (std::getline(f, line)) {
        if (line.rfind("cpu", 0) != 0 || line.size() < 4 || line[3] == ' ') continue;
        std::istringstream is(line.substr(line.find(' '))); double v[10] = {}; for (auto& x : v) is >> x;
        cpu_times t; t.user = (v[0] + v[1]) / hz; t.system = (v[2] + v[5] + v[6]) / hz; t.idle = (v[3] + v[4]) / hz; t.other = (v[7]) / hz;
        out.push_back(t);
    }
    if (out.empty()) return {{}, last_error("/proc/stat")};
    return {std::move(out), {}};
#elif defined(__APPLE__)
    natural_t n = 0; processor_info_array_t info = nullptr; mach_msg_type_number_t cnt = 0;
    if (host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &n, &info, &cnt) != KERN_SUCCESS) return {{}, error{-1, "host_processor_info failed"}};
    double hz = double(::sysconf(_SC_CLK_TCK)); if (hz <= 0) hz = 100;
    auto* l = reinterpret_cast<processor_cpu_load_info_t>(info);
    for (natural_t i = 0; i < n; ++i) {
        cpu_times t; t.user = (double(l[i].cpu_ticks[CPU_STATE_USER]) + double(l[i].cpu_ticks[CPU_STATE_NICE])) / hz;
        t.system = double(l[i].cpu_ticks[CPU_STATE_SYSTEM]) / hz; t.idle = double(l[i].cpu_ticks[CPU_STATE_IDLE]) / hz; out.push_back(t);
    }
    vm_deallocate(mach_task_self(), vm_address_t(info), vm_size_t(cnt) * sizeof(integer_t));
    return {std::move(out), {}};
#elif defined(__FreeBSD__)
    std::size_t sz = 0; if (::sysctlbyname("kern.cp_times", nullptr, &sz, nullptr, 0)) return {{}, last_error("kern.cp_times")};
    std::vector<long> v(sz / sizeof(long)); if (::sysctlbyname("kern.cp_times", v.data(), &sz, nullptr, 0)) return {{}, last_error("kern.cp_times")};
    clockinfo ci{}; std::size_t cn = sizeof ci; double hz = ::sysctlbyname("kern.clockrate", &ci, &cn, nullptr, 0) == 0 && ci.stathz > 0 ? double(ci.stathz) : 128;
    for (std::size_t i = 0; i + CPUSTATES <= sz / sizeof(long); i += CPUSTATES) {   // ticks at stathz
        cpu_times t; t.user = double(v[i + CP_USER] + v[i + CP_NICE]) / hz; t.system = double(v[i + CP_SYS]) / hz;
        t.idle = double(v[i + CP_IDLE]) / hz; t.other = double(v[i + CP_INTR]) / hz; out.push_back(t);
    }
    if (out.empty()) return {{}, error{-1, "kern.cp_times: empty"}};
    return {std::move(out), {}};
#elif defined(FORGE_OS_WINDOWS)
    // NtQuerySystemInformationEx(SystemProcessorPerformanceInformation = 8) per processor group (100 ns units;
    // kernel time includes idle). ntdll is always loaded. Where the Ex call is missing or refuses the class
    // (Windows 7 before SP1, wine), NtQuerySystemInformation answers for the caller's group.
    struct sppi { LARGE_INTEGER idle, kernel, user, dpc, interrupt; ULONG interrupts; };
    using fn = LONG(WINAPI*)(int, void*, ULONG, void*, ULONG, ULONG*);
    using fn0 = LONG(WINAPI*)(int, void*, ULONG, ULONG*);
    HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    auto q = nt ? reinterpret_cast<fn>(reinterpret_cast<void*>(GetProcAddress(nt, "NtQuerySystemInformationEx"))) : nullptr;
    auto q0 = nt ? reinterpret_cast<fn0>(reinterpret_cast<void*>(GetProcAddress(nt, "NtQuerySystemInformation"))) : nullptr;
    if (!q && !q0) return {{}, unsupported("per_cpu_times (NtQuerySystemInformation)")};
    auto add = [&](const std::vector<sppi>& b, ULONG got) {
        for (std::size_t i = 0; i < got / sizeof(sppi) && i < b.size(); ++i) {
            cpu_times t; t.idle = double(b[i].idle.QuadPart) / 1e7; t.user = double(b[i].user.QuadPart) / 1e7;
            t.system = double(b[i].kernel.QuadPart - b[i].idle.QuadPart) / 1e7; if (t.system < 0) t.system = 0; out.push_back(t);
        }
    };
    WORD groups = GetActiveProcessorGroupCount(); if (!groups) groups = 1;
    LONG st = -1;
    for (WORD g = 0; q && g < groups; ++g) {
        USHORT grp = g; DWORD c = GetActiveProcessorCount(g); std::vector<sppi> b(c ? c : 1); ULONG got = 0;
        st = q(8, &grp, sizeof grp, b.data(), ULONG(b.size() * sizeof(sppi)), &got);
        if (st < 0) { out.clear(); break; }
        add(b, got);
    }
    if (out.empty() && q0) {
        SYSTEM_INFO si{}; GetSystemInfo(&si); std::vector<sppi> b(si.dwNumberOfProcessors ? si.dwNumberOfProcessors : 1); ULONG got = 0;
        st = q0(8, b.data(), ULONG(b.size() * sizeof(sppi)), &got);
        if (st >= 0) add(b, got);
    }
    if (out.empty() && st < 0) return {{}, error{int(st), "NtQuerySystemInformation(SystemProcessorPerformanceInformation) failed"}};
    if (out.empty()) return {{}, error{-1, "per_cpu_times: no processors reported"}};
    return {std::move(out), {}};
#else
    return {{}, unsupported("per_cpu_times")};
#endif
}
inline double cpu_usage(const cpu_times& a, const cpu_times& b) {
    double t = b.total() - a.total(); if (t <= 0) return 0;
    return 1.0 - (b.idle - a.idle) / t;
}

struct process_usage { double user_s = 0, system_s = 0; std::uint64_t max_rss = 0; long major_faults = 0, minor_faults = 0, voluntary_switches = 0, involuntary_switches = 0; };
inline std::uint64_t current_rss();
inline result<process_usage> self_usage() {
#if defined(FORGE_OS_POSIX)
    rusage r{}; if (::getrusage(RUSAGE_SELF, &r)) return {{}, last_error("getrusage")};
    process_usage u; u.user_s = double(r.ru_utime.tv_sec) + double(r.ru_utime.tv_usec) / 1e6; u.system_s = double(r.ru_stime.tv_sec) + double(r.ru_stime.tv_usec) / 1e6;
#if defined(__APPLE__)
    u.max_rss = std::uint64_t(r.ru_maxrss);
#else
    u.max_rss = std::uint64_t(r.ru_maxrss) * 1024;
#endif
    if (!u.max_rss) u.max_rss = current_rss();   // FreeBSD samples ru_maxrss on clock ticks: 0 for a young process
    u.major_faults = r.ru_majflt; u.minor_faults = r.ru_minflt; u.voluntary_switches = r.ru_nvcsw; u.involuntary_switches = r.ru_nivcsw;
    return {u, {}};
#else
    FILETIME c, e, k, us; if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &us)) return {{}, last_error("GetProcessTimes")};
    auto s = [](FILETIME f) { return double((std::uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime) / 1e7; };
    process_usage u; u.user_s = s(us); u.system_s = s(k);
    PROCESS_MEMORY_COUNTERS pmc{}; pmc.cb = sizeof pmc;   // Windows counts soft and hard faults together: minor_faults
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) { u.max_rss = std::uint64_t(pmc.PeakWorkingSetSize); u.minor_faults = long(pmc.PageFaultCount); }
    return {u, {}};
#endif
}
// Resident set size of this process now, in bytes (0 if unknown). Linux/Android: /proc/self/statm;
// macOS/iOS: task_info(MACH_TASK_BASIC_INFO); FreeBSD: sysctl KERN_PROC_PID ki_rssize; Windows: working set.
inline std::uint64_t current_rss() {
#if defined(__linux__)
    long pages = 0, rss = 0;
    if (FILE* f = std::fopen("/proc/self/statm", "r")) { if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) rss = 0; std::fclose(f); }
    return std::uint64_t(rss) * std::uint64_t(page_size());
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{}; mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &n) != KERN_SUCCESS) return 0;
    return std::uint64_t(info.resident_size);
#elif defined(__FreeBSD__)
    int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, int(::getpid())}; struct kinfo_proc kp{}; std::size_t n = sizeof kp;
    if (::sysctl(mib, 4, &kp, &n, nullptr, 0) != 0 || n != sizeof kp) return 0;
    return std::uint64_t(kp.ki_rssize) * std::uint64_t(page_size());
#elif defined(FORGE_OS_WINDOWS)
    PROCESS_MEMORY_COUNTERS pmc{}; pmc.cb = sizeof pmc;
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return 0;
    return std::uint64_t(pmc.WorkingSetSize);
#else
    return 0;
#endif
}

// ---------------------------------------------------------------- machine identity, boot time, resource budgets
// A stable identifier of this OS installation (not of the hardware; treat it as confidential and hash it before it
// leaves the machine).  Linux: /etc/machine-id (systemd / D-Bus).  macOS: the hardware UUID (kern.uuid).
// FreeBSD: kern.hostuuid.  Windows: HKLM\SOFTWARE\Microsoft\Cryptography MachineGuid.  iOS: identifierForVendor
// (per app vendor).  Android: unsupported until app glue passes Settings.Secure.ANDROID_ID to set_machine_id().
namespace detail {
inline std::string& machine_id_override_() { static std::string s; return s; }
inline std::mutex& machine_id_mutex_() { static std::mutex m; return m; }
}
inline void set_machine_id(std::string id) { std::lock_guard<std::mutex> g(detail::machine_id_mutex_()); detail::machine_id_override_() = std::move(id); }
inline result<std::string> machine_id() {
    { std::lock_guard<std::mutex> g(detail::machine_id_mutex_()); if (!detail::machine_id_override_().empty()) return {detail::machine_id_override_(), {}}; }
    [[maybe_unused]] auto clean = [](std::string s) {
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\0')) s.pop_back();
        for (auto& c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        return s;
    };
#if defined(FORGE_OS_ANDROID)
    return {{}, unsupported("machine id (pass Settings.Secure.ANDROID_ID to set_machine_id)")};
#elif defined(__linux__)
    for (const char* p : {"/etc/machine-id", "/var/lib/dbus/machine-id"}) {
        std::ifstream f(p); std::string s; if (f >> s && s.size() >= 16) return {clean(s), {}};
    }
    return {{}, unsupported("machine id (no /etc/machine-id)")};
#elif defined(FORGE_OS_IOS)
    void* objc = ::dlopen("/usr/lib/libobjc.A.dylib", RTLD_LAZY);
    auto get_class = objc ? reinterpret_cast<void* (*)(const char*)>(::dlsym(objc, "objc_getClass")) : nullptr;
    auto sel = objc ? reinterpret_cast<void* (*)(const char*)>(::dlsym(objc, "sel_registerName")) : nullptr;
    auto msg = objc ? reinterpret_cast<void* (*)(void*, void*)>(::dlsym(objc, "objc_msgSend")) : nullptr;
    if (get_class && sel && msg) if (void* cls = get_class("UIDevice")) if (void* dev = msg(cls, sel("currentDevice")))
        if (void* uuid = msg(dev, sel("identifierForVendor"))) if (void* str = msg(uuid, sel("UUIDString")))
            if (auto c = reinterpret_cast<const char*>(msg(str, sel("UTF8String")))) return {clean(c), {}};
    return {{}, unsupported("machine id (UIDevice identifierForVendor not available)")};
#elif defined(__APPLE__) || defined(FORGE_OS_FREEBSD)
#if defined(__APPLE__)
    const char* key = "kern.uuid";
#else
    const char* key = "kern.hostuuid";
#endif
    char b[128] = {}; std::size_t n = sizeof b - 1;
    if (::sysctlbyname(key, b, &n, nullptr, 0) == 0 && b[0]) return {clean(b), {}};
    return {{}, last_error(key)};
#elif defined(FORGE_OS_WINDOWS)
    HKEY k = nullptr;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Cryptography", 0, KEY_QUERY_VALUE | KEY_WOW64_64KEY, &k) != ERROR_SUCCESS) return {{}, unsupported("machine id (MachineGuid not readable)")};
    wchar_t w[128] = {}; DWORD sz = sizeof w - sizeof(wchar_t), ty = 0;
    LSTATUS rc = RegQueryValueExW(k, L"MachineGuid", nullptr, &ty, reinterpret_cast<BYTE*>(w), &sz); RegCloseKey(k);
    if (rc != ERROR_SUCCESS || ty != REG_SZ || !w[0]) return {{}, unsupported("machine id (MachineGuid not set)")};
    return {clean(narrow(w)), {}};
#else
    return {{}, unsupported("machine id")};
#endif
}
// When the system booted, as wall-clock nanoseconds since the Unix epoch (Linux: btime in /proc/stat; Apple /
// FreeBSD: kern.boottime; otherwise now - uptime, rounded to the second).
inline result<std::int64_t> boot_time_ns() {
#if defined(__linux__)
    { std::ifstream f("/proc/stat"); std::string k; long long v = 0;
      while (f >> k) { if (k == "btime" && f >> v && v > 0) return {std::int64_t(v) * 1000000000ll, {}}; std::string rest; std::getline(f, rest); } }
#elif defined(__APPLE__) || defined(FORGE_OS_FREEBSD)
    { timeval tv{}; std::size_t n = sizeof tv; int mib[2] = {CTL_KERN, KERN_BOOTTIME};
      if (::sysctl(mib, 2, &tv, &n, nullptr, 0) == 0 && tv.tv_sec > 0) return {std::int64_t(tv.tv_sec) * 1000000000ll + std::int64_t(tv.tv_usec) * 1000ll, {}}; }
#endif
    auto up = uptime_seconds(); if (!up) return {0, up.err};
    std::int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::int64_t b = now - std::int64_t(*up * 1e9); return {(b + 500000000ll) / 1000000000ll * 1000000000ll, {}};
}

// How many CPUs this process may actually use. affinity: logical CPUs in its affinity mask / cpuset; quota: CPU
// time the container or job allows, in CPUs (0 = no quota); usable = max(1, min(affinity, ceil(quota))).
//   Linux/Android: sched_getaffinity + cgroup v2 cpu.max (tightest on the path to the root) or the v1 cfs quota.
//   Windows: the process affinity mask + the job object's hard CPU rate cap (Windows containers use one).
//   FreeBSD: cpuset_getaffinity (jails with a cpuset) + an rctl pcpu rule on the process (needs kern.racct.enable).
//   macOS/iOS: hw.activecpu; those systems have no quotas.
struct cpu_budget { unsigned affinity = 1; double quota = 0; unsigned usable = 1; std::string source = "none"; };
// Memory this process may use. limit: the tightest container / job / resource-limit cap (0 = none);
// usable = the limit when one is set and below physical memory, else physical.
//   Linux/Android: cgroup v2 memory.max / v1 limit_in_bytes.  Windows: job object process and job memory limits.
//   FreeBSD: an rctl memoryuse rule on the process.  iOS: the jetsam budget (footprint + os_proc_available_memory).
//   Every POSIX system: a finite RLIMIT_AS counts too.
struct memory_budget { std::uint64_t physical = 0, limit = 0, usable = 0; std::string source = "none"; };
namespace detail {
// cgroup v2 cpu.max text ("max 100000" / "150000 100000"), or v1 "quota period" -> CPUs; 0 = unlimited.
inline double parse_cpu_max(const std::string& text) {
    std::istringstream in(text); std::string q; double period = 0; if (!(in >> q >> period) || q == "max" || period <= 0) return 0;
    double quota = std::strtod(q.c_str(), nullptr); return quota > 0 ? quota / period : 0;
}
#if defined(__linux__)
inline std::string slurp_small_(const std::string& p) { std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }
// The cgroup v1 cpu controller directory of this process ("" when not mounted).
inline std::string cgroup1_cpu_dir_() {
    std::ifstream f("/proc/self/cgroup"); std::string line, rel;
    while (std::getline(f, line)) {
        auto a = line.find(':'), b = a == std::string::npos ? a : line.find(':', a + 1); if (b == std::string::npos) continue;
        if (("," + line.substr(a + 1, b - a - 1) + ",").find(",cpu,") != std::string::npos) { rel = line.substr(b + 1); break; }
    }
    for (const char* root : {"/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup/cpu"}) {
        if (!rel.empty() && rel != "/" && os::stat(std::string(root) + rel + "/cpu.cfs_quota_us").exists) return std::string(root) + rel;
        if (os::stat(std::string(root) + "/cpu.cfs_quota_us").exists) return root;
    }
    return {};
}
#endif
#if defined(FORGE_OS_FREEBSD)
extern "C" int rctl_get_limits(const char*, size_t, char*, size_t);
// The amount of the first rctl rule on this process for `resource` ("memoryuse", "pcpu"); 0 when there is none
// or resource accounting is off.
inline std::uint64_t rctl_limit_(const char* resource) {
    std::string filter = "process:" + std::to_string(::getpid()); char out[4096] = {};
    if (rctl_get_limits(filter.c_str(), filter.size() + 1, out, sizeof out) != 0) return 0;
    std::string s(out), key = std::string(":") + resource + ":"; auto p = s.find(key); if (p == std::string::npos) return 0;
    auto eq = s.find('=', p); if (eq == std::string::npos) return 0;
    return std::strtoull(s.c_str() + eq + 1, nullptr, 10);
}
#endif
}
inline cpu_budget cpu_limits() {
    cpu_budget b; b.affinity = cpu_count();
#if defined(__linux__)
    if (auto d = detail::cgroup_dir_(true); !d.empty()) {
        for (std::string p = d; p.size() >= 14 && p.rfind("/sys/fs/cgroup", 0) == 0; p = p.substr(0, p.rfind('/'))) {
            double q = detail::parse_cpu_max(detail::slurp_small_(p + "/cpu.max")); if (q > 0 && (b.quota == 0 || q < b.quota)) { b.quota = q; b.source = "cgroup2"; }
            if (p == "/sys/fs/cgroup") break;
        }
    }
    if (b.quota == 0) if (auto d1 = detail::cgroup1_cpu_dir_(); !d1.empty()) {
        double q = detail::parse_cpu_max(detail::slurp_small_(d1 + "/cpu.cfs_quota_us") + " " + detail::slurp_small_(d1 + "/cpu.cfs_period_us"));
        if (q > 0) { b.quota = q; b.source = "cgroup1"; }
    }
#elif defined(FORGE_OS_WINDOWS)
    DWORD_PTR pm = 0, sm = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm) && pm) {   // 0 when the process spans processor groups
        unsigned n = 0; for (DWORD_PTR m = pm; m; m &= m - 1) ++n;
        if (n) b.affinity = std::min(b.affinity, n);
    }
    struct { DWORD flags; DWORD rate; } cr{};   // JOBOBJECT_CPU_RATE_CONTROL_INFORMATION
    if (QueryInformationJobObject(nullptr, static_cast<JOBOBJECTINFOCLASS>(15), &cr, sizeof cr, nullptr) && (cr.flags & 0x1)) {
        const double total = double(cpu_count());
        if ((cr.flags & 0x10) && HIWORD(cr.rate)) { b.quota = total * double(HIWORD(cr.rate)) / 10000.0; b.source = "job"; }   // MIN_MAX_RATE: MaxRate
        else if ((cr.flags & 0x4) && !(cr.flags & 0x2) && cr.rate) { b.quota = total * double(cr.rate) / 10000.0; b.source = "job"; }   // HARD_CAP
    }
#elif defined(FORGE_OS_FREEBSD)
    cpuset_t s; CPU_ZERO(&s);
    if (::cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_PID, -1, sizeof s, &s) == 0) { int n = CPU_COUNT(&s); if (n > 0) b.affinity = unsigned(n); }
    if (auto pc = detail::rctl_limit_("pcpu")) { b.quota = double(pc) / 100.0; b.source = "rctl"; }
#elif defined(__APPLE__)
    int n = 0; std::size_t len = sizeof n; if (::sysctlbyname("hw.activecpu", &n, &len, nullptr, 0) == 0 && n > 0) b.affinity = unsigned(n);
#endif
    b.usable = b.affinity ? b.affinity : 1;
    if (b.quota > 0) { auto q = unsigned(b.quota); if (double(q) < b.quota) ++q; b.usable = std::max(1u, std::min(b.usable, q)); }
    return b;
}
inline unsigned usable_cpu_count() { return cpu_limits().usable; }
inline memory_budget memory_limits() {
    memory_budget b; b.physical = total_memory();
    [[maybe_unused]] auto take = [&](std::uint64_t v, const char* src) { if (v && v < (1ull << 60) && (!b.limit || v < b.limit)) { b.limit = v; b.source = src; } };
#if defined(__linux__)
    take(memory().cgroup_limit, "cgroup");
#elif defined(FORGE_OS_WINDOWS)
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
    if (QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &li, sizeof li, nullptr)) {
        if (li.BasicLimitInformation.LimitFlags & 0x100 /*JOB_OBJECT_LIMIT_PROCESS_MEMORY*/) take(li.ProcessMemoryLimit, "job");
        if (li.BasicLimitInformation.LimitFlags & 0x200 /*JOB_OBJECT_LIMIT_JOB_MEMORY*/) take(li.JobMemoryLimit, "job");
    }
#elif defined(FORGE_OS_FREEBSD)
    take(detail::rctl_limit_("memoryuse"), "rctl");
#elif defined(FORGE_OS_IOS)
    take(current_rss() + available_memory(), "jetsam");
#endif
#if defined(FORGE_OS_POSIX) && defined(RLIMIT_AS)
    { rlimit rl{}; if (::getrlimit(RLIMIT_AS, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) take(std::uint64_t(rl.rlim_cur), "rlimit"); }
#endif
    b.usable = b.limit && (!b.physical || b.limit < b.physical) ? b.limit : b.physical;
    return b;
}
} // namespace forge::os
