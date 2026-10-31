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
#endif
#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__FreeBSD__)
#include <sys/sysctl.h>
#include <sys/user.h>
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
        if (f(&v) == 0) { r.version = std::to_string(v.dwMajorVersion) + "." + std::to_string(v.dwMinorVersion); r.kernel_version = r.version + "." + std::to_string(v.dwBuildNumber); }
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
inline result<load_avg> load_average() {
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_ANDROID)
    double l[3] = {}; if (::getloadavg(l, 3) != 3) return {{}, error{-1, "getloadavg failed"}};
    return {{l[0], l[1], l[2]}, {}};
#elif defined(FORGE_OS_ANDROID)
    std::ifstream f("/proc/loadavg"); load_avg a; if (f >> a.one >> a.five >> a.fifteen) return {a, {}};
    return {{}, last_error("/proc/loadavg")};
#else
    return {{}, unsupported("load_average")};
#endif
}

struct memory_info { std::uint64_t total = 0, available = 0, free = 0, swap_total = 0, swap_free = 0, cgroup_limit = 0, cgroup_usage = 0; };
inline memory_info memory() {
    memory_info m; m.total = total_memory();
#if defined(__linux__)
    std::ifstream f("/proc/meminfo"); std::string k; std::uint64_t v; std::string unit;
    while (f >> k >> v) {
        std::getline(f, unit); v *= 1024;
        if (k == "MemAvailable:") m.available = v; else if (k == "MemFree:") m.free = v;
        else if (k == "SwapTotal:") m.swap_total = v; else if (k == "SwapFree:") m.swap_free = v;
    }
    auto rd = [](const char* p) -> std::uint64_t { std::ifstream g(p); std::string s; if (!(g >> s) || s == "max") return 0; return std::strtoull(s.c_str(), nullptr, 10); };
    m.cgroup_limit = rd("/sys/fs/cgroup/memory.max"); m.cgroup_usage = rd("/sys/fs/cgroup/memory.current");
    if (!m.cgroup_limit) { auto l = rd("/sys/fs/cgroup/memory/memory.limit_in_bytes"); if (l && l < (1ull << 60)) m.cgroup_limit = l; m.cgroup_usage = rd("/sys/fs/cgroup/memory/memory.usage_in_bytes"); }
#elif defined(FORGE_OS_WINDOWS)
    MEMORYSTATUSEX s{}; s.dwLength = sizeof s;
    if (GlobalMemoryStatusEx(&s)) { m.available = m.free = s.ullAvailPhys; m.swap_total = s.ullTotalPageFile - s.ullTotalPhys; m.swap_free = s.ullAvailPageFile > s.ullAvailPhys ? s.ullAvailPageFile - s.ullAvailPhys : 0; }
#else
    m.available = m.free = available_memory();
#endif
    return m;
}

// Cumulative busy/idle time per logical CPU (seconds since boot). usage(a, b) gives the busy fraction between samples.
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
#else
    return {{}, unsupported("per_cpu_times")};
#endif
}
inline double cpu_usage(const cpu_times& a, const cpu_times& b) {
    double t = b.total() - a.total(); if (t <= 0) return 0;
    return 1.0 - (b.idle - a.idle) / t;
}

struct process_usage { double user_s = 0, system_s = 0; std::uint64_t max_rss = 0; long major_faults = 0, minor_faults = 0, voluntary_switches = 0, involuntary_switches = 0; };
inline result<process_usage> self_usage() {
#if defined(FORGE_OS_POSIX)
    rusage r{}; if (::getrusage(RUSAGE_SELF, &r)) return {{}, last_error("getrusage")};
    process_usage u; u.user_s = double(r.ru_utime.tv_sec) + double(r.ru_utime.tv_usec) / 1e6; u.system_s = double(r.ru_stime.tv_sec) + double(r.ru_stime.tv_usec) / 1e6;
#if defined(__APPLE__)
    u.max_rss = std::uint64_t(r.ru_maxrss);
#else
    u.max_rss = std::uint64_t(r.ru_maxrss) * 1024;
#endif
    u.major_faults = r.ru_majflt; u.minor_faults = r.ru_minflt; u.voluntary_switches = r.ru_nvcsw; u.involuntary_switches = r.ru_nivcsw;
    return {u, {}};
#else
    FILETIME c, e, k, us; if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &us)) return {{}, last_error("GetProcessTimes")};
    auto s = [](FILETIME f) { return double((std::uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime) / 1e7; };
    process_usage u; u.user_s = s(us); u.system_s = s(k); return {u, {}};
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
} // namespace forge::os
