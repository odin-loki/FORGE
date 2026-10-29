#pragma once
// High-resolution timing and hardware performance counters.
//   cycle_counter(): rdtsc (x86) / cntvct_el0 (arm64) / monotonic_ns elsewhere; counter_hz() its frequency
//   (cntfrq_el0, or TSC calibrated against the monotonic clock).
//   precise_sleep_ns(): Windows high-resolution waitable timer; POSIX absolute clock_nanosleep + short spin.
//   perf_counters: Linux/Android perf_event_open (cycles, instructions, cache refs/misses, branch misses; obeys
//   perf_event_paranoid); macOS kperf (kpc_* fixed counters: cycles, instructions; needs root); Windows
//   QueryThreadCycleTime (cycles only); FreeBSD/iOS: TSC/CNTVCT "cycles" only, the rest unsupported.
#include <forge/os/detail.hpp>
#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#endif
#if (defined(__x86_64__) || defined(__i386__)) && !defined(_MSC_VER)
#include <x86intrin.h>
#endif

namespace forge::os {
inline std::uint64_t cycle_counter() {
#if defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#elif defined(__aarch64__)
    std::uint64_t v; asm volatile("isb; mrs %0, cntvct_el0" : "=r"(v)); return v;
#else
    return monotonic_ns();
#endif
}
inline std::uint64_t counter_hz() {
#if defined(__aarch64__)
    std::uint64_t f; asm volatile("mrs %0, cntfrq_el0" : "=r"(f)); return f;
#elif defined(__x86_64__) || defined(__i386__)
    static const std::uint64_t hz = [] {
        std::uint64_t t0 = monotonic_ns(), c0 = cycle_counter(); while (monotonic_ns() - t0 < 20000000ull) {}
        std::uint64_t t1 = monotonic_ns(), c1 = cycle_counter(); return (c1 - c0) * 1000000000ull / (t1 - t0);
    }();
    return hz;
#else
    return 1000000000ull;
#endif
}
inline std::uint64_t timer_resolution_ns() {
#if defined(FORGE_OS_WINDOWS)
    LARGE_INTEGER f; QueryPerformanceFrequency(&f); return std::uint64_t(1000000000ull / std::uint64_t(f.QuadPart ? f.QuadPart : 1)) + 1;
#else
    timespec r{}; ::clock_getres(CLOCK_MONOTONIC, &r); return std::uint64_t(r.tv_sec) * 1000000000ull + std::uint64_t(r.tv_nsec);
#endif
}
// Sleep for ns with sub-millisecond accuracy (spins for the final stretch).
inline void precise_sleep_ns(std::uint64_t ns) {
    std::uint64_t end = monotonic_ns() + ns; const std::uint64_t spin = 200000;   // 200us
    if (ns > spin) {
#if defined(FORGE_OS_WINDOWS)
        static thread_local HANDLE t = [] { HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, 0x2 /*HIGH_RESOLUTION*/, TIMER_ALL_ACCESS); return h ? h : CreateWaitableTimerW(nullptr, TRUE, nullptr); }();
        LARGE_INTEGER due; due.QuadPart = -std::int64_t((ns - spin) / 100);
        if (t && SetWaitableTimer(t, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(t, INFINITE); else sleep_ns(ns - spin);
#elif defined(__APPLE__)
        sleep_ns(ns - spin);
#else
        timespec ts; ::clock_gettime(CLOCK_MONOTONIC, &ts);
        std::uint64_t a = std::uint64_t(ts.tv_sec) * 1000000000ull + std::uint64_t(ts.tv_nsec) + ns - spin;
        timespec abs{time_t(a / 1000000000ull), long(a % 1000000000ull)};
        while (::clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &abs, nullptr) == EINTR) {}
#endif
    }
    while (monotonic_ns() < end) {}
}

enum class perf_event { cycles, instructions, cache_references, cache_misses, branch_misses };
inline const char* perf_event_name(perf_event e) {
    switch (e) { case perf_event::cycles: return "cycles"; case perf_event::instructions: return "instructions"; case perf_event::cache_references: return "cache-references";
                 case perf_event::cache_misses: return "cache-misses"; default: return "branch-misses"; }
}
// Counts for the calling thread between start() and stop().
class perf_counters {
public:
    struct reading { perf_event event; std::uint64_t value = 0; bool valid = false; };
    perf_counters() = default;
    perf_counters(const perf_counters&) = delete; perf_counters& operator=(const perf_counters&) = delete;
    ~perf_counters() { close_(); }
    status open(const std::vector<perf_event>& evs) {
        close_(); evs_ = evs; if (evs.empty()) return {{}, {22, "perf_counters: no events"}};
#if defined(__linux__)
        for (auto e : evs) {
            perf_event_attr a{}; a.size = sizeof a; a.type = PERF_TYPE_HARDWARE; a.disabled = fds_.empty() ? 1 : 0; a.exclude_kernel = 1; a.exclude_hv = 1;
            a.read_format = PERF_FORMAT_GROUP;
            a.config = e == perf_event::cycles ? PERF_COUNT_HW_CPU_CYCLES : e == perf_event::instructions ? PERF_COUNT_HW_INSTRUCTIONS : e == perf_event::cache_references ? PERF_COUNT_HW_CACHE_REFERENCES
                     : e == perf_event::cache_misses ? PERF_COUNT_HW_CACHE_MISSES : PERF_COUNT_HW_BRANCH_MISSES;
            int fd = int(::syscall(SYS_perf_event_open, &a, 0, -1, fds_.empty() ? -1 : fds_[0], 0));
            if (fd < 0 && evs.size() == 1 && e == perf_event::cycles) { close_(); backend_ = "cycle_counter"; return {}; }   // perf_event locked down: TSC/CNTVCT
            if (fd < 0) { auto er = last_error(std::string("perf_event_open ") + perf_event_name(e) + " (perf_event_paranoid=" + detail::trim(detail::slurp("/proc/sys/kernel/perf_event_paranoid")) + ")"); close_(); return {{}, er}; }
            fds_.push_back(fd);
        }
        backend_ = "perf_event"; return {};
#elif defined(FORGE_OS_MACOS)
        kp_ = ::dlopen("/System/Library/PrivateFrameworks/kperf.framework/kperf", RTLD_LAZY);
        if (!kp_) return {{}, unsupported("kperf.framework")};
        auto set_counting = reinterpret_cast<int (*)(std::uint32_t)>(::dlsym(kp_, "kpc_set_counting"));
        auto set_thread = reinterpret_cast<int (*)(std::uint32_t)>(::dlsym(kp_, "kpc_set_thread_counting"));
        get_thread_ = reinterpret_cast<int (*)(int, unsigned, void*)>(::dlsym(kp_, "kpc_get_thread_counters"));
        if (!set_counting || !set_thread || !get_thread_) return {{}, unsupported("kpc API")};
        for (auto e : evs) if (e != perf_event::cycles && e != perf_event::instructions) return {{}, unsupported(std::string("kperf configurable event ") + perf_event_name(e))};
        if (set_counting(1) || set_thread(1)) return {{}, {1, "kpc_set_counting failed (needs root)"}};
        backend_ = "kperf"; return {};
#elif defined(FORGE_OS_WINDOWS)
        for (auto e : evs) if (e != perf_event::cycles) return {{}, unsupported(std::string("windows perf event ") + perf_event_name(e))};
        backend_ = "QueryThreadCycleTime"; return {};
#else
        for (auto e : evs) if (e != perf_event::cycles) return {{}, unsupported(std::string("perf event ") + perf_event_name(e))};
        backend_ = "cycle_counter"; return {};
#endif
    }
    const char* backend() const { return backend_; }
    void start() {
#if defined(__linux__)
        if (!fds_.empty()) { ::ioctl(fds_[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP); ::ioctl(fds_[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP); return; }
#endif
        base_ = snapshot_();
    }
    std::vector<reading> stop() {
        std::vector<reading> r; for (auto e : evs_) r.push_back({e, 0, false});
#if defined(__linux__)
        if (!fds_.empty()) {
            ::ioctl(fds_[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
            std::vector<std::uint64_t> buf(1 + fds_.size());
            if (::read(fds_[0], buf.data(), buf.size() * 8) > 0) for (std::size_t i = 0; i < r.size() && i < buf[0]; ++i) { r[i].value = buf[1 + i]; r[i].valid = true; }
            return r;
        }
        if (std::string(backend_) != "cycle_counter") return r;
#endif
        {
        auto now = snapshot_();
        for (std::size_t i = 0; i < r.size(); ++i) { std::size_t k = r[i].event == perf_event::instructions ? 1 : 0; if (k < now.size() && k < base_.size()) { r[i].value = now[k] - base_[k]; r[i].valid = true; } }
        }
        return r;
    }
private:
    void close_() {
#if defined(__linux__)
        for (int f : fds_) ::close(f); fds_.clear();
#endif
    }
    std::vector<std::uint64_t> snapshot_() {
#if defined(FORGE_OS_MACOS)
        std::uint64_t c[32] = {}; if (get_thread_ && get_thread_(0, 32, c) == 0) return {c[0], c[1]}; return {};
#elif defined(FORGE_OS_WINDOWS)
        ULONG64 c = 0; QueryThreadCycleTime(GetCurrentThread(), &c); return {c};
#else
        return {cycle_counter()};
#endif
    }
    std::vector<std::uint64_t> base_;
#if defined(__linux__)
    std::vector<int> fds_;
#endif
#if defined(FORGE_OS_MACOS)
    void* kp_ = nullptr; int (*get_thread_)(int, unsigned, void*) = nullptr;
#endif
    std::vector<perf_event> evs_; const char* backend_ = "none";
};
} // namespace forge::os
