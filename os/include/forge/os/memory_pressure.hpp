#pragma once
// Memory-pressure notifications, one subscription API:
//   Linux: PSI triggers on /proc/pressure/memory (poll POLLPRI); Android: the same where readable, plus the
//   onTrimMemory hook (call memory_pressure::on_trim_memory(level) from ComponentCallbacks2 via JNI);
//   iOS/macOS: DISPATCH_SOURCE_TYPE_MEMORYPRESSURE (also what UIApplicationDidReceiveMemoryWarning is built on;
//   app code may forward warnings with notify());  Windows: CreateMemoryResourceNotification;
//   everywhere else (FreeBSD, PSI-less kernels): a 1s poll of available memory.
#include <forge/os/detail.hpp>

namespace forge::os {
enum class pressure_level { normal = 0, moderate = 1, critical = 2 };
inline const char* pressure_name(pressure_level l) { return l == pressure_level::critical ? "critical" : l == pressure_level::moderate ? "moderate" : "normal"; }

class memory_pressure {
public:
    using callback = std::function<void(pressure_level)>;
    static memory_pressure& instance() { static memory_pressure m; return m; }
    // Subscribe; the monitor starts on first subscription. Callbacks run on the monitor thread (or the hook's caller).
    int subscribe(callback cb) {
        std::lock_guard<std::mutex> g(m_); int id = ++next_; subs_[id] = std::move(cb); if (!running_) start_locked_(); return id;
    }
    void unsubscribe(int id) { std::lock_guard<std::mutex> g(m_); subs_.erase(id); }
    // Deliver a level from anywhere (tests, platform glue, iOS app delegate).
    void notify(pressure_level l) {
        std::vector<callback> cbs; { std::lock_guard<std::mutex> g(m_); last_ = l; ++events_; for (auto& kv : subs_) cbs.push_back(kv.second); }
        for (auto& c : cbs) c(l);
    }
    // Android ComponentCallbacks2.onTrimMemory levels: RUNNING_MODERATE=5, RUNNING_LOW=10, RUNNING_CRITICAL=15,
    // UI_HIDDEN=20, BACKGROUND=40, MODERATE=60, COMPLETE=80.
    static void on_trim_memory(int level) {
        instance().notify(level >= 60 || level == 15 ? pressure_level::critical : level >= 5 && level != 20 ? pressure_level::moderate : pressure_level::normal);
    }
    pressure_level last() const { std::lock_guard<std::mutex> g(m_); return last_; }
    std::uint64_t events() const { std::lock_guard<std::mutex> g(m_); return events_; }
    std::string source() const { std::lock_guard<std::mutex> g(m_); return source_; }
    // Current level by sampling (independent of the monitor).
    static pressure_level sample() {
        std::uint64_t tot = total_memory(), av = available_memory(); if (!tot) return pressure_level::normal;
        double f = double(av) / double(tot); return f < 0.05 ? pressure_level::critical : f < 0.10 ? pressure_level::moderate : pressure_level::normal;
    }
    void stop() {
        { std::lock_guard<std::mutex> g(m_); if (!running_) return; stop_ = true; }
#if defined(FORGE_OS_WINDOWS)
        if (stop_ev_) SetEvent(stop_ev_);
#elif defined(FORGE_OS_POSIX)
        if (wake_[1] >= 0) { char c = 1; (void)!::write(wake_[1], &c, 1); }
#endif
        if (th_.joinable()) th_.join();
        std::lock_guard<std::mutex> g(m_); running_ = false; stop_ = false;
    }
    ~memory_pressure() { stop(); }
private:
    memory_pressure() = default;
    void start_locked_() {
        running_ = true;
#if defined(__APPLE__)
        if (start_dispatch_()) return;
#endif
#if defined(FORGE_OS_WINDOWS)
        stop_ev_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
#elif defined(FORGE_OS_POSIX)
        if (wake_[0] < 0 && ::pipe(wake_)) wake_[0] = wake_[1] = -1;
#endif
        th_ = std::thread([this] { run_(); });
    }
    bool stopping_() { std::lock_guard<std::mutex> g(m_); return stop_; }
    void set_source_(const char* s) { std::lock_guard<std::mutex> g(m_); source_ = s; }
    void run_() {
#if defined(__linux__)
        int some = ::open("/proc/pressure/memory", O_RDWR | O_NONBLOCK | O_CLOEXEC), full = ::open("/proc/pressure/memory", O_RDWR | O_NONBLOCK | O_CLOEXEC);
        // Unprivileged PSI triggers need a 2s-multiple window.
        if (some >= 0 && full >= 0 && ::write(some, "some 150000 2000000", 20) > 0 && ::write(full, "full 100000 2000000", 20) > 0) {
            set_source_("psi");
            pollfd f[3] = {{some, POLLPRI, 0}, {full, POLLPRI, 0}, {wake_[0], POLLIN, 0}};
            while (!stopping_()) {
                int n = ::poll(f, 3, 2000); if (n < 0 && errno != EINTR) break;
                if (f[2].revents) break;
                if (f[1].revents & POLLPRI) notify(pressure_level::critical);
                else if (f[0].revents & POLLPRI) notify(pressure_level::moderate);
                else if (n == 0 && last() != pressure_level::normal && sample() == pressure_level::normal) notify(pressure_level::normal);
                if ((f[0].revents | f[1].revents) & (POLLERR | POLLNVAL)) break;
            }
            ::close(some); ::close(full); return;
        }
        if (some >= 0) ::close(some); if (full >= 0) ::close(full);
#elif defined(FORGE_OS_WINDOWS)
        HANDLE lo = CreateMemoryResourceNotification(LowMemoryResourceNotification), hi = CreateMemoryResourceNotification(HighMemoryResourceNotification);
        if (lo && hi) {
            set_source_("memory_resource_notification");
            bool low = false;
            while (!stopping_()) {
                HANDLE hs[2] = {stop_ev_, lo};
                DWORD r = WaitForMultipleObjects(2, hs, FALSE, low ? 1000 : INFINITE);
                if (r == WAIT_OBJECT_0) break;
                BOOL st = FALSE; QueryMemoryResourceNotification(lo, &st);
                if (st && !low) { low = true; notify(sample() == pressure_level::critical ? pressure_level::critical : pressure_level::moderate); }
                BOOL h = FALSE; QueryMemoryResourceNotification(hi, &h);
                if (low && !st && h) { low = false; notify(pressure_level::normal); }
            }
            CloseHandle(lo); CloseHandle(hi); return;
        }
        if (lo) CloseHandle(lo); if (hi) CloseHandle(hi);
#endif
        set_source_("poll");
        pressure_level cur = pressure_level::normal;
        while (!stopping_()) {
            auto l = sample(); if (l != cur) { cur = l; notify(l); }
#if defined(FORGE_OS_WINDOWS)
            if (WaitForSingleObject(stop_ev_, 1000) == WAIT_OBJECT_0) break;
#else
            pollfd f{wake_[0], POLLIN, 0}; if (::poll(&f, 1, 1000) > 0) break;
#endif
        }
    }
#if defined(__APPLE__)
    bool start_dispatch_() {
        using src_t = void*;
        auto create = reinterpret_cast<src_t (*)(const void*, std::uintptr_t, std::uintptr_t, void*)>(::dlsym(RTLD_DEFAULT, "dispatch_source_create"));
        auto type = ::dlsym(RTLD_DEFAULT, "_dispatch_source_type_memorypressure");
        auto gq = reinterpret_cast<void* (*)(long, std::uintptr_t)>(::dlsym(RTLD_DEFAULT, "dispatch_get_global_queue"));
        auto seth = reinterpret_cast<void (*)(src_t, void (*)(void*))>(::dlsym(RTLD_DEFAULT, "dispatch_source_set_event_handler_f"));
        auto ctx = reinterpret_cast<void (*)(void*, void*)>(::dlsym(RTLD_DEFAULT, "dispatch_set_context"));
        auto res = reinterpret_cast<void (*)(void*)>(::dlsym(RTLD_DEFAULT, "dispatch_resume"));
        data_ = reinterpret_cast<std::uintptr_t (*)(src_t)>(::dlsym(RTLD_DEFAULT, "dispatch_source_get_data"));
        if (!create || !type || !gq || !seth || !ctx || !res || !data_) return false;
        src_ = create(type, 0, 0x1 | 0x2 | 0x4 /*NORMAL|WARN|CRITICAL*/, gq(0, 0));
        if (!src_) return false;
        ctx(src_, this);
        seth(src_, [](void* c) {
            auto* self = static_cast<memory_pressure*>(c); auto d = self->data_(self->src_);
            self->notify(d & 0x4 ? pressure_level::critical : d & 0x2 ? pressure_level::moderate : pressure_level::normal);
        });
        res(src_); source_ = "dispatch_source"; return true;
    }
    void* src_ = nullptr; std::uintptr_t (*data_)(void*) = nullptr;
#endif
    mutable std::mutex m_; std::map<int, callback> subs_; int next_ = 0; bool running_ = false, stop_ = false;
    pressure_level last_ = pressure_level::normal; std::uint64_t events_ = 0; std::string source_ = "none"; std::thread th_;
#if defined(FORGE_OS_WINDOWS)
    HANDLE stop_ev_ = nullptr;
#elif defined(FORGE_OS_POSIX)
    int wake_[2] = {-1, -1};
#endif
};
} // namespace forge::os
