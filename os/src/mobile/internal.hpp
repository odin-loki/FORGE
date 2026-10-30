#pragma once
// Internals shared by the forge_mobile backends.
#include <forge/os/mobile.hpp>
#include <forge/core/memory.hpp>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory_resource>
#include <mutex>
#include <thread>

namespace forge::os::mobile::detail {

struct item { bool touch = false; sensor_event s; touch_event t; };

// Bounded MPSC queue. Storage comes from forge::mem (host budget, tag "mobile"); when the budget or
// the capacity is exhausted the oldest event is dropped and counted.
std::pmr::memory_resource* mobile_resource();
struct queue {
    explicit queue(std::size_t cap = 4096) : cap(cap), q(mobile_resource()) {}
    std::mutex m; std::condition_variable cv; std::size_t cap; std::pmr::deque<item> q;
    std::atomic<std::uint64_t> dropped{0}; std::atomic<bool> finished{false};
    void push(const item& it) {
        { std::lock_guard g(m);
          try { if (q.size() >= cap) { q.pop_front(); ++dropped; } q.push_back(it); }
          catch (...) { ++dropped; } }
        cv.notify_all();
    }
    void finish() { finished = true; cv.notify_all(); }
    template <class F> void drain(int timeout_ms, std::size_t max, F&& f) {
        std::unique_lock g(m);
        auto ready = [&] { return !q.empty() || finished.load(); };
        if (timeout_ms < 0) cv.wait(g, ready);
        else if (timeout_ms > 0) cv.wait_for(g, std::chrono::milliseconds(timeout_ms), ready);
        while (!q.empty() && max--) { f(q.front()); q.pop_front(); }
    }
};

// Process-wide routing for feed_touch/feed_sensor and report_* (app glue -> open streams).
struct bridge {
    static bridge& get();
    std::mutex m;
    std::vector<std::weak_ptr<queue>> touch;
    std::vector<std::pair<sensor_kind, std::weak_ptr<queue>>> sensor;
    std::optional<display_info> disp;
    std::array<std::optional<permission_state>, 5> perms{};
    action_hooks hooks;
    void add_touch(const std::shared_ptr<queue>& q) { std::lock_guard g(m); touch.push_back(q); }
    void add_sensor(sensor_kind k, const std::shared_ptr<queue>& q) { std::lock_guard g(m); sensor.emplace_back(k, q); }
};

struct backend {
    virtual ~backend() = default;
    virtual std::string name() const = 0;
    virtual result<std::vector<sensor_info>> sensors() { return {{}, unsupported("mobile sensors")}; }
    virtual result<sensor_stream> open_sensor(sensor_kind k, double) { return {{}, unsupported(std::string("sensor ") + sensor_name(k))}; }
    virtual result<touch_stream> open_touch() { return {{}, unsupported("touch input")}; }
    virtual status vibrate(int ms, double amp);
    virtual result<display_info> display();
    virtual result<orientation> screen_orientation() { auto d = display(); if (!d) return {{}, d.err}; return {d.value.orient, {}}; }
    virtual result<std::vector<camera_info>> cameras() { return {{}, unsupported("camera enumeration")}; }
    virtual result<permission_state> permission_status(permission p);
    virtual status request_permission(permission p);
    virtual status notify(std::string_view title, std::string_view body);
    virtual std::vector<sim_record> sim_log() const { return {}; }

    // Stream construction (backends are friends of the stream types through this base).
    static sensor_stream make_sensor(sensor_kind k, std::shared_ptr<queue> q, std::shared_ptr<void> keep) {
        sensor_stream s; s.kind_ = k; s.q_ = std::move(q); s.keep_ = std::move(keep); return s;
    }
    static touch_stream make_touch(std::shared_ptr<queue> q, std::shared_ptr<void> keep) {
        touch_stream s; s.q_ = std::move(q); s.keep_ = std::move(keep); return s;
    }
};

std::shared_ptr<backend> make_null();
std::shared_ptr<backend> make_sim(std::string_view json, bool realtime, error& err);
#if defined(__linux__) && !defined(__ANDROID__)
std::shared_ptr<backend> make_linux(std::string sysfs, std::string input);
#endif
#if defined(__ANDROID__)
std::shared_ptr<backend> make_android();
#endif
#if defined(__APPLE__)
std::shared_ptr<backend> make_apple();   // iOS/iPadOS: CoreMotion etc.; macOS: returns null backend
#endif
// Background thread owner: stop flag + interruptible sleep; joins on destruction (RAII, no detached threads).
struct worker {
    std::atomic<bool> stop{false}; std::mutex m; std::condition_variable cv; std::thread t;
    template <class F> void start(F&& f) { t = std::thread(std::forward<F>(f)); }
    bool sleep_ns(std::uint64_t ns) {   // false when asked to stop
        std::unique_lock g(m); return !cv.wait_for(g, std::chrono::nanoseconds(ns), [&] { return stop.load(); });
    }
    ~worker() { { std::lock_guard g(m); stop = true; } cv.notify_all(); if (t.joinable()) t.join(); }
};
inline std::uint64_t now_ns() { return monotonic_ns(); }

} // namespace forge::os::mobile::detail
