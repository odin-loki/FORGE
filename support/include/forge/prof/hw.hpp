#pragma once
// prof hardware-counter zones: a scoped zone that also reads the calling thread's hardware counters through
// os::perf_counters (cycles, instructions, cache misses, branch misses) and stores them with the event.
//
//   { prof::hw_zone z("attention"); work(); }        // recorded as kind::hw_zone when profiling is on
//   prof::hw_reading r = prof::hw_measure([] { work(); });   // or measured directly, profiling on or off
//
// What is available depends on the OS and its permissions (see os/perf.hpp): Linux perf_event gives all four
// when perf_event_paranoid allows it; otherwise only a cycle (time-stamp) counter may be left, and then
// `counters` is false and only `cycles` is meaningful. The counter group is opened once per thread and
// reused. Zones do not nest: an hw_zone opened inside another on the same thread records its time but
// zeros for the counters.
#include <forge/os/perf.hpp>
#include <forge/prof/prof.hpp>

namespace forge::prof {
struct hw_reading {
    std::uint64_t cycles = 0, instructions = 0, cache_misses = 0, branch_misses = 0, ns = 0;
    bool counters = false;   // false: no full counter group on this system (cycles may still be valid)
    bool cycles_valid = false;
    const char* backend = "none";
};
namespace detail {
struct hw_thread {
    os::perf_counters pc; bool full = false, cycles_only = false, tried = false; int depth = 0;
    void open() {
        if (tried) return;
        tried = true;
        if (pc.open({os::perf_event::cycles, os::perf_event::instructions, os::perf_event::cache_misses, os::perf_event::branch_misses})) full = true;
        else if (pc.open({os::perf_event::cycles})) cycles_only = true;
    }
    static hw_thread& get() { thread_local hw_thread t; return t; }
};
inline void hw_fill(hw_thread& t, hw_reading& r) {
    r.backend = t.pc.backend();
    for (auto& v : t.pc.stop()) {
        if (!v.valid) continue;
        switch (v.event) {
        case os::perf_event::cycles: r.cycles = v.value; r.cycles_valid = true; break;
        case os::perf_event::instructions: r.instructions = v.value; break;
        case os::perf_event::cache_misses: r.cache_misses = v.value; break;
        case os::perf_event::branch_misses: r.branch_misses = v.value; break;
        default: break;
        }
    }
    r.counters = t.full && r.cycles_valid;
}
} // namespace detail

// True when at least a cycle counter can be read on this thread.
inline bool hw_available() { auto& t = detail::hw_thread::get(); t.open(); return t.full || t.cycles_only; }

template <class F> hw_reading hw_measure(F&& f) {
    auto& t = detail::hw_thread::get(); t.open();
    hw_reading r; const bool mine = (t.full || t.cycles_only) && t.depth == 0;
    ++t.depth;
    const std::uint64_t t0 = recorder::now();
    if (mine) t.pc.start();
    try { f(); } catch (...) { if (mine) (void)t.pc.stop(); --t.depth; throw; }
    if (mine) detail::hw_fill(t, r);
    r.ns = recorder::now() - t0; --t.depth;
    return r;
}

class hw_zone {
public:
    explicit hw_zone(const char* name) : name_(name), live_(enabled()) {
        if (!live_) return;
        auto& t = detail::hw_thread::get(); t.open();
        mine_ = (t.full || t.cycles_only) && t.depth == 0; ++t.depth;
        t0_ = recorder::now();
        if (mine_) t.pc.start();
    }
    ~hw_zone() {
        if (!live_) return;
        auto& t = detail::hw_thread::get(); hw_reading r;
        if (mine_) detail::hw_fill(t, r);
        --t.depth;
        event e{name_, t0_, recorder::now() - t0_, recorder::tid(), kind::hw_zone};
        e.aux[0] = r.cycles; e.aux[1] = r.instructions; e.aux[2] = r.cache_misses; e.aux[3] = r.branch_misses;
        recorder::get().add(e);
    }
    hw_zone(const hw_zone&) = delete; hw_zone& operator=(const hw_zone&) = delete;
private:
    const char* name_; bool live_, mine_ = false; std::uint64_t t0_ = 0;
};
} // namespace forge::prof

#if defined(FORGE_PROF_DISABLE)
#define FORGE_HW_ZONE(name) ((void)0)
#else
#define FORGE_HW_ZONE(name) ::forge::prof::hw_zone FORGE_PROF_CAT(forge_hw_zone_, __LINE__)(name)
#endif
