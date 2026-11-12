#pragma once
// prof: scoped zones, counters, gauges, allocation tracking and GPU zones, recorded per thread and exported
// as Chrome trace JSON, Perfetto-compatible JSON or a plain summary table.
//
//   FORGE_ZONE("decode");                       scoped zone (name: a string that outlives the recorder's data)
//   FORGE_COUNTER("tokens", n);                 adds n to a named running total
//   FORGE_GAUGE("queue_depth", v);              sets a named value (last, min, max are kept)
//   FORGE_INSTANT("frame");                     a point event
//   prof::note_alloc("kv", p, bytes); prof::note_free("kv", p, bytes);     allocation hooks, per tag
//   prof::gpu_zone("matmul", device, t0, t1);   a finished GPU zone from device timestamps (backends call it)
//   auto& r = prof::recorder::get(); r.on = true; ... r.on = false;
//   r.chrome_json(); r.perfetto_json(); r.summary_table(); r.stats();
//
// Cost. Recording is off until recorder::on is set. While off, every macro and hook costs one relaxed atomic
// load and records nothing (counters and gauges do not accumulate either). With FORGE_PROF_DISABLE defined
// before this header is included, the macros expand to nothing at all; the functions remain callable.
// While on, an event takes two clock reads and one uncontended lock of the calling thread's own buffer.
//
// Storage. Each thread writes to its own ring buffer (recorder::set_capacity events, default 16384, allocated
// on the thread's first event); when it is full the oldest events are overwritten and counted in dropped().
// stats() and the exports see what the rings still hold.
//
// FORGE_PROF_TRACY forwards zones, counters, gauges and allocation hooks to Tracy's C API (<tracy/TracyC.h>,
// or the header named by FORGE_PROF_TRACY_INCLUDE) in addition to the recorder. That path is compiled in the
// tests against a stub header only; it has not been run against a real Tracy client or server.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if defined(FORGE_PROF_TRACY)
#ifndef FORGE_PROF_TRACY_INCLUDE
#define FORGE_PROF_TRACY_INCLUDE <tracy/TracyC.h>
#endif
#include FORGE_PROF_TRACY_INCLUDE
#endif

namespace forge::prof {
enum class kind : std::uint8_t { zone = 0, counter, gauge, instant, gpu_zone, alloc, release, hw_zone };
// zone / gpu_zone / hw_zone: [t0_ns, t0_ns + dur_ns). counter: value = the new total. gauge: value. alloc /
// release: value = live bytes of the tag afterwards, aux[0] = address, aux[1] = size. gpu_zone: tid = device.
// hw_zone: aux = cycles, instructions, cache misses, branch misses (0 where unavailable; see prof/hw.hpp).
struct event { const char* name; std::uint64_t t0_ns, dur_ns; std::uint32_t tid; kind k = kind::zone; double value = 0; std::uint64_t aux[4] = {}; };
struct zone_stat { std::string name; std::uint64_t count = 0, total_ns = 0, min_ns = 0, max_ns = 0, p50_ns = 0, p95_ns = 0, p99_ns = 0; double mean_ns = 0; };
struct counter_value { std::string name; std::int64_t total = 0; };
struct gauge_value { std::string name; double last = 0, min = 0, max = 0; std::uint64_t samples = 0; };
struct alloc_stat { std::string tag; std::uint64_t live_bytes = 0, peak_bytes = 0, allocs = 0, frees = 0; };

struct recorder {
    std::atomic<bool> on{false};
    static recorder& get() { static recorder r; return r; }
    static std::uint64_t now() { return std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); }
    static std::uint32_t tid() { static std::atomic<std::uint32_t> n{0}; thread_local std::uint32_t t = n++; return t; }

    // Append to the calling thread's ring (whether or not `on` is set: the callers check it).
    void add(const event& e) {
        ring& r = mine_();
        std::lock_guard l(r.mu);
        if (r.buf.size() < r.cap) r.buf.push_back(e); else { r.buf[r.head % r.cap] = e; ++r.overwritten; }
        ++r.head;
    }
    // Everything the rings hold, ordered by end time (so a nested zone comes before the zone around it).
    std::vector<event> events() const {
        std::vector<event> out;
        {
            std::lock_guard l(mu_);
            for (auto& r : rings_) { std::lock_guard l2(r->mu); out.insert(out.end(), r->buf.begin(), r->buf.end()); }
        }
        std::stable_sort(out.begin(), out.end(), [](const event& a, const event& b) { return a.t0_ns + a.dur_ns < b.t0_ns + b.dur_ns; });
        return out;
    }
    // Empties every ring and resets counters, gauges and allocation statistics.
    void clear() {
        std::lock_guard l(mu_);
        for (auto& r : rings_) { std::lock_guard l2(r->mu); r->buf.clear(); r->head = 0; r->overwritten = 0; r->cap = cap_; }
        for (auto& [k, c] : counters_) c->total.store(0);
        for (auto& [k, g] : gauges_) { std::lock_guard l2(g->mu); g->v = gauge_value{}; }
        for (auto& [k, a] : allocs_) { a->live.store(0); a->peak.store(0); a->allocs.store(0); a->frees.store(0); }
    }
    // Events kept per thread; applies to rings created afterwards and to existing ones at the next clear().
    void set_capacity(std::size_t events_per_thread) { std::lock_guard l(mu_); cap_ = std::max<std::size_t>(1, events_per_thread); }
    std::uint64_t dropped() const { std::lock_guard l(mu_); std::uint64_t d = 0; for (auto& r : rings_) { std::lock_guard l2(r->mu); d += r->overwritten; } return d; }
    void name_thread(const std::string& name) { ring& r = mine_(); std::lock_guard l(r.mu); r.name = name; }

    // ---- counters, gauges, allocation tags (registered by name on first use) ----
    struct counter_cell { std::atomic<std::int64_t> total{0}; std::string name; };
    struct gauge_cell { std::mutex mu; gauge_value v; };
    struct alloc_cell { std::atomic<std::uint64_t> live{0}, peak{0}, allocs{0}, frees{0}; std::string tag; };
    counter_cell& counter_named(const std::string& name) { std::lock_guard l(mu_); auto& p = counters_[name]; if (!p) { p = std::make_unique<counter_cell>(); p->name = name; } return *p; }
    gauge_cell& gauge_named(const std::string& name) { std::lock_guard l(mu_); auto& p = gauges_[name]; if (!p) { p = std::make_unique<gauge_cell>(); p->v.name = name; } return *p; }
    alloc_cell& alloc_named(const std::string& tag) { std::lock_guard l(mu_); auto& p = allocs_[tag]; if (!p) { p = std::make_unique<alloc_cell>(); p->tag = tag; } return *p; }
    std::vector<counter_value> counters() const { std::lock_guard l(mu_); std::vector<counter_value> v; for (auto& [k, c] : counters_) v.push_back({k, c->total.load()}); return v; }
    std::vector<gauge_value> gauges() const { std::lock_guard l(mu_); std::vector<gauge_value> v; for (auto& [k, g] : gauges_) { std::lock_guard l2(g->mu); v.push_back(g->v); } return v; }
    std::vector<alloc_stat> allocations() const { std::lock_guard l(mu_); std::vector<alloc_stat> v; for (auto& [k, a] : allocs_) v.push_back({k, a->live.load(), a->peak.load(), a->allocs.load(), a->frees.load()}); return v; }

    // GPU clock: ns = cpu_ns_at_sync + (ticks - gpu_ticks_at_sync) * ns_per_tick. Uncalibrated devices are
    // treated as already reporting nanoseconds on the recorder's clock.
    void gpu_calibrate(std::uint32_t device, std::uint64_t gpu_ticks_at_sync, std::uint64_t cpu_ns_at_sync, double ns_per_tick) { std::lock_guard l(mu_); gpu_[device] = {gpu_ticks_at_sync, cpu_ns_at_sync, ns_per_tick}; }
    std::uint64_t gpu_to_ns(std::uint32_t device, std::uint64_t ticks) const {
        std::lock_guard l(mu_); auto it = gpu_.find(device);
        if (it == gpu_.end()) return ticks;
        const double d = (double(ticks) - double(it->second.ticks)) * it->second.ns_per_tick + double(it->second.cpu_ns);
        return d <= 0 ? 0 : std::uint64_t(d + 0.5);
    }

    // ---- aggregated statistics per zone name (zones, GPU zones and hardware zones the rings still hold) ----
    // Percentiles are nearest-rank: the smallest duration with at least p% of the samples at or below it.
    std::vector<zone_stat> stats() const {
        std::map<std::string, std::vector<std::uint64_t>> by;
        for (auto& e : events()) if (e.k == kind::zone || e.k == kind::gpu_zone || e.k == kind::hw_zone) by[e.name].push_back(e.dur_ns);
        std::vector<zone_stat> out;
        for (auto& [name, d] : by) {
            std::sort(d.begin(), d.end());
            zone_stat s; s.name = name; s.count = d.size(); s.min_ns = d.front(); s.max_ns = d.back();
            for (auto v : d) s.total_ns += v;
            s.mean_ns = double(s.total_ns) / double(d.size());
            auto pct = [&](double p) { std::size_t rank = std::size_t(std::ceil(p * double(d.size()))); return d[std::min(d.size() - 1, rank ? rank - 1 : 0)]; };
            s.p50_ns = pct(0.50); s.p95_ns = pct(0.95); s.p99_ns = pct(0.99);
            out.push_back(std::move(s));
        }
        std::sort(out.begin(), out.end(), [](const zone_stat& a, const zone_stat& b) { return a.total_ns > b.total_ns; });
        return out;
    }
    // A fixed-width text table: zones by total time, then counters, gauges and allocation tags.
    std::string summary_table() const {
        std::string s; char line[512];
        std::snprintf(line, sizeof line, "%-32s %10s %12s %12s %12s %12s %12s %12s\n", "zone", "count", "total ms", "mean us", "p50 us", "p95 us", "p99 us", "max us"); s += line;
        for (auto& z : stats()) {
            std::snprintf(line, sizeof line, "%-32.32s %10llu %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n", z.name.c_str(), static_cast<unsigned long long>(z.count), double(z.total_ns) / 1e6,
                          z.mean_ns / 1e3, double(z.p50_ns) / 1e3, double(z.p95_ns) / 1e3, double(z.p99_ns) / 1e3, double(z.max_ns) / 1e3);
            s += line;
        }
        auto cs = counters(); auto gs = gauges(); auto as = allocations();
        if (!cs.empty()) { s += "\ncounter                               total\n"; for (auto& c : cs) { std::snprintf(line, sizeof line, "%-32.32s %10lld\n", c.name.c_str(), static_cast<long long>(c.total)); s += line; } }
        if (!gs.empty()) { std::snprintf(line, sizeof line, "\n%-32s %12s %12s %12s %10s\n", "gauge", "last", "min", "max", "samples"); s += line;
            for (auto& g : gs) { std::snprintf(line, sizeof line, "%-32.32s %12.6g %12.6g %12.6g %10llu\n", g.name.c_str(), g.last, g.min, g.max, static_cast<unsigned long long>(g.samples)); s += line; } }
        if (!as.empty()) { std::snprintf(line, sizeof line, "\n%-32s %14s %14s %10s %10s\n", "memory tag", "live bytes", "peak bytes", "allocs", "frees"); s += line;
            for (auto& a : as) { std::snprintf(line, sizeof line, "%-32.32s %14llu %14llu %10llu %10llu\n", a.tag.c_str(), static_cast<unsigned long long>(a.live_bytes), static_cast<unsigned long long>(a.peak_bytes),
                                               static_cast<unsigned long long>(a.allocs), static_cast<unsigned long long>(a.frees)); s += line; } }
        if (auto d = dropped()) { std::snprintf(line, sizeof line, "\n%llu event(s) overwritten (ring full)\n", static_cast<unsigned long long>(d)); s += line; }
        return s;
    }

    // ---- trace exports ----
    // Chrome trace-event JSON (chrome://tracing): one entry per recorded event, in events() order. Zones are
    // complete events ("X"), counters / gauges / allocation tags counter tracks ("C"), instants "i"; GPU zones
    // sit in process 2 with the device as thread.
    std::string chrome_json() const { return json_(false); }
    // The same events in the form Perfetto's JSON importer (ui.perfetto.dev, trace_processor) expects: sorted
    // by timestamp, with process and thread name metadata. Written to the documented legacy-JSON format.
    std::string perfetto_json() const { return json_(true); }

private:
    struct ring { std::mutex mu; std::vector<event> buf; std::size_t cap = 16384; std::uint64_t head = 0, overwritten = 0; std::uint32_t tid = 0; std::string name; };
    struct gpu_clock { std::uint64_t ticks = 0, cpu_ns = 0; double ns_per_tick = 1; };
    ring& mine_() {
        thread_local std::shared_ptr<ring> r;
        if (!r) { r = std::make_shared<ring>(); r->tid = tid(); std::lock_guard l(mu_); r->cap = cap_; rings_.push_back(r); }
        return *r;
    }
    static void escape_(std::string& s, const char* p) {
        for (; *p; ++p) {
            const unsigned char c = static_cast<unsigned char>(*p);
            if (c == '"' || c == '\\') { s += '\\'; s += char(c); }
            else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); s += b; }
            else s += char(c);
        }
    }
    std::string json_(bool perfetto) const {
        auto ev = events();
        if (perfetto) std::stable_sort(ev.begin(), ev.end(), [](const event& a, const event& b) { return a.t0_ns < b.t0_ns; });
        std::string s = "{\"traceEvents\":["; char buf[256]; bool first = true;
        auto sep = [&] { if (!first) s += ','; first = false; };
        if (perfetto) {
            bool gpu = false; for (auto& e : ev) gpu |= e.k == kind::gpu_zone;
            sep(); s += "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,\"args\":{\"name\":\"forge\"}}";
            if (gpu) { sep(); s += "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":2,\"tid\":0,\"args\":{\"name\":\"GPU\"}}"; }
            std::lock_guard l(mu_);
            for (auto& r : rings_) {
                std::lock_guard l2(r->mu);
                sep(); std::snprintf(buf, sizeof buf, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":%u,\"args\":{\"name\":\"", r->tid); s += buf;
                if (r->name.empty()) { std::snprintf(buf, sizeof buf, "thread %u", r->tid); s += buf; } else escape_(s, r->name.c_str());
                s += "\"}}";
            }
        }
        for (auto& e : ev) {
            sep(); s += "{\"name\":\""; escape_(s, e.name); s += "\",";
            const double ts = double(e.t0_ns) / 1e3, dur = double(e.dur_ns) / 1e3;
            switch (e.k) {
            case kind::zone:
                std::snprintf(buf, sizeof buf, "\"ph\":\"X\",\"pid\":1,\"tid\":%u,\"ts\":%.3f,\"dur\":%.3f}", e.tid, ts, dur); break;
            case kind::hw_zone:
                std::snprintf(buf, sizeof buf, "\"ph\":\"X\",\"pid\":1,\"tid\":%u,\"ts\":%.3f,\"dur\":%.3f,\"args\":{\"cycles\":%llu,\"instructions\":%llu,\"cache_misses\":%llu,\"branch_misses\":%llu}}", e.tid, ts, dur,
                              static_cast<unsigned long long>(e.aux[0]), static_cast<unsigned long long>(e.aux[1]), static_cast<unsigned long long>(e.aux[2]), static_cast<unsigned long long>(e.aux[3])); break;
            case kind::gpu_zone:
                std::snprintf(buf, sizeof buf, "\"cat\":\"gpu\",\"ph\":\"X\",\"pid\":2,\"tid\":%u,\"ts\":%.3f,\"dur\":%.3f}", e.tid, ts, dur); break;
            case kind::instant:
                std::snprintf(buf, sizeof buf, "\"ph\":\"i\",\"s\":\"t\",\"pid\":1,\"tid\":%u,\"ts\":%.3f}", e.tid, ts); break;
            default:   // counter, gauge, alloc, release: a counter track named after the counter / tag
                std::snprintf(buf, sizeof buf, "\"ph\":\"C\",\"pid\":1,\"tid\":0,\"ts\":%.3f,\"args\":{\"value\":%.17g}}", ts, std::isfinite(e.value) ? e.value : 0.0); break;
            }
            s += buf;
        }
        return s + "],\"displayTimeUnit\":\"ns\"}";
    }
    mutable std::mutex mu_;
    std::vector<std::shared_ptr<ring>> rings_;
    std::size_t cap_ = 16384;
    std::map<std::string, std::unique_ptr<counter_cell>> counters_;
    std::map<std::string, std::unique_ptr<gauge_cell>> gauges_;
    std::map<std::string, std::unique_ptr<alloc_cell>> allocs_;
    std::map<std::uint32_t, gpu_clock> gpu_;
};

inline bool enabled() { return recorder::get().on.load(std::memory_order_relaxed); }

struct zone {
    const char* name; std::uint64_t t0 = 0; bool live;
#if defined(FORGE_PROF_TRACY)
    TracyCZoneCtx tracy_;
#endif
    explicit zone(const char* n) : name(n), live(enabled()) {
        if (live) t0 = recorder::now();
#if defined(FORGE_PROF_TRACY)
#if defined(FORGE_PROF_TRACY_SRCLOC_COLOR)   // Tracy 0.11 and later take a colour as the last argument
        tracy_ = ___tracy_emit_zone_begin_alloc(___tracy_alloc_srcloc_name(0, "", 0, "", 0, n, std::strlen(n), 0), 1);
#else
        tracy_ = ___tracy_emit_zone_begin_alloc(___tracy_alloc_srcloc_name(0, "", 0, "", 0, n, std::strlen(n)), 1);
#endif
#endif
    }
    ~zone() {
#if defined(FORGE_PROF_TRACY)
        ___tracy_emit_zone_end(tracy_);
#endif
        if (live) recorder::get().add({name, t0, recorder::now() - t0, recorder::tid()});
    }
    zone(const zone&) = delete; zone& operator=(const zone&) = delete;
};

// A named running total. Construct once (FORGE_COUNTER keeps a function-local static) and add from any thread.
class counter {
public:
    explicit counter(const char* name) : name_(name), cell_(&recorder::get().counter_named(name)) {}
    void add(std::int64_t delta = 1) {
        if (!enabled()) return;
        const std::int64_t total = cell_->total.fetch_add(delta, std::memory_order_relaxed) + delta;
        recorder::get().add({name_, recorder::now(), 0, recorder::tid(), kind::counter, double(total)});
#if defined(FORGE_PROF_TRACY)
        ___tracy_emit_plot(name_, double(total));
#endif
    }
    std::int64_t total() const { return cell_->total.load(std::memory_order_relaxed); }
private:
    const char* name_; recorder::counter_cell* cell_;
};
// A named value that goes up and down.
class gauge {
public:
    explicit gauge(const char* name) : name_(name), cell_(&recorder::get().gauge_named(name)) {}
    void set(double v) {
        if (!enabled()) return;
        { std::lock_guard l(cell_->mu); auto& g = cell_->v; g.min = g.samples ? std::min(g.min, v) : v; g.max = g.samples ? std::max(g.max, v) : v; g.last = v; ++g.samples; }
        recorder::get().add({name_, recorder::now(), 0, recorder::tid(), kind::gauge, v});
#if defined(FORGE_PROF_TRACY)
        ___tracy_emit_plot(name_, v);
#endif
    }
private:
    const char* name_; recorder::gauge_cell* cell_;
};
inline void instant(const char* name) { if (enabled()) recorder::get().add({name, recorder::now(), 0, recorder::tid(), kind::instant}); }

// ---- allocation tracking hooks ----
// Allocators call these around their own bookkeeping; `tag` names the pool (a string literal or other
// long-lived string). note_free must be given the size that was passed to note_alloc.
namespace detail {
inline recorder::alloc_cell& alloc_cell_for(const char* tag) {
    // per-thread one-entry cache: hooks are usually called with the same literal over and over
    thread_local const char* last_tag = nullptr; thread_local recorder::alloc_cell* last = nullptr;
    if (last_tag != tag) { last = &recorder::get().alloc_named(tag); last_tag = tag; }
    return *last;
}
}
inline void note_alloc(const char* tag, const void* p, std::size_t bytes) {
#if defined(FORGE_PROF_TRACY)
    ___tracy_emit_memory_alloc_named(p, bytes, 0, tag);
#endif
    if (!enabled()) return;
    auto& c = detail::alloc_cell_for(tag);
    const std::uint64_t live = c.live.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    for (std::uint64_t pk = c.peak.load(std::memory_order_relaxed); live > pk && !c.peak.compare_exchange_weak(pk, live, std::memory_order_relaxed);) {}
    c.allocs.fetch_add(1, std::memory_order_relaxed);
    event e{tag, recorder::now(), 0, recorder::tid(), kind::alloc, double(live)}; e.aux[0] = std::uint64_t(reinterpret_cast<std::uintptr_t>(p)); e.aux[1] = bytes;
    recorder::get().add(e);
}
inline void note_free(const char* tag, const void* p, std::size_t bytes) {
#if defined(FORGE_PROF_TRACY)
    ___tracy_emit_memory_free_named(p, 0, tag);
#endif
    if (!enabled()) return;
    auto& c = detail::alloc_cell_for(tag);
    std::uint64_t live = c.live.load(std::memory_order_relaxed), next;
    do { next = live >= bytes ? live - bytes : 0; } while (!c.live.compare_exchange_weak(live, next, std::memory_order_relaxed));   // a free whose alloc predates recording must not wrap
    c.frees.fetch_add(1, std::memory_order_relaxed);
    event e{tag, recorder::now(), 0, recorder::tid(), kind::release, double(next)}; e.aux[0] = std::uint64_t(reinterpret_cast<std::uintptr_t>(p)); e.aux[1] = bytes;
    recorder::get().add(e);
}
// A std allocator that reports to the hooks under one tag: std::vector<float, prof::tracked_allocator<float>>
// v(prof::tracked_allocator<float>("weights")).
template <class T> struct tracked_allocator {
    using value_type = T;
    const char* tag;
    explicit tracked_allocator(const char* t = "untagged") noexcept : tag(t) {}
    template <class U> tracked_allocator(const tracked_allocator<U>& o) noexcept : tag(o.tag) {}
    T* allocate(std::size_t n) { T* p = std::allocator<T>().allocate(n); note_alloc(tag, p, n * sizeof(T)); return p; }
    void deallocate(T* p, std::size_t n) noexcept { note_free(tag, p, n * sizeof(T)); std::allocator<T>().deallocate(p, n); }
    template <class U> bool operator==(const tracked_allocator<U>& o) const noexcept { return tag == o.tag; }
};

// ---- GPU zones ----
// A backend that reads device timestamps reports each finished zone here, in device ticks (see
// recorder::gpu_calibrate for the tick-to-nanosecond mapping). Nothing is recorded while profiling is off.
// No FORGE GPU backend calls this yet; a backend without timestamp queries simply never does.
inline void gpu_zone(const char* name, std::uint32_t device, std::uint64_t begin_ticks, std::uint64_t end_ticks) {
    if (!enabled()) return;
    auto& r = recorder::get();
    const std::uint64_t t0 = r.gpu_to_ns(device, begin_ticks), t1 = r.gpu_to_ns(device, end_ticks);
    r.add({name, t0, t1 > t0 ? t1 - t0 : 0, device, kind::gpu_zone});
}
} // namespace forge::prof

#define FORGE_PROF_CAT2(a, b) a##b
#define FORGE_PROF_CAT(a, b) FORGE_PROF_CAT2(a, b)
#if defined(FORGE_PROF_DISABLE)
#define FORGE_ZONE(name) ((void)0)
#define FORGE_COUNTER(name, delta) ((void)0)
#define FORGE_GAUGE(name, value) ((void)0)
#define FORGE_INSTANT(name) ((void)0)
#else
#define FORGE_ZONE(name) ::forge::prof::zone FORGE_PROF_CAT(forge_zone_, __LINE__)(name)
#define FORGE_COUNTER(name, delta) do { static ::forge::prof::counter forge_prof_c_(name); forge_prof_c_.add(delta); } while (0)
#define FORGE_GAUGE(name, value) do { static ::forge::prof::gauge forge_prof_g_(name); forge_prof_g_.set(value); } while (0)
#define FORGE_INSTANT(name) ::forge::prof::instant(name)
#endif
