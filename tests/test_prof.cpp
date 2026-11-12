// prof: counters, gauges, per-thread rings, allocation hooks, GPU zones, statistics and the three exports.
#include "check.hpp"
#include <forge/json/json.hpp>
#include <forge/prof/hw.hpp>
#include <forge/prof/prof.hpp>
#include <set>
#include <string>
#include <thread>
#include <vector>
using namespace forge;

namespace {
struct recording {   // a clean recorder that is on for the scope
    prof::recorder& r = prof::recorder::get();
    recording() { r.on = false; r.set_capacity(16384); r.clear(); r.on = true; }
    ~recording() { r.on = false; }
};
std::size_t count_kind(const std::vector<prof::event>& ev, prof::kind k) { std::size_t n = 0; for (auto& e : ev) n += e.k == k; return n; }
}

TEST(prof_off_records_nothing) {
    auto& r = prof::recorder::get(); r.on = false; r.clear();
    { FORGE_ZONE("z"); FORGE_COUNTER("c_off", 3); FORGE_GAUGE("g_off", 1.5); FORGE_INSTANT("i"); }
    int x = 0; prof::note_alloc("t_off", &x, 100); prof::note_free("t_off", &x, 100); prof::gpu_zone("g", 0, 1, 2);
    { prof::hw_zone h("hw_off"); }
    CHECK(r.events().empty()); CHECK(r.dropped() == 0);
    for (auto& c : r.counters()) CHECK(c.total == 0);
    for (auto& a : r.allocations()) CHECK(a.allocs == 0 && a.live_bytes == 0);
}

TEST(prof_counters_gauges_instants) {
    recording rec; auto& r = rec.r;
    for (int i = 0; i < 5; ++i) FORGE_COUNTER("tokens", 3);
    prof::counter c("tokens"); c.add(); c.add(-6);                      // the same named total from another object
    for (double v : {2.0, -1.5, 7.25}) FORGE_GAUGE("queue", v);
    FORGE_INSTANT("frame");
    CHECK(c.total() == 10);
    auto cs = r.counters(); bool found = false; for (auto& cv : cs) if (cv.name == "tokens") { found = true; CHECK(cv.total == 10); }
    CHECK(found);
    found = false; for (auto& g : r.gauges()) if (g.name == "queue") { found = true; CHECK(g.last == 7.25 && g.min == -1.5 && g.max == 7.25 && g.samples == 3); }
    CHECK(found);
    auto ev = r.events();
    CHECK(count_kind(ev, prof::kind::counter) == 7); CHECK(count_kind(ev, prof::kind::gauge) == 3); CHECK(count_kind(ev, prof::kind::instant) == 1);
    double last_total = 0; for (auto& e : ev) if (e.k == prof::kind::counter) last_total = e.value;
    CHECK(last_total == 10);
    r.clear(); CHECK(c.total() == 0); CHECK(r.events().empty());
}

TEST(prof_stats_percentiles_exact) {
    // Synthetic durations 1..100 ns for "a" (nearest rank: p50 = 50, p95 = 95, p99 = 99) and one sample for "b".
    recording rec; auto& r = rec.r;
    for (std::uint64_t d = 100; d >= 1; --d) r.add({"a", 1000 * d, d, 0});
    r.add({"b", 5, 7, 0});
    r.add({"c", 10, 40, 1, prof::kind::gpu_zone}); r.add({"c", 60, 60, 1, prof::kind::gpu_zone});
    r.add({"not_a_zone", 1, 0, 0, prof::kind::counter, 3.0});
    auto st = r.stats();
    CHECK(st.size() == 3); CHECK(st[0].name == "a" && st[1].name == "c" && st[2].name == "b");   // by total time
    CHECK(st[0].count == 100 && st[0].total_ns == 5050 && st[0].min_ns == 1 && st[0].max_ns == 100);
    CHECK(st[0].mean_ns == 50.5); CHECK(st[0].p50_ns == 50 && st[0].p95_ns == 95 && st[0].p99_ns == 99);
    CHECK(st[2].count == 1 && st[2].p50_ns == 7 && st[2].p99_ns == 7 && st[2].mean_ns == 7);
    CHECK(st[1].count == 2 && st[1].p50_ns == 40 && st[1].p95_ns == 60);
    auto table = r.summary_table();
    CHECK(table.find("zone") == 0); CHECK(table.find("\na ") != std::string::npos); CHECK(table.find("p99 us") != std::string::npos);
    CHECK(table.find("0.050") != std::string::npos);   // p50 of "a" in microseconds
}

TEST(prof_ring_overwrites_oldest_per_thread) {
    auto& r = prof::recorder::get(); r.on = false; r.set_capacity(8); r.clear(); r.on = true;
    for (std::uint64_t i = 0; i < 20; ++i) r.add({"z", i * 10, 1, prof::recorder::tid()});
    r.on = false;
    auto ev = r.events();
    CHECK(ev.size() == 8); CHECK(r.dropped() == 12);
    std::set<std::uint64_t> t0; for (auto& e : ev) t0.insert(e.t0_ns);
    CHECK(*t0.begin() == 120 && *t0.rbegin() == 190);   // the newest eight survive
    r.set_capacity(16384); r.clear(); CHECK(r.dropped() == 0);
}

TEST(prof_threads_record_into_their_own_rings) {
    recording rec; auto& r = rec.r;
    const int threads = 4, per = 500;
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) ts.emplace_back([&, t] {
        prof::recorder::get().name_thread("worker " + std::to_string(t));
        for (int i = 0; i < per; ++i) { FORGE_ZONE("work"); FORGE_COUNTER("shared", 1); }
    });
    std::string mid = r.chrome_json();   // exporting while threads record must be safe (TSan)
    for (auto& t : ts) t.join();
    auto ev = r.events();
    CHECK(count_kind(ev, prof::kind::zone) == std::size_t(threads * per)); CHECK(count_kind(ev, prof::kind::counter) == std::size_t(threads * per));
    std::set<std::uint32_t> tids; for (auto& e : ev) if (e.k == prof::kind::zone) tids.insert(e.tid);
    CHECK(tids.size() == std::size_t(threads));
    bool ordered = true; for (std::size_t i = 1; i < ev.size(); ++i) ordered &= ev[i - 1].t0_ns + ev[i - 1].dur_ns <= ev[i].t0_ns + ev[i].dur_ns;
    CHECK(ordered);
    for (auto& c : r.counters()) if (c.name == "shared") CHECK(c.total == threads * per);
    auto st = r.stats(); CHECK(st.size() == 1 && st[0].count == std::uint64_t(threads * per) && st[0].p50_ns <= st[0].p95_ns && st[0].p95_ns <= st[0].p99_ns && st[0].p99_ns <= st[0].max_ns);
    CHECK(!mid.empty());
    auto pj = r.perfetto_json(); CHECK(pj.find("worker 2") != std::string::npos);
}

TEST(prof_allocation_hooks_and_tracked_allocator) {
    recording rec; auto& r = rec.r;
    int a = 0, b = 0;
    prof::note_alloc("pool", &a, 1000); prof::note_alloc("pool", &b, 500); prof::note_free("pool", &a, 1000); prof::note_alloc("pool", &a, 200);
    prof::note_free("other", &b, 64);   // a free with no recorded alloc must not wrap below zero
    {
        std::vector<float, prof::tracked_allocator<float>> v(prof::tracked_allocator<float>("vec"));
        v.resize(256);
        for (auto& s : r.allocations()) if (s.tag == "vec") CHECK(s.live_bytes == 256 * sizeof(float) && s.allocs == 1);
    }
    for (auto& s : r.allocations()) {
        if (s.tag == "pool") CHECK(s.live_bytes == 700 && s.peak_bytes == 1500 && s.allocs == 3 && s.frees == 1);
        if (s.tag == "other") CHECK(s.live_bytes == 0 && s.frees == 1);
        if (s.tag == "vec") CHECK(s.live_bytes == 0 && s.peak_bytes == 256 * sizeof(float) && s.frees == s.allocs);
    }
    auto ev = r.events(); bool saw = false;
    for (auto& e : ev) if (e.k == prof::kind::alloc && e.aux[1] == 500) { saw = true; CHECK(e.value == 1500 && e.aux[0] == std::uint64_t(reinterpret_cast<std::uintptr_t>(&b))); }
    CHECK(saw); CHECK(count_kind(ev, prof::kind::release) == 3);
    CHECK(r.summary_table().find("memory tag") != std::string::npos);
}

TEST(prof_gpu_zones_and_clock_calibration) {
    recording rec; auto& r = rec.r;
    prof::gpu_zone("raw", 3, 1000, 1600);                 // uncalibrated: ticks are nanoseconds
    r.gpu_calibrate(7, 500, 2'000'000, 2.5);              // tick 500 == 2 ms on the CPU clock, 2.5 ns per tick
    prof::gpu_zone("scaled", 7, 900, 1300);               // -> [2'001'000, 2'002'000)
    prof::gpu_zone("backwards", 7, 1300, 900);            // clamped to zero length
    CHECK(r.gpu_to_ns(7, 500) == 2'000'000); CHECK(r.gpu_to_ns(7, 100) == 1'999'000); CHECK(r.gpu_to_ns(9, 42) == 42);
    std::size_t seen = 0;
    for (auto& e : r.events()) {
        CHECK(e.k == prof::kind::gpu_zone);
        if (std::string(e.name) == "raw") { ++seen; CHECK(e.t0_ns == 1000 && e.dur_ns == 600 && e.tid == 3); }
        if (std::string(e.name) == "scaled") { ++seen; CHECK(e.t0_ns == 2'001'000 && e.dur_ns == 1000 && e.tid == 7); }
        if (std::string(e.name) == "backwards") { ++seen; CHECK(e.dur_ns == 0); }
    }
    CHECK(seen == 3);
    auto j = json::parse(r.perfetto_json()); bool gpu_proc = false, gpu_ev = false;
    for (auto& e : j["traceEvents"].arr()) {
        if (e["ph"].str() == "M" && e["name"].str() == "process_name" && e["pid"].num() == 2) gpu_proc = e["args"]["name"].str() == "GPU";
        if (e["ph"].str() == "X" && e["name"].str() == "scaled") gpu_ev = e["pid"].num() == 2 && e["tid"].num() == 7 && e["ts"].num() == 2001.0 && e["dur"].num() == 1.0;
    }
    CHECK(gpu_proc); CHECK(gpu_ev);
}

TEST(prof_exports_are_valid_json) {
    recording rec; auto& r = rec.r;
    r.name_thread("main \"quoted\"\n");
    { FORGE_ZONE("outer"); { FORGE_ZONE("in\\ner\t"); } FORGE_COUNTER("n", 2); FORGE_GAUGE("g", 0.25); FORGE_INSTANT("mark"); }
    int x = 0; prof::note_alloc("tag", &x, 64); prof::gpu_zone("gz", 1, 10, 20);
    r.on = false;
    auto cj = json::parse(r.chrome_json()), pj = json::parse(r.perfetto_json());
    CHECK(cj["traceEvents"].arr().size() == r.events().size());      // Chrome form: one entry per event, nothing else
    CHECK(cj["displayTimeUnit"].str() == "ns");
    std::size_t meta = 0, x_events = 0, c_events = 0, i_events = 0; double last_ts = -1; bool sorted = true, named = false;
    for (auto& e : pj["traceEvents"].arr()) {
        const std::string ph = e["ph"].str();
        if (ph == "M") { ++meta; if (e["name"].str() == "thread_name" && e["args"]["name"].str() == "main \"quoted\"\n") named = true; continue; }
        if (ph == "X") { ++x_events; CHECK(e.find("dur") && e["dur"].num() >= 0); }
        if (ph == "C") { ++c_events; CHECK(e["args"].find("value") != nullptr); }
        if (ph == "i") { ++i_events; CHECK(e["s"].str() == "t"); }
        sorted &= e["ts"].num() >= last_ts; last_ts = e["ts"].num();
        CHECK(e.find("pid") && e.find("tid") && e.find("name"));
    }
    CHECK(meta >= 3);              // forge process, GPU process, this thread
    CHECK(named); CHECK(sorted);   // Perfetto form: metadata first, events in timestamp order
    CHECK(x_events == 3 && c_events == 3 && i_events == 1);
    bool esc = false; for (auto& e : cj["traceEvents"].arr()) esc |= e["name"].str() == "in\\ner\t";
    CHECK(esc);
}

TEST(prof_hardware_counter_zone) {
    recording rec; auto& r = rec.r;
    volatile double sink = 0;
    auto work = [&] { double s = 0; for (int i = 1; i < 200000; ++i) s += 1.0 / double(i); sink = s; };
    auto m = prof::hw_measure(work);
    std::printf("  hw counters: backend=%s full=%d cycles=%llu instructions=%llu cache-misses=%llu branch-misses=%llu ns=%llu\n", m.backend, int(m.counters),
                static_cast<unsigned long long>(m.cycles), static_cast<unsigned long long>(m.instructions), static_cast<unsigned long long>(m.cache_misses),
                static_cast<unsigned long long>(m.branch_misses), static_cast<unsigned long long>(m.ns));
    CHECK(m.ns > 0);
    if (m.counters) CHECK(m.instructions > 200000);   // at least one instruction per iteration
    if (m.cycles_valid) CHECK(m.cycles > 0);
    { FORGE_HW_ZONE("hw_outer"); work(); { prof::hw_zone inner("hw_inner"); work(); } }
    std::size_t n = 0;
    for (auto& e : r.events()) if (e.k == prof::kind::hw_zone) {
        ++n; CHECK(e.dur_ns > 0);
        if (std::string(e.name) == "hw_inner") CHECK(e.aux[0] == 0 && e.aux[1] == 0);            // nested: time only
        if (std::string(e.name) == "hw_outer" && m.counters) CHECK(e.aux[1] > 400000);            // both loops
    }
    CHECK(n == 2);
    CHECK(prof::hw_available() == (m.counters || m.cycles_valid) || !m.cycles_valid);
    auto j = json::parse(r.chrome_json()); bool args = false;
    for (auto& e : j["traceEvents"].arr()) if (e["name"].str() == "hw_outer") args = e["args"].find("instructions") != nullptr;
    CHECK(args);
}
