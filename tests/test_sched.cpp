// task::scheduler (sched.hpp): work-stealing deque, task graphs, placement, parallel_for determinism, frame
// pacing, worker-local storage. No test asserts a wall-clock bound; the pacing arithmetic uses a fake clock.
// Run under TSan with -DFORGE_TSAN=ON.
#include "check.hpp"
#include <forge/task/sched.hpp>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
using namespace forge;
using namespace std::chrono_literals;

TEST(ws_deque_owner_and_thieves_take_each_item_once) {
    // One owner pushes and pops, three thieves steal; every item must be seen exactly once. The deque starts
    // at 4 slots so it grows many times while thieves are reading.
    constexpr int N = 60000;
    task::detail::ws_deque<int> dq(4);
    std::vector<int> items(N); std::iota(items.begin(), items.end(), 0);
    std::vector<std::atomic<int>> seen(N);
    std::atomic<bool> done{false}; std::atomic<int> got{0};
    std::vector<std::thread> thieves;
    for (int t = 0; t < 3; ++t) thieves.emplace_back([&] {
        while (!done.load() || dq.maybe_nonempty()) {
            bool lost; if (int* x = dq.steal(lost)) { ++seen[std::size_t(*x)]; ++got; } else std::this_thread::yield();
        }
    });
    for (int i = 0; i < N; ++i) {
        dq.push(&items[std::size_t(i)]);
        if (i % 3 == 2) if (int* x = dq.take()) { ++seen[std::size_t(*x)]; ++got; }
    }
    while (int* x = dq.take()) { ++seen[std::size_t(*x)]; ++got; }
    done = true;
    for (auto& t : thieves) t.join();
    bool ok = true; for (auto& s : seen) ok &= s == 1;
    CHECK(ok); CHECK(got == N);
    bool lost; CHECK(dq.steal(lost) == nullptr); CHECK(!lost); CHECK(dq.take() == nullptr);
}

TEST(sched_spawn_values_exceptions_and_many_tasks) {
    task::scheduler s(4);
    auto a = s.spawn([] { return 6 * 7; });
    auto b = s.spawn([] { return std::string("ok"); });
    auto c = s.spawn([]() -> int { throw std::runtime_error("boom"); });
    CHECK(a.get() == 42); CHECK(b.get() == "ok"); CHECK_THROWS(c.get()); CHECK(c.failed()); CHECK(!a.failed());
    std::atomic<int> n{0}; std::vector<task::task<void>> ts;
    for (int i = 0; i < 2000; ++i) ts.push_back(s.spawn([&] { ++n; }));
    for (auto& t : ts) t.get();
    CHECK(n == 2000);
    s.wait_idle(); CHECK(s.pending() == 0);
    CHECK(s.total_stats().executed >= 2003);
}

TEST(sched_tasks_spawned_by_a_blocked_worker_are_stolen) {
    // The parent spawns two children onto its own deque and then spins (without helping) until both have
    // run: only a steal by another worker can run them.
    task::scheduler s(3);
    std::atomic<int> ran{0}; std::atomic<int> parent_worker{-2}; std::atomic<int> same{0};
    auto p = s.spawn([&] {
        int me = s.current_worker(); parent_worker = me;
        for (int i = 0; i < 2; ++i) (void)s.spawn([&, me] { if (s.current_worker() == me) ++same; ++ran; });
        while (ran.load() < 2) std::this_thread::yield();
    });
    while (!p.done()) std::this_thread::sleep_for(1ms);   // the test thread must not help either
    CHECK(ran == 2); CHECK(parent_worker >= 0); CHECK(same == 0);
    s.wait_idle();
    CHECK(s.total_stats().stolen >= 2);
}

TEST(sched_continuations_and_when_all) {
    task::scheduler s(4);
    auto a = s.spawn([] { std::this_thread::sleep_for(5ms); return 6; });
    auto b = a.then([](int v) { return v * 7; });
    auto c = b.then([](int v) { return std::to_string(v); });
    CHECK(c.get() == "42"); CHECK(b.get() == 42);
    auto late = a.then([](int v) { return v + 1; });          // predecessor already finished
    CHECK(late.get() == 7);
    auto v = s.spawn([] {}); auto v2 = v.then([] { return 5; }); CHECK(v2.get() == 5);
    // a failed predecessor fails the continuation without running it
    std::atomic<bool> ran{false};
    auto bad = s.spawn([]() -> int { throw std::runtime_error("x"); });
    auto after = bad.then([&](int) { ran = true; return 1; });
    CHECK_THROWS(after.get()); CHECK(!ran);
    // dependency counting with a diamond
    std::atomic<int> order{0}; int at[4] = {};
    auto r = s.spawn([&] { std::this_thread::sleep_for(3ms); at[0] = ++order; });
    auto l1 = s.spawn_after({r}, [&] { at[1] = ++order; });
    auto l2 = s.spawn_after({r}, [&] { std::this_thread::sleep_for(2ms); at[2] = ++order; });
    auto j = s.spawn_after({l1, l2}, [&] { at[3] = ++order; });
    j.get();
    CHECK(at[0] == 1); CHECK(at[3] == 4); CHECK(at[1] > 1 && at[2] > 1);
    auto all = s.when_all({a, b, bad}); CHECK_THROWS(all.get());
    auto all2 = s.when_all({a, b}); all2.get(); CHECK(!all2.failed());
    std::vector<task::task<int>> sq; for (int i = 0; i < 50; ++i) sq.push_back(s.spawn([i] { return i * i; }));
    auto vals = s.when_all(sq).get(); long sum = 0; for (int x : vals) sum += x;
    CHECK(vals.size() == 50 && vals[7] == 49 && sum == 40425);
    // a wide graph: 200 leaves -> 20 mids -> 1 root
    std::vector<task::task_ref> leaves; std::atomic<int> cnt{0};
    for (int i = 0; i < 200; ++i) leaves.push_back(s.spawn([&] { ++cnt; }));
    std::vector<task::task_ref> mids;
    for (int m = 0; m < 20; ++m) mids.push_back(s.spawn_after(std::span<const task::task_ref>(leaves.data() + m * 10, 10), [&] { cnt += 1000; }));
    auto root = s.spawn_after(std::span<const task::task_ref>(mids), [&] { return cnt.load(); });
    CHECK(root.get() == 200 + 20 * 1000);
}

TEST(sched_cancellation) {
    task::scheduler s(2);
    std::atomic<bool> go{false};
    auto b1 = s.spawn([&] { while (!go) std::this_thread::sleep_for(1ms); });
    auto b2 = s.spawn([&] { while (!go) std::this_thread::sleep_for(1ms); });
    std::stop_source src; std::atomic<bool> ran{false};
    auto f = s.spawn([&] { ran = true; }, {.stop = src.get_token()});
    auto g = f.then([] { return 1; });
    src.request_stop(); go = true;
    bool threw = false; try { f.get(); } catch (const task::cancelled&) { threw = true; }
    CHECK(threw); CHECK(!ran);
    threw = false; try { g.get(); } catch (const task::cancelled&) { threw = true; }
    CHECK(threw);
    b1.get(); b2.get();
    std::stop_source s2; std::atomic<bool> started{false};
    auto h = s.spawn([&](std::stop_token st) { started = true; int spins = 0; while (!st.stop_requested() && spins < 20000) { std::this_thread::sleep_for(1ms); ++spins; } return spins; }, {.stop = s2.get_token()});
    while (!started) std::this_thread::sleep_for(1ms);
    s2.request_stop(); CHECK(h.get() < 20000);
}

TEST(sched_priorities_are_served_highest_first) {
    task::scheduler s(1);   // one worker: the order it drains the injector is observable
    std::atomic<bool> go{false}; std::atomic<bool> busy{false};
    auto blocker = s.spawn([&] { busy = true; while (!go) std::this_thread::sleep_for(1ms); });
    while (!busy) std::this_thread::sleep_for(1ms);
    std::mutex m; std::vector<int> order; std::vector<task::task<void>> ts;
    using P = task::priority;
    for (P p : {P::normal, P::high, P::low, P::high, P::normal})
        ts.push_back(s.spawn([&, p] { std::lock_guard l(m); order.push_back(int(p)); }, {.prio = p}));
    go = true;
    while (s.pending() != 0) std::this_thread::sleep_for(1ms);   // do not help: the worker alone decides the order
    CHECK((order == std::vector<int>{2, 2, 1, 1, 0}));
}

TEST(sched_parallel_for_chunks_are_deterministic_and_cover_once) {
    // Chunk k is [k * grain, min(n, (k + 1) * grain)) for every thread count.
    for (unsigned threads : {1u, 2u, 5u}) {
        task::scheduler s(threads);
        for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(7), std::size_t(64), std::size_t(1000), std::size_t(100003)})
            for (std::size_t grain : {std::size_t(1), std::size_t(7), std::size_t(64), std::size_t(5000)}) {
                if (n / grain > 20000) continue;
                std::vector<std::atomic<int>> hit(n); std::atomic<bool> shape{true};
                s.parallel_for(n, grain, [&](std::size_t b, std::size_t e) {
                    if (b % grain != 0 || e != std::min(n, b + grain)) shape = false;
                    for (auto i = b; i < e; ++i) ++hit[i];
                });
                bool ok = true; for (auto& h : hit) ok &= h == 1;
                CHECK_MSG(ok && shape, "threads=%u n=%zu grain=%zu", threads, n, grain);
            }
    }
    CHECK(task::simd_grain(100, 16) == 112); CHECK(task::simd_grain(0, 8) == 8); CHECK(task::simd_grain(64, 16) == 64); CHECK(task::simd_grain(5, 0) == 5);
}

TEST(sched_parallel_reduce_is_bit_identical_across_thread_counts) {
    // A float sum whose value depends on association order: the chunked reduction must give the same bits
    // for 1, 2, 3 and 6 threads and on every repeat, and must equal the same chunking done serially.
    std::vector<float> x(200001); std::uint32_t r = 12345;
    for (auto& v : x) { r = r * 1664525u + 1013904223u; v = (float(r >> 8) / float(1 << 24) - 0.5f) * (1.0f + float(r & 1023)); }
    const std::size_t grain = 1000;
    auto chunk = [&](std::size_t b, std::size_t e) { float t = 0; for (auto i = b; i < e; ++i) t += x[i]; return t; };
    float ref = 0; for (std::size_t b = 0; b < x.size(); b += grain) ref += chunk(b, std::min(x.size(), b + grain));
    for (unsigned threads : {1u, 2u, 3u, 6u}) {
        task::scheduler s(threads);
        for (int rep = 0; rep < 5; ++rep) {
            float got = s.parallel_reduce(x.size(), grain, 0.0f, chunk, [](float a, float b) { return a + b; });
            CHECK_MSG(std::memcmp(&got, &ref, 4) == 0, "threads=%u rep=%d got=%a ref=%a", threads, rep, double(got), double(ref));
        }
    }
}

TEST(sched_nested_parallel_for_and_waits_do_not_deadlock) {
    task::scheduler s(3);
    std::atomic<long> sum{0};
    s.parallel_for(48, 1, [&](std::size_t b, std::size_t e) {
        for (auto i = b; i < e; ++i)
            s.parallel_for(60, 4, [&](std::size_t b2, std::size_t e2) {
                s.parallel_for(e2 - b2, 1, [&](std::size_t b3, std::size_t e3) { sum += long(e3 - b3); });
            });
    });
    CHECK(sum == 48 * 60);
    // every worker waits on tasks it spawned: more blocked waits than threads
    std::function<int(int)> fib = [&](int n) -> int {
        if (n < 2) return n;
        auto a = s.spawn([&, n] { return fib(n - 1); });
        int b = fib(n - 2);
        return a.get() + b;
    };
    CHECK(s.spawn([&] { return fib(16); }).get() == 987);
    // parallel_for from several outside threads at once
    std::vector<std::thread> ts; std::atomic<long> tot{0};
    for (int t = 0; t < 4; ++t) ts.emplace_back([&] { for (int i = 0; i < 100; ++i) s.parallel_for(1000, 16, [&](std::size_t b, std::size_t e) { tot += long(e - b); }); });
    for (auto& t : ts) t.join();
    CHECK(tot == 4L * 100 * 1000);
    bool threw = false;
    try { s.parallel_for(10000, 1, [&](std::size_t b, std::size_t) { if (b > 5000) throw std::runtime_error("x"); }); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    sum = 0; s.parallel_for(10000, 3, [&](std::size_t b, std::size_t e) { sum += long(e - b); });
    CHECK(sum == 10000);
}

TEST(sched_sleeping_workers_are_woken) {
    task::scheduler s(task::scheduler::config{.threads = 4, .spin = 1});
    for (int round = 0; round < 20; ++round) {
        std::this_thread::sleep_for(3ms);   // workers park
        std::atomic<int> n{0}; std::vector<task::task<void>> ts;
        for (int i = 0; i < 8; ++i) ts.push_back(s.spawn([&] { ++n; }));
        while (s.pending() != 0) std::this_thread::sleep_for(1ms);   // no helping: parked workers must wake by themselves
        CHECK(n == 8);
    }
    std::uint64_t parked = 0; for (unsigned i = 0; i < s.size(); ++i) parked += s.worker_stats(i).parked;
    CHECK(parked > 0);
}

static os::cpu_topology fake_hybrid(unsigned p_cores, unsigned e_cores, bool smt) {
    // P cores first (two SMT threads each when smt), then E cores: the layout of an Intel hybrid desktop part.
    os::cpu_topology t; t.hybrid = e_cores > 0; unsigned id = 0, core = 0;
    for (unsigned i = 0; i < p_cores; ++i, ++core) for (unsigned k = 0; k < (smt ? 2u : 1u); ++k) { os::logical_cpu c; c.id = id++; c.core = core; c.kind = os::core_kind::performance; t.cpus.push_back(c); }
    for (unsigned i = 0; i < e_cores; ++i, ++core) { os::logical_cpu c; c.id = id++; c.core = core; c.kind = os::core_kind::efficiency; t.cpus.push_back(c); }
    t.logical = id; t.physical = core;
    return t;
}

TEST(plan_workers_orders_performance_cores_first) {
    auto t = fake_hybrid(2, 4, true);   // cpus 0,1 = P0; 2,3 = P1; 4..7 = E
    auto w = task::plan_workers(t, 10);
    std::vector<unsigned> cpus; for (auto& x : w) cpus.push_back(x.cpu);
    CHECK((cpus == std::vector<unsigned>{0, 2, 4, 5, 6, 7, 1, 3, 0, 2}));   // P cores, E cores, P siblings, then wrap
    CHECK(w[0].kind == os::core_kind::performance && w[2].kind == os::core_kind::efficiency && w[6].kind == os::core_kind::performance);
    std::vector<os::numa_node> nodes(2); nodes[0].id = 0; nodes[0].cpus = {0, 1, 4, 5}; nodes[1].id = 1; nodes[1].cpus = {2, 3, 6, 7};
    auto wn = task::plan_workers(t, 4, nodes);
    CHECK(wn[0].node == 0 && wn[1].node == 1 && wn[2].node == 0 && wn[3].node == 0);
    auto homo = task::plan_workers(fake_hybrid(4, 0, false), 3);
    CHECK(homo.size() == 3 && homo[2].cpu == 2 && homo[2].kind == os::core_kind::performance);
    CHECK(task::plan_workers(os::cpu_topology{}, 2).size() == 2);   // an empty topology still yields workers
}

TEST(sched_performance_tasks_never_run_on_efficiency_workers) {
    // Injected topology: 2 P cores + 4 E cores, 6 workers, no pinning (the machine running the test need not
    // be hybrid). Tasks marked performance must only ever execute on the two P workers; unmarked tasks must
    // be able to reach E workers.
    task::scheduler s(task::scheduler::config{.threads = 6, .topology = fake_hybrid(2, 4, false)});
    CHECK(s.hybrid());
    unsigned np = 0; for (auto& sl : s.slots()) np += sl.kind == os::core_kind::performance;
    CHECK(np == 2);
    std::atomic<int> on_e{0}, on_p{0}, any_on_e{0}, eff_total{0};
    auto kind_here = [&] { int w = s.current_worker(); return w < 0 ? os::core_kind::unknown : s.slots()[std::size_t(w)].kind; };
    for (int i = 0; i < 300; ++i) {
        (void)s.spawn([&] {
            auto k = kind_here(); if (k == os::core_kind::efficiency) ++on_e; else if (k == os::core_kind::performance) ++on_p;
            // spawned from a P worker: goes to its own performance deque, which E workers never steal from
            (void)s.spawn([&] { if (kind_here() == os::core_kind::efficiency) ++on_e; }, {.place = task::placement::performance});
        }, {.place = task::placement::performance});
        (void)s.spawn([&] {
            if (kind_here() == os::core_kind::efficiency) ++any_on_e;
            (void)s.spawn([&] { if (kind_here() == os::core_kind::efficiency) ++on_e; }, {.place = task::placement::performance});   // possibly from an E worker: re-routed
            std::this_thread::sleep_for(200us);
        });
        (void)s.spawn([&] { ++eff_total; }, {.place = task::placement::efficiency});
    }
    while (s.pending() != 0) std::this_thread::sleep_for(1ms);   // the test thread does not help
    CHECK(on_e == 0); CHECK(on_p == 300); CHECK(eff_total == 300);
    CHECK_MSG(any_on_e > 0, "no unmarked task reached an efficiency worker");
    // On a homogeneous topology the marks are ignored (nothing is starved).
    task::scheduler h(task::scheduler::config{.threads = 2, .topology = fake_hybrid(2, 0, false)});
    CHECK(!h.hybrid());
    CHECK(h.spawn([] { return 1; }, {.place = task::placement::efficiency}).get() == 1);
    CHECK(h.spawn([] { return 2; }, {.place = task::placement::performance}).get() == 2);
}

TEST(sched_pinning_and_numa_binding_start_and_run) {
    // Real pinning against the machine's own topology: the calls may be refused (containers, cpusets); the
    // scheduler must work either way. An injected two-node layout exercises the victim ordering.
    {
        task::scheduler s(task::scheduler::config{.threads = 2, .pin = true});
        CHECK(s.spawn([] { return 3; }).get() == 3);
        CHECK(s.pin_failures() <= 2);
    }
    {
        task::scheduler s(task::scheduler::config{.threads = 2, .numa_bind = true});
        CHECK(s.spawn([] { return 4; }).get() == 4);
    }
    auto topo = fake_hybrid(4, 0, false);
    std::vector<os::numa_node> nodes(2); nodes[0].id = 0; nodes[0].cpus = {0, 1}; nodes[1].id = 1; nodes[1].cpus = {2, 3};
    task::scheduler s(task::scheduler::config{.threads = 4, .topology = topo, .numa = nodes});
    CHECK(s.slots()[0].node == 0 && s.slots()[3].node == 1);
    std::atomic<long> sum{0};
    s.parallel_for(10000, 10, [&](std::size_t b, std::size_t e) { sum += long(e - b); });
    CHECK(sum == 10000);
}

TEST(sched_worker_local_storage) {
    task::scheduler s(4);
    task::worker_local<long> acc(s);
    acc.local() += 0;   // the calling thread's slot exists even if the workers take every chunk
    s.parallel_for(100000, 100, [&](std::size_t b, std::size_t e) { long& a = acc.local(); for (auto i = b; i < e; ++i) a += long(i); });
    long total = 0; int slots = 0; acc.for_each([&](long& v) { total += v; ++slots; });
    CHECK(total == 100000L * 99999 / 2);
    CHECK(slots == 5);   // four workers and the calling thread
    task::worker_local<std::vector<int>> bufs(s, std::vector<int>(3, 7));
    CHECK(bufs.local().size() == 3);
    CHECK(s.spawn([&] { return bufs.local()[2]; }).get() == 7);
}

TEST(frame_pacer_runs_must_work_and_defers_by_budget) {
    // A fake clock (advanced only by the tasks) makes the budget arithmetic exact on any machine.
    task::scheduler s(1);
    std::atomic<std::uint64_t> now{1000};
    task::frame_pacer fp(s, 10'000, [&] { return now.load(); });
    std::atomic<int> must{0}, bg{0};
    for (int i = 0; i < 6; ++i) fp.defer([&] { now += 3000; ++bg; }, 3000);
    fp.begin_frame();
    for (int i = 0; i < 4; ++i) fp.run([&] { now += 1000; ++must; });
    auto st = fp.end_frame();
    // 4000 of must-run work, then deferred tasks of 3000 each while elapsed + 3000 <= 10000: two fit.
    CHECK(must == 4); CHECK(st.ran == 4); CHECK(st.deferred_ran == 2); CHECK(bg == 2); CHECK(st.deferred_left == 4);
    CHECK(st.elapsed_ns == 10'000); CHECK(!st.over_budget); CHECK(fp.remaining_ns() == 0);
    fp.begin_frame();                       // an empty frame drains three more
    st = fp.end_frame();
    CHECK(st.deferred_ran == 3); CHECK(st.deferred_left == 1); CHECK(st.frame == 2);
    fp.begin_frame();
    fp.run([&] { now += 25'000; });         // the frame's own work overruns: nothing deferred starts
    st = fp.end_frame();
    CHECK(st.over_budget); CHECK(st.deferred_ran == 0); CHECK(st.deferred_left == 1);
    fp.begin_frame(); st = fp.end_frame(); CHECK(st.deferred_left == 0); CHECK(bg == 6);
    // unknown cost: the observed average (3000 so far) decides: 0 + 3000, 4000 + 3250 <= 10000, then 8000 + 3438 > 10000
    for (int i = 0; i < 5; ++i) fp.defer([&] { now += 4000; ++bg; });
    fp.begin_frame(); st = fp.end_frame();
    CHECK(st.deferred_ran == 2); CHECK(st.deferred_left == 3);
    fp.begin_frame(); fp.run([] { throw std::runtime_error("frame"); }); CHECK_THROWS(fp.end_frame());
}

TEST(frame_pacer_real_clock_many_frames) {
    task::scheduler s(3);
    task::frame_pacer fp(s, 5'000'000);
    std::atomic<int> must{0}, bg{0};
    for (int i = 0; i < 40; ++i) fp.defer([&] { std::this_thread::sleep_for(200us); ++bg; });
    for (int f = 0; f < 30 && fp.deferred(); ++f) {
        fp.begin_frame();
        for (int i = 0; i < 8; ++i) fp.run([&] { ++must; });
        (void)fp.end_frame();
    }
    CHECK(must % 8 == 0 && must >= 8);
    for (int f = 0; f < 100000 && fp.deferred(); ++f) { fp.begin_frame(); (void)fp.end_frame(); }
    CHECK(bg == 40);   // everything deferred eventually runs (no wall-clock assertion)
}

TEST(sched_destructor_drains_and_global_exists) {
    std::atomic<int> n{0};
    { task::scheduler s(2); for (int i = 0; i < 200; ++i) (void)s.spawn([&] { ++n; }); }
    CHECK(n == 200);
    CHECK(task::scheduler::global().size() >= 1);
    CHECK(task::scheduler::global().spawn([] { return 9; }).get() == 9);
}

TEST(sched_stress_mixed_spawn_wait_cancel) {
    // Many short graphs from several threads; checks counts only. The main case for TSan.
    task::scheduler s(4);
    std::atomic<long> total{0}; std::atomic<int> wrong{0}; std::vector<std::thread> ts;
    for (int t = 0; t < 3; ++t) ts.emplace_back([&, t] {
        for (int i = 0; i < 300; ++i) {
            auto a = s.spawn([&] { total += 1; return 1; }, {.prio = task::priority(i % 3)});
            auto b = a.then([&](int v) { total += v; return v + 1; });
            auto c = s.spawn_after({a, b}, [&] { s.parallel_for(32, 4, [&](std::size_t x, std::size_t y) { total += long(y - x); }); });
            std::stop_source src; bool cancel = (i + t) % 7 == 0; if (cancel) src.request_stop();
            auto d = s.spawn([&] { total += 1000000; }, {.stop = src.get_token()});
            c.get(); d.wait();
            if (d.failed() != cancel) ++wrong;
        }
    });
    for (auto& t : ts) t.join();
    s.wait_idle();
    long cancelled = 0; for (int t = 0; t < 3; ++t) for (int i = 0; i < 300; ++i) cancelled += (i + t) % 7 == 0;
    CHECK(wrong == 0);
    CHECK(total == 3L * 300 * (1 + 1 + 32) + (900 - cancelled) * 1000000L);
}
