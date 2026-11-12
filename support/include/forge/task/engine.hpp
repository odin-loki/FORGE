#pragma once
// The engine's data-parallel entry points, task::parallel_for(n, grain, f) and (n, lo, hi, f), on one of two
// executors chosen once per process:
//
//   FORGE_TASK_SCHEDULER=pool    task::pool::global() (pool.hpp). The default.
//   FORGE_TASK_SCHEDULER=sched   a task::scheduler (sched.hpp: work stealing) with engine_threads() - 1 workers;
//                                the calling thread is the last participant, as with the pool.
//
// graph, inference and gpu call these two functions (and engine_threads()) and nothing else of the pool, so
// with `sched` the pool's threads are never created: there is one thread set either way. task::submit stays
// on the pool (nothing in the engine uses it).
//
// The interface is the pool's on both executors:
//   * f(begin, end) is called for disjoint ranges covering [0, n), each of at least min(lo, n) and at most hi
//     items except possibly the last; the caller runs ranges too;
//   * n <= lo, one thread, or a call from inside another engine parallel_for runs f(0, n) on the caller;
//   * the first exception thrown by f is rethrown to the caller.
// The pool hands out ranges of max(lo, remaining / (2 * threads)) items as threads ask for them, so where the
// range boundaries fall depends on timing. The scheduler cuts [0, n) into fixed ranges of
// clamp(n / (4 * threads), lo, hi) items. Neither order is something a kernel may depend on: every engine
// kernel computes each item (row, block, head) independently of how items are grouped into ranges, which is
// why both executors give the same bits (tests: engine_pool / engine_sched and the engine's tests rerun under
// `sched`). A reduction ACROSS items must use parallel_reduce-style per-item results combined in item order.
//
// pool.hpp and sched.hpp both end by including this header, so including either gives the free functions.
#include <forge/task/pool.hpp>
#include <forge/task/sched.hpp>
#include <string_view>

namespace forge::task {

enum class engine_kind { pool, sched };

namespace detail {
inline engine_kind read_engine_kind() {
    const char* e = std::getenv("FORGE_TASK_SCHEDULER");
    return e && std::string_view(e) == "sched" ? engine_kind::sched : engine_kind::pool;
}
// the thread count the pool would use (FORGE_THREADS, else physical cores, else hardware_concurrency())
inline unsigned engine_thread_count() {
    unsigned n = 0;
    if (const char* e = std::getenv("FORGE_THREADS")) n = unsigned(std::strtoul(e, nullptr, 10));
    if (!n) n = physical_cores();
    if (!n) n = std::max(1u, std::thread::hardware_concurrency());
    return n;
}
inline int& engine_depth() { thread_local int d = 0; return d; }   // > 0 while this thread runs a range of an engine parallel_for
struct engine_scope { engine_scope() { ++engine_depth(); } ~engine_scope() { --engine_depth(); } };
} // namespace detail

// Which executor this process uses (read once from FORGE_TASK_SCHEDULER).
inline engine_kind engine() { static const engine_kind k = detail::read_engine_kind(); return k; }
inline const char* engine_name() { return engine() == engine_kind::sched ? "sched" : "pool"; }

// The scheduler behind FORGE_TASK_SCHEDULER=sched: engine_threads() - 1 workers (nullptr when that is zero).
// Created on first use; never created under `pool`.
inline scheduler* engine_scheduler() {
    static const unsigned n = detail::engine_thread_count();
    if (n <= 1) return nullptr;
    static scheduler s(scheduler::config{.threads = n - 1});
    return &s;
}
// Threads that take part in a parallel_for, the caller included.
inline unsigned engine_threads() {
    if (engine() == engine_kind::pool) return pool::global().size();
    scheduler* s = engine_scheduler();
    return s ? s->size() + 1 : 1;
}

template <class F> void parallel_for(std::size_t n, std::size_t lo, std::size_t hi, F&& f) {
    if (engine() == engine_kind::pool) { pool::global().parallel_for(n, lo, hi, std::forward<F>(f)); return; }
    if (n == 0) return;
    lo = std::max<std::size_t>(lo, 1); hi = std::max(hi, lo);
    scheduler* s = engine_scheduler();
    if (!s || n <= lo || detail::engine_depth() > 0) { f(std::size_t(0), n); return; }
    const std::size_t chunk = std::clamp(n / (4 * (std::size_t(s->size()) + 1)), lo, hi);
    if (chunk >= n) { f(std::size_t(0), n); return; }
    s->parallel_for(n, chunk, [&f](std::size_t b, std::size_t e) { detail::engine_scope in; f(b, e); });
}
template <class F> void parallel_for(std::size_t n, std::size_t grain, F&& f) { parallel_for(n, grain, std::size_t(-1), std::forward<F>(f)); }

} // namespace forge::task
