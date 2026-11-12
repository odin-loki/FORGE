#pragma once
// task::scheduler: a work-stealing scheduler with task graphs. It sits next to task::pool (pool.hpp), which
// keeps its API and behaviour; nothing in the engine was moved onto the scheduler.
//
//   scheduler s({.threads = 8});
//   auto a = s.spawn([] { return 6; });                         // task<int>
//   auto b = a.then([](int v) { return v * 7; });               // continuation, runs when `a` is done
//   auto c = s.when_all({a, b});                                // task<void>, done when both are
//   int v = b.get();                                            // waits by running other tasks (never blocks a core)
//   s.parallel_for(n, grain, [&](std::size_t b, std::size_t e) { ... });
//
// Structure
//   * Every worker owns Chase-Lev deques (one per priority, per placement set). It pushes and pops at the
//     bottom (LIFO, cache-warm); idle workers steal from the top (FIFO, the oldest and usually largest work)
//     of a victim chosen at random, same NUMA node first. Threads that are not workers submit through a
//     mutex-protected injector queue.
//   * Priorities (low, normal, high) are scanned highest first across own deques, injector and victims. They
//     are a preference, not a strict order: a running task is never preempted.
//   * Idle workers spin briefly, then park on an event count (std::atomic wait/notify, or a mutex and
//     condition variable with FORGE_TASK_PARK_CONDVAR or where the library has no atomic wait).
//   * wait / get / parallel_for called on any thread help: they run queued tasks until the awaited work is
//     done, so nested parallel_for and task-in-task waits cannot deadlock on an exhausted pool. A task must
//     not hold a lock across such a call (the helper may run a task that wants the same lock).
//   * Placement: a task marked placement::performance only runs on workers that sit on performance cores
//     (os::cpu_topology core classes; on a homogeneous machine the mark has no effect). placement::efficiency
//     is a preference for efficiency cores; performance workers take such tasks only when they have nothing
//     else. The topology can be injected (config::topology) so the policy is testable without the hardware.
//   * Determinism: parallel_for chunk k is always [k * grain, min(n, (k + 1) * grain)), whatever the thread
//     count or timing. parallel_reduce combines the chunk results in chunk order. Which thread runs a chunk,
//     and when, is not deterministic.
#include <forge/os/cpu_topology.hpp>
#include <forge/os/numa.hpp>
#include <forge/task/pool.hpp>
#include <deque>
#include <map>
#include <optional>
#include <span>
#include <tuple>

namespace forge::task {

enum class priority : std::uint8_t { low = 0, normal = 1, high = 2 };
enum class placement : std::uint8_t { any = 0, performance = 1, efficiency = 2 };
struct spawn_options { priority prio = priority::normal; placement place = placement::any; std::stop_token stop{}; };

class scheduler;

namespace detail {

// ---- parking primitive: wait while *w == old; wake one / all ----
struct parker {
#if defined(__cpp_lib_atomic_wait) && !defined(FORGE_TASK_PARK_CONDVAR)
    void wait(std::atomic<std::uint32_t>& w, std::uint32_t old) { w.wait(old, std::memory_order_seq_cst); }
    void wake_one(std::atomic<std::uint32_t>& w) { w.notify_one(); }
    void wake_all(std::atomic<std::uint32_t>& w) { w.notify_all(); }
#else
    std::mutex m; std::condition_variable cv;
    void wait(std::atomic<std::uint32_t>& w, std::uint32_t old) { std::unique_lock l(m); cv.wait(l, [&] { return w.load() != old; }); }
    void wake_one(std::atomic<std::uint32_t>&) { { std::lock_guard l(m); } cv.notify_all(); }   // waiters differ in what they wait for
    void wake_all(std::atomic<std::uint32_t>&) { { std::lock_guard l(m); } cv.notify_all(); }
#endif
};

// ---- Chase-Lev work-stealing deque of T* (Chase & Lev 2005; orderings after Le et al. 2013, all seq_cst here).
// One owner thread calls push/take; any thread calls steal. Grown rings are kept until destruction because a
// thief may still be reading an old one.
template <class T> class ws_deque {
    struct ring {
        std::int64_t cap; std::unique_ptr<std::atomic<T*>[]> s;
        explicit ring(std::int64_t c) : cap(c), s(new std::atomic<T*>[std::size_t(c)]) {}
        T* get(std::int64_t i) const { return s[std::size_t(i & (cap - 1))].load(std::memory_order_relaxed); }
        void put(std::int64_t i, T* x) { s[std::size_t(i & (cap - 1))].store(x, std::memory_order_relaxed); }
    };
    alignas(64) std::atomic<std::int64_t> top_{0};
    alignas(64) std::atomic<std::int64_t> bottom_{0};
    alignas(64) std::atomic<ring*> ring_;
    std::vector<std::unique_ptr<ring>> rings_;   // owner only
public:
    explicit ws_deque(std::int64_t cap = 64) { rings_.push_back(std::make_unique<ring>(cap)); ring_.store(rings_.back().get()); }
    ws_deque(const ws_deque&) = delete; ws_deque& operator=(const ws_deque&) = delete;
    void push(T* x) {
        std::int64_t b = bottom_.load(), t = top_.load(); ring* a = ring_.load();
        if (b - t > a->cap - 1) {
            auto g = std::make_unique<ring>(a->cap * 2);
            for (std::int64_t i = t; i < b; ++i) g->put(i, a->get(i));
            a = g.get(); rings_.push_back(std::move(g)); ring_.store(a);
        }
        a->put(b, x);
        bottom_.store(b + 1);
    }
    T* take() {
        std::int64_t b = bottom_.load() - 1; ring* a = ring_.load();
        bottom_.store(b);
        std::int64_t t = top_.load();
        if (t > b) { bottom_.store(b + 1); return nullptr; }
        T* x = a->get(b);
        if (t == b) {   // last item: race the thieves for it
            if (!top_.compare_exchange_strong(t, t + 1)) x = nullptr;
            bottom_.store(b + 1);
        }
        return x;
    }
    // nullptr with lost = true means another thread won the item (worth retrying); lost = false means empty.
    T* steal(bool& lost) {
        lost = false;
        std::int64_t t = top_.load(), b = bottom_.load();
        if (t >= b) return nullptr;
        ring* a = ring_.load();
        T* x = a->get(t);
        if (!top_.compare_exchange_strong(t, t + 1)) { lost = true; return nullptr; }
        return x;
    }
    bool maybe_nonempty() const { return top_.load() < bottom_.load(); }
};

struct spin_lock {
    std::atomic_flag f = ATOMIC_FLAG_INIT;
    void lock() { for (unsigned i = 1; f.test_and_set(std::memory_order_acquire); ++i) { if (i & 63) cpu_relax(); else std::this_thread::yield(); } }
    void unlock() { f.clear(std::memory_order_release); }
};

struct node {
    std::atomic<std::uint32_t> refs{2};      // the handle and the scheduler
    std::atomic<std::uint32_t> state{0};     // 0 not started, 1 running, 2 done
    std::atomic<std::int32_t> deps{1};       // unfinished dependencies, plus one held while they are being registered
    std::atomic<std::uint32_t> waiters{0};
    spin_lock mu;                            // guards succ and the move to done
    std::vector<node*> succ;
    std::exception_ptr err;
    std::stop_token stop; priority prio = priority::normal; placement place = placement::any;
    scheduler* owner = nullptr;
    virtual void invoke() = 0;
    virtual ~node() = default;
    void retain() { refs.fetch_add(1, std::memory_order_relaxed); }
    void release() { if (refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this; }
    bool done() const { return state.load() == 2; }
};
template <class R> struct node_r : node { std::optional<R> val; };
template <> struct node_r<void> : node {};
template <class F, class R> struct node_f final : node_r<R> {
    std::optional<F> f;
    explicit node_f(F&& g) : f(std::move(g)) {}
    void invoke() override {
        struct drop { std::optional<F>& f; ~drop() { f.reset(); } } d{f};   // release captures (e.g. predecessor handles) once run
        if constexpr (std::is_invocable_v<F&, std::stop_token>) {
            if constexpr (std::is_void_v<R>) (*f)(this->stop); else this->val.emplace((*f)(this->stop));
        } else {
            if constexpr (std::is_void_v<R>) (*f)(); else this->val.emplace((*f)());
        }
    }
};
template <class F> using result_of = typename std::conditional_t<std::is_invocable_v<F&, std::stop_token>, std::invoke_result<F&, std::stop_token>, std::invoke_result<F&>>::type;

} // namespace detail

// An untyped handle to a spawned task. Copies share the task.
class task_ref {
public:
    task_ref() = default;
    task_ref(const task_ref& o) : n_(o.n_) { if (n_) n_->retain(); }
    task_ref(task_ref&& o) noexcept : n_(o.n_) { o.n_ = nullptr; }
    task_ref& operator=(task_ref o) noexcept { std::swap(n_, o.n_); return *this; }
    ~task_ref() { if (n_) n_->release(); }
    bool valid() const { return n_ != nullptr; }
    bool done() const { return n_ && n_->done(); }
    inline void wait() const;                       // helps: runs other tasks until this one is done
    std::exception_ptr error() const { wait(); return n_->err; }
    bool failed() const { return error() != nullptr; }
protected:
    friend class scheduler;
    explicit task_ref(detail::node* n) : n_(n) {}
    detail::node* n_ = nullptr;
};

template <class R> class task : public task_ref {
public:
    task() = default;
    // The value (a reference that lives as long as a handle does), or rethrows what the task threw;
    // task::cancelled if its stop_token fired before it started.
    decltype(auto) get() const {
        wait();
        if (n_->err) std::rethrow_exception(n_->err);
        if constexpr (!std::is_void_v<R>) return static_cast<R&>(*static_cast<detail::node_r<R>*>(n_)->val);
    }
    // f(value) (or f() for task<void>) once this task has finished. If this task failed or was cancelled the
    // continuation fails with the same exception and f is not called.
    template <class F> auto then(F&& f, spawn_options o = {}) const;
private:
    friend class scheduler;
    explicit task(detail::node* n) : task_ref(n) {}
};

// A worker and the CPU it is meant for.
struct worker_slot { unsigned cpu = 0, core = 0; os::core_kind kind = os::core_kind::performance; int node = 0; };

// Chooses CPUs for n workers: one logical CPU per performance core first, then one per efficiency core, then
// the remaining SMT siblings in the same order; wraps around when n exceeds the logical CPU count. `nodes`
// (os::numa_nodes()) fills worker_slot::node; without it every worker is on node 0. Cores of unknown class
// count as performance.
inline std::vector<worker_slot> plan_workers(const os::cpu_topology& t, unsigned n, const std::vector<os::numa_node>& nodes = {}) {
    std::vector<worker_slot> all;
    auto node_of = [&](unsigned cpu) { for (auto& nd : nodes) for (auto c : nd.cpus) if (c == cpu) return int(nd.id); return 0; };
    std::vector<const os::logical_cpu*> cpus; for (auto& c : t.cpus) cpus.push_back(&c);
    std::stable_sort(cpus.begin(), cpus.end(), [](auto* a, auto* b) { return a->id < b->id; });
    std::map<std::pair<unsigned, unsigned>, unsigned> rank;   // (package, core) -> SMT sibling index handed out so far
    std::vector<std::tuple<unsigned, int, const os::logical_cpu*>> order;   // (sibling index, is efficiency, cpu)
    for (auto* c : cpus) order.emplace_back(rank[{c->package, c->core}]++, c->kind == os::core_kind::efficiency ? 1 : 0, c);
    std::stable_sort(order.begin(), order.end(), [](auto& a, auto& b) { return std::tie(std::get<0>(a), std::get<1>(a)) < std::tie(std::get<0>(b), std::get<1>(b)); });
    for (auto& [sib, eff, c] : order) all.push_back({c->id, c->core, eff ? os::core_kind::efficiency : os::core_kind::performance, node_of(c->id)});
    if (all.empty()) all.push_back({});
    std::vector<worker_slot> out;
    for (unsigned i = 0; i < n; ++i) out.push_back(all[i % all.size()]);
    return out;
}

// parallel_for grain rounded up to a whole number of SIMD vectors (so only the last chunk has a tail).
constexpr std::size_t simd_grain(std::size_t grain, std::size_t lanes) { return lanes ? (std::max<std::size_t>(grain, 1) + lanes - 1) / lanes * lanes : std::max<std::size_t>(grain, 1); }

class scheduler {
public:
    struct config {
        unsigned threads = 0;                            // workers; 0 = FORGE_THREADS, else physical cores, else hardware_concurrency()
        bool pin = false;                                // pin each worker to its planned CPU (os::pin_thread_to)
        bool numa_bind = false;                          // instead bind each worker to the CPUs of its NUMA node
        std::optional<os::cpu_topology> topology{};      // injected topology (tests); default os::cpu_topology_query()
        std::optional<std::vector<os::numa_node>> numa{};// injected NUMA layout; default os::numa_nodes() when numa_bind, else one node
        unsigned spin = 200;                             // idle rounds of looking for work before a worker parks
    };
    struct stats { std::uint64_t executed = 0, stolen = 0, parked = 0; };

    scheduler() : scheduler(config{}) {}
    explicit scheduler(unsigned threads) : scheduler(config{.threads = threads}) {}
    explicit scheduler(config c) : cfg_(std::move(c)) {
        unsigned n = cfg_.threads;
        if (!n) if (const char* e = std::getenv("FORGE_THREADS")) n = unsigned(std::strtoul(e, nullptr, 10));
        os::cpu_topology topo = cfg_.topology ? *cfg_.topology : os::cpu_topology_query();
        if (!n) n = topo.physical;
        if (!n) n = std::max(1u, std::thread::hardware_concurrency());
        std::vector<os::numa_node> nodes = cfg_.numa ? *cfg_.numa : cfg_.numa_bind ? os::numa_nodes() : std::vector<os::numa_node>{};
        slots_ = plan_workers(topo, n, nodes);
        bool p = false, e = false;
        for (auto& s : slots_) (s.kind == os::core_kind::efficiency ? e : p) = true;
        mixed_ = p && e;
        for (unsigned i = 0; i < n; ++i) workers_.push_back(std::make_unique<worker>());
        for (unsigned i = 0; i < n; ++i) {   // victims: same node first
            for (unsigned j = 0; j < n; ++j) if (j != i) (slots_[j].node == slots_[i].node ? workers_[i]->near : workers_[i]->far).push_back(j);
            workers_[i]->rng = 0x9E3779B97F4A7C15ull * (i + 1);
        }
        for (unsigned i = 0; i < n; ++i) workers_[i]->th = std::thread([this, i, nodes] { loop_(i, nodes); });
    }
    // Runs everything still queued (or skips it, if cancelled), then stops the workers.
    ~scheduler() {
        wait_idle();
        stop_.store(true);
        epoch_.fetch_add(1); park_.wake_all(epoch_);
        for (auto& w : workers_) w->th.join();
    }
    scheduler(const scheduler&) = delete; scheduler& operator=(const scheduler&) = delete;

    static scheduler& global() { static scheduler s; return s; }
    unsigned size() const { return unsigned(workers_.size()); }
    const std::vector<worker_slot>& slots() const { return slots_; }
    bool hybrid() const { return mixed_; }
    // Index of the calling worker in this scheduler, -1 for any other thread.
    int current_worker() const { return tls_().s == this ? tls_().idx : -1; }
    stats worker_stats(unsigned i) const { auto& w = *workers_[i]; return {w.executed.load(), w.stolen.load(), w.parked.load()}; }
    stats total_stats() const { stats t; for (unsigned i = 0; i < size(); ++i) { auto s = worker_stats(i); t.executed += s.executed; t.stolen += s.stolen; t.parked += s.parked; } t.executed += ext_executed_.load(); return t; }
    unsigned pin_failures() const { return pin_fail_.load(); }
    std::size_t pending() const { return pending_.load(); }

    // Queue f() (or f(std::stop_token)). The result, or the exception, is read through the handle.
    template <class F> auto spawn(F&& f, spawn_options o = {}) { return spawn_after(std::span<const task_ref>{}, std::forward<F>(f), std::move(o)); }
    // The same, started only after every task in deps has finished (whether it succeeded or not).
    template <class F> auto spawn_after(std::span<const task_ref> deps, F&& f, spawn_options o = {}) {
        using Fd = std::decay_t<F>; using R = detail::result_of<Fd>;
        auto* n = new detail::node_f<Fd, R>(Fd(std::forward<F>(f)));
        n->stop = std::move(o.stop); n->prio = o.prio; n->place = mixed_ ? o.place : placement::any; n->owner = this;
        pending_.fetch_add(1);
        for (auto& d : deps) {
            if (!d.n_) continue;
            std::lock_guard l(d.n_->mu);
            if (d.n_->state.load() == 2) continue;
            n->deps.fetch_add(1); n->retain(); d.n_->succ.push_back(n);
        }
        task<R> h(n);
        if (n->deps.fetch_sub(1) == 1) enqueue_(n);
        return h;
    }
    template <class F> auto spawn_after(std::initializer_list<task_ref> deps, F&& f, spawn_options o = {}) {
        return spawn_after(std::span<const task_ref>(deps.begin(), deps.size()), std::forward<F>(f), std::move(o));
    }
    // Done when every task in deps is done; fails with the first (in argument order) exception among them.
    task<void> when_all(std::span<const task_ref> deps, spawn_options o = {}) {
        std::vector<task_ref> keep(deps.begin(), deps.end());
        return spawn_after(deps, [keep = std::move(keep)] { for (auto& d : keep) if (d.valid() && d.n_->err) std::rethrow_exception(d.n_->err); }, std::move(o));
    }
    task<void> when_all(std::initializer_list<task_ref> deps, spawn_options o = {}) { return when_all(std::span<const task_ref>(deps.begin(), deps.size()), std::move(o)); }
    // The values of every task, in order.
    template <class R> task<std::vector<R>> when_all(const std::vector<task<R>>& ts, spawn_options o = {}) {
        std::vector<task_ref> deps(ts.begin(), ts.end());
        return spawn_after(std::span<const task_ref>(deps), [ts] { std::vector<R> out; out.reserve(ts.size()); for (auto& t : ts) out.push_back(t.get()); return out; }, std::move(o));
    }

    // Block until t is done, running other tasks meanwhile.
    void wait(const task_ref& t) { if (t.n_) wait_until_([n = t.n_] { return n->done(); }, t.n_->waiters); }
    // Block until every spawned task has finished. Not for use inside a task (it would wait for itself).
    void wait_idle() { wait_until_([this] { return pending_.load() == 0; }, idle_waiters_); }

    // f(begin, end) over [0, n) in chunks of exactly `grain` items (the last may be shorter); chunk k is
    // [k * grain, ...) regardless of thread count. The caller runs chunks too. The first exception thrown by a
    // chunk is rethrown here after the chunks already running have finished; unstarted chunks are skipped.
    template <class F> void parallel_for(std::size_t n, std::size_t grain, F&& f, spawn_options o = {}) {
        if (n == 0) return;
        grain = std::max<std::size_t>(grain, 1);
        const std::size_t chunks = (n - 1) / grain + 1;
        if (chunks == 1) { f(std::size_t(0), n); return; }
        using Fn = std::remove_reference_t<F>;
        struct job {
            std::atomic<std::size_t> next{0}, finished{0}; std::size_t chunks = 0, n = 0, grain = 0; Fn* f = nullptr;
            std::atomic<bool> failed{false}; std::mutex em; std::exception_ptr err; std::atomic<std::uint32_t> waiters{0}; scheduler* s = nullptr;
            void work() {
                for (;;) {
                    std::size_t c = next.fetch_add(1);
                    if (c >= chunks) return;
                    if (!failed.load(std::memory_order_relaxed)) {
                        try { (*f)(c * grain, std::min(n, (c + 1) * grain)); }
                        catch (...) { std::lock_guard l(em); if (!err) err = std::current_exception(); failed.store(true); }
                    }
                    if (finished.fetch_add(1) + 1 == chunks && waiters.load() > 0) s->wake_all_();
                }
            }
        };
        auto j = std::make_shared<job>();
        j->chunks = chunks; j->n = n; j->grain = grain; j->f = std::addressof(f); j->s = this;
        std::size_t helpers = std::min<std::size_t>(chunks - 1, size() - (current_worker() >= 0 ? 1 : 0));
        for (std::size_t i = 0; i < helpers; ++i) (void)spawn([j] { j->work(); }, o);   // a helper that starts late finds no chunk left and never touches f
        j->work();
        wait_until_([&] { return j->finished.load() == chunks; }, j->waiters);
        if (j->err) std::rethrow_exception(j->err);
    }
    // acc = combine(acc, map(begin, end)) over the chunks of parallel_for, combined in chunk order starting
    // from init: the result does not depend on the thread count or on timing.
    template <class T, class Map, class Combine> T parallel_reduce(std::size_t n, std::size_t grain, T init, Map&& map, Combine&& combine, spawn_options o = {}) {
        if (n == 0) return init;
        grain = std::max<std::size_t>(grain, 1);
        std::vector<std::optional<T>> part((n - 1) / grain + 1);
        parallel_for(n, grain, [&](std::size_t b, std::size_t e) { part[b / grain].emplace(map(b, e)); }, std::move(o));
        T acc = std::move(init);
        for (auto& p : part) acc = combine(std::move(acc), std::move(*p));
        return acc;
    }

private:
    friend class task_ref;
    using node = detail::node;
    struct worker {
        detail::ws_deque<node> dq[2][3];   // [0 = any, 1 = performance-only][priority]
        std::vector<unsigned> near, far; std::uint64_t rng = 1;
        std::atomic<std::uint64_t> executed{0}, stolen{0}, parked{0};
        std::thread th;
    };
    struct tls_t { const scheduler* s = nullptr; int idx = -1; };
    static tls_t& tls_() { thread_local tls_t t; return t; }
    bool is_eff_(int w) const { return w >= 0 && slots_[std::size_t(w)].kind == os::core_kind::efficiency; }

    void wake_all_() { epoch_.fetch_add(1); park_.wake_all(epoch_); }
    void enqueue_(node* n) {
        int w = current_worker(); unsigned p = unsigned(n->prio);
        if (w >= 0 && n->place == placement::any) workers_[std::size_t(w)]->dq[0][p].push(n);
        else if (w >= 0 && n->place == placement::performance && !is_eff_(w)) workers_[std::size_t(w)]->dq[1][p].push(n);
        else {
            std::lock_guard l(inj_mu_);
            inj_[unsigned(n->place)][p].push_back(n); inj_n_[unsigned(n->place)].fetch_add(1);
        }
        if (sleepers_.load() > 0) {
            epoch_.fetch_add(1);
            if (n->place == placement::any) park_.wake_one(epoch_); else park_.wake_all(epoch_);   // a restricted task needs the right kind of worker
        }
    }
    node* inject_pop_(unsigned place, unsigned p) {
        if (inj_n_[place].load() == 0) return nullptr;
        std::lock_guard l(inj_mu_);
        auto& q = inj_[place][p];
        if (q.empty()) return nullptr;
        node* n = q.front(); q.pop_front(); inj_n_[place].fetch_sub(1);
        return n;
    }
    node* steal_from_(unsigned v, unsigned set, unsigned p) {
        for (int tries = 0; tries < 4; ++tries) { bool lost; if (node* n = workers_[v]->dq[set][p].steal(lost)) return n; if (!lost) break; }
        return nullptr;
    }
    // w = the calling worker or -1. Highest priority first: own deques, injector, then victims.
    node* find_work_(int w) {
        const bool eff = is_eff_(w);
        worker* me = w >= 0 ? workers_[std::size_t(w)].get() : nullptr;
        std::uint64_t r = 0;
        if (me) { me->rng ^= me->rng << 13; me->rng ^= me->rng >> 7; me->rng ^= me->rng << 17; r = me->rng; }
        else { thread_local std::uint64_t x = 0x2545F4914F6CDD1Dull; x ^= x << 13; x ^= x >> 7; x ^= x << 17; r = x; }
        for (int p = 2; p >= 0; --p) {
            unsigned up = unsigned(p);
            if (me) { if (!eff) if (node* n = me->dq[1][up].take()) return n; if (node* n = me->dq[0][up].take()) return n; }
            if (!eff) if (node* n = inject_pop_(1, up)) return n;
            if (eff) if (node* n = inject_pop_(2, up)) return n;
            if (node* n = inject_pop_(0, up)) return n;
            auto sweep = [&](const std::vector<unsigned>& vs) -> node* {
                for (std::size_t k = 0, m = vs.size(); k < m; ++k) {
                    unsigned v = vs[(k + r) % m];
                    if (!eff) if (node* n = steal_from_(v, 1, up)) return n;
                    if (node* n = steal_from_(v, 0, up)) return n;
                }
                return nullptr;
            };
            node* n = nullptr;
            if (me) { n = sweep(me->near); if (!n) n = sweep(me->far); }
            else { for (std::size_t k = 0, m = workers_.size(); k < m && !n; ++k) { unsigned v = unsigned((k + r) % m); n = steal_from_(v, 1, up); if (!n) n = steal_from_(v, 0, up); } }
            if (n) { if (me) me->stolen.fetch_add(1, std::memory_order_relaxed); return n; }
        }
        if (!eff) for (int p = 2; p >= 0; --p) if (node* n = inject_pop_(2, unsigned(p))) return n;   // efficiency-preferred work, when nothing else is left
        return nullptr;
    }
    bool has_work_(int w) const {
        const bool eff = is_eff_(w);
        if (inj_n_[0].load() > 0 || inj_n_[2].load() > 0 || (!eff && inj_n_[1].load() > 0)) return true;
        for (auto& x : workers_) for (int p = 0; p < 3; ++p) if (x->dq[0][p].maybe_nonempty() || (!eff && x->dq[1][p].maybe_nonempty())) return true;
        return false;
    }
    void execute_(node* n, int w) {
        if (n->stop.stop_requested()) n->err = std::make_exception_ptr(cancelled{});
        else {
            n->state.store(1);
            try { n->invoke(); } catch (...) { n->err = std::current_exception(); }
        }
        std::vector<node*> succ;
        { std::lock_guard l(n->mu); n->state.store(2); succ.swap(n->succ); }
        for (node* s : succ) { if (s->deps.fetch_sub(1) == 1) enqueue_(s); s->release(); }
        if (w >= 0) workers_[std::size_t(w)]->executed.fetch_add(1, std::memory_order_relaxed); else ext_executed_.fetch_add(1, std::memory_order_relaxed);
        bool idle = pending_.fetch_sub(1) == 1;
        if (n->waiters.load() > 0 || (idle && idle_waiters_.load() > 0)) wake_all_();
        n->release();
    }
    // Sleep unless ready() holds. The sleeper count is raised and the epoch read before ready() is evaluated;
    // whoever makes ready() true bumps the epoch afterwards if it sees a sleeper, so no wake-up is lost.
    template <class Ready> void park_if_(Ready&& ready) {
        sleepers_.fetch_add(1);
        std::uint32_t e = epoch_.load();
        if (!ready()) park_.wait(epoch_, e);
        sleepers_.fetch_sub(1);
    }
    template <class Done> void wait_until_(Done&& done, std::atomic<std::uint32_t>& waiters) {
        if (done()) return;
        const int w = current_worker();
        waiters.fetch_add(1);
        while (!done()) {
            if (node* n = find_work_(w)) { execute_(n, w); continue; }
            park_if_([&] { return done() || has_work_(w); });
        }
        waiters.fetch_sub(1);
    }
    void loop_(unsigned i, const std::vector<os::numa_node>& nodes) {
        tls_() = {this, int(i)};
        if (cfg_.pin) { if (!os::pin_thread_to({slots_[i].cpu})) pin_fail_.fetch_add(1); }
        else if (cfg_.numa_bind) {
            bool ok = false;
            for (auto& nd : nodes) if (int(nd.id) == slots_[i].node && !nd.cpus.empty()) ok = bool(os::pin_thread_to(nd.cpus));
            if (!ok) pin_fail_.fetch_add(1);
        }
        worker& me = *workers_[i];
        for (unsigned idle = 0;;) {
            if (node* n = find_work_(int(i))) {
                idle = 0;
                if (sleepers_.load() > 0 && has_work_(-1)) { epoch_.fetch_add(1); park_.wake_one(epoch_); }   // more left: pass the wake-up on
                execute_(n, int(i));
                continue;
            }
            if (stop_.load()) return;
            if (++idle < cfg_.spin) { if (idle & 15) detail::cpu_relax(); else std::this_thread::yield(); continue; }
            idle = 0;
            me.parked.fetch_add(1, std::memory_order_relaxed);
            park_if_([&] { return stop_.load() || has_work_(int(i)); });
        }
    }

    config cfg_;
    std::vector<worker_slot> slots_;
    std::vector<std::unique_ptr<worker>> workers_;
    bool mixed_ = false;
    std::mutex inj_mu_;
    std::deque<node*> inj_[3][3];              // [placement][priority]
    std::atomic<std::size_t> inj_n_[3] = {};
    alignas(64) std::atomic<std::uint32_t> epoch_{0};
    alignas(64) std::atomic<std::uint32_t> sleepers_{0};
    detail::parker park_;
    std::atomic<std::size_t> pending_{0};
    std::atomic<std::uint32_t> idle_waiters_{0};
    std::atomic<std::uint64_t> ext_executed_{0};
    std::atomic<unsigned> pin_fail_{0};
    std::atomic<bool> stop_{false};
};

inline void task_ref::wait() const { if (n_ && !n_->done()) n_->owner->wait(*this); }

template <class R> template <class F> auto task<R>::then(F&& f, spawn_options o) const {
    task<R> pred = *this;
    task_ref dep = *this;
    if constexpr (std::is_void_v<R>)
        return n_->owner->spawn_after(std::span<const task_ref>(&dep, 1), [pred, g = std::decay_t<F>(std::forward<F>(f))]() mutable { pred.get(); return g(); }, std::move(o));
    else
        return n_->owner->spawn_after(std::span<const task_ref>(&dep, 1), [pred, g = std::decay_t<F>(std::forward<F>(f))]() mutable { return g(pred.get()); }, std::move(o));
}

// One T per worker of a scheduler, each on its own cache line; a worker reaches its own with local() without
// locking. Threads that are not workers of that scheduler get a slot each on first use (under a mutex).
template <class T> class worker_local {
    struct alignas(64) cell { T v{}; };
public:
    explicit worker_local(const scheduler& s) : s_(&s), cells_(s.size()) {}
    worker_local(const scheduler& s, const T& init) : s_(&s), cells_(s.size()) { for (auto& c : cells_) c.v = init; proto_ = init; }
    T& local() {
        int w = s_->current_worker();
        if (w >= 0) return cells_[std::size_t(w)].v;
        std::lock_guard l(m_);
        auto& p = ext_[std::this_thread::get_id()];
        if (!p) { p = std::make_unique<cell>(); if (proto_) p->v = *proto_; }
        return p->v;
    }
    // Visit every slot (workers first). Call when no task is touching them.
    template <class F> void for_each(F&& f) { for (auto& c : cells_) f(c.v); std::lock_guard l(m_); for (auto& [id, p] : ext_) f(p->v); }
    std::size_t size() const { return cells_.size(); }
private:
    const scheduler* s_; std::vector<cell> cells_; std::optional<T> proto_;
    std::mutex m_; std::map<std::thread::id, std::unique_ptr<cell>> ext_;
};

// Frame-paced execution: per frame, work that must finish in the frame and work that may slip to a later one.
//   frame_pacer fp(sched, 16'000'000);            // 16 ms budget
//   fp.begin_frame();
//   fp.run([] { animate(); });                    // must complete before end_frame returns
//   fp.defer([] { stream_asset(); }, 2'000'000);  // runs only if the frame still has ~2 ms of slack
//   frame_stats st = fp.end_frame();
// end_frame waits (helping) for the frame's run() tasks, then starts deferred tasks, oldest first, in waves of
// at most one per worker while now + estimate <= frame start + budget; the rest stay queued for the next
// frame. An estimate of 0 uses a running average of observed deferred-task times. A deferred task that has
// started is never interrupted, so a frame can overrun by one deferred task's length. The clock can be
// injected (tests).
struct frame_stats { std::uint64_t frame = 0, elapsed_ns = 0, budget_ns = 0; std::size_t ran = 0, deferred_ran = 0, deferred_left = 0; bool over_budget = false; };
class frame_pacer {
public:
    using clock_fn = std::function<std::uint64_t()>;
    frame_pacer(scheduler& s, std::uint64_t budget_ns, clock_fn clock = {})
        : s_(s), budget_(budget_ns), clock_(clock ? std::move(clock) : clock_fn([] { return std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); })) {}
    void set_budget(std::uint64_t ns) { budget_ = ns; }
    void begin_frame() { t0_ = clock_(); ++frame_; must_.clear(); }
    std::uint64_t remaining_ns() const { std::uint64_t e = clock_() - t0_; return e >= budget_ ? 0 : budget_ - e; }
    template <class F> void run(F&& f, spawn_options o = {.prio = priority::high, .place = placement::performance}) { must_.push_back(s_.spawn(std::forward<F>(f), std::move(o))); }
    template <class F> void defer(F&& f, std::uint64_t est_ns = 0) { std::lock_guard l(m_); deferred_.push_back({std::function<void()>(std::forward<F>(f)), est_ns}); }
    std::size_t deferred() const { std::lock_guard l(m_); return deferred_.size(); }
    // Rethrows the first exception of a run() task, after all of them have finished.
    frame_stats end_frame() {
        frame_stats st; st.frame = frame_; st.budget_ns = budget_; st.ran = must_.size();
        for (auto& t : must_) t.wait();
        std::exception_ptr err; for (auto& t : must_) if (!err) err = t.error();
        must_.clear();
        for (;;) {
            std::vector<item> wave;
            {
                std::lock_guard l(m_);
                std::uint64_t now = clock_() - t0_;
                while (!deferred_.empty() && wave.size() < s_.size()) {
                    std::uint64_t est = deferred_.front().est ? deferred_.front().est : avg_;
                    if (now + est > budget_ || now >= budget_) break;
                    wave.push_back(std::move(deferred_.front())); deferred_.pop_front();
                }
            }
            if (wave.empty()) break;
            std::vector<task<void>> ts;
            for (auto& it : wave) ts.push_back(s_.spawn([this, fn = std::move(it.fn)] {
                std::uint64_t a = clock_(); fn(); std::uint64_t d = clock_() - a;
                std::lock_guard l(m_); avg_ = avg_ ? (avg_ * 3 + d) / 4 : d;
            }, {.prio = priority::low, .place = placement::efficiency}));
            for (auto& t : ts) { t.wait(); if (!err) err = t.error(); }
            st.deferred_ran += ts.size();
        }
        st.deferred_left = deferred();
        st.elapsed_ns = clock_() - t0_; st.over_budget = st.elapsed_ns > budget_;
        if (err) std::rethrow_exception(err);
        return st;
    }
    // Sleep out what is left of the frame (after end_frame).
    void pace() { if (auto r = remaining_ns()) std::this_thread::sleep_for(std::chrono::nanoseconds(r)); }
private:
    struct item { std::function<void()> fn; std::uint64_t est; };
    scheduler& s_; std::uint64_t budget_; clock_fn clock_;
    std::uint64_t t0_ = 0, frame_ = 0, avg_ = 0;
    std::vector<task_ref> must_;
    mutable std::mutex m_; std::deque<item> deferred_;
};

} // namespace forge::task
