#pragma once
// task: a small shared work pool (one scheduler for every layer; no oversubscription).
//   parallel_for(n, grain, f)          data-parallel loop; the caller participates.
//   submit(f, {priority, stop_token})  queued task -> std::future. Higher priority runs first, FIFO within a
//                                      priority. A task whose stop_token is triggered before it starts is
//                                      skipped and its future throws task::cancelled; a running task sees the
//                                      token if f takes a std::stop_token. parallel_for jobs take precedence.
#include <cstdlib>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <queue>
#include <stdexcept>
#include <stop_token>
#include <type_traits>
#include <mutex>
#include <thread>
#include <vector>

namespace forge::task {

struct cancelled : std::runtime_error { cancelled() : std::runtime_error("task cancelled before it started") {} };
struct options { int priority = 0; std::stop_token stop{}; };

class pool {
public:
    explicit pool(unsigned n = 0) {
        if (!n) if (const char* e = std::getenv("FORGE_THREADS")) n = unsigned(std::strtoul(e, nullptr, 10));   // cap for constrained hosts/emulators
        if (!n) n = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 1; i < n; ++i) workers_.emplace_back([this] { loop(); });
        size_ = n;
    }
    ~pool() {   // queued tasks still run (or are skipped if cancelled) before the workers exit
        if (workers_.empty()) { std::unique_lock l(m_); drain_inline_(l); }
        { std::lock_guard l(m_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    unsigned size() const { return size_; }

    // Calls f(begin, end) over [0, n) split into chunks of at least `grain`. Caller participates.
    void parallel_for(std::size_t n, std::size_t grain, const std::function<void(std::size_t, std::size_t)>& f) {
        if (n == 0) return;
        std::size_t chunks = std::min<std::size_t>((n + grain - 1) / std::max<std::size_t>(grain, 1), size_ * 4);
        if (chunks <= 1 || size_ == 1) { f(0, n); return; }
        std::size_t step = (n + chunks - 1) / chunks;
        std::atomic<std::size_t> next{0}, done{0};
        auto body = [&] {
            for (;;) {
                std::size_t c = next.fetch_add(1);
                if (c >= chunks) break;
                std::size_t b = c * step, e = std::min(n, b + step);
                if (b < e) f(b, e);
                done.fetch_add(1, std::memory_order_release);
            }
        };
        {
            std::lock_guard l(m_);
            job_ = body; ++gen_;
        }
        cv_.notify_all();
        body();
        while (done.load(std::memory_order_acquire) < chunks) std::this_thread::yield();
        std::lock_guard l(m_);
        job_ = nullptr;
        while (active_) { m_.unlock(); std::this_thread::yield(); m_.lock(); }
    }

    // Queue f; returns its future. With a single-thread pool (no workers) the task runs inline at submit,
    // after any higher-or-equal priority tasks already queued.
    template <class F> auto submit(F&& f, options o = {}) {
        using R = std::conditional_t<std::is_invocable_v<F&, std::stop_token>, std::invoke_result<F&, std::stop_token>, std::invoke_result<F&>>::type;
        auto pr = std::make_shared<std::promise<R>>();
        std::future<R> fut = pr->get_future();
        item it{o.priority, 0, [fn = std::make_shared<std::decay_t<F>>(std::forward<F>(f)), pr, st = o.stop]() {
            if (st.stop_requested()) { pr->set_exception(std::make_exception_ptr(cancelled{})); return; }
            try {
                if constexpr (std::is_void_v<R>) {
                    if constexpr (std::is_invocable_v<F&, std::stop_token>) (*fn)(st); else (*fn)();
                    pr->set_value();
                } else {
                    if constexpr (std::is_invocable_v<F&, std::stop_token>) pr->set_value((*fn)(st)); else pr->set_value((*fn)());
                }
            } catch (...) { pr->set_exception(std::current_exception()); }
        }};
        {
            std::unique_lock l(m_);
            it.seq = seq_++; tasks_.push(std::move(it)); ++queued_;
            if (workers_.empty()) { drain_inline_(l); return fut; }
        }
        cv_.notify_one();
        return fut;
    }
    std::size_t pending() const { std::lock_guard l(m_); return queued_; }
    // Block until every task queued so far has finished (or been skipped).
    void wait_idle() { std::unique_lock l(m_); idle_cv_.wait(l, [&] { return queued_ == 0 && running_tasks_ == 0; }); }

    static pool& global() { static pool p; return p; }

private:
    void loop() {
        std::size_t seen = 0;
        std::unique_lock l(m_);
        for (;;) {
            cv_.wait(l, [&] { return stop_ || (gen_ != seen && job_) || !tasks_.empty(); });
            if (!(gen_ != seen && job_) && !tasks_.empty()) {   // a queued task (parallel_for jobs go first)
                auto t = tasks_.top().fn; tasks_.pop(); --queued_; ++running_tasks_;
                l.unlock(); t(); l.lock();
                if (--running_tasks_ == 0 && queued_ == 0) idle_cv_.notify_all();
                continue;
            }
            if (stop_) return;
            seen = gen_;
            auto j = job_;
            ++active_;
            l.unlock();
            j();
            l.lock();
            --active_;
        }
    }
    struct item {
        int priority; std::uint64_t seq; std::function<void()> fn;
        bool operator<(const item& o) const { return priority != o.priority ? priority < o.priority : seq > o.seq; }
    };
    void drain_inline_(std::unique_lock<std::mutex>& l) {
        while (!tasks_.empty()) { auto t = tasks_.top().fn; tasks_.pop(); --queued_; l.unlock(); t(); l.lock(); }
        idle_cv_.notify_all();
    }
    std::priority_queue<item> tasks_;
    std::uint64_t seq_ = 0;
    std::size_t queued_ = 0, running_tasks_ = 0;
    std::condition_variable idle_cv_;
    std::vector<std::thread> workers_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::function<void()> job_;
    std::size_t gen_ = 0;
    int active_ = 0;
    bool stop_ = false;
    unsigned size_ = 1;
};

inline void parallel_for(std::size_t n, std::size_t grain, const std::function<void(std::size_t, std::size_t)>& f) {
    pool::global().parallel_for(n, grain, f);
}
template <class F> auto submit(F&& f, options o = {}) { return pool::global().submit(std::forward<F>(f), std::move(o)); }

} // namespace forge::task
