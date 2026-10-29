#pragma once
// task: a small shared work pool with parallel_for (one scheduler for every layer; no oversubscription).
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace forge::task {

class pool {
public:
    explicit pool(unsigned n = 0) {
        if (!n) n = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 1; i < n; ++i) workers_.emplace_back([this] { loop(); });
        size_ = n;
    }
    ~pool() {
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

    static pool& global() { static pool p; return p; }

private:
    void loop() {
        std::size_t seen = 0;
        std::unique_lock l(m_);
        for (;;) {
            cv_.wait(l, [&] { return stop_ || (gen_ != seen && job_); });
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
    std::vector<std::thread> workers_;
    std::mutex m_;
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

} // namespace forge::task
