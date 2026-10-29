#pragma once
// Continuous batching scheduler: many requests share one model; each step admits waiting requests up to
// `max_active`, prefills newcomers in chunks and advances every active sequence by one token, so short
// requests finish and free their KV pages while long ones keep decoding (iteration-level scheduling).
// Finished sequences are kept in a small prefix cache: a new prompt that shares a prefix with a cached
// sequence copies those KV rows instead of recomputing them.
#include <forge/mind/grammar.hpp>
#include <forge/mind/session.hpp>
#include <condition_variable>
#include <atomic>
#include <deque>
#include <list>
#include <thread>

namespace forge::mind {

struct request {
    std::vector<int> prompt;
    int max_new = 64;
    sampler_config sampling;
    std::shared_ptr<const forge::mind::grammar> grammar;    // optional constraint
    std::vector<int> stop_tokens;              // in addition to EOS
    int priority = 0;                          // higher is admitted first; FIFO within a priority
    std::function<bool(int)> on_token;         // streaming; return false to cancel
    std::function<void(const std::vector<int>&, const std::string& finish_reason)> on_done;
};

struct batch_options { int max_active = 8; i64 prefill_chunk = 256; std::size_t prefix_cache = 4; session_options session; };

struct batch_stats { std::size_t steps = 0, completed = 0, prefix_hits = 0, prefix_tokens_reused = 0, peak_active = 0, peak_kv_bytes = 0; };

class batch_scheduler {
public:
    batch_scheduler(std::shared_ptr<model> m, batch_options o = {}) : m_(std::move(m)), o_(o) {}
    ~batch_scheduler() { stop(); }

    std::uint64_t submit(request r) {
        std::lock_guard lk(mu_);
        auto id = ++next_id_;
        auto pos = std::find_if(waiting_.begin(), waiting_.end(), [&](const pending& w) { return w.r.priority < r.priority; });
        waiting_.insert(pos, {id, std::move(r)});
        cv_.notify_all();
        return id;
    }
    // Cancel a waiting or active request; its on_done gets finish_reason "cancelled". False if unknown/finished.
    bool cancel(std::uint64_t id) {
        std::lock_guard lk(mu_);
        for (auto it = waiting_.begin(); it != waiting_.end(); ++it)
            if (it->id == id) { auto r = std::move(it->r); waiting_.erase(it); ++stats_.completed; if (r.on_done) r.on_done({}, "cancelled"); return true; }
        for (auto& a : active_) if (a->id == id && !a->cancelled) { a->cancelled = true; return true; }
        return false;
    }
    bool idle() { std::lock_guard lk(mu_); return waiting_.empty() && active_.empty(); }
    const batch_stats& stats() const { return stats_; }

    // One scheduling iteration. Returns false when there was nothing to do.
    // Memory pressure: drop cached prefix sessions (their KV state) down to `keep`, applied on the worker at the
    // next step. Safe from any thread (e.g. a forge::os::memory_pressure callback). Returns immediately.
    void request_kv_eviction(std::size_t keep = 0) { evict_keep_.store(keep); evict_.store(true); std::lock_guard lk(mu_); cv_.notify_all(); }
    std::size_t evictions() const { return evictions_.load(); }

    bool step() {
        {
            std::lock_guard lk(mu_);
            if (evict_.exchange(false)) { auto k = evict_keep_.load(); while (cache_.size() > k) { cache_.pop_back(); ++evictions_; } }
            while (!waiting_.empty() && int(active_.size()) < o_.max_active) {
                auto w = std::move(waiting_.front()); waiting_.pop_front();
                active_.push_back(admit(w.id, std::move(w.r)));
            }
            if (active_.empty()) return false;
        }
        stats_.peak_active = std::max(stats_.peak_active, active_.size());
        std::size_t kv = 0;
        for (auto it = active_.begin(); it != active_.end();) {
            auto& a = **it;
            bool fin;
            { std::lock_guard lk(mu_); fin = a.cancelled; }
            if (fin) finish(a, "cancelled"); else fin = advance(a);
            kv += a.s->state().bytes();
            if (fin) { retire(std::move(*it)); it = active_.erase(it); }
            else ++it;
        }
        stats_.peak_kv_bytes = std::max(stats_.peak_kv_bytes, kv);
        ++stats_.steps;
        return true;
    }
    void run_until_idle() { while (step()) {} }

    // background worker (forge-serve)
    void start() {
        if (worker_.joinable()) return;
        quit_ = false;
        worker_ = std::thread([this] {
            for (;;) {
                {
                    std::unique_lock lk(mu_);
                    cv_.wait(lk, [&] { return quit_ || evict_.load() || !waiting_.empty() || !active_.empty(); });
                    if (quit_) return;
                }
                try { step(); } catch (const std::exception& e) { fail_all(e.what()); }
            }
        });
    }
    void stop() {
        { std::lock_guard lk(mu_); quit_ = true; cv_.notify_all(); }
        if (worker_.joinable()) worker_.join();
    }

private:
    struct pending { std::uint64_t id; request r; };
    struct active {
        std::uint64_t id; request r;
        std::unique_ptr<session> s;
        std::unique_ptr<sampler> smp;
        std::unique_ptr<grammar_constraint> gc;
        std::vector<int> hist, out;
        std::size_t fed = 0;           // prompt tokens already in the KV state
        std::vector<float> logits;
        bool cancelled = false;
    };
    std::unique_ptr<active> admit(std::uint64_t id, request r) {
        auto a = std::make_unique<active>();
        a->id = id; a->r = std::move(r);
        a->s = std::make_unique<session>(m_, o_.session);
        a->smp = std::make_unique<sampler>(a->r.sampling);
        if (a->r.grammar) {
            auto* tok = m_->tokenizer();
            if (!tok) raise(errc::unsupported, "grammar constraints need a tokenizer");
            a->gc = std::make_unique<grammar_constraint>(a->r.grammar, *tok, int(m_->vocab()), tok->eos());
            a->smp->filter = [g = a->gc.get()](std::span<float> l) { g->apply(l); };
        }
        if (a->r.prompt.empty()) raise(errc::invalid_argument, "request with empty prompt");
        a->hist = a->r.prompt;
        // prefix cache: reuse the longest shared prefix (always leave >= 1 prompt token to compute logits)
        std::size_t best = 0; session* src = nullptr;
        for (auto& c : cache_) {
            std::size_t n = 0, lim = std::min(c.tokens.size(), a->r.prompt.size() - 1);
            while (n < lim && c.tokens[n] == a->r.prompt[n]) ++n;
            if (n > best) { best = n; src = c.s.get(); }
        }
        if (src && best > 0) {
            a->s->copy_prefix(*src, i64(best)); a->fed = best;
            ++stats_.prefix_hits; stats_.prefix_tokens_reused += best;
        }
        return a;
    }
    // returns true when finished
    bool advance(active& a) {
        if (a.fed < a.r.prompt.size()) {   // chunked prefill
            std::size_t n = std::min<std::size_t>(a.r.prompt.size() - a.fed, std::size_t(o_.prefill_chunk));
            a.logits = a.s->forward(std::span<const int>(a.r.prompt.data() + a.fed, n));
            a.fed += n;
            if (a.fed < a.r.prompt.size()) return false;
        }
        int t = a.smp->sample(a.logits, a.hist);
        if (a.gc && !a.gc->accept(t)) { finish(a, "grammar"); return true; }
        a.out.push_back(t); a.hist.push_back(t);
        int eos = m_->tokenizer() ? m_->tokenizer()->eos() : -1;
        bool stop = t == eos || std::find(a.r.stop_tokens.begin(), a.r.stop_tokens.end(), t) != a.r.stop_tokens.end();
        if (a.r.on_token && !a.r.on_token(t)) { finish(a, "cancelled"); return true; }
        if (stop) { finish(a, "stop"); return true; }
        if (a.gc && a.gc->can_end() && a.gc->finished()) { finish(a, "stop"); return true; }
        if (int(a.out.size()) >= a.r.max_new || a.s->position() >= a.s->context()) { finish(a, "length"); return true; }
        a.logits = a.s->forward(std::span<const int>(&t, 1));
        return false;
    }
    void finish(active& a, const std::string& why) { ++stats_.completed; if (a.r.on_done) a.r.on_done(a.out, why); }
    void retire(std::unique_ptr<active> a) {
        if (o_.prefix_cache == 0) return;
        std::vector<int> toks(a->hist.begin(), a->hist.begin() + std::min<std::size_t>(a->hist.size(), std::size_t(a->s->position())));
        cache_.push_front({std::move(toks), std::move(a->s)});
        while (cache_.size() > o_.prefix_cache) cache_.pop_back();
    }
    void fail_all(const std::string& msg) {
        std::lock_guard lk(mu_);
        for (auto& a : active_) if (a->r.on_done) a->r.on_done(a->out, "error: " + msg);
        active_.clear();
    }
    struct cached { std::vector<int> tokens; std::unique_ptr<session> s; };
    std::shared_ptr<model> m_;
    batch_options o_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<pending> waiting_;
    std::list<std::unique_ptr<active>> active_;
    std::list<cached> cache_;
    std::uint64_t next_id_ = 0;
    batch_stats stats_;
    std::thread worker_;
    bool quit_ = false;
    std::atomic<bool> evict_{false};
    std::atomic<std::size_t> evict_keep_{0}, evictions_{0};
};

} // namespace forge::mind
