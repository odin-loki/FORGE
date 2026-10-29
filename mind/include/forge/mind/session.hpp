#pragma once
// Generic inference: the .fpk's blueprint declares the session kind; the engine attaches services.
//
// model   : the immutable, shareable part (mapped file, blueprint, tokenizer, decoded weights, plans).
// session : one sequence's mutable state (KV state store, position). Many sessions may share a model;
//           runs are serialised on the model (plans own a scratch arena), state is per session.
#include <forge/flow/flow.hpp>
#include <forge/mind/sampler.hpp>
#include <forge/pack/blueprint.hpp>
#include <forge/pack/fpk.hpp>
#include <forge/pack/sign.hpp>
#include <forge/pack/tokenizer.hpp>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>

namespace forge::mind {

struct session_options {
    i64 context = 0;       // 0 = manifest max_position_embeddings
    bool verify = false;   // check BLAKE3 hashes on open
    i64 page_rows = 256;   // KV page size in rows; 0 = reserve the whole context up front
};

struct generate_stats { std::size_t prompt_tokens = 0, new_tokens = 0; double prefill_s = 0, decode_s = 0; };

class model {
public:
    explicit model(const std::string& fpk_path, bool verify = false) : r_(fpk_path) {
        // Strict mode (FORGE_PLAN: "verify before load; refuse unsigned files"): FORGE_REQUIRE_SIGNED=1
        // and/or FORGE_TRUSTED_KEYS=<hex pubkey>[,<hex pubkey>...] require a valid (trusted) signature.
        std::vector<std::string> trusted;
        if (auto* e = std::getenv("FORGE_TRUSTED_KEYS")) {
            std::string s = e;
            for (std::size_t p = 0, q; p < s.size(); p = q + 1) { q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) trusted.push_back(s.substr(p, q - p)); }
        }
        bool strict = !trusted.empty() || (std::getenv("FORGE_REQUIRE_SIGNED") && std::string(std::getenv("FORGE_REQUIRE_SIGNED")) == "1");
        if (verify || strict) {
            auto bad = r_.verify();
            if (!bad.empty()) raise(errc::integrity_error, "{}: hash mismatch in {}", fpk_path, bad.front());
        }
        if (strict || (verify && r_.has_section("signature"))) {
            auto rep = pack::verify_signature(r_, trusted);
            if (!rep.trusted && (strict || rep.signed_)) raise(errc::integrity_error, "{}: signature check failed: {}", fpk_path, rep.message);
        }
        man_ = r_.manifest();
        bp_ = pack::parse_blueprint(r_.section_text("blueprint"));
        if (r_.has_section("tokenizer")) {
            tok_.emplace(r_.section_text("tokenizer"));
            tok_->set_special(int(man_.num_or("bos_token_id", -1)), int(man_.num_or("eos_token_id", -1)));
        }
        vocab_ = i64(bp_.param_or("vocab", 0));
    }
    static std::shared_ptr<model> open(const std::string& p, bool verify = false) { return std::make_shared<model>(p, verify); }

    const pack::blueprint& blueprint() const { return bp_; }
    const json::value& manifest() const { return man_; }
    const pack::reader& file() const { return r_; }
    const pack::bpe_tokenizer* tokenizer() const { return tok_ ? &*tok_ : nullptr; }
    i64 vocab() const { return vocab_; }
    i64 max_context() const { return i64(man_.num_or("max_position_embeddings", 2048)); }
    flow::plan_cache& plans() { return cache_; }
    std::mutex& run_mutex() { return run_mu_; }

    // plan for `n` new tokens; all_rows keeps logits for every position
    flow::plan& plan_for(i64 n, bool all_rows, flow::state_store* st) {
        return cache_.get(std::format("{}:{}", all_rows ? "all" : "seq", n), [&] {
            pack::build_options bo;
            bo.dims["seq"] = n;
            bo.all_rows = all_rows;
            bo.weight = [&](const std::string& name) -> std::pair<shape_t, const float*> {
                std::lock_guard lk(w_mu_);
                auto it = weights_.find(name);
                if (it == weights_.end()) it = weights_.emplace(name, r_.load(name)).first;
                return {it->second.shape, it->second.data.data()};
            };
            return flow::compile(pack::build_graph(bp_, bo), st);
        });
    }
private:
    pack::reader r_;
    json::value man_;
    pack::blueprint bp_;
    std::optional<pack::bpe_tokenizer> tok_;
    flow::plan_cache cache_;
    std::map<std::string, tensor> weights_;
    std::mutex w_mu_, run_mu_;
    i64 vocab_ = 0;
};

class session {
public:
    explicit session(const std::string& fpk_path, session_options o = {}) : session(model::open(fpk_path, o.verify), o) {}
    explicit session(std::shared_ptr<model> m, session_options o = {}) : m_(std::move(m)), opt_(o) {
        st_.capacity = o.context ? o.context : m_->max_context();
        st_.page_rows = o.page_rows;
    }
    const std::shared_ptr<model>& shared_model() const { return m_; }
    const pack::blueprint& blueprint() const { return m_->blueprint(); }
    const json::value& manifest() const { return m_->manifest(); }
    const pack::reader& file() const { return m_->file(); }
    const pack::bpe_tokenizer* tokenizer() const { return m_->tokenizer(); }
    i64 vocab() const { return m_->vocab(); }
    i64 position() const { return pos_; }
    i64 context() const { return st_.capacity; }
    const flow::state_store& state() const { return st_; }
    void reset() { st_.reset(); pos_ = 0; }
    // roll back to `pos` tokens (speculative rejection, prefix reuse)
    void rewind(i64 pos) { if (pos < pos_) { st_.truncate(pos); pos_ = pos; } }
    // copy the first `n` tokens of another session's state (prefix sharing across sequences)
    void copy_prefix(const session& o, i64 n) {
        if (o.m_ != m_) raise(errc::invalid_argument, "copy_prefix: sessions use different models");
        n = std::min(n, o.pos_);
        st_.reset();
        for (auto& [k, s] : o.st_.slots) {
            auto& d = st_.slots[k];
            shape_t row = s.buf.shape;
            st_.ensure(d, row, std::max<i64>(n, 1));
            i64 rw = numel(s.buf.shape) / std::max<i64>(s.capacity, 1);
            std::copy_n(s.buf.data.data(), std::size_t(rw * n), d.buf.data.data());
            d.len = n;
        }
        pos_ = n;
    }
    flow::plan_cache& plans() { return m_->plans(); }

    // Runs the graph on `ids` continuing from the current position; returns logits of the last position.
    std::vector<float> forward(std::span<const int> ids) { return run(ids, false); }
    // Same, but returns logits for every input position ([ids.size(), vocab] row-major).
    std::vector<float> forward_all(std::span<const int> ids) { return run(ids, true); }

    // generative_sequence: prefill + decode loop with a sampler chain. `on_token` returning false stops.
    std::vector<int> generate(std::span<const int> prompt, int max_new, sampler& s,
                              const std::function<bool(int)>& on_token = {}, generate_stats* stats = nullptr) {
        check_generative();
        std::vector<int> hist(prompt.begin(), prompt.end()), out;
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now();
        auto logits = forward(prompt);
        auto t1 = clk::now();
        int eos = tokenizer() ? tokenizer()->eos() : -1;
        for (int i = 0; i < max_new; ++i) {
            int t = s.sample(logits, hist);
            out.push_back(t); hist.push_back(t);
            if (on_token && !on_token(t)) break;
            if (t == eos) break;
            if (i + 1 == max_new || pos_ >= st_.capacity) break;
            logits = forward(std::span<const int>(&t, 1));
        }
        auto t2 = clk::now();
        if (stats) *stats = {prompt.size(), out.size(), std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count()};
        return out;
    }
    void check_generative() const {
        if (blueprint().session != "generative_sequence") raise(errc::unsupported, "session kind '{}' cannot generate", blueprint().session);
    }

private:
    std::vector<float> run(std::span<const int> ids, bool all) {
        if (ids.empty()) raise(errc::invalid_argument, "forward: no tokens");
        if (pos_ + i64(ids.size()) > st_.capacity) raise(errc::out_of_memory, "context {} exceeded", st_.capacity);
        i64 n = i64(ids.size());
        std::lock_guard lk(m_->run_mutex());
        auto& p = m_->plan_for(n, all, &st_);
        auto it = execs_.find(&p);
        if (it == execs_.end()) it = execs_.emplace(&p, std::make_unique<flow::executor>(p, &st_)).first;
        auto& ex = *it->second;
        std::vector<float> tk(ids.begin(), ids.end()); std::vector<float> ps(static_cast<std::size_t>(n));
        for (i64 i = 0; i < n; ++i) ps[std::size_t(i)] = float(pos_ + i);
        ex.set_input("tokens", tk);
        ex.set_input("positions", ps);
        ex.run();
        pos_ += n;
        auto v = ex.output_view(blueprint().outputs.front().name);
        return {v.data, v.data + v.size()};
    }
    std::shared_ptr<model> m_;
    session_options opt_;
    flow::state_store st_;
    std::map<flow::plan*, std::unique_ptr<flow::executor>> execs_;
    i64 pos_ = 0;
};

} // namespace forge::mind
