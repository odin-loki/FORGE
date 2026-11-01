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
    std::string kv_codec;  // "", "f32", "int8", "fp8", "tbq4", "tbq3" or "k8v4"; empty = $FORGE_KV_CACHE_TYPE, $FORGE_KV_CODEC or f32
    int kv_offload = -1;   // 1 = keep KV on the host, drop device copies after each attention; -1 = $FORGE_KV_OFFLOAD
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
        if (auto* d = std::getenv("FORGE_DEVICE")) set_device(d);
    }
    // Select the execution device ("cpu", "vulkan"). Unavailable devices fall back to the CPU with a warning.
    // Returns the device actually used. Clears cached plans.
    std::string set_device(const std::string& dev) {
        std::lock_guard lk(run_mu_);
        std::string why;
        reg_ = flow::device_registry(dev, &why);
        if (!reg_ && !why.empty()) log::warn("{}; using cpu", why);
        device_ = reg_ ? dev : "cpu";
        cache_.clear();
        return device_;
    }
    const std::string& device() const { return device_; }
    static std::shared_ptr<model> open(const std::string& p, bool verify = false) { return std::make_shared<model>(p, verify); }

    const pack::blueprint& blueprint() const { return bp_; }
    const json::value& manifest() const { return man_; }
    const pack::reader& file() const { return r_; }
    const pack::bpe_tokenizer* tokenizer() const { return tok_ ? &*tok_ : nullptr; }
    i64 vocab() const { return vocab_; }
    i64 max_context() const { return i64(man_.num_or("max_position_embeddings", 2048)); }
    flow::plan_cache& plans() { return cache_; }
    std::mutex& run_mutex() { return run_mu_; }

    // LoRA adapters: W' = W + (alpha / r) * B @ A for each named weight (A: [r, in], B: [out, r], PEFT layout).
    // Only the targeted matrices are materialised per adapter; base weights stay mapped and shared, and each
    // adapter gets its own plans, so sessions switch adapters per request without reloading anything.
    struct lora { tensor a, b; };
    void add_adapter(const std::string& name, const std::map<std::string, lora>& deltas, float alpha) {
        if (name.empty() || adapters_.count(name)) raise(errc::invalid_argument, "adapter: empty or duplicate name '{}'", name);
        std::map<std::string, tensor> merged;
        for (auto& [w, d] : deltas) {
            tensor base = r_.load(w);
            if (d.a.shape.size() != 2 || d.b.shape.size() != 2 || base.shape.size() != 2 || d.b.shape[1] != d.a.shape[0] ||
                d.b.shape[0] != base.shape[0] || d.a.shape[1] != base.shape[1])
                raise(errc::invalid_argument, "adapter {}: shape mismatch for {}", name, w);
            i64 r = d.a.shape[0], O = base.shape[0], I = base.shape[1]; float sc = alpha / float(r);
            for (i64 o = 0; o < O; ++o) for (i64 k = 0; k < r; ++k) {
                float bk = sc * d.b.data[std::size_t(o * r + k)];
                for (i64 i = 0; i < I; ++i) base.data[std::size_t(o * I + i)] += bk * d.a.data[std::size_t(k * I + i)];
            }
            merged.emplace(w, std::move(base));
        }
        std::lock_guard lk(w_mu_);
        // Plans (possibly held by live sessions) point at the merged data, so an adapter name is immutable.
        if (!adapters_.emplace(name, std::move(merged)).second) raise(errc::invalid_argument, "adapter '{}' already loaded", name);
    }
    // Load an adapter from a .fpk holding "<weight>.lora_a" / "<weight>.lora_b" tensors; alpha from the manifest.
    void add_adapter_file(const std::string& name, const std::string& path) {
        pack::reader ar(path); std::map<std::string, lora> d;
        for (auto& [n, e] : ar.tensors()) {
            auto cut = n.rfind(".lora_"); if (cut == std::string::npos) continue;
            auto& l = d[n.substr(0, cut)]; (n.substr(cut) == ".lora_a" ? l.a : l.b) = ar.load(n);
        }
        add_adapter(name, d, float(ar.manifest().num_or("lora_alpha", 1)));
    }
    bool has_adapter(const std::string& n) const { return n.empty() || adapters_.count(n); }
    std::size_t loaded_weights() const { return weights_.size(); }

    // plan for `n` new tokens; all_rows keeps logits for every position
    flow::plan& plan_for(i64 n, bool all_rows, flow::state_store* st, const std::string& adapter = {}) {
        return cache_.get(std::format("{}:{}:{}:{}", all_rows ? "all" : "seq", n, adapter, st ? flow::kv_codec_pair_name({st->codec, st->codec_v == flow::kv_codec::f32 ? st->codec : st->codec_v}) : "f32"), [&] {
            pack::build_options bo;
            bo.dims["seq"] = n;
            bo.all_rows = all_rows;
            bo.weight = [&](const std::string& name) -> std::pair<shape_t, const float*> {
                std::lock_guard lk(w_mu_);
                if (!adapter.empty()) {
                    auto& m = adapters_.at(adapter);
                    if (auto a = m.find(name); a != m.end()) return {a->second.shape, a->second.data.data()};
                }
                auto it = weights_.find(name);
                if (it == weights_.end()) it = weights_.emplace(name, r_.load(name)).first;
                return {it->second.shape, it->second.data.data()};
            };
            // Weights stay in the mapped file: f32 tensors are used in place, other codecs are read in place by
            // quantised matmul (the plan decodes a shared f32 copy only for other consumers). FORGE_DECODE_WEIGHTS=1
            // restores the old behaviour (every weight decoded to f32 up front).
            static const bool decode_all = [] { auto* e = std::getenv("FORGE_DECODE_WEIGHTS"); return e && std::string(e) == "1"; }();
            if (!decode_all) bo.qweight = [&](const std::string& name) -> pack::build_options::encoded {
                if (!adapter.empty() && adapters_.at(adapter).count(name)) return {};
                const auto& e = r_.entry(name);
                if (e.codec == "f32") return {};  // handled by `weight` (zero-copy below)
                return {e.shape, r_.section_bytes("data").data() + e.offset, e.codec};
            };
            if (!decode_all) bo.weight = [&, inner = bo.weight](const std::string& name) -> std::pair<shape_t, const float*> {
                const auto& e = r_.entry(name);
                bool adapted = !adapter.empty() && adapters_.at(adapter).count(name);
                const auto* base = r_.section_bytes("data").data() + e.offset;
                if (!adapted && e.codec == "f32" && reinterpret_cast<std::uintptr_t>(base) % alignof(float) == 0)
                    return {e.shape, reinterpret_cast<const float*>(base)};
                return inner(name);
            };
            flow::pass_options po; po.lowerings = reg_.get();
            return flow::compile(pack::build_graph(bp_, bo), st, po);
        });
    }
private:
    pack::reader r_;
    json::value man_;
    pack::blueprint bp_;
    std::optional<pack::bpe_tokenizer> tok_;
    flow::plan_cache cache_;
    std::map<std::string, tensor> weights_;
    std::map<std::string, std::map<std::string, tensor>> adapters_;
    std::mutex w_mu_, run_mu_;
    std::unique_ptr<flow::registry> reg_;
    std::string device_ = "cpu";
    i64 vocab_ = 0;
};

class session {
public:
    explicit session(const std::string& fpk_path, session_options o = {}) : session(model::open(fpk_path, o.verify), o) {}
    explicit session(std::shared_ptr<model> m, session_options o = {}) : m_(std::move(m)), opt_(o) {
        st_.capacity = o.context ? o.context : m_->max_context();
        st_.page_rows = o.page_rows;
        std::string kc = o.kv_codec;
        if (kc.empty()) if (const char* e = std::getenv("FORGE_KV_CACHE_TYPE")) kc = e;
        if (kc.empty()) if (const char* e = std::getenv("FORGE_KV_CODEC")) kc = e;
        auto kp = flow::kv_codec_pair_from_name(kc);
        st_.codec = kp.k; st_.codec_v = kp.v != kp.k ? kp.v : flow::kv_codec::f32;
        int off = o.kv_offload;
        if (off < 0) { const char* e = std::getenv("FORGE_KV_OFFLOAD"); off = e && *e && *e != '0'; }
        st_.offload = off != 0;
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
    // Select a LoRA adapter registered on the model ("" = base). The KV cache is cleared, since cached
    // keys/values were computed with the previous weights.
    void set_adapter(const std::string& a) {
        if (!m_->has_adapter(a)) raise(errc::not_found, "adapter '{}' not loaded", a);
        if (a != adapter_) { adapter_ = a; execs_.clear(); reset(); }
    }
    const std::string& adapter() const { return adapter_; }
    // roll back to `pos` tokens (speculative rejection, prefix reuse)
    void rewind(i64 pos) { if (pos < pos_) { st_.truncate(pos); pos_ = pos; } }
    // copy the first `n` tokens of another session's state (prefix sharing across sequences)
    void copy_prefix(const session& o, i64 n) {
        if (o.m_ != m_) raise(errc::invalid_argument, "copy_prefix: sessions use different models");
        n = std::min(n, o.pos_);
        if (o.st_.codec != st_.codec) raise(errc::invalid_argument, "copy_prefix: sessions use different KV codecs");
        o.st_.flush(); st_.forget();
        st_.reset();
        for (auto& [k, s] : o.st_.slots) {
            auto& d = st_.slots[k];
            if (s.fixed) {   // recurrent state: only a full-length copy is meaningful
                if (n != s.len) raise(errc::unsupported, "copy_prefix: recurrent slot '{}' holds {} tokens, cannot share {}", k, s.len, n);
                d = s; continue;
            }
            if (!d.capacity) d.codec = s.codec;
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

    // one_shot graphs (e.g. imported ONNX): bind named f32 inputs, run once, return every declared output.
    std::map<std::string, std::vector<float>> infer(const std::map<std::string, std::vector<float>>& in, i64 seq = 1) {
        std::lock_guard lk(m_->run_mutex());
        auto& p = m_->plan_for(seq, false, &st_);
        auto it = execs_.find(&p);
        if (it == execs_.end()) it = execs_.emplace(&p, std::make_unique<flow::executor>(p, &st_)).first;
        auto& ex = *it->second;
        for (auto& d : blueprint().inputs)
            if (!in.count(d.name)) raise(errc::invalid_argument, "infer: missing input '{}'", d.name);
        for (auto& [k, v] : in) ex.set_input(k, v);
        ex.run();
        std::map<std::string, std::vector<float>> out;
        for (auto& o : blueprint().outputs) { auto v = ex.output_view(o.name); out[o.name].assign(v.data, v.data + v.size()); }
        return out;
    }
private:
    std::vector<float> run(std::span<const int> ids, bool all) {
        if (ids.empty()) raise(errc::invalid_argument, "forward: no tokens");
        if (pos_ + i64(ids.size()) > st_.capacity) raise(errc::out_of_memory, "context {} exceeded", st_.capacity);
        i64 n = i64(ids.size());
        std::lock_guard lk(m_->run_mutex());
        auto& p = m_->plan_for(n, all, &st_, adapter_);
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
    std::string adapter_;
};

} // namespace forge::mind
