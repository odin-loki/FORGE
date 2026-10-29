#pragma once
// Generic inference: the .fpk's blueprint declares the session kind; the engine attaches services.
#include <forge/flow/flow.hpp>
#include <forge/mind/sampler.hpp>
#include <forge/pack/blueprint.hpp>
#include <forge/pack/fpk.hpp>
#include <forge/pack/tokenizer.hpp>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>

namespace forge::mind {

struct session_options { i64 context = 0; bool verify = false; };

struct generate_stats { std::size_t prompt_tokens = 0, new_tokens = 0; double prefill_s = 0, decode_s = 0; };

class session {
public:
    explicit session(const std::string& fpk_path, session_options o = {}) : r_(fpk_path), opt_(o) {
        if (o.verify) {
            auto bad = r_.verify();
            if (!bad.empty()) raise(errc::integrity_error, "{}: hash mismatch in {}", fpk_path, bad.front());
        }
        man_ = r_.manifest();
        bp_ = pack::parse_blueprint(r_.section_text("blueprint"));
        if (r_.has_section("tokenizer")) {
            tok_.emplace(r_.section_text("tokenizer"));
            tok_->set_special(int(man_.num_or("bos_token_id", -1)), int(man_.num_or("eos_token_id", -1)));
        }
        i64 ctx = o.context ? o.context : i64(man_.num_or("max_position_embeddings", 2048));
        st_.capacity = ctx;
    }
    const pack::blueprint& blueprint() const { return bp_; }
    const json::value& manifest() const { return man_; }
    const pack::reader& file() const { return r_; }
    const pack::bpe_tokenizer* tokenizer() const { return tok_ ? &*tok_ : nullptr; }
    i64 position() const { return pos_; }
    i64 context() const { return st_.capacity; }
    void reset() { st_.reset(); pos_ = 0; }
    flow::plan_cache& plans() { return cache_; }

    // Runs the graph on `ids` continuing from the current position; returns logits of the last position.
    std::vector<float> forward(std::span<const int> ids) {
        if (ids.empty()) raise(errc::invalid_argument, "forward: no tokens");
        if (pos_ + i64(ids.size()) > st_.capacity) raise(errc::out_of_memory, "context {} exceeded", st_.capacity);
        i64 n = i64(ids.size());
        auto& p = cache_.get(std::format("seq:{}", n), [&] {
            pack::build_options bo;
            bo.dims["seq"] = n;
            bo.weight = [&](const std::string& name) -> std::pair<shape_t, const float*> {
                auto it = weights_.find(name);
                if (it == weights_.end()) it = weights_.emplace(name, r_.load(name)).first;
                return {it->second.shape, it->second.data.data()};
            };
            return flow::compile(pack::build_graph(bp_, bo), &st_);
        });
        auto& ex = exec_for(p);
        std::vector<float> tk(ids.begin(), ids.end()); std::vector<float> ps(static_cast<std::size_t>(n));
        for (i64 i = 0; i < n; ++i) ps[std::size_t(i)] = float(pos_ + i);
        ex.set_input("tokens", tk);
        ex.set_input("positions", ps);
        ex.run();
        pos_ += n;
        auto v = ex.output_view(bp_.outputs.front().name);
        return {v.data, v.data + v.size()};
    }

    // generative_sequence: prefill + decode loop with a sampler chain. `on_token` returning false stops.
    std::vector<int> generate(std::span<const int> prompt, int max_new, sampler& s,
                              const std::function<bool(int)>& on_token = {}, generate_stats* stats = nullptr) {
        if (bp_.session != "generative_sequence") raise(errc::unsupported, "session kind '{}' cannot generate", bp_.session);
        std::vector<int> hist(prompt.begin(), prompt.end()), out;
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now();
        auto logits = forward(prompt);
        auto t1 = clk::now();
        int eos = tok_ ? tok_->eos() : -1;
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

private:
    flow::executor& exec_for(flow::plan& p) {
        auto it = execs_.find(&p);
        if (it == execs_.end()) it = execs_.emplace(&p, std::make_unique<flow::executor>(p, &st_)).first;
        return *it->second;
    }
    pack::reader r_;
    session_options opt_;
    json::value man_;
    pack::blueprint bp_;
    std::optional<pack::bpe_tokenizer> tok_;
    flow::state_store st_;
    flow::plan_cache cache_;
    std::map<flow::plan*, std::unique_ptr<flow::executor>> execs_;
    std::map<std::string, tensor> weights_;
    i64 pos_ = 0;
};

} // namespace forge::mind
