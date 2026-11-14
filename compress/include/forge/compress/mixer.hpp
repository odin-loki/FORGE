// Online context mixing (in the spirit of PAQ / cmix) of a network's next-token distribution with cheap adaptive
// models of the text seen so far. Shared by evaluation (ContextMixer in eval.hpp: log p_mix of the observed token)
// and generation (StreamMixer: the FULL mixed next-token distribution, used by `forge run --mix 1` through the
// sampler's logit filter).
//
// Components:
//   0 the network          1 the network sharpened (logits x 1.25)     2 the network flattened (logits x 0.8)
//   3 unigram cache        4 bigram cache                               5 match model (token after the most recent
//                                                                         earlier occurrence of the last 3 tokens)
// A component with nothing to say (empty cache, no match) falls back to the network, so the mixture is always a
// proper distribution. Weight sets are selected by context (match-length bucket x bigram seen), as PAQ selects mixer
// weights by context; updates are Bayesian (w_i <- w_i * (p_i(y) / p(y))^eta) with fixed-share smoothing (alpha) so
// the mixer keeps adapting. Only past tokens are used: a valid online predictor.
//
// The full distribution is the dense part (w0 p0 + w1 p1 + w2 p2 + the weight of every component that falls back,
// on p0) plus sparse parts over the tokens the caches have seen (unigram counts, the successors of the previous token,
// the match token): O(V) for the three softmaxes, O(seen) for the rest.
#pragma once
#include <forge/compress/core.hpp>
#include <forge/inference/session.hpp>

#include <array>
#include <cmath>
#include <span>
#include <unordered_map>
#include <vector>

namespace forge::compress {

struct MixModels {
    static constexpr int K = 6, SETS = 8;
    static constexpr double scale[3] = {1.0, 1.25, 0.8};   // logit scales of components 0..2
    double eta = 1.0, alpha = 0.002;
    std::array<std::array<double, K>, SETS> w{};
    std::unordered_map<int, int> uni;
    std::unordered_map<std::uint64_t, std::pair<int, int>> bi;   // (prev,next) count, prev count under key prev<<32|~0
    std::unordered_map<int, std::vector<int>> succ;               // prev -> distinct successors seen (sparse bigram)
    std::unordered_map<std::uint64_t, i64> ctx3;                  // hash of 3 tokens -> last position after them
    std::vector<int> hist;
    std::array<bool, SETS> used{};   // weight sets that have learned

    MixModels() { reset(); }
    void reset() {
        for (auto& ws : w) ws = {0.94, 0.01, 0.01, 0.01, 0.01, 0.02};
        used.fill(false);
        uni.clear(); bi.clear(); succ.clear(); ctx3.clear(); hist.clear();
    }
    static std::uint64_t key3(const int* p) { return (std::uint64_t(std::uint32_t(p[0])) * 0x9E3779B97F4A7C15ull) ^ (std::uint64_t(std::uint32_t(p[1])) << 21) ^ std::uint64_t(std::uint32_t(p[2])) * 0xC2B2AE3D27D4EB4Full; }
    static std::uint64_t bkey(int prev, int next) { return (std::uint64_t(std::uint32_t(prev)) << 32) | std::uint32_t(next); }
    static std::uint64_t pkey(int prev) { return (std::uint64_t(std::uint32_t(prev)) << 32) | 0xFFFFFFFFull; }

    // What the text models say about the next token (computed once per position).
    struct Query {
        i64 t = 0;                 // tokens seen
        bool uni_ok = false;       // unigram has data
        int prev = -1, prev_n = 0; // bigram context and its count (bigram has data when prev_n > 0)
        int match = -1;            // match model's token (-1: no match: falls back)
        int set = 0;               // weight set
    };
    Query query() const {
        Query q; q.t = i64(hist.size());
        q.uni_ok = q.t > 0;
        if (q.t > 0) {
            q.prev = hist.back();
            if (auto pt = bi.find(pkey(q.prev)); pt != bi.end()) q.prev_n = pt->second.first;
        }
        int mlen = 0;
        if (q.t >= 3) {
            if (auto it = ctx3.find(key3(hist.data() + q.t - 3)); it != ctx3.end()) {
                const i64 p = it->second;   // position whose 3 preceding tokens hash like ours
                while (mlen < 32 && p - 1 - mlen >= 0 && hist[std::size_t(p - 1 - mlen)] == hist[std::size_t(q.t - 1 - mlen)]) ++mlen;
                q.match = p < q.t ? hist[std::size_t(p)] : -2;   // -2: matched but nothing follows (probability 0 for all)
            }
        }
        const int mb = mlen == 0 ? 0 : mlen < 4 ? 1 : mlen < 16 ? 2 : 3;
        q.set = mb * 2 + (q.prev_n > 0 ? 1 : 0);
        return q;
    }
    // A weight set used for the first time starts from the nearest set below it (shorter match, same bigram flag)
    // that has learned: long matches are first met after shorter ones, whose weights already trust the match model.
    void select(const Query& q) {
        if (used[std::size_t(q.set)]) return;
        for (int s = q.set - 2; s >= 0; s -= 2) if (used[std::size_t(s)]) { w[std::size_t(q.set)] = w[std::size_t(s)]; break; }
    }
    // p_3..p_5 of token y; p0 is the network's probability (the fallback)
    void text_probs(const Query& q, int y, double p0, double out[3]) const {
        if (q.uni_ok) { auto it = uni.find(y); out[0] = it == uni.end() ? 0.0 : double(it->second) / double(q.t); } else out[0] = p0;
        if (q.prev_n > 0) { auto it = bi.find(bkey(q.prev, y)); out[1] = it == bi.end() ? 0.0 : double(it->second.first) / double(q.prev_n); } else out[1] = p0;
        out[2] = q.match == -1 ? p0 : (q.match == y ? 1.0 : 0.0);
    }
    // Bayesian update with fixed share; p[] the components' probabilities of the observed token, pm their mixture
    void learn(const Query& q, const double p[K], double pm) {
        auto& ws = w[std::size_t(q.set)];
        used[std::size_t(q.set)] = true;
        pm = std::max(pm, 1e-300);
        double sum = 0;
        for (int i = 0; i < K; ++i) { ws[std::size_t(i)] *= std::pow(std::max(p[i], 1e-300) / pm, eta); sum += ws[std::size_t(i)]; }
        for (int i = 0; i < K; ++i) ws[std::size_t(i)] = (1 - alpha) * ws[std::size_t(i)] / sum + alpha / K;
    }
    // the text models see token y
    void push(int y) {
        const i64 t = i64(hist.size());
        if (t >= 3) ctx3[key3(hist.data() + t - 3)] = t;
        if (t > 0) {
            const int pv = hist.back();
            ++bi[pkey(pv)].first;
            if (++bi[bkey(pv, y)].first == 1) succ[pv].push_back(y);
        }
        ++uni[y];
        hist.push_back(y);
    }
    // the network's three components at token y from logits z (lse: log-sum-exp of scale[i] * z)
    static void net_probs(const float* z, const double lse[3], int y, double out[3]) {
        for (int i = 0; i < 3; ++i) out[i] = std::exp(scale[i] * double(z[y]) - lse[i]);
    }
    static void lses(std::span<const float> z, double lse[3]) {
        double m = -INFINITY; for (float x : z) m = std::max(m, double(x));
        double s[3] = {0, 0, 0};
        for (float x : z) { const double d = double(x) - m; s[0] += std::exp(d); s[1] += std::exp(scale[1] * d); s[2] += std::exp(scale[2] * d); }
        for (int i = 0; i < 3; ++i) lse[i] = scale[i] * m + std::log(s[i]);
    }
};

// Streaming mixer for generation. Per position: distribution(logits) gives the full mixed next-token distribution
// (normalised), observe(y) then learns from the token that followed (weights, when a distribution was asked for
// this position) and adds it to the text models. observe() without a distribution (prompt tokens fed without their
// logits) only updates the text models.
class StreamMixer {
public:
    MixModels m;

    void reset() { m.reset(); have_ = false; }

    const std::vector<double>& distribution(std::span<const float> z) {
        const std::size_t V = z.size();
        z_.assign(z.begin(), z.end());
        MixModels::lses(z, lse_);
        q_ = m.query();
        m.select(q_);
        const auto& ws = m.w[std::size_t(q_.set)];
        // dense part: components 0..2, plus the weight of every text model that falls back to the network
        double c0 = ws[0];
        if (!q_.uni_ok) c0 += ws[3];
        if (q_.prev_n <= 0) c0 += ws[4];
        if (q_.match == -1) c0 += ws[5];
        P_.resize(V);
        for (std::size_t v = 0; v < V; ++v) {
            const double x = double(z[v]);
            P_[v] = c0 * std::exp(x - lse_[0]) + ws[1] * std::exp(MixModels::scale[1] * x - lse_[1]) + ws[2] * std::exp(MixModels::scale[2] * x - lse_[2]);
        }
        // sparse parts
        if (q_.uni_ok) for (auto [tok, n] : m.uni) if (tok >= 0 && std::size_t(tok) < V) P_[std::size_t(tok)] += ws[3] * double(n) / double(q_.t);
        if (q_.prev_n > 0)
            if (auto it = m.succ.find(q_.prev); it != m.succ.end())
                for (int tok : it->second) if (tok >= 0 && std::size_t(tok) < V) P_[std::size_t(tok)] += ws[4] * double(m.bi.at(MixModels::bkey(q_.prev, tok)).first) / double(q_.prev_n);
        if (q_.match >= 0 && std::size_t(q_.match) < V) P_[std::size_t(q_.match)] += ws[5];
        double tot = 0; for (double p : P_) tot += p;
        if (tot > 0) for (double& p : P_) p /= tot;   // (already 1 up to rounding, or less a match/-2 weight)
        have_ = true;
        return P_;
    }

    // log p_mix(y) of the last distribution (requires one)
    double logprob(int y) const { return std::log(std::max(P_[std::size_t(y)], 1e-300)); }

    void observe(int y) {
        if (have_ && y >= 0 && std::size_t(y) < z_.size()) {
            double p[MixModels::K];
            MixModels::net_probs(z_.data(), lse_, y, p);
            m.text_probs(q_, y, p[0], p + 3);
            const auto& ws = m.w[std::size_t(q_.set)];
            double pm = 0; for (int i = 0; i < MixModels::K; ++i) pm += ws[std::size_t(i)] * p[i];
            m.learn(q_, p, pm);
        }
        have_ = false;
        m.push(y);
    }

    // Sampler logit filter: replaces the logits by the log of the mixed distribution.
    void filter(std::span<float> l) {
        const auto& P = distribution(l);
        for (std::size_t v = 0; v < l.size(); ++v) l[v] = float(std::log(std::max(P[v], 1e-30)));
    }

private:
    std::vector<float> z_;
    std::vector<double> P_;
    double lse_[3] = {0, 0, 0};
    MixModels::Query q_;
    bool have_ = false;
};

// Generation with context mixing (forge run --mix 1): the sampler draws from the mixed distribution (its logit filter
// replaces the logits by log p_mix; temperature / top-k / top-p then apply to it as to logits), and the mixer
// observes every prompt and emitted token. With `warm`, the prompt is first run with all rows so the mixing weights
// also learn from the prompt's own tokens (one extra prefill). A filter the sampler already has (a grammar) runs
// after the mixer; the sampler's per-token `allow` shortcut is dropped, since it would bypass the filter.
inline std::vector<int> generate_mixed(inference::session& s, std::span<const int> prompt, int max_new, inference::sampler& smp, StreamMixer& mx,
                                       const std::function<bool(int)>& on_token = {}, inference::generate_stats* stats = nullptr, bool warm = true) {
    if (prompt.empty()) die("generate_mixed: empty prompt");
    mx.observe(prompt[0]);
    if (warm && prompt.size() > 1) {
        s.reset();
        auto L = s.forward_all(prompt.subspan(0, prompt.size() - 1));
        const std::size_t V = std::size_t(s.vocab());
        for (std::size_t t = 1; t < prompt.size(); ++t) { mx.distribution(std::span<const float>(L.data() + (t - 1) * V, V)); mx.observe(prompt[t]); }
        s.reset();
    } else for (std::size_t t = 1; t < prompt.size(); ++t) mx.observe(prompt[t]);
    auto inner = smp.filter;
    smp.filter = [&mx, inner](std::span<float> l) { mx.filter(l); if (inner) inner(l); };
    smp.allow = {};
    auto out = s.generate(prompt, max_new, smp, [&](int t) { mx.observe(t); return on_token ? on_token(t) : true; }, stats);
    smp.filter = inner;
    return out;
}

}  // namespace forge::compress
