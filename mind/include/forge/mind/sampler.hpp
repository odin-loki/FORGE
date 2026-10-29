#pragma once
// Composable sampler chain: repetition penalty -> temperature -> top-k -> top-p -> min-p -> draw.
#include <algorithm>
#include <functional>
#include <cmath>
#include <numeric>
#include <random>
#include <span>
#include <unordered_set>
#include <vector>

namespace forge::mind {

struct sampler_config {
    float temperature = 0.f;   // 0 = greedy
    int top_k = 0;             // 0 = off
    float top_p = 1.f;
    float min_p = 0.f;
    float repetition_penalty = 1.f;
    int repetition_window = 64;
    std::uint64_t seed = 0;
};

class sampler {
public:
    explicit sampler(sampler_config c = {}) : c_(c), rng_(c.seed) {}
    const sampler_config& config() const { return c_; }
    // Optional logit filter run after the repetition penalty (grammar constraints set disallowed logits to -inf).
    std::function<void(std::span<float>)> filter;

    // Full next-token distribution after the whole chain (one-hot when greedy).
    std::vector<double> probs(std::span<const float> logits_in, std::span<const int> history = {}) {
        auto l = prepare(logits_in, history);
        std::vector<double> out(l.size(), 0.0);
        if (c_.temperature <= 0.f) { out[std::size_t(std::max_element(l.begin(), l.end()) - l.begin())] = 1; return out; }
        std::vector<int> idx; std::vector<double> p; truncate(l, idx, p);
        double tot = 0; for (auto x : p) tot += x;
        for (std::size_t i = 0; i < p.size(); ++i) out[std::size_t(idx[i])] = p[i] / tot;
        return out;
    }
    // Draw from an explicit distribution (speculative decoding residuals).
    int draw(std::span<const double> p) {
        double tot = 0; for (auto x : p) tot += x;
        if (tot <= 0) return int(std::max_element(p.begin(), p.end()) - p.begin());
        double u = std::uniform_real_distribution<double>(0, tot)(rng_);
        for (std::size_t i = 0; i < p.size(); ++i) { u -= p[i]; if (u <= 0 && p[i] > 0) return int(i); }
        for (std::size_t i = p.size(); i-- > 0;) if (p[i] > 0) return int(i);
        return 0;
    }
    double uniform() { return std::uniform_real_distribution<double>(0, 1)(rng_); }

    int sample(std::span<const float> logits_in, std::span<const int> history = {}) {
        auto l = prepare(logits_in, history);
        if (c_.temperature <= 0.f) return int(std::max_element(l.begin(), l.end()) - l.begin());
        std::vector<int> idx; std::vector<double> p; truncate(l, idx, p);
        double tot = 0; for (auto x : p) tot += x;
        double u = std::uniform_real_distribution<double>(0, tot)(rng_);
        for (std::size_t i = 0; i < p.size(); ++i) { u -= p[i]; if (u <= 0) return idx[i]; }
        return idx.back();
    }
private:
    std::vector<float> prepare(std::span<const float> logits_in, std::span<const int> history) {
        std::vector<float> l(logits_in.begin(), logits_in.end());
        if (c_.repetition_penalty != 1.f) {
            std::unordered_set<int> seen;
            std::size_t from = history.size() > std::size_t(c_.repetition_window) ? history.size() - std::size_t(c_.repetition_window) : 0;
            for (std::size_t i = from; i < history.size(); ++i) seen.insert(history[i]);
            for (int t : seen) if (t >= 0 && std::size_t(t) < l.size()) l[std::size_t(t)] = l[std::size_t(t)] > 0 ? l[std::size_t(t)] / c_.repetition_penalty : l[std::size_t(t)] * c_.repetition_penalty;
        }
        if (filter) filter(l);
        return l;
    }
    // sorted candidate indices + unnormalised probabilities after top-k / top-p / min-p
    void truncate(const std::vector<float>& l, std::vector<int>& idx, std::vector<double>& p) {
        idx.resize(l.size()); std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return l[std::size_t(a)] > l[std::size_t(b)] || (l[std::size_t(a)] == l[std::size_t(b)] && a < b); });
        std::size_t keep = idx.size();
        if (c_.top_k > 0) keep = std::min(keep, std::size_t(c_.top_k));
        p.assign(keep, 0.0);
        double mx = l[std::size_t(idx[0])] / c_.temperature, sum = 0;
        for (std::size_t i = 0; i < keep; ++i) { p[i] = std::exp(double(l[std::size_t(idx[i])]) / c_.temperature - mx); sum += p[i]; }
        for (auto& x : p) x /= sum;
        if (c_.top_p < 1.f) {
            double cum = 0; std::size_t k = 0;
            while (k < keep) { cum += p[k++]; if (cum >= c_.top_p) break; }
            keep = std::max<std::size_t>(k, 1);
        }
        if (c_.min_p > 0.f) {
            std::size_t k = 1; while (k < keep && p[k] >= c_.min_p * p[0]) ++k;
            keep = k;
        }
        // drop -inf (filtered) candidates, keep at least one
        std::size_t k2 = keep; while (k2 > 1 && !(p[k2 - 1] > 0)) --k2;
        keep = k2;
        p.resize(keep); idx.resize(keep);
    }
    sampler_config c_;
    std::mt19937_64 rng_;
};

} // namespace forge::mind
