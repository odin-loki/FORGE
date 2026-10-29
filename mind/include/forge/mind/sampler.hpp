#pragma once
// Composable sampler chain: repetition penalty -> temperature -> top-k -> top-p -> min-p -> draw.
#include <algorithm>
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

    int sample(std::span<const float> logits_in, std::span<const int> history = {}) {
        std::vector<float> l(logits_in.begin(), logits_in.end());
        if (c_.repetition_penalty != 1.f) {
            std::unordered_set<int> seen;
            std::size_t from = history.size() > std::size_t(c_.repetition_window) ? history.size() - std::size_t(c_.repetition_window) : 0;
            for (std::size_t i = from; i < history.size(); ++i) seen.insert(history[i]);
            for (int t : seen) if (t >= 0 && std::size_t(t) < l.size()) l[std::size_t(t)] = l[std::size_t(t)] > 0 ? l[std::size_t(t)] / c_.repetition_penalty : l[std::size_t(t)] * c_.repetition_penalty;
        }
        if (c_.temperature <= 0.f) return int(std::max_element(l.begin(), l.end()) - l.begin());
        std::vector<int> idx(l.size()); std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return l[std::size_t(a)] > l[std::size_t(b)] || (l[std::size_t(a)] == l[std::size_t(b)] && a < b); });
        std::size_t keep = idx.size();
        if (c_.top_k > 0) keep = std::min(keep, std::size_t(c_.top_k));
        std::vector<double> p(keep);
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
        double tot = 0; for (std::size_t i = 0; i < keep; ++i) tot += p[i];
        double u = std::uniform_real_distribution<double>(0, tot)(rng_);
        for (std::size_t i = 0; i < keep; ++i) { u -= p[i]; if (u <= 0) return idx[i]; }
        return idx[keep - 1];
    }
private:
    sampler_config c_;
    std::mt19937_64 rng_;
};

} // namespace forge::mind
