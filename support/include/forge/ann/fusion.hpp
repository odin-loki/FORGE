#pragma once
// Rank fusion and re-ranking helpers for hybrid retrieval.
//
//   reciprocal_rank_fusion({bm25_ids, ann_ids}, k)   score(id) = sum over lists of weight / (c + rank), rank from 1 (c = 60)
//   rerank(candidates, q, get, exact, k)             recompute the distance with an exact metric, keep the k closest
// Results are ordered by (score descending, id ascending) / (dist, id): total orders, so they never depend on timing.
#include <forge/ann/metrics.hpp>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace forge::ann {

struct fused { std::uint64_t id; double score; bool operator==(const fused&) const = default; };

// Each list is ranked best first. A duplicate id inside one list counts once (its best rank). `weights` (optional, one per
// list) scale a list's contribution.
inline std::vector<fused> reciprocal_rank_fusion(const std::vector<std::vector<std::uint64_t>>& lists, std::size_t k, double c = 60.0,
                                                 const std::vector<double>& weights = {}) {
    std::unordered_map<std::uint64_t, double> score;
    for (std::size_t li = 0; li < lists.size(); ++li) {
        const double w = li < weights.size() ? weights[li] : 1.0;
        std::unordered_map<std::uint64_t, bool> seen;
        std::size_t rank = 0;
        for (std::uint64_t id : lists[li]) {
            ++rank;
            if (!seen.emplace(id, true).second) continue;
            score[id] += w / (c + double(rank));
        }
    }
    std::vector<fused> out; out.reserve(score.size());
    for (auto& [id, s] : score) out.push_back({id, s});
    std::sort(out.begin(), out.end(), [](const fused& a, const fused& b) { return a.score != b.score ? a.score > b.score : a.id < b.id; });
    if (out.size() > k) out.resize(k);
    return out;
}

template <class V, class Exact>
std::vector<hit> rerank(std::span<const hit> cand, const V& q, std::type_identity_t<std::function<const V&(std::uint64_t)>> get, const Exact& exact, std::size_t k) {
    std::vector<hit> out; out.reserve(cand.size());
    for (const hit& h : cand) out.push_back({h.id, float(exact.distance(q, get(h.id)))});
    std::sort(out.begin(), out.end(), hit_before);
    if (out.size() > k) out.resize(k);
    return out;
}

} // namespace forge::ann
