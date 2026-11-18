#pragma once
// forge::ann::flat_index: exact nearest-neighbour search by scanning every vector, in parallel blocks.
//   flat_index<std::vector<float>, ann::l2> ix; ix.add(id, v); auto hits = ix.search(q, 10, &pool);
// search returns the k closest ordered by (dist, id); it equals the brute-force sort of all (dist, id) pairs, for any pool
// size (nullptr: serial on the caller). When the metric has a batch `distances(q, span<const V>, span<float>)` it is called
// once per block of 2048 vectors.
#include <forge/ann/metrics.hpp>
#include <forge/task/algorithms.hpp>
#include <algorithm>
#include <vector>

namespace forge::ann {

template <class V, class M> requires metric<M, V>
class flat_index {
public:
    explicit flat_index(M m = {}) : m_(std::move(m)) {}
    void add(std::uint64_t id, V v) { ids_.push_back(id); vs_.push_back(std::move(v)); }
    std::size_t size() const { return ids_.size(); }
    const V& vector_at(std::size_t i) const { return vs_[i]; }
    std::uint64_t id_at(std::size_t i) const { return ids_[i]; }

    std::vector<hit> search(const V& q, std::size_t k, forge::task::pool* pool = nullptr) const {
        const std::size_t n = ids_.size();
        if (k == 0 || n == 0) return {};
        constexpr std::size_t kBlock = 2048;
        const std::size_t nb = (n + kBlock - 1) / kBlock;
        std::vector<std::vector<hit>> runs(nb);
        forge::task::parallel_blocks(pool, n, kBlock, [&](std::size_t bi, std::size_t b, std::size_t e) {
            std::vector<float> d(e - b);
            if constexpr (batch_metric<M, V>) m_.distances(q, std::span<const V>(vs_.data() + b, e - b), std::span<float>(d));
            else for (std::size_t i = b; i < e; ++i) d[i - b] = m_.distance(q, vs_[i]);
            std::vector<hit> heap; heap.reserve(std::min(k, e - b));
            for (std::size_t i = b; i < e; ++i) {
                const hit h{ids_[i], d[i - b]};
                if (heap.size() < k) { heap.push_back(h); std::push_heap(heap.begin(), heap.end(), hit_before); }
                else if (hit_before(h, heap.front())) { std::pop_heap(heap.begin(), heap.end(), hit_before); heap.back() = h; std::push_heap(heap.begin(), heap.end(), hit_before); }
            }
            std::sort_heap(heap.begin(), heap.end(), hit_before);
            runs[bi] = std::move(heap);
        });
        auto out = forge::task::merge_runs(pool, std::move(runs), hit_before);
        if (out.size() > k) out.resize(k);
        return out;
    }

private:
    M m_;
    std::vector<std::uint64_t> ids_;
    std::vector<V> vs_;
};

} // namespace forge::ann
