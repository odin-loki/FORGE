#pragma once
// task/algorithms: deterministic parallel algorithms on forge::task::pool. Every function takes a `pool*`;
// nullptr runs the same algorithm serially on the caller. The result never depends on the pool's size.
//
//   parallel_blocks(p, n, block, f(block, begin, end))   fixed blocks of `block` items, run concurrently
//   parallel_sort(p, first, last, cmp)                    stable sort == std::stable_sort(first, last, cmp)
//   parallel_merge(p, a0, a1, b0, b1, out, cmp)           stable merge == std::merge (ties: first range first)
//   merge_runs(p, runs, cmp)                              stable k-way merge of sorted vectors (run order breaks ties)
//   parallel_topk_by(p, n, k, make(i), cmp)               the first k of stable_sort({make(0)..make(n-1)}, cmp)
//   parallel_topk(p, first, last, k, cmp)                 the same over a range
//   parallel_reduce(p, first, last, init, op, block)      blocked fold, see below
//   parallel_transform_reduce(p, n, init, map, op, block)
//   parallel_inclusive_scan / parallel_exclusive_scan     blocked prefix sums
//
// "Exactly the serial result" means:
//   * sort, merge, merge_runs, top-k: the output is unique (stable order), so it equals the standard serial
//     algorithm bit for bit for any comparator that is a strict weak ordering, whatever the thread count.
//   * reduce and scans: the items are grouped into fixed blocks of `block` items (default 2048 / 4096), each block
//     is folded left to right, and the block results are combined left to right. The grouping depends only on n and
//     `block`, never on the thread count, so every thread count (and the serial run, p == nullptr) gives the same
//     value. For an associative op (integers, min, max, bitwise) this equals std::accumulate / std::inclusive_scan;
//     for floating-point addition it equals the serial run of the same blocked algorithm, not a plain left fold.
// If f throws, the first exception is rethrown after the running blocks finish.
#include <forge/task/pool.hpp>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

namespace forge::task {

namespace detail {
// f(i) for i in [0, n); concurrent when p has workers.
template <class F> void for_each_index(pool* p, std::size_t n, F&& f) {
    if (!p || p->size() <= 1 || n <= 1) { for (std::size_t i = 0; i < n; ++i) f(i); return; }
    p->parallel_for(n, 1, [&](std::size_t b, std::size_t e) { for (std::size_t i = b; i < e; ++i) f(i); });
}
inline std::size_t threads_of(pool* p) { return p ? std::max<std::size_t>(1, p->size()) : 1; }
} // namespace detail

// Runs f(block_index, begin, end) for the blocks [i*block, min(n, (i+1)*block)).
template <class F> void parallel_blocks(pool* p, std::size_t n, std::size_t block, F&& f) {
    block = std::max<std::size_t>(block, 1);
    const std::size_t nb = (n + block - 1) / block;
    detail::for_each_index(p, nb, [&](std::size_t i) { f(i, i * block, std::min(n, (i + 1) * block)); });
}

// ---- merge ---------------------------------------------------------------------------------------------------
// Stable parallel merge of the sorted ranges [a0,a1) and [b0,b1) into `out` (disjoint from both). Splits the output
// into parts by co-ranking (binary search on the two inputs) and merges the parts with std::merge.
template <class It1, class It2, class Out, class Cmp = std::less<>>
Out parallel_merge(pool* p, It1 a0, It1 a1, It2 b0, It2 b1, Out out, Cmp cmp = {}, std::size_t grain = 8192) {
    const std::size_t n1 = std::size_t(a1 - a0), n2 = std::size_t(b1 - b0), n = n1 + n2;
    const std::size_t parts = std::min<std::size_t>(detail::threads_of(p) * 4, std::max<std::size_t>(1, n / std::max<std::size_t>(grain, 1)));
    if (!p || parts <= 1) return std::merge(a0, a1, b0, b1, out, cmp);
    // split[k] = (i, j): the first k*n/parts outputs are a[0,i) and b[0,j)
    std::vector<std::pair<std::size_t, std::size_t>> split(parts + 1);
    split[0] = {0, 0}; split[parts] = {n1, n2};
    detail::for_each_index(p, parts - 1, [&](std::size_t k0) {
        const std::size_t k = k0 + 1, t = k * n / parts;
        std::size_t lo = t > n2 ? t - n2 : 0, hi = std::min(t, n1);
        while (lo < hi) {
            const std::size_t mid = (lo + hi) / 2, j = t - mid;   // mid < hi <= t, so j >= 1; mid >= lo >= t - n2, so j <= n2
            if (cmp(b0[std::ptrdiff_t(j - 1)], a0[std::ptrdiff_t(mid)])) hi = mid; else lo = mid + 1;
        }
        split[k] = {lo, t - lo};
    });
    detail::for_each_index(p, parts, [&](std::size_t k) {
        const auto [i0, j0] = split[k]; const auto [i1, j1] = split[k + 1];
        std::merge(a0 + std::ptrdiff_t(i0), a0 + std::ptrdiff_t(i1), b0 + std::ptrdiff_t(j0), b0 + std::ptrdiff_t(j1),
                   out + std::ptrdiff_t(i0 + j0), cmp);
    });
    return out + std::ptrdiff_t(n);
}

// ---- sort ----------------------------------------------------------------------------------------------------
// Stable. value_type must be default constructible and movable (a scratch buffer of n elements is used).
template <class It, class Cmp = std::less<>>
void parallel_sort(pool* p, It first, It last, Cmp cmp = {}, std::size_t grain = 4096) {
    const std::size_t n = std::size_t(last - first);
    const std::size_t th = detail::threads_of(p);
    if (th <= 1 || n <= grain) { std::stable_sort(first, last, cmp); return; }
    const std::size_t chunks = std::min<std::size_t>(th * 2, std::max<std::size_t>(1, n / grain));
    const std::size_t csz = (n + chunks - 1) / chunks;
    detail::for_each_index(p, chunks, [&](std::size_t c) {
        const std::size_t b = c * csz, e = std::min(n, b + csz);
        if (b < e) std::stable_sort(first + std::ptrdiff_t(b), first + std::ptrdiff_t(e), cmp);
    });
    using T = typename std::iterator_traits<It>::value_type;
    std::vector<T> tmp(n);
    bool in_tmp = false;
    for (std::size_t w = csz; w < n; w *= 2) {
        const std::size_t pairs = (n + 2 * w - 1) / (2 * w);
        auto round = [&](auto src, auto dst) {
            auto one = [&](pool* q, std::size_t j) {
                const std::size_t b = j * 2 * w, m = std::min(n, b + w), e = std::min(n, b + 2 * w);
                parallel_merge(q, src + std::ptrdiff_t(b), src + std::ptrdiff_t(m), src + std::ptrdiff_t(m), src + std::ptrdiff_t(e),
                               dst + std::ptrdiff_t(b), cmp, grain);
            };
            if (pairs >= th) detail::for_each_index(p, pairs, [&](std::size_t j) { one(nullptr, j); });
            else for (std::size_t j = 0; j < pairs; ++j) one(p, j);
        };
        if (!in_tmp) round(first, tmp.begin()); else round(tmp.begin(), first);
        in_tmp = !in_tmp;
    }
    if (in_tmp) std::move(tmp.begin(), tmp.end(), first);
}

// Stable k-way merge of sorted runs: elements of an earlier run precede equal elements of a later run.
template <class T, class Cmp = std::less<>>
std::vector<T> merge_runs(pool* p, std::vector<std::vector<T>> runs, Cmp cmp = {}, std::size_t grain = 8192) {
    if (runs.empty()) return {};
    while (runs.size() > 1) {
        const std::size_t pairs = runs.size() / 2;
        std::vector<std::vector<T>> next((runs.size() + 1) / 2);
        auto one = [&](pool* q, std::size_t j) {
            auto& a = runs[2 * j]; auto& b = runs[2 * j + 1];
            next[j].resize(a.size() + b.size());
            parallel_merge(q, a.begin(), a.end(), b.begin(), b.end(), next[j].begin(), cmp, grain);
            std::vector<T>().swap(a); std::vector<T>().swap(b);
        };
        if (pairs >= detail::threads_of(p)) detail::for_each_index(p, pairs, [&](std::size_t j) { one(nullptr, j); });
        else for (std::size_t j = 0; j < pairs; ++j) one(p, j);
        if (runs.size() % 2) next.back() = std::move(runs.back());
        runs = std::move(next);
    }
    return std::move(runs[0]);
}

// ---- top-k ---------------------------------------------------------------------------------------------------
// The first k elements of the stable sort of make(0) .. make(n-1) under `cmp` (cmp(a, b): a before b), in that order.
// Each block keeps a bounded heap; the block results are merged in block order. make(i) is called exactly once per
// i and may run on any thread. Memory: O(k) per block.
template <class Make, class Cmp>
auto parallel_topk_by(pool* p, std::size_t n, std::size_t k, Make&& make, Cmp cmp, std::size_t block = 4096) {
    using T = std::decay_t<decltype(make(std::size_t(0)))>;
    std::vector<T> out;
    if (k == 0 || n == 0) return out;
    const std::size_t nb = (n + block - 1) / block;
    std::vector<std::vector<T>> runs(nb);
    detail::for_each_index(p, nb, [&](std::size_t bi) {
        const std::size_t b = bi * block, e = std::min(n, b + block);
        struct entry { T v; std::size_t idx; };
        auto before = [&](const entry& x, const entry& y) { return cmp(x.v, y.v) || (!cmp(y.v, x.v) && x.idx < y.idx); };
        std::vector<entry> heap;   // max-heap: the worst kept entry on top
        heap.reserve(std::min(k, e - b));
        for (std::size_t i = b; i < e; ++i) {
            entry en{make(i), i};
            if (heap.size() < k) { heap.push_back(std::move(en)); std::push_heap(heap.begin(), heap.end(), before); }
            else if (before(en, heap.front())) {
                std::pop_heap(heap.begin(), heap.end(), before); heap.back() = std::move(en); std::push_heap(heap.begin(), heap.end(), before);
            }
        }
        std::sort_heap(heap.begin(), heap.end(), before);   // ascending: best first
        auto& r = runs[bi]; r.reserve(heap.size());
        for (auto& x : heap) r.push_back(std::move(x.v));
    });
    out = merge_runs(p, std::move(runs), cmp);
    if (out.size() > k) out.resize(k);
    return out;
}
template <class It, class Cmp = std::less<>>
auto parallel_topk(pool* p, It first, It last, std::size_t k, Cmp cmp = {}, std::size_t block = 4096) {
    return parallel_topk_by(p, std::size_t(last - first), k, [&](std::size_t i) { return *(first + std::ptrdiff_t(i)); }, cmp, block);
}

// ---- reduce and scans ----------------------------------------------------------------------------------------
// result = op(...op(op(init, B0), B1)..., Bm-1) where Bj is the left fold of map(i) over block j (first item is the
// seed of the block; no init inside blocks). n == 0 returns init.
template <class T, class Map, class Op>
T parallel_transform_reduce(pool* p, std::size_t n, T init, Map&& map, Op op, std::size_t block = 2048) {
    if (n == 0) return init;
    block = std::max<std::size_t>(block, 1);
    const std::size_t nb = (n + block - 1) / block;
    std::vector<std::optional<T>> part(nb);
    detail::for_each_index(p, nb, [&](std::size_t bi) {
        const std::size_t b = bi * block, e = std::min(n, b + block);
        T acc = map(b);
        for (std::size_t i = b + 1; i < e; ++i) acc = op(std::move(acc), map(i));
        part[bi] = std::move(acc);
    });
    for (auto& x : part) init = op(std::move(init), std::move(*x));
    return init;
}
template <class It, class T, class Op = std::plus<>>
T parallel_reduce(pool* p, It first, It last, T init, Op op = {}, std::size_t block = 2048) {
    return parallel_transform_reduce(p, std::size_t(last - first), std::move(init), [&](std::size_t i) -> T { return *(first + std::ptrdiff_t(i)); }, op, block);
}

// out[i] = op(in[0], ..., in[i]) (inclusive) or op(init, in[0], ..., in[i-1]) (exclusive). `out` may be `in`.
// Blocks of `block` items are scanned independently, then each block is offset by the combined totals before it.
template <class InIt, class OutIt, class Op = std::plus<>>
void parallel_inclusive_scan(pool* p, InIt first, InIt last, OutIt out, Op op = {}, std::size_t block = 4096) {
    using T = typename std::iterator_traits<InIt>::value_type;
    const std::size_t n = std::size_t(last - first);
    if (n == 0) return;
    block = std::max<std::size_t>(block, 1);
    const std::size_t nb = (n + block - 1) / block;
    std::vector<std::optional<T>> total(nb);
    parallel_blocks(p, n, block, [&](std::size_t bi, std::size_t b, std::size_t e) {
        T acc = *(first + std::ptrdiff_t(b)); *(out + std::ptrdiff_t(b)) = acc;
        for (std::size_t i = b + 1; i < e; ++i) { acc = op(std::move(acc), *(first + std::ptrdiff_t(i))); *(out + std::ptrdiff_t(i)) = acc; }
        total[bi] = std::move(acc);
    });
    std::vector<std::optional<T>> carry(nb);   // carry[j] = combined totals of blocks 0..j-1
    for (std::size_t j = 1; j < nb; ++j) carry[j] = j == 1 ? *total[0] : op(*carry[j - 1], *total[j - 1]);
    parallel_blocks(p, n, block, [&](std::size_t bi, std::size_t b, std::size_t e) {
        if (bi == 0) return;
        for (std::size_t i = b; i < e; ++i) *(out + std::ptrdiff_t(i)) = op(*carry[bi], *(out + std::ptrdiff_t(i)));
    });
}
template <class InIt, class OutIt, class T, class Op = std::plus<>>
void parallel_exclusive_scan(pool* p, InIt first, InIt last, OutIt out, T init, Op op = {}, std::size_t block = 4096) {
    const std::size_t n = std::size_t(last - first);
    if (n == 0) return;
    block = std::max<std::size_t>(block, 1);
    std::vector<T> inc(n);
    parallel_inclusive_scan(p, first, last, inc.begin(), op, block);
    detail::for_each_index(p, (n + block - 1) / std::max<std::size_t>(block, 1), [&](std::size_t bi) {
        const std::size_t b = bi * block, e = std::min(n, b + block);
        for (std::size_t i = b; i < e; ++i) *(out + std::ptrdiff_t(i)) = i == 0 ? init : op(init, inc[i - 1]);
    });
}

} // namespace forge::task
