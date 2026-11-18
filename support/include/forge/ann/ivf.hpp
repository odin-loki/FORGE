#pragma once
// forge::ann::ivf_flat: an inverted-file index over dense float vectors (l2): k-means coarse centroids trained on a sample, one
// inverted list per centroid, a query scans the rows of its nprobe nearest lists exactly.
//
//   auto cents = ann::kmeans(sample.data(), sample_rows, dim, nlist, 20, seed, &pool);   // one-off training
//   ann::ivf_flat ix(cents, dim);  ix.add(id, vec); ...  ix.finalize(&pool);
//   auto hits = ix.search(q, 10, nprobe);                // the 10 nearest among the rows of the nprobe nearest lists, (dist, id) order
//   auto hb = ix.search_batch_device(queries, 10, nprobe, graph::cuda_ivf(rt), &pool);   // the same scan on a device
//
// Approximate: a true neighbour whose list is not probed is missed (nprobe = nlist is exact). Within the probed lists the CPU search
// is exact and its distances are those of the l2 metric (as flat_index). Probe order: centroids by (squared distance, index).
// k-means: Lloyd iterations from k distinct sample rows chosen by a seeded shuffle; assignment ties go to the lower centroid; an
// empty cluster is re-seeded with the row farthest from its centroid in the largest cluster. Sums are accumulated per fixed block
// of rows in double and added in block order, so the centroids are the same for any thread count.
// Devices: search_batch_device sends the rows once (resident, sorted by list) and per query the [begin, end) row ranges of its probed
// lists; the device returns the 16 best (approximate squared distance, row) and the host re-scores them with the l2 metric. Results
// can differ from search() only where f32 rounding reorders near-ties around the 16th candidate of a query; k <= 16.
// search_batch_matmul runs the scan through a plain matmul callable (gpu_flat_l2's: graph::ann_matmul(reg)): for every list touched by
// the batch, one product of the queries probing it against the list's rows, |q|^2 + |x|^2 - 2 q.x, the best 32 per (query, list)
// re-scored exactly on the host.
#include <forge/ann/metrics.hpp>
#include <forge/core/error.hpp>
#include <forge/task/algorithms.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace forge::ann {

namespace detail {
inline float ivf_sq(const float* a, const float* b, std::size_t d) { float s = 0; for (std::size_t j = 0; j < d; ++j) { const float t = a[j] - b[j]; s += t * t; } return s; }
inline std::uint32_t ivf_nearest(const float* x, const float* cents, std::size_t k, std::size_t d, float* best = nullptr) {
    std::uint32_t bi = 0; float bd = std::numeric_limits<float>::infinity();
    for (std::size_t c = 0; c < k; ++c) { const float s = ivf_sq(x, cents + c * d, d); if (s < bd) { bd = s; bi = std::uint32_t(c); } }
    if (best) *best = bd;
    return bi;
}
}

// k centroids (row-major k * d) of the n rows x (n >= k).
inline std::vector<float> kmeans(const float* x, std::size_t n, std::size_t d, std::size_t k, std::size_t iters, std::uint64_t seed, forge::task::pool* pool = nullptr) {
    if (k == 0 || d == 0 || n < k) forge::raise(forge::errc::invalid_argument, "kmeans: need n >= k > 0 rows (n {}, k {})", n, k);
    std::vector<std::size_t> perm(n); std::iota(perm.begin(), perm.end(), std::size_t(0));
    std::mt19937_64 g(seed); for (std::size_t i = 0; i < k; ++i) std::swap(perm[i], perm[i + std::size_t(g() % (n - i))]);
    std::vector<float> c(k * d); for (std::size_t i = 0; i < k; ++i) std::copy(x + perm[i] * d, x + perm[i] * d + d, c.begin() + std::ptrdiff_t(i * d));
    constexpr std::size_t kBlock = 4096; const std::size_t nb = (n + kBlock - 1) / kBlock;
    std::vector<std::uint32_t> as(n); std::vector<float> ad(n);
    for (std::size_t it = 0; it < iters; ++it) {
        std::vector<std::vector<double>> bs(nb); std::vector<std::vector<std::uint64_t>> bc(nb);
        forge::task::parallel_blocks(pool, n, kBlock, [&](std::size_t bi, std::size_t b, std::size_t e) {
            auto& s = bs[bi]; auto& cn = bc[bi]; s.assign(k * d, 0.0); cn.assign(k, 0);
            for (std::size_t i = b; i < e; ++i) { const auto a = detail::ivf_nearest(x + i * d, c.data(), k, d, &ad[i]); as[i] = a; ++cn[a]; for (std::size_t j = 0; j < d; ++j) s[a * d + j] += x[i * d + j]; }
        });
        std::vector<double> sum(k * d, 0.0); std::vector<std::uint64_t> cnt(k, 0);
        for (std::size_t bi = 0; bi < nb; ++bi) { for (std::size_t t = 0; t < k * d; ++t) sum[t] += bs[bi][t]; for (std::size_t t = 0; t < k; ++t) cnt[t] += bc[bi][t]; }
        for (std::size_t t = 0; t < k; ++t) if (cnt[t]) for (std::size_t j = 0; j < d; ++j) c[t * d + j] = float(sum[t * d + j] / double(cnt[t]));
        for (std::size_t t = 0; t < k; ++t) if (!cnt[t]) {   // re-seed: the farthest row of the largest cluster
            const std::size_t big = std::size_t(std::max_element(cnt.begin(), cnt.end()) - cnt.begin());
            std::size_t far = n; float fd = -1;
            for (std::size_t i = 0; i < n; ++i) if (as[i] == big && ad[i] > fd) { fd = ad[i]; far = i; }
            if (far == n) break;
            std::copy(x + far * d, x + far * d + d, c.begin() + std::ptrdiff_t(t * d)); --cnt[big]; cnt[t] = 1; as[far] = std::uint32_t(t); ad[far] = 0;
        }
    }
    return c;
}

class ivf_flat {
public:
    // od/oi[q * 16 ..]: the 16 best (squared distance, row of x) of query q over the row ranges rng[(q * nprobe + p) * 2 ..] (begin, end).
    using scan_fn = std::function<bool(const float* x, std::size_t n, std::size_t d, const float* q, std::size_t nq, const std::uint32_t* rng, std::size_t nprobe, float* od, std::uint32_t* oi, double* upload_s)>;
    using matmul_fn = std::function<bool(const float* x, std::size_t rows, std::size_t k, const float* w, std::size_t cols, float* y)>;
    static constexpr std::size_t kCand = 16;

    ivf_flat(std::vector<float> centroids, std::size_t dim) : c_(std::move(centroids)), d_(dim), k_(dim ? c_.size() / dim : 0) {
        if (!dim || k_ == 0 || c_.size() != k_ * dim) forge::raise(forge::errc::invalid_argument, "ivf_flat: centroids are not a k * {} matrix", dim);
    }
    std::size_t nlist() const { return k_; }
    std::size_t dim() const { return d_; }
    std::size_t size() const { return ids_.size() + pend_ids_.size(); }
    const std::vector<float>& centroids() const { return c_; }
    std::size_t list_of(std::span<const float> v) const { return detail::ivf_nearest(v.data(), c_.data(), k_, d_); }

    void add(std::uint64_t id, std::span<const float> v) {
        if (v.size() != d_) forge::raise(forge::errc::invalid_argument, "ivf_flat::add: expected {} floats, got {}", d_, v.size());
        pend_ids_.push_back(id); pend_.insert(pend_.end(), v.begin(), v.end()); fin_ = false;
    }
    // Groups the rows by list (stable: insertion order inside a list). Called by search if needed; not thread-safe against searches.
    void finalize(forge::task::pool* pool = nullptr) {
        if (fin_) return;
        // existing rows plus pending ones, re-bucketed
        std::vector<std::uint64_t> ids = std::move(ids_); std::vector<float> data = std::move(data_); std::vector<std::uint32_t> lst;
        lst.reserve(ids.size() + pend_ids_.size());
        for (std::size_t l = 0; l < k_; ++l) for (std::size_t r = off_.empty() ? 0 : off_[l]; !off_.empty() && r < off_[l + 1]; ++r) lst.push_back(std::uint32_t(l));
        const std::size_t base = ids.size(), np = pend_ids_.size(); lst.resize(base + np);
        forge::task::parallel_blocks(pool, np, 1024, [&](std::size_t, std::size_t b, std::size_t e) { for (std::size_t i = b; i < e; ++i) lst[base + i] = detail::ivf_nearest(pend_.data() + i * d_, c_.data(), k_, d_); });
        ids.insert(ids.end(), pend_ids_.begin(), pend_ids_.end()); data.insert(data.end(), pend_.begin(), pend_.end()); pend_ids_.clear(); pend_.clear();
        off_.assign(k_ + 1, 0); for (auto l : lst) ++off_[l + 1]; for (std::size_t l = 0; l < k_; ++l) off_[l + 1] += off_[l];
        std::vector<std::size_t> pos(off_.begin(), off_.end() - 1);
        ids_.resize(ids.size()); data_.resize(data.size());
        for (std::size_t i = 0; i < ids.size(); ++i) { const std::size_t p = pos[lst[i]]++; ids_[p] = ids[i]; std::copy(data.begin() + std::ptrdiff_t(i * d_), data.begin() + std::ptrdiff_t((i + 1) * d_), data_.begin() + std::ptrdiff_t(p * d_)); }
        fin_ = true;
    }
    std::size_t list_begin(std::size_t l) const { return off_[l]; }
    std::size_t list_end(std::size_t l) const { return off_[l + 1]; }
    const float* data() const { return data_.data(); }
    std::uint64_t id_at(std::size_t r) const { return ids_[r]; }

    // the nprobe nearest lists of q, by (squared distance, index)
    std::vector<std::uint32_t> probes(std::span<const float> q, std::size_t nprobe) const {
        nprobe = std::min(nprobe, k_);
        std::vector<std::pair<float, std::uint32_t>> cd(k_);
        for (std::size_t c = 0; c < k_; ++c) cd[c] = {detail::ivf_sq(q.data(), c_.data() + c * d_, d_), std::uint32_t(c)};
        std::partial_sort(cd.begin(), cd.begin() + std::ptrdiff_t(nprobe), cd.end());
        std::vector<std::uint32_t> out(nprobe); for (std::size_t i = 0; i < nprobe; ++i) out[i] = cd[i].second;
        return out;
    }
    // exact k nearest among the rows of the given lists (finalize() first); rows_scanned += rows read
    std::vector<hit> search_lists(std::span<const float> q, std::size_t k, std::span<const std::uint32_t> lists, std::size_t* rows_scanned = nullptr) const {
        std::vector<hit> heap; heap.reserve(k + 1); const l2 m;
        for (auto l : lists) for (std::size_t r = off_[l]; r < off_[l + 1]; ++r) {
            const hit h{ids_[r], m.distance(q, std::span<const float>(data_.data() + r * d_, d_))};
            if (heap.size() < k) { heap.push_back(h); std::push_heap(heap.begin(), heap.end(), hit_before); }
            else if (hit_before(h, heap.front())) { std::pop_heap(heap.begin(), heap.end(), hit_before); heap.back() = h; std::push_heap(heap.begin(), heap.end(), hit_before); }
            if (rows_scanned) ++*rows_scanned;
        }
        std::sort_heap(heap.begin(), heap.end(), hit_before);
        return heap;
    }
    std::vector<hit> search(std::span<const float> q, std::size_t k, std::size_t nprobe) {
        finalize(); if (q.size() != d_) forge::raise(forge::errc::invalid_argument, "ivf_flat: query has {} floats, expected {}", q.size(), d_);
        auto p = probes(q, nprobe); return search_lists(q, k, p);
    }

    std::vector<std::vector<hit>> search_batch_device(std::span<const std::vector<float>> qs, std::size_t k, std::size_t nprobe, const scan_fn& fn, forge::task::pool* pool = nullptr, bool* fell_back = nullptr, double* upload_s = nullptr) {
        finalize(pool);
        const std::size_t nq = qs.size(); nprobe = std::min(nprobe, k_);
        std::vector<std::vector<hit>> out(nq);
        if (fell_back) *fell_back = false;
        if (nq == 0 || k == 0) return out;
        std::vector<std::uint32_t> rng(nq * nprobe * 2); std::vector<std::vector<std::uint32_t>> pr(nq);
        forge::task::parallel_blocks(pool, nq, 1, [&](std::size_t i, std::size_t, std::size_t) { pr[i] = probes(qs[i], nprobe);
            for (std::size_t p = 0; p < nprobe; ++p) { rng[(i * nprobe + p) * 2] = std::uint32_t(off_[pr[i][p]]); rng[(i * nprobe + p) * 2 + 1] = std::uint32_t(off_[pr[i][p] + 1]); } });
        std::vector<float> qm(nq * d_), od(nq * kCand); std::vector<std::uint32_t> oi(nq * kCand);
        for (std::size_t i = 0; i < nq; ++i) std::copy(qs[i].begin(), qs[i].end(), qm.begin() + std::ptrdiff_t(i * d_));
        double up = 0;
        const bool ok = k <= kCand && fn && fn(data_.data(), ids_.size(), d_, qm.data(), nq, rng.data(), nprobe, od.data(), oi.data(), &up);
        if (upload_s) *upload_s = up;
        if (!ok && fell_back) *fell_back = true;
        const l2 m;
        forge::task::parallel_blocks(pool, nq, 1, [&](std::size_t i, std::size_t, std::size_t) {
            if (!ok) { out[i] = search_lists(qs[i], k, pr[i]); return; }
            std::vector<hit> hs;
            for (std::size_t c = 0; c < kCand; ++c) { const auto r = oi[i * kCand + c]; if (r < ids_.size()) hs.push_back({ids_[r], m.distance(std::span<const float>(qs[i]), std::span<const float>(data_.data() + std::size_t(r) * d_, d_))}); }
            std::sort(hs.begin(), hs.end(), hit_before); if (hs.size() > k) hs.resize(k); out[i] = std::move(hs);
        });
        return out;
    }

    std::vector<std::vector<hit>> search_batch_matmul(std::span<const std::vector<float>> qs, std::size_t k, std::size_t nprobe, const matmul_fn& mm, forge::task::pool* pool = nullptr, std::size_t* device_calls = nullptr) {
        finalize(pool);
        const std::size_t nq = qs.size(); nprobe = std::min(nprobe, k_);
        std::vector<std::vector<hit>> out(nq);
        if (nq == 0 || k == 0) return out;
        std::vector<std::vector<std::uint32_t>> pr(nq);
        forge::task::parallel_blocks(pool, nq, 1, [&](std::size_t i, std::size_t, std::size_t) { pr[i] = probes(qs[i], nprobe); });
        std::vector<std::vector<std::uint32_t>> who(k_); for (std::size_t i = 0; i < nq; ++i) for (auto l : pr[i]) who[l].push_back(std::uint32_t(i));
        std::vector<std::vector<std::uint32_t>> cand(nq);   // candidate rows per query
        constexpr std::size_t kKeep = 32;
        std::vector<float> norm(ids_.size()); for (std::size_t r = 0; r < ids_.size(); ++r) { double s = 0; for (std::size_t j = 0; j < d_; ++j) s += double(data_[r * d_ + j]) * data_[r * d_ + j]; norm[r] = float(s); }
        std::vector<float> qm, y;
        for (std::size_t l = 0; l < k_; ++l) {
            const std::size_t rb = off_[l], rn = off_[l + 1] - rb, qn = who[l].size();
            if (!qn || !rn) continue;
            qm.resize(qn * d_); y.resize(qn * rn);
            for (std::size_t i = 0; i < qn; ++i) std::copy(qs[who[l][i]].begin(), qs[who[l][i]].end(), qm.begin() + std::ptrdiff_t(i * d_));
            bool ok = mm && mm(qm.data(), qn, d_, data_.data() + rb * d_, rn, y.data());
            if (ok && device_calls) ++*device_calls;
            for (std::size_t i = 0; i < qn; ++i) {
                auto& cs = cand[who[l][i]];
                if (!ok) { for (std::size_t r = 0; r < rn; ++r) cs.push_back(std::uint32_t(rb + r)); continue; }
                double qq = 0; for (std::size_t j = 0; j < d_; ++j) qq += double(qm[i * d_ + j]) * qm[i * d_ + j];
                std::vector<std::pair<float, std::uint32_t>> ds(rn);
                for (std::size_t r = 0; r < rn; ++r) ds[r] = {float(qq) + norm[rb + r] - 2.0f * y[i * rn + r], std::uint32_t(rb + r)};
                const std::size_t keep = std::min(kKeep, rn);
                std::partial_sort(ds.begin(), ds.begin() + std::ptrdiff_t(keep), ds.end());
                for (std::size_t r = 0; r < keep; ++r) cs.push_back(ds[r].second);
            }
        }
        const l2 m;
        forge::task::parallel_blocks(pool, nq, 1, [&](std::size_t i, std::size_t, std::size_t) {
            std::vector<hit> hs; for (auto r : cand[i]) hs.push_back({ids_[r], m.distance(std::span<const float>(qs[i]), std::span<const float>(data_.data() + std::size_t(r) * d_, d_))});
            std::sort(hs.begin(), hs.end(), hit_before); if (hs.size() > k) hs.resize(k); out[i] = std::move(hs); });
        return out;
    }

private:
    std::vector<float> c_; std::size_t d_, k_;
    std::vector<std::uint64_t> ids_, pend_ids_; std::vector<float> data_, pend_; std::vector<std::size_t> off_; bool fin_ = false;
};

} // namespace forge::ann
