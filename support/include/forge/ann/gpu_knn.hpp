#pragma once
// forge::ann::gpu_knn_l2: batched exact nearest-neighbour search with the rows kept on a device and the top-k selected there.
//
//   ann::gpu_knn_l2 ix(graph::cuda_knn(), dim);   // knn: the device's 16 nearest per query (gpu::cuda_hip::runtime::knn_l2_16)
//   ix.add(id, vec); ...
//   auto hits = ix.search_batch(queries, 10);     // hits[q] == flat_index<std::vector<float>, l2>::search(queries[q], 10)
//
// Unlike gpu_flat_l2 (a device matmul whose whole distance matrix comes back to the host), the rows are uploaded once and stay
// on the device, and only 16 (approximate distance, row) pairs per query come back. The host re-scores those 16 with the plain
// l2 metric and keeps the k best by (distance, id), so ties and order are flat_index's. That is exact whenever the 16 hold the
// true k nearest, which the host checks per query: the device computes sum (q_j - x_j)^2 in f32 with every term non-negative,
// so each approximate value A is within a relative error g = 2 (dim + 4) 2^-24 of the true squared distance S, and the host's
// own l2 value (a float sum of non-negative terms, then sqrt) is within the same g of S. With A_k and A_16 the k-th and 16th
// smallest approximations, a row left out has S >= A_16 / (1 + g) and so a host value >= A_16 (1 - g) / (1 + g); the k
// candidates below A_k have host values <= A_k (1 + g) / (1 - g). When A_16 (1 - g)^2 > A_k (1 + g)^2 every row left out is
// strictly after the k-th candidate in flat_index's order. Otherwise (k > 16, ties or near-ties at the boundary, fewer than 16 finite candidates, non-finite data)
// that query is answered by the CPU flat path; report::fallback_queries counts them. If the device call fails the whole batch
// runs on the CPU path, with the same result.
#include <forge/ann/flat.hpp>
#include <forge/core/error.hpp>
#include <forge/task/algorithms.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace forge::ann {

class gpu_knn_l2 {
public:
    static constexpr std::size_t kCand = 16;   // candidates per query from the device
    // od/oi[q * 16 ..] = the 16 smallest squared distances of query q over the n rows x[n * d] (ascending, row 0xFFFFFFFF: none).
    // x stays the same pointer between calls while the index is unchanged, so a device can keep it resident.
    // *upload_s: seconds spent uploading x in this call (0 when it was already resident). false: the device could not run it.
    using knn_fn = std::function<bool(const float* x, std::size_t n, std::size_t d, const float* q, std::size_t nq, float* od, std::uint32_t* oi, double* upload_s)>;
    struct report { std::size_t device_calls = 0, fallback_queries = 0; double upload_s = 0; bool fell_back = false; };

    gpu_knn_l2(knn_fn fn, std::size_t dim, std::size_t max_batch = 4096) : fn_(std::move(fn)), dim_(dim), max_batch_(max_batch ? max_batch : 1) {}

    void add(std::uint64_t id, std::span<const float> v) {
        if (v.size() != dim_) forge::raise(forge::errc::invalid_argument, "gpu_knn_l2::add: expected {} floats, got {}", dim_, v.size());
        for (float f : v) if (!std::isfinite(f)) finite_ = false;
        ids_.push_back(id); data_.insert(data_.end(), v.begin(), v.end());
    }
    std::size_t size() const { return ids_.size(); }
    const float* data() const { return data_.data(); }

    std::vector<std::vector<hit>> search_batch(std::span<const std::vector<float>> queries, std::size_t k, forge::task::pool* pool = nullptr, report* rep = nullptr) const {
        const std::size_t nq = queries.size(), n = ids_.size();
        std::vector<std::vector<hit>> out(nq);
        if (rep) *rep = {};
        if (nq == 0 || k == 0 || n == 0) return out;
        for (auto& q : queries) if (q.size() != dim_) forge::raise(forge::errc::invalid_argument, "gpu_knn_l2: query has {} floats, expected {}", q.size(), dim_);
        const double g = 2.0 * double(dim_ + 4) * std::ldexp(1.0, -24);   // relative error bound of a squared distance, per side
        const l2 metric;
        auto exact = [&](std::size_t qi) {   // the CPU flat path for one query
            std::vector<hit> heap; heap.reserve(k + 1);
            std::span<const float> qs(queries[qi]);
            for (std::size_t j = 0; j < n; ++j) {
                const hit h{ids_[j], metric.distance(qs, std::span<const float>(data_.data() + j * dim_, dim_))};
                if (heap.size() < k) { heap.push_back(h); std::push_heap(heap.begin(), heap.end(), hit_before); }
                else if (hit_before(h, heap.front())) { std::pop_heap(heap.begin(), heap.end(), hit_before); heap.back() = h; std::push_heap(heap.begin(), heap.end(), hit_before); }
            }
            std::sort_heap(heap.begin(), heap.end(), hit_before);
            return heap;
        };
        bool device_ok = fn_ && k <= kCand && n > kCand && finite_;
        std::vector<float> qm, od; std::vector<std::uint32_t> oi;
        for (std::size_t q0 = 0; q0 < nq; q0 += max_batch_) {
            const std::size_t qn = std::min(max_batch_, nq - q0);
            if (device_ok) {
                qm.resize(qn * dim_); od.resize(qn * kCand); oi.resize(qn * kCand);
                for (std::size_t i = 0; i < qn; ++i) std::copy(queries[q0 + i].begin(), queries[q0 + i].end(), qm.begin() + std::ptrdiff_t(i * dim_));
                double up = 0;
                if (fn_(data_.data(), n, dim_, qm.data(), qn, od.data(), oi.data(), &up)) {
                    if (rep) { ++rep->device_calls; rep->upload_s += up; }
                } else { device_ok = false; if (rep) rep->fell_back = true; }
            }
            std::vector<char> fb(qn, 0);
            forge::task::parallel_blocks(pool, qn, 1, [&](std::size_t i, std::size_t, std::size_t) {
                const std::size_t qi = q0 + i;
                bool ok = device_ok;
                for (std::size_t c = 0; ok && c < kCand; ++c) { const std::uint32_t r = oi[i * kCand + c]; ok = r < n && std::isfinite(od[i * kCand + c]); }
                if (ok) ok = double(od[i * kCand + kCand - 1]) * (1.0 - g) * (1.0 - g) > double(od[i * kCand + k - 1]) * (1.0 + g) * (1.0 + g);
                if (!ok) { out[qi] = exact(qi); fb[i] = 1; return; }
                std::vector<hit> hs; hs.reserve(kCand);
                std::span<const float> qs(queries[qi]);
                for (std::size_t c = 0; c < kCand; ++c) { const std::uint32_t r = oi[i * kCand + c]; hs.push_back({ids_[r], metric.distance(qs, std::span<const float>(data_.data() + std::size_t(r) * dim_, dim_))}); }
                std::sort(hs.begin(), hs.end(), hit_before);
                hs.resize(k);
                out[qi] = std::move(hs);
            });
            if (rep && device_ok) for (char f : fb) rep->fallback_queries += std::size_t(f);
        }
        return out;
    }

private:
    knn_fn fn_; std::size_t dim_, max_batch_;
    std::vector<std::uint64_t> ids_; std::vector<float> data_; bool finite_ = true;
};

} // namespace forge::ann
