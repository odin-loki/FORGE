#pragma once
// forge::ann::gpu_flat_l2: batched exact nearest-neighbour search where a device does the heavy part.
//
//   ann::gpu_flat_l2 ix(matmul, dim);            // matmul: y[rows,cols] = x[rows,k] . w[cols,k]^T on f32, e.g. graph::ann_matmul(reg)
//   ix.add(id, vec); ...
//   auto hits = ix.search_batch(queries, k, &pool);     // hits[q] == flat_index<std::vector<float>, l2>::search(queries[q], k)
//
// The result is exactly flat_index's, not an approximation. The device computes all squared distances of a chunk of vectors with one
// matrix product (|q|^2 + |x|^2 - 2 q.x); those values carry rounding error, bounded for any order of summation by
//     err <= 2 * (dim + 6) * 2^-24 * (|q|^2 + max |x|^2)      (error_scale multiplies it)
// so the true k-th distance is at most A_k + err, where A_k is the k-th smallest approximate value. Every vector whose approximate
// distance is within A_k + 2 err of the best is kept and re-ranked on the host with the plain l2 metric (the same arithmetic as
// flat_index), then the k best by (distance, id) are returned: ties and order are flat_index's. The matmul must be IEEE f32 (not TF32
// or half precision) for the bound to hold. If the callable fails (returns false: no device, out of memory) the batch runs on the CPU
// path and the result is the same.
#include <forge/ann/flat.hpp>
#include <forge/core/error.hpp>
#include <limits>
#include <forge/task/algorithms.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace forge::ann {

class gpu_flat_l2 {
public:
    // y[rows * cols] (row-major) = x[rows * k] . w[cols * k]^T. Returns false when the device could not do it.
    using matmul_fn = std::function<bool(const float* x, std::size_t rows, std::size_t k, const float* w, std::size_t cols, float* y)>;
    struct report { std::size_t device_calls = 0, candidates = 0; bool fell_back = false; };

    gpu_flat_l2(matmul_fn mm, std::size_t dim, std::size_t chunk_rows = 1u << 16, std::size_t query_block = 128, double error_scale = 1.0)
        : mm_(std::move(mm)), dim_(dim), chunk_(chunk_rows ? chunk_rows : 1), qblock_(query_block ? query_block : 1), scale_(error_scale) {}

    void add(std::uint64_t id, std::span<const float> v) {
        if (v.size() != dim_) forge::raise(forge::errc::invalid_argument, "gpu_flat_l2::add: expected {} floats, got {}", dim_, v.size());
        ids_.push_back(id); data_.insert(data_.end(), v.begin(), v.end());
        double s = 0; for (float f : v) s += double(f) * double(f);
        norm_.push_back(float(s)); max_norm_ = std::max(max_norm_, double(norm_.back()));
    }
    std::size_t size() const { return ids_.size(); }

    // hits[q] = the k nearest of queries[q] by (l2 distance, id): equal to flat_index::search. pool = nullptr: serial.
    std::vector<std::vector<hit>> search_batch(std::span<const std::vector<float>> queries, std::size_t k, forge::task::pool* pool = nullptr, report* rep = nullptr) const {
        const std::size_t nq = queries.size(), n = ids_.size();
        std::vector<std::vector<hit>> out(nq);
        if (rep) *rep = {};
        if (nq == 0 || k == 0 || n == 0) return out;
        for (auto& q : queries) if (q.size() != dim_) forge::raise(forge::errc::invalid_argument, "gpu_flat_l2: query has {} floats, expected {}", q.size(), dim_);
        const double u = std::ldexp(1.0, -24);
        for (std::size_t q0 = 0; q0 < nq; q0 += qblock_) {
            const std::size_t qn = std::min(qblock_, nq - q0);
            std::vector<float> qm(qn * dim_), qnorm(qn);
            for (std::size_t i = 0; i < qn; ++i) {
                std::copy(queries[q0 + i].begin(), queries[q0 + i].end(), qm.begin() + std::ptrdiff_t(i * dim_));
                double s = 0; for (float f : queries[q0 + i]) s += double(f) * double(f); qnorm[i] = float(s);
            }
            struct cand { float approx; std::uint32_t idx; };
            std::vector<std::vector<cand>> cands(qn);
            bool ok = true;
            std::vector<float> y(qn * std::min(chunk_, n));
            for (std::size_t c0 = 0; c0 < n && ok; c0 += chunk_) {
                const std::size_t cn = std::min(chunk_, n - c0);
                if (!mm_(qm.data(), qn, dim_, data_.data() + c0 * dim_, cn, y.data())) { ok = false; break; }
                if (rep) ++rep->device_calls;
                forge::task::parallel_blocks(pool, qn, 1, [&](std::size_t i, std::size_t, std::size_t) {
                    std::vector<float> d(cn);
                    const float* yi = y.data() + i * cn;
                    for (std::size_t j = 0; j < cn; ++j) d[j] = qnorm[i] + norm_[c0 + j] - 2.0f * yi[j];
                    const double err = 2.0 * scale_ * double(dim_ + 6) * u * (double(qnorm[i]) + max_norm_);
                    float kth;
                    if (cn > k) { std::vector<float> t = d; std::nth_element(t.begin(), t.begin() + std::ptrdiff_t(k - 1), t.end()); kth = t[k - 1]; } else kth = std::numeric_limits<float>::infinity();
                    const float limit = float(double(kth) + 2.0 * err * 1.0000001) + std::numeric_limits<float>::min();
                    for (std::size_t j = 0; j < cn; ++j) if (d[j] <= limit || d[j] != d[j]) cands[i].push_back({d[j], std::uint32_t(c0 + j)});
                });
            }
            if (!ok) {   // the device could not run: the plain exact path for this and every further block
                if (rep) rep->fell_back = true;
                flat_index<std::vector<float>, l2> flat;
                for (std::size_t j = 0; j < n; ++j) flat.add(ids_[j], std::vector<float>(data_.begin() + std::ptrdiff_t(j * dim_), data_.begin() + std::ptrdiff_t((j + 1) * dim_)));
                forge::task::parallel_blocks(pool, qn, 1, [&](std::size_t i, std::size_t, std::size_t) { out[q0 + i] = flat.search(queries[q0 + i], k, nullptr); });
                continue;
            }
            forge::task::parallel_blocks(pool, qn, 1, [&](std::size_t i, std::size_t, std::size_t) {
                auto& cs = cands[i];
                const double err = 2.0 * scale_ * double(dim_ + 6) * u * (double(qnorm[i]) + max_norm_);
                if (cs.size() > k) {   // global threshold from all collected approximations, then drop what cannot be in the top k
                    std::vector<float> t; t.reserve(cs.size()); for (auto& c : cs) t.push_back(c.approx);
                    std::nth_element(t.begin(), t.begin() + std::ptrdiff_t(k - 1), t.end());
                    const float limit = float(double(t[k - 1]) + 2.0 * err * 1.0000001) + std::numeric_limits<float>::min();
                    std::erase_if(cs, [&](const cand& c) { return !(c.approx <= limit) && c.approx == c.approx; });
                }
                std::vector<hit> hs; hs.reserve(cs.size());
                l2 metric; std::span<const float> qs(queries[q0 + i]);
                for (auto& c : cs) hs.push_back({ids_[c.idx], metric.distance(qs, std::span<const float>(data_.data() + std::size_t(c.idx) * dim_, dim_))});
                std::sort(hs.begin(), hs.end(), hit_before);
                if (hs.size() > k) hs.resize(k);
                out[q0 + i] = std::move(hs);
            });
            if (rep) for (auto& cs : cands) rep->candidates += cs.size();
        }
        return out;
    }

private:
    matmul_fn mm_; std::size_t dim_, chunk_, qblock_; double scale_;
    std::vector<std::uint64_t> ids_; std::vector<float> data_, norm_; double max_norm_ = 0;
};

} // namespace forge::ann
