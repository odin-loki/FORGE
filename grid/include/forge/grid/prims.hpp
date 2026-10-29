#pragma once
// grid::prims — portable primitives (CPU backend implementations; GPU backends lower the same specs).
#include <forge/grid/grid.hpp>
#include <forge/rand/philox.hpp>
#include <algorithm>
#include <numeric>

namespace forge::grid::prims {

template <class T, class Op = std::plus<T>> T reduce(span<const T> x, T init = T{}, Op op = {}) {
    std::size_t n = x.size(), parts = std::max<std::size_t>(1, std::min<std::size_t>(64, n / 4096));
    std::vector<T> partial(parts, init);
    task::parallel_for(parts, 1, [&](std::size_t b, std::size_t e) {
        for (std::size_t p = b; p < e; ++p) {
            std::size_t lo = n * p / parts, hi = n * (p + 1) / parts; T acc = init;
            for (std::size_t i = lo; i < hi; ++i) acc = op(acc, x[i]);
            partial[p] = acc;
        }
    });
    T r = init; for (auto& v : partial) r = op(r, v); return r;
}

// Two-pass blocked scan (inclusive or exclusive).
template <class T> void scan(span<const T> x, span<T> y, bool inclusive = true) {
    std::size_t n = x.size(), parts = std::max<std::size_t>(1, std::min<std::size_t>(64, n / 4096));
    std::vector<T> sums(parts + 1, T{});
    task::parallel_for(parts, 1, [&](std::size_t b, std::size_t e) {
        for (std::size_t p = b; p < e; ++p) {
            std::size_t lo = n * p / parts, hi = n * (p + 1) / parts; T s{};
            for (std::size_t i = lo; i < hi; ++i) s += x[i];
            sums[p + 1] = s;
        }
    });
    for (std::size_t p = 1; p <= parts; ++p) sums[p] += sums[p - 1];
    task::parallel_for(parts, 1, [&](std::size_t b, std::size_t e) {
        for (std::size_t p = b; p < e; ++p) {
            std::size_t lo = n * p / parts, hi = n * (p + 1) / parts; T s = sums[p];
            for (std::size_t i = lo; i < hi; ++i) { if (inclusive) { s += x[i]; y[i] = s; } else { y[i] = s; s += x[i]; } }
        }
    });
}

// Stream compaction: keep x[i] where pred(x[i]); returns count.
template <class T, class P> std::size_t compact(span<const T> x, span<T> out, P pred) {
    std::vector<std::uint32_t> flags(x.size()), pos(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) flags[i] = pred(x[i]) ? 1u : 0u;
    scan<std::uint32_t>(flags, pos, false);
    std::size_t c = x.empty() ? 0 : pos.back() + flags.back();
    for (std::size_t i = 0; i < x.size(); ++i) if (flags[i]) out[pos[i]] = x[i];
    return c;
}

// LSD radix sort on 32-bit keys (optionally carrying values).
inline void radix_sort(span<std::uint32_t> keys, span<std::uint32_t> vals = {}) {
    std::vector<std::uint32_t> k2(keys.size()), v2(vals.size());
    span<std::uint32_t> ka = keys, kb = k2, va = vals, vb = v2;
    for (int sh = 0; sh < 32; sh += 8) {
        std::array<std::size_t, 257> cnt{};
        for (auto k : ka) ++cnt[((k >> sh) & 0xff) + 1];
        for (int i = 0; i < 256; ++i) cnt[std::size_t(i) + 1] += cnt[std::size_t(i)];
        for (std::size_t i = 0; i < ka.size(); ++i) {
            auto d = cnt[(ka[i] >> sh) & 0xff]++;
            kb[d] = ka[i]; if (!va.empty()) vb[d] = va[i];
        }
        std::swap(ka, kb); std::swap(va, vb);
    }
    // 4 passes: result is back in the original buffers.
}

inline void histogram(span<const std::uint32_t> x, span<std::uint32_t> bins) {
    for (auto v : x) if (v < bins.size()) ++bins[v];
}

// Top-k (values descending) returning indices.
inline std::vector<std::uint32_t> top_k(span<const float> x, std::size_t k) {
    std::vector<std::uint32_t> idx(x.size());
    std::iota(idx.begin(), idx.end(), 0u);
    k = std::min(k, x.size());
    std::partial_sort(idx.begin(), idx.begin() + std::ptrdiff_t(k), idx.end(),
                      [&](auto a, auto b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
    idx.resize(k);
    return idx;
}

// GEMM: C[m,n] = A[m,k] * B[n,k]^T, rows split across the pool, micro-kernel from lane dispatch.
inline void gemm_nt(const float* A, const float* B, float* C, std::size_t m, std::size_t n, std::size_t k) {
    const auto& ker = lane::active();
    if (m == 1) {   // GEMV: split over output columns
        task::parallel_for(n, 64, [&](std::size_t b, std::size_t e) { ker.gemm_nt(A, B + b * k, C + b, 1, e - b, k);
            // gemm_nt writes C[j] with stride n=e-b for m=1 -> contiguous; fine.
        });
        return;
    }
    task::parallel_for((m + 3) / 4, 1, [&](std::size_t b, std::size_t e) {
        std::size_t r0 = b * 4, r1 = std::min(m, e * 4);
        // compute rows [r0, r1) against all n columns, in column blocks to keep B in cache
        constexpr std::size_t NB = 64;
        std::vector<float> tmp((r1 - r0) * NB);
        for (std::size_t j = 0; j < n; j += NB) {
            std::size_t nb = std::min(NB, n - j);
            ker.gemm_nt(A + r0 * k, B + j * k, tmp.data(), r1 - r0, nb, k);
            for (std::size_t r = 0; r < r1 - r0; ++r) std::copy_n(&tmp[r * nb], nb, C + (r0 + r) * n + j);
        }
    });
}

// Philox RNG fill: identical sequence on every backend for (seed, offset).
inline void philox_uniform(span<float> out, std::uint64_t seed, std::uint64_t offset = 0) {
    rand::philox4x32 g{{std::uint32_t(seed), std::uint32_t(seed >> 32)}};
    for (std::size_t i = 0; i < out.size(); i += 4) {
        auto b = g(offset + i / 4);
        for (std::size_t j = 0; j < 4 && i + j < out.size(); ++j) out[i + j] = rand::philox4x32::to_unit(b[j]);
    }
}

} // namespace forge::grid::prims
