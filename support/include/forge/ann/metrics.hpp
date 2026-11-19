#pragma once
// forge::ann: hits, the metric concept, the vector codec, and the built-in metrics.
//
//   hit{id, dist}              ordered by (dist, id); NaN distances count as +infinity
//   metric<M, V>               m.distance(a, b) -> float, smaller = closer. Optional on M, used when present by flat_index:
//                              void distances(const V& q, std::span<const V> xs, std::span<float> out) const;
//   codec<V>                   encode(const V&, std::vector<std::byte>&) / decode(std::span<const std::byte>), for hnsw save/load
//
// Built-in metrics (all are stateless value types, usable as M):
//   l2                  Euclidean distance on contiguous float ranges (std::vector<float>, std::span<const float>)
//   l2_squared          the same without the square root (same ordering)
//   cosine              1 - a.b / (|a||b|); 1 when either vector is zero
//   inner_product       -(a.b): smaller = larger dot product
//   hamming             popcount(a xor b) over bit vectors held in std::vector<std::uint64_t> (any contiguous range of uint64)
//   weighted_jaccard    1 - sum(min w) / sum(max w) over sparse vectors: sparse_vec = sorted-by-feature (feature, weight) pairs
//   minhash_jaccard     1 - (fraction of equal slots) between two minhash_sig signatures; minhash_sketch / wminhash_sketch
//                       build them from a feature set / from a sparse_vec (Ioffe's consistent weighted sampling, an unbiased
//                       estimator of weighted Jaccard).
// All arithmetic is plain scalar C++ in a fixed order, so distances are bit-identical on every thread and across runs.
#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace forge::ann {

struct hit {
    std::uint64_t id;
    float dist;
    bool operator==(const hit&) const = default;
};
// The total order (dist, id); NaN distances sort after every number.
inline bool hit_before(const hit& a, const hit& b) {
    const bool an = a.dist != a.dist, bn = b.dist != b.dist;
    if (an != bn) return bn;
    if (!an && a.dist != b.dist) return a.dist < b.dist;
    return a.id < b.id;
}
inline bool operator<(const hit& a, const hit& b) { return hit_before(a, b); }

template <class M, class V>
concept metric = requires(const M& m, const V& a, const V& b) {
    { m.distance(a, b) } -> std::convertible_to<float>;
};

template <class M, class V>
concept batch_metric = metric<M, V> && requires(const M& m, const V& q, std::span<const V> xs, std::span<float> out) {
    { m.distances(q, xs, out) };
};

// Specialise for the vector type stored in an hnsw that is saved: encode appends the bytes, decode reverses it.
template <class V> struct codec;

namespace detail {
template <class R> concept float_range = std::ranges::contiguous_range<R> && std::ranges::sized_range<R> && std::same_as<std::remove_cv_t<std::ranges::range_value_t<R>>, float>;
template <class R> concept u64_range = std::ranges::contiguous_range<R> && std::ranges::sized_range<R> && std::same_as<std::remove_cv_t<std::ranges::range_value_t<R>>, std::uint64_t>;

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
// 8 independent accumulators in a fixed order: vectorisable, and the same value everywhere.
template <class A, class B> inline void dot_and_norms(const A& a, const B& b, float& dot, float& na, float& nb) {
    const std::size_t n = std::min<std::size_t>(std::ranges::size(a), std::ranges::size(b));
    const float* x = std::ranges::data(a); const float* y = std::ranges::data(b);
    float d[8] = {}, p[8] = {}, q[8] = {};
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; ++j) { d[j] += x[i + j] * y[i + j]; p[j] += x[i + j] * x[i + j]; q[j] += y[i + j] * y[i + j]; }
    for (; i < n; ++i) { d[0] += x[i] * y[i]; p[0] += x[i] * x[i]; q[0] += y[i] * y[i]; }
    auto red = [](const float* v) { return ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7])); };
    dot = red(d); na = red(p); nb = red(q);
}
template <class A, class B> inline float dot(const A& a, const B& b) {
    const std::size_t n = std::min<std::size_t>(std::ranges::size(a), std::ranges::size(b));
    const float* x = std::ranges::data(a); const float* y = std::ranges::data(b);
    float d[8] = {}; std::size_t i = 0;
    for (; i + 8 <= n; i += 8) for (int j = 0; j < 8; ++j) d[j] += x[i + j] * y[i + j];
    for (; i < n; ++i) d[0] += x[i] * y[i];
    return ((d[0] + d[4]) + (d[1] + d[5])) + ((d[2] + d[6]) + (d[3] + d[7]));
}
template <class A, class B> inline float sqdist(const A& a, const B& b) {
    const std::size_t n = std::min<std::size_t>(std::ranges::size(a), std::ranges::size(b));
    const float* x = std::ranges::data(a); const float* y = std::ranges::data(b);
    float d[8] = {}; std::size_t i = 0;
    for (; i + 8 <= n; i += 8) for (int j = 0; j < 8; ++j) { const float t = x[i + j] - y[i + j]; d[j] += t * t; }
    for (; i < n; ++i) { const float t = x[i] - y[i]; d[0] += t * t; }
    return ((d[0] + d[4]) + (d[1] + d[5])) + ((d[2] + d[6]) + (d[3] + d[7]));
}
} // namespace detail

struct l2_squared {
    template <detail::float_range V> float distance(const V& a, const V& b) const { return detail::sqdist(a, b); }
    template <detail::float_range V> void distances(const V& q, std::span<const V> xs, std::span<float> out) const {
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = detail::sqdist(q, xs[i]);
    }
};
struct l2 {
    template <detail::float_range V> float distance(const V& a, const V& b) const { return std::sqrt(detail::sqdist(a, b)); }
    template <detail::float_range V> void distances(const V& q, std::span<const V> xs, std::span<float> out) const {
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = std::sqrt(detail::sqdist(q, xs[i]));
    }
};
struct cosine {
    template <detail::float_range V> float distance(const V& a, const V& b) const {
        float d, na, nb; detail::dot_and_norms(a, b, d, na, nb);
        if (na <= 0 || nb <= 0) return 1.0f;
        return 1.0f - d / (std::sqrt(na) * std::sqrt(nb));
    }
};
struct inner_product {
    template <detail::float_range V> float distance(const V& a, const V& b) const { return -detail::dot(a, b); }
    template <detail::float_range V> void distances(const V& q, std::span<const V> xs, std::span<float> out) const {
        for (std::size_t i = 0; i < xs.size(); ++i) out[i] = -detail::dot(q, xs[i]);
    }
};
struct hamming {
    template <detail::u64_range V> float distance(const V& a, const V& b) const {
        const std::size_t n = std::min<std::size_t>(std::ranges::size(a), std::ranges::size(b));
        const std::uint64_t* x = std::ranges::data(a); const std::uint64_t* y = std::ranges::data(b);
        std::uint32_t c = 0;
        for (std::size_t i = 0; i < n; ++i) c += std::uint32_t(std::popcount(x[i] ^ y[i]));
        for (std::size_t i = n; i < std::ranges::size(a); ++i) c += std::uint32_t(std::popcount(x[i]));   // missing words count as zero
        for (std::size_t i = n; i < std::ranges::size(b); ++i) c += std::uint32_t(std::popcount(y[i]));
        return float(c);
    }
};

// Sparse feature vector: (feature, weight) pairs sorted by strictly increasing feature, weights >= 0.
using sparse_vec = std::vector<std::pair<std::uint64_t, float>>;
struct weighted_jaccard {
    float distance(const sparse_vec& a, const sparse_vec& b) const {
        double mn = 0, mx = 0; std::size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i].first == b[j].first) { mn += std::min(a[i].second, b[j].second); mx += std::max(a[i].second, b[j].second); ++i; ++j; }
            else if (a[i].first < b[j].first) mx += a[i++].second;
            else mx += b[j++].second;
        }
        for (; i < a.size(); ++i) mx += a[i].second;
        for (; j < b.size(); ++j) mx += b[j].second;
        return mx <= 0 ? 0.0f : float(1.0 - mn / mx);
    }
};

using minhash_sig = std::vector<std::uint32_t>;
// k-slot MinHash of a feature set (duplicates are harmless). Empty set: every slot 0xFFFFFFFF.
inline minhash_sig minhash_sketch(std::span<const std::uint64_t> features, std::size_t k, std::uint64_t seed = 0x1234abcd) {
    minhash_sig s(k, 0xFFFFFFFFu);
    for (std::uint64_t f : features) {
        const std::uint64_t base = detail::splitmix64(f ^ seed);
        for (std::size_t i = 0; i < k; ++i) {
            const std::uint32_t h = std::uint32_t(detail::splitmix64(base + i * 0x9E3779B97F4A7C15ull) >> 32);
            if (h < s[i]) s[i] = h;
        }
    }
    return s;
}
// Weighted MinHash by Ioffe's improved consistent weighted sampling (ICWS): P[slot equal] = weighted Jaccard.
inline minhash_sig wminhash_sketch(const sparse_vec& v, std::size_t k, std::uint64_t seed = 0x1234abcd) {
    minhash_sig s(k, 0xFFFFFFFFu);
    auto unit = [](std::uint64_t h) { return (double(h >> 11) + 0.5) * (1.0 / 9007199254740992.0); };   // (0, 1)
    for (std::size_t i = 0; i < k; ++i) {
        double best = std::numeric_limits<double>::infinity(); std::uint32_t val = 0xFFFFFFFFu;
        for (auto [f, w] : v) {
            if (!(w > 0)) continue;
            std::uint64_t h = detail::splitmix64(f ^ seed ^ (std::uint64_t(i + 1) * 0xD6E8FEB86659FD93ull));
            const double u1 = unit(h = detail::splitmix64(h)), u2 = unit(h = detail::splitmix64(h)), u3 = unit(h = detail::splitmix64(h)),
                         u4 = unit(h = detail::splitmix64(h)), beta = unit(detail::splitmix64(h));
            const double r = -std::log(u1 * u2), c = -std::log(u3 * u4);   // Gamma(2, 1)
            const double t = std::floor(std::log(double(w)) / r + beta);
            const double y = std::exp(r * (t - beta));
            const double a = c / (y * std::exp(r));
            if (a < best) { best = a; val = std::uint32_t(detail::splitmix64(f * 31 + std::uint64_t(std::int64_t(t))) >> 32); }
        }
        s[i] = val;
    }
    return s;
}
struct minhash_jaccard {
    float distance(const minhash_sig& a, const minhash_sig& b) const {
        const std::size_t n = std::min(a.size(), b.size());
        if (n == 0) return 1.0f;
        std::size_t eq = 0;
        for (std::size_t i = 0; i < n; ++i) eq += a[i] == b[i];
        return 1.0f - float(eq) / float(n);
    }
};

// ---- codecs ----------------------------------------------------------------------------------------------------
static_assert(std::endian::native == std::endian::little, "forge::ann codecs write little-endian by copying memory");
namespace detail {
template <class T> inline void put(std::vector<std::byte>& out, const T* p, std::size_t n) {
    const auto* b = reinterpret_cast<const std::byte*>(p); out.insert(out.end(), b, b + n * sizeof(T));
}
template <class T> inline void put1(std::vector<std::byte>& out, T v) { put(out, &v, 1); }
} // namespace detail
template <class T> requires (std::same_as<T, float> || std::same_as<T, std::uint64_t> || std::same_as<T, std::uint32_t>)
struct codec<std::vector<T>> {
    static void encode(const std::vector<T>& v, std::vector<std::byte>& out) { detail::put(out, v.data(), v.size()); }
    static std::vector<T> decode(std::span<const std::byte> b) {
        std::vector<T> v(b.size() / sizeof(T));
        if (!v.empty()) std::memcpy(v.data(), b.data(), v.size() * sizeof(T));
        return v;
    }
};
template <> struct codec<sparse_vec> {
    static void encode(const sparse_vec& v, std::vector<std::byte>& out) {
        for (auto& [f, w] : v) { detail::put1(out, f); detail::put1(out, w); }
    }
    static sparse_vec decode(std::span<const std::byte> b) {
        sparse_vec v(b.size() / 12);
        for (std::size_t i = 0; i < v.size(); ++i) { std::memcpy(&v[i].first, b.data() + 12 * i, 8); std::memcpy(&v[i].second, b.data() + 12 * i + 8, 4); }
        return v;
    }
};

} // namespace forge::ann
