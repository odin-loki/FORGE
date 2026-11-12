#pragma once
// Tier A numeric algorithms: min/max/argmin/argmax, compare-count/any/all/find, clamp, prefix sums,
// histograms, transpose, dot/axpy for f64 and integers, bitset operations.
//
// Every function is a template on the target so each per-target translation unit (simd/src/tiera_*.cpp) gets
// its own copy compiled with its own flags; nothing here is a plain inline function, because the linker
// would keep one body for all tables. How each one is vectorised:
//   * f32 reductions, compares, clamp      explicit, through vec<float, Tg> / strip (tail by mask)
//   * i32 prefix sum                       explicit AVX2 / AVX-512 in-register scan; a scalar loop elsewhere
//   * f32 / f64 prefix sum                 a sequential loop: the result is defined as the left-to-right sum, so
//                                          it is bit-identical on every target (and not vectorised)
//   * f64 and integer dot/axpy, transpose,
//     histograms, bitset ops               loops written so the compiler vectorises them with the TU's flags
#include <forge/simd/vec.hpp>
#include <algorithm>
#include <cstdint>
#include <limits>
#if defined(_MSC_VER) && !defined(__clang__)
#include <bit>
#endif

namespace forge::simd::algo {

enum class cmp : int { eq = 0, ne = 1, lt = 2, le = 3, gt = 4, ge = 5 };

namespace bits {
template <Target Tg> inline int popcount64(std::uint64_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
    return std::popcount(v);
#else
    return __builtin_popcountll(v);
#endif
}
template <Target Tg> inline int ctz64(std::uint64_t v) {   // v != 0
#if defined(_MSC_VER) && !defined(__clang__)
    return std::countr_zero(v);
#else
    return __builtin_ctzll(v);
#endif
}
} // namespace bits

namespace numdetail {
// Bits of the lanes of v (restricted to the lanes valid in m) for which `v op t` holds, with C++ comparison
// semantics for NaN (only ne is true).
template <Target Tg> inline unsigned cmp_bits(cmp op, vec<float, Tg> v, vec<float, Tg> t, mask<float, Tg> m) {
    const unsigned valid = m.bits();
    switch (op) {
    case cmp::lt: return lt(v, t).bits() & valid;
    case cmp::gt: return lt(t, v).bits() & valid;
    case cmp::le: return ~(lt(t, v) | isnan(v) | isnan(t)).bits() & valid;
    case cmp::ge: return ~(lt(v, t) | isnan(v) | isnan(t)).bits() & valid;
    case cmp::eq: return ~(lt(v, t) | lt(t, v) | isnan(v) | isnan(t)).bits() & valid;
    default:      return (lt(v, t) | lt(t, v) | isnan(v) | isnan(t)).bits() & valid;
    }
}
template <Target Tg> inline bool cmp_scalar(cmp op, float v, float t) {
    switch (op) {
    case cmp::lt: return v < t; case cmp::gt: return v > t; case cmp::le: return v <= t;
    case cmp::ge: return v >= t; case cmp::eq: return v == t; default: return v != t;
    }
}
} // namespace numdetail

// ---- min / max over the non-NaN elements (+inf / -inf when there are none) ----
template <Target Tg> float min_f32(const float* x, std::size_t n) {
    const auto inf = splat<Tg>(std::numeric_limits<float>::infinity());
    auto acc = inf;
    strip<Tg, float>(n, [&](std::size_t i, auto m) {
        auto v = load(x + i, m);
        acc = simd::min(acc, select(m, select(isnan(v), inf, v), inf));
    });
    float t[64]; store(t, acc);
    float r = t[0]; for (int i = 1; i < vec<float, Tg>::N; ++i) r = t[i] < r ? t[i] : r;
    return r;
}
template <Target Tg> float max_f32(const float* x, std::size_t n) {
    const auto ninf = splat<Tg>(-std::numeric_limits<float>::infinity());
    auto acc = ninf;
    strip<Tg, float>(n, [&](std::size_t i, auto m) {
        auto v = load(x + i, m);
        acc = simd::max(acc, select(m, select(isnan(v), ninf, v), ninf));
    });
    float t[64]; store(t, acc);
    float r = t[0]; for (int i = 1; i < vec<float, Tg>::N; ++i) r = t[i] > r ? t[i] : r;
    return r;
}

// ---- compare against a threshold: count, any, all, first index (n when there is none) ----
template <Target Tg> std::size_t count_f32(const float* x, std::size_t n, cmp op, float t) {
    const auto vt = splat<Tg>(t); std::size_t c = 0;
    strip<Tg, float>(n, [&](std::size_t i, auto m) { c += std::size_t(bits::popcount64<Tg>(numdetail::cmp_bits<Tg>(op, load(x + i, m), vt, m))); });
    return c;
}
template <Target Tg> std::size_t find_f32(const float* x, std::size_t n, cmp op, float t) {
    const auto vt = splat<Tg>(t);
    constexpr std::size_t W = std::size_t(vec<float, Tg>::N);
    std::size_t i = 0;
    const auto full = mask<float, Tg>::all();
    for (; i + W <= n; i += W)
        if (unsigned b = numdetail::cmp_bits<Tg>(op, load<Tg>(x + i), vt, full)) return i + std::size_t(bits::ctz64<Tg>(b));
    if (i < n) {
        auto m = mask<float, Tg>::first_n(n - i);
        if (unsigned b = numdetail::cmp_bits<Tg>(op, load(x + i, m), vt, m)) return i + std::size_t(bits::ctz64<Tg>(b));
    }
    return n;
}
template <Target Tg> bool any_f32(const float* x, std::size_t n, cmp op, float t) { return find_f32<Tg>(x, n, op, t) != n; }
template <Target Tg> bool all_f32(const float* x, std::size_t n, cmp op, float t) {
    // all(v op t) == none(!(v op t)); the complement is taken per lane so NaN keeps C++ semantics
    const auto vt = splat<Tg>(t); bool ok = true;
    strip<Tg, float>(n, [&](std::size_t i, auto m) { ok = ok && numdetail::cmp_bits<Tg>(op, load(x + i, m), vt, m) == m.bits(); });
    return ok;
}

// Generic forms: pred(vec) -> mask. find_if returns the first index whose lane is set, n when none.
template <Target Tg, class P> std::size_t find_if(const float* x, std::size_t n, P&& pred) {
    constexpr std::size_t W = std::size_t(vec<float, Tg>::N);
    for (std::size_t i = 0; i < n; i += W) {
        auto m = mask<float, Tg>::first_n(n - i);
        if (unsigned b = pred(load(x + i, m)).bits() & m.bits()) return i + std::size_t(bits::ctz64<Tg>(b));
    }
    return n;
}
template <Target Tg, class P> std::size_t count_if(const float* x, std::size_t n, P&& pred) {
    std::size_t c = 0;
    strip<Tg, float>(n, [&](std::size_t i, auto m) { c += std::size_t(bits::popcount64<Tg>(pred(load(x + i, m)).bits() & m.bits())); });
    return c;
}
template <Target Tg, class P> bool any_of(const float* x, std::size_t n, P&& pred) { return find_if<Tg>(x, n, pred) != n; }
template <Target Tg, class P> bool all_of(const float* x, std::size_t n, P&& pred) {
    bool ok = true;
    strip<Tg, float>(n, [&](std::size_t i, auto m) { ok = ok && (pred(load(x + i, m)).bits() & m.bits()) == m.bits(); });
    return ok;
}

// Index of the first element equal to the minimum / maximum of the non-NaN elements; n if there is none.
template <Target Tg> std::size_t argmax_f32(const float* x, std::size_t n) { return find_f32<Tg>(x, n, cmp::eq, max_f32<Tg>(x, n)); }
template <Target Tg> std::size_t argmin_f32(const float* x, std::size_t n) { return find_f32<Tg>(x, n, cmp::eq, min_f32<Tg>(x, n)); }

// y = x < lo ? lo : x > hi ? hi : x (NaN passes through). x and y may be the same array.
template <Target Tg> void clamp_f32(const float* x, float* y, std::size_t n, float lo, float hi) {
    const auto vl = splat<Tg>(lo), vh = splat<Tg>(hi);
    strip<Tg, float>(n, [&](std::size_t i, auto m) {
        auto v = load(x + i, m);
        store(y + i, select(lt(v, vl), vl, select(lt(vh, v), vh, v)), m);
    });
}

// ---- prefix sums. inclusive: y[i] = x[0] + ... + x[i]; exclusive: y[i] = x[0] + ... + x[i-1], y[0] = 0. ----
// x and y may be the same array. Floating-point sums are accumulated left to right, one element at a time.
template <Target Tg, class T> void scan_add_seq(const T* x, T* y, std::size_t n, bool exclusive) {
    T acc = T(0);
    if (exclusive) for (std::size_t i = 0; i < n; ++i) { T v = x[i]; y[i] = acc; acc = T(acc + v); }
    else for (std::size_t i = 0; i < n; ++i) { acc = T(acc + x[i]); y[i] = acc; }
}
template <Target Tg> void scan_add_f32(const float* x, float* y, std::size_t n, bool exclusive) { scan_add_seq<Tg, float>(x, y, n, exclusive); }
template <Target Tg> void scan_add_f64(const double* x, double* y, std::size_t n, bool exclusive) { scan_add_seq<Tg, double>(x, y, n, exclusive); }
// 32-bit integers wrap modulo 2^32 (computed unsigned), so the vector form is exact.
template <Target Tg> void scan_add_i32(const std::int32_t* x, std::int32_t* y, std::size_t n, bool exclusive) {
    constexpr int K = impl::kind<Tg>();
    std::size_t i = 0; std::uint32_t carry = 0;
    if constexpr (K == 2) {
#if defined(__AVX512F__)
        __m512i c = _mm512_setzero_si512();
        const __m512i last = _mm512_set1_epi32(15);
        for (; i + 16 <= n; i += 16) {
            __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(x + i)), s = v;
            s = _mm512_add_epi32(s, _mm512_maskz_alignr_epi32(__mmask16(0xfffe), s, s, 15));   // lane j += lane j-1
            s = _mm512_add_epi32(s, _mm512_maskz_alignr_epi32(__mmask16(0xfffc), s, s, 14));
            s = _mm512_add_epi32(s, _mm512_maskz_alignr_epi32(__mmask16(0xfff0), s, s, 12));
            s = _mm512_add_epi32(s, _mm512_maskz_alignr_epi32(__mmask16(0xff00), s, s, 8));
            s = _mm512_add_epi32(s, c);
            _mm512_storeu_si512(reinterpret_cast<void*>(y + i), exclusive ? _mm512_sub_epi32(s, v) : s);
            c = _mm512_permutexvar_epi32(last, s);
        }
        carry = std::uint32_t(_mm_cvtsi128_si32(_mm512_castsi512_si128(c)));
#endif
    } else if constexpr (K == 1) {
#if defined(__AVX2__)
        __m256i c = _mm256_setzero_si256();
        const __m256i last = _mm256_set1_epi32(7);
        for (; i + 8 <= n; i += 8) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)), s = v;
            s = _mm256_add_epi32(s, _mm256_slli_si256(s, 4));    // within each 128-bit half
            s = _mm256_add_epi32(s, _mm256_slli_si256(s, 8));
            __m256i lo_total = _mm256_permutevar8x32_epi32(s, _mm256_setr_epi32(3, 3, 3, 3, 3, 3, 3, 3));
            s = _mm256_add_epi32(s, _mm256_blend_epi32(_mm256_setzero_si256(), lo_total, 0xf0));   // low half's total into the high half
            s = _mm256_add_epi32(s, c);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(y + i), exclusive ? _mm256_sub_epi32(s, v) : s);
            c = _mm256_permutevar8x32_epi32(s, last);
        }
        carry = std::uint32_t(_mm_cvtsi128_si32(_mm256_castsi256_si128(c)));
#endif
    }
    for (; i < n; ++i) {
        std::uint32_t v = std::uint32_t(x[i]);
        if (exclusive) { y[i] = std::int32_t(carry); carry += v; } else { carry += v; y[i] = std::int32_t(carry); }
    }
}

// ---- histograms ----
// hist[b] += number of bytes equal to b (hist has 256 counters and is added to, not cleared).
template <Target Tg> void histogram_u8(const std::uint8_t* x, std::size_t n, std::uint32_t* hist) {
    std::uint32_t h[4][256] = {};   // four tables break the store-to-load dependency between equal neighbours
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) { ++h[0][x[i]]; ++h[1][x[i + 1]]; ++h[2][x[i + 2]]; ++h[3][x[i + 3]]; }
    for (; i < n; ++i) ++h[0][x[i]];
    for (int b = 0; b < 256; ++b) hist[b] += h[0][b] + h[1][b] + h[2][b] + h[3][b];
}
// `bins` equal-width bins over [lo, hi): hist[min(bins - 1, floor((x - lo) * bins / (hi - lo)))] += 1 for
// lo <= x < hi; everything else (including NaN) is skipped. Returns how many were counted. The bin of a value
// is computed in single precision as float(x - lo) * float(bins / (hi - lo)), the same on every target.
template <Target Tg> std::size_t histogram_f32(const float* x, std::size_t n, float lo, float hi, std::uint32_t* hist, std::size_t bins) {
    if (bins == 0 || !(lo < hi)) return 0;
    const float scale = float(double(bins) / (double(hi) - double(lo)));
    const std::int32_t top = std::int32_t(bins - 1);
    std::size_t counted = 0; std::int32_t idx[256];
    for (std::size_t i = 0; i < n; i += 256) {
        const std::size_t m = std::min<std::size_t>(256, n - i);
        for (std::size_t j = 0; j < m; ++j) {   // vectorisable: compare, multiply, truncate, select
            float v = x[i + j]; bool in = v >= lo && v < hi;
            float s = in ? v - lo : 0.0f;
            float q = s * scale;
            std::int32_t b = std::int32_t(q);
            idx[j] = in ? (b > top ? top : b) : -1;
        }
        for (std::size_t j = 0; j < m; ++j) if (idx[j] >= 0) { ++hist[idx[j]]; ++counted; }
    }
    return counted;
}

// ---- transpose ----
// b[cols x rows] = a[rows x cols]^T, row-major, in 32 x 32 blocks so both arrays are walked with locality.
template <Target Tg, class T> void transpose(const T* a, T* b, std::size_t rows, std::size_t cols) {
    constexpr std::size_t B = 32;
    for (std::size_t i0 = 0; i0 < rows; i0 += B)
        for (std::size_t j0 = 0; j0 < cols; j0 += B) {
            const std::size_t i1 = std::min(rows, i0 + B), j1 = std::min(cols, j0 + B);
            for (std::size_t j = j0; j < j1; ++j) for (std::size_t i = i0; i < i1; ++i) b[j * rows + i] = a[i * cols + j];
        }
}
// In-place transpose of a square n x n matrix.
template <Target Tg, class T> void transpose_square(T* a, std::size_t n) {
    constexpr std::size_t B = 32;
    for (std::size_t i0 = 0; i0 < n; i0 += B)
        for (std::size_t j0 = i0; j0 < n; j0 += B) {
            const std::size_t i1 = std::min(n, i0 + B), j1 = std::min(n, j0 + B);
            for (std::size_t i = i0; i < i1; ++i)
                for (std::size_t j = (i0 == j0 ? i + 1 : j0); j < j1; ++j) { T t = a[i * n + j]; a[i * n + j] = a[j * n + i]; a[j * n + i] = t; }
        }
}

// ---- dot / axpy for f64 and integers ----
// f64 dot: eight running sums (element i goes to sum i mod 8), combined as ((0+4)+(2+6)) + ((1+5)+(3+7)).
// The compiler turns the eight sums into vector lanes; whether a*b+s is fused depends on the target, so
// tables can differ in the last bits.
template <Target Tg> double dot_f64(const double* a, const double* b, std::size_t n) {
    double s[8] = {};
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) for (int j = 0; j < 8; ++j) s[j] += a[i + j] * b[i + j];
    for (int j = 0; i < n; ++i, ++j) s[j] += a[i] * b[i];
    return ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
}
template <Target Tg> void axpy_f64(double a, const double* x, double* y, std::size_t n) { for (std::size_t i = 0; i < n; ++i) y[i] += a * x[i]; }
template <Target Tg> double sum_f64(const double* x, std::size_t n) {
    double s[8] = {};
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) for (int j = 0; j < 8; ++j) s[j] += x[i + j];
    for (int j = 0; i < n; ++i, ++j) s[j] += x[i];
    return ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
}
// Integer dots widen before multiplying, so the result is exact unless the 64-bit sum itself overflows (wraps).
template <Target Tg> std::int64_t dot_i32(const std::int32_t* a, const std::int32_t* b, std::size_t n) {
    std::uint64_t s = 0; for (std::size_t i = 0; i < n; ++i) s += std::uint64_t(std::int64_t(a[i]) * std::int64_t(b[i])); return std::int64_t(s);
}
template <Target Tg> std::int64_t dot_i16(const std::int16_t* a, const std::int16_t* b, std::size_t n) {
    std::int64_t s = 0; std::size_t i = 0;
    while (i < n) {   // blocks of 2^15 products fit an int32 partial sum of pairs only if widened: sum in int64 per block
        const std::size_t e = std::min(n, i + 32768); std::int64_t t = 0;
        for (; i < e; ++i) t += std::int32_t(a[i]) * std::int32_t(b[i]);
        s += t;
    }
    return s;
}
template <Target Tg> std::int64_t dot_i8(const std::int8_t* a, const std::int8_t* b, std::size_t n) {
    std::int64_t s = 0; std::size_t i = 0;
    while (i < n) {   // 2^16 products of at most 2^14 fit in int32
        const std::size_t e = std::min(n, i + 65536); std::int32_t t = 0;
        for (; i < e; ++i) t += std::int32_t(a[i]) * std::int32_t(b[i]);
        s += t;
    }
    return s;
}
// y += a * x, modulo 2^32.
template <Target Tg> void axpy_i32(std::int32_t a, const std::int32_t* x, std::int32_t* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] = std::int32_t(std::uint32_t(y[i]) + std::uint32_t(a) * std::uint32_t(x[i]));
}
template <Target Tg> std::int64_t sum_i32(const std::int32_t* x, std::size_t n) { std::int64_t s = 0; for (std::size_t i = 0; i < n; ++i) s += x[i]; return s; }
// Integer min / max / first index of them (n == 0: max_i32 = INT32_MIN, min_i32 = INT32_MAX, arg* = 0).
template <Target Tg> std::int32_t max_i32(const std::int32_t* x, std::size_t n) { std::int32_t m = std::numeric_limits<std::int32_t>::min(); for (std::size_t i = 0; i < n; ++i) m = x[i] > m ? x[i] : m; return m; }
template <Target Tg> std::int32_t min_i32(const std::int32_t* x, std::size_t n) { std::int32_t m = std::numeric_limits<std::int32_t>::max(); for (std::size_t i = 0; i < n; ++i) m = x[i] < m ? x[i] : m; return m; }
template <Target Tg> std::size_t count_eq_i32(const std::int32_t* x, std::size_t n, std::int32_t v) { std::size_t c = 0; for (std::size_t i = 0; i < n; ++i) c += x[i] == v; return c; }
template <Target Tg> std::size_t find_i32(const std::int32_t* x, std::size_t n, std::int32_t v) {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {   // a block test the compiler vectorises, then a short scalar scan
        std::uint32_t any = 0; for (int j = 0; j < 16; ++j) any |= std::uint32_t(x[i + j] == v);
        if (any) break;
    }
    for (; i < n; ++i) if (x[i] == v) return i;
    return n;
}

// ---- bitsets (arrays of 64-bit words; bit i of the set is bit i % 64 of word i / 64) ----
template <Target Tg> std::uint64_t popcount(const std::uint64_t* w, std::size_t n) { std::uint64_t c = 0; for (std::size_t i = 0; i < n; ++i) c += std::uint64_t(bits::popcount64<Tg>(w[i])); return c; }
template <Target Tg> std::uint64_t popcount_and(const std::uint64_t* a, const std::uint64_t* b, std::size_t n) { std::uint64_t c = 0; for (std::size_t i = 0; i < n; ++i) c += std::uint64_t(bits::popcount64<Tg>(a[i] & b[i])); return c; }
enum class bitop : int { and_ = 0, or_ = 1, xor_ = 2, andnot = 3 };   // andnot: a & ~b
template <Target Tg> void bitset_op(bitop op, const std::uint64_t* a, const std::uint64_t* b, std::uint64_t* out, std::size_t n) {
    switch (op) {
    case bitop::and_: for (std::size_t i = 0; i < n; ++i) out[i] = a[i] & b[i]; break;
    case bitop::or_:  for (std::size_t i = 0; i < n; ++i) out[i] = a[i] | b[i]; break;
    case bitop::xor_: for (std::size_t i = 0; i < n; ++i) out[i] = a[i] ^ b[i]; break;
    default:          for (std::size_t i = 0; i < n; ++i) out[i] = a[i] & ~b[i]; break;
    }
}
// Index of the first set bit at or after `from` in a set of n words; n * 64 when there is none.
template <Target Tg> std::size_t bitset_find_first(const std::uint64_t* w, std::size_t n, std::size_t from = 0) {
    std::size_t i = from / 64;
    if (i >= n) return n * 64;
    std::uint64_t cur = w[i] & (~std::uint64_t(0) << (from % 64));
    for (;;) {
        if (cur) return i * 64 + std::size_t(bits::ctz64<Tg>(cur));
        if (++i >= n) return n * 64;
        cur = w[i];
    }
}

} // namespace forge::simd::algo
