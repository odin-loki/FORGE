#pragma once
// Tier A sorting: sorting networks for small fixed N, a quicksort / network hybrid with a vectorised
// partition, selection, partial sort and top-k.
//
//   sort_network<N>(a)        fixed N: optimal-size networks for N <= 8, Batcher's merge-exchange above
//   sort_small(a, n)          n <= 32, a data-independent sequence of compare-exchanges
//   sort_i32 / sort_u32       ascending
//   sort_f32                  ascending in the IEEE total order: -NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN
//   partial_sort_f32(a, n, k) the k smallest, sorted, at the front (same order); the rest in unspecified order
//   topk_f32(x, n, k, idx, val) indices (and values) of the k largest, largest first, ties by lower index;
//                             NaN ranks below every number
//
// The sorts work on 32-bit keys (floats are mapped to integers that order the same way and mapped back, so
// zeros keep their signs and NaNs their payloads). The partition step is explicit AVX-512 (compress-store) or
// AVX2 (permute by a 256-entry table); other targets use a branch-free scalar loop. Sub-arrays of at most 32
// keys are finished by the network, whose min/max pairs the compiler vectorises. Recursion depth is bounded
// (introsort: a heap sort takes over on adversarial input), so the worst case is O(n log n).
// sort_*, partial_sort_f32 and topk_f32 allocate O(n) scratch keys with new[].
#include <forge/simd/algo_num.hpp>
#include <array>
#include <cstring>
#include <memory>

namespace forge::simd::algo {

namespace sortdetail {
template <Target Tg, class T> inline void cswap(T& a, T& b) { T lo = b < a ? b : a, hi = b < a ? a : b; a = lo; b = hi; }
// Batcher's merge-exchange (Knuth 5.2.2 M): a sorting network for any n.
template <Target Tg, class T> inline void merge_exchange(T* a, std::size_t n) {
    for (std::size_t p = 1; p < n; p += p)
        for (std::size_t k = p; k >= 1; k /= 2)
            for (std::size_t j = k % p; j + k < n; j += 2 * k)
                for (std::size_t i = 0; i < k && i + j + k < n; ++i)
                    if ((i + j) / (2 * p) == (i + j + k) / (2 * p)) cswap<Tg, T>(a[i + j], a[i + j + k]);
}
#define FORGE_CS(i, j) cswap<Tg, T>(a[i], a[j]);
template <Target Tg, class T> inline void net2(T* a) { FORGE_CS(0, 1) }
template <Target Tg, class T> inline void net3(T* a) { FORGE_CS(0, 2) FORGE_CS(0, 1) FORGE_CS(1, 2) }
template <Target Tg, class T> inline void net4(T* a) { FORGE_CS(0, 2) FORGE_CS(1, 3) FORGE_CS(0, 1) FORGE_CS(2, 3) FORGE_CS(1, 2) }
template <Target Tg, class T> inline void net5(T* a) { FORGE_CS(0, 3) FORGE_CS(1, 4) FORGE_CS(0, 2) FORGE_CS(1, 3) FORGE_CS(0, 1) FORGE_CS(2, 4) FORGE_CS(1, 2) FORGE_CS(3, 4) FORGE_CS(2, 3) }
template <Target Tg, class T> inline void net6(T* a) {
    FORGE_CS(0, 5) FORGE_CS(1, 3) FORGE_CS(2, 4) FORGE_CS(1, 2) FORGE_CS(3, 4) FORGE_CS(0, 3) FORGE_CS(2, 5)
    FORGE_CS(0, 1) FORGE_CS(2, 3) FORGE_CS(4, 5) FORGE_CS(1, 2) FORGE_CS(3, 4)
}
template <Target Tg, class T> inline void net7(T* a) {
    FORGE_CS(0, 6) FORGE_CS(2, 3) FORGE_CS(4, 5) FORGE_CS(0, 2) FORGE_CS(1, 4) FORGE_CS(3, 6) FORGE_CS(0, 1) FORGE_CS(2, 5) FORGE_CS(3, 4)
    FORGE_CS(1, 2) FORGE_CS(4, 6) FORGE_CS(2, 3) FORGE_CS(4, 5) FORGE_CS(1, 2) FORGE_CS(3, 4) FORGE_CS(5, 6)
}
template <Target Tg, class T> inline void net8(T* a) {
    FORGE_CS(0, 2) FORGE_CS(1, 3) FORGE_CS(4, 6) FORGE_CS(5, 7) FORGE_CS(0, 4) FORGE_CS(1, 5) FORGE_CS(2, 6) FORGE_CS(3, 7)
    FORGE_CS(0, 1) FORGE_CS(2, 3) FORGE_CS(4, 5) FORGE_CS(6, 7) FORGE_CS(2, 4) FORGE_CS(3, 5) FORGE_CS(1, 4) FORGE_CS(3, 6)
    FORGE_CS(1, 2) FORGE_CS(3, 4) FORGE_CS(5, 6)
}
#undef FORGE_CS

template <Target Tg> struct perm_lut {
    // row m: the lanes whose bit is set in m, in order, then the others
    static constexpr std::array<std::array<std::uint32_t, 8>, 256> make() {
        std::array<std::array<std::uint32_t, 8>, 256> t{};
        for (unsigned m = 0; m < 256; ++m) {
            unsigned k = 0;
            for (unsigned b = 0; b < 8; ++b) if (m >> b & 1) t[m][k++] = b;
            for (unsigned b = 0; b < 8; ++b) if (!(m >> b & 1)) t[m][k++] = b;
        }
        return t;
    }
    static constexpr std::array<std::array<std::uint32_t, 8>, 256> table = make();
};

// Moves the keys < pivot to the front of a (keeping their order) and the rest after them; returns how many
// are < pivot. scratch must hold n + 16 keys.
template <Target Tg> inline std::size_t partition(std::int32_t* a, std::size_t n, std::int32_t pivot, std::int32_t* scratch) {
    constexpr int K = impl::kind<Tg>();
    std::size_t i = 0, l = 0, r = 0;
    if constexpr (K == 2) {
#if defined(__AVX512F__)
        const __m512i vp = _mm512_set1_epi32(pivot);
        for (; i + 16 <= n; i += 16) {
            __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
            __mmask16 k = _mm512_cmplt_epi32_mask(v, vp);
            _mm512_mask_compressstoreu_epi32(a + l, k, v);
            _mm512_mask_compressstoreu_epi32(scratch + r, __mmask16(~k), v);
            std::size_t c = std::size_t(bits::popcount64<Tg>(k)); l += c; r += 16 - c;
        }
#endif
    } else if constexpr (K == 1) {
#if defined(__AVX2__)
        const __m256i vp = _mm256_set1_epi32(pivot);
        for (; i + 8 <= n; i += 8) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            unsigned k = unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(vp, v))));
            __m256i left = _mm256_permutevar8x32_epi32(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(perm_lut<Tg>::table[k].data())));
            __m256i right = _mm256_permutevar8x32_epi32(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(perm_lut<Tg>::table[k ^ 0xffu].data())));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + l), left);        // l <= i: only overwrites keys already loaded
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(scratch + r), right);
            std::size_t c = std::size_t(bits::popcount64<Tg>(k)); l += c; r += 8 - c;
        }
#endif
    }
    for (; i < n; ++i) { std::int32_t v = a[i]; bool lt = v < pivot; a[l] = v; scratch[r] = v; l += lt; r += !lt; }
    std::memcpy(a + l, scratch, r * sizeof(std::int32_t));
    return l;
}
template <Target Tg> inline std::int32_t med3(std::int32_t a, std::int32_t b, std::int32_t c) {
    return a < b ? (b < c ? b : (a < c ? c : a)) : (a < c ? a : (b < c ? c : b));
}
template <Target Tg> inline std::int32_t pick_pivot(const std::int32_t* a, std::size_t n) {
    if (n < 128) return med3<Tg>(a[0], a[n / 2], a[n - 1]);
    const std::size_t s = n / 8;
    return med3<Tg>(med3<Tg>(a[0], a[s], a[2 * s]), med3<Tg>(a[n / 2 - s], a[n / 2], a[n / 2 + s]), med3<Tg>(a[n - 1 - 2 * s], a[n - 1 - s], a[n - 1]));
}
template <Target Tg> inline void small_i32(std::int32_t* a, std::size_t n) {
    switch (n) {
    case 0: case 1: return;
    case 2: net2<Tg, std::int32_t>(a); return; case 3: net3<Tg, std::int32_t>(a); return; case 4: net4<Tg, std::int32_t>(a); return;
    case 5: net5<Tg, std::int32_t>(a); return; case 6: net6<Tg, std::int32_t>(a); return; case 7: net7<Tg, std::int32_t>(a); return;
    case 8: net8<Tg, std::int32_t>(a); return;
    default: merge_exchange<Tg, std::int32_t>(a, n);
    }
}
template <Target Tg> void sort_keys(std::int32_t* a, std::size_t n, std::int32_t* scratch, int depth) {
    while (n > 32) {
        if (depth-- <= 0) {
            auto less = [](std::int32_t x, std::int32_t y) { return x < y; };
            std::make_heap(a, a + n, less); std::sort_heap(a, a + n, less);
            return;
        }
        const std::int32_t pivot = pick_pivot<Tg>(a, n);
        std::size_t l = partition<Tg>(a, n, pivot, scratch);
        if (l == 0) {   // the pivot is the minimum: peel off everything equal to it
            if (pivot == std::numeric_limits<std::int32_t>::max()) return;
            l = partition<Tg>(a, n, pivot + 1, scratch);
            a += l; n -= l;
            continue;
        }
        if (l < n - l) { sort_keys<Tg>(a, l, scratch, depth); a += l; n -= l; }
        else { sort_keys<Tg>(a + l, n - l, scratch, depth); n = l; }
    }
    small_i32<Tg>(a, n);
}
template <Target Tg> inline int depth_limit(std::size_t n) { int d = 8; while (n > 1) { n >>= 1; d += 2; } return d; }
// Rearranges a so that a[k] is the key a full sort would put there, with nothing larger before it and
// nothing smaller after it.
template <Target Tg> void select_keys(std::int32_t* a, std::size_t n, std::size_t k, std::int32_t* scratch) {
    std::size_t lo = 0, hi = n;
    for (int budget = depth_limit<Tg>(n) * 2; hi - lo > 32; --budget) {
        if (budget <= 0) { sort_keys<Tg>(a + lo, hi - lo, scratch, depth_limit<Tg>(hi - lo)); return; }
        const std::int32_t pivot = pick_pivot<Tg>(a + lo, hi - lo);
        std::size_t l = partition<Tg>(a + lo, hi - lo, pivot, scratch);
        if (l == 0) {
            if (pivot == std::numeric_limits<std::int32_t>::max()) return;
            l = partition<Tg>(a + lo, hi - lo, pivot + 1, scratch);
            if (k < lo + l) return;   // k falls among the keys equal to the pivot
            lo += l;
            continue;
        }
        if (k < lo + l) hi = lo + l; else lo += l;
    }
    small_i32<Tg>(a + lo, hi - lo);
}
// float <-> key with the same order as the IEEE total order
template <Target Tg> inline std::int32_t f32_key(float f) { std::int32_t b; std::memcpy(&b, &f, 4); return b < 0 ? b ^ 0x7fffffff : b; }
template <Target Tg> inline float key_f32(std::int32_t k) { std::int32_t b = k < 0 ? k ^ 0x7fffffff : k; float f; std::memcpy(&f, &b, 4); return f; }
} // namespace sortdetail

// Sorting network for exactly N elements of any type with operator< (a strict weak order; for floats: no NaN).
template <std::size_t N, Target Tg, class T> void sort_network(T* a) {
    if constexpr (N < 2) (void)a;
    else if constexpr (N == 2) sortdetail::net2<Tg, T>(a);
    else if constexpr (N == 3) sortdetail::net3<Tg, T>(a);
    else if constexpr (N == 4) sortdetail::net4<Tg, T>(a);
    else if constexpr (N == 5) sortdetail::net5<Tg, T>(a);
    else if constexpr (N == 6) sortdetail::net6<Tg, T>(a);
    else if constexpr (N == 7) sortdetail::net7<Tg, T>(a);
    else if constexpr (N == 8) sortdetail::net8<Tg, T>(a);
    else sortdetail::merge_exchange<Tg, T>(a, N);
}
// n elements by a network chosen at run time (meant for n <= 32; correct for any n, O(n log^2 n) exchanges).
template <Target Tg, class T> void sort_small(T* a, std::size_t n) {
    switch (n) {
    case 0: case 1: return;
    case 2: sortdetail::net2<Tg, T>(a); return; case 3: sortdetail::net3<Tg, T>(a); return; case 4: sortdetail::net4<Tg, T>(a); return;
    case 5: sortdetail::net5<Tg, T>(a); return; case 6: sortdetail::net6<Tg, T>(a); return; case 7: sortdetail::net7<Tg, T>(a); return;
    case 8: sortdetail::net8<Tg, T>(a); return;
    default: sortdetail::merge_exchange<Tg, T>(a, n);
    }
}

template <Target Tg> void sort_i32(std::int32_t* a, std::size_t n) {
    if (n <= 32) { sortdetail::small_i32<Tg>(a, n); return; }
    std::unique_ptr<std::int32_t[]> scratch(new std::int32_t[n + 16]);
    sortdetail::sort_keys<Tg>(a, n, scratch.get(), sortdetail::depth_limit<Tg>(n));
}
template <Target Tg> void sort_u32(std::uint32_t* a, std::size_t n) {
    if (n < 2) return;
    std::unique_ptr<std::int32_t[]> buf(new std::int32_t[2 * n + 16]);
    std::int32_t* k = buf.get();
    for (std::size_t i = 0; i < n; ++i) k[i] = std::int32_t(a[i] ^ 0x80000000u);
    sortdetail::sort_keys<Tg>(k, n, k + n, sortdetail::depth_limit<Tg>(n));
    for (std::size_t i = 0; i < n; ++i) a[i] = std::uint32_t(k[i]) ^ 0x80000000u;
}
template <Target Tg> void sort_f32(float* a, std::size_t n) {
    if (n < 2) return;
    std::unique_ptr<std::int32_t[]> buf(new std::int32_t[2 * n + 16]);
    std::int32_t* k = buf.get();
    for (std::size_t i = 0; i < n; ++i) k[i] = sortdetail::f32_key<Tg>(a[i]);
    sortdetail::sort_keys<Tg>(k, n, k + n, sortdetail::depth_limit<Tg>(n));
    for (std::size_t i = 0; i < n; ++i) a[i] = sortdetail::key_f32<Tg>(k[i]);
}
template <Target Tg> void partial_sort_f32(float* a, std::size_t n, std::size_t k) {
    if (k >= n) { sort_f32<Tg>(a, n); return; }
    if (k == 0) return;
    std::unique_ptr<std::int32_t[]> buf(new std::int32_t[2 * n + 16]);
    std::int32_t* key = buf.get();
    for (std::size_t i = 0; i < n; ++i) key[i] = sortdetail::f32_key<Tg>(a[i]);
    sortdetail::select_keys<Tg>(key, n, k - 1, key + n);
    sortdetail::sort_keys<Tg>(key, k, key + n, sortdetail::depth_limit<Tg>(k));
    for (std::size_t i = 0; i < n; ++i) a[i] = sortdetail::key_f32<Tg>(key[i]);
}
// k-th smallest (0-based) of x in the IEEE total order; x is not modified. k < n.
template <Target Tg> float select_f32(const float* x, std::size_t n, std::size_t k) {
    std::unique_ptr<std::int32_t[]> buf(new std::int32_t[2 * n + 16]);
    std::int32_t* key = buf.get();
    for (std::size_t i = 0; i < n; ++i) key[i] = sortdetail::f32_key<Tg>(x[i]);
    sortdetail::select_keys<Tg>(key, n, k, key + n);
    return sortdetail::key_f32<Tg>(key[k]);
}

// Writes the indices of the min(k, n) largest elements of x to idx, largest first (equal values: lower index
// first; -0 and +0 are distinct, +0 larger; NaN ranks below everything), and their values to val unless val
// is null. Returns the count written. n must be below 2^32.
template <Target Tg> std::size_t topk_f32(const float* x, std::size_t n, std::size_t k, std::uint32_t* idx, float* val) {
    k = std::min(k, n);
    if (k == 0) return 0;
    constexpr std::int32_t lowest = std::numeric_limits<std::int32_t>::min();
    auto key_of = [](float f) { return f != f ? lowest : sortdetail::f32_key<Tg>(f); };
    std::unique_ptr<std::int32_t[]> buf(new std::int32_t[2 * n + 16]);
    std::int32_t* key = buf.get();
    for (std::size_t i = 0; i < n; ++i) key[i] = key_of(x[i]);
    sortdetail::select_keys<Tg>(key, n, n - k, key + n);
    const std::int32_t t = key[n - k];   // the k-th largest key
    std::unique_ptr<std::uint64_t[]> packed(new std::uint64_t[k + 1]);
    std::uint64_t* p = packed.get();
    auto pack = [](std::int32_t ky, std::size_t i) { return (std::uint64_t(std::uint32_t(ky) ^ 0x80000000u) << 32) | std::uint64_t(0xffffffffu - std::uint32_t(i)); };
    std::size_t c = 0;
    for (std::size_t i = 0; i < n; ++i) { std::int32_t ky = key_of(x[i]); p[c] = pack(ky, i); c += ky > t; }   // fewer than k keys exceed t, so c <= k - 1 and p[c] stays in range
    for (std::size_t i = 0; i < n && c < k; ++i) if (key_of(x[i]) == t) p[c++] = pack(t, i);
    std::sort(p, p + k, [](std::uint64_t a, std::uint64_t b) { return a > b; });
    for (std::size_t j = 0; j < k; ++j) {
        std::uint32_t i = 0xffffffffu - std::uint32_t(p[j]);
        idx[j] = i; if (val) val[j] = x[i];
    }
    return k;
}

} // namespace forge::simd::algo
