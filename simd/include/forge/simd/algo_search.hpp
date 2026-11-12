#pragma once
// Tier A byte search: first byte (memchr), byte count, substring (memmem), first mismatch.
// Explicit AVX-512BW (64 bytes per step) and AVX2 (32 bytes per step) paths; other targets read 8 bytes at a
// time with integer tricks (SWAR). No function reads outside [p, p + n).
#include <forge/simd/algo_num.hpp>
#include <cstring>

namespace forge::simd::algo {

namespace searchdetail {
template <Target Tg> inline std::uint64_t load64(const std::uint8_t* p) { std::uint64_t v; std::memcpy(&v, p, 8); return v; }
// 0x80 in every byte of v that is zero (exact, no false positives from borrows)
template <Target Tg> inline std::uint64_t zero_bytes(std::uint64_t v) {
    const std::uint64_t lo7 = 0x7f7f7f7f7f7f7f7full;
    return ~(((v & lo7) + lo7) | v | lo7);
}
template <Target Tg> inline bool little_endian() { const std::uint16_t x = 1; std::uint8_t b; std::memcpy(&b, &x, 1); return b == 1; }
} // namespace searchdetail

// Index of the first byte equal to c; n when there is none.
template <Target Tg> std::size_t find_byte(const std::uint8_t* p, std::size_t n, std::uint8_t c) {
    constexpr int K = impl::kind<Tg>();
    std::size_t i = 0;
    if constexpr (K == 2) {
#if defined(__AVX512BW__)
        const __m512i vc = _mm512_set1_epi8(char(c));
        for (; i + 64 <= n; i += 64)
            if (std::uint64_t m = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(reinterpret_cast<const void*>(p + i)), vc)) return i + std::size_t(bits::ctz64<Tg>(m));
        if (i < n) {
            const __mmask64 k = ~std::uint64_t(0) >> (64 - (n - i));
            if (std::uint64_t m = _mm512_mask_cmpeq_epi8_mask(k, _mm512_maskz_loadu_epi8(k, p + i), vc)) return i + std::size_t(bits::ctz64<Tg>(m));
        }
        return n;
#endif
    } else if constexpr (K == 1) {
#if defined(__AVX2__)
        const __m256i vc = _mm256_set1_epi8(char(c));
        for (; i + 32 <= n; i += 32)
            if (std::uint32_t m = std::uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i)), vc)))) return i + std::size_t(bits::ctz64<Tg>(m));
#endif
    }
    if (searchdetail::little_endian<Tg>()) {
        const std::uint64_t rep = 0x0101010101010101ull * c;
        for (; i + 8 <= n; i += 8)
            if (std::uint64_t z = searchdetail::zero_bytes<Tg>(searchdetail::load64<Tg>(p + i) ^ rep)) return i + std::size_t(bits::ctz64<Tg>(z)) / 8;
    }
    for (; i < n; ++i) if (p[i] == c) return i;
    return n;
}

template <Target Tg> std::size_t count_byte(const std::uint8_t* p, std::size_t n, std::uint8_t c) {
    constexpr int K = impl::kind<Tg>();
    std::size_t i = 0, cnt = 0;
    if constexpr (K == 2) {
#if defined(__AVX512BW__)
        const __m512i vc = _mm512_set1_epi8(char(c));
        for (; i + 64 <= n; i += 64) cnt += std::size_t(bits::popcount64<Tg>(_mm512_cmpeq_epi8_mask(_mm512_loadu_si512(reinterpret_cast<const void*>(p + i)), vc)));
#endif
    } else if constexpr (K == 1) {
#if defined(__AVX2__)
        const __m256i vc = _mm256_set1_epi8(char(c));
        for (; i + 32 <= n; i += 32) cnt += std::size_t(bits::popcount64<Tg>(std::uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i)), vc)))));
#endif
    }
    for (; i < n; ++i) cnt += p[i] == c;
    return cnt;
}

// Index of the first position where a and b differ; n when the ranges are equal.
template <Target Tg> std::size_t mismatch(const std::uint8_t* a, const std::uint8_t* b, std::size_t n) {
    constexpr int K = impl::kind<Tg>();
    std::size_t i = 0;
    if constexpr (K == 2) {
#if defined(__AVX512BW__)
        for (; i + 64 <= n; i += 64)
            if (std::uint64_t m = _mm512_cmpneq_epi8_mask(_mm512_loadu_si512(reinterpret_cast<const void*>(a + i)), _mm512_loadu_si512(reinterpret_cast<const void*>(b + i)))) return i + std::size_t(bits::ctz64<Tg>(m));
#endif
    } else if constexpr (K == 1) {
#if defined(__AVX2__)
        for (; i + 32 <= n; i += 32) {
            std::uint32_t eq = std::uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)))));
            if (eq != 0xffffffffu) return i + std::size_t(bits::ctz64<Tg>(std::uint32_t(~eq)));
        }
#endif
    }
    if (searchdetail::little_endian<Tg>())
        for (; i + 8 <= n; i += 8)
            if (std::uint64_t d = searchdetail::load64<Tg>(a + i) ^ searchdetail::load64<Tg>(b + i)) return i + std::size_t(bits::ctz64<Tg>(d)) / 8;
    for (; i < n; ++i) if (a[i] != b[i]) return i;
    return n;
}

// Index of the first occurrence of needle[0, m) in hay[0, n); n when there is none; 0 for an empty needle.
// Candidates are the positions where both the first and the last needle byte match (a vector compare of two
// windows), each confirmed with memcmp. The worst case is O(n * m) (e.g. a needle of one repeated byte in a
// haystack of the same byte with one mismatch in the middle of every window).
template <Target Tg> std::size_t find_bytes(const std::uint8_t* hay, std::size_t n, const std::uint8_t* needle, std::size_t m) {
    if (m == 0) return 0;
    if (m > n) return n;
    if (m == 1) return find_byte<Tg>(hay, n, needle[0]);
    constexpr int K = impl::kind<Tg>();
    const std::size_t last = n - m;   // highest valid start
    const std::uint8_t c0 = needle[0], c1 = needle[m - 1];
    std::size_t i = 0;
    auto confirm = [&](std::size_t at) { return std::memcmp(hay + at + 1, needle + 1, m - 2) == 0; };
    if constexpr (K == 2) {
#if defined(__AVX512BW__)
        const __m512i v0 = _mm512_set1_epi8(char(c0)), v1 = _mm512_set1_epi8(char(c1));
        for (; i + 64 <= last + 1; i += 64) {
            std::uint64_t k = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(reinterpret_cast<const void*>(hay + i)), v0)
                            & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(reinterpret_cast<const void*>(hay + i + m - 1)), v1);
            for (; k; k &= k - 1) { std::size_t at = i + std::size_t(bits::ctz64<Tg>(k)); if (confirm(at)) return at; }
        }
#endif
    } else if constexpr (K == 1) {
#if defined(__AVX2__)
        const __m256i v0 = _mm256_set1_epi8(char(c0)), v1 = _mm256_set1_epi8(char(c1));
        for (; i + 32 <= last + 1; i += 32) {
            std::uint32_t k = std::uint32_t(_mm256_movemask_epi8(_mm256_and_si256(
                _mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(hay + i)), v0),
                _mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(hay + i + m - 1)), v1))));
            for (; k; k &= k - 1) { std::size_t at = i + std::size_t(bits::ctz64<Tg>(k)); if (confirm(at)) return at; }
        }
#endif
    }
    while (i <= last) {   // remaining starts: jump to the next first-byte match
        std::size_t j = find_byte<Tg>(hay + i, last + 1 - i, c0);
        if (j == last + 1 - i) return n;
        i += j;
        if (hay[i + m - 1] == c1 && confirm(i)) return i;
        ++i;
    }
    return n;
}

} // namespace forge::simd::algo
