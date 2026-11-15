#pragma once
// 2:4 sparse matmul_t on the CPU, straight from the s24_f16 / s24_q8 / s24_q4 blocks (graph/codecs.hpp).
//
// A 32-weight block keeps 16 values and 4 bytes of 2-bit positions. Instead of decoding the row to 32 dense floats
// (half of them zero) and running a dense dot, each block's 16 kept values are multiplied with the 16 activations
// they sit over, gathered by position: half the multiply-adds, and the weights are read at their stored size
// (1.125 / 0.6875 / 0.4375 bytes per weight). Kept value j of a block sits in group j / 2 at position
//   ((pos_word >> 2j) & 3), i.e. activation 4 * (j / 2) + that,
// where pos_word is the block's 4 position bytes read as a little-endian u32 (group g = bits 4g..4g+3).
//
// Kernels: AVX-512 (one two-source permute gathers the 16 activations of a block), AVX2 (two 8-lane permutes and a
// blend per 8), and a portable scalar loop for every other CPU. They are compiled here with target attributes and
// picked at run time, so this header needs no ISA flags. The result is the dense product's up to summation order
// (pruned positions contribute exactly 0 there), within matmul_t's oracle tolerance; tests/test_pack_sparse.cpp
// checks it against decode + dense dot. FORGE_S24=0 turns the path off (generic decode + dot), FORGE_S24_ISA=
// scalar|avx2|avx512 caps the kernel tier.
#include <forge/core/dtypes.hpp>
#include <forge/graph/codecs.hpp>
#include <forge/simd/dispatch.hpp>
#include <forge/task/engine.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define FORGE_S24_X86 1
#define FORGE_S24_T512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma,f16c")))
#define FORGE_S24_T256 __attribute__((target("avx2,fma,f16c")))
#else
#define FORGE_S24_X86 0
#endif

namespace forge::graph::s24 {

// 16, 8 or 4 for the s24 codecs (value bits), 0 for every other codec
inline int bits_of(const codecs::codec* c) {
    if (!c) return 0;
    return c->name == "s24_f16" ? 16 : c->name == "s24_q8" ? 8 : c->name == "s24_q4" ? 4 : 0;
}
inline constexpr std::size_t block_bytes(int bits) { return bits == 16 ? 36 : bits == 8 ? 22 : 14; }
inline bool enabled() {
    static const bool on = [] { const char* e = std::getenv("FORGE_S24"); return !(e && *e == '0'); }();
    return on;
}
enum class isa : int { scalar = 0, avx2 = 1, avx512 = 2 };
inline isa best_isa() {
    static const isa b = [] {
        isa r = isa::scalar;
#if FORGE_S24_X86
        const auto& f = simd::detect();
        if (f.avx2 && f.fma && f.f16c && f.os_avx) r = isa::avx2;
        if (r == isa::avx2 && f.avx512f && f.avx512bw && f.avx512vl && f.avx512dq && f.os_avx512) r = isa::avx512;
#endif
        if (const char* e = std::getenv("FORGE_S24_ISA"); e && *e) {
            std::string_view v(e);
            isa cap = v == "scalar" ? isa::scalar : v == "avx2" ? isa::avx2 : isa::avx512;
            if (int(cap) < int(r)) r = cap;
        }
        return r;
    }();
    return b;
}

// Accumulation (every tier): per (weight row, activation row) two chains of partial sums, block i of the row going
// to chain i % 2, reduced once at the end, plus a partial last block added in scalar. The chains are kept in memory
// between calls, so the k dimension can be cut into chunks (the activations of a chunk stay in L1 while a tile of
// weight rows passes over them) without changing a single rounding: a row's result does not depend on how many
// activation rows ran with it, how the work was split, or which thread ran it.
namespace detail {
inline float h2f(const std::uint8_t* p) { std::uint16_t b; std::memcpy(&b, p, 2); return float(f16::from_bits(b)); }
inline std::uint32_t ld32(const std::uint8_t* p) { std::uint32_t v; std::memcpy(&v, p, 4); return v; }

// One block, portable: the 16 kept values (as the codec decodes them) and their activation offsets 0..31.
template <int Bits> inline void unpack_block(const std::uint8_t* b, float* kv, std::uint8_t* at) {
    const std::uint8_t* p = b;
    if constexpr (Bits == 16) { for (int i = 0; i < 16; ++i) kv[i] = h2f(p + 2 * i); p += 32; }
    else {
        const float d = h2f(p); p += 2;
        if constexpr (Bits == 8) { for (int i = 0; i < 16; ++i) kv[i] = d * float(std::int8_t(p[i])); p += 16; }
        else { for (int i = 0; i < 8; ++i) { kv[2 * i] = d * float(int(p[i] & 15) - 8); kv[2 * i + 1] = d * float(int(p[i] >> 4) - 8); } p += 8; }
    }
    const std::uint32_t w = ld32(p);
    for (int j = 0; j < 16; ++j) at[j] = std::uint8_t(4 * (j / 2) + ((w >> (2 * j)) & 3));
}
// the partial last block (k % 32 weights): positions past k are dropped, as the dense decode drops them
template <int Bits> inline float tail_dot(const std::uint8_t* b, const float* x, std::size_t len) {
    float kv[16]; std::uint8_t at[16]; unpack_block<Bits>(b, kv, at);
    float s = 0;
    for (int j = 0; j < 16; ++j) if (at[j] < len) s += kv[j] * x[at[j]];
    return s;
}
// ---- scalar: 4 floats of state per (weight row, activation row): chain i % 2 of block i, two lanes each ----
inline constexpr std::size_t acc_scalar = 4;
template <int Bits> inline void step_scalar(const std::uint8_t* b, std::size_t nb, const float* x, std::size_t ldx, std::size_t nr, float* acc) {
    constexpr std::size_t BB = block_bytes(Bits);
    for (std::size_t i = 0; i < nb; ++i, b += BB) {
        float kv[16]; std::uint8_t at[16]; unpack_block<Bits>(b, kv, at);
        const std::size_t ch = 2 * (i & 1);
        for (std::size_t r = 0; r < nr; ++r) {
            const float* xb = x + r * ldx + 32 * i; float* a = acc + r * acc_scalar + ch;
            for (int j = 0; j < 16; j += 2) { a[0] += kv[j] * xb[at[j]]; a[1] += kv[j + 1] * xb[at[j + 1]]; }
        }
    }
}
inline float fin_scalar(const float* a) { return (a[0] + a[2]) + (a[1] + a[3]); }

#if FORGE_S24_X86
// The block scale in every lane, by F16C (exact, as f16::from_bits). Built from a broadcast of the 16-bit word rather
// than a scalar conversion: for that the compiler may insert the word into whatever register is free, the accumulator
// included, which chains every block on the previous one's multiply-add.
FORGE_S24_T512 inline __m512 scale512(const std::uint8_t* p) {
    std::uint16_t b; std::memcpy(&b, p, 2);
    return _mm512_cvtph_ps(_mm256_set1_epi16(short(b)));
}
FORGE_S24_T256 inline __m256 scale256(const std::uint8_t* p) {
    std::uint16_t b; std::memcpy(&b, p, 2);
    return _mm256_cvtph_ps(_mm_set1_epi16(short(b)));
}
// Weight bytes are prefetched this far ahead of the block being multiplied: matvec streams the matrix from DRAM, and
// the hardware prefetcher alone left a core at about half the bandwidth the integer q8_0 kernels reach.
inline constexpr std::size_t prefetch_bytes = 2048;
// ---- AVX-512: 32 floats of state per (weight row, activation row), two 16-lane chains ----
inline constexpr std::size_t acc512 = 32;
FORGE_S24_T512 inline __m512i idx512(const std::uint8_t* pos) {
    const __m512i sh = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i base = _mm512_setr_epi32(0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28);
    __m512i v = _mm512_srlv_epi32(_mm512_set1_epi32(int(ld32(pos))), sh);
    return _mm512_add_epi32(_mm512_and_si512(v, _mm512_set1_epi32(3)), base);
}
template <int Bits> FORGE_S24_T512 inline __m512 kv512(const std::uint8_t* b) {
    if constexpr (Bits == 16) return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b)));
    const __m512 d = scale512(b);
    if constexpr (Bits == 8)
        return _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 2)))));
    else {
        const __m512i sh = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28);
        __m512i w = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(int(ld32(b + 2)))), _mm256_set1_epi32(int(ld32(b + 6))), 1);
        __m512i q = _mm512_sub_epi32(_mm512_and_si512(_mm512_srlv_epi32(w, sh), _mm512_set1_epi32(15)), _mm512_set1_epi32(8));
        return _mm512_mul_ps(d, _mm512_cvtepi32_ps(q));
    }
}
FORGE_S24_T512 inline __m512 gather512(const float* xb, __m512i id) { return _mm512_permutex2var_ps(_mm512_loadu_ps(xb), id, _mm512_loadu_ps(xb + 16)); }
// block i of a chunk into the T chains a[]: decoded in registers (Buf false) or read from kv / at (Buf true)
template <int Bits, int T, bool Buf>
FORGE_S24_T512 inline void block512(const std::uint8_t* b, const float* kvb, const std::int32_t* atb, std::size_t i, const float* x, std::size_t ldx, __m512* a) {
    constexpr std::size_t BB = block_bytes(Bits);
    __m512 kv; __m512i id;
    if constexpr (Buf) { kv = _mm512_loadu_ps(kvb + 16 * i); id = _mm512_loadu_si512(atb + 16 * i); }
    else { _mm_prefetch(reinterpret_cast<const char*>(b + i * BB) + prefetch_bytes, _MM_HINT_T0); kv = kv512<Bits>(b + i * BB); id = idx512(b + i * BB + BB - 4); }
    for (int r = 0; r < T; ++r) a[r] = _mm512_fmadd_ps(kv, gather512(x + r * ldx + 32 * i, id), a[r]);
}
// T activation rows (T <= 8) over nb blocks
template <int Bits, int T, bool Buf>
FORGE_S24_T512 inline void step512(const std::uint8_t* b, const float* kvb, const std::int32_t* atb, std::size_t nb, const float* x, std::size_t ldx, float* acc) {
    __m512 a0[T], a1[T];
    for (int r = 0; r < T; ++r) { a0[r] = _mm512_loadu_ps(acc + r * acc512); a1[r] = _mm512_loadu_ps(acc + r * acc512 + 16); }
    std::size_t i = 0;
    for (; i + 2 <= nb; i += 2) { block512<Bits, T, Buf>(b, kvb, atb, i, x, ldx, a0); block512<Bits, T, Buf>(b, kvb, atb, i + 1, x, ldx, a1); }
    if (i < nb) block512<Bits, T, Buf>(b, kvb, atb, i, x, ldx, a0);
    for (int r = 0; r < T; ++r) { _mm512_storeu_ps(acc + r * acc512, a0[r]); _mm512_storeu_ps(acc + r * acc512 + 16, a1[r]); }
}
template <int Bits> FORGE_S24_T512 inline void unpack512(const std::uint8_t* b, std::size_t nb, float* kvb, std::int32_t* atb) {
    constexpr std::size_t BB = block_bytes(Bits);
    for (std::size_t i = 0; i < nb; ++i) { _mm512_storeu_ps(kvb + 16 * i, kv512<Bits>(b + i * BB)); _mm512_storeu_si512(atb + 16 * i, idx512(b + i * BB + BB - 4)); }
}
FORGE_S24_T512 inline float fin512(const float* a) { return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(a + 16))); }

// ---- AVX2: 16 floats of state per (weight row, activation row), one 8-lane chain per half block ----
inline constexpr std::size_t acc256 = 16;
FORGE_S24_T256 inline __m256i idx256(std::uint32_t w, int h) {   // kept values 8h..8h+7, offsets within activations [16h, 16h + 16)
    const __m256i sh = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i base = _mm256_setr_epi32(0, 0, 4, 4, 8, 8, 12, 12);
    __m256i v = _mm256_srlv_epi32(_mm256_set1_epi32(int(w >> (16 * h))), sh);
    return _mm256_add_epi32(_mm256_and_si256(v, _mm256_set1_epi32(3)), base);
}
template <int Bits> FORGE_S24_T256 inline __m256 kv256(const std::uint8_t* b, int h) {
    if constexpr (Bits == 16) return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 16 * h)));
    const __m256 d = scale256(b);
    if constexpr (Bits == 8)
        return _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + 2 + 8 * h)))));
    else {
        const __m256i sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
        __m256i q = _mm256_srlv_epi32(_mm256_set1_epi32(int(ld32(b + 2 + 4 * h))), sh);
        q = _mm256_sub_epi32(_mm256_and_si256(q, _mm256_set1_epi32(15)), _mm256_set1_epi32(8));
        return _mm256_mul_ps(d, _mm256_cvtepi32_ps(q));
    }
}
FORGE_S24_T256 inline __m256 gather256(const float* xb, __m256i id) {   // xb: 16 activations
    return _mm256_blend_ps(_mm256_permutevar8x32_ps(_mm256_loadu_ps(xb), id), _mm256_permutevar8x32_ps(_mm256_loadu_ps(xb + 8), id), 0xF0);
}
template <int Bits, int T, bool Buf>
FORGE_S24_T256 inline void step256(const std::uint8_t* b, const float* kvb, const std::int32_t* atb, std::size_t nb, const float* x, std::size_t ldx, float* acc) {
    constexpr std::size_t BB = block_bytes(Bits);
    __m256 a[T][2];
    for (int r = 0; r < T; ++r) { a[r][0] = _mm256_loadu_ps(acc + r * acc256); a[r][1] = _mm256_loadu_ps(acc + r * acc256 + 8); }
    for (std::size_t i = 0; i < nb; ++i) {
        if constexpr (!Buf) _mm_prefetch(reinterpret_cast<const char*>(b + i * BB) + prefetch_bytes, _MM_HINT_T0);
        const std::uint32_t w = Buf ? 0 : ld32(b + i * BB + BB - 4);
        for (int h = 0; h < 2; ++h) {
            __m256 kv; __m256i id;
            if constexpr (Buf) { kv = _mm256_loadu_ps(kvb + 16 * i + 8 * h); id = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(atb + 16 * i + 8 * h)); }
            else { kv = kv256<Bits>(b + i * BB, h); id = idx256(w, h); }
            for (int r = 0; r < T; ++r) a[r][h] = _mm256_fmadd_ps(kv, gather256(x + r * ldx + 32 * i + 16 * h, id), a[r][h]);
        }
    }
    for (int r = 0; r < T; ++r) { _mm256_storeu_ps(acc + r * acc256, a[r][0]); _mm256_storeu_ps(acc + r * acc256 + 8, a[r][1]); }
}
template <int Bits> FORGE_S24_T256 inline void unpack256(const std::uint8_t* b, std::size_t nb, float* kvb, std::int32_t* atb) {
    constexpr std::size_t BB = block_bytes(Bits);
    for (std::size_t i = 0; i < nb; ++i) {
        const std::uint32_t w = ld32(b + i * BB + BB - 4);
        for (int h = 0; h < 2; ++h) {
            _mm256_storeu_ps(kvb + 16 * i + 8 * h, kv256<Bits>(b + i * BB, h));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(atb + 16 * i + 8 * h), idx256(w, h));
        }
    }
}
FORGE_S24_T256 inline float fin256(const float* a) {
    __m256 v = _mm256_add_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(a + 8));
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s)); s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
// up to 8 activation rows: the step instantiation for that many
template <int Bits, bool Buf, bool Wide>
inline void step_n(std::size_t t, const std::uint8_t* b, const float* kvb, const std::int32_t* atb, std::size_t nb, const float* x, std::size_t ldx, float* acc) {
#define FORGE_S24_CASE(T) case T: if constexpr (Wide) step512<Bits, T, Buf>(b, kvb, atb, nb, x, ldx, acc); else step256<Bits, T, Buf>(b, kvb, atb, nb, x, ldx, acc); break;
    switch (t) { FORGE_S24_CASE(1) FORGE_S24_CASE(2) FORGE_S24_CASE(3) FORGE_S24_CASE(4) FORGE_S24_CASE(5) FORGE_S24_CASE(6) FORGE_S24_CASE(7) default: FORGE_S24_CASE(8) }
#undef FORGE_S24_CASE
}
#endif

// Weight rows [b, e) times activation rows [0, nr) (row i at x + i * k) -> y[i * ldy + j].
template <int Bits> inline void rows(isa t, const std::uint8_t* w, std::size_t rb, std::size_t b, std::size_t e, const float* x, std::size_t nr,
                                     std::size_t k, float* y, std::size_t ldy) {
    constexpr std::size_t BB = block_bytes(Bits), JT = 8;
    const std::size_t nbf = k / 32, tail = k % 32;
    const std::size_t A = t == isa::avx512 ? 32 : t == isa::avx2 ? 16 : acc_scalar;
    // blocks per chunk: even (chunks start on chain 0), the chunk of every activation row within ~1 MiB (L2). Timed:
    // smaller chunks (L1-sized) lose more to saving and reloading the chains than they gain in reuse.
    const std::size_t kcb = std::max<std::size_t>(2, ((std::size_t(1) << 20) / (128 * std::max<std::size_t>(nr, 1))) & ~std::size_t(1));
    thread_local std::vector<float> acc, kvb; thread_local std::vector<std::int32_t> atb;
    if (acc.size() < JT * nr * A) acc.resize(JT * nr * A);
    if (kvb.size() < 16 * kcb) { kvb.resize(16 * kcb); atb.resize(16 * kcb); }
    for (std::size_t j0 = b; j0 < e; j0 += JT) {
        const std::size_t jt = std::min(JT, e - j0);
        std::fill_n(acc.begin(), jt * nr * A, 0.f);
        for (std::size_t c0 = 0; c0 < nbf; c0 += kcb) {
            const std::size_t nb = std::min(kcb, nbf - c0);
            const float* xc = x + 32 * c0;
            for (std::size_t jj = 0; jj < jt; ++jj) {
                const std::uint8_t* row = w + (j0 + jj) * rb + c0 * BB;
                float* a = acc.data() + jj * nr * A;
#if FORGE_S24_X86
                if (t != isa::scalar) {
                    const bool wide = t == isa::avx512, buf = nr > 8;   // many rows: decode the chunk once, then 8 rows a pass
                    if (buf) { if (wide) unpack512<Bits>(row, nb, kvb.data(), atb.data()); else unpack256<Bits>(row, nb, kvb.data(), atb.data()); }
                    for (std::size_t g = 0; g < nr; g += 8) {
                        const std::size_t tn = std::min<std::size_t>(8, nr - g);
                        if (wide) { if (buf) step_n<Bits, true, true>(tn, row, kvb.data(), atb.data(), nb, xc + g * k, k, a + g * A);
                                    else step_n<Bits, false, true>(tn, row, nullptr, nullptr, nb, xc + g * k, k, a + g * A); }
                        else      { if (buf) step_n<Bits, true, false>(tn, row, kvb.data(), atb.data(), nb, xc + g * k, k, a + g * A);
                                    else step_n<Bits, false, false>(tn, row, nullptr, nullptr, nb, xc + g * k, k, a + g * A); }
                    }
                    continue;
                }
#endif
                step_scalar<Bits>(row, nb, xc, k, nr, a);
            }
        }
        for (std::size_t jj = 0; jj < jt; ++jj)
            for (std::size_t r = 0; r < nr; ++r) {
                const float* a = acc.data() + (jj * nr + r) * A;
                float v;
#if FORGE_S24_X86
                if (t == isa::avx512) v = fin512(a); else if (t == isa::avx2) v = fin256(a); else
#endif
                v = fin_scalar(a);
                if (tail) v += tail_dot<Bits>(w + (j0 + jj) * rb + nbf * BB, x + r * k + 32 * nbf, tail);
                y[r * ldy + j0 + jj] = v;
            }
    }
}
template <int Bits> inline void gemm_bits(isa t, const std::uint8_t* w, std::size_t rb, std::size_t n, const float* x, std::size_t m, std::size_t k, float* y) {
    auto body = [&](std::size_t b, std::size_t e) {
        for (std::size_t r0 = 0; r0 < m; r0 += 64)   // activation rows in panels of 64
            rows<Bits>(t, w, rb, b, e, x + r0 * k, std::min<std::size_t>(64, m - r0), k, y + r0 * n, n);
    };
    if (m <= 8) task::parallel_for(n, 16, 256, body);
    else task::parallel_for(n, 8, 64, body);
}
} // namespace detail

// y[m, n] = x[m, k] . W[n, k]^T for W stored as s24_f16 / s24_q8 / s24_q4 rows of rb = codec.bytes(k) bytes
inline void gemm_nt(int bits, const std::uint8_t* w, std::size_t rb, std::size_t n, const float* x, std::size_t m, std::size_t k, float* y,
                    isa t = best_isa()) {
    switch (bits) {
        case 16: detail::gemm_bits<16>(t, w, rb, n, x, m, k, y); break;
        case 8: detail::gemm_bits<8>(t, w, rb, n, x, m, k, y); break;
        default: detail::gemm_bits<4>(t, w, rb, n, x, m, k, y); break;
    }
}

} // namespace forge::graph::s24
