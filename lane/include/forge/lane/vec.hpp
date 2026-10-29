#pragma once
// Tier P: portable fixed-width vectors and masks over native registers.
// Native code paths are only instantiated in TUs compiled with the matching -m flags (see lane/src).
#include <forge/core/dtypes.hpp>
#include <forge/lane/caps.hpp>
#include <cmath>
#include <cstddef>
#include <cstring>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define FORGE_IFNEON(...) __VA_ARGS__
#else
#define FORGE_IFNEON(...)
#endif

namespace forge::lane {

template <class T, Target Tg> struct vec;
template <class T, Target Tg> struct mask;

// ---------------- scalar emulation (any target) ----------------
template <int N> struct femu { float v[N]; };
template <int N> struct iemu { i32 v[N]; };

namespace impl {
template <Target Tg> constexpr int kind() {   // 3 = neon, 2 = avx512, 1 = avx2, 0 = emulated
#if defined(__aarch64__) && defined(__ARM_NEON)
    if constexpr (Has<Tg, neon>) return 3;
#endif
#if defined(__AVX512F__)
    if constexpr (Has<Tg, avx512f>) return 2;
#endif
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr (Has<Tg, avx2> && Has<Tg, fma>) return 1;
#endif
    return 0;
}
} // namespace impl

template <Target Tg> struct mask<float, Tg> {
    static constexpr int K = impl::kind<Tg>();
    static constexpr int N = native_f32_lanes<Tg>() > 4 && K == 0 ? 4 : native_f32_lanes<Tg>();
#if defined(__AVX512F__)
    using native_t = std::conditional_t<K == 2, __mmask16,
#else
    using native_t = std::conditional_t<false, int,
#endif
#if defined(__AVX2__)
                     std::conditional_t<K == 1, __m256, u32>>;
#elif defined(__aarch64__) && defined(__ARM_NEON)
                     std::conditional_t<K == 3, uint32x4_t, u32>>;
#else
                     u32>;
#endif
    native_t m;
    static mask first_n(std::size_t n) {
        mask r;
        if constexpr (K == 2) {
#if defined(__AVX512F__)
            r.m = __mmask16(n >= 16 ? 0xffffu : ((1u << n) - 1u));
#endif
        } else if constexpr (K == 1) {
#if defined(__AVX2__)
            __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            r.m = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(int(n > 8 ? 8 : n)), idx));
#endif
        } else if constexpr (K == 3) {
            FORGE_IFNEON(const uint32_t idx[4] = {0, 1, 2, 3}; r.m = vcltq_u32(vld1q_u32(idx), vdupq_n_u32(uint32_t(n > 4 ? 4 : n)));)
        } else {
            r.m = n >= u32(N) ? ((1u << N) - 1u) : ((1u << n) - 1u);
        }
        return r;
    }
    static mask all() { return first_n(N); }
    unsigned bits() const {
        if constexpr (K == 2) return unsigned(m);
#if defined(__AVX2__)
        else if constexpr (K == 1) return unsigned(_mm256_movemask_ps(m));
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
        else if constexpr (K == 3) { const uint32_t w[4] = {1, 2, 4, 8}; return unsigned(vaddvq_u32(vandq_u32(m, vld1q_u32(w)))); }
#endif
        else return unsigned(m);
    }
    bool any() const { return bits() != 0; }
    bool all_set() const { return bits() == (N == 32 ? ~0u : ((1u << N) - 1u)); }
    bool none() const { return bits() == 0; }
    int count() const { return __builtin_popcount(bits()); }
    int first() const { auto b = bits(); return b ? __builtin_ctz(b) : -1; }
};

template <Target Tg> struct vec<float, Tg> {
    static constexpr int K = impl::kind<Tg>();
    static constexpr int N = mask<float, Tg>::N;
    using mask_t = mask<float, Tg>;
#if defined(__AVX512F__)
    using native_t = std::conditional_t<K == 2, __m512,
#else
    using native_t = std::conditional_t<false, int,
#endif
#if defined(__AVX2__)
                     std::conditional_t<K == 1, __m256, femu<N>>>;
#elif defined(__aarch64__) && defined(__ARM_NEON)
                     std::conditional_t<K == 3, float32x4_t, femu<N>>>;
#else
                     femu<N>>;
#endif
    native_t r;
    native_t native() const { return r; }
    static vec from_native(native_t n) { vec v; v.r = n; return v; }
    static constexpr int size() { return N; }
};

#define FORGE_VOP(name, a512, a256, aneon, scalar_expr)                                   \
    template <Target Tg> inline vec<float, Tg> name(vec<float, Tg> a, vec<float, Tg> b) { \
        vec<float, Tg> o;                                                                 \
        if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(o.r = aneon(a.r, b.r);) }    \
        else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(o.r = a512(a.r, b.r);) } \
        else if constexpr (vec<float, Tg>::K == 1) { FORGE_IF256(o.r = a256(a.r, b.r);) } \
        else for (int i = 0; i < vec<float, Tg>::N; ++i) { float x = a.r.v[i], y = b.r.v[i]; o.r.v[i] = (scalar_expr); } \
        return o;                                                                         \
    }
#if defined(__AVX512F__)
#define FORGE_IF512(...) __VA_ARGS__
#else
#define FORGE_IF512(...)
#endif
#if defined(__AVX2__)
#define FORGE_IF256(...) __VA_ARGS__
#else
#define FORGE_IF256(...)
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
// x86 max/min semantics (return the second operand when unordered), so every target agrees with the oracle
inline float32x4_t forge_neon_max(float32x4_t a, float32x4_t b) { return vbslq_f32(vcgtq_f32(a, b), a, b); }
inline float32x4_t forge_neon_min(float32x4_t a, float32x4_t b) { return vbslq_f32(vcltq_f32(a, b), a, b); }
#endif
FORGE_VOP(add, _mm512_add_ps, _mm256_add_ps, vaddq_f32, x + y)
FORGE_VOP(sub, _mm512_sub_ps, _mm256_sub_ps, vsubq_f32, x - y)
FORGE_VOP(mul, _mm512_mul_ps, _mm256_mul_ps, vmulq_f32, x * y)
FORGE_VOP(div, _mm512_div_ps, _mm256_div_ps, vdivq_f32, x / y)
FORGE_VOP(max, _mm512_max_ps, _mm256_max_ps, forge_neon_max, x > y ? x : y)
FORGE_VOP(min, _mm512_min_ps, _mm256_min_ps, forge_neon_min, x < y ? x : y)

template <Target Tg> inline vec<float, Tg> operator+(vec<float, Tg> a, vec<float, Tg> b) { return add(a, b); }
template <Target Tg> inline vec<float, Tg> operator-(vec<float, Tg> a, vec<float, Tg> b) { return sub(a, b); }
template <Target Tg> inline vec<float, Tg> operator*(vec<float, Tg> a, vec<float, Tg> b) { return mul(a, b); }
template <Target Tg> inline vec<float, Tg> operator/(vec<float, Tg> a, vec<float, Tg> b) { return div(a, b); }

template <Target Tg> inline vec<float, Tg> splat(float s) {
    vec<float, Tg> o;
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(o.r = vdupq_n_f32(s);) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(o.r = _mm512_set1_ps(s);) }
    else if constexpr (vec<float, Tg>::K == 1) { FORGE_IF256(o.r = _mm256_set1_ps(s);) }
    else for (auto& x : o.r.v) x = s;
    return o;
}
template <Target Tg> inline vec<float, Tg> fma(vec<float, Tg> a, vec<float, Tg> b, vec<float, Tg> c) {
    vec<float, Tg> o;
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(o.r = vfmaq_f32(c.r, a.r, b.r);) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(o.r = _mm512_fmadd_ps(a.r, b.r, c.r);) }
    else if constexpr (vec<float, Tg>::K == 1) { FORGE_IF256(o.r = _mm256_fmadd_ps(a.r, b.r, c.r);) }
    else for (int i = 0; i < vec<float, Tg>::N; ++i) o.r.v[i] = std::fma(a.r.v[i], b.r.v[i], c.r.v[i]);
    return o;
}
template <Target Tg> inline vec<float, Tg> load(const float* p) {
    vec<float, Tg> o;
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(o.r = vld1q_f32(p);) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(o.r = _mm512_loadu_ps(p);) }
    else if constexpr (vec<float, Tg>::K == 1) { FORGE_IF256(o.r = _mm256_loadu_ps(p);) }
    else std::memcpy(o.r.v, p, sizeof(o.r.v));
    return o;
}
// Masked load: lanes outside the mask read as 0 and never touch memory.
template <Target Tg> inline vec<float, Tg> load(const float* p, mask<float, Tg> m) {
    vec<float, Tg> o;
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(for (int i = 0; i < 4; ++i) o.r[i] = (m.bits() >> i) & 1 ? p[i] : 0.f;) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(o.r = _mm512_maskz_loadu_ps(m.m, p);) }
    else if constexpr (vec<float, Tg>::K == 1) { FORGE_IF256(o.r = _mm256_maskload_ps(p, _mm256_castps_si256(m.m));) }
    else for (int i = 0; i < vec<float, Tg>::N; ++i) o.r.v[i] = (m.m >> i) & 1 ? p[i] : 0.f;
    return o;
}
template <Target Tg> inline void store(float* p, vec<float, Tg> v) {
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(vst1q_f32(p, v.r);) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(_mm512_storeu_ps(p, v.r);) }
    else if constexpr (vec<float, Tg>::K == 1) { FORGE_IF256(_mm256_storeu_ps(p, v.r);) }
    else std::memcpy(p, v.r.v, sizeof(v.r.v));
}
template <Target Tg> inline void store(float* p, vec<float, Tg> v, mask<float, Tg> m) {
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(for (int i = 0; i < 4; ++i) if ((m.bits() >> i) & 1) p[i] = v.r[i];) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(_mm512_mask_storeu_ps(p, m.m, v.r);) }
    else if constexpr (vec<float, Tg>::K == 1) { FORGE_IF256(_mm256_maskstore_ps(p, _mm256_castps_si256(m.m), v.r);) }
    else for (int i = 0; i < vec<float, Tg>::N; ++i) if ((m.m >> i) & 1) p[i] = v.r.v[i];
}
// select(m, a, b): m ? a : b
template <Target Tg> inline vec<float, Tg> select(mask<float, Tg> m, vec<float, Tg> a, vec<float, Tg> b) {
    vec<float, Tg> o;
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(o.r = vbslq_f32(m.m, a.r, b.r);) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(o.r = _mm512_mask_blend_ps(m.m, b.r, a.r);) }
    else if constexpr (vec<float, Tg>::K == 1) { FORGE_IF256(o.r = _mm256_blendv_ps(b.r, a.r, m.m);) }
    else for (int i = 0; i < vec<float, Tg>::N; ++i) o.r.v[i] = (m.m >> i) & 1 ? a.r.v[i] : b.r.v[i];
    return o;
}
template <Target Tg> inline mask<float, Tg> lt(vec<float, Tg> a, vec<float, Tg> b) {
    mask<float, Tg> m;
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(m.m = vcltq_f32(a.r, b.r);) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(m.m = _mm512_cmp_ps_mask(a.r, b.r, _CMP_LT_OQ);) }
    else if constexpr (vec<float, Tg>::K == 1) { FORGE_IF256(m.m = _mm256_cmp_ps(a.r, b.r, _CMP_LT_OQ);) }
    else { m.m = 0; for (int i = 0; i < vec<float, Tg>::N; ++i) m.m |= u32(a.r.v[i] < b.r.v[i]) << i; }
    return m;
}
template <Target Tg> inline mask<float, Tg> isnan(vec<float, Tg> a) {
    mask<float, Tg> m;
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(m.m = vmvnq_u32(vceqq_f32(a.r, a.r));) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(m.m = _mm512_cmp_ps_mask(a.r, a.r, _CMP_UNORD_Q);) }
    else if constexpr (vec<float, Tg>::K == 1) { FORGE_IF256(m.m = _mm256_cmp_ps(a.r, a.r, _CMP_UNORD_Q);) }
    else { m.m = 0; for (int i = 0; i < vec<float, Tg>::N; ++i) m.m |= u32(a.r.v[i] != a.r.v[i]) << i; }
    return m;
}
template <Target Tg> inline mask<float, Tg> operator|(mask<float, Tg> a, mask<float, Tg> b) {
    mask<float, Tg> m;
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(m.m = vorrq_u32(a.m, b.m);) }
    else if constexpr (vec<float, Tg>::K == 2) m.m = a.m | b.m;
    else if constexpr (vec<float, Tg>::K == 1) { FORGE_IF256(m.m = _mm256_or_ps(a.m, b.m);) }
    else m.m = a.m | b.m;
    return m;
}
template <Target Tg> inline float reduce_add(vec<float, Tg> v) {
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(return vaddvq_f32(v.r);) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(return _mm512_reduce_add_ps(v.r);) }
    else if constexpr (vec<float, Tg>::K == 1) {
        FORGE_IF256(__m128 lo = _mm_add_ps(_mm256_castps256_ps128(v.r), _mm256_extractf128_ps(v.r, 1));
                    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
                    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
                    return _mm_cvtss_f32(lo);)
    }
    float s = 0; if constexpr (vec<float, Tg>::K == 0) for (float x : v.r.v) s += x;
    return s;
}
template <Target Tg> inline float reduce_max(vec<float, Tg> v) {
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(return vmaxvq_f32(v.r);) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(return _mm512_reduce_max_ps(v.r);) }
    float tmp[vec<float, Tg>::N]; store(tmp, v);
    float m = tmp[0]; for (float x : tmp) m = x > m ? x : m; return m;
}

// Scale by 2^n for integer-valued float n (used by exp). Exact for normal range.
template <Target Tg> inline vec<float, Tg> ldexp_int(vec<float, Tg> x, vec<float, Tg> n) {
    vec<float, Tg> o;
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(o.r = vmulq_f32(x.r, vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vcvtnq_s32_f32(n.r), vdupq_n_s32(127)), 23)));) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(o.r = _mm512_scalef_ps(x.r, n.r);) }
    else if constexpr (vec<float, Tg>::K == 1) {
        FORGE_IF256(__m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n.r), _mm256_set1_epi32(127)), 23);
                    o.r = _mm256_mul_ps(x.r, _mm256_castsi256_ps(e));)
    } else for (int i = 0; i < vec<float, Tg>::N; ++i) o.r.v[i] = std::ldexp(x.r.v[i], int(n.r.v[i]));
    return o;
}
template <Target Tg> inline vec<float, Tg> round_nearest(vec<float, Tg> x) {
    vec<float, Tg> o;
    if constexpr (vec<float, Tg>::K == 3) { FORGE_IFNEON(o.r = vrndnq_f32(x.r);) }
    else if constexpr (vec<float, Tg>::K == 2) { FORGE_IF512(o.r = _mm512_roundscale_ps(x.r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);) }
    else if constexpr (vec<float, Tg>::K == 1) { FORGE_IF256(o.r = _mm256_round_ps(x.r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);) }
    else for (int i = 0; i < vec<float, Tg>::N; ++i) o.r.v[i] = std::nearbyint(x.r.v[i]);
    return o;
}

// strip: strip-mining loop hiding width and tail. f(i, mask) is called with a full mask for
// full vectors and a partial native predicate for the tail (no scalar epilogue).
template <Target Tg, class T, class F> inline void strip(std::size_t n, F&& f) {
    constexpr std::size_t W = std::size_t(vec<float, Tg>::N);
    std::size_t i = 0;
    auto full = mask<float, Tg>::all();
    for (; i + W <= n; i += W) f(i, full);
    if (i < n) f(i, mask<float, Tg>::first_n(n - i));
}

} // namespace forge::lane
