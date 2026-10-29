#pragma once
// Vectorised libm with accuracy policies. Errors are measured by tests/test_lane_math.cpp.
#include <forge/lane/vec.hpp>

namespace forge::lane {

struct precise {}; // target <= 1-2 ULP over normal range
struct fast {};    // target <= 4 ULP
struct ml {};      // relative error ~1e-4..1e-3

template <class P> concept Policy = std::is_same_v<P, precise> || std::is_same_v<P, fast> || std::is_same_v<P, ml>;

template <Target Tg> inline vec<float, Tg> clamp(vec<float, Tg> x, float lo, float hi) {
    return min(max(x, splat<Tg>(lo)), splat<Tg>(hi));
}

template <Policy P = precise, Target Tg> inline vec<float, Tg> exp(vec<float, Tg> x) {
    using V = vec<float, Tg>;
    auto nan = isnan(x);
    auto over = lt(splat<Tg>(88.7228394f), x);
    auto under = lt(x, splat<Tg>(-104.0f));
    V x0 = x;
    x = clamp(x, -103.9f, 88.72f);
    V n = round_nearest(x * splat<Tg>(1.44269504088896341f));
    V r = fma(n, splat<Tg>(-0.693359375f), x);
    r = fma(n, splat<Tg>(2.12194440e-4f), r);
    V p;
    if constexpr (std::is_same_v<P, ml>) {
        p = splat<Tg>(4.16666716e-2f);
        p = fma(p, r, splat<Tg>(1.66666672e-1f));
        p = fma(p, r, splat<Tg>(0.5f));
        p = fma(p, r, splat<Tg>(1.0f));
        p = fma(p, r, splat<Tg>(1.0f));
    } else {
        p = splat<Tg>(1.9875691500e-4f);
        p = fma(p, r, splat<Tg>(1.3981999507e-3f));
        p = fma(p, r, splat<Tg>(8.3334519073e-3f));
        p = fma(p, r, splat<Tg>(4.1665795894e-2f));
        p = fma(p, r, splat<Tg>(1.6666665459e-1f));
        p = fma(p, r, splat<Tg>(5.0000001201e-1f));
        p = fma(p, r * r, r + splat<Tg>(1.0f));
    }
    // split scaling so n down to -150 stays representable through subnormals
    V half = round_nearest(n * splat<Tg>(0.5f));
    V y = ldexp_int(ldexp_int(p, half), n - half);
    y = select(over, splat<Tg>(INFINITY), y);
    y = select(under, splat<Tg>(0.0f), y);
    return select(nan, x0, y);
}

// frexp-style split: x = m * 2^e with m in [sqrt(1/2), sqrt(2)). x must be positive and normal.
template <Target Tg> inline void split_exp(vec<float, Tg> x, vec<float, Tg>& m, vec<float, Tg>& e) {
    if constexpr (vec<float, Tg>::K == 2) {
        FORGE_IF512(
        __m512 mm = _mm512_getmant_ps(x.r, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);
        __m512 ee = _mm512_getexp_ps(x.r);
        __mmask16 big = _mm512_cmp_ps_mask(mm, _mm512_set1_ps(1.41421356f), _CMP_GT_OQ);
        mm = _mm512_mask_mul_ps(mm, big, mm, _mm512_set1_ps(0.5f));
        ee = _mm512_mask_add_ps(ee, big, ee, _mm512_set1_ps(1.0f));
        m.r = mm; e.r = ee;)
    } else if constexpr (vec<float, Tg>::K == 1) {
        FORGE_IF256(
        __m256i xi = _mm256_castps_si256(x.r);
        __m256i ei = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
        __m256 mm = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(xi, _mm256_set1_epi32(0x007fffff)), _mm256_set1_epi32(0x3f800000)));
        __m256 ee = _mm256_cvtepi32_ps(ei);
        __m256 big = _mm256_cmp_ps(mm, _mm256_set1_ps(1.41421356f), _CMP_GT_OQ);
        mm = _mm256_blendv_ps(mm, _mm256_mul_ps(mm, _mm256_set1_ps(0.5f)), big);
        ee = _mm256_blendv_ps(ee, _mm256_add_ps(ee, _mm256_set1_ps(1.0f)), big);
        m.r = mm; e.r = ee;)
    } else {
        for (int i = 0; i < vec<float, Tg>::N; ++i) {
            int ex; float mm = std::frexp(x.r.v[i], &ex); mm *= 2; --ex;
            if (mm > 1.41421356f) { mm *= 0.5f; ++ex; }
            m.r.v[i] = mm; e.r.v[i] = float(ex);
        }
    }
}

template <Policy P = precise, Target Tg> inline vec<float, Tg> log(vec<float, Tg> x) {
    using V = vec<float, Tg>;
    V m, e;
    auto bad = lt(x, splat<Tg>(1.17549435e-38f));  // <= 0, denormal handled below
    bad = bad | lt(splat<Tg>(3.40282347e38f), x) | isnan(x);
    V xs = select(bad, splat<Tg>(1.0f), x);
    split_exp(xs, m, e);
    V f = m - splat<Tg>(1.0f);
    // log(1+f) = 2 atanh(s), s = f / (2 + f)
    V s = f / (f + splat<Tg>(2.0f));
    V z = s * s;
    V p;
    if constexpr (std::is_same_v<P, ml>) {
        p = fma(z, splat<Tg>(0.4f), splat<Tg>(0.66666667f));
    } else {
        p = splat<Tg>(0.24279078841f);
        p = fma(p, z, splat<Tg>(0.28498786688f));
        p = fma(p, z, splat<Tg>(0.40000972152f));
        p = fma(p, z, splat<Tg>(0.66666662693f));
    }
    V r = fma(s * z, p, s + s); // 2s + 2s^3/3 + ...
    V y = fma(e, splat<Tg>(0.693147180559945f), r);
    if (bad.any()) {
        // slow path for non-normal lanes keeps semantics identical to std::log
        float tx[V::N], ty[V::N]; store(tx, x); store(ty, y);
        for (int i = 0; i < V::N; ++i) if ((bad.bits() >> i) & 1 || !(tx[i] == tx[i])) ty[i] = std::log(tx[i]);
        y = load<Tg>(ty);
    }
    return y;
}

template <Policy P = precise, Target Tg> inline vec<float, Tg> sigmoid(vec<float, Tg> x) {
    auto one = splat<Tg>(1.0f);
    return one / (one + exp<P>(splat<Tg>(0.0f) - x));
}
template <Policy P = precise, Target Tg> inline vec<float, Tg> silu(vec<float, Tg> x) { return x * sigmoid<P>(x); }

template <Policy P = precise, Target Tg> inline vec<float, Tg> tanh(vec<float, Tg> x) {
    using V = vec<float, Tg>;
    // small |x|: odd polynomial (avoids cancellation); else 1 - 2/(e^{2x}+1)
    V ax = max(x, splat<Tg>(0.0f) - x);
    V e2 = exp<P>(clamp(x + x, -88.f, 88.f));
    V big = splat<Tg>(1.0f) - splat<Tg>(2.0f) / (e2 + splat<Tg>(1.0f));
    V z = x * x;
    V p = splat<Tg>(-5.70498872745e-3f);
    p = fma(p, z, splat<Tg>(2.06390887954e-2f));
    p = fma(p, z, splat<Tg>(-5.37397155531e-2f));
    p = fma(p, z, splat<Tg>(1.33314422036e-1f));
    p = fma(p, z, splat<Tg>(-3.33332819422e-1f));
    V small = fma(p * z, x, x);
    return select(lt(ax, splat<Tg>(0.625f)), small, big);
}

template <Policy P = precise, Target Tg> inline vec<float, Tg> gelu(vec<float, Tg> x) {
    // tanh form is used for ml/fast; precise uses tanh form too but with precise tanh
    auto c = splat<Tg>(0.7978845608028654f);
    auto inner = c * fma(splat<Tg>(0.044715f) * x, x * x, x);
    return splat<Tg>(0.5f) * x * (splat<Tg>(1.0f) + tanh<P>(inner));
}

} // namespace forge::lane
