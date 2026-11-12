// AArch64 integer matvec kernels (qdot_i8.hpp), written once and compiled per ISA. The including TU defines
//   QI8_NS            a namespace unique to the TU (the bodies differ per ISA: no ODR clashes)
// and, unless QI8_ARM_NO_ROWS is defined (the SVE TU brings its own products), after including "part 1"
// (QI8_ARM_PART 1: helpers, quantisation, weight unpacking) these three in namespace forge::simd::qi8::QI8_NS:
//   sc_t / mksc(int)                       a broadcast sub-block scale
//   dot16(acc, w, a, sc)                   acc + sc * sum of 16 s8*s8 products, spread over the int32 lanes
//   dot32(acc, w0, w1, a0, a1, sc)         the same over 32 products (|w| <= 31: K-quant values of q4_k / q5_k)
//   sdot32(w0, w1, a0, a1)                 sum of 32 s8*s8 products over the int32 lanes (any w, |a| <= 127)
// followed by part 2 (QI8_ARM_PART 2: the products). All vectors are 128 bits.
// Every kernel forms the reference's exact int32 sums and then the reference's float steps (qdot_i8.hpp).
// Every unsigned K-quant value fits in 7 bits (q4_k <= 15, q5_k <= 31, q6_k <= 63), so the signed dot product
// is exact on them and the unsigned*signed form (i8mm usdot) has nothing to add.
#if QI8_ARM_PART == 1
#include <forge/simd/qdot_i8.hpp>
#if defined(_MSC_VER) && !defined(__clang__)
#include <arm64_neon.h>
#include <intrin.h>
#else
#include <arm_neon.h>
#endif
#include <algorithm>

namespace forge::simd::qi8::QI8_NS {

static inline int8x16_t lds(const i8* p) { return vld1q_s8(p); }
static inline uint8x16_t ldu(const u8* p) { return vld1q_u8(p); }
static inline float h2f_(const u8* p) {
#if defined(_MSC_VER) && !defined(__clang__)
    return h2f(p);
#else
    std::uint16_t b; std::memcpy(&b, p, 2);
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(b))), 0);   // exact, like the reference's decode
#endif
}
static inline void pf(const u8* p, std::size_t bytes) {
    const int d = qi8_prefetch_bytes;
    if (d > 0) for (std::size_t o = 0; o < bytes; o += 64) {
#if defined(_MSC_VER) && !defined(__clang__)
        __prefetch(p + d + o);
#else
        __builtin_prefetch(p + d + o, 0, 3);
#endif
    }
}

// ---- activation quantisation (the reference's rounding: x * (1/d), nearest-even, clamp to +-127, NaN -> -127) ----
static inline float amax_n(const float* x, int n) {
    float32x4_t m = vdupq_n_f32(0);
    for (int i = 0; i < n; i += 4) m = vmaxnmq_f32(m, vabsq_f32(vld1q_f32(x + i)));   // fmax: a NaN operand is ignored
    return vmaxnmvq_f32(m);
}
static inline int32x4_t q4f(const float* x, float32x4_t id) {
    float32x4_t v = vmulq_f32(vld1q_f32(x), id);
    // fcvtns: nearest-even, saturating; NaN gives 0 there, the reference gives -127
    return vbslq_s32(vceqq_f32(v, v), vcvtnq_s32_f32(v), vdupq_n_s32(-127));
}
// 16 floats -> 16 int8
static inline int8x16_t q16(const float* x, float32x4_t id) {
    int16x8_t lo = vcombine_s16(vqmovn_s32(q4f(x, id)), vqmovn_s32(q4f(x + 4, id)));
    int16x8_t hi = vcombine_s16(vqmovn_s32(q4f(x + 8, id)), vqmovn_s32(q4f(x + 12, id)));
    return vmaxq_s8(vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)), vdupq_n_s8(-127));
}
static void quantize(int fmt, const float* x, std::size_t k, void* out) {
    if (qfmt(fmt) == qfmt::q8_0) {
        auto* o = static_cast<act_32*>(out);
        for (std::size_t b = 0; b < k / 32; ++b, x += 32, ++o) {
            float d = amax_n(x, 32) / 127.f, id = d != 0.f ? 1.f / d : 0.f;
            float32x4_t vid = vdupq_n_f32(id);
            vst1q_s8(o->q, q16(x, vid)); vst1q_s8(o->q + 16, q16(x + 16, vid));
            o->d = d;
        }
        return;
    }
    auto* o = static_cast<act_k*>(out);
    for (std::size_t b = 0; b < k / 256; ++b, x += 256, ++o) {
        float d = amax_n(x, 256) / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        float32x4_t vid = vdupq_n_f32(id);
        for (int s = 0; s < 16; ++s) {
            int8x16_t q = q16(x + 16 * s, vid);
            vst1q_s8(o->q + 16 * s, q);
            o->bsum16[s] = i16(vaddlvq_s8(q));   // sum of 16 (|.| <= 2032)
        }
        for (int s = 0; s < 8; ++s) o->bsum32[s] = i16(o->bsum16[2 * s] + o->bsum16[2 * s + 1]);
        o->d = d;
    }
}

// ---- K-quant weights of one 256-value block as 16 chunks of 16 values (0..63), in activation order ----
// q4_k / q5_k: chunks 2s and 2s+1 carry sub-scale s (8 per block). q6_k: chunk c carries sub-scale c (16 per block).
static inline void unpack_q4_k(const u8* qs, int8x16_t* q) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int j = 0; j < 4; ++j) {
        const uint8x16_t a = ldu(qs + 32 * j), b = ldu(qs + 32 * j + 16);
        q[4 * j + 0] = vreinterpretq_s8_u8(vandq_u8(a, m4)); q[4 * j + 1] = vreinterpretq_s8_u8(vandq_u8(b, m4));
        q[4 * j + 2] = vreinterpretq_s8_u8(vshrq_n_u8(a, 4)); q[4 * j + 3] = vreinterpretq_s8_u8(vshrq_n_u8(b, 4));
    }
}
static inline void unpack_q5_k(const u8* qh, const u8* qs, int8x16_t* q) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m10 = vdupq_n_u8(0x10);
    const uint8x16_t ha = ldu(qh), hb = ldu(qh + 16);
    for (int j = 0; j < 4; ++j) {
        const uint8x16_t a = ldu(qs + 32 * j), b = ldu(qs + 32 * j + 16);
        const uint8x16_t bl = vdupq_n_u8(u8(1u << (2 * j))), bh = vdupq_n_u8(u8(2u << (2 * j)));
        q[4 * j + 0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a, m4), vandq_u8(vtstq_u8(ha, bl), m10)));
        q[4 * j + 1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(b, m4), vandq_u8(vtstq_u8(hb, bl), m10)));
        q[4 * j + 2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(a, 4), vandq_u8(vtstq_u8(ha, bh), m10)));
        q[4 * j + 3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(b, 4), vandq_u8(vtstq_u8(hb, bh), m10)));
    }
}
static inline void unpack_q6_k(const u8* ql, const u8* qh, int8x16_t* q) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m30 = vdupq_n_u8(0x30);
    for (int h = 0; h < 2; ++h)
        for (int t = 0; t < 2; ++t) {
            const uint8x16_t l0 = ldu(ql + 64 * h + 16 * t), l1 = ldu(ql + 64 * h + 32 + 16 * t), hh = ldu(qh + 32 * h + 16 * t);
            int8x16_t* o = q + 8 * h + t;
            o[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l0, m4), vandq_u8(vshlq_n_u8(hh, 4), m30)));
            o[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l1, m4), vandq_u8(vshlq_n_u8(hh, 2), m30)));
            o[4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l0, 4), vandq_u8(hh, m30)));
            o[6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l1, 4), vandq_u8(vshrq_n_u8(hh, 2), m30)));
        }
}
// sum of mn[s] * bsum32[s] (q4_k / q5_k mins) and of sc[s] * bsum16[s] (q6_k offset)
static inline std::int32_t min_dot(const u8* mn, const i16* bsum32) {
    const int16x8_t m = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mn))), b = vld1q_s16(bsum32);
    return vaddvq_s32(vmlal_high_s16(vmull_s16(vget_low_s16(m), vget_low_s16(b)), m, b));
}
static inline std::int32_t off_dot(int16x8_t s0, int16x8_t s1, const i16* bsum16) {
    const int16x8_t b0 = vld1q_s16(bsum16), b1 = vld1q_s16(bsum16 + 8);
    int32x4_t t = vmull_s16(vget_low_s16(s0), vget_low_s16(b0));
    t = vmlal_high_s16(t, s0, b0); t = vmlal_s16(t, vget_low_s16(s1), vget_low_s16(b1)); t = vmlal_high_s16(t, s1, b1);
    return vaddvq_s32(t);
}

} // namespace forge::simd::qi8::QI8_NS

#elif QI8_ARM_PART == 2
namespace forge::simd::qi8::QI8_NS {

// ---- products: NV activation vectors per weight row (weights unpacked once per block for all of them) ----
// q4_k (Q5 = false) and q5_k (Q5 = true): the same block layout apart from the 32 bytes of fifth bits.
template <int NV, bool Q5> static void rows_q45_k(const u8* w, std::size_t rb, std::size_t rows, const act_k* const* av, std::size_t nb, float* y, std::size_t ys) {
    const std::size_t bb = Q5 ? 176 : 144;
    for (std::size_t r = 0; r < rows; ++r) {
        const u8* blk = w + r * rb;
        float acc[NV] = {}, accm[NV] = {};
        for (std::size_t b = 0; b < nb; ++b, blk += bb) {
            pf(blk, bb);
            u8 sc[8], mn[8]; unpack_scales(blk + 4, sc, mn);
            int8x16_t q[16];
            if constexpr (Q5) unpack_q5_k(blk + 16, blk + 48, q); else unpack_q4_k(blk + 16, q);
            int32x4_t ai[NV];
            for (int v = 0; v < NV; ++v) ai[v] = vdupq_n_s32(0);
            for (int s = 0; s < 8; ++s) {
                const sc_t sv = mksc(sc[s]);
                for (int v = 0; v < NV; ++v) {
                    const i8* aq = av[v][b].q + 32 * s;
                    ai[v] = dot32(ai[v], q[2 * s], q[2 * s + 1], lds(aq), lds(aq + 16), sv);
                }
            }
            const float d = h2f_(blk), dmin = h2f_(blk + 2);
            for (int v = 0; v < NV; ++v) {
                const act_k& a = av[v][b];
                const std::int32_t imin = min_dot(mn, a.bsum32);
                acc[v] = fmac(d * a.d, float(vaddvq_s32(ai[v])), acc[v]);
                accm[v] = fmac(dmin * a.d, float(imin), accm[v]);
            }
        }
        for (int v = 0; v < NV; ++v) y[std::size_t(v) * ys + r] = acc[v] - accm[v];
    }
}
template <int NV> static void rows_q6_k(const u8* w, std::size_t rb, std::size_t rows, const act_k* const* av, std::size_t nb, float* y, std::size_t ys) {
    for (std::size_t r = 0; r < rows; ++r) {
        const u8* blk = w + r * rb;
        float acc[NV] = {};
        for (std::size_t b = 0; b < nb; ++b, blk += 210) {
            pf(blk, 210);
            const auto* sc = reinterpret_cast<const i8*>(blk + 192);
            int8x16_t q[16];
            unpack_q6_k(blk, blk + 128, q);
            int32x4_t ai[NV];
            for (int v = 0; v < NV; ++v) ai[v] = vdupq_n_s32(0);
            for (int c = 0; c < 16; ++c) {
                const sc_t sv = mksc(sc[c]);
                for (int v = 0; v < NV; ++v) ai[v] = dot16(ai[v], q[c], lds(av[v][b].q + 16 * c), sv);
            }
            const int16x8_t s0 = vmovl_s8(vld1_s8(sc)), s1 = vmovl_s8(vld1_s8(sc + 8));
            const float d = h2f_(blk + 208);
            for (int v = 0; v < NV; ++v) {
                const act_k& a = av[v][b];
                const std::int32_t off = off_dot(s0, s1, a.bsum16);
                acc[v] = fmac(d * a.d, float(vaddvq_s32(ai[v]) - 32 * off), acc[v]);
            }
        }
        for (int v = 0; v < NV; ++v) y[std::size_t(v) * ys + r] = acc[v];
    }
}
template <int NV> static void rows_q8_0(const u8* w, std::size_t rb, std::size_t rows, const act_32* const* av, std::size_t nb, float* y, std::size_t ys) {
    for (std::size_t r = 0; r < rows; ++r) {
        const u8* blk = w + r * rb;
        float acc[NV] = {};
        std::size_t b = 0;
        // four blocks at a time: one pairwise reduction for their four sums
        for (; b + 4 <= nb; b += 4, blk += 136) {
            pf(blk, 136);
            int8x16_t w0[4], w1[4];
            for (int i = 0; i < 4; ++i) { w0[i] = lds(reinterpret_cast<const i8*>(blk + 34 * i + 2)); w1[i] = lds(reinterpret_cast<const i8*>(blk + 34 * i + 18)); }
            float dw[4]; for (int i = 0; i < 4; ++i) dw[i] = h2f_(blk + 34 * i);
            for (int v = 0; v < NV; ++v) {
                int32x4_t p[4];
                for (int i = 0; i < 4; ++i) p[i] = sdot32(w0[i], w1[i], lds(av[v][b + i].q), lds(av[v][b + i].q + 16));
                std::int32_t is[4]; vst1q_s32(is, vpaddq_s32(vpaddq_s32(p[0], p[1]), vpaddq_s32(p[2], p[3])));   // sums of blocks 0..3
                for (int i = 0; i < 4; ++i) acc[v] = fmac(dw[i] * av[v][b + i].d, float(is[i]), acc[v]);
            }
        }
        for (; b < nb; ++b, blk += 34) {
            const int8x16_t w0 = lds(reinterpret_cast<const i8*>(blk + 2)), w1 = lds(reinterpret_cast<const i8*>(blk + 18));
            const float dw = h2f_(blk);
            for (int v = 0; v < NV; ++v)
                acc[v] = fmac(dw * av[v][b].d, float(vaddvq_s32(sdot32(w0, w1, lds(av[v][b].q), lds(av[v][b].q + 16)))), acc[v]);
        }
        for (int v = 0; v < NV; ++v) y[std::size_t(v) * ys + r] = acc[v];
    }
}

template <int NV> static void group(int fmt, const u8* w, std::size_t rb, std::size_t rows, const u8* act, std::size_t as, std::size_t k, float* y, std::size_t ys) {
    switch (qfmt(fmt)) {
        case qfmt::q8_0: { const act_32* p[NV]; for (int v = 0; v < NV; ++v) p[v] = reinterpret_cast<const act_32*>(act + v * as); return rows_q8_0<NV>(w, rb, rows, p, k / 32, y, ys); }
        case qfmt::q4_k: case qfmt::q5_k: case qfmt::q6_k: {
            const act_k* p[NV]; for (int v = 0; v < NV; ++v) p[v] = reinterpret_cast<const act_k*>(act + v * as);
            if (qfmt(fmt) == qfmt::q4_k) return rows_q45_k<NV, false>(w, rb, rows, p, k / 256, y, ys);
            if (qfmt(fmt) == qfmt::q5_k) return rows_q45_k<NV, true>(w, rb, rows, p, k / 256, y, ys);
            return rows_q6_k<NV>(w, rb, rows, p, k / 256, y, ys);
        }
        default: return;
    }
}
static void matvec(int fmt, const u8* w, std::size_t rb, std::size_t rows, const void* act, std::size_t as, std::size_t k, std::size_t nv, float* y, std::size_t ys) {
    const u8* a = static_cast<const u8*>(act);
    while (nv) {
        std::size_t n = std::min<std::size_t>(nv, 8);
        switch (n) {
            case 1: group<1>(fmt, w, rb, rows, a, as, k, y, ys); break;
            case 2: group<2>(fmt, w, rb, rows, a, as, k, y, ys); break;
            case 3: group<3>(fmt, w, rb, rows, a, as, k, y, ys); break;
            case 4: group<4>(fmt, w, rb, rows, a, as, k, y, ys); break;
            case 5: group<5>(fmt, w, rb, rows, a, as, k, y, ys); break;
            case 6: group<6>(fmt, w, rb, rows, a, as, k, y, ys); break;
            case 7: group<7>(fmt, w, rb, rows, a, as, k, y, ys); break;
            default: group<8>(fmt, w, rb, rows, a, as, k, y, ys); break;
        }
        nv -= n; a += n * as; y += n * ys;
    }
}

} // namespace forge::simd::qi8::QI8_NS
#endif
