// Integer matvec kernels, AArch64 SVE (svdot), vector-length agnostic: one object serves every register
// length (128..2048 bits, powers of two or not). CMake defines FORGE_QI8_HAVE_SVE when the compiler accepts
// the flags; the file is built WITHOUT -msve-vector-bits.
//
// Sizeless vectors cannot sit in arrays, so the per-vector accumulators of the NEON kernels are not available.
// Instead the K-quant weights of a block are unpacked once (NEON, 16 bytes at a time) into 256 bytes on the
// stack next to one int32 scale per four weights; each activation vector then takes one predicated pass:
//     acc += svdot(0, weights, activations) * scale        (int32 lanes; one lane = 4 products, one scale)
// Activation quantisation is the NEON code (qi8_arm.inc).
#if defined(FORGE_QI8_HAVE_SVE) && defined(__aarch64__)
#define QI8_NS sve
#define QI8_ARM_PART 1
#include "qi8_arm.inc"
#include <arm_sve.h>

namespace forge::simd::qi8::sve {

// sum over 256 weights of scale[i / 4] * w[i] * a[i]
static inline std::int32_t dot256(const i8* w, const std::int32_t* scl, const i8* a) {
    svint32_t acc = svdup_n_s32(0);
    const std::uint64_t step = svcntb();
    for (std::uint64_t o = 0; o < 256; o += step) {
        const svbool_t p8 = svwhilelt_b8_u64(o, 256), p32 = svwhilelt_b32_u64(o / 4, 64);
        // inactive bytes load as zero: their lanes add nothing
        const svint32_t d = svdot_s32(svdup_n_s32(0), svld1_s8(p8, w + o), svld1_s8(p8, a + o));
        acc = svmla_s32_m(p32, acc, d, svld1_s32(p32, scl + o / 4));
    }
    return std::int32_t(svaddv_s32(svptrue_b32(), acc));
}
// sum over 32 products
static inline std::int32_t dot32v(const i8* w, const i8* a) {
    svint32_t acc = svdup_n_s32(0);
    const std::uint64_t step = svcntb();
    for (std::uint64_t o = 0; o < 32; o += step) {
        const svbool_t p8 = svwhilelt_b8_u64(o, 32);
        acc = svdot_s32(acc, svld1_s8(p8, w + o), svld1_s8(p8, a + o));
    }
    return std::int32_t(svaddv_s32(svptrue_b32(), acc));
}

// Q: 4 = q4_k, 5 = q5_k, 6 = q6_k
template <int Q> static void rows_k(const u8* w, std::size_t rb, std::size_t rows, const act_k* const* av, int nv, std::size_t nb, float* y, std::size_t ys) {
    const std::size_t bb = Q == 4 ? 144 : Q == 5 ? 176 : 210;
    for (std::size_t r = 0; r < rows; ++r) {
        const u8* blk = w + r * rb;
        float acc[8] = {}, accm[8] = {};
        for (std::size_t b = 0; b < nb; ++b, blk += bb) {
            pf(blk, bb);
            int8x16_t q[16];
            alignas(64) i8 wq[256];
            alignas(64) std::int32_t scl[64];
            if constexpr (Q == 6) {
                const auto* sc = reinterpret_cast<const i8*>(blk + 192);
                unpack_q6_k(blk, blk + 128, q);
                for (int c = 0; c < 16; ++c) { vst1q_s8(wq + 16 * c, q[c]); vst1q_s32(scl + 4 * c, vdupq_n_s32(sc[c])); }
                const int16x8_t s0 = vmovl_s8(vld1_s8(sc)), s1 = vmovl_s8(vld1_s8(sc + 8));
                const float d = h2f_(blk + 208);
                for (int v = 0; v < nv; ++v) {
                    const act_k& a = av[v][b];
                    const std::int32_t isum = dot256(wq, scl, a.q), off = off_dot(s0, s1, a.bsum16);
                    acc[v] = fmac(d * a.d, float(isum - 32 * off), acc[v]);
                }
            } else {
                u8 sc[8], mn[8]; unpack_scales(blk + 4, sc, mn);
                if constexpr (Q == 5) unpack_q5_k(blk + 16, blk + 48, q); else unpack_q4_k(blk + 16, q);
                for (int c = 0; c < 16; ++c) vst1q_s8(wq + 16 * c, q[c]);
                for (int s = 0; s < 8; ++s) { vst1q_s32(scl + 8 * s, vdupq_n_s32(sc[s])); vst1q_s32(scl + 8 * s + 4, vdupq_n_s32(sc[s])); }
                const float d = h2f_(blk), dmin = h2f_(blk + 2);
                for (int v = 0; v < nv; ++v) {
                    const act_k& a = av[v][b];
                    const std::int32_t isum = dot256(wq, scl, a.q), imin = min_dot(mn, a.bsum32);
                    acc[v] = fmac(d * a.d, float(isum), acc[v]);
                    accm[v] = fmac(dmin * a.d, float(imin), accm[v]);
                }
            }
        }
        for (int v = 0; v < nv; ++v) y[std::size_t(v) * ys + r] = Q == 6 ? acc[v] : acc[v] - accm[v];
    }
}
static void rows_q8_0(const u8* w, std::size_t rb, std::size_t rows, const act_32* const* av, int nv, std::size_t nb, float* y, std::size_t ys) {
    for (std::size_t r = 0; r < rows; ++r) {
        const u8* blk = w + r * rb;
        float acc[8] = {};
        for (std::size_t b = 0; b < nb; ++b, blk += 34) {
            if (b % 4 == 0) pf(blk, 136);
            const float dw = h2f_(blk);
            const auto* wq = reinterpret_cast<const i8*>(blk + 2);
            for (int v = 0; v < nv; ++v) acc[v] = fmac(dw * av[v][b].d, float(dot32v(wq, av[v][b].q)), acc[v]);
        }
        for (int v = 0; v < nv; ++v) y[std::size_t(v) * ys + r] = acc[v];
    }
}

static void matvec(int fmt, const u8* w, std::size_t rb, std::size_t rows, const void* act, std::size_t as, std::size_t k, std::size_t nv, float* y, std::size_t ys) {
    const u8* a = static_cast<const u8*>(act);
    while (nv) {
        const int n = int(std::min<std::size_t>(nv, 8));
        switch (qfmt(fmt)) {
            case qfmt::q8_0: {
                const act_32* p[8]; for (int v = 0; v < n; ++v) p[v] = reinterpret_cast<const act_32*>(a + v * as);
                rows_q8_0(w, rb, rows, p, n, k / 32, y, ys); break;
            }
            case qfmt::q4_k: case qfmt::q5_k: case qfmt::q6_k: {
                const act_k* p[8]; for (int v = 0; v < n; ++v) p[v] = reinterpret_cast<const act_k*>(a + v * as);
                if (qfmt(fmt) == qfmt::q4_k) rows_k<4>(w, rb, rows, p, n, k / 256, y, ys);
                else if (qfmt(fmt) == qfmt::q5_k) rows_k<5>(w, rb, rows, p, n, k / 256, y, ys);
                else rows_k<6>(w, rb, rows, p, n, k / 256, y, ys);
                break;
            }
            default: return;
        }
        nv -= std::size_t(n); a += std::size_t(n) * as; y += std::size_t(n) * ys;
    }
}

} // namespace forge::simd::qi8::sve
namespace forge::simd::qi8_tables {
extern const qi8_kernels sve;
const qi8_kernels sve = {"sve", &qi8::sve::quantize, &qi8::sve::matvec};
}
#endif
