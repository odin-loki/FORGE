// Integer matvec kernels (qdot_i8.hpp) written once with GNU vector types and compiled per ISA by
// isa_extra.cpp (SSE2 / SSE4.2, RVV, LSX / LASX, VSX, s390x VXE, WebAssembly SIMD128, MSA): 16 int8 lanes are
// widened to int16 products and then to int32 sums with __builtin_convertvector, which each compiler lowers to
// the ISA's widening multiplies. The including file defines QI8_NS. Every kernel forms the reference's exact
// int32 sums and then the reference's float steps, so the bits are the scalar reference's on every ISA.
//
// int16 bounds per lane: one product of |w| <= 63, |a| <= 127 is 8001; two products of |w| <= 31 are 7874;
// one product of |w| <= 128 is 16256.
#include <forge/simd/qdot_i8.hpp>
#include <algorithm>

namespace forge::simd::qi8::QI8_NS {

typedef i8 vb __attribute__((vector_size(16)));
typedef u8 vub __attribute__((vector_size(16)));
typedef i16 vh __attribute__((vector_size(32)));             // 16 x int16
typedef std::int32_t vw __attribute__((vector_size(64)));    // 16 x int32

#define QI8_INL [[maybe_unused]] static inline __attribute__((always_inline))
QI8_INL vb lds(const i8* p) { vb v; std::memcpy(&v, p, 16); return v; }
QI8_INL vub ldu(const u8* p) { vub v; std::memcpy(&v, p, 16); return v; }
QI8_INL vub bu(unsigned x) { return vub{} + u8(x); }
QI8_INL void pf(const u8* p, std::size_t bytes) {
    const int d = qi8_prefetch_bytes;
    if (d > 0) for (std::size_t o = 0; o < bytes; o += 64) __builtin_prefetch(p + d + o, 0, 3);
}
QI8_INL std::int32_t hsum(const vw& v) { std::int32_t s = 0; for (int i = 0; i < 16; ++i) s += v[i]; return s; }
// acc += sc * (w . a) over 16 products, spread over the int32 lanes
QI8_INL void dot16(vw& acc, vb w, vb a, std::int32_t sc) {
    const vh p = __builtin_convertvector(w, vh) * __builtin_convertvector(a, vh);
    acc += __builtin_convertvector(p, vw) * sc;
}
// the same over 32 products (|w| <= 31: the K-quant values of q4_k / q5_k)
QI8_INL void dot32(vw& acc, vb w0, vb w1, vb a0, vb a1, std::int32_t sc) {
    const vh p = __builtin_convertvector(w0, vh) * __builtin_convertvector(a0, vh) + __builtin_convertvector(w1, vh) * __builtin_convertvector(a1, vh);
    acc += __builtin_convertvector(p, vw) * sc;
}
// sum of 32 products, any w
QI8_INL std::int32_t sdot32(vb w0, vb w1, vb a0, vb a1) {
    const vh p0 = __builtin_convertvector(w0, vh) * __builtin_convertvector(a0, vh);
    const vh p1 = __builtin_convertvector(w1, vh) * __builtin_convertvector(a1, vh);
    const vw s = __builtin_convertvector(p0, vw) + __builtin_convertvector(p1, vw);
    return hsum(s);
}

// ---- activation quantisation: the reference's amax (fmax fold: NaN ignored), x * (1/d), nearest-even, clamp ----
QI8_INL float amax_n(const float* x, int n) {
    float m[16] = {};
    for (int i = 0; i < n; i += 16)
        for (int j = 0; j < 16; ++j) { const float a = __builtin_fabsf(x[i + j]); m[j] = a > m[j] ? a : m[j]; }
    float r = 0;
    for (int j = 0; j < 16; ++j) r = m[j] > r ? m[j] : r;
    return r;
}
// 16 floats -> 16 int8. |x * id| <= 127 (1 + ulp) for finite input, so adding and subtracting 1.5 * 2^23 rounds
// to the nearest integer, ties to even, like nearbyint; NaN fails both compares and becomes -127 as in q8_round.
QI8_INL void q16(const float* x, float id, i8* q) {
    for (int i = 0; i < 16; ++i) {
        const float v = x[i] * id;
        float r = v + 12582912.0f;
        r = r - 12582912.0f;
        r = r > -127.0f ? r : -127.0f;
        r = r >= 127.0f ? 127.0f : r;
        q[i] = i8(int(r));
    }
}
static void quantize(int fmt, const float* x, std::size_t k, void* out) {
    if (qfmt(fmt) == qfmt::q8_0) {
        auto* o = static_cast<act_32*>(out);
        for (std::size_t b = 0; b < k / 32; ++b, x += 32, ++o) {
            const float d = amax_n(x, 32) / 127.f, id = d != 0.f ? 1.f / d : 0.f;
            q16(x, id, o->q); q16(x + 16, id, o->q + 16);
            o->d = d;
        }
        return;
    }
    auto* o = static_cast<act_k*>(out);
    for (std::size_t b = 0; b < k / 256; ++b, x += 256, ++o) {
        const float d = amax_n(x, 256) / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        for (int s = 0; s < 16; ++s) {
            q16(x + 16 * s, id, o->q + 16 * s);
            int t = 0; for (int i = 0; i < 16; ++i) t += o->q[16 * s + i];
            o->bsum16[s] = i16(t);
        }
        for (int s = 0; s < 8; ++s) o->bsum32[s] = i16(o->bsum16[2 * s] + o->bsum16[2 * s + 1]);
        o->d = d;
    }
}

// ---- K-quant weights of one 256-value block as 16 chunks of 16 values (0..63), in activation order ----
// q4_k / q5_k: chunks 2s and 2s+1 carry sub-scale s. q6_k: chunk c carries sub-scale c.
QI8_INL void unpack_q4_k(const u8* qs, vb* q) {
    for (int j = 0; j < 4; ++j) {
        const vub a = ldu(qs + 32 * j), b = ldu(qs + 32 * j + 16);
        q[4 * j + 0] = vb(a & bu(0x0F)); q[4 * j + 1] = vb(b & bu(0x0F));
        q[4 * j + 2] = vb(a >> 4); q[4 * j + 3] = vb(b >> 4);
    }
}
QI8_INL void unpack_q5_k(const u8* qh, const u8* qs, vb* q) {
    const vub ha = ldu(qh), hb = ldu(qh + 16), m10 = bu(0x10), z = vub{};
    for (int j = 0; j < 4; ++j) {
        const vub a = ldu(qs + 32 * j), b = ldu(qs + 32 * j + 16);
        const vub bl = bu(1u << (2 * j)), bh = bu(2u << (2 * j));
        q[4 * j + 0] = vb((a & bu(0x0F)) | (vub((ha & bl) != z) & m10));
        q[4 * j + 1] = vb((b & bu(0x0F)) | (vub((hb & bl) != z) & m10));
        q[4 * j + 2] = vb((a >> 4) | (vub((ha & bh) != z) & m10));
        q[4 * j + 3] = vb((b >> 4) | (vub((hb & bh) != z) & m10));
    }
}
QI8_INL void unpack_q6_k(const u8* ql, const u8* qh, vb* q) {
    const vub m4 = bu(0x0F), m30 = bu(0x30);
    for (int h = 0; h < 2; ++h)
        for (int t = 0; t < 2; ++t) {
            const vub l0 = ldu(ql + 64 * h + 16 * t), l1 = ldu(ql + 64 * h + 32 + 16 * t), hh = ldu(qh + 32 * h + 16 * t);
            vb* o = q + 8 * h + t;
            o[0] = vb((l0 & m4) | ((hh << 4) & m30));
            o[2] = vb((l1 & m4) | ((hh << 2) & m30));
            o[4] = vb((l0 >> 4) | (hh & m30));
            o[6] = vb((l1 >> 4) | ((hh >> 2) & m30));
        }
}

// ---- products: nv <= 8 activation vectors per weight row (weights unpacked once per block for all of them) ----
static void rows_q45_k(bool q5, const u8* w, std::size_t rb, std::size_t rows, const act_k* const* av, int nv, std::size_t nb, float* y, std::size_t ys) {
    const std::size_t bb = q5 ? 176 : 144;
    for (std::size_t r = 0; r < rows; ++r) {
        const u8* blk = w + r * rb;
        float acc[8] = {}, accm[8] = {};
        for (std::size_t b = 0; b < nb; ++b, blk += bb) {
            pf(blk, bb);
            u8 sc[8], mn[8]; unpack_scales(blk + 4, sc, mn);
            vb q[16];
            if (q5) unpack_q5_k(blk + 16, blk + 48, q); else unpack_q4_k(blk + 16, q);
            const float d = h2f(blk), dmin = h2f(blk + 2);
            for (int v = 0; v < nv; ++v) {
                const act_k& a = av[v][b];
                vw ai = vw{};
                for (int s = 0; s < 8; ++s) dot32(ai, q[2 * s], q[2 * s + 1], lds(a.q + 32 * s), lds(a.q + 32 * s + 16), sc[s]);
                std::int32_t imin = 0;
                for (int s = 0; s < 8; ++s) imin += mn[s] * a.bsum32[s];
                acc[v] = fmac(d * a.d, float(hsum(ai)), acc[v]);
                accm[v] = fmac(dmin * a.d, float(imin), accm[v]);
            }
        }
        for (int v = 0; v < nv; ++v) y[std::size_t(v) * ys + r] = acc[v] - accm[v];
    }
}
static void rows_q6_k(const u8* w, std::size_t rb, std::size_t rows, const act_k* const* av, int nv, std::size_t nb, float* y, std::size_t ys) {
    for (std::size_t r = 0; r < rows; ++r) {
        const u8* blk = w + r * rb;
        float acc[8] = {};
        for (std::size_t b = 0; b < nb; ++b, blk += 210) {
            pf(blk, 210);
            const auto* sc = reinterpret_cast<const i8*>(blk + 192);
            vb q[16];
            unpack_q6_k(blk, blk + 128, q);
            const float d = h2f(blk + 208);
            for (int v = 0; v < nv; ++v) {
                const act_k& a = av[v][b];
                vw ai = vw{};
                for (int c = 0; c < 16; ++c) dot16(ai, q[c], lds(a.q + 16 * c), sc[c]);
                std::int32_t off = 0;
                for (int s = 0; s < 16; ++s) off += sc[s] * a.bsum16[s];
                acc[v] = fmac(d * a.d, float(hsum(ai) - 32 * off), acc[v]);
            }
        }
        for (int v = 0; v < nv; ++v) y[std::size_t(v) * ys + r] = acc[v];
    }
}
static void rows_q8_0(const u8* w, std::size_t rb, std::size_t rows, const act_32* const* av, int nv, std::size_t nb, float* y, std::size_t ys) {
    for (std::size_t r = 0; r < rows; ++r) {
        const u8* blk = w + r * rb;
        float acc[8] = {};
        for (std::size_t b = 0; b < nb; ++b, blk += 34) {
            if ((b & 3) == 0) pf(blk, 136);
            const vb w0 = lds(reinterpret_cast<const i8*>(blk + 2)), w1 = lds(reinterpret_cast<const i8*>(blk + 18));
            const float dw = h2f(blk);
            for (int v = 0; v < nv; ++v)
                acc[v] = fmac(dw * av[v][b].d, float(sdot32(w0, w1, lds(av[v][b].q), lds(av[v][b].q + 16))), acc[v]);
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
                const act_32* p[8]; for (int v = 0; v < n; ++v) p[v] = reinterpret_cast<const act_32*>(a + std::size_t(v) * as);
                rows_q8_0(w, rb, rows, p, n, k / 32, y, ys); break;
            }
            case qfmt::q4_k: case qfmt::q5_k: case qfmt::q6_k: {
                const act_k* p[8]; for (int v = 0; v < n; ++v) p[v] = reinterpret_cast<const act_k*>(a + std::size_t(v) * as);
                if (qfmt(fmt) == qfmt::q6_k) rows_q6_k(w, rb, rows, p, n, k / 256, y, ys);
                else rows_q45_k(qfmt(fmt) == qfmt::q5_k, w, rb, rows, p, n, k / 256, y, ys);
                break;
            }
            default: return;
        }
        nv -= std::size_t(n); a += std::size_t(n) * as; y += std::size_t(n) * ys;
    }
}
#undef QI8_INL

} // namespace forge::simd::qi8::QI8_NS
