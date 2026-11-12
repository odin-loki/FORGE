#pragma once
// Quantisation codecs: block layout + pack + unpack. Verified against dequantise-then-compute.
#include <forge/core/dtypes.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

namespace forge::graph::codecs {

struct codec {
    std::string_view name;
    dtype tag;
    std::size_t block;        // elements per block
    std::size_t block_bytes;  // bytes per block
    void (*encode)(const float*, std::size_t, u8*);
    void (*decode)(const u8*, std::size_t, float*);
    std::size_t bytes(std::size_t n) const { return (n + block - 1) / block * block_bytes; }
};

namespace detail {
inline void enc_f32(const float* x, std::size_t n, u8* o) { std::memcpy(o, x, n * 4); }
inline void dec_f32(const u8* i, std::size_t n, float* x) { std::memcpy(x, i, n * 4); }
inline void enc_f16(const float* x, std::size_t n, u8* o) { for (std::size_t i = 0; i < n; ++i) { u16 b = f16(x[i]).bits; std::memcpy(o + 2 * i, &b, 2); } }
inline void dec_f16(const u8* in, std::size_t n, float* x) { for (std::size_t i = 0; i < n; ++i) { u16 b; std::memcpy(&b, in + 2 * i, 2); x[i] = float(f16::from_bits(b)); } }
inline void enc_bf16(const float* x, std::size_t n, u8* o) { for (std::size_t i = 0; i < n; ++i) { u16 b = bf16(x[i]).bits; std::memcpy(o + 2 * i, &b, 2); } }
inline void dec_bf16(const u8* in, std::size_t n, float* x) { for (std::size_t i = 0; i < n; ++i) { u16 b; std::memcpy(&b, in + 2 * i, 2); x[i] = float(bf16::from_bits(b)); } }
inline void enc_e4m3(const float* x, std::size_t n, u8* o) { for (std::size_t i = 0; i < n; ++i) o[i] = fp8_e4m3(x[i]).bits; }
inline void dec_e4m3(const u8* in, std::size_t n, float* x) { for (std::size_t i = 0; i < n; ++i) x[i] = float(fp8_e4m3::from_bits(in[i])); }

// Q8_0 (GGUF-compatible): 32 values, f16 scale d + 32 x int8. x = d * q
inline void enc_q8_0(const float* x, std::size_t n, u8* o) {
    for (std::size_t b = 0; b < n; b += 32, o += 34) {
        float amax = 0; std::size_t m = std::min<std::size_t>(32, n - b);
        for (std::size_t i = 0; i < m; ++i) amax = std::max(amax, std::fabs(x[b + i]));
        float d = amax / 127.f; float id = d ? 1.f / d : 0.f;
        u16 hb = f16(d).bits; std::memcpy(o, &hb, 2);
        for (std::size_t i = 0; i < 32; ++i) o[2 + i] = u8(i8(i < m ? std::lround(x[b + i] * id) : 0));
    }
}
inline void dec_q8_0(const u8* in, std::size_t n, float* x) {
    for (std::size_t b = 0; b < n; b += 32, in += 34) {
        u16 hb; std::memcpy(&hb, in, 2); float d = float(f16::from_bits(hb));
        for (std::size_t i = 0; i < 32 && b + i < n; ++i) x[b + i] = d * float(i8(in[2 + i]));
    }
}

// Q4_K (GGUF-compatible layout): super-block of 256 = 8 sub-blocks of 32.
// struct { f16 d; f16 dmin; u8 scales[12]; u8 qs[128]; }  x = d*sc*q - dmin*m
inline void get_scale_min_k4(int j, const u8* q, u8& d, u8& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = u8((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); m = u8((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}
inline void enc_q4_k(const float* x, std::size_t n, u8* o) {
    for (std::size_t b = 0; b < n; b += 256, o += 144) {
        float blk[256] = {};
        std::copy_n(x + b, std::min<std::size_t>(256, n - b), blk);
        float sc[8], mn[8];
        for (int s = 0; s < 8; ++s) {
            float lo = 0, hi = 0;   // include 0 so min >= 0 offset is representable
            for (int i = 0; i < 32; ++i) { lo = std::min(lo, blk[s * 32 + i]); hi = std::max(hi, blk[s * 32 + i]); }
            sc[s] = (hi - lo) / 15.f; mn[s] = -lo;
        }
        float msc = *std::max_element(sc, sc + 8), mmn = *std::max_element(mn, mn + 8);
        float d = msc / 63.f, dmin = mmn / 63.f;
        u16 hd = f16(d).bits, hm = f16(dmin).bits; std::memcpy(o, &hd, 2); std::memcpy(o + 2, &hm, 2);
        d = float(f16::from_bits(hd)); dmin = float(f16::from_bits(hm));
        u8 ls[8], lm[8];
        for (int s = 0; s < 8; ++s) {
            ls[s] = u8(std::clamp(d ? int(std::lround(sc[s] / d)) : 0, 0, 63));
            lm[s] = u8(std::clamp(dmin ? int(std::lround(mn[s] / dmin)) : 0, 0, 63));
            // make sure the scale covers the range after rounding
            if (d && ls[s] < 63 && ls[s] * d * 15.f < (sc[s] * 15.f) - 1e-6f) ++ls[s];
        }
        u8* scales = o + 4;
        std::memset(scales, 0, 12);
        for (int j = 0; j < 4; ++j) { scales[j] = u8(ls[j] | ((ls[j + 4] >> 4) << 6)); scales[j + 4] = u8(lm[j] | ((lm[j + 4] >> 4) << 6)); }
        for (int j = 4; j < 8; ++j) scales[j + 4] = u8((ls[j] & 0xF) | ((lm[j] & 0xF) << 4));
        u8* qs = o + 16;
        for (int j = 0; j < 256; j += 64) {
            int is = j / 32;
            u8 s1, m1, s2, m2; get_scale_min_k4(is, scales, s1, m1); get_scale_min_k4(is + 1, scales, s2, m2);
            float d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
            for (int l = 0; l < 32; ++l) {
                int q1 = d1 ? int(std::lround((blk[j + l] + mm1) / d1)) : 0;
                int q2 = d2 ? int(std::lround((blk[j + 32 + l] + mm2) / d2)) : 0;
                qs[j / 2 + l] = u8(std::clamp(q1, 0, 15) | (std::clamp(q2, 0, 15) << 4));
            }
        }
    }
}
inline void dec_q4_k(const u8* in, std::size_t n, float* x) {
    for (std::size_t b = 0; b < n; b += 256, in += 144) {
        u16 hd, hm; std::memcpy(&hd, in, 2); std::memcpy(&hm, in + 2, 2);
        float d = float(f16::from_bits(hd)), dmin = float(f16::from_bits(hm));
        const u8* scales = in + 4; const u8* q = in + 16;
        float out[256];
        int is = 0;
        for (int j = 0; j < 256; j += 64, q += 32, is += 2) {
            u8 sc, m; get_scale_min_k4(is, scales, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) out[j + l] = d1 * float(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) out[j + 32 + l] = d2 * float(q[l] >> 4) - m2;
        }
        std::copy_n(out, std::min<std::size_t>(256, n - b), x + b);
    }
}

// FP8 E5M2 per element
inline void enc_e5m2(const float* x, std::size_t n, u8* o) { for (std::size_t i = 0; i < n; ++i) o[i] = fp8_e5m2(x[i]).bits; }
inline void dec_e5m2(const u8* in, std::size_t n, float* x) { for (std::size_t i = 0; i < n; ++i) x[i] = float(fp8_e5m2::from_bits(in[i])); }

inline void put16(u8* o, float v) { u16 b = f16(v).bits; std::memcpy(o, &b, 2); }
inline float get16(const u8* i) { u16 b; std::memcpy(&b, i, 2); return float(f16::from_bits(b)); }

// Group-wise asymmetric weight-only quantisation (AWQ / GPTQ style), group 32:
// { f16 scale; f16 min; q[32] } with x = q * scale + min, q in [0, 2^bits - 1].
template <int BITS> inline void enc_qg(const float* x, std::size_t n, u8* o) {
    constexpr int L = (1 << BITS) - 1;
    constexpr std::size_t BB = 4 + 32 * BITS / 8;
    for (std::size_t b = 0; b < n; b += 32, o += BB) {
        std::size_t m = std::min<std::size_t>(32, n - b);
        float lo = x[b], hi = x[b];
        for (std::size_t i = 0; i < m; ++i) { lo = std::min(lo, x[b + i]); hi = std::max(hi, x[b + i]); }
        float sc = (hi - lo) / float(L);
        float mn = float(f16(lo));                          // the stored min is what decode sees
        float scs = float(f16(sc));
        if (scs * float(L) + mn < hi) scs = float(f16(sc * 1.001f + 1e-7f));
        put16(o, scs); put16(o + 2, mn);
        float id = scs > 0 ? 1.f / scs : 0.f;
        std::memset(o + 4, 0, BB - 4);
        for (std::size_t i = 0; i < m; ++i) {
            int q = int(std::lround((x[b + i] - mn) * id)); q = std::clamp(q, 0, L);
            if constexpr (BITS == 8) o[4 + i] = u8(q);
            else o[4 + i / 2] |= u8(q << (4 * (i & 1)));
        }
    }
}
template <int BITS> inline void dec_qg(const u8* in, std::size_t n, float* x) {
    constexpr std::size_t BB = 4 + 32 * BITS / 8;
    for (std::size_t b = 0; b < n; b += 32, in += BB) {
        float sc = get16(in), mn = get16(in + 2);
        for (std::size_t i = 0; i < 32 && b + i < n; ++i) {
            int q = BITS == 8 ? in[4 + i] : (in[4 + i / 2] >> (4 * (i & 1))) & 0xf;
            x[b + i] = float(q) * sc + mn;
        }
    }
}

// OCP Microscaling (MX v1.0): blocks of 32 elements share an E8M0 scale X (a power of two, bias 127).
// X = floor(log2(amax)) - emax_elem; elements store x / 2^X in the element format.
inline int e8m0_exp(float amax, int emax) {
    if (!(amax > 0) || std::isinf(amax)) return 0;
    int e; std::frexp(amax, &e);                   // amax = f * 2^e, f in [0.5, 1)
    return std::clamp(e - 1 - emax, -127, 127);
}
template <class EL, int EMAX, int ELBITS> inline void enc_mx(const float* x, std::size_t n, u8* o) {
    constexpr std::size_t BB = 1 + 32 * ELBITS / 8;
    for (std::size_t b = 0; b < n; b += 32, o += BB) {
        std::size_t m = std::min<std::size_t>(32, n - b);
        float amax = 0; for (std::size_t i = 0; i < m; ++i) amax = std::max(amax, std::fabs(x[b + i]));
        int X = e8m0_exp(amax, EMAX);
        o[0] = u8(X + 127);
        std::memset(o + 1, 0, BB - 1);
        for (std::size_t i = 0; i < m; ++i) {
            float v = std::ldexp(x[b + i], -X);
            u32 code;
            if constexpr (std::is_same_v<EL, i8>) code = u8(i8(std::clamp<long>(std::lround(v * 64.f), -127, 127)));   // MXINT8: 1.6 fixed point
            else code = EL(v).bits;
            std::size_t bit = i * ELBITS;
            for (int k = 0; k < ELBITS; ++k) if ((code >> k) & 1) o[1 + (bit + k) / 8] |= u8(1u << ((bit + k) % 8));
        }
    }
}
template <class EL, int EMAX, int ELBITS> inline void dec_mx(const u8* in, std::size_t n, float* x) {
    constexpr std::size_t BB = 1 + 32 * ELBITS / 8;
    for (std::size_t b = 0; b < n; b += 32, in += BB) {
        int X = int(in[0]) - 127;
        for (std::size_t i = 0; i < 32 && b + i < n; ++i) {
            u32 code = 0; std::size_t bit = i * ELBITS;
            for (int k = 0; k < ELBITS; ++k) code |= u32((in[1 + (bit + k) / 8] >> ((bit + k) % 8)) & 1) << k;
            float v;
            if constexpr (std::is_same_v<EL, i8>) v = float(i8(u8(code))) / 64.f;
            else v = float(EL::from_bits(u8(code)));
            x[b + i] = std::ldexp(v, X);
        }
    }
}

// NVFP4: blocks of 16 E2M1 elements with an FP8 E4M3 block scale (amax / 6).
// (The optional per-tensor f32 second-level scale is folded to 1 here.)
inline void enc_nvfp4(const float* x, std::size_t n, u8* o) {
    for (std::size_t b = 0; b < n; b += 16, o += 9) {
        std::size_t m = std::min<std::size_t>(16, n - b);
        float amax = 0; for (std::size_t i = 0; i < m; ++i) amax = std::max(amax, std::fabs(x[b + i]));
        fp8_e4m3 sc(amax / 6.f);
        if (float(sc) * 6.f < amax && sc.bits < 0x7e) sc = fp8_e4m3::from_bits(u8(sc.bits + 1));
        o[0] = sc.bits;
        float s = float(sc), is = s > 0 ? 1.f / s : 0.f;
        std::memset(o + 1, 0, 8);
        for (std::size_t i = 0; i < m; ++i) o[1 + i / 2] |= u8(fp4_e2m1(x[b + i] * is).bits << (4 * (i & 1)));
    }
}
inline void dec_nvfp4(const u8* in, std::size_t n, float* x) {
    for (std::size_t b = 0; b < n; b += 16, in += 9) {
        float s = float(fp8_e4m3::from_bits(in[0]));
        for (std::size_t i = 0; i < 16 && b + i < n; ++i) x[b + i] = s * float(fp4_e2m1::from_bits(u8((in[1 + i / 2] >> (4 * (i & 1))) & 0xf)));
    }
}

// Ternary (BitNet b1.58 style): blocks of 64, f16 absmean scale, values in {-1, 0, +1} packed 2 bits each.
inline void enc_tq2(const float* x, std::size_t n, u8* o) {
    for (std::size_t b = 0; b < n; b += 64, o += 18) {
        std::size_t m = std::min<std::size_t>(64, n - b);
        double sa = 0; for (std::size_t i = 0; i < m; ++i) sa += std::fabs(x[b + i]);
        float g0 = float(sa / double(m)), ig = g0 > 0 ? 1.f / g0 : 0.f;
        int q[64] = {};
        double nz = 0, cnt = 0;
        for (std::size_t i = 0; i < m; ++i) {
            q[i] = std::clamp(int(std::lround(x[b + i] * ig)), -1, 1);   // absmean quantisation (BitNet b1.58)
            if (q[i]) { nz += std::fabs(x[b + i]); cnt += 1; }
        }
        put16(o, float(cnt > 0 ? nz / cnt : 0.0));                       // least-squares scale for the chosen signs
        std::memset(o + 2, 0, 16);
        for (std::size_t i = 0; i < m; ++i) o[2 + i / 4] |= u8(u8(q[i] + 1) << (2 * (i & 3)));
    }
}
inline void dec_tq2(const u8* in, std::size_t n, float* x) {
    for (std::size_t b = 0; b < n; b += 64, in += 18) {
        float g = get16(in);
        for (std::size_t i = 0; i < 64 && b + i < n; ++i) x[b + i] = g * float(int((in[2 + i / 4] >> (2 * (i & 3))) & 3) - 1);
    }
}

// Q6_K (GGUF-compatible layout): super-block of 256 = 16 sub-blocks of 16.
// struct { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; }  x = d * scales[s] * (q - 32), q in [0, 63]
inline void dec_q6_k(const u8* in, std::size_t n, float* x) {
    for (std::size_t b = 0; b < n; b += 256, in += 210) {
        const u8* ql = in; const u8* qh = in + 128; const i8* sc = reinterpret_cast<const i8*>(in + 192);
        float d = get16(in + 208), out[256];
        for (int h = 0; h < 2; ++h, ql += 64, qh += 32, sc += 8) {
            float* y = out + 128 * h;
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * sc[is] * q1; y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3; y[l + 96] = d * sc[is + 6] * q4;
            }
        }
        std::copy_n(out, std::min<std::size_t>(256, n - b), x + b);
    }
}
inline void enc_q6_k(const float* x, std::size_t n, u8* o) {
    for (std::size_t b = 0; b < n; b += 256, o += 210) {
        float blk[256] = {};
        std::copy_n(x + b, std::min<std::size_t>(256, n - b), blk);
        float ss[16], amax = 0;
        for (int s = 0; s < 16; ++s) {
            float m = 0; for (int i = 0; i < 16; ++i) m = std::max(m, std::fabs(blk[s * 16 + i]));
            ss[s] = m / 31.f; amax = std::max(amax, ss[s]);
        }
        float d = amax / 127.f; put16(o + 208, d); d = get16(o + 208);
        i8 sc[16]; int q[256];
        for (int s = 0; s < 16; ++s) {
            int v = d ? int(std::lround(ss[s] / d)) : 0;
            if (d && v < 127 && v * d * 31.f < ss[s] * 31.f - 1e-6f) ++v;   // cover the range after rounding
            sc[s] = i8(std::clamp(v, 0, 127));
            float ds = d * sc[s];
            for (int i = 0; i < 16; ++i) q[s * 16 + i] = std::clamp(ds ? int(std::lround(blk[s * 16 + i] / ds)) : 0, -32, 31) + 32;
        }
        std::memcpy(o + 192, sc, 16);
        u8* ql = o; u8* qh = o + 128; std::memset(o, 0, 192);
        for (int h = 0; h < 2; ++h) {
            const int* qq = q + 128 * h; u8* L = ql + 64 * h; u8* H = qh + 32 * h;
            for (int l = 0; l < 32; ++l) {
                L[l] = u8((qq[l] & 0xF) | ((qq[l + 64] & 0xF) << 4));
                L[l + 32] = u8((qq[l + 32] & 0xF) | ((qq[l + 96] & 0xF) << 4));
                H[l] = u8((qq[l] >> 4) | ((qq[l + 32] >> 4) << 2) | ((qq[l + 64] >> 4) << 4) | ((qq[l + 96] >> 4) << 6));
            }
        }
    }
}

// Q5_K (GGUF-compatible layout): super-block of 256 = 8 sub-blocks of 32.
// struct { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; }  x = d*sc*q - dmin*m, q in [0, 31]
inline void dec_q5_k(const u8* in, std::size_t n, float* x) {
    for (std::size_t b = 0; b < n; b += 256, in += 176) {
        float d = get16(in), dmin = get16(in + 2);
        const u8* scales = in + 4; const u8* qh = in + 16; const u8* ql = in + 48;
        float out[256]; int is = 0; u8 u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64, ql += 32, is += 2, u1 <<= 2, u2 <<= 2) {
            u8 sc, m; get_scale_min_k4(is, scales, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) out[j + l] = d1 * float((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) out[j + 32 + l] = d2 * float((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        }
        std::copy_n(out, std::min<std::size_t>(256, n - b), x + b);
    }
}
inline void enc_q5_k(const float* x, std::size_t n, u8* o) {
    for (std::size_t b = 0; b < n; b += 256, o += 176) {
        float blk[256] = {};
        std::copy_n(x + b, std::min<std::size_t>(256, n - b), blk);
        float sc[8], mn[8];
        for (int s = 0; s < 8; ++s) {
            float lo = 0, hi = 0;
            for (int i = 0; i < 32; ++i) { lo = std::min(lo, blk[s * 32 + i]); hi = std::max(hi, blk[s * 32 + i]); }
            sc[s] = (hi - lo) / 31.f; mn[s] = -lo;
        }
        float d = *std::max_element(sc, sc + 8) / 63.f, dmin = *std::max_element(mn, mn + 8) / 63.f;
        put16(o, d); put16(o + 2, dmin); d = get16(o); dmin = get16(o + 2);
        u8 ls[8], lm[8];
        for (int s = 0; s < 8; ++s) {
            ls[s] = u8(std::clamp(d ? int(std::lround(sc[s] / d)) : 0, 0, 63));
            lm[s] = u8(std::clamp(dmin ? int(std::lround(mn[s] / dmin)) : 0, 0, 63));
            if (d && ls[s] < 63 && ls[s] * d * 31.f < (sc[s] * 31.f) - 1e-6f) ++ls[s];
        }
        u8* scales = o + 4; std::memset(scales, 0, 12);
        for (int j = 0; j < 4; ++j) { scales[j] = u8(ls[j] | ((ls[j + 4] >> 4) << 6)); scales[j + 4] = u8(lm[j] | ((lm[j + 4] >> 4) << 6)); }
        for (int j = 4; j < 8; ++j) scales[j + 4] = u8((ls[j] & 0xF) | ((lm[j] & 0xF) << 4));
        u8* qh = o + 16; u8* qs = o + 48; std::memset(qh, 0, 32);
        u8 u1 = 1, u2 = 2;
        for (int j = 0, is = 0; j < 256; j += 64, is += 2, u1 <<= 2, u2 <<= 2) {
            u8 s1, m1, s2, m2; get_scale_min_k4(is, scales, s1, m1); get_scale_min_k4(is + 1, scales, s2, m2);
            float d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
            for (int l = 0; l < 32; ++l) {
                int q1 = std::clamp(d1 ? int(std::lround((blk[j + l] + mm1) / d1)) : 0, 0, 31);
                int q2 = std::clamp(d2 ? int(std::lround((blk[j + 32 + l] + mm2) / d2)) : 0, 0, 31);
                qs[j / 2 + l] = u8((q1 & 0xF) | ((q2 & 0xF) << 4));
                if (q1 & 16) qh[l] |= u1;
                if (q2 & 16) qh[l] |= u2;
            }
        }
    }
}

// ---- GGUF legacy, low-bit K, non-linear and ternary block formats ------------------------------------------------
// The decoders follow the public GGUF block layouts and give the bits of the reference dequantisers (checked
// against gguf-py in verify/golden/gguf_types.py). The encoders are FORGE's own (plain min/max or absmax fits).

// Q4_0 / Q5_0: 32 values, { f16 d; [u8 qh[4];] u8 qs[16] }, x = d * (q - 2^(BITS-1)).
// Q4_1 / Q5_1: 32 values, { f16 d; f16 m; [u8 qh[4];] u8 qs[16] }, x = d * q + m.
// Element j (< 16) is the low nibble of qs[j], element j + 16 the high nibble; with 5 bits, bit i of the little-endian
// u32 qh is the fifth bit of element i.
template <int BITS, bool MIN> inline void dec_q45(const u8* in, std::size_t n, float* x) {
    constexpr std::size_t H = MIN ? 4 : 2, QS = H + (BITS == 5 ? 4 : 0), BB = QS + 16;
    [[maybe_unused]] constexpr int OFF = 1 << (BITS - 1);
    for (std::size_t b = 0; b < n; b += 32, in += BB) {
        float d = get16(in), out[32];
        [[maybe_unused]] float m = MIN ? get16(in + 2) : 0.f;
        u32 qh = BITS == 5 ? u32(in[H]) | u32(in[H + 1]) << 8 | u32(in[H + 2]) << 16 | u32(in[H + 3]) << 24 : 0;
        for (int j = 0; j < 16; ++j) {
            int q0 = int(in[QS + j] & 0xF) | int(((qh >> j) & 1) << 4), q1 = int(in[QS + j] >> 4) | int(((qh >> (j + 16)) & 1) << 4);
            if constexpr (MIN) { out[j] = float(q0) * d + m; out[j + 16] = float(q1) * d + m; }
            else { out[j] = float(q0 - OFF) * d; out[j + 16] = float(q1 - OFF) * d; }
        }
        std::copy_n(out, std::min<std::size_t>(32, n - b), x + b);
    }
}
template <int BITS, bool MIN> inline void enc_q45(const float* x, std::size_t n, u8* o) {
    constexpr std::size_t H = MIN ? 4 : 2, QS = H + (BITS == 5 ? 4 : 0), BB = QS + 16;
    [[maybe_unused]] constexpr int L = (1 << BITS) - 1, OFF = 1 << (BITS - 1);
    for (std::size_t b = 0; b < n; b += 32, o += BB) {
        float blk[32] = {}; std::copy_n(x + b, std::min<std::size_t>(32, n - b), blk);
        int q[32]; std::memset(o, 0, BB);
        if constexpr (MIN) {
            float lo = blk[0], hi = blk[0];
            for (float v : blk) { lo = std::min(lo, v); hi = std::max(hi, v); }
            put16(o, (hi - lo) / float(L)); put16(o + 2, lo);
            float d = get16(o), m = get16(o + 2);
            if (d * float(L) + m < hi) { put16(o, (hi - m) / float(L) * 1.001f + 1e-7f); d = get16(o); }   // cover the range after rounding to f16
            for (int i = 0; i < 32; ++i) q[i] = std::clamp(d > 0 ? int(std::lround((blk[i] - m) / d)) : 0, 0, L);
        } else {
            float mx = 0;   // the largest magnitude, with its sign, maps to -2^(BITS-1) (the one code without a mirror)
            for (float v : blk) if (std::fabs(v) > std::fabs(mx)) mx = v;
            put16(o, mx / float(-OFF)); float d = get16(o);
            for (int i = 0; i < 32; ++i) q[i] = std::clamp(d != 0 ? int(std::lround(blk[i] / d)) + OFF : OFF, 0, L);
        }
        for (int j = 0; j < 16; ++j) {
            o[QS + j] = u8((q[j] & 0xF) | ((q[j + 16] & 0xF) << 4));
            if constexpr (BITS == 5) { if (q[j] & 16) o[H + j / 8] |= u8(1u << (j % 8)); if (q[j + 16] & 16) o[H + 2 + j / 8] |= u8(1u << (j % 8)); }
        }
    }
}

// Q2_K: super-block of 256 = 16 sub-blocks of 16. struct { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin; }
// x = d * (scales[s] & 0xF) * q - dmin * (scales[s] >> 4), q in [0, 3]. Element e of half h (128 values) is bits
// 2*(e/32).. of qs[32*h + e % 32].
inline void dec_q2_k(const u8* in, std::size_t n, float* x) {
    for (std::size_t b = 0; b < n; b += 256, in += 84) {
        float d = get16(in + 80), dmin = get16(in + 82), out[256];
        for (int s = 0; s < 16; ++s) {
            float dl = d * float(in[s] & 0xF), ml = dmin * float(in[s] >> 4);
            const u8* q = in + 16 + (s / 8) * 32 + (s & 1) * 16; int sh = 2 * ((s & 7) / 2);
            for (int l = 0; l < 16; ++l) out[s * 16 + l] = dl * float((q[l] >> sh) & 3) - ml;
        }
        std::copy_n(out, std::min<std::size_t>(256, n - b), x + b);
    }
}
inline void enc_q2_k(const float* x, std::size_t n, u8* o) {
    for (std::size_t b = 0; b < n; b += 256, o += 84) {
        float blk[256] = {}; std::copy_n(x + b, std::min<std::size_t>(256, n - b), blk);
        float sc[16], mn[16], msc = 0, mmn = 0;
        for (int s = 0; s < 16; ++s) {
            float lo = 0, hi = 0;
            for (int i = 0; i < 16; ++i) { lo = std::min(lo, blk[s * 16 + i]); hi = std::max(hi, blk[s * 16 + i]); }
            sc[s] = (hi - lo) / 3.f; mn[s] = -lo; msc = std::max(msc, sc[s]); mmn = std::max(mmn, mn[s]);
        }
        std::memset(o, 0, 80);
        put16(o + 80, msc / 15.f); put16(o + 82, mmn / 15.f);
        float d = get16(o + 80), dmin = get16(o + 82);
        for (int s = 0; s < 16; ++s) {
            int ls = std::clamp(d ? int(std::lround(sc[s] / d)) : 0, 0, 15), lm = std::clamp(dmin ? int(std::lround(mn[s] / dmin)) : 0, 0, 15);
            if (d && ls < 15 && float(ls) * d * 3.f < sc[s] * 3.f - 1e-6f) ++ls;
            o[s] = u8(ls | (lm << 4));
            float dl = d * float(ls), ml = dmin * float(lm);
            u8* q = o + 16 + (s / 8) * 32 + (s & 1) * 16; int sh = 2 * ((s & 7) / 2);
            for (int l = 0; l < 16; ++l) q[l] |= u8(std::clamp(dl ? int(std::lround((blk[s * 16 + l] + ml) / dl)) : 0, 0, 3) << sh);
        }
    }
}

// Q3_K: super-block of 256 = 16 sub-blocks of 16. struct { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d; }
// x = d * (sc[s] - 32) * q, q in [-4, 3]: the low two bits sit in qs as for Q2_K, and bit (4*h + e/32) of
// hmask[e % 32] is set when q >= 0 (q = low - 4 otherwise). The sixteen 6-bit scales: low four bits in the nibbles of
// scales[0..7], high two bits packed four to a byte in scales[8..11].
inline int q3_k_scale(const u8* sc, int s) {
    int lo = s < 8 ? sc[s] & 0xF : sc[s - 8] >> 4;
    return (lo | (((sc[8 + (s & 3)] >> (2 * (s / 4))) & 3) << 4)) - 32;
}
inline void dec_q3_k(const u8* in, std::size_t n, float* x) {
    for (std::size_t b = 0; b < n; b += 256, in += 110) {
        float d = get16(in + 108), out[256];
        for (int s = 0; s < 16; ++s) {
            float dl = d * float(q3_k_scale(in + 96, s));
            int h = s / 8, j = (s & 7) / 2, r0 = (s & 1) * 16; u8 m = u8(1u << (4 * h + j));
            const u8* q = in + 32 + h * 32 + r0; const u8* hm = in + r0;
            for (int l = 0; l < 16; ++l) out[s * 16 + l] = dl * float(int((q[l] >> (2 * j)) & 3) - ((hm[l] & m) ? 0 : 4));
        }
        std::copy_n(out, std::min<std::size_t>(256, n - b), x + b);
    }
}
inline void enc_q3_k(const float* x, std::size_t n, u8* o) {
    for (std::size_t b = 0; b < n; b += 256, o += 110) {
        float blk[256] = {}; std::copy_n(x + b, std::min<std::size_t>(256, n - b), blk);
        float ss[16], big = 0;   // per sub-block step: the largest magnitude maps to -4; the largest step maps to -32
        for (int s = 0; s < 16; ++s) {
            float mx = 0; for (int i = 0; i < 16; ++i) if (std::fabs(blk[s * 16 + i]) > std::fabs(mx)) mx = blk[s * 16 + i];
            ss[s] = mx / -4.f; if (std::fabs(ss[s]) > std::fabs(big)) big = ss[s];
        }
        std::memset(o, 0, 108);
        put16(o + 108, big / -32.f); float d = get16(o + 108);
        for (int s = 0; s < 16; ++s) {
            int ls = std::clamp(d != 0 ? int(std::lround(ss[s] / d)) : 0, -32, 31);
            if (ls == 0 && ss[s] != 0 && d != 0) ls = (ss[s] / d) < 0 ? -1 : 1;
            int u = ls + 32;
            if (s < 8) o[96 + s] |= u8(u & 0xF); else o[96 + s - 8] |= u8((u & 0xF) << 4);
            o[104 + (s & 3)] |= u8((u >> 4) << (2 * (s / 4)));
            float dl = d * float(ls);
            int h = s / 8, j = (s & 7) / 2, r0 = (s & 1) * 16; u8 m = u8(1u << (4 * h + j));
            u8* q = o + 32 + h * 32 + r0; u8* hm = o + r0;
            for (int l = 0; l < 16; ++l) {
                int v = std::clamp(dl != 0 ? int(std::lround(blk[s * 16 + l] / dl)) : 0, -4, 3);
                q[l] |= u8((v & 3) << (2 * j)); if (v >= 0) hm[l] |= m;
            }
        }
    }
}

// IQ4_NL: 32 values, { f16 d; u8 qs[16] }, x = d * K[q] with the 16-entry non-linear codebook K; nibble order as Q4_0.
// IQ4_XS: super-block of 256 = 8 sub-blocks of 32. struct { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; }
// x = d * (ls[s] - 32) * K[q], ls[s] = nibble s of scales_l | bits 2s..2s+1 of scales_h << 4.
inline constexpr i8 iq4_k[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
inline void dec_iq4_nl(const u8* in, std::size_t n, float* x) {
    for (std::size_t b = 0; b < n; b += 32, in += 18) {
        float d = get16(in), out[32];
        for (int j = 0; j < 16; ++j) { out[j] = d * float(iq4_k[in[2 + j] & 0xF]); out[j + 16] = d * float(iq4_k[in[2 + j] >> 4]); }
        std::copy_n(out, std::min<std::size_t>(32, n - b), x + b);
    }
}
inline void dec_iq4_xs(const u8* in, std::size_t n, float* x) {
    for (std::size_t b = 0; b < n; b += 256, in += 136) {
        float d = get16(in), out[256]; u32 sh = u32(in[2]) | u32(in[3]) << 8;
        for (int s = 0; s < 8; ++s) {
            int ls = int((in[4 + s / 2] >> (4 * (s & 1))) & 0xF) | int(((sh >> (2 * s)) & 3) << 4);
            float dl = d * float(ls - 32); const u8* q = in + 8 + 16 * s;
            for (int j = 0; j < 16; ++j) { out[s * 32 + j] = dl * float(iq4_k[q[j] & 0xF]); out[s * 32 + j + 16] = dl * float(iq4_k[q[j] >> 4]); }
        }
        std::copy_n(out, std::min<std::size_t>(256, n - b), x + b);
    }
}
inline int iq4_nearest(float v) {   // index of the codebook value closest to v
    int best = 0; float be = std::fabs(v - float(iq4_k[0]));
    for (int i = 1; i < 16; ++i) { float e = std::fabs(v - float(iq4_k[i])); if (e < be) { be = e; best = i; } }
    return best;
}
// Least-squares step for 32 values: the codebook is asymmetric (-127 .. 113), so both signs are tried.
inline float iq4_fit(const float* x) {
    float amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    if (!(amax > 0)) return 0.f;
    float best = 0; double be = -1;
    const float cand[4] = {amax / 127.f, -amax / 127.f, amax / 113.f, -amax / 113.f};
    for (float d0 : cand) {
        double sxk = 0, skk = 0;
        for (int i = 0; i < 32; ++i) { double k = iq4_k[iq4_nearest(x[i] / d0)]; sxk += double(x[i]) * k; skk += k * k; }
        float d = skk > 0 ? float(sxk / skk) : d0; double e = 0;
        if (d == 0) d = d0;
        for (int i = 0; i < 32; ++i) { double r = double(x[i]) - double(d) * iq4_k[iq4_nearest(x[i] / d)]; e += r * r; }
        if (be < 0 || e < be) { be = e; best = d; }
    }
    return best;
}
inline void iq4_pack(const float* x, float d, u8* qs) {
    for (int j = 0; j < 16; ++j) qs[j] = u8((d != 0 ? iq4_nearest(x[j] / d) : 8) | ((d != 0 ? iq4_nearest(x[j + 16] / d) : 8) << 4));
}
inline void enc_iq4_nl(const float* x, std::size_t n, u8* o) {
    for (std::size_t b = 0; b < n; b += 32, o += 18) {
        float blk[32] = {}; std::copy_n(x + b, std::min<std::size_t>(32, n - b), blk);
        put16(o, iq4_fit(blk)); iq4_pack(blk, get16(o), o + 2);
    }
}
inline void enc_iq4_xs(const float* x, std::size_t n, u8* o) {
    for (std::size_t b = 0; b < n; b += 256, o += 136) {
        float blk[256] = {}; std::copy_n(x + b, std::min<std::size_t>(256, n - b), blk);
        float ds[8], big = 0;
        for (int s = 0; s < 8; ++s) { ds[s] = iq4_fit(blk + 32 * s); if (std::fabs(ds[s]) > std::fabs(big)) big = ds[s]; }
        std::memset(o, 0, 8);
        put16(o, big / -32.f); float d = get16(o); u32 sh = 0;
        for (int s = 0; s < 8; ++s) {
            int ls = std::clamp(d != 0 ? int(std::lround(ds[s] / d)) : 0, -32, 31);
            if (ls == 0 && ds[s] != 0 && d != 0) ls = (ds[s] / d) < 0 ? -1 : 1;
            int u = ls + 32;
            o[4 + s / 2] |= u8((u & 0xF) << (4 * (s & 1))); sh |= u32(u >> 4) << (2 * s);
            iq4_pack(blk + 32 * s, d * float(ls), o + 8 + 16 * s);
        }
        o[2] = u8(sh); o[3] = u8(sh >> 8);
    }
}

// TQ2_0: 256 ternary values, { u8 qs[64]; f16 d; }, x = d * (q - 1); element e of half h (128 values) is bits
// 2*(e/32).. of qs[32*h + e % 32] (the Q2_K order).
// TQ1_0: 256 ternary values in 1.6875 bits each, { u8 qs[48]; u8 qh[4]; f16 d; }. A byte holds five base-3 digits
// t0..t4 as ceil((81 t0 + 27 t1 + 9 t2 + 3 t3 + t4) * 256 / 243); digit n is ((byte * 3^n mod 256) * 3) >> 8.
// qs[0..31] hold elements n*32 + m (byte m, digit n), qs[32..47] elements 160 + n*16 + m, and qh[0..3] elements
// 240 + n*4 + m (digits 0..3 only).
inline int tq1_digit(u8 byte, int n) { constexpr u8 p3[5] = {1, 3, 9, 27, 81}; return int((u32(u8(byte * p3[n])) * 3) >> 8); }
inline void tq1_pos(int i, int& byte, int& digit) {   // element -> (byte in qs|qh, digit)
    if (i < 160) { byte = i % 32; digit = i / 32; }
    else if (i < 240) { byte = 32 + (i - 160) % 16; digit = (i - 160) / 16; }
    else { byte = 48 + (i - 240) % 4; digit = (i - 240) / 4; }
}
inline void dec_tq1_0(const u8* in, std::size_t n, float* x) {
    for (std::size_t b = 0; b < n; b += 256, in += 54) {
        float d = get16(in + 52);
        for (int i = 0; i < 256 && b + std::size_t(i) < n; ++i) { int by, dg; tq1_pos(i, by, dg); x[b + std::size_t(i)] = float(tq1_digit(in[by], dg) - 1) * d; }
    }
}
inline void dec_tq2_0(const u8* in, std::size_t n, float* x) {
    for (std::size_t b = 0; b < n; b += 256, in += 66) {
        float d = get16(in + 64);
        for (std::size_t i = 0; i < 256 && b + i < n; ++i) x[b + i] = float(int((in[i / 128 * 32 + i % 32] >> (2 * (i % 128 / 32))) & 3) - 1) * d;
    }
}
// absmean ternarisation with a least-squares scale (as tq2), over the 256-value block; q in {0, 1, 2}
inline float ternarise(const float* x, std::size_t m, int* q) {
    double sa = 0; for (std::size_t i = 0; i < m; ++i) sa += std::fabs(x[i]);
    float g0 = m ? float(sa / double(m)) : 0.f, ig = g0 > 0 ? 1.f / g0 : 0.f;
    double nz = 0, cnt = 0;
    for (std::size_t i = 0; i < m; ++i) { int t = std::clamp(int(std::lround(x[i] * ig)), -1, 1); q[i] = t + 1; if (t) { nz += std::fabs(x[i]); cnt += 1; } }
    return float(cnt > 0 ? nz / cnt : 0.0);
}
inline void enc_tq2_0(const float* x, std::size_t n, u8* o) {
    for (std::size_t b = 0; b < n; b += 256, o += 66) {
        std::size_t m = std::min<std::size_t>(256, n - b); int q[256]; std::fill_n(q, 256, 1);
        put16(o + 64, ternarise(x + b, m, q)); std::memset(o, 0, 64);
        for (int i = 0; i < 256; ++i) o[i / 128 * 32 + i % 32] |= u8(q[i] << (2 * (i % 128 / 32)));
    }
}
inline void enc_tq1_0(const float* x, std::size_t n, u8* o) {
    for (std::size_t b = 0; b < n; b += 256, o += 54) {
        std::size_t m = std::min<std::size_t>(256, n - b); int q[256]; std::fill_n(q, 256, 1);
        put16(o + 52, ternarise(x + b, m, q));
        int acc[52] = {}; constexpr int w3[5] = {81, 27, 9, 3, 1};
        for (int i = 0; i < 256; ++i) { int by, dg; tq1_pos(i, by, dg); acc[by] += q[i] * w3[dg]; }
        for (int i = 0; i < 52; ++i) o[i] = u8((acc[i] * 256 + 242) / 243);
    }
}
} // namespace detail

inline const std::vector<codec>& all() {
    static const std::vector<codec> c = {
        {"f32", dtype::f32, 1, 4, detail::enc_f32, detail::dec_f32},
        {"f16", dtype::f16, 1, 2, detail::enc_f16, detail::dec_f16},
        {"bf16", dtype::bf16, 1, 2, detail::enc_bf16, detail::dec_bf16},
        {"fp8_e4m3", dtype::fp8_e4m3, 1, 1, detail::enc_e4m3, detail::dec_e4m3},
        {"q8_0", dtype::q8_0, 32, 34, detail::enc_q8_0, detail::dec_q8_0},
        {"q4_k", dtype::q4_k, 256, 144, detail::enc_q4_k, detail::dec_q4_k},
        {"fp8_e5m2", dtype::fp8_e5m2, 1, 1, detail::enc_e5m2, detail::dec_e5m2},
        {"q4_g32", dtype::q4_g32, 32, 20, detail::enc_qg<4>, detail::dec_qg<4>},
        {"q8_g32", dtype::q8_g32, 32, 36, detail::enc_qg<8>, detail::dec_qg<8>},
        {"mxfp8", dtype::mxfp8, 32, 33, detail::enc_mx<fp8_e4m3, 8, 8>, detail::dec_mx<fp8_e4m3, 8, 8>},
        {"mxfp6", dtype::mxfp6, 32, 25, detail::enc_mx<fp6_e2m3, 2, 6>, detail::dec_mx<fp6_e2m3, 2, 6>},
        {"mxfp4", dtype::mxfp4, 32, 17, detail::enc_mx<fp4_e2m1, 2, 4>, detail::dec_mx<fp4_e2m1, 2, 4>},
        {"mxint8", dtype::mxint8, 32, 33, detail::enc_mx<i8, 0, 8>, detail::dec_mx<i8, 0, 8>},
        {"nvfp4", dtype::nvfp4, 16, 9, detail::enc_nvfp4, detail::dec_nvfp4},
        {"tq2", dtype::tq2, 64, 18, detail::enc_tq2, detail::dec_tq2},
        {"q6_k", dtype::q6_k, 256, 210, detail::enc_q6_k, detail::dec_q6_k},
        {"q5_k", dtype::q5_k, 256, 176, detail::enc_q5_k, detail::dec_q5_k},
        {"q4_0", dtype::q4_0, 32, 18, detail::enc_q45<4, false>, detail::dec_q45<4, false>},
        {"q4_1", dtype::q4_1, 32, 20, detail::enc_q45<4, true>, detail::dec_q45<4, true>},
        {"q5_0", dtype::q5_0, 32, 22, detail::enc_q45<5, false>, detail::dec_q45<5, false>},
        {"q5_1", dtype::q5_1, 32, 24, detail::enc_q45<5, true>, detail::dec_q45<5, true>},
        {"q2_k", dtype::q2_k, 256, 84, detail::enc_q2_k, detail::dec_q2_k},
        {"q3_k", dtype::q3_k, 256, 110, detail::enc_q3_k, detail::dec_q3_k},
        {"iq4_nl", dtype::iq4_nl, 32, 18, detail::enc_iq4_nl, detail::dec_iq4_nl},
        {"iq4_xs", dtype::iq4_xs, 256, 136, detail::enc_iq4_xs, detail::dec_iq4_xs},
        {"tq1_0", dtype::tq1_0, 256, 54, detail::enc_tq1_0, detail::dec_tq1_0},
        {"tq2_0", dtype::tq2_0, 256, 66, detail::enc_tq2_0, detail::dec_tq2_0},
    };
    return c;
}
inline const codec* find(std::string_view n) {
    for (auto& c : all()) if (c.name == n) return &c;
    return nullptr;
}
inline const codec* find(dtype t) {
    for (auto& c : all()) if (c.tag == t) return &c;
    return nullptr;
}

} // namespace forge::graph::codecs
