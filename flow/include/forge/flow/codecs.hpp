#pragma once
// Quantisation codecs: block layout + pack + unpack. Verified against dequantise-then-compute.
#include <forge/core/dtypes.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

namespace forge::flow::codecs {

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
} // namespace detail

inline const std::vector<codec>& all() {
    static const std::vector<codec> c = {
        {"f32", dtype::f32, 1, 4, detail::enc_f32, detail::dec_f32},
        {"f16", dtype::f16, 1, 2, detail::enc_f16, detail::dec_f16},
        {"bf16", dtype::bf16, 1, 2, detail::enc_bf16, detail::dec_bf16},
        {"fp8_e4m3", dtype::fp8_e4m3, 1, 1, detail::enc_e4m3, detail::dec_e4m3},
        {"q8_0", dtype::q8_0, 32, 34, detail::enc_q8_0, detail::dec_q8_0},
        {"q4_k", dtype::q4_k, 256, 144, detail::enc_q4_k, detail::dec_q4_k},
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

} // namespace forge::flow::codecs
