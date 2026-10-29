#pragma once
// KV-cache codecs. A quantised state slot stores each (row, head) as one float32 scale word followed by
// ceil(d/4) words holding d packed bytes (little-endian, byte e of the head in word e/4, bits 8*(e%4)).
// The slot's physical view is therefore [len, heads, 1 + ceil(d/4)] floats; attention kernels (CPU and
// Vulkan) decode on the fly. int8: symmetric, scale = amax/127. fp8: E4M3 (saturating), scale = amax/448.
#include <forge/core/dtypes.hpp>
#include <forge/core/error.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

namespace forge::flow {

enum class kv_codec : int { f32 = 0, int8 = 1, fp8 = 2 };

inline kv_codec kv_codec_from_name(const std::string& s) {
    if (s.empty() || s == "f32" || s == "none") return kv_codec::f32;
    if (s == "int8" || s == "i8" || s == "q8") return kv_codec::int8;
    if (s == "fp8" || s == "fp8_e4m3" || s == "e4m3") return kv_codec::fp8;
    raise(errc::invalid_argument, "unknown KV codec '{}' (f32, int8, fp8)", s);
}
inline const char* kv_codec_name(kv_codec c) { return c == kv_codec::int8 ? "int8" : c == kv_codec::fp8 ? "fp8" : "f32"; }
inline i64 kv_head_words(i64 d) { return 1 + (d + 3) / 4; }

// Exact E4M3 decode by bit arithmetic (the GLSL kernel uses the same formula, so CPU and GPU agree bitwise).
inline float kv_fp8_decode(std::uint32_t b) {
    std::uint32_t e = (b >> 3) & 15u, m = b & 7u;
    float v = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(float(8u + m), int(e) - 10);
    return (b & 0x80u) ? -v : v;
}

// Encode one head of d floats into kv_head_words(d) words at dst.
inline void kv_encode_head(const float* x, i64 d, kv_codec c, float* dst) {
    float amax = 0; for (i64 e = 0; e < d; ++e) amax = std::max(amax, std::fabs(x[e]));
    float scale = amax > 0 ? amax / (c == kv_codec::int8 ? 127.f : 448.f) : 1.f;
    std::memcpy(dst, &scale, 4);
    auto* w = reinterpret_cast<std::uint32_t*>(dst + 1);
    std::memset(w, 0, std::size_t((d + 3) / 4) * 4);
    float inv = 1.f / scale;
    for (i64 e = 0; e < d; ++e) {
        std::uint32_t byte;
        if (c == kv_codec::int8) byte = std::uint32_t(std::uint8_t(std::int8_t(std::clamp(std::lround(x[e] * inv), -127l, 127l))));
        else { byte = fp8_e4m3(x[e] * inv).bits; if ((byte & 0x7fu) == 0x7fu) byte -= 1; }   // never emit NaN
        w[e / 4] |= byte << (8 * (e % 4));
    }
}
inline void kv_decode_head(const float* src, i64 d, kv_codec c, float* out) {
    float scale; std::memcpy(&scale, src, 4);
    auto* w = reinterpret_cast<const std::uint32_t*>(src + 1);
    for (i64 e = 0; e < d; ++e) {
        std::uint32_t b = (w[e / 4] >> (8 * (e % 4))) & 0xffu;
        out[e] = (c == kv_codec::int8 ? float(std::int8_t(b)) : kv_fp8_decode(b)) * scale;
    }
}

} // namespace forge::flow
