#pragma once
// Tier A text kernels: UTF-8 validation, base64 and hex encode/decode.
//   utf8_error      ASCII runs are skipped 64 (AVX-512BW), 32 (AVX2) or 8 (integer) bytes at a time; non-ASCII
//                   sequences are checked one by one (this is not the table-lookup vector validator)
//   hex_encode/decode  branch-free per-byte arithmetic the compiler vectorises
//   base64          scalar table loops, three bytes / four characters per step
// Decoders are strict: they reject anything a matching encoder would not have produced.
#include <forge/simd/algo_search.hpp>
#include <array>

namespace forge::simd::algo {

// Index of the lead byte of the first ill-formed UTF-8 sequence (overlong forms, surrogates U+D800..DFFF,
// code points above U+10FFFF, stray continuation bytes and truncated sequences are ill-formed); n when the
// whole range is valid.
template <Target Tg> std::size_t utf8_error(const std::uint8_t* p, std::size_t n) {
    constexpr int K = impl::kind<Tg>();
    std::size_t i = 0;
    auto cont = [&](std::size_t at, std::uint8_t lo, std::uint8_t hi) { return at < n && p[at] >= lo && p[at] <= hi; };
    while (i < n) {
        if (p[i] < 0x80) {
            if constexpr (K == 2) {
#if defined(__AVX512BW__)
                while (i + 64 <= n && _mm512_movepi8_mask(_mm512_loadu_si512(reinterpret_cast<const void*>(p + i))) == 0) i += 64;
#endif
            } else if constexpr (K == 1) {
#if defined(__AVX2__)
                while (i + 32 <= n && _mm256_movemask_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i))) == 0) i += 32;
#endif
            }
            while (i + 8 <= n && (searchdetail::load64<Tg>(p + i) & 0x8080808080808080ull) == 0) i += 8;
            while (i < n && p[i] < 0x80) ++i;
            continue;
        }
        const std::uint8_t b = p[i];
        std::size_t len = 0;
        if (b >= 0xC2 && b <= 0xDF) len = cont(i + 1, 0x80, 0xBF) ? 2 : 0;
        else if (b == 0xE0) len = cont(i + 1, 0xA0, 0xBF) && cont(i + 2, 0x80, 0xBF) ? 3 : 0;
        else if (b == 0xED) len = cont(i + 1, 0x80, 0x9F) && cont(i + 2, 0x80, 0xBF) ? 3 : 0;
        else if (b >= 0xE1 && b <= 0xEF) len = cont(i + 1, 0x80, 0xBF) && cont(i + 2, 0x80, 0xBF) ? 3 : 0;
        else if (b == 0xF0) len = cont(i + 1, 0x90, 0xBF) && cont(i + 2, 0x80, 0xBF) && cont(i + 3, 0x80, 0xBF) ? 4 : 0;
        else if (b >= 0xF1 && b <= 0xF3) len = cont(i + 1, 0x80, 0xBF) && cont(i + 2, 0x80, 0xBF) && cont(i + 3, 0x80, 0xBF) ? 4 : 0;
        else if (b == 0xF4) len = cont(i + 1, 0x80, 0x8F) && cont(i + 2, 0x80, 0xBF) && cont(i + 3, 0x80, 0xBF) ? 4 : 0;
        if (!len) return i;
        i += len;
    }
    return n;
}
template <Target Tg> bool utf8_valid(const std::uint8_t* p, std::size_t n) { return utf8_error<Tg>(p, n) == n; }

// ---- hex ----
// out receives 2 * n characters (no terminator).
template <Target Tg> void hex_encode(const std::uint8_t* in, std::size_t n, char* out, bool upper) {
    const int alpha = (upper ? 'A' : 'a') - 10 - '0';
    for (std::size_t i = 0; i < n; ++i) {
        int hi = in[i] >> 4, lo = in[i] & 15;
        out[2 * i] = char('0' + hi + (hi > 9 ? alpha : 0));
        out[2 * i + 1] = char('0' + lo + (lo > 9 ? alpha : 0));
    }
}
// n characters (n even) to n / 2 bytes; accepts both letter cases. False (out contents unspecified) when n
// is odd or a character is not a hex digit.
template <Target Tg> bool hex_decode(const char* in, std::size_t n, std::uint8_t* out) {
    if (n & 1) return false;
    unsigned bad = 0;
    auto nib = [](unsigned c, unsigned& err) {
        unsigned d = c - '0', l = (c | 0x20) - 'a';
        unsigned v = d <= 9 ? d : l + 10;
        err |= unsigned(!(d <= 9 || l <= 5));
        return v;
    };
    for (std::size_t i = 0; i < n / 2; ++i) {
        unsigned hi = nib(static_cast<unsigned char>(in[2 * i]), bad), lo = nib(static_cast<unsigned char>(in[2 * i + 1]), bad);
        out[i] = std::uint8_t((hi << 4) | (lo & 15));
    }
    return bad == 0;
}

// ---- base64 (RFC 4648) ----
// url = false: alphabet "+/", output padded with '=' to a multiple of four.
// url = true:  alphabet "-_", no padding.
constexpr std::size_t base64_encoded_size(std::size_t n, bool url) { return url ? (n * 4 + 2) / 3 : (n + 2) / 3 * 4; }
constexpr std::size_t base64_decoded_max(std::size_t n) { return n / 4 * 3 + 3; }
namespace textdetail {
template <Target Tg> struct b64 {
    static constexpr std::array<char, 64> enc(bool url) {
        std::array<char, 64> t{};
        for (int i = 0; i < 26; ++i) { t[std::size_t(i)] = char('A' + i); t[std::size_t(26 + i)] = char('a' + i); }
        for (int i = 0; i < 10; ++i) t[std::size_t(52 + i)] = char('0' + i);
        t[62] = url ? '-' : '+'; t[63] = url ? '_' : '/';
        return t;
    }
    static constexpr std::array<std::int8_t, 256> dec(bool url) {
        std::array<std::int8_t, 256> t{};
        for (auto& v : t) v = -1;
        auto e = enc(url);
        for (int i = 0; i < 64; ++i) t[static_cast<unsigned char>(e[std::size_t(i)])] = std::int8_t(i);
        return t;
    }
    static constexpr std::array<char, 64> enc_std = enc(false), enc_url = enc(true);
    static constexpr std::array<std::int8_t, 256> dec_std = dec(false), dec_url = dec(true);
};
} // namespace textdetail
// Writes base64_encoded_size(n, url) characters (no terminator) and returns that count.
template <Target Tg> std::size_t base64_encode(const std::uint8_t* in, std::size_t n, char* out, bool url) {
    const auto& e = url ? textdetail::b64<Tg>::enc_url : textdetail::b64<Tg>::enc_std;
    std::size_t i = 0, o = 0;
    for (; i + 3 <= n; i += 3, o += 4) {
        std::uint32_t v = std::uint32_t(in[i]) << 16 | std::uint32_t(in[i + 1]) << 8 | in[i + 2];
        out[o] = e[v >> 18]; out[o + 1] = e[(v >> 12) & 63]; out[o + 2] = e[(v >> 6) & 63]; out[o + 3] = e[v & 63];
    }
    if (n - i == 1) {
        std::uint32_t v = std::uint32_t(in[i]) << 16;
        out[o++] = e[v >> 18]; out[o++] = e[(v >> 12) & 63];
        if (!url) { out[o++] = '='; out[o++] = '='; }
    } else if (n - i == 2) {
        std::uint32_t v = std::uint32_t(in[i]) << 16 | std::uint32_t(in[i + 1]) << 8;
        out[o++] = e[v >> 18]; out[o++] = e[(v >> 12) & 63]; out[o++] = e[(v >> 6) & 63];
        if (!url) out[o++] = '=';
    }
    return o;
}
// Decodes n characters into out (room for base64_decoded_max(n) bytes). Returns the byte count, or -1 for
// input base64_encode(url) could not have produced: a character outside the alphabet (whitespace included),
// wrong or missing padding (url = false), any padding (url = true), an impossible length, or non-zero unused
// bits in the last character.
template <Target Tg> std::ptrdiff_t base64_decode(const char* in, std::size_t n, std::uint8_t* out, bool url) {
    const auto& d = url ? textdetail::b64<Tg>::dec_url : textdetail::b64<Tg>::dec_std;
    if (!url) {
        if (n % 4 != 0) return -1;
        if (n && in[n - 1] == '=') { --n; if (in[n - 1] == '=') --n; }   // n was >= 4, so n - 1 is valid
    }
    if (n % 4 == 1) return -1;
    std::size_t i = 0, o = 0; int bad = 0;
    auto at = [&](std::size_t k) { return int(d[static_cast<unsigned char>(in[k])]); };
    for (; i + 4 <= n; i += 4, o += 3) {
        int a = at(i), b = at(i + 1), c = at(i + 2), e = at(i + 3);
        bad |= a | b | c | e;   // any -1 sets the sign bit
        std::uint32_t v = std::uint32_t(a & 63) << 18 | std::uint32_t(b & 63) << 12 | std::uint32_t(c & 63) << 6 | std::uint32_t(e & 63);
        out[o] = std::uint8_t(v >> 16); out[o + 1] = std::uint8_t(v >> 8); out[o + 2] = std::uint8_t(v);
    }
    if (bad < 0) return -1;
    if (n - i == 2) {
        int a = at(i), b = at(i + 1);
        if ((a | b) < 0 || (b & 15)) return -1;
        out[o++] = std::uint8_t(a << 2 | b >> 4);
    } else if (n - i == 3) {
        int a = at(i), b = at(i + 1), c = at(i + 2);
        if ((a | b | c) < 0 || (c & 3)) return -1;
        out[o++] = std::uint8_t(a << 2 | b >> 4); out[o++] = std::uint8_t((b & 15) << 4 | c >> 2);
    }
    return std::ptrdiff_t(o);
}

} // namespace forge::simd::algo
