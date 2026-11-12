#pragma once
// Image file decoders and encoders with no dependencies, written for untrusted input: every read is bounds
// checked, every size computation is overflow checked, output size is capped (decode_limits), and malformed
// input raises forge::forge_exception (errc::format_error, or errc::unsupported for valid files that use a
// feature this code does not implement). Nothing here is derived from another library's source.
//
//   decode       PNG  (all colour types and bit depths 1/2/4/8/16, palette, tRNS, Adam7 interlace; CRCs and the
//                      zlib Adler-32 are verified)
//                BMP  (uncompressed 1/4/8-bit palette, 16/24/32-bit, BI_BITFIELDS masks; RLE is unsupported)
//                PNM  (P2/P3 text and P5/P6 binary PGM/PPM, maxval up to 65535; P1/P4 bitmaps and PAM unsupported)
//   encode       PNG (8/16-bit, 1-4 channels, adaptive filters, deflate with fixed Huffman codes or stored
//                blocks, whichever is smaller), BMP (8-bit grey palette, 24-bit, 32-bit with alpha), PGM/PPM
//   building blocks  inflate / zlib_decompress, zlib_compress, crc32, adler32
// JPEG is not implemented.
#include <forge/core/error.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace forge::vision {

// Decoded pixels: c interleaved samples per pixel, rows top to bottom. depth 8 uses u8, depth 16 uses u16
// (native byte order); the other vector is empty. c: 1 grey, 2 grey + alpha, 3 RGB, 4 RGBA.
struct raster {
    std::size_t w = 0, h = 0, c = 0; unsigned depth = 8;
    std::vector<std::uint8_t> u8; std::vector<std::uint16_t> u16;
    std::size_t samples() const { return w * h * c; }
};
struct decode_limits {
    std::size_t max_pixels = std::size_t(1) << 26;   // width * height above this is refused before any allocation
    bool verify_checksums = true;                    // PNG chunk CRCs and the zlib Adler-32. Off only for fuzzing: it lets mutated inputs reach the parser
};

namespace codec {
[[noreturn]] inline void bad(const char* what) { throw forge_exception(errc::format_error, std::string("vision: ") + what); }
[[noreturn]] inline void unsupported(const char* what) { throw forge_exception(errc::unsupported, std::string("vision: unsupported ") + what); }
inline std::size_t mul(std::size_t a, std::size_t b) { if (b && a > std::numeric_limits<std::size_t>::max() / b) bad("size overflow"); return a * b; }
inline std::size_t add(std::size_t a, std::size_t b) { if (a > std::numeric_limits<std::size_t>::max() - b) bad("size overflow"); return a + b; }
inline std::uint32_t be32(const std::uint8_t* p) { return std::uint32_t(p[0]) << 24 | std::uint32_t(p[1]) << 16 | std::uint32_t(p[2]) << 8 | p[3]; }
inline std::uint32_t le32(const std::uint8_t* p) { return std::uint32_t(p[3]) << 24 | std::uint32_t(p[2]) << 16 | std::uint32_t(p[1]) << 8 | p[0]; }
inline std::uint16_t le16(const std::uint8_t* p) { return std::uint16_t(p[1] << 8 | p[0]); }
inline void put_be32(std::vector<std::uint8_t>& o, std::uint32_t v) { o.push_back(std::uint8_t(v >> 24)); o.push_back(std::uint8_t(v >> 16)); o.push_back(std::uint8_t(v >> 8)); o.push_back(std::uint8_t(v)); }
inline void put_le32(std::vector<std::uint8_t>& o, std::uint32_t v) { o.push_back(std::uint8_t(v)); o.push_back(std::uint8_t(v >> 8)); o.push_back(std::uint8_t(v >> 16)); o.push_back(std::uint8_t(v >> 24)); }
inline void put_le16(std::vector<std::uint8_t>& o, std::uint32_t v) { o.push_back(std::uint8_t(v)); o.push_back(std::uint8_t(v >> 8)); }
} // namespace codec

// CRC-32 (IEEE 802.3, reflected, as PNG and zlib use it). Pass the previous result to continue a running CRC.
inline std::uint32_t crc32(const std::uint8_t* p, std::size_t n, std::uint32_t crc = 0) {
    static constexpr std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) { std::uint32_t c = i; for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1; t[i] = c; }
        return t;
    }();
    crc = ~crc;
    for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}
inline std::uint32_t adler32(const std::uint8_t* p, std::size_t n, std::uint32_t adler = 1) {
    std::uint32_t a = adler & 0xffff, b = adler >> 16;
    while (n) {
        std::size_t k = n < 5552 ? n : 5552;   // the largest run for which b cannot overflow 32 bits
        n -= k;
        while (k--) { a += *p++; b += a; }
        a %= 65521; b %= 65521;
    }
    return b << 16 | a;
}

namespace codec {
struct bit_reader {
    const std::uint8_t* p; std::size_t n, pos = 0; std::uint32_t buf = 0; int cnt = 0;
    std::uint32_t bits(int need) {   // need <= 16
        while (cnt < need) { if (pos >= n) bad("deflate stream is truncated"); buf |= std::uint32_t(p[pos++]) << cnt; cnt += 8; }
        const std::uint32_t v = buf & ((std::uint32_t(1) << need) - 1);
        buf >>= need; cnt -= need;
        return v;
    }
    void align() { buf = 0; cnt = 0; }             // drop the rest of the current byte (whole buffered bytes are never held: bits() refills lazily)
    std::size_t consumed() const { return pos; }   // bytes taken from the input, including a partly used one
};
// Canonical Huffman code as counts per length plus symbols in code order.
struct huffman { std::uint16_t count[16]; std::uint16_t symbol[288]; };
// Returns 0 for a complete code, a negative number for an over-subscribed one, positive for an incomplete one.
inline int build(huffman& h, const std::uint8_t* length, int n) {
    for (auto& c : h.count) c = 0;
    for (int s = 0; s < n; ++s) ++h.count[length[s]];
    if (h.count[0] == n) return 0;
    int left = 1;
    for (int len = 1; len <= 15; ++len) { left <<= 1; left -= h.count[len]; if (left < 0) return left; }
    std::uint16_t offs[16]; offs[1] = 0;
    for (int len = 1; len < 15; ++len) offs[len + 1] = std::uint16_t(offs[len] + h.count[len]);
    for (int s = 0; s < n; ++s) if (length[s]) h.symbol[offs[length[s]]++] = std::uint16_t(s);
    return left;
}
inline int decode_symbol(bit_reader& br, const huffman& h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= 15; ++len) {
        code |= int(br.bits(1));
        const int count = h.count[len];
        if (code - count < first) return h.symbol[index + (code - first)];
        index += count; first += count; first <<= 1; code <<= 1;
    }
    bad("invalid Huffman code in deflate stream");
}
} // namespace codec

// Raw DEFLATE (RFC 1951). Appends at most max_out bytes to a fresh buffer; more output is an error (a guard
// against decompression bombs). If consumed is given it receives the number of input bytes used.
inline std::vector<std::uint8_t> inflate(const std::uint8_t* data, std::size_t n, std::size_t max_out, std::size_t* consumed = nullptr, std::size_t reserve = 0) {
    using namespace codec;
    static constexpr std::uint16_t lbase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static constexpr std::uint8_t lext[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    static constexpr std::uint16_t dbase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    static constexpr std::uint8_t dext[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    static constexpr std::uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
    std::vector<std::uint8_t> out; out.reserve(reserve < max_out ? reserve : max_out);
    bit_reader br{data, n};
    huffman lencode, distcode;
    for (bool last = false; !last;) {
        last = br.bits(1) != 0;
        const std::uint32_t type = br.bits(2);
        if (type == 0) {
            br.align();
            if (br.pos > n || n - br.pos < 4) bad("deflate stored block is truncated");
            const std::uint32_t len = le16(data + br.pos), nlen = le16(data + br.pos + 2);
            br.pos += 4;
            if ((len ^ 0xffffu) != nlen) bad("deflate stored block length check failed");
            if (n - br.pos < len) bad("deflate stored block is truncated");
            if (max_out - out.size() < len) bad("decompressed data exceeds the expected size");
            out.insert(out.end(), data + br.pos, data + br.pos + len);
            br.pos += len;
            continue;
        }
        if (type == 3) bad("deflate block type 3");
        if (type == 1) {
            std::uint8_t l[288];
            for (int i = 0; i < 288; ++i) l[i] = std::uint8_t(i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8);
            build(lencode, l, 288);
            for (int i = 0; i < 30; ++i) l[i] = 5;
            build(distcode, l, 30);
        } else {
            const int nlen = int(br.bits(5)) + 257, ndist = int(br.bits(5)) + 1, ncode = int(br.bits(4)) + 4;
            if (nlen > 286 || ndist > 30) bad("deflate dynamic block has too many codes");
            std::uint8_t l[320] = {};
            for (int i = 0; i < ncode; ++i) l[order[i]] = std::uint8_t(br.bits(3));
            if (build(lencode, l, 19) != 0) bad("deflate code-length code is not complete");
            for (int i = 0; i < nlen + ndist;) {
                const int sym = decode_symbol(br, lencode);
                if (sym < 16) { l[i++] = std::uint8_t(sym); continue; }
                int rep, val = 0;
                if (sym == 16) { if (i == 0) bad("deflate repeat with no previous length"); val = l[i - 1]; rep = 3 + int(br.bits(2)); }
                else if (sym == 17) rep = 3 + int(br.bits(3));
                else rep = 11 + int(br.bits(7));
                if (i + rep > nlen + ndist) bad("deflate code lengths overrun");
                while (rep--) l[i++] = std::uint8_t(val);
            }
            if (l[256] == 0) bad("deflate block has no end-of-block code");
            int err = build(lencode, l, nlen);
            if (err < 0 || (err > 0 && nlen - lencode.count[0] != 1)) bad("deflate literal/length code is invalid");
            err = build(distcode, l + nlen, ndist);
            if (err < 0 || (err > 0 && ndist - distcode.count[0] != 1)) bad("deflate distance code is invalid");
        }
        for (;;) {
            int sym = decode_symbol(br, lencode);
            if (sym < 256) { if (out.size() >= max_out) bad("decompressed data exceeds the expected size"); out.push_back(std::uint8_t(sym)); continue; }
            if (sym == 256) break;
            sym -= 257;
            if (sym >= 29) bad("deflate length symbol out of range");
            const std::size_t len = lbase[sym] + br.bits(lext[sym]);
            const int ds = decode_symbol(br, distcode);
            if (ds >= 30) bad("deflate distance symbol out of range");
            const std::size_t dist = dbase[ds] + br.bits(dext[ds]);
            if (dist > out.size()) bad("deflate distance reaches before the start of the output");
            if (max_out - out.size() < len) bad("decompressed data exceeds the expected size");
            for (std::size_t i = 0, from = out.size() - dist; i < len; ++i) out.push_back(out[from + i]);   // may overlap its own output
        }
    }
    if (consumed) *consumed = br.consumed();
    return out;
}
// zlib stream (RFC 1950): header checks, inflate, Adler-32 verified. Bytes after the checksum are ignored.
inline std::vector<std::uint8_t> zlib_decompress(const std::uint8_t* data, std::size_t n, std::size_t max_out, std::size_t reserve = 0, bool verify = true) {
    using namespace codec;
    if (n < 6) bad("zlib stream is too short");
    if ((data[0] & 0x0f) != 8 || (data[0] >> 4) > 7 || ((std::uint32_t(data[0]) << 8 | data[1]) % 31) != 0) bad("zlib header is invalid");
    if (data[1] & 0x20) bad("zlib preset dictionary is not allowed");
    std::size_t used = 0;
    auto out = inflate(data + 2, n - 2, max_out, &used, reserve);
    if (n - 2 - used < 4) bad("zlib checksum is missing");
    if (verify && be32(data + 2 + used) != adler32(out.data(), out.size())) bad("zlib Adler-32 mismatch");
    return out;
}

// zlib stream of the input: one fixed-Huffman block over a greedy LZ77 parse (hash chains, 32 KB window), or
// stored blocks when that is smaller. Any inflate implementation can read it.
inline std::vector<std::uint8_t> zlib_compress(const std::uint8_t* data, std::size_t n) {
    using namespace codec;
    struct writer {
        std::vector<std::uint8_t> o; std::uint32_t buf = 0; int cnt = 0;
        void bits(std::uint32_t v, int k) { buf |= v << cnt; cnt += k; while (cnt >= 8) { o.push_back(std::uint8_t(buf)); buf >>= 8; cnt -= 8; } }   // k <= 16, cnt < 8 on entry
        void code(std::uint32_t c, int len) { std::uint32_t r = 0; for (int i = 0; i < len; ++i) r |= ((c >> i) & 1u) << (len - 1 - i); bits(r, len); }   // Huffman codes go most significant bit first
        void lit(unsigned s) { if (s < 144) code(0x30 + s, 8); else if (s < 256) code(0x190 + (s - 144), 9); else if (s < 280) code(s - 256, 7); else code(0xC0 + (s - 280), 8); }
        void flush() { if (cnt) { o.push_back(std::uint8_t(buf)); buf = 0; cnt = 0; } }
    } w;
    static constexpr std::uint16_t lbase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static constexpr std::uint8_t lext[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    static constexpr std::uint16_t dbase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    static constexpr std::uint8_t dext[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    w.o.push_back(0x78); w.o.push_back(0x01);
    w.bits(1, 1); w.bits(1, 2);   // final block, fixed codes
    constexpr std::size_t HASH = 1 << 15, WIN = 32768, npos = std::size_t(-1);
    std::vector<std::size_t> head(HASH, npos), prev(WIN, npos);
    auto hash = [&](std::size_t i) { return (std::size_t(data[i]) << 10 ^ std::size_t(data[i + 1]) << 5 ^ data[i + 2]) & (HASH - 1); };
    for (std::size_t i = 0; i < n;) {
        std::size_t best = 0, bdist = 0;
        if (i + 3 <= n) {
            const std::size_t maxlen = n - i < 258 ? n - i : 258;
            int chain = 48;
            for (std::size_t c = head[hash(i)]; c != npos && i - c <= WIN && chain-- > 0; c = prev[c % WIN]) {
                std::size_t l = 0; while (l < maxlen && data[c + l] == data[i + l]) ++l;
                if (l > best) { best = l; bdist = i - c; if (l == maxlen) break; }
                if (prev[c % WIN] != npos && prev[c % WIN] >= c) break;   // the slot was reused by a newer position
            }
        }
        const std::size_t step = best >= 3 ? best : 1;
        if (best >= 3) {
            int ls = 28; while (lbase[ls] > best) --ls;
            w.lit(unsigned(257 + ls)); if (lext[ls]) w.bits(std::uint32_t(best - lbase[ls]), lext[ls]);
            int ds = 29; while (dbase[ds] > bdist) --ds;
            w.code(std::uint32_t(ds), 5); if (dext[ds]) w.bits(std::uint32_t(bdist - dbase[ds]), dext[ds]);
        } else w.lit(data[i]);
        for (std::size_t k = 0; k < step; ++k, ++i) if (i + 3 <= n) { const std::size_t hsh = hash(i); prev[i % WIN] = head[hsh]; head[hsh] = i; }
    }
    w.lit(256); w.flush();
    const std::size_t stored = 2 + n + 5 * (n / 65535 + 1);
    if (w.o.size() > stored) {
        w.o.assign({0x78, 0x01});
        std::size_t i = 0;
        do {
            const std::size_t k = n - i < 65535 ? n - i : 65535;
            w.o.push_back(i + k == n ? 1 : 0); put_le16(w.o, std::uint32_t(k)); put_le16(w.o, std::uint32_t(k) ^ 0xffffu);
            w.o.insert(w.o.end(), data + i, data + i + k); i += k;
        } while (i < n);
    }
    put_be32(w.o, adler32(data, n));
    return w.o;
}

// ------------------------------------------------- PNM -------------------------------------------------
inline raster decode_pnm(std::span<const std::uint8_t> in, decode_limits lim = {}) {
    using namespace codec;
    if (in.size() < 3 || in[0] != 'P') bad("not a PNM file");
    const int kind = in[1] - '0';
    if (kind == 1 || kind == 4 || kind == 7) unsupported("PNM variant (P1, P4 and P7 are not read)");
    if (kind != 2 && kind != 3 && kind != 5 && kind != 6) bad("not a PNM file");
    std::size_t pos = 2;
    auto skip_space = [&] {   // whitespace and '#' comments
        for (;;) {
            while (pos < in.size() && (in[pos] == ' ' || in[pos] == '\t' || in[pos] == '\n' || in[pos] == '\r' || in[pos] == '\v' || in[pos] == '\f')) ++pos;
            if (pos < in.size() && in[pos] == '#') { while (pos < in.size() && in[pos] != '\n') ++pos; continue; }
            return;
        }
    };
    auto number = [&](std::size_t cap) {
        skip_space();
        if (pos >= in.size() || in[pos] < '0' || in[pos] > '9') bad("PNM header: expected a number");
        std::size_t v = 0;
        while (pos < in.size() && in[pos] >= '0' && in[pos] <= '9') { v = v * 10 + std::size_t(in[pos++] - '0'); if (v > cap) bad("PNM value out of range"); }
        return v;
    };
    raster r; r.w = number(std::size_t(1) << 30); r.h = number(std::size_t(1) << 30);
    const std::size_t maxval = number(65535);
    if (r.w == 0 || r.h == 0 || maxval == 0) bad("PNM header has a zero dimension or maxval");
    if (mul(r.w, r.h) > lim.max_pixels) bad("image is larger than the decode limit");
    r.c = (kind == 3 || kind == 6) ? 3 : 1; r.depth = maxval < 256 ? 8 : 16;
    const std::size_t count = mul(mul(r.w, r.h), r.c), full = r.depth == 8 ? 255 : 65535;
    auto scale = [&](std::size_t v) { if (v > maxval) bad("PNM sample exceeds maxval"); return (v * full + maxval / 2) / maxval; };
    if (kind >= 5) {
        if (pos >= in.size()) bad("PNM data is missing");
        ++pos;   // the single whitespace byte that ends the header
        const std::size_t bytes = r.depth == 8 ? 1 : 2;
        if (in.size() - pos < mul(count, bytes)) bad("PNM data is truncated");
        if (r.depth == 8) { r.u8.resize(count); for (std::size_t i = 0; i < count; ++i) r.u8[i] = std::uint8_t(scale(in[pos + i])); }
        else { r.u16.resize(count); for (std::size_t i = 0; i < count; ++i) r.u16[i] = std::uint16_t(scale(std::size_t(in[pos + 2 * i]) << 8 | in[pos + 2 * i + 1])); }
    } else {
        if ((in.size() - pos) / 2 + 1 < count) bad("PNM text data is truncated");   // each sample needs a digit and a separator
        if (r.depth == 8) r.u8.resize(count); else r.u16.resize(count);
        for (std::size_t i = 0; i < count; ++i) { const std::size_t v = scale(number(65535)); if (r.depth == 8) r.u8[i] = std::uint8_t(v); else r.u16[i] = std::uint16_t(v); }
    }
    return r;
}
// Binary PGM (c == 1) or PPM (c == 3), maxval 255 or 65535.
inline std::vector<std::uint8_t> encode_pnm(const raster& r) {
    using namespace codec;
    if ((r.c != 1 && r.c != 3) || (r.depth != 8 && r.depth != 16)) throw forge_exception(errc::invalid_argument, "vision: PNM needs 1 or 3 channels at depth 8 or 16");
    if ((r.depth == 8 ? r.u8.size() : r.u16.size()) != r.samples() || r.w == 0 || r.h == 0) throw forge_exception(errc::invalid_argument, "vision: raster size does not match its data");
    std::string hdr = std::string(r.c == 1 ? "P5\n" : "P6\n") + std::to_string(r.w) + " " + std::to_string(r.h) + "\n" + (r.depth == 8 ? "255" : "65535") + "\n";
    std::vector<std::uint8_t> o(hdr.begin(), hdr.end());
    if (r.depth == 8) o.insert(o.end(), r.u8.begin(), r.u8.end());
    else for (auto v : r.u16) { o.push_back(std::uint8_t(v >> 8)); o.push_back(std::uint8_t(v)); }
    return o;
}

// ------------------------------------------------- BMP -------------------------------------------------
inline raster decode_bmp(std::span<const std::uint8_t> in, decode_limits lim = {}) {
    using namespace codec;
    if (in.size() < 26 || in[0] != 'B' || in[1] != 'M') bad("not a BMP file");
    const std::uint8_t* p = in.data();
    const std::size_t data_off = le32(p + 10), hdr = le32(p + 14);
    if (hdr != 12 && hdr != 40 && hdr != 52 && hdr != 56 && hdr != 108 && hdr != 124) bad("BMP header size is not a known version");
    if (in.size() - 14 < hdr) bad("BMP header is truncated");
    std::int64_t w, h; unsigned planes, bpp; std::uint32_t comp = 0, used = 0;
    if (hdr == 12) { w = le16(p + 18); h = le16(p + 20); planes = le16(p + 22); bpp = le16(p + 24); }
    else { w = std::int32_t(le32(p + 18)); h = std::int32_t(le32(p + 22)); planes = le16(p + 26); bpp = le16(p + 28); comp = le32(p + 30); used = le32(p + 46); }
    const bool top_down = h < 0; if (top_down) h = -h;
    if (w <= 0 || h <= 0 || planes != 1) bad("BMP dimensions are invalid");
    if (bpp != 1 && bpp != 4 && bpp != 8 && bpp != 16 && bpp != 24 && bpp != 32) bad("BMP bit count is invalid");
    if (comp == 1 || comp == 2 || comp == 4 || comp == 5) unsupported("BMP compression (RLE / embedded JPEG / PNG)");
    if (comp != 0 && comp != 3 && comp != 6) bad("BMP compression value is invalid");
    if ((comp == 3 || comp == 6) && bpp != 16 && bpp != 32) bad("BMP bit masks need 16 or 32 bits per pixel");
    if (top_down && hdr == 12) bad("BMP core header cannot be top-down");
    raster r; r.w = std::size_t(w); r.h = std::size_t(h);
    if (mul(r.w, r.h) > lim.max_pixels) bad("image is larger than the decode limit");
    // channel masks for 16 / 32 bits per pixel
    std::uint32_t mask[4] = {0, 0, 0, 0};
    if (bpp == 16) { mask[0] = 0x7C00; mask[1] = 0x03E0; mask[2] = 0x001F; }
    if (bpp == 32) { mask[0] = 0x00FF0000; mask[1] = 0x0000FF00; mask[2] = 0x000000FF; }
    std::size_t after_hdr = 14 + hdr;
    if (comp == 3 || comp == 6) {
        const std::size_t nmask = hdr >= 52 ? (hdr >= 56 ? 4 : 3) : (comp == 6 ? 4 : 3);
        if (hdr >= 52) { for (std::size_t i = 0; i < nmask; ++i) mask[i] = le32(p + 54 + 4 * i); }   // masks are part of the header
        else { if (in.size() - after_hdr < 4 * nmask) bad("BMP bit masks are truncated"); for (std::size_t i = 0; i < nmask; ++i) mask[i] = le32(p + after_hdr + 4 * i); after_hdr += 4 * nmask; }
        if (nmask == 3) mask[3] = 0;
        for (int i = 0; i < 3; ++i) if (mask[i] == 0) bad("BMP colour mask is empty");
    }
    int shift[4] = {}; std::uint32_t maxv[4] = {};
    for (int i = 0; i < 4; ++i) if (mask[i]) {
        while (!((mask[i] >> shift[i]) & 1u)) ++shift[i];
        maxv[i] = mask[i] >> shift[i];
        if (maxv[i] & (maxv[i] + 1)) bad("BMP colour mask is not contiguous");
    }
    // palette
    std::vector<std::uint8_t> pal;   // RGB triples
    if (bpp <= 8) {
        const std::size_t esz = hdr == 12 ? 3 : 4, cap = std::size_t(1) << bpp;
        std::size_t count = used ? used : cap;
        if (count > cap) bad("BMP palette is larger than the bit depth allows");
        if (in.size() - after_hdr < count * esz) bad("BMP palette is truncated");
        pal.resize(count * 3);
        for (std::size_t i = 0; i < count; ++i) { const std::uint8_t* e = p + after_hdr + i * esz; pal[3 * i] = e[2]; pal[3 * i + 1] = e[1]; pal[3 * i + 2] = e[0]; }
        after_hdr += count * esz;
    }
    if (data_off < after_hdr || data_off > in.size()) bad("BMP pixel data offset is invalid");
    const std::size_t stride = mul(add(mul(r.w, bpp), 31) / 32, 4);
    if ((in.size() - data_off) / stride < r.h) bad("BMP pixel data is truncated");
    r.c = mask[3] ? 4 : 3; r.depth = 8; r.u8.resize(mul(mul(r.w, r.h), r.c));
    for (std::size_t y = 0; y < r.h; ++y) {
        const std::uint8_t* row = p + data_off + (top_down ? y : r.h - 1 - y) * stride;
        std::uint8_t* out = r.u8.data() + y * r.w * r.c;
        for (std::size_t x = 0; x < r.w; ++x, out += r.c) {
            if (bpp <= 8) {
                const std::size_t idx = bpp == 8 ? row[x] : bpp == 4 ? (row[x / 2] >> (4 * (1 - x % 2))) & 15u : (row[x / 8] >> (7 - x % 8)) & 1u;
                if (idx * 3 >= pal.size()) bad("BMP pixel refers to a colour outside the palette");
                out[0] = pal[3 * idx]; out[1] = pal[3 * idx + 1]; out[2] = pal[3 * idx + 2];
            } else if (bpp == 24) { out[0] = row[3 * x + 2]; out[1] = row[3 * x + 1]; out[2] = row[3 * x]; }
            else {
                const std::uint32_t v = bpp == 16 ? le16(row + 2 * x) : le32(row + 4 * x);
                for (std::size_t k = 0; k < r.c; ++k) { const std::uint32_t q = (v & mask[k]) >> shift[k]; out[k] = std::uint8_t((std::uint64_t(q) * 255 + maxv[k] / 2) / maxv[k]); }
            }
        }
    }
    return r;
}
// 8-bit only. c == 1: 8-bit with a grey palette; c == 3: 24-bit; c == 4: 32-bit with channel masks (alpha kept).
// c == 2 is not representable.
inline std::vector<std::uint8_t> encode_bmp(const raster& r) {
    using namespace codec;
    if (r.depth != 8 || (r.c != 1 && r.c != 3 && r.c != 4) || r.w == 0 || r.h == 0 || r.u8.size() != r.samples()) throw forge_exception(errc::invalid_argument, "vision: BMP needs 8-bit data with 1, 3 or 4 channels");
    if (r.w > 0x7fffffffu || r.h > 0x7fffffffu) throw forge_exception(errc::invalid_argument, "vision: image too large for BMP");
    const std::size_t bpp = r.c * 8, stride = (r.w * bpp + 31) / 32 * 4, hdr = r.c == 4 ? 108 : 40, palbytes = r.c == 1 ? 1024 : 0;
    const std::size_t off = 14 + hdr + palbytes, total = off + stride * r.h;
    if (total > 0xffffffffu) throw forge_exception(errc::invalid_argument, "vision: image too large for BMP");
    std::vector<std::uint8_t> o; o.reserve(total);
    o.push_back('B'); o.push_back('M'); put_le32(o, std::uint32_t(total)); put_le32(o, 0); put_le32(o, std::uint32_t(off));
    put_le32(o, std::uint32_t(hdr)); put_le32(o, std::uint32_t(r.w)); put_le32(o, std::uint32_t(r.h)); put_le16(o, 1); put_le16(o, std::uint32_t(bpp));
    put_le32(o, r.c == 4 ? 3 : 0); put_le32(o, std::uint32_t(stride * r.h)); put_le32(o, 2835); put_le32(o, 2835); put_le32(o, r.c == 1 ? 256 : 0); put_le32(o, 0);
    if (r.c == 4) { put_le32(o, 0x00FF0000); put_le32(o, 0x0000FF00); put_le32(o, 0x000000FF); put_le32(o, 0xFF000000); put_le32(o, 0x73524742 /* 'sRGB' */); o.resize(14 + hdr, 0); }
    if (r.c == 1) for (unsigned i = 0; i < 256; ++i) { o.push_back(std::uint8_t(i)); o.push_back(std::uint8_t(i)); o.push_back(std::uint8_t(i)); o.push_back(0); }
    for (std::size_t y = r.h; y-- > 0;) {
        const std::uint8_t* s = r.u8.data() + y * r.w * r.c; const std::size_t start = o.size();
        for (std::size_t x = 0; x < r.w; ++x, s += r.c) {
            if (r.c == 1) o.push_back(s[0]);
            else { o.push_back(s[2]); o.push_back(s[1]); o.push_back(s[0]); if (r.c == 4) o.push_back(s[3]); }
        }
        o.resize(start + stride, 0);
    }
    return o;
}

// ------------------------------------------------- PNG -------------------------------------------------
namespace codec {
inline int paeth(int a, int b, int c) { const int p = a + b - c, pa = p > a ? p - a : a - p, pb = p > b ? p - b : b - p, pc = p > c ? p - c : c - p; return pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
// Undo the PNG filters of `rows` scanlines of `rowbytes` bytes (each preceded by its filter byte) in place;
// the unfiltered rows end up packed at the start of the buffer region (dst may equal src region start).
inline void unfilter(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows, std::size_t rowbytes, std::size_t bpp) {
    const std::uint8_t* prev = nullptr;
    for (std::size_t y = 0; y < rows; ++y) {
        const std::uint8_t f = src[0]; const std::uint8_t* s = src + 1;
        if (f > 4) bad("PNG filter type is invalid");
        for (std::size_t i = 0; i < rowbytes; ++i) {
            const int a = i >= bpp ? dst[i - bpp] : 0, b = prev ? prev[i] : 0, c = (prev && i >= bpp) ? prev[i - bpp] : 0;
            int v = s[i];
            switch (f) { case 1: v += a; break; case 2: v += b; break; case 3: v += (a + b) >> 1; break; case 4: v += paeth(a, b, c); break; default: break; }
            dst[i] = std::uint8_t(v);
        }
        prev = dst; dst += rowbytes; src += rowbytes + 1;
    }
}
} // namespace codec

inline raster decode_png(std::span<const std::uint8_t> in, decode_limits lim = {}) {
    using namespace codec;
    static constexpr std::uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (in.size() < 8 || std::memcmp(in.data(), sig, 8) != 0) bad("not a PNG file");
    std::size_t pos = 8; bool have_ihdr = false, have_plte = false, have_trns = false, have_idat = false, idat_closed = false, have_iend = false;
    std::uint32_t w = 0, h = 0; unsigned depth = 0, ctype = 0, interlace = 0;
    std::vector<std::uint8_t> idat, plte, trns;
    while (!have_iend) {
        if (in.size() - pos < 12) bad("PNG chunk is truncated");
        const std::uint32_t len = be32(in.data() + pos);
        if (len > 0x7fffffffu || in.size() - pos - 12 < len) bad("PNG chunk length runs past the end of the file");
        const std::uint8_t* type = in.data() + pos + 4; const std::uint8_t* body = type + 4;
        if (lim.verify_checksums && crc32(type, std::size_t(len) + 4) != be32(body + len)) bad("PNG chunk CRC mismatch");
        auto is = [&](const char* t) { return std::memcmp(type, t, 4) == 0; };
        if (!have_ihdr && !is("IHDR")) bad("PNG does not start with IHDR");
        if (is("IHDR")) {
            if (have_ihdr || len != 13) bad("PNG IHDR is invalid");
            have_ihdr = true; w = be32(body); h = be32(body + 4); depth = body[8]; ctype = body[9]; interlace = body[12];
            if (w == 0 || h == 0 || w > 0x7fffffffu || h > 0x7fffffffu) bad("PNG dimensions are invalid");
            if (body[10] != 0 || body[11] != 0 || interlace > 1) bad("PNG compression, filter or interlace method is invalid");
            const bool ok = (ctype == 0 && (depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16)) || (ctype == 3 && (depth == 1 || depth == 2 || depth == 4 || depth == 8)) ||
                            ((ctype == 2 || ctype == 4 || ctype == 6) && (depth == 8 || depth == 16));
            if (!ok) bad("PNG colour type and bit depth do not go together");
            if (mul(w, h) > lim.max_pixels) bad("image is larger than the decode limit");
        } else if (is("PLTE")) {
            if (have_plte || have_idat || len == 0 || len % 3 || len > 768 || ctype == 0 || ctype == 4) bad("PNG PLTE is invalid or misplaced");
            if (ctype == 3 && len / 3 > (std::size_t(1) << depth)) bad("PNG palette has more entries than the bit depth allows");
            have_plte = true; plte.assign(body, body + len);
        } else if (is("tRNS")) {
            if (have_trns || have_idat || ctype == 4 || ctype == 6) bad("PNG tRNS is invalid or misplaced");
            if (ctype == 3 && (!have_plte || len > plte.size() / 3 || len == 0)) bad("PNG tRNS does not match the palette");
            if ((ctype == 0 && len != 2) || (ctype == 2 && len != 6)) bad("PNG tRNS has the wrong length");
            have_trns = true; trns.assign(body, body + len);
        } else if (is("IDAT")) {
            if (idat_closed) bad("PNG IDAT chunks are not consecutive");
            if (ctype == 3 && !have_plte) bad("PNG palette image has no PLTE before IDAT");
            have_idat = true; idat.insert(idat.end(), body, body + len);
        } else if (is("IEND")) {
            if (len != 0) bad("PNG IEND is not empty");
            have_iend = true;
        } else if (!(type[0] & 0x20)) bad("PNG has an unknown critical chunk");
        if (have_idat && !is("IDAT")) idat_closed = true;
        pos += std::size_t(len) + 12;
    }
    if (!have_idat) bad("PNG has no image data");
    const std::size_t in_ch = ctype == 0 ? 1 : ctype == 2 ? 3 : ctype == 3 ? 1 : ctype == 4 ? 2 : 4;
    const std::size_t bits_pp = in_ch * depth, bpp = (bits_pp + 7) / 8;
    auto row_bytes = [&](std::size_t width) { return (mul(width, bits_pp) + 7) / 8; };
    // Adam7 pass geometry: x start, y start, x step, y step
    static constexpr std::uint8_t px0[7] = {0, 4, 0, 2, 0, 1, 0}, py0[7] = {0, 0, 4, 0, 2, 0, 1}, pdx[7] = {8, 8, 4, 4, 2, 2, 1}, pdy[7] = {8, 8, 8, 4, 4, 2, 2};
    std::size_t expect = 0;
    if (!interlace) expect = mul(add(row_bytes(w), 1), h);
    else for (int p = 0; p < 7; ++p) { const std::size_t pw = (w + pdx[p] - 1 - px0[p]) / pdx[p], ph = (h + pdy[p] - 1 - py0[p]) / pdy[p]; if (pw && ph) expect = add(expect, mul(add(row_bytes(pw), 1), ph)); }
    // reserve no more than this input could possibly inflate to (deflate tops out near 1032:1)
    std::vector<std::uint8_t> raw = zlib_decompress(idat.data(), idat.size(), expect, std::min(expect, add(mul(idat.size(), 1032), 64)), lim.verify_checksums);
    if (raw.size() != expect) bad("PNG image data is shorter than the header implies");
    idat.clear(); idat.shrink_to_fit();
    // output layout
    raster r; r.w = w; r.h = h;
    const bool key = have_trns && ctype != 3;
    r.c = ctype == 3 ? (have_trns ? 4 : 3) : in_ch + (key ? 1 : 0);
    r.depth = depth == 16 ? 16 : 8;
    const std::size_t total = mul(mul(r.w, r.h), r.c);
    if (r.depth == 8) r.u8.resize(total); else r.u16.resize(total);
    const unsigned key_g = key && ctype == 0 ? unsigned(trns[0] << 8 | trns[1]) : 0;
    const unsigned key_rgb[3] = {key && ctype == 2 ? unsigned(trns[0] << 8 | trns[1]) : 0u, key && ctype == 2 ? unsigned(trns[2] << 8 | trns[3]) : 0u, key && ctype == 2 ? unsigned(trns[4] << 8 | trns[5]) : 0u};
    // Convert one unfiltered scanline of `width` pixels and write them at (x0 + i * dx, y).
    auto emit = [&](const std::uint8_t* line, std::size_t width, std::size_t y, std::size_t x0, std::size_t dx) {
        for (std::size_t i = 0; i < width; ++i) {
            const std::size_t o = (y * r.w + x0 + i * dx) * r.c;
            if (depth == 16) {
                unsigned v[4];
                for (std::size_t k = 0; k < in_ch; ++k) v[k] = unsigned(line[(i * in_ch + k) * 2] << 8 | line[(i * in_ch + k) * 2 + 1]);
                for (std::size_t k = 0; k < in_ch; ++k) r.u16[o + k] = std::uint16_t(v[k]);
                if (key) r.u16[o + in_ch] = (ctype == 0 ? v[0] == key_g : (v[0] == key_rgb[0] && v[1] == key_rgb[1] && v[2] == key_rgb[2])) ? 0 : 65535;
            } else if (depth == 8 && ctype != 3) {
                for (std::size_t k = 0; k < in_ch; ++k) r.u8[o + k] = line[i * in_ch + k];
                if (key) r.u8[o + in_ch] = (ctype == 0 ? line[i] == key_g : (line[3 * i] == key_rgb[0] && line[3 * i + 1] == key_rgb[1] && line[3 * i + 2] == key_rgb[2])) ? 0 : 255;
            } else {   // one sample of 1, 2, 4 or 8 bits: grey or a palette index
                const unsigned s = depth == 8 ? line[i] : (line[i * depth / 8] >> (8 - depth - (i * depth) % 8)) & ((1u << depth) - 1);
                if (ctype == 3) {
                    if (std::size_t(s) * 3 >= plte.size()) bad("PNG pixel refers to a colour outside the palette");
                    r.u8[o] = plte[3 * s]; r.u8[o + 1] = plte[3 * s + 1]; r.u8[o + 2] = plte[3 * s + 2];
                    if (have_trns) r.u8[o + 3] = s < trns.size() ? trns[s] : 255;
                } else {
                    r.u8[o] = std::uint8_t(s * 255 / ((1u << depth) - 1));
                    if (key) r.u8[o + 1] = s == key_g ? 0 : 255;
                }
            }
        }
    };
    if (!interlace) {
        const std::size_t rb = row_bytes(w);
        unfilter(raw.data(), raw.data(), h, rb, bpp);   // rows move down by their index: the write position never passes the read position
        for (std::size_t y = 0; y < h; ++y) emit(raw.data() + y * rb, w, y, 0, 1);
    } else {
        std::size_t off = 0;
        for (int p = 0; p < 7; ++p) {
            const std::size_t pw = (w + pdx[p] - 1 - px0[p]) / pdx[p], ph = (h + pdy[p] - 1 - py0[p]) / pdy[p];
            if (!pw || !ph) continue;
            const std::size_t rb = row_bytes(pw);
            unfilter(raw.data() + off, raw.data() + off, ph, rb, bpp);
            for (std::size_t y = 0; y < ph; ++y) emit(raw.data() + off + y * rb, pw, py0[p] + y * pdy[p], px0[p], pdx[p]);
            off += (rb + 1) * ph;
        }
    }
    return r;
}
// PNG with colour type from c (1 grey, 2 grey + alpha, 3 RGB, 4 RGBA) at depth 8 or 16, not interlaced.
// Each row uses the filter with the smallest sum of absolute filtered values.
inline std::vector<std::uint8_t> encode_png(const raster& r) {
    using namespace codec;
    if (r.c < 1 || r.c > 4 || (r.depth != 8 && r.depth != 16) || r.w == 0 || r.h == 0 || r.w > 0x7fffffffu || r.h > 0x7fffffffu || (r.depth == 8 ? r.u8.size() : r.u16.size()) != r.samples())
        throw forge_exception(errc::invalid_argument, "vision: PNG needs 1-4 channels at depth 8 or 16 and data matching the size");
    const std::size_t bps = r.depth / 8, bpp = r.c * bps, rb = r.w * bpp;
    std::vector<std::uint8_t> cur(rb), prev(rb, 0), cand[5], raw; raw.reserve((rb + 1) * r.h);
    for (auto& c : cand) c.resize(rb);
    for (std::size_t y = 0; y < r.h; ++y) {
        if (r.depth == 8) std::memcpy(cur.data(), r.u8.data() + y * rb, rb);
        else for (std::size_t i = 0; i < r.w * r.c; ++i) { const std::uint16_t v = r.u16[y * r.w * r.c + i]; cur[2 * i] = std::uint8_t(v >> 8); cur[2 * i + 1] = std::uint8_t(v); }
        std::size_t best = 0, best_sum = std::size_t(-1);
        for (std::size_t f = 0; f < 5; ++f) {
            std::size_t sum = 0;
            for (std::size_t i = 0; i < rb; ++i) {
                const int a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
                const int pred = f == 0 ? 0 : f == 1 ? a : f == 2 ? b : f == 3 ? (a + b) >> 1 : paeth(a, b, c);
                const std::uint8_t v = std::uint8_t(cur[i] - pred); cand[f][i] = v; sum += v < 128 ? v : 256 - v;
            }
            if (sum < best_sum) { best_sum = sum; best = f; }
        }
        raw.push_back(std::uint8_t(best)); raw.insert(raw.end(), cand[best].begin(), cand[best].end());
        prev.swap(cur);
    }
    std::vector<std::uint8_t> o = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    auto chunk = [&](const char* type, const std::uint8_t* body, std::size_t len) {
        put_be32(o, std::uint32_t(len)); const std::size_t at = o.size();
        o.insert(o.end(), type, type + 4); o.insert(o.end(), body, body + len);
        put_be32(o, crc32(o.data() + at, len + 4));
    };
    std::uint8_t ihdr[13] = {}; static constexpr std::uint8_t ct[5] = {0, 0, 4, 2, 6};
    ihdr[0] = std::uint8_t(r.w >> 24); ihdr[1] = std::uint8_t(r.w >> 16); ihdr[2] = std::uint8_t(r.w >> 8); ihdr[3] = std::uint8_t(r.w);
    ihdr[4] = std::uint8_t(r.h >> 24); ihdr[5] = std::uint8_t(r.h >> 16); ihdr[6] = std::uint8_t(r.h >> 8); ihdr[7] = std::uint8_t(r.h);
    ihdr[8] = std::uint8_t(r.depth); ihdr[9] = ct[r.c];
    chunk("IHDR", ihdr, 13);
    const auto z = zlib_compress(raw.data(), raw.size());
    for (std::size_t i = 0; i < z.size() || i == 0; i += 0x40000000u) chunk("IDAT", z.data() + i, z.size() - i < 0x40000000u ? z.size() - i : 0x40000000u);
    chunk("IEND", nullptr, 0);
    return o;
}

// Detects the format from the first bytes and decodes it.
inline raster decode_image(std::span<const std::uint8_t> in, decode_limits lim = {}) {
    if (in.size() >= 8 && in[0] == 0x89 && in[1] == 'P') return decode_png(in, lim);
    if (in.size() >= 2 && in[0] == 'B' && in[1] == 'M') return decode_bmp(in, lim);
    if (in.size() >= 2 && in[0] == 'P' && in[1] >= '1' && in[1] <= '7') return decode_pnm(in, lim);
    if (in.size() >= 2 && in[0] == 0xFF && in[1] == 0xD8) codec::unsupported("format (JPEG decoding is not implemented)");
    codec::bad("unrecognised image format");
}

} // namespace forge::vision
