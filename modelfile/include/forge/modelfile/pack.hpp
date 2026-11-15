#pragma once
// .fpkz: lossless, entropy-coded storage of a .fpk model file (`forge pack` / `forge unpack`; docs/format.md "Packed
// files"). Unpacking gives back the original file byte for byte, checked against the BLAKE3 of the original that the
// packed file records.
//
// The file is cut into segments that tile it: one or more per tensor (whole rows, ~8 MiB each, so segments are coded
// on all cores) and one per stretch of bytes between tensors (header, sections, padding, section table). Each segment
// is coded on its own:
//   * The segment's codec gives a byte layout: a period (4 for f32, 2 for f16 / bf16, the block size for block codecs)
//     and a class per byte position (sign / exponent byte, mantissa bytes, block scale, quantised values, 2:4
//     positions...). Multi-byte fields are coded most significant byte first, so the sign / exponent byte is known
//     when the mantissa bytes that depend on it are coded.
//   * Every class gets its own coding, chosen by the encoder by two-part code length (minimum description length:
//     entropy under a context plus the cost of learning or storing the model), and recorded in the segment:
//       - context mixing (PAQ / lpaq style): an adaptive binary arithmetic coder whose bit predictions come from up
//         to four models (tables of probabilities with hit counts, adaptation rate 1/n) - order 0 in the class, the
//         previous byte, the byte one period back (the same field of the previous element / block), the byte one row
//         back (the same column of the previous row) - mixed by a logistic mixer (online gradient descent on coding
//         cost) whose weights are selected by class and partial byte. Only the models that pay for themselves on this
//         class are mixed in: block scales (previous block / row), the third f32 byte;
//       - static rANS of order 0 (one stored histogram) for bytes no context predicts better (quantised values,
//         2:4 positions, exponent bytes), or of order 1 on the previous byte (the mantissa byte after a float's
//         exponent byte, whose distribution depends on it): as short as the adaptive coder there and many times faster;
//       - stored, when nothing gets the class below ~8 bits per byte (low mantissa bytes of trained f32 weights);
//       - one value (the zero low halves of f32 weights that were bf16).
//     A segment that does not get smaller is stored as is.
// Both sides start every segment from the same deterministic model state and update it identically, so no adaptive
// model is transmitted (the "shared predictor" idea of the Izaac notes, without any secret: the shared state is code).
//
// Layout (little endian):
//   header 96 B : "FORGEPKZ" | u32 version (1) | u32 n_layouts | u32 n_segments | u32 reserved | u64 original size
//                 | u64 index offset | u64 index bytes | 32 B BLAKE3 of the original file | 8 B reserved
//   payloads    : one per segment, in file order
//   index       : n_layouts x { u16 period; u8 n_classes; u8 pad; u8 order[period]; u8 class[period] }
//                 n_segments x { u64 offset; u64 bytes; u64 payload offset; u64 payload bytes; u32 layout;
//                                u32 row bytes; u8 method (0 stored, 1 modelled); 7 B pad }   (48 B each)
//   modelled payload: u8 n_classes | n_classes x u8 mode | n_classes x u8 constant
//                     | per rANS class: order 0: 256 x u16 frequencies (sum 2^15); order 1: u16 n, n x (u8 context,
//                       256 x u16 frequencies, sum 2^12)
//                     | u64 n | n stored bytes (coding order) | u64 n | n rANS bytes | arithmetic-coded bytes
//     mode: 1..15 context mixing (bit q set: model q mixed in), 16 stored, 17 constant, 18 rANS order 0, 19 rANS order 1
#include <forge/core/error.hpp>
#include <forge/core/mapped.hpp>
#include <forge/graph/codecs.hpp>
#include <forge/hash/blake3.hpp>
#include <forge/json/json.hpp>
#include <forge/security/enforce.hpp>
#include <forge/task/engine.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace forge::modelfile::pack {

inline constexpr char magic[8] = {'F', 'O', 'R', 'G', 'E', 'P', 'K', 'Z'};
inline constexpr std::uint32_t version = 1;
inline constexpr std::size_t header_bytes = 96, segment_record = 48;
inline constexpr std::string_view ext = ".fpkz";

// ---- byte layouts ----
struct layout {
    std::uint16_t period = 1;
    std::uint8_t ncls = 1;
    std::array<std::uint8_t, 256> order{};   // coding position j of a period codes byte order[j] of it
    std::array<std::uint8_t, 256> cls{};     // class of coding position j
    bool operator==(const layout& o) const { return period == o.period && ncls == o.ncls && order == o.order && cls == o.cls; }
};
inline constexpr int max_classes = 8;
namespace detail {
struct field { int size; int cls; };   // cls: class of the field's most significant byte, the next ones cls + 1, ...
inline layout from_fields(std::initializer_list<std::pair<field, int>> fs) {   // (field, repeat)
    layout L; int at = 0, j = 0, nc = 0;
    for (auto& [f, rep] : fs)
        for (int r = 0; r < rep; ++r) {
            for (int b = 0; b < f.size; ++b) { L.order[std::size_t(j)] = std::uint8_t(at + f.size - 1 - b); L.cls[std::size_t(j)] = std::uint8_t(f.cls + b); ++j; nc = std::max(nc, f.cls + b + 1); }
            at += f.size;
        }
    L.period = std::uint16_t(at); L.ncls = std::uint8_t(nc);
    return L;
}
}
// The layout of a tensor stored with codec `name` (an unknown codec: one class, bytes in order).
inline layout layout_for(std::string_view name) {
    using detail::field; using detail::from_fields;
    if (name == "f32") return from_fields({{{4, 0}, 1}});
    if (name == "f16" || name == "bf16") return from_fields({{{2, 0}, 1}});
    if (name == "q8_0") return from_fields({{{2, 0}, 1}, {{1, 2}, 32}});
    if (name == "q4_0" || name == "iq4_nl") return from_fields({{{2, 0}, 1}, {{1, 2}, 16}});
    if (name == "q4_1") return from_fields({{{2, 0}, 1}, {{2, 2}, 1}, {{1, 4}, 16}});
    if (name == "q5_0") return from_fields({{{2, 0}, 1}, {{1, 2}, 4}, {{1, 3}, 16}});
    if (name == "q5_1") return from_fields({{{2, 0}, 1}, {{2, 2}, 1}, {{1, 4}, 4}, {{1, 5}, 16}});
    if (name == "s24_f16") return from_fields({{{2, 0}, 16}, {{1, 2}, 4}});
    if (name == "s24_q8") return from_fields({{{2, 0}, 1}, {{1, 2}, 16}, {{1, 3}, 4}});
    if (name == "s24_q4") return from_fields({{{2, 0}, 1}, {{1, 2}, 8}, {{1, 3}, 4}});
    if (name == "q8_g32" || name == "q4_g32") {   // f16 scale, f16 min, values
        const auto* c = graph::codecs::find(name);
        return from_fields({{{2, 0}, 1}, {{2, 2}, 1}, {{1, 4}, int(c->block_bytes) - 4}});
    }
    if (const auto* c = graph::codecs::find(name); c && c->block > 1 && c->block_bytes <= 256) {   // other block codecs
        layout L; L.period = std::uint16_t(c->block_bytes); L.ncls = std::uint8_t(std::min<std::size_t>(c->block_bytes, max_classes));
        for (std::size_t j = 0; j < c->block_bytes; ++j) { L.order[j] = std::uint8_t(j); L.cls[j] = std::uint8_t(j < 4 ? j : 4 + (j & 3)); }
        if (c->block_bytes < 8) for (std::size_t j = 0; j < c->block_bytes; ++j) L.cls[j] = std::uint8_t(j);
        return L;
    }
    if (const auto* c = graph::codecs::find(name); c && c->block == 1 && c->block_bytes == 1) return layout{};
    return layout{};
}

// ---- context-mixing coder ----
namespace cm {
struct tables {
    std::array<std::int16_t, 4096> stretch{};
    std::array<std::int16_t, 4096> squash{};   // index d + 2048, d in [-2048, 2047]
    std::array<std::int32_t, 1024> dt{};       // adaptation rate of a counter seen n times
    tables() {
        for (int i = 0; i < 4096; ++i) {
            double d = (i - 2048) / 256.0;
            squash[std::size_t(i)] = std::int16_t(std::clamp(int(std::lround(4096.0 / (1.0 + std::exp(-d)))), 1, 4095));
        }
        int pi = 0;   // stretch = inverse of squash
        for (int x = -2047; x <= 2047; ++x) {
            int v = squash[std::size_t(x + 2048)];
            for (int j = pi; j <= v; ++j) stretch[std::size_t(j)] = std::int16_t(x);
            pi = v + 1;
        }
        for (int j = pi; j < 4096; ++j) stretch[std::size_t(j)] = 2047;
        for (int i = 0; i < 1024; ++i) dt[std::size_t(i)] = 16384 / (i + i + 3);
    }
};
inline const tables& T() { static const tables t; return t; }
inline int squash(int d) { return T().squash[std::size_t(std::clamp(d, -2047, 2047) + 2048)]; }

inline constexpr int n_models = 4, n_inputs = n_models + 1;
inline constexpr std::uint32_t counter_limit = 1023;
// A counter: 22-bit probability of a 1 in the high bits, hit count in the low 10.
inline void update(std::uint32_t& e, int y, const std::int32_t* dt) {
    const std::uint32_t n = e & 1023, p = e >> 10;
    if (n < counter_limit) ++e; else e = (e & 0xfffffc00u) | counter_limit;
    const std::int64_t d = (std::int64_t((std::uint32_t(y) << 22)) - std::int64_t(p)) >> 3;
    e += std::uint32_t(d * dt[n]) & 0xfffffc00u;
}

// Model ids: 0 order 0 in the class, 1 previous byte, 2 byte one period back, 3 byte one row back.
// A class's mode: 1..15 a mask of the models mixed for it; mode_raw stored; mode_const one value.
inline constexpr std::uint8_t mode_raw = 16, mode_const = 17;
inline constexpr int mixer_lr = 6;
struct model {
    std::array<std::vector<std::uint32_t>, n_models> t;   // t[0]: [class][256], the others [class][context byte][256]
    std::vector<std::int32_t> w;                          // [class][c0][n_inputs]
    void reset(int nc, const std::uint8_t* mode) {
        int used = 0;
        for (int c = 0; c < nc; ++c) if (mode[c] >= 1 && mode[c] <= 15) used |= mode[c];
        for (int q = 0; q < n_models; ++q) {
            if (!(used >> q & 1)) continue;
            const std::size_t sz = q == 0 ? std::size_t(nc) << 8 : std::size_t(nc) << 16;
            if (t[std::size_t(q)].size() < sz) t[std::size_t(q)].resize(sz);
            for (int c = 0; c < nc; ++c) if (mode[c] >= 1 && mode[c] <= 15 && (mode[c] >> q & 1)) {   // only the classes that use it
                const std::size_t per = q == 0 ? 256 : 65536;
                std::fill_n(t[std::size_t(q)].begin() + std::ptrdiff_t(per * std::size_t(c)), per, 1u << 31);
            }
        }
        w.resize((std::size_t(nc) << 8) * n_inputs);
        for (int c = 0; c < nc; ++c) {
            const int nm = std::popcount(unsigned(mode[c] >= 1 && mode[c] <= 15 ? mode[c] : 0));
            for (std::size_t i = std::size_t(c) << 8; i < (std::size_t(c) + 1) << 8; ++i) {
                for (int q = 0; q < n_inputs; ++q) w[i * n_inputs + std::size_t(q)] = q < nm ? 65536 / std::max(nm, 1) : 0;
            }
        }
    }
};
inline model& thread_model() { thread_local model m; return m; }

struct encoder {
    std::uint32_t x1 = 0, x2 = 0xffffffffu;
    std::vector<std::uint8_t>* out;
    void code(int y, int p) {   // p: probability of a 1, 12 bits
        const std::uint32_t xm = x1 + ((x2 - x1) >> 12) * std::uint32_t(p);
        if (y) x2 = xm; else x1 = xm + 1;
        while (((x1 ^ x2) & 0xff000000u) == 0) { out->push_back(std::uint8_t(x2 >> 24)); x1 <<= 8; x2 = (x2 << 8) | 255; }
    }
    void flush() { for (int i = 0; i < 4; ++i) { out->push_back(std::uint8_t(x1 >> 24)); x1 <<= 8; } }
};
struct decoder {
    std::uint32_t x1 = 0, x2 = 0xffffffffu, x = 0;
    const std::uint8_t *p, *end;
    decoder(const std::uint8_t* b, const std::uint8_t* e) : p(b), end(e) { for (int i = 0; i < 4; ++i) x = (x << 8) | next(); }
    std::uint32_t next() { return p < end ? *p++ : 0; }   // past the end: zeros (a corrupt stream decodes to garbage, caught by the hash)
    int code(int /*unused*/, int pr) {
        const std::uint32_t xm = x1 + ((x2 - x1) >> 12) * std::uint32_t(pr);
        int y = x <= xm;
        if (y) x2 = xm; else x1 = xm + 1;
        while (((x1 ^ x2) & 0xff000000u) == 0) { x1 <<= 8; x2 = (x2 << 8) | 255; x = (x << 8) | next(); }
        return y;
    }
};

// One byte through the NM models at s[0..NM) (NM == 1: the order-0 counter alone, no mixer).
template <bool Enc, int NM, class Coder>
inline std::uint32_t code_byte(std::uint32_t* const* s, std::int32_t* W, std::uint32_t c, Coder& ac, const std::int16_t* st_t, const std::int32_t* dt) {
    std::uint32_t c0 = 1;
    for (int k = 7; k >= 0; --k) {
        int pr, y;
        [[maybe_unused]] int x[NM];
        [[maybe_unused]] std::int32_t* w = W + c0 * n_inputs;
        if constexpr (NM == 1) pr = std::clamp(int(s[0][c0] >> 20), 1, 4095);
        else {
            std::int64_t dot = std::int64_t(w[NM]) * 256;
            for (int q = 0; q < NM; ++q) { x[q] = st_t[s[q][c0] >> 20]; dot += std::int64_t(w[q]) * x[q]; }
            pr = std::clamp(squash(int(dot >> 16)), 1, 4095);
        }
        if constexpr (Enc) { y = int(c >> k) & 1; ac.code(y, pr); } else y = ac.code(0, pr);
        if constexpr (NM > 1) {
            const int err = ((y << 12) - pr) * mixer_lr;
            constexpr std::int32_t wmax = 1 << 24;   // weights stay within +-256.0: no overflow whatever the data
            for (int q = 0; q < NM; ++q) w[q] = std::clamp(w[q] + ((x[q] * err) >> 10), -wmax, wmax);
            w[NM] = std::clamp(w[NM] + ((256 * err) >> 10), -wmax, wmax);
        }
        for (int q = 0; q < NM; ++q) update(s[q][c0], y, dt);
        c0 = (c0 << 1) | std::uint32_t(y);
    }
    return c0 & 255;
}
// Codes (Enc) or decodes the modelled classes of `buf` [n bytes, coding order]. Raw / constant classes must already be
// in place in `buf` (the decoder fills them before calling).
template <bool Enc, class Coder>
inline void run(model& m, const layout& L, const std::uint8_t* mode, std::size_t row, std::uint8_t* buf, std::size_t n, Coder& ac) {
    const auto& tb = T();
    const std::int16_t* st_t = tb.stretch.data(); const std::int32_t* dt = tb.dt.data();
    const std::size_t P = L.period;
    const std::size_t d2 = P > 1 ? P : 2;
    const std::size_t d3 = row && row % P == 0 ? row : P > 1 ? 2 * P : 3;
    std::uint32_t* t[n_models];
    for (int q = 0; q < n_models; ++q) t[q] = m.t[std::size_t(q)].data();
    std::size_t j = 0;   // position in the period
    for (std::size_t i = 0; i < n; ++i, j = j + 1 == P ? 0 : j + 1) {
        const int cl = L.cls[j];
        const unsigned md = mode[cl];
        if (md == 0 || md > 15) continue;
        const std::size_t c8 = std::size_t(cl) << 8;
        std::uint32_t* s[n_models]; int nm = 0;
        if (md & 1) s[nm++] = t[0] + c8;
        if (md & 2) s[nm++] = t[1] + ((c8 + (i ? buf[i - 1] : 0)) << 8);
        if (md & 4) s[nm++] = t[2] + ((c8 + (i >= d2 ? buf[i - d2] : 0)) << 8);
        if (md & 8) s[nm++] = t[3] + ((c8 + (i >= d3 ? buf[i - d3] : 0)) << 8);
        std::int32_t* W = m.w.data() + c8 * n_inputs;
        const std::uint32_t c = Enc ? buf[i] : 0;
        std::uint32_t r;
        switch (nm) {
            case 1: r = code_byte<Enc, 1>(s, W, c, ac, st_t, dt); break;
            case 2: r = code_byte<Enc, 2>(s, W, c, ac, st_t, dt); break;
            case 3: r = code_byte<Enc, 3>(s, W, c, ac, st_t, dt); break;
            default: r = code_byte<Enc, 4>(s, W, c, ac, st_t, dt); break;
        }
        if constexpr (!Enc) buf[i] = std::uint8_t(r);
    }
}
} // namespace cm

// ---- one segment ----
namespace detail {
inline void put64(std::vector<std::uint8_t>& o, std::uint64_t v) { for (int i = 0; i < 8; ++i) o.push_back(std::uint8_t(v >> (8 * i))); }
inline std::uint64_t get64(const std::uint8_t* p) { std::uint64_t v = 0; for (int i = 7; i >= 0; --i) v = (v << 8) | p[i]; return v; }
inline std::uint32_t get32(const std::uint8_t* p) { return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8 | std::uint32_t(p[2]) << 16 | std::uint32_t(p[3]) << 24; }
inline void set64(std::uint8_t* p, std::uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = std::uint8_t(v >> (8 * i)); }
inline void set32(std::uint8_t* p, std::uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = std::uint8_t(v >> (8 * i)); }
// bytes of `src` in coding order (n a multiple of the period; else the layout is not used)
inline void to_coding_order(const layout& L, const std::uint8_t* src, std::size_t n, std::uint8_t* dst) {
    const std::size_t P = L.period;
    if (P == 1) { std::memcpy(dst, src, n); return; }
    for (std::size_t q = 0; q < n; q += P) for (std::size_t j = 0; j < P; ++j) dst[q + j] = src[q + L.order[j]];
}
inline void from_coding_order(const layout& L, const std::uint8_t* src, std::size_t n, std::uint8_t* dst) {
    const std::size_t P = L.period;
    if (P == 1) { std::memcpy(dst, src, n); return; }
    for (std::size_t q = 0; q < n; q += P) for (std::size_t j = 0; j < P; ++j) dst[q + L.order[j]] = src[q + j];
}
}

// Static order-0 rANS (Duda's asymmetric numeral systems, byte-wise renormalisation as in ryg_rans) for the classes
// whose bytes no context predicts: the encoder stores each such class's histogram (normalised to 2^15) and the
// decoder needs one table lookup and one multiply per byte. For an order-0 source the stored histogram costs ~4 kbit
// per class and segment against the adaptive coder's learning cost; choose_modes picks the shorter (MDL) and the
// faster when they tie.
namespace rans {
inline constexpr std::uint32_t scale_bits = 15, L = 1u << 23;
struct table { std::array<std::uint32_t, 256> f{}, c{}; };
// counts -> frequencies summing to 2^bits, every present symbol at least 1
inline void normalise(const std::uint64_t* h, table& t, std::uint32_t bits) {
    const std::int64_t M = std::int64_t(1) << bits;
    std::uint64_t tot = 0; for (int s = 0; s < 256; ++s) tot += h[s];
    std::int64_t sum = 0; int big = 0;
    for (int s = 0; s < 256; ++s) {
        t.f[std::size_t(s)] = h[s] ? std::max<std::uint32_t>(1, std::uint32_t((double(h[s]) * double(M)) / double(tot) + 0.5)) : 0;
        sum += t.f[std::size_t(s)];
        if (h[s] > h[big]) big = s;
    }
    while (sum != M) {   // put the rounding difference on the largest frequencies
        if (sum < M) { t.f[std::size_t(big)] += std::uint32_t(M - sum); break; }
        int s = 0; for (int i = 0; i < 256; ++i) if (t.f[std::size_t(i)] > t.f[std::size_t(s)]) s = i;
        const std::uint32_t take = std::uint32_t(std::min<std::int64_t>(sum - M, std::int64_t(t.f[std::size_t(s)]) - 1));
        t.f[std::size_t(s)] -= take; sum -= take;
    }
    std::uint32_t c = 0; for (int s = 0; s < 256; ++s) { t.c[std::size_t(s)] = c; c += t.f[std::size_t(s)]; }
}
inline double cost_bits(const std::uint64_t* h, const table& t, std::uint32_t bits) {
    double b = 0; for (int s = 0; s < 256; ++s) if (h[s]) b -= double(h[s]) * (std::log2(double(t.f[std::size_t(s)])) - double(bits));
    return b;
}
}
namespace detail {
// Description length in bits of the bytes of class `cl` under a context (ctx(i) in [0, 256)) estimated from the
// segment itself: the empirical conditional entropy plus, per context, (symbols - 1) / 2 * log2(n) bits for learning
// it (a two-part MDL code length).
template <class Ctx>
inline double code_length(const layout& L, const std::uint8_t* buf, std::size_t n, int cl, Ctx ctx, std::vector<std::uint32_t>& cnt) {
    cnt.assign(std::size_t(1) << 16, 0);
    for (std::size_t i = 0, j = 0; i < n; ++i, j = j + 1 == L.period ? 0 : j + 1)
        if (L.cls[j] == cl) cnt[(std::size_t(ctx(i)) << 8) | buf[i]]++;
    double bits = 0;
    for (std::size_t c = 0; c < 256; ++c) {
        const std::uint32_t* h = cnt.data() + (c << 8);
        double tot = 0; int k = 0;
        for (int v = 0; v < 256; ++v) if (h[v]) { tot += h[v]; ++k; }
        if (tot == 0) continue;
        for (int v = 0; v < 256; ++v) if (h[v]) bits -= double(h[v]) * std::log2(double(h[v]) / tot);
        bits += 0.5 * double(k - 1) * std::log2(tot + 1) + 8;
    }
    return bits;
}
}
// Class codings besides the context-mixing masks (1..15), mode_raw and mode_const: static rANS of order 0, and of
// order 1 on the previous byte in coding order (for the byte after a float's sign / exponent byte: its distribution
// depends on that byte, a handful of whose values occur, so one table per value present is cheap to store).
inline constexpr std::uint8_t mode_rans = 18, mode_rans1 = 19;
inline constexpr std::uint32_t rans1_bits = 12;
struct class_code {
    rans::table t;                       // mode_rans
    std::vector<std::uint8_t> ctx;       // mode_rans1: the context bytes present, and their tables
    std::vector<rans::table> t1;
    std::array<std::int16_t, 256> at{};  // context byte -> index into t1, -1 none
};
namespace detail {
inline bool is_cm(std::uint8_t m) { return m >= 1 && m <= 15; }
// position j's predecessor in coding order is position j - 1 of the period (the last one of the previous period)
inline bool pred_is_cm(const layout& L, const std::uint8_t* mode, int c) {
    for (std::size_t j = 0; j < L.period; ++j)
        if (L.cls[j] == c && is_cm(mode[L.cls[j == 0 ? L.period - 1 : j - 1]])) return true;
    return false;
}
}
// The coding of each class, chosen by two-part code length (MDL) over the segment, among: context mixing of the
// models that describe the class in clearly fewer bits than order 0 (estimated with a 4% allowance for the mixer),
// static rANS of order 0 or 1 (histograms stored), stored bytes, a single value. Ties go to the faster coding: most
// weight bytes are close to order-0 sources, for which static rANS is both as short and many times faster.
inline void choose_modes(const layout& L, std::size_t row, const std::uint8_t* buf, std::size_t n, std::uint8_t* mode, std::uint8_t* cval,
                         class_code* cc) {
    const std::size_t P = L.period;
    const std::size_t d2 = P > 1 ? P : 2, d3 = row && row % P == 0 ? row : P > 1 ? 2 * P : 3;
    std::vector<std::uint32_t> cnt;
    struct option { double bits; std::uint8_t mode; };
    std::vector<std::vector<option>> opts(L.ncls);
    for (int c = 0; c < L.ncls; ++c) {
        std::array<std::uint64_t, 256> h{}; std::uint64_t tot = 0;
        for (std::size_t i = 0, j = 0; i < n; ++i, j = j + 1 == P ? 0 : j + 1) if (L.cls[j] == c) { h[buf[i]]++; ++tot; }
        int nz = 0, last = 0;
        for (int v = 0; v < 256; ++v) if (h[std::size_t(v)]) { ++nz; last = v; }
        cval[c] = 0;
        auto& o = opts[std::size_t(c)];
        if (tot == 0 || nz == 1) { cval[c] = std::uint8_t(last); o.push_back({0, cm::mode_const}); continue; }
        o.push_back({8.0 * double(tot), cm::mode_raw});
        const double l0 = detail::code_length(L, buf, n, c, [](std::size_t) { return 0; }, cnt);
        const double l2 = detail::code_length(L, buf, n, c, [&](std::size_t i) { return i >= d2 ? buf[i - d2] : 0; }, cnt);
        const double l3 = detail::code_length(L, buf, n, c, [&](std::size_t i) { return i >= d3 ? buf[i - d3] : 0; }, cnt);
        const double l1 = detail::code_length(L, buf, n, c, [&](std::size_t i) { return i ? buf[i - 1] : 0; }, cnt);   // last: cnt kept for rANS-1
        std::uint8_t m = 1;
        constexpr double gain = 0.97;   // a context model must save 3% to be worth its time
        if (l1 < gain * l0) m |= 2;
        if (l2 < gain * l0) m |= 4;
        if (l3 < gain * l0) m |= 8;
        o.push_back({1.04 * std::min({l0, l1, l2, l3}), m});
        auto& k = cc[c];
        rans::normalise(h.data(), k.t, rans::scale_bits);
        o.push_back({rans::cost_bits(h.data(), k.t, rans::scale_bits) + 256 * 16, mode_rans});
        k.ctx.clear(); k.t1.clear(); k.at.fill(-1);
        double b1 = 16;
        std::array<std::uint64_t, 256> hc{};
        for (int x = 0; x < 256; ++x) {
            const std::uint32_t* q = cnt.data() + (std::size_t(x) << 8);
            bool any = false;
            for (int v = 0; v < 256; ++v) { hc[std::size_t(v)] = q[v]; any |= q[v] != 0; }
            if (!any) continue;
            k.at[std::size_t(x)] = std::int16_t(k.t1.size());
            k.ctx.push_back(std::uint8_t(x)); k.t1.emplace_back();
            rans::normalise(hc.data(), k.t1.back(), rans1_bits);
            b1 += rans::cost_bits(hc.data(), k.t1.back(), rans1_bits) + 8 + 256 * 16;
        }
        o.push_back({b1, mode_rans1});
        std::stable_sort(o.begin(), o.end(), [](const option& a, const option& b) { return a.bits < b.bits; });
    }
    // best option per class; an order-1 rANS class needs its predecessor bytes decoded before the mixing pass, so
    // it cannot follow a context-mixed class: fall back to its next best option until no conflict is left
    std::vector<std::size_t> pick(L.ncls, 0);
    for (int c = 0; c < L.ncls; ++c) mode[c] = opts[std::size_t(c)][0].mode;
    for (bool changed = true; changed;) {
        changed = false;
        for (int c = 0; c < L.ncls; ++c)
            if (mode[c] == mode_rans1 && detail::pred_is_cm(L, mode, c)) {
                auto& o = opts[std::size_t(c)];
                do ++pick[std::size_t(c)]; while (o[pick[std::size_t(c)]].mode == mode_rans1);
                mode[c] = o[pick[std::size_t(c)]].mode; changed = true;
            }
    }
}
namespace detail {
inline void put_freqs(std::vector<std::uint8_t>& out, const rans::table& t) {
    for (int s = 0; s < 256; ++s) { out.push_back(std::uint8_t(t.f[std::size_t(s)])); out.push_back(std::uint8_t(t.f[std::size_t(s)] >> 8)); }
}
}
// Encodes n bytes (a multiple of L.period) -> modelled payload; returns false when storing them is no larger.
inline bool encode_segment(const layout& L, std::size_t row, const std::uint8_t* src, std::size_t n, std::vector<std::uint8_t>& out) {
    out.clear();
    thread_local std::vector<std::uint8_t> buf; buf.resize(n);
    detail::to_coding_order(L, src, n, buf.data());
    std::vector<std::uint8_t> mode(L.ncls, 0), cval(L.ncls, 0);
    std::vector<class_code> cc(L.ncls);
    choose_modes(L, row, buf.data(), n, mode.data(), cval.data(), cc.data());
    out.push_back(L.ncls);
    for (auto m : mode) out.push_back(m);
    for (auto v : cval) out.push_back(v);
    for (int c = 0; c < L.ncls; ++c) {
        if (mode[std::size_t(c)] == mode_rans) detail::put_freqs(out, cc[std::size_t(c)].t);
        if (mode[std::size_t(c)] == mode_rans1) {
            auto& k = cc[std::size_t(c)];
            out.push_back(std::uint8_t(k.ctx.size())); out.push_back(std::uint8_t(k.ctx.size() >> 8));
            for (std::size_t q = 0; q < k.ctx.size(); ++q) { out.push_back(k.ctx[q]); detail::put_freqs(out, k.t1[q]); }
        }
    }
    std::uint64_t nraw = 0;
    for (std::size_t i = 0, j = 0; i < n; ++i, j = j + 1 == L.period ? 0 : j + 1) nraw += mode[L.cls[j]] == cm::mode_raw;
    detail::put64(out, nraw);
    for (std::size_t i = 0, j = 0; i < n; ++i, j = j + 1 == L.period ? 0 : j + 1) if (mode[L.cls[j]] == cm::mode_raw) out.push_back(buf[i]);
    {   // rANS stream: symbols coded last to first, the bytes reversed at the end so the decoder reads forwards
        std::vector<std::uint8_t> rs;
        if (std::any_of(mode.begin(), mode.end(), [](std::uint8_t m) { return m == mode_rans || m == mode_rans1; })) {
            std::uint32_t x = rans::L;
            for (std::size_t i = n; i-- > 0;) {
                const int c = L.cls[i % L.period];
                const std::uint8_t md = mode[std::size_t(c)];
                if (md != mode_rans && md != mode_rans1) continue;
                const auto& k = cc[std::size_t(c)];
                const rans::table& t = md == mode_rans ? k.t : k.t1[std::size_t(k.at[i ? buf[i - 1] : 0])];
                const std::uint32_t sb = md == mode_rans ? rans::scale_bits : rans1_bits;
                const std::uint32_t f = t.f[buf[i]];
                const std::uint32_t xmax = ((rans::L >> sb) << 8) * f;
                while (x >= xmax) { rs.push_back(std::uint8_t(x)); x >>= 8; }
                x = ((x / f) << sb) + (x % f) + t.c[buf[i]];
            }
            for (int q = 3; q >= 0; --q) rs.push_back(std::uint8_t(x >> (8 * q)));
            std::reverse(rs.begin(), rs.end());
        }
        detail::put64(out, rs.size());
        out.insert(out.end(), rs.begin(), rs.end());
    }
    if (std::any_of(mode.begin(), mode.end(), detail::is_cm)) {
        auto& m = cm::thread_model(); m.reset(L.ncls, mode.data());
        cm::encoder ac; ac.out = &out;
        cm::run<true>(m, L, mode.data(), row, buf.data(), n, ac);
        ac.flush();
    }
    return out.size() < n;
}
namespace detail {
struct dtab { std::array<std::uint32_t, 256> f{}, c{}; std::vector<std::uint8_t> sym; std::uint32_t bits = 0; };
inline const std::uint8_t* read_freqs(const std::uint8_t* p, const std::uint8_t* end, dtab& t, std::uint32_t bits) {
    if (end - p < 512) raise(errc::format_error, "packed segment: truncated frequency table");
    const std::uint32_t M = 1u << bits; std::uint32_t cum = 0;
    t.bits = bits; t.sym.resize(M);
    for (int s = 0; s < 256; ++s) {
        const std::uint32_t f = std::uint32_t(p[2 * s]) | std::uint32_t(p[2 * s + 1]) << 8;
        if (f > M - cum) raise(errc::format_error, "packed segment: bad frequency table");
        t.f[std::size_t(s)] = f; t.c[std::size_t(s)] = cum;
        std::fill_n(t.sym.begin() + cum, f, std::uint8_t(s));
        cum += f;
    }
    if (cum != M) raise(errc::format_error, "packed segment: bad frequency table");
    return p + 512;
}
}
// Decodes a modelled payload into n bytes at dst. Throws format_error on a malformed payload.
inline void decode_segment(const layout& L, std::size_t row, const std::uint8_t* in, std::size_t nin, std::uint8_t* dst, std::size_t n) {
    const std::uint8_t* end = in + nin;
    if (nin < 1 + 2 * std::size_t(L.ncls) || in[0] != L.ncls) raise(errc::format_error, "packed segment: bad header");
    const std::uint8_t* mode = in + 1; const std::uint8_t* cval = in + 1 + L.ncls;
    const std::uint8_t* p = in + 1 + 2 * std::size_t(L.ncls);
    struct ctab { detail::dtab t0; std::vector<detail::dtab> t1; std::array<std::int16_t, 256> at; };
    thread_local std::vector<ctab> dt; dt.resize(L.ncls);
    for (int c = 0; c < L.ncls; ++c) {
        const std::uint8_t md = mode[c];
        if (md == 0 || (md > cm::mode_const && md != mode_rans && md != mode_rans1)) raise(errc::format_error, "packed segment: bad class mode");
        if (md == mode_rans1 && detail::pred_is_cm(L, mode, c)) raise(errc::format_error, "packed segment: bad class mode");
        auto& k = dt[std::size_t(c)];
        if (md == mode_rans) p = detail::read_freqs(p, end, k.t0, rans::scale_bits);
        if (md == mode_rans1) {
            if (end - p < 2) raise(errc::format_error, "packed segment: truncated");
            const std::size_t nk = std::size_t(p[0] | p[1] << 8); p += 2;
            if (nk > 256) raise(errc::format_error, "packed segment: bad context count");
            k.at.fill(-1); k.t1.resize(nk);
            for (std::size_t q = 0; q < nk; ++q) {
                if (end - p < 1) raise(errc::format_error, "packed segment: truncated");
                const std::uint8_t x = *p++;
                if (k.at[x] >= 0) raise(errc::format_error, "packed segment: repeated context");
                k.at[x] = std::int16_t(q);
                p = detail::read_freqs(p, end, k.t1[q], rans1_bits);
            }
        }
    }
    if (end - p < 8) raise(errc::format_error, "packed segment: truncated");
    const std::uint64_t nraw = detail::get64(p); p += 8;
    if (nraw > std::uint64_t(end - p)) raise(errc::format_error, "packed segment: raw bytes out of bounds");
    const std::uint8_t* raw = p; const std::uint8_t* raw_end = p + nraw; p = raw_end;
    if (end - p < 8) raise(errc::format_error, "packed segment: truncated");
    const std::uint64_t nr = detail::get64(p); p += 8;
    if (nr > std::uint64_t(end - p)) raise(errc::format_error, "packed segment: rANS bytes out of bounds");
    const std::uint8_t* rp = p; const std::uint8_t* rend = p + nr; p = rend;
    std::uint32_t x = 0;
    if (nr) { if (nr < 4) raise(errc::format_error, "packed segment: rANS stream too short"); x = detail::get32(rp); rp += 4; }
    thread_local std::vector<std::uint8_t> buf; buf.resize(n);
    for (std::size_t i = 0, j = 0; i < n; ++i, j = j + 1 == L.period ? 0 : j + 1) {
        const int c = L.cls[j];
        const std::uint8_t md = mode[c];
        if (md == mode_rans || md == mode_rans1) {
            const auto& k = dt[std::size_t(c)];
            const detail::dtab* t = &k.t0;
            if (md == mode_rans1) {
                const int q = k.at[i ? buf[i - 1] : 0];
                if (q < 0) raise(errc::format_error, "packed segment: context without a table");
                t = &k.t1[std::size_t(q)];
            }
            const std::uint32_t mask = (1u << t->bits) - 1;
            const std::uint8_t s = t->sym[x & mask];
            x = t->f[s] * (x >> t->bits) + (x & mask) - t->c[s];
            for (int g = 0; x < rans::L && g < 4; ++g) x = (x << 8) | (rp < rend ? *rp++ : 0);   // a corrupt stream: garbage, caught by the hash
            buf[i] = s;
        } else if (md == cm::mode_raw) { if (raw == raw_end) raise(errc::format_error, "packed segment: raw bytes short"); buf[i] = *raw++; }
        else if (md == cm::mode_const) buf[i] = cval[c];
    }
    if (raw != raw_end) raise(errc::format_error, "packed segment: raw bytes left over");
    if (std::any_of(mode, mode + L.ncls, detail::is_cm)) {
        auto& m = cm::thread_model(); m.reset(L.ncls, mode);
        cm::decoder ac(p, end);
        cm::run<false>(m, L, mode, row, buf.data(), n, ac);
    }
    detail::from_coding_order(L, buf.data(), n, dst);
}

// ---- files ----
struct segment {
    std::uint64_t offset = 0, bytes = 0, poff = 0, pbytes = 0;
    std::uint32_t layout = 0, row = 0;
    std::uint8_t method = 0;
    std::string codec;   // pack side only: what the bytes are (statistics)
};
inline constexpr std::uint64_t max_segment_bytes = std::uint64_t(256) << 20;   // what an unpacker allocates at most per segment
struct options {
    std::size_t segment_bytes = std::size_t(8) << 20;   // target size of a tensor segment (whole rows), at most max_segment_bytes
};
struct stats {
    struct per { std::uint64_t in = 0, out = 0; std::size_t tensors = 0; };
    std::map<std::string, per> by_codec;   // "(other)": bytes outside tensors
    std::uint64_t in = 0, out = 0;
    double seconds = 0;
    std::size_t segments = 0;
};

namespace detail {
struct tensor_span { std::uint64_t off, bytes, row; std::string codec; };
// Tensor byte ranges of the .fpk image `m` (file offsets), sorted, non-overlapping; empty when the index cannot be read
// (encrypted data or not an .fpk: then the whole file is one stream of generic bytes).
inline std::vector<tensor_span> tensor_spans(const std::uint8_t* m, std::size_t size) {
    std::vector<tensor_span> r;
    static constexpr char fpk_magic[8] = {'F', 'O', 'R', 'G', 'E', 'P', 'K', '1'};
    if (size < 64 || std::memcmp(m, fpk_magic, 8)) return r;
    const std::uint32_t n = get32(m + 12); const std::uint64_t toff = get64(m + 16);
    if (toff > size || std::uint64_t(n) * 96 > size - toff) return r;
    std::uint64_t doff = 0, dsize = 0, ioff = 0, isize = 0; bool have_d = false, have_i = false;
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::uint8_t* rec = m + toff + 96 * std::uint64_t(i);
        std::string nm(reinterpret_cast<const char*>(rec), strnlen(reinterpret_cast<const char*>(rec), 32));
        const std::uint64_t so = get64(rec + 32), ss = get64(rec + 40);
        if (so > size || ss > size - so) return {};
        if (nm == "data") { doff = so; dsize = ss; have_d = true; }
        else if (nm == "tensors") { ioff = so; isize = ss; have_i = true; }
        else if (nm == "encryption" || nm == "shards") return {};
    }
    if (!have_d || !have_i) return r;
    json::value idx;
    try { idx = json::parse(std::string_view(reinterpret_cast<const char*>(m + ioff), isize)); } catch (const std::exception&) { return {}; }
    if (!idx.is_arr()) return {};
    for (auto& t : idx.arr()) {
        if (!t.is_obj()) continue;
        const double off = t.num_or("offset", -1), nb = t.num_or("bytes", -1);
        if (!(off >= 0 && nb > 0 && off + nb <= double(dsize))) continue;
        tensor_span s{doff + std::uint64_t(off), std::uint64_t(nb), 0, t.str_or("codec", "")};
        if (const auto* sh = t.find("shape"); sh && sh->is_arr() && !sh->arr().empty()) {
            const auto* c = graph::codecs::find(s.codec);
            const auto& cv = sh->arr().back(); const double cols = cv.is_num() ? cv.num() : 0;
            if (c && cols >= 1 && cols <= 1e9) s.row = c->bytes(std::size_t(cols));
        }
        r.push_back(std::move(s));
    }
    std::sort(r.begin(), r.end(), [](auto& a, auto& b) { return a.off < b.off; });
    std::vector<tensor_span> u;   // aliased / overlapping entries: the first one covers the bytes
    for (auto& s : r) if (u.empty() || s.off >= u.back().off + u.back().bytes) u.push_back(std::move(s));
    return u;
}
inline void write_all(std::FILE* f, const void* p, std::size_t n, const std::string& path) {
    if (n && std::fwrite(p, 1, n, f) != n) raise(errc::io_error, "short write {}", path);
}
// Runs f(i) for i in [0, n) on the task engine; the first exception is rethrown on the caller.
template <class F> void for_each_parallel(std::size_t n, F&& f) {
    std::exception_ptr err; std::mutex mu; std::atomic<bool> failed{false};
    task::parallel_for(n, 1, 1, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e && !failed.load(); ++i)
            try { f(i); } catch (...) { std::lock_guard l(mu); if (!err) err = std::current_exception(); failed = true; }
    });
    if (err) std::rethrow_exception(err);
}
// Batches of segments: produce(b0, b1, slot) codes batch [b0, b1) on the task engine while consume(...) of the batch
// before it (the other slot) hashes and writes it on a helper thread, in order.
template <class P, class C> void pipeline(std::size_t n, P&& produce, C&& consume) {
    const std::size_t batch = std::max<std::size_t>(8, 2 * std::size_t(task::engine_threads()));
    std::exception_ptr err; std::thread th;
    auto join = [&] { if (th.joinable()) th.join(); };
    int slot = 0;
    try {
        for (std::size_t b0 = 0; b0 < n; b0 += batch, slot ^= 1) {
            const std::size_t b1 = std::min(n, b0 + batch);
            produce(b0, b1, slot);
            join();
            if (err) std::rethrow_exception(err);
            th = std::thread([&consume, &err, b0, b1, slot] { try { consume(b0, b1, slot); } catch (...) { err = std::current_exception(); } });
        }
        join();
        if (err) std::rethrow_exception(err);
    } catch (...) { join(); throw; }
}
struct file_closer { std::FILE* f; ~file_closer() { if (f) std::fclose(f); } };
struct unmapper { const std::uint8_t* p; std::size_t n; ~unmapper() { forge::unmap_readonly(p, n); } };
}

inline bool is_packed(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char b[8] = {}; bool ok = std::fread(b, 1, 8, f) == 8 && std::memcmp(b, magic, 8) == 0;
    std::fclose(f);
    return ok;
}

// BLAKE3 (hex) of the original file a packed file records; empty when `path` is not a packed file.
inline std::string recorded_blake3(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::uint8_t h[header_bytes] = {};
    const bool ok = std::fread(h, 1, header_bytes, f) == header_bytes && std::memcmp(h, magic, 8) == 0;
    std::fclose(f);
    if (!ok) return {};
    hash::blake3::digest d; std::memcpy(d.data(), h + 48, 32);
    return hash::blake3::hex(d);
}

// Packs the file at `in` (normally an .fpk) into `out`.
inline stats pack_file(const std::string& in, const std::string& out, options opt = {}) {
    opt.segment_bytes = std::clamp<std::size_t>(opt.segment_bytes, 4096, std::size_t(max_segment_bytes) / 2);
    security::enforce(security::capability::fs_read, in);
    security::enforce(security::capability::fs_write, out);
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t size = 0;
    const std::uint8_t* m = forge::map_readonly(in, size);
    detail::unmapper um{m, size};
    // segments
    std::vector<layout> layouts{layout{}};
    auto layout_id = [&](const layout& L) {
        for (std::size_t i = 0; i < layouts.size(); ++i) if (layouts[i] == L) return std::uint32_t(i);
        layouts.push_back(L); return std::uint32_t(layouts.size() - 1);
    };
    std::vector<segment> segs;
    auto add_generic = [&](std::uint64_t a, std::uint64_t b) {
        for (std::uint64_t p = a; p < b;) { std::uint64_t e = std::min<std::uint64_t>(b, p + opt.segment_bytes); segs.push_back({p, e - p, 0, 0, 0, 0, 1, ""}); p = e; }
    };
    std::uint64_t at = 0;
    const auto spans = detail::tensor_spans(m, size);
    for (auto& t : spans) {
        if (t.off > at) add_generic(at, t.off);
        layout L = layout_for(t.codec);
        if (t.bytes % L.period) L = layout{};
        std::uint64_t row = t.row && t.row % L.period == 0 && t.bytes % t.row == 0 && t.row <= opt.segment_bytes ? t.row : 0;
        const std::uint64_t unit = row ? row : L.period;
        const std::uint64_t per = std::max<std::uint64_t>(unit, opt.segment_bytes / unit * unit);
        const std::uint32_t lid = layout_id(L);
        for (std::uint64_t p = 0; p < t.bytes;) {
            std::uint64_t e = std::min<std::uint64_t>(t.bytes, p + per);
            segs.push_back({t.off + p, e - p, 0, 0, lid, std::uint32_t(std::min<std::uint64_t>(row, 0xffffffffu)), 1, t.codec});
            p = e;
        }
        at = t.off + t.bytes;
    }
    if (at < size) add_generic(at, size);
    for (auto& s : segs) if (s.row > s.bytes) s.row = 0;
    // code batches of segments on all threads; hash the original and write the previous batch meanwhile
    hash::blake3 hasher;
    const std::string part = out + ".partial";
    std::FILE* f = std::fopen(part.c_str(), "wb");
    if (!f) raise(errc::io_error, "cannot write {}", part);
    detail::file_closer fc{f};
    stats st;
    try {
    std::vector<std::uint8_t> head(header_bytes, 0);
    detail::write_all(f, head.data(), head.size(), part);
    std::uint64_t pos = header_bytes;
    std::vector<std::vector<std::uint8_t>> payload[2];
    detail::pipeline(segs.size(), [&](std::size_t b0, std::size_t b1, int slot) {
        auto& pl = payload[slot]; pl.assign(b1 - b0, {});
        detail::for_each_parallel(b1 - b0, [&](std::size_t i) {
            auto& s = segs[b0 + i];
            if (!encode_segment(layouts[s.layout], s.row, m + s.offset, std::size_t(s.bytes), pl[i])) { s.method = 0; pl[i].clear(); }
        });
    }, [&](std::size_t b0, std::size_t b1, int slot) {
        for (std::size_t i = b0; i < b1; ++i) {
            auto& s = segs[i]; auto& pl = payload[slot][i - b0];
            hasher.update(m + s.offset, std::size_t(s.bytes));
            s.poff = pos;
            if (s.method == 0) detail::write_all(f, m + s.offset, std::size_t(s.bytes), part), s.pbytes = s.bytes;
            else detail::write_all(f, pl.data(), pl.size(), part), s.pbytes = pl.size();
            pos += s.pbytes;
            auto& c = st.by_codec[s.codec.empty() ? std::string("(other)") : s.codec];
            c.in += s.bytes; c.out += s.pbytes;
            std::vector<std::uint8_t>().swap(pl);
        }
    });
    for (auto& t : spans) st.by_codec[t.codec].tensors++;
    // index
    std::vector<std::uint8_t> idx;
    for (auto& L : layouts) {
        idx.push_back(std::uint8_t(L.period)); idx.push_back(std::uint8_t(L.period >> 8)); idx.push_back(L.ncls); idx.push_back(0);
        idx.insert(idx.end(), L.order.begin(), L.order.begin() + L.period);
        idx.insert(idx.end(), L.cls.begin(), L.cls.begin() + L.period);
    }
    for (auto& s : segs) {
        std::uint8_t r[segment_record] = {};
        detail::set64(r, s.offset); detail::set64(r + 8, s.bytes); detail::set64(r + 16, s.poff); detail::set64(r + 24, s.pbytes);
        detail::set32(r + 32, s.layout); detail::set32(r + 36, s.row); r[40] = s.method;
        idx.insert(idx.end(), r, r + segment_record);
    }
    detail::write_all(f, idx.data(), idx.size(), part);
    const auto digest = hasher.final();
    std::memcpy(head.data(), magic, 8); detail::set32(head.data() + 8, version); detail::set32(head.data() + 12, std::uint32_t(layouts.size()));
    detail::set32(head.data() + 16, std::uint32_t(segs.size())); detail::set64(head.data() + 24, size); detail::set64(head.data() + 32, pos);
    detail::set64(head.data() + 40, idx.size()); std::memcpy(head.data() + 48, digest.data(), 32);
    if (std::fseek(f, 0, SEEK_SET) != 0) raise(errc::io_error, "seek {}", part);
    detail::write_all(f, head.data(), head.size(), part);
    const int rc = std::fclose(f); fc.f = nullptr;
    if (rc != 0) raise(errc::io_error, "cannot finish {}", part);
    if (auto s = os::rename(part, out); !s) raise(errc::io_error, "cannot rename {} to {}: {}", part, out, s.err.what);
    st.out = pos + idx.size();
    } catch (...) { if (fc.f) { std::fclose(fc.f); fc.f = nullptr; } (void)os::remove(part); throw; }
    st.in = size; st.segments = segs.size();
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

struct unpack_stats { std::uint64_t in = 0, out = 0; double seconds = 0; std::size_t segments = 0; std::string blake3; };
// Unpacks `in` (.fpkz) into `out`; the result is checked against the recorded BLAKE3 of the original before it is
// renamed into place, so `out` is either the original file or absent.
inline unpack_stats unpack_file(const std::string& in, const std::string& out) {
    security::enforce(security::capability::fs_read, in);
    security::enforce(security::capability::fs_write, out);
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t size = 0;
    const std::uint8_t* m = forge::map_readonly(in, size);
    detail::unmapper um{m, size};
    if (size < header_bytes || std::memcmp(m, magic, 8)) raise(errc::format_error, "{}: not a packed model (.fpkz)", in);
    if (detail::get32(m + 8) != version) raise(errc::unsupported, "{}: packed format version {} not supported", in, detail::get32(m + 8));
    const std::uint32_t nl = detail::get32(m + 12), ns = detail::get32(m + 16);
    const std::uint64_t osize = detail::get64(m + 24), ioff = detail::get64(m + 32), ibytes = detail::get64(m + 40);
    hash::blake3::digest want; std::memcpy(want.data(), m + 48, 32);
    if (ioff > size || ibytes > size - ioff || nl == 0 || nl > 65536) raise(errc::format_error, "{}: corrupt index", in);
    const std::uint8_t* ip = m + ioff; const std::uint8_t* iend = ip + ibytes;
    std::vector<layout> layouts(nl);
    for (auto& L : layouts) {
        if (iend - ip < 4) raise(errc::format_error, "{}: corrupt index", in);
        L.period = std::uint16_t(ip[0] | ip[1] << 8); L.ncls = ip[2]; ip += 4;
        if (L.period < 1 || L.period > 256 || L.ncls < 1 || L.ncls > max_classes || std::size_t(iend - ip) < 2 * std::size_t(L.period)) raise(errc::format_error, "{}: corrupt layout", in);
        std::copy_n(ip, L.period, L.order.begin()); std::copy_n(ip + L.period, L.period, L.cls.begin()); ip += 2 * std::size_t(L.period);
        std::vector<bool> seen(L.period, false);
        for (std::size_t j = 0; j < L.period; ++j) {
            if (L.order[j] >= L.period || seen[L.order[j]] || L.cls[j] >= L.ncls) raise(errc::format_error, "{}: corrupt layout", in);
            seen[L.order[j]] = true;
        }
    }
    if (std::uint64_t(iend - ip) != std::uint64_t(ns) * segment_record) raise(errc::format_error, "{}: corrupt index", in);
    std::vector<segment> segs(ns);
    std::uint64_t at = 0;
    for (auto& s : segs) {
        s.offset = detail::get64(ip); s.bytes = detail::get64(ip + 8); s.poff = detail::get64(ip + 16); s.pbytes = detail::get64(ip + 24);
        s.layout = detail::get32(ip + 32); s.row = detail::get32(ip + 36); s.method = ip[40]; ip += segment_record;
        if (s.offset != at || s.bytes == 0 || s.bytes > osize - at || s.layout >= nl || s.method > 1 || s.poff > size || s.pbytes > size - s.poff ||
            (s.method == 0 && s.pbytes != s.bytes) || s.bytes % layouts[s.layout].period || s.bytes > max_segment_bytes)
            raise(errc::format_error, "{}: corrupt segment table", in);
        at += s.bytes;
    }
    if (at != osize) raise(errc::format_error, "{}: segments do not cover the original file", in);
    const std::string part = out + ".partial";
    std::FILE* f = std::fopen(part.c_str(), "wb");
    if (!f) raise(errc::io_error, "cannot write {}", part);
    detail::file_closer fc{f};
    hash::blake3 hasher;
    std::vector<std::vector<std::uint8_t>> buf[2];
    try {
        detail::pipeline(segs.size(), [&](std::size_t b0, std::size_t b1, int slot) {
            auto& bs = buf[slot]; bs.resize(b1 - b0);
            detail::for_each_parallel(b1 - b0, [&](std::size_t i) {
                const auto& s = segs[b0 + i];
                if (s.method == 0) return;
                bs[i].resize(std::size_t(s.bytes));
                decode_segment(layouts[s.layout], s.row, m + s.poff, std::size_t(s.pbytes), bs[i].data(), std::size_t(s.bytes));
            });
        }, [&](std::size_t b0, std::size_t b1, int slot) {
            for (std::size_t i = b0; i < b1; ++i) {
                const auto& s = segs[i];
                const std::uint8_t* p = s.method == 0 ? m + s.poff : buf[slot][i - b0].data();
                hasher.update(p, std::size_t(s.bytes));
                detail::write_all(f, p, std::size_t(s.bytes), part);
            }
        });
        const int rc = std::fclose(f); fc.f = nullptr;
        if (rc != 0) raise(errc::io_error, "cannot finish {}", part);
        if (hasher.final() != want) raise(errc::integrity_error, "{}: unpacked bytes do not match the recorded BLAKE3 (corrupt packed file)", in);
    } catch (...) { if (fc.f) { std::fclose(fc.f); fc.f = nullptr; } (void)os::remove(part); throw; }
    if (auto s = os::rename(part, out); !s) raise(errc::io_error, "cannot rename {} to {}: {}", part, out, s.err.what);
    unpack_stats r; r.in = size; r.out = osize; r.segments = segs.size(); r.blake3 = hash::blake3::hex(want);
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

} // namespace forge::modelfile::pack
