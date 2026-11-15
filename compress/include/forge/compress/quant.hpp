// Weight quantisation into FORGE's block codecs (q8_0, q4_0, q4_k, q5_k, q6_k, q4_g32, mxfp4, nvfp4, ...).
//   rtn   round-to-nearest: each row encoded / decoded by the codec (what `forge import --codec` does)
//   gptq  GPTQ (Frantar et al., 2023) generalised to any FORGE codec: columns are visited in order with the upper
//         Cholesky factor of H^-1; when a column is reached, its codec block (of the current, error-updated row) is
//         encoded and decoded by the codec itself and the column takes its decoded value, and the rounding error is
//         pushed onto the columns not yet quantised (lazy blocked updates, as in SparseGPT). Because the codec does
//         the rounding, the result is exactly representable by the codec the model is saved in.
// Zeros of sparse weights stay zero (codecs whose grid contains 0: q8_0, q4_0, the symmetric g32 / mx / nv
// formats), so quantisation composes with pruning.
#pragma once
#include <forge/compress/prune.hpp>

namespace forge::compress {

inline const forge::graph::codecs::codec& codec_of(const std::string& name) {
    const auto* c = forge::graph::codecs::find(name);
    if (!c) die("unknown codec " + name);
    return *c;
}

// Encode + decode n values (a multiple of the codec block) in place.
inline void codec_roundtrip(const forge::graph::codecs::codec& c, float* x, std::size_t n, std::vector<std::uint8_t>& buf) {
    buf.resize(c.bytes(n));
    c.encode(x, n, buf.data());
    c.decode(buf.data(), n, x);
}

// True when every group of 4 consecutive weights along each row has at most 2 non-zeros.
inline bool is_2of4(const float* W, std::size_t n, i64 cols) {
    if (cols % 4) return false;
    for (std::size_t g = 0; g < n; g += 4) {
        int nz = (W[g] != 0.f) + (W[g + 1] != 0.f) + (W[g + 2] != 0.f) + (W[g + 3] != 0.f);
        if (nz > 2) return false;
    }
    return true;
}
// The dense codec with the value format of a 2:4 codec (the codec itself otherwise).
inline std::string dense_sibling(const std::string& c) { return c == "s24_f16" ? "f16" : c == "s24_q8" ? "q8_0" : c == "s24_q4" ? "q4_0" : c; }

inline bool codec_fits(const forge::graph::codecs::codec& c, i64 cols) { return cols % i64(c.block) == 0; }

inline void rtn_quantise(float* W, i64 rows, i64 cols, const forge::graph::codecs::codec& c, bool keep_zeros) {
    la::parallel_for(static_cast<std::size_t>(rows), 16, [&](std::size_t b, std::size_t e) {
        std::vector<std::uint8_t> buf; std::vector<std::uint8_t> z;
        for (std::size_t r = b; r < e; ++r) {
            float* w = W + i64(r) * cols;
            if (keep_zeros) { z.assign(static_cast<std::size_t>(cols), 0); for (i64 j = 0; j < cols; ++j) z[std::size_t(j)] = w[j] == 0.f; }
            codec_roundtrip(c, w, std::size_t(cols), buf);
            if (keep_zeros) for (i64 j = 0; j < cols; ++j) if (z[std::size_t(j)]) w[j] = 0.f;
        }
    });
}

inline void gptq_quantise(float* W, i64 rows, i64 cols, const la::Sym& H, const forge::graph::codecs::codec& c, double damp, bool keep_zeros) {
    const i64 G = i64(c.block);
    if (cols % G) die(std::format("gptq: {} columns are not a multiple of the {} block ({})", cols, c.name, G));
    std::vector<std::uint8_t> zero(static_cast<std::size_t>(rows * cols), 0);
    if (keep_zeros) for (std::size_t i = 0; i < zero.size(); ++i) zero[i] = W[i] == 0.f;
    la::Sym Hd = H;
    for (i64 j = 0; j < cols; ++j) if (Hd(j, j) == 0) Hd(j, j) = 1;
    la::Sym U = la::inverse_upper_cholesky(Hd, damp);
    const i64 B = std::max<i64>(G, (128 / G) * G);   // lazy block: whole codec blocks
    std::vector<float> W1, Err;
    for (i64 i1 = 0; i1 < cols; i1 += B) {
        const i64 i2 = std::min(cols, i1 + B), cnt = i2 - i1;
        W1.assign(static_cast<std::size_t>(rows * cnt), 0.f); Err.assign(static_cast<std::size_t>(rows * cnt), 0.f);
        for (i64 r = 0; r < rows; ++r) std::copy_n(W + r * cols + i1, cnt, W1.data() + r * cnt);
        la::parallel_for(static_cast<std::size_t>(rows), 8, [&](std::size_t b, std::size_t e) {
            std::vector<std::uint8_t> buf; std::vector<float> grp(static_cast<std::size_t>(G));
            for (std::size_t r = b; r < e; ++r) {
                float* w1 = W1.data() + r * std::size_t(cnt);
                float* er = Err.data() + r * std::size_t(cnt);
                const std::uint8_t* zr = zero.data() + r * std::size_t(cols) + std::size_t(i1);
                for (i64 i = 0; i < cnt; ++i) {
                    const i64 g0 = i - (i % G);
                    std::copy_n(w1 + g0, G, grp.data());
                    for (i64 t = 0; t < G; ++t) if (zr[g0 + t]) grp[std::size_t(t)] = 0.f;
                    codec_roundtrip(c, grp.data(), std::size_t(G), buf);
                    const double w = w1[i];
                    const double q = zr[i] ? 0.0 : double(grp[std::size_t(i - g0)]);
                    const double err = (w - q) / U(i1 + i, i1 + i);
                    w1[i] = float(q);
                    if (err != 0) { const double* ur = U.row(i1 + i); for (i64 j = i + 1; j < cnt; ++j) w1[j] -= float(err * ur[i1 + j]); }
                    er[i] = float(err);
                }
            }
        });
        for (i64 r = 0; r < rows; ++r) std::copy_n(W1.data() + r * cnt, cnt, W + r * cols + i1);
        if (i2 < cols) {
            const i64 rest = cols - i2;
            std::vector<float> Ut(static_cast<std::size_t>(rest * cnt)), upd(static_cast<std::size_t>(rows * rest));
            for (i64 j = 0; j < rest; ++j) for (i64 i = 0; i < cnt; ++i) Ut[std::size_t(j * cnt + i)] = float(U(i1 + i, i2 + j));
            la::gemm_nt(Err.data(), Ut.data(), upd.data(), rows, rest, cnt);
            la::parallel_for(static_cast<std::size_t>(rows), 16, [&](std::size_t b, std::size_t e) {
                for (std::size_t r = b; r < e; ++r) for (i64 j = 0; j < rest; ++j) W[i64(r) * cols + i2 + j] -= upd[r * std::size_t(rest) + std::size_t(j)];
            });
        }
    }
    // the stored form: one final encode of every row (it reproduces the values chosen above up to the codec's
    // re-fit of a block's scale; zeros stay zero)
    rtn_quantise(W, rows, cols, c, false);
    if (keep_zeros) for (std::size_t i = 0; i < zero.size(); ++i) if (zero[i]) W[i] = 0.f;
}

}  // namespace forge::compress
