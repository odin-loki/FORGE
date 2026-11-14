// Function-preserving reparameterisations (data-free, embarrassingly parallel).
//
// A network is one point of a large set of weight settings that compute the same function: an RMSNorm gain can move
// into the linear layers that read the norm, the residual stream can be expressed in any orthonormal basis (RMSNorm
// only sees ||x||), a SwiGLU neuron's up-row and down-column can trade a scale (the activation is linear in `up`),
// and the value channels of a KV head can trade a scale or rotation with the output projection (attention is linear
// in V). These transforms leave the logits unchanged (up to rounding) but change how hard the weights are to
// compress. They need no data:
//   fold_norms       RMSNorm gains folded into the readers (attn / ffn / output norms become 1)
//   rotate           randomised Hadamard rotation of the residual stream (QuaRot-style, Ashkboos et al. 2024):
//                    outlier channels are spread over all channels, so block quantisation sees flat blocks.
//                    Every reader row becomes w H D / sqrt(d), every writer column D H w / sqrt(d) (FWHT, O(d log d)).
//                    Plus a Hadamard rotation of each head's value channels (V rows / O columns).
//   equalize         per-channel scale balancing (DFQ cross-layer equalisation, Nagel et al. 2019) for the SwiGLU
//                    up/down pair and the V/O pair: the input columns of w_down / wo get equal norms, the rows of
//                    w_up / wv absorb the inverse scale.
// Rotation densifies the residual readers/writers, so it is skipped on sparse models (it would destroy the 2:4
// masks); equalisation keeps zeros.
#pragma once
#include <forge/compress/calib.hpp>

#include <random>

namespace forge::compress {

inline bool is_pow2(i64 n) { return n > 0 && (n & (n - 1)) == 0; }

// In-place fast Walsh-Hadamard transform of x[0..n) (n a power of 2), unnormalised.
inline void fwht(float* x, i64 n) {
    for (i64 h = 1; h < n; h <<= 1)
        for (i64 i = 0; i < n; i += h << 1)
            for (i64 j = i; j < i + h; ++j) { const float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
}

// The orthogonal map Q = H diag(s) / sqrt(n); rows of a reader W [r, n] become w Q, columns of a writer [n, c] become Q^T w.
struct Rotation {
    i64 n = 0;
    std::vector<float> s;   // random signs
    float scale = 1;
    Rotation() = default;
    Rotation(i64 n_, std::uint64_t seed) : n(n_), s(static_cast<std::size_t>(n_)), scale(float(1.0 / std::sqrt(double(n_)))) {
        std::mt19937_64 g(seed);
        for (auto& v : s) v = (g() & 1) ? 1.f : -1.f;
    }
    // reader rows: w <- (w H) * s / sqrt(n)   (H symmetric: w H = fwht(w))
    void rows(float* W, i64 rows) const {
        la::parallel_for(static_cast<std::size_t>(rows), 16, [&](std::size_t b, std::size_t e) {
            for (std::size_t r = b; r < e; ++r) { float* w = W + i64(r) * n; fwht(w, n); for (i64 j = 0; j < n; ++j) w[j] *= s[std::size_t(j)] * scale; }
        });
    }
    // writer columns of W [n, c]: col <- s * (H col) / sqrt(n)
    void cols(float* W, i64 c) const {
        std::vector<float> T(static_cast<std::size_t>(n * c));
        la::transpose(W, T.data(), n, c);   // [c, n]
        la::parallel_for(static_cast<std::size_t>(c), 16, [&](std::size_t b, std::size_t e) {
            for (std::size_t r = b; r < e; ++r) { float* w = T.data() + i64(r) * n; fwht(w, n); for (i64 j = 0; j < n; ++j) w[j] *= s[std::size_t(j)] * scale; }
        });
        la::transpose(T.data(), W, c, n);
    }
};

// Scales the input columns of W [r, n] by g (W <- W diag(g)).
inline void scale_cols(float* W, i64 r, i64 n, const float* g) {
    la::parallel_for(static_cast<std::size_t>(r), 64, [&](std::size_t b, std::size_t e) { for (std::size_t i = b; i < e; ++i) for (i64 j = 0; j < n; ++j) W[i64(i) * n + j] *= g[j]; });
}

// Residual readers of layer l (tensor, rows) and writers (tensor) for the current spec.
struct Rw { std::vector<std::string> readers_attn, readers_ffn, writers; };
inline Rw residual_tensors(const Model& m, i64 l) {
    Rw r;
    const auto& L = m.spec.layers[std::size_t(l)];
    const std::string p = std::format("layer.{}.", l);
    if (L.attn) { r.readers_attn = {p + "wq", p + "wk", p + "wv"}; r.writers.push_back(p + "wo"); }
    switch (L.ffn) {
        case LayerSpec::Ffn::dense: r.readers_ffn = {p + "w_gate", p + "w_up"}; r.writers.push_back(p + "w_down"); break;
        case LayerSpec::Ffn::moe: r.readers_ffn = {p + "router", p + "experts.w_gate", p + "experts.w_up"}; r.writers.push_back(p + "experts.w_down"); break;
        case LayerSpec::Ffn::linear: r.readers_ffn = {p + "lin_a"}; r.writers.push_back(p + "lin_b"); break;
        case LayerSpec::Ffn::none: break;
    }
    return r;
}

// rows of a reader tensor (2-D [r, d] or stacked [e, f, d]) as one [rows, d] matrix
inline i64 reader_rows(const tensor& t) { i64 r = 1; for (std::size_t i = 0; i + 1 < t.shape.size(); ++i) r *= t.shape[i]; return r; }

inline void fold_norms(Model& m) {
    const i64 d = m.spec.d_model;
    auto fold = [&](const std::string& norm, const std::vector<std::string>& readers) {
        auto& g = m.w.at(norm);
        for (auto& n : readers) { auto& t = m.w.at(n); scale_cols(t.data.data(), reader_rows(t), d, g.data.data()); }
        std::fill(g.data.begin(), g.data.end(), 1.f);
    };
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        auto rw = residual_tensors(m, l);
        if (!rw.readers_attn.empty()) fold(std::format("layer.{}.attn_norm", l), rw.readers_attn);
        if (!rw.readers_ffn.empty()) fold(std::format("layer.{}.ffn_norm", l), rw.readers_ffn);
    }
    // a tied head keeps its norm gain (rotate_model turns it into head_mix), so it stays the embedding, stored once
    if (m.w.at("lm_head").data != m.w.at("tok_emb").data) fold("out_norm", {"lm_head"});
}

inline value rotate_model(Model& m, std::uint64_t seed, bool heads) {
    const i64 d = m.spec.d_model;
    if (!is_pow2(d)) die(std::format("rotate: d_model {} is not a power of 2", d));
    const bool tied = m.w.at("lm_head").data == m.w.at("tok_emb").data && !m.spec.head_mix;
    fold_norms(m);
    Rotation Q(d, seed);
    if (tied) {
        // logits = (rms(x') Q^T * g) E^T = rms(x') M (E Q)^T with M = Q^T diag(g) Q: one d x d map, embedding shared
        auto& g = m.w.at("out_norm");
        std::vector<float> Qm(static_cast<std::size_t>(d * d), 0.f);
        for (i64 i = 0; i < d; ++i) Qm[std::size_t(i * d + i)] = 1.f;
        Q.rows(Qm.data(), d);                                   // I Q = Q
        std::vector<float> A(Qm), QT(Qm.size()), AT(Qm.size()), M(Qm.size());
        for (i64 k = 0; k < d; ++k) for (i64 j = 0; j < d; ++j) A[std::size_t(k * d + j)] *= g.data[std::size_t(k)];   // diag(g) Q
        la::transpose(Qm.data(), QT.data(), d, d);
        la::transpose(A.data(), AT.data(), d, d);
        la::gemm_nt(QT.data(), AT.data(), M.data(), d, d, d);   // Q^T (diag(g) Q)
        tensor hm({d, d}); hm.data = M;                           // symmetric: linear(h, M) = h M^T = h M
        m.w.t["head_mix"] = hm;
        m.spec.head_mix = true;
        std::fill(g.data.begin(), g.data.end(), 1.f);
    }
    auto rot_reader = [&](const std::string& n) { auto& t = m.w.at(n); Q.rows(t.data.data(), reader_rows(t)); };
    auto rot_writer = [&](const std::string& n) {
        auto& t = m.w.at(n);
        if (t.shape.size() == 3) { const i64 E = t.shape[0], c = t.shape[2]; for (i64 e = 0; e < E; ++e) Q.cols(t.data.data() + e * d * c, c); }
        else Q.cols(t.data.data(), t.shape[1]);
    };
    rot_reader("tok_emb");   // embedding rows live in the residual basis
    rot_reader("lm_head");
    i64 rotated = 2;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        auto rw = residual_tensors(m, l);
        for (auto& n : rw.readers_attn) { rot_reader(n); ++rotated; }
        for (auto& n : rw.readers_ffn) { rot_reader(n); ++rotated; }
        for (auto& n : rw.writers) { rot_writer(n); ++rotated; }
        if (heads && m.spec.layers[std::size_t(l)].attn && is_pow2(m.spec.head_dim)) {
            // value channels of every KV head: v_h <- v_h R (R = Hadamard / sqrt(hd)); wv rows of a head <- R^T rows,
            // wo columns of every query head <- columns R
            const i64 hd = m.spec.head_dim;
            Rotation R(hd, seed + 17 + std::uint64_t(l));
            auto& wv = m.w.at(std::format("layer.{}.wv", l));
            for (i64 h = 0; h < m.spec.n_kv; ++h) R.cols(wv.data.data() + h * hd * d, d);   // rows h*hd.. as an [hd, d] writer block
            if (m.spec.qkv_bias) { auto& bv = m.w.at(std::format("layer.{}.bv", l)); for (i64 h = 0; h < m.spec.n_kv; ++h) R.cols(bv.data.data() + h * hd, 1); }
            auto& wo = m.w.at(std::format("layer.{}.wo", l));
            const i64 cols = wo.shape[1];
            // wo [d, n_heads*hd]: each row's head segments <- segment R  (reader-style on hd-sized segments)
            la::parallel_for(static_cast<std::size_t>(d), 16, [&](std::size_t b, std::size_t e) {
                for (std::size_t r = b; r < e; ++r) for (i64 h = 0; h < cols / hd; ++h) {
                    float* w = wo.data.data() + i64(r) * cols + h * hd;
                    fwht(w, hd); for (i64 j = 0; j < hd; ++j) w[j] *= R.s[std::size_t(j)] * R.scale;
                }
            });
        }
    }
    return value(jobj{{"rotated_tensors", value(rotated)}, {"d_model", value(d)}, {"heads", value(heads)}});
}

// Cross-layer equalisation: the input columns of w_down (and wo) get the mean column norm; w_up rows (wv rows)
// absorb the inverse scale. Exact: silu(g) * (s u) = s (silu(g) u), attention is linear in v.
inline value equalize_model(Model& m, double alpha = 1.0) {
    i64 pairs = 0;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        const auto& L = m.spec.layers[std::size_t(l)];
        const std::string p = std::format("layer.{}.", l);
        auto eq = [&](float* up, float* down, i64 f, i64 d, i64 down_cols, i64 col0) {
            // up [f, d]; down [d, down_cols], neurons at columns col0..col0+f
            std::vector<double> cn(static_cast<std::size_t>(f), 0.0);
            for (i64 r = 0; r < d; ++r) for (i64 j = 0; j < f; ++j) { double x = down[r * down_cols + col0 + j]; cn[std::size_t(j)] += x * x; }
            double mean = 0; for (auto& c : cn) { c = std::sqrt(c); mean += c; } mean /= double(f);
            std::vector<float> s(static_cast<std::size_t>(f));
            for (i64 j = 0; j < f; ++j) s[std::size_t(j)] = cn[std::size_t(j)] > 0 ? float(std::pow(cn[std::size_t(j)] / mean, alpha)) : 1.f;
            for (i64 j = 0; j < f; ++j) for (i64 c = 0; c < d; ++c) up[j * d + c] *= s[std::size_t(j)];
            for (i64 r = 0; r < d; ++r) for (i64 j = 0; j < f; ++j) down[r * down_cols + col0 + j] /= s[std::size_t(j)];
            ++pairs;
        };
        const i64 d = m.spec.d_model;
        if (L.ffn == LayerSpec::Ffn::dense) {
            auto& up = m.w.at(p + "w_up"); auto& dn = m.w.at(p + "w_down");
            eq(up.data.data(), dn.data.data(), up.shape[0], d, dn.shape[1], 0);
        } else if (L.ffn == LayerSpec::Ffn::moe) {
            auto& up = m.w.at(p + "experts.w_up"); auto& dn = m.w.at(p + "experts.w_down");
            const i64 E = up.shape[0], f = up.shape[1];
            for (i64 e = 0; e < E; ++e) eq(up.data.data() + e * f * d, dn.data.data() + e * d * f, f, d, f, 0);
        }
        if (L.attn && !m.spec.qkv_bias) {
            // value channel c of KV head k feeds wo columns (h * hd + c) of the query heads h in k's group
            auto& wv = m.w.at(p + "wv"); auto& wo = m.w.at(p + "wo");
            const i64 hd = m.spec.head_dim, nkv = m.spec.n_kv, g = m.spec.n_heads / nkv, cols = wo.shape[1];
            std::vector<double> cn(static_cast<std::size_t>(nkv * hd), 0.0);
            for (i64 r = 0; r < d; ++r) for (i64 h = 0; h < m.spec.n_heads; ++h) for (i64 c = 0; c < hd; ++c) { double x = wo.data[std::size_t(r * cols + h * hd + c)]; cn[std::size_t((h / g) * hd + c)] += x * x; }
            double mean = 0; for (auto& c : cn) { c = std::sqrt(c); mean += c; } mean /= double(cn.size());
            for (i64 k = 0; k < nkv; ++k) for (i64 c = 0; c < hd; ++c) {
                const double v = cn[std::size_t(k * hd + c)];
                const float s = v > 0 ? float(std::pow(v / mean, alpha)) : 1.f;
                for (i64 j = 0; j < d; ++j) wv.data[std::size_t((k * hd + c) * d + j)] *= s;
                for (i64 h = k * g; h < (k + 1) * g; ++h) for (i64 r = 0; r < d; ++r) wo.data[std::size_t(r * cols + h * hd + c)] /= s;
            }
            ++pairs;
        }
    }
    return value(jobj{{"equalized_pairs", value(pairs)}});
}

}  // namespace forge::compress
