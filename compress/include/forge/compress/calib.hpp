// Calibration: token samples, the hidden states of the calibration set as they flow block by block through the
// model (FORGE block graphs), and the per-layer statistics compression needs:
//   * Gram matrices H = sum x x^T of every linear layer's input (second-order information: SparseGPT, OBS, refits)
//   * column energies diag(H) (Wanda / RIA / OWL)
//   * MoE routing (FORGE's moe_route), per-expert Gram matrices of the routed inputs and of the expert hidden
//     activations, expert usage, gate mass and REAP saliency
//   * residual-stream energies and sublayer cosine influence (depth pruning)
#pragma once
#include <forge/compress/linalg.hpp>
#include <forge/ops/reference.hpp>

#include <fstream>
#include <random>
#include <sstream>

namespace forge::compress {

// ---------------------------------------------------------------------------------------------------- data
inline std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) die("cannot read " + p);
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

// Minimal JSON-lines reader on FORGE's json parser.
inline std::vector<forge::json::value> read_jsonl(const std::string& p, std::size_t limit = SIZE_MAX) {
    std::vector<forge::json::value> v;
    std::ifstream f(p);
    if (!f) die("cannot read " + p);
    std::string line;
    while (v.size() < limit && std::getline(f, line)) if (!line.empty()) v.push_back(forge::json::parse(line));
    return v;
}

struct CalibSet {
    i64 seq = 0;
    std::vector<std::vector<int>> seqs;
    i64 tokens() const { return i64(seqs.size()) * seq; }
};

// SparseGPT/Wanda-style sampling: random documents long enough for a window, random window start (seeded).
// `source` is a .jsonl with {"text"} documents or a plain text file (windows of the whole tokenised text).
inline CalibSet make_calib(const Model& m, const std::string& source, i64 n, i64 seq, std::uint64_t seed = 0) {
    CalibSet c; c.seq = seq;
    std::mt19937_64 rng(seed);
    const bool bos = m.manifest.find("add_bos") && m.manifest["add_bos"].is_bool() && m.manifest["add_bos"].b();
    const int bos_id = m.manifest.find("bos_token_id") && m.manifest["bos_token_id"].is_num() ? int(m.manifest["bos_token_id"].num()) : -1;
    if (source.ends_with(".jsonl")) {
        auto docs = read_jsonl(source);
        std::uniform_int_distribution<std::size_t> pick(0, docs.size() - 1);
        for (int tries = 0; i64(c.seqs.size()) < n && tries < 100000; ++tries) {
            auto ids = m.encode(docs[pick(rng)]["text"].str());
            if (bos && bos_id >= 0) ids.insert(ids.begin(), bos_id);
            if (i64(ids.size()) <= seq) continue;
            std::uniform_int_distribution<i64> st(0, i64(ids.size()) - seq - 1);
            i64 s0 = st(rng);
            c.seqs.emplace_back(ids.begin() + s0, ids.begin() + s0 + seq);
        }
    } else {
        auto ids = m.encode(read_file(source));
        if (i64(ids.size()) < seq + 1) die("calibration text too short");
        std::uniform_int_distribution<i64> st(0, i64(ids.size()) - seq - 1);
        for (i64 i = 0; i < n; ++i) { i64 s0 = st(rng); c.seqs.emplace_back(ids.begin() + s0, ids.begin() + s0 + seq); }
    }
    if (i64(c.seqs.size()) < n) die(std::format("only {} calibration windows of {} tokens found", c.seqs.size(), seq));
    return c;
}

// ---------------------------------------------------------------------------------------------------- hidden
// The residual stream of every calibration sequence at one depth: [n_seq, seq, d].
struct Hidden {
    i64 n = 0, seq = 0, d = 0;
    std::vector<float> x;
    float* at(i64 i) { return x.data() + i * seq * d; }
    const float* at(i64 i) const { return x.data() + i * seq * d; }
    static Hidden embed(const Model& m, const CalibSet& c) {
        Hidden h; h.n = i64(c.seqs.size()); h.seq = c.seq; h.d = m.spec.d_model;
        h.x.resize(static_cast<std::size_t>(h.n * h.seq * h.d));
        const auto& E = m.w.at("tok_emb");
        for (i64 i = 0; i < h.n; ++i) for (i64 t = 0; t < h.seq; ++t)
            std::copy_n(E.data.data() + i64(c.seqs[std::size_t(i)][std::size_t(t)]) * h.d, h.d, h.at(i) + t * h.d);
        return h;
    }
};

// Which statistics a pass collects.
struct StatsReq {
    bool gram = true;             // full Gram matrices (else diagonals only)
    bool moe = false;             // routing + per-expert statistics
    bool moe_dense_acts = false;  // all experts on all tokens + true MoE output (MoE -> dense conversion)
    bool sublayer = false;        // cosine influence of the attention / FFN sublayers
    bool ffn_io = false;          // FFN input (normalised) and output: least-squares replacement of the FFN
    bool moe_means = false;       // each expert's mean output over all tokens (merge partner similarity)
    std::set<std::string> only;   // when non-empty: Gram matrices only for these groups (diagonals for all)
};

struct ExpertStats {
    i64 tokens = 0;         // routed (token, slot) pairs
    double gate_mass = 0;   // sum of routing weights
    double reap = 0;        // sum of gate * ||expert output||  (REAP saliency numerator)
    double out_energy = 0;  // sum ||expert output||^2
    la::Gram in;            // routed inputs h (gate/up)
    la::Gram hid;           // expert hidden activations silu(g) * u (down)
};

struct LayerStats {
    i64 layer = -1;
    std::map<std::string, la::Gram> gram;        // group -> Gram of its input ("attn", "o", "mlp", "down", "moe")
    std::map<std::string, std::vector<double>> diag;
    double resid_in = 0, resid_out = 0;          // sum ||x||^2 entering / leaving the block
    double cos_attn = 0, cos_ffn = 0, cos_block = 0;   // sums of cos(x_in, x_out) per token
    i64 tokens = 0;
    // MoE
    std::vector<ExpertStats> experts;
    i64 n_experts = 0;
    std::vector<double> coact;                   // [E*E] co-selection counts
    la::Gram dense_hid;                          // all experts' activations concatenated [E*f]
    la::Cross dense_cross;                       // MoE output^T x dense activations
    double y_energy = 0;                         // sum ||MoE output||^2
    std::vector<double> exp_mean_out;            // [E*d] mean output of each expert over all tokens
    // FFN replacement
    la::Gram ffn_in;
    la::Cross ffn_cross;                         // FFN output^T x FFN input
    std::vector<double> ffn_out_mean, ffn_in_mean;
};

// Group of a weight name in its layer: inputs shared by several projections share a Gram matrix.
inline std::string weight_group(const std::string& w) {
    auto leaf = w.substr(w.find('.', 6) + 1);   // after "layer.N."
    if (leaf.ends_with("_b") && leaf != "lin_b") return "lr:" + leaf.substr(0, leaf.size() - 2);   // input of a factor's 2nd half
    if (leaf.ends_with("_a") && leaf != "lin_a") leaf = leaf.substr(0, leaf.size() - 2);           // same input as the full projection
    if (leaf == "wq" || leaf == "wk" || leaf == "wv") return "attn";
    if (leaf == "wo") return "o";
    if (leaf == "w_gate" || leaf == "w_up") return "mlp";
    if (leaf == "w_down") return "down";
    if (leaf == "experts.w_gate") return "moe";
    if (leaf == "lin_a") return "lin";
    if (leaf == "ffn_norm") return "ffn_x";
    return "";
}
inline i64 layer_of(const std::string& w) { return w.rfind("layer.", 0) == 0 ? std::stoll(w.substr(6)) : -1; }
inline std::string leaf_of(const std::string& w) { return w.substr(w.find('.', 6) + 1); }

inline void add_diag(std::vector<double>& dg, const float* X, i64 rows, i64 d) {
    if (dg.empty()) dg.assign(static_cast<std::size_t>(d), 0.0);
    la::parallel_for(static_cast<std::size_t>(d), 64, [&](std::size_t b, std::size_t e) {
        for (i64 r = 0; r < rows; ++r) { const float* x = X + r * d; for (std::size_t j = b; j < e; ++j) dg[j] += double(x[j]) * x[j]; }
    });
}

// Per-token SwiGLU hidden activations silu(X Wg^T) * (X Wu^T) for one expert / FFN (FORGE GEMM).
inline std::vector<float> swiglu_hidden(const float* X, i64 rows, i64 d, const float* Wg, const float* Wu, i64 f) {
    std::vector<float> g(static_cast<std::size_t>(rows * f)), u(static_cast<std::size_t>(rows * f));
    la::gemm_nt(X, Wg, g.data(), rows, f, d);
    la::gemm_nt(X, Wu, u.data(), rows, f, d);
    la::parallel_for(static_cast<std::size_t>(rows * f), 4096, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) { float a = g[i]; g[i] = a / (1.f + std::exp(-a)) * u[i]; }
    });
    return g;
}

inline double cos_sum(const float* a, const float* b, i64 rows, i64 d) {
    double s = 0;
    for (i64 r = 0; r < rows; ++r) {
        double ab = 0, aa = 0, bb = 0;
        for (i64 j = 0; j < d; ++j) { ab += double(a[r * d + j]) * b[r * d + j]; aa += double(a[r * d + j]) * a[r * d + j]; bb += double(b[r * d + j]) * b[r * d + j]; }
        s += ab / std::sqrt(std::max(aa * bb, 1e-30));
    }
    return s;
}

// Runs layer `l` (as one FORGE block graph) over all calibration sequences, collecting `req`; if `next` is given it
// receives the block outputs (the hidden states entering layer l+1).
inline LayerStats layer_pass(Model& m, i64 l, const Hidden& in, const StatsReq& req, Hidden* next) {
    LayerStats S; S.layer = l;
    const auto& L = m.spec.layers[std::size_t(l)];
    const i64 d = m.spec.d_model, T = in.seq;
    const std::string pre = std::format("layer.{}.", l);
    EmitOptions eo; eo.mode = EmitOptions::Mode::block; eo.block = l;
    Runner r(m.spec, m.w, T, eo, true, [&](const std::string& w) -> std::string {
        if (layer_of(w) != l) return "";
        auto g = weight_group(w);
        if (g == "ffn_x" && !req.sublayer && !req.ffn_io) return "";
        return g;
    });
    if (next) { next->n = in.n; next->seq = T; next->d = d; next->x.resize(in.x.size()); }
    const bool moe = L.ffn == LayerSpec::Ffn::moe;
    i64 E = 0, f = 0;
    if (moe) {
        const auto& wg = m.w.at(pre + "experts.w_gate");
        E = wg.shape[0]; f = wg.shape[1];
        S.n_experts = E;
        if (req.moe) {
            S.experts.resize(static_cast<std::size_t>(E));
            for (auto& e : S.experts) { e.in = la::Gram(req.gram ? d : 0); e.hid = la::Gram(req.gram ? f : 0); }
            S.coact.assign(static_cast<std::size_t>(E * E), 0.0);
        }
        if (req.moe_dense_acts) { S.dense_hid = la::Gram(E * f); S.dense_cross = la::Cross(d, E * f); }
        if (req.moe_dense_acts || req.moe_means) S.exp_mean_out.assign(static_cast<std::size_t>(E * d), 0.0);
    }
    if (req.ffn_io) { S.ffn_in = la::Gram(d); S.ffn_cross = la::Cross(d, d); S.ffn_out_mean.assign(static_cast<std::size_t>(d), 0); S.ffn_in_mean.assign(static_cast<std::size_t>(d), 0); }

    std::set<std::string> groups;
    for (auto& [w, o] : r.tap_of) groups.insert(o);
    for (i64 i = 0; i < in.n; ++i) {
        r.run_hidden({in.at(i), std::size_t(T * d)});
        auto out = r.output();
        if (next) std::copy(out.begin(), out.end(), next->at(i));
        const float* xi = in.at(i);
        for (i64 t = 0; t < T; ++t) for (i64 j = 0; j < d; ++j) { S.resid_in += double(xi[t * d + j]) * xi[t * d + j]; S.resid_out += double(out[std::size_t(t * d + j)]) * out[std::size_t(t * d + j)]; }
        S.cos_block += cos_sum(xi, out.data(), T, d);
        S.tokens += T;
        for (auto& gname : groups) {
            const std::string g = gname.substr(4);   // strip "tap:"
            auto x = r.tap(gname);
            const i64 dim = i64(x.size()) / T;
            if (g == "ffn_x") continue;
            add_diag(S.diag[g], x.data(), T, dim);
            if (req.gram && (req.only.empty() || req.only.count(g))) {
                auto it = S.gram.find(g);
                if (it == S.gram.end()) it = S.gram.emplace(g, la::Gram(dim)).first;
                it->second.add(x.data(), T);
            }
        }
        // sublayer influence: x_in -> x_mid (after attention) -> x_out
        if ((req.sublayer || req.ffn_io) && r.has_tap("tap:ffn_x")) {
            auto xm = r.tap("tap:ffn_x");
            if (L.attn) S.cos_attn += cos_sum(xi, xm.data(), T, d);
            S.cos_ffn += cos_sum(xm.data(), out.data(), T, d);
            if (req.ffn_io) {
                std::vector<float> y(static_cast<std::size_t>(T * d)), hn(static_cast<std::size_t>(T * d));
                const auto& g = m.w.at(pre + "ffn_norm");
                for (i64 t = 0; t < T; ++t) {
                    double ss = 0; for (i64 j = 0; j < d; ++j) ss += double(xm[std::size_t(t * d + j)]) * xm[std::size_t(t * d + j)];
                    const float inv = float(1.0 / std::sqrt(ss / double(d) + m.spec.eps));
                    for (i64 j = 0; j < d; ++j) {
                        const std::size_t k = std::size_t(t * d + j);
                        y[k] = out[k] - xm[k];
                        hn[k] = xm[k] * inv * g.data[std::size_t(j)];
                        S.ffn_out_mean[std::size_t(j)] += y[k]; S.ffn_in_mean[std::size_t(j)] += hn[k];
                    }
                }
                S.ffn_in.add(hn.data(), T);
                S.ffn_cross.add(y.data(), hn.data(), T);
            }
        } else if (!L.attn) S.cos_ffn += cos_sum(xi, out.data(), T, d);
        else if (L.ffn == LayerSpec::Ffn::none) S.cos_attn += cos_sum(xi, out.data(), T, d);

        if (moe && (req.moe || req.moe_dense_acts || req.moe_means)) {
            auto h = r.tap("tap:moe");
            auto lg = r.tap("tap:moe.router");
            const auto& wg = m.w.at(pre + "experts.w_gate");
            const auto& wu = m.w.at(pre + "experts.w_up");
            const auto& wd = m.w.at(pre + "experts.w_down");
            const i64 k = std::min(L.moe_k, E);
            std::vector<std::vector<i64>> rows(static_cast<std::size_t>(E));
            std::vector<std::vector<float>> gates(static_cast<std::size_t>(E));
            std::vector<float> gate_full(static_cast<std::size_t>(T * E), 0.f);
            std::vector<i64> ids(static_cast<std::size_t>(k)); std::vector<float> wts(static_cast<std::size_t>(k));
            for (i64 t = 0; t < T; ++t) {
                forge::ops::ref::moe_route(lg.data() + t * E, E, k, L.moe_norm, L.moe_scale, 0, ids.data(), wts.data());
                for (i64 j = 0; j < k; ++j) {
                    rows[std::size_t(ids[std::size_t(j)])].push_back(t);
                    gates[std::size_t(ids[std::size_t(j)])].push_back(wts[std::size_t(j)]);
                    gate_full[std::size_t(t * E + ids[std::size_t(j)])] = wts[std::size_t(j)];
                    if (req.moe) for (i64 j2 = 0; j2 < k; ++j2) S.coact[std::size_t(ids[std::size_t(j)] * E + ids[std::size_t(j2)])] += 1;
                }
            }
            if (req.moe) {
                for (i64 e = 0; e < E; ++e) {
                    const auto& R = rows[std::size_t(e)];
                    if (R.empty()) continue;
                    const i64 n = i64(R.size());
                    std::vector<float> xs(static_cast<std::size_t>(n * d));
                    for (i64 q = 0; q < n; ++q) std::copy_n(h.data() + R[std::size_t(q)] * d, d, xs.data() + q * d);
                    auto a = swiglu_hidden(xs.data(), n, d, wg.data.data() + e * f * d, wu.data.data() + e * f * d, f);
                    std::vector<float> o(static_cast<std::size_t>(n * d));
                    la::gemm_nt(a.data(), wd.data.data() + e * d * f, o.data(), n, d, f);
                    auto& es = S.experts[std::size_t(e)];
                    es.tokens += n;
                    for (i64 q = 0; q < n; ++q) {
                        double nn = 0; for (i64 j = 0; j < d; ++j) nn += double(o[std::size_t(q * d + j)]) * o[std::size_t(q * d + j)];
                        const double gq = gates[std::size_t(e)][std::size_t(q)];
                        es.gate_mass += gq; es.reap += gq * std::sqrt(nn); es.out_energy += nn;
                    }
                    if (req.gram) { es.in.add(xs.data(), n); es.hid.add(a.data(), n); }
                }
            }
            if (req.moe_dense_acts || req.moe_means) {
                // every expert on every token: the activations a dense FFN built from all experts would see
                std::vector<float> A(static_cast<std::size_t>(T * E * f)), Y(static_cast<std::size_t>(T * d), 0.f);
                for (i64 e = 0; e < E; ++e) {
                    auto a = swiglu_hidden(h.data(), T, d, wg.data.data() + e * f * d, wu.data.data() + e * f * d, f);
                    std::vector<float> o(static_cast<std::size_t>(T * d));
                    la::gemm_nt(a.data(), wd.data.data() + e * d * f, o.data(), T, d, f);
                    for (i64 t = 0; t < T; ++t) {
                        std::copy_n(a.data() + t * f, f, A.data() + t * E * f + e * f);
                        for (i64 j = 0; j < d; ++j) S.exp_mean_out[std::size_t(e * d + j)] += o[std::size_t(t * d + j)];
                        const float gt = gate_full[std::size_t(t * E + e)];
                        if (gt != 0.f) for (i64 j = 0; j < d; ++j) Y[std::size_t(t * d + j)] += gt * o[std::size_t(t * d + j)];
                    }
                }
                if (req.moe_dense_acts) {
                    for (float v : Y) S.y_energy += double(v) * v;
                    S.dense_hid.add(A.data(), T);
                    S.dense_cross.add(Y.data(), A.data(), T);
                }
            }
        }
    }
    return S;
}

// Propagates the hidden states through layer l without collecting statistics.
inline void layer_forward(Model& m, i64 l, const Hidden& in, Hidden& out) {
    EmitOptions eo; eo.mode = EmitOptions::Mode::block; eo.block = l;
    Runner r(m.spec, m.w, in.seq, eo, true);
    out.n = in.n; out.seq = in.seq; out.d = in.d; out.x.resize(in.x.size());
    for (i64 i = 0; i < in.n; ++i) {
        r.run_hidden({in.at(i), std::size_t(in.seq * in.d)});
        auto o = r.output();
        std::copy(o.begin(), o.end(), out.at(i));
    }
}

}  // namespace forge::compress
