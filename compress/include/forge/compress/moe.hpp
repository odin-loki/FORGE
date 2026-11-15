// Mixture-of-experts compression.
//   * expert pruning: drop the least salient experts of a layer (router rows and stacked expert slices removed)
//       freq : routed-token count                       (frequency pruning, e.g. NAEE / Lu et al., 2024)
//       gate : total routing weight
//       reap : mean over routed tokens of gate * ||expert output||   (REAP, Lasby et al., ICLR 2026)
//   * expert merging
//       avg  : routing-weighted parameter average of the merged experts (HC-SMoE / M-SMoE style baseline; hidden
//              neurons are not aligned, which is the weakness REAP points out)
//       ls   : "concatenate-then-select" (new here): the merged expert starts as the concatenation of both
//              experts' neurons with down-projections scaled by their routing share, then greedy structured OBS
//              selects f neurons and refits w_down by least squares to the pair's true gated output on the
//              tokens routed to either expert. No neuron alignment is needed.
//   * MoE -> dense conversion (new here, "router-expected dense distillation"): all experts' neurons form one wide
//     SwiGLU FFN; w_down is solved in closed form against the true MoE output (so the constant-gate
//     approximation is absorbed by the fit), and structured OBS on the activation Gram shrinks it to the target
//     width. Optionally per layer, keeping the layers that do not convert well as MoE.
#pragma once
#include <forge/compress/structured.hpp>

namespace forge::compress {

enum class ExpertScore { freq, gate, reap };
inline ExpertScore expert_score_from(const std::string& s) {
    if (s == "freq") return ExpertScore::freq;
    if (s == "gate") return ExpertScore::gate;
    if (s == "reap") return ExpertScore::reap;
    die("unknown expert score " + s);
}
inline std::vector<double> expert_scores(const LayerStats& S, ExpertScore sc) {
    std::vector<double> v;
    for (auto& e : S.experts) {
        if (sc == ExpertScore::freq) v.push_back(double(e.tokens));
        else if (sc == ExpertScore::gate) v.push_back(e.gate_mass);
        else v.push_back(e.tokens ? e.reap / double(e.tokens) : 0.0);
    }
    return v;
}

// Keeps the experts in `keep` (sorted) of layer l.
inline void keep_experts(Model& m, i64 l, const std::vector<i64>& keep) {
    const std::string p = std::format("layer.{}.", l);
    for (const char* n : {"experts.w_gate", "experts.w_up", "experts.w_down"}) {
        auto& t = m.w.at(p + n);
        const i64 per = t.shape[1] * t.shape[2];
        t.data = take_rows(t.data.data(), per, keep);
        t.shape[0] = i64(keep.size());
    }
    auto& r = m.w.at(p + "router");
    r.data = take_rows(r.data.data(), r.shape[1], keep);
    r.shape[0] = i64(keep.size());
    auto& L = m.spec.layers[std::size_t(l)];
    L.moe_k = std::min<i64>(L.moe_k, i64(keep.size()));
}

inline std::vector<i64> prune_experts(Model& m, i64 l, const LayerStats& S, i64 drop, ExpertScore sc) {
    auto s = expert_scores(S, sc);
    const i64 E = i64(s.size());
    std::vector<i64> idx(static_cast<std::size_t>(E)); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](i64 a, i64 b) { return s[std::size_t(a)] > s[std::size_t(b)]; });
    std::vector<i64> keep(idx.begin(), idx.begin() + std::max<i64>(1, E - drop));
    std::sort(keep.begin(), keep.end());
    keep_experts(m, l, keep);
    return keep;
}

// Merge plan: repeatedly fold the least salient expert into the most similar remaining one (cosine of mean outputs).
inline std::vector<std::pair<i64, i64>> plan_merges(const LayerStats& S, i64 drop, ExpertScore sc) {
    const i64 E = S.n_experts, d = i64(S.exp_mean_out.size()) / std::max<i64>(E, 1);
    auto s = expert_scores(S, sc);
    std::vector<std::uint8_t> alive(static_cast<std::size_t>(E), 1), used(static_cast<std::size_t>(E), 0);   // pairs are disjoint
    std::vector<std::pair<i64, i64>> plan;   // (absorbed, into)
    for (i64 q = 0; q < drop; ++q) {
        i64 lo = -1;
        for (i64 e = 0; e < E; ++e) if (alive[std::size_t(e)] && !used[std::size_t(e)] && (lo < 0 || s[std::size_t(e)] < s[std::size_t(lo)])) lo = e;
        if (lo < 0) break;
        i64 best = -1; double bc = -2;
        for (i64 e = 0; e < E; ++e) {
            if (!alive[std::size_t(e)] || used[std::size_t(e)] || e == lo) continue;
            double ab = 0, aa = 0, bb = 0;
            for (i64 j = 0; j < d; ++j) { double a = S.exp_mean_out[std::size_t(lo * d + j)], b = S.exp_mean_out[std::size_t(e * d + j)]; ab += a * b; aa += a * a; bb += b * b; }
            double c = ab / std::sqrt(std::max(aa * bb, 1e-30));
            if (c > bc) { bc = c; best = e; }
        }
        if (best < 0) break;
        plan.push_back({lo, best});
        alive[std::size_t(lo)] = 0;
        used[std::size_t(best)] = 1;
    }
    return plan;
}

// Statistics for fitting merged experts: for one (absorbed a, into b) pair, the concatenated activations [a_a ; a_b]
// on the tokens routed to either, and the routing-share-normalised target (g_a o_a + g_b o_b) / (g_a + g_b).
struct MergeFitStats { la::Gram G; la::Cross C; double t_energy = 0; double ga = 0, gb = 0; };

inline std::vector<MergeFitStats> merge_pass(Model& m, i64 l, const Hidden& in, const std::vector<std::pair<i64, i64>>& pairs) {
    const auto& L = m.spec.layers[std::size_t(l)];
    const std::string p = std::format("layer.{}.", l);
    const auto& wg = m.w.at(p + "experts.w_gate"); const auto& wu = m.w.at(p + "experts.w_up"); const auto& wd = m.w.at(p + "experts.w_down");
    const i64 E = wg.shape[0], f = wg.shape[1], d = wg.shape[2], T = in.seq;
    std::vector<MergeFitStats> st(pairs.size());
    for (auto& s : st) { s.G = la::Gram(2 * f); s.C = la::Cross(d, 2 * f); }
    EmitOptions eo; eo.mode = EmitOptions::Mode::block; eo.block = l;
    Runner r(m.spec, m.w, T, eo, true, [&](const std::string& w) { return layer_of(w) == l && weight_group(w) == "moe" ? std::string("moe") : std::string(); });
    const i64 k = std::min(L.moe_k, E);
    std::vector<i64> ids(static_cast<std::size_t>(k)); std::vector<float> wts(static_cast<std::size_t>(k));
    for (i64 i = 0; i < in.n; ++i) {
        r.run_hidden({in.at(i), std::size_t(T * d)});
        auto h = r.tap("tap:moe"); auto lg = r.tap("tap:moe.router");
        std::vector<float> gate(static_cast<std::size_t>(T * E), 0.f);
        for (i64 t = 0; t < T; ++t) {
            forge::ops::ref::moe_route(lg.data() + t * E, E, k, L.moe_norm, L.moe_scale, 0, ids.data(), wts.data());
            for (i64 j = 0; j < k; ++j) gate[std::size_t(t * E + ids[std::size_t(j)])] = wts[std::size_t(j)];
        }
        for (std::size_t q = 0; q < pairs.size(); ++q) {
            auto [ea, eb] = pairs[q];
            std::vector<i64> rows;
            for (i64 t = 0; t < T; ++t) if (gate[std::size_t(t * E + ea)] > 0 || gate[std::size_t(t * E + eb)] > 0) rows.push_back(t);
            if (rows.empty()) continue;
            const i64 n = i64(rows.size());
            std::vector<float> xs(static_cast<std::size_t>(n * d));
            for (i64 q2 = 0; q2 < n; ++q2) std::copy_n(h.data() + rows[std::size_t(q2)] * d, d, xs.data() + q2 * d);
            auto aa = swiglu_hidden(xs.data(), n, d, wg.data.data() + ea * f * d, wu.data.data() + ea * f * d, f);
            auto ab = swiglu_hidden(xs.data(), n, d, wg.data.data() + eb * f * d, wu.data.data() + eb * f * d, f);
            std::vector<float> oa(static_cast<std::size_t>(n * d)), ob(static_cast<std::size_t>(n * d));
            la::gemm_nt(aa.data(), wd.data.data() + ea * d * f, oa.data(), n, d, f);
            la::gemm_nt(ab.data(), wd.data.data() + eb * d * f, ob.data(), n, d, f);
            std::vector<float> A(static_cast<std::size_t>(n * 2 * f)), Y(static_cast<std::size_t>(n * d));
            for (i64 t = 0; t < n; ++t) {
                const float g1 = gate[std::size_t(rows[std::size_t(t)] * E + ea)], g2 = gate[std::size_t(rows[std::size_t(t)] * E + eb)];
                st[q].ga += g1; st[q].gb += g2;
                const float den = std::max(g1 + g2, 1e-12f);
                std::copy_n(aa.data() + t * f, f, A.data() + t * 2 * f);
                std::copy_n(ab.data() + t * f, f, A.data() + t * 2 * f + f);
                for (i64 j = 0; j < d; ++j) { float y = (g1 * oa[std::size_t(t * d + j)] + g2 * ob[std::size_t(t * d + j)]) / den; Y[std::size_t(t * d + j)] = y; st[q].t_energy += double(y) * y; }
            }
            st[q].G.add(A.data(), n);
            st[q].C.add(Y.data(), A.data(), n);
        }
    }
    return st;
}

enum class MergeMethod { avg, ls };

// Applies a merge plan to layer l. Returns the relative fit error per merge (ls) or 0 (avg).
inline std::vector<double> merge_experts(Model& m, i64 l, const std::vector<std::pair<i64, i64>>& plan, MergeMethod how,
                                         const std::vector<MergeFitStats>* fit, double damp = 1e-4) {
    const std::string p = std::format("layer.{}.", l);
    auto& wg = m.w.at(p + "experts.w_gate"); auto& wu = m.w.at(p + "experts.w_up"); auto& wd = m.w.at(p + "experts.w_down");
    auto& router = m.w.at(p + "router");
    const i64 E = wg.shape[0], f = wg.shape[1], d = wg.shape[2];
    std::vector<double> errs;
    std::vector<std::uint8_t> alive(static_cast<std::size_t>(E), 1);
    for (std::size_t q = 0; q < plan.size(); ++q) {
        auto [a, b] = plan[q];
        double ga = 0.5, gb = 0.5;
        if (fit && q < fit->size()) { double s = (*fit)[q].ga + (*fit)[q].gb; if (s > 0) { ga = (*fit)[q].ga / s; gb = (*fit)[q].gb / s; } }
        float* Ga = wg.data.data() + a * f * d; float* Gb = wg.data.data() + b * f * d;
        float* Ua = wu.data.data() + a * f * d; float* Ub = wu.data.data() + b * f * d;
        float* Da = wd.data.data() + a * d * f; float* Db = wd.data.data() + b * d * f;
        if (how == MergeMethod::avg) {
            for (i64 i = 0; i < f * d; ++i) { Gb[i] = float(ga * Ga[i] + gb * Gb[i]); Ub[i] = float(ga * Ua[i] + gb * Ub[i]); }
            for (i64 i = 0; i < d * f; ++i) Db[i] = float(ga * Da[i] + gb * Db[i]);
            errs.push_back(0);
        } else {
            const auto& S = (*fit)[q];
            // full-width least squares down projection over the 2f concatenated neurons, then OBS down to f
            auto W = la::ls_solve(S.C.C, d, S.G.H, damp);   // [d, 2f]
            std::vector<double> curve;
            auto kept = obs_columns(W.data(), d, 2 * f, S.G.H, f, damp, &curve, true);
            // error of the fitted expert relative to the target energy
            std::vector<float> Wk = take_cols(W.data(), d, 2 * f, kept);
            la::Sym Gk = la::submatrix(S.G.H, kept);
            std::vector<float> Gf = la::to_f32(Gk), WG(static_cast<std::size_t>(d * f));
            la::gemm_nt(Wk.data(), Gf.data(), WG.data(), d, f, f);
            double fitted = 0, cross = 0;
            for (std::size_t i = 0; i < WG.size(); ++i) fitted += double(Wk[i]) * WG[i];
            for (i64 r = 0; r < d; ++r) for (std::size_t j = 0; j < kept.size(); ++j) cross += double(Wk[std::size_t(r) * kept.size() + j]) * S.C.C[std::size_t(r * 2 * f + kept[j])];
            errs.push_back((S.t_energy - 2 * cross + fitted) / std::max(S.t_energy, 1e-30));
            std::vector<float> ng(static_cast<std::size_t>(f * d)), nu(static_cast<std::size_t>(f * d));
            for (std::size_t j = 0; j < kept.size(); ++j) {
                const i64 src = kept[j];
                const float* g = src < f ? Ga + src * d : Gb + (src - f) * d;
                const float* u = src < f ? Ua + src * d : Ub + (src - f) * d;
                std::copy_n(g, d, ng.data() + i64(j) * d); std::copy_n(u, d, nu.data() + i64(j) * d);
            }
            std::copy(ng.begin(), ng.end(), Gb); std::copy(nu.begin(), nu.end(), Ub);
            std::copy(Wk.begin(), Wk.end(), Db);
        }
        // router: the merged expert's logit row is the routing-weighted average of both rows
        float* ra = router.data.data() + a * d; float* rb = router.data.data() + b * d;
        for (i64 j = 0; j < d; ++j) rb[j] = float(ga * ra[j] + gb * rb[j]);
        alive[std::size_t(a)] = 0;
    }
    std::vector<i64> keep;
    for (i64 e = 0; e < E; ++e) if (alive[std::size_t(e)]) keep.push_back(e);
    keep_experts(m, l, keep);
    return errs;
}

// MoE -> dense for layer l: width `keep` neurons out of E*f. Returns the relative error vs the true MoE output.
inline double moe_to_dense(Model& m, i64 l, const LayerStats& S, i64 keep, double damp = 1e-4) {
    const std::string p = std::format("layer.{}.", l);
    const auto& wg = m.w.at(p + "experts.w_gate"); const auto& wu = m.w.at(p + "experts.w_up");
    const i64 E = wg.shape[0], f = wg.shape[1], d = wg.shape[2], n = E * f;
    keep = std::min(keep, n);
    auto W = la::ls_solve(S.dense_cross.C, d, S.dense_hid.H, damp);   // [d, E*f]
    std::vector<i64> kept;
    if (keep < n) kept = obs_columns(W.data(), d, n, S.dense_hid.H, keep, damp, nullptr, true);
    else { kept.resize(static_cast<std::size_t>(n)); std::iota(kept.begin(), kept.end(), 0); }
    std::vector<float> Wk = take_cols(W.data(), d, n, kept);
    // error vs the true output: ||Y||^2 - 2 tr(W C^T) + tr(W G W^T)
    la::Sym Gk = la::submatrix(S.dense_hid.H, kept);
    std::vector<float> Gf = la::to_f32(Gk), WG(static_cast<std::size_t>(d) * kept.size());
    la::gemm_nt(Wk.data(), Gf.data(), WG.data(), d, i64(kept.size()), i64(kept.size()));
    double fitted = 0, cross = 0;
    for (std::size_t i = 0; i < WG.size(); ++i) fitted += double(Wk[i]) * WG[i];
    for (i64 r = 0; r < d; ++r) for (std::size_t j = 0; j < kept.size(); ++j) cross += double(Wk[std::size_t(r) * kept.size() + j]) * S.dense_cross.C[std::size_t(r * n + kept[j])];
    const double err = (S.y_energy - 2 * cross + fitted) / std::max(S.y_energy, 1e-30);
    tensor G({i64(kept.size()), d}), U({i64(kept.size()), d}), D({d, i64(kept.size())});
    G.data = take_rows(wg.data.data(), d, kept);
    U.data = take_rows(wu.data.data(), d, kept);
    D.data = Wk;
    auto& L = m.spec.layers[std::size_t(l)];
    for (auto& nm : layer_tensors(m.spec, l)) if (nm.find("experts.") != std::string::npos || nm.ends_with("router")) m.w.t.erase(nm);
    L.ffn = LayerSpec::Ffn::dense;
    m.w.t[p + "w_gate"] = std::move(G); m.w.t[p + "w_up"] = std::move(U); m.w.t[p + "w_down"] = std::move(D);
    return err;
}

}  // namespace forge::compress
