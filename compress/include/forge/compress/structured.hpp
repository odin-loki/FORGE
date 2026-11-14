// Structured compression (the model gets genuinely smaller dense matrices, no sparse kernels needed):
//   * FFN width pruning (removes SwiGLU neurons: rows of w_gate / w_up, columns of w_down)
//       score  mag  : ||w_down[:,j]|| * (||w_gate[j]|| + ||w_up[j]||)
//              act  : H_jj * ||w_down[:,j]||^2                         (Wanda-sp style, FLAP-like)
//              obs  : greedy structured OBS on H (exact output-error increments, ZipLM/SlimGPT style); the
//                     surviving columns of w_down are updated optimally (= least-squares refit)
//       refit  ls   : closed-form least squares of w_down on the kept neurons (FORGE Cholesky solve)
//              mcr  : masked refit preserving an existing sparsity mask (for already-sparse models)
//   * Sublayer removal (depth pruning at attention / FFN granularity, BlockPruner-style), scored by
//       bi     : block influence 1 - cos(x_in, x_out) (ShortGPT)
//       probe  : the measured calibration loss with the sublayer switched off (FORGE forward passes)
//   * Linear FFN replacement (new here): a removed FFN is replaced by the rank-r linear map minimising the
//     output error under the measured input covariance, M_r = P_r C G^-1 with P_r the top-r eigenspace of
//     M G M^T (closed form, FORGE eigh) - a training-free variant of LLM-Streamline's lightweight replacement.
#pragma once
#include <forge/compress/prune.hpp>
#include <forge/linalg/linalg.hpp>

namespace forge::compress {

// Greedy structured OBS over the columns of W [rows, n] with input Gram H [n, n]; removes columns until `keep`
// remain. W is updated in place (removed columns are zero). Returns the kept columns (sorted) and fills
// `curve` with the cumulative relative error after each removal (curve[i] = error with i columns removed).
inline std::vector<i64> obs_columns(float* W, i64 rows, i64 n, const la::Sym& H, i64 keep, double damp,
                                    std::vector<double>* curve = nullptr, bool update = true) {
    la::Sym L = H;
    la::cholesky_damped(L, damp);
    la::Sym Hi = la::inverse_from_cholesky(L);
    std::vector<float> Wt(static_cast<std::size_t>(n * rows));   // neurons as rows
    for (i64 r = 0; r < rows; ++r) for (i64 j = 0; j < n; ++j) Wt[std::size_t(j * rows + r)] = W[r * n + j];
    double ref = 0;   // tr(W H W^T)
    {
        std::vector<float> Hf = la::to_f32(H), WH(static_cast<std::size_t>(rows * n));
        la::gemm_nt(W, Hf.data(), WH.data(), rows, n, n);
        for (std::size_t i = 0; i < WH.size(); ++i) ref += double(W[i]) * WH[i];
        ref = std::max(ref, 1e-30);
    }
    std::vector<std::uint8_t> alive(static_cast<std::size_t>(n), 1);
    i64 n_alive = n;
    double acc = 0;
    if (curve) { curve->assign(1, 0.0); }
    std::vector<double> cost(static_cast<std::size_t>(n));
    while (n_alive > keep) {
        la::parallel_for(static_cast<std::size_t>(n), 64, [&](std::size_t b, std::size_t e) {
            for (std::size_t j = b; j < e; ++j) {
                if (!alive[j]) { cost[j] = INFINITY; continue; }
                double s = 0; const float* w = Wt.data() + j * std::size_t(rows);
                for (i64 r = 0; r < rows; ++r) s += double(w[r]) * w[r];
                cost[j] = s / Hi(i64(j), i64(j));
            }
        });
        const i64 batch = std::max<i64>(1, std::min<i64>(32, (n_alive - keep) / 16));
        std::vector<i64> order(static_cast<std::size_t>(n)); std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + batch, order.end(), [&](i64 a, i64 b) { return cost[std::size_t(a)] < cost[std::size_t(b)]; });
        for (i64 q = 0; q < batch && n_alive > keep; ++q) {
            const i64 j = order[std::size_t(q)];
            // recompute the exact cost of j after the earlier removals of this batch
            double s = 0; const float* wj = Wt.data() + j * rows;
            for (i64 r = 0; r < rows; ++r) s += double(wj[r]) * wj[r];
            const double hjj = Hi(j, j);
            acc += s / hjj;
            if (update) {
                std::vector<float> wjc(wj, wj + rows);
                la::parallel_for(static_cast<std::size_t>(n), 64, [&](std::size_t b, std::size_t e) {
                    for (std::size_t i = b; i < e; ++i) {
                        if (!alive[i] || i64(i) == j) continue;
                        const float c = float(Hi(i64(i), j) / hjj);
                        float* wi = Wt.data() + i * std::size_t(rows);
                        for (i64 r = 0; r < rows; ++r) wi[r] -= c * wjc[std::size_t(r)];
                    }
                });
            }
            std::vector<double> hj(Hi.row(j), Hi.row(j) + n);
            la::parallel_for(static_cast<std::size_t>(n), 32, [&](std::size_t b, std::size_t e) {
                for (std::size_t i = b; i < e; ++i) {
                    if (!alive[i]) continue;
                    static const auto axpy = forge::simd::tier_a_active().axpy_f64;
                    axpy(-hj[i] / hjj, hj.data(), Hi.row(i64(i)), std::size_t(n));
                }
            });
            alive[std::size_t(j)] = 0; --n_alive;
            std::fill_n(Wt.data() + j * rows, rows, 0.f);
            for (i64 k = 0; k < n; ++k) { Hi(j, k) = 0; Hi(k, j) = 0; }
            Hi(j, j) = 1;
            if (curve) curve->push_back(acc / ref);
        }
    }
    if (update) for (i64 r = 0; r < rows; ++r) for (i64 j = 0; j < n; ++j) W[r * n + j] = Wt[std::size_t(j * rows + r)];
    std::vector<i64> kept;
    for (i64 j = 0; j < n; ++j) if (alive[std::size_t(j)]) kept.push_back(j);
    return kept;
}

// Rows of a [r, c] matrix (or slices of a stacked [e, r, c] tensor along r) kept by `idx`.
inline std::vector<float> take_rows(const float* A, i64 c, const std::vector<i64>& idx) {
    std::vector<float> o(idx.size() * std::size_t(c));
    for (std::size_t i = 0; i < idx.size(); ++i) std::copy_n(A + idx[i] * c, c, o.data() + i64(i) * c);
    return o;
}
inline std::vector<float> take_cols(const float* A, i64 r, i64 c, const std::vector<i64>& idx) {
    std::vector<float> o(static_cast<std::size_t>(r) * idx.size());
    for (i64 i = 0; i < r; ++i) for (std::size_t j = 0; j < idx.size(); ++j) o[std::size_t(i) * idx.size() + j] = A[i * c + idx[j]];
    return o;
}

enum class WidthScore { mag, act, obs };
inline WidthScore width_score_from(const std::string& s) {
    if (s == "mag") return WidthScore::mag;
    if (s == "act") return WidthScore::act;
    if (s == "obs") return WidthScore::obs;
    die("unknown width score " + s);
}

struct WidthResult { std::vector<i64> kept; double err = 0; };

// Prunes the dense FFN of layer l to `keep` neurons, given the Gram of the w_down input. `sparse_refit` keeps the
// existing zero pattern of the surviving weights (refit by masked CG instead of a dense update).
inline WidthResult prune_ffn_width(Model& m, i64 l, const la::Sym& Hd, i64 keep, WidthScore sc, double damp, bool sparse_refit) {
    const std::string p = std::format("layer.{}.", l);
    auto& wg = m.w.at(p + "w_gate"); auto& wu = m.w.at(p + "w_up"); auto& wd = m.w.at(p + "w_down");
    const i64 f = wg.shape[0], d = wg.shape[1];
    if (keep >= f) return {{}, 0.0};
    std::vector<float> W0 = wd.data;
    std::vector<i64> kept;
    WidthResult res;
    if (sc == WidthScore::obs) {
        std::vector<float> W = wd.data;
        kept = obs_columns(W.data(), d, f, Hd, keep, damp, nullptr, !sparse_refit);
        if (!sparse_refit) { wd.data = W; }
    } else {
        std::vector<double> score(static_cast<std::size_t>(f));
        for (i64 j = 0; j < f; ++j) {
            double dn = 0; for (i64 r = 0; r < d; ++r) dn += double(wd.data[std::size_t(r * f + j)]) * wd.data[std::size_t(r * f + j)];
            if (sc == WidthScore::act) score[std::size_t(j)] = dn * Hd(j, j);
            else {
                double gn = 0, un = 0;
                for (i64 c = 0; c < d; ++c) { gn += double(wg.data[std::size_t(j * d + c)]) * wg.data[std::size_t(j * d + c)]; un += double(wu.data[std::size_t(j * d + c)]) * wu.data[std::size_t(j * d + c)]; }
                score[std::size_t(j)] = std::sqrt(dn) * (std::sqrt(gn) + std::sqrt(un));
            }
        }
        std::vector<i64> idx(static_cast<std::size_t>(f)); std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), [&](i64 a, i64 b) { return score[std::size_t(a)] > score[std::size_t(b)]; });
        kept.assign(idx.begin(), idx.begin() + keep);
        std::sort(kept.begin(), kept.end());
        if (!sparse_refit) {   // dense least-squares refit of w_down on the kept neurons
            auto Wn = la::ls_refit_columns(W0.data(), d, Hd, kept, damp);
            std::vector<float> full(static_cast<std::size_t>(d * f), 0.f);
            for (i64 r = 0; r < d; ++r) for (std::size_t j = 0; j < kept.size(); ++j) full[std::size_t(r * f + kept[j])] = Wn[std::size_t(r) * kept.size() + j];
            wd.data = full;
        }
    }
    if (sparse_refit) {   // keep the zero pattern; drop the removed columns; refit the rest
        std::vector<std::uint8_t> mask = mask_of(W0.data(), W0.size());
        std::vector<std::uint8_t> keepc(static_cast<std::size_t>(f), 0); for (i64 j : kept) keepc[std::size_t(j)] = 1;
        for (i64 r = 0; r < d; ++r) for (i64 j = 0; j < f; ++j) if (!keepc[std::size_t(j)]) mask[std::size_t(r * f + j)] = 0;
        std::vector<float> W = W0;
        for (std::size_t i = 0; i < W.size(); ++i) if (!mask[i]) W[i] = 0;
        mcr_refit(W.data(), W0.data(), d, f, Hd, mask, 30, damp);
        wd.data = W;
    }
    res.err = rel_error(wd.data.data(), W0.data(), d, f, Hd);
    // shrink the tensors
    wd.data = take_cols(wd.data.data(), d, f, kept); wd.shape = {d, i64(kept.size())};
    wg.data = take_rows(wg.data.data(), d, kept); wg.shape = {i64(kept.size()), d};
    wu.data = take_rows(wu.data.data(), d, kept); wu.shape = {i64(kept.size()), d};
    res.kept = std::move(kept);
    return res;
}

// ------------------------------------------------------------------------------------------------- sublayers
struct Sublayer { i64 layer; bool attn; std::string key() const { return std::format("{}{}", attn ? "attn" : "ffn", layer); } };

// Switches a sublayer off in place by zeroing its output projection (exactly removes its contribution); returns the
// saved tensor so it can be restored. Used for loss probes without recompiling the FORGE plan.
inline std::vector<float> zero_sublayer(Model& m, const Sublayer& s) {
    const auto& L = m.spec.layers[std::size_t(s.layer)];
    std::string n = std::format("layer.{}.", s.layer);
    if (s.attn) n += "wo";
    else if (L.ffn == LayerSpec::Ffn::moe) n += "experts.w_down";
    else if (L.ffn == LayerSpec::Ffn::linear) n += "lin_b";
    else n += "w_down";
    auto& t = m.w.at(n);
    std::vector<float> saved = t.data;
    std::fill(t.data.begin(), t.data.end(), 0.f);
    return saved;
}
inline void restore_sublayer(Model& m, const Sublayer& s, std::vector<float> saved) {
    const auto& L = m.spec.layers[std::size_t(s.layer)];
    std::string n = std::format("layer.{}.", s.layer);
    if (s.attn) n += "wo";
    else if (L.ffn == LayerSpec::Ffn::moe) n += "experts.w_down";
    else if (L.ffn == LayerSpec::Ffn::linear) n += "lin_b";
    else n += "w_down";
    m.w.at(n).data = std::move(saved);
}

// Structural removal (the tensors are dropped and the blueprint loses the sublayer).
inline void remove_sublayer(Model& m, const Sublayer& s) {
    auto& L = m.spec.layers[std::size_t(s.layer)];
    auto before = layer_tensors(m.spec, s.layer);
    if (s.attn) L.attn = false; else L.ffn = LayerSpec::Ffn::none;
    auto after = layer_tensors(m.spec, s.layer);
    for (auto& n : before) {
        if (std::find(after.begin(), after.end(), n) != after.end()) continue;
        m.spec.alias.erase(n);
        // a tensor other layers still read through an alias (a shared basis) moves to the first of them
        std::vector<std::string> readers;
        for (auto& [k, v] : m.spec.alias) if (v == n) readers.push_back(k);
        if (!readers.empty() && m.w.has(n)) {
            const std::string heir = readers.front();
            m.w.t[heir] = m.w.at(n);
            m.spec.alias.erase(heir);
            for (auto& k : readers) if (k != heir) m.spec.alias[k] = heir;
        }
        m.w.t.erase(n);
    }
}

// Replaces the FFN of layer l by the best rank-r linear map of its normalised input (ffn_norm kept).
inline double replace_ffn_linear(Model& m, i64 l, const LayerStats& S, i64 rank, double damp = 1e-4) {
    const i64 d = m.spec.d_model;
    auto M = la::ls_solve(S.ffn_cross.C, d, S.ffn_in.H, damp);   // [d, d]: y ~ M h
    // P_r = top-r eigenspace of M G M^T
    std::vector<float> Gf = la::to_f32(S.ffn_in.H), MG(static_cast<std::size_t>(d * d)), MGM(static_cast<std::size_t>(d * d));
    la::gemm_nt(M.data(), Gf.data(), MG.data(), d, d, d);   // M G (G symmetric)
    la::gemm_nt(MG.data(), M.data(), MGM.data(), d, d, d);  // (M G) M^T
    forge::linalg::mat<double> A(static_cast<std::size_t>(d), std::size_t(d));
    for (i64 i = 0; i < d; ++i) for (i64 j = 0; j < d; ++j) A(static_cast<std::size_t>(i), std::size_t(j)) = 0.5 * (double(MGM[std::size_t(i * d + j)]) + MGM[std::size_t(j * d + i)]);
    std::vector<double> w; forge::linalg::mat<double> V;
    forge::linalg::eigh(A, w, V);   // ascending; eigenvectors in columns
    rank = std::min(rank, d);
    std::vector<float> B(static_cast<std::size_t>(d * rank)), Ar(static_cast<std::size_t>(rank * d), 0.f);
    for (i64 k = 0; k < rank; ++k) {
        const std::size_t col = std::size_t(d - 1 - k);
        for (i64 i = 0; i < d; ++i) B[std::size_t(i * rank + k)] = float(V(static_cast<std::size_t>(i), col));
    }
    // A_r = U_r^T M
    for (i64 k = 0; k < rank; ++k) for (i64 j = 0; j < d; ++j) {
        double s = 0; for (i64 i = 0; i < d; ++i) s += double(B[std::size_t(i * rank + k)]) * M[std::size_t(i * d + j)];
        Ar[std::size_t(k * d + j)] = float(s);
    }
    // explained fraction of the FFN output energy (diagnostic)
    double tot = 0, kept = 0; for (double x : w) tot += std::max(x, 0.0);
    for (i64 k = 0; k < rank; ++k) kept += std::max(w[std::size_t(d - 1 - k)], 0.0);
    auto& L = m.spec.layers[std::size_t(l)];
    auto before = layer_tensors(m.spec, l);
    L.ffn = LayerSpec::Ffn::linear;
    auto after = layer_tensors(m.spec, l);
    for (auto& n : before) if (std::find(after.begin(), after.end(), n) == after.end()) m.w.t.erase(n);
    m.w.t[std::format("layer.{}.lin_a", l)] = tensor({rank, d}); m.w.t[std::format("layer.{}.lin_a", l)].data = Ar;
    m.w.t[std::format("layer.{}.lin_b", l)] = tensor({d, rank}); m.w.t[std::format("layer.{}.lin_b", l)].data = B;
    return tot > 0 ? kept / tot : 0.0;
}

}  // namespace forge::compress
