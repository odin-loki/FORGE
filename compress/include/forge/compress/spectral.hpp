// Spectral structure of trained weights: diagnostics, spectrum-driven sparsity allocation and low-rank factoring.
//
//   diagnostics   the empirical spectral density (ESD) of every projection, eigenvalues of W W^T / N
//                 (N = the larger dimension); the heavy-tail exponent alpha (Hill estimator over the top of the
//                 ESD, as in HT-SR / WeightWatcher and AlphaPruning, Lu et al. 2024); the Marchenko-Pastur bulk
//                 edge lambda+ = sigma^2 (1 + sqrt(gamma))^2 with sigma^2 estimated from the median eigenvalue
//                 (Gavish-Donoho style), the signal rank (eigenvalues above the edge) and the fraction of the
//                 squared Frobenius norm in the signal.
//   alloc=alpha   prune(...,alloc=alpha): AlphaPruning layer-wise sparsity. Smaller alpha (heavier tail, more
//                 learned structure) -> lower sparsity; alpha is mapped linearly onto [lo, hi] and shifted so the
//                 parameter-weighted mean sparsity equals the target. Data-free.
//   lowrank       factors projections into X_a [r, in], X_b [out, r] (W ~ X_b X_a) when that saves >= min_save of
//                 the projection's parameters. Rank selection:
//                   mp      data-free: the MP signal rank, truncated SVD of W
//                   energy  data-free: smallest rank keeping `keep` of ||W||_F^2
//                   whiten  data-aware (SVD-LLM / ASVD): truncated SVD of W L with H = L L^T the input Gram, which
//                           minimises ||(W - W_r) X|| exactly for the rank; rank by `keep` on the whitened spectrum
//                   mdl     minimum description length (NMP): argmin_r bits * r (out + in) + (N/2) log2(E_r / E_0)
//                           with E_r the (whitened when data=1) residual energy, floored at the storage precision
//                           E_0 = ||W||^2 2^(-2 bits); N = out * in
//                 The factors are a projection onto the top singular subspace: for out <= in, X_b = U_r,
//                 X_a = U_r^T W (U_r the top eigenvectors of W H W^T, H = I data-free); otherwise X_a = V_r^T L^-1,
//                 X_b = W L V_r. At full rank X_b X_a = W up to rounding.
//   share         cross-layer basis sharing (Basis Sharing, Wang et al. 2024; the "Grassmannian backbone"): for each
//                 group of `group` adjacent layers and each projection type, one output-side basis U (top-r left
//                 singular vectors of the stacked [W_1, W_2, ...], i.e. eigenvectors of sum_l W_l W_l^T; with data=1
//                 of sum_l W_l H_l W_l^T, H_l each layer's input Gram per token, which minimises the summed output
//                 error exactly) is shared: X_b = U (stored once in the group's first layer, aliased by the others)
//                 and X_a_l = U^T W_l per layer. (The shared factor is X_b, not X_a: the blueprint parser recovers
//                 a factored projection from its own X_a reference, so X_a must stay per layer.)
//   spectra       report only: the diagnostics of every projection.
//
// Recipe syntax:
//   lowrank   method=mp|energy|whiten|mdl  keep=<energy fraction, 0.9>  ratio=<max rank / min(out,in), 1>
//             rank=<fixed rank>  scope=attn|ffn|all  min_save=<0.1>  bits=<16, mdl>  data=0|1 (mdl whitening)
//             damp=<relative Gram damping, 1e-6>  edge=<MP edge multiplier, 1>
//   share     group=<2>  keep=<0.9>  ratio=<1>  rank=<r>  scope=attn|ffn|all  min_save=<0.1>  data=0|1
//   prune     ...,alloc=alpha  eps=<0.2>  alpha_lo=<min sparsity, s-eps>  alpha_hi=<max sparsity, s+eps>
//             alpha_unit=layer|proj  tail=<Hill tail fraction, 0.5>
//   spectra   scope=all  tail=0.5
#pragma once
#include <forge/compress/compress.hpp>
#include <forge/linalg/linalg.hpp>

#include <numbers>
#include <optional>

namespace forge::compress {

namespace spectral {

// ------------------------------------------------------------------------------------------------ helpers
// Eigendecomposition of a symmetric f32 matrix [n, n] in f64, eigenvalues descending, eigenvectors in columns.
struct EigDesc { std::vector<double> w; forge::linalg::mat<double> V; };
inline EigDesc eig_desc(const float* G, i64 n) {
    forge::linalg::mat<double> A(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) for (i64 j = 0; j < n; ++j) A(std::size_t(i), std::size_t(j)) = 0.5 * (double(G[i * n + j]) + double(G[j * n + i]));
    EigDesc e; std::vector<double> w; forge::linalg::mat<double> V;
    forge::linalg::eigh(A, w, V);   // ascending
    e.w.resize(std::size_t(n)); e.V = forge::linalg::mat<double>(std::size_t(n), std::size_t(n));
    for (i64 j = 0; j < n; ++j) {
        const std::size_t src = std::size_t(n - 1 - j);
        e.w[std::size_t(j)] = std::max(w[src], 0.0);
        for (i64 i = 0; i < n; ++i) e.V(std::size_t(i), std::size_t(j)) = V(std::size_t(i), src);
    }
    return e;
}

// Gram of the smaller side: W W^T [out, out] when out <= in, else W^T W [in, in].
inline std::vector<float> small_gram(const float* W, i64 out, i64 in) {
    const i64 p = std::min(out, in);
    std::vector<float> G(static_cast<std::size_t>(p * p));
    if (out <= in) la::gemm_nt(W, W, G.data(), out, out, in);
    else {
        std::vector<float> Wt(static_cast<std::size_t>(out * in));
        la::transpose(W, Wt.data(), out, in);
        la::gemm_nt(Wt.data(), Wt.data(), G.data(), in, in, out);
    }
    return G;
}

// Squared singular values of W (descending, min(out, in) of them).
inline std::vector<double> sq_singular_values(const float* W, i64 out, i64 in) {
    auto G = small_gram(W, out, in);
    return eig_desc(G.data(), std::min(out, in)).w;
}

// Median of the Marchenko-Pastur law with unit variance and aspect ratio gamma in (0, 1].
inline double mp_median(double gamma) {
    gamma = std::clamp(gamma, 1e-6, 1.0);
    const double a = (1 - std::sqrt(gamma)) * (1 - std::sqrt(gamma)), b = (1 + std::sqrt(gamma)) * (1 + std::sqrt(gamma));
    // x = a + (b - a) sin^2(phi): density * dx = (b-a)^2 2 sin^2 cos^2 / (2 pi gamma x) dphi (no endpoint singularity)
    constexpr int n = 4000;
    std::vector<double> cdf(n + 1, 0.0), xs(n + 1);
    const double h = (std::numbers::pi / 2) / n;
    for (int i = 0; i <= n; ++i) xs[std::size_t(i)] = a + (b - a) * std::pow(std::sin(i * h), 2);
    for (int i = 0; i < n; ++i) {
        const double ph = (i + 0.5) * h, s = std::sin(ph), c = std::cos(ph);
        const double x = a + (b - a) * s * s;
        const double f = (b - a) * (b - a) * 2 * s * s * c * c / (2 * std::numbers::pi * gamma * std::max(x, 1e-300));
        cdf[std::size_t(i) + 1] = cdf[std::size_t(i)] + f * h;
    }
    const double tot = cdf[n];
    for (int i = 0; i < n; ++i) if (cdf[std::size_t(i) + 1] >= 0.5 * tot) {
        const double t = (0.5 * tot - cdf[std::size_t(i)]) / std::max(cdf[std::size_t(i) + 1] - cdf[std::size_t(i)], 1e-300);
        return xs[std::size_t(i)] + t * (xs[std::size_t(i) + 1] - xs[std::size_t(i)]);
    }
    return b;
}

// ------------------------------------------------------------------------------------------------ diagnostics
struct Spectrum {
    std::vector<double> esd;     // eigenvalues of W W^T / N, descending (N = max(out, in))
    double alpha = 0;            // Hill heavy-tail exponent of the ESD
    double sigma2 = 0;           // robust (median-based) noise variance
    double gamma = 1;            // min / max dimension
    double lambda_plus = 0;      // MP bulk edge
    i64 mp_rank = 0;             // eigenvalues above the edge
    double signal_frac = 0;      // fraction of ||W||_F^2 above the edge
};

// Hill estimator of the density exponent alpha (p(lambda) ~ lambda^-alpha) over the top `tail` fraction.
inline double hill_alpha(const std::vector<double>& ev_desc, double tail = 0.5) {
    const i64 p = i64(ev_desc.size());
    if (p < 3) return 2.0;
    i64 k = std::clamp<i64>(i64(std::llround(tail * double(p))), 2, p - 1);
    const double xk = std::max(ev_desc[std::size_t(k)], 1e-300);
    double s = 0;
    for (i64 i = 0; i < k; ++i) s += std::log(std::max(ev_desc[std::size_t(i)], 1e-300) / xk);
    return 1.0 + double(k) / std::max(s, 1e-12);
}

inline Spectrum analyse_esd(std::vector<double> sq_sv, i64 out, i64 in, double tail = 0.5, double edge_mult = 1.0) {
    Spectrum S;
    const double N = double(std::max(out, in));
    S.gamma = double(std::min(out, in)) / N;
    for (auto& x : sq_sv) x /= N;
    S.esd = std::move(sq_sv);
    S.alpha = hill_alpha(S.esd, tail);
    std::vector<double> srt = S.esd;
    std::nth_element(srt.begin(), srt.begin() + std::ptrdiff_t(srt.size() / 2), srt.end());
    const double med = srt.empty() ? 0.0 : srt[srt.size() / 2];
    S.sigma2 = med / mp_median(S.gamma);
    S.lambda_plus = edge_mult * S.sigma2 * std::pow(1 + std::sqrt(S.gamma), 2);
    double tot = 0, sig = 0;
    for (double x : S.esd) { tot += x; if (x > S.lambda_plus) { sig += x; ++S.mp_rank; } }
    S.signal_frac = tot > 0 ? sig / tot : 0.0;
    return S;
}

inline Spectrum spectrum_of(const float* W, i64 out, i64 in, double tail = 0.5, double edge_mult = 1.0) {
    return analyse_esd(sq_singular_values(W, out, in), out, in, tail, edge_mult);
}

// ------------------------------------------------------------------------------------------------ rank selection
inline i64 energy_rank(const std::vector<double>& ev, double keep) {
    double tot = 0; for (double x : ev) tot += x;
    if (tot <= 0) return 1;
    double acc = 0;
    for (std::size_t i = 0; i < ev.size(); ++i) { acc += ev[i]; if (acc >= keep * tot * (1 - 1e-12)) return i64(i) + 1; }
    return i64(ev.size());
}

// MDL: bits * r (out + in) for the factors (capped at the dense out * in), plus the Gaussian code length of the
// residual relative to the storage precision floor.
inline i64 mdl_rank(const std::vector<double>& ev, i64 out, i64 in, double bits, std::vector<double>* cost = nullptr) {
    const i64 p = i64(ev.size());
    double tot = 0; for (double x : ev) tot += x;
    if (tot <= 0 || p == 0) return 1;
    const double N = double(out) * double(in), floor_e = tot * std::pow(2.0, -2.0 * bits);
    std::vector<double> tail(static_cast<std::size_t>(p) + 1, 0.0);
    for (i64 i = p - 1; i >= 0; --i) tail[std::size_t(i)] = tail[std::size_t(i) + 1] + ev[std::size_t(i)];
    i64 best = 1; double bc = INFINITY;
    if (cost) cost->assign(static_cast<std::size_t>(p) + 1, 0.0);
    for (i64 r = 1; r <= p; ++r) {
        const double params = std::min(double(r) * double(out + in), N);
        const double c = bits * params + 0.5 * N * std::log2(std::max(tail[std::size_t(r)], floor_e) / floor_e);
        if (cost) (*cost)[std::size_t(r)] = c;
        if (c < bc) { bc = c; best = r; }
    }
    return best;
}

// ------------------------------------------------------------------------------------------------ factoring
// The spectrum of W (whitened by H when given) and the factors of its rank-r projection.
struct Factoring {
    i64 out = 0, in = 0;
    bool left = true;                    // left basis (out <= in) or right basis
    std::vector<double> ev;              // squared singular values of W L (descending)
    forge::linalg::mat<double> V;        // basis (columns): left U [out, p] or right V [in, p]
    std::vector<float> M;                // right: W L [out, in]
    la::Sym Linv_T;                      // right + whitened: L^-T
    bool whitened = false;
    const float* W = nullptr;
};

inline Factoring analyse(const float* W, i64 out, i64 in, const la::Sym* H, double damp = 1e-6) {
    Factoring F; F.out = out; F.in = in; F.W = W; F.left = out <= in; F.whitened = H != nullptr;
    if (F.left) {
        std::vector<float> G(static_cast<std::size_t>(out * out));
        if (H) {   // W H W^T
            auto Hf = la::to_f32(*H);
            std::vector<float> WH(static_cast<std::size_t>(out * in));
            la::gemm_nt(W, Hf.data(), WH.data(), out, in, in);
            la::gemm_nt(WH.data(), W, G.data(), out, out, in);
        } else la::gemm_nt(W, W, G.data(), out, out, in);
        auto e = eig_desc(G.data(), out);
        F.ev = std::move(e.w); F.V = std::move(e.V);
    } else {
        F.M.assign(W, W + out * in);
        if (H) {
            la::Sym L = *H;
            la::cholesky_damped(L, damp);
            std::vector<float> Lt(static_cast<std::size_t>(in * in));
            for (i64 i = 0; i < in; ++i) for (i64 j = 0; j < in; ++j) Lt[std::size_t(j * in + i)] = float(L(i, j));   // row j of L^T = column j of L
            la::gemm_nt(W, Lt.data(), F.M.data(), out, in, in);                                                     // W L
            F.Linv_T = la::tri_inverse_T(L);
        }
        auto G = small_gram(F.M.data(), out, in);
        auto e = eig_desc(G.data(), in);
        F.ev = std::move(e.w); F.V = std::move(e.V);
    }
    return F;
}

// X_a [r, in], X_b [out, r] with X_b X_a the rank-r (whitened-)optimal approximation of W.
inline std::pair<tensor, tensor> factors(const Factoring& F, i64 r) {
    const i64 out = F.out, in = F.in;
    tensor A({r, in}), B({out, r});
    if (F.left) {
        std::vector<float> Ut(static_cast<std::size_t>(r * out)), Wt(static_cast<std::size_t>(in * out));
        for (i64 k = 0; k < r; ++k) for (i64 i = 0; i < out; ++i) {
            const float u = float(F.V(std::size_t(i), std::size_t(k)));
            Ut[std::size_t(k * out + i)] = u; B.data[std::size_t(i * r + k)] = u;
        }
        la::transpose(F.W, Wt.data(), out, in);
        la::gemm_nt(Ut.data(), Wt.data(), A.data.data(), r, in, out);   // U_r^T W
    } else {
        std::vector<float> Vt(static_cast<std::size_t>(r * in));
        for (i64 k = 0; k < r; ++k) for (i64 i = 0; i < in; ++i) Vt[std::size_t(k * in + i)] = float(F.V(std::size_t(i), std::size_t(k)));
        la::gemm_nt(F.M.data(), Vt.data(), B.data.data(), out, r, in);   // (W L) V_r
        if (F.whitened) {
            auto Uf = la::to_f32(F.Linv_T);
            la::gemm_nt(Vt.data(), Uf.data(), A.data.data(), r, in, in);   // A[k, j] = sum_i L^-T[j, i] v_k[i]
        } else A.data = Vt;
    }
    return {std::move(A), std::move(B)};
}

// ||(W - X_b X_a) L||^2 / ||W L||^2 from the spectrum (exact for the projection factors).
inline double rel_residual(const std::vector<double>& ev, i64 r) {
    double tot = 0, rest = 0;
    for (std::size_t i = 0; i < ev.size(); ++i) { tot += ev[i]; if (i64(i) >= r) rest += ev[i]; }
    return tot > 0 ? rest / tot : 0.0;
}

// Stored parameters, each aliased tensor counted once (count() in compress.hpp counts names).
inline i64 stored_params(const Model& m) {
    std::set<std::string> used;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) for (auto& n : layer_tensors(m.spec, l)) { auto a = m.spec.alias.find(n); used.insert(a == m.spec.alias.end() ? n : a->second); }
    i64 s = 0;
    for (auto& n : used) if (m.w.has(n)) s += forge::numel(m.w.at(n).shape);
    return s;
}

inline std::vector<std::string> scope_projections(const LayerSpec& L, const std::string& scope) {
    std::vector<std::string> v;
    const bool attn_ok = scope == "all" || scope == "attn", ffn_ok = scope == "all" || scope == "ffn";
    if (!attn_ok && !ffn_ok) die("unknown scope " + scope + " (attn, ffn, all)");
    if (L.attn && attn_ok) for (const char* n : {"wq", "wk", "wv", "wo"}) v.push_back(n);
    if (L.ffn == LayerSpec::Ffn::dense && ffn_ok) for (const char* n : {"w_gate", "w_up", "w_down"}) v.push_back(n);
    return v;
}

inline std::string group_of_proj(const std::string& leaf) {
    if (leaf == "wq" || leaf == "wk" || leaf == "wv") return "attn";
    if (leaf == "wo") return "o";
    if (leaf == "w_gate" || leaf == "w_up") return "mlp";
    return "down";
}

// Replaces layer.L.X by its factors (spec + store).
inline void install_factors(Model& m, i64 l, const std::string& leaf, tensor A, tensor B) {
    const std::string p = std::format("layer.{}.{}", l, leaf);
    m.w.t.erase(p);
    m.w.t[p + "_a"] = std::move(A);
    m.w.t[p + "_b"] = std::move(B);
    m.spec.lowrank.insert(p);
}

// A data-free view of every linear layer of layer l (MoE experts included), for spectrum-based allocations.
inline std::vector<LinView> all_linears(Model& m, i64 l, const std::string& scope, LayerStats& S) {
    { LayerStats fresh; S = std::move(fresh); } S.layer = l;
    if (m.spec.layers[std::size_t(l)].ffn == LayerSpec::Ffn::moe) S.experts.resize(std::size_t(m.w.at(std::format("layer.{}.experts.w_gate", l)).shape[0]));
    return layer_linears(m, l, S, scope);
}

}  // namespace spectral

// ------------------------------------------------------------------------------------------------ alloc=alpha
inline std::map<std::string, double> alloc_alpha(Ctx& c, const PruneCfg& pc, double target, const Stage& st) {
    (void)pc;
    Model& m = c.m;
    const std::string scope = st.str("scope", "all"), unit = st.str("alpha_unit", "layer");
    const double tail = st.num("tail", 0.5), eps = st.num("eps", 0.2);
    const double lo = std::clamp(st.num("alpha_lo", target - eps), 0.0, 0.99), hi = std::clamp(st.num("alpha_hi", target + eps), lo, 0.99);
    struct U { std::string name; i64 layer; double params, alpha; };
    std::vector<U> units;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        LayerStats S;
        auto lin = spectral::all_linears(m, l, scope, S);
        std::vector<double> al(lin.size());
        // the matrices are independent: one per task (the inner GEMMs run on the same pool)
        for (std::size_t i = 0; i < lin.size(); ++i) al[i] = spectral::spectrum_of(lin[i].W, lin[i].rows, lin[i].cols, tail).alpha;
        for (std::size_t i = 0; i < lin.size(); ++i) units.push_back({lin[i].name, l, double(lin[i].rows * lin[i].cols), al[i]});
    }
    if (units.empty()) return {};
    if (unit == "layer") {   // AlphaPruning: one alpha per block (mean of its matrices)
        std::map<i64, std::pair<double, int>> acc;
        for (auto& u : units) { acc[u.layer].first += u.alpha; acc[u.layer].second++; }
        for (auto& u : units) u.alpha = acc[u.layer].first / acc[u.layer].second;
    } else if (unit != "proj") die("alpha_unit must be layer or proj");
    double amin = INFINITY, amax = -INFINITY;
    for (auto& u : units) { amin = std::min(amin, u.alpha); amax = std::max(amax, u.alpha); }
    std::vector<double> x(units.size());
    for (std::size_t i = 0; i < units.size(); ++i) x[i] = amax - amin > 1e-12 ? lo + (units[i].alpha - amin) / (amax - amin) * (hi - lo) : target;
    // shift so the parameter-weighted mean (after clamping) is the target: monotone in the shift -> bisection
    double P = 0; for (auto& u : units) P += u.params;
    auto mean_at = [&](double sh) { double s = 0; for (std::size_t i = 0; i < units.size(); ++i) s += units[i].params * std::clamp(x[i] + sh, lo, hi); return s / P; };
    double a = -1.0, b = 1.0;
    const double tgt = std::clamp(target, lo, hi);
    for (int it = 0; it < 100; ++it) { const double mid = 0.5 * (a + b); (mean_at(mid) < tgt ? a : b) = mid; }
    const double sh = 0.5 * (a + b);
    std::map<std::string, double> out;
    for (std::size_t i = 0; i < units.size(); ++i) out[units[i].name] = std::clamp(x[i] + sh, lo, hi);
    return out;
}

// ------------------------------------------------------------------------------------------------ spectra stage
inline value stage_spectra(Ctx& c, const Stage& st) {
    Model& m = c.m;
    const std::string scope = st.str("scope", "all");
    const double tail = st.num("tail", 0.5), edge = st.num("edge", 1.0);
    std::vector<value> rows;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        LayerStats S;
        for (auto& lv : spectral::all_linears(m, l, scope, S)) {
            auto sp = spectral::spectrum_of(lv.W, lv.rows, lv.cols, tail, edge);
            rows.push_back(value(jobj{{"name", value(lv.name)}, {"alpha", value(sp.alpha)}, {"sigma2", value(sp.sigma2)}, {"lambda_plus", value(sp.lambda_plus)},
                                      {"mp_rank", value(sp.mp_rank)}, {"dims", value(i64(sp.esd.size()))}, {"signal_frac", value(sp.signal_frac)}}));
        }
    }
    return value(jobj{{"stage", value(st.text())}, {"matrices", value(rows)}});
}

// ------------------------------------------------------------------------------------------------ lowrank stage
inline value stage_lowrank(Ctx& c, const Stage& st) {
    Model& m = c.m;
    const std::string method = st.str("method", "energy"), scope = st.str("scope", "all");
    if (method != "mp" && method != "energy" && method != "whiten" && method != "mdl") die("lowrank: unknown method " + method + " (mp, energy, whiten, mdl)");
    const double keep = st.num("keep", 0.9), ratio = st.num("ratio", 1.0), min_save = st.num("min_save", 0.1), bits = st.num("bits", 16);
    const double damp = st.num("damp", 1e-6), edge = st.num("edge", 1.0), tail = st.num("tail", 0.5);
    const i64 fixed = i64(st.num("rank", 0));
    const bool data = method == "whiten" || (method == "mdl" && st.str("data", c.calib.seqs.empty() ? "0" : "1") == "1");
    if (data && c.calib.seqs.empty()) die("lowrank: method " + method + " needs calibration data");
    jobj rep{{"stage", value(st.text())}, {"params_before", value(spectral::stored_params(m))}};
    std::vector<value> mats;
    Hidden h, nx;
    if (data) h = Hidden::embed(m, c.calib);
    i64 n_fact = 0;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        const auto& L = m.spec.layers[std::size_t(l)];
        std::vector<std::string> todo;
        for (auto& leaf : spectral::scope_projections(L, scope))
            if (!m.spec.factored(l, leaf.c_str()) && m.w.has(std::format("layer.{}.{}", l, leaf))) todo.push_back(leaf);
        LayerStats S;
        if (data && !todo.empty()) {
            StatsReq rq; rq.gram = true;
            for (auto& leaf : todo) rq.only.insert(spectral::group_of_proj(leaf));
            S = layer_pass(m, l, h, rq, nullptr);
        }
        for (auto& leaf : todo) {
            const std::string name = std::format("layer.{}.{}", l, leaf);
            const auto& W = m.w.at(name);
            const i64 out = W.shape[0], in = W.shape[1], p = std::min(out, in);
            const la::Sym* H = nullptr;
            if (data) {
                auto it = S.gram.find(spectral::group_of_proj(leaf));
                if (it != S.gram.end() && it->second.count > 0) H = &it->second.H;
            }
            auto F = spectral::analyse(W.data.data(), out, in, H, damp);
            i64 r = 1;
            jobj mr{{"name", value(name)}, {"dims", value(p)}};
            if (method == "mp") {
                auto sp = spectral::analyse_esd(H ? spectral::sq_singular_values(W.data.data(), out, in) : F.ev, out, in, tail, edge);
                r = sp.mp_rank; mr.push_back({"alpha", value(sp.alpha)}); mr.push_back({"signal_frac", value(sp.signal_frac)});
            } else if (method == "mdl") r = spectral::mdl_rank(F.ev, out, in, bits);
            else r = spectral::energy_rank(F.ev, keep);
            if (fixed > 0) r = fixed;
            r = std::clamp<i64>(std::min<i64>(r, i64(std::floor(ratio * double(p) + 1e-9))), 1, p);
            const double save = 1.0 - double(r * (out + in)) / double(out * in);
            mr.push_back({"rank", value(r)}); mr.push_back({"save", value(save)});
            mr.push_back({"rel_err", value(spectral::rel_residual(F.ev, r))}); mr.push_back({"whitened", value(H != nullptr)});
            if (save >= min_save) {
                auto [A, B] = spectral::factors(F, r);
                spectral::install_factors(m, l, leaf, std::move(A), std::move(B));
                ++n_fact;
                mr.push_back({"factored", value(true)});
            } else mr.push_back({"factored", value(false)});
            mats.push_back(value(mr));
        }
        if (data) { layer_forward(m, l, h, nx); std::swap(h, nx); }
        say("lowrank layer {}/{} done", l + 1, m.spec.n_layers());
    }
    rep.push_back({"factored", value(n_fact)});
    rep.push_back({"matrices", value(mats)});
    rep.push_back({"params_after", value(spectral::stored_params(m))});
    return value(rep);
}

// ------------------------------------------------------------------------------------------------ share stage
inline value stage_share(Ctx& c, const Stage& st) {
    Model& m = c.m;
    const i64 group = std::max<i64>(1, i64(st.num("group", 2)));
    const std::string scope = st.str("scope", "all");
    const double keep = st.num("keep", 0.9), ratio = st.num("ratio", 1.0), min_save = st.num("min_save", 0.1);
    const i64 fixed = i64(st.num("rank", 0));
    const bool data = st.str("data", "0") == "1";
    if (data && c.calib.seqs.empty()) die("share: data=1 needs calibration data");
    jobj rep{{"stage", value(st.text())}, {"params_before", value(spectral::stored_params(m))}};
    std::vector<value> groups;
    Hidden h, nx;
    if (data) h = Hidden::embed(m, c.calib);
    const i64 nL = m.spec.n_layers();
    for (i64 l0 = 0; l0 < nL; l0 += group) {
        const i64 l1 = std::min(nL, l0 + group);
        // projection types present, unfactored and of one shape in every layer of the group
        std::vector<std::string> leaves;
        if (l1 - l0 >= 2 || group == 1) {
            for (auto& leaf : spectral::scope_projections(m.spec.layers[std::size_t(l0)], scope)) {
                bool ok = true;
                for (i64 l = l0; l < l1 && ok; ++l) {
                    const std::string n = std::format("layer.{}.{}", l, leaf);
                    auto ps = spectral::scope_projections(m.spec.layers[std::size_t(l)], scope);
                    ok = std::find(ps.begin(), ps.end(), leaf) != ps.end() && !m.spec.factored(l, leaf.c_str()) && m.w.has(n)
                         && m.w.at(n).shape == m.w.at(std::format("layer.{}.{}", l0, leaf)).shape;
                }
                if (ok) leaves.push_back(leaf);
            }
        }
        std::vector<LayerStats> stats;
        Hidden hg;
        if (data && !leaves.empty()) {   // statistics of the group's layers as they are (before sharing)
            hg = h;
            StatsReq rq; rq.gram = true;
            for (auto& leaf : leaves) rq.only.insert(spectral::group_of_proj(leaf));
            for (i64 l = l0; l < l1; ++l) { stats.push_back(layer_pass(m, l, h, rq, &nx)); std::swap(h, nx); }
            h = std::move(hg);
        }
        for (auto& leaf : leaves) {
            const auto& W0 = m.w.at(std::format("layer.{}.{}", l0, leaf));
            const i64 out = W0.shape[0], in = W0.shape[1], G = l1 - l0;
            // shared output-side basis U: the top eigenvectors of sum_l W_l H_l W_l^T (H_l = I data-free, else each
            // layer's input Gram per token), i.e. the top left singular vectors of [W_1 L_1, W_2 L_2, ...]
            std::vector<float> Gs(static_cast<std::size_t>(out * out), 0.f), Gl(static_cast<std::size_t>(out * out)), WH(static_cast<std::size_t>(out * in));
            bool whitened = false;
            for (i64 l = l0; l < l1; ++l) {
                const auto& W = m.w.at(std::format("layer.{}.{}", l, leaf));
                const la::Gram* H = nullptr;
                if (data) {
                    auto it = stats[std::size_t(l - l0)].gram.find(spectral::group_of_proj(leaf));
                    if (it != stats[std::size_t(l - l0)].gram.end() && it->second.count > 0) H = &it->second;
                }
                if (H) {
                    auto Hf = la::to_f32(H->H);
                    const float inv = float(1.0 / double(H->count));
                    for (auto& x : Hf) x *= inv;
                    la::gemm_nt(W.data.data(), Hf.data(), WH.data(), out, in, in);
                    la::gemm_nt(WH.data(), W.data.data(), Gl.data(), out, out, in);
                    whitened = true;
                } else la::gemm_nt(W.data.data(), W.data.data(), Gl.data(), out, out, in);
                for (std::size_t i = 0; i < Gs.size(); ++i) Gs[i] += Gl[i];
            }
            auto E = spectral::eig_desc(Gs.data(), out);
            const i64 p = std::min(out, G * in);
            E.w.resize(std::size_t(p));
            i64 r = fixed > 0 ? fixed : spectral::energy_rank(E.w, keep);
            r = std::clamp<i64>(std::min<i64>(r, i64(std::floor(ratio * double(p) + 1e-9))), 1, p);
            const double before = double(G * out * in), after = double(r * out + G * r * in);
            const double save = 1.0 - after / before;
            jobj gr{{"layers", value(std::format("{}-{}", l0, l1 - 1))}, {"proj", value(leaf)}, {"rank", value(r)}, {"save", value(save)},
                    {"rel_err", value(spectral::rel_residual(E.w, r))}, {"whitened", value(whitened)}};
            if (save >= min_save) {
                tensor B({out, r});
                std::vector<float> Ut(static_cast<std::size_t>(r * out)), Wt(static_cast<std::size_t>(in * out));
                for (i64 k = 0; k < r; ++k) for (i64 i = 0; i < out; ++i) {
                    const float u = float(E.V(std::size_t(i), std::size_t(k)));
                    Ut[std::size_t(k * out + i)] = u; B.data[std::size_t(i * r + k)] = u;
                }
                const std::string base = std::format("layer.{}.{}_b", l0, leaf);
                for (i64 l = l0; l < l1; ++l) {
                    const auto& W = m.w.at(std::format("layer.{}.{}", l, leaf));
                    tensor A({r, in});
                    la::transpose(W.data.data(), Wt.data(), out, in);
                    la::gemm_nt(Ut.data(), Wt.data(), A.data.data(), r, in, out);   // U^T W_l
                    spectral::install_factors(m, l, leaf, std::move(A), B);   // a mirror of the basis keeps m.w.at() valid
                    if (l != l0) m.spec.alias[std::format("layer.{}.{}_b", l, leaf)] = base;
                }
                gr.push_back({"shared", value(true)});
            } else gr.push_back({"shared", value(false)});
            groups.push_back(value(gr));
        }
        if (data) for (i64 l = l0; l < l1; ++l) { layer_forward(m, l, h, nx); std::swap(h, nx); }
    }
    rep.push_back({"groups", value(groups)});
    rep.push_back({"params_after", value(spectral::stored_params(m))});
    return value(rep);
}

inline const bool reg_lowrank = register_stage("lowrank", &stage_lowrank);
inline const bool reg_share = register_stage("share", &stage_share);
inline const bool reg_spectra = register_stage("spectra", &stage_spectra);
inline const bool reg_alpha = register_alloc("alpha", &alloc_alpha);

}  // namespace forge::compress
