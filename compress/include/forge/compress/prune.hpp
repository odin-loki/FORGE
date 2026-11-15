// Unstructured and N:M semi-structured pruning of linear layers W [rows=out, cols=in] (y = x W^T).
//
// Selection criteria (published baselines):
//   magnitude  |W|
//   wanda      |W| * ||x_j||                                  (Sun et al., 2023)
//   ria        (|W|/sum_row|W| + |W|/sum_col|W|) * ||x_j||^a  (Zhang et al., 2024, relative importance + activation)
//   sparsegpt  OBS with lazy blocked updates on H = X^T X     (Frantar & Alistarh, 2023)
// Weight reconstruction (new in this framework):
//   mcr        masked conjugate-gradient refit: for a fixed sparsity mask, solve each row's least-squares problem
//              min ||(w - w0) X||^2 s.t. w = 0 off the mask, exactly (to CG tolerance), batched over rows with FORGE
//              GEMMs. It turns a selection-only method (magnitude / Wanda / RIA) into a reconstructing one and
//              polishes SparseGPT's greedy updates. The same routine refits under any mask (an existing 2:4 mask,
//              a mask with whole columns removed), so it is also the recovery step of structured pruning on
//              already-sparse models.
// Layer-wise allocation:
//   owl        outlier-weighted layerwise sparsity (Yin et al., 2024)
//   eba        error-budget allocation (new): per projection, the curve of residual-relative output error vs
//              sparsity is read off the sorted saliencies; a greedy marginal-cost (convex-hull) solver spends a
//              global parameter budget where it costs the least error.
#pragma once
#include <forge/compress/calib.hpp>

#include <algorithm>
#include <queue>

namespace forge::compress {

enum class Criterion { magnitude, wanda, ria, sparsegpt };
inline Criterion criterion_from(const std::string& s) {
    if (s == "magnitude" || s == "mag") return Criterion::magnitude;
    if (s == "wanda") return Criterion::wanda;
    if (s == "ria") return Criterion::ria;
    if (s == "sparsegpt") return Criterion::sparsegpt;
    die("unknown pruning criterion " + s);
}

struct PruneCfg {
    Criterion crit = Criterion::wanda;
    double sparsity = 0.5;
    int nm_n = 0, nm_m = 0;     // N:M - keep n of every m consecutive inputs (2:4 -> n=2, m=4); 0 = unstructured
    double damp = 0.01;         // relative Hessian damping (SparseGPT / refits)
    i64 block = 128;            // SparseGPT lazy-update block
    double ria_a = 0.5;
    bool keep_existing_zeros = true;   // weights already zero stay zero (sparse source models)
};

// Saliency per weight (higher = more important).
inline std::vector<float> saliency(const float* W, i64 rows, i64 cols, const std::vector<double>& diag, const PruneCfg& c) {
    std::vector<float> s(static_cast<std::size_t>(rows * cols));
    std::vector<double> xn(static_cast<std::size_t>(cols), 1.0);
    if (c.crit != Criterion::magnitude) for (i64 j = 0; j < cols; ++j) xn[std::size_t(j)] = std::sqrt(std::max(diag[std::size_t(j)], 0.0));
    if (c.crit == Criterion::ria) {
        std::vector<double> rs(static_cast<std::size_t>(rows), 0), cs(static_cast<std::size_t>(cols), 0);
        for (i64 i = 0; i < rows; ++i) for (i64 j = 0; j < cols; ++j) { double a = std::abs(W[i * cols + j]); rs[std::size_t(i)] += a; cs[std::size_t(j)] += a; }
        la::parallel_for(static_cast<std::size_t>(rows), 16, [&](std::size_t b, std::size_t e) {
            for (std::size_t i = b; i < e; ++i) for (i64 j = 0; j < cols; ++j) {
                double a = std::abs(W[i64(i) * cols + j]);
                s[i * std::size_t(cols) + std::size_t(j)] = float((a / std::max(rs[i], 1e-30) + a / std::max(cs[std::size_t(j)], 1e-30)) * std::pow(xn[std::size_t(j)], c.ria_a));
            }
        });
        return s;
    }
    la::parallel_for(static_cast<std::size_t>(rows), 16, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) for (i64 j = 0; j < cols; ++j)
            s[i * std::size_t(cols) + std::size_t(j)] = float(std::abs(W[i64(i) * cols + j]) * xn[std::size_t(j)]);
    });
    return s;
}

// Mask from saliency: per output row keep the top (1 - sparsity), or the top n of every m (N:M). 1 = keep.
inline std::vector<std::uint8_t> select_mask(const std::vector<float>& s, const float* W, i64 rows, i64 cols, const PruneCfg& c) {
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(rows * cols), 1);
    la::parallel_for(static_cast<std::size_t>(rows), 8, [&](std::size_t b, std::size_t e) {
        std::vector<i64> idx(static_cast<std::size_t>(cols));
        for (std::size_t i = b; i < e; ++i) {
            const float* si = s.data() + i * std::size_t(cols);
            std::uint8_t* mi = mask.data() + i * std::size_t(cols);
            const float* wi = W + i64(i) * cols;
            if (c.nm_m > 0) {
                for (i64 g = 0; g + c.nm_m <= cols; g += c.nm_m) {
                    std::vector<i64> q(static_cast<std::size_t>(c.nm_m));
                    std::iota(q.begin(), q.end(), g);
                    std::sort(q.begin(), q.end(), [&](i64 a, i64 bb) { return si[a] > si[bb]; });
                    for (i64 t = c.nm_n; t < c.nm_m; ++t) mi[q[std::size_t(t)]] = 0;
                }
            } else {
                std::iota(idx.begin(), idx.end(), 0);
                const i64 np = i64(std::llround(c.sparsity * double(cols)));
                std::nth_element(idx.begin(), idx.begin() + np, idx.end(), [&](i64 a, i64 bb) { return si[a] < si[bb]; });
                for (i64 t = 0; t < np; ++t) mi[idx[std::size_t(t)]] = 0;
            }
            if (c.keep_existing_zeros) for (i64 j = 0; j < cols; ++j) if (wi[j] == 0.f) mi[j] = 0;
        }
    });
    return mask;
}

inline void apply_mask(float* W, const std::vector<std::uint8_t>& mask) { for (std::size_t i = 0; i < mask.size(); ++i) if (!mask[i]) W[i] = 0.f; }

// SparseGPT (OBS with lazy batch updates); supports unstructured (adaptive per-block masks) and N:M.
inline void sparsegpt(float* W, i64 rows, i64 cols, const la::Sym& H, const PruneCfg& c) {
    std::vector<std::uint8_t> pre(static_cast<std::size_t>(rows * cols), 1);
    if (c.keep_existing_zeros) for (std::size_t i = 0; i < pre.size(); ++i) pre[i] = W[i] != 0.f;
    la::Sym Hd = H;
    for (i64 j = 0; j < cols; ++j) if (Hd(j, j) == 0) { Hd(j, j) = 1; for (i64 i = 0; i < rows; ++i) W[i * cols + j] = 0; }
    la::Sym U = la::inverse_upper_cholesky(Hd, c.damp);
    const i64 B = c.block;
    std::vector<float> W1, Err;
    for (i64 i1 = 0; i1 < cols; i1 += B) {
        const i64 i2 = std::min(cols, i1 + B), cnt = i2 - i1;
        W1.assign(static_cast<std::size_t>(rows * cnt), 0.f); Err.assign(static_cast<std::size_t>(rows * cnt), 0.f);
        for (i64 r = 0; r < rows; ++r) std::copy_n(W + r * cols + i1, cnt, W1.data() + r * cnt);
        std::vector<std::uint8_t> m1(static_cast<std::size_t>(rows * cnt), 1);   // 1 = keep
        if (c.nm_m == 0) {
            // adaptive mask: per row, prune the `sparsity` fraction of this block with the smallest w^2 / U_jj^2
            la::parallel_for(static_cast<std::size_t>(rows), 16, [&](std::size_t b, std::size_t e) {
                std::vector<i64> idx(static_cast<std::size_t>(cnt)); std::vector<double> sc(static_cast<std::size_t>(cnt));
                for (std::size_t r = b; r < e; ++r) {
                    for (i64 j = 0; j < cnt; ++j) { double u = U(i1 + j, i1 + j); double w = W1[r * std::size_t(cnt) + std::size_t(j)]; sc[std::size_t(j)] = w * w / (u * u); }
                    std::iota(idx.begin(), idx.end(), 0);
                    const i64 np = i64(std::llround(c.sparsity * double(cnt)));
                    std::nth_element(idx.begin(), idx.begin() + np, idx.end(), [&](i64 a, i64 bb) { return sc[std::size_t(a)] < sc[std::size_t(bb)]; });
                    for (i64 t = 0; t < np; ++t) m1[r * std::size_t(cnt) + std::size_t(idx[std::size_t(t)])] = 0;
                }
            });
        }
        la::parallel_for(static_cast<std::size_t>(rows), 8, [&](std::size_t b, std::size_t e) {
            for (std::size_t r = b; r < e; ++r) {
                float* w1 = W1.data() + r * std::size_t(cnt);
                std::uint8_t* mk = m1.data() + r * std::size_t(cnt);
                float* er = Err.data() + r * std::size_t(cnt);
                const std::uint8_t* pr = pre.data() + r * std::size_t(cols) + std::size_t(i1);
                for (i64 i = 0; i < cnt; ++i) {
                    if (c.nm_m > 0 && i % c.nm_m == 0 && i + c.nm_m <= cnt) {
                        std::vector<std::pair<double, i64>> q;
                        for (i64 t = 0; t < c.nm_m; ++t) { double u = U(i1 + i + t, i1 + i + t); double w = w1[i + t]; q.push_back({w * w / (u * u), i + t}); }
                        std::sort(q.begin(), q.end());
                        for (i64 t = 0; t < c.nm_m - c.nm_n; ++t) mk[q[std::size_t(t)].second] = 0;
                    }
                    if (!pr[i]) mk[i] = 0;
                    const double dd = U(i1 + i, i1 + i);
                    const double w = w1[i], qv = mk[i] ? w : 0.0;
                    const double err = (w - qv) / dd;
                    w1[i] = float(qv);
                    if (err != 0) {
                        const double* ur = U.row(i1 + i);
                        for (i64 j = i + 1; j < cnt; ++j) w1[j] -= float(err * ur[i1 + j]);
                    }
                    er[i] = float(err);
                }
            }
        });
        for (i64 r = 0; r < rows; ++r) std::copy_n(W1.data() + r * cnt, cnt, W + r * cols + i1);
        if (i2 < cols) {   // W[:, i2:] -= Err @ U[i1:i2, i2:]
            const i64 rest = cols - i2;
            std::vector<float> Ut(static_cast<std::size_t>(rest * cnt)), upd(static_cast<std::size_t>(rows * rest));
            for (i64 j = 0; j < rest; ++j) for (i64 i = 0; i < cnt; ++i) Ut[std::size_t(j * cnt + i)] = float(U(i1 + i, i2 + j));
            la::gemm_nt(Err.data(), Ut.data(), upd.data(), rows, rest, cnt);
            la::parallel_for(static_cast<std::size_t>(rows), 16, [&](std::size_t b, std::size_t e) {
                for (std::size_t r = b; r < e; ++r) for (i64 j = 0; j < rest; ++j) W[i64(r) * cols + i2 + j] -= upd[r * std::size_t(rest) + std::size_t(j)];
            });
        }
    }
    for (std::size_t i = 0; i < pre.size(); ++i) if (!pre[i]) W[i] = 0.f;
}

// Masked CG refit: rows of W solve (H_SS + lam I) w_S = (H w0)_S over their own support S (mask), warm-started
// from W. Returns the relative reconstruction error tr(dW H dW^T) / tr(W0 H W0^T) after the refit.
struct RefitReport { double err_before = 0, err_after = 0; int iters = 0; };
inline RefitReport mcr_refit(float* W, const float* W0, i64 rows, i64 cols, const la::Sym& H, const std::vector<std::uint8_t>& mask,
                             int iters = 25, double damp = 0.01, double tol = 1e-4) {
    RefitReport rep;
    std::vector<float> Hf = la::to_f32(H);
    const double lam = damp * H.mean_diag();
    for (i64 j = 0; j < cols; ++j) Hf[std::size_t(j * cols + j)] += float(lam);
    std::vector<double> dg(static_cast<std::size_t>(cols));
    for (i64 j = 0; j < cols; ++j) dg[std::size_t(j)] = double(Hf[std::size_t(j * cols + j)]);
    const std::size_t N = std::size_t(rows * cols);
    std::vector<float> D(N), R(N), Z(N), P(N), AP(N);
    auto energy = [&](const std::vector<float>& M, const std::vector<float>& MH) { double s = 0; for (std::size_t i = 0; i < N; ++i) s += double(M[i]) * MH[i]; return s; };
    // reference energy tr(W0 H W0^T)
    std::vector<float> W0v(W0, W0 + N), W0H(N);
    la::gemm_nt(W0v.data(), Hf.data(), W0H.data(), rows, cols, cols);
    const double ref = std::max(energy(W0v, W0H), 1e-30);
    // residual R = mask * ((W0 - W) H)   (gradient of 0.5 * err, negated)
    for (std::size_t i = 0; i < N; ++i) D[i] = W0[i] - W[i];
    la::gemm_nt(D.data(), Hf.data(), R.data(), rows, cols, cols);
    rep.err_before = energy(D, R) / ref;
    for (std::size_t i = 0; i < N; ++i) if (!mask[i]) R[i] = 0;
    std::vector<double> rz(static_cast<std::size_t>(rows)), r0(static_cast<std::size_t>(rows));
    for (i64 r = 0; r < rows; ++r) {
        double s = 0, n2 = 0;
        for (i64 j = 0; j < cols; ++j) { std::size_t k = std::size_t(r * cols + j); Z[k] = float(R[k] / dg[std::size_t(j)]); P[k] = Z[k]; s += double(R[k]) * Z[k]; n2 += double(R[k]) * R[k]; }
        rz[std::size_t(r)] = s; r0[std::size_t(r)] = std::sqrt(n2);
    }
    std::vector<std::uint8_t> done(static_cast<std::size_t>(rows), 0);
    for (int it = 0; it < iters; ++it) {
        la::gemm_nt(P.data(), Hf.data(), AP.data(), rows, cols, cols);
        bool any = false;
        la::parallel_for(static_cast<std::size_t>(rows), 16, [&](std::size_t b, std::size_t e) {
            for (std::size_t r = b; r < e; ++r) {
                if (done[r]) continue;
                double pap = 0;
                for (i64 j = 0; j < cols; ++j) { std::size_t k = r * std::size_t(cols) + std::size_t(j); if (mask[k]) pap += double(P[k]) * AP[k]; }
                if (pap <= 0) { done[r] = 1; continue; }
                const double alpha = rz[r] / pap;
                double nrz = 0, n2 = 0;
                for (i64 j = 0; j < cols; ++j) {
                    std::size_t k = r * std::size_t(cols) + std::size_t(j);
                    if (!mask[k]) continue;
                    W[k] += float(alpha * P[k]);
                    R[k] -= float(alpha * AP[k]);
                    Z[k] = float(R[k] / dg[std::size_t(j)]);
                    nrz += double(R[k]) * Z[k]; n2 += double(R[k]) * R[k];
                }
                const double beta = nrz / std::max(rz[r], 1e-300);
                rz[r] = nrz;
                for (i64 j = 0; j < cols; ++j) { std::size_t k = r * std::size_t(cols) + std::size_t(j); P[k] = mask[k] ? Z[k] + float(beta) * P[k] : 0.f; }
                if (std::sqrt(n2) < tol * std::max(r0[r], 1e-30)) done[r] = 1;
            }
        });
        for (std::size_t r = 0; r < std::size_t(rows); ++r) any |= !done[r];
        rep.iters = it + 1;
        if (!any) break;
    }
    for (std::size_t i = 0; i < N; ++i) { if (!mask[i]) W[i] = 0.f; D[i] = W0[i] - W[i]; }
    la::gemm_nt(D.data(), Hf.data(), R.data(), rows, cols, cols);
    rep.err_after = energy(D, R) / ref;
    return rep;
}

inline std::vector<std::uint8_t> mask_of(const float* W, std::size_t n) { std::vector<std::uint8_t> m(n); for (std::size_t i = 0; i < n; ++i) m[i] = W[i] != 0.f; return m; }

// Relative output error tr(dW H dW^T) / tr(W0 H W0^T) of a pruned layer (diagnostic).
inline double rel_error(const float* W, const float* W0, i64 rows, i64 cols, const la::Sym& H) {
    std::vector<float> Hf = la::to_f32(H), D(static_cast<std::size_t>(rows * cols)), DH(D.size()), W0v(W0, W0 + D.size()), W0H(D.size());
    for (std::size_t i = 0; i < D.size(); ++i) D[i] = W0[i] - W[i];
    la::gemm_nt(D.data(), Hf.data(), DH.data(), rows, cols, cols);
    la::gemm_nt(W0v.data(), Hf.data(), W0H.data(), rows, cols, cols);
    double a = 0, b = 0;
    for (std::size_t i = 0; i < D.size(); ++i) { a += double(D[i]) * DH[i]; b += double(W0v[i]) * W0H[i]; }
    return a / std::max(b, 1e-30);
}

// ---------------------------------------------------------------------------------------------------- allocation
// A prunable unit (one projection of one layer) with its error-vs-sparsity curve on a grid.
struct Unit {
    std::string name;
    i64 params = 0;
    std::vector<double> err;   // err[g] at sparsity grid[g]
    double lod = 0;            // outlier ratio (OWL)
    i64 layer = 0;
};

inline std::vector<double> sparsity_grid(double step = 0.05, double max_s = 0.9) { std::vector<double> g; for (double s = 0; s <= max_s + 1e-9; s += step) g.push_back(s); return g; }

// Error curve of one projection under per-row Wanda/RIA/magnitude selection, diag approximation:
// err(s) = sum of the pruned (saliency)^2 (= sum w^2 ||x||^2), normalised by `norm` (residual-stream energy).
inline Unit make_unit(const std::string& name, const float* W, i64 rows, i64 cols, const std::vector<double>& diag, const PruneCfg& c,
                      const std::vector<double>& grid, double norm, double owl_m = 5.0) {
    Unit u; u.name = name; u.params = rows * cols; u.layer = layer_of(name);
    PruneCfg cw = c; if (cw.crit == Criterion::sparsegpt) cw.crit = Criterion::wanda;
    auto s = saliency(W, rows, cols, diag, cw);
    std::vector<double> e(grid.size(), 0.0);
    double mean = 0; for (float v : s) mean += v; mean /= double(s.size());
    i64 out = 0; for (float v : s) out += double(v) > owl_m * mean;
    u.lod = double(out) / double(s.size());
    // squared Wanda saliency approximates the output error contribution
    std::vector<std::vector<double>> per(static_cast<std::size_t>(rows));
    la::parallel_for(static_cast<std::size_t>(rows), 16, [&](std::size_t b, std::size_t en) {
        std::vector<double> v(static_cast<std::size_t>(cols));
        for (std::size_t r = b; r < en; ++r) {
            const std::vector<double> xn = {};
            for (i64 j = 0; j < cols; ++j) { double w = W[i64(r) * cols + j]; v[std::size_t(j)] = w * w * diag[std::size_t(j)]; }
            // order by the criterion's saliency, accumulate the error proxy in that order
            std::vector<i64> idx(static_cast<std::size_t>(cols)); std::iota(idx.begin(), idx.end(), 0);
            const float* sr = s.data() + r * std::size_t(cols);
            std::sort(idx.begin(), idx.end(), [&](i64 a, i64 bb) { return sr[a] < sr[bb]; });
            std::vector<double> pre(static_cast<std::size_t>(cols) + 1, 0.0);
            for (i64 j = 0; j < cols; ++j) pre[std::size_t(j) + 1] = pre[std::size_t(j)] + v[std::size_t(idx[std::size_t(j)])];
            per[r].resize(grid.size());
            for (std::size_t g = 0; g < grid.size(); ++g) per[r][g] = pre[std::size_t(std::llround(grid[g] * double(cols)))];
        }
    });
    for (auto& p : per) for (std::size_t g = 0; g < grid.size(); ++g) e[g] += p[g];
    for (auto& x : e) x /= std::max(norm, 1e-30);
    u.err = std::move(e);
    return u;
}

// EBA: choose a grid index per unit minimising sum err subject to sum params * s >= target * sum params.
// Greedy on the lower convex hull of each curve (optimal for convex curves; the proxy curves are convex by
// construction because saliencies are visited in increasing order).
inline std::map<std::string, double> eba_allocate(const std::vector<Unit>& units, const std::vector<double>& grid, double target,
                                                  double min_s = 0.0, double max_s = 0.9) {
    double total = 0; for (auto& u : units) total += double(u.params);
    const double need = target * total;
    std::vector<std::size_t> at(units.size(), 0);
    std::size_t g_min = 0, g_max = grid.size() - 1;
    while (g_min < grid.size() && grid[g_min] < min_s - 1e-9) ++g_min;
    while (g_max > 0 && grid[g_max] > max_s + 1e-9) --g_max;
    double have = 0;
    for (std::size_t i = 0; i < units.size(); ++i) { at[i] = g_min; have += grid[g_min] * double(units[i].params); }
    using item = std::tuple<double, std::size_t, std::size_t>;   // (cost per param, unit, to-index)
    auto best_step = [&](std::size_t i) -> std::optional<item> {
        const auto& u = units[i];
        std::optional<item> b;
        for (std::size_t g = at[i] + 1; g <= g_max; ++g) {
            const double dp = (grid[g] - grid[at[i]]) * double(u.params);
            const double c = (u.err[g] - u.err[at[i]]) / dp;
            if (!b || c < std::get<0>(*b)) b = item{c, i, g};
        }
        return b;
    };
    std::priority_queue<item, std::vector<item>, std::greater<item>> pq;
    for (std::size_t i = 0; i < units.size(); ++i) if (auto s = best_step(i)) pq.push(*s);
    while (have < need && !pq.empty()) {
        auto [c, i, g] = pq.top(); pq.pop();
        if (g <= at[i]) continue;
        have += (grid[g] - grid[at[i]]) * double(units[i].params);
        at[i] = g;
        if (auto s = best_step(i)) pq.push(*s);
    }
    std::map<std::string, double> out;
    for (std::size_t i = 0; i < units.size(); ++i) out[units[i].name] = grid[at[i]];
    return out;
}

// OWL: layer keep-ratio grows with the layer's outlier ratio (LOD), mean sparsity preserved.
inline std::map<i64, double> owl_allocate(const std::vector<Unit>& units, double target, double lambda = 0.08) {
    std::map<i64, std::pair<double, double>> acc;   // layer -> (sum lod*params, params)
    for (auto& u : units) { acc[u.layer].first += u.lod * double(u.params); acc[u.layer].second += double(u.params); }
    std::vector<i64> ls; std::vector<double> lod;
    for (auto& [l, p] : acc) { ls.push_back(l); lod.push_back(p.first / p.second); }
    const double mn = *std::min_element(lod.begin(), lod.end()), mx = *std::max_element(lod.begin(), lod.end());
    std::vector<double> ratio(lod.size());
    for (std::size_t i = 0; i < lod.size(); ++i) ratio[i] = mx > mn ? (lod[i] - mn) / (mx - mn) * lambda * 2 : 0.0;
    double mean = 0; for (double r : ratio) mean += r; mean /= double(ratio.size());
    std::map<i64, double> out;
    for (std::size_t i = 0; i < ls.size(); ++i) out[ls[i]] = std::clamp(1.0 - (ratio[i] - mean + (1.0 - target)), 0.0, 0.95);
    return out;
}

}  // namespace forge::compress
