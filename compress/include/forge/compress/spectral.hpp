// Spectral structure of trained weights: diagnostics, spectrum-driven sparsity allocation and low-rank factoring.
//
//   diagnostics   the empirical spectral density (ESD) of every projection, eigenvalues of W W^T / N
//                 (N = the larger dimension); the heavy-tail exponent alpha (Hill estimator over the top of the
//                 ESD, as in HT-SR / WeightWatcher and AlphaPruning, Lu et al. 2024); the Marchenko-Pastur bulk
//                 edge lambda+ = sigma^2 (1 + sqrt(gamma))^2 with sigma^2 the mean of the bulk below the edge
//                 (spiked-model fixed point, robust to the spikes it removes), the signal rank (eigenvalues above the edge) and the fraction of the
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
//             damp=<relative Gram damping, 1e-6>  edge=<MP edge multiplier, 1>  tw=<Tracy-Widom margin, 2>
//   share     group=<2>  keep=<0.9>  ratio=<1>  rank=<r>  scope=attn|ffn|all  min_save=<0.1>  data=0|1
//   prune     ...,alloc=alpha  eps=<0.2>  alpha_lo=<min sparsity, s-eps>  alpha_hi=<max sparsity, s+eps>
//             alpha_unit=layer|proj  tail=<Hill tail fraction, 0.5>
//   spectra   scope=all  tail=0.5  mp_cap=<max eigenpairs computed for the MP rank>
//   every stage / alloc=alpha, the spectral solver (see "fast spectra" below):
//             exact_max=<512>  power=<3>  maxit=<25>  cheb=<6>  over=<16>  block0=<64>  block=<128>  probes=<32>  lanczos=<80>
#pragma once
#include <forge/compress/compress.hpp>
#include <forge/linalg/linalg.hpp>

#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <random>

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

// ------------------------------------------------------------------------------------------------ fast spectra
// Matrices whose Gram side p exceeds `exact_max` (512) never go through the serial dense eigh (O(p^3), ~2 min at
// p = 2048). They use, on the explicit p x p Gram G (one threaded GEMM):
//   top-k eigenpairs   subspace iteration with Rayleigh-Ritz and locking, in blocks: a block of b (64 first, then
//                      128) + `over` (16) vectors, kept orthogonal to the locked eigenvectors (projection) and
//                      re-orthonormalised by damped Cholesky-QR (twice) on threaded GEMMs; each iteration applies a
//                      Chebyshev filter bounded on [0, first oversampling Ritz value] (Zhou & Saad) instead of a power
//                      step, of degree <= `cheb` (6) and low enough that the block's dynamic range stays < 1e4 (f32):
//                      a plain power step on steep spectra, a strong filter on flat ones, where power steps crawl.
//                      After `power` (3) iterations the leading Ritz pairs are locked once converged: residual
//                      ||G u - t u|| <= tol t (1e-4), or the remaining Ritz-value error, extrapolated from its last
//                      change with the last step's per-iteration factor ((t_min / t)^2 for a power step,
//                      1 / T_d(2 t / a - 1)^2 for the filter), <= stag t (2e-5). After `maxit` (25) iterations (a flat
//                      bulk, where only the captured energy matters) at least the first half of the block is locked.
//                      The unlocked Ritz vectors warm-start the next block; once the remaining complement fits in one
//                      block a single Rayleigh-Ritz on it is exact. The small m x m problems go to
//                      forge::linalg::eigh. k grows on demand (TopEig::grow). Measured on a 2048 x 5632 power-law
//                      matrix (4 cores): alpha 0.6 s, full diagnostics 2.7 s, energy rank 0.9 (k = 832) 6.5 s, MDL
//                      1.9 s, against 165 s for the dense eigh; same ranks, sigma^2 and MP rank, alpha within 2%.
//   tail energies      exact: trace(G) - sum of the top-k (energy and MDL ranks never need the bulk eigenvalues).
//   MP noise level     both paths: the spiked-model fixed point (mp_noise): sigma^2 = mean of the eigenvalues below the
//                      edge = trace minus the top s, over p - s; it needs only the trace and the top s exactly, so the
//                      fast path agrees with the dense one. The MP signal rank is that s (k grows while every
//                      computed eigenvalue is above the edge).
//   Hill alpha         exact over the top-k; when its tail (the top `tail` * p eigenvalues) reaches below lambda_k,
//                      the rest [x_q, lambda_k) comes from a stochastic Lanczos quadrature estimate of the spectral
//                      measure (SLQ; Ubaru, Chen & Saad 2017: `probes` (32) Rademacher vectors, `lanczos` (80) steps
//                      with full reorthogonalisation, probes batched into one GEMM per step), renormalised to the
//                      known count of eigenvalues there; x_q is the SLQ quantile.
//   rank decisions     energy: grow k until the kept energy reaches keep * trace, or k reaches the largest rank that
//                      could still be factored (ratio cap / min_save), beyond which the answer cannot change the
//                      outcome. MDL: the best rank r <= k is certified when no r > k can beat it, using the lower bound
//                      tail(r) >= tail(k) - (the largest possible sum of r - k remaining eigenvalues, each <= lambda_k,
//                      with their known sum and sum of squares from tr(G^2)); uncertain beyond the useful ranks ->
//                      kept dense.
struct SpecOpts {
    i64 exact_max = 512;            // Gram side <= exact_max: dense eigh (exact, as before)
    int power = 3, maxit = 25;      // subspace iterations (Rayleigh-Ritz rounds) per block: minimum, maximum
    int cheb = 6;                   // Chebyshev filter degree per iteration (1: plain power steps)
    i64 over = 16;                  // oversampling
    i64 block0 = 64, block = 128;   // eigenpairs per block (first, later)
    double tol = 1e-4, stag = 2e-5; // locking: relative residual, extrapolated relative Ritz-value error
    int probes = 32, lanczos = 80;  // SLQ
    double tw = 2.0;                // Tracy-Widom margin of the MP edge (mp_edge_factor)
    std::uint64_t seed = 1;
};
inline SpecOpts spec_opts(const Stage& st) {
    SpecOpts o;
    o.exact_max = i64(st.num("exact_max", double(o.exact_max)));
    o.power = int(st.num("power", o.power)); o.maxit = int(st.num("maxit", o.maxit)); o.cheb = std::max(1, int(st.num("cheb", o.cheb)));
    o.over = std::max<i64>(1, i64(st.num("over", double(o.over))));
    o.block0 = std::max<i64>(1, i64(st.num("block0", double(o.block0)))); o.block = std::max<i64>(1, i64(st.num("block", double(o.block))));
    o.probes = int(st.num("probes", o.probes)); o.lanczos = int(st.num("lanczos", o.lanczos)); o.tw = st.num("tw", o.tw);
    return o;
}

namespace detail {
// Rows of X [m, p] orthonormalised: X <- L^-1 X with X X^T = L L^T (damped Cholesky; call twice for stability).
inline void cholqr(std::vector<float>& X, i64 m, i64 p) {
    std::vector<float> C(static_cast<std::size_t>(m * m)), Linv(static_cast<std::size_t>(m * m)), Xc(static_cast<std::size_t>(p * m)), out(static_cast<std::size_t>(m * p));
    la::gemm_nt(X.data(), X.data(), C.data(), m, m, p);
    la::Sym S(m);
    for (i64 i = 0; i < m; ++i) for (i64 j = 0; j < m; ++j) S(i, j) = 0.5 * (double(C[std::size_t(i * m + j)]) + double(C[std::size_t(j * m + i)]));
    la::cholesky_damped(S, 1e-6);
    la::Sym U = la::tri_inverse_T(S);   // L^-T
    for (i64 i = 0; i < m; ++i) for (i64 j = 0; j < m; ++j) Linv[std::size_t(i * m + j)] = float(U(j, i));
    la::transpose(X.data(), Xc.data(), m, p);
    la::gemm_nt(Linv.data(), Xc.data(), out.data(), m, p, m);
    X.swap(out);
}
}  // namespace detail

// Top eigenpairs of an explicit symmetric PSD matrix G [p, p], computed on demand (descending).
struct TopEig {
    i64 p = 0;
    std::vector<float> G;
    double trace = 0, trace2 = 0;
    SpecOpts o;
    std::vector<double> w;        // locked eigenvalues, descending
    std::vector<float> Ut, Uc;    // locked eigenvectors as rows [k, p] and transposed [p, k]
    std::vector<float> carry;     // unlocked Ritz vectors of the last block (warm start), rows
    i64 n_carry = 0;
    int iters = 0;                // subspace iterations used (diagnostic)
    std::mt19937_64 rng;

    TopEig() = default;
    TopEig(std::vector<float> G_, i64 p_, const SpecOpts& o_) : p(p_), G(std::move(G_)), o(o_), rng(o_.seed) {
        for (i64 i = 0; i < p; ++i) trace += double(G[std::size_t(i * p + i)]);
        for (float x : G) trace2 += double(x) * double(x);   // tr(G^2) = ||G||_F^2
    }
    bool exact() const { return p <= o.exact_max; }
    i64 k() const { return i64(w.size()); }
    double top_sum(i64 r) const { double s = 0; for (i64 i = 0; i < std::min(r, k()); ++i) s += w[std::size_t(i)]; return s; }

    void project(std::vector<float>& X, i64 m) const {   // X <- X (I - U^T U)
        const i64 kk = k();
        if (kk == 0) return;
        std::vector<float> C(static_cast<std::size_t>(m * kk)), D(static_cast<std::size_t>(m * p));
        la::gemm_nt(X.data(), Ut.data(), C.data(), m, kk, p);
        la::gemm_nt(C.data(), Uc.data(), D.data(), m, p, kk);
        for (std::size_t i = 0; i < D.size(); ++i) X[i] -= D[i];
    }
    void orth(std::vector<float>& X, i64 m) const { project(X, m); detail::cholqr(X, m, p); project(X, m); detail::cholqr(X, m, p); }

    void lock(const std::vector<float>& X, const std::vector<double>& th, i64 c) {
        const i64 k0 = k();
        Ut.resize(static_cast<std::size_t>((k0 + c) * p));
        std::copy_n(X.data(), c * p, Ut.data() + k0 * p);
        for (i64 i = 0; i < c; ++i) w.push_back(std::max(th[std::size_t(i)], 0.0));
        if (!std::is_sorted(w.begin(), w.end(), std::greater<double>())) {   // a later block found a larger value
            std::vector<i64> ord(w.size()); std::iota(ord.begin(), ord.end(), 0);
            std::stable_sort(ord.begin(), ord.end(), [&](i64 a, i64 b) { return w[std::size_t(a)] > w[std::size_t(b)]; });
            std::vector<double> w2(w.size()); std::vector<float> U2(Ut.size());
            for (std::size_t i = 0; i < ord.size(); ++i) { w2[i] = w[std::size_t(ord[i])]; std::copy_n(Ut.data() + ord[i] * p, p, U2.data() + i64(i) * p); }
            w.swap(w2); Ut.swap(U2);
        }
        Uc.resize(Ut.size());
        la::transpose(Ut.data(), Uc.data(), k(), p);
    }

    // X <- T_d((2 G - a) / a) X (rows), the degree-d Chebyshev polynomial that is bounded by 1 on the unwanted
    // part [0, a] of the spectrum and grows fast above it (Zhou & Saad's filtered subspace iteration). GX = G X on
    // entry. Each term is projected off the locked eigenvectors (which the filter would amplify most) and the
    // recurrence is rescaled as it goes (it is linear, so only directions matter).
    void cheb_filter(std::vector<float>& X, const std::vector<float>& GX, i64 m, double a, int deg) const {
        const float e = float(a / 2), cc = float(a / 2);
        std::vector<float> X0 = X, X1(X.size()), Gt(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) X1[i] = (GX[i] - cc * X0[i]) / e;
        project(X1, m);
        for (int j = 1; j < deg; ++j) {
            la::gemm_nt(X1.data(), G.data(), Gt.data(), m, p, p);
            float mx = 0;
            for (std::size_t i = 0; i < X.size(); ++i) { const float v = 2.f * (Gt[i] - cc * X1[i]) / e - X0[i]; X0[i] = v; mx = std::max(mx, std::abs(v)); }
            project(X0, m);
            std::swap(X0, X1);   // X1 = newest, X0 = previous
            const float sc = mx > 0 ? 1.f / mx : 1.f;
            for (std::size_t i = 0; i < X.size(); ++i) { X0[i] *= sc; X1[i] *= sc; }
        }
        X.swap(X1);
    }

    void grow(i64 want) {
        want = std::min(want, p);
        if (k() >= want) return;
        if (exact()) {   // dense, all at once
            auto e = eig_desc(G.data(), p);
            w = e.w;
            Ut.resize(static_cast<std::size_t>(p * p));
            for (i64 j = 0; j < p; ++j) for (i64 i = 0; i < p; ++i) Ut[std::size_t(j * p + i)] = float(e.V(std::size_t(i), std::size_t(j)));
            Uc.resize(Ut.size()); la::transpose(Ut.data(), Uc.data(), p, p);
            return;
        }
        std::normal_distribution<float> N01(0.f, 1.f);
        while (k() < want) {
            const i64 rem = p - k(), b = k() == 0 ? o.block0 : o.block;
            const i64 m = std::min(b + o.over, rem);
            const bool all = m == rem;   // the whole complement: one Rayleigh-Ritz is exact
            const i64 lim = all ? m : m - o.over;
            std::vector<float> X(static_cast<std::size_t>(m * p));
            const i64 nc = std::min(n_carry, m);
            std::copy_n(carry.data(), nc * p, X.data());
            for (std::size_t i = std::size_t(nc * p); i < X.size(); ++i) X[i] = N01(rng);
            n_carry = 0;
            orth(X, m);
            std::vector<float> Y(X.size()), T(static_cast<std::size_t>(m * m)), St(static_cast<std::size_t>(m * m)), X2(X.size()), Xc(X.size());
            std::vector<double> th, prev, res(static_cast<std::size_t>(m));
            double last_a = 0;
            int last_d = 1;
            i64 c = 0;
            for (int it = 0;; ++it) {
                ++iters;
                la::gemm_nt(X.data(), G.data(), Y.data(), m, p, p);   // rows: G x_i
                la::gemm_nt(Y.data(), X.data(), T.data(), m, m, p);   // X G X^T
                auto E = eig_desc(T.data(), m);
                th = E.w;
                for (i64 a = 0; a < m; ++a) for (i64 q = 0; q < m; ++q) St[std::size_t(a * m + q)] = float(E.V(std::size_t(q), std::size_t(a)));
                la::transpose(X.data(), Xc.data(), m, p); la::gemm_nt(St.data(), Xc.data(), X2.data(), m, p, m); X.swap(X2);   // Ritz vectors
                la::transpose(Y.data(), Xc.data(), m, p); la::gemm_nt(St.data(), Xc.data(), X2.data(), m, p, m); Y.swap(X2);   // G * Ritz vectors
                if (all) { c = m; break; }
                la::parallel_for(static_cast<std::size_t>(m), 4, [&](std::size_t b0, std::size_t e0) {
                    for (std::size_t i = b0; i < e0; ++i) {
                        double s = 0; const float* x = X.data() + i * std::size_t(p); const float* y = Y.data() + i * std::size_t(p);
                        for (i64 j = 0; j < p; ++j) { const double d = double(y[j]) - th[i] * double(x[j]); s += d * d; }
                        res[i] = std::sqrt(s);
                    }
                });
                // converged: small residual, or the Ritz value's remaining error, extrapolated from its last change
                // with the per-iteration factor rho^2 (rho = smallest / this Ritz value), below stag
                c = 0;
                while (c < lim) {
                    const std::size_t i = std::size_t(c);
                    bool conv = res[i] <= o.tol * th[i];
                    if (!conv && it + 1 >= o.power && !prev.empty() && th[i] > 0) {
                        // per-iteration error factor of the last step: (lambda_min / lambda)^2 for a power step,
                        // 1 / T_d(2 lambda / a - 1)^2 for the Chebyshev filter on [0, a]
                        double rho2 = std::pow(std::min(0.999, th[std::size_t(m - 1)] / th[i]), 2);
                        if (last_d > 1) { const double t = 2 * th[i] / last_a - 1; rho2 = t > 1 ? std::min(0.999, std::pow(1 / std::cosh(double(last_d) * std::acosh(t)), 2)) : 0.999; }
                        conv = std::abs(th[i] - prev[i]) * rho2 / (1 - rho2) <= o.stag * th[i];
                    }
                    if (!conv) break;
                    ++c;
                }
                const i64 need = std::min(lim, want - k());
                if (it + 1 >= o.power && c >= need) break;
                // out of iterations (a flat bulk): lock at least the better-converged first half of the block
                if (it + 1 >= o.maxit) { c = std::max(c, std::max<i64>(1, std::min(need, lim / 2))); break; }
                prev = th;
                // Chebyshev degree: at most `cheb`, and low enough that the block's own dynamic range after the
                // filter, T_d(2 t_0 / a - 1), stays below 1e4 (f32 Cholesky-QR keeps every direction of the block)
                const double a = th[std::size_t(lim)];   // upper estimate of the unwanted part: the first oversampling Ritz value
                int d = 1;
                if (o.cheb > 1 && a > 0 && th[0] > a * (1 + 1e-6)) d = std::clamp(int(std::acosh(1e4) / std::acosh(2 * th[0] / a - 1)), 1, o.cheb);
                if (d > 1) cheb_filter(X, Y, m, a, d);   // Chebyshev-filtered step
                else X.swap(Y);                          // power step
                last_a = a; last_d = d;
                orth(X, m);
            }
            lock(X, th, c);
            n_carry = m - c;
            carry.assign(X.begin() + c * p, X.end());
        }
    }
};

// ------------------------------------------------------------------------------------------------ SLQ density
struct Density {
    std::vector<double> x, wt, cw;   // nodes ascending, weights, cumulative weights (total 1)
    double quantile(double q) const {
        if (x.empty()) return 0.0;
        auto it = std::lower_bound(cw.begin(), cw.end(), q);
        if (it == cw.end()) return x.back();
        const std::size_t i = std::size_t(it - cw.begin());
        if (i == 0) return x[0];
        const double t = (q - cw[i - 1]) / std::max(cw[i] - cw[i - 1], 1e-300);
        return x[i - 1] + t * (x[i] - x[i - 1]);
    }
};

// Stochastic Lanczos quadrature estimate of the eigenvalue distribution of G (each eigenvalue weight 1/p).
inline Density slq_density(const TopEig& T, int probes, int steps, std::uint64_t seed) {
    const i64 p = T.p, nv = std::max(1, probes);
    const i64 ms = std::clamp<i64>(steps, 1, p);
    std::vector<float> Q(static_cast<std::size_t>(nv * ms * p)), V(static_cast<std::size_t>(nv * p)), Wv(static_cast<std::size_t>(nv * p));
    std::vector<double> al(static_cast<std::size_t>(nv * ms), 0.0), be(static_cast<std::size_t>(nv * ms), 0.0);
    std::vector<i64> len(static_cast<std::size_t>(nv), ms);
    std::vector<char> active(static_cast<std::size_t>(nv), 1);
    std::mt19937_64 rng(seed);
    const float inv = float(1.0 / std::sqrt(double(p)));
    for (i64 v = 0; v < nv; ++v) for (i64 j = 0; j < p; ++j) { const float s = (rng() & 1) ? inv : -inv; V[std::size_t(v * p + j)] = s; Q[std::size_t((v * ms) * p + j)] = s; }
    auto q = [&](i64 v, i64 j) { return Q.data() + (v * ms + j) * p; };
    for (i64 j = 0; j < ms; ++j) {
        la::gemm_nt(V.data(), T.G.data(), Wv.data(), nv, p, p);   // rows G q_j (G symmetric)
        la::parallel_for(static_cast<std::size_t>(nv), 1, [&](std::size_t b0, std::size_t e0) {
            std::vector<double> w(static_cast<std::size_t>(p));
            for (std::size_t vv = b0; vv < e0; ++vv) {
                const i64 v = i64(vv);
                if (!active[vv]) continue;
                const float* qj = q(v, j);
                for (i64 t = 0; t < p; ++t) w[std::size_t(t)] = Wv[std::size_t(v * p + t)];
                double a = 0; for (i64 t = 0; t < p; ++t) a += double(qj[t]) * w[std::size_t(t)];
                for (i64 t = 0; t < p; ++t) w[std::size_t(t)] -= a * qj[t];
                if (j > 0) { const float* qp = q(v, j - 1); const double bp = be[std::size_t(v * ms + j - 1)]; for (i64 t = 0; t < p; ++t) w[std::size_t(t)] -= bp * qp[t]; }
                for (int pass = 0; pass < 2; ++pass) for (i64 s = 0; s <= j; ++s) {   // full reorthogonalisation
                    const float* qs = q(v, s);
                    double cdot = 0; for (i64 t = 0; t < p; ++t) cdot += double(qs[t]) * w[std::size_t(t)];
                    for (i64 t = 0; t < p; ++t) w[std::size_t(t)] -= cdot * qs[t];
                }
                double bn = 0; for (double x : w) bn += x * x;
                bn = std::sqrt(bn);
                al[std::size_t(v * ms + j)] = a;
                if (j + 1 == ms || bn <= 1e-9 * std::max(T.trace, 1e-300)) { len[vv] = j + 1; active[vv] = 0; continue; }
                be[std::size_t(v * ms + j)] = bn;
                float* qn = q(v, j + 1);
                for (i64 t = 0; t < p; ++t) { qn[t] = float(w[std::size_t(t)] / bn); V[std::size_t(v * p + t)] = qn[t]; }
            }
        });
        for (i64 v = 0; v < nv; ++v) if (!active[std::size_t(v)]) std::fill_n(V.data() + v * p, p, 0.f);
    }
    std::vector<std::pair<double, double>> nodes;
    for (i64 v = 0; v < nv; ++v) {
        const i64 L = len[std::size_t(v)];
        forge::linalg::mat<double> A(static_cast<std::size_t>(L), static_cast<std::size_t>(L)), U;
        for (i64 i = 0; i < L; ++i) {
            A(std::size_t(i), std::size_t(i)) = al[std::size_t(v * ms + i)];
            if (i + 1 < L) A(std::size_t(i), std::size_t(i + 1)) = A(std::size_t(i + 1), std::size_t(i)) = be[std::size_t(v * ms + i)];
        }
        std::vector<double> th;
        forge::linalg::eigh(A, th, U);
        for (i64 i = 0; i < L; ++i) { const double u0 = U(0, std::size_t(i)); nodes.push_back({std::max(th[std::size_t(i)], 0.0), u0 * u0 / double(nv)}); }
    }
    std::sort(nodes.begin(), nodes.end());
    Density D;
    double acc = 0, tot = 0;
    for (auto& [x, wgt] : nodes) tot += wgt;
    for (auto& [x, wgt] : nodes) { D.x.push_back(x); D.wt.push_back(wgt / tot); D.cw.push_back(acc + 0.5 * wgt / tot); acc += wgt / tot; }   // midpoint CDF at each node
    return D;
}

// ------------------------------------------------------------------------------------------------ diagnostics
struct Spectrum {
    std::vector<double> esd;     // eigenvalues of W W^T / N, descending (N = max(out, in)); fast path: the top-k found
    i64 p = 0;                   // number of eigenvalues (min(out, in))
    bool exact = true;           // full dense spectrum (else top-k + SLQ bulk)
    double alpha = 0;            // Hill heavy-tail exponent of the ESD
    double sigma2 = 0;           // noise variance of the bulk (spiked-model fixed point, see mp_noise)
    double gamma = 1;            // min / max dimension
    double lambda_plus = 0;      // MP bulk edge
    i64 mp_rank = 0;             // eigenvalues above the edge
    bool mp_capped = false;      // mp_rank stopped at the cap (true rank may be larger)
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

// Noise level of a spiked model: sigma^2 is the mean of the eigenvalues below the MP edge (the bulk mean of an MP
// law is sigma^2 exactly), the edge is ef * sigma^2 (mp_edge_factor), iterated to the fixed point: the
// number s above the edge only grows, since removing eigenvalues above the mean lowers the mean. Needs only the
// trace and the top s eigenvalues (top(i), descending; more() makes more available, false when none can be added),
// so the exact and the fast path give the same answer. Stops if half the spectrum is above the edge (no bulk).
// Returns {sigma^2 (ESD units), s}.
template <class Top, class More>
std::pair<double, i64> mp_noise(Top&& top, More&& more, double trace, i64 p, double N, double ef) {
    i64 s = 0; double sum_top = 0, sig = trace / (double(p) * N);
    for (int it = 0; it < 1000 && s < p / 2; ++it) {
        sig = (trace - sum_top) / (double(p - s) * N);
        const double edge = ef * sig * N;
        i64 s2 = s;
        for (;;) {
            i64 avail = 0;
            while (s2 < (avail = top.avail()) && top(s2) > edge) ++s2;
            if (s2 < avail || !more()) break;
        }
        if (s2 == s) break;
        for (i64 i = s; i < s2; ++i) sum_top += top(i);
        s = s2;
    }
    return {sig, s};
}

// The MP edge in ESD units per unit noise variance, with a finite-size Tracy-Widom margin (Johnstone 2001): the
// largest noise eigenvalue of an [p, N] Gaussian matrix is ~ mu + s TW1 with mu = (sqrt(N-1) + sqrt(p))^2,
// s = (sqrt(N-1) + sqrt(p)) (1/sqrt(N-1) + 1/sqrt(p))^(1/3); tw = 2 (about the 99% quantile of TW1) keeps the
// largest noise eigenvalue out of the signal rank. tw = 0 is the asymptotic edge (1 + sqrt(gamma))^2.
inline double mp_edge_factor(i64 p, i64 Ni, double edge_mult, double tw) {
    const double n1 = std::max(1.0, double(Ni) - 1), sp = std::sqrt(double(p)), sn = std::sqrt(n1);
    const double mu = (sn + sp) * (sn + sp), sc = (sn + sp) * std::cbrt(1 / sn + 1 / sp);
    return edge_mult * (mu + tw * sc) / double(Ni);
}

// Exact analysis from the full list of squared singular values.
inline Spectrum analyse_esd(std::vector<double> sq_sv, i64 out, i64 in, double tail = 0.5, double edge_mult = 1.0, double tw = 2.0) {
    Spectrum S;
    const double N = double(std::max(out, in));
    S.gamma = double(std::min(out, in)) / N;
    for (auto& x : sq_sv) x /= N;
    S.esd = std::move(sq_sv);
    S.p = i64(S.esd.size());
    S.alpha = hill_alpha(S.esd, tail);
    double tot = 0; for (double x : S.esd) tot += x;
    struct { const std::vector<double>& e; i64 avail() const { return i64(e.size()); } double operator()(i64 i) const { return e[std::size_t(i)]; } } top{S.esd};
    const double ef = mp_edge_factor(S.p, std::max(out, in), edge_mult, tw);
    auto [sig, s] = mp_noise(top, [] { return false; }, tot, S.p, 1.0, ef);
    S.sigma2 = sig; S.mp_rank = s;
    S.lambda_plus = ef * S.sigma2;
    double sg = 0; for (i64 i = 0; i < s; ++i) sg += S.esd[std::size_t(i)];
    S.signal_frac = tot > 0 ? sg / tot : 0.0;
    return S;
}

// Analysis of the Gram of an [out, in] matrix (T.p = min(out, in)): exact below exact_max, else top-k + SLQ.
// mp_cap bounds the growth of k while counting the MP signal rank (0: alpha / sigma only).
inline Spectrum analyse_spectrum(TopEig& T, i64 out, i64 in, double tail = 0.5, double edge_mult = 1.0, i64 mp_cap = std::numeric_limits<i64>::max()) {
    if (T.exact()) { T.grow(T.p); return analyse_esd(T.w, out, in, tail, edge_mult, T.o.tw); }
    Spectrum S;
    S.exact = false; S.p = T.p;
    const double N = double(std::max(out, in));
    S.gamma = double(std::min(out, in)) / N;
    T.grow(std::min(T.o.block0, T.p));
    struct { TopEig& t; i64 avail() const { return t.k(); } double operator()(i64 i) const { return t.w[std::size_t(i)]; } } top{T};
    auto more = [&] { if (T.k() >= std::min(mp_cap, T.p)) return false; T.grow(T.k() + T.o.block); return true; };
    const double ef = mp_edge_factor(T.p, std::max(out, in), edge_mult, T.o.tw);
    auto [sig, ns] = mp_noise(top, more, T.trace, T.p, N, ef);
    S.sigma2 = sig; S.mp_rank = ns;
    S.mp_capped = ns == T.k() && T.k() < T.p;
    S.lambda_plus = ef * S.sigma2;
    S.signal_frac = T.trace > 0 ? T.top_sum(S.mp_rank) / T.trace : 0.0;
    // Hill: exact top-k, the rest of the tail from the SLQ measure (renormalised to its known count)
    const i64 p = T.p, kh = std::clamp<i64>(i64(std::llround(tail * double(p))), 2, p - 1), k = T.k();
    double s = 0;
    if (kh < k) {
        const double x = std::max(T.w[std::size_t(kh)], 1e-300);
        for (i64 i = 0; i < kh; ++i) s += std::log(std::max(T.w[std::size_t(i)], 1e-300) / x);
    } else {
        const Density D = slq_density(T, T.o.probes, T.o.lanczos, T.o.seed + 7);
        const double wl = T.w.back();
        const double x = std::max(std::min(D.quantile(1.0 - double(kh) / double(p)), wl), 1e-300);
        for (i64 i = 0; i < k; ++i) s += std::log(std::max(T.w[std::size_t(i)], 1e-300) / x);
        double cnt = 0, lg = 0;
        for (std::size_t i = 0; i < D.x.size(); ++i) if (D.x[i] >= x && D.x[i] < wl) { cnt += D.wt[i]; lg += D.wt[i] * std::log(D.x[i] / x); }
        const double need = double(kh - k);
        s += cnt > 0 ? lg / cnt * need : need * 0.5 * std::log(wl / x);
    }
    S.alpha = 1.0 + double(kh) / std::max(s, 1e-12);
    S.esd = T.w;
    for (auto& x : S.esd) x /= N;
    return S;
}

inline Spectrum spectrum_of(const float* W, i64 out, i64 in, double tail = 0.5, double edge_mult = 1.0, const SpecOpts& o = {}, i64 mp_cap = std::numeric_limits<i64>::max()) {
    TopEig T(small_gram(W, out, in), std::min(out, in), o);
    return analyse_spectrum(T, out, in, tail, edge_mult, mp_cap);
}

// ------------------------------------------------------------------------------------------------ rank selection
inline i64 energy_rank(const std::vector<double>& ev, double keep) {
    double tot = 0; for (double x : ev) tot += x;
    if (tot <= 0) return 1;
    double acc = 0;
    for (std::size_t i = 0; i < ev.size(); ++i) { acc += ev[i]; if (acc >= keep * tot * (1 - 1e-12)) return i64(i) + 1; }
    return i64(ev.size());
}

// Smallest rank keeping `keep` of the trace; cap + 1 when it exceeds `cap` (k never grows beyond the cap).
inline i64 energy_rank(TopEig& T, double keep, i64 cap) {
    if (T.trace <= 0) return 1;
    cap = std::min(cap, T.p);
    if (cap < 1) return 1;
    const double goal = keep * T.trace * (1 - 1e-12);
    for (;;) {
        double acc = 0;
        for (i64 i = 0; i < T.k(); ++i) { acc += T.w[std::size_t(i)]; if (acc >= goal) return i + 1 > cap ? cap + 1 : i + 1; }
        if (T.k() >= T.p) return T.p;
        if (T.k() >= cap) return cap + 1;
        T.grow(std::min(cap, T.k() + (T.k() ? T.o.block : T.o.block0)));
    }
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

// MDL on the top-k: the best r <= k is returned once certified against every r > k (tail(r) >= tail(k) -
// (r - k) lambda_k); if that needs k beyond `cap` (no larger rank could be factored) the projection stays dense (p).
inline i64 mdl_rank(TopEig& T, i64 out, i64 in, double bits, i64 cap) {
    const i64 p = T.p;
    if (T.trace <= 0 || p == 0) return 1;
    const double N = double(out) * double(in), fl = T.trace * std::pow(2.0, -2.0 * bits), oi = double(out + in);
    auto cost = [&](i64 r, double tail) { return bits * std::min(double(r) * oi, N) + 0.5 * N * std::log2(std::max(tail, fl) / fl); };
    if (T.k() == 0) T.grow(std::min(T.o.block0, p));
    for (;;) {
        const i64 k = T.k();
        double S = 0, bc = INFINITY; i64 best = p;
        for (i64 r = 1; r <= k; ++r) { S += T.w[std::size_t(r - 1)]; const double c = cost(r, T.trace - S); if (c < bc) { bc = c; best = r; } }
        if (k >= p) return best;
        if (const double c = cost(p, 0.0); c < bc) { bc = c; best = p; }
        // the n = p - k remaining eigenvalues are <= lambda_k, sum to S1 and their squares to S2 (tr(G^2) minus the
        // top-k squares): the j largest sum to at most min(j lambda_k, (j S1 + sqrt(j (n - j) (n S2 - S1^2))) / n)
        const double tk = std::max(T.trace - S, 0.0), wl = T.w[std::size_t(k - 1)];
        double sq = 0; for (i64 i = 0; i < k; ++i) sq += T.w[std::size_t(i)] * T.w[std::size_t(i)];
        const double n = double(p - k), S2 = std::max(T.trace2 - sq, 0.0), var = std::max(n * S2 - tk * tk, 0.0);
        double lb = INFINITY;
        for (i64 r = k + 1; r < p; ++r) {
            const double j = double(r - k), top = std::min(j * wl, (j * tk + std::sqrt(j * (n - j) * var)) / n);
            lb = std::min(lb, cost(r, std::max(tk - top, 0.0)));
        }
        if (bc <= lb) return best;
        if (k >= cap) return p;
        T.grow(std::min(p, k + T.o.block));
    }
}

// ------------------------------------------------------------------------------------------------ factoring
// The spectrum of W (whitened by H when given) and the factors of its rank-r projection.
struct Factoring {
    i64 out = 0, in = 0;
    bool left = true;                    // left basis (out <= in) or right basis
    TopEig eig;                          // eigenpairs of W H W^T [out, out] (left) or (W L)^T (W L) [in, in] (right)
    std::vector<float> M;                // right: W L [out, in]
    la::Sym Linv_T;                      // right + whitened: L^-T
    bool whitened = false;
    const float* W = nullptr;
};

inline Factoring analyse(const float* W, i64 out, i64 in, const la::Sym* H, double damp = 1e-6, const SpecOpts& o = {}) {
    Factoring F; F.out = out; F.in = in; F.W = W; F.left = out <= in; F.whitened = H != nullptr;
    if (F.left) {
        std::vector<float> G(static_cast<std::size_t>(out * out));
        if (H) {   // W H W^T
            auto Hf = la::to_f32(*H);
            std::vector<float> WH(static_cast<std::size_t>(out * in));
            la::gemm_nt(W, Hf.data(), WH.data(), out, in, in);
            la::gemm_nt(WH.data(), W, G.data(), out, out, in);
        } else la::gemm_nt(W, W, G.data(), out, out, in);
        F.eig = TopEig(std::move(G), out, o);
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
        F.eig = TopEig(small_gram(F.M.data(), out, in), in, o);
    }
    return F;
}

// X_a [r, in], X_b [out, r] with X_b X_a the rank-r (whitened-)optimal approximation of W.
inline std::pair<tensor, tensor> factors(Factoring& F, i64 r) {
    const i64 out = F.out, in = F.in;
    F.eig.grow(r);
    if (F.eig.k() < r) die("lowrank: eigenvectors missing");
    tensor A({r, in}), B({out, r});
    const float* U = F.eig.Ut.data();   // rows: basis vectors
    if (F.left) {
        std::vector<float> Wt(static_cast<std::size_t>(in * out));
        for (i64 k = 0; k < r; ++k) for (i64 i = 0; i < out; ++i) B.data[std::size_t(i * r + k)] = U[k * out + i];
        la::transpose(F.W, Wt.data(), out, in);
        la::gemm_nt(U, Wt.data(), A.data.data(), r, in, out);   // U_r^T W
    } else {
        la::gemm_nt(F.M.data(), U, B.data.data(), out, r, in);   // (W L) V_r
        if (F.whitened) {
            auto Uf = la::to_f32(F.Linv_T);
            la::gemm_nt(U, Uf.data(), A.data.data(), r, in, in);   // A[k, j] = sum_i L^-T[j, i] v_k[i]
        } else std::copy_n(U, r * in, A.data.data());
    }
    return {std::move(A), std::move(B)};
}

// ||(W - X_b X_a) L||^2 / ||W L||^2 from the spectrum (exact for the projection factors).
inline double rel_residual(const std::vector<double>& ev, i64 r) {
    double tot = 0, rest = 0;
    for (std::size_t i = 0; i < ev.size(); ++i) { tot += ev[i]; if (i64(i) >= r) rest += ev[i]; }
    return tot > 0 ? rest / tot : 0.0;
}
inline double rel_residual(const TopEig& T, i64 r) { return T.trace > 0 ? std::max(T.trace - T.top_sum(r), 0.0) / T.trace : 0.0; }   // needs r <= k

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
    const auto so = spectral::spec_opts(st);
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        LayerStats S;
        auto lin = spectral::all_linears(m, l, scope, S);
        std::vector<double> al(lin.size());
        for (std::size_t i = 0; i < lin.size(); ++i) al[i] = spectral::spectrum_of(lin[i].W, lin[i].rows, lin[i].cols, tail, 1.0, so, 0).alpha;   // alpha only
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
    const auto so = spectral::spec_opts(st);
    const i64 mp_cap = i64(st.num("mp_cap", 1e15));
    std::vector<value> rows;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        LayerStats S;
        for (auto& lv : spectral::all_linears(m, l, scope, S)) {
            auto sp = spectral::spectrum_of(lv.W, lv.rows, lv.cols, tail, edge, so, mp_cap);
            rows.push_back(value(jobj{{"name", value(lv.name)}, {"alpha", value(sp.alpha)}, {"sigma2", value(sp.sigma2)}, {"lambda_plus", value(sp.lambda_plus)},
                                      {"mp_rank", value(sp.mp_rank)}, {"mp_capped", value(sp.mp_capped)}, {"dims", value(sp.p)},
                                      {"signal_frac", value(sp.signal_frac)}, {"exact", value(sp.exact)}}));
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
    const auto so = spectral::spec_opts(st);
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
            auto F = spectral::analyse(W.data.data(), out, in, H, damp, so);
            // the largest rank that could still be factored: beyond it no decision depends on the spectrum
            const i64 rcap = std::clamp<i64>(i64(std::floor(ratio * double(p) + 1e-9)), 1, p);
            const i64 rsave = i64(std::min(double(p), std::floor((1.0 - min_save) * double(out) * double(in) / double(out + in) + 1e-9)));
            const i64 kc = std::min(rcap, rsave);
            i64 r = 1;
            jobj mr{{"name", value(name)}, {"dims", value(p)}};
            if (method == "mp") {
                auto sp = spectral::analyse_spectrum(F.eig, out, in, tail, edge, kc + 1);
                r = sp.mp_rank; mr.push_back({"alpha", value(sp.alpha)}); mr.push_back({"signal_frac", value(sp.signal_frac)});
            } else if (method == "mdl") r = spectral::mdl_rank(F.eig, out, in, bits, kc);
            else r = spectral::energy_rank(F.eig, keep, kc);
            if (fixed > 0) r = fixed;
            r = std::clamp<i64>(std::min<i64>(r, rcap), 1, p);
            const double save = 1.0 - double(r * (out + in)) / double(out * in);
            mr.push_back({"rank", value(r)}); mr.push_back({"save", value(save)});
            if (r <= F.eig.k()) mr.push_back({"rel_err", value(spectral::rel_residual(F.eig, r))});
            mr.push_back({"whitened", value(H != nullptr)}); mr.push_back({"eig_k", value(F.eig.k())}); mr.push_back({"exact", value(F.eig.exact())});
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
    const auto so = spectral::spec_opts(st);
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
            spectral::TopEig E(std::move(Gs), out, so);
            const i64 p = std::min(out, G * in);
            const i64 rcap = std::clamp<i64>(i64(std::floor(ratio * double(p) + 1e-9)), 1, p);
            const i64 rsave = i64(std::min(double(p), std::floor((1.0 - min_save) * double(G * out * in) / double(out + G * in) + 1e-9)));
            i64 r = fixed > 0 ? fixed : spectral::energy_rank(E, keep, std::min(rcap, rsave));
            r = std::clamp<i64>(std::min<i64>(r, rcap), 1, p);
            const double before = double(G * out * in), after = double(r * out + G * r * in);
            const double save = 1.0 - after / before;
            jobj gr{{"layers", value(std::format("{}-{}", l0, l1 - 1))}, {"proj", value(leaf)}, {"rank", value(r)}, {"save", value(save)},
                    {"whitened", value(whitened)}, {"eig_k", value(E.k())}, {"exact", value(E.exact())}};
            if (r <= E.k()) gr.push_back({"rel_err", value(spectral::rel_residual(E, r))});
            if (save >= min_save) {
                E.grow(r);
                tensor B({out, r});
                std::vector<float> Ut(E.Ut.begin(), E.Ut.begin() + r * out), Wt(static_cast<std::size_t>(in * out));
                for (i64 k = 0; k < r; ++k) for (i64 i = 0; i < out; ++i) B.data[std::size_t(i * r + k)] = Ut[std::size_t(k * out + i)];
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
