// Dense linear algebra for compression, on FORGE's thread pool (forge::task::parallel_for) and FORGE's threaded f32
// GEMM (forge::gpu::prims::gemm_nt). FORGE's own forge::linalg is serial and unblocked (fine to a few hundred rows);
// the Hessians here are up to 8192 x 8192, so the factorisations below are blocked and parallel, in f64.
#pragma once
#include <forge/compress/core.hpp>
#include <forge/gpu/prims.hpp>
#include <forge/simd/tier_a.hpp>

#include <cmath>
#include <numeric>

namespace forge::compress::la {

using forge::task::parallel_for;

// C[m,n] = A[m,k] * B[n,k]^T (row-major f32), threaded.
inline void gemm_nt(const float* A, const float* B, float* C, i64 m, i64 n, i64 k) {
    forge::gpu::prims::gemm_nt(A, B, C, std::size_t(m), std::size_t(n), std::size_t(k));
}

inline void transpose(const float* a, float* b, i64 rows, i64 cols) {
    forge::simd::tier_a_active().transpose_f32(a, b, std::size_t(rows), std::size_t(cols));
}

// f64 dot product on FORGE's dispatched SIMD table (scalar / AVX2 / AVX-512 chosen at run time).
inline double dot(const double* a, const double* b, i64 n) {
    static const auto f = forge::simd::tier_a_active().dot_f64;
    return n > 0 ? f(a, b, std::size_t(n)) : 0.0;
}

// Symmetric n x n matrix in f64, row-major (both triangles stored).
struct Sym {
    i64 n = 0;
    std::vector<double> a;
    Sym() = default;
    explicit Sym(i64 n_) : n(n_), a(static_cast<std::size_t>(n_ * n_), 0.0) {}
    double& operator()(i64 i, i64 j) { return a[std::size_t(i * n + j)]; }
    double operator()(i64 i, i64 j) const { return a[std::size_t(i * n + j)]; }
    double* row(i64 i) { return a.data() + i * n; }
    const double* row(i64 i) const { return a.data() + i * n; }
    double trace() const { double t = 0; for (i64 i = 0; i < n; ++i) t += (*this)(i, i); return t; }
    double mean_diag() const { return n ? trace() / double(n) : 0; }
};

// Accumulates H = sum X^T X over row blocks of X [rows, d] (f32 GEMM per block, f64 accumulation across blocks).
struct Gram {
    i64 d = 0, count = 0;
    Sym H;
    std::vector<float> xt, tmp;
    explicit Gram(i64 d_ = 0) : d(d_), H(d_) {}
    void add(const float* X, i64 rows) {
        constexpr i64 chunk = 4096;
        for (i64 r0 = 0; r0 < rows; r0 += chunk) {
            const i64 r = std::min(chunk, rows - r0);
            xt.resize(static_cast<std::size_t>(d * r));
            tmp.resize(static_cast<std::size_t>(d * d));
            transpose(X + r0 * d, xt.data(), r, d);
            gemm_nt(xt.data(), xt.data(), tmp.data(), d, d, r);
            parallel_for(static_cast<std::size_t>(d), 16, [&](std::size_t b, std::size_t e) {
                for (std::size_t i = b; i < e; ++i) {
                    double* h = H.row(i64(i));
                    const float* t = tmp.data() + i * std::size_t(d);
                    for (i64 j = 0; j < d; ++j) h[j] += double(t[j]);
                }
            });
            count += r;
        }
    }
    // Weighted rows: H += sum_i w_i x_i x_i^T (used for routed / gated statistics).
    void add_weighted(const float* X, const float* w, i64 rows) {
        std::vector<float> s(static_cast<std::size_t>(rows * d));
        for (i64 i = 0; i < rows; ++i) {
            const float sw = std::sqrt(std::max(0.f, w[i]));
            for (i64 j = 0; j < d; ++j) s[std::size_t(i * d + j)] = X[i * d + j] * sw;
        }
        i64 c = count;
        add(s.data(), rows);
        count = c + rows;
    }
};

// In-place lower Cholesky A = L L^T (blocked, right-looking, parallel). Returns false if not positive definite.
inline bool cholesky(Sym& A, i64 nb = 96) {
    const i64 n = A.n;
    for (i64 k0 = 0; k0 < n; k0 += nb) {
        const i64 k1 = std::min(n, k0 + nb);
        for (i64 j = k0; j < k1; ++j) {   // diagonal block
            double* rj = A.row(j);
            double s = rj[j] - dot(rj + k0, rj + k0, j - k0);
            if (!(s > 0) || !std::isfinite(s)) return false;
            const double ljj = std::sqrt(s);
            rj[j] = ljj;
            for (i64 i = j + 1; i < k1; ++i) {
                double* ri = A.row(i);
                ri[j] = (ri[j] - dot(ri + k0, rj + k0, j - k0)) / ljj;
            }
        }
        if (k1 == n) break;
        // panel: rows below solve A[i, k0:k1] L11^T = A[i, k0:k1]
        parallel_for(static_cast<std::size_t>(n - k1), 8, [&](std::size_t b, std::size_t e) {
            for (std::size_t ii = b; ii < e; ++ii) {
                double* ri = A.row(k1 + i64(ii));
                for (i64 j = k0; j < k1; ++j) {
                    const double* rj = A.row(j);
                    ri[j] = (ri[j] - dot(ri + k0, rj + k0, j - k0)) / rj[j];
                }
            }
        });
        // trailing update of the lower triangle: A[i][j] -= A[i, k0:k1] . A[j, k0:k1]
        parallel_for(static_cast<std::size_t>(n - k1), 4, [&](std::size_t b, std::size_t e) {
            for (std::size_t ii = b; ii < e; ++ii) {
                const i64 i = k1 + i64(ii);
                double* ri = A.row(i);
                const double* pi = ri + k0;
                for (i64 j = k1; j <= i; ++j) {
                    ri[j] -= dot(pi, A.row(j) + k0, k1 - k0);
                }
            }
        });
    }
    for (i64 i = 0; i < n; ++i) for (i64 j = i + 1; j < n; ++j) A(i, j) = 0;
    return true;
}

// Cholesky with escalating damping (H + lambda I); returns the damping used. Dead (zero) diagonals become 1.
inline double cholesky_damped(Sym& A, double rel_damp) {
    Sym base = A;
    double md = 0; i64 cnt = 0;
    for (i64 i = 0; i < A.n; ++i) if (base(i, i) > 0) { md += base(i, i); ++cnt; }
    md = cnt ? md / double(cnt) : 1.0;
    for (int attempt = 0; attempt < 12; ++attempt) {
        const double lam = rel_damp * md * std::pow(10.0, attempt);
        A = base;
        for (i64 i = 0; i < A.n; ++i) { if (A(i, i) <= 0) A(i, i) = md; A(i, i) += lam; }
        if (cholesky(A)) return lam;
    }
    die("cholesky failed even with heavy damping");
}

// Lower-triangular L -> (L^{-1})^T stored row-major as U (upper): U = L^{-T}.
inline Sym tri_inverse_T(const Sym& L) {
    const i64 n = L.n;
    Sym U(n);
    // column j of L^{-1} solves L x = e_j; x is stored as row j of U (U[j][i] = Linv[i][j], i >= j)
    parallel_for(static_cast<std::size_t>(n), 4, [&](std::size_t b, std::size_t e) {
        for (std::size_t jj = b; jj < e; ++jj) {
            const i64 j = i64(jj);
            double* x = U.row(j);
            for (i64 i = j; i < n; ++i) {
                const double* li = L.row(i);
                x[i] = ((i == j ? 1.0 : 0.0) - dot(li + j, x + j, i - j)) / li[i];
            }
        }
    });
    return U;
}

// H^{-1} from its lower Cholesky factor: H^{-1} = L^{-T} L^{-1}.
inline Sym inverse_from_cholesky(const Sym& L) {
    const i64 n = L.n;
    Sym U = tri_inverse_T(L);   // U[a][i] = Linv[i][a]
    Sym Hi(n);
    parallel_for(static_cast<std::size_t>(n), 4, [&](std::size_t b, std::size_t e) {
        for (std::size_t aa = b; aa < e; ++aa) {
            const i64 a = i64(aa);
            const double* ua = U.row(a);
            for (i64 c = 0; c <= a; ++c) {
                Hi(a, c) = dot(ua + a, U.row(c) + a, n - a);
            }
        }
    });
    for (i64 a = 0; a < n; ++a) for (i64 c = a + 1; c < n; ++c) Hi(a, c) = Hi(c, a);
    return Hi;
}

// Upper Cholesky factor of H^{-1} (SparseGPT / OBS): U with H^{-1} = U^T U. Row i of U, from i on, holds the
// inverse-Hessian of the columns not yet eliminated.
inline Sym inverse_upper_cholesky(Sym H, double rel_damp, double* used_damp = nullptr) {
    double lam = cholesky_damped(H, rel_damp);
    if (used_damp) *used_damp = lam;
    Sym Hi = inverse_from_cholesky(H);
    if (!cholesky(Hi)) { double l2 = cholesky_damped(Hi, 1e-8); (void)l2; }
    Sym U(Hi.n);
    for (i64 i = 0; i < Hi.n; ++i) for (i64 j = i; j < Hi.n; ++j) U(i, j) = Hi(j, i);
    return U;
}

// Solves (H_SS + lam I) X^T = B^T for every row of B [rows, |S|] in place (rows independent, parallel).
inline void spd_solve_rows(const Sym& Hs, double rel_damp, float* B, i64 rows) {
    Sym L = Hs;
    cholesky_damped(L, rel_damp);
    const i64 n = L.n;
    Sym U(n);   // L^T for contiguous back substitution
    for (i64 i = 0; i < n; ++i) for (i64 j = 0; j <= i; ++j) U(j, i) = L(i, j);
    parallel_for(static_cast<std::size_t>(rows), 4, [&](std::size_t b, std::size_t e) {
        std::vector<double> y(static_cast<std::size_t>(n));
        for (std::size_t r = b; r < e; ++r) {
            float* x = B + i64(r) * n;
            for (i64 i = 0; i < n; ++i) {
                const double* li = L.row(i);
                y[std::size_t(i)] = (double(x[i]) - dot(li, y.data(), i)) / li[i];
            }
            for (i64 i = n - 1; i >= 0; --i) {
                const double* ui = U.row(i);
                y[std::size_t(i)] = (y[std::size_t(i)] - dot(ui + i + 1, y.data() + i + 1, n - i - 1)) / ui[i];
            }
            for (i64 i = 0; i < n; ++i) x[i] = float(y[std::size_t(i)]);
        }
    });
}

inline Sym submatrix(const Sym& H, const std::vector<i64>& S) {
    Sym R(i64(S.size()));
    for (std::size_t i = 0; i < S.size(); ++i) for (std::size_t j = 0; j < S.size(); ++j) R(i64(i), i64(j)) = H(S[i], S[j]);
    return R;
}

inline std::vector<float> to_f32(const Sym& H) { std::vector<float> f(H.a.size()); for (std::size_t i = 0; i < f.size(); ++i) f[i] = float(H.a[i]); return f; }

// Least-squares refit of a linear layer onto a column subset: W' = argmin ||W X - W' X_S||^2 = (W H_{.,S}) (H_SS)^-1.
// W [rows, n] (row-major), returns W' [rows, |S|].
inline std::vector<float> ls_refit_columns(const float* W, i64 rows, const Sym& H, const std::vector<i64>& S, double rel_damp = 1e-4) {
    const i64 n = H.n, k = i64(S.size());
    std::vector<float> HS(static_cast<std::size_t>(k * n));   // rows S of H
    for (i64 i = 0; i < k; ++i) for (i64 j = 0; j < n; ++j) HS[std::size_t(i * n + j)] = float(H(S[std::size_t(i)], j));
    std::vector<float> out(static_cast<std::size_t>(rows * k));
    gemm_nt(W, HS.data(), out.data(), rows, k, n);   // W H_{.,S}
    spd_solve_rows(submatrix(H, S), rel_damp, out.data(), rows);
    return out;
}

// Least squares with a separate target: M = argmin sum ||y - M x||^2 = (Y^T X)(X^T X)^-1, from accumulated
// cross-moment C = Y^T X [dy, dx] and Gram G = X^T X. Returns M [dy, dx].
inline std::vector<float> ls_solve(const std::vector<double>& C, i64 dy, const Sym& G, double rel_damp = 1e-4) {
    std::vector<float> M(C.size());
    for (std::size_t i = 0; i < C.size(); ++i) M[i] = float(C[i]);
    spd_solve_rows(G, rel_damp, M.data(), dy);
    return M;
}

// Cross moment accumulator C += Y^T X  (Y [rows, dy], X [rows, dx]).
struct Cross {
    i64 dy = 0, dx = 0;
    std::vector<double> C;
    std::vector<float> yt, xt, tmp;
    Cross(i64 dy_ = 0, i64 dx_ = 0) : dy(dy_), dx(dx_), C(static_cast<std::size_t>(dy_ * dx_), 0.0) {}
    void add(const float* Y, const float* X, i64 rows) {
        constexpr i64 chunk = 4096;
        for (i64 r0 = 0; r0 < rows; r0 += chunk) {
            const i64 r = std::min(chunk, rows - r0);
            yt.resize(static_cast<std::size_t>(dy * r)); xt.resize(static_cast<std::size_t>(dx * r)); tmp.resize(static_cast<std::size_t>(dy * dx));
            transpose(Y + r0 * dy, yt.data(), r, dy);
            transpose(X + r0 * dx, xt.data(), r, dx);
            gemm_nt(yt.data(), xt.data(), tmp.data(), dy, dx, r);
            parallel_for(static_cast<std::size_t>(dy), 16, [&](std::size_t b, std::size_t e) {
                for (std::size_t i = b; i < e; ++i) for (i64 j = 0; j < dx; ++j) C[i * std::size_t(dx) + std::size_t(j)] += double(tmp[i * std::size_t(dx) + std::size_t(j)]);
            });
        }
    }
};

// Symmetric eigendecomposition through FORGE's linalg (Householder + implicit QL), for low-rank fits.
}  // namespace forge::compress::la
