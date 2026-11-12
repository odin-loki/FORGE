#pragma once
// Dense linear algebra on row-major float / double matrices, for small and medium sizes.
//   products      matmul, matvec (float: blocked over the simd GEMM kernels; double: blocked, rows combined
//                 with the dispatched f64 axpy), transpose, identity
//   factorisations lu / lu_factor (partial pivoting), cholesky, qr (explicit Q) and qr_factor (compact
//                 Householder), svd (one-sided Jacobi), sym_eig (cyclic Jacobi), eigh (Householder
//                 tridiagonalisation + implicit QL; Jacobi below 16 x 16)
//   solvers       solve, lu_solve, cholesky_solve, solve_triangular, inverse, det, lstsq (QR; SVD when rank
//                 deficient or underdetermined), pinv
//   norms         norm_fro, norm_1, norm_inf, norm_2, cond_2 (via SVD), rcond_1 (Hager / Higham estimate from LU)
// Sparse formats and the iterative solvers with preconditioners are in sparse.hpp; csr, spmv, cg and gmres
// stay here for existing callers. The factorisations are unblocked O(n^3) loops: fine up to a few thousand
// rows, not a LAPACK replacement. Link forge_simd_mv when using matmul / matvec.
#include <forge/simd/dispatch.hpp>
#include <forge/simd/tier_a.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace forge::linalg {
template <class T> struct mat {
    std::size_t r = 0, c = 0; std::vector<T> a;
    mat() = default; mat(std::size_t r_, std::size_t c_) : r(r_), c(c_), a(r_ * c_) {}
    T& operator()(std::size_t i, std::size_t j) { return a[i * c + j]; }
    const T& operator()(std::size_t i, std::size_t j) const { return a[i * c + j]; }
    T* row(std::size_t i) { return a.data() + i * c; }
    const T* row(std::size_t i) const { return a.data() + i * c; }
};
template <class T> mat<T> identity(std::size_t n) { mat<T> I(n, n); for (std::size_t i = 0; i < n; ++i) I(i, i) = T(1); return I; }
template <class T> mat<T> transpose(const mat<T>& A) {
    mat<T> B(A.c, A.r);
    if constexpr (std::is_same_v<T, float>) simd::tier_a_active().transpose_f32(A.a.data(), B.a.data(), A.r, A.c);
    else if constexpr (std::is_same_v<T, double>) simd::tier_a_active().transpose_f64(A.a.data(), B.a.data(), A.r, A.c);
    else for (std::size_t i = 0; i < A.r; ++i) for (std::size_t j = 0; j < A.c; ++j) B(j, i) = A(i, j);
    return B;
}

namespace detail {
// C[m,n] = A[m,k] * B[k,n], float. Small products go straight to the GEMM kernel; larger ones are cut into
// panels (k by 256, n by 1024, m by 256) that are packed contiguously so the kernel's working set stays in
// cache, and the partial products are accumulated.
inline void gemm_f32(const float* A, const float* B, float* C, std::size_t m, std::size_t n, std::size_t k) {
    const auto& K = simd::active();
    constexpr std::size_t KB = 256, NB = 1024, MB = 256;
    if (m == 0 || n == 0) return;
    if (k == 0) { std::fill(C, C + m * n, 0.0f); return; }
    if (k <= KB && n <= NB) { K.gemm_nn(A, B, C, m, n, k); return; }
    std::fill(C, C + m * n, 0.0f);
    std::vector<float> pa(MB * KB), pb(KB * NB), t(MB * NB);
    for (std::size_t k0 = 0; k0 < k; k0 += KB) {
        const std::size_t kb = std::min(KB, k - k0);
        for (std::size_t n0 = 0; n0 < n; n0 += NB) {
            const std::size_t nb = std::min(NB, n - n0);
            for (std::size_t p = 0; p < kb; ++p) std::copy_n(B + (k0 + p) * n + n0, nb, pb.data() + p * nb);
            for (std::size_t m0 = 0; m0 < m; m0 += MB) {
                const std::size_t mb = std::min(MB, m - m0);
                for (std::size_t i = 0; i < mb; ++i) std::copy_n(A + (m0 + i) * k + k0, kb, pa.data() + i * kb);
                K.gemm_nn(pa.data(), pb.data(), t.data(), mb, nb, kb);
                for (std::size_t i = 0; i < mb; ++i) { float* c = C + (m0 + i) * n + n0; const float* s = t.data() + i * nb; for (std::size_t j = 0; j < nb; ++j) c[j] += s[j]; }
            }
        }
    }
}
// C = A * B, double: row i of C is a combination of rows of B (axpy), with k cut so the rows of B in use fit
// in about 512 KB.
inline void gemm_f64(const double* A, const double* B, double* C, std::size_t m, std::size_t n, std::size_t k) {
    std::fill(C, C + m * n, 0.0);
    if (m == 0 || n == 0 || k == 0) return;
    const auto axpy = simd::tier_a_active().axpy_f64;
    const std::size_t KB = std::max<std::size_t>(8, 65536 / n);
    for (std::size_t k0 = 0; k0 < k; k0 += KB) {
        const std::size_t k1 = std::min(k, k0 + KB);
        for (std::size_t i = 0; i < m; ++i) {
            double* c = C + i * n; const double* a = A + i * k;
            if (n < 8) { for (std::size_t p = k0; p < k1; ++p) { const double v = a[p]; const double* b = B + p * n; for (std::size_t j = 0; j < n; ++j) c[j] += v * b[j]; } }
            else for (std::size_t p = k0; p < k1; ++p) axpy(a[p], B + p * n, c, n);
        }
    }
}
template <class T> T eps() { return std::numeric_limits<T>::epsilon(); }
} // namespace detail

template <class T> mat<T> matmul(const mat<T>& A, const mat<T>& B) {
    if (A.c != B.r) throw std::invalid_argument("matmul: inner dimensions differ");
    mat<T> C(A.r, B.c);
    if constexpr (std::is_same_v<T, float>) detail::gemm_f32(A.a.data(), B.a.data(), C.a.data(), A.r, B.c, A.c);
    else if constexpr (std::is_same_v<T, double>) detail::gemm_f64(A.a.data(), B.a.data(), C.a.data(), A.r, B.c, A.c);
    else for (std::size_t i = 0; i < A.r; ++i) for (std::size_t k = 0; k < A.c; ++k) { T v = A(i, k); for (std::size_t j = 0; j < B.c; ++j) C(i, j) += v * B(k, j); }
    return C;
}
// The plain triple loop (reference for tests, and for element types without kernels).
template <class T> mat<T> matmul_naive(const mat<T>& A, const mat<T>& B) {
    if (A.c != B.r) throw std::invalid_argument("matmul: inner dimensions differ");
    mat<T> C(A.r, B.c);
    for (std::size_t i = 0; i < A.r; ++i) for (std::size_t k = 0; k < A.c; ++k) { T v = A(i, k); for (std::size_t j = 0; j < B.c; ++j) C(i, j) += v * B(k, j); }
    return C;
}
template <class T> std::vector<T> matvec(const mat<T>& A, const std::vector<T>& x) {
    if (A.c != x.size()) throw std::invalid_argument("matvec: dimensions differ");
    std::vector<T> y(A.r);
    if constexpr (std::is_same_v<T, float>) { if (A.r && A.c) simd::active().gemm_nt(A.a.data(), x.data(), y.data(), A.r, 1, A.c); }
    else if constexpr (std::is_same_v<T, double>) { const auto dot = simd::tier_a_active().dot_f64; for (std::size_t i = 0; i < A.r; ++i) y[i] = dot(A.row(i), x.data(), A.c); }
    else for (std::size_t i = 0; i < A.r; ++i) { T s = 0; for (std::size_t j = 0; j < A.c; ++j) s += A(i, j) * x[j]; y[i] = s; }
    return y;
}

// ---- norms ----
template <class T> T norm2(const std::vector<T>& v) {   // Euclidean, scaled against overflow / underflow
    T scale = 0; for (T e : v) scale = std::max(scale, std::fabs(e));
    if (scale == T(0) || !std::isfinite(scale)) return scale;
    T s = 0; for (T e : v) { T q = e / scale; s += q * q; }
    return scale * std::sqrt(s);
}
template <class T> T norm_fro(const mat<T>& A) { return norm2(A.a); }
template <class T> T norm_1(const mat<T>& A) {     // largest column sum
    std::vector<T> cs(A.c, T(0)); for (std::size_t i = 0; i < A.r; ++i) for (std::size_t j = 0; j < A.c; ++j) cs[j] += std::fabs(A(i, j));
    T m = 0; for (T v : cs) m = std::max(m, v); return m;
}
template <class T> T norm_inf(const mat<T>& A) {   // largest row sum
    T m = 0; for (std::size_t i = 0; i < A.r; ++i) { T s = 0; for (std::size_t j = 0; j < A.c; ++j) s += std::fabs(A(i, j)); m = std::max(m, s); } return m;
}

// ---- LU ----
// LU with partial pivoting, in place: row i of L U is row p[i] of A (unit lower L below the diagonal, U on
// and above). Returns the permutation; throws on an exactly singular matrix.
template <class T> std::vector<std::size_t> lu(mat<T>& A) {
    if (A.r != A.c) throw std::invalid_argument("lu: matrix is not square");
    std::size_t n = A.r; std::vector<std::size_t> p(n); for (std::size_t i = 0; i < n; ++i) p[i] = i;
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t m = k; for (std::size_t i = k + 1; i < n; ++i) if (std::fabs(A(i, k)) > std::fabs(A(m, k))) m = i;
        if (A(m, k) == T(0)) throw std::runtime_error("lu: singular");
        if (m != k) { for (std::size_t j = 0; j < n; ++j) std::swap(A(k, j), A(m, j)); std::swap(p[k], p[m]); }
        const T* rk = A.row(k);
        for (std::size_t i = k + 1; i < n; ++i) { T* ri = A.row(i); ri[k] /= rk[k]; const T f = ri[k]; for (std::size_t j = k + 1; j < n; ++j) ri[j] -= f * rk[j]; }
    }
    return p;
}
template <class T> std::vector<T> lu_solve(const mat<T>& LU, const std::vector<std::size_t>& p, const std::vector<T>& b) {
    std::size_t n = LU.r; std::vector<T> x(n);
    for (std::size_t i = 0; i < n; ++i) { T s = b[p[i]]; for (std::size_t j = 0; j < i; ++j) s -= LU(i, j) * x[j]; x[i] = s; }
    for (std::size_t i = n; i-- > 0;) { T s = x[i]; for (std::size_t j = i + 1; j < n; ++j) s -= LU(i, j) * x[j]; x[i] = s / LU(i, i); }
    return x;
}
// Solves A^T x = b from the factors of A.
template <class T> std::vector<T> lu_solve_transposed(const mat<T>& LU, const std::vector<std::size_t>& p, const std::vector<T>& b) {
    std::size_t n = LU.r; std::vector<T> w(b);
    for (std::size_t i = 0; i < n; ++i) { T s = w[i]; for (std::size_t j = 0; j < i; ++j) s -= LU(j, i) * w[j]; w[i] = s / LU(i, i); }   // U^T y = b
    for (std::size_t i = n; i-- > 0;) { T s = w[i]; for (std::size_t j = i + 1; j < n; ++j) s -= LU(j, i) * w[j]; w[i] = s; }             // L^T z = y
    std::vector<T> x(n); for (std::size_t i = 0; i < n; ++i) x[p[i]] = w[i];
    return x;
}
template <class T> struct lu_factors {
    mat<T> LU; std::vector<std::size_t> perm; int sign = 1;   // sign of the permutation
    std::vector<T> solve(const std::vector<T>& b) const { return lu_solve(LU, perm, b); }
    mat<T> solve(const mat<T>& B) const {   // one right-hand side per column
        mat<T> X(B.r, B.c); std::vector<T> col(B.r);
        for (std::size_t j = 0; j < B.c; ++j) { for (std::size_t i = 0; i < B.r; ++i) col[i] = B(i, j); auto x = lu_solve(LU, perm, col); for (std::size_t i = 0; i < B.r; ++i) X(i, j) = x[i]; }
        return X;
    }
    T det() const { T d = T(sign); for (std::size_t i = 0; i < LU.r; ++i) d *= LU(i, i); return d; }
};
template <class T> lu_factors<T> lu_factor(mat<T> A) {
    lu_factors<T> f; f.perm = lu(A); f.LU = std::move(A);
    std::vector<bool> seen(f.perm.size(), false);   // parity = product over cycles of (-1)^(length - 1)
    for (std::size_t i = 0; i < f.perm.size(); ++i) { if (seen[i]) continue; std::size_t len = 0; for (std::size_t j = i; !seen[j]; j = f.perm[j]) { seen[j] = true; ++len; } if (len % 2 == 0) f.sign = -f.sign; }
    return f;
}
template <class T> std::vector<T> solve(const mat<T>& A, const std::vector<T>& b) { return lu_factor(A).solve(b); }
template <class T> mat<T> solve(const mat<T>& A, const mat<T>& B) { return lu_factor(A).solve(B); }
template <class T> mat<T> inverse(const mat<T>& A) { return lu_factor(A).solve(identity<T>(A.r)); }
// Determinant; 0 for an exactly singular matrix.
template <class T> T det(const mat<T>& A) {
    if (A.r != A.c) throw std::invalid_argument("det: matrix is not square");
    if (A.r == 0) return T(1);
    try { return lu_factor(A).det(); } catch (const std::runtime_error&) { return T(0); }
}
// Estimate of 1 / (||A||_1 ||A^-1||_1) from the factors (Hager's method as refined by Higham, at most five
// solves with A and A^T). It never overestimates the condition number's reciprocal by much and is usually
// within a factor of 3 of the truth; a_norm1 = norm_1 of the original matrix.
template <class T> T rcond_1(const lu_factors<T>& f, T a_norm1) {
    const std::size_t n = f.LU.r; if (n == 0) return T(1);
    if (a_norm1 == T(0)) return T(0);
    std::vector<T> x(n, T(1) / T(n)); T est = 0; std::size_t last = n;
    for (int it = 0; it < 5; ++it) {
        auto y = lu_solve(f.LU, f.perm, x);
        T e = 0; for (T v : y) e += std::fabs(v);
        if (it > 0 && e <= est) break;
        est = e;
        std::vector<T> xi(n); for (std::size_t i = 0; i < n; ++i) xi[i] = y[i] >= 0 ? T(1) : T(-1);
        auto z = lu_solve_transposed(f.LU, f.perm, xi);
        std::size_t j = 0; T zx = 0; for (std::size_t i = 0; i < n; ++i) { if (std::fabs(z[i]) > std::fabs(z[j])) j = i; zx += z[i] * x[i]; }
        if (std::fabs(z[j]) <= zx || j == last) break;
        std::fill(x.begin(), x.end(), T(0)); x[j] = T(1); last = j;
    }
    return est > 0 ? T(1) / (a_norm1 * est) : T(0);
}

// ---- triangular solves ----
struct tri { bool lower = true, transposed = false, unit_diagonal = false; };
// Solves op(A) x = b using only the lower (or upper) triangle of A; op(A) = A or A^T.
template <class T> std::vector<T> solve_triangular(const mat<T>& A, std::vector<T> b, tri o = {}) {
    const std::size_t n = A.r;
    if (A.c != n || b.size() != n) throw std::invalid_argument("solve_triangular: dimensions differ");
    const bool forward = o.lower != o.transposed;
    auto el = [&](std::size_t i, std::size_t j) { return o.transposed ? A(j, i) : A(i, j); };
    for (std::size_t s = 0; s < n; ++s) {
        const std::size_t i = forward ? s : n - 1 - s;
        T v = b[i];
        if (forward) for (std::size_t j = 0; j < i; ++j) v -= el(i, j) * b[j];
        else for (std::size_t j = i + 1; j < n; ++j) v -= el(i, j) * b[j];
        if (!o.unit_diagonal) { if (A(i, i) == T(0)) throw std::runtime_error("solve_triangular: zero on the diagonal"); v /= A(i, i); }
        b[i] = v;
    }
    return b;
}

// ---- Cholesky ----
// A = L L^T (lower). Throws if A is not positive definite. Only the lower triangle of A is read.
template <class T> mat<T> cholesky(const mat<T>& A) {
    if (A.r != A.c) throw std::invalid_argument("cholesky: matrix is not square");
    std::size_t n = A.r; mat<T> L(n, n);
    for (std::size_t j = 0; j < n; ++j) {
        T d = A(j, j); for (std::size_t k = 0; k < j; ++k) d -= L(j, k) * L(j, k);
        if (!(d > T(0))) throw std::runtime_error("cholesky: not positive definite");
        L(j, j) = std::sqrt(d);
        for (std::size_t i = j + 1; i < n; ++i) { T s = A(i, j); const T* li = L.row(i); const T* lj = L.row(j); for (std::size_t k = 0; k < j; ++k) s -= li[k] * lj[k]; L(i, j) = s / L(j, j); }
    }
    return L;
}
template <class T> std::vector<T> cholesky_solve(const mat<T>& L, const std::vector<T>& b) {
    return solve_triangular(L, solve_triangular(L, b, {.lower = true}), {.lower = true, .transposed = true});
}

// ---- QR ----
// Householder QR of an m x n matrix: Q is m x m orthogonal, R is m x n upper triangular (trapezoidal when m < n).
template <class T> void qr(const mat<T>& A, mat<T>& Q, mat<T>& R) {
    std::size_t m = A.r, n = A.c; R = A; Q = mat<T>(m, m); for (std::size_t i = 0; i < m; ++i) Q(i, i) = 1;
    std::vector<T> v(m);
    for (std::size_t k = 0; k < n && k + 1 < m + (m > n); ++k) {
        T nrm = 0; for (std::size_t i = k; i < m; ++i) nrm += R(i, k) * R(i, k); nrm = std::sqrt(nrm);
        if (nrm == T(0)) continue;
        T alpha = R(k, k) > 0 ? -nrm : nrm;
        for (std::size_t i = 0; i < m; ++i) v[i] = i < k ? 0 : R(i, k); v[k] -= alpha;
        T vv = 0; for (std::size_t i = k; i < m; ++i) vv += v[i] * v[i]; if (vv == T(0)) continue;
        for (std::size_t j = 0; j < n; ++j) { T s = 0; for (std::size_t i = k; i < m; ++i) s += v[i] * R(i, j); s = 2 * s / vv; for (std::size_t i = k; i < m; ++i) R(i, j) -= s * v[i]; }
        for (std::size_t j = 0; j < m; ++j) { T s = 0; for (std::size_t i = k; i < m; ++i) s += Q(j, i) * v[i]; s = 2 * s / vv; for (std::size_t i = k; i < m; ++i) Q(j, i) -= s * v[i]; }
    }
}
// Compact form: R on and above the diagonal of QR, the Householder vectors below it (v_k = [1, QR(k+1.., k)]),
// H_k = I - tau[k] v_k v_k^T, A = H_0 H_1 ... R.
template <class T> struct qr_factors {
    mat<T> QR; std::vector<T> tau;
    void apply_qt(std::vector<T>& b) const {   // b <- Q^T b
        const std::size_t m = QR.r;
        for (std::size_t k = 0; k < tau.size(); ++k) {
            if (tau[k] == T(0)) continue;
            T s = b[k]; for (std::size_t i = k + 1; i < m; ++i) s += QR(i, k) * b[i];
            s *= tau[k]; b[k] -= s; for (std::size_t i = k + 1; i < m; ++i) b[i] -= s * QR(i, k);
        }
    }
    void apply_q(std::vector<T>& b) const {    // b <- Q b
        const std::size_t m = QR.r;
        for (std::size_t k = tau.size(); k-- > 0;) {
            if (tau[k] == T(0)) continue;
            T s = b[k]; for (std::size_t i = k + 1; i < m; ++i) s += QR(i, k) * b[i];
            s *= tau[k]; b[k] -= s; for (std::size_t i = k + 1; i < m; ++i) b[i] -= s * QR(i, k);
        }
    }
    mat<T> r() const { std::size_t k = std::min(QR.r, QR.c); mat<T> R(k, QR.c); for (std::size_t i = 0; i < k; ++i) for (std::size_t j = i; j < QR.c; ++j) R(i, j) = QR(i, j); return R; }
    mat<T> thin_q() const {   // m x min(m, n), orthonormal columns
        const std::size_t m = QR.r, k = std::min(QR.r, QR.c); mat<T> Q(m, k); std::vector<T> e(m);
        for (std::size_t j = 0; j < k; ++j) { std::fill(e.begin(), e.end(), T(0)); e[j] = 1; apply_q(e); for (std::size_t i = 0; i < m; ++i) Q(i, j) = e[i]; }
        return Q;
    }
};
template <class T> qr_factors<T> qr_factor(mat<T> A) {
    const std::size_t m = A.r, n = A.c, steps = std::min(m, n);
    qr_factors<T> f; f.tau.assign(steps, T(0));
    for (std::size_t k = 0; k < steps; ++k) {
        T scale = 0; for (std::size_t i = k; i < m; ++i) scale = std::max(scale, std::fabs(A(i, k)));
        if (scale == T(0)) continue;
        T nrm = 0; for (std::size_t i = k; i < m; ++i) { T q = A(i, k) / scale; nrm += q * q; } nrm = scale * std::sqrt(nrm);
        const T alpha = A(k, k) > 0 ? -nrm : nrm, v0 = A(k, k) - alpha;
        if (v0 == T(0)) continue;   // the column is already alpha * e_k
        for (std::size_t i = k + 1; i < m; ++i) A(i, k) /= v0;
        f.tau[k] = -v0 / alpha;     // (alpha - a_kk) / alpha
        A(k, k) = alpha;
        for (std::size_t j = k + 1; j < n; ++j) {
            T s = A(k, j); for (std::size_t i = k + 1; i < m; ++i) s += A(i, k) * A(i, j);
            s *= f.tau[k]; A(k, j) -= s; for (std::size_t i = k + 1; i < m; ++i) A(i, j) -= s * A(i, k);
        }
    }
    f.QR = std::move(A);
    return f;
}

// ---- symmetric eigenproblem ----
// Cyclic Jacobi: A = V diag(w) V^T, eigenvalues ascending. Accurate and simple, O(n^3) per sweep.
template <class T> void sym_eig(mat<T> A, std::vector<T>& w, mat<T>& V, int sweeps = 100) {
    std::size_t n = A.r; V = mat<T>(n, n); for (std::size_t i = 0; i < n; ++i) V(i, i) = 1;
    T total = 0; for (T v : A.a) total += v * v;
    const T stop = total * detail::eps<T>() * detail::eps<T>() * T(1e-2);   // off-diagonal norm below 0.1 * epsilon * ||A||_F
    for (int s = 0; s < sweeps; ++s) {
        T off = 0; for (std::size_t i = 0; i < n; ++i) for (std::size_t j = i + 1; j < n; ++j) off += A(i, j) * A(i, j);
        if (off <= stop) break;
        for (std::size_t p = 0; p < n; ++p) for (std::size_t q = p + 1; q < n; ++q) {
            if (A(p, q) == T(0)) continue;
            T th = (A(q, q) - A(p, p)) / (2 * A(p, q));
            T t = (th >= 0 ? T(1) : T(-1)) / (std::fabs(th) + std::sqrt(th * th + 1)), c = 1 / std::sqrt(t * t + 1), sn = t * c;
            for (std::size_t k = 0; k < n; ++k) { T akp = A(k, p), akq = A(k, q); A(k, p) = c * akp - sn * akq; A(k, q) = sn * akp + c * akq; }
            for (std::size_t k = 0; k < n; ++k) { T apk = A(p, k), aqk = A(q, k); A(p, k) = c * apk - sn * aqk; A(q, k) = sn * apk + c * aqk; }
            for (std::size_t k = 0; k < n; ++k) { T vkp = V(k, p), vkq = V(k, q); V(k, p) = c * vkp - sn * vkq; V(k, q) = sn * vkp + c * vkq; }
        }
    }
    std::vector<std::size_t> o(n); for (std::size_t i = 0; i < n; ++i) o[i] = i;
    std::sort(o.begin(), o.end(), [&](std::size_t x, std::size_t y) { return A(x, x) < A(y, y); });
    mat<T> V2(n, n); w.assign(n, 0);
    for (std::size_t j = 0; j < n; ++j) { w[j] = A(o[j], o[j]); for (std::size_t i = 0; i < n; ++i) V2(i, j) = V(i, o[j]); }
    V = V2;
}
namespace detail {
// Householder reduction to tridiagonal form (EISPACK tred2): on return V holds the accumulated orthogonal
// transform, d the diagonal and e the sub-diagonal (e[0] = 0).
template <class T> void tridiagonalise(mat<T>& V, std::vector<T>& d, std::vector<T>& e) {
    const std::size_t n = V.r;
    for (std::size_t j = 0; j < n; ++j) d[j] = V(n - 1, j);
    for (std::size_t i = n - 1; i > 0; --i) {
        T scale = 0, h = 0;
        for (std::size_t k = 0; k < i; ++k) scale += std::fabs(d[k]);
        if (scale == T(0)) {
            e[i] = d[i - 1];
            for (std::size_t j = 0; j < i; ++j) { d[j] = V(i - 1, j); V(i, j) = 0; V(j, i) = 0; }
        } else {
            for (std::size_t k = 0; k < i; ++k) { d[k] /= scale; h += d[k] * d[k]; }
            T f = d[i - 1], g = std::sqrt(h); if (f > 0) g = -g;
            e[i] = scale * g; h -= f * g; d[i - 1] = f - g;
            for (std::size_t j = 0; j < i; ++j) e[j] = 0;
            for (std::size_t j = 0; j < i; ++j) {
                f = d[j]; V(j, i) = f; g = e[j] + V(j, j) * f;
                for (std::size_t k = j + 1; k < i; ++k) { g += V(k, j) * d[k]; e[k] += V(k, j) * f; }
                e[j] = g;
            }
            f = 0;
            for (std::size_t j = 0; j < i; ++j) { e[j] /= h; f += e[j] * d[j]; }
            const T hh = f / (h + h);
            for (std::size_t j = 0; j < i; ++j) e[j] -= hh * d[j];
            for (std::size_t j = 0; j < i; ++j) {
                f = d[j]; g = e[j];
                for (std::size_t k = j; k < i; ++k) V(k, j) -= f * e[k] + g * d[k];
                d[j] = V(i - 1, j); V(i, j) = 0;
            }
        }
        d[i] = h;
    }
    for (std::size_t i = 0; i + 1 < n; ++i) {
        V(n - 1, i) = V(i, i); V(i, i) = 1;
        const T h = d[i + 1];
        if (h != T(0)) {
            for (std::size_t k = 0; k <= i; ++k) d[k] = V(k, i + 1) / h;
            for (std::size_t j = 0; j <= i; ++j) {
                T g = 0; for (std::size_t k = 0; k <= i; ++k) g += V(k, i + 1) * V(k, j);
                for (std::size_t k = 0; k <= i; ++k) V(k, j) -= g * d[k];
            }
        }
        for (std::size_t k = 0; k <= i; ++k) V(k, i + 1) = 0;
    }
    for (std::size_t j = 0; j < n; ++j) { d[j] = V(n - 1, j); V(n - 1, j) = 0; }
    V(n - 1, n - 1) = 1; e[0] = 0;
}
// Implicit QL iterations on the tridiagonal matrix (EISPACK tql2), rotating the columns of V along.
template <class T> void tridiagonal_ql(mat<T>& V, std::vector<T>& d, std::vector<T>& e) {
    const std::size_t n = V.r;
    for (std::size_t i = 1; i < n; ++i) e[i - 1] = e[i];
    e[n - 1] = 0;
    T f = 0, tst1 = 0; const T eps = detail::eps<T>();
    for (std::size_t l = 0; l < n; ++l) {
        tst1 = std::max(tst1, std::fabs(d[l]) + std::fabs(e[l]));
        std::size_t m = l;
        while (m < n - 1 && std::fabs(e[m]) > eps * tst1) ++m;
        if (m > l) {
            int iter = 0;
            do {
                if (++iter > 300) throw std::runtime_error("eigh: QL iteration did not converge");
                T g = d[l], p = (d[l + 1] - g) / (2 * e[l]), r = std::hypot(p, T(1));
                if (p < 0) r = -r;
                d[l] = e[l] / (p + r); d[l + 1] = e[l] * (p + r);
                const T dl1 = d[l + 1]; T h = g - d[l];
                for (std::size_t i = l + 2; i < n; ++i) d[i] -= h;
                f += h;
                p = d[m]; T c = 1, c2 = c, c3 = c; const T el1 = e[l + 1]; T s = 0, s2 = 0;
                for (std::size_t i = m; i-- > l;) {
                    c3 = c2; c2 = c; s2 = s;
                    g = c * e[i]; h = c * p; r = std::hypot(p, e[i]);
                    e[i + 1] = s * r; s = e[i] / r; c = p / r; p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for (std::size_t k = 0; k < n; ++k) { h = V(k, i + 1); V(k, i + 1) = s * V(k, i) + c * h; V(k, i) = c * V(k, i) - s * h; }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p; d[l] = c * p;
            } while (std::fabs(e[l]) > eps * tst1);
        }
        d[l] += f; e[l] = 0;
    }
}
} // namespace detail
// Symmetric eigendecomposition A = V diag(w) V^T, eigenvalues ascending, eigenvectors in the columns of V.
// Householder tridiagonalisation and implicit QL (about 4n^3/3 + O(n^2) per eigenvalue pass); matrices smaller
// than 16 x 16 use Jacobi. Only the lower triangle of A is read by the tridiagonal path; pass a symmetric A.
template <class T> void eigh(const mat<T>& A, std::vector<T>& w, mat<T>& V) {
    if (A.r != A.c) throw std::invalid_argument("eigh: matrix is not square");
    const std::size_t n = A.r;
    if (n < 16) { sym_eig(A, w, V); return; }
    V = A; w.assign(n, T(0)); std::vector<T> e(n, T(0));
    detail::tridiagonalise(V, w, e);
    detail::tridiagonal_ql(V, w, e);
    std::vector<std::size_t> o(n); for (std::size_t i = 0; i < n; ++i) o[i] = i;
    std::sort(o.begin(), o.end(), [&](std::size_t x, std::size_t y) { return w[x] < w[y]; });
    mat<T> V2(n, n); std::vector<T> w2(n);
    for (std::size_t j = 0; j < n; ++j) { w2[j] = w[o[j]]; for (std::size_t i = 0; i < n; ++i) V2(i, j) = V(i, o[j]); }
    V = std::move(V2); w = std::move(w2);
}

// ---- SVD ----
// Thin SVD by one-sided Jacobi (m >= n): A = U diag(s) V^T, singular values descending. Columns of U that
// belong to zero singular values are zero, not an orthonormal completion.
template <class T> void svd(const mat<T>& A, mat<T>& U, std::vector<T>& s, mat<T>& V, int sweeps = 100) {
    std::size_t m = A.r, n = A.c; if (m < n) throw std::invalid_argument("svd: needs rows >= cols");
    U = A; V = mat<T>(n, n); for (std::size_t i = 0; i < n; ++i) V(i, i) = 1;
    const T tol = 4 * detail::eps<T>();
    for (int sw = 0; sw < sweeps; ++sw) {
        bool rot = false;
        for (std::size_t p = 0; p < n; ++p) for (std::size_t q = p + 1; q < n; ++q) {
            T a = 0, b = 0, g = 0;
            for (std::size_t i = 0; i < m; ++i) { a += U(i, p) * U(i, p); b += U(i, q) * U(i, q); g += U(i, p) * U(i, q); }
            if (std::fabs(g) <= tol * std::sqrt(a * b) || g == 0) continue;
            rot = true;
            T z = (b - a) / (2 * g), t = (z >= 0 ? T(1) : T(-1)) / (std::fabs(z) + std::sqrt(1 + z * z)), c = 1 / std::sqrt(1 + t * t), sn = c * t;
            for (std::size_t i = 0; i < m; ++i) { T up = U(i, p), uq = U(i, q); U(i, p) = c * up - sn * uq; U(i, q) = sn * up + c * uq; }
            for (std::size_t i = 0; i < n; ++i) { T vp = V(i, p), vq = V(i, q); V(i, p) = c * vp - sn * vq; V(i, q) = sn * vp + c * vq; }
        }
        if (!rot) break;
    }
    s.assign(n, 0);
    for (std::size_t j = 0; j < n; ++j) { T nn = 0; for (std::size_t i = 0; i < m; ++i) nn += U(i, j) * U(i, j); s[j] = std::sqrt(nn);
        if (s[j] > 0) for (std::size_t i = 0; i < m; ++i) U(i, j) /= s[j]; }
    std::vector<std::size_t> o(n); for (std::size_t i = 0; i < n; ++i) o[i] = i;
    std::stable_sort(o.begin(), o.end(), [&](std::size_t x, std::size_t y) { return s[x] > s[y]; });
    mat<T> U2(m, n), V2(n, n); std::vector<T> s2(n);
    for (std::size_t j = 0; j < n; ++j) { s2[j] = s[o[j]]; for (std::size_t i = 0; i < m; ++i) U2(i, j) = U(i, o[j]); for (std::size_t i = 0; i < n; ++i) V2(i, j) = V(i, o[j]); }
    U = U2; V = V2; s = s2;
}
// Thin SVD of any shape: U is m x k, V is n x k, k = min(m, n); A = U diag(s) V^T.
template <class T> struct svd_result { mat<T> U; std::vector<T> s; mat<T> V; };
template <class T> svd_result<T> svd_thin(const mat<T>& A) {
    svd_result<T> r;
    if (A.r >= A.c) svd(A, r.U, r.s, r.V);
    else svd(transpose(A), r.V, r.s, r.U);
    return r;
}
template <class T> T norm_2(const mat<T>& A) { if (A.a.empty()) return T(0); return svd_thin(A).s.front(); }
// 2-norm condition number sigma_max / sigma_min; infinity for a singular (or rank-deficient rectangular) matrix.
template <class T> T cond_2(const mat<T>& A) {
    if (A.a.empty()) return T(0);
    auto s = svd_thin(A).s;
    return s.back() > T(0) ? s.front() / s.back() : std::numeric_limits<T>::infinity();
}
// Moore-Penrose pseudo-inverse (n x m). Singular values at or below rcond * sigma_max are treated as zero;
// rcond < 0 selects max(m, n) * epsilon.
template <class T> mat<T> pinv(const mat<T>& A, T rcond = T(-1)) {
    mat<T> P(A.c, A.r);
    if (A.a.empty()) return P;
    auto f = svd_thin(A);
    if (rcond < 0) rcond = T(std::max(A.r, A.c)) * detail::eps<T>();
    const T cut = rcond * f.s.front();
    for (std::size_t k = 0; k < f.s.size(); ++k) {
        if (!(f.s[k] > cut)) continue;
        const T inv = T(1) / f.s[k];
        for (std::size_t i = 0; i < A.c; ++i) { const T v = f.V(i, k) * inv; if (v == T(0)) continue; for (std::size_t j = 0; j < A.r; ++j) P(i, j) += v * f.U(j, k); }
    }
    return P;
}
// Least squares: x minimising ||A x - b||_2. Full column rank with m >= n: Householder QR. Otherwise (rank
// deficient to within max(m, n) * epsilon on R's diagonal, or m < n): the minimum-norm solution through the SVD.
template <class T> std::vector<T> lstsq(const mat<T>& A, const std::vector<T>& b) {
    const std::size_t m = A.r, n = A.c;
    if (b.size() != m) throw std::invalid_argument("lstsq: dimensions differ");
    if (m >= n && n > 0) {
        auto f = qr_factor(A);
        T dmax = 0, dmin = std::numeric_limits<T>::infinity();
        for (std::size_t i = 0; i < n; ++i) { dmax = std::max(dmax, std::fabs(f.QR(i, i))); dmin = std::min(dmin, std::fabs(f.QR(i, i))); }
        if (dmin > T(std::max(m, n)) * detail::eps<T>() * dmax) {
            std::vector<T> y(b); f.apply_qt(y);
            std::vector<T> x(n);
            for (std::size_t i = n; i-- > 0;) { T s = y[i]; for (std::size_t j = i + 1; j < n; ++j) s -= f.QR(i, j) * x[j]; x[i] = s / f.QR(i, i); }
            return x;
        }
    }
    return matvec(pinv(A), b);
}

// ---- sparse (CSR) basics; more formats and solvers in sparse.hpp ----
template <class T> struct csr { std::size_t rows = 0, cols = 0; std::vector<std::size_t> ptr{0}, idx; std::vector<T> val; };
template <class T> csr<T> to_csr(const mat<T>& A) {
    csr<T> s; s.rows = A.r; s.cols = A.c;
    for (std::size_t i = 0; i < A.r; ++i) { for (std::size_t j = 0; j < A.c; ++j) if (A(i, j) != T(0)) { s.idx.push_back(j); s.val.push_back(A(i, j)); } s.ptr.push_back(s.idx.size()); }
    return s;
}
template <class T> void spmv(const csr<T>& A, const T* x, T* y) {
    for (std::size_t i = 0; i < A.rows; ++i) { T s = 0; for (std::size_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) s += A.val[k] * x[A.idx[k]]; y[i] = s; }
}
// Conjugate gradients for SPD A; returns the iteration count.
template <class T> int cg(const csr<T>& A, const std::vector<T>& b, std::vector<T>& x, T tol = T(1e-10), int maxit = 1000) {
    std::size_t n = b.size(); x.assign(n, T(0)); std::vector<T> r = b, p = b, Ap(n);
    auto dot = [&](const std::vector<T>& u, const std::vector<T>& v) { T s = 0; for (std::size_t i = 0; i < n; ++i) s += u[i] * v[i]; return s; };
    T rr = dot(r, r), bb = rr;
    for (int it = 0; it < maxit; ++it) {
        if (std::sqrt(rr) <= tol * std::sqrt(bb)) return it;
        spmv(A, p.data(), Ap.data()); T al = rr / dot(p, Ap);
        for (std::size_t i = 0; i < n; ++i) { x[i] += al * p[i]; r[i] -= al * Ap[i]; }
        T rn = dot(r, r); for (std::size_t i = 0; i < n; ++i) p[i] = r[i] + (rn / rr) * p[i]; rr = rn;
    }
    return maxit;
}
// Restarted GMRES(k) for general (non-symmetric) CSR systems. Returns iterations used, -1 if not converged.
template <class T> int gmres(const csr<T>& A, const std::vector<T>& b, std::vector<T>& x, int restart = 30, T tol = T(1e-10), int maxit = 1000) {
    std::size_t n = b.size(); if (x.size() != n) x.assign(n, 0);
    auto nrm = [](const std::vector<T>& v) { T s = 0; for (T e : v) s += e * e; return std::sqrt(s); };
    T bn = nrm(b); if (bn == 0) bn = 1;
    int it = 0;
    while (it < maxit) {
        std::vector<T> r(n); spmv(A, x.data(), r.data()); for (std::size_t i = 0; i < n; ++i) r[i] = b[i] - r[i];
        T beta = nrm(r); if (beta / bn < tol) return it;
        std::size_t k = std::size_t(restart);
        std::vector<std::vector<T>> Vb(k + 1, std::vector<T>(n)); mat<T> H(k + 1, k);
        std::vector<T> cs(k), sn(k), g(k + 1); g[0] = beta;
        for (std::size_t i = 0; i < n; ++i) Vb[0][i] = r[i] / beta;
        std::size_t j = 0;
        for (; j < k && it < maxit; ++j, ++it) {
            spmv(A, Vb[j].data(), Vb[j + 1].data());
            for (std::size_t i = 0; i <= j; ++i) { T h = 0; for (std::size_t q = 0; q < n; ++q) h += Vb[j + 1][q] * Vb[i][q]; H(i, j) = h;
                for (std::size_t q = 0; q < n; ++q) Vb[j + 1][q] -= h * Vb[i][q]; }
            H(j + 1, j) = nrm(Vb[j + 1]);
            if (H(j + 1, j) > 0) for (auto& e : Vb[j + 1]) e /= H(j + 1, j);
            for (std::size_t i = 0; i < j; ++i) { T t = cs[i] * H(i, j) + sn[i] * H(i + 1, j); H(i + 1, j) = -sn[i] * H(i, j) + cs[i] * H(i + 1, j); H(i, j) = t; }
            T d = std::sqrt(H(j, j) * H(j, j) + H(j + 1, j) * H(j + 1, j)); cs[j] = H(j, j) / d; sn[j] = H(j + 1, j) / d;
            H(j, j) = d; H(j + 1, j) = 0; g[j + 1] = -sn[j] * g[j]; g[j] = cs[j] * g[j];
            if (std::fabs(g[j + 1]) / bn < tol) { ++j; ++it; break; }
        }
        std::vector<T> y(j);
        for (std::size_t i = j; i-- > 0;) { T sum = g[i]; for (std::size_t q = i + 1; q < j; ++q) sum -= H(i, q) * y[q]; y[i] = sum / H(i, i); }
        for (std::size_t i = 0; i < j; ++i) for (std::size_t q = 0; q < n; ++q) x[q] += y[i] * Vb[i][q];
        std::vector<T> r2(n); spmv(A, x.data(), r2.data()); for (std::size_t i = 0; i < n; ++i) r2[i] = b[i] - r2[i];
        if (nrm(r2) / bn < tol) return it;
    }
    return -1;
}
}
