#pragma once
// Dense and sparse linear algebra on row-major double/float matrices: LU (partial pivoting), Cholesky,
// Householder QR, triangular solves, CSR SpMV and conjugate gradients.
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace forge::linalg {
template <class T> struct mat {
    std::size_t r = 0, c = 0; std::vector<T> a;
    mat() = default; mat(std::size_t r_, std::size_t c_) : r(r_), c(c_), a(r_ * c_) {}
    T& operator()(std::size_t i, std::size_t j) { return a[i * c + j]; }
    const T& operator()(std::size_t i, std::size_t j) const { return a[i * c + j]; }
};
template <class T> mat<T> matmul(const mat<T>& A, const mat<T>& B) {
    mat<T> C(A.r, B.c);
    for (std::size_t i = 0; i < A.r; ++i) for (std::size_t k = 0; k < A.c; ++k) { T v = A(i, k); for (std::size_t j = 0; j < B.c; ++j) C(i, j) += v * B(k, j); }
    return C;
}
// LU with partial pivoting, in place: A = P^T L U. Returns the permutation; throws on a singular matrix.
template <class T> std::vector<std::size_t> lu(mat<T>& A) {
    std::size_t n = A.r; std::vector<std::size_t> p(n); for (std::size_t i = 0; i < n; ++i) p[i] = i;
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t m = k; for (std::size_t i = k + 1; i < n; ++i) if (std::fabs(A(i, k)) > std::fabs(A(m, k))) m = i;
        if (A(m, k) == T(0)) throw std::runtime_error("lu: singular");
        if (m != k) { for (std::size_t j = 0; j < n; ++j) std::swap(A(k, j), A(m, j)); std::swap(p[k], p[m]); }
        for (std::size_t i = k + 1; i < n; ++i) { A(i, k) /= A(k, k); T f = A(i, k); for (std::size_t j = k + 1; j < n; ++j) A(i, j) -= f * A(k, j); }
    }
    return p;
}
template <class T> std::vector<T> lu_solve(const mat<T>& LU, const std::vector<std::size_t>& p, const std::vector<T>& b) {
    std::size_t n = LU.r; std::vector<T> x(n);
    for (std::size_t i = 0; i < n; ++i) { T s = b[p[i]]; for (std::size_t j = 0; j < i; ++j) s -= LU(i, j) * x[j]; x[i] = s; }
    for (std::size_t i = n; i-- > 0;) { T s = x[i]; for (std::size_t j = i + 1; j < n; ++j) s -= LU(i, j) * x[j]; x[i] = s / LU(i, i); }
    return x;
}
// Cholesky A = L L^T (lower). Throws if A is not positive definite.
template <class T> mat<T> cholesky(const mat<T>& A) {
    std::size_t n = A.r; mat<T> L(n, n);
    for (std::size_t j = 0; j < n; ++j) {
        T d = A(j, j); for (std::size_t k = 0; k < j; ++k) d -= L(j, k) * L(j, k);
        if (!(d > T(0))) throw std::runtime_error("cholesky: not positive definite");
        L(j, j) = std::sqrt(d);
        for (std::size_t i = j + 1; i < n; ++i) { T s = A(i, j); for (std::size_t k = 0; k < j; ++k) s -= L(i, k) * L(j, k); L(i, j) = s / L(j, j); }
    }
    return L;
}
// Householder QR of an m x n matrix (m >= n): Q is m x m orthogonal, R is m x n upper triangular.
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
}
