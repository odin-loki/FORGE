#pragma once
// Sparse matrices and iterative solvers.
//   formats   csr (linalg.hpp), csc, coo, bsr (square dense blocks); conversions between them and to dense
//   products  spmv for every format, spmv_t (A^T x), spmm (sparse x dense), spgemm (sparse x sparse)
//   solvers   pcg (SPD), bicgstab and pgmres (general), each with a preconditioner: identity_prec,
//             jacobi_prec (inverse diagonal) or ilu0 (incomplete LU on the matrix's own pattern)
// A solver takes the matrix as a csr or as any callable op(const T* x, T* y) computing y = A x, and a
// preconditioner callable prec(const T* r, T* z) computing z ~ A^-1 r. x is the starting guess when its size
// matches (zero otherwise). All loops are serial.
#include <forge/linalg/linalg.hpp>
#include <concepts>
#include <numeric>

namespace forge::linalg {

template <class T> struct coo {
    std::size_t rows = 0, cols = 0; std::vector<std::size_t> row, col; std::vector<T> val;
    void add(std::size_t i, std::size_t j, T v) { row.push_back(i); col.push_back(j); val.push_back(v); }
    std::size_t nnz() const { return val.size(); }
};
// Column-compressed: ptr[j]..ptr[j+1] index the row indices and values of column j.
template <class T> struct csc { std::size_t rows = 0, cols = 0; std::vector<std::size_t> ptr{0}, idx; std::vector<T> val; };
// Block-compressed rows with square bs x bs blocks: block row i holds blocks ptr[i]..ptr[i+1], block k sits in
// block column idx[k] and its values are val[k * bs * bs ...] row-major.
template <class T> struct bsr { std::size_t rows = 0, cols = 0, bs = 1; std::vector<std::size_t> ptr{0}, idx; std::vector<T> val; };

// Throws std::invalid_argument unless the structure is consistent (monotone ptr, indices in range).
template <class T> void validate(const csr<T>& A) {
    if (A.ptr.size() != A.rows + 1 || A.ptr.front() != 0 || A.ptr.back() != A.idx.size() || A.idx.size() != A.val.size()) throw std::invalid_argument("csr: inconsistent sizes");
    for (std::size_t i = 0; i < A.rows; ++i) if (A.ptr[i] > A.ptr[i + 1]) throw std::invalid_argument("csr: row pointers decrease");
    for (std::size_t c : A.idx) if (c >= A.cols) throw std::invalid_argument("csr: column index out of range");
}
// Sort the column indices of every row.
template <class T> void sort_indices(csr<T>& A) {
    std::vector<std::size_t> o; std::vector<std::size_t> ti; std::vector<T> tv;
    for (std::size_t i = 0; i < A.rows; ++i) {
        const std::size_t b = A.ptr[i], n = A.ptr[i + 1] - b;
        if (std::is_sorted(A.idx.begin() + std::ptrdiff_t(b), A.idx.begin() + std::ptrdiff_t(b + n))) continue;
        o.resize(n); std::iota(o.begin(), o.end(), std::size_t(0));
        std::sort(o.begin(), o.end(), [&](std::size_t x, std::size_t y) { return A.idx[b + x] < A.idx[b + y]; });
        ti.resize(n); tv.resize(n);
        for (std::size_t k = 0; k < n; ++k) { ti[k] = A.idx[b + o[k]]; tv[k] = A.val[b + o[k]]; }
        std::copy(ti.begin(), ti.end(), A.idx.begin() + std::ptrdiff_t(b)); std::copy(tv.begin(), tv.end(), A.val.begin() + std::ptrdiff_t(b));
    }
}
// COO -> CSR: entries sorted by (row, column), duplicates summed. Explicit zeros are kept.
template <class T> csr<T> to_csr(const coo<T>& C) {
    if (C.row.size() != C.val.size() || C.col.size() != C.val.size()) throw std::invalid_argument("coo: inconsistent sizes");
    for (std::size_t k = 0; k < C.val.size(); ++k) if (C.row[k] >= C.rows || C.col[k] >= C.cols) throw std::invalid_argument("coo: index out of range");
    std::vector<std::size_t> o(C.val.size()); std::iota(o.begin(), o.end(), std::size_t(0));
    std::stable_sort(o.begin(), o.end(), [&](std::size_t x, std::size_t y) { return C.row[x] != C.row[y] ? C.row[x] < C.row[y] : C.col[x] < C.col[y]; });
    csr<T> A; A.rows = C.rows; A.cols = C.cols; A.ptr.assign(C.rows + 1, 0);
    for (std::size_t q = 0; q < o.size(); ++q) {
        const std::size_t k = o[q];
        if (q && C.row[k] == C.row[o[q - 1]] && C.col[k] == C.col[o[q - 1]]) { A.val.back() += C.val[k]; continue; }
        A.idx.push_back(C.col[k]); A.val.push_back(C.val[k]); ++A.ptr[C.row[k] + 1];
    }
    for (std::size_t i = 0; i < C.rows; ++i) A.ptr[i + 1] += A.ptr[i];
    return A;
}
template <class T> coo<T> to_coo(const csr<T>& A) {
    coo<T> C; C.rows = A.rows; C.cols = A.cols;
    for (std::size_t i = 0; i < A.rows; ++i) for (std::size_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) C.add(i, A.idx[k], A.val[k]);
    return C;
}
// A^T in CSR (column indices come out sorted).
template <class T> csr<T> transpose(const csr<T>& A) {
    csr<T> B; B.rows = A.cols; B.cols = A.rows; B.ptr.assign(A.cols + 1, 0); B.idx.resize(A.idx.size()); B.val.resize(A.val.size());
    for (std::size_t c : A.idx) ++B.ptr[c + 1];
    for (std::size_t j = 0; j < A.cols; ++j) B.ptr[j + 1] += B.ptr[j];
    std::vector<std::size_t> at(B.ptr.begin(), B.ptr.end() - 1);
    for (std::size_t i = 0; i < A.rows; ++i) for (std::size_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) { const std::size_t d = at[A.idx[k]]++; B.idx[d] = i; B.val[d] = A.val[k]; }
    return B;
}
template <class T> csc<T> to_csc(const csr<T>& A) { auto t = transpose(A); csc<T> C; C.rows = A.rows; C.cols = A.cols; C.ptr = std::move(t.ptr); C.idx = std::move(t.idx); C.val = std::move(t.val); return C; }
template <class T> csr<T> to_csr(const csc<T>& C) { csr<T> t; t.rows = C.cols; t.cols = C.rows; t.ptr = C.ptr; t.idx = C.idx; t.val = C.val; return transpose(t); }
template <class T> mat<T> to_dense(const csr<T>& A) {
    mat<T> D(A.rows, A.cols);
    for (std::size_t i = 0; i < A.rows; ++i) for (std::size_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) D(i, A.idx[k]) += A.val[k];
    return D;
}
// CSR -> BSR with bs x bs blocks; rows and cols must be multiples of bs. A block is stored if any of its
// entries is stored in A.
template <class T> bsr<T> to_bsr(const csr<T>& A, std::size_t bs) {
    if (bs == 0 || A.rows % bs || A.cols % bs) throw std::invalid_argument("to_bsr: dimensions are not multiples of the block size");
    bsr<T> B; B.rows = A.rows; B.cols = A.cols; B.bs = bs;
    const std::size_t brows = A.rows / bs, bcols = A.cols / bs, npos = std::size_t(-1);
    std::vector<std::size_t> where(bcols, npos), used;
    for (std::size_t bi = 0; bi < brows; ++bi) {
        used.clear();
        for (std::size_t i = bi * bs; i < (bi + 1) * bs; ++i) for (std::size_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) { const std::size_t bj = A.idx[k] / bs; if (where[bj] == npos) { where[bj] = 0; used.push_back(bj); } }
        std::sort(used.begin(), used.end());
        for (std::size_t bj : used) { where[bj] = B.idx.size(); B.idx.push_back(bj); }
        B.val.resize(B.idx.size() * bs * bs, T(0));
        for (std::size_t i = bi * bs; i < (bi + 1) * bs; ++i) for (std::size_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k)
            B.val[where[A.idx[k] / bs] * bs * bs + (i - bi * bs) * bs + A.idx[k] % bs] += A.val[k];
        for (std::size_t bj : used) where[bj] = npos;
        B.ptr.push_back(B.idx.size());
    }
    return B;
}
// BSR -> CSR; zeros inside stored blocks are dropped.
template <class T> csr<T> to_csr(const bsr<T>& B) {
    csr<T> A; A.rows = B.rows; A.cols = B.cols; const std::size_t bs = B.bs;
    for (std::size_t bi = 0; bi + 1 < B.ptr.size(); ++bi) for (std::size_t r = 0; r < bs; ++r) {
        for (std::size_t k = B.ptr[bi]; k < B.ptr[bi + 1]; ++k) for (std::size_t c = 0; c < bs; ++c) { const T v = B.val[k * bs * bs + r * bs + c]; if (v != T(0)) { A.idx.push_back(B.idx[k] * bs + c); A.val.push_back(v); } }
        A.ptr.push_back(A.idx.size());
    }
    return A;
}
template <class T> std::vector<T> diagonal(const csr<T>& A) {
    std::vector<T> d(std::min(A.rows, A.cols), T(0));
    for (std::size_t i = 0; i < d.size(); ++i) for (std::size_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) if (A.idx[k] == i) d[i] += A.val[k];
    return d;
}

// ---- products ----
template <class T> void spmv(const csc<T>& A, const T* x, T* y) {
    std::fill(y, y + A.rows, T(0));
    for (std::size_t j = 0; j < A.cols; ++j) { const T v = x[j]; for (std::size_t k = A.ptr[j]; k < A.ptr[j + 1]; ++k) y[A.idx[k]] += A.val[k] * v; }
}
template <class T> void spmv(const coo<T>& A, const T* x, T* y) {
    std::fill(y, y + A.rows, T(0));
    for (std::size_t k = 0; k < A.val.size(); ++k) y[A.row[k]] += A.val[k] * x[A.col[k]];
}
template <class T> void spmv(const bsr<T>& A, const T* x, T* y) {
    const std::size_t bs = A.bs;
    std::fill(y, y + A.rows, T(0));
    for (std::size_t bi = 0; bi + 1 < A.ptr.size(); ++bi) for (std::size_t k = A.ptr[bi]; k < A.ptr[bi + 1]; ++k) {
        const T* blk = A.val.data() + k * bs * bs; const T* xs = x + A.idx[k] * bs; T* ys = y + bi * bs;
        for (std::size_t r = 0; r < bs; ++r) { T s = 0; for (std::size_t c = 0; c < bs; ++c) s += blk[r * bs + c] * xs[c]; ys[r] += s; }
    }
}
// y = A^T x without forming the transpose.
template <class T> void spmv_t(const csr<T>& A, const T* x, T* y) {
    std::fill(y, y + A.cols, T(0));
    for (std::size_t i = 0; i < A.rows; ++i) { const T v = x[i]; for (std::size_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) y[A.idx[k]] += A.val[k] * v; }
}
// Sparse x dense: C = A * B.
template <class T> mat<T> spmm(const csr<T>& A, const mat<T>& B) {
    if (A.cols != B.r) throw std::invalid_argument("spmm: inner dimensions differ");
    mat<T> C(A.rows, B.c);
    for (std::size_t i = 0; i < A.rows; ++i) { T* c = C.row(i); for (std::size_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) { const T v = A.val[k]; const T* b = B.row(A.idx[k]); for (std::size_t j = 0; j < B.c; ++j) c[j] += v * b[j]; } }
    return C;
}
template <class T> mat<T> spmm(const bsr<T>& A, const mat<T>& B) {
    if (A.cols != B.r) throw std::invalid_argument("spmm: inner dimensions differ");
    mat<T> C(A.rows, B.c); const std::size_t bs = A.bs;
    for (std::size_t bi = 0; bi + 1 < A.ptr.size(); ++bi) for (std::size_t k = A.ptr[bi]; k < A.ptr[bi + 1]; ++k)
        for (std::size_t r = 0; r < bs; ++r) { T* c = C.row(bi * bs + r); for (std::size_t q = 0; q < bs; ++q) { const T v = A.val[k * bs * bs + r * bs + q]; const T* b = B.row(A.idx[k] * bs + q); for (std::size_t j = 0; j < B.c; ++j) c[j] += v * b[j]; } }
    return C;
}
// Sparse x sparse (Gustavson): C = A * B with sorted column indices; entries that cancel to zero are kept.
template <class T> csr<T> spgemm(const csr<T>& A, const csr<T>& B) {
    if (A.cols != B.rows) throw std::invalid_argument("spgemm: inner dimensions differ");
    csr<T> C; C.rows = A.rows; C.cols = B.cols;
    const std::size_t npos = std::size_t(-1);
    std::vector<std::size_t> where(B.cols, npos), cols; std::vector<T> acc(B.cols, T(0));
    for (std::size_t i = 0; i < A.rows; ++i) {
        cols.clear();
        for (std::size_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) {
            const T v = A.val[k]; const std::size_t r = A.idx[k];
            for (std::size_t q = B.ptr[r]; q < B.ptr[r + 1]; ++q) { const std::size_t c = B.idx[q]; if (where[c] == npos) { where[c] = i; cols.push_back(c); } acc[c] += v * B.val[q]; }
        }
        std::sort(cols.begin(), cols.end());
        for (std::size_t c : cols) { C.idx.push_back(c); C.val.push_back(acc[c]); acc[c] = T(0); where[c] = npos; }
        C.ptr.push_back(C.idx.size());
    }
    return C;
}

// ---- preconditioners ----
template <class T> struct identity_prec { std::size_t n = 0; void operator()(const T* r, T* z) const { std::copy(r, r + n, z); } };
// z = D^-1 r. Throws if a diagonal entry is zero.
template <class T> struct jacobi_prec {
    std::vector<T> inv;
    explicit jacobi_prec(const csr<T>& A) : inv(diagonal(A)) { for (T& d : inv) { if (d == T(0)) throw std::runtime_error("jacobi_prec: zero on the diagonal"); d = T(1) / d; } }
    void operator()(const T* r, T* z) const { for (std::size_t i = 0; i < inv.size(); ++i) z[i] = r[i] * inv[i]; }
};
// Incomplete LU with no fill-in: L and U have the pattern of A's lower and upper triangles. Needs a square
// matrix with every diagonal entry stored; throws if a pivot becomes zero. For a symmetric positive definite
// A the product L U is symmetric, so it can precondition pcg (it may still fail to be positive definite for
// matrices that are far from diagonally dominant).
template <class T> struct ilu0 {
    csr<T> F; std::vector<std::size_t> diag;
    explicit ilu0(csr<T> A) : F(std::move(A)) {
        if (F.rows != F.cols) throw std::invalid_argument("ilu0: matrix is not square");
        sort_indices(F);
        const std::size_t n = F.rows, npos = std::size_t(-1);
        diag.assign(n, npos);
        for (std::size_t i = 0; i < n; ++i) for (std::size_t k = F.ptr[i]; k < F.ptr[i + 1]; ++k) if (F.idx[k] == i) diag[i] = k;
        for (std::size_t i = 0; i < n; ++i) if (diag[i] == npos) throw std::runtime_error("ilu0: missing diagonal entry");
        std::vector<std::size_t> pos(n, npos);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t k = F.ptr[i]; k < F.ptr[i + 1]; ++k) pos[F.idx[k]] = k;
            for (std::size_t k = F.ptr[i]; k < diag[i]; ++k) {
                const std::size_t c = F.idx[k];
                F.val[k] /= F.val[diag[c]];
                const T f = F.val[k];
                for (std::size_t q = diag[c] + 1; q < F.ptr[c + 1]; ++q) { const std::size_t p = pos[F.idx[q]]; if (p != npos) F.val[p] -= f * F.val[q]; }
            }
            if (F.val[diag[i]] == T(0)) throw std::runtime_error("ilu0: zero pivot");
            for (std::size_t k = F.ptr[i]; k < F.ptr[i + 1]; ++k) pos[F.idx[k]] = npos;
        }
    }
    void operator()(const T* r, T* z) const {
        const std::size_t n = F.rows;
        for (std::size_t i = 0; i < n; ++i) { T s = r[i]; for (std::size_t k = F.ptr[i]; k < diag[i]; ++k) s -= F.val[k] * z[F.idx[k]]; z[i] = s; }
        for (std::size_t i = n; i-- > 0;) { T s = z[i]; for (std::size_t k = diag[i] + 1; k < F.ptr[i + 1]; ++k) s -= F.val[k] * z[F.idx[k]]; z[i] = s / F.val[diag[i]]; }
    }
};

// ---- iterative solvers ----
// iterations = matrix applications' worth of steps taken; residual = ||b - A x|| / ||b|| recomputed from x at
// the end (||b - A x|| when b = 0); converged = residual <= tol at that point.
struct solve_info { int iterations = 0; double residual = 0; bool converged = false; };
template <class F, class T> concept linear_op = std::invocable<F&, const T*, T*>;   // f(in, out)
namespace detail {
template <class T> T vdot(const std::vector<T>& a, const std::vector<T>& b) { T s = 0; for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s; }
template <class T, class Op> solve_info finish(Op& A, const std::vector<T>& b, const std::vector<T>& x, T bnorm, T tol, int it) {
    std::vector<T> r(b.size()); A(x.data(), r.data());
    for (std::size_t i = 0; i < b.size(); ++i) r[i] = b[i] - r[i];
    solve_info s; s.iterations = it; s.residual = double(norm2(r) / bnorm); s.converged = s.residual <= double(tol) * 1.0000001 || s.residual == 0;
    return s;
}
template <class T> auto op_of(const csr<T>& A) { return [&A](const T* x, T* y) { spmv(A, x, y); }; }
} // namespace detail

// Preconditioned conjugate gradients for symmetric positive definite A (and symmetric positive definite M).
// Stops when the recurrence residual reaches tol * ||b||.
template <class T, linear_op<T> Op, linear_op<T> Prec> solve_info pcg(Op&& A, const std::vector<T>& b, std::vector<T>& x, Prec&& M, std::type_identity_t<T> tol = T(1e-8), int maxit = 1000) {
    const std::size_t n = b.size(); if (x.size() != n) x.assign(n, T(0));
    T bn = norm2(b); if (bn == T(0)) bn = T(1);
    std::vector<T> r(n), z(n), p(n), q(n);
    A(x.data(), r.data()); for (std::size_t i = 0; i < n; ++i) r[i] = b[i] - r[i];
    M(r.data(), z.data()); p = z;
    T rz = detail::vdot(r, z); int it = 0;
    for (; it < maxit && norm2(r) > tol * bn; ++it) {
        A(p.data(), q.data());
        const T pq = detail::vdot(p, q);
        if (!(pq > T(0))) break;   // not positive definite (or breakdown)
        const T al = rz / pq;
        for (std::size_t i = 0; i < n; ++i) { x[i] += al * p[i]; r[i] -= al * q[i]; }
        M(r.data(), z.data());
        const T rz2 = detail::vdot(r, z), be = rz2 / rz; rz = rz2;
        for (std::size_t i = 0; i < n; ++i) p[i] = z[i] + be * p[i];
    }
    return detail::finish(A, b, x, bn, tol, it);
}
// BiCGSTAB (van der Vorst) with right preconditioning, for general square systems. One iteration applies A
// and M twice. Stops early on breakdown (rho or omega zero); check solve_info::converged.
template <class T, linear_op<T> Op, linear_op<T> Prec> solve_info bicgstab(Op&& A, const std::vector<T>& b, std::vector<T>& x, Prec&& M, std::type_identity_t<T> tol = T(1e-8), int maxit = 1000) {
    const std::size_t n = b.size(); if (x.size() != n) x.assign(n, T(0));
    T bn = norm2(b); if (bn == T(0)) bn = T(1);
    std::vector<T> r(n), r0(n), p(n, T(0)), v(n, T(0)), s(n), t(n), y(n), z(n);
    A(x.data(), r.data()); for (std::size_t i = 0; i < n; ++i) r[i] = b[i] - r[i];
    r0 = r; T rho = 1, alpha = 1, omega = 1; int it = 0;
    for (; it < maxit && norm2(r) > tol * bn; ++it) {
        const T rho2 = detail::vdot(r0, r);
        if (rho2 == T(0) || omega == T(0)) break;
        const T beta = (rho2 / rho) * (alpha / omega); rho = rho2;
        for (std::size_t i = 0; i < n; ++i) p[i] = r[i] + beta * (p[i] - omega * v[i]);
        M(p.data(), y.data()); A(y.data(), v.data());
        const T r0v = detail::vdot(r0, v);
        if (r0v == T(0)) break;
        alpha = rho / r0v;
        for (std::size_t i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];
        if (norm2(s) <= tol * bn) { for (std::size_t i = 0; i < n; ++i) x[i] += alpha * y[i]; r = s; ++it; break; }
        M(s.data(), z.data()); A(z.data(), t.data());
        const T tt = detail::vdot(t, t);
        omega = tt > T(0) ? detail::vdot(t, s) / tt : T(0);
        for (std::size_t i = 0; i < n; ++i) { x[i] += alpha * y[i] + omega * z[i]; r[i] = s[i] - omega * t[i]; }
    }
    return detail::finish(A, b, x, bn, tol, it);
}
// Restarted GMRES(restart) with right preconditioning (the residual it minimises is the true one), modified
// Gram-Schmidt and Givens rotations.
template <class T, linear_op<T> Op, linear_op<T> Prec> solve_info pgmres(Op&& A, const std::vector<T>& b, std::vector<T>& x, Prec&& M, int restart = 30, std::type_identity_t<T> tol = T(1e-8), int maxit = 1000) {
    const std::size_t n = b.size(); if (x.size() != n) x.assign(n, T(0));
    T bn = norm2(b); if (bn == T(0)) bn = T(1);
    const std::size_t k = std::size_t(std::max(1, restart));
    std::vector<std::vector<T>> V(k + 1, std::vector<T>(n)); mat<T> H(k + 1, k);
    std::vector<T> cs(k), sn(k), g(k + 1), w(n), z(n), r(n);
    int it = 0;
    while (it < maxit) {
        A(x.data(), r.data()); for (std::size_t i = 0; i < n; ++i) r[i] = b[i] - r[i];
        const T beta = norm2(r);
        if (beta <= tol * bn) break;
        std::fill(g.begin(), g.end(), T(0)); g[0] = beta;
        for (std::size_t i = 0; i < n; ++i) V[0][i] = r[i] / beta;
        std::size_t j = 0;
        while (j < k && it < maxit) {
            M(V[j].data(), z.data()); A(z.data(), w.data()); ++it;
            for (std::size_t i = 0; i <= j; ++i) { const T h = detail::vdot(w, V[i]); H(i, j) = h; for (std::size_t q = 0; q < n; ++q) w[q] -= h * V[i][q]; }
            const T hn = norm2(w); H(j + 1, j) = hn;
            if (hn > T(0)) for (std::size_t q = 0; q < n; ++q) V[j + 1][q] = w[q] / hn;
            for (std::size_t i = 0; i < j; ++i) { const T a = cs[i] * H(i, j) + sn[i] * H(i + 1, j); H(i + 1, j) = -sn[i] * H(i, j) + cs[i] * H(i + 1, j); H(i, j) = a; }
            const T d = std::hypot(H(j, j), H(j + 1, j));
            if (d == T(0)) { ++j; break; }
            cs[j] = H(j, j) / d; sn[j] = H(j + 1, j) / d; H(j, j) = d; H(j + 1, j) = 0;
            g[j + 1] = -sn[j] * g[j]; g[j] = cs[j] * g[j];
            ++j;
            if (std::fabs(g[j]) <= tol * bn || hn == T(0)) break;
        }
        std::vector<T> yv(j, T(0));
        for (std::size_t i = j; i-- > 0;) { T s = g[i]; for (std::size_t q = i + 1; q < j; ++q) s -= H(i, q) * yv[q]; yv[i] = H(i, i) != T(0) ? s / H(i, i) : T(0); }
        std::fill(w.begin(), w.end(), T(0));
        for (std::size_t i = 0; i < j; ++i) for (std::size_t q = 0; q < n; ++q) w[q] += yv[i] * V[i][q];
        M(w.data(), z.data());
        for (std::size_t q = 0; q < n; ++q) x[q] += z[q];
    }
    return detail::finish(A, b, x, bn, tol, it);
}
// CSR conveniences (the default preconditioner is the identity).
template <class T, linear_op<T> Prec> solve_info pcg(const csr<T>& A, const std::vector<T>& b, std::vector<T>& x, Prec&& M, std::type_identity_t<T> tol = T(1e-8), int maxit = 1000) { return pcg(detail::op_of(A), b, x, M, tol, maxit); }
template <class T> solve_info pcg(const csr<T>& A, const std::vector<T>& b, std::vector<T>& x, std::type_identity_t<T> tol = T(1e-8), int maxit = 1000) { return pcg(detail::op_of(A), b, x, identity_prec<T>{b.size()}, tol, maxit); }
template <class T, linear_op<T> Prec> solve_info bicgstab(const csr<T>& A, const std::vector<T>& b, std::vector<T>& x, Prec&& M, std::type_identity_t<T> tol = T(1e-8), int maxit = 1000) { return bicgstab(detail::op_of(A), b, x, M, tol, maxit); }
template <class T> solve_info bicgstab(const csr<T>& A, const std::vector<T>& b, std::vector<T>& x, std::type_identity_t<T> tol = T(1e-8), int maxit = 1000) { return bicgstab(detail::op_of(A), b, x, identity_prec<T>{b.size()}, tol, maxit); }
template <class T, linear_op<T> Prec> solve_info pgmres(const csr<T>& A, const std::vector<T>& b, std::vector<T>& x, Prec&& M, int restart = 30, std::type_identity_t<T> tol = T(1e-8), int maxit = 1000) { return pgmres(detail::op_of(A), b, x, M, restart, tol, maxit); }
template <class T> solve_info pgmres(const csr<T>& A, const std::vector<T>& b, std::vector<T>& x, int restart = 30, std::type_identity_t<T> tol = T(1e-8), int maxit = 1000) { return pgmres(detail::op_of(A), b, x, identity_prec<T>{b.size()}, restart, tol, maxit); }

} // namespace forge::linalg
