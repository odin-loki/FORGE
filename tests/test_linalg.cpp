// linalg (linalg.hpp, sparse.hpp): analytic cases and reconstruction residuals, in float and double.
// Tolerances are stated per check as multiples of the type's epsilon (2.2e-16 double, 1.2e-7 float) times
// the problem's size or conditioning; they are loose by a factor of 10 to 100 over what was observed, to
// leave room for different rounding (FMA or not, another summation order) on other targets.
#include "check.hpp"
#include <forge/linalg/sparse.hpp>
#include <cmath>
#include <random>
#include <vector>
using namespace forge;
using linalg::mat;

namespace {
template <class T> T eps() { return std::numeric_limits<T>::epsilon(); }
template <class T> mat<T> random_mat(std::size_t r, std::size_t c, unsigned seed) {
    std::mt19937 g(seed); std::normal_distribution<double> d; mat<T> A(r, c); for (auto& v : A.a) v = T(d(g)); return A;
}
template <class T> std::vector<T> random_vec(std::size_t n, unsigned seed) { std::mt19937 g(seed); std::normal_distribution<double> d; std::vector<T> v(n); for (auto& e : v) e = T(d(g)); return v; }
template <class T> T diff_fro(const mat<T>& A, const mat<T>& B) { double s = 0; for (std::size_t i = 0; i < A.a.size(); ++i) { double d = double(A.a[i]) - double(B.a[i]); s += d * d; } return T(std::sqrt(s)); }
template <class T> T ortho_error(const mat<T>& Q) {   // max |Q^T Q - I|
    double e = 0;
    for (std::size_t i = 0; i < Q.c; ++i) for (std::size_t j = 0; j < Q.c; ++j) { double s = 0; for (std::size_t k = 0; k < Q.r; ++k) s += double(Q(k, i)) * double(Q(k, j)); e = std::max(e, std::fabs(s - (i == j))); }
    return T(e);
}
template <class T> mat<T> spd(std::size_t n, unsigned seed) {   // M M^T + n I
    auto M = random_mat<T>(n, n, seed); mat<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) { double s = i == j ? double(n) : 0; for (std::size_t k = 0; k < n; ++k) s += double(M(i, k)) * double(M(j, k)); A(i, j) = T(s); }
    return A;
}
template <class T> T vec_err(const std::vector<T>& a, const std::vector<T>& b) { double e = 0; for (std::size_t i = 0; i < a.size(); ++i) e = std::max(e, std::fabs(double(a[i]) - double(b[i]))); return T(e); }

template <class T> void products() {
    // blocked / kernel matmul against the triple loop: ||C - Cref||_F <= 8 k eps ||A||_F ||B||_F
    for (auto [m, k, n] : {std::tuple<std::size_t, std::size_t, std::size_t>{1, 1, 1}, {3, 5, 7}, {7, 5, 3}, {64, 64, 64}, {33, 17, 129}, {37, 300, 1100}, {260, 520, 40}, {5, 0, 4}, {0, 3, 4}}) {
        auto A = random_mat<T>(m, k, 1), B = random_mat<T>(k, n, 2);
        auto C = linalg::matmul(A, B), R = linalg::matmul_naive(A, B);
        CHECK(C.r == m && C.c == n);
        T bound = T(8) * T(k + 1) * eps<T>() * (linalg::norm_fro(A) * linalg::norm_fro(B) + T(1e-30));
        CHECK_MSG(diff_fro(C, R) <= bound, "matmul %zux%zux%zu err %g bound %g", m, k, n, double(diff_fro(C, R)), double(bound));
        if (k && m) {
            auto x = random_vec<T>(k, 3); auto y = linalg::matvec(A, x); double e = 0, scale = 0;
            for (std::size_t i = 0; i < m; ++i) { double s = 0, a = 0; for (std::size_t j = 0; j < k; ++j) { s += double(A(i, j)) * double(x[j]); a += std::fabs(double(A(i, j)) * double(x[j])); } e = std::max(e, std::fabs(s - double(y[i]))); scale = std::max(scale, a); }
            CHECK_MSG(e <= 4.0 * double(k) * double(eps<T>()) * scale, "matvec %zux%zu err %g", m, k, e);
        }
    }
    CHECK_THROWS(linalg::matmul(mat<T>(2, 3), mat<T>(2, 3)));
    auto A = random_mat<T>(5, 9, 4); auto At = linalg::transpose(A); bool ok = At.r == 9 && At.c == 5;
    for (std::size_t i = 0; i < 5; ++i) for (std::size_t j = 0; j < 9; ++j) ok &= At(j, i) == A(i, j);
    CHECK(ok);
    // norms of [[1, -2], [3, 4]]: column sums 4, 6; row sums 3, 7; Frobenius sqrt(30)
    mat<T> N(2, 2); N(0, 0) = 1; N(0, 1) = -2; N(1, 0) = 3; N(1, 1) = 4;
    CHECK(linalg::norm_1(N) == T(6)); CHECK(linalg::norm_inf(N) == T(7)); CHECK(std::fabs(linalg::norm_fro(N) - std::sqrt(T(30))) <= 4 * eps<T>() * 6);
    std::vector<T> big{std::numeric_limits<T>::max() / 2, std::numeric_limits<T>::max() / 2};   // no overflow in norm2
    CHECK(std::isfinite(linalg::norm2(big)) && linalg::norm2(big) > big[0]);
}

template <class T> void lu_family() {
    // analytic: det [[4,3,2],[1,3,1],[2,1,3]] = 19
    mat<T> A(3, 3); const T a[9] = {4, 3, 2, 1, 3, 1, 2, 1, 3}; std::copy(a, a + 9, A.a.begin());
    CHECK(std::fabs(linalg::det(A) - T(19)) <= 64 * eps<T>() * 19);
    mat<T> P(3, 3); P(0, 1) = 1; P(1, 0) = 1; P(2, 2) = 1;                 // one row swap: det -1
    CHECK(linalg::det(P) == T(-1));
    mat<T> C3(3, 3); C3(0, 1) = 1; C3(1, 2) = 1; C3(2, 0) = 1;             // a 3-cycle: det +1
    CHECK(linalg::det(C3) == T(1));
    mat<T> S(3, 3); const T s[9] = {1, 2, 3, 2, 4, 6, 1, 0, 1}; std::copy(s, s + 9, S.a.begin());   // rank 2
    CHECK(linalg::det(S) == T(0)); CHECK_THROWS(linalg::inverse(S)); CHECK_THROWS(linalg::lu_factor(mat<T>(2, 3)));
    CHECK(linalg::det(mat<T>(0, 0)) == T(1));
    for (std::size_t n : {std::size_t(1), std::size_t(2), std::size_t(12), std::size_t(60)}) {
        auto M = random_mat<T>(n, n, unsigned(10 + n)); for (std::size_t i = 0; i < n; ++i) M(i, i) += T(3);   // keep it well conditioned
        auto f = linalg::lu_factor(M);
        auto x0 = random_vec<T>(n, 5); auto b = linalg::matvec(M, x0);
        auto x = f.solve(b);
        const T kappa = linalg::cond_2(M);
        CHECK_MSG(vec_err(x, x0) <= 50 * T(n) * eps<T>() * kappa * (linalg::norm2(x0) + 1), "lu solve n=%zu err %g cond %g", n, double(vec_err(x, x0)), double(kappa));
        auto Inv = linalg::inverse(M); auto I = linalg::matmul(M, Inv);
        CHECK_MSG(diff_fro(I, linalg::identity<T>(n)) <= 50 * T(n) * eps<T>() * kappa, "inverse n=%zu resid %g", n, double(diff_fro(I, linalg::identity<T>(n))));
        auto X = linalg::solve(M, linalg::identity<T>(n)); CHECK(diff_fro(X, Inv) == T(0));
        auto xt = linalg::lu_solve_transposed(f.LU, f.perm, linalg::matvec(linalg::transpose(M), x0));
        CHECK(vec_err(xt, x0) <= 50 * T(n) * eps<T>() * kappa * (linalg::norm2(x0) + 1));
        // det(M) det(M^-1) = 1 to within conditioning (small n only: a 60 x 60 determinant overflows float)
        if (n <= 12) CHECK(std::fabs(linalg::det(M) * linalg::det(Inv) - T(1)) <= 200 * T(n) * eps<T>() * kappa);
        // rcond estimate: never above the true value by more than rounding, and within a factor 10 below... of it
        const T truth = T(1) / (linalg::norm_1(M) * linalg::norm_1(Inv)), est = linalg::rcond_1(f, linalg::norm_1(M));
        CHECK_MSG(est >= truth * T(0.99) && est <= truth * T(10), "rcond n=%zu est %g true %g", n, double(est), double(truth));
    }
    mat<T> D(4, 4); const T d[4] = {8, T(0.5), 2, T(0.125)}; for (std::size_t i = 0; i < 4; ++i) D(i, i) = d[i];
    auto fd = linalg::lu_factor(D);
    CHECK(std::fabs(linalg::rcond_1(fd, linalg::norm_1(D)) - T(0.125) / 8) <= 8 * eps<T>());   // diagonal: exact
    CHECK(std::fabs(linalg::cond_2(D) - T(64)) <= 64 * 16 * eps<T>()); CHECK(std::fabs(linalg::norm_2(D) - T(8)) <= 64 * eps<T>());
    CHECK(std::isinf(linalg::cond_2(S)) || linalg::cond_2(S) > T(1) / (64 * eps<T>()));
}

template <class T> void cholesky_and_triangular() {
    for (std::size_t n : {std::size_t(1), std::size_t(9), std::size_t(40)}) {
        auto A = spd<T>(n, unsigned(20 + n)); auto L = linalg::cholesky(A);
        auto LLt = linalg::matmul(L, linalg::transpose(L));
        CHECK_MSG(diff_fro(LLt, A) <= 20 * T(n) * eps<T>() * linalg::norm_fro(A), "cholesky n=%zu", n);
        bool lower = true; for (std::size_t i = 0; i < n; ++i) for (std::size_t j = i + 1; j < n; ++j) lower &= L(i, j) == T(0);
        CHECK(lower);
        auto x0 = random_vec<T>(n, 6); auto x = linalg::cholesky_solve(L, linalg::matvec(A, x0));
        CHECK(vec_err(x, x0) <= 100 * T(n) * eps<T>() * linalg::cond_2(A) * (linalg::norm2(x0) + 1));
        // every triangular variant against an explicit product
        auto M = random_mat<T>(n, n, unsigned(30 + n)); for (std::size_t i = 0; i < n; ++i) M(i, i) = T(2) + std::fabs(M(i, i));
        for (int lower_i = 0; lower_i < 2; ++lower_i) for (int tr = 0; tr < 2; ++tr) for (int unit = 0; unit < 2; ++unit) {
            mat<T> Tm(n, n);   // the triangle actually used, with the unit diagonal made explicit
            for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) if (lower_i ? j <= i : j >= i) Tm(i, j) = i == j && unit ? T(1) : M(i, j);
            auto op = tr ? linalg::transpose(Tm) : Tm;
            auto b = linalg::matvec(op, x0);
            auto xs = linalg::solve_triangular(M, b, {.lower = lower_i != 0, .transposed = tr != 0, .unit_diagonal = unit != 0});
            CHECK_MSG(vec_err(xs, x0) <= 100 * T(n) * eps<T>() * linalg::cond_2(op) * (linalg::norm2(x0) + 1), "trsv n=%zu lower=%d tr=%d unit=%d err %g", n, lower_i, tr, unit, double(vec_err(xs, x0)));
        }
    }
    mat<T> Z(2, 2); CHECK_THROWS(linalg::cholesky(Z)); CHECK_THROWS(linalg::solve_triangular(Z, std::vector<T>(2)));
    mat<T> Neg(2, 2); Neg(0, 0) = 1; Neg(1, 1) = -1; CHECK_THROWS(linalg::cholesky(Neg));
}

template <class T> void qr_and_least_squares() {
    for (auto [m, n] : {std::pair<std::size_t, std::size_t>{1, 1}, {9, 5}, {5, 5}, {5, 9}, {60, 23}}) {
        auto A = random_mat<T>(m, n, unsigned(40 + m + n));
        auto f = linalg::qr_factor(A); auto Q = f.thin_q(); auto R = f.r();
        CHECK_MSG(diff_fro(linalg::matmul(Q, R), A) <= 30 * T(m + n) * eps<T>() * linalg::norm_fro(A), "||A - QR|| %zux%zu = %g", m, n, double(diff_fro(linalg::matmul(Q, R), A)));
        CHECK_MSG(ortho_error(Q) <= 30 * T(m + n) * eps<T>(), "Q^T Q %zux%zu", m, n);
        bool upper = true; for (std::size_t i = 0; i < R.r; ++i) for (std::size_t j = 0; j < i && j < R.c; ++j) upper &= R(i, j) == T(0);
        CHECK(upper);
        auto b = random_vec<T>(m, 7), qtb = b; f.apply_qt(qtb); f.apply_q(qtb);   // Q Q^T b = b
        CHECK(vec_err(qtb, b) <= 30 * T(m + n) * eps<T>() * (linalg::norm2(b) + 1));
        mat<T> Qf, Rf; linalg::qr(A, Qf, Rf);   // the explicit-Q form
        CHECK(diff_fro(linalg::matmul(Qf, Rf), A) <= 30 * T(m + n) * eps<T>() * linalg::norm_fro(A)); CHECK(ortho_error(Qf) <= 30 * T(m + n) * eps<T>());
    }
    // exact polynomial data: fitting 1, t, t^2 to y = 2 - 3t + 0.5 t^2 recovers the coefficients
    const std::size_t m = 30; mat<T> V(m, 3); std::vector<T> y(m);
    for (std::size_t i = 0; i < m; ++i) { T t = T(i) / T(m - 1) * 2 - 1; V(i, 0) = 1; V(i, 1) = t; V(i, 2) = t * t; y[i] = T(2) - 3 * t + T(0.5) * t * t; }
    auto c = linalg::lstsq(V, y);
    CHECK_MSG(vec_err(c, std::vector<T>{2, -3, T(0.5)}) <= 200 * eps<T>(), "polyfit err %g", double(vec_err(c, std::vector<T>{2, -3, T(0.5)})));
    // noisy overdetermined system: the residual is orthogonal to the columns, A^T (A x - b) = 0
    auto A = random_mat<T>(40, 7, 50); auto b = random_vec<T>(40, 51); auto x = linalg::lstsq(A, b);
    auto r = linalg::matvec(A, x); for (std::size_t i = 0; i < r.size(); ++i) r[i] -= b[i];
    auto g = linalg::matvec(linalg::transpose(A), r);
    CHECK_MSG(linalg::norm2(g) <= 200 * eps<T>() * linalg::norm_fro(A) * linalg::norm_fro(A) * (linalg::norm2(x) + linalg::norm2(b)), "normal equations %g", double(linalg::norm2(g)));
    // underdetermined: A x = b exactly and x is the minimum-norm solution, i.e. x lies in the row space (x = A^T w)
    auto W = random_mat<T>(4, 9, 52); auto bw = random_vec<T>(4, 53); auto xw = linalg::lstsq(W, bw);
    CHECK(vec_err(linalg::matvec(W, xw), bw) <= 500 * eps<T>() * linalg::cond_2(W) * (linalg::norm2(bw) + 1));
    auto proj = linalg::matvec(linalg::matmul(linalg::pinv(W), W), xw);   // P_row x = x
    CHECK(vec_err(proj, xw) <= 500 * eps<T>() * linalg::cond_2(W) * (linalg::norm2(xw) + 1));
    // rank deficient: two identical columns; the minimum-norm solution splits the weight equally
    mat<T> Dp(6, 2); std::vector<T> bd(6); for (std::size_t i = 0; i < 6; ++i) { Dp(i, 0) = Dp(i, 1) = T(i + 1); bd[i] = 4 * T(i + 1); }
    auto xd = linalg::lstsq(Dp, bd);
    CHECK_MSG(std::fabs(xd[0] - 2) <= 400 * eps<T>() && std::fabs(xd[1] - 2) <= 400 * eps<T>(), "rank-deficient lstsq %g %g", double(xd[0]), double(xd[1]));
}

template <class T> void eigen() {
    // 1-D Laplacian tridiag(-1, 2, -1): eigenvalues 2 - 2 cos(k pi / (n + 1)), k = 1..n
    for (std::size_t n : {std::size_t(1), std::size_t(2), std::size_t(15), std::size_t(16), std::size_t(17), std::size_t(50)}) {
        mat<T> L(n, n); for (std::size_t i = 0; i < n; ++i) { L(i, i) = 2; if (i) L(i, i - 1) = -1; if (i + 1 < n) L(i, i + 1) = -1; }
        std::vector<T> w; mat<T> V; linalg::eigh(L, w, V);
        double e = 0; for (std::size_t k = 0; k < n; ++k) e = std::max(e, std::fabs(double(w[k]) - (2 - 2 * std::cos(double(k + 1) * 3.14159265358979323846 / double(n + 1)))));
        CHECK_MSG(e <= 30 * double(n) * double(eps<T>()), "laplacian eigenvalues n=%zu err %g", n, e);
        CHECK(ortho_error(V) <= 30 * T(n) * eps<T>());
    }
    for (std::size_t n : {std::size_t(8), std::size_t(40), std::size_t(90)}) {
        auto M = random_mat<T>(n, n, unsigned(60 + n)); mat<T> S(n, n);
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) S(i, j) = M(i, j) + M(j, i);
        std::vector<T> w, wj; mat<T> V, Vj; linalg::eigh(S, w, V); linalg::sym_eig(S, wj, Vj);
        mat<T> D(n, n); for (std::size_t i = 0; i < n; ++i) D(i, i) = w[i];
        // ||S V - V diag(w)||_F <= 30 n eps ||S||_F, V orthogonal, ascending, and the same eigenvalues as Jacobi
        CHECK_MSG(diff_fro(linalg::matmul(S, V), linalg::matmul(V, D)) <= 30 * T(n) * eps<T>() * linalg::norm_fro(S), "eigh residual n=%zu %g", n, double(diff_fro(linalg::matmul(S, V), linalg::matmul(V, D))));
        CHECK(ortho_error(V) <= 30 * T(n) * eps<T>()); CHECK(std::is_sorted(w.begin(), w.end()));
        CHECK_MSG(vec_err(w, wj) <= 30 * T(n) * eps<T>() * linalg::norm_fro(S), "eigh vs jacobi n=%zu %g", n, double(vec_err(w, wj)));
        mat<T> Dj(n, n); for (std::size_t i = 0; i < n; ++i) Dj(i, i) = wj[i];
        CHECK(diff_fro(linalg::matmul(S, Vj), linalg::matmul(Vj, Dj)) <= 30 * T(n) * eps<T>() * linalg::norm_fro(S));
    }
    // repeated eigenvalues: diag(3 x 20, 5 x 20) rotated by an orthogonal matrix
    const std::size_t n = 40; auto f = linalg::qr_factor(random_mat<T>(n, n, 70)); auto Q = f.thin_q(); mat<T> D(n, n);
    for (std::size_t i = 0; i < n; ++i) D(i, i) = i < 20 ? T(3) : T(5);
    auto S = linalg::matmul(linalg::matmul(Q, D), linalg::transpose(Q));
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < i; ++j) S(i, j) = S(j, i) = (S(i, j) + S(j, i)) / 2;
    std::vector<T> w; mat<T> V; linalg::eigh(S, w, V);
    bool ok = true; for (std::size_t i = 0; i < n; ++i) ok &= std::fabs(w[i] - (i < 20 ? T(3) : T(5))) <= 200 * T(n) * eps<T>();
    CHECK(ok); CHECK(ortho_error(V) <= 60 * T(n) * eps<T>());
    CHECK_THROWS(linalg::eigh(mat<T>(2, 3), w, V));
}

template <class T> void svd_family() {
    for (auto [m, n] : {std::pair<std::size_t, std::size_t>{1, 1}, {9, 5}, {5, 9}, {30, 30}, {3, 40}}) {
        auto A = random_mat<T>(m, n, unsigned(80 + m + n)); auto f = linalg::svd_thin(A);
        const std::size_t k = std::min(m, n); mat<T> S(k, k); for (std::size_t i = 0; i < k; ++i) S(i, i) = f.s[i];
        auto R = linalg::matmul(linalg::matmul(f.U, S), linalg::transpose(f.V));
        CHECK_MSG(diff_fro(R, A) <= 40 * T(m + n) * eps<T>() * linalg::norm_fro(A), "||A - U S V^T|| %zux%zu = %g", m, n, double(diff_fro(R, A)));
        CHECK(f.U.r == m && f.U.c == k && f.V.r == n && f.V.c == k);
        CHECK(ortho_error(f.U) <= 40 * T(m + n) * eps<T>()); CHECK(ortho_error(f.V) <= 40 * T(m + n) * eps<T>());
        CHECK(std::is_sorted(f.s.rbegin(), f.s.rend())); CHECK(f.s.back() >= T(0));
        // sum of squared singular values = squared Frobenius norm
        double ss = 0; for (T v : f.s) ss += double(v) * double(v);
        CHECK(std::fabs(std::sqrt(ss) - double(linalg::norm_fro(A))) <= 40 * double(m + n) * double(eps<T>()) * double(linalg::norm_fro(A)));
        // Penrose conditions for the pseudo-inverse
        auto P = linalg::pinv(A); const T tol = 200 * T(m + n) * eps<T>() * linalg::cond_2(A);
        CHECK(diff_fro(linalg::matmul(linalg::matmul(A, P), A), A) <= tol * linalg::norm_fro(A));
        CHECK(diff_fro(linalg::matmul(linalg::matmul(P, A), P), P) <= tol * linalg::norm_fro(P));
        auto AP = linalg::matmul(A, P), PA = linalg::matmul(P, A);
        CHECK(diff_fro(AP, linalg::transpose(AP)) <= tol * T(m)); CHECK(diff_fro(PA, linalg::transpose(PA)) <= tol * T(n));
    }
    // rank one: u v^T has the single singular value ||u|| ||v||
    auto u = random_vec<T>(12, 90), v = random_vec<T>(7, 91); mat<T> A(12, 7);
    for (std::size_t i = 0; i < 12; ++i) for (std::size_t j = 0; j < 7; ++j) A(i, j) = u[i] * v[j];
    auto f = linalg::svd_thin(A); const T s0 = linalg::norm2(u) * linalg::norm2(v);
    CHECK(std::fabs(f.s[0] - s0) <= 100 * eps<T>() * s0); CHECK(f.s[1] <= 100 * eps<T>() * s0);
    auto P = linalg::pinv(A);   // pinv(u v^T) = v u^T / (|u|^2 |v|^2)
    double e = 0; for (std::size_t i = 0; i < 7; ++i) for (std::size_t j = 0; j < 12; ++j) e = std::max(e, std::fabs(double(P(i, j)) - double(v[i]) * double(u[j]) / (double(s0) * double(s0))));
    CHECK_MSG(e <= 400 * double(eps<T>()) / double(s0), "rank-1 pinv err %g", e);
    mat<T> Z(3, 2); CHECK(diff_fro(linalg::pinv(Z), mat<T>(2, 3)) == T(0)); CHECK(linalg::norm_2(Z) == T(0));
}

template <class T> linalg::csr<T> poisson2d(std::size_t g) {   // 5-point Laplacian on a g x g grid, Dirichlet
    linalg::coo<T> C; C.rows = C.cols = g * g;
    for (std::size_t y = 0; y < g; ++y) for (std::size_t x = 0; x < g; ++x) {
        const std::size_t i = y * g + x; C.add(i, i, 4);
        if (x) C.add(i, i - 1, -1); if (x + 1 < g) C.add(i, i + 1, -1); if (y) C.add(i, i - g, -1); if (y + 1 < g) C.add(i, i + g, -1);
    }
    return linalg::to_csr(C);
}

template <class T> void sparse_formats() {
    std::mt19937 gen(100); std::uniform_real_distribution<double> u(0, 1); std::normal_distribution<double> nd;
    const std::size_t r = 24, c = 36; mat<T> D(r, c);
    for (auto& v : D.a) if (u(gen) < 0.12) v = T(nd(gen));
    for (std::size_t j = 0; j < c; ++j) D(5, j) = 0;   // an empty row
    auto A = linalg::to_csr(D); linalg::validate(A);
    CHECK(diff_fro(linalg::to_dense(A), D) == T(0));
    auto C = linalg::to_coo(A); CHECK(C.nnz() == A.val.size());
    // shuffled COO with duplicates that must be summed
    linalg::coo<T> C2; C2.rows = r; C2.cols = c;
    for (std::size_t k = C.nnz(); k-- > 0;) { C2.add(C.row[k], C.col[k], C.val[k] / 2); C2.add(C.row[k], C.col[k], C.val[k] / 2); }
    auto A2 = linalg::to_csr(C2); linalg::validate(A2);
    CHECK(A2.idx == A.idx && A2.ptr == A.ptr); CHECK(diff_fro(linalg::to_dense(A2), D) <= 2 * eps<T>() * linalg::norm_fro(D));
    auto Cc = linalg::to_csc(A); CHECK(diff_fro(linalg::to_dense(linalg::to_csr(Cc)), D) == T(0));
    CHECK(diff_fro(linalg::to_dense(linalg::transpose(A)), linalg::transpose(D)) == T(0));
    auto x = random_vec<T>(c, 101), xt = random_vec<T>(r, 102); std::vector<T> y(r), yt(c);
    auto ref = linalg::matvec(D, x), reft = linalg::matvec(linalg::transpose(D), xt);
    const T tol = 20 * T(c) * eps<T>() * (linalg::norm_fro(D) * linalg::norm2(x) + 1);
    linalg::spmv(A, x.data(), y.data()); CHECK(vec_err(y, ref) <= tol);
    linalg::spmv(Cc, x.data(), y.data()); CHECK(vec_err(y, ref) <= tol);
    linalg::spmv(C2, x.data(), y.data()); CHECK(vec_err(y, ref) <= tol);
    linalg::spmv_t(A, xt.data(), yt.data()); CHECK(vec_err(yt, reft) <= tol);
    for (std::size_t bs : {std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(12)}) {
        auto B = linalg::to_bsr(A, bs); CHECK(B.ptr.size() == r / bs + 1 && B.val.size() == B.idx.size() * bs * bs);
        linalg::spmv(B, x.data(), y.data()); CHECK_MSG(vec_err(y, ref) <= tol, "bsr spmv bs=%zu", bs);
        CHECK(diff_fro(linalg::to_dense(linalg::to_csr(B)), D) == T(0));
        auto Xd = random_mat<T>(c, 5, 103);
        CHECK(diff_fro(linalg::spmm(B, Xd), linalg::matmul_naive(D, Xd)) <= tol * 5);
    }
    CHECK_THROWS(linalg::to_bsr(A, 5)); CHECK_THROWS(linalg::to_bsr(A, 0));
    auto Xd = random_mat<T>(c, 11, 104);
    CHECK(diff_fro(linalg::spmm(A, Xd), linalg::matmul_naive(D, Xd)) <= tol * 11);
    mat<T> E(c, 17); for (auto& v : E.a) if (u(gen) < 0.2) v = T(nd(gen));
    auto P = linalg::spgemm(A, linalg::to_csr(E)); linalg::validate(P);
    CHECK(diff_fro(linalg::to_dense(P), linalg::matmul_naive(D, E)) <= tol * 17);
    bool sorted = true; for (std::size_t i = 0; i < P.rows; ++i) sorted &= std::is_sorted(P.idx.begin() + std::ptrdiff_t(P.ptr[i]), P.idx.begin() + std::ptrdiff_t(P.ptr[i + 1]));
    CHECK(sorted);
    CHECK_THROWS(linalg::spgemm(A, A)); CHECK_THROWS(linalg::spmm(A, mat<T>(3, 3)));
    auto bad = A; bad.idx[0] = c + 4; CHECK_THROWS(linalg::validate(bad));
    bad = A; bad.ptr[3] = bad.ptr[4] + 1; CHECK_THROWS(linalg::validate(bad));
    linalg::coo<T> oob; oob.rows = oob.cols = 2; oob.add(2, 0, 1); CHECK_THROWS(linalg::to_csr(oob));
    auto dg = linalg::diagonal(poisson2d<T>(4)); bool four = dg.size() == 16; for (T v : dg) four &= v == T(4); CHECK(four);
}

template <class T> void iterative_solvers(T tol) {
    // SPD: 2-D Poisson, n = 400. All three preconditioners converge; ILU(0) needs the fewest iterations.
    const std::size_t g = 20, n = g * g; auto A = poisson2d<T>(g);
    auto x0 = random_vec<T>(n, 110); std::vector<T> b(n); linalg::spmv(A, x0.data(), b.data());
    std::vector<T> x; auto plain = linalg::pcg(A, b, x, tol, 2000);
    const T sol_tol = tol * 400;   // error <= cond * residual; cond(Poisson 20 x 20) ~ 180
    CHECK_MSG(plain.converged && vec_err(x, x0) <= sol_tol * linalg::norm2(x0), "pcg plain it=%d res=%g", plain.iterations, plain.residual);
    x.clear(); auto jac = linalg::pcg(A, b, x, linalg::jacobi_prec<T>(A), tol, 2000);
    CHECK(jac.converged && vec_err(x, x0) <= sol_tol * linalg::norm2(x0));
    x.clear(); auto ilu = linalg::pcg(A, b, x, linalg::ilu0<T>(A), tol, 2000);
    CHECK_MSG(ilu.converged && vec_err(x, x0) <= sol_tol * linalg::norm2(x0), "pcg ilu0 it=%d res=%g", ilu.iterations, ilu.residual);
    CHECK_MSG(ilu.iterations < plain.iterations, "ilu0 %d vs plain %d iterations", ilu.iterations, plain.iterations);
    std::printf("  pcg(%s) poisson 400: plain %d, jacobi %d, ilu0 %d iterations\n", sizeof(T) == 4 ? "f32" : "f64", plain.iterations, jac.iterations, ilu.iterations);
    // analytic: -x[i-1] + 2 x[i] - x[i+1] = 1 has the solution x[i] = (i + 1)(m - i) / 2 (0-based), matrix-free
    const std::size_t m = 60; std::vector<T> ones(m, T(1)), xl;
    auto lap = [m](const T* in, T* out) { for (std::size_t i = 0; i < m; ++i) out[i] = 2 * in[i] - (i ? in[i - 1] : T(0)) - (i + 1 < m ? in[i + 1] : T(0)); };
    auto li = linalg::pcg(lap, ones, xl, linalg::identity_prec<T>{m}, tol, 500);
    double e = 0; for (std::size_t i = 0; i < m; ++i) e = std::max(e, std::fabs(double(xl[i]) - double(i + 1) * double(m - i) / 2));
    CHECK_MSG(li.converged && e <= double(tol) * 1e5, "matrix-free laplacian err %g it %d", e, li.iterations);   // cond ~ 1500, |x| ~ 450
    // non-symmetric: 1-D convection-diffusion tridiag(-1.5, 2.1, -0.5), n = 300; reference by dense LU
    const std::size_t q = 300; linalg::coo<T> C; C.rows = C.cols = q;
    for (std::size_t i = 0; i < q; ++i) { C.add(i, i, T(2.1)); if (i) C.add(i, i - 1, T(-1.5)); if (i + 1 < q) C.add(i, i + 1, T(-0.5)); }
    auto N = linalg::to_csr(C); auto xn0 = random_vec<T>(q, 111); std::vector<T> bn(q); linalg::spmv(N, xn0.data(), bn.data());
    auto dense = linalg::solve(linalg::to_dense(N), bn);
    const T ntol = tol * 2000 * (linalg::norm2(xn0) + 1);
    for (int prec = 0; prec < 3; ++prec) {
        std::vector<T> xb, xg; linalg::solve_info ib, ig;
        if (prec == 0) { ib = linalg::bicgstab(N, bn, xb, tol, 3000); ig = linalg::pgmres(N, bn, xg, 40, tol, 3000); }
        else if (prec == 1) { linalg::jacobi_prec<T> M(N); ib = linalg::bicgstab(N, bn, xb, M, tol, 3000); ig = linalg::pgmres(N, bn, xg, M, 40, tol, 3000); }
        else { linalg::ilu0<T> M(N); ib = linalg::bicgstab(N, bn, xb, M, tol, 3000); ig = linalg::pgmres(N, bn, xg, M, 40, tol, 3000);
               CHECK_MSG(ig.iterations <= 3 && ib.iterations <= 3, "ILU(0) of a tridiagonal matrix is its exact LU: gmres %d, bicgstab %d iterations", ig.iterations, ib.iterations); }
        CHECK_MSG(ib.converged && vec_err(xb, dense) <= ntol, "bicgstab prec=%d it=%d res=%g err=%g", prec, ib.iterations, ib.residual, double(vec_err(xb, dense)));
        CHECK_MSG(ig.converged && vec_err(xg, dense) <= ntol, "pgmres prec=%d it=%d res=%g err=%g", prec, ig.iterations, ig.residual, double(vec_err(xg, dense)));
    }
    // the older entry points still work
    std::vector<T> xo; CHECK(linalg::gmres(N, bn, xo, 40, tol, 3000) >= 0); CHECK(vec_err(xo, dense) <= ntol);
    // zero right-hand side, a non-positive-definite matrix for pcg, and a tiny iteration budget
    std::vector<T> zero(n, T(0)), xz; auto iz = linalg::pcg(A, zero, xz, tol); CHECK(iz.converged && iz.iterations == 0 && linalg::norm2(xz) == T(0));
    auto Neg = A; for (auto& v : Neg.val) v = -v;
    std::vector<T> xneg; auto in = linalg::pcg(Neg, b, xneg, tol, 50); CHECK(!in.converged);
    std::vector<T> xs; auto is = linalg::pgmres(N, bn, xs, 5, tol, 3); CHECK(!is.converged && is.iterations == 3);
    linalg::coo<T> hole; hole.rows = hole.cols = 2; hole.add(0, 1, 1); hole.add(1, 0, 1);
    CHECK_THROWS(linalg::ilu0<T>(linalg::to_csr(hole))); CHECK_THROWS(linalg::jacobi_prec<T>(linalg::to_csr(hole)));
    // a warm start at the solution takes no iterations
    std::vector<T> warm = x0; auto iw = linalg::pcg(A, b, warm, tol * 10); CHECK(iw.converged && iw.iterations == 0);
}
} // namespace

TEST(linalg_products_f64) { products<double>(); }
TEST(linalg_products_f32) { products<float>(); }
TEST(linalg_lu_det_inverse_rcond_f64) { lu_family<double>(); }
TEST(linalg_lu_det_inverse_rcond_f32) { lu_family<float>(); }
TEST(linalg_cholesky_triangular_f64) { cholesky_and_triangular<double>(); }
TEST(linalg_cholesky_triangular_f32) { cholesky_and_triangular<float>(); }
TEST(linalg_qr_lstsq_f64) { qr_and_least_squares<double>(); }
TEST(linalg_qr_lstsq_f32) { qr_and_least_squares<float>(); }
TEST(linalg_eigh_f64) { eigen<double>(); }
TEST(linalg_eigh_f32) { eigen<float>(); }
TEST(linalg_svd_pinv_f64) { svd_family<double>(); }
TEST(linalg_svd_pinv_f32) { svd_family<float>(); }
TEST(linalg_sparse_formats_f64) { sparse_formats<double>(); }
TEST(linalg_sparse_formats_f32) { sparse_formats<float>(); }
TEST(linalg_iterative_solvers_f64) { iterative_solvers<double>(1e-10); }
TEST(linalg_iterative_solvers_f32) { iterative_solvers<float>(2e-5f); }
