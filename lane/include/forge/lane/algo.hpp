#pragma once
// Tier A algorithms written once against vec/strip; instantiated per target in lane/src.
#include <forge/lane/math.hpp>
#include <span>

namespace forge::lane::algo {

template <Target Tg> void saxpy(float a, const float* x, float* y, std::size_t n) {
    auto va = splat<Tg>(a);
    strip<Tg, float>(n, [&](std::size_t i, auto m) {
        store(y + i, fma(va, load(x + i, m), load(y + i, m)), m);
    });
}

template <Target Tg> float sum(const float* x, std::size_t n) {
    auto acc = splat<Tg>(0.f);
    strip<Tg, float>(n, [&](std::size_t i, auto m) { acc = acc + load(x + i, m); });
    return reduce_add(acc);
}

template <Target Tg> float dot(const float* a, const float* b, std::size_t n) {
    using V = vec<float, Tg>;
    constexpr std::size_t W = V::N;
    V a0 = splat<Tg>(0.f), a1 = a0, a2 = a0, a3 = a0;
    std::size_t i = 0;
    for (; i + 4 * W <= n; i += 4 * W) {
        a0 = fma(load<Tg>(a + i), load<Tg>(b + i), a0);
        a1 = fma(load<Tg>(a + i + W), load<Tg>(b + i + W), a1);
        a2 = fma(load<Tg>(a + i + 2 * W), load<Tg>(b + i + 2 * W), a2);
        a3 = fma(load<Tg>(a + i + 3 * W), load<Tg>(b + i + 3 * W), a3);
    }
    strip<Tg, float>(n - i, [&](std::size_t j, auto m) { a0 = fma(load(a + i + j, m), load(b + i + j, m), a0); });
    return reduce_add((a0 + a1) + (a2 + a3));
}

template <Target Tg> float max(const float* x, std::size_t n) {
    auto acc = splat<Tg>(-INFINITY);
    strip<Tg, float>(n, [&](std::size_t i, auto m) {
        acc = lane::max(acc, select(m, load(x + i, m), splat<Tg>(-INFINITY)));
    });
    return reduce_max(acc);
}

// Softmax row kernel: y = exp(x - max) / sum.
template <Target Tg, Policy P = precise> void softmax_row(const float* x, float* y, std::size_t n) {
    float mx = algo::max<Tg>(x, n);
    auto vm = splat<Tg>(mx);
    auto acc = splat<Tg>(0.f);
    strip<Tg, float>(n, [&](std::size_t i, auto m) {
        auto e = select(m, lane::exp<P>(load(x + i, m) - vm), splat<Tg>(0.f));
        store(y + i, e, m);
        acc = acc + e;
    });
    auto inv = splat<Tg>(1.0f / reduce_add(acc));
    strip<Tg, float>(n, [&](std::size_t i, auto m) { store(y + i, load(y + i, m) * inv, m); });
}

template <Target Tg> void rms_norm_row(const float* x, const float* w, float* y, std::size_t n, float eps) {
    float ss = dot<Tg>(x, x, n);
    auto r = splat<Tg>(1.0f / std::sqrt(ss / float(n) + eps));
    strip<Tg, float>(n, [&](std::size_t i, auto m) { store(y + i, load(x + i, m) * r * load(w + i, m), m); });
}

#define FORGE_LANE_UNARY(fname, expr)                                              \
    template <Target Tg, Policy P = precise> void fname(const float* x, float* y, std::size_t n) { \
        strip<Tg, float>(n, [&](std::size_t i, auto m) { auto v = load(x + i, m); store(y + i, (expr), m); }); \
    }
FORGE_LANE_UNARY(exp_n, lane::exp<P>(v))
FORGE_LANE_UNARY(log_n, lane::log<P>(v))
FORGE_LANE_UNARY(tanh_n, lane::tanh<P>(v))
FORGE_LANE_UNARY(sigmoid_n, lane::sigmoid<P>(v))
FORGE_LANE_UNARY(silu_n, lane::silu<P>(v))
FORGE_LANE_UNARY(gelu_n, lane::gelu<P>(v))
#undef FORGE_LANE_UNARY

template <Target Tg, Policy P = precise> vec<float, Tg> apply1(fn1 f, vec<float, Tg> v) {
    switch (f) {
    case fn1::exp2: return lane::exp2<P>(v);     case fn1::expm1: return lane::expm1<P>(v);
    case fn1::log2: return lane::log2<P>(v);     case fn1::log10: return lane::log10<P>(v);
    case fn1::log1p: return lane::log1p<P>(v);   case fn1::sqrt: return lane::sqrt(v);
    case fn1::rsqrt: return lane::rsqrt<P>(v);   case fn1::cbrt: return lane::cbrt<P>(v);
    case fn1::sin: return lane::sin<P>(v);       case fn1::cos: return lane::cos<P>(v);
    case fn1::tan: return lane::tan<P>(v);       case fn1::asin: return lane::asin<P>(v);
    case fn1::acos: return lane::acos<P>(v);     case fn1::atan: return lane::atan<P>(v);
    case fn1::sinh: return lane::sinh<P>(v);     case fn1::cosh: return lane::cosh<P>(v);
    case fn1::erf: return lane::erf<P>(v);       case fn1::erfc: return lane::erfc<P>(v);
    case fn1::gelu_erf: return lane::gelu_erf<P>(v); case fn1::softplus: return lane::softplus<P>(v);
    case fn1::lgamma: return lane::lgamma<P>(v);
    default: return splat<Tg>(NAN);
    }
}
// Dispatched entry: one function pointer covers the whole extended libm; the switch is hoisted per call.
template <Target Tg> void math1_n(int f, const float* x, float* y, std::size_t n) {
    strip<Tg, float>(n, [&](std::size_t i, auto m) { store(y + i, apply1<Tg>(fn1(f), load(x + i, m)), m); });
}
template <Target Tg> void math2_n(int f, const float* a, const float* b, float* y, std::size_t n) {
    strip<Tg, float>(n, [&](std::size_t i, auto m) {
        auto va = load(a + i, m), vb = load(b + i, m);
        auto r = fn2(f) == fn2::pow ? lane::pow(va, vb) : fn2(f) == fn2::atan2 ? lane::atan2(va, vb) : lane::hypot(va, vb);
        store(y + i, r, m);
    });
}
// Fused log-sum-exp row: max + log(sum exp(x - max)).
template <Target Tg> float logsumexp_row(const float* x, std::size_t n) {
    float mx = algo::max<Tg>(x, n);
    if (!(mx > -INFINITY && mx < INFINITY)) return mx;
    auto vm = splat<Tg>(mx);
    auto acc = splat<Tg>(0.f);
    strip<Tg, float>(n, [&](std::size_t i, auto m) { acc = acc + select(m, lane::exp(load(x + i, m) - vm), splat<Tg>(0.f)); });
    return mx + std::log(reduce_add(acc));
}

template <Target Tg> void silu_mul_n(const float* a, const float* b, float* y, std::size_t n) {
    strip<Tg, float>(n, [&](std::size_t i, auto m) {
        auto va = load(a + i, m);
        store(y + i, lane::silu<precise>(va) * load(b + i, m), m);
    });
}

#define FORGE_LANE_BIN(fname, op)                                                              \
    template <Target Tg> void fname(const float* a, const float* b, float* y, std::size_t n) { \
        strip<Tg, float>(n, [&](std::size_t i, auto m) { store(y + i, op(load(a + i, m), load(b + i, m)), m); }); \
    }
FORGE_LANE_BIN(add_n, lane::add)
FORGE_LANE_BIN(sub_n, lane::sub)
FORGE_LANE_BIN(mul_n, lane::mul)
#undef FORGE_LANE_BIN

// ---- GEMM: C[m,n] (+)= A[m,k] * B^T where B is stored [n,k] (linear-layer layout). ----
// Micro-kernel computes a 4 x (2W) tile of C using broadcast-free dot-products on packed rows.
// For the [n,k] layout the natural kernel is a register-blocked multi-dot: 4 rows of A x 4 rows of B.
template <Target Tg> void gemm_nt(const float* A, const float* B, float* C, std::size_t m, std::size_t n, std::size_t k) {
    using V = vec<float, Tg>;
    constexpr std::size_t W = V::N;
    std::size_t i = 0;
    for (; i + 4 <= m; i += 4) {
        std::size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            V acc[4][4];
            for (auto& r : acc) for (auto& c : r) c = splat<Tg>(0.f);
            std::size_t p = 0;
            for (; p + W <= k; p += W) {
                V a[4], b[4];
                for (int r = 0; r < 4; ++r) a[r] = load<Tg>(A + (i + r) * k + p);
                for (int c = 0; c < 4; ++c) b[c] = load<Tg>(B + (j + c) * k + p);
                for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) acc[r][c] = fma(a[r], b[c], acc[r][c]);
            }
            if (p < k) {
                auto mk = mask<float, Tg>::first_n(k - p);
                V a[4], b[4];
                for (int r = 0; r < 4; ++r) a[r] = load(A + (i + r) * k + p, mk);
                for (int c = 0; c < 4; ++c) b[c] = load(B + (j + c) * k + p, mk);
                for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) acc[r][c] = fma(a[r], b[c], acc[r][c]);
            }
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) C[(i + r) * n + j + c] = reduce_add(acc[r][c]);
        }
        for (; j < n; ++j) for (int r = 0; r < 4; ++r) C[(i + r) * n + j] = dot<Tg>(A + (i + r) * k, B + j * k, k);
    }
    for (; i < m; ++i) for (std::size_t j = 0; j < n; ++j) C[i * n + j] = dot<Tg>(A + i * k, B + j * k, k);
}

// C[m,n] = A[m,k] * B[k,n] (row-major B): outer-product micro-kernel 4 x 2W.
template <Target Tg> void gemm_nn(const float* A, const float* B, float* C, std::size_t m, std::size_t n, std::size_t k) {
    using V = vec<float, Tg>;
    constexpr std::size_t W = V::N;
    for (std::size_t i = 0; i < m; i += 4) {
        std::size_t rows = std::min<std::size_t>(4, m - i);
        for (std::size_t j = 0; j < n; j += 2 * W) {
            auto m0 = mask<float, Tg>::first_n(n - j);
            auto m1 = mask<float, Tg>::first_n(n > j + W ? n - j - W : 0);
            V acc[4][2];
            for (auto& r : acc) r[0] = r[1] = splat<Tg>(0.f);
            for (std::size_t p = 0; p < k; ++p) {
                V b0 = load(B + p * n + j, m0);
                V b1 = n > j + W ? load(B + p * n + j + W, m1) : splat<Tg>(0.f);
                for (std::size_t r = 0; r < rows; ++r) {
                    V a = splat<Tg>(A[(i + r) * k + p]);
                    acc[r][0] = fma(a, b0, acc[r][0]);
                    acc[r][1] = fma(a, b1, acc[r][1]);
                }
            }
            for (std::size_t r = 0; r < rows; ++r) {
                store(C + (i + r) * n + j, acc[r][0], m0);
                if (n > j + W) store(C + (i + r) * n + j + W, acc[r][1], m1);
            }
        }
    }
}

} // namespace forge::lane::algo
