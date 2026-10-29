#include "check.hpp"
#include <forge/lane/dispatch.hpp>
#include <forge/lane/caps.hpp>
#include <forge/verify/oracle.hpp>
#include <cmath>
#include <random>

using namespace forge;
using namespace forge::lane;

// ---- concept subsumption: most constrained overload wins, no manual ordering ----
template <Target T> requires Has<T, avx2> constexpr int pick() { return 1; }
template <Target T> requires Has<T, avx2> && Has<T, avx512f> constexpr int pick() { return 2; }
template <Target T> constexpr int pick() { return 0; }
static_assert(pick<scalar_t>() == 0);
static_assert(pick<x86_64_v3>() == 1);
static_assert(pick<x86_64_v4>() == 2);
static_assert(pick<sapphire>() == 2);
static_assert(Has<sapphire, amx_int8> && !Has<zen4, amx_int8> && Has<zen4, avx512vnni>);
static_assert(!Has<neoverse2, avx2> && Has<neoverse2, sve2>);

static std::vector<const kernels*> tables() {
    std::vector<const kernels*> v;
    for (const char* t : {"scalar", "avx2", "avx512", "neon"}) if (auto* k = by_name(t)) v.push_back(k);
    return v;
}

TEST(detection_reports) {
    auto& f = detect();
    std::printf("  cpu: %s\n  active: %s\n", f.summary().c_str(), active().target);
#if defined(__x86_64__)
    CHECK(f.sse2);
#elif defined(__aarch64__)
    CHECK(f.neon);
#endif
}

// strip/masks: every length 0..100, results must match scalar and never touch past the end.
TEST(strip_tails_all_lengths) {
    for (auto* k : tables()) {
        for (std::size_t n = 0; n <= 100; ++n) {
            std::vector<float> x(n + 32), y(n + 32, 7.f);
            for (std::size_t i = 0; i < n; ++i) { x[i] = float(i) * 0.5f; y[i] = 1.f; }
            k->saxpy(2.f, x.data(), y.data(), n);
            for (std::size_t i = 0; i < n; ++i) CHECK(y[i] == 1.f + float(i));
            for (std::size_t i = n; i < n + 32; ++i) CHECK_MSG(y[i] == 7.f, "%s wrote past end n=%zu", k->target, n);
            float s = k->sum(x.data(), n), e = 0; for (std::size_t i = 0; i < n; ++i) e += x[i];
            CHECK(std::fabs(s - e) <= 1e-4f * std::max(1.f, e));
        }
    }
}

// ULP harness for math functions over sampled ranges plus hard cases.
template <class F, class R>
static void math_sweep(const char* name, F kernel_of, R ref, float lo, float hi, double max_ulp, double abs_floor) {
    std::vector<float> x;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(lo, hi);
    for (int i = 0; i < 200000; ++i) x.push_back(d(rng));
    for (float h : {0.f, -0.f, 1e-30f, -1e-30f, 1.f, -1.f, lo, hi}) x.push_back(h);
    std::vector<float> y(x.size());
    for (auto* k : tables()) {
        kernel_of(*k)(x.data(), y.data(), x.size());
        std::int64_t worst = 0; double worst_x = 0;
        bool ok = true;
        for (std::size_t i = 0; i < x.size(); ++i) {
            float e = float(ref(double(x[i])));
            auto u = verify::ulp_distance(y[i], e);
            if (std::fabs(double(y[i]) - e) <= abs_floor) continue;
            if (u > worst) { worst = u; worst_x = x[i]; }
            if (double(u) > max_ulp) ok = false;
        }
        std::printf("  %-8s %-7s max %lld ulp (x=%g)\n", name, k->target, (long long)worst, worst_x);
        CHECK_MSG(ok, "%s on %s exceeds %g ulp", name, k->target, max_ulp);
    }
}
TEST(math_ulp_harness) {
    math_sweep("exp", [](const kernels& k) { return k.exp; }, [](double x) { return std::exp(x); }, -87.f, 88.f, 2, 0);
    math_sweep("exp-sub", [](const kernels& k) { return k.exp; }, [](double x) { return std::exp(x); }, -103.f, -87.f, 2, 1e-44);
    math_sweep("log", [](const kernels& k) { return k.log; }, [](double x) { return std::log(x); }, 1e-30f, 1e30f, 2, 0);
    math_sweep("log01", [](const kernels& k) { return k.log; }, [](double x) { return std::log(x); }, 0.5f, 2.f, 2, 1e-7);
    math_sweep("tanh", [](const kernels& k) { return k.tanh; }, [](double x) { return std::tanh(x); }, -10.f, 10.f, 4, 1e-7);
    math_sweep("sigmoid", [](const kernels& k) { return k.sigmoid; }, [](double x) { return 1 / (1 + std::exp(-x)); }, -30.f, 30.f, 4, 1e-7);
    math_sweep("silu", [](const kernels& k) { return k.silu; }, [](double x) { return x / (1 + std::exp(-x)); }, -30.f, 30.f, 4, 1e-6);
    math_sweep("gelu", [](const kernels& k) { return k.gelu; },
               [](double x) { return 0.5 * x * (1 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); }, -10.f, 10.f, 8, 1e-6);
}
TEST(math_special_values) {
    float in[6] = {0.f, -1.f, INFINITY, -INFINITY, NAN, 1e-40f}, out[6];
    for (auto* k : tables()) {
        k->log(in, out, 6);
        CHECK(std::isinf(out[0]) && out[0] < 0); CHECK(std::isnan(out[1])); CHECK(std::isinf(out[2]));
        CHECK(std::isnan(out[3])); CHECK(std::isnan(out[4])); CHECK(std::fabs(out[5] - std::log(1e-40f)) < 1e-4f);
        k->exp(in, out, 6);
        CHECK(out[0] == 1.f); CHECK(std::isinf(out[2])); CHECK(out[3] == 0.f);
    }
}

TEST(kernels_vs_spec_oracle) {
    std::mt19937 rng(7);
    for (auto* k : tables()) {
        for (i64 n : {1, 7, 16, 33, 100, 257}) {
            auto x = verify::random_tensor({3, n}, rng, -5, 5);
            auto w = verify::random_tensor({n}, rng, -2, 2);
            tensor y(x.shape);
            for (int r = 0; r < 3; ++r) k->softmax_row(&x.data[std::size_t(r * n)], &y.data[std::size_t(r * n)], std::size_t(n));
            auto ref = spec::ref::softmax({x}, {});
            auto c = verify::compare(y, ref, spec::info(spec::op::softmax).tol);
            CHECK_MSG(c.ok, "softmax %s n=%lld %s", k->target, (long long)n, c.message.c_str());
            for (int r = 0; r < 3; ++r) k->rms_norm_row(&x.data[std::size_t(r * n)], w.data.data(), &y.data[std::size_t(r * n)], std::size_t(n), 1e-5f);
            c = verify::compare(y, spec::ref::rms_norm({x, w}, {{"eps", 1e-5}}), spec::info(spec::op::rms_norm).tol);
            CHECK_MSG(c.ok, "rms_norm %s %s", k->target, c.message.c_str());
            auto b = verify::random_tensor({3, n}, rng, -5, 5);
            k->silu_mul(x.data.data(), b.data.data(), y.data.data(), y.size());
            c = verify::compare(y, spec::ref::silu_mul({x, b}, {}), spec::info(spec::op::silu_mul).tol);
            CHECK_MSG(c.ok, "silu_mul %s %s", k->target, c.message.c_str());
        }
    }
}

TEST(gemm_vs_oracle) {
    std::mt19937 rng(3);
    for (auto* k : tables()) {
        for (auto [m, n, kk] : {std::tuple<i64, i64, i64>{1, 1, 1}, {4, 4, 16}, {5, 7, 3}, {13, 29, 67}, {64, 48, 128}, {3, 100, 17}}) {
            auto A = verify::random_tensor({m, kk}, rng, -1, 1);
            auto Bt = verify::random_tensor({n, kk}, rng, -1, 1);
            auto B = verify::random_tensor({kk, n}, rng, -1, 1);
            tensor C({m, n});
            k->gemm_nt(A.data.data(), Bt.data.data(), C.data.data(), std::size_t(m), std::size_t(n), std::size_t(kk));
            auto c = verify::compare(C, spec::ref::matmul_t({A, Bt}, {}), spec::info(spec::op::matmul_t).tol);
            CHECK_MSG(c.ok, "gemm_nt %s %lldx%lldx%lld %s", k->target, (long long)m, (long long)n, (long long)kk, c.message.c_str());
            k->gemm_nn(A.data.data(), B.data.data(), C.data.data(), std::size_t(m), std::size_t(n), std::size_t(kk));
            c = verify::compare(C, spec::ref::matmul({A, B}, {}), spec::info(spec::op::matmul).tol);
            CHECK_MSG(c.ok, "gemm_nn %s %s", k->target, c.message.c_str());
        }
    }
}
