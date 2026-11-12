// KV-cache quantisation (int8 / fp8 e4m3) and host offload, checked against the f32 KV path on CPU and Vulkan.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/inference/inference.hpp>
#include <forge/graph/attn_cpu.hpp>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

using namespace forge;
using namespace forge::inference;
using forge::graph::kv_codec;

static std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / ("forge_kvq_" + std::to_string(::getpid()) + "_" + n)).string(); }
static const std::string& model_a() { static std::string p = forge_test::make_tiny_model(tmp("a.fpk")); return p; }
static float maxabs(const std::vector<float>& a) { float m = 0; for (float x : a) m = std::max(m, std::abs(x)); return m; }
static float maxdiff(const std::vector<float>& a, const std::vector<float>& b) { float m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i])); return m; }

// logits for a prompt then `steps` single-token decode steps (fixed tokens, so paths are comparable)
static std::vector<std::vector<float>> trace(inference::session& s, int steps) {
    std::vector<int> p = {1, 5, 9, 33, 70, 101, 7, 12, 3};
    std::vector<std::vector<float>> out; s.reset();
    out.push_back(s.forward(p));
    for (int i = 0; i < steps; ++i) { int t = (i * 37 + 11) % 100 + 1; out.push_back(s.forward(std::span<const int>(&t, 1))); }
    return out;
}

TEST(codec_roundtrip_bounds) {
    std::mt19937 g(3); std::normal_distribution<float> nd(0, 2);
    for (i64 d : {1, 3, 4, 7, 16, 64, 129}) {
        std::vector<float> x(static_cast<std::size_t>(d)), y(static_cast<std::size_t>(d)), buf(static_cast<std::size_t>(graph::kv_head_words(d)));
        for (auto& v : x) v = nd(g);
        x[0] = 9.5f;   // amax
        for (auto c : {kv_codec::int8, kv_codec::fp8}) {
            graph::kv_encode_head(x.data(), d, c, buf.data());
            graph::kv_decode_head(buf.data(), d, c, y.data());
            for (i64 e = 0; e < d; ++e) {
                float err = std::abs(x[e] - y[e]);
                float bound = c == kv_codec::int8 ? 9.5f / 254.f * 1.0001f : std::abs(x[e]) / 16.f + 9.5f / 448.f * std::ldexp(1.f, -9);
                CHECK_MSG(err <= bound + 1e-7f, "codec %d d %lld e %lld x %g y %g", int(c), (long long)d, (long long)e, x[e], y[e]);
            }
            CHECK(std::abs(y[0] - 9.5f) < 1e-5f);   // the amax element is exact up to scale rounding
        }
    }
    std::vector<float> z(8, 0.f), b(3), o(8, 1.f);
    graph::kv_encode_head(z.data(), 8, kv_codec::fp8, b.data()); graph::kv_decode_head(b.data(), 8, kv_codec::fp8, o.data());
    for (float v : o) CHECK(v == 0.f);
    // the fp8 decode formula matches core's E4M3 decoder on every non-NaN code
    for (std::uint32_t c = 0; c < 256; ++c) if ((c & 0x7f) != 0x7f) CHECK(graph::kv_fp8_decode(c) == float(fp8_e4m3::from_bits(std::uint8_t(c))));
}

static void accuracy_vs_f32(const std::string& dev) {
    auto mf = model::open(model_a()), m8 = model::open(model_a()), mq = model::open(model_a());
    if (!dev.empty()) { mf->set_device(dev); m8->set_device(dev); mq->set_device(dev); }
    inference::session sf(mf), s8(m8, {.kv_codec = "int8"}), sq(mq, {.kv_codec = "fp8"});
    auto rf = trace(sf, 24), r8 = trace(s8, 24), rq = trace(sq, 24);
    float w8 = 0, wq = 0;
    for (std::size_t i = 0; i < rf.size(); ++i) {
        float sc = maxabs(rf[i]);
        w8 = std::max(w8, maxdiff(rf[i], r8[i]) / sc); wq = std::max(wq, maxdiff(rf[i], rq[i]) / sc);
    }
    std::printf("  %s: max logit error vs f32 KV: int8 %.2e, fp8 %.2e (relative to max |logit|)\n", dev.empty() ? "cpu" : dev.c_str(), w8, wq);
    CHECK(w8 > 0 && w8 < 2e-2f);
    CHECK(wq > 0 && wq < 8e-2f);
    // memory: each (row, head) is one scale word + ceil(d/4) words instead of d words
    for (auto& [k, sl] : s8.state().slots) {
        CHECK(sl.codec == kv_codec::int8);
        auto& fs = sf.state().slots.at(k);
        i64 d = fs.buf.shape.back();
        CHECK(sl.buf.shape.back() == graph::kv_head_words(d));
    }
    CHECK(s8.state().bytes() < sf.state().bytes());
    // greedy decoding agrees with f32 KV
    sampler g1, g2; std::vector<int> p = {1, 5, 9, 33};
    sf.reset(); s8.reset();
    auto a = sf.generate(p, 12, g1), b = s8.generate(p, 12, g2);
    // (random tiny model: logits are near-tied, so require only the leading tokens to agree)
    std::size_t same = 0; while (same < a.size() && a[same] == b[same]) ++same;
    std::printf("  greedy tokens identical to f32 KV: %zu/%zu\n", same, a.size());
    CHECK(same >= 3);
}

TEST(kv_quant_cpu_accuracy) { accuracy_vs_f32(""); }

TEST(kv_quant_prefix_copy_and_rewind) {
    auto m = model::open(model_a());
    inference::session a(m, {.kv_codec = "int8"}), b(m, {.kv_codec = "int8"}), c(m);
    std::vector<int> p = {4, 8, 15, 16, 23, 42};
    a.reset(); a.forward(p);
    b.copy_prefix(a, 4);
    int t = 9;
    auto lb = b.forward(std::span<const int>(&p[4], 2));
    a.rewind(4); auto la = a.forward(std::span<const int>(&p[4], 2));
    CHECK(maxdiff(la, lb) == 0.f);
    (void)t;
    CHECK_THROWS(c.copy_prefix(a, 2));
}

TEST(kv_quant_vulkan) {
#ifdef FORGE_GPU_VULKAN
    std::string why;
    if (!graph::device_registry("vulkan", &why)) { std::printf("  vulkan unavailable (%s): skipped\n", why.c_str()); return; }
    accuracy_vs_f32("vulkan");
    // Vulkan int8 KV matches CPU int8 KV (same decode, f32 accumulation differences only)
    auto mc = model::open(model_a()), mv = model::open(model_a()); mv->set_device("vulkan");
    inference::session sc(mc, {.kv_codec = "int8"}), sv(mv, {.kv_codec = "int8"});
    auto rc = trace(sc, 8), rv = trace(sv, 8);
    float w = 0; for (std::size_t i = 0; i < rc.size(); ++i) w = std::max(w, maxdiff(rc[i], rv[i]) / maxabs(rc[i]));
    std::printf("  vulkan int8 vs cpu int8: %.2e\n", w);
    CHECK(w < 1e-4f);
    // host offload: identical results, lower peak device memory
    auto mo = model::open(model_a()); mo->set_device("vulkan");
    auto rt = graph::last_vulkan_runtime();
    inference::session so(mo, {.kv_offload = 1});
    auto mn = model::open(model_a()); mn->set_device("vulkan"); auto rt2 = graph::last_vulkan_runtime();
    inference::session sn(mn, {.kv_offload = 0});
    std::vector<int> p = {1, 5, 9, 33, 70, 101, 7, 12, 3, 44, 55, 66, 77, 88, 99, 10};
    so.reset(); auto lo = so.forward(p); sn.reset(); auto ln = sn.forward(p);
    CHECK(maxdiff(lo, ln) == 0.f);
    auto po = rt->residency().peak_resident_bytes, pn = rt2->residency().peak_resident_bytes;
    std::printf("  peak device bytes: offload %zu, resident %zu\n", po, pn);
    CHECK(po < pn);
#else
    std::puts("  vulkan not built: skipped");
#endif
}

// ---- TurboQuant (arXiv 2504.19874): random-sign FWHT rotation + Lloyd-Max scalar codes ----
TEST(tbq_roundtrip_mse) {
    std::mt19937 g(11); std::normal_distribution<float> nd(0, 1);
    for (i64 d : {8, 64, 96, 128, 256}) {
        for (auto c : {kv_codec::tbq4, kv_codec::tbq3}) {
            double num = 0, den = 0;
            std::vector<float> x(static_cast<std::size_t>(d)), y(static_cast<std::size_t>(d)), buf(static_cast<std::size_t>(graph::kv_head_words(d, c)));
            for (int r = 0; r < 200; ++r) {
                for (auto& v : x) v = nd(g) * (r % 3 == 0 ? 5.f : 1.f);
                x[std::size_t(r) % std::size_t(d)] += 12.f;   // an outlier channel: the rotation spreads it
                graph::kv_encode_head(x.data(), d, c, buf.data());
                graph::kv_decode_head(buf.data(), d, c, y.data());
                for (i64 e = 0; e < d; ++e) { num += double(x[e] - y[e]) * (x[e] - y[e]); den += double(x[e]) * x[e]; }
            }
            double rel = num / den;
            std::printf("  %s d %lld: relative MSE %.4f, words/head %lld (f32 %lld)\n", c == kv_codec::tbq4 ? "tbq4" : "tbq3", (long long)d, rel,
                        (long long)graph::kv_head_words(d, c), (long long)d);
            // Lloyd-Max on N(0,1): 4-bit 0.0095, 3-bit 0.0345 (paper Thm 1); small d is a touch worse
            CHECK_MSG(rel < (c == kv_codec::tbq4 ? (d < 64 ? 0.03 : 0.013) : (d < 64 ? 0.07 : 0.045)), "rel %g", rel);
        }
    }
    std::vector<float> z(64, 0.f), b(std::size_t(graph::kv_head_words(64, kv_codec::tbq4))), o(64, 1.f);
    graph::kv_encode_head(z.data(), 64, kv_codec::tbq4, b.data()); graph::kv_decode_head(b.data(), 64, kv_codec::tbq4, o.data());
    for (float v : o) CHECK(v == 0.f);
    auto p = graph::kv_codec_pair_from_name("k8v4");
    CHECK(p.k == kv_codec::int8 && p.v == kv_codec::tbq4);
    CHECK(graph::kv_codec_pair_from_name("tbq3").k == kv_codec::tbq3);
    CHECK_THROWS(graph::kv_codec_pair_from_name("tbq9"));
}

// TurboQuant_prod: (b-1)-bit MSE codes + 1-bit QJL residual sketch. Inner products should be unbiased.
TEST(tbq_qjl_inner_product) {
    std::mt19937 g(5); std::normal_distribution<float> nd(0, 1);
    for (i64 d : {64, 128}) {
        std::vector<float> q(static_cast<std::size_t>(d)); for (auto& v : q) v = nd(g);
        for (auto c : {kv_codec::tbq4, kv_codec::tbq4p, kv_codec::tbq3, kv_codec::tbq3p}) {
            std::vector<float> k(static_cast<std::size_t>(d)), y(static_cast<std::size_t>(d)), buf(static_cast<std::size_t>(graph::kv_head_words(d, c)));
            double bias = 0, se = 0, ref2 = 0, rel = 0, xx = 0; const int n = 4000;
            for (int r = 0; r < n; ++r) {
                for (auto& v : k) v = nd(g);
                // keys aligned with q: shrinkage bias of a pure MSE quantizer shows up here
                for (i64 e = 0; e < d; ++e) k[e] += 0.5f * q[e];
                graph::kv_encode_head(k.data(), d, c, buf.data());
                graph::kv_decode_head(buf.data(), d, c, y.data());
                double a = 0, b = 0;
                for (i64 e = 0; e < d; ++e) { a += double(q[e]) * k[e]; b += double(q[e]) * y[e]; rel += double(k[e] - y[e]) * (k[e] - y[e]); xx += double(k[e]) * k[e]; }
                bias += b - a; se += (b - a) * (b - a); ref2 += a * a;
            }
            bias /= n; double rmse = std::sqrt(se / n), rms = std::sqrt(ref2 / n);
            std::printf("  %s d %lld: <q,k> bias %+.4f (%.2e of rms), rmse %.4f, vector rel MSE %.4f, words %lld\n", std::string(graph::kv_codec_name(c)).c_str(), (long long)d,
                        bias, bias / rms, rmse / rms, rel / xx, (long long)graph::kv_head_words(d, c));
            if (graph::kv_is_qjl(c)) {
                CHECK_MSG(std::fabs(bias / rms) < 6e-3,   // MSE-only: 8e-3 (tbq4), 3e-2 (tbq3); sampling noise ~1e-3 (stdlib RNGs differ)
                          "bias %g", bias / rms);
                CHECK_MSG(rel / xx < (c == kv_codec::tbq4p ? 0.06 : 0.2), "rel %g", rel / xx);
                CHECK(graph::kv_head_words(d, c) <= graph::kv_head_words(d, c == kv_codec::tbq4p ? kv_codec::tbq4 : kv_codec::tbq3) + 2);
            }
        }
    }
    CHECK(graph::kv_codec_pair_from_name("tbq4p").k == kv_codec::tbq4p);
    CHECK(graph::kv_codec_pair_from_name("tbq3_qjl").v == kv_codec::tbq3p);
}

// The QJL stage in numbers. With y = R x the rotated head, yhat the (b-1)-bit Lloyd-Max reconstruction and
// r = y - yhat, the codec stores |r| and sign(S r) for the orthonormal S = H D3 H D2 (n = dp rows) and decodes
// rhat = c |r| S^T sign(S r). <r, S^T sign(S r)> = |S r|_1, so <r, rhat> = |r|^2 in expectation exactly when
// c = |r| / E|S r|_1 = sqrt(pi) G((n+1)/2) / (n G(n/2)) for S r uniform on its sphere (kv_qjl_gain; the paper's
// sqrt(pi/2)/d for unit-variance Gaussian rows becomes sqrt(pi/2)/sqrt(n) for unit-norm rows, times 1 - 1/(4n)).
// Checked here: the stored norm, that constant (fitted from the data: sum |r|^2 / sum |r| |S r|_1),
// <x, xhat> / <x, x> = 1 with the sketch against 1 - MSE without it, the bias of <q, k> for correlated q,
// the variance bound (pi/2) |r|^2 |q|^2 / n of the paper, and exact zeros.
TEST(tbq_qjl_constant_and_unbiasedness) {
    std::mt19937 g(23); std::normal_distribution<float> nd(0, 1);
    for (auto c : {kv_codec::tbq4p, kv_codec::tbq3p})
    for (i64 d : {16, 64, 128, 256}) {
        const std::size_t n = std::size_t(d), hw = std::size_t(graph::kv_head_words(d, c));
        CHECK(graph::kv_tbq_dim(d) == d);   // powers of two: no padding, so norms carry over to the rotated domain
        // layout: gain, |r|, the (b-1)-bit codes, one sign bit per rotated coordinate
        CHECK(hw == 2 + std::size_t((d + graph::kv_tbq_per_word(c) - 1) / graph::kv_tbq_per_word(c)) + std::size_t((d + 31) / 32));
        std::vector<float> x(n), q(n), full(n), mse(n), buf(hw), nob(hw);
        double self_full = 0, self_mse = 0, rr = 0, rrhat = 0, r_l1 = 0, se = 0, bound = 0, bias_full = 0, bias_mse = 0, ref2 = 0;
        const int N = 20000;
        const double gain = double(graph::kv_qjl_gain(d));
        for (int it = 0; it < N; ++it) {
            const float amp = it % 3 == 0 ? 4.f : 1.f;
            for (auto& v : x) v = nd(g) * amp;
            for (auto& v : q) v = nd(g);
            for (std::size_t e = 0; e < n; ++e) q[e] += 0.5f * x[e] / amp;   // queries correlated with the key
            graph::kv_encode_head(x.data(), d, c, buf.data());
            graph::kv_decode_head(buf.data(), d, c, full.data());
            nob = buf; nob[1] = 0.f;   // |r| = 0 switches the sketch off: the plain MSE reconstruction of the same codes
            graph::kv_decode_head(nob.data(), d, c, mse.data());
            double xx = 0, xf = 0, xm = 0, r2 = 0, rh = 0, rh2 = 0, qx = 0, qf = 0, qm = 0, qq = 0;
            for (std::size_t e = 0; e < n; ++e) {
                const double r = double(x[e]) - mse[e], rhat = double(full[e]) - mse[e];
                xx += double(x[e]) * x[e]; xf += double(x[e]) * full[e]; xm += double(x[e]) * mse[e];
                r2 += r * r; rh += r * rhat; rh2 += rhat * rhat;
                qx += double(q[e]) * x[e]; qf += double(q[e]) * full[e]; qm += double(q[e]) * mse[e]; qq += double(q[e]) * q[e];
            }
            CHECK_MSG(std::fabs(double(buf[1]) - std::sqrt(r2)) <= 1e-3 * std::sqrt(r2) + 1e-6, "stored |r| %g vs %g", double(buf[1]), std::sqrt(r2));
            // rhat = c |r| S^T sign(S r) has norm c sqrt(n) |r| whatever the signs are
            CHECK_MSG(std::fabs(std::sqrt(rh2) - gain * std::sqrt(double(n)) * std::sqrt(r2)) <= 2e-3 * std::sqrt(r2) + 1e-6, "|rhat| %g, |r| %g", std::sqrt(rh2), std::sqrt(r2));
            self_full += xf / xx; self_mse += xm / xx;
            rr += r2; rrhat += rh;
            r_l1 += rh / gain;   // = |r| |S r|_1
            se += (qf - qx) * (qf - qx); bound += 1.5707963267948966 / double(n) * r2 * qq;
            bias_full += qf - qx; bias_mse += qm - qx; ref2 += qx * qx;
        }
        const double c_fit = rr / r_l1, c_asym = 1.2533141373155003 / std::sqrt(double(n)), rms = std::sqrt(ref2 / N);
        const double c_exact = std::sqrt(3.141592653589793) * std::exp(std::lgamma((double(n) + 1) / 2) - std::lgamma(double(n) / 2)) / double(n);
        std::printf("  %s d %lld: <x,xhat>/<x,x> %.4f with the sketch, %.4f without; fitted constant %.5f, exact %.5f, sqrt(pi/2)/sqrt(n) %.5f; "
                    "<q,k> bias/rms %+.1e (MSE only %+.1e); error variance / bound %.3f\n", graph::kv_codec_name(c), (long long)d, self_full / N, self_mse / N,
                    c_fit, c_exact, c_asym, bias_full / N / rms, bias_mse / N / rms, se / bound);
        CHECK_MSG(std::fabs(gain / c_exact - 1) < 1e-4, "kv_qjl_gain %g vs the closed form %g", gain, c_exact);
        // Measured with libstdc++'s RNG: the fitted constant is within 1e-3 of the closed form at every d (the
        // asymptotic sqrt(pi/2)/sqrt(n) is 1.6% off at d 16, 0.4% at d 64); the self ratio is within 4e-4 (tbq4p)
        // and 7e-4 (tbq3p) of 1, against 1 - 0.034 and 1 - 0.115 without the sketch. Sampling noise at N = 20000
        // is about 7e-4 (d 16) .. 2e-4 (d 256) of the constant and 1e-4 (tbq4p) / 2e-4 (tbq3p) of the self ratio.
        const double shrink = c == kv_codec::tbq4p ? 0.0345 : 0.1175;   // Lloyd-Max distortion at 3 / 2 bits
        CHECK_MSG(std::fabs(c_fit / c_exact - 1) < (d < 64 ? 4e-3 : 2.5e-3), "fitted QJL constant %g vs %g", c_fit, c_exact);
        CHECK_MSG(std::fabs(rrhat / rr - 1) < (d < 64 ? 4e-3 : 2.5e-3), "<r, rhat> / |r|^2 = %g", rrhat / rr);
        CHECK_MSG(std::fabs(self_full / N - 1) < std::max(1.2e-3, 0.025 * shrink), "QJL self inner product ratio %g", self_full / N);
        // the MSE-only estimate shrinks: <x, xhat> = (1 - relative MSE) <x, x> for a least-squares gain
        CHECK_MSG(1 - self_mse / N > 0.6 * shrink && 1 - self_mse / N < 1.2 * shrink, "MSE-only shrinkage %g (expected about %g)", 1 - self_mse / N, shrink);
        CHECK_MSG(std::fabs(bias_mse / N / rms) > 4 * std::fabs(bias_full / N / rms), "bias: MSE only %g, QJL %g", bias_mse / N / rms, bias_full / N / rms);
        CHECK_MSG(std::fabs(bias_full / N / rms) < (c == kv_codec::tbq4p ? 2.5e-3 : 4e-3), "QJL bias %g of rms", bias_full / N / rms);
        CHECK_MSG(se <= bound, "QJL inner-product error variance %g above the bound %g", se, bound);
    }
    // padded head dimension (d = 96 -> 128 rotated coordinates): still unbiased against the true inner product
    {
        const i64 d = 96; const auto c = kv_codec::tbq4p;
        std::vector<float> x(96), y(96), buf(std::size_t(graph::kv_head_words(d, c)));
        double ratio = 0; const int N = 20000;
        for (int it = 0; it < N; ++it) {
            for (auto& v : x) v = nd(g);
            graph::kv_encode_head(x.data(), d, c, buf.data()); graph::kv_decode_head(buf.data(), d, c, y.data());
            double xx = 0, xy = 0; for (std::size_t e = 0; e < 96; ++e) { xx += double(x[e]) * x[e]; xy += double(x[e]) * y[e]; }
            ratio += xy / xx;
        }
        CHECK_MSG(std::fabs(ratio / N - 1) < 3e-3, "padded QJL self inner product ratio %g", ratio / N);
    }
    for (auto c : {kv_codec::tbq4p, kv_codec::tbq3p}) {   // zeros stay exact zeros; a head re-encodes to the same words
        std::vector<float> z(64, 0.f), b(std::size_t(graph::kv_head_words(64, c))), b2(b.size()), o(64, 1.f);
        graph::kv_encode_head(z.data(), 64, c, b.data()); graph::kv_decode_head(b.data(), 64, c, o.data());
        for (float v : o) CHECK(v == 0.f);
        for (auto& v : z) v = nd(g);
        graph::kv_encode_head(z.data(), 64, c, b.data()); graph::kv_encode_head(z.data(), 64, c, b2.data());
        CHECK(std::memcmp(b.data(), b2.data(), b.size() * 4) == 0);
    }
    CHECK(std::string(graph::kv_codec_pair_name(graph::kv_codec_pair_from_name("turbo4p"))) == "tbq4p");
    CHECK(!graph::kv_is_qjl(kv_codec::tbq4) && graph::kv_is_qjl(kv_codec::tbq3p) && graph::kv_is_tbq(kv_codec::tbq4p));
}

// A QJL KV cache through the session paths that hold or move quantised rows: prefix blobs (save_prefix /
// load_prefix, what the spill store writes), copy_prefix and rewind give bit-identical continuations.
TEST(kv_qjl_prefix_blob_copy_rewind) {
    auto m = model::open(model_a());
    for (const char* name : {"tbq4p", "tbq3p"}) {
        inference::session a(m, {.kv_codec = name}), b(m, {.kv_codec = name}), c(m, {.kv_codec = name}), other(m, {.kv_codec = "tbq3"});
        std::vector<int> p = {4, 8, 15, 16, 23, 42, 9, 77};
        a.reset(); a.forward(p);
        auto blob = a.save_prefix();
        b.load_prefix(blob);
        CHECK(b.save_prefix() == blob);
        for (auto& [k, sl] : b.state().slots) CHECK(graph::kv_is_qjl(sl.codec));
        int t = 31;
        auto la = a.forward(std::span<const int>(&t, 1)), lb = b.forward(std::span<const int>(&t, 1));
        CHECK(maxdiff(la, lb) == 0.f);
        c.copy_prefix(a, 5);
        auto lc = c.forward(std::span<const int>(&p[5], 3));
        a.rewind(5); auto lr = a.forward(std::span<const int>(&p[5], 3));
        CHECK(maxdiff(lc, lr) == 0.f);
        CHECK_THROWS(other.load_prefix(blob));   // a different codec refuses the blob
        CHECK_THROWS(other.copy_prefix(a, 2));
        sampler g1; a.reset();
        CHECK(a.generate(p, 6, g1).size() == 6);
    }
}

// The int16-query attention path (FORGE_CPU_ATTN=q16, here through attn_cpu::set_query_i16) on the tiny model
// with an int8 KV cache: logits stay within the int8 tolerance of f32 KV and next to the float-query kernel.
TEST(kv_int8_int16_query_end_to_end) {
    namespace ac = forge::graph::attn_cpu;
    if (!ac::enabled()) { std::puts("  FORGE_CPU_ATTN=old: skipped"); return; }
    auto mf = model::open(model_a()), m8 = model::open(model_a()), mq = model::open(model_a());
    inference::session sf(mf), s8(m8, {.kv_codec = "int8"}), sq(mq, {.kv_codec = "int8"});
    ac::set_query_i16(0);
    auto rf = trace(sf, 24), r8 = trace(s8, 24);
    sampler g0; std::vector<int> p = {1, 5, 9, 33};
    s8.reset(); auto t8 = s8.generate(p, 12, g0);
    ac::set_query_i16(1);
    auto rq = trace(sq, 24);
    // one token at a time against one prefill block (other ops differ in the last bits between the two shapes)
    std::vector<int> toks = {1, 5, 9, 33, 70, 101, 7, 12, 3};
    sq.reset(); auto all = sq.forward_all(toks);
    sq.reset(); std::vector<float> last;
    for (int tk : toks) last = sq.forward(std::span<const int>(&tk, 1));
    std::vector<float> tail(all.end() - std::ptrdiff_t(last.size()), all.end());
    CHECK_MSG(maxdiff(last, tail) < 1e-4f * maxabs(last), "decode vs prefill %g", maxdiff(last, tail) / maxabs(last));
    sampler g1; sq.reset(); auto tq = sq.generate(p, 12, g1);
    ac::set_query_i16(-1);
    float wf = 0, w8 = 0;
    for (std::size_t i = 0; i < rf.size(); ++i) { wf = std::max(wf, maxdiff(rf[i], rq[i]) / maxabs(rf[i])); w8 = std::max(w8, maxdiff(r8[i], rq[i]) / maxabs(r8[i])); }
    std::size_t same = 0; while (same < t8.size() && same < tq.size() && t8[same] == tq[same]) ++same;
    std::printf("  int8 KV, int16 query: max logit error vs f32 KV %.2e, vs the float-query kernel %.2e; greedy tokens identical to the float-query kernel %zu/%zu\n",
                wf, w8, same, t8.size());
    CHECK(wf > 0 && wf < 2e-2f);
    CHECK(w8 < 1e-3f);
    CHECK(tq.size() == 12 && same >= 3);
}

static double kld(const std::vector<float>& p, const std::vector<float>& q) {
    auto sm = [](const std::vector<float>& l) { float m = *std::max_element(l.begin(), l.end()); std::vector<double> o(l.size()); double s = 0;
        for (std::size_t i = 0; i < l.size(); ++i) s += o[i] = std::exp(double(l[i] - m)); for (auto& v : o) v /= s; return o; };
    auto a = sm(p), b = sm(q); double k = 0;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i] > 0) k += a[i] * std::log(a[i] / std::max(b[i], 1e-300));
    return k;
}
static std::size_t argmax(const std::vector<float>& v) { return std::size_t(std::max_element(v.begin(), v.end()) - v.begin()); }

static void tbq_vs_f32(const std::string& dev) {
    for (const char* name : {"tbq4", "tbq3", "k8v4", "tbq4p", "tbq3p"}) {
        if (!dev.empty() && name[4] == 'p') continue;   // QJL types are CPU only
        auto mf = model::open(model_a()), mq = model::open(model_a());
        if (!dev.empty()) { mf->set_device(dev); mq->set_device(dev); }
        inference::session sf(mf), sq(mq, {.kv_codec = name});
        auto rf = trace(sf, 24), rq = trace(sq, 24);
        double kl = 0; std::size_t agree = 0; float w = 0;
        for (std::size_t i = 0; i < rf.size(); ++i) { kl += kld(rf[i], rq[i]); agree += argmax(rf[i]) == argmax(rq[i]); w = std::max(w, maxdiff(rf[i], rq[i]) / maxabs(rf[i])); }
        kl /= double(rf.size());
        std::printf("  %s %s: mean KLD vs f32 KV %.2e, top-1 agreement %zu/%zu, max rel logit err %.2e, KV bytes %zu vs %zu\n", dev.empty() ? "cpu" : dev.c_str(), name, kl, agree,
                    rf.size(), w, std::size_t(sq.state().bytes()), std::size_t(sf.state().bytes()));
        CHECK(w > 0);
        // (random tiny model: near-tied logits make KLD large in absolute terms; tbq4 ~1.5e-2, tbq3 ~4e-2 here)
        const bool loose = std::string(name) == "tbq3" || std::string(name) == "tbq3p";
        CHECK_MSG(kl < (loose ? 1.2e-1 : std::string(name) == "tbq4p" ? 6e-2 : 3e-2), "kld %g", kl);
        // (tbq4p sits between tbq4 and tbq3 in vector MSE; top-1 on near-tied logits is noisy: 19/25 here)
        CHECK(agree * 10 >= rf.size() * (loose || std::string(name) == "tbq4p" ? 6 : 8));
        CHECK(sq.state().bytes() < sf.state().bytes());
        bool any_v = false, any_k = false;
        for (auto& [k, sl] : sq.state().slots) {
            i64 d = sf.state().slots.at(k).buf.shape.back();
            CHECK(sl.buf.shape.back() == graph::kv_head_words(d, sl.codec));
            if (std::string(name) == "k8v4") { any_k |= sl.codec == kv_codec::int8; any_v |= sl.codec == kv_codec::tbq4; }
        }
        if (std::string(name) == "k8v4") CHECK(any_k && any_v);
    }
}

TEST(kv_tbq_cpu_accuracy) { tbq_vs_f32(""); }

TEST(kv_tbq_cuda) {
    std::string why;
    if (!graph::device_registry("cuda", &why)) { std::printf("  cuda unavailable (%s): skipped\n", why.c_str()); return; }
    tbq_vs_f32("cuda");
    for (const char* name : {"tbq4", "tbq3", "k8v4"}) {   // CUDA fused-decode attention vs the CPU decode path, same codes
        auto mc = model::open(model_a()), mg = model::open(model_a()); mg->set_device("cuda");
        inference::session sc(mc, {.kv_codec = name}), sg(mg, {.kv_codec = name});
        auto rc = trace(sc, 8), rg = trace(sg, 8);
        float w = 0; for (std::size_t i = 0; i < rc.size(); ++i) w = std::max(w, maxdiff(rc[i], rg[i]) / maxabs(rc[i]));
        std::printf("  cuda %s vs cpu %s: %.2e\n", name, name, w);
        CHECK(w < 1e-3f);
    }
}

TEST(kv_tbq_vulkan) {
#ifdef FORGE_GPU_VULKAN
    std::string why;
    if (!graph::device_registry("vulkan", &why)) { std::printf("  vulkan unavailable (%s): skipped\n", why.c_str()); return; }
    for (const char* name : {"tbq4", "tbq3", "k8v4"}) {
        auto mc = model::open(model_a()), mv = model::open(model_a()); mv->set_device("vulkan");
        inference::session sc(mc, {.kv_codec = name}), sv(mv, {.kv_codec = name});
        auto rc = trace(sc, 8), rv = trace(sv, 8);
        float w = 0; for (std::size_t i = 0; i < rc.size(); ++i) w = std::max(w, maxdiff(rc[i], rv[i]) / maxabs(rc[i]));
        std::printf("  vulkan %s vs cpu %s: %.2e\n", name, name, w);
        CHECK(w < 1e-3f);
    }
#else
    std::puts("  vulkan not built: skipped");
#endif
}
