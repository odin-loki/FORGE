// KV-cache quantisation (int8 / fp8 e4m3) and host offload, checked against the f32 KV path on CPU and Vulkan.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/mind/mind.hpp>
#include <filesystem>
#include <random>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace forge;
using namespace forge::mind;
using forge::flow::kv_codec;

static std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / ("forge_kvq_" + std::to_string(::getpid()) + "_" + n)).string(); }
static const std::string& model_a() { static std::string p = forge_test::make_tiny_model(tmp("a.fpk")); return p; }
static float maxabs(const std::vector<float>& a) { float m = 0; for (float x : a) m = std::max(m, std::abs(x)); return m; }
static float maxdiff(const std::vector<float>& a, const std::vector<float>& b) { float m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i])); return m; }

// logits for a prompt then `steps` single-token decode steps (fixed tokens, so paths are comparable)
static std::vector<std::vector<float>> trace(mind::session& s, int steps) {
    std::vector<int> p = {1, 5, 9, 33, 70, 101, 7, 12, 3};
    std::vector<std::vector<float>> out; s.reset();
    out.push_back(s.forward(p));
    for (int i = 0; i < steps; ++i) { int t = (i * 37 + 11) % 100 + 1; out.push_back(s.forward(std::span<const int>(&t, 1))); }
    return out;
}

TEST(codec_roundtrip_bounds) {
    std::mt19937 g(3); std::normal_distribution<float> nd(0, 2);
    for (i64 d : {1, 3, 4, 7, 16, 64, 129}) {
        std::vector<float> x(static_cast<std::size_t>(d)), y(static_cast<std::size_t>(d)), buf(static_cast<std::size_t>(flow::kv_head_words(d)));
        for (auto& v : x) v = nd(g);
        x[0] = 9.5f;   // amax
        for (auto c : {kv_codec::int8, kv_codec::fp8}) {
            flow::kv_encode_head(x.data(), d, c, buf.data());
            flow::kv_decode_head(buf.data(), d, c, y.data());
            for (i64 e = 0; e < d; ++e) {
                float err = std::abs(x[e] - y[e]);
                float bound = c == kv_codec::int8 ? 9.5f / 254.f * 1.0001f : std::abs(x[e]) / 16.f + 9.5f / 448.f * std::ldexp(1.f, -9);
                CHECK_MSG(err <= bound + 1e-7f, "codec %d d %lld e %lld x %g y %g", int(c), (long long)d, (long long)e, x[e], y[e]);
            }
            CHECK(std::abs(y[0] - 9.5f) < 1e-5f);   // the amax element is exact up to scale rounding
        }
    }
    std::vector<float> z(8, 0.f), b(3), o(8, 1.f);
    flow::kv_encode_head(z.data(), 8, kv_codec::fp8, b.data()); flow::kv_decode_head(b.data(), 8, kv_codec::fp8, o.data());
    for (float v : o) CHECK(v == 0.f);
    // the fp8 decode formula matches core's E4M3 decoder on every non-NaN code
    for (std::uint32_t c = 0; c < 256; ++c) if ((c & 0x7f) != 0x7f) CHECK(flow::kv_fp8_decode(c) == float(fp8_e4m3::from_bits(std::uint8_t(c))));
}

static void accuracy_vs_f32(const std::string& dev) {
    auto mf = model::open(model_a()), m8 = model::open(model_a()), mq = model::open(model_a());
    if (!dev.empty()) { mf->set_device(dev); m8->set_device(dev); mq->set_device(dev); }
    mind::session sf(mf), s8(m8, {.kv_codec = "int8"}), sq(mq, {.kv_codec = "fp8"});
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
        CHECK(sl.buf.shape.back() == flow::kv_head_words(d));
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
    mind::session a(m, {.kv_codec = "int8"}), b(m, {.kv_codec = "int8"}), c(m);
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
#ifdef FORGE_GRID_VULKAN
    std::string why;
    if (!flow::device_registry("vulkan", &why)) { std::printf("  vulkan unavailable (%s): skipped\n", why.c_str()); return; }
    accuracy_vs_f32("vulkan");
    // Vulkan int8 KV matches CPU int8 KV (same decode, f32 accumulation differences only)
    auto mc = model::open(model_a()), mv = model::open(model_a()); mv->set_device("vulkan");
    mind::session sc(mc, {.kv_codec = "int8"}), sv(mv, {.kv_codec = "int8"});
    auto rc = trace(sc, 8), rv = trace(sv, 8);
    float w = 0; for (std::size_t i = 0; i < rc.size(); ++i) w = std::max(w, maxdiff(rc[i], rv[i]) / maxabs(rc[i]));
    std::printf("  vulkan int8 vs cpu int8: %.2e\n", w);
    CHECK(w < 1e-4f);
    // host offload: identical results, lower peak device memory
    auto mo = model::open(model_a()); mo->set_device("vulkan");
    auto rt = flow::last_vulkan_runtime();
    mind::session so(mo, {.kv_offload = 1});
    auto mn = model::open(model_a()); mn->set_device("vulkan"); auto rt2 = flow::last_vulkan_runtime();
    mind::session sn(mn, {.kv_offload = 0});
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
