#include "check.hpp"
#include <forge/gpu/prims.hpp>
#include <forge/gpu/select.hpp>
#include <forge/ops/reference.hpp>
#include <forge/ops/ops.gen.hpp>
#include <complex>
#include <random>

using namespace forge;

struct Scale {
    float a;
    FORGE_KERNEL void operator()(gpu::item<3> it, gpu::span<float> y) const {
        if (auto i = it.global(0); i < i64(y.size())) y[std::size_t(i)] *= a;
    }
};

TEST(graph_first_record_compile_replay) {
    auto dev = gpu::select(gpu::prefer::discrete);
    std::size_t n = 100000;
    std::vector<float> host(n, 1.f), back(n, 0.f);
    auto buf = dev.alloc<float>(n);
    gpu::graph g;
    auto up = g.copy(gpu::span<const float>(host), buf);
    auto k = g.launch(Scale{2.0f}, gpu::extent{i64(n)}, buf); k.after(up);
    g.copy(buf, gpu::span<float>(back)).after(k);
    auto exe = g.compile(dev);
    CHECK(exe.steps() == 3);
    exe.run();
    CHECK(back[0] == 2.f && back[n - 1] == 2.f);
    exe.run();  // replay: host -> 2x again (host unchanged)
    CHECK(back[12345] == 2.f);
}
TEST(backend_probe_and_caps) {
    auto pr = gpu::probe_backends();
    CHECK(pr.size() == 7 && pr[0].available);
    for (std::size_t i = 0; i < pr.size(); ++i) CHECK(int(pr[i].kind) == int(i));   // one entry per backend_kind, OpenCL included
    for (auto& p : pr) std::printf("  %-10s %s %s\n", gpu::name(p.kind), p.available ? "yes" : "no ", p.detail.c_str());
    auto d = gpu::select();
    CHECK(d.caps().compute_units >= 1);
}
// Automatic device selection (gpu/select.hpp): the scoring rule over synthetic capability records.
static gpu::device_record rec(const char* dev, const char* backend, gpu::device_class cls, std::uint64_t mem_gb, std::vector<std::string> q = {}, double gflops = 0, int ord = 0) {
    gpu::device_record r; r.device = dev; r.backend = backend; r.cls = cls; r.memory_bytes = mem_gb << 30; r.qcodecs = std::move(q); r.gflops = gflops; r.ordinal = ord; return r;
}
TEST(device_selection_scoring) {
    using gpu::device_class; using gpu::select_device; using gpu::select_request;
    const std::vector<std::string> kq{"q8_0", "q4_k", "q5_k", "q6_k"};
    auto disc = rec("cuda:0", "cuda", device_class::discrete, 24, kq), integ = rec("opencl:0", "opencl", device_class::integrated, 8, kq);
    auto soft = rec("vulkan", "vulkan", device_class::software, 16), mock = rec("opencl:1", "opencl", device_class::software, 1, kq, 0, 1);
    select_request rq;
    CHECK(select_device({}, rq) == "cpu");                                   // nothing but the CPU
    CHECK(select_device({disc, integ, soft}, rq) == "cuda:0");               // discrete > integrated > cpu
    CHECK(select_device({integ, soft}, rq) == "opencl:0");
    // software devices never win over the CPU ...
    CHECK(select_device({soft, mock}, rq) == "cpu");
    { auto fast = soft; fast.gflops = 1e6; CHECK(select_device({fast}, rq) == "cpu"); }   // ... whatever they measure
    // ... unless allowed; then they still have to beat it (prior 0.5 < 4), or be preferred
    { auto a = rq; a.allow_software = true; CHECK(select_device({soft}, a) == "cpu");
      a.prefer = {"vulkan"}; CHECK(select_device({soft, mock}, a) == "vulkan"); }
    // memory: a model that does not fit sends the choice to the next device
    { auto a = rq; a.model_bytes = std::uint64_t(20) << 30; CHECK(select_device({disc, integ}, a) == "cuda:0");
      a.model_bytes = std::uint64_t(30) << 30; CHECK(select_device({disc}, a) == "cpu"); }   // 200 * 0.01 < the CPU's 4
    { auto a = rq; a.model_bytes = std::uint64_t(30) << 30;
      auto small = rec("cuda:0", "cuda", device_class::discrete, 24, kq), big = rec("cuda:1", "cuda", device_class::discrete, 48, kq, 0, 1);
      CHECK(select_device({small, big}, a) == "cuda:1"); }
    // codec coverage: the device with a quantised matmul for the model's codec wins over one of the same class without
    { auto a = rq; a.codecs = {"q4_k"};
      auto none = rec("vulkan", "vulkan", device_class::discrete, 24), has = rec("opencl:0", "opencl", device_class::discrete, 24, kq);
      CHECK(select_device({none, has}, a) == "opencl:0");
      CHECK(gpu::score(none, a).score < gpu::score(has, a).score); }
    // exclusion by backend or by device; the CPU cannot be excluded
    { auto a = rq; a.exclude = {"cuda"}; CHECK(select_device({disc, integ}, a) == "opencl:0");
      a.exclude = {"cuda", "opencl:0", "cpu"}; CHECK(select_device({disc, integ}, a) == "cpu"); }
    // preference is a strong override among eligible devices
    { auto a = rq; a.prefer = {"opencl"}; CHECK(select_device({disc, integ}, a) == "opencl:0"); }
    // measured throughput is used only when every candidate has one: here it reverses the class priors
    { auto d = disc, i = integ; d.gflops = 3; i.gflops = 9; auto cpu = rec("cpu", "cpu", device_class::cpu, 0, {}, 5);
      CHECK(select_device({d, i, cpu}, rq) == "opencl:0");
      i.gflops = 0; CHECK(select_device({d, i, cpu}, rq) == "cuda:0");   // one unmeasured: back to priors
      d.gflops = 3; i.gflops = 4; cpu.gflops = 50; CHECK(select_device({d, i, cpu}, rq) == "cpu"); }   // a fast CPU beats slow devices
    // ties: backend order, then ordinal, not enumeration order
    { auto a0 = rec("opencl:0", "opencl", device_class::discrete, 8, kq), a1 = rec("opencl:1", "opencl", device_class::discrete, 8, kq, 0, 1), c0 = rec("cuda:0", "cuda", device_class::discrete, 8, kq);
      CHECK(select_device({a1, a0}, rq) == "opencl:0"); CHECK(select_device({a1, a0, c0}, rq) == "cuda:0"); }
    CHECK(gpu::software_name("llvmpipe (LLVM 17.0.6, 256 bits)") && gpu::software_name("SwiftShader Device") && gpu::software_name("forge mock cuda 0") && !gpu::software_name("NVIDIA GeForce RTX 3090"));
    std::string why; select_device({disc, soft}, rq, &why);
    CHECK(why.find("cuda:0 (discrete") != std::string::npos && why.find("skipped, software device") != std::string::npos);
}
TEST(prims_reduce_scan_compact) {
    std::vector<std::int64_t> x(1 << 18); std::mt19937 r(5);
    for (auto& v : x) v = std::int64_t(r() % 100);
    std::int64_t ref = 0; for (auto v : x) ref += v;
    CHECK(gpu::prims::reduce<std::int64_t>(x) == ref);
    std::vector<std::int64_t> y(x.size());
    gpu::prims::scan<std::int64_t>(x, y, true);
    std::int64_t s = 0; bool ok = true;
    for (std::size_t i = 0; i < x.size(); ++i) { s += x[i]; ok &= y[i] == s; }
    CHECK(ok);
    gpu::prims::scan<std::int64_t>(x, y, false);
    CHECK(y[0] == 0 && y[1] == x[0]);
    std::vector<std::int64_t> out(x.size());
    auto c = gpu::prims::compact<std::int64_t>(x, out, [](auto v) { return v < 10; });
    std::size_t e = 0; for (auto v : x) e += v < 10; CHECK(c == e);
    for (std::size_t i = 0; i < c; ++i) CHECK(out[i] < 10);
}
TEST(prims_radix_sort_topk) {
    std::vector<std::uint32_t> k(50000), v(50000); std::mt19937 r(9);
    for (std::size_t i = 0; i < k.size(); ++i) { k[i] = r(); v[i] = std::uint32_t(i); }
    auto orig = k;
    gpu::prims::radix_sort(k, v);
    CHECK(std::is_sorted(k.begin(), k.end()));
    bool ok = true; for (std::size_t i = 0; i < k.size(); ++i) ok &= orig[v[i]] == k[i];
    CHECK(ok);
    std::vector<float> f = {0.1f, 5.f, 3.f, 5.f, -1.f};
    auto t = gpu::prims::top_k(f, 3);
    CHECK(t.size() == 3 && t[0] == 1 && t[1] == 3 && t[2] == 2);
}
TEST(prims_gemm_and_philox) {
    std::mt19937 r(2); std::uniform_real_distribution<float> d(-1, 1);
    std::size_t m = 37, n = 129, k = 70;
    std::vector<float> A(m * k), B(n * k), C(m * n);
    for (auto& x : A) x = d(r); for (auto& x : B) x = d(r);
    gpu::prims::gemm_nt(A.data(), B.data(), C.data(), m, n, k);
    double maxe = 0;
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < n; ++j) {
        double s = 0; for (std::size_t p = 0; p < k; ++p) s += double(A[i * k + p]) * B[j * k + p];
        maxe = std::max(maxe, std::fabs(s - C[i * n + j]));
    }
    CHECK(maxe < 1e-4);
    std::vector<float> a(10), b(10);
    gpu::prims::philox_uniform(a, 42); gpu::prims::philox_uniform(b, 42);
    CHECK(a == b);
    for (float x : a) CHECK(x >= 0.f && x < 1.f);
    // known-answer test: Philox4x32-10 with zero key/counter (Random123 KAT)
    auto kat = rand::philox4x32{{0, 0}}(0, 0);
    CHECK(kat[0] == 0x6627e8d5u && kat[1] == 0xe169c58du && kat[2] == 0xbc57ac4cu && kat[3] == 0x9b00dbd8u);
}

// ---- FORGE_PLAN B.6 primitives on the CPU, each against the op's float64 reference (ops::ref) ----
namespace {
tensor rnd_t(shape_t s, float lo, float hi, std::uint32_t seed) {
    tensor t(s); for (auto& v : t.data) { seed = seed * 1664525u + 1013904223u; v = lo + (hi - lo) * float(seed >> 8) / float(1u << 24); } return t;
}
double max_err(const float* got, const tensor& ref) {   // max |got - ref| / max(1, |ref|)
    double w = 0; for (std::size_t i = 0; i < ref.data.size(); ++i) w = std::max(w, std::fabs(double(got[i]) - ref.data[i]) / std::max(1.0, std::fabs(double(ref.data[i]))));
    return w;
}
tensor ref_act(const tensor& t, gpu::prims::act a) {
    using gpu::prims::act;
    const char* n = a == act::relu ? "relu" : a == act::silu ? "silu" : a == act::gelu_tanh ? "gelu_tanh" : a == act::tanh ? "tanh" : a == act::sigmoid ? "sigmoid" : nullptr;
    return n ? ops::reference(*ops::op_from_name(n))({t}, {}) : t;
}
}
TEST(prims_gemv_gemm_epilogue) {
    using namespace gpu::prims;
    const std::size_t n = 37, k = 129, batch = 3, m = 5;
    auto W = rnd_t({i64(n), i64(k)}, -1, 1, 1), x = rnd_t({1, i64(k)}, -1, 1, 2), bias = rnd_t({i64(n)}, -1, 1, 3);
    for (auto a : {act::none, act::relu, act::silu, act::gelu_tanh, act::tanh, act::sigmoid}) for (bool with_bias : {false, true}) {
        std::vector<float> y(n);
        gemv(W.data.data(), x.data.data(), y.data(), n, k, {with_bias ? bias.data.data() : nullptr, a});
        tensor r = ops::ref::matmul_t({x, W}, {});
        if (with_bias) r = ops::ref::add({r, bias}, {});
        CHECK_MSG(max_err(y.data(), ref_act(r, a)) < 2e-5, "gemv act %d bias %d", int(a), int(with_bias));
    }
    auto A = rnd_t({i64(batch), i64(m), i64(k)}, -1, 1, 4), Bb = rnd_t({i64(batch), i64(n), i64(k)}, -1, 1, 5);
    std::vector<float> C(batch * m * n);
    gemm_batched(A.data.data(), W.data.data(), C.data(), batch, m, n, k, true, {bias.data.data(), act::silu});
    tensor r = ref_act(ops::ref::add({ops::ref::matmul_t({A, W}, {}), bias}, {}), act::silu);
    CHECK(max_err(C.data(), r) < 2e-5);
    gemm_batched(A.data.data(), Bb.data.data(), C.data(), batch, m, n, k, false, {nullptr, act::relu});
    for (std::size_t b = 0; b < batch; ++b) {   // one weight per batch
        tensor ab({i64(m), i64(k)}), bb({i64(n), i64(k)});
        std::copy_n(A.data.data() + b * m * k, m * k, ab.data.data()); std::copy_n(Bb.data.data() + b * n * k, n * k, bb.data.data());
        CHECK(max_err(C.data() + b * m * n, ref_act(ops::ref::matmul_t({ab, bb}, {}), act::relu)) < 2e-5);
    }
}
TEST(prims_convolution) {
    using namespace gpu::prims;
    {   // 2 groups, 2x3 kernel, per-axis stride / pad / dilation, a shared bias
        auto x = rnd_t({4, 12, 10}, -1, 1, 6), w = rnd_t({6, 2, 2, 3}, -1, 1, 7), b = rnd_t({1}, -1, 1, 8);
        conv_geom g; g.stride[0] = 2; g.pad[1] = 2; g.dilation[1] = 2;
        ops::attrs a{{"stride", std::vector<i64>{2, 1}}, {"pad", std::vector<i64>{0, 2}}, {"dilation", std::vector<i64>{1, 2}}, {"groups", 2.0}};
        tensor r = ops::ref::conv2d({x, w, b}, a); std::vector<float> y(r.data.size(), -7.f);
        conv2d(x.data.data(), w.data.data(), b.data.data(), y.data(), 4, 12, 10, 6, 2, 2, 3, g, 1);
        CHECK(max_err(y.data(), r) < 1e-5);
    }
    {   // depthwise 3x3, pad 1
        auto x = rnd_t({6, 9, 8}, -1, 1, 9), w = rnd_t({6, 1, 3, 3}, -1, 1, 10), b = rnd_t({6}, -1, 1, 11);
        conv_geom g; g.pad[0] = g.pad[1] = 1;
        tensor r = ops::ref::conv2d({x, w, b}, {{"pad", 1.0}, {"groups", 6.0}}); std::vector<float> y(r.data.size());
        conv2d(x.data.data(), w.data.data(), b.data.data(), y.data(), 6, 9, 8, 6, 1, 3, 3, g);
        CHECK(max_err(y.data(), r) < 1e-5);
    }
    {   // conv1d, 2 groups, stride 2, dilation 2
        auto x = rnd_t({6, 40}, -1, 1, 12), w = rnd_t({4, 3, 5}, -1, 1, 13), b = rnd_t({4}, -1, 1, 14);
        tensor r = ops::ref::conv1d({x, w, b}, {{"stride", 2.0}, {"pad", 3.0}, {"dilation", 2.0}, {"groups", 2.0}}); std::vector<float> y(r.data.size());
        conv1d(x.data.data(), w.data.data(), b.data.data(), y.data(), 6, 40, 4, 3, 5, 2, 3, 2);
        CHECK(max_err(y.data(), r) < 1e-5);
    }
}
TEST(prims_fft) {
    using cf = std::complex<float>;
    for (std::size_t n : {1u, 2u, 8u, 64u, 1024u, 3u, 5u, 12u, 97u, 1000u}) {   // powers of two (radix 2) and other lengths (Bluestein)
        std::vector<cf> x(n); std::uint32_t seed = std::uint32_t(n) * 7919u;
        for (auto& v : x) { seed = seed * 1664525u + 1013904223u; float re = float(seed >> 8) / float(1u << 24) - 0.5f; seed = seed * 1664525u + 1013904223u; v = {re, float(seed >> 8) / float(1u << 24) - 0.5f}; }
        auto y = x; gpu::prims::fft(std::span<cf>(y));
        double worst = 0, scale = 0;
        for (std::size_t k = 0; k < n; ++k) {   // the definition, in double
            std::complex<double> s = 0;
            for (std::size_t j = 0; j < n; ++j) { const double a = -2.0 * 3.14159265358979323846 * double((j * k) % n) / double(n); s += std::complex<double>(x[j]) * std::complex<double>(std::cos(a), std::sin(a)); }
            worst = std::max(worst, std::abs(std::complex<double>(y[k]) - s)); scale = std::max(scale, std::abs(s));
        }
        CHECK_MSG(worst <= 2e-5 * std::max(1.0, scale), "fft n %zu: worst %.3g (scale %.3g)", n, worst, scale);
        gpu::prims::fft(std::span<cf>(y), true);
        double back = 0; for (std::size_t j = 0; j < n; ++j) back = std::max(back, double(std::abs(y[j] - x[j])));
        CHECK_MSG(back < 2e-5, "ifft(fft(x)) n %zu: worst %.3g", n, back);
    }
}
TEST(prims_softmax_norms) {
    using namespace gpu::prims;
    const std::size_t rows = 7, cols = 133;
    auto x = rnd_t({i64(rows), i64(cols)}, -4, 4, 15), w = rnd_t({i64(cols)}, 0.5f, 1.5f, 16), b = rnd_t({i64(cols)}, -1, 1, 17);
    std::vector<float> y(rows * cols);
    softmax(x.data.data(), y.data(), rows, cols);
    CHECK(max_err(y.data(), ops::ref::softmax({x}, {})) < 1e-6);
    rms_norm(x.data.data(), w.data.data(), y.data(), rows, cols, 1e-6f);
    CHECK(max_err(y.data(), ops::ref::rms_norm({x, w}, {{"eps", 1e-6}})) < 1e-5);
    layer_norm(x.data.data(), w.data.data(), b.data.data(), y.data(), rows, cols, 1e-5f);
    CHECK(max_err(y.data(), ops::ref::layer_norm({x, w, b}, {{"eps", 1e-5}})) < 1e-5);
    y.assign(x.data.begin(), x.data.end());   // in place
    layer_norm(y.data(), w.data.data(), b.data.data(), y.data(), rows, cols, 1e-5f);
    CHECK(max_err(y.data(), ops::ref::layer_norm({x, w, b}, {{"eps", 1e-5}})) < 1e-5);
}
TEST(prims_flash_attention) {
    using namespace gpu::prims;
    struct cfg { i64 s, t, hq, hk, d, dv; bool causal; i64 window; std::size_t tile; };
    for (auto c : {cfg{1, 1, 4, 2, 32, 32, true, 0, 64}, cfg{6, 6, 4, 2, 32, 24, true, 0, 4}, cfg{5, 300, 8, 2, 64, 64, true, 0, 64}, cfg{3, 9, 4, 4, 32, 32, false, 0, 2},
                   cfg{70, 90, 4, 1, 48, 40, true, 33, 16}, cfg{9, 9, 2, 2, 16, 16, true, 1, 64}}) {
        auto q = rnd_t({c.s, c.hq, c.d}, -1, 1, 18), k = rnd_t({c.t, c.hk, c.d}, -1, 1, 19), v = rnd_t({c.t, c.hk, c.dv}, -1, 1, 20);
        ops::attrs a{{"causal", c.causal ? 1.0 : 0.0}, {"window", double(c.window)}};
        tensor r = ops::ref::attention({q, k, v}, a); std::vector<float> y(r.data.size(), -7.f);
        attention_opts o; o.causal = c.causal; o.window = std::size_t(c.window); o.tile = c.tile;
        flash_attention(q.data.data(), k.data.data(), v.data.data(), y.data(), std::size_t(c.s), std::size_t(c.hq), std::size_t(c.d), std::size_t(c.t), std::size_t(c.hk), std::size_t(c.dv), o);
        double worst = 0; for (std::size_t i = 0; i < y.size(); ++i) worst = std::max(worst, std::fabs(double(y[i]) - r.data[i]));
        CHECK_MSG(worst < 1e-5, "flash_attention s %lld t %lld tile %zu: worst abs %.3g", (long long)c.s, (long long)c.t, c.tile, worst);
    }
}

// "pipe:auto" (gpu::plan_pipeline): shares follow throughput, are capped by device memory, overflow to the CPU, and
// only one backend's devices are used.
TEST(pipeline_auto_partition) {
    using namespace gpu;
    auto dev = [](const char* device, const char* backend, int ord, device_class cls, double gib, double gflops) {
        device_record d; d.device = device; d.backend = backend; d.ordinal = ord; d.cls = cls; d.memory_bytes = std::uint64_t(gib * double(1ull << 30)); d.gflops = gflops; return d; };
    device_record cpu; cpu.device = cpu.backend = "cpu"; cpu.cls = device_class::cpu; cpu.gflops = 5;
    std::vector<device_record> two{cpu, dev("cuda:0", "cuda", 0, device_class::discrete, 24, 100), dev("cuda:1", "cuda", 1, device_class::discrete, 8, 50)};
    auto near = [](double a, double b) { return std::fabs(a - b) < 1e-3; };
    select_request rq;
    {   // no model size: throughput alone
        auto st = plan_pipeline(two, rq);
        CHECK(st.size() == 2 && st[0].device == "cuda:0" && st[1].device == "cuda:1" && near(st[0].share, 2.0 / 3) && near(st[1].share, 1.0 / 3));
        CHECK(pipeline_string(st) == "pipe:cuda:0*0.6667,cuda:1*0.3333");
    }
    {   // 20 GiB: both shares fit (caps 1.0 and 0.36)
        rq.model_bytes = 20ull << 30; auto st = plan_pipeline(two, rq);
        CHECK(st.size() == 2 && near(st[0].share, 2.0 / 3) && near(st[1].share, 1.0 / 3));
    }
    {   // 28 GiB: the 8 GiB card is capped at 0.9 * 8 / 28 and the rest moves to the 24 GiB one
        rq.model_bytes = 28ull << 30; std::string why; auto st = plan_pipeline(two, rq, &why);
        CHECK(st.size() == 2 && near(st[1].share, 0.9 * 8 / 28) && near(st[0].share, 1 - 0.9 * 8 / 28));
        CHECK(why.find("cuda:1") != std::string::npos && why.find("limited by its memory") != std::string::npos);
    }
    {   // 40 GiB: neither can take its share; what is left goes to a last CPU stage
        rq.model_bytes = 40ull << 30; auto st = plan_pipeline(two, rq);
        CHECK(st.size() == 3 && st[2].device == "cpu" && near(st[0].share, 0.9 * 24 / 40) && near(st[1].share, 0.9 * 8 / 40) && near(st[2].share, 1 - 0.9 * 32 / 40));
        double sum = 0; for (auto& s : st) sum += s.share;
        CHECK(near(sum, 1.0));
    }
    rq.model_bytes = 0;
    {   // the same GPUs listed under a second API are not counted twice; a software device is never a stage
        auto v = two; v.push_back(dev("opencl:0", "opencl", 0, device_class::discrete, 24, 90)); v.push_back(dev("vulkan", "vulkan", 0, device_class::software, 0, 0));
        std::string why; auto st = plan_pipeline(v, rq, &why);
        CHECK(st.size() == 2 && st[0].device == "cuda:0" && st[1].device == "cuda:1");
        CHECK(why.find("opencl:0: not used") != std::string::npos && why.find("vulkan: not used, software device") != std::string::npos);
    }
    {   // an excluded device, a missing benchmark (class priors: equal shares), one device, none
        rq.exclude = {"cuda:1"}; CHECK(pipeline_string(plan_pipeline(two, rq)) == "cuda:0"); rq.exclude.clear();
        auto v = two; v[2].gflops = 0; auto st = plan_pipeline(v, rq);
        CHECK(st.size() == 2 && near(st[0].share, 0.5) && near(st[1].share, 0.5));
        CHECK(pipeline_string(plan_pipeline({cpu}, rq)) == "cpu");
        rq.model_bytes = 100ull << 30; auto one = plan_pipeline({cpu, two[1]}, rq);   // one card, model too large: the card and the CPU
        CHECK(one.size() == 2 && one[0].device == "cuda:0" && one[1].device == "cpu" && near(one[0].share, 0.9 * 24 / 100));
    }
}
