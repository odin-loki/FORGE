// CUDA / HIP runtime. `test_cuda_hip images` checks the embedded kernel images (runs everywhere);
// `test_cuda_hip cuda|hip` runs every kernel on the device against the reference (exit 77 = skip, no device).
#include <forge/graph/graph.hpp>
#include <forge/gpu/cuda.hpp>
#include <forge/graph/vulkan.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstring>
#include <format>
#include <functional>
#include <string>

using namespace forge;
namespace cuda_hip = forge::gpu::cuda_hip;

static const char* kernel_ops[] = {"neg", "abs", "sqrt", "rsqrt", "recip", "square", "exp", "exp2", "log", "log2", "tanh", "sigmoid",
    "silu", "relu", "softplus", "gelu_tanh", "add", "sub", "mul", "div", "maximum", "minimum", "silu_mul", "scale", "softmax",
    "rms_norm", "matmul_t", "rope", "gather", "slice", "attention", "qmatmul_t"};

static bool has(std::string_view hay, const std::string& needle) { return hay.find(needle) != std::string_view::npos; }

static int images() {
    int fails = 0;
    auto ptx = cuda_hip::module_images(cuda_hip::api::cuda);
    auto hip = cuda_hip::module_images(cuda_hip::api::hip);
    if (ptx.empty()) { std::puts("SKIP images: built without a GPU-capable clang"); return 77; }
    for (auto& [arch, img] : ptx) {
        const bool sm80 = arch == "ptx sm_80";   // the sm_80 image adds the int8 tensor-core prefill matmul
        bool ok = has(img, ".version") && has(img, sm80 ? ".target sm_80" : ".target sm_70");
        for (auto op : kernel_ops) if (!has(img, std::string(".entry forge_") + op + "(")) { std::printf("FAIL %s: no entry forge_%s\n", arch.c_str(), op); ok = false; }
        if (sm80) for (auto op : {"qmatmul_mma_q8_0", "qmatmul_mma_q4_k", "qmatmul_mma_q6_k", "quant_q8r"}) if (!has(img, std::string(".entry forge_") + op + "(")) { std::printf("FAIL %s: no entry forge_%s\n", arch.c_str(), op); ok = false; }
        std::printf("%s %s: %zu bytes, %zu kernels\n", ok ? "ok  " : "FAIL", arch.c_str(), img.size(), std::size(kernel_ops)); fails += !ok;
    }
    for (auto& [arch, img] : hip) {
        bool elf = img.size() > 64 && std::memcmp(img.data(), "\x7f" "ELF", 4) == 0;
        std::uint16_t machine = 0; if (elf) std::memcpy(&machine, img.data() + 18, 2);
        bool ok = elf && machine == 224;   // EM_AMDGPU
        for (auto op : kernel_ops) if (!has(img, std::string("forge_") + op + ".kd")) { std::printf("FAIL %s: no kernel descriptor forge_%s\n", arch.c_str(), op); ok = false; }
        std::printf("%s hsaco %s: %zu bytes, ELF machine %u\n", ok ? "ok  " : "FAIL", arch.c_str(), img.size(), unsigned(machine)); fails += !ok;
    }
    if (hip.empty()) std::puts("note: no HIP code objects (no lld at build time)");
    for (auto a : {cuda_hip::api::cuda, cuda_hip::api::hip}) {   // without a driver this must fail cleanly, not crash
        std::string why; auto rt = cuda_hip::runtime::open(a, &why);
        std::printf("     %s runtime: %s\n", cuda_hip::api_name(a), rt ? ("device " + rt->device_name()).c_str() : why.c_str());
    }
    std::string why; auto reg = graph::device_registry("cuda", &why);
    if (!reg && why.empty()) { std::puts("FAIL device_registry(cuda) gave no reason for the fallback"); ++fails; }
    return fails ? 1 : 0;
}

// Quantised matmul_t kept encoded on the device vs the CPU quantised kernel (graph::qweights::gemm_nt).
// mma_runs: how many prefill matmuls ran on the int8 tensor-core kernel so far (it quantises x like the MMVQ path).
template<class RT> static int qmatmul_check(graph::registry& reg, const std::string& api, const std::function<std::size_t()>& mma_runs) {
    if (!reg.device_qgemm) { std::printf("note %s: no device quantised matmul\n", api.c_str()); return 0; }
    int fails = 0, n = 0;
    for (std::string cn : {"q8_0", "q4_k", "q5_k", "q6_k"}) for (std::size_t rows : {1, 3, 6, 16, 70, 130}) {   // >= 16 rows: the tiled MMQ kernel
        const auto* c = graph::codecs::find(cn);
        const std::size_t k = 512, cols = 37, rb = c->bytes(k);
        std::vector<float> wf(cols * k), x(rows * k);
        for (std::size_t i = 0; i < wf.size(); ++i) wf[i] = std::sin(float(i) * 0.731f + float(cn.size())) * (1.f + float(i % 97) / 50.f);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::cos(float(i) * 0.217f);
        std::vector<std::uint8_t> enc(cols * rb);
        for (std::size_t j = 0; j < cols; ++j) c->encode(wf.data() + j * k, k, enc.data() + j * rb);
        tensor_view xv; xv.data = x.data(); xv.shape = {i64(rows), i64(k)};
        tensor_view wv; wv.shape = {i64(cols), i64(k)}; wv.qdata = enc.data(); wv.qcodec = c;
        std::vector<float> yg(rows * cols, -1.f), yc(rows * cols);
        tensor_view og; og.data = yg.data(); og.shape = {i64(rows), i64(cols)};
        tensor_view oc; oc.data = yc.data(); oc.shape = og.shape;
        std::vector<tensor_view> ins{xv, wv};
        const std::size_t mma0 = mma_runs();
        reg.device_qgemm->fn(ins, og, {});
        const bool mma = mma_runs() != mma0;
        // rows < 16 on q8_0/q4_k/q6_k: the MMVQ path quantises x to q8_1 (32-value blocks, d = amax/127) like llama.cpp,
        // so the reference uses the same rounded activations.
        const bool mmvq = mma || (rows < 16 && (cn == "q8_0" || cn == "q4_k" || cn == "q6_k") && !(std::getenv("FORGE_MMVQ") && std::string(std::getenv("FORGE_MMVQ")) == "0"));
        std::vector<float> xr(x);
        if (mmvq) for (std::size_t b = 0; b < xr.size(); b += 32) {
            float amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xr[b + i]));
            const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
            for (int i = 0; i < 32; ++i) xr[b + i] = d * float(int(std::nearbyint(xr[b + i] * id)));
        }
        std::vector<tensor_view> insr{xv, wv}; insr[0].data = xr.data();
        graph::qweights::gemm_nt(insr, oc, {});
        double worst = 0;
        for (std::size_t i = 0; i < yc.size(); ++i) worst = std::max(worst, std::fabs(double(yg[i]) - yc[i]) / (1.0 + std::fabs(double(yc[i]))));
        ++n; if (worst > (mmvq ? 2e-3 : 1e-4)) { std::printf("FAIL %s.qmatmul_t %s rows %zu: worst rel error %.3g\n", api.c_str(), cn.c_str(), rows, worst); ++fails; }
    }
    std::printf("%s %d/%d %s quantised matmul_t cases match the CPU kernel\n", fails ? "FAIL" : "ok  ", n - fails, n, api.c_str());
    return fails;
}

static int state_ops_check_one(graph::registry& reg, const std::string& api, const i64 dk, const i64 dv);
// Gated DeltaNet state ops on the device (run_state) vs the CPU executor: 3-token prefill, then token by token,
// then reset() and again (the resident state must restart from zeros).
static int state_ops_check(graph::registry& reg, const std::string& api) {
    if (!reg.hooks || !reg.hooks->state) { std::printf("note %s: no device state ops\n", api.c_str()); return 0; }
    int fails = 0;
    for (auto [dk, dv] : {std::pair<i64, i64>{40, 136}, {128, 128}, {64, 32}}) fails += state_ops_check_one(reg, api, dk, dv);
    return fails;
}
// One shape: dk 40 / dv 136 takes the per-head kernel (dv > 128 and C > 128: multi-column threads / blocks); dk = dv = 128
// (Qwen3.5) and dk 64 / dv 32 take the register-state kernels forge_delta_rule_p16 / _p8 (_r16 / _r8 with FORGE_DELTA_RULE_R).
static int state_ops_check_one(graph::registry& reg, const std::string& api, const i64 dk, const i64 dv) {
    using namespace graph;
    const i64 S = 7, Hk = 2, Hv = 4, C = 150, K = 4;
    auto rnd = [](std::size_t n, float ph) { std::vector<float> v(n); for (std::size_t i = 0; i < n; ++i) v[i] = std::sin(float(i) * 0.37f + ph); return v; };
    auto q = rnd(S * Hk * dk, 1), k = rnd(S * Hk * dk, 2), v = rnd(S * Hv * dv, 3), gl = rnd(S * Hv, 4), bl = rnd(S * Hv, 5), xc = rnd(S * C, 6), wc = rnd(C * K, 7);
    for (auto& x : gl) x = -std::fabs(x) * 0.5f;
    for (auto& x : bl) x = 1.f / (1.f + std::exp(-x));
    auto build = [&](i64 s) {
        tensor_graph g;
        auto iq = g.input("q", {s, Hk, dk}), ik = g.input("k", {s, Hk, dk}), iv = g.input("v", {s, Hv, dv});
        auto ig = g.input("g", {s, Hv}), ib = g.input("b", {s, Hv}), ix = g.input("x", {s, C});
        auto w = g.constant(tensor({C, K}, 0.f)); g.at(w).constant->data = wc;
        g.output("o", g.state_op("ssm", "delta_rule", {iq, ik, iv, ig, ib}));
        g.output("c", g.state_op("conv", "causal_conv", {ix, w}));
        return g;
    };
    auto run = [&](registry* r) {
        std::vector<float> out;
        state_store st; pass_options po; po.lowerings = r;
        auto p3 = compile(build(3), &st, po), p1 = compile(build(1), &st, po);
        executor e3(*p3, &st), e1(*p1, &st);
        auto feed = [&](executor& e, i64 t0, i64 s) {
            auto sl = [&](const std::vector<float>& a, i64 row) { return std::span<const float>(a.data() + t0 * row, std::size_t(s * row)); };
            e.set_input("q", sl(q, Hk * dk)); e.set_input("k", sl(k, Hk * dk)); e.set_input("v", sl(v, Hv * dv));
            e.set_input("g", sl(gl, Hv)); e.set_input("b", sl(bl, Hv)); e.set_input("x", sl(xc, C)); e.run();
            for (auto nm : {"o", "c"}) { auto t = e.output(nm); out.insert(out.end(), t.data.begin(), t.data.end()); }
        };
        for (int pass = 0; pass < 2; ++pass) { feed(e3, 0, 3); for (i64 t = 3; t < S; ++t) feed(e1, t, 1); st.reset(); }
        return out;
    };
    auto dev = run(&reg), cpu = run(nullptr);
    double worst = dev.size() == cpu.size() ? 0 : 1e9;
    for (std::size_t i = 0; i < std::min(dev.size(), cpu.size()); ++i) worst = std::max(worst, std::fabs(double(dev[i]) - cpu[i]) / (1.0 + std::fabs(double(cpu[i]))));
    bool ok = worst < 1e-4;
    std::printf("%s %s delta_rule/causal_conv (dk %lld, dv %lld) on the device match the CPU (worst rel %.3g)\n", ok ? "ok  " : "FAIL", api.c_str(),
                (long long)dk, (long long)dv, worst);
    return ok ? 0 : 1;
}

// attention against the reference: the split-K decode kernels' shapes (forge_attention_dec + _comb: head dims a
// multiple of 32 up to 256, rows x heads <= 256) and prefill shapes for the tiled kernel; GQA, causal and not.
template<class RT> static int attention_dec_check(RT& rt, const std::string& api) {
    int fails = 0, n = 0;
    struct cfg { i64 s, t, hq, hk, d; bool causal; i64 window = 0; };
    for (auto c : {cfg{1, 1, 4, 2, 32, true}, cfg{1, 37, 4, 2, 32, true}, cfg{6, 6, 4, 2, 32, true}, cfg{5, 12, 8, 2, 64, true},
                   cfg{1, 300, 16, 4, 128, true}, cfg{3, 9, 4, 4, 32, false}, cfg{16, 16, 8, 8, 32, true},
                   // prefill shapes (rows x heads > 256: the tiled forge_attention_fa2_* kernels on the sm_80 image)
                   cfg{100, 100, 8, 2, 128, true}, cfg{70, 90, 4, 4, 256, true}, cfg{40, 40, 8, 4, 64, false}, cfg{130, 130, 4, 1, 96, true},
                   cfg{50, 120, 8, 2, 128, true}, cfg{150, 150, 4, 2, 128, true, 40}, cfg{90, 90, 4, 2, 256, true, 33}}) {
        auto rnd = [](std::size_t m, float ph) { std::vector<float> v(m); for (std::size_t i = 0; i < m; ++i) v[i] = std::sin(float(i) * 0.613f + ph); return v; };
        tensor q({c.s, c.hq, c.d}), k({c.t, c.hk, c.d}), v({c.t, c.hk, c.d}), y({c.s, c.hq, c.d});
        q.data = rnd(q.data.size(), 1); k.data = rnd(k.data.size(), 2); v.data = rnd(v.data.size(), 3);
        ops::attrs a{{"causal", c.causal ? 1.0 : 0.0}, {"window", double(c.window)}};
        ++n;
        auto tc_runs = [&]() -> std::size_t { auto dd = rt.dispatches(); auto it = dd.find("attention_fa2tc"); return it == dd.end() ? 0 : it->second; };
        const std::size_t tc0 = tc_runs();
        if (!rt.run_op("attention", {{q.data.data(), q.shape}, {k.data.data(), k.shape}, {v.data.data(), v.shape}}, a, {y.data.data(), y.shape})) {
            std::printf("FAIL %s.attention s %lld t %lld d %lld: run_op refused the shape\n", api.c_str(), (long long)c.s, (long long)c.t, (long long)c.d); ++fails; continue;
        }
        auto want = ops::ref::attention({q, k, v}, a);
        double worst = 0; for (std::size_t i = 0; i < y.data.size(); ++i) worst = std::max(worst, std::fabs(double(y.data[i]) - want.data[i]));
        // fp16 tensor-core kernel (forge_attention_fa2tc_*): Q, K, V and P are rounded to fp16 (relative 2^-11) before
        // the f32-accumulated products; outputs here carry ~4e-4 absolute error (bound 2e-3).
        const bool tc = tc_runs() > tc0; const double tol = tc ? 2e-3 : 1e-4;
        if (!(worst < tol)) { std::printf("FAIL %s.attention s %lld t %lld hq %lld hk %lld d %lld: worst abs error %.3g\n", api.c_str(),
                                                (long long)c.s, (long long)c.t, (long long)c.hq, (long long)c.hk, (long long)c.d, worst); ++fails; }
        else if (tc) std::printf("ok   %s.attention s %lld t %lld d %lld on fp16 tensor cores: worst abs error %.3g\n", api.c_str(),
                                 (long long)c.s, (long long)c.t, (long long)c.d, worst);
    }
    std::printf("%s %d/%d %s attention shapes match the reference\n", fails ? "FAIL" : "ok  ", n - fails, n, api.c_str());
    return fails;
}

// Every runtime with the cuda.hpp contract: open(why, ordinal) via the Open functor.
template<class RT, class Open> static int device_t(Open open) {
    std::string why; auto rt = open(&why, 0);
    if (!rt) { std::printf("SKIP: %s\n", why.c_str()); return 77; }
    const std::string api = rt->backend();
    std::printf("%s device: %s (image %s)\n", api.c_str(), rt->device_name().c_str(), rt->image_arch().c_str());
    int dec_fails = 0;
    if constexpr (requires { rt->run_op("attention", {}, ops::attrs{}, tensor_view{}); })
        if (api == "cuda" || api == "hip") dec_fails = attention_dec_check(*rt, api);
    auto shared = std::shared_ptr<RT>(std::move(rt));
    auto reg = graph::gpu_registry(shared);
    auto mma_runs = [&]() -> std::size_t { auto d = shared->dispatches(); auto it = d.find("qmatmul_mma"); return it == d.end() ? 0 : it->second; };
    int fails = 0, n = 0;
    for (std::size_t i = 0; i < ops::op_count; ++i) for (auto& l : reg->lowerings(ops::op(i))) {
        if (l.name.rfind(api + ".", 0) != 0) continue;
        std::string msg; ++n;
        if (!graph::registry::oracle_check(l, &msg)) { std::printf("FAIL %s: %s\n", l.name.c_str(), msg.c_str()); ++fails; }
    }
    std::printf("%s %d/%d %s lowerings pass the oracle\n", fails ? "FAIL" : "ok  ", n - fails, n, api.c_str());
    fails += dec_fails;
    fails += qmatmul_check<RT>(*reg, api, mma_runs);
    fails += state_ops_check(*reg, api);
    return fails ? 1 : 0;
}

// Streams, residency, graphs, multi-device and peer copies (runs on hardware or the mock drivers).
template<class Open, class Count> static int features_t(Open open, Count device_count) {
    std::string why; auto rt = open(&why, 0);
    if (!rt) { std::printf("SKIP: %s\n", why.c_str()); return 77; }
    int fails = 0;
    auto chk = [&](bool c, const char* what) { std::printf("%s %s\n", c ? "ok  " : "FAIL", what); fails += !c; };
    std::printf("%s: %d device(s), streams %d, graphs %d\n", rt->backend(), device_count(), rt->has_streams(), rt->has_graphs());
    // residency: a chain of ops inside one run touches the host only at the end
    std::vector<float> x(1000), y(1000), z(1000);
    for (int i = 0; i < 1000; ++i) x[std::size_t(i)] = float(i % 17) * 0.25f - 2;
    shape_t sh{1000};
    rt->begin_run();
    rt->run_op("exp", {{x.data(), sh}}, {}, {y.data(), sh});
    rt->run_op("mul", {{y.data(), sh}, {x.data(), sh}}, {}, {z.data(), sh});
    auto st = rt->residency();
    rt->end_run();
    bool ok = true; for (int i = 0; i < 1000; ++i) ok &= std::abs(z[std::size_t(i)] - std::exp(x[std::size_t(i)]) * x[std::size_t(i)]) <= 1e-5f * (1 + std::abs(z[std::size_t(i)]));
    chk(ok, "resident exp*x chain matches the host");
    chk(st.uploads == 1 && st.downloads == 0, "residency: one upload, no downloads mid-run");
    chk(rt->residency().resident == 0, "end_run frees unpinned buffers");
    // device-side copy into a slot (KV append path)
    std::vector<float> slot(64, 0.f), row(16, 3.f);
    rt->begin_run(); rt->copy_resident(row.data(), 16, slot.data(), 64, 16); rt->to_host(slot.data(), 64);
    chk(slot[16] == 3.f && slot[31] == 3.f && slot[15] == 0.f && slot[32] == 0.f, "copy_resident writes the slot on the device"); rt->end_run();
    // greedy argmax on a device-newer buffer (multi-block over a vocabulary-sized row; ties: the lowest index), twice
    // per size so the blocks' ticket counter is seen to reset
    if constexpr (requires { rt->argmax_resident((const float*)nullptr, std::size_t(0)); }) {
        bool am_ok = true;
        for (std::size_t n : {std::size_t(1000), std::size_t(248320), std::size_t(3)})
            for (std::size_t at : {n - 1, n / 3}) {
                std::vector<float> lv(n), lo(n);
                for (std::size_t i = 0; i < n; ++i) lv[i] = float((i * 7919) % 1000) * 1e-3f;
                lv[at] = 5.f; if (at + 1 < n) lv[n - 1] = 5.f;   // a tie after `at`
                shape_t ls{i64(n)};
                rt->begin_run(); rt->run_op("scale", {{lv.data(), ls}}, {{"alpha", 1.0}}, {lo.data(), ls});
                const long long r = rt->argmax_resident(lo.data(), n); rt->end_run();
                if (r != (long long)at) { am_ok = false; std::printf("     argmax n %zu: %lld, want %zu\n", n, r, at); }
            }
        chk(am_ok, "argmax_resident finds the first largest value");
    }
    // graphs
    if (rt->has_graphs()) {
        // warm run makes the buffers resident, then capture the same op and replay it on new input
        rt->begin_run();
        rt->run_op("exp", {{x.data(), sh}}, {}, {y.data(), sh});
        rt->capture_begin();
        rt->run_op("scale", {{y.data(), sh}}, {{"alpha", 2.0}}, {z.data(), sh});
        auto g = rt->capture_end();
        std::fill(y.begin(), y.end(), 1.5f); rt->host_wrote(y.data(), y.size());
        rt->graph_launch(g); rt->to_host(z.data(), z.size());
        chk(z[0] == 3.f && z[999] == 3.f, "captured graph replays on updated input");
        rt->end_run();
    } else std::puts("note: driver has no graph API");
    // multi-device + peer copy
    if (device_count() > 1) {
        auto r1 = open(&why, 1);
        chk(r1 != nullptr, "open device 1");
        if (r1) {
            auto s0 = rt->device_alloc(4000), s1 = r1->device_alloc(4000);
            rt->upload(s0, x.data(), 4000); rt->copy_to_peer(*r1, s1, s0, 4000);
            std::vector<float> back(1000); r1->download(back.data(), s1, 4000);
            chk(back == x, "peer copy device 0 -> 1");
            rt->device_free(s0); r1->device_free(s1);
        }
    }
    chk(!open(&why, 99) && !why.empty(), "an out-of-range ordinal fails with a reason");
    auto m = rt->memory(); std::printf("     memory: %zu / %zu bytes free\n", m.free, m.total);
    return fails ? 1 : 0;
}

static void set_env(const char* k, const std::string& v) {
#ifdef _WIN32
    _putenv_s(k, v.c_str());
#else
    if (v.empty()) unsetenv(k); else setenv(k, v.c_str(), 1);
#endif
}

// Device memory accounting (gpu/device_memory.hpp, docs/memory.md): every allocation is charged to the pool
// "<backend>:0"; resident buffers are evicted least recently used first (synced to the host) when an allocation
// would cross the pool's limit or the driver runs out; an op that cannot fit even then fails with out_of_memory.
template<class Open> static int memory_t(Open open) {
    std::string why; auto rt = open(&why, 0);
    if (!rt) { std::printf("SKIP: %s\n", why.c_str()); return 77; }
    int fails = 0;
    auto chk = [&](bool c, const std::string& what) { std::printf("%s %s\n", c ? "ok  " : "FAIL", what.c_str()); fails += !c; };
    auto* pool = rt->memory_pool();
    chk(pool && pool->name() == std::string(rt->backend()) + ":0", std::string("runtime charges the pool ") + (pool ? pool->name() : "?"));
    if (!pool) return 1;
    pool->set_limit(0);
    const std::size_t N = 1000, B = N * 4;
    shape_t sh{i64(N)};
    std::vector<float> w0(N), w1(N), w2(N), y(N), z(N);
    for (std::size_t i = 0; i < N; ++i) { w0[i] = float(i % 13) * 0.5f - 3; w1[i] = float(i % 7) + 1; w2[i] = float(i % 5) - 2; }
    for (auto* w : {&w0, &w1, &w2}) rt->pin(w->data(), N);   // weights: resident across runs
    auto same = [&](const std::vector<float>& v, auto f) { for (std::size_t i = 0; i < N; ++i) if (std::abs(v[i] - f(i)) > 1e-5f) return false; return true; };
    auto pooled = [&] { if constexpr (requires { rt->residency().pooled_bytes; }) return std::uint64_t(rt->residency().pooled_bytes); else return std::uint64_t(0); };
    auto base = pool->used();

    // 1. accounting: a resident op charges its buffers; after the run the pinned weights (and pooled buffers) stay charged
    rt->begin_run();
    rt->run_op("add", {{w0.data(), sh}, {w1.data(), sh}}, {}, {y.data(), sh});
    chk(pool->used() - base == 3 * B && rt->charged_bytes() == 3 * B, std::format("add(w0, w1) charges 3 x {} bytes (used {})", B, pool->used() - base));
    chk(pool->stats().by_tag["resident"] >= 3 * B, "charged under the tag \"resident\"");
    rt->end_run();
    chk(same(y, [&](std::size_t i) { return w0[i] + w1[i]; }), "add result");
    chk(pool->used() - base == 2 * B + pooled(), std::format("after end_run: 2 pinned weights + {} pooled bytes stay charged (used {})", pooled(), pool->used() - base));

    // 2. a limit of 3 buffers: the next op needs 4, so the least recently used weight not bound by the op goes
    pool->set_limit(base + 3 * B);
    auto ev0 = rt->residency().evictions, up0 = rt->residency().uploads;
    rt->begin_run();
    rt->run_op("mul", {{w2.data(), sh}, {w0.data(), sh}}, {}, {z.data(), sh});
    rt->end_run();
    chk(same(z, [&](std::size_t i) { return w2[i] * w0[i]; }), "mul result under a 3-buffer budget");
    chk(rt->residency().evictions > ev0 && pool->stats().evictions > 0, std::format("an allocation over the budget evicted ({} evictions)", rt->residency().evictions - ev0));
    chk(pool->stats().peak <= base + 3 * B && pool->used() <= base + 3 * B, std::format("usage never crossed the limit (peak {})", pool->stats().peak - base));
    // w1 was evicted (LRU, not bound by mul): using it again uploads it; w0 stayed resident
    up0 = rt->residency().uploads;
    rt->begin_run(); rt->run_op("scale", {{w0.data(), sh}}, {{"alpha", 2.0}}, {y.data(), sh}); rt->end_run();
    chk(rt->residency().uploads == up0 && same(y, [&](std::size_t i) { return 2 * w0[i]; }), "w0 (used most recently) stayed resident");
    rt->begin_run(); rt->run_op("scale", {{w1.data(), sh}}, {{"alpha", 2.0}}, {y.data(), sh}); rt->end_run();
    chk(rt->residency().uploads == up0 + 1 && same(y, [&](std::size_t i) { return 2 * w1[i]; }), "w1 (least recently used) was evicted and is uploaded again");

    // 3. eviction requested from outside (pressure response): least recently used first
    ev0 = rt->residency().evictions;
    pool->evict(pooled() + 1);   // the pool's free buffers, then one resident weight: w0 (used before w1)
    up0 = rt->residency().uploads;
    rt->begin_run(); rt->run_op("scale", {{w1.data(), sh}}, {{"alpha", 3.0}}, {y.data(), sh}); rt->end_run();
    chk(rt->residency().evictions > ev0 && rt->residency().uploads == up0, "budget::evict drops the LRU weight and keeps the recent one");

    // 4. an op that cannot fit even after evicting everything fails with out_of_memory, and leaves the runtime usable
    pool->set_limit(base + 2 * B);
    auto den = pool->stats().denials; bool oom = false;
    rt->begin_run();
    try { rt->run_op("add", {{w0.data(), sh}, {w1.data(), sh}}, {}, {y.data(), sh}); }
    catch (const forge_exception& e) { oom = e.code == errc::out_of_memory; std::printf("     %s\n", e.what()); }
    rt->end_run();
    chk(oom && pool->stats().denials > den, "an op larger than the budget fails with errc::out_of_memory");
    pool->set_limit(0);
    rt->begin_run(); rt->run_op("add", {{w0.data(), sh}, {w1.data(), sh}}, {}, {y.data(), sh}); rt->end_run();
    chk(same(y, [&](std::size_t i) { return w0[i] + w1[i]; }), "the runtime recovers once the limit is lifted");

#if !defined(_WIN32)   // the mock reads MOCK_MEM_LIMIT through its own C runtime
    // 5. (mock driver) the driver runs out with no budget set: evict resident buffers, including an unsynced output, and retry
    if (rt->device_name().find("forge mock") != std::string::npos && (std::string(rt->backend()) == "cuda" || std::string(rt->backend()) == "hip")) {
        pool->evict(0);
        const std::size_t M = 1500;   // a size the pool has no buffer for
        shape_t sm{i64(M)};
        std::vector<float> a(M), b(M), c(M), d(M);
        for (std::size_t i = 0; i < M; ++i) { a[i] = float(i % 11) * 0.1f; c[i] = float(i % 3) * 0.2f; }
        auto m = rt->memory(); std::size_t live = m.total - m.free;
        set_env("MOCK_MEM_LIMIT", std::to_string(live + 2 * M * 4));
        ev0 = rt->residency().evictions;
        rt->begin_run();
        rt->run_op("exp", {{a.data(), sm}}, {}, {b.data(), sm});   // a, b: the driver is now full
        rt->run_op("exp", {{c.data(), sm}}, {}, {d.data(), sm});   // c evicts a, d evicts b (downloaded first)
        rt->end_run();
        set_env("MOCK_MEM_LIMIT", "");
        bool ok = true; for (std::size_t i = 0; i < M; ++i) ok &= std::abs(b[i] - std::exp(a[i])) < 1e-5f && std::abs(d[i] - std::exp(c[i])) < 1e-5f;
        chk(ok && rt->residency().evictions >= ev0 + 2, std::format("driver out of memory: {} evictions, results intact", rt->residency().evictions - ev0));
    }
#endif
    rt.reset();
    chk(pool->used() == base, std::format("destroying the runtime returns every byte (used {})", pool->used() - base));
    return fails ? 1 : 0;
}

// test_cuda_hip bench <codec> <rows> <cols> <k> [reps]: time one quantised matmul on the device (x and y stay resident
// inside a run, so only the kernels are timed). Not part of ctest; for kernel work (e.g. q4_k 512 25600 5120).
static int bench_qmm(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: test_cuda_hip bench <codec> <rows> <cols> <k> [reps]\n"); return 2; }
    std::string why; auto rt = cuda_hip::runtime::open(cuda_hip::api::cuda, &why, 0);
    if (!rt) { std::printf("SKIP: %s\n", why.c_str()); return 77; }
    const auto* c = graph::codecs::find(argv[2]);
    if (!c) { std::fprintf(stderr, "unknown codec %s\n", argv[2]); return 2; }
    const std::size_t rows = std::stoul(argv[3]), cols = std::stoul(argv[4]), k = std::stoul(argv[5]), rb = c->bytes(k);
    const int reps = argc > 6 ? std::stoi(argv[6]) : 20;
    std::vector<float> wrow(k), x(rows * k), y(rows * cols);
    std::vector<std::uint8_t> enc(cols * rb);
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < k; ++i) wrow[i] = std::sin(float(j * k + i) * 0.731f) * 0.05f;
        c->encode(wrow.data(), k, enc.data() + j * rb);
    }
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::cos(float(i) * 0.217f);
    std::uint32_t fmt = c->name == "q8_0" ? 1 : c->name == "q4_k" ? 2 : c->name == "q6_k" ? 3 : c->name == "q5_k" ? 4 : 0;
    tensor_view xv{x.data(), {i64(rows), i64(k)}}, yv{y.data(), {i64(rows), i64(cols)}};
    rt->begin_run();
    rt->run_qmatmul(xv, enc.data(), rb, std::uint32_t(cols), fmt, yv);   // upload + warm up
    rt->wait();
    // batches of `reps` back-to-back calls, one wait each; the fastest batch is kept (the GPU may be shared, and
    // the minimum best approximates an idle device). Per-call time = batch time / reps.
    std::vector<double> ts;
    for (int b = 0; b < 7; ++b) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) rt->run_qmatmul(xv, enc.data(), rb, std::uint32_t(cols), fmt, yv);
        rt->wait();
        ts.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps);
    }
    std::sort(ts.begin(), ts.end()); const double us = ts.front(), med = ts[ts.size() / 2];
    rt->end_run();
    auto d = rt->dispatches(); std::string path;
    for (const char* p : {"qmatmul_mma", "qmatmul_mmq", "qmatmul_q81"}) if (d.count(p)) path = p;
    std::printf("%s rows %zu cols %zu k %zu: min %.1f us (median %.1f), %.1f TOPS, %.0f GB/s weights (%s)\n", argv[2], rows, cols, k, us, med,
                2.0 * double(rows) * double(cols) * double(k) / us / 1e6, double(enc.size()) / us / 1e3, path.c_str());
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "images";
    if (mode == "images") return images();
    if (mode == "bench") return bench_qmm(argc, argv);
    auto cu = [](cuda_hip::api a) { return [a](std::string* w, int o) { return cuda_hip::runtime::open(a, w, o); }; };
    auto cl = [](std::string* w, int o) { return forge::gpu::ocl::runtime::open(w, o); };
    auto ze = [](std::string* w, int o) { return forge::gpu::ze::runtime::open(w, o); };
    if (mode == "cuda") return device_t<cuda_hip::runtime>(cu(cuda_hip::api::cuda));
    if (mode == "hip") return device_t<cuda_hip::runtime>(cu(cuda_hip::api::hip));
    if (mode == "opencl") return device_t<forge::gpu::ocl::runtime>(cl);
    if (mode == "ze") return device_t<forge::gpu::ze::runtime>(ze);
    if (mode == "metal") return device_t<forge::gpu::metal::runtime>([](std::string* w, int o) { return forge::gpu::metal::runtime::open(w, o); });
    if (mode == "features") {
        std::string w = argc > 2 ? argv[2] : "cuda";
        if (w == "opencl") return features_t(cl, [] { return forge::gpu::ocl::runtime::device_count(); });
        if (w == "ze") return features_t(ze, [] { return forge::gpu::ze::runtime::device_count(); });
        auto a = w == "hip" ? cuda_hip::api::hip : cuda_hip::api::cuda;
        return features_t(cu(a), [a] { return cuda_hip::runtime::device_count(a); });
    }
    if (mode == "memory") {
        std::string w = argc > 2 ? argv[2] : "cuda";
        if (w == "opencl") return memory_t(cl);
        if (w == "ze") return memory_t(ze);
        return memory_t(cu(w == "hip" ? cuda_hip::api::hip : cuda_hip::api::cuda));
    }
    std::fprintf(stderr, "usage: test_cuda_hip [images|cuda|hip|opencl|ze|features <api>]\n"); return 2;
}
