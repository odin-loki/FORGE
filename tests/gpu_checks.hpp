// Device checks shared by tests/test_cuda_hip.cpp (CUDA, HIP, OpenCL, Level Zero, Metal: hardware or the mock
// drivers) and tests/test_vulkan.cpp (lavapipe / a real Vulkan device): the kernel families beyond the per-op
// lowerings, each against the CPU path or the op reference. They take the backend's graph::registry (and the runtime
// where a check drives it directly), so every backend is held to the same references and bounds.
#pragma once
#include <forge/graph/graph.hpp>
#include <forge/graph/vulkan.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace forge_test::gpu_checks {
using namespace forge;

inline void set_env(const char* k, const std::string& v) {
#ifdef _WIN32
    _putenv_s(k, v.c_str());
#else
    if (v.empty()) ::unsetenv(k); else ::setenv(k, v.c_str(), 1);
#endif
}

// Quantised matmul_t kept encoded on the device vs the CPU quantised kernel (graph::qweights::gemm_nt).
// mma_runs: how many prefill matmuls ran on the int8 tensor-core kernel so far (it quantises x like the MMVQ path).
template<class RT> int qmatmul_check(graph::registry& reg, const std::string& api, const std::function<std::size_t()>& mma_runs) {
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
        bool mmvq = mma || (rows < 16 && (cn == "q8_0" || cn == "q4_k" || cn == "q6_k") && !(std::getenv("FORGE_MMVQ") && std::string(std::getenv("FORGE_MMVQ")) == "0"));
        if constexpr (requires { RT::exact_activations; }) mmvq = false;   // the portable kernel (OpenCL, Level Zero) dots the f32 activations
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

inline int state_ops_check_one(graph::registry& reg, const std::string& api, const i64 dk, const i64 dv);
// Gated DeltaNet state ops on the device (run_state) vs the CPU executor: 3-token prefill, then token by token,
// then reset() and again (the resident state must restart from zeros).
inline int state_ops_check(graph::registry& reg, const std::string& api) {
    if (!reg.hooks || !reg.hooks->state) { std::printf("note %s: no device state ops\n", api.c_str()); return 0; }
    int fails = 0;
    for (auto [dk, dv] : {std::pair<i64, i64>{40, 136}, {128, 128}, {64, 32}}) fails += state_ops_check_one(reg, api, dk, dv);
    return fails;
}

// One shape: dk 40 / dv 136 takes the per-head kernel (dv > 128 and C > 128: multi-column threads / blocks); dk = dv = 128
// (Qwen3.5) and dk 64 / dv 32 take the register-state kernels forge_delta_rule_p16 / _p8 (_r16 / _r8 with FORGE_DELTA_RULE_R).
inline int state_ops_check_one(graph::registry& reg, const std::string& api, const i64 dk, const i64 dv) {
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
        // the state slots are pinned on the device by host address: drop the device copies before the store is freed
        if (r && r->hooks) { st.hooks = r->hooks.get(); st.forget(); }
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
template<class RT> int attention_dec_check(RT& rt, const std::string& api) {
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

// FFN gate/up/SiLU fusion (plan.hpp fuse_glu + run_qglu): y = (silu(x Wg^T) * (x Wu^T)) Wd^T as a compiled plan on the
// device, fused vs compiled with FORGE_FUSE_GLU=0 vs the CPU executor. Rows < 16: decode (forge_qmatmul_q81 over the
// interleaved weight + forge_glu_q81, whose q8_1 the down projection reuses); >= 16: forge_qmatmul_glu_* (tensor cores),
// whose q8r output the down projection reuses. q5_k has no fused kernel: the plan must stay unfused.
template<class RT> int glu_check(graph::registry& reg, RT& rt, const std::string& api) {
    if (!reg.device_qglu) { std::printf("note %s: no fused FFN gate/up\n", api.c_str()); return 0; }
    using namespace graph;
    int fails = 0, n = 0;
    // static: device weights are cached by host address, and glu_check runs more than once per process
    static std::deque<std::vector<std::uint8_t>> keep;
    auto disp = [&](const char* key) -> std::size_t { auto d = rt.dispatches(); auto it = d.find(key); return it == d.end() ? 0 : it->second; };
    for (std::string cn : {"q8_0", "q4_k", "q6_k", "q5_k"}) for (i64 rows : {1, 3, 16, 70, 130}) {
        const auto* c = codecs::find(cn);
        const i64 k = 512, ff = 256, od = 40;
        auto enc = [&](i64 r, i64 kk, float ph) {
            std::vector<float> w(std::size_t(r * kk)); std::vector<std::uint8_t> e(std::size_t(r) * c->bytes(std::size_t(kk)));
            for (std::size_t i = 0; i < w.size(); ++i) w[i] = std::sin(float(i) * 0.731f + ph) * (0.05f + float(i % 89) / 900.f);
            for (i64 j = 0; j < r; ++j) c->encode(w.data() + j * kk, std::size_t(kk), e.data() + std::size_t(j) * c->bytes(std::size_t(kk)));
            return e; };
        // kept alive for the whole check: device weights are cached by host address, so a freed buffer's address must
        // not come back with other contents
        auto& eg = keep.emplace_back(enc(ff, k, 1.f)); auto& eu = keep.emplace_back(enc(ff, k, 2.f)); auto& ed = keep.emplace_back(enc(od, ff, 3.f));
        std::vector<float> x(std::size_t(rows * k)); for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::cos(float(i) * 0.217f);
        auto build = [&] {
            tensor_graph g; auto xi = g.input("x", {rows, k});
            auto gt = g.apply(ops::op::matmul_t, {xi, g.qweight("wg", {ff, k}, eg.data(), cn)});
            auto up = g.apply(ops::op::matmul_t, {xi, g.qweight("wu", {ff, k}, eu.data(), cn)});
            g.output("y", g.apply(ops::op::matmul_t, {g.mul(g.apply(ops::op::silu, {gt}), up), g.qweight("wd", {od, ff}, ed.data(), cn)}));
            return g; };
        int fused_steps = 0;
        auto run = [&](registry* r) {
            pass_options po; po.lowerings = r; auto p = compile(build(), nullptr, po);
            for (auto& s : p->steps) if (s.low && s.low->name.ends_with(".qglu")) ++fused_steps;
            executor e(*p); e.set_input("x", x); e.run(); return e.output("y").data; };
        run(&reg);   // warm-up: the first run uploads the weights (an upload drops the quantised-activation reuse)
        fused_steps = 0;
        const std::size_t glu0 = disp("qmatmul_glu"), re0 = disp("quant_q8r_reused") + disp("quant_q81_reused"), pg0 = disp("qglu");
        auto fused = run(&reg);
        const int fs = fused_steps;
        const bool took = disp("qmatmul_glu") > glu0, reused = disp("quant_q8r_reused") + disp("quant_q81_reused") > re0;
        set_env("FORGE_FUSE_GLU", "0"); fused_steps = 0;
        auto plain = run(&reg);
        set_env("FORGE_FUSE_GLU", "");
        const int ps = fused_steps;
        auto cpu = run(nullptr);
        double mx = 0, dp = 0, dc = 0;
        for (auto v : cpu) mx = std::max(mx, double(std::fabs(v)));
        for (std::size_t i = 0; i < cpu.size() && fused.size() == cpu.size() && plain.size() == cpu.size(); ++i) {
            dp = std::max(dp, std::fabs(double(fused[i]) - plain[i])); dc = std::max(dc, std::fabs(double(fused[i]) - cpu[i])); }
        const bool want = std::find(reg.device_qglu_codecs.begin(), reg.device_qglu_codecs.end(), cn) != reg.device_qglu_codecs.end();
        bool ok = fused.size() == cpu.size() && dp <= 2e-3 * (1 + mx) && dc <= 2e-2 * (1 + mx) && ps == 0 && fs == (want ? 1 : 0);
        // where the runtime takes it (sm_80 image / mock), the fused kernel ran and the down projection reused its quantised output
        if (want && took && !reused) ok = false;
        if constexpr (requires { RT::qglu_any_shape; }) { if (want && disp("qglu") == pg0) ok = false; }   // the portable forge_qglu must have run
        ++n;
        if (!ok) { std::printf("FAIL %s.qglu %s rows %lld: fused steps %d/%d, ran %d, reused %d, |fused-unfused| %.3g, |fused-cpu| %.3g (max %.3g)\n", api.c_str(),
                               cn.c_str(), (long long)rows, fs, ps, int(took), int(reused), dp, dc, mx); ++fails; }
    }
    std::printf("%s %d/%d %s fused FFN gate/up cases match the unfused plan and the CPU\n", fails ? "FAIL" : "ok  ", n - fails, n, api.c_str());
    return fails;
}

// Embedding gather from an encoded table kept on the device (registry::device_qgather) vs the codec's own decoder:
// in-range rows must match the decoded row, an index past the table gives a zero row (the kernel's contract).
inline int qgather_check(graph::registry& reg, const std::string& api) {
    if (!reg.device_qgather) { std::printf("note %s: no device quantised gather\n", api.c_str()); return 0; }
    int fails = 0, n = 0;
    static std::deque<std::vector<std::uint8_t>> keep;   // device tables are cached by host address
    for (std::string cn : {"q8_0", "q4_k", "q5_k", "q6_k"}) {
        const auto* c = graph::codecs::find(cn);
        const std::size_t d = 512, nrows = 19, rb = c->bytes(d);
        std::vector<float> tf(nrows * d);
        for (std::size_t i = 0; i < tf.size(); ++i) tf[i] = std::sin(float(i) * 0.413f + float(cn.size())) * (0.5f + float(i % 61) / 40.f);
        auto& enc = keep.emplace_back(nrows * rb);
        for (std::size_t r = 0; r < nrows; ++r) c->encode(tf.data() + r * d, d, enc.data() + r * rb);
        std::vector<float> idx{3, 0, 18, 7, 7, 25};   // 25: out of range
        std::vector<float> y(idx.size() * d, -1.f), row(d);
        tensor_view tv; tv.shape = {i64(nrows), i64(d)}; tv.qdata = enc.data(); tv.qcodec = c;
        tensor_view iv; iv.data = idx.data(); iv.shape = {i64(idx.size())};
        tensor_view ov; ov.data = y.data(); ov.shape = {i64(idx.size()), i64(d)};
        std::vector<tensor_view> ins{tv, iv};
        reg.device_qgather->fn(ins, ov, {});
        double worst = 0;
        for (std::size_t i = 0; i < idx.size(); ++i) {
            if (std::size_t(idx[i]) < nrows) c->decode(enc.data() + std::size_t(idx[i]) * rb, d, row.data()); else std::fill(row.begin(), row.end(), 0.f);
            for (std::size_t e = 0; e < d; ++e) worst = std::max(worst, std::fabs(double(y[i * d + e]) - row[e]) / (1.0 + std::fabs(double(row[e]))));
        }
        ++n; if (!(worst <= 1e-6)) { std::printf("FAIL %s.qgather %s: worst rel error %.3g\n", api.c_str(), cn.c_str(), worst); ++fails; }
    }
    std::printf("%s %d/%d %s quantised gather cases match the codec decoder\n", fails ? "FAIL" : "ok  ", n - fails, n, api.c_str());
    return fails;
}

// Sparse mixture of experts on the device (registry::device_moe, run_moe: host router + forge_moe_hidden / forge_moe_down
// over the stacked expert weights kept on the device) vs the CPU moe-sparse path (plan.hpp moe_ffn) through compiled
// plans, for encoded and f32 expert weights, softmax and sigmoid gating, prefill and single-token shapes; and the
// lowering called directly against ops::ref::moe_ffn. The plan must have picked the device lowering.
inline int moe_check(graph::registry& reg, const std::string& api, const std::function<std::size_t(const char*)>& disp) {
    if (!reg.device_moe) { std::printf("note %s: no device mixture of experts\n", api.c_str()); return 0; }
    using namespace graph;
    int fails = 0, n = 0;
    static std::deque<std::vector<std::uint8_t>> keep;
    static std::deque<tensor> keepf;
    for (std::string cn : {"f32", "q8_0", "q4_k", "q5_k", "q6_k"}) for (i64 t : {1, 5}) for (int gating : {0, 1}) {
        const i64 d = 256, f = 512, ne = 6, k = cn == "q4_k" ? 3 : 2;
        auto wf = [&](i64 rows, i64 kd, float ph) { tensor w({ne, rows, kd});
            for (std::size_t i = 0; i < w.data.size(); ++i) w.data[i] = std::sin(float(i) * 0.731f + ph) * (0.02f + float(i % 89) / 2000.f); return w; };
        auto& G = keepf.emplace_back(wf(f, d, 1.f)); auto& U = keepf.emplace_back(wf(f, d, 2.f)); auto& D = keepf.emplace_back(wf(d, f, 3.f));
        std::vector<float> h(std::size_t(t * d)), lg(std::size_t(t * ne));
        for (std::size_t i = 0; i < h.size(); ++i) h[i] = std::cos(float(i) * 0.217f);
        for (std::size_t i = 0; i < lg.size(); ++i) lg[i] = std::sin(float(i) * 1.37f + float(t));
        ops::attrs a{{"k", double(k)}, {"norm", gating ? 0.0 : 1.0}, {"scale", 1.0}, {"gating", double(gating)}};
        auto build = [&] {
            tensor_graph g; auto hi = g.input("h", {t, d}); auto li = g.input("lg", {t, ne});
            auto wt = [&](const char* nm, tensor& w) {
                if (cn == "f32") { auto id = g.constant(tensor(w.shape, 0.f)); g.at(id).constant->data = w.data; return id; }
                const auto* c = codecs::find(cn); const std::size_t kd = std::size_t(w.shape[2]), rows = std::size_t(w.shape[0] * w.shape[1]), rb = c->bytes(kd);
                auto& e = keep.emplace_back(rows * rb);
                for (std::size_t r = 0; r < rows; ++r) c->encode(w.data.data() + r * kd, kd, e.data() + r * rb);
                return g.qweight(nm, w.shape, e.data(), cn); };
            auto g1 = wt("wg", G), u1 = wt("wu", U), d1 = wt("wd", D);
            // an op before and after, so the experts' input and output live on the device inside the run
            auto hh = g.apply(ops::op::scale, {hi}, {{"alpha", 0.5}});
            g.output("y", g.apply(ops::op::neg, {g.apply(ops::op::moe_ffn, {hh, li, g1, u1, d1}, a)}));
            return g; };
        int dev_steps = 0;
        auto run = [&](registry* r) {
            pass_options po; po.lowerings = r; auto p = compile(build(), nullptr, po);
            for (auto& s : p->steps) if (s.low && s.low->name == api + ".moe_ffn") ++dev_steps;
            executor e(*p); e.set_input("h", h); e.set_input("lg", lg); e.run(); auto y1 = e.output("y").data;
            e.set_input("h", h); e.set_input("lg", lg); e.run(); auto y2 = e.output("y").data;   // second run: weights already resident
            if (y1 != y2) y1.clear();
            return y1; };
        const std::size_t d0 = disp("moe_ffn");
        auto dev = run(&reg); const int ds = dev_steps;
        auto cpu = run(nullptr);
        double worst = dev.size() == cpu.size() && !dev.empty() ? 0 : 1e9, mx = 0;
        for (auto v : cpu) mx = std::max(mx, double(std::fabs(v)));
        for (std::size_t i = 0; i < std::min(dev.size(), cpu.size()); ++i) worst = std::max(worst, std::fabs(double(dev[i]) - cpu[i]));
        bool ok = worst <= 2e-4 * (1 + mx) && ds == 1 && disp("moe_ffn") == d0 + 2;
        ++n; if (!ok) { std::printf("FAIL %s.moe_ffn %s t %lld gating %d: device steps %d, dispatches %zu, worst abs %.3g (max %.3g)\n", api.c_str(), cn.c_str(),
                                    (long long)t, gating, ds, disp("moe_ffn") - d0, worst, mx); ++fails; }
    }
    {   // the lowering outside a run, f32 experts, against the op reference
        const i64 t = 3, d = 64, f = 96, ne = 5;
        auto mk = [](shape_t s, float ph) { tensor w(s); for (std::size_t i = 0; i < w.data.size(); ++i) w.data[i] = std::sin(float(i) * 0.531f + ph) * 0.1f; return w; };
        static tensor G = mk({ne, f, d}, 1), U = mk({ne, f, d}, 2), D = mk({ne, d, f}, 3);
        tensor h = mk({t, d}, 4), lg = mk({t, ne}, 5), y({t, d});
        ops::attrs a{{"k", 2.0}};
        std::vector<tensor_view> ins{{h.data.data(), h.shape}, {lg.data.data(), lg.shape}, {G.data.data(), G.shape}, {U.data.data(), U.shape}, {D.data.data(), D.shape}};
        reg.device_moe->fn(ins, {y.data.data(), y.shape}, a);
        auto want = ops::ref::moe_ffn({h, lg, G, U, D}, a);
        double worst = 0; for (std::size_t i = 0; i < y.data.size(); ++i) worst = std::max(worst, std::fabs(double(y.data[i]) - want.data[i]));
        ++n; if (!(worst < 1e-4)) { std::printf("FAIL %s.moe_ffn outside a run: worst abs error %.3g\n", api.c_str(), worst); ++fails; }
    }
    std::printf("%s %d/%d %s mixture-of-experts cases match the CPU sparse path\n", fails ? "FAIL" : "ok  ", n - fails, n, api.c_str());
    return fails;
}

// Every backend's quantised matmul_t (and fused FFN gate/up) against ONE reference: the f32 activations dotted with
// the codec-decoded weight rows in float64, i.e. the exact value of the quantised model. The inputs are shaped like
// the tensors of verify/golden/tiny_gemma.py's Gemma-2 model, where the paths were first seen to differ: 64-wide
// projections whose weights are three times the usual scale, and activations with a few large entries per row.
//   exact_activations backends (the portable forge_qmatmul_t / forge_qglu of OpenCL and Level Zero) must match the
//     reference to f32 rounding: they decode each weight group and accumulate in f32.
//   CUDA / HIP (q8_1 activations, integer dot products) and the CPU int8 kernels round the activations to 8 bits per
//     32-value block first; they are held to that rounding's bound against the same reference.
// So a q8_0 logit difference between OpenCL and the other paths is the int8 paths' activation rounding, not a
// portable-kernel defect (docs/gpu.md "Quantised matmul: exact and int8-activation paths").
template<class RT> int exact_reference_check(graph::registry& reg, const std::string& api) {
    if (!reg.device_qgemm) return 0;
    using namespace graph;
    bool exact = false;
    if constexpr (requires { RT::exact_activations; }) exact = true;
    int fails = 0, n = 0; double worst_dev = 0, worst_cpu = 0;
    static std::deque<std::vector<std::uint8_t>> keep;   // device weights are cached by host address
    std::uint32_t seed = 12345;
    auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return float(seed >> 8) / float(1u << 24) - 0.5f; };   // uniform in [-0.5, 0.5)
    auto gauss = [&] { float s = 0; for (int i = 0; i < 6; ++i) s += rnd(); return s * 1.4142f; };                // ~N(0, 1)
    for (std::string cn : {"q8_0", "q4_k", "q5_k", "q6_k"}) for (std::size_t rows : {1, 13, 70}) for (std::size_t k : {256, 512}) {
        const auto* c = codecs::find(cn);
        const std::size_t cols = 48, rb = c->bytes(k);
        std::vector<float> wf(cols * k), x(rows * k), row(k);
        for (auto& v : wf) v = gauss() * 3.f / std::sqrt(float(k));
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = gauss() * (i % 37 == 5 ? 8.f : 1.f);   // a few entries dominate each 32-value block
        auto& enc = keep.emplace_back(cols * rb);
        for (std::size_t j = 0; j < cols; ++j) c->encode(wf.data() + j * k, k, enc.data() + j * rb);
        std::vector<double> ref(rows * cols); double mx = 0;
        for (std::size_t j = 0; j < cols; ++j) {
            c->decode(enc.data() + j * rb, k, row.data());
            for (std::size_t i = 0; i < rows; ++i) { double s = 0; for (std::size_t e = 0; e < k; ++e) s += double(x[i * k + e]) * double(row[e]); ref[i * cols + j] = s; mx = std::max(mx, std::fabs(s)); }
        }
        tensor_view xv; xv.data = x.data(); xv.shape = {i64(rows), i64(k)};
        tensor_view wv; wv.shape = {i64(cols), i64(k)}; wv.qdata = enc.data(); wv.qcodec = c;
        std::vector<float> yd(rows * cols, -1.f), yc(rows * cols, -2.f);
        tensor_view od; od.data = yd.data(); od.shape = {i64(rows), i64(cols)};
        tensor_view oc; oc.data = yc.data(); oc.shape = od.shape;
        std::vector<tensor_view> ins{xv, wv};
        reg.device_qgemm->fn(ins, od, {});
        qweights::gemm_nt(ins, oc, {});
        double ed = 0, ec = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) { ed = std::max(ed, std::fabs(yd[i] - ref[i])); ec = std::max(ec, std::fabs(yc[i] - ref[i])); }
        ed /= mx; ec /= mx; worst_dev = std::max(worst_dev, ed); worst_cpu = std::max(worst_cpu, ec);
        // exact: f32 accumulation over k <= 512 terms. int8 activations: half a step of amax / 127 per value.
        const double bound = exact ? 2e-5 : 3e-2;
        ++n;
        if (!(ed <= bound)) { std::printf("FAIL %s.qmatmul_t %s rows %zu k %zu vs decoded-weight reference: %.3g of the largest output (bound %.3g)\n", api.c_str(), cn.c_str(), rows, k, ed, bound); ++fails; }
        if (!(ec <= 3e-2)) { std::printf("FAIL cpu gemm_nt %s rows %zu k %zu vs decoded-weight reference: %.3g of the largest output\n", cn.c_str(), rows, k, ec); ++fails; }
    }
    // the fused FFN gate/up/SiLU + down projection as a compiled plan (forge_qglu on the portable backends)
    if (reg.device_qglu) for (std::string cn : {"q8_0", "q4_k"}) for (i64 rows : {1, 13}) {
        const auto* c = codecs::find(cn);
        const i64 k = 256, ff = 256, od = 32;
        auto enc = [&](i64 r, i64 kk, std::vector<float>& dec) {
            std::vector<float> w(std::size_t(r * kk)); auto& e = keep.emplace_back(std::size_t(r) * c->bytes(std::size_t(kk)));
            for (auto& v : w) v = gauss() * 3.f / std::sqrt(float(kk));
            dec.resize(w.size());
            for (i64 j = 0; j < r; ++j) { c->encode(w.data() + j * kk, std::size_t(kk), e.data() + std::size_t(j) * c->bytes(std::size_t(kk)));
                                          c->decode(e.data() + std::size_t(j) * c->bytes(std::size_t(kk)), std::size_t(kk), dec.data() + j * kk); }
            return e.data(); };
        std::vector<float> dg, du, dd;
        const std::uint8_t* eg = enc(ff, k, dg); const std::uint8_t* eu = enc(ff, k, du); const std::uint8_t* ed_ = enc(od, ff, dd);
        std::vector<float> x(std::size_t(rows * k)); for (std::size_t i = 0; i < x.size(); ++i) x[i] = gauss() * (i % 37 == 5 ? 4.f : 0.5f);
        std::vector<double> ref(static_cast<std::size_t>(rows * od)), hid(static_cast<std::size_t>(ff)); double mx = 0;
        for (i64 i = 0; i < rows; ++i) {
            for (i64 f = 0; f < ff; ++f) {
                double g = 0, u = 0;
                for (i64 e = 0; e < k; ++e) { g += double(x[std::size_t(i * k + e)]) * dg[std::size_t(f * k + e)]; u += double(x[std::size_t(i * k + e)]) * du[std::size_t(f * k + e)]; }
                hid[std::size_t(f)] = g / (1.0 + std::exp(-g)) * u;
            }
            for (i64 o = 0; o < od; ++o) { double s = 0; for (i64 f = 0; f < ff; ++f) s += hid[std::size_t(f)] * dd[std::size_t(o * ff + f)]; ref[std::size_t(i * od + o)] = s; mx = std::max(mx, std::fabs(s)); }
        }
        tensor_graph g; auto xi = g.input("x", {rows, k});
        auto gt = g.apply(ops::op::matmul_t, {xi, g.qweight("wg", {ff, k}, eg, cn)});
        auto up = g.apply(ops::op::matmul_t, {xi, g.qweight("wu", {ff, k}, eu, cn)});
        g.output("y", g.apply(ops::op::matmul_t, {g.mul(g.apply(ops::op::silu, {gt}), up), g.qweight("wd", {od, ff}, ed_, cn)}));
        pass_options po; po.lowerings = &reg; auto p = compile(std::move(g), nullptr, po);
        executor ex(*p); ex.set_input("x", x); ex.run(); auto y = ex.output("y").data;
        double e = 0; for (std::size_t i = 0; i < ref.size() && y.size() == ref.size(); ++i) e = std::max(e, std::fabs(y[i] - ref[i]));
        e = y.size() == ref.size() ? e / mx : 1.0; worst_dev = std::max(worst_dev, e);
        const double bound = exact ? 1e-4 : 6e-2;   // two matmuls and a product deep
        ++n;
        if (!(e <= bound)) { std::printf("FAIL %s FFN gate/up/down %s rows %lld vs decoded-weight reference: %.3g of the largest output (bound %.3g)\n", api.c_str(), cn.c_str(), (long long)rows, e, bound); ++fails; }
    }
    std::printf("%s %d/%d %s quantised cases within the %s bound of the decoded-weight reference (worst: device %.2g, CPU f32 kernel %.2g of the largest output)\n",
                fails ? "FAIL" : "ok  ", n - fails, n, api.c_str(), exact ? "exact (f32 rounding)" : "int8-activation", worst_dev, worst_cpu);
    return fails;
}

} // namespace forge_test::gpu_checks
