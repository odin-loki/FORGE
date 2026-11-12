#pragma once
// Vulkan lowerings for the graph executor: every spec op with a generated GPU kernel gets a "vk.<op>" lowering
// (priority above the CPU ones). Like every lowering it must pass the oracle before it is selected, and any
// op/shape without a GPU kernel runs on the CPU, so a whole model runs with the GPU doing what it can.
#include <forge/graph/kernels.hpp>
#include <forge/graph/codecs.hpp>
#include <forge/gpu/vulkan.hpp>
#include <forge/gpu/cuda.hpp>
#include <forge/gpu/opencl.hpp>
#include <forge/gpu/levelzero.hpp>
#include <forge/gpu/metal.hpp>
#include <forge/graph/rpc.hpp>
#include <forge/ops/reference.hpp>
#include <forge/gpu/select.hpp>
#include <forge/gpu/kernel_cache.hpp>
#include <chrono>

namespace forge::graph {
// The lowerings over weights kept encoded on the device, shared by every runtime with the entry points of
// gpu::cuda_hip::runtime / gpu::portable_kernels<> (CUDA, HIP, OpenCL, Level Zero, Vulkan): quantised matmul_t, the
// fused FFN gate/up/SiLU step, quantised gather and the sparse mixture of experts. `pre` prefixes the lowering names.
template<class RT> void add_encoded_lowerings(registry& r, std::shared_ptr<RT> rt, const std::string& pre, bool resident) {
    // Quantised matmul_t with the weight kept encoded on the device (CUDA/HIP forge_qmatmul_t); FORGE_GPU_QMATMUL=0
    // falls back to the f32 path (decoded weights within FORGE_GPU_WEIGHT_BUDGET, the rest on the CPU).
    if constexpr (requires(RT& t, const tensor_view& v, tensor_view o) { t.run_qmatmul(v, nullptr, std::size_t(0), 0u, 0u, o); t.has_function(""); }) {
        auto* qe = std::getenv("FORGE_GPU_QMATMUL");
        if (!(qe && std::string(qe) == "0") && rt->has_function("forge_qmatmul_t")) {
            r.device_qgemm = lowering{pre + ".qmatmul_t", ops::op::matmul_t, 30, [](auto, auto&, auto&) { return true; },
                [rt](std::span<const tensor_view> in, tensor_view out, const ops::attrs&) {
                    const auto* c = static_cast<const codecs::codec*>(in[1].qcodec);
                    std::uint32_t fmt = c->name == "q8_0" ? 1 : c->name == "q4_k" ? 2 : c->name == "q6_k" ? 3 : c->name == "q5_k" ? 4 : 0;
                    std::size_t k = std::size_t(in[0].shape.back());
                    if (!rt->run_qmatmul(in[0], in[1].qdata, c->bytes(k), std::uint32_t(in[1].shape[0]), fmt, out))
                        raise(errc::unsupported, "{}: qmatmul_t rejected a {} weight", rt->backend(), c->name); },
                1, resident};
            r.device_qgemm_codecs = {"q8_0", "q4_k", "q5_k", "q6_k"};
            // FFN gate/up/SiLU in one step (CUDA run_qglu; $FORGE_FUSE_GLU=0 off): inputs (x, Wg, Wu, gate, up). When the
            // runtime does not take it, the two matmuls and silu_mul run as before, into the gate/up buffers.
            if constexpr (requires(RT& t, const tensor_view& v, tensor_view o) { t.run_qglu(v, nullptr, nullptr, std::size_t(0), 0u, 0u, o); }) {
                auto* fe = std::getenv("FORGE_FUSE_GLU");
                if (resident && !(fe && std::string(fe) == "0")) {
                    r.device_qglu = lowering{pre + ".qglu", ops::op::silu_mul, 30, [](auto, auto&, auto&) { return true; },
                        [rt](std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
                            const auto* c = static_cast<const codecs::codec*>(in[1].qcodec);
                            std::uint32_t fmt = c->name == "q8_0" ? 1 : c->name == "q4_k" ? 2 : c->name == "q6_k" ? 3 : c->name == "q5_k" ? 4 : 0;
                            std::size_t rb = c->bytes(std::size_t(in[0].shape.back()));
                            std::uint32_t n = std::uint32_t(in[1].shape[0]);
                            if (rt->run_qglu(in[0], in[1].qdata, in[2].qdata, rb, n, fmt, out)) return;
                            if (!rt->run_qmatmul(in[0], in[1].qdata, rb, n, fmt, in[3]) || !rt->run_qmatmul(in[0], in[2].qdata, rb, n, fmt, in[4]) ||
                                !rt->run_op("silu_mul", {in[3], in[4]}, a, out))
                                raise(errc::unsupported, "{}: qglu rejected a {} weight", rt->backend(), c->name); },
                        1, true};
                    r.device_qglu_codecs = {"q8_0", "q4_k", "q6_k"};
                    if constexpr (requires { RT::qglu_any_shape; }) r.device_qglu_codecs.push_back("q5_k");   // portable forge_qglu: every codec
                }
            }
            if constexpr (requires(RT& t, const tensor_view& v, tensor_view o) { t.run_qgather(v, nullptr, std::size_t(0), 0u, 0u, 0u, o); }) {
                auto* ge = std::getenv("FORGE_GPU_QGATHER");
                if (!(ge && std::string(ge) == "0") && rt->has_function("forge_qgather"))
                    r.device_qgather = lowering{pre + ".qgather", ops::op::gather, 30, [](auto, auto&, auto&) { return true; },
                        [rt](std::span<const tensor_view> in, tensor_view out, const ops::attrs&) {
                            const auto* c = static_cast<const codecs::codec*>(in[0].qcodec);
                            std::uint32_t fmt = c->name == "q8_0" ? 1 : c->name == "q4_k" ? 2 : c->name == "q6_k" ? 3 : c->name == "q5_k" ? 4 : 0;
                            std::size_t d = std::size_t(in[0].shape[1]);
                            if (!rt->run_qgather(in[1], in[0].qdata, c->bytes(d), std::uint32_t(in[0].shape[0]), std::uint32_t(d), fmt, out))
                                raise(errc::unsupported, "{}: qgather rejected a {} table", rt->backend(), c->name); },
                        1, resident};
            }
        }
    }
    // Sparse mixture of experts on the device (run_moe; $FORGE_GPU_MOE=0 off): the router's top-k runs on the host
    // (ops::ref::moe_route over the logits, synced back first), the routed experts' gate/up/down rows are evaluated on
    // the device from the stacked weights, which are uploaded once and stay resident. Encoded or f32 expert weights.
    if constexpr (requires(RT& t, const tensor_view& v, tensor_view o, const std::vector<float>& rt2, const gpu::moe_weight& w) { t.run_moe(v, rt2, 0u, w, w, w, 0u, o); }) {
        auto* me = std::getenv("FORGE_GPU_MOE");
        if (resident && !(me && std::string(me) == "0") && rt->has_function("forge_moe_hidden") && rt->has_function("forge_moe_down")) {
            r.device_moe = lowering{pre + ".moe_ffn", ops::op::moe_ffn, 30, [](auto, auto&, auto&) { return true; },
                [rt](std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
                    const auto& h = in[0]; const auto& lg = in[1];
                    const i64 t = h.shape[0], d = h.shape[1], ne = in[2].shape[0], f = in[2].shape[1];
                    const i64 k = std::min<i64>(i64(a.num("k", 2)), ne);
                    rt->to_host(lg.data, lg.size());
                    std::vector<i64> ids(static_cast<std::size_t>(k)); std::vector<float> w(static_cast<std::size_t>(k));
                    std::vector<float> route(std::size_t(t * k) * 3);
                    for (i64 row = 0; row < t; ++row) {
                        ops::ref::moe_route(lg.data + row * ne, ne, k, a.num("norm", 1) != 0, a.num("scale", 1.0), int(a.num("gating", 0)), ids.data(), w.data());
                        for (i64 j = 0; j < k; ++j) { float* p = route.data() + std::size_t(row * k + j) * 3; p[0] = float(ids[std::size_t(j)]); p[1] = float(row); p[2] = w[std::size_t(j)]; }
                    }
                    auto mw = [](const tensor_view& wv, std::size_t kd) {
                        gpu::moe_weight m; std::size_t rows = std::size_t(wv.shape[0] * wv.shape[1]);
                        if (wv.qdata) {
                            const auto* c = static_cast<const codecs::codec*>(wv.qcodec);
                            m.fmt = c->name == "q8_0" ? 1 : c->name == "q4_k" ? 2 : c->name == "q6_k" ? 3 : c->name == "q5_k" ? 4 : 99;
                            m.rb = std::uint32_t(c->bytes(kd)); m.data = wv.qdata; m.bytes = rows * m.rb;
                        } else { m.data = reinterpret_cast<const std::uint8_t*>(wv.data); m.bytes = rows * kd * 4; }
                        return m; };
                    if (!rt->run_moe(h, route, std::uint32_t(k), mw(in[2], std::size_t(d)), mw(in[3], std::size_t(d)), mw(in[4], std::size_t(f)), std::uint32_t(f), out))
                        raise(errc::unsupported, "{}: moe_ffn rejected its inputs", rt->backend()); },
                1, true};
            r.device_moe_codecs = {"q8_0", "q4_k", "q5_k", "q6_k", "f32"};
        }
    }
}
#ifdef FORGE_GPU_VULKAN
// Last runtime opened by device_registry (for reporting dispatch counts).
inline std::shared_ptr<gpu::vk::runtime>& last_vulkan_runtime() { static std::shared_ptr<gpu::vk::runtime> r; return r; }
// resident = false: the registry of vulkan_registry_flat (no run hooks; the encoded lowerings stage through the host).
inline std::unique_ptr<registry> vulkan_registry(std::shared_ptr<gpu::vk::runtime> rt, bool resident = true) {
    last_vulkan_runtime() = rt;
    auto r = registry::make();
    auto h = std::make_shared<device_hooks>();
    gpu::vk::runtime* raw = rt.get();
    h->begin_run = [raw] { raw->begin_run(); };
    h->end_run = [raw] { raw->end_run(); };
    h->pin = [raw](const float* p, std::size_t n) { raw->pin(p, n); };
    h->to_host = [raw](const float* p, std::size_t n) { raw->to_host(p, n); };
    h->host_wrote = [raw](const float* p, std::size_t n) { raw->host_wrote(p, n); };
    h->release = [raw](const float* p, std::size_t n) { raw->release(p, n); };
    h->forget = [rt](const float* p, std::size_t n) { rt->forget(p, n); };   // owning: plans may outlive the registry
    h->copy = [raw](float* s, std::size_t n, float* d, std::size_t dn, std::size_t off) { raw->copy_resident(s, n, d, dn, off); };
    // Gated DeltaNet state ops on the device (forge_delta_rule / forge_causal_conv ports; the state slot stays resident)
    if (resident && !std::getenv("FORGE_GPU_NO_STATE_OPS") && rt->has_function("forge_delta_rule") && rt->has_function("forge_causal_conv"))
        h->state = [raw](const std::string& fn, std::span<const tensor_view> in, float* S, std::size_t sn, tensor_view out, const ops::attrs& a) {
            return raw->run_state(fn, std::vector<tensor_view>(in.begin(), in.end()), S, sn, out, a); };
    r->hooks = h;
    const bool bcast = rt->has_function("forge_bcast");
    for (std::size_t i = 0; i < ops::op_count; ++i) {
        auto o = ops::op(i);
        std::string name(ops::info(o).name);
        if (!gpu::vk::runtime::has_kernel(name)) continue;
        r->add({"vk." + name, o, 20,
                [name, bcast](std::span<const shape_t> in, const shape_t&, const ops::attrs& a) {
                    if (bcast && gpu::cuda_hip::bcast_op(name, std::vector<shape_t>(in.begin(), in.end())) >= 0) return true;   // forge_bcast
                    bool ok = gpu::vk::supports(name, std::vector<shape_t>(in.begin(), in.end()), a); if (!ok && std::getenv("FORGE_VK_DEBUG")) { std::string s; for (auto& x : in) { for (auto d : x) s += std::to_string(d) + ","; s += "| "; } log::warn("vk: {} unsupported for {}", name, s); } return ok; },
                [rt, name, bcast](std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
                    if (bcast) {
                        std::vector<shape_t> v; for (auto& t : in) v.push_back(t.shape);
                        if (int bo = gpu::cuda_hip::bcast_op(name, v); bo >= 0 && numel(out.shape) == std::max(numel(v[0]), numel(v[1]))) {
                            if (!rt->run_bcast(bo, std::vector<tensor_view>(in.begin(), in.end()), out)) raise(errc::unsupported, "vulkan: broadcast {} rejected its inputs", name);
                            return; }
                    }
                    if (!gpu::vk::run_op(*rt, name, std::vector<tensor_view>(in.begin(), in.end()), a, out))
                        raise(errc::unsupported, "vulkan: {} rejected its inputs", name); }, -1, true,
                gpu::vk::transcendental_tol(name)});
    }
    // reshape on the device: a no-op when the planner aliased it in place, else a copy (scale by 1 is exact).
    r->add({"vk.reshape", ops::op::reshape, 20, nullptr,
            [rt](std::span<const tensor_view> in, tensor_view out, const ops::attrs&) {
                if (in[0].data == out.data) return;
                gpu::vk::push pc; pc.n = std::uint32_t(out.size()); pc.alpha = 1.f;
                std::vector<gpu::vk::hbuf> b{{in[0].data, in[0].size()}, {out.data, out.size()}};
                if (rt->resident()) rt->run_resident("scale", b, pc, pc.n); else rt->run("scale", b, pc, pc.n); }, -1, true});
    // The encoder / vision ops (layer_norm, group_norm, conv1d, conv2d, pooling, attention_bias): GLSL ports of the
    // PORT_NN kernels, shapes as the other device backends take them (cuda_hip::supports), run through run_nn.
    for (auto nm : {"layer_norm", "group_norm", "conv1d", "conv2d", "max_pool2d", "avg_pool2d", "attention_bias"}) {
        const std::string name = nm; auto o = ops::op_from_name(name);
        if (!o || !rt->has_function("forge_" + name)) continue;
        r->add({"vk." + name, *o, 20,
                [name](std::span<const shape_t> in, const shape_t&, const ops::attrs& a) { return gpu::cuda_hip::supports(name, std::vector<shape_t>(in.begin(), in.end()), a); },
                [rt, name](std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
                    if (!rt->run_nn(name, std::vector<tensor_view>(in.begin(), in.end()), a, out)) raise(errc::unsupported, "vulkan: {} rejected its inputs", name); },
                -1, true});
    }
    add_encoded_lowerings(*r, rt, "vk", resident);
    return r;
}
#endif
// CUDA / HIP lowerings ("cuda.<op>" / "hip.<op>"), oracle-gated. With the residency hooks, buffers stay on the
// device for a whole executor run (weights across runs), like the Vulkan backend.
// The same registry serves every runtime with the cuda.hpp contract: CUDA, HIP, OpenCL, Level Zero.
template<class RT> std::unique_ptr<registry> gpu_registry(std::shared_ptr<RT> rt, bool resident = true) {
    auto r = registry::make();
    std::string pre = rt->backend();
    resident = resident && !std::getenv("FORGE_GPU_NO_RESIDENCY");
    if (resident) {
        auto h = std::make_shared<device_hooks>();
        RT* raw = rt.get();
        h->begin_run = [raw] { raw->begin_run(); };
        h->end_run = [raw] { raw->end_run(); };
        if constexpr (requires { raw->end_run_discard(); })
            if (!std::getenv("FORGE_GPU_EAGER_SYNC")) {
                h->end_run_discard = [raw] { raw->end_run_discard(); };
                h->keep = [raw](const float* p, std::size_t n) { raw->keep(p, n); };
                if constexpr (requires { raw->arena((float*)nullptr, std::size_t(0)); raw->forget((const float*)nullptr, std::size_t(0)); }) {
                    h->arena = [rt](float* p, std::size_t n) { rt->arena(p, n); };      // owning: plans may outlive the registry
                    h->forget = [rt](const float* p, std::size_t n) { rt->forget(p, n); };
                }
                if constexpr (requires { raw->ran((const void*)nullptr, true); raw->replay((const void*)nullptr); })
                    if (std::getenv("FORGE_GPU_REPLAY") && !gpu::deterministic()) {   // opt-in while experimental (no measured gain yet)
                        h->ran = [raw](const void* k, bool c) { raw->ran(k, c); };
                        h->replay = [raw](const void* k) { return raw->replay(k); };
                    }
            }
        // every resident backend: a dying plan drops its constants' device copies and pins (plan::~plan)
        if constexpr (requires { raw->forget((const float*)nullptr, std::size_t(0)); })
            if (!h->forget) h->forget = [rt](const float* p, std::size_t n) { rt->forget(p, n); };
        h->pin = [raw](const float* p, std::size_t n) { raw->pin(p, n); };
        h->to_host = [raw](const float* p, std::size_t n) { raw->to_host(p, n); };
        h->host_wrote = [raw](const float* p, std::size_t n) { raw->host_wrote(p, n); };
        h->release = [raw](const float* p, std::size_t n) { raw->release(p, n); };
        h->copy = [raw](float* s, std::size_t n, float* d, std::size_t dn, std::size_t off) { raw->copy_resident(s, n, d, dn, off); };
        if constexpr (requires { raw->copy2d_resident((float*)nullptr, std::size_t(0), (float*)nullptr, std::size_t(0), std::size_t(0), std::size_t(0), std::size_t(0), std::size_t(0)); })
            h->copy2d = [raw](float* s, std::size_t sn, float* d, std::size_t dn, std::size_t rows, std::size_t w, std::size_t ds, std::size_t off) {
                return raw->copy2d_resident(s, sn, d, dn, rows, w, ds, off); };
        if constexpr (requires { raw->run_state(std::string(), std::vector<tensor_view>(), (float*)nullptr, std::size_t(0), tensor_view(), ops::attrs()); })
            if (!std::getenv("FORGE_GPU_NO_STATE_OPS"))
                h->state = [raw](const std::string& fn, std::span<const tensor_view> in, float* S, std::size_t sn, tensor_view out, const ops::attrs& a) {
                    return raw->run_state(fn, std::vector<tensor_view>(in.begin(), in.end()), S, sn, out, a); };
        if constexpr (requires { raw->argmax_resident((const float*)nullptr, std::size_t(0)); })
            h->argmax = [raw](const float* p, std::size_t n) { return raw->argmax_resident(p, n); };
        if constexpr (requires { raw->run_fused(std::string(), std::vector<tensor_view>(), std::vector<tensor_view>(), ops::attrs()); })
            h->fused = [raw](const std::string& k, std::span<const tensor_view> in, std::span<const tensor_view> out, const ops::attrs& a) {
                return raw->run_fused(k, std::vector<tensor_view>(in.begin(), in.end()), std::vector<tensor_view>(out.begin(), out.end()), a); };
        r->hooks = h;
    }
    for (std::size_t i = 0; i < ops::op_count; ++i) {
        auto o = ops::op(i);
        std::string name(ops::info(o).name);
        gpu::cuda_hip::sig sg; if (!gpu::cuda_hip::kernel_sig(name, sg)) continue;
        r->add({pre + "." + name, o, 20,
                [name](std::span<const shape_t> in, const shape_t&, const ops::attrs& a) {
                    std::vector<shape_t> v(in.begin(), in.end());
                    if constexpr (requires { RT::has_bcast; }) if (gpu::cuda_hip::bcast_op(name, v) >= 0) return true;
                    return gpu::cuda_hip::supports(name, v, a); },
                [rt, name](std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
                    if constexpr (requires { RT::has_bcast; }) {
                        std::vector<shape_t> v; for (auto& t : in) v.push_back(t.shape);
                        if (int bo = gpu::cuda_hip::bcast_op(name, v); bo >= 0 && numel(out.shape) == std::max(numel(v[0]), numel(v[1]))) {
                            if (!rt->run_bcast(bo, std::vector<tensor_view>(in.begin(), in.end()), out))
                                raise(errc::unsupported, "{}: broadcast {} rejected its inputs", rt->backend(), name);
                            return; }
                    }
                    if (!rt->run_op(name, std::vector<tensor_view>(in.begin(), in.end()), a, out))
                        raise(errc::unsupported, "{}: {} rejected its inputs", rt->backend(), name); }, -1, resident});
    }
    add_encoded_lowerings(*r, rt, pre, resident);
    return r;
}
// Registry for a device name ("cpu", "vulkan", "cuda[:N]", "hip[:N]", "opencl[:N]", "level_zero[:N]"); null = CPU. `why` explains a fallback.
inline std::unique_ptr<registry> device_registry(const std::string& dev, std::string* why = nullptr, bool resident = true);
// "pipe:<dev>[*w],<dev>[*w],..." e.g. "pipe:cuda:0,cuda:1" or "pipe:vulkan*3,cpu" or "pipe:rpc:host:7000,cpu".
inline std::unique_ptr<registry> pipeline_registry(const std::string& list, std::string* why) {
    auto r = registry::make();
    std::size_t i = 0;
    while (i <= list.size()) {
        // device names contain ':' but never ',' -> split on ','
        auto j = list.find(',', i); if (j == std::string::npos) j = list.size();
        auto d = list.substr(i, j - i); double w = 1;
        if (auto st = d.rfind('*'); st != std::string::npos) { w = std::atof(d.c_str() + st + 1); d = d.substr(0, st); }
        if (d.empty()) { if (why) *why = "pipe: empty stage"; return nullptr; }
        std::string e; auto sr = device_registry(d, &e, false);
        if (!sr && d != "cpu") { if (why) *why = "pipe stage '" + d + "': " + e; return nullptr; }
        r->stages.push_back(std::move(sr)); r->stage_weight.push_back(w > 0 ? w : 1);
        i = j + 1;
    }
    if (r->stages.size() < 2 && why) *why = "pipe: single stage";
    return r;
}
#ifdef FORGE_GPU_VULKAN
// Non-resident Vulkan (pipeline stages / RPC servers): every op round-trips host memory.
inline std::unique_ptr<registry> vulkan_registry_flat(std::shared_ptr<gpu::vk::runtime> rt) {
    auto r = vulkan_registry(rt, false); r->hooks.reset();
    for (std::size_t i = 0; i < ops::op_count; ++i) for (auto& l : r->lowerings(ops::op(i))) if (l.on_device) {
        auto fn = l.fn; l.on_device = false;
        l.fn = [rt, fn](std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
            rt->begin_run(); try { fn(in, out, a); rt->to_host(out.data, out.size()); rt->end_run(); } catch (...) { rt->end_run(); throw; } };
    }
    return r;
}
#endif
// ---- automatic device selection (gpu/select.hpp; docs/gpu.md "Automatic device selection") ----
// GFLOP/s of a 128 x 512 . 512 x 512 f32 matmul_t through `reg`'s own lowering (null = the CPU's), best of 3; the
// data crosses host memory each time, so a device's transfer overhead is part of the number. 0 when it cannot run.
inline double bench_matmul(registry* reg) {
    const i64 rows = 128, k = 512, cols = 512;
    std::vector<float> x(std::size_t(rows * k)), w(std::size_t(cols * k)), y(std::size_t(rows * cols));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = float(i % 13) * 0.01f;
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = float(i % 7) * 0.02f;
    std::vector<shape_t> ss{{rows, k}, {cols, k}}; shape_t os{rows, cols};
    registry& r = reg ? *reg : registry::cpu();
    const lowering* low = nullptr;
    for (auto& l : r.lowerings(ops::op::matmul_t)) if ((!l.supports || l.supports(ss, os, {})) && (!low || l.priority > low->priority)) low = &l;
    if (!low) return 0;
    std::vector<tensor_view> in{{x.data(), ss[0]}, {w.data(), ss[1]}}; tensor_view out{y.data(), os};
    double best = 0;
    try {
        for (int i = 0; i < 3; ++i) {
            auto t0 = std::chrono::steady_clock::now(); low->fn(in, out, {});
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (s > 0) best = std::max(best, 2.0 * double(rows * k * cols) / s / 1e9);
        }
    } catch (...) { return 0; }
    return best;
}
// The cached micro-benchmark of a device (kernel cache directory, keyed by backend + device name; measured once).
// FORGE_DEVICE_BENCH=0: never measure (0 = the class prior is used).
inline double device_gflops(const std::string& backend, const std::string& name, registry* reg) {
    if (const char* e = std::getenv("FORGE_DEVICE_BENCH"); e && std::string(e) == "0") return 0;
    const std::string key = "device-bench v1\n" + backend + "\n" + name;
    if (auto d = gpu::kernel_cache::load("device-bench", key); d.size() == sizeof(double)) { double v; std::memcpy(&v, d.data(), sizeof v); if (v > 0 && v < 1e7) return v; }
    double v = bench_matmul(reg);
    if (v > 0) gpu::kernel_cache::store("device-bench", key, reinterpret_cast<const unsigned char*>(&v), sizeof v);
    return v;
}
// One record per device that opens, plus the CPU. Opening a device builds (or loads) its kernels, so this costs what
// device_registry() costs once per device; `bench` also takes the cached micro-benchmark.
inline std::vector<gpu::device_record> enumerate_devices(bool bench = true) {
    std::vector<gpu::device_record> out;
    { gpu::device_record c; c.device = c.backend = "cpu"; c.name = std::string("cpu/") + simd::active().target; c.cls = gpu::device_class::cpu;
      if (bench) c.gflops = device_gflops("cpu", c.name, nullptr);
      out.push_back(c); }
    auto add = [&](const std::string& dev, const std::string& backend, int ord, const std::string& name, gpu::device_class cls, std::uint64_t mem, registry* reg) {
        gpu::device_record r; r.device = dev; r.backend = backend; r.ordinal = ord; r.name = name; r.cls = cls; r.memory_bytes = mem;
        if (reg) {
            if (reg->device_qgemm) r.qcodecs = reg->device_qgemm_codecs;
            r.moe = bool(reg->device_moe); r.qgather = bool(reg->device_qgather); r.state_ops = reg->hooks && reg->hooks->state;
            if (bench && cls != gpu::device_class::software) r.gflops = device_gflops(backend, name, reg);   // software devices are never benchmarked: they do not compete
        }
        out.push_back(std::move(r));
    };
    auto each = [&](const char* backend, int count, auto open) {
        for (int i = 0; i < count; ++i) {
            std::string why; auto rt0 = open(&why, i); if (!rt0) continue;
            auto rt = std::shared_ptr<typename decltype(rt0)::element_type>(std::move(rt0));
            auto name = rt->device_name(); auto cls = rt->device_class(); std::uint64_t mem = 0;
            if constexpr (requires { rt->memory(); }) mem = rt->memory().total;
            auto reg = gpu_registry(rt, true);
            add(std::string(backend) + ":" + std::to_string(i), backend, i, name, cls, mem, reg.get());
        }
    };
    namespace ch = gpu::cuda_hip;
    each("cuda", ch::runtime::device_count(ch::api::cuda), [](std::string* w, int o) { return ch::runtime::open(ch::api::cuda, w, o); });
    each("hip", ch::runtime::device_count(ch::api::hip), [](std::string* w, int o) { return ch::runtime::open(ch::api::hip, w, o); });
    each("level_zero", gpu::ze::runtime::device_count(), [](std::string* w, int o) { return gpu::ze::runtime::open(w, o); });
    each("opencl", gpu::ocl::runtime::device_count(), [](std::string* w, int o) { return gpu::ocl::runtime::open(w, o); });
    each("metal", gpu::metal::runtime::device_count(), [](std::string* w, int o) { return gpu::metal::runtime::open(w, o); });
#ifdef FORGE_GPU_VULKAN
    { std::string why;
      if (auto rt0 = gpu::vk::runtime::open(&why)) {
          auto rt = std::shared_ptr<gpu::vk::runtime>(std::move(rt0));
          auto name = rt->device_name(); auto cls = rt->device_class(); auto mem = rt->device_memory_bytes();
          auto reg = vulkan_registry_flat(rt);
          add("vulkan", "vulkan", 0, name, cls, mem, reg.get());
      } }
#endif
    return out;
}
// What the caller knows about the model (inference::session fills it before resolving "auto").
inline gpu::select_request& auto_request() { static gpu::select_request r; return r; }
// The device "auto" resolves to for `rq` plus the environment overrides (FORGE_DEVICE_PREFER / _EXCLUDE /
// _ALLOW_SOFTWARE); `why` gets the scoring table. "cpu" when no device beats the CPU.
inline std::string auto_device(const gpu::select_request& rq, std::string* why = nullptr) {
    return gpu::select_device(enumerate_devices(), gpu::with_env(rq), why);
}
// What "pipe:auto" resolves to for `rq` plus the environment overrides: a "pipe:<dev>*<share>,..." string over the
// devices of one backend (gpu::plan_pipeline), a single device's name, or "cpu". `why` gets the table.
inline std::string auto_pipeline(const gpu::select_request& rq, std::string* why = nullptr) {
    return gpu::pipeline_string(gpu::plan_pipeline(enumerate_devices(), gpu::with_env(rq), why));
}
inline std::unique_ptr<registry> device_registry(const std::string& dev, std::string* why, bool resident) {
    if (dev.empty() || dev == "cpu") return nullptr;
    if (dev == "pipe:auto") {
        std::string table; const std::string pick = auto_pipeline(auto_request(), &table);
        log::info("FORGE_DEVICE=pipe:auto -> {}\n{}", pick, table);
        return pick == "cpu" ? nullptr : device_registry(pick, why, resident);
    }
    if (dev == "auto") {
        std::string table; const std::string pick = auto_device(auto_request(), &table);
        log::info("FORGE_DEVICE=auto -> {}\n{}", pick, table);
        return pick == "cpu" ? nullptr : device_registry(pick, why, resident);
    }
    if (dev.rfind("pipe:", 0) == 0) return pipeline_registry(dev.substr(5), why);
    if (dev.rfind("rpc:", 0) == 0) return rpc::open(dev.substr(4), why);
#ifdef FORGE_GPU_VULKAN
    if (dev == "vulkan") {
        std::string e;
        auto rt = gpu::vk::runtime::open(&e);
        if (rt) return resident ? vulkan_registry(std::shared_ptr<gpu::vk::runtime>(std::move(rt))) : vulkan_registry_flat(std::shared_ptr<gpu::vk::runtime>(std::move(rt)));
        if (why) *why = "vulkan unavailable: " + e;
        return nullptr;
    }
#endif
    auto base = dev.substr(0, dev.find(':'));   // "cuda:1" selects device ordinal 1
    if (base == "cuda" || base == "hip") {
        std::string e; int ord = dev.size() > base.size() ? std::atoi(dev.c_str() + base.size() + 1) : 0;
        auto rt = gpu::cuda_hip::runtime::open(base == "cuda" ? gpu::cuda_hip::api::cuda : gpu::cuda_hip::api::hip, &e, ord);
        if (rt) return gpu_registry(std::shared_ptr<gpu::cuda_hip::runtime>(std::move(rt)), resident);
        if (why) *why = e;
        return nullptr;
    }
    if (base == "opencl") {   // "opencl:N" = Nth device over all platforms (GPUs first)
        std::string e; int ord = dev.size() > base.size() ? std::atoi(dev.c_str() + base.size() + 1) : 0;
        auto rt = gpu::ocl::runtime::open(&e, ord);
        if (rt) return gpu_registry(std::shared_ptr<gpu::ocl::runtime>(std::move(rt)), resident);
        if (why) *why = e;
        return nullptr;
    }
    if (base == "metal") {
        std::string e; int ord = dev.size() > base.size() ? std::atoi(dev.c_str() + base.size() + 1) : 0;
        auto rt = gpu::metal::runtime::open(&e, ord);
        if (rt) return gpu_registry(std::shared_ptr<gpu::metal::runtime>(std::move(rt)), resident);
        if (why) *why = e;
        return nullptr;
    }
    if (base == "level_zero" || base == "ze") {
        std::string e; int ord = dev.size() > base.size() ? std::atoi(dev.c_str() + base.size() + 1) : 0;
        auto rt = gpu::ze::runtime::open(&e, ord);
        if (rt) return gpu_registry(std::shared_ptr<gpu::ze::runtime>(std::move(rt)), resident);
        if (why) *why = e;
        return nullptr;
    }
    if (why) *why = "device '" + dev + "' not available in this build";
    return nullptr;
}
} // namespace forge::graph
