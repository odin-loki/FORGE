#pragma once
// Vulkan lowerings for the flow executor: every spec op with a generated GPU kernel gets a "vk.<op>" lowering
// (priority above the CPU ones). Like every lowering it must pass the oracle before it is selected, and any
// op/shape without a GPU kernel runs on the CPU, so a whole model runs with the GPU doing what it can.
#include <forge/flow/kernels.hpp>
#include <forge/flow/codecs.hpp>
#include <forge/grid/vulkan.hpp>
#include <forge/grid/cuda.hpp>
#include <forge/grid/opencl.hpp>
#include <forge/grid/levelzero.hpp>
#include <forge/grid/metal.hpp>
#include <forge/flow/rpc.hpp>

namespace forge::flow {
#ifdef FORGE_GRID_VULKAN
// Last runtime opened by device_registry (for reporting dispatch counts).
inline std::shared_ptr<grid::vk::runtime>& last_vulkan_runtime() { static std::shared_ptr<grid::vk::runtime> r; return r; }
inline std::unique_ptr<registry> vulkan_registry(std::shared_ptr<grid::vk::runtime> rt) {
    last_vulkan_runtime() = rt;
    auto r = registry::make();
    auto h = std::make_shared<device_hooks>();
    grid::vk::runtime* raw = rt.get();
    h->begin_run = [raw] { raw->begin_run(); };
    h->end_run = [raw] { raw->end_run(); };
    h->pin = [raw](const float* p, std::size_t n) { raw->pin(p, n); };
    h->to_host = [raw](const float* p, std::size_t n) { raw->to_host(p, n); };
    h->host_wrote = [raw](const float* p, std::size_t n) { raw->host_wrote(p, n); };
    h->release = [raw](const float* p, std::size_t n) { raw->release(p, n); };
    h->copy = [raw](float* s, std::size_t n, float* d, std::size_t dn, std::size_t off) { raw->copy_resident(s, n, d, dn, off); };
    r->hooks = h;
    for (std::size_t i = 0; i < spec::op_count; ++i) {
        auto o = spec::op(i);
        std::string name(spec::info(o).name);
        if (!grid::vk::runtime::has_kernel(name)) continue;
        r->add({"vk." + name, o, 20,
                [name](std::span<const shape_t> in, const shape_t&, const spec::attrs& a) {
                    bool ok = grid::vk::supports(name, std::vector<shape_t>(in.begin(), in.end()), a); if (!ok && std::getenv("FORGE_VK_DEBUG")) { std::string s; for (auto& x : in) { for (auto d : x) s += std::to_string(d) + ","; s += "| "; } log::warn("vk: {} unsupported for {}", name, s); } return ok; },
                [rt, name](std::span<const tensor_view> in, tensor_view out, const spec::attrs& a) {
                    if (!grid::vk::run_op(*rt, name, std::vector<tensor_view>(in.begin(), in.end()), a, out))
                        raise(errc::unsupported, "vulkan: {} rejected its inputs", name); }, -1, true,
                grid::vk::transcendental_tol(name)});
    }
    // reshape on the device: a no-op when the planner aliased it in place, else a copy (scale by 1 is exact).
    r->add({"vk.reshape", spec::op::reshape, 20, nullptr,
            [rt](std::span<const tensor_view> in, tensor_view out, const spec::attrs&) {
                if (in[0].data == out.data) return;
                grid::vk::push pc; pc.n = std::uint32_t(out.size()); pc.alpha = 1.f;
                std::vector<grid::vk::hbuf> b{{in[0].data, in[0].size()}, {out.data, out.size()}};
                if (rt->resident()) rt->run_resident("scale", b, pc, pc.n); else rt->run("scale", b, pc, pc.n); }, -1, true});
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
            if (!std::getenv("FORGE_GPU_EAGER_SYNC")) h->end_run_discard = [raw] { raw->end_run_discard(); };
        h->pin = [raw](const float* p, std::size_t n) { raw->pin(p, n); };
        h->to_host = [raw](const float* p, std::size_t n) { raw->to_host(p, n); };
        h->host_wrote = [raw](const float* p, std::size_t n) { raw->host_wrote(p, n); };
        h->release = [raw](const float* p, std::size_t n) { raw->release(p, n); };
        h->copy = [raw](float* s, std::size_t n, float* d, std::size_t dn, std::size_t off) { raw->copy_resident(s, n, d, dn, off); };
        if constexpr (requires { raw->run_state(std::string(), std::vector<tensor_view>(), (float*)nullptr, std::size_t(0), tensor_view(), spec::attrs()); })
            if (!std::getenv("FORGE_GPU_NO_STATE_OPS"))
                h->state = [raw](const std::string& fn, std::span<const tensor_view> in, float* S, std::size_t sn, tensor_view out, const spec::attrs& a) {
                    return raw->run_state(fn, std::vector<tensor_view>(in.begin(), in.end()), S, sn, out, a); };
        r->hooks = h;
    }
    for (std::size_t i = 0; i < spec::op_count; ++i) {
        auto o = spec::op(i);
        std::string name(spec::info(o).name);
        grid::gpu::sig sg; if (!grid::gpu::kernel_sig(name, sg)) continue;
        r->add({pre + "." + name, o, 20,
                [name](std::span<const shape_t> in, const shape_t&, const spec::attrs& a) {
                    return grid::gpu::supports(name, std::vector<shape_t>(in.begin(), in.end()), a); },
                [rt, name](std::span<const tensor_view> in, tensor_view out, const spec::attrs& a) {
                    if (!rt->run_op(name, std::vector<tensor_view>(in.begin(), in.end()), a, out))
                        raise(errc::unsupported, "{}: {} rejected its inputs", rt->backend(), name); }, -1, resident});
    }
    // Quantised matmul_t with the weight kept encoded on the device (CUDA/HIP forge_qmatmul_t); FORGE_GPU_QMATMUL=0
    // falls back to the f32 path (decoded weights within FORGE_GPU_WEIGHT_BUDGET, the rest on the CPU).
    if constexpr (requires(RT& t, const tensor_view& v, tensor_view o) { t.run_qmatmul(v, nullptr, std::size_t(0), 0u, 0u, o); t.has_function(""); }) {
        auto* qe = std::getenv("FORGE_GPU_QMATMUL");
        if (!(qe && std::string(qe) == "0") && rt->has_function("forge_qmatmul_t")) {
            r->device_qgemm = lowering{pre + ".qmatmul_t", spec::op::matmul_t, 30, [](auto, auto&, auto&) { return true; },
                [rt](std::span<const tensor_view> in, tensor_view out, const spec::attrs&) {
                    const auto* c = static_cast<const codecs::codec*>(in[1].qcodec);
                    std::uint32_t fmt = c->name == "q8_0" ? 1 : c->name == "q4_k" ? 2 : c->name == "q6_k" ? 3 : c->name == "q5_k" ? 4 : 0;
                    std::size_t k = std::size_t(in[0].shape.back());
                    if (!rt->run_qmatmul(in[0], in[1].qdata, c->bytes(k), std::uint32_t(in[1].shape[0]), fmt, out))
                        raise(errc::unsupported, "{}: qmatmul_t rejected a {} weight", rt->backend(), c->name); },
                1, resident};
            r->device_qgemm_codecs = {"q8_0", "q4_k", "q5_k", "q6_k"};
        }
    }
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
#ifdef FORGE_GRID_VULKAN
// Non-resident Vulkan (pipeline stages / RPC servers): every op round-trips host memory.
inline std::unique_ptr<registry> vulkan_registry_flat(std::shared_ptr<grid::vk::runtime> rt) {
    auto r = vulkan_registry(rt); r->hooks.reset();
    for (std::size_t i = 0; i < spec::op_count; ++i) for (auto& l : r->lowerings(spec::op(i))) if (l.on_device) {
        auto fn = l.fn; l.on_device = false;
        l.fn = [rt, fn](std::span<const tensor_view> in, tensor_view out, const spec::attrs& a) {
            rt->begin_run(); try { fn(in, out, a); rt->to_host(out.data, out.size()); rt->end_run(); } catch (...) { rt->end_run(); throw; } };
    }
    return r;
}
#endif
inline std::unique_ptr<registry> device_registry(const std::string& dev, std::string* why, bool resident) {
    if (dev.empty() || dev == "cpu") return nullptr;
    if (dev.rfind("pipe:", 0) == 0) return pipeline_registry(dev.substr(5), why);
    if (dev.rfind("rpc:", 0) == 0) return rpc::open(dev.substr(4), why);
#ifdef FORGE_GRID_VULKAN
    if (dev == "vulkan") {
        std::string e;
        auto rt = grid::vk::runtime::open(&e);
        if (rt) return resident ? vulkan_registry(std::shared_ptr<grid::vk::runtime>(std::move(rt))) : vulkan_registry_flat(std::shared_ptr<grid::vk::runtime>(std::move(rt)));
        if (why) *why = "vulkan unavailable: " + e;
        return nullptr;
    }
#endif
    auto base = dev.substr(0, dev.find(':'));   // "cuda:1" selects device ordinal 1
    if (base == "cuda" || base == "hip") {
        std::string e; int ord = dev.size() > base.size() ? std::atoi(dev.c_str() + base.size() + 1) : 0;
        auto rt = grid::gpu::runtime::open(base == "cuda" ? grid::gpu::api::cuda : grid::gpu::api::hip, &e, ord);
        if (rt) return gpu_registry(std::shared_ptr<grid::gpu::runtime>(std::move(rt)), resident);
        if (why) *why = e;
        return nullptr;
    }
    if (base == "opencl") {   // "opencl:N" = Nth device over all platforms (GPUs first)
        std::string e; int ord = dev.size() > base.size() ? std::atoi(dev.c_str() + base.size() + 1) : 0;
        auto rt = grid::ocl::runtime::open(&e, ord);
        if (rt) return gpu_registry(std::shared_ptr<grid::ocl::runtime>(std::move(rt)), resident);
        if (why) *why = e;
        return nullptr;
    }
    if (base == "metal") {
        std::string e; int ord = dev.size() > base.size() ? std::atoi(dev.c_str() + base.size() + 1) : 0;
        auto rt = grid::metal::runtime::open(&e, ord);
        if (rt) return gpu_registry(std::shared_ptr<grid::metal::runtime>(std::move(rt)), resident);
        if (why) *why = e;
        return nullptr;
    }
    if (base == "level_zero" || base == "ze") {
        std::string e; int ord = dev.size() > base.size() ? std::atoi(dev.c_str() + base.size() + 1) : 0;
        auto rt = grid::ze::runtime::open(&e, ord);
        if (rt) return gpu_registry(std::shared_ptr<grid::ze::runtime>(std::move(rt)), resident);
        if (why) *why = e;
        return nullptr;
    }
    if (why) *why = "device '" + dev + "' not available in this build";
    return nullptr;
}
} // namespace forge::flow
