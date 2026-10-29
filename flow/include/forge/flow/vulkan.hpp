#pragma once
// Vulkan lowerings for the flow executor: every spec op with a generated GPU kernel gets a "vk.<op>" lowering
// (priority above the CPU ones). Like every lowering it must pass the oracle before it is selected, and any
// op/shape without a GPU kernel runs on the CPU, so a whole model runs with the GPU doing what it can.
#include <forge/flow/kernels.hpp>
#include <forge/grid/vulkan.hpp>

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
                        raise(errc::unsupported, "vulkan: {} rejected its inputs", name); }, -1, true});
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
// Registry for a device name ("cpu" or "vulkan"); null = CPU. `why` explains a fallback.
inline std::unique_ptr<registry> device_registry(const std::string& dev, std::string* why = nullptr) {
    if (dev.empty() || dev == "cpu") return nullptr;
#ifdef FORGE_GRID_VULKAN
    if (dev == "vulkan") {
        std::string e;
        auto rt = grid::vk::runtime::open(&e);
        if (rt) return vulkan_registry(std::shared_ptr<grid::vk::runtime>(std::move(rt)));
        if (why) *why = "vulkan unavailable: " + e;
        return nullptr;
    }
#endif
    if (why) *why = "device '" + dev + "' not available in this build";
    return nullptr;
}
} // namespace forge::flow
