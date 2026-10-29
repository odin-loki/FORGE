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
    for (std::size_t i = 0; i < spec::op_count; ++i) {
        auto o = spec::op(i);
        std::string name(spec::info(o).name);
        if (!grid::vk::runtime::has_kernel(name)) continue;
        r->add({"vk." + name, o, 20,
                [name](std::span<const shape_t> in, const shape_t&, const spec::attrs& a) {
                    bool ok = grid::vk::supports(name, std::vector<shape_t>(in.begin(), in.end()), a); if (!ok && std::getenv("FORGE_VK_DEBUG")) { std::string s; for (auto& x : in) { for (auto d : x) s += std::to_string(d) + ","; s += "| "; } log::warn("vk: {} unsupported for {}", name, s); } return ok; },
                [rt, name](std::span<const tensor_view> in, tensor_view out, const spec::attrs& a) {
                    if (!grid::vk::run_op(*rt, name, std::vector<tensor_view>(in.begin(), in.end()), a, out))
                        raise(errc::unsupported, "vulkan: {} rejected its inputs", name); }});
    }
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
