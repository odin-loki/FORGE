// Runs every generated GPU kernel through the Vulkan backend and checks it against the spec reference.
// Exit 77 (ctest SKIP) when no Vulkan device is available.
#include <forge/grid/vulkan.hpp>
#include <cstdio>
#ifndef FORGE_GRID_VULKAN
int main() { std::puts("vulkan backend not built (no vulkan headers / glslangValidator)"); return 77; }
#else
#include <forge/spec/ops.gen.hpp>
#include <cmath>
#include <random>
using namespace forge;
int main() {
    std::string why;
    auto rt = grid::vk::runtime::open(&why);
    if (!rt) { std::printf("SKIP: %s\n", why.c_str()); return 77; }
    std::printf("vulkan device: %s\n", rt->device_name().c_str());
    std::mt19937 g(7);
    auto rnd = [&](shape_t s, float lo, float hi) { tensor t(s); std::uniform_real_distribution<float> d(lo, hi); for (auto& v : t.data) v = d(g); return t; };
    int fails = 0, n = 0;
    for (auto& b : grid::spv::table) {
        std::string name = b.name;
        auto o = spec::op_from_name(name);
        if (!o) { std::printf("FAIL %s not in spec\n", b.name); ++fails; continue; }
        auto& info = spec::info(*o);
        bool positive = name == "sqrt" || name == "rsqrt" || name == "log" || name == "log2" || name == "recip";
        std::vector<tensor> in;
        spec::attrs a;
        float alpha = 0;
        if (name == "matmul_t") in = {rnd({3, 5, 17}, -1, 1), rnd({9, 17}, -1, 1)};
        else if (name == "rms_norm") { in = {rnd({7, 33}, -2, 2), rnd({33}, -1, 1)}; alpha = 1e-5f; a = {{"eps", 1e-5}}; }
        else { in.push_back(rnd({4, 129}, positive ? 0.1f : -4.f, 4.f)); if (info.inputs == 2) in.push_back(rnd({4, 129}, 0.5f, 3.f)); }
        if (name == "scale") { alpha = 1.75f; a = {{"alpha", 1.75}}; }
        tensor ref = spec::reference(*o)(in, a), got;
        if (!grid::vk::run_op(*rt, name, in, alpha, got)) { std::printf("FAIL %s: no lowering\n", b.name); ++fails; continue; }
        // GLSL float precision: transcendental builtins are not correctly rounded, so use a relaxed bound.
        double worst = 0;
        for (std::size_t i = 0; i < ref.data.size(); ++i) {
            double e = std::fabs(double(got.data[i]) - ref.data[i]) / std::max(1.0, std::fabs(double(ref.data[i])));
            if (!(e <= worst)) worst = std::isnan(e) ? 1e30 : e;
        }
        bool ok = got.shape == ref.shape && worst <= 2e-5;
        std::printf("%s %-10s max err %.2e\n", ok ? "ok  " : "FAIL", b.name, worst);
        fails += !ok; ++n;
    }
    std::printf("%d kernels, %d failures\n", n, fails);
    return fails ? 1 : 0;
}
#endif
