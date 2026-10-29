// (1) every generated GPU kernel vs the spec reference on the Vulkan backend;
// (2) a whole tiny Llama run end-to-end with FORGE's Vulkan lowerings vs the CPU path.
// Exit 77 (ctest SKIP) when no Vulkan device is available.
#include <forge/grid/vulkan.hpp>
#include <cstdio>
#ifndef FORGE_GRID_VULKAN
int main() { std::puts("vulkan backend not built (no vulkan headers / glslangValidator)"); return 77; }
#else
#include "tiny_model.hpp"
#include <forge/mind/mind.hpp>
#include <cmath>
#include <filesystem>
#include <unistd.h>
#include <random>
using namespace forge;

static int kernels(grid::vk::runtime& rt) {
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
        if (name == "matmul_t") in = {rnd({3, 5, 17}, -1, 1), rnd({9, 17}, -1, 1)};
        else if (name == "rms_norm") { in = {rnd({7, 33}, -2, 2), rnd({33}, -1, 1)}; a = {{"eps", 1e-5}}; }
        else if (name == "rope") { in = {rnd({6, 3, 16}, -1, 1), tensor({6})}; for (int i = 0; i < 6; ++i) in[1].data[std::size_t(i)] = float(i * 7); a = {{"base", 10000.0}, {"neox", 1.0}}; }
        else if (name == "gather") { in = {rnd({11, 8}, -1, 1), tensor({5})}; for (int i = 0; i < 5; ++i) in[1].data[std::size_t(i)] = float((i * 4) % 11); }
        else if (name == "slice") { in = {rnd({3, 10, 4}, -1, 1)}; a = {{"axis", 1.0}, {"start", 2.0}, {"end", 7.0}}; }
        else if (name == "attention") { in = {rnd({4, 4, 8}, -1, 1), rnd({9, 2, 8}, -1, 1), rnd({9, 2, 6}, -1, 1)}; a = {{"causal", 1.0}}; }
        else { in.push_back(rnd({4, 129}, positive ? 0.1f : -4.f, 4.f)); if (info.inputs == 2) in.push_back(rnd({4, 129}, 0.5f, 3.f)); }
        if (name == "scale") a = {{"alpha", 1.75}};
        tensor ref = spec::reference(*o)(in, a), got;
        if (!grid::vk::run_op(rt, name, in, a, got)) { std::printf("FAIL %s: no lowering\n", b.name); ++fails; continue; }
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
    return fails;
}

static int end_to_end() {
    auto path = (std::filesystem::temp_directory_path() / ("forge_test_vk_e2e_" + std::to_string(::getpid()) + ".fpk")).string();
    forge_test::make_tiny_model(path);
    auto cpu = mind::model::open(path), gpu = mind::model::open(path);
    cpu->set_device("cpu");
    if (gpu->set_device("vulkan") != "vulkan") { std::puts("FAIL e2e: vulkan device not selected"); return 1; }
    {   // every vk lowering must pass the same oracle gate the registry applies
        auto reg = flow::vulkan_registry(flow::last_vulkan_runtime());
        for (std::size_t i = 0; i < spec::op_count; ++i) for (auto& l : reg->lowerings(spec::op(i))) {
            if (l.name.rfind("vk.", 0) != 0) continue;
            std::string msg;
            if (!flow::registry::oracle_check(l, &msg)) { std::printf("FAIL oracle %s: %s\n", l.name.c_str(), msg.c_str()); return 1; }
        }
    }
    std::vector<int> prompt = {1, 5, 9, 33, 70, 101, 7};
    mind::session a(cpu), b(gpu);
    auto la = a.forward(prompt), lb = b.forward(prompt);
    float worst = 0;
    auto diff = [&](const std::vector<float>& x, const std::vector<float>& y) { for (std::size_t i = 0; i < x.size(); ++i) worst = std::max(worst, std::fabs(x[i] - y[i])); };
    diff(la, lb);
    mind::sampler g1, g2;
    a.reset(); b.reset();
    auto ta = a.generate(prompt, 12, g1), tb = b.generate(prompt, 12, g2);
    std::size_t vk_ops = 0;
    for (auto& [k, v] : flow::last_vulkan_runtime()->dispatches()) vk_ops += v;
    // Which lowering every step of the decode plan uses: all compute should be on the device.
    auto& plan = gpu->plan_for(1, false, const_cast<flow::state_store*>(&b.state()));
    std::printf("%s", plan.describe().c_str());
    int cpu_compute = 0;
    for (auto& st : plan.steps) if (st.low && st.low->name.rfind("vk.", 0) != 0) {
        ++cpu_compute; std::printf("  on cpu: %s (%s)\n", std::string(spec::info(st.low->op).name).c_str(), st.low->name.c_str()); }
    // Residency: weights upload once, and a decode step is one submit, not one per op.
    auto rs = flow::last_vulkan_runtime()->residency();
    std::printf("  residency: %zu uploads (%.1f KiB), %zu downloads, %zu submits for %zu dispatches, %zu pinned buffers\n",
                rs.uploads, double(rs.upload_bytes) / 1024, rs.downloads, rs.submits, vk_ops, rs.resident);
    bool batched = rs.submits * 4 < vk_ops;
    if (!batched) std::puts("FAIL residency: dispatches are not batched");
    bool ok = la.size() == lb.size() && worst < 1e-3f && ta == tb && vk_ops > 0 && cpu_compute == 0 && batched;
    std::printf("%s e2e tiny llama: prefill logits max|err| %.2e, 12 greedy tokens %s, %zu vulkan dispatches, %d CPU compute steps\n",
                ok ? "ok  " : "FAIL", double(worst), ta == tb ? "identical" : "DIFFER", vk_ops, cpu_compute);
    std::error_code ec; std::filesystem::remove(path, ec);
    return ok ? 0 : 1;
}

static int pipeline_cache() {
    // A second runtime must pick up the cache file written by the first one.
    std::string path;
    { auto a = grid::vk::runtime::open(); if (!a) return 1;
      path = a->pipeline_cache_path();
      std::vector<float> x(64, 1.f), y(64, 2.f), o(64);
      if (grid::vk::runtime::has_kernel("add")) a->run("add", {{x.data(), 64}, {y.data(), 64}, {o.data(), 64}}, {}, 64); }
    if (path.empty() || !std::filesystem::exists(path)) { std::printf("FAIL pipeline cache not written (%s)\n", path.c_str()); return 1; }
    auto b = grid::vk::runtime::open(); if (!b) return 1;
    if (b->pipeline_cache_path() != path || b->pipeline_cache_loaded_bytes() == 0) { std::printf("FAIL pipeline cache not reloaded\n"); return 1; }
    std::printf("pipeline cache: %s (%zu bytes reloaded)\n", path.c_str(), b->pipeline_cache_loaded_bytes());
    return 0;
}

int main() {
    std::string why;
    auto tmp = std::filesystem::temp_directory_path() / ("forge-vkcache-" + std::to_string(::getpid()));
    
#if defined(_WIN32)
    _putenv_s("FORGE_CACHE_DIR", tmp.c_str());
#else
    ::setenv("FORGE_CACHE_DIR", tmp.c_str(), 1);
#endif
    auto rt = grid::vk::runtime::open(&why);
    if (!rt) { std::printf("SKIP: %s\n", why.c_str()); return 77; }
    std::printf("vulkan device: %s\n", rt->device_name().c_str());
    int f = kernels(*rt);
    f += end_to_end();
    rt.reset();
    f += pipeline_cache();
    std::error_code ec; std::filesystem::remove_all(tmp, ec);
    return f ? 1 : 0;
}
#endif
