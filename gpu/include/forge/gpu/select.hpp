#pragma once
// Automatic device selection (FORGE_DEVICE=auto; docs/gpu.md, "Automatic device selection").
//
// This header is the pure part: capability records and the scoring rule, with no driver calls, so it can be unit
// tested with synthetic records (tests/test_gpu.cpp). graph/vulkan.hpp fills the records from the runtimes that open
// (graph::enumerate_devices) and resolves "auto" with select_device().
//
// Score of a device for a model (higher is better; the CPU is always a candidate):
//   throughput   the cached micro-benchmark (GFLOP/s of a small f32 matmul through the device's own kernel) when
//                every eligible candidate has one (measured and prior numbers are never mixed), else a prior per
//                device class (discrete 200, integrated 20, virtual 10, cpu 4, software 0.5);
//   x coverage   0.25 + 0.75 * (share of the model's weight codecs the device has a quantised matmul for): an
//                uncovered codec is decoded to f32 on the device or runs on the host;
//   x memory     1 when the model fits in 90% of the device memory (or either size is unknown), else 0.01: the
//                device would evict and re-upload on every token;
//   x preference 1000 for a backend or device named in FORGE_DEVICE_PREFER (it wins whenever it is eligible).
// Hard rules, applied before the score:
//   * a device named in FORGE_DEVICE_EXCLUDE (backend "opencl" or device "cuda:1") is never chosen;
//   * a software device (lavapipe, SwiftShader, an OpenCL CPU runtime, the mock drivers) never wins over the CPU
//     unless FORGE_DEVICE_ALLOW_SOFTWARE=1: it is the same cores with more overhead.
// Ties go to the backend with the more complete kernel set (cuda, hip, metal, level_zero, opencl, vulkan, cpu), then
// to the lower ordinal, so the choice does not depend on enumeration order.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

namespace forge::gpu {

enum class device_class { discrete, integrated, virtual_gpu, software, cpu, unknown };
inline const char* name(device_class c) {
    switch (c) { case device_class::discrete: return "discrete"; case device_class::integrated: return "integrated";
                 case device_class::virtual_gpu: return "virtual"; case device_class::software: return "software";
                 case device_class::cpu: return "cpu"; default: return "unknown"; }
}
// Device names that are software rasterisers / CPU runtimes whatever the driver says about their type.
inline bool software_name(std::string n) {
    std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    for (auto s : {"llvmpipe", "lavapipe", "swiftshader", "softpipe", "pocl", "portable computing language", "mock", "swrast", "warp", "basic render"})
        if (n.find(s) != std::string::npos) return true;
    return false;
}

struct device_record {
    std::string device;                 // what FORGE_DEVICE takes: "cpu", "cuda:0", "opencl:1", "vulkan", ...
    std::string backend;                // "cpu", "cuda", "hip", "opencl", "level_zero", "vulkan", "metal"
    std::string name;                   // the driver's device name
    int ordinal = 0;
    device_class cls = device_class::unknown;
    std::uint64_t memory_bytes = 0;     // device memory (0 = unknown / host memory)
    std::vector<std::string> qcodecs;   // weight codecs with a device quantised matmul
    bool state_ops = false, moe = false, qgather = false;
    double gflops = 0;                  // micro-benchmark (0 = not measured: the class prior is used)
};
struct select_request {
    std::uint64_t model_bytes = 0;      // bytes the model keeps on the device (0 = unknown)
    std::vector<std::string> codecs;    // weight codecs the model uses (empty = unknown / f32)
    bool allow_software = false;        // FORGE_DEVICE_ALLOW_SOFTWARE
    std::vector<std::string> prefer;    // FORGE_DEVICE_PREFER: backends or devices
    std::vector<std::string> exclude;   // FORGE_DEVICE_EXCLUDE
};
inline std::vector<std::string> split_list(const char* e) {
    std::vector<std::string> r; if (!e) return r;
    std::string s = e;
    for (std::size_t p = 0, q; p <= s.size(); p = q + 1) { q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) r.push_back(s.substr(p, q - p)); }
    return r;
}
// The user overrides from the environment, on top of `r`.
inline select_request with_env(select_request r) {
    if (const char* e = std::getenv("FORGE_DEVICE_ALLOW_SOFTWARE"); e && *e && std::string(e) != "0") r.allow_software = true;
    for (auto& s : split_list(std::getenv("FORGE_DEVICE_PREFER"))) r.prefer.push_back(s);
    for (auto& s : split_list(std::getenv("FORGE_DEVICE_EXCLUDE"))) r.exclude.push_back(s);
    return r;
}
inline bool names_device(const std::vector<std::string>& list, const device_record& d) {
    for (auto& s : list) if (s == d.device || s == d.backend || (d.ordinal == 0 && s + ":0" == d.device)) return true;
    return false;
}
inline double class_prior(device_class c) {
    switch (c) { case device_class::discrete: return 200; case device_class::integrated: return 20; case device_class::virtual_gpu: return 10;
                 case device_class::cpu: return 4; case device_class::software: return 0.5; default: return 10; }
}
inline int backend_rank(const std::string& b) {
    int i = 0; for (auto s : {"cuda", "hip", "metal", "level_zero", "opencl", "vulkan", "cpu"}) { if (b == s) return i; ++i; }
    return i;
}
struct scored { double score = 0; bool eligible = true; std::string why; };
inline scored score(const device_record& d, const select_request& rq, bool measured = false) {
    scored s;
    if (d.cls != device_class::cpu && names_device(rq.exclude, d)) { s.eligible = false; s.why = "excluded"; return s; }
    if (d.cls == device_class::software && !rq.allow_software) { s.eligible = false; s.why = "software device (FORGE_DEVICE_ALLOW_SOFTWARE=1 to allow)"; return s; }
    double thr = measured && d.gflops > 0 ? d.gflops : class_prior(d.cls);
    double cov = 1;
    if (!rq.codecs.empty() && d.cls != device_class::cpu) {
        std::size_t have = 0;
        for (auto& c : rq.codecs) if (c == "f32" || std::find(d.qcodecs.begin(), d.qcodecs.end(), c) != d.qcodecs.end()) ++have;
        cov = 0.25 + 0.75 * double(have) / double(rq.codecs.size());
    }
    double mem = 1;
    if (d.cls != device_class::cpu && rq.model_bytes && d.memory_bytes && double(rq.model_bytes) > 0.9 * double(d.memory_bytes)) mem = 0.01;
    double pref = names_device(rq.prefer, d) ? 1000 : 1;
    s.score = thr * cov * mem * pref;
    s.why = std::string(measured && d.gflops > 0 ? "measured" : "prior") + (cov < 1 ? ", partial codec coverage" : "") + (mem < 1 ? ", model does not fit" : "") + (pref > 1 ? ", preferred" : "");
    return s;
}
// The best device of `devs` for the request; "cpu" when nothing else is eligible (a "cpu" record need not be listed:
// without one the CPU scores its class prior). `why` gets one line per candidate.
inline std::string select_device(const std::vector<device_record>& devs, const select_request& rq, std::string* why = nullptr) {
    device_record cpu; cpu.device = cpu.backend = "cpu"; cpu.cls = device_class::cpu;
    std::vector<device_record> all = devs;
    if (std::none_of(all.begin(), all.end(), [](const device_record& d) { return d.device == "cpu"; })) all.push_back(cpu);
    bool measured = true;   // use the micro-benchmarks only when every eligible candidate has one
    for (auto& d : all) if (score(d, rq).eligible && !(d.gflops > 0)) measured = false;
    const device_record* best = nullptr; double bs = -1;
    for (auto& d : all) {
        auto s = score(d, rq, measured);
        if (why) *why += d.device + " (" + name(d.cls) + (d.name.empty() ? "" : ", " + d.name) + "): " + (s.eligible ? "score " + std::to_string(s.score) + ", " : "skipped, ") + s.why + "\n";
        if (!s.eligible) continue;
        bool better = !best || s.score > bs ||
                      (s.score == bs && (backend_rank(d.backend) < backend_rank(best->backend) || (d.backend == best->backend && d.ordinal < best->ordinal)));
        if (better) { best = &d; bs = s.score; }
    }
    return best ? best->device : std::string("cpu");
}

// ---- automatic multi-device split ("pipe:auto"; docs/graph.md) ----
// One stage of a pipeline: the device and the share of the model's work it gets.
struct pipe_stage { std::string device; double share = 0; };
// Split a model over the devices of one backend. Candidates are the eligible devices (score(): not excluded, not
// software unless allowed) of the best-ranked backend that has any; devices of other backends are left out, because
// one physical GPU usually shows up under several APIs (cuda:0, opencl:0, vulkan) and must not be counted twice.
// Shares start proportional to throughput (the micro-benchmark when every candidate has one, else the class prior).
// With a known model size no device gets more than fits in 90% of its memory: the excess moves to the devices that
// still have room (water-filling), and what fits on none of them becomes a last "cpu" stage. Stages are ordered by
// device ordinal, so consecutive layer ranges go to consecutive devices. Returns no stage when no device is eligible
// and one stage when a single device takes everything; `why` gets one line per candidate.
inline std::vector<pipe_stage> plan_pipeline(const std::vector<device_record>& devs, const select_request& rq, std::string* why = nullptr) {
    std::vector<const device_record*> cand; int rank = 1 << 20;
    for (auto& d : devs) if (d.cls != device_class::cpu && score(d, rq).eligible) rank = std::min(rank, backend_rank(d.backend));
    for (auto& d : devs) {
        if (d.cls == device_class::cpu) continue;
        auto sc = score(d, rq);
        const bool take = sc.eligible && backend_rank(d.backend) == rank;
        if (take) cand.push_back(&d);
        else if (why) *why += d.device + ": not used, " + (sc.eligible ? "another backend's devices were chosen (the same GPU may be listed under several APIs)" : sc.why) + "\n";
    }
    std::sort(cand.begin(), cand.end(), [](const device_record* a, const device_record* b) { return a->ordinal < b->ordinal; });
    if (cand.empty()) return {};
    bool measured = true; for (auto* d : cand) if (!(d->gflops > 0)) measured = false;
    const std::size_t n = cand.size();
    std::vector<double> thr(n), cap(n, 1.0), share(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        thr[i] = measured ? cand[i]->gflops : class_prior(cand[i]->cls);
        if (rq.model_bytes && cand[i]->memory_bytes) cap[i] = std::min(1.0, 0.9 * double(cand[i]->memory_bytes) / double(rq.model_bytes));
    }
    double left = 1.0; std::vector<bool> full(n, false);
    for (std::size_t round = 0; round < n && left > 1e-12; ++round) {   // water-filling: each round at least one device fills up or all is placed
        double t = 0; for (std::size_t i = 0; i < n; ++i) if (!full[i]) t += thr[i];
        if (t <= 0) break;
        const double give = left; bool capped = false;
        for (std::size_t i = 0; i < n; ++i) if (!full[i]) {
            const double want = give * thr[i] / t, room = cap[i] - share[i];
            if (want >= room) { share[i] = cap[i]; left -= room; full[i] = true; capped = true; }
        }
        if (capped) continue;
        for (std::size_t i = 0; i < n; ++i) if (!full[i]) share[i] += give * thr[i] / t;
        left = 0;
    }
    std::vector<pipe_stage> out;
    for (std::size_t i = 0; i < n; ++i) {
        if (why) *why += cand[i]->device + " (" + name(cand[i]->cls) + (cand[i]->name.empty() ? "" : ", " + cand[i]->name) + "): share " + std::to_string(share[i]) +
                         (measured ? ", measured throughput" : ", class prior") + (full[i] && share[i] < 1.0 ? ", limited by its memory" : "") + "\n";
        if (share[i] > 1e-9) out.push_back({cand[i]->device, share[i]});
    }
    if (left > 1e-9) { out.push_back({"cpu", left}); if (why) *why += "cpu: share " + std::to_string(left) + ", what does not fit in the devices' memory\n"; }
    return out;
}
// The stages as a device string: "pipe:cuda:0*0.67,cuda:1*0.33", a single device's own name, or "cpu".
inline std::string pipeline_string(const std::vector<pipe_stage>& st) {
    if (st.empty()) return "cpu";
    if (st.size() == 1) return st[0].device;
    std::string s = "pipe:";
    for (std::size_t i = 0; i < st.size(); ++i) { char b[32]; std::snprintf(b, sizeof b, "*%.4f", st[i].share); s += (i ? "," : "") + st[i].device + b; }
    return s;
}

} // namespace forge::gpu
