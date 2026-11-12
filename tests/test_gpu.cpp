#include "check.hpp"
#include <forge/gpu/prims.hpp>
#include <forge/gpu/select.hpp>
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
