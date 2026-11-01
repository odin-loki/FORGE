// CUDA / HIP runtime. `test_gpu images` checks the embedded kernel images (runs everywhere);
// `test_gpu cuda|hip` runs every kernel on the device against the reference (exit 77 = skip, no device).
#include <forge/flow/flow.hpp>
#include <forge/grid/cuda.hpp>
#include <forge/flow/vulkan.hpp>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>

using namespace forge;
namespace gpu = forge::grid::gpu;

static const char* kernel_ops[] = {"neg", "abs", "sqrt", "rsqrt", "recip", "square", "exp", "exp2", "log", "log2", "tanh", "sigmoid",
    "silu", "relu", "softplus", "gelu_tanh", "add", "sub", "mul", "div", "maximum", "minimum", "silu_mul", "scale", "softmax",
    "rms_norm", "matmul_t", "rope", "gather", "slice", "attention", "qmatmul_t"};

static bool has(std::string_view hay, const std::string& needle) { return hay.find(needle) != std::string_view::npos; }

static int images() {
    int fails = 0;
    auto ptx = gpu::module_images(gpu::api::cuda);
    auto hip = gpu::module_images(gpu::api::hip);
    if (ptx.empty()) { std::puts("SKIP images: built without a GPU-capable clang"); return 77; }
    for (auto& [arch, img] : ptx) {
        bool ok = has(img, ".version") && has(img, ".target sm_70");
        for (auto op : kernel_ops) if (!has(img, std::string(".entry forge_") + op + "(")) { std::printf("FAIL ptx: no entry forge_%s\n", op); ok = false; }
        std::printf("%s ptx: %zu bytes, %zu kernels\n", ok ? "ok  " : "FAIL", img.size(), std::size(kernel_ops)); fails += !ok;
    }
    for (auto& [arch, img] : hip) {
        bool elf = img.size() > 64 && std::memcmp(img.data(), "\x7f" "ELF", 4) == 0;
        std::uint16_t machine = 0; if (elf) std::memcpy(&machine, img.data() + 18, 2);
        bool ok = elf && machine == 224;   // EM_AMDGPU
        for (auto op : kernel_ops) if (!has(img, std::string("forge_") + op + ".kd")) { std::printf("FAIL %s: no kernel descriptor forge_%s\n", arch.c_str(), op); ok = false; }
        std::printf("%s hsaco %s: %zu bytes, ELF machine %u\n", ok ? "ok  " : "FAIL", arch.c_str(), img.size(), unsigned(machine)); fails += !ok;
    }
    if (hip.empty()) std::puts("note: no HIP code objects (no lld at build time)");
    for (auto a : {gpu::api::cuda, gpu::api::hip}) {   // without a driver this must fail cleanly, not crash
        std::string why; auto rt = gpu::runtime::open(a, &why);
        std::printf("     %s runtime: %s\n", gpu::api_name(a), rt ? ("device " + rt->device_name()).c_str() : why.c_str());
    }
    std::string why; auto reg = flow::device_registry("cuda", &why);
    if (!reg && why.empty()) { std::puts("FAIL device_registry(cuda) gave no reason for the fallback"); ++fails; }
    return fails ? 1 : 0;
}

// Quantised matmul_t kept encoded on the device vs the CPU quantised kernel (flow::qweights::gemm_nt).
template<class RT> static int qmatmul_check(flow::registry& reg, const std::string& api) {
    if (!reg.device_qgemm) { std::printf("note %s: no device quantised matmul\n", api.c_str()); return 0; }
    int fails = 0, n = 0;
    for (std::string cn : {"q8_0", "q4_k", "q5_k", "q6_k"}) for (std::size_t rows : {1, 3, 6, 16, 70, 130}) {   // >= 16 rows: the tiled MMQ kernel
        const auto* c = flow::codecs::find(cn);
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
        reg.device_qgemm->fn(ins, og, {});
        flow::qweights::gemm_nt(ins, oc, {});
        double worst = 0;
        for (std::size_t i = 0; i < yc.size(); ++i) worst = std::max(worst, std::fabs(double(yg[i]) - yc[i]) / (1.0 + std::fabs(double(yc[i]))));
        ++n; if (worst > 1e-4) { std::printf("FAIL %s.qmatmul_t %s rows %zu: worst rel error %.3g\n", api.c_str(), cn.c_str(), rows, worst); ++fails; }
    }
    std::printf("%s %d/%d %s quantised matmul_t cases match the CPU kernel\n", fails ? "FAIL" : "ok  ", n - fails, n, api.c_str());
    return fails;
}

// Gated DeltaNet state ops on the device (run_state) vs the CPU executor: 3-token prefill, then token by token,
// then reset() and again (the resident state must restart from zeros).
static int state_ops_check(flow::registry& reg, const std::string& api) {
    if (!reg.hooks || !reg.hooks->state) { std::printf("note %s: no device state ops\n", api.c_str()); return 0; }
    using namespace flow;
    const i64 S = 7, Hk = 2, Hv = 4, dk = 40, dv = 136, C = 150, K = 4;   // dv > 128 and C > 128: multi-column threads / blocks
    auto rnd = [](std::size_t n, float ph) { std::vector<float> v(n); for (std::size_t i = 0; i < n; ++i) v[i] = std::sin(float(i) * 0.37f + ph); return v; };
    auto q = rnd(S * Hk * dk, 1), k = rnd(S * Hk * dk, 2), v = rnd(S * Hv * dv, 3), gl = rnd(S * Hv, 4), bl = rnd(S * Hv, 5), xc = rnd(S * C, 6), wc = rnd(C * K, 7);
    for (auto& x : gl) x = -std::fabs(x) * 0.5f;
    for (auto& x : bl) x = 1.f / (1.f + std::exp(-x));
    auto build = [&](i64 s) {
        graph g;
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
    std::printf("%s %s delta_rule/causal_conv on the device match the CPU (worst rel %.3g)\n", ok ? "ok  " : "FAIL", api.c_str(), worst);
    return ok ? 0 : 1;
}

// Every runtime with the cuda.hpp contract: open(why, ordinal) via the Open functor.
template<class RT, class Open> static int device_t(Open open) {
    std::string why; auto rt = open(&why, 0);
    if (!rt) { std::printf("SKIP: %s\n", why.c_str()); return 77; }
    const std::string api = rt->backend();
    std::printf("%s device: %s (image %s)\n", api.c_str(), rt->device_name().c_str(), rt->image_arch().c_str());
    auto reg = flow::gpu_registry(std::shared_ptr<RT>(std::move(rt)));
    int fails = 0, n = 0;
    for (std::size_t i = 0; i < spec::op_count; ++i) for (auto& l : reg->lowerings(spec::op(i))) {
        if (l.name.rfind(api + ".", 0) != 0) continue;
        std::string msg; ++n;
        if (!flow::registry::oracle_check(l, &msg)) { std::printf("FAIL %s: %s\n", l.name.c_str(), msg.c_str()); ++fails; }
    }
    std::printf("%s %d/%d %s lowerings pass the oracle\n", fails ? "FAIL" : "ok  ", n - fails, n, api.c_str());
    fails += qmatmul_check<RT>(*reg, api);
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

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "images";
    if (mode == "images") return images();
    auto cu = [](gpu::api a) { return [a](std::string* w, int o) { return gpu::runtime::open(a, w, o); }; };
    auto cl = [](std::string* w, int o) { return forge::grid::ocl::runtime::open(w, o); };
    auto ze = [](std::string* w, int o) { return forge::grid::ze::runtime::open(w, o); };
    if (mode == "cuda") return device_t<gpu::runtime>(cu(gpu::api::cuda));
    if (mode == "hip") return device_t<gpu::runtime>(cu(gpu::api::hip));
    if (mode == "opencl") return device_t<forge::grid::ocl::runtime>(cl);
    if (mode == "ze") return device_t<forge::grid::ze::runtime>(ze);
    if (mode == "metal") return device_t<forge::grid::metal::runtime>([](std::string* w, int o) { return forge::grid::metal::runtime::open(w, o); });
    if (mode == "features") {
        std::string w = argc > 2 ? argv[2] : "cuda";
        if (w == "opencl") return features_t(cl, [] { return forge::grid::ocl::runtime::device_count(); });
        if (w == "ze") return features_t(ze, [] { return forge::grid::ze::runtime::device_count(); });
        auto a = w == "hip" ? gpu::api::hip : gpu::api::cuda;
        return features_t(cu(a), [a] { return gpu::runtime::device_count(a); });
    }
    std::fprintf(stderr, "usage: test_gpu [images|cuda|hip|opencl|ze|features <api>]\n"); return 2;
}
