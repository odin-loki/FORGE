// CUDA / HIP runtime. `test_gpu images` checks the embedded kernel images (runs everywhere);
// `test_gpu cuda|hip` runs every kernel on the device against the reference (exit 77 = skip, no device).
#include <forge/flow/flow.hpp>
#include <forge/grid/cuda.hpp>
#include <cstdio>
#include <cstring>
#include <string>

using namespace forge;
namespace gpu = forge::grid::gpu;

static const char* kernel_ops[] = {"neg", "abs", "sqrt", "rsqrt", "recip", "square", "exp", "exp2", "log", "log2", "tanh", "sigmoid",
    "silu", "relu", "softplus", "gelu_tanh", "add", "sub", "mul", "div", "maximum", "minimum", "silu_mul", "scale", "softmax",
    "rms_norm", "matmul_t"};

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

static int device(gpu::api a) {
    std::string why; auto rt = gpu::runtime::open(a, &why);
    if (!rt) { std::printf("SKIP: %s\n", why.c_str()); return 77; }
    std::printf("%s device: %s (image %s)\n", gpu::api_name(a), rt->device_name().c_str(), rt->image_arch().c_str());
    auto reg = flow::gpu_registry(std::shared_ptr<gpu::runtime>(std::move(rt)));
    int fails = 0, n = 0;
    for (std::size_t i = 0; i < spec::op_count; ++i) for (auto& l : reg->lowerings(spec::op(i))) {
        if (l.name.rfind(gpu::api_name(a), 0) != 0) continue;
        std::string msg; ++n;
        if (!flow::registry::oracle_check(l, &msg)) { std::printf("FAIL %s: %s\n", l.name.c_str(), msg.c_str()); ++fails; }
    }
    std::printf("%s %d/%d %s lowerings pass the oracle\n", fails ? "FAIL" : "ok  ", n - fails, n, gpu::api_name(a));
    return fails ? 1 : 0;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "images";
    if (mode == "images") return images();
    if (mode == "cuda") return device(gpu::api::cuda);
    if (mode == "hip") return device(gpu::api::hip);
    std::fprintf(stderr, "usage: test_gpu [images|cuda|hip]\n"); return 2;
}
