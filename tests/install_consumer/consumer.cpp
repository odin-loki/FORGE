// C++ consumer of an installed FORGE: headers from several layers (which pull in the generated and the vendored
// headers) plus the compiled simd kernels, through forge::inference and forge::simd_mv.
// Usage: consumer_cpp [expected-template-dir]. With the argument, and installed in <prefix>/bin, it also checks that
// the import templates are found relative to the executable.
#include <forge/os/os.hpp>
#include <forge/simd/dispatch.hpp>
#include <forge/import/hf.hpp>
#include <forge/inference/inference.hpp>
#include <cstdio>
#include <filesystem>
#include <vector>

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    auto exe = forge::os::exe_path();
    if (exe.empty() || !forge::os::exists(exe)) { std::puts("os::exe_path() failed"); return 1; }

    const forge::simd::kernels& k = forge::simd::active();
    std::vector<float> x(1000, 1.f), y(1000, 2.f);
    k.saxpy(3.f, x.data(), y.data(), x.size());
    if (k.dot(x.data(), y.data(), x.size()) != 5000.f) { std::puts("simd kernel gave a wrong result"); return 2; }

    forge::inference::sampler_config sc; sc.temperature = 0.f;
    forge::inference::sampler smp(sc);
    (void)smp;

    auto td = forge::import::template_dir();
    std::printf("consumer_cpp: simd target %s, exe %s, templates %s\n", k.target, exe.c_str(), td.string().c_str());
    if (argc > 1) {
        std::error_code ec;
        if (!fs::equivalent(td, fs::path(argv[1]), ec)) { std::printf("template_dir() is not %s\n", argv[1]); return 3; }
        if (forge::import::read_file(td / "llama.fbp").empty()) { std::puts("llama.fbp is empty"); return 4; }
    }
    return 0;
}
