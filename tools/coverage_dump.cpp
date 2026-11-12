// forge_coverage_dump: the op x backend lowering table as the built code sees it, as JSON on stdout.
// tools/gen_docs.py derives the same table from the sources for docs/reference.md (no build needed, and the same
// on every platform); the `coverage_table` ctest runs `gen_docs.py --check-coverage <this output>` so the two can
// not drift apart.
//   cpu      op -> names of the lowerings registry::cpu() holds for it (walked from the live registry)
//   gpu      ops with a kernel in the CUDA / HIP / OpenCL / Level Zero / Metal runtimes (gpu::cuda_hip::kernel_sig,
//            the test graph::gpu_registry registers by)
//   vulkan   ops with an embedded SPIR-V kernel; null in a build without the Vulkan backend
//   codecs   weight codecs (graph/codecs.hpp)
#include <forge/graph/vulkan.hpp>
#include <cstdio>
#include <string>

int main() {
    using namespace forge;
    auto& cpu = graph::registry::cpu();
    std::string cpu_j, gpu_j, vk_j, codec_j;
    for (std::size_t i = 0; i < ops::op_count; ++i) {
        auto o = ops::op(i);
        std::string name(ops::info(o).name), names;
        for (auto& l : cpu.lowerings(o)) names += (names.empty() ? "\"" : ", \"") + l.name + "\"";
        cpu_j += (cpu_j.empty() ? "\n  \"" : ",\n  \"") + name + "\": [" + names + "]";
        gpu::cuda_hip::sig s;
        if (gpu::cuda_hip::kernel_sig(name, s)) gpu_j += (gpu_j.empty() ? "\"" : ", \"") + name + "\"";
#ifdef FORGE_GPU_VULKAN
        // a spec shader of the op's name, or the GLSL port of an encoder / vision kernel (vulkan_registry, run_nn)
        const bool nn = gpu::cuda_hip::kernel_sig(name, s) && gpu::cuda_hip::nn_sig(s) && gpu::vk::runtime::has_kernel("forge_" + name);
        if (gpu::vk::runtime::has_kernel(name) || nn) vk_j += (vk_j.empty() ? "\"" : ", \"") + name + "\"";
#endif
    }
    for (auto& c : graph::codecs::all()) codec_j += (codec_j.empty() ? "\"" : ", \"") + std::string(c.name) + "\"";
#ifdef FORGE_GPU_VULKAN
    vk_j = "[" + vk_j + "]";
#else
    vk_j = "null";
#endif
    std::printf("{\n \"cpu\": {%s\n },\n \"gpu\": [%s],\n \"vulkan\": %s,\n \"codecs\": [%s]\n}\n", cpu_j.c_str(), gpu_j.c_str(), vk_j.c_str(), codec_j.c_str());
    return 0;
}
