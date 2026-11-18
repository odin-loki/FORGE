#pragma once
// Glue between forge::ann::gpu_flat_l2 and a device registry (graph::device_registry("cuda"), "vulkan", ...): the matrix product
// y = x . w^T runs through the registry's own matmul_t lowering (the one graph::bench_matmul times); the data crosses host memory
// on every call. A registry without a device lowering for it (or a CPU-only registry, null) gives the CPU lowering.
#include <forge/ann/gpu_flat.hpp>
#include <forge/ann/gpu_knn.hpp>
#include <forge/ann/ivf.hpp>
#include <forge/graph/vulkan.hpp>
#include <cstdlib>
#include <memory>
#include <string>

namespace forge::graph {
inline ann::gpu_flat_l2::matmul_fn ann_matmul(registry* reg) {
    return [reg](const float* x, std::size_t rows, std::size_t k, const float* w, std::size_t cols, float* y) -> bool {
        registry& r = reg ? *reg : registry::cpu();
        std::vector<shape_t> ss{{i64(rows), i64(k)}, {i64(cols), i64(k)}}; shape_t os{i64(rows), i64(cols)};
        const lowering* low = nullptr;
        for (auto& l : r.lowerings(ops::op::matmul_t)) if ((!l.supports || l.supports(ss, os, {})) && (!low || l.priority > low->priority)) low = &l;
        if (!low) return false;
        std::vector<tensor_view> in{{const_cast<float*>(x), ss[0]}, {const_cast<float*>(w), ss[1]}}; tensor_view out{y, os};
        try { low->fn(in, out, {}); } catch (...) { return false; }
        return true;
    };
}
// Glue for forge::ann::gpu_knn_l2: the 16-nearest kernel of a CUDA / HIP runtime (gpu::cuda_hip::runtime::knn_l2_16), the rows
// resident on the device. dev: "cuda", "cuda:N", "hip", "hip:N". Null when the device cannot be opened (why says why); the
// index then answers on the CPU.
inline ann::gpu_knn_l2::knn_fn cuda_knn(std::shared_ptr<gpu::cuda_hip::runtime> rt) {
    if (!rt) return nullptr;
    return [rt](const float* x, std::size_t n, std::size_t d, const float* q, std::size_t nq, float* od, std::uint32_t* oi, double* up) {
        return rt->knn_l2_16(x, n, d, q, nq, od, oi, up); };
}
// Glue for forge::ann::ivf_flat::search_batch_device: the IVF scan kernel of a CUDA / HIP runtime (runtime::ivf_l2_16).
inline ann::ivf_flat::scan_fn cuda_ivf(std::shared_ptr<gpu::cuda_hip::runtime> rt) {
    if (!rt) return nullptr;
    return [rt](const float* x, std::size_t n, std::size_t d, const float* q, std::size_t nq, const std::uint32_t* rng, std::size_t np, float* od, std::uint32_t* oi, double* up) {
        return rt->ivf_l2_16(x, n, d, q, nq, rng, np, od, oi, up); };
}
inline std::shared_ptr<gpu::cuda_hip::runtime> open_knn_device(const std::string& dev = "cuda", std::string* why = nullptr) {
    auto base = dev.substr(0, dev.find(':'));
    if (base != "cuda" && base != "hip") { if (why) *why = "knn: not a CUDA/HIP device: " + dev; return nullptr; }
    const int ord = dev.size() > base.size() ? std::atoi(dev.c_str() + base.size() + 1) : 0;
    return std::shared_ptr<gpu::cuda_hip::runtime>(gpu::cuda_hip::runtime::open(base == "cuda" ? gpu::cuda_hip::api::cuda : gpu::cuda_hip::api::hip, why, ord));
}
} // namespace forge::graph
