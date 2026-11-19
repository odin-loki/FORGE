#pragma once
// Glue between forge::ann::gpu_flat_l2 and a device registry (graph::device_registry("cuda"), "vulkan", ...): the matrix product
// y = x . w^T runs through the registry's own matmul_t lowering (the one graph::bench_matmul times); the data crosses host memory
// on every call. A registry without a device lowering for it (or a CPU-only registry, null) gives the CPU lowering.
#include <forge/ann/gpu_flat.hpp>
#include <forge/graph/vulkan.hpp>

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
} // namespace forge::graph
