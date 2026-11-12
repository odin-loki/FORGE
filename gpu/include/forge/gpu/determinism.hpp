#pragma once
// Determinism mode for the device runtimes (docs/gpu.md, "Determinism mode").
//
// On: every backend takes only order-stable paths. The reduction order of each kernel then depends on the shapes
// alone, never on the grid a launch happens to get, the number of resident blocks, the free memory, a tuning
// variable or the host thread count. Guarantee: the same build on the same device and driver gives bit-identical
// results for the same inputs, run after run. It does not make two devices, two backends or two builds agree.
//
// What it turns off (each runtime reads deterministic() where it picks a kernel):
//   * CUDA/HIP: split-K and stream-K tensor-core matmuls (the k-sum order depends on the launch grid, which depends
//     on how many blocks the device holds at once), the split-K decode attention, the multi-block argmax (one block
//     scans the whole buffer instead of a merge in completion order), whole-run replay (predicted arguments).
//   * OpenCL / Level Zero / Metal: the split-over-keys decode attention.
//   * CPU: gpu::prims launches use fixed chunking (the same chunk boundaries whatever the worker count).
// Switch: FORGE_DETERMINISTIC=1 in the environment, or set_deterministic(true) (graph::session_options /
// inference sessions call it); the explicit call wins over the environment.
#include <atomic>
#include <cstdlib>
#include <string>

namespace forge::gpu {

inline std::atomic<int>& deterministic_flag() { static std::atomic<int> f{-1}; return f; }   // -1: follow the environment
inline bool deterministic() {
    int f = deterministic_flag().load(std::memory_order_relaxed);
    if (f >= 0) return f != 0;
    const char* e = std::getenv("FORGE_DETERMINISTIC");
    return e && *e && std::string(e) != "0";
}
// on: force the mode; off: force it off. reset_deterministic() goes back to following $FORGE_DETERMINISTIC.
inline void set_deterministic(bool on) { deterministic_flag().store(on ? 1 : 0, std::memory_order_relaxed); }
inline void reset_deterministic() { deterministic_flag().store(-1, std::memory_order_relaxed); }

} // namespace forge::gpu
