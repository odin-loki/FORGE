# grid: devices, graphs and parallel primitives

## Concepts
- **Graph first.** Record copies and kernel launches into a `grid::graph`, `compile` it once for a device, then `run()` the resulting executable as often as needed.
- **Kernels** are function objects whose call operator is marked `FORGE_KERNEL`. They take a `grid::item<3>` (`it.global(d)` gives the global index in dimension `d`) followed by buffer arguments.
- **Backends are loaded at run time.** Nothing links against a vendor SDK; a runtime that cannot be loaded is reported as absent and work stays on the CPU.

There are two levels of GPU support:
1. The **graph API** (`grid.hpp`) executes on the CPU backend.
2. The **op runtimes** (`vulkan.hpp`, `cuda.hpp` for CUDA and HIP, `opencl.hpp`, `levelzero.hpp`, `metal.hpp`) run individual spec ops with kernels generated from `spec/ops` by `spec/gen/gpu_gen.py` (33 kernels: unary/binary elementwise, scale, silu_mul, softmax, rms_norm, matmul_t, rope, gather, attention, slice). These are what model execution uses through flow (select with `FORGE_DEVICE`, see [configuration.md](configuration.md)). They keep tensors resident on the device between ops and batch submissions.

## API (`grid/include/forge/grid/grid.hpp`, `prims.hpp`)
- `probe_backends()` returns one `backend_probe {kind, available, detail}` per `backend_kind`: `cpu`, `cuda`, `hip`, `level_zero`, `vulkan`, `metal`. `name(kind)` gives the string. (OpenCL is not in this list; it is probed by its own runtime.)
- `select(prefer = prefer::any)` (`any`, `discrete`, `cpu`) currently always returns the first device, i.e. the CPU.
- `device::alloc<T>(n)` → `buffer<T>`; `device::caps()` → capabilities, including `compute_units`.
- `graph::copy(src, dst)`, `graph::launch(kernel, extent, args...)` return nodes; `node.after(other)` orders them; `graph::compile(device)` → `executable` with `run()` and `steps()`.
- `grid::prims` (parallel CPU algorithms): `reduce`, `scan(x, y, inclusive = true)`, `compact(x, out, pred)`, `radix_sort(keys, vals = {})`, `histogram`, `segmented_scan`, `segmented_reduce`, `bitonic_sort`, `top_k`, `gemm_nt`, `philox_uniform`, and a `hash_map`.
- Op runtimes expose `supports(op, shapes, attrs)` and `run_op(...)`.
- `interop.hpp`: share memory with an application's renderer: export/import Vulkan buffers and semaphores as OS handles, and import them into CUDA.

## Example
`docs/examples/grid_graph.cpp` (ctest `doc_example_grid_graph`):

```cpp
struct Scale {
    float a;
    FORGE_KERNEL void operator()(grid::item<3> it, grid::span<float> y) const {
        if (auto i = it.global(0); i < i64(y.size())) y[std::size_t(i)] *= a;
    }
};
auto dev = grid::select();
auto buf = dev.alloc<float>(n);
grid::graph g;
auto up = g.copy(grid::span<const float>(host), buf);
auto k  = g.launch(Scale{3.f}, grid::extent{i64(n)}, buf); k.after(up);
g.copy(buf, grid::span<float>(back)).after(k);
auto exe = g.compile(dev);
exe.run();
grid::prims::scan<int>(x, y, true);   // inclusive prefix sum
```

## Platform notes and limits
- Link `forge_grid` and `forge_lane_mv`.
- **Vulkan** needs `glslangValidator` and Vulkan headers at build time (otherwise it is compiled out even if `forge info` finds `libvulkan`). It has run on Mesa lavapipe and SwiftShader (software devices) only. A pipeline cache is kept on disk (`FORGE_CACHE_DIR`, `FORGE_VK_PIPELINE_CACHE=0` disables).
- **CUDA / HIP** kernel images (PTX sm_70+, HSACO for `FORGE_HIP_ARCHS`) are built with clang at build time; without a suitable clang the runtimes report "no embedded kernel image".
- **CUDA, HIP, OpenCL, Level Zero and Metal have never run on real hardware.** They are exercised against mock driver libraries (`tests/mock_*.c`) and compile checks only. Graphics interop's CUDA import is likewise untested on hardware; the Vulkan export/import round trip has run on lavapipe.
