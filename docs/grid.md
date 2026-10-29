# grid: devices, graphs and parallel primitives

## Concepts
- **Graph first.** You record copies and kernel launches into a `grid::graph`. You then `compile` it once for a device and `run` the resulting executable as often as needed.
- **Kernels** are function objects marked `FORGE_KERNEL`. Each one takes an `item<3>` (use `global(d)` to get the global index) followed by its buffer arguments.
- **Backends** are found at run time by dynamic loading, so nothing links against a vendor runtime. A backend whose runtime is missing reports itself as absent.

## API (`grid/include/forge/grid/grid.hpp`, `prims.hpp`)
- `probe_backends()` returns 6 `backend_probe` entries: cpu, cuda, hip, level_zero, vulkan and metal. Each has `kind`, `available` and `detail`. `name(backend_kind)` returns the backend's name.
- `select(prefer = prefer::any)`: `prefer` is `any`, `discrete` or `cpu`. Today it returns `all().front()`, which is the CPU device.
- `device::alloc<T>(n)` allocates a `buffer<T>`. `device::caps()` returns the device capabilities, including `compute_units`.
- `graph::copy(src, dst)` and `graph::launch(kernel, extent, args...)` return nodes. Use `.after(node)` to order them. `graph::compile(device)` returns an `executable`, which provides `run()` and `steps()`.
- `grid::prims` (CPU, parallel):
  - `reduce`, `scan(x, y, inclusive=true)`, `compact(x, out, pred)`
  - `radix_sort(keys, vals={})`, `histogram`, `segmented_scan`, `segmented_reduce`, `bitonic_sort`, `top_k`
  - `gemm_nt`, `philox_uniform`
- Vendor headers (`vulkan.hpp`, `cuda.hpp`, `levelzero.hpp`, `metal.hpp`, `opencl.hpp`) provide `supports(op, shapes, attrs)` and `run_op(...)` for single spec ops. `interop.hpp` covers buffer sharing between devices.

## Example
`docs/examples/grid_graph.cpp` runs as the ctest `doc_example_grid_graph`:

```cpp
grid::graph g;
auto up = g.copy(grid::span<const float>(host), buf);
auto k  = g.launch(Scale{3.f}, grid::extent{i64(n)}, buf); k.after(up);
g.copy(buf, grid::span<float>(back)).after(k);
auto exe = g.compile(dev);
exe.run();
```

## Platform notes and limits
- The graph executor runs on the CPU backend only.
- Vulkan runs SPIR-V kernels generated from `spec/ops` by `spec/gen/gpu_gen.py`. It has been tested on Mesa lavapipe only.
- CUDA, HIP, Level Zero, OpenCL and Metal have only run against mock drivers: the `golden_tiny_llama_*_mock` tests. **They are untested on real hardware.**
- Link with `forge_grid` and `forge_lane_mv`.
