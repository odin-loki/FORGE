# flow: graph IR, compilation and execution

## Concepts
flow treats a tensor graph as a program that is compiled once and replayed:

1. **Build** a `flow::graph` (or let `mind` build one from a model's blueprint).
2. **Optimise**: verify, constant folding, algebraic simplification, CSE, dead-code elimination, then fusion (for example `silu(x) * y` → `silu_mul`).
3. **Plan memory**: liveness analysis gives static arena offsets, view aliasing and in-place reuse, so peak memory is known before running.
4. **Lower** each op to an implementation from a lowering registry (CPU by default; Vulkan, CUDA/HIP, OpenCL, Level Zero, Metal, RPC or a multi-device pipeline when a device is selected).
5. **Replay** the flat plan. `mind` caches plans per shape bucket.

A lowering is only used if it agrees with the scalar reference in `spec/` within the op's tolerance (the oracle, `verify/`). Ops a device does not support fall back to the CPU.

## API (`flow/include/forge/flow/*.hpp`)
- `graph` (`ir.hpp`): `input(name, shape)`, `scalar(v)`, `output(name, value)`; helpers `add`, `mul`, `matmul`; any spec op via `apply(spec::op::X, {inputs}, attrs)`.
- `verify_graph(g)` throws with the offending node (e.g. `node %2 (matmul)`); `optimise(g, pass_options)` runs the passes. `pass_options{cse, fold, simplify, fuse}` all default to `true`; `lowerings` (registry, `nullptr` = CPU).
- `compile(graph, state_store* = nullptr, pass_options = {})` → `std::unique_ptr<plan>` (`plan.hpp`).
- `executor(plan&, state_store* = nullptr)`: `set_input(name, data)`, `run()`, `output(name)` (tensor copy), `output_view(name)`.
- `grad(g, loss, wrt)` (`autodiff.hpp`): reverse-mode autodiff; appends the backward graph and returns a map from each `wrt` value to its gradient. Supported ops are listed in the header.
- `codecs.hpp`: weight codecs `f32 f16 bf16 fp8_e4m3 fp8_e5m2 q8_0 q4_k q4_g32 q8_g32 mxfp8 mxfp6 mxfp4 mxint8 nvfp4 tq2`.
- `kv_codec.hpp`: KV cache codecs via `kv_codec_from_name`: `f32` (also `""`, `none`), `int8` (`i8`, `q8`), `fp8` (`fp8_e4m3`, `e4m3`). int8/fp8 store a per-(row, head) f32 scale and are decoded on the fly by the attention kernels.
- `device_registry(name)` (`vulkan.hpp`): lowering registry for a device string: `vulkan`, `cuda[:N]`, `hip[:N]`, `opencl[:N]`, `ze[:N]`/`level_zero[:N]`, `metal[:N]`, `rpc:host:port`, or `pipe:<dev>[*w],<dev>...` to split layers across devices by weight.
- `rpc.hpp`: remote device over TCP. Start the server with `forge rpc-serve [--port 7000] [--bind 127.0.0.1] [--device cpu]`. Every op is still oracle-checked on the client. Weights are sent with every op (no remote residency). **No authentication or encryption: trusted networks only.**

## Example
`docs/examples/flow_graph.cpp` (ctest `doc_example_flow_graph`) checks `silu(1)·exp(1)` and that the gradient at x = 0 is 0.5:

```cpp
graph g;
auto x = g.input("x", {2, 4});
auto y = g.mul(g.apply(spec::op::silu, {x}), g.apply(spec::op::exp, {x}));
g.output("y", y);
auto flat = g.apply(spec::op::reshape, {y}, {{"shape", std::vector<i64>{8}}});
auto loss = g.apply(spec::op::reduce_sum, {flat}, {{"axis", 0.0}});
auto grads = grad(g, loss, {x});
g.output("dx", grads[x]);
auto plan = compile(g);
executor ex(*plan);
ex.set_input("x", in);
ex.run();
tensor out = ex.output("y");
```

## Platform notes and limits
- Link `forge_flow`, `forge_verify` and `forge_lane_mv`.
- GPU lowerings other than Vulkan have only run against mock drivers; see [grid.md](grid.md).
