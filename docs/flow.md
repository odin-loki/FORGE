# flow: graph IR, compilation and execution

## Concepts
A tensor graph is treated as a program: it is compiled once and replayed. The pipeline is:

1. Build the graph with `flow::graph`.
2. Run the passes: verify, canonicalise (CSE, constant folding, DCE, algebraic simplification), then fuse (for example `silu(x) * y` becomes `silu_mul`).
3. Partition the graph across devices.
4. Plan memory. Liveness analysis gives static arena offsets, so peak memory is known exactly before the graph runs.
5. Lower each op to a device implementation.
6. Produce a flat plan for replay. Plans are cached per shape bucket.

A lowering only becomes eligible if it passes the oracle, which compares it against the scalar reference in `spec`.

## API (`flow/include/forge/flow/*.hpp`)
- `graph` (`ir.hpp`):
  - Inputs and outputs: `input(name, shape)`, `scalar(v)` and `output(name, value)`.
  - Ops: `add` and `mul` are named helpers; any other spec op goes through `apply(spec::op::X, {inputs}, attrs)`. `matmul` builds a matrix product.
- `verify_graph(g)` throws with the offending node, for example `node %2 (matmul)`. `optimise(g, pass_options)` runs the passes.
- `pass_options` has `{cse, fold, simplify, fuse}`, all `true` by default, plus `lowerings` (null means CPU).
- `compile(graph, state_store* = nullptr, pass_options = {})` returns a `std::unique_ptr<plan>`.
- `executor(plan&, state_store* = nullptr)` methods: `set_input(name, data)`, `run()`, `output(name)` (returns a tensor) and `output_view(name)`.
- `grad(g, loss, wrt)` (`autodiff.hpp`) appends the backward graph and returns a map from each input to its gradient value. Supported ops are listed in the header.
- `codecs.hpp` holds the weight codecs (for example Q8_0, Q4_K, FP8, group-32, MX). `kv_codec.hpp` holds the KV codecs `f32`, `int8` and `fp8` (see `kv_codec_from_name`).
- `rpc.hpp` provides the remote device `"rpc:host:port"`. `client`/`server`, or `forge rpc-serve [--port 7000] [--bind 127.0.0.1] [--device ...]`, start the remote side. **There is no authentication, so only use it on a trusted network.**
- `vulkan.hpp` registers `vk.<op>` lowerings for every op with a generated GPU kernel. Any other op falls back to the CPU.

## Example
`docs/examples/flow_graph.cpp` runs as the ctest `doc_example_flow_graph`. It checks the result of `silu(1) * exp(1)` and checks the gradient at x = 0, which is 0.5.

```cpp
graph g;
auto x = g.input("x", {2, 4});
g.output("y", g.mul(g.apply(spec::op::silu, {x}), g.apply(spec::op::exp, {x})));
auto plan = compile(g);
executor ex(*plan);
ex.set_input("x", in);
ex.run();
tensor y = ex.output("y");
```

## Platform notes and limits
- Link with `forge_flow`, `forge_verify` and `forge_lane_mv`.
- GPU lowerings other than Vulkan have only run against mock drivers. See [grid.md](grid.md).
