# graph: graph IR, compilation and execution

## Concepts
graph treats a tensor graph as a program that is compiled once and replayed:

1. **Build** a `tensor_graph` (or let [inference](inference.md) build one from a model's blueprint).
2. **Optimise** (`passes.hpp`): verify shapes, then CSE, constant folding, algebraic simplification, dead-code elimination, and fusion: `silu(a) * b` → `silu_mul(a, b)` when `silu(a)` has no other users, and a `batch_norm` folded into the convolution in front of it. Host plans get two more fusions after lowering (step 4). All of them are listed under [Fusion](#fusion).
3. **Plan memory** (`memplan.hpp`): liveness analysis gives static arena offsets, view aliasing (`reshape`) and in-place reuse, so peak memory is known before running.
4. **Lower** each op to an implementation from a lowering registry: CPU by default, or a device registry (Vulkan, CUDA, HIP, OpenCL, Level Zero, Metal, RPC, or a multi-device pipeline).
5. **Replay** the flat plan with an `executor`. inference caches plans per shape bucket (`plan_cache`).

Steps 2 to 4 can be done ahead of time: `forge aot` stores compiled plans in the model file, or compiles them to native code, and a session then rebuilds the plan without running the passes ([aot-models.md](aot-models.md)). `compile()` is split for that: `lower_plan(plan&, state_store*, pass_options, pick)` is its second half (buffer binding and kernel selection), shared by both paths.

A lowering is used only if it agrees with the float64 scalar reference in `ops/` within the op's declared tolerance (the oracle, `verify/`). The check runs the first time the lowering is selected; a lowering that fails is disabled with a warning. Ops or shapes a device does not support run on the CPU. On a device run, `FORGE_PLAN_TRACE` lists those host ops at exit.

Quantised weights (`tensor_graph::qweight`) stay encoded. On the CPU, `matmul_t` and `gather` read the encoded rows in place, so a q4_k model costs its file size in memory. On a device, see [Device execution](#device-execution).

## API (`graph/include/forge/graph/*.hpp`)
`graph.hpp` includes the IR, passes, codecs, memory planner, kernels, plans and the device registries. `autodiff.hpp`, `kv_codec.hpp` and `rpc.hpp` are included separately.

| Item | Header | Notes |
|------|--------|-------|
| `tensor_graph` | `ir.hpp` | `input(name, shape)`, `weight(name, shape, data)`, `qweight(name, shape, bytes, codec)`, `constant(tensor)`, `scalar(v)`, `apply(op, {inputs}, attrs)` (op enum or name), `state_append(slot, x)`, `state_op(slot, fn, {inputs}, attrs)`, `output(name, value)`; helpers `add`, `mul`, `matmul`. `to_string(g)` prints it |
| `verify_graph(g)` | `passes.hpp` | Infers shapes; errors name the node (`node %2 (matmul)`) |
| `optimise(g, pass_options)` | `passes.hpp` | `pass_options{cse, fold, simplify, fuse}` all default to `true`; `lowerings` is the registry (`nullptr` = CPU); `fold_conv_bn`, `fuse_chain`, `fuse_qact` (default `true`) and `chain_min_elems` (4096) control the [fusions](#fusion) |
| `compile(g, state_store* = nullptr, pass_options = {})` | `plan.hpp` | → `std::unique_ptr<plan>`; `plan::describe()` prints it |
| `executor(plan&, state_store* = nullptr)` | `plan.hpp` | `set_input(name, data)`, `run()`, `output(name)` (tensor copy), `output_view(name)` |
| `aot::builder`, `aot::image`, `aot::instantiate` | `aot.hpp` | Compiled plans as a checked, versioned blob, and back ([aot-models.md](aot-models.md)). `aot_codegen.hpp` writes the straight-line C++ for native compilation; `aot_rt.hpp` is what that code calls |
| `state_store` | `plan.hpp` | Per-session slots: KV history (`state_append`) and fixed-size state (`state_op`). `reset()`, `truncate(len)`, paged growth (`page_rows`), KV codec, host offload |
| `state_op` kernels | `state_ops.hpp` | `causal_conv` (depthwise causal conv1d) and `delta_rule` (gated delta rule recurrence), for Gated DeltaNet layers (Qwen3.5) |
| `grad(g, loss, wrt)` | `autodiff.hpp` | Reverse-mode autodiff. Appends the backward graph and returns a map from each `wrt` value to its gradient. Supported ops: add sub mul div neg scale square sqrt rsqrt recip exp log tanh sigmoid silu relu abs, matmul (rank-2 b), matmul_t, transpose, reshape, reduce_sum, reduce_mean, logsumexp, softmax, rms_norm. Others raise `errc::unsupported` |
| `codecs::find(name)`, `codecs::all()` | `codecs.hpp` | Weight codecs, [below](#weight-codecs) |
| `kv_codec_from_name`, `kv_codec_pair_from_name` | `kv_codec.hpp` | KV cache codecs, below |
| `device_registry(name, &why)` | `vulkan.hpp` | Lowering registry for a device string, or `nullptr` (CPU) with the reason in `why` |
| `rpc::open(addr, &why)`, `rpc::server`, `rpc::client` | `rpc.hpp` | Remote device over TCP |

### Weight codecs
Each codec is a block layout with an encoder and a decoder (`codec{name, tag, block, block_bytes, encode, decode}`). A weight whose rows are whole blocks stays encoded and is read in place.

| Name | Elements / bytes per block | Layout | CPU row kernel |
|------|----------------------------|--------|----------------|
| `f32`, `f16`, `bf16`, `fp8_e4m3`, `fp8_e5m2` | 1 / 4, 2, 2, 1, 1 | plain elements | f32 in place; the others decode + dot |
| `q8_0` | 32 / 34 | GGUF Q8_0 | AVX2 decode, int8 kernels |
| `q4_k`, `q5_k`, `q6_k` | 256 / 144, 176, 210 | GGUF K-quants | AVX2 decode, int8 kernels |
| `q4_0`, `q5_0` | 32 / 18, 22 | GGUF: `{f16 d; [qh[4];] qs[16]}`, `x = d * (q - 8 or 16)` | scalar decode + SIMD dot |
| `q4_1`, `q5_1` | 32 / 20, 24 | GGUF: `{f16 d; f16 m; [qh[4];] qs[16]}`, `x = d * q + m` | scalar decode + SIMD dot |
| `q2_k`, `q3_k` | 256 / 84, 110 | GGUF K-quants at 2 and 3 bits, 16 sub-blocks of 16 | scalar decode + SIMD dot |
| `iq4_nl`, `iq4_xs` | 32 / 18, 256 / 136 | GGUF non-linear 4-bit: codes index a 16-entry table | scalar decode + SIMD dot |
| `tq1_0`, `tq2_0` | 256 / 54, 66 | GGUF ternary: 5 trits per byte, or 2 bits per value | scalar decode + SIMD dot |
| `q4_g32`, `q8_g32` | 32 / 20, 36 | `{f16 scale; f16 min; q[32]}`, `x = q * scale + min` (AWQ / GPTQ style groups) | scalar decode + SIMD dot |
| `mxfp8`, `mxfp6`, `mxfp4`, `mxint8` | 32 / 33, 25, 17, 33 | OCP Microscaling: one E8M0 scale per block | scalar decode + SIMD dot |
| `nvfp4` | 16 / 9 | E2M1 elements with an E4M3 block scale | scalar decode + SIMD dot |
| `tq2` | 64 / 18 | ternary, f16 absmean scale | scalar decode + SIMD dot |

The decoders of the GGUF layouts give the reference dequantiser's bits ([modelfile.md](modelfile.md#gguf-tensor-types) says what each was checked against). The encoders are FORGE's own min/max or absmax fits, not ports of ggml's search-based quantisers: `forge import --codec q3_k` gives a valid Q3_K tensor with more error than llama.cpp's quantiser would. For the K-quants a GGUF made by llama.cpp and imported with `--codec keep` is the better source.

### KV cache codecs

| Name (aliases) | Storage | Runs on |
|----------------|---------|---------|
| `f32` (`""`, `f16`, `none`) | f32 | all |
| `int8` (`i8`, `q8`) | symmetric int8, one f32 scale per (row, head) | CPU, Vulkan, CUDA/HIP |
| `fp8` (`fp8_e4m3`, `e4m3`) | E4M3 (saturating), one f32 scale per (row, head) | CPU, Vulkan, CUDA/HIP |
| `tbq4` (`tbq4_0`, `turbo4`), `tbq3` (`tbq3_0`, `turbo3`) | TurboQuant: Walsh-Hadamard rotation, then 4- or 3-bit Lloyd-Max codes and one gain per head | CPU; GPU up to head dim 256 |
| `tbq4p`, `tbq3p` (`*_qjl`, `turbo*p`) | TurboQuant_prod at the same bit budget: 3- or 2-bit Lloyd-Max codes, plus the residual's norm and one sign bit per coordinate of a seeded projection of it (QJL). Inner products against the decoded head are unbiased (measured within 1e-3 of ‖x‖² where `tbq4` / `tbq3` shrink by 1% / 3.4%), at about twice the vector MSE of `tbq4` / `tbq3` | CPU only (the GPU kernels raise `unsupported`) |
| `k8v4` | int8 keys, tbq4 values (`kv_codec_pair_from_name` only) | as its parts |

The attention kernels decode quantised slots on the fly. inference selects the codec with `session_options.kv_codec` or `FORGE_KV_CODEC` ([inference.md](inference.md)).

### Device strings

| String | Registry |
|--------|----------|
| `cpu` or empty | `nullptr`: CPU lowerings |
| `vulkan` | `vk.<op>` lowerings |
| `cuda[:N]`, `hip[:N]` | `cuda.<op>` / `hip.<op>`, plus device quantised matmul and gather |
| `opencl[:N]`, `ze[:N]` / `level_zero[:N]`, `metal[:N]` | `<backend>.<op>` |
| `rpc:host:port` | ops run by a remote `forge rpc-serve` |
| `pipe:<dev>[*w],<dev>[*w],...` | Split contiguous layer ranges across devices, balanced by cost × weight `w` (e.g. `pipe:vulkan*3,cpu`, `pipe:cuda:0,cuda:1`) |
| `pipe:auto` | The same split with the devices and weights chosen from device memory and measured throughput ([below](#automatic-split-pipeauto)) |

Device lowerings have priority 20, above the CPU ones. The backends themselves are described in [gpu.md](gpu.md).

### Automatic split: `pipe:auto`
`pipe:auto` builds the `pipe:` string itself (`gpu::plan_pipeline` in `gpu/select.hpp`, `graph::auto_pipeline`) from the device records `FORGE_DEVICE=auto` gathers (class, memory, cached micro-benchmark; [gpu.md](gpu.md#automatic-device-selection)):

1. **Devices.** The eligible devices (not in `FORGE_DEVICE_EXCLUDE`, not a software device unless `FORGE_DEVICE_ALLOW_SOFTWARE=1`) of one backend: the first of cuda, hip, metal, level_zero, opencl, vulkan that has any. Other backends' devices are left out, because one GPU is usually listed under several APIs and would be counted twice. A machine with one NVIDIA and one AMD card is therefore not split by `pipe:auto`; write that `pipe:` string by hand.
2. **Shares.** Proportional to throughput: the micro-benchmark when every chosen device has one, else the class prior.
3. **Memory.** With a known model size (`inference::model` passes the data section's size) no device gets a larger share than 90% of its memory holds; the excess moves to the devices with room, and what fits on none becomes a last `cpu` stage.
4. **Result.** Stages in device order with their shares as weights (`pipe:cuda:0*0.7429,cuda:1*0.2571`), a single device's own name when one device takes everything (a plain resident device, not a pipeline), or `cpu`. The choice and the table are logged at info level.

The share is a share of the planner's cost estimate (op cost × tensor size), which the pipeline splits into contiguous node ranges. It tracks weight bytes only for models with uniform layers, so the memory cap is approximate: there is no per-layer byte accounting. As for every `pipe:` device, the stages are non-resident and activations cross a stage boundary through host memory; device-to-device peer copies (`copy_to_peer`) are not used for the hand-off. Tested with synthetic records (`test_gpu`: `pipeline_auto_partition`) and on the mock drivers with `MOCK_DEVICES=2` (`test_gpu_select_{cuda,hip,opencl,ze}_mock`: the split, a graph run over both devices against the CPU, the one-device case; `golden_tiny_llama_pipe_auto_cuda_mock`); the peer copy itself only by `test_gpu_features_*_mock`. No multi-GPU machine has run it.

### CPU lowerings
`kernels.hpp` registers the SIMD lowerings of the decoder ops (elementwise strips, softmax, rms_norm, gemm, attention, rope, gather); `kernels_ext.hpp` adds the spatial and encoder families:

| Op | CPU lowering |
|----|--------------|
| `conv2d`, `conv1d` | im2col + the blocked gemm (`gpu::prims::gemm_nt`), one gemm per group; depthwise (one input channel per filter) runs direct loops |
| `gelu`, `quick_gelu`, `mish`, `hard_swish`, `floor` | float strips over the task pool |
| `layer_norm`, `l2_normalize` | one row per task (sums in double) |
| `attention_bias` | the plain attention loop (SIMD dot / saxpy) with the bias, soft-cap and masks; f32 keys and values only |

Every other op of those families (`conv_transpose2d`, pooling, resize, `group_norm`, `batch_norm`, `pad`, `alibi_bias`, `sinusoid_embed`, `topk`, `index_select`, `masked_mean`) runs its scalar reference: correct, not fast. The same holds for older ops without a lowering (the reductions, `transpose`, `concat`, `slice`).

`add`, `sub` and `mul` of a tensor with a row vector broadcast over the last axis (`x[..., n] op v[n]`, either operand order: a bias add, a per-channel scale) have SIMD lowerings (`lane.add_row`, `lane.mul_row`, …; before, any broadcast went to the float64 reference, one element at a time on one thread). `relu` (`cpu.relu`) and `scale` (`cpu.scale`) have float loops. Each is a single float operation per element, so the bits are the reference's (`row_vector_lowerings_match_reference_bits`). Other broadcasts still run the reference.

### Fusion

| Fusion | Where | What it does | Against the unfused result | Off switch |
|--------|-------|--------------|----------------------------|------------|
| `silu(a) * b` → `silu_mul` | graph pass (`fuse`) | One op, one kernel | within the ops' tolerances | `pass_options::fuse = false` |
| conv + `batch_norm` folding | graph pass (`fold_conv_bn`) | `batch_norm(conv2d / conv1d(x, W, b), w, bb, mean, var)` with constant f32 parameters becomes the convolution with `W' = W · s`, `b' = (b − mean) · s + bb`, `s = w / sqrt(var + eps)` (computed in double, rounded once). Needs the convolution's result, `W` and `b` to have no other user. Applies on every backend: it is a rewrite of constants | **Not bit-identical.** Stated bound: 2·10⁻⁵ of the largest output (measured 2.8·10⁻⁷ over 24 random cases), and inside the convolution's oracle tolerance (rel 10⁻⁴ of the float64 reference) | `FORGE_FUSE_CONV_BN=0`, `pass_options::fold_conv_bn` |
| Row chains | host plans, after lowering (`fuse_cpu.hpp`) | A run of adjacent steps that each compute row r of their output from row r of their inputs (elementwise ops, bias adds, `rms_norm`, `layer_norm`) is run block by block: the tensor is cut into blocks of rows of about 64 KB and each block goes through all steps while it is in cache, so the chain is one pass over memory instead of one per step. This is what `matmul_t → add(bias) → gelu / silu / relu` (the epilogue; the matmul itself is not blocked), `add → rms_norm`, `norm → mul / scale` and longer elementwise expressions become | **Bit-identical**: a block is run by calling each step's own lowering on row slices of its views | `FORGE_FUSE_CHAIN=0`, `pass_options::fuse_chain`; tensors below `chain_min_elems` (4096 elements) are left alone |
| Shared int8 activations | host plans, after lowering | Quantised `matmul_t` on the integer path quantises its activation rows to int8 first. Steps that multiply the same activation (q / k / v after one norm; an FFN's gate and up) share that: the first to run quantises into a buffer the plan owns, the others reuse it | **Bit-identical** (quantisation is a function of the row) | `FORGE_FUSE_QACT=0`, `pass_options::fuse_qact` |

`FORGE_FUSE=0` turns off all but the first. The last two do not rewrite anything: the step list, the lowerings and the buffers stay as they are and steps are only marked (`step::chain`, `step::qact`; `plan::describe()` prints `fused: …` lines). That is why they cannot interfere with the other paths:

- **Devices.** They are applied only to plans lowered from the CPU registry (no device hooks, no pipeline stages). On a device, over RPC, and in every stage of a `pipe:` split (CPU stages included), each op lowers and runs exactly as before; the device-side pair fusions (`fuse_steps`, below) are separate and unchanged.
- **Precompiled plans** (`forge aot`) store the same steps with or without them; `aot::pipeline_version` is 3 because of the new lowerings and the conv folding. Natively compiled plans run their steps unfused (the compiled code is a kernel call per step).
- **Memory.** The memory plan was made for the unfused order. A chain interleaves its steps, so it is formed only when every pair of buffers it touches is disjoint or the same buffer used row for row (an in-place step).

What a chain is, exactly: step t + 1 takes step t's output as a whole-tensor operand; every other operand is a whole tensor of the same size (sliced by rows with it), a row vector of the last dimension, or a scalar; any other broadcast ends the chain. An intermediate with other users is still written in full. A chain of only elementwise steps with no row vector is cut by elements, not rows.

Not fused (and why): the matmul with its epilogue inside the GEMM loop (the epilogue is a separate block-wise pass over the output); `rope` with the `attention` after it (two kernels, as before; devices have the `rms_rope` pair); `layer_norm` followed by a separate scale only fuses as a row chain, the scale is not folded into the norm's weights; chains on devices. No measurement of the row chains' effect on a real model was made (the tiny / wide synthetic models have neither biases nor separate activations after their matmuls, so `forge bench` on them does not exercise it).

Tests (`test_graph`): `fusion_row_chains_are_bit_identical` (60 random graphs of 13 unary ops, 5 binary ops, bias adds, row-vector scales, `rms_norm`, `layer_norm`, a `matmul_t` head and intermediates with a second user: outputs `memcmp`-equal to the plan compiled without fusion, over two runs, and equal to the unoptimised pipeline; the three switches), `fusion_shared_int8_activations` (q4_k, q8_0, q6_k; 1, 5 and 40 rows; three runs), `fusion_conv_bn_folds` (24 random conv2d / conv1d cases with stride, padding, groups and a `[1]` bias; the cases that must not fold).

A growing state view (`state_append`) may only feed `attention` and `attention_bias`. `attention_bias` reads f32 rows, so a graph that feeds it from a state slot is refused at plan time when the store uses a quantised KV codec.

## Device execution
Device registries have lowerings for 42 ops, 45 on Vulkan ([reference.md](reference.md)), since 2026-11-12 including `gelu`, `layer_norm`, `group_norm`, `conv1d`, `conv2d`, `max_pool2d`, `avg_pool2d` and `attention_bias` ([gpu.md](gpu.md#encoder-and-vision-kernels)). A step whose op has no device kernel (`transpose`, `l2_normalize`, `masked_mean`, the resize and transposed-convolution ops, `batch_norm`, ...) is lowered to the CPU: the executor fetches its inputs to the host, runs it there and reports the write back to the device. The mock-driver goldens `golden_tiny_{gpt2,gemma,phi3,bert,vit,whisper}_{cuda,opencl}_mock` and their `_vulkan` versions run such mixed plans; `FORGE_PLAN_TRACE` lists the host steps.

A resident device registry carries `device_hooks` (`kernels.hpp`) that the executor calls: `begin_run`/`end_run`, `pin` (weights and constants, once), `to_host`, `host_wrote`, `release`, device-side `copy`/`copy2d` (reshape and concat stay on the device), and optionally `arena`, `keep`, `end_run_discard`, `state`, `ran`/`replay`, `argmax` and `fused`. With `fused`, `compile` marks adjacent step pairs the backend may run as one kernel (`fuse_steps`: a residual `add` followed by the `rms_norm` of its result; an `rms_norm` whose only consumer is the next step, a `rope`); the executor offers each pair to the hook and runs the two steps as usual when it declines. The host-plan fusions of [Fusion](#fusion) are never applied to a device plan. Which hooks a backend provides, and the environment variables that turn them off, are in [gpu.md](gpu.md#environment).

Quantised weights on a device:
- CUDA and HIP keep q8_0, q4_k, q5_k and q6_k weights encoded on the device and run them with their own kernels.
- Other resident backends get an f32 copy, decoded on the host and mirrored on the device, up to `FORGE_GPU_WEIGHT_BUDGET` bytes (default `8G`; K/M/G suffixes). Weights beyond the budget stay encoded and their `matmul_t`/`gather` run on the CPU.

Sparse MoE (`moe_ffn(h, router_logits, gate[e,f,d], up[e,f,d], down[e,d,f], k, norm, scale, gating)`): each token is routed to its top-k experts; the (token, slot) pairs are grouped by expert and only the selected experts' slices of the stacked (encoded) weights are read, with the same SIMD quantised dot / row-decode kernels as `matmul_t`; tokens are combined in slot order (deterministic). Expert weights always stay encoded. On CUDA, HIP, OpenCL, Level Zero and Vulkan `moe_ffn` runs on the device (`registry::device_moe`): the router's top-k is computed on the host from the synced logits with the same `moe_route`, and two kernels evaluate the routed experts from the stacked weights, which stay resident (encoded or f32 constants; see [gpu.md](gpu.md#mixture-of-experts-on-the-device); `FORGE_GPU_MOE=0` turns it off). On Metal and RPC devices it runs on the host.

On the CPU, `FORGE_QDOT=0` replaces the SIMD quantised-row kernels with plain decode + dot (for benchmarking and bisecting).

## Example
`docs/examples/graph_build.cpp` (ctest `doc_example_graph_build`) checks `silu(1)·exp(1)` and that the gradient at x = 0 is 0.5:

```cpp
tensor_graph g;
auto x = g.input("x", {2, 4});
auto y = g.mul(g.apply(ops::op::silu, {x}), g.apply(ops::op::exp, {x}));   // fused to silu_mul
g.output("y", y);
auto flat = g.apply(ops::op::reshape, {y}, {{"shape", std::vector<i64>{8}}});
auto loss = g.apply(ops::op::reduce_sum, {flat}, {{"axis", 0.0}});
auto grads = grad(g, loss, {x});
g.output("dx", grads[x]);
auto plan = compile(g);
executor ex(*plan);
ex.set_input("x", in);
ex.run();
tensor out = ex.output("y");
```

## RPC device
`rpc:host:port` sends each op to a server started with `forge rpc-serve [--port 7000] [--bind 127.0.0.1] [--device cpu]`. The server runs it with its own device registry. Every op is still oracle-checked on the client. The plan's f32 weights and constants are uploaded once per connection and referred to by BLAKE3 hash afterwards ([rpc.md](rpc.md#weights-on-the-server)); activations travel with every op.

Optional TLS (mTLS too) and token or Ed25519 key authentication are available. The client reads its credentials from `FORGE_RPC_KEY`, `FORGE_RPC_TOKEN`, `FORGE_RPC_TLS=1`, `FORGE_RPC_CA`, `FORGE_RPC_FINGERPRINT`, `FORGE_RPC_SERVER_NAME`, `FORGE_RPC_TLS_CERT` and `FORGE_RPC_TLS_KEY`. A server with no tokens or keys accepts anyone, so bind it to loopback. Details: [rpc.md](rpc.md).

## Platform notes and limits
- Link `forge_graph`, `forge_verify` and `forge_simd_mv`.
- Device lowerings have run on an NVIDIA RTX 3090 (CUDA; not the kernels added on 2026-11-12), Mesa lavapipe and SwiftShader (Vulkan), mock drivers (CUDA, HIP, OpenCL, Level Zero) and, for the Metal lowerings that existed at PR #53, the GitHub macOS runner's GPU. The Metal ports added on 2026-11-12 have never been compiled by a Metal toolchain or executed. See [gpu.md](gpu.md) and [UNTESTED.md](UNTESTED.md#4-gpu-backends).
- Pipeline splits have only been run with CPU and lavapipe stages and, for `pipe:auto`, two devices of a mock driver; never across real GPUs.
