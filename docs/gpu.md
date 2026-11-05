# gpu: devices, graphs, op runtimes and parallel primitives

## Concepts
`forge::gpu` has two parts that are used differently.

1. **Programming model** (`gpu.hpp`, `prims.hpp`). Record copies and kernel launches into a `gpu::graph`, `compile` it once for a device, then `run()` the result as often as needed. Kernels are function objects whose call operator is marked `FORGE_KERNEL`. This part runs on the **CPU only**: `select()` always returns the CPU device, and launches are split across the task pool.
2. **Op runtimes** (`vulkan.hpp`, `cuda.hpp`, `opencl.hpp`, `levelzero.hpp`, `metal.hpp`). Each runs single spec ops (`run_op`) on a real device. They are what model execution uses: [graph](graph.md) wraps them as lowerings, selected with `FORGE_DEVICE` or `device_registry(name)`.

Nothing links against a vendor SDK. Each runtime `dlopen`s its driver library at run time. A runtime that cannot load reports why, and work stays on the CPU.

### Kernels come from the op spec
`ops/gen/gpu_gen.py` generates every kernel from `ops/defs/*.fspec` at build time:

| Output | Used by | Built with |
|--------|---------|------------|
| one GLSL compute shader per kernel (33) | Vulkan | `glslangValidator` → SPIR-V, embedded |
| `kernels.cu` | CUDA, HIP | clang, no SDK: two PTX images, `sm_70` and `sm_80` (JIT-compiled by the driver; the `sm_80` one adds the tensor-core kernels); AMDGPU code objects for `FORGE_HIP_ARCHS` (default `gfx90a gfx942 gfx1100`, needs lld) |
| `kernels.cl` | OpenCL (built by the driver at open), Level Zero (offline to SPIR-V with clang + `llvm-spirv`) | |
| `kernels.metal` | Metal (compiled at open with `newLibraryWithSource`) | |

The 33 spec kernels are: 18 unary (`neg abs sqrt rsqrt recip square exp exp2 log log2 sin cos tanh sigmoid silu relu softplus gelu_tanh`), 6 binary (`add sub mul div maximum minimum`) and `scale softmax rms_norm matmul_t silu_mul rope gather attention slice`. CUDA, HIP, OpenCL, Level Zero and Metal have no `sin`/`cos`, so they register **31** lowerings. [reference.md](reference.md) marks which ops have a GPU kernel.

`kernels.cu` also holds CUDA/HIP-only kernels that the runtime picks itself: see [CUDA and HIP](#cuda-and-hip).

### Backends

| Backend | Header | Device string | Memory pool | Run on |
|---------|--------|---------------|-------------|--------|
| Vulkan | `gpu/vulkan.hpp` | `vulkan` (first compute device) | `vulkan:0` | Mesa lavapipe and SwiftShader (software) |
| CUDA | `gpu/cuda.hpp` | `cuda[:N]` | `cuda:N` | NVIDIA RTX 3090 and mock driver |
| HIP | `gpu/cuda.hpp` | `hip[:N]` | `hip:N` | mock driver only |
| OpenCL | `gpu/opencl.hpp` | `opencl[:N]` (all platforms, GPUs first) | `opencl:N` | mock driver only |
| Level Zero | `gpu/levelzero.hpp` | `ze[:N]` / `level_zero[:N]` | `level_zero:N` | mock driver only |
| Metal | `gpu/metal.hpp` + `gpu/src/metal.mm` | `metal[:N]` | none | never compiled (no Mac) |

All runtimes except Metal share one contract: `run_op` over host tensor views, and a **residency protocol** (`begin_run`, `end_run`, `pin`, `to_host`, `host_wrote`, `release`, `copy_resident`). Between `begin_run` and `end_run`, buffers stay on the device keyed by host address. Pinned ranges (weights) survive across runs. Metal uses unified memory, so its residency hooks do nothing.

Every driver allocation of Vulkan, CUDA, HIP, OpenCL and Level Zero is charged to the memory pool named above (`gpu/device_memory.hpp`). When an allocation would cross the pool limit (`FORGE_MEM_DEVICE_LIMIT`) or the driver runs out, the runtime evicts: pooled free buffers first, then resident tensors and encoded weights, least recently used first. Device-only data is copied back to the host before it is freed. Buffers used by the op in progress are never evicted. See [memory.md](memory.md).

## CUDA and HIP
One runtime class (`gpu::cuda_hip::runtime`) serves both APIs. CUDA loads `libcuda.so.1` / `nvcuda.dll`; HIP loads `libamdhip64.so.6` (or `.5`, or unversioned) / `amdhip64.dll`. Each runtime has one non-blocking stream.

CUDA embeds two PTX images built from the same `kernels.cu`: `ptx sm_80` (`kernels.sm80.ptx`) and `ptx` (`sm_70`). The runtime tries `sm_80` first and falls back to `sm_70` on a device that rejects it. Kernels guarded by `__CUDA_ARCH__ >= 800` (the tensor-core matmul and the tiled prefill attention) exist only in the `sm_80` image; everything else is in both. `FORGE_CUDA_IMAGE=ptx` (or `ptx sm_80`) loads only that image. `image_arch()` reports the one in use.

### Kernel selection
`run_op` and the quantised entry points choose kernels by shape:

| Work | Kernel | When | Disable |
|------|--------|------|---------|
| Attention, prefill (tensor cores) | `forge_attention_fa2tc_d64` / `_d128` / `_d256` (`sm_80` image): FlashAttention-2 on fp16 `mma.sync` m16n8k16 with f32 accumulation; Q, K, V (and P) rounded to fp16 in shared memory (the KV cache stays f32), online softmax in f32 (exp2). 64 query rows per 128-thread block (16 per warp), 64/32/16-key tiles (d64/d128/d256), Q fragments in registers, the next K/V tile loaded into registers while the current one is computed, `ldmatrix` (`.trans` for V); causal, sliding window and GQA as below. Agrees with the f32 kernel to ~1e-3 | same as the tiled row below | `FORGE_FA_TC=0` (uses the f32 tiled kernel) |
| Attention, prefill (tiled) | `forge_attention_fa2_d128` / `_d256` (`sm_80` image): fp32 flash attention, 64 (head dim ≤ 128) or 32 (≤ 256) query rows per block, K/V tiles in shared memory | rows ≥ 16, plain f32 KV, head dims multiples of 4 and ≤ 256, split-K decode path not taken | `FORGE_GPU_NO_FA` (any value) |
| Attention, prefill | `forge_attention_fa`: a block per (query row, head), tiled online softmax | otherwise | `FORGE_GPU_NO_FA` (any value) falls back to `forge_attention` |
| Attention, decode | `forge_attention_dec` + `forge_attention_comb`: split-K over 16 key chunks, then a combine pass | rows × heads ≤ 256, f32 KV, head dims multiples of 32 and ≤ 256 | `FORGE_ATTN_DEC=0` |
| Quantised matmul_t, many rows (tensor cores) | `forge_quant_q8r` (activations to int8 per 32 values, plus per-group `d` and `d·sum`) + `forge_qmatmul_mma_{q8_0,q8_0g,q4_k,q6_k}` (`sm_80` image): int8 `mma.sync`, 128 weight rows × 64 tokens per block, `cp.async` staging; `forge_qmatmul_mma128_*` use 128-token tiles. Activations are stored with each group's 4-byte words interleaved so a B fragment is one 8-byte shared load. When the grid is smaller than the blocks the GPU holds at once, k is split into up to 8 slices (chosen by wave efficiency) and `forge_sum_splits` adds them in order | rows ≥ `FORGE_MMQ_MIN_ROWS`, q8_0 / q4_k / q6_k, k % 256 = 0. q4_k uses 64-token tiles (two blocks per SM); q6_k and q8_0 use 128 when rows ≥ 128 | `FORGE_MMA=0` (falls back to MMQ); `FORGE_MMA_BT=64\|128` forces the token tile; `FORGE_MMA_SPLITK=0` turns split-K off |
| Quantised matmul_t, many rows | `forge_qmatmul_mmq`: 64×64 tiles | rows ≥ `FORGE_MMQ_MIN_ROWS` (default 16), MMA not used | `FORGE_MMQ_MIN_ROWS=0` |
| Quantised matmul_t, few rows (decode) | `forge_quant_q81` + `forge_qmatmul_q81`: activations quantised to q8_1, dp4a dot, a 128-thread block per output column whose 4 warps split k (up to 4 activation rows per block) | q8_0, q4_k, q6_k weights | `FORGE_MMVQ=0` |
| Quantised matmul_t, otherwise | `forge_qmatmul_t` | q5_k at few rows, or MMVQ off | |
| Embedding gather from encoded rows | `forge_qgather` | q8_0, q4_k, q6_k, q5_k tables | `FORGE_GPU_QGATHER=0` |
| rms_norm | `forge_rms_norm_b` (block per row) | when present | |
| FFN gate/up/SiLU, prefill | `forge_quant_q8r` + `forge_qmatmul_glu_{q8_0g,q4_k,q6_k}` / `forge_qmatmul_glu128_*` (`sm_80` image): the tensor-core matmul over the interleaved gate/up weight with a SiLU epilogue that writes `silu(gate) * up` and its q8r form; the down projection uses that instead of running `forge_quant_q8r` | a fused GLU step (below), rows ≥ `FORGE_MMQ_MIN_ROWS` | `FORGE_FUSE_GLU=0` |
| FFN gate/up/SiLU, decode | `forge_quant_q81` + `forge_qmatmul_q81` over the interleaved weight (2n rows) + `forge_glu_q81` (`silu(gate) * up` and its q8_1 blocks, reused by the down projection) | a fused GLU step, fewer rows | `FORGE_FUSE_GLU=0` |
| q8_1 fusion | `forge_rms_norm_q81`, `forge_silu_mul_q81` also write the q8_1 form of their output for the next matmul. `forge_rms_norm_q81` reduces with warp shuffles and uses one 1024-thread block for a wide row (rows ≤ 4, cols ≥ 2048) | decode rows inside a run, with MMVQ on | `FORGE_FUSE_Q81=0` |
| Broadcast add/sub/mul/div | `forge_bcast` | the smaller shape is a suffix of the larger | |
| Gated DeltaNet state ops | `forge_delta_rule_p8` / `_p16` / `_p32`: the recurrent state in registers, a 64-thread block per (value head, 8 state columns); tokens in chunks of 4, the next chunk's q/k/v/g/beta loaded and l2-normalised into a double-buffered shared tile while the current one is computed. Bitwise the same results as `_rN` | CUDA (both images), inside a run; dk ∈ {64, 128, 256}, dv % 8 = 0 | `FORGE_DELTA_RULE_R` (any value) |
| | `forge_delta_rule_r8` / `_r16` / `_r32`: the state in registers, a 256-thread block per (value head, 32 state columns), q/k read per token | with `FORGE_DELTA_RULE_R`; dv % 32 = 0 | `FORGE_DELTA_RULE_OLD` (any value) |
| | `forge_causal_conv_t`: a thread per (channel, 8 tokens), kernel width 1–9 | inside a run | `FORGE_CAUSAL_CONV_OLD` (any value) |
| | `forge_delta_rule` (dk ≤ 256), `forge_causal_conv` (a thread per channel; kernel width 1–9) | inside a run; state slot stays resident | `FORGE_GPU_NO_STATE_OPS` (any value) |
| Device copies | `forge_copy2d` (concat), `forge_copy_words` (copies inside a captured segment) | | |

Encoded weights (q8_0, q4_k, q6_k, q5_k) are uploaded once and stay on the device. With the `sm_80` image and both MMA and MMVQ on, two codecs are repacked on upload so the kernels can use aligned 16-byte loads (k % 256 = 0 only): q6_k blocks are padded from 210 to 224 bytes (about 6.7% more device memory), and q8_0 is regrouped so each 8 blocks become 8 fp16 scales followed by 256 int8 values (272 bytes, the same size). Decode MMVQ reads both layouts; q4_k reads its 16-byte super-block header in one load. Consecutive decode matmuls that read the same activation reuse its q8_1 form instead of re-quantising.

### FFN gate/up fusion
`graph::compile` (`fuse_glu` in `plan.hpp`) turns `silu_mul(matmul_t(x, Wg), matmul_t(x, Wu))` into one step when the backend registers `registry::device_qglu` (CUDA): the two matmuls must be device quantised matmuls in the two steps right before the `silu_mul`, read the same `x`, have weights of the same codec (q8_0, q4_k or q6_k) and shape, and feed nothing else. This matches `linear(silu(linear(h, w_gate)) * linear(h, w_up), w_down)` in the llama, qwen3, qwen3_5, mixtral and deepseek_v2 blueprints. The step runs `run_qglu`:

- **One weight.** Gate and up are uploaded once as a single `[2n, k]` weight that interleaves them in groups of 16 rows (gate rows 8i…8i+7, then up rows 8i…8i+7), repacked like the separate weights (q6_k padded to 224-byte blocks, q8_0 regrouped). The separate copies are never uploaded, so device memory does not grow.
- **Prefill** (rows ≥ `FORGE_MMQ_MIN_ROWS`): one `forge_qmatmul_glu_*` launch (llama.cpp's `mul_mat_q` with a GLU epilogue does the same). Each 256-thread block covers 128 interleaved rows, that is the gate and up rows of 64 output columns, so a thread holds matching gate and up accumulators. The epilogue computes `silu(gate) * up` exactly as `forge_silu_mul` does, stages it in shared memory, writes `y` with coalesced stores, and also writes its q8r form (int8 values plus `(d, d·sum)` per 32 values, as `forge_quant_q8r` computes them). The down projection's `run_qmatmul` uses that form when nothing was launched, copied or freed in between, and skips `forge_quant_q8r` (`quant_q8r_reused` in `dispatches()`). No split-K.
- **Decode**: `forge_qmatmul_q81` over the 2n interleaved rows into a scratch buffer, then `forge_glu_q81` writes `y` and its q8_1 blocks, which the down projection reuses (unless `FORGE_FUSE_Q81=0`). That is two kernels where there were three (gate, up, `forge_silu_mul_q81`).

Measured on the RTX 3090 (2026-11-06, interleaved runs with `FORGE_FUSE_GLU` on and off): Goedel-Prover-V2 32B Q4_K_M prefill (512 tokens) 960 → 990 tok/s. Per prefill, the gate/up matmuls took 229 ms as 128 `forge_qmatmul_mma_q4_k` launches and take 237 ms as 64 `forge_qmatmul_glu_q4_k` launches (the epilogue adds about 5% instructions). `forge_silu_mul` (13 ms) is gone, and `forge_quant_q8r` drops from 18.3 to 9.1 ms. Qwen3.5 9B prefill is 3700 → 3890 tok/s. Decode gains 1–2%: one launch fewer per layer.

Results are bitwise those of the unfused path, except where the unfused prefill matmul would have used split-K (rounding of the k-sum order). `run_qglu` takes the step only inside a run, for k % 256 = 0 and n % 64 = 0, and only when both the prefill and decode kernels exist (the `sm_80` image with MMA and MMVQ on), so a process never holds both weight layouts. Otherwise the step runs the two matmuls and `silu_mul` into the planned gate/up buffers, as before. HIP, the `sm_70` image, OpenCL, Level Zero and the CPU never fuse.

Attention supports the KV codecs `int8`, `fp8`, `tbq4` and `tbq3` (tbq only up to head dim 256). `tbq4p` and `tbq3p` run on the CPU only. The same holds for Vulkan.

### Graphs and replay (CUDA)
- **User capture**: `capture_begin()`, `capture_end()` → id, `graph_launch(id)`. Streams are captured in relaxed mode so `run_op` may allocate while capturing. Needs the driver's graph API (`has_graphs()`); both CUDA and HIP export it.
- **Decode graphs** (CUDA only, on by default; `FORGE_GRAPHS=0` turns them off). Inside a run, kernel launches are stream-captured in segments. A segment ends at a host copy, a free, a sync, or after `FORGE_GRAPH_SEG` kernels (default 64; 0 = no split). Each segment's exec graph is cached by index. On later tokens, if the sequence of kernels and launch shapes matches, only the kernel nodes whose arguments changed are patched (`cuGraphExecKernelNodeSetParams`) and the graph is relaunched, without capturing. This needs `cuFuncGetParamInfo` (CUDA 12.4+). Without it, or with `FORGE_GRAPH_REPLAY=0`, each segment is re-captured and refreshed with `cuGraphExecUpdate`.
- **Whole-run replay** (`FORGE_GPU_REPLAY` set, experimental, opt-in; also needs decode graphs on and the fixed arena). When every step of a run was a recorded kernel, the runtime predicts the next run's arguments from the last two (decode moves the KV write row and attention length by one each token) and relaunches all segments without the executor. A run becomes replayable only after a real run confirmed a prediction. The executor does one real run every `FORGE_REPLAY_VERIFY` runs (default 32). A wrong prediction disables replay for that plan and prints a warning.
- **Fixed arena**: a plan's whole activation arena is held as one device buffer across runs, so every intermediate keeps its device address. Used when the arena is at most `FORGE_GPU_ARENA_MB` (default 1024; 0 = off).

Freed device buffers are kept in a per-runtime pool for reuse, up to `FORGE_GPU_POOL_MB` (default 4096). The pool is the first thing given back under memory pressure.

`FORGE_GPU_EAGER_SYNC` (any value) turns off the fixed arena, whole-run replay, and the end-of-run shortcut that leaves intermediates unread on the device. Every run then ends by syncing all resident buffers to the host.

### Runtime API (`gpu::cuda_hip`)

| Member | Purpose |
|--------|---------|
| `runtime::open(api, &why, ordinal)`, `runtime::device_count(api)` | Open a device (`api::cuda` / `api::hip`); `nullptr` + reason on failure |
| `device_name()`, `image_arch()`, `has_graphs()`, `has_streams()`, `memory()` | Device facts; `memory()` is free/total bytes |
| `supports(op, shapes, attrs)` (free function), `run_op(op, in, attrs, out)` | One spec op; `false` = no kernel for this shape |
| `run_qmatmul`, `run_qgather`, `run_state` | Quantised matmul_t, quantised gather, DeltaNet state ops |
| `begin_run` … `end_run`, `pin`, `to_host`, `host_wrote`, `release` | Residency protocol |
| `argmax_resident(p, n)` | Greedy decode: `forge_argmax` over a device-newer buffer, one `int` read back (ties: lowest index); `-1` when not resident. Wired as the `device_hooks::argmax` hook: with a plain greedy sampler, `session::generate` asks the executor for the logits' argmax instead of downloading them |
| `capture_begin`, `capture_end`, `graph_launch` | User stream capture |
| `copy_to_peer`, `device_alloc`, `device_free`, `upload`, `download` | Raw device memory and peer copies (host-staged when the APIs differ) |
| `residency()`, `dispatches()`, `memory_pool()`, `charged_bytes()` | Statistics: uploads, downloads, resident and pooled bytes, evictions, per-kernel dispatch counts |
| `cuda_hip::counters()` | Process-wide driver call counters (launches, captured, graph launches, copies, syncs) read by `forge bench` and `tools/bench_llm.sh` |

OpenCL (`gpu::ocl`) and Level Zero (`gpu::ze`) runtimes have `open(&why, ordinal)`, `device_count()`, `run_op`, the residency protocol, peer copies, and recorded "graphs" (enqueues replayed with the same arguments). They have none of the CUDA/HIP-only kernels above.

## Programming model API (`gpu.hpp`, `prims.hpp`)

| Item | Notes |
|------|-------|
| `probe_backends()` | One `backend_probe {kind, available, detail}` per `backend_kind` (`cpu`, `cuda`, `hip`, `level_zero`, `vulkan`, `metal`). It only checks that the driver library loads. `metal` is always reported absent. OpenCL is not listed. `name(kind)` gives the string |
| `select(prefer = prefer::any)`, `all()` | Always the CPU device (`caps().name` is `cpu/<simd target>`) |
| `device::alloc<T>(n)` → `buffer<T>`; `device::caps()` | Host buffers; capabilities incl. `compute_units` |
| `graph::copy(src, dst)`, `graph::launch(kernel, extent, args...)`, `graph::host(fn)` | Return a node. Nodes run in recorded order by default; `node.after(other)` adds a dependency |
| `graph::compile(device)` → `executable` | `run()` replays the flat order; `steps()` counts nodes |
| `item<3>` | `global(d)`, `local(d)`, `group(d)`, `range(d)` |
| `gpu::prims` | `reduce`, `scan(x, y, inclusive = true)`, `compact(x, out, pred)`, `radix_sort(keys, vals = {})`, `histogram`, `segmented_scan`, `segmented_reduce`, `bitonic_sort`, `top_k`, `gemm_nt`, `philox_uniform`, `hash_map`. CPU implementations |
| `interop.hpp` | `vk_external`: allocate exportable Vulkan buffers and semaphores on the application's `VkDevice`, export/import them as OS handles (fd or NT handle). `cuda_import`: map them into CUDA (`cuImportExternalMemory`) |

## Example
`docs/examples/gpu_graph.cpp` (ctest `doc_example_gpu_graph`):

```cpp
struct Scale {
    float a;
    FORGE_KERNEL void operator()(gpu::item<3> it, gpu::span<float> y) const {
        if (auto i = it.global(0); i < i64(y.size())) y[std::size_t(i)] *= a;
    }
};
auto dev = gpu::select();
auto buf = dev.alloc<float>(n);
gpu::graph g;
auto up = g.copy(gpu::span<const float>(host), buf);
auto k  = g.launch(Scale{3.f}, gpu::extent{i64(n)}, buf); k.after(up);
g.copy(buf, gpu::span<float>(back)).after(k);
auto exe = g.compile(dev);
exe.run();                               // replay as often as needed
gpu::prims::scan<int>(x, y, true);       // inclusive prefix sum
```

Running a model on a device goes through graph or inference instead: `FORGE_DEVICE=cuda forge run model.fpk ...`.

## Environment

| Variable | Default | Backend | Effect |
|----------|---------|---------|--------|
| `FORGE_CUDA_IMAGE` | unset | CUDA | `ptx` (sm_70) or `ptx sm_80`: load only that PTX image |
| `FORGE_GPU_NO_FA` | unset | CUDA/HIP | Set: no flash-attention kernels (`forge_attention_fa`, `forge_attention_fa2_*`, `forge_attention_fa2tc_*`) |
| `FORGE_FA_TC` | on | CUDA | `0`: tiled prefill attention uses the f32 SIMT kernel (`forge_attention_fa2_*`) instead of the fp16 tensor-core one |
| `FORGE_ATTN_DEC` | on | CUDA/HIP | `0`: no split-K decode attention |
| `FORGE_MMVQ` | on | CUDA/HIP | `0`: no q8_1/dp4a decode matmul |
| `FORGE_MMQ_MIN_ROWS` | 16 | CUDA/HIP | Rows at which the tiled MMA / MMQ kernels are used; `0` disables them |
| `FORGE_MMA` | on | CUDA | `0`: no int8 tensor-core prefill matmul (and no q6_k/q8_0 repacking); MMQ is used instead |
| `FORGE_MMA_BT` | per codec | CUDA | `64` or `128`: tokens per MMA block |
| `FORGE_DELTA_RULE_OLD` | unset | CUDA | Set: use `forge_delta_rule` instead of the register-state kernels |
| `FORGE_DELTA_RULE_R` | unset | CUDA | Set: use the `forge_delta_rule_rN` register-state kernels instead of the staged `_pN` ones |
| `FORGE_CAUSAL_CONV_OLD` | unset | CUDA/HIP | Set: use the per-channel `forge_causal_conv` instead of `forge_causal_conv_t` |
| `FORGE_FUSE_Q81` | on | CUDA/HIP | `0`: rms_norm/silu_mul do not emit q8_1 |
| `FORGE_FUSE_GLU` | on | CUDA | `0`: FFN gate and up run as two matmuls plus `silu_mul` (see [FFN gate/up fusion](#ffn-gateup-fusion)) |
| `FORGE_GPU_QMATMUL` | on | CUDA/HIP | `0`: no device quantised matmul; weights are decoded to f32 within `FORGE_GPU_WEIGHT_BUDGET` |
| `FORGE_GPU_QGATHER` | on | CUDA/HIP | `0`: no device quantised gather |
| `FORGE_GPU_NO_STATE_OPS` | unset | CUDA/HIP | Set: DeltaNet state ops run on the host |
| `FORGE_GRAPHS` | on | CUDA | `0`: do not capture decode runs into CUDA graph segments |
| `FORGE_GRAPH_SEG` | 64 | CUDA | Kernels per captured segment; `0` = unlimited |
| `FORGE_GRAPH_REPLAY` | on | CUDA | `0`: re-capture segments instead of patching kernel nodes |
| `FORGE_GPU_REPLAY` | unset | CUDA | Set: experimental whole-run replay |
| `FORGE_REPLAY_VERIFY` | 32 | CUDA | One real run every N during whole-run replay |
| `FORGE_GPU_ARENA_MB` | 1024 | CUDA/HIP | Largest arena held at fixed device addresses; `0` = off |
| `FORGE_GPU_POOL_MB` | 4096 | CUDA/HIP | Bytes of freed device buffers kept for reuse |
| `FORGE_GPU_EAGER_SYNC` | unset | CUDA/HIP | Set: sync every resident buffer to the host at the end of each run |
| `FORGE_GPU_NO_RESIDENCY` | unset | CUDA/HIP, OpenCL, Level Zero, Metal | Set: no residency hooks; every op uploads and downloads. Does not affect Vulkan |
| `FORGE_GPU_WEIGHT_BUDGET` | 8G | resident backends | f32 bytes of decoded quantised weights mirrored on the device (K/M/G suffixes); the rest stay encoded and run on the CPU. Backends with a device quantised matmul keep those codecs encoded instead |
| `FORGE_OPENCL_TYPE` | `all` | OpenCL | `gpu`, `cpu` or `all`: which device types are enumerated |
| `FORGE_VK_PIPELINE_CACHE` | on | Vulkan | `0`: no on-disk pipeline cache |
| `FORGE_CACHE_DIR` | `$XDG_CACHE_HOME/forge` or `~/.cache/forge` | Vulkan | Pipeline cache directory |
| `FORGE_VK_DEBUG` | unset | Vulkan | Set: log each op/shape the Vulkan lowering refuses |
| `FORGE_PLAN_TRACE` | unset | all devices | Set: at exit, print the ops that ran on the host during device runs |
| `FORGE_MEM_DEVICE_LIMIT` | unset | all pools | Device pool limit ([memory.md](memory.md)) |

Test hooks, not for normal use: `FORGE_VK_SSBO_ALIGN` (emulate a stricter storage-buffer alignment), `FORGE_OPENCL_TEST_ALIGN`, `FORGE_ZE_TEST_IMAGE=opencl-c` (hand OpenCL C to the mock Level Zero driver).

## Platform notes and limits
- Link `forge_gpu` and `forge_simd_mv`.
- **Vulkan** needs `glslangValidator` and `vulkan/vulkan.h` at build time. Without them it is compiled out, even if `forge info` finds `libvulkan`. One device only. The pipeline cache is keyed by vendor, device, driver, pipeline-cache UUID and a hash of the SPIR-V.
- **CUDA / HIP** images need a clang with the NVPTX / AMDGPU backends at build time (HIP also needs lld; zig's `ld.lld` works). Without them `open()` reports "this build has no embedded kernel image". Mobile builds have no CUDA/HIP images.
- **Level Zero** needs `llvm-spirv` at build time; without it `open()` says so. `opencl` on the same GPU is the alternative.
- **Metal** builds only on Apple platforms (`FORGE_GPU_METAL`); elsewhere `open()` reports "not an Apple platform build". On Linux only an MSL syntax check runs (`gpu_compile_metal_syntax`). Metal has no graphs. Each op charges its buffers to the `metal:<ordinal>` memory pool while it runs; with unified memory and no residency there is nothing to evict, so an op that does not fit fails (see [memory.md](memory.md)).
- Tested on hardware:
  - **CUDA** on an NVIDIA RTX 3090. 2026-10-31 (WSL2, CUDA 13, clang 23.1.4): full ctest 104/104. 2026-11-04 (Windows 11, mingw-w64 build): `test_gpu_cuda` passed 31/31 lowerings against the oracle, 24/24 quantised matmul cases, 7/7 split-K decode-attention shapes, and delta_rule/causal_conv matched the CPU. The same day a bug was fixed where the split-K combine kernel was launched with dangling argument pointers (commit 7fe8a83); the 3090 had passed by luck, the mocks failed. 2026-11-05 (WSL2, CUDA 13 driver 610.88, clang 23): the `sm_80` image (tensor-core prefill matmul, tiled prefill attention), register-state delta_rule and graphs on by default; full Linux ctest passes and greedy output is identical with graphs on and off. Prefill (512 tokens) / decode (128 tokens) tok/s on Q4_K_M GGUF weights, against llama.cpp `llama-bench` (bc428e4) on the same file: Qwen3.5 9B 2720 / 107.5 (llama.cpp 4740 / 120.5); Goedel-Prover-V2 32B (Qwen3) 839 / 35.3 (llama.cpp 1360 / 38.4). See [STATUS.md](STATUS.md).
  - **Vulkan** on Mesa lavapipe and SwiftShader only (software devices).
  - **Metal** on the GitHub `macos-latest` runner's GPU: `test_gpu_metal` passes in CI (oracle check of every `metal.*` lowering). The device model is not logged.
  - **HIP, OpenCL, Level Zero**: never on real hardware. HIP, OpenCL and Level Zero run against mock driver libraries (`tests/mock_cuda.c` serves CUDA and HIP, `mock_opencl.c`, `mock_ze.c`) in `test_gpu_*_mock`, `test_gpu_memory_*_mock` and `golden_tiny_llama_*_mock`. The OpenCL and Level Zero mocks compile the generated OpenCL C as C and run it. The AMD code objects are only compile-checked (`gpu_compile_hip`).
- Graphics interop: the Vulkan export/import round trip has run on lavapipe. The CUDA import has not run on hardware.
- Known gaps: no eviction while a CUDA/HIP stream capture or an OpenCL/Level Zero recording is open, and graphs recorded before an eviction must not be replayed after it.
