# gpu: devices, graphs, op runtimes and parallel primitives

## Concepts
`forge::gpu` has two parts that are used differently.

1. **Programming model** (`gpu.hpp`, `prims.hpp`). Record copies and kernel launches into a `gpu::graph`, `compile` it once for a device, then `run()` the result as often as needed. Kernels are function objects whose call operator is marked `FORGE_KERNEL`. This part runs on the **CPU only**: `select()` always returns the CPU device, and launches are split across the task pool.
2. **Op runtimes** (`vulkan.hpp`, `cuda.hpp`, `opencl.hpp`, `levelzero.hpp`, `metal.hpp`). Each runs single spec ops (`run_op`) on a real device. They are what model execution uses: [graph](graph.md) wraps them as lowerings, selected with `FORGE_DEVICE` or `device_registry(name)`; `FORGE_DEVICE=auto` scores the devices that open ([Automatic device selection](#automatic-device-selection)).

Nothing links against a vendor SDK. Each runtime `dlopen`s its driver library at run time. A runtime that cannot load reports why, and work stays on the CPU.

### Kernels come from the op spec
`ops/gen/gpu_gen.py` generates every kernel from `ops/defs/*.fspec` at build time:

| Output | Used by | Built with |
|--------|---------|------------|
| one GLSL compute shader per kernel (33) | Vulkan | `glslangValidator` → SPIR-V, embedded |
| `kernels.cu` | CUDA, HIP | clang, no SDK: two PTX images, `sm_70` and `sm_80` (JIT-compiled by the driver; the `sm_80` one adds the tensor-core kernels); AMDGPU code objects for `FORGE_HIP_ARCHS` (default `gfx90a gfx942 gfx1100`, needs lld) |
| `kernels.cl` | OpenCL (built by the driver at open, then loaded from the [kernel program cache](#kernel-program-cache)), Level Zero (offline to SPIR-V with clang + `llvm-spirv`; the driver's native binary is cached the same way) | |
| `kernels.metal` | Metal (compiled at open with `newLibraryWithSource`) | |

The 33 spec kernels are: 18 unary (`neg abs sqrt rsqrt recip square exp exp2 log log2 sin cos tanh sigmoid silu relu softplus gelu_tanh`), 6 binary (`add sub mul div maximum minimum`) and `scale softmax rms_norm matmul_t silu_mul rope gather attention slice`. CUDA, HIP, OpenCL, Level Zero and Metal have no `sin`/`cos`, so they register **31** lowerings. [reference.md](reference.md) marks which ops have a GPU kernel.

`rope` with frequency scaling (`scaling`, `freq_factors`; `ops/include/forge/ops/rope.hpp`): the CUDA, HIP, OpenCL, Level Zero and Metal kernels read the inverse frequencies from a table the host builds, so they run every variant unchanged except dynamic NTK, whose table depends on the positions of each call and runs on the CPU. The Vulkan kernel computes `base^(-2j/d)` itself and hands any scaled `rope` to the CPU. The mock drivers cover the CUDA, HIP, OpenCL and Level Zero paths; the Metal and Vulkan paths for scaled `rope` have not been run.

`kernels.cu` also holds CUDA/HIP-only kernels that the runtime picks itself: see [CUDA and HIP](#cuda-and-hip). `kernels.cl` holds portable versions of the same families (the `PORT_*` section of `gpu_gen.py`): see [Portable kernels](#portable-kernels-opencl-and-level-zero).

### Kernels per backend
Beyond the spec kernels. "tuned" = the block / shared-memory / tensor-core kernels of [CUDA and HIP](#cuda-and-hip); "portable" = one work item per output, no workgroup-shared memory; "–" = runs on the host (or, for quantised weights, decoded to f32 within `FORGE_GPU_WEIGHT_BUDGET`).

| Kernel family | CUDA | HIP | OpenCL | Level Zero | Vulkan | Metal |
|---|---|---|---|---|---|---|
| Quantised `matmul_t` (q8_0, q4_k, q5_k, q6_k weights kept encoded) | tuned (MMVQ, MMQ, + tensor cores on `sm_80`) | `forge_qmatmul_t`, `forge_qmatmul_mmq` | portable `forge_qmatmul_t` | portable | – | – |
| Quantised gather (embedding from encoded rows) | `forge_qgather` | `forge_qgather` | portable `forge_qgather` | portable | – | – |
| FFN gate/up/SiLU fusion | `forge_qmatmul_glu_*`, `forge_glu_q81` (`sm_80`) | – | portable `forge_qglu` (all four codecs) | portable | – | – |
| Attention, prefill | `forge_attention_fa`, `_fa2_*`, `_fa2tc_*` | `forge_attention_fa` | portable `forge_attention_fa` (one pass) | portable | spec kernel | spec kernel |
| Attention, decode (split over keys) | `forge_attention_dec` + `_comb` | – (`forge_attention_fa`) | portable `forge_attention_dec` + `_comb` | portable | spec kernel | spec kernel |
| Gated DeltaNet state ops | `forge_delta_rule*`, `forge_causal_conv*` | `forge_delta_rule`, `forge_causal_conv*` | portable `forge_delta_rule`, `forge_causal_conv` | portable | – | – |
| Sparse mixture of experts (`moe_ffn`) | `forge_moe_hidden` + `forge_moe_down` | same | same | same | – | – |
| Broadcast add/sub/mul/div | `forge_bcast` | `forge_bcast` | portable `forge_bcast` | portable | – | – |
| q8_1 / q8r activation fusions, add+rms_norm, rms+rope | yes | – | – | – | – | – |
| Device concat (`forge_copy2d`), device argmax | yes | yes | – | – | – | – |

Verification of the non-CUDA columns is by the mock drivers only (they compile `kernels.cl` as C and run it); see [Platform notes](#platform-notes-and-limits).

### Backends

| Backend | Header | Device string | Memory pool | Run on |
|---------|--------|---------------|-------------|--------|
| Vulkan | `gpu/vulkan.hpp` | `vulkan` (first compute device) | `vulkan:0` | Mesa lavapipe and SwiftShader (software) |
| CUDA | `gpu/cuda.hpp` | `cuda[:N]` | `cuda:N` | NVIDIA RTX 3090 and mock driver |
| HIP | `gpu/cuda.hpp` | `hip[:N]` | `hip:N` | mock driver only |
| OpenCL | `gpu/opencl.hpp` | `opencl[:N]` (all platforms, GPUs first) | `opencl:N` | mock driver only |
| Level Zero | `gpu/levelzero.hpp` | `ze[:N]` / `level_zero[:N]` | `level_zero:N` | mock driver only |
| Metal | `gpu/metal.hpp` + `gpu/src/metal.mm` | `metal[:N]` | none | the GitHub macOS runner's GPU in CI (`test_gpu_metal`, last on PR #53). The Metal changes of 2026-11-12 (the RoPE frequency table, the kernels `msl_port` translates) have only passed the MSL syntax check on Linux |

All runtimes except Metal share one contract: `run_op` over host tensor views, and a **residency protocol** (`begin_run`, `end_run`, `pin`, `to_host`, `host_wrote`, `release`, `forget`, `copy_resident`). `forget(p, n)` drops the device copies and pin records of host memory that is going away; `graph::plan`'s destructor calls it for the plan's constants on every backend. (Until 2026-11-12 only CUDA/HIP had it: on OpenCL, Level Zero and Vulkan a destroyed plan left pins and resident entries behind, and the next plan's buffers at the same addresses read stale device data or were overwritten at sync-back. Regression test `test_gpu_stale_*_mock`.) The OpenCL and Level Zero `run_op` also acquire every buffer of a launch before binding any: acquiring a range that partly overlaps a resident entry drops that entry, and an input bound earlier in the same launch could live in it (the SIGBUS in `forge_matmul_t` with `FORGE_DEVICE=opencl` on the mock). Between `begin_run` and `end_run`, buffers stay on the device keyed by host address. Pinned ranges (weights) survive across runs. Metal uses unified memory, so its residency hooks do nothing.

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
| Quantised matmul_t, many rows (tensor cores) | `forge_quant_q8r` (activations to int8 per 32 values, plus per-group `d` and `d·sum`) + `forge_qmatmul_mma_{q8_0,q8_0g,q4_k,q6_k}` (`sm_80` image): int8 `mma.sync`, 128 weight rows × 64 tokens per block, `cp.async` staging; `forge_qmatmul_mma128_*` use 128-token tiles. Activations are stored with each group's 4-byte words interleaved so a B fragment is one 8-byte shared load. Grids of 1/4 to 4 waves that do not fill whole waves run as a persistent stream-K launch (opt-in, see [Stream-K](#stream-k-tensor-core-matmul)); smaller grids split k into up to 8 slices (chosen by wave efficiency) and `forge_sum_splits` adds them in order | rows ≥ `FORGE_MMQ_MIN_ROWS`, q8_0 / q4_k / q6_k, k % 256 = 0. q4_k uses 64-token tiles (two blocks per SM); q6_k and q8_0 use 128 when rows ≥ 128 | `FORGE_MMA=0` (falls back to MMQ); `FORGE_MMA_BT=64\|128` forces the token tile; `FORGE_MMA_STREAMK=1` turns stream-K on; `FORGE_MMA_SPLITK=0` turns split-K off |
| Quantised matmul_t, many rows | `forge_qmatmul_mmq`: 64×64 tiles | rows ≥ `FORGE_MMQ_MIN_ROWS` (default 16), MMA not used | `FORGE_MMQ_MIN_ROWS=0` |
| Quantised matmul_t, few rows (decode) | `forge_quant_q81` + `forge_qmatmul_q81`: activations quantised to q8_1, dp4a dot, a 128-thread block per output column whose 4 warps split k (up to 4 activation rows per block). Every lane loads 16 weight bytes at once (q4_k: 8 lanes per super-block; padded q6_k: 8 lanes; regrouped q8_0: 2 lanes per 32-value block, 16 per 272-byte group) and its activations as 8-byte loads. q4_k with k ≤ 6144 puts two columns in a block (2 warps each), since 4 warps would leave most lanes idle on the second step | q8_0, q4_k, q6_k weights | `FORGE_MMVQ=0`; `FORGE_MMVQ_CPB=1\|2\|4` forces the columns per block; `FORGE_MMVQ_Q8V=0` keeps the older 4-bytes-per-lane q8_0 loop |
| Quantised matmul_t, otherwise | `forge_qmatmul_t` | q5_k at few rows, or MMVQ off | |
| Embedding gather from encoded rows | `forge_qgather` | q8_0, q4_k, q6_k, q5_k tables | `FORGE_GPU_QGATHER=0` |
| rms_norm | `forge_rms_norm_b` (block per row) | when present | |
| FFN gate/up/SiLU, prefill | `forge_quant_q8r` + `forge_qmatmul_glu_{q8_0g,q4_k,q6_k}` / `forge_qmatmul_glu128_*` (`sm_80` image): the tensor-core matmul over the interleaved gate/up weight with a SiLU epilogue that writes `silu(gate) * up` and its q8r form; the down projection uses that instead of running `forge_quant_q8r` | a fused GLU step (below), rows ≥ `FORGE_MMQ_MIN_ROWS` | `FORGE_FUSE_GLU=0` |
| FFN gate/up/SiLU, decode | `forge_quant_q81` + `forge_qmatmul_q81` over the interleaved weight (2n rows) + `forge_glu_q81` (`silu(gate) * up` and its q8_1 blocks, reused by the down projection) | a fused GLU step, fewer rows | `FORGE_FUSE_GLU=0` |
| q8_1 fusion | `forge_rms_norm_q81`, `forge_silu_mul_q81` and `forge_attention_comb_q81` (decode attention's combine pass, feeding the output projection) also write the q8_1 form of their output for the next matmul. `forge_rms_norm_q81` reduces with warp shuffles and uses one 1024-thread block for a wide row (rows ≤ 4, cols ≥ 2048) | decode rows inside a run, with MMVQ on | `FORGE_FUSE_Q81=0` |
| q8r fusion (prefill) | `forge_rms_norm_f` writes the int8 rows and per-group `(d, d·sum)` of its output in `forge_quant_q8r`'s layout (same bytes), so the next tensor-core matmul (or a fused GLU step's gate/up matmul) skips `forge_quant_q8r`; a matmul whose input is the one the q8r scratch already holds (q/k/v, gate/up) also skips it. The scratch's source is tracked like the q8_1 one: dropped by a launch, upload or copy that writes it (or declares no writes), or a free | rms_norm rows ≥ `FORGE_MMQ_MIN_ROWS`, cols % 256 = 0 and ≥ 1024 (not per-head norms), MMA on (`sm_80` image) | `FORGE_FUSE_Q8R=0` |
| Residual add + rms_norm | `forge_rms_norm_f`: `s = a + b` and `y = rms_norm(s)` in one pass, plus y's q8_1 (decode) or q8r (prefill) output when the unfused rms_norm would write it. Reduces like the kernel run_op would pick (`forge_rms_norm_b` tree or `forge_rms_norm_q81` shuffles), so results are bit-identical | an `add` step followed by an `rms_norm` of its result (graph `fuse_steps`, device runs; `device_hooks::fused`); same-size inputs, cols % 32 = 0 | `FORGE_FUSE_ADD_NORM=0` |
| QK-norm + RoPE | `forge_rms_rope`: a 256-thread block per (token, up to 16 heads; 256 / head dim at decode), cos/sin computed once per block (double, as `forge_rope`) instead of per head; bit-identical to `rms_norm` + `forge_rope` | an `rms_norm` whose only consumer is the next step, a `rope` (Qwen3 per-head Q/K norm); head dim a power of two in 32..256 | `FORGE_FUSE_QK_ROPE=0` |
| Broadcast add/sub/mul/div | `forge_bcast` | the smaller shape is a suffix of the larger | |
| Gated DeltaNet state ops | `forge_delta_rule_p8` / `_p16` / `_p32`: the recurrent state in registers, a 64-thread block per (value head, 8 state columns); tokens in chunks of 4, the next chunk's q/k/v/g/beta loaded and l2-normalised into a double-buffered shared tile while the current one is computed. Bitwise the same results as `_rN` | CUDA (both images), inside a run; dk ∈ {64, 128, 256}, dv % 8 = 0 | `FORGE_DELTA_RULE_R` (any value) |
| | `forge_delta_rule_r8` / `_r16` / `_r32`: the state in registers, a 256-thread block per (value head, 32 state columns), q/k read per token | with `FORGE_DELTA_RULE_R`; dv % 32 = 0 | `FORGE_DELTA_RULE_OLD` (any value) |
| | `forge_causal_conv_t`: a thread per (channel, 8 tokens), kernel width 1–9 | inside a run | `FORGE_CAUSAL_CONV_OLD` (any value) |
| | `forge_delta_rule` (dk ≤ 256), `forge_causal_conv` (a thread per channel; kernel width 1–9) | inside a run; state slot stays resident | `FORGE_GPU_NO_STATE_OPS` (any value) |
| Device copies | `forge_copy2d` (concat), `forge_copy_words` (copies inside a captured segment) | | |

Encoded weights (q8_0, q4_k, q6_k, q5_k) are uploaded once and stay on the device. With the `sm_80` image and both MMA and MMVQ on, two codecs are repacked on upload so the kernels can use aligned 16-byte loads (k % 256 = 0 only): q6_k blocks are padded from 210 to 224 bytes (about 6.7% more device memory), and q8_0 is regrouped so each 8 blocks become 8 fp16 scales followed by 256 int8 values (272 bytes, the same size). Decode MMVQ reads both layouts; q4_k reads its 16-byte super-block header in one load. Consecutive decode matmuls that read the same activation reuse its q8_1 form instead of re-quantising.

### Stream-K tensor-core matmul
`forge_qmatmul_mma*` and `forge_qmatmul_glu*` also run as a persistent launch, after llama.cpp's `mul_mat_q` stream-k loop (`ggml-cuda/mmq.cuh`, MIT) and CUTLASS's data-parallel + stream-K hybrid. The grid is G = the blocks the GPU holds at once (`cuOccupancyMaxActiveBlocksPerMultiprocessor` × SMs: 164 for 64-token tiles and 82 for 128-token tiles on an RTX 3090), capped so each block gets at least `FORGE_MMA_SK_MIN` (4) 256-value k chunks. Tiles (128 weight rows × 64 or 128 tokens) are numbered token tile fastest. Each block first takes whole tiles b, b + G, ... for all but the last one or two waves, in the usual order, so concurrent blocks share weight rows in L2. The remaining G to 2G tiles are shared out as (tile, k chunk) iterations, an equal range per block.

A range splits at most two tiles. If its last piece ends mid-tile, the partial accumulators go to the block's slot in a fixup buffer (64 KB per SM) and the block sets a flag. If its first piece starts mid-tile and reaches the tile's end, it finishes that tile. The block computes that piece last, after publishing its own partial, so no block waits on another block that is itself waiting. It then waits for the flags of the earlier blocks that hold the rest of the tile and adds their partials in a fixed order (b − 1, b − 2, ...). It clears the flags and runs the normal epilogue, which for the GLU kernel is SiLU and the q8r hand-off. Results are deterministic, but the k-sum order differs from the one-tile-per-block launch (relative differences around 1e-6).

The flags are cleared by the block that consumes them, so the next launch or a graph replay starts from zeros. The buffers are allocated once.

**Off by default for now**: with `FORGE_MMA_STREAMK=1`, Qwen3.5 9B's greedy ids (17-token prompt) diverged from the default path after 9 tokens. That may be only the changed k-sum order flipping a near-tie, but it has not been checked yet. When enabled (`FORGE_MMA_STREAMK=1`; `2` forces it for every grid that does not fill whole waves, which the tests use), it runs for grids of 1/4 to 4 waves that do not fill whole waves. Below 1/4 wave, split-K's parallel sum beats the fixup's serial adds (Qwen3.5's 32-column DeltaNet projection: 54 µs either way, against 64 µs forced). From 4 waves on, the ragged tail costs about as much as the stream-K share. Not in the `sm_70` image or on HIP, which have no tensor-core kernels.

RTX 3090, 512 tokens, in-process A/B (`test_cuda_hip bench <codec> 512 <cols> <k> 20 FORGE_MMA_STREAMK=0,1`; the GPU was shared, so the minimum of 7 interleaved batches; the batch that runs first in each pair has a few percent of advantage, so each was also run in reverse order):

| Shape (cols × k) | split-K / one tile per block | stream-K |
|---|---|---|
| q4_k 1024 × 5120 (Goedel k/v) | 146–155 µs | 131–135 µs |
| q4_k 8192 × 5120 (Goedel q) | 691–698 µs | 627–638 µs |
| q4_k 5120 × 8192 (Goedel o) | 633–668 µs | 628–661 µs |
| q4_k 5120 × 25600 (Goedel down) | 2054–2070 µs | 1975–2031 µs |
| q6_k 5120 × 25600 (Goedel down) | 2252–2267 µs | 2189–2195 µs |
| q4_k 8192 × 4096 (Qwen3.5) | 547–556 µs | 498–506 µs |
| q4_k 4096 × 4096 (Qwen3.5) | 323 µs | 277–295 µs |
| q4_k 4096 × 12288 (Qwen3.5 down) | 834–843 µs | 758–768 µs |
| q8_0 4096 × 4096 | 307 µs | 270 µs |

Shapes with 4 or more waves (q4_k 25600 × 5120, 12288 × 4096, the fused GLUs) keep the one-tile-per-block launch. With stream-K on, the token tile defaults stay the best: q4_k 64 tokens (8192 × 5120: 631 against 687 µs), q6_k and q8_0 128 tokens.

### FFN gate/up fusion
`graph::compile` (`fuse_glu` in `plan.hpp`) turns `silu_mul(matmul_t(x, Wg), matmul_t(x, Wu))` into one step when the backend registers `registry::device_qglu` (CUDA): the two matmuls must be device quantised matmuls in the two steps right before the `silu_mul`, read the same `x`, have weights of the same codec (q8_0, q4_k or q6_k) and shape, and feed nothing else. This matches `linear(silu(linear(h, w_gate)) * linear(h, w_up), w_down)` in the llama, qwen3, qwen3_5, mixtral and deepseek_v2 blueprints. The step runs `run_qglu`:

- **One weight.** Gate and up are uploaded once as a single `[2n, k]` weight that interleaves them in groups of 16 rows (gate rows 8i…8i+7, then up rows 8i…8i+7), repacked like the separate weights (q6_k padded to 224-byte blocks, q8_0 regrouped). The separate copies are never uploaded, so device memory does not grow.
- **Prefill** (rows ≥ `FORGE_MMQ_MIN_ROWS`): one `forge_qmatmul_glu_*` launch (llama.cpp's `mul_mat_q` with a GLU epilogue does the same). Each 256-thread block covers 128 interleaved rows, that is the gate and up rows of 64 output columns, so a thread holds matching gate and up accumulators. The epilogue computes `silu(gate) * up` exactly as `forge_silu_mul` does, stages it in shared memory, writes `y` with coalesced stores, and also writes its q8r form (int8 values plus `(d, d·sum)` per 32 values, as `forge_quant_q8r` computes them). The down projection's `run_qmatmul` uses that form when nothing was launched, copied or freed in between, and skips `forge_quant_q8r` (`quant_q8r_reused` in `dispatches()`). No split-K; stream-K as for the plain kernel (`qmatmul_glu_streamk`).
- **Decode**: `forge_qmatmul_q81` over the 2n interleaved rows into a scratch buffer, then `forge_glu_q81` writes `y` and its q8_1 blocks, which the down projection reuses (unless `FORGE_FUSE_Q81=0`). That is two kernels where there were three (gate, up, `forge_silu_mul_q81`).

Measured on the RTX 3090 (2026-11-06, interleaved runs with `FORGE_FUSE_GLU` on and off): Goedel-Prover-V2 32B Q4_K_M prefill (512 tokens) 960 → 990 tok/s. Per prefill, the gate/up matmuls took 229 ms as 128 `forge_qmatmul_mma_q4_k` launches and take 237 ms as 64 `forge_qmatmul_glu_q4_k` launches (the epilogue adds about 5% instructions). `forge_silu_mul` (13 ms) is gone, and `forge_quant_q8r` drops from 18.3 to 9.1 ms. Qwen3.5 9B prefill is 3700 → 3890 tok/s. Decode gains 1–2%: one launch fewer per layer.

Results are bitwise those of the unfused path, except where the unfused prefill matmul would have used split-K (rounding of the k-sum order). `run_qglu` takes the step only inside a run, for k % 256 = 0 and n % 64 = 0, and only when both the prefill and decode kernels exist (the `sm_80` image with MMA and MMVQ on), so a process never holds both weight layouts. Otherwise the step runs the two matmuls and `silu_mul` into the planned gate/up buffers, as before. HIP, the `sm_70` image and the CPU never fuse. OpenCL and Level Zero fuse with their own kernel (`forge_qglu`, below).

Attention supports the KV codecs `int8`, `fp8`, `tbq4` and `tbq3` (tbq only up to head dim 256). `tbq4p` and `tbq3p` run on the CPU only. The same holds for Vulkan.

### Graphs and replay (CUDA)
- **User capture**: `capture_begin()`, `capture_end()` → id, `graph_launch(id)`. Streams are captured in relaxed mode so `run_op` may allocate while capturing. Needs the driver's graph API (`has_graphs()`); both CUDA and HIP export it.
- **Decode graphs** (CUDA only, on by default; `FORGE_GRAPHS=0` turns them off). Inside a run, kernel launches are stream-captured in segments. A segment ends at a host copy, a free, a sync, or after `FORGE_GRAPH_SEG` kernels (default 64; 0 = no split). The first segments of a run are shorter, `FORGE_GRAPH_SEG0` kernels (default 8; 0 = off) doubling up to `FORGE_GRAPH_SEG`, so the device starts on the new token while the host is still recording the rest. Each segment's exec graph is cached by index. On later tokens, if the sequence of kernels and launch shapes matches, only the kernel nodes whose arguments changed are patched (`cuGraphExecKernelNodeSetParams`) and the graph is relaunched, without capturing. This needs `cuFuncGetParamInfo` (CUDA 12.4+). Without it, or with `FORGE_GRAPH_REPLAY=0`, each segment is re-captured and refreshed with `cuGraphExecUpdate`.
- **Whole-run replay** (`FORGE_GPU_REPLAY` set, experimental, opt-in; also needs decode graphs on and the fixed arena). When every step of a run was a recorded kernel, the runtime predicts the next run's arguments from the last two (decode moves the KV write row and attention length by one each token) and relaunches all segments without the executor. A run becomes replayable only after a real run confirmed a prediction. The executor does one real run every `FORGE_REPLAY_VERIFY` runs (default 32). A wrong prediction disables replay for that plan and prints a warning. Segments are patched and launched one at a time, so the device starts on the first while the host patches the next (`FORGE_REPLAY_EAGER=0`: patch all, then launch).
- **Small copies** (CUDA): host-to-device copies of up to 4 KB (decode: the token id and position) are staged through a 64 KB page-locked ring and issued as real async copies; from pageable memory the driver stages them synchronously, which costs tens of µs per copy under WSL. A slot is reused only after the next stream sync. `FORGE_GPU_STAGE=0` turns it off.
- **Host waits**: `FORGE_CUDA_SCHED=spin|yield|block` picks the context's scheduling flag (`CU_CTX_SCHED_*`; default auto).
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
| `argmax_resident(p, n)` | Greedy decode: `forge_argmax` over a device-newer buffer, one `int` read back into page-locked memory (ties: lowest index); `-1` when not resident. Up to 128 blocks scan the buffer and the last one to finish merges their partials (a 248k vocabulary: ~10 µs, against ~85 µs for one block). Wired as the `device_hooks::argmax` hook: with a plain greedy sampler, `session::generate` asks the executor for the logits' argmax instead of downloading them |
| `capture_begin`, `capture_end`, `graph_launch` | User stream capture |
| `copy_to_peer`, `device_alloc`, `device_free`, `upload`, `download` | Raw device memory and peer copies (host-staged when the APIs differ) |
| `residency()`, `dispatches()`, `memory_pool()`, `charged_bytes()` | Statistics: uploads, downloads, resident and pooled bytes, evictions, per-kernel dispatch counts |
| `cuda_hip::counters()` | Process-wide driver call counters (launches, captured, graph launches, copies, syncs) read by `forge bench` and `tools/bench_llm.sh` |

OpenCL (`gpu::ocl`) and Level Zero (`gpu::ze`) runtimes have `open(&why, ordinal)`, `device_count()`, `run_op`, the residency protocol, peer copies, and recorded "graphs" (enqueues replayed with the same arguments). They also have `run_qmatmul`, `run_qglu`, `run_qgather`, `run_state`, `run_bcast`, `run_moe` and `has_function` with the CUDA/HIP signatures, from one shared implementation (`gpu/portable_kernels.hpp`), plus `program_from_cache()` / `kernel_cache_path()`.

## Portable kernels (OpenCL and Level Zero)
`ops/gen/gpu_gen.py` has one portable source (`PORT_HELPERS`, `PORT_SHARED`, `PORT_KERNELS`) for the kernel families that used to exist only for CUDA/HIP. Every kernel is one work item per output element (or per row-head, per state column), uses no workgroup-shared memory, barriers or vendor intrinsics, and so builds on any OpenCL 1.2 device. `gpu::portable_kernels<Runtime>` (`gpu/portable_kernels.hpp`) is the host side for both runtimes.

| Work | Kernel | Launch | Disable |
|------|--------|--------|---------|
| Quantised `matmul_t` | `forge_qmatmul_t`: decodes each 32-value group of the weight row (same decoders as `kernels.cu`) and accumulates with f32 `fma` over k in index order. The f32 activations are used as they are: no q8_1 / int8 rounding, so results agree with the CPU's exact quantised kernel to ~1e-6 | rows × cols | `FORGE_GPU_QMATMUL=0` |
| FFN gate/up/SiLU | `forge_qglu`: both dots and `silu(gate) * up` (the expression of `forge_silu_mul`) in one launch; q8_0, q4_k, q5_k, q6_k; any shape, inside or outside a run. Bit-identical to two `forge_qmatmul_t` + `forge_silu_mul` on the same device | rows × n | `FORGE_FUSE_GLU=0` |
| Quantised gather | `forge_qgather`: one element of one encoded row per work item; an index past the table gives zeros | indices × dim | `FORGE_GPU_QGATHER=0` |
| Attention, prefill | `forge_attention_fa`: one pass with an online softmax (the spec kernel `forge_attention` reads every key twice). Plain f32 K/V; causal, sliding window, GQA | rows × heads | `FORGE_GPU_NO_FA` (any value) |
| Attention, decode | `forge_attention_dec` + `forge_attention_comb`: the visible keys in 16 contiguous chunks per (row, head), merged in chunk order; the launch shape does not change as the context grows | rows × heads ≤ 256, f32 K/V | `FORGE_ATTN_DEC=0`; off in [determinism mode](#determinism-mode) |
| Gated DeltaNet | `forge_delta_rule`: a work item per (value head, state column), any dk; `forge_causal_conv`: a work item per channel, kernel width 1–9 | inside a run; the state slot stays resident and pinned | `FORGE_GPU_NO_STATE_OPS` (any value) |
| Broadcast binaries | `forge_bcast` | the smaller shape is a suffix of the larger | |
| Mixture of experts | `forge_moe_hidden` + `forge_moe_down` (also in `kernels.cu`): see [Mixture of experts](#mixture-of-experts-on-the-device) | | `FORGE_GPU_MOE=0` |

KV codecs (`int8`, `fp8`, `tbq4`, `tbq3`) keep using the spec kernel `forge_attention`. Encoded weights are uploaded once, keyed by host address and size, and are evicted least recently used first under memory pressure (the host copy is re-uploaded on the next use). These kernels are correct-first: a work item re-decodes its weight row and the DeltaNet item recomputes the token's q/k norms, so they are slower per flop than the tuned CUDA kernels. None of them has run on real hardware.

### Quantised matmul: exact and int8-activation paths
A quantised `matmul_t` has two families of kernels, and they give slightly different numbers for the same model file:

| Path | Where | What it computes |
|------|-------|------------------|
| exact | the portable `forge_qmatmul_t` / `forge_qglu` (OpenCL, Level Zero), the CPU with `FORGE_CPU_INT8=0` or `FORGE_DECODE_WEIGHTS=1`, AArch64 CPUs by default | f32 activations times the decoded weights: the exact output of the quantised model, to f32 rounding |
| int8 activations | the CPU's integer kernels (x86 default), CUDA / HIP (`forge_qmatmul_q81`, the tensor-core kernels) | the activations are rounded to 8 bits per 32-value block first (q8_1, as llama.cpp does), then integer dot products: up to about 1.6 % of the largest output on test inputs with a wide dynamic range |

`tests/test_cuda_hip.cpp` (`exact_reference_check`) holds every backend to one reference, the float64 dot product of the f32 activations with the codec-decoded weights: within 2e-5 of the largest output for the exact backends (measured on the OpenCL mock: 1e-6), within 3e-2 for the int8-activation ones (CUDA / HIP mocks: 1.6e-2).

This is what was behind a q8_0 logit cosine of 0.9984 on the OpenCL mock against 0.9990 on the CPU and the CUDA mock for the Gemma-2 model of `verify/golden/tiny_gemma.py`. The OpenCL run was bit-for-bit the exact path (cosine 1.000000 with `FORGE_CPU_INT8=0` on the CPU); soft-capping, scale rounding and intermediates in the portable kernels were not involved. That model's attention scores reached 28 against a soft-cap of 5, which made its logits so sensitive to q8_0 weight rounding that over 12 prompts the cosine against the unquantised reference ran from 0.945 to 0.9995 on the int8 path and from 0.979 to 0.9995 on the exact one (the exact path was the closer one for 11 of the 12; the golden's prompt was the twelfth). The golden now uses weights whose scores pass the cap by a factor of 3 (6.4 against 2), and its bound is 0.999 for every generation on every path (measured: 0.99936 CPU and CUDA mock, 0.99977 OpenCL mock).

## Mixture of experts on the device
`moe_ffn` runs on CUDA, HIP, OpenCL and Level Zero when the three stacked expert weights are encoded (q8_0, q4_k, q5_k, q6_k) or f32 constants (`registry::device_moe`, `run_moe`; `FORGE_GPU_MOE=0` keeps it on the host):

1. **Router on the host.** The router logits are synced back (`to_host`), and each token's top-k experts and weights come from `ops::ref::moe_route`, the same function the CPU path uses, so ties and the softmax / sigmoid gating are identical.
2. **Route table.** Three floats per (token, slot) pair (expert id, token row, weight) are uploaded to a scratch buffer.
3. **`forge_moe_hidden`**: a work item per (pair, hidden column) computes `silu(gate_e[c] · h) * (up_e[c] · h)` straight from the stacked `[e, f, d]` weights.
4. **`forge_moe_down`**: a work item per (token, output column) sums `w * (down_e[c] · hidden)` over the token's slots in slot order.

Only the routed experts' rows are read (sparse dispatch, like the CPU "moe-sparse" path); the stacked weights are uploaded once and stay on the device. The same two kernels are in `kernels.cu` (CUDA/HIP dialect) and `kernels.cl`. The CUDA/HIP launches sync at the end of the op (they use temporaries), so a decode graph segment ends there. Verified against the CPU path by `moe_check` in `test_cuda_hip` and by `golden_moe_mla_{cuda,hip,opencl,ze}_mock`; never run on hardware. Vulkan and Metal have no MoE kernel.

## Kernel program cache
OpenCL builds `kernels.cl` with the driver at `open()`; Level Zero compiles the SPIR-V. Both are cached on disk (`gpu/kernel_cache.hpp`), in the same directory as the Vulkan pipeline cache: `$FORGE_CACHE_DIR`, else `$XDG_CACHE_HOME/forge`, else `~/.cache/forge` (`%LOCALAPPDATA%\forge` when none of those is set).

| | OpenCL | Level Zero |
|---|---|---|
| Saved | `clGetProgramInfo(CL_PROGRAM_BINARIES)` after a source build | `zeModuleGetNativeBinary` after the IL build |
| Loaded with | `clCreateProgramWithBinary` + `clBuildProgram` | `zeModuleCreate` with `ZE_MODULE_FORMAT_NATIVE` |
| File | `cl-program-<hash>.bin` | `ze-module-<hash>.bin` |
| Key | device name, vendor, device version, driver version, hash of the OpenCL C, build options | device name, vendor and device id, device UUID, driver version, hash of the image, build flags |

The whole key is stored in the file and compared, not just its hash. The file also carries the payload size and an FNV-1a checksum. A missing, truncated, edited or foreign file, a binary the driver refuses, or one without the kernels (`forge_scale` is probed) falls back to the source build, which rewrites the file. Files are written through a temporary file and a rename, with owner-only permissions (0600; a directory the cache creates is 0700). On POSIX a file that is a symlink, is not owned by the current user or is writable by group/others is not read. `FORGE_GPU_KERNEL_CACHE=0` turns reading and writing off. `program_from_cache()` says whether `open()` used the cache.

A program binary is code the driver runs: do not point `FORGE_CACHE_DIR` at a directory other users can write. Metal has no cache (`MTLBinaryArchive` is not used). Tested with the mock drivers only (`test_gpu_kernel_cache_{opencl,ze}_mock`); whether a given real driver accepts its own binaries back has not been checked.

## Automatic device selection
`FORGE_DEVICE=auto` (or `device_registry("auto")`, `session::set_device("auto")`) opens every device of every backend, builds a capability record for each and picks the best score (`gpu/select.hpp` is the rule, `graph::enumerate_devices` / `graph::auto_device` in `graph/vulkan.hpp` fill it in). The table with every candidate's score is logged at info level. The default without `FORGE_DEVICE` is still the CPU.

Record: backend, ordinal, driver name, class (`discrete`, `integrated`, `virtual`, `software`, `cpu`), device memory, the weight codecs with a device quantised matmul, whether gather / state ops / MoE run on the device, and the micro-benchmark.

Score = throughput × codec coverage × memory fit × preference:

| Factor | Value |
|---|---|
| Throughput | the cached micro-benchmark when **every** eligible candidate has one, else the class prior: discrete 200, integrated 20, virtual 10, cpu 4, software 0.5 |
| Codec coverage | 0.25 + 0.75 × the share of the model's weight codecs the device has a quantised matmul for (unknown model: 1) |
| Memory fit | 1 if the model's data section fits in 90% of the device memory (or a size is unknown), else 0.01 |
| Preference | 1000 for a backend or device named in `FORGE_DEVICE_PREFER` |

Hard rules: a backend or device in `FORGE_DEVICE_EXCLUDE` is never chosen; a **software device** (Vulkan `CPU` type such as lavapipe or SwiftShader, an OpenCL CPU runtime, a device whose name says llvmpipe / SwiftShader / PoCL / mock) never wins over the CPU unless `FORGE_DEVICE_ALLOW_SOFTWARE=1`, and then only if it scores higher (in practice: when it is also preferred). Ties go to the backend order cuda, hip, metal, level_zero, opencl, vulkan, then the lower ordinal. Naming a device explicitly (`FORGE_DEVICE=vulkan`) bypasses all of this.

The micro-benchmark is a 128×512 · 512×512 f32 `matmul_t` through the device's own lowering, best of 3, data crossing host memory each time (GFLOP/s). It is measured once per (backend, device name) and stored in the kernel cache directory (`device-bench-<hash>.bin`); `FORGE_DEVICE_BENCH=0` never measures. It is a tie-breaker between real devices, not a model benchmark: it does not exercise the quantised kernels or residency.

Classes come from the driver: Vulkan `VkPhysicalDeviceType`; OpenCL `CL_DEVICE_TYPE` and `CL_DEVICE_HOST_UNIFIED_MEMORY`; Level Zero device type and the `INTEGRATED` flag; CUDA and HIP devices count as discrete; Metal as integrated. Cost: `auto` opens every device once (kernel builds are cached, see above).

Tested with synthetic records (`test_gpu`: `device_selection_scoring`) and with the mock drivers at `MOCK_DEVICES=2` (`test_gpu_select_*_mock`). No real multi-GPU machine has run it; the priors are guesses, not measurements.

## Determinism mode
`FORGE_DETERMINISTIC=1`, `gpu::set_deterministic(true)` or `inference::session_options::deterministic = 1` (`gpu/determinism.hpp`; the explicit calls win over the variable, and the mode is process-wide).

**Guarantee:** the same build on the same device and driver gives bit-identical results for the same inputs, run after run. It does not make two devices, two backends, two driver versions or two builds agree, and it is not a statement about the CPU against a device.

With the mode on, every runtime takes only paths whose reduction order depends on the shapes alone:

| Backend | Turned off |
|---|---|
| CUDA | split-K and stream-K tensor-core matmuls (their k-sum order follows the launch grid, which follows how many blocks the GPU holds at once), the split-K decode attention, the multi-block `forge_argmax` (one block scans the buffer), whole-run replay |
| HIP | the multi-block argmax (HIP has no split-K kernels: they are in the NVPTX-only part of `kernels.cu`) |
| OpenCL, Level Zero | the split-over-keys decode attention (`forge_attention_fa` / `forge_attention` run instead) |
| Vulkan, Metal | nothing: their kernels have one fixed order per shape already |
| CPU (`gpu::graph::launch`) | chunk boundaries are fixed at 4096 items whatever the worker count |

The runtimes read the switch at each launch, so it can be changed between runs; `plan::deterministic` records the mode a plan was compiled under and `describe()` prints it. `FORGE_MMA_STREAMK` and `FORGE_MMA_SPLITK` are ignored while the mode is on. Cost: prefill matmuls on ragged grids lose the split-K / stream-K speedup, and decode attention runs as one work item (block) per row-head. Tested with the mock drivers (`test_gpu_determinism_{cuda,hip,opencl,ze}_mock`: no order-varying launch is dispatched, repeated runs are bit-identical). The mocks execute serially, so they cannot show a real device's scheduling; on hardware this is untested.

## Programming model API (`gpu.hpp`, `prims.hpp`)

| Item | Notes |
|------|-------|
| `probe_backends()` | One `backend_probe {kind, available, detail}` per `backend_kind` (`cpu`, `cuda`, `hip`, `level_zero`, `vulkan`, `metal`, `opencl`). It only checks that the driver library loads. `metal` is always reported absent. `name(kind)` gives the string |
| `select(prefer = prefer::any)`, `all()` | Always the CPU device (`caps().name` is `cpu/<simd target>`): the programming model executes on the CPU. Devices for model execution are scored by `graph::auto_device` |
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
| `FORGE_GPU_NO_FA` | unset | CUDA/HIP, OpenCL, Level Zero | Set: no flash-attention kernels (`forge_attention_fa`, `forge_attention_fa2_*`, `forge_attention_fa2tc_*`) |
| `FORGE_FA_TC` | on | CUDA | `0`: tiled prefill attention uses the f32 SIMT kernel (`forge_attention_fa2_*`) instead of the fp16 tensor-core one |
| `FORGE_ATTN_DEC` | on | CUDA/HIP, OpenCL, Level Zero | `0`: no split-K decode attention |
| `FORGE_MMVQ` | on | CUDA/HIP | `0`: no q8_1/dp4a decode matmul |
| `FORGE_MMVQ_CPB` | per shape | CUDA/HIP | `1`, `2` or `4`: output columns per decode-matmul block (default 2 for q4_k with k ≤ 6144, else 1) |
| `FORGE_MMVQ_Q8V` | on | CUDA/HIP | `0`: q8_0 decode matmul uses the older 4-bytes-per-lane loop |
| `FORGE_MMVQ_NC` | off | CUDA | `1`: decode-matmul weight loads use `ld.global.nc.L1::no_allocate` (no faster on an RTX 3090) |
| `FORGE_MMQ_MIN_ROWS` | 16 | CUDA/HIP | Rows at which the tiled MMA / MMQ kernels are used; `0` disables them |
| `FORGE_MMA` | on | CUDA | `0`: no int8 tensor-core prefill matmul (and no q6_k/q8_0 repacking); MMQ is used instead |
| `FORGE_MMA_BT` | per codec | CUDA | `64` or `128`: tokens per MMA block |
| `FORGE_MMA_STREAMK` | off | CUDA | `1`: persistent stream-K tensor-core matmul for ragged grids of 1/4 to 4 waves; `2`: for every ragged grid |
| `FORGE_MMA_SK_MIN` | 4 | CUDA | Minimum 256-value k chunks per stream-K block |
| `FORGE_DELTA_RULE_OLD` | unset | CUDA | Set: use `forge_delta_rule` instead of the register-state kernels |
| `FORGE_DELTA_RULE_R` | unset | CUDA | Set: use the `forge_delta_rule_rN` register-state kernels instead of the staged `_pN` ones |
| `FORGE_CAUSAL_CONV_OLD` | unset | CUDA/HIP | Set: use the per-channel `forge_causal_conv` instead of `forge_causal_conv_t` |
| `FORGE_FUSE_Q81` | on | CUDA/HIP | `0`: rms_norm/silu_mul/attention_comb do not emit q8_1 |
| `FORGE_FUSE_Q8R` | on | CUDA | `0`: prefill rms_norm does not emit the tensor-core matmul's int8 rows, and matmuls on the same activation do not share them |
| `FORGE_FUSE_ADD_NORM` | on | CUDA | `0`: residual add and the following rms_norm run as two kernels |
| `FORGE_FUSE_QK_ROPE` | on | CUDA | `0`: QK-norm rms_norm and rope run as two kernels |
| `FORGE_ROPE_HPB` | per shape | CUDA | Heads per `forge_rms_rope` block (tuning) |
| `FORGE_FUSE_GLU` | on | CUDA, OpenCL, Level Zero | `0`: FFN gate and up run as two matmuls plus `silu_mul` (see [FFN gate/up fusion](#ffn-gateup-fusion)) |
| `FORGE_GPU_QMATMUL` | on | CUDA/HIP, OpenCL, Level Zero | `0`: no device quantised matmul; weights are decoded to f32 within `FORGE_GPU_WEIGHT_BUDGET` |
| `FORGE_GPU_QGATHER` | on | CUDA/HIP, OpenCL, Level Zero | `0`: no device quantised gather |
| `FORGE_GPU_NO_STATE_OPS` | unset | CUDA/HIP, OpenCL, Level Zero | Set: DeltaNet state ops run on the host |
| `FORGE_GPU_MOE` | on | CUDA/HIP, OpenCL, Level Zero | `0`: `moe_ffn` runs on the host ([Mixture of experts](#mixture-of-experts-on-the-device)) |
| `FORGE_DETERMINISTIC` | off | all | `1`: [determinism mode](#determinism-mode) |
| `FORGE_DEVICE_PREFER` | unset | `auto` | Comma list of backends or devices that win whenever eligible (`cuda`, `opencl:1`) |
| `FORGE_DEVICE_EXCLUDE` | unset | `auto` | Comma list of backends or devices never chosen |
| `FORGE_DEVICE_ALLOW_SOFTWARE` | off | `auto` | `1`: software devices may be chosen |
| `FORGE_DEVICE_BENCH` | on | `auto` | `0`: no micro-benchmark; class priors only |
| `FORGE_GPU_KERNEL_CACHE` | on | OpenCL, Level Zero | `0`: no on-disk [kernel program cache](#kernel-program-cache) |
| `FORGE_GRAPHS` | on | CUDA | `0`: do not capture decode runs into CUDA graph segments |
| `FORGE_GRAPH_SEG` | 64 | CUDA | Kernels per captured segment; `0` = unlimited |
| `FORGE_GRAPH_SEG0` | 8 | CUDA | Kernels in a run's first segment, doubling per segment up to `FORGE_GRAPH_SEG`; `0` = all segments full size |
| `FORGE_REPLAY_EAGER` | on | CUDA | `0`: whole-run replay patches every segment before launching any |
| `FORGE_GPU_STAGE` | on | CUDA | `0`: small host-to-device copies go from pageable memory |
| `FORGE_CUDA_SCHED` | auto | CUDA | `spin`, `yield` or `block`: how the host waits on the device |
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
| `FORGE_CACHE_DIR` | `$XDG_CACHE_HOME/forge` or `~/.cache/forge` | Vulkan, OpenCL, Level Zero | Directory of the pipeline cache and the kernel program cache |
| `FORGE_VK_DEBUG` | unset | Vulkan | Set: log each op/shape the Vulkan lowering refuses |
| `FORGE_PLAN_TRACE` | unset | all devices | Set: at exit, print the ops that ran on the host during device runs |
| `FORGE_MEM_DEVICE_LIMIT` | unset | all pools | Device pool limit ([memory.md](memory.md)) |

Test hooks, not for normal use: `FORGE_VK_SSBO_ALIGN` (emulate a stricter storage-buffer alignment), `FORGE_OPENCL_TEST_ALIGN`, `FORGE_ZE_TEST_IMAGE=opencl-c` (hand OpenCL C to the mock Level Zero driver). The mock drivers read `MOCK_DEVICES`, `MOCK_MEM_LIMIT`, `MOCK_ALIGN`, `MOCK_REJECT_BINARY` (refuse every cached program binary) and `MOCK_DRIVER_VERSION`; the OpenCL and Level Zero mocks put a canary after every device buffer and abort when a kernel writes past one.

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
  - **HIP, OpenCL, Level Zero**: never on real hardware. HIP, OpenCL and Level Zero run against mock driver libraries (`tests/mock_cuda.c` serves CUDA and HIP, `mock_opencl.c`, `mock_ze.c`) in `test_gpu_*_mock`, `test_gpu_memory_*_mock`, `test_gpu_determinism_*_mock`, `test_gpu_kernel_cache_*_mock`, `golden_tiny_llama_*_mock` and `golden_moe_mla_*_mock`. The OpenCL and Level Zero mocks compile the generated OpenCL C as C and run it, so the portable kernels (quantised matmul / gather / GLU, attention, DeltaNet, MoE) are checked as generated; whether a real OpenCL C compiler accepts them has not been checked. `mock_cuda.c` has hand-written C versions of the CUDA kernels, so for CUDA/HIP the mock checks the host code and the algorithm, not the PTX / code object; the two MoE kernels were compiled to PTX and AMDGPU code objects (2026-11-12) and have not run on a GPU. The AMD code objects are only compile-checked (`gpu_compile_hip`).
- Graphics interop: the Vulkan export/import round trip has run on lavapipe. The CUDA import has not run on hardware.
- Known gaps: no eviction while a CUDA/HIP stream capture or an OpenCL/Level Zero recording is open, and graphs recorded before an eviction must not be replayed after it.
