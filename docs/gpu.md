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
| `kernels.cu` | CUDA, HIP | clang, no SDK: PTX for `sm_70` (JIT-compiled by the driver); AMDGPU code objects for `FORGE_HIP_ARCHS` (default `gfx90a gfx942 gfx1100`, needs lld) |
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

### Kernel selection
`run_op` and the quantised entry points choose kernels by shape:

| Work | Kernel | When | Disable |
|------|--------|------|---------|
| Attention, prefill | `forge_attention_fa`: a block per (query row, head), tiled online softmax | default | `FORGE_GPU_NO_FA` (any value) falls back to `forge_attention` |
| Attention, decode | `forge_attention_dec` + `forge_attention_comb`: split-K over 16 key chunks, then a combine pass | rows × heads ≤ 256, f32 KV, head dims multiples of 32 and ≤ 256 | `FORGE_ATTN_DEC=0` |
| Quantised matmul_t, many rows | `forge_qmatmul_mmq`: 64×64 tiles | rows ≥ `FORGE_MMQ_MIN_ROWS` (default 16) | `FORGE_MMQ_MIN_ROWS=0` |
| Quantised matmul_t, few rows (decode) | `forge_quant_q81` + `forge_qmatmul_q81`: activations quantised to q8_1, dp4a dot, a warp per output column | q8_0, q4_k, q6_k weights | `FORGE_MMVQ=0` |
| Quantised matmul_t, otherwise | `forge_qmatmul_t` | q5_k at few rows, or MMVQ off | |
| Embedding gather from encoded rows | `forge_qgather` | q8_0, q4_k, q6_k, q5_k tables | `FORGE_GPU_QGATHER=0` |
| rms_norm | `forge_rms_norm_b` (block per row) | when present | |
| q8_1 fusion | `forge_rms_norm_q81`, `forge_silu_mul_q81` also write the q8_1 form of their output for the next matmul | decode rows inside a run, with MMVQ on | `FORGE_FUSE_Q81=0` |
| Broadcast add/sub/mul/div | `forge_bcast` | the smaller shape is a suffix of the larger | |
| Gated DeltaNet state ops | `forge_delta_rule` (dk ≤ 256), `forge_causal_conv` (kernel width 1–9) | inside a run; state slot stays resident | `FORGE_GPU_NO_STATE_OPS` (any value) |
| Device copies | `forge_copy2d` (concat), `forge_copy_words` (copies inside a captured segment) | | |

Encoded weights (q8_0, q4_k, q6_k, q5_k) are uploaded once and stay on the device. Consecutive decode matmuls that read the same activation reuse its q8_1 form instead of re-quantising.

Attention supports the KV codecs `int8`, `fp8`, `tbq4` and `tbq3` (tbq only up to head dim 256). `tbq4p` and `tbq3p` run on the CPU only. The same holds for Vulkan.

### Graphs and replay (CUDA)
- **User capture**: `capture_begin()`, `capture_end()` → id, `graph_launch(id)`. Streams are captured in relaxed mode so `run_op` may allocate while capturing. Needs the driver's graph API (`has_graphs()`); both CUDA and HIP export it.
- **Decode graphs** (`FORGE_GRAPHS=1`, CUDA only, opt-in). Inside a run, kernel launches are stream-captured in segments. A segment ends at a host copy, a free, a sync, or after `FORGE_GRAPH_SEG` kernels (default 64; 0 = no split). Each segment's exec graph is cached by index. On later tokens, if the sequence of kernels and launch shapes matches, only the kernel nodes whose arguments changed are patched (`cuGraphExecKernelNodeSetParams`) and the graph is relaunched, without capturing. This needs `cuFuncGetParamInfo` (CUDA 12.4+). Without it, or with `FORGE_GRAPH_REPLAY=0`, each segment is re-captured and refreshed with `cuGraphExecUpdate`.
- **Whole-run replay** (`FORGE_GPU_REPLAY` set, experimental, opt-in; also needs `FORGE_GRAPHS=1` and the fixed arena). When every step of a run was a recorded kernel, the runtime predicts the next run's arguments from the last two (decode moves the KV write row and attention length by one each token) and relaunches all segments without the executor. A run becomes replayable only after a real run confirmed a prediction. The executor does one real run every `FORGE_REPLAY_VERIFY` runs (default 32). A wrong prediction disables replay for that plan and prints a warning.
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
| `FORGE_GPU_NO_FA` | unset | CUDA/HIP | Set: no flash-attention kernel |
| `FORGE_ATTN_DEC` | on | CUDA/HIP | `0`: no split-K decode attention |
| `FORGE_MMVQ` | on | CUDA/HIP | `0`: no q8_1/dp4a decode matmul |
| `FORGE_MMQ_MIN_ROWS` | 16 | CUDA/HIP | Rows at which the tiled MMQ kernel is used; `0` disables it |
| `FORGE_FUSE_Q81` | on | CUDA/HIP | `0`: rms_norm/silu_mul do not emit q8_1 |
| `FORGE_GPU_QMATMUL` | on | CUDA/HIP | `0`: no device quantised matmul; weights are decoded to f32 within `FORGE_GPU_WEIGHT_BUDGET` |
| `FORGE_GPU_QGATHER` | on | CUDA/HIP | `0`: no device quantised gather |
| `FORGE_GPU_NO_STATE_OPS` | unset | CUDA/HIP | Set: DeltaNet state ops run on the host |
| `FORGE_GRAPHS` | off | CUDA | `1`: capture decode runs into CUDA graph segments |
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
  - **CUDA** on an NVIDIA RTX 3090. 2026-10-31 (WSL2, CUDA 13, clang 23.1.4): full ctest 104/104. 2026-11-04 (Windows 11, mingw-w64 build): `test_gpu_cuda` passed 31/31 lowerings against the oracle, 24/24 quantised matmul cases, 7/7 split-K decode-attention shapes, and delta_rule/causal_conv matched the CPU. The same day a bug was fixed where the split-K combine kernel was launched with dangling argument pointers (commit 7fe8a83); the 3090 had passed by luck, the mocks failed.
  - **Vulkan** on Mesa lavapipe and SwiftShader only (software devices).
  - **Metal** on the GitHub `macos-latest` runner's GPU: `test_gpu_metal` passes in CI (oracle check of every `metal.*` lowering). The device model is not logged.
  - **HIP, OpenCL, Level Zero**: never on real hardware. HIP, OpenCL and Level Zero run against mock driver libraries (`tests/mock_cuda.c` serves CUDA and HIP, `mock_opencl.c`, `mock_ze.c`) in `test_gpu_*_mock`, `test_gpu_memory_*_mock` and `golden_tiny_llama_*_mock`. The OpenCL and Level Zero mocks compile the generated OpenCL C as C and run it. The AMD code objects are only compile-checked (`gpu_compile_hip`).
- Graphics interop: the Vulkan export/import round trip has run on lavapipe. The CUDA import has not run on hardware.
- Known gaps: no eviction while a CUDA/HIP stream capture or an OpenCL/Level Zero recording is open, and graphs recorded before an eviction must not be replayed after it.
