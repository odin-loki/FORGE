# FORGE implementation status

Implemented and tested on an 8-core Xeon (AVX-512, AMX; no GPU). aarch64 was cross-built with zig and tested under qemu.

## Done (tested by ctest: 8/8 on x86_64, 6/6 on aarch64 under qemu)
- **core / spec**: dtypes, shapes, the op spec and a float64 oracle for every op.
- **lane**: the target-tag SIMD library with native lowerings for scalar, AVX2, AVX-512 and NEON. Math (exp/log/tanh/sigmoid/silu/gelu) is ≤3 ulp and agrees across lowerings. Runtime dispatch, an ULP harness, and GEMM.
- **verify**: oracle comparison, ULP and cosine metrics.
- **grid**: a CPU backend model with graph replay, backend probing, prims, GEMM (241 GFLOP/s at 512³ on all cores) and Philox RNG.
- **flow**: IR, verifier, CSE/fold/fuse/DCE passes (random-graph equivalence tested), memory planner, plans, quant codecs (q8_0, q4_k) and state_append.
- **pack**: the `.fpk` container (mmap, 64-byte alignment, BLAKE3 hashes), JSON, the BPE tokenizer, and the blueprint language (`.fbp`) feeding the graph builder.
- **import**: HF safetensors (F32/F16/BF16/F64) through the Llama template, and GGUF (F32/F16/BF16/Q8_0/Q4_K kept verbatim, with a vocabulary-derived tokenizer).
- **mind**: session with a KV cache, a plan cache, sampler (temperature, top-k, top-p, min-p, repetition penalty), and the C ABI (`forge.h`, `libforge.so`).
- **CLI** `forge`: info, import, inspect, run, bench (`--gemm`), dump.
- **Golden test**: a tiny Llama checked against a NumPy float64 reference (f32 logit error 6e-7; q8_0 and GGUF cosine ≥0.9999).

## Remaining
- GPU backends: CUDA, HIP, Level Zero, Vulkan and Metal. Only the probe stubs exist; none can be built or tested without GPU hardware.
- SVE/SVE2 and RVV lowerings. The tags exist, but these targets fall back to emulation.
- Paged KV, continuous batching, speculative decoding, grammar-constrained sampling, MoE/MLA blueprints.
- `forge-serve` HTTP server and Python bindings.
- Pack signing, fuzzing harnesses, ONNX import, and on-the-fly `forge run <hf-dir>`.
- CI: the workflow file is at `ci/github-actions-ci.yml`. The push token lacks `workflow` scope, so move it to `.github/workflows/ci.yml` manually.

## Build
    cmake -S . -B build -G Ninja && ninja -C build && (cd build && ctest)
Cross aarch64: use a toolchain file that sets `CMAKE_SYSTEM_PROCESSOR=aarch64` and `CMAKE_CROSSCOMPILING_EMULATOR=qemu-aarch64`, then build with `-DCMAKE_EXE_LINKER_FLAGS=-static -DCMAKE_POSITION_INDEPENDENT_CODE=ON`.
