# FORGE implementation status

Built and tested on an 8-core Xeon (AVX-512, AMX) with no GPU. The Vulkan backend runs on Mesa lavapipe (a software Vulkan device). aarch64 and riscv64 are cross-built with zig and tested under qemu-user.

## Test matrix (ctest)
| Build | Result |
|---|---|
| x86_64 dev | 25/25. This includes 8 fuzz targets, the Vulkan kernels on lavapipe, and the CUDA/HIP compile checks. |
| aarch64, qemu `-cpu max` (VL 512, so the NEON path runs) | 13/13 without the fuzz label. `test_vulkan` skips because the build is static. |
| aarch64 SVE-256, qemu `-cpu max,sve256=on` | 13/13 without the fuzz label. The `sve` lane is active. |
| riscv64 RVV-256, qemu `-cpu rv64,v=true,vlen=256` | 13/13 without the fuzz label. The `rvv` lane is active. At VLEN=128 it falls back to scalar. |

## Done
- **core / spec**: dtypes, shapes, a declarative op spec (`spec/ops/*.fspec`) with codegen, and a float64 oracle for every op.
- **lane**: target-tag SIMD with native lowerings for scalar, AVX2, AVX-512, NEON, **SVE-256** and **RVV-256**.
  - SVE and RVV use fixed-length vector types. The runtime selects them only when the hardware VL/VLEN is 256 bits; any other length falls back to NEON or scalar.
  - Math functions are within 3 ulp and agree across lowerings. Also included: runtime dispatch (HWCAP/cpuid), an ULP harness, and GEMM.
- **grid**:
  - CPU backend with graph replay, backend probing, prims, GEMM and Philox RNG.
  - **GPU kernels are generated from the spec** (`spec/gen/gpu_gen.py`): 29 kernels covering elementwise unary and binary ops, scale, silu_mul, softmax, rms_norm and matmul_t.
    - GLSL is compiled to SPIR-V with glslangValidator, validated with spirv-val, and embedded in the binary.
    - The **Vulkan backend** (`grid/vulkan.hpp`) dlopens libvulkan, so there is no link dependency. Every kernel is checked against the spec reference (`test_vulkan`, max relative error 3e-7).
    - The same source compiles to **CUDA PTX (sm_80) and HIP AMDGCN (gfx90a)** with clang, without an SDK (`gpu_compile_cuda`, `gpu_compile_hip`).
- **flow**: IR, verifier, CSE/fold/fuse/DCE passes, memory planner, quant codecs (q8_0, q4_k).
- **pack**:
  - `.fpk` container (mmap, BLAKE3) and **Ed25519 signing** (`forge keygen/sign/verify`, vendored Monocypher, cross-checked against pyca/cryptography).
  - JSON, BPE tokenizer, blueprint language, GBNF grammars.
- **import**:
  - HF safetensors: Llama, Mixtral **MoE**, and DeepSeek-V2 **MLA** (absorbed).
  - GGUF.
  - **ONNX**: a protobuf decoder translating to a blueprint. Supports MatMul, Gemm, elementwise ops, Gelu, Softmax, LayerNorm, Transpose, Reshape, Flatten and Reduce*. Run with `forge run model.onnx --input x=x.f32`.
- **mind**: paged KV, continuous batching with a prefix cache, speculative decoding, grammar-constrained sampling, samplers, and the C ABI v2.
- **apps**: the `forge` CLI (info/import/inspect/run/bench/dump/keygen/sign/verify), the OpenAI-compatible `forge-serve`, and Python bindings.
- **Fuzzing** (`fuzz/`): json, blueprint, grammar, gguf, fpk, tokenizer, onnx and more, each with a corpus. Parsers were hardened against every crash the fuzzers found.
- **Goldens**: tiny Llama and MoE/MLA, checked against NumPy float64 references.

## Remaining
- Vulkan is not wired into graph execution yet. Sessions still run on the CPU; the backend runs single ops through `grid::vk::run_op`.
- The CUDA/HIP kernels are compiled but have not been run: there is no GPU hardware, and the CUDA/HIP host runtime is not written.
- Level Zero and Metal: only probes exist.
- SVE and RVV kernels are fixed at 256-bit. Other vector lengths fall back to NEON or scalar.
- CI: the workflow file is at `ci/github-actions-ci.yml`. The push token lacks `workflow` scope, so it has to be moved to `.github/workflows/ci.yml` by hand.

## Build
    cmake -S . -B build -G Ninja && ninja -C build && (cd build && ctest)
The Vulkan backend needs `glslangValidator` and `vulkan/vulkan.h` at build time. At run time any loader and ICD works; lavapipe is enough for testing. The GPU compile checks need clang ≥18.

### Cross builds under qemu
Toolchain files are in `cmake/toolchains/` (`aarch64-sve-qemu.cmake`, `riscv64-rvv-qemu.cmake`). Configure with:

    -DCMAKE_TOOLCHAIN_FILE=... -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXE_LINKER_FLAGS=-static -DCMAKE_POSITION_INDEPENDENT_CODE=ON

To compile with zig instead (`-DFORGE_CROSS_CXX=<wrapper running "zig c++ -target aarch64-linux-musl">`), note that zig rejects `-march`. The CMake probes select the zig-compatible spellings:
- SVE: `-mcpu=generic+sve -msve-vector-bits=256`
- RVV: `-mcpu=baseline_rv64+v+zvl256b -mrvv-vector-bits=256`

The emulators are `qemu-aarch64 -cpu max,sve256=on` and `qemu-riscv64 -cpu rv64,v=true,vlen=256`.
