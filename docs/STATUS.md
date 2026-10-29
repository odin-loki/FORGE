# FORGE implementation status

Built and tested on an 8-core Xeon (AVX-512, AMX) with no GPU. The Vulkan backend runs on Mesa lavapipe (a software Vulkan device). aarch64 and riscv64 are cross-built with zig and tested under qemu-user.

## Test matrix (ctest)
| Build | Result |
|---|---|
| x86_64 dev | 27/27. This includes 8 fuzz targets, the Vulkan kernels on lavapipe, both goldens end-to-end on Vulkan (`FORGE_DEVICE=vulkan`), and the CUDA/HIP compile checks. |
| aarch64, qemu `-cpu max` (VL 512, so the NEON path runs) | 13/13 without the fuzz label. `test_vulkan` skips because the build is static. |
| aarch64 SVE, qemu `-cpu max,sveN=on`, N = 128/256/512 (1024 requested; qemu caps `max` at 512) | 13/13 at each length without the fuzz label. The `sve` lane is active at each VL. |
| riscv64 RVV, qemu `-cpu rv64,v=true,vlen=N`, N = 128/256/512/1024 | 13/13 at each length without the fuzz label. The `rvv` lane is active at each VLEN. |

## Done
- **core / spec**: dtypes, shapes, a declarative op spec (`spec/ops/*.fspec`) with codegen, and a float64 oracle for every op.
- **lane**: target-tag SIMD with native lowerings for scalar, AVX2, AVX-512, NEON, **SVE** and **RVV** at any vector length from 128 to 1024 bits.
  - SVE and RVV are built once per length (128/256/512/1024) as fixed-length tables. At startup, dispatch reads the hardware VL (`rdvl` / `vlenb`) and picks the matching table.
  - Math functions are within 3 ulp and agree across lowerings. Also included: runtime dispatch (HWCAP/cpuid), an ULP harness, and GEMM.
- **grid**:
  - CPU backend with graph replay, backend probing, prims, GEMM and Philox RNG.
  - **GPU kernels are generated from the spec** (`spec/gen/gpu_gen.py`): 33 kernels covering elementwise unary and binary ops, scale, silu_mul, softmax, rms_norm and matmul_t, rope, gather, attention and slice.
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
- Vulkan runs whole sessions (`FORGE_DEVICE=vulkan` or `session::set_device`). The vk.* lowerings have priority over the CPU ones and are each oracle-checked; any op without a GPU kernel falls back to the CPU. Host↔device copies are made per op, not kept resident on the device.
- The CUDA/HIP kernels are compiled but have not been run: there is no GPU hardware, and the CUDA/HIP host runtime is not written.
- Level Zero and Metal: only probes exist.
- Non-power-of-two SVE lengths (e.g. 384) have no table and fall back to NEON.
- CI: the workflow file is at `ci/github-actions-ci.yml`. The push token lacks `workflow` scope, so it has to be moved to `.github/workflows/ci.yml` by hand.

## Build
    cmake -S . -B build -G Ninja && ninja -C build && (cd build && ctest)
The Vulkan backend needs `glslangValidator` and `vulkan/vulkan.h` at build time. At run time any loader and ICD works; lavapipe is enough for testing. The GPU compile checks need clang ≥18.

### Cross builds under qemu
Toolchain files are in `cmake/toolchains/` (`aarch64-sve-qemu.cmake`, `riscv64-rvv-qemu.cmake`). Configure with:

    -DCMAKE_TOOLCHAIN_FILE=... -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXE_LINKER_FLAGS=-static -DCMAKE_POSITION_INDEPENDENT_CODE=ON

To compile with zig instead (`-DFORGE_CROSS_CXX=<wrapper running "zig c++ -target aarch64-linux-musl">`), note that zig rejects `-march`. The CMake probes select the zig-compatible spellings:
- SVE: `-mcpu=generic+sve -msve-vector-bits=256`
- RVV: `-mcpu=baseline_rv64+v+zvl<B>b -mrvv-vector-bits=B`

The emulators are `qemu-aarch64 -cpu max,sve256=on` and `qemu-riscv64 -cpu rv64,v=true,vlen=256`.

## Test flake (test_vulkan under parallel load)

Concurrent runs shared fixed temp paths in /tmp (`forge_test_vk_e2e.fpk` and others), which caused SIGBUS and bad-magic errors. Temp files are now named per pid in test_vulkan, test_mind and test_pack. After the fix, 120 concurrent runs and `ctest -j8 --repeat until-fail:4` under full CPU load all passed.
