# What has not been tested

The single list of what in this tree has not been run, or has been run only in a weaker way than on real hardware
(emulator, wine, mock driver, software device, syntax check). It is written to be worked through on other machines:
each row says what exists, how far verification got, and the command that would close it.

State described: integration branch at commit c452a61, 2026-11-13. Everything here is taken from the dated entries of
[STATUS.md](STATUS.md), the notes of the guides, the commit messages of the 2026-11-12/13 work
(`git log b135941..c452a61`) and the final-state record in the 2026-11-13 STATUS entry. Where none of those says
whether something ran, the row says **not recorded**. Nothing was re-run to write this page.

Words used:

| Word | Meaning |
|---|---|
| native | built and run on the machine named |
| qemu / node / wine | run under `qemu-user`, node 22 (wasm) or wine 9 on the x86_64 Linux machine; no hardware or OS of that kind |
| mock | run against the mock driver libraries in `tests/` (`mock_cuda.c`, `mock_opencl.c`, `mock_ze.c`), which stand in for the vendor runtimes |
| lavapipe | run on Mesa lavapipe (llvmpipe), a CPU implementation of Vulkan |
| syntax | compiled with `-fsyntax-only` or against stub headers; nothing linked or run |
| cross | compiled and linked by a cross compiler; never run |
| never | no compiler of that platform has seen it, or nothing has executed it |

## 1. The one configuration that was verified

| What | Result |
|---|---|
| Machine | Linux x86_64, 4-core VM (kernel 6.18), no GPU |
| Compiler | clang 23.1.6, `dev` preset (RelWithDebInfo) |
| Command | `ctest --test-dir build/dev -LE "slow|emu"` |
| Result at c452a61 (2026-11-13) | 247 tests: 237 passed, 10 skipped, 0 failed |
| The 10 skips | `test_text_conformance` (Unicode conformance files not downloaded); `test_gpu_cuda`, `test_gpu_hip`, `test_gpu_features_cuda`, `test_gpu_features_hip`, `test_gpu_metal`, `test_gpu_opencl`, `test_gpu_features_opencl`, `test_gpu_ze`, `test_gpu_features_ze` (no device) |
| Not run on the final tree | labels `slow` and `emu`; any sanitizer build; any libFuzzer build; any other compiler, OS or CPU |

To reproduce:

```sh
cmake --preset dev -DCMAKE_CXX_COMPILER=clang++-23 -DCMAKE_C_COMPILER=clang-23
cmake --build build/dev -j4
ctest --test-dir build/dev -j4 -LE "slow|emu"
ctest --test-dir build/dev -L slow        # not run on the final tree
```

Python packages the 2026-11-12 run had: numpy, safetensors, onnx, cryptography, gguf, tokenizers, sentencepiece
(STATUS); the package set of the final run is not recorded separately. The CUDA / HIP mock tests
(`test_gpu_*_cuda_mock`, `golden_*_cuda_mock`, ...) only run when configure found a clang with the NVPTX / AMDGPU
backends (the kernel images are built with it); with none they skip.

## 2. Platforms and compilers

No CI run has happened for any of the 2026-11-12/13 work. The last CI run was PR #53 (2026-11-04, main at ccc0a7d);
its numbers are in the STATUS tables and describe the tree before this work. "Since" below means the roughly 191,000
lines added after it.

| Platform / compiler | State of the final tree | Command to verify |
|---|---|---|
| Linux x86_64, clang 23 | native, see section 1 | above |
| Linux x86_64, GCC 14 | **never** compiled the final tree. g++ 13 compiled parts by hand: `-fsyntax-only` on the model-container headers, the `sse2` / `sse4.2` companion tables and three `vec_types` checks, script AOT output, cross builds of the simd tests (STATUS 2026-11-12) | `cmake --preset dev -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_C_COMPILER=gcc-14 && cmake --build --preset dev -j4 && ctest --test-dir build/dev -LE "slow|emu" -j2 --timeout 1200` |
| Linux, clang 19 to 22 | configure accepts them with a note (zig cc, the NDK, FreeBSD base clang); none has built the whole tree. zig 0.17 (clang 22) built the simd tests and the OS cross-link checks only | — |
| Windows, MSVC `cl.exe` | **never**. clang-cl 23 against the MSVC STL and Windows SDK 10.0.22621 compiled every os header and linked every `test_os*`, run under wine 9 (os-parity.md). Whether anything outside the OS layer was compiled for the MSVC ABI since PR #53 is not recorded | `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=cl -DCMAKE_C_COMPILER=cl -DFORGE_BUILD_EXAMPLES=OFF && cmake --build build -j4 && ctest --test-dir build -LE "slow|emu" -j2 --timeout 1200`, then `set FORGE_TEST_ONLY=emit&& build\test_script.exe` |
| Windows, MinGW on real Windows | **never** since PR #53. MinGW-w64 g++ 13 cross builds of the `test_os*` suites ran under wine 9; whether the rest of the new code was built with MinGW is not recorded | MSYS2 UCRT64: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DFORGE_BUILD_EXAMPLES=OFF && cmake --build build -j4 && ctest --test-dir build -LE "slow|emu" -j2 --timeout 1200` |
| Windows under wine | OS-layer tests of waves 3 and 4 only (`tests/wine_run.sh`); full-suite wine passes are from 2026-10-29/31 and predate this work | `tests/wine_run.sh` (see [building.md](building.md)) |
| macOS, Apple clang | **never** compiled the final tree. zig 0.17 cross-compiled and linked the OS tests (`aarch64-macos`, `x86_64-macos`); nothing ran | `cmake --preset dev && cmake --build --preset dev -j4 && ctest --test-dir build/dev -LE "slow|emu" -j2 --timeout 1200` |
| iOS | **never**: the iOS SDK has not compiled the wave 3 / 4 OS branches or anything else since PR #53; nothing has ever run on a device; the Swift package and XCTest have never run | the `ios` job of `.github/workflows/ci.yml` (Xcode device build of `forge_c`, `forge_mobile`, `forge_text`; five tests in the simulator through `.github/ios_sim_check.sh`) |
| Android, NDK | NDK r27c (clang 18, API 26) compiled every os header and linked every `test_os*` for arm64-v8a; for x86_64 everything up to wave 3 and most of wave 4 (os-parity.md). **Never run** for the new code. The rest of the tree (inference, serve, the JNI library against ABI 4) with the NDK: not recorded. Never on a physical device. Note: r27c's compiler is clang 18, which the CMake configure refuses since 2026-11-12, so a CMake build needs an NDK with clang 19 or newer; how the r27c check was driven, and which NDK the CI runner provides, is not recorded | `sh tests/mobile_build.sh android arm64-v8a`; the `android` job (x86_64 emulator, `.github/android_check.sh`); `tests/android_emulator.sh apk <build-dir>` |
| FreeBSD 14 | **never** compiled natively since PR #53. zig 0.17 (`x86_64-freebsd`) compiled and linked the OS tests; nothing ran | in a FreeBSD 14.3 VM: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DFORGE_BUILD_EXAMPLES=OFF && cmake --build build -j4 && ctest --test-dir build -LE "slow|emu|gpu" -j2 --timeout 1200 --output-on-failure` |

### The CI workflow itself

`.github/workflows/ci.yml` (and its two copies under `ci/`) was rewritten during this work and has never executed
on GitHub in its present form.

| Job | Ever run on GitHub | Note |
|---|---|---|
| `linux-gcc`, `linux-clang` | the older form, PR #53 (GCC 14, clang 18) | now clang 23 from apt.llvm.org; whether that repository serves `llvm-toolchain-noble-23` is unchecked |
| `linux-vulkan`, `macos`, `ios`, `windows-msvc`, `windows-mingw`, `freebsd`, `android`, `shells` | PR #53 form; the 2026-11-05 additions (`linux-vulkan`, the iOS simulator tests, five more Android suites) were recorded as "first CI run pending" | package lists changed since |
| `linux-asan` (whole suite), `linux-tsan` (threaded suites) | only as `test_script`-only jobs (PR #53) | never in the present form |
| `linux-fuzz`, `emu` (aarch64 SVE, riscv64 RVV, loongarch64, ppc64le, s390x, mips64el), `wasm-simd`, `cross-zig`, `perf`, `repro` | **never** | whether Ubuntu 24.04 has `g++` cross packages for loongarch64 and mips64el is unchecked; those two rows are `continue-on-error` |

Checked locally instead: the file parses as YAML, every ctest name, label and build target it names exists in a
clang 23 build tree, and the three copies are identical (`ci_workflow_copies`).

## 3. CPU architectures

x86_64 is the only architecture anything has run on natively during this work. The 2026-10-31 qemu runs of the wider
suite on aarch64 and riscv64 (50/54, STATUS) predate it.

| Architecture / tables | Compiled with | Ran | What ran | Not run |
|---|---|---|---|---|
| x86_64 `scalar`, `sse2`, `sse4.2`, `avx2`, `avx512`; int8 `avxvnni`, `avx512vnni` | clang 23 (g++ 13 for parts) | native (Xeon with AVX-512 and AVX-VNNI); `sse2` / `sse4.2` also under `qemu-x86_64 -cpu core2duo` / `Nehalem` | the whole suite; each lower table forced with `FORGE_SIMD_TARGET` for `test_simd`, `test_simd_raw`, `test_graph`; `test_simd_tables` 3/3, `test_simd_tiera` 14/14, `vec_types` 0 failed | an AMD CPU, an AVX2-only or SSE-only machine natively: not recorded. AMX ran natively and under Intel SDE before this work |
| aarch64 `neon`, `sve`; int8 `neon`, `dotprod`, `sve` | g++ 13 cross; clang 23 cross for the int8 kernels | qemu (`-cpu cortex-a57`, `max,sve=off`, SVE at 128 to 2048 bits) | `test_simd`, `test_simd_tables` 3/3, `test_simd_tiera` 14/14, `vec_types`, `qi8_aarch64_*`, `test_graph` (also with `FORGE_CPU_INT8=1`) under `max,sve256=on` | **any Arm hardware**; any timing; Apple, FreeBSD, Windows detection branches (zig compile only); Apple clang, NDK, MSVC arm64; SME only under qemu |
| riscv64 `rvv` (VLEN 128 to 1024) | zig 0.17 (clang 22) | qemu (`-cpu rv64,v=true,vlen=N`) | `test_simd_tables`, `test_simd_tiera`, `vec_types` at all four lengths | hardware; timing; the rest of the suite on the final tree |
| loongarch64 `lsx`, `lasx` | zig 0.17 | qemu (`-cpu la464`) | `test_simd` 7/7, `test_simd_tables` 3/3, `test_simd_tiera` 14/14, `vec_types` | everything else; hardware. Known: 46,976 differing checks in the unused portable-array path, a compiler defect, reported without failing |
| ppc64le `vsx` | zig 0.17; g++ 13 by hand | qemu (`-cpu power9`, `power8`) | the same four | everything else; hardware; big-endian POWER |
| s390x `vxe` | zig 0.17; g++ 13 by hand | qemu (`-cpu max`) | the same four; the int8 reference hash is compared only with its own scalar table (big-endian) | everything else; hardware; a z14-only CPU model |
| mips64el `msa` | zig 0.17 | qemu (`-cpu Loongson-3A4000`) | the same four | everything else; hardware |
| wasm32 `simd128` | zig 0.17 `wasm32-wasi` | node 22 | `tests/wasm_simd.sh`: `test_simd` 7/7, tables 3/3, Tier A 14/14 one test per process (node segfaults on the whole binary), `vec_types`; also with `WASM_SIMD=0` | any other engine; FORGE as a whole is not ported to WebAssembly |
| 32-bit x86 `sse2`, `sse4.2` | zig 0.17, by hand | qemu (`qemu-i386`) | `test_simd` 7/7 | no CMake configure; companion tables and `vec_types` not built |
| 32-bit Arm `neon` | — | — | being fixed, see [STATUS.md](STATUS.md) | — |
| Raw intrinsic wrappers (Tier R, eight ISA families, 103,666 wrappers) | clang 23 `--target=<triple> -fsyntax-only` (`raw_intrinsics_cross`) | x86 only (`test_simd_raw`) | — | **no non-x86 wrapper has executed**; MSVC; GCC beyond the recorded per-entry rejections |

Defaults that wait on hardware: every int8 table outside the x86 AVX ones (`sse2`, `sse4.2`, AArch64, `rvv`, `lsx`,
`lasx`, `vsx`, `vxe`, `simd128`, `msa`) is off by default because none has been timed.

Commands:

```sh
# aarch64 hardware: correctness, then time the int8 kernels and decide their default
ctest --test-dir build/dev -R "test_simd|test_graph|test_tile"
FORGE_CPU_INT8=1 ./build/dev/forge bench --kernels 1024        # compare with FORGE_CPU_INT8=0

# emulated rows (qemu-user installed; GNU cross compilers or zig wrappers, see simd.md "Other architectures")
cmake -S . -B build/loongarch64 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/loongarch64-lasx-qemu.cmake -DFORGE_BUILD_EXAMPLES=OFF
cmake --build build/loongarch64 --target forge_simd_tests && ctest --test-dir build/loongarch64 -L emu
sh tests/wasm_simd.sh && WASM_SIMD=0 sh tests/wasm_simd.sh     # wasm32-wasi under node
ctest --test-dir build/dev -R raw_intrinsics_cross             # syntax of every Tier R wrapper
```

The toolchain files for the other emulated rows are `aarch64-sve-qemu.cmake`, `riscv64-rvv-qemu.cmake`,
`ppc64le-vsx-qemu.cmake`, `s390x-vxe-qemu.cmake` and `mips64el-msa-qemu.cmake` in `cmake/toolchains/`.

## 4. GPU backends

No GPU was available during the 2026-11-12/13 work. Nothing written in it has run on a physical GPU.

| Backend | Real hardware | Since then, verified by | Never executed |
|---|---|---|---|
| CUDA | RTX 3090: WSL2 2026-10-31 and 2026-11-05, Windows 11 2026-11-04. **Predates this work** | mock (`test_gpu_*_cuda_mock`, `golden_*_cuda_mock`); kernels compiled to PTX (`sm_70`, `sm_80`) by `gpu_compile_cuda` | on a GPU: device MoE, determinism mode, the device kernels for `gelu` / `layer_norm` / `group_norm` / `conv1d` / `conv2d` / pooling / `attention_bias`, buffer generations and re-recording after eviction (the `cuFuncGetParamInfo` path), `gpu::prims` device versions, `pipe:auto`, the scaled-RoPE table, graphics-interop import, multi-GPU and peer copies |
| HIP | **never** | mock (`mock_cuda.c` built as HIP); AMDGPU code objects compile (`gpu_compile_hip`) | everything |
| OpenCL | **never** | mock: the generated `kernels.cl` compiled as C and run | everything; no real OpenCL C compiler has compiled the portable kernels; whether a driver accepts its own cached program binaries |
| Level Zero | **never** | mock, handed the OpenCL C source (`FORGE_ZE_TEST_IMAGE=opencl-c`): the build had no `llvm-spirv`, so the SPIR-V image path did not run even on the mock | everything, including building the SPIR-V image |
| Vulkan | **never** on a GPU | lavapipe 25.2.8 (LLVM 20): `test_vulkan`, `test_vulkan_align256`, 13 `golden_*_vulkan`; all 47 shaders pass `spirv-val`. SwiftShader not since 2026-11-05 | real drivers' SPIR-V compilers, descriptor limits, buffers beyond the tested sizes. `golden_tiny_qwen3_5` on Vulkan was skipped (no torch) |
| Metal | `test_gpu_metal` on the GitHub macOS runner, PR #53. **Predates this work** | syntax only: MSL as C++ (`gpu_compile_metal_syntax`, 45 kernels; address-space keywords vanish) and `metal.mm` against stub declarations (`gpu_compile_metal_host_syntax`) | the Metal ports of the portable kernels (quantised matmul / gather / FFN, attention, state ops, MoE, broadcast, the encoder / vision kernels), the RoPE table and their host code: never compiled by a Metal toolchain, never executed |

Mock limits to keep in mind: the mocks run serially (the determinism tests show kernel selection, not a device's
scheduling); `mock_cuda.c` has hand-written C versions of the tuned CUDA kernels, so for those it checks host code and
algorithm, not the PTX; the mocks report equal device priors, so only the 50 / 50 `pipe:auto` split ran on a driver.

The late fix in c452a61 belongs here: the CUDA mock sized captured kernel arguments from a hand table that did not
know the generated conv / pool / norm / `attention_bias` kernels, so a graph replay truncated pointers
(`test_gpu_cuda_mock` and five `*_cuda_mock` goldens crashed). It was found only at the very end. Those mock tests only
run when configure found a clang with the NVPTX backend; with none they skip.

Commands. The STATUS entries of the GPU work name tests, not a command list for the RTX 3090; these are the ctest
names and tools they and [gpu.md](gpu.md) / [testing.md](testing.md) give:

```sh
# CUDA on the RTX 3090 (WSL2 or Linux; clang with NVPTX at configure time)
ctest --test-dir build/dev -L gpu                                   # real-device tests run instead of skipping
ctest --test-dir build/dev -R "^test_gpu_(features_)?cuda$" --output-on-failure
./build/dev/test_cuda_hip cuda                                      # the same binary directly: prints each check
FORGE_DEVICE=cuda ./build/dev/forge run model.fpk --prompt "Hello" -n 32      # compare ids with FORGE_GRAPHS=0 and with the CPU
./build/dev/test_cuda_hip bench q4_k 512 25600 5120                 # one quantised matmul, timed
tools/bench_llm.sh --gguf model.gguf --device cuda -p 512 -n 128    # against llama-bench

# HIP, OpenCL, Level Zero on a device of that kind
ctest --test-dir build/dev -R "test_gpu_(hip|opencl|ze)$|test_gpu_features_(hip|opencl|ze)$" --output-on-failure
FORGE_DEVICE=opencl ./build/dev/forge run build/dev/golden/tiny_f32.fpk --prompt "Hello" -n 8

# Vulkan on a real GPU (needs glslangValidator and the Vulkan headers at build time); do not pin the lavapipe ICD
ctest --test-dir build/dev -R vulkan --output-on-failure

# Metal on a Mac: first real compile and run of the ports
ctest --test-dir build/dev -R "gpu_compile_metal|test_gpu_metal" --output-on-failure
```

The determinism, memory, stale-plan, kernel-cache and device-selection checks (`test_cuda_hip determinism|memory|stale|cache|select <api>`)
are registered in ctest only against the mocks; whether they can run against a real driver is not recorded.

## 5. Features

Every model in every test is a tiny random-weight model built by the test. **No published checkpoint or tokenizer of
any family has been imported or run** during this work. (Before it, Qwen3.5 9B and a Qwen3 32B GGUF ran on the RTX
3090: STATUS 2026-11-05.)

### Models, importers, tokenizers

| Item | Exists | Verified | Not verified | Command |
|---|---|---|---|---|
| Model families: llama, qwen2, qwen3 (+MoE), qwen3.5 (+MoE), mixtral, deepseek_v2 | templates, HF and GGUF import | tiny goldens against numpy / transformers references | a published checkpoint on the final tree | `forge import <hf_dir or .gguf> -o m.fpk && forge run m.fpk --prompt "..." -n 32`, compare ids with the source runtime |
| GPT-2, Gemma 1 / 2 / 3, Phi-3 / Phi-4-mini | templates, HF and GGUF import | `golden_tiny_{gpt2,gemma,phi3}` (f32, q8_0) against float64 numpy; the same on the CUDA and OpenCL mocks and lavapipe | any published checkpoint; GGUF tensor names and conventions (section 8) | as above |
| BERT / RoBERTa, ViT / CLIP / SigLIP, Whisper encoder (embedding sessions) | templates, `forge embed`, `/v1/embeddings` | `golden_tiny_{bert,vit,whisper}`, `serve_embeddings` | any published checkpoint; no image or audio front end (inputs are preprocessed f32) | `forge embed m.fpk --text "..."` against sentence-transformers / transformers |
| Qwen2 bias template, RoPE scaling (linear, dynamic, YaRN, llama3) | importer + `rope` attrs | `golden_tiny_qwen2`, `golden_rope_scaling` against NumPy transcriptions | transformers or llama.cpp themselves (section 8); LongRoPE is refused | import a model that declares `rope_scaling` and compare logits |
| GGUF tensor types | every type with a codec; `--codec keep` | `golden_gguf_types` against gguf-py, bit for bit | published GGUF files of the less common types. The seven grid-codebook types are refused by decision ([modelfile.md](modelfile.md#gguf-tensor-types)) | `ctest --test-dir build/dev -R golden_gguf_types` |
| AWQ / GPTQ checkpoints | AutoAWQ `gemm` and AutoGPTQ `QuantLinear` layouts | `golden_hf_quant`, `test_modelfile`: **only against a packer written from the same description** | any checkpoint written by AutoAWQ, AutoGPTQ or GPTQModel. A misread layout imports without error and computes with wrong weights | import a real AWQ / GPTQ checkpoint; compare a few logits with the exporter's runtime |
| Tokenizers (BPE, SentencePiece BPE, Unigram, WordPiece, WordLevel; the HF pipeline) | `tokenizer.json`, `tokenizer.model`, `vocab.txt`, GGUF | `test_tokenizer` against `tokenizers` 0.23 and `sentencepiece` 0.2 on small trained fixtures; text → ids → output in four family goldens | the tokenizer of any full-size published model; llama.cpp's per-token `lstrip` / `rstrip` attributes are not carried over | `forge run m.fpk --prompt "..." --ids` against `AutoTokenizer` ids |
| Chat templates (Jinja subset) | `forge run --chat`, forge-serve | `test_chat_template` against jinja2 3.1 for 13 template families; `chat_template_http` | the vendors' actual template files (section 8) | render a real `tokenizer_config.json` template and diff with transformers |
| ONNX importer | CNN, shape, reduction and movement operators; external data | `onnx_import`: 78 outputs of 10 graphs against the ONNX reference evaluator and onnxruntime 1.29 | any published ONNX model; convolution family only at batch 1 | `forge run model.onnx --input name=file.f32` against onnxruntime |
| Unicode conformance | `forge_text` | passed 100 % on 8 suites before this work (STATUS 2026-10-29) | `test_text_conformance` skipped on the final tree (files not downloaded) | configure with the UCD files fetched (`cmake/fetch_ucd.cmake`), `ctest -R test_text_conformance` |

### Model container

| Item | Verified (clang 23, Linux; g++ 13 syntax check of the headers) | Not verified |
|---|---|---|
| Chunked encryption, on-demand decryption | `test_modelfile` `fpk_chunked_encryption`; `fuzz_fpk` with sealed seeds | reading a file written by a build from before the change (none was kept); locked plaintext depends on `RLIMIT_MEMLOCK` |
| Tensor alignment 64 / 4096 / 65536 | `fpk_tensor_alignment`, `golden_tiny_llama` | whether page alignment helps loading or device upload: never measured |
| `state`, `adapters`, `calib` sections; `forge requant`, `forge adapter` | three `test_modelfile` cases, `golden_tiny_llama` | the calibration rule is a heuristic with no quality measurement |
| Binary blueprint | `blueprint_binary_form_round_trips` (16 templates, 7 corpus blueprints, every truncation and bit flip); `fuzz_blueprint` | — |
| Shards and blob store | `fpk_shards_and_blob_store`, `golden_tiny_llama` | models of a few hundred kilobytes only: `--shard-size 2G` has never split a file that large; local directories only |
| Signing | `modelfile_signing` against pyca/cryptography | — |

### Inference and ahead-of-time compilation

| Item | Verified | Not verified | Command |
|---|---|---|---|
| Model AOT, plan mode | `test_aot` (bitwise against run-time plans, hostile blobs, signed files), 26 tiny models; `test_aot` under ASan + UBSan on 2026-11-12, no report | models larger than the tiny ones; `fuzz_aot` under a sanitizer: not recorded | `forge aot m.fpk && forge run m.fpk ...` |
| Model AOT, native mode | `test_aot_native` (label `slow`; run by hand 2026-11-12, **not on the final tree**), 25 tiny models by hand | cross compilation (`--target`); a compiler other than the one that built FORGE; Mach-O and PE/COFF; AArch64, RISC-V, macOS, Windows; linking against `--emit static`; an installed tree; embedding near the size limit | `ctest --test-dir build/dev -R test_aot_native` |
| Script AOT (`forge script --aot`) | `test_script_aot`; `test_script_aot_native_sweep` with clang++ 23 and g++ 13 (label `slow`; `ctest -R script_aot` 3/3 on 2026-11-12, not on the final tree); `fuzz_script_aot` | Windows (`cl` command line, PE trailer), macOS (Mach-O trailer + `codesign`), FreeBSD, Android, iOS; cross-compiling; `cmake --install` of the runtime; baked `--mem-limit`, baked FSS policy and `integrity.scripts` have no automated test | `ctest --test-dir build/dev -R script_aot` |
| Diffusion and streaming sessions | `test_sessions`, `golden_tiny_sessions` (numpy references) | a UNet / DiT, a VAE, an audio front end; any device (CPU only) | `forge denoise`, `forge stream` on a real model |
| Self-speculative decoding (early exit) | exact against plain decoding (`test_inference`, `golden_tiny_llama`) | a trained model (acceptance on random weights says nothing); speed never measured | `forge run m.fpk --self-spec --exit-layer K` and time it |
| CPU fusions (conv + batch_norm folding, row chains, shared int8 activations) | bit-identical or within the stated bound on random graphs, x86_64 only | any other architecture; effect on a real model never timed | — |
| Engine on the work-stealing scheduler (`FORGE_TASK_SCHEDULER=sched`) | the engine's tests under both executors (`sched_*`, `engine_sched`) | more cores, NUMA, hybrid machines, a quiet machine (section 7) | `FORGE_TASK_SCHEDULER=sched forge bench m.fpk` against the default |

### Serving, C ABI, bindings, RPC

| Item | Verified (Linux x86_64, clang 23, tiny golden models) | Not verified | Command |
|---|---|---|---|
| forge-serve: stops, request fields, keep-alive, limits, shutdown | `serve_api`, `serve_http`, `fuzz_serve_http` | a real chat checkpoint; the Windows shutdown path; IPv6 listen (host had none); under systemd, the Windows SCM or launchd | `forge-serve m.fpk --port 8080`, then an OpenAI client |
| Tool calling (four formats) | `test_chat_tools`, `serve_api` | **no tool-calling checkpoint was served** | chat request with `tools` against a real model |
| API keys, rate limits, quotas, TLS / mTLS | `serve_auth` (certificates from the `openssl` command; OpenSSL 3.0) | other TLS stacks; other OSes | `ctest --test-dir build/dev -R serve_auth` |
| Per-request LoRA | `test_chat_tools`, `serve_api` | a real adapter on a real model | `forge-serve m.fpk --lora name=adapter.safetensors` |
| `/v1/embeddings` | `serve_embeddings` | published embedding models | — |
| C ABI 4 | `capi_smoke`, `doc_example_capi_generate` | any other compiler or OS; `"auto"` and GPU devices through the C ABI (only `"cpu"` and a fallback ran) | `ctest --test-dir build/dev -R "capi_smoke|doc_example_capi"` |
| Python binding | `python_bindings` (CPython 3.13, Linux) | macOS, Windows; a built wheel | `ctest --test-dir build/dev -R python_bindings` |
| Kotlin binding | JNI C bridge only: `jni_smoke` on a host JVM (OpenJDK 21, `-Xcheck:jni`), `jni_decl_matches` | **`ForgeNative.kt` and `ForgeModel.kt` (ABI 4) have never been compiled**; Gradle, the AAR, `ForgeModelTest`, any Android device or emulator | `cd bindings/android && gradle :example:assembleDebug`; `gradle :forge:connectedAndroidTest` with `FORGE_TEST_FPK` set; `tests/android_emulator.sh apk <build-dir>` |
| Swift binding | nothing | **`Forge.swift` (ABI 4) has never been compiled**; the xcframework script, `swift test`, any Apple device | on a Mac: `tools/build_xcframework.sh`, then `swift test` in `bindings/apple` with `FORGE_TEST_FPK` set |
| RPC protocol 2 (weight residency) | `test_rpc` over loopback, plain TCP, CPU server; `fuzz_rpc` 150,000 inputs in a build without sanitizers | two machines; TLS-specific paths of the new frames; a GPU behind the server; concurrent connections under load | `forge rpc-serve --device cuda:0` on one machine, `FORGE_DEVICE=rpc:host:port` on another ([rpc.md](rpc.md)) |
| RPC certificates: SAN rules, CRL and fingerprint revocation, fine-grained permissions | `test_rpc`, `test_rpc_tls` (OpenSSL 3.0, Linux) | other TLS stacks and platforms; `test_rpc_tls` on Windows has never run | `ctest --test-dir build/dev -R test_rpc` |
| Load balancer and proxy | `test_loadbalancer`, `fuzz_proxy_http` | — | — |

### OS layer

Per-call detail is in [os-parity.md](os-parity.md); this is the summary by OS for waves 1 to 4.

| OS | Compiled | Run | Did not run |
|---|---|---|---|
| Linux x86_64 | clang 23 | every `test_os*`; `test_os_dbus` also against a real `dbus-daemon`; `test_os_clipboard` against Xvfb, cross-checked with `xsel` | cgroup v2 limit groups on a real v2 hierarchy (the machine had v1); hybrid P/E placement; multi-node NUMA; `perf_event` hardware zones (`hw_zone` ran its fallback); under a real systemd; PAC scripts are not evaluated (no JavaScript engine) |
| Windows | MinGW-w64 g++ 13; clang-cl 23 with the MSVC STL; **not `cl.exe`** | under wine 9 only | on real Windows: all of waves 3 and 4. Under wine these could not run at all: reparse points (`junction`, `read_link`, file `symlink`), stream enumeration, `preallocate` / `punch_hole`, ConPTY output, background I/O priority, integrity levels and job enforcement in the sandbox, AppContainers, nested job objects, job limits, ReFS `clone_file`, `PowerCreateRequest`; `network_monitor`, `devices::monitor`, toast notifications and `run_elevated` ran their fallback paths |
| macOS | zig 0.17 cross (`aarch64-macos`, `x86_64-macos`), tests linked | **never** | everything, including the kqueue reactor backend, OS-evaluated PAC, CFPreferences appearance, `ntp_adjtime` |
| FreeBSD 14 | zig 0.17 cross (`x86_64-freebsd`), tests linked | **never** | everything, including the kqueue reactor backend, rctl limit groups, the ConsoleKit inhibitor, `elf_aux_info` detection |
| Android | NDK r27c: arm64-v8a complete; x86_64 without the wave 4 tests and `timesync.hpp` | **never** | everything |
| iOS | **not compiled** (no SDK); waves 1 to 3 syntax-checked against macOS headers with `TARGET_OS_IPHONE` forced | **never** | everything |

Command on each OS: build, then `ctest --test-dir <build> -R "test_os|test_process_pool|test_mobile|test_text|test_locale"`.
The zig cross-link checks are `ctest --test-dir build/dev -L cross` (label `slow`; not run on the final tree).

### Support modules, codecs, tooling

| Item | Verified | Not verified |
|---|---|---|
| Work-stealing scheduler | `test_sched` (also under TSan) | pinning and NUMA binding beyond a 4-CPU single-node VM; placement on hybrid hardware (injected topologies only) |
| Linear algebra, Tier A algorithms, reflection | `test_linalg`, `test_simd_tiera`, `test_reflect` | the C++26 static-reflection branch (`FORGE_REFLECT_STATIC`) has never been compiled |
| Profiler | `test_prof` (structure of `perfetto_json()`) | the output has not been loaded into Perfetto; Tracy forwarding has never been compiled; `perf_event` was unavailable; overhead not measured |
| Image codecs (PNG, BMP, PNM, inflate) | `test_vision`, `fuzz_png`, `fuzz_bmp`, `fuzz_pnm`, `fuzz_inflate` | throughput never measured |
| Video | bitstream parsing in `test_vision` | NVDEC, MediaCodec, VideoToolbox, VA-API decoders have never run |
| Install rules, CMake package | `install_consumer` (label `install`) on Linux | other platforms; the script AOT runtime through `cmake --install` |
| Release tooling | `tools/release.py version` | every other release command; `tools/build_xcframework.sh`; the `macos-osxcross` preset; there are no release tags |
| SBOM, docs checks | `sbom_spdx`, `coverage_table`, `docs_links`, `docs_up_to_date` | — |

## 6. Sanitizers and fuzzing

The hardening pass that was to run ASan / UBSan, TSan and libFuzzer over the newest code was cut short by machine
restarts. Its eleven fixes are merged; its planned full passes did not complete and it left no run table. **Treat the
second-round code as not sanitized**: forge-serve chat features and C ABI 4, the simd companion tables and
`vec_types.hpp`, the CPU fusions, the Vulkan / Metal ports and RPC protocol 2, the model-container sections and
session kinds.

| Run | When | Code covered | Result |
|---|---|---|---|
| CI `linux-asan`, `linux-tsan` | PR #53, 2026-11-04 | `test_script` only | pass |
| ASan + UBSan and TSan of `test_script`, `test_rpc` | 2026-10-30 | as then | clean |
| ASan + UBSan: D-Bus, proxy, clipboard and misc OS tests; TSan: reactor; `fuzz_dbus` 300,000 inputs under ASan + UBSan | 2026-11-12, OS wave 4 | wave 4 | clean |
| TSan: `test_sched` | 2026-11-12 | scheduler as first merged | clean |
| Whole-suite ASan + UBSan and TSan over the 25 threaded suites | 2026-11-12, "before the final merges" (STATUS, commit 13af50e) | the tree before the tokenizer, family, serve, GPU follow-up, simd follow-up and container merges | the list of suites and counts was **not recorded** |
| ASan + UBSan with leak detection: `test_tokenizer`, `test_modelfile`, `test_chat_template`, `test_ops_ext`, `test_aot`, the gpt2 / gemma / phi3 / bert / llama goldens, `serve_embeddings` | 2026-11-12, after the first round of merges | first-round code | no report |
| Hardening pass: whole-suite ASan + UBSan (started) | 2026-11-12/13 | the commit messages name what it reached: `test_script*` (hang, heap-cap use-after-free), `signing_cli`, `security_cli`, `fuzz_gguf`, the 23 `*_cuda_mock` / `*_hip_mock` ctests | eleven defects fixed (list in the 2026-11-13 STATUS entry); **the pass did not complete**; which suites finished clean is not recorded |
| Hardening pass: TSan; libFuzzer campaigns | planned | second-round code | **did not run** |
| Any sanitizer on the final tree (c452a61) | — | — | **not run** |

Not re-run under a sanitizer after the first merges (STATUS 2026-11-12): `test_sched`, `test_simd_tiera`,
`test_linalg`, `test_prof`, `test_vision`, `test_reflect`, `test_script_aot`, the `test_os*` suites.

Fuzzing:

| What | State |
|---|---|
| Built-in mutation driver, every `fuzz_*` target as a ctest (label `fuzz`, `FORGE_FUZZ_ITERS` inputs) | part of the 237 passing tests on the final tree, without sanitizers |
| `fuzz_rpc` | 150,000 mutated inputs with the protocol 2 seeds, no finding, **no sanitizers** |
| `fuzz_dbus` | 300,000 inputs under ASan + UBSan |
| `fuzz_gguf` under ASan + UBSan on its corpus | found the empty-tensor `memcpy`; fixed |
| `fuzz_aot` | a structure-aware seed generator was added (`fuzz/aot_seeds.py`); a campaign with it is not recorded |
| libFuzzer (`-DFORGE_LIBFUZZER=ON`) campaigns on any target | **not recorded** for this work; the `linux-fuzz` job has never run |

Commands (from [testing.md](testing.md)):

```sh
# ASan + UBSan, whole suite
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++-23 -DCMAKE_C_COMPILER=clang-23 -DFORGE_SANITIZE=ON -DFORGE_BUILD_EXAMPLES=OFF
cmake --build build-asan -j4
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ctest --test-dir build-asan -LE "slow|emu" -j2 --timeout 3600

# TSan, the threaded suites: set TSAN_TESTS to the list of that name in the linux-tsan job of .github/workflows/ci.yml
cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=clang++-23 -DCMAKE_C_COMPILER=clang-23 -DFORGE_TSAN=ON -DFORGE_BUILD_EXAMPLES=OFF
cmake --build build-tsan -j4 --target forge forge_script_example_plugin $TSAN_TESTS
TSAN_OPTIONS="halt_on_error=1 second_deadlock_stack=1" ctest --test-dir build-tsan -R "^($(echo $TSAN_TESTS | tr ' ' '|'))$" -j2 --timeout 3600 --no-tests=error

# libFuzzer with ASan + UBSan, one target
cmake -S . -B build-fuzz -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++-23 -DCMAKE_C_COMPILER=clang-23 -DFORGE_SANITIZE=ON -DFORGE_LIBFUZZER=ON -DFORGE_BUILD_EXAMPLES=OFF
cmake --build build-fuzz --target fuzz_rpc
mkdir -p /tmp/rpc-corpus && cp fuzz/corpus/rpc/* /tmp/rpc-corpus/ && ./build-fuzz/fuzz_rpc -max_total_time=300 -timeout=30 /tmp/rpc-corpus
```

On kernels with `vm.mmap_rnd_bits` above 28 the sanitizer runtimes fail at start: `sudo sysctl vm.mmap_rnd_bits=28`.
A sanitizer tree does not register `python_bindings` or `install_consumer`.

## 7. Performance

Everything timed during this work was timed on one loaded 4-core x86_64 VM shared with other builds. The guides
that hold the tables say to read them as "no difference larger than the noise". No number below was taken on a quiet
machine or on a real model.

**Model AOT** ([aot-models.md](aot-models.md#measured); load average 22 to 27; random-weight 12-layer models; the
same plan measured twice differs by up to 30 %):

| | small: start-up | small: decode | wide: start-up | wide: decode |
|---|---|---|---|---|
| `forge run model.fpk` (plans compiled at run time) | 6.6 ms | 1608 tok/s | 73 ms | 78 tok/s |
| the file with precompiled plans | 6.5 ms | 1240 tok/s | 70 ms | 57 tok/s |
| `forge run model.so` (native library) | 8.0 ms | 1471 tok/s | 56 ms | 82 tok/s |
| `./model` (native executable) | 5.8 ms | 1332 tok/s | 66 ms | 94 tok/s |

The first two rows run the same plan through the same executor: their difference is the noise.

**Script AOT** ([aot.md](aot.md#performance); load average 17 to 20; CPU seconds, best of 5; three small programs):

| program | VM | embed | native, clang++ | native, g++ | native vs VM |
|---|---|---|---|---|---|
| `bench_ints.cpp 300000` | 5.50 | 5.47 | 1.36 | 1.41 | 4.0x |
| `bench_calls.cpp 32` | 2.04 | 2.01 | 1.59 | 1.74 | 1.3x |
| `bench_objects.cpp 20000` | 1.42 | 1.37 | 1.18 | 1.26 | 1.2x |

**Engine executor** ([support.md](support.md#engine-executor-taskenginehpp); `forge bench`, 9 interleaved runs, runs
of one configuration differed by up to 40 %), tok/s:

| | `pool` (default) | `sched` |
|---|---|---|
| wide model, prefill | 1472 | 1458 |
| wide model, decode | 678 | 418 |
| deep model, prefill | 28260 | 13800 |
| deep model, decode | 5719 | 2197 |

`sched` is 1.6 to 2.6 times slower where loops are short; the cause was not profiled. `pool` stays the default.

Also measured on that machine, with the same caveat: the int16-query attention path (`FORGE_CPU_ATTN=q16`; "too noisy
to change a default", [inference.md](inference.md#notes-and-limits)).

Measured before this work, on real hardware: CUDA prefill and decode on the RTX 3090 against llama.cpp (2026-11-05)
and the x86 int8 kernels on a shared Xeon Gold 6242 (2026-11-06); both in [STATUS.md](STATUS.md).

**Never timed:**

- every int8, `seq` and Tier A table outside x86 AVX (hence off by default where a default exists); the SSE and
  wasm tables against `scalar`
- the AArch64 int8 kernels on Arm hardware: `FORGE_CPU_INT8=1 ./build/dev/forge bench --kernels 1024`
- every device kernel added in this work, on any backend: the encoder / vision kernels, device MoE, the Vulkan and
  Metal ports, `gpu::prims`, `pipe:auto` splits, the cost of determinism mode
- CPU fusions on a model; self-speculative decoding; RPC weight residency over a real network
- embeddings throughput; forge-serve under load; tokenizer speed
- chunked decryption and page alignment at load; sharded models of real size
- the profiler's overhead; the inflate decoder; the scheduler on more than 4 cores
- `forge script --aot` build time with `--aot-pch`
- the CI `perf` job has never run; `perf/baseline-x86_64.json` has not been compared with this tree (not recorded)

## 8. Written from memory, never diffed against the source

These were written from recollection or from a description of another project's format and have **not** been compared
with that project's files. Each can be wrong while every test passes, because the tests compare FORGE with a
reference built from the same recollection.

| What | Where | What the tests do show | To close |
|---|---|---|---|
| AutoAWQ and AutoGPTQ packing layouts | `import/hf_quant.hpp`, `verify/golden/hf_quant.py` | agreement with a NumPy packer written from the same description | import a checkpoint written by each exporter; compare logits with its runtime |
| The llama.cpp pre-tokenizer regex table and the `ignore_merges` list | `import/tokenizer_import.hpp` | `llama-bpe`, `qwen2` and `gpt-2` against `tokenizers`; `bert` on ASCII only | diff against the llama.cpp source; tokenise real GGUF vocabularies and compare ids with llama.cpp |
| Vendor chat-template texts | `tests/data/chat_template/make_goldens.py` | FORGE renders those texts as jinja2 does | diff each against the vendor's `tokenizer_config.json` |
| GGUF tensor names and conventions for gemma, phi3, bert and gpt2, and several Gemma specifics | `import/gguf_arch.hpp`, the templates in `import/templates/` | the tiny goldens write GGUF files with the same names the importer expects | import a published GGUF of each family |
| The RoPE-scaling reference | `verify/golden/tiny_rope.py` (a NumPy transcription of transformers' `modeling_rope_utils.py`) | FORGE equals the transcription | run the reference through transformers itself |
| The GGUF YaRN attention-factor formula | the GGUF importer | as above | compare with llama.cpp on a YaRN model |
| The per-family "leave-out" tensor lists (tensors an import skips as harmless) | `import/hf_arch.hpp`, `import/gguf_arch.hpp` | tiny models built by the tests | import published checkpoints without `--allow-unused` and read the report |

The last four rows and the `ignore_merges` list are recorded in the 2026-11-13 STATUS entry; the guides do not carry
a note for them.

## 9. Licence and provenance facts

Not test debt, listed so that it is in one place: the licence of the `riscv-non-isa/rvv-intrinsic-doc` prototype
lists behind the RISC-V files in `ops/intrinsics/` is recorded in [NOTICE](../NOTICE) as not established. The owner
decided on 2026-11-13 not to pursue it now; the NOTICE statement stands.
