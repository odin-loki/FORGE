# FORGE test status

Current as of 2026-11-13, integration branch at c452a61. What has been built and tested where.

- **The final tree was verified in one configuration only**: Linux x86_64 (4-core VM), clang 23.1.6, `dev` preset,
  `ctest -LE "slow|emu"`: 247 tests, 237 passed, 10 skipped, 0 failed. Labels `slow` and `emu` were not run on it.
- **No CI run has happened for the 2026-11-12/13 work** (about 191,000 lines since main at b135941). The CI results
  in the tables below are from PR #53 (2026-11-04, main at ccc0a7d) and describe the tree before that work. GCC 14,
  MSVC `cl.exe`, MinGW on real Windows, Apple clang (macOS, iOS), the Android NDK build in CI and FreeBSD have not
  compiled the final tree.
- The other platforms saw parts of the new code under emulators, wine, mock drivers or a cross compiler only. The
  changelog entries say, part by part, which.
- **What has not been tested, in one place, with the commands to test it: [UNTESTED.md](UNTESTED.md).**

CI is `.github/workflows/ci.yml`; it runs on every pull request and every push to main. The workflow in the tree is
newer than its last run: its emulation, wasm, cross-zig, perf, repro and fuzz jobs and the whole-suite sanitizer jobs
have never run on GitHub, and `linux-clang` now uses clang 23 (clang 18 in the last run).
The compiler baseline: clang 23 is the project compiler; configure refuses clang 18 and older; clang 19 to 22
configure with a note (zig cc, the NDK, FreeBSD base clang).
"Skip" means a test exited 77 because its device or tool was missing.

## Platforms

### The final tree (local, 2026-11-13)

| Platform | Built | Tested | How |
|---|---|---|---|
| Linux x86_64, clang 23.1.6 | locally, c452a61 | ctest `-LE "slow\|emu"`: 247 tests, 237 pass, 10 skip (`test_text_conformance`: Unicode files not downloaded; the nine real-device GPU tests: no device), 0 fail. Labels `slow`, `emu`: not run | 4-core VM, `cmake --preset dev` with `clang++-23` |

### CI (last run PR #53, 2026-11-04) and local emulation

The CI rows are results of the tree **before** the 2026-11-12/13 work; no CI job has built the tree since. The
emulation rows are local and dated.

| Platform | Built | Tested | How |
|---|---|---|---|
| Linux x86_64, GCC 14 | CI `linux-gcc` | ctest `-LE "slow\|emu"`: 111/111 pass (GPU device tests skip) | Ubuntu 24.04, `cmake --preset dev` |
| Linux x86_64, Clang 18 | CI `linux-clang` | same, 111/111 | Ubuntu 24.04. The job now installs clang 23 from apt.llvm.org; that form has never run on GitHub |
| Linux, ASan + UBSan | CI `linux-asan` | `test_script` only | Clang 18, `FORGE_SANITIZE=ON`. The job now runs the whole suite with clang 23: never run on GitHub |
| Linux, TSan | CI `linux-tsan` | `test_script` only | Clang 18, `FORGE_TSAN=ON`. The job now runs 25 threaded suites with clang 23: never run on GitHub |
| CI jobs `linux-fuzz`, `emu` (six rows), `wasm-simd`, `cross-zig`, `perf`, `repro` | never run on GitHub | — | added 2026-11-12; checked locally only for YAML syntax and that every ctest name, label and target they name exists |
| Linux, Vulkan on lavapipe | CI `linux-vulkan` | `test_vulkan`, `test_vulkan_align256`, `golden_tiny_llama_vulkan`, `golden_moe_mla_vulkan` (added 2026-11-05; first CI run pending) | GCC 14, Ubuntu `glslang-tools` + `libvulkan-dev` + `mesa-vulkan-drivers`; loader pinned to the lavapipe ICD |
| macOS | CI `macos` | ctest 111/111, including `test_gpu_metal` | `macos-latest` runner, Xcode toolchain |
| iOS | CI `ios` | `test_os`, `test_text`, `test_locale`, `test_mobile`, `test_util` in an arm64 iPhone simulator via `xcrun simctl spawn` (added 2026-11-05; first CI run pending) | `forge_c`, `forge_mobile`, `forge_text` built with Xcode for device, signing off; tests built with Ninja for the simulator (`FORGE_IOS_SIMULATOR=ON`, arm64) |
| Windows, MSVC | CI `windows-msvc` | ctest 106/106; script emit goldens | `windows-latest`, Ninja + `cl` |
| Windows, MinGW | CI `windows-mingw` | ctest: 84 run, all pass | MSYS2 UCRT64 GCC |
| FreeBSD 14.3 | CI `freebsd` | ctest `-LE "slow\|emu\|gpu"`: 60/60 | `vmactions/freebsd-vm` VM on Ubuntu |
| Android arm64-v8a | CI `android` | build only | NDK, `tests/mobile_build.sh` |
| Android x86_64 | CI `android` | `test_script`, `--emit android` goldens, `test_mobile`, `test_locale`, `test_text`, `test_os`, `test_os_ext`, `test_os_sys`, `test_util`, `test_loadbalancer` on an API 34 emulator (the last five added 2026-11-05; first CI run pending) | NDK build of `forge`, `test_script`, the example plugin; KVM emulator |
| Android, physical device | no | never | — |
| Linux aarch64 (SVE) | locally, zig cross | under qemu-user only (2026-10-31: 50/54, see changelog) | `qemu-aarch64 -cpu max,sveN=on` |
| Linux riscv64 (RVV) | locally, zig cross | under qemu-user only (same) | `qemu-riscv64 -cpu rv64,v=true,vlen=N` |
| Linux loongarch64 (LSX, LASX) | locally, zig cross: the simd tests only | `test_simd` 7/7, `test_simd_tables` 3/3, `test_simd_tiera` 14/14, `vec_types` checks under qemu-user only (2026-11-12) | `qemu-loongarch64 -cpu la464` |
| Linux ppc64le (VSX) | locally, zig cross and g++ 13: the simd tests only | `test_simd` 7/7, `test_simd_tables` 3/3, `test_simd_tiera` 14/14, `vec_types` checks under qemu-user only (2026-11-12) | `qemu-ppc64le -cpu power9`, `power8` |
| Linux s390x (z14 vector) | locally, zig cross and g++ 13: the simd tests only | `test_simd` 7/7, `test_simd_tables` 3/3, `test_simd_tiera` 14/14, `vec_types` checks under qemu-user only (2026-11-12) | `qemu-s390x -cpu max` |
| wasm32-wasi (SIMD128) | simd layer and its tests only, `tests/wasm_simd.sh` | `test_simd` 7/7, `test_simd_tables` 3/3, `test_simd_tiera` 14/14 (one test per process), `vec_types` checks under node 22 (2026-11-12) | zig `wasm32-wasi` |
| Linux mips64el (MSA) | locally, zig cross: the simd tests only | `test_simd` 7/7, `test_simd_tables` 3/3, `test_simd_tiera` 14/14, `vec_types` checks under qemu-user only (2026-11-12) | `qemu-mips64el -cpu Loongson-3A4000` |
| 32-bit x86 (SSE) | `test_simd` only, compiled by hand, no CMake configure | qemu-user only: 7/7 (2026-11-12) | `qemu-i386` |
| 32-bit Arm (NEON + VFPv4) | The simd tests only (`forge_simd_tests`), CMake + `armv7-neon-qemu.cmake` | qemu-user only: `test_simd` 7/7, `test_simd_tables` 3/3, `test_simd_tiera` 14/14, types 0 failed, `neon` the default table (2026-11-13). Never on hardware; never timed | `qemu-arm -cpu cortex-a7`, `-cpu cortex-a15` |
| Windows under wine | locally, mingw cross | historical full-suite passes (2026-10-29/31) | `tests/wine_run.sh` |

Windows emit goldens (`FORGE_TEST_ONLY=emit test_script`, `windows-msvc`): sh, bash, dash, pwsh and PowerShell 5.1
for all three goldens; cmd for `batch` (the other two are outside the cmd subset and refused).

Shell emit goldens (`shells` job, 2026-11-04 run): only sh, bash, dash, ash (busybox), pwsh and powershell were
found and tested. The other shells in the job's install list were not present in that run.

## GPU backends

| Backend | Real hardware | Mock / software | CI (PR #53) |
|---|---|---|---|
| CPU | every platform above | — | all jobs |
| CUDA | NVIDIA GeForce RTX 3090: WSL2 (2026-10-31, 2026-11-05) and native Windows 11, CUDA 13.2 driver (2026-11-04). None of the CUDA code of 2026-11-12/13 has run on it | mock driver (`test_gpu_*cuda_mock`, `golden_*_cuda_mock`; they skip when configure found no clang with the NVPTX backend) | mocks pass on `linux-gcc`/`linux-clang`; skip on macOS and Windows |
| HIP | **never** | mock driver | mocks pass on Linux |
| OpenCL | **never** | mock driver runs the generated OpenCL C as C | mocks pass on Linux |
| Level Zero | **never** | mock driver | mocks pass on Linux and `windows-msvc` |
| Metal | GitHub `macos-latest` runner's Metal device: `test_gpu_metal` passes (oracle check of every `metal.*` lowering; PR #53). The Metal ports of 2026-11-12 have **never** been compiled by a Metal toolchain or executed | syntax checks on Linux: the MSL as C++, `metal.mm` against stub declarations | `macos` |
| Vulkan | **never** on a GPU | Mesa lavapipe (local 2026-11-05: `test_vulkan` 33 kernels, 0 failures, both Vulkan goldens pass on llvmpipe; local 2026-11-12: 36 spec kernels, the 11 ported kernels and 13 Vulkan goldens, see the changelog); SwiftShader in the Android emulator (local, not since 2026-11-05) | `linux-vulkan` on lavapipe; compiled out in every other job |

No GPU was available during the 2026-11-12/13 work: everything GPU-related in it is mock-verified (CUDA, HIP, OpenCL,
Level Zero), lavapipe-verified (Vulkan) or syntax-checked only (Metal). Detail and the commands to run on hardware:
[UNTESTED.md](UNTESTED.md#4-gpu-backends).

`test_gpu_metal` passing means the runtime opened a device and all checks passed; the device name is not in the
CI log because ctest only prints output on failure.

### CUDA on RTX 3090, Windows 11 (2026-11-04)

`test_cuda_hip` cross-built with mingw-w64 and run natively:

| Check | Result |
|---|---|
| CUDA lowerings vs the float64 oracle | 31/31 |
| Quantised `matmul_t` cases | 24/24 |
| `delta_rule`, `causal_conv` state ops | match the reference |
| Split-K decode attention shapes | 7/7 |

HIP on that machine: mock only (no ROCm runtime).

### CUDA performance on RTX 3090 (2026-11-05)

Idle RTX 3090, Windows 11 host, WSL2 Ubuntu, clang 23, CUDA 13 driver 610.88. 512-token prefill and 128-token
decode, tok/s. llama.cpp is `llama-bench` at bc428e4 on the same GGUF file; FORGE imports that GGUF to `.fpk`.
"Before" is main before the `cuda-mma-prefill` branch.

| Model (Q4_K_M) | Prefill before → after | llama.cpp prefill | Decode before → after | llama.cpp decode |
|---|---|---|---|---|
| Qwen3.5 9B | 326 → 2720 | 4740 | 75.6 → 107.5 | 120.5 |
| Goedel-Prover-V2 32B (Qwen3) | 88 → 839 | 1360 | 26.2 → 35.3 | 38.4 |

Greedy output is identical with CUDA graphs on and off.

## CPU quantised matmul (2026-11-06, Xeon Gold 6242 x2, WSL2, 32 threads, machine shared with GPU jobs: noisy)
- `forge bench --kernels 1024` (1 GB matrix, k=4096), int8 avx512vnni vs f32 decode+dot: q4_k 27.1 vs 17.9 GB/s, q5_k 28.9 vs 19.5, q6_k 30.3 vs 21.5, q8_0 24.8 vs 21.2. Two vectors cost the same as one; eight run at 19–25 GB/s of weights (8x the work).
- q35 (Qwen3.5 9B Q4_K_M) `bench -p 16 -n 8`, same build and load: FORGE_CPU_INT8=0 prefill 4.4 / decode 0.6 tok/s, int8 8.0–8.2 / 1.1–1.4 tok/s. An earlier quiet-machine baseline (old build) was prefill 16.5 / decode 1.8 tok/s at -p 128; the load on the machine made later absolute numbers incomparable. Greedy tokens on the test prompt are identical to the f32 path.
- AArch64 integer kernels (2026-11-12, **emulation only**: qemu-user 8.2.2 on an x86_64 host, cross builds with GCC 13.3 and with clang 23.1, static glibc; never run or timed on Arm hardware): tables `neon` (ARMv8.0), `dotprod` (ARMv8.2 sdot) and `sve` (svdot, vector-length agnostic). `qdot_i8_matches_reference_on_every_isa` passes under `-cpu cortex-a57` (neon), `max,sve=off` (neon, dotprod) and `max` with SVE at 128, 256, 384, 512 and 2048 bits (`FORGE_SVE_KEEP_VL=1`, so 384 and 2048 are real lengths): every table equals the scalar reference by memcmp, including raw blocks at the ends of the integer ranges, and the reference's hash on fixed inputs is the same value on x86-64 (avx2, avxvnni, avx512vnni) and AArch64. Full `test_graph` (also with `FORGE_CPU_INT8=1`) and `test_simd` pass under `max,sve256=on`. ctest: `qi8_aarch64_*` (label `emu`, cross builds with a qemu emulator only). Default on AArch64 is still the f32 path: `FORGE_CPU_INT8=1` opts in. Compiled but never executed: the Apple, FreeBSD and Windows feature-detection branches (zig compile-only for aarch64-macos, -freebsd, -windows-gnu), Apple clang, the Android NDK, MSVC arm64.
- Not done: §2 interleaved prefill tiles, multi-product dispatch, per-token serial work on the pool, timing the AArch64 integer kernels on hardware (`forge bench --kernels <MB>` with `FORGE_CPU_INT8=1`) and then deciding their default.

## Known gaps

The full list, with commands, is [UNTESTED.md](UNTESTED.md). The headline items:

- **No CI run and no second compiler for the 2026-11-12/13 work.** One machine, one compiler (Linux x86_64,
  clang 23). GCC 14, MSVC, MinGW on Windows, Apple clang, the NDK in CI and FreeBSD have not compiled the final tree.
- **No sanitizer run on the final tree.** The hardening pass was cut short; the code of the second round of merges
  (forge-serve chat features and C ABI 4, simd companion tables and `vec_types.hpp`, fusion, the Vulkan / Metal ports
  and RPC protocol 2, model-container sections and session kinds) has not been through ASan / UBSan, TSan or
  libFuzzer as a whole. See the 2026-11-13 entry.
- **No published model.** Every importer, tokenizer, chat template and model family is tested on tiny random-weight
  models; several tables were written from memory and never diffed against their sources
  ([UNTESTED.md](UNTESTED.md#8-written-from-memory-never-diffed-against-the-source)).
- **No GPU for the new GPU code.** HIP, OpenCL and Level Zero have never run on real hardware; Vulkan never on a real
  GPU (lavapipe only; CI builds it only in `linux-vulkan`); the Metal ports have never been compiled by a Metal
  toolchain; the CUDA additions have not run on the RTX 3090. CUDA: one GPU model, no multi-GPU or peer-copy test on
  hardware, no graphics-interop import.
- **Kotlin and Swift bindings (ABI 4): never compiled.**
- iOS: the last CI run only built the library; the simulator tests added 2026-11-05 have not had a CI run; never
  run on a device. The Swift package and XCTest have not been run. The wave 3 / 4 OS branches have not been compiled by the iOS SDK.
- Android: never run on a physical device. The new code was compiled with NDK r27c for the OS layer and never run.
- macOS, FreeBSD: OS-layer waves 1 to 4 cross-compiled with zig, never run. Windows: waves 3 and 4 under wine only.
- Non-x86 CPUs: emulation only (qemu-user, node); most architectures ran the simd tests only.
- `test_rpc_tls` on Windows CI: the `openssl` CLI probe no longer uses a shell redirect, but it has not yet run there.
- AMX/SME: AMX ran natively and under Intel SDE; SME only under qemu.
- GPU memory: no eviction while a CUDA/HIP stream capture is open. Graphs recorded before an eviction are re-recorded
  at launch (2026-11-12, mock-verified only); a stale CUDA/HIP capture that holds an evicted encoded weight or a
  macro-stamped kernel throws instead ([gpu.md](gpu.md#eviction-and-recorded-graphs)).
- RPC: revocation is by CRL file and fingerprint list (no OCSP, no CRL fetching); device ops have no per-model scope. The finer permissions, SAN rules and revocation (`test_rpc`, `test_rpc_tls`) have run on Linux x86-64 with OpenSSL 3.0 only.
- Not tested anywhere: hybrid P/E cores, multi-node NUMA, cgroup v2 limit groups on a v2 hierarchy, batteries and
  thermal sensors, running under systemd or the Windows SCM, macOS kperf, VA-API/NVDEC/MediaCodec/VideoToolbox
  decode, CUDA graphics interop.
- Performance: everything timed in the 2026-11-12/13 work was timed on a loaded 4-core VM
  ([UNTESTED.md](UNTESTED.md#7-performance)).
- Security gaps: [security.md](security.md#known-gaps).

## Changelog of checks

Newest first. Results are from the date given; later code may differ.

### 2026-11-13 (simd: the 32-bit Arm `neon` table gives IEEE results and is selected automatically; emulation only)
Advanced SIMD on AArch32 runs flush-to-zero whatever `FPSCR` says, so the `neon` table failed `exp-sub`, `cbrt-sub`
and `erfc` and was opt-in. The Armv7 backend of `vec_gn.hpp` now tests each `add` / `sub` / `mul` / `fma` / `lt` and
recomputes the lanes NEON may have got wrong (subnormal operand, flushed result, NaN) with VFP instructions
([simd.md](simd.md), "32-bit Arm"). `FORGE_SIMD_ARM32_NEON` is gone; `neon` is chosen when `AT_HWCAP` has NEON and
VFPv4; the table has Tier A, `seq` and int8 companions (`isa_extra.cpp`).
- zig 0.17.1 (clang 22) `arm-linux-musleabihf`, static, CMake with the new `cmake/toolchains/armv7-neon-qemu.cmake`,
  `ninja forge_simd_tests`, under `qemu-arm -cpu cortex-a7` and `-cpu cortex-a15`, same results on both: `test_simd`
  7/7 with `neon` active by default and forced, 7/7 with `FORGE_SIMD_TARGET=scalar`; `test_simd_tables` 3/3 (`seq` and
  int8 tables bit-identical to scalar); `test_simd_tiera` 14/14; `test_simd_types` 3/3; `test_simd_types_neon32`
  0 of 25.5 million type checks failed and 0 of 13.5 million lane results differ from VFP arithmetic. ctest
  `lane_neon`, `tables_neon`, `tiera_neon`, `types_neon32` and the four `_a15` ones pass. No tolerance or sweep
  was changed; `math_special_values` gained three exact subnormal cases for every table.
- g++ 13 `arm-linux-gnueabihf` (glibc, the table's files with `-mfpu=neon-vfpv4`), the same CMake configure,
  `QEMU_LD_PREFIX=/usr/arm-linux-gnueabihf`: the same executables and the eight ctest entries pass (13 ctest entries in all).
- x86_64, clang 23: `forge_simd_tests`, `ctest -R "simd|lane|^types_|^tiera_|^tables_"` 16/16.
- Not done: any run on 32-bit Arm hardware; any timing (every checked operation carries one to four extra NEON
  compares and a branch, so whether `neon` beats `scalar` on a real core is unknown); any test other than the simd
  ones on 32-bit Arm; the other `vec_gn.hpp` tables (LSX / LASX, VSX, VXE, SIMD128, MSA) were not re-run after the
  shared `ldexp_int` / rounding helpers were reworded (same operations, compare spelled `raw_lt`); the `emu` CI row
  added for 32-bit Arm has never run on GitHub.

### 2026-11-13 (final state of the integration branch; what was not run; owner decisions)
Integration branch `ccr-54bb5525-7f3iii` at c452a61: about 27 feature branches merged over 2026-11-12/13, roughly
191,000 lines added across about 1,030 files since main at b135941. This entry is the record of the final state; the
2026-11-12 entries below are the per-topic records and were written before the last merges.

- **Build and test.** Full build with clang 23.1.6 on Linux x86_64 (4-core VM). `ctest -LE "slow|emu"`: 247 tests,
  237 passed, 10 skipped, 0 failed. Skipped: `test_text_conformance` (the Unicode conformance files are not
  downloaded) and nine real-device GPU tests (`test_gpu_cuda`, `test_gpu_hip`, `test_gpu_features_cuda`,
  `test_gpu_features_hip`, `test_gpu_metal`, `test_gpu_opencl`, `test_gpu_features_opencl`, `test_gpu_ze`,
  `test_gpu_features_ze`: no device).
- **Not run on the final tree.** Labels `slow` and `emu`. Any sanitizer build. Any libFuzzer build. Anything on a
  GPU, on another CPU architecture, or on another operating system.
- **No CI run has happened for any of this work.** GCC 14, MSVC `cl.exe`, MinGW on real Windows, Apple clang (macOS,
  iOS), the Android NDK build in CI and FreeBSD have not compiled the final tree. The CI workflow gained many jobs
  (emulation matrix, wasm, cross-zig, perf, repro, fuzz, whole-suite sanitizers) that have never executed on GitHub.
- **Compiler baseline.** clang 23 is the project compiler. Configure refuses clang 18 and older; clang 19 to 22
  configure with a note (zig cc, the NDK, FreeBSD base clang). Owner decision: this stays as it is.
- **Hardening pass cut short.** The pass that was to run ASan / UBSan, TSan and libFuzzer over the newest code was
  interrupted by machine restarts. Its eleven fix commits are merged:
  - 0b50d40 script: the heap mutexes are taken around `fork()` (a forked ProcessPool worker could deadlock at start;
    seen as `test_script` hanging in the whole-suite ASan run)
  - 089a5c2 script: a container constructor that hits the heap cap left a dangling pointer in the cycle collector
    (heap-use-after-free, found by ASan in `test_script_golden` / `test_script_security`)
  - 97a391e script: references no longer copy or destroy the object; delegating and base constructors; typed members
  - b4ec7e2 graph/codecs: f32 encode / decode of an empty tensor must not call `memcpy` with null pointers (UBSan,
    through `fuzz_gguf` on its corpus)
  - 86af32e gpu/cuda: the stream-K fix-up scratch and flag buffers are freed when the runtime is destroyed
    (LeakSanitizer, through the CUDA mock)
  - e5c73cc tests/mock_cuda: captured kernel arguments are copied by their own size; function handles interned
    (stack-buffer-overflow and leaks in the 23 `*_cuda_mock` / `*_hip_mock` ctests under ASan)
  - 18c559b `forge rpc-serve`: no address-space cap on the script child in ASan / TSan builds (`signing_cli` and
    `security_cli` failed in the whole-suite ASan run)
  - 1952e96 cmake: `python_bindings` is not registered in sanitizer builds
  - a715424 tests: the script golden runners' 20 s whole-run timeout becomes 300 s (an ASan Debug build on a loaded
    machine exceeded it)
  - d540514 fuzz: a structure-aware seed generator for `fuzz_aot` (`fuzz/aot_seeds.py`)
  - 4d01ea3 cmake: AOT binaries built in a sanitizer tree link the sanitizer runtime

  Its planned full ASan / UBSan pass, its TSan pass and its libFuzzer campaigns over the second-round code (forge-serve
  chat features and C ABI 4; the simd companion tables, `vec_types.hpp` and fusion; the Vulkan / Metal ports and RPC
  protocol 2; the model-container sections and session kinds) **did not complete**, and it left no run table. Which
  suites finished clean before the interruption is not recorded. Earlier sanitizer coverage is what the dated entries
  below record; [UNTESTED.md](UNTESTED.md#6-sanitizers-and-fuzzing) lists it in one table.
- **One integration bug found at the very end**, fixed in c452a61: the CUDA test mock sized captured kernel
  arguments from a hand table that did not know the generated conv / pool / norm / `attention_bias` kernels, so a
  graph replay launched them with truncated pointers (crash in `test_gpu_cuda_mock` and five `*_cuda_mock` goldens).
  The generated trampoline table now carries each kernel's pointer-argument mask. Those mock tests only run when
  configure found a clang with the NVPTX backend; with none they skip.
- **Written from memory and never diffed against the source** (so a test passing does not show agreement with the
  original): the AutoAWQ / AutoGPTQ packing layouts (verified only against a self-written packer); the llama.cpp
  pre-tokenizer regex table and `ignore_merges` list; the vendor chat-template texts in `tests/data/chat_template`;
  the GGUF tensor names and conventions for gemma / phi3 / bert / gpt2 and several Gemma specifics; the RoPE-scaling
  reference (a transcription of transformers' `modeling_rope_utils.py`) and the GGUF YaRN attention-factor formula;
  the per-family "leave-out" tensor lists. **No published checkpoint or tokenizer of any family has been imported or
  run in this work**; all model tests use tiny random-weight models.
- **Owner decisions (2026-11-13).**
  1. The licence of the `riscv-non-isa/rvv-intrinsic-doc` prototype lists behind the RISC-V intrinsic databases stays
     as recorded in [NOTICE](../NOTICE) ("not established"). It is not being pursued now and is no longer listed as
     an open action item.
  2. The seven grid-codebook GGUF types (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M) will not be supported:
     FORGE is clean-room and the codebooks will not be copied. The importer keeps refusing them with an error that
     names the type ([modelfile.md](modelfile.md#gguf-tensor-types)).
  3. The 32-bit Arm NEON table is kept and fixed; see its own entry above.
  4. The clang floor stays as described above.
- **Documentation.** [UNTESTED.md](UNTESTED.md) is the single list of what has not been tested, organised by
  platform, CPU architecture, GPU backend and feature, with commands. README.md, TODO.md, docs/index.md and the
  guides were brought in line with the final tree ([claims-audit.md](claims-audit.md), 2026-11-13).

### 2026-11-12 (model container, importer and session kinds: the plan-versus-code gaps of FORGE_PLAN Parts D and E)
Same machine as below: Linux x86_64, **clang 23** (`-fsyntax-only` with g++ 13 on the new headers also passes). Nothing
here has been through CI or built by MSVC, MinGW, Apple clang, the NDK or FreeBSD's compiler, and nothing was run on a
published model: every model in these tests is a tiny one built by the tests.

| Added | Verified how | Not verified / not done |
|---|---|---|
| Chunked `.fpk` encryption, tensors decrypted on demand ([format.md](format.md#encryption)) | `test_modelfile` `fpk_chunked_encryption`: three chunk sizes; reorder, three truncations, four splices, bit flips in ciphertext / tag / header, index edit, all refused; session on a sealed file; the single-AEAD scheme still reads. `fuzz_fpk` with sealed seeds | No file written by a FORGE build from before the change was kept to read back (the single-AEAD writer path produces that format). `forge encrypt` still holds the tensor data in memory. Locked plaintext depends on `RLIMIT_MEMLOCK` |
| Tensor alignment 64 / 4096 / 65536 ([format.md](format.md#tensor-alignment)) | `fpk_tensor_alignment` (writers, `sign`, `encrypt`, `aot` keep it); `golden_tiny_llama` (`--align`, logits unchanged) | No measurement that page alignment helps loading or device upload; the 64 MiB threshold is a size rule, not a tuned one |
| `state`, `adapters`, `calib` sections; `forge requant`, `forge adapter` | `fpk_state_schema_section`, `fpk_embedded_adapters`, `fpk_calibration_section_and_requant`; `golden_tiny_llama` (statistics against numpy, requant equals a direct import tensor for tensor) | The calibration rule (crest factor) is a heuristic with no quality measurement. Weight statistics only, no activation calibration. No PEFT adapter importer |
| Binary blueprint form ([format.md](format.md#binary-blueprint)) | `blueprint_binary_form_round_trips`: 16 templates and 7 corpus blueprints, byte-identical canonical text, every truncation and bit flip; `fuzz_blueprint` on both forms | Importers still write text; conversion is `forge blueprint` |
| Sharded models, content-addressed blob store ([format.md](format.md#shards)) | `fpk_shards_and_blob_store` (dedup counts, missing / corrupt / swapped blobs, forged index against the signature, traversal and symlink names); `golden_tiny_llama` (CLI, aot and binary blueprint on a sharded model) | Only local directories: no download, no garbage collection of unreferenced blobs. Models of a few hundred kilobytes only; `--shard-size 2G` has not split a file that large |
| ONNX importer: CNN, shape, reduction and movement operators; shape-subgraph folding; external data; integer inputs; several symbolic dimensions ([modelfile.md](modelfile.md#onnx-graphs)) | `onnx_import`: 78 outputs of 10 graphs against `onnx.reference.ReferenceEvaluator` and onnxruntime 1.29; 8 refusals; 7 bad external-data references. `fuzz_onnx` with 8 new seeds | No published ONNX model. Convolution-family operators only at batch 1. See the "Not done" list in the guide |
| Blueprint opset 4: session parameters, entry points; `session diffusion_loop`, `session streaming` through `inference::session`; `forge denoise`, `forge stream`; blueprint-directory import ([inference.md](inference.md#session-kinds-chosen-by-the-file-diffusion_loop-and-streaming-file_sessionshpp)) | `test_sessions` (runners against the graph computed by hand); `golden_tiny_sessions` (numpy references: five schedule / prediction / sampler combinations, guidance, streaming with carried conv state, chunk-size invariance) | Tiny MLP denoiser and a 6-channel streaming encoder only: no UNet / DiT, no VAE, no audio front end. Not exposed by `forge-serve` or the C ABI. CPU only |
| Self-speculative decoding (early exit) ([inference.md](inference.md#speculative-decoding-speculativehpp)) | `test_inference` `self_speculative_early_exit_is_exact`: equal to plain decoding for greedy and three seeded samplers, exit layers 1 to 3, a recurrent hybrid; `golden_tiny_llama` (CLI) | No trained model: acceptance (2 % random, 97 % with damped late layers) says nothing about real models. No speed measurement. The early layers' KV is not shared with the verify pass |
### 2026-11-12 (GPU follow-up: Vulkan and Metal ports, device conv / norm ops, eviction and captures, prims, pipe:auto, RPC residency)
Linux x86_64, clang 23, no GPU. "lavapipe" = Mesa 25.2.8 llvmpipe (LLVM 20), a CPU implementation of Vulkan that
executes the embedded SPIR-V; "mock" = the mock drivers of the entry below. Nothing in this entry ran on a physical
GPU, and nothing was built by a compiler other than clang 23.
- **Vulkan ports of the portable kernels** (`VK_PORT` in `ops/gen/gpu_gen.py`: quantised matmul_t / gather / FFN
  gate-up for q8_0, q4_k, q5_k, q6_k, one-pass and split-over-keys attention, delta_rule, causal_conv, broadcast,
  device MoE; host side `gpu::portable_kernels<>` on `gpu::vk::runtime`). **Lavapipe-verified**: all 47 shaders build
  with glslang 15.1 and pass `spirv-val`; `test_vulkan` and `test_vulkan_align256` (`FORGE_VK_SSBO_ALIGN=256`: views
  staged through scratch buffers) run the checks of `tests/gpu_checks.hpp`, which `test_cuda_hip` runs on the mocks:
  14/14 attention shapes, 24/24 quantised matmul cases, 28/28 against the decoded-weight reference (worst 1.2e-6 of
  the largest output), 20/20 fused FFN cases, state ops (3 shapes, 2e-8), 4/4 quantised gather, 21/21 MoE cases, 8/8
  broadcast cases, every `vk.*` lowering through the oracle. Goldens with `FORGE_DEVICE=vulkan`: the existing three
  plus `golden_{hf_quant,quant_inplace,tiny_qwen2,tiny_qwen3,tiny_gpt2,tiny_gemma,tiny_phi3,tiny_bert,tiny_vit,tiny_whisper}_vulkan`
  (13 in all) pass. `tiny_qwen3_5` (DeltaNet end to end) was skipped: no torch here; the state ops are covered by the
  check above only. First attempt failed the oracle on decode attention by 1.2e-5 against 1e-5 (one `fma` chain per
  score); the ports sum scores in 4 partial sums like the spec shader. Never on a GPU: descriptor limits, real
  drivers' SPIR-V compilers and buffers over the tested sizes are untested.
- **Vulkan dispatch folding**: launches are folded over the second dispatch dimension (rows of 32768 groups); before,
  a launch of more than 65535 × 64 work items exceeded what `maxComputeWorkGroupCount` guarantees. Lavapipe: a
  2,098,152-item `add` (0 wrong). The 36 spec shaders were regenerated with the linear index and pass as before.
- **Metal ports of the same families**: `kernels.metal` gets the eleven portable kernels through a textual
  translation of the OpenCL C (`msl_port`), and `gpu/src/metal.mm` gets the host side (`gpu::portable_kernels<>`).
  **Never executed, never compiled by a Metal toolchain.** Checked here: the generated MSL passes the C++ syntax
  check `gpu_compile_metal_syntax` (45 kernels; address-space keywords vanish in that check), and `metal.mm` parses
  as Objective-C++ against stub declarations of the Metal API (`gpu_compile_metal_host_syntax`, new). That second
  check found that `forge_metal` lacked the `third_party/monocypher` include directory `core/memory.hpp` needs since
  the 2026-11-12 merges (not verified on macOS, but the include path was missing from the target); added. `test_gpu_metal` now runs the attention, quantised,
  state-op and MoE checks too, so the macOS CI job is the first place any of this runs.
- **Device kernels for `gelu`, `layer_norm`, `group_norm`, `conv1d`, `conv2d`, `max_pool2d`, `avg_pool2d`,
  `attention_bias`** on every backend (one portable source, `PORT_NN`). **Mock-verified** on CUDA, HIP, OpenCL, Level
  Zero: `test_gpu_{cuda,hip,opencl,ze}_mock` 42/42 lowerings through the oracle (34 before) and 26/26 cases of the new
  `nn_ops_check`. The CUDA / HIP mock now compiles the generated portable source as C for these kernels (no hand
  copy); the PTX (sm_70, sm_80) and AMDGPU builds of that text compile and have not run. **Lavapipe-verified** on
  Vulkan: `gelu.comp` and the seven GLSL ports pass the oracle and the same 26 cases, also with
  `FORGE_VK_SSBO_ALIGN=256`. **Metal: never executed** (`msl_port` translation, C++ syntax check only). Goldens of the
  six template families pass on the CUDA mock, the OpenCL mock and lavapipe; with `FORGE_PLAN_TRACE` the OpenCL-mock
  runs show no `attention_bias`, `layer_norm`, `gelu` or convolution left on the host (remaining: `transpose`,
  `l2_normalize`, `masked_mean`). A broadcast with a single-element operand (`x * [1]`) now also stays on the device.
  No GPU has run any of it and nothing was timed.
- **Eviction and recorded graphs** (buffer generations, `gpu/graph_refs.hpp`): a graph whose buffers were evicted,
  released or re-created is detected at `graph_launch` and re-recorded from the host views (per launch on OpenCL /
  Level Zero, the whole captured graph on CUDA / HIP); OpenCL and Level Zero now evict while a recording is open.
  **Mock-verified** only: `test_gpu_memory_{cuda,hip,opencl,ze}_mock` step 6 (6 checks per backend, see
  [gpu.md](gpu.md#eviction-and-recorded-graphs)). On the mocks the CUDA / HIP re-record uses the generated
  parameter-size table, since the mock has no `cuFuncGetParamInfo`; the driver path of that lookup has not run.
  Still not possible: eviction while a CUDA / HIP stream capture is open. Not run on the RTX 3090.
- **`gpu::prims` (FORGE_PLAN B.6)**: `gemv`, `gemm_batched` with an epilogue, `conv1d` / `conv2d`, `fft`,
  `softmax`, `rms_norm`, `layer_norm`, `flash_attention` on the CPU (`prims.hpp`), and all but the FFT on a device
  through the generated kernels (`prims_device.hpp`). CPU: 5 new `test_gpu` tests against the float64 references on
  this x86_64 machine only. Device: `prims_check` 22/22 on the CUDA, HIP, OpenCL, Level Zero **mocks**
  and on **lavapipe**; Metal never executed; no GPU. The device epilogue is separate launches, not a fused kernel.
- **`pipe:auto`** (`gpu::plan_pipeline`): the layer split over the devices of one backend, shares from throughput,
  capped by device memory, overflow to a CPU stage. Synthetic records in `test_gpu` (`pipeline_auto_partition`);
  **mock-verified** with `MOCK_DEVICES=2` on CUDA, HIP, OpenCL, Level Zero (`test_gpu_select_*_mock`: the two-stage
  split, an 8-layer graph running 8 + 8 ops over both devices against the CPU, the one-device case) and
  `golden_tiny_llama_pipe_auto_cuda_mock`. The mocks report equal priors, so only the 50 / 50 split ran on a driver;
  the memory-capped and benchmark-weighted splits ran on synthetic records only. Stage hand-off is through host
  memory: peer copies are not used by the pipeline (the peer-copy API is covered by `test_gpu_features_*_mock` as
  before). No multi-GPU machine has run it.
- **RPC remote weight residency** (protocol version 2 in `graph/rpc.hpp`: the `rpc.caps` probe, `FRPW` uploads keyed
  by BLAKE3, `FRPR` ops with handles, a per-connection LRU under a shared byte budget, invalidation on reconnect).
  Tested over **loopback only**, plain TCP, CPU server: `test_rpc` 7 new tests (upload once and the bytes saved, LRU
  eviction and re-upload, reconnect by hand and after a server restart, a version 1 server and a version 1 client,
  `allow=` / `max-bytes=` on uploads and on handles, hostile frames, a whole model against the CPU); the 24 older
  `test_rpc` tests pass unchanged (the TLS ones included: this build has OpenSSL). `fuzz_rpc`: 32 new seeds, 150,000 mutated inputs without a finding, in a build
  **without** sanitizers. Not run: two machines, TLS-specific paths, a GPU behind the server (the device pin of a
  stored weight has only run against the CPU registry, where it is a no-op), concurrent connections under load. `forge rpc-serve` still opens its device non-resident, so only the network
  transfer of a weight is saved, not the server's host-to-GPU copy.
- The checks shared by `test_cuda_hip` and `test_vulkan` moved to `tests/gpu_checks.hpp`; the 26 `test_gpu_*_mock`
  tests pass unchanged after the move.

### 2026-11-12 (session summary: everything merged on this day, and how each part was verified)
One machine did all of this: Linux x86_64 (4 cores, kernel 6.18), **clang 23**, no GPU. Nothing below has been
through GitHub CI, and GCC 14, MSVC, MinGW on Windows, Apple clang and FreeBSD's compiler have not built most of it.
The per-topic entries further down (simd tables, GPU parity) are the detailed records of two of the rows; this entry
is the index of the whole day. "Native" means built and run on that machine.

Whole tree after the last merge (`ctest -LE "slow|emu" -j2`, clang 23, RelWithDebInfo): 209 tests: 199 passed, 10 skipped, 0 failed. The skips are the nine real-device GPU tests (`test_gpu_{cuda,hip,opencl,ze,metal}` and their `features` variants: no driver) and `test_text_conformance` (the Unicode conformance files are not downloaded). `test_script` passes with 15 shells installed, including `rc` behind an unrelated npm tool of the same name on `PATH`. Also run by hand: `test_aot_native` (new family case) and `ctest -R script_aot` 3/3 (labels `slow`, `fuzz`). Not run: the rest of label `slow`, label `emu`.

| Added | Verified how | Not verified |
|---|---|---|
| Model AOT: `forge aot` plan mode and native mode ([aot-models.md](aot-models.md)) | Native. `test_aot` (bitwise against run-time plans, hostile and corrupt blobs, signed files), `test_aot_native` (label `slow`; shared library, embedded weights, executable against `forge run`), `fuzz_aot`. After the merges: 25 tiny models of every template family in both modes by hand, all bit-identical | Cross compilation, another compiler than the build's, Mach-O / PE output, any non-x86 or non-Linux host, models larger than the tiny ones |
| Script AOT: `forge script --aot` ([aot.md](aot.md)) | Native: `test_script_aot`, `test_script_aot_native_sweep` (clang++ 23 and g++ 13 as the script compiler), `fuzz_script_aot` | Windows, macOS, FreeBSD, Android, iOS: never run. Cross-compiling in native mode |
| Script language fixes (catch-clause matching, fatal errors, `.fsc` version 4), conformance sweep | Native: `test_script*`, `script_conformance` (each program also built as real C++23 and the outputs compared) | Other platforms through CI only |
| Tokenizers (Hugging Face pipeline, SentencePiece BPE, Unigram, WordPiece, WordLevel) and chat templates (Jinja subset) | Native: `test_tokenizer` and `test_chat_template` against goldens produced by the `tokenizers` 0.23, `sentencepiece` 0.2 and `jinja2` packages; `chat_template_http`; fuzz targets. Text to ids to model output through `forge run --prompt` / `forge embed --text` in the family goldens | Tokenizers of full-size published models; the llama.cpp regex table and the model chat templates were written from memory |
| Template families GPT-2, Gemma 1 / 2 / 3, Phi-3, BERT, ViT / CLIP / SigLIP, Whisper encoder; 24 spatial and encoder ops; embedding sessions; opset 3 | Native: `test_ops_ext`, `golden_tiny_{gpt2,gemma,phi3,bert,vit,whisper}` against float64 numpy references (HF and GGUF imports, f32 and q8_0); the same goldens on the CUDA and OpenCL mock drivers | Any published checkpoint; any real GPU |
| forge-serve: end-of-turn stops, tool calls, request fields (`n`, `logprobs`, `logit_bias`, ...), keep-alive HTTP, API keys, TLS / mTLS, `--lora`; C ABI 4 (shared weights, sampler chain, adapters, KV state, embeddings, batching); Python / Kotlin / Swift bindings on it | Native (clang 23, Linux x86_64, tiny golden models): `serve_api`, `serve_auth`, `test_chat_tools`, `fuzz_serve_http`, `capi_smoke`, `doc_example_capi_generate`, `python_bindings`, `jni_smoke` (the JNI C bridge on a host JVM), `jni_decl_matches`. forge-serve without TLS: built by hand once and run through `serve_auth` | No real chat or tool-calling checkpoint served. **Kotlin and Swift sources never compiled.** Other compilers and OSes through CI only; Windows shutdown path, `device = auto` through the C ABI: not run |
| `/v1/embeddings`, `forge embed --text` / `--text-pair`, opset enforcement per feature | Native: `serve_embeddings`, `test_modelfile` (`blueprint_opset_enforced_per_feature`, `shipped_templates_declare_their_opset`), the family goldens | — |
| Importer fixes: Qwen2 bias template, RoPE scaling, tensor accounting, GGUF tensor types and `--codec` for GGUF, AWQ / GPTQ checkpoints, licence capture | Native: `golden_tiny_qwen2`, `golden_rope_scaling`, `golden_gguf_types` (against gguf-py), `golden_hf_quant`, `test_modelfile` | Published checkpoints |
| GPU: portable OpenCL / Level Zero kernels, device MoE, kernel program cache, determinism mode, automatic device selection | **Mock drivers only** (the OpenCL mock compiles the generated OpenCL C as C and runs it). The Level Zero mock is handed the OpenCL C source (`FORGE_ZE_TEST_IMAGE=opencl-c`): this build has no SPIR-V image because `llvm-spirv` is not installed, so the SPIR-V path itself did not run. See the GPU entry below | Any real HIP, OpenCL, Level Zero or Vulkan device; the CUDA changes on the RTX 3090; Metal (MSL syntax check only) |
| Gemma-2 q8_0 "OpenCL less accurate": found to be the exact path against the int8-activation path, golden re-conditioned ([gpu.md](gpu.md#quantised-matmul-exact-and-int8-activation-paths)) | Native CPU (`FORGE_CPU_INT8=0` bit-equal to the OpenCL mock), CUDA and OpenCL mocks; `test_cuda_hip` `exact_reference_check` on the CUDA, HIP and OpenCL mocks | Real devices |
| SIMD kernel tables: SSE2 / SSE4.2, LoongArch LSX / LASX, POWER VSX, s390x, wasm SIMD128, MIPS MSA, 32-bit Arm NEON | x86 tables native and under `qemu-x86_64`; the others **`test_simd` only, under qemu-user or node**, built with zig. See the simd entry below | Real hardware of those architectures; any other test there |
| AArch64 int8 quantised kernels (NEON, dotprod, SVE) | qemu-user (`qemu-aarch64`); off by default | Hardware; timing |
| Raw intrinsic wrappers for eight ISA families (103,666 wrappers) | x86 natively (`test_simd_raw`); every wrapper of every architecture **compiled only** with `clang++-23 --target=<triple> -fsyntax-only` (`raw_intrinsics_cross`) | Execution of any non-x86 wrapper; MSVC; GCC beyond the recorded rejections |
| OS layer, waves 1 to 4 (processes, files, IPC, system, hardware, network, services, reactor, limit groups, D-Bus client, X11 clipboard, proxy, appearance, SNTP) | Linux native (`test_os*`; D-Bus also against a real `dbus-daemon`, clipboard against Xvfb, ASan + UBSan and TSan on the wave 4 tests). Windows: MinGW and clang-cl + MSVC STL builds **under wine 9**. macOS, FreeBSD: **cross-compiled and linked with zig, never run**. Android: **compiled with NDK r27c, never run**. iOS: **not compiled**. Table: [os-parity.md](os-parity.md) | Real Windows, macOS, FreeBSD, Android, iOS; `cl.exe`; cgroup v2 on a v2 hierarchy |
| Support modules: work-stealing scheduler, Tier A algorithms, linear algebra, profiler, reflection, vision (PNG / BMP / PNM, video bitstream parsing) | Native: `test_sched` (also TSan), `test_simd_tiera`, `test_linalg`, `test_prof`, `test_reflect`, `test_vision`; image fuzz targets | Perfetto / Tracy output in the tools; `perf_event`; hardware video decode; the C++26 reflection branch (never compiled) |
| RPC: SAN rules, revocation, fine-grained permissions; spill encryption; `safe_join` in template and bundle paths; script `--mem-limit` and indirect-call capability analysis | Native: `test_rpc`, `test_rpc_tls` (OpenSSL 3.0), `test_memory`, `test_script_security` | Other TLS stacks and platforms |
| Install rules, CMake package, relocatable templates | Native: `install_consumer` | Other platforms |
| CI workflow: emulation rows, wasm-simd, cross-zig, perf, repro, fuzz, whole-suite sanitizer jobs; SBOM (SPDX, CycloneDX); coverage table | The workflow has **never run on GitHub**. Checked here: it parses as YAML, every ctest name, label and build target it names exists in a clang 23 build tree, and the three copies are one file (`ci_workflow_copies`). `sbom_spdx`, `coverage_table`, `docs_links` pass | Every job on a runner; the apt package names for loongarch64 / mips64el cross compilers and `llvm-toolchain-noble-23` |
| NOTICE / SBOM: attribution of derived data | Licence statements read from clang 23's headers and from unicode.org on this day | The licence of `rvv-intrinsic-doc` (recorded as not established); the WHATWG / vendor-table statements |

Sanitizers on this day: after the last merge, clang 23 `-DFORGE_SANITIZE=ON` (ASan + UBSan, leak detection on) over `test_tokenizer`, `test_modelfile`, `test_chat_template`, `test_ops_ext`, `test_aot`, the gpt2 / gemma / phi3 / bert / llama goldens with the sanitized `forge`, and `serve_embeddings` with the sanitized `forge-serve`: no report. The whole-suite ASan run and the TSan run of the threaded suites that the workflow defines were last done before the final merges (see [testing.md](testing.md)); `test_sched`, `test_simd_tiera`, `test_linalg`, `test_prof`, `test_vision`, `test_reflect`, `test_script_aot` and the `test_os*` suites were not re-run under a sanitizer on the merged tree.

Environment notes for reproducing the run: Python packages numpy, safetensors, onnx, cryptography, gguf, tokenizers,
sentencepiece; `llvm-spirv` absent; shells found for the script emit goldens: sh, bash, zsh, dash, ash, ksh, mksh, yash, posh, fish, tcsh, csh, xonsh, elvish, rc.

### 2026-11-12 (simd: companion tables and element types on every ISA; std interop; CPU fusions; engine on the scheduler)
One machine, as above: Linux x86_64, 4 cores shared with other jobs, clang 23. Everything for other architectures
ran under qemu-user 8.2 or node 22 only; nothing ran on real non-x86 hardware and nothing was through CI.
- **Tier A, `seq` and int8 tables for every Tier P backend** (`simd/src/isa_extra.cpp`, `qi8_gn.inc`). x86_64 natively:
  `test_simd_tables` 3/3 (tables `avx512 avx2 sse4.2 sse2 scalar`; int8 `avx512vnni avxvnni avx2 sse4.2 sse2`),
  `test_simd_tiera` 14/14, `test_graph` with the extended table lists. Emulated, per CPU model, `test_simd_tables` 3/3
  and `test_simd_tiera` 14/14: AArch64 (g++ 13) `cortex-a57`, `max,sve=off`, SVE at 128 / 256 / 384 / 512 / 640 / 1024
  / 1536 / 2048 bits; riscv64 (zig 0.17) VLEN 128 / 256 / 512 / 1024; loongarch64 `la464` (`lasx`, `lsx`); ppc64le
  `power9`; s390x `max`; mips64el `Loongson-3A4000`; wasm32 under node (Tier A one test per process: node segfaults on
  the whole binary). The int8 reference hash of fixed blocks equals x86's on every little-endian target; s390x is
  only compared with its own scalar reference. g++ 13, by hand on x86_64: the `sse2` / `sse4.2` companion tables
  3/3 and the `sse4.2` / `avx2` / `avx512` `vec_types` checks 0 failed. No timing anywhere; the new int8 tables are
  off by default.
- **`vec_types.hpp`** (`vec<T>` for double and the integers): about 25.5 million checks per register width, exhaustive
  for 8-bit binary and 16-bit unary operations. 0 failed: x86 `scalar`, `emulated`, `sse2`, `sse4.2`, `avx2`, `avx512`
  natively; `neon`, `sve` 128 / 256 / 512 / 1024, `rvv` 128 / 256 / 512 / 1024, `lsx`, `lasx`, `vsx`, `vxe`, `msa`, `simd128`
  under emulation. Known: on loongarch64 with zig 0.17 the portable-array path (`emulated_t`, unused there) has 46,976
  differing checks from a compiler defect, reported by the test without failing it. Compiler defects worked around:
  MSA double `fma` / `abs`, LASX half-precision conversion (`tools/repro_llvm_vector_codegen.cpp`). MSVC is untested
  and uses the portable arrays.
- **`interop.hpp`**: `std::experimental::simd` conversions compiled and run with libstdc++ 13 (x86_64, and AArch64 under
  qemu); `std::mdspan` overloads compiled and run with libc++ 23 (`test_simd_std_libcxx`) and with zig's libc++ on
  s390x and riscv64 under qemu. C++26 `std::simd`: nothing (no library has it here). libstdc++ 15's `<mdspan>`: not tried.
- **CPU fusions** (`fuse_cpu.hpp`, `passes.hpp`; [graph.md](graph.md#fusion)): row chains and shared int8 activations
  bit-identical to the unfused plans on 60 random graphs / 27 quantised cases; conv + batch_norm folding within
  2.8e-7 of the largest output over 24 random cases (stated bound 2e-5). x86_64 only. Host plans only: device plans
  and `pipe:` stages are untouched. No model-level timing of the fusions was made.
- **Engine on `task::scheduler`** (`FORGE_TASK_SCHEDULER=sched`): `test_engine` and `test_graph`, `test_inference`,
  `test_kvquant`, `test_sessions`, `test_aot` pass under both executors; identical token ids from `forge run` on the
  wide and deep synthetic models. `forge bench` (9 interleaved runs, loaded machine): wide prefill 1472 (pool) vs 1458
  (sched) tok/s, wide decode 678 vs 418, deep prefill 28260 vs 13800, deep decode 5719 vs 2197. `pool` stays the default.
- Not done: Tier A's hand-written byte search / partition for non-x86 ISAs (they use the 8-byte integer path or the
  compiler); a fused GEMM epilogue; rope + attention fusion; 32-bit x86 / Arm companions; any of this on MSVC, GCC 14,
  Apple clang, NDK, FreeBSD.

### 2026-11-12 (simd tables for more architectures; emulation only)
New Tier P backend `vec_gn.hpp` and kernel tables `sse2`, `sse4.2`, `lasx`, `lsx`, `vsx`, `vxe`, `simd128`, `msa` and
(opt-in) 32-bit Arm `neon`. Only `test_simd` (7 cases: detection, tails, two ULP sweeps, special values, two oracle
comparisons; every usable table in one run) was built for the new architectures. Nothing ran on real hardware other
than x86_64.
- x86_64, clang 23: `test_simd`, `test_simd_raw`, `test_graph` pass with `FORGE_SIMD_TARGET` = scalar, sse2, sse4.2,
  avx2, avx512. `qemu-x86_64 -cpu Nehalem` (active `sse4.2`) and `-cpu core2duo` (active `sse2`): `test_simd` 7/7.
- zig 0.17.1 (clang 22) musl static builds through CMake and the new toolchain files, `test_simd` under qemu-user:
  loongarch64 `-cpu la464` 7/7 (also with `lsx` forced), ppc64le `-cpu power9` and `-cpu power8` 7/7, s390x
  `-cpu max` 7/7, mips64el `-cpu Loongson-3A4000` 7/7. g++ 13 cross builds of the same sources by hand: ppc64le 7/7, s390x 7/7.
- wasm32-wasi (zig), `tests/wasm_simd.sh`, node 22: 7/7 with `simd128` active.
- By hand, no CMake configure: 32-bit x86 under
  `qemu-i386 -cpu Nehalem` / `core2duo` 7/7; 32-bit Arm NEON under `qemu-arm -cpu cortex-a7` 5/7 (`exp-sub`,
  `cbrt-sub`, `erfc`: AArch32 NEON flushes subnormals), so that table is never selected automatically.
- zig's clang 22 hangs compiling musl with `-mmsa`, so the CMake probes for these tables compile only (no link).
- Found on the way: with clang 22 on MSA the and/or form of a vector select gave wrong `expm1`/`log1p`/rounding
  results; that backend calls `bsel.v` directly.
- Not done: any other test on these architectures; MSVC, Apple clang, NDK and FreeBSD builds of the new code (CI
  will be the first); the wasm, 32-bit Arm and 32-bit x86 CMake branches; a z14-only s390x CPU model.

### 2026-11-12 (GPU backend parity work; mock drivers only, no hardware)
Linux x86_64, clang 23, no GPU. Everything below ran against the mock drivers (`tests/mock_cuda.c` as CUDA and HIP,
`mock_opencl.c`, `mock_ze.c`); real hardware for HIP, OpenCL, Level Zero and Vulkan remains "never", and none of the
new CUDA code ran on the RTX 3090.
- Portable kernels for OpenCL and Level Zero (quantised matmul_t / gather / FFN gate-up, one-pass and split-over-keys
  attention, delta_rule / causal_conv, broadcast): `test_cuda_hip opencl|ze` now runs the same checks as CUDA
  (14/14 attention shapes, 24/24 quantised matmul cases, 20/20 fused FFN cases, state ops, 4/4 quantised gather).
  The OpenCL/Level Zero mocks execute the generated `kernels.cl` as C. Not compiled by any real OpenCL C compiler.
- Device mixture of experts (`run_moe`) on CUDA, HIP, OpenCL, Level Zero: 21/21 cases match the CPU sparse path on
  each mock. The two kernels compile to PTX (sm_70, sm_80) and AMDGPU code objects. `golden_moe_mla_{cuda,hip,opencl,ze}_mock`
  (Mixtral and DeepSeek-V2 tiny models, f32 expert weights) pass.
- Kernel program cache (OpenCL program binaries, Level Zero native binaries): `test_cuda_hip cache opencl|ze` 12/12.
- Determinism mode: `test_cuda_hip determinism <api>` 6/6 on the four mocks. The mocks run serially, so this shows
  the kernel selection, not a device's scheduling.
- Automatic device selection: `test_gpu` scoring test; `test_cuda_hip select <api>` 11/11 with `MOCK_DEVICES=2`.
- Fixed: plans destroyed on OpenCL, Level Zero and Vulkan left pins and resident constants behind (no
  `device_hooks::forget`), corrupting later plans (heap corruption / SIGBUS with `FORGE_DEVICE=opencl`). The OpenCL
  sub-buffer cache also ignored the view size. Regression: `test_cuda_hip stale <api>` 7/7 on the four mocks.
- Fixed: OpenCL and Level Zero `run_op` bound its inputs before acquiring the output; an output that partly
  overlapped the resident entry an input lived in (a reused plan arena) dropped that entry and left the input's
  handle freed. That was the SIGBUS in `forge_matmul_t` with `FORGE_DEVICE=opencl` (DeepSeek-V2 tiny golden; reported
  for Gemma / Phi-3 plans). `run_op` now settles the residency map first. `golden_moe_mla_opencl_mock` failed before
  and passes now; the Gemma / Phi-3 goldens of the reporting branch were not run here (not merged yet).
- Vulkan on lavapipe: `test_vulkan` still passes (the new `forget` hook); no new Vulkan kernels.
- Not done in this round: Vulkan (GLSL) and Metal (MSL) ports of the kernels above, eviction during captures with
  buffer generations, the missing `gpu::prims`, `pipe:auto` partitioning, RPC remote weight residency.

### 2026-11-05 (branch `cuda-mma-prefill`, RTX 3090)
- CUDA prefill and decode speedups; numbers in [CUDA performance](#cuda-performance-on-rtx-3090-2026-11-05).
  A second PTX image for `sm_80` (tried first, `sm_70` fallback) carries int8 tensor-core prefill matmul for
  q8_0/q4_k/q6_k and tiled prefill attention. Gated DeltaNet keeps its state in registers. Decode MMVQ uses a
  4-warp block per output column, with q6_k padded to 224-byte blocks and q8_0 regrouped on upload. CUDA graph
  capture is now on by default (`FORGE_GRAPHS=0` turns it off). Less host work per decode token: no device fetch
  before overwriting an input, no sync after uploads, logits read in place, plain greedy takes the argmax directly.
- Fixed: a destroyed plan left its pinned constants resident on the device, so a later buffer at the same host
  address could find a stale entry; `~plan` now forgets them and `forget` drops their pin records.
- Full Linux ctest passes on that machine, including the CUDA/HIP mock goldens and the tiny Qwen3.5 golden
  against transformers. `test_cuda_hip` adds prefill attention shapes, three delta_rule shapes, an MMA-aware
  quantised matmul reference and an `sm_80` image check.

### 2026-11-04
- PR #53 merged as ccc0a7d. Its last PR run (41802863931) passed all 11 CI jobs: linux-gcc, linux-clang,
  linux-asan, linux-tsan, macos, ios, freebsd, android, windows-msvc, windows-mingw, shells.
- CI now covers native macOS, iOS (build), Windows MSVC and MinGW, and FreeBSD in a VM. This supersedes
  older notes that macOS was only cross-linked, iOS never built, and the Windows event log sink and
  `os/dynlib.hpp` Windows/Apple code never compiled: `test_log` (including `eventlog_sink` on Windows) and
  `test_os_dynlib` pass on macOS, both Windows jobs and FreeBSD.
- CUDA on an RTX 3090 under Windows 11 (results above).
- Fixed: split-K decode attention launched its combine kernel with dangling argument pointers (7fe8a83).
- Fixed: on Windows, `find_executable` let an extensionless file on PATH shadow `cmd.exe` (1dea828); this was the
  `windows-mingw` `limit_counts.fs` failure.
- Fixed: Windows `ProcessPool` workers re-ran the whole test suite, an unbounded chain of test processes
  (d6377d3).
- Also: `strtod` fallback where libc++ lacks floating `from_chars` (FreeBSD 14, f6f1d11); CRLF-tolerant emit text
  goldens (8a90508); Metal build fixes after the module rename (f17c83d, d7d423d).
- Modules renamed: lane→simd, grid→gpu, flow→graph, pack→modelfile, mind→inference, balance→loadbalancer,
  spec→ops. `FORGE_LANE_TARGET` still works as an alias of `FORGE_SIMD_TARGET`.
- Added with tests: async and Windows event log sinks, `FORGE_LOG_MIN_LEVEL` (`test_log`, `log_min_level_cmake`);
  `limit {}` blocks and `allocs`/`handles`/`procs` limits; capability escape tests, which closed three gaps
  (`std::getenv` ignored `env`; pre-opened files and processes were usable in a narrower sandbox); RPC mTLS,
  channel binding and `allow=` permissions (`test_rpc`, `test_rpc_tls`; clients built before this cannot
  authenticate to a TLS server); `forge::remote_call`; GPU memory pools with LRU eviction
  (`test_gpu_memory_*_mock`); prefix KV spill (bit-identical logits); `os/dynlib.hpp`.
- PR #50: merges to main are gated on all CI checks.

### 2026-11-01 to 2026-11-03 (CUDA decode work, RTX 3090)
PRs #33–#48 moved CUDA decode onto the device. Decode speed reported in the PR titles: 1.2 → 4.5 tok/s at
p=1024 (#33), 4.8 → 17 (#35), 17 → 24 (#36), 24 → 29 with CUDA graphs (#38), split-K decode attention +29% (#48).
PR #49 (whole-run decode replay) gave no gain; PR #52 made it opt-in.

### 2026-10-31
- PR #18: CUDA on an RTX 3090 under WSL2, clang 23.1.4: ctest 104/104, CUDA tests on the device. Fixes: graph
  capture uses relaxed mode (global mode fails with error 900 on a real driver).
- PR #17: first GitHub Actions workflow.
- qemu-user, zig 0.17 musl static builds:
  - aarch64 (`-cpu max,sve256=on`) and riscv64 (`-cpu rv64,v=true,vlen=256`): `ctest -E '^(xos_|mobile_)'` 50/54
    with 6 GPU skips on both. All simd, SVE/RVV VL and fuzz tests pass. The 4 failures come from qemu-user (no
    dlopen in static builds, no binfmt for self-exec, `/proc/self` RSS 0) in `test_os`, `test_os_ext`,
    `test_os_process`, `test_script` (plugin).
  - Fixed: riscv64 has no `__NR_renameat` (cff8244).
- `ctest -L cross` (zig compile + link for FreeBSD, macOS, MinGW, musl): 39/39 after compiling each file with `-c`
  (ee0100c).
- QEMU TCG VMs: FreeBSD 14.5 amd64, all real tests pass after fixes; wine 11.20 (llvm-mingw build) every test
  passes, GPU tests skip. Bugs fixed then include kqueue missing in-place writes, a safetensors header-length
  overflow, Winsock `shutdown()` not waking `accept()`, and several cmd emitter errors.
- Shell emit goldens run on: sh, bash, zsh, dash, ash/busybox, yash, fish, tcsh, csh, nu, xonsh, elvish, pwsh,
  powershell (via pwsh), ksh, osh, ysh, hush, mksh, oksh, rc, es, and cmd under wine. Not tested: posh.
- `forge_mobile`: `test_mobile` 13/13 on x86_64 Linux (simulator, fake sysfs). Windows, FreeBSD, macOS, musl and
  Android backends compile-checked only; `apple.mm` not compiled then (now built in the `macos` job).

### 2026-10-30
- Linux dev builds: ctest 73/73 (zig), 76/76 (g++), 84/84 (clang 18, after scripting milestones 6–7). ASan and
  TSan runs of `test_script` and `test_rpc` clean.
- Android x86_64 emulator (API 34, TCG): 23 pass, 2 skip, 1 fail (`test_os_ext` socket binds; fix 7b7602c
  restored after 21c58e0 reverted it). The Android rerun after the fix was not done.
- wine: 32 pass, `test_vulkan` skips; one `test_os` timing flake fixed.
- `forge::os` extension, 16 features (`test_os_ext`): all pass on Linux and the Android emulator; under wine all
  pass except sandbox exec (wine ignores job limits).
- `test_vulkan` on SwiftShader: fixed a real misaligned-sub-buffer bug; lavapipe and the emulator now batch 571
  dispatches into 63 submits. `test_vulkan_align256` covers it.

### 2026-10-29 and earlier
- 8-core Xeon (AVX-512, AMX), no GPU. Vulkan on lavapipe: every kernel within 3e-7 relative of the reference;
  both goldens end-to-end on Vulkan.
- SVE at VL 128/256/512 and RVV at VLEN 128–1024 under qemu: simd tests pass at every length.
- AMX intermittent mismatches traced to the VM losing tile registers across context switches
  (`tools/amx_state_probe.cpp`); after fences and retries, 0/300 native and 0/10 SDE failures.
- SME f32 outer products pass under qemu at SVL 128–2048.
- KV-cache quantisation on the tiny model: max logit error vs f32 KV 6.8e-3 (int8), 3.1e-2 (fp8); Vulkan int8
  matches CPU int8 to 4e-7.
- Windows under wine (pure win64 prefix): 19 tests, 5 examples, 8 fuzzers pass. Real Windows bugs fixed then:
  IOCP on sync handles, unfused mingw `fmaf` (now `forge::fmaf_exact`), `std::rename` not replacing, mappings
  leaked when constructors threw.
- Android emulator: `test_os`, `test_loadbalancer`, `test_os_ext` pass; `test_inference` too heavy for TCG.
  Kotlin sources compiled with kotlinc 2.0.21.
- Text and locale: `test_text` 115/115, `test_locale` 223/223, UCD conformance 100% on 8 suites.
- `test_vulkan` parallel-run flake (shared `/tmp` paths) fixed with per-pid temp names.
