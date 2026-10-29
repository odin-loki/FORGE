# FORGE implementation status

Built and tested on an 8-core Xeon (AVX-512, AMX) with no GPU. The Vulkan backend runs on Mesa lavapipe (a software Vulkan device). aarch64 and riscv64 are cross-built with zig and tested under qemu-user.

## Test matrix (ctest)
| Build | Result |
|---|---|
| x86_64 dev | 59/59 with `-LE slow` (hardware tests skip with code 77: CUDA/HIP/OpenCL/Level Zero/Metal devices, adb, iOS). With `ANDROID_NDK_HOME` set, `mobile_android_arm64-v8a` and `mobile_android_x86_64` (label `slow`) pass: NDK r27c cross-build plus a check that all 12 JNI exports are present. This includes 8 fuzz targets, the Vulkan kernels on lavapipe, both goldens end-to-end on Vulkan (`FORGE_DEVICE=vulkan`), and the CUDA/HIP compile checks. |
| aarch64 SVE build (`build/a64sve`), latest run | 31/31 without the fuzz label (3 skip: Vulkan/CUDA/HIP). |
| riscv64 build (`build/rv64`), latest run | 27/27 without the fuzz label (3 skip). |
| aarch64, qemu `-cpu max` (VL 512, so the NEON path runs) | 13/13 without the fuzz label. `test_vulkan` skips because the build is static. |
| aarch64 SVE, qemu `-cpu max,sveN=on`, N = 128/256/512 (1024 requested; qemu caps `max` at 512) | 13/13 at each length without the fuzz label. The `sve` lane is active at each VL. |
| riscv64 RVV, qemu `-cpu rv64,v=true,vlen=N`, N = 128/256/512/1024 | 13/13 at each length without the fuzz label. The `rvv` lane is active at each VLEN. |

## Done
- **core / spec**: dtypes, shapes, a declarative op spec (`spec/ops/*.fspec`) with codegen, and a float64 oracle for every op.
- **lane**: target-tag SIMD with native lowerings for scalar, AVX2, AVX-512, NEON, **SVE** and **RVV** at any vector length from 128 to 1024 bits. SVE lengths that are not a power of two, or are above 1024, are lowered with PR_SVE_SET_VL at load time.
  - SVE and RVV are built once per length (128/256/512/1024) as fixed-length tables. At startup, dispatch reads the hardware VL (`rdvl` / `vlenb`) and picks the matching table.
  - Math functions are within 3 ulp and agree across lowerings. Also included: runtime dispatch (HWCAP/cpuid), an ULP harness, and GEMM.
- **grid**:
  - CPU backend with graph replay, backend probing, prims, GEMM and Philox RNG.
  - **GPU kernels are generated from the spec** (`spec/gen/gpu_gen.py`): 33 kernels covering elementwise unary and binary ops, scale, silu_mul, softmax, rms_norm and matmul_t, rope, gather, attention and slice.
    - GLSL is compiled to SPIR-V with glslangValidator, validated with spirv-val, and embedded in the binary.
    - The **Vulkan backend** (`grid/vulkan.hpp`) dlopens libvulkan, so there is no link dependency. Every kernel is checked against the spec reference (`test_vulkan`, max relative error 3e-7).
    - The same source compiles to **CUDA PTX (sm_80) and HIP AMDGCN (gfx90a)** with clang, without an SDK (`gpu_compile_cuda`, `gpu_compile_hip`).
- **flow**: IR, verifier, CSE/fold/fuse/DCE passes, memory planner, quant codecs: f16, bf16, fp8 e4m3/e5m2, q8_0, q4_k, groupwise q4_g32/q8_g32 (AWQ/GPTQ style), OCP MX (mxfp8, mxfp6, mxfp4, mxint8), nvfp4 and ternary tq2 (BitNet absmean). All are usable via `forge import --codec`.
- **core dtypes**: FP4 E2M1 / FP6 E2M3 / E3M2 element formats (checked exhaustively), stochastic rounding (checked for bias), complex c32/c64.
- **lane extended libm**: exp2, expm1, log2/10/1p, sqrt, rsqrt, cbrt, sin/cos/tan, asin/acos/atan/atan2, sinh/cosh, erf/erfc, gelu_erf, softplus, lgamma, pow, hypot, logsumexp. Written on portable vec ops so every lowering gets them; worst error is 4 ulp (`math_extended_ulp`).
- **Vulkan residency**: tensors stay on the device across ops; submits are batched (571 dispatches in 63 submits for the e2e golden).
- **CUDA/HIP host runtime**: dlopens libcuda/libamdhip64 at run time and loads the embedded PTX/HSACO. Tested against a mock driver library; the hardware tests skip when there is no device.
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
- **support**: arena allocator, linalg (LU/Cholesky/QR/CSR/CG), FFT (radix-2 + Bluestein, rfft, STFT, mel), vision (resize/YUV/NMS), profiler zones with Chrome-trace export, and grid prims (segmented scan/reduce, bitonic sort, hash map).
- **mind, newer**: typical-p and Mirostat v2 samplers; JSON-Schema to GBNF (`response_format` in forge-serve); LoRA hot-swap (in memory or from an `.fpk` with `*.lora_a/b`); C ABI 3 (`forge_generate_ex2` + size-prefixed `forge_sampling_ext`), also exposed in the Python bindings.
- **Session kinds**: `diffusion_loop` (`mind/diffusion.hpp`): DDIM/DDPM/flow-Euler; epsilon/v/x0 prediction; classifier-free guidance; step plans; a latent cache that resumes bit-identically; Philox noise. `streaming` (`mind/streaming.hpp`): SPSC ring buffer, window/hop framing, and a latency budget that drops the oldest frames.
- **flow**: reverse-mode autodiff (`flow/autodiff.hpp`), checked by finite differences. Blueprint composite ops (`op name(args){...}`).
- **pack security**: `.fpk` encryption with XChaCha20-Poly1305 (`forge encrypt`; key from `FORGE_MODEL_KEY` or `FORGE_MODEL_KEY_FILE`). Without the key, metadata still opens.
- **Vulkan pipeline cache** on disk, keyed by vendor/device/driver/pipelineCacheUUID and a SPIR-V hash (`FORGE_CACHE_DIR`; `FORGE_VK_PIPELINE_CACHE=0` turns it off).
- **Supply chain**: SPDX 2.3 SBOM (`tools/sbom.py`, deterministic, tested). Release builds are bit-identical (`FORGE_REPRODUCIBLE`, `ci/repro_check.sh`, which compares two builds in different directories). NOTICE lists Monocypher.
- **support, more**: Threefry4x32-20 (Random123 known-answer tests); linalg sym_eig/SVD (Jacobi) and GMRES; TLSF + pool allocators with leak reports (`mem/tlsf.hpp`); `FORGE_REFLECT` JSON reflection; audio resample/biquad; vision roi_align.
- **Async I/O** (`io/aio.hpp`): io_uring through raw syscalls (no liburing), falling back to a pread thread pool; short reads are resubmitted (`test_aio` runs both backends).
- **mind, serving**: prompt-lookup (n-gram) speculation; presence/frequency penalties (also in forge-serve); batch request priority and `cancel(id)`.
- **Docs/perf**: generated API reference (`tools/gen_docs.py`, `docs/reference.md`, checked by ctest `docs_up_to_date`); perf baselines (`tools/perf_baseline.py`, `perf/baseline-x86_64.json`).
- **KV-cache quantisation + host offload** (`flow/kv_codec.hpp`): int8 (symmetric) and fp8 E4M3 per (row, head) with an f32 scale; `state_append` quantises, the CPU and Vulkan attention kernels decode on the fly. Select with `session_options.kv_codec` or `FORGE_KV_CODEC`. Measured on the tiny model: max logit error vs f32 KV is 6.8e-3 (int8) and 3.1e-2 (fp8) relative to max |logit|; Vulkan int8 matches CPU int8 to 4e-7. `kv_offload` / `FORGE_KV_OFFLOAD=1` drops device KV after each attention (peak device bytes 153 KB vs 219 KB). Tested by `test_kvquant`.
- **GPU runtimes sharing one contract** (`grid/cuda.hpp`): run_op, residency, graph capture/replay, peer copy, multiple devices (`cuda[:N]`, `hip[:N]`, `opencl[:N]`, `ze[:N]`, `metal[:N]`), and one kernel-argument packer.
  - **OpenCL** (`grid/opencl.hpp`): kernels.cl is generated from the CUDA bodies.
  - **Level Zero** (`grid/levelzero.hpp`): vendored MIT `ze_api.h`; SPIR-V is built when llvm-spirv is available.
  - **Metal** (`grid/metal.hpp` + `src/metal.mm`): MSL is generated; unified memory.
  - **Mock-tested only.** For OpenCL and Level Zero, mock drivers run the real generated OpenCL C as C, and all 31 lowerings pass the oracle plus golden_tiny_llama. The MSL passes a host syntax check. **None of these have run on real hardware.**
- **Android** (*cross-compiled, never run on a device*):
  - NDK CMake build for arm64-v8a (NEON + Vulkan) and x86_64. The CUDA/HIP image builds are disabled on mobile. Portable entropy (`pack/entropy.hpp`, arc4random_buf on Bionic/Apple) because API 26 lacks getentropy.
  - JNI bridge at `bindings/android/jni/forge_jni.c` (CMake `-DFORGE_JNI=ON` builds `libforge_jni.so`).
  - Kotlin `dev.forge.ForgeModel`, a Gradle library module `:forge` (externalNativeBuild on the root CMakeLists), and an example app `:example`.
  - The Kotlin and Gradle files have **not been compiled**: there is no JDK on the build box.
- **iOS / macOS** (*written without a Mac; never compiled*):
  - `cmake/toolchains/ios.cmake` (device, or simulator with `-DFORGE_IOS_SIMULATOR=ON`) and `tools/build_xcframework.sh`, which produces FORGE.xcframework.
  - SwiftPM package `bindings/apple`: a binaryTarget, a `CForge` module map over forge.h, and a `Forge` Swift API (ForgeModel/Sampling, streaming callback).
  - XCTest (skips without `FORGE_TEST_FPK`) and a SwiftUI example. Metal is the default device.
- **Goldens**: tiny Llama and MoE/MLA, checked against NumPy float64 references.

- **Multi-GPU pipeline split + socket RPC device** (`flow/rpc.hpp`, `FORGE_DEVICE=pipe:a*w,b` / `rpc:host:port`, `forge rpc-serve`): contiguous stage split balanced by cost; loopback RPC tested (test_rpc). Only CPU/lavapipe stages exercised — no real multi-GPU. RPC has no auth; binds 127.0.0.1 by default.
- **AMX / SME tiles** (`lane/tile.hpp`): AMX bf16/int8 runs natively and under Intel SDE (`tile_amx_sde_{spr,gnr,icl}`); SME f32 outer products pass under qemu at SVL 128–2048 (`tile_sme_svl*`, weak SME ABI routines for runtimes lacking them). **Open bug:** AMX bf16/s8 gives wrong blocks intermittently (~1.5% native, most SDE runs) after interleaved calls; not yet root-caused.
- **Graphics interop** (`grid/interop.hpp`): Vulkan external memory export/import round-trip passes on lavapipe (also on the Android emulator); external semaphores optional (lavapipe lacks them); CUDA import code untested (no GPU).
- **Hardware video decode** (`vision/video.hpp`): Annex-B/SPS parsing tested; NVDEC (built only with nvcuvid.h), MediaCodec, VideoToolbox written but untested; VA-API is probe-only.
- **Windows**: mingw-w64 cross-build compiles and links all targets (`--preset windows-mingw`; Winsock via `core/net.hpp`, dlopen shim, ReadFile pread). **Not run**: Kron4ek Wine 11.20 crashes/cannot init a prefix on this VM. macOS: osxcross toolchain file only (no SDK on the box) — not built.
- **App integrations** (`examples/integrations`, `mind/plugin.hpp` custom-session registry): Cypha, seatbelt one_shot, NeuralVFX streaming, FUSE runtime, CameraDetector examples run as ctest `-L integration`.
- **v1.0 release tooling**: `tools/release.py` (bump/changelog/build/verify/tag), install rules + CPack (TGZ/DEB/ZIP), platform presets and a `release` workflow preset, `docs/RELEASING.md`. The full `build` flow has not been run end to end yet.
- **Android emulator pass** (`tests/android_emulator.sh`, x86_64 AVD API 34 under TCG — KVM is not usable in this VM): native tests run on device; Kotlin sources compile with kotlinc 2.0.21 + JDK 17.

## Remaining
- Not testable on this machine: real CUDA/HIP GPUs, Level Zero, Metal, Windows/macOS/mobile, multi-GPU/NCCL-style collectives, and SDE-validated AMX/SME tiles.
- Vulkan runs whole sessions (`FORGE_DEVICE=vulkan` or `session::set_device`). The vk.* lowerings have priority over the CPU ones and are each oracle-checked; any op without a GPU kernel falls back to the CPU.
- CUDA/HIP/OpenCL/Level Zero/Metal have not been run on real hardware; only the mock drivers have run them.
- Android: has not run on a device, and the Kotlin/Gradle files have not been compiled. iOS/Swift: not compiled at all.
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

### Android (NDK r26+)
    export ANDROID_NDK_HOME=/path/to/android-ndk-r27c
    cmake -S . -B build/android-arm64 -G Ninja -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DCMAKE_BUILD_TYPE=Release -DFORGE_JNI=ON
    ninja -C build/android-arm64 forge_jni      # libforge.so + libforge_jni.so; use ANDROID_ABI=x86_64 for the emulator
    cd bindings/android && gradle :example:assembleDebug     # AAR + APK (Gradle invokes the same CMake per ABI)
    gradle :forge:connectedAndroidTest                       # on a device; set FORGE_TEST_FPK for the model test
On the device, `FORGE_DEVICE=vulkan` selects the Vulkan backend; otherwise the NEON CPU lanes run. The ctest entries are `mobile_android_<abi>` and `mobile_android_device`.

### iOS / macOS (Xcode 15+)
    tools/build_xcframework.sh                 # iOS device + simulator + macOS -> bindings/apple/FORGE.xcframework
    cd bindings/apple && swift test            # XCTest; set FORGE_TEST_FPK=/path/model.fpk
Add `bindings/apple` as a local Swift package in Xcode and `import Forge` (see `Example/ForgeExample`). ctest `mobile_ios` runs both steps on macOS and skips elsewhere.

## Test flake (test_vulkan under parallel load)

Concurrent runs shared fixed temp paths in /tmp (`forge_test_vk_e2e.fpk` and others), which caused SIGBUS and bad-magic errors. Temp files are now named per pid in test_vulkan, test_mind and test_pack. After the fix, 120 concurrent runs and `ctest -j8 --repeat until-fail:4` under full CPU load all passed.
