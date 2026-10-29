# FORGE implementation status

Built and tested on an 8-core Xeon (AVX-512, AMX) with no GPU. The Vulkan backend runs on Mesa lavapipe (a software Vulkan device). aarch64 and riscv64 are cross-built with zig and tested under qemu-user.

## Test matrix (ctest)
| Build | Result |
|---|---|
| x86_64 dev | 73/73 with `-LE "slow|emu"` (2026-10-30; earlier rounds 59/59 with `-LE slow`) (hardware tests skip with code 77: CUDA/HIP/OpenCL/Level Zero/Metal devices, adb, iOS). With `ANDROID_NDK_HOME` set, `mobile_android_arm64-v8a` and `mobile_android_x86_64` (label `slow`) pass: NDK r27c cross-build plus a check that all 12 JNI exports are present. This includes 8 fuzz targets, the Vulkan kernels on lavapipe, both goldens end-to-end on Vulkan (`FORGE_DEVICE=vulkan`), and the CUDA/HIP compile checks. |
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
- Not testable on this machine: real CUDA/HIP GPUs, Level Zero, Metal, native Windows/macOS/iOS (Windows runs under wine, Android under the x86_64 emulator), multi-GPU/NCCL-style collectives, and SDE-validated AMX/SME tiles.
- Vulkan runs whole sessions (`FORGE_DEVICE=vulkan` or `session::set_device`). The vk.* lowerings have priority over the CPU ones and are each oracle-checked; any op without a GPU kernel falls back to the CPU.
- CUDA/HIP/OpenCL/Level Zero/Metal have not been run on real hardware; only the mock drivers have run them.
- Android: native tests run on the x86_64 emulator (see below), not on a physical device; the Kotlin/Gradle files have not been compiled. iOS/Swift: not compiled at all.
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

### Android emulator (low-memory settings used on the shared box)
    emulator -avd forge -no-window -no-audio -no-snapshot -no-boot-anim -gpu swiftshader_indirect \
      -memory 2048 -cores 2 -no-metrics -accel off -feature -Netsim     # TCG boot: 4–12 min
    adb push build/android-x64/test_os_ext $NDK/.../x86_64-linux-android/libc++_shared.so /data/local/tmp/forge/
    adb shell 'cd /data/local/tmp/forge && FORGE_THREADS=2 LD_LIBRARY_PATH=. TMPDIR=. ./test_os_ext'
Run one test at a time with a timeout; skip test_mind (OOMs the emulator). `tests/android_emulator.sh` automates this.

### Windows under wine64
    cmake -S . -B build/win64 -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/mingw-w64.cmake; ninja -C build/win64 -j4
    WINEPREFIX=~/.wine-forge64 tests/wine_run.sh build/win64     # pure win64 prefix; 77 = skip
Do not leave `FORGE_OS_EXT_CHILD` exported in the shell — test_os_ext children use it.

### zig cross compile+link (FreeBSD, macOS, MinGW, musl)
`ctest -L cross` runs `zig c++ -target <x86_64-freebsd|aarch64-macos|x86_64-macos|x86_64-windows-gnu|aarch64-linux-musl>`
on test_os, test_balance and test_os_ext. Use `ninja -j4` or lower on shared machines.

### iOS / macOS (Xcode 15+)
    tools/build_xcframework.sh                 # iOS device + simulator + macOS -> bindings/apple/FORGE.xcframework
    cd bindings/apple && swift test            # XCTest; set FORGE_TEST_FPK=/path/model.fpk
Add `bindings/apple` as a local Swift package in Xcode and `import Forge` (see `Example/ForgeExample`). ctest `mobile_ios` runs both steps on macOS and skips elsewhere.

## Test flake (test_vulkan under parallel load)

Concurrent runs shared fixed temp paths in /tmp (`forge_test_vk_e2e.fpk` and others), which caused SIGBUS and bad-magic errors. Temp files are now named per pid in test_vulkan, test_mind and test_pack. After the fix, 120 concurrent runs and `ctest -j8 --repeat until-fail:4` under full CPU load all passed.

### AMX intermittent-failure root cause (resolved)
The intermittent bf16/s8 AMX mismatches are a host problem: this VM loses tile registers across
context switches. `tools/amx_state_probe.cpp` loads a tile, spins, stores it back, and sees 42-329 of 3000
round-trips corrupted, with no FORGE code involved. SDE shows the same failures because on an AMX host it
runs tile instructions natively. Changes: explicit compiler fences around the tile asm (GCC's intrinsics have
no memory clobber), reused thread_local pack buffers, and test_tile retries before failing. 0/300 native and
0/10 SDE failures after the change.

## OS wrapper (`forge::os`) + load balancer (`forge::balance`) — 2026-10-29

- `os/include/forge/os/os.hpp`: one header that wraps files and the filesystem, mmap, processes, threads and sync, time, sockets, polling (epoll, kqueue, or WSAPoll), IOCP `completion_port` (emulated off Windows), dl, env, signals, memory, sysinfo, paths and entropy. It targets Linux, Android, Windows, macOS, iOS (`run` unsupported there) and FreeBSD. `core/mapped`, `core/net` and `pack/entropy` now go through it. Not yet converted: dl.hpp, rpc.hpp, aio.hpp, video.cpp, interop.hpp.
- `balance/`: strategies are RR, smooth-weighted, least-conn, EWMA, consistent hash (vnodes + bounded load), P2C and prefix/KV affinity with load spill. Also includes health checks, a circuit breaker, retries, and a backpressure queue. `proxy.hpp` provides the L4/L7 proxy behind `forge-serve proxy`.
- Verification:
  - Linux native: test_os (12 tests) and test_balance (13 tests) pass, as does the full dev ctest.
  - Android x86_64 emulator (TCG): test_os and test_balance pass; the l7 stickiness check is occasionally flaky under TCG.
  - zig cross compile+link (`xos_*` ctests): x86_64-freebsd, aarch64/x86_64-macos, x86_64-windows-gnu and aarch64-linux-musl all OK.
- Android full-suite run: everything passes except test_vulkan. SwiftShader's sin/cos come in at about 1.9e-4 abs error, which the Vulkan spec allows (bound 2^-11). Fix: vk lowerings for sin/cos/rope now carry `lowering::tol = 5e-4`. test_mind and example_seatbelt hit `adb device offline` because the emulator ran out of memory; that is not a test failure.
- Windows under wine (`tests/wine_run.sh`): the earlier `kernel32.dll c0000135` came from a wow64 prefix. A pure 64-bit prefix (`WINEARCH=win64`, `/usr/lib/wine/wine64 wineboot -i`) works. All 19 win64 tests, 5 examples and 8 fuzzers pass; test_vulkan skips (no ICD). Real Windows bugs this run exposed and fixed:
  - IOCP: `completion_port` now reopens the file handle with `FILE_FLAG_OVERLAPPED` (`ReOpenFile`); sync handles never completed.
  - msvcrt/mingw `fmaf` is unfused (2.4M of 10M random inputs wrong), which broke the scalar lane math (asin/erfc 1 ulp over). New `forge::fmaf_exact` (TwoSum + round-to-odd, or the hardware FMA) is used by the lane scalar backend and the spec reference.
  - `std::rename` does not replace on Windows, so fpk write/sign silently kept the old file. They now use `os::rename` (MoveFileEx REPLACE_EXISTING) and raise on failure.
  - `pack::reader`, `safetensors` and `gguf_file` leaked their mapping when the constructor threw (bad key, corrupt or fuzzed input), which held files open on Windows.
  - Tests: autodiff data is now portable Box-Muller (std::normal_distribution differs between libstdc++ and libc++). The paged-KV capacity check now allows for EOS.
- All platform code now goes through forge::os: rpc (os::socket), aio (os::file; CRT fds kept on Windows), video/interop (os::library), and the dl.hpp shim (UTF-8 LoadLibraryW).
- Not done:
  - FreeBSD qemu VM: not run (memory limits on the shared box); only cross-linked.
  - iOS: no SDK (zig lacks iOS libc headers, and there is no osxcross/darling), so iOS paths are unverified beyond review.

### Android emulator re-run (FORGE_THREADS cap, one test at a time) — 2026-10-29
- 21 pass (aio, balance, dtypes, flow, grid, interop, kvquant, lane, lane_raw, os, pack, rpc, sessions, support, support2, tile, video, plus 5 examples); test_gpu skips.
- test_vulkan: the 33 kernel oracles and vk.rope now pass with the relative 2^-11 bound. Still failing on SwiftShader: "residency: dispatches are not batched" and the e2e tiny llama step, which only fails on that batching flag (logits err 8.8e-4, greedy tokens identical). Not investigated yet.
- test_mind: exceeds 10–20 min under TCG with 2 threads and has taken the emulator down from memory twice. Treated as too heavy for this box; it passes on Linux and wine.
- `tests/android_emulator.sh` accepts FORGE_TEST_TIMEOUT and FORGE_THREADS, and retries when adb drops.

## test_vulkan batching on SwiftShader — fixed
Root cause was a real backend bug. SwiftShader requires a 256-byte minStorageBufferOffsetAlignment. Sub-range views of a resident buffer at misaligned offsets made `acquire` drop the parent buffer, which forced a submit and a host sync (143 submits/571 dispatches). Such views are now staged through scratch buffers with device-side copies, including write-back of outputs. `FORGE_VK_SSBO_ALIGN` emulates stricter drivers, and the `test_vulkan_align256` ctest covers this on lavapipe. Lavapipe (with and without align=256) and the Android emulator now give 63 submits/571 dispatches and test_vulkan passes. Linux dev suite: all pass.

## forge::os extension (16 features) — 2026-10-30
Code: `os/include/forge/os/*.hpp` (guide: [`docs/os.md`](os.md)); test: `tests/test_os_ext.cpp` (one TEST per feature).
Wiring: forge-serve `--watch` hot reload, `service::ready`, P-core pinning, `--numa-node`, memory pressure → prefix KV
eviction (`batch_scheduler::request_kv_eviction`), `FORGE_HUGEPAGES=1` on weight mappings (see [`docs/serve.md`](serve.md)).

| Feature | Linux (native) | Windows (wine64) | Android x86_64 emu | FreeBSD / macOS / musl |
|---|---|---|---|---|
| 1 NUMA | pass (1 node) | pass (affinity-mask fallback) | pass | link only |
| 2 Huge pages | pass (2M via THP; strict 1G ENOMEM) | pass (large pages need privilege → fallback) | pass | link only |
| 3 Async file I/O | pass (io_uring) | pass (IOCP) | pass | link only |
| 4 CPU topology P/E | pass (non-hybrid VM) | pass | pass | link only |
| 5 Power/thermal | pass (no battery) | pass | pass | link only |
| 6 Memory pressure | pass (PSI) | pass (resource notification) | pass | link only |
| 7 Shared memory | pass | pass | pass | link only |
| 8 GPU discovery | pass (no GPU visible) | pass (DXGI via wine) | pass | link only |
| 9 Unix sockets / pipes | pass | pass (named pipes) | pass | link only |
| 10 Crash + minidump | pass (.crash) | pass (.dmp) | pass | link only |
| 11 Sandboxing | pass (seccomp net/exec/write) | exec skipped (wine ignores job limits); net/fs unsupported by design | pass | link only |
| 12 Key storage | pass (libsecret + file) | pass (DPAPI + file) | pass | link only |
| 13 File watching | pass (inotify) | pass (ReadDirectoryChangesW) | pass | link only |
| 14 Timers/perf | pass (TSC fallback; perf_event_paranoid=2) | pass (cycle fallback) | pass | link only |
| 15 Background hooks | pass (not_required) | pass (execution state) | pass | link only |
| 16 Service integration | pass (no manager) | pass (not under SCM) | pass | link only |

forge-serve `--watch` verified live on Linux (two reloads, `/health` OK after); `--numa-node 0` verified. Linux dev ctest 73/73.
Android x86_64 emulator (API 34, TCG, 2 GB, FORGE_THREADS=2): test_os_ext 16/16, test_os and test_balance pass. Android-specific
results: background → unsupported without JNI hooks (graceful skip), perf counters unavailable (cycle fallback), GPU = emulator EGL device,
unix sockets must be abstract (SELinux) — `local_endpoint` now defaults to `@forge-<name>` on Linux/Android.
Also compiled and passed with g++ (Linux) in addition to the zig/clang dev build.
Untested: iOS (no SDK),
FreeBSD/macOS execution, hybrid P/E hardware, multi-node NUMA, batteries/thermal sensors, real GPUs, running under
systemd/SCM, Windows sandbox exec denial on real Windows, macOS kperf (root).

## Latest runs (2026-10-30)
- Linux, g++ dev build: ctest 76/76 pass, including the two new `doc_example_*` tests. The Python golden, signing and ONNX tests need `numpy`, `safetensors`, `onnx` and `cryptography` in the configured Python.
- Windows under wine, mingw build, `tests/wine_run.sh`: 32 pass, `test_vulkan` skipped, `test_os.exe` failed once inside the suite, at `cpu_time_ns() > 0` in `time_and_sleep`. Windows process times tick at about 15.6 ms, so the test's short loop could read 0. The test now busy-spins for 100 ms of wall time first, and passed 5 out of 5 reruns under wine.
- Android x86_64 emulator, full suite (`tests/android_emulator.sh native build/android-x64`, with `test_mind`): 23 pass, 2 skipped (`test_gpu`, `test_vulkan`), 1 failed. The failure was `test_os_ext` on socket-file binds. Commit 21c58e0 had accidentally reverted the abstract-socket fix; commit 7b7602c restores it. Linux `test_os_ext` passes again. **The Android rerun of `test_os_ext` after the restore is still pending.**
