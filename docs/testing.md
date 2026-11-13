# Testing

## Concepts
- Each `tests/test_*.cpp` builds into its own executable and ctest entry of the same name. It runs its `TEST(...)` cases (`tests/check.hpp`), prints `[ ok ]` or `[FAIL]` per case and a failure count, and exits 1 on any failure.
- **Exit code 77 means skipped** (missing hardware, runtime, SDK or data). Hardware tests declare `SKIP_RETURN_CODE 77`.
- `FORGE_TEST_ONLY=<substring>` runs only the cases whose name contains it (useful for locating a hang).
- **The oracle** (`verify/include/forge/verify/oracle.hpp`): every SIMD, GPU and quantised lowering is compared against the scalar reference in `ops/` within each op's declared tolerance. `test_graph`'s `oracle_matrix_all_ops` covers every registered lowering of every op; `test_cuda_hip` does the same for each GPU runtime.
- ctest runs most tests with the source tree as the working directory.

## Kinds of test

| Kind | Where | ctest names / label |
|------|-------|---------------------|
| Unit and differential | `tests/test_*.cpp` | `test_<name>` |
| Script golden tests | `tests/script/*.fs` with expected `*.out` | inside `test_script_golden`: each script runs on the tree-walker and the VM, and both outputs must match the `.out` file (`FORGE_GOLDEN_UPDATE=1` rewrites them) |
| Script conformance | `tests/script_conformance/*.cpp`: ordinary C++23 programs covering the documented subset (statements, operators, numbers, strings, containers, algorithms, formatting, functions and lambdas, classes, exceptions). `run.py` builds each with the host compiler and runs it, then runs it with `forge script --tree` and `forge script`; stdout and exit status must be identical | `script_conformance`; exits 77 (skipped) when the host compiler has no C++23 `<print>` (it tries `-std=c++23`, then `-stdlib=libc++`). `python3 tests/script_conformance/run.py --forge build/dev/forge --only strings` runs one program and prints the diff |
| Golden models | `verify/golden/tiny_llama.py`, `tiny_moe.py`, `tiny_qwen3.py`, `tiny_qwen3_5.py`, `quant_inplace.py`, `gguf_types.py`, `hf_quant.py` | `golden_tiny_llama`, `golden_moe_mla`, `golden_tiny_qwen3`, `golden_tiny_qwen3_5`, `golden_quant_inplace`, `golden_gguf_types` (every GGUF tensor type against gguf-py), `golden_hf_quant` (AWQ / GPTQ checkpoints from the script's own packer); `golden_tiny_llama_vulkan`, `golden_moe_mla_vulkan` and `golden_<name>_vulkan` for the other golden scripts (label `gpu`) when Vulkan is built. `golden_tiny_llama` writes `build/<dir>/golden/tiny_f32.fpk`, which later tests use |
| Golden models, other families | `verify/golden/tiny_gpt2.py`, `tiny_gemma.py`, `tiny_phi3.py`, `tiny_bert.py`, `tiny_vit.py`, `tiny_whisper.py`, `tiny_qwen2.py`, `tiny_rope.py`: HF and GGUF imports of tiny random models against float64 numpy references, f32 and q8_0 (cosine at least 0.999), and for the first four text through the file's own tokenizer against the reference tokenizers' ids | `golden_tiny_{gpt2,gemma,phi3,bert,vit,whisper}`, `golden_tiny_qwen2`, `golden_rope_scaling`; the same on mock devices as `golden_tiny_<family>_{cuda,opencl}_mock`, `golden_rope_scaling_{cuda,opencl}_mock` (label `gpu`). Disabled at configure time without numpy and safetensors |
| Real GPU runtimes | `tests/test_cuda_hip.cpp` | `test_gpu_{cuda,hip,opencl,ze,metal}`, `test_gpu_features_*`, label `gpu`; skip without a driver and device |
| Mock GPU drivers | `tests/mock_cuda.c` (also built as the HIP mock), `mock_opencl.c`, `mock_ze.c` stand in for the vendor libraries | `test_gpu_*_mock`, `test_gpu_features_*_mock`, `test_gpu_memory_*_mock`, `test_gpu_determinism_*_mock`, `test_gpu_kernel_cache_{opencl,ze}_mock`, `test_gpu_select_*_mock`, `test_gpu_stale_*_mock`, `golden_tiny_llama_{cuda,hip,opencl,ze}_mock`, `golden_moe_mla_*_mock`, `golden_tiny_llama_pipe_auto_cuda_mock`, label `gpu`. Not built when cross-compiling. The CUDA and HIP mock tests skip when configure found no clang with the NVPTX / AMDGPU backends (no kernel image) |
| Vulkan | `tests/test_vulkan.cpp` | `test_vulkan`, `test_vulkan_align256` (`FORGE_VK_SSBO_ALIGN=256`), label `gpu` |
| GPU compile checks | generated kernels compiled with clang | `gpu_compile_{cuda,hip,opencl,opencl_nofp64,metal_syntax,metal_host_syntax}`, and `gpu_compile_metal` with `xcrun`; label `gpu` |
| Script tests | `tests/*.py` | `modelfile_signing`, `onnx_import`, `python_bindings`, `serve_http`, `sbom_spdx` (`tools/sbom.py`: the SPDX 2.3 and the CycloneDX 1.5 output) |
| JNI bridge | `tests/jni/JniSmoke.java` on a host JVM with `-Xcheck:jni`: every native of `bindings/android/jni/forge_jni.c`, through `tests/jni/dev/forge/ForgeNative.java` (generated from the Kotlin externals by `tests/jni_decl_check.py --write`). `jni_decl_matches` compares the Kotlin, Java and C declarations. The Kotlin classes are not compiled by either | `jni_smoke` (Linux with a JDK; needs the llama and bert goldens), `jni_decl_matches`, `jni_syntax` |
| Server API | `tests/serve_api_test.py`: request fields (`n`, `logprobs`, `logit_bias`, stop ids, ...), end-of-turn stops, tool calling, keep-alive / pipelining / chunked bodies / limits, graceful shutdown, `--lora`. `tests/serve_auth_test.py`: API keys, rate limits, quotas, TLS and mutual TLS (certificates from the `openssl` command; the TLS part is skipped, with a message, when forge-serve has no OpenSSL) | `serve_api`, `serve_auth` (need `golden_tiny_llama` to have run) |
| Embedding server | `tests/serve_embeddings_test.py`: `/v1/embeddings` on the tiny BERT of the bert golden against `forge embed`, 400 on a generative model | `serve_embeddings` (needs `golden_tiny_bert` and `golden_tiny_llama` to have run) |
| C ABI | `tests/capi_smoke.c` | `capi_smoke` (not built when cross-compiling) |
| Installed package | `tests/install_consumer/` (install to a scratch prefix, then `find_package(forge)` from a C++ and a C program; see [building.md](building.md#the-install_consumer-test)) | `install_consumer`, label `install` (not for cross or sanitizer builds) |
| TLS-only RPC cases | `test_rpc` with `FORGE_TEST_ONLY=rpc_tls` | `test_rpc_tls` (skips in a build without OpenSSL, or without the `openssl` command) |
| Log strip | `tests/log_min_level.cmake` | `log_min_level_cmake` |
| Doc examples | `docs/examples/*` | `doc_example_*` |
| Integration examples | `examples/integrations/*.cpp` | `example_*`, label `integration` |
| Docs freshness | `tools/gen_docs.py --check` | `docs_up_to_date` (fails if `docs/reference.md` is stale) |
| Docs links | `tools/check_docs_links.py`: every relative link and anchor, and every backticked repository path, of `README.md`, the other top-level guides and `docs/*.md` resolves; every guide is linked from `docs/index.md` | `docs_links` |
| CI workflow copies | `.github/workflows/ci.yml`, `ci/github/ci.yml` and `ci/github-actions-ci.yml` are the same file | `ci_workflow_copies`, `ci_workflow_copy_flat` |
| Fuzz | `fuzz/fuzz_*.cpp`: file formats (aot, blueprint, bmp, chat_template, dbus, fpk, fsc, gguf, grammar, inflate, json, onnx, png, pnm, reflect, safetensors, script, script_aot, tokenizer) and network or request input (serve_http, proxy_http, rpc, json_schema, policy, signature) | `fuzz_*`, label `fuzz` |
| Lowering table | `tools/coverage_dump.cpp` walks the registries of the built code | `coverage_table`: the op × backend table `tools/gen_docs.py` reads from the sources for `docs/reference.md` must match them |
| JNI syntax | `bindings/android/jni/forge_jni.c` against a JDK's `jni.h` | `jni_syntax` |
| Mobile | `tests/mobile_build.sh` | `mobile_android_{arm64-v8a,x86_64}`, `mobile_android_device`, `mobile_ios`, labels `mobile;slow` |
| Cross compile/link | zig cross builds (`zig` on PATH, `ZIG_PATH`, or the `ziglang` Python wheel): every `tests/test_os*.cpp` and `test_loadbalancer` linked per target through `cmake/xos_link.cmake` (C sources compiled separately); `xos_headers_<target>` compiles each os header as its own translation unit | `xos_*`, labels `cross;slow` |
| Scalable vectors under qemu | `test_simd`, `test_tile` | `lane_sve_vl<N>`, `lane_rvv_vl<N>`, `tile_sme_svl<N>`, label `emu` (defined only in the qemu cross builds) |
| simd companion tables and element types | `test_simd_tables` (seq and int8 tables against scalar, bit for bit), `test_simd_tiera`, `test_simd_types` and the per-table `test_simd_types_<table>` (`vec_types.hpp`, exhaustive for 8-bit binary and 16-bit unary operations), `test_simd_std` / `test_simd_std_libcxx` (`interop.hpp`). They link only the simd library; the CMake target `forge_simd_tests` builds all of them | natively `types_<table>` (skipped when the CPU lacks the ISA), `tiera_sse2`, `tiera_sse4.2`, `tiera_avx2`; in the qemu cross builds `tables_*`, `tiera_*`, `types_*` per CPU model or vector length, label `emu` |
| Engine executor | `test_engine` | `engine_pool`, `engine_sched`; `sched_test_graph`, `sched_test_inference`, `sched_test_kvquant`, `sched_test_sessions`, `sched_test_aot` (the engine's tests under `FORGE_TASK_SCHEDULER=sched`, label `sched`) |
| Emulated AMX tiles | Intel SDE | `tile_amx_sde_*`, label `emu` |

The `gpu_compile_*`, `jni_syntax` and `mobile_*` tests are only defined when CMake finds a GPU-capable clang (`FORGE_GPU_CLANG`).

Tests whose Python modules are missing are marked "Not Run (Disabled)" at configure time rather than failing. The configure log lists them (`FORGE: python lacks one of {...}`).

## Running
```sh
ctest --test-dir build/dev -j4                      # everything configured
ctest --test-dir build/dev -j4 -LE "slow|emu"       # what CI runs: skips the long cross/mobile/emulator tests
ctest --test-dir build/dev -R doc_example           # the documentation examples
ctest --test-dir build/dev -L gpu                   # GPU runtimes, mocks and compile checks
ctest --test-dir build/dev -L fuzz                  # fuzzers (FORGE_FUZZ_ITERS inputs each)
./build/dev/test_os                                 # one suite directly
FORGE_TEST_ONLY=emit ./build/dev/test_script        # one group of cases
```

Wine, the Android emulator and qemu: see [building.md](building.md). Results per platform: [STATUS.md](STATUS.md).

## CI
`.github/workflows/ci.yml` runs on every pull request, on pushes to `main`, and on manual dispatch. All jobs set `FORGE_THREADS=2`. The table describes the workflow file as it is in the tree. **It has not run in this form**: the last CI run was PR #53 (2026-11-04), and `linux-fuzz`, the `emu` rows, `wasm-simd`, `cross-zig`, `perf`, `repro` and the whole-suite `linux-asan` / threaded `linux-tsan` jobs have never run on GitHub ([UNTESTED.md](UNTESTED.md#the-ci-workflow-itself)).

| Job | Runner | What it does |
|-----|--------|--------------|
| `linux-gcc`, `linux-clang` | ubuntu-24.04 | `dev` preset with g++-14 / clang-23 (from apt.llvm.org) and ccache; `ctest -LE "slow|emu"` |
| `linux-vulkan` | ubuntu-24.04 | g++-14 with glslang, Vulkan headers and Mesa lavapipe; builds `forge` and `test_vulkan`, checks `test_vulkan` reports an `llvmpipe` device (a skip fails the job), then `ctest -R vulkan` (both `test_vulkan` runs and the Vulkan goldens: llama, MoE, RoPE scaling, quantised, the template families) |
| `linux-asan` | ubuntu-24.04 | clang-23, `FORGE_SANITIZE=ON` (ASan + UBSan, Debug), leak detection on, `FORGE_FUZZ_ITERS=1000`: builds everything and runs `ctest -LE "slow|emu"` |
| `linux-tsan` | ubuntu-24.04 | clang-23, `FORGE_TSAN=ON`: builds and runs the suites that start threads (`test_pool`, `test_process_pool`, `test_loadbalancer`, `test_inference`, `test_sessions`, `test_memory`, `test_aio`, `test_log`, `test_rpc`, `test_os*`, `test_script*`, `test_modelfile`, `test_kvquant`, `test_graph`) |
| `linux-fuzz` | ubuntu-24.04 | clang-23, `FORGE_SANITIZE=ON` + `FORGE_LIBFUZZER=ON`: every `fuzz/fuzz_*.cpp` target for 60 s on its seed corpus; a finding fails the job and uploads the input (`fuzz-findings`) |
| `emu-aarch64-sve`, `emu-riscv64-rvv` | ubuntu-24.04 + qemu-user | Cross build with `g++-14-<triple>` and `cmake/toolchains/<arch>-qemu.cmake`; `ctest -L emu` (`lane_*_vl<N>`, `tile_sme_svl<N>`, `qi8_aarch64_*`), then `test_simd`, `test_tile`, `test_dtypes`, `test_graph`, `test_kvquant`, `test_support`, `test_util` under each `-cpu` model of the row (SVE 256 and 512 bits; RVV VLEN 256). One matrix row per target |
| `emu-loongarch64-lasx`, `emu-ppc64le-vsx`, `emu-s390x-vxe`, `emu-mips64el-msa` | ubuntu-24.04 + qemu-user | The same job for the fixed-width tables: `cmake/toolchains/<arch>-qemu.cmake` sets the one CPU model (`la464`, `power9`, `max`, `Loongson-3A4000`); `lane_lasx` and `lane_lsx`, `lane_vsx`, `lane_vxe`, `lane_msa` force each table and require it to be the active one, then the same SIMD-facing suites run |
| `emu-armv7-neon` | ubuntu-24.04 + qemu-user | 32-bit Arm (`armv7-neon-qemu.cmake`, `qemu-arm -cpu cortex-a7`): `lane_neon`, `tables_neon`, `tiera_neon`, `types_neon32` and the same four on a Cortex-A15 (`…_a15`), then the SIMD-facing suites. Does not fail the workflow: only the simd tests have been built for 32-bit Arm (locally), and this job has never run on GitHub |
| `wasm-simd` | ubuntu-24.04 | zig from the `ziglang` Python package, node 22: `tests/wasm_simd.sh` (test_simd with the scalar and SIMD128 tables on wasm32-wasi), then again with `WASM_SIMD=0` |
| `cross-zig` | ubuntu-24.04 | zig from the `ziglang` Python package; `ctest -L cross` (the `xos_*` compile and link checks) |
| `perf` | ubuntu-24.04 | Release build of `forge`; `tools/perf_baseline.py --check --tolerance 0.75 --allow-missing` against `perf/baseline-x86_64.json`. Never blocks (`continue-on-error`): the table is in the job summary, the numbers in the `perf-report` artefact |
| `repro` | ubuntu-24.04 | `ci/repro_check.sh`: two clean Release builds in directories of different path length; `forge`, `forge-serve` and `libforge.so` must be bit-identical |
| `shells` | ubuntu-24.04 | Installs many shells; runs `FORGE_TEST_ONLY=emit test_script` |
| `macos` | macos-latest | `dev` preset; `ctest -LE "slow|emu"` |
| `ios` | macos-latest | Xcode device build of `forge_c`, `forge_mobile`, `forge_text` with `ios.cmake`; then a Ninja arm64 simulator build of `test_os`, `test_text`, `test_locale`, `test_mobile`, `test_util`, run in a booted iPhone simulator with `xcrun simctl spawn` (`.github/ios_sim_check.sh`) |
| `windows-msvc` | windows-latest | MSVC, Ninja; `ctest -LE "slow|emu"`, then the emitted cmd/pwsh/PowerShell goldens |
| `windows-mingw` | windows-latest | MSYS2 UCRT64 GCC; `ctest -LE "slow|emu"` |
| `freebsd` | FreeBSD 14.3 VM | Native build; `ctest -LE "slow|emu|gpu"` |
| `android` | ubuntu-24.04 + API 34 x86_64 emulator | NDK cross builds; `--emit android` goldens, `test_script`, `test_mobile`, `test_locale`, `test_text`, `test_os`, `test_os_ext`, `test_os_sys`, `test_util` and `test_loadbalancer` on the emulator (`.github/android_check.sh`) |

Only `linux-vulkan` installs `glslangValidator` and the Vulkan headers; every other job compiles the Vulkan backend and its tests out. To reproduce `linux-vulkan` locally, set `VK_ICD_FILENAMES` and `VK_DRIVER_FILES` to the lavapipe ICD (`/usr/share/vulkan/icd.d/lvp_icd*.json`) so the loader cannot pick a real GPU. `test_rpc_tls` has not run on Windows CI: its `openssl` probe used a shell redirect to `/dev/null` in the last run; the probe no longer does, and the fixed form has not run there yet. Pass counts per job: [STATUS.md](STATUS.md).

`ci/run-local.sh [job...]` mirrors these jobs with user-local toolchains for a machine without GitHub Actions. It hard-codes default toolchain paths under `/workspace` that you override with environment variables (`LLVM`, `MINGW_TC`, `WINE`, `ZIG`, `NDK`, `SHELL_DIRS`, ...). `ci/repro_check.sh` checks that two clean Release builds give bit-identical `forge`, `forge-serve` and `libforge.so` (the `repro` job; `REPRO_JOBS=N` limits its compile jobs).

## Sanitizers
```sh
# ASan + UBSan, everything (what linux-asan runs)
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++-23 -DCMAKE_C_COMPILER=clang-23 -DFORGE_SANITIZE=ON -DFORGE_BUILD_EXAMPLES=OFF
cmake --build build-asan -j4
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ctest --test-dir build-asan -LE "slow|emu" -j2 --timeout 3600
# TSan, one suite
cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=clang++-23 -DCMAKE_C_COMPILER=clang-23 -DFORGE_TSAN=ON -DFORGE_BUILD_EXAMPLES=OFF
cmake --build build-tsan --target test_rpc && TSAN_OPTIONS="halt_on_error=1 second_deadlock_stack=1" ./build-tsan/test_rpc
```
- On kernels with `vm.mmap_rnd_bits` above 28 the sanitizer runtimes fail at start; `sudo sysctl vm.mmap_rnd_bits=28` (the CI jobs do this).
- A full ASan Debug tree needs several GB of disk; a test that crashes on purpose (a child that segfaults) resets the signal to `SIG_DFL` first, because the sanitizer's own handler would report and exit instead.
- Suppressions belong in `ci/sanitizers/` with a comment saying why each is a false positive; there are none at present.
- A sanitizer tree does not register `python_bindings` or `install_consumer`; `forge rpc-serve` leaves the address-space cap off its script child there; AOT binaries built by the tests link the sanitizer runtime.
- State on 2026-11-13: no sanitizer build of the final tree has been run. The last whole-suite pass was interrupted; what did run, and on which code, is in [UNTESTED.md](UNTESTED.md#6-sanitizers-and-fuzzing).

## Useful environment variables
- `FORGE_THREADS`: worker pool size (use 2 on small machines and emulators).
- `FORGE_SIMD_TARGET=scalar`: run with a specific kernel table.
- `FORGE_TEST_ONLY`: run a subset of cases.
- `FORGE_VK_SSBO_ALIGN=256`: emulate a strict Vulkan driver.
- `MOCK_DEVICES=N`, `MOCK_MEM_LIMIT=<bytes>`: devices and memory the mock GPU drivers report.
- `FORGE_UPDATE_EMIT_GOLDEN`, `FORGE_GOLDEN_UPDATE`: rewrite the emit and script golden files instead of comparing.

The full list is in [configuration.md](configuration.md).

## Fuzzing
By default each fuzz target links a small built-in mutation driver (`fuzz/driver.cpp`) and runs `FORGE_FUZZ_ITERS` inputs over its corpus in `fuzz/corpus/<name>/`, with a 600 s timeout. With clang and `-DFORGE_LIBFUZZER=ON` they link libFuzzer instead and are not registered with ctest.

| Target | Input | What counts as a finding besides a crash or sanitizer report |
|--------|-------|------|
| `fuzz_serve_http` | forge-serve's HTTP request reader (`apps/serve_http.hpp`), fed in fuzzer-chosen `recv()` sizes as a kept-alive connection: pipelined requests, `Content-Length` and chunked bodies, `Expect: 100-continue`, with small limits | any exception; a body over the limit, a body without framing, both framings accepted, a malformed method / target / header name accepted |
| `fuzz_proxy_http` | the L7 proxy (`loadbalancer/proxy.hpp`): the client loop against an empty pool, then the upstream response parser | any exception |
| `fuzz_rpc` | `authorized_keys`, token file, client map and permission list parsers; FRPC op frames, FRPS/FRPZ script frames and the FRPH handshake sent to real `rpc::server` instances on 127.0.0.1 | a reply larger than a frame allows |
| `fuzz_json_schema` | JSON Schema → GBNF (`response_format`) | an accepted schema whose grammar the GBNF parser rejects |
| `fuzz_policy` | security policy files | `to_json` output that does not parse back to the same policy |
| `fuzz_signature` | bundle `FORGE-SIGNATURE.json` manifests and detached `.sig` files | a manifest that verifies against a key nobody holds |
| `fuzz_aot` | precompiled plan blobs (`fuzz/fuzz_aot.cpp`) | an exception that is not a `forge_exception` |
| `fuzz_fpk`, `fuzz_fsc`, `fuzz_gguf`, `fuzz_onnx`, `fuzz_safetensors`, `fuzz_tokenizer`, `fuzz_blueprint`, `fuzz_json`, `fuzz_grammar`, `fuzz_script` | the file formats and the script front end | an exception that is not a `forge_exception` |

The `aot` seed is written by `FORGE_AOT_WRITE_SEED=fuzz/corpus/aot/seed.fplan build/dev/test_aot`; `test_aot` fails when the checked-in seed no longer loads.

Seeds: `fuzz/make_corpus.py <forge>` regenerates the binary model seeds (it needs numpy, safetensors and gguf), `fuzz/make_net_corpus.py` the HTTP, RPC and JSON Schema seeds (standard library only). The policy and signature seeds are written by the code under test (`cmake --build <build> --target fuzz_make_sig_corpus && <build>/fuzz_make_sig_corpus fuzz/corpus`), so they carry real digests and signatures by a fixed throwaway key. An input that found a bug is kept as `fuzz/corpus/<name>/regress_<what>` in the commit that fixes it, so every later build replays it.

libFuzzer with ASan and UBSan, as the `linux-fuzz` CI job runs it:
```sh
cmake -S . -B build-fuzz -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++-23 -DCMAKE_C_COMPILER=clang-23 \
      -DFORGE_SANITIZE=ON -DFORGE_LIBFUZZER=ON -DFORGE_BUILD_EXAMPLES=OFF
cmake --build build-fuzz --target fuzz_rpc
mkdir -p /tmp/rpc-corpus && cp fuzz/corpus/rpc/* /tmp/rpc-corpus/      # libFuzzer adds files to its first corpus directory
./build-fuzz/fuzz_rpc -max_total_time=300 -timeout=30 /tmp/rpc-corpus
```

## Performance tools (not part of ctest)
- `tools/perf_baseline.py --forge build/release/forge --model <model.fpk> [--gemm N] [--reps R] [--baseline perf/baseline-x86_64.json] [--check] [--tolerance T] [--allow-missing] [--report out.json] [--markdown FILE]` runs `forge bench` and records or checks throughput against a stored baseline. `--allow-missing` reports, without failing, baseline metrics this CPU cannot produce (no AVX-512 lane); `--report` and `--markdown` write the comparison as JSON and as a table. The `perf` CI job runs it without blocking.
- `test_cuda_hip bench <codec> <rows> <cols> <k> [reps]` times one quantised matmul on CUDA device 0 (e.g. `q4_k 512 25600 5120`). Inputs stay resident inside a run, so only the kernels are timed; it reports the fastest of 7 batches of `reps` calls (default 20), the median, TOPS, weight GB/s and the kernel path (`qmatmul_mma`, `qmatmul_mmq` or `qmatmul_q81`).
- `tools/bench_llm.sh --gguf <model.gguf> [--fpk <model.fpk>] [--forge PATH] [--llama-bench PATH] [-p 512] [-n 128] [-r 3] [--device cuda] [--codec q4_k] [--nsys PREFIX] [--no-llama]` compares FORGE with llama.cpp's `llama-bench` on the same model: prefill and decode tok/s, FORGE with and without `FORGE_GRAPHS=1`, and FORGE's GPU driver calls per decode token from `forge bench`. `--nsys` adds an Nsight Systems kernel summary per FORGE run. It needs a GPU and a model.

## Golden test Python environment
`golden_tiny_qwen3_5` skips without torch and transformers 5 or newer. `golden_quant_inplace` skips its GGUF sub-check without `gguf`, and `golden_gguf_types` is skipped as a whole. Install the pinned set from `verify/golden/requirements.txt` into a venv and configure with `-DFORGE_GOLDEN_PYTHON=<venv>/bin/python`.
