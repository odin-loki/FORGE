# Testing

## Concepts
- Each `tests/test_*.cpp` builds into its own executable and ctest entry. It runs its `TEST(...)` cases (`tests/check.hpp`) and prints a failure count.
- **Exit code 77 means skipped** (missing hardware, runtime or SDK). Most hardware tests declare `SKIP_RETURN_CODE 77`.
- **The oracle** (`verify/include/forge/verify/oracle.hpp`): every SIMD, GPU and quantised lowering is compared against the scalar reference in `spec/` within each op's declared tolerance. `test_flow`'s `oracle_matrix_all_ops` covers every registered lowering of every op.

## Kinds of test

| Kind | Where | ctest names / label |
|------|-------|---------------------|
| Unit and differential | `tests/test_*.cpp` | `test_<name>` |
| Golden models | `verify/golden/tiny_llama.py`, `tiny_moe.py` | `golden_tiny_llama`, `golden_moe_mla` (+ `_vulkan` when Vulkan is built). Writes `build/<preset>/golden/tiny_f32.fpk`, which later tests use |
| Mock GPU drivers | `tests/mock_cuda.c`, `mock_opencl.c`, `mock_ze.c` stand in for the vendor libraries | `test_gpu_*_mock`, `golden_tiny_llama_*_mock`, label `gpu` |
| GPU compile checks | generated kernels compiled with clang | `gpu_compile_*`, label `gpu` |
| Script tests | `tests/*.py` | `pack_signing`, `onnx_import`, `python_bindings`, `serve_http`, `sbom_spdx` |
| C ABI | `tests/capi_smoke.c` | `capi_smoke` |
| Doc examples | `docs/examples/*` | `doc_example_*` |
| Integration examples | `examples/integrations/*.cpp` | `example_*`, label `integration` |
| Docs freshness | `tools/gen_docs.py --check` | `docs_up_to_date` (fails if `docs/reference.md` is stale) |
| Fuzz | `fuzz/fuzz_*.cpp` (blueprint, fpk, gguf, grammar, json, onnx, safetensors, tokenizer) | `fuzz_*`, label `fuzz` |
| Cross link | zig cross builds | `xos_*`, labels `cross;slow` |
| Emulated tiles | Intel SDE | `tile_amx_sde_*`, label `emu` |
| Mobile | `tests/mobile_build.sh` | `mobile_*`, labels `mobile;slow` |

## Running
```sh
(cd build/dev && ctest -j4)                    # everything configured
(cd build/dev && ctest -j4 -LE "slow|emu")     # skip the long cross/mobile/emulator tests
(cd build/dev && ctest -R doc_example)         # just the documentation examples
(cd build/dev && ctest -L gpu)                 # GPU mocks and compile checks
(cd build/dev && ctest -L fuzz)                # fuzzers (FORGE_FUZZ_ITERS inputs each)
./build/dev/test_os                            # one suite directly
```
Run ctest from the build directory; tests use the source tree as their working directory where needed.

Wine and the Android emulator: see [building.md](building.md). Results per platform: [STATUS.md](STATUS.md).

## Useful environment variables
- `FORGE_THREADS`: worker pool size (use 2 on small emulators).
- `FORGE_LANE_TARGET=scalar`: run with a specific kernel table.
- `FORGE_VK_SSBO_ALIGN=256`: emulate strict Vulkan drivers (used by `test_vulkan_align256`).
- `MOCK_DEVICES=N`: number of devices the mock GPU drivers report.

## Fuzzing
By default each fuzz target is linked with a small built-in mutation driver (`fuzz/driver.cpp`) and runs over the corpus in `fuzz/corpus/`. With clang and `-DFORGE_LIBFUZZER=ON` they link against libFuzzer instead. `fuzz/make_corpus.py` regenerates seed inputs.

## Performance baselines
`tools/perf_baseline.py --forge build/release/forge --model <model.fpk> [--gemm N] [--reps R] [--baseline perf/baseline-x86_64.json] [--check] [--tolerance T]` runs `forge bench` and records or checks throughput against the stored baseline. It is not part of ctest.
