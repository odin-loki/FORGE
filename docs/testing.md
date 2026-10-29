# Testing

## Concepts
- Every file `tests/test_*.cpp` becomes its own executable and its own ctest entry. The executable runs its `TEST(...)` cases and prints `N failure(s)`.
- Exit code 77 means *skipped*, for example when the hardware or SDK is missing.
- **The oracle:** every SIMD, GPU and quantised path is compared against the scalar reference implementation in `spec`, within the tolerance declared for each op.

## Test kinds

| Kind | Where | Notes |
|------|-------|-------|
| Unit and differential | `tests/test_*.cpp` | One ctest per file |
| Golden models | `verify/golden/*.py` | Generates `golden/tiny_f32.fpk`. Later tests depend on it |
| Mock GPU drivers | `golden_tiny_llama_{cuda,opencl,ze,hip}_mock` | Label `gpu`. Mock runtimes stand in for the vendor libraries |
| Script tests | `tests/*.py` | Signing, ONNX import, Python bindings, `forge-serve` HTTP, SBOM |
| Doc examples | `docs/examples/*` | `doc_example_*`: the code shown in the guides |
| Cross link | `xos_*` | Label `cross`. Uses zig |
| Fuzz | `fuzz/` | Built-in driver, or libFuzzer with `FORGE_LIBFUZZER=ON` |

## Running
```sh
(cd build/dev && ctest -j4)                 # everything
(cd build/dev && ctest -R doc_example)      # just the documentation examples
(cd build/dev && ctest -L gpu)              # mock-driver GPU tests
./build/dev/test_os                         # one suite directly
```
For running on wine and the Android emulator, see [building.md](building.md). Current results are in [STATUS.md](STATUS.md).

## Environment variables used by tests
- `FORGE_THREADS`: size of the worker pool (2 on the emulator).
- `FORGE_LANE_TARGET`: forces a lane table, for example `scalar`.
- `FORGE_TEST_TIMEOUT`: per-test timeout in `android_emulator.sh` (default 600 s).
