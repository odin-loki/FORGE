# Building

## Requirements
- CMake **3.25+** and Ninja (the presets use the Ninja generator; `ios` uses Xcode).
- A **C++23** compiler (the build sets `CMAKE_CXX_STANDARD 23`) and a C compiler. Tested with g++ and with clang (including `zig c++`).
- **Python 3** (required at configure time: it runs the code generators in `spec/gen/`).
- Optional, each enabling more at configure time:
  - `glslangValidator` + Vulkan headers (`vulkan/vulkan.h`): builds and embeds the SPIR-V kernels and enables the Vulkan backend. `spirv-val` additionally validates them.
  - `clang` (named `clang`, `clang-18/19/20`) and `ld.lld` (or the `ziglang` Python package for its lld): builds the embedded CUDA PTX and HIP code objects. Without it the CUDA/HIP runtimes have no kernels.
  - Python packages `numpy`, `safetensors`, `onnx`, `cryptography` for the golden-model, ONNX and signing script tests.

Configure messages tell you what was found, for example `FORGE: Vulkan backend disabled (need glslangValidator + vulkan/vulkan.h)`.

## Host build

```sh
cmake --preset dev                 # build/dev, RelWithDebInfo
cmake --build build/dev -j4
(cd build/dev && ctest -j4)
```

Other host presets: `release` (`build/release`) and `asan` (`build/asan`, Debug with ASan+UBSan). `cmake --workflow --preset release` runs configure, build and test in one go.

## CMake options

| Option | Default | Meaning |
|--------|---------|---------|
| `FORGE_BUILD_TESTS` | ON | Tests, script tests, fuzz targets and doc examples |
| `FORGE_BUILD_APPS` | ON | `forge` and `forge-serve` (golden-model tests need them) |
| `FORGE_BUILD_EXAMPLES` | ON | `examples/integrations/*.cpp` as `example_*` |
| `FORGE_SANITIZE` | OFF | ASan + UBSan |
| `FORGE_REPRODUCIBLE` | ON | Map build paths out of artefacts and drop the build-id, for bit-identical rebuilds |
| `FORGE_JNI` | OFF | Build `libforge_jni` for the Kotlin binding (ON in `android-arm64`) |
| `FORGE_LIBFUZZER` | OFF | Link fuzz targets with libFuzzer instead of the built-in driver |
| `FORGE_FUZZ_ITERS` | 4000 | Mutated inputs per fuzz target under ctest |
| `FORGE_HIP_ARCHS` | `gfx90a gfx942 gfx1100` | AMDGPU targets embedded for HIP |

## Targets
Most layers are header-only INTERFACE libraries: `forge_core`, `forge_os`, `forge_spec`, `forge_lane`, `forge_verify`, `forge_grid`, `forge_flow`, `forge_pack`, `forge_mind`, `forge_balance`. Compiled targets:

| Target | Output |
|--------|--------|
| `forge_lane_mv` | Static library with every compiled kernel variant (link this whenever you use lane, grid, flow or mind) |
| `forge_trust` | Static library: vendored Monocypher |
| `forge_video` | Static library: video decode support |
| `forge_metal` | Static library, Apple only (`grid/src/metal.mm`) |
| `forge_c` | `libforge.so`, soname `libforge.so.1`, version `0.1.0`: the C ABI |
| `forge_jni` | `libforge_jni.so` (with `FORGE_JNI=ON`) |
| `forge`, `forge-serve` | The apps |

Generated files go to `build/<preset>/generated/`: the op table (`forge/spec/ops.gen.hpp`, `ops.md`), x86 intrinsic wrappers (`forge/lane/raw/x86.gen.hpp`) and GPU kernels (`gpu/*.comp`, `kernels.cu`, `kernels.cl`, `kernels.metal`, and embedded SPIR-V).

## Installing and packaging
`cmake --install build/release --prefix /opt/forge` installs `libforge`, `forge_lane_mv`, `forge_trust`, the apps, the licence files and headers. CPack (`cd build/release && cpack`, or `tools/release.py build`) produces binary and source archives: TGZ (plus DEB on Linux), or ZIP on Windows.

**Known gap:** the install rule copies headers from `core`, `lane`, `spec`, `grid`, `flow`, `pack`, `mind`, `support` and `verify`, but not from `os/include` or `balance/include`, and not the generated headers. Since `core` includes `<forge/os/os.hpp>`, an installed tree is not yet usable for compiling against the C++ headers; build against the source tree (or use the C ABI) for now. Installed binaries also look for import templates in the source tree unless `FORGE_TEMPLATES` is set.

## Presets

| Preset | Build dir | Toolchain | Run? |
|--------|-----------|-----------|------|
| `dev`, `release`, `asan` | `build/dev`, `build/release`, `build/asan` | host | Linux: yes |
| `windows-mingw` | `build/win64` | `cmake/toolchains/mingw-w64.cmake` | under wine |
| `android-arm64` | `build/android-arm64` | `cmake/toolchains/android-arm64.cmake` (`ANDROID_NDK_HOME`) | x86_64 emulator build only |
| `macos-osxcross` | `build/macos` | `cmake/toolchains/macos-osxcross.cmake` (`OSXCROSS_ROOT`, `OSXCROSS_HOST`) | **never run** |
| `ios` | `build/ios` | `cmake/toolchains/ios.cmake`, Xcode generator | **never built** |

## Windows (mingw-w64, run under wine)
```sh
cmake --preset windows-mingw && cmake --build build/win64 -j4
tests/wine_run.sh build/win64      # runs every test_*/example_*/fuzz_* .exe, prints PASS/SKIP/FAIL
```
`wine_run.sh` uses `WINEPREFIX` (default `~/.wine-forge64`) and initialises it on first use. Use a pure 64-bit prefix; a wow64 prefix fails with `kernel32.dll c0000135`.

## Android (NDK)
```sh
export ANDROID_NDK_HOME=/path/to/android-ndk-r27c
cmake -S . -B build/android-x64 -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/android-arm64.cmake -DANDROID_ABI=x86_64
cmake --build build/android-x64 -j4
ANDROID_HOME=/path/to/sdk tests/android_emulator.sh native build/android-x64
```
- The toolchain defaults to `arm64-v8a`, API 26. CUDA/HIP images are not built for mobile.
- `android_emulator.sh native` pushes every `test_*`/`example_*` binary plus `libc++_shared.so`, runs them with `FORGE_THREADS=2`, retries when adb drops, and exits 77 when no device is attached. `FORGE_TEST_TIMEOUT` sets the per-test limit (default 600 s).
- A software (TCG) emulator that boots on a shared machine: `emulator -avd <name> -no-window -no-audio -no-snapshot -gpu swiftshader_indirect -memory 2048 -cores 2 -accel off` (about 6 minutes to boot).

## qemu and cross-link checks
- `cmake/toolchains/aarch64-sve-qemu.cmake` and `riscv64-rvv-qemu.cmake` cross-build and run ctest under qemu-user; extra `lane_<sve|rvv>_vl<N>` tests run the lane suite at several vector lengths and check that the scalable lane is active; `tile_sme_svl<N>` does the same for SME when the compiler supports it.
- With zig available, the `xos_*` tests (labels `cross;slow`) compile and link `test_os`, `test_balance` and `test_os_ext` for x86_64-freebsd, aarch64-macos, x86_64-macos, x86_64-windows-gnu and aarch64-linux-musl. They are link checks only.
- With Intel SDE installed, `tile_amx_sde_*` tests run the AMX tile kernels under emulation (label `emu`).

## Apple
`tools/build_xcframework.sh` builds `bindings/apple/FORGE.xcframework` (iOS device + simulator + macOS) and needs Xcode. **Not run.** A native macOS build needs no toolchain file but has also not been run.

## Building with zig as the compiler
When no system compiler is available, `pip install ziglang` and point CMake at wrapper scripts that run `python -m ziglang c++ "$@"` / `cc "$@"` (`CC=... CXX=... cmake --preset dev`). This is how the Linux results in this documentation revision were produced. Note that the CUDA/HIP image build looks for a binary named `clang`, so with zig alone the `test_gpu_cuda_mock` and `test_gpu_features_cuda_mock` tests fail (the test binary exits 77, "no embedded kernel image", but those two mock tests do not treat 77 as a skip).
