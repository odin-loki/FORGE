# Building, cross-building and emulators

## Concepts
FORGE uses CMake (3.x) with Ninja and needs a C++20 compiler (g++ or clang). Most of the library is headers, exposed as INTERFACE targets: `forge_core`, `forge_os`, `forge_lane`, `forge_grid`, `forge_flow`, `forge_pack`, `forge_mind`, `forge_balance`, `forge_spec` and `forge_verify`. The compiled targets are:
- `forge_lane_mv`: static library of kernel variants.
- `forge_trust`: static library wrapping Monocypher.
- `forge_video`: static library.
- `forge_c`: shared library `libforge.so.1`.
- `forge_jni`: optional JNI shim.
- The apps `forge` and `forge-serve`.

## Options (`CMakeLists.txt`)

| Option | Default | Meaning |
|--------|---------|---------|
| `FORGE_BUILD_TESTS` | ON | Build `tests/test_*.cpp`, script tests and the doc examples |
| `FORGE_BUILD_APPS` | ON | Build `forge` and `forge-serve` (the golden-model tests need these) |
| `FORGE_BUILD_EXAMPLES` | ON | Build `examples/integrations` |
| `FORGE_SANITIZE` | OFF | Build with ASan and UBSan (the `asan` preset) |
| `FORGE_REPRODUCIBLE` | ON | Strip build paths and timestamps from artefacts |
| `FORGE_JNI` | OFF | Build `libforge_jni`. ON in the `android-arm64` preset |
| `FORGE_LIBFUZZER` | OFF | Link fuzz targets with libFuzzer instead of the built-in driver |

## Host build (Linux, tested)
```sh
cmake --preset dev && ninja -C build/dev -j4 && (cd build/dev && ctest -j4)
```
Several Python script tests (golden models, signing, ONNX import) need `numpy`, `safetensors`, `onnx` and `cryptography` in the Python that CMake finds.

## Windows (mingw-w64, run under wine: tested)
```sh
cmake --preset windows-mingw && ninja -C build/win64 -j4
tests/wine_run.sh build/win64      # runs every test_/example_/fuzz_ .exe; PASS/SKIP/FAIL per binary
```
`wine_run.sh` uses `WINEPREFIX` (default `~/.wine-forge64`) and initialises it with `wineboot` on first use.

## Android (NDK; tested on the x86_64 emulator)
```sh
export ANDROID_NDK_HOME=/path/to/android-ndk-r27c
cmake -S . -B build/android-x64 -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/android-arm64.cmake -DANDROID_ABI=x86_64
ninja -C build/android-x64 -j4
ANDROID_HOME=/path/to/sdk tests/android_emulator.sh native build/android-x64
```
- The toolchain defaults to `arm64-v8a` and API level 26 (`android-26`).
- `android_emulator.sh native` pushes every `test_*` and `example_*` binary together with `libc++_shared.so`. It runs them with `FORGE_THREADS=2` and retries when adb drops. It exits 77 if no device is attached.
- A software-only (TCG) emulator that boots reliably on a shared box:
  `emulator -avd <name> -no-window -no-audio -no-snapshot -gpu swiftshader_indirect -memory 2048 -cores 2 -accel off`. Booting takes about 6 minutes.
- Unix socket files are refused in `/data/local/tmp` by SELinux. For that reason `forge::os` local IPC uses abstract socket names on Linux and Android.

## qemu targets (tested)
- `cmake/toolchains/aarch64-sve-qemu.cmake` runs ctest under `qemu-aarch64 -cpu max,sve256=on`.
- `cmake/toolchains/riscv64-rvv-qemu.cmake` does the same for RVV.
- See [STATUS.md](STATUS.md) for building with zig as the cross compiler.

## Cross-link only (built and linked, never run)
- `ctest -L cross` runs the `xos_*` tests. They use `zig c++` to compile and link `test_os`, `test_balance` and `test_os_ext` for x86_64-freebsd, aarch64-macos, x86_64-macos, x86_64-windows-gnu and aarch64-linux-musl.
- The `macos-osxcross` preset needs an osxcross SDK (`OSXCROSS_ROOT`, `OSXCROSS_HOST`). **It has not been run.** A native macOS build needs no toolchain file.
- The `ios` preset and `tools/build_xcframework.sh` need Xcode. **They have not been built.**
