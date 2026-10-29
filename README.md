# FORGE

FORGE is a single C++20 library that covers three jobs usually split across separate projects:

- **SIMD kernels** with runtime CPU dispatch (`lane`)
- **GPU compute** through one API: CPU, Vulkan, CUDA, HIP, Level Zero and Metal (`grid`)
- **AI inference** from a self-describing model file, `.fpk` (`flow`, `pack`, `mind`)

The model file carries its own computation graph, so supporting a new architecture means importing a file, not writing engine code. FORGE is a clean-room design: it is not a fork or wrapper of another project.

> **Status:** in development. Every layer has code and tests. Tests have run on Linux, on Windows under wine, on the Android emulator and on qemu cross targets. FreeBSD and macOS builds have been cross-linked but never run. iOS has not been built. GPU vendor backends have only run against mock drivers. See [docs/STATUS.md](docs/STATUS.md) for results per platform.

## Features

| Layer | Headers | What it provides |
|-------|---------|------------------|
| core | `core/include/forge/core` | Tensors, memory-mapped files (`FORGE_HUGEPAGES=1` enables huge pages for mappings of 2 MiB or more) |
| os | `os/include/forge/os` | Portability layer: threads, sockets, IPC, CPU topology, NUMA, perf counters, sandbox, service integration and more |
| lane | `lane/include/forge/lane` | CPU feature detection and a kernel table resolved once (scalar, AVX2, AVX-512, NEON, plus AMX/SME tile units) |
| grid | `grid/include/forge/grid` | Device backends loaded at runtime; scan, compaction, radix sort and similar primitives |
| flow | `flow/include/forge/flow` | Graph IR, optimisation passes, memory planning, quantisation codecs, autodiff, RPC devices |
| pack | `pack/include/forge/pack` | The `.fpk` format: BLAKE3 hashes, Ed25519 signing, optional encryption, BPE tokenizer |
| mind | `mind/include/forge/mind` | Sessions, samplers, batching, speculative decoding, GBNF grammars, LoRA, plus the C ABI `forge.h` |
| balance | `balance/include/forge/balance` | Load-balancing strategies (p2c, round robin, least connections, EWMA, hash ring, KV-prefix affinity) |

There are two apps:
- `forge`, the command-line tool. Its commands are `info`, `import`, `inspect`, `run`, `bench`, `dump`, `keygen`, `sign`, `verify`, `encrypt` and `rpc-serve`.
- `forge-serve`, an OpenAI-compatible HTTP server.

## Quick start (Linux)

```sh
cmake --preset dev              # configures into build/dev (RelWithDebInfo)
ninja -C build/dev -j4
./build/dev/forge info          # CPU features, lane target, grid backends
(cd build/dev && ctest -j4)     # golden, signing and ONNX tests need numpy, safetensors, onnx, cryptography
./build/dev/forge run model.fpk --prompt "Hello" -n 32
```

You can also pass a Hugging Face directory or a `.gguf` file to `forge run`. It is imported once into `~/.cache/forge`.

## Building on other platforms

Each preset in `CMakePresets.json` has its own build directory:

| Preset | Directory | Toolchain | Tested? |
|--------|-----------|-----------|---------|
| `dev`, `release`, `asan` | `build/dev`, `build/release`, `build/asan` | host | yes (Linux) |
| `windows-mingw` | `build/win64` | `cmake/toolchains/mingw-w64.cmake` | runs under wine |
| `android-arm64` | `build/android-arm64` | `cmake/toolchains/android-arm64.cmake` (needs `ANDROID_NDK_HOME`) | x86_64 emulator |
| `macos-osxcross` | `build/macos` | `cmake/toolchains/macos-osxcross.cmake` | not run (the zig `xos_*` cross-link tests cover macOS) |
| `ios` | `build/ios` | `cmake/toolchains/ios.cmake` | not built |

The `aarch64-sve-qemu` and `riscv64-rvv-qemu` toolchains run their tests under qemu. Wine, emulator and qemu steps are in [docs/building.md](docs/building.md).

## Repository layout

```
core os lane grid flow pack mind balance   library layers (header-mostly)
spec/       operation specs and generators      verify/   oracle + golden models
support/    blake3, aio, json, linalg, arenas   import/   import templates
apps/       forge, forge-serve                  bindings/ android (Kotlin/JNI), apple (Swift)
python/     Python package                      examples/integrations/  sample apps
tests/      test_*.cpp + script tests           fuzz/, perf/, tools/, ci/
docs/       guides (start at docs/index.md)
```

## Documentation

Start at **[docs/index.md](docs/index.md)**. The design and roadmap are in [FORGE_PLAN.md](FORGE_PLAN.md), and security reporting is covered in [SECURITY.md](SECURITY.md).

## Licence

FORGE is dual licensed.

- **AGPL-3.0 or later** — see [`LICENSE`](LICENSE). Free for individuals, charities and not-for-profits, education and research, and organisations with annual income under AUD 50,000.
- **Commercial licence** — required for organisations at or above that threshold, or anyone who needs to keep modifications private. See [`COMMERCIAL-LICENCE.md`](COMMERCIAL-LICENCE.md).

Attribution line for AGPL users:

> Powered by FORGE, developed by Odin Loch. Licensed under AGPL-3.0+.

---

Copyright © 2026 Odin Loch, trading as Imortek. Sydney, Australia.
Enquiries: odin.loch@outlook.com.au · Licensing terms: https://imortek.com.au/licensing.html
