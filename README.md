# FORGE

FORGE is a C++23 library and toolset for running AI models on CPUs and GPUs. One codebase covers:

- **SIMD kernels** picked at run time for the CPU you are on (`lane`)
- **GPU compute** behind one API, with backends loaded at run time: CPU, Vulkan, CUDA, HIP, Level Zero, OpenCL and Metal (`grid`)
- **Inference** from a self-describing model file, `.fpk`, which carries its own computation graph (`flow`, `pack`, `mind`)
- **Serving**: an OpenAI-compatible HTTP server with a built-in load-balancing proxy (`forge-serve`, `balance`)
- **A portable OS layer** for Linux, Android, Windows, macOS, iOS and FreeBSD (`os`)

Because the model file describes its own graph, adding an architecture means importing a file, not writing engine code. FORGE is a clean-room design, not a fork or wrapper of another project.

> **Status: in development (version 0.1.0).** Every layer has code and tests. Tests run on Linux x86_64, on Windows under wine, on the Android x86_64 emulator and on aarch64/riscv64 under qemu. FreeBSD and macOS have only been cross-compiled and linked; iOS has never been built. CUDA, HIP, OpenCL, Level Zero and Metal have only run against mock drivers, never real GPUs. Vulkan has run on software devices (Mesa lavapipe, SwiftShader). Details: [docs/STATUS.md](docs/STATUS.md).

## Quick start (Linux)

You need CMake 3.25 or newer, Ninja, Python 3 and a C++23 compiler (recent g++ or clang).

```sh
cmake --preset dev                      # configure into build/dev (RelWithDebInfo)
cmake --build build/dev -j4             # or: ninja -C build/dev -j4
./build/dev/forge info                  # CPU features, chosen lane target, GPU backends found
(cd build/dev && ctest -j4)             # full test suite; also generates build/dev/golden/tiny_f32.fpk
./build/dev/forge run build/dev/golden/tiny_f32.fpk --prompt "Hello" -n 8
```

The golden test model has random weights, so the text it prints is meaningless; it shows the pipeline runs end to end. Some script tests need `numpy`, `safetensors`, `onnx` and `cryptography` installed in the Python that CMake finds.

To run a real model, import it first, or pass a Hugging Face directory, `.gguf` or `.onnx` file straight to `forge run` (it is imported once into `~/.cache/forge`):

```sh
./build/dev/forge import /path/to/hf_model -o model.fpk --codec q8_0
./build/dev/forge run model.fpk --prompt "Hello" -n 32
./build/dev/forge-serve model.fpk --port 8080
```

## What is in the box

| Layer | Headers | Provides |
|-------|---------|----------|
| core | `core/include/forge/core` | Tensors, dtypes (incl. FP4/FP6/FP8, bf16, complex), errors, logging, memory-mapped files |
| os | `os/include/forge/os` | Files, processes, threads, sockets, polling, IPC, NUMA, huge pages, CPU topology, sandboxing, keystore, service integration, … |
| spec | `spec/` | One declaration per tensor op (`.fspec`) plus a scalar reference; code generators |
| lane | `lane/include/forge/lane` | CPU feature detection and kernel tables: scalar, AVX2, AVX-512, NEON, SVE, RVV; AMX/SME tile kernels |
| grid | `grid/include/forge/grid` | Devices, recorded graphs, parallel primitives, GPU backends loaded with `dlopen` |
| flow | `flow/include/forge/flow` | Graph IR, optimisation passes, memory planning, weight and KV codecs, autodiff, remote (RPC) devices |
| pack | `pack/include/forge/pack`, `forge/import` | The `.fpk` container (BLAKE3, Ed25519 signing, XChaCha20-Poly1305 encryption), BPE tokenizer, importers for Hugging Face safetensors, GGUF and ONNX |
| mind | `mind/include/forge/mind`, `forge.h` | Sessions, samplers, continuous batching, speculative decoding, GBNF grammars, JSON Schema, LoRA, diffusion and streaming sessions, the C ABI |
| balance | `balance/include/forge/balance` | Load-balancing pool (7 strategies, circuit breakers, retries, backpressure) and L4/L7 proxy |
| support | `support/include/forge/*` | BLAKE3, JSON, async I/O, thread pool, allocators, linear algebra, FFT/audio, vision, profiler, RNGs |
| verify | `verify/` | The oracle that checks every lowering against the spec, and golden-model scripts |

Programs built by default:

- `forge`: command-line tool (`info`, `import`, `inspect`, `run`, `bench`, `dump`, `keygen`, `sign`, `verify`, `encrypt`, `rpc-serve`). See [docs/cli.md](docs/cli.md).
- `forge-serve`: OpenAI-compatible HTTP server, and `forge-serve proxy` load balancer. See [docs/serve.md](docs/serve.md). Memory budgets and OOM avoidance: [docs/memory.md](docs/memory.md).
- `libforge.so` (`forge_c` target): the stable C ABI, used by the Python, Kotlin and Swift bindings.

## Repository layout

```
core/ os/ spec/ lane/ grid/ flow/ pack/ mind/ balance/ support/ verify/   library layers
apps/                  forge and forge-serve
import/templates/      blueprint templates used by the importers (llama, mixtral, deepseek_v2)
python/ bindings/      Python (ctypes), Android (Kotlin + JNI) and Apple (Swift) bindings
examples/integrations/ embedding examples built as example_* executables
tests/ fuzz/           unit, differential, script and fuzz tests
docs/                  guides; start at docs/index.md (docs/examples/*.cpp are compiled and run by ctest)
cmake/ ci/ tools/ perf/ third_party/   toolchains, CI and release tooling, perf baselines, vendored Monocypher and ze_api.h
FORGE_PLAN.md          original design and roadmap
```

## Documentation

Start at **[docs/index.md](docs/index.md)**. Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). Security: [SECURITY.md](SECURITY.md).

## Licence

FORGE is dual licensed.

- **AGPL-3.0 or later** ([`LICENSE`](LICENSE)). Free for individuals, charities and not-for-profits, education and research, and organisations with annual income under AUD 50,000.
- **Commercial licence** for organisations at or above that threshold, or anyone who needs to keep modifications private. See [`COMMERCIAL-LICENCE.md`](COMMERCIAL-LICENCE.md).

Attribution line for AGPL users:

> Powered by FORGE, developed by Odin Loch. Licensed under AGPL-3.0+.

Third-party code: Monocypher (see [`NOTICE`](NOTICE)) and the MIT-licensed Level Zero API header in `third_party/level-zero`.

---

Copyright © 2026 Odin Loch, trading as Imortek. Sydney, Australia.
Enquiries: odin.loch@outlook.com.au · Licensing terms: https://imortek.com.au/licensing.html
