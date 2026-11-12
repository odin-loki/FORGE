# FORGE

FORGE is a C++23 library and toolset for running AI models on CPUs and GPUs. One codebase covers:

- **SIMD kernels** chosen at run time for the CPU in use (`simd`)
- **GPU compute** behind one API, with runtimes loaded at run time: Vulkan, CUDA, HIP, OpenCL, Level Zero and Metal (`gpu`)
- **Inference** from a self-describing model file, `.fpk`, which carries its own computation graph (`graph`, `modelfile`, `inference`): generative decoders and embedding models, optionally compiled ahead of time (`forge aot`)
- **Serving**: an OpenAI-compatible HTTP server (completions, chat, embeddings) with a load-balancing proxy (`forge-serve`, `loadbalancer`)
- **A portable OS layer** for Linux, Android, Windows, macOS, iOS and FreeBSD (`os`)
- **Scripting**: `forge script`, an interpreted, memory-safe subset of C++23 that can also be translated to shell scripts or compiled ahead of time to a native executable with `forge script --aot` (`script`, [docs/aot.md](docs/aot.md))

Because the model file describes its own graph, a new architecture needs an import template, not new engine code. FORGE is a clean-room design, not a fork or wrapper of another project.

> **Status: in development (version 0.1.0).** Two things are true at once, and they are different kinds of evidence:
>
> - **What CI last ran (PR #53, 2026-11-04):** Linux GCC 14 and Clang 18 111/111, macOS 111/111, Windows MSVC 106/106, Windows MinGW 84/84 run, FreeBSD 14.3 (VM) 60/60; ASan and TSan runs of the script tests; Android arm64-v8a and x86_64 cross-builds, with the script tests run on an x86_64 emulator. The iOS job only built `libforge`.
> - **What has landed since (2026-11-12) and has not been through CI:** model ahead-of-time compilation (`forge aot`), script ahead-of-time compilation (`forge script --aot`), the full Hugging Face tokenizer pipeline and chat templates, the GPT-2 / Gemma / Phi-3 / BERT / ViT / Whisper-encoder import families and embedding sessions (`forge embed`, `/v1/embeddings`), portable OpenCL / Level Zero kernels, more SIMD kernel tables and the raw intrinsic wrappers, four waves of OS-layer additions, the scheduler / linear algebra / profiling / reflection / vision support modules, and a much larger CI workflow. All of it was built and tested **on one machine: Linux x86_64 with clang 23** ({FULLSUITE}). Everything else is weaker evidence: other CPU architectures under qemu-user, Windows code under wine, GPU backends against mock drivers, macOS / FreeBSD / Android code cross-compiled and never run, iOS code not compiled. GCC 14, MSVC, MinGW, Apple clang and the NDK have not compiled most of the new code yet; the new CI jobs have never run on GitHub.
>
> Hardware: aarch64 and riscv64 have run under qemu only. CUDA has run on a real RTX 3090 (WSL2 and native Windows 11; not the code added on 2026-11-12). Metal's `test_gpu_metal` passed on the GitHub macOS runner's GPU (PR #53). HIP, OpenCL and Level Zero have only run against mock drivers. Vulkan has run only on software devices (lavapipe, SwiftShader). Nothing has run on iOS or on a physical Android device. Per-feature detail, with how each part was checked: [docs/STATUS.md](docs/STATUS.md); open work: [TODO.md](TODO.md).

## Quick start (Linux)

You need CMake 3.25 or newer, Ninja, Python 3 and a C++23 compiler (the CI workflow uses g++-14 and clang-23; the 2026-11-12 work has only been built with clang 23).

```sh
cmake --preset dev                      # configure into build/dev (RelWithDebInfo)
cmake --build build/dev -j4
./build/dev/forge info                  # CPU features, simd target, GPU runtimes found, memory budget
ctest --test-dir build/dev -j4 -LE "slow|emu"   # tests; also writes build/dev/golden/tiny_f32.fpk
./build/dev/forge run build/dev/golden/tiny_f32.fpk --prompt "Hello" -n 8
```

The golden test model has random weights, so its output text is meaningless; it shows the pipeline runs end to end. The golden-model tests, and the tests that use the golden model, need the Python packages `numpy` and `safetensors` (`onnx` for `onnx_import`, `cryptography` for `modelfile_signing`). Without them CMake marks those tests "Not Run (Disabled)". Windows, macOS, FreeBSD and cross builds: [docs/building.md](docs/building.md).

To run a real model, import it first, or pass a Hugging Face directory, `.gguf` or `.onnx` file straight to `forge run` (it is imported once into `~/.cache/forge`):

```sh
./build/dev/forge import /path/to/hf_model -o model.fpk --codec q8_0
./build/dev/forge run model.fpk --prompt "Hello" -n 32
./build/dev/forge-serve model.fpk --port 8080
```

To skip plan compilation at start-up, compile the model ahead of time ([docs/aot-models.md](docs/aot-models.md)):

```sh
./build/dev/forge aot model.fpk                                   # precompiled plans, stored in the .fpk
./build/dev/forge aot model.fpk --aot-mode native -o model.so     # plans compiled to native code; forge run model.so
./build/dev/forge aot model.fpk --aot-mode native --emit exe -o model && ./model --prompt "Hello" -n 32
```

## What is in the box

| Layer | Headers | Provides |
|-------|---------|----------|
| core | `core/include/forge/core` | Tensors, dtypes (FP4/FP6/FP8, bf16, complex), errors, logging, memory budget, memory-mapped files |
| os | `os/include/forge/os` | Files, processes, process pools and limit groups, sockets, an event reactor, IPC, NUMA, huge pages, CPU topology, sandboxing, keystore, services, dynamic libraries, desktop integration (D-Bus client, X11 clipboard, proxy and appearance settings, notifications), text and locale (`forge_text`), mobile sensors (`forge_mobile`). Which calls are native, degraded or unsupported per OS, and which were only cross-compiled: [docs/os-parity.md](docs/os-parity.md) |
| ops | `ops/` | One declaration per tensor op (`ops/defs/*.fspec`), a scalar reference, and the code generators |
| simd | `simd/include/forge/simd` | CPU feature detection and kernel tables: scalar, SSE2/SSE4.2, AVX2, AVX-512, NEON, SVE, RVV, and (emulator-tested only) LoongArch LSX/LASX, POWER VSX, s390x, MIPS MSA, wasm SIMD128; AMX/SME tile kernels; int8 quantised-matmul kernels; generated one-to-one wrappers for the raw intrinsics of eight ISA families |
| gpu | `gpu/include/forge/gpu` | Devices, recorded graphs, parallel primitives, GPU runtimes loaded with `dlopen`; tuned CUDA/HIP kernels and portable OpenCL / Level Zero kernels for quantised matmul, attention and mixture of experts; automatic device selection |
| graph | `graph/include/forge/graph` | Graph IR, optimisation passes, memory planning, weight and KV codecs, autodiff, remote (RPC) devices, serialised plans and generated native code for ahead-of-time compilation |
| modelfile | `modelfile/include/forge/modelfile`, `modelfile/include/forge/import` | The `.fpk` container (BLAKE3, Ed25519 signing, XChaCha20-Poly1305 encryption), the blueprint language with versioned opsets, tokenizers (the Hugging Face pipeline: BPE, SentencePiece BPE, Unigram, WordPiece, WordLevel), importers for Hugging Face safetensors (also AWQ / GPTQ checkpoints), GGUF and ONNX |
| inference | `inference/include/forge/inference`, `inference/include/forge.h` | Generative and embedding sessions, samplers, continuous batching, speculative decoding, GBNF grammars, JSON Schema, chat templates (a Jinja subset), LoRA, diffusion and streaming sessions, precompiled plans, the C ABI |
| loadbalancer | `loadbalancer/include/forge/loadbalancer` | Load-balancing pool (7 strategies, circuit breakers, retries, backpressure) and L4/L7 proxy |
| script | `script/include/forge/script` | The `forge script` interpreter, bytecode VM, static checker, shell emitters and AOT compiler (`forge_script`, `forge_script_rt`) |
| support | `support/include/forge/*` | Hashing, JSON, async I/O, thread pool and work-stealing scheduler, caches, linear algebra, FFT/audio, vision (PNG / BMP / PNM decoding, video bitstream parsing), profiler, reflection, RNGs |
| verify | `verify/` | The oracle that checks every lowering against the op reference, and the golden-model scripts |

Programs built by default (`FORGE_BUILD_APPS=ON`):

- `forge`: command-line tool (`info`, `import`, `inspect`, `run`, `embed`, `aot`, `bench`, `dump`, `keygen`, `sign`, `verify`, `encrypt`, `rpc-serve`, `script`, `security`). See [docs/cli.md](docs/cli.md).
- `forge-serve`: OpenAI-compatible HTTP server (`/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`), and `forge-serve proxy` load balancer. See [docs/serve.md](docs/serve.md).
- `libforge` (`forge_c` target; `libforge.so.1` on Linux, `forge.dll` on Windows): the C ABI used by the Python, Kotlin and Swift bindings.

## Repository layout

```
core/ os/ ops/ simd/ gpu/ graph/ modelfile/ inference/ loadbalancer/ script/ support/ verify/   library layers
apps/                  forge and forge-serve
import/templates/      blueprint templates used by the importers (llama, qwen2, qwen3, qwen3_5(+moe), qwen3_moe, mixtral, deepseek_v2,
                       gpt2, gemma, gemma2, gemma3, phi3, bert, vit, whisper_encoder)
python/ bindings/      Python (ctypes), Android (Kotlin + JNI) and Apple (Swift) bindings
examples/              embedding examples (integrations/, built as example_*), script examples, a script plugin
tests/ fuzz/           unit, differential, script and fuzz tests
docs/                  guides; start at docs/index.md (docs/examples/* are compiled and run by ctest)
cmake/ ci/ tools/ perf/ third_party/   toolchains, CI and release tooling, perf baselines, vendored Monocypher and Level Zero header
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

Third-party code and data: Monocypher and the MIT-licensed Level Zero API header are vendored in `third_party/`; the intrinsic databases, the Unicode / CLDR / encoding tables and one ported routine are derived from third-party sources. [`NOTICE`](NOTICE) lists all of them with their licences (one, the source of the RISC-V intrinsic lists, is recorded there as not established) and `tools/sbom.py` emits the same list as an SBOM.

---

Copyright © 2026 Odin Loch, trading as Imortek. Sydney, Australia.
Enquiries: odin.loch@outlook.com.au · Licensing terms: https://imortek.com.au/licensing.html
