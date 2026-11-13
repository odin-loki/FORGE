# FORGE

FORGE is a C++23 library and toolset for running AI models on CPUs and GPUs. One codebase covers:

- **SIMD kernels** chosen at run time for the CPU in use (`simd`)
- **GPU compute** behind one API, with runtimes loaded at run time: Vulkan, CUDA, HIP, OpenCL, Level Zero and Metal (`gpu`)
- **Inference** from a self-describing model file, `.fpk`, which carries its own computation graph and session kind (`graph`, `modelfile`, `inference`): generative decoders, embedding models, diffusion loops and streaming encoders, optionally compiled ahead of time (`forge aot`)
- **Serving**: an OpenAI-compatible HTTP server (completions, chat with tool calling, embeddings; API keys and TLS) with a load-balancing proxy (`forge-serve`, `loadbalancer`), and remote devices over RPC (`forge rpc-serve`)
- **A portable OS layer** for Linux, Android, Windows, macOS, iOS and FreeBSD (`os`)
- **Scripting**: `forge script`, an interpreted, memory-safe subset of C++23 that can also be translated to shell scripts or compiled ahead of time to a native executable with `forge script --aot` (`script`, [docs/aot.md](docs/aot.md))

Because the model file describes its own graph, a new architecture needs an import template, not new engine code. FORGE is a clean-room design, not a fork or wrapper of another project.

> **Status: in development (version 0.1.0).**
>
> - **Verified:** the tree as of 2026-11-13 was built and tested in one configuration: Linux x86_64 (a 4-core VM), clang 23.1.6, `ctest -LE "slow|emu"`: 247 tests, 237 passed, 10 skipped (no GPU device; Unicode conformance files not downloaded), 0 failed.
> - **Not verified:** nothing else has built or run that tree. No CI run has happened for the work merged on 2026-11-12/13 (about 191,000 lines): GCC 14, MSVC, MinGW on Windows, Apple clang, the Android NDK in CI and FreeBSD have not compiled it; the labels `slow` and `emu` and the sanitizers were not run on it; the new GPU code has run only against mock drivers and a software Vulkan device (the Metal ports have never been compiled by a Metal toolchain); other CPU architectures only under emulators; the Kotlin and Swift bindings have never been compiled; no published model or tokenizer has been imported or run, only tiny random-weight test models. The full list, with the command to verify each item: **[docs/UNTESTED.md](docs/UNTESTED.md)**.
>
> Before that work, CI last ran on PR #53 (2026-11-04): Linux GCC 14 and Clang 18, macOS, Windows MSVC and MinGW, FreeBSD 14.3 and an Android emulator passed their suites, and CUDA ran on a real RTX 3090. Those results describe the older tree. What was checked where, entry by entry: [docs/STATUS.md](docs/STATUS.md); open work: [TODO.md](TODO.md).

## Quick start (Linux)

You need CMake 3.25 or newer, Ninja, Python 3 and clang 23, the project compiler and the only one that has built the current tree (configure refuses clang 18 and older and accepts 19 to 22 with a note; the CI workflow also names g++-14, which has not built this tree yet).

```sh
cmake --preset dev -DCMAKE_CXX_COMPILER=clang++-23 -DCMAKE_C_COMPILER=clang-23   # configure into build/dev (RelWithDebInfo)
cmake --build build/dev -j4
./build/dev/forge info                  # CPU features, simd target, GPU runtimes found, memory budget
ctest --test-dir build/dev -j4 -LE "slow|emu"   # tests; also writes build/dev/golden/tiny_f32.fpk
./build/dev/forge run build/dev/golden/tiny_f32.fpk --prompt "Hello" -n 8
```

The golden test model has random weights, so its output text is meaningless; it shows the pipeline runs end to end. The golden-model tests, and the tests that use the golden model, need the Python packages `numpy` and `safetensors` (`onnx` for `onnx_import`, `cryptography` for `modelfile_signing`, `gguf`, `tokenizers` and `sentencepiece` for the GGUF and tokenizer goldens). Without them CMake marks those tests "Not Run (Disabled)". Windows, macOS, FreeBSD and cross builds: [docs/building.md](docs/building.md).

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
| simd | `simd/include/forge/simd` | CPU feature detection and kernel tables: scalar, SSE2/SSE4.2, AVX2, AVX-512, NEON, SVE, RVV, and (emulator-tested only) LoongArch LSX/LASX, POWER VSX, s390x, MIPS MSA, wasm SIMD128; for each of them companion tables for attention, int8 quantised matmul and general algorithms (search, sort, top-k, text); AMX/SME tile kernels; `vec<T>` vector types for float, double and the integers; conversions to `std::experimental::simd` and `std::mdspan` overloads; generated one-to-one wrappers for the raw intrinsics of eight ISA families |
| gpu | `gpu/include/forge/gpu` | Devices, recorded graphs, parallel primitives, GPU runtimes loaded with `dlopen`; tuned CUDA/HIP kernels and portable kernels for quantised matmul, attention, mixture of experts and the encoder / vision ops on OpenCL, Level Zero, Vulkan (run on a software device only) and Metal (never executed); device memory pools with eviction; kernel program cache; determinism mode; automatic device selection |
| graph | `graph/include/forge/graph` | Graph IR, optimisation passes and fusions, memory planning, weight and KV codecs, autodiff, pipelines over several devices, remote (RPC) devices with server-side weight residency, serialised plans and generated native code for ahead-of-time compilation |
| modelfile | `modelfile/include/forge/modelfile`, `modelfile/include/forge/import` | The `.fpk` container (BLAKE3, Ed25519 signing, chunked XChaCha20-Poly1305 encryption, sharded files with a content-addressed blob store, optional state / adapter / calibration sections), the blueprint language with versioned opsets in a text and a binary form, tokenizers (the Hugging Face pipeline: BPE, SentencePiece BPE, Unigram, WordPiece, WordLevel), importers for Hugging Face safetensors (also AWQ / GPTQ checkpoints), GGUF, ONNX and blueprint directories |
| inference | `inference/include/forge/inference`, `inference/include/forge.h` | Generative and embedding sessions, diffusion-loop and streaming sessions chosen by the model file, samplers, continuous batching, speculative decoding (draft model, prompt lookup, early exit), GBNF grammars, JSON Schema, chat templates (a Jinja subset), tool-call grammars and parsers, LoRA, precompiled plans, the C ABI (ABI 4) |
| loadbalancer | `loadbalancer/include/forge/loadbalancer` | Load-balancing pool (7 strategies, circuit breakers, retries, backpressure) and L4/L7 proxy |
| script | `script/include/forge/script` | The `forge script` interpreter, bytecode VM, static checker, shell emitters and AOT compiler (`forge_script`, `forge_script_rt`) |
| security | `security/include/forge/security` | The security policy file and profiles, enforcement, `safe_join`, audit log, lockdown, integrity checks ([docs/security-api.md](docs/security-api.md)) |
| support | `support/include/forge/*` | Hashing, JSON, async I/O, thread pool and work-stealing scheduler, caches, allocators, linear algebra (dense and sparse), FFT/audio, vision (PNG / BMP / PNM decoding, transforms, video bitstream parsing), profiler, reflection, RNGs |
| verify | `verify/` | The oracle that checks every lowering against the op reference, and the golden-model scripts |

Programs built by default (`FORGE_BUILD_APPS=ON`):

- `forge`: command-line tool (`info`, `import`, `inspect`, `run`, `embed`, `denoise`, `stream`, `aot`, `bench`, `dump`, `requant`, `adapter`, `blueprint`, `shard`, `keygen`, `sign`, `verify`, `encrypt`, `rpc-serve`, `script`, `security`). See [docs/cli.md](docs/cli.md).
- `forge-serve`: OpenAI-compatible HTTP server (`/v1/completions`, `/v1/chat/completions` with end-of-turn stops and tool calling, `/v1/embeddings`, `/v1/models`; keep-alive HTTP/1.1, API keys, rate limits, TLS and mutual TLS, per-request LoRA adapters), and `forge-serve proxy` load balancer. See [docs/serve.md](docs/serve.md).
- `forge-script-run` (built with the script library, whatever `FORGE_BUILD_APPS` says): the runner stub that `forge script --aot` copies in embed mode. See [docs/aot.md](docs/aot.md).
- `libforge` (`forge_c` target; `libforge.so.1` on Linux, `forge.dll` on Windows): the C ABI used by the Python, Kotlin and Swift bindings. The Python binding is tested on Linux; the Kotlin and Swift sources for the current ABI have never been compiled ([docs/bindings.md](docs/bindings.md)).

## Repository layout

```
core/ os/ ops/ simd/ gpu/ graph/ modelfile/ inference/ loadbalancer/ script/ security/ support/ verify/   library layers
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

Start at **[docs/index.md](docs/index.md)**. What has not been tested: [docs/UNTESTED.md](docs/UNTESTED.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). Security: [SECURITY.md](SECURITY.md).

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
