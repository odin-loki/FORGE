# FORGE documentation

Each guide follows the same shape: **concepts**, **API or command reference**, **examples**, and **platform notes and limits**. Anything that has not been run is labelled as such, and [UNTESTED.md](UNTESTED.md) gathers all of those labels in one list. The tree as it stands (2026-11-13) has been built and tested with clang 23 on Linux x86_64 only; see [STATUS.md](STATUS.md).

## Getting started
- [README](../README.md): what FORGE is, quick start, layout
- [building.md](building.md): requirements (clang 23), presets, CMake options, targets, installing and the CMake package, native and cross builds
- [testing.md](testing.md): kinds of test, ctest names and labels, how to run them, the CI jobs, sanitizers, fuzzing, performance tools
- [cli.md](cli.md): every `forge` command and flag
- [configuration.md](configuration.md): every environment variable in one place
- [memory.md](memory.md): memory budgets, eviction, admission control and OOM avoidance

## Library layers, bottom up
- [os.md](os.md): `forge::os`, the portability layer and its extension modules
- [os-parity.md](os-parity.md): every `forge::os` call per operating system: native, degraded or unsupported, and whether that path was run, only cross-compiled or never compiled
- [os-audit.md](os-audit.md): the audit of the OS layer: findings by wave, what was fixed, what is open
- [os-wrapper.md](os-wrapper.md): the `forge::os` headers beyond the core layer: processes, pipes, signals, file-system extras, users, time, system information, console, networking, file watching, service control, Windows registry, devices, clipboard, dynamic libraries, process pools. Built; the missing parts are listed in the page
- [text-locale.md](text-locale.md): `forge::os::text` and `forge::os::locale` (Unicode, encodings, locales; `forge_text`)
- [mobile.md](mobile.md): `forge::os::mobile`, sensors, touch and device services (`forge_mobile`)
- [support.md](support.md): logger, process wait and leases, thread pool, work-stealing scheduler and the engine executor, cache, deadlines, settings registry, linear algebra, profiler, reflection, vision (image codecs, video bitstream parsing)
- [simd.md](simd.md): CPU feature detection and SIMD kernel dispatch, the kernel tables per architecture, raw intrinsic wrappers, Tier A algorithms, `vec<T>` element types, std interop, and what ran under which emulator
- [gpu.md](gpu.md): devices, graphs, parallel primitives, GPU runtimes, the kernel families per backend (and which were only mock-verified, lavapipe-verified or never executed), device memory and eviction, device selection, determinism mode
- [graph.md](graph.md): graph IR, passes and fusions, plans, codecs, autodiff, device pipelines (`pipe:`, `pipe:auto`), RPC devices
- [modelfile.md](modelfile.md): the `.fpk` container, signing, encryption, shards, adapters, blueprints and opsets, tokenizers, importing (Hugging Face, GGUF, AWQ / GPTQ, ONNX, blueprint directories) and the template families
- [format.md](format.md): `.fpk` byte layout: sections, chunked encryption, tensor alignment, the binary blueprint, shards, the state / adapters / calib sections
- [inference.md](inference.md): models, sessions (generative, embedding, diffusion loop, streaming), samplers, generation, speculative decoding, chat templates, tool calls
- [compress.md](compress.md): model compression at import time (`forge compress`, `forge eval`, `forge import` optimising by default): pruning, sparsity, structured / depth / head / low-rank / MoE compression, mixed-precision quantisation, function-preserving rotations, calibration, context mixing, checkpoints and SSD-backed buffers
- [aot-models.md](aot-models.md): `forge aot`: precompiled plans in the `.fpk`, and models compiled to native libraries and executables

## Serving, remote devices and embedding
- [serve.md](serve.md): `forge-serve` HTTP server (completions, chat with end-of-turn stops and tool calling, embeddings, API keys, TLS, LoRA adapters)
- [loadbalancer.md](loadbalancer.md): load-balancing pool and `forge-serve proxy`
- [rpc.md](rpc.md): `forge rpc-serve`, authentication, TLS, certificate rules and revocation, weights kept on the server (protocol 2), remote scripts
- [capi.md](capi.md): the C ABI (`forge.h`, ABI 4)
- [bindings.md](bindings.md): Python, Kotlin and Swift (the Kotlin and Swift sources for ABI 4 have never been compiled)

## Scripting
- [scripting.md](scripting.md): `forge script`, the REPL, static checks, shell blocks and emitting shell scripts
- [shells.md](shells.md): shell backends and emit targets
- [aot.md](aot.md): `forge script --aot`: compile a script to a native executable (native and embed modes, baked policy, runtime library)

## Security
- [security.md](security.md): what is enforced today: script capabilities, OS sandbox, plugins, AI confirmation, RPC auth, signing, known gaps
- [security-standard.md](security-standard.md): draft security standard and the state of each control
- [security-api.md](security-api.md): `forge::security`: policy file and profiles, `enforce`, `safe_join`, audit, `lockdown`, `check`, `forge security check`
- [../SECURITY.md](../SECURITY.md): how to report a vulnerability, threat model

## Reference and project
- [reference.md](reference.md): op table, GPU kernel list and C ABI, **generated** by `tools/gen_docs.py` (do not edit by hand; the `docs_up_to_date` test checks it)
- [../ops/README.md](../ops/README.md): op specs and code generators
- [STATUS.md](STATUS.md): what has been tested where, and how each part was verified (native, emulator, wine, mock driver, cross-compile only, never compiled); a dated changelog of checks, newest first
- [UNTESTED.md](UNTESTED.md): the single list of what has not been tested, by platform, CPU architecture, GPU backend and feature, with the command that would verify each item; sanitizer and fuzzing coverage; what was measured and what was never timed; what was written from memory
- [RELEASING.md](RELEASING.md): release procedure
- [../CONTRIBUTING.md](../CONTRIBUTING.md): how to change code and docs, and how to merge
- [../TODO.md](../TODO.md): open work items (features not built, with the guide that describes each) and the list of things decided against
- [../NOTICE](../NOTICE): third-party code and derived data, with licences
- [../FORGE_PLAN.md](../FORGE_PLAN.md): the original design plan (a roadmap, not a description of current behaviour)
- [claims-audit.md](claims-audit.md): dated record of audits of doc claims against the code and what each pass corrected; not a description of current behaviour
- `wip/dn-scratch.patch`: a saved work-in-progress diff from a scratch directory. It is not applied and uses the old module names (`flow`, `grid`)

## Compiled examples
Each file in `docs/examples/` below is built by CMake and run by ctest as `doc_example_<name>`, so the snippets in the guides cannot silently drift from the API:

```sh
ctest --test-dir build/dev -R doc_example
```

| Example | Guide |
|---------|-------|
| `simd_dispatch.cpp` | simd.md |
| `gpu_graph.cpp` | gpu.md |
| `graph_build.cpp` | graph.md |
| `modelfile_roundtrip.cpp` | modelfile.md |
| `inference_generate.cpp` | inference.md (needs the golden model) |
| `os_features.cpp` | os.md |
| `loadbalancer_pool.cpp` | loadbalancer.md |
| `capi_generate.c` | capi.md (needs the golden model; not built when cross-compiling) |

`docs/examples/script/reflection.cpp` is a `forge script` example for scripting.md; it is not a ctest target.
