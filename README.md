# FORGE — intrinsics, GPU compute and AI inference in one library

> **Status: design stage.** The architecture and delivery plan is in [`FORGE_PLAN.md`](FORGE_PLAN.md) (v0.1, 2026-10-29). No code has landed yet. The first target is **Convergence 1**: one model file running on CPU, NVIDIA and AMD with zero model-specific code.

---

## What this is

FORGE is one C++ library that covers three jobs usually split across separate projects:

- **SIMD intrinsics** on every CPU instruction set (the role xsimd plays today)
- **GPU compute** across NVIDIA, AMD and Intel, plus Vulkan, Metal and a CPU fallback (the role CUDA, HIP and SYCL play)
- **AI inference** from a self-describing model file (the role llama.cpp plays)

It is a new design, not a fork or wrapper of any of those projects. Code and model are fully decoupled: the model file carries its own computation graph, so running a new architecture means importing a file, not writing engine code.

---

## 🧠 The pattern — One Spec, Many Lowerings, One Oracle

1. **Spec once.** Every operation is declared once, with a scalar reference implementation.
2. **Capabilities, not hierarchies.** Targets advertise feature sets; the most specific valid implementation wins.
3. **Plan once, replay forever.** Dispatch, GPU work and tensor graphs are compiled once and replayed.
4. **Data, not code.** Model architectures, quantisation formats and tokenizers are data.
5. **One oracle.** Every SIMD, GPU and quantised path is differential-tested against the reference.

---

## Layers

| Module | Role |
|---|---|
| `lane` | SIMD: generated wrappers for every intrinsic, a portable fixed + scalable vector layer, vectorised math, algorithms, AMX/SME tiles |
| `grid` | GPU: graph-first programming model; CUDA, HIP, Level Zero/SYCL, Vulkan, Metal and CPU backends; portable primitives |
| `flow` | Tensor graph compiler: verify → optimise → partition → memory-plan → replay; pluggable quantisation codecs |
| `pack` | `.fpk` model container: blueprint graph, weights, tokenizer, state schema, content hashes, signatures |
| `mind` | Generic inference engine: sessions, paged KV cache, sampling, batching, speculative decoding, LoRA |
| `core` · `spec` · `verify` | Types, op definitions and code generators, oracle and differential tests |
| `os` | Portable OS layer for Linux, Android, Windows, macOS, iOS, FreeBSD — [`docs/os.md`](docs/os.md) |
| `balance` | Load balancer + L4/L7 proxy — [`docs/balance.md`](docs/balance.md) |
| `apps` | `forge` CLI and `forge-serve` HTTP server — [`docs/serve.md`](docs/serve.md) |
| `bindings` | Android (Kotlin/JNI, Gradle) and Apple (Swift package / XCFramework) |

Build, cross-build, Android emulator and wine instructions: [`docs/STATUS.md#build`](docs/STATUS.md#build).
CI: the workflow lives at `ci/github-actions-ci.yml`; move it to `.github/workflows/ci.yml` by hand (the push token lacks `workflow` scope).

Full detail, repository layout, roadmap and first steps: [`FORGE_PLAN.md`](FORGE_PLAN.md).

---

## Roadmap at a glance

| Phase | Milestone |
|---|---|
| 0–3 | Foundations, `lane` core (x86 + NEON), `grid` core (CPU + CUDA + HIP), `flow` |
| 4 | **Convergence 1** — one `.fpk` runs on CPU, NVIDIA and AMD with no model-specific code |
| 5 | **Convergence 2** — Intel GPUs, SVE/RVV, batching, serving, multi-GPU |
| 6 | **Convergence 3** — Vulkan + Metal; Android and iOS |
| 7–8 | Vision, audio and diffusion sessions; hardening; v1.0 |

---

## 🚧 Honest framing

- **Work in progress.** See [`docs/STATUS.md`](docs/STATUS.md) for what is implemented and what has actually been tested on each platform.
- **Scope is large.** The minimum useful version (Convergence 1) is estimated at 60–90k hand-written lines; the full library at roughly 185–275k, plus tests. Breadth comes after Convergence 1, not before.
- **Clean-room design.** No code is copied from xsimd, llama.cpp or ggml, and nothing from CUDA's headers or libraries. Any third-party code that is used will be recorded in a `NOTICE` file with its licence.
- **Vendor libraries are optional.** cuBLAS, hipBLASLt and oneMKL may be used where present and faster, but are never required.

---

## Licence

FORGE is dual licensed.

- **AGPL-3.0 or later** — see [`LICENSE`](LICENSE). Free for individuals, charities and not-for-profits, education and research, and organisations with annual income under AUD 50,000.
- **Commercial licence** — required for organisations at or above that threshold, or anyone who needs to keep modifications private. See [`COMMERCIAL-LICENCE.md`](COMMERCIAL-LICENCE.md).

Attribution line for AGPL users:

> Powered by FORGE, developed by Odin Loch. Licensed under AGPL-3.0+.

---

Copyright © 2026 Odin Loch, trading as Imortek. Sydney, Australia.
Enquiries: odin.loch@outlook.com.au · Licensing terms: https://imortek.com.au/licensing.html
