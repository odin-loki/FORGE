# Ahead-of-time compilation of models (`forge aot`)

This page is about models and graphs. Ahead-of-time compilation of `forge script` programs is a separate feature of the script module ([scripting.md](scripting.md)).

## Concepts
A session normally compiles a plan the first time it sees a sequence length: build the graph from the blueprint, run the passes, plan memory, pick a kernel per step ([graph.md](graph.md)). `forge aot` does that work once, ahead of time, in one of two modes.

| Mode | Output | At load | Needs a compiler |
|------|--------|---------|------------------|
| `plan` (default) | The compiled plans, stored in the `.fpk` as the `aot` section (or in a sidecar file) | The plan is rebuilt from the stored form: no blueprint build, no passes, no memory planning | No. Works on every platform FORGE builds on |
| `native` | A shared library, static library or executable containing the plans **and** straight-line C++ code that executes them | The plan is rebuilt as in `plan` mode and its kernel steps run as compiled code instead of being interpreted | Yes: a GCC- or Clang-compatible C++23 compiler on the build machine |

Both modes give **bit-identical outputs** to plans compiled at run time (see [What is bit-identical](#what-is-bit-identical)). Anything that does not match at load time (another model, another FORGE version, a sequence length that was not precompiled, a damaged or hostile blob) falls back to compiling the plan at run time. The fallback is logged at debug level (`FORGE_LOG=debug`) and is otherwise silent.

What is precompiled is a **target**: a sequence length (the blueprint's `seq` dimension), whether logits are kept for every row (`all_rows`), a KV codec, and the device, which is always `cpu`. Plans are cached per exact sequence length at run time, so the precompiled lengths are exact lengths too: `--shapes 1,8,16` covers decode (1) and prompts of exactly 8 or 16 tokens. A prompt of 13 tokens compiles its plan at run time, as before; decoding after it uses the precompiled length-1 plan.

The CPU lowerings pick their SIMD kernels at run time (`simd::active()`), so a precompiled plan is not tied to an instruction set: a plan blob built on an AVX2 machine loads on AVX-512, NEON or scalar machines.

## Usage

### Plan mode
```sh
forge aot model.fpk                                   # adds the "aot" section in place: lengths 1,2,4,...,64, KV codec f32
forge aot model.fpk -o model-aot.fpk --shapes 1,16,32,128 --kv-codec f32,int8 --all-rows
forge aot model.fpk --sidecar                         # writes model.fpk.fplan and leaves model.fpk alone
forge aot model.fpk --list                            # what is in the section, and whether it would be used
forge aot model.fpk --strip                           # remove the section
forge run model-aot.fpk --prompt "Hello" -n 32        # nothing to pass: the session finds the plans
```

`forge run` reports on stderr how many plans came precompiled (`[aot: 2 of the plans were precompiled ...]`).

### Native mode
```sh
forge aot model.fpk --aot-mode native -o model.so                      # shared library; weights stay in model.fpk
forge aot model.fpk --aot-mode native -o model.so --embed-weights      # the library is the whole model
forge aot model.fpk --aot-mode native --emit static -o libmodel.a
forge aot model.fpk --aot-mode native --emit exe -o model
forge run model.so --prompt "Hello" -n 32                              # a compiled library is a model path
./model --prompt "Hello" -n 32                                         # the executable is `forge run` for this model
```

A compiled library is accepted wherever a model path is: `inference::model::open("model.so")`, `inference::session("model.so")`, `forge_open("model.so", ...)` in the C ABI and therefore the Python, Kotlin and Swift bindings, `forge run`, `forge bench`, `forge-serve`. It is recognised by its first bytes (ELF, Mach-O or PE), not by its name. Everything on top of a session works unchanged (samplers, grammars, speculative decoding, rewind, prefix sharing, batching), because the session state and the plan stay in the engine; only the interpretation of kernel steps is replaced.

The weights of a library built without `--embed-weights` are read from, in order: `$FORGE_AOT_FPK`, a file with the recorded name next to the library, the absolute path recorded at build time. If that file is not the model the library was compiled from, the library's code is not used: a warning is printed and the model runs on plans compiled at run time.

The executable takes `--prompt TEXT | --tokens 1,2,3`, `-n`, `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--seed`, `--ctx`, `--ids`, `--no-bos`, `--grammar FILE`, `--verify` and `--model FILE.fpk`. It prints what `forge run` prints for the same arguments (the test `test_aot_native` compares them). `--temp`, `--top-k`, `--top-p` and `-n` given to `forge aot` become its defaults.

### Options

| Option | Mode | Meaning |
|--------|------|---------|
| `-o PATH` | both | Output. Plan mode: the `.fpk` to write (default: in place), or the sidecar with `--sidecar`. Native mode: the artefact (required) |
| `--aot-mode plan\|native` | | Default `plan` |
| `--shapes 1,8,16` | both | Sequence lengths. Default for `generative_sequence` models: `1,2,4,8,16,32,64`; for other session kinds: `1` |
| `--all-rows` | both | Also precompile the all-rows variant of every length (`forward_all`, speculative verification) |
| `--kv-codec f32,int8,...` | both | KV codecs to precompile for (default `f32`). A session using another codec compiles at run time |
| `--ctx N` | both | Context used while compiling (default: the model's maximum). Only bounds `--shapes` |
| `--sidecar` | plan | Write `<model>.fplan` instead of changing the `.fpk` |
| `--key NAME.key` | plan | Sign the result (a signed input must be signed again, see [Security](#security-of-precompiled-plans)) |
| `--drop-signature` | plan | Accept that a signed input becomes unsigned |
| `--strip`, `--list` | plan | Remove / show the section |
| `--emit shared\|static\|exe` | native | Default `shared` |
| `--embed-weights` | native | Put the `.fpk` into the artefact (`.incbin`). Refused above `--embed-limit-mb` (default 1024) |
| `--cxx C` | native | Compiler. Default: `$CXX`, then the compiler FORGE was built with (recorded in the build description) if it is still there, then `c++`, `clang++`, `clang++-23`, `g++`, `cl` on `PATH`. `cl.exe` is found only to be refused with a message: the driver needs GCC-style options |
| `--target TRIPLE` | native | Passed to the compiler as `--target=TRIPLE` (cross compiling; see limits) |
| `--simd T`, `--march M` | native | See [Target description](#target-description-native-mode) |
| `--cxxflag F` (repeatable) | native | Extra compiler flags |
| `--sdk FILE` | native | The build description (`forge_aot_sdk.txt`); default: `$FORGE_AOT_SDK`, then the build tree the `forge` binary came from |
| `--work DIR` | native | Keep the generated source (`DIR/forge_aot_model.cpp`) |

Environment: `FORGE_AOT=0` ignores precompiled plans (every plan is compiled at run time; a compiled library then runs without its code). `FORGE_AOT_PLAN=FILE` names the sidecar. `FORGE_AOT_FPK=FILE` names the weights of a compiled library. `FORGE_AOT_SDK=FILE` names the build description.

## How it works

### Plan mode
`forge aot` opens the model, and for every target calls the same `model::plan_for` a session calls, so the plan is compiled by the normal pipeline. Each plan is then written out (`graph/aot.hpp`): the optimised graph (node kinds, op names, inputs, attributes, shapes, weight names and codecs, folded constants by value), the memory plan (buffer kind, arena offset, size, alias root per value) and the step list (the lowering of each step **by name**, never by address). The layout is in [format.md](format.md#the-aot-section).

Every entry is keyed by a BLAKE3 hash of: the plan format version, the pass pipeline version (`graph::aot::pipeline_version`), the FORGE version, the model's `blueprint` section, the model's `tensors` index (names, shapes, codecs and content hashes of every weight), and the target (sequence length, `all_rows`, KV codec, device). The whole blob ends in its own BLAKE3.

At load `model::plan_for` looks for an entry with the key it computes for the request. If there is one, `graph::aot::instantiate`:

1. parses the entry with bounds checks on every count, string and offset;
2. resolves each weight by name through the model file and requires the stored codec and shape;
3. re-infers every shape from the op spec (`verify_graph`) and requires the stored shapes;
4. checks the memory plan against the graph (see [Security](#security-of-precompiled-plans));
5. runs the lowering half of `compile()` (`lower_plan`) with the stored lowering names: buffers are bound, state slots are created, quantised weights are bound in place, and each named lowering must exist for that op, support the shapes and pass the oracle on this machine, as in `compile()`;
6. requires the resulting step list to equal the stored one.

Any failure raises, is logged at debug level, and the plan is compiled the usual way. Skipped at load: the blueprint build, the passes (CSE, constant folding, simplification, fusion, DCE) and the memory planner.

### Native mode
`forge aot --aot-mode native` compiles the plans as above and then writes one C++ translation unit (`graph/aot_codegen.hpp`). For every plan it contains the plan's tables and one function per run of consecutive kernel steps. Inside a function every step is one statement:

```cpp
void seg1(rt::instance& I) {
    float* const A = I.arena; const tensor_view* const V = I.v.data(); (void)A; (void)V;
    rt::rms_norm(A + 0, V[4].data, A + 256, 4, 64, float(1.0000000000000001e-05));   // 3 %5 = lane.rms_norm [4,64]
    rt::gemm_nt(A + 256, V[6].data, A + 512, 4, 64, 64);                             // 4 %7 = lane.gemm_nt [4,64]
    /* a view of its input */                                                        // 5 %8 = cpu.reshape [4,4,16]
    I.call(6);                                                                       // 6 %10 = cpu.rope [4,4,16]
    ...
}
```

- Element-wise ops, softmax, RMS norm, f32 matmuls and reshapes become direct calls with the shapes and arena offsets as constants (`graph/aot_rt.hpp`).
- Everything else (attention, RoPE, gather, quantised matmul, MoE, ops that only have the reference kernel) is a call of the lowering's kernel over views that were built once when the code was bound to the plan.
- State steps (KV appends, recurrent `state_op`s) are not compiled. They split the plan into segments, and the engine's executor runs them between the segments.

The unit includes the FORGE headers, so the kernels are the same code the interpreted plan calls. It is compiled with the flags of the FORGE build it is compiled against (recorded by CMake in `build/<preset>/forge_aot_sdk.txt`: compiler, flags, include directories, definitions, static libraries) and linked with `libforge_simd_mv.a`. The library exports one symbol, `forge_aot_model_v1_get()`, returning the `forge_aot_model_v1` description declared in `inference/include/forge_aot.h`: FORGE version, SIMD target, compiler, the model file's path or its embedded bytes, the plan blob, and per plan the `bind` / `run` / `unbind` entry points and the step ranges that are compiled.

At load the engine instantiates the plan from the library's blob exactly as in plan mode, binds the compiled code to the plan's buffers (the code checks that every arena value is where it was generated for), and the executor then calls `run` for each compiled segment.

**What native mode does not do.** The SIMD kernels are *not* re-instantiated for one instruction set inside the generated unit: they stay the precompiled per-target tables of `libforge_simd_mv`, reached through `simd::active()` as in the interpreter. Recompiling them in the unit would mean compiling the kernel templates and the generic header code with one set of flags, and on x86 the two are built with different floating-point contraction today (the kernel units with FMA available, the rest for baseline x86-64), so the results would stop being bit-identical. What native mode removes is the interpretation: per-step dispatch, building the input views of every step on every run (a heap allocation per input), and the generic executor loop. The measured effect is under [Measured](#measured).

## Target description (native mode)
- **SIMD target** (`--simd`, default: the build machine's, `simd::active().target`). It decides `-march` for the generated unit: `avx2` → `-march=x86-64-v3`, `avx512` → `-march=x86-64-v4`, anything else → no `-march`. `--march M` overrides it; `--march none` compiles for the baseline. When no `-march` is applied the recorded target is `scalar`.
- On x86 `-march` is always paired with `-ffp-contract=off`, which is what keeps the generic code bit-identical to the FORGE build (see below). If the FORGE build itself was configured with `-m...` flags in `CMAKE_CXX_FLAGS`, no flags are added: the unit is compiled exactly like the build.
- The library records its target and refuses to load (`check()`) on a CPU or FORGE build where `simd::by_name(target)` is not usable, for example an `avx512` library on an AVX2 machine. A loader that cannot use the code falls back to run-time plans with a warning.
- Thread count is not part of a target: plans do not depend on it.
- KV codec and sequence lengths are per plan (`--kv-codec`, `--shapes`). The weight codec is the model file's.

## What is bit-identical
- **Plan mode**: the instantiated plan has the same graph, memory plan and kernels as the plan `compile()` builds, so outputs are equal bit for bit, on every SIMD target. Tested for f32, f16, q8_0, q4_k, q4_0, q5_1, iq4_nl, q3_k and tq2_0 weights, KV codecs f32 / int8 / fp8 / tbq4 / k8v4, paged and preallocated KV state, the Qwen3.5-style recurrent model, a `one_shot` model, and the golden Mixtral and DeepSeek-V2 (MLA + MoE) models with f32, q8_0 and q4_k weights.
- **Native mode**: equal bit for bit to the interpreted plan in the same process environment, when the unit is compiled by the compiler and with the flags of the FORGE build (the default). The direct calls repeat the bodies of the CPU lowerings in `kernels.hpp` (same kernel, same chunking); `tests/test_aot_native.cpp` compares logits and generated tokens.
- **Another compiler than the one that built FORGE** (`--cxx g++` against a Clang build, for example): the generated unit's generic code (RoPE, the scalar parts of attention, state ops, reference kernels) is then compiled by that compiler, while the SIMD kernels stay the build's object code. Two things can differ between compilers. Floating-point contraction: GCC defaults to `-ffp-contract=fast`, Clang to `on`; `forge aot` therefore adds `-ffp-contract=off` whenever the compiler is not the build's, and prints a note. On x86 that restores the build's behaviour exactly (its generic code is compiled for baseline x86-64, which has no FMA to contract into). On AArch64 and other targets with fused multiply-add in the baseline, the build's own generic code *is* contracted by its compiler, so a foreign compiler with contraction off can differ in the last bit. Vectorisation: neither compiler may reorder floating-point reductions without `-ffast-math`, so auto-vectorisation does not change results. Bit-identity is tested only for the build's compiler (Clang 23 here); with another compiler treat it as unverified and compare outputs yourself.
- Not guaranteed either: extra floating-point flags via `--cxxflag` (`-ffast-math`, `-ffp-contract=fast`, `-Ofast`).

## Security of precompiled plans
A plan blob is untrusted input: it arrives inside a model file or next to it. The rules:

- **Memory safety does not depend on trusting the blob.** Nothing in it is used as an address. Arena offsets are checked against the arena; buffer sizes are recomputed from shapes that are re-inferred from the op spec; liveness is recomputed from the graph and no two live buffers may overlap; an alias is accepted only where the memory planner makes one (a reshape of its input, or an element-wise op over a buffer that dies at that op); weights are resolved by name and must have the stored shape and codec; a lowering must be registered for that op under that name. A state slot may be used by one node per plan, and a slot the session already holds must have the same kind and row layout. The attention kernel's KV codec attributes are recomputed from the session's state store, not read from the blob. `tests/test_aot.cpp` (`aot_hostile_plans_are_rejected`) alters a real plan in 22 ways and checks each is rejected; `fuzz/fuzz_aot.cpp` fuzzes the parser and the checks behind the hash.
- **What a blob still decides** is which valid graph runs, the same thing the blueprint decides. The key detects mismatch, not malice: someone who can rewrite the file can compute a matching key. Integrity therefore comes from the signature, as for the rest of the file.
- **Signed files.** The `aot` section is an ordinary section: a signature made after `forge aot` covers it. Adding, changing or removing the section afterwards changes the signed digest. `forge aot` on a signed file needs `--key` (sign again) or `--drop-signature`. A file whose signature does not cover its `aot` section is refused by strict loading (`FORGE_REQUIRE_SIGNED=1`, `FORGE_TRUSTED_KEYS`, a policy with `integrity.models required`), like any other modified file; without strict loading the model loads and the section is ignored. That check does not re-hash the weights: the signed digest is rebuilt from the section table with only the `aot` section's hash recomputed.
- **Sidecars** are covered by nothing. A sidecar is ignored for a signed model and under strict loading.
- **Compiled libraries are code.** Loading `model.so` runs it, like any plugin. A compiled library is refused under strict loading and for a signed model, since no model signature covers it. A library stays loaded until the process exits.
- The plan blob's own BLAKE3 and the section table hash catch accidental damage (`forge inspect --verify`).

## API
| Item | Header | Notes |
|------|--------|-------|
| `graph::aot::target` | `graph/aot.hpp` | `seq`, `all_rows`, `kv_codec`, `device` |
| `graph::aot::model_key`, `entry_key` | `graph/aot.hpp` | The hashes described above |
| `graph::aot::builder(model_key)` | `graph/aot.hpp` | `add(target, plan)`, `finish(simd_note)` → blob bytes. `serialise_plan(plan)` raises `errc::unsupported` for a device plan |
| `graph::aot::image(blob, check_hash = true)` | `graph/aot.hpp` | Checked directory over borrowed bytes: `entries()`, `find(model_key, target)` |
| `graph::aot::instantiate(image, entry, binder, state_store*, pass_options, max_arena_bytes)` | `graph/aot.hpp` | → `std::unique_ptr<plan>`; raises `forge_exception` on any mismatch |
| `graph::lower_plan(plan&, state_store*, pass_options, pick)` | `graph/plan.hpp` | The lowering half of `compile()`, shared by both paths |
| `graph::aot::codegen` | `graph/aot_codegen.hpp` | `add(target, plan)`, `finish(blob, options)` → the translation unit |
| `graph::aot::rt` | `graph/aot_rt.hpp` | What generated code calls |
| `modelfile::aot::write_section(in, out, blob)`, `write_sidecar`, `covered_by_signature(reader, &why)`, `model_key(reader)` | `modelfile/aot.hpp` | File side. `write_section` keeps every other section at its file offset and drops a signature |
| `inference::aot_compile_plans(model&, aot_build_options)` | `inference/aot_build.hpp` | → `aot_plans{blob, targets}` |
| `inference::aot_build_native(fpk, aot_build_options, aot_native_options)` | `inference/aot_build.hpp` | Generate, compile, link. `errc::unsupported` where no compiler can run |
| `model::aot()`, `model::set_aot(bool)` | `inference/session.hpp` | `aot().stats()`: plans used, native, rejected, origin, last rejection. `set_aot(false)` compiles everything at run time |
| `model(path, verify, lib)` | `inference/session.hpp` | `path` may be a compiled library; `lib` attaches code linked into the program (`aot_native::library::in_image(forge_aot_model_v1_get())`) |
| `forge_aot_model_v1`, `forge_aot_plan_v1`, `forge_aot_value` | `inference/include/forge_aot.h` | ABI of a compiled model, version `FORGE_AOT_ABI_VERSION` 1 |

No function was added to `forge.h`: `forge_open` accepts a compiled library as its path.

A static library (`--emit static`) is used from a program that links FORGE: `auto m = std::make_shared<inference::model>("model.fpk", false, inference::aot_native::library::in_image(forge_aot_model_v1_get()));`.

## Measured
All numbers are from one machine on 2026-11-12: a 4-core x86-64 VM (AVX-512 target), Clang 23.1.6, `RelWithDebInfo`. The machine was shared with other builds at a load average of 22 to 27 throughout, so wall-clock numbers are noisy; the same plan measured twice differs by up to 30 % below. Read them as "no difference larger than the noise", not as a ranking. Nothing here was measured on a real model or a quiet machine.

Models: random-weight Llama-style models from `tests/tiny_model.hpp`, 12 layers. "small": d = 64, f32, 3 MB. "wide": d = 512, FFN 2048, q8_0 weights, 51 MB. A 12-layer model has the same graph size whatever its width, so the plan work is that of a real 12-layer model; the kernel work is not.

**Getting the plans, in process** (`test_aot`, `aot_startup_and_throughput`; best of 9; two plans: prefill 16 and decode 1, 738 nodes and 512 steps together; plan blob 75 kB):

| | time |
|---|---|
| compiled at run time (blueprint build, passes, memory plan, lowering) | 1.36 ms |
| from precompiled plans (parse, checks, lowering) | 0.46 ms |
| opening the model (map, manifest, blueprint, tokenizer) | 0.27 ms |

So plan mode removes about 0.9 ms of start-up for these two plans, roughly two thirds of the plan work. FORGE's run-time compile was already cheap.

**Whole process** (`forge run <model> --tokens <16 ids> -n 1` for start-up, best of 25 runs for "small" and 12 for "wide"; `-n 200` for decode, best run):

| | small: start-up | small: decode | wide: start-up | wide: decode |
|---|---|---|---|---|
| `forge run model.fpk` (plans compiled at run time) | 6.6 ms | 1608 tok/s | 73 ms | 78 tok/s |
| `forge run` on the file with precompiled plans | 6.5 ms | 1240 tok/s | 70 ms | 57 tok/s |
| `forge run model.so` (native library) | 8.0 ms | 1471 tok/s | 56 ms | 82 tok/s |
| `./model` (native executable) | 5.8 ms | 1332 tok/s | 66 ms | 94 tok/s |

The first two rows run the identical plan through the identical executor, so their decode difference (1608 vs 1240, 78 vs 57) is the measurement noise. Within that noise native code is neither faster nor slower at decoding. This is the expected result for the design described above: the kernels are the same object code, and the per-step overhead that native mode removes is small next to them. The native library's extra start-up on the small model is its own thread pool being created (see limits).

**Build time**: generating the unit takes about 0.1 s; compiling it took 37 to 48 s for a shared library and 133 to 147 s for the executable (which also compiles the tokenizer, samplers and grammar code) on the loaded machine. Sizes: shared library 2.7 MB, executable 7.5 MB, without weights.

## Limits
- **CPU only.** A device plan (Vulkan, CUDA, HIP, OpenCL, Level Zero, Metal, RPC, `pipe:`) holds kernels and device hooks that the backend creates per process, and the weight residency decisions depend on the device's memory budget, so device plans are not serialised. `forge aot --device X` and `FORGE_DEVICE=X forge aot` fail with that reason. At run time a model on a device ignores precompiled plans and compiles as before.
- **LoRA adapters** are not baked in. `forge aot --adapter` is refused; a session that selects an adapter compiles its plans at run time, and the base model keeps using the precompiled ones.
- **Sequence lengths are exact**, as in the plan cache. There is no bucketing or prompt chunking.
- `FORGE_DECODE_WEIGHTS=1` (decode every weight to f32 up front) disables precompiled plans.
- **Diffusion and streaming sessions** (`diffusion.hpp`, `streaming.hpp`) have no plans of their own: they drive a callable. A model used as that callable through `session::infer` or `session::forward` is precompiled like any other.
- **Embedding weights** needs `.incbin` (GCC or Clang) and stays under `--embed-limit-mb` (default 1 GiB): the embedded file lives in read-only data of the default (small) code model, which cannot address 2 GB. Larger models keep their weights in the `.fpk`. An encrypted model cannot be embedded.
- **Native mode needs a compiler at build time** and the build description of a FORGE build tree. On iOS, or where a process cannot be spawned, it fails with `errc::unsupported`; plan mode works everywhere. MSVC's `cl.exe` is not driven; use Clang on Windows.
- The pass pipeline version is a constant that has to be bumped by hand when the passes, the memory planner or the lowering selection change. A stale plan that still validates would run the old (valid) graph, so the FORGE version is part of the key as well.
- A compiled **shared** library has its own copy of the header-only runtime, including its own thread pool: loading it into a FORGE host creates a second set of worker threads (a few milliseconds once, and two pools in one process). The executable and a static library linked into a FORGE program have one copy.
- **Model families.** Tested on the tiny models of the golden tests, both modes bit-identical to run-time compiled plans (`test_aot` `aot_golden_template_families`: plan mode, 26 models; `test_aot_native` `native_golden_template_families`: one library each for Gemma-2, a RoPE-scaling model, Qwen2 q8_0 and BERT; and by hand through `forge aot` / `forge run` / `forge embed` for 25 models in both modes, 2026-11-12): GPT-2, Gemma 1 / 2 / 3 (`attention_bias` soft-capping and sliding window), Phi-3 / Phi-4-mini, Qwen2 (biases), the RoPE frequency-scaling variants (opset 2 attrs), and the embedding sessions BERT / RoBERTa, ViT / CLIP / SigLIP and the Whisper encoder. Nothing had to be changed in the plan format or the native generator for them. `forge embed` reports precompiled plans on stderr like `forge run` (`[aot: 1 of the plans were precompiled ...]`); an embedding model's plans are per exact token count (`--shapes`), and an image or audio encoder has one plan.
- Weight codecs: a plan refers to a weight by name and codec name, so every codec in `graph/codecs.hpp` is handled the same way; which kernels read it is decided by `lower_plan` at load, as at run time. Tested: f32, f16, q8_0, q4_k, q4_0, q5_1, iq4_nl, tq2_0, and q3_k on rows its 256-element block does not divide (the decoded-f32 path).
- `forge aot` copies every section it does not know through unchanged and at its old offset; only `aot` (replaced) and `signature` (dropped) are touched.

Not run: cross compilation with `--target` (the build description's libraries must be for that target); a compiler other than the one that built FORGE (GCC against a Clang build); Mach-O and PE/COFF output (`--embed-weights` uses different assembler directives there); native mode on AArch64, RISC-V, macOS and Windows; `--emit static` beyond producing the archive (no program was linked against it); native mode against an installed tree (`<prefix>/share/forge/forge_aot_sdk.txt` is installed by the `dev` component with paths relative to the prefix, and found next to `../bin/forge`, but no install was made and tried); embedding a file close to the limit; models larger than the tiny test models. Sanitizers: `test_aot` ran under ASan + UBSan on 2026-11-12 without a report ([STATUS.md](STATUS.md)); a sanitizer run of `test_aot_native` and of `fuzz_aot` is not recorded.
