# Docs claims audit (historical record)

This is a record of past audits of documentation claims against the code. It is not a description of the
current code. For current behaviour read the topic docs; for test results read [STATUS.md](STATUS.md).

## 2026-11-13: documentation pass over the final integration tree (c452a61)

About 27 branches were merged over 2026-11-12/13; several landed after the pass recorded below, and one (the
hardening pass) was cut short. This pass added [UNTESTED.md](UNTESTED.md), a dated STATUS entry for the final state,
and brought README.md, TODO.md, docs/index.md and the guides in line with the tree. Documentation only; nothing was
built or re-run except the documentation checks (`docs_up_to_date`, `docs_links`, `ci_workflow_copies`, `sbom_spdx`).
Method: README.md, TODO.md, STATUS.md, index.md and testing.md were read in full; of the other guides the
"limits", "not verified" and "platform notes" sections were read, and all of docs/ was searched for the stale
phrases listed under the table. No guide was re-read line by line, so a wrong sentence outside those sections and
phrases can remain; `scripting.md`, `security*.md`, `os.md`, `os-audit.md`, `mobile.md`, `shells.md`,
`loadbalancer.md`, `memory.md` and `configuration.md` were only searched. The 32-bit Arm lines of `simd.md`,
`configuration.md` and `TODO.md` were left alone: they were being changed separately.

| Claim found | Why it was wrong for the final tree | Now |
|---|---|---|
| README, STATUS header: "209 tests, 199 passed, 10 skipped" as the state of the tree | That was the count before the last merges | 247 tests, 237 passed, 10 skipped, 0 failed at c452a61, clang 23.1.6; `slow` and `emu` not run |
| README status: two bullets presenting the PR #53 CI results first | CI has not run on any of the 2026-11-12/13 work; the older results describe another tree | One sentence on what was verified, one on what was not, link to UNTESTED.md; PR #53 is named as history |
| README quick start: `cmake --preset dev` with "a C++23 compiler" | Only clang 23 has built the tree | The command names `clang++-23` / `clang-23` |
| README "What is in the box" and program list | Missing: `denoise`, `stream`, `requant`, `adapter`, `blueprint`, `shard`; tool calling, API keys, TLS; session kinds; shards, chunked encryption, binary blueprint; the Vulkan / Metal ports; `vec<T>`; the security layer; `forge-script-run` | Listed, with how far the GPU ports were checked |
| STATUS Platforms table: CI rows read as the state of the tree; "Clang 18" | CI rows are PR #53; the workflow now uses clang 23 and has never run in that form | A separate row for the final local result; CI rows labelled as the last run; the never-run jobs listed |
| STATUS Known gaps: "graphs recorded before an eviction must not be replayed after it"; no eviction during an OpenCL / Level Zero recording | Buffer generations (2026-11-12): graphs are re-recorded; OpenCL / Level Zero evict during a recording | Rewritten; mock-verified only |
| STATUS: no record of the hardening pass | Its fixes are merged, its full passes did not complete, it left no run table | 2026-11-13 entry lists the eleven fixes and says what did not run |
| TODO "Licence and provenance": establish the `rvv-intrinsic-doc` licence, "must be done before a release"; RELEASING.md "only then ship those two files"; NOTICE "has to be confirmed ... before FORGE is distributed" | Owner decision 2026-11-13: not pursued now | The factual "not established" statement stays in NOTICE and the SBOM; the action wording is gone; TODO lists it under "Decided against" |
| modelfile.md: the IQ grid GGUF types "not loadable" for want of tables, read as a gap | Owner decision 2026-11-13: will not be supported (clean-room; the codebooks will not be copied) | Documented as a deliberate non-goal; TODO "Decided against" |
| TODO "Verification debt": seventeen items duplicating the guides, some overtaken (Vulkan / Metal ports "not done", sanitizers "see the STATUS entry") | Later branches landed after the list was written | A pointer to UNTESTED.md; open features kept with their guides; closed items listed as closed |
| graph.md: "Metal has not been compiled" | The Metal backend is built and tested in the macOS CI job since PR #53; what has never been compiled is the 2026-11-12 ports | Says exactly that |
| graph.md: pipelines "only been run with CPU and lavapipe stages" | `pipe:auto` ran on two devices of a mock driver | Added |
| building.md: "CI builds with g++-14, clang-23, AppleClang, MSVC, MinGW, FreeBSD's clang" | True of the workflow file, but none of them except clang 23 has built the current tree | Says so |
| building.md: "The CI runners have no `glslangValidator`, so Vulkan is compiled out in every CI job" | `linux-vulkan` installs it (since 2026-11-05) | Corrected |
| building.md: "Only `test_simd` has been built and run" under the emulator toolchains | The companion-table and `vec_types.hpp` tests run there too | Corrected |
| building.md: Android "needs NDK r26 or newer" | Configure refuses clang older than 19; r27c is clang 18 | States the conflict; no CMake NDK build of this tree has been run |
| testing.md: `linux-vulkan` runs "the two Vulkan goldens" | The Vulkan golden list grew to thirteen | Corrected |
| testing.md: "`test_rpc_tls` skips on Windows CI (its `openssl` probe redirects to `/dev/null`)" | The probe no longer redirects; the fixed form has not run on Windows | Corrected |
| testing.md: mock and compile-check rows | Missing the determinism, kernel-cache, select, stale and `metal_host_syntax` tests; did not say the CUDA / HIP mocks need a clang with NVPTX / AMDGPU | Added |
| testing.md CI table read as results | The workflow has not run in this form | A sentence before the table |
| capi.md: "other compilers see ABI 4 for the first time in CI" | Reads as if CI had run | "no other compiler has built ABI 4" with the date |
| os-wrapper.md: "All of these passed in CI ... on 2026-11-04" under a table that includes the wave 4 tests | Seven of the listed tests were added on 2026-11-12 | Split by date |
| inference.md: embeddings are "One sequence per call" | `session::embed_batch` takes several inputs (as a loop) | Corrected |
| aot-models.md: "Not run: ... an ASan or UBSan build of the tests" | `test_aot` ran under ASan + UBSan on 2026-11-12 (STATUS) | Corrected; `test_aot_native` and `fuzz_aot` under a sanitizer are not recorded |
| text-locale.md: `test_text` "pass in CI" | The regex additions of 2026-11-12 have not been through CI | Dated |
| modelfile.md: lists two things as written from memory | The `ignore_merges` list, the GGUF conventions of gemma / phi3 / bert / gpt2, the YaRN attention-factor formula and the leave-out lists were too | Added, with a pointer to UNTESTED.md |
| docs/index.md descriptions | Several predate the guides' new sections | Updated |

Searched for and found nothing left to fix outside dated changelog entries: "clang-18" / "clang 18" as the current
compiler, "ABI 3" as the current C ABI, "opset 1..", "62 ops" / "63 ops", "111/111" outside the PR #53 rows, "There
is no chat template", "one handle owns both", "Tier P is f32 only".

Could not be established from the record (written as "not recorded" in UNTESTED.md): which suites of the
interrupted whole-suite ASan pass finished clean; the suite list and counts of the whole-suite ASan / TSan run that
preceded the final merges; whether MinGW or the NDK built anything beyond the OS-layer tests; a command list for the
RTX 3090 (the GPU entries name tests, not commands); any libFuzzer campaign; whether `perf/baseline-x86_64.json` was
compared with this tree; which NDK the CI runner provides.

## 2026-11-12: pass over the tree after the day's merges

About nineteen branches written in parallel were merged on this day. The guides were re-read against the merged code
for claims one branch made that another branch, or the merge itself, had made false. Scope: `README.md`, `NOTICE`,
`TODO.md`, `docs/index.md`, `docs/STATUS.md`, the tokenizer, importer, serving, GPU and AOT sections, and every
relative link and backticked repository path (now checked by `tools/check_docs_links.py`, ctest `docs_links`). Not
re-read line by line: `os*.md`, `scripting.md`, `security*.md`, `support.md`, `simd.md` beyond the lines quoted in
STATUS.

| Claim | Finding | Now |
|---|---|---|
| `NOTICE`: "one vendored third-party library"; `ops/intrinsics/x86.json` "is a hand-curated subset" | The Level Zero header is vendored too; the eight intrinsic databases are derived by scripts from clang 23's headers and, for RISC-V, from the prototype lists of another project whose licence is recorded nowhere | NOTICE lists each source with the licence statement read from it, and says "not established" for the RISC-V source; `tools/sbom.py` carries the same list |
| `os/src/text/ucd_data.inc`: "used under the Unicode License v3 (see NOTICE)" | NOTICE had no Unicode entry | The licence text is in NOTICE |
| `docs/cli.md`, `docs/modelfile.md`: "the runtime tokenizer is byte-level BPE", BERT-family models "need `--tokens`", SentencePiece-BPE files "are not encoded exactly" | True for the branch that wrote it, false after the tokenizer branch merged, and the family importers were not using the new code | The family importers go through the same tokenizer import; the text is rewritten; the family goldens check text to ids to output |
| `verify/golden/tiny_gemma.py`: Gemma-2 q8_0 "cosine 0.9999 on the CPU and CUDA paths, 0.9984 on the OpenCL path"; the bound lowered to 0.997 "for the OpenCL weight path" | The CPU and CUDA value was 0.99900, and the OpenCL value is the exact result of the quantised model: the other two round activations to int8. The model was ill-conditioned (0.945 to 0.9995 over prompts) | Explained in `docs/gpu.md`; the golden is re-conditioned and the bound is 0.999 everywhere |
| `docs/modelfile.md`: opsets are checked (`build_graph` "refuse[s] any other opset") | Only the range was checked; a blueprint could declare opset 1 and use opset-3 features, and the template example in the same guide did | Enforced per feature; the example declares opset 3 |
| `docs/gpu.md` backends table: Metal "never compiled (no Mac)" | Contradicted `STATUS.md` and the same guide's limits section: `test_gpu_metal` passes on the GitHub macOS runner (PR #53) | The row says what ran where, and that the day's Metal changes only passed the MSL syntax check |
| `README.md` status paragraph: CI results of PR #53 presented as the state of the tree; "CI uses ... clang-18" | A large part of the tree is newer than any CI run and was built with one compiler on one machine | The paragraph separates what CI last ran from what has landed since and how that was checked |
| `TODO.md`: "No open items." | The guides hold several dozen "not run" / "never compiled" / "not done" notes | TODO.md indexes them |
| `docs/index.md` links every guide | `os-audit.md` and `os-parity.md` were not linked | Linked; the link checker fails when a guide is missing from the index |
| `docs/security-api.md` → `security-standard.md#policy-file-planned` | The heading is "Policy file" | Fixed |
| `ci/install-workflows.sh` installs the workflow from `ci/github/ci.yml` | That copy (174 lines) and `ci/github-actions-ci.yml` (40 lines) were far behind `.github/workflows/ci.yml` (452 lines): running the script would have replaced the workflow with an old one | The three are one file; ctest `ci_workflow_copies` and `ci_workflow_copy_flat` compare them |
| `docs/inference.md`: `tokenizer()` returns `const modelfile::bpe_tokenizer*` | The class is `tokenizer`; `bpe_tokenizer` is an alias kept for old code | Fixed |
| `STATUS.md` header: "Current as of 2026-11-04" | The changelog below it had entries of 2026-11-12 | The header says which parts are of which date |

## 2026-11-04: full docs rewrite

All docs were rewritten and re-checked against the source at main ccc0a7d. Corrections made in the
security and status docs:

| Old claim | Now |
|---|---|
| No `.fsc` script cache (dropped) | It exists (`script/src/fsc.cpp`, `--cache-dir`); it has no tests. |
| Remote scripts run one at a time | Each runs in its own child process (`forge script --rpc-child`) with timeout and memory cap. |
| FSS v1 (security-standard.md) is normative | Marked as a draft: `forge::security`, the policy file, profiles, `forge security check` and `FORGE_HARDENING` do not exist. Each control now states what exists today. |
| `run_sandboxed` applies the whole policy | On POSIX it drops `deny_exec` for the target so `execvp` can run. |
| OS sandbox run-tested on Linux only | `test_os_ext` sandboxing runs in CI on Linux, macOS, Windows and FreeBSD. |
| macOS only cross-linked; iOS never built; Windows event log sink and dynlib Windows/Apple code never compiled | All built in CI; `test_log` and `test_os_dynlib` pass on macOS and Windows. iOS is build-only. |
| CUDA/HIP/OpenCL/Level Zero/Metal never on hardware | CUDA ran on an RTX 3090; Metal passes on the GitHub macOS runner. HIP, OpenCL, Level Zero still mock-only. |
| CI workflow must be moved by hand | It is at `.github/workflows/ci.yml`. |
| Keystore fallback (implied encrypted) | The file fallback is XOR-padded, not encrypted. (True at ccc0a7d; replaced the next day, in 0b76e23, by XChaCha20-Poly1305 under a store key file. See [security.md](security.md).) |

## 2026-10-31: audit on branch docs-verified (base 5940593), merged in PR #12

Scope: scripting.md, security claims in rpc.md/modelfile.md/os.md, index.md, TODO.md, STATUS.md headlines,
every backticked file path, and `forge script` CLI options. Other docs were not checked line by line.

False claims found and fixed then:

| Claim | Finding then | Status 2026-11-04 |
|---|---|---|
| Scripts sandboxed "using sandbox.hpp" | No OS sandbox in script/src | Now true: `engine.cpp` applies it for `--no-fs/--no-net/--no-proc` (PR #27). |
| `.fsc` bytecode cache | Absent | Now exists. |
| `--mem-limit` for scripts | Absent | Still absent on 2026-11-04. Added 2026-11-12: `forge script --mem-limit SIZE` (`options::mem_limit_bytes`); what it bounds is in docs/scripting.md "Resource limits". |
| `limit {mem: "2G", ...}` block syntax | Absent | Now exists (`limit {...} { body }`, size/duration strings). |
| Recursion depth default 10 000 | It was 800 | See `script/src/interp.hpp` `Limits`. |
| Allocation, handle and child-count limits | Absent | Now exist (`allocs`, `handles`, `procs`). |
| `tests/script/*.fs` goldens | Absent | Added in PR #11. |
| Script parser fuzz target | Absent | Added in PR #11 (`fuzz/fuzz_script.cpp`). |
| `dynlib.hpp` module | Absent | Now exists (`os/include/forge/os/dynlib.hpp`). |
| Scripting is "design, not yet built" | It was built | — |
| No C++ exception escapes the VM | Partly true; the engine catches all, the VM does not | Wording kept as "engine catches all". |

A second pass re-derived the changed sentences from source and found 5 more errors (an example map literal,
lazy REPL compilation wrongly marked false, goldens, fuzz target, `forge::limit` test). Claims verified then
included: capability checks raise `PermissionError`; AI code needs confirmation; plugins are refused in
sandboxed runs; remote scripts need auth; `.fpk` Ed25519 signing.

## 2026-11-12: script memory limit and indirect capability checks

| Claim | Finding | Status |
|---|---|---|
| security.md: "Static capability checks miss indirect calls" | True, and wider than stated: the error-level check saw only the `lib.fn(...)` spelling, so `forge::fs::read(p)` and a `forge::sandbox(std::map<std::string, bool>{{"fs", false}}, ...)` literal were not seen either | `script/src/capflow.hpp` follows callables and reports warnings (docs/scripting.md "Static checks", *Indirect calls*); the error-level check is unchanged. |
| security.md: "Nested non-isolated `sandbox` blocks are interpreter-checked only" | True and inherent: every OS mechanism is one-way for the process | Behaviour kept; the docs now say why. |
| security.md: "`sandbox({isolate: true})` ... unavailable on Windows" | True | Not changed; a re-spawn design is noted under Known gaps. |
| scripting.md: only `limit-option` has a dedicated checker test | Was true | The indirect `capability` warnings now have tests too. |
