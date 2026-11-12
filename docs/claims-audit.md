# Docs claims audit (historical record)

This is a record of past audits of documentation claims against the code. It is not a description of the
current code. For current behaviour read the topic docs; for test results read [STATUS.md](STATUS.md).

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
