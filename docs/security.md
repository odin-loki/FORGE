# Security model

What FORGE enforces today, where, and what is only planned. Every statement below points at code.

## Script capabilities (interpreter-level)
- `struct Caps { fs, net, proc, env, reflect }` in `script/src/interp.hpp`; all default on.
- Narrowed by `forge::sandbox(opts, fn)` (`script/src/builtins.cpp`, can only turn flags off) and by
  `forge script --no-fs / --no-net / --no-proc` (`apps/forge.cpp`).
- Built-ins that touch files, network, processes, devices or reflection check the flag first and raise
  `PermissionError` (helpers `need()` in `builtins.cpp`, `need_dev()` in `mobile.cpp`, `need_reflect()` in
  `reflect.cpp`; also `runtime.cpp` for processes and `import`, `ai.cpp`, `remote.cpp`, `text.cpp`,
  `automation.cpp`). About 59 such checks in `script/src`.
- **Limit:** these are checks inside the interpreter. They are not an OS boundary: a bug in the
  interpreter or in a native built-in could bypass them. Nothing in `script/` calls `os/sandbox.hpp`.

## OS sandbox (`os/include/forge/os/sandbox.hpp`)
`sandbox_policy { deny_network, deny_exec, deny_fs_write }`, `apply_sandbox`, `run_sandboxed`, `sandbox_status`.
| Platform | Mechanism |
|---|---|
| Linux / Android | `no_new_privs` + seccomp-BPF returning EPERM for denied syscalls (`unsupported` on architectures without an audit arch) |
| FreeBSD | Capsicum `cap_enter()` (coarser than requested) |
| OpenBSD | `pledge()` |
| macOS | `sandbox_init` named profiles, looked up with `dlsym` |
| Windows | in-process denial unsupported; `run_sandboxed` starts children with `CreateRestrictedToken` + a job object |
Tested on Linux in `tests/test_os_ext.cpp`; other platforms are compile-checked only.

## Plugins
Native plugins (`os::library`) are refused unless fs, net, proc and env are all enabled
(`builtins.cpp`, "native plugins are refused in sandboxed runs").

## AI-drafted code
`forge::ai_run` / `forge::ai_eval` run drafted code only if a confirm callback (per call `{"confirm", fn}` or
the engine's `confirm`) returns true; with neither, `confirmed()` returns false and nothing runs
(`script/src/ai.cpp`). The CLI prompts `[y/N]`. `ai_run` goes through the normal shell built-ins, so
capability flags and child limits still apply.

## RPC
Token and Ed25519 key auth, optional TLS (`flow/include/forge/flow/rpc_secure.hpp`, docs/rpc.md).
Remote scripts (`rpc-serve --allow-scripts`) are refused at server construction without auth
(`flow/include/forge/flow/rpc.hpp`). Remote scripts run one at a time with a timeout (`--script-timeout`,
default 60000 ms) and optional `--script-no-proc/-fs/-net` (`apps/forge.cpp`). Without auth on a non-loopback
bind the server prints a warning but still serves non-script RPC.

## Signing
`.fpk` model packages: Ed25519 signatures over the section-table digest (`pack/include/forge/pack/sign.hpp`,
`verify_signature`); with no trusted keys any valid signature passes (integrity only). Encryption uses
XChaCha20-Poly1305 (vendored Monocypher). Release artifacts: `tools/release.py` (docs/RELEASING.md).
Secrets at rest: `os/include/forge/os/keystore.hpp`.

## Exceptions as a security boundary
See docs/scripting.md (Errors): `Interp::from_cpp` maps C++ exceptions to script exceptions and `engine`
catches all of them, so an embedder never sees a C++ exception from `run`/`eval`.

## Known gaps / not yet implemented
- On `main`, scripts are not OS-sandboxed. On `forge-script-os-sandbox` the engine applies the OS sandbox (row
  below); OS denials map to `SandboxViolationError` only when the active policy denies that kind of operation.
- `limit {}` block syntax (only `forge::limit(opts, fn)` exists), no allocation/handle/child-count limits.
- No escape tests for the capability checks (the parser is fuzzed by `fuzz/fuzz_script.cpp`).
- RPC: no mTLS or per-key permissions (docs/STATUS.md, Deferred).

## Pending features (not on `main` yet)

Everything above describes `main`. The following work is on feature branches and is **not merged**; do not rely on it from `main`.

| Feature | Branch | What it adds | State |
|---|---|---|---|
| ForgeScript OS sandbox on by default | `forge-script-os-sandbox` | When any of fs/net/proc is disallowed, the engine also applies `os::apply_sandbox` (deny network, deny starting processes, deny file writes); mode `off`/default/`require`, where `require` fails the run if the sandbox cannot be applied; exceptions: Python-style `OSError` hierarchy (`PermissionError`, `TimeoutError`, `OutOfMemoryError`, `CancelledError`, `DeadlockError` and the file/connection errors are all `OSError` subclasses) and `SandboxViolationError` (a `PermissionError`); `ProcessPool` workers run in sandboxed processes; test `tests/test_script_os_sandbox.cpp` | Compiles on Linux; non-Linux paths untested |
| Typed OS exceptions | `forge-script-os-sandbox` | `os/include/forge/os/oserror.hpp`: errno / Win32 / Capsicum errors, sandbox violations and fault signals become typed, catchable ForgeScript exceptions; test `tests/test_script_oserror.cpp` | Compiles on Linux; non-Linux paths untested |
| Generic Ed25519 bundle signing | `forge-bundle-sign` | `pack/include/forge/pack/bundle.hpp`: detached `sign_bytes`/`verify_bytes` and signed bundles (`FORGE-SIGNATURE.json`); test `tests/test_bundle_sign.cpp` | Compile-checked only, tests not run |
| Qwen 3.5 models | `qwen35` (PR #13) | Gated DeltaNet + gated full attention (`flow/state_ops.hpp`, `import/templates/qwen3_5.fbp`) | Open PR |
| Building FORGE as a subproject | local fix, not pushed | Lets FORGE build inside another project's CMake build | Not on remote |

Known gaps still open on those branches: nested `sandbox{}` blocks are interpreter-checked only; on Windows only process creation is blocked.

## Static capability checks

Before a script runs, the static checker (docs/scripting.md "Static checks") reports every direct call to an
`fs`, `net`, `proc`, `service` or environment builtin, `run()` and every shell block that the run's capabilities,
or an enclosing literal `sandbox({...}, fn)` block, deny. These are errors, so by default the script is refused
before any of it executes rather than failing part-way with PermissionError. It is a convenience layer: indirect
calls (a builtin stored in a variable, `eval`, imported modules) are not seen statically, and the interpreter's
capability checks and the OS sandbox remain the enforcement.
