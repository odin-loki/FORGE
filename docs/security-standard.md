# FORGE Security Standard (FSS) v1 — draft

**Status: draft; the core API exists and FORGE's own programs use it when a policy is active.** As of 2026-11-05 the
`forge::security` module ([security-api.md](security-api.md)) provides the policy file and profiles, `enforce()`,
`safe_join` / `canonical_within`, the audit stream, `lockdown()`, `check()`, the `security_error` family and
`forge security check`. With an active policy, `forge script` derives its capabilities from it, scopes the script
built-ins' file, network, process and environment operations through `enforce()` and calls `lockdown()` before
running; `forge-serve` and `forge rpc-serve` (and its script children) call `lockdown()` after binding and loading;
model loads check `fs.read` and `integrity.models`; native plugins and script files (including imports and
`rpc-serve` remote scripts) are checked against `integrity.plugins` / `integrity.scripts` with detached `.sig`
signatures, and plugins against OS code signing when `integrity.os_code_signing` is set
([security-api.md, "Integration"](security-api.md#integration)). With no policy (profile `none`) all of this is
skipped. There is no `FORGE_HARDENING` CMake option in this module. What FORGE enforces outside the module is in
[security.md](security.md). The "Today" column says what exists for each control; everything else is a requirement
for future work.

**Deviation from this draft: no policy means no profile.** Without a policy file and without `FORGE_SECURITY_PROFILE`,
`load_policy()` returns profile `none` ("no FSS profile active", FORGE's behaviour before the standard), not `strict`.
P3 ("default profile `strict`") is therefore not met; flipping the default for every FORGE program is a separate
decision.

The key words MUST, SHOULD and MAY are as in RFC 2119 / RFC 8174. Controls are modelled on OWASP ASVS 4.0,
SEI CERT C++, NIST SP 800-218 (SSDF) and the CIS Benchmarks.

## Principles

| ID | Principle | Requirement |
|----|-----------|-------------|
| P1 | Default deny | Every capability (fs, net, exec, plugins, dynamic code, env, credentials) is denied unless the policy grants it. |
| P2 | Least privilege | Drop unneeded privileges early; never regain them. |
| P3 | Secure defaults | Default profile `strict`; weakening is explicit and audited. |
| P4 | Fail closed | If a control cannot be applied, refuse the operation. |
| P5 | One standard | One policy, equivalent behaviour on every OS; a missing native mechanism is emulated or, in `strict`, refused, and reported. |
| P6 | Auditability | Denials, weakenings and integrity failures go to the OS audit facility. |

Today: `forge::security` implements P1 (profiles deny by default), P4 (`lockdown` refuses a required control it
cannot apply, `plugin_load` is refused when a required check cannot run, a model failing `integrity.models:
required` is not loaded) and P6 (`audit`). FORGE's programs follow P1 when a policy is active: `forge script` turns
net / proc / env off unless the policy grants them and confines file operations to its roots. No profile is active
unless a policy file or `FORGE_SECURITY_PROFILE` says so; without one, script capabilities are default **allow** (all
`Caps` flags start on) as before. P3 (strict by default) is not met.

## Profiles

| Control | strict (default) | standard | permissive |
|---------|------------------|----------|------------|
| OS sandbox | required; missing ⇒ emulate or refuse | should; missing ⇒ emulate + warn | may |
| Filesystem | read cwd + model/plugin dirs; write cwd + private temp | read any; write cwd + temp | any |
| Network | none (loopback for servers) | loopback + listed hosts | any |
| Process exec | denied | allowed, audited | allowed |
| Dynamic code / JIT pages | denied (W^X) | allowed | allowed |
| Models (.fpk) | valid signature by a trusted key | verified when signed; unsigned audited | unchecked |
| Native plugins | signed by a trusted key + OS code signing | signed | allowed, audited |
| Scripts | unsigned: no net/proc/env, fs confined | full caps, audited | full caps |
| Credentials | OS store or encrypted file | same | plaintext forbidden |
| Resource limits | no core dumps; fds, address space, CPU per policy | no core dumps | unchanged |

Today: the three profiles exist as policy defaults (`security::profile_defaults`, table in
[security-api.md](security-api.md#policy)) and are applied by `lockdown()` / `enforce()` for programs that call them.
"strict (default)" is not the default yet (see the status note). The credentials row is not part of the policy.
The scripts row: `forge script` under a policy runs scripts with the policy's capabilities, so `strict` gives no net /
proc / env and a confined file system; `standard` and `permissive` give what their policy grants, audited. Signed
scripts do not get more capabilities; a signature only decides whether a script may run (`integrity.scripts`).

## Controls

### Sandboxing (FSS-SBX)

| ID | Requirement | Today |
|----|-------------|-------|
| SBX-1 | Servers and script hosts call `lockdown(policy)` before handling untrusted input. | With an active policy: `forge script` (and `rpc-serve` script children) lock down before running the script; `forge rpc-serve`, `forge-serve` and `forge-serve proxy` after binding the listener and loading, before the first request (audited; controls that are not `ok` are printed). Linux: the seccomp filter is synchronised to threads that already exist (TSYNC); Landlock write rules bind only the locking thread and threads started after it. Without a policy: `forge script` applies `os::apply_sandbox` for `--no-fs/--no-net/--no-proc` as before, `rpc-serve` script children likewise; `forge-serve` applies none. |
| SBX-2 | Order: rlimits → no-new-privs → privilege drop → fs confinement → syscall confinement; each irreversible. | `lockdown` follows this order: `setrlimit` / job object limits, `PR_SET_NO_NEW_PRIVS` (FreeBSD `procctl`), a root check instead of a privilege drop (FORGE never changes uid), Landlock write rules, `os::apply_sandbox`. Child rlimits exist in `os::spawn_spec`. |
| SBX-3 | Backends: Linux seccomp + Landlock + caps; FreeBSD Capsicum; OpenBSD pledge/unveil; macOS sandbox_init; Windows mitigation policies, job object, AppContainer; iOS/Android app sandbox. | seccomp + Landlock (write / exec / TCP denial in `os::apply_sandbox`; per-root write rules in `lockdown`), Capsicum `cap_enter`, `pledge` (no `unveil`), `sandbox_init`, Windows job object / AppContainer / restricted token (`os/sandbox.hpp`); `lockdown` adds Windows mitigation policies (ACG, extension points, image load; strict handle checks in strict). No capability dropping. `lockdown` does not use Capsicum (all-or-nothing) on the current process. |
| SBX-4 | Missing mechanism in `strict` ⇒ `SandboxError` unless emulated. | `lockdown` throws `sandbox_error` (before applying anything) when a required control is missing, or emulated with `"emulation": "deny"`. The emulation layer is `enforce()`, which covers only its callers. `--os-sandbox require` aborts a script if the sandbox cannot be applied. |
| SBX-5 | No root/Administrator in `strict` unless `allow_root`. | `lockdown` refuses root / an elevated token when `allow_root` is false (the strict default). |
| SBX-6 | Report a missing MAC label. | `check` / `lockdown` report the Linux SELinux / AppArmor label (`/proc/self/attr/current`). Other OSes: not checked. |

### Capability enforcement (FSS-CAP)

| ID | Requirement | Today |
|----|-------------|-------|
| CAP-1 | Every privileged operation goes through `enforce(capability, target)`. | `security::enforce` / `try_enforce` (path-, host- and port-scoped, audited). Script built-ins call it through the runtime hook `enforce_scope` against the active policy: file reads, writes and listings (path), `net.http`, `net.port_open`, `net.resolve`, `forge::remote*` and the AI endpoint (`host:port`), `run`, `proc.start`, shell blocks and `ProcessPool` (exec), `os.env` / `std::getenv` (env), imports and native plugins. Model loads check `fs_read`; the servers check `net_listen` (the proxy also `net_connect`). Not routed through it: the C++ API used directly, importers, the `.fsc` cache, KV spill files, open file handles. |
| CAP-2 | Canonicalise paths, resolve symlinks, reject `..`, NUL, device namespaces. | `security::canonical_within` and `check_relative` (also used by `enforce`). Bundle manifests keep their own `safe_rel_path`. |
| CAP-3 | `safe_join(root, rel)` for untrusted relative paths. | `security::safe_join` / `try_safe_join`; not yet used by FORGE's loaders (script paths go through `canonical_within` in `enforce`). |

### Integrity (FSS-INT)

| ID | Requirement | Today |
|----|-------------|-------|
| INT-1 | Verify models, plugins and scripts before load; policy decides if a signature is required. | Models: `inference::model` reads the active policy. `integrity.models` `required` verifies the hashes and the embedded Ed25519 signature against `integrity.trusted_keys` (any valid signature when the list is empty) before the blueprint or any tensor is used; `optional` verifies signed files and audits unsigned ones. `FORGE_REQUIRE_SIGNED` / `FORGE_TRUSTED_KEYS` (also set by `lockdown`) still apply and add keys. `forge verify` / `verify_signature` on request; `.fsc` cache entries can require signatures. Plugins (`forge::plugin`, `import "x.so"`, `security::open_plugin`): `integrity.plugins` `required` needs `<plugin>.sig` (INT-2) signed by a trusted key, checked on the bytes that are loaded; `optional` verifies signed plugins and audits unsigned ones (`weaken`). `enforce(plugin_load)` alone also refuses a plugin whose `.sig` does not verify under `required`. Scripts: `engine::run_file` (`forge script FILE`) and every import check `<file>.sig` on the buffer they parse; `rpc-serve` children check the signature the client sent with the source (`FRPZ`, `--rpc-sig`) against the script name. `required` refuses unsigned, tampered or untrusted scripts and code without a file (`-e`, REPL); `optional` runs unsigned scripts, audited. |
| INT-2 | Ed25519 over BLAKE3 digests; `.fpk` embedded section, detached `.sig` elsewhere. `.fpk` embedded signature: yes. Directories: signed bundles with `FORGE-SIGNATURE.json`. Other files: detached `<file>.sig` (`forge-detached-sig` v1, JSON): Ed25519 over BLAKE3 of a domain-separated message binding the file's BLAKE3 digest, its size and its base name, several signers per file (`security/integrity.hpp`, [modelfile.md](modelfile.md#detached-signatures)); `forge sign` / `forge verify` produce and check it for any non-`.fpk` file. |
| INT-3 | Plugins also pass OS code signing (Authenticode, `SecStaticCodeCheckValidity`, IMA). | `security::os_code_signature(path)` → state (`signed`, `unsigned`, `invalid`, `n/a`, `error`) and signer. Windows: `WinVerifyTrust` (`WINTRUST_ACTION_GENERIC_VERIFY_V2`, embedded signature, else the system catalog), no UI, revocation from the cache only, `wintrust.dll` loaded from System32. macOS: `SecStaticCodeCreateWithPath` + `SecStaticCodeCheckValidity` (Security.framework through `dlopen`); an ad-hoc signature counts as unsigned (CI only). Linux, FreeBSD and others: no standard mechanism for shared objects, reported `n/a` (Linux notes whether a `security.ima` xattr exists; IMA appraisal is not checked). With `integrity.os_code_signing` a plugin that is not OS-signed is refused (`integrity_error`, audited `INT-3`) on Windows and macOS; on the others the load proceeds and is audited `weaken`. `forge security check` shows INT-3 with the running executable's signature. |
| INT-4 | Integrity failure ⇒ `IntegrityError`, audited, artefact not mapped. | Under an active policy with `integrity.models` not `off`, a failed check throws `security::integrity_error` (script class `IntegrityError`) and is audited (`INT-4`, `fail`). The file is mapped read-only and hashed before the check, but no blueprint, tokenizer or tensor is read from it. Plugins and scripts: the same exception and audit record; a plugin that fails is never passed to `LoadLibrary` / `dlopen`, a script is never parsed. Without a policy the environment path still raises `forge_exception` (`errc::integrity_error`), unaudited. |

### Cryptography and secrets (FSS-CRY)

| ID | Requirement | Today |
|----|-------------|-------|
| CRY-1 | Randomness from the OS CSPRNG; failure is an error. | `os::entropy`: `getrandom`, `arc4random_buf`, `BCryptGenRandom`; returns false on failure. |
| CRY-2 | Keys in locked, non-dumpable memory, wiped on free. | Monocypher `crypto_wipe` on key buffers in signing, RPC and `.fsc` code. `os` has `mlock`/`VirtualLock` helpers, not used for keys. No `secure_buffer`. |
| CRY-3 | Credentials in the OS store; fallback XChaCha20-Poly1305 with an OS-held or Argon2i key; no plaintext files. | `os::keystore`: Keychain, DPAPI, libsecret, Android backend. File fallback is only XOR-padded, not encrypted. |
| CRY-4 | Constant-time secret comparison. | RPC tokens: yes (`ct_equal`). |

### Errors (FSS-ERR)

| ID | Requirement | Today |
|----|-------------|-------|
| ERR-1 | `SecurityError` ⊃ `PolicyViolation`, `IntegrityError`, `PermissionDenied`, `SandboxError`, also as script classes. | C++: `security_error` ⊃ `policy_violation`, `integrity_error`, `permission_denied`, `sandbox_error`, derived from `os::os_exception`. Scripts: `SecurityError` (under `OSError`) ⊃ `IntegrityError`, `PolicyViolation`; `permission_denied` maps to the existing `PermissionError` and `sandbox_error` to `SandboxViolationError`, which keep their place under `OSError` (not under `SecurityError`). |
| ERR-2 | `os::error` codes in a `0x5EC0x` range. | `0x5EC01`-`0x5EC04` (`security::sec_code`); `security::classify` maps them, plain `os::classify` reports `other`. |
| ERR-3 | Security errors are never swallowed. | Not audited. |
| ERR-4 | No secrets in messages. | Not audited. |

### Audit (FSS-AUD)

| ID | Requirement | Today |
|----|-------------|-------|
| AUD-1 | Events to journald/syslog, `os_log`, Windows Event Log, logcat, plus a JSON-lines file. | `security::audit` writes to syslog (POSIX; journald) or the Windows event log, the FORGE logger (module `security`), callbacks and a 0600 JSON-lines file. No `os_log` or logcat sink. Events come from `forge::security` itself and, under a policy, from model loads (INT-1 / INT-4), `forge script` (the unsigned-script decision), script built-in denials (CAP-1) and the servers (SBX-1). |
| AUD-2 | Each event: UTC time, pid, profile, category, decision, control ID, target. | Yes (`audit_record`). |
| AUD-3 | Audit sink failure never unblocks a denial. | `audit()` never throws; sink failures are counted (`audit_failures()`). |

### Files (FSS-FIO)

| ID | Requirement | Today |
|----|-------------|-------|
| FIO-1 | Temp files exclusive-create, 0600, private dir. | `os::temp_file` exists; its guarantees were not reviewed against this rule. |
| FIO-2 | Owner-only permission check on keys, credentials, policies in `strict`. | Not implemented. `.fsc` requires signatures when its directory is not private. |

### Build hardening (FSS-BLD)

| ID | Requirement | Today |
|----|-------------|-------|
| BLD-1 | `FORGE_HARDENING` (default ON): stack protector, `_FORTIFY_SOURCE=3`, PIE, full RELRO, noexecstack, CET/BTI; MSVC `/guard:cf /CETCOMPAT /sdl`. | Not implemented; CMake sets none of these. Existing options: `FORGE_SANITIZE` (ASan/UBSan), `FORGE_TSAN`, `FORGE_REPRODUCIBLE`. |
| BLD-2 | `forge security check` reports hardening. | Reports `forge::build_hardening()` when `core/build_info.hpp` exists, else the compiler's macros (stack protector, `_FORTIFY_SOURCE`, PIE, CET) for the `forge` binary. |

## Policy file

`forge-security.json`, searched in order: `--policy FILE`, `$FORGE_SECURITY_POLICY`, `./forge-security.json`,
the user config dir. `$FORGE_SECURITY_PROFILE` may pick a base profile (audited). No file ⇒ `strict` (draft; the
implementation returns profile `none` instead, see the status note). A file that fails to parse is a `PolicyViolation`.
Implemented by `security::load_policy` / `parse_policy`; field values in [security-api.md](security-api.md#policy).

```json
{
  "profile": "strict",
  "fs": { "read": ["."], "write": ["."] },
  "net": { "mode": "none", "hosts": [] },
  "exec": false, "dynamic_code": false, "env": false, "allow_root": false,
  "integrity": { "models": "required", "plugins": "required", "scripts": "optional",
                 "trusted_keys": ["<64 hex ed25519 pubkey>"], "os_code_signing": true },
  "limits": { "core": 0, "nofile": 1024, "address_space_mb": 0, "cpu_seconds": 0, "nproc": 0 },
  "audit": { "os": true, "file": "" },
  "emulation": "allow"
}
```

Omitted fields take the profile's value. `emulation: "deny"` makes a missing native mechanism fatal.

## Verification

`forge security check [--policy F] [--json]` prints each control's state (`active`, `available`, `emulated`,
`missing`, `n/a`), the backend and a verdict, and exits non-zero if a required control cannot be met.

Today: implemented (`security::check`, `forge security check`). It also reports `failed` (applying a step failed) and
a level per control. With no profile active it reports and exits 0.
