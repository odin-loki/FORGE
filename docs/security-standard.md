# FORGE Security Standard (FSS) v1 — draft

**Status: draft; the core API exists, FORGE itself does not use it yet.** As of 2026-11-05 the `forge::security`
module ([security-api.md](security-api.md)) provides the policy file and profiles, `enforce()`, `safe_join` /
`canonical_within`, the audit stream, `lockdown()`, `check()`, the `security_error` family and
`forge security check`. No FORGE program calls `lockdown()` or routes its operations through `enforce()` yet (script
built-ins, `forge-serve`, `rpc-serve` and loaders keep their own checks), and there is no `FORGE_HARDENING` CMake
option in this module. What FORGE enforces outside the module is in [security.md](security.md). The "Today" column
says what exists for each control; everything else is a requirement for future work.

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
cannot apply, `plugin_load` is refused when a required check cannot run) and P6 (`audit`) for code that uses it. No
profile is active unless a policy file or `FORGE_SECURITY_PROFILE` says so, and script capabilities are still default
**allow** (all `Caps` flags start on) and narrowed by flags or `forge::sandbox`. P1 and P3 are not met for FORGE's
own programs.

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

## Controls

### Sandboxing (FSS-SBX)

| ID | Requirement | Today |
|----|-------------|-------|
| SBX-1 | Servers and script hosts call `lockdown(policy)` before handling untrusted input. | `security::lockdown` exists; no FORGE program calls it yet. `forge script` applies `os::apply_sandbox` when `--no-fs/--no-net/--no-proc` is given; `rpc-serve` script children do the same. `forge-serve` applies none. |
| SBX-2 | Order: rlimits → no-new-privs → privilege drop → fs confinement → syscall confinement; each irreversible. | `lockdown` follows this order: `setrlimit` / job object limits, `PR_SET_NO_NEW_PRIVS` (FreeBSD `procctl`), a root check instead of a privilege drop (FORGE never changes uid), Landlock write rules, `os::apply_sandbox`. Child rlimits exist in `os::spawn_spec`. |
| SBX-3 | Backends: Linux seccomp + Landlock + caps; FreeBSD Capsicum; OpenBSD pledge/unveil; macOS sandbox_init; Windows mitigation policies, job object, AppContainer; iOS/Android app sandbox. | seccomp + Landlock (write / exec / TCP denial in `os::apply_sandbox`; per-root write rules in `lockdown`), Capsicum `cap_enter`, `pledge` (no `unveil`), `sandbox_init`, Windows job object / AppContainer / restricted token (`os/sandbox.hpp`); `lockdown` adds Windows mitigation policies (ACG, extension points, image load; strict handle checks in strict). No capability dropping. `lockdown` does not use Capsicum (all-or-nothing) on the current process. |
| SBX-4 | Missing mechanism in `strict` ⇒ `SandboxError` unless emulated. | `lockdown` throws `sandbox_error` (before applying anything) when a required control is missing, or emulated with `"emulation": "deny"`. The emulation layer is `enforce()`, which covers only its callers. `--os-sandbox require` aborts a script if the sandbox cannot be applied. |
| SBX-5 | No root/Administrator in `strict` unless `allow_root`. | `lockdown` refuses root / an elevated token when `allow_root` is false (the strict default). |
| SBX-6 | Report a missing MAC label. | `check` / `lockdown` report the Linux SELinux / AppArmor label (`/proc/self/attr/current`). Other OSes: not checked. |

### Capability enforcement (FSS-CAP)

| ID | Requirement | Today |
|----|-------------|-------|
| CAP-1 | Every privileged operation goes through `enforce(capability, target)`. | `security::enforce` / `try_enforce` exist (path-, host- and port-scoped, audited). FORGE's own operations do not call them yet: script built-ins check `Caps` flags (fs, net, proc, env, reflect), not path- or host-scoped. |
| CAP-2 | Canonicalise paths, resolve symlinks, reject `..`, NUL, device namespaces. | `security::canonical_within` and `check_relative` (also used by `enforce`). Bundle manifests keep their own `safe_rel_path`. |
| CAP-3 | `safe_join(root, rel)` for untrusted relative paths. | `security::safe_join` / `try_safe_join`; not yet used by FORGE's loaders. |

### Integrity (FSS-INT)

| ID | Requirement | Today |
|----|-------------|-------|
| INT-1 | Verify models, plugins and scripts before load; policy decides if a signature is required. | `.fpk` signatures are checked on request (`forge verify`, `verify_signature`) or at load with `FORGE_REQUIRE_SIGNED` / `FORGE_TRUSTED_KEYS`, which `lockdown` sets when `integrity.models` is required. `.fsc` cache entries can require signatures. Plugins and scripts are not verified; `enforce(plugin_load)` refuses plugins when `integrity.plugins` is required. |
| INT-2 | Ed25519 over BLAKE3 digests; `.fpk` embedded section, detached `.sig` elsewhere. | `.fpk` embedded signature: yes. Directories: signed bundles with `FORGE-SIGNATURE.json`. No detached `.sig` format. |
| INT-3 | Plugins also pass OS code signing (Authenticode, `SecStaticCodeCheckValidity`, IMA). | Not implemented. |
| INT-4 | Integrity failure ⇒ `IntegrityError`, audited, artefact not mapped. | `security::integrity_error` exists; the loaders still raise `forge_exception` (`errc::integrity_error`) and do not audit. |

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
| ERR-1 | `SecurityError` ⊃ `PolicyViolation`, `IntegrityError`, `PermissionDenied`, `SandboxError`, also as script classes. | C++: `security_error` ⊃ `policy_violation`, `integrity_error`, `permission_denied`, `sandbox_error`, derived from `os::os_exception`. Scripts: `permission_denied` maps to `PermissionError`, `sandbox_error` to `SandboxViolationError`; the other script classes do not exist yet. |
| ERR-2 | `os::error` codes in a `0x5EC0x` range. | `0x5EC01`-`0x5EC04` (`security::sec_code`); `security::classify` maps them, plain `os::classify` reports `other`. |
| ERR-3 | Security errors are never swallowed. | Not audited. |
| ERR-4 | No secrets in messages. | Not audited. |

### Audit (FSS-AUD)

| ID | Requirement | Today |
|----|-------------|-------|
| AUD-1 | Events to journald/syslog, `os_log`, Windows Event Log, logcat, plus a JSON-lines file. | `security::audit` writes to syslog (POSIX; journald) or the Windows event log, the FORGE logger (module `security`), callbacks and a 0600 JSON-lines file. No `os_log` or logcat sink. Only `forge::security` itself emits events so far. |
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
