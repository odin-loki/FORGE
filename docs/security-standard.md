# FORGE Security Standard (FSS) v1 — draft

**Status: target design, mostly not implemented.** As of 2026-11-04 there is no `forge::security` API, no
`forge-security.json` policy file, no `forge security check` command, no profiles, no `FORGE_HARDENING` option
and no `SecurityError` exception family. What FORGE enforces today is in [security.md](security.md).
The "Today" column below says what exists for each control; everything else is a requirement for future work.

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

Today: script capabilities are default **allow** (all `Caps` flags start on) and are narrowed by flags or
`forge::sandbox`. P1 and P3 are not met.

## Profiles (planned)

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

Today: none of these profiles exist.

## Controls

### Sandboxing (FSS-SBX)

| ID | Requirement | Today |
|----|-------------|-------|
| SBX-1 | Servers and script hosts call `lockdown(policy)` before handling untrusted input. | No `lockdown`. `forge script` applies `os::apply_sandbox` when `--no-fs/--no-net/--no-proc` is given; `rpc-serve` script children do the same. `forge-serve` applies none. |
| SBX-2 | Order: rlimits → no-new-privs → privilege drop → fs confinement → syscall confinement; each irreversible. | Linux: no-new-privs + seccomp only. No privilege drop, no Landlock. Child rlimits exist in `os::spawn_spec`. |
| SBX-3 | Backends: Linux seccomp + Landlock + caps; FreeBSD Capsicum; OpenBSD pledge/unveil; macOS sandbox_init; Windows mitigation policies, job object, AppContainer; iOS/Android app sandbox. | seccomp, Capsicum `cap_enter`, `pledge` (no `unveil`), `sandbox_init`, Windows job object / AppContainer / restricted token (`os/sandbox.hpp`). No Landlock, no Windows mitigation policies. |
| SBX-4 | Missing mechanism in `strict` ⇒ `SandboxError` unless emulated. | `--os-sandbox require` aborts the script if the sandbox cannot be applied. No emulation layer. |
| SBX-5 | No root/Administrator in `strict` unless `allow_root`. | Not checked. |
| SBX-6 | Report a missing MAC label. | Not implemented. |

### Capability enforcement (FSS-CAP)

| ID | Requirement | Today |
|----|-------------|-------|
| CAP-1 | Every privileged operation goes through `enforce(capability, target)`. | Script built-ins check `Caps` flags (fs, net, proc, env, reflect). Not path- or host-scoped. C++ API callers are not checked. |
| CAP-2 | Canonicalise paths, resolve symlinks, reject `..`, NUL, device namespaces. | Only bundle manifests refuse escaping paths (`bundle.hpp` `safe_rel_path`). |
| CAP-3 | `safe_join(root, rel)` for untrusted relative paths. | Not implemented. |

### Integrity (FSS-INT)

| ID | Requirement | Today |
|----|-------------|-------|
| INT-1 | Verify models, plugins and scripts before load; policy decides if a signature is required. | `.fpk` signatures are checked on request (`forge verify`, `verify_signature`), not enforced at load. `.fsc` cache entries can require signatures. Plugins and scripts are not verified. |
| INT-2 | Ed25519 over BLAKE3 digests; `.fpk` embedded section, detached `.sig` elsewhere. | `.fpk` embedded signature: yes. Directories: signed bundles with `FORGE-SIGNATURE.json`. No detached `.sig` format. |
| INT-3 | Plugins also pass OS code signing (Authenticode, `SecStaticCodeCheckValidity`, IMA). | Not implemented. |
| INT-4 | Integrity failure ⇒ `IntegrityError`, audited, artefact not mapped. | No such error type. |

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
| ERR-1 | `SecurityError` ⊃ `PolicyViolation`, `IntegrityError`, `PermissionDenied`, `SandboxError`, also as script classes. | Scripts raise `PermissionError` and `SandboxViolationError`. No C++ hierarchy. |
| ERR-2 | `os::error` codes in a `0x5EC0x` range. | Not implemented. |
| ERR-3 | Security errors are never swallowed. | Not audited. |
| ERR-4 | No secrets in messages. | Not audited. |

### Audit (FSS-AUD)

| ID | Requirement | Today |
|----|-------------|-------|
| AUD-1 | Events to journald/syslog, `os_log`, Windows Event Log, logcat, plus a JSON-lines file. | The logger has syslog (POSIX), Windows event log, file and callback sinks, and a JSON-lines format (`core/log.hpp`). No security event stream feeds them. |
| AUD-2 | Each event: UTC time, pid, profile, category, decision, control ID, target. | Not implemented. |
| AUD-3 | Audit sink failure never unblocks a denial. | Not implemented. |

### Files (FSS-FIO)

| ID | Requirement | Today |
|----|-------------|-------|
| FIO-1 | Temp files exclusive-create, 0600, private dir. | `os::temp_file` exists; its guarantees were not reviewed against this rule. |
| FIO-2 | Owner-only permission check on keys, credentials, policies in `strict`. | Not implemented. `.fsc` requires signatures when its directory is not private. |

### Build hardening (FSS-BLD)

| ID | Requirement | Today |
|----|-------------|-------|
| BLD-1 | `FORGE_HARDENING` (default ON): stack protector, `_FORTIFY_SOURCE=3`, PIE, full RELRO, noexecstack, CET/BTI; MSVC `/guard:cf /CETCOMPAT /sdl`. | Not implemented; CMake sets none of these. Existing options: `FORGE_SANITIZE` (ASan/UBSan), `FORGE_TSAN`, `FORGE_REPRODUCIBLE`. |
| BLD-2 | `forge security check` reports hardening. | Not implemented. |

## Policy file (planned)

`forge-security.json`, searched in order: `--policy FILE`, `$FORGE_SECURITY_POLICY`, `./forge-security.json`,
the user config dir. `$FORGE_SECURITY_PROFILE` may pick a base profile (audited). No file ⇒ `strict`. A file that
fails to parse is a `PolicyViolation`.

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

## Verification (planned)

`forge security check [--policy F] [--json]` prints each control's state (`active`, `available`, `emulated`,
`missing`, `n/a`), the backend and a verdict, and exits non-zero if a required control cannot be met.
