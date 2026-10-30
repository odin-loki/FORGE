# FORGE Security Standard (FSS) v1

Status: normative. Applies to every FORGE component on every OS: the `forge` CLI, `forge-serve`,
`forge rpc-serve`, `forge script`, native plugins, and anything that loads models through
`forge::pack`. There is ONE standard and ONE API: `forge::security` (C++) / `forge::security::`
(scripts). Applications never write OS-specific security code; each OS's native mechanism is a
backend behind the same call and the same policy file.

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY and
OPTIONAL are to be interpreted as described in RFC 2119 and RFC 8174.

Modelled on: OWASP ASVS 4.0 (verification levels, V1/V6/V8/V10/V12/V14), SEI CERT C++ (MEM, ERR,
FIO, MSC rules), NIST SP 800-218 SSDF (PO, PS, PW, RV practices), and the CIS Benchmarks
(hardening baselines per OS). Cross-references are given per control as `[ASVS x.y] [CERT ABC00]
[SSDF PW.x] [CIS]`.

## 0. Principles

| ID | Principle | Requirement |
|----|-----------|-------------|
| P1 | Default deny | Every capability (fs, net, exec, plugins, dynamic code, env, credentials) MUST be denied unless the active policy grants it. |
| P2 | Least privilege | A process MUST drop every privilege it does not need as early as possible and MUST NOT regain it. |
| P3 | Secure defaults | The default profile MUST be `strict`. Weakening MUST be explicit (policy file, flag or env) and MUST be audited. |
| P4 | Fail closed | When a control cannot be applied or a check cannot be completed, the operation MUST be refused, not allowed. |
| P5 | One standard | The same policy MUST produce equivalent behaviour on every OS. Where a native mechanism is missing, the implementation MUST either emulate it in-process or, in `strict`, refuse to run; either outcome MUST be reported by `forge security check`. |
| P6 | Auditability | Every policy decision that denies, every weakening, and every integrity failure MUST be audited to the OS audit facility. |

## 1. Profiles

| Control | strict (default) | standard | permissive |
|---------|------------------|----------|------------|
| Unknown/absent policy file | strict | — | — |
| OS sandbox (`lockdown`) | REQUIRED; missing mechanism ⇒ emulate or refuse | SHOULD; missing ⇒ emulate + warn | MAY |
| Filesystem | read: cwd + model/plugin dirs; write: cwd + private temp | read any, write cwd + temp | any |
| Network | none (loopback only for servers) | loopback + explicit hosts | any |
| Process exec | denied | allowed, audited | allowed |
| Dynamic code / JIT pages | denied (W^X) | allowed | allowed |
| Models (.fpk) | MUST carry a valid signature by a trusted key | signature verified when present; unsigned allowed + audited | unchecked |
| Native plugins | MUST be signed by a trusted key AND pass OS code-signing where available | MUST be signed | allowed, audited |
| Scripts | unsigned allowed but run with no net/proc/env and fs confined; signed scripts get the policy's caps | full caps, audited | full caps |
| Credentials | OS store or encrypted file only | same | plaintext file forbidden in all profiles |
| Resource limits | core dumps 0; fds, address space, cpu per policy | core dumps 0 | unchanged |

## 2. Controls

### 2.1 Sandboxing and privilege (FSS-SBX) [ASVS 1.14, 14.1] [CIS] [SSDF PW.9]

- **FSS-SBX-1** A FORGE server or script host MUST call `security::lockdown(policy)` before handling untrusted input.
- **FSS-SBX-2** `lockdown` MUST apply, in order: rlimits (no core dumps) → no-new-privileges → capability / token / privilege drop → filesystem confinement → syscall / API confinement. Each step MUST be irreversible.
- **FSS-SBX-3** Backends: Linux seccomp-bpf + Landlock + `PR_SET_NO_NEW_PRIVS` + capability bounding/effective drop + rlimits (namespaces detected); FreeBSD Capsicum `cap_enter`/`cap_rights_limit` (jail detected); OpenBSD `pledge`/`unveil`; macOS `sandbox_init` + App Sandbox/hardened-runtime entitlement check; iOS App Sandbox (verified); Windows process mitigation policies (DEP, ASLR, CFG, dynamic-code prohibition, image-load policy, strict handles), DLL search hardening, a kill-on-close job object, integrity level + AppContainer detection; Android SELinux context + app sandbox (verified).
- **FSS-SBX-4** If a step's mechanism is absent, `strict` MUST fail with `SandboxError` unless an in-process emulation exists (path/capability guard), in which case the posture MUST report the control as `emulated`.
- **FSS-SBX-5** Processes MUST NOT run as root/Administrator in `strict` unless the policy sets `allow_root`.
- **FSS-SBX-6** A MAC label (SELinux/AppArmor/Android) SHOULD be present; its absence MUST be reported.

### 2.2 Capability enforcement (FSS-CAP) [ASVS 4.1, 12.3] [CERT FIO02]
- **FSS-CAP-1** Every privileged operation MUST pass through `security::enforce(capability, target)`.
- **FSS-CAP-2** Path targets MUST be canonicalised and checked against allow-lists after symlink resolution; `..` escapes, NUL bytes and device namespaces (`\\?\`, `\\.\`) MUST be rejected.
- **FSS-CAP-3** Untrusted relative paths MUST be joined with `security::safe_join(root, rel)`.

### 2.3 Integrity (FSS-INT) [ASVS 10.3, 14.2] [SSDF PS.2, PW.4]
- **FSS-INT-1** Models, plugins and scripts MUST be verified with `security::verify_artifact` before load; the policy decides whether a signature is required.
- **FSS-INT-2** Signatures are Ed25519 over BLAKE3 digests (`.fpk`: embedded `signature` section; other files: detached `<file>.sig` JSON `{"scheme":"forge-detached-v1","key":hex,"sig":hex}`).
- **FSS-INT-3** Native plugins SHOULD additionally pass OS code signing (Authenticode `WinVerifyTrust`, macOS `SecStaticCodeCheckValidity`, Linux IMA `security.ima` xattr when IMA is active). In `strict`, a present-but-invalid OS signature MUST fail; an absent OS facility is reported.
- **FSS-INT-4** Integrity failures MUST raise `IntegrityError`, MUST be audited, and the artefact MUST NOT be mapped or executed.

### 2.4 Cryptography and secrets (FSS-CRY) [ASVS 6.2, 6.3, 6.4] [CERT MSC50, MEM03]
- **FSS-CRY-1** Randomness MUST come from the OS CSPRNG (`getrandom`, `arc4random_buf`, `BCryptGenRandom`, `SecRandomCopyBytes`); failure MUST be an error, never a fallback to a PRNG.
- **FSS-CRY-2** Key material MUST live in `security::secure_buffer`/`secure_arena`: locked (mlock/VirtualLock), excluded from core dumps where supported (`MADV_DONTDUMP`), and zeroised on free with a non-elidable wipe.
- **FSS-CRY-3** Credentials MUST be stored via `security::credentials` (Keychain, DPAPI/Credential Manager, libsecret, kernel keyring, Android Keystore). The fallback MUST be XChaCha20-Poly1305 encrypted with a key held in the OS store or derived with Argon2i from a passphrase. Plaintext credential files are forbidden.
- **FSS-CRY-4** Comparisons of secrets MUST be constant time.

### 2.5 Errors and exceptions (FSS-ERR) [ASVS 7.4] [CERT ERR50-ERR62]
- **FSS-ERR-1** Security failures MUST use the taxonomy: `SecurityError` ⊃ {`PolicyViolation`, `IntegrityError`, `PermissionDenied`, `SandboxError`}. The same names exist as script exception classes.
- **FSS-ERR-2** Each maps to an `os::error` code in the `0x5EC0x` range so `os::result` callers can test `security::is_security_error`.
- **FSS-ERR-3** A security error MUST NOT be caught and ignored; a handler MAY translate it only into another security error or a refusal.
- **FSS-ERR-4** Error messages MUST NOT include secret material.

### 2.6 Audit (FSS-AUD) [ASVS 7.1-7.3] [SSDF RV.1]
- **FSS-AUD-1** Events go to the OS facility: journald/syslog (Linux, BSD), `os_log`/syslog (Apple), the Windows Event Log (`ReportEventW`), logcat (Android); plus an optional JSON-lines file sink.
- **FSS-AUD-2** Each event carries time (UTC), pid, profile, category, decision (allow/deny), control ID and target; no secrets.
- **FSS-AUD-3** Failure of the audit sink in `strict` MUST NOT unblock the denied operation.

### 2.7 Files (FSS-FIO) [CERT FIO03, FIO21] [ASVS 12.3]
- **FSS-FIO-1** Temporary files MUST be created with `security::temp_file` (exclusive create, 0600, private dir; `O_TMPFILE` / `FILE_FLAG_DELETE_ON_CLOSE` where available).
- **FSS-FIO-2** Sensitive files (keys, credentials, policies) MUST be checked for owner-only permissions before use in `strict`.

### 2.8 Build hardening (FSS-BLD) [SSDF PW.6] [CIS]
- **FSS-BLD-1** Release builds MUST enable `FORGE_HARDENING` (default ON): `-fstack-protector-strong`, `-D_FORTIFY_SOURCE=3`, `-D_GLIBCXX_ASSERTIONS`, PIE, `-Wl,-z,relro,-z,now,-z,noexecstack`, `-fstack-clash-protection`, `-fcf-protection=full` (x86) / `-mbranch-protection=standard` (arm64); MSVC `/guard:cf /DYNAMICBASE /HIGHENTROPYVA /NXCOMPAT /CETCOMPAT /sdl`.
- **FSS-BLD-2** `forge security check` MUST report which hardening features the running binary was built with.

## 3. Policy file

`forge-security.json`, searched in order: `--policy FILE`, `$FORGE_SECURITY_POLICY`, `./forge-security.json`,
the user config dir (`~/.config/forge/`, `%APPDATA%\forge\`, `~/Library/Application Support/forge/`).
`$FORGE_SECURITY_PROFILE` MAY select a base profile; it is audited. Absent everything, `strict`.
A policy file that fails to parse MUST be a `PolicyViolation` (fail closed, never "fall back to permissive").

```json
{
  "profile": "strict",
  "fs": { "read": ["."], "write": ["."] },
  "net": { "mode": "none", "hosts": [] },
  "exec": false,
  "dynamic_code": false,
  "env": false,
  "allow_root": false,
  "integrity": { "models": "required", "plugins": "required", "scripts": "optional",
                 "trusted_keys": ["<64 hex ed25519 pubkey>"], "os_code_signing": true },
  "limits": { "core": 0, "nofile": 1024, "address_space_mb": 0, "cpu_seconds": 0, "nproc": 0 },
  "audit": { "os": true, "file": "" },
  "emulation": "allow"
}
```

Fields not given take the profile's value. `emulation: "deny"` makes a missing native mechanism fatal
even when an in-process guard exists.

## 4. Verification

`forge security check [--policy F] [--json]` prints the posture: each control's state
(`active`, `available`, `emulated`, `missing`, `n/a`), the backend used, and an overall verdict.
It exits non-zero if the effective profile's REQUIRED controls are not satisfiable.
