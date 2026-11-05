# `forge::security` API

The C++ side of the [FORGE Security Standard](security-standard.md) (FSS). Header-only, target `forge_security`
(`security/include/forge/security/`, umbrella header `security.hpp`). What FORGE enforces outside this module is in
[security.md](security.md).

**Default: no profile.** With no policy file and no `FORGE_SECURITY_PROFILE`, `load_policy()` returns
`profile::none`: nothing is confined, `enforce()` allows everything and audits nothing, `lockdown()` does nothing.
This is FORGE's behaviour before the standard. The draft says "no file ⇒ `strict`"; making strict the default for
every FORGE program is a separate, later decision.

**Not wired in yet.** FORGE's own subsystems (script built-ins, `forge-serve`, `rpc-serve`, model and plugin
loading) do not call `enforce()` or `lockdown()` yet. The module is usable by embedders and by `forge security
check`; routing FORGE itself through it is the next step.

## Headers

| Header | Contents | FSS |
|---|---|---|
| `errors.hpp` | `security_error` ⊃ `policy_violation`, `integrity_error`, `permission_denied`, `sandbox_error`; codes `0x5EC01`-`0x5EC04` | ERR-1, ERR-2 |
| `audit.hpp` | `audit()`, `add_audit_sink`, `set_audit_os`, `open_audit_file`, `audit_record`, `to_json` | AUD-1..3 |
| `path.hpp` | `check_relative`, `safe_join`, `try_safe_join`, `canonical_within` | CAP-2, CAP-3 |
| `policy.hpp` | `policy`, `profile`, `profile_defaults`, `parse_policy`, `load_policy`, `to_json` | Profiles, policy file |
| `enforce.hpp` | `enforce`, `try_enforce`, `capability`, `set_active_policy`, `active_policy`, `expand_root` | CAP-1 |
| `lockdown.hpp` | `lockdown`, `check`, `report`, `control`, `locked_down` | SBX-1..6, Verification, BLD-2 |

## Example

```cpp
#include <forge/security/security.hpp>
namespace sec = forge::security;

int main(int argc, char** argv) {
    auto p = sec::load_policy(argc > 1 ? argv[1] : "");   // throws sec::policy_violation on a bad file
    auto rep = sec::lockdown(p);                           // throws sec::sandbox_error if a required control is missing
    std::fputs(rep.text().c_str(), stderr);
    sec::enforce(sec::capability::fs_write, "out/result.json");   // throws sec::permission_denied, audited
    if (auto s = sec::try_enforce(sec::capability::net_connect, "api.example.com:443"); !s) { /* s.err.what */ }
    auto path = sec::safe_join(model_dir, untrusted_name);  // CAP-3
}
```

## Policy

`forge-security.json` (shape as in [security-standard.md](security-standard.md#policy-file-planned)). Omitted fields take
the profile's values. Load order: `--policy FILE` (the argument to `load_policy`), `$FORGE_SECURITY_POLICY`,
`./forge-security.json`, `<config dir>/forge/forge-security.json` (`$XDG_CONFIG_HOME/forge`, `%APPDATA%\forge`,
`~/Library/Application Support/forge`). The first one that exists is used. A file named by `--policy` or
`$FORGE_SECURITY_POLICY` that cannot be read is an error, never a fallback.

`$FORGE_SECURITY_PROFILE` (`strict`, `standard`, `permissive`) is the base profile for a file without `"profile"`, or the
whole policy when there is no file. Using it is audited (`weaken` unless it is `strict`). A file's own `"profile"` wins.

Parsing fails closed. A syntax error, unknown key, duplicate key, wrong type or bad value throws `policy_violation`
with `file:line:col` (the position of the offending key, or of the JSON syntax error).

| Field | Values |
|---|---|
| `profile` | `strict`, `standard`, `permissive` |
| `fs.read`, `fs.write` | paths; `"*"` = anything; tokens `.` (working directory), `$TMP` (`os::temp_dir()`), `$CACHE` (`$FORGE_CACHE`, else `~/.cache/forge`), `$CONFIG` (`<config dir>/forge`), `~`. Write roots are readable too |
| `net.mode` | `none` (no sockets but `AF_UNIX`; `enforce` still allows loopback listen), `loopback`, `hosts` (loopback + `net.hosts`), `any` |
| `net.hosts` | `host`, `host:port`, `*.suffix[:port]`, `[v6]:port` |
| `exec`, `dynamic_code`, `env`, `allow_root` | booleans |
| `integrity.models/plugins/scripts` | `required`, `optional`, `off` |
| `integrity.trusted_keys` | 64-hex Ed25519 public keys |
| `limits` | `core` (null = unchanged), `nofile`, `address_space_mb`, `cpu_seconds`, `nproc` (0 = unchanged) |
| `audit` | `os` (bool), `file` (JSON-lines path) |
| `emulation` | `allow`, `deny` (a control FORGE can only emulate becomes fatal at level `required`) |

Profile defaults (`profile_defaults`):

| | strict | standard | permissive |
|---|---|---|---|
| fs.read / fs.write | `.`, `$CACHE` / `.`, `$TMP` | `*` / `.`, `$TMP` | `*` / `*` |
| net.mode | none | hosts | any |
| exec, dynamic_code, env | false | true | true |
| allow_root | false | true | true |
| models / plugins / scripts | required / required / optional | optional / required / optional | off / optional / off |
| os_code_signing | true | false | false |
| limits | core 0, nofile 1024 | core 0 | unchanged |
| audit.os | true | true | true |

## Errors

| Class | Code | `os_exception::kind` | Script class today |
|---|---|---|---|
| `policy_violation` | `0x5EC01` | `other` | `OSError` (planned: `PolicyViolation`) |
| `integrity_error` | `0x5EC02` | `other` | `OSError` (planned: `IntegrityError`) |
| `permission_denied` | `0x5EC03` | `permission_denied` | `PermissionError` |
| `sandbox_error` | `0x5EC04` | `sandbox_violation` | `SandboxViolationError` |

All derive from `security_error`, which derives from `os::os_exception`, so existing `catch (os::os_exception&)` sites and
the interpreter's C++-to-script mapping keep working. `security_error::control` holds the FSS control ID,
`fss_name()` the FSS class name. Non-throwing calls return `os::status` whose `err.code` is one of these codes;
`security::classify(err)` maps them to `os_errc` (plain `os::classify` reports `other`), `security::throw_if(err)` throws
the matching class. Script classes `SecurityError`, `PolicyViolation` and `IntegrityError` are a later step; until then
the two existing script classes are the mapping.

## Audit

`audit(category, decision, control, target, detail)` records UTC time (ms), pid, profile, category, decision
(`allow`, `deny`, `weaken`, `fail`), control ID and target. Sinks:

- callbacks (`add_audit_sink`);
- the OS facility, `set_audit_os(true)` (done by `lockdown` when `audit.os`): `syslog(3)` (journald) on POSIX, the
  Windows event log (source `forge-security`). Called directly, so a `FORGE_LOG` level cannot hide it. Android logcat
  and Apple `os_log` go through syslog only where it exists;
- the FORGE logger, module `security` (deny / weaken at warn, fail at error, allow at info);
- a JSON-lines file, `open_audit_file(path)` (done by `lockdown` when `audit.file` is set, before the file system is
  confined): `O_APPEND | O_CREAT | O_NOFOLLOW`, mode 0600 (an existing file of this user is tightened, another user's
  is refused); on Windows a DACL for the owner and SYSTEM, and a reparse point is refused.

`audit()` never throws. A sink that throws or a short write is counted in `audit_failures()`; the decision that was
being audited does not change (AUD-3).

## Paths

`check_relative(rel)` validates an untrusted relative path with one rule set on every OS: `/` and `\` both separate;
absolute, drive, UNC and `\\?\` / `\\.\` paths, `:` (alternate data streams), NUL and control characters, Windows
device names (`CON`, `NUL`, `COM1`, `LPT¹`, `CONIN$`, ... with any extension), segments ending in `.` or space,
dot-only segments other than `.`/`..`, and `..` that would leave the root are refused. `safe_join(root, rel)` joins and
then checks the result with `canonical_within`, so a symlink inside the root that points outside is refused too
(throws `permission_denied`, control `CAP-3`). `canonical_within(path, roots)` returns the canonical path (absolute,
symlinks of the existing prefix resolved) if it lies inside a root, compared by whole components (case-insensitive on
Windows). These are checks, not handles: a link swapped in between the check and the open is outside their reach.

`modelfile::detail::safe_rel_path` (signed bundle manifests) is stricter (it refuses `\`, `.` and empty segments) and
is left as it is.

## enforce

`enforce(capability, target)` checks the active policy (`set_active_policy`, set by `lockdown`), throws
`permission_denied` and audits the denial. `try_enforce` returns `os::status`. Both have overloads taking an explicit
policy.

| Capability | Allowed when |
|---|---|
| `fs_read` | canonical path under an `fs.read` or `fs.write` root, or `"*"` |
| `fs_write` | canonical path under an `fs.write` root, or `"*"` |
| `net_connect` | mode `any`; `hosts`: loopback or a `net.hosts` match; `loopback`: loopback; `none`: never |
| `net_listen` | mode `any`; otherwise a loopback address, or a `net.hosts` match in mode `hosts` |
| `exec`, `env`, `dynamic_code` | the policy flag |
| `plugin_load` | `integrity.plugins` is not `required` (there is no plugin signature verifier, so required means refused) |

`exec`, `dynamic_code` and `plugin_load` allowed under a non-strict profile are audited as `allow`.

## lockdown and check

`check(policy)` reports, per control, what this process could apply; it changes nothing. `lockdown(policy)` applies
it, once per process. Both return a `report` (`json()`, `text()`, `find(name)`, `ok`). The plan is evaluated first; if a
required control would fail, `lockdown` throws `sandbox_error` before changing anything. If applying a step fails
afterwards, it throws after the fact (the steps already applied stay). Emulated, missing and failed controls are
audited (`weaken` / `fail`). With `profile::none` it does nothing.

Order: audit sinks → resource limits → no-new-privs → root check (SBX-5) → file-system confinement → syscall confinement
→ Windows mitigation policies → integrity settings → the policy becomes active for `enforce`.

States: `active`, `available`, `emulated` (FORGE's in-process `enforce()` stands in for a missing kernel mechanism),
`missing`, `n/a` (not asked for), `failed`. Levels: `required` (strict), `should` (standard), `may` (permissive),
`info`. Verdicts: missing / failed → `fail` when required, `warn` when should; emulated → `fail` only when required and
`emulation` is `deny`.

| Control | Linux / Android | Windows | macOS | FreeBSD | OpenBSD | iOS |
|---|---|---|---|---|---|---|
| limits | `setrlimit` (core, nofile, AS, CPU, NPROC) | job object (memory, CPU time, active processes, die-on-unhandled-exception); nofile n/a | `setrlimit`; AS missing | `setrlimit` | `setrlimit` | `setrlimit` |
| no-new-privs | `prctl(PR_SET_NO_NEW_PRIVS)` | n/a | n/a | `procctl` (14+) | n/a | n/a |
| SBX-5 root | `geteuid() == 0` refused unless `allow_root` | elevated token refused unless `allow_root` | same as Linux | same | same | same |
| fs write, empty list | seccomp + Landlock (`os::apply_sandbox`) | emulated | `sandbox_init` no-write | emulated | `pledge` | app sandbox |
| fs write, paths | Landlock write rules beneath each root (+ `/dev/null`); without Landlock emulated | emulated | emulated | emulated | emulated | app sandbox |
| fs read | emulated everywhere (kernel read rules would break late `dlopen`) | | | | | |
| exec denial | seccomp + Landlock | job object, one active process | `sandbox_init` pure-computation only with an empty write list, else emulated | emulated (Capsicum is all-or-nothing) | `pledge` | app sandbox |
| net `none` | seccomp (no sockets but `AF_UNIX`), Landlock TCP from ABI 4 | active inside an AppContainer without network capabilities, else emulated | `sandbox_init` no-internet | emulated | `pledge` | emulated |
| net `loopback` / `hosts` | emulated (no kernel filter by address) | emulated | emulated | emulated | emulated | emulated |
| dynamic code (W^X) | emulated | `ProcessDynamicCodePolicy` (ACG) | emulated | emulated | emulated | emulated |
| mitigations | n/a | extension points off, no remote / low-label images; strict: strict handle checks | n/a | n/a | n/a | n/a |
| MAC label (SBX-6) | `/proc/self/attr/current`, report only | not checked | not checked | not checked | not checked | not checked |
| models required | sets `FORGE_REQUIRE_SIGNED=1` and `FORGE_TRUSTED_KEYS` (honoured by `inference::model`) | same | same | same | same | same |
| plugins required | emulated: `enforce(plugin_load)` refuses | same | same | same | same | same |
| BLD-1 hardening | `forge::build_hardening()` when `core/build_info.hpp` exists, else the compiler's macros (`__SSP__`, `_FORTIFY_SOURCE`, `__PIE__`, `__CET__`) | same (MSVC: `/GS`) | same | same | same | same |

Also reported: CAP-1/CAP-2 (in-process, `info`), CRY-1 (OS CSPRNG probe), AUD-1 (OS facility, audit file).

`forge security check [--policy F] [--json]` prints `check()` for `load_policy()` and exits 1 when `ok` is false (see
[cli.md](cli.md)). A strict policy with `"emulation": "deny"` fails on every OS today, because file reads, W^X and
plugin signatures are only emulated.

## Tests

`tests/test_security.cpp` (ctest `test_security`, label `security`): profile defaults, parsing (full document,
omitted fields, unknown and duplicate keys, wrong types, `line:col`), the load order and environment variables,
attack strings for `check_relative` / `safe_join` / `canonical_within` (POSIX and Windows forms, symlink escapes),
`enforce` decisions and audit records through a callback sink, a throwing sink, audit file mode and symlink refusal,
`check()` JSON shape, and three child processes (the binary re-runs itself with `FORGE_SECURITY_TEST_CHILD`):
`lockdown` (writes outside the write root fail under Landlock, exec fails, core and nofile limits, no-new-privs, ACG on
Windows, the audit file), `refuse` (strict with `emulation: deny` is refused before anything is applied) and `models`.
Run on Linux (WSL2, kernel Landlock ABI 3; also with `FORGE_LANDLOCK=0`) and Windows (MinGW). macOS, FreeBSD and MSVC:
CI only.
