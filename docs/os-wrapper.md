# OS wrapper design (forge::os, full surface)

Status: **design, not implemented.** This extends the existing header-only `forge::os` layer
(`os/include/forge/os/`) into a complete, portable wrapper over the operating system. It is the
foundation for the scripting layer in [scripting.md](scripting.md).

## Goals

- One C++23 API for the whole OS surface on Linux, Android, Windows, macOS, iOS and FreeBSD.
- Keep the current conventions: header-only, `forge::os::result<T>` / `status`, `error{code, what}`,
  `unsupported(...)` instead of silent no-ops, UTF-8 strings everywhere (wide conversion only at the
  Windows boundary), no exceptions across the API.
- Every call is either implemented natively or returns `is_unsupported(e) == true`, so callers and
  scripts can feature-test at runtime. `os::caps()` reports what the current platform supports.
- No new third-party dependencies.

## What already exists

`os.hpp` already covers files (`file`, `read_file`, `stat`, `mkdirs`, `list_dir`, `rename`), paths,
`mapping` (mmap), virtual memory (`reserve`/`commit`/`protect`/`lock_memory`), `library` (dlopen),
env, clocks, threads/mutex/condvar/semaphore, `run()` (blocking exec), basic signals, `sysinfo`,
`entropy`, sockets and `poll`. Separate headers cover async files, file watch, CPU topology, NUMA,
huge pages, memory pressure, shared memory, local IPC, keystore, sandbox, service, power, perf,
crash handling and GPU device enumeration.

## New and extended modules

| Header | Surface |
|---|---|
| `process.hpp` | `spawn(spec)` with argv/env/cwd, stdin/stdout/stderr as pipe, file, null or inherit; `child.wait(timeout)`, `kill(sig)`, `pid`; process groups / Job Objects so a tree dies together; `list_processes()`, `process_info(pid)` (cmdline, RSS, CPU, parent) ; resource limits applied at spawn (memory, CPU time, open files) via rlimit / cgroups v2 / Job Objects |
| `pipe.hpp` | anonymous pipes, named pipes / FIFOs, non-blocking read with deadline, `read_all` helpers |
| `signal.hpp` | full portable signal set, `sigint`/`sigterm` → console ctrl events on Windows, signal-to-fd / self-pipe for event loops |
| `fs.hpp` (extends file API) | `copy`, `copy_tree`, symlinks/hardlinks, `chmod`/`chown`, timestamps, `glob`, recursive `walk`, file locks, disk usage (`statvfs` / `GetDiskFreeSpaceEx`), mount list, temp files, atomic `write_file_atomic` |
| `file_watch.hpp` (extend) | recursive watch, debounce, inotify / FSEvents / kqueue / ReadDirectoryChangesW, polling fallback |
| `users.hpp` | current user/group, uid/gid or SID, `lookup_user`, `is_elevated()`, effective permissions (`access`), home/config/cache/data dirs per platform convention |
| `net.hpp` (extends sockets) | TCP/UDP/Unix-domain, IPv4/IPv6, DNS `resolve`, interface list with addresses/MAC, simple blocking HTTP/1.1 client (no TLS), port probe |
| `time.hpp` | monotonic/wall/CPU clocks (existing), timezone offset, ISO-8601 format/parse, high-resolution timers, `timer_queue` for scheduled callbacks |
| `sysinfo.hpp` (extends) | OS name/version/kernel, arch, uptime, load average, per-CPU usage, memory (incl. cgroup limit, shared with `forge::mem`), disks, battery/thermal (via `power.hpp`), GPU list (via `gpu_devices.hpp`) |
| `devices.hpp` | block devices, USB/PCI enumeration where the OS exposes it (sysfs, SetupAPI, IOKit); read-only |
| `service.hpp` (extend) | install/start/stop/status for systemd user units, launchd agents, Windows SCM; `unsupported` on mobile |
| `dynlib.hpp` | existing `library` plus symbol enumeration where available and search-path helpers |
| `registry.hpp` | Windows registry read/write/enumerate; `unsupported` elsewhere |
| `console.hpp` | is-a-tty, terminal size, ANSI/VT enabling on Windows, raw mode for the REPL line editor |
| `clipboard.hpp` | text get/set on desktop platforms (optional, low priority) |

### Platform notes

- **Android / iOS:** no `fork`, restricted spawn (iOS: `unsupported`), no services, sandboxed paths.
  App-container dirs are returned by the path helpers.
- **Windows:** `CreateProcessW` + Job Objects, overlapped pipes, `ReadDirectoryChangesW`, SCM, registry.
- **macOS / FreeBSD:** `posix_spawn`, kqueue, `sysctl`; FSEvents on macOS.
- **Linux:** `posix_spawn`/`pidfd`, inotify, cgroups v2 for limits, `/proc` and `/sys` for info.

## Capability model

`os::caps()` returns a flat set of named capabilities (`"spawn"`, `"process.limits.memory"`,
`"registry"`, `"watch.recursive"`, ...). The scripting layer exposes this as `os.caps()` so scripts
can branch on features rather than on platform names.

## Testing

- Unit tests per module under `tests/os/`, running on the host; platform-specific tests skip with
  the standard skip code when `is_unsupported` is returned.
- Cross-compile build checks for Windows, macOS, FreeBSD, Android, iOS with the zig toolchain.
- Fuzz target for path/glob parsing.

## Phasing

1. `process` + `pipe` + `signal` (needed by scripting `sh {}` blocks and `run`).
2. `fs` extensions, `users`, `time`, `sysinfo`, `console`.
3. `net` extensions, `file_watch` recursive, `service`, `registry`.
4. `devices`, `clipboard`, remaining platform gaps.
