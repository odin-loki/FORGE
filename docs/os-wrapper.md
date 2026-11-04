# OS wrapper: processes, files, system, network

This page covers the `forge::os` headers beyond the core layer in [os.md](os.md): process control, pipes, signals, file-system extras, users, time, system information, console, networking, recursive file watching, service control, the Windows registry, devices, the clipboard, dynamic libraries and process pools. It started as a design; it is now built, and the parts that are still missing are listed under [Not built yet](#not-built-yet).

The scripting layer uses these headers for `run`, `proc`, `fs`, `os`, `time`, `net` and `service` and for `ProcessPool` (see [scripting.md](scripting.md)).

## Concepts
- Same conventions as `os.hpp`: header-only, UTF-8 strings (wide strings only at the Windows boundary), `os::result<T>` / `os::status` returns, no exceptions (except `process_pool`, below).
- A call either works natively or returns an error with `os::is_unsupported(err) == true`, so callers can test for a feature at run time.
- All headers below are included by `<forge/os/all.hpp>` except `process_pool.hpp`.

## Module reference
| Header | Namespace | Main API |
|---|---|---|
| `process.hpp` | `forge::os` | `spawn_wait(spawn_spec)` → `proc_result`; `child` (streaming); `wait_any`, `wait_all`; `find_executable`; `apply_limits`; `kill_group`; `list_processes`, `process_info`; `process_caps` |
| `pipe.hpp` | `forge::os` | `make_pipe()` → `{read, write}` RAII `pipe_end`s with `write_all`, `read_some(max, timeout_ms)`, `read_all` |
| `signal.hpp` | `forge::os` | `signal_id` (interrupt, terminate, kill, hangup, quit, user1, user2), `sig_name`, `sig_from_name`, `sig_native`, `send_signal(pid, sig, group)` |
| `fs.hpp` | `forge::os::fs` | `copy`, `copy_tree`, `move`, `remove_all`, `symlink`, `hardlink`, `read_link`, `mode`, `chmod`, `chown`, `mtime_ns`, `set_mtime_ns`, `touch`, `glob_match`, `glob`, `walk`, `file_lock`, `disk_usage`, `mounts`, `make_temp_file`, `make_temp_dir`, `write_file_atomic` |
| `users.hpp` | `forge::os` | `current_user`, `lookup_user`, `lookup_uid`, `is_elevated`, `user_directory(user_dir)`, `config_dir`, `cache_dir`, `data_dir` |
| `time.hpp` | `forge::os` | `unix_time_ns`, civil-time conversion, `utc_offset_seconds`, `timezone_name`, `format_iso8601`, `format_local_iso8601`, `parse_iso8601`, `timer_queue`, `stopwatch`, `deadline` |
| `sysinfo.hpp` | `forge::os` | `release_info`, `uptime_seconds`, `load_average`, `memory()`, `per_cpu_times`, `cpu_usage`, `self_usage`, `current_rss` |
| `console.hpp` | `forge::os::console` | `is_tty`, `size`, `enable_vt`, `color_enabled`, `paint`, `mode_guard` (raw or no-echo), `read_password`, `confirm` |
| `net.hpp` | `forge::os::net` | `resolve`, `interfaces`, `port_open`, `parse_url`, `http(request)`, `http_get` |
| `file_watch_tree.hpp` | `forge::os` | `tree_watcher::open(root, tree_watch_opts)`, `poll(ms)` |
| `service_control.hpp` | `forge::os::service` | `service_spec`, `render_systemd_unit`, `render_launchd_plist`, `install`, `start`, `stop`, `query`, `uninstall` |
| `registry.hpp` | `forge::os::registry` | `get`, `set` (string or 64-bit integer), `remove_value`, `remove_key`, `list` |
| `devices.hpp` | `forge::os::devices` | `block()`, `pci()`, `usb()` |
| `clipboard.hpp` | `forge::os::clipboard` | `available`, `get`, `set` |
| `dynlib.hpp` | `forge::os` | `dynlib::open`, `get<F>`, `require<F>`, `path()`, `symbols()`, `file_symbols(path)`, `file_format(path)` |
| `process_pool.hpp` | `forge::os` (alias `forge::process_pool`) | `process_pool`, `FORGE_PROCESS_TASK`, `register_process_task` |

## Processes (`process.hpp`)
`spawn_wait(spec)` starts a program, feeds stdin, collects output and waits.

| `spawn_spec` field | Meaning |
|---|---|
| `argv`, `cwd` | Program and arguments; empty `cwd` inherits |
| `env_set`, `env_unset`, `clear_env` | Environment changes |
| `stdin_data` | Bytes for stdin (`nullopt` = null device) |
| `capture_stdout`, `capture_stderr`, `merge_stderr` | Capture into `out`/`err`, or inherit; merge stderr into `out` |
| `stdin_file`, `stdout_file`, `stderr_file`, `append_output` | Redirect a stream to a file (wins over capture / `stdin_data`) |
| `timeout_ms` | Kill the whole process tree on expiry; `timed_out = true` |
| `max_output` | Per-stream cap (default 64 MiB); `truncated = true` |
| `limits` | `proc_limits`: `memory_bytes`, `cpu_seconds`, `open_files`, `file_size_bytes`, `no_core` |
| `stop` | `std::stop_token`; on request the tree is killed, `cancelled = true`, exit code 125 |
| `new_session` | POSIX `setsid()`; Windows `CREATE_NEW_PROCESS_GROUP` |
| `watch_memory_bytes` | Kill the tree past this many bytes; `memory_killed = true`, exit code 137 |

`proc_result` holds `exit_code` (128 + signal when killed by a signal), `term_signal`, `out`, `err`, `wall_ms`, `peak_rss_bytes`, and the flags `timed_out`, `truncated`, `cancelled`, `memory_killed` and `unreaped` (killed but not reaped within 2 s; a detached thread reaps it later).

How it works per platform:

| | POSIX (Linux, Android, macOS, FreeBSD) | Windows | iOS |
|---|---|---|---|
| Start | `serialized_fork` + `execve`; the child leads its own process group | `CreateProcessW` in a kill-on-close Job Object | unsupported |
| Tree kill | the process group (and, for a session leader, the session) | the Job Object | — |
| `memory_bytes` | `RLIMIT_AS`; macOS and Android use a resident-size watchdog instead | Job Object process memory | — |
| `cpu_seconds` | `RLIMIT_CPU` (soft, hard = soft + 1) | Job Object user time | — |
| `open_files`, `file_size_bytes`, `no_core` | `RLIMIT_NOFILE`, `RLIMIT_FSIZE`, `RLIMIT_CORE` | not applied | — |
| `watch_memory_bytes` | Linux/Android: RSS of the process group; elsewhere use `limits.memory_bytes` | Job Object committed memory | — |
| Handle inheritance | close-on-exec pipes | only the three std handles (`PROC_THREAD_ATTRIBUTE_HANDLE_LIST`), so one child never holds another child's pipe open | — |

`apply_limits` lowers any request above an existing hard limit to that limit, so an unprivileged process still gets a limit. It is async-signal-safe and can run in a forked child.

`find_executable(name)` searches `PATH`. A name with a separator is checked as is. On Windows a bare name without an extension resolves only to `.exe`, `.cmd`, `.bat` or `.com`, as `CreateProcess` would, so an extensionless shell script on `PATH` (MSYS2, Git) cannot shadow `cmd.exe`. `spawn_wait` runs `.cmd`/`.bat` files through `cmd.exe /d /c`.

`child::start(argv)` or `child::start(spec)` starts a process without waiting: `read(max, timeout_ms)` streams stdout (and merged stderr), `try_wait`, `wait(timeout_ms)`, `kill(sig)` (whole group). The destructor kills and reaps a child still running. `wait_any(children, timeout_ms)` and `wait_all(...)` wait on several. On Windows `child` uses `CreateProcessW` (only the std handles inherited) in a Job Object that carries the limits; stdout is a pipe read with `PeekNamedPipe` polling for the timeout, and `kill(sig)` (and the destructor) ends the whole tree with exit code 128 + sig, so `wait()` reports what POSIX would. `kill(0)` only checks that it is running. Unsupported on iOS.

`list_processes()` and `process_info(pid)` return pid, ppid, RSS, name and command line: `/proc` on Linux and Android; Toolhelp32 + `K32GetProcessMemoryInfo` + `NtQueryInformationProcess` (command line) on Windows; `proc_listpids` / `proc_pid_rusage` / `KERN_PROCARGS2` on macOS; `kern.proc.proc` / `kern.proc.args` on FreeBSD. A field the caller may not read (another user's argv or RSS on macOS, protected processes on Windows) stays empty or 0. Unsupported on iOS.

`process_caps()` lists what the platform supports: `spawn`, `process.limits.memory`, `process.limits.cpu`, `process.limits.files` (POSIX), `process.tree_kill`, `process.timeout`, `signals` (POSIX). It is empty on iOS. ForgeScript's `os.caps()` returns this list with the platform name and the script's own capabilities.

`mem::spawn_leased` ([memory.md](memory.md)) starts a `child` whose memory cap is a lease on a memory budget.

## Pipes and signals
- `pipe.hpp` is POSIX only. Pipes are close-on-exec. `read_some` with a timeout returns an empty string on timeout or EOF.
- `send_signal(pid, s, group)` delivers any `signal_id` on POSIX (`group = true` signals the process group). On Windows only `kill` and `terminate` work (`TerminateProcess`); the rest return unsupported (a console control event cannot be aimed at one process).
- Handler installation is `os::on_signal` in `os.hpp`; on Windows it maps Ctrl-C to `interrupt` and close / Ctrl-Break / shutdown to `terminate`.

## Files (`fs.hpp`)
| Function | Notes |
|---|---|
| `copy`, `copy_tree`, `move` | `move` across devices falls back to copy then remove |
| `symlink`, `hardlink`, `read_link` | |
| `mode`, `chmod`, `chown` | `chown` is unsupported on Windows (owners are SIDs, and changing them needs privileges) |
| `mtime_ns`, `set_mtime_ns`, `touch` | |
| `glob_match(pattern, path)`, `glob(pattern, base)` | `*`, `?`, `**`, `[a-z]`, `[!x]`, `\` escapes |
| `walk(root, walk_opts)` | `max_depth`, `follow_symlinks`, `include_dirs`, `include_hidden` |
| `file_lock` | RAII advisory lock: `flock` / `LockFileEx`, shared or exclusive, wait or try |
| `disk_usage(path)` | total, free, available |
| `mounts()` | `/proc/self/mounts` (Linux, Android), volume enumeration (Windows), `getfsstat` (macOS, FreeBSD) |
| `make_temp_file`, `make_temp_dir` | |
| `write_file_atomic(path, data, mode)` | temp file, fsync, rename, then fsync of the directory (POSIX) |

## Users, time, system, console
- **`users.hpp`.** `current_user()` everywhere. `lookup_user` / `lookup_uid` (with groups) everywhere but iOS. Windows has no numeric ids: `uid` is the last sub-authority (RID) of the account SID and `gid` that of its primary group; `user_info::sid` holds the SID string, the home directory comes from the profile list and full name / groups from netapi32 (loaded at run time). `current_user()` reports the process token's primary group, which can differ from the account's. `is_elevated()` is root or an elevated token. `user_directory(config|cache|data|state|runtime|documents|downloads|desktop)` follows XDG on Linux and FreeBSD, `~/Library` on Apple, Known Folders on Windows.
- **`time.hpp`.** `unix_time_ns`, civil-time conversion, DST-aware `utc_offset_seconds`, `timezone_name`, `format_iso8601` / `format_local_iso8601` / `parse_iso8601` (RFC 3339 with offsets and fractions to ns). `timer_queue` runs `after(ms, fn)` and `every(ms, fn)` callbacks on one worker thread; `cancel(id)`. `stopwatch` and `deadline` are described in [support.md](support.md).
- **`sysinfo.hpp`.**

  | Call | Linux | Android | Windows | macOS / iOS | FreeBSD |
  |---|---|---|---|---|---|
  | `release_info` | os-release | yes | `RtlGetVersion` (`version` "10" / "11" by build; NT version in `kernel_version`) | sysctl | sysctl |
  | `uptime_seconds` | `/proc/uptime` | yes | `GetTickCount64` | `kern.boottime` | `kern.boottime` |
  | `load_average` | `getloadavg` | `/proc/loadavg` | unsupported (no run-queue average) | `getloadavg` | `getloadavg` |
  | `memory()` | `/proc/meminfo`; cgroup v2/v1 of this process (`/proc/self/cgroup`), tightest `memory.max` up the tree | yes | free = zeroed + free lists (else available − cache); page file as swap | free pages; `vm.swapusage` | `v_free_count`; swap from `vm.swap_info` |
  | `per_cpu_times`, `cpu_usage` | `/proc/stat` | yes | `NtQuerySystemInformationEx` per processor group | `host_processor_info` | `kern.cp_times` |
  | `self_usage`, `current_rss` | yes | yes | peak working set, page faults (soft and hard, in `minor_faults`) | yes | yes |

- **`console.hpp`.** `is_tty`, `size`, `enable_vt` (Windows VT mode), `color_enabled` (honours `NO_COLOR`, `FORCE_COLOR`, `TERM=dumb`), `paint`, RAII `mode_guard` (raw or no-echo), `read_password`, `confirm`.

## Networking (`net.hpp`)
| Call | Notes |
|---|---|
| `resolve(host)` | IPv4 and IPv6 address strings |
| `interfaces()` | name, addresses, up, loopback: `getifaddrs`; `GetAdaptersAddresses` (iphlpapi, loaded at run time) on Windows; Android `getifaddrs` from API 24, else an IPv4-only `SIOCGIFCONF` fallback |
| `port_open(host, port, timeout_ms)` | TCP connect probe |
| `parse_url(url)` | scheme, host, port, path, query; `resolve_reference(base, ref)` (RFC 3986), `to_string(url)`, `host_port(url)` (IPv6 in brackets) |
| `http(request)`, `http_get(url)` | Blocking HTTP/1.1 client: Content-Length and chunked bodies, redirects, size cap. `https://` returns unsupported |

`http` rejects a method, header name or header value with CR, LF, NUL or other non-token bytes (`EINVAL`) and percent-encodes the path and query. Redirects resolve relative `Location` values against the current URL and drop `Authorization`, `Proxy-Authorization` and `Cookie` when the scheme, host or port changes. A body shorter than its `Content-Length`, a chunked body without its final chunk or with a bad chunk size, or conflicting lengths are `EPROTO` errors. `timeout_ms` bounds the connect and each read; `deadline_ms` (default 120 s) bounds the whole exchange including redirects (`ETIMEDOUT`). Response headers are in `headers` (repeats joined by ", ", `Set-Cookie` by "\n") and, one per line, in `header_list` / `values(name)`; `url` is the URL that answered.

## Recursive file watching (`file_watch_tree.hpp`)
`tree_watcher::open(root, {debounce_ms = 50, force_polling, include_hidden, ignore})` registers every directory with the native `file_watcher` (inotify, kqueue, `ReadDirectoryChangesW`) and picks up new directories as they appear, reporting their existing contents as created. A per-path snapshot turns coarse kqueue directory events into exact created / modified / removed events. Where native watching is unavailable, or with `force_polling`, it rescans on each `poll()`. `poll(ms)` waits until the tree has been quiet for `debounce_ms` and coalesces bursts: create + modify → created, create + remove → nothing, one event per path. FSEvents is not used on macOS. A plain kqueue `file_watcher` on a directory also rescans and reports per-name created / removed / modified events; an in-place edit of a file inside a watched directory needs `add_file()` for that file.

Limits and loss. kqueue holds a descriptor per watched path, capped at half the `RLIMIT_NOFILE` soft limit (at most 8192; `max_native_handles` or `file_watcher::set_max_handles` to change it); a directory or file past the cap, or any directory the native backend refuses, is polled every 100 ms instead (`polled_dirs()`, backend "native+polling"). Windows completes all directories through one I/O completion port, so there is no 64-handle limit, and matches names case-insensitively. When the kernel drops events (inotify `IN_Q_OVERFLOW`, a Windows buffer overflow) `file_watcher::poll` returns `watch_kind::overflow` for each affected directory and `tree_watcher` rescans the whole tree. Directories are entered once by identity (device and inode; volume and file id on Windows), so symlink and junction loops are not followed. `file_watcher::remove(path)` drops a watch.

## Service control (`service_control.hpp`)
A `service_spec` (`name`, `description`, `argv`, `working_dir`, `env`, `restart_on_failure`, `start_at_login`, optional `dir`) becomes a per-user service:

| Platform | Backend |
|---|---|
| Linux, FreeBSD with systemd | user unit in `$XDG_CONFIG_HOME/systemd/user`, driven by `systemctl --user` |
| macOS | LaunchAgent plist in `~/Library/LaunchAgents`, driven by `launchctl` |
| Windows | Service Control Manager service through `sc.exe` (needs an elevated process) |
| Android, iOS | unsupported |

`render_systemd_unit` and `render_launchd_plist` return the file text without touching the system. `query` returns `not_installed`, `stopped`, `running`, `failed` or `unknown`. Setting `spec.dir` writes the unit elsewhere without registering it (the tests use this).

## Registry, devices, clipboard
- **`registry.hpp`.** Windows only: `get` / `set` string and 64-bit integer values (`REG_SZ`, `REG_DWORD`, `REG_QWORD`), `remove_value`, `remove_key`, `list`. Every call is unsupported elsewhere.
- **`devices.hpp`.** Read-only enumeration: `block()` (name, model, vendor, serial, size, removable, rotational, read-only, partitions), `pci()` (address, vendor/device/class ids, driver), `usb()` (vendor/product ids, strings, bus/devnum). Linux and Android read sysfs. Windows lists `PhysicalDriveN` disks with drive letters as partitions (storage IOCTLs, no admin needed) and PCI/USB through SetupAPI loaded at run time; USB `bus` / `devnum` are the hub number and hub port (Windows has no USB address). FreeBSD uses the GEOM sysctls, `/dev/pci` (`PCIOCGETCONF`) and the newbus device tree. macOS uses IOKit loaded at run time. Unsupported on iOS.
- **`clipboard.hpp`.** Text get/set: Win32 clipboard on Windows (retried for ~100 ms while another program holds it; foreign text bounded by its allocation), `pbcopy`/`pbpaste` on macOS (run with a UTF-8 locale), `wl-copy`/`wl-paste` or `xclip`/`xsel` on Linux and FreeBSD. An empty selection reads as "". Unsupported without a display or helper, and on mobile.

## Dynamic libraries (`dynlib.hpp`)
`os::dynlib` sits on top of `os::library`.

```cpp
auto m = os::dynlib::open("forge_plugin", {.search_dirs = {plugin_dir}, .env_var = "MYAPP_PLUGIN_PATH"});
auto add = m->get<int (*)(int, int)>("add");          // nullptr when missing; get<int(int, int)> works too
auto sub = m->require<int(int, int)>("sub");          // error naming the symbol when missing
std::printf("%s\n", m->path().c_str());               // file actually loaded
for (auto& s : m->symbols().value) std::printf("%s %p\n", s.name.c_str(), s.address);
auto syms = os::file_symbols("libm.so.6");            // without loading it
```

- **Search.** `open(name, opts)` tries `name` and `library::decorate(name)` in each of `search_dirs`, then in the dirs listed in `$env_var` (`:`-separated, `;` on Windows), then (if `system_search`) hands both names to the system loader exactly as `library::open` does. `""` is the running program (`dynlib::self()`); a name with a path separator is opened as is. `adopt(library, hint)` wraps an existing `os::library`. `find_library(name, dirs)` only searches the dirs.
- **`path()`.** Found through `dl_iterate_phdr` + `RTLD_NOLOAD` handle matching (ELF), `GetModuleFileNameW` (Windows) or the dyld image list (Apple). Empty if the platform cannot tell.
- **Symbols.** `dynlib_symbol{name, type (function/data/tls/other), weak, indirect, value (offset from the load base), address, version, default_version, ordinal, forwarder}`.
  - `file_symbols(path)` and `file_format(path)` parse any format on any host. ELF: `.dynsym` with `.gnu.version` / `.gnu.version_d` (both `foo@@V2` and hidden `foo@V1` are listed), or `PT_DYNAMIC` + `DT_HASH` / `DT_GNU_HASH` when section headers are stripped; 32/64-bit, either endianness. PE32/PE32+: the export directory (named exports, ordinals, forwarders). Mach-O: the export trie (`LC_DYLD_INFO[_ONLY]`, `LC_DYLD_EXPORTS_TRIE`) or external `LC_SYMTAB` entries; fat files use the host slice; the leading `_` is stripped.
  - `symbols()` adds run-time addresses. ELF reads the file and adds the load base (IFUNCs through `dlsym`, hidden versions through `dlvsym` on glibc). Windows parses the mapped module. Apple parses the image in memory, which also works for the dyld shared cache.
  - The parsers are bounds-checked: bad input gives an error or a shorter list, never an out-of-range read.

## Process pools (`process_pool.hpp`)
A pool of worker processes with an API like `task::pool` ([support.md](support.md)).

```cpp
FORGE_PROCESS_TASK(square, [](std::string in) { int v = std::stoi(in); return std::to_string(v * v); });
int main(int argc, char** argv) {
    if (forge::os::process_pool::run_worker_if_requested(argc, argv)) return 0;   // needed for spawn mode
    forge::os::process_pool pp({.workers = 4});
    auto f = pp.submit("square", "7");        // std::future<std::string>
    pp.wait_all(); std::string r = f.get();   // "49"
}
```

| API | Meaning |
|---|---|
| `options{workers, mode, exe, sandbox}` | `workers = 0` uses `FORGE_PROCESSES`, then `hardware_concurrency()` |
| `submit(name, payload)` | `std::future<std::string>`; payloads and results are byte strings you serialise yourself |
| `wait_all()`, `wait_all_for(ms)` | Wait until every queued and running task is done; `wait_all_for` returns false on timeout |
| `pending()`, `size()`, `set_workers(n)` | Queue length, worker count, grow or retire workers after their current task |
| `shutdown()` | Graceful: finishes queued tasks, sends each worker a quit message, reaps it. Also run by the destructor. Waits without a bound |
| `terminate()` | Abrupt: queued and running tasks fail with `os_errc::cancelled`, every worker is killed and reaped |

- **Modes.** `fork` (default on Linux, macOS, FreeBSD, Android) forks workers that run the registered tasks. `spawn` re-launches the program with `--forge-process-worker=...`: fork + exec on POSIX, `CreateProcessW` with the worker's stdin/stdout as the IPC pipes on Windows, where it is the only mode (`fork` throws `os_exception{not_supported}`). Spawn mode needs `run_worker_if_requested(argc, argv)` first in `main()`; a test harness must do the same, or a worker re-runs the test suite. On iOS the constructor throws `os_exception{not_supported}`.
- **IPC.** One pipe pair per worker; frames are length-prefixed (type, id, task name, payload).
- **Errors.** A task that throws fails its future with `os_exception` (kind carried over from the worker). A worker that dies mid-task fails that future with `os_errc::signal` and the signal number (Windows access violation / illegal instruction / divide by zero map to 1 / 2 / 3), or `os_errc::io` for a non-zero exit. The worker is respawned; other tasks keep running.
- **Sandbox.** `options.sandbox` applies `os::apply_sandbox` in every worker before its first task.
- ForgeScript exposes it as `ProcessPool(n, {...})`.

## Platform notes
- **iOS.** No spawning: `spawn_wait`, `child` and `os::run` return unsupported; `process_pool` throws. No service control. Paths come from the app container.
- **Android.** Spawning works (fork/exec). `RLIMIT_AS` is not used because bionic reserves far more address space than it uses; `memory_bytes` uses a resident-size watchdog. No service control.
- **Windows.** `CreateProcessW` + Job Objects (`spawn_wait` and `child`), `ReadDirectoryChangesW`, SCM through `sc.exe`, registry, event-log service log. `pipe.hpp`, `kill_group` and `chown` are POSIX only. `pin_thread_to` uses `SetThreadGroupAffinity`, so CPUs 64 and up work; a set spanning processor groups is an error. CPU ids past what the OS mask holds are `EINVAL` everywhere (Linux allocates a larger set with `CPU_ALLOC` beyond `CPU_SETSIZE`). System DLLs loaded at run time (`iphlpapi`, `setupapi`, `powrprof`, `dxgi`, `dxcore`) come from System32 only (`LOAD_LIBRARY_SEARCH_SYSTEM32`).
- **macOS.** `RLIMIT_AS` is ignored by the kernel, so `memory_bytes` uses a resident-size watchdog. kqueue for file watching.
- **Linux.** fork/exec, inotify, `/proc` and `/sys`.

## Not built yet
| Item | Today |
|---|---|
| `os::caps()` for the whole OS layer | Only `process_caps()`; ForgeScript `os.caps()` wraps it |
| `pipe.hpp` on Windows | POSIX only (`child` and `local_ipc.hpp` cover the uses) |
| Named pipes / FIFOs in `pipe.hpp`, signal-to-fd for event loops | Not present (`local_ipc.hpp` covers named pipes) |
| cgroup v2 limits for spawned processes | rlimits, Job Objects and watchdogs instead |
| HTTPS in `net::http` | unsupported; use the RPC client or a TLS library |
| FSEvents on macOS | kqueue is used instead (immediate per-file events) |
| Fuzz target for glob/path parsing | none |
| `load_average` on Windows, `chown` on Windows, signals other than kill/terminate on Windows, `advise_huge` on Windows/macOS, hard CPU affinity on macOS | no OS facility (see the sections above) |

## Tests
| Test | Covers |
|---|---|
| `test_os_process` | spawn, limits, timeouts, tree kill, redirection, cancellation, memory watchdog, `find_executable` |
| `test_os_sys` | `fs`, users, time, sysinfo, console |
| `test_os_net` | net, local HTTP servers (injection, cross-origin redirects, truncated bodies, deadline, `Set-Cookie`), native and polling tree watch (80 directories, overflow, symlink/junction loop, kqueue cap), service units in a temp dir |
| `test_os_dev` | devices, FreeBSD GEOM partition fixture, clipboard |
| `test_os_dynlib` | loads `tests/dynlib_fixture.c` through search dirs and an env var, calls it through `get<>` and through enumerated addresses, enumerates libm; when clang + lld are installed, parses fixtures cross-linked by `cmake/dynlib_fixtures.cmake` (PE32/PE32+ with a forwarder, arm64/x86_64/exports-trie/fat Mach-O, ELF32, big-endian ELF, section-stripped ELF, versions + IFUNC + TLS), plus ~14k truncated and mutated inputs |
| `test_process_pool` | fork and spawn modes, crash recovery, typed task errors, grow/shrink, shutdown, `terminate`, sandboxed workers, Windows spawn |
| `test_script` `os_phase2_surface`, `os_phase3_surface`; `test_script_oserror` `script_process_pool` | script bindings |

All of these passed in CI on Linux, macOS (arm64), Windows (MSVC and MinGW) and FreeBSD 14.3 on 2026-11-04. Cases a platform does not support are skipped inside each test. With `zig` installed, `xos_test_os_{process,sys,net,dev,dynlib}_*` compile and link them for FreeBSD, macOS, MinGW and aarch64 musl. iOS has not been built.
