# forge::os parity matrix

What every public call of `forge::os` does on each OS, read from the code branch by branch (2026-11-12).
✓ = a native implementation. **degraded** = works, with the stated difference. **unsupported** = returns
`os::unsupported(...)` (`os::is_unsupported(err)`); the reason says whether the OS has no such facility
(*inherent*) or it is not built yet. Guides: [os.md](os.md), [os-wrapper.md](os-wrapper.md).

## Processes, files, IPC, core

Headers: `os.hpp`, `oserror.hpp`, `fs.hpp`, `async_file.hpp`, `pipe.hpp`, `process.hpp`, `process_pool.hpp`,
`sandbox.hpp`, `signal.hpp`, `crash.hpp`, `local_ipc.hpp`, `shared_memory.hpp`, `console.hpp`, `time.hpp`,
`locked_memory.hpp`, `private_files.hpp`, `sync.hpp` (new), `pty.hpp` (new).

### How each OS was checked

| OS | Compiled | Run |
|---|---|---|
| Linux x86_64 | clang 23 (C++23), native | `test_os`, `test_os_ext`, `test_os_process`, `test_os_sys`, `test_process_pool` pass (kernel 6.18: Landlock ABI 7, seccomp user notification; also with `FORGE_LANDLOCK=0`, `FORGE_EXEC_GATE=0` and both) |
| Windows | MinGW-w64 g++ 13 (C++23, full link) and clang 23 `--target=x86_64-w64-mingw32` (syntax). **MSVC: not compiled here** | The new tests ran under wine 9.0. Wine lacks, so these paths did **not** run: reparse points (`junction`, `read_link`, file `symlink`), stream enumeration (`xattr_list`), `FileAllocationInfo` / `FSCTL_SET_ZERO_DATA` (`preallocate`, `punch_hole`), ConPTY output (`pty::read`), background mode (`set_io_priority`), integrity-level and job enforcement (`apply_sandbox`), AppContainers |
| macOS arm64 | clang 23 and zig 0.17 against zig's macOS headers (tests linked) | **never run** |
| FreeBSD 14 x86_64 | clang 23 and zig 0.17 against zig's FreeBSD headers (tests linked) | **never run** |
| iOS | macOS headers with `TARGET_OS_IPHONE` forced to 1 (syntax of every header here and of `test_os.cpp`); **not the iOS SDK** | **never run** |
| Android | Linux headers with `__ANDROID__` forced (syntax of every header here); **not the NDK** | **never run** |

CI (Windows MSVC and MinGW, macOS, iOS simulator, FreeBSD VM, Android NDK) is the first real compile and run
for everything marked "never run" above.

### Core (`os.hpp`)

| Feature | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `file` open / read / write / pread / pwrite / size / resize / sync | ✓ | ✓ | ✓ (1 GiB per call; `pread` moves the handle position) | ✓ (`F_FULLFSYNC`) | ✓ | ✓ |
| `stat`, `mkdir(s)`, `remove(_all)`, `rename`, `list_dir` | ✓ | ✓ | ✓; paths of 248+ characters go through `\\?\` (`wpath`), so `MAX_PATH` does not apply | ✓ | ✓ | ✓ |
| `mapping` (file, anonymous), `reserve` / `commit` / `protect`, `lock_memory`, `advise_willneed` | ✓ | ✓ | ✓ (64 KiB offset granularity) | ✓ | ✓ | ✓ |
| `library`, environment, `exe_path`, paths | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `monotonic_ns`, `wall_ns`, `cpu_time_ns`, `thread_cpu_time_ns`, `sleep_ns` | ✓ | ✓ | ✓ (high-resolution timer from 1803) | ✓ | ✓ | ✓ |
| `thread`, `mutex`, `condvar`, `semaphore`, `set_thread_name` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `set_thread_priority` | ✓ nice per thread (`high` needs privilege) | ✓ | ✓ | ✓ QoS class | ✓ QoS class | ✓ `rtprio_thread` |
| `pin_thread` | ✓ | ✓ | ✓ (across processor groups) | returns false: inherent, XNU has no hard affinity | returns false: inherent | ✓ |
| `run(argv)` | ✓ `posix_spawn` | ✓ (fork + exec before API 28) | ✓ | ✓ | unsupported: inherent (now a proper `unsupported` error; it was code -1) | ✓ |
| `on_signal`, `ignore_sigpipe` | ✓ | ✓ | degraded: console control events mapped to interrupt / terminate / hangup; no SIGPIPE (inherent) | ✓ | ✓ | ✓ |
| `socket`, `poller` | ✓ epoll | ✓ epoll | ✓ WSAPoll | ✓ kqueue | ✓ kqueue | ✓ kqueue |
| `completion_port` | emulated (threads) | emulated | ✓ IOCP | emulated | emulated | emulated |
| `entropy`, `sysinfo`, memory counters | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `android_api_level()` (new) | n/a (10000) | ✓ | n/a | n/a | n/a | n/a |

### Waiting primitives (`sync.hpp`, new)

| Feature | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `wait_on_address`, `wake_one`, `wake_all` | ✓ futex | ✓ futex | ✓ `WaitOnAddress` | ✓ `os_sync_wait_on_address` (14.4+); degraded before: mutex + condvar table (the only older interface is private) | ✓ 17.4+; table before | ✓ `_umtx_op` |
| `wait_event` | ✓ eventfd | ✓ eventfd | ✓ manual-reset event | ✓ non-blocking pipe (2 descriptors) | ✓ pipe | ✓ eventfd (13+), else pipe |
| `wait_timer` (one-shot, periodic, ns) | ✓ timerfd | ✓ timerfd | ✓ high-resolution waitable timer (15.6 ms steps before 1803) | ✓ kqueue `EVFILT_TIMER` | ✓ kqueue | ✓ kqueue |
| `signal_listener` | ✓ self-pipe | ✓ | degraded: console control events only; `sig::pipe` unsupported (inherent) | ✓ | ✓ | ✓ |

### Files (`fs.hpp`, `private_files.hpp`, `async_file.hpp`)

| Feature | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `copy`, `copy_tree`, `move`, `walk`, `glob`, `mode`, `chmod`, times, `disk_usage` | ✓ | ✓ | ✓ (`chmod`: read-only bit only, inherent) | ✓ | ✓ | ✓ |
| `chown` | ✓ | ✓ | unsupported: inherent (owners are SIDs; changing them needs a privilege) | ✓ | ✓ | ✓ |
| `symlink`, `read_link`, `hardlink` | ✓ | ✓ | ✓ native now (`CreateSymbolicLinkW`, reparse data, `CreateHardLinkW`; libstdc++ on MinGW could not). File symlinks need Developer Mode or the privilege; directories fall back to a junction | ✓ | ✓ | ✓ |
| `junction` (new) | symlink to the absolute path | same | ✓ mount-point reparse point, no privilege | symlink | symlink | symlink |
| `copy_file` (new) | ✓ `FICLONE`, then `copy_file_range`, then read/write | ✓ (`copy_file_range` from API 34) | ✓ `CopyFileExW` (clones by itself on ReFS / Dev Drive) | ✓ `clonefile`, then `fcopyfile` | ✓ | ✓ `copy_file_range` (13+; clones on ZFS) |
| `clone_file` (new) | ✓ `FICLONE` (Btrfs, XFS reflink, bcachefs); unsupported on other file systems | ✓ | unsupported: no call that clones or fails (`FSCTL_DUPLICATE_EXTENTS_TO_FILE` on ReFS is not built) | ✓ `clonefile` (APFS) | ✓ | unsupported: inherent, no explicit clone call |
| `preallocate` (new) | ✓ `posix_fallocate` | ✓ | ✓ `FileAllocationInfo` | ✓ `F_PREALLOCATE` | ✓ | ✓ (ZFS refuses: unsupported) |
| `punch_hole` (new) | ✓ `fallocate(PUNCH_HOLE)` | ✓ | ✓ `FSCTL_SET_SPARSE` + `FSCTL_SET_ZERO_DATA` | ✓ `F_PUNCHHOLE` (APFS, block-aligned) | ✓ | ✓ `fspacectl` (14+); unsupported on 13 |
| `data_ranges` (new) | ✓ `SEEK_DATA` / `SEEK_HOLE` | ✓ | ✓ `FSCTL_QUERY_ALLOCATED_RANGES` | ✓ | ✓ | ✓ |
| `file_id`, `same_file` (new) | ✓ dev + inode | ✓ | ✓ volume serial + file index | ✓ | ✓ | ✓ |
| `xattr_get` / `set` / `remove` / `list` (new) | ✓ `user.` namespace | ✓ | ✓ NTFS alternate data streams | ✓ | ✓ | ✓ `extattr` user namespace |
| `rename_noreplace` (new) | ✓ `renameat2` (link + unlink on file systems without it) | ✓ (API 30; link + unlink before) | ✓ `MoveFileEx` without replace | ✓ `renamex_np(RENAME_EXCL)` | ✓ | degraded: link + unlink (files; both names exist for a moment); directories unsupported (inherent) |
| `rename_exchange` (new) | ✓ `renameat2` | ✓ (API 30) | unsupported: inherent, no atomic exchange | ✓ `RENAME_SWAP` | ✓ | unsupported: inherent |
| `sync_dir` (new) | ✓ | ✓ | ✓ `FlushFileBuffers` on the directory | ✓ | ✓ | ✓ |
| `anonymous_file` (new) | ✓ `O_TMPFILE` | ✓ | degraded: `FILE_FLAG_DELETE_ON_CLOSE` (the name exists, privately, until close) | exclusive create + unlink | same | same |
| `file_lock` | ✓ `flock` (advisory) | ✓ | ✓ `LockFileEx` (mandatory for the locked range: other handles' reads and writes fail; inherent difference) | ✓ | ✓ | ✓ |
| `mounts` | ✓ | ✓ | ✓ drive letters | ✓ | ✓ | ✓ |
| `make_temp_file` / `dir`, `write_file_atomic`, `check_private`, `ensure_private_dir`, `write_new_private_file` | ✓ | ✓ | ✓ (private DACL; `mode` only as read-only) | ✓ | ✓ | ✓ |
| `async_file_io` | ✓ io_uring, else threads | ✓ (threads when seccomp refuses io_uring) | ✓ IOCP | ✓ POSIX AIO | ✓ POSIX AIO | ✓ POSIX AIO + kqueue |

### Processes (`process.hpp`, `process_pool.hpp`, `signal.hpp`, `pipe.hpp`, `pty.hpp`)

| Feature | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `spawn_wait`, `child`, `wait_any` / `wait_all`, `find_executable` | ✓ | ✓ | ✓ Job Object | ✓ | unsupported: inherent | ✓ |
| `child::wait(timeout)` | ✓ pidfd (5.3+; 1 ms naps before) | ✓ pidfd from API 31; naps before | ✓ handle wait | ✓ kqueue `EVFILT_PROC` | — | ✓ kqueue |
| `wait_for_exit(pid)` (new; any process) | ✓ pidfd | ✓ | ✓ `SYNCHRONIZE` handle | ✓ kqueue | unsupported: inherent | ✓ kqueue |
| `spawn_spec::inherit`, `inheritable_handle` (new) | ✓ | ✓ | ✓ handle list | ✓ | — | ✓ |
| `limits.memory_bytes` | ✓ `RLIMIT_AS` | degraded: RSS watchdog (bionic reserves address space) | ✓ Job Object | degraded: RSS watchdog (the kernel ignores `RLIMIT_AS`) | — | ✓ |
| `limits.cpu_seconds`, `no_core` | ✓ | ✓ | ✓ Job Object | ✓ | — | ✓ |
| `limits.open_files`, `file_size_bytes` | ✓ | ✓ | ignored: inherent, no per-process cap | ✓ | — | ✓ |
| `apply_limits` (calling process) | ✓ | ✓ | ✓ new: joins a Job Object (memory, CPU time, no_core) | ✓ | ✓ rlimits | ✓ |
| `watch_memory_bytes` | ✓ group RSS | ✓ | ✓ job memory | ✓ group RSS | — | ✓ group RSS |
| `kill_group` | ✓ | ✓ | returns false: inherent, no process groups to signal (trees: Job Object, `kill_tree`) | ✓ | returns false | ✓ |
| `list_processes`, `process_info` | ✓ procfs | ✓ | ✓ Toolhelp | ✓ libproc | unsupported: inherent | ✓ sysctl |
| `process_descendants`, `kill_tree` (new) | ✓ | ✓ | ✓ (children must be younger than their parent, so reused pids are not followed) | ✓ | unsupported | ✓ |
| `set_process_priority` / `get_process_priority` (new) | ✓ nice | ✓ | ✓ priority class | ✓ | ✓ (own process) | ✓ |
| `set_process_affinity` / `get_process_affinity` (new) | ✓ | ✓ | ✓ (CPUs 0-63 of the process's group) | unsupported: inherent | unsupported: inherent | ✓ `cpuset` |
| `set_io_priority` (new) | ✓ `ioprio_set` idle class (calling thread and what it starts later) | unsupported: bionic and the app seccomp policy have no `ioprio_set` | ✓ background mode (lowers CPU and memory priority too) | ✓ `setiopolicy_np` | ✓ | unsupported: inherent |
| `send_signal` | ✓ | ✓ | degraded: kill / terminate only (inherent) | ✓ | ✓ (own processes) | ✓ |
| `make_pipe`, `pipe_end` | ✓ | ✓ | ✓ overlapped named pipe | ✓ | ✓ | ✓ |
| `pty` (new) | ✓ `posix_openpt` | ✓ | ✓ ConPTY (Windows 10 1809+; unsupported before) | ✓ | unsupported: inherent | ✓ |
| `process_pool` | ✓ fork, spawn | ✓ | spawn only (no fork: inherent) | ✓ | throws `not_supported`: inherent | ✓ |
| cgroup v2 / `rctl` limits | not built (rlimits and watchdogs instead) | — | Job Objects | — | — | not built |

### Sandbox (`sandbox.hpp`)

| Feature | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `apply_sandbox` deny_network | ✓ seccomp (+ Landlock ABI 4) | ✓ | unsupported outside an AppContainer: inherent (no unprivileged per-process network filter) | ✓ Seatbelt profile (named profile as fallback) | no-op: always sandboxed | degraded: Capsicum capability mode denies everything at once |
| `apply_sandbox` deny_fs_write | ✓ seccomp + Landlock | ✓ | ✓ new: low integrity level. degraded: `%LOCALAPPDATA%Low`, volumes without ACLs and handles already open stay writable | ✓ | no-op | degraded (Capsicum) |
| `apply_sandbox` deny_exec | ✓ | ✓ | ✓ Job Object; new fallback: child-process mitigation policy | ✓ | no-op | degraded (Capsicum) |
| `run_sandboxed` | ✓ | ✓ | ✓ AppContainer / restricted token + job | ✓ | unsupported: inherent | ✓ `fexecve` in capability mode (ELF only) |
| `run_sandboxed` deny_exec applies to the target | ✓ new: one-exec seccomp gate (5.5+) and Landlock (5.13+); also stops `ld.so prog`. Neither available: target not confined for exec | ✓ (gate probed in a throwaway child) | ✓ | ✓ new: Seatbelt `process-exec` allowed on the target only (126 if the profile is refused) | — | ✓ |
| `sandbox_status` | ✓ | ✓ | ✓ (+ `low-integrity`) | degraded: App Sandbox only (a Seatbelt profile cannot be queried) | ✓ | ✓ |

### IPC, shared memory, console, time, crash

| Feature | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `local_listener`, `local_stream`, `peer()` | ✓ (abstract names too) | ✓ | ✓ named pipes; peer = pid + SID | ✓ (`@abstract` unsupported: inherent) | ✓ | ✓ |
| `send_handle` / `recv_handle` (new) | ✓ `SCM_RIGHTS` | ✓ | ✓ `DuplicateHandle` into the peer; the value travels in-band, so the receiver must trust the sender (inherent) | ✓ | ✓ | ✓ |
| `shared_memory` create / open | ✓ | degraded: a file in `set_directory()` (no `shm_open`) | ✓ | ✓ | ✓ | ✓ |
| `shared_memory::anonymous`, `attach`, `share` (new) | ✓ memfd | ✓ memfd (API 30), else an unlinked file | ✓ unnamed section | ✓ shm object unlinked at once | ✓ | ✓ `SHM_ANON` |
| `shared_memory::seal` (new) | ✓ | ✓ | unsupported: inherent (a section cannot change size anyway) | unsupported: inherent | unsupported | ✓ (13+) |
| `console` tty, size, colours, `mode_guard`, `read_password` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `set_feature` mouse / bracketed paste / focus / alternate screen, `resize_watcher` (new) | ✓ | ✓ | ✓ (VT; input reports need `mode_guard(raw)`) | ✓ | ✓ | ✓ |
| `timezone_name`, ISO 8601, `timer_queue` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `boottime_ns`, `awake_ns` (new) | ✓ | ✓ | ✓ | ✓ | ✓ | degraded: `awake_ns` = `boottime_ns` (no clock that excludes suspend: inherent) |
| `stack_trace`, `crash_handler` | ✓ text report | ✓ | ✓ minidump + text | ✓ text report | ✓ | ✓ |
| `write_dump_now` | text report (no minidump format: inherent) | same | ✓ minidump | text report | text | text |
| `set_core_dumps` (new) | ✓ rlimit + `PR_SET_DUMPABLE` | ✓ | degraded: Windows Error Reporting off for the process; a debugger can still dump | ✓ rlimit | ✓ | ✓ rlimit + `PROC_TRACE_CTL` |
| `locked_buffer` | ✓ | ✓ | degraded: excluded from WER only | degraded: no dump exclusion (inherent) | same | ✓ |

### Not built (the OS could do it)

- cgroup v2 (Linux) and `rctl` (FreeBSD) limits for spawned processes.
- `clone_file` on Windows ReFS (`FSCTL_DUPLICATE_EXTENTS_TO_FILE`).
- One reactor over sockets, events, timers and child exits (`poller` takes sockets; `wait_event` / `wait_timer`
  expose `native()` for epoll / kqueue / `WaitForMultipleObjects`).
- Resizing a `mapping` in place (`mremap`).
- Named FIFOs in `pipe.hpp` (`local_ipc.hpp` covers named endpoints).
## System, hardware, network, services, mobile

Read from the code, branch by branch, on 2026-11-12. One row per public function or class of the headers in this
half of the layer (`net`, `netinfo`, `sockopt`, `sysinfo`, `virtualization`, `cpu_topology`, `numa`, `hugepages`,
`memory_pressure`, `power`, `perf`, `gpu_devices`, `devices`, `device_watch`, `volumes`, `display`, `idle`,
`clipboard`, `dynlib`, `file_watch`, `file_watch_tree`, `service`, `service_control`, `scheduler`, `background`,
`users`, `sessions`, `registry`, `preferences`, `settings`, `single_instance`, `autostart`, `shell_open`,
`notification`, `trash`, `mobile`).

Legend: **yes** a real implementation; **part** works with the stated limit or through a fallback; **hook** the OS
only offers it to Java / Swift code, so app glue passes it in through the named function; **no** returns
`os::unsupported(...)` (the reason is in "Inherent limits" unless a note says otherwise). Mechanisms are named in
the header comments and in [os.md](os.md); this table says only whether the call works.

### Network (`net.hpp`, `netinfo.hpp`, `sockopt.hpp`)
| API | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `resolve`, `port_open`, `parse_url`, `resolve_reference` | yes | yes | yes | yes | yes | yes |
| `interfaces` | yes | part (API < 24: IPv4 only) | yes | yes | yes | yes |
| `http`, `http_get` over http | yes | yes | yes | yes | yes | yes |
| `http` over https, `tls_backend` | yes (system OpenSSL ≥ 1.1) | part (only when the app ships libssl) | yes (Schannel) | yes (Network.framework) | yes (Network.framework) | yes (base OpenSSL) |
| `links` (MAC, MTU, index, type, speed, prefix) | yes | part (MAC and sysfs hidden from apps: type by name) | yes | yes (no speed) | part (MAC hidden, no media type) | yes (no speed) |
| `default_routes` | yes | part (unsupported when `/proc/net` is denied, Android 10+) | yes | yes | no (SDK has no `<net/route.h>`) | yes |
| `dns_servers` | yes | part (`net.dnsN`, gone in Android 8+) | yes | yes | part (`/etc/resolv.conf`, absent in the sandbox) | yes |
| `proxy_from_env`, `proxy_bypassed`, `proxy_for` | yes | yes | yes | yes | yes | yes |
| `system_proxy` beyond the environment | part (GNOME via `gsettings`; KDE not read) | hook (`set_proxy_hook`) | yes (WinINet) | yes (CFNetwork) | yes (CFNetwork) | part (GNOME via `gsettings`) |
| `network_monitor` | yes (rtnetlink) | part (netlink refused from API 30: 1 s polling) | yes (iphlpapi) | yes (PF_ROUTE) | part (polling) | yes (PF_ROUTE) |
| `set_keepalive`, `set_v6only`, `set_buffer_sizes`, `set_linger`, `listen_dual_stack` | yes | yes | yes (count fixed at 10 before 10 1703) | yes | yes | yes |
| `set_fastopen` (listener) | yes | yes | yes (10 1607+) | yes | yes | yes |
| `set_fastopen_connect` (client) | yes | yes | no | no | no | no |
| `set_reuseport` | yes | yes | no | yes | yes | yes (`SO_REUSEPORT_LB`) |
| `unix_listen`, `unix_connect`, `unix_sockets_available` | yes | yes | yes (10 1803+, else no) | yes | yes | yes |
| `udp_bind`, `send_to`, `recv_from`, `join_multicast`, `leave_multicast`, `set_multicast` | yes | yes | yes | yes | yes (multicast needs the entitlement) | yes |
| `dns_encode_query`, `dns_parse`, `mdns_query` | yes | yes | yes | yes | yes (entitlement) | yes |
| `connect_happy_eyeballs`, `capabilities` | yes | yes | yes | yes | yes | yes |

### System information (`sysinfo.hpp`, `virtualization.hpp`, `cpu_topology.hpp`, `numa.hpp`, `hugepages.hpp`)
| API | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `release_info`, `uptime_seconds`, `boot_time_ns`, `memory`, `current_rss`, `self_usage` | yes | yes | yes | yes | yes | yes |
| `load_average` | yes | yes (`/proc/loadavg`) | part (emulated: 5 s sampler) | yes | yes | yes |
| `per_cpu_times` | yes | yes | yes | yes | yes | yes |
| `machine_id` | yes | hook (`set_machine_id`) | yes | yes | part (per-vendor id) | yes |
| `cpu_limits`, `usable_cpu_count` | yes (affinity + cgroup v1/v2 quota) | yes | yes (affinity + job CPU cap) | part (no quotas exist) | part (no quotas exist) | part (cpuset; rctl only with `kern.racct.enable`) |
| `memory_limits` | yes (cgroup) | yes | yes (job object) | part (RLIMIT_AS only) | yes (jetsam budget) | part (rctl / RLIMIT_AS) |
| `virtualization().hypervisor` | yes | part (emulator only) | yes | yes | no (not visible to apps) | yes |
| `virtualization().container` | yes (docker, podman, lxc, k8s, nspawn, WSL, flatpak, snap) | no | part (Windows containers) | no (none exist) | no | yes (jail) |
| `virtualization().translator` / `.emulator` | no (qemu-user leaves no mark) | emulator: yes | yes (ARM64 emulation, wine) | yes (Rosetta) | emulator: yes (simulator) | no |
| `cpu_topology_query` packages / cores / SMT / caches | yes | yes | yes | yes | yes | yes |
| P/E core classification | yes (sysfs hybrid PMUs, capacity) | yes (capacity) | yes (`EfficiencyClass`) | yes (`hw.perflevelN`) | yes | part (x86: CPUID leaf 0x1A; ARM: none) |
| `pin_thread_to` | yes | yes | yes | no | no | yes |
| `pin_thread_to_kind` | yes | yes | yes | part (QoS class, a hint) | part (QoS) | yes |
| `numa_nodes`, `numa_alloc`, `bind_thread_to_node`, `numa_node_of_address` | yes | yes | yes (`total_bytes` 0 on multi-node) | part (one node) | part (one node) | yes (`total_bytes` 0 on multi-node; no address lookup) |
| `huge_alloc`, `huge_page_sizes` | yes (2M / 1G, THP) | part (THP) | yes (needs "Lock pages in memory") | part (2M on Intel only) | no (falls back to normal pages) | yes |
| `advise_huge`, `thp_enabled` | yes | yes | no | no | no | yes (automatic promotion) |

### Power, pressure, counters, devices
| API | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `power_status` battery / AC | yes | part (sysfs where readable) + hook (`power_hooks`) | yes | yes | yes | yes |
| `power_status` thermal state / temperature | yes | hook | part (frequency cap heuristic, no temperature) | part (state, no temperature without SMC) | part (state) | yes (ACPI zone 0) |
| `memory_pressure` | yes (PSI, cgroup first) | yes (PSI) + hook (`on_trim_memory`) | yes | yes | yes | part (1 s poll) |
| `cycle_counter`, `counter_hz`, `precise_sleep_ns` | yes | yes | yes (MSVC ARM64: ns) | yes | yes | yes |
| `perf_counters` cycles | yes | yes (when seccomp allows) | yes | yes | part (CNTVCT) | yes |
| `perf_counters` instructions | yes | yes | no | yes | no | yes (hwpmc) |
| `perf_counters` cache / branch events | yes | yes | no | no (kperf is private and root-only) | no | yes (hwpmc) |
| `gpu_devices` | yes | part (kgsl / mali / EGL property) | yes | yes | part (one integrated GPU from the model) | yes |
| `devices::block`, `pci`, `usb` | yes | part (sysfs where readable) | yes | yes | no | yes |
| `devices::monitor` (hot-plug) | yes (uevent) | part (uevent refused for apps: polling) | yes (cfgmgr32) | yes (IOKit) | no | yes (devd) |
| `volumes`, `volume_for` | yes | yes | yes | yes | yes | yes (removable flag unknown) |
| `displays` | yes (DRM + EDID; X11 screen fallback) | hook (`set_display_hook`) | yes | yes | hook | part (X11 screen only) |
| `idle_seconds` | part (X11 exact; logind coarse) | no | yes | yes | no | part (X11 only) |
| `screen_locked` | yes (logind) | no | yes | yes | no | no (nothing records it) |
| `clipboard::get` / `set` | part (needs wl-clipboard, xclip or xsel) | hook (`clipboard::set_backend`) | yes | yes | yes (UIPasteboard) | part (same helpers) |

### Libraries and file watching
| API | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `dynlib::open`, `get`, `require`, `path`, `find_library`, `file_symbols`, `file_format` | yes | yes | yes | yes | yes | yes |
| `dynlib::symbols` (loaded image) | yes | yes | yes | yes | yes | yes |
| `dynlib::open_with` flags | yes | yes | part (`global`, `lazy` have no meaning) | part (`deps_beside` has no meaning on POSIX) | part | part |
| `loaded_modules`, `module_of_address` | yes | yes | yes | yes | yes | yes |
| `file_watcher::add`, `remove`, `poll`, overflow events | yes | yes | yes | yes | yes | yes |
| `file_watcher::add_file` (in-place writes) | yes | yes | yes | yes | yes | yes |
| `file_watcher::add_tree` (one recursive watch) | no (`tree_watcher` registers each directory) | no | yes | yes (FSEvents) | no | no |
| `tree_watcher` | yes | yes | yes | yes | yes (kqueue, descriptor cap → polling) | yes (descriptor cap → polling) |

A rename is reported the same way on every OS: the old name as `renamed`, the new name as `created`. The two events
are not paired (kqueue and FSEvents cannot pair them).

### Services, scheduling, background work
| API | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `service::manager`, `log` | yes | yes (liblog) | yes | yes | yes | yes |
| `service::notify` / `ready` / `watchdog` | yes (systemd) | no-op | no-op (SCM status is set by `run_as_service`) | no-op | no-op | no-op |
| `run_as_service` | runs the body | runs the body | yes (SCM) | runs the body | runs the body | runs the body |
| `service::install` / `start` / `stop` / `restart` / `query` / `uninstall`, user scope | yes (systemd user) | no | yes (Task Scheduler) | yes (LaunchAgent) | no | no (no per-user manager) |
| same, system scope | yes | no | yes (SCM) | yes (LaunchDaemon) | no | yes (rc.d) |
| `service::logs` | yes (journal) | no | yes (Application event log) | part (unified log by process name, slow) | no | part (`/var/log/messages` by tag) |
| `schedule::install` / `uninstall` / `installed` | yes (systemd timer, else crontab) | no | yes (`schtasks`) | yes (launchd) | no | part (crontab: whole minutes / hours only) |
| `background::begin` / `end` | yes (logind inhibitor; `not_required` without logind) | hook (`background::set_hooks`) | yes | yes | yes | part (`not_required`: no idle-suspend policy exists) |

### Users, settings, desktop integration
| API | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `current_user`, `lookup_user`, `lookup_uid`, `is_elevated` | yes | part (no group list) | yes | yes | yes | yes |
| `list_users`, `list_groups`, `lookup_group` | yes | part (API 26+; an app sees itself) | yes (local accounts) | yes | part (two accounts) | yes |
| `is_admin` | yes | no (false) | yes | yes | no (false) | yes |
| `user_directory` (all 17 kinds) | yes | part (needs `HOME`; app dirs come from `mobile::paths`) | yes | yes | yes | yes |
| `login_sessions` | yes (utmpx) | no | yes (WTS) | yes | no | yes |
| `run_elevated` without a prompt | yes (`sudo -n` / `doas -n`) | no | part (only when already elevated) | yes (`sudo -n`) | no | yes |
| `run_elevated` with a prompt | yes (pkexec / sudo) | no | yes (UAC) | yes (osascript) | no | yes |
| `registry::*` | no | no | yes | no | no | no |
| `preferences` | yes (XDG file) | part (file under `options.dir`) | yes (registry) | yes (CFPreferences) | yes (CFPreferences) | yes (XDG file) |
| `settings::*` (environment table) | yes | yes | yes | yes | yes | yes |
| `single_instance` | yes | yes (`set_directory`) | yes | yes | yes | yes |
| `autostart` | yes (XDG autostart) | no | yes (Run key) | yes (LaunchAgent) | no | yes (XDG autostart) |
| `open_url`, `open_path` | part (desktop session + xdg-open / gio) | hook (`set_open_hook`) | yes | yes | `open_url` yes, `open_path` no | part (as Linux) |
| `reveal_path` | part (FileManager1 over gdbus, else opens the folder) | hook | yes | yes | no | part |
| `notify` | part (session bus + notify-send or gdbus) | hook (`set_notification_hook`) | yes | yes (osascript) | hook | part (as Linux) |
| `trash` | yes (freedesktop spec, native) | no | yes | yes | no | yes |

### Mobile (`mobile.hpp`, `forge_mobile`)
Summarised from the backend classes in `os/src/mobile` (each overrides the `backend` virtuals; a missing override is
`unsupported`). [mobile.md](mobile.md) lists the mechanisms.

| Area | Android | iOS | Linux | Windows | macOS | FreeBSD |
|---|---|---|---|---|---|---|
| Motion / environment sensors | yes (ASensorManager) | yes (CoreMotion; no light sensor) | yes (IIO) | yes (Sensor API) | part (ambient light, lid) | part (temperature) |
| Location, GNSS | hook | yes | yes (GeoClue2) | hook (WinRT) | yes | no |
| Touch, gestures | yes | hook (`feed_touch` from UIKit) | yes (evdev) | hook (`windows_feed_pointer`) | hook | yes (evdev) |
| Display, orientation, brightness, keep awake | hook / yes | yes | yes | yes | yes | part (backlight(9); keep awake under X11 only) |
| Cameras, torch, photo | yes (camera2ndk) | yes | yes (V4L2, MJPEG only) | yes (Media Foundation) | yes | no |
| Audio record / play | hook | hook | yes (arecord / aplay) | yes (WASAPI) | hook | hook |
| Haptics | hook | yes | yes (force feedback) | hook | no | no |
| Notifications, clipboard, open URL, share | hook | yes | yes (helpers) | yes | yes | part (open URL) |
| Bluetooth scan, NFC, biometrics | hook | hook / yes | yes (BlueZ, fprintd) | part (classic inquiry) | part (Touch ID) | no |
| Lifecycle, permissions | hook | yes | not required | not required | yes | not required |
| Simulator (JSON trace), null device | yes | yes | yes | yes | yes | yes |

### Closed in this pass
| Gap | Now |
|---|---|
| No link-level interface data, routes, DNS servers, proxy or change notifications on any OS | `netinfo.hpp` on all six |
| No keep-alive tuning, fast open, reuseport, dual stack, AF_UNIX on Windows, multicast, mDNS, happy eyeballs | `sockopt.hpp` on all six |
| No way to ask whether https will work | `net::tls_backend()` |
| No machine id, boot time, container-aware CPU / memory budget (Windows job objects, FreeBSD rctl, iOS jetsam) | `sysinfo.hpp` |
| No VM / container / WSL / Rosetta / emulator detection | `virtualization.hpp` |
| No device hot-plug | `device_watch.hpp` (uevent, devd, IOKit, cfgmgr32, polling) |
| Clipboard unsupported on iOS and Android | UIPasteboard; `clipboard::set_backend` |
| No loaded-module list, no loader flags | `loaded_modules`, `module_of_address`, `dynlib::open_with` |
| No service restart or logs | `service::restart`, `service::logs` |
| No scheduled tasks | `scheduler.hpp` |
| No user / group enumeration, admin check, sessions, elevation | `users.hpp`, `sessions.hpp` |
| Registry was the only settings store (Windows only) | `preferences.hpp` |
| `service_control.hpp` did not compile on its own (used `fs::` without including `fs.hpp`) | include added |
| Missing everywhere: volumes, displays, idle / lock state, single instance, autostart, open URL / file, notifications, trash, more known folders | one header each, all six OSes answered |

### Not compiled or not run here
The work was done on Linux x86_64. Linux paths were built with clang 23 and run. Windows paths were built with
MinGW-w64 (g++ 13) and run under wine 9, and compiled with zig's `x86_64-windows-gnu`; wine implements only part of
iphlpapi, cfgmgr32 and the shell, so `network_monitor` (iphlpapi), `devices::monitor` (cfgmgr), toast notifications
and `run_elevated` with a prompt ran their fallback or error paths there, not the native one. FreeBSD and macOS
paths were compiled and linked with zig (`x86_64-freebsd`, `aarch64-macos`) and never run. The following have had
no compiler at all (no Android NDK or iOS SDK on the machine): every `FORGE_OS_ANDROID` / `__ANDROID__` branch added
(`dns_servers`, `machine_id`, the emulator check in `virtualization`, `list_users` API gate) and every
`FORGE_OS_IOS` branch added (`machine_id` identifierForVendor, `memory_limits` jetsam, `clipboard` UIPasteboard,
`shell_open` `ios_open`, the `links` path without `<net/if_media.h>` / `<net/route.h>`). MSVC has not compiled any
of it. Nothing in `os/src/mobile/apple.mm` or `os/src/text/locale_apple.mm` was changed.

### Inherent limits (not gaps)
- **macOS / iOS**: no hard CPU affinity, no NUMA, no CPU quotas, superpages only on Intel Macs, PMU events beyond
  cycles and instructions are private (kperf) and root-only, no temperature without SMC.
- **iOS**: no routing table header, no device registry, no services, no scheduler, no trash, no other apps' input
  state; multicast needs an entitlement.
- **Android**: ClipboardManager, Intents, notifications, ANDROID_ID, proxies, display metrics and foreground
  services are Java APIs; apps cannot bind netlink (API 30+) or read most of `/proc/net` and sysfs.
- **Windows**: no `SO_REUSEPORT`, no user-mode PMU, no transparent huge pages, no silent elevation, no client
  fast open without `ConnectEx`.
- **FreeBSD**: no per-user service manager, no idle inhibitor or lock-state registry, no display registry outside
  X11 / Wayland, cron cannot express arbitrary intervals, removable flag not exposed by `statfs`.
- **Linux**: desktop features (clipboard, open, notifications, idle) depend on a session and on helper programs;
  qemu-user translation cannot be detected. KDE proxy settings are not read.
- **File watching**: kqueue and FSEvents do not pair the two halves of a rename, so no backend reports pairs.
