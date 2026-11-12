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
