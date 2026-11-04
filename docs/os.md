# os: the portability layer

## Concepts
`forge::os` (`os/include/forge/os/`) is a header-only OS layer for Linux, Android, Windows, macOS, iOS and FreeBSD. The rest of FORGE (mapped files, sockets, RPC, async I/O, dynamic loading, entropy, the memory governor, the logger) goes through it.

- **Targets.** `forge_os` is a CMake INTERFACE target. It links `Threads::Threads`, plus `ws2_32` and `bcrypt` on Windows and `libdl` on Linux, Android and FreeBSD. Nothing else is needed at link time.
- **Late binding.** Optional system libraries are loaded at run time: `dbghelp`, `crypt32`, `powrprof`, `dxgi` and `dxcore` on Windows (system DLLs from System32 only); CoreFoundation, IOKit, Security, kperf and `libobjc` on Apple; `libpmc` on FreeBSD; `libsecret` and `liblog` on Linux and Android. With MSVC, `process.hpp` and `sandbox.hpp` add `psapi` and `userenv` through `#pragma comment`.
- **Errors are values.** Calls return `os::result<T>` (`.value`, `.err`, `operator bool`) or `os::status`. `error{code, what}` holds the native code and a message. Nothing in `os.hpp` throws.
- **Unsupported features.** When a platform lacks a feature the call returns `os::unsupported(what)`. Test it with `os::is_unsupported(err)` (code `os::unsupported_code`, `0x0F0E5`) and carry on.
- **Typed errors (optional).** `oserror.hpp` maps errno, Win32 and NTSTATUS codes to `os_errc` (`classify`, `classify_errno`, `classify_win32`), throws `os_exception` from `throw_if(err)`, turns fault signals into exceptions with `fault_guard`, and sets a flag on Ctrl-C / SIGTERM with `install_cancel_signals`.

Include `<forge/os/os.hpp>` for the core API, or `<forge/os/all.hpp>` for the core plus every header listed below except `process_pool.hpp`, `oserror.hpp`, `text.hpp`, `locale.hpp` and `mobile.hpp`. `text.hpp` and `locale.hpp` need the `forge_text` library ([text-locale.md](text-locale.md)); `mobile.hpp` needs `forge_mobile` ([mobile.md](mobile.md)).

## Core API (`os.hpp`)
| Area | API |
|---|---|
| Platform | `os::current()`, `platform_name()`, `arch_name()`; macros `FORGE_OS_WINDOWS`, `FORGE_OS_POSIX` and per-OS `FORGE_OS_LINUX` etc. |
| Files | `file` (open flags `rd`/`wr`/`create`/`truncate`/`append`/`excl`, `pread`/`pwrite`), `read_file`, `write_file`, `stat`, `exists`, `mkdir`, `mkdirs`, `remove`, `remove_all`, `rename` (atomic replace, also on Windows), `list_dir` |
| Paths | `join`, `basename`, `dirname`, `absolute`, `cwd`, `chdir`, `temp_dir`, `home_dir`, `exe_path` |
| Memory | `mapping` (mmap / file mapping), `page_size`, `aligned_alloc`/`aligned_free`, `reserve`/`commit`/`release`/`protect`, `lock_memory`, `advise_willneed` |
| Dynamic libraries | `library` (`open`, `symbol`, `get<F>`, `decorate("m")` → `libm.so` / `libm.dylib` / `m.dll`). `dynlib.hpp` adds search paths, typed lookup, `path()` and symbol listing ([os-wrapper.md](os-wrapper.md#dynamic-libraries-dynlibhpp)) |
| Environment | `getenv(k, default)`, `has_env`, `setenv`, `unsetenv` |
| Time | `monotonic_ns`, `wall_ns`, `cpu_time_ns`, `thread_cpu_time_ns`, `sleep_ns` |
| Threads | `thread`, `mutex`, `condvar`, `semaphore`, `lock_guard`, `thread_id`, `yield`, `set_thread_name`, `set_thread_priority`, `pin_thread` |
| Processes | `pid`, `run(argv, merge_stderr)` (blocking; unsupported on iOS), `serialized_fork` / `fork_mutex` (every FORGE fork holds one process-wide lock, so `pthread_atfork` handlers never run concurrently) |
| Signals | `on_signal(sig, handler)` (`interrupt`, `terminate`, `hangup`, `pipe`; console control events on Windows), `ignore_sigpipe` |
| System | `sysinfo()`, `cpu_count`, `total_memory`, `available_memory`, `hostname`, `entropy(p, n)` |
| Sockets | `socket` (TCP), `poller` (epoll / kqueue / WSAPoll), `completion_port` (IOCP on Windows, emulated elsewhere), `net_init` |

## Extension modules
One header each. The table lists the mechanism each header uses per platform.

| Header | Main API | Linux | Android | Windows | macOS / iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `numa.hpp` | `numa_nodes`, `bind_thread_to_node`, `numa_alloc`/`numa_free`, `numa_node_of_address` | sysfs + `mbind` | same | `VirtualAllocExNuma`, group affinity, nodes across groups, `QueryWorkingSetEx` | one node (UMA) | `cpuset_setdomain`, per-domain free pages |
| `hugepages.hpp` | `huge_page_sizes`, `huge_alloc`, `advise_huge`, `thp_enabled` | `MAP_HUGETLB` 2M/1G, else THP | THP | large pages (needs "Lock pages in memory"); 1G via `VirtualAlloc2` | superpages on x86_64 macOS; Apple Silicon and iOS fall back | `shm_create_largepage` 2M/1G, else `MAP_ALIGNED_SUPER` |
| `async_file.hpp` | `async_file_io` read/write/wait | io_uring (raw syscalls), else threads | same (seccomp often forces threads) | IOCP | POSIX AIO + `aio_suspend` | POSIX AIO + kqueue |
| `cpu_topology.hpp` | `cpu_topology_query`, `core_kind` P/E, caches, `pin_thread_to`, `pin_thread_to_kind` | sysfs `cpu_core`/`cpu_atom`, capacity | capacity (big.LITTLE) | `EfficiencyClass` | `hw.perflevelN` + QoS | sysctl |
| `power.hpp` | `power_status`, `power_hooks::report` | sysfs power_supply / thermal | sysfs + app hooks | `GetSystemPowerStatus` | NSProcessInfo, IOKit (macOS), UIDevice (iOS) | `hw.acpi.*` |
| `memory_pressure.hpp` | `memory_pressure::instance().subscribe`, `on_trim_memory`, `notify` | PSI triggers (the cgroup's own `memory.pressure` first) | PSI + `onTrimMemory` hook | memory resource notification | dispatch source | 1 s poll |
| `shared_memory.hpp` | `shared_memory::create/open/unlink` | `shm_open` | file under `$TMPDIR` | `CreateFileMappingW` `Local\forge-*` | `shm_open` | `shm_open` |
| `gpu_devices.hpp` | `gpu_devices()` (no compute runtime initialised) | PCI sysfs + DRM (NVIDIA names from the driver) | kgsl / mali / EGL property | DXGI + DXCore `IsIntegrated` | IOKit (macOS), model lookup (iOS) | `/dev/pci` |
| `local_ipc.hpp` | `local_listener`, `local_stream`, `local_endpoint` | AF_UNIX, abstract `@name` | same | named pipes | AF_UNIX | AF_UNIX |
| `crash.hpp` | `stack_trace`, `crash_handler::install`, `write_dump_now` | signal handler → `.crash` text report | same | SEH → minidump `.dmp` + text | signal handler | signal handler |
| `sandbox.hpp` | `apply_sandbox`, `run_sandboxed`, `sandbox_status` | `no_new_privs` + seccomp-BPF | same | job object; children in an AppContainer or restricted token | `sandbox_init` (macOS); iOS is always sandboxed | Capsicum |
| `keystore.hpp` | `keystore::put/get/remove`, `backend()` | libsecret, else file | backend set from JNI, else file | DPAPI | Keychain | file |
| `file_watch.hpp` | `file_watcher::add/remove/poll` (+ `overflow` events) | inotify | inotify | `ReadDirectoryChangesW` on one IOCP | kqueue (descriptor cap) | kqueue (descriptor cap) |
| `perf.hpp` | `cycle_counter`, `counter_hz`, `precise_sleep_ns`, `perf_counters` | `perf_event_open` (a group per hybrid PMU, multiplex scaling), else TSC | same | `QueryThreadCycleTime` (cycles only) | `mach_absolute_time`; kperf counters (root), else `proc_pid_rusage` cycles/instructions | libpmc (needs `hwpmc`), else TSC cycles |
| `background.hpp` | `background::begin/end`, `background_scope` | not required | JNI hooks, else unsupported | `SetThreadExecutionState` | IOPMAssertion / `beginBackgroundTask` | not required |
| `service.hpp` | `service::ready/reloading/stopping/watchdog`, `log`, `run_as_service` | `sd_notify` + journald, else syslog | liblog | SCM + Event Log | launchd detection + syslog | rc.d detection + syslog |

Process control, pipes, signals, `fs`, users, time, sysinfo, console, networking, recursive file watching, service control, registry, devices, clipboard, `dynlib` and `process_pool` are described in [os-wrapper.md](os-wrapper.md). `settings.hpp` is in [support.md](support.md).

Environment: `FORGE_AIO=threads` forces the portable async-file backend; `FORGE_KEYSTORE=file` and `FORGE_KEYSTORE_DIR` select the file keystore (see [configuration.md](configuration.md)).

## Example
`docs/examples/os_features.cpp` (ctest `doc_example_os_features`):

```cpp
auto t = os::cpu_topology_query();
auto nodes = os::numa_nodes();                               // always at least one node
auto shm = os::shared_memory::create(name, 4096);            // result<shared_memory>
if (!shm) { std::printf("shm failed: %s\n", shm.err.what.c_str()); return 1; }
auto pin = os::pin_thread_to_kind(os::core_kind::performance);
if (!pin && os::is_unsupported(pin.err)) { /* platform lacks it: carry on */ }
```

## Platform notes and limits
- **Local IPC names.** On Linux and Android, `local_endpoint(name)` returns the abstract socket `@forge-<name>`, because Android's SELinux policy refuses socket files in `/data/local/tmp`. Other POSIX systems use `<temp>/<name>.sock` unless the name is absolute. Windows uses the bare pipe name.
- **File keystore.** One mode-0600 file per secret in a 0700 directory, obfuscated with a per-store pad. It is a fallback, not real secret storage; `keystore::backend()` reports when it is in use.
- **Mobile hooks.** `power_hooks`, `memory_pressure::on_trim_memory`, `background::set_hooks` and `keystore::set_backend` are plain C++ entry points. The app's JNI or Swift code must call them.
- **Windows sandbox.** Network and filesystem denial cannot be added to a running process. `apply_sandbox` only blocks child processes; `run_sandboxed` starts a child under the full policy.
- **Tests.** `tests/test_os.cpp` (core) and `tests/test_os_ext.cpp` (one test per extension module). `test_os_ext` re-runs itself with `FORGE_OS_EXT_CHILD` set for the sandbox and crash cases, so never export that variable. When `zig` is installed, the `xos_*` ctests (label `cross;slow`) compile and link these tests for FreeBSD, macOS, MinGW Windows and aarch64 musl.

## Where it has run
| Platform | Status |
|---|---|
| Linux x86_64 (gcc 14, clang) | Full test suite passes, locally and in CI |
| Windows x86_64 (MSVC and MinGW-w64 UCRT) | `test_os*` pass natively in GitHub Actions CI |
| macOS arm64 | `test_os*` pass in GitHub Actions CI |
| FreeBSD 14.3 | `test_os*` pass in CI (VM) |
| Android x86_64 | Native tests ran on an API 34 emulator (local). CI runs only `test_script` on the emulator |
| aarch64 / riscv64 Linux (qemu-user, static musl) | `test_os`, `test_os_ext` and `test_os_process` fail under qemu-user because the static build has no `dlopen`, self-exec gets ENOEXEC and `/proc` RSS reads 0. Not a FORGE fault, but not a pass either |
| iOS | Only the code `forge_c` uses is compiled (CI `ios` job). Nothing has run |

Tests skip cases a platform does not support, so "pass" does not mean every feature exists there; the tables above say what is unsupported. See [STATUS.md](STATUS.md) for dates and details.
