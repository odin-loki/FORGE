# `forge::os` — portable OS layer

Header-only (`forge_os` is an INTERFACE target). Include `<forge/os/os.hpp>` for the core or `<forge/os/all.hpp>` for everything.
Targets: Linux, Android, Windows, macOS, iOS and FreeBSD. Apple frameworks (CoreFoundation, IOKit, Security, objc, kperf,
libdispatch) and Windows DLLs (dbghelp, crypt32, powrprof, dxgi) are late-bound with `dlopen`/`LoadLibrary`, so no extra link
flags are needed. Every call returns `result<T>`/`status`; when the OS lacks a feature the error satisfies
`os::is_unsupported(err)` (code `0x0F0E5`) instead of crashing.

## Core (`os.hpp`)
Files and filesystem, `mapping` (mmap / file mapping), processes (`run`, `exe_path`; `run` unsupported on iOS), threads and sync,
time, TCP `socket`, `poller` (epoll / kqueue / WSAPoll), `completion_port` (IOCP, emulated elsewhere), `library` (dl), env,
signals (`on_signal`), memory, sysinfo, paths and entropy.

## Extension modules
| # | Header | API | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|---|---|
| 1 | `numa.hpp` | `numa_nodes`, `bind_thread_to_node`, `numa_alloc/free`, `numa_node_of_address` | sysfs + mbind + move_pages | same | `VirtualAllocExNuma`, group affinity | single node | single node | `cpuset_setdomain` |
| 2 | `hugepages.hpp` | `huge_page_sizes`, `huge_alloc(n, kind, fallback)`, `advise_huge`, `thp_enabled` | MAP_HUGETLB 2M/1G, THP | THP | large pages (needs SeLockMemoryPrivilege) | superpages (VM_FLAGS) | normal | `MAP_ALIGNED_SUPER` |
| 3 | `async_file.hpp` | `async_file_io` read/write/wait | io_uring (fallback threads) | io_uring/threads | IOCP | POSIX AIO | POSIX AIO | kqueue AIO |
| 4 | `cpu_topology.hpp` | `cpu_topology_query`, P/E `core_kind`, caches, `pin_thread_to(_kind)` | sysfs cpu_core/cpu_atom, capacity | capacity (big.LITTLE) | `EfficiencyClass` | perflevels + QoS | perflevels + QoS | sysctl + CPUID |
| 5 | `power.hpp` | `power_status`, `power_hooks::report` | sysfs power_supply/thermal | sysfs + hooks | `GetSystemPowerStatus`, powrprof | IOKit / NSProcessInfo | hooks | sysctl acpi |
| 6 | `memory_pressure.hpp` | `memory_pressure::instance().subscribe`, `on_trim_memory` | PSI triggers | PSI / `onTrimMemory` glue | memory resource notification | dispatch source | dispatch source | poll fallback |
| 7 | `shared_memory.hpp` | `shared_memory::create/open/unlink` | shm_open | file in TMPDIR | `CreateFileMapping Local\forge-*` | shm_open | shm_open | shm_open |
| 8 | `gpu_devices.hpp` | `gpu_devices()` | PCI sysfs + DRM | kgsl/mali/DRM | DXGI | IOKit IOAccelerator | Metal device name | /dev/pci |
| 9 | `local_ipc.hpp` | `local_listener`, `local_stream` | AF_UNIX + abstract `@name` | AF_UNIX | overlapped named pipes | AF_UNIX | AF_UNIX | AF_UNIX |
| 10 | `crash.hpp` | `stack_trace`, `crash_handler::install`, `write_dump_now` | signal handler, `.crash` report | same | SEH + minidump `.dmp` | signal handler | signal handler | signal handler |
| 11 | `sandbox.hpp` | `apply_sandbox`, `run_sandboxed`, `sandbox_status` | seccomp BPF | seccomp | job object (exec) / restricted token for children; in-process net/fs → unsupported | `sandbox_init` profiles | already sandboxed (no-op) | Capsicum `cap_enter` |
| 12 | `keystore.hpp` | `keystore::put/get/remove` | libsecret, else file | file | DPAPI | Keychain | Keychain | file |
| 13 | `file_watch.hpp` | `file_watcher::add/poll` | inotify | inotify | `ReadDirectoryChangesW` | kqueue | kqueue | kqueue |
| 14 | `perf.hpp` | `cycle_counter`, `precise_sleep_ns`, `perf_counters` | perf_event_open, falls back to TSC | same | `QueryThreadCycleTime` / TSC | kperf (root) | CNTVCT | TSC |
| 15 | `background.hpp` | `background::begin/end`, `background_scope` | not required | app hooks (JNI) | `SetThreadExecutionState` | IOPMAssertion | `beginBackgroundTask` | not required |
| 16 | `service.hpp` | `service::ready/reloading/stopping/watchdog`, `log`, `run_as_service` | sd_notify + journald | liblog | SCM + Event Log | syslog | os_log via syslog | syslog |

Environment knobs: `FORGE_AIO=threads` (force the thread backend), `FORGE_KEYSTORE=file`, `FORGE_KEYSTORE_DIR`.
The file keystore stores 0600 files obfuscated with a per-store pad; it is a fallback, not real secret storage.
Android/iOS hooks (`power_hooks`, `memory_pressure::on_trim_memory`, `background::set_hooks`) are plain C++ entry points
for the app's JNI or Swift glue to call.

Tests: `tests/test_os.cpp` (core), `tests/test_os_ext.cpp` (one TEST per feature above; child-process modes via
`FORGE_OS_EXT_CHILD`). Cross compile+link checks: the `xos_*` ctests (label `cross;slow`). See `docs/STATUS.md` for which
platforms were actually run.
