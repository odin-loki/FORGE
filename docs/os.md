# os: the portable OS layer

## Concepts
`forge::os` is header-only (`forge_os` is an INTERFACE target). Include `<forge/os/os.hpp>` for the core, or `<forge/os/all.hpp>` for everything.
- **Targets:** Linux, Android, Windows, macOS, iOS and FreeBSD.
- **No extra link flags:** Apple frameworks (CoreFoundation, IOKit, Security, objc, kperf, libdispatch) and Windows DLLs (dbghelp, crypt32, powrprof, dxgi) are loaded late with `dlopen` or `LoadLibrary`.
- **Errors:** every call returns `result<T>` or `status`. If the OS lacks a feature, the error satisfies `os::is_unsupported(err)` (`unsupported_code = 0x0F0E5`) instead of crashing.

## API reference

### Core (`os.hpp`)
- Files and filesystem, and `mapping` (mmap or Windows file mapping).
- Processes: `run` and `exe_path`. `run` is unsupported on iOS.
- Threads and sync, time (`monotonic_ns`, `wall_ns`, `cpu_time_ns`, `sleep_ns`).
- Networking: TCP `socket`, `poller` (epoll, kqueue or WSAPoll) and `completion_port` (IOCP, emulated on other platforms).
- Dynamic libraries (`library`), env, signals (`on_signal`), memory, sysinfo, paths and entropy.

### Extension modules (one header each, all included by `all.hpp`)
Each column shows the backend used on that platform.

| # | Header | API | Linux | Android | Windows | macOS | iOS | FreeBSD |
|---|---|---|---|---|---|---|---|---|
| 1 | `numa.hpp` | `numa_nodes`, `bind_thread_to_node`, `numa_alloc/free`, `numa_node_of_address` | sysfs + mbind + move_pages | same | `VirtualAllocExNuma`, group affinity | single node | single node | `cpuset_setdomain` |
| 2 | `hugepages.hpp` | `huge_page_sizes`, `huge_alloc(n, kind, fallback)`, `advise_huge`, `thp_enabled` | MAP_HUGETLB 2M/1G, THP | THP | large pages (needs SeLockMemoryPrivilege) | superpages (VM_FLAGS) | normal | `MAP_ALIGNED_SUPER` |
| 3 | `async_file.hpp` | `async_file_io` read/write/wait | io_uring (fallback threads) | io_uring/threads | IOCP | POSIX AIO | POSIX AIO | kqueue AIO |
| 4 | `cpu_topology.hpp` | `cpu_topology_query`, P/E `core_kind`, caches, `pin_thread_to(_kind)` | sysfs cpu_core/cpu_atom, capacity | capacity (big.LITTLE) | `EfficiencyClass` | perflevels + QoS | perflevels + QoS | sysctl + CPUID |
| 5 | `power.hpp` | `power_status`, `power_hooks::report` | sysfs power_supply/thermal | sysfs + hooks | `GetSystemPowerStatus`, powrprof | IOKit / NSProcessInfo | hooks | sysctl acpi |
| 6 | `memory_pressure.hpp` | `memory_pressure::instance().subscribe`, `on_trim_memory` | PSI triggers | PSI / `onTrimMemory` glue | memory resource notification | dispatch source | dispatch source | poll fallback |
| 7 | `shared_memory.hpp` | `shared_memory::create/open/unlink` | shm_open | file in TMPDIR | `CreateFileMapping Local\forge-*` | shm_open | shm_open | shm_open |
| 8 | `gpu_devices.hpp` | `gpu_devices()` | PCI sysfs + DRM | kgsl/mali/DRM | DXGI | IOKit IOAccelerator | integrated GPU from `hw.machine` | /dev/pci `PCIOCGETCONF` |
| 9 | `local_ipc.hpp` | `local_listener`, `local_stream` | AF_UNIX + abstract `@name` | AF_UNIX | overlapped named pipes | AF_UNIX | AF_UNIX | AF_UNIX |
| 10 | `crash.hpp` | `stack_trace`, `crash_handler::install`, `write_dump_now` | signal handler, `.crash` report | same | SEH + minidump `.dmp` | signal handler | signal handler | signal handler |
| 11 | `sandbox.hpp` | `apply_sandbox`, `run_sandboxed`, `sandbox_status` | seccomp BPF | seccomp | job object (exec) / restricted token for children; in-process net/fs → unsupported | `sandbox_init` profiles | already sandboxed (no-op) | Capsicum `cap_enter` |
| 12 | `keystore.hpp` | `keystore::put/get/remove` | libsecret, else file | JNI-installed backend (`set_backend`), else file | DPAPI | Keychain | Keychain | file |
| 13 | `file_watch.hpp` | `file_watcher::add/poll` | inotify | inotify | `ReadDirectoryChangesW` | kqueue | kqueue | kqueue |
| 14 | `perf.hpp` | `cycle_counter`, `precise_sleep_ns`, `perf_counters` | perf_event_open, falls back to TSC | same | `QueryThreadCycleTime` / TSC | kperf (root) | CNTVCT | TSC |
| 15 | `background.hpp` | `background::begin/end`, `background_scope` | not required | JNI hooks (`set_hooks`), else unsupported | `SetThreadExecutionState` | IOPMAssertion | `beginBackgroundTask` | not required |
| 16 | `service.hpp` | `service::ready/reloading/stopping/watchdog`, `log`, `run_as_service` | sd_notify + journald (else syslog) | liblog | SCM + Event Log | launchd detection + syslog (→ os_log) | syslog (→ os_log) | rc.d detection + syslog |

### Environment variables
- `FORGE_AIO=threads` forces the thread backend for `async_file_io`.
- `FORGE_KEYSTORE=file` forces the file keystore.
- `FORGE_KEYSTORE_DIR` sets the file keystore's directory.

## Example
`docs/examples/os_features.cpp` runs as the ctest `doc_example_os_features`:

```cpp
auto t = os::cpu_topology_query();
auto shm = os::shared_memory::create(name, 4096);   // result<shared_memory>
if (!shm) std::puts(shm.err.what.c_str());
auto pin = os::pin_thread_to_kind(os::core_kind::performance);
if (!pin && os::is_unsupported(pin.err)) { /* fine: the platform lacks it */ }
```

## Platform notes and limits
- **Local IPC names:** on Linux and Android, `local_endpoint(name)` produces an abstract socket name (`@forge-<name>`), because SELinux on Android refuses socket files in `/data/local/tmp`. Other POSIX systems use a `.sock` file in the temp directory unless the name is absolute.
- **File keystore:** stores files with mode 0600, obfuscated with a per-store pad. It is a fallback, not real secret storage.
- **Mobile hooks:** `power_hooks`, `memory_pressure::on_trim_memory` and `background::set_hooks` are plain C++ entry points for the app's JNI or Swift code to call.
- **Tests:**
  - `tests/test_os.cpp` covers the core.
  - `tests/test_os_ext.cpp` has one TEST per extension module. Its child-process modes are selected with `FORGE_OS_EXT_CHILD`; never export that variable in a shell.
  - The `xos_*` ctests (label `cross`) cross-compile and link the tests for other platforms.
- **Where it has actually run:** Linux, wine and the Android emulator. FreeBSD and macOS are link-only, and iOS has not been built. See [STATUS.md](STATUS.md).
