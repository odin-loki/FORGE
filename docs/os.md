# os: the portability layer

## Concepts
`forge::os` (`os/include/forge/os/`) is a header-only OS abstraction (`forge_os` INTERFACE target) for **Linux, Android, Windows, macOS, iOS and FreeBSD**. The rest of FORGE (mapped files, sockets, RPC, async I/O, dynamic loading, entropy) goes through it.

- Include `<forge/os/os.hpp>` for the core API, or `<forge/os/all.hpp>` for the core plus all 16 extension modules.
- **No extra link flags.** Apple frameworks (CoreFoundation, IOKit, Security, objc, …) and Windows DLLs (dbghelp, crypt32, powrprof, dxgi, …) are bound at run time with `dlopen`/`LoadLibrary`. On Linux the target links only `libdl`; on Windows `ws2_32` and `bcrypt`.
- **Errors are values.** Calls return `result<T>` or `status`. When a platform lacks a feature the error satisfies `os::is_unsupported(err)` (code `os::unsupported_code`, `0x0F0E5`), so callers can degrade gracefully.

## Core API (`os.hpp`)
Files and filesystem; `mapping` (mmap / file mapping); processes (`run`, `exe_path`; `run` is unsupported on iOS); threads and synchronisation; time (`monotonic_ns`, `wall_ns`, `cpu_time_ns`, `sleep_ns`); TCP `socket`, `poller` (epoll / kqueue / WSAPoll) and `completion_port` (IOCP on Windows, emulated elsewhere); dynamic libraries (`library`); environment; signals (`on_signal`); memory; system info; standard paths; entropy; `os::rename` (atomic replace, also on Windows).

## Extension modules
One header each. The table lists the mechanism used per platform, as documented in each header.

| Header | Main API | Linux | Android | Windows | macOS / iOS | FreeBSD |
|---|---|---|---|---|---|---|
| `numa.hpp` | `numa_nodes`, `bind_thread_to_node`, `numa_alloc/free`, `numa_node_of_address` | sysfs + mbind | same | `VirtualAllocExNuma`, group affinity | single node (UMA) | `cpuset_setdomain` |
| `hugepages.hpp` | `huge_page_sizes`, `huge_alloc`, `advise_huge`, `thp_enabled` | MAP_HUGETLB 2M/1G, else THP | THP | large pages (needs "Lock pages in memory") | superpages on x86_64 macOS | super pages |
| `async_file.hpp` | `async_file_io` read/write/wait | io_uring (raw syscalls), else threads | same | IOCP | POSIX AIO | POSIX AIO + kqueue |
| `cpu_topology.hpp` | `cpu_topology_query`, `core_kind` P/E, caches, `pin_thread_to(_kind)` | sysfs `cpu_core`/`cpu_atom`, capacity | capacity (big.LITTLE) | `EfficiencyClass` | `hw.perflevelN` + QoS | sysctl |
| `power.hpp` | `power_status`, `power_hooks::report` | sysfs power_supply / thermal | sysfs + app hooks | `GetSystemPowerStatus` | IOKit / process info; iOS via hooks | sysctl |
| `memory_pressure.hpp` | `memory_pressure::instance().subscribe`, `on_trim_memory` | PSI triggers | PSI + `onTrimMemory` hook | memory resource notification | dispatch source | polling |
| `shared_memory.hpp` | `shared_memory::create/open/unlink` | shm_open | file under `$TMPDIR` | `CreateFileMappingW Local\forge-*` | shm_open | shm_open |
| `gpu_devices.hpp` | `gpu_devices()` (no runtime initialised) | PCI sysfs + DRM | kgsl / mali / EGL | DXGI | IOKit (macOS), model lookup (iOS) | `/dev/pci` |
| `local_ipc.hpp` | `local_listener`, `local_stream`, `local_endpoint` | AF_UNIX, abstract `@name` | same | named pipes | AF_UNIX | AF_UNIX |
| `crash.hpp` | `stack_trace`, `crash_handler::install`, `write_dump_now` | signal handler → `.crash` text report | same | SEH + minidump `.dmp` | signal handler | signal handler |
| `sandbox.hpp` | `apply_sandbox`, `run_sandboxed`, `sandbox_status` | no_new_privs + seccomp-BPF | same | job object / restricted token for children; in-process net/fs denial unsupported | `sandbox_init` (macOS); iOS already sandboxed | Capsicum |
| `keystore.hpp` | `keystore::put/get/remove` | libsecret, else file | backend set from JNI, else file | DPAPI | Keychain | file |
| `file_watch.hpp` | `file_watcher::add/poll` | inotify | inotify | `ReadDirectoryChangesW` | kqueue | kqueue |
| `perf.hpp` | `cycle_counter`, `precise_sleep_ns`, `perf_counters` | perf_event_open, else TSC | same | thread cycle time / TSC | kperf (root) / CNTVCT | TSC |
| `background.hpp` | `background::begin/end`, `background_scope` | not required | JNI hooks, else unsupported | `SetThreadExecutionState` | IOPMAssertion / `beginBackgroundTask` | not required |
| `service.hpp` | `service::ready/reloading/stopping/watchdog`, `log`, `run_as_service` | sd_notify + journald, else syslog | liblog | SCM + Event Log | launchd detection + syslog | rc.d detection + syslog |

Environment: `FORGE_AIO=threads`, `FORGE_KEYSTORE=file`, `FORGE_KEYSTORE_DIR` (see [configuration.md](configuration.md)).

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
- **Local IPC names:** on Linux and Android `local_endpoint(name)` gives an abstract socket `@forge-<name>`, because Android's SELinux policy refuses socket files in `/data/local/tmp`. Other POSIX systems use a `.sock` file in the temp directory unless the name is absolute.
- **File keystore** writes mode-0600 files obfuscated with a per-store pad. It is a fallback, not real secret storage.
- **Mobile hooks** (`power_hooks`, `memory_pressure::on_trim_memory`, `background::set_hooks`, `keystore::set_backend`) are plain C++ entry points that the app's JNI or Swift code must call.
- **Tests:** `tests/test_os.cpp` (core) and `tests/test_os_ext.cpp` (one test per module; never export `FORGE_OS_EXT_CHILD`). `xos_*` ctests cross-link them for other OSes.
- **Where it has run:** Linux, Windows under wine, the Android x86_64 emulator. FreeBSD and macOS are link-checked only; iOS has not been built. See [STATUS.md](STATUS.md).
