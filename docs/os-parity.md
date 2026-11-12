# forge::os parity matrix

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
