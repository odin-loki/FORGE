# mobile: sensors, touch and device services

## Concepts
`forge::os::mobile` (`#include <forge/os/mobile.hpp>`, link `forge_mobile`) is one API for sensors, touch, haptics, display, cameras, audio, permissions, notifications and app lifecycle. Backends exist for Android, iOS/iPadOS, macOS, Windows, Linux (desktops and Linux phones such as PinePhone and Librem 5) and FreeBSD, plus a null backend and a JSON-trace simulator.

- **Errors.** Calls return `os::result<T>` / `os::status`. Anything a platform cannot do returns `os::unsupported(...)` (`is_unsupported(r.err)` is true). Nothing throws.
- **RAII.** `device`, `sensor_stream`, `touch_stream` and `subscription` are move-only owners. Dropping a stream stops the hardware.
- **Memory.** Event queues are bounded (4096 events; a full queue drops the oldest and counts it in `dropped()`) and allocate through `forge::mem::budget_resource(host_budget(), "mobile")`.
- **App glue.** Some things only Java, Swift or WndProc code can see. The app pushes them in through the glue functions below.

```cpp
namespace m = forge::os::mobile;
auto dev = m::device::open();                 // platform backend; FORGE_MOBILE_TRACE=trace.json selects the simulator
auto acc = dev->open_sensor(m::sensor_kind::accelerometer, 100);
for (auto& e : acc->poll(50)) use(e.v[0], e.v[1], e.v[2]);
auto touch = dev->open_touch();
for (auto& g : touch->gestures(16)) if (g.kind == m::gesture_kind::pinch) zoom(g.scale);
auto sub = dev->on_lifecycle([](m::app_event e) { if (e == m::app_event::background) save(); });
```

## API reference
| Area | API |
|---|---|
| Opening | `device::open()` (platform backend, or the simulator when `FORGE_MOBILE_TRACE` is set), `null_device()`, `simulated(json, realtime)`, `simulated_file(path, realtime)`, `linux_with(sysfs_root, input_dir)` (Linux, for tests), `backend_name()` |
| Sensors (16 kinds) | accelerometer, gyroscope, magnetometer, light, proximity, pressure, location, battery, step_counter, step_detector, rotation_vector, gravity, linear_acceleration, heart_rate, temperature, humidity. `sensors()`, `has_sensor`, `open_sensor(kind, rate_hz)` (rate in (0, 10000]) → `sensor_stream` with `poll(timeout_ms, max)`, `dropped()`, `finished()`. `sensor_event::v` units are listed in the header |
| Touch | `open_touch()` → `touch_stream` with `poll` and `gestures`. `gesture_recognizer` (tap, double_tap, long_press, swipe, pinch, rotate) is a pure state machine you can feed yourself |
| Location | `location` sensor (lat, lon, altitude, accuracy), `gnss()` (fix, satellites), `geofence_monitor` (`add`, `remove`, `feed`; haversine enter/exit, works on any backend that produces location) |
| Display | `display()`, `screen_orientation()`, `lock_orientation`, `brightness` / `set_brightness`, `keep_awake`, `show_keyboard` |
| Camera / audio | `cameras()`, `torch`, `capture_photo` (JPEG bytes), `record_audio(ms, rate)`, `play_audio` |
| Haptics | `vibrate(ms, amplitude)` (ms in (0, 60000], amplitude in [0, 1]), `haptic_pattern({on, off, ...})` (falls back to repeated vibrate) |
| System | `power()`, `connectivity()`, `bluetooth_scan(timeout_ms)`, `nfc_available()`, `clipboard_get` / `clipboard_set`, `notify`, `biometrics()`, `info()` (incl. `form_factor`: phone, tablet, watch, tv, desktop, laptop, car), `paths()`, `open_url`, `share` |
| Permissions | `permission_status`, `request_permission` for sensors, location, camera, notifications, vibrate |
| Lifecycle | `on_lifecycle(cb)` → `subscription` for foreground, background, low_memory, terminate. `low_memory` calls `forge::mem::governor::on_pressure(critical)`, which drops every cache, whether or not anyone subscribed. Once a backend has a subscriber, `os::memory_pressure` critical events are reported as `low_memory` |
| Simulator | `sim_log()` returns the actions the simulator recorded |

App glue: `feed_touch`, `feed_sensor`, `feed_lifecycle`, `report_display`, `report_permission`, `set_action_hooks({vibrate, notify, request_permission})`, `set_service_hook(fn(op, json_args) -> optional<json>)`, `android_feed_motion(AInputEvent*)` (Android), `windows_feed_pointer(hwnd, msg, wparam, lparam)` (Windows). A service hook answers any device method by name (`"connectivity"`, `"torch"`, ...) with the same JSON shape as the simulator's `services` entries; returning `nullopt` falls through to the native backend.

## Backends
| OS | Native | Needs app glue or unsupported |
|---|---|---|
| Android (NDK, API 26+) | ASensorManager for accelerometer, gyroscope, magnetometer, light, proximity, pressure, gravity, linear acceleration, rotation vector, humidity, ambient temperature, step detector/counter, heart rate. Battery from `os::power_status`. Touch from `android_feed_motion`. Cameras through ACameraManager. Device info and form factor from system properties (`ro.build.characteristics`) | Location (FusedLocationProvider → `feed_sensor`), display (`report_display`), vibrate, notify and permission requests (`set_action_hooks`), and clipboard, share, biometrics, brightness, torch through `set_service_hook` |
| iOS / iPadOS (`apple.mm`) | CoreMotion (accelerometer, gyroscope, magnetometer), CMAltimeter (pressure), CoreLocation, UIDevice battery, touch through `FORGETouchView`, UIImpactFeedbackGenerator, UIScreen (display, brightness), idle-timer keep-awake, AVCapture cameras and torch, LAContext biometrics, UIDevice info (pad → tablet, TV, CarPlay), openURL, UNUserNotificationCenter, lifecycle and memory warnings | Light and proximity are not exposed by iOS. watchOS uses the null backend + glue |
| macOS (`apple.mm`) | IOPMAssertion keep-awake, NSWorkspace `open_url`, device info (laptop when a battery is present, else desktop), plus the portable power, clipboard and connectivity services | Trackpad touch and CoreLocation through `feed_touch` / `feed_sensor`; CoreBluetooth through the service hook |
| Windows | WM_POINTER touch and pen pressure (`windows_feed_pointer`), display (GetSystemMetrics, DPI, EnumDisplaySettings, rotation), battery, `SetThreadExecutionState` keep-awake, `ShellExecuteW` `open_url`, form factor (`SM_CONVERTIBLESLATEMODE`, battery) | WinRT sensors, geolocation, Bluetooth and toast notifications through `feed_sensor` / the service hook |
| Linux | IIO sensors (accel, gyro, magnetometer, light, proximity, pressure, temperature, humidity, steps, gravity), power_supply battery, evdev multitouch (protocol B) and single touch, force-feedback rumble, DRM / fbdev display with accelerometer orientation, V4L2 cameras, backlight brightness, LED-flash torch, `/sys/class/nfc`, NetworkManager (`nmcli`) and ModemManager (`mmcli`), BlueZ (`bluetoothctl`) scan, fprintd biometrics, DMI / device-tree info, `arecord` / `aplay`, `notify-send`. Permissions are `not_required` | Location (needs GeoClue). `FORGE_MOBILE_SYSFS` and `FORGE_MOBILE_INPUT` relocate the sysfs and input trees |
| FreeBSD | Temperature from `dev.cpu.0.temperature` or the ACPI thermal zone, ACPI battery, evdev multitouch | Display through `report_display`; devd hot-plug forwarded by the app |
| null | — | Everything unsupported |

## Simulator (JSON trace)
`device::simulated(json)`, `simulated_file(path)` or `FORGE_MOBILE_TRACE=<file>`. Keys:

| Key | Content |
|---|---|
| `display`, `sensors`, `cameras`, `permissions` | Static device description |
| `services` | One entry per service op (`power`, `connectivity`, `bluetooth_scan`, `info`, `gnss`, `clipboard`, `brightness`, `photo`, `audio`, ...). A missing op returns unsupported |
| `events` | `sensor`, `touch`, `orientation` and `lifecycle` events, each with `t_ms` |

Actions (vibrate, notify, torch, open_url, ...) are recorded in `sim_log()` and honour `permissions`; for example `torch` fails when camera permission is denied. With `realtime = true` events are replayed at their `t_ms`.

## Scripting
`forge::mobile::device`, `forge::sensors::*` and `forge::touch::open` in [scripting.md](scripting.md).

## Platform status and tests
| Platform | Status |
|---|---|
| Linux x86_64 | `test_mobile` (simulator, gestures incl. pinch/rotate, services, lifecycle → memory pressure, geofencing, Linux backend on a fake sysfs + evdev FIFO) and `test_script` `mobile_sensors_touch` pass |
| Windows (MSVC, MinGW), macOS arm64, FreeBSD 14.3 | `test_mobile` passes in CI. It exercises the simulator and the backend's `open()`, not real sensors |
| Android | CI builds `test_script` (which links `forge_mobile`) for x86_64 with the NDK and runs it on an API 34 emulator. Not run against real sensors or on a device |
| iOS / iPadOS | The UIKit part of `apple.mm` has never been compiled (`mobile_ios` skips without an Xcode SDK; the CI `ios` job does not build `forge_mobile`) |

With `zig` installed, `xos_mobile_<target>` compiles every C++ backend for FreeBSD, macOS, MinGW and aarch64 musl.
