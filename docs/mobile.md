# forge::os::mobile — sensors, touch and device services on every OS

`#include <forge/os/mobile.hpp>`, link `forge_mobile`. One API on Android, iOS/iPadOS, macOS, Windows,
Linux (desktop, phones such as PinePhone/Librem, tablets), FreeBSD, and wearables, plus a null backend and a
JSON-trace simulator. Anything a platform cannot do returns `os::unsupported(...)`: `is_unsupported(r.err)`
is true and nothing throws. Errors are `os::result<T>` / `os::status`. Queues and buffers are charged to
`forge::mem` (`budget_resource(host_budget(), "mobile")`). Streams, subscriptions and devices are move-only
RAII owners: dropping them stops the hardware.

```cpp
namespace m = forge::os::mobile;
auto dev = m::device::open();                 // platform backend; FORGE_MOBILE_TRACE=trace.json -> simulator
auto acc = dev->open_sensor(m::sensor_kind::accelerometer, 100);
for (auto& e : acc->poll(50)) use(e.v[0], e.v[1], e.v[2]);
auto touch = dev->open_touch();
for (auto& g : touch->gestures(16)) if (g.kind == m::gesture_kind::pinch) zoom(g.scale);
auto sub = dev->on_lifecycle([](m::app_event e) { if (e == m::app_event::background) save(); });
```

## Surface

| Area | API |
|---|---|
| Sensors (16) | accelerometer, gyroscope, magnetometer, light, proximity, pressure, location, battery, step_counter, step_detector, rotation_vector, gravity, linear_acceleration, heart_rate, temperature, humidity. `sensors()`, `has_sensor`, `open_sensor(kind, hz)` → `sensor_stream` (`poll(timeout, max)`, `dropped`, `finished`). Queue cap 4096 (oldest dropped). |
| Touch | `open_touch()` → `touch_stream` (`poll`, `gestures`). `gesture_recognizer` (tap, double_tap, long_press, swipe, pinch, rotate) is a pure state machine you can feed yourself. |
| GNSS / geofence | `location` sensor (lat, lon, alt, accuracy), `gnss()` (fix, satellites), `geofence_monitor` (haversine enter/exit on location events, every backend). |
| Display | `display()`, `screen_orientation()`, `lock_orientation`, `brightness`/`set_brightness`, `keep_awake`, `show_keyboard`. |
| Camera / audio | `cameras()`, `torch`, `capture_photo` (JPEG), `record_audio`, `play_audio`. |
| Haptics | `vibrate(ms, amp)`, `haptic_pattern({on, off, ...})` (falls back to repeated vibrate). |
| System | `power()` (battery, charging, low-power, thermal), `connectivity()`, `bluetooth_scan()` (BLE), `nfc_available()`, `clipboard_get/set`, `notify`, `biometrics()`, `info()` (incl. `form_factor`: phone, tablet, watch, tv, desktop, laptop, car), `paths()`, `open_url`, `share`. |
| Permissions | `permission_status`, `request_permission` (sensors, location, camera, notifications, vibrate). |
| Lifecycle | `on_lifecycle(cb)` → `subscription`: foreground, background, low_memory, terminate. **low_memory drives `forge::mem::governor::on_pressure(critical)`**; on desktops `os::memory_pressure` critical is reported as low_memory. |

App glue (for things only Java/Swift/WndProc code can see): `feed_touch`, `feed_sensor`, `feed_lifecycle`,
`report_display`, `report_permission`, `set_action_hooks`, `set_service_hook(fn(op, json_args) -> optional<json>)`,
`android_feed_motion(AInputEvent*)`, `windows_feed_pointer(hwnd, msg, wparam, lparam)`.

## Backends

| OS | Native | Through glue / not_supported |
|---|---|---|
| Android (NDK) | ASensorManager for all 16 sensor types incl. step counter/detector, rotation vector, gravity, linear accel, heart rate, ambient temperature, humidity; AInputEvent touch; ACameraManager; battery; `__system_property_get` device info + form factor (watch/tv/car/tablet via `ro.build.characteristics`). | Location, notify, clipboard, share, biometrics, brightness, torch: Java side via `set_service_hook`. |
| iOS / iPadOS (`apple.mm`) | CoreMotion, CMAltimeter, CoreLocation, touch view, UIImpactFeedback, UIScreen (brightness), idleTimer keep-awake, AVCapture torch/cameras, LAContext biometrics, UIDevice info + idiom (pad → tablet), openURL, UNUserNotification, UIApplication lifecycle + memory warnings. | watchOS: null backend + glue. |
| macOS (`apple.mm`) | IOPMAssertion keep-awake, NSWorkspace open_url, device info (laptop/desktop), portable power/clipboard/connectivity. | Trackpad touches and CoreLocation via `feed_touch` / `feed_sensor` from the NSView/app; CoreBluetooth via service hook. |
| Windows | WM_POINTER (`windows_feed_pointer`, pen pressure), display (EnumDisplaySettings, DPI, rotation), battery, SetThreadExecutionState, ShellExecuteW, tablet/laptop/desktop form factor (SM_CONVERTIBLESLATEMODE). | WinRT sensors, geolocation, Bluetooth, toast notifications via feed_sensor / service hook. |
| Linux | IIO (accel, gyro, magn, light, proximity, pressure, temperature, humidity, steps, gravity), power_supply battery, evdev multitouch (protocol B), force-feedback rumble, DRM/fbdev display + accelerometer orientation, v4l2 cameras, backlight brightness, LED flash torch, `/sys/class/nfc`, NetworkManager (`nmcli`) + ModemManager (`mmcli`), BlueZ (`bluetoothctl`) scan, fprintd biometrics, DMI/device-tree info, `arecord`/`aplay`, `notify-send`. | `FORGE_MOBILE_SYSFS`, `FORGE_MOBILE_INPUT` relocate the trees (tests use fake ones). |
| FreeBSD | sysctl temperature (`dev.cpu.0.temperature`, ACPI thermal zone), ACPI battery, evdev multitouch (EVIOCGBIT). | Display via report_display; devd hot-plug forwarded by the app. |
| null | — | everything unsupported. |

## Simulator (JSON trace)

`device::simulated(json)` / `simulated_file(path)` / `FORGE_MOBILE_TRACE`. Keys: `display`, `sensors`,
`cameras`, `permissions`, `services` (one entry per service op: `power`, `connectivity`, `bluetooth_scan`,
`info`, `gnss`, `clipboard`, `brightness`, `photo`, `audio`, …; a missing op returns unsupported) and
`events` (`sensor`, `touch`, `orientation`, `lifecycle`, each with `t_ms`). Actions (vibrate, notify, torch,
open_url, …) are recorded in `sim_log()` and honour `permissions` (e.g. torch fails when camera is denied).

## Scripting

See `forge::mobile::device`, `forge::sensors::*`, `forge::touch::open` in [scripting.md](scripting.md).

## Testing

`test_mobile` (simulator, gestures incl. pinch/rotate, services, lifecycle → mem pressure, geofencing, Linux
backend on a fake sysfs + evdev FIFO). `test_script` `mobile_sensors_touch`. `xos_mobile_<target>` compiles
every C++ backend for FreeBSD, macOS, Windows (MinGW) and musl with zig. `apple.mm` needs an Apple SDK and is
only built on macOS CI; Android sources are checked with NDK r27c.
