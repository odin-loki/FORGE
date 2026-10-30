// forge_mobile Windows backend (Win32; tablets, 2-in-1s and desktops).
//   touch     WM_POINTER messages from the app's WndProc via windows_feed_pointer (GetPointerInfo /
//             GetPointerPenInfo pressure); pinch/rotate come from the shared gesture recognizer.
//   display   GetSystemMetrics + GetDpiForSystem + EnumDisplaySettings; orientation from DEVMODE.
//   power     GetSystemPowerStatus (forge::os::power).  keep_awake: SetThreadExecutionState.
//   open_url  ShellExecuteW.  form factor: SM_CONVERTIBLESLATEMODE / battery presence.
//   Sensors, geolocation, Bluetooth and toast notifications live in WinRT (Windows.Devices.Sensors,
//   Windows.Devices.Geolocation, Windows.Devices.Bluetooth, ToastNotificationManager); a C++/WinRT or
//   C# app supplies them through set_service_hook / feed_sensor. Without that they are not_supported.
#include "internal.hpp"
#if defined(_WIN32)
#include <windows.h>
#include <shellapi.h>

namespace forge::os::mobile {
bool windows_feed_pointer(void* hwnd, unsigned msg, std::uintptr_t wp, std::intptr_t) {
#ifndef WM_POINTERDOWN
    (void)hwnd; (void)msg; (void)wp; return false;
#else
    if (msg != WM_POINTERDOWN && msg != WM_POINTERUPDATE && msg != WM_POINTERUP && msg != WM_POINTERCAPTURECHANGED) return false;
    using get_info_t = BOOL(WINAPI*)(UINT32, POINTER_INFO*);
    static auto get_info = reinterpret_cast<get_info_t>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"user32.dll"), "GetPointerInfo")));
    if (!get_info) return false;
    UINT32 id = UINT32(wp & 0xFFFF); POINTER_INFO pi{};
    if (!get_info(id, &pi)) return false;
    POINT pt = pi.ptPixelLocation; if (hwnd) ScreenToClient(static_cast<HWND>(hwnd), &pt);
    touch_phase ph = msg == WM_POINTERDOWN ? touch_phase::down : msg == WM_POINTERUP ? touch_phase::up : msg == WM_POINTERCAPTURECHANGED ? touch_phase::cancel : touch_phase::move;
    double pressure = 1;
    if (pi.pointerType == PT_PEN) {
        using pen_t = BOOL(WINAPI*)(UINT32, POINTER_PEN_INFO*);
        static auto pen = reinterpret_cast<pen_t>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"user32.dll"), "GetPointerPenInfo")));
        POINTER_PEN_INFO pp{}; if (pen && pen(id, &pp)) pressure = pp.pressure / 1024.0;
    }
    feed_touch({std::uint64_t(pi.PerformanceCount), ph, int(id), double(pt.x), double(pt.y), pressure, 1});
    return true;
#endif
}
} // namespace forge::os::mobile

namespace forge::os::mobile::detail {
namespace {
struct windows_backend final : backend {
    std::string name() const override { return "windows"; }
    result<std::vector<sensor_info>> sensors() override { return {{{sensor_kind::battery, "GetSystemPowerStatus", "Microsoft", 1, 100, 1}}, {}}; }
    result<sensor_stream> open_sensor(sensor_kind k, double hz) override {
        auto q = std::make_shared<queue>(); bridge::get().add_sensor(k, q);
        if (k != sensor_kind::battery) {
            // WinRT sensors arrive through feed_sensor from app code; without a provider this is unsupported.
            return {{}, unsupported(std::string("sensor ") + sensor_name(k) + " (Win32; supply via feed_sensor from WinRT)")};
        }
        auto w = std::make_shared<worker>(); auto period = std::uint64_t(1e9 / hz);
        w->start([q, wp = w.get(), period] {
            do { auto p = power_status(); if (p.battery_percent >= 0) q->push({false, {sensor_kind::battery, now_ns(), {double(p.battery_percent), p.charging ? 1.0 : 0.0}, 2}, {}}); } while (wp->sleep_ns(period));
        });
        return {make_sensor(k, q, w), {}};
    }
    result<touch_stream> open_touch() override { auto q = std::make_shared<queue>(); bridge::get().add_touch(q); return {make_touch(q, nullptr), {}}; }
    result<display_info> display() override {
        if (auto b = backend::display()) return b;
        display_info d; DEVMODEW dm{}; dm.dmSize = sizeof dm;
        if (EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &dm)) {
            d.width_px = int(dm.dmPelsWidth); d.height_px = int(dm.dmPelsHeight); d.refresh_hz = dm.dmDisplayFrequency;
            switch (dm.dmDisplayOrientation) { case 0: d.orient = d.width_px >= d.height_px ? orientation::landscape_left : orientation::portrait; break;
                case 1: d.orient = orientation::portrait; break; case 2: d.orient = orientation::landscape_right; break; case 3: d.orient = orientation::portrait_upside_down; break; }
        } else { d.width_px = GetSystemMetrics(SM_CXSCREEN); d.height_px = GetSystemMetrics(SM_CYSCREEN); }
        using dpi_t = UINT(WINAPI*)(); auto f = reinterpret_cast<dpi_t>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"user32.dll"), "GetDpiForSystem")));
        d.dpi = f ? f() : 96; d.scale = d.dpi / 96.0;
        return {d, {}};
    }
    status keep_awake(bool on) override {
        return SetThreadExecutionState(on ? (ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED) : ES_CONTINUOUS) ? status{} : status{{}, last_error("SetThreadExecutionState")};
    }
    status open_url(std::string_view u) override {
        std::string s(u); if (s.find("://") == std::string::npos && s.rfind("mailto:", 0) && s.rfind("tel:", 0)) return {{}, {22, "open_url: not a URL"}};
        int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0); std::wstring w(std::size_t(n), L'\0'); MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
        auto r = reinterpret_cast<INT_PTR>(ShellExecuteW(nullptr, L"open", w.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
        return r > 32 ? status{} : status{{}, {int(r), "ShellExecuteW failed"}};
    }
    result<device_info> info() override {
        auto d = backend::info(); if (!d) return d;
        bool slate = GetSystemMetrics(0x2003 /*SM_CONVERTIBLESLATEMODE*/) == 0, tablet = GetSystemMetrics(SM_TABLETPC) != 0;
        d.value.form = slate && tablet ? form_factor::tablet : power_status().has_battery ? form_factor::laptop : form_factor::desktop;
        d.value.manufacturer = "Microsoft"; return d;
    }
    result<permission_state> permission_status(permission p) override { if (auto b = backend::permission_status(p)) return b; return {permission_state::not_required, {}}; }
    status request_permission(permission) override { return {}; }
};
} // namespace
std::shared_ptr<backend> make_windows() { return std::make_shared<windows_backend>(); }
} // namespace forge::os::mobile::detail
#endif
