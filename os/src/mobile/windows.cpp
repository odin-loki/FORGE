// forge_mobile Windows backend (Win32; tablets, 2-in-1s and desktops).
//   touch     WM_POINTER messages from the app's WndProc via windows_feed_pointer (GetPointerInfo /
//             GetPointerPenInfo pressure, GetPointerFrameInfo for the contact count, QPC timestamps on the
//             monotonic_ns clock); pinch/rotate come from the shared gesture recognizer.
//   display   GetSystemMetrics + GetDpiForSystem + EnumDisplaySettings; orientation from DEVMODE.
//   power     GetSystemPowerStatus (forge::os::power).  keep_awake: a power request object
//             (PowerCreateRequest / PowerSetRequest, display + system required), process-wide.
//   open_url  ShellExecuteW, only for allow-listed URL schemes.  info: SMBIOS manufacturer/product from
//             HKLM\HARDWARE\DESCRIPTION\System\BIOS; form factor: SM_CONVERTIBLESLATEMODE / battery presence.
//   Sensors, geolocation, Bluetooth and toast notifications live in WinRT (Windows.Devices.Sensors,
//   Windows.Devices.Geolocation, Windows.Devices.Bluetooth, ToastNotificationManager); a C++/WinRT or
//   C# app supplies them through provide_sensor + feed_sensor / set_service_hook. Without that they are not_supported.
#include "internal.hpp"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>

namespace forge::os::mobile {
namespace {
// POINTER_INFO.PerformanceCount is a QPC tick count (0 when the source gave none); monotonic_ns uses the same counter.
std::uint64_t qpc_to_ns(std::uint64_t t) {
    static const std::uint64_t f = [] { LARGE_INTEGER q; QueryPerformanceFrequency(&q); return std::uint64_t(q.QuadPart); }();
    return t / f * 1000000000ull + t % f * 1000000000ull / f;
}
}  // namespace
bool windows_feed_pointer(void* hwnd, unsigned msg, std::uintptr_t wp, std::intptr_t) {
#ifndef WM_POINTERDOWN
    (void)hwnd; (void)msg; (void)wp; return false;
#else
    if (msg != WM_POINTERDOWN && msg != WM_POINTERUPDATE && msg != WM_POINTERUP && msg != WM_POINTERCAPTURECHANGED) return false;
    HMODULE u32 = GetModuleHandleW(L"user32.dll");
    using get_info_t = BOOL(WINAPI*)(UINT32, POINTER_INFO*);
    static auto get_info = reinterpret_cast<get_info_t>(reinterpret_cast<void*>(GetProcAddress(u32, "GetPointerInfo")));
    if (!get_info) return false;
    UINT32 id = UINT32(wp & 0xFFFF); POINTER_INFO pi{};
    if (!get_info(id, &pi)) return false;
    POINT pt = pi.ptPixelLocation; if (hwnd) ScreenToClient(static_cast<HWND>(hwnd), &pt);
    touch_phase ph = msg == WM_POINTERDOWN ? touch_phase::down : msg == WM_POINTERUP ? touch_phase::up : msg == WM_POINTERCAPTURECHANGED ? touch_phase::cancel : touch_phase::move;
    double pressure = 1;
    if (pi.pointerType == PT_PEN) {
        using pen_t = BOOL(WINAPI*)(UINT32, POINTER_PEN_INFO*);
        static auto pen = reinterpret_cast<pen_t>(reinterpret_cast<void*>(GetProcAddress(u32, "GetPointerPenInfo")));
        POINTER_PEN_INFO pp{}; if (pen && pen(id, &pp)) pressure = pp.pressure / 1024.0;
    }
    // Contacts in this pointer's frame (all fingers down together), not always 1.
    int pointers = 1;
    using frame_t = BOOL(WINAPI*)(UINT32, UINT32*, POINTER_INFO*);
    static auto frame = reinterpret_cast<frame_t>(reinterpret_cast<void*>(GetProcAddress(u32, "GetPointerFrameInfo")));
    if (frame) { POINTER_INFO buf[16]; UINT32 n = 16; if (frame(id, &n, buf) && n > 0) pointers = int(n); }
    std::uint64_t t = pi.PerformanceCount ? qpc_to_ns(std::uint64_t(pi.PerformanceCount)) : detail::now_ns();
    feed_touch({t, ph, int(id), double(pt.x), double(pt.y), pressure, pointers});
    return true;
#endif
}
} // namespace forge::os::mobile

namespace forge::os::mobile::detail {
namespace {
std::string reg_string(const wchar_t* key, const wchar_t* value) {
    wchar_t buf[256]; DWORD size = sizeof buf;
    if (RegGetValueW(HKEY_LOCAL_MACHINE, key, value, RRF_RT_REG_SZ, nullptr, buf, &size) != ERROR_SUCCESS) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, buf, -1, nullptr, 0, nullptr, nullptr); if (n <= 1) return {};
    std::string s(std::size_t(n - 1), '\0'); WideCharToMultiByte(CP_UTF8, 0, buf, -1, s.data(), n, nullptr, nullptr);
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}
// kernel32 power requests (Windows 7+), resolved at run time. REASON_CONTEXT with a simple reason string.
struct reason_context { ULONG version; DWORD flags; union { LPWSTR simple; struct { HMODULE m; ULONG id; ULONG count; LPWSTR* strings; } detailed; } reason; };
struct power_api {
    using create_t = HANDLE(WINAPI*)(reason_context*); using set_t = BOOL(WINAPI*)(HANDLE, int);
    create_t create = nullptr; set_t set = nullptr, clear = nullptr;
    power_api() {
        HMODULE k = GetModuleHandleW(L"kernel32.dll");
        create = reinterpret_cast<create_t>(reinterpret_cast<void*>(GetProcAddress(k, "PowerCreateRequest")));
        set = reinterpret_cast<set_t>(reinterpret_cast<void*>(GetProcAddress(k, "PowerSetRequest")));
        clear = reinterpret_cast<set_t>(reinterpret_cast<void*>(GetProcAddress(k, "PowerClearRequest")));
    }
};
constexpr int k_display_required = 0, k_system_required = 1;   // POWER_REQUEST_TYPE

struct windows_backend final : backend {
    std::mutex awake_m; HANDLE awake = nullptr; bool awake_on = false;
    ~windows_backend() override { if (awake) { keep_awake(false); CloseHandle(awake); } }
    std::string name() const override { return "windows"; }
    result<std::vector<sensor_info>> sensors() override { return {{{sensor_kind::battery, "GetSystemPowerStatus", "Microsoft", 1, 100, 1}}, {}}; }
    result<sensor_stream> open_sensor(sensor_kind k, double hz) override {
        // WinRT sensors arrive through feed_sensor from app code (provide_sensor); without a provider this is unsupported.
        if (k != sensor_kind::battery) return {{}, unsupported(std::string("sensor ") + sensor_name(k) + " (Win32; supply via provide_sensor + feed_sensor from WinRT)")};
        auto q = std::make_shared<queue>(); bridge::get().add_sensor(k, q);
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
    // A power request belongs to the process (any thread may toggle it), unlike SetThreadExecutionState, which is
    // per thread and is dropped when that thread exits.
    status keep_awake(bool on) override {
        static const power_api api;
        std::lock_guard g(awake_m);
        if (!api.create || !api.set || !api.clear) {   // pre-Windows 7: the thread-scoped fallback
            return SetThreadExecutionState(on ? (ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED) : ES_CONTINUOUS) ? status{} : status{{}, last_error("SetThreadExecutionState")};
        }
        if (on == awake_on) return {};
        if (!awake) {
            static wchar_t why[] = L"forge keep_awake";
            reason_context rc{}; rc.version = 0 /*POWER_REQUEST_CONTEXT_VERSION*/; rc.flags = 0x1 /*POWER_REQUEST_CONTEXT_SIMPLE_STRING*/; rc.reason.simple = why;
            awake = api.create(&rc);
            if (awake == INVALID_HANDLE_VALUE || !awake) { awake = nullptr; return {{}, last_error("PowerCreateRequest")}; }
        }
        if (on) {
            if (!api.set(awake, k_display_required)) return {{}, last_error("PowerSetRequest")};
            if (!api.set(awake, k_system_required)) { auto e = last_error("PowerSetRequest"); api.clear(awake, k_display_required); return {{}, e}; }
        } else { api.clear(awake, k_display_required); api.clear(awake, k_system_required); }
        awake_on = on; return {};
    }
    status open_url(std::string_view u) override {
        if (auto c = check_url(u); !c) return c;
        std::string s(u);
        int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0); std::wstring w(std::size_t(n), L'\0'); MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
        auto r = reinterpret_cast<INT_PTR>(ShellExecuteW(nullptr, L"open", w.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
        return r > 32 ? status{} : status{{}, {int(r), "ShellExecuteW failed"}};
    }
    result<device_info> info() override {
        auto d = backend::info(); if (!d) return d;
        bool slate = GetSystemMetrics(0x2003 /*SM_CONVERTIBLESLATEMODE*/) == 0, tablet = GetSystemMetrics(SM_TABLETPC) != 0;
        d.value.form = slate && tablet ? form_factor::tablet : power_status().has_battery ? form_factor::laptop : form_factor::desktop;
        const wchar_t* bios = L"HARDWARE\\DESCRIPTION\\System\\BIOS";   // SMBIOS system information, as msinfo32 shows it
        d.value.manufacturer = reg_string(bios, L"SystemManufacturer"); d.value.model = reg_string(bios, L"SystemProductName");
        return d;
    }
    result<permission_state> permission_status(permission p) override { if (auto b = backend::permission_status(p)) return b; return {permission_state::not_required, {}}; }
    status request_permission(permission) override { return {}; }
};
} // namespace
std::shared_ptr<backend> make_windows() { return std::make_shared<windows_backend>(); }
} // namespace forge::os::mobile::detail
#endif
