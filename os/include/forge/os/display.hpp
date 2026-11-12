#pragma once
// forge::os::displays(): the connected monitors with pixel size, physical size, DPI, scale and refresh rate,
// without a GUI toolkit and without creating a window.
//   Linux:    the kernel's DRM connectors (/sys/class/drm/card*-*: status, modes and the EDID block, which carries
//             the monitor name, physical size and timing) - works under X11, Wayland and on a bare console; when
//             that lists nothing (WSLg, remote X) and DISPLAY is set, the X11 screen through libX11 (dlopen).
//   FreeBSD:  the X11 screen through libX11 (dlopen) when DISPLAY is set; no display server: unsupported.
//   Windows:  EnumDisplayMonitors + GetMonitorInfoW + EnumDisplaySettingsW (pixels, position, refresh) and
//             GetDpiForMonitor (shcore, 8.1+) for the effective DPI; scale = dpi / 96. Values follow the process's
//             DPI awareness (a DPI-unaware process sees virtualised sizes and 96 DPI).
//   macOS:    CoreGraphics (CGGetActiveDisplayList, CGDisplayBounds, CGDisplayScreenSize, the current display
//             mode), loaded at run time; scale = backing pixels per point (2 on Retina).
//   Android / iOS: the screen belongs to the app's activity / scene: glue passes it to set_display_hook()
//             (forge_mobile's device::display() does this natively); unsupported without it.
// A machine with no monitor (servers, CI) gives an empty list, not an error.
#include <forge/os/detail.hpp>
#if defined(_MSC_VER)
#pragma comment(lib, "user32.lib")   // MSVC links only kernel32 on its own
#endif

namespace forge::os {
struct display_info {
    std::string name;            // connector or device name ("HDMI-A-1", "\\\\.\\DISPLAY1", "display 1")
    std::string model;           // monitor name from the EDID where available
    int x = 0, y = 0;            // position in the virtual desktop (0,0 where unknown)
    int width = 0, height = 0;   // pixels
    int width_mm = 0, height_mm = 0;   // physical size, 0 unknown
    double dpi = 0;              // pixels per inch: from the physical size, or the OS's effective DPI; 0 unknown
    double scale = 1.0;          // UI scale factor (1.0 where the OS does not say)
    double refresh_hz = 0;       // 0 unknown
    bool primary = false;
};
using display_hook = std::function<std::vector<display_info>()>;
namespace detail_display {
inline std::mutex& hook_mutex() { static std::mutex m; return m; }
inline display_hook& hook() { static display_hook h; return h; }
struct edid_info { std::string name, vendor; int width = 0, height = 0, width_mm = 0, height_mm = 0; double refresh_hz = 0; bool ok = false; };
// A 128-byte (or longer) EDID block: preferred timing, physical size, monitor name. Bounds-checked.
inline edid_info parse_edid(std::string_view e) {
    edid_info r; auto b = [&](std::size_t i) { return static_cast<unsigned>(static_cast<unsigned char>(e[i])); };
    static const unsigned char magic[8] = {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0};
    if (e.size() < 128 || std::memcmp(e.data(), magic, 8) != 0) return r;
    unsigned m = (b(8) << 8) | b(9); for (int sh : {10, 5, 0}) { unsigned c = (m >> sh) & 0x1f; r.vendor += c >= 1 && c <= 26 ? char('A' + c - 1) : '?'; }
    r.width_mm = int(b(21)) * 10; r.height_mm = int(b(22)) * 10;
    for (std::size_t d = 54; d + 18 <= 126; d += 18) {
        unsigned clk = b(d) | (b(d + 1) << 8);
        if (clk) {
            if (r.width) continue;   // the first detailed timing is the preferred mode
            unsigned ha = b(d + 2) | ((b(d + 4) & 0xf0) << 4), hb = b(d + 3) | ((b(d + 4) & 0x0f) << 8), va = b(d + 5) | ((b(d + 7) & 0xf0) << 4), vb = b(d + 6) | ((b(d + 7) & 0x0f) << 8);
            unsigned wmm = b(d + 12) | ((b(d + 14) & 0xf0) << 4), hmm = b(d + 13) | ((b(d + 14) & 0x0f) << 8);
            r.width = int(ha); r.height = int(va);
            if (wmm && hmm) { r.width_mm = int(wmm); r.height_mm = int(hmm); }
            if ((ha + hb) && (va + vb)) r.refresh_hz = double(clk) * 10000.0 / (double(ha + hb) * double(va + vb));
        } else if (b(d + 3) == 0xfc) {
            std::string n; for (std::size_t i = d + 5; i < d + 18; ++i) { char c = e[i]; if (c == '\n' || c == 0) break; n += (static_cast<unsigned char>(c) < 0x20 || static_cast<unsigned char>(c) > 0x7e) ? '?' : c; }
            while (!n.empty() && n.back() == ' ') n.pop_back();
            r.name = n;
        }
    }
    r.ok = true; return r;
}
inline void finish(display_info& d) {
    if (d.dpi <= 0 && d.width > 0 && d.width_mm > 0) d.dpi = double(d.width) * 25.4 / double(d.width_mm);
}
#if (defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD))
inline bool x11_screen(display_info& d) {
    if (!has_env("DISPLAY")) return false;
    void* x = ::dlopen("libX11.so.6", RTLD_NOW | RTLD_LOCAL); if (!x) x = ::dlopen("libX11.so", RTLD_NOW | RTLD_LOCAL);
    if (!x) return false;
    auto open = reinterpret_cast<void* (*)(const char*)>(::dlsym(x, "XOpenDisplay")); auto close = reinterpret_cast<int (*)(void*)>(::dlsym(x, "XCloseDisplay"));
    auto screen = reinterpret_cast<int (*)(void*)>(::dlsym(x, "XDefaultScreen"));
    using dim = int (*)(void*, int);
    auto w = reinterpret_cast<dim>(::dlsym(x, "XDisplayWidth")), h = reinterpret_cast<dim>(::dlsym(x, "XDisplayHeight")), wm = reinterpret_cast<dim>(::dlsym(x, "XDisplayWidthMM")), hm = reinterpret_cast<dim>(::dlsym(x, "XDisplayHeightMM"));
    bool ok = false;
    if (open && close && screen && w && h && wm && hm) if (void* dpy = open(nullptr)) {
        int s = screen(dpy); d.name = "X11 screen " + std::to_string(s); d.width = w(dpy, s); d.height = h(dpy, s); d.width_mm = wm(dpy, s); d.height_mm = hm(dpy, s); d.primary = true;
        close(dpy); ok = d.width > 0 && d.height > 0;
    }
    ::dlclose(x); return ok;
}
#endif
#if defined(FORGE_OS_WINDOWS)
inline BOOL CALLBACK monitor_cb(HMONITOR mon, HDC, LPRECT, LPARAM lp) {
    auto* out = reinterpret_cast<std::vector<display_info>*>(lp);
    MONITORINFOEXW mi{}; mi.cbSize = sizeof mi; if (!GetMonitorInfoW(mon, &mi)) return TRUE;
    display_info d; d.name = narrow(mi.szDevice); d.x = mi.rcMonitor.left; d.y = mi.rcMonitor.top;
    d.width = mi.rcMonitor.right - mi.rcMonitor.left; d.height = mi.rcMonitor.bottom - mi.rcMonitor.top; d.primary = (mi.dwFlags & MONITORINFOF_PRIMARY) != 0;
    DEVMODEW dm{}; dm.dmSize = sizeof dm;
    if (EnumDisplaySettingsW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm)) { if (dm.dmDisplayFrequency > 1) d.refresh_hz = double(dm.dmDisplayFrequency); }
    static HMODULE shcore = LoadLibraryExW(L"shcore.dll", nullptr, 0x800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/);
    using dpi_fn = HRESULT(WINAPI*)(HMONITOR, int, UINT*, UINT*);
    static dpi_fn get_dpi = shcore ? reinterpret_cast<dpi_fn>(reinterpret_cast<void*>(GetProcAddress(shcore, "GetDpiForMonitor"))) : nullptr;
    UINT dx = 0, dy = 0;
    if (get_dpi && SUCCEEDED(get_dpi(mon, 0 /*MDT_EFFECTIVE_DPI*/, &dx, &dy)) && dx) { d.dpi = double(dx); d.scale = double(dx) / 96.0; }
    out->push_back(std::move(d)); return TRUE;
}
#endif
}

inline void set_display_hook(display_hook h) { std::lock_guard<std::mutex> g(detail_display::hook_mutex()); detail_display::hook() = std::move(h); }

inline result<std::vector<display_info>> displays() {
    { display_hook h; { std::lock_guard<std::mutex> g(detail_display::hook_mutex()); h = detail_display::hook(); } if (h) return {h(), {}}; }
    std::vector<display_info> out;
#if defined(FORGE_OS_LINUX)
    if (auto l = list_dir("/sys/class/drm")) {
        std::vector<std::string> names; for (auto& e : l.value) if (e.name.rfind("card", 0) == 0 && e.name.find('-') != std::string::npos) names.push_back(e.name);
        std::sort(names.begin(), names.end());
        for (auto& n : names) {
            std::string d = "/sys/class/drm/" + n;
            if (detail::trim(detail::slurp(d + "/status")) != "connected") continue;
            display_info di; di.name = n.substr(n.find('-') + 1);
            auto ed = detail_display::parse_edid(detail::slurp(d + "/edid"));
            if (ed.ok) { di.model = ed.name; di.width = ed.width; di.height = ed.height; di.width_mm = ed.width_mm; di.height_mm = ed.height_mm; di.refresh_hz = ed.refresh_hz; }
            auto modes = detail::slurp(d + "/modes");   // first line = the preferred mode, "1920x1080"
            if (!di.width && !modes.empty()) { int w = 0, h = 0; if (std::sscanf(modes.c_str(), "%dx%d", &w, &h) == 2) { di.width = w; di.height = h; } }
            if (di.width <= 0) continue;
            di.primary = out.empty(); detail_display::finish(di); out.push_back(std::move(di));
        }
    }
    if (out.empty()) { display_info d; if (detail_display::x11_screen(d)) { detail_display::finish(d); out.push_back(std::move(d)); } }
    return {std::move(out), {}};
#elif defined(FORGE_OS_FREEBSD)
    if (!has_env("DISPLAY")) return {{}, unsupported("displays (no X11 display; FreeBSD has no display registry outside the display server)")};
    { display_info d; if (detail_display::x11_screen(d)) { detail_display::finish(d); out.push_back(std::move(d)); } }
    return {std::move(out), {}};
#elif defined(FORGE_OS_WINDOWS)
    EnumDisplayMonitors(nullptr, nullptr, detail_display::monitor_cb, reinterpret_cast<LPARAM>(&out));
    return {std::move(out), {}};
#elif defined(FORGE_OS_MACOS)
    static void* cg = ::dlopen("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics", RTLD_LAZY);
    if (!cg) return {{}, unsupported("displays (CoreGraphics not loadable)")};
    struct cgsize { double w, h; }; struct cgrect { double x, y, w, h; };
    auto list = reinterpret_cast<int (*)(std::uint32_t, std::uint32_t*, std::uint32_t*)>(::dlsym(cg, "CGGetActiveDisplayList"));
    auto bounds = reinterpret_cast<cgrect (*)(std::uint32_t)>(::dlsym(cg, "CGDisplayBounds"));
    auto size_mm = reinterpret_cast<cgsize (*)(std::uint32_t)>(::dlsym(cg, "CGDisplayScreenSize"));
    auto main_id = reinterpret_cast<std::uint32_t (*)()>(::dlsym(cg, "CGMainDisplayID"));
    auto copy_mode = reinterpret_cast<void* (*)(std::uint32_t)>(::dlsym(cg, "CGDisplayCopyDisplayMode"));
    auto mode_pw = reinterpret_cast<std::size_t (*)(void*)>(::dlsym(cg, "CGDisplayModeGetPixelWidth")), mode_ph = reinterpret_cast<std::size_t (*)(void*)>(::dlsym(cg, "CGDisplayModeGetPixelHeight"));
    auto mode_w = reinterpret_cast<std::size_t (*)(void*)>(::dlsym(cg, "CGDisplayModeGetWidth"));
    auto mode_hz = reinterpret_cast<double (*)(void*)>(::dlsym(cg, "CGDisplayModeGetRefreshRate")); auto mode_free = reinterpret_cast<void (*)(void*)>(::dlsym(cg, "CGDisplayModeRelease"));
    if (!list || !bounds || !size_mm || !main_id) return {{}, unsupported("displays (CoreGraphics display functions)")};
    std::uint32_t ids[32] = {}, n = 0;
    if (list(32, ids, &n) != 0) return {std::move(out), {}};   // no window server session (ssh, launchd daemon): no displays
    for (std::uint32_t i = 0; i < n && i < 32; ++i) {
        display_info d; d.name = "display " + std::to_string(ids[i]); d.primary = ids[i] == main_id();
        cgrect r = bounds(ids[i]); d.x = int(r.x); d.y = int(r.y); d.width = int(r.w); d.height = int(r.h);
        cgsize mm = size_mm(ids[i]); d.width_mm = int(mm.w + 0.5); d.height_mm = int(mm.h + 0.5);
        if (copy_mode && mode_pw && mode_ph && mode_w && mode_free) if (void* m = copy_mode(ids[i])) {
            std::size_t pw = mode_pw(m), ph = mode_ph(m), w = mode_w(m);
            if (pw && ph) { if (w) d.scale = double(pw) / double(w); d.width = int(pw); d.height = int(ph); }
            if (mode_hz) d.refresh_hz = mode_hz(m);
            mode_free(m);
        }
        detail_display::finish(d); out.push_back(std::move(d));
    }
    return {std::move(out), {}};
#else
    return {{}, unsupported("displays (pass the screen to set_display_hook from the app)")};
#endif
}
} // namespace forge::os
