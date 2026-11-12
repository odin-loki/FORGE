#pragma once
// forge::os system appearance: light / dark, high contrast, reduced motion, accent colour, and a watcher that
// reports changes. Read-only; nothing here opens a window. Guide: docs/os-wrapper.md.
//   Windows   registry: Themes\Personalize AppsUseLightTheme, DWM AccentColor; SystemParametersInfo for high
//             contrast (SPI_GETHIGHCONTRAST) and animations (SPI_GETCLIENTAREAANIMATION).
//   macOS     CFPreferences: AppleInterfaceStyle, AppleAccentColor, and com.apple.universalaccess reduceMotion /
//             increaseContrast.
//   iOS       UITraitCollection.currentTraitCollection.userInterfaceStyle, UIAccessibilityIsReduceMotionEnabled,
//             UIAccessibilityDarkerSystemColorsEnabled (UIKit loaded in the app).
//   Linux / FreeBSD   the desktop portal's Settings over D-Bus (org.freedesktop.appearance color-scheme / contrast /
//             accent-color, org.gnome.desktop.interface enable-animations); without a portal GNOME's dconf database
//             and KDE's kdeglobals are read directly.
//   Android   Configuration.uiMode is Java-only: the app installs set_appearance_hook().
//   No desktop session (servers, CI): unsupported.
#include <forge/os/detail.hpp>
#include <chrono>
#include <functional>
#include <forge/os/desktop_bus.hpp>
#include <forge/os/proxy.hpp>   // detail_net::gvdb_values (the dconf reader)
#if defined(_MSC_VER)
#pragma comment(lib, "user32.lib")   // MSVC links only kernel32 on its own
#pragma comment(lib, "advapi32.lib")
#endif

namespace forge::os {
enum class color_scheme { unknown, light, dark };
struct appearance {
    color_scheme scheme = color_scheme::unknown;
    bool high_contrast = false, reduced_motion = false;
    bool has_accent = false; std::uint8_t accent_r = 0, accent_g = 0, accent_b = 0;
    std::string source;   // registry, cfpreferences, uikit, portal, dconf, kdeglobals, hook
    bool operator==(const appearance& o) const {
        return scheme == o.scheme && high_contrast == o.high_contrast && reduced_motion == o.reduced_motion && has_accent == o.has_accent &&
               accent_r == o.accent_r && accent_g == o.accent_g && accent_b == o.accent_b;
    }
};
using appearance_hook = std::function<result<appearance>()>;
namespace detail_appearance {
inline std::mutex& hook_mutex() { static std::mutex m; return m; }
inline appearance_hook& hook() { static appearance_hook h; return h; }
// GNOME settings from a dconf database image.
inline appearance parse_dconf(const std::string& db) {
    appearance a; auto kv = net::detail_net::gvdb_values(db, "/org/gnome/desktop/");
    auto get = [&](const char* k) { auto it = kv.find(k); return it == kv.end() ? std::string() : it->second; };
    std::string cs = get("interface/color-scheme"), theme = net::detail_net::lower(get("interface/gtk-theme"));
    if (cs == "prefer-dark") a.scheme = color_scheme::dark; else if (cs == "prefer-light") a.scheme = color_scheme::light;
    else if (!theme.empty()) a.scheme = theme.find("dark") != std::string::npos ? color_scheme::dark : color_scheme::light;
    else if (cs == "default") a.scheme = color_scheme::light;
    a.reduced_motion = get("interface/enable-animations") == "false";
    a.high_contrast = get("a11y/interface/high-contrast") == "true" || theme.find("highcontrast") != std::string::npos;
    if (a.scheme != color_scheme::unknown || a.reduced_motion || a.high_contrast) a.source = "dconf";
    return a;
}
// KDE: kdeglobals ([General] ColorScheme, [KDE] AnimationDurationFactor).
inline appearance parse_kdeglobals(const std::string& text) {
    appearance a; std::string group; std::size_t i = 0;
    while (i < text.size()) {
        auto e = text.find('\n', i); std::string line = detail::trim(text.substr(i, e == std::string::npos ? std::string::npos : e - i)); i = e == std::string::npos ? text.size() : e + 1;
        if (line.empty() || line[0] == '#') continue;
        if (line[0] == '[') { group = line; continue; }
        auto eq = line.find('='); if (eq == std::string::npos) continue;
        std::string k = detail::trim(line.substr(0, eq)), v = detail::trim(line.substr(eq + 1));
        if (group == "[General]" && k == "ColorScheme" && !v.empty()) { a.scheme = net::detail_net::lower(v).find("dark") != std::string::npos ? color_scheme::dark : color_scheme::light; a.source = "kdeglobals"; }
        if (group == "[KDE]" && k == "AnimationDurationFactor") { a.reduced_motion = std::strtod(v.c_str(), nullptr) == 0.0; a.source = "kdeglobals"; }
        if (group == "[General]" && k == "AccentColor") {   // "r,g,b"
            int r = 0, g = 0, b = 0; if (std::sscanf(v.c_str(), "%d,%d,%d", &r, &g, &b) == 3) { a.has_accent = true; a.accent_r = std::uint8_t(r); a.accent_g = std::uint8_t(g); a.accent_b = std::uint8_t(b); }
        }
    }
    return a;
}
}
// Android / iOS glue or tests supply the appearance. Pass {} to remove it.
inline void set_appearance_hook(appearance_hook h) { std::lock_guard<std::mutex> g(detail_appearance::hook_mutex()); detail_appearance::hook() = std::move(h); }

inline result<appearance> system_appearance() {
    { appearance_hook h; { std::lock_guard<std::mutex> g(detail_appearance::hook_mutex()); h = detail_appearance::hook(); } if (h) { auto r = h(); if (r && r->source.empty()) r->source = "hook"; return r; } }
    appearance a;
#if defined(FORGE_OS_WINDOWS)
    auto dword = [](const wchar_t* key, const wchar_t* name, DWORD& out) { DWORD n = sizeof out; return RegGetValueW(HKEY_CURRENT_USER, key, name, RRF_RT_REG_DWORD, nullptr, &out, &n) == ERROR_SUCCESS; };
    DWORD v = 0; a.source = "registry";
    if (dword(L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize", L"AppsUseLightTheme", v)) a.scheme = v ? color_scheme::light : color_scheme::dark;
    else a.scheme = color_scheme::light;   // before Windows 10 1607 there is no dark mode
    if (dword(L"Software\\Microsoft\\Windows\\DWM", L"AccentColor", v)) { a.has_accent = true; a.accent_r = std::uint8_t(v); a.accent_g = std::uint8_t(v >> 8); a.accent_b = std::uint8_t(v >> 16); }   // 0xAABBGGRR
    HIGHCONTRASTW hc{}; hc.cbSize = sizeof hc;
    if (SystemParametersInfoW(SPI_GETHIGHCONTRAST, sizeof hc, &hc, 0)) a.high_contrast = (hc.dwFlags & HCF_HIGHCONTRASTON) != 0;
    BOOL anim = TRUE; if (SystemParametersInfoW(0x1042 /*SPI_GETCLIENTAREAANIMATION*/, 0, &anim, 0)) a.reduced_motion = !anim;
    return {a, {}};
#elif defined(FORGE_OS_IOS)
    auto& o = detail::objc_api::get(); void* tc = o.ok() && o.cls("UITraitCollection") ? o.send(o.cls("UITraitCollection"), "currentTraitCollection") : nullptr;
    if (!tc) return {a, unsupported("appearance (UIKit not available)")};
    long style = o.send<long>(tc, "userInterfaceStyle"); a.scheme = style == 2 ? color_scheme::dark : style == 1 ? color_scheme::light : color_scheme::unknown;
    if (auto f = reinterpret_cast<unsigned char (*)()>(::dlsym(RTLD_DEFAULT, "UIAccessibilityIsReduceMotionEnabled"))) a.reduced_motion = f() != 0;
    if (auto f = reinterpret_cast<unsigned char (*)()>(::dlsym(RTLD_DEFAULT, "UIAccessibilityDarkerSystemColorsEnabled"))) a.high_contrast = f() != 0;
    a.source = "uikit"; return {a, {}};
#elif defined(FORGE_OS_MACOS)
    auto& cf = detail::cf_api::get(); if (!cf.ok) return {a, unsupported("appearance (CoreFoundation not loadable)")};
    static auto copy = reinterpret_cast<const void* (*)(const void*, const void*)>(::dlsym(cf.h, "CFPreferencesCopyAppValue"));
    static auto cur = static_cast<const void**>(::dlsym(cf.h, "kCFPreferencesCurrentApplication"));
    if (!copy || !cur || !*cur) return {a, unsupported("appearance (CFPreferences not available)")};
    auto read = [&](const char* key, const void* app, std::string& text, long long& num) {
        auto k = cf.str(key); auto v = copy(k, app); cf.CFRelease(k); if (!v) return false;
        text = cf.to_string(v); num = cf.to_ll(v, 0); cf.CFRelease(v); return true;
    };
    std::string s; long long n = 0;
    a.scheme = read("AppleInterfaceStyle", *cur, s, n) && s == "Dark" ? color_scheme::dark : color_scheme::light;   // the key is absent in light mode
    if (read("AppleAccentColor", *cur, s, n)) {
        static const std::uint8_t tbl[8][3] = {{142, 142, 147}, {255, 59, 48}, {255, 149, 0}, {255, 204, 0}, {40, 205, 65}, {0, 122, 255}, {175, 82, 222}, {255, 45, 85}};   // -1 graphite .. 6 pink
        if (n >= -1 && n <= 6) { a.has_accent = true; a.accent_r = tbl[n + 1][0]; a.accent_g = tbl[n + 1][1]; a.accent_b = tbl[n + 1][2]; }
    } else { a.has_accent = true; a.accent_r = 0; a.accent_g = 122; a.accent_b = 255; }   // unset: the default blue
    auto ua = cf.str("com.apple.universalaccess");
    if (read("reduceMotion", ua, s, n)) a.reduced_motion = n != 0 || s == "true";
    if (read("increaseContrast", ua, s, n)) a.high_contrast = n != 0 || s == "true";
    cf.CFRelease(ua); a.source = "cfpreferences"; return {a, {}};
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    if (desktop_bus::has_bus(dbus::bus::session)) {
        if (auto cs = desktop_bus::portal_setting("org.freedesktop.appearance", "color-scheme")) {
            a.scheme = cs->u == 1 ? color_scheme::dark : cs->u == 2 ? color_scheme::light : color_scheme::unknown; a.source = "portal";
            if (auto c = desktop_bus::portal_setting("org.freedesktop.appearance", "contrast")) a.high_contrast = c->u == 1;
            if (auto ac = desktop_bus::portal_setting("org.freedesktop.appearance", "accent-color"); ac && ac->type == 'r' && ac->items.size() == 3) {
                auto ch = [](double d) { return std::uint8_t(d < 0 ? 0 : d > 1 ? 255 : d * 255.0 + 0.5); };
                if (ac->items[0].d >= 0 && ac->items[0].d <= 1) { a.has_accent = true; a.accent_r = ch(ac->items[0].d); a.accent_g = ch(ac->items[1].d); a.accent_b = ch(ac->items[2].d); }   // out of range: "no accent"
            }
            if (auto an = desktop_bus::portal_setting("org.gnome.desktop.interface", "enable-animations"); an && an->type == 'b') a.reduced_motion = an->u == 0;
            return {a, {}};
        }
    }
    {
        std::string cfg = getenv("XDG_CONFIG_HOME"); if (cfg.empty()) cfg = join(home_dir(), ".config");
        const bool kde_first = net::detail_net::lower(getenv("XDG_CURRENT_DESKTOP")).find("kde") != std::string::npos;
        auto kde = [&] { auto t = read_file(join(cfg, "kdeglobals")); return t ? detail_appearance::parse_kdeglobals(*t) : appearance{}; };
        auto gnome = [&] { auto t = read_file(join(join(cfg, "dconf"), "user")); return t ? detail_appearance::parse_dconf(*t) : appearance{}; };
        if (auto x = kde_first ? kde() : gnome(); !x.source.empty()) return {x, {}};
        if (auto y = kde_first ? gnome() : kde(); !y.source.empty()) return {y, {}};
    }
    return {a, unsupported("appearance (no desktop portal and no desktop settings file)")};
#else
    return {a, unsupported("appearance (install set_appearance_hook from the app)")};
#endif
}

// Calls back (on its own thread) whenever system_appearance() changes; polls every interval_ms. The first
// callback reports the state found at start(). stop() (or destruction) ends it promptly.
class appearance_watcher {
public:
    appearance_watcher() = default;
    appearance_watcher(const appearance_watcher&) = delete; appearance_watcher& operator=(const appearance_watcher&) = delete;
    ~appearance_watcher() { stop(); }
    status start(std::function<void(const appearance&)> cb, int interval_ms = 1000) {
        stop();
        auto first = system_appearance(); if (!first) return {{}, first.err};
        stop_ = false;
        th_ = std::thread([this, cb = std::move(cb), last = first.value, interval_ms]() mutable {
            cb(last);
            std::unique_lock<std::mutex> lk(m_);
            while (!stop_) {
                cv_.wait_for(lk, std::chrono::milliseconds(interval_ms < 10 ? 10 : interval_ms), [&] { return stop_; });
                if (stop_) break;
                lk.unlock(); auto now = system_appearance(); if (now && !(now.value == last)) { last = now.value; cb(last); } lk.lock();
            }
        });
        return {};
    }
    void stop() { { std::lock_guard<std::mutex> g(m_); stop_ = true; } cv_.notify_all(); if (th_.joinable()) th_.join(); }
    bool running() const { return th_.joinable(); }
private:
    std::mutex m_; std::condition_variable cv_; bool stop_ = false; std::thread th_;
};
} // namespace forge::os
