#pragma once
// forge::os::notify(): a desktop notification (title + body) from a program without a GUI toolkit.
//   Linux/FreeBSD: org.freedesktop.Notifications on the session bus, spoken by `notify-send` (libnotify) or, when
//                  that is missing, `gdbus call` (GLib). No session bus (ssh, containers, CI): unsupported.
//   macOS:         `osascript -e 'display notification ...'` (works for unbundled command-line programs; a bundled
//                  app that wants its own icon uses UNUserNotificationCenter, see forge_mobile).
//   Windows:       a balloon / toast from a notification-area icon (Shell_NotifyIconW, shell32 loaded at run time)
//                  owned by a hidden message-only window on its own thread; the icon is removed at exit.
//   Android/iOS:   notifications need Java / an app bundle: glue installs set_notification_hook(); unsupported
//                  without it. (forge_mobile's device::notify covers UNUserNotificationCenter on iOS.)
// A hook installed with set_notification_hook() takes over on every OS.
#include <forge/os/detail.hpp>
#include <forge/os/process.hpp>
#if defined(FORGE_OS_WINDOWS)
#include <shellapi.h>   // NOTIFYICONDATAW (types only)
#endif

namespace forge::os {
enum class notification_urgency { low, normal, critical };
struct notification {
    std::string title, body;
    std::string app_name = "forge";
    notification_urgency urgency = notification_urgency::normal;
    int timeout_ms = -1;   // -1: the notification server's default; 0: until dismissed (Linux/FreeBSD only)
};
using notification_hook = std::function<status(const notification&)>;
namespace detail_notify {
inline std::mutex& hook_mutex() { static std::mutex m; return m; }
inline notification_hook& hook() { static notification_hook h; return h; }
inline notification_hook get_hook() { std::lock_guard<std::mutex> g(hook_mutex()); return hook(); }
inline std::string no_ctl(std::string s) { for (auto& c : s) if (static_cast<unsigned char>(c) < 0x20 && c != '\n') c = ' '; return s; }
// GVariant text string literal.
inline std::string gvariant_string(const std::string& s) {
    std::string o = "'";
    for (char c : s) { if (c == '\'' || c == '\\') { o += '\\'; o += c; } else if (c == '\n') o += "\\n"; else o += c; }
    return o + "'";
}
// AppleScript string literal.
inline std::string applescript_string(const std::string& s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += static_cast<unsigned char>(c) < 0x20 ? ' ' : c; }
    return o + "\"";
}
inline bool session_bus() {
#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    if (has_env("DBUS_SESSION_BUS_ADDRESS")) return true;
    auto rt = getenv("XDG_RUNTIME_DIR"); return !rt.empty() && stat(rt + "/bus").exists;
#else
    return false;
#endif
}
#if defined(FORGE_OS_WINDOWS)
// The notification-area icon and the thread that owns its window. One per process, created on first use.
struct tray {
    std::mutex m; std::condition_variable cv; std::thread t; HWND hwnd = nullptr; DWORD tid = 0; bool ready = false, added = false;
    using notify_fn = BOOL(WINAPI*)(DWORD, NOTIFYICONDATAW*);
    static notify_fn shell_notify() {
        static HMODULE sh = LoadLibraryExW(L"shell32.dll", nullptr, 0x800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/);
        static notify_fn f = sh ? reinterpret_cast<notify_fn>(reinterpret_cast<void*>(GetProcAddress(sh, "Shell_NotifyIconW"))) : nullptr;
        return f;
    }
    bool start() {
        std::unique_lock<std::mutex> g(m);
        if (ready) return hwnd != nullptr;
        t = std::thread([this] {
            HWND w = CreateWindowExW(0, L"STATIC", L"forge", 0, 0, 0, 0, 0, HWND_MESSAGE, nullptr, GetModuleHandleW(nullptr), nullptr);   // a predefined class: nothing to register
            MSG msg; PeekMessageW(&msg, nullptr, 0, 0, PM_NOREMOVE);   // gives the thread its queue before anyone posts WM_QUIT
            { std::lock_guard<std::mutex> g2(m); hwnd = w; tid = GetCurrentThreadId(); ready = true; } cv.notify_all();
            if (!w) return;
            while (GetMessageW(&msg, nullptr, 0, 0) > 0) { TranslateMessage(&msg); DispatchMessageW(&msg); }
            { std::lock_guard<std::mutex> g2(m); if (added) { NOTIFYICONDATAW nid{}; nid.cbSize = sizeof nid; nid.hWnd = w; nid.uID = 1; if (auto f = shell_notify()) f(NIM_DELETE, &nid); added = false; } }
            DestroyWindow(w);
        });
        cv.wait(g, [&] { return ready; });
        return hwnd != nullptr;
    }
    status show(const notification& n) {
        auto f = shell_notify(); if (!f) return {{}, unsupported("notifications (shell32 Shell_NotifyIconW)")};
        if (!start()) return {{}, last_error("CreateWindowExW (notification window)")};
        std::lock_guard<std::mutex> g(m);
        NOTIFYICONDATAW nid{}; nid.cbSize = sizeof nid; nid.hWnd = hwnd; nid.uID = 1;
        nid.uFlags = NIF_ICON | NIF_TIP | NIF_INFO;
        nid.hIcon = LoadIconW(nullptr, MAKEINTRESOURCEW(n.urgency == notification_urgency::critical ? 32515 /*IDI_WARNING*/ : 32516 /*IDI_INFORMATION*/));
        auto copy = [](wchar_t* dst, std::size_t cap, const std::string& s) { auto w = widen(s); std::size_t k = std::min(w.size(), cap - 1); std::memcpy(dst, w.data(), k * sizeof(wchar_t)); dst[k] = 0; };
        copy(nid.szTip, sizeof nid.szTip / sizeof(wchar_t), n.app_name);
        copy(nid.szInfoTitle, sizeof nid.szInfoTitle / sizeof(wchar_t), no_ctl(n.title)); copy(nid.szInfo, sizeof nid.szInfo / sizeof(wchar_t), n.body.empty() ? std::string(" ") : no_ctl(n.body));
        nid.dwInfoFlags = n.urgency == notification_urgency::critical ? NIIF_WARNING : NIIF_INFO;
        if (!added) {
            if (!f(NIM_ADD, &nid)) return {{}, error{-1, "notifications: Shell_NotifyIconW(NIM_ADD) failed (no notification area in this session)"}};
            added = true; NOTIFYICONDATAW v = nid; v.uVersion = 4 /*NOTIFYICON_VERSION_4*/; f(NIM_SETVERSION, &v);
            return {};
        }
        if (!f(NIM_MODIFY, &nid)) return {{}, error{-1, "notifications: Shell_NotifyIconW(NIM_MODIFY) failed"}};
        return {};
    }
    ~tray() {
        DWORD id = 0; { std::lock_guard<std::mutex> g(m); id = tid; }
        if (t.joinable()) { if (id) PostThreadMessageW(id, WM_QUIT, 0, 0); t.join(); }
    }
    static tray& instance() { static tray x; return x; }
};
#endif
}

// Takes over notify() on every OS (Android / iOS glue, tests). Pass {} to remove it.
inline void set_notification_hook(notification_hook h) { std::lock_guard<std::mutex> g(detail_notify::hook_mutex()); detail_notify::hook() = std::move(h); }

// The helper command notify() would run on this OS; empty on Windows (API call), mobile, or without a session bus.
inline std::vector<std::string> notification_command(const notification& n) {
#if defined(FORGE_OS_MACOS)
    return {"/usr/bin/osascript", "-e", "display notification " + detail_notify::applescript_string(n.body) + " with title " + detail_notify::applescript_string(n.title)};
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    if (!detail_notify::session_bus()) return {};
    const char* urg = n.urgency == notification_urgency::low ? "low" : n.urgency == notification_urgency::critical ? "critical" : "normal";
    if (!find_executable("notify-send").empty()) {
        std::vector<std::string> a = {"notify-send", "--app-name=" + detail_notify::no_ctl(n.app_name), std::string("--urgency=") + urg};
        if (n.timeout_ms >= 0) a.push_back("--expire-time=" + std::to_string(n.timeout_ms));
        a.push_back("--"); a.push_back(detail_notify::no_ctl(n.title)); a.push_back(detail_notify::no_ctl(n.body));
        return a;
    }
    if (!find_executable("gdbus").empty()) {
        int u = n.urgency == notification_urgency::low ? 0 : n.urgency == notification_urgency::critical ? 2 : 1;
        return {"gdbus", "call", "--session", "--dest", "org.freedesktop.Notifications", "--object-path", "/org/freedesktop/Notifications",
                "--method", "org.freedesktop.Notifications.Notify", detail_notify::gvariant_string(detail_notify::no_ctl(n.app_name)), "uint32 0", "''",
                detail_notify::gvariant_string(detail_notify::no_ctl(n.title)), detail_notify::gvariant_string(detail_notify::no_ctl(n.body)), "@as []",
                "{'urgency': <byte " + std::to_string(u) + ">}", "int32 " + std::to_string(n.timeout_ms)};
    }
    return {};
#else
    (void)n; return {};
#endif
}
inline bool notifications_available() {
    if (detail_notify::get_hook()) return true;
#if defined(FORGE_OS_WINDOWS)
    return detail_notify::tray::shell_notify() != nullptr;
#elif defined(FORGE_OS_MACOS)
    return stat("/usr/bin/osascript").exists;
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    return !notification_command({}).empty();
#else
    return false;
#endif
}
inline status notify(const notification& n) {
    if (n.title.empty() && n.body.empty()) return {{}, errno_error(EINVAL, "notify: empty notification")};
    if (n.title.size() > 1024 || n.body.size() > 8192) return {{}, errno_error(EINVAL, "notify: title or body too long")};
    if (auto h = detail_notify::get_hook()) return h(n);
#if defined(FORGE_OS_WINDOWS)
    return detail_notify::tray::instance().show(n);
#elif defined(FORGE_OS_MACOS) || defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    auto cmd = notification_command(n);
    if (cmd.empty()) return {{}, unsupported("notifications (no session bus, or neither notify-send nor gdbus)")};
    spawn_spec sp; sp.argv = std::move(cmd); sp.timeout_ms = 10000;
    auto r = spawn_wait(sp); if (!r) return {{}, r.err};
    if (r->exit_code != 0) return {{}, error{r->exit_code, "notify: " + sp.argv[0] + " failed: " + detail::trim(r->err)}};
    return {};
#else
    return {{}, unsupported("notifications (install set_notification_hook from the app)")};
#endif
}
inline status notify(const std::string& title, const std::string& body) { notification n; n.title = title; n.body = body; return notify(n); }
} // namespace forge::os
