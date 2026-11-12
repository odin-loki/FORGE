#pragma once
// forge::os: hand a URL or a file to the user's default application, or show a file in the file manager.
//   open_url(url)      Linux/FreeBSD: xdg-open (else `gio open`), only inside a desktop session (DISPLAY or
//                      WAYLAND_DISPLAY).  macOS: /usr/bin/open.  Windows: ShellExecuteW "open".  iOS: UIApplication
//                      openURL:options:completionHandler: on the main queue.  Android: an Intent can only be sent
//                      from Java, so app glue installs set_open_hook(); without it unsupported.
//   open_path(path)    the same openers on an existing file or directory (never on something that looks like an
//                      option or a URL).
//   reveal_path(path)  Windows: explorer /select.  macOS: open -R.  Linux/FreeBSD: the org.freedesktop.FileManager1
//                      ShowItems D-Bus call through gdbus, else the parent directory is opened.
// open_url checks the URL first (check_url): absolute, an RFC 3986 scheme at position 0 that is on the allow-list
// (default http, https, mailto), no spaces or control characters, so neither a local program nor an option can be
// smuggled to the opener. A hook installed with set_open_hook() takes over on every OS (mobile glue, tests).
// Nothing here blocks on the opened application: helpers are waited for at most 10 s (they return at once).
#include <forge/os/detail.hpp>
#include <forge/os/process.hpp>
#include <forge/os/desktop_bus.hpp>

namespace forge::os {
enum class open_kind { url, path, reveal };
using open_hook = std::function<status(const std::string& target, open_kind kind)>;
namespace detail_open {
inline std::mutex& hook_mutex() { static std::mutex m; return m; }
inline open_hook& hook() { static open_hook h; return h; }
inline open_hook get_hook() { std::lock_guard<std::mutex> g(hook_mutex()); return hook(); }
inline bool desktop_session() {
#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    return has_env("DISPLAY") || has_env("WAYLAND_DISPLAY");
#else
    return true;
#endif
}
inline std::string file_uri(const std::string& abs) {
    std::string o = "file://"; char b[4];
    for (unsigned char c : abs) {
        bool keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '/' || c == '-' || c == '_' || c == '.' || c == '~';
        if (keep) o += char(c); else { std::snprintf(b, sizeof b, "%%%02X", unsigned(c)); o += b; }
    }
    return o;
}
inline status run_helper(std::vector<std::string> argv, const char* what) {
    spawn_spec sp; sp.argv = std::move(argv); sp.timeout_ms = 10000; sp.new_session = true;
    auto r = spawn_wait(sp); if (!r) return {{}, r.err};
    if (r->exit_code != 0) return {{}, error{r->exit_code, std::string(what) + ": " + sp.argv[0] + " exited with " + std::to_string(r->exit_code)}};
    return {};
}
#if defined(FORGE_OS_WINDOWS)
inline status shell_execute(const std::wstring& file, const std::wstring& params) {
    static HMODULE sh = LoadLibraryExW(L"shell32.dll", nullptr, 0x800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/);
    using fn = HINSTANCE(WINAPI*)(HWND, const wchar_t*, const wchar_t*, const wchar_t*, const wchar_t*, int);
    auto f = sh ? reinterpret_cast<fn>(reinterpret_cast<void*>(GetProcAddress(sh, "ShellExecuteW"))) : nullptr;
    if (!f) return {{}, unsupported("ShellExecuteW")};
    auto r = reinterpret_cast<INT_PTR>(f(nullptr, L"open", file.c_str(), params.empty() ? nullptr : params.c_str(), nullptr, 1 /*SW_SHOWNORMAL*/));
    if (r <= 32) return {{}, error{int(r), "ShellExecuteW failed (" + std::to_string(int(r)) + ")"}};
    return {};
}
#endif
#if defined(FORGE_OS_IOS)
inline status ios_open(const std::string& url) {
    auto& o = detail::objc_api::get();
    static auto async_f = reinterpret_cast<void (*)(void*, void*, void (*)(void*))>(::dlsym(RTLD_DEFAULT, "dispatch_async_f"));
    static void* main_q = ::dlsym(RTLD_DEFAULT, "_dispatch_main_q");
    if (!o.ok() || !o.cls("UIApplication") || !async_f || !main_q) return {{}, unsupported("open_url (UIApplication not available)")};
    async_f(main_q, new std::string(url), [](void* c) {
        std::unique_ptr<std::string> u(static_cast<std::string*>(c)); auto& oo = detail::objc_api::get();
        void* app = oo.send(oo.cls("UIApplication"), "sharedApplication"); if (!app) return;
        void* s = oo.send<void*, const char*>(oo.cls("NSString"), "stringWithUTF8String:", u->c_str());
        void* nsurl = s ? oo.send<void*, void*>(oo.cls("NSURL"), "URLWithString:", s) : nullptr; if (!nsurl) return;
        void* opts = oo.send(oo.cls("NSDictionary"), "dictionary");
        oo.send<void, void*, void*, void*>(app, "openURL:options:completionHandler:", nsurl, opts, nullptr);
    });
    return {};
}
#endif
}

// Takes over open_url / open_path / reveal_path (after validation). Pass {} to remove it.
inline void set_open_hook(open_hook h) { std::lock_guard<std::mutex> g(detail_open::hook_mutex()); detail_open::hook() = std::move(h); }

inline status check_url(std::string_view u, const std::vector<std::string>& schemes = {"http", "https", "mailto"}) {
    if (u.empty() || u.size() > 8192) return {{}, errno_error(EINVAL, "open_url: empty or too long")};
    for (unsigned char c : u) if (c <= 0x20 || c == 0x7f) return {{}, errno_error(EINVAL, "open_url: control character or space in URL")};
    auto alpha = [](unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
    if (!alpha(static_cast<unsigned char>(u[0]))) return {{}, errno_error(EINVAL, "open_url: not an absolute URL")};
    std::size_t i = 1; while (i < u.size() && (alpha(static_cast<unsigned char>(u[i])) || (u[i] >= '0' && u[i] <= '9') || u[i] == '+' || u[i] == '-' || u[i] == '.')) ++i;
    if (i >= u.size() || u[i] != ':' || i + 1 == u.size()) return {{}, errno_error(EINVAL, "open_url: not an absolute URL")};
    std::string scheme(u.substr(0, i)); for (auto& c : scheme) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    bool allowed = false; for (auto& s : schemes) allowed |= s == scheme;
    if (!allowed) return {{}, errno_error(EPERM, "open_url: scheme '" + scheme + "' is not allowed")};
    if ((scheme == "http" || scheme == "https") && (u.substr(i + 1, 2) != "//" || u.size() <= i + 3 || u[i + 3] == '/')) return {{}, errno_error(EINVAL, "open_url: " + scheme + " URL without a host")};
    return {};
}
// The helper program open_url / open_path would run for `target` on this OS; empty when the OS API is called
// directly (Windows, iOS) or no opener exists (headless Linux, Android).
inline std::vector<std::string> opener_command(const std::string& target) {
#if defined(FORGE_OS_MACOS)
    return {"/usr/bin/open", target};
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    if (!detail_open::desktop_session()) return {};
    if (!find_executable("xdg-open").empty()) return {"xdg-open", target};
    if (!find_executable("gio").empty()) return {"gio", "open", target};
    return {};
#else
    (void)target; return {};
#endif
}
// True when open_url can reach a default application (a hook, a desktop session with an opener, or an OS API).
inline bool can_open() {
    if (detail_open::get_hook()) return true;
#if defined(FORGE_OS_WINDOWS) || defined(FORGE_OS_MACOS)
    return true;
#elif defined(FORGE_OS_IOS)
    return detail::objc_api::get().ok() && detail::objc_api::get().cls("UIApplication") != nullptr;
#elif defined(FORGE_OS_ANDROID)
    return false;
#else
    return !opener_command("x").empty();
#endif
}
namespace detail_open {
inline status open_target(const std::string& target, open_kind kind) {
    if (auto h = get_hook()) return h(target, kind);
#if defined(FORGE_OS_WINDOWS)
    return shell_execute(widen(target), {});
#elif defined(FORGE_OS_IOS)
    if (kind != open_kind::url) return {{}, unsupported("open_path (iOS apps open documents through UIDocumentInteractionController)")};
    return ios_open(target);
#elif defined(FORGE_OS_ANDROID)
    (void)kind; return {{}, unsupported("open (install set_open_hook from JNI: startActivity(Intent.ACTION_VIEW))")};
#else
    auto cmd = opener_command(target);
#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    // The desktop portal over D-Bus: the only way out of a Flatpak / Snap sandbox, and the fallback when no
    // opener program is installed. It takes URIs (files need a descriptor: those keep using the helper).
    const bool sandboxed = stat("/.flatpak-info").exists || has_env("SNAP");
    if (kind == open_kind::url && (cmd.empty() || sandboxed) && desktop_session()) {
        auto r = desktop_bus::open_uri(target);
        if (r || cmd.empty()) { if (!r && is_unsupported(r.err)) return {{}, unsupported("open (no xdg-open / gio and no desktop portal)")}; return r; }
    }
#endif
    (void)kind;
    if (cmd.empty()) return {{}, unsupported("open (no desktop session or no xdg-open / gio)")};
    return run_helper(std::move(cmd), "open");
#endif
}
}
inline status open_url(const std::string& url, const std::vector<std::string>& schemes = {"http", "https", "mailto"}) {
    if (auto c = check_url(url, schemes); !c) return c;
    return detail_open::open_target(url, open_kind::url);
}
// Opens an existing file or directory with its default application.
inline status open_path(const std::string& path) {
    if (path.empty()) return {{}, errno_error(EINVAL, "open_path: empty path")};
    std::string abs = absolute(path);   // absolute: never mistaken for an option or a URL by the helper
    if (!stat(abs).exists) return {{}, errno_error(ENOENT, "open_path: " + path + ": no such file or directory")};
    return detail_open::open_target(abs, open_kind::path);
}
// Shows the file selected in the desktop's file manager.
inline status reveal_path(const std::string& path) {
    if (path.empty()) return {{}, errno_error(EINVAL, "reveal_path: empty path")};
    std::string abs = absolute(path);
    if (!stat(abs).exists) return {{}, errno_error(ENOENT, "reveal_path: " + path + ": no such file or directory")};
    if (auto h = detail_open::get_hook()) return h(abs, open_kind::reveal);
#if defined(FORGE_OS_WINDOWS)
    std::wstring w = widen(abs); for (auto& c : w) if (c == L'/') c = L'\\';
    if (w.find(L'"') != std::wstring::npos) return {{}, errno_error(EINVAL, "reveal_path: quote in path")};
    return detail_open::shell_execute(L"explorer.exe", L"/select,\"" + w + L"\"");
#elif defined(FORGE_OS_MACOS)
    return detail_open::run_helper({"/usr/bin/open", "-R", abs}, "reveal_path");
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    if (!detail_open::desktop_session()) return {{}, unsupported("reveal_path (no desktop session)")};
    if (desktop_bus::show_items({detail_open::file_uri(abs)})) return {};   // org.freedesktop.FileManager1, spoken directly
    if (has_env("DBUS_SESSION_BUS_ADDRESS") && !find_executable("gdbus").empty()) {
        std::string uri = detail_open::file_uri(abs);   // percent-encoded: no quote can end the GVariant string
        auto r = detail_open::run_helper({"gdbus", "call", "--session", "--dest", "org.freedesktop.FileManager1", "--object-path", "/org/freedesktop/FileManager1",
                                          "--method", "org.freedesktop.FileManager1.ShowItems", "['" + uri + "']", ""}, "reveal_path");
        if (r) return r;
    }
    return detail_open::open_target(dirname(abs), open_kind::path);
#else
    return {{}, unsupported("reveal_path")};
#endif
}
} // namespace forge::os
