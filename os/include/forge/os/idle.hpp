#pragma once
// forge::os user-session state: how long the user has been idle and whether the screen is locked. Useful for
// deferring heavy work until nobody is at the machine.
//   idle_seconds()   Windows: GetLastInputInfo.  macOS: CGEventSourceSecondsSinceLastEventType (CoreGraphics,
//                    loaded at run time).  Linux/FreeBSD under X11: the MIT-SCREEN-SAVER extension through
//                    libXss (dlopen).  Linux without X11 (Wayland, consoles): logind's IdleHint for the session
//                    (coarse: 0 until the desktop declares the session idle, then the time since it did).
//   screen_locked()  Windows: the session's WTS lock flag (wtsapi32), else whether the input desktop can be
//                    opened.  macOS: CGSSessionScreenIsLocked from CGSessionCopyCurrentDictionary.  Linux: logind
//                    LockedHint (`loginctl show-session`).  FreeBSD: unsupported (no session manager records it).
//   Android / iOS:   unsupported - an app is not told about other apps' input; use the lifecycle events of
//                    forge_mobile (background / foreground) instead.
// No graphical or login session (ssh, services, CI): unsupported.
#include <forge/os/detail.hpp>
#include <forge/os/process.hpp>

namespace forge::os {
namespace detail_idle {
struct logind_session { bool found = false, idle = false, locked = false; std::uint64_t idle_since_us = 0; };
// `loginctl show-session <id> -p IdleHint -p IdleSinceHintMonotonic -p LockedHint` output.
inline logind_session parse_loginctl(const std::string& text) {
    logind_session s; std::size_t i = 0;
    while (i < text.size()) {
        auto e = text.find('\n', i); std::string line = text.substr(i, e == std::string::npos ? std::string::npos : e - i); i = e == std::string::npos ? text.size() : e + 1;
        auto eq = line.find('='); if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq), v = detail::trim(line.substr(eq + 1));
        if (k == "IdleHint") { s.idle = v == "yes"; s.found = true; }
        else if (k == "LockedHint") { s.locked = v == "yes"; s.found = true; }
        else if (k == "IdleSinceHintMonotonic") s.idle_since_us = std::strtoull(v.c_str(), nullptr, 10);
    }
    return s;
}
#if defined(FORGE_OS_LINUX)
inline result<logind_session> logind() {
    if (find_executable("loginctl").empty()) return {{}, unsupported("session state (no loginctl)")};
    spawn_spec sp; sp.argv = {"loginctl", "show-session"};
    if (auto id = getenv("XDG_SESSION_ID"); !id.empty()) sp.argv.push_back(id);
    for (const char* p : {"IdleHint", "IdleSinceHintMonotonic", "LockedHint"}) { sp.argv.push_back("-p"); sp.argv.push_back(p); }
    sp.timeout_ms = 3000;
    auto r = spawn_wait(sp);
    if (!r || r->exit_code != 0) return {{}, unsupported("session state (no logind session)")};
    auto s = parse_loginctl(r->out); if (!s.found) return {{}, unsupported("session state (no logind session)")};
    return {s, {}};
}
#endif
#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
// Milliseconds since the last X11 input event; -1 when X11 or the screen-saver extension is not available.
inline long long x11_idle_ms() {
    if (!has_env("DISPLAY")) return -1;
    void* x = ::dlopen("libX11.so.6", RTLD_NOW | RTLD_LOCAL); void* ss = ::dlopen("libXss.so.1", RTLD_NOW | RTLD_LOCAL);
    long long ms = -1;
    if (x && ss) {
        struct info { unsigned long window; int state, kind; unsigned long til_or_since, idle, event_mask; };
        auto open = reinterpret_cast<void* (*)(const char*)>(::dlsym(x, "XOpenDisplay")); auto close = reinterpret_cast<int (*)(void*)>(::dlsym(x, "XCloseDisplay"));
        auto root = reinterpret_cast<unsigned long (*)(void*)>(::dlsym(x, "XDefaultRootWindow")); auto xfree = reinterpret_cast<int (*)(void*)>(::dlsym(x, "XFree"));
        auto alloc = reinterpret_cast<info* (*)()>(::dlsym(ss, "XScreenSaverAllocInfo"));
        auto query = reinterpret_cast<int (*)(void*, unsigned long, info*)>(::dlsym(ss, "XScreenSaverQueryInfo"));
        auto ext = reinterpret_cast<int (*)(void*, int*, int*)>(::dlsym(ss, "XScreenSaverQueryExtension"));
        if (open && close && root && xfree && alloc && query && ext) if (void* dpy = open(nullptr)) {
            int ev = 0, er = 0;
            if (ext(dpy, &ev, &er)) if (info* i = alloc()) { if (query(dpy, root(dpy), i)) ms = static_cast<long long>(i->idle); xfree(i); }
            close(dpy);
        }
    }
    if (ss) ::dlclose(ss);
    if (x) ::dlclose(x);
    return ms;
}
#endif
#if defined(FORGE_OS_MACOS)
inline void* coregraphics() { static void* cg = ::dlopen("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics", RTLD_LAZY); return cg; }
#endif
}

// Seconds since the user last touched keyboard or pointer.
inline result<double> idle_seconds() {
#if defined(FORGE_OS_WINDOWS)
    LASTINPUTINFO li{}; li.cbSize = sizeof li;
    if (!GetLastInputInfo(&li)) return {0, unsupported("idle time (no interactive session)")};
    return {double(GetTickCount() - li.dwTime) / 1000.0, {}};   // both are 32-bit tick counts: the difference survives wrap-around
#elif defined(FORGE_OS_MACOS)
    void* cg = detail_idle::coregraphics();
    auto f = cg ? reinterpret_cast<double (*)(int, std::uint32_t)>(::dlsym(cg, "CGEventSourceSecondsSinceLastEventType")) : nullptr;
    auto sess = cg ? reinterpret_cast<detail::CFDictionaryRef (*)()>(::dlsym(cg, "CGSessionCopyCurrentDictionary")) : nullptr;
    if (!f) return {0, unsupported("idle time (CoreGraphics not loadable)")};
    if (sess) { auto d = sess(); if (!d) return {0, unsupported("idle time (no window server session)")}; detail::cf_api::get().CFRelease(d); }
    double s = f(0 /*kCGEventSourceStateCombinedSessionState*/, ~std::uint32_t(0) /*kCGAnyInputEventType*/);
    return {s < 0 ? 0 : s, {}};
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    if (long long ms = detail_idle::x11_idle_ms(); ms >= 0) return {double(ms) / 1000.0, {}};
#if defined(FORGE_OS_LINUX)
    auto s = detail_idle::logind(); if (!s) return {0, s.err};
    if (!s->idle || !s->idle_since_us) return {0.0, {}};
    timespec ts{}; ::clock_gettime(CLOCK_MONOTONIC, &ts);
    std::uint64_t now_us = std::uint64_t(ts.tv_sec) * 1000000ull + std::uint64_t(ts.tv_nsec) / 1000ull;
    return {now_us > s->idle_since_us ? double(now_us - s->idle_since_us) / 1e6 : 0.0, {}};
#else
    return {0, unsupported("idle time (no X11 display or no libXss)")};
#endif
#else
    return {0, unsupported("idle time")};
#endif
}

inline result<bool> screen_locked() {
#if defined(FORGE_OS_WINDOWS)
    static HMODULE wts = LoadLibraryExW(L"wtsapi32.dll", nullptr, 0x800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/);
    using q_fn = BOOL(WINAPI*)(HANDLE, DWORD, int, wchar_t**, DWORD*); using f_fn = void(WINAPI*)(void*);
    auto q = wts ? reinterpret_cast<q_fn>(reinterpret_cast<void*>(GetProcAddress(wts, "WTSQuerySessionInformationW"))) : nullptr;
    auto fr = wts ? reinterpret_cast<f_fn>(reinterpret_cast<void*>(GetProcAddress(wts, "WTSFreeMemory"))) : nullptr;
    if (q && fr) {
        // WTSINFOEXW: DWORD Level; then WTSINFOEX_LEVEL1_W { ULONG SessionId; WTS_CONNECTSTATE_CLASS SessionState; LONG SessionFlags; ... }
        wchar_t* buf = nullptr; DWORD n = 0;
        if (q(nullptr /*WTS_CURRENT_SERVER_HANDLE*/, 0xFFFFFFFFu /*WTS_CURRENT_SESSION*/, 25 /*WTSSessionInfoEx*/, &buf, &n) && buf) {
            long level = 0, flags = -1;
            // The level-1 data holds LARGE_INTEGERs, so the union is 8-aligned: SessionId at 8, SessionState at 12, SessionFlags at 16.
            if (n >= 20) { std::memcpy(&level, buf, 4); std::memcpy(&flags, reinterpret_cast<const char*>(buf) + 16, 4); }
            fr(buf);
            if (level == 1 && (flags == 0 || flags == 1)) return {flags == 0 /*WTS_SESSIONSTATE_LOCK*/, {}};
        }
    }
    HDESK d = OpenInputDesktop(0, FALSE, 0x0100 /*DESKTOP_SWITCHDESKTOP*/);
    if (!d) return {false, unsupported("screen lock state (no interactive desktop)")};
    BOOL can = SwitchDesktop(d); CloseDesktop(d);   // switching to the desktop that already has input is a no-op, and fails while locked
    return {!can, {}};
#elif defined(FORGE_OS_MACOS)
    void* cg = detail_idle::coregraphics(); auto& cf = detail::cf_api::get();
    auto sess = cg ? reinterpret_cast<detail::CFDictionaryRef (*)()>(::dlsym(cg, "CGSessionCopyCurrentDictionary")) : nullptr;
    if (!sess || !cf.ok) return {false, unsupported("screen lock state (CoreGraphics not loadable)")};
    auto d = sess(); if (!d) return {false, unsupported("screen lock state (no window server session)")};
    auto k = cf.str("CGSSessionScreenIsLocked"); auto v = cf.CFDictionaryGetValue(d, k);
    bool locked = v && (v == cf.kCFBooleanTrue || cf.to_ll(v, 0) == 1);
    cf.CFRelease(k); cf.CFRelease(d);
    return {locked, {}};
#elif defined(FORGE_OS_LINUX)
    auto s = detail_idle::logind(); if (!s) return {false, s.err};
    return {s->locked, {}};
#else
    return {false, unsupported("screen lock state")};
#endif
}
} // namespace forge::os
