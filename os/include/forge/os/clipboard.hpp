#pragma once
// forge::os::clipboard: get / set plain UTF-8 text. Windows uses the Win32 clipboard (CF_UNICODETEXT); macOS
// runs pbcopy / pbpaste; Linux/FreeBSD run wl-copy / wl-paste under Wayland, else xclip or xsel under X11.
// No display or helper → unsupported(). Mobile: unsupported. Guide: docs/os-wrapper.md.
#include <forge/os/process.hpp>

namespace forge::os::clipboard {
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS) && !defined(FORGE_OS_ANDROID)
// The helper command lines for reading and writing, or empty when none is usable.
inline std::pair<std::vector<std::string>, std::vector<std::string>> helpers_() {
#if defined(__APPLE__)
    return {{"pbpaste"}, {"pbcopy"}};
#else
    if (!getenv("WAYLAND_DISPLAY").empty() && !find_executable("wl-copy").empty()) return {{"wl-paste", "--no-newline"}, {"wl-copy"}};
    if (!getenv("DISPLAY").empty()) {
        if (!find_executable("xclip").empty()) return {{"xclip", "-selection", "clipboard", "-o"}, {"xclip", "-selection", "clipboard", "-i"}};
        if (!find_executable("xsel").empty()) return {{"xsel", "--clipboard", "--output"}, {"xsel", "--clipboard", "--input"}};
    }
    return {};
#endif
}
#endif
inline bool available() {
#if defined(FORGE_OS_WINDOWS)
    return true;
#elif defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS) && !defined(FORGE_OS_ANDROID)
    return !helpers_().first.empty();
#else
    return false;
#endif
}
inline result<std::string> get() {
#if defined(FORGE_OS_WINDOWS)
    if (!OpenClipboard(nullptr)) return {{}, last_error("OpenClipboard")};
    std::string out; HANDLE h = GetClipboardData(CF_UNICODETEXT);
    if (h) if (auto* w = static_cast<const wchar_t*>(GlobalLock(h))) {
        int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
        if (n > 1) { out.resize(std::size_t(n)); WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), n, nullptr, nullptr); out.pop_back(); }
        GlobalUnlock(h);
    }
    CloseClipboard(); return {out, {}};
#elif defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS) && !defined(FORGE_OS_ANDROID)
    auto h = helpers_(); if (h.first.empty()) return {{}, unsupported("clipboard (no display or clipboard helper)")};
    spawn_spec s; s.argv = h.first; s.timeout_ms = 5000; auto r = spawn_wait(s);
    if (!r) return {{}, r.err}; if (r->exit_code) return {{}, {r->exit_code, "clipboard read: " + r->err}};
    return {r->out, {}};
#else
    return {{}, unsupported("clipboard")};
#endif
}
inline status set(const std::string& text) {
#if defined(FORGE_OS_WINDOWS)
    int n = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    HGLOBAL g = GlobalAlloc(GMEM_MOVEABLE, std::size_t(n) * sizeof(wchar_t)); if (!g) return {{}, last_error("GlobalAlloc")};
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, static_cast<wchar_t*>(GlobalLock(g)), n); GlobalUnlock(g);
    if (!OpenClipboard(nullptr)) { GlobalFree(g); return {{}, last_error("OpenClipboard")}; }
    EmptyClipboard();
    if (!SetClipboardData(CF_UNICODETEXT, g)) { auto e = last_error("SetClipboardData"); CloseClipboard(); GlobalFree(g); return {{}, e}; }
    CloseClipboard(); return {};
#elif defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS) && !defined(FORGE_OS_ANDROID)
    auto h = helpers_(); if (h.second.empty()) return {{}, unsupported("clipboard (no display or clipboard helper)")};
    spawn_spec s; s.argv = h.second; s.stdin_data = text; s.timeout_ms = 5000;
    s.capture_stdout = false; s.capture_stderr = false;   // xclip/wl-copy fork a server that keeps the clipboard alive
    auto r = spawn_wait(s); if (!r) return {{}, r.err}; if (r->exit_code) return {{}, {r->exit_code, "clipboard write failed"}};
    return {};
#else
    (void)text; return {{}, unsupported("clipboard")};
#endif
}
} // namespace forge::os::clipboard
