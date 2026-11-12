#pragma once
// forge::os::autostart: start a program when the user logs in (a desktop session concept; for a supervised
// background service use service_control.hpp).
//   Linux/FreeBSD: an XDG autostart entry $XDG_CONFIG_HOME/autostart/<name>.desktop (Desktop Entry spec quoting).
//   macOS:         a LaunchAgent ~/Library/LaunchAgents/<name>.plist with RunAtLoad (launchd loads it at the next
//                  login; it is not bootstrapped into the running session).
//   Windows:       a value <name> under HKCU\Software\Microsoft\Windows\CurrentVersion\Run (MSVCRT-quoted command line).
//   Android/iOS:   unsupported (apps cannot register themselves to start at boot from native code).
// The text is produced by render_*() so it can be inspected and tested on any OS. entry::dir redirects the write:
// a directory for the file on POSIX, an HKCU subkey on Windows (used by the tests).
#include <forge/os/os.hpp>

namespace forge::os::autostart {
struct entry {
    std::string name;                 // 1-64 characters of [A-Za-z0-9._-]; macOS convention: reverse DNS ("com.example.app")
    std::vector<std::string> argv;    // argv[0] should be an absolute path
    std::string description;          // shown by desktop session managers
    std::string dir;                  // "" = the system location
};
inline bool valid_name(const std::string& n) {
    if (n.empty() || n.size() > 64 || n[0] == '.' || n[0] == '-') return false;
    for (unsigned char c : n) if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-')) return false;
    return true;
}
inline status validate(const entry& e) {
    if (!valid_name(e.name)) return {{}, errno_error(EINVAL, "autostart: invalid name '" + e.name + "'")};
    if (e.argv.empty() || e.argv[0].empty()) return {{}, errno_error(EINVAL, "autostart: empty command")};
    auto ctl = [](const std::string& s) { for (unsigned char c : s) if (c < 0x20 || c == 0x7f) return true; return false; };
    if (ctl(e.description)) return {{}, errno_error(EINVAL, "autostart: control character in description")};
    for (auto& a : e.argv) if (ctl(a)) return {{}, errno_error(EINVAL, "autostart: control character in command")};
    return {};
}
namespace detail_autostart {
// One Exec= argument: quoted, with the quoting rule's escapes (" ` $ \), then the string-value escape (\ -> \\)
// and the field-code escape (% -> %%).
inline std::string desktop_exec_arg(const std::string& a) {
    std::string q = "\"";
    for (char c : a) { if (c == '"' || c == '`' || c == '$' || c == '\\') q += '\\'; q += c; }
    q += '"';
    std::string o;
    for (char c : q) { if (c == '\\') o += "\\\\"; else if (c == '%') o += "%%"; else o += c; }
    return o;
}
inline std::string desktop_string(const std::string& s) {   // string value: \ -> \\, control characters dropped
    std::string o; for (unsigned char c : s) { if (c == '\\') o += "\\\\"; else if (c >= 0x20 && c != 0x7f) o += char(c); } return o;
}
inline std::string xml_text(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) { case '&': o += "&amp;"; break; case '<': o += "&lt;"; break; case '>': o += "&gt;"; break; case '"': o += "&quot;"; break; case '\'': o += "&apos;"; break;
                     default: o += (c < 0x20 && c != '\t') ? ' ' : char(c); }
    }
    return o;
}
inline std::string win_quote(const std::string& a) {   // MSVCRT / CommandLineToArgvW
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    std::string o = "\""; std::size_t bs = 0;
    for (char c : a) {
        if (c == '\\') { ++bs; continue; }
        if (c == '"') { o.append(bs * 2 + 1, '\\'); o += '"'; bs = 0; continue; }
        o.append(bs, '\\'); bs = 0; o += c;
    }
    o.append(bs * 2, '\\'); o += '"'; return o;
}
inline constexpr const char* run_key = "Software\\Microsoft\\Windows\\CurrentVersion\\Run";
}
inline std::string render_desktop_entry(const entry& e) {
    std::string exec; for (auto& a : e.argv) { if (!exec.empty()) exec += ' '; exec += detail_autostart::desktop_exec_arg(a); }
    return "[Desktop Entry]\nType=Application\nName=" + detail_autostart::desktop_string(e.name) + "\n" +
           (e.description.empty() ? std::string() : "Comment=" + detail_autostart::desktop_string(e.description) + "\n") +
           "Exec=" + exec + "\nTerminal=false\nX-GNOME-Autostart-enabled=true\n";
}
inline std::string render_launch_agent(const entry& e) {
    std::string args; for (auto& a : e.argv) args += "        <string>" + detail_autostart::xml_text(a) + "</string>\n";
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
           "<plist version=\"1.0\">\n<dict>\n    <key>Label</key>\n    <string>" + detail_autostart::xml_text(e.name) + "</string>\n    <key>ProgramArguments</key>\n    <array>\n" + args +
           "    </array>\n    <key>RunAtLoad</key>\n    <true/>\n    <key>ProcessType</key>\n    <string>Interactive</string>\n</dict>\n</plist>\n";
}
inline std::string render_command_line(const entry& e) {
    std::string c; for (auto& a : e.argv) { if (!c.empty()) c += ' '; c += detail_autostart::win_quote(a); } return c;
}
// Where the entry lives: a file path, or on Windows "HKCU\<key>\<name>".
inline std::string location(const entry& e) {
#if defined(FORGE_OS_WINDOWS)
    return "HKCU\\" + (e.dir.empty() ? std::string(detail_autostart::run_key) : e.dir) + "\\" + e.name;
#elif defined(__APPLE__)
    return (e.dir.empty() ? home_dir() + "/Library/LaunchAgents" : e.dir) + "/" + e.name + ".plist";
#else
    std::string d = e.dir;
    if (d.empty()) { d = getenv("XDG_CONFIG_HOME"); if (d.empty() || d[0] != '/') d = home_dir() + "/.config"; d += "/autostart"; }
    return d + "/" + e.name + ".desktop";
#endif
}
inline status enable(const entry& e) {
    if (auto v = validate(e); !v) return v;
#if defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    return {{}, unsupported("autostart")};
#elif defined(FORGE_OS_WINDOWS)
    std::wstring key = widen(e.dir.empty() ? std::string(detail_autostart::run_key) : e.dir), val = widen(render_command_line(e)); HKEY k = nullptr;
    LSTATUS rc = RegCreateKeyExW(HKEY_CURRENT_USER, key.c_str(), 0, nullptr, 0, KEY_SET_VALUE, nullptr, &k, nullptr);
    if (rc != ERROR_SUCCESS) return {{}, error{int(rc), "autostart: open " + location(e) + ": " + error_text(int(rc))}};
    rc = RegSetValueExW(k, widen(e.name).c_str(), 0, REG_SZ, reinterpret_cast<const BYTE*>(val.c_str()), DWORD((val.size() + 1) * sizeof(wchar_t)));
    RegCloseKey(k);
    if (rc != ERROR_SUCCESS) return {{}, error{int(rc), "autostart: write " + location(e) + ": " + error_text(int(rc))}};
    return {};
#else
    std::string p = location(e);
    if (auto m = mkdirs(dirname(p)); !m) return m;
#if defined(__APPLE__)
    return write_file(p, render_launch_agent(e));
#else
    return write_file(p, render_desktop_entry(e));
#endif
#endif
}
inline result<bool> is_enabled(const entry& e) {
    if (!valid_name(e.name)) return {false, errno_error(EINVAL, "autostart: invalid name '" + e.name + "'")};
#if defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    return {false, unsupported("autostart")};
#elif defined(FORGE_OS_WINDOWS)
    std::wstring key = widen(e.dir.empty() ? std::string(detail_autostart::run_key) : e.dir); HKEY k = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, key.c_str(), 0, KEY_QUERY_VALUE, &k) != ERROR_SUCCESS) return {false, {}};
    LSTATUS rc = RegQueryValueExW(k, widen(e.name).c_str(), nullptr, nullptr, nullptr, nullptr); RegCloseKey(k);
    return {rc == ERROR_SUCCESS, {}};
#else
    auto st = stat(location(e)); if (!st.exists) return {false, {}};
#if !defined(__APPLE__)
    auto f = read_file(location(e));   // a desktop environment disables an entry by adding Hidden=true
    if (f && (f.value.find("\nHidden=true") != std::string::npos || f.value.find("\nX-GNOME-Autostart-enabled=false") != std::string::npos)) return {false, {}};
#endif
    return {true, {}};
#endif
}
// Removing an entry that is not there succeeds.
inline status disable(const entry& e) {
    if (!valid_name(e.name)) return {{}, errno_error(EINVAL, "autostart: invalid name '" + e.name + "'")};
#if defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    return {{}, unsupported("autostart")};
#elif defined(FORGE_OS_WINDOWS)
    std::wstring key = widen(e.dir.empty() ? std::string(detail_autostart::run_key) : e.dir); HKEY k = nullptr;
    LSTATUS rc = RegOpenKeyExW(HKEY_CURRENT_USER, key.c_str(), 0, KEY_SET_VALUE, &k);
    if (rc == ERROR_FILE_NOT_FOUND) return {};
    if (rc != ERROR_SUCCESS) return {{}, error{int(rc), "autostart: open " + location(e) + ": " + error_text(int(rc))}};
    rc = RegDeleteValueW(k, widen(e.name).c_str()); RegCloseKey(k);
    if (rc != ERROR_SUCCESS && rc != ERROR_FILE_NOT_FOUND) return {{}, error{int(rc), "autostart: delete " + location(e) + ": " + error_text(int(rc))}};
    return {};
#else
    std::string p = location(e); if (!stat(p).exists) return {};
    return remove(p);
#endif
}
} // namespace forge::os::autostart
