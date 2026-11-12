#pragma once
// forge::os login sessions and privilege elevation.
//   login_sessions()  who is logged in.  Linux, macOS, FreeBSD: the utmpx database (USER_PROCESS records).
//                     Windows: WTSEnumerateSessionsW + WTSQuerySessionInformationW (wtsapi32, loaded at run
//                     time).  Android / iOS: unsupported (one app, one user).
//   run_elevated()    run a command with administrator rights and wait for it.
//                     Already elevated (root / an elevated token): the command simply runs.
//                     Otherwise, without allow_prompt: `sudo -n` (or `doas -n`) on POSIX - it works only when
//                     no password is needed and fails at once otherwise; Windows cannot elevate silently, so the
//                     call fails with needs_prompt(err).
//                     With allow_prompt: Linux/FreeBSD pkexec in a desktop session (a polkit dialog), else sudo
//                     on the terminal; macOS `osascript ... with administrator privileges` (the system dialog);
//                     Windows ShellExecuteExW "runas" (the UAC consent prompt).
//                     Android / iOS: unsupported.
#include <forge/os/users.hpp>
#include <forge/os/process.hpp>
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_ANDROID) && !defined(FORGE_OS_IOS)
#include <utmpx.h>
#define FORGE_OS_UTMPX 1
#endif
#if defined(FORGE_OS_WINDOWS)
#include <shellapi.h>   // SHELLEXECUTEINFOW (types only: shell32 is loaded at run time)
#endif

namespace forge::os {
struct login_session {
    std::string user, terminal, host;   // terminal: tty line ("pts/0", "console") or the Windows station ("Console", "RDP-Tcp#0")
    long id = 0;                        // Windows session id; the session leader's pid elsewhere
    std::int64_t login_time_ns = 0;     // wall clock, 0 unknown
    bool active = true;                 // Windows: the session is connected and in use
};
inline result<std::vector<login_session>> login_sessions() {
    std::vector<login_session> out;
#if defined(FORGE_OS_UTMPX)
    std::lock_guard<std::mutex> g(detail_users::db_mutex());
    ::setutxent();
    while (utmpx* u = ::getutxent()) {
        if (u->ut_type != USER_PROCESS) continue;
        login_session s;
        s.user.assign(u->ut_user, ::strnlen(u->ut_user, sizeof u->ut_user)); s.terminal.assign(u->ut_line, ::strnlen(u->ut_line, sizeof u->ut_line));
        s.host.assign(u->ut_host, ::strnlen(u->ut_host, sizeof u->ut_host));
        s.id = long(u->ut_pid); s.login_time_ns = std::int64_t(u->ut_tv.tv_sec) * 1000000000ll + std::int64_t(u->ut_tv.tv_usec) * 1000ll;
        if (!s.user.empty()) out.push_back(std::move(s));
    }
    ::endutxent();
    return {std::move(out), {}};
#elif defined(FORGE_OS_WINDOWS)
    static HMODULE wts = LoadLibraryExW(L"wtsapi32.dll", nullptr, 0x800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/);
    struct info { DWORD id; wchar_t* station; int state; };
    using enum_fn = BOOL(WINAPI*)(HANDLE, DWORD, DWORD, info**, DWORD*); using q_fn = BOOL(WINAPI*)(HANDLE, DWORD, int, wchar_t**, DWORD*); using f_fn = void(WINAPI*)(void*);
    auto en = wts ? reinterpret_cast<enum_fn>(reinterpret_cast<void*>(GetProcAddress(wts, "WTSEnumerateSessionsW"))) : nullptr;
    auto q = wts ? reinterpret_cast<q_fn>(reinterpret_cast<void*>(GetProcAddress(wts, "WTSQuerySessionInformationW"))) : nullptr;
    auto fr = wts ? reinterpret_cast<f_fn>(reinterpret_cast<void*>(GetProcAddress(wts, "WTSFreeMemory"))) : nullptr;
    if (!en || !q || !fr) return {{}, unsupported("login_sessions (wtsapi32)")};
    info* list = nullptr; DWORD n = 0;
    if (!en(nullptr, 0, 1, &list, &n)) return {{}, last_error("WTSEnumerateSessions")};
    for (DWORD i = 0; i < n; ++i) {
        wchar_t* name = nullptr; DWORD bytes = 0; login_session s;
        if (q(nullptr, list[i].id, 5 /*WTSUserName*/, &name, &bytes) && name) { s.user = narrow(name); fr(name); }
        if (s.user.empty()) continue;   // the services session, listeners
        s.id = long(list[i].id); s.terminal = list[i].station ? narrow(list[i].station) : std::string(); s.active = list[i].state == 0 /*WTSActive*/;
        wchar_t* client = nullptr;
        if (q(nullptr, list[i].id, 10 /*WTSClientName*/, &client, &bytes) && client) { s.host = narrow(client); fr(client); }
        out.push_back(std::move(s));
    }
    fr(list);
    return {std::move(out), {}};
#else
    return {{}, unsupported("login_sessions")};
#endif
}

inline constexpr int needs_prompt_code = 0x0F0E7;
// True when run_elevated() failed only because elevating needs a password / consent prompt (retry with allow_prompt).
inline bool needs_prompt(const error& e) { return e.code == needs_prompt_code; }
namespace detail_elevate {
inline std::string sh_quote(const std::string& s) { std::string o = "'"; for (char c : s) { if (c == '\'') o += "'\\''"; else o += c; } return o + "'"; }
inline std::string applescript_string(const std::string& s) { std::string o = "\""; for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; } return o + "\""; }
inline std::string win_quote(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    std::string o = "\""; std::size_t bs = 0;
    for (char c : a) { if (c == '\\') { ++bs; continue; } if (c == '"') { o.append(bs * 2 + 1, '\\'); o += '"'; bs = 0; continue; } o.append(bs, '\\'); bs = 0; o += c; }
    o.append(bs * 2, '\\'); o += '"'; return o;
}
}
// The command line run_elevated() would execute to elevate argv on this OS (empty: an OS API is used, or there
// is no way). Exposed so callers can show it and tests can check the quoting.
inline std::vector<std::string> elevation_command(const std::vector<std::string>& argv, bool allow_prompt) {
    std::vector<std::string> v;
#if defined(FORGE_OS_MACOS)
    if (allow_prompt) {
        std::string sh; for (auto& a : argv) { if (!sh.empty()) sh += ' '; sh += detail_elevate::sh_quote(a); }
        return {"/usr/bin/osascript", "-e", "do shell script " + detail_elevate::applescript_string(sh) + " with administrator privileges"};
    }
    v = {"/usr/bin/sudo", "-n", "--"};
#elif defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    if (allow_prompt && (has_env("DISPLAY") || has_env("WAYLAND_DISPLAY")) && !find_executable("pkexec").empty()) v = {"pkexec"};
    else if (!find_executable("sudo").empty()) { v = {"sudo"}; if (!allow_prompt) v.push_back("-n"); v.push_back("--"); }
    else if (!find_executable("doas").empty()) { v = {"doas"}; if (!allow_prompt) v.push_back("-n"); v.push_back("--"); }
    else return {};
#else
    (void)allow_prompt; return {};
#endif
    v.insert(v.end(), argv.begin(), argv.end()); return v;
}
// Runs argv elevated and returns its exit code. timeout_ms 0 = wait forever (a prompt may be on screen).
inline result<int> run_elevated(const std::vector<std::string>& argv, bool allow_prompt = false, std::uint64_t timeout_ms = 0) {
    if (argv.empty() || argv[0].empty()) return {-1, errno_error(EINVAL, "run_elevated: empty command")};
    for (auto& a : argv) if (a.find('\0') != std::string::npos) return {-1, errno_error(EINVAL, "run_elevated: NUL in argument")};
#if defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    (void)allow_prompt; (void)timeout_ms; return {-1, unsupported("run_elevated")};
#else
    if (is_elevated()) {
        spawn_spec sp; sp.argv = argv; sp.timeout_ms = timeout_ms; sp.capture_stdout = sp.capture_stderr = false;
        auto r = spawn_wait(sp); if (!r) return {-1, r.err};
        return {r->exit_code, {}};
    }
#if defined(FORGE_OS_WINDOWS)
    if (!allow_prompt) return {-1, error{needs_prompt_code, "run_elevated: Windows elevates only through the UAC consent prompt"}};
    static HMODULE sh = LoadLibraryExW(L"shell32.dll", nullptr, 0x800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/);
    auto exec = sh ? reinterpret_cast<BOOL(WINAPI*)(SHELLEXECUTEINFOW*)>(reinterpret_cast<void*>(GetProcAddress(sh, "ShellExecuteExW"))) : nullptr;
    if (!exec) return {-1, unsupported("run_elevated (shell32 ShellExecuteExW)")};
    std::string params; for (std::size_t i = 1; i < argv.size(); ++i) { if (!params.empty()) params += ' '; params += detail_elevate::win_quote(argv[i]); }
    std::wstring file = widen(argv[0]), wparams = widen(params);
    SHELLEXECUTEINFOW ei{}; ei.cbSize = sizeof ei; ei.fMask = 0x00000040 /*SEE_MASK_NOCLOSEPROCESS*/ | 0x00000400 /*SEE_MASK_FLAG_NO_UI*/;
    ei.lpVerb = L"runas"; ei.lpFile = file.c_str(); ei.lpParameters = wparams.empty() ? nullptr : wparams.c_str(); ei.nShow = 0 /*SW_HIDE*/;
    if (!exec(&ei) || !ei.hProcess) { DWORD e = GetLastError(); if (e == ERROR_CANCELLED) return {-1, errno_error(EACCES, "run_elevated: the consent prompt was declined")}; return {-1, error{int(e ? e : DWORD(-1)), "run_elevated: ShellExecuteEx runas: " + error_text(int(e))}}; }
    DWORD w = WaitForSingleObject(ei.hProcess, timeout_ms ? DWORD(timeout_ms) : INFINITE), code = DWORD(-1);
    if (w == WAIT_TIMEOUT) { CloseHandle(ei.hProcess); return {-1, errno_error(ETIMEDOUT, "run_elevated: timed out")}; }
    GetExitCodeProcess(ei.hProcess, &code); CloseHandle(ei.hProcess);
    return {int(code), {}};
#else
    auto cmd = elevation_command(argv, allow_prompt);
    if (cmd.empty()) return {-1, unsupported("run_elevated (no sudo, doas or pkexec)")};
    if (cmd[0].find('/') != std::string::npos && !stat(cmd[0]).exists) return {-1, unsupported("run_elevated (no " + cmd[0] + ")")};
    spawn_spec sp; sp.argv = cmd; sp.timeout_ms = timeout_ms; sp.capture_stdout = false; sp.capture_stderr = !allow_prompt;   // the prompt goes to the terminal
    auto r = spawn_wait(sp); if (!r) return {-1, r.err};
    if (!allow_prompt && r->exit_code == 1 && (r->err.find("password") != std::string::npos || r->err.find("a terminal is required") != std::string::npos || r->err.find("Authentication") != std::string::npos || r->err.find("not in the sudoers") != std::string::npos || r->err.find("Operation not permitted") != std::string::npos))
        return {-1, error{needs_prompt_code, "run_elevated: " + cmd[0] + " needs a password"}};
    return {r->exit_code, {}};
#endif
#endif
}
} // namespace forge::os
