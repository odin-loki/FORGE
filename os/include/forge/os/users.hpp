#pragma once
// forge::os users and standard directories: the current user, user lookup by name or id, elevation, and the
// per-user config / cache / data / state / runtime directories (XDG on Linux/FreeBSD, ~/Library on Apple,
// Known Folders on Windows). Guide: docs/os-wrapper.md.
#include <forge/os/os.hpp>
#if defined(FORGE_OS_POSIX)
#include <grp.h>
#include <pwd.h>
#endif

namespace forge::os {
struct user_info { std::string name, home, shell, full_name; long uid = -1, gid = -1; std::vector<std::string> groups; };

#if defined(FORGE_OS_POSIX)
inline user_info user_from_passwd_(const passwd& p) {
    user_info u; u.name = p.pw_name ? p.pw_name : ""; u.home = p.pw_dir ? p.pw_dir : ""; u.shell = p.pw_shell ? p.pw_shell : "";
    if (p.pw_gecos) { u.full_name = p.pw_gecos; auto c = u.full_name.find(','); if (c != std::string::npos) u.full_name.resize(c); }
    u.uid = long(p.pw_uid); u.gid = long(p.pw_gid);
#if !defined(FORGE_OS_ANDROID)
    std::vector<gid_t> g;
#if defined(__APPLE__)
    int n = 64; std::vector<int> gi(64);
    if (::getgrouplist(p.pw_name, int(p.pw_gid), gi.data(), &n) < 0) { gi.resize(std::size_t(n > 64 ? n : 256)); n = int(gi.size()); ::getgrouplist(p.pw_name, int(p.pw_gid), gi.data(), &n); }
    for (int i = 0; i < n && std::size_t(i) < gi.size(); ++i) g.push_back(gid_t(gi[std::size_t(i)]));
#else
    int n = 64; g.resize(64);
    if (::getgrouplist(p.pw_name, p.pw_gid, g.data(), &n) < 0) { g.resize(std::size_t(n > 64 ? n : 256)); n = int(g.size()); ::getgrouplist(p.pw_name, p.pw_gid, g.data(), &n); }
    g.resize(std::size_t(n) < g.size() ? std::size_t(n) : g.size());
#endif
    for (auto id : g) { std::vector<char> b(4096); group gr{}, *r = nullptr; if (::getgrgid_r(id, &gr, b.data(), b.size(), &r) == 0 && r) u.groups.push_back(r->gr_name); }
#endif
    return u;
}
#endif

inline result<user_info> lookup_user(const std::string& name) {
#if defined(FORGE_OS_POSIX)
    std::vector<char> b(16384); passwd p{}, *r = nullptr;
    int rc = ::getpwnam_r(name.c_str(), &p, b.data(), b.size(), &r);
    if (rc || !r) return {{}, error{rc ? rc : ENOENT, "user " + name + ": not found"}};
    return {user_from_passwd_(p), {}};
#else
    (void)name; return {{}, unsupported("lookup_user")};
#endif
}
inline result<user_info> lookup_uid(long uid) {
#if defined(FORGE_OS_POSIX)
    std::vector<char> b(16384); passwd p{}, *r = nullptr;
    int rc = ::getpwuid_r(uid_t(uid), &p, b.data(), b.size(), &r);
    if (rc || !r) return {{}, error{rc ? rc : ENOENT, "uid " + std::to_string(uid) + ": not found"}};
    return {user_from_passwd_(p), {}};
#else
    (void)uid; return {{}, unsupported("lookup_uid")};
#endif
}
inline user_info current_user() {
#if defined(FORGE_OS_POSIX)
    if (auto r = lookup_uid(long(::getuid()))) return r.value;
    user_info u; u.uid = long(::getuid()); u.gid = long(::getgid()); u.name = getenv("USER"); u.home = home_dir(); return u;
#else
    user_info u; char b[256]; DWORD n = sizeof b;
    if (GetUserNameA(b, &n)) u.name = b; else u.name = getenv("USERNAME");
    u.home = home_dir(); u.shell = getenv("COMSPEC");
    return u;
#endif
}
// Root on POSIX (effective uid 0); an elevated token on Windows.
inline bool is_elevated() {
#if defined(FORGE_OS_POSIX)
    return ::geteuid() == 0;
#else
    HANDLE t = nullptr; if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &t)) return false;
    TOKEN_ELEVATION e{}; DWORD n = 0; bool r = GetTokenInformation(t, TokenElevation, &e, sizeof e, &n) && e.TokenIsElevated;
    CloseHandle(t); return r;
#endif
}

// ---------------------------------------------------------------- standard per-user directories
enum class user_dir { config, cache, data, state, runtime, documents, downloads, desktop };
inline std::string user_directory(user_dir d) {
    std::string h = home_dir();
#if defined(FORGE_OS_WINDOWS)
    std::string ad = getenv("APPDATA", h + "\\AppData\\Roaming"), lad = getenv("LOCALAPPDATA", h + "\\AppData\\Local");
    switch (d) {
        case user_dir::config: case user_dir::data: return ad;
        case user_dir::cache: return lad + "\\Cache";
        case user_dir::state: return lad;
        case user_dir::runtime: return temp_dir();
        case user_dir::documents: return h + "\\Documents";
        case user_dir::downloads: return h + "\\Downloads";
        case user_dir::desktop: return h + "\\Desktop";
    }
#elif defined(__APPLE__)
    switch (d) {
        case user_dir::config: case user_dir::data: case user_dir::state: return h + "/Library/Application Support";
        case user_dir::cache: return h + "/Library/Caches";
        case user_dir::runtime: return temp_dir();
        case user_dir::documents: return h + "/Documents";
        case user_dir::downloads: return h + "/Downloads";
        case user_dir::desktop: return h + "/Desktop";
    }
#else
    auto x = [&](const char* var, const std::string& dflt) { auto v = getenv(var); return v.empty() || v[0] != '/' ? dflt : v; };
    switch (d) {
        case user_dir::config: return x("XDG_CONFIG_HOME", h + "/.config");
        case user_dir::cache: return x("XDG_CACHE_HOME", h + "/.cache");
        case user_dir::data: return x("XDG_DATA_HOME", h + "/.local/share");
        case user_dir::state: return x("XDG_STATE_HOME", h + "/.local/state");
        case user_dir::runtime: return x("XDG_RUNTIME_DIR", temp_dir());
        case user_dir::documents: return x("XDG_DOCUMENTS_DIR", h + "/Documents");
        case user_dir::downloads: return x("XDG_DOWNLOAD_DIR", h + "/Downloads");
        case user_dir::desktop: return x("XDG_DESKTOP_DIR", h + "/Desktop");
    }
#endif
    return h;
}
inline std::string config_dir() { return user_directory(user_dir::config); }
inline std::string cache_dir() { return user_directory(user_dir::cache); }
inline std::string data_dir() { return user_directory(user_dir::data); }
} // namespace forge::os
