#pragma once
// forge::os::preferences: a small persistent per-user key/value store, kept where each OS keeps application
// settings (settings.hpp is the table of *environment* settings; registry.hpp the raw Windows registry).
//   Windows:      values (REG_SZ) under HKCU\Software\<app>            backend "registry"
//   macOS / iOS:  CFPreferences for the application id <app> (the store behind NSUserDefaults and `defaults`;
//                 CoreFoundation loaded at run time)                   backend "cfpreferences"
//   Linux / FreeBSD: $XDG_CONFIG_HOME/<app>/preferences.conf           backend "file"
//   Android:      a file under options.dir (pass Context.getFilesDir(); there is no usable default without it)
// options.dir forces the file backend on any OS (tests, portable installs): <dir>/<app>.conf.
// The file is "key=value" lines (UTF-8; \\ \n \r escaped), rewritten atomically (temp file + rename) under an
// advisory lock, so concurrent writers from several processes do not lose each other's keys.
// Names: app 1-64 and keys 1-128 characters of [A-Za-z0-9._-]. Values are strings; get_int / get_bool parse them
// (values written by `defaults write -int/-bool` or as REG_DWORD are read as their decimal / true|false text).
#include <forge/os/detail.hpp>
#include <optional>
#if defined(FORGE_OS_POSIX)
#include <sys/file.h>
#endif

namespace forge::os {
struct preferences_options { std::string dir; };
class preferences {
public:
    using options = preferences_options;
    static bool valid_name(const std::string& n, std::size_t max) {
        if (n.empty() || n.size() > max || n[0] == '.' || n[0] == '-') return false;
        for (unsigned char c : n) if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-')) return false;
        return true;
    }
    static result<preferences> open(const std::string& app, options o = {}) {
        if (!valid_name(app, 64)) return {{}, errno_error(EINVAL, "preferences: invalid application name '" + app + "'")};
        preferences p; p.app_ = app;
        if (!o.dir.empty()) { p.file_ = o.dir + "/" + app + ".conf"; p.kind_ = kind::file; return {std::move(p), {}}; }
#if defined(FORGE_OS_WINDOWS)
        p.kind_ = kind::registry;
#elif defined(__APPLE__)
        auto& a = cfp_(); if (!a.ok) return {{}, unsupported("preferences (CFPreferences not loadable)")};
        p.kind_ = kind::cfprefs;
#elif defined(FORGE_OS_ANDROID)
        return {{}, unsupported("preferences (pass the app's files directory as options.dir)")};
#else
        std::string d = getenv("XDG_CONFIG_HOME"); if (d.empty() || d[0] != '/') { auto h = home_dir(); if (h.empty()) return {{}, unsupported("preferences (no home directory)")}; d = h + "/.config"; }
        p.file_ = d + "/" + app + "/preferences.conf"; p.kind_ = kind::file;
#endif
        return {std::move(p), {}};
    }
    const char* backend() const { return kind_ == kind::registry ? "registry" : kind_ == kind::cfprefs ? "cfpreferences" : kind_ == kind::file ? "file" : "none"; }
    // The file path, "HKCU\Software\<app>", or the CFPreferences application id.
    std::string location() const { return kind_ == kind::file ? file_ : kind_ == kind::registry ? "HKCU\\Software\\" + app_ : app_; }

    result<std::optional<std::string>> get(const std::string& key) const {
        if (!valid_name(key, 128)) return {{}, errno_error(EINVAL, "preferences: invalid key '" + key + "'")};
        if (kind_ == kind::file) { auto m = load_(); if (!m) return {{}, m.err}; auto it = m->find(key); if (it == m->end()) return {std::nullopt, {}}; return {it->second, {}}; }
#if defined(FORGE_OS_WINDOWS)
        if (kind_ == kind::registry) {
            HKEY k = nullptr; LSTATUS rc = RegOpenKeyExW(HKEY_CURRENT_USER, widen("Software\\" + app_).c_str(), 0, KEY_QUERY_VALUE, &k);
            if (rc == ERROR_FILE_NOT_FOUND) return {std::nullopt, {}};
            if (rc != ERROR_SUCCESS) return {{}, rerr_("open", rc)};
            std::wstring name = widen(key); DWORD ty = 0, sz = 0;
            rc = RegQueryValueExW(k, name.c_str(), nullptr, &ty, nullptr, &sz);
            if (rc == ERROR_FILE_NOT_FOUND) { RegCloseKey(k); return {std::nullopt, {}}; }
            std::vector<BYTE> buf(sz + sizeof(wchar_t) * 2);
            if (rc == ERROR_SUCCESS) { sz = DWORD(buf.size() - sizeof(wchar_t)); rc = RegQueryValueExW(k, name.c_str(), nullptr, &ty, buf.data(), &sz); }
            RegCloseKey(k);
            if (rc != ERROR_SUCCESS) return {{}, rerr_("read " + key, rc)};
            if (ty == REG_DWORD && sz >= 4) { DWORD v; std::memcpy(&v, buf.data(), 4); return {std::to_string(v), {}}; }
            if (ty == REG_QWORD && sz >= 8) { unsigned long long v; std::memcpy(&v, buf.data(), 8); return {std::to_string(v), {}}; }
            if (ty != REG_SZ && ty != REG_EXPAND_SZ) return {{}, errno_error(EINVAL, "preferences: " + key + " is not a string value")};
            std::wstring w(reinterpret_cast<const wchar_t*>(buf.data()), sz / sizeof(wchar_t)); while (!w.empty() && w.back() == L'\0') w.pop_back();
            return {narrow(w), {}};
        }
#elif defined(__APPLE__)
        if (kind_ == kind::cfprefs) {
            auto& a = cfp_(); auto& cf = detail::cf_api::get();
            auto k = cf.str(key.c_str()), id = cf.str(app_.c_str());
            auto v = a.copy(k, id); cf.CFRelease(k); cf.CFRelease(id);
            if (!v) return {std::nullopt, {}};
            std::optional<std::string> out;
            if (cf.CFGetTypeID(v) == cf.CFStringGetTypeID()) out = cf_string_(v);
            else if (cf.CFGetTypeID(v) == cf.CFNumberGetTypeID()) out = std::to_string(cf.to_ll(v, 0));
            else if (a.bool_type && cf.CFGetTypeID(v) == a.bool_type()) out = v == cf.kCFBooleanTrue ? "true" : "false";
            cf.CFRelease(v);
            if (!out) return {{}, errno_error(EINVAL, "preferences: " + key + " is not a string, number or boolean")};
            return {std::move(out), {}};
        }
#endif
        return {{}, unsupported("preferences")};
    }
    std::string get_or(const std::string& key, const std::string& dflt) const { auto v = get(key); return v && *v ? **v : dflt; }
    std::int64_t get_int(const std::string& key, std::int64_t dflt) const {
        auto v = get(key); if (!v || !*v || (*v)->empty()) return dflt;
        char* e = nullptr; errno = 0; long long x = std::strtoll((*v)->c_str(), &e, 10); return (*e || errno) ? dflt : std::int64_t(x);
    }
    bool get_bool(const std::string& key, bool dflt) const {
        auto v = get(key); if (!v || !*v) return dflt;
        std::string s = **v; for (auto& c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
        if (s == "0" || s == "false" || s == "no" || s == "off") return false;
        return dflt;
    }
    status set(const std::string& key, const std::string& value) {
        if (!valid_name(key, 128)) return {{}, errno_error(EINVAL, "preferences: invalid key '" + key + "'")};
        if (value.size() > (1u << 20) || value.find('\0') != std::string::npos) return {{}, errno_error(EINVAL, "preferences: value too long or contains NUL")};
        if (kind_ == kind::file) return update_([&](map_t& m) { m[key] = value; });
#if defined(FORGE_OS_WINDOWS)
        if (kind_ == kind::registry) {
            HKEY k = nullptr; LSTATUS rc = RegCreateKeyExW(HKEY_CURRENT_USER, widen("Software\\" + app_).c_str(), 0, nullptr, 0, KEY_SET_VALUE, nullptr, &k, nullptr);
            if (rc != ERROR_SUCCESS) return {{}, rerr_("open", rc)};
            std::wstring w = widen(value);
            rc = RegSetValueExW(k, widen(key).c_str(), 0, REG_SZ, reinterpret_cast<const BYTE*>(w.c_str()), DWORD((w.size() + 1) * sizeof(wchar_t))); RegCloseKey(k);
            if (rc != ERROR_SUCCESS) return {{}, rerr_("write " + key, rc)};
            return {};
        }
#elif defined(__APPLE__)
        if (kind_ == kind::cfprefs) return cf_set_(key, &value);
#endif
        return {{}, unsupported("preferences")};
    }
    status set_int(const std::string& key, std::int64_t v) { return set(key, std::to_string(v)); }
    status set_bool(const std::string& key, bool v) { return set(key, v ? "true" : "false"); }
    // Removing a key that is not there succeeds.
    status remove(const std::string& key) {
        if (!valid_name(key, 128)) return {{}, errno_error(EINVAL, "preferences: invalid key '" + key + "'")};
        if (kind_ == kind::file) { if (!stat(file_).exists) return {}; return update_([&](map_t& m) { m.erase(key); }); }
#if defined(FORGE_OS_WINDOWS)
        if (kind_ == kind::registry) {
            HKEY k = nullptr; LSTATUS rc = RegOpenKeyExW(HKEY_CURRENT_USER, widen("Software\\" + app_).c_str(), 0, KEY_SET_VALUE, &k);
            if (rc == ERROR_FILE_NOT_FOUND) return {};
            if (rc != ERROR_SUCCESS) return {{}, rerr_("open", rc)};
            rc = RegDeleteValueW(k, widen(key).c_str()); RegCloseKey(k);
            if (rc != ERROR_SUCCESS && rc != ERROR_FILE_NOT_FOUND) return {{}, rerr_("delete " + key, rc)};
            return {};
        }
#elif defined(__APPLE__)
        if (kind_ == kind::cfprefs) return cf_set_(key, nullptr);
#endif
        return {{}, unsupported("preferences")};
    }
    result<std::vector<std::string>> keys() const {
        std::vector<std::string> out;
        if (kind_ == kind::file) { auto m = load_(); if (!m) return {{}, m.err}; for (auto& kv : *m) out.push_back(kv.first); return {std::move(out), {}}; }
#if defined(FORGE_OS_WINDOWS)
        if (kind_ == kind::registry) {
            HKEY k = nullptr; LSTATUS rc = RegOpenKeyExW(HKEY_CURRENT_USER, widen("Software\\" + app_).c_str(), 0, KEY_QUERY_VALUE, &k);
            if (rc == ERROR_FILE_NOT_FOUND) return {std::move(out), {}};
            if (rc != ERROR_SUCCESS) return {{}, rerr_("open", rc)};
            for (DWORD i = 0;; ++i) {
                wchar_t name[16384]; DWORD n = 16383;
                rc = RegEnumValueW(k, i, name, &n, nullptr, nullptr, nullptr, nullptr); if (rc != ERROR_SUCCESS) break;
                auto s = narrow(std::wstring_view(name, n)); if (valid_name(s, 128)) out.push_back(std::move(s));
            }
            RegCloseKey(k); std::sort(out.begin(), out.end()); return {std::move(out), {}};
        }
#elif defined(__APPLE__)
        if (kind_ == kind::cfprefs) {
            auto& a = cfp_(); auto& cf = detail::cf_api::get(); auto id = cf.str(app_.c_str());
            a.sync(id);
            if (auto arr = a.key_list && a.current_user && a.any_host ? a.key_list(id, a.current_user, a.any_host) : nullptr) {
                for (detail::CFIndex i = 0; i < cf.CFArrayGetCount(arr); ++i) { auto s = cf_string_(cf.CFArrayGetValueAtIndex(arr, i)); if (valid_name(s, 128)) out.push_back(std::move(s)); }
                cf.CFRelease(arr);
            }
            cf.CFRelease(id); std::sort(out.begin(), out.end()); return {std::move(out), {}};
        }
#endif
        return {{}, unsupported("preferences")};
    }
    // Remove every key (and the file / registry key).
    status clear() {
        if (kind_ == kind::file) { if (stat(file_).exists) { if (auto r = os::remove(file_); !r) return r; } if (stat(file_ + ".lock").exists) os::remove(file_ + ".lock"); return {}; }
#if defined(FORGE_OS_WINDOWS)
        if (kind_ == kind::registry) {
            LSTATUS rc = RegDeleteTreeW(HKEY_CURRENT_USER, widen("Software\\" + app_).c_str());
            if (rc != ERROR_SUCCESS && rc != ERROR_FILE_NOT_FOUND) return {{}, rerr_("delete", rc)};
            return {};
        }
#elif defined(__APPLE__)
        if (kind_ == kind::cfprefs) { auto ks = keys(); if (!ks) return {{}, ks.err}; for (auto& k : *ks) if (auto r = cf_set_(k, nullptr); !r) return r; return {}; }
#endif
        return {{}, unsupported("preferences")};
    }

    using map_t = std::map<std::string, std::string>;
    // The file format, exposed for tests: "key=value" lines; '#' comments and malformed lines are skipped.
    static map_t parse(const std::string& text) {
        map_t m; std::size_t i = 0;
        while (i < text.size()) {
            auto e = text.find('\n', i); std::string line = text.substr(i, e == std::string::npos ? std::string::npos : e - i); i = e == std::string::npos ? text.size() : e + 1;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            auto eq = line.find('='); if (line.empty() || line[0] == '#' || eq == std::string::npos) continue;
            std::string k = line.substr(0, eq), v; if (!valid_name(k, 128)) continue;
            for (std::size_t j = eq + 1; j < line.size(); ++j) {
                if (line[j] == '\\' && j + 1 < line.size()) { char c = line[++j]; v += c == 'n' ? '\n' : c == 'r' ? '\r' : c; }
                else v += line[j];
            }
            m[k] = v;
        }
        return m;
    }
    static std::string render(const map_t& m) {
        std::string o = "# forge preferences\n";
        for (auto& [k, v] : m) { o += k; o += '='; for (char c : v) { if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r"; else o += c; } o += '\n'; }
        return o;
    }
private:
    enum class kind { none, file, registry, cfprefs };
    result<map_t> load_() const {
        if (!stat(file_).exists) return {map_t{}, {}};
        auto f = read_file(file_); if (!f) return {{}, f.err};
        return {parse(f.value), {}};
    }
    template <class F> status update_(F&& change) {
        static std::mutex mu; std::lock_guard<std::mutex> g(mu);   // threads of this process; the file lock covers other processes
        if (auto m = mkdirs(dirname(file_)); !m) return m;
#if defined(FORGE_OS_POSIX)
        int lock = ::open((file_ + ".lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (lock >= 0) { int rc; do { rc = ::flock(lock, LOCK_EX); } while (rc != 0 && errno == EINTR); }
#endif
        status r;
        auto m = load_();
        if (!m) r = {{}, m.err};
        else {
            change(*m);
            std::string tmp = file_ + ".tmp" + std::to_string(pid());
            r = write_file(tmp, render(*m));
            if (r) { r = rename(tmp, file_); if (!r) os::remove(tmp); }
        }
#if defined(FORGE_OS_POSIX)
        if (lock >= 0) ::close(lock);
#endif
        return r;
    }
#if defined(FORGE_OS_WINDOWS)
    error rerr_(const std::string& what, LSTATUS rc) const { return {int(rc), "preferences: " + what + " (" + location() + "): " + error_text(int(rc))}; }
#elif defined(__APPLE__)
    struct cfp_api {
        detail::CFTypeRef (*copy)(detail::CFStringRef, detail::CFStringRef) = nullptr;
        void (*set)(detail::CFStringRef, detail::CFTypeRef, detail::CFStringRef) = nullptr;
        detail::Boolean (*sync)(detail::CFStringRef) = nullptr;
        detail::CFTypeRef (*key_list)(detail::CFStringRef, detail::CFStringRef, detail::CFStringRef) = nullptr;
        unsigned long (*bool_type)() = nullptr; detail::CFIndex (*str_len)(detail::CFStringRef) = nullptr;
        detail::CFIndex (*max_size)(detail::CFIndex, std::uint32_t) = nullptr;
        detail::CFStringRef current_user = nullptr, any_host = nullptr; bool ok = false;
    };
    static const cfp_api& cfp_() {
        static const cfp_api a = [] {
            cfp_api x; auto& cf = detail::cf_api::get(); if (!cf.ok) return x;
            x.copy = reinterpret_cast<decltype(x.copy)>(::dlsym(cf.h, "CFPreferencesCopyAppValue")); x.set = reinterpret_cast<decltype(x.set)>(::dlsym(cf.h, "CFPreferencesSetAppValue"));
            x.sync = reinterpret_cast<decltype(x.sync)>(::dlsym(cf.h, "CFPreferencesAppSynchronize")); x.key_list = reinterpret_cast<decltype(x.key_list)>(::dlsym(cf.h, "CFPreferencesCopyKeyList"));
            x.bool_type = reinterpret_cast<decltype(x.bool_type)>(::dlsym(cf.h, "CFBooleanGetTypeID")); x.str_len = reinterpret_cast<decltype(x.str_len)>(::dlsym(cf.h, "CFStringGetLength"));
            x.max_size = reinterpret_cast<decltype(x.max_size)>(::dlsym(cf.h, "CFStringGetMaximumSizeForEncoding"));
            x.current_user = detail::framework_const(cf.h, "kCFPreferencesCurrentUser"); x.any_host = detail::framework_const(cf.h, "kCFPreferencesAnyHost");
            x.ok = x.copy && x.set && x.sync && cf.CFArrayGetCount && cf.CFArrayGetValueAtIndex && cf.CFStringGetCString;
            return x;
        }();
        return a;
    }
    // Any length (cf_api::to_string stops at 1 KiB).
    static std::string cf_string_(detail::CFTypeRef s) {
        auto& a = cfp_(); auto& cf = detail::cf_api::get();
        if (!s || cf.CFGetTypeID(s) != cf.CFStringGetTypeID()) return {};
        if (!a.str_len || !a.max_size) return cf.to_string(s);
        detail::CFIndex cap = a.max_size(a.str_len(s), 0x08000100) + 1; if (cap <= 0 || cap > (8 << 20)) return {};
        std::string b(std::size_t(cap), '\0'); if (!cf.CFStringGetCString(s, b.data(), cap, 0x08000100)) return {};
        b.resize(std::strlen(b.c_str())); return b;
    }
    status cf_set_(const std::string& key, const std::string* value) {
        auto& a = cfp_(); auto& cf = detail::cf_api::get();
        auto k = cf.str(key.c_str()), id = cf.str(app_.c_str()); detail::CFStringRef v = value ? cf.str(value->c_str()) : nullptr;
        if (value && !v) { cf.CFRelease(k); cf.CFRelease(id); return {{}, errno_error(EINVAL, "preferences: value is not valid UTF-8")}; }
        a.set(k, v, id); bool ok = a.sync(id) != 0;
        if (v) cf.CFRelease(v);
        cf.CFRelease(k); cf.CFRelease(id);
        if (!ok) return {{}, error{-1, "preferences: CFPreferencesAppSynchronize failed for " + app_}};
        return {};
    }
#endif
    std::string app_, file_; kind kind_ = kind::none;
};
} // namespace forge::os
