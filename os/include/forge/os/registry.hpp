#pragma once
// forge::os::registry: Windows registry read / write / enumerate / delete for HKCU, HKLM, HKCR, HKU.
// Keys are "HKCU\\Software\\Vendor\\App"; values are strings (REG_SZ / REG_EXPAND_SZ) or 64-bit integers
// (REG_DWORD / REG_QWORD). Every call returns unsupported() on other platforms. Guide: docs/os-wrapper.md.
#include <forge/os/os.hpp>
#include <variant>

namespace forge::os::registry {
using value = std::variant<std::string, std::int64_t>;
#if defined(FORGE_OS_WINDOWS)
inline std::wstring w_(const std::string& s) { if (s.empty()) return {}; int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0); std::wstring w(std::size_t(n), L'\0'); MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n); return w; }
inline std::string u_(const wchar_t* w, int len = -1) { int n = WideCharToMultiByte(CP_UTF8, 0, w, len, nullptr, 0, nullptr, nullptr); std::string s(std::size_t(n > 0 ? n : 0), '\0'); WideCharToMultiByte(CP_UTF8, 0, w, len, s.data(), n, nullptr, nullptr); if (len < 0 && !s.empty()) s.pop_back(); return s; }
inline bool split_(const std::string& key, HKEY& root, std::wstring& sub) {
    auto p = key.find('\\'); std::string r = key.substr(0, p); sub = p == std::string::npos ? L"" : w_(key.substr(p + 1));
    if (r == "HKCU" || r == "HKEY_CURRENT_USER") root = HKEY_CURRENT_USER;
    else if (r == "HKLM" || r == "HKEY_LOCAL_MACHINE") root = HKEY_LOCAL_MACHINE;
    else if (r == "HKCR" || r == "HKEY_CLASSES_ROOT") root = HKEY_CLASSES_ROOT;
    else if (r == "HKU" || r == "HKEY_USERS") root = HKEY_USERS;
    else return false;
    return true;
}
inline error rerr_(LSTATUS e, const std::string& what) { return {int(e), what + ": " + error_text(int(e))}; }
#endif
inline result<value> get(const std::string& key, const std::string& name) {
#if defined(FORGE_OS_WINDOWS)
    HKEY root; std::wstring sub; if (!split_(key, root, sub)) return {{}, {EINVAL, "registry: bad root in " + key}};
    DWORD type = 0, size = 0; auto wn = w_(name);
    LSTATUS e = RegGetValueW(root, sub.c_str(), wn.c_str(), RRF_RT_ANY | RRF_NOEXPAND, &type, nullptr, &size);
    if (e != ERROR_SUCCESS) return {{}, rerr_(e, "registry get " + key + "\\" + name)};
    std::vector<std::uint8_t> buf(size + 2);
    e = RegGetValueW(root, sub.c_str(), wn.c_str(), RRF_RT_ANY | RRF_NOEXPAND, &type, buf.data(), &size);
    if (e != ERROR_SUCCESS) return {{}, rerr_(e, "registry get " + key + "\\" + name)};
    if (type == REG_DWORD) { DWORD v; std::memcpy(&v, buf.data(), sizeof v); return {value(std::int64_t(v)), {}}; }
    if (type == REG_QWORD) { std::uint64_t v; std::memcpy(&v, buf.data(), sizeof v); return {value(std::int64_t(v)), {}}; }
    if (type == REG_SZ || type == REG_EXPAND_SZ) return {value(u_(reinterpret_cast<const wchar_t*>(buf.data()))), {}};
    return {{}, unsupported("registry value type " + std::to_string(type))};
#else
    (void)key; (void)name; return {{}, unsupported("registry")};
#endif
}
inline status set(const std::string& key, const std::string& name, const value& v) {
#if defined(FORGE_OS_WINDOWS)
    HKEY root; std::wstring sub; if (!split_(key, root, sub)) return {{}, {EINVAL, "registry: bad root in " + key}};
    LSTATUS e;
    if (auto* s = std::get_if<std::string>(&v)) { auto w = w_(*s); e = RegSetKeyValueW(root, sub.c_str(), w_(name).c_str(), REG_SZ, w.c_str(), DWORD((w.size() + 1) * sizeof(wchar_t))); }
    else { std::uint64_t q = std::uint64_t(std::get<std::int64_t>(v)); e = RegSetKeyValueW(root, sub.c_str(), w_(name).c_str(), REG_QWORD, &q, sizeof q); }
    if (e != ERROR_SUCCESS) return {{}, rerr_(e, "registry set " + key + "\\" + name)};
    return {};
#else
    (void)key; (void)name; (void)v; return {{}, unsupported("registry")};
#endif
}
inline status remove_value(const std::string& key, const std::string& name) {
#if defined(FORGE_OS_WINDOWS)
    HKEY root; std::wstring sub; if (!split_(key, root, sub)) return {{}, {EINVAL, "registry: bad root in " + key}};
    LSTATUS e = RegDeleteKeyValueW(root, sub.c_str(), w_(name).c_str()); if (e != ERROR_SUCCESS) return {{}, rerr_(e, "registry delete " + key + "\\" + name)};
    return {};
#else
    (void)key; (void)name; return {{}, unsupported("registry")};
#endif
}
inline status remove_key(const std::string& key) {
#if defined(FORGE_OS_WINDOWS)
    HKEY root; std::wstring sub; if (!split_(key, root, sub) || sub.empty()) return {{}, {EINVAL, "registry: bad key " + key}};
    LSTATUS e = RegDeleteTreeW(root, sub.c_str()); if (e == ERROR_SUCCESS) e = RegDeleteKeyW(root, sub.c_str());
    if (e != ERROR_SUCCESS && e != ERROR_FILE_NOT_FOUND) return {{}, rerr_(e, "registry delete " + key)};
    return {};
#else
    (void)key; return {{}, unsupported("registry")};
#endif
}
struct listing { std::vector<std::string> subkeys, values; };
inline result<listing> list(const std::string& key) {
#if defined(FORGE_OS_WINDOWS)
    HKEY root; std::wstring sub; if (!split_(key, root, sub)) return {{}, {EINVAL, "registry: bad root in " + key}};
    HKEY h; LSTATUS e = RegOpenKeyExW(root, sub.c_str(), 0, KEY_READ, &h); if (e != ERROR_SUCCESS) return {{}, rerr_(e, "registry open " + key)};
    listing l; wchar_t name[16384];
    for (DWORD i = 0;; ++i) { DWORD n = 16384; if (RegEnumKeyExW(h, i, name, &n, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) break; l.subkeys.push_back(u_(name, int(n))); }
    for (DWORD i = 0;; ++i) { DWORD n = 16384; if (RegEnumValueW(h, i, name, &n, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) break; l.values.push_back(u_(name, int(n))); }
    RegCloseKey(h);
    return {std::move(l), {}};
#else
    (void)key; return {{}, unsupported("registry")};
#endif
}
} // namespace forge::os::registry
