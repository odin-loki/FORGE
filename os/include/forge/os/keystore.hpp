#pragma once
// Secure key storage: put/get/remove a secret by (service, account).
//   macOS/iOS: Keychain (SecItemAdd/SecItemCopyMatching/SecItemDelete, generic passwords) via Security.framework.
//   Windows: DPAPI (CryptProtectData, user scope) blobs under %LOCALAPPDATA%\forge\keys.
//   Linux: libsecret (Secret Service over D-Bus) when a session bus + keyring is available, else the file backend.
//   Android: the app's JNI glue installs a backend (Android Keystore wrapping key + EncryptedFile) with
//   keystore::set_backend(); without it the file backend is used under FORGE_KEYSTORE_DIR / app files dir.
//   File backend (fallback everywhere): one 0600 file per secret in a 0700 dir; NOT encrypted at rest beyond a
//   per-store XOR pad, so it protects only against other users (documented, reported by keystore::backend()).
#include <forge/os/detail.hpp>

namespace forge::os {
struct keystore_backend {
    const char* name = "custom";
    std::function<status(const std::string&, const std::string&, const std::string&)> put;
    std::function<result<std::string>(const std::string&, const std::string&)> get;
    std::function<status(const std::string&, const std::string&)> remove;
};
class keystore {
public:
    static void set_backend(keystore_backend b) { std::lock_guard<std::mutex> g(m_()); custom_() = std::make_unique<keystore_backend>(std::move(b)); }
    static void reset_backend() { std::lock_guard<std::mutex> g(m_()); custom_().reset(); }
    static std::string backend() {
        { std::lock_guard<std::mutex> g(m_()); if (custom_()) return custom_()->name; }
        if (force_file_()) return "file";
#if defined(__APPLE__)
        return "keychain";
#elif defined(FORGE_OS_WINDOWS)
        return "dpapi";
#elif defined(__linux__) && !defined(__ANDROID__)
        return secret_api::get().usable() ? "libsecret" : "file";
#else
        return "file";
#endif
    }
    static status put(const std::string& service, const std::string& account, const std::string& secret) {
        if (auto c = custom_copy_()) return c->put(service, account, secret);
        if (force_file_()) return file_put_(service, account, secret);
#if defined(__APPLE__)
        return kc_put_(service, account, secret);
#elif defined(FORGE_OS_WINDOWS)
        auto blob = dpapi_(secret, true); if (!blob) return {{}, blob.err};
        auto d = dir_(); if (auto m = mkdirs(d); !m) return m;
        return write_file(join(d, file_name_(service, account)), blob.value);
#else
#if defined(__linux__) && !defined(__ANDROID__)
        if (auto& s = secret_api::get(); s.usable()) {
            void* err = nullptr;
            if (s.store(s.schema(), nullptr, ("forge " + service).c_str(), secret.c_str(), nullptr, &err, "service", service.c_str(), "account", account.c_str(), nullptr) && !err) return {};
            if (err) s.clear_error(&err);
        }
#endif
        return file_put_(service, account, secret);
#endif
    }
    static result<std::string> get(const std::string& service, const std::string& account) {
        if (auto c = custom_copy_()) return c->get(service, account);
        if (force_file_()) return file_get_(service, account);
#if defined(__APPLE__)
        return kc_get_(service, account);
#elif defined(FORGE_OS_WINDOWS)
        auto r = read_file(join(dir_(), file_name_(service, account))); if (!r) return {{}, {2, "keystore: no such secret"}};
        return dpapi_(r.value, false);
#else
#if defined(__linux__) && !defined(__ANDROID__)
        if (auto& s = secret_api::get(); s.usable()) {
            void* err = nullptr; char* v = s.lookup(s.schema(), nullptr, &err, "service", service.c_str(), "account", account.c_str(), nullptr);
            if (err) s.clear_error(&err);
            if (v) { std::string out = v; s.free_pw(v); return {out, {}}; }
        }
#endif
        return file_get_(service, account);
#endif
    }
    static status remove(const std::string& service, const std::string& account) {
        if (auto c = custom_copy_()) return c->remove(service, account);
        if (force_file_()) return os::remove(join(dir_(), file_name_(service, account)));
#if defined(__APPLE__)
        return kc_remove_(service, account);
#else
#if defined(__linux__) && !defined(__ANDROID__)
        if (auto& s = secret_api::get(); s.usable()) { void* err = nullptr; s.clear(s.schema(), nullptr, &err, "service", service.c_str(), "account", account.c_str(), nullptr); if (err) s.clear_error(&err); }
#endif
        auto p = join(dir_(), file_name_(service, account)); if (!exists(p)) return {};
        return os::remove(p);
#endif
    }
private:
    static std::mutex& m_() { static std::mutex m; return m; }
    static std::unique_ptr<keystore_backend>& custom_() { static std::unique_ptr<keystore_backend> b; return b; }
    static std::unique_ptr<keystore_backend> custom_copy_() { std::lock_guard<std::mutex> g(m_()); return custom_() ? std::make_unique<keystore_backend>(*custom_()) : nullptr; }
    static bool force_file_() { return getenv("FORGE_KEYSTORE") == "file"; }
    static std::string dir_() {
        if (auto d = getenv("FORGE_KEYSTORE_DIR"); !d.empty()) return d;
#if defined(FORGE_OS_WINDOWS)
        auto b = getenv("LOCALAPPDATA"); return join(b.empty() ? temp_dir() : b, "forge/keys");
#elif defined(__ANDROID__)
        return join(temp_dir(), "forge-keys");
#else
        auto x = getenv("XDG_DATA_HOME"); return join(x.empty() ? join(home_dir(), ".local/share") : x, "forge/keys");
#endif
    }
    static std::string file_name_(const std::string& s, const std::string& a) {   // FNV-1a of both parts, hex
        std::uint64_t h = 1469598103934665603ull; for (char c : s + '\x1f' + a) { h ^= std::uint8_t(c); h *= 1099511628211ull; }
        char b[24]; std::snprintf(b, sizeof b, "%016llx.key", (unsigned long long)h); return b;
    }
    static std::string pad_(const std::string& d) {   // per-store pad (random, created once, 0600)
        auto p = join(d, ".pad"); auto r = read_file(p); if (r && r.value.size() == 64) return r.value;
        std::string k(64, '\0'); entropy(k.data(), k.size()); write_file(p, k);
#if defined(FORGE_OS_POSIX)
        ::chmod(p.c_str(), 0600);
#endif
        return k;
    }
    static status file_put_(const std::string& s, const std::string& a, const std::string& v) {
        auto d = dir_(); if (auto m = mkdirs(d); !m) return m;
#if defined(FORGE_OS_POSIX)
        ::chmod(d.c_str(), 0700);
#endif
        auto k = pad_(d); std::string e = v; for (std::size_t i = 0; i < e.size(); ++i) e[i] = char(e[i] ^ k[i % k.size()]);
        auto p = join(d, file_name_(s, a)), tmp = p + ".tmp";
        if (auto w = write_file(tmp, e); !w) return w;
#if defined(FORGE_OS_POSIX)
        ::chmod(tmp.c_str(), 0600);
#endif
        return rename(tmp, p);
    }
    static result<std::string> file_get_(const std::string& s, const std::string& a) {
        auto d = dir_(); auto r = read_file(join(d, file_name_(s, a))); if (!r) return {{}, {2, "keystore: no such secret"}};
        auto k = pad_(d); for (std::size_t i = 0; i < r.value.size(); ++i) r.value[i] = char(r.value[i] ^ k[i % k.size()]);
        return {r.value, {}};
    }
#if defined(FORGE_OS_WINDOWS)
    static result<std::string> dpapi_(const std::string& in, bool protect) {
        struct blob_t { DWORD cbData; BYTE* pbData; };
        using fn = BOOL(WINAPI*)(blob_t*, LPCWSTR, blob_t*, PVOID, PVOID, DWORD, blob_t*);
        using ufn = BOOL(WINAPI*)(blob_t*, LPWSTR*, blob_t*, PVOID, PVOID, DWORD, blob_t*);
        static HMODULE h = LoadLibraryW(L"crypt32.dll"); if (!h) return {{}, unsupported("crypt32")};
        blob_t i{DWORD(in.size()), reinterpret_cast<BYTE*>(const_cast<char*>(in.data()))}, o{0, nullptr};
        BOOL ok;
        if (protect) ok = reinterpret_cast<fn>(reinterpret_cast<void*>(GetProcAddress(h, "CryptProtectData")))(&i, L"forge", nullptr, nullptr, nullptr, 0x1 /*UI_FORBIDDEN*/, &o);
        else ok = reinterpret_cast<ufn>(reinterpret_cast<void*>(GetProcAddress(h, "CryptUnprotectData")))(&i, nullptr, nullptr, nullptr, nullptr, 0x1, &o);
        if (!ok) return {{}, last_error(protect ? "CryptProtectData" : "CryptUnprotectData")};
        std::string r(reinterpret_cast<char*>(o.pbData), o.cbData); LocalFree(o.pbData); return {r, {}};
    }
#endif
#if defined(__linux__) && !defined(__ANDROID__)
    struct secret_api {
        struct schema_attr { const char* name; int type; };
        struct schema_t { const char* name; int flags; schema_attr attributes[32]; int reserved; void *r1, *r2, *r3, *r4, *r5, *r6, *r7; };
        void* h = nullptr; schema_t sc{};
        int (*store)(const schema_t*, const char*, const char*, const char*, void*, void**, ...) = nullptr;
        char* (*lookup)(const schema_t*, void*, void**, ...) = nullptr;
        int (*clear)(const schema_t*, void*, void**, ...) = nullptr;
        void (*free_pw)(char*) = nullptr; void (*clear_error)(void**) = nullptr; bool ok = false;
        const schema_t* schema() const { return &sc; }
        bool usable() const { return ok && !getenv("DBUS_SESSION_BUS_ADDRESS").empty(); }
        static secret_api& get() {
            static secret_api a = [] {
                secret_api s; s.sc.name = "org.forge.Secret"; s.sc.attributes[0] = {"service", 0}; s.sc.attributes[1] = {"account", 0};
                s.h = ::dlopen("libsecret-1.so.0", RTLD_LAZY); if (!s.h) return s;
                s.store = reinterpret_cast<decltype(s.store)>(::dlsym(s.h, "secret_password_store_sync"));
                s.lookup = reinterpret_cast<decltype(s.lookup)>(::dlsym(s.h, "secret_password_lookup_sync"));
                s.clear = reinterpret_cast<decltype(s.clear)>(::dlsym(s.h, "secret_password_clear_sync"));
                s.free_pw = reinterpret_cast<decltype(s.free_pw)>(::dlsym(s.h, "secret_password_free"));
                if (void* g = ::dlopen("libglib-2.0.so.0", RTLD_LAZY)) s.clear_error = reinterpret_cast<decltype(s.clear_error)>(::dlsym(g, "g_clear_error"));
                s.ok = s.store && s.lookup && s.clear && s.free_pw && s.clear_error; return s;
            }();
            return a;
        }
    };
#endif
#if defined(__APPLE__)
    struct sec_api {
        void* h = nullptr; int (*add)(const void*, const void**) = nullptr; int (*copy)(const void*, const void**) = nullptr; int (*del)(const void*) = nullptr;
        const void *kClass, *kGeneric, *kService, *kAccount, *kValueData, *kReturnData, *kMatchLimit, *kMatchOne;
        static sec_api& get() {
            static sec_api s = [] { sec_api x{}; x.h = ::dlopen("/System/Library/Frameworks/Security.framework/Security", RTLD_LAZY); if (!x.h) return x;
                x.add = reinterpret_cast<decltype(x.add)>(::dlsym(x.h, "SecItemAdd")); x.copy = reinterpret_cast<decltype(x.copy)>(::dlsym(x.h, "SecItemCopyMatching"));
                x.del = reinterpret_cast<decltype(x.del)>(::dlsym(x.h, "SecItemDelete"));
                x.kClass = detail::framework_const(x.h, "kSecClass"); x.kGeneric = detail::framework_const(x.h, "kSecClassGenericPassword");
                x.kService = detail::framework_const(x.h, "kSecAttrService"); x.kAccount = detail::framework_const(x.h, "kSecAttrAccount");
                x.kValueData = detail::framework_const(x.h, "kSecValueData"); x.kReturnData = detail::framework_const(x.h, "kSecReturnData");
                x.kMatchLimit = detail::framework_const(x.h, "kSecMatchLimit"); x.kMatchOne = detail::framework_const(x.h, "kSecMatchLimitOne");
                return x; }();
            return s;
        }
        bool ok() const { return add && copy && del && kClass && detail::cf_api::get().ok; }
    };
    static const void* kc_query_(const std::string& s, const std::string& a, const void* extra_k[], const void* extra_v[], int nextra, std::vector<const void*>& owned) {
        auto& cf = detail::cf_api::get(); auto& k = sec_api::get();
        auto ss = cf.str(s.c_str()), as = cf.str(a.c_str()); owned.push_back(ss); owned.push_back(as);
        const void* keys[8] = {k.kClass, k.kService, k.kAccount}; const void* vals[8] = {k.kGeneric, ss, as};
        for (int i = 0; i < nextra; ++i) { keys[3 + i] = extra_k[i]; vals[3 + i] = extra_v[i]; }
        return cf.CFDictionaryCreate(nullptr, keys, vals, 3 + nextra, cf.kCFTypeDictionaryKeyCallBacks, cf.kCFTypeDictionaryValueCallBacks);
    }
    static status kc_put_(const std::string& s, const std::string& a, const std::string& v) {
        auto& k = sec_api::get(); if (!k.ok()) return {{}, unsupported("Security.framework")};
        kc_remove_(s, a);
        auto& cf = detail::cf_api::get(); std::vector<const void*> owned;
        auto data = cf.CFDataCreate(nullptr, reinterpret_cast<const std::uint8_t*>(v.data()), detail::CFIndex(v.size())); owned.push_back(data);
        const void* ek[] = {k.kValueData}; const void* ev[] = {data};
        auto q = kc_query_(s, a, ek, ev, 1, owned); int rc = k.add(q, nullptr); cf.CFRelease(q); for (auto o : owned) cf.CFRelease(o);
        if (rc) return {{}, {rc, "SecItemAdd failed (OSStatus " + std::to_string(rc) + ")"}}; return {};
    }
    static result<std::string> kc_get_(const std::string& s, const std::string& a) {
        auto& k = sec_api::get(); if (!k.ok()) return {{}, unsupported("Security.framework")};
        auto& cf = detail::cf_api::get(); std::vector<const void*> owned;
        const void* ek[] = {k.kReturnData, k.kMatchLimit}; const void* ev[] = {cf.kCFBooleanTrue, k.kMatchOne};
        auto q = kc_query_(s, a, ek, ev, 2, owned); const void* out = nullptr; int rc = k.copy(q, &out); cf.CFRelease(q); for (auto o : owned) cf.CFRelease(o);
        if (rc || !out) return {{}, {rc ? rc : 2, "keystore: no such secret"}};
        std::string r(reinterpret_cast<const char*>(cf.CFDataGetBytePtr(out)), std::size_t(cf.CFDataGetLength(out))); cf.CFRelease(out); return {r, {}};
    }
    static status kc_remove_(const std::string& s, const std::string& a) {
        auto& k = sec_api::get(); if (!k.ok()) return {{}, unsupported("Security.framework")};
        auto& cf = detail::cf_api::get(); std::vector<const void*> owned;
        auto q = kc_query_(s, a, nullptr, nullptr, 0, owned); int rc = k.del(q); cf.CFRelease(q); for (auto o : owned) cf.CFRelease(o);
        if (rc && rc != -25300 /*errSecItemNotFound*/) return {{}, {rc, "SecItemDelete failed"}}; return {};
    }
#endif
};
} // namespace forge::os
