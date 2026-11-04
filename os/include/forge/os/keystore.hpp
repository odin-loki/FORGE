#pragma once
// Secure key storage: put/get/remove a secret by (service, account).
//   macOS/iOS: Keychain generic passwords (SecItemUpdate, else SecItemAdd with AfterFirstUnlockThisDeviceOnly).
//   Windows: DPAPI (CryptProtectData, user scope) blobs under %LOCALAPPDATA%\forge\keys.
//   Linux/FreeBSD: libsecret (Secret Service over D-Bus) when a session bus has a Secret Service provider; its
//   failures (locked keyring, dismissed prompt) are returned, not papered over. With no bus or no provider the
//   file backend is used and keystore::backend() says "file".
//   Android: the app's JNI glue installs a backend (Android Keystore wrapping key + EncryptedFile) with
//   keystore::set_backend(), or sets FORGE_KEYSTORE_DIR to getFilesDir() for the file backend; otherwise every
//   call fails.
//   File backend (FORGE_KEYSTORE=file forces it): one file per secret, XChaCha20-Poly1305 (Monocypher) under a
//   random 32-byte store key, with a random nonce and the (service, account) pair as associated data, so a
//   tampered, truncated or swapped file fails get(). The store key lives in `.master` beside the secrets (0600,
//   created exclusively; DPAPI-wrapped on Windows), so this protects against other users and against copies of
//   one secret file, not against an attacker who can read the whole directory as this user. Key wrapping by a
//   TPM or a passphrase is planned (docs/os-audit.md W2-4). Files from the old XOR format are re-encrypted on
//   first read and the old pad is deleted once no old file remains.
// Link: the file backend needs Monocypher (forge_os links forge_trust in CMake).
#include <forge/os/detail.hpp>
#include <monocypher.h>
#include <thread>
#if (defined(__linux__) && !defined(__ANDROID__)) || defined(FORGE_OS_FREEBSD)
#define FORGE_OS_KS_LIBSECRET_ 1
#endif

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
#elif defined(__ANDROID__)
        return getenv("FORGE_KEYSTORE_DIR").empty() ? "none" : "file";
#elif defined(FORGE_OS_KS_LIBSECRET_)
        return use_libsecret_() ? "libsecret" : "file";
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
        auto d = dir_(); if (!d) return {{}, d.err}; if (auto m = ensure_dir_(d.value); !m) return m;
        return replace_(d.value, join(d.value, file_name_(service, account)), blob.value);
#else
#if defined(FORGE_OS_KS_LIBSECRET_)
        if (use_libsecret_()) {
            auto& s = secret_api::get(); void* err = nullptr;
            if (s.store(s.schema(), nullptr, ("forge " + service).c_str(), secret.c_str(), nullptr, &err, "service", service.c_str(), "account", account.c_str(), nullptr) && !err) return {};
            return {{}, s.take_error(err, "libsecret store")};
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
        auto d = dir_(); if (!d) return {{}, d.err};
        auto r = read_file(join(d.value, file_name_(service, account))); if (!r) return {{}, {2, "keystore: no such secret"}};
        return dpapi_(r.value, false);
#else
#if defined(FORGE_OS_KS_LIBSECRET_)
        if (use_libsecret_()) {
            auto& s = secret_api::get(); void* err = nullptr; char* v = s.lookup(s.schema(), nullptr, &err, "service", service.c_str(), "account", account.c_str(), nullptr);
            if (err) { if (v) s.free_pw(v); return {{}, s.take_error(err, "libsecret lookup")}; }
            if (!v) return {{}, {2, "keystore: no such secret"}};
            std::string out = v; s.free_pw(v); return {out, {}};
        }
#endif
        return file_get_(service, account);
#endif
    }
    static status remove(const std::string& service, const std::string& account) {
        if (auto c = custom_copy_()) return c->remove(service, account);
        if (force_file_()) return file_remove_(service, account);
#if defined(__APPLE__)
        return kc_remove_(service, account);
#else
#if defined(FORGE_OS_KS_LIBSECRET_)
        if (use_libsecret_()) {
            auto& s = secret_api::get(); void* err = nullptr; s.clear(s.schema(), nullptr, &err, "service", service.c_str(), "account", account.c_str(), nullptr);
            if (err) return {{}, s.take_error(err, "libsecret clear")}; return {};
        }
#endif
        return file_remove_(service, account);
#endif
    }
private:
    static std::mutex& m_() { static std::mutex m; return m; }
    static std::unique_ptr<keystore_backend>& custom_() { static std::unique_ptr<keystore_backend> b; return b; }
    static std::unique_ptr<keystore_backend> custom_copy_() { std::lock_guard<std::mutex> g(m_()); return custom_() ? std::make_unique<keystore_backend>(*custom_()) : nullptr; }
    static bool force_file_() { return getenv("FORGE_KEYSTORE") == "file"; }
    static result<std::string> dir_() {
        if (auto d = getenv("FORGE_KEYSTORE_DIR"); !d.empty()) return {d, {}};
#if defined(FORGE_OS_WINDOWS)
        auto b = getenv("LOCALAPPDATA"); if (b.empty()) return {{}, {EINVAL, "keystore: LOCALAPPDATA is not set (set FORGE_KEYSTORE_DIR)"}};
        return {join(b, "forge/keys"), {}};
#elif defined(__ANDROID__)
        return {{}, {EINVAL, "keystore: set FORGE_KEYSTORE_DIR to the app's files dir or install keystore::set_backend() from JNI"}};
#else
        auto x = getenv("XDG_DATA_HOME"); return {join(x.empty() || x[0] != '/' ? join(home_dir(), ".local/share") : x, "forge/keys"), {}};
#endif
    }
    // Creates the store directory 0700 when missing; an existing directory's mode is left alone.
    static status ensure_dir_(const std::string& d) {
        if (stat(d).is_dir) return {};
        if (auto m = mkdirs(dirname(d)); !m) return m;
#if defined(FORGE_OS_POSIX)
        if (::mkdir(d.c_str(), 0700) && errno != EEXIST) return {{}, last_error("mkdir " + d)};
        return {};
#else
        return mkdir(d);
#endif
    }
    static std::string file_name_(const std::string& s, const std::string& a) {   // FNV-1a of both parts, hex
        std::uint64_t h = 1469598103934665603ull; for (char c : s + '\x1f' + a) { h ^= std::uint8_t(c); h *= 1099511628211ull; }
        char b[24]; std::snprintf(b, sizeof b, "%016llx.key", (unsigned long long)h); return b;
    }
    static std::string tmp_name_(const std::string& d, const std::string& base) {
        std::uint64_t r = 0; entropy(&r, sizeof r); char b[48]; std::snprintf(b, sizeof b, ".%llu.%016llx.tmp", (unsigned long long)pid(), (unsigned long long)r);
        return join(d, base + b);
    }
    // New file with owner-only access from the first byte (O_EXCL, 0600, no symlink follow; CREATE_NEW on
    // Windows, where the directory ACL applies), written and flushed.
    static status create_excl_(const std::string& p, std::string_view data) {
#if defined(FORGE_OS_WINDOWS)
        HANDLE h = CreateFileW(widen(p).c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return {{}, last_error("create " + p)};
        std::size_t off = 0; bool ok = true;
        while (ok && off < data.size()) { DWORD w = 0; ok = WriteFile(h, data.data() + off, DWORD(std::min<std::size_t>(data.size() - off, 1u << 30)), &w, nullptr) && w; off += w; }
        ok = ok && FlushFileBuffers(h); auto e = ok ? error{} : last_error("write " + p); CloseHandle(h);
        if (!ok) { DeleteFileW(widen(p).c_str()); return {{}, e}; }
        return {};
#else
        int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600); if (fd < 0) return {{}, last_error("create " + p)};
        std::size_t off = 0;
        while (off < data.size()) {
            auto w = ::write(fd, data.data() + off, data.size() - off);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) { auto e = last_error("write " + p); ::close(fd); ::unlink(p.c_str()); return {{}, e}; }
            off += std::size_t(w);
        }
        if (::fsync(fd)) { auto e = last_error("fsync " + p); ::close(fd); ::unlink(p.c_str()); return {{}, e}; }
        ::close(fd); return {};
#endif
    }
    static status replace_(const std::string& d, const std::string& p, std::string_view data) {   // atomic replace
        auto tmp = tmp_name_(d, basename(p)); if (auto w = create_excl_(tmp, data); !w) return w;
        if (auto r = rename(tmp, p); !r) { os::remove(tmp); return r; }
        return {};
    }
    // Publish tmp at p only if p does not exist. true = published, false = someone else's p already exists.
    static result<bool> publish_excl_(const std::string& tmp, const std::string& p, std::string_view data) {
#if defined(FORGE_OS_WINDOWS)
        (void)data;
        if (MoveFileExW(widen(tmp).c_str(), widen(p).c_str(), MOVEFILE_WRITE_THROUGH)) return {true, {}};
        auto e = GetLastError(); DeleteFileW(widen(tmp).c_str());
        if (e == ERROR_ALREADY_EXISTS || e == ERROR_FILE_EXISTS) return {false, {}};
        return {false, {int(e), "keystore: publish " + p + ": " + error_text(int(e))}};
#else
        int rc = ::link(tmp.c_str(), p.c_str()); int e = rc ? errno : 0; ::unlink(tmp.c_str());
        if (!rc) return {true, {}};
        if (e == EEXIST) return {false, {}};
        // No hard links here (some FUSE/Android mounts): create in place; readers retry a short file.
        if (auto w = create_excl_(p, data); !w) { if (w.err.code == EEXIST) return {false, {}}; return {false, w.err}; }
        return {true, {}};
#endif
    }
    static result<std::string> wrap_key_(const std::string& k) {
#if defined(FORGE_OS_WINDOWS)
        auto b = dpapi_(k, true); if (!b) return {{}, b.err}; return {"FKKD" + b.value, {}};
#else
        return {"FKK1" + k, {}};
#endif
    }
    static result<std::string> unwrap_key_(const std::string& f) {
#if defined(FORGE_OS_WINDOWS)
        if (f.size() > 4 && f.compare(0, 4, "FKKD") == 0) { auto k = dpapi_(f.substr(4), false); if (k && k.value.size() == 32) return k; }
#else
        if (f.size() == 36 && f.compare(0, 4, "FKK1") == 0) return {f.substr(4), {}};
#endif
        return {{}, {EIO, "keystore: store key is corrupt"}};
    }
    // The store key: read it, or create it exclusively; concurrent creators converge on the first one published.
    static result<std::string> key_(const std::string& d) {
        auto p = join(d, ".master"); error last{EIO, "keystore: store key is corrupt"};
        for (int attempt = 0; attempt < 100; ++attempt) {
            if (auto r = read_file(p)) {
                auto k = unwrap_key_(r.value); crypto_wipe(r.value.data(), r.value.size()); if (k) return k;
                last = k.err; std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue;   // maybe still being written
            }
            if (exists(p)) return {{}, last_error("keystore: read " + p)};
            std::string k(32, '\0'); if (!entropy(k.data(), k.size())) return {{}, {EIO, "keystore: no entropy"}};
            auto w = wrap_key_(k); crypto_wipe(k.data(), k.size()); if (!w) return {{}, w.err};
            auto tmp = tmp_name_(d, ".master");
            auto c = create_excl_(tmp, w.value); if (!c) { crypto_wipe(w.value.data(), w.value.size()); return {{}, c.err}; }
            auto pub = publish_excl_(tmp, p, w.value); crypto_wipe(w.value.data(), w.value.size());
            if (!pub && pub.err.code) return {{}, pub.err};
            // published or lost the race: read back whichever key won
        }
        return {{}, last};
    }
    static std::string ad_(const std::string& s, const std::string& a) { return "forge-ks-v1" + s + '\x1f' + a; }
    static status file_put_(const std::string& s, const std::string& a, const std::string& v) {
        auto d = dir_(); if (!d) return {{}, d.err}; if (auto m = ensure_dir_(d.value); !m) return m;
        auto k = key_(d.value); if (!k) return {{}, k.err};
        std::string out(4 + 24 + 16 + v.size(), '\0'); std::memcpy(out.data(), "FKS1", 4);
        auto* nonce = reinterpret_cast<std::uint8_t*>(out.data() + 4); auto* mac = nonce + 24; auto* ct = mac + 16;
        if (!entropy(nonce, 24)) { crypto_wipe(k.value.data(), 32); return {{}, {EIO, "keystore: no entropy"}}; }
        auto ad = ad_(s, a);
        crypto_aead_lock(ct, mac, reinterpret_cast<const std::uint8_t*>(k.value.data()), nonce, reinterpret_cast<const std::uint8_t*>(ad.data()), ad.size(),
                         reinterpret_cast<const std::uint8_t*>(v.data()), v.size());
        crypto_wipe(k.value.data(), k.value.size());
        return replace_(d.value, join(d.value, file_name_(s, a)), out);
    }
    static result<std::string> file_get_(const std::string& s, const std::string& a) {
        auto d = dir_(); if (!d) return {{}, d.err};
        auto r = read_file(join(d.value, file_name_(s, a))); if (!r) return {{}, {2, "keystore: no such secret"}};
        if (r.value.size() >= 44 && r.value.compare(0, 4, "FKS1") == 0) {
            auto k = key_(d.value); if (!k) return {{}, k.err};
            auto* nonce = reinterpret_cast<const std::uint8_t*>(r.value.data() + 4); auto* mac = nonce + 24; auto* ct = mac + 16;
            std::string pt(r.value.size() - 44, '\0'); auto ad = ad_(s, a);
            int bad = crypto_aead_unlock(reinterpret_cast<std::uint8_t*>(pt.data()), mac, reinterpret_cast<const std::uint8_t*>(k.value.data()), nonce,
                                         reinterpret_cast<const std::uint8_t*>(ad.data()), ad.size(), ct, pt.size());
            crypto_wipe(k.value.data(), k.value.size());
            if (bad) return {{}, {EIO, "keystore: secret failed authentication (tampered, corrupt or store key replaced)"}};
            return {std::move(pt), {}};
        }
        return legacy_get_(d.value, s, a, std::move(r.value));
    }
    // Old format: plaintext XOR a 64-byte `.pad`. Decode, re-store encrypted, drop the pad when no old file is left.
    static result<std::string> legacy_get_(const std::string& d, const std::string& s, const std::string& a, std::string v) {
        auto pad = read_file(join(d, ".pad"));
        if (!pad || pad.value.size() != 64) return {{}, {EIO, "keystore: secret is in an unknown format"}};
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = char(v[i] ^ pad.value[i % 64]);
        if (auto st = file_put_(s, a, v); !st) return {{}, st.err};
        bool old_left = false;
        if (auto l = list_dir(d))
            for (auto& e : l.value)
                if (e.name.size() == 20 && e.name.compare(16, 4, ".key") == 0) { auto t = read_file(join(d, e.name)); if (t && (t.value.size() < 44 || t.value.compare(0, 4, "FKS1") != 0)) { old_left = true; break; } }
        if (!old_left) os::remove(join(d, ".pad"));
        return {std::move(v), {}};
    }
    static status file_remove_(const std::string& s, const std::string& a) {
        auto d = dir_(); if (!d) return {{}, d.err};
        auto p = join(d.value, file_name_(s, a)); if (!exists(p)) return {};
        return os::remove(p);
    }
#if defined(FORGE_OS_WINDOWS)
    static result<std::string> dpapi_(const std::string& in, bool protect) {
        struct blob_t { DWORD cbData; BYTE* pbData; };
        using fn = BOOL(WINAPI*)(blob_t*, LPCWSTR, blob_t*, PVOID, PVOID, DWORD, blob_t*);
        using ufn = BOOL(WINAPI*)(blob_t*, LPWSTR*, blob_t*, PVOID, PVOID, DWORD, blob_t*);
        static HMODULE h = LoadLibraryExW(L"crypt32.dll", nullptr, 0x00000800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/); if (!h) return {{}, unsupported("crypt32")};
        blob_t i{DWORD(in.size()), reinterpret_cast<BYTE*>(const_cast<char*>(in.data()))}, o{0, nullptr};
        BOOL ok;
        if (protect) ok = reinterpret_cast<fn>(reinterpret_cast<void*>(GetProcAddress(h, "CryptProtectData")))(&i, L"forge", nullptr, nullptr, nullptr, 0x1 /*UI_FORBIDDEN*/, &o);
        else ok = reinterpret_cast<ufn>(reinterpret_cast<void*>(GetProcAddress(h, "CryptUnprotectData")))(&i, nullptr, nullptr, nullptr, nullptr, 0x1, &o);
        if (!ok) return {{}, last_error(protect ? "CryptProtectData" : "CryptUnprotectData")};
        std::string r(reinterpret_cast<char*>(o.pbData), o.cbData); SecureZeroMemory(o.pbData, o.cbData); LocalFree(o.pbData); return {r, {}};
    }
#endif
#if defined(FORGE_OS_KS_LIBSECRET_)
    struct secret_api {
        struct schema_attr { const char* name; int type; };
        struct schema_t { const char* name; int flags; schema_attr attributes[32]; int reserved; void *r1, *r2, *r3, *r4, *r5, *r6, *r7; };
        struct gerror_t { std::uint32_t domain; int code; char* message; };
        void* h = nullptr; schema_t sc{};
        int (*store)(const schema_t*, const char*, const char*, const char*, void*, void**, ...) = nullptr;
        char* (*lookup)(const schema_t*, void*, void**, ...) = nullptr;
        int (*clear)(const schema_t*, void*, void**, ...) = nullptr;
        void (*free_pw)(char*) = nullptr; void (*clear_error)(void**) = nullptr; std::uint32_t (*quark)(const char*) = nullptr; bool ok = false;
        const schema_t* schema() const { return &sc; }
        // GError -> error, freeing it.
        error take_error(void* e, const char* what) const {
            auto* g = static_cast<gerror_t*>(e); error r{g->code ? g->code : EIO, std::string("keystore: ") + what + ": " + (g->message ? g->message : "failed")};
            void* p = e; clear_error(&p); return r;
        }
        // No bus or no Secret Service provider on it (as opposed to a provider that refused).
        bool absent(void* e) const {
            auto* g = static_cast<gerror_t*>(e); if (!quark) return false;
            if (g->domain == quark("g-io-error-quark")) return true;
            if (g->domain != quark("g-dbus-error-quark")) return false;
            int c = g->code;   // SERVICE_UNKNOWN, NAME_HAS_NO_OWNER, NO_SERVER, NO_NETWORK, DISCONNECTED, SPAWN_*
            return c == 2 || c == 3 || c == 11 || c == 13 || c == 15 || (c >= 23 && c <= 31);
        }
        static secret_api& get() {
            static secret_api a = [] {
                secret_api s; s.sc.name = "org.forge.Secret"; s.sc.attributes[0] = {"service", 0}; s.sc.attributes[1] = {"account", 0};
                s.h = ::dlopen("libsecret-1.so.0", RTLD_LAZY); if (!s.h) return s;
                s.store = reinterpret_cast<decltype(s.store)>(::dlsym(s.h, "secret_password_store_sync"));
                s.lookup = reinterpret_cast<decltype(s.lookup)>(::dlsym(s.h, "secret_password_lookup_sync"));
                s.clear = reinterpret_cast<decltype(s.clear)>(::dlsym(s.h, "secret_password_clear_sync"));
                s.free_pw = reinterpret_cast<decltype(s.free_pw)>(::dlsym(s.h, "secret_password_free"));
                if (void* g = ::dlopen("libglib-2.0.so.0", RTLD_LAZY)) {
                    s.clear_error = reinterpret_cast<decltype(s.clear_error)>(::dlsym(g, "g_clear_error"));
                    s.quark = reinterpret_cast<decltype(s.quark)>(::dlsym(g, "g_quark_from_string"));
                }
                s.ok = s.store && s.lookup && s.clear && s.free_pw && s.clear_error; return s;
            }();
            return a;
        }
    };
    // libsecret when it loads, a session bus exists and a provider answers a probe lookup; decided once per process.
    static bool use_libsecret_() {
        static const bool v = [] {
            auto& s = secret_api::get(); if (!s.ok) return false;
            auto rt = getenv("XDG_RUNTIME_DIR");
            if (getenv("DBUS_SESSION_BUS_ADDRESS").empty() && (rt.empty() || !exists(rt + "/bus"))) return false;
            void* err = nullptr; char* r = s.lookup(s.schema(), nullptr, &err, "service", "forge-keystore-probe", "account", "", nullptr);
            if (r) s.free_pw(r);
            if (!err) return true;
            bool gone = s.absent(err); s.clear_error(&err); return !gone;
        }();
        return v;
    }
#endif
#if defined(__APPLE__)
    struct sec_api {
        void* h = nullptr; int (*add)(const void*, const void**) = nullptr; int (*copy)(const void*, const void**) = nullptr; int (*del)(const void*) = nullptr;
        int (*update)(const void*, const void*) = nullptr;
        const void *kClass, *kGeneric, *kService, *kAccount, *kValueData, *kReturnData, *kMatchLimit, *kMatchOne, *kAccessible, *kAfterFirstUnlockThisDevice;
        static sec_api& get() {
            static sec_api s = [] { sec_api x{}; x.h = ::dlopen("/System/Library/Frameworks/Security.framework/Security", RTLD_LAZY); if (!x.h) return x;
                x.add = reinterpret_cast<decltype(x.add)>(::dlsym(x.h, "SecItemAdd")); x.copy = reinterpret_cast<decltype(x.copy)>(::dlsym(x.h, "SecItemCopyMatching"));
                x.del = reinterpret_cast<decltype(x.del)>(::dlsym(x.h, "SecItemDelete")); x.update = reinterpret_cast<decltype(x.update)>(::dlsym(x.h, "SecItemUpdate"));
                x.kClass = detail::framework_const(x.h, "kSecClass"); x.kGeneric = detail::framework_const(x.h, "kSecClassGenericPassword");
                x.kService = detail::framework_const(x.h, "kSecAttrService"); x.kAccount = detail::framework_const(x.h, "kSecAttrAccount");
                x.kValueData = detail::framework_const(x.h, "kSecValueData"); x.kReturnData = detail::framework_const(x.h, "kSecReturnData");
                x.kMatchLimit = detail::framework_const(x.h, "kSecMatchLimit"); x.kMatchOne = detail::framework_const(x.h, "kSecMatchLimitOne");
                x.kAccessible = detail::framework_const(x.h, "kSecAttrAccessible");
                x.kAfterFirstUnlockThisDevice = detail::framework_const(x.h, "kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly");
                return x; }();
            return s;
        }
        bool ok() const { return add && copy && del && update && kClass && detail::cf_api::get().ok; }
    };
    static const void* kc_query_(const std::string& s, const std::string& a, const void* extra_k[], const void* extra_v[], int nextra, std::vector<const void*>& owned) {
        auto& cf = detail::cf_api::get(); auto& k = sec_api::get();
        auto ss = cf.str(s.c_str()), as = cf.str(a.c_str()); owned.push_back(ss); owned.push_back(as);
        const void* keys[8] = {k.kClass, k.kService, k.kAccount}; const void* vals[8] = {k.kGeneric, ss, as};
        for (int i = 0; i < nextra; ++i) { keys[3 + i] = extra_k[i]; vals[3 + i] = extra_v[i]; }
        return cf.CFDictionaryCreate(nullptr, keys, vals, 3 + nextra, cf.kCFTypeDictionaryKeyCallBacks, cf.kCFTypeDictionaryValueCallBacks);
    }
    // Update in place (the old secret survives a failure); add only when the item does not exist yet.
    static status kc_put_(const std::string& s, const std::string& a, const std::string& v) {
        auto& k = sec_api::get(); if (!k.ok()) return {{}, unsupported("Security.framework")};
        auto& cf = detail::cf_api::get(); std::vector<const void*> owned;
        auto data = cf.CFDataCreate(nullptr, reinterpret_cast<const std::uint8_t*>(v.data()), detail::CFIndex(v.size())); owned.push_back(data);
        const void* uk[] = {k.kValueData}; const void* uv[] = {data};
        auto attrs = cf.CFDictionaryCreate(nullptr, uk, uv, 1, cf.kCFTypeDictionaryKeyCallBacks, cf.kCFTypeDictionaryValueCallBacks); owned.push_back(attrs);
        auto q = kc_query_(s, a, nullptr, nullptr, 0, owned); int rc = k.update(q, attrs); cf.CFRelease(q);
        if (rc == -25300 /*errSecItemNotFound*/) {
            const void* ek[2] = {k.kValueData, k.kAccessible}; const void* ev[2] = {data, k.kAfterFirstUnlockThisDevice};
            auto aq = kc_query_(s, a, ek, ev, k.kAccessible && k.kAfterFirstUnlockThisDevice ? 2 : 1, owned); rc = k.add(aq, nullptr); cf.CFRelease(aq);
            for (auto o : owned) cf.CFRelease(o);
            if (rc) return {{}, {rc, "SecItemAdd failed (OSStatus " + std::to_string(rc) + ")"}}; return {};
        }
        for (auto o : owned) cf.CFRelease(o);
        if (rc) return {{}, {rc, "SecItemUpdate failed (OSStatus " + std::to_string(rc) + ")"}}; return {};
    }
    static result<std::string> kc_get_(const std::string& s, const std::string& a) {
        auto& k = sec_api::get(); if (!k.ok()) return {{}, unsupported("Security.framework")};
        auto& cf = detail::cf_api::get(); std::vector<const void*> owned;
        const void* ek[] = {k.kReturnData, k.kMatchLimit}; const void* ev[] = {cf.kCFBooleanTrue, k.kMatchOne};
        auto q = kc_query_(s, a, ek, ev, 2, owned); const void* out = nullptr; int rc = k.copy(q, &out); cf.CFRelease(q); for (auto o : owned) cf.CFRelease(o);
        if (rc == -25300 || (!rc && !out)) return {{}, {2, "keystore: no such secret"}};
        if (rc) return {{}, {rc, "SecItemCopyMatching failed (OSStatus " + std::to_string(rc) + ")"}};
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
#undef FORGE_OS_KS_LIBSECRET_
