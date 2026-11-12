#pragma once
// forge::os::net proxy resolution beyond the static settings of netinfo.hpp. Guide: docs/os-wrapper.md.
//   desktop_proxy()   Linux / FreeBSD without helper programs: GNOME's dconf database read directly
//                     (~/.config/dconf/user, GVDB), KDE's kioslaverc, then what system_proxy() finds (environment,
//                     gsettings). Elsewhere it is system_proxy().
//   resolve_proxy()   which proxies to try for one URL. Static settings are applied directly. A PAC script is
//                     JavaScript: FORGE embeds no engine, so it is evaluated only where the OS does it -
//                     Windows WinHttpGetProxyForUrl, macOS / iOS CFNetworkExecuteProxyAutoConfigurationURL. On
//                     Linux, FreeBSD and Android the result says pac_url_present with pac_evaluated == false and
//                     no proxies: the caller decides (connect directly, or ask the user). Auto-detection (WPAD
//                     over DHCP / DNS) is never performed.
#include <forge/os/netinfo.hpp>

namespace forge::os::net {

struct proxy_resolution {
    std::vector<std::string> proxies;   // in the order to try: "http://host:port", "socks5://host:port", or "direct"
    bool pac_url_present = false;       // the settings name a PAC script (or ask for auto-detection)
    bool pac_evaluated = false;         // ... and the OS evaluated it for this URL
    std::string pac_url, source;        // source: the settings' source, or "winhttp-pac" / "cfnetwork-pac"
    std::string note;                   // why a PAC script was not evaluated, in words
    bool direct() const { return proxies.empty() || proxies.front() == "direct"; }
};

namespace detail_net {
// ---- GVDB (the dconf database format): every value under `prefix`, rendered as text (s as is, i / u decimal,
// b "true" / "false", as joined with '\n'). The file is read defensively: all offsets are checked.
inline std::map<std::string, std::string> gvdb_values(const std::string& db, std::string_view prefix) {
    std::map<std::string, std::string> out;
    const auto* p = reinterpret_cast<const unsigned char*>(db.data()); const std::size_t n = db.size();
    auto u32 = [&](std::size_t at) -> std::uint32_t { return at + 4 <= n ? std::uint32_t(p[at]) | std::uint32_t(p[at + 1]) << 8 | std::uint32_t(p[at + 2]) << 16 | std::uint32_t(p[at + 3]) << 24 : 0; };
    if (n < 24 || std::memcmp(p, "GVariant", 8) != 0 || u32(8) != 0) return out;
    const std::size_t rs = u32(16), re = u32(20); if (rs > re || re > n || re - rs < 8) return out;
    const std::size_t blooms = u32(rs) & ((1u << 27) - 1), buckets = u32(rs + 4);
    const std::uint64_t items_at64 = std::uint64_t(rs) + 8 + 4ull * blooms + 4ull * buckets; if (items_at64 > re) return out;
    const std::size_t items_at = std::size_t(items_at64), count = (re - items_at) / 24;
    auto key_of = [&](std::size_t i) -> std::string {   // the full path: parents' segments first
        std::string k; std::size_t hops = 0;
        while (i < count && hops++ < 64) {
            std::size_t at = items_at + i * 24, ks = u32(at + 8), kl = std::size_t(p[at + 12]) | std::size_t(p[at + 13]) << 8;
            if (ks > n || kl > n - ks) return {};
            k.insert(0, db, ks, kl);
            std::uint32_t parent = u32(at + 4); if (parent == 0xffffffffu) return k;
            i = parent;
        }
        return {};
    };
    for (std::size_t i = 0; i < count && out.size() < 4096; ++i) {
        std::size_t at = items_at + i * 24; if (p[at + 14] != 'v') continue;
        std::string key = key_of(i); if (key.size() < prefix.size() || key.compare(0, prefix.size(), prefix) != 0) continue;
        std::size_t vs = u32(at + 16), ve = u32(at + 20); if (vs > ve || ve > n) continue;
        std::string_view v(db.data() + vs, ve - vs);       // a serialized variant: payload, NUL, type string
        auto z = v.rfind('\0'); if (z == std::string_view::npos) continue;
        std::string_view type = v.substr(z + 1), body = v.substr(0, z); std::string text;
        if (type == "s") { text.assign(body.substr(0, body.find('\0'))); }
        else if (type == "i" && body.size() >= 4) { std::int32_t x; std::memcpy(&x, body.data(), 4); text = std::to_string(x); }
        else if (type == "u" && body.size() >= 4) { std::uint32_t x; std::memcpy(&x, body.data(), 4); text = std::to_string(x); }
        else if (type == "b" && !body.empty()) text = body[0] ? "true" : "false";
        else if (type == "as") {   // strings, then one end offset per string (1, 2 or 4 bytes wide by container size)
            if (!body.empty()) {
                const std::size_t w = body.size() <= 0xff ? 1 : body.size() <= 0xffff ? 2 : 4;
                auto off = [&](std::size_t pos) { std::size_t x = 0; for (std::size_t b = 0; b < w; ++b) x |= std::size_t(static_cast<unsigned char>(body[pos + b])) << (8 * b); return x; };
                if (body.size() < w) continue;
                const std::size_t last = off(body.size() - w); if (last > body.size() - w) continue;
                const std::size_t cnt = (body.size() - last) / w; std::size_t start = 0; bool ok = true;
                for (std::size_t k = 0; k < cnt && ok; ++k) {
                    std::size_t end = off(last + k * w); if (end < start || end > last) { ok = false; break; }
                    std::string_view s = body.substr(start, end - start); s = s.substr(0, s.find('\0'));
                    if (k) text += '\n'; text.append(s); start = end;
                }
                if (!ok) continue;
            }
        } else continue;
        out[key.substr(prefix.size())] = std::move(text);
    }
    return out;
}
// GNOME proxy settings from a dconf database image (keys under /system/proxy/).
inline proxy_settings parse_dconf_proxy(const std::string& db) {
    proxy_settings p; auto kv = gvdb_values(db, "/system/proxy/");
    auto get = [&](const char* k) { auto it = kv.find(k); return it == kv.end() ? std::string() : it->second; };
    const std::string mode = get("mode");
    auto hp = [&](const char* sub, const char* scheme) -> std::string {
        std::string h = get((std::string(sub) + "/host").c_str()), port = get((std::string(sub) + "/port").c_str());
        if (h.empty()) return {};
        return std::string(scheme) + "://" + (h.find(':') != std::string::npos && h[0] != '[' ? "[" + h + "]" : h) + (port.empty() || port == "0" ? "" : ":" + port);
    };
    if (mode == "manual") { p.http = hp("http", "http"); p.https = hp("https", "http"); p.socks = hp("socks", "socks5"); }
    else if (mode == "auto") { p.pac_url = get("autoconfig-url"); p.auto_detect = p.pac_url.empty(); }
    else return p;
    if (kv.count("ignore-hosts")) p.bypass = split_list(get("ignore-hosts"), "\n");
    else p.bypass = {"localhost", "127.0.0.0/8", "::1"};   // the schema's default
    p.source = "dconf"; return p;
}
// KDE: ~/.config/kioslaverc, group [Proxy Settings]. ProxyType 0 none, 1 manual, 2 PAC URL, 3 auto-detect, 4 environment.
inline proxy_settings parse_kioslaverc(const std::string& text) {
    proxy_settings p; std::map<std::string, std::string> kv; std::istringstream in(text); std::string line; bool in_group = false;
    while (std::getline(in, line)) {
        line = detail::trim(line); if (line.empty() || line[0] == '#') continue;
        if (line[0] == '[') { in_group = line == "[Proxy Settings]"; continue; }
        if (!in_group) continue;
        auto eq = line.find('='); if (eq == std::string::npos) continue;
        std::string k = detail::trim(line.substr(0, eq)); auto br = k.find('['); if (br != std::string::npos) k = k.substr(0, br);   // "key[$e]"
        kv[k] = detail::trim(line.substr(eq + 1));
    }
    const std::string type = kv["ProxyType"];
    auto one = [&](const char* k, const char* scheme) -> std::string {   // "http://host port" or "http://host:port" or "host port"
        std::string v = kv[k]; if (v.empty()) return {};
        auto sp = v.find(' '); if (sp != std::string::npos) { std::string port = detail::trim(v.substr(sp + 1)); v = v.substr(0, sp); if (!port.empty() && port != "0") v += ":" + port; }
        auto ss = v.find("://");
        if (ss != std::string::npos) { std::string sc = lower(v.substr(0, ss)); if (std::string(scheme) == "socks5" && (sc == "socks" || sc == "socks5")) return "socks5" + v.substr(ss); if (std::string(scheme) == "http") return "http" + v.substr(ss); return v; }
        return std::string(scheme) + "://" + v;
    };
    if (type == "1") { p.http = one("httpProxy", "http"); p.https = one("httpsProxy", "http"); p.socks = one("socksProxy", "socks5"); }
    else if (type == "2") { p.pac_url = kv["Proxy Config Script"]; p.auto_detect = p.pac_url.empty(); }
    else if (type == "3") p.auto_detect = true;
    else return p;   // 0: none; 4: the environment, which proxy_from_env() already covers
    if (kv["ReversedException"] != "true") p.bypass = split_list(kv["NoProxyFor"], ", \t");
    if (p.any()) p.source = "kioslaverc";
    return p;
}
inline std::string pac_token(std::string t) {   // one WinHTTP proxy-list entry -> URL
    auto eq = t.find('=');
    if (eq != std::string::npos) { std::string k = lower(t.substr(0, eq)); t = t.substr(eq + 1); if (k == "socks") return with_scheme(t, "socks5"); }
    return with_scheme(t, "http");
}
}

// The desktop's proxy settings, read without helper programs where a file holds them (see the top of this file).
inline proxy_settings desktop_proxy() {
    proxy_settings p = proxy_from_env(); if (p.any()) return p;
#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    {
        std::function<proxy_settings()> h; { std::lock_guard<std::mutex> g(detail_net::proxy_hook_mutex()); h = detail_net::proxy_hook(); }
        if (h) return system_proxy();
    }
    std::string cfg = getenv("XDG_CONFIG_HOME"); if (cfg.empty()) cfg = join(home_dir(), ".config");
    const std::string desktop = detail_net::lower(getenv("XDG_CURRENT_DESKTOP"));
    auto kde = [&] { auto t = read_file(join(cfg, "kioslaverc")); return t ? detail_net::parse_kioslaverc(*t) : proxy_settings{}; };
    auto gnome = [&] { auto t = read_file(join(join(cfg, "dconf"), "user")); return t ? detail_net::parse_dconf_proxy(*t) : proxy_settings{}; };
    const bool kde_first = desktop.find("kde") != std::string::npos;
    if (auto a = kde_first ? kde() : gnome(); a.any()) return a;
    if (auto b = kde_first ? gnome() : kde(); b.any()) return b;
#endif
    return system_proxy();
}

// The proxies to try for `target` under settings `p` (desktop_proxy() by default).
inline proxy_resolution resolve_proxy(const std::string& target, const proxy_settings& p) {
    proxy_resolution r; r.source = p.source; r.pac_url = p.pac_url; r.pac_url_present = !p.pac_url.empty() || p.auto_detect;
    auto fixed = [&] { std::string s = proxy_for(target, p); r.proxies.push_back(s.empty() ? std::string("direct") : s); };
    if (!r.pac_url_present) { fixed(); return r; }
    if (p.pac_url.empty()) { r.note = "auto-detection (WPAD) is not performed"; if (!p.http.empty() || !p.https.empty() || !p.socks.empty()) fixed(); return r; }
    auto u = parse_url(target); if (!u) { r.note = "target is not a URL"; return r; }
    if (proxy_bypassed(u->host, p.bypass)) { r.proxies.push_back("direct"); return r; }
#if defined(FORGE_OS_WINDOWS)
    struct options { DWORD flags, detect; LPCWSTR url; LPVOID reserved; DWORD reserved2; BOOL logon; };   // WINHTTP_AUTOPROXY_OPTIONS
    struct info { DWORD access; LPWSTR proxy, bypass; };                                              // WINHTTP_PROXY_INFO
    static HMODULE h = LoadLibraryExW(L"winhttp.dll", nullptr, 0x800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/);
    auto open = h ? reinterpret_cast<void*(WINAPI*)(LPCWSTR, DWORD, LPCWSTR, LPCWSTR, DWORD)>(reinterpret_cast<void*>(GetProcAddress(h, "WinHttpOpen"))) : nullptr;
    auto get = h ? reinterpret_cast<BOOL(WINAPI*)(void*, LPCWSTR, options*, info*)>(reinterpret_cast<void*>(GetProcAddress(h, "WinHttpGetProxyForUrl"))) : nullptr;
    auto close = h ? reinterpret_cast<BOOL(WINAPI*)(void*)>(reinterpret_cast<void*>(GetProcAddress(h, "WinHttpCloseHandle"))) : nullptr;
    if (!open || !get || !close) { r.note = "winhttp.dll is not available"; return r; }
    void* s = open(L"forge", 1 /*WINHTTP_ACCESS_TYPE_NO_PROXY*/, nullptr, nullptr, 0);
    if (!s) { r.note = "WinHttpOpen: " + error_text(int(GetLastError())); return r; }
    const std::wstring wurl = widen(target), wpac = widen(p.pac_url);
    options o{}; o.flags = 0x2 /*WINHTTP_AUTOPROXY_CONFIG_URL*/; o.url = wpac.c_str(); o.logon = FALSE;
    info pi{}; BOOL ok = get(s, wurl.c_str(), &o, &pi);
    if (!ok && GetLastError() == 12015 /*ERROR_WINHTTP_LOGIN_FAILURE*/) { o.logon = TRUE; ok = get(s, wurl.c_str(), &o, &pi); }
    if (!ok) { r.note = "WinHttpGetProxyForUrl: error " + std::to_string(GetLastError()); close(s); return r; }
    r.pac_evaluated = true; r.source = "winhttp-pac";
    if (pi.access == 3 /*WINHTTP_ACCESS_TYPE_NAMED_PROXY*/ && pi.proxy) for (auto& t : detail_net::split_list(narrow(pi.proxy), "; \t")) r.proxies.push_back(detail_net::pac_token(t));
    if (r.proxies.empty()) r.proxies.push_back("direct");
    if (pi.proxy) GlobalFree(pi.proxy);
    if (pi.bypass) GlobalFree(pi.bypass);
    close(s);
#elif defined(__APPLE__)
    auto& cf = detail::cf_api::get();
    static void* cn = ::dlopen("/System/Library/Frameworks/CFNetwork.framework/CFNetwork", RTLD_LAZY);
    static void* cfh = ::dlopen("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation", RTLD_LAZY);
    struct client_context { long version; void* info; void* retain; void* release; void* describe; };   // CFStreamClientContext
    using result_cb = void (*)(void*, const void*, const void*);
    auto exec = cn ? reinterpret_cast<const void* (*)(const void*, const void*, result_cb, client_context*)>(::dlsym(cn, "CFNetworkExecuteProxyAutoConfigurationURL")) : nullptr;
    auto mkurl = cfh ? reinterpret_cast<const void* (*)(const void*, const void*, const void*)>(::dlsym(cfh, "CFURLCreateWithString")) : nullptr;
    auto rl_current = cfh ? reinterpret_cast<void* (*)()>(::dlsym(cfh, "CFRunLoopGetCurrent")) : nullptr;
    auto rl_add = cfh ? reinterpret_cast<void (*)(void*, const void*, const void*)>(::dlsym(cfh, "CFRunLoopAddSource")) : nullptr;
    auto rl_remove = cfh ? reinterpret_cast<void (*)(void*, const void*, const void*)>(::dlsym(cfh, "CFRunLoopRemoveSource")) : nullptr;
    auto rl_invalidate = cfh ? reinterpret_cast<void (*)(const void*)>(::dlsym(cfh, "CFRunLoopSourceInvalidate")) : nullptr;
    auto rl_run = cfh ? reinterpret_cast<std::int32_t (*)(const void*, double, unsigned char)>(::dlsym(cfh, "CFRunLoopRunInMode")) : nullptr;
    static auto rl_stop = cfh ? reinterpret_cast<void (*)(void*)>(::dlsym(cfh, "CFRunLoopStop")) : nullptr;
    auto konst = [&](const char* name) -> const void* { auto pp = cn ? static_cast<const void**>(::dlsym(cn, name)) : nullptr; return pp ? *pp : nullptr; };
    if (!cf.ok || !exec || !mkurl || !rl_current || !rl_add || !rl_remove || !rl_run || !rl_stop || !cf.CFArrayGetCount || !cf.CFArrayGetValueAtIndex || !cf.CFDictionaryGetValue) { r.note = "CFNetwork is not available"; return r; }
    struct state { proxy_resolution* r; const void *k_type, *k_host, *k_port; bool done = false; void* loop = nullptr; } st{&r, konst("kCFProxyTypeKey"), konst("kCFProxyHostNameKey"), konst("kCFProxyPortNumberKey")};
    if (!st.k_type || !st.k_host || !st.k_port) { r.note = "CFNetwork proxy keys are not available"; return r; }
    auto pac_s = cf.str(p.pac_url.c_str()), tgt_s = cf.str(target.c_str());
    const void* pac_u = pac_s ? mkurl(nullptr, pac_s, nullptr) : nullptr; const void* tgt_u = tgt_s ? mkurl(nullptr, tgt_s, nullptr) : nullptr;
    if (pac_s) cf.CFRelease(pac_s);
    if (tgt_s) cf.CFRelease(tgt_s);
    if (!pac_u || !tgt_u) { if (pac_u) cf.CFRelease(pac_u); if (tgt_u) cf.CFRelease(tgt_u); r.note = "the PAC URL or the target is not a valid URL"; return r; }
    st.loop = rl_current();
    client_context ctx{0, &st, nullptr, nullptr, nullptr};
    const void* src = exec(pac_u, tgt_u, [](void* client, const void* list, const void* err) {
        auto* s2 = static_cast<state*>(client); auto& c = detail::cf_api::get(); s2->done = true;
        if (!err && list) {
            s2->r->pac_evaluated = true; s2->r->source = "cfnetwork-pac";
            for (detail::CFIndex i = 0; i < c.CFArrayGetCount(list); ++i) {
                auto d = c.CFArrayGetValueAtIndex(list, i); if (!d) continue;
                std::string type = c.to_string(c.CFDictionaryGetValue(d, s2->k_type)), host = c.to_string(c.CFDictionaryGetValue(d, s2->k_host));
                long long port = c.to_ll(c.CFDictionaryGetValue(d, s2->k_port), 0);
                if (type.find("None") != std::string::npos || host.empty()) { s2->r->proxies.push_back("direct"); continue; }
                std::string hp = (host.find(':') != std::string::npos && host[0] != '[' ? "[" + host + "]" : host) + (port > 0 ? ":" + std::to_string(port) : "");
                s2->r->proxies.push_back((type.find("SOCKS") != std::string::npos ? "socks5://" : "http://") + hp);
            }
        } else s2->r->note = "the PAC script could not be fetched or run";
        if (rl_stop && s2->loop) rl_stop(s2->loop);
    }, &ctx);
    if (!src) { cf.CFRelease(pac_u); cf.CFRelease(tgt_u); r.note = "CFNetworkExecuteProxyAutoConfigurationURL failed"; return r; }
    auto mode = cf.str("forge.proxy.pac");   // a private mode: nothing else of this thread's run loop is serviced
    rl_add(st.loop, src, mode);
    for (int i = 0; i < 100 && !st.done; ++i) rl_run(mode, 0.1, 0);   // up to 10 s (the script is downloaded first)
    rl_remove(st.loop, src, mode);
    if (rl_invalidate) rl_invalidate(src);   // no callback after this point: `st` is about to go away
    cf.CFRelease(mode); cf.CFRelease(src); cf.CFRelease(pac_u); cf.CFRelease(tgt_u);
    if (!st.done) r.note = "the PAC script did not answer within 10 s";
    if (r.pac_evaluated && r.proxies.empty()) r.proxies.push_back("direct");
#else
    r.note = "pac_url present, not evaluated (no JavaScript engine; the OS offers none)";
#endif
    return r;
}
inline proxy_resolution resolve_proxy(const std::string& target) { return resolve_proxy(target, desktop_proxy()); }

} // namespace forge::os::net
