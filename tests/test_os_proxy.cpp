// forge::os::net proxy resolution (proxy.hpp): GNOME's dconf database and KDE's kioslaverc read directly, static
// rules, and PAC scripts evaluated only where the OS has an engine (the result says so otherwise).
#include <forge/os/proxy.hpp>
#include <forge/os/fs.hpp>
#include <atomic>
#include <thread>
#include "check.hpp"
using namespace forge;
namespace net = forge::os::net;
namespace {
// A real dconf database: written by `gsettings set org.gnome.system.proxy ...` through the dconf backend (GLib 2.80).
const char* dconf_user_hex =
    "4756617269616e74000000000000000018000000e0010000000000281000000000000000020000000200000002000000020000000300000003000000"
    "0400000004000000080000000b0000000b0000000e0000000e0000000f00000010000000609b294b0c000000e001000004007600e8010000ee010000"
    "1050edff0b000000ee01000004007600f8010000fe010000d4b50200fffffffffe01000001004c00000200000402000006a9fc900400000004020000"
    "04007600080200001c02000028e6eae60a0000001c02000005004c00240200002c020000e837d592020000002c02000007004c003402000038020000"
    "48906cf90a000000380200000e007600480200004b020000886fdd190a0000004b0200000e00760060020000840200007938254b0c00000084020000"
    "04007600880200009202000029ede8ff0b000000920200000400760098020000ac0200001922f93105000000ac02000006004c00b4020000d0020000"
    "cb908cdd0a000000d002000006004c00d8020000e00200001bb447c40a000000e002000006004c00e8020000f0020000db583d4e0a000000f0020000"
    "0c007600000300002e030000ed0b0191040000002e03000004007600380300003e0300005e31bcef0a0000003e030000040076004803000051030000"
    "706f727400000000fb2000000069706f72740000000000003804000000692f0005000000686f737470726f78792e6578616d706c652e636f6d000073"
    "687474702f000000030000000e00000073797374656d2f000a0000007573652d73616d652d70726f787900000000626175746f636f6e6669672d7572"
    "6c00000000000000687474703a2f2f777061642e6578616d706c652e636f6d2f70726f78792e706163000073686f7374666538303a3a31000073686f"
    "73740000736f636b732e6578616d706c652e636f6d00007370726f78792f000007000000040000000c0000000d0000000f0000000b00000006000000"
    "736f636b732f0000090000000100000068747470732f0000080000000000000069676e6f72652d686f737473000000006c6f63616c686f7374002a2e"
    "696e7465726e616c2e6578616d706c650031302e302e302e302f38000a1d28006173706f7274000000000000380c000000696d6f6465000000000000"
    "6d616e75616c000073";
std::string unhex(const char* h) { std::string o; auto v = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; }; for (; h[0] && h[1]; h += 2) o.push_back(char(v(h[0]) * 16 + v(h[1]))); return o; }
struct env_guard {   // the proxy variables of the machine running the tests must not leak into them
    std::vector<std::pair<std::string, std::string>> saved;
    env_guard() { for (const char* k : {"http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY", "XDG_CONFIG_HOME", "XDG_CURRENT_DESKTOP"}) if (os::has_env(k)) { saved.push_back({k, os::getenv(k)}); os::unsetenv(k); } }
    ~env_guard() { for (auto& [k, v] : saved) os::setenv(k, v); }
};
}

TEST(proxy_dconf_database) {
    const std::string db = unhex(dconf_user_hex); CHECK(db.size() > 100); CHECK(db.compare(0, 8, "GVariant") == 0);
    auto kv = net::detail_net::gvdb_values(db, "/system/proxy/");
    for (auto& [k, v] : kv) { std::string one = v; for (auto& c : one) if (c == '\n') c = '|'; std::printf("  %s = %s\n", k.c_str(), one.c_str()); }
    CHECK(kv["mode"] == "manual"); CHECK(kv["http/host"] == "proxy.example.com"); CHECK(kv["http/port"] == "3128"); CHECK(kv["https/host"] == "fe80::1");
    CHECK(kv["socks/port"] == "1080"); CHECK(kv["autoconfig-url"] == "http://wpad.example.com/proxy.pac"); CHECK(kv["use-same-proxy"] == "false");
    CHECK(kv["ignore-hosts"] == "localhost\n*.internal.example\n10.0.0.0/8");
    CHECK(net::detail_net::gvdb_values(db, "/nothing/here/").empty());
    auto p = net::detail_net::parse_dconf_proxy(db);
    CHECK(p.source == "dconf"); CHECK(p.http == "http://proxy.example.com:3128"); CHECK(p.https == "http://[fe80::1]:8443"); CHECK(p.socks == "socks5://socks.example.com:1080");
    CHECK(p.pac_url.empty()); CHECK(p.bypass.size() == 3 && p.bypass[1] == "*.internal.example");
    // damaged files: nothing is read out of bounds (ASan / UBSan watch), and nothing is invented
    for (std::size_t n = 0; n < db.size(); n += 7) (void)net::detail_net::gvdb_values(db.substr(0, n), "/");
    for (std::size_t i = 0; i < db.size(); ++i) for (int d : {1, 0x80, 0xff}) { std::string c = db; c[i] = char(c[i] ^ d); (void)net::detail_net::parse_dconf_proxy(c); }
    CHECK(net::detail_net::gvdb_values("", "/").empty()); CHECK(net::detail_net::gvdb_values(std::string(4096, '\xff'), "/").empty());
    CHECK(!net::detail_net::parse_dconf_proxy("GVariant").any());
}

TEST(proxy_kioslaverc) {
    auto p = net::detail_net::parse_kioslaverc(
        "[$Version]\nupdate_info=kioslave.upd\n\n[Proxy Settings]\nNoProxyFor=localhost, .corp.example,10.0.0.0/8\nProxy Config Script=\nProxyType=1\nReversedException=false\n"
        "ftpProxy=\nhttpProxy=http://proxy.kde.example 8080\nhttpsProxy[$e]=proxy2.kde.example:8443\nsocksProxy=socks://s.kde.example 1080\n\n[Other]\nhttpProxy=wrong\n");
    CHECK(p.source == "kioslaverc"); CHECK(p.http == "http://proxy.kde.example:8080"); CHECK(p.https == "http://proxy2.kde.example:8443"); CHECK(p.socks == "socks5://s.kde.example:1080");
    CHECK(p.bypass.size() == 3 && p.bypass[1] == ".corp.example");
    auto pac = net::detail_net::parse_kioslaverc("[Proxy Settings]\nProxyType=2\nProxy Config Script=http://wpad.kde.example/proxy.pac\n");
    CHECK(pac.pac_url == "http://wpad.kde.example/proxy.pac"); CHECK(!pac.auto_detect); CHECK(pac.http.empty());
    CHECK(net::detail_net::parse_kioslaverc("[Proxy Settings]\nProxyType=3\n").auto_detect);
    CHECK(!net::detail_net::parse_kioslaverc("[Proxy Settings]\nProxyType=0\nhttpProxy=http://x 1\n").any());
    CHECK(!net::detail_net::parse_kioslaverc("[Proxy Settings]\nProxyType=4\n").any());   // "use the environment": proxy_from_env has it
    CHECK(!net::detail_net::parse_kioslaverc("").any()); CHECK(!net::detail_net::parse_kioslaverc("ProxyType=1\nhttpProxy=http://x 1\n").any());   // outside the group
    auto rev = net::detail_net::parse_kioslaverc("[Proxy Settings]\nProxyType=1\nhttpProxy=h 1\nNoProxyFor=a.example\nReversedException=true\n");
    CHECK(rev.http == "http://h:1"); CHECK(rev.bypass.empty());   // "only use the proxy for these": not expressible as a bypass list
}

TEST(proxy_desktop_files) {
    env_guard env;
    auto d = os::fs::make_temp_dir("forge-proxy-"); CHECK(d); if (!d) return;
    os::setenv("XDG_CONFIG_HOME", *d);
#if defined(FORGE_OS_LINUX) || defined(FORGE_OS_FREEBSD)
    os::mkdirs(*d + "/dconf"); CHECK(os::write_file(*d + "/dconf/user", unhex(dconf_user_hex)));
    CHECK(os::write_file(*d + "/kioslaverc", "[Proxy Settings]\nProxyType=1\nhttpProxy=http://kde.example 3128\n"));
    auto g = net::desktop_proxy(); CHECK_MSG(g.source == "dconf", "%s", g.source.c_str()); CHECK(g.http == "http://proxy.example.com:3128");
    os::setenv("XDG_CURRENT_DESKTOP", "KDE");
    auto k = net::desktop_proxy(); CHECK(k.source == "kioslaverc"); CHECK(k.http == "http://kde.example:3128");
    os::remove(*d + "/kioslaverc");
    CHECK(net::desktop_proxy().source == "dconf");   // KDE session without KDE settings: the other file still counts
    os::unsetenv("XDG_CURRENT_DESKTOP");
#endif
    os::setenv("http_proxy", "http://env.example:1");   // the environment wins everywhere
    auto e = net::desktop_proxy(); CHECK(e.source == "env"); CHECK(e.http == "http://env.example:1");
    os::unsetenv("http_proxy"); os::remove_all(*d);
}

TEST(proxy_resolution_static_and_pac) {
    net::proxy_settings none;
    { auto r = net::resolve_proxy("https://example.com/", none); CHECK(r.direct()); CHECK(r.proxies.size() == 1 && r.proxies[0] == "direct"); CHECK(!r.pac_url_present); CHECK(!r.pac_evaluated); }
    net::proxy_settings fixed; fixed.http = "http://p.example:8080"; fixed.socks = "socks5://s.example:1080"; fixed.bypass = {"*.internal", "localhost"}; fixed.source = "test";
    { auto r = net::resolve_proxy("http://example.com/x", fixed); CHECK(!r.direct()); CHECK(r.proxies[0] == "http://p.example:8080"); CHECK(r.source == "test"); }
    { auto r = net::resolve_proxy("https://example.com/x", fixed); CHECK(r.proxies[0] == "socks5://s.example:1080"); }   // no https proxy: SOCKS carries it
    { auto r = net::resolve_proxy("http://db.internal/", fixed); CHECK(r.direct()); }
    // auto-detect alone: never performed, said so
    net::proxy_settings wpad; wpad.auto_detect = true;
    { auto r = net::resolve_proxy("http://example.com/", wpad); CHECK(r.pac_url_present); CHECK(!r.pac_evaluated); CHECK(r.proxies.empty()); CHECK(r.direct()); CHECK(r.note.find("WPAD") != std::string::npos); }
    // A PAC script served from this process: evaluated where the OS has an engine, reported as not evaluated elsewhere.
    auto ls = os::socket::listen("127.0.0.1", 0); CHECK(ls); if (!ls) return;
    const std::string script = "function FindProxyForURL(url, host) { if (host == \"direct.example\") return \"DIRECT\"; return \"PROXY p.example:8080; SOCKS s.example:1080; DIRECT\"; }\n";
    std::atomic<bool> stop{false}; std::atomic<int> served{0};
    std::thread srv([&] {
        while (!stop) {
            if (!ls->wait_readable(100)) continue;
            auto c = ls->accept(); if (!c) continue;
            c->set_timeout_ms(2000); char buf[2048]; std::string rq;
            while (rq.find("\r\n\r\n") == std::string::npos) { long n = c->recv(buf, sizeof buf); if (n <= 0) break; rq.append(buf, std::size_t(n)); }
            std::string rs = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ns-proxy-autoconfig\r\nContent-Length: " + std::to_string(script.size()) + "\r\nConnection: close\r\n\r\n" + script;
            c->send_all(rs.data(), rs.size()); ++served;
        }
    });
    net::proxy_settings pac; pac.pac_url = "http://127.0.0.1:" + std::to_string(ls->local_port()) + "/proxy.pac"; pac.bypass = {"skip.example"}; pac.source = "test";
    auto t0 = os::monotonic_ns();
    auto r = net::resolve_proxy("http://www.example.com/page", pac);
    auto byp = net::resolve_proxy("http://skip.example/", pac);
    stop = true; srv.join();
    CHECK(os::monotonic_ns() - t0 < 40'000'000'000ull);
    CHECK(r.pac_url_present); CHECK(r.pac_url == pac.pac_url);
    CHECK(byp.direct()); CHECK(byp.proxies.size() == 1);   // the bypass list is applied before any script
    std::printf("  PAC: evaluated %d, %zu proxies%s%s (script fetched %d time(s))\n", int(r.pac_evaluated), r.proxies.size(), r.note.empty() ? "" : ", note: ", r.note.c_str(), served.load());
#if defined(_WIN32) || defined(__APPLE__)
    if (r.pac_evaluated) {
        CHECK(r.proxies.size() >= 1); if (!r.proxies.empty()) CHECK_MSG(r.proxies[0] == "http://p.example:8080", "%s", r.proxies[0].c_str());
        CHECK(r.source == "winhttp-pac" || r.source == "cfnetwork-pac");
    } else CHECK(!r.note.empty());   // e.g. wine without a script engine: said so, no guess
#else
    CHECK(!r.pac_evaluated); CHECK(r.proxies.empty()); CHECK(r.note.find("not evaluated") != std::string::npos); CHECK(served == 0);   // the script is not even fetched
#endif
    CHECK(!net::resolve_proxy("not a url", pac).pac_evaluated);
}
