// forge::os phase 3: recursive/debounced file watch, net extensions, service control, registry.
#include <forge/os/file_watch_tree.hpp>
#include <forge/os/net.hpp>
#include <forge/os/service_control.hpp>
#include <forge/os/registry.hpp>
#include <thread>
#include "check.hpp"
using namespace forge;
namespace fs = forge::os::fs;

static bool has_event(const std::vector<os::watch_event>& ev, const std::string& suffix, os::watch_kind k) {
    for (auto& e : ev) if (e.kind == k && e.path.size() >= suffix.size() && e.path.compare(e.path.size() - suffix.size(), suffix.size(), suffix) == 0) return true;
    return false;
}
static void tree_watch_case(bool polling) {
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d;
    CHECK(fs::make_dirs(r + "/a/b"));
    os::tree_watch_opts o; o.force_polling = polling; o.debounce_ms = 60; o.ignore = {"*.tmp"};
    auto w = os::tree_watcher::open(r, o); CHECK(w);
    CHECK((*w)->watched_dirs() == 3);
    // A burst: new nested dir with a file, a write deep in the tree, an ignored file.
    os::sleep_ns(20'000'000);
    CHECK(fs::make_dirs(r + "/a/b/c"));
    CHECK(os::write_file(r + "/a/b/c/new.txt", "1"));
    CHECK(os::write_file(r + "/a/b/x.txt", "x"));
    CHECK(os::write_file(r + "/a/skip.tmp", "t"));
    std::vector<os::watch_event> ev;
    for (int i = 0; i < 20 && !(has_event(ev, "/c/new.txt", os::watch_kind::created) && has_event(ev, "/b/x.txt", os::watch_kind::created)); ++i) {
        auto more = (*w)->poll(500); ev.insert(ev.end(), more.begin(), more.end());
    }
    CHECK(has_event(ev, "/a/b/c", os::watch_kind::created));
    CHECK(has_event(ev, "/c/new.txt", os::watch_kind::created));
    CHECK(has_event(ev, "/b/x.txt", os::watch_kind::created));
    for (auto& e : ev) CHECK_MSG(e.path.find("skip.tmp") == std::string::npos, "ignored path reported: %s", e.path.c_str());
    // Each path once per poll (coalesced).
    ev.clear(); os::sleep_ns(30'000'000);
    CHECK(os::write_file(r + "/a/b/c/new.txt", "22")); CHECK(os::write_file(r + "/a/b/c/new.txt", "333"));
    for (int i = 0; i < 20 && ev.empty(); ++i) ev = (*w)->poll(500);
    CHECK(ev.size() == 1); CHECK(has_event(ev, "/c/new.txt", os::watch_kind::modified));
    // Removing a subtree reports the dir and its contents.
    CHECK(fs::remove_all(r + "/a/b/c"));
    ev.clear(); for (int i = 0; i < 20 && !has_event(ev, "/a/b/c", os::watch_kind::removed); ++i) { auto m = (*w)->poll(500); ev.insert(ev.end(), m.begin(), m.end()); }
    CHECK(has_event(ev, "/a/b/c", os::watch_kind::removed));
    // Created then deleted inside one debounce window: nothing.
    CHECK(os::write_file(r + "/flash", "f")); CHECK(fs::remove_all(r + "/flash"));
    auto q = (*w)->poll(polling ? 250 : 150); for (auto& e : q) CHECK_MSG(e.path.find("flash") == std::string::npos, "flash reported as %s", os::watch_kind_name(e.kind));
    fs::remove_all(r);
}
TEST(tree_watch_native) { tree_watch_case(false); }
TEST(tree_watch_polling) { tree_watch_case(true); }

TEST(url_parse) {
    auto u = os::net::parse_url("http://user@example.com:8080/a/b?x=1#frag"); CHECK(u);
    CHECK(u->host == "example.com" && u->port == 8080 && u->path == "/a/b" && u->query == "x=1");
    auto v = os::net::parse_url("https://[::1]/"); CHECK(v); CHECK(v->host == "::1" && v->port == 443);
    CHECK(!os::net::parse_url("nope")); CHECK(!os::net::parse_url("http:///x"));
}
TEST(resolve_and_interfaces) {
    auto r = os::net::resolve("localhost"); CHECK(r); CHECK(!r->empty());
    auto i = os::net::interfaces(); CHECK(i); bool lo = false, lo4 = false;
    for (auto& x : *i) { CHECK(!x.name.empty()); lo |= x.loopback; if (x.loopback) for (auto& a : x.addresses) lo4 |= a.ip == "127.0.0.1"; }
    CHECK(lo); CHECK(lo4);
}
TEST(http_client_against_local_server) {
    auto l = os::socket::listen("127.0.0.1", 0); CHECK(l); int port = l->local_port();
    std::thread srv([&] {
        for (int k = 0; k < 3; ++k) {
            auto c = l->accept(); if (!c) return;
            std::string req; char b[4096];
            while (req.find("\r\n\r\n") == std::string::npos) { long n = c->recv(b, sizeof b); if (n <= 0) break; req.append(b, std::size_t(n)); }
            std::string resp;
            if (req.rfind("GET /redirect", 0) == 0) resp = "HTTP/1.1 302 Found\r\nLocation: /chunked\r\nContent-Length: 0\r\n\r\n";
            else if (req.rfind("GET /chunked", 0) == 0) resp = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nX-A: 1\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
            else { auto bl = req.substr(req.find("\r\n\r\n") + 4); while (bl.size() < 3) { long n = c->recv(b, sizeof b); if (n <= 0) break; bl.append(b, std::size_t(n)); } resp = "HTTP/1.1 201 Created\r\nContent-Length: " + std::to_string(bl.size() + 5) + "\r\n\r\necho:" + bl; }
            c->send_all(resp.data(), resp.size());
        }
    });
    auto base = "http://127.0.0.1:" + std::to_string(port);
    auto r = os::net::http_get(base + "/redirect"); CHECK(r); CHECK(r->status == 200); CHECK(r->body == "hello world"); CHECK(r->headers["x-a"] == "1");
    os::net::http_request p; p.method = "POST"; p.url = base + "/post"; p.body = "abc";
    auto q = os::net::http(p); CHECK(q); CHECK(q->status == 201); CHECK(q->body == "echo:abc");
    srv.join();
    CHECK(os::net::port_open("127.0.0.1", port)); l->close(); CHECK(!os::net::port_open("127.0.0.1", port, 300));
    CHECK(os::is_unsupported(os::net::http_get("https://example.com/").err));
}
TEST(service_units) {
    os::service::service_spec s; s.name = "forge-test-svc"; s.description = "Forge test"; s.argv = {"/usr/bin/forge", "script", "my file.fg"};
    s.env["A"] = "1 2"; s.working_dir = "/tmp";
    auto u = os::service::render_systemd_unit(s);
    CHECK(u.find("ExecStart=/usr/bin/forge script \"my file.fg\"") != std::string::npos);
    CHECK(u.find("Environment=\"A=1 2\"") != std::string::npos); CHECK(u.find("Restart=on-failure") != std::string::npos);
    auto p = os::service::render_launchd_plist(s); CHECK(p.find("<string>my file.fg</string>") != std::string::npos);
    CHECK(!os::service::valid_name("bad name")); CHECK(!os::service::install({}));
    CHECK(os::service::validate(s));
    {   // injection: a newline must not start a new directive; install refuses control characters
        auto bad = s; bad.description = "x\nExecStartPre=/bin/evil"; CHECK(!os::service::validate(bad)); CHECK(!os::service::install(bad));
        auto t = os::service::render_systemd_unit(bad); CHECK(t.find("\nExecStartPre=") == std::string::npos);
        bad = s; bad.argv.push_back("a\nExecStartPre=/bin/evil"); CHECK(!os::service::validate(bad));
        t = os::service::render_systemd_unit(bad); CHECK(t.find("\nExecStartPre=") == std::string::npos); CHECK(t.find("\"a\\x0aExecStartPre=/bin/evil\"") != std::string::npos);
        bad = s; bad.env["B"] = "1\nExecStartPre=/bin/evil"; CHECK(!os::service::validate(bad)); CHECK(os::service::render_systemd_unit(bad).find("\nExecStartPre=") == std::string::npos);
        bad = s; bad.working_dir = "/tmp\nExecStartPre=/bin/evil"; CHECK(!os::service::validate(bad)); CHECK(os::service::render_systemd_unit(bad).find("\nExecStartPre=") == std::string::npos);
        bad = s; bad.env["A=B"] = "1"; CHECK(!os::service::validate(bad));
        bad = s; bad.argv[0] = "-/usr/bin/forge"; CHECK(!os::service::validate(bad));
    }
    {   // escaping per systemd.syntax / systemd.exec
        auto e = s; e.argv = {"/bin/sh", "-c", "echo $HOME 100%", ";", "x;y"}; e.env = {{"P", "$HOME 5%"}}; e.description = "50% done"; e.working_dir = "/srv/my dir";
        auto t = os::service::render_systemd_unit(e);
        CHECK(t.find("ExecStart=/bin/sh -c \"echo $$HOME 100%%\" \\; \"x;y\"\n") != std::string::npos);
        CHECK(t.find("Environment=\"P=$HOME 5%%\"\n") != std::string::npos);   // Environment= does not expand $
        CHECK(t.find("Description=50%% done\n") != std::string::npos); CHECK(t.find("WorkingDirectory=/srv/my dir\n") != std::string::npos);
        e.argv = {"/bin/x", "a\"b\\c"}; CHECK(os::service::render_systemd_unit(e).find("ExecStart=/bin/x \"a\\\"b\\\\c\"\n") != std::string::npos);
    }
    {   // macOS `launchctl print` parsing
        using st = os::service::service_state;
        CHECK(os::service::parse_launchctl_print("gui/501/x = {\n\tactive count = 1\n\tstate = running\n\tpid = 4242\n\tendpoints = {\n\t\tstate = active\n\t}\n}\n") == st::running);
        CHECK(os::service::parse_launchctl_print("gui/501/x = {\n\tstate = not running\n\tlast exit code = (never exited)\n}\n") == st::stopped);
        CHECK(os::service::parse_launchctl_print("gui/501/x = {\n\tstate = not running\n\tlast exit code = 0\n}\n") == st::stopped);
        CHECK(os::service::parse_launchctl_print("gui/501/x = {\n\tstate = not running\n\tlast exit code = 78: EX_CONFIG\n}\n") == st::failed);
        CHECK(os::service::parse_launchctl_print("gui/501/x = {\n\tstate = waiting\n\t\tstate = running\n}\n") == st::stopped);
    }
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_ANDROID) && !defined(FORGE_OS_IOS)
    auto d = fs::make_temp_dir(); CHECK(d); s.dir = *d;
    CHECK(os::service::query(s) == os::service::service_state::not_installed);
    CHECK(os::service::install(s)); CHECK(fs::is_file(os::service::unit_path(s)));
    CHECK(os::service::query(s) != os::service::service_state::not_installed);
    CHECK(os::service::uninstall(s)); CHECK(!fs::exists(os::service::unit_path(s)));
    fs::remove_all(*d);
#endif
}
TEST(registry_portable) {
#if defined(FORGE_OS_WINDOWS)
    namespace R = os::registry; using T = R::type;
    const std::string k = "HKCU\\Software\\FORGE-test\\" + std::to_string(os::pid()), sub = k + "\\child";
    CHECK(R::set(k, "n", std::int64_t(42))); CHECK(R::value_type(k, "n").value == T::qword);
    auto v = R::get(k, "n"); CHECK(v); CHECK(std::get<std::int64_t>(v.value) == 42);
    // DWORD stays DWORD while the number fits; a larger one becomes QWORD
    CHECK(R::set_as(k, "d", std::int64_t(7), T::dword)); CHECK(R::set(k, "d", std::int64_t(0xFFFFFFFF))); CHECK(R::value_type(k, "d").value == T::dword);
    CHECK(std::get<std::int64_t>(R::get(k, "d").value) == 0xFFFFFFFF);
    CHECK(R::set(k, "d", std::int64_t(1) << 40)); CHECK(R::value_type(k, "d").value == T::qword); CHECK(!R::set_as(k, "x", std::int64_t(1) << 40, T::dword));
    // REG_EXPAND_SZ stays expandable and is read unexpanded; non-ASCII round-trips
    CHECK(R::set_as(k, "e", std::string("%USERPROFILE%\\x"), T::expand_sz)); CHECK(R::set(k, "e", std::string("%TEMP%\\caf\xc3\xa9")));
    CHECK(R::value_type(k, "e").value == T::expand_sz); CHECK(std::get<std::string>(R::get(k, "e").value) == "%TEMP%\\caf\xc3\xa9");
    CHECK(R::set(k, "s", std::string("plain"))); CHECK(R::value_type(k, "s").value == T::sz); CHECK(std::get<std::string>(R::get(k, "s").value) == "plain");
    CHECK(R::set(k, "empty", std::string())); CHECK(std::get<std::string>(R::get(k, "empty").value).empty());
    // MULTI_SZ and BINARY
    std::vector<std::string> ms = {"one", "tw\xc3\xb6", "three"}; CHECK(R::set(k, "m", ms)); CHECK(R::value_type(k, "m").value == T::multi_sz);
    CHECK(std::get<std::vector<std::string>>(R::get(k, "m").value) == ms); CHECK(!R::set(k, "m2", std::vector<std::string>{"a", ""}));
    CHECK(R::set(k, "m0", std::vector<std::string>{})); CHECK(std::get<std::vector<std::string>>(R::get(k, "m0").value).empty());
    std::vector<std::uint8_t> bin = {0, 1, 2, 0xFF, 0}; CHECK(R::set(k, "b", bin)); CHECK(R::value_type(k, "b").value == T::binary);
    CHECK(std::get<std::vector<std::uint8_t>>(R::get(k, "b").value) == bin);
    CHECK(!R::set_as(k, "bad", std::string("x"), T::dword));
    // list / remove_value / missing values
    CHECK(R::set(sub, "c", std::int64_t(1)));
    auto l = R::list(k); CHECK(l); if (l) { CHECK(l->subkeys.size() == 1 && l->subkeys[0] == "child"); CHECK(l->values.size() == 8); }
    CHECK(R::remove_value(k, "s")); CHECK(!R::get(k, "s")); CHECK(R::value_type(k, "s").value == T::none); CHECK(!R::remove_value(k, "s"));
    CHECK(R::list(k)->values.size() == 7);
    // explicit 64-bit view (the native one for a 64-bit process; HKCU\Software is shared anyway)
    CHECK(R::set(k, "v64", std::int64_t(5), R::view::wow64_64)); CHECK(std::get<std::int64_t>(R::get(k, "v64", R::view::wow64_64).value) == 5);
    CHECK(R::remove_key(k)); CHECK(!R::get(k, "n")); CHECK(R::remove_key(k));
    if (auto p = R::list("HKCU\\Software\\FORGE-test"); p && p->subkeys.empty() && p->values.empty()) CHECK(R::remove_key("HKCU\\Software\\FORGE-test"));
    CHECK(!R::get("HKXX\\Software", "x")); CHECK(!R::remove_key("HKCU"));
#else
    CHECK(os::is_unsupported(os::registry::get("HKCU\\Software", "x").err));
#endif
}
