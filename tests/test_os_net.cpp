// forge::os phase 3: recursive/debounced file watch, net extensions, service control, registry.
#include <forge/os/file_watch_tree.hpp>
#include <forge/os/net.hpp>
#include <forge/os/service_control.hpp>
#include <forge/os/registry.hpp>
#include <atomic>
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
static std::vector<os::watch_event> drain(os::file_watcher& w, const std::string& suffix, int ms) {
    std::vector<os::watch_event> all;
    for (auto end = os::monotonic_ns() + std::uint64_t(ms) * 1000000ull; os::monotonic_ns() < end;) {
        auto e = w.poll(100); all.insert(all.end(), e.begin(), e.end());
        bool hit = false; for (auto& x : all) hit |= x.path.size() >= suffix.size() && x.path.compare(x.path.size() - suffix.size(), suffix.size(), suffix) == 0;
        if (hit && !suffix.empty()) break;
    }
    return all;
}
TEST(watch_many_dirs_and_removal) {   // > 64 directories: one Windows wait could not cover them
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d;
    os::file_watcher w; const int n = 80;
    for (int i = 0; i < n; ++i) { auto s = r + "/d" + std::to_string(i); CHECK(fs::make_dirs(s)); CHECK(w.add(s)); }
    os::sleep_ns(20'000'000);
    CHECK(os::write_file(r + "/d75/x.txt", "1"));
    auto ev = drain(w, "/d75/x.txt", 3000); CHECK(has_event(ev, "/d75/x.txt", os::watch_kind::created));
    CHECK(os::write_file(r + "/d3/y.txt", "1"));
    ev = drain(w, "/d3/y.txt", 3000); CHECK(has_event(ev, "/d3/y.txt", os::watch_kind::created));
    // remove(): no more events from that directory
    CHECK(w.remove(r + "/d5")); CHECK(!w.remove(r + "/not-watched"));
    CHECK(os::write_file(r + "/d5/z.txt", "1")); CHECK(os::write_file(r + "/d6/z.txt", "1"));
    ev = drain(w, "/d6/z.txt", 3000); CHECK(has_event(ev, "/d6/z.txt", os::watch_kind::created));
    for (auto& e : drain(w, "", 200)) ev.push_back(e);
    for (auto& e : ev) CHECK_MSG(e.path.find("/d5/") == std::string::npos, "event after remove: %s", e.path.c_str());
#if defined(_WIN32)
    {   // names match case-insensitively, as NTFS does
        os::file_watcher wc; CHECK(os::write_file(r + "/d0/CaseFile.txt", "a")); CHECK(wc.add(r + "/d0/CaseFile.txt"));
        CHECK(os::write_file(r + "/d0/casefile.TXT", "bb"));
        auto ce = drain(wc, "CaseFile.txt", 3000); CHECK(has_event(ce, "/d0/CaseFile.txt", os::watch_kind::modified));
    }
#endif
    fs::remove_all(r);
}
TEST(tree_watch_symlink_loop) {
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d;
    CHECK(fs::make_dirs(r + "/a/b"));
    auto s = fs::symlink(r, r + "/a/b/up");   // a/b/up -> root: following it forever would never end
#if defined(_WIN32)
    if (!s) {   // symlinks need Developer Mode or a privilege; a directory junction does not
        auto bs = [](std::string p) { for (auto& c : p) if (c == '/') c = '\\'; return p; };
        std::string cmd = "cmd /c mklink /J \"" + bs(r + "/a/b/up") + "\" \"" + bs(r) + "\" >NUL";
        if (std::system(cmd.c_str()) == 0 && os::stat(r + "/a/b/up").is_dir) s = {};
    }
#endif
    if (!s) { std::printf("  skip: %s\n", s.err.what.c_str()); fs::remove_all(r); return; }
    auto w = os::tree_watcher::open(r); CHECK(w); if (!w) return;
    std::printf("  watched %zu dirs\n", (*w)->watched_dirs());
    CHECK((*w)->watched_dirs() == 3);   // root, a, a/b: the loop back to the root is not entered
    CHECK(os::write_file(r + "/a/b/f.txt", "1"));
    std::vector<os::watch_event> ev; for (int i = 0; i < 20 && !has_event(ev, "/a/b/f.txt", os::watch_kind::created); ++i) { auto m = (*w)->poll(300); ev.insert(ev.end(), m.begin(), m.end()); }
    CHECK(has_event(ev, "/a/b/f.txt", os::watch_kind::created));
    w.value.reset();
#if defined(_WIN32)
    RemoveDirectoryW(os::widen(r + "/a/b/up").c_str());   // the link itself, before remove_all could walk through it
#endif
    fs::remove_all(r);
}
TEST(tree_watch_native_cap_falls_back_to_polling) {   // kqueue: more paths than descriptors allowed
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d;
    for (int i = 0; i < 6; ++i) { CHECK(fs::make_dirs(r + "/s" + std::to_string(i) + "/deep")); CHECK(os::write_file(r + "/s" + std::to_string(i) + "/deep/f.txt", "1")); }
    os::tree_watch_opts o; o.max_native_handles = 4; o.debounce_ms = 30;
    auto w = os::tree_watcher::open(r, o); CHECK(w); if (!w) return;
    std::printf("  backend %s, %zu of %zu dirs polled\n", (*w)->backend(), (*w)->polled_dirs(), (*w)->watched_dirs());
#if defined(__APPLE__) || defined(__FreeBSD__)
    CHECK((*w)->polled_dirs() > 0);
#endif
    os::sleep_ns(30'000'000);
    CHECK(os::write_file(r + "/s5/deep/f.txt", "changed!")); CHECK(os::write_file(r + "/s4/deep/new.txt", "n"));
    std::vector<os::watch_event> ev;
    for (int i = 0; i < 20 && !(has_event(ev, "/s5/deep/f.txt", os::watch_kind::modified) && has_event(ev, "/s4/deep/new.txt", os::watch_kind::created)); ++i) { auto m = (*w)->poll(300); ev.insert(ev.end(), m.begin(), m.end()); }
    CHECK(has_event(ev, "/s5/deep/f.txt", os::watch_kind::modified)); CHECK(has_event(ev, "/s4/deep/new.txt", os::watch_kind::created));
    fs::remove_all(r);
}
TEST(watch_overflow_rescans) {   // more changes than the kernel buffers between two polls
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d;
    int n = 3000;
#if defined(__linux__)
    long q = std::atol(os::read_file("/proc/sys/fs/inotify/max_queued_events").value.c_str());
    if (q <= 0 || q > 60000) { std::printf("  skip: max_queued_events=%ld\n", q); fs::remove_all(r); return; }
    n = int(q / 2) + 200;   // >= 2 events per file (create, close_write)
#endif
    {
        os::file_watcher w; CHECK(w.add(r));
        for (int i = 0; i < n; ++i) CHECK(os::write_file(r + "/file-with-a-longish-name-" + std::to_string(i) + ".txt", ""));
        auto ev = w.poll(1000); bool overflow = false; std::size_t created = 0;
        for (int i = 0; i < 50; ++i) { for (auto& e : ev) { overflow |= e.kind == os::watch_kind::overflow; created += e.kind == os::watch_kind::created; } ev = w.poll(50); if (ev.empty()) break; }
        std::printf("  %d files: overflow=%d, %zu created events\n", n, int(overflow), created);
#if defined(__linux__)
        CHECK(overflow);
#endif
#if defined(_WIN32)
        CHECK(overflow || created >= std::size_t(n));
#endif
    }
    {   // the tree watcher turns an overflow into a rescan: every new file is reported
        auto w = os::tree_watcher::open(r); CHECK(w); if (!w) return;
        for (int i = 0; i < n; ++i) CHECK(os::write_file(r + "/second-" + std::to_string(i) + ".txt", ""));
        std::set<std::string> seen;
        for (int i = 0; i < 40 && seen.size() < std::size_t(n); ++i) for (auto& e : (*w)->poll(300)) if (e.kind == os::watch_kind::created && e.path.find("/second-") != std::string::npos) seen.insert(e.path);
        CHECK_MSG(seen.size() == std::size_t(n), "%zu of %d created reported", seen.size(), n);
    }
    fs::remove_all(r);
}

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
TEST(url_resolution) {
    auto b = os::net::parse_url("http://h/a/b/c?q=1"); CHECK(b);
    auto r = [&](const char* ref) { auto u = os::net::resolve_reference(*b, ref); return u ? os::net::to_string(*u) : std::string("ERR"); };
    CHECK(r("d") == "http://h/a/b/d"); CHECK(r("../d") == "http://h/a/d"); CHECK(r("./d/../e") == "http://h/a/b/e"); CHECK(r("../../../../x") == "http://h/x");
    CHECK(r("/x?y") == "http://h/x?y"); CHECK(r("?z") == "http://h/a/b/c?z"); CHECK(r("#f") == "http://h/a/b/c?q=1");
    CHECK(r("//other:81/p") == "http://other:81/p"); CHECK(r("http://x:8080/") == "http://x:8080/");
    os::net::url v6; v6.scheme = "http"; v6.host = "::1"; v6.port = 8080; CHECK(os::net::host_port(v6) == "[::1]:8080");
    v6.port = 80; CHECK(os::net::host_port(v6) == "[::1]");
    CHECK(os::net::detail_net::encode_target("/a b\r\n%41%zz", false) == "/a%20b%0D%0A%41%25zz");
    CHECK(!os::net::parse_url("http://h\r\nX: 1/")); CHECK(!os::net::parse_url("http://h:99999/")); CHECK(!os::net::parse_url("http://h:8x/"));
}
// A tiny HTTP server: answers `n` connections with handler(request head, connection).
template <class F> static std::thread serve(os::socket& l, int n, F handler) {
    return std::thread([&l, n, handler] {
        for (int k = 0; k < n; ++k) {
            auto c = l.accept(); if (!c) return;
            std::string req; char b[4096];
            while (req.find("\r\n\r\n") == std::string::npos) { long m = c->recv(b, sizeof b); if (m <= 0) break; req.append(b, std::size_t(m)); }
            handler(req, *c);
        }
    });
}
static void reply(os::socket& c, const std::string& s) { c.send_all(s.data(), s.size()); }
TEST(http_rejects_injection) {
    auto l = os::socket::listen("127.0.0.1", 0); CHECK(l); auto base = "http://127.0.0.1:" + std::to_string(l->local_port());
    os::net::http_request q; q.url = base + "/";
    q.method = "GET / HTTP/1.1\r\nX-Evil: 1\r\n"; auto a = os::net::http(q); CHECK(!a); CHECK(a.err.code == EINVAL);
    q.method = "GET"; q.headers["X-Ok"] = "v\r\nX-Evil: 1"; auto b = os::net::http(q); CHECK(!b); CHECK(b.err.code == EINVAL);
    q.headers.clear(); q.headers["Bad Name"] = "v"; CHECK(!os::net::http(q));
    q.headers.clear(); q.headers[std::string("X-Nul")] = std::string("a\0b", 3); CHECK(!os::net::http(q));
    // A path with CR/LF and spaces is percent-encoded, never sent raw.
    std::string seen;
    auto srv = serve(*l, 1, [&](const std::string& req, os::socket& c) { seen = req; reply(c, "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"); });
    q.headers.clear(); q.url = base + "/a b\r\nX-Injected: 1";
    auto r = os::net::http(q); srv.join();
    CHECK(r); CHECK(r->body == "ok");
    CHECK_MSG(seen.rfind("GET /a%20b%0D%0AX-Injected:%201 HTTP/1.1\r\n", 0) == 0, "request: %s", seen.c_str());
    CHECK(seen.find("\r\nX-Injected") == std::string::npos);
}
TEST(http_redirect_credentials_and_relative) {
    auto a = os::socket::listen("127.0.0.1", 0), b = os::socket::listen("127.0.0.1", 0); CHECK(a); CHECK(b);
    auto pa = std::to_string(a->local_port()), pb = std::to_string(b->local_port());
    std::vector<std::string> ra, rb;
    auto sa = serve(*a, 3, [&](const std::string& req, os::socket& c) {
        ra.push_back(req);
        if (req.rfind("GET /dir/same", 0) == 0) reply(c, "HTTP/1.1 302 Found\r\nLocation: next?x=1\r\nContent-Length: 0\r\n\r\n");
        else if (req.rfind("GET /dir/next", 0) == 0) reply(c, "HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nsame");
        else reply(c, "HTTP/1.1 307 Temporary Redirect\r\nLocation: http://127.0.0.1:" + pb + "/other\r\nContent-Length: 0\r\n\r\n");
    });
    auto sb = serve(*b, 1, [&](const std::string& req, os::socket& c) { rb.push_back(req); reply(c, "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nother"); });
    os::net::http_request q; q.headers["Authorization"] = "Bearer secret"; q.headers["Cookie"] = "s=1"; q.headers["X-Keep"] = "1";
    q.url = "http://127.0.0.1:" + pa + "/dir/same";
    auto r = os::net::http(q); CHECK(r); CHECK(r->body == "same"); CHECK(r->url == "http://127.0.0.1:" + pa + "/dir/next?x=1");
    q.url = "http://127.0.0.1:" + pa + "/cross";
    auto x = os::net::http(q); CHECK(x); CHECK(x->body == "other");
    sa.join(); sb.join();
    CHECK(ra.size() == 3 && rb.size() == 1);
    CHECK_MSG(ra[1].rfind("GET /dir/next?x=1 ", 0) == 0, "%s", ra[1].c_str());
    CHECK(ra[1].find("Authorization: Bearer secret") != std::string::npos);   // same origin: kept
    CHECK(rb[0].find("Authorization") == std::string::npos); CHECK(rb[0].find("Cookie") == std::string::npos);   // other origin: dropped
    CHECK(rb[0].find("X-Keep: 1") != std::string::npos); CHECK(rb[0].find("Host: 127.0.0.1:" + pb) != std::string::npos);
}
TEST(http_truncation_and_headers) {
    auto l = os::socket::listen("127.0.0.1", 0); CHECK(l); auto base = "http://127.0.0.1:" + std::to_string(l->local_port());
    auto srv = serve(*l, 5, [&](const std::string& req, os::socket& c) {
        if (req.rfind("GET /short", 0) == 0) reply(c, "HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc");
        else if (req.rfind("GET /nochunkend", 0) == 0) reply(c, "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n");
        else if (req.rfind("GET /badchunk", 0) == 0) reply(c, "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\nabc\r\n0\r\n\r\n");
        else if (req.rfind("GET /cookies", 0) == 0) reply(c, "HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nSet-Cookie: a=1; Expires=Wed, 21 Oct 2026 07:28:00 GMT\r\nSet-Cookie: b=2\r\nX-L: 1\r\nX-L: 2\r\nContent-Length: 2\r\n\r\nokEXTRA");
        else reply(c, "HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nab");
    });
    auto a = os::net::http_get(base + "/short"); CHECK(!a); CHECK(a.err.code == EPROTO);
    auto b = os::net::http_get(base + "/nochunkend"); CHECK(!b); CHECK(b.err.code == EPROTO);
    auto c = os::net::http_get(base + "/badchunk"); CHECK(!c); CHECK(c.err.code == EPROTO);
    auto d = os::net::http_get(base + "/cookies"); CHECK(d); CHECK(d->status == 200); CHECK(d->body == "ok");
    CHECK(d->values("set-cookie").size() == 2); CHECK(d->values("set-cookie")[0] == "a=1; Expires=Wed, 21 Oct 2026 07:28:00 GMT");
    CHECK(d->headers["set-cookie"].find('\n') != std::string::npos); CHECK(d->headers["x-l"] == "1, 2");
    auto e = os::net::http_get(base + "/conflict"); CHECK(!e); CHECK(e.err.code == EPROTO);
    srv.join();
}
TEST(http_overall_deadline) {
    auto l = os::socket::listen("127.0.0.1", 0); CHECK(l); auto base = "http://127.0.0.1:" + std::to_string(l->local_port());
    std::atomic<bool> done{false};
    auto srv = serve(*l, 1, [&](const std::string&, os::socket& c) {   // drips a byte every 50 ms: each read succeeds
        reply(c, "HTTP/1.1 200 OK\r\nX-Slow: ");
        for (int i = 0; i < 60 && !done; ++i) { if (c.send("x", 1) != 1) break; os::sleep_ns(50'000'000); }
    });
    os::net::http_request q; q.url = base + "/slow"; q.timeout_ms = 2000; q.deadline_ms = 400;
    auto t0 = os::monotonic_ns(); auto r = os::net::http(q); auto ms = (os::monotonic_ns() - t0) / 1000000;
    done = true; srv.join();
    CHECK(!r); CHECK(r.err.code == ETIMEDOUT); CHECK_MSG(ms < 1500, "took %llu ms", (unsigned long long)ms);
}
TEST(service_units) {
    os::service::service_spec s; s.name = "forge-test-svc"; s.description = "Forge test"; s.argv = {"/usr/bin/forge", "script", "my file.fg"};
    s.env["A"] = "1 2"; s.working_dir = "/tmp";
    auto u = os::service::render_systemd_unit(s);
    CHECK(u.find("ExecStart=/usr/bin/forge script \"my file.fg\"") != std::string::npos);
    CHECK(u.find("Environment=\"A=1 2\"") != std::string::npos); CHECK(u.find("Restart=on-failure") != std::string::npos);
    auto p = os::service::render_launchd_plist(s); CHECK(p.find("<string>my file.fg</string>") != std::string::npos);
    CHECK(!os::service::valid_name("bad name")); CHECK(!os::service::install({}));
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
    CHECK(os::registry::set("HKCU\\Software\\ForgeTest", "n", std::int64_t(42)));
    auto v = os::registry::get("HKCU\\Software\\ForgeTest", "n"); CHECK(v); CHECK(std::get<std::int64_t>(v.value) == 42);
    CHECK(os::registry::remove_key("HKCU\\Software\\ForgeTest"));
#else
    CHECK(os::is_unsupported(os::registry::get("HKCU\\Software", "x").err));
#endif
}
