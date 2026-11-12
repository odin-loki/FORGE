// forge::os::reactor: timers, posts, sockets, pipes, cancellation, child exit, events, signals, files, watches,
// coroutines, shutdown. Every test arms a watchdog timer, so a broken backend fails instead of hanging.
#include <forge/os/reactor.hpp>
#include <forge/os/pipe.hpp>
#include <forge/os/fs.hpp>
#include <forge/os/file_watch.hpp>
#include <forge/os/local_ipc.hpp>
#include <atomic>
#include <thread>
#include <vector>
#include "check.hpp"
using namespace forge;
namespace {
constexpr std::uint64_t ms = 1'000'000ull;
// Stops the loop and fails the test when it runs longer than `seconds`.
struct watchdog {
    os::reactor& r; os::reactor::op_id id; bool fired = false;
    explicit watchdog(os::reactor& x, int seconds = 20) : r(x) {
        id = r.after(std::uint64_t(seconds) * 1000 * ms, [this](const os::error& e) { if (!e) { fired = true; r.stop(); } });
    }
    void done() { r.cancel(id); r.run(); CHECK_MSG(!fired, "watchdog fired: the loop did not finish in time"); }
};
bool on_wine() {
#if defined(_WIN32)
    return GetProcAddress(GetModuleHandleA("ntdll.dll"), "wine_get_version") != nullptr;
#else
    return false;
#endif
}
}

TEST(reactor_basics) {
    os::reactor r; CHECK(r.valid()); CHECK(!r.init_error());
    std::printf("  backend: %s\n", r.backend());
    CHECK(std::string(r.backend()) == "epoll" || std::string(r.backend()) == "kqueue" || std::string(r.backend()) == "iocp");
    CHECK(r.pending() == 0); CHECK(r.run() == 0);   // nothing to do: returns at once
    int n = 0; r.post([&] { ++n; }); r.post([&] { ++n; });
    CHECK(r.run() == 2); CHECK(n == 2);
    CHECK(!r.cancel({}));  CHECK(!r.cancel({12345}));
    CHECK(os::is_cancelled(os::cancelled_error())); CHECK(!os::is_cancelled(os::error{})); CHECK(!os::is_cancelled(os::unsupported("x")));
}

TEST(reactor_timers_fire_in_deadline_order) {
    os::reactor r; watchdog wd(r);
    std::vector<int> order; auto t0 = os::monotonic_ns();
    r.after(60 * ms, [&](const os::error& e) { CHECK(!e); order.push_back(3); });
    r.after(20 * ms, [&](const os::error& e) { CHECK(!e); order.push_back(1); });
    r.after(40 * ms, [&](const os::error& e) { CHECK(!e); order.push_back(2); });
    r.after(0, [&](const os::error& e) { CHECK(!e); order.push_back(0); });
    std::uint64_t same = os::monotonic_ns() + 80 * ms;   // equal deadlines: creation order
    for (int i = 0; i < 5; ++i) r.at(same, [&, i](const os::error&) { order.push_back(10 + i); });
    std::uint64_t fired_at = 0; r.after(100 * ms, [&](const os::error&) { fired_at = os::monotonic_ns(); r.cancel(wd.id); });
    CHECK(r.pending() == 11);
    r.run();
    CHECK(!wd.fired);
    CHECK(order == (std::vector<int>{0, 1, 2, 3, 10, 11, 12, 13, 14}));
    CHECK_MSG(fired_at - t0 >= 100 * ms, "a timer fired early: %llu ns", static_cast<unsigned long long>(fired_at - t0));
    CHECK(r.pending() == 0);
}

TEST(reactor_periodic_timer_and_cancel) {
    os::reactor r; watchdog wd(r);
    int ticks = 0, cancelled = 0; os::reactor::op_id per;
    per = r.every(5 * ms, [&](const os::error& e) {
        if (e) { CHECK(os::is_cancelled(e)); ++cancelled; return; }
        if (++ticks == 4) CHECK(r.cancel(per));   // from inside its own callback
    });
    int never = 0; auto dead = r.after(3600ull * 1000 * ms, [&](const os::error& e) { CHECK(os::is_cancelled(e)); ++never; });
    r.after(10 * ms, [&](const os::error&) { CHECK(r.cancel(dead)); CHECK(!r.cancel(dead)); });
    wd.done();   // cancels the watchdog and runs to completion
    CHECK(ticks == 4); CHECK(cancelled == 1); CHECK(never == 1); CHECK(r.pending() == 0);
}

TEST(reactor_post_from_other_threads) {
    os::reactor r; watchdog wd(r);
    constexpr int threads = 4, per = 500;
    std::atomic<int> sent{0}; int got = 0; std::vector<int> last(threads, -1); bool ordered = true;
    auto loop_thread = std::this_thread::get_id(); bool same_thread = true;
    r.hold();   // keep run() alive while the producers are still starting
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t) th.emplace_back([&, t] {
        for (int i = 0; i < per; ++i) { r.post([&, t, i] { same_thread &= std::this_thread::get_id() == loop_thread; ordered &= last[t] == i - 1; last[t] = i; if (++got == threads * per) { r.unhold(); r.cancel(wd.id); } }); ++sent; }
    });
    r.run();
    for (auto& t : th) t.join();
    CHECK(!wd.fired); CHECK(got == threads * per); CHECK(sent == threads * per); CHECK(ordered); CHECK(same_thread);
    // stop() from another thread ends a loop that would otherwise wait
    r.restart(); r.hold();
    std::thread stopper([&] { os::sleep_ns(20 * ms); r.stop(); });
    auto t0 = os::monotonic_ns(); r.run(); stopper.join();
    CHECK(r.stopped()); CHECK(os::monotonic_ns() - t0 < 10'000 * ms); r.unhold(); r.restart();
    CHECK(r.run_for(10) == 0);
}

TEST(reactor_tcp_echo_many_sockets) {
    os::reactor r; watchdog wd(r, 60);
    auto ls = os::socket::listen("127.0.0.1", 0); CHECK(ls); if (!ls) return;
    int port = ls->local_port(); CHECK(port > 0);
    constexpr int N = 48; constexpr std::size_t len = 20000;
    struct conn { os::socket s; std::vector<char> buf; std::size_t echoed = 0; };
    std::vector<std::unique_ptr<conn>> server; int accepted = 0, served = 0, verified = 0, failures = 0;
    std::function<void(conn*)> serve = [&](conn* c) {
        r.read(c->s, c->buf.data(), c->buf.size(), [&, c](const os::error& e, std::size_t n) {
            if (e) { ++failures; return; }
            if (n == 0) { ++served; c->s.close(); return; }   // client finished
            r.write(c->s, c->buf.data(), n, [&, c, n](const os::error& e2, std::size_t w) { if (e2 || w != n) { ++failures; return; } c->echoed += n; serve(c); });
        });
    };
    std::function<void()> accept_one = [&] {
        r.accept(*ls, [&](const os::error& e, os::socket s) {
            if (e) { ++failures; return; }
            CHECK(s.valid()); server.push_back(std::make_unique<conn>()); server.back()->s = std::move(s); server.back()->buf.resize(4096);
            serve(server.back().get());
            if (++accepted < N) accept_one();
        });
    };
    accept_one();
    struct client { os::socket s; std::vector<char> out, in; };
    std::vector<std::unique_ptr<client>> clients;
    for (int i = 0; i < N; ++i) {
        clients.push_back(std::make_unique<client>()); client* c = clients.back().get();
        c->out.resize(len); for (std::size_t k = 0; k < len; ++k) c->out[k] = char((k * 31 + std::size_t(i)) & 0xff); c->in.resize(len);
        r.connect("127.0.0.1", port, [&, c](const os::error& e, os::socket s) {
            if (e) { ++failures; std::printf("  connect: %s\n", e.what.c_str()); return; }
            c->s = std::move(s);
            r.write(c->s, c->out.data(), c->out.size(), [&](const os::error& e2, std::size_t w) { if (e2 || w != len) ++failures; });
            r.read_exact(c->s, c->in.data(), c->in.size(), [&, c](const os::error& e2, std::size_t n) {
                if (e2 || n != len || c->in != c->out) ++failures; else ++verified;
                c->s.shutdown_write();   // the server reads 0 and closes
            });
        });
    }
    r.after(0, [&](const os::error&) {});
    int failed_at = -1;   // on a failure: cancel what is left so the loop ends (the cancellations are not failures)
    std::function<void()> check_done = [&] {
        if (failures) { failed_at = failures; r.cancel_all(); return; }
        if (served == N) r.cancel(wd.id); else r.after(5 * ms, [&](const os::error& e) { if (!e) check_done(); });
    };
    check_done();
    r.run();
    CHECK(!wd.fired); CHECK_MSG(failures == 0, "%d failures", failed_at); CHECK(accepted == N); CHECK(verified == N); CHECK(served == N);
    for (auto& c : server) CHECK(c->echoed == len);
    CHECK(r.pending() == 0);
    // connecting to a closed port fails with the OS error, not a hang
    int dead_port = 0; { auto tmp = os::socket::listen("127.0.0.1", 0); CHECK(tmp); dead_port = tmp->local_port(); }
    bool refused = false; watchdog wd2(r);
    r.connect("127.0.0.1", dead_port, [&](const os::error& e, os::socket s) { refused = bool(e) && !s.valid(); r.cancel(wd2.id); });
    r.run(); CHECK(refused); CHECK(!wd2.fired);
    // a name that cannot resolve reports the resolver's error
    bool noname = false; watchdog wd3(r, 60);   // below: a label over 63 bytes fails in the resolver without a query
    r.connect(std::string(70, 'x') + ".invalid", 80,[&](const os::error& e, os::socket) { noname = bool(e); r.cancel(wd3.id); });
    r.run(); CHECK(noname); CHECK(!wd3.fired);
}

TEST(reactor_pipe_read_write_eof) {
    os::reactor r; watchdog wd(r);
    auto p = os::make_pipe(); CHECK(p); if (!p) return;
    std::string got; char buf[256]; bool eof = false; int reads = 0;
    std::function<void()> rd = [&] {
        r.read(p->read, buf, sizeof buf, [&](const os::error& e, std::size_t n) {
            CHECK_MSG(!e, "%s", e.what.c_str()); ++reads;
            if (e || n == 0) { eof = !e; r.cancel(wd.id); return; }
            got.append(buf, n); rd();
        });
    };
    rd();
    std::string msg(3000, 'x'); for (std::size_t i = 0; i < msg.size(); ++i) msg[i] = char('a' + i % 26);
    r.after(10 * ms, [&](const os::error&) {
        r.write(p->write, msg.data(), msg.size(), [&](const os::error& e, std::size_t n) { CHECK(!e); CHECK(n == msg.size()); p->write.close(); });
    });
    r.run();
    CHECK(!wd.fired); CHECK(eof); CHECK(got == msg); CHECK(reads >= 2);
    // zero-length operations complete at once without touching the handle
    bool z = false; r.read(p->read, buf, 0, [&](const os::error& e, std::size_t n) { z = !e && n == 0; }); r.run(); CHECK(z);
}

TEST(reactor_cancel_pending_io) {
    os::reactor r; watchdog wd(r);
    auto p = os::make_pipe(); CHECK(p); if (!p) return;
    char buf[16]; int state = 0; std::size_t nread = 99;
    auto id = r.read(p->read, buf, sizeof buf, [&](const os::error& e, std::size_t n) { state = os::is_cancelled(e) ? 1 : 2; nread = n; });
    CHECK(id); CHECK(r.pending() == 2);
    r.after(20 * ms, [&](const os::error&) { CHECK(r.cancel(id)); CHECK(state == 0); });   // the callback never runs inside cancel()
    wd.done();
    CHECK(state == 1); CHECK(nread == 0); CHECK(!r.cancel(id)); CHECK(r.pending() == 0);
    // the pipe still works afterwards
    watchdog wd2(r); std::string got;
    r.read(p->read, buf, sizeof buf, [&](const os::error& e, std::size_t n) { CHECK(!e); got.assign(buf, n); r.cancel(wd2.id); });
    r.write(p->write, "ping", 4, [&](const os::error& e, std::size_t n) { CHECK(!e); CHECK(n == 4); });
    r.run(); CHECK(got == "ping"); CHECK(!wd2.fired);
    // sockets: a pending accept and a pending read are cancelled; cancel_all sweeps the rest
    auto ls = os::socket::listen("127.0.0.1", 0); CHECK(ls); if (!ls) return;
    int c1 = 0, c2 = 0, c3 = 0;
    auto a = r.accept(*ls, [&](const os::error& e, os::socket s) { c1 = os::is_cancelled(e) && !s.valid(); });
    r.after(3600ull * 1000 * ms, [&](const os::error& e) { c2 = os::is_cancelled(e); });
    r.wait_readable(*ls, [&](const os::error& e) { c3 = os::is_cancelled(e); });
    CHECK(r.run_for(20) == 0); CHECK(r.pending() == 3);
    CHECK(r.cancel(a)); r.cancel_all(); r.run();
    CHECK(c1 == 1); CHECK(c2 == 1); CHECK(c3 == 1); CHECK(r.pending() == 0);
    r.forget(*ls);
}

TEST(reactor_readiness_waits) {
    os::reactor r; watchdog wd(r);
    auto ls = os::socket::listen("127.0.0.1", 0); CHECK(ls); if (!ls) return;
    bool readable = false, writable = false; os::socket peer;
    r.wait_readable(*ls, [&](const os::error& e) { CHECK(!e); readable = true; auto a = ls->accept(); CHECK(a); peer = std::move(a.value); });
    auto c = os::socket::connect("127.0.0.1", ls->local_port()); CHECK(c); if (!c) return;
    r.wait_writable(*c, [&](const os::error& e) { CHECK(!e); writable = true; });
    wd.done();
    CHECK(readable); CHECK(writable); CHECK(peer.valid());
}

TEST(reactor_child_exit) {
    os::reactor r; watchdog wd(r, 60);
#if defined(FORGE_OS_IOS)
    bool unsup = false; r.on_exit(1, [&](const os::error& e) { unsup = os::is_unsupported(e); }); wd.done(); CHECK(unsup);
#else
#if defined(FORGE_OS_WINDOWS)
    auto c = os::child::start({"cmd", "/c", "exit 3"});
#else
    auto c = os::child::start({"sh", "-c", "sleep 0.1; exit 3"});
#endif
    CHECK_MSG(bool(c), "%s", c.err.what.c_str()); if (!c) { wd.done(); return; }
    bool exited = false; r.on_exit(c->pid(), [&](const os::error& e) { CHECK_MSG(!e, "%s", e.what.c_str()); exited = true; });
    // a pid that does not exist completes at once; a cancelled watch reports cancellation
    bool gone = false; r.on_exit(0x3ffffff0, [&](const os::error& e) { gone = !e; });
    bool cancelled = false; auto self = r.on_exit(os::pid(), [&](const os::error& e) { cancelled = os::is_cancelled(e); });
    r.after(5 * ms, [&](const os::error&) { CHECK(r.cancel(self)); });
    wd.done();
    CHECK(exited); CHECK(gone); CHECK(cancelled);
    CHECK(c->wait(5000) == 3);   // the reactor did not reap it
#endif
}

TEST(reactor_events_and_offload) {
    os::reactor r; watchdog wd(r);
    auto ev = os::wait_event::create(); CHECK(ev); if (!ev) return;
    bool seen = false; r.on_event(*ev, [&](const os::error& e) { CHECK(!e); seen = true; });
    CHECK(r.run_for(20) == 0); CHECK(!seen);
    std::thread t([&] { os::sleep_ns(10 * ms); ev->set(); });
    std::atomic<int> worked{0}; bool done = false; auto main_id = std::this_thread::get_id(); bool off_thread = false;
    r.offload([&] { off_thread = std::this_thread::get_id() != main_id; ++worked; }, [&](const os::error& e) { CHECK(!e); done = true; });
    wd.done(); t.join();
    CHECK(seen); CHECK(ev->is_set()); CHECK(done); CHECK(worked == 1); CHECK(off_thread);
    // an event that is already set completes; a cancelled one reports it
    bool again = false, can = false; r.on_event(*ev, [&](const os::error& e) { again = !e; });
    auto e2 = os::wait_event::create(); auto id = r.on_event(*e2, [&](const os::error& e) { can = os::is_cancelled(e); });
    r.after(5 * ms, [&](const os::error&) { r.cancel(id); });
    r.run(); CHECK(again); CHECK(can);
}

TEST(reactor_signals) {
#if defined(FORGE_OS_POSIX)
    os::reactor r; watchdog wd(r);
    int hups = 0; os::reactor::op_id id;
    id = r.on_signal(os::sig::hangup, [&](os::sig s) { CHECK(s == os::sig::hangup); if (++hups == 2) { r.cancel(id); r.cancel(wd.id); } else ::raise(SIGHUP); });
    CHECK(id);
    r.after(5 * ms, [&](const os::error&) { ::raise(SIGHUP); });
    r.run();
    CHECK(hups == 2); CHECK(!wd.fired);
    os::signal_listener::instance().ignore(os::sig::hangup);
#else
    os::reactor r;   // console control events cannot be raised at ourselves safely; the registration must work
    auto id = r.on_signal(os::sig::interrupt, [](os::sig) {}); CHECK(id); CHECK(r.pending() == 1);
    CHECK(!r.on_signal(os::sig::pipe, [](os::sig) {}));   // no SIGPIPE on Windows
    CHECK(r.run_for(20) == 0); CHECK(r.cancel(id)); CHECK(r.pending() == 0);
    os::signal_listener::instance().ignore(os::sig::interrupt);
#endif
}

TEST(reactor_file_io) {
    os::reactor r; watchdog wd(r);
    auto d = os::fs::make_temp_dir("forge-reactor-"); CHECK(d); if (!d) return;
    auto path = os::join(*d, "data.bin");
    std::string blob(200000, '\0'); for (std::size_t i = 0; i < blob.size(); ++i) blob[i] = char(i * 7);
    auto f = os::file::open(path, os::rdwr | os::create | os::truncate); CHECK(f); if (!f) return;
    std::string back(blob.size(), '\0'); bool ok = false, eof = false; char tail[8];
    r.write_file(*f, blob.data(), blob.size(), 0, [&](const os::error& e, std::size_t n) {
        CHECK_MSG(!e, "%s", e.what.c_str()); CHECK(n == blob.size());
        r.read_file(*f, back.data(), back.size(), 0, [&](const os::error& e2, std::size_t n2) {
            CHECK(!e2); ok = n2 == blob.size() && back == blob;
            r.read_file(*f, tail, sizeof tail, blob.size(), [&](const os::error& e3, std::size_t n3) { eof = !e3 && n3 == 0; });
        });
    });
    wd.done();
    CHECK(ok); CHECK(eof); std::printf("  file backend: %s\n", r.file_backend().c_str()); CHECK(!r.file_backend().empty());
    f->close(); os::remove_all(*d);
}

TEST(reactor_file_watch) {
    os::reactor r; watchdog wd(r, 30);
    auto d = os::fs::make_temp_dir("forge-reactor-watch-"); CHECK(d); if (!d) return;
    os::file_watcher fw; auto st = fw.add(*d);
    if (!st) { std::printf("  skip: %s\n", st.err.what.c_str()); wd.done(); os::remove_all(*d); return; }
    bool seen = false; os::reactor::op_id id;
    id = r.watch(fw, [&](std::vector<os::watch_event> ev) { for (auto& e : ev) if (e.path.find("made.txt") != std::string::npos) seen = true; if (seen) { r.cancel(id); r.cancel(wd.id); } });
    CHECK(id);
    r.after(30 * ms, [&](const os::error&) { CHECK(os::write_file(os::join(*d, "made.txt"), "x")); });
    r.run();
    CHECK(seen); CHECK(!wd.fired); os::remove_all(*d);
}

TEST(reactor_local_ipc_stream) {
    // local_stream / local_listener: AF_UNIX on POSIX, named pipes on Windows (handle I/O path there).
    os::reactor r; watchdog wd(r, 30);
    auto name = os::local_endpoint("forge-reactor-" + std::to_string(os::pid()));
    auto ls = os::local_listener::listen(name); CHECK_MSG(bool(ls), "%s", ls.err.what.c_str()); if (!ls) { wd.done(); return; }
    os::local_stream server, client; std::string got; char buf[64]; bool replied = false; char rbuf[8];
    r.offload([&] { auto a = ls->accept(10000); if (a) server = std::move(a.value); }, [&](const os::error& e) {
        CHECK(!e); CHECK(server.valid()); if (!server.valid()) { r.cancel(wd.id); return; }
        r.read(server, buf, sizeof buf, [&](const os::error& e2, std::size_t n) {
            CHECK_MSG(!e2, "%s", e2.what.c_str()); got.assign(buf, n);
            r.write(server, "pong", 4, [&](const os::error& e3, std::size_t w) { CHECK(!e3); CHECK(w == 4); });
        });
    });
    r.after(20 * ms, [&](const os::error&) {
        auto c = os::local_stream::connect(name, 5000); CHECK_MSG(bool(c), "%s", c.err.what.c_str()); if (!c) { r.cancel(wd.id); return; }
        client = std::move(c.value);
        r.write(client, "ping", 4, [&](const os::error& e, std::size_t n) { CHECK(!e); CHECK(n == 4); });
        r.read_exact(client, rbuf, 4, [&](const os::error& e, std::size_t n) { CHECK(!e); replied = n == 4 && std::string(rbuf, 4) == "pong"; r.cancel(wd.id); });
    });
    r.run();
    CHECK(!wd.fired); CHECK(got == "ping"); CHECK(replied);
}

namespace {
os::reactor::task echo_server(os::reactor& r, os::socket& ls, int& served) {
    auto a = co_await r.async_accept(ls);
    if (!*a) co_return;
    os::socket s = std::move(a->sock); char buf[512];
    for (;;) {
        auto rd = co_await r.async_read(s, buf, sizeof buf);
        if (!rd || rd.n == 0) break;
        auto wr = co_await r.async_write(s, buf, rd.n);
        if (!wr) break;
    }
    ++served;
}
os::reactor::task echo_client(os::reactor& r, int port, std::string msg, std::string& out, bool& slept) {
    auto t0 = os::monotonic_ns();
    auto e = co_await r.async_sleep(15 * ms); slept = !e && os::monotonic_ns() - t0 >= 15 * ms;
    auto c = co_await r.async_connect("127.0.0.1", port);
    if (!*c) co_return;
    os::socket s = std::move(c->sock);
    auto wr = co_await r.async_write(s, msg.data(), msg.size());
    if (!wr || wr.n != msg.size()) co_return;
    std::string in(msg.size(), '\0');
    auto rd = co_await r.async_read_exact(s, in.data(), in.size());
    if (rd) out = in;
    co_await r.async_yield();
    s.shutdown_write();
}
}
TEST(reactor_coroutines) {
    os::reactor r; watchdog wd(r);
    auto ls = os::socket::listen("127.0.0.1", 0); CHECK(ls); if (!ls) return;
    int served = 0; std::string out; bool slept = false;
    echo_server(r, *ls, served);
    echo_client(r, ls->local_port(), "hello over a coroutine", out, slept);
    wd.done();
    CHECK(served == 1); CHECK(out == "hello over a coroutine"); CHECK(slept);
}

TEST(reactor_shutdown_with_pending_operations) {
    auto p = os::make_pipe(); CHECK(p); if (!p) return;
    auto ls = os::socket::listen("127.0.0.1", 0); CHECK(ls); if (!ls) return;
    auto ev = os::wait_event::create(); CHECK(ev);
    auto d = os::fs::make_temp_dir("forge-reactor-sd-"); CHECK(d); if (!d) return;
    auto f = os::file::open(os::join(*d, "f"), os::rdwr | os::create); CHECK(f);
    int called = 0; char buf[32]; std::string big(1 << 20, 'z');
    std::atomic<bool> release{false};
    {
        os::reactor r;
        r.read(p->read, buf, sizeof buf, [&](const os::error&, std::size_t) { ++called; });
        r.accept(*ls, [&](const os::error&, os::socket) { ++called; });
        r.after(3600ull * 1000 * ms, [&](const os::error&) { ++called; });
        r.every(1 * ms, [&](const os::error&) {});
        r.on_event(*ev, [&](const os::error&) { ++called; });
        r.on_exit(os::pid(), [&](const os::error&) { ++called; });
        r.write_file(*f, big.data(), big.size(), 0, [&](const os::error&, std::size_t) { ++called; });
        r.offload([&] { os::sleep_ns(20 * ms); }, [&](const os::error&) { ++called; });
        r.connect("127.0.0.1", ls->local_port(), [&](const os::error&, os::socket) { ++called; });
        r.post([&] { ++called; });
        CHECK(r.pending() >= 8);
        r.run_for(5);   // some of it starts, none of the long waits finish
    }   // destructor: waits for in-flight OS calls, drops the rest without calling back
    int after = called; os::sleep_ns(30 * ms); CHECK(called == after);
    // the handles are still usable by someone else
    CHECK(p->write.write_all("ok")); CHECK(p->read.read_some(16, 2000) == "ok");
    {   // cancel_all + run delivers every callback before destruction
        auto ls2 = os::socket::listen("127.0.0.1", 0); CHECK(ls2);   // a fresh listener: the first one may still hold the connection made above
        os::reactor r; int cancelled = 0, total = 0;
        r.read(p->read, buf, sizeof buf, [&](const os::error& e, std::size_t) { ++total; cancelled += os::is_cancelled(e); });
        r.accept(*ls2, [&](const os::error& e, os::socket) { ++total; cancelled += os::is_cancelled(e); });
        r.after(3600ull * 1000 * ms, [&](const os::error& e) { ++total; cancelled += os::is_cancelled(e); });
        r.on_event(*ev, [&](const os::error& e) { ++total; cancelled += os::is_cancelled(e); });
        r.offload([&] { while (!release.load()) os::sleep_ns(ms); }, [&](const os::error& e) { ++total; cancelled += os::is_cancelled(e); });
        r.run_for(10);
        r.cancel_all(); release.store(true);
        watchdog wd(r); wd.done();
        CHECK_MSG(total == 5 && cancelled == 5, "total %d cancelled %d", total, cancelled); CHECK(r.pending() == 0);
    }
    (void)on_wine();
    f->close(); os::remove_all(*d);
}
