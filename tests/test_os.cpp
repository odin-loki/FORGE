// forge::os: the unified OS layer, exercised on whatever OS this runs on.
#include "check.hpp"
#include <forge/os/os.hpp>
#include <atomic>
using namespace forge;

static std::string scratch() { auto d = os::join(os::temp_dir(), "forge_os_" + std::to_string(os::pid())); os::remove_all(d); CHECK(os::mkdirs(d)); return d; }

TEST(identity_and_sysinfo) {
    auto s = os::sysinfo();
    std::printf("  %s/%s cpus=%u ram=%llu MiB avail=%llu MiB page=%zu host=%s\n", s.os.c_str(), s.arch.c_str(), s.logical_cpus,
                (unsigned long long)(s.total_ram >> 20), (unsigned long long)(s.avail_ram >> 20), s.page, s.hostname.c_str());
    CHECK(s.logical_cpus >= 1); CHECK(s.total_ram > 0); CHECK(s.page >= 4096); CHECK(s.os != "other");
}
TEST(files_and_fs) {
    auto d = scratch(); auto p = os::join(d, "a.bin");
    CHECK(os::write_file(p, "hello world"));
    auto r = os::read_file(p); CHECK(r); CHECK(r.value == "hello world");
    auto st = os::stat(p); CHECK(st.exists && st.is_file && st.size == 11 && st.mtime_ns > 0);
    { auto f = os::file::open(p, os::rdwr); CHECK(f); char b[5] = {}; CHECK(f->pread(b, 5, 6).value == 5); CHECK(std::string(b, 5) == "world");
      CHECK(f->pwrite("W", 1, 6)); CHECK(f->resize(8)); CHECK(f->sync()); CHECK(f->size().value == 8); }
    CHECK(os::read_file(p).value == "hello Wo");
    { auto f = os::file::open(p, os::wr | os::append); CHECK(f); CHECK(f->write_all("!!", 2)); }
    CHECK(os::read_file(p).value == "hello Wo!!");
    CHECK(!os::file::open(p, os::wr | os::excl));
    CHECK(os::mkdirs(os::join(d, "x/y/z")));
    CHECK(os::rename(p, os::join(d, "x/b.bin")));
    auto l = os::list_dir(os::join(d, "x")); CHECK(l); CHECK(l.value.size() == 2);
    bool sawdir = false; for (auto& e : l.value) if (e.name == "y") sawdir = e.is_dir; CHECK(sawdir);
    auto bad = os::file::open(os::join(d, "nope"), os::rd); CHECK(!bad); CHECK(!bad.err.what.empty());
    CHECK(os::remove_all(d)); CHECK(!os::exists(d));
}
TEST(paths) {
    CHECK(os::basename("a/b/c.txt") == "c.txt"); CHECK(os::dirname("a/b/c.txt") == "a/b"); CHECK(os::dirname("c") == ".");
    CHECK(os::join("a", "b") == "a/b"); CHECK(os::join("a/", "b") == "a/b");
    CHECK(!os::cwd().empty()); CHECK(!os::exe_path().empty()); CHECK(os::exists(os::exe_path()));
    CHECK(!os::temp_dir().empty()); CHECK(os::stat(os::temp_dir()).is_dir);
}
TEST(mmap_and_memory) {
    auto d = scratch(); auto p = os::join(d, "m.bin");
    CHECK(os::write_file(p, std::string(10000, 'x')));
    { auto m = os::mapping::map_file(p, os::prot_read | os::prot_write); CHECK(m); CHECK(m->size() == 10000); m->data()[5] = 'Y'; CHECK(m->flush()); }
    { auto m = os::mapping::map_file(p); CHECK(m); CHECK(m->data()[5] == 'Y' && m->data()[9999] == 'x'); }
    auto a = os::mapping::anonymous(1 << 20); CHECK(a); a->data()[(1 << 20) - 1] = 7; CHECK(a->data()[(1 << 20) - 1] == 7);
    std::size_t pg = os::page_size(); void* r = os::reserve(pg * 16); CHECK(r);
    CHECK(os::commit(r, pg * 2)); static_cast<char*>(r)[pg] = 1; CHECK(os::protect(r, pg * 2, os::prot_read)); os::release(r, pg * 16);
    void* al = os::aligned_alloc(4096, 256); CHECK(al && (reinterpret_cast<std::uintptr_t>(al) % 256) == 0); os::aligned_free(al);
    char buf[4096]; os::lock_memory(buf, sizeof buf); os::unlock_memory(buf, sizeof buf);   // may fail without privilege; must not crash
    os::remove_all(d);
}
TEST(time_and_sleep) {
    auto t0 = os::monotonic_ns(); os::sleep_ns(20'000'000); auto dt = os::monotonic_ns() - t0;
    CHECK(dt >= 15'000'000 && dt < 2'000'000'000);
    CHECK(os::wall_ns() > 1'700'000'000'000'000'000LL);
    volatile double x = 0; for (int i = 0; i < 2'000'000; ++i) x = x + i; CHECK(os::cpu_time_ns() > 0);
}
TEST(threads_and_sync) {
    os::mutex m; os::condvar cv; int counter = 0; std::atomic<std::uint64_t> other{0};
    { std::vector<os::thread> ts;
      for (int t = 0; t < 4; ++t) ts.emplace_back([&] { other = os::thread_id(); for (int i = 0; i < 10000; ++i) { os::lock_guard g(m); ++counter; } }, "forge-w");
    }
    CHECK(counter == 40000); CHECK(other != 0 && other != os::thread_id());
    os::semaphore s; os::thread t([&] { os::sleep_ns(5'000'000); s.post(); });
    CHECK(s.wait_for(2'000'000'000)); CHECK(!s.wait_for(10'000'000)); t.join();
    { os::lock_guard g(m); CHECK(!cv.wait_for(m, 5'000'000)); }
    os::set_thread_name("forge-test"); os::set_thread_priority(os::thread_priority::normal); os::yield();
}
TEST(env_and_dl) {
    CHECK(os::setenv("FORGE_OS_T", "v1")); CHECK(os::getenv("FORGE_OS_T") == "v1"); CHECK(os::has_env("FORGE_OS_T"));
    CHECK(os::unsetenv("FORGE_OS_T")); CHECK(os::getenv("FORGE_OS_T", "dflt") == "dflt");
    auto self = os::library::open(""); CHECK(self);
#if defined(_WIN32)
    auto k = os::library::open("kernel32.dll"); CHECK(k); CHECK(k->symbol("GetTickCount") != nullptr);
#elif defined(__APPLE__)
    auto k = os::library::open("/usr/lib/libSystem.B.dylib"); CHECK(k); CHECK(k->symbol("malloc") != nullptr);
#else
    auto k = os::library::open(
#if defined(__ANDROID__) || defined(__FreeBSD__)
        "libc.so"
#else
        "libc.so.6"
#endif
    ); CHECK(k); CHECK(k->symbol("malloc") != nullptr);
#endif
    CHECK(!os::library::open("definitely_missing_forge_lib"));
    CHECK(os::library::decorate("x").find('x') != std::string::npos);
}
TEST(entropy) {
    std::uint8_t a[64] = {}, b[64] = {}; CHECK(os::entropy(a, 64)); CHECK(os::entropy(b, 64)); CHECK(std::memcmp(a, b, 64) != 0);
}
static std::atomic<int> g_sig{0};
TEST(signals) {
    CHECK(os::on_signal(os::sig::interrupt, [](int) { g_sig = 1; }));
    os::ignore_sigpipe();
#if !defined(_WIN32)
    ::raise(SIGINT); CHECK(g_sig == 1);
#endif
    os::on_signal(os::sig::interrupt, nullptr);
}
TEST(process) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    CHECK(!os::run({"true"}));
#elif defined(_WIN32)
    auto r = os::run({"cmd", "/c", "echo forge&& exit 3"}); CHECK(r); CHECK(r->exit_code == 3); CHECK(r->out.find("forge") != std::string::npos);
#else
    auto r = os::run({"sh", "-c", "echo forge; echo err >&2; exit 3"}, true); CHECK(r); CHECK(r->exit_code == 3);
    CHECK(r->out.find("forge") != std::string::npos && r->out.find("err") != std::string::npos);
    CHECK(!os::run({"/nonexistent/forge-binary"}) || os::run({"/nonexistent/forge-binary"})->exit_code != 0);
#endif
    CHECK(os::pid() > 0);
}
TEST(sockets_and_poller) {
    auto l = os::socket::listen("127.0.0.1", 0); CHECK(l); int port = l->local_port(); CHECK(port > 0);
    os::poller p; std::printf("  poller backend: %s\n", p.backend());
    CHECK(p.add(l->native(), os::readable, &*l));
    auto c = os::socket::connect("127.0.0.1", port); CHECK(c);
    auto ev = p.wait(2000); CHECK(ev.size() == 1 && (ev[0].events & os::readable) && ev[0].user == &*l);
    auto s = l->accept(); CHECK(s);
    CHECK(p.add(s->native(), os::readable)); CHECK(c->send_all("ping", 4));
    ev = p.wait(2000); bool got = false; for (auto& e : ev) if (e.fd == s->native() && (e.events & os::readable)) got = true; CHECK(got);
    char b[4]; CHECK(s->recv_all(b, 4)); CHECK(std::memcmp(b, "ping", 4) == 0);
    CHECK(p.modify(s->native(), os::writable)); ev = p.wait(1000); CHECK(!ev.empty() && (ev[0].events & os::writable));
    CHECK(p.remove(s->native())); CHECK(p.size() == 1);
    CHECK(p.wait(10).empty());
    c->close(); CHECK(s->wait_readable(2000)); CHECK(s->recv(b, 4) == 0);
    CHECK(!os::socket::connect("127.0.0.1", 1, 300));
}
