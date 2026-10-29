// forge::os extension modules: one test per feature, run on whatever OS this is.
// Child-process scenarios (crash dump, sandbox, cross-process shared memory) re-exec this binary with
// FORGE_OS_EXT_CHILD=<mode> and are dispatched from a static initializer before the test runner starts.
#include "check.hpp"
#include <forge/os/all.hpp>
#include <atomic>
using namespace forge;

static std::string scratch(const char* tag) { auto d = os::join(os::temp_dir(), std::string("forge_osx_") + tag + "_" + std::to_string(os::pid())); os::remove_all(d); os::mkdirs(d); return d; }
static void skip(const char* what, const std::string& why) { std::printf("  skip %s: %s\n", what, why.c_str()); }

// ---------------------------------------------------------------- child modes
static int child_main(const std::string& mode) {
    auto arg = mode.substr(mode.find(':') == std::string::npos ? mode.size() : mode.find(':') + 1);
    if (mode.rfind("crash", 0) == 0) {
        os::crash_handler::config c; c.dir = arg; c.app = "osx"; if (!os::crash_handler::install(c)) return 2;
        volatile int* p = nullptr; *p = 42; return 3;
    }
    if (mode.rfind("shm", 0) == 0) {
        auto s = os::shared_memory::open(arg); if (!s) return 2;
        std::memcpy(s->data() + 64, "child-was-here", 15); return 0;
    }
    os::sandbox_policy p;
    if (mode == "sandbox_net") p.deny_network = true;
    else if (mode == "sandbox_exec") p.deny_exec = true;
    else if (mode.rfind("sandbox_write", 0) == 0) p.deny_fs_write = true;
    else return 9;
    auto st = os::apply_sandbox(p);
    if (!st) { std::fprintf(stderr, "sandbox: %s\n", st.err.what.c_str()); return os::is_unsupported(st.err) ? 77 : 2; }
    if (p.deny_network) {
        auto r = os::socket::listen("127.0.0.1", 0); if (r) return 4;   // must be refused
#if !defined(_WIN32)
        int u = ::socket(AF_UNIX, SOCK_STREAM, 0); if (u < 0) return 5; ::close(u);
#endif
        return 0;
    }
    if (p.deny_exec) { auto r = os::run({os::exe_path()}); return r && r->exit_code == 0 ? 4 : 0; }
    if (p.deny_fs_write) {
        if (os::write_file(os::join(arg, "nope.txt"), "x")) return 4;
        if (!os::read_file(os::exe_path())) return 5;
        return 0;
    }
    return 9;
}
static const int child_dispatch = [] {
    auto m = os::getenv("FORGE_OS_EXT_CHILD"); if (m.empty()) return 0;
    os::unsetenv("FORGE_OS_EXT_CHILD"); std::fflush(stdout); std::exit(child_main(m));
}();
static int run_child(const std::string& mode) {
    os::setenv("FORGE_OS_EXT_CHILD", mode); auto r = os::run({os::exe_path()}, true); os::unsetenv("FORGE_OS_EXT_CHILD");
    if (!r) { std::printf("  run child failed: %s\n", r.err.what.c_str()); return -1; }
    if (!r->out.empty()) std::printf("  child[%s]: %s", mode.c_str(), r->out.c_str());
    return r->exit_code;
}

// ---------------------------------------------------------------- 1 NUMA
TEST(numa) {
    auto ns = os::numa_nodes(); CHECK(!ns.empty());
    for (auto& n : ns) std::printf("  node %u: %zu cpus, %llu MiB\n", n.id, n.cpus.size(), (unsigned long long)(n.total_bytes >> 20));
    CHECK(!ns[0].cpus.empty());
    auto a = os::numa_alloc(1 << 20, ns[0].id); CHECK(a);
    if (a) { std::memset(a.value, 7, 1 << 20); int nd = os::numa_node_of_address(a.value); CHECK(nd == -1 || nd == int(ns[0].id)); os::numa_free(a.value, 1 << 20); }
    std::thread t([&] { CHECK(os::bind_thread_to_node(ns[0].id)); int c = os::current_numa_node(); CHECK(c == -1 || c == int(ns[0].id)); }); t.join();
    CHECK(!os::bind_thread_to_node(9999));
}
// ---------------------------------------------------------------- 2 huge pages
TEST(huge_pages) {
    auto sz = os::huge_page_sizes(); std::printf("  sizes:"); for (auto s : sz) std::printf(" %s", os::detail::format_size(s).c_str()); std::printf(" thp=%d\n", int(os::thp_enabled()));
    auto r = os::huge_alloc(3 << 20, os::page_kind::huge_2m); CHECK(r); CHECK(r->size >= (3u << 20));
    if (r) { std::printf("  2M request -> %s\n", os::page_kind_name(r->kind)); std::memset(r->ptr, 1, r->size); os::huge_free(r.value); CHECK(!r->ptr); }
    auto g = os::huge_alloc(1 << 20, os::page_kind::huge_1g, false);   // strict: may legitimately fail
    std::printf("  strict 1G -> %s\n", g ? os::page_kind_name(g->kind) : g.err.what.c_str());
    if (g) os::huge_free(g.value);
    auto m = os::mapping::anonymous(4 << 20); CHECK(m);
    auto a = os::advise_huge(m->data(), m->size()); if (!a) std::printf("  advise_huge: %s\n", a.err.what.c_str());
}
// ---------------------------------------------------------------- 3 async file I/O
static void aio_roundtrip(os::async_file_io& io, const std::string& path) {
    const std::size_t blk = 64 << 10, nblk = 16;
    std::vector<std::uint8_t> src(blk * nblk); for (std::size_t i = 0; i < src.size(); ++i) src[i] = std::uint8_t(i * 31 + 7);
    auto f = os::file::open(path, os::rdwr | os::create | os::truncate); CHECK(f); if (!f) return;
    for (std::size_t i = 0; i < nblk; ++i) CHECK(io.write(f.value, src.data() + i * blk, blk, i * blk, 100 + i));
    std::size_t got = 0; while (got < nblk) { auto c = io.wait(1, 5000); if (c.empty()) break; for (auto& x : c) { CHECK(x.result == std::int64_t(blk)); CHECK(x.user >= 100 && x.user < 100 + nblk); } got += c.size(); }
    CHECK(got == nblk); CHECK(f->sync());
    std::vector<std::uint8_t> dst(src.size());
    for (std::size_t i = 0; i < nblk; ++i) CHECK(io.read(f.value, dst.data() + i * blk, blk, i * blk, i));
    got = 0; while (got < nblk) { auto c = io.wait(nblk - got, 5000); if (c.empty()) break; for (auto& x : c) CHECK(x.result == std::int64_t(blk)); got += c.size(); }
    CHECK(got == nblk); CHECK(dst == src); CHECK(io.in_flight() == 0);
    CHECK(io.wait(1, 10).empty());
}
TEST(async_file_io) {
    auto d = scratch("aio");
    { os::async_file_io io; std::printf("  backend %s\n", io.backend_name()); aio_roundtrip(io, os::join(d, "a.bin")); }
    os::setenv("FORGE_AIO", "threads");
    { os::async_file_io io; CHECK(io.backend() == os::aio_backend::threads); aio_roundtrip(io, os::join(d, "b.bin")); }
    os::unsetenv("FORGE_AIO"); os::remove_all(d);
}
// ---------------------------------------------------------------- 4 CPU topology
TEST(cpu_topology) {
    auto t = os::cpu_topology_query();
    std::printf("  %s: %u logical, %u physical, %u pkg, hybrid=%d P=%zu E=%zu\n", t.brand.c_str(), t.logical, t.physical, t.packages, int(t.hybrid),
                t.of_kind(os::core_kind::performance).size(), t.of_kind(os::core_kind::efficiency).size());
    for (auto& c : t.caches) if (c.shared_cpus.empty() || c.shared_cpus[0] == t.cpus[0].id) std::printf("  L%u%c %s line %u\n", c.level, c.type, os::detail::format_size(c.size).c_str(), c.line);
    CHECK(t.logical >= 1); CHECK(t.physical >= 1 && t.physical <= t.logical); CHECK(t.packages >= 1);
    CHECK(t.logical == t.cpus.size());
    for (auto& c : t.cpus) CHECK(c.kind != os::core_kind::unknown);
#if defined(__x86_64__) || defined(__linux__) || defined(_WIN32)
    CHECK(!t.caches.empty()); CHECK(t.cache_size(1, 'D') > 0);
#endif
    std::thread th([] { auto s = os::pin_thread_to_kind(os::core_kind::performance); CHECK(s || os::is_unsupported(s.err)); }); th.join();
    CHECK(os::detail::parse_cpulist("0-2,5,7-8") == (std::vector<unsigned>{0, 1, 2, 5, 7, 8}));
}
// ---------------------------------------------------------------- 5 power / thermal
TEST(power_thermal) {
    auto p = os::power_status();
    std::printf("  [%s] battery=%d %d%% on_battery=%d charging=%d low_power=%d thermal=%s throttled=%d temp=%.1fC\n", p.source.c_str(), int(p.has_battery), p.battery_percent,
                int(p.on_battery), int(p.charging), int(p.low_power_mode), os::thermal_name(p.thermal), int(p.throttled), p.temperature_c);
    CHECK(p.battery_percent >= -1 && p.battery_percent <= 100);
    os::power_state h; h.has_battery = true; h.battery_percent = 12; h.low_power_mode = true; h.thermal = os::thermal_state::serious; os::power_hooks::report(h);
    auto q = os::power_status(); CHECK(q.battery_percent == 12 && q.low_power_mode && q.thermal == os::thermal_state::serious && q.source == "hook");
    os::power_hooks::clear(); CHECK(os::power_status().source != "hook");
}
// ---------------------------------------------------------------- 6 memory pressure
TEST(memory_pressure) {
    auto& mp = os::memory_pressure::instance();
    std::atomic<int> seen{0}; std::atomic<int> last{-1};
    int id = mp.subscribe([&](os::pressure_level l) { ++seen; last = int(l); });
    os::sleep_ns(50000000);
    std::printf("  source %s, sampled %s\n", mp.source().c_str(), os::pressure_name(os::memory_pressure::sample()));
    int s0 = seen;
    mp.notify(os::pressure_level::moderate); CHECK(seen == s0 + 1); CHECK(last == int(os::pressure_level::moderate));
    os::memory_pressure::on_trim_memory(80); CHECK(last == int(os::pressure_level::critical));
    os::memory_pressure::on_trim_memory(5); CHECK(last == int(os::pressure_level::moderate));
    os::memory_pressure::on_trim_memory(20); CHECK(last == int(os::pressure_level::normal));
    mp.unsubscribe(id); int s1 = seen; mp.notify(os::pressure_level::critical); CHECK(seen == s1);
    mp.stop();
}
// ---------------------------------------------------------------- 7 shared memory
TEST(shared_memory) {
    std::string name = "osx" + std::to_string(os::pid());
    auto a = os::shared_memory::create(name, 1 << 16); CHECK_MSG(a, "%s", a.err.what.c_str()); if (!a) return;
    std::memcpy(a->data(), "hello", 6);
    { auto b = os::shared_memory::open(name, 1 << 16); CHECK(b); if (b) { CHECK(std::string(reinterpret_cast<char*>(b->data())) == "hello"); b->data()[10] = 'Z'; } }
    CHECK(a->data()[10] == 'Z');
#if !defined(__APPLE__) || !TARGET_OS_IPHONE
    int rc = run_child("shm:" + name); CHECK(rc == 0);
    CHECK(std::string(reinterpret_cast<char*>(a->data() + 64)) == "child-was-here");
#endif
    a->close(); CHECK(!os::shared_memory::open("osx-missing-" + std::to_string(os::pid())));
}
// ---------------------------------------------------------------- 8 GPU discovery
TEST(gpu_discovery) {
    auto g = os::gpu_devices();
    for (auto& d : g) std::printf("  %s [%04x:%04x] %s driver=%s bus=%s vram=%llu MiB%s\n", d.name.c_str(), d.vendor_id, d.device_id, d.vendor.c_str(), d.driver.c_str(), d.bus.c_str(),
                                  (unsigned long long)(d.dedicated_memory >> 20), d.integrated ? " (integrated)" : "");
    if (g.empty()) std::printf("  no GPUs visible\n");
    for (auto& d : g) CHECK(!d.name.empty());
    CHECK(std::string(os::gpu_vendor_name(0x10de)) == "NVIDIA");
}
// ---------------------------------------------------------------- 9 local IPC
TEST(local_ipc) {
    auto ep = os::local_endpoint("osx" + std::to_string(os::pid()));
    auto l = os::local_listener::listen(ep); CHECK_MSG(l, "%s", l.err.what.c_str()); if (!l) return;
    std::thread cl([&] {
        auto c = os::local_stream::connect(ep); CHECK(c); if (!c) return;
        CHECK(c->write_all("ping", 4)); char b[5] = {}; CHECK(c->read_exact(b, 4)); CHECK(std::string(b) == "pong");
    });
    auto s = l->accept(5000); CHECK_MSG(s, "%s", s.err.what.c_str());
    if (s) { char b[5] = {}; CHECK(s->read_exact(b, 4)); CHECK(std::string(b) == "ping"); CHECK(s->write_all("pong", 4)); }
    cl.join();
    auto t = l->accept(50); CHECK(!t);   // timeout
#if defined(__linux__)
    auto ab = os::local_listener::listen("@forge-osx-" + std::to_string(os::pid())); CHECK(ab);
#endif
}
// ---------------------------------------------------------------- 10 crash handling
static __attribute__((noinline)) std::vector<os::stack_frame> deep_trace(int n) { if (n) { auto v = deep_trace(n - 1); asm volatile(""); return v; } return os::stack_trace(); }
TEST(crash_handling) {
    auto st = deep_trace(3); CHECK(st.size() >= 3);
    std::printf("%s", os::format_stack_trace(std::vector<os::stack_frame>(st.begin(), st.begin() + std::ptrdiff_t(std::min<std::size_t>(st.size(), 5)))).c_str());
#if !defined(__APPLE__) || !TARGET_OS_IPHONE
    auto d = scratch("crash");
    int rc = run_child("crash:" + d); CHECK(rc != 0);
    auto l = os::list_dir(d); bool report = false, dump = false;
    if (l) for (auto& e : l.value) {
        auto p = os::join(d, e.name);
        if (e.name.size() > 6 && e.name.substr(e.name.size() - 6) == ".crash") { auto t = os::read_file(p); report = t && (t->find("SIGSEGV") != std::string::npos || t->find("exception 0xc0000005") != std::string::npos) && t->find("backtrace") != std::string::npos; }
        if (e.name.size() > 4 && e.name.substr(e.name.size() - 4) == ".dmp") { auto t = os::read_file(p); dump = t && t->size() > 32 && t->compare(0, 4, "MDMP") == 0; }
    }
    CHECK(report);
#if defined(_WIN32)
    CHECK(dump);
#else
    (void)dump;
#endif
    os::remove_all(d);
#endif
}
// ---------------------------------------------------------------- 11 sandboxing
TEST(sandboxing) {
    auto s = os::sandbox_status(); std::printf("  current: sandboxed=%d %s\n", int(s.sandboxed), s.mechanism.c_str());
#if !defined(__APPLE__) || !TARGET_OS_IPHONE
    auto d = scratch("sbx");
    for (std::string m : {std::string("sandbox_net"), std::string("sandbox_exec"), "sandbox_write:" + d}) {
        int rc = run_child(m);
        if (rc == 77) { skip(m.c_str(), "unsupported here"); continue; }
#if defined(__FreeBSD__)
        if (rc != 0) { skip(m.c_str(), "capsicum is all-or-nothing"); continue; }
#endif
        CHECK_MSG(rc == 0, "%s rc=%d", m.c_str(), rc);
    }
    os::remove_all(d);
#endif
}
// ---------------------------------------------------------------- 12 secure key storage
TEST(key_storage) {
    auto d = scratch("keys"); os::setenv("FORGE_KEYSTORE_DIR", d);
    std::printf("  default backend %s\n", os::keystore::backend().c_str());
    std::string svc = "forge-test-" + std::to_string(os::pid()), secret = std::string("s3cr\0et\xff", 8);
    auto p = os::keystore::put(svc, "alice", secret);
    if (p) { auto g = os::keystore::get(svc, "alice"); CHECK(g); CHECK(g.value == secret); CHECK(os::keystore::remove(svc, "alice")); CHECK(!os::keystore::get(svc, "alice")); }
    else skip("default keystore", p.err.what);
    os::setenv("FORGE_KEYSTORE", "file"); CHECK(os::keystore::backend() == "file");
    CHECK(os::keystore::put(svc, "bob", secret)); CHECK(os::keystore::put(svc, "bob", "v2"));
    auto g = os::keystore::get(svc, "bob"); CHECK(g && g.value == "v2");
    auto raw = os::list_dir(d); bool plain = false;
    if (raw) for (auto& e : raw.value) { auto t = os::read_file(os::join(d, e.name)); if (t && t->find("v2") != std::string::npos && e.name != ".pad") plain = true; }
    CHECK(!plain);
    CHECK(os::keystore::remove(svc, "bob")); CHECK(!os::keystore::get(svc, "bob"));
    os::unsetenv("FORGE_KEYSTORE");
    std::map<std::string, std::string> mem;
    os::keystore::set_backend({"mem", [&](auto& s, auto& a, auto& v) { mem[s + "/" + a] = v; return os::status{}; },
        [&](auto& s, auto& a) { auto it = mem.find(s + "/" + a); return it == mem.end() ? os::result<std::string>{{}, {2, "none"}} : os::result<std::string>{it->second, {}}; },
        [&](auto& s, auto& a) { mem.erase(s + "/" + a); return os::status{}; }});
    CHECK(os::keystore::backend() == "mem"); CHECK(os::keystore::put("x", "y", "z")); CHECK(mem["x/y"] == "z"); CHECK(os::keystore::get("x", "y").value == "z");
    os::keystore::reset_backend(); os::unsetenv("FORGE_KEYSTORE_DIR"); os::remove_all(d);
}
// ---------------------------------------------------------------- 13 file watching
static bool saw(const std::vector<os::watch_event>& v, const std::string& name) { for (auto& e : v) if (os::basename(e.path) == name) return true; return false; }
static std::vector<os::watch_event> collect(os::file_watcher& w, const std::string& name, int ms) {
    std::vector<os::watch_event> all; auto end = os::monotonic_ns() + std::uint64_t(ms) * 1000000ull;
    while (os::monotonic_ns() < end) { auto e = w.poll(100); all.insert(all.end(), e.begin(), e.end()); if (saw(all, name)) break; }
    return all;
}
TEST(file_watching) {
    auto d = scratch("watch"); auto f = os::join(d, "model.fpk"); CHECK(os::write_file(f, "v1"));
    os::file_watcher w; std::printf("  backend %s\n", w.backend());
    CHECK(w.add(f));
    CHECK(os::write_file(f + ".tmp", "v2")); CHECK(os::rename(f + ".tmp", f));   // atomic replace
    auto ev = collect(w, "model.fpk", 3000); CHECK(saw(ev, "model.fpk"));
    for (auto& e : ev) std::printf("  %s %s\n", os::watch_kind_name(e.kind), os::basename(e.path).c_str());
    { auto fh = os::file::open(f, os::wr | os::append); CHECK(fh); fh->write_all("!", 1); }
    CHECK(saw(collect(w, "model.fpk", 3000), "model.fpk"));
    os::file_watcher wd; CHECK(wd.add(d)); CHECK(os::write_file(os::join(d, "new.txt"), "x"));
    auto ev2 = collect(wd, "new.txt", 3000);
#if defined(__linux__) || defined(_WIN32)
    CHECK(saw(ev2, "new.txt"));
#else
    CHECK(!ev2.empty());   // kqueue reports the directory itself
#endif
    CHECK(!w.add(os::join(d, "no/such/dir/x")));
    os::remove_all(d);
}
// ---------------------------------------------------------------- 14 timers + perf counters
TEST(timers_and_perf) {
    auto c0 = os::cycle_counter(); volatile double x = 1; for (int i = 0; i < 100000; ++i) x = x * 1.0000001 + 1e-9; auto c1 = os::cycle_counter();
    CHECK(c1 > c0); auto hz = os::counter_hz(); CHECK(hz > 1000000);
    std::printf("  counter %.1f MHz, timer res %llu ns\n", double(hz) / 1e6, (unsigned long long)os::timer_resolution_ns());
    auto t0 = os::monotonic_ns(); os::precise_sleep_ns(2000000); auto dt = os::monotonic_ns() - t0;
    std::printf("  precise_sleep 2ms -> %.3f ms\n", double(dt) / 1e6); CHECK(dt >= 2000000 && dt < 50000000);
    os::perf_counters pc; auto o = pc.open({os::perf_event::cycles, os::perf_event::instructions});
    if (!o) { skip("hw counters", o.err.what); o = pc.open({os::perf_event::cycles}); }
    if (o) {
        pc.start(); for (int i = 0; i < 1000000; ++i) x = x * 1.0000001 + 1e-9; auto r = pc.stop();
        for (auto& v : r) std::printf("  [%s] %s = %llu%s\n", pc.backend(), os::perf_event_name(v.event), (unsigned long long)v.value, v.valid ? "" : " (invalid)");
        CHECK(!r.empty() && r[0].valid && r[0].value > 0);
    } else skip("cycle counter", o.err.what);
}
// ---------------------------------------------------------------- 15 background execution
TEST(background_execution) {
    auto t = os::background::begin("forge-test");
    if (!t) { skip("background", t.err.what); CHECK(os::is_unsupported(t.err)); }
    else { std::printf("  kind %s\n", os::background::kind(t.value).c_str()); CHECK(os::background::active() == 1); CHECK(os::background::end(t.value)); CHECK(os::background::active() == 0); CHECK(!os::background::end(t.value)); }
    bool began = false, ended = false;
    os::background::set_hooks({[&](auto&, auto&, auto&) { began = true; return true; }, [&](auto&) { ended = true; }});
#if defined(__ANDROID__)
    { os::background_scope s("hooked"); CHECK(began && s.token); } CHECK(ended);
#else
    (void)began; (void)ended;
#endif
}
// ---------------------------------------------------------------- 16 service integration
TEST(service_integration) {
    std::printf("  manager '%s'\n", os::service::manager().c_str());
    CHECK(os::service::ready());   // no-op outside a service manager
#if defined(__linux__)
    auto path = os::join(os::temp_dir(), "osx-notify-" + std::to_string(os::pid()));
    int fd = ::socket(AF_UNIX, SOCK_DGRAM, 0); sockaddr_un a{}; a.sun_family = AF_UNIX; std::strcpy(a.sun_path, path.c_str()); ::unlink(path.c_str());
    CHECK(::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
    os::setenv("NOTIFY_SOCKET", path); os::setenv("WATCHDOG_USEC", "2000000");
    CHECK(os::service::ready("loaded")); char b[256] = {}; auto n = ::recv(fd, b, sizeof b - 1, 0);
    CHECK(n > 0 && std::string(b) == "READY=1\nSTATUS=loaded"); CHECK(os::service::watchdog_interval_us() == 1000000);
    CHECK(os::service::watchdog()); std::memset(b, 0, sizeof b); ::recv(fd, b, sizeof b - 1, 0); CHECK(std::string(b) == "WATCHDOG=1");
    CHECK(os::service::manager() == "systemd" || os::platform_name() == std::string("android"));
    os::unsetenv("NOTIFY_SOCKET"); os::unsetenv("WATCHDOG_USEC"); ::close(fd); ::unlink(path.c_str());
#endif
    auto l = os::service::log(os::log_level::info, "forge test_os_ext service log line");
    if (!l) skip("service log", l.err.what);
    int ran = 0; auto r = os::service::run_as_service("forge-test", [&] { ++ran; return 0; }, [] {});
    CHECK(r); CHECK(ran == 1);
}
