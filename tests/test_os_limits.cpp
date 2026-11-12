// forge::os wave 4, processes and files: limit_group (cgroup v2 / job object / rctl / rlimit fallback),
// mapped_file::resize, named FIFOs, clone_file on Windows.
#include <forge/os/process_limits.hpp>
#include <forge/os/fs.hpp>
#include <forge/os/pipe.hpp>
#include <thread>
#include "check.hpp"
using namespace forge;
namespace {
bool on_wine() {
#if defined(_WIN32)
    return GetProcAddress(GetModuleHandleA("ntdll.dll"), "wine_get_version") != nullptr;
#else
    return false;
#endif
}
std::vector<std::string> shell(const std::string& script) {
#if defined(_WIN32)
    return {"cmd", "/c", script};
#else
    return {"sh", "-c", script};
#endif
}
}

TEST(limit_group_runs_children) {
    os::process_limits lim; lim.memory_bytes = 1ull << 30; lim.max_processes = 64;
    auto g = os::limit_group::create(lim, "forge test");
#if defined(__APPLE__) || defined(__ANDROID__)
    CHECK(!g); CHECK(os::is_unsupported(g.err));   // no mechanism an application may use
    auto plain = os::limit_group::create({});      // membership only
#if defined(FORGE_OS_IOS)
    CHECK(!plain); CHECK(os::is_unsupported(plain.err));
#else
    CHECK(plain); if (plain) CHECK(std::string(plain->backend()) == "rlimit");
#endif
#else
    CHECK_MSG(bool(g), "%s", g.err.what.c_str()); if (!g) return;
    std::string be = g->backend();
    std::printf("  backend %s, enforced 0x%x%s%s\n", be.c_str(), g->enforced(), g->fallback_reason().empty() ? "" : ", fallback: ", g->fallback_reason().c_str());
    CHECK(be == "cgroup2" || be == "job" || be == "rctl" || be == "rlimit"); CHECK(g->valid()); CHECK(g->attach() != nullptr);
    if (be == "cgroup2" || be == "job") CHECK(g->enforced() & os::limit_group_wide);
    if (be != "rlimit") CHECK(g->fallback_reason().empty());
    CHECK(g->enforced() & os::limit_memory);
    // a child spawned into the group runs normally
    os::spawn_spec sp; sp.argv = shell("echo inside"); sp.timeout_ms = 30000;
    auto r = g->run(sp);
    if (!r && on_wine()) { std::printf("  skip: wine refuses the nested job: %s\n", r.err.what.c_str()); return; }
    CHECK_MSG(bool(r), "%s", r.err.what.c_str());
    if (r) { CHECK_MSG(r->exit_code == 0, "exit %d", r->exit_code); CHECK(r->out.find("inside") != std::string::npos); }
    // kill() ends a long-running member; usage() sees it while it lives
    os::spawn_spec slow;
#if defined(_WIN32)
    slow.argv = {"cmd", "/c", "ping -n 60 127.0.0.1 >NUL"};
#else
    slow.argv = {"sleep", "60"};
#endif
    auto c = g->start(slow);
    if (!c && on_wine()) { std::printf("  skip: wine has no nested jobs: %s\n", c.err.what.c_str()); return; }
    CHECK_MSG(bool(c), "%s", c.err.what.c_str()); if (!c) return;
    os::sleep_ns(50'000'000);
    auto u = g->usage(); CHECK_MSG(bool(u), "%s", u.err.what.c_str());
    if (u) { std::printf("  usage: %llu processes, %llu bytes, %llu ns cpu\n", static_cast<unsigned long long>(u->processes), static_cast<unsigned long long>(u->memory_bytes), static_cast<unsigned long long>(u->cpu_ns)); if (!on_wine()) CHECK(u->processes >= 1); }
    auto t0 = os::monotonic_ns(); CHECK(g->kill());
    int code = c->wait(20000);
    if (on_wine() && code == -1) { std::printf("  skip: wine did not terminate the job\n"); c->kill(); c->wait(5000); return; }
    CHECK_MSG(code != 0 && os::monotonic_ns() - t0 < 15'000'000'000ull, "exit %d", code);
    // update() and close() are safe; a closed group refuses work
    lim.memory_bytes = 2ull << 30; CHECK(g->update(lim));
    g->close(); g->close(); CHECK(!g->valid()); CHECK(g->attach() == nullptr); CHECK(!g->add(os::pid()));
    // allow_fallback = false: either a real mechanism, or a clean unsupported
    os::process_limits strict; strict.memory_bytes = 1ull << 30; strict.allow_fallback = false;
    auto s = os::limit_group::create(strict);
    if (!s) CHECK_MSG(os::is_unsupported(s.err), "%s", s.err.what.c_str()); else CHECK(std::string(s->backend()) != "rlimit");
#endif
}

#if defined(__linux__) && !defined(__ANDROID__)
// The cgroup v2 file protocol against a stand-in hierarchy: which files are written, with what.
TEST(limit_group_cgroup_files) {
    auto d = os::fs::make_temp_dir("forge-cg-"); CHECK(d); if (!d) return;
    os::mkdirs(*d + "/self");
    CHECK(os::write_file(*d + "/cgroup.controllers", "cpuset cpu io memory pids\n"));
    CHECK(os::write_file(*d + "/cgroup.subtree_control", "cpu memory pids\n"));
    os::detail::cg::test_root() = *d;
    {
        os::process_limits lim; lim.memory_bytes = 256ull << 20; lim.memory_soft_bytes = 200ull << 20; lim.cpus = 1.5; lim.cpu_weight = 50; lim.max_processes = 32;
        lim.allow_fallback = false;
        auto g = os::limit_group::create(lim, "my/../svc"); CHECK_MSG(bool(g), "%s", g.err.what.c_str());
        if (g) {
            CHECK(std::string(g->backend()) == "cgroup2");
            CHECK(g->enforced() == (os::limit_memory | os::limit_memory_soft | os::limit_cpu | os::limit_cpu_weight | os::limit_processes | os::limit_group_wide));
            const std::string dir = g->path();
            CHECK(os::dirname(dir) == *d);   // a sibling of our own cgroup, not inside it
            CHECK(os::basename(dir).rfind("my..svc-", 0) == 0);   // the name is reduced to safe characters
            auto rd = [&](const char* f) { return os::detail::trim(os::read_file(dir + "/" + f).value); };
            CHECK(rd("memory.max") == "268435456"); CHECK(rd("memory.high") == "209715200"); CHECK(rd("cpu.max") == "150000 100000");
            CHECK(rd("cpu.weight") == "50"); CHECK(rd("pids.max") == "32"); CHECK(rd("memory.oom.group") == "1");
            CHECK(g->add(4242)); CHECK(rd("cgroup.procs") == "4242");
            CHECK(g->attach()->cgroup_procs_fd >= 0);
            lim.cpus = 0; lim.memory_bytes = 0; lim.max_processes = 0; lim.memory_soft_bytes = 0; CHECK(g->update(lim));
            CHECK(rd("memory.max") == "max"); CHECK(rd("cpu.max") == "max 100000"); CHECK(rd("pids.max") == "max");
            CHECK(g->enforced() == (os::limit_cpu_weight | os::limit_group_wide));
            CHECK(g->kill()); CHECK(rd("cgroup.kill") == "1");
        }
    }
    {   // a parent that does not hand out the memory controller: the reason names it, and the fallback takes over
        CHECK(os::remove(*d + "/cgroup.subtree_control"));   // nothing to enable controllers through
        os::process_limits lim; lim.memory_bytes = 1 << 20; lim.allow_fallback = false;
        auto g = os::limit_group::create(lim);
        CHECK(!g); CHECK(os::is_unsupported(g.err)); CHECK_MSG(g.err.what.find("memory") != std::string::npos, "%s", g.err.what.c_str());
        lim.allow_fallback = true; auto f = os::limit_group::create(lim); CHECK(f);
        if (f) { CHECK(std::string(f->backend()) == "rlimit"); CHECK(!f->fallback_reason().empty()); CHECK(f->enforced() == os::limit_memory); CHECK(f->attach()->use_rlimits); }
    }
    os::detail::cg::test_root().clear(); os::remove_all(*d);
}
#endif

TEST(mapped_file_resize) {
    auto d = os::fs::make_temp_dir("forge-mf-"); CHECK(d); if (!d) return;
    auto path = os::join(*d, "grow.bin");
    {
        auto m = os::fs::mapped_file::open(path, os::rdwr | os::create); CHECK_MSG(bool(m), "%s", m.err.what.c_str()); if (!m) return;
        CHECK(m->size() == 0); CHECK(m->data() == nullptr); CHECK(m->flush());
        CHECK(m->resize(10000)); CHECK(m->size() == 10000); CHECK(m->data() != nullptr);
        for (std::size_t i = 0; i < 10000; ++i) { CHECK(m->data()[i] == 0); m->data()[i] = std::uint8_t(i * 13); if (forge_test::failures()) break; }
        const std::size_t big = 8u << 20;   // far past the first mapping: the view has to move or extend
        CHECK(m->resize(big)); CHECK(m->size() == big);
        bool kept = true, zero = true;
        for (std::size_t i = 0; i < 10000; ++i) kept &= m->data()[i] == std::uint8_t(i * 13);
        for (std::size_t i = 10000; i < big; i += 4093) zero &= m->data()[i] == 0;
        CHECK(kept); CHECK(zero);
        m->data()[big - 1] = 0x5a; CHECK(m->flush());
        CHECK(os::stat(path).size == big);
        CHECK(m->resize(5000)); CHECK(m->size() == 5000); CHECK(os::stat(path).size == 5000);
        kept = true; for (std::size_t i = 0; i < 5000; ++i) kept &= m->data()[i] == std::uint8_t(i * 13);
        CHECK(kept);
        CHECK(m->resize(5000));   // same size: nothing to do
        CHECK(m->resize(0)); CHECK(m->size() == 0); CHECK(m->data() == nullptr); CHECK(os::stat(path).size == 0);
        CHECK(m->resize(4096)); m->data()[0] = 'F'; m->data()[4095] = 'G';
    }   // closed: the bytes are in the file
    auto back = os::read_file(path); CHECK(back); CHECK(back->size() == 4096); if (back->size() == 4096) { CHECK((*back)[0] == 'F'); CHECK((*back)[4095] == 'G'); }
    {   // read-only: mapped, not resizable; moved-from objects are empty
        auto ro = os::fs::mapped_file::open(path, os::rd); CHECK(ro); if (!ro) return;
        CHECK(ro->size() == 4096); CHECK(ro->data()[0] == 'F');
        auto st = ro->resize(8192); CHECK(!st); CHECK(ro->size() == 4096);
        os::fs::mapped_file moved = std::move(*ro); CHECK(moved.size() == 4096); CHECK(ro->size() == 0); CHECK(!ro->valid());
        moved.close(); CHECK(!moved.resize(1));
    }
    CHECK(!os::fs::mapped_file::open(os::join(*d, "missing.bin"), os::rd));
    os::remove_all(*d);
}

TEST(fifo_named_pipe) {
    const std::string name = "forge-fifo-" + std::to_string(os::pid());
    CHECK(!os::fifo::create(""));
    auto st = os::fifo::create(name); CHECK_MSG(bool(st), "%s", st.err.what.c_str()); if (!st) return;
    CHECK(os::fifo::create(name));   // already there: fine
#if defined(FORGE_OS_POSIX)
    CHECK(os::fifo::path(name) == os::join(os::temp_dir(), name + ".fifo")); CHECK(os::fifo::path("/x/y") == "/x/y");
    {   // something else of that name is refused
        auto plain = os::join(os::temp_dir(), name + "-plain.fifo"); CHECK(os::write_file(plain, "x"));
        CHECK(!os::fifo::create(name + "-plain")); os::remove(plain);
    }
#else
    CHECK(os::fifo::path(name) == "\\\\.\\pipe\\" + name);
#endif
    // nobody on the other side: both opens time out instead of hanging
    auto t0 = os::monotonic_ns();
    auto lonely = os::fifo::open_write(name, 100); CHECK(!lonely); CHECK(lonely.err.code == ETIMEDOUT);
    auto lonely_r = os::fifo::open_read(name, 100); CHECK(!lonely_r); CHECK(lonely_r.err.code == ETIMEDOUT);
    CHECK(os::monotonic_ns() - t0 < 10'000'000'000ull);
    // reader first (the order Windows needs), writer from another thread
    std::string msg(100000, '\0'); for (std::size_t i = 0; i < msg.size(); ++i) msg[i] = char('A' + i % 23);
    std::string got; bool wrote = false;
    std::thread w([&] { auto e = os::fifo::open_write(name, 10000); if (e) { wrote = bool(e->write_all(msg)); } });
    auto r = os::fifo::open_read(name, 10000); CHECK_MSG(bool(r), "%s", r.err.what.c_str());
    if (r) got = r->read_all();
    w.join();
    CHECK(wrote); CHECK(got == msg);
    CHECK(os::fifo::remove(name)); CHECK(os::fifo::remove(name));   // gone twice is fine
#if defined(FORGE_OS_POSIX)
    CHECK(!os::exists(os::fifo::path(name)));
#endif
}

#if defined(_WIN32)
TEST(clone_file_windows) {
    auto d = os::fs::make_temp_dir("forge-clone-"); CHECK(d); if (!d) return;
    auto a = os::join(*d, "a.bin"), b = os::join(*d, "b.bin");
    std::string data(300000, '\0'); for (std::size_t i = 0; i < data.size(); ++i) data[i] = char(i * 11);
    CHECK(os::write_file(a, data));
    auto st = os::fs::clone_file(a, b);
    if (!st) {   // NTFS, FAT, wine: no block sharing; nothing may be left behind
        CHECK_MSG(os::is_unsupported(st.err), "%s", st.err.what.c_str()); CHECK(!os::exists(b));
        std::printf("  skip: %s\n", st.err.what.c_str());
    } else {     // ReFS / Dev Drive
        CHECK(os::read_file(b).value == data); CHECK(!os::fs::clone_file(a, b));   // the target must not exist
    }
    CHECK(!os::fs::clone_file(os::join(*d, "missing"), b));
    os::remove_all(*d);
}
#endif
