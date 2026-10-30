// forge::os phase 2: fs, users, time, sysinfo, console.
#include <forge/os/fs.hpp>
#include <forge/os/users.hpp>
#include <forge/os/time.hpp>
#include <forge/os/sysinfo.hpp>
#include <forge/os/console.hpp>
#include <atomic>
#include "check.hpp"
using namespace forge;
namespace fs = forge::os::fs;

TEST(glob_match_rules) {
    CHECK(fs::glob_match("*.cpp", "a.cpp")); CHECK(!fs::glob_match("*.cpp", "d/a.cpp"));
    CHECK(fs::glob_match("**/*.cpp", "a.cpp")); CHECK(fs::glob_match("**/*.cpp", "x/y/a.cpp"));
    CHECK(fs::glob_match("src/**", "src/a/b")); CHECK(fs::glob_match("a?c", "abc")); CHECK(!fs::glob_match("a?c", "a/c"));
    CHECK(fs::glob_match("[a-c]x", "bx")); CHECK(!fs::glob_match("[!a-c]x", "bx")); CHECK(fs::glob_match("\\*", "*"));
    CHECK(fs::glob_match("a*b*c", "aXXbYYc")); CHECK(!fs::glob_match("a*b*c", "aXXbYY"));
}
TEST(fs_tree_ops) {
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d;
    CHECK(fs::make_dirs(r + "/src/sub"));
    CHECK(fs::write_file_atomic(r + "/src/a.txt", "alpha")); CHECK(fs::write_file_atomic(r + "/src/sub/b.cpp", "beta"));
    CHECK(fs::write_file_atomic(r + "/src/sub/b.cpp", "beta2"));
    CHECK(os::read_file(r + "/src/sub/b.cpp").value == "beta2");
    CHECK(fs::copy_tree(r + "/src", r + "/dst")); CHECK(os::read_file(r + "/dst/sub/b.cpp").value == "beta2");
    auto g = fs::glob("**/*.cpp", r); CHECK(g); CHECK(g->size() == 2);
    auto g2 = fs::glob("src/*.txt", r); CHECK(g2); CHECK(g2->size() == 1 && (*g2)[0] == "src/a.txt");
    auto w = fs::walk(r + "/src"); CHECK(w); CHECK(w->size() == 3);
    { fs::walk_opts o; o.include_dirs = false; auto w2 = fs::walk(r, o); CHECK(w2 && w2->size() == 4); }
    CHECK(fs::move(r + "/dst", r + "/moved")); CHECK(!fs::exists(r + "/dst")); CHECK(fs::is_file(r + "/moved/a.txt"));
    CHECK(*fs::file_size(r + "/src/a.txt") == 5);
    std::int64_t t = 1700000000123000000LL; CHECK(fs::set_mtime_ns(r + "/src/a.txt", t));
    auto m = fs::mtime_ns(r + "/src/a.txt"); CHECK(m); CHECK(*m / 1000000 == t / 1000000);
#if defined(FORGE_OS_POSIX)
    CHECK(fs::chmod(r + "/src/a.txt", 0600)); CHECK(*fs::mode(r + "/src/a.txt") == 0600);
    CHECK(fs::symlink(r + "/src/a.txt", r + "/ln")); CHECK(fs::is_symlink(r + "/ln")); CHECK(fs::read_link(r + "/ln").value == r + "/src/a.txt");
#endif
    auto du = fs::disk_usage(r); CHECK(du); CHECK(du->total > 0);
    auto n = fs::remove_all(r); CHECK(n); CHECK(*n >= 8); CHECK(!fs::exists(r));
}
TEST(file_lock_exclusive) {
    auto p = fs::make_temp_file(); CHECK(p);
    auto a = fs::file_lock::acquire(*p); CHECK(a); CHECK(a->held());
#if defined(FORGE_OS_POSIX)
    auto b = fs::file_lock::acquire(*p, true, false); CHECK(!b);   // flock is per open file description
#endif
    a->unlock(); auto c = fs::file_lock::acquire(*p, true, false); CHECK(c);
    c->unlock(); fs::remove_all(*p);
}
#if defined(__linux__)
TEST(mounts_list) { auto m = fs::mounts(); CHECK(m); CHECK(!m->empty()); }
#endif
TEST(users_dirs) {
    auto u = os::current_user(); CHECK(!u.name.empty() || !u.home.empty());
#if defined(FORGE_OS_POSIX)
    CHECK(u.uid >= 0); auto r = os::lookup_uid(u.uid); CHECK(r); CHECK(r->name == u.name);
    CHECK(!os::lookup_user("no-such-user-forge-xyz"));
#endif
    CHECK(!os::config_dir().empty()); CHECK(!os::cache_dir().empty());
}
TEST(time_civil_and_iso) {
    CHECK(os::format_iso8601(0) == "1970-01-01T00:00:00.000Z");
    CHECK(os::format_iso8601(951782400LL * 1000000000LL, 0, 0) == "2000-02-29T00:00:00Z");
    CHECK(os::format_iso8601(1793318400123000000LL, 11 * 3600) == "2026-10-30T11:00:00.123+11:00");
    auto p = os::parse_iso8601("2026-10-30T11:00:00.123+11:00"); CHECK(p); CHECK(*p == 1793318400123000000LL);
    CHECK(*os::parse_iso8601("1969-12-31T23:59:59Z") == -1000000000LL);
    CHECK(!os::parse_iso8601("2026-13-01")); CHECK(!os::parse_iso8601("nope"));
    auto c = os::civil_from_unix_ns(0); CHECK(c.weekday == 4 && c.year == 1970);
    auto now = os::unix_time_ns(); CHECK(*os::parse_iso8601(os::format_iso8601(now, 0, 9)) == now);
    CHECK(*os::parse_iso8601(os::format_local_iso8601(now, 9)) == now);
}
TEST(timer_queue_runs) {
    std::atomic<int> once{0}, periodic{0};
    {
        os::timer_queue q;
        q.after(10, [&] { ++once; });
        auto id = q.every(5, [&] { ++periodic; });
        auto cancelled = q.after(10000, [&] { once += 100; });
        CHECK(q.cancel(cancelled));
        os::sleep_ns(120'000'000);
        q.cancel(id);
    }
    CHECK(once == 1); CHECK(periodic >= 3);
}
TEST(sysinfo_ext) {
    auto r = os::release_info(); CHECK(!r.name.empty());
    auto m = os::memory(); CHECK(m.total > 0);
    auto u = os::self_usage(); CHECK(u);
#if defined(__linux__)
    CHECK(*os::uptime_seconds() > 0); CHECK(os::load_average());
    auto a = os::per_cpu_times(); CHECK(a); CHECK(!a->empty()); CHECK(m.available > 0);
    double x = os::cpu_usage((*a)[0], (*a)[0]); CHECK(x == 0);
#endif
}
TEST(console_bits) {
    CHECK(os::console::paint("x", os::console::color::red, false) == "x");
    CHECK(os::console::paint("x", os::console::color::red, true) == "\x1b[31mx\x1b[0m");
    auto s = os::console::size(); CHECK(s.value.cols > 0);
    (void)os::console::is_tty(os::console::stream::out);
}
