// forge::os phase 2: fs, users, time, sysinfo, console.
#include <forge/os/fs.hpp>
#include <forge/os/users.hpp>
#include <forge/os/time.hpp>
#include <forge/os/sysinfo.hpp>
#include <forge/os/console.hpp>
#include <atomic>
#include <vector>
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
#if !defined(FORGE_OS_ANDROID) && !defined(FORGE_OS_IOS)
TEST(mounts_list) {
    auto m = fs::mounts(); CHECK(m); CHECK(!m->empty());
    bool root = false; for (auto& x : *m) { CHECK(!x.path.empty()); root |= x.path == "/" || (x.path.size() == 3 && x.path[1] == ':'); }
    CHECK(root);
}
#endif
TEST(users_dirs) {
    auto u = os::current_user(); CHECK(!u.name.empty() || !u.home.empty());
    CHECK(u.uid >= 0); auto r = os::lookup_uid(u.uid); CHECK(r); CHECK(r->name == u.name); CHECK(r->uid == u.uid);
    auto n = os::lookup_user(u.name); CHECK(n); CHECK(n->uid == u.uid); CHECK(!n->home.empty());
    CHECK(!os::lookup_user("no-such-user-forge-xyz"));
#if defined(FORGE_OS_WINDOWS)
    CHECK(u.sid.rfind("S-1-", 0) == 0); CHECK(n->sid == u.sid); CHECK(u.gid >= 0);
    CHECK(!os::lookup_uid(0x7FFFFFF0));
#else
    CHECK(u.sid.empty());
#endif
    CHECK(!os::config_dir().empty()); CHECK(!os::cache_dir().empty());
#if defined(FORGE_OS_WINDOWS)
    std::printf("  full name '%s', %zu groups\n", u.full_name.c_str(), u.groups.size()); CHECK(!u.groups.empty());
    {   // Known Folders, as the shell resolves them (OneDrive / GPO redirection included)
        auto r = os::run({"powershell", "-NoProfile", "-Command", "[Console]::OutputEncoding=[Text.Encoding]::UTF8; foreach ($f in 'Desktop','MyDocuments','ApplicationData','LocalApplicationData') { [Environment]::GetFolderPath($f) }"});
        CHECK(r); std::vector<std::string> want; std::string cur;
        if (r) for (char c : r->out) { if (c == '\n') { if (!cur.empty()) want.push_back(cur); cur.clear(); } else if (c != '\r') cur += c; }
        CHECK(want.size() == 4);
        if (want.size() == 4) {
            CHECK_MSG(os::user_directory(os::user_dir::desktop) == want[0], "%s vs %s", os::user_directory(os::user_dir::desktop).c_str(), want[0].c_str());
            CHECK(os::user_directory(os::user_dir::documents) == want[1]); CHECK(os::config_dir() == want[2]); CHECK(os::user_directory(os::user_dir::state) == want[3]);
        }
    }
#elif !defined(__APPLE__)
    {   // user-dirs.dirs wins over the environment and the defaults
        auto cfg = os::join(os::temp_dir(), "forge_userdirs_" + std::to_string(os::pid())); os::mkdirs(cfg);
        CHECK(os::write_file(os::join(cfg, "user-dirs.dirs"), "# written by xdg-user-dirs-update\nXDG_DESKTOP_DIR=\"$HOME/Schreibtisch\"\n  XDG_DOWNLOAD_DIR=\"/srv/dl\"\nXDG_DOCUMENTS_DIR=\"$HOME/My \\\"Docs\\\"\"\nXDG_MUSIC_DIR=relative\n"));
        auto old = os::getenv("XDG_CONFIG_HOME"); bool had = os::has_env("XDG_CONFIG_HOME"); auto oldd = os::getenv("XDG_DESKTOP_DIR"); bool hadd = os::has_env("XDG_DESKTOP_DIR");
        os::setenv("XDG_CONFIG_HOME", cfg); os::setenv("XDG_DESKTOP_DIR", "/env/desktop");
        auto h = os::home_dir();
        CHECK(os::user_directory(os::user_dir::desktop) == h + "/Schreibtisch"); CHECK(os::user_directory(os::user_dir::downloads) == "/srv/dl");
        CHECK(os::user_directory(os::user_dir::documents) == h + "/My \"Docs\"");
        CHECK(os::write_file(os::join(cfg, "user-dirs.dirs"), "XDG_DOWNLOAD_DIR=\"/srv/dl\"\n")); CHECK(os::user_directory(os::user_dir::desktop) == "/env/desktop");
        if (had) os::setenv("XDG_CONFIG_HOME", old); else os::unsetenv("XDG_CONFIG_HOME");
        if (hadd) os::setenv("XDG_DESKTOP_DIR", oldd); else os::unsetenv("XDG_DESKTOP_DIR");
        os::remove_all(cfg);
    }
#endif
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
    std::printf("  %s (%s %s) mem total %llu MiB avail %llu free %llu swap %llu/%llu cgroup %llu/%llu\n", r.pretty.c_str(), r.kernel.c_str(), r.kernel_version.c_str(),
                (unsigned long long)(m.total >> 20), (unsigned long long)(m.available >> 20), (unsigned long long)(m.free >> 20), (unsigned long long)(m.swap_free >> 20),
                (unsigned long long)(m.swap_total >> 20), (unsigned long long)(m.cgroup_usage >> 20), (unsigned long long)(m.cgroup_limit >> 20));
    CHECK(m.free > 0); CHECK(m.free <= m.total); CHECK(m.swap_free <= m.swap_total);
#if defined(FORGE_OS_WINDOWS)
    CHECK(r.version == "10" || r.version == "11"); CHECK(m.free <= m.available);
    CHECK(m.swap_total > 0);   // the page file
#elif defined(__linux__)
    if (os::stat("/sys/fs/cgroup/cgroup.controllers").exists) { auto d = os::detail::cgroup2_dir(); CHECK(!d.empty()); CHECK(os::stat(d).is_dir); }
#endif
    auto u = os::self_usage(); CHECK(u);
    CHECK(u->max_rss > 0); CHECK(u->minor_faults + u->major_faults > 0);
    {   // current_rss grows with touched memory, on every OS
        std::uint64_t r0 = os::current_rss(); CHECK(r0 > 0);
        std::vector<char> big(64u << 20); volatile char* vp = big.data(); for (std::size_t i = 0; i < big.size(); i += 4096) vp[i] = char(i | 1);
        std::uint64_t r1 = os::current_rss(); CHECK(r1 >= r0 + (32ull << 20)); CHECK(r1 < m.total); CHECK(vp[4096] != 0);
    }
    CHECK(*os::uptime_seconds() > 0); CHECK(m.available > 0);
#if defined(FORGE_OS_WINDOWS)
    CHECK(os::is_unsupported(os::load_average().err));
#else
    CHECK(os::load_average());
#endif
    auto a = os::per_cpu_times(); CHECK(a); CHECK(!a->empty()); CHECK(a->size() >= 1u);
    double x = os::cpu_usage((*a)[0], (*a)[0]); CHECK(x == 0);
    {   // the counters advance, and a busy loop shows up as busy time somewhere
        double busy0 = 0, all0 = 0; for (auto& t : *a) { CHECK(t.idle >= 0 && t.user >= 0 && t.system >= 0); busy0 += t.total() - t.idle; all0 += t.total(); }
        CHECK(all0 > 0);
        volatile std::uint64_t spin = 0; auto t0 = os::unix_time_ns(); while (os::unix_time_ns() - t0 < 300'000'000) spin = spin + 1;
        auto b = os::per_cpu_times(); CHECK(b); CHECK(b->size() == a->size());
        double busy1 = 0, all1 = 0; for (auto& t : *b) { busy1 += t.total() - t.idle; all1 += t.total(); }
        CHECK(all1 > all0); CHECK(busy1 > busy0);
    }
}
TEST(console_bits) {
    CHECK(os::console::paint("x", os::console::color::red, false) == "x");
    CHECK(os::console::paint("x", os::console::color::red, true) == "\x1b[31mx\x1b[0m");
    auto s = os::console::size(); CHECK(s.value.cols > 0);
    (void)os::console::is_tty(os::console::stream::out);
}
