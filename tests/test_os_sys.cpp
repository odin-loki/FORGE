// forge::os phase 2: fs, users, time, sysinfo, console.
#include <forge/os/fs.hpp>
#include <forge/os/users.hpp>
#include <forge/os/time.hpp>
#include <forge/os/sysinfo.hpp>
#include <forge/os/console.hpp>
#include <forge/os/oserror.hpp>
#include <forge/os/pipe.hpp>
#include <atomic>
#include <thread>
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
// ---------------------------------------------------------------- audit W1-B regressions
TEST(fs_non_ascii_paths) {
    auto d = fs::make_temp_dir("forge-\xC3\xA9-\xE6\x97\xA5-"); CHECK(d); if (!d) return;   // "é", "日"
    CHECK(os::stat(*d).is_dir);
    auto p = *d + "/\xE6\x97\xA5\xE6\x9C\xAC-caf\xC3\xA9.txt";
    CHECK(fs::write_file_atomic(p, "one")); CHECK(fs::write_file_atomic(p, "two")); CHECK(os::read_file(p).value == "two");
    auto l = os::list_dir(*d); CHECK(l && l->size() == 1 && (*l)[0].name == "\xE6\x97\xA5\xE6\x9C\xAC-caf\xC3\xA9.txt");   // no stray temp file
    { auto k = fs::file_lock::acquire(*d + "/l\xC3\xA9.lock"); CHECK(k); }
    CHECK(fs::exists(*d + "/l\xC3\xA9.lock"));   // created under its real (UTF-8) name, not a byte-widened one
    auto t = fs::make_temp_file("t\xC3\xA9-", ".tmp"); CHECK(t); if (t) { CHECK(os::stat(*t).is_file); CHECK(t->find("t\xC3\xA9-") != std::string::npos); fs::remove_all(*t); }
#if defined(FORGE_OS_WINDOWS)
    CHECK(fs::write_file_atomic(*d + "/ro.txt", "r", 0444)); CHECK(!os::file::open(*d + "/ro.txt", os::wr));
    SetFileAttributesW(os::widen(*d + "/ro.txt").c_str(), FILE_ATTRIBUTE_NORMAL);
#else
    CHECK(fs::write_file_atomic(*d + "/ro.txt", "r", 0600)); CHECK((*fs::mode(*d + "/ro.txt") & 0077) == 0);
#endif
    CHECK(fs::remove_all(*d)); CHECK(!fs::exists(*d));
}
TEST(mount_unescape_is_safe) {
    CHECK(fs::mount_unescape("/mnt/with\\040space") == "/mnt/with space");
    CHECK(fs::mount_unescape("a\\134b") == "a\\b");
    CHECK(fs::mount_unescape("bad\\9zz") == "bad\\9zz"); CHECK(fs::mount_unescape("end\\04") == "end\\04"); CHECK(fs::mount_unescape("\\") == "\\");
}
TEST(classify_by_domain) {
    CHECK(os::classify(os::errno_error(ENOENT, "x")) == os::os_errc::not_found);
    CHECK(os::classify(os::errno_error(EEXIST, "x")) == os::os_errc::already_exists);   // 17 = ERROR_NOT_SAME_DEVICE as Win32
    CHECK(os::classify(os::errno_error(EINVAL, "x")) == os::os_errc::invalid_argument);
#if defined(EAI_NONAME)
    CHECK(os::classify(os::error{EAI_NONAME, "x", os::err_domain::gai}) == os::os_errc::not_found);
    CHECK(os::classify_gai(EAI_AGAIN) == os::os_errc::would_block);
#endif
    auto missing = fs::file_size("definitely/not/here-forge"); CHECK(!missing); CHECK(os::classify(missing.err) == os::os_errc::not_found);
    auto parse = os::parse_iso8601("nope"); CHECK(os::classify(parse.err) == os::os_errc::invalid_argument);
#if defined(FORGE_OS_WINDOWS)
    CHECK(os::classify(os::error{int(ERROR_FILE_NOT_FOUND), "x"}) == os::os_errc::not_found);
    CHECK(os::classify(os::error{int(ERROR_FILE_NOT_FOUND), "x", os::err_domain::win32}) == os::os_errc::not_found);
    struct { unsigned long code; os::os_errc k; } t[] = {
        {ERROR_HANDLE_EOF, os::os_errc::io}, {11001, os::os_errc::not_found}, {11002, os::os_errc::would_block}, {ERROR_SEM_TIMEOUT, os::os_errc::timed_out},
        {ERROR_PIPE_BUSY, os::os_errc::busy}, {ERROR_NOT_FOUND, os::os_errc::not_found}, {10052, os::os_errc::connection_reset},
        {ERROR_NETWORK_UNREACHABLE, os::os_errc::network_unreachable}, {ERROR_CANT_RESOLVE_FILENAME, os::os_errc::too_many_links},
        {10047, os::os_errc::not_supported}, {ERROR_COMMITMENT_LIMIT, os::os_errc::out_of_memory}, {ERROR_ALREADY_EXISTS, os::os_errc::already_exists}};
    for (auto& e : t) CHECK_MSG(os::classify_win32(e.code) == e.k, "%lu -> %s", e.code, os::name(os::classify_win32(e.code)));
#else
    CHECK(os::classify(os::error{ENOENT, "x"}) == os::os_errc::not_found);
    CHECK(os::classify(os::error{2, "x", os::err_domain::win32}) == os::os_errc::other);
#endif
}
TEST(timezone_and_offsets) {
    auto tz = os::timezone_name(); std::printf("  timezone_name: '%s'\n", tz.c_str());
#if defined(FORGE_OS_WINDOWS) || defined(__APPLE__) || defined(FORGE_OS_ANDROID) || defined(FORGE_OS_FREEBSD)
    CHECK(!tz.empty());
#endif
#if defined(FORGE_OS_WINDOWS)
    CHECK(std::string(os::windows_zone_to_iana("W. Europe Standard Time")) == "Europe/Berlin"); CHECK(!os::windows_zone_to_iana("No Such Zone"));
#endif
    int o = os::utc_offset_seconds(-86400LL * 365 * 300); CHECK(o >= -16 * 3600 && o <= 16 * 3600);   // year 1670: 0 or a sane offset, no garbage
}
TEST(pipe_read_timeout) {
    auto p = os::make_pipe(); CHECK(p); if (!p) return;
    auto t0 = os::monotonic_ns(); CHECK(p->read.read_some(100, 50).empty()); CHECK(os::monotonic_ns() - t0 >= 40'000'000ull);
    CHECK(p->write.write_all("hello")); CHECK(p->read.read_some(100, 2000) == "hello");
    std::thread w([&] { os::sleep_ns(30'000'000); p->write.write_all(" world"); p->write.close(); });
    std::string all; for (std::string c; !(c = p->read.read_some(100, 5000)).empty();) all += c;
    w.join(); CHECK(all == " world");
}
TEST(console_bits) {
    CHECK(os::console::paint("x", os::console::color::red, false) == "x");
    CHECK(os::console::paint("x", os::console::color::red, true) == "\x1b[31mx\x1b[0m");
    auto s = os::console::size(); CHECK(s.value.cols > 0);
    (void)os::console::is_tty(os::console::stream::out);
}
