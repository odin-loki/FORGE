// forge::os phase 2: fs, users, time, sysinfo, console.
#include <forge/os/fs.hpp>
#include <forge/os/users.hpp>
#include <forge/os/time.hpp>
#include <forge/os/sysinfo.hpp>
#include <forge/os/console.hpp>
#include <forge/os/oserror.hpp>
#include <forge/os/pipe.hpp>
#include <forge/os/service_control.hpp>
#include <algorithm>
#include <atomic>
#include <random>
#include <thread>
#include <vector>
#include "check.hpp"
using namespace forge;
namespace fs = forge::os::fs;
static bool wine_host() {   // wine lacks some APIs real Windows always has
#if defined(_WIN32)
    return GetProcAddress(GetModuleHandleA("ntdll.dll"), "wine_get_version") != nullptr;
#else
    return false;
#endif
}
// ---- machine identity, budgets, virtualization, account database, sessions, scheduler, service logs (begin)
#include <forge/os/sysinfo.hpp>
#include <forge/os/virtualization.hpp>
#include <forge/os/users.hpp>
#include <forge/os/sessions.hpp>
#include <forge/os/scheduler.hpp>
TEST(sysinfo_identity_and_budgets) {
    auto id = os::machine_id();
    std::printf("  machine id: %s\n", id ? (id->substr(0, 6) + "...").c_str() : id.err.what.c_str());
    CHECK(id || os::is_unsupported(id.err)); if (id) { CHECK(id->size() >= 16); CHECK(*os::machine_id() == *id); }
    os::set_machine_id("test-machine-id-0001"); CHECK(*os::machine_id() == "test-machine-id-0001"); os::set_machine_id("");
    auto bt = os::boot_time_ns(); CHECK(bt); auto up = os::uptime_seconds();
    if (bt && up) {
        std::int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        CHECK(*bt > 0 && *bt < now); double d = double(now - *bt) / 1e9 - *up; CHECK_MSG(d > -120 && d < 120, "boot time vs uptime differ by %.1f s", d);
    }
    auto c = os::cpu_limits(); std::printf("  cpus: affinity %u quota %.2f usable %u (%s)\n", c.affinity, c.quota, c.usable, c.source.c_str());
    CHECK(c.affinity >= 1 && c.usable >= 1 && c.usable <= c.affinity); CHECK(os::usable_cpu_count() == c.usable);
    auto m = os::memory_limits(); std::printf("  memory: physical %llu limit %llu usable %llu (%s)\n", (unsigned long long)m.physical, (unsigned long long)m.limit, (unsigned long long)m.usable, m.source.c_str());
    CHECK(m.physical > 0 && m.usable > 0 && m.usable <= m.physical);
    CHECK(os::detail::parse_cpu_max("max 100000") == 0); CHECK(os::detail::parse_cpu_max("150000 100000") == 1.5); CHECK(os::detail::parse_cpu_max("-1 100000") == 0); CHECK(os::detail::parse_cpu_max("") == 0);
}
TEST(virtualization_detect) {
    auto v = os::virtualization();
    std::printf("  hypervisor '%s' container '%s' translator '%s' emulator %d\n", v.hypervisor.c_str(), v.container.c_str(), v.translator.c_str(), v.emulator);
    CHECK(v.virtualized() == !v.hypervisor.empty());
    namespace d = os::detail;
    CHECK(d::hypervisor_from_cpuid_vendor("KVMKVMKVM\0\0\0") == "kvm"); CHECK(d::hypervisor_from_cpuid_vendor("Microsoft Hv") == "hyperv");
    CHECK(d::hypervisor_from_cpuid_vendor("VMwareVMware") == "vmware"); CHECK(d::hypervisor_from_cpuid_vendor("GenuineIntel").empty());
    CHECK(d::hypervisor_from_dmi("QEMU", "Standard PC (Q35 + ICH9, 2009)") == "qemu"); CHECK(d::hypervisor_from_dmi("Microsoft Corporation", "Virtual Machine") == "hyperv");
    CHECK(d::hypervisor_from_dmi("Microsoft Corporation", "Surface Laptop 5").empty()); CHECK(d::hypervisor_from_dmi("Dell Inc.", "XPS 13").empty());
    CHECK(d::hypervisor_from_dmi("innotek GmbH", "VirtualBox") == "virtualbox"); CHECK(d::hypervisor_from_dmi("Amazon EC2", "t3.micro") == "kvm");
    CHECK(d::container_from_cgroup("0::/system.slice/docker-abc.scope\n") == "docker"); CHECK(d::container_from_cgroup("12:memory:/kubepods/burstable/pod1/abc\n") == "kubernetes");
    CHECK(d::container_from_cgroup("0::/user.slice/user-1000.slice/session-3.scope\n").empty()); CHECK(d::container_from_cgroup("0::/machine.slice/libpod-123.scope") == "podman");
    CHECK(d::wsl_kernel("5.15.153.1-microsoft-standard-WSL2")); CHECK(d::wsl_kernel("4.4.0-19041-Microsoft")); CHECK(!d::wsl_kernel("6.8.0-45-generic"));
}
TEST(users_database_and_groups) {
    auto me = os::current_user();
    auto us = os::list_users(); std::printf("  users: %s\n", us ? std::to_string(us->size()).c_str() : us.err.what.c_str());
    CHECK(us || os::is_unsupported(us.err));
    if (us) { for (auto& u : *us) CHECK(!u.name.empty()); }
#if defined(__linux__) && !defined(__ANDROID__)
    if (us) { bool root = false; for (auto& u : *us) root |= u.name == "root" && u.uid == 0; CHECK(root); }
#endif
    auto gs = os::list_groups(); std::printf("  groups: %s\n", gs ? std::to_string(gs->size()).c_str() : gs.err.what.c_str());
    CHECK(gs || os::is_unsupported(gs.err));
    if (gs && !gs->empty()) {
        for (auto& g : *gs) CHECK(!g.name.empty());
        auto g0 = os::lookup_group((*gs)[0].name); CHECK_MSG(bool(g0), "%s", (*gs)[0].name.c_str()); if (g0) CHECK(g0->name == (*gs)[0].name && g0->gid == (*gs)[0].gid);
    }
    CHECK(!os::lookup_group("forge-no-such-group-xyz"));
    if (!me.groups.empty()) { auto g = os::lookup_group(me.groups[0]); std::printf("  group %s: %s\n", me.groups[0].c_str(), g ? ("gid " + std::to_string(g->gid)).c_str() : g.err.what.c_str()); }
    std::printf("  is_admin %d, is_elevated %d\n", os::is_admin(), os::is_elevated()); if (os::is_elevated()) CHECK(os::is_admin());
    for (auto d : {os::user_dir::music, os::user_dir::pictures, os::user_dir::videos, os::user_dir::public_share, os::user_dir::templates, os::user_dir::fonts, os::user_dir::applications, os::user_dir::temp})
        CHECK(!os::user_directory(d).empty());
    CHECK(os::user_directory(os::user_dir::home) == os::home_dir()); CHECK(os::user_directory(os::user_dir::temp) == os::temp_dir());
}
TEST(sessions_and_elevation) {
    auto s = os::login_sessions(); std::printf("  sessions: %s\n", s ? std::to_string(s->size()).c_str() : s.err.what.c_str());
    CHECK(s || os::is_unsupported(s.err));
    for (auto& x : s.value) { std::printf("  session %ld user %s on '%s' from '%s'%s\n", x.id, x.user.c_str(), x.terminal.c_str(), x.host.c_str(), x.active ? "" : " (inactive)"); CHECK(!x.user.empty()); }
    CHECK(!os::run_elevated({})); CHECK(!os::run_elevated({"a", std::string("b\0c", 3)}));
    namespace de = os::detail_elevate;
    CHECK(de::sh_quote("it's") == "'it'\\''s'"); CHECK(de::win_quote("a b\\") == "\"a b\\\\\""); CHECK(de::win_quote("plain") == "plain");
    auto cmd = os::elevation_command({"/bin/echo", "a b"}, false); std::printf("  elevation command:"); for (auto& a : cmd) std::printf(" %s", a.c_str()); std::printf("\n");
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IPHONE)
    CHECK(os::is_unsupported(os::run_elevated({"true"}).err));
#else
#if defined(_WIN32)
    std::vector<std::string> argv = {"cmd.exe", "/c", "exit 7"};
#else
    std::vector<std::string> argv = {"/bin/sh", "-c", "exit 7"};
#endif
    auto r = os::run_elevated(argv, false, 20000);   // never prompts: runs (root, or passwordless sudo) or fails cleanly
    std::printf("  run_elevated without prompt: %s\n", r ? ("exit " + std::to_string(*r)).c_str() : r.err.what.c_str());
    if (r) CHECK(*r == 7 || !os::is_elevated()); else CHECK(os::needs_prompt(r.err) || os::is_unsupported(r.err) || r.err.code != 0);
    if (os::is_elevated()) CHECK(r && *r == 7);
#endif
}
TEST(schedule_render_and_validate) {
    namespace sc = os::schedule;
    sc::task t; t.name = "forge.nightly"; t.argv = {"/opt/forge/bin/forge", "sync", "--pct=100%", "it's"}; t.minute = 30; t.hour = 2; t.description = "Nightly <sync>";
    CHECK(sc::validate(t));
    { auto b = t; b.name = "a b"; CHECK(!sc::validate(b)); } { auto b = t; b.argv = {}; CHECK(!sc::validate(b)); } { auto b = t; b.interval_s = 600; CHECK(!sc::validate(b)); }
    { auto b = t; b.minute = 60; CHECK(!sc::validate(b)); } { auto b = t; b.minute = -1; CHECK(!sc::validate(b)); } { auto b = t; b.day = 3; b.weekday = 1; CHECK(!sc::validate(b)); }
    { auto b = t; b.month = 6; CHECK(!sc::validate(b)); } { auto b = t; b.argv[1] = "a\nb"; CHECK(!sc::validate(b)); } { sc::task i; i.name = "x"; i.argv = {"/bin/true"}; i.interval_s = 30; CHECK(!sc::validate(i)); }
    CHECK(sc::render_cron_line(t) == "30 2 * * * '/opt/forge/bin/forge' 'sync' '--pct=100\\%' 'it'\\''s' # forge-schedule:forge.nightly");
    auto timer = sc::render_systemd_timer(t); CHECK(timer.find("OnCalendar=*-*-* 02:30:00\nPersistent=true\nUnit=forge.nightly.service\n") != std::string::npos);
    auto svc = sc::render_systemd_service(t); CHECK(svc.find("Type=oneshot\nExecStart=") != std::string::npos); CHECK(svc.find("100%%") != std::string::npos);
    auto pl = sc::render_launchd_plist(t); CHECK(pl.find("<key>StartCalendarInterval</key>\n  <dict>\n    <key>Minute</key><integer>30</integer>\n    <key>Hour</key><integer>2</integer>\n  </dict>") != std::string::npos);
    auto xml = sc::render_task_xml(t);
    CHECK(xml.find("<StartBoundary>2020-01-01T02:30:00</StartBoundary>") != std::string::npos); CHECK(xml.find("<ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>") != std::string::npos);
    CHECK(xml.find("<Description>Nightly &lt;sync&gt;</Description>") != std::string::npos); CHECK(xml.find("<Command>/opt/forge/bin/forge</Command>") != std::string::npos);
    CHECK(xml.find("<Arguments>sync --pct=100% it&apos;s</Arguments>") != std::string::npos);
    sc::task w = t; w.weekday = 1; w.working_dir = "/var/lib/forge data";
    CHECK(sc::render_cron_line(w).rfind("30 2 * * 1 cd '/var/lib/forge data' && ", 0) == 0); CHECK(sc::render_systemd_timer(w).find("OnCalendar=Mon *-*-* 02:30:00") != std::string::npos);
    CHECK(sc::render_task_xml(w).find("<ScheduleByWeek><DaysOfWeek><Monday /></DaysOfWeek>") != std::string::npos); CHECK(sc::render_launchd_plist(w).find("<key>Weekday</key><integer>1</integer>") != std::string::npos);
    sc::task m = t; m.day = 15; m.month = 6; CHECK(sc::validate(m)); CHECK(sc::render_systemd_timer(m).find("OnCalendar=*-06-15 02:30:00") != std::string::npos);
    CHECK(sc::render_task_xml(m).find("<DaysOfMonth><Day>15</Day></DaysOfMonth><Months><June /></Months>") != std::string::npos);
    sc::task h; h.name = "hourly"; h.argv = {"/bin/true"}; h.minute = 5; CHECK(sc::validate(h)); CHECK(sc::render_cron_line(h).rfind("5 * * * * ", 0) == 0);
    CHECK(sc::render_task_xml(h).find("<Interval>PT1H</Interval>") != std::string::npos); CHECK(sc::render_systemd_timer(h).find("OnCalendar=*-*-* *:05:00") != std::string::npos);
    sc::task i; i.name = "every"; i.argv = {"/bin/true"}; i.interval_s = 900; CHECK(sc::validate(i));
    CHECK(sc::render_cron_line(i).rfind("*/15 * * * * ", 0) == 0); CHECK(sc::render_systemd_timer(i).find("OnUnitActiveSec=900s") != std::string::npos);
    CHECK(sc::render_launchd_plist(i).find("<key>StartInterval</key><integer>900</integer>") != std::string::npos); CHECK(sc::render_task_xml(i).find("<Interval>PT15M</Interval>") != std::string::npos);
    i.interval_s = 7200; CHECK(sc::render_cron_line(i).rfind("0 */2 * * * ", 0) == 0); i.interval_s = 86400; CHECK(sc::render_cron_line(i).rfind("0 0 * * * ", 0) == 0);
    i.interval_s = 420; CHECK(sc::render_cron_line(i).empty()); i.interval_s = 90; CHECK(sc::render_cron_line(i).empty());   // 7 min / 90 s: not expressible
    namespace ds = sc::detail_schedule;
    bool found = false; std::string tab = "MAILTO=x\n0 1 * * * /a # forge-schedule:other\n30 2 * * * /b # forge-schedule:forge.nightly\n";
    CHECK(ds::cron_without(tab, "forge.nightly", &found) == "MAILTO=x\n0 1 * * * /a # forge-schedule:other\n"); CHECK(found);
    CHECK(ds::cron_without(tab, "forge", &found) == tab); CHECK(!found);   // a name that is a prefix of another does not match
    CHECK(ds::utf16_bom("aé𝄞") == std::string("\xFF\xFE" "a\0" "\xE9\0" "\x34\xD8\x1E\xDD", 10));
}
TEST(schedule_definition_files) {
    namespace sc = os::schedule;
    std::printf("  scheduler backend: %s\n", sc::backend().c_str());
    sc::task t; t.name = "forge-test-" + std::to_string(os::pid()); t.argv = {"/bin/true"}; t.interval_s = 3600;
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IPHONE)
    CHECK(os::is_unsupported(sc::install(t).err)); CHECK(os::is_unsupported(sc::uninstall(t).err)); CHECK(os::is_unsupported(sc::installed(t).err)); CHECK(sc::backend() == "none");
#else
    auto d = os::fs::make_temp_dir("fsched-"); CHECK(d); if (!d) return;
    t.dir = *d;   // definitions only: nothing is registered with the scheduler
    auto in0 = sc::installed(t); CHECK(in0 && !*in0);
    auto r = sc::install(t); CHECK_MSG(bool(r), "%s", r.err.what.c_str());
    auto files = sc::definition_files(t); CHECK(!files.empty());
    for (auto& f : files) { auto st = os::stat(f); CHECK_MSG(st.exists && st.size > 0, "%s", f.c_str()); }
    auto in1 = sc::installed(t); CHECK(in1 && *in1);
#if defined(_WIN32)
    auto x = os::read_file(files[0]); CHECK(x && x->size() > 4 && (unsigned char)(*x)[0] == 0xFF && (unsigned char)(*x)[1] == 0xFE);
#endif
    CHECK(sc::uninstall(t)); auto in2 = sc::installed(t); CHECK(in2 && !*in2); CHECK(sc::uninstall(t));
    os::fs::remove_all(*d);
    sc::task bad; bad.name = "../x"; bad.argv = {"/bin/true"}; bad.interval_s = 60; CHECK(!sc::install(bad)); CHECK(!sc::installed(bad)); CHECK(!sc::uninstall(bad));
#endif
}
TEST(service_logs_and_restart) {
    namespace svc = os::service;
    CHECK(svc::tail_lines_("a\nb\nc\n", 2) == "b\nc\n"); CHECK(svc::tail_lines_("a\nb\nc", 1) == "c"); CHECK(svc::tail_lines_("a\nb\n", 5) == "a\nb\n"); CHECK(svc::tail_lines_("", 3).empty());
    svc::service_spec s; s.name = "forge-test-nonexistent-" + std::to_string(os::pid()); s.argv = {"/bin/true"};
    auto cmd = svc::logs_command(s, 20); std::printf("  logs command:"); for (auto& a : cmd) std::printf(" %s", a.c_str()); std::printf("\n");
    { auto bad = s; bad.name = "a b"; CHECK(svc::logs_command(bad).empty()); CHECK(!svc::logs(bad)); }
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IPHONE)
    CHECK(cmd.empty()); CHECK(os::is_unsupported(svc::logs(s).err)); CHECK(os::is_unsupported(svc::restart(s).err));
#else
    CHECK(!cmd.empty());
#if !defined(__APPLE__)   // `log show` takes many seconds on a busy Mac: the command line is checked above, not run
    auto l = svc::logs(s, 20); std::printf("  logs: %s\n", l ? (std::to_string(l->size()) + " bytes").c_str() : l.err.what.c_str());
    CHECK(l || os::is_unsupported(l.err) || l.err.code != 0);
#endif
    auto r = svc::restart(s); CHECK(!r);   // not installed (or no service manager): an error, never a hang
#endif
}
// ---- machine identity, budgets, virtualization, account database, sessions, scheduler, service logs (end)

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
        std::vector<std::string> want; std::string cur;
        bool no_ps = wine_host() && (!r || r->exit_code != 0 || r->out.empty());   // wine ships no powershell
        if (no_ps) std::printf("  skip known-folder cross-check: no powershell under wine\n"); else CHECK(r);
        if (r && !no_ps) for (char c : r->out) { if (c == '\n') { if (!cur.empty()) want.push_back(cur); cur.clear(); } else if (c != '\r') cur += c; }
        if (!no_ps) CHECK(want.size() == 4);
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
    if (!u) return;
    CHECK(u->max_rss > 0);
    if (wine_host() && u->minor_faults + u->major_faults == 0) std::printf("  skip fault counters: wine reports none\n");
    else CHECK(u->minor_faults + u->major_faults > 0);
    {   // current_rss grows with touched memory, on every OS
        std::uint64_t r0 = os::current_rss(); CHECK(r0 > 0);
        std::vector<char> big(64u << 20); volatile char* vp = big.data(); for (std::size_t i = 0; i < big.size(); i += 4096) vp[i] = char(i | 1);
        std::uint64_t r1 = os::current_rss(); CHECK(r1 >= r0 + (32ull << 20)); CHECK(r1 < m.total); CHECK(vp[4096] != 0);
    }
    CHECK(*os::uptime_seconds() > 0); CHECK(m.available > 0);
    CHECK(os::load_average());
    auto a = os::per_cpu_times();
    if (!a && wine_host()) { std::printf("  skip per_cpu_times: wine: %s\n", a.err.what.c_str()); return; }
    CHECK(a); if (!a || a->empty()) { CHECK_MSG(false, "per_cpu_times returned no CPUs"); return; }
    double x = os::cpu_usage((*a)[0], (*a)[0]); CHECK(x == 0);
    {   // the counters advance, and a busy loop shows up as busy time somewhere
        double busy0 = 0, all0 = 0; for (auto& t : *a) { CHECK(t.idle >= 0 && t.user >= 0 && t.system >= 0); busy0 += t.total() - t.idle; all0 += t.total(); }
        CHECK(all0 > 0);
        volatile std::uint64_t spin = 0; auto t0 = os::unix_time_ns(); while (os::unix_time_ns() - t0 < 300'000'000) spin = spin + 1;
        auto b = os::per_cpu_times(); CHECK(b); if (!b) return; CHECK(b->size() == a->size());
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
    CHECK(fs::write_file_atomic(*d + "/ro.txt", "r", 0444)); CHECK(wine_host() || !os::file::open(*d + "/ro.txt", os::wr));   // wine as root opens read-only files
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

// ---- service control wave 2: SCM parameters, Task Scheduler XML, rc.d scripts (rendering everywhere) ----
namespace svc = forge::os::service;
static std::size_t count_of(const std::string& h, const std::string& n) { std::size_t c = 0; for (auto p = h.find(n); p != std::string::npos; p = h.find(n, p + 1)) ++c; return c; }
TEST(service_render_windows) {
    svc::service_spec s; s.name = "forge-test-svc"; s.description = "Forge test"; s.argv = {"C:\\Program Files\\Forge\\forge.exe", "a b", "q\"x", "tail\\"};
    s.working_dir = "C:\\srv"; s.env = {{"A", "1"}};
    auto c = svc::render_scm(s);
    CHECK(c.command_line == "\"C:\\Program Files\\Forge\\forge.exe\" \"a b\" \"q\\\"x\" tail\\");   // argv[0] quoted: no unquoted-path hijack
    CHECK(c.display_name == "Forge test" && c.description == "Forge test" && c.auto_start && c.restart);
    CHECK(c.environment.size() == 2 && c.environment[0] == "A=1" && c.environment[1] == "FORGE_SERVICE_DIR=C:\\srv");
    s.argv[0] = "C:\\Forge\\forge.exe"; s.argv[3] = "x y\\"; CHECK(svc::render_scm(s).command_line == "C:\\Forge\\forge.exe \"a b\" \"q\\\"x\" \"x y\\\\\"");
    // Task XML: markup in any field stays text
    auto t = s; t.env.clear(); t.description = "x</Description></RegistrationInfo><Actions><Exec><Command>evil.exe</Command></Exec></Actions>";
    t.argv = {"C:\\a&b\\p.exe", "<x>", "it's \"q\""}; t.working_dir = "C:\\w&<>";
    auto x = svc::render_task_xml(t, "S-1-5-21-1\"<evil/>");
    CHECK(count_of(x, "<Exec>") == 1); CHECK(count_of(x, "<Command>") == 1); CHECK(x.find("evil.exe</Command>") == std::string::npos);
    CHECK(x.find("<Description>x&lt;/Description&gt;&lt;/RegistrationInfo&gt;") != std::string::npos);
    CHECK(x.find("<Command>C:\\a&amp;b\\p.exe</Command>") != std::string::npos);
    CHECK(x.find("<Arguments>&lt;x&gt; &quot;it&apos;s \\&quot;q\\&quot;&quot;</Arguments>") != std::string::npos);
    CHECK(x.find("<WorkingDirectory>C:\\w&amp;&lt;&gt;</WorkingDirectory>") != std::string::npos);
    CHECK(x.find("<UserId>S-1-5-21-1&quot;&lt;evil/&gt;</UserId>") != std::string::npos); CHECK(count_of(x, "<LogonTrigger>") == 1);
    CHECK(x.find("<RestartOnFailure>") != std::string::npos);
    t.start_at_login = false; t.restart_on_failure = false; x = svc::render_task_xml(t, "S-1-5-21-1");
    CHECK(x.find("<LogonTrigger>") == std::string::npos); CHECK(x.find("<RestartOnFailure>") == std::string::npos);
    t.description = "a\nb"; CHECK(!svc::validate(t)); CHECK(svc::render_task_xml(t, "S").find("a\nb") == std::string::npos);
    // env through PowerShell: literals single-quoted with every quote variant doubled, nothing on a command line
    CHECK(svc::ps_quote_("a'b") == "'a''b'"); CHECK(svc::ps_quote_("$(evil) `n") == "'$(evil) `n'");
    CHECK(svc::ps_quote_("x\xE2\x80\x99y") == "'x\xE2\x80\x99\xE2\x80\x99y'"); CHECK(svc::ps_quote_("\xE2\x80\x98") == "'\xE2\x80\x98\xE2\x80\x98'");
    CHECK(svc::ps_encode_("A") == "QQA="); CHECK(svc::ps_encode_("ab") == "YQBiAA=="); CHECK(svc::ps_encode_("\xC3\xA9") == "6QA=");
    CHECK(svc::ps_encode_("\xF0\x9F\x98\x80") == "PdgA3g==");   // U+1F600 as a surrogate pair
    auto e = s; e.env = {{"K", "v'; Remove-Item C:\\ -Recurse; '"}};
    auto ps = svc::task_powershell_script_(e);
    CHECK(ps.find("SetEnvironmentVariable('K','v''; Remove-Item C:\\ -Recurse; ''')") != std::string::npos);
    CHECK(ps.find("$i.Arguments='\"a b\" \"q\\\"x\" \"x y\\\\\"'") != std::string::npos);
    auto xe = svc::render_task_xml(e, "S"); CHECK(xe.find("<Command>powershell.exe</Command>") != std::string::npos);
    CHECK(xe.find("-EncodedCommand " + svc::ps_encode_(ps) + "</Arguments>") != std::string::npos); CHECK(xe.find("Remove-Item") == std::string::npos);
}
TEST(service_render_rcd) {
    svc::service_spec s; s.name = "forge-test.svc"; s.description = "it's"; s.argv = {"/usr/local/bin/forge", "a b", "'; rm -rf /; '", "$(id)", "`id`", "${pidfile}"};
    s.working_dir = "/srv/my dir'"; s.env = {{"GOOD", "x'y $HOME"}, {"BAD-NAME", "1"}};
    CHECK(svc::rc_name("forge-test.svc") == "forge_test_svc"); CHECK(svc::rc_name("9x") == "svc_9x");
    auto r = svc::render_rcd_script(s);
    CHECK(r.rfind("#!/bin/sh\n", 0) == 0); CHECK(r.find("\nname=forge_test_svc\n") != std::string::npos); CHECK(r.find("\nrcvar=forge_test_svc_enable\n") != std::string::npos);
    CHECK(r.find("\ndesc='it'\\''s'\n") != std::string::npos); CHECK(r.find("BAD-NAME") == std::string::npos);
    CHECK(r.find("\n\tcd -- '/srv/my dir'\\''' || return 1\n") != std::string::npos); CHECK(r.find("\n\texport GOOD='x'\\''y $HOME'\n") != std::string::npos);
    CHECK(r.find("-r -R 5") != std::string::npos); auto nr = s; nr.restart_on_failure = false; CHECK(svc::render_rcd_script(nr).find("-r -R 5") == std::string::npos);
    CHECK(svc::render_systemd_unit(s).find("WantedBy=default.target") != std::string::npos);
    auto sys = s; sys.scope = svc::service_scope::system; CHECK(svc::render_systemd_unit(sys).find("WantedBy=multi-user.target") != std::string::npos);
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
    {   // what rc.subr does: the assignment, then eval of "$command $command_args"; and the precmd's exports
        auto ln = r.substr(r.find("\ncommand_args=") + 1); ln = ln.substr(0, ln.find('\n'));
        auto ex = r.substr(r.find("\texport GOOD=") + 1); ex = ex.substr(0, ex.find('\n'));
        os::spawn_spec sp; sp.argv = {"/bin/sh", "-c", "pidfile=/var/run/p.pid; " + ln + "; eval \"set -- $command_args\"; for a; do printf '%s\\n' \"$a\"; done; " + ex + "; printf '%s\\n' \"$GOOD\""};
        auto o = os::spawn_wait(sp); CHECK(o && o->exit_code == 0);
        if (o) CHECK_MSG(o->out == "-P\n/var/run/p.pid\n-r\n-R\n5\n-S\n-T\nforge_test_svc\n--\n/usr/local/bin/forge\na b\n'; rm -rf /; '\n$(id)\n`id`\n${pidfile}\nx'y $HOME\n", "%s", o->out.c_str());
    }
#endif
#if defined(FORGE_OS_FREEBSD)
    {   // rc.d written to a directory, not registered; names that are not shell identifiers refused
        auto d = fs::make_temp_dir(); CHECK(d); auto t = s; t.dir = *d;
        CHECK(!svc::install(t)); t.env.erase("BAD-NAME");
        CHECK(svc::install(t)); CHECK(fs::is_file(svc::unit_path(t))); CHECK(svc::query(t) == svc::service_state::stopped);
        CHECK(svc::uninstall(t)); CHECK(svc::query(t) == svc::service_state::not_installed); fs::remove_all(*d);
        t.dir.clear(); t.scope = svc::service_scope::user; CHECK(os::is_unsupported(svc::install(t).err));
    }
    // real rc.d service (root only, opt-in): daemon(8) running sleep, started, queried, stopped, removed
    if (::geteuid() == 0 && os::getenv("FORGE_TEST_RCD") == "1") {
        svc::service_spec d; d.name = "forge-test-rcd-" + std::to_string(os::pid()); d.argv = {"/bin/sleep", "300"}; d.scope = svc::service_scope::system;
        d.working_dir = "/tmp"; d.env = {{"FORGE_X", "a'b"}};
        struct cleanup { svc::service_spec& s; ~cleanup() { svc::uninstall(s); } } guard{d};
        CHECK(svc::install(d)); CHECK(svc::query(d) == svc::service_state::stopped);
        auto st = svc::start(d); CHECK_MSG(st, "%s", st.err.what.c_str());
        auto i = svc::query_info(d); CHECK(i.state == svc::service_state::running); CHECK(i.pid > 0);
        std::printf("  rc.d service %s running, daemon pid %lld\n", d.name.c_str(), static_cast<long long>(i.pid));
        CHECK(svc::stop(d)); CHECK(svc::query(d) == svc::service_state::stopped);
        CHECK(svc::uninstall(d)); CHECK(svc::query(d) == svc::service_state::not_installed); CHECK(!fs::exists(svc::unit_path(d)));
    }
#endif
}
#if defined(FORGE_OS_WINDOWS)
TEST(service_windows_scm_query) {   // read-only SCM paths (no admin needed); installing a system service is CI / elevated only
    svc::service_spec s; s.name = "forge-test-no-such-svc"; s.argv = {"C:\\x.exe"}; s.scope = svc::service_scope::system;
    CHECK(svc::query(s) == svc::service_state::not_installed); CHECK(svc::uninstall(s));
    s.name = "EventLog"; auto i = svc::query_info(s); CHECK(i.state == svc::service_state::running); CHECK(i.pid > 0);
    if (!os::is_elevated()) { s.name = "forge-test-denied"; auto r = svc::install(s); CHECK(!r); CHECK(r.err.code == ERROR_ACCESS_DENIED); }
    s.name = "forge-test-bat"; s.argv = {"C:\\x.bat", "a"}; CHECK(!svc::install(s));
    svc::service_spec t; t.name = "forge-test-dir"; t.argv = {"C:\\Windows\\System32\\cmd.exe", "/c", "exit 0"};
    auto d = fs::make_temp_dir(); CHECK(d); t.dir = *d;
    CHECK(svc::install(t)); auto txt = os::read_file(svc::unit_path(t)); CHECK(txt && txt->size() > 2 && (*txt)[0] == '\xFF' && (*txt)[1] == '\xFE');
    CHECK(svc::query(t) == svc::service_state::stopped); CHECK(svc::uninstall(t)); CHECK(svc::query(t) == svc::service_state::not_installed);
    fs::remove_all(*d);
}
TEST(service_windows_task) {   // a real per-user scheduled task: install, start, query, stop, env + exit code, remove
    if (wine_host()) { std::printf("  skip under wine: its Task Scheduler cannot run tasks (E_NOTIMPL)\n"); return; }
    std::mt19937_64 rng(std::random_device{}() ^ os::monotonic_ns());
    svc::service_spec s; s.name = "forge-test-" + std::to_string(rng() % 1000000000ull); s.description = "FORGE test task (safe to delete)";
    s.argv = {"PING.EXE", "-n", "30", "127.0.0.1"}; s.start_at_login = false; s.working_dir = os::getenv("SystemRoot", "C:\\Windows");
    struct cleanup { svc::service_spec& s; ~cleanup() { svc::uninstall(s); } } guard{s};
    auto wait_for = [&](auto pred) { for (int i = 0; i < 300; ++i) { auto q = svc::query_info(s); if (pred(q)) return q; os::sleep_ns(100'000'000); } return svc::query_info(s); };
    CHECK(svc::query(s) == svc::service_state::not_installed);
    auto in = svc::install(s); CHECK_MSG(in, "%s", in.err.what.c_str()); if (!in) return;
    CHECK(svc::query(s) == svc::service_state::stopped);
    auto st = svc::start(s); CHECK_MSG(st, "%s", st.err.what.c_str());
    auto q = wait_for([](const svc::service_info& i) { return i.state == svc::service_state::running; });
    CHECK(q.state == svc::service_state::running); CHECK(q.pid > 0);
    CHECK(svc::start(s));   // already running: IgnoreNew, not an error
    auto sp = svc::stop(s); CHECK_MSG(sp, "%s", sp.err.what.c_str());
    q = svc::query_info(s); CHECK_MSG(q.state == svc::service_state::stopped, "state %s result 0x%lx", svc::state_name(q.state), (unsigned long)q.exit_code);
    CHECK(q.pid == 0);
    // environment through the PowerShell wrapper (with quoting traps), exit code reported as failed
    s.env = {{"FORGE_T", "7"}, {"FORGE_Q", "it's $(x) \xE2\x80\x99q\" `n"}}; s.argv = {"cmd.exe", "/d", "/c", "exit %FORGE_T%"};
    in = svc::install(s); CHECK_MSG(in, "%s", in.err.what.c_str());
    CHECK(svc::start(s));
    os::sleep_ns(500'000'000);
    q = wait_for([](const svc::service_info& i) { return i.state != svc::service_state::running; });
    CHECK_MSG(q.state == svc::service_state::failed && q.exit_code == 7, "state %s result 0x%lx", svc::state_name(q.state), (unsigned long)q.exit_code);
    // stop ends the wrapper's child too (Task Scheduler alone would leave it running)
    s.argv = {"PING.EXE", "-n", "60", "127.0.0.1"}; CHECK(svc::install(s)); CHECK(svc::start(s));
    q = wait_for([](const svc::service_info& i) { return i.state == svc::service_state::running; });
    CHECK(q.state == svc::service_state::running);
    os::sleep_ns(1'500'000'000);   // let PowerShell start the child
    auto kids = q.pid ? svc::open_descendants_(DWORD(q.pid)) : std::vector<HANDLE>{}; CHECK(!kids.empty());
    CHECK(svc::stop(s));
    for (HANDLE h : kids) { CHECK(WaitForSingleObject(h, 5000) == WAIT_OBJECT_0); CloseHandle(h); }
    CHECK(svc::uninstall(s)); CHECK(svc::query(s) == svc::service_state::not_installed); CHECK(svc::uninstall(s));
}
#endif
TEST(load_average_emulation) {   // Windows: sampler started by the first call; elsewhere the kernel's average
    auto a = os::load_average(); CHECK(a); if (!a) return;
    CHECK(a->one >= 0 && a->five >= 0 && a->fifteen >= 0);
#if defined(FORGE_OS_WINDOWS)
    const unsigned n = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<bool> stop{false}; std::vector<std::thread> spin;
    for (unsigned i = 0; i < n; ++i) spin.emplace_back([&] { volatile std::uint64_t x = 0; while (!stop.load(std::memory_order_relaxed)) x = x + 1; });
    os::sleep_ns(11'000'000'000ull);   // two 5 s ticks
    auto b = os::load_average(); stop = true; for (auto& t : spin) t.join();
    CHECK(b); if (!b) return;
    CHECK(b->one >= 0 && b->five >= 0 && b->fifteen >= 0);
    if (wine_host() && b->one <= a->one + 0.05 * n) { std::printf("  skip under wine: no PDH queue length, busy time does not move the estimate\n"); return; }
    CHECK_MSG(b->one > a->one + 0.05 * n,"before %.2f after %.2f (n=%u)", a->one, b->one, n);
    CHECK(b->one - a->one > b->fifteen - a->fifteen);   // the 1-minute average reacts fastest
#endif
}

// ---------------------------------------------------------------- parity additions (every OS)
#include <forge/os/sync.hpp>
namespace {
// A step the file system (not the OS layer) may lack: a clean `unsupported` is a skip, anything else a failure.
bool ok_or_unsupported(const os::error& e, const char* what) {
    if (!e) return true;
    if (os::is_unsupported(e)) { std::printf("  skip %s: %s\n", what, e.what.c_str()); return false; }
    CHECK_MSG(false, "%s: %s", what, e.what.c_str()); return false;
}
bool on_wine() {   // wine lacks stream enumeration and reparse points; real Windows must pass
#if defined(_WIN32)
    return GetProcAddress(GetModuleHandleA("ntdll.dll"), "wine_get_version") != nullptr;
#else
    return false;
#endif
}
std::string pattern(std::size_t n, unsigned seed) { std::string s(n, '\0'); std::mt19937 g(seed); for (auto& c : s) c = char(g()); return s; }
}
TEST(fs_copy_file_and_clone) {
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d;
    const std::string data = pattern(3 * 1024 * 1024 + 123, 7);
    CHECK(os::write_file(r + "/src.bin", data));
    auto c = fs::copy_file(r + "/src.bin", r + "/dst.bin"); CHECK_MSG(c, "%s", c.err.what.c_str());
    if (c) std::printf("  copy_file: %s\n", c.value == fs::copy_method::clone ? "clone" : c.value == fs::copy_method::kernel ? "kernel" : "user");
    CHECK(os::read_file(r + "/dst.bin").value == data);
    auto again = fs::copy_file(r + "/src.bin", r + "/dst.bin"); CHECK(!again);          // exists, overwrite = false
    if (!again) CHECK(os::classify(again.err) == os::os_errc::already_exists);
    CHECK(os::read_file(r + "/dst.bin").value == data);
    CHECK(os::write_file(r + "/small.bin", "tiny"));
    auto over = fs::copy_file(r + "/small.bin", r + "/dst.bin", true); CHECK_MSG(over, "%s", over.err.what.c_str());
    CHECK(os::read_file(r + "/dst.bin").value == "tiny");
    CHECK(os::write_file(r + "/empty.bin", "")); CHECK(fs::copy_file(r + "/empty.bin", r + "/empty2.bin")); CHECK(*fs::file_size(r + "/empty2.bin") == 0);
    CHECK(!fs::copy_file(r + "/nope.bin", r + "/x.bin")); CHECK(!os::exists(r + "/x.bin"));
    auto cl = fs::clone_file(r + "/src.bin", r + "/clone.bin");
    if (ok_or_unsupported(cl.err, "clone_file")) {
        CHECK(os::read_file(r + "/clone.bin").value == data);
        CHECK(os::write_file(r + "/clone.bin", "changed")); CHECK(os::read_file(r + "/src.bin").value == data);   // copy-on-write
        CHECK(!fs::clone_file(r + "/src.bin", r + "/clone.bin"));                                                 // must not exist
    } else CHECK(!os::exists(r + "/clone.bin"));
#if defined(_WIN32) || defined(__FreeBSD__)
    CHECK(!cl && os::is_unsupported(cl.err));
#endif
    CHECK(fs::remove_all(r));
}
TEST(fs_preallocate_holes_and_ranges) {
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d;
    auto f = os::file::open(r + "/sparse.bin", os::rdwr | os::create); CHECK(f); if (!f) return;
    auto pre = fs::preallocate(*f, 1 << 20);
    if (ok_or_unsupported(pre.err, "preallocate")) { auto sz = f->size(); CHECK(sz && sz.value == (1u << 20)); }
    CHECK(fs::preallocate(*f, 0));                                         // never shrinks, never fails
    const std::string data = pattern(4 << 20, 11);
    CHECK(f->pwrite(data.data(), data.size(), 0)); CHECK(f->sync());
    auto all = fs::data_ranges(*f); CHECK_MSG(all, "%s", all.err.what.c_str());
    if (all) {
        std::uint64_t sum = 0, prev_end = 0; bool ordered = true;
        for (auto& x : *all) { if (x.offset < prev_end || !x.length) ordered = false; prev_end = x.offset + x.length; sum += x.length; }
        CHECK(ordered); CHECK(!all->empty()); CHECK_MSG(sum >= data.size() && prev_end >= data.size(), "ranges cover %llu of %zu", static_cast<unsigned long long>(sum), data.size());
    }
    auto ph = fs::punch_hole(*f, 1 << 20, 2 << 20);
    if (ok_or_unsupported(ph.err, "punch_hole")) {
        auto sz = f->size(); CHECK(sz && sz.value == data.size());        // the size stays
        std::string back(data.size(), 'x'); std::size_t got = 0;
        while (got < back.size()) { auto n = f->pread(back.data() + got, back.size() - got, got); if (!n || !n.value) break; got += n.value; }
        CHECK(got == back.size());
        CHECK(back.compare(0, 1 << 20, data, 0, 1 << 20) == 0);
        CHECK(back.compare(3 << 20, 1 << 20, data, 3 << 20, 1 << 20) == 0);
        CHECK(back.find_first_not_of('\0', 1 << 20) >= (3u << 20));       // the hole reads as zeros
        auto holes = fs::data_ranges(*f); CHECK(holes);
        if (holes) {
            std::uint64_t sum = 0; for (auto& x : *holes) sum += x.length;
            std::printf("  after punch: %zu data range(s), %llu of %zu bytes\n", holes->size(), static_cast<unsigned long long>(sum), data.size());
            CHECK(sum <= data.size() + (1u << 20));
        }
    }
    CHECK(fs::punch_hole(*f, 0, 0));
    auto e = os::file::open(r + "/empty.bin", os::rdwr | os::create); CHECK(e);
    if (e) { auto er = fs::data_ranges(*e); CHECK(er && er->empty()); }
    f->close(); if (e) e->close();
    CHECK(fs::remove_all(r));
}
TEST(fs_file_identity_and_links) {
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d;
    CHECK(os::write_file(r + "/a.txt", "a")); CHECK(os::write_file(r + "/b.txt", "b"));
    auto a = fs::file_id(r + "/a.txt"), b = fs::file_id(r + "/b.txt"); CHECK(a); CHECK(b);
    if (a && b) { CHECK(!(a.value == b.value)); CHECK(a->links == 1); CHECK(a->inode != 0 || a->device != 0); }
    CHECK(fs::same_file(r + "/a.txt", r + "/a.txt")); CHECK(!fs::same_file(r + "/a.txt", r + "/b.txt")); CHECK(!fs::same_file(r + "/a.txt", r + "/nope"));
    CHECK(!fs::file_id(r + "/nope"));
    auto h = fs::hardlink(r + "/a.txt", r + "/a2.txt"); CHECK_MSG(h, "%s", h.err.what.c_str());
    if (h) { CHECK(fs::same_file(r + "/a.txt", r + "/a2.txt")); auto a2 = fs::file_id(r + "/a2.txt"); CHECK(a2 && a2->links == 2); }
    auto dir = fs::file_id(r); CHECK(dir);
    CHECK(fs::remove_all(r));
}
TEST(fs_extended_attributes) {
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d, f = r + "/x.txt";
    CHECK(os::write_file(f, "content"));
    CHECK(!fs::xattr_set(f, "", "v")); CHECK(!fs::xattr_set(f, "a:b", "v")); CHECK(!fs::xattr_get(f, "a/b"));
    auto s = fs::xattr_set(f, "forge.test", "value-1");
    if (ok_or_unsupported(s.err, "xattr_set")) {
        auto g = fs::xattr_get(f, "forge.test"); CHECK_MSG(g, "%s", g.err.what.c_str()); if (g) CHECK(g.value == "value-1");
        std::string big = pattern(3000, 5);
        CHECK(fs::xattr_set(f, "forge.big", big)); auto gb = fs::xattr_get(f, "forge.big"); CHECK(gb && gb.value == big);
        CHECK(fs::xattr_set(f, "forge.test", "v2")); CHECK(fs::xattr_get(f, "forge.test").value == "v2");            // replace
        CHECK(fs::xattr_set(f, "forge.empty", "")); auto ge = fs::xattr_get(f, "forge.empty"); CHECK(ge && ge.value.empty());
        auto l = fs::xattr_list(f); CHECK_MSG(l, "%s", l.err.what.c_str());
        if (l && l->empty() && on_wine()) std::printf("  skip xattr_list: wine does not enumerate streams\n");
        else if (l) { for (const char* n : {"forge.test", "forge.big", "forge.empty"}) CHECK_MSG(std::find(l->begin(), l->end(), n) != l->end(), "%s not listed (%zu names)", n, l->size()); }
        CHECK(os::read_file(f).value == "content");                                                                  // the file itself is untouched
        CHECK(fs::xattr_remove(f, "forge.test")); CHECK(!fs::xattr_get(f, "forge.test")); CHECK(!fs::xattr_remove(f, "forge.test"));
        auto l2 = fs::xattr_list(f); CHECK(l2); if (l2) CHECK(std::find(l2->begin(), l2->end(), "forge.test") == l2->end());
    }
    CHECK(!fs::xattr_get(f, "forge.never-set"));
    CHECK(!fs::xattr_get(r + "/nope", "forge.test"));
    CHECK(fs::remove_all(r));
}
TEST(fs_rename_noreplace_and_exchange) {
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d;
    CHECK(os::write_file(r + "/a", "A")); CHECK(os::write_file(r + "/b", "B"));
    auto no = fs::rename_noreplace(r + "/a", r + "/b"); CHECK(!no);
    if (!no) CHECK_MSG(os::classify(no.err) == os::os_errc::already_exists, "%s", no.err.what.c_str());
    CHECK(os::read_file(r + "/a").value == "A"); CHECK(os::read_file(r + "/b").value == "B");
    auto yes = fs::rename_noreplace(r + "/a", r + "/c"); CHECK_MSG(yes, "%s", yes.err.what.c_str());
    CHECK(!os::exists(r + "/a")); CHECK(os::read_file(r + "/c").value == "A");
    CHECK(!fs::rename_noreplace(r + "/nope", r + "/z"));
    auto x = fs::rename_exchange(r + "/b", r + "/c");
    if (ok_or_unsupported(x.err, "rename_exchange")) { CHECK(os::read_file(r + "/b").value == "A"); CHECK(os::read_file(r + "/c").value == "B"); }
    else { CHECK(os::read_file(r + "/b").value == "B"); CHECK(os::read_file(r + "/c").value == "A"); }
#if defined(_WIN32) || defined(__FreeBSD__)
    CHECK(!x && os::is_unsupported(x.err));
#endif
    CHECK(fs::make_dirs(r + "/dir1"));
    auto dr = fs::rename_noreplace(r + "/dir1", r + "/dir2");
    if (ok_or_unsupported(dr.err, "rename_noreplace (directory)")) CHECK(fs::is_dir(r + "/dir2"));
    CHECK(fs::remove_all(r));
}
TEST(fs_sync_dir_and_anonymous_file) {
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d;
    CHECK(os::write_file(r + "/f", "x"));
    auto s = fs::sync_dir(r); CHECK_MSG(s, "%s", s.err.what.c_str());
    CHECK(!fs::sync_dir(r + "/nope"));
    auto before = os::list_dir(r);
    auto a = fs::anonymous_file(r); CHECK_MSG(a, "%s", a.err.what.c_str());
    if (a) {
        CHECK(a->write_all("hello anonymous", 15)); char b[16] = {}; auto n = a->pread(b, 15, 0); CHECK(n && n.value == 15); CHECK(std::string(b) == "hello anonymous");
        auto sz = a->size(); CHECK(sz && sz.value == 15);
#if !defined(_WIN32)
        auto during = os::list_dir(r); CHECK(before && during && before->size() == during->size());   // never visible (Windows: until close)
#endif
        a->close();
    }
    auto after = os::list_dir(r); CHECK(before && after && before->size() == after->size());
    auto t = fs::anonymous_file(); CHECK(t);
    CHECK(!fs::anonymous_file(r + "/no-such-dir"));
    CHECK(fs::remove_all(r));
}
TEST(fs_symlink_junction_read_link) {
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d;
    CHECK(os::write_file(r + "/t.txt", "target")); CHECK(fs::make_dirs(r + "/dir")); CHECK(os::write_file(r + "/dir/in.txt", "inside"));
    auto s = fs::symlink("t.txt", r + "/link.txt");
    if (!s) {
#if defined(_WIN32)
        std::printf("  skip file symlink: %s\n", s.err.what.c_str());   // needs Developer Mode or SeCreateSymbolicLinkPrivilege
#else
        CHECK_MSG(false, "%s", s.err.what.c_str());
#endif
    } else if (on_wine() && !os::exists(r + "/link.txt")) {
        std::printf("  skip file symlink: wine's CreateSymbolicLinkW creates nothing\n");
    } else {
        auto t = fs::read_link(r + "/link.txt"); CHECK_MSG(t, "%s", t.err.what.c_str()); if (t) CHECK_MSG(t.value == "t.txt", "[%s]", t.value.c_str());
        CHECK(os::stat(r + "/link.txt").is_link); CHECK(os::read_file(r + "/link.txt").value == "target");
        auto li = fs::file_id(r + "/link.txt", false), ti = fs::file_id(r + "/link.txt", true), fi = fs::file_id(r + "/t.txt");
        CHECK(li && ti && fi); if (li && ti && fi) { CHECK(ti.value == fi.value); CHECK(!(li.value == fi.value)); }
    }
    auto j = fs::junction(r + "/dir", r + "/jn");
    if (!j && on_wine()) std::printf("  skip junction: wine has no reparse points (%s)\n", j.err.what.c_str());
    else CHECK_MSG(j, "%s", j.err.what.c_str());
    if (j) {
        auto i = os::stat(r + "/jn"); CHECK(i.is_link); CHECK(i.is_dir);
        CHECK(os::read_file(r + "/jn/in.txt").value == "inside");
        auto t = fs::read_link(r + "/jn"); CHECK_MSG(t, "%s", t.err.what.c_str()); if (t) CHECK(fs::same_file(t.value, r + "/dir"));
        CHECK(os::remove(r + "/jn")); CHECK(os::read_file(r + "/dir/in.txt").value == "inside");   // the link goes, the target stays
    }
    CHECK(!fs::read_link(r + "/t.txt")); CHECK(!fs::read_link(r + "/nope"));
    CHECK(!fs::junction(r + "/dir", r + "/dir"));   // the link path exists
    CHECK(fs::remove_all(r));
}
TEST(long_paths) {   // 400+ characters: past MAX_PATH on Windows, ordinary elsewhere
    auto d = fs::make_temp_dir(); CHECK(d); std::string r = *d, p = r;
    for (int i = 0; i < 8; ++i) p += "/" + std::string(48, char('a' + i)) + std::to_string(i);
    CHECK(p.size() > 400);
    auto m = os::mkdirs(p); CHECK_MSG(m, "%s", m.err.what.c_str());
    CHECK(os::stat(p).is_dir);
    CHECK_MSG(os::write_file(p + "/file.txt", "deep"), "write");
    CHECK(os::read_file(p + "/file.txt").value == "deep");
    auto i = os::stat(p + "/file.txt"); CHECK(i.exists && i.is_file && i.size == 4);
    auto l = os::list_dir(p); CHECK(l); if (l) CHECK(l->size() == 1 && (*l)[0].name == "file.txt");
    CHECK(os::rename(p + "/file.txt", p + "/renamed.txt")); CHECK(os::exists(p + "/renamed.txt"));
    CHECK(fs::copy_file(p + "/renamed.txt", p + "/copy.txt")); CHECK(fs::file_id(p + "/copy.txt"));
    auto mp = os::mapping::map_file(p + "/renamed.txt"); CHECK(mp); if (mp) CHECK(mp->size() == 4);
    mp = {};
    CHECK(os::remove(p + "/copy.txt"));
    CHECK(os::remove_all(r)); CHECK(!os::exists(r));
}
TEST(clocks_across_suspend) {
    const auto b0 = os::boottime_ns(), a0 = os::awake_ns();
    CHECK(b0 > 0); CHECK(a0 > 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    const auto b1 = os::boottime_ns(), a1 = os::awake_ns();
    CHECK_MSG(b1 - b0 >= 20000000ull && b1 - b0 < 5000000000ull, "boottime advanced %llu ns", static_cast<unsigned long long>(b1 - b0));
    CHECK_MSG(a1 - a0 >= 20000000ull && a1 - a0 < 5000000000ull, "awake advanced %llu ns", static_cast<unsigned long long>(a1 - a0));
    CHECK(b1 + 1000000000ull >= a1);   // time asleep is only ever added
}
TEST(console_features_and_resize) {
    namespace con = os::console;
    CHECK(con::feature_sequence(con::feature::bracketed_paste, true) == "\x1b[?2004h");
    CHECK(con::feature_sequence(con::feature::bracketed_paste, false) == "\x1b[?2004l");
    CHECK(con::feature_sequence(con::feature::focus_events, true) == "\x1b[?1004h");
    CHECK(con::feature_sequence(con::feature::alternate_screen, false) == "\x1b[?1049l");
    CHECK(con::feature_sequence(con::feature::mouse, true).find("\x1b[?1006h") != std::string::npos);
    CHECK(con::feature_sequence(con::feature::mouse, false).find("\x1b[?1000l") != std::string::npos);
    if (!con::is_tty(con::stream::out)) {   // redirected (CI): nothing may be written
        for (auto f : {con::feature::mouse, con::feature::bracketed_paste, con::feature::focus_events, con::feature::alternate_screen}) {
            auto s = con::set_feature(f, true); CHECK(!s); if (!s) CHECK(!os::is_unsupported(s.err));
        }
    } else {
        CHECK(con::set_feature(con::feature::bracketed_paste, true)); CHECK(con::set_feature(con::feature::bracketed_paste, false));
    }
    CHECK(con::write_raw("", con::stream::err));
    con::resize_watcher w; CHECK(w.current().cols > 0 && w.current().rows > 0); CHECK(!w.poll());
}
