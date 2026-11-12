// forge::os phase 4: devices, clipboard.
#include <forge/os/devices.hpp>
#include <forge/os/clipboard.hpp>
#include "check.hpp"
using namespace forge;

TEST(devices_enumerate) {
    auto b = os::devices::block(); auto p = os::devices::pci(); auto u = os::devices::usb();
    std::printf("  block %s, pci %s, usb %s\n", b ? std::to_string(b->size()).c_str() : b.err.what.c_str(), p ? std::to_string(p->size()).c_str() : p.err.what.c_str(),
                u ? std::to_string(u->size()).c_str() : u.err.what.c_str());
    for (auto& d : b.value) {
        std::printf("  disk %s %llu B '%s' '%s'%s%s%s parts:", d.name.c_str(), (unsigned long long)d.size_bytes, d.vendor.c_str(), d.model.c_str(), d.removable ? " removable" : "", d.rotational ? " rotational" : "", d.read_only ? " ro" : "");
        for (auto& x : d.partitions) std::printf(" %s", x.c_str());
        std::printf("\n"); CHECK(!d.name.empty());
    }
    for (auto& d : p.value) {
        CHECK(d.vendor_id.size() == 6 && d.vendor_id.rfind("0x", 0) == 0); CHECK(d.device_id.rfind("0x", 0) == 0);
        CHECK(d.class_id.empty() || d.class_id.size() == 8); CHECK(!d.address.empty());
    }
    if (!p.value.empty()) { auto& d = p.value[0]; std::printf("  pci %s %s:%s class %s driver '%s'\n", d.address.c_str(), d.vendor_id.c_str(), d.device_id.c_str(), d.class_id.c_str(), d.driver.c_str()); }
    for (auto& d : u.value) {
        CHECK(d.vendor_id.size() == 4 && d.product_id.size() == 4); CHECK(!d.path.empty());
        std::printf("  usb %s %s:%s '%s' '%s' bus %d dev %d\n", d.path.c_str(), d.vendor_id.c_str(), d.product_id.c_str(), d.manufacturer.c_str(), d.product.c_str(), d.bus, d.devnum);
    }
#if defined(__linux__) || defined(_WIN32) || defined(__FreeBSD__) || (defined(__APPLE__) && !TARGET_OS_IPHONE)
    CHECK(b);   // may be empty in containers, but readable
    CHECK(p || os::is_unsupported(p.err)); CHECK(u || os::is_unsupported(u.err));   // Linux containers may lack /sys/bus/*
#if !defined(__linux__)
    CHECK(p); CHECK(u);
#endif
#if defined(_WIN32) || defined(__FreeBSD__) || defined(__APPLE__)
    CHECK(!b.value.empty());   // a real or virtual machine always has a boot disk
#endif
#else
    CHECK(!b ? os::is_unsupported(b.err) : true);
#endif
}
TEST(freebsd_geom_partitions) {   // fixture: ada1 must not claim ada10's partitions, MBR slices and BSD labels count
    const char* conftxt =
        "0 DISK ada1 21474836480 512 hd 16 sc 63\n"
        "1 PART ada1p1 524288 512 i 1 o 20480 ty freebsd-boot xs GPT xt 83bd6b9d\n"
        "1 PART ada1p2 21474000000 512 i 2 o 544768 ty freebsd-ufs xs GPT xt 516e7cb6\n"
        "0 DISK ada10 1000204886016 512 hd 16 sc 63\n"
        "1 PART ada10p1 1000204000000 512 i 1 o 40960 ty freebsd-zfs xs GPT xt 516e7cba\n"
        "0 DISK da0 8053063680 512 hd 255 sc 63\n"
        "1 PART da0s1 8053000000 512 i 1 o 32256 ty freebsd xs MBR xt 165\n"
        "2 PART da0s1a 8000000000 512 i 1 o 0 ty freebsd-ufs xs BSD xt 7\n"
        "0 DISK nvd0 512110190592 512 hd 0 sc 0\n"
        "1 PART nvd0p1 209715200 512 i 1 o 20480 ty efi xs GPT xt c12a7328\n"
        "0 DISK cd0 0 2048 hd 0 sc 0\n";
    auto v = os::devices::geom_conftxt_disks(conftxt);
    CHECK(v.size() == 5); if (v.size() != 5) return;
    CHECK(v[0].name == "ada1" && v[0].size_bytes == 21474836480ull); CHECK((v[0].partitions == std::vector<std::string>{"ada1p1", "ada1p2"}));
    CHECK(v[1].name == "ada10"); CHECK((v[1].partitions == std::vector<std::string>{"ada10p1"}));
    CHECK((v[2].partitions == std::vector<std::string>{"da0s1", "da0s1a"}));
    CHECK((v[3].partitions == std::vector<std::string>{"nvd0p1"})); CHECK(v[4].partitions.empty());
    CHECK(!os::devices::geom_partition_of("ada1", "ada10p1")); CHECK(!os::devices::geom_partition_of("da0", "da0p")); CHECK(os::devices::geom_partition_of("da0", "da0s2"));
}
TEST(clipboard_roundtrip_or_unsupported) {
    if (!os::clipboard::available()) { CHECK(os::is_unsupported(os::clipboard::get().err)); return; }
    auto old = os::clipboard::get();
    CHECK(os::clipboard::set("forge clipboard ✓"));
    auto g = os::clipboard::get(); CHECK(g); CHECK(g.value == "forge clipboard ✓");
    if (old) os::clipboard::set(old.value);
}
// ---- volumes, trash, single instance, autostart, open, notifications, displays, idle, preferences, hot-plug, clipboard backend (begin)
#include <forge/os/volumes.hpp>
#include <forge/os/trash.hpp>
#include <forge/os/single_instance.hpp>
#include <forge/os/autostart.hpp>
#include <forge/os/shell_open.hpp>
#include <forge/os/notification.hpp>
#include <forge/os/fs.hpp>
#include <forge/os/display.hpp>
#include <forge/os/idle.hpp>
#include <forge/os/preferences.hpp>
#include <thread>
#include <forge/os/device_watch.hpp>
#include <forge/os/clipboard.hpp>
TEST(volumes_list) {
    auto v = os::volumes(); CHECK_MSG(bool(v) || os::is_unsupported(v.err), "%s", v.err.what.c_str()); if (!v) return;
    for (auto& x : *v) std::printf("  vol %-24s %-10s dev %-28s total %llu avail %llu%s%s%s%s label '%s'\n", x.mount_point.c_str(), x.fs_type.c_str(), x.device.c_str(), (unsigned long long)x.total, (unsigned long long)x.available,
                                   x.removable ? " removable" : "", x.network ? " network" : "", x.read_only ? " ro" : "", x.ram ? " ram" : "", x.label.c_str());
    CHECK(!v->empty());
    for (auto& x : *v) { CHECK(!x.mount_point.empty()); CHECK(x.free <= x.total); CHECK(x.available <= x.total); }
    auto all = os::volumes(true); CHECK(all && all->size() >= v->size());
    auto here = os::volume_for(os::temp_dir()); CHECK_MSG(bool(here), "%s", here.err.what.c_str());
    if (here) { std::printf("  temp dir is on %s (%s)\n", here->mount_point.c_str(), here->fs_type.c_str()); CHECK(!here->mount_point.empty()); }
    os::volume_info m;
    CHECK(os::detail::parse_mountinfo_line("36 35 98:0 /mnt1 /mnt/with\\040space rw,noatime master:1 - ext3 /dev/root rw,errors=continue", m));
    CHECK(m.mount_point == "/mnt/with space" && m.fs_type == "ext3" && m.device == "/dev/root" && !m.read_only && !m.pseudo && !m.network);
    CHECK(os::detail::parse_mountinfo_line("50 30 0:44 / /net ro,relatime - nfs4 srv:/export rw,vers=4.2", m)); CHECK(m.network && m.read_only && m.device == "srv:/export");
    CHECK(os::detail::parse_mountinfo_line("22 30 0:21 / /proc rw - proc proc rw", m)); CHECK(m.pseudo);
    CHECK(!os::detail::parse_mountinfo_line("garbage", m)); CHECK(!os::detail::parse_mountinfo_line("1 2 3:4 / /x rw shared:1", m));
}
TEST(trash_moves_to_trash) {
    namespace dt = os::detail_trash;
    CHECK(dt::escape_path("/home/u/a b/ü.txt") == "/home/u/a%20b/%C3%BC.txt");
    CHECK(dt::trashinfo("/x/y z", "2026-11-12T10:00:00") == "[Trash Info]\nPath=/x/y%20z\nDeletionDate=2026-11-12T10:00:00\n");
    CHECK(!os::trash(""));
    auto d = os::fs::make_temp_dir("ftrash-"); CHECK(d); if (!d) return;
    auto missing = os::trash(*d + "/does-not-exist");
    CHECK(!missing);
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IPHONE)
    CHECK(os::is_unsupported(missing.err)); CHECK(!os::trash_available());
#elif defined(__linux__) || defined(__FreeBSD__)
    // The real implementation against a private XDG_DATA_HOME: nothing lands in the user's trash.
    forge_test::set_env("XDG_DATA_HOME", (*d + "/xdg").c_str());
    CHECK(os::trash_available());
    CHECK(os::write_file(*d + "/victim one.txt", "data")); CHECK(os::mkdirs(*d + "/dir/sub")); CHECK(os::write_file(*d + "/dir/sub/f", "x"));
    auto r = os::trash(*d + "/victim one.txt"); CHECK_MSG(bool(r), "%s", r.err.what.c_str());
    if (r) {
        CHECK(!os::stat(*d + "/victim one.txt").exists); CHECK(*r == *d + "/xdg/Trash/files/victim one.txt"); CHECK(os::read_file(*r).value == "data");
        auto info = os::read_file(*d + "/xdg/Trash/info/victim one.txt.trashinfo"); CHECK(info);
        if (info) { CHECK(info->rfind("[Trash Info]\nPath=", 0) == 0); CHECK(info->find("victim%20one.txt\nDeletionDate=20") != std::string::npos); }
    }
    CHECK(os::write_file(*d + "/victim one.txt", "second"));
    auto r2 = os::trash(*d + "/victim one.txt"); CHECK(r2); if (r2) CHECK(*r2 == *d + "/xdg/Trash/files/victim one.txt.1");   // same name: a numbered copy
    auto r3 = os::trash(*d + "/dir"); CHECK(r3); if (r3) CHECK(os::read_file(*r3 + "/sub/f").value == "x");
    forge_test::set_env("XDG_DATA_HOME", nullptr);
#else
    // Windows recycle bin / macOS trash: a real move of a throw-away file (no UI). Unsupported is accepted where
    // the session has no recycle bin.
    CHECK(os::write_file(*d + "/forge-trash-test.txt", "data"));
    auto r = os::trash(*d + "/forge-trash-test.txt");
    std::printf("  trash: %s\n", r ? ("ok '" + *r + "'").c_str() : r.err.what.c_str());
    CHECK(r || os::is_unsupported(r.err)); if (r) CHECK(!os::stat(*d + "/forge-trash-test.txt").exists);
#endif
    os::fs::remove_all(*d);
}
TEST(single_instance_guard) {
    using si = os::single_instance;
    CHECK(si::valid_name("forge.app-1")); CHECK(!si::valid_name("")); CHECK(!si::valid_name("a/b")); CHECK(!si::valid_name(std::string(65, 'a'))); CHECK(!si::valid_name("a b"));
    CHECK(!si::acquire("../evil"));
    std::string name = "forge-test-" + std::to_string(os::pid()) + "-" + std::to_string(os::monotonic_ns() % 100000);
    auto a = si::acquire(name); CHECK_MSG(bool(a), "%s", a.err.what.c_str()); if (!a) return;
    CHECK(a->held()); CHECK(a->name() == name);
    auto b = si::acquire(name); CHECK(!b); CHECK(si::already_running(b.err)); CHECK(!os::is_unsupported(b.err));
#if !defined(_WIN32)
    CHECK(si::holder_pid(name) == long(os::pid()));
#endif
    si moved = std::move(*a); CHECK(moved.held()); CHECK(!a->held());
    moved.release(); CHECK(!moved.held());
    auto c = si::acquire(name); CHECK(c);   // free again after release
#if !defined(_WIN32)
    if (c) { c->release(); os::remove(si::lock_path(name)); }
#endif
}
TEST(autostart_entries) {
    namespace as = os::autostart;
    as::entry e; e.name = "com.forge.test"; e.argv = {"/opt/forge app/bin/forge", "--serve", "a\"b", "100%", "$HOME", "back\\slash"}; e.description = "Forge test";
    CHECK(as::validate(e)); { auto bad = e; bad.name = "../x"; CHECK(!as::validate(bad)); CHECK(!as::enable(bad)); } { auto bad = e; bad.argv = {}; CHECK(!as::validate(bad)); } { auto bad = e; bad.argv[1] = "a\nb"; CHECK(!as::validate(bad)); }
    auto de = as::render_desktop_entry(e);
    CHECK(de.find("[Desktop Entry]\nType=Application\nName=com.forge.test\nComment=Forge test\n") == 0);
    CHECK(de.find("Exec=\"/opt/forge app/bin/forge\" \"--serve\" \"a\\\\\"b\" \"100%%\" \"\\\\$HOME\" \"back\\\\\\\\slash\"\n") != std::string::npos);
    auto pl = as::render_launch_agent(e); CHECK(pl.find("<key>RunAtLoad</key>\n    <true/>") != std::string::npos); CHECK(pl.find("<string>a&quot;b</string>") != std::string::npos);
    CHECK(as::render_command_line(e) == "\"/opt/forge app/bin/forge\" --serve \"a\\\"b\" 100% $HOME back\\slash");
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IPHONE)
    CHECK(os::is_unsupported(as::enable(e).err)); CHECK(os::is_unsupported(as::disable(e).err)); CHECK(os::is_unsupported(as::is_enabled(e).err));
#else
#if defined(_WIN32)
    e.dir = "Software\\ForgeTest\\Autostart" + std::to_string(os::pid());
#else
    auto d = os::fs::make_temp_dir("fauto-"); CHECK(d); if (!d) return; e.dir = *d + "/autostart";
#endif
    auto en0 = as::is_enabled(e); CHECK(en0 && !*en0); CHECK(as::disable(e));
    auto en = as::enable(e); CHECK_MSG(bool(en), "%s", en.err.what.c_str());
    auto en1 = as::is_enabled(e); CHECK(en1 && *en1); CHECK(as::enable(e));   // idempotent
    std::printf("  autostart at %s\n", as::location(e).c_str());
    CHECK(as::disable(e)); auto en2 = as::is_enabled(e); CHECK(en2 && !*en2);
#if defined(_WIN32)
    RegDeleteKeyW(HKEY_CURRENT_USER, os::widen(e.dir).c_str());
#else
    os::fs::remove_all(*d);
#endif
#endif
}
TEST(shell_open_policy) {
    CHECK(os::check_url("https://example.com/a?b=c")); CHECK(os::check_url("mailto:someone@example.com")); CHECK(os::check_url("HTTP://EXAMPLE.COM"));
    for (const char* bad : {"", "example.com", "/etc/passwd", "file:///etc/passwd", "javascript:alert(1)", "-oProxyCommand=x", "https://", "https:///x", "http://a b", "http://a\nb", "ssh://host", "C:\\Windows\\cmd.exe", "data:text/html,x"})
        CHECK_MSG(!os::check_url(bad), "accepted '%s'", bad);
    CHECK(os::check_url("ssh://host", {"ssh"})); CHECK(!os::open_url("javascript:alert(1)")); CHECK(!os::open_path("")); CHECK(!os::open_path("/no/such/file/anywhere")); CHECK(!os::reveal_path("/no/such/file/anywhere"));
    std::vector<std::pair<std::string, os::open_kind>> seen;
    os::set_open_hook([&](const std::string& t, os::open_kind k) { seen.push_back({t, k}); return os::status{}; });
    CHECK(os::can_open()); CHECK(os::open_url("https://example.com/")); CHECK(!os::open_url("file:///x")); CHECK(os::open_path(os::temp_dir())); CHECK(os::reveal_path(os::temp_dir()));
    os::set_open_hook({});
    CHECK(seen.size() == 3); if (seen.size() == 3) { CHECK(seen[0].first == "https://example.com/" && seen[0].second == os::open_kind::url); CHECK(seen[1].second == os::open_kind::path); CHECK(seen[2].second == os::open_kind::reveal); }
    auto cmd = os::opener_command("https://example.com/"); std::printf("  can_open %d, opener:", os::can_open()); for (auto& a : cmd) std::printf(" %s", a.c_str()); std::printf("\n");
    if (!os::can_open()) { auto r = os::open_url("https://example.com/"); CHECK(!r && os::is_unsupported(r.err)); }   // headless: a clean error, nothing launched
    else if (os::getenv("FORGE_UI_TESTS") == "1") CHECK(os::open_url("https://example.com/"));
    CHECK(os::detail_open::file_uri("/a b/ü'x") == "file:///a%20b/%C3%BC%27x");
}
TEST(notification_policy) {
    namespace dn = os::detail_notify;
    CHECK(dn::gvariant_string("it's a \\ test\nline") == "'it\\'s a \\\\ test\\nline'"); CHECK(dn::applescript_string("say \"hi\"\\") == "\"say \\\"hi\\\"\\\\\"");
    CHECK(!os::notify("", "")); CHECK(!os::notify(std::string(2000, 't'), "b"));
    std::vector<os::notification> seen;
    os::set_notification_hook([&](const os::notification& n) { seen.push_back(n); return os::status{}; });
    CHECK(os::notifications_available()); CHECK(os::notify("Title", "Body")); os::set_notification_hook({});
    CHECK(seen.size() == 1 && seen[0].title == "Title" && seen[0].body == "Body" && seen[0].app_name == "forge");
    os::notification n; n.title = "forge test"; n.body = "notification test"; n.urgency = os::notification_urgency::low; n.timeout_ms = 1000;
    auto cmd = os::notification_command(n); std::printf("  notifications available %d, helper '%s'\n", os::notifications_available(), cmd.empty() ? "" : cmd[0].c_str());
    if (!os::notifications_available()) { auto r = os::notify(n); CHECK(!r && os::is_unsupported(r.err)); }
    else if (os::getenv("FORGE_UI_TESTS") == "1") { auto r = os::notify(n); CHECK_MSG(bool(r), "%s", r.err.what.c_str()); }
}
TEST(display_enumeration) {
    auto d = os::displays();
    std::printf("  displays: %s\n", d ? std::to_string(d->size()).c_str() : d.err.what.c_str());
    CHECK(d || os::is_unsupported(d.err));
    for (auto& x : d.value) {
        std::printf("  %s '%s' %dx%d+%d+%d %dx%d mm dpi %.1f scale %.2f %.1f Hz%s\n", x.name.c_str(), x.model.c_str(), x.width, x.height, x.x, x.y, x.width_mm, x.height_mm, x.dpi, x.scale, x.refresh_hz, x.primary ? " primary" : "");
        CHECK(x.width > 0 && x.height > 0); CHECK(x.scale > 0); CHECK(!x.name.empty());
    }
    // EDID fixture: 1920x1080@60 (148.5 MHz), 527x296 mm, vendor "DEL", name "FORGE TEST".
    unsigned char e[128] = {0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0};
    e[8] = 0x10; e[9] = 0xac; e[21] = 53; e[22] = 30;
    unsigned char dtd[18] = {0x02, 0x3a, 0x80, 0x18, 0x71, 0x38, 0x2d, 0x40, 0x58, 0x2c, 0x45, 0x00, 0x0f, 0x28, 0x21, 0x00, 0x00, 0x1e};
    std::memcpy(e + 54, dtd, 18);
    unsigned char nm[18] = {0, 0, 0, 0xfc, 0, 'F', 'O', 'R', 'G', 'E', ' ', 'T', 'E', 'S', 'T', '\n', ' ', ' '};
    std::memcpy(e + 72, nm, 18);
    auto p = os::detail_display::parse_edid(std::string_view(reinterpret_cast<const char*>(e), sizeof e));
    CHECK(p.ok); CHECK(p.vendor == "DEL"); CHECK(p.name == "FORGE TEST"); CHECK(p.width == 1920 && p.height == 1080); CHECK(p.width_mm == 527 && p.height_mm == 296);
    CHECK_MSG(p.refresh_hz > 59.9 && p.refresh_hz < 60.1, "%.3f", p.refresh_hz);
    CHECK(!os::detail_display::parse_edid("short").ok); CHECK(!os::detail_display::parse_edid(std::string(128, 'x')).ok);
    os::display_info di; di.width = 1920; di.width_mm = 527; os::detail_display::finish(di); CHECK(di.dpi > 92 && di.dpi < 93);
    os::set_display_hook([] { os::display_info h; h.name = "hook"; h.width = 100; h.height = 50; return std::vector<os::display_info>{h}; });
    auto hd = os::displays(); CHECK(hd && hd->size() == 1 && (*hd)[0].name == "hook"); os::set_display_hook({});
}
TEST(idle_and_lock_state) {
    auto i = os::idle_seconds(); auto l = os::screen_locked();
    std::printf("  idle: %s, locked: %s\n", i ? std::to_string(*i).c_str() : i.err.what.c_str(), l ? (*l ? "yes" : "no") : l.err.what.c_str());
    CHECK(i || os::is_unsupported(i.err)); CHECK(l || os::is_unsupported(l.err)); if (i) CHECK(*i >= 0);
    auto s = os::detail_idle::parse_loginctl("IdleHint=yes\nIdleSinceHintMonotonic=123456789\nLockedHint=no\n");
    CHECK(s.found && s.idle && !s.locked && s.idle_since_us == 123456789ull);
    auto s2 = os::detail_idle::parse_loginctl("LockedHint=yes\n"); CHECK(s2.found && s2.locked && !s2.idle);
    CHECK(!os::detail_idle::parse_loginctl("Failed to get session: no session\n").found);
}
TEST(preferences_store) {
    using pr = os::preferences;
    CHECK(!pr::open("")); CHECK(!pr::open("a/b")); CHECK(!pr::open("..")); CHECK(pr::valid_name("com.forge.App_1", 64));
    auto m = pr::parse("# c\na=1\nbad line\nb=x\\ny\\\\z\n=novalue\nc=\r\nwith space=1\n");
    CHECK(m.size() == 3 && m["a"] == "1" && m["b"] == "x\ny\\z" && m["c"].empty()); CHECK(pr::parse(pr::render(m)) == m);
    auto d = os::fs::make_temp_dir("fprefs-"); CHECK(d); if (!d) return;
    pr::options o; o.dir = *d;
    auto p = pr::open("forge-test", o); CHECK(p); if (!p) return;
    CHECK(std::string(p->backend()) == "file");
    auto none = p->get("missing"); CHECK(none && !*none); CHECK(p->get_or("missing", "dflt") == "dflt"); CHECK(p->remove("missing")); CHECK(p->keys() && p->keys()->empty());
    CHECK(p->set("name", "värde = \"x\"\nline2\\")); CHECK(p->set_int("count", -42)); CHECK(p->set_bool("flag", true)); CHECK(p->set("empty", ""));
    CHECK(p->get_or("name", "") == "värde = \"x\"\nline2\\"); CHECK(p->get_int("count", 0) == -42); CHECK(p->get_bool("flag", false)); CHECK(p->get_int("name", 7) == 7);
    auto e = p->get("empty"); CHECK(e && *e && (*e)->empty());
    CHECK(!p->set("bad key", "x")); CHECK(!p->get("a=b")); CHECK(!p->set("k", std::string("a\0b", 3)));
    auto ks = p->keys(); CHECK(ks && (*ks == std::vector<std::string>{"count", "empty", "flag", "name"}));
    auto again = pr::open("forge-test", o); CHECK(again && again->get_int("count", 0) == -42);   // persisted
    CHECK(p->remove("count")); CHECK(p->get_int("count", 5) == 5);
    {   // concurrent writers keep each other's keys
        std::vector<std::thread> th;
        for (int t = 0; t < 4; ++t) th.emplace_back([&, t] { auto q = pr::open("forge-test", o); for (int i = 0; i < 20; ++i) q->set("t" + std::to_string(t) + "_" + std::to_string(i), "v"); });
        for (auto& x : th) x.join();
        auto k2 = p->keys(); CHECK(k2 && k2->size() == 3 + 80);
    }
    CHECK(p->clear()); CHECK(p->keys() && p->keys()->empty()); os::fs::remove_all(*d);
    // The OS store (registry / CFPreferences / XDG file), with a throw-away application id.
    std::string app = "forge-test-prefs-" + std::to_string(os::pid());
#if !defined(_WIN32) && !defined(__APPLE__)
    auto cfg = os::fs::make_temp_dir("fprefs-xdg-"); CHECK(cfg); if (cfg) forge_test::set_env("XDG_CONFIG_HOME", cfg->c_str());
#endif
    auto n = pr::open(app);
    std::printf("  native preferences: %s\n", n ? (std::string(n->backend()) + " at " + n->location()).c_str() : n.err.what.c_str());
    CHECK(n || os::is_unsupported(n.err));
    if (n) {
        CHECK(n->set("greeting", "hello wörld")); CHECK(n->set_int("n", 123456789012345ll)); CHECK(n->get_or("greeting", "") == "hello wörld"); CHECK(n->get_int("n", 0) == 123456789012345ll);
        auto nk = n->keys(); CHECK(nk && nk->size() == 2); CHECK(n->remove("greeting")); auto g = n->get("greeting"); CHECK(g && !*g);
        CHECK(n->clear()); auto nk2 = n->keys(); CHECK(nk2 && nk2->empty());
    }
#if !defined(_WIN32) && !defined(__APPLE__)
    forge_test::set_env("XDG_CONFIG_HOME", nullptr); if (cfg) os::fs::remove_all(*cfg);
#endif
}
TEST(device_monitor_events) {
    namespace dv = os::devices; namespace dd = os::devices::detail_dev;
    dv::device_event e;
    std::string add("add@/devices/pci0000:00/usb1/1-2\0ACTION=add\0DEVPATH=/devices/pci0000:00/usb1/1-2\0SUBSYSTEM=usb\0DEVNAME=bus/usb/001/004\0SEQNUM=1\0", 120);
    CHECK(dd::parse_uevent(add, e)); CHECK(e.action == dv::device_action::added && e.subsystem == "usb" && e.name == "bus/usb/001/004");
    std::string rem("remove@/devices/virtual/block/loop3\0ACTION=remove\0DEVPATH=/devices/virtual/block/loop3\0SUBSYSTEM=block\0", 102);
    CHECK(dd::parse_uevent(rem, e)); CHECK(e.action == dv::device_action::removed && e.subsystem == "block" && e.name == "loop3");
    CHECK(!dd::parse_uevent(std::string("libudev\0\xfe\xed\xca\xfe junk", 17), e)); CHECK(!dd::parse_uevent("no nul here", e)); CHECK(!dd::parse_uevent(std::string("x@/y\0SUBSYSTEM=usb\0", 19), e));
    CHECK(!dd::parse_uevent(std::string("bind@/d\0ACTION=bind\0DEVPATH=/d\0", 31), e));
    CHECK(dd::parse_devd("+ugen0.2 at bus=0 hubaddr=1 port=2 vendor=0x80ee on uhub0\n", e)); CHECK(e.action == dv::device_action::added && e.subsystem == "usb" && e.name == "ugen0.2");
    CHECK(dd::parse_devd("-da0 at scbus2 on umass-sim0\n", e)); CHECK(e.action == dv::device_action::removed && e.subsystem == "block" && e.name == "da0");
    CHECK(dd::parse_devd("!system=DEVFS subsystem=CDEV type=CREATE cdev=da0s1\n", e)); CHECK(e.action == dv::device_action::added && e.name == "da0s1");
    CHECK(dd::parse_devd("!system=IFNET subsystem=ue0 type=ATTACH\n", e)); CHECK(e.subsystem == "net" && e.name == "ue0");
    CHECK(!dd::parse_devd("!system=ACPI subsystem=ACAD type=\\_SB_.AC notify=0x01\n", e)); CHECK(!dd::parse_devd("", e)); CHECK(!dd::parse_devd("+", e));
    std::vector<dv::device_event> diff; dd::diff_names({"a", "b"}, {"b", "c"}, "usb", diff);
    CHECK(diff.size() == 2 && diff[0].action == dv::device_action::added && diff[0].name == "c" && diff[1].action == dv::device_action::removed && diff[1].name == "a");
    auto m = dv::monitor::open();
#if defined(__APPLE__) && TARGET_OS_IPHONE
    CHECK(!m && os::is_unsupported(m.err));
#else
    CHECK_MSG(bool(m), "%s", m.err.what.c_str());
    if (m) { std::printf("  device monitor: %s\n", (*m)->backend()); auto ev = (*m)->poll(100); std::printf("  %zu event(s)\n", ev.size()); for (auto& x : ev) CHECK(!x.subsystem.empty()); }
    dv::monitor_options o; o.force_polling = true; o.poll_interval_ms = 20;
    auto p = dv::monitor::open(o); CHECK(p);
    if (p) { CHECK(std::string((*p)->backend()) == "polling"); auto t0 = os::monotonic_ns(); (*p)->poll(80); CHECK(os::monotonic_ns() - t0 < 5000000000ull); }
#endif
}
TEST(clipboard_backend_hook) {
    std::string store = "initial"; int sets = 0;
    os::clipboard::backend b;
    b.get = [&]() -> os::result<std::string> { return {store, {}}; };
    b.set = [&](const std::string& t) -> os::status { store = t; ++sets; return {}; };
    os::clipboard::set_backend(b);
    CHECK(os::clipboard::available()); auto g = os::clipboard::get(); CHECK(g && *g == "initial");
    CHECK(os::clipboard::set("from the hook ✓")); CHECK(store == "from the hook ✓" && sets == 1); CHECK(*os::clipboard::get() == "from the hook ✓");
    os::clipboard::set_backend({});
    if (!os::clipboard::available()) { CHECK(os::is_unsupported(os::clipboard::get().err)); CHECK(os::is_unsupported(os::clipboard::set("x").err)); }
}
// ---- volumes, trash, single instance, autostart, open, notifications, displays, idle, preferences, hot-plug, clipboard backend (end)
