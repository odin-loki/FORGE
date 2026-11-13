// forge::os desktop services over D-Bus (os::desktop_bus) and the public calls built on them: notify, open_url,
// reveal_path, idle / lock state, sleep inhibition, power profile. A mock bus stands in for the session and system
// buses, so this runs on headless CI; without it every call must report `unsupported`, never hang.
#include <forge/os/desktop_bus.hpp>
#include <forge/os/notification.hpp>
#include <forge/os/shell_open.hpp>
#include <forge/os/idle.hpp>
#include <forge/os/power.hpp>
#include "check.hpp"
#include "dbus_mock.hpp"
using namespace forge;
namespace bus = forge::os::desktop_bus;

TEST(desktop_bus_without_a_bus) {
    bus::set_addresses("", "");   // as on a server: neither bus exists
    CHECK(!bus::has_bus(os::dbus::bus::session)); CHECK(!bus::has_bus(os::dbus::bus::system));
    auto t0 = os::monotonic_ns();
    { auto r = bus::notify("a", "t", "b"); CHECK(!r); CHECK(os::is_unsupported(r.err)); }
    { auto r = bus::open_uri("https://example.com/"); CHECK(!r); CHECK(os::is_unsupported(r.err)); }
    { auto r = bus::show_items({"file:///tmp"}); CHECK(!r); CHECK(os::is_unsupported(r.err)); }
    { auto r = bus::login_session(); CHECK(!r); CHECK(os::is_unsupported(r.err)); }
    { auto r = bus::screensaver_active(); CHECK(!r); CHECK(os::is_unsupported(r.err)); }
    { auto r = bus::mutter_idle_ms(); CHECK(!r); CHECK(os::is_unsupported(r.err)); }
    { auto r = bus::inhibit_sleep_fd("sleep", "forge", "test"); CHECK(!r); CHECK(os::is_unsupported(r.err)); }
    { auto r = bus::inhibit_idle("forge", "test"); CHECK(!r); CHECK(os::is_unsupported(r.err)); }
    { auto r = bus::portal_setting("org.freedesktop.appearance", "color-scheme"); CHECK(!r); CHECK(os::is_unsupported(r.err)); }
    { auto r = bus::power_profile(); CHECK(!r); CHECK(os::is_unsupported(r.err)); }
    CHECK(os::monotonic_ns() - t0 < 5'000'000'000ull);
    // an address nobody listens on behaves the same
    bus::set_addresses("unix:path=/nonexistent/forge/session", "unix:path=/nonexistent/forge/system");
    { auto r = bus::notify("a", "t", "b"); CHECK(!r); CHECK(os::is_unsupported(r.err)); }
    { auto r = bus::power_profile(); CHECK(!r); CHECK(os::is_unsupported(r.err)); }
    bus::clear_addresses();
}

TEST(sleep_inhibitor_everywhere) {
    // The real thing on this machine: either something holds the machine awake, or the OS says why not.
    auto s = os::power::inhibit_sleep("forge test");
    if (!s) { std::printf("  skip: %s\n", s.err.what.c_str()); CHECK(os::is_unsupported(s.err)); return; }
    std::printf("  inhibitor kind: %s\n", s->kind().c_str());
    CHECK(s->active()); CHECK(!s->kind().empty());
#if defined(_WIN32)
    CHECK(s->kind() == "power-request");
#elif defined(FORGE_OS_MACOS)
    CHECK(s->kind() == "iopm-assertion" || s->kind() == "not_required");
#endif
    os::power::sleep_inhibitor moved = std::move(*s); CHECK(moved.active()); CHECK(!s->active());
    moved.release(); moved.release(); CHECK(!moved.active());
    auto again = os::power::inhibit_sleep(""); CHECK(again);   // an empty reason gets a default name
}

#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS) && !defined(FORGE_OS_ANDROID)
TEST(desktop_bus_against_mock) {
    mock_bus mock; CHECK(mock.start());
    const std::string addr = "unix:path=" + mock.path;
    bus::set_addresses(addr, addr);
    struct restore { ~restore() { bus::clear_addresses(); } } guard;
    CHECK(bus::has_bus(os::dbus::bus::session));
    { auto r = bus::notify("forge", "Title", "Body", 2, 1500); CHECK_MSG(bool(r), "%s", r.err.what.c_str()); CHECK(r.value == 7); }
    {
        const auto& m = mock.calls.back(); CHECK(m.member == "Notify"); CHECK(m.signature == "susssasa{sv}i");
        if (m.body.size() == 8) { CHECK(m.body[0].s == "forge"); CHECK(m.body[3].s == "Title"); CHECK(m.body[4].s == "Body"); CHECK(m.body[7].as_int() == 1500);
                                  const value* u = m.body[6].find("urgency"); CHECK(u && u->unwrap().type == 'y' && u->unwrap().u == 2); }
    }
    { auto r = bus::open_uri("https://example.com/a?b=c"); CHECK(r); const auto& m = mock.calls.back(); CHECK(m.iface == "org.freedesktop.portal.OpenURI"); CHECK(m.signature == "ssa{sv}"); CHECK(m.body[1].s == "https://example.com/a?b=c"); }
    { auto r = bus::show_items({"file:///tmp/a%20b", "file:///tmp/c"}); CHECK(r); const auto& m = mock.calls.back(); CHECK(m.signature == "ass"); CHECK(m.body[0].items.size() == 2 && m.body[0].items[0].s == "file:///tmp/a%20b"); }
    { auto r = bus::login_session(); CHECK(r); if (r) { CHECK(r->idle && r->locked); CHECK(r->idle_since_monotonic_us == 1); } }
    { auto r = bus::screensaver_active(); CHECK(r); CHECK(r.value); }
    { auto r = bus::mutter_idle_ms(); CHECK(r); CHECK(r.value == 4200); }
    { auto r = bus::portal_setting("org.freedesktop.appearance", "color-scheme"); CHECK(r); CHECK(r && r->type == 'u' && r->u == 1); }   // doubly wrapped variant
    { auto r = bus::power_profile(); CHECK(r); CHECK(r.value == "performance"); }
    { auto r = bus::inhibit_idle("forge", "busy"); CHECK(r); CHECK(mock.calls.back().member == "Inhibit"); }   // the connection closes here: released
    // a descriptor arrives with the logind reply; closing it is seen by the other side
    {
        auto fd = bus::inhibit_sleep_fd("sleep:idle", "forge", "testing"); CHECK_MSG(bool(fd), "%s", fd.err.what.c_str());
        if (fd) {
            CHECK(fd.value > 2); CHECK(::fcntl(fd.value, F_GETFD) != -1);
            const auto& m = mock.calls.back(); CHECK(m.iface == "org.freedesktop.login1.Manager"); CHECK(m.signature == "ssss"); CHECK(m.body[0].s == "sleep:idle"); CHECK(m.body[3].s == "block");
            int rd = mock.inhibit_read_end.load(); CHECK(rd >= 0);
            struct pollfd p{rd, POLLIN, 0}; CHECK(::poll(&p, 1, 0) == 0);   // still held
            ::close(fd.value);
            CHECK(::poll(&p, 1, 2000) == 1);                                  // hang-up: the inhibitor is gone
        }
    }
#if !defined(__APPLE__)   // macOS: notify / reveal_path use Notification Center and Finder, not the session bus
    // The public calls take the native path (no helper program is installed for them to find here).
    os::notification n; n.title = "Hello"; n.body = "from the mock"; n.app_name = "forge-test"; n.timeout_ms = 10;
    { auto r = os::notify(n); CHECK_MSG(bool(r), "%s", r.err.what.c_str()); CHECK(mock.calls.back().member == "Notify"); CHECK(mock.calls.back().body[0].s == "forge-test"); }
    CHECK(os::notifications_available());
#if defined(FORGE_OS_LINUX)
    { auto l = os::screen_locked(); CHECK_MSG(bool(l), "%s", l.err.what.c_str()); CHECK(l.value); }
    mock.session_flags = false;
    { auto l = os::screen_locked(); CHECK(l); CHECK(!l.value); }
    {   // keep-awake through logind: the inhibitor is held until release
        auto s = os::power::inhibit_sleep("mock run"); CHECK_MSG(bool(s), "%s", s.err.what.c_str());
        if (s) { CHECK(s->kind() == "logind-inhibit"); int rd = mock.inhibit_read_end.load(); struct pollfd p{rd, POLLIN, 0}; CHECK(::poll(&p, 1, 0) == 0); s->release(); CHECK(::poll(&p, 1, 2000) == 1); }
    }
#endif
    {   // reveal_path / open_url reach the bus only inside a desktop session
        bool had_w = os::has_env("WAYLAND_DISPLAY"); std::string old_w = os::getenv("WAYLAND_DISPLAY");
        os::setenv("WAYLAND_DISPLAY", "forge-test-0");
        auto before = mock.calls.size();
        auto r = os::reveal_path(os::temp_dir()); CHECK_MSG(bool(r), "%s", r.err.what.c_str());
        CHECK(mock.calls.size() > before); bool shown = false; for (std::size_t i = before; i < mock.calls.size(); ++i) shown |= mock.calls[i].member == "ShowItems"; CHECK(shown);
        if (os::opener_command("x").empty()) {   // no xdg-open / gio here: the portal carries open_url
            auto o = os::open_url("https://example.com/"); CHECK_MSG(bool(o), "%s", o.err.what.c_str()); CHECK(mock.calls.back().member == "OpenURI");
        }
        if (had_w) os::setenv("WAYLAND_DISPLAY", old_w); else os::unsetenv("WAYLAND_DISPLAY");
    }
#endif
}
#endif
