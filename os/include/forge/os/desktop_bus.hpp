#pragma once
// forge::os::desktop_bus: the freedesktop services forge::os uses, called over os::dbus (no notify-send, gdbus,
// loginctl, xdg-open or libsystemd needed). notification.hpp, shell_open.hpp, idle.hpp, background.hpp,
// appearance.hpp and power.hpp try these first and keep their helper programs as a fallback.
//   notify               org.freedesktop.Notifications.Notify                      (session bus)
//   open_uri             org.freedesktop.portal.OpenURI.OpenURI                    (session bus; Flatpak / Snap too)
//   show_items           org.freedesktop.FileManager1.ShowItems                    (session bus)
//   session_state        org.freedesktop.login1.Session IdleHint / LockedHint      (system bus)
//   screensaver_active   org.freedesktop.ScreenSaver / org.gnome.ScreenSaver       (session bus)
//   mutter_idle_ms       org.gnome.Mutter.IdleMonitor.GetIdletime                  (session bus, GNOME)
//   inhibit_sleep_fd     login1 / ConsoleKit Manager.Inhibit -> a descriptor       (system bus)
//   inhibit_idle         org.freedesktop.ScreenSaver.Inhibit -> a cookie           (session bus)
//   portal_setting       org.freedesktop.portal.Settings.Read                      (session bus)
//   power_profile        net.hadess.PowerProfiles / org.freedesktop.UPower.PowerProfiles (system bus)
// Every call returns `unsupported` when the bus or the service is not there (headless machines, CI, other OSes).
// set_addresses() points both buses somewhere else (tests).
#include <forge/os/dbus.hpp>
#include <mutex>

namespace forge::os::desktop_bus {
namespace detail_db {
struct overrides { std::mutex m; std::string session, system; bool set = false; };
inline overrides& ov() { static overrides o; return o; }
}
// Use these addresses instead of the environment's (empty = that bus does not exist). clear_addresses() undoes it.
inline void set_addresses(const std::string& session, const std::string& system) { auto& o = detail_db::ov(); std::lock_guard<std::mutex> g(o.m); o.session = session; o.system = system; o.set = true; }
inline void clear_addresses() { auto& o = detail_db::ov(); std::lock_guard<std::mutex> g(o.m); o.set = false; }
inline std::string address(dbus::bus b) {
    { auto& o = detail_db::ov(); std::lock_guard<std::mutex> g(o.m); if (o.set) return b == dbus::bus::session ? o.session : o.system; }
    return dbus::default_address(b);
}
inline bool has_bus(dbus::bus b) { return !address(b).empty(); }
inline result<dbus::connection> open(dbus::bus b, int timeout_ms = 3000) {
    std::string a = address(b);
    if (a.empty()) return {{}, unsupported(b == dbus::bus::session ? "D-Bus session bus (none running)" : "D-Bus system bus (none running)")};
    return dbus::connection::open_address(a, timeout_ms);
}

// urgency: 0 low, 1 normal, 2 critical. timeout_ms < 0: the server's default. Returns the notification id.
inline result<std::uint32_t> notify(const std::string& app, const std::string& title, const std::string& body, int urgency = 1, int timeout_ms = -1) {
    auto c = open(dbus::bus::session); if (!c) return {0, c.err};
    using dbus::value;
    auto r = c->call("org.freedesktop.Notifications", "/org/freedesktop/Notifications", "org.freedesktop.Notifications", "Notify",
                     {value::str(app), value::u32(0), value::str(""), value::str(title), value::str(body), value::array("s"),
                      value::dict({{"urgency", value::byte(std::uint8_t(urgency < 0 ? 0 : urgency > 2 ? 2 : urgency))}}), value::i32(timeout_ms)}, 5000);
    if (!r) return {0, r.err};
    return {r->body.empty() ? 0u : std::uint32_t(r->body[0].u), {}};
}
// Asks the desktop portal to open a URI with the user's default application.
inline status open_uri(const std::string& uri) {
    auto c = open(dbus::bus::session); if (!c) return {{}, c.err};
    using dbus::value;
    auto r = c->call("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.portal.OpenURI", "OpenURI",
                     {value::str(""), value::str(uri), value::dict()}, 10000);
    if (!r) return {{}, r.err};
    return {};
}
// Selects the given file:// URIs in the file manager.
inline status show_items(const std::vector<std::string>& file_uris) {
    auto c = open(dbus::bus::session); if (!c) return {{}, c.err};
    using dbus::value;
    std::vector<value> u; for (auto& s : file_uris) u.push_back(value::str(s));
    auto r = c->call("org.freedesktop.FileManager1", "/org/freedesktop/FileManager1", "org.freedesktop.FileManager1", "ShowItems", {value::array("s", std::move(u)), value::str("")}, 10000);
    if (!r) return {{}, r.err};
    return {};
}
struct session_state { bool idle = false, locked = false; std::uint64_t idle_since_monotonic_us = 0; };
// logind's view of this login session (XDG_SESSION_ID, else the caller's own session).
inline result<session_state> login_session() {
    auto c = open(dbus::bus::system); if (!c) return {{}, c.err};
    std::string path = "/org/freedesktop/login1/session/auto";   // logind 243+; older ones need the id
    if (auto id = getenv("XDG_SESSION_ID"); !id.empty() && dbus::valid_object_path("/s/" + id) ) {
        std::string esc; char b[8];
        for (unsigned char ch : id) { if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) esc.push_back(char(ch)); else { std::snprintf(b, sizeof b, "_%02x", ch); esc += b; } }
        if (!esc.empty() && esc[0] >= '0' && esc[0] <= '9') { std::snprintf(b, sizeof b, "_%02x", static_cast<unsigned char>(esc[0])); esc = b + esc.substr(1); }   // sd-bus path escaping
        path = "/org/freedesktop/login1/session/" + esc;
    }
    const char* iface = "org.freedesktop.login1.Session";
    auto idle = c->get_property("org.freedesktop.login1", path, iface, "IdleHint"); if (!idle) return {{}, is_unsupported(idle.err) ? idle.err : unsupported("login1 session (" + idle.err.what + ")")};
    session_state s; s.idle = idle->u != 0;
    if (auto l = c->get_property("org.freedesktop.login1", path, iface, "LockedHint")) s.locked = l->u != 0;
    if (auto t = c->get_property("org.freedesktop.login1", path, iface, "IdleSinceHintMonotonic")) s.idle_since_monotonic_us = t->u;
    return {s, {}};
}
// True while the screen saver / lock screen is showing, as the desktop's ScreenSaver service reports it.
inline result<bool> screensaver_active() {
    auto c = open(dbus::bus::session); if (!c) return {false, c.err};
    error last = unsupported("ScreenSaver service");
    for (auto [name, path] : {std::pair<const char*, const char*>{"org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver"}, {"org.gnome.ScreenSaver", "/org/gnome/ScreenSaver"}}) {
        auto r = c->call(name, path, name, "GetActive", {}, 3000);
        if (r && !r->body.empty() && r->body[0].type == 'b') return {r->body[0].u != 0, {}};
        if (!r) last = r.err;
    }
    return {false, is_unsupported(last) ? last : unsupported("ScreenSaver service (" + last.what + ")")};
}
// Milliseconds since the last input event, from GNOME's compositor.
inline result<std::uint64_t> mutter_idle_ms() {
    auto c = open(dbus::bus::session); if (!c) return {0, c.err};
    auto r = c->call("org.gnome.Mutter.IdleMonitor", "/org/gnome/Mutter/IdleMonitor/Core", "org.gnome.Mutter.IdleMonitor", "GetIdletime", {}, 3000);
    if (!r) return {0, is_unsupported(r.err) ? r.err : unsupported("Mutter idle monitor (" + r.err.what + ")")};
    if (r->body.empty()) return {0, unsupported("Mutter idle monitor (empty reply)")};
    return {r->body[0].u, {}};
}
// A block inhibitor from logind (or ConsoleKit2): the machine does not sleep / idle while the returned descriptor
// is open. what: "sleep", "idle", "shutdown" joined by ':'. unsupported: no such service on the system bus.
inline result<int> inhibit_sleep_fd(const std::string& what, const std::string& who, const std::string& why) {
#if defined(FORGE_OS_POSIX)
    auto c = open(dbus::bus::system); if (!c) return {-1, c.err};
    if (!c->passes_fds()) return {-1, unsupported("inhibitor (the bus does not pass descriptors)")};
    using dbus::value;
    error last = unsupported("login1 / ConsoleKit");
    for (auto [name, path, iface] : {std::tuple<const char*, const char*, const char*>{"org.freedesktop.login1", "/org/freedesktop/login1", "org.freedesktop.login1.Manager"},
                                     {"org.freedesktop.ConsoleKit", "/org/freedesktop/ConsoleKit/Manager", "org.freedesktop.ConsoleKit.Manager"}}) {
        auto r = c->call(name, path, iface, "Inhibit", {value::str(what), value::str(who), value::str(why), value::str("block")}, 5000);
        if (!r) { if (!is_unsupported(r.err) || is_unsupported(last)) last = r.err; continue; }
        int fd = r->body.empty() ? -1 : r->take_fd(r->body[0]);
        if (fd >= 0) return {fd, {}};
        last = errno_error(EPROTO, "Inhibit reply without a descriptor");
    }
    return {-1, last};
#else
    (void)what; (void)who; (void)why; return {-1, unsupported("inhibitor")};
#endif
}
// Keeps the session from going idle (screen saver, auto-suspend on most desktops) while the connection that
// asked stays open: the returned connection owns the inhibition; close it (or let it go) to release.
inline result<dbus::connection> inhibit_idle(const std::string& app, const std::string& reason) {
    auto c = open(dbus::bus::session); if (!c) return {{}, c.err};
    using dbus::value;
    auto r = c->call("org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver", "org.freedesktop.ScreenSaver", "Inhibit", {value::str(app), value::str(reason)}, 5000);
    if (!r) return {{}, r.err};
    return {std::move(c.value), {}};
}
// org.freedesktop.portal.Settings.Read(namespace, key), the variant unwrapped: e.g. ("org.freedesktop.appearance",
// "color-scheme") -> u 0 no preference / 1 dark / 2 light.
inline result<dbus::value> portal_setting(const std::string& ns, const std::string& key) {
    auto c = open(dbus::bus::session); if (!c) return {{}, c.err};
    using dbus::value;
    auto r = c->call("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.portal.Settings", "Read", {value::str(ns), value::str(key)}, 3000);
    if (!r) return {{}, is_unsupported(r.err) ? r.err : unsupported("portal setting " + ns + " " + key + " (" + r.err.what + ")")};
    if (r->body.empty()) return {{}, unsupported("portal setting (empty reply)")};
    return {r->body[0].unwrap(), {}};
}
// "power-saver", "balanced" or "performance" from power-profiles-daemon (or tuned's compatible service).
inline result<std::string> power_profile() {
    auto c = open(dbus::bus::system); if (!c) return {{}, c.err};
    error last = unsupported("power-profiles-daemon");
    for (auto [name, path] : {std::pair<const char*, const char*>{"org.freedesktop.UPower.PowerProfiles", "/org/freedesktop/UPower/PowerProfiles"}, {"net.hadess.PowerProfiles", "/net/hadess/PowerProfiles"}}) {
        auto v = c->get_property(name, path, name, "ActiveProfile", 3000);
        if (v && v->is_text()) return {v->s, {}};
        if (!v) last = v.err;
    }
    return {{}, is_unsupported(last) ? last : unsupported("power-profiles-daemon (" + last.what + ")")};
}

} // namespace forge::os::desktop_bus
