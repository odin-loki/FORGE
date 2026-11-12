// forge::os::dbus: codec (golden bytes, round trips, hostile input), then the client against an in-process mock
// bus (AF_UNIX, SASL EXTERNAL, Hello, calls, errors, signals, timeouts) and, when installed, a real dbus-daemon.
#include <forge/os/dbus.hpp>
#include <forge/os/fs.hpp>
#include <forge/os/process.hpp>
#include <atomic>
#include <thread>
#include "check.hpp"
#include "dbus_mock.hpp"
using namespace forge;
namespace db = forge::os::dbus;
using db::value;

namespace {
std::string hex(const std::string& s) { static const char* d = "0123456789abcdef"; std::string o; for (unsigned char c : s) { o.push_back(d[c >> 4]); o.push_back(d[c & 15]); } return o; }
os::result<db::message> dec(const std::string& b, const db::limits& l = {}) { return db::decode(reinterpret_cast<const std::uint8_t*>(b.data()), b.size(), l); }
}

TEST(dbus_signatures_and_paths) {
    for (const char* ok : {"", "s", "a{sv}", "(ii)", "a(yv)", "susssasa{sv}i", "aaay", "a{s(ia{uv})}", "v", "h"}) CHECK_MSG(db::valid_signature(ok), "%s", ok);
    for (const char* bad : {"a", "(", ")", "()", "{sv}", "a{vs}", "a{s}", "a{sss}", "z", "(i", "a{(i)s}", "e", "r"}) CHECK_MSG(!db::valid_signature(bad), "%s", bad);
    CHECK(!db::valid_signature(std::string(256, 'i'))); CHECK(db::valid_signature(std::string(255, 'i')));
    CHECK(!db::valid_signature(std::string(33, 'a') + "i")); CHECK(db::valid_signature(std::string(32, 'a') + "i"));
    CHECK(!db::valid_signature(std::string(33, '(') + "i" + std::string(33, ')'))); CHECK(db::valid_signature(std::string(32, '(') + "i" + std::string(32, ')')));
    CHECK(db::valid_single_type("a{sv}")); CHECK(!db::valid_single_type("ii")); CHECK(!db::valid_single_type(""));
    for (const char* ok : {"/", "/a", "/org/freedesktop/DBus", "/a_b/C9"}) CHECK(db::valid_object_path(ok));
    for (const char* bad : {"", "a", "//", "/a/", "/a//b", "/a-b", "/a.b"}) CHECK(!db::valid_object_path(bad));
    CHECK(value::dict({{"k", value::u32(1)}}).signature() == "a{sv}");
    CHECK(value::strukt({value::str("x"), value::array("i"), value::variant(value::byte(1))}).signature() == "(saiv)");
}

TEST(dbus_golden_bytes) {
    // A Hello call, byte for byte as the specification lays it out (little endian, fields in code order).
    db::message m; m.type = db::msg_type::method_call; m.serial = 1; m.destination = "org.freedesktop.DBus"; m.path = "/org/freedesktop/DBus";
    m.iface = "org.freedesktop.DBus"; m.member = "Hello";
    auto b = db::encode(m); CHECK(b); if (!b) return;
    const std::string want =
        std::string("l\1\0\1", 4) + std::string("\0\0\0\0", 4) + std::string("\1\0\0\0", 4) + std::string("\x6d\0\0\0", 4) +
        std::string("\1\1o\0", 4) + std::string("\x15\0\0\0", 4) + "/org/freedesktop/DBus" + std::string(3, '\0') +
        std::string("\2\1s\0", 4) + std::string("\x14\0\0\0", 4) + "org.freedesktop.DBus" + std::string(4, '\0') +
        std::string("\3\1s\0", 4) + std::string("\5\0\0\0", 4) + "Hello" + std::string(3, '\0') +
        std::string("\6\1s\0", 4) + std::string("\x14\0\0\0", 4) + "org.freedesktop.DBus" + std::string(4, '\0');
    CHECK_MSG(*b == want, "got  %s\n  want %s", hex(*b).c_str(), hex(want).c_str());
    auto sz = db::message_size(reinterpret_cast<const std::uint8_t*>(b->data()), b->size()); CHECK(sz); CHECK(sz.value == b->size());
    CHECK(db::message_size(reinterpret_cast<const std::uint8_t*>(b->data()), 15).value == 0);   // needs 16 bytes
    auto back = dec(*b); CHECK(back);
    if (back) { CHECK(back->type == db::msg_type::method_call); CHECK(back->member == "Hello"); CHECK(back->path == "/org/freedesktop/DBus"); CHECK(back->serial == 1); CHECK(back->body.empty()); }
    // Body alignment: byte, then a uint32 (3 pad bytes), a string, a uint64 (padded to 8), a double.
    db::message s; s.type = db::msg_type::signal; s.serial = 7; s.path = "/p"; s.iface = "a.b"; s.member = "M";
    s.body = {value::byte(0xab), value::u32(0x01020304), value::str("hi"), value::u64(0x1122334455667788ull), value::f64(1.5), value::boolean(true), value::i16(-2)};
    auto sb = db::encode(s); CHECK(sb); if (!sb) return;
    std::string body = sb->substr(sb->size() - 38);
    CHECK_MSG(hex(body) == "ab000000" "04030201" "02000000" "686900" "00" "8877665544332211" "000000000000f83f" "01000000" "feff", "%s", hex(body).c_str());
    auto sd = dec(*sb); CHECK(sd);
    if (sd) { CHECK(sd->signature == "yustdbn"); CHECK(sd->body.size() == 7); CHECK(sd->body[4].d == 1.5); CHECK(sd->body[6].as_int() == -2); CHECK(sd->body[3].u == 0x1122334455667788ull); }
}

TEST(dbus_round_trip_containers_and_big_endian) {
    db::message m; m.type = db::msg_type::method_call; m.serial = 99; m.path = "/x"; m.member = "Notify"; m.destination = "d.e"; m.iface = "i.f";
    m.body = {value::str("app"), value::u32(0), value::str(""), value::str("title \xC3\xA9"), value::str("body"), value::array("s", {value::str("a"), value::str("b")}),
              value::dict({{"urgency", value::byte(2)}, {"x", value::array("(is)", {value::strukt({value::i32(-5), value::str("q")})})}, {"empty", value::array("a{sv}")}}),
              value::i32(-1), value::byte_array(std::string("\0\1\2\xff", 4)), value::array("ay", {value::byte_array("zz"), value::byte_array("")}),
              value::variant(value::variant(value::i64(-7))), value::path("/a/b"), value::sig("a{sv}"), value::array("x")};
    auto b = db::encode(m); CHECK_MSG(bool(b), "%s", b.err.what.c_str()); if (!b) return;
    auto d = dec(*b); CHECK_MSG(bool(d), "%s", d.err.what.c_str()); if (!d) return;
    CHECK(d->signature == "susssasa{sv}iayaayvogax"); CHECK(d->body.size() == m.body.size());
    CHECK(d->body[3].s == "title \xC3\xA9"); CHECK(d->body[5].items.size() == 2); CHECK(d->body[5].items[1].s == "b");
    const value* urg = d->body[6].find("urgency"); CHECK(urg && urg->unwrap().type == 'y' && urg->unwrap().u == 2);
    const value* x = d->body[6].find("x"); CHECK(x && x->unwrap().items.size() == 1 && x->unwrap().items[0].items[0].as_int() == -5 && x->unwrap().items[0].items[1].s == "q");
    CHECK(d->body[6].find("nope") == nullptr); CHECK(d->body[6].find("empty")->unwrap().items.empty());
    CHECK(d->body[8].bytes == std::string("\0\1\2\xff", 4)); CHECK(d->body[9].items.size() == 2 && d->body[9].items[0].bytes == "zz");
    CHECK(d->body[10].unwrap().as_int() == -7); CHECK(d->body[11].s == "/a/b"); CHECK(d->body[12].s == "a{sv}"); CHECK(d->body[13].items.empty());
    auto again = db::encode(*d); CHECK(again && *again == *b);   // canonical: decode + encode is the identity
    // The same message as a big-endian peer would send it: swap every multi-byte number by hand-building a tiny one.
    const std::string be = std::string("B\2\0\1", 4) + std::string("\0\0\0\4", 4) + std::string("\0\0\0\x2a", 4) + std::string("\0\0\0\x0f", 4) +
        std::string("\5\1u\0", 4) + std::string("\0\0\0\7", 4) + std::string("\x08\1g\0", 4) + std::string("\1u\0", 3) + std::string(1, '\0') + std::string("\x12\x34\x56\x78", 4);
    auto bd = dec(be); CHECK_MSG(bool(bd), "%s", bd.err.what.c_str());
    if (bd) { CHECK(bd->type == db::msg_type::method_return); CHECK(bd->serial == 42); CHECK(bd->reply_serial == 7); CHECK(bd->body.size() == 1 && bd->body[0].u == 0x12345678u); }
    // values that cannot be sent are refused, not mangled
    db::message bad = m; bad.body = {value::path("not/a/path")}; CHECK(!db::encode(bad));
    bad.body = {value::array("i", {value::str("wrong element type")})}; CHECK(!db::encode(bad));
    bad.body = {value::str(std::string("nul\0inside", 10))}; CHECK(!db::encode(bad));
    bad.body = {value::strukt({})}; CHECK(!db::encode(bad));
    bad.body = {}; bad.serial = 0; CHECK(!db::encode(bad));
}

TEST(dbus_decode_rejects_hostile_input) {
    db::message m; m.type = db::msg_type::method_return; m.serial = 5; m.reply_serial = 4;
    m.body = {value::str("hello"), value::array("u", {value::u32(1), value::u32(2)}), value::variant(value::str("v")), value::boolean(true)};
    auto good = db::encode(m); CHECK(good); if (!good) return;
    CHECK(dec(*good));
    // every truncation fails cleanly; no prefix is accepted as a whole message
    for (std::size_t n = 0; n < good->size(); ++n) CHECK(!dec(good->substr(0, n)));
    CHECK(!dec(*good + "x"));
    // every single-byte corruption either decodes to something or fails: never crashes (ASan / the fuzz target watch the rest)
    std::size_t accepted = 0;
    for (std::size_t i = 0; i < good->size(); ++i) for (int delta : {1, 0x80, 0xff}) { std::string c = *good; c[i] = char(c[i] ^ delta); if (dec(c)) ++accepted; }
    std::printf("  %zu of %zu corruptions still decode\n", accepted, good->size() * 3);
    auto patch32 = [](std::string s, std::size_t at, std::uint32_t v) { for (int i = 0; i < 4; ++i) s[at + std::size_t(i)] = char(v >> (8 * i)); return s; };
    CHECK(!dec(patch32(*good, 4, 0xffffffffu)));   // body length past the buffer
    CHECK(!dec(patch32(*good, 12, 0x7fffffffu)));  // header field array past the buffer
    CHECK(!dec(patch32(*good, 8, 0)));             // serial 0
    { std::string c = *good; c[0] = 'x'; CHECK(!dec(c)); c = *good; c[3] = 2; CHECK(!dec(c)); c = *good; c[1] = 0; CHECK(!dec(c)); }
    const std::size_t body_at = good->size() - 40;   // "hello" length word
    CHECK(dec(*good)->body[0].s == "hello");
    CHECK(!dec(patch32(*good, body_at, 6)));          // string longer than stated: NUL is not where it must be
    CHECK(!dec(patch32(*good, body_at, 0x40000000)));  // string length far past the end
    CHECK(!dec(patch32(*good, body_at + 12, 0x04000001)));   // array longer than the 64 MiB cap
    CHECK(!dec(patch32(*good, body_at + 12, 7)));            // array that ends inside an element
    CHECK(!dec(patch32(*good, good->size() - 4, 2)));        // boolean that is neither 0 nor 1
    // limits: a message over max_message is refused from its header; node and depth caps hold
    db::limits tiny; tiny.max_message = 32; CHECK(!dec(*good, tiny));
    CHECK(!db::message_size(reinterpret_cast<const std::uint8_t*>(good->data()), good->size(), tiny));
    db::message wide = m; std::vector<value> many; for (int i = 0; i < 5000; ++i) many.push_back(value::u32(std::uint32_t(i)));
    wide.body = {value::array("u", many)}; auto wb = db::encode(wide); CHECK(wb);
    db::limits few; few.max_nodes = 1000; CHECK(!dec(*wb, few)); CHECK(dec(*wb));
    value deep = value::u32(1); for (int i = 0; i < 70; ++i) deep = value::variant(deep);
    db::message dm = m; dm.body = {deep}; CHECK(!db::encode(dm));   // deeper than the protocol allows: not even sent
    deep = value::u32(1); for (int i = 0; i < 40; ++i) deep = value::variant(deep);
    dm.body = {deep}; auto db40 = db::encode(dm); CHECK(db40); db::limits shallow; shallow.max_depth = 10; CHECK(!dec(*db40, shallow)); CHECK(dec(*db40));
    // header rules
    db::message call; call.type = db::msg_type::method_call; call.serial = 1; call.member = "NoPath"; auto cb = db::encode(call); CHECK(cb); CHECK(!dec(*cb));
    db::message sig; sig.type = db::msg_type::signal; sig.serial = 1; sig.path = "/p"; sig.member = "NoInterface"; auto sgb = db::encode(sig); CHECK(sgb); CHECK(!dec(*sgb));
    db::message err; err.type = db::msg_type::error; err.serial = 1; err.reply_serial = 1; auto eb = db::encode(err); CHECK(eb); CHECK(!dec(*eb));
}

TEST(dbus_addresses) {
    auto a = db::parse_address("unix:path=/run/user/1000/bus,guid=abc;unix:abstract=/tmp/dbus%2dX%20y;tcp:host=localhost,port=1;nonsense;unix:tmpdir=/tmp");
    CHECK(a.size() == 4);
    if (a.size() == 4) { CHECK(a[0].kind == "unix" && a[0].path == "/run/user/1000/bus" && !a[0].abstract); CHECK(a[1].abstract && a[1].path == "/tmp/dbus-X y"); CHECK(a[2].kind == "tcp" && a[2].path.empty()); CHECK(a[3].path.empty()); }
    CHECK(db::parse_address("").empty()); CHECK(db::parse_address(";;;").empty()); CHECK(db::parse_address("unix:path=%zz%4").at(0).path == "%zz%4");
#if !defined(FORGE_OS_POSIX) || defined(FORGE_OS_IOS) || defined(FORGE_OS_ANDROID)
    CHECK(db::default_address(db::bus::session).empty());
    auto c = db::connection::open(db::bus::session); CHECK(!c); CHECK(os::is_unsupported(c.err));
    auto s = db::connection::open(db::bus::system); CHECK(!s); CHECK(os::is_unsupported(s.err));
#else
    // no bus at that path: a clean unsupported, not a hang; a non-unix transport likewise
    auto none = db::connection::open_address("unix:path=/nonexistent/forge/bus"); CHECK(!none); CHECK(os::is_unsupported(none.err));
    auto tcp = db::connection::open_address("tcp:host=127.0.0.1,port=1"); CHECK(!tcp); CHECK(os::is_unsupported(tcp.err));
    auto sess = db::connection::open(db::bus::session);
    std::printf("  session bus: %s\n", sess ? sess->unique_name().c_str() : sess.err.what.c_str());
    if (!sess && db::default_address(db::bus::session).empty()) CHECK(os::is_unsupported(sess.err));
#endif
}

#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS)
TEST(dbus_client_against_mock_bus) {
    mock_bus bus; CHECK(bus.start());
    auto c = db::connection::open_address("unix:path=/nonexistent/first;unix:path=" + bus.path + ",guid=00"); CHECK_MSG(bool(c), "%s", c.err.what.c_str()); if (!c) return;
    CHECK(c->valid()); CHECK(c->unique_name() == ":1.42"); CHECK(c->passes_fds()); CHECK(c->native() >= 0);
    {   // EXTERNAL carries the uid as hex of its decimal digits
        std::string uid = std::to_string(::getuid()), want; for (char ch : uid) { char b[3]; std::snprintf(b, sizeof b, "%02x", ch); want += b; }
        CHECK_MSG(bus.auth_seen == want, "%s vs %s", bus.auth_seen.c_str(), want.c_str());
    }
    // a Notify call exactly as notification.hpp sends it
    auto r = c->call("org.freedesktop.Notifications", "/org/freedesktop/Notifications", "org.freedesktop.Notifications", "Notify",
                     {value::str("forge"), value::u32(0), value::str(""), value::str("Title"), value::str("Body"), value::array("s"), value::dict({{"urgency", value::byte(1)}}), value::i32(5000)}, 5000);
    CHECK_MSG(bool(r), "%s", r.err.what.c_str());
    if (r) { CHECK(r->body.size() == 1 && r->body[0].type == 'u' && r->body[0].u == 7); CHECK(r->sender == "org.freedesktop.DBus"); }
    CHECK(bus.calls.size() == 2);
    if (bus.calls.size() == 2) {
        auto& m = bus.calls[1]; CHECK(m.signature == "susssasa{sv}i"); CHECK(m.destination == "org.freedesktop.Notifications"); CHECK(m.member == "Notify");
        CHECK(m.body[3].s == "Title"); CHECK(m.body[7].as_int() == 5000); CHECK(m.serial == 2);
    }
    // big payloads cross several reads
    std::string blob(300000, '\0'); for (std::size_t i = 0; i < blob.size(); ++i) blob[i] = char(i * 131);
    auto e = c->call("x.y", "/x", "x.y", "Echo", {value::byte_array(blob), value::str("tail")}, 10000); CHECK(e);
    if (e) { CHECK(e->body.size() == 2 && e->body[0].bytes == blob && e->body[1].s == "tail"); }
    // errors: a D-Bus error keeps its name; a missing service is `unsupported`
    auto f = c->call("x.y", "/x", "x.y", "Fail"); CHECK(!f); CHECK(!os::is_unsupported(f.err)); CHECK(f.err.what.find("org.example.Error.Broken: it broke") != std::string::npos);
    CHECK(c->last_error_name() == "org.example.Error.Broken");
    auto miss = c->call("x.y", "/x", "x.y", "Missing"); CHECK(!miss); CHECK(os::is_unsupported(miss.err));
    // a signal that arrives before the reply is queued, not lost
    auto er = c->call("x.y", "/x", "x.y", "EmitThenReply"); CHECK(er); CHECK(er && er->body[0].u == 1);
    auto s = c->next_signal(1000); CHECK(s);
    if (s) { CHECK(s->member == "Changed" && s->iface == "org.example.Events"); CHECK(s->body.size() == 2 && s->body[1].unwrap().u == 1); }
    auto nos = c->next_signal(50); CHECK(!nos); CHECK(nos.err.code == ETIMEDOUT);
    CHECK(c->add_match("type='signal',interface='org.example.Events'"));
    auto p = c->get_property("net.hadess.PowerProfiles", "/net/hadess/PowerProfiles", "net.hadess.PowerProfiles", "ActiveProfile"); CHECK(p); CHECK(p && p->s == "performance");
    // no reply: the call times out, the connection stays usable
    auto t0 = os::monotonic_ns(); auto slow = c->call("x.y", "/x", "x.y", "Slow", {}, 150);
    CHECK(!slow); CHECK(slow.err.code == ETIMEDOUT); CHECK(os::monotonic_ns() - t0 < 5'000'000'000ull);
    CHECK(c->call("x.y", "/x", "x.y", "Echo", {value::u32(3)}));
    CHECK(c->send([] { db::message m; m.destination = "x.y"; m.path = "/x"; m.iface = "x.y"; m.member = "OneWay"; return m; }()));
    CHECK(c->call("x.y", "/x", "x.y", "Echo"));   // the one-way call produced no stray reply
    // a peer that answers with bytes that are not D-Bus: the call fails and the connection is closed
    bus.garbage_reply = std::string(64, '\x7f');
    auto g = c->call("x.y", "/x", "x.y", "Garbage", {}, 2000); CHECK(!g); CHECK(!c->valid());
    CHECK(!c->call("x.y", "/x", "x.y", "Echo"));
}

TEST(dbus_client_auth_variants) {
    { mock_bus bus; bus.agree_fds = false; CHECK(bus.start()); auto c = db::connection::open_address("unix:path=" + bus.path); CHECK(c); if (c) CHECK(!c->passes_fds()); }
    { mock_bus bus; bus.reject_auth = true; CHECK(bus.start()); auto c = db::connection::open_address("unix:path=" + bus.path, 2000); CHECK(!c); CHECK(c.err.what.find("rejected") != std::string::npos); }
}

// The same client against the reference implementation, when it is installed (skipped otherwise).
TEST(dbus_client_against_dbus_daemon) {
    if (os::find_executable("dbus-daemon").empty()) { std::printf("  skip: no dbus-daemon\n"); return; }
    auto d = os::fs::make_temp_dir("forge-dbusd-"); CHECK(d); if (!d) return;
    const std::string sock = *d + "/bus", conf = *d + "/session.conf";
    CHECK(os::write_file(conf, "<!DOCTYPE busconfig PUBLIC \"-//freedesktop//DTD D-Bus Bus Configuration 1.0//EN\" \"http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd\">\n"
        "<busconfig><type>session</type><listen>unix:path=" + sock + "</listen><auth>EXTERNAL</auth>"
        "<policy context=\"default\"><allow send_destination=\"*\" eavesdrop=\"true\"/><allow eavesdrop=\"true\"/><allow own=\"*\"/></policy></busconfig>\n"));
    auto daemon = os::child::start({"dbus-daemon", "--config-file=" + conf, "--nofork"}, true); CHECK(daemon); if (!daemon) { os::remove_all(*d); return; }
    for (int i = 0; i < 200 && !os::exists(sock); ++i) os::sleep_ns(10'000'000);
    if (!os::exists(sock)) { std::printf("  skip: dbus-daemon did not start: %s\n", daemon->read(4096, 200).c_str()); daemon->kill(); daemon->wait(5000); os::remove_all(*d); return; }
    {
        auto c = db::connection::open_address("unix:path=" + sock); CHECK_MSG(bool(c), "%s", c.err.what.c_str());
        if (c) {
            std::printf("  unique name %s, fd passing %d\n", c->unique_name().c_str(), int(c->passes_fds()));
            CHECK(c->unique_name().rfind(":1.", 0) == 0);
            auto id = c->call("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "GetId"); CHECK(id); CHECK(id && id->text().size() == 32);
            auto names = c->call("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "ListNames"); CHECK(names);
            if (names) { bool me = false, busname = false; for (auto& n : names->body[0].items) { me |= n.s == c->unique_name(); busname |= n.s == "org.freedesktop.DBus"; } CHECK(me); CHECK(busname); }
            // owning a name raises NameAcquired (delivered to us without a match rule) and NameOwnerChanged (with one)
            CHECK(c->add_match("type='signal',interface='org.freedesktop.DBus',member='NameOwnerChanged',arg0='org.forge.Test'"));
            auto own = c->call("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "RequestName", {value::str("org.forge.Test"), value::u32(4)}); CHECK(own); CHECK(own && own->body[0].u == 1);
            bool acquired = false, changed = false;
            for (int i = 0; i < 6 && !(acquired && changed); ++i) { auto s = c->next_signal(2000); if (!s) break; acquired |= s->member == "NameAcquired" && s->text() == "org.forge.Test"; changed |= s->member == "NameOwnerChanged" && s->text() == "org.forge.Test"; }
            CHECK(acquired); CHECK(changed);
            auto has = c->call("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "NameHasOwner", {value::str("org.forge.Test")}); CHECK(has && has->body[0].type == 'b' && has->body[0].u == 1);
            // the daemon's own errors: unknown method (an error), unknown service (unsupported)
            auto um = c->call("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "NoSuchMethod"); CHECK(!um); CHECK(c->last_error_name() == "org.freedesktop.DBus.Error.UnknownMethod");
            auto us = c->call("org.forge.Nobody", "/", "org.forge.Nobody", "Ping", {}, 5000); CHECK(!us); CHECK_MSG(os::is_unsupported(us.err), "%s", us.err.what.c_str());
            // the daemon validates what we send as strictly as anything: a{sv} with nested containers must be accepted by AddMatch-free calls
            auto sid = c->call("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "GetConnectionUnixUser", {value::str(c->unique_name())}); CHECK(sid); CHECK(sid && sid->body[0].u == std::uint64_t(::getuid()));
            auto props = c->call("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus.Properties", "GetAll", {value::str("org.freedesktop.DBus")});
            if (props) { CHECK(props->signature == "a{sv}"); const value* f = props->body[0].find("Features"); CHECK(f && f->unwrap().type == 'a'); }
        }
    }
    daemon->kill(); daemon->wait(5000); os::remove_all(*d);
}
#endif
