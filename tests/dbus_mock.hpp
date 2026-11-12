#pragma once
// A stand-in D-Bus bus for tests: one AF_UNIX listener that speaks SASL EXTERNAL and answers a handful of members
// (see serve()). One client at a time; `calls` records what arrived. POSIX only.
#include <forge/os/dbus.hpp>
#include <forge/os/fs.hpp>
#include <atomic>
#include <thread>
#if defined(FORGE_OS_POSIX)
namespace {
namespace db = forge::os::dbus;
namespace os = forge::os;
using db::value;
// Speaks just enough of the bus side: one client, SASL EXTERNAL, then message dispatch by member name.
struct mock_bus {
    std::string dir, path; int lfd = -1; std::thread th; std::atomic<bool> stop{false};
    std::string auth_seen; std::vector<db::message> calls; bool agree_fds = true, reject_auth = false, session_flags = true; std::string garbage_reply;
    std::atomic<int> inhibit_read_end{-1};
    bool start() {
        auto d = os::fs::make_temp_dir("forge-dbus-"); if (!d) return false;
        dir = *d; path = dir + "/bus";
        lfd = os::cloexec_socket(AF_UNIX, SOCK_STREAM); if (lfd < 0) return false;
        sockaddr_un sa{}; sa.sun_family = AF_UNIX; if (path.size() >= sizeof sa.sun_path) return false;
        std::memcpy(sa.sun_path, path.data(), path.size());
        if (::bind(lfd, reinterpret_cast<sockaddr*>(&sa), sizeof sa) != 0 || ::listen(lfd, 4) != 0) return false;
        th = std::thread([this] { serve(); }); return true;
    }
    ~mock_bus() { stop = true; if (inhibit_read_end >= 0) ::close(inhibit_read_end); if (lfd >= 0) { ::shutdown(lfd, SHUT_RDWR); ::close(lfd); } if (th.joinable()) th.join(); if (!dir.empty()) os::remove_all(dir); }
    static bool readable(int fd, int ms) { struct pollfd p{fd, POLLIN, 0}; return ::poll(&p, 1, ms) > 0; }
    static void put(int fd, const std::string& s) {
#if defined(MSG_NOSIGNAL)
        (void)!::send(fd, s.data(), s.size(), MSG_NOSIGNAL);
#else
        (void)!::send(fd, s.data(), s.size(), 0);
#endif
    }
    void reply(int fd, const db::message& to, std::vector<value> body, const char* error_name = nullptr) {
        static std::uint32_t serial = 1000;
        db::message r; r.type = error_name ? db::msg_type::error : db::msg_type::method_return; r.serial = ++serial; r.reply_serial = to.serial;
        r.sender = "org.freedesktop.DBus"; r.destination = ":1.42"; if (error_name) r.error_name = error_name; r.body = std::move(body);
        auto b = db::encode(r); if (b) put(fd, *b);
    }
    void serve() {
        while (!stop) {
            if (!readable(lfd, 100)) continue;
            int c = ::accept(lfd, nullptr, nullptr); if (c < 0) continue;
            std::string rx; char buf[4096]; bool begun = false;
            while (!stop) {
                if (!readable(c, 100)) continue;
                auto n = ::recv(c, buf, sizeof buf, 0); if (n <= 0) break;
                rx.append(buf, std::size_t(n));
                while (!begun) {
                    if (!rx.empty() && rx[0] == '\0') rx.erase(0, 1);
                    auto nl = rx.find("\r\n"); if (nl == std::string::npos) break;
                    std::string line = rx.substr(0, nl); rx.erase(0, nl + 2);
                    if (line.rfind("AUTH EXTERNAL ", 0) == 0) { auth_seen = line.substr(14); put(c, reject_auth ? "REJECTED EXTERNAL\r\n" : "OK 0123456789abcdef0123456789abcdef\r\n"); }
                    else if (line == "NEGOTIATE_UNIX_FD") put(c, agree_fds ? "AGREE_UNIX_FD\r\n" : "ERROR\r\n");
                    else if (line == "BEGIN") begun = true;
                    else put(c, "ERROR\r\n");
                }
                while (begun) {
                    auto sz = db::message_size(reinterpret_cast<const std::uint8_t*>(rx.data()), rx.size()); if (!sz || !sz.value || rx.size() < sz.value) break;
                    auto m = db::decode(reinterpret_cast<const std::uint8_t*>(rx.data()), sz.value); rx.erase(0, sz.value); if (!m) break;
                    calls.push_back(*m);
                    if (m->flags & 1) continue;   // no reply expected
                    if (m->member == "Hello") reply(c, *m, {value::str(":1.42")});
                    else if (m->member == "Echo") reply(c, *m, m->body);
                    else if (m->member == "Notify") reply(c, *m, {value::u32(7)});
                    else if (m->member == "AddMatch") reply(c, *m, {});
                    else if (m->member == "Fail") reply(c, *m, {value::str("it broke")}, "org.example.Error.Broken");
                    else if (m->member == "Missing") reply(c, *m, {value::str("The name is not activatable")}, "org.freedesktop.DBus.Error.ServiceUnknown");
                    else if (m->member == "Garbage") put(c, garbage_reply);
                    else if (m->member == "EmitThenReply") {
                        db::message s; s.type = db::msg_type::signal; s.serial = 5000; s.path = "/org/example"; s.iface = "org.example.Events"; s.member = "Changed"; s.body = {value::str("color-scheme"), value::variant(value::u32(1))};
                        auto b = db::encode(s); if (b) put(c, *b);
                        reply(c, *m, {value::boolean(true)});
                    }
                    else if (m->member == "Get") {   // org.freedesktop.DBus.Properties.Get(interface, name)
                        const std::string name = m->body.size() == 2 ? m->body[1].s : std::string();
                        if (name == "ActiveProfile") reply(c, *m, {value::variant(value::str("performance"))});
                        else if (name == "IdleHint" || name == "LockedHint") reply(c, *m, {value::variant(value::boolean(session_flags))});
                        else if (name == "IdleSinceHintMonotonic") reply(c, *m, {value::variant(value::u64(1))});
                        else reply(c, *m, {value::str("no such property")}, "org.freedesktop.DBus.Error.UnknownProperty");
                    }
                    else if (m->member == "OpenURI") reply(c, *m, {value::path("/org/freedesktop/portal/desktop/request/1_42/t")});
                    else if (m->member == "ShowItems") reply(c, *m, {});
                    else if (m->member == "GetActive") reply(c, *m, {value::boolean(true)});
                    else if (m->member == "GetIdletime") reply(c, *m, {value::u64(4200)});
                    else if (m->member == "Read") reply(c, *m, {value::variant(value::variant(value::u32(1)))});
                    else if (m->member == "Inhibit" && m->iface == "org.freedesktop.ScreenSaver") reply(c, *m, {value::u32(99)});
                    else if (m->member == "Inhibit") {   // login1: a descriptor rides along (SCM_RIGHTS + the UNIX_FDS field)
                        int pp[2]; if (os::cloexec_pipe(pp) != 0) { reply(c, *m, {value::str("pipe")}, "org.example.Error.Failed"); continue; }
                        db::message r; r.type = db::msg_type::method_return; r.serial = 7777; r.reply_serial = m->serial; r.unix_fds = 1;
                        value h; h.type = 'h'; h.u = 0; r.body = {h};
                        { int old = inhibit_read_end.exchange(pp[0]); if (old >= 0) ::close(old); }   // before the reply: the client may look at once
                        auto b = db::encode(r);
                        if (b) {
                            alignas(cmsghdr) char ctl[CMSG_SPACE(sizeof(int))] = {}; iovec iov{b->data(), b->size()}; msghdr mh{};
                            mh.msg_iov = &iov; mh.msg_iovlen = 1; mh.msg_control = ctl; mh.msg_controllen = sizeof ctl;
                            cmsghdr* cm = CMSG_FIRSTHDR(&mh); cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS; cm->cmsg_len = CMSG_LEN(sizeof(int));
                            std::memcpy(CMSG_DATA(cm), &pp[1], sizeof(int));
                            (void)!::sendmsg(c, &mh, 0);
                        }
                        ::close(pp[1]);   // readable (EOF) once the client closes its copy
                    }
                    else if (m->member != "Slow") reply(c, *m, {value::str("no such method")}, "org.freedesktop.DBus.Error.UnknownMethod");
                    // "Slow": no reply at all
                }
            }
            ::close(c);
        }
    }
};
}
#endif
