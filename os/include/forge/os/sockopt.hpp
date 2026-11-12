#pragma once
// forge::os::net socket options and connection helpers, with one meaning on every OS:
//   set_keepalive       SO_KEEPALIVE + idle / interval / count (TCP_KEEPIDLE or Apple's TCP_KEEPALIVE, TCP_KEEPINTVL,
//                       TCP_KEEPCNT; Windows before 10 1709: SIO_KEEPALIVE_VALS, where the count is fixed at 10).
//   set_fastopen        TCP Fast Open on a listening socket (Linux, FreeBSD, macOS, Windows 10 1607+);
//                       set_fastopen_connect for clients (Linux TCP_FASTOPEN_CONNECT; elsewhere unsupported: macOS
//                       needs connectx, Windows ConnectEx).
//   set_reuseport       several sockets on one port (SO_REUSEPORT; SO_REUSEPORT_LB on FreeBSD so connections are
//                       balanced). Windows has no equivalent: unsupported.
//   set_v6only, listen_dual_stack   one IPv6 socket that also accepts IPv4 (v4-mapped addresses).
//   unix_listen / unix_connect      AF_UNIX stream sockets by path, also on Windows (10 1803+; unsupported before).
//   udp_bind, join_multicast, leave_multicast, set_multicast, send_to, recv_from   UDP and multicast groups
//                       (MCAST_JOIN_GROUP, protocol-independent, by interface index).
//   mdns_query          a one-shot multicast DNS question (224.0.0.251:5353, unicast responses requested) with a
//                       bounds-checked DNS message parser (dns_parse).
//   connect_happy_eyeballs          RFC 8305 style connect: IPv6 and IPv4 addresses interleaved, attempts started
//                       stagger_ms apart, the first to connect wins.
//   capabilities()      what this machine can do (IPv6, dual stack, AF_UNIX, reuseport, fast open, TLS backend).
#include <forge/os/net.hpp>
#if defined(FORGE_OS_POSIX)
#include <net/if.h>
#include <sys/un.h>
#endif

namespace forge::os::net {
namespace detail_net {
inline error sock_error(const std::string& what) { int e = sock_errno(); return {e, what + ": " + error_text(e)}; }
template <class T> inline bool setopt(sock_t s, int level, int name, const T& v) { return ::setsockopt(s, level, name, reinterpret_cast<const char*>(&v), sizeof v) == 0; }
#if defined(FORGE_OS_WINDOWS)
inline constexpr int tcp_keepidle_ = 3, tcp_keepcnt_ = 16, tcp_keepintvl_ = 17, tcp_fastopen_ = 15, af_unix_ = 1;
struct sockaddr_un_ { ADDRESS_FAMILY sun_family; char sun_path[108]; };
#endif
// Text form of a socket address and its port.
inline std::string ss_ip(const sockaddr_storage& a, int* port = nullptr) {
    char b[INET6_ADDRSTRLEN] = {};
    if (a.ss_family == AF_INET) { auto* s = reinterpret_cast<const sockaddr_in*>(&a); ::inet_ntop(AF_INET, &s->sin_addr, b, sizeof b); if (port) *port = ntohs(s->sin_port); }
    else if (a.ss_family == AF_INET6) { auto* s = reinterpret_cast<const sockaddr_in6*>(&a); ::inet_ntop(AF_INET6, &s->sin6_addr, b, sizeof b); if (port) *port = ntohs(s->sin6_port); }
    return b;
}
// ip literal + port -> sockaddr (no name resolution).
inline bool ip_sockaddr(const std::string& ip, int port, sockaddr_storage& out, socklen_t& len) {
    std::memset(&out, 0, sizeof out);
    auto* a4 = reinterpret_cast<sockaddr_in*>(&out); auto* a6 = reinterpret_cast<sockaddr_in6*>(&out);
    if (::inet_pton(AF_INET, ip.c_str(), &a4->sin_addr) == 1) { a4->sin_family = AF_INET; a4->sin_port = htons(static_cast<unsigned short>(port)); len = sizeof(sockaddr_in); return true; }
    std::string host = ip; unsigned scope = 0;
    if (auto pc = host.find('%'); pc != std::string::npos) { std::string z = host.substr(pc + 1); host.resize(pc);
#if defined(FORGE_OS_POSIX)
        scope = ::if_nametoindex(z.c_str());   // Windows zones are numeric (if_nametoindex lives in iphlpapi, not linked)
#endif
        if (!scope) scope = unsigned(std::strtoul(z.c_str(), nullptr, 10));
    }
    if (::inet_pton(AF_INET6, host.c_str(), &a6->sin6_addr) == 1) { a6->sin6_family = AF_INET6; a6->sin6_port = htons(static_cast<unsigned short>(port)); a6->sin6_scope_id = scope; len = sizeof(sockaddr_in6); return true; }
    return false;
}
} // namespace detail_net

// ---------------------------------------------------------------- TCP options
// idle_s: seconds of silence before the first probe; interval_s: between probes; count: unanswered probes before
// the connection is dropped.
inline status set_keepalive(socket& s, bool on = true, int idle_s = 60, int interval_s = 10, int count = 5) {
    if (!s.valid()) return {{}, errno_error(EBADF, "set_keepalive: closed socket")};
    if (!detail_net::setopt(s.native(), SOL_SOCKET, SO_KEEPALIVE, int(on))) return {{}, detail_net::sock_error("SO_KEEPALIVE")};
    if (!on) return {};
    if (idle_s < 1 || interval_s < 1 || count < 1) return {{}, errno_error(EINVAL, "set_keepalive: idle, interval and count must be positive")};
#if defined(FORGE_OS_WINDOWS)
    if (detail_net::setopt(s.native(), IPPROTO_TCP, detail_net::tcp_keepidle_, DWORD(idle_s)) && detail_net::setopt(s.native(), IPPROTO_TCP, detail_net::tcp_keepintvl_, DWORD(interval_s))) {
        detail_net::setopt(s.native(), IPPROTO_TCP, detail_net::tcp_keepcnt_, DWORD(count));   // 1703+; older keeps 10
        return {};
    }
    struct { u_long onoff, time_ms, interval_ms; } ka{1, u_long(idle_s) * 1000u, u_long(interval_s) * 1000u}; DWORD got = 0;
    if (::WSAIoctl(s.native(), 0x98000004u /*SIO_KEEPALIVE_VALS*/, &ka, sizeof ka, nullptr, 0, &got, nullptr, nullptr)) return {{}, detail_net::sock_error("SIO_KEEPALIVE_VALS")};
    return {};
#else
#if defined(__APPLE__)
    const int idle_opt = 0x10 /*TCP_KEEPALIVE*/, intvl_opt = 0x101 /*TCP_KEEPINTVL*/, cnt_opt = 0x102 /*TCP_KEEPCNT*/;
#else
    const int idle_opt = TCP_KEEPIDLE, intvl_opt = TCP_KEEPINTVL, cnt_opt = TCP_KEEPCNT;
#endif
    if (!detail_net::setopt(s.native(), IPPROTO_TCP, idle_opt, idle_s)) return {{}, detail_net::sock_error("TCP_KEEPIDLE")};
    if (!detail_net::setopt(s.native(), IPPROTO_TCP, intvl_opt, interval_s)) return {{}, detail_net::sock_error("TCP_KEEPINTVL")};
    if (!detail_net::setopt(s.native(), IPPROTO_TCP, cnt_opt, count)) return {{}, detail_net::sock_error("TCP_KEEPCNT")};
    return {};
#endif
}
// Accept data in the SYN on a listening socket. queue: pending fast-open requests (Linux); a flag elsewhere.
inline status set_fastopen(socket& listener, int queue = 16) {
    if (!listener.valid()) return {{}, errno_error(EBADF, "set_fastopen: closed socket")};
#if defined(FORGE_OS_WINDOWS)
    (void)queue; if (!detail_net::setopt(listener.native(), IPPROTO_TCP, detail_net::tcp_fastopen_, DWORD(1))) return {{}, detail_net::sock_error("TCP_FASTOPEN")};
    return {};
#elif defined(__linux__)
    if (!detail_net::setopt(listener.native(), IPPROTO_TCP, 23 /*TCP_FASTOPEN*/, queue > 0 ? queue : 1)) return {{}, detail_net::sock_error("TCP_FASTOPEN")};
    return {};
#elif defined(__APPLE__)
    (void)queue; if (!detail_net::setopt(listener.native(), IPPROTO_TCP, 0x105 /*TCP_FASTOPEN*/, 1)) return {{}, detail_net::sock_error("TCP_FASTOPEN")};
    return {};
#elif defined(FORGE_OS_FREEBSD) && defined(TCP_FASTOPEN)
    (void)queue; if (!detail_net::setopt(listener.native(), IPPROTO_TCP, TCP_FASTOPEN, 1)) return {{}, detail_net::sock_error("TCP_FASTOPEN")};
    return {};
#else
    (void)queue; return {{}, unsupported("TCP fast open")};
#endif
}
// Client side: the next connect() defers the SYN until the first write, which then travels in it.
inline status set_fastopen_connect(socket& s) {
#if defined(__linux__)
    if (!s.valid()) return {{}, errno_error(EBADF, "set_fastopen_connect: closed socket")};
    if (!detail_net::setopt(s.native(), IPPROTO_TCP, 30 /*TCP_FASTOPEN_CONNECT*/, 1)) return {{}, detail_net::sock_error("TCP_FASTOPEN_CONNECT")};
    return {};
#else
    (void)s; return {{}, unsupported("TCP fast open connect")};
#endif
}
inline status set_reuseport(socket& s, bool on = true) {
#if defined(FORGE_OS_WINDOWS)
    (void)s; (void)on; return {{}, unsupported("SO_REUSEPORT")};
#else
    if (!s.valid()) return {{}, errno_error(EBADF, "set_reuseport: closed socket")};
#if defined(FORGE_OS_FREEBSD) && defined(SO_REUSEPORT_LB)
    if (detail_net::setopt(s.native(), SOL_SOCKET, SO_REUSEPORT_LB, int(on))) return {};
#endif
    if (!detail_net::setopt(s.native(), SOL_SOCKET, SO_REUSEPORT, int(on))) return {{}, detail_net::sock_error("SO_REUSEPORT")};
    return {};
#endif
}
// IPV6_V6ONLY: false lets an IPv6 socket accept IPv4 peers as ::ffff:a.b.c.d. Set before bind.
inline status set_v6only(socket& s, bool on) {
    if (!s.valid()) return {{}, errno_error(EBADF, "set_v6only: closed socket")};
    if (!detail_net::setopt(s.native(), IPPROTO_IPV6, IPV6_V6ONLY, int(on))) return {{}, detail_net::sock_error("IPV6_V6ONLY")};
    return {};
}
// Kernel buffer sizes in bytes (0 = leave alone). The kernel may round or cap them.
inline status set_buffer_sizes(socket& s, int send_bytes, int recv_bytes) {
    if (!s.valid()) return {{}, errno_error(EBADF, "set_buffer_sizes: closed socket")};
    if (send_bytes > 0 && !detail_net::setopt(s.native(), SOL_SOCKET, SO_SNDBUF, send_bytes)) return {{}, detail_net::sock_error("SO_SNDBUF")};
    if (recv_bytes > 0 && !detail_net::setopt(s.native(), SOL_SOCKET, SO_RCVBUF, recv_bytes)) return {{}, detail_net::sock_error("SO_RCVBUF")};
    return {};
}
// seconds < 0: default close (background flush); 0: close resets the connection; > 0: close waits that long.
inline status set_linger(socket& s, int seconds) {
    if (!s.valid()) return {{}, errno_error(EBADF, "set_linger: closed socket")};
    linger l{}; l.l_onoff = seconds >= 0; l.l_linger = static_cast<decltype(l.l_linger)>(seconds > 0 ? seconds : 0);
    if (!detail_net::setopt(s.native(), SOL_SOCKET, SO_LINGER, l)) return {{}, detail_net::sock_error("SO_LINGER")};
    return {};
}
// TCP listener on [::]:port that accepts IPv6 and IPv4 (port 0 = ephemeral). Fails where IPv6 is disabled.
inline result<socket> listen_dual_stack(int port, int backlog = 128) {
    socket s = socket::make(AF_INET6, SOCK_STREAM);
    if (!s.valid()) return {{}, detail_net::sock_error("socket(AF_INET6)")};
    if (auto v = set_v6only(s, false); !v) return {{}, v.err};
    s.set_reuseaddr();
    sockaddr_in6 a{}; a.sin6_family = AF_INET6; a.sin6_port = htons(static_cast<unsigned short>(port));
    if (::bind(s.native(), reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) return {{}, detail_net::sock_error("bind [::]:" + std::to_string(port))};
    if (::listen(s.native(), backlog) != 0) return {{}, detail_net::sock_error("listen")};
    return {std::move(s), {}};
}

// ---------------------------------------------------------------- AF_UNIX
inline bool unix_sockets_available() {
#if defined(FORGE_OS_WINDOWS)
    static const bool ok = [] { socket s = socket::make(detail_net::af_unix_, SOCK_STREAM); return s.valid(); }();
    return ok;
#else
    return true;
#endif
}
namespace detail_net {
// Fills a sockaddr_un for path; false when the path does not fit (104-108 bytes depending on the OS).
template <class SA> inline bool unix_addr(const std::string& path, SA& a, socklen_t& len) {
    std::memset(&a, 0, sizeof a);
    if (path.empty() || path.size() >= sizeof a.sun_path) return false;
    std::memcpy(a.sun_path, path.data(), path.size());
    len = socklen_t(offsetof(SA, sun_path) + path.size() + 1);
    return true;
}
}
// Stream listener on a socket file. An existing socket file at path is replaced (a regular file is not).
inline result<socket> unix_listen(const std::string& path, int backlog = 128) {
    if (!unix_sockets_available()) return {{}, unsupported("AF_UNIX sockets (needs Windows 10 1803)")};
#if defined(FORGE_OS_WINDOWS)
    detail_net::sockaddr_un_ a; socklen_t len = 0; const int fam = detail_net::af_unix_;
    if (!detail_net::unix_addr(path, a, len)) return {{}, errno_error(ENAMETOOLONG, "unix_listen: path too long: " + path)};
    a.sun_family = ADDRESS_FAMILY(fam);
    DWORD attr = GetFileAttributesW(widen(path).c_str());
    if (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_REPARSE_POINT) && !(attr & FILE_ATTRIBUTE_DIRECTORY)) DeleteFileW(widen(path).c_str());   // a socket file is a reparse point
#else
    sockaddr_un a; socklen_t len = 0; const int fam = AF_UNIX;
    if (!detail_net::unix_addr(path, a, len)) return {{}, errno_error(ENAMETOOLONG, "unix_listen: path too long: " + path)};
    a.sun_family = AF_UNIX;
    struct ::stat st{}; if (::lstat(path.c_str(), &st) == 0 && S_ISSOCK(st.st_mode)) ::unlink(path.c_str());
#endif
    socket s = socket::make(fam, SOCK_STREAM);
    if (!s.valid()) return {{}, detail_net::sock_error("socket(AF_UNIX)")};
    if (::bind(s.native(), reinterpret_cast<sockaddr*>(&a), len) != 0) return {{}, detail_net::sock_error("bind " + path)};
    if (::listen(s.native(), backlog) != 0) return {{}, detail_net::sock_error("listen " + path)};
    return {std::move(s), {}};
}
inline result<socket> unix_connect(const std::string& path) {
    if (!unix_sockets_available()) return {{}, unsupported("AF_UNIX sockets (needs Windows 10 1803)")};
#if defined(FORGE_OS_WINDOWS)
    detail_net::sockaddr_un_ a; socklen_t len = 0; const int fam = detail_net::af_unix_;
    if (!detail_net::unix_addr(path, a, len)) return {{}, errno_error(ENAMETOOLONG, "unix_connect: path too long: " + path)};
    a.sun_family = ADDRESS_FAMILY(fam);
#else
    sockaddr_un a; socklen_t len = 0; const int fam = AF_UNIX;
    if (!detail_net::unix_addr(path, a, len)) return {{}, errno_error(ENAMETOOLONG, "unix_connect: path too long: " + path)};
    a.sun_family = AF_UNIX;
#endif
    socket s = socket::make(fam, SOCK_STREAM);
    if (!s.valid()) return {{}, detail_net::sock_error("socket(AF_UNIX)")};
    for (;;) {
        if (::connect(s.native(), reinterpret_cast<sockaddr*>(&a), len) == 0) break;
#if defined(FORGE_OS_POSIX)
        if (errno == EINTR) continue;
#endif
        return {{}, detail_net::sock_error("connect " + path)};
    }
    return {std::move(s), {}};
}

// ---------------------------------------------------------------- UDP and multicast
// UDP socket bound to ip:port ("" or "0.0.0.0" = any IPv4, "::" = any IPv6; port 0 = ephemeral). reuse: several
// sockets may bind the port (multicast listeners).
inline result<socket> udp_bind(const std::string& ip = "", int port = 0, bool reuse = false) {
    sockaddr_storage a{}; socklen_t len = 0;
    if (!detail_net::ip_sockaddr(ip.empty() ? "0.0.0.0" : ip, port, a, len)) return {{}, errno_error(EINVAL, "udp_bind: not an IP address: " + ip)};
    socket s = socket::make(a.ss_family, SOCK_DGRAM);
    if (!s.valid()) return {{}, detail_net::sock_error("socket(SOCK_DGRAM)")};
    if (reuse) {
        detail_net::setopt(s.native(), SOL_SOCKET, SO_REUSEADDR, 1);
#if defined(SO_REUSEPORT) && !defined(FORGE_OS_WINDOWS)
        detail_net::setopt(s.native(), SOL_SOCKET, SO_REUSEPORT, 1);
#endif
    }
    if (::bind(s.native(), reinterpret_cast<sockaddr*>(&a), len) != 0) return {{}, detail_net::sock_error("bind " + ip + ":" + std::to_string(port))};
    return {std::move(s), {}};
}
namespace detail_net {
inline status mcast_(socket& s, const std::string& group, unsigned ifindex, bool join) {
    sockaddr_storage g{}; socklen_t len = 0;
    if (!s.valid() || !ip_sockaddr(group, 0, g, len)) return {{}, errno_error(EINVAL, "multicast: not a group address: " + group)};
    const bool v4 = g.ss_family == AF_INET;
    if (v4 ? (ntohl(reinterpret_cast<sockaddr_in*>(&g)->sin_addr.s_addr) >> 28) != 0xe : reinterpret_cast<sockaddr_in6*>(&g)->sin6_addr.s6_addr[0] != 0xff)
        return {{}, errno_error(EINVAL, "multicast: not a multicast address: " + group)};
#if defined(MCAST_JOIN_GROUP)
    group_req r{}; r.gr_interface = ifindex; std::memcpy(&r.gr_group, &g, len);
    if (::setsockopt(s.native(), v4 ? IPPROTO_IP : IPPROTO_IPV6, join ? MCAST_JOIN_GROUP : MCAST_LEAVE_GROUP, reinterpret_cast<const char*>(&r), sizeof r) != 0)
        return {{}, sock_error(join ? "MCAST_JOIN_GROUP " + group : "MCAST_LEAVE_GROUP " + group)};
#else
    if (v4) {
        ip_mreq m{}; m.imr_multiaddr = reinterpret_cast<sockaddr_in*>(&g)->sin_addr; (void)ifindex;
        if (!setopt(s.native(), IPPROTO_IP, join ? IP_ADD_MEMBERSHIP : IP_DROP_MEMBERSHIP, m)) return {{}, sock_error("IP_ADD_MEMBERSHIP " + group)};
    } else {
        ipv6_mreq m{}; m.ipv6mr_multiaddr = reinterpret_cast<sockaddr_in6*>(&g)->sin6_addr; m.ipv6mr_interface = ifindex;
        if (!setopt(s.native(), IPPROTO_IPV6, join ? IPV6_JOIN_GROUP : IPV6_LEAVE_GROUP, m)) return {{}, sock_error("IPV6_JOIN_GROUP " + group)};
    }
#endif
    return {};
}
}
// ifindex 0 = the default multicast interface (link_info::index names a specific one).
inline status join_multicast(socket& s, const std::string& group, unsigned ifindex = 0) { return detail_net::mcast_(s, group, ifindex, true); }
inline status leave_multicast(socket& s, const std::string& group, unsigned ifindex = 0) { return detail_net::mcast_(s, group, ifindex, false); }
// Outgoing multicast: hop limit, whether our own packets loop back to local listeners, and the interface.
inline status set_multicast(socket& s, int family, int ttl, bool loopback, unsigned ifindex = 0) {
    if (!s.valid() || ttl < 0 || ttl > 255) return {{}, errno_error(EINVAL, "set_multicast: bad socket or ttl")};
    bool ok = true;
    if (family == 6) {
        ok &= detail_net::setopt(s.native(), IPPROTO_IPV6, IPV6_MULTICAST_HOPS, ttl);
        ok &= detail_net::setopt(s.native(), IPPROTO_IPV6, IPV6_MULTICAST_LOOP, static_cast<unsigned>(loopback));
        if (ifindex) ok &= detail_net::setopt(s.native(), IPPROTO_IPV6, IPV6_MULTICAST_IF, ifindex);
    } else {
#if defined(FORGE_OS_WINDOWS)
        ok &= detail_net::setopt(s.native(), IPPROTO_IP, IP_MULTICAST_TTL, DWORD(ttl)); ok &= detail_net::setopt(s.native(), IPPROTO_IP, IP_MULTICAST_LOOP, DWORD(loopback));
        if (ifindex) ok &= detail_net::setopt(s.native(), IPPROTO_IP, IP_MULTICAST_IF, htonl(ifindex));   // 0.0.0.x = interface index
#else
        ok &= detail_net::setopt(s.native(), IPPROTO_IP, IP_MULTICAST_TTL, static_cast<unsigned char>(ttl));
        ok &= detail_net::setopt(s.native(), IPPROTO_IP, IP_MULTICAST_LOOP, static_cast<unsigned char>(loopback));
        if (ifindex) {
#if defined(__linux__)
            ip_mreqn m{}; m.imr_ifindex = int(ifindex); ok &= detail_net::setopt(s.native(), IPPROTO_IP, IP_MULTICAST_IF, m);
#else
            in_addr ia{}; ia.s_addr = htonl(ifindex); ok &= detail_net::setopt(s.native(), IPPROTO_IP, IP_MULTICAST_IF, ia);   // BSD/Apple: 0.0.0.x = index
#endif
        }
#endif
    }
    if (!ok) return {{}, detail_net::sock_error("set_multicast")};
    return {};
}
// Datagram send / receive by IP literal. recv_from waits up to timeout_ms (< 0: forever); -1 with err set on
// failure or timeout (ETIMEDOUT).
inline result<std::size_t> send_to(socket& s, const void* p, std::size_t n, const std::string& ip, int port) {
    sockaddr_storage a{}; socklen_t len = 0;
    if (!detail_net::ip_sockaddr(ip, port, a, len)) return {0, errno_error(EINVAL, "send_to: not an IP address: " + ip)};
    auto r = ::sendto(s.native(), static_cast<const char*>(p), static_cast<int>(n), 0, reinterpret_cast<sockaddr*>(&a), len);
    if (r < 0) return {0, detail_net::sock_error("sendto " + ip)};
    return {std::size_t(r), {}};
}
struct datagram { std::string data, ip; int port = 0; };
inline result<datagram> recv_from(socket& s, int timeout_ms, std::size_t max = 65536) {
    if (!s.wait_readable(timeout_ms)) return {{}, errno_error(ETIMEDOUT, "recv_from: timed out")};
    datagram d; d.data.resize(max); sockaddr_storage a{}; socklen_t len = sizeof a;
    auto r = ::recvfrom(s.native(), d.data.data(), static_cast<int>(max), 0, reinterpret_cast<sockaddr*>(&a), &len);
    if (r < 0) return {{}, detail_net::sock_error("recvfrom")};
    d.data.resize(std::size_t(r)); d.ip = detail_net::ss_ip(a, &d.port);
    return {std::move(d), {}};
}

// ---------------------------------------------------------------- DNS messages / mDNS
namespace dns_type { inline constexpr int a = 1, cname = 5, ptr = 12, txt = 16, aaaa = 28, srv = 33, any = 255; }
// data: A/AAAA the address; PTR/CNAME the name; SRV "priority weight port target"; TXT the strings joined by '\n';
// other types the raw bytes.
struct dns_record { std::string name; int type = 0; std::uint32_t ttl = 0; std::string data; bool answer = true; };
// One question in wire format. unicast_reply sets the mDNS "QU" bit (responders answer our port directly).
inline result<std::string> dns_encode_query(const std::string& name, int qtype, std::uint16_t id = 0, bool unicast_reply = false) {
    std::string m; m.push_back(char(id >> 8)); m.push_back(char(id & 0xff));
    m.append("\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00", 10);   // flags 0, QDCOUNT 1
    std::size_t start = 0, total = 0;
    while (start < name.size()) {
        std::size_t dot = name.find('.', start); if (dot == std::string::npos) dot = name.size();
        std::size_t n = dot - start; if (n == 0) { if (dot == name.size()) break; return {{}, errno_error(EINVAL, "dns: empty label in " + name)}; }
        if (n > 63 || (total += n + 1) > 254) return {{}, errno_error(EINVAL, "dns: name too long: " + name)};
        m.push_back(char(n)); m.append(name, start, n); start = dot + 1;
    }
    m.push_back('\0');
    m.push_back(char((qtype >> 8) & 0xff)); m.push_back(char(qtype & 0xff));
    m.push_back(char(unicast_reply ? 0x80 : 0x00)); m.push_back('\x01');   // class IN
    return {std::move(m), {}};
}
namespace detail_net {
// A possibly compressed name at offset o; o is advanced past it. Pointers may only go backwards (no loops).
inline bool dns_name(std::string_view p, std::size_t& o, std::string& out) {
    std::size_t pos = o, limit = o; bool jumped = false; out.clear();
    for (int hops = 0; hops < 128; ++hops) {
        if (pos >= p.size()) return false;
        unsigned len = static_cast<unsigned char>(p[pos]);
        if (len == 0) { if (!jumped) o = pos + 1; return true; }
        if ((len & 0xc0) == 0xc0) {
            if (pos + 1 >= p.size()) return false;
            std::size_t tgt = ((len & 0x3f) << 8) | static_cast<unsigned char>(p[pos + 1]);
            if (tgt >= limit) return false;
            if (!jumped) o = pos + 2;
            jumped = true; limit = tgt; pos = tgt; continue;
        }
        if (len > 63 || pos + 1 + len > p.size() || out.size() + len + 1 > 255) return false;
        if (!out.empty()) out += '.';
        for (unsigned i = 0; i < len; ++i) { char c = p[pos + 1 + i]; out += (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) ? '?' : c; }
        pos += 1 + len;
    }
    return false;
}
inline unsigned be16(std::string_view p, std::size_t o) { return (unsigned(static_cast<unsigned char>(p[o])) << 8) | static_cast<unsigned char>(p[o + 1]); }
}
// Every resource record of a DNS message (answer, authority and additional sections; `answer` marks the first).
// Malformed input is an error, never an out-of-range read.
inline result<std::vector<dns_record>> dns_parse(std::string_view p) {
    auto bad = [] { return result<std::vector<dns_record>>{{}, errno_error(EPROTO, "dns: malformed message")}; };
    if (p.size() < 12) return bad();
    unsigned qd = detail_net::be16(p, 4), an = detail_net::be16(p, 6), ns = detail_net::be16(p, 8), ar = detail_net::be16(p, 10);
    std::size_t o = 12; std::string name;
    for (unsigned i = 0; i < qd; ++i) { if (!detail_net::dns_name(p, o, name) || o + 4 > p.size()) return bad(); o += 4; }
    std::vector<dns_record> out;
    for (unsigned i = 0; i < an + ns + ar; ++i) {
        dns_record r; if (!detail_net::dns_name(p, o, r.name) || o + 10 > p.size()) return bad();
        r.type = int(detail_net::be16(p, o)); r.ttl = (std::uint32_t(detail_net::be16(p, o + 4)) << 16) | detail_net::be16(p, o + 6);
        std::size_t rdlen = detail_net::be16(p, o + 8); o += 10; if (o + rdlen > p.size()) return bad();
        std::string_view rd = p.substr(o, rdlen); r.answer = i < an;
        if (r.type == dns_type::a && rdlen == 4) { char b[INET_ADDRSTRLEN] = {}; ::inet_ntop(AF_INET, rd.data(), b, sizeof b); r.data = b; }
        else if (r.type == dns_type::aaaa && rdlen == 16) { char b[INET6_ADDRSTRLEN] = {}; ::inet_ntop(AF_INET6, rd.data(), b, sizeof b); r.data = b; }
        else if (r.type == dns_type::ptr || r.type == dns_type::cname) { std::size_t q = o; if (!detail_net::dns_name(p, q, r.data) || q > o + rdlen) return bad(); }
        else if (r.type == dns_type::srv && rdlen >= 7) {
            std::size_t q = o + 6; std::string tgt; if (!detail_net::dns_name(p, q, tgt) || q > o + rdlen) return bad();
            r.data = std::to_string(detail_net::be16(p, o)) + " " + std::to_string(detail_net::be16(p, o + 2)) + " " + std::to_string(detail_net::be16(p, o + 4)) + " " + tgt;
        } else if (r.type == dns_type::txt) {
            for (std::size_t q = 0; q < rdlen;) { std::size_t n = static_cast<unsigned char>(rd[q]); if (q + 1 + n > rdlen) return bad(); if (!r.data.empty()) r.data += '\n'; r.data.append(rd.substr(q + 1, n)); q += 1 + n; }
        } else r.data.assign(rd);
        o += rdlen;
        if (out.size() < 4096) out.push_back(std::move(r));
    }
    return {std::move(out), {}};
}
// Ask the local link (multicast DNS) one question and collect the records that arrive within timeout_ms.
// e.g. mdns_query("_services._dns-sd._udp.local", dns_type::ptr) lists the service types being advertised.
// No answer is an empty list; an error means the question could not be sent (no multicast-capable network).
inline result<std::vector<dns_record>> mdns_query(const std::string& name, int qtype = dns_type::ptr, int timeout_ms = 1000) {
    auto q = dns_encode_query(name, qtype, 0, true); if (!q) return {{}, q.err};
    auto s = udp_bind("0.0.0.0", 0); if (!s) return {{}, s.err};
    set_multicast(*s, 4, 255, false);
    if (auto w = send_to(*s, q->data(), q->size(), "224.0.0.251", 5353); !w) return {{}, w.err};
    std::vector<dns_record> out;
    const std::uint64_t end = monotonic_ns() + std::uint64_t(timeout_ms < 0 ? 0 : timeout_ms) * 1000000ull;
    for (;;) {
        std::uint64_t now = monotonic_ns(); if (now >= end) break;
        auto d = recv_from(*s, int((end - now + 999999) / 1000000), 9000); if (!d) break;
        auto recs = dns_parse(d->data); if (!recs) continue;
        for (auto& r : recs.value) {
            bool dup = false; for (auto& x : out) dup |= x.name == r.name && x.type == r.type && x.data == r.data;
            if (!dup && out.size() < 4096) out.push_back(std::move(r));
        }
    }
    return {std::move(out), {}};
}

// ---------------------------------------------------------------- happy eyeballs
struct connect_options { int timeout_ms = 5000; int stagger_ms = 250; bool prefer_ipv6 = true; };
namespace detail_net {
// Waits until one of the connecting sockets is writable or failed; ready[i] = 1 connected-or-failed.
inline int wait_connecting(const std::vector<sock_t>& socks, int ms, std::vector<char>& ready) {
    ready.assign(socks.size(), 0); if (socks.empty()) { if (ms > 0) sleep_ns(std::uint64_t(ms) * 1000000ull); return 0; }
#if defined(FORGE_OS_WINDOWS)
    fd_set w, x; FD_ZERO(&w); FD_ZERO(&x);
    for (auto s : socks) { FD_SET(s, &w); FD_SET(s, &x); }   // a failed connect shows up in the except set
    timeval tv{ms / 1000, (ms % 1000) * 1000};
    int n = ::select(0, nullptr, &w, &x, &tv); if (n <= 0) return n;
    for (std::size_t i = 0; i < socks.size(); ++i) ready[i] = FD_ISSET(socks[i], &w) || FD_ISSET(socks[i], &x);
    return n;
#else
    std::vector<pollfd> p(socks.size()); for (std::size_t i = 0; i < socks.size(); ++i) p[i] = {socks[i], POLLOUT, 0};
    int n; do { n = ::poll(p.data(), nfds_t(p.size()), ms); } while (n < 0 && errno == EINTR);
    for (std::size_t i = 0; i < socks.size(); ++i) ready[i] = p[i].revents != 0;
    return n;
#endif
}
}
// Connect to host:port trying its IPv6 and IPv4 addresses in turn, a new attempt every stagger_ms while earlier
// ones are still in flight (so a dead address family costs stagger_ms, not a full timeout). The returned socket
// is blocking. connected_ip receives the address that won.
inline result<socket> connect_happy_eyeballs(const std::string& host, int port, connect_options o = {}, std::string* connected_ip = nullptr) {
    auto ai = socket::resolve(host.c_str(), port, false); if (!ai) return {{}, ai.err};
    std::vector<addrinfo*> v6, v4, order;
    for (auto* a = ai.value; a; a = a->ai_next) (a->ai_family == AF_INET6 ? v6 : a->ai_family == AF_INET ? v4 : order).push_back(a);
    order.clear();
    auto& first = o.prefer_ipv6 ? v6 : v4; auto& second = o.prefer_ipv6 ? v4 : v6;
    for (std::size_t i = 0; i < std::max(first.size(), second.size()); ++i) { if (i < first.size()) order.push_back(first[i]); if (i < second.size()) order.push_back(second[i]); }
    if (order.size() > 32) order.resize(32);
    struct attempt { socket s; addrinfo* a; };
    std::vector<attempt> live; std::size_t next = 0; int last = 0;
    const std::uint64_t t0 = monotonic_ns(), end = t0 + std::uint64_t(o.timeout_ms > 0 ? o.timeout_ms : 0) * 1000000ull;
    std::uint64_t next_start = t0;
    auto win = [&](attempt& w) -> result<socket> {
        w.s.set_nonblocking(false);
        if (connected_ip) { sockaddr_storage ss{}; std::memcpy(&ss, w.a->ai_addr, std::min<std::size_t>(sizeof ss, std::size_t(w.a->ai_addrlen))); *connected_ip = detail_net::ss_ip(ss); }
        socket s = std::move(w.s); live.clear(); ::freeaddrinfo(ai.value); return {std::move(s), {}};
    };
    for (;;) {
        std::uint64_t now = monotonic_ns();
        if (next < order.size() && (live.empty() || now >= next_start)) {
            addrinfo* a = order[next++]; next_start = now + std::uint64_t(o.stagger_ms > 0 ? o.stagger_ms : 0) * 1000000ull;
            socket s = socket::make(a->ai_family, a->ai_socktype);
            if (!s.valid()) { last = sock_errno(); continue; }
            s.set_nonblocking(true);
            if (::connect(s.native(), a->ai_addr, socklen_t(a->ai_addrlen)) == 0) { attempt w{std::move(s), a}; return win(w); }
            int e = sock_errno();
#if defined(FORGE_OS_POSIX)
            if (e != EINPROGRESS && !would_block(e)) { last = e; continue; }
#else
            if (!would_block(e)) { last = e; continue; }
#endif
            live.push_back({std::move(s), a}); continue;
        }
        if (live.empty() && next >= order.size()) break;
        if (o.timeout_ms > 0 && now >= end) { last = ETIMEDOUT; break; }
        std::uint64_t until = o.timeout_ms > 0 ? end : now + 1000000000ull;
        if (next < order.size()) until = std::min(until, next_start);
        int ms = until > now ? int((until - now + 999999) / 1000000) : 0;
        std::vector<sock_t> socks; for (auto& l : live) socks.push_back(l.s.native());
        std::vector<char> ready; detail_net::wait_connecting(socks, ms, ready);
        for (std::size_t i = 0; i < live.size();) {
            if (!ready[i]) { ++i; continue; }
            int so = live[i].s.so_error();
            if (so == 0) return win(live[i]);
            last = so; live.erase(live.begin() + std::ptrdiff_t(i)); ready.erase(ready.begin() + std::ptrdiff_t(i));
            next_start = monotonic_ns();   // a failure frees the slot: start the next address now
        }
    }
    live.clear(); ::freeaddrinfo(ai.value);
    if (last == ETIMEDOUT) return {{}, errno_error(ETIMEDOUT, "connect " + host + ":" + std::to_string(port) + ": timed out")};
    return {{}, {last ? last : -1, "connect " + host + ":" + std::to_string(port) + ": " + (last ? error_text(last) : std::string("no usable address"))}};
}

// ---------------------------------------------------------------- capabilities
struct net_capabilities {
    bool ipv6 = false, dual_stack = false, unix_sockets = false, reuseport = false, fastopen = false, multicast = true;
    std::string tls;   // schannel, network.framework, openssl or "" (no TLS backend loadable)
};
inline net_capabilities capabilities() {
    net_capabilities c; net_init();
    { socket s = socket::make(AF_INET6, SOCK_STREAM); c.ipv6 = s.valid(); if (s.valid()) c.dual_stack = bool(set_v6only(s, false)); }
    c.unix_sockets = unix_sockets_available();
    { socket s = socket::make(AF_INET, SOCK_STREAM); if (s.valid()) { c.reuseport = bool(set_reuseport(s)); c.fastopen = bool(set_fastopen(s)); } }
    auto t = tls_backend(); if (t.available) c.tls = t.name;
    return c;
}
} // namespace forge::os::net
