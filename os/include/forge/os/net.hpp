#pragma once
// forge::os networking beyond os::socket: DNS resolution to address strings, network interfaces, TCP port
// probing, URL parsing and a small blocking HTTP/1.1 client (http:// only; no TLS: use forge's rpc client or a
// TLS library for https). Guide: docs/os-wrapper.md.
#include <forge/os/os.hpp>
#include <map>
#if defined(FORGE_OS_POSIX) && (!defined(FORGE_OS_ANDROID) || __ANDROID_API__ >= 24)   // bionic: getifaddrs from API 24
#define FORGE_NET_IFADDRS 1
#include <ifaddrs.h>
#include <net/if.h>
#elif defined(FORGE_OS_ANDROID)
#include <net/if.h>
#include <sys/ioctl.h>
#elif defined(FORGE_OS_WINDOWS)
#include <iphlpapi.h>   // types only: iphlpapi.dll is loaded at run time
#endif

namespace forge::os::net {
struct address { std::string ip; int family = 4; };
// All addresses for host (IPv4 and IPv6, de-duplicated, resolver order).
inline result<std::vector<address>> resolve(const std::string& host) {
    net_init();
    addrinfo hints{}, *r = nullptr; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    if (int e = ::getaddrinfo(host.c_str(), nullptr, &hints, &r); e) return {{}, {e, "resolve " + host + ": " + ::gai_strerror(e)}};
    std::vector<address> out;
    for (auto* p = r; p; p = p->ai_next) {
        char b[INET6_ADDRSTRLEN] = {};
        const void* src = p->ai_family == AF_INET ? static_cast<const void*>(&reinterpret_cast<sockaddr_in*>(p->ai_addr)->sin_addr) : static_cast<const void*>(&reinterpret_cast<sockaddr_in6*>(p->ai_addr)->sin6_addr);
        if (!::inet_ntop(p->ai_family, src, b, sizeof b)) continue;
        address a{b, p->ai_family == AF_INET ? 4 : 6};
        bool dup = false; for (auto& x : out) dup |= x.ip == a.ip; if (!dup) out.push_back(a);
    }
    ::freeaddrinfo(r);
    return {std::move(out), {}};
}
// Windows names an interface by its friendly name ("Ethernet", "Wi-Fi"); old Android (API < 24) lists IPv4 only.
struct interface_info { std::string name; std::vector<address> addresses; bool up = false, loopback = false; };
inline result<std::vector<interface_info>> interfaces() {
#if defined(FORGE_NET_IFADDRS)
    ifaddrs* ifs = nullptr; if (::getifaddrs(&ifs)) return {{}, last_error("getifaddrs")};
    std::vector<interface_info> out;
    for (auto* p = ifs; p; p = p->ifa_next) {
        interface_info* it = nullptr; for (auto& x : out) if (x.name == p->ifa_name) it = &x;
        if (!it) { out.push_back({p->ifa_name, {}, (p->ifa_flags & IFF_UP) != 0, (p->ifa_flags & IFF_LOOPBACK) != 0}); it = &out.back(); }
        if (!p->ifa_addr || (p->ifa_addr->sa_family != AF_INET && p->ifa_addr->sa_family != AF_INET6)) continue;
        char b[INET6_ADDRSTRLEN] = {}; int fam = p->ifa_addr->sa_family;
        const void* src = fam == AF_INET ? static_cast<const void*>(&reinterpret_cast<sockaddr_in*>(p->ifa_addr)->sin_addr) : static_cast<const void*>(&reinterpret_cast<sockaddr_in6*>(p->ifa_addr)->sin6_addr);
        if (::inet_ntop(fam, src, b, sizeof b)) it->addresses.push_back({b, fam == AF_INET ? 4 : 6});
    }
    ::freeifaddrs(ifs);
    return {std::move(out), {}};
#elif defined(FORGE_OS_ANDROID)   // SIOCGIFCONF: interfaces with an IPv4 address
    int fd = ::socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0); if (fd < 0) return {{}, last_error("socket")};
    std::vector<ifreq> rq(64); ifconf ic{};
    for (;;) {
        ic.ifc_len = int(rq.size() * sizeof(ifreq)); ic.ifc_req = rq.data();
        if (::ioctl(fd, SIOCGIFCONF, &ic) < 0) { auto e = last_error("SIOCGIFCONF"); ::close(fd); return {{}, e}; }
        if (std::size_t(ic.ifc_len) < rq.size() * sizeof(ifreq)) break;
        rq.resize(rq.size() * 2);
    }
    std::vector<interface_info> out;
    for (std::size_t i = 0; i < std::size_t(ic.ifc_len) / sizeof(ifreq); ++i) {
        ifreq& r = rq[i]; interface_info* it = nullptr; for (auto& x : out) if (x.name == r.ifr_name) it = &x;
        if (!it) {
            ifreq f{}; std::memcpy(f.ifr_name, r.ifr_name, IFNAMSIZ); short fl = ::ioctl(fd, SIOCGIFFLAGS, &f) == 0 ? f.ifr_flags : 0;
            out.push_back({r.ifr_name, {}, (fl & IFF_UP) != 0, (fl & IFF_LOOPBACK) != 0}); it = &out.back();
        }
        char b[INET_ADDRSTRLEN] = {};
        if (r.ifr_addr.sa_family == AF_INET && ::inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(&r.ifr_addr)->sin_addr, b, sizeof b)) it->addresses.push_back({b, 4});
    }
    ::close(fd);
    return {std::move(out), {}};
#elif defined(FORGE_OS_WINDOWS)
    net_init();
    static HMODULE h = LoadLibraryW(L"iphlpapi.dll"); if (!h) return {{}, last_error("iphlpapi.dll")};
    auto gaa = reinterpret_cast<decltype(&GetAdaptersAddresses)>(reinterpret_cast<void*>(GetProcAddress(h, "GetAdaptersAddresses")));
    if (!gaa) return {{}, unsupported("GetAdaptersAddresses")};
    ULONG n = 16384, rc = 0; std::vector<unsigned char> buf;
    for (int k = 0; k < 4; ++k) {   // the list can grow between the size query and the call
        buf.assign(n, 0);
        rc = gaa(AF_UNSPEC, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER, nullptr, reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data()), &n);
        if (rc != ERROR_BUFFER_OVERFLOW) break;
    }
    std::vector<interface_info> out;
    if (rc == ERROR_NO_DATA) return {std::move(out), {}};
    if (rc != ERROR_SUCCESS) return {{}, error{int(rc), "GetAdaptersAddresses failed"}};
    for (auto* a = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data()); a; a = a->Next) {
        interface_info it{a->FriendlyName ? narrow(a->FriendlyName) : std::string(a->AdapterName), {}, a->OperStatus == IfOperStatusUp, a->IfType == IF_TYPE_SOFTWARE_LOOPBACK};
        for (auto* u = a->FirstUnicastAddress; u; u = u->Next) {
            auto* sa = u->Address.lpSockaddr; if (!sa || (sa->sa_family != AF_INET && sa->sa_family != AF_INET6)) continue;
            char b[INET6_ADDRSTRLEN] = {}; int fam = sa->sa_family;
            const void* src = fam == AF_INET ? static_cast<const void*>(&reinterpret_cast<sockaddr_in*>(sa)->sin_addr) : static_cast<const void*>(&reinterpret_cast<sockaddr_in6*>(sa)->sin6_addr);
            if (::inet_ntop(fam, src, b, sizeof b)) it.addresses.push_back({b, fam == AF_INET ? 4 : 6});
        }
        out.push_back(std::move(it));
    }
    return {std::move(out), {}};
#else
    return {{}, unsupported("network interfaces")};
#endif
}
// True when a TCP connection to host:port succeeds within timeout_ms.
inline bool port_open(const std::string& host, int port, int timeout_ms = 1000) { auto s = socket::connect(host, port, timeout_ms); return bool(s); }

struct url { std::string scheme, host, path = "/", query; int port = 0; };
inline result<url> parse_url(const std::string& s) {
    url u; auto k = s.find("://"); if (k == std::string::npos) return {{}, {EINVAL, "bad url '" + s + "'"}};
    u.scheme = s.substr(0, k); for (auto& c : u.scheme) c = char(std::tolower(static_cast<unsigned char>(c)));
    auto rest = s.substr(k + 3); auto slash = rest.find_first_of("/?#");
    std::string hostport = rest.substr(0, slash); if (slash != std::string::npos) { auto p = rest.substr(slash); auto h = p.find('#'); if (h != std::string::npos) p.resize(h); auto q = p.find('?'); if (q != std::string::npos) { u.query = p.substr(q + 1); p.resize(q); } u.path = p.empty() ? "/" : p; }
    if (auto at = hostport.rfind('@'); at != std::string::npos) hostport = hostport.substr(at + 1);
    if (!hostport.empty() && hostport[0] == '[') { auto e = hostport.find(']'); if (e == std::string::npos) return {{}, {EINVAL, "bad url '" + s + "'"}}; u.host = hostport.substr(1, e - 1); if (e + 1 < hostport.size() && hostport[e + 1] == ':') u.port = std::atoi(hostport.c_str() + e + 2); }
    else { auto c = hostport.rfind(':'); if (c != std::string::npos) { u.host = hostport.substr(0, c); u.port = std::atoi(hostport.c_str() + c + 1); } else u.host = hostport; }
    if (!u.port) u.port = u.scheme == "https" ? 443 : u.scheme == "http" ? 80 : 0;
    if (u.host.empty()) return {{}, {EINVAL, "bad url '" + s + "' (no host)"}};
    return {u, {}};
}
struct http_response { int status = 0; std::string reason, body; std::map<std::string, std::string> headers; };   // header names lower-case
struct http_request { std::string method = "GET", url, body; std::map<std::string, std::string> headers; int timeout_ms = 10000; std::size_t max_body = 64u << 20; int max_redirects = 5; };
inline result<http_response> http(const http_request& rq) {
    auto u = parse_url(rq.url); if (!u) return {{}, u.err};
    if (u->scheme != "http") return {{}, unsupported("http: scheme '" + u->scheme + "' (only http:// is built in)")};
    auto s = socket::connect(u->host, u->port, rq.timeout_ms); if (!s) return {{}, s.err};
    s->set_timeout_ms(unsigned(rq.timeout_ms));
    std::string req = rq.method + " " + u->path + (u->query.empty() ? "" : "?" + u->query) + " HTTP/1.1\r\nHost: " + u->host + (u->port != 80 ? ":" + std::to_string(u->port) : "") + "\r\nConnection: close\r\n";
    bool ua = false, len = false; for (auto& [k, v] : rq.headers) { req += k + ": " + v + "\r\n"; std::string l = k; for (auto& c : l) c = char(std::tolower(static_cast<unsigned char>(c))); ua |= l == "user-agent"; len |= l == "content-length"; }
    if (!ua) req += "User-Agent: forge\r\n";
    if (!rq.body.empty() || rq.method == "POST" || rq.method == "PUT") { if (!len) req += "Content-Length: " + std::to_string(rq.body.size()) + "\r\n"; }
    req += "\r\n"; req += rq.body;
    if (!s->send_all(req.data(), req.size())) return {{}, last_error("http send")};
    std::string raw; char buf[16384];
    for (;;) { long n = s->recv(buf, sizeof buf); if (n < 0) return {{}, last_error("http recv")}; if (n == 0) break; raw.append(buf, std::size_t(n)); if (raw.size() > rq.max_body + 65536) return {{}, {EFBIG, "http: response too large"}}; }
    auto he = raw.find("\r\n\r\n"); if (he == std::string::npos) return {{}, {EPROTO, "http: malformed response"}};
    http_response r; std::size_t pos = raw.find("\r\n"); std::string sl = raw.substr(0, pos);
    { auto a = sl.find(' '); if (a == std::string::npos || sl.compare(0, 5, "HTTP/")) return {{}, {EPROTO, "http: bad status line"}}; r.status = std::atoi(sl.c_str() + a + 1); auto b = sl.find(' ', a + 1); if (b != std::string::npos) r.reason = sl.substr(b + 1); }
    while (pos < he) {
        auto e = raw.find("\r\n", pos + 2); auto line = raw.substr(pos + 2, e - pos - 2); pos = e;
        auto c = line.find(':'); if (c == std::string::npos) continue;
        std::string k = line.substr(0, c), v = line.substr(c + 1); for (auto& ch : k) ch = char(std::tolower(static_cast<unsigned char>(ch)));
        while (!v.empty() && v.front() == ' ') v.erase(v.begin());
        r.headers[k] = r.headers.count(k) ? r.headers[k] + ", " + v : v;
    }
    std::string body = raw.substr(he + 4);
    if (r.headers["transfer-encoding"].find("chunked") != std::string::npos) {
        std::string out; std::size_t i = 0;
        for (;;) { auto e = body.find("\r\n", i); if (e == std::string::npos) break; auto n = std::strtoull(body.c_str() + i, nullptr, 16); i = e + 2; if (!n) break; if (i + n > body.size()) return {{}, {EPROTO, "http: truncated chunk"}}; out.append(body, i, n); i += n + 2; }
        body = std::move(out);
    } else if (auto it = r.headers.find("content-length"); it != r.headers.end()) {
        auto n = std::strtoull(it->second.c_str(), nullptr, 10); if (body.size() > n) body.resize(n);
    }
    r.body = std::move(body);
    if (r.status >= 300 && r.status < 400 && rq.max_redirects > 0 && r.headers.count("location")) {
        http_request n = rq; n.max_redirects--; auto loc = r.headers["location"];
        n.url = loc.find("://") != std::string::npos ? loc : "http://" + u->host + ":" + std::to_string(u->port) + (loc.empty() || loc[0] != '/' ? "/" : "") + loc;
        if (r.status == 303) { n.method = "GET"; n.body.clear(); }
        return http(n);
    }
    return {std::move(r), {}};
}
inline result<http_response> http_get(const std::string& u, int timeout_ms = 10000) { http_request r; r.url = u; r.timeout_ms = timeout_ms; return http(r); }
} // namespace forge::os::net
