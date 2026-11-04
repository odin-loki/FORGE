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
    static HMODULE h = LoadLibraryExW(L"iphlpapi.dll", nullptr, 0x800 /*LOAD_LIBRARY_SEARCH_SYSTEM32*/); if (!h) return {{}, last_error("iphlpapi.dll")};
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
namespace detail_net {
inline bool alnum(unsigned char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
inline int hexv(unsigned char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }
inline std::string lower(std::string s) { for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c))); return s; }
// RFC 9110 token (methods, header names) and field value (no CR, LF or NUL: request splitting / header injection).
inline bool token(std::string_view s) {
    if (s.empty()) return false;
    for (unsigned char c : s) if (!alnum(c) && !(c && std::strchr("!#$%&'*+-.^_`|~", c))) return false;
    return true;
}
inline bool field_value(std::string_view s) { for (unsigned char c : s) if (c == '\r' || c == '\n' || c == 0) return false; return true; }
// Percent-encode what may not appear raw in a request target (controls, space, non-ASCII, "<>\"\\^`{|}[]#");
// valid %XX escapes are kept, a stray '%' becomes %25.
inline std::string encode_target(std::string_view s, bool query) {
    std::string o; o.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        bool ok = alnum(c) || std::strchr("-._~!$&'()*+,;=:@/", c) || (query && c == '?');
        if (c == '%' && i + 2 < s.size() && hexv(static_cast<unsigned char>(s[i + 1])) >= 0 && hexv(static_cast<unsigned char>(s[i + 2])) >= 0) ok = true;
        if (ok && c) { o += char(c); continue; }
        char b[4]; std::snprintf(b, sizeof b, "%%%02X", c); o += b;
    }
    return o;
}
// RFC 3986 5.2.4.
inline std::string remove_dot_segments(std::string in) {
    std::string out;
    while (!in.empty()) {
        if (in.rfind("../", 0) == 0) in.erase(0, 3);
        else if (in.rfind("./", 0) == 0) in.erase(0, 2);
        else if (in.rfind("/./", 0) == 0) in.replace(0, 3, "/");
        else if (in == "/.") in = "/";
        else if (in.rfind("/../", 0) == 0 || in == "/..") { in = in.size() == 3 ? "/" : in.substr(3); auto k = out.rfind('/'); out.resize(k == std::string::npos ? 0 : k); }
        else if (in == "." || in == "..") in.clear();
        else { auto k = in.find('/', in[0] == '/' ? 1 : 0); out += in.substr(0, k); in = k == std::string::npos ? std::string() : in.substr(k); }
    }
    return out;
}
inline bool valid_host(const std::string& h) {   // reg-name / IPv4 / IPv6 (with %zone): nothing that could split a request line
    if (h.empty()) return false;
    for (unsigned char c : h) if (!alnum(c) && !(c && std::strchr("-._~%:", c))) return false;
    return true;
}
}
// "host", "host:port" or "[v6]:port" (port omitted when it is the scheme default), as a Host header wants it.
inline std::string host_port(const url& u) {
    std::string h = u.host.find(':') != std::string::npos ? "[" + u.host + "]" : u.host;
    bool dflt = (u.scheme == "http" && u.port == 80) || (u.scheme == "https" && u.port == 443);
    return dflt || !u.port ? h : h + ":" + std::to_string(u.port);
}
inline std::string to_string(const url& u) { return u.scheme + "://" + host_port(u) + (u.path.empty() ? "/" : u.path) + (u.query.empty() ? "" : "?" + u.query); }
inline result<url> parse_url(const std::string& s) {
    url u; auto k = s.find("://"); if (k == std::string::npos) return {{}, {EINVAL, "bad url '" + s + "'"}};
    u.scheme = detail_net::lower(s.substr(0, k));
    auto rest = s.substr(k + 3); auto slash = rest.find_first_of("/?#");
    std::string hostport = rest.substr(0, slash); if (slash != std::string::npos) { auto p = rest.substr(slash); auto h = p.find('#'); if (h != std::string::npos) p.resize(h); auto q = p.find('?'); if (q != std::string::npos) { u.query = p.substr(q + 1); p.resize(q); } u.path = p.empty() ? "/" : p; }
    if (auto at = hostport.rfind('@'); at != std::string::npos) hostport = hostport.substr(at + 1);
    std::string port;
    if (!hostport.empty() && hostport[0] == '[') { auto e = hostport.find(']'); if (e == std::string::npos) return {{}, {EINVAL, "bad url '" + s + "'"}}; u.host = hostport.substr(1, e - 1); if (e + 1 < hostport.size()) { if (hostport[e + 1] != ':') return {{}, {EINVAL, "bad url '" + s + "'"}}; port = hostport.substr(e + 2); } }
    else { auto c = hostport.rfind(':'); if (c != std::string::npos) { u.host = hostport.substr(0, c); port = hostport.substr(c + 1); } else u.host = hostport; }
    for (char c : port) if (c < '0' || c > '9') return {{}, {EINVAL, "bad url '" + s + "' (port)"}};
    if (!port.empty()) { u.port = port.size() > 5 ? 0 : std::atoi(port.c_str()); if (u.port <= 0 || u.port > 65535) return {{}, {EINVAL, "bad url '" + s + "' (port)"}}; }
    if (!u.port) u.port = u.scheme == "https" ? 443 : u.scheme == "http" ? 80 : 0;
    if (u.host.empty()) return {{}, {EINVAL, "bad url '" + s + "' (no host)"}};
    if (!detail_net::valid_host(u.host)) return {{}, {EINVAL, "bad url '" + s + "' (host)"}};
    return {u, {}};
}
// Resolve a reference (a redirect Location) against base, RFC 3986 section 5.2: absolute URLs, "//host/p",
// "/p", "p", "../p", "?q".
inline result<url> resolve_reference(const url& base, std::string ref) {
    if (auto h = ref.find('#'); h != std::string::npos) ref.resize(h);
    while (!ref.empty() && (ref.back() == ' ' || ref.back() == '\t')) ref.pop_back();
    auto colon = ref.find(':'), stop = ref.find_first_of("/?");
    bool has_scheme = colon != std::string::npos && colon > 0 && (stop == std::string::npos || colon < stop) && std::isalpha(static_cast<unsigned char>(ref[0]));
    for (std::size_t i = 0; has_scheme && i < colon; ++i) has_scheme = detail_net::alnum(static_cast<unsigned char>(ref[i])) || std::strchr("+-.", ref[i]);
    if (has_scheme) return parse_url(ref);
    if (ref.rfind("//", 0) == 0) return parse_url(base.scheme + ":" + ref);
    url u = base; auto q = ref.find('?'); std::string path = ref.substr(0, q);
    if (path.empty()) { if (q != std::string::npos) u.query = ref.substr(q + 1); return {u, {}}; }
    u.query = q == std::string::npos ? std::string() : ref.substr(q + 1);
    if (path[0] == '/') u.path = detail_net::remove_dot_segments(path);
    else { auto d = base.path.rfind('/'); u.path = detail_net::remove_dot_segments((d == std::string::npos ? "/" : base.path.substr(0, d + 1)) + path); }
    if (u.path.empty() || u.path[0] != '/') u.path = "/" + u.path;
    return {u, {}};
}
struct http_response {
    int status = 0; std::string reason, body;
    // Lower-case names. A repeated header is joined with ", ", except Set-Cookie (cookies contain commas): "\n".
    std::map<std::string, std::string> headers;
    std::vector<std::pair<std::string, std::string>> header_list;   // every header line, in order (lower-case names)
    std::string url;                                                // the URL that answered (after redirects)
    std::vector<std::string> values(const std::string& lower_name) const { std::vector<std::string> v; for (auto& [k, x] : header_list) if (k == lower_name) v.push_back(x); return v; }
};
// timeout_ms bounds the connect and each read/write; deadline_ms the whole exchange including redirects (<= 0: none).
// Method, header names and values with CR/LF/NUL or other non-token bytes are rejected (EINVAL). Credentials
// (Authorization, Proxy-Authorization, Cookie) are not forwarded when a redirect leaves the origin.
struct http_request { std::string method = "GET", url, body; std::map<std::string, std::string> headers; int timeout_ms = 10000, deadline_ms = 300000; std::size_t max_body = 64u << 20; int max_redirects = 5; };
namespace detail_net {
// Decode a chunked body: 1 complete, 0 needs more bytes, -1 malformed, -2 larger than max.
inline int dechunk(const std::string& in, std::size_t i, std::string& out, std::size_t max) {
    out.clear();
    for (;;) {
        auto e = in.find("\r\n", i); if (e == std::string::npos) return in.size() - i > 1024 ? -1 : 0;
        std::string_view line(in.data() + i, e - i); if (auto semi = line.find(';'); semi != std::string_view::npos) line = line.substr(0, semi);
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) line.remove_suffix(1);
        if (line.empty() || line.size() > 15) return -1;
        std::uint64_t n = 0; for (char c : line) { int v = hexv(static_cast<unsigned char>(c)); if (v < 0) return -1; n = n * 16 + std::uint64_t(v); }
        i = e + 2;
        if (!n) { for (;;) { auto t = in.find("\r\n", i); if (t == std::string::npos) return 0; if (t == i) return 1; i = t + 2; } }   // trailers
        if (n > max || out.size() + n > max) return -2;
        if (in.size() < i + n + 2) return 0;
        if (in.compare(i + std::size_t(n), 2, "\r\n")) return -1;
        out.append(in, i, std::size_t(n)); i += std::size_t(n) + 2;
    }
}
inline bool timed_out(int e) {
#if defined(FORGE_OS_WINDOWS)
    return would_block(e) || e == WSAETIMEDOUT;
#else
    return would_block(e) || e == ETIMEDOUT;
#endif
}
}
inline result<http_response> http(const http_request& rq);
namespace detail_net {
inline result<http_response> http_once(const http_request& rq, std::uint64_t end_ns) {
    auto u = parse_url(rq.url); if (!u) return {{}, u.err};
    if (u->scheme != "http") return {{}, unsupported("http: scheme '" + u->scheme + "' (only http:// is built in)")};
    if (!token(rq.method)) return {{}, {EINVAL, "http: invalid method"}};
    bool ua = false, len = false, host = false;
    for (auto& [k, v] : rq.headers) {
        if (!token(k) || !field_value(v)) return {{}, {EINVAL, "http: invalid header '" + k + "'"}};
        auto l = lower(k); ua |= l == "user-agent"; len |= l == "content-length"; host |= l == "host";
    }
    auto left = [&]() -> int {   // ms for the next blocking step: min(per-step timeout, what remains of the deadline)
        int t = rq.timeout_ms > 0 ? rq.timeout_ms : 1 << 30; if (!end_ns) return t;
        auto now = monotonic_ns(); if (now >= end_ns) return 0;
        return int(std::min<std::uint64_t>((end_ns - now) / 1000000 + 1, std::uint64_t(t)));
    };
    const error deadline{ETIMEDOUT, "http: deadline exceeded"};
    int t = left(); if (!t) return {{}, deadline};
    auto s = socket::connect(u->host, u->port, t); if (!s) return {{}, s.err};
    std::string req = rq.method + " " + encode_target(u->path, false) + (u->query.empty() ? "" : "?" + encode_target(u->query, true)) + " HTTP/1.1\r\n";
    if (!host) req += "Host: " + host_port(*u) + "\r\n";
    req += "Connection: close\r\n";
    for (auto& [k, v] : rq.headers) if (lower(k) != "connection") req += k + ": " + v + "\r\n";
    if (!ua) req += "User-Agent: forge\r\n";
    if (!rq.body.empty() || rq.method == "POST" || rq.method == "PUT" || rq.method == "PATCH") { if (!len) req += "Content-Length: " + std::to_string(rq.body.size()) + "\r\n"; }
    req += "\r\n"; req += rq.body;
    if (!(t = left())) return {{}, deadline};
    s->set_timeout_ms(unsigned(t));
    if (!s->send_all(req.data(), req.size())) return {{}, last_error("http send")};
    http_response r; r.url = to_string(*u);
    std::string raw; char buf[16384]; std::size_t he = std::string::npos, body_at = 0;
    bool head = false, chunked = false, no_body = false; std::uint64_t clen = ~0ull;
    auto parse_head = [&]() -> error {   // status line + headers of raw[0, he)
        std::size_t pos = raw.find("\r\n"); std::string sl = raw.substr(0, pos);
        auto a = sl.find(' '); if (a == std::string::npos || sl.compare(0, 5, "HTTP/")) return {EPROTO, "http: bad status line"};
        r.status = std::atoi(sl.c_str() + a + 1); if (r.status < 100 || r.status > 999) return {EPROTO, "http: bad status line"};
        auto b = sl.find(' ', a + 1); r.reason = b == std::string::npos ? std::string() : sl.substr(b + 1);
        r.headers.clear(); r.header_list.clear();
        while (pos < he) {
            auto e = raw.find("\r\n", pos + 2); auto line = raw.substr(pos + 2, e - pos - 2); pos = e;
            auto c = line.find(':'); if (c == std::string::npos) continue;
            std::string k = lower(line.substr(0, c)), v = line.substr(c + 1);
            while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
            while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.pop_back();
            r.header_list.emplace_back(k, v);
            auto it = r.headers.find(k);
            if (it == r.headers.end()) r.headers.emplace(k, v); else it->second += (k == "set-cookie" ? "\n" : ", ") + v;
        }
        chunked = false; clen = ~0ull;
        if (auto te = r.headers.find("transfer-encoding"); te != r.headers.end()) {
            if (lower(te->second).find("chunked") == std::string::npos) return {EPROTO, "http: unsupported transfer-encoding '" + te->second + "'"};
            chunked = true;   // RFC 9112 6.3: Transfer-Encoding overrides Content-Length
        } else for (auto& v : r.values("content-length")) {
            if (v.empty() || v.size() > 19 || v.find_first_not_of("0123456789") != std::string::npos) return {EPROTO, "http: bad content-length '" + v + "'"};
            auto n = std::strtoull(v.c_str(), nullptr, 10); if (clen != ~0ull && clen != n) return {EPROTO, "http: conflicting content-length"};
            clen = n;
        }
        no_body = head || r.status == 204 || r.status == 304 || r.status < 200;
        if (!no_body && clen != ~0ull && clen > rq.max_body) return {EFBIG, "http: response too large"};
        return {};
    };
    head = rq.method == "HEAD";
    for (bool closed = false; !closed;) {
        if (!(t = left())) return {{}, deadline};
        s->set_timeout_ms(unsigned(t));
        long n = s->recv(buf, sizeof buf);
        if (n < 0) { int e = sock_errno(); if (timed_out(e)) return {{}, {ETIMEDOUT, "http: timed out reading the response"}}; return {{}, last_error("http recv")}; }
        if (n == 0) closed = true; else raw.append(buf, std::size_t(n));
        if (raw.size() - body_at > rq.max_body + 65536) return {{}, {EFBIG, "http: response too large"}};
        while (he == std::string::npos) {   // head not parsed yet (or an interim 1xx was skipped)
            auto e = raw.find("\r\n\r\n"); if (e == std::string::npos) break;
            he = e; if (auto err = parse_head()) return {{}, err};
            if (r.status >= 100 && r.status < 200 && r.status != 101) { raw.erase(0, e + 4); he = std::string::npos; continue; }
            body_at = e + 4;
        }
        if (he == std::string::npos) { if (closed) return {{}, {EPROTO, raw.empty() ? "http: connection closed without a response" : "http: malformed response"}}; continue; }
        if (no_body) break;
        if (chunked) {
            int st = dechunk(raw, body_at, r.body, rq.max_body);
            if (st == 1) break;
            if (st == -1) return {{}, {EPROTO, "http: malformed chunked body"}};
            if (st == -2) return {{}, {EFBIG, "http: response too large"}};
            if (closed) return {{}, {EPROTO, "http: truncated chunked body (no final chunk)"}};
        } else if (clen != ~0ull) {
            if (raw.size() - body_at >= clen) break;
            if (closed) return {{}, {EPROTO, "http: truncated body (" + std::to_string(raw.size() - body_at) + " of " + std::to_string(clen) + " bytes)"}};
        } else if (raw.size() - body_at > rq.max_body) return {{}, {EFBIG, "http: response too large"}};
    }
    if (!no_body && !chunked) r.body = raw.substr(body_at, clen == ~0ull ? std::string::npos : std::size_t(clen));
    if (no_body) r.body.clear();
    if (r.status >= 300 && r.status < 400 && r.status != 304 && rq.max_redirects > 0 && r.headers.count("location")) {
        auto to = resolve_reference(*u, r.headers["location"]); if (!to) return {{}, to.err};
        http_request n = rq; n.max_redirects--; n.url = to_string(*to);
        auto drop = [&](std::initializer_list<const char*> names) { for (auto it = n.headers.begin(); it != n.headers.end();) { bool d = false; auto l = lower(it->first); for (auto x : names) d |= l == x; it = d ? n.headers.erase(it) : std::next(it); } };
        if (to->scheme != u->scheme || lower(to->host) != lower(u->host) || to->port != u->port) drop({"authorization", "proxy-authorization", "cookie", "host"});
        if (r.status == 303 || ((r.status == 301 || r.status == 302) && rq.method == "POST")) {   // what browsers do
            if (n.method != "HEAD") n.method = "GET";
            n.body.clear(); drop({"content-length", "content-type", "transfer-encoding"});
        }
        return http_once(n, end_ns);
    }
    return {std::move(r), {}};
}
}
inline result<http_response> http(const http_request& rq) {
    net_init();
    return detail_net::http_once(rq, rq.deadline_ms > 0 ? monotonic_ns() + std::uint64_t(rq.deadline_ms) * 1000000ull : 0);
}
inline result<http_response> http_get(const std::string& u, int timeout_ms = 10000) { http_request r; r.url = u; r.timeout_ms = timeout_ms; return http(r); }
} // namespace forge::os::net
