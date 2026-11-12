#pragma once
// forge::os clock synchronisation: is the system clock disciplined, and how far is it from an NTP server.
//   time_sync_status()   Linux: adjtimex(2) (STA_UNSYNC, estimated error) - what timedatectl and chronyc read.
//                        FreeBSD / macOS: ntp_adjtime(2). Windows, iOS, Android: unsupported (the state lives in
//                        the time service; ask an NTP server with sntp_query instead).
//   sntp_query()         one SNTP (RFC 4330) exchange over UDP, on every OS: clock offset, round-trip delay,
//                        stratum. The reply is checked (source address, mode, version, originate timestamp,
//                        kiss-of-death) before it is believed. Never called implicitly: it sends a packet.
#include <forge/os/os.hpp>
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_IOS) && !defined(FORGE_OS_ANDROID) && __has_include(<sys/timex.h>)
#include <sys/timex.h>
#define FORGE_OS_HAVE_TIMEX 1
#endif

namespace forge::os {
struct time_sync { bool synchronized = false; double estimated_error_s = 0, max_error_s = 0; std::string source; };
inline result<time_sync> time_sync_status() {
#if defined(FORGE_OS_HAVE_TIMEX)
    struct timex tx; std::memset(&tx, 0, sizeof tx);
#if defined(__linux__)
    int r = ::adjtimex(&tx); const char* src = "adjtimex";
#else
    int r = ::ntp_adjtime(&tx); const char* src = "ntp_adjtime";
#endif
    if (r < 0) return {{}, last_error(src)};
    time_sync t; t.source = src;
    t.synchronized = !(tx.status & STA_UNSYNC) && r != TIME_ERROR;
    t.estimated_error_s = double(tx.esterror) / 1e6; t.max_error_s = double(tx.maxerror) / 1e6;
    return {t, {}};
#else
    return {{}, unsupported("time_sync_status (use sntp_query)")};
#endif
}

struct sntp_result {
    double offset_s = 0;        // add this to the local clock to get the server's time
    double delay_s = 0;         // round trip, minus the server's processing time
    int stratum = 0;            // 1 = reference clock, 2.. = that many hops away
    std::int64_t server_unix_ns = 0;   // the server's transmit time
};
namespace detail_sntp {
inline std::uint64_t be(const unsigned char* p, int n) { std::uint64_t v = 0; for (int i = 0; i < n; ++i) v = v << 8 | p[i]; return v; }
// NTP timestamp (seconds since 1900 . 32-bit fraction) <-> Unix nanoseconds.
inline std::int64_t to_unix_ns(const unsigned char* p) {
    std::uint64_t sec = be(p, 4), frac = be(p + 4, 4);
    std::int64_t unix_s = std::int64_t(sec) - 2208988800ll; if (sec < 0x80000000ull) unix_s += 4294967296ll;   // era 1 (after 2036)
    return unix_s * 1000000000ll + std::int64_t((frac * 1000000000ull) >> 32);
}
inline void from_unix_ns(std::int64_t ns, unsigned char* p) {
    std::uint64_t sec = std::uint64_t(ns / 1000000000ll + 2208988800ll) & 0xffffffffull, frac = (std::uint64_t(ns % 1000000000ll) << 32) / 1000000000ull;
    for (int i = 0; i < 4; ++i) { p[i] = static_cast<unsigned char>(sec >> (24 - 8 * i)); p[4 + i] = static_cast<unsigned char>(frac >> (24 - 8 * i)); }
}
// Checks a reply against the request's transmit timestamp and computes offset and delay (t1 / t4 local, ns).
inline result<sntp_result> evaluate(const unsigned char* reply, std::size_t n, const unsigned char* sent_transmit, std::int64_t t1, std::int64_t t4) {
    if (n < 48) return {{}, errno_error(EPROTO, "sntp: short reply")};
    const int mode = reply[0] & 7, version = (reply[0] >> 3) & 7, leap = reply[0] >> 6, stratum = reply[1];
    if (mode != 4 && mode != 5) return {{}, errno_error(EPROTO, "sntp: reply is not from a server")};
    if (version < 1 || version > 4) return {{}, errno_error(EPROTO, "sntp: unknown protocol version")};
    if (std::memcmp(reply + 24, sent_transmit, 8) != 0) return {{}, errno_error(EPROTO, "sntp: reply does not answer this request")};
    if (stratum == 0) return {{}, errno_error(EPROTO, "sntp: kiss-of-death (" + std::string(reinterpret_cast<const char*>(reply + 12), 4) + ")")};
    if (stratum > 15 || leap == 3) return {{}, errno_error(EPROTO, "sntp: the server's clock is not synchronised")};
    if (be(reply + 40, 8) == 0) return {{}, errno_error(EPROTO, "sntp: reply without a transmit time")};
    const std::int64_t t2 = to_unix_ns(reply + 32), t3 = to_unix_ns(reply + 40);
    sntp_result r; r.stratum = stratum; r.server_unix_ns = t3;
    r.offset_s = (double(t2 - t1) + double(t3 - t4)) / 2e9;
    r.delay_s = (double(t4 - t1) - double(t3 - t2)) / 1e9; if (r.delay_s < 0) r.delay_s = 0;
    return {r, {}};
}
}
// Asks `server` (name or address; UDP `port`) for the time. Blocks up to timeout_ms.
inline result<sntp_result> sntp_query(const std::string& server, int port = 123, int timeout_ms = 2000) {
    net_init();
    addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_DGRAM; addrinfo* ai = nullptr;
    if (int e = ::getaddrinfo(server.c_str(), std::to_string(port).c_str(), &hints, &ai); e || !ai) return {{}, {e ? e : -1, "sntp: resolve " + server + ": " + (e ? ::gai_strerror(e) : "no address"), err_domain::gai}};
    error last = errno_error(ETIMEDOUT, "sntp: no reply from " + server);
    for (addrinfo* a = ai; a; a = a->ai_next) {
        socket s = socket::make(a->ai_family, SOCK_DGRAM); if (!s.valid()) { last = error{sock_errno(), "sntp: socket: " + error_text(sock_errno())}; continue; }
        if (::connect(s.native(), a->ai_addr, socklen_t(a->ai_addrlen)) != 0) { last = error{sock_errno(), "sntp: connect: " + error_text(sock_errno())}; continue; }   // only this peer's datagrams are delivered
        unsigned char rq[48] = {}; rq[0] = 0x23;   // no leap warning, version 4, mode 3 (client)
        std::uint64_t nonce = 0; entropy(&nonce, sizeof nonce);
        const std::int64_t t1 = wall_ns();
        detail_sntp::from_unix_ns(t1, rq + 40); rq[44] ^= static_cast<unsigned char>(nonce); rq[45] ^= static_cast<unsigned char>(nonce >> 8); rq[46] ^= static_cast<unsigned char>(nonce >> 16); rq[47] ^= static_cast<unsigned char>(nonce >> 24);   // unpredictable low bits: an off-path reply cannot match
        if (s.send(rq, sizeof rq) != long(sizeof rq)) { last = error{sock_errno(), "sntp: send: " + error_text(sock_errno())}; continue; }
        const std::uint64_t end = monotonic_ns() + std::uint64_t(timeout_ms < 0 ? 0 : timeout_ms) * 1000000ull;
        for (;;) {
            auto now = monotonic_ns(); if (now >= end) break;
            if (!s.wait_readable(int((end - now + 999999ull) / 1000000ull))) continue;
            unsigned char rp[512]; long n = s.recv(rp, sizeof rp); const std::int64_t t4 = wall_ns();
            if (n < 0) { last = error{sock_errno(), "sntp: " + server + ": " + error_text(sock_errno())}; break; }   // e.g. port unreachable
            auto r = detail_sntp::evaluate(rp, std::size_t(n), rq + 40, t1, t4);
            if (r) { ::freeaddrinfo(ai); return r; }
            last = r.err;   // a stray or forged datagram: keep waiting for the real one
        }
    }
    ::freeaddrinfo(ai);
    return {{}, last};
}
} // namespace forge::os
