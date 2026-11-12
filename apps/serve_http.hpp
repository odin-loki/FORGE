#pragma once
// forge-serve's HTTP/1.1 request reader, kept apart from the socket so the parser can be driven from a buffer
// (fuzz/fuzz_serve_http.cpp). One request_reader per connection: it keeps the bytes that follow a request, so
// keep-alive connections and pipelined requests are read in order.
//
// Bodies: Content-Length, or Transfer-Encoding: chunked (chunk extensions and trailers are read and dropped).
// "Expect: 100-continue" calls back before the body is read. Everything is bounded (http_limits): header bytes
// (request line, header lines, chunk-size lines, trailers), body bytes, and the bytes of blank lines before a request.
// A request that carries both Content-Length and Transfer-Encoding, two different Content-Lengths, a header name with
// blanks, a folded header line or a transfer coding other than chunked is refused (these are the request-smuggling
// shapes), never guessed at.
#include <cctype>
#include <charconv>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace forge::serve {

struct http_limits { std::size_t max_header = 1u << 20, max_body = 64u << 20; };

struct http_request {
    std::string method, path, body, version;        // version: "HTTP/1.1" or "HTTP/1.0"
    std::map<std::string, std::string> headers;     // names in lower case; a repeated header keeps "a, b"
    bool keep_alive = false;                        // the client wants the connection kept (1.1 default, 1.0 opt-in)
    bool chunked = false;                           // the body arrived with Transfer-Encoding: chunked
};

enum class read_status {
    ok,
    closed,              // end of stream before the first byte of a request (an idle keep-alive connection went away)
    truncated,           // the stream ended, failed or timed out inside a request
    bad,                 // 400: malformed request line, header, length or chunk framing
    header_too_large,    // 431
    body_too_large,      // 413
    not_implemented,     // 501: a transfer coding other than chunked
    bad_version,         // 505: not HTTP/1.x
    expectation_failed,  // 417: an Expect other than 100-continue
};
inline int status_code(read_status s) {
    switch (s) {
    case read_status::ok: return 200;
    case read_status::truncated: return 408;
    case read_status::header_too_large: return 431;
    case read_status::body_too_large: return 413;
    case read_status::not_implemented: return 501;
    case read_status::bad_version: return 505;
    case read_status::expectation_failed: return 417;
    default: return 400;
    }
}

namespace detail_http {
inline bool ieq(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) return false;
    return true;
}
inline std::string_view trim(std::string_view v) {
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.remove_prefix(1);
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.remove_suffix(1);
    return v;
}
// does the comma-separated list `v` hold `item` (case-insensitive)?
inline bool has_token(std::string_view v, std::string_view item) {
    while (!v.empty()) {
        auto c = v.find(','); auto t = trim(v.substr(0, c));
        if (ieq(t, item)) return true;
        if (c == std::string_view::npos) break;
        v.remove_prefix(c + 1);
    }
    return false;
}
}

// `recv(char* buf, std::size_t cap)` returns the number of bytes read, or <= 0 at end of stream, on error or on a
// timeout (the contract of ::recv).
template <class Recv> class request_reader {
public:
    explicit request_reader(Recv recv, http_limits lim = {}) : recv_(std::move(recv)), lim_(lim) {}
    // bytes of a following (pipelined) request are already here
    bool buffered() const { return !buf_.empty(); }

    // Reads one request. on_continue() is called once, before the body is read, when the client sent
    // "Expect: 100-continue" and the body has not arrived yet: send "HTTP/1.1 100 Continue\r\n\r\n" and return true
    // (false: stop, the read ends as truncated). After any status but ok the connection must be closed.
    template <class Cont> read_status read(http_request& r, Cont&& on_continue) {
        r = {};
        // blank lines before a request line are allowed (RFC 9112 2.2), within the header limit
        std::size_t hdr_end, skipped = 0;
        for (;;) {
            std::size_t lead = 0;
            while (lead + 1 < buf_.size() && buf_[lead] == '\r' && buf_[lead + 1] == '\n') lead += 2;
            if (lead) { buf_.erase(0, lead); scan_ = 0; skipped += lead; if (skipped > lim_.max_header) return read_status::header_too_large; }
            if ((hdr_end = buf_.find("\r\n\r\n", scan_ > 3 ? scan_ - 3 : 0)) != std::string::npos) break;
            scan_ = buf_.size();
            if (buf_.size() > lim_.max_header) return read_status::header_too_large;
            if (!fill()) return buf_.empty() && !skipped ? read_status::closed : read_status::truncated;
        }
        scan_ = 0;
        if (hdr_end > lim_.max_header) return read_status::header_too_large;
        std::string_view head(buf_.data(), hdr_end);
        auto line_end = head.find("\r\n");
        auto first = head.substr(0, line_end);
        auto sp1 = first.find(' '), sp2 = first.rfind(' ');
        if (sp1 == std::string_view::npos || sp2 == sp1 || sp1 == 0) return read_status::bad;
        r.method = std::string(first.substr(0, sp1)); r.path = std::string(first.substr(sp1 + 1, sp2 - sp1 - 1)); r.version = std::string(first.substr(sp2 + 1));
        for (char c : r.method) if (!std::isalpha(static_cast<unsigned char>(c))) return read_status::bad;
        if (r.path.empty() || r.path.find_first_of(" \t\r\n") != std::string::npos) return read_status::bad;
        if (r.version.rfind("HTTP/", 0) != 0) return read_status::bad;
        if (r.version != "HTTP/1.1" && r.version != "HTTP/1.0") return read_status::bad_version;
        bool two_lengths = false;
        for (std::size_t p = line_end == std::string_view::npos ? head.size() : line_end + 2; p < head.size();) {
            auto e = head.find("\r\n", p); if (e == std::string_view::npos) e = head.size();
            auto l = head.substr(p, e - p); p = e + 2;
            auto c = l.find(':');
            if (c == std::string_view::npos || c == 0 || l[0] == ' ' || l[0] == '\t') return read_status::bad;   // no name, or a folded line
            std::string k(l.substr(0, c));
            for (auto& ch : k) { if (ch == ' ' || ch == '\t') return read_status::bad; ch = char(std::tolower(static_cast<unsigned char>(ch))); }
            std::string v(detail_http::trim(l.substr(c + 1)));
            auto [it, fresh] = r.headers.emplace(k, v);
            if (!fresh) { if (k == "content-length" && it->second != v) two_lengths = true; else if (k != "content-length") it->second += ", " + v; }
        }
        if (two_lengths) return read_status::bad;
        const bool v11 = r.version == "HTTP/1.1";
        auto conn = r.headers.find("connection");
        r.keep_alive = v11 ? !(conn != r.headers.end() && detail_http::has_token(conn->second, "close"))
                           : conn != r.headers.end() && detail_http::has_token(conn->second, "keep-alive");
        auto te = r.headers.find("transfer-encoding"), cl = r.headers.find("content-length");
        if (te != r.headers.end()) {
            if (cl != r.headers.end()) return read_status::bad;   // both: a smuggling vector, refused
            if (!detail_http::ieq(detail_http::trim(te->second), "chunked")) return read_status::not_implemented;
            if (!v11) return read_status::bad;
            r.chunked = true;
        }
        // Content-Length: decimal digits only. std::stoul threw on "abc", "" or a value out of range, and the
        // exception left the connection thread uncaught: one request ended the server.
        std::size_t len = 0;
        if (cl != r.headers.end()) {
            std::string_view v = cl->second;
            auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), len);
            if (v.empty() || ec != std::errc() || end != v.data() + v.size()) return read_status::bad;
        }
        if (len > lim_.max_body) return read_status::body_too_large;
        bool expect = false;
        if (auto ex = r.headers.find("expect"); ex != r.headers.end()) {
            if (!detail_http::ieq(detail_http::trim(ex->second), "100-continue")) return read_status::expectation_failed;
            expect = v11;
        }
        buf_.erase(0, hdr_end + 4);
        if (expect && (r.chunked || buf_.size() < len) && !on_continue()) return read_status::truncated;
        if (r.chunked) return read_chunked(r);
        while (buf_.size() < len) if (!fill()) return read_status::truncated;
        r.body.assign(buf_, 0, len);
        buf_.erase(0, len);
        return read_status::ok;
    }
    read_status read(http_request& r) { return read(r, [] { return true; }); }
private:
    bool fill() {
        char tmp[8192];
        auto n = recv_(tmp, sizeof tmp);
        if (n <= 0) return false;
        buf_.append(tmp, std::size_t(n));
        return true;
    }
    // one CRLF-terminated line at buf_[at ...] of at most `cap` bytes; false + status on end of stream or overflow
    bool line(std::size_t at, std::size_t cap, std::size_t& end, read_status& st) {
        for (;;) {
            end = buf_.find("\r\n", at);
            if (end != std::string::npos) { if (end - at > cap) { st = read_status::header_too_large; return false; } return true; }
            if (buf_.size() - at > cap + 1) { st = read_status::header_too_large; return false; }
            if (!fill()) { st = read_status::truncated; return false; }
        }
    }
    read_status read_chunked(http_request& r) {
        std::size_t meta = 0;   // bytes of chunk-size lines and trailers, counted against the header limit
        read_status st = read_status::ok;
        for (;;) {
            std::size_t e;
            if (!line(0, 4096, e, st)) return st;
            std::string_view l(buf_.data(), e);
            if (auto semi = l.find(';'); semi != std::string_view::npos) l = l.substr(0, semi);   // chunk extensions
            l = detail_http::trim(l);
            std::size_t n = 0;
            auto [end, ec] = std::from_chars(l.data(), l.data() + l.size(), n, 16);
            if (l.empty() || l.size() > 16 || ec != std::errc() || end != l.data() + l.size()) return read_status::bad;
            if ((meta += e + 2) > lim_.max_header) return read_status::header_too_large;
            buf_.erase(0, e + 2);
            if (n == 0) break;
            if (n > lim_.max_body || r.body.size() > lim_.max_body - n) return read_status::body_too_large;
            while (buf_.size() < n + 2) if (!fill()) return read_status::truncated;
            if (buf_[n] != '\r' || buf_[n + 1] != '\n') return read_status::bad;
            r.body.append(buf_, 0, n);
            buf_.erase(0, n + 2);
        }
        for (;;) {   // trailers up to the empty line: read and dropped
            std::size_t e;
            if (!line(0, 8192, e, st)) return st;
            if ((meta += e + 2) > lim_.max_header) return read_status::header_too_large;
            buf_.erase(0, e + 2);
            if (e == 0) return read_status::ok;
        }
    }
    Recv recv_;
    http_limits lim_;
    std::string buf_;
    std::size_t scan_ = 0;
};
template <class Recv> request_reader(Recv) -> request_reader<Recv>;
template <class Recv> request_reader(Recv, http_limits) -> request_reader<Recv>;

// One request off a stream (no connection reuse). False for a request that is truncated, malformed or over a limit.
template <class Recv> bool read_request(Recv&& recv, http_request& r, http_limits lim = {}) {
    request_reader<std::decay_t<Recv>> rd(std::forward<Recv>(recv), lim);
    return rd.read(r) == read_status::ok;
}

} // namespace forge::serve
