#pragma once
// forge-serve's HTTP/1.1 request reader, kept apart from the socket so the parser can be driven from a buffer
// (fuzz/fuzz_serve_http.cpp). Limits: 1 MiB of header bytes, 64 MiB of body.
#include <cctype>
#include <charconv>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <system_error>

namespace forge::serve {

struct http_request { std::string method, path, body; std::map<std::string, std::string> headers; };

// `recv(char* buf, std::size_t cap)` returns the number of bytes read, or <= 0 at end of stream or on error
// (the contract of ::recv). Returns false for a request that is truncated, malformed or over a limit.
template <class Recv> bool read_request(Recv&& recv, http_request& r) {
    std::string buf; char tmp[8192];
    std::size_t hdr_end;
    while ((hdr_end = buf.find("\r\n\r\n")) == std::string::npos) {
        auto n = recv(tmp, sizeof tmp); if (n <= 0) return false; buf.append(tmp, std::size_t(n));
        if (buf.size() > (1u << 20)) return false;
    }
    auto line_end = buf.find("\r\n");
    auto first = buf.substr(0, line_end);
    auto sp1 = first.find(' '), sp2 = first.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    r.method = first.substr(0, sp1); r.path = first.substr(sp1 + 1, sp2 - sp1 - 1);
    std::size_t p = line_end + 2;
    while (p < hdr_end) {
        auto e = buf.find("\r\n", p); auto l = buf.substr(p, e - p); p = e + 2;
        auto c = l.find(':'); if (c == std::string::npos) continue;
        std::string k = l.substr(0, c); for (auto& ch : k) ch = char(std::tolower(static_cast<unsigned char>(ch)));
        auto v = l.substr(c + 1); while (!v.empty() && v[0] == ' ') v.erase(0, 1);
        r.headers[k] = v;
    }
    // Content-Length: decimal digits only (trailing blanks allowed). std::stoul threw on "abc", "" or a value
    // out of range, and the exception left the connection thread uncaught: one request ended the server.
    std::size_t len = 0;
    if (auto it = r.headers.find("content-length"); it != r.headers.end()) {
        std::string_view v = it->second;
        while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.remove_suffix(1);
        auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), len);
        if (v.empty() || ec != std::errc() || end != v.data() + v.size()) return false;
    }
    if (len > (64u << 20)) return false;
    r.body = buf.substr(hdr_end + 4);
    while (r.body.size() < len) { auto n = recv(tmp, sizeof tmp); if (n <= 0) return false; r.body.append(tmp, std::size_t(n)); }
    r.body.resize(len);
    return true;
}

} // namespace forge::serve
