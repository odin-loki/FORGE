#include "fuzz.hpp"
#include "serve_http.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
// forge-serve's HTTP request reader (apps/serve_http.hpp), fed from a buffer. Byte 0 picks how many bytes each
// recv() hands over, so header/body/chunk boundaries fall at every offset; the rest is the client's byte stream,
// read as a kept-alive connection: request after request (Content-Length and chunked bodies, pipelined) until the
// reader reports an error or the stream ends. The limits are small so that inputs of a few hundred bytes reach them.
FORGE_FUZZ({
    if (size < 1) return;
    std::size_t chunk = data[0] >= 200 ? 8192 : 1 + data[0] % 97, pos = 1;
    auto recv = [&](char* b, std::size_t cap) -> long {
        std::size_t n = std::min({cap, chunk, size - pos});
        std::memcpy(b, data + pos, n); pos += n;
        return long(n);   // 0 = peer closed
    };
    const forge::serve::http_limits lim{512, 2048};
    forge::serve::request_reader rd(recv, lim);
    for (int served = 0; served < 64; ++served) {
        forge::serve::http_request r;
        int continues = 0;
        auto st = rd.read(r, [&] { ++continues; return true; });
        if (st != forge::serve::read_status::ok) {
            if (st == forge::serve::read_status::closed && served == 0 && size > 1 && pos < size) throw std::logic_error("serve_http: closed with input left");
            break;
        }
        if (r.body.size() > lim.max_body) throw std::logic_error("serve_http: body over the limit");
        if (!r.headers.count("content-length") && !r.chunked && !r.body.empty()) throw std::logic_error("serve_http: body without a length");
        if (r.headers.count("content-length") && r.chunked) throw std::logic_error("serve_http: both framings accepted");
        if (r.chunked != (r.headers.count("transfer-encoding") != 0)) throw std::logic_error("serve_http: chunked flag and header disagree");
        if (r.method.empty() || r.method.find(' ') != std::string::npos) throw std::logic_error("serve_http: bad method accepted");
        if (r.path.empty() || r.path.find_first_of(" \r\n") != std::string::npos) throw std::logic_error("serve_http: bad target accepted");
        if (r.version != "HTTP/1.1" && r.version != "HTTP/1.0") throw std::logic_error("serve_http: version");
        if (continues > 1 || (continues && !r.headers.count("expect"))) throw std::logic_error("serve_http: 100-continue");
        for (auto& [k, v] : r.headers) for (char c : k) if (c == ' ' || c == '\t' || (c >= 'A' && c <= 'Z')) throw std::logic_error("serve_http: header name");
    }
    // the one-shot form used before keep-alive existed
    pos = 1;
    forge::serve::http_request one;
    (void)forge::serve::read_request(recv, one, lim);
})
