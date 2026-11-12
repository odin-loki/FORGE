#include "fuzz.hpp"
#include "serve_http.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
// forge-serve's HTTP request reader (apps/serve_http.hpp), fed from a buffer. Byte 0 picks how many bytes each
// recv() hands over, so header/body boundaries fall at every offset; the rest is the client's byte stream.
FORGE_FUZZ({
    if (size < 1) return;
    std::size_t chunk = data[0] >= 200 ? 8192 : 1 + data[0] % 97, pos = 1;
    auto recv = [&](char* b, std::size_t cap) -> long {
        std::size_t n = std::min({cap, chunk, size - pos});
        std::memcpy(b, data + pos, n); pos += n;
        return long(n);   // 0 = peer closed
    };
    forge::serve::http_request r;
    if (forge::serve::read_request(recv, r)) {
        if (!r.headers.count("content-length") && !r.body.empty()) throw std::logic_error("serve_http: body without a length");
        if (r.method.find(' ') != std::string::npos) throw std::logic_error("serve_http: space in method");
    }
})
