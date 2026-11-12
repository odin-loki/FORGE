#include "fuzz.hpp"
#include <forge/loadbalancer/proxy.hpp>
#include <algorithm>
#include <cstring>
// loadbalancer L7 proxy (loadbalancer/proxy.hpp): the client side of serve_l7_ (request line, headers,
// Content-Length / chunked bodies, keep-alive loop) against a pool with no backends, then the upstream
// response parser (read_head_ / parse_headers / read_body_) on the same bytes. Sockets are replaced by a
// buffer; byte 0 picks the recv() chunk size.
namespace forge::loadbalancer {
struct mem_sock {
    const std::uint8_t* p; std::size_t n, chunk; std::size_t sent = 0;
    bool wait_readable(int) const { return n > 0; }
    long recv(void* b, std::size_t cap) { std::size_t k = std::min({cap, chunk, n}); std::memcpy(b, p, k); p += k; n -= k; return long(k); }
    bool send_all(const void*, std::size_t k) { sent += k; return true; }
};
struct proxy_test_access {
    static void client(proxy& px, mem_sock& s) { px.running_ = true; px.serve_l7_(s); px.running_ = false; }
    static void upstream(proxy& px, mem_sock& s) {
        std::string buf, head, first, body; std::vector<std::pair<std::string, std::string>> h;
        if (!px.read_head_(s, buf, head)) return;
        proxy::parse_headers(head, first, h);
        if (proxy::header(h, "content-length") || proxy::header(h, "transfer-encoding")) (void)px.read_body_(s, buf, h, body);
    }
};
}
FORGE_FUZZ({
    using namespace forge::loadbalancer;
    if (size < 1) return;
    static proxy* px = [] {
        proxy_config c; c.mode = proxy_config::l7; c.queue_wait_ns = 0; c.idle_timeout_ms = 0; c.retries = 1;
        return new proxy(std::make_shared<pool>(config{}), c);
    }();
    std::size_t chunk = data[0] >= 200 ? 65536 : 1 + data[0] % 61;
    mem_sock a{data + 1, size - 1, chunk}; proxy_test_access::client(*px, a);
    mem_sock b{data + 1, size - 1, chunk}; proxy_test_access::upstream(*px, b);
})
