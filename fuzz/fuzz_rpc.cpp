#include "fuzz.hpp"
#include <forge/graph/rpc.hpp>
#include <stdexcept>
// RPC (graph/rpc.hpp, rpc_secure.hpp). Byte 0 picks what the rest of the input is:
//   0..3  text of an authorized_keys file / a token file / a client-certificate map / a permission list
//   4     a byte stream sent to a server without authentication: FRPC op frames (name, attrs, tensors, shapes)
//   5     the same after a valid token handshake, on a server that also takes FRPS / FRPZ script frames
//   6     a byte stream sent to that server as the first thing it sees: the FRPH handshake decoder
// The streams of 4 and 5 also reach the remote-weight frames (the rpc.caps probe, FRPW uploads, FRPR ops with handles).
// A fuzzer cannot guess the BLAKE3 hash of an upload, so when (byte 0 / 7) is odd the harness first sends a valid FRPW
// of a fixed 16-float weight and replaces every run of 32 bytes 0xA5 in the input by that weight's hash: FRPR frames
// of the corpus then name a handle the server really holds, and mutations around it stay on the handle paths (lookup,
// size check, LRU, mixed inline inputs). The servers keep at most 4096 bytes of weights, so uploads evict each other.
// The servers are real rpc::server instances on 127.0.0.1 (the frame decoder reads from a socket), started once
// per process; each input is one connection: send, half-close, read the replies until the server closes.
namespace {
using namespace forge;
namespace rpc = forge::graph::rpc;
struct servers {
    std::unique_ptr<rpc::server> plain, authed;
    servers() {
        plain = std::make_unique<rpc::server>(nullptr, 0);
        rpc::server_security sec; sec.tokens = {"tok"}; sec.allow_scripts = true;
        sec.authorized_keys = {"00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"};
        authed = std::make_unique<rpc::server>(nullptr, 0, "127.0.0.1", sec);
        authed->script_handler = [](const std::string& src, const std::string&, const std::vector<std::string>& argv, std::uint32_t) {
            return rpc::server::script_result{true, "", "", std::to_string(src.size() + argv.size())};
        };
        for (auto* s : {plain.get(), authed.get()}) { s->max_tensor_elems = 1u << 16; s->weight_budget_bytes = 4096; s->start(); }
    }
    ~servers() { plain->stop(); authed->stop(); }
};
void talk(int port, std::string_view prefix, const std::uint8_t* d, std::size_t n) {
    auto c = os::socket::connect("127.0.0.1", port, 2000);
    if (!c) return;   // out of ephemeral ports under a long run: skip the input
    c->set_timeout_ms(2000);
    if (!prefix.empty() && !c->send_all(prefix.data(), prefix.size())) return;
    if (n) (void)c->send_all(d, n);
    c->shutdown_write();
    char buf[4096]; std::size_t got = 0;
    for (long k; (k = c->recv(buf, sizeof buf)) > 0;) if ((got += std::size_t(k)) > (64u << 20)) throw std::logic_error("rpc: reply larger than any frame allows");
}
}
FORGE_FUZZ({
    if (size < 1) return;
    std::string_view text{reinterpret_cast<const char*>(data + 1), size - 1};
    switch (data[0] % 7) {
    case 0: (void)rpc::parse_authorized_keys(text); return;
    case 1: (void)rpc::parse_tokens(text); return;
    case 2: (void)rpc::parse_client_map(text); return;
    case 3: (void)rpc::perms_string(rpc::parse_perms(text)); return;
    default: break;
    }
    static servers sv;
    // the fixed weight, its hash, and its upload frame (see the header comment)
    static const std::vector<float> fixed = [] { std::vector<float> v(16); for (std::size_t i = 0; i < v.size(); ++i) v[i] = float(i) - 7.5f; return v; }();
    static const hash::blake3::digest fixed_hash = [] { hash::blake3 h; h.update(fixed.data(), fixed.size() * 4); return h.final(); }();
    static const std::string upload = [] { rpc::detail::writer w; w.raw("FRPW", 4); w.raw(fixed_hash.data(), 32); w.shape({16}); w.raw(fixed.data(), 64); return std::string(w.b.begin(), w.b.end()); }();
    const bool resident = (data[0] / 7) % 2 == 1 && data[0] % 7 != 6;
    std::string body(text);
    if (resident) {
        static const std::string mark(32, char(0xA5));
        for (std::size_t at = 0; (at = body.find(mark, at)) != std::string::npos; at += 32) body.replace(at, 32, reinterpret_cast<const char*>(fixed_hash.data()), 32);
    }
    const auto* bd = reinterpret_cast<const std::uint8_t*>(body.data());
    if (data[0] % 7 == 4) talk(sv.plain->port(), resident ? upload : std::string(), bd, body.size());
    else if (data[0] % 7 == 5) {
        static const std::string hello = [] { rpc::detail::writer w; w.raw("FRPH", 4); w.u32(1); w.str("tok"); return std::string(w.b.begin(), w.b.end()); }();
        talk(sv.authed->port(), resident ? hello + upload : hello, bd, body.size());
    } else talk(sv.authed->port(), {}, bd, body.size());
    // the budget holds whatever the streams did: a counter that drifted would show here
    for (auto* s : {sv.plain.get(), sv.authed.get()}) if (s->weight_bytes() > s->weight_budget_bytes) throw std::logic_error("rpc: more weight bytes stored than the budget allows");
})
