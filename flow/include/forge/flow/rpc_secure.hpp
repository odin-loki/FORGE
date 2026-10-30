#pragma once
// Secure transport for forge rpc (docs/rpc.md): optional TLS (OpenSSL, when built with FORGE_HAVE_TLS) under an
// authentication handshake. Authentication reuses FORGE's .fpk signing primitives: Ed25519 via the vendored
// Monocypher and the same key files as `forge sign` (secret = 64 hex seed, public = 64 hex).
//   hello  = "FRPH" u32 method                      (1 = token, 2 = ed25519 key)
//   token  : client str token                       -> server u32 status [str error]
//   key    : client pub[32] -> server nonce[32] -> client sig[64] over ("forge-rpc-auth-v1" || nonce || pub)
//            -> server u32 status [str error]
// A server with no tokens/keys configured accepts unauthenticated peers (legacy, loopback only by default) and never
// runs scripts. Tokens are compared in constant time; nonces come from the OS CSPRNG (os::entropy).
#include <forge/os/os.hpp>
#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
extern "C" {
#include "monocypher.h"
#include "monocypher-ed25519.h"
}
#ifdef FORGE_HAVE_TLS
#include <csignal>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>
#endif

namespace forge::flow::rpc {
#ifdef FORGE_HAVE_TLS
inline namespace tls_v1 {
#else
inline namespace plain_v1 {
#endif

inline constexpr std::string_view auth_context = "forge-rpc-auth-v1";

namespace sec {
inline std::string hex(const std::uint8_t* p, std::size_t n) { static const char* d = "0123456789abcdef"; std::string s; for (std::size_t i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; } return s; }
inline bool unhex(std::string_view s, std::uint8_t* out, std::size_t n) {
    auto v = [](char c) -> int { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
    if (s.size() != n * 2) return false;
    for (std::size_t i = 0; i < n; ++i) { int a = v(s[2 * i]), b = v(s[2 * i + 1]); if (a < 0 || b < 0) return false; out[i] = std::uint8_t(a * 16 + b); }
    return true;
}
inline bool ct_equal(std::string_view a, std::string_view b) {
    unsigned char d = a.size() == b.size() ? 0 : 1; std::size_t n = std::max(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) d |= static_cast<unsigned char>((i < a.size() ? a[i] : 0) ^ (i < b.size() ? b[i] : 0));
    return d == 0;
}
inline std::vector<std::uint8_t> auth_message(const std::uint8_t* nonce, const std::uint8_t* pub) {
    std::vector<std::uint8_t> m(auth_context.begin(), auth_context.end()); m.insert(m.end(), nonce, nonce + 32); m.insert(m.end(), pub, pub + 32); return m;
}
} // namespace sec

// Ed25519 identity (same format as pack::keypair / `forge keygen`).
struct identity { std::array<std::uint8_t, 32> seed{}, pub{}; };
inline identity identity_from_seed_hex(std::string_view h) {
    identity k; if (!sec::unhex(h, k.seed.data(), 32)) raise(errc::parse_error, "rpc: secret key must be 64 hex chars");
    std::uint8_t sk[64], s2[32]; std::memcpy(s2, k.seed.data(), 32); crypto_ed25519_key_pair(sk, k.pub.data(), s2); crypto_wipe(sk, 64);
    return k;
}
inline std::string read_key_file(const std::string& path) {
    auto r = os::read_file(path); if (!r) raise(errc::io_error, "rpc: cannot read key {}: {}", path, r.err.what);
    std::string s; for (char c : r.value) if (!std::isspace(static_cast<unsigned char>(c))) s += c; return s;
}

struct tls_server_config { std::string cert_file, key_file; };
struct tls_client_config { bool enabled = false; std::string ca_file, server_name, fingerprint_sha256; bool insecure_skip_verify = false; };

struct server_security {
    std::vector<std::string> tokens;            // accepted bearer tokens
    std::vector<std::string> authorized_keys;   // accepted ed25519 public keys (hex)
    tls_server_config tls;                      // cert+key set = TLS required
    bool allow_scripts = false;                 // FRPS frames (remote script execution); requires auth
    bool auth_required() const { return !tokens.empty() || !authorized_keys.empty(); }
};
struct client_security {
    std::string token;                          // method 1
    std::shared_ptr<identity> key;              // method 2 (preferred when set)
    tls_client_config tls;
    bool any_auth() const { return key || !token.empty(); }
};

#ifdef FORGE_HAVE_TLS
namespace sec {
struct ctx_free { void operator()(SSL_CTX* c) const { SSL_CTX_free(c); } };
struct ssl_free { void operator()(SSL* s) const { SSL_free(s); } };
// OpenSSL writes with write(2), not send(MSG_NOSIGNAL): a peer that hangs up must not kill the process.
inline void no_sigpipe() {
#if !defined(_WIN32)
    static const bool once = [] { std::signal(SIGPIPE, SIG_IGN); return true; }(); (void)once;
#endif
}
inline std::string ssl_error() { char b[256] = {}; ERR_error_string_n(ERR_get_error(), b, sizeof b); return b; }
}
#endif

// A connected stream: plain TCP or TLS over it.
class channel {
public:
    channel() = default;
    explicit channel(os::socket s) : s_(std::move(s)) {}
    channel(channel&&) noexcept = default;
    channel& operator=(channel&&) noexcept = default;
    ~channel() { close(); }
    bool tls() const {
#ifdef FORGE_HAVE_TLS
        return ssl_ != nullptr;
#else
        return false;
#endif
    }
    os::socket& sock() { return s_; }
    bool send_all(const void* p, std::size_t n) {
#ifdef FORGE_HAVE_TLS
        if (ssl_) { auto* c = static_cast<const char*>(p); while (n) { int k = SSL_write(ssl_.get(), c, int(std::min<std::size_t>(n, 1 << 30))); if (k <= 0) return false; c += k; n -= std::size_t(k); } return true; }
#endif
        return s_.send_all(p, n);
    }
    bool recv_all(void* p, std::size_t n) {
#ifdef FORGE_HAVE_TLS
        if (ssl_) { auto* c = static_cast<char*>(p); while (n) { int k = SSL_read(ssl_.get(), c, int(std::min<std::size_t>(n, 1 << 30))); if (k <= 0) return false; c += k; n -= std::size_t(k); } return true; }
#endif
        return s_.recv_all(p, n);
    }
    void close() {
#ifdef FORGE_HAVE_TLS
        if (ssl_) { SSL_shutdown(ssl_.get()); ssl_.reset(); }
        ctx_.reset();
#endif
        s_.close();
    }
    void shutdown_both() { s_.shutdown_both(); }

#ifdef FORGE_HAVE_TLS
    // Server side: wrap an accepted socket. `ctx` is shared by all connections.
    bool accept_tls(std::shared_ptr<SSL_CTX> ctx, std::string* why) {
        ctx_ = std::move(ctx); ssl_.reset(SSL_new(ctx_.get()));
        if (!ssl_ || SSL_set_fd(ssl_.get(), int(s_.native())) != 1 || SSL_accept(ssl_.get()) != 1) { if (why) *why = "tls accept: " + sec::ssl_error(); ssl_.reset(); return false; }
        return true;
    }
    bool connect_tls(const tls_client_config& c, const std::string& host, std::string* why) {
        auto fail = [&](std::string m) { if (why) *why = "tls: " + m; ssl_.reset(); return false; };
        sec::no_sigpipe();
        ctx_.reset(SSL_CTX_new(TLS_client_method()), sec::ctx_free{});
        if (!ctx_) return fail(sec::ssl_error());
        SSL_CTX_set_min_proto_version(ctx_.get(), TLS1_2_VERSION);
        bool verify_chain = !c.ca_file.empty();
        if (verify_chain && SSL_CTX_load_verify_locations(ctx_.get(), c.ca_file.c_str(), nullptr) != 1) return fail("cannot load CA " + c.ca_file);
        if (!verify_chain && c.fingerprint_sha256.empty() && !c.insecure_skip_verify) return fail("no way to verify the server: set a CA file or a certificate fingerprint");
        SSL_CTX_set_verify(ctx_.get(), verify_chain ? SSL_VERIFY_PEER : SSL_VERIFY_NONE, nullptr);
        ssl_.reset(SSL_new(ctx_.get())); if (!ssl_) return fail(sec::ssl_error());
        std::string sni = c.server_name.empty() ? host : c.server_name;
        SSL_set_tlsext_host_name(ssl_.get(), sni.c_str());
        if (verify_chain) SSL_set1_host(ssl_.get(), sni.c_str());
        if (SSL_set_fd(ssl_.get(), int(s_.native())) != 1 || SSL_connect(ssl_.get()) != 1) return fail("handshake failed: " + sec::ssl_error());
        if (!c.fingerprint_sha256.empty()) {
            X509* cert = SSL_get1_peer_certificate(ssl_.get()); if (!cert) return fail("server sent no certificate");
            unsigned char md[EVP_MAX_MD_SIZE]; unsigned n = 0; X509_digest(cert, EVP_sha256(), md, &n); X509_free(cert);
            std::string want; for (char ch : c.fingerprint_sha256) if (std::isxdigit(static_cast<unsigned char>(ch))) want += char(std::tolower(static_cast<unsigned char>(ch)));
            if (!sec::ct_equal(sec::hex(md, n), want)) return fail("server certificate fingerprint mismatch");
        }
        return true;
    }
#endif
private:
    os::socket s_;
#ifdef FORGE_HAVE_TLS
    std::shared_ptr<SSL_CTX> ctx_; std::unique_ptr<SSL, sec::ssl_free> ssl_;
#endif
};

inline bool tls_available() {
#ifdef FORGE_HAVE_TLS
    return true;
#else
    return false;
#endif
}
#ifdef FORGE_HAVE_TLS
inline std::shared_ptr<SSL_CTX> make_server_tls(const tls_server_config& t) {
    sec::no_sigpipe();
    std::shared_ptr<SSL_CTX> c(SSL_CTX_new(TLS_server_method()), sec::ctx_free{});
    if (!c) raise(errc::io_error, "rpc tls: {}", sec::ssl_error());
    SSL_CTX_set_min_proto_version(c.get(), TLS1_2_VERSION);
    if (SSL_CTX_use_certificate_chain_file(c.get(), t.cert_file.c_str()) != 1) raise(errc::io_error, "rpc tls: cannot load certificate {}", t.cert_file);
    if (SSL_CTX_use_PrivateKey_file(c.get(), t.key_file.c_str(), SSL_FILETYPE_PEM) != 1 || SSL_CTX_check_private_key(c.get()) != 1) raise(errc::io_error, "rpc tls: cannot load key {}", t.key_file);
    return c;
}
#endif

namespace sec {
inline bool put_u32(channel& c, std::uint32_t v) { return c.send_all(&v, 4); }
inline bool get_u32(channel& c, std::uint32_t& v) { return c.recv_all(&v, 4); }
inline bool put_str(channel& c, std::string_view s) { return put_u32(c, std::uint32_t(s.size())) && c.send_all(s.data(), s.size()); }
inline bool get_str(channel& c, std::string& s, std::size_t cap) { std::uint32_t n; if (!get_u32(c, n) || n > cap) return false; s.assign(n, '\0'); return c.recv_all(s.data(), n); }
inline bool status(channel& c, std::string* why) {
    std::uint32_t st; if (!get_u32(c, st)) { if (why) *why = "rpc auth: connection closed"; return false; }
    if (st == 0) return true; std::string e; get_str(c, e, 4096); if (why) *why = "rpc auth: " + e; return false;
}
inline void reject(channel& c, std::string_view why) { put_u32(c, 1); put_str(c, why); }
} // namespace sec

// Client half of the handshake. No-op when the client has no credentials.
inline bool client_auth(channel& c, const client_security& s, std::string* why) {
    if (!s.any_auth()) return true;
    if (!c.send_all("FRPH", 4)) { if (why) *why = "rpc auth: send failed"; return false; }
    if (s.key) {
        if (!sec::put_u32(c, 2) || !c.send_all(s.key->pub.data(), 32)) { if (why) *why = "rpc auth: send failed"; return false; }
        std::uint8_t nonce[32]; if (!c.recv_all(nonce, 32)) { if (why) *why = "rpc auth: server closed (key not authorized?)"; return false; }
        auto m = sec::auth_message(nonce, s.key->pub.data());
        std::uint8_t sk[64], pub[32], seed[32], sig[64]; std::memcpy(seed, s.key->seed.data(), 32);
        crypto_ed25519_key_pair(sk, pub, seed); crypto_ed25519_sign(sig, sk, m.data(), m.size()); crypto_wipe(sk, 64);
        if (!c.send_all(sig, 64)) { if (why) *why = "rpc auth: send failed"; return false; }
    } else if (!sec::put_u32(c, 1) || !sec::put_str(c, s.token)) { if (why) *why = "rpc auth: send failed"; return false; }
    return sec::status(c, why);
}

// Server half. `first4` is the frame magic already read. Returns true when the peer may proceed;
// `authed` reports whether it proved an identity.
inline bool server_auth(channel& c, const server_security& s, const char first4[4], bool& authed, std::string* who) {
    authed = false;
    if (std::memcmp(first4, "FRPH", 4) != 0) { if (s.auth_required()) { sec::reject(c, "authentication required"); return false; } return true; }
    std::uint32_t method; if (!sec::get_u32(c, method)) return false;
    if (method == 1) {
        std::string tok; if (!sec::get_str(c, tok, 4096)) return false;
        bool ok = false; for (auto& t : s.tokens) ok |= sec::ct_equal(t, tok);
        if (!ok) { sec::reject(c, "bad token"); return false; }
        if (who) *who = "token";
    } else if (method == 2) {
        std::uint8_t pub[32]; if (!c.recv_all(pub, 32)) return false;
        auto ph = sec::hex(pub, 32); bool known = false; for (auto& k : s.authorized_keys) known |= sec::ct_equal(k, ph);
        std::uint8_t nonce[32]; if (!os::entropy(nonce, 32)) return false;
        if (!c.send_all(nonce, 32)) return false;
        std::uint8_t sig[64]; if (!c.recv_all(sig, 64)) return false;
        auto m = sec::auth_message(nonce, pub);
        if (!known || crypto_ed25519_check(sig, pub, m.data(), m.size()) != 0) { sec::reject(c, "key not authorized"); return false; }
        if (who) *who = "key:" + ph.substr(0, 8);
    } else { sec::reject(c, "unknown auth method"); return false; }
    authed = true;
    return sec::put_u32(c, 0);
}
} // inline namespace
} // namespace forge::flow::rpc
