#pragma once
// Secure transport for forge rpc (docs/rpc.md): optional TLS (OpenSSL, when built with FORGE_HAVE_TLS, optionally
// with client certificates = mTLS) under an authentication handshake. Authentication reuses FORGE's .fpk signing
// primitives: Ed25519 via the vendored Monocypher and the same key files as `forge sign` (secret = 64 hex seed,
// public = 64 hex).
//   hello  = "FRPH" u32 method   plain TCP: 1 = token, 2 = ed25519 key;  TLS: 3 = bound token, 4 = bound key
//   1 token : client str token                                     -> server u32 status [str error]
//   2 key   : client pub[32] -> server nonce[32] -> client sig[64] over ("forge-rpc-auth-v1" || nonce || pub)
//             -> server u32 status [str error]
//   3 token : client mac[64] = BLAKE2b-512(key = BLAKE2b-512(token), "forge-rpc-token-v2" || cb) -> server u32 status
//   4 key   : as 2, but signing ("forge-rpc-auth-v2" || nonce || pub || cb)
// cb is the TLS channel binding: RFC 9266 tls-exporter (32 bytes, label "EXPORTER-Channel-Binding", no context) on
// TLS 1.3 and on TLS 1.2 with extended master secret, else RFC 5929 tls-unique (the handshake's first Finished).
// Over TLS the server accepts only methods 3/4, so an auth exchange relayed or replayed through another TLS
// connection (which has a different cb) fails, and a bound token never crosses the wire. Plain TCP keeps 1/2.
// A server with no tokens/keys configured accepts unauthenticated peers (legacy, loopback only by default) and never
// runs scripts. Tokens are compared in constant time; nonces come from the OS CSPRNG (os::entropy).
// authorized_keys lines may carry an ssh-style options prefix limiting what the key may do:
//   [allow=PERM[,PERM...]] [ops=OP[,OP...]] [max-bytes=N] [script-timeout=MS] <64-hex public key> [comment]
// PERM: device.<op class> | scripts.run | scripts.fs | scripts.net | scripts.proc, or the aliases device | scripts | all.
// A line without options grants everything (the format before options existed). Token files use the same options
// ("[options] <token> [name]"), and an mTLS client map ("[options] sha256:<hex> | san:<type>:<value> | cn:<name>
// [comment]") limits what each client certificate may do; a request must be allowed by both its token/key and its
// certificate. Revocation: PEM CRLs checked in the TLS handshake (tls_server_config::crl_files) and a deny-list of
// certificate fingerprints (server_security::revoked_certs / revoked_files).
#include <forge/os/os.hpp>
#include <forge/security/files.hpp>
#include <forge/security/secure_buffer.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
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
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>
#endif

namespace forge::graph::rpc {
#ifdef FORGE_HAVE_TLS
inline namespace tls_v1 {
#else
inline namespace plain_v1 {
#endif

inline constexpr std::string_view auth_context = "forge-rpc-auth-v1";
inline constexpr std::string_view auth_context_bound = "forge-rpc-auth-v2";
inline constexpr std::string_view token_context_bound = "forge-rpc-token-v2";

// What an authenticated peer may do ("allow=" option of authorized_keys, token files and client maps). Bare tokens,
// keys without options and unauthenticated legacy peers get all.
// Device ops (FRPC frames) are split by the op's class in the op table (ops::op_class, ops/defs/*.fspec); remote
// scripts (FRPS / FRPZ frames) into running one at all and the capabilities its child process keeps. `device` and
// `scripts` (the only permissions before the split) are the full sets.
enum perm : unsigned {
    perm_none = 0,
    perm_device_elementwise = 1u << 0, perm_device_reduction = 1u << 1, perm_device_contraction = 1u << 2,
    perm_device_data_movement = 1u << 3, perm_device_stateful = 1u << 4,
    perm_scripts_run = 1u << 8,     // send a script at all; without the three below its child gets no fs / net / proc
    perm_scripts_fs = 1u << 9, perm_scripts_net = 1u << 10, perm_scripts_proc = 1u << 11,
    perm_device = perm_device_elementwise | perm_device_reduction | perm_device_contraction | perm_device_data_movement | perm_device_stateful,
    perm_scripts = perm_scripts_run | perm_scripts_fs | perm_scripts_net | perm_scripts_proc,
    perm_all = perm_device | perm_scripts };
// Bit for device ops of class index `cls` (the value of ops::op_class).
inline constexpr unsigned perm_for_op_class(unsigned cls) { return cls < 5 ? 1u << cls : 0u; }

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
// Method 4: the v1 message under its own context, followed by the TLS channel binding.
inline std::vector<std::uint8_t> auth_message_bound(const std::uint8_t* nonce, const std::uint8_t* pub, const std::vector<std::uint8_t>& cb) {
    std::vector<std::uint8_t> m(auth_context_bound.begin(), auth_context_bound.end()); m.insert(m.end(), nonce, nonce + 32); m.insert(m.end(), pub, pub + 32);
    m.insert(m.end(), cb.begin(), cb.end()); return m;
}
// Method 3: a MAC of the channel binding keyed by the token, so the token itself never crosses the wire.
inline std::array<std::uint8_t, 64> token_mac(std::string_view token, const std::vector<std::uint8_t>& cb) {
    std::uint8_t k[64]; crypto_blake2b(k, 64, reinterpret_cast<const std::uint8_t*>(token.data()), token.size());
    std::vector<std::uint8_t> m(token_context_bound.begin(), token_context_bound.end()); m.insert(m.end(), cb.begin(), cb.end());
    std::array<std::uint8_t, 64> out{}; crypto_blake2b_keyed(out.data(), 64, k, 64, m.data(), m.size()); crypto_wipe(k, 64);
    return out;
}
inline bool ct_equal_bytes(const std::uint8_t* a, const std::uint8_t* b, std::size_t n) { unsigned char d = 0; for (std::size_t i = 0; i < n; ++i) d |= a[i] ^ b[i]; return d == 0; }
} // namespace sec

// Ed25519 identity (same format as modelfile::keypair / `forge keygen`). The seed is in locked, non-dumpable memory,
// wiped on destruction (FSS-CRY-2); identity is move-only (it is shared through std::shared_ptr).
struct identity { security::secure_bytes<32> seed; std::array<std::uint8_t, 32> pub{}; };
inline identity identity_from_seed_hex(std::string_view h) {
    identity k; if (!sec::unhex(h, k.seed.data(), 32)) raise(errc::parse_error, "rpc: secret key must be 64 hex chars");
    std::uint8_t sk[64], s2[32]; std::memcpy(s2, k.seed.data(), 32); crypto_ed25519_key_pair(sk, k.pub.data(), s2); crypto_wipe(sk, 64);
    return k;
}
// A secret key file: private under a strict security policy (FSS-FIO-2), a warning otherwise. The file's bytes are
// wiped after reading; identity_from_seed_hex copies the returned hex into locked memory.
inline std::string read_key_file(const std::string& path) {
    security::require_private_file("rpc key", path, security::file_kind::secret);
    auto r = os::read_file(path); if (!r) raise(errc::io_error, "rpc: cannot read key {}: {}", path, r.err.what);
    std::string s; s.reserve(r.value.size()); for (char c : r.value) if (!std::isspace(static_cast<unsigned char>(c))) s += c;
    if (!r.value.empty()) crypto_wipe(r.value.data(), r.value.size());
    return s;
}

// client_ca_file set = mTLS: every client must present a certificate that chains to it.
// crl_files (PEM, one or more CRLs each; needs client_ca_file): the handshake refuses client certificates they
// revoke. Every certificate of the client's chain is checked (X509_V_FLAG_CRL_CHECK_ALL), so each CA in it needs a
// CRL that is currently valid; a missing or expired CRL refuses the client (fail closed).
struct tls_server_config { std::string cert_file, key_file, client_ca_file; std::vector<std::string> crl_files; };
// cert_file/key_file: client certificate presented to servers that require one (mTLS).
struct tls_client_config { bool enabled = false; std::string ca_file, server_name, fingerprint_sha256; bool insecure_skip_verify = false; std::string cert_file, key_file; };

// Per-rule limits next to the permissions (options "ops=", "max-bytes=", "script-timeout="). Default: none.
struct limits {
    bool ops_limited = false; std::vector<std::string> ops;   // ops=: the only device ops allowed (sorted, unique)
    std::uint64_t max_bytes = 0;                // max-bytes=: tensor bytes per device op (inputs + output) and source bytes per script; 0 = no limit
    std::uint32_t script_timeout_ms = 0;        // script-timeout=: upper limit per script; 0 = the server's
    bool any() const { return ops_limited || max_bytes || script_timeout_ms; }
    bool op_allowed(std::string_view name) const { return !ops_limited || std::binary_search(ops.begin(), ops.end(), name, std::less<>{}); }
    // Both apply (a token or key and its client certificate): the stricter of each.
    void intersect(const limits& o) {
        if (o.ops_limited) {
            if (!ops_limited) { ops = o.ops; ops_limited = true; }
            else { std::vector<std::string> both; std::set_intersection(ops.begin(), ops.end(), o.ops.begin(), o.ops.end(), std::back_inserter(both)); ops = std::move(both); }
        }
        if (o.max_bytes && (!max_bytes || o.max_bytes < max_bytes)) max_bytes = o.max_bytes;
        if (o.script_timeout_ms && (!script_timeout_ms || o.script_timeout_ms < script_timeout_ms)) script_timeout_ms = o.script_timeout_ms;
    }
    // " ops=a,b max-bytes=N script-timeout=MS" for logs ("" without limits).
    std::string str() const {
        std::string s;
        if (ops_limited) { s += " ops="; for (std::size_t i = 0; i < ops.size(); ++i) { if (i) s += ','; s += ops[i]; } }
        if (max_bytes) s += " max-bytes=" + std::to_string(max_bytes);
        if (script_timeout_ms) s += " script-timeout=" + std::to_string(script_timeout_ms);
        return s;
    }
};

// One authorized_keys entry. Constructible from a bare hex key (full access), as before options existed.
struct authorized_key {
    std::string pub;                            // 64 hex, lower case
    unsigned perms = perm_all;
    std::string comment;
    limits lim;
    authorized_key() = default;
    authorized_key(std::string hex, unsigned p = perm_all, std::string c = {}) : pub(std::move(hex)), perms(p), comment(std::move(c)) {
        for (auto& ch : pub) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    }
    authorized_key(const char* hex) : authorized_key(std::string(hex)) {}
};
namespace sec {
struct perm_name { std::string_view name; unsigned bits; };
// The aliases for the full sets, then one name per permission bit.
inline constexpr perm_name perm_names[] = {
    {"all", perm_all}, {"device", perm_device}, {"scripts", perm_scripts},
    {"device.elementwise", perm_device_elementwise}, {"device.reduction", perm_device_reduction}, {"device.contraction", perm_device_contraction},
    {"device.data_movement", perm_device_data_movement}, {"device.stateful", perm_device_stateful},
    {"scripts.run", perm_scripts_run}, {"scripts.fs", perm_scripts_fs}, {"scripts.net", perm_scripts_net}, {"scripts.proc", perm_scripts_proc}};
}
// "device.contraction,scripts" -> perm bits. Unknown names are errors (fail closed).
inline unsigned parse_perms(std::string_view list) {
    unsigned p = perm_none;
    while (!list.empty()) {
        auto c = list.find(','); auto w = list.substr(0, c); list = c == std::string_view::npos ? std::string_view{} : list.substr(c + 1);
        if (w.empty()) continue;
        bool known = false;
        for (auto& n : sec::perm_names) if (n.name == w) { p |= n.bits; known = true; }
        if (!known) raise(errc::parse_error, "rpc: unknown permission '{}' (want device, scripts, all, device.<op class> or scripts.run/fs/net/proc)", w);
    }
    return p;
}
// "all", "none", "device", "device.contraction,scripts.run", ...: a full set by its alias, otherwise each permission.
inline std::string perms_string(unsigned p) {
    p &= perm_all;
    if (p == perm_all) return "all";
    if (p == perm_none) return "none";
    std::string s; auto add = [&](std::string_view n) { if (!s.empty()) s += ','; s += n; };
    const unsigned sets[] = {perm_device, perm_scripts};
    for (unsigned set : sets) {
        if ((p & set) == set) { add(set == perm_device ? "device" : "scripts"); continue; }
        for (auto& n : sec::perm_names) if ((n.bits & (n.bits - 1)) == 0 && (n.bits & set) && (p & n.bits)) add(n.name);
    }
    return s;
}
namespace sec {
// Config text (authorized_keys, token files, client-certificate maps): one entry per line, fields separated by
// blanks, blank lines and lines starting with '#' ignored. fn(lineno, fields).
template <class F> void for_each_config_line(std::string_view text, F&& fn) {
    std::size_t lineno = 0; auto is_ws = [](char c) { return c == ' ' || c == '\t' || c == '\r'; };
    while (!text.empty()) {
        ++lineno; auto nl = text.find('\n'); auto line = text.substr(0, nl); text = nl == std::string_view::npos ? std::string_view{} : text.substr(nl + 1);
        std::vector<std::string_view> f;
        for (std::size_t i = 0; i < line.size();) { while (i < line.size() && is_ws(line[i])) ++i; auto j = i; while (j < line.size() && !is_ws(line[j])) ++j; if (j > i) f.push_back(line.substr(i, j - i)); i = j; }
        if (f.empty() || f[0][0] == '#') continue;
        fn(lineno, f);
    }
}
// Does this field start like an option (allow=, ops=, max-bytes=, script-timeout=)? Token files use it to tell
// options from the token.
inline bool is_option(std::string_view f) { return f.starts_with("allow=") || f.starts_with("ops=") || f.starts_with("max-bytes=") || f.starts_with("script-timeout="); }
// Decimal with an optional k / m / g suffix (powers of 1024); false on anything else or on overflow.
inline bool parse_size(std::string_view v, std::uint64_t& out) {
    std::uint64_t n = 0, mul = 1; std::size_t i = 0;
    for (; i < v.size() && v[i] >= '0' && v[i] <= '9'; ++i) { if (n > (UINT64_MAX - 9) / 10) return false; n = n * 10 + std::uint64_t(v[i] - '0'); }
    if (i == 0) return false;
    if (i + 1 == v.size()) { char u = char(std::tolower(static_cast<unsigned char>(v[i]))); mul = u == 'k' ? 1ull << 10 : u == 'm' ? 1ull << 20 : u == 'g' ? 1ull << 30 : 0; if (!mul) return false; }
    else if (i != v.size()) return false;
    if (mul > 1 && n > UINT64_MAX / mul) return false;
    out = n * mul; return true;
}
// One option. "allow=PERM[,PERM...]" and "ops=OP[,OP...]" add up when repeated; "max-bytes=N[k|m|g]" and
// "script-timeout=MS" may be given once and must be positive. Anything else is an error (fail closed).
inline void apply_option(std::string_view what, std::size_t lineno, std::string_view f, unsigned& p, bool& have_allow, limits& lim) {
    auto eq = f.find('='); auto key = f.substr(0, eq);
    if (eq == std::string_view::npos) raise(errc::parse_error, "{} line {}: unknown option '{}'", what, lineno, key);
    auto val = f.substr(eq + 1);
    if (key == "allow") {
        unsigned v;
        try { v = parse_perms(val); }
        catch (const std::exception& e) { std::string_view m = e.what(); if (m.starts_with("rpc: ")) m.remove_prefix(5); raise(errc::parse_error, "{} line {}: {}", what, lineno, m); }
        p = have_allow ? (p | v) : v; have_allow = true;
    } else if (key == "ops") {
        lim.ops_limited = true;
        while (!val.empty()) { auto c = val.find(','); auto w = val.substr(0, c); val = c == std::string_view::npos ? std::string_view{} : val.substr(c + 1); if (!w.empty()) lim.ops.emplace_back(w); }
        std::sort(lim.ops.begin(), lim.ops.end()); lim.ops.erase(std::unique(lim.ops.begin(), lim.ops.end()), lim.ops.end());
    } else if (key == "max-bytes") {
        if (lim.max_bytes) raise(errc::parse_error, "{} line {}: max-bytes= given twice", what, lineno);
        if (!parse_size(val, lim.max_bytes) || !lim.max_bytes) raise(errc::parse_error, "{} line {}: max-bytes= wants a positive size (bytes, or with a k, m or g suffix)", what, lineno);
    } else if (key == "script-timeout") {
        std::uint64_t ms = 0;
        if (lim.script_timeout_ms) raise(errc::parse_error, "{} line {}: script-timeout= given twice", what, lineno);
        if (!parse_size(val, ms) || !ms || ms > UINT32_MAX || !(val.back() >= '0' && val.back() <= '9')) raise(errc::parse_error, "{} line {}: script-timeout= wants positive milliseconds", what, lineno);
        lim.script_timeout_ms = std::uint32_t(ms);
    } else raise(errc::parse_error, "{} line {}: unknown option '{}'", what, lineno, key);
}
inline std::string join_from(const std::vector<std::string_view>& f, std::size_t i) { std::string s; for (; i < f.size(); ++i) { if (!s.empty()) s += ' '; s += f[i]; } return s; }
} // namespace sec

// authorized_keys text: one key per line, '#' comments, optional options before the key (see the header comment).
// Unknown options or permissions are errors (fail closed).
inline std::vector<authorized_key> parse_authorized_keys(std::string_view text) {
    std::vector<authorized_key> out;
    sec::for_each_config_line(text, [&](std::size_t lineno, const std::vector<std::string_view>& f) {
        std::uint8_t tmp[32]; std::size_t k = 0; unsigned p = perm_all; bool have_allow = false; limits lim;
        for (; k < f.size() && !sec::unhex(f[k], tmp, 32); ++k) {
            if (f[k].find('=') == std::string_view::npos) raise(errc::parse_error, "authorized_keys line {}: '{}' is neither a 64-hex key nor an option", lineno, f[k]);
            sec::apply_option("authorized_keys", lineno, f[k], p, have_allow, lim);
        }
        if (k == f.size()) raise(errc::parse_error, "authorized_keys line {}: no 64-hex public key", lineno);
        out.emplace_back(std::string(f[k]), p, sec::join_from(f, k + 1)); out.back().lim = std::move(lim);
    });
    return out;
}

// One accepted bearer token and what it may do. Constructible from a bare token (full access), as before scopes.
struct token_entry {
    security::secure_string token;              // locked, non-dumpable, wiped on destruction (FSS-CRY-2)
    unsigned perms = perm_all;
    std::string name;                           // comment from the token file; used in logs instead of the token
    limits lim;
    token_entry() = default;
    token_entry(std::string t, unsigned p = perm_all, std::string n = {}) : token(t), perms(p), name(std::move(n)) { if (!t.empty()) crypto_wipe(t.data(), t.size()); }
    token_entry(const char* t) : token_entry(std::string(t)) {}
};
// Token file (`--token-file`): one token per line, as authorized_keys but with a token instead of a key:
//   [allow=PERM[,PERM...]] [ops=...] [max-bytes=N] [script-timeout=MS] <token> [name]
// A line without options grants everything (a file holding just a token keeps working). Leading fields that start
// with an option name and '=' (sec::is_option) are options, so a token may contain '=' (base64) but may not itself
// start with "allow=", "ops=", "max-bytes=" or "script-timeout=".
inline std::vector<token_entry> parse_tokens(std::string_view text, std::string_view what = "token file") {
    std::vector<token_entry> out;
    sec::for_each_config_line(text, [&](std::size_t lineno, const std::vector<std::string_view>& f) {
        std::size_t k = 0; unsigned p = perm_all; bool have_allow = false; limits lim;
        for (; k < f.size() && sec::is_option(f[k]); ++k) sec::apply_option(what, lineno, f[k], p, have_allow, lim);
        if (k == f.size()) raise(errc::parse_error, "{} line {}: no token", what, lineno);
        out.emplace_back(std::string(f[k]), p, sec::join_from(f, k + 1)); out.back().lim = std::move(lim);
    });
    return out;
}

// ---- subjectAltName matching (client map "san:" rules). Pure text, so it is built and tested without OpenSSL.
namespace sec {
inline std::string lower(std::string_view v) { std::string s(v); for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c))); return s; }
// IPv4 dotted quad or IPv6 (with "::" and an optional trailing dotted quad) -> 4 or 16 bytes.
inline bool parse_ip(std::string_view s, std::vector<std::uint8_t>& out) {
    auto v4 = [](std::string_view t, std::uint8_t* o) {
        for (int k = 0; k < 4; ++k) {
            auto d = t.find('.'); auto part = t.substr(0, d);
            if ((k < 3) != (d != std::string_view::npos) || part.empty() || part.size() > 3) return false;
            unsigned v = 0; for (char c : part) { if (c < '0' || c > '9') return false; v = v * 10 + unsigned(c - '0'); }
            if (v > 255) return false;
            o[k] = std::uint8_t(v); t = d == std::string_view::npos ? std::string_view{} : t.substr(d + 1);
        }
        return true;
    };
    out.clear();
    if (s.empty()) return false;
    if (s.find(':') == std::string_view::npos) { out.resize(4); if (v4(s, out.data())) return true; out.clear(); return false; }
    std::vector<std::uint16_t> head, tail; bool gap = false; std::size_t i = 0;
    if (s.starts_with("::")) { gap = true; i = 2; } else if (s[0] == ':') return false;
    while (i < s.size()) {
        auto j = s.find(':', i); auto g = s.substr(i, j == std::string_view::npos ? j : j - i);
        auto& dst = gap ? tail : head;
        if (g.find('.') != std::string_view::npos) {
            std::uint8_t b[4]; if (j != std::string_view::npos || !v4(g, b)) return false;
            dst.push_back(std::uint16_t(b[0] << 8 | b[1])); dst.push_back(std::uint16_t(b[2] << 8 | b[3]));
        } else {
            if (g.empty() || g.size() > 4) return false;
            unsigned v = 0; for (char c : g) { int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; if (d < 0) return false; v = v * 16 + unsigned(d); }
            dst.push_back(std::uint16_t(v));
        }
        if (j == std::string_view::npos) break;
        i = j + 1;
        if (i == s.size()) return false;                       // a single trailing ':'
        if (s[i] == ':') { if (gap) return false; gap = true; ++i; }
    }
    std::size_t n = head.size() + tail.size();
    if (gap ? n > 7 : n != 8) return false;
    head.resize(8 - tail.size(), 0); head.insert(head.end(), tail.begin(), tail.end());
    for (auto g : head) { out.push_back(std::uint8_t(g >> 8)); out.push_back(std::uint8_t(g & 255)); }
    return true;
}
// Canonical text of an address: "10.1.2.3", or eight lower-case hex groups without "::" ("fd00:0:0:0:0:0:0:7").
// "" for any other length.
inline std::string ip_text(const std::uint8_t* p, std::size_t n) {
    std::string s;
    if (n == 4) { for (int k = 0; k < 4; ++k) { if (k) s += '.'; s += std::to_string(unsigned(p[k])); } }
    else if (n == 16) { static const char* d = "0123456789abcdef";
        for (int k = 0; k < 8; ++k) { if (k) s += ':'; unsigned g = unsigned(p[2 * k]) << 8 | p[2 * k + 1]; bool lead = true;
            for (int sh = 12; sh >= 0; sh -= 4) { unsigned x = (g >> sh) & 15; if (x || !lead || sh == 0) { s += d[x]; lead = false; } } } }
    return s;
}
// Canonical form of a subjectAltName value, for rules and for certificate entries alike ("" = not usable):
//   dns   lower case, one trailing '.' dropped        email  the domain (after the last '@') in lower case
//   uri   as written (compared byte for byte)          ip     ip_text of the parsed address
// Values with blanks, control characters or NUL are not usable.
inline std::string san_canonical(std::string_view type, std::string_view v) {
    for (char c : v) if (static_cast<unsigned char>(c) <= ' ' || c == 0x7f) return {};
    if (v.empty()) return {};
    if (type == "dns") { auto s = lower(v); if (s.size() > 1 && s.back() == '.') s.pop_back(); return s.find("..") != std::string::npos || s.front() == '.' ? std::string{} : s; }
    if (type == "email") { auto at = v.rfind('@'); if (at == std::string_view::npos || at == 0 || at + 1 == v.size()) return {}; return std::string(v.substr(0, at + 1)) + lower(v.substr(at + 1)); }
    if (type == "uri") return v.find(':') == std::string_view::npos ? std::string{} : std::string(v);
    if (type == "ip") { std::vector<std::uint8_t> b; return parse_ip(v, b) ? ip_text(b.data(), b.size()) : std::string{}; }
    return {};
}
// A san:dns: rule against one DNS name of the certificate (both canonical). "*" is a wildcard only in a rule and
// only as its whole leftmost label, where it stands for exactly one label: "*.ci.example" matches "a.ci.example",
// not "ci.example" or "a.b.ci.example". A name in the certificate is never expanded: a certificate for
// "*.ci.example" matches only a rule spelled the same.
inline bool dns_rule_matches(std::string_view rule, std::string_view name) {
    if (rule.empty() || name.empty()) return false;
    if (rule == name) return true;
    if (!rule.starts_with("*.") || name.find('*') != std::string_view::npos) return false;
    auto dot = name.find('.');
    return dot != std::string_view::npos && dot > 0 && name.substr(dot) == rule.substr(1);
}
} // namespace sec

// mTLS client-certificate map (`--tls-client-map`): which verified client certificates are accepted and what each
// may do. One rule per line (options as in authorized_keys):
//   [options] sha256:<64 hex, ':' separators allowed>          [comment]
//   [options] san:dns:<name or *.suffix> | san:uri:<uri> | san:email:<address> | san:ip:<v4 or v6 address> [comment]
//   [options] cn:<subject common name, no blanks>              [comment]
// A certificate matches its fingerprint rule first, else the first san: rule (in file order) that one of its
// subjectAltName entries satisfies, else the rule for its (single) subject CN. With a map, a certificate that matches
// no rule is refused.
struct cert_rule {
    enum kind_t { sha256, cn, san_dns, san_uri, san_email, san_ip } kind = sha256;
    std::string value;                          // lower-case hex fingerprint, the CN, or the canonical SAN value (sec::san_canonical)
    unsigned perms = perm_all;
    std::string comment;
    limits lim;
    bool is_san() const { return kind >= san_dns; }
    // "dns" / "uri" / "email" / "ip" for san rules.
    std::string_view san_type() const { return kind == san_dns ? "dns" : kind == san_uri ? "uri" : kind == san_email ? "email" : kind == san_ip ? "ip" : ""; }
    // Does a certificate entry "<type>:<canonical value>" (channel::peer_cert_sans) satisfy this san rule?
    bool matches_san(std::string_view entry) const {
        auto t = san_type(); if (t.empty() || entry.size() <= t.size() || !entry.starts_with(t) || entry[t.size()] != ':') return false;
        auto v = entry.substr(t.size() + 1);
        return kind == san_dns ? sec::dns_rule_matches(value, v) : !value.empty() && v == std::string_view(value);
    }
};
inline std::vector<cert_rule> parse_client_map(std::string_view text, std::string_view what = "client map") {
    std::vector<cert_rule> out;
    sec::for_each_config_line(text, [&](std::size_t lineno, const std::vector<std::string_view>& f) {
        std::size_t k = 0; unsigned p = perm_all; bool have_allow = false; limits lim;
        for (; k < f.size() && !f[k].starts_with("sha256:") && !f[k].starts_with("cn:") && !f[k].starts_with("san:"); ++k) sec::apply_option(what, lineno, f[k], p, have_allow, lim);
        if (k == f.size()) raise(errc::parse_error, "{} line {}: no certificate identity (want sha256:<hex>, san:<type>:<value> or cn:<name>)", what, lineno);
        cert_rule r; r.perms = p; r.lim = std::move(lim); r.comment = sec::join_from(f, k + 1);
        if (f[k].starts_with("cn:")) {
            r.kind = cert_rule::cn; r.value = std::string(f[k].substr(3));
            if (r.value.empty()) raise(errc::parse_error, "{} line {}: empty cn:", what, lineno);
        } else if (f[k].starts_with("san:")) {
            auto rest = f[k].substr(4); auto c = rest.find(':'); auto type = rest.substr(0, c); auto v = c == std::string_view::npos ? std::string_view{} : rest.substr(c + 1);
            r.kind = type == "dns" ? cert_rule::san_dns : type == "uri" ? cert_rule::san_uri : type == "email" ? cert_rule::san_email : type == "ip" ? cert_rule::san_ip : cert_rule::sha256;
            if (!r.is_san()) raise(errc::parse_error, "{} line {}: san: wants a type (san:dns:, san:uri:, san:email: or san:ip:)", what, lineno);
            r.value = sec::san_canonical(type, v);
            if (r.value.empty()) raise(errc::parse_error, "{} line {}: bad san:{}: value '{}'", what, lineno, type, v);
            // a wildcard is the whole leftmost label of a dns rule, with a name after it; nowhere else
            auto star = r.value.find('*');
            if (star != std::string::npos && (r.kind != cert_rule::san_dns || star != 0 || r.value.size() < 3 || r.value[1] != '.' || r.value.find('*', 1) != std::string::npos))
                raise(errc::parse_error, "{} line {}: '*' is only a wildcard as the whole leftmost label of a san:dns: rule (*.example.org)", what, lineno);
        } else {
            std::string h; for (char c : f[k].substr(7)) if (c != ':') h += char(std::tolower(static_cast<unsigned char>(c)));
            std::uint8_t tmp[32];
            if (!sec::unhex(h, tmp, 32)) raise(errc::parse_error, "{} line {}: sha256: wants a 64-hex certificate fingerprint", what, lineno);
            r.value = h;
        }
        for (auto& o : out) if (o.kind == r.kind && o.value == r.value) raise(errc::parse_error, "{} line {}: duplicate rule for {}", what, lineno, f[k]);
        out.push_back(std::move(r));
    });
    return out;
}
// Deny-list of client certificates (`--tls-revoked`): one SHA-256 certificate fingerprint per line, as in a client
// map's sha256: rule ("sha256:" prefix and ':' separators optional), then a comment. Returns lower-case hex.
inline std::vector<std::string> parse_revoked(std::string_view text, std::string_view what = "revoked list") {
    std::vector<std::string> out;
    sec::for_each_config_line(text, [&](std::size_t lineno, const std::vector<std::string_view>& f) {
        auto v = f[0]; if (v.starts_with("sha256:")) v.remove_prefix(7);
        std::string h; for (char c : v) if (c != ':') h += char(std::tolower(static_cast<unsigned char>(c)));
        std::uint8_t tmp[32];
        if (!sec::unhex(h, tmp, 32)) raise(errc::parse_error, "{} line {}: want a 64-hex SHA-256 certificate fingerprint", what, lineno);
        out.push_back(std::move(h));
    });
    return out;
}

struct server_security {
    std::vector<token_entry> tokens;            // accepted bearer tokens and what each may do
    std::vector<authorized_key> authorized_keys;// accepted ed25519 public keys and what each may do
    tls_server_config tls;                      // cert+key set = TLS required; client_ca_file = mTLS
    std::vector<cert_rule> client_map;          // mTLS identities -> permissions (needs tls.client_ca_file)
    bool allow_scripts = false;                 // FRPS frames (remote script execution); requires tokens or keys
    // Deny-list of client certificates by SHA-256 fingerprint (lower-case hex; needs tls.client_ca_file): a listed
    // certificate is refused whatever token, key or client-map rule it comes with. revoked_files are parse_revoked
    // files the server reads at start and again when they change (rpc::server).
    std::vector<std::string> revoked_certs, revoked_files;
    // Tokens or keys configured: every peer must present one (a client certificate alone is not enough).
    bool has_credentials() const { return !tokens.empty() || !authorized_keys.empty(); }
    // Some proof of identity is required: credentials, or (with only a client map) a mapped client certificate.
    bool auth_required() const { return has_credentials() || !client_map.empty(); }
};
struct client_security {
    security::secure_string token;              // method 1 (locked memory, FSS-CRY-2)
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
    // TLS channel binding of this connection (empty on plain TCP): RFC 9266 tls-exporter on TLS 1.3 and on TLS 1.2
    // with extended master secret, otherwise RFC 5929 tls-unique (the first Finished message of the handshake).
    std::vector<std::uint8_t> channel_binding() const {
#ifdef FORGE_HAVE_TLS
        if (!ssl_) return {};
        SSL* s = ssl_.get();
        if (SSL_version(s) >= TLS1_3_VERSION || SSL_get_extms_support(s) == 1) {
            static constexpr char label[] = "EXPORTER-Channel-Binding";
            std::vector<std::uint8_t> cb(32);
            if (SSL_export_keying_material(s, cb.data(), cb.size(), label, sizeof label - 1, nullptr, 0, 0) == 1) return cb;
            return {};
        }
        // tls-unique: the client's Finished on a full handshake, the server's on a resumed one.
        std::uint8_t b[EVP_MAX_MD_SIZE];
        bool first_is_mine = (SSL_is_server(s) == 1) == (SSL_session_reused(s) == 1);
        std::size_t n = first_is_mine ? SSL_get_finished(s, b, sizeof b) : SSL_get_peer_finished(s, b, sizeof b);
        return std::vector<std::uint8_t>(b, b + std::min(n, sizeof b));
#else
        return {};
#endif
    }
    // mTLS: subject of the verified client certificate ("" when there is none).
    std::string peer_subject() const {
#ifdef FORGE_HAVE_TLS
        if (!ssl_) return {};
        X509* c = SSL_get1_peer_certificate(ssl_.get()); if (!c) return {};
        char b[256] = {}; X509_NAME_oneline(X509_get_subject_name(c), b, sizeof b); X509_free(c); return b;
#else
        return {};
#endif
    }
    // mTLS: SHA-256 of the client certificate (DER), lower-case hex ("" when there is none).
    std::string peer_cert_sha256() const {
#ifdef FORGE_HAVE_TLS
        if (!ssl_) return {};
        X509* c = SSL_get1_peer_certificate(ssl_.get()); if (!c) return {};
        unsigned char md[EVP_MAX_MD_SIZE]; unsigned n = 0; bool ok = X509_digest(c, EVP_sha256(), md, &n) == 1; X509_free(c);
        return ok ? sec::hex(md, n) : std::string{};
#else
        return {};
#endif
    }
    // mTLS: the subject common name, UTF-8 ("" when there is no certificate, no CN, more than one CN, or a CN with NUL).
    std::string peer_cert_cn() const {
#ifdef FORGE_HAVE_TLS
        if (!ssl_) return {};
        X509* c = SSL_get1_peer_certificate(ssl_.get()); if (!c) return {};
        std::string out; auto* nm = const_cast<X509_NAME*>(X509_get_subject_name(c));   // const in OpenSSL 3, not 1.1
        int i = X509_NAME_get_index_by_NID(nm, NID_commonName, -1);
        if (i >= 0 && X509_NAME_get_index_by_NID(nm, NID_commonName, i) < 0) {
            unsigned char* u = nullptr; int n = ASN1_STRING_to_UTF8(&u, X509_NAME_ENTRY_get_data(X509_NAME_get_entry(nm, i)));
            if (n > 0 && std::memchr(u, 0, std::size_t(n)) == nullptr) out.assign(reinterpret_cast<char*>(u), std::size_t(n));
            OPENSSL_free(u);
        }
        X509_free(c); return out;
#else
        return {};
#endif
    }
    // mTLS: the certificate's subjectAltName entries of the types client-map rules know, as "<type>:<canonical
    // value>" ("dns:host.example", "uri:spiffe://...", "email:a@b.example", "ip:10.1.2.3"; sec::san_canonical).
    // Entries of other types, with NUL or blanks, or that do not parse are left out (they match nothing).
    std::vector<std::string> peer_cert_sans() const {
        std::vector<std::string> out;
#ifdef FORGE_HAVE_TLS
        if (!ssl_) return out;
        X509* c = SSL_get1_peer_certificate(ssl_.get()); if (!c) return out;
        auto* gens = static_cast<GENERAL_NAMES*>(X509_get_ext_d2i(c, NID_subject_alt_name, nullptr, nullptr));
        for (int i = 0; gens && i < sk_GENERAL_NAME_num(gens); ++i) {
            int type = 0; auto* v = static_cast<const ASN1_STRING*>(GENERAL_NAME_get0_value(sk_GENERAL_NAME_value(gens, i), &type));
            const char* t = type == GEN_DNS ? "dns" : type == GEN_URI ? "uri" : type == GEN_EMAIL ? "email" : type == GEN_IPADD ? "ip" : nullptr;
            if (!t || !v) continue;
            const unsigned char* d = ASN1_STRING_get0_data(v); int n = ASN1_STRING_length(v);
            if (!d || n <= 0) continue;
            std::string val = type == GEN_IPADD ? sec::ip_text(d, std::size_t(n)) : sec::san_canonical(t, std::string_view(reinterpret_cast<const char*>(d), std::size_t(n)));
            if (!val.empty()) out.push_back(std::string(t) + ":" + val);
        }
        if (gens) GENERAL_NAMES_free(gens);
        X509_free(c);
#endif
        return out;
    }

#ifdef FORGE_HAVE_TLS
    // Server side: wrap an accepted socket. `ctx` is shared by all connections.
    bool accept_tls(std::shared_ptr<SSL_CTX> ctx, std::string* why) {
        ctx_ = std::move(ctx); ssl_.reset(SSL_new(ctx_.get()));
        if (!ssl_ || SSL_set_fd(ssl_.get(), int(s_.native())) != 1 || SSL_accept(ssl_.get()) != 1) {
            if (why) {   // name the verification failure (certificate revoked, CRL has expired, ...) when there is one
                *why = "tls accept: " + sec::ssl_error();
                if (long v = ssl_ ? SSL_get_verify_result(ssl_.get()) : long(X509_V_OK); v != X509_V_OK) *why += std::string(" (client certificate: ") + X509_verify_cert_error_string(v) + ")";
            }
            ERR_clear_error(); ssl_.reset(); return false;
        }
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
        if (c.cert_file.empty() != c.key_file.empty()) return fail("client certificate and key go together");
        if (!c.cert_file.empty()) {   // mTLS: present a client certificate
            if (SSL_CTX_use_certificate_chain_file(ctx_.get(), c.cert_file.c_str()) != 1) return fail("cannot load client certificate " + c.cert_file);
            if (SSL_CTX_use_PrivateKey_file(ctx_.get(), c.key_file.c_str(), SSL_FILETYPE_PEM) != 1 || SSL_CTX_check_private_key(ctx_.get()) != 1)
                return fail("cannot load client key " + c.key_file);
        }
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
    if (!t.client_ca_file.empty()) {   // mTLS: demand a client certificate that chains to this CA
        if (SSL_CTX_load_verify_locations(c.get(), t.client_ca_file.c_str(), nullptr) != 1) raise(errc::io_error, "rpc tls: cannot load client CA {}", t.client_ca_file);
        if (auto* names = SSL_load_client_CA_file(t.client_ca_file.c_str())) SSL_CTX_set_client_CA_list(c.get(), names);
        SSL_CTX_set_verify(c.get(), SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, nullptr);
    }
    if (!t.crl_files.empty()) {   // revocation: only the CRLs of each file are loaded (never certificates it may also hold)
        if (t.client_ca_file.empty()) raise(errc::invalid_argument, "rpc tls: a CRL needs client certificates (mTLS: a client CA)");
        X509_STORE* store = SSL_CTX_get_cert_store(c.get());
        for (auto& f : t.crl_files) {
            BIO* b = BIO_new_file(f.c_str(), "r"); if (!b) { ERR_clear_error(); raise(errc::io_error, "rpc tls: cannot read CRL {}", f); }
            int n = 0; bool ok = true;
            while (X509_CRL* crl = PEM_read_bio_X509_CRL(b, nullptr, nullptr, nullptr)) { ok = X509_STORE_add_crl(store, crl) == 1; X509_CRL_free(crl); if (!ok) break; ++n; }
            BIO_free(b); ERR_clear_error();   // the read loop ends on "no start line"
            if (!ok || n == 0) raise(errc::io_error, "rpc tls: {} {}", ok ? "no PEM CRL in" : "cannot load a CRL of", f);
        }
        X509_STORE_set_flags(store, X509_V_FLAG_CRL_CHECK | X509_V_FLAG_CRL_CHECK_ALL);
    }
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

// Client half of the handshake. No-op when the client has no credentials. Over TLS it binds to the channel (3/4).
inline bool client_auth(channel& c, const client_security& s, std::string* why) {
    if (!s.any_auth()) return true;
    bool bound = c.tls(); auto cb = c.channel_binding();
    if (bound && cb.empty()) { if (why) *why = "rpc auth: cannot derive the TLS channel binding"; return false; }
    auto sent = [&](bool ok) { if (!ok && why) *why = "rpc auth: send failed"; return ok; };
    if (!sent(c.send_all("FRPH", 4))) return false;
    if (s.key) {
        if (!sent(sec::put_u32(c, bound ? 4 : 2) && c.send_all(s.key->pub.data(), 32))) return false;
        std::uint8_t nonce[32]; if (!c.recv_all(nonce, 32)) { if (why) *why = "rpc auth: server closed (key not authorized?)"; return false; }
        auto m = bound ? sec::auth_message_bound(nonce, s.key->pub.data(), cb) : sec::auth_message(nonce, s.key->pub.data());
        std::uint8_t sk[64], pub[32], seed[32], sig[64]; std::memcpy(seed, s.key->seed.data(), 32);
        crypto_ed25519_key_pair(sk, pub, seed); crypto_ed25519_sign(sig, sk, m.data(), m.size()); crypto_wipe(sk, 64);
        if (!sent(c.send_all(sig, 64))) return false;
    } else if (bound) {
        auto mac = sec::token_mac(s.token, cb);
        if (!sent(sec::put_u32(c, 3) && c.send_all(mac.data(), mac.size()))) return false;
    } else if (!sent(sec::put_u32(c, 1) && sec::put_str(c, s.token))) return false;
    return sec::status(c, why);
}

// The peer of one connection: whether it proved an identity, who (for logs: "token#2 (ci)", "key:d75a9801 (alice)",
// "cert:cn=forge-client", or both credential and certificate), and what it may do. handshake = an FRPH hello was
// consumed (the frame that came with it is done).
struct peer_info { bool authed = false; unsigned perms = perm_all; std::string who; bool handshake = false; limits lim; };

// mTLS client map lookup: the rule for this connection's client certificate (fingerprint rules first, then
// subjectAltName rules in map order, then CN), or null. `who` names the certificate either way.
inline const cert_rule* match_client_cert(const channel& c, const std::vector<cert_rule>& map, std::string& who) {
    auto fp = c.peer_cert_sha256(), cn = c.peer_cert_cn();
    who = fp.empty() ? "cert:none" : "cert:sha256:" + fp.substr(0, 16) + (cn.empty() ? "" : " cn=" + cn);
    if (fp.empty()) return nullptr;
    for (auto& r : map) if (r.kind == cert_rule::sha256 && sec::ct_equal(r.value, fp)) { who = "cert:sha256:" + fp.substr(0, 16) + (r.comment.empty() ? "" : " (" + r.comment + ")"); return &r; }
    std::vector<std::string> sans; bool any_san_rule = false;
    for (auto& r : map) any_san_rule = any_san_rule || r.is_san();
    if (any_san_rule) sans = c.peer_cert_sans();
    for (auto& r : map) if (r.is_san()) for (auto& e : sans) if (r.matches_san(e)) { who = "cert:san=" + e + (r.comment.empty() ? "" : " (" + r.comment + ")"); return &r; }
    if (!cn.empty()) for (auto& r : map) if (r.kind == cert_rule::cn && r.value == cn) { who = "cert:cn=" + cn + (r.comment.empty() ? "" : " (" + r.comment + ")"); return &r; }
    return nullptr;
}

// Server half. `first4` is the frame magic already read. Returns true when the peer may proceed; `peer` gets its
// identity and permissions (an unauthenticated peer of a no-auth server: authed = false, perms = all). On refusal
// `why` (if given) gets a description for the server log.
// With a client map (mTLS): the certificate must match a rule. Its permissions are intersected with those of the
// token or key; when the server has no tokens or keys, the mapped certificate alone authenticates, for device ops
// only (remote scripts always need a token or key).
// `revoked`: the certificate deny-list in force (null = s.revoked_certs; rpc::server passes its reloaded list). A
// client certificate on it is refused, with or without a client map, after the credential exchange.
inline bool server_auth(channel& c, const server_security& s, const char first4[4], peer_info& peer, std::string* why = nullptr,
                        const std::vector<std::string>* revoked = nullptr) {
    peer = {};
    const auto& deny = revoked ? *revoked : s.revoked_certs;
    bool is_revoked = false; std::string revoked_fp;
    if (!deny.empty()) { revoked_fp = c.peer_cert_sha256(); if (!revoked_fp.empty()) for (auto& d : deny) if (sec::ct_equal(d, revoked_fp)) is_revoked = true; }
    auto refuse_revoked = [&](const std::string& extra) { sec::reject(c, "permission denied: client certificate revoked"); if (why) *why = extra + "cert:sha256:" + revoked_fp.substr(0, 16) + " is on the revoked list"; return false; };
    auto refuse = [&](std::string_view to_client, std::string log = {}) {
        sec::reject(c, to_client); if (why) *why = log.empty() ? std::string(to_client) : std::move(log); return false; };
    const cert_rule* cert = nullptr; std::string cert_who;
    bool mapped = !s.client_map.empty();
    if (mapped) cert = match_client_cert(c, s.client_map, cert_who);
    if (std::memcmp(first4, "FRPH", 4) != 0) {
        if (is_revoked) return refuse_revoked("");
        if (mapped && !cert) return refuse("permission denied: client certificate not authorized", cert_who + " is not in the client map");
        if (s.has_credentials()) return refuse("authentication required");
        if (cert) { peer.authed = true; peer.perms = cert->perms & perm_device; peer.who = cert_who; peer.lim = cert->lim; }
        return true;
    }
    peer.handshake = true;
    std::uint32_t method; if (!sec::get_u32(c, method)) return false;
    bool tls = c.tls(); auto cb = c.channel_binding();
    if (tls && cb.empty()) return refuse("cannot derive the TLS channel binding");
    if (tls && (method == 1 || method == 2)) return refuse("TLS connections need channel-bound authentication (client too old?)");
    if (!tls && (method == 3 || method == 4)) return refuse("channel-bound authentication needs TLS");
    if (method == 1 || method == 3) {
        const token_entry* hit = nullptr; std::size_t idx = 0;
        if (method == 1) {
            std::string tok; if (!sec::get_str(c, tok, 4096)) return false;
            for (std::size_t i = 0; i < s.tokens.size(); ++i) if (sec::ct_equal(s.tokens[i].token, tok)) { hit = &s.tokens[i]; idx = i; }
        } else {
            std::uint8_t mac[64]; if (!c.recv_all(mac, 64)) return false;
            for (std::size_t i = 0; i < s.tokens.size(); ++i) { auto want = sec::token_mac(s.tokens[i].token, cb); if (sec::ct_equal_bytes(want.data(), mac, 64)) { hit = &s.tokens[i]; idx = i; } }
        }
        if (!hit) return refuse("bad token");
        peer.who = "token#" + std::to_string(idx + 1) + (hit->name.empty() ? "" : " (" + hit->name + ")"); peer.perms = hit->perms; peer.lim = hit->lim;
    } else if (method == 2 || method == 4) {
        std::uint8_t pub[32]; if (!c.recv_all(pub, 32)) return false;
        auto ph = sec::hex(pub, 32); const authorized_key* known = nullptr;
        for (auto& k : s.authorized_keys) if (sec::ct_equal(k.pub, ph)) known = &k;
        std::uint8_t nonce[32]; if (!os::entropy(nonce, 32)) return false;
        if (!c.send_all(nonce, 32)) return false;
        std::uint8_t sig[64]; if (!c.recv_all(sig, 64)) return false;
        auto m = method == 4 ? sec::auth_message_bound(nonce, pub, cb) : sec::auth_message(nonce, pub);
        if (!known || crypto_ed25519_check(sig, pub, m.data(), m.size()) != 0) return refuse("key not authorized", "key not authorized: " + ph.substr(0, 16));
        peer.who = "key:" + ph.substr(0, 8) + (known->comment.empty() ? "" : " (" + known->comment + ")"); peer.perms = known->perms; peer.lim = known->lim;
    } else return refuse("unknown auth method");
    if (is_revoked) return refuse_revoked(peer.who + " with ");
    if (mapped) {
        if (!cert) return refuse("permission denied: client certificate not authorized", peer.who + " with " + cert_who + " not in the client map");
        peer.perms &= cert->perms; peer.lim.intersect(cert->lim); peer.who += " " + cert_who;
    }
    peer.authed = true;
    return sec::put_u32(c, 0);
}
} // inline namespace
} // namespace forge::graph::rpc
