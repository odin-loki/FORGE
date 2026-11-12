#pragma once
// RPC device: run lowerings on a remote FORGE process over TCP ("rpc:host:port").
// The server (`forge rpc-serve`, or rpc::serve()) executes each op with its own device registry (CPU, Vulkan, CUDA...),
// so a machine without a GPU can borrow one over the network. Every op is still oracle-gated on the client.
// Wire format (little endian): request  = "FRPC" u32 op-name-len name u32 nattrs {key, tag, value} u32 nin {u32 rank, i64 dims, f32 data} u32 rank i64 out-dims
//                              response = u32 status(0 ok) then f32 out data, or u32 len + error text.
// Remote weight residency (protocol version 2; docs/rpc.md "Weights on the server"): a plan's weights and constants are
// uploaded once per connection and referred to by the BLAKE3 hash of their bytes afterwards.
//   probe   = an FRPC frame for the op "rpc.caps" (no inputs, rank-0 output)
//             reply: u32 0, u32 version, u32 flags (bit 0: weights), u64 weight budget in bytes.
//             A version 1 server answers it like any unknown op (status 1 + text) and stays usable: no residency.
//   upload  = "FRPW" hash[32] u32 rank i64 dims f32 data
//             reply: u32 status: 0 stored; 1 refused + str (permission, max-bytes, hash mismatch); 2 no room + str
//   op      = "FRPR": as FRPC, but each input starts with u32 kind: 0 = inline (rank, dims, data), 1 = by handle
//             (hash[32], rank, dims: the shape the op sees)
//             reply: as FRPC, or u32 3, u32 n, n x hash[32]: handles the server does not hold (evicted, or a new
//             connection); the client uploads them again and repeats the request.
// The server keeps the uploads of one connection in an LRU cache under a byte budget shared by all connections
// (server::weight_budget_bytes, $FORGE_RPC_WEIGHT_CACHE); they are gone when the connection ends. Every handle,
// shape and size from the wire is checked; an upload is hashed by the server before it is stored.
// Security (rpc_secure.hpp, docs/rpc.md): optional TLS + token / Ed25519 key auth before the first frame; per-rule
// permissions and limits are checked on every request (a refused request gets "permission denied: ..." and the
// connection stays usable); client certificates can be revoked by CRL or by a fingerprint deny-list.
// Remote scripts:  request  = "FRPS" str source, str name, u32 argc {str}, u32 timeout_ms
//                  or "FRPZ" (signed) with the same fields then str signature (the script's detached .sig text, FSS
//                  INT-2; servers older than FRPZ close the connection)
//                  response = u32 status(0 ok / 1 refused + str error) then u32 ok, str error_type, str error, str output
#include <forge/graph/kernels.hpp>
#include <forge/graph/rpc_secure.hpp>
#include <forge/hash/blake3.hpp>
#include <array>
#include <deque>
#include <list>
#include <map>
#include <functional>
#include <cstring>
#include <string>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <forge/os/os.hpp>

namespace forge::graph::rpc {
#ifdef FORGE_HAVE_TLS
inline namespace tls_v1 {
#else
inline namespace plain_v1 {
#endif
namespace detail {
inline void wsa() { os::net_init(); }
inline bool send_all(channel& s, const void* p, std::size_t n) { return s.send_all(p, n); }
inline bool recv_all(channel& s, void* p, std::size_t n) { return s.recv_all(p, n); }
struct writer {
    std::vector<char> b;
    void raw(const void* p, std::size_t n) { auto* c = static_cast<const char*>(p); b.insert(b.end(), c, c + n); }
    void u32(std::uint32_t v) { raw(&v, 4); }
    void i64(std::int64_t v) { raw(&v, 8); }
    void str(std::string_view s) { u32(std::uint32_t(s.size())); raw(s.data(), s.size()); }
    void shape(const shape_t& s) { u32(std::uint32_t(s.size())); for (auto d : s) i64(d); }
};
struct reader {
    channel& s; bool ok = true;
    void raw(void* p, std::size_t n) { if (ok) ok = recv_all(s, p, n); }
    std::uint32_t u32() { std::uint32_t v = 0; raw(&v, 4); return v; }
    std::int64_t i64() { std::int64_t v = 0; raw(&v, 8); return v; }
    std::string str(std::size_t cap = 1 << 20) { auto n = u32(); if (n > cap) { ok = false; return {}; } std::string r(n, '\0'); raw(r.data(), n); return r; }
    shape_t shape() { auto r = u32(); shape_t s; if (r > 16) { ok = false; return s; } for (std::uint32_t i = 0; i < r; ++i) s.push_back(i64()); return s; }
};
inline void put_attrs(writer& w, const ops::attrs& a) {
    w.u32(std::uint32_t(a.v.size()));
    for (auto& [k, v] : a.v) {
        w.str(k); w.u32(std::uint32_t(v.index()));
        if (auto* d = std::get_if<double>(&v)) w.raw(d, 8);
        else if (auto* iv = std::get_if<std::vector<i64>>(&v)) { w.u32(std::uint32_t(iv->size())); for (auto x : *iv) w.i64(x); }
        else w.str(std::get<std::string>(v));
    }
}
inline ops::attrs get_attrs(reader& r) {
    ops::attrs a; auto n = r.u32(); if (n > 4096) { r.ok = false; return a; }
    for (std::uint32_t i = 0; i < n && r.ok; ++i) {
        auto k = r.str(); auto tag = r.u32();
        if (tag == 0) {   // ops cast number attrs to i64: no NaN, infinity or out-of-range value from the wire
            double d = 0; r.raw(&d, 8); if (!(d >= -0x1p53 && d <= 0x1p53)) { r.ok = false; break; }
            a.v[k] = d;
        }
        else if (tag == 1) { auto m = r.u32(); if (m > (1u << 20)) { r.ok = false; break; } std::vector<i64> iv(m); for (auto& x : iv) x = r.i64(); a.v[k] = iv; }
        else if (tag == 2) a.v[k] = r.str();
        else r.ok = false;
    }
    return a;
}
inline bool valid_shape(const shape_t& s, i64 cap = i64(1) << 32) { i64 n = 1; for (auto d : s) { if (d < 0 || d > cap) return false; n *= std::max<i64>(d, 1); if (n > cap) return false; } return true; }
inline int op_index(std::string_view name) {
    for (std::size_t i = 0; i < ops::op_count; ++i) if (ops::info(ops::op(i)).name == name) return int(i);
    return -1;
}
// A frame is run only if it is a well-formed call of the op: the declared number of inputs, operands the op's
// shape rule accepts, and an output of the size that rule gives. Lowerings index their inputs and write the
// output without checks of their own (the planner guarantees them locally; a peer does not).
inline void check_call(ops::op o, const std::vector<shape_t>& in, const shape_t& out, const ops::attrs& a) {
    const auto& inf = ops::info(o);
    if (inf.shape == ops::shape_rule::concat ? in.size() < 2 : in.size() != std::size_t(inf.inputs))
        raise(errc::invalid_argument, "{} takes {} inputs, got {}", inf.name, inf.inputs, in.size());
    for (auto& s : in) if (numel(s) == 0) raise(errc::shape_mismatch, "{}: empty input", inf.name);
    auto want = ops::infer_shape(inf.shape, in, a);
    if (!valid_shape(want) || numel(want) != numel(out))
        raise(errc::shape_mismatch, "{}: output shape {} does not fit the inputs (expected {})", inf.name, shape_str(out), shape_str(want));
    auto rank = [&](std::size_t i) { return in[i].size(); };
    bool ok = true;   // operands of the ops whose shape rule only says "same as input 0"
    if (inf.name == "rms_norm") ok = rank(0) >= 1 && rank(1) == 1 && in[1][0] == in[0].back();
    else if (inf.name == "layer_norm") ok = rank(0) >= 1 && rank(1) == 1 && rank(2) == 1 && in[1][0] == in[0].back() && in[2][0] == in[0].back();
    else if (inf.name == "softmax" || inf.name == "moe_gate") ok = rank(0) >= 1;
    else if (inf.name == "rope") ok = rank(0) == 3 && in[0][2] % 2 == 0 && numel(in[1]) == in[0][0];
    else if (inf.name == "moe_ffn") ok = rank(0) == 2 && rank(1) == 2 && rank(2) == 3 && rank(4) == 3 && in[3] == in[2] && in[1][0] == in[0][0] &&
                                         in[1][1] == in[2][0] && in[2][2] == in[0][1] && in[4] == shape_t{in[2][0], in[0][1], in[2][1]};
    if (!ok) raise(errc::shape_mismatch, "{}: operand shapes do not fit together", inf.name);
}
// Read and drop n bytes (the payload of a request that is refused anyway), without buffering them.
inline void skip(reader& r, std::uint64_t n) {
    char buf[1 << 15];
    while (n && r.ok) { auto k = std::size_t(std::min<std::uint64_t>(n, sizeof buf)); r.raw(buf, k); n -= k; }
}
inline std::string_view op_class_name(ops::op_class c) {
    static constexpr std::string_view names[] = {"elementwise", "reduction", "contraction", "data_movement", "stateful"};
    return std::size_t(c) < std::size(names) ? names[std::size_t(c)] : std::string_view("unknown");
}
inline constexpr std::string_view caps_op = "rpc.caps";   // not an op: the capability probe of protocol version 2
} // namespace detail

// Connection to one server; serialises calls (one op in flight).
class client {
public:
    static std::shared_ptr<client> connect(const std::string& host, const std::string& port, std::string* why, const client_security& sec = env_security()) {
        detail::wsa();
        auto c = std::shared_ptr<client>(new client); c->host_ = host; c->port_ = port; c->sec_ = sec;
        if (const char* e = std::getenv("FORGE_RPC_RESIDENT_MIN"); e && *e) c->min_bytes_ = std::strtoull(e, nullptr, 10);
        return c->open_(why) ? c : nullptr;
    }
    // Credentials from the environment: FORGE_RPC_KEY (secret key file), FORGE_RPC_TOKEN, FORGE_RPC_TLS=1,
    // FORGE_RPC_CA, FORGE_RPC_FINGERPRINT (sha256 of the server cert), FORGE_RPC_SERVER_NAME,
    // FORGE_RPC_TLS_CERT + FORGE_RPC_TLS_KEY (client certificate for mTLS servers).
    static client_security env_security() {
        client_security s; auto env = [](const char* k) { const char* v = std::getenv(k); return std::string(v ? v : ""); };
        if (auto k = env("FORGE_RPC_KEY"); !k.empty()) s.key = std::make_shared<identity>(identity_from_seed_hex(read_key_file(k)));
        s.token = env("FORGE_RPC_TOKEN");
        s.tls.ca_file = env("FORGE_RPC_CA"); s.tls.fingerprint_sha256 = env("FORGE_RPC_FINGERPRINT"); s.tls.server_name = env("FORGE_RPC_SERVER_NAME");
        s.tls.cert_file = env("FORGE_RPC_TLS_CERT"); s.tls.key_file = env("FORGE_RPC_TLS_KEY");
        s.tls.enabled = env("FORGE_RPC_TLS") == "1" || !s.tls.ca_file.empty() || !s.tls.fingerprint_sha256.empty() || !s.tls.cert_file.empty();
        return s;
    }
    struct script_reply { bool ok = false; std::string error_type, error, output; };
    // Run a FORGE script on the server (needs an authenticated connection to a server started with scripts allowed).
    // `signature`: the script's detached signature (`forge sign script.fs` -> script.fs.sig text; `name` must be the
    // signed file name), checked by servers whose policy has integrity.scripts.
    script_reply run_script(std::string_view source, std::string_view name = "<remote>", const std::vector<std::string>& argv = {}, std::uint32_t timeout_ms = 0,
                            std::string_view signature = {}) {
        std::lock_guard g(m_);
        detail::writer w; w.raw(signature.empty() ? "FRPS" : "FRPZ", 4); w.str(source); w.str(name); w.u32(std::uint32_t(argv.size())); for (auto& a : argv) w.str(a); w.u32(timeout_ms);
        if (!signature.empty()) w.str(signature);
        if (!detail::send_all(s_, w.b.data(), w.b.size())) raise(errc::io_error, "rpc: send failed");
        detail::reader r{s_};
        auto st = r.u32(); if (!r.ok) raise(errc::io_error, "rpc: connection lost");
        if (st != 0) raise(errc::unsupported, "rpc server: {}", r.str());
        script_reply out; out.ok = r.u32() != 0; out.error_type = r.str(); out.error = r.str(); out.output = r.str(64u << 20);
        if (!r.ok) raise(errc::io_error, "rpc: short reply");
        return out;
    }
    std::uint64_t calls() const { return calls_; }
    // ---- remote weight residency ----
    // A weight or constant of a plan (registry::weights reports them): from now on an op input that is exactly this
    // buffer goes to the server once per connection and by hash afterwards. forget_weight: the buffer is going away.
    void note_weight(const float* p, std::size_t n) {
        std::lock_guard g(m_); if (!p || !n) return;
        auto& w = weights_[reinterpret_cast<std::uintptr_t>(p)]; if (w.n != n) { w = wrec{}; w.n = n; }
    }
    void forget_weight(const float* p, std::size_t n) {
        std::lock_guard g(m_);
        const auto lo = reinterpret_cast<std::uintptr_t>(p), hi = lo + std::max<std::size_t>(n, 1) * 4;
        for (auto it = weights_.begin(); it != weights_.end();) { if (it->first < hi && lo < it->first + it->second.n * 4) it = weights_.erase(it); else ++it; }
    }
    bool resident() const { return resident_; }                       // the server keeps weights (protocol version 2, a budget > 0); known after the first weight was about to be sent
    std::uint64_t weight_budget() const { return budget_; }           // the server's budget, as it reported it
    std::uint64_t uploads() const { return uploads_; }                // weights sent with FRPW
    std::uint64_t upload_bytes() const { return upload_bytes_; }
    std::uint64_t weight_refs() const { return refs_; }               // op inputs sent as a handle
    std::uint64_t weight_misses() const { return misses_; }           // "handle not held" replies (the weight was uploaded again)
    std::uint64_t reconnects() const { return reconnects_; }
    std::uint64_t bytes_sent() const { return sent_; }
    // Drop the connection and open a new one (same address and credentials). The server's copy of every weight is
    // gone with the old connection, so each is uploaded again on its next use.
    bool reconnect(std::string* why = nullptr) { std::lock_guard g(m_); if (!open_(why)) return false; ++reconnects_; return true; }
    // One op. A lost connection is reopened once and the op sent again (ops are pure).
    void run(std::string_view op, std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
        std::lock_guard g(m_);
        for (int attempt = 0;; ++attempt) {
            try { run_(op, in, out, a); ++calls_; return; }
            catch (const lost& l) {
                if (!refused_.empty()) raise(errc::unsupported, "rpc server: {}", refused_);   // the server refused us, then closed
                if (attempt || !open_(nullptr)) raise(errc::io_error, "rpc: {}", l.what);
                ++reconnects_;
            }
        }
    }
private:
    client() = default;
    struct lost { const char* what; };
    struct wrec { std::size_t n = 0; bool hashed = false, uploaded = false, inline_only = false; hash::blake3::digest hash{}; std::uint64_t fp = 0; };
    bool open_(std::string* why) {
        int pn = std::atoi(port_.c_str());
        auto r = os::socket::connect(host_, pn, 10000);
        if (!r) { if (why) *why = "rpc: cannot connect to " + host_ + ":" + port_ + " (" + r.err.what + ")"; return false; }
        r->set_nodelay();
        s_ = channel(std::move(r.value));
        if (sec_.tls.enabled) {
#ifdef FORGE_HAVE_TLS
            if (!s_.connect_tls(sec_.tls, host_, why)) return false;
#else
            if (why) *why = "rpc: TLS requested but this build has no TLS (rebuild with OpenSSL)"; return false;
#endif
        }
        if (!client_auth(s_, sec_, why)) return false;
        for (auto& [k, w] : weights_) w.uploaded = false;   // a new connection: the server holds none of our weights
        probed_ = false; resident_ = false; budget_ = 0;    // asked again when the first weight is about to be sent
        return true;
    }
    bool send_(const void* p, std::size_t n) { sent_ += n; return detail::send_all(s_, p, n); }
    // Ask what the server can do, once per connection and only when an op is about to carry a noted weight (a
    // connection that never has one sends exactly the frames it always did). A version 1 server answers "unknown op":
    // weights then travel with every op. Any other refusal (authentication, permissions) is kept: if the server
    // then closes the connection, the op reports that text instead of a bare "connection lost".
    void probe_() {
        probed_ = true; resident_ = false; budget_ = 0; refused_.clear();
        if (const char* e = std::getenv("FORGE_RPC_RESIDENT"); e && std::string(e) == "0") return;
        detail::writer w; w.raw("FRPC", 4); w.str(detail::caps_op); detail::put_attrs(w, {}); w.u32(0); w.shape({});
        if (!send_(w.b.data(), w.b.size())) throw lost{"send failed"};
        detail::reader r{s_};
        auto st = r.u32(); if (!r.ok) throw lost{"connection lost"};
        if (st != 0) { auto e = r.str(); if (r.ok && e.find("unknown op") == std::string::npos) refused_ = e; return; }
        auto ver = r.u32(), flags = r.u32(); std::uint64_t b = 0; r.raw(&b, 8);
        if (r.ok && ver >= 2 && (flags & 1u) && b) { resident_ = true; budget_ = b; }
    }
    // 64 words spread over the data: notices a buffer that was reused for other contents (a check, not a proof)
    static std::uint64_t fingerprint(const float* p, std::size_t n) {
        std::uint64_t h = 1469598103934665603ull; const std::size_t step = std::max<std::size_t>(1, n / 64);
        auto mix = [&](std::size_t i) { std::uint32_t u; std::memcpy(&u, p + i, 4); h = (h ^ u) * 1099511628211ull; };
        for (std::size_t i = 0; i < n; i += step) mix(i);
        mix(n - 1); return h;
    }
    // The record of this input if it goes by handle: a noted weight of exactly this extent, large enough to be worth
    // it and small enough for the server's budget. The hash is computed on first use.
    wrec* weight_for_(const tensor_view& t) {
        if (!t.data || weights_.empty()) return nullptr;
        auto it = weights_.find(reinterpret_cast<std::uintptr_t>(t.data));
        if (it == weights_.end() || it->second.n != t.size() || it->second.inline_only) return nullptr;
        if (!probed_) probe_();
        if (!resident_) return nullptr;
        auto& w = it->second;
        if (std::uint64_t(w.n) * 4 < min_bytes_ || std::uint64_t(w.n) * 4 > budget_) return nullptr;
        const auto fp = fingerprint(t.data, w.n);
        if (!w.hashed || w.fp != fp) {   // first use, or the buffer holds other data now
            hash::blake3 h; h.update(t.data, w.n * 4); w.hash = h.final(); w.fp = fp; w.hashed = true; w.uploaded = false;
        }
        return &w;
    }
    bool upload_(const tensor_view& t, wrec& w) {   // false: the server refused it; this weight travels inline from now on
        detail::writer hd; hd.raw("FRPW", 4); hd.raw(w.hash.data(), 32); hd.shape(t.shape);
        if (!send_(hd.b.data(), hd.b.size()) || !send_(t.data, w.n * 4)) throw lost{"send failed"};
        detail::reader r{s_}; auto st = r.u32(); if (!r.ok) throw lost{"connection lost"};
        if (st == 0) { w.uploaded = true; ++uploads_; upload_bytes_ += std::uint64_t(w.n) * 4; return true; }
        (void)r.str(); if (!r.ok) throw lost{"connection lost"};
        w.inline_only = true; return false;
    }
    void run_(std::string_view op, std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
        std::vector<wrec*> ref(in.size(), nullptr); bool any = false;
        for (std::size_t i = 0; i < in.size(); ++i) if ((ref[i] = weight_for_(in[i]))) any = true;
        for (int round = 0; any && round < 3; ++round) {   // a handle the server evicted is uploaded again, at most twice
            for (std::size_t i = 0; i < in.size(); ++i) if (ref[i] && !ref[i]->uploaded && !upload_(in[i], *ref[i])) ref[i] = nullptr;
            std::uint64_t nref = 0; for (auto* w : ref) nref += w != nullptr;
            if (!nref) break;
            detail::writer w; w.raw("FRPR", 4); w.str(op); detail::put_attrs(w, a);
            w.u32(std::uint32_t(in.size()));
            for (std::size_t i = 0; i < in.size(); ++i) {
                if (ref[i]) { w.u32(1); w.raw(ref[i]->hash.data(), 32); w.shape(in[i].shape); }
                else { w.u32(0); w.shape(in[i].shape); w.raw(in[i].data, in[i].size() * 4); }
            }
            w.shape(out.shape);
            if (!send_(w.b.data(), w.b.size())) throw lost{"send failed"};
            detail::reader r{s_};
            auto st = r.u32(); if (!r.ok) throw lost{"connection lost"};
            if (st == 0) { r.raw(out.data, out.size() * 4); if (!r.ok) throw lost{"short reply"}; refs_ += nref; return; }
            if (st != 3) raise(errc::unsupported, "rpc server: {}", r.str());
            auto n = r.u32(); if (!r.ok || n > in.size()) throw lost{"malformed reply"};
            for (std::uint32_t k = 0; k < n; ++k) {
                hash::blake3::digest h{}; r.raw(h.data(), 32);
                for (auto& [key, wr] : weights_) if (wr.hashed && wr.hash == h) wr.uploaded = false;
            }
            if (!r.ok) throw lost{"connection lost"};
            ++misses_;
        }
        detail::writer w; w.raw("FRPC", 4); w.str(op); detail::put_attrs(w, a);
        w.u32(std::uint32_t(in.size()));
        for (auto& t : in) { w.shape(t.shape); w.raw(t.data, t.size() * 4); }
        w.shape(out.shape);
        if (!send_(w.b.data(), w.b.size())) throw lost{"send failed"};
        detail::reader r{s_};
        auto st = r.u32();
        if (!r.ok) throw lost{"connection lost"};
        if (st != 0) raise(errc::unsupported, "rpc server: {}", r.str());
        r.raw(out.data, out.size() * 4);
        if (!r.ok) throw lost{"short reply"};
    }
    channel s_; std::mutex m_; std::uint64_t calls_ = 0;
    std::string host_, port_; client_security sec_;
    bool resident_ = false, probed_ = false; std::uint64_t budget_ = 0, min_bytes_ = 4096;
    std::string refused_;   // why the server refused the capability probe (not "unknown op"), if it did
    std::map<std::uintptr_t, wrec> weights_;
    std::uint64_t uploads_ = 0, upload_bytes_ = 0, refs_ = 0, misses_ = 0, reconnects_ = 0, sent_ = 0;
};

// Registry whose every op runs remotely ("rpc.<op>", priority 20). Ops the server lacks a lowering for run there on its reference.
inline std::unique_ptr<registry> rpc_registry(std::shared_ptr<client> c) {
    auto r = registry::make();
    for (std::size_t i = 0; i < ops::op_count; ++i) {
        auto o = ops::op(i); std::string name(ops::info(o).name);
        r->add({"rpc." + name, o, 20, nullptr,
                [c, name](std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) { c->run(name, in, out, a); }});
    }
    // the plan's weights and constants: uploaded once per connection, then referred to by hash (remote residency)
    auto ws = std::make_shared<weight_sink>();
    ws->note = [c](const float* p, std::size_t n) { c->note_weight(p, n); };
    ws->forget = [c](const float* p, std::size_t n) { c->forget_weight(p, n); };
    r->weights = ws;
    return r;
}
// "rpc:host:port" (host may be [v6]).
inline std::unique_ptr<registry> open(const std::string& addr, std::string* why) {
    auto hp = addr; auto colon = hp.rfind(':');
    if (colon == std::string::npos) { if (why) *why = "rpc: want rpc:host:port"; return nullptr; }
    auto host = hp.substr(0, colon), port = hp.substr(colon + 1);
    if (host.size() > 1 && host.front() == '[') host = host.substr(1, host.size() - 2);
    auto c = client::connect(host, port, why);
    return c ? rpc_registry(c) : nullptr;
}

// Server. `reg` (null = CPU) picks the lowering per op exactly like a local plan would.
class server {
public:
    // port 0 = ephemeral; see port(). bind = "127.0.0.1" by default (pass "0.0.0.0" to expose; there is no auth).
    struct script_result { bool ok = false; std::string error_type, error, output; };
    // Runs a remote script; set by the host (apps/forge.cpp wires forge::script). Null = scripts unsupported.
    std::function<script_result(const std::string& src, const std::string& name, const std::vector<std::string>& argv, std::uint32_t timeout_ms)> script_handler;
    // Same, with the client's detached signature ("" for an unsigned FRPS request). Preferred over script_handler when set.
    std::function<script_result(const std::string& src, const std::string& name, const std::vector<std::string>& argv, std::uint32_t timeout_ms,
                                const std::string& signature)> signed_script_handler;
    // The whole request: also who sent it and which capabilities its permissions leave the script (scripts.fs,
    // scripts.net, scripts.proc). Preferred over the two above when set, and the only handler that serves peers
    // without the full `scripts` set: the other two cannot withhold a capability, so such peers are refused.
    // timeout_ms already has the peer's script-timeout= limit applied (0 = the client asked for none and has no limit).
    struct script_request {
        std::string source, name; std::vector<std::string> argv; std::uint32_t timeout_ms = 0;
        std::string signature;                  // the client's detached signature ("" for an unsigned FRPS request)
        std::string who;                        // the authenticated principal, as in the security log
        bool allow_fs = true, allow_net = true, allow_proc = true;
    };
    std::function<script_result(const script_request&)> script_request_handler;
    // Security log (optional): authentications with the principal and its permissions, refusals, permission denials.
    // Called from connection threads.
    std::function<void(const std::string&)> log;
    // Largest tensor, in elements, a request may name (inputs and the output); a frame naming a larger one is
    // dropped before anything is allocated for it. Set before start().
    std::uint64_t max_tensor_elems = std::uint64_t(1) << 32;
    // Remote weight residency: bytes of uploaded weights the server keeps, over all connections (each connection
    // evicts its own least recently used weights to make room). 0 = keep none (clients then send weights with every
    // op). Default: $FORGE_RPC_WEIGHT_CACHE (bytes, or with a k, m or g suffix), else 1 GiB. Set before start().
    std::uint64_t weight_budget_bytes = default_weight_budget();
    // 2 = with weight residency (the rpc.caps probe, FRPW, FRPR); 1 = behave like a server from before it: the probe
    // is an unknown op and the new frames close the connection (interoperability tests).
    int protocol_version = 2;
    static std::uint64_t default_weight_budget() {
        std::uint64_t v = 0;
        if (const char* e = std::getenv("FORGE_RPC_WEIGHT_CACHE"); e && *e && sec::parse_size(e, v)) return v;
        return std::uint64_t(1) << 30;
    }
    std::uint64_t weight_uploads() const { return weight_uploads_; }       // uploads stored
    std::uint64_t weight_hits() const { return weight_hits_; }             // op inputs served from a stored weight
    std::uint64_t weight_evictions() const { return weight_evictions_; }
    std::uint64_t weight_bytes() const { std::lock_guard g(wm_); return resident_bytes_; }   // stored now, all connections
    server(registry* reg, int port = 0, const std::string& bind = "127.0.0.1", server_security sec = {}) : reg_(reg ? reg : &registry::cpu()), sec_(std::move(sec)) {
        detail::wsa();
        if (!sec_.tls.client_ca_file.empty() && sec_.tls.cert_file.empty()) raise(errc::invalid_argument, "rpc: client certificates (mTLS) need TLS (a server certificate and key)");
        if (!sec_.tls.cert_file.empty()) {
#ifdef FORGE_HAVE_TLS
            tls_ = make_server_tls(sec_.tls);
#else
            raise(errc::unsupported, "rpc: TLS requested but this build has no TLS (rebuild with OpenSSL)");
#endif
        }
        if (!sec_.client_map.empty() && sec_.tls.client_ca_file.empty()) raise(errc::invalid_argument, "rpc: a client certificate map needs client certificates (mTLS: a client CA)");
        if ((!sec_.tls.crl_files.empty() || !sec_.revoked_certs.empty() || !sec_.revoked_files.empty()) && sec_.tls.client_ca_file.empty())
            raise(errc::invalid_argument, "rpc: certificate revocation (a CRL or a revoked list) needs client certificates (mTLS: a client CA)");
        for (auto& h : sec_.revoked_certs) {   // "AB:CD:..." or "sha256:..." -> lower-case hex; anything else is an error
            auto one = parse_revoked(h, "revoked certificate");
            if (one.size() != 1) raise(errc::invalid_argument, "rpc: revoked certificate: want one 64-hex SHA-256 fingerprint");
            h = one[0];
        }
        // ops= names are checked against the op table, so a typo is an error instead of a rule that allows nothing
        auto check_ops = [](const limits& l, std::string_view where) {
            for (auto& o : l.ops) if (detail::op_index(o) < 0) raise(errc::invalid_argument, "rpc: ops= of {} names an unknown op '{}'", where, o);
        };
        for (std::size_t i = 0; i < sec_.tokens.size(); ++i) check_ops(sec_.tokens[i].lim, "token#" + std::to_string(i + 1));
        for (auto& k : sec_.authorized_keys) check_ops(k.lim, "key " + k.pub.substr(0, 8));
        for (auto& c : sec_.client_map) check_ops(c.lim, "client-map rule " + c.value.substr(0, 16));
        revoked_ = load_revoked(); stamp_ = watch_stamp(); has_tls_ = !sec_.tls.cert_file.empty();
        if (sec_.allow_scripts && !sec_.has_credentials()) raise(errc::invalid_argument, "rpc: remote scripts need authentication (tokens or authorized keys)");
        auto l = os::socket::listen(bind, port, 8);
        if (!l) raise(errc::io_error, "rpc: cannot listen on {}:{}: {}", bind, port, l.err.what);
        ls_ = std::move(l.value); port_ = ls_.local_port();
    }
    ~server() { stop(); }
    int port() const { return port_; }
    std::uint64_t ops() const { return ops_; }
    // Accepts connections until stop(); one thread per connection.
    void run() {
        while (!stop_) {
            // macOS/BSD: shutdown() on a listening socket does not wake a blocked accept() (Linux's does), so
            // wait for a connection in 200 ms slices and re-check stop_.
            if (!ls_.wait_readable(200)) continue;
            auto c = ls_.accept();
            if (!c) { if (stop_) break; continue; }
            auto ch = std::make_shared<channel>(std::move(c.value));
            auto done = std::make_shared<std::atomic<bool>>(false);
            reload_if_changed();   // CRLs / revoked lists edited since the last connection
            conn_env env; env.revoked = revoked_;
#ifdef FORGE_HAVE_TLS
            env.tls = tls_;
#endif
            std::lock_guard g(m_);
            for (std::size_t i = 0; i < conns_.size();)   // reap finished connections
                if (conns_[i].done->load()) { if (conns_[i].th.joinable()) conns_[i].th.join(); conns_.erase(conns_.begin() + std::ptrdiff_t(i)); } else ++i;
            conns_.push_back({ch, done, std::thread([this, ch, done, env] { serve_conn(*ch, env); ch->shutdown_both(); done->store(true); })});
        }
    }
    void start() { th_ = std::thread([this] { run(); }); }
    void stop() {
        if (stop_.exchange(true)) return;
#if defined(_WIN32)
        ls_.close();                            // Winsock: shutdown() does not wake a blocked accept(); closesocket() does
        if (th_.joinable()) th_.join();
#else
        ls_.shutdown_both();                    // wakes accept() on Linux (run() polls elsewhere); close after the thread is gone
        if (th_.joinable()) th_.join();
        ls_.close();
#endif
        std::vector<conn> cs; { std::lock_guard g(m_); for (auto& c : conns_) c.ch->shutdown_both(); cs.swap(conns_); }
        for (auto& c : cs) if (c.th.joinable()) c.th.join();     // workers never outlive the server
    }
private:
    // What one connection is served with: the TLS context and certificate deny-list in force when it was accepted
    // (a reload replaces them for later connections only).
    struct conn_env {
#ifdef FORGE_HAVE_TLS
        std::shared_ptr<SSL_CTX> tls;
#endif
        std::shared_ptr<const std::vector<std::string>> revoked;
    };
    // Deny-list in force: the configured fingerprints plus those of the revoked files.
    std::shared_ptr<const std::vector<std::string>> load_revoked() const {
        auto v = std::make_shared<std::vector<std::string>>(sec_.revoked_certs);
        for (auto& f : sec_.revoked_files) {
            auto r = os::read_file(f); if (!r) raise(errc::io_error, "rpc: cannot read revoked list {}: {}", f, r.err.what);
            auto x = parse_revoked(r.value, f); v->insert(v->end(), x.begin(), x.end());
        }
        return v;
    }
    // Modification time and size of every CRL and revoked file ("" = nothing to watch).
    std::string watch_stamp() const {
        std::string st;
        auto add = [&](const std::string& f) {
            auto m = os::fs::mtime_ns(f); auto z = os::fs::file_size(f);
            st += f + '|' + (m ? std::to_string(m.value) : std::string("?")) + '|' + (z ? std::to_string(z.value) : std::string("?")) + '\n';
        };
        for (auto& f : sec_.tls.crl_files) add(f);
        for (auto& f : sec_.revoked_files) add(f);
        return st;
    }
    // Called from run() before each connection is served: when a CRL or revoked file changed on disk, load them
    // again (the TLS context is rebuilt, so the certificate, key and client CA files are read again too). A failed
    // reload keeps the previous lists and is logged; it is retried when a file changes again.
    void reload_if_changed() {
        if (sec_.tls.crl_files.empty() && sec_.revoked_files.empty()) return;
        auto st = watch_stamp(); if (st == stamp_) return;
        stamp_ = std::move(st);
        try {
            auto rv = load_revoked();
#ifdef FORGE_HAVE_TLS
            if (tls_) tls_ = make_server_tls(sec_.tls);
#endif
            revoked_ = std::move(rv);
            if (log) log("reloaded certificate revocation lists (" + std::to_string(sec_.tls.crl_files.size()) + " CRL file(s), " + std::to_string(revoked_->size()) + " revoked fingerprint(s))");
        } catch (const std::exception& e) { if (log) log(std::string("certificate revocation reload failed, keeping the previous lists: ") + e.what()); }
    }
    void serve_conn(channel& s, const conn_env& env) {
        auto conn_id = ++conn_seq_;
        auto say = [&](const std::string& m) { if (log) log("conn " + std::to_string(conn_id) + ": " + m); };
#ifdef FORGE_HAVE_TLS
        if (env.tls) {
            std::string why; s.sock().set_timeout_ms(15000);
            if (!s.accept_tls(env.tls, &why)) { say("refused: " + why); return; }
            s.sock().set_timeout_ms(0);
        }
#endif
        peer_info peer; bool first = true;
        weight_cache wc;   // this connection's uploaded weights; gone with the connection
        struct wc_guard { server* sv; weight_cache* c; ~wc_guard() { sv->wc_clear(*c); } } wcg{this, &wc};
        for (;;) {
#if defined(_WIN32)
            // Winsock: shutdown() from stop() does not reliably wake a blocked recv(), so idle between requests in slices.
            while (!stop_ && !s.sock().wait_readable(200)) {}
            if (stop_) return;
#endif
            detail::reader r{s};
            char magic[4]; r.raw(magic, 4);
            if (!r.ok) return;
            if (first) {
                first = false;
                std::string why;
                if (!server_auth(s, sec_, magic, peer, &why, env.revoked.get())) { say("refused: " + (why.empty() ? std::string("connection lost during authentication") : why)); return; }
                if (peer.authed) say("authenticated " + peer.who + " allow=" + perms_string(peer.perms) + peer.lim.str());
                else say("unauthenticated (no auth configured)");
                if (peer.handshake) continue;   // handshake consumed; next frame
            }
            if (std::memcmp(magic, "FRPS", 4) == 0 || std::memcmp(magic, "FRPZ", 4) == 0) { if (!serve_script(s, r, peer, say, magic[3] == 'Z')) return; continue; }
            if (protocol_version >= 2 && std::memcmp(magic, "FRPW", 4) == 0) { if (!serve_upload(s, r, peer, say, wc)) return; continue; }
            const bool refs = protocol_version >= 2 && std::memcmp(magic, "FRPR", 4) == 0;   // inputs may be weight handles
            if (!refs && std::memcmp(magic, "FRPC", 4) != 0) return;
            auto name = r.str(256); auto a = detail::get_attrs(r);
            auto n = r.u32(); if (!r.ok || n > 64) return;
            if (protocol_version >= 2 && !refs && n == 0 && name == detail::caps_op) {   // the capability probe
                auto os = r.shape(); if (!r.ok) return;
                detail::writer w; w.u32(0); w.u32(2); w.u32(weight_budget_bytes ? 1u : 0u); w.raw(&weight_budget_bytes, 8);
                if (!detail::send_all(s, w.b.data(), w.b.size())) return;
                continue;
            }
            int oi = detail::op_index(name);
            // Permission checks (per request). What is known before the tensors arrive is decided first, so a refused
            // request is not buffered: its payload is read and dropped, which keeps the connection usable.
            std::string deny;   // why this peer may not run this request ("" = it may)
            if (!(peer.perms & perm_device)) deny = "may not run device ops (allow=)";
            else if (oi >= 0) {
                auto cls = ops::info(ops::op(oi)).cls;
                if (!(peer.perms & perm_for_op_class(unsigned(cls)))) deny = "may not run " + std::string(detail::op_class_name(cls)) + " ops such as " + name + " (allow=device." + std::string(detail::op_class_name(cls)) + ")";
                else if (!peer.lim.op_allowed(name)) deny = "may not run the op " + name + " (ops=)";
            }
            const std::uint64_t cap = peer.lim.max_bytes; std::uint64_t bytes = 0;
            auto over = [&] { return cap && bytes > cap; };
            // A weight handle counts as the bytes of the tensor it names, like an inline input: max-bytes= cannot be
            // sidestepped by uploading first. Views of stored weights stay valid for the op: nothing is uploaded or
            // evicted on this connection while it runs.
            std::deque<tensor> ins; std::vector<tensor_view> views; std::vector<shape_t> shapes;
            std::vector<hash::blake3::digest> missing; std::string bad;
            for (std::uint32_t i = 0; i < n && r.ok; ++i) {
                const std::uint32_t kind = refs ? r.u32() : 0; if (!r.ok || kind > 1) return;
                hash::blake3::digest hsh{}; if (kind == 1) r.raw(hsh.data(), 32);
                auto sh = r.shape(); if (!r.ok || !detail::valid_shape(sh, i64(max_tensor_elems))) return;
                auto nb = std::uint64_t(numel(sh)) * 4; bytes += nb;
                shapes.push_back(sh); views.push_back({nullptr, sh});
                if (kind == 1) {
                    if (!deny.empty() || over()) continue;
                    auto it = wc.by.find(hsh);
                    if (it == wc.by.end()) { missing.push_back(hsh); continue; }
                    if (it->second->data.size() != std::size_t(numel(sh))) {
                        if (bad.empty()) bad = "a weight handle names " + std::to_string(it->second->data.size()) + " elements, the request " + std::to_string(numel(sh));
                        continue; }
                    wc.lru.splice(wc.lru.begin(), wc.lru, it->second);   // most recently used
                    views.back().data = it->second->data.data(); ++weight_hits_;
                    continue;
                }
                if (!deny.empty() || over()) { detail::skip(r, nb); continue; }
                // grow with the bytes that actually arrive: a 40-byte frame may name a 16 GiB tensor
                tensor t; t.shape = sh; const std::size_t want = std::size_t(numel(sh));
                while (r.ok && t.data.size() < want) {
                    const std::size_t at = t.data.size(), step = std::min<std::size_t>(want - at, std::size_t(1) << 22);
                    t.data.resize(at + step); r.raw(t.data.data() + at, step * 4);
                }
                ins.push_back(std::move(t)); views.back().data = ins.back().data.data();
            }
            auto os = r.shape(); if (!r.ok || !detail::valid_shape(os, i64(max_tensor_elems))) return;
            bytes += std::uint64_t(numel(os)) * 4;
            if (deny.empty() && over()) deny = "asked for an op over " + std::to_string(bytes) + " tensor bytes, above its limit (max-bytes=" + std::to_string(cap) + ")";
            detail::writer w;
            try {
                if (!deny.empty()) { say("denied device op " + name + " for " + peer.who + ": " + deny); raise(errc::unsupported, "permission denied: {} {}", peer.who, deny); }
                if (oi < 0) raise(errc::unsupported, "unknown op '{}'", name);
                if (!bad.empty()) raise(errc::invalid_argument, "{}", bad);
                if (!missing.empty()) {   // the client uploads them (again) and repeats the request
                    w.u32(3); w.u32(std::uint32_t(missing.size())); for (auto& h : missing) w.raw(h.data(), 32);
                } else {
                    detail::check_call(ops::op(oi), shapes, os, a);
                    tensor out(os);
                    auto& l = reg_->select(ops::op(oi), shapes, os, a);
                    auto* h = l.on_device ? reg_->hooks.get() : nullptr;
                    std::lock_guard dg(dev_m_);   // device runtimes are single-stream
                    if (h) h->begin_run();
                    try { l.fn(views, {out.data.data(), os}, a); if (h) { h->to_host(out.data.data(), out.data.size()); h->end_run(); } }
                    catch (...) { if (h) h->end_run(); throw; }
                    w.u32(0); w.raw(out.data.data(), out.data.size() * 4); ++ops_;
                }
            } catch (const std::exception& e) { w.b.clear(); w.u32(1); w.str(e.what()); }
            if (!detail::send_all(s, w.b.data(), w.b.size())) return;
        }
    }
    // ---- remote weight residency: one connection's uploads, least recently used last ----
    struct weight_cache {
        struct ent { hash::blake3::digest h; std::vector<float> data; };
        std::list<ent> lru;
        std::map<hash::blake3::digest, std::list<ent>::iterator> by;
    };
    // Stored weights stay on the server's device too: pinned like a local plan's weights, dropped when evicted.
    void wc_pin(const std::vector<float>& d) {
        auto* h = reg_->hooks.get(); if (!h || !h->pin) return;
        std::lock_guard dg(dev_m_); h->pin(d.data(), d.size());
    }
    void wc_evict_last(weight_cache& wc) {   // wm_ held
        auto& e = wc.lru.back();
        if (auto* h = reg_->hooks.get()) { std::lock_guard dg(dev_m_); try { if (h->forget) h->forget(e.data.data(), e.data.size()); else if (h->release) h->release(e.data.data(), e.data.size()); } catch (...) {} }
        resident_bytes_ -= std::uint64_t(e.data.size()) * 4; ++weight_evictions_;
        wc.by.erase(e.h); wc.lru.pop_back();
    }
    // Reserve nb bytes of the budget for this connection, evicting its own oldest weights; false = no room.
    bool wc_reserve(weight_cache& wc, std::uint64_t nb) {
        std::lock_guard g(wm_);
        while (resident_bytes_ + nb > weight_budget_bytes && !wc.lru.empty()) wc_evict_last(wc);
        if (resident_bytes_ + nb > weight_budget_bytes) return false;
        resident_bytes_ += nb; return true;
    }
    void wc_clear(weight_cache& wc) { std::lock_guard g(wm_); while (!wc.lru.empty()) { wc_evict_last(wc); --weight_evictions_; } }
    // "FRPW": hash, shape, data. Nothing the frame says is trusted: the size is checked against the peer's limits and
    // the budget before a byte of payload is buffered, and the data is hashed here before it is stored under the
    // hash, so a handle can only ever name bytes this connection really sent.
    template <class Log> bool serve_upload(channel& s, detail::reader& r, const peer_info& peer, Log& say, weight_cache& wc) {
        hash::blake3::digest claim{}; r.raw(claim.data(), 32);
        auto sh = r.shape(); if (!r.ok || !detail::valid_shape(sh, i64(max_tensor_elems))) return false;
        const std::uint64_t nel = std::uint64_t(numel(sh)), nb = nel * 4;
        std::uint32_t status = 0; std::string err;
        if (!(peer.perms & perm_device)) { status = 1; err = "permission denied: " + peer.who + " may not upload weights (allow= has no device permission)"; }
        else if (peer.lim.max_bytes && nb > peer.lim.max_bytes) {
            status = 1; err = "permission denied: " + peer.who + " sent a weight of " + std::to_string(nb) + " bytes, above its limit (max-bytes=" + std::to_string(peer.lim.max_bytes) + ")"; }
        else if (!nel) { status = 1; err = "an empty weight"; }
        else if (nb > weight_budget_bytes) { status = 2; err = "a weight of " + std::to_string(nb) + " bytes does not fit the server's weight budget (" + std::to_string(weight_budget_bytes) + " bytes)"; }
        if (status) {
            if (status == 1) say("refused weight upload for " + peer.who + ": " + err);
            detail::skip(r, nb); if (!r.ok) return false;
        } else {
            std::vector<float> data;   // grows with the bytes that actually arrive
            while (r.ok && data.size() < nel) {
                const std::size_t at = data.size(), step = std::size_t(std::min<std::uint64_t>(nel - at, std::uint64_t(1) << 22));
                data.resize(at + step); r.raw(data.data() + at, step * 4);
            }
            if (!r.ok) return false;
            hash::blake3 h; h.update(data.data(), std::size_t(nb));
            if (h.final() != claim) { status = 1; err = "the weight's bytes do not have the hash the upload names"; say("refused weight upload for " + peer.who + ": hash mismatch"); }
            else if (auto it = wc.by.find(claim); it != wc.by.end()) wc.lru.splice(wc.lru.begin(), wc.lru, it->second);   // already here
            else if (!wc_reserve(wc, nb)) { status = 2; err = "no room in the server's weight budget"; }
            else {
                wc.lru.push_front({claim, std::move(data)}); wc.by[claim] = wc.lru.begin();
                wc_pin(wc.lru.front().data); ++weight_uploads_;
            }
        }
        detail::writer w; w.u32(status); if (status) w.str(err);
        return detail::send_all(s, w.b.data(), w.b.size());
    }
    template <class Log> bool serve_script(channel& s, detail::reader& r, const peer_info& peer, Log& say, bool with_sig = false) {
        auto src = r.str(16u << 20); auto name = r.str(4096); auto n = r.u32(); if (!r.ok || n > 4096) return false;
        std::vector<std::string> argv; for (std::uint32_t i = 0; i < n && r.ok; ++i) argv.push_back(r.str(1 << 20));
        auto tmo = r.u32(); if (!r.ok) return false;
        std::string sig; if (with_sig) { sig = r.str(64u << 10); if (!r.ok) return false; }
        detail::writer w;
        // scripts.run without scripts.fs / .net / .proc: the script runs, its child without those capabilities
        const bool fs = (peer.perms & perm_scripts_fs) != 0, net = (peer.perms & perm_scripts_net) != 0, proc = (peer.perms & perm_scripts_proc) != 0;
        if (!sec_.allow_scripts || !peer.authed) { w.u32(1); w.str("remote scripts are disabled on this server"); }
        else if (!(peer.perms & perm_scripts_run)) { say("denied script for " + peer.who); w.u32(1); w.str("permission denied: " + peer.who + " may not run scripts (allow=)"); }
        else if (peer.lim.max_bytes && src.size() > peer.lim.max_bytes) {
            say("denied script for " + peer.who + ": " + std::to_string(src.size()) + " source bytes, max-bytes=" + std::to_string(peer.lim.max_bytes));
            w.u32(1); w.str("permission denied: " + peer.who + " sent a script of " + std::to_string(src.size()) + " bytes, above its limit (max-bytes=" + std::to_string(peer.lim.max_bytes) + ")");
        }
        else if (!script_request_handler && !script_handler && !signed_script_handler) { w.u32(1); w.str("this server has no script runtime"); }
        else if (!(fs && net && proc) && !script_request_handler) {   // fail closed: the older handlers cannot withhold a capability
            say("denied script for " + peer.who + ": the script runtime cannot restrict capabilities");
            w.u32(1); w.str("permission denied: " + peer.who + " has no full scripts permission and this server's script runtime cannot restrict capabilities");
        }
        else {
            if (auto cap = peer.lim.script_timeout_ms) tmo = tmo ? std::min(tmo, cap) : cap;
            script_result res; std::string without = std::string(fs ? "" : " fs") + (net ? "" : " net") + (proc ? "" : " proc");
            say("script '" + name + "' for " + peer.who + (sig.empty() ? "" : " (signed)") + (without.empty() ? "" : " without" + without));
            try {
                if (script_request_handler) { script_request q; q.source = std::move(src); q.name = name; q.argv = std::move(argv); q.timeout_ms = tmo; q.signature = sig; q.who = peer.who; q.allow_fs = fs; q.allow_net = net; q.allow_proc = proc; res = script_request_handler(q); }
                else res = signed_script_handler ? signed_script_handler(src, name, argv, tmo, sig) : script_handler(src, name, argv, tmo);
            } catch (const std::exception& e) { res.ok = false; res.error_type = "RuntimeError"; res.error = e.what(); }
            w.u32(0); w.u32(res.ok ? 1 : 0); w.str(res.error_type); w.str(res.error); w.str(res.output); ++scripts_;
        }
        return detail::send_all(s, w.b.data(), w.b.size());
    }
public:
    std::uint64_t scripts() const { return scripts_; }
    bool tls() const { return has_tls_; }
private:
    registry* reg_; server_security sec_; os::socket ls_; int port_ = 0; std::atomic<std::uint64_t> scripts_{0};
    // tls_, revoked_ and stamp_ belong to the constructor and then to the run() thread (reload_if_changed);
    // connection threads get their own references (conn_env).
#ifdef FORGE_HAVE_TLS
    std::shared_ptr<SSL_CTX> tls_;
#endif
    bool has_tls_ = false; std::shared_ptr<const std::vector<std::string>> revoked_; std::string stamp_;
    std::atomic<bool> stop_{false}; std::atomic<std::uint64_t> ops_{0}, conn_seq_{0};
    mutable std::mutex wm_; std::uint64_t resident_bytes_ = 0;   // remote weight residency: bytes stored over all connections (wm_)
    std::atomic<std::uint64_t> weight_uploads_{0}, weight_hits_{0}, weight_evictions_{0};
    std::thread th_; std::mutex m_, dev_m_; struct conn { std::shared_ptr<channel> ch; std::shared_ptr<std::atomic<bool>> done; std::thread th; };
    std::vector<conn> conns_;
};
} // inline namespace
} // namespace forge::graph::rpc
