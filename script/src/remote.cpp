// forge::remote(addr, source[, opts]) -> {ok, output, error_type, error}
// Runs a FORGE script on a `forge rpc-serve --allow-scripts` node (docs/rpc.md). The connection is authenticated
// (token or Ed25519 key) and optionally TLS; credentials come from opts or the FORGE_RPC_* environment.
//   opts: {"args", list} {"timeout_ms", n} {"token", s} {"key", secret-key-file} {"tls", bool} {"ca", file}
//         {"fingerprint", sha256-hex} {"server_name", s} {"name", script-name}
#include "interp.hpp"
#include <forge/flow/rpc.hpp>

namespace forge::script {
namespace {
using Args = std::vector<Value>;
Value strv(std::string s) { return obj(make<String>(std::move(s))); }
const Value* kv(const Args& a, std::size_t i, const std::string& k) {
    if (i >= a.size()) return nullptr; const Value* r = nullptr;
    if (auto* m = as<Map>(a[i])) r = m->find_str(k);
    else if (auto* l = as<List>(a[i])) for (auto& e : l->items) if (auto* p = as<List>(e); p && p->items.size() == 2) if (auto* s = as<String>(p->items[0]); s && s->s == k) r = &p->items[1];
    return r && !is_nil(*r) ? r : nullptr;
}
} // namespace

void install_remote(Shared& sh) {
    auto n = make<Native>("forge::remote", [](Interp& in, Args& a) -> Value {
        if (!in.caps.net) in.raise("PermissionError", "network access is disabled in this sandbox");
        std::string addr = in.str(a[0]), src = in.str(a[1]);
        if (addr.starts_with("rpc:")) addr = addr.substr(4);
        auto colon = addr.rfind(':'); if (colon == std::string::npos) in.raise("ValueError", "forge::remote: want host:port");
        auto host = addr.substr(0, colon), port = addr.substr(colon + 1);
        if (host.size() > 1 && host.front() == '[') host = host.substr(1, host.size() - 2);
        flow::rpc::client_security sec;
        try { sec = flow::rpc::client::env_security(); } catch (const std::exception& e) { in.raise("ValueError", std::string("forge::remote: ") + e.what()); }
        if (auto* v = kv(a, 2, "token")) sec.token = in.str(*v);
        try { if (auto* v = kv(a, 2, "key")) sec.key = std::make_shared<flow::rpc::identity>(flow::rpc::identity_from_seed_hex(flow::rpc::read_key_file(in.str(*v)))); }
        catch (const std::exception& e) { in.raise("ValueError", std::string("forge::remote: ") + e.what()); }
        if (auto* v = kv(a, 2, "ca")) { sec.tls.ca_file = in.str(*v); sec.tls.enabled = true; }
        if (auto* v = kv(a, 2, "fingerprint")) { sec.tls.fingerprint_sha256 = in.str(*v); sec.tls.enabled = true; }
        if (auto* v = kv(a, 2, "server_name")) sec.tls.server_name = in.str(*v);
        if (auto* v = kv(a, 2, "tls")) sec.tls.enabled = Interp::truthy(*v);
        std::vector<std::string> argv; if (auto* v = kv(a, 2, "args")) if (auto* l = as<List>(*v)) for (auto& x : l->items) argv.push_back(in.str(x));
        std::uint32_t tmo = 0; if (auto* v = kv(a, 2, "timeout_ms")) if (auto* i = std::get_if<std::int64_t>(v)) tmo = std::uint32_t(std::max<std::int64_t>(0, *i));
        std::string name = "<remote>"; if (auto* v = kv(a, 2, "name")) name = in.str(*v);
        std::string why;
        auto c = flow::rpc::client::connect(host, port, &why, sec);
        if (!c) in.raise(why.find("auth") != std::string::npos ? "PermissionError" : "IOError", why);
        flow::rpc::client::script_reply r;
        try { r = c->run_script(src, name, argv, tmo); }
        catch (const std::exception& e) { std::string m = e.what(); in.raise(m.find("disabled") != std::string::npos ? "PermissionError" : "IOError", m); }
        auto m = make<Map>(); m->set_str("ok", r.ok); m->set_str("output", strv(r.output)); m->set_str("error_type", strv(r.error_type)); m->set_str("error", strv(r.error));
        return obj(m);
    }, 2, 3);
    n->frozen = true; sh.builtins["forge::remote"] = obj(n);
}
} // namespace forge::script
