// forge::script AOT runtime (docs/aot.md): what an AOT binary runs before and around the script.
//   - the payload (script source, bundled modules, baked policy): encoding, the embed-mode trailer;
//   - binding natively compiled function bodies to the parsed program;
//   - the runner: run-time options that can only narrow the baked policy, the security policy, then the same
//     forge::script::engine the `forge script` command uses;
//   - the out-of-line helpers of rt/forge_aot_rt.hpp.
#include <forge/script/aot.hpp>
#include "aot.hpp"
#include "../rt/forge_aot_rt.hpp"
#include <forge/hash/blake3.hpp>
#include <forge/os/os.hpp>
#include <forge/security/security.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>

namespace forge::script {
namespace aot {

// ---------------------------------------------------------------- payload
namespace {
constexpr char magic_head[8] = {'F', 'O', 'R', 'G', 'E', 'A', 'O', 'T'};
constexpr char magic_tail[8] = {'T', 'O', 'A', 'E', 'G', 'R', 'O', 'F'};
constexpr std::uint32_t payload_version = 1;
constexpr std::size_t max_payload = 256u << 20;

struct W {
    std::vector<std::uint8_t> b;
    void u8(std::uint8_t v) { b.push_back(v); }
    void u32(std::uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(std::uint8_t(v >> (8 * i))); }
    void u64(std::uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back(std::uint8_t(v >> (8 * i))); }
    void str(const std::string& s) { u64(s.size()); b.insert(b.end(), s.begin(), s.end()); }
};
struct R {
    const std::uint8_t* p; std::size_t n, i = 0; bool ok = true;
    std::uint8_t u8() { if (i + 1 > n) { ok = false; return 0; } return p[i++]; }
    std::uint32_t u32() { if (n - i < 4 || i > n) { ok = false; return 0; } std::uint32_t v = 0; for (int k = 0; k < 4; ++k) v |= std::uint32_t(p[i + std::size_t(k)]) << (8 * k); i += 4; return v; }
    std::uint64_t u64() { if (n - i < 8 || i > n) { ok = false; return 0; } std::uint64_t v = 0; for (int k = 0; k < 8; ++k) v |= std::uint64_t(p[i + std::size_t(k)]) << (8 * k); i += 8; return v; }
    std::string str() { std::uint64_t l = u64(); if (!ok || l > n - i) { ok = false; return {}; } std::string s(reinterpret_cast<const char*>(p + i), std::size_t(l)); i += std::size_t(l); return s; }
};
}

std::vector<std::uint8_t> encode_payload(const payload& p) {
    W w;
    w.u32(payload_version); w.u32(abi_version);
    w.str(p.name); w.str(p.source);
    w.u64(p.modules.size());
    for (auto& m : p.modules) { w.str(m.importer); w.str(m.path); w.str(m.key); w.str(m.source); }
    auto& b = p.baked;
    w.u8(std::uint8_t((b.allow_fs ? 1 : 0) | (b.allow_net ? 2 : 0) | (b.allow_proc ? 4 : 0) | (b.allow_env ? 8 : 0)));
    w.u64(b.max_steps); w.u64(b.timeout_ms); w.u64(b.heap_bytes); w.u32(std::uint32_t(b.max_depth));
    w.u8(b.os_sandbox);
    w.u8(std::uint8_t((b.static_check ? 1 : 0) | (b.strict_check ? 2 : 0) | (b.debug ? 4 : 0) | (b.allow_widen ? 8 : 0)));
    w.str(b.policy_json); w.str(b.policy_origin);
    w.u64(b.mem_limit_bytes);
    return std::move(w.b);
}

bool decode_payload(const std::uint8_t* data, std::size_t size, payload& out, std::string* why) {
    auto bad = [&](const char* m) { if (why) *why = m; return false; };
    R r{data, size};
    if (r.u32() != payload_version) return bad("unknown payload version");
    if (r.u32() != abi_version) return bad("payload was written for another runtime version");
    out.name = r.str(); out.source = r.str();
    std::uint64_t nm = r.u64();
    if (!r.ok || nm > size) return bad("truncated payload");
    for (std::uint64_t i = 0; i < nm && r.ok; ++i) { payload::module m; m.importer = r.str(); m.path = r.str(); m.key = r.str(); m.source = r.str(); out.modules.push_back(std::move(m)); }
    auto& b = out.baked;
    std::uint8_t caps = r.u8();
    b.allow_fs = caps & 1; b.allow_net = caps & 2; b.allow_proc = caps & 4; b.allow_env = caps & 8;
    b.max_steps = r.u64(); b.timeout_ms = r.u64(); b.heap_bytes = r.u64(); b.max_depth = std::int32_t(r.u32());
    b.os_sandbox = r.u8();
    std::uint8_t fl = r.u8();
    b.static_check = fl & 1; b.strict_check = fl & 2; b.debug = fl & 4; b.allow_widen = fl & 8;
    b.policy_json = r.str(); b.policy_origin = r.str();
    b.mem_limit_bytes = r.u64();
    if (!r.ok) return bad("truncated payload");
    if (r.i != size) return bad("trailing bytes in payload");
    if (b.os_sandbox > 2) return bad("bad OS sandbox mode in payload");
    return true;
}

std::vector<std::uint8_t> make_trailer(const std::vector<std::uint8_t>& pb) {
    W w;
    w.b.insert(w.b.end(), magic_head, magic_head + 8);
    w.u32(payload_version); w.u64(pb.size());
    auto d = hash::blake3::of(std::span<const std::uint8_t>(pb));
    w.b.insert(w.b.end(), d.begin(), d.end());
    w.b.insert(w.b.end(), magic_tail, magic_tail + 8);
    return std::move(w.b);
}

int read_embedded(const std::string& exe, std::vector<std::uint8_t>& out, std::string* why) {
    std::ifstream f(exe, std::ios::binary);
    if (!f) return 0;
    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    if (size < std::streamoff(trailer_size)) return 0;
    std::uint8_t t[trailer_size];
    f.seekg(size - std::streamoff(trailer_size));
    if (!f.read(reinterpret_cast<char*>(t), std::streamsize(trailer_size))) return 0;
    if (std::memcmp(t, magic_head, 8) != 0 || std::memcmp(t + trailer_size - 8, magic_tail, 8) != 0) return 0;
    auto bad = [&](const char* m) { if (why) *why = m; return -1; };
    R r{t + 8, 12};
    if (r.u32() != payload_version) return bad("embedded program has an unknown format version");
    const std::uint64_t len = r.u64();
    if (len > max_payload || std::streamoff(len) > size - std::streamoff(trailer_size)) return bad("embedded program has a bad length");
    out.resize(std::size_t(len));
    f.seekg(size - std::streamoff(trailer_size) - std::streamoff(len));
    if (len && !f.read(reinterpret_cast<char*>(out.data()), std::streamsize(len))) return bad("embedded program could not be read");
    auto d = hash::blake3::of(std::span<const std::uint8_t>(out));
    if (std::memcmp(d.data(), t + 20, 32) != 0) return bad("embedded program is damaged (checksum mismatch)");
    return 1;
}

// ---------------------------------------------------------------- binding
namespace {
struct State {
    std::mutex m;
    const unit* units = nullptr; std::size_t n_units = 0;
    std::vector<char> bound;
    std::deque<std::vector<const Node*>> tables;   // node tables stay put: generated code holds pointers into them
    payload pl;
    bool active = false;
};
State& state() { static State s; return s; }
}

void bind_program(const std::shared_ptr<const FnDecl>& top, const std::string& name) {
    State& s = state();
    if (!s.active || !s.n_units || !top) return;
    std::lock_guard g(s.m);
    for (std::size_t i = 0; i < s.n_units; ++i) {
        const unit& u = s.units[i];
        if (s.bound[i] || name != u.name) continue;
        program_index ix = index_program(*top);
        if (ix.nodes.size() != u.n_nodes || ix.fns.size() != u.n_fns || ix.fingerprint != u.fingerprint) {
            // The main program must match: the code was generated from this very source. Anything else that
            // happens to share a unit's name (a file on disk shadowing a bundled module) just runs interpreted.
            if (i != 0) return;
            std::fprintf(stderr, "%s: the compiled code does not match the embedded program (AOT runtime library and code generator are from different builds)\n", name.c_str());
            std::fflush(stderr);
            std::_Exit(70);
        }
        s.tables.push_back(std::move(ix.nodes));
        *u.nodes = s.tables.back().data();
        for (std::size_t k = 0; k < ix.fns.size(); ++k) ix.fns[k]->aot_body = reinterpret_cast<void (*)()>(u.fns[k]);
        s.bound[i] = 1;
        return;
    }
}

bool bundled_module(const Interp& in, const std::string& path, std::string& key, std::string& source) {
    State& s = state();
    if (!s.active || s.pl.modules.empty()) return false;
    const payload::module* hit = nullptr;
    for (auto& m : s.pl.modules) if (m.path == path && m.importer == in.file) { hit = &m; break; }
    if (!hit) for (auto& m : s.pl.modules) if (m.path == path) { hit = &m; break; }   // imported at run time from a function of another unit
    if (!hit) return false;
    key = hit->key;
    for (auto& m : s.pl.modules) if (m.key == key && !m.source.empty()) { source = m.source; return true; }
    return false;
}

// ---------------------------------------------------------------- runner
namespace {

bool truthy_env(const char* name) { auto v = os::getenv(name); return !v.empty() && v != "0"; }
std::uint64_t num_arg(const std::string& what, const std::string& v) {
    try { std::size_t used = 0; auto n = std::stoull(v, &used); if (used == v.size() && !v.empty() && v[0] != '-') return n; } catch (...) {}
    throw std::runtime_error(what + " needs a non-negative integer, got '" + v + "'");
}
// A limit can only be tightened: 0 means unlimited, so a baked limit stays unless the request is smaller.
std::uint64_t narrow(std::uint64_t baked, std::uint64_t asked, bool widen) {
    if (widen) return asked;
    if (!baked) return asked;
    if (!asked) return baked;
    return std::min(baked, asked);
}
int sandbox_mode(const std::string& m) {
    if (m == "off") return 0; if (m == "best-effort") return 1; if (m == "require") return 2;
    throw std::runtime_error("--forge-os-sandbox takes off|best-effort|require");
}

void print_info(const payload& pl, std::size_t n_units, const unit* units) {
    auto& b = pl.baked;
    std::printf("forge script AOT binary (%s mode, runtime ABI %u)\n", n_units ? "native" : "embed", unsigned(abi_version));
    std::printf("program: %s (%zu bytes of source, %zu bundled module(s))\n", pl.name.c_str(), pl.source.size(), pl.modules.size());
    if (n_units) { std::size_t fns = 0; for (std::size_t i = 0; i < n_units; ++i) fns += units[i].n_fns; std::printf("native functions: %zu in %zu unit(s)\n", fns, n_units); }
    std::printf("capabilities: fs=%s net=%s proc=%s env=%s\n", b.allow_fs ? "on" : "off", b.allow_net ? "on" : "off", b.allow_proc ? "on" : "off", b.allow_env ? "on" : "off");
    std::printf("limits: timeout_ms=%llu max_steps=%llu heap_bytes=%llu mem_limit_bytes=%llu max_depth=%d\n", static_cast<unsigned long long>(b.timeout_ms), static_cast<unsigned long long>(b.max_steps), static_cast<unsigned long long>(b.heap_bytes), static_cast<unsigned long long>(b.mem_limit_bytes), int(b.max_depth));
    std::printf("os sandbox: %s\n", b.os_sandbox == 0 ? "off" : b.os_sandbox == 1 ? "best-effort" : "require");
    std::printf("security policy: %s\n", b.policy_json.empty() ? "none baked (the usual policy search applies)" : ("baked from " + b.policy_origin).c_str());
    std::printf("run-time widening: %s\n", b.allow_widen ? "allowed" : "not allowed");
}

// FSS for an AOT binary: what apps/forge.cpp script_policy() does for `forge script`, with two differences. A
// policy baked in at compile time is the policy (the host's policy search is not consulted, and --forge-policy is
// refused unless the binary allows widening). And the artefact integrity.scripts checks is the executable itself:
// its detached signature `<exe>.sig` (forge sign <exe> --key k.key) covers the program and every bundled module.
void apply_policy(options& o, const baked_policy& b, const std::string& cli_path, const std::string& what) {
    security::policy p;
    if (!b.policy_json.empty() && !(b.allow_widen && !cli_path.empty())) {
        if (!cli_path.empty()) throw std::runtime_error("--forge-policy: this binary carries a baked security policy (" + b.policy_origin + ") and was not compiled with --aot-allow-widen");
        p = security::parse_policy(b.policy_json, "baked:" + b.policy_origin);
        p.source = "baked:" + b.policy_origin;
    } else p = security::load_policy(cli_path);
    if (!p.active()) return;
    if (p.integrity.scripts != security::requirement::off) {
        const std::string exe = os::exe_path();
        auto bytes = os::read_file(exe);
        if (exe.empty() || !bytes) throw security::integrity_error(what + ": integrity.scripts is set in " + p.source + ", and this executable could not be read to verify its signature", "INT-4");
        security::verify_artefact(p, security::artefact::script, exe, {reinterpret_cast<const std::uint8_t*>(bytes.value.data()), bytes.value.size()});
    }
    if (!o.allow_fs) { p.fs.read.clear(); p.fs.write.clear(); }
    if (!o.allow_net) p.net.mode = security::net_mode::none;
    if (!o.allow_proc) p.exec = false;
    if (!o.allow_env) p.env = false;
    o.allow_fs = !(p.fs.read.empty() && p.fs.write.empty());
    o.allow_net = p.net.mode != security::net_mode::none;
    o.allow_proc = p.exec;
    o.allow_env = p.env;
    o.os_sandbox = options::os_sandbox_mode::off;   // lockdown() applies the OS confinement
    security::audit("script", security::decision::allow, "CAP-1", what,
                    std::string("AOT binary runs with fs=") + (o.allow_fs ? "scoped" : "off") + " net=" + security::name(p.net.mode) +
                    " proc=" + (o.allow_proc ? "on" : "off") + " env=" + (o.allow_env ? "on" : "off") + " (profile " + security::name(p.prof) + ")");
    security::lockdown(p);
}

int run_payload(int argc, char** argv, payload&& pl_in, const unit* units, std::size_t n_units) {
    if (run_process_worker_if_requested(argc, argv)) return 0;   // ProcessPool spawn-mode worker
    State& s = state();
    { std::lock_guard g(s.m); s.pl = std::move(pl_in); s.units = units; s.n_units = n_units; s.bound.assign(n_units, 0); s.active = true; }
    const payload& pl = s.pl;
    const baked_policy& b = pl.baked;
    const std::string what = pl.name;
    try {
        options o;
        o.allow_fs = b.allow_fs; o.allow_net = b.allow_net; o.allow_proc = b.allow_proc; o.allow_env = b.allow_env;
        o.max_steps = b.max_steps; o.timeout_ms = b.timeout_ms; o.heap_bytes = b.heap_bytes; o.max_depth = b.max_depth;
        o.mem_limit_bytes = b.mem_limit_bytes;
        o.static_check = b.static_check; o.strict_check = b.strict_check; o.debug = b.debug;
        int sandbox = b.os_sandbox;
        std::string policy_path;
        const bool widen = b.allow_widen;
        auto need_widen = [&](const std::string& opt) { if (!widen) throw std::runtime_error(opt + " would widen the policy baked into this binary, which was not compiled with --aot-allow-widen"); };
        auto set_sandbox = [&](const std::string& opt, const std::string& v) { int m = sandbox_mode(v); if (m < sandbox) need_widen(opt); sandbox = m; };

        // Environment first, then leading --forge-* arguments; both can only narrow (unless the binary allows widening).
        if (truthy_env("FORGE_AOT_NO_FS")) o.allow_fs = false;
        if (truthy_env("FORGE_AOT_NO_NET")) o.allow_net = false;
        if (truthy_env("FORGE_AOT_NO_PROC")) o.allow_proc = false;
        if (truthy_env("FORGE_AOT_NO_ENV")) o.allow_env = false;
        if (auto v = os::getenv("FORGE_AOT_TIMEOUT"); !v.empty()) o.timeout_ms = narrow(o.timeout_ms, num_arg("FORGE_AOT_TIMEOUT", v), widen);
        if (auto v = os::getenv("FORGE_AOT_MAX_STEPS"); !v.empty()) o.max_steps = narrow(o.max_steps, num_arg("FORGE_AOT_MAX_STEPS", v), widen);
        if (auto v = os::getenv("FORGE_AOT_HEAP"); !v.empty()) o.heap_bytes = narrow(o.heap_bytes, num_arg("FORGE_AOT_HEAP", v), widen);
        auto size_arg = [](const std::string& what, const std::string& v) { std::uint64_t n = 0; std::string why; if (!parse_size(v, n, &why) || !n) throw std::runtime_error(what + ": " + (why.empty() ? "the limit must be greater than 0" : why)); return n; };
        if (auto v = os::getenv("FORGE_AOT_MEM_LIMIT"); !v.empty()) o.mem_limit_bytes = narrow(o.mem_limit_bytes, size_arg("FORGE_AOT_MEM_LIMIT", v), widen);
        if (auto v = os::getenv("FORGE_AOT_OS_SANDBOX"); !v.empty()) set_sandbox("FORGE_AOT_OS_SANDBOX", v);
        int i = 1;
        for (; i < argc; ++i) {
            std::string a = argv[i];
            if (!a.starts_with("--forge-")) break;
            std::string k = a, v; bool has_v = false;
            if (auto eq = a.find('='); eq != std::string::npos) { k = a.substr(0, eq); v = a.substr(eq + 1); has_v = true; }
            auto val = [&]() -> std::string { if (has_v) return v; if (i + 1 >= argc) throw std::runtime_error("missing value for " + k); return argv[++i]; };
            if (k == "--forge-args") { ++i; break; }
            else if (k == "--forge-aot-info") { print_info(pl, n_units, units); return 0; }
            else if (k == "--forge-no-fs") o.allow_fs = false;
            else if (k == "--forge-no-net") o.allow_net = false;
            else if (k == "--forge-no-proc") o.allow_proc = false;
            else if (k == "--forge-no-env") o.allow_env = false;
            else if (k == "--forge-allow-fs") { need_widen(k); o.allow_fs = true; }
            else if (k == "--forge-allow-net") { need_widen(k); o.allow_net = true; }
            else if (k == "--forge-allow-proc") { need_widen(k); o.allow_proc = true; }
            else if (k == "--forge-allow-env") { need_widen(k); o.allow_env = true; }
            else if (k == "--forge-timeout") o.timeout_ms = narrow(o.timeout_ms, num_arg(k, val()), widen);
            else if (k == "--forge-max-steps") o.max_steps = narrow(o.max_steps, num_arg(k, val()), widen);
            else if (k == "--forge-heap") o.heap_bytes = narrow(o.heap_bytes, num_arg(k, val()), widen);
            else if (k == "--forge-mem-limit") o.mem_limit_bytes = narrow(o.mem_limit_bytes, size_arg(k, val()), widen);
            else if (k == "--forge-os-sandbox") set_sandbox(k, val());
            else if (k == "--forge-policy") policy_path = val();
            else if (k == "--forge-debug") o.debug = true;
            else if (k == "--forge-strict-check") o.strict_check = true;
            else if (k == "--forge-no-check") { need_widen(k); o.static_check = false; }
            else throw std::runtime_error("unknown run-time option " + k + " (script arguments that start with --forge- go after --forge-args)");
        }
        for (; i < argc; ++i) o.argv.push_back(argv[i]);
        o.os_sandbox = options::os_sandbox_mode(sandbox);
        // AI-drafted code (forge::ai_run / ai_eval) runs only after an explicit "y" on the terminal, as in `forge script`.
        o.confirm = [](const std::string& code) {
            std::fprintf(stderr, "\nAI drafted:\n%s\nRun it? [y/N] ", code.c_str()); std::fflush(stderr);
            char buf[16] = {}; if (!std::fgets(buf, sizeof buf, stdin)) return false;
            return buf[0] == 'y' || buf[0] == 'Y';
        };
        try { apply_policy(o, b, policy_path, what); }
        catch (const security::integrity_error& e) { std::fprintf(stderr, "IntegrityError: %s\n", e.what()); return 1; }
        engine e(o);
        auto r = e.run(pl.source, pl.name);
        std::fflush(stdout);
        if (!r.ok) std::fprintf(stderr, "%s\n", r.error.c_str());
        return r.exit_code;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", what.c_str(), e.what());
        return 2;
    }
}

} // namespace

int run_native(int argc, char** argv, const unsigned char* payload_bytes, std::size_t payload_size, const unit* units, std::size_t n_units, std::uint32_t abi) {
    if (abi != abi_version) { std::fprintf(stderr, "this program was compiled for AOT runtime ABI %u, but is linked against ABI %u\n", unsigned(abi), unsigned(abi_version)); return 70; }
    payload pl; std::string why;
    if (!decode_payload(payload_bytes, payload_size, pl, &why)) { std::fprintf(stderr, "bad embedded program: %s\n", why.c_str()); return 70; }
    return run_payload(argc, argv, std::move(pl), units, n_units);
}

} // namespace aot

std::optional<int> aot_run_embedded(int argc, char** argv) {
    std::string exe = os::exe_path();
    if (exe.empty()) return std::nullopt;
    std::vector<std::uint8_t> bytes; std::string why;
    int st = aot::read_embedded(exe, bytes, &why);
    if (st == 0) return std::nullopt;
    if (st < 0) { std::fprintf(stderr, "%s: %s\n", exe.c_str(), why.c_str()); return 70; }
    aot::payload pl;
    if (!aot::decode_payload(bytes.data(), bytes.size(), pl, &why)) { std::fprintf(stderr, "%s: bad embedded program: %s\n", exe.c_str(), why.c_str()); return 70; }
    return aot::run_payload(argc, argv, std::move(pl), nullptr, 0);
}

// ---------------------------------------------------------------- helpers of rt/forge_aot_rt.hpp
namespace aot_rt {

void int_error(Interp& in, Op op, I64 x, I64 y) {
    if (op == Op::None) in.unary("-", Value(x)); else in.binary(op, Value(x), Value(y));
    in.raise("OverflowError", "integer overflow");   // not reached: the calls above raise
}

I64 need_int(Interp& in, const Value& v, const Node& let) {
    if (auto* i = std::get_if<I64>(&v)) return *i;
    in.raise("TypeError", "let " + let.s + " expects int, got " + Interp::type_of(v));
}

// tree.cpp, Interp::exec, case NK::Throw
void throw_value(Interp& in, Value v, int line) {
    if (auto* i = as<Instance>(v); i && i->cls->derives("Error")) {   // a thrown string stays a string, as in tree.cpp
        auto user_field = [&](const char* f) {   // a user exception's own `trace` / `line` member is the user's
            for (const Class* c = i->cls.get(); c; c = c->base.get()) {
                if (auto it = in.sh->errors.find(c->name); it != in.sh->errors.end() && it->second.get() == c) return false;
                for (auto& x : c->fields) if (x == f) return true;
            }
            return false;
        };
        if (auto it = i->fields.find("trace"); !user_field("trace") && (it == i->fields.end() || is_nil(it->second) || (as<String>(it->second) && as<String>(it->second)->s.empty())))
            i->fields["trace"] = in.new_string(in.trace_text());
        if (!user_field("line")) i->fields["line"] = std::int64_t(line);
    }
    throw ScriptError(v, in.error_message(v));
}

// tree.cpp, Interp::run_try: which catch clause takes this error
// The same rule as the interpreter, through the same function: a clause handles only its own type
// (Interp::catch_matches), and a fatal error (whole-run limit, std::exit, cancellation, a limit block's timeout
// inside the block) is handled by no clause, catch (...) included.
int match_catch(Interp& in, const ScriptError& e, const Node& n) {
    if (e.fatal || in.sh->dead.load()) return 0;
    for (std::size_t i = 1; i < n.kids.size(); ++i) if (in.catch_matches(n.kids[i]->s, e.v)) return int(i);
    return 0;
}

void let_check(Interp& in, const Value& v, const Node& let) { if (let.type) in.check_type(v, *let.type, "let " + let.s); }

bool same_binding(const Value& x, const Value& y) {
    auto* p = std::get_if<Ref<Obj>>(&x); auto* q = std::get_if<Ref<Obj>>(&y);
    return (p && q) ? p->get() == q->get() : (!p && !q && x == y);
}

void moved_error(Interp& in, const Node& var) { in.raise("BorrowError", "use of moved-from '" + var.var.name + "'"); }

void bad_flow(Interp& in) { in.raise("SyntaxError", "break/continue outside a loop"); }

} // namespace aot_rt
} // namespace forge::script
