// forge::security wired into FORGE (docs/security-api.md "Integration"): the script runtime's scope hook
// (enforce_scope: fs paths, net hosts, exec, env, plugins), the SecurityError script classes, model integrity at
// load (INT-1 / INT-4) and the no-policy path. The policy is made active with set_active_policy for the in-process
// cases; the `lockdown` child (this binary re-run with FORGE_SCRIPT_SECURITY_CHILD) applies a real lockdown and runs
// a script under it. The forge / forge-serve / rpc-serve command lines are covered by tests/security_cli_test.py.
#include <forge/script/script.hpp>
#include <forge/script/check.hpp>
#include <forge/security/security.hpp>
#include <forge/inference/inference.hpp>
#include <forge/modelfile/sign.hpp>
#include <forge/os/os.hpp>
#include <forge/os/process.hpp>
#include "check.hpp"
#include "tiny_model.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
using namespace forge;
namespace sec = forge::security;
namespace fs = std::filesystem;
TEST_PROCESS_WORKER(script::run_process_worker_if_requested);   // sandbox_escape.fs starts a ProcessPool

namespace {
std::string scratch(const char* tag) {
    auto d = os::join(os::temp_dir(), std::string("forge_ssec_") + tag + "_" + std::to_string(os::pid()));
    os::remove_all(d); os::mkdirs(d);
    return os::absolute(d);
}
std::string slurp(const fs::path& p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }
// "!<ErrorType>" appended on failure, like the golden tests
std::string run(const std::string& src, script::options o = {}, bool vm = true) {
    std::string out; o.use_vm = vm; o.jit_threshold = 1; o.timeout_ms = 300000;
    o.out = [&](std::string_view s) { out += s; };
    script::engine e(o);
    auto r = e.run(src, "<sec>");
    if (!r.ok) out += "!" + r.error_type + (r.error_type == "PermissionError" || r.error_type == "IntegrityError" ? "" : ": " + r.error);
    return out;
}
struct active {   // a policy active for the scope of one test; profile none again afterwards
    explicit active(sec::policy p) { sec::set_active_policy(std::move(p)); }
    ~active() { sec::set_active_policy(sec::profile_defaults(sec::profile::none)); }
};
struct capture {
    std::mutex m; std::vector<sec::audit_record> recs; int id = 0;
    capture() { id = sec::add_audit_sink([this](const sec::audit_record& r) { std::lock_guard g(m); recs.push_back(r); }); }
    ~capture() { sec::remove_audit_sink(id); }
    bool has(std::string_view cat, sec::decision d, std::string_view target_part, std::string_view control = {}) {
        std::lock_guard g(m);
        for (auto& r : recs) if (r.category == cat && r.d == d && r.target.find(target_part) != std::string::npos && (control.empty() || r.control == control)) return true;
        return false;
    }
};
std::string quoted(std::string s) { std::string o = "\""; for (char c : s) { if (c == '\\' || c == '"') o += '\\'; o += c; } return o + "\""; }
sec::policy confined(const std::string& root) {   // strict, with fs confined to `root` (reads and writes)
    auto p = sec::profile_defaults(sec::profile::strict);
    p.fs.read = {root}; p.fs.write = {root};
    p.audit.os = false;
    return p;
}
} // namespace

// ---------------------------------------------------------------- child: a real lockdown, then a script
static int child_main(const std::string& dir) {
    auto p = confined(dir);
    p.integrity = {sec::requirement::off, sec::requirement::off, sec::requirement::off, {}, false};
    p.allow_root = os::is_elevated();   // CI containers may run as root
    try { sec::lockdown(p); } catch (const std::exception& e) { std::printf("lockdown threw: %s\n", e.what()); return 2; }
    script::options o; o.allow_net = false; o.allow_proc = false; o.allow_env = false; o.os_sandbox = script::options::os_sandbox_mode::off;
    auto outside = os::join(os::temp_dir(), "forge_ssec_outside_" + std::to_string(os::pid()));
    const std::string src = "fs.write(" + quoted(os::join(dir, "in.txt")) + ", \"x\"); std::println(\"{}\", fs.read(" + quoted(os::join(dir, "in.txt")) + "));\n"
                            "try { fs.write(" + quoted(outside) + ", \"y\"); std::println(\"wrote\"); } catch (const PermissionError& e) { std::println(\"denied\"); }\n";
    auto got = run(src, o);
    std::printf("%s", got.c_str());
    if (got != "x\ndenied\n") std::printf("--- source\n%s", src.c_str());
    if (os::exists(outside)) { os::remove_all(outside); return 3; }
    return got == "x\ndenied\n" ? 0 : 4;
}
// Run as a test (not from a static initializer: the script runtime's own statics must be initialised first). The
// parent re-runs this binary with FORGE_TEST_ONLY=zz_lockdown_child and FORGE_SCRIPT_SECURITY_CHILD=<dir>; in a
// normal run the variable is unset and this does nothing.
TEST(zz_lockdown_child) {
    auto d = os::getenv("FORGE_SCRIPT_SECURITY_CHILD"); if (d.empty()) return;
    os::unsetenv("FORGE_SCRIPT_SECURITY_CHILD");
    std::fflush(stdout);
    std::exit(child_main(d));
}

// ---------------------------------------------------------------- no policy: unchanged
TEST(no_policy_is_profile_none) {
    // the test runs without a policy file or FORGE_SECURITY_* in the environment (ctest's working directory has none)
    CHECK(!sec::active_policy()->active());
    capture cap;
    auto d = scratch("none");
    auto got = run("fs.write(" + quoted(os::join(d, "a.txt")) + ", \"1\"); std::println(\"{}\", fs.read(" + quoted(os::join(d, "a.txt")) + "));");
    CHECK_MSG(got == "1\n", "%s", got.c_str());
    CHECK(cap.recs.empty());   // profile none audits nothing
    os::remove_all(d);
}

TEST(goldens_same_with_permissive_policy) {
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator("tests/script")) if (e.path().extension() == ".fs") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    CHECK(files.size() >= 8);
    auto p = sec::profile_defaults(sec::profile::permissive); p.audit.os = false;
    for (auto& f : files) {
        auto src = slurp(f); auto name = f.filename().string();
        auto runf = [&](bool vm) {
            std::string out; script::options o; o.use_vm = vm; o.jit_threshold = 1; o.timeout_ms = 300000;
            o.out = [&](std::string_view s) { out += s; };
            script::engine e(o); auto r = e.run(src, name);
            if (!r.ok) out += "!" + r.error_type + "\n";
            return out;
        };
        auto plain = runf(true);
        std::string with;
        { active a(p); with = runf(true); }
        auto want = slurp(fs::path(f).replace_extension(".out")); std::erase(want, '\r');
        CHECK_MSG(plain == want, "%s: no-policy output differs from the golden", name.c_str());
        CHECK_MSG(with == plain, "%s: output differs under a permissive policy\n--- none\n%s--- permissive\n%s", name.c_str(), plain.c_str(), with.c_str());
    }
}

// ---------------------------------------------------------------- strict: path- and host-scoped
TEST(strict_policy_scopes_fs) {
    auto root = scratch("root"), other = scratch("other");
    capture cap;
    active a(confined(root));
    auto in = os::join(root, "in.txt"), out = os::join(other, "out.txt");
    for (bool vm : {false, true}) {
        CHECK_MSG(run("fs.write(" + quoted(in) + ", \"ok\"); std::println(\"{}\", fs.read(" + quoted(in) + "));", {}, vm) == "ok\n", "write inside the root");
        CHECK(run("fs.write(" + quoted(out) + ", \"no\");", {}, vm) == "!PermissionError");
        CHECK(run("fs.read(" + quoted(os::join(other, "x")) + ");", {}, vm) == "!PermissionError");
        CHECK(run("fs.list(" + quoted(other) + ");", {}, vm) == "!PermissionError");
        CHECK(run("fs.exists(" + quoted(out) + ");", {}, vm) == "!PermissionError");
        CHECK(run("fs.mkdir(" + quoted(os::join(root, "sub")) + "); std::println(\"{}\", fs.is_dir(" + quoted(os::join(root, "sub")) + "));", {}, vm) == "true\n");
        CHECK(run("fs.copy(" + quoted(in) + ", " + quoted(out) + ");", {}, vm) == "!PermissionError");
        CHECK(run("fs.rename(" + quoted(in) + ", " + quoted(out) + ");", {}, vm) == "!PermissionError");
        CHECK(run("auto f = fs.open(" + quoted(out) + ", \"w\");", {}, vm) == "!PermissionError");
        CHECK(run("auto f = fs.open(" + quoted(in) + ", \"r\"); std::println(\"{}\", f.read()); f.close();", {}, vm) == "ok\n");
        // "../" out of the root, and a glob whose literal prefix leaves the root
        CHECK(run("fs.read(" + quoted(os::join(root, "../" + fs::path(other).filename().string() + "/x")) + ");", {}, vm) == "!PermissionError");
        CHECK(run("fs.glob(" + quoted(other + "/*") + ");", {}, vm) == "!PermissionError");
        CHECK(run("try { fs.write(" + quoted(out) + ", \"no\"); } catch (const OSError& e) { std::println(\"oserror\"); }", {}, vm) == "oserror\n");
    }
    CHECK(!os::exists(out));
    CHECK(cap.has("fs", sec::decision::deny, out, "CAP-1"));
    os::remove_all(root); os::remove_all(other);
}

TEST(strict_policy_scopes_net_exec_env) {
    auto root = scratch("net");
    active a(confined(root));
    // net.mode none: connect denied by the policy even when the Caps flag is on (an embedder that keeps caps)
    CHECK(run("net.port_open(\"127.0.0.1\", 9, 100);") == "!PermissionError");
    CHECK(run("net.http(\"http://example.com/\");") == "!PermissionError");
    CHECK(run("net.resolve(\"example.com\");") == "!PermissionError");
    CHECK(run("forge::remote(\"127.0.0.1:9\", \"1;\");") == "!PermissionError");
    CHECK(run("run(\"echo hi\");") == "!PermissionError");
    CHECK(run("os.env(\"PATH\");") == "!PermissionError");
    CHECK(run("std::getenv(\"PATH\");") == "!PermissionError");
    os::remove_all(root);
}

TEST(hosts_policy_allows_listed_hosts) {
    auto p = sec::profile_defaults(sec::profile::standard); p.audit.os = false;
    p.net.hosts = {"127.0.0.1:9"};
    active a(p);
    // loopback is allowed in mode hosts: the call reaches the OS (port 9 is normally closed -> false, not an error)
    auto got = run("std::println(\"{}\", net.port_open(\"127.0.0.1\", 9, 200));");
    CHECK_MSG(got == "false\n" || got == "true\n", "%s", got.c_str());
    CHECK(run("net.port_open(\"192.0.2.1\", 80, 100);") == "!PermissionError");   // not listed
}

// ---------------------------------------------------------------- errors and plugins
TEST(security_error_script_classes) {
    script::options o; std::string out; o.out = [&](std::string_view s) { out += s; };
    script::engine e(o);
    e.define("integrity", 0, 0, [](const std::string&) -> std::string { throw sec::integrity_error("bad signature"); });
    e.define("policy", 0, 0, [](const std::string&) -> std::string { throw sec::policy_violation("bad policy"); });
    e.define("denied", 0, 0, [](const std::string&) -> std::string { throw sec::permission_denied("no"); });
    e.define("sandbox", 0, 0, [](const std::string&) -> std::string { throw sec::sandbox_error("cannot confine"); });
    auto r = e.run(R"(
try { integrity(); } catch (const IntegrityError& x) { std::println("IntegrityError"); }
try { integrity(); } catch (const SecurityError& x) { std::println("SecurityError"); }
try { policy(); } catch (const PolicyViolation& x) { std::println("PolicyViolation"); }
try { policy(); } catch (const OSError& x) { std::println("OSError"); }
try { denied(); } catch (const PermissionError& x) { std::println("PermissionError"); }
try { sandbox(); } catch (const SandboxViolationError& x) { std::println("SandboxViolationError"); }
)", "<classes>");
    CHECK_MSG(r.ok, "%s", r.error.c_str());
    CHECK_MSG(out == "IntegrityError\nSecurityError\nPolicyViolation\nOSError\nPermissionError\nSandboxViolationError\n", "%s", out.c_str());
}

#ifdef FORGE_TEST_PLUGIN
TEST(plugins_follow_integrity_plugins) {
    const char* plug = std::getenv("FORGE_TEST_PLUGIN_PATH"); if (!plug || !*plug) plug = FORGE_TEST_PLUGIN;
    const std::string src = "auto h = forge::plugin(" + quoted(plug) + "); std::println(\"loaded\");";
    capture cap;
    {
        auto p = sec::profile_defaults(sec::profile::permissive); p.audit.os = false;   // plugins optional: allowed, audited
        active a(p);
        CHECK_MSG(run(src) == "loaded\n", "%s", run(src).c_str());
        CHECK(cap.has("plugin", sec::decision::allow, "hello_plugin", "CAP-1"));
    }
    {
        auto p = sec::profile_defaults(sec::profile::standard); p.audit.os = false;   // plugins required: no .sig -> refused
        active a(p);
        CHECK(run(src) == "!IntegrityError");
        CHECK(cap.has("integrity", sec::decision::fail, "hello_plugin", "INT-4"));
        CHECK(!sec::try_enforce(sec::capability::plugin_load, plug));               // the gate alone refuses too (fail closed)
        CHECK(cap.has("plugin", sec::decision::deny, "hello_plugin", "CAP-1"));
    }
}
#endif

// ---------------------------------------------------------------- model integrity (INT-1 / INT-4)
TEST(model_integrity_under_policy) {
    auto d = scratch("model");
    auto plain = os::join(d, "plain.fpk"), signed_ = os::join(d, "signed.fpk"), other = os::join(d, "other.fpk"), bad = os::join(d, "tampered.fpk");
    forge_test::make_tiny_model(plain);
    auto k = modelfile::generate_keypair(), k2 = modelfile::generate_keypair();
    modelfile::sign_file(plain, signed_, k);
    modelfile::sign_file(plain, other, k2);
    {   // tampered: flip one byte of tensor data after signing
        auto bytes = slurp(signed_);
        bytes[bytes.size() / 2] = char(bytes[bytes.size() / 2] ^ 0x5a);
        std::ofstream(bad, std::ios::binary) << bytes;
    }
    auto loads = [](const std::string& f) -> std::string {
        try { inference::model m(f); return "ok"; }
        catch (const sec::integrity_error&) { return "integrity"; }
        catch (const std::exception& e) { return std::string("other: ") + e.what(); }
    };
    // no policy: unsigned and tampered files load as before (no verification requested)
    CHECK(loads(plain) == "ok");
    CHECK(loads(signed_) == "ok");
    capture cap;
    auto p = sec::profile_defaults(sec::profile::permissive); p.audit.os = false;
    p.integrity.models = sec::requirement::required; p.integrity.trusted_keys = {modelfile::detail::to_hex(k.pub)};
    {
        active a(p);
        CHECK(loads(signed_) == "ok");
        CHECK(loads(plain) == "integrity");
        CHECK(loads(bad) == "integrity");
        CHECK(loads(other) == "integrity");   // valid signature, untrusted key
        CHECK(cap.has("integrity", sec::decision::fail, plain, "INT-4"));
        CHECK(cap.has("integrity", sec::decision::allow, signed_, "INT-1"));
    }
    p.integrity.models = sec::requirement::optional; p.integrity.trusted_keys.clear();
    {
        active a(p);
        CHECK(loads(plain) == "ok");          // unsigned: loaded, audited
        CHECK(cap.has("integrity", sec::decision::weaken, plain, "INT-1"));
        CHECK(loads(other) == "ok");          // signed, any valid signature (no trusted_keys)
        CHECK(loads(bad) == "integrity");     // signed but modified
    }
    {   // fs.read scope on the model path
        auto q = sec::profile_defaults(sec::profile::strict); q.audit.os = false; q.integrity.models = sec::requirement::off;
        q.fs.read = {os::join(d, "nowhere")}; q.fs.write = {};
        active a(q);
        bool denied = false;
        try { inference::model m(plain); } catch (const sec::permission_denied&) { denied = true; }
        CHECK(denied);
    }
    os::remove_all(d);
}

// ---------------------------------------------------------------- lockdown + script in a child process
TEST(lockdown_child_runs_script_confined) {
    auto d = scratch("child");
    os::setenv("FORGE_SCRIPT_SECURITY_CHILD", d); os::setenv("FORGE_TEST_ONLY", "zz_lockdown_child");
    auto r = os::run({os::exe_path()}, true);
    os::unsetenv("FORGE_SCRIPT_SECURITY_CHILD"); os::unsetenv("FORGE_TEST_ONLY");
    CHECK_MSG(r && r->exit_code == 0, "child: %s", r ? (std::to_string(r->exit_code) + ": " + r->out).c_str() : r.err.what.c_str());
    os::remove_all(d);
}

// ---------------------------------------------------------------- static capability checks through indirect calls
// script/src/capflow.hpp: the checker follows callables (aliases, std::function, lambdas in variables, containers
// and struct members, callbacks, virtual dispatch) and warns where a call inside a context that denies a capability
// may reach a built-in that needs it. All of it is warning level: tests/script/indirect_calls.fs is the run-time side.
namespace {
// `capability` diagnostics as "line:severity:message\n"
std::string caps_of(const std::string& src, script::check_options o = {}) {
    std::string s;
    for (auto& d : script::check(src, "<t>", o)) if (d.id == "capability") s += std::to_string(d.line) + ":" + (d.severity == script::diagnostic::level::error ? "error:" : "warning:") + d.message + "\n";
    return s;
}
// Code placed inside forge::sandbox(<literal with fs off>, [] { ... }) after `decls`; the sandbox call is on the last line.
std::string in_no_fs(const std::string& decls, const std::string& body) {
    return decls + "\nforge::sandbox(std::map<std::string, bool>{{\"fs\", false}}, [] { " + body + " });\n";
}
bool flags(const std::string& diags, const std::string& part) { return diags.find("warning:") != std::string::npos && diags.find(part) != std::string::npos; }
const char* kDecls =
    "std::string helper(std::string p) { return forge::fs::read(p); }\n"          // 1
    "std::string outer(std::string p) { return helper(p); }\n"                    // 2
    "std::string pure(std::string p) { return p + \"!\"; }\n"                     // 3
    "auto lam = [] { return forge::fs::read(\"x\"); };\n"                         // 4
    "auto pure_lam = [] { return std::string(\"p\"); };\n";                       // 5
}

TEST(static_capability_indirect_forms) {
    struct form { const char* name; std::string decls, body, expect; };
    const std::string D = kDecls;
    const form forms[] = {
        {"direct qualified call", "", "forge::fs::read(\"x\");", "forge::fs::read needs the 'fs' capability (denied by the enclosing sandbox block)"},
        {"using namespace", "using namespace forge;", "fs::read(\"x\");", "fs::read needs the 'fs' capability"},
        {"script function", D, "helper(\"x\");", "'helper' may need the 'fs' capability (denied by the enclosing sandbox block): helper -> forge::fs::read (line 1)"},
        {"function calling a function", D, "outer(\"x\");", "outer -> helper -> forge::fs::read (line 1)"},
        {"alias of a built-in", "auto f = forge::fs::read;", "f(\"x\");", "'f' may be the built-in forge::fs::read"},
        {"alias of an alias", "auto f = forge::fs::read; auto g = f;", "g(\"x\");", "'g' may be the built-in forge::fs::read"},
        {"reference to a function", D + "auto& r = helper;", "r(\"x\");", "'r' may need the 'fs' capability"},
        {"function pointer variable", D + "auto fp = helper;", "fp(\"x\");", "'fp' may need the 'fs' capability"},
        {"std::function of a function", D + "std::function<std::string(std::string)> sf = helper;", "sf(\"x\");", "'sf' may need the 'fs' capability"},
        {"std::function of a built-in", "std::function<std::string(std::string)> sf = forge::fs::read;", "sf(\"x\");", "may be the built-in forge::fs::read"},
        {"lambda in a variable", D, "lam();", "'lam' may need the 'fs' capability (denied by the enclosing sandbox block): lambda (line 4) -> forge::fs::read (line 4)"},
        {"lambda reassigned", D + "std::function<std::string()> cur = pure_lam; void swap_in() { cur = lam; }", "cur();", "'cur' may need the 'fs' capability"},
        {"lambda in a vector", D + "std::vector<std::function<std::string()>> v{lam};", "v[0]();", "may need the 'fs' capability"},
        {"lambda pushed into a vector", D + "std::vector<std::function<std::string()>> v; v.push_back(lam);", "v.at(0)();", "may need the 'fs' capability"},
        {"vector range-for", D + "std::vector<std::function<std::string()>> v{lam};", "for (auto& f : v) f();", "'f' may need the 'fs' capability"},
        {"lambda in a map", D + "std::map<std::string, std::function<std::string()>> t{{\"r\", lam}};", "t[\"r\"]();", "may need the 'fs' capability"},
        {"map assigned by key", D + "std::map<std::string, std::function<std::string()>> t; t[\"r\"] = lam;", "t[\"r\"]();", "may need the 'fs' capability"},
        {"built-in in a map", "std::map<std::string, std::function<std::string(std::string)>> t{{\"r\", forge::fs::read}};", "t[\"r\"](\"x\");", "forge::fs::read"},
        {"struct member (aggregate)", D + "struct H { std::function<std::string()> cb; }; H h{lam};", "h.cb();", "'h.cb' may need the 'fs' capability"},
        {"struct member (assigned)", D + "struct H { std::function<std::string()> cb; }; H h; h.cb = lam;", "h.cb();", "'h.cb' may need the 'fs' capability"},
        {"struct member (constructor)", D + "struct H { std::function<std::string()> cb; H(std::function<std::string()> f) : cb(f) {} std::string go() { return cb(); } }; H h(lam);", "h.go();", "'h.go' may need the 'fs' capability"},
        {"struct member (default initialiser)", "struct H { std::function<std::string()> cb = [] { return forge::fs::read(\"x\"); }; }; H h;", "h.cb();", "'h.cb' may need the 'fs' capability"},
        {"method", "struct R { std::string get() { return forge::fs::read(\"x\"); } }; R r;", "r.get();", "'r.get' may need the 'fs' capability (denied by the enclosing sandbox block): get -> forge::fs::read (line 1)"},
        {"method calling a method", "struct R { std::string raw() { return forge::fs::read(\"x\"); } std::string get() { return raw(); } }; R r;", "r.get();", "get -> raw -> forge::fs::read"},
        {"virtual dispatch through a base reference",
         "struct S { virtual std::string get() { return \"m\"; } };\nstruct F : S { std::string get() override { return forge::fs::read(\"x\"); } };\nstd::string pull(S& s) { return s.get(); }\nF f;",
         "pull(f);", "pull -> get -> forge::fs::read (line 2)"},
        {"virtual dispatch through shared_ptr",
         "struct S { virtual std::string get() { return \"m\"; } };\nstruct F : S { std::string get() override { return forge::fs::read(\"x\"); } };\nstd::shared_ptr<S> s = std::make_shared<F>();",
         "s->get();", "may need the 'fs' capability"},
        {"static member function", "struct U { static std::string load() { return forge::fs::read(\"x\"); } };", "U::load();", "may need the 'fs' capability"},
        {"call operator", "struct C { std::string operator()() { return forge::fs::read(\"x\"); } }; C c;", "c();", "'c' may need the 'fs' capability"},
        {"callback argument", D + "std::string apply(std::function<std::string()> f) { return f(); }", "apply(lam);", "'apply' may need the 'fs' capability (denied by the enclosing sandbox block): apply -> its parameter 'f' -> lambda (line 4) -> forge::fs::read (line 4)"},
        {"callback passed on", D + "std::string apply(std::function<std::string()> f) { return f(); } std::string relay(std::function<std::string()> g) { return apply(g); }", "relay(lam);", "relay -> its parameter 'g' -> lambda (line 4) -> forge::fs::read (line 4)"},
        {"built-in as a callback argument", "std::string apply(std::function<std::string(std::string)> f) { return f(\"x\"); }", "apply(forge::fs::read);", "its parameter 'f' -> forge::fs::read"},
        {"callback returned by a function", D + "std::function<std::string()> pick() { return lam; }", "pick()();", "may need the 'fs' capability"},
        {"callback captured by a lambda", D + "auto wrap = [] { return lam(); };", "wrap();", "'wrap' may need the 'fs' capability"},
        {"callable handed to a built-in", D, "std::vector<std::string> ps{\"x\"}; std::for_each(ps.begin(), ps.end(), helper);", "a callable passed to std::for_each may need the 'fs' capability"},
        {"built-in handed to a built-in", "", "std::vector<std::string> ps{\"x\"}; std::for_each(ps.begin(), ps.end(), forge::fs::remove);", "a callable passed to std::for_each may need the 'fs' capability"},
        {"std::bind", D + "auto b = std::bind(helper, \"x\");", "b();", "'b' may need the 'fs' capability"},
        {"thread body defined outside", D, "std::async(lam).get();", "a callable passed to std::async may need the 'fs' capability"},
        {"nested limit block", D, "limit {.timeout = \"5s\"} { helper(\"x\"); }", "'helper' may need the 'fs' capability"},
        {"forward reference", "std::string early() { return late(); }\nstd::string late() { return forge::fs::read(\"x\"); }", "early();", "early -> late -> forge::fs::read (line 2)"},
        {"recursion", "std::string walk(int n) { return n ? walk(n - 1) : forge::fs::read(\"x\"); }", "walk(3);", "walk -> forge::fs::read"},
    };
    for (auto& f : forms) {
        auto src = in_no_fs(f.decls, f.body);
        auto got = caps_of(src);
        CHECK_MSG(flags(got, f.expect), "%s: expected a warning with \"%s\"\n--- source\n%s--- got\n%s", f.name, f.expect.c_str(), src.c_str(), got.c_str());
        CHECK_MSG(got.find("error:") == std::string::npos, "%s: indirect findings are warnings\n%s", f.name, got.c_str());
        // warnings never refuse a run; --strict-check does
        if (std::string(f.name) == "script function") {
            script::options strict; strict.strict_check = true;
            CHECK(run(src, strict).rfind("!StaticCheckError", 0) == 0);
            CHECK(run(src) == "!PermissionError");
        }
    }
}

TEST(static_capability_indirect_no_false_positives) {
    const std::string D = kDecls;
    const std::pair<const char*, std::pair<std::string, std::string>> clean[] = {
        {"pure function", {D, "pure(\"x\");"}},
        {"pure lambda", {D, "pure_lam();"}},
        {"alias of a pure built-in", {"auto p = std::to_string;", "p(1);"}},
        {"std::function of a pure function", {D + "std::function<std::string(std::string)> sf = pure;", "sf(\"x\");"}},
        {"pure lambdas in a vector", {D + "std::vector<std::function<std::string()>> v{pure_lam}; v.push_back(pure_lam);", "v[0](); for (auto& f : v) f();"}},
        {"pure lambda in a map", {D + "std::map<std::string, std::function<std::string()>> t{{\"p\", pure_lam}};", "t[\"p\"]();"}},
        {"pure struct member", {D + "struct H { std::function<std::string()> on_pure; }; H h{pure_lam};", "h.on_pure();"}},
        // another member of the same struct holds a capability callable: fields are tracked by name
        {"other member of a struct", {D + "struct H { std::function<std::string()> a; std::function<std::string()> b; }; H h{lam, pure_lam};", "h.b();"}},
        {"pure callback argument", {D + "std::string apply(std::function<std::string()> f) { return f(); }\nauto elsewhere = apply(lam);", "apply(pure_lam);"}},
        {"pure callback passed on", {D + "std::string apply(std::function<std::string()> f) { return f(); } std::string relay(std::function<std::string()> g) { return apply(g); }\nauto elsewhere = relay(lam);", "relay(pure_lam);"}},
        {"literal lambda callback", {D + "std::string apply(std::function<std::string()> f) { return f(); }\nauto elsewhere = apply(lam);", "apply([] { return std::string(\"q\"); });"}},
        // the receiver's class is known, so the override in another class does not count
        {"method of a known class", {"struct S { virtual std::string get() { return \"m\"; } };\nstruct F : S { std::string get() override { return forge::fs::read(\"x\"); } };\nS s; F f;", "s.get();"}},
        {"same method name in an unrelated class", {"struct A { std::string get() { return forge::fs::read(\"x\"); } };\nstruct B { std::string get() { return \"b\"; } };\nB b;", "b.get();"}},
        {"callable handed to a built-in", {D, "std::vector<std::string> ps{\"x\"}; std::for_each(ps.begin(), ps.end(), pure); std::sort(ps.begin(), ps.end(), [](auto a, auto b) { return a < b; });"}},
        {"storing is not calling", {D, "std::vector<std::function<std::string()>> keep; keep.push_back(lam); auto copy = lam; std::println(\"{}\", keep.size());"}},
        {"built-in methods on a container of capability callables", {D + "std::vector<std::function<std::string()>> v{lam};", "std::println(\"{} {}\", v.size(), v.empty()); v.clear();"}},
        {"capability function that is never reached", {D, "std::println(\"{}\", pure(\"a\")); int n = 0; for (int i = 0; i < 3; ++i) n += i;"}},
        {"other capability", {"std::string host() { return forge::net::resolve(\"localhost\"); }", "host();"}},
        {"data from a capability call made outside", {"auto text = forge::fs::read(\"x\");", "std::println(\"{}\", text.size());"}},
        {"recursion over pure code", {"int fact(int n) { return n < 2 ? 1 : n * fact(n - 1); }", "fact(5);"}},
    };
    for (auto& [name, src2] : clean) {
        auto src = in_no_fs(src2.first, src2.second);
        auto got = caps_of(src);
        CHECK_MSG(got.empty(), "%s: no capability diagnostic expected\n--- source\n%s--- got\n%s", name, src.c_str(), got.c_str());
    }
    // nothing is denied: no capability diagnostics at all, whatever the script does
    CHECK(caps_of(D + "helper(\"x\"); lam(); auto f = forge::fs::read; f(\"x\"); std::function<void()> u; u();").empty());
    // the repository's own scripts stay clean (no new diagnostics on the goldens)
    for (auto& e : fs::directory_iterator("tests/script")) if (e.path().extension() == ".fs") {
        auto got = caps_of(slurp(e.path()));
        CHECK_MSG(got.empty(), "%s:\n%s", e.path().string().c_str(), got.c_str());
    }
}

TEST(static_capability_indirect_contexts) {
    const std::string D = kDecls;
    // every literal spelling of the sandbox options is recognised
    for (const char* open : {"forge::sandbox(std::map<std::string, bool>{{\"fs\", false}}, [] { ", "forge::sandbox({{\"fs\", false}}, [] { ", "sandbox({{\"fs\", false}, {\"net\", false}}, [] { "}) {
        auto got = caps_of(D + open + "helper(\"x\"); });");
        CHECK_MSG(flags(got, "'helper' may need the 'fs' capability"), "%s\n%s", open, got.c_str());
    }
    // options that are not a literal, or that leave the capability on, say nothing
    CHECK(caps_of(D + "const std::map<std::string, bool> o{{\"fs\", false}}; forge::sandbox(o, [] { helper(\"x\"); });").empty());
    CHECK(caps_of(D + "forge::sandbox({{\"fs\", true}, {\"net\", false}}, [] { helper(\"x\"); });").empty());
    // the block is a named function instead of a literal lambda
    CHECK(flags(caps_of(D + "void work() { helper(\"x\"); }\nforge::sandbox({{\"fs\", false}}, work);"), "a callable passed to forge::sandbox may need the 'fs' capability (denied by this sandbox block): work -> helper -> forge::fs::read (line 1)"));
    CHECK(caps_of(D + "void work() { pure(\"x\"); }\nforge::sandbox({{\"fs\", false}}, work);").empty());
    // nested blocks add up; a nested block cannot turn a capability back on
    CHECK(flags(caps_of(D + "std::string host() { return forge::net::resolve(\"h\"); }\nforge::sandbox({{\"fs\", false}}, [] { forge::sandbox({{\"net\", false}, {\"fs\", true}}, [] { helper(\"x\"); host(); }); });"), "'host' may need the 'net' capability"));
    CHECK(flags(caps_of(D + "forge::sandbox({{\"fs\", false}}, [] { forge::sandbox({{\"fs\", true}}, [] { helper(\"x\"); }); });"), "'helper' may need the 'fs' capability"));
    // each capability: proc (shell call, run), net, env
    CHECK(flags(caps_of("void sh_it() { forge::sh(\"echo hi\"); }\nforge::sandbox({{\"proc\", false}}, [] { sh_it(); });"), "sh_it -> forge::sh (line 1)"));
    CHECK(flags(caps_of("void go() { forge::run(\"ls\"); }\nforge::sandbox({{\"proc\", false}}, [] { go(); });"), "'go' may need the 'proc' capability"));
    CHECK(flags(caps_of("auto e = [] { return std::getenv(\"HOME\"); };\nforge::sandbox({{\"env\", false}}, [] { e(); });"), "'e' may need the 'env' capability"));
    CHECK(flags(caps_of("auto r = forge::net::resolve;\nforge::sandbox({{\"net\", false}}, [] { r(\"h\"); });"), "'r' may be the built-in forge::net::resolve"));

    // capabilities denied for the whole run (--no-fs): a built-in reached through a value is flagged at the call;
    // script functions are not flagged at their call sites, because the place that uses the built-in is
    script::check_options no_fs; no_fs.allow_fs = false;
    auto run_level = caps_of(D + "auto f = forge::fs::read;\nvoid use(std::function<std::string(std::string)> g) { g(\"x\"); }\nuse(f);\nhelper(\"y\");", no_fs);
    CHECK_MSG(flags(run_level, "1:warning:forge::fs::read needs the 'fs' capability (denied by this run)"), "%s", run_level.c_str());
    CHECK_MSG(flags(run_level, "7:warning:'g' may be the built-in forge::fs::read, which needs the 'fs' capability (denied by this run)"), "%s", run_level.c_str());
    CHECK_MSG(run_level.find("'helper' may need") == std::string::npos && run_level.find("'use' may need") == std::string::npos, "%s", run_level.c_str());
    // the direct lib.fn spelling stays an error and is not reported twice
    auto dotted = caps_of("fs.read(\"x\");", no_fs);
    CHECK_MSG(dotted == "1:error:fs.read needs the 'fs' capability, which this run denies (raises PermissionError)\n", "%s", dotted.c_str());

    // an unresolved callee: it may be any address-taken function
    const std::string unk = "std::function<std::string()> saved = lam;\nstd::string fire(std::function<std::string()> cb) { return cb(); }\n";   // nobody calls fire()
    auto u1 = caps_of(D + unk + "forge::sandbox({{\"fs\", false}}, [] { auto f = eval(\"lam\"); f(); });");
    CHECK_MSG(flags(u1, "the target of 'f' cannot be resolved; it may be any function whose address is taken, and one of them needs the 'fs' capability"), "%s", u1.c_str());
    // ... and says nothing when no address-taken function needs the capability
    CHECK(caps_of("auto p = [] { return 1; };\nforge::sandbox({{\"fs\", false}}, [] { auto f = eval(\"p\"); f(); });").empty());
    // suppression comments work as for every other check
    CHECK(caps_of(D + "forge::sandbox({{\"fs\", false}}, [] {\n    helper(\"x\");   // forge-check: ignore[capability]\n});").empty());
}
