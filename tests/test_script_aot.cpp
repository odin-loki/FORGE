// forge script --aot (docs/aot.md): AOT binaries must behave exactly like the interpreter.
//   aot_embed_sweep      every corpus script, embed mode: stdout, exit status and the uncaught-error text
//   aot_native_subset    a fast subset in native mode with the first host compiler found
//   aot_native_sweep     every corpus script with every host compiler (clang++-23, g++); only with
//                        FORGE_AOT_SWEEP=full (ctest -L slow), exit 77 when no compiler is installed
//   aot_semantics_*      errors, limits, capabilities, the baked policy, plugins, arguments, determinism
// The corpus: tests/script/*.fs, tests/aot/*.fs, fuzz/corpus/script/*.fs, examples/scripts/*.cpp, tests/data/emit/*.cpp.
// FORGE_AOT_TEST_CXX names the compilers to use (comma separated) instead of the default search.
#include <forge/script/aot.hpp>
#include <forge/script/script.hpp>
#include <forge/os/fs.hpp>
#include <forge/os/os.hpp>
#include <forge/os/process.hpp>
#include "check.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
using namespace forge;
namespace fs = std::filesystem;
TEST_PROCESS_WORKER(script::run_process_worker_if_requested);

#ifndef FORGE_AOT_TEST_PCH_DIR
#define FORGE_AOT_TEST_PCH_DIR ""
#endif

namespace {

std::string slurp(const fs::path& p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }

const std::string& work_dir() {
    static const std::string d = [] { auto t = os::fs::make_temp_dir("forge-aot-test-"); return t ? t.value : std::string("."); }();
    return d;
}
struct Cleanup { ~Cleanup() { std::error_code ec; if (work_dir() != ".") fs::remove_all(work_dir(), ec); } } cleanup_at_exit;

std::vector<std::string> compilers() {
    static const std::vector<std::string> found = [] {
        std::vector<std::string> names, out;
        if (auto e = os::getenv("FORGE_AOT_TEST_CXX"); !e.empty()) { std::stringstream ss(e); std::string n; while (std::getline(ss, n, ',')) if (!n.empty()) names.push_back(n); }
        else names = {"clang++-23", "clang++", "g++"};
        if (!script::aot_supported()) return out;
        for (auto& n : names) {
            auto p = os::find_executable(n);
            if (p.empty()) continue;
            std::error_code ec; auto canon = fs::weakly_canonical(p, ec).string();
            bool dup = false; for (auto& o : out) if (fs::weakly_canonical(o, ec).string() == canon) dup = true;
            if (!dup) out.push_back(p);
        }
        return out;
    }();
    return found;
}

struct Run { std::string out, err; int code = 0; bool refused = false; };   // refused: the script never started (exit 2)

// The interpreter, in process, the way test_script_golden runs it.
Run interpret(const std::string& src, const std::string& name, script::options o = {}) {
    Run r;
    o.out = [&](std::string_view s) { r.out += s; };
    o.os_sandbox = script::options::os_sandbox_mode::off;   // never confine the test process itself
    script::engine e(o);
    auto res = e.run(src, name);
    r.code = res.exit_code;
    if (!res.ok) { r.err = res.error + "\n"; r.refused = res.exit_code == 2; }
    return r;
}

Run run_binary(const std::string& exe, std::vector<std::string> args = {}, std::map<std::string, std::string> env = {}) {
    os::spawn_spec sp;
    sp.argv.push_back(exe); for (auto& a : args) sp.argv.push_back(a);
    sp.env_set = std::move(env); sp.timeout_ms = 120000;
    auto p = os::spawn_wait(sp);
    Run r;
    if (!p) { r.code = -1; r.err = "spawn failed: " + p.err.what; return r; }
    r.out = p->out; r.err = p->err; r.code = p->timed_out ? -2 : p->exit_code;
    return r;
}

int g_serial = 0;
std::vector<std::string> g_built;
// Delete the executables built so far (an embed-mode binary is a copy of the whole runner stub).
void purge() { std::error_code ec; for (auto& p : g_built) fs::remove(p, ec); g_built.clear(); }
// Compile; on refusal returns an empty path and fills *why / *exit_code.
std::string compile(const std::string& src, const std::string& name, script::aot_mode mode, const std::string& cxx, script::aot_options o = {}, std::string* why = nullptr, int* exit_code = nullptr) {
    o.mode = mode; o.cxx = cxx; o.opt_level = "1";
    if (o.pch_dir.empty()) o.pch_dir = FORGE_AOT_TEST_PCH_DIR;
    std::string out = (fs::path(work_dir()) / ("p" + std::to_string(++g_serial) + (mode == script::aot_mode::native ? "_native" : "_embed"))).string();
#if defined(_WIN32)
    out += ".exe";
#endif
    try { auto r = script::aot_compile(src, name, out, o); if (why) *why = r.detail; g_built.push_back(r.output); return r.output; }
    catch (const script::aot_error& e) { if (why) *why = e.what(); if (exit_code) *exit_code = e.exit_code; return {}; }
}

std::vector<fs::path> corpus(bool subset) {
    std::vector<fs::path> files;
    auto add = [&](const char* dir, const char* ext) {
        std::error_code ec;
        std::vector<fs::path> v;
        for (auto& e : fs::directory_iterator(dir, ec)) if (e.path().extension() == ext) v.push_back(e.path());
        std::sort(v.begin(), v.end());
        files.insert(files.end(), v.begin(), v.end());
    };
    add("tests/aot", ".fs");
    if (subset) { files.push_back("tests/script/exceptions.fs"); files.push_back("tests/script/functions.fs"); return files; }
    add("tests/script", ".fs"); add("fuzz/corpus/script", ".fs"); add("examples/scripts", ".cpp"); add("tests/data/emit", ".cpp");
    return files;
}

// One script, one mode: the binary's stdout, exit status and uncaught-error text must be the interpreter's.
void check_same(const fs::path& f, script::aot_mode mode, const std::string& cxx) {
    const std::string name = f.generic_string(), src = slurp(f);
    const char* what = mode == script::aot_mode::native ? "native" : "embed";
    script::options o; o.timeout_ms = 60000;
    Run want = interpret(src, name, o);
    script::aot_options ao; ao.run = o;
    std::string why; int ec = 0;
    std::string exe = compile(src, name, mode, cxx, ao, &why, &ec);
    std::fprintf(stderr, "== %s [%s%s%s] %s\n", name.c_str(), what, cxx.empty() ? "" : " ", cxx.c_str(), exe.empty() ? "refused" : "built");
    if (want.refused) {   // a syntax or static-check error: the compiler must refuse it the same way
        CHECK_MSG(exe.empty() && ec == 2, "%s (%s): the interpreter refuses this script, AOT compiled it", name.c_str(), what);
        return;
    }
    CHECK_MSG(!exe.empty(), "%s (%s): AOT compile failed: %s", name.c_str(), what, why.c_str());
    if (exe.empty()) return;
    Run got = run_binary(exe);
    CHECK_MSG(got.out == want.out, "%s (%s): stdout differs\n--- aot\n%s--- interpreter\n%s", name.c_str(), what, got.out.c_str(), want.out.c_str());
    CHECK_MSG(got.code == want.code, "%s (%s): exit status %d, interpreter %d\n%s", name.c_str(), what, got.code, want.code, got.err.c_str());
    // (what the script itself writes to stderr comes first; the uncaught error, if any, is the last thing printed)
    CHECK_MSG(got.err.ends_with(want.err), "%s (%s): stderr differs\n--- aot\n%s--- interpreter\n%s", name.c_str(), what, got.err.c_str(), want.err.c_str());
    purge();
}

// Both modes for the semantics tests: embed always, native with the first compiler when there is one.
std::vector<std::pair<script::aot_mode, std::string>> modes() {
    std::vector<std::pair<script::aot_mode, std::string>> m{{script::aot_mode::embed, ""}};
    if (!compilers().empty()) m.push_back({script::aot_mode::native, compilers().front()});
    return m;
}
bool has(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }
const bool sweep_only = os::getenv("FORGE_AOT_SWEEP") == "full";

} // namespace

TEST(aot_embed_sweep) {
    if (sweep_only) return;
    auto files = corpus(false);
    CHECK(files.size() >= 20);
    for (auto& f : files) check_same(f, script::aot_mode::embed, "");
}

TEST(aot_native_subset) {
    if (sweep_only) return;
    if (compilers().empty()) { std::fprintf(stderr, "aot_native_subset: no host C++ compiler, native mode not tested\n"); return; }
    for (auto& f : corpus(true)) check_same(f, script::aot_mode::native, compilers().front());
}

TEST(aot_native_sweep) {
    if (!sweep_only) return;
    if (compilers().empty()) { std::fprintf(stderr, "aot_native_sweep: no host C++ compiler found, skipping\n"); std::fflush(stderr); std::_Exit(77); }
    for (auto& cxx : compilers()) for (auto& f : corpus(false)) check_same(f, script::aot_mode::native, cxx);
}

TEST(aot_semantics_errors_and_exit_status) {
    if (sweep_only) return;
    const char* bounds = "#include <vector>\nint main() {\n    std::vector<int> v{1, 2, 3};\n    std::println(\"before\");\n    int i = 7;\n    std::println(\"{}\", v.at(i));\n    return 0;\n}\n";
    const char* thrown = "struct Bad : std::runtime_error { using std::runtime_error::runtime_error; };\nvoid inner(int n) { if (n == 0) throw Bad(\"deep\"); inner(n - 1); }\nint main() { std::println(\"start\"); inner(3); return 0; }\n";
    const char* exits = "int main() { std::println(\"a\"); std::exit(41); return 1; }\n";
    const char* overflow = "int twice(int x) { int y = x * 2; return y; }\nint main() { int big = 9223372036854775807; std::println(\"{}\", twice(big)); return 0; }\n";
    for (auto& [mode, cxx] : modes()) {
        purge();
        for (const char* src : {bounds, thrown, exits, overflow}) {
            Run want = interpret(src, "case.fs");
            std::string why; std::string exe = compile(src, "case.fs", mode, cxx, {}, &why);
            CHECK_MSG(!exe.empty(), "compile failed: %s", why.c_str());
            if (exe.empty()) continue;
            Run got = run_binary(exe);
            CHECK_MSG(got.out == want.out && got.code == want.code && got.err.ends_with(want.err), "mode %d\n--- aot (%d)\n%s%s--- interpreter (%d)\n%s%s", int(mode), got.code, got.out.c_str(), got.err.c_str(), want.code, want.out.c_str(), want.err.c_str());
        }
        Run b = run_binary(compile(bounds, "case.fs", mode, cxx));
        CHECK(b.code == 1 && has(b.err, "IndexError") && b.out == "before\n");
        Run x = run_binary(compile(exits, "case.fs", mode, cxx));
        CHECK(x.code == 41 && x.out == "a\n" && x.err.empty());
        Run ov = run_binary(compile(overflow, "case.fs", mode, cxx));
        CHECK(ov.code == 1 && has(ov.err, "OverflowError"));
        // Refused before anything is built: a syntax error and a static-check error (exit 2), like `forge script`.
        std::string why; int ec = 0;
        CHECK(compile("int main() { return 0 \n", "bad.fs", mode, cxx, {}, &why, &ec).empty() && ec == 2 && has(why, "SyntaxError"));
        ec = 0; script::aot_options nofs; nofs.run.allow_fs = false;
        CHECK(compile("int main() { fs.write(\"x\", \"y\"); return 0; }\n", "cap.fs", mode, cxx, nofs, &why, &ec).empty() && ec == 2 && has(why, "StaticCheckError"));
    }
}

TEST(aot_semantics_limits) {
    if (sweep_only) return;
    const char* spin = "int main() { std::println(\"go\"); int n = 0; while (true) { n = n + 1; if (n < 0) break; } return 0; }\n";
    const char* grow = "#include <vector>\n#include <string>\nint main() { std::vector<std::string> v; try { while (true) v.push_back(std::string(4096, 'x')); } catch (const std::bad_alloc& e) { v.clear(); std::println(\"caught oom\"); return 9; } return 0; }\n";
    const char* deep = "int down(int n) { return down(n + 1) + 1; }\nint main() { try { down(0); } catch (const std::exception& e) { std::println(\"recursion stopped\"); return 5; } return 0; }\n";
    for (auto& [mode, cxx] : modes()) {
        purge();
        // Limits baked at compile time.
        script::aot_options o; o.run.timeout_ms = 300;
        Run t = run_binary(compile(spin, "spin.fs", mode, cxx, o));
        CHECK_MSG(t.code == 1 && has(t.err, "TimeoutError") && t.out == "go\n", "baked timeout: %d %s", t.code, t.err.c_str());
        script::aot_options s; s.run.max_steps = 5000;
        Run st = run_binary(compile(spin, "spin.fs", mode, cxx, s));
        CHECK_MSG(st.code == 1 && has(st.err, "LimitExceededError") && has(st.err, "5000"), "baked steps: %d %s", st.code, st.err.c_str());
        script::aot_options h; h.run.heap_bytes = 8u << 20;
        Run hp = run_binary(compile(grow, "grow.fs", mode, cxx, h));
        CHECK_MSG(hp.code == 9 && hp.out == "caught oom\n", "baked heap: %d %s %s", hp.code, hp.out.c_str(), hp.err.c_str());
        Run dp = run_binary(compile(deep, "deep.fs", mode, cxx));
        CHECK_MSG(dp.code == 5 && dp.out == "recursion stopped\n", "depth: %d %s %s", dp.code, dp.out.c_str(), dp.err.c_str());
        // The same limits given at run time, as leading --forge-* arguments and through the environment.
        std::string plain = compile(spin, "spin.fs", mode, cxx);
        Run a = run_binary(plain, {"--forge-timeout=300"});
        CHECK(a.code == 1 && has(a.err, "TimeoutError"));
        Run b = run_binary(plain, {"--forge-max-steps", "4000"});
        CHECK(b.code == 1 && has(b.err, "4000"));
        Run c = run_binary(plain, {}, {{"FORGE_AOT_MAX_STEPS", "3000"}});
        CHECK(c.code == 1 && has(c.err, "3000"));
        Run d = run_binary(compile(grow, "grow.fs", mode, cxx), {"--forge-heap=8388608"});
        CHECK(d.code == 9);
        // A baked limit can be tightened at run time, never raised or removed.
        std::string capped = compile(spin, "spin.fs", mode, cxx, s);
        Run e1 = run_binary(capped, {"--forge-max-steps=100"});
        CHECK(e1.code == 1 && has(e1.err, "step limit of 100 "));
        Run e2 = run_binary(capped, {"--forge-max-steps=900000000"});
        CHECK(e2.code == 1 && has(e2.err, "step limit of 5000 "));
        Run e3 = run_binary(capped, {"--forge-max-steps=0"});
        CHECK(e3.code == 1 && has(e3.err, "step limit of 5000 "));
        Run bad = run_binary(plain, {"--forge-nonsense"});
        CHECK(bad.code == 2 && has(bad.err, "unknown run-time option"));
    }
}

TEST(aot_semantics_capabilities) {
    if (sweep_only) return;
    const std::string probe = (fs::path(work_dir()) / "cap-probe.txt").generic_string();
    // Indirect calls: the static checker cannot refuse these, so the run-time check is what is being tested.
    const std::string src =
        "#include <string>\nint main() {\n"
        "    auto w = forge::fs::write; auto e = forge::os::env; auto r = forge::proc::run;\n"
        "    std::string path = forge::args()[0];\n"
        "    try { w(path, \"data\"); std::println(\"fs ok\"); } catch (const PermissionError& x) { std::println(\"fs denied\"); }\n"
        "    try { e(\"HOME\", \"\"); std::println(\"env ok\"); } catch (const PermissionError& x) { std::println(\"env denied\"); }\n"
        "    try { r(\"forge-aot-test-no-such-program\"); std::println(\"proc ok\"); } catch (const std::exception& x) { std::string m = x.what(); std::println(m.starts_with(\"spawn\") ? \"proc ok\" : \"proc denied\"); }\n"
        "    forge::sandbox(std::map<std::string, bool>{{\"fs\", false}}, [&] { try { w(path, \"again\"); std::println(\"sandbox fs ok\"); } catch (const PermissionError& x) { std::println(\"sandbox fs denied\"); } });\n"
        "    return 0;\n}\n";
    for (auto& [mode, cxx] : modes()) {
        purge();
        script::aot_options open; open.run.static_check = false; open.run.os_sandbox = script::options::os_sandbox_mode::off;
        std::string free_bin = compile(src, "caps.fs", mode, cxx, open);
        std::error_code ec; fs::remove(probe, ec);
        Run a = run_binary(free_bin, {probe});
        CHECK_MSG(a.code == 0 && a.out == "fs ok\nenv ok\nproc ok\nsandbox fs denied\n", "all on: %d\n%s%s", a.code, a.out.c_str(), a.err.c_str());
        CHECK(fs::exists(probe));
        // Narrowed at run time: arguments and environment.
        Run b = run_binary(free_bin, {"--forge-no-fs", "--forge-no-proc", "--forge-os-sandbox=off", probe});
        CHECK_MSG(b.code == 0 && b.out == "fs denied\nenv ok\nproc denied\nsandbox fs denied\n", "narrowed: %d\n%s%s", b.code, b.out.c_str(), b.err.c_str());
        Run c = run_binary(free_bin, {probe}, {{"FORGE_AOT_NO_FS", "1"}, {"FORGE_AOT_NO_ENV", "1"}});
        CHECK_MSG(c.code == 0 && c.out == "fs denied\nenv denied\nproc ok\nsandbox fs denied\n", "env narrowed: %d\n%s%s", c.code, c.out.c_str(), c.err.c_str());
        // Baked at compile time: denied, and no run-time option turns it back on.
        script::aot_options baked = open; baked.run.allow_fs = false; baked.run.allow_proc = false;
        std::string locked = compile(src, "caps.fs", mode, cxx, baked);
        fs::remove(probe, ec);
        Run d = run_binary(locked, {probe});
        CHECK_MSG(d.code == 0 && d.out == "fs denied\nenv ok\nproc denied\nsandbox fs denied\n", "baked: %d\n%s%s", d.code, d.out.c_str(), d.err.c_str());
        CHECK(!fs::exists(probe));
        Run w1 = run_binary(locked, {"--forge-allow-fs", probe});
        CHECK_MSG(w1.code == 2 && w1.out.empty() && has(w1.err, "--aot-allow-widen"), "widen refused: %d %s", w1.code, w1.err.c_str());
        Run w2 = run_binary(locked, {"--forge-no-check", probe});
        CHECK(w2.code == 2 && w2.out.empty());
        CHECK(!fs::exists(probe));
        Run info = run_binary(locked, {"--forge-aot-info"});
        CHECK(info.code == 0 && has(info.out, "fs=off") && has(info.out, "proc=off") && has(info.out, "net=on") && has(info.out, "not allowed"));
        // Compiled with allow_widen: the same option is honoured.
        script::aot_options wide = baked; wide.allow_widen = true;
        Run w3 = run_binary(compile(src, "caps.fs", mode, cxx, wide), {"--forge-allow-fs", "--forge-allow-proc", probe});
        CHECK_MSG(w3.code == 0 && w3.out == "fs ok\nenv ok\nproc ok\nsandbox fs denied\n", "widened: %d\n%s%s", w3.code, w3.out.c_str(), w3.err.c_str());
        // With the static checks on, a script whose direct calls the baked policy denies is refused at start-up
        // when the policy is narrowed at run time, exactly like `forge script --no-fs`.
        std::string direct = compile("int main() { std::println(\"run\"); fs.exists(\"x\"); return 0; }\n", "direct.fs", mode, cxx);
        Run ok = run_binary(direct);
        CHECK(ok.code == 0 && ok.out == "run\n");
        Run refused = run_binary(direct, {"--forge-no-fs"});
        CHECK_MSG(refused.code == 2 && refused.out.empty() && has(refused.err, "StaticCheckError"), "static refusal: %d %s", refused.code, refused.err.c_str());
    }
}

#if defined(__linux__)
// The OS sandbox is the engine's: a binary with a capability baked off confines itself at start-up.
TEST(aot_semantics_os_sandbox) {
    if (sweep_only) return;
    const char* src = "int main() { auto w = forge::fs::write; try { w(forge::args()[0], \"x\"); } catch (const std::exception& e) { std::println(\"denied\"); } return 0; }\n";
    const std::string probe = (fs::path(work_dir()) / "os-probe.txt").generic_string();
    for (auto& [mode, cxx] : modes()) {
        purge();
        script::aot_options o; o.run.allow_net = false; o.run.static_check = false;
        Run r = run_binary(compile(src, "sb.fs", mode, cxx, o), {probe});
        CHECK_MSG(r.code == 0 || r.code == 1, "os sandbox run: %d %s", r.code, r.err.c_str());
        script::aot_options req = o; req.run.os_sandbox = script::options::os_sandbox_mode::require;
        std::string strict = compile(src, "sb.fs", mode, cxx, req);
        Run weaker = run_binary(strict, {"--forge-os-sandbox=off", probe});
        CHECK_MSG(weaker.code == 2 && has(weaker.err, "--aot-allow-widen"), "weakening the OS sandbox: %d %s", weaker.code, weaker.err.c_str());
    }
}
#endif

TEST(aot_semantics_plugins) {
    if (sweep_only) return;
#ifdef FORGE_TEST_PLUGIN
    const std::string src = "int main() {\n    try { auto h = forge::plugin(forge::args()[0]); std::println(\"{}\", h.add(2, 3.5)); }\n    catch (const std::exception& e) { std::println(\"refused\"); return 3; }\n    return 0;\n}\n";
    script::options io; io.argv = {FORGE_TEST_PLUGIN};
    Run want = interpret(src, "plug.fs", io);
    for (auto& [mode, cxx] : modes()) {
        purge();
        script::aot_options o; o.run.os_sandbox = script::options::os_sandbox_mode::off; o.run.static_check = false;
        std::string exe = compile(src, "plug.fs", mode, cxx, o);
        Run loaded = run_binary(exe, {FORGE_TEST_PLUGIN});
        CHECK_MSG(loaded.code == want.code && loaded.out == want.out && want.code == 0, "plugin loaded: %d %s %s (interpreter %d %s)", loaded.code, loaded.out.c_str(), loaded.err.c_str(), want.code, want.out.c_str());
        Run refused = run_binary(exe, {"--forge-no-net", FORGE_TEST_PLUGIN});   // native code is refused once any capability is off
        CHECK_MSG(refused.code == 3 && refused.out == "refused\n", "plugin refused: %d %s %s", refused.code, refused.out.c_str(), refused.err.c_str());
    }
#else
    std::fprintf(stderr, "aot_semantics_plugins: example plugin not built\n");
#endif
}

TEST(aot_semantics_arguments_and_environment) {
    if (sweep_only) return;
    const char* src = "int main() {\n    for (auto a : forge::args()) std::println(\"[{}]\", a);\n    std::println(\"{}\", args.size());\n    std::println(\"{}\", forge::os::env(\"FORGE_AOT_TEST_VALUE\", \"unset\"));\n    return args.size();\n}\n";
    for (auto& [mode, cxx] : modes()) {
        purge();
        std::string exe = compile(src, "args.fs", mode, cxx);
        Run a = run_binary(exe, {"one", "two words", "--flag", "--forge-looks-like-ours"}, {{"FORGE_AOT_TEST_VALUE", "from env"}});
        CHECK_MSG(a.code == 4 && a.out == "[one]\n[two words]\n[--flag]\n[--forge-looks-like-ours]\n4\nfrom env\n", "args: %d\n%s%s", a.code, a.out.c_str(), a.err.c_str());
        Run b = run_binary(exe, {"--forge-args", "--forge-timeout=5", "x"});
        CHECK_MSG(b.code == 2 && b.out == "[--forge-timeout=5]\n[x]\n2\nunset\n", "--forge-args: %d\n%s%s", b.code, b.out.c_str(), b.err.c_str());
        Run c = run_binary(exe);
        CHECK(c.code == 0 && c.out == "0\nunset\n");
    }
}

TEST(aot_semantics_embed_integrity) {
    if (sweep_only) return;
    std::string exe = compile("int main() { std::println(\"intact\"); return 0; }\n", "t.fs", script::aot_mode::embed, "");
    CHECK(!exe.empty());
    if (exe.empty()) return;
    CHECK(run_binary(exe).out == "intact\n");
    std::string bytes = slurp(exe);
    auto pos = bytes.rfind("intact");
    CHECK(pos != std::string::npos);
    if (pos == std::string::npos) return;
    bytes[pos] = 'I';
    const std::string bad = exe + ".damaged";
    { std::ofstream f(bad, std::ios::binary); f << bytes; }
    std::error_code ec; fs::permissions(bad, fs::perms::owner_all, ec);
    Run r = run_binary(bad);
    CHECK_MSG(r.code == 70 && r.out.empty() && has(r.err, "checksum"), "damaged payload: %d %s%s", r.code, r.out.c_str(), r.err.c_str());
}

TEST(aot_generated_source_is_deterministic) {
    if (sweep_only) return;
    for (auto& f : corpus(false)) {
        const std::string name = f.generic_string(), src = slurp(f);
        std::string a, b;
        try { a = script::aot_translate(src, name); b = script::aot_translate(src, name); }
        catch (const script::aot_error&) { continue; }   // refused scripts are covered by the sweeps
        CHECK_MSG(a == b && !a.empty(), "%s: two translations differ", name.c_str());
        CHECK_MSG(a.find(work_dir()) == std::string::npos, "%s: generated source mentions a temporary path", name.c_str());
    }
    // Options that change the baked policy change the output; nothing else (time, host, output path) does.
    script::aot_options o; o.run.allow_net = false;
    const std::string src = "int main() { return 0; }\n";
    CHECK(script::aot_translate(src, "x.fs", o) != script::aot_translate(src, "x.fs"));
    script::aot_options p; p.opt_level = "3"; p.keep_source_dir = work_dir(); p.cxx = "whatever";
    CHECK(script::aot_translate(src, "x.fs", p) == script::aot_translate(src, "x.fs"));
}
