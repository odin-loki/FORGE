// forge script --aot code generator (script/src/aot_codegen.cpp): any script the parser accepts must translate to
// C++ or be refused with an aot_error. Nothing is executed and no compiler runs, so hostile inputs cannot loop or
// touch the host; a crash, a hang or any other exception is a finding. Translation must also be deterministic.
//
// Differential mode (slow, opt-in): FORGE_FUZZ_AOT_DIFF=embed or =<C++ compiler> also builds each accepted input
// and compares the binary's stdout and exit status with the interpreter's. Both run with every capability off, a
// step limit and a time limit, so inputs cannot reach the file system, the network or other processes.
#include "fuzz.hpp"
#include <forge/script/aot.hpp>
#include <forge/script/script.hpp>
#include <forge/os/fs.hpp>
#include <forge/os/os.hpp>
#include <forge/os/process.hpp>
#include "parser.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace {
[[noreturn]] void finding(const char* what, std::string_view src, const std::string& detail) {
    std::fprintf(stderr, "fuzz_script_aot: %s\n%s\n--- input\n%.*s\n", what, detail.c_str(), int(src.size()), src.data());
    std::abort();
}

void differential(std::string_view src, const std::string& how) {
    namespace fs = std::filesystem;
    forge::script::options o;
    o.allow_fs = o.allow_net = o.allow_proc = o.allow_env = false;
    o.os_sandbox = forge::script::options::os_sandbox_mode::off;
    o.max_steps = 200000; o.timeout_ms = 5000; o.heap_bytes = 64u << 20;
    std::string want_out;
    forge::script::options io = o; io.out = [&](std::string_view s) { want_out += s; };
    if (how != "embed") io.jit_threshold = 1;   // every function on the VM from its first call: native code counts steps like the VM
    int want_code;
    { forge::script::engine e(io); want_code = e.run(src, "<fuzz>").exit_code; }
    static const std::string dir = [] { auto t = forge::os::fs::make_temp_dir("forge-fuzz-script-aot-"); return t ? t.value : std::string("."); }();
    struct Cleanup { ~Cleanup() { std::error_code ec; if (dir != ".") fs::remove_all(dir, ec); } };
    static Cleanup cleanup_at_exit;   // the directory (program, precompiled header) goes away when the fuzzer exits
    forge::script::aot_options ao; ao.run = o;
    ao.mode = how == "embed" ? forge::script::aot_mode::embed : forge::script::aot_mode::native;
    if (how != "embed") { ao.cxx = how; ao.opt_level = "0"; ao.pch_dir = (fs::path(dir) / "pch").string(); }
    const std::string exe = (fs::path(dir) / "prog").string();
    try { forge::script::aot_compile(src, "<fuzz>", exe, ao); }
    catch (const forge::script::aot_error& e) {
        if (want_code == 2 && e.exit_code == 2) return;   // refused by both
        finding("the interpreter ran this script but AOT refused it (or the reverse)", src, e.what());
    }
    if (want_code == 2) finding("AOT compiled a script the interpreter refuses", src, "");
    forge::os::spawn_spec sp; sp.argv = {exe}; sp.timeout_ms = 60000;
    auto r = forge::os::spawn_wait(sp);
    { std::error_code ec; fs::remove(exe, ec); }
    if (!r) finding("could not run the AOT binary", src, r.err.what);
    // Step and time limits fire at slightly different points of output only if the program is cut off mid-run by
    // the wall clock; a deterministic step limit cuts both at the same statement.
    if (r->out != want_out || r->exit_code != want_code)
        finding("AOT binary and interpreter disagree", src, "--- aot (" + std::to_string(r->exit_code) + ")\n" + r->out + "--- interpreter (" + std::to_string(want_code) + ")\n" + want_out);
}
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    std::string_view src(reinterpret_cast<const char*>(data), size);
    try { (void)forge::script::parse_program(src, "<fuzz>"); }
    catch (const forge::script::syntax_error&) { return 0; }
    forge::script::aot_options o;
    o.run.static_check = false;   // translate everything the parser accepts, not just what the checker lets run
    std::string a, b;
    try { a = forge::script::aot_translate(src, "<fuzz>", o); b = forge::script::aot_translate(src, "<fuzz>", o); }
    catch (const forge::script::aot_error&) { return 0; }   // refused with a diagnostic: fine
    catch (const std::exception& e) { finding("aot_translate threw something other than aot_error", src, e.what()); }
    if (a != b) finding("aot_translate is not deterministic", src, "");
    if (a.find("int main(int argc, char** argv)") == std::string::npos) finding("generated source has no entry point", src, "");
    static const std::string diff = [] { const char* e = std::getenv("FORGE_FUZZ_AOT_DIFF"); return std::string(e ? e : ""); }();
    if (!diff.empty()) differential(src, diff);
    return 0;
}
