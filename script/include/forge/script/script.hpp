#pragma once
#include <stdexcept>
// forge::script — the FORGE managed scripting runtime (see docs/scripting.md).
// ARC + backup cycle collector, RAII handles, hybrid tree-walker / verified bytecode VM, written in safe
// ISO C++23 (no raw owning pointers, no unsafe escape hatch in the language).
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace forge::script {

struct options {
    std::uint64_t max_steps = 0;        // 0 = unlimited
    std::uint64_t timeout_ms = 0;       // whole-run wall clock; 0 = unlimited
    std::uint64_t heap_bytes = 0;       // managed heap cap; 0 = forge::mem budget only
    int max_depth = 800;
    bool use_vm = true;                 // compile hot functions and whole scripts to bytecode
    bool debug = false;                 // race detector + lock-order / stall deadlock detector (slower)
    std::uint64_t deadlock_ms = 1000;   // debug: how long every thread may be blocked before DeadlockError
    int jit_threshold = 2;              // calls before a function is compiled
    bool allow_fs = true, allow_net = true, allow_proc = true, allow_env = true;
    std::vector<std::string> argv;      // visible to scripts as `args`
    std::function<void(std::string_view)> out;   // print(); default stdout
    // AI-assisted commands (ai.*). `ai` completes (system, prompt) -> text; null = use FORGE_AI_ENDPOINT
    // (OpenAI-compatible /v1/chat/completions over http) if set. `confirm` is shown AI-drafted code and
    // returns true to run it; null = AI code never runs (unless the script passes its own confirm callback).
    std::function<std::string(const std::string& system, const std::string& prompt)> ai;
    std::function<bool(const std::string& code)> confirm;
};

struct result {
    bool ok = true;
    std::string value;                  // repr of the result (eval / REPL); empty for nil
    std::string error_type, error;      // e.g. "TypeError", "TypeError: ... \n  at f (x.fs:3)"
    int line = 0;
    int exit_code = 0;                  // int main()'s return value / std::exit(n); 1 on an uncaught error
};

class engine {
public:
    explicit engine(options o = {});
    ~engine();
    engine(const engine&) = delete; engine& operator=(const engine&) = delete;

    result run_file(const std::string& path);
    result run(std::string_view src, const std::string& name = "<string>");   // VM path if enabled
    result eval(std::string_view src);                                       // tree-walker, keeps globals (REPL)
    std::string disassemble(std::string_view src);                           // bytecode listing of a program
    std::string type_of(std::string_view expr);
    void interrupt();                                                          // async-safe: cancels the running script
    void reset();                                                              // drop all globals
    int repl();                                                                // interactive loop on stdin/stdout
    std::string stats();

    struct impl;
private:
    std::unique_ptr<impl> p_;
};

// Write once, emit anywhere (docs/scripting.md): translate a script in the portable subset into a
// standalone native script for `target` (a shell backend name/alias, "auto" = host default, or "ast").
// Throws emit_error with the offending line when the script uses something the target can't express.
struct emit_error : std::runtime_error { int line; emit_error(int l, const std::string& m) : std::runtime_error(m), line(l) {} };
std::string emit(std::string_view src, const std::string& file, const std::string& target);
std::vector<std::string> emit_targets();      // backends that have an emitter
std::string host_shell();                     // default shell for this OS ("pwsh"/"cmd" on Windows, "sh" elsewhere)

} // namespace forge::script
