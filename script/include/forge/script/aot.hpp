#pragma once
// forge::script ahead-of-time compilation (docs/aot.md): turn a script into a native executable.
//
//   native  translate the checked program to standalone C++23 that links against the AOT runtime library
//           (forge_script_rt: the interpreter's own value model, heap and built-ins) and compile it with the
//           host C++ compiler.
//   embed   no compiler needed: append the program and its baked policy to a copy of a runner stub
//           (forge-script-run, or the forge binary itself), which runs it on the bytecode VM at start-up.
//
// Both kinds of binary enforce the capabilities, limits and OS sandbox baked in at compile time. Run-time options
// (leading --forge-* arguments, FORGE_AOT_* environment variables) can narrow them; they can widen them only when
// the binary was compiled with allow_widen.
#include <forge/script/script.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace forge::script {

enum class aot_mode { automatic, native, embed };

struct aot_options {
    aot_mode mode = aot_mode::automatic;   // automatic: native when a C++ compiler and the runtime are found, else embed
    std::string cxx;                       // C++ compiler; empty = $CXX, then c++ / clang++ / clang++-23 / g++ / cl on PATH
    std::string target;                    // cross-compilation triple: passed as --target=<triple> (clang) with the runtime in runtime_dir
    std::string opt_level = "2";           // -O<level>: 0, 1, 2, 3, s, z
    std::string keep_source_dir;           // native: also write the generated C++ (and the compiler command) here
    std::string runtime_dir;               // directory holding forge-aot-runtime.cfg; empty = $FORGE_AOT_RUNTIME, then next to this
                                           // executable (../lib/forge/aot, ./aot-runtime), then the build tree this library came from
    std::string stub;                      // embed: runner stub to copy; empty = forge-script-run from the runtime dir or next to this
                                           // executable, then this executable itself when self_is_runner
    bool self_is_runner = false;           // this executable calls aot_run_embedded() first in main() (the forge CLI does)
    std::vector<std::string> cxx_flags;    // extra compiler / linker arguments (native)
    std::string pch_dir;                   // native, clang / GCC: keep a precompiled runtime header here and reuse it (several times faster
                                           // compiles). Empty = $FORGE_AOT_PCH_DIR, else none. Must be a directory only you can write.
    bool allow_widen = false;              // the binary lets run-time options widen the baked policy
    std::string policy_file;               // FSS policy to bake in (docs/scripting.md "Security policy"); empty = none
    std::vector<std::string> module_dirs;  // extra directories searched for statically imported modules
    // Baked policy: capabilities, limits, OS sandbox mode, static-check settings and debug mode come from here.
    // The script's static checks run at compile time with these capabilities; a script with errors is refused.
    options run;
};

struct aot_error : std::runtime_error {
    int line;                              // script line (0: not tied to a line)
    int exit_code;                         // 2: the script or the command is at fault; 1: the tool chain failed
    aot_error(int l, const std::string& m, int code = 2) : std::runtime_error(m), line(l), exit_code(code) {}
};

struct aot_result {
    aot_mode mode = aot_mode::embed;       // what was produced
    std::string output;                    // path of the executable
    std::string detail;                    // one line for the user: compiler and runtime used, or the stub copied
    std::string fallback_reason;           // automatic mode only: why native was not used (empty when it was)
    std::string compiler_output;           // native: what the compiler printed
};

// Whether this build can compile at all (false on iOS and Android, where spawning a compiler is not possible;
// binaries built elsewhere still run their payload there).
bool aot_supported();

// The C++23 source native mode compiles: deterministic for a given (source, file name, options). Throws aot_error
// naming the construct and its line when the script cannot be translated, and for syntax or static-check errors.
std::string aot_translate(std::string_view src, const std::string& file, const aot_options& o = {});

// Compile `src` (the script in `file`, used for messages, imports and as the program's name) to the executable
// `out`. Throws aot_error. In automatic mode a native failure that is not the script's fault falls back to embed.
aot_result aot_compile(std::string_view src, const std::string& file, const std::string& out, const aot_options& o = {});

// For runner executables (the forge CLI, forge-script-run): if this executable carries an embedded program, run
// it with these arguments and return its exit status; nullopt when there is none.
std::optional<int> aot_run_embedded(int argc, char** argv);

} // namespace forge::script
