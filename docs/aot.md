# Ahead-of-time compilation of scripts (`forge script --aot`)

`forge script --aot <out> script.cpp` turns a [ForgeScript](scripting.md) program into one executable that runs
without `forge` and without the script file. The executable behaves like `forge script script.cpp`: same output,
same exit status, same errors, same capability checks, limits and OS sandbox.

Code: `script/src/aot_codegen.cpp` (translator), `aot_compile.cpp` (driver), `aot_rt.cpp` (runtime),
`script/rt/forge_aot_rt.hpp` (header for generated code), `script/rt/stub_main.cpp` (runner stub). Public header:
`forge/script/aot.hpp`. This is AOT for scripts only; model plans are compiled with `forge aot` ([aot-models.md](aot-models.md)).

Status: both modes are implemented and tested on Linux x86-64 (see *Tests*). Everything under *Not verified* has
been written but never run.

## Usage

```
forge script --aot <out> [options] <script.cpp>
forge script --aot <out> [options] -e '<code>'
```

Options go before the script file, like every `forge script` option.

| option | effect |
|---|---|
| `--aot OUT` | write the executable `OUT` instead of running the script |
| `--aot-mode native\|embed\|auto` | which kind of binary (default `auto`: native when a C++ compiler and the runtime are found, else embed; the reason is printed on stderr) |
| `--cxx COMPILER` | C++ compiler for native mode. Default: `$CXX`, then `c++`, `clang++`, `clang++-23`, `g++` on `PATH` (Windows: `cl` first) |
| `-O0`, `-O1`, `-O2` (default), `-O3`, `-Os`, `-Oz`, `-Og` | optimisation level passed to the compiler; `-Og` also keeps debug information |
| `--target TRIPLE` | cross-compile: passed to the compiler as `--target=TRIPLE` (see *Cross-compiling*) |
| `--aot-runtime DIR` | directory holding `forge-aot-runtime.cfg` (see *The runtime library*) |
| `--aot-keep-source DIR` | native: also write the generated `<out>.aot.cpp` and the compiler command `<out>.aot.cmd` to `DIR` |
| `--aot-pch DIR` | native, clang and GCC: keep a precompiled runtime header in `DIR` and reuse it (also `$FORGE_AOT_PCH_DIR`) |
| `--aot-cxx-flag FLAG` | extra compiler or linker argument (repeatable) |
| `--aot-stub FILE` | embed: runner stub to copy (default: `forge-script-run` from the runtime directory or next to `forge`, else `forge` itself) |
| `--aot-allow-widen` | let run-time options widen the baked policy (see *Security model*) |
| `--no-fs`, `--no-net`, `--no-proc`, `--timeout MS`, `--max-steps N`, `--heap BYTES`, `--mem-limit SIZE`, `--os-sandbox MODE`, `--policy FILE`, `--no-check`, `--strict-check`, `--debug` | baked into the binary as its policy |
| `--module-dir DIR` | extra directory searched for statically imported modules |

`forge script --aot` prints one line on stderr saying what it wrote and how. Exit status: 0 on success; 2 for a
syntax error, a script refused by the static checks or a bad command line; 1 when the tool chain failed (no
runtime, compiler error) in a forced mode.

```
$ forge script --aot hello examples/scripts/hello_emit.cpp
forge script --aot: wrote hello (native: /usr/bin/c++ -O2, runtime /usr/lib/forge/aot; 0 unboxed int local(s))
$ ./hello
```

From C++: `forge::script::aot_compile(src, file, out, options)` and `aot_translate(src, file, options)` (the
generated C++ only) in `forge/script/aot.hpp`. `aot_supported()` is false on iOS and Android, where no compiler
can be started; `aot_compile` throws there. A program built elsewhere still runs its payload on those systems.

## The two modes

### native

1. The script is parsed, resolved and statically checked exactly as for a run. A syntax error or a static-check
   error stops here with the same text and exit status 2.
2. Every function, method, lambda and the top-level code is translated to one C++ function over the
   interpreter's own `Value` type. The translation follows the bytecode compiler instruction for instruction: each
   VM instruction becomes a call of the same `Interp` operation the VM calls (`binary`, `get_field`, `get_index`,
   `call`, `iterate`, `copy_value`, ...), so bounds checks, overflow checks, moved-from checks, borrow checks,
   reference counting, destructors and the exception model are the interpreter's code, not a second copy.
3. The generated source is compiled with the host C++ compiler and linked against `forge_script_rt`, which is the
   whole script runtime (`forge_script` and the static libraries it needs) in one archive.
4. At start-up the binary parses its embedded copy of the source with the same parser, checks a fingerprint of
   the tree against the one the code generator saw, and attaches each compiled function to its declaration. Calls
   then enter native code instead of the tree-walker or the VM.

Typed fast paths, all with the interpreter's semantics:

- `int`/`int` and number/number arithmetic and comparisons are inlined; every error case (overflow, division by
  zero, shift count) falls back to `Interp::binary`, which raises the usual error.
- **Unboxed integer locals.** A local or by-value `int` parameter is kept in a C++ `std::int64_t` when the code
  generator can prove it only ever holds an int: every initialiser and assignment is integer arithmetic over
  literals and other proven locals (or, for a declared `int`, a checked conversion), and nothing else can reach
  the slot (it is not captured by a lambda, bound to a `T&`, moved from, a loop variable of a range-`for`, a
  `catch` variable, or assigned inside a `try` block). Arithmetic on such locals is checked for overflow exactly
  as in the interpreter. Top-level variables are globals and are never unboxed. The compile line reports how many
  locals were unboxed.

The binary still contains the interpreter: `eval`, `forge::import` of a file found at run time, default-argument
expressions, class member initialisers, `ProcessPool` tasks, `forge::remote_call` (which ships source text) and
reflection all need it.

### embed

No compiler is needed. The program (source, bundled modules, baked policy) is appended to a copy of a runner stub,
followed by a 60-byte trailer:

```
<stub executable> <payload> "FORGEAOT" u32 version u64 payload_len u8[32] BLAKE3(payload) "TOAEGROF"
```

At start-up the stub looks for the trailer at the end of its own file, checks the length and the BLAKE3 checksum
(a damaged payload is refused with exit status 70) and runs the program on the bytecode VM, exactly like
`forge script`. The stub is `forge-script-run` (built by CMake, installed in the AOT runtime directory); the
`forge` binary itself is also a valid stub and is used when no `forge-script-run` is found.

The payload holds the script **source**, not `.fsc` bytecode: `.fsc` is a parse cache, and the binary re-runs the
static checks at start-up with the capabilities in force for that run, which needs the source. The same payload
is compiled into a native-mode binary as a byte array.

Appending data works on ELF and PE. On Mach-O it invalidates the code signature, and Apple Silicon refuses to run
such a file: `forge script --aot` re-signs the output ad hoc with `codesign -s - -f` when `codesign` is on `PATH`
and says so; otherwise sign it yourself. See *Not verified*.

### Imports and plugins

- `#include "file"` and `import "name"` with a literal name are resolved at compile time with the interpreter's
  search order (the importing script's directory, the main script's directory, `--module-dir`,
  `$FORGE_SCRIPT_PATH`, `~/.forge/modules`, the current directory). The module sources are bundled into the
  binary, and in native mode their functions are compiled like the main script's. A module that is not found at
  compile time is looked for at run time, as the interpreter would, and raises `ImportError` if it is still missing.
- `forge::import(expr)` and any module not bundled are read from disk at run time and run on the VM. Relative
  names resolve against the main script's compile-time directory as written on the command line (so against the
  current directory of the run) and the rest of the usual search path.
- Native plugins (`.so`, `.dylib`, `.dll`) are never bundled. They are loaded from disk at run time under the
  interpreter's rules: refused unless `fs`, `net`, `proc` and `env` are all on, and subject to `integrity.plugins`
  under a security policy.

## What is and is not translated

Native mode translates every node kind the parser produces: declarations, assignments (including compound and
indexed), `if`, `while`, `do`/`while`, `for(;;)`, range-`for` by value and by reference, `switch` (the parser
lowers it to loops), `break`/`continue`/`return`, `try`/`catch`/`throw`/`throw;`, functions, lambdas, classes,
templates (type-erased by the parser), calls, member access, indexing, list and map literals, `?:`, `&&`, `||`.

Run by the embedded tree-walker inside a native binary (correct, but not faster than the interpreter):

| construct | why |
|---|---|
| a call that binds an lvalue argument to a non-const `T&` parameter (`void inc(int& x)`; `inc(n)`) | the write-back protocol lives in `Interp::call_writeback`; only the argument evaluation and the write-back are interpreted, the callee is native |
| `std::move(x)` | `Interp::move_from` places the moved-from marker |
| `Base::method(...)` from a derived method, base-constructor delegation | `super` lookup in `Interp::eval` |
| default-argument expressions, class member initialisers | evaluated by `Interp::call_function` / `instantiate` |
| `import` / `#include "file"` statements | `Interp::import_module` (the imported functions themselves are native) |
| `try` with a `finally` block, `using`, `defer`, string interpolation, shell blocks, `is`, `spawn`/`await` nodes | node kinds the current parser never produces; kept for trees from other front ends |

The compile line lists the constructs it left to the tree-walker. There is currently **no construct that native
mode refuses**: anything the translator does not know is handed to the tree-walker on the same node, never guessed
at. Two limits make native mode fail at compile time with a diagnostic naming the file and line (`auto` mode then
falls back to embed):

- statements or expressions nested more than 180 levels deep (MSVC's block-nesting limit is 256);
- a tree the translator cannot make sense of (`malformed ... node`), which a parsed script never produces.

Differences from `forge script` that remain:

- The interpreter runs a function's first call on the tree-walker and later calls on the VM (`jit_threshold`).
  Native code always follows the VM's rules. Where the two interpreter paths differ (the exact statement at which
  a step limit fires, whether block locals are released on `break`) a native binary behaves like the VM. One
  exception: the moved-from `BorrowError` names the variable, as the tree-walker's does.
- The program's name in stack traces is the path given at compile time, not the absolute path `forge script`
  prints.
- `--tree`, the `.fsc` cache options, the REPL, `--emit` and `--check` do not exist in a binary.
- `--mem-limit`: the interpreter part (managed heap cap, default memory cap of child processes) is baked and
  enforced. The OS-level cap `forge script` also puts on its own process is CLI code and is **not** applied by an
  AOT binary.
- `forge::ask` and related calls have no in-process model (`--ask-model`): the runtime library does not link the
  inference module. An OpenAI-compatible endpoint in `$FORGE_AI_ENDPOINT` is used as in the interpreter.
- Scripts cannot read stdin in any mode; a binary leaves stdin untouched, and children started by the script get
  what the interpreter gives them.

## Running an AOT binary

Arguments are the script's arguments (`forge::args()`), with one exception: **leading** arguments that start
with `--forge-` are run-time options for the binary and are not passed on. `--forge-args` ends them, so a script
argument that starts with `--forge-` goes after it.

| run-time option | environment | effect |
|---|---|---|
| `--forge-no-fs`, `--forge-no-net`, `--forge-no-proc`, `--forge-no-env` | `FORGE_AOT_NO_FS`, `..._NO_NET`, `..._NO_PROC`, `..._NO_ENV` (any value but empty or `0`) | turn a capability off |
| `--forge-timeout=MS`, `--forge-max-steps=N`, `--forge-heap=BYTES`, `--forge-mem-limit=SIZE` | `FORGE_AOT_TIMEOUT`, `FORGE_AOT_MAX_STEPS`, `FORGE_AOT_HEAP`, `FORGE_AOT_MEM_LIMIT` | whole-run limits |
| `--forge-os-sandbox=off\|best-effort\|require` | `FORGE_AOT_OS_SANDBOX` | OS enforcement of the capability flags |
| `--forge-policy=FILE` | the usual policy search (`$FORGE_SECURITY_POLICY`, ...) | FORGE Security Standard policy |
| `--forge-debug`, `--forge-strict-check` | | debug detectors; refuse on static-check warnings |
| `--forge-allow-fs`, `--forge-allow-net`, `--forge-allow-proc`, `--forge-allow-env`, `--forge-no-check` | | widen the baked policy; refused unless the binary was compiled with `--aot-allow-widen` |
| `--forge-aot-info` | | print the mode, the baked policy and whether widening is allowed, then exit |
| `--forge-args` | | end of run-time options |

Values may also follow as the next argument (`--forge-timeout 500`). An unknown `--forge-` option is an error
(exit status 2) so a typo never silently becomes a script argument.

Exit status: the script's (`main()`'s value, `std::exit(n)`, 1 for an uncaught error with the error on stderr, 2
when the static checks refuse the script under the capabilities of this run); 2 for a bad run-time option; 70 for
a damaged payload or a binary whose compiled code does not match its runtime.

## Security model

An AOT binary enforces what `forge script` enforces, with the same code:

- **Capabilities** (`fs`, `net`, `proc`, `env`, `reflect`) are checked by the built-ins at the call, against the
  running interpreter state; `forge::sandbox` and `forge::limit` blocks work unchanged. Native code reaches
  built-ins only through `Interp::call`, so there is no path around a check.
- **Static checks** run at compile time with the baked capabilities and again at every start-up with the
  capabilities of that run. A script whose direct calls are denied is refused before it starts (exit status 2).
- **OS sandbox.** When `fs`, `net` or `proc` is off for the run, the engine applies the OS sandbox to the process
  before any script code runs, as in `forge script` ([security.md](security.md)).
- **Limits**: `--timeout`, `--max-steps`, `--heap` and the call-depth limit are the engine's. Native code calls
  `Interp::step()` at every statement and loop iteration, where the interpreter does, so time limits,
  cancellation (Ctrl-C), step limits and destructor timing are the same.

**The baked policy cannot be widened at run time.** What was given to `forge script --aot` is stored in the
payload. Run-time options and environment variables are applied on top and can only narrow: a capability that is
off stays off, a limit can be lowered but not raised or removed, the OS sandbox mode can be strengthened
(`off` < `best-effort` < `require`) but not weakened. A widening request is an error (exit status 2), not a silent
no-op. `--aot-allow-widen` at compile time removes this rule for that binary; `--forge-aot-info` shows which kind
you have.

**Security policy (FSS).** `--policy FILE` at compile time bakes the policy text into the binary. A binary with a
baked policy uses that policy and nothing else: the host's policy search is not consulted and `--forge-policy` is
refused (unless widening is allowed). A binary without one does what `forge script` does: `--forge-policy`, then
`$FORGE_SECURITY_POLICY`, `./forge-security.json`, the config directory, `$FORGE_SECURITY_PROFILE`. With an active
policy the capabilities come from the policy, file / network / process / environment built-ins are scoped by it
and `security::lockdown` is applied before the script runs. `integrity.scripts` is checked on **the executable**:
`forge sign app --key k.key` writes `app.sig`, and a binary without a valid, trusted `app.sig` next to it refuses
to start with `IntegrityError` when the policy says `required`. The signature covers the program and every
bundled module; modules read from disk at run time need their own `.sig` as usual.

What the model does **not** give you:

- The payload checksum detects corruption, not tampering. Someone who can modify the file can replace the program,
  the policy or the code. Protect a binary like any executable: file permissions, a detached signature
  (`forge sign`), OS code signing.
- The source text is inside the binary, readable with `strings`. AOT is not obfuscation.
- The baked policy binds the binary, not the person running it: it says what this program may do, and stops a
  caller from talking the program into doing more. It does not stop the caller from running other programs.
- Interpreter checks are checks in code. As in `forge script`, the OS sandbox is the second line.

## The runtime library

Native mode needs three things, described by a `forge-aot-runtime.cfg` file: the headers generated code includes,
the library `forge_script_rt`, and the system libraries to link. The driver looks for the file in
`--aot-runtime DIR`, else `$FORGE_AOT_RUNTIME`, `<forge's directory>/../lib/forge/aot`,
`<forge's directory>/aot-runtime`, and finally the build tree `forge` was built in.

- CMake builds `forge_script_rt` (target of the same name) and writes `<build>/aot-runtime/forge-aot-runtime.cfg`
  pointing at the source tree's headers.
- `cmake --install` puts the runtime under `<prefix>/lib/forge/aot`: `libforge_script_rt.a`, `forge-script-run`,
  `forge-aot-runtime.cfg` and `include/` (`forge_aot_rt.hpp` and the interpreter's internal headers it builds on).
  The `forge/core` and `forge/os` headers come from the package's own include directory, which the installed
  file names relative to itself. An installed `forge` (in `<prefix>/bin`) finds the runtime there. The library is
  also part of the CMake package as `forge::script_rt`.

The file is `key=value` lines; `include`, `lib`, `cxxflag`, `ldflag`, `define` repeat; relative paths are relative
to the file. `ldflag` entries are what the build that produced the library linked (for example the absolute paths
of the OpenSSL libraries); edit them if the file is moved to a machine where those differ.

Generated code and runtime are versioned together (`abi=`, currently 1). A binary checks at start-up that the
code it was compiled from matches the program it carries, and refuses to run (exit status 70) if the runtime
library and the code generator came from different builds.

A native build spends most of its time parsing the runtime's headers. `--aot-pch DIR` keeps one precompiled header
per (compiler, flags, runtime) in `DIR`; use a directory only you can write, since a precompiled header is trusted
like source. A stale one is detected by the compiler, deleted and the build repeated without it.

## Cross-compiling

- **embed** cross-compiles by construction: pass `--aot-stub` a `forge-script-run` (or `forge`) built for the
  target. The payload is the same for every platform.
- **native**: `--target TRIPLE` is passed to the compiler (clang's `--target=`), or name a cross compiler with
  `--cxx`. The runtime library must be one built for that target: build FORGE for the target (the
  `forge_script_rt` and `forge-script-run` targets are enough), install it, and point `--aot-runtime` at its
  `lib/forge/aot`. The target's sysroot and linker are the compiler's business; pass what it needs with
  `--aot-cxx-flag` (`--sysroot=...`, `-fuse-ld=lld`).

## Reproducible output

For a given script text, file name (as written on the command line), baked options and bundled modules, the
generated C++ is byte-for-byte the same: names are numbered in program order and nothing depends on the time, the
host, the output path or addresses. `--aot-keep-source` shows it. Whether the executable is reproducible is up to
the compiler and linker; an embed-mode binary is a deterministic function of the stub and the payload.

## Performance

PERF_SECTION

## Tests

- `tests/test_script_aot.cpp` (ctest `test_script_aot`):
  - `aot_embed_sweep`: every script in `tests/script/*.fs`, `tests/aot/*.fs`, `fuzz/corpus/script/*.fs`,
    `examples/scripts/*.cpp` and `tests/data/emit/*.cpp` is built in embed mode; stdout, exit status and the
    uncaught-error text must equal the interpreter's. Scripts the interpreter refuses must be refused at compile
    time.
  - `aot_native_subset`: `tests/aot/*.fs` and two golden scripts in native mode with the first host compiler.
  - `aot_semantics_*`: uncaught errors and exit status; baked and run-time limits (timeout, steps, heap, call
    depth); capability denial at run time, baked, not widenable, widenable; static-check refusal at start-up; the
    OS sandbox mode cannot be weakened (Linux); native plugins loaded and refused; argument and environment
    passthrough; a damaged embed payload; deterministic generated source.
  - Native cases are skipped (with a message) when no host compiler is found.
- ctest `test_script_aot_native_sweep` (label `slow`): the whole corpus in native mode with every host compiler
  found among `clang++-23` and `g++` (`FORGE_AOT_TEST_CXX=a,b` overrides). Exit 77 (skipped) when there is
  none.
- `fuzz/fuzz_script_aot.cpp` (ctest `fuzz_script_aot`, label `fuzz`): everything the parser accepts must translate or be refused
  with a diagnostic, deterministically; nothing is executed. `FORGE_FUZZ_AOT_DIFF=embed` or `=<compiler>` adds a
  slow differential mode that builds each input and compares it with the interpreter under a no-capability policy.

## Not verified

Written to be portable, never run:

- **Windows** (MSVC, MinGW): the `cl` command line (`/std:c++latest /EHsc /MD ...`), `.lib` naming in the runtime
  file, PE trailer append, and whether the runtime library's CRT settings match `/MD` in every configuration.
- **macOS**: Mach-O trailer append plus `codesign -s - -f` (codesign may reject a file with data after the
  signature; if so embed mode needs a different container there), `-Wl,-S`.
- **FreeBSD, Android, iOS**: only the "compiling is not supported" path on Android / iOS is by construction; running
  an embed payload on those systems has not been tried.
- **Cross-compiling in native mode**: the `--target` pass-through exists; no cross runtime has been built or linked.
- **The install tree**: the install rules were written and inspected, not exercised by an AOT build from an
  installed prefix.
- **A baked or run-time FSS policy** and `integrity.scripts` on the executable: implemented from the `forge script`
  logic, no automated test.
- MSVC's limits on function size and nesting for very large scripts.
