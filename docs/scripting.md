# FORGE scripting design (`forge script`)

Status: **core runtime implemented** (C++23-subset language, tree-walker + verified bytecode VM, ARC +
cycle collector, exceptions, tasks/channels, shell calls, limits, sandbox, `forge script` CLI/REPL).
Automation (pipelines, file-watch triggers, schedules) and modules/native plugins implemented.
Still to come: AI commands, remote execution. See *Phasing* at the end.

## Purpose

A small, interpreted, dynamic language built into the `forge` binary. One script runs the same on
Windows, Linux and macOS, because built-ins go through the [OS wrapper](os-wrapper.md). When you need
a real shell, embedded blocks call it and give back structured results.

```
forge script build.cpp [args...]                  # run a file (int main() return value = exit status)
forge script -e 'std::println("{}", 6 * 7);'      # run a snippet
forge script                                      # REPL (`x + 2` without ';' prints the value)
forge script --tree | --timeout MS | --max-steps N | --heap BYTES | --no-proc | --no-fs | --no-net ...
```

## Language

Scripts are written in an **interpreted, memory-safe subset of ISO C++23**. A script is ordinary-looking
C++: top-level statements run first (like a REPL), then `int main()` if present. `#include <...>` lines
are accepted and ignored; `#include "util.cpp"` loads another script and brings its names in.

Not allowed (rejected by the parser with a message saying what to use instead): raw pointers (`T*`),
`new`/`delete`, address-of, `reinterpret_cast`/`const_cast`/C-style casts, pointer arithmetic, explicit
destructor calls. Everything else that a typical tool script needs is supported:

- Declarations: `int`/`long long`/`double`/`bool`/`char`/`std::string`/`auto`, `const`/`constexpr`,
  several declarators, brace/paren initialisation, structured bindings `auto [k, v] = ...`.
  Narrowing (`int x = 3.5;`) and assigning to a `const` are compile-time errors.
- Control flow: `if`/`else`, `while`, `do`/`while`, `for(;;)`, range-`for` (incl. bindings), `switch`
  with fall-through, `break`/`continue`, `try`/`catch`/`throw`/`throw;`, `catch (...)`.
- Functions with defaults, by-value (copied) and by-reference parameters, recursion, templates
  (type-erased), lambdas with `[=]`, `[&]`, `[x]`, `[&x]` captures and `mutable`.
- Classes and structs with value semantics, constructors with init lists and delegation to base
  constructors, `using Base::Base;`, virtual/override/pure virtual (abstract classes cannot be
  instantiated), static members, operator overloading (`operator+`, `==`, `<`, `<=>`, `()`,
  `= default` comparisons), `friend` free operators.
- `enum`/`enum class`, `namespace` (nested, `using namespace`), `typedef`/`using` aliases.
- Integer semantics match C++ (`/` truncates, `%` keeps the sign of the dividend) except that 64-bit
  overflow throws `OverflowError` instead of being UB; division by zero throws `ArithmeticError`.

Standard library surface (implemented natively, `std::` prefix optional after `using namespace std;`):
`std::cout`/`cerr` with `endl`, `boolalpha`, `fixed`, `setprecision`, `setw`; `std::print`/`println`/
`format` (full format specs); `std::string` (mutable, value semantics: `size`, `substr`, `find`,
`rfind`, `find_first_of`, `push_back`, `append`, `insert`, `erase`, `replace`, `starts_with`, ...);
`std::vector`, `std::map`/`unordered_map` (`operator[]` inserts a default), `std::set`/`unordered_set`,
`std::pair`/`tuple`/`get<N>`, `std::optional`; iterators (`begin`/`end`, `find`, `it - v.begin()`,
`it->first`); `<algorithm>`/`<numeric>`: `sort`, `stable_sort`, `reverse`, `find(_if)`, `count(_if)`,
`any_of`/`all_of`/`none_of`, `for_each`, `min/max_element`, `fill`, `iota`, `binary_search`,
`transform`, `accumulate`, and the `std::ranges::` forms; `<cmath>`, `<cctype>`, `to_string`, `stoi`,
`stod`, `numeric_limits`; exceptions `std::exception`, `runtime_error`, `logic_error`,
`invalid_argument`, `out_of_range`, ... with `what()`.

### Memory model

- **Value types** (`std::string`, containers, structs/classes held by value) are copied on
  copy-initialisation, by-value parameters and insertion into containers, exactly like C++.
- `std::make_shared<T>` / `std::shared_ptr` share one object (ARC); `std::weak_ptr` (`lock`,
  `expired`) does not keep it alive; the cycle collector reclaims `shared_ptr` cycles.
- `std::make_unique<T>` / `std::unique_ptr` cannot be copied (`BorrowError`); `std::move` transfers it.
- `std::move(x)` leaves `x` moved-from: any later read of `x` throws `BorrowError` (use-after-move is a
  checked error, not UB). Scalars are unaffected, as in C++.
- Dereferencing an empty pointer/optional or an `end()` iterator throws `NullPointerError` /
  `IndexError`; `at()`, `substr`, `erase` etc. are bounds-checked.
- All allocation is charged to `forge::mem`; exhaustion throws `OutOfMemoryError`.
- **Destructors (RAII).** `~T()` runs deterministically when the last reference goes away: block locals
  are destroyed at the closing brace in reverse declaration order (also on `break`, `return` and
  exceptions), `unique_ptr` on reset/scope exit, `shared_ptr` when the last owner drops. Order is
  most-derived destructor, then bases, then members. A constructor that throws destroys nothing.
  Destructors run on the owning thread at the next statement boundary, so they may print, call
  functions and throw (an exception from a destructor propagates from that statement). Prvalue
  initialisation (`G g("x");`, `G g = make();`) elides the copy as in C++17; default member
  initialisers (`G g{"m"};`) are evaluated for each object.
- **Borrows while iterating.** A range-`for` borrows its container; structural changes to it
  (`push_back`, `insert`, `erase`, `clear`, `resize`, inserting a new key with `operator[]`, ...)
  throw `BorrowError` instead of invalidating iterators. Changing elements in place is allowed.
- **References.** Non-const `T&` parameters (functions, methods, lambdas, `std::for_each`,
  `forge::shared::with`) and `for (T& x : c)` loops write through to the caller's variable, element or
  member, for scalars and strings as well as objects; `const T&` and by-value parameters never do.

### Shell calls, threads and FORGE built-ins

- `forge::sh(fmt, args...)`, `forge::bash`, `forge::pwsh`, `forge::cmd` run a command; each `{}` is
  replaced by the argument **quoted for that shell**. They return a result with `out`, `err`,
  `code`, `ok`, `ms` plus `lines()` and `json()`.
- `std::async(f, args...)` / `std::thread t(f, args...)` / `forge::spawn` run on another thread;
  `get()`/`join()` wait and rethrow. Arguments cross threads only deep-copied, frozen
  (`forge::freeze`), moved, or as `forge::shared<T>` (locked on every access: `with(fn)`, `load`,
  `store`). `forge::channel<T> ch(cap)` has `send`, `recv`, `close` and works in range-`for`.
- **Debug mode** (`forge script --debug`, `options::debug`) adds concurrency checks:
  - *race detector*: every mutable object belongs to the thread that created or received it (task
    arguments, channel messages and task results are adopted by the receiver); modifying one from
    another thread without holding its `forge::shared` lock raises `RaceError`;
  - *lock order*: nested `forge::shared::with` calls record lock-order edges; acquiring locks in an
    order that closes a cycle raises `DeadlockError` (before the threads can actually deadlock);
  - *stall detector*: if every script thread is blocked (channel send/recv, `get`/`join`, a shared
    lock) for `options::deadlock_ms` (default 1000 ms), the waiting thread raises `DeadlockError`.
  CI additionally builds the runtime with ThreadSanitizer (`-DFORGE_TSAN=ON`, job `linux-tsan`).
- `forge::args()`, `forge::fs::...`, `forge::os::...`, `forge::json::...`, `forge::gc::collect()`
  and the other built-in modules (see below) are reachable as `forge::<module>::<name>`.

Example:

```cpp
#include <vector>
#include <map>

struct Job { std::string name; std::string cmd; };

int main() {
    std::vector<Job> jobs{{"kernel", "uname -r"}, {"files", "ls / | wc -l"}};
    std::map<std::string, std::string> out;
    for (const auto& j : jobs) {
        auto r = forge::sh(j.cmd);                  // no {} placeholders: run as written
        if (!r.ok) { std::println(stderr, "{} failed: {}", j.name, r.err); return 1; }
        out[j.name] = r.lines().front();
    }
    for (const auto& [k, v] : out) std::println("{:>8}: {}", k, v);
    return 0;
}
```

### Execution model (hybrid)

- Scripts and modules are compiled to a **compact, verified stack bytecode** (NETMF uses a compact IL
  interpreter for the same reasons: small footprint, fast startup, no JIT, predictable on small
  devices). Bytecode is compiled in memory from the parsed, resolved program on each run.
- **`.fsc` cache** (`script/src/fsc.cpp`). `run_file()` can cache the parsed and resolved program (the
  input to the bytecode compiler: bytecode chunks reference AST nodes for delegated statements, so the
  tree is what is stored) and skip lexing, parsing and resolution on later runs. Enable it with
  `options::fsc_cache_dir`, `forge script --cache-dir DIR`, or `$FORGE_SCRIPT_CACHE`; off otherwise.
  - File: `<dir>/<key>.fsc`, key = BLAKE3(`"forge-fsc\0"` ‖ VM version ‖ canonical file name ‖ source).
    A changed source, name or VM version gives a different key, so stale files are never used.
  - Load checks: magic, VM version, the key recomputed from the current source, every length and count
    bounds-checked, node kinds range-checked, nesting capped at 4096, no trailing bytes. Any failure means
    the source is parsed normally; a bad cache can cost time but not change behaviour. The decoded chunks
    still go through the bytecode verifier.
  - Signing: with `options::fsc_sign_seed` / `--cache-sign-seed HEX` / `$FORGE_SCRIPT_CACHE_SIGN_SEED`
    (32-byte Ed25519 seed) the files written are signed over `"forge-fsc-v1"` ‖ key ‖ BLAKE3(payload). A
    signed file loads only if the signature verifies and the signer is in `fsc_trusted_keys` /
    `--cache-trust HEX` (the engine's own signing key is trusted automatically).
  - Unsigned files load only from a directory owned by the current user and not writable by group or
    other (POSIX), and never on Windows; `fsc_require_signed` / `--cache-require-signed` rejects them
    everywhere. That matters because the decoder trusts resolved slot numbers: anyone who can write the
    cache directory could otherwise feed the tree-walker a crafted tree.
  - `engine::cache_status()` reports `off`, `hit`, `miss (stored): <reason>` or `miss: <reason>`.
  - Bump `fsc::vm_version` whenever `Node`, `FnDecl`, `ClassDecl` or the compiler change meaning.
- The **REPL and `eval`** use a direct **AST tree-walker** path: no compile step, best error locations,
  instant feedback. Functions are compiled to bytecode lazily once called `jit_threshold` times (default 2, `script::options`;
  `script/src/runtime.cpp`), so hot REPL code also runs on the VM.
- Both paths share one value model, one object heap and one built-in table, so behaviour is identical;
  golden tests run every test script through both paths.
- The verifier checks stack depth, jump targets, constant/local indices and opcode operands before a
  chunk runs; malformed bytecode is rejected, never executed. No JIT (keeps it portable and safe).
- Rationale: a tree-walker alone is simple but 3-10x slower for loops and pipelines; bytecode alone
  makes the REPL and `eval` slower to start and harder to give good errors. Hybrid gets both.

## Managed runtime

### Object lifetime: automatic reference counting

- Every heap object carries an intrusive atomic refcount; values hold `forge::script::Ref<T>` (an
  intrusive smart pointer). Releasing the last reference frees the object immediately: no tracing GC,
  no pauses.
- **Cycles**: both mechanisms, layered.
  1. `weak` references in the language (`weak let parent = p`, `weak` fields); upgrading a dead weak ref
     yields `nil`. Library types (parent links, event subscriptions, caches) use weak refs by design.
  2. A **backup cycle collector** (synchronous trial deletion, Bacon-Rajan style) over container
     objects only (lists, maps, class instances, closures). It runs on memory-budget pressure, every N
     possible-root decrements, at REPL idle, or via `gc.collect()`. Strings, numbers and bytes cannot
     form cycles and are never scanned.
  - Choice: weak refs are the primary tool so most programs never need the collector; the collector
    is a safety net so a forgotten cycle leaks only until the next collection, not forever.
- Finalisation order: `dispose` runs before memory is released; objects freed by the cycle collector
  have `dispose` called but must not be resurrected (resurrection attempts are ignored and logged).

### Deterministic cleanup (RAII in the language)

- `Disposable` interface with `dispose()`. All OS-backed values (files, sockets, processes, mappings,
  locks, watchers, models, GPU buffers) are `handle`s that implement it and wrap a C++ RAII owner.
- `using f = fs.open("log.txt") { ... }` (and `using` declarations scoped to the block) call `dispose`
  on every exit path, including `throw`, `return` and `break`, in reverse order.
- Handles are also released when the last reference drops, so forgetting `using` cannot leak the OS
  resource; `using` just makes the timing explicit. Disposed handles throw `ObjectDisposedError`.
- `defer { ... }` runs code at block exit for non-object cleanup.

### Exceptions

- `try { } catch e: IOError { } catch e { } finally { }`, `throw Error("msg")`; user classes derive
  from `Error`. Exceptions carry message, type, script stack trace and optional `cause`.
- Built-in hierarchy: `Error` > `IOError`, `OSError` (with `errno`/`win32` code), `TypeError`,
  `IndexError`, `KeyError`, `OverflowError`, `ObjectDisposedError`, `ShellError`, `LimitExceededError`.
- OS errors (as in Python, all under `OSError`, so `catch (OSError e)` gets every one): `PermissionError`
  (and its subclass `SandboxViolationError`), `TimeoutError`, `OutOfMemoryError`, `CancelledError`, `DeadlockError`,
  `FileNotFoundError`, `FileExistsError`, `NotADirectoryError`, `IsADirectoryError`, `NotSupportedError`,
  `InterruptedError`, `BlockingIOError`, `ProcessLookupError`, `NoSpaceError`, `TooManyFilesError`, `BusyError`,
  `ReadOnlyFileSystemError`, `SignalError` (a fault caught by `fault_guard`, or a crashed process-pool worker), and
  `ConnectionError` with `ConnectionRefusedError`, `ConnectionResetError`, `BrokenPipeError`. The mapping from
  errno / GetLastError / WSA / NTSTATUS / Mach codes is in `forge/os/oserror.hpp`. `EPERM` is reported as
  `SandboxViolationError` only when the active OS sandbox policy denies that kind of operation (network, exec,
  file write); `EACCES` stays `PermissionError`. SIGINT/SIGTERM (Ctrl-C/Ctrl-Break/console close on Windows)
  raise `CancelledError` during a top-level run (`options::cancel_signals`, default on; the previous handlers
  are restored afterwards).
- C++ exceptions are mapped to script exceptions by `Interp::from_cpp` (`script/src/runtime.cpp`):
  `heap_exhausted`/`std::bad_alloc` -> `OutOfMemoryError`, `syntax_error` -> `SyntaxError`, `forge_exception`
  by `errc` (`IOError`, `ValueError`, `OSError`, `OutOfMemoryError`, else `Error`), `std::out_of_range` ->
  `IndexError`, `std::invalid_argument` -> `ValueError`, anything else -> `Error`. Built-ins turn failed
  `forge::os` results into script exceptions via their `ok(...)` helpers. Script `try`/`catch` sees these
  mapped errors, and `engine` (`script/src/engine.cpp`) catches everything at the embedding API, so no
  C++ exception leaves `engine::run`/`eval`. The VM itself has no separate catch layer; this is enforced at
  the interpreter and engine boundaries, not proven for every native call.
  `std::bad_alloc` maps to `OutOfMemoryError`.

### Threads and async

- `spawn f(x)` returns a `task`; `await t`, `await all(ts)`, `parallel(xs, fn, n)`, `channel<T>(cap)`
  for message passing, `lock`/`mutex` handles usable with `using`. `async fn` + `await` for I/O.
- Tasks run on the `forge::os` thread pool; async I/O uses the OS wrapper's event loop.
- Safety with ARC: refcounts are atomic (relaxed increment, acq-rel decrement). Objects are **owned
  by one task unless frozen or shared**: `freeze(x)` makes a deep-immutable value that is freely
  shareable; mutable objects passed to `spawn`/channels are moved (the sender's binding becomes
  invalid) or must be `shared` objects whose methods take an internal lock. This rules out data races
  in script code without a global interpreter lock.
- Cancellation is cooperative: `t.cancel()` raises `CancelledError` at the next await or safepoint.
- The cycle collector stops the world at safepoints only.

### Interop with `forge::os` and `forge::mem`

- Built-ins are thin C++ bindings over `forge::os` (fs, proc, net, time, env, registry) and return
  handles or plain values; native resources never appear as raw addresses.
- The managed heap uses a `forge::mem` allocator: every object allocation is charged to the script's
  memory budget (`FORGE_MEM_LIMIT`, `forge script --heap BYTES`, or `forge::limit({{"mem", bytes}}, fn)`). When an allocation
  would exceed the budget the runtime first evicts caches and runs the cycle collector, then throws a
  catchable `OutOfMemoryError` instead of being OOM-killed. A small reserve is kept so handlers and
  `finally` blocks can still run.
- `mem.usage()`, `mem.budget()`, `gc.stats()` expose live bytes, object counts and collector runs.

### Resource limits

`forge::limit(opts, fn)` runs `fn` with narrower limits and restores them afterwards. The keys are
`timeout` (wall-clock ms, raising `TimeoutError`), `cpu_s` (CPU seconds of the running thread, raising
`LimitExceededError`), `mem` (extra heap bytes, raising `OutOfMemoryError`), and `proc_mem` / `proc_cpu`
(OS limits for child processes started inside the block). Example:
`forge::limit(std::map<std::string, double>{{"timeout", 500}, {"cpu_s", 0.2}}, [] { work(); });`

Also implemented (`Limits` in `script/src/interp.hpp`; CLI `--timeout MS`, `--max-steps N`, `--heap BYTES`):
step count, wall-clock timeout, managed-heap cap, call depth (default 800, checked so the C++ stack is
safe) and per-child output size (64 MiB). Tested by `tests/script/limits.fs`. **Planned, not implemented:**
`limit {}` block syntax, string sizes such as `"2G"`, allocation-count, open-handle and child-count limits.

### Interpreter implementation rules (safe ISO C++23)

- ISO C++23 only, no compiler extensions; `script/` module, library `forge_script`.
- Ownership via `std::unique_ptr`, `std::shared_ptr`/`std::weak_ptr` where shared lifetimes are needed,
  and an intrusive `Ref<T>` for script objects. **No raw `new`/`delete`**, no owning raw pointers,
  no `malloc`/`free`.
- RAII for every resource (handles, locks, frames, temporary roots).
- No `reinterpret_cast`, no C-style casts, no `const_cast` removing constness; value variants use
  `std::variant`/tagged unions with checked access; `std::bit_cast` only for float/int bits.
- Bounds-checked access everywhere (`.at()`, `std::span` with checked helpers, hardened libstdc++/libc++
  assertions enabled in all builds); bytecode reads go through a checked reader.
- Errors via exceptions/`std::expected` internally; mapped to script exceptions as described under Errors.
- Enforced by clang-tidy (cppcoreguidelines, bugprone), `-Wall -Wextra -Werror`, ASan/UBSan/TSan CI
  jobs and fuzzing of the lexer, parser, compiler and verifier (`fuzz/fuzz_script.cpp`).

## REPL

- `forge script` with no file starts the REPL; line editing, history (`~/.forge/history`), multi-line
  input detected from unbalanced brackets, tab completion of names, members, paths and commands.
- Every expression result is printed with its type; `_` holds the last result.
- Commands: `:help`, `:type expr`, `:doc name`, `:load file`, `:reset`, `:time expr`, `:mem`
  (heap and budget), `:gc`, `:dis fn` (show bytecode), `:trace on|off`, `:quit`.
- Errors show the source line with a caret and the script stack; the session keeps running.
- Ctrl-C cancels the running statement (`CancelledError`) without killing the REPL.
- Each REPL session has its own heap and limits; `:reset` disposes everything deterministically.

## Shell blocks

```
let r = sh { ls -la | wc -l }      # also bash { }, zsh { }, ps { } / pwsh { }, cmd { }
print(r.out, r.err, r.code, r.ok)
let procs = pwsh { Get-Process | ConvertTo-Json }.json   # parsed into lists/maps
```

- Block bodies are passed verbatim; `${expr}` is substituted with shell-appropriate quoting.
- Result object: `out`, `err`, `code`, `ok`, `ms`, `.json` (parse), `.lines`.
- Shells are located via the OS wrapper; a missing shell raises a catchable error.
- `run(["git", "status"])` runs without any shell (no quoting issues).

## Write once, emit anywhere

`forge script --emit <shell> file.cpp [-o out]` translates a script into a standalone native script for that shell.
The output runs without FORGE. `--emit auto` (or `forge::host_shell()` at runtime) picks the host's shell:
`$FORGE_SHELL` if set, otherwise pwsh, then powershell, then cmd on Windows; mksh on Android; and sh everywhere else.
`--emit android` (alias `toybox`) emits for Android's mksh with a `/system/bin/sh` shebang. `--emit ast` dumps the parse tree.
`forge script --emit-targets` lists every backend that has an emitter.

```
$ forge script --emit bash -o hello.bash examples/scripts/hello_emit.cpp && ./hello.bash
$ forge script --emit pwsh -o hello.ps1  examples/scripts/hello_emit.cpp && pwsh -File hello.ps1
```

**Portable subset** (the same for every target):

- `int`, `bool`, `std::string`, `auto` locals, and a few `int`/`std::string` parameters on free functions, which may recurse.
- `+ - * / %` (C truncation), comparisons, `&& || !` (short-circuit), `?:`, `+=`, `++`, string `+`, and `.size()`.
- `if`/`else`, `while`, `for(;;)`, range-for over a `{...}` literal list, `break`, `continue`, `return`, and `std::exit`.
- `std::println`/`std::print` with `{}`, `std::cout << ... << std::endl`, and `std::cerr`.
- `os.env(name, default)`, `os.setenv`, `os.cwd()`, and `os.platform` (linux/windows/macos/freebsd/android/ios/other).
- `fs.exists/is_dir/is_file/mkdir/remove/write/append/read`.
- `forge::sh(tmpl, args...)` and `forge::shell(...)` results (`.out`, `.code`, `.ok`), with `{}` quoted for the target.

Anything else is a clear compile-time error naming the file and line. Examples: `x.cpp:1: emit bash: classes and structs can't
be expressed in bash`. Unsupported constructs include classes, lambdas, containers, exceptions, templates, and tasks.

| Target | Extension | Not expressible (emit error) |
|---|---|---|
| sh, bash, zsh, dash, ash, yash, ksh, mksh/android, oksh, posh, hush, osh | none | none |
| fish | `.fish` | none |
| pwsh / powershell (output avoids 7-only syntax) | `.ps1` | on Windows, `forge::sh` needs an `sh` on PATH |
| xonsh (plain Python) | `.xsh` | none |
| elvish | `.elv` | none |
| nu | `.nu` | functions that read or write script-level variables (nu `def` can't capture) |
| tcsh, csh | `.csh` | functions (only `int main()` is inlined), `fs.read`, `forge::sh` results (csh drops newlines) |
| cmd (golden-tested under wine) | `.cmd` | functions, `.size()`, string ordering, loops inside if/for blocks, break/continue in range-for, `fs.read`, `forge::sh` |
| rc (Byron rc), es | `.rc`, `.es` | string ordering. Integers go through `expr(1)`, and locals use dynamic-scope blocks |
| ysh | | no emitter (osh covers Oils) |

**Tests.** `emit_golden_all_shells` in `tests/test_script.cpp` emits `tests/data/emit/functions.cpp` and `flat.cpp` for every
installed backend. It runs each output with the real shell and requires exactly the interpreter's stdout and exit code. Shells that
are missing are skipped. ctest sets `FORGE_TEST_SHELL_PATH` from the `FORGE_TEST_SHELL_PATH` cache variable, which defaults to
`../opt/shells/bin` when that directory exists. **Golden-tested on Linux**: sh, bash, zsh, dash, ash (busybox), yash, fish, tcsh,
pwsh, xonsh, elvish, nu, ksh, mksh, oksh, osh, hush, rc and es, plus csh and powershell through tcsh and pwsh, and cmd under wine
(`batch.cpp`). **Untested**: posh, real csh, Windows PowerShell 5.1, and mksh as Android's /system/bin/sh.

**OS targets.**

- **Linux**: tested.
- **macOS and FreeBSD**: the POSIX, fish, zsh, nu, elvish, xonsh and pwsh output is the same code. `_platform` maps `uname -s`.
  This output has not been run on either OS.
- **Windows**: pwsh/powershell and cmd output. `os.platform` is `windows`.
- **Android**: `--emit android` gives mksh/toybox output. Untested.
- **iOS**: *interpreter only*. iOS has no user shell to run emitted scripts, so use the embedded interpreter (`bindings/apple`).
- **Cross-compile checks**: `ctest -L cross` syntax-checks the emitter and backends with
  `zig c++ -target x86_64-freebsd|aarch64-macos|x86_64-windows-gnu|aarch64-linux-musl`. These tests are registered only when zig
  is found. zig is not installed on the current box, so they have not been run here.

## Built-ins (from the OS wrapper)

`fs.*` (read, write, append, open, copy, move, glob, walk, match, symlink, chmod, touch, write_atomic,
temp_file, make_temp_dir, disk_usage, lock, remove, watch, watch_tree), `os.*` (env, cwd, platform, caps, user, dir,
release, uptime, loadavg, memory, is_tty, term_size, devices, clipboard), `proc.*` (spawn, kill, list), `net.*` (resolve, interfaces, port_open, http, parse_url), `service.*` (unit, install, start, stop, status, uninstall), `time.*`, `json.*`, `path.*`,
`str.*`, `re.*` (regex), `reg.*` (Windows).

## FORGE built-ins

- `model.load(path)`, `model.generate(prompt, opts)` (inference via `mind`).
- `gpu.devices()`, grid dispatch helpers.
- `mem.budget()`, `mem.usage()`, `mem.limit(size)` from `forge::mem`.
- `serve.start(opts)` / `serve.stop()`.

## Mobile and device built-ins (implemented)

Gated by the `env` capability (device access). Handles own their device/stream; methods copy the owner per
call, so disposing a handle while a task uses it is safe.

```cpp
auto dev = forge::mobile::device();              // or ({{"sim", trace_json}}), ({{"sim_file", p}}), ({{"null", true}})
std::println("{} {}", dev.backend(), dev.info()["form"]);
auto acc = forge::sensors::open("accelerometer", 100, dev);   // forge::sensors::list(dev) lists kinds
forge::channel<int> ch(64);
while (acc.pump(ch, 32, 50) == 0) {}            // pump on the interpreter thread; consumers recv from ch (tasks too)
auto e = ch.recv();                              // {"type":"sensor","kind":..., "t_ns":..., "v":[...]}
auto touch = forge::touch::open(dev);
for (auto g : touch.gestures(16)) std::println("{}", g["kind"]);   // tap, double_tap, long_press, swipe, pinch, rotate
```

Sensor handle: `poll(timeout)`, `pump(ch, max, timeout)` (never blocks; stops at channel capacity), `kind()`,
`dropped()`, `finished()`, `close()`. Touch handle: `poll`, `gestures`, `pump` (gestures), `finished`, `close`.
Device handle: `backend`, `display`, `power`, `connectivity`, `info`, `vibrate`, `notify`, `clipboard_get/set`,
`brightness`, `set_brightness`, `keep_awake`, `torch`, `open_url`, `permission`, `request_permission`,
`sim_log`, `close`. Unsupported features raise `NotSupportedError`; other failures raise `IOError`.

## Higher-level features (approved, all in scope)

1. **Parallel tasks** — `spawn fn`, `await`, `parallel(list, fn, n)`; thread pool from `forge::os`.
2. **Resource-limited runs** — `forge::limit(opts, fn)` with an options map (`timeout` ms, `cpu_s`, `mem` bytes, `proc_mem`, `proc_cpu`) (block syntax `limit {...} {}` is planned) applies to
   child processes (cgroups / Job Objects) and to script work via the `forge::mem` budget, so
   jobs fail cleanly instead of OOM-killing the box.
3. **Job pipelines** — declare steps with dependencies (`job build after fetch { }`), run as a DAG,
   skip up-to-date steps, cache results; a portable make/CI runner.
4. **File-watch triggers** — `watch "src/**/*.cpp" { |changes| ... }` with debounce.
5. **Scheduled tasks** — `every "10m" { }`, `at "02:00" { }`; optionally installed as an OS service.
6. **AI-assisted commands** — `ai "find the 5 largest files here"` asks a local model to draft a
   script, shows it, and runs only on confirmation; `ai.explain(r.err)` for failed commands.
7. **Remote execution** — `forge::remote("host:7000", source, opts)` runs on `forge rpc-serve
   --allow-scripts`, authenticated (token / Ed25519) and optionally over TLS. Implemented; see docs/rpc.md.
8. **Modules/plugins** — `import` from a search path, native plugins via `os::library`.
9. **Sandboxed runs** — run untrusted scripts with fs/net/proc capabilities off. ForgeScript enforces
   the flags twice: the interpreter raises `PermissionError`, and the engine applies the OS sandbox from
   `forge/os/sandbox.hpp` to the whole process (Linux/Android seccomp-BPF, FreeBSD Capsicum, OpenBSD
   pledge, macOS `sandbox_init`, Windows job object). Mapping: `--no-net` → deny sockets other than
   AF_UNIX; `--no-proc` → deny exec; `--no-fs` → deny file writes (reads are blocked by the interpreter
   only). It is on by default (`options::os_sandbox = best_effort`; `forge script --os-sandbox
   off|best-effort|require`), is one-way, and `engine::sandbox_status()` reports the mechanism or the
   failure.
   - **Per-block OS sandbox:** `sandbox({net: false, isolate: true}) { ... }` runs the block in a forked
     child that applies the narrower OS policy (only the capabilities the block turns off) and sends back
     the block's return value (JSON-representable values only), its `print` output and any exception
     (re-raised with its type). The child works on a copy of the heap, so assignments to captured
     variables, globals and objects inside the block are not seen afterwards. Without `isolate` a block is
     interpreter-only: an OS sandbox is process-wide and one-way, so it cannot be narrowed for a block and
     then widened again in the same process. `isolate` is POSIX only (Windows and iOS have no `fork`;
     there it raises `PermissionError` without running the block). Forking a process with other running
     threads copies only the calling thread: avoid `isolate` while tasks hold locks the block needs.
   - **rpc-serve:** each remote script runs in its own child (`forge script --rpc-child`, source on
     stdin) under `--script-timeout`, a `--script-mem-mb` memory cap (default 2048), and with
     `--os-sandbox require` whenever a `--script-no-*` flag is set, so the OS sandbox applies to remote
     scripts without touching the server's sockets. Scripts can now run concurrently.
   - **Windows:** a running process cannot drop fs/net access, so in-process enforcement is the job
     object (no child processes) and the interpreter checks the rest. `os::run_sandboxed` starts children
     in an AppContainer (no network capability unless allowed, writes only to the container folder, user
     files unreadable unless their ACL grants ALL APPLICATION PACKAGES) inside a kill-on-close job with UI
     limits; an exec-only policy uses a restricted token in the job. rpc-serve children on Windows use
     `best-effort`. Not yet run on Windows (cross-compiled with llvm-mingw only). See docs/security.md.

## Process pools (implemented)

`ProcessPool(workers, {fs, net, proc, env, timeout})` runs tasks in worker *processes* (`forge::os::process_pool`:
fork on Linux/macOS/FreeBSD/Android, spawn of the same executable on Windows; not available on iOS, where it raises
`NotSupportedError`). A task is ForgeScript source that defines `task(arg)`; each call runs it in a fresh engine in the
worker with a JSON-representable argument and result:

```
auto pp = ProcessPool(4);                                   // capabilities default to off
auto f = pp.submit("auto task(auto x) { return x * x; }", 7);
f.get();                                                     // 49; f.get(timeout_ms), f.done()
pp.map("auto task(auto x) { return x + 1; }", [1, 2, 3]);    // [2, 3, 4]
pp.shutdown();                                               // also pp.size(), pp.resize(n)
```

Workers never get a capability the calling script lacks, and the pool applies the matching OS sandbox to each
worker process (no network / no exec / no file writes unless enabled). Errors keep their type (`ValueError` raised
in the task is `ValueError` in the caller); a worker that crashes raises `SignalError` and is replaced. A task's
`print` output is replayed in the caller when the result is collected. Programs embedding the engine that use spawn
mode must call `forge::script::run_process_worker_if_requested(argc, argv)` first in `main()` (the `forge` CLI does).

## AI-assisted commands (implemented)

`script/src/ai.cpp`. Drafted code never runs without confirmation.

```cpp
std::println("{}", forge::ask("summarise this log", fs.read("build.log")));
auto cmd = forge::suggest_cmd("find the 5 largest files here");     // string only, never run
auto r = forge::ai_run("find the 5 largest files here");            // {command, ran, result}
auto c = forge::ai_eval("list files changed today");                 // drafts script code -> eval on yes
auto j = forge::ask_json("give {name, version} for this package", fs.read("package.json"));
std::println("{}", forge::explain(forge::sh("make").err));
```

- Backend: the host's `options::ai` callback, else `FORGE_AI_ENDPOINT` — any OpenAI-compatible
  `/v1/chat/completions` server, normally FORGE's own `forge-serve` (`FORGE_AI_MODEL`, `FORGE_AI_KEY`
  optional). Endpoint calls need the `net` capability; plain `http://` only.
- Confirmation: `{{"confirm", fn}}` per call, else `options::confirm`. `forge script` prompts
  `Run it? [y/N]` on the terminal. With no confirmer, `ran` is `false` and nothing executes.
- `ai_run` takes `{{"shell", "sh"|"bash"|"pwsh"|"cmd"}}`; it runs through `forge::<shell>` so sandbox
  caps and child limits still apply. Failures raise `AIError`.

## Automation (implemented)

C++-shaped library calls rather than new keywords (`script/src/automation.cpp`):

```cpp
auto p = forge::pipeline({{"cache", ".forge-cache"}});          // optional input-fingerprint cache file
p.job("gen", [&]() { return sh("./gen.sh").code; });
p.job("build", {"gen"}, [&]() { ... }, {{"inputs", std::vector<std::string>{"src/*.cpp"}},
                                         {"outputs", std::vector<std::string>{"out/app"}}, {"retries", 1}});
auto r = p.run({{"parallel", 4}});   // also: targets, keep_going, dry_run, force, raise
// r["ok"], r["order"], r["jobs"][name] = {status: ok|skipped|failed|cancelled|would_run, ms, value, error, error_type}

forge::watch("src/**/*.cpp", [&](auto changes) { ...; return true; },   // false stops
             {{"debounce_ms", 100}, {"count", 1}, {"for", "10m"}, {"ignore", ...}, {"polling", false}});

forge::every("10m", fn, {{"count", 3}});  forge::at("02:00", fn);  forge::cron("*/5 * * * *", fn);
auto s = forge::scheduler(); s.every("1h", f, {{"times", 2}}); s.cron("@daily", g); s.upcoming(); s.run({{"for", "1d"}});
forge::cron_next(expr[, unix_s]);  forge::duration_ms("1h30m");
```

- Pipelines: DAG with unknown-dependency and cycle checks (`PipelineError`); a failed job cancels its
  dependants, and without `keep_going` everything not yet started. A job is skipped when all outputs
  exist and are newer than its inputs (make rule) or its input fingerprint (path, size, mtime) matches
  the cache, unless a dependency re-ran or `force`. `parallel > 1` runs each ready wave as tasks (same
  deep-copy rules as `std::async`; handles and tasks held in globals are not copied into them).
  `TimeoutError`/`OutOfMemoryError` are never retried or swallowed.
- Watch: `tree_watcher` (inotify/FSEvents/ReadDirectoryChangesW or polling) with debounce; events are
  filtered by glob relative to the pattern's literal root.
- Schedules: run on the calling thread, sleeping in slices so script timeouts apply. Callbacks returning
  `false` stop their entry. Cron is 5-field POSIX (lists, ranges, steps, `@daily` etc.; day-of-month and
  day-of-week OR together when both are set) in local time. To survive reboots, install the script
  as an OS service with `service.install`.
- Callbacks are stored in the handle's props, which the cycle collector traces (handles are tracked),
  so a lambda capturing its own pipeline/scheduler does not leak.

## Modules and plugins (implemented)

```cpp
#include "util.cpp"            // splice another script's names in (C++ style)
import "geom" as g;            // module object: g.area(2, 3); finds geom.fs, geom.cpp, geom/mod.fs, geom/mod.cpp
import tools.strings;          // dotted form = tools/strings
auto m = forge::import("strs");                 // at run time
import "libhello_plugin.so" as hello;           // native plugin (also .dylib / .dll), or forge::plugin(path)
hello.add(2, 3.5); hello.VERSION;
forge::module_path(); forge::modules();
```

- Search order: the importing script's directory, the engine base dir, `FORGE_SCRIPT_PATH`
  (`:`/`;` separated; `;` on Windows), `~/.forge/modules`, then the current directory. Absolute paths
  are used as is. Modules load once per engine (cached by canonical path); import cycles raise
  `ImportError`.
- Native plugins use the C ABI in `script/include/forge/script/plugin.h`: export
  `forge_script_plugin_init(const forge_plugin_api*)`, register functions with `def(reg, name, min, max,
  fn, user)` and constants with `constant(reg, name, json)`. Arguments/results cross as JSON;
  a non-zero return raises `PluginError` with the message from `*out`. Example:
  `examples/script_plugin/hello_plugin.c` (built as `hello_plugin` by CMake and used by tests).
- Plugins run native code, so they are refused unless all capabilities (fs, net, proc, env) are on.
  Loaded libraries stay loaded for the life of the process.

## Reflection (implemented)

Runtime reflection in the spirit of C++26 reflection (P2996), exposed as ordinary builtins rather
than a `^^` operator, and identical in the tree-walker and the bytecode VM (classes are defined by
shared runtime code, and the builtins are natives called through `Interp::call`). Implementation:
`script/src/reflect.cpp`. Example: `docs/examples/script/reflection.cpp`.

**Metadata.** The parser records a `ClassMeta` for each `class`/`struct`. It holds the class
`[[attributes]]`, and for each data member the name, the C++-style type name, the type annotation,
`const`/`static`, access, attributes and line. Function declarations record parameter type names,
the return type, `const`/`static`/`virtual`, access and attributes. `ClassMeta` is held by
`shared_ptr` from the runtime `Class`, so it outlives the AST. An `enum` becomes a frozen map tagged
with `Map::enum_name` (via `forge::__enum`). Metadata memory comes from the script heap (`forge::mem`,
charged under "script"), like every other script object.

**API** (`forge::reflect::`; `forge::typeof` and `forge::type_info` are also top-level):

| builtin | result |
|---|---|
| `typeof(v)`, `kind(v)`, `type_info(v\|T)`, `find_type("Name")` | type name, kind, or an info map (name, kind, bases, fields, methods, attributes) |
| `fields(T)`, `statics(T)`, `methods(T)`, `bases(T)`, `is_base_of(B, D)`, `has(T, "m")` | member lists as maps `{name, type, const, static, access, attributes, line}` |
| `attributes(T[, "member"])`, `attributes(fn)` | attribute texts, e.g. `"key(\"x\")"` |
| `get(obj, "f")`, `set(obj, "f", v)` | read or write a field by name |
| `invoke(obj, "m", args...)`, `call("fn", args...)`, `construct("T"\|T, args...)` | call or construct by name |
| `enum_values(E)`, `enum_name(E, v)` | `[{name, value}]` in declaration order; value back to name |
| `symbols([module])` | names a module (or the globals) defines |
| `signature(fn\|T\|"name")` | `{name, kind, returns, params[{name, type, default, ref, out, const}], arity, variadic, const, static, virtual, access, attributes, file, line}` |

**Rules.** Reflection never bypasses the language:
- Private and protected members are listed, but `get`/`set`/`invoke` on them raise `AccessError`.
- `set` on a `const` member or a frozen object raises `TypeError`. `invoke` of a non-`const` method
  on a frozen object is refused too.
- `set` while the object or the field's value is borrowed (for example, during iteration) raises
  `BorrowError`, exactly like a normal assignment.
- `set` checks the value against the member's declared type. Enums are read-only.
- Mutation and dynamic dispatch (`set`, `invoke`, `call`, `construct`) need the `reflect`
  capability (on by default). `forge::sandbox(std::map<std::string, bool>{{"reflect", false}}, fn)` turns it off, and those
  calls then raise `PermissionError`. Read-only queries always work.

**Tests.** `TEST(reflection)` in `tests/test_script.cpp` runs every builtin and each error case
in both execution modes.

## Safety

- Scripts run with the user's privileges. Capability flags (`forge::sandbox`, `--no-*`) are checked by
  the interpreter only; there is no OS-level sandbox for scripts yet. See [security.md](security.md).
- AI-generated code is never run without confirmation.
- Remote execution is refused unless rpc-serve has authentication (`flow/include/forge/flow/rpc.hpp`).

## Testing

`tests/test_script.cpp` runs every snippet through both the VM and the tree-walker and fails on any divergence,
plus per-built-in cases, emit goldens (`tests/data/emit/`) and shell tests that skip when the shell is missing.
`tests/test_script_golden.cpp` runs every `tests/script/*.fs` on both paths against `<name>.out`
(`FORGE_GOLDEN_UPDATE=1` rewrites them). `fuzz/fuzz_script.cpp` fuzzes the lexer, parser, resolver, compiler
and verifier without executing anything.

## Phasing

All phases are in scope.

1. **Core runtime**: value model, ARC heap on `forge::mem`, weak refs, exceptions, lexer/parser,
   tree-walker, REPL, `run` + `sh/bash/zsh/ps/cmd` blocks, fs/os/json built-ins, `using`/`defer`.
2. **Bytecode VM**: compiler, verifier, lazy REPL compilation, dual-path golden tests, the `.fsc` cache (see Execution model);
   classes, structs, interfaces, optional annotations and generics-lite.
3. **Concurrency and limits**: tasks, async, channels, freeze/move sharing rules, cycle collector,
   resource limits and `OutOfMemoryError`, FORGE built-ins (model, gpu, mem, serve).
   Features 1-2 (parallel tasks, resource-limited runs).
4. **Automation**: features 3-5 and 8 (done) (job pipelines, file-watch triggers, scheduled tasks, modules
   and plugins).
5. **Advanced**: features 6, 7 and 9 (AI-assisted commands, remote execution once rpc-serve has
   auth/TLS, sandboxed runs).

## Static checks (implemented, compile-checked only; tests pending)

`forge script --check file.fs` (add `--json` for `[{"file","line","col","severity","id","message"}]`) runs the
static checker (`forge/script/check.hpp`, `script/src/check.cpp`) over the parsed and resolved AST without running
anything; exit 2 when there are errors. Every normal run also runs it first: `options::static_check` (default on,
`--no-check` turns it off) refuses a script with any **error** (exit 2, `StaticCheckError`); `options::strict_check`
(`--strict-check`) refuses on warnings too. Errors are claimed only when the flagged operation is certain to raise
or be refused whenever it is reached; anything uncertain is a warning. The runtime checks (BorrowError, RaceError,
DeadlockError, PermissionError) remain the backstop.

Suppress with `// forge-check: ignore` or `// forge-check: ignore[id,id]` on the line or the line above, or
`// forge-check: ignore-file[id]` for the whole file.

| id | severity | what |
|---|---|---|
| `use-after-move` | error on every path / warning on some path | flow-sensitive over the function's locals (branches, `&&`/`||`/`?:`, loops with a back-edge pass, try/catch) |
| `unique-copy` | error | copy-initialising from a `std::make_unique` local |
| `iter-mutation` | error for a list/map-literal or `list`/`map`/`vector`-typed local, else warning | size-changing method on the container a range-for is iterating |
| `capability` | error | `fs.*`, `net.*`, `proc.*`, `service.*`, `os.env/setenv/chdir/processes/kill`, `run()`, shell blocks when the run's `--no-fs/--no-net/--no-proc` or an enclosing `sandbox({fs: false, ...}, fn)` literal denies them |
| `pool-task` | error | `submit`/`map` with a literal task source that does not parse, does not define `task`, or whose `task` does not take one argument |
| `pool-json` | error | a lambda or shell value (or a literal containing one) passed as a ProcessPool argument |
| `thread-send` | warning | an argument to `forge::spawn`/`std::thread`/`std::async` that is not a literal, moved, frozen, const or `forge::shared` |
| `thread-capture` | warning | a thread lambda reading or writing a captured mutable local |
| `lock-order` | warning | two `forge::shared` names locked in both orders by nested `x.with(fn)` (static lock graph, 2-cycles) |
| `unreachable` | warning | statements after return/throw/break/continue |
| `unused-variable` | warning | local initialised from a literal and never read (names starting `_` exempt) |
| `shadow` | warning | local redeclaring a name from an enclosing block of the same function |
| `missing-return` | warning | function with a declared non-void return type that can fall off its end |
| `arg-count` | warning | call of a top-level script function (never reassigned) with too few/many arguments |
| `catch-order` | warning | a catch after `catch (...)` or after a built-in ancestor (e.g. `OSError` before `TimeoutError`) |
| `infinite-loop` | warning | `while (true)`/`for (;;)` with no break, return, throw or exit |
| `format` | warning | `std::format/print/println` literal with a placeholder count that does not match the arguments |
| `syntax` | error | parse failure |

| `channel-send` | warning | a non-moved, non-frozen, non-const, non-shared value sent with `.send/.push` on a local made by `channel()` |
| `deprecated` | warning | built-in table (`std::random_shuffle`, `std::auto_ptr`, `std::bind1st/2nd`, `std::ptr_fun`, `std::gets`, `std::aligned_storage/union`) and calls of script functions marked `[[deprecated]]` |
| `type` | warning | arithmetic/comparison between a string and a number where both sides are literals or `const` locals initialised from literals |

Scope of the flow checks: function locals, and **script globals at top level**: a global declared by the script's
own top-level `let` is tracked through the top-level statements; globals the host injects are never tracked. A move
of a global is downgraded to "maybe" after any call to script code when some function assigns or moves that global.
Field paths (`std::move(o.f)` then `o.f`) are tracked; assigning `o` or `o.f` clears them. `capability` also covers a
builtin stored in a local (`auto r = fs.read; r(p)`: error when the local is `const`, warning otherwise) and
`eval("literal")`, whose source is checked with the same capabilities. `iter-mutation` recognises `std::vector`,
`deque`, `list`, `forward_list`, `map`, `unordered_map`, `multimap`, `set`, `unordered_set`, `multiset` and `array`.
`lock-order` finds cycles of any length in the nested-`with` graph (strongly connected components). Columns come from
the token positions recorded on AST nodes (`Node::col`).

Considered and not built (yet): **undefined variables** (the resolver already rejects unknown locals; unknown globals
may be host-injected, so only a warning would be sound); **await outside async** (`await` on a task value is legal
anywhere in ForgeScript, so there is nothing to flag); **type inference beyond literals** (values are dynamic and
conversions implicit).

References: `let` nodes carry a `ref` flag (`T& r = x`), so a reference is an alias of its target: moving
through the reference marks the target as maybe moved, and reading a reference whose target may have been moved is a
warning. Capability flow also follows: string-path `import`s of script modules (checked once each, cycles stopped,
same capabilities; errors reported on the import line), builtins passed as arguments to top-level script functions
(the parameter is tainted; calls through it are warnings, since other call sites may pass something else), and
builtins stored in list/map literals (calling an element is a warning). Columns are the start of the construct: the
leftmost recorded token column among the node and its sub-expressions on that line. `[[deprecated]]` on class
methods warns at `obj.m()` calls (the receiver's class is unknown statically, so the message says when only some
classes defining `m` mark it).

Capability taint is a small value analysis: a builtin (`fs.read`, ...) taints the local it is stored in, a container
built from it (literal, `std::vector<auto> t = {fs.read}` and other standard constructors, `push_back`/`insert`/
`emplace*`/`append`), an element read from such a container, the return value of a script function or method that
returns it, and the parameters of script functions and methods it is passed to. Lambdas see the taint of captured
locals. Calls through any tainted value are warnings (the value may differ at run time); only a `const` local
initialised directly from the builtin is an error. `import` resolves modules like the runtime (`module_search_path`:
script dir, `FORGE_SCRIPT_PATH`, `~/.forge/modules`, cwd; `name.fs`, `name.cpp`, `name/mod.fs`, `name/mod.cpp`), and
with `fs` denied an import is itself an error (the runtime refuses it). A native `.so/.dylib/.dll` import is an error
unless every capability is granted, matching `load_plugin`. Statement nodes carry the column of their first token.

The checker runs in two passes: the first summarises every function and lambda (returned builtins, builtins
stored into object fields) and the second reports with those summaries, so a call may come before the definition.
Builtins assigned to an object field taint that field name on every object (coarse, warning only); calling such a
field warns. Calling a lambda, a local holding a lambda, or a method whose body returns a builtin taints the result.
`forge::import("lit")` and `forge::plugin("lit")` are checked like `import` statements. Extra import directories:
`forge script --check --module-dir DIR` (repeatable) or `check_options::module_dirs`; the engine passes its
`base_dir` automatically, so embedded runs resolve modules like the runtime.

Known gaps: field taint is by field name, not per object or class (warnings only); summaries iterate twice, so
taint passed through more than two levels of function-returns-function may be missed; non-literal paths to
`forge::import`/`plugin` are not followed; taint is per variable, not per element or key. Test scripts that deliberately trigger the runtime errors carry
`// forge-check: ignore-file`.
