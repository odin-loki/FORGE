# ForgeScript (`forge script`)

ForgeScript is the scripting language built into the `forge` binary. Scripts are written in an interpreted,
memory-safe subset of C++23. Built-ins go through the [OS wrapper](os-wrapper.md), so one script runs the same on
Windows, Linux and macOS. Shell calls give back structured results, and a script in the portable subset can be
translated into a standalone script for a native shell.

Code: `script/` (library `forge_script`). Public headers: `forge/script/script.hpp` (engine), `check.hpp` (static
checker), `local_ai.hpp` (in-process model for the AI calls), `plugin.h` (native plugin ABI).

Status: implemented and tested unless a section says otherwise. Sections marked **untested** compile but
have no automated test.

## Running scripts

```
forge script build.cpp [args...]               # run a file
forge script -e 'std::println("{}", 6 * 7);'   # run a snippet (remaining arguments go to the script)
forge script                                   # REPL
```

A file may start with `#!`. Top-level statements run first, then `int main()` if the file has one.

**Exit status**: the value returned by `main()` or passed to `std::exit(n)`; 1 for an uncaught exception; 2 for a
syntax error, a script refused by the static checks, a failed `--emit`, or a bad command line.

| option | effect |
|---|---|
| `--tree` | tree-walker only (no bytecode VM) |
| `--debug` | race, lock-order and stall detectors (see *Threads*) |
| `--timeout MS`, `--max-steps N`, `--heap BYTES` | whole-run wall clock, step count, managed-heap cap |
| `--mem-limit SIZE` | hard memory cap for the run and its child processes, e.g. `512M`, `2G` (see *Resource limits*) |
| `--no-fs`, `--no-net`, `--no-proc` | turn a capability off (see *Capabilities and sandboxing*) |
| `--os-sandbox off\|best-effort\|require` | OS enforcement of the `--no-*` flags (default `best-effort`) |
| `--policy FILE` | FORGE Security Standard policy (see *Security policy*); without it `$FORGE_SECURITY_POLICY`, `./forge-security.json`, the config dir or `$FORGE_SECURITY_PROFILE` is used, and with none of them nothing changes |
| `--check [--json] [--module-dir DIR]` | static checks only, nothing runs; exit 2 on errors |
| `--no-check`, `--strict-check` | skip the static checks before a run / refuse on warnings too |
| `--emit TARGET [-o FILE]`, `--emit-targets` | translate to a shell script instead of running (see *Write once, emit anywhere*) |
| `--ask-model M.fpk`, `--ask-max-tokens N` | in-process model for `forge::ask` & co |
| `--cache-dir DIR`, `--cache-trust HEX`, `--cache-require-signed`, `--cache-sign-seed HEX` | `.fsc` parse cache |
| `--args` | end of options; the rest are script arguments |
| `--rpc-child NAME` | internal: used by `forge rpc-serve` to run a remote script from stdin |

There is no `--no-env` flag; the `env` capability can only be turned off with `forge::sandbox` or by an embedder.
Script arguments are `forge::args()` (also the global `args`).

## Language

A script looks like ordinary C++. `#include <...>` lines are accepted and ignored; `#include "util.cpp"` loads
another script and brings its names in.

Rejected by the parser, with a message saying what to use instead: raw pointers (`T*`), `new`/`delete`,
address-of, `reinterpret_cast`/`const_cast`/C-style casts, pointer arithmetic, explicit destructor calls.

Supported:

- Declarations: `int`, `long long`, `double`, `bool`, `char`, `std::string`, `auto`, `const`/`constexpr`, several
  declarators, brace and paren initialisation, structured bindings `auto [k, v] = ...`. Narrowing from a literal
  (`int x = 3.5;`) and assigning to a `const` are errors.
- A variable keeps its declared arithmetic type: storing into an `int`, `double`, `bool` or `char` variable (`=`,
  compound assignment, `++`/`--`) and returning from a function declared with one converts as C++ does
  (`double x = 7; x = 1;` leaves `1.0`, `int i; i = 2.9;` leaves `2`, `int f() { return 2.7; }` returns `2`). The
  conversion is added by the parser for named variables and return values; it is not applied to struct members,
  container elements or values that reach a variable through a reference (see *Differences from C++*).
- `char` is a character and a small integer: `'a' + 1` is `98`, `c - '0'` is the digit, `char n = c + 1;` and
  `static_cast<char>(...)` convert back, `s[i]`, `s.at(i)`, `s.front()`, `*it` and `for (char c : s)` yield chars.
  `const char*` is accepted as a string type.
- Expressions: every C++ operator on the supported types, with C++ precedence; assignment, compound assignment
  and `++`/`--` are expressions (`f(i++)`, `while (--n > 0)`, `a = b = c`, `v[i++] = x`); the comma operator inside
  parentheses; `v.at(i)`, `v.front()`, `v.back()`, `std::get<N>(t)`, `*it` and `it->second` as assignment targets;
  `std::tie(a, b) = expr;` as a statement.
- Control flow: `if`/`else`, `while`, `do`/`while`, `for(;;)`, range-`for` (with bindings), `switch` with
  fall-through, `if (init; cond)` and `switch (init; value)`, `break`/`continue`,
  `try`/`catch`/`throw`/`throw;` (see *Catch clauses*). There is no `finally`.
- Functions with default arguments, by-value and by-reference parameters, recursion, templates (type-erased),
  lambdas with `[=]`, `[&]`, `[x]`, `[&x]`, `[x = expr]` captures and `mutable`. `[=]` and `[x]` copy when the
  lambda is created; a `std::function` variable may be called from its own initialiser (`[&f]` recursion).
- Classes and structs with value semantics, constructors with init lists and base-constructor delegation,
  `using Base::Base;`, `virtual`/`override`/pure virtual (abstract classes cannot be instantiated), static members,
  operator overloading (`+`, `==`, `<`, `<=>`, `()`, `= default` comparisons), `friend` free operators.
- `enum`/`enum class`, `namespace` (nested, `using namespace`), `typedef`/`using` aliases.
- Integers follow C++ (`/` truncates, `%` keeps the sign of the dividend), except that 64-bit overflow throws
  `OverflowError` and division by zero throws `ArithmeticError`.

Standard library surface (native; the `std::` prefix is optional after `using namespace std;`):

- I/O: `std::cout`/`cerr`/`clog` with `endl`, `flush`, `boolalpha`, `fixed`, `setprecision`, `setw`;
  `std::print`/`println`/`format` (full format specs); `std::println(stderr, ...)`.
- `std::string` (mutable value type: `size`, `substr`, `find`, `rfind`, `find_first_of`, `find_last_of`,
  `find_first_not_of`, `find_last_not_of`, `push_back`, `pop_back`, `append`, `insert`, `erase`, `replace`,
  `resize`, `assign`, `swap`, `clear`, `compare`, `starts_with`, `ends_with`, `contains`, the substring and
  iterator-range constructors, ...), `std::string_view`.
- Containers: `vector`, `deque` (`push_front`, `pop_front`, `emplace_front`), `list` (`sort`, `reverse`), `array`
  (`fill`), `span` (a read-only copy), `map`/`unordered_map` (`operator[]` inserts a default; `insert`, `emplace`,
  `try_emplace` and `insert_or_assign` return `{iterator, inserted}`; `lower_bound`/`upper_bound` on ordered ones),
  `set`/`unordered_set`, `pair`, `tuple`/`get<N>`/`tie`, `optional`, `any`. Vectors, pairs and tuples compare
  lexicographically; `vector(first, last)`, `insert(pos, n, v)`, `insert(pos, first, last)`, `assign`.
- Iterators (`begin`/`end`, `find`, `it - v.begin()`, `it->first`, `std::next`/`prev`/`distance`).
- `<algorithm>`/`<numeric>`: `sort`, `stable_sort`, `is_sorted`, `reverse`, `rotate`, `partition`,
  `next_permutation`, `find(_if)(_not)`, `count(_if)`, `any_of`/`all_of`/`none_of`, `for_each`,
  `min/max/minmax_element`, `lower_bound`/`upper_bound`/`binary_search`, `equal`, `fill`, `iota`, `replace`,
  `copy(_if)`, `transform` (one or two inputs), `remove(_if)`/`unique` with `erase`, `std::erase(_if)`,
  `back_inserter`, `accumulate`, `reduce`, `partial_sum`, `inner_product`, `gcd`, `lcm`, `min`/`max`/`clamp`, the
  `std::ranges::` forms, and the function objects `std::less`, `std::greater`, `std::plus`, ...
- `<cmath>`, `<cctype>`, `to_string`, `stoi`/`stol`/`stoll`/`stod`, `numeric_limits`, `std::getenv`, `std::exit`.
- Exceptions `std::exception`, `runtime_error`, `logic_error`, `invalid_argument`, `out_of_range`, ... with
  `what()`. They are aliases of the script error classes (table under *Errors*).

```cpp
struct Job { std::string name; std::string cmd; };

int main() {
    std::vector<Job> jobs{{"kernel", "uname -r"}, {"files", "ls / | wc -l"}};
    std::map<std::string, std::string> out;
    for (const auto& j : jobs) {
        auto r = forge::sh(j.cmd);                     // no {} placeholders: run as written
        if (!r.ok) { std::println(stderr, "{} failed: {}", j.name, r.err); return 1; }
        out[j.name] = r.lines().front();
    }
    for (const auto& [k, v] : out) std::println("{:>8}: {}", k, v);
    return 0;
}
```

### Differences from C++

`tests/script_conformance` compiles a set of ordinary C++23 programs with the host compiler and requires
`forge script` (tree-walker and VM) to print exactly what the compiled program prints
([testing.md](testing.md)). What the programs had to leave out is listed here.

Rejected with a diagnostic that names the construct:

- raw pointers and address-of, `new`/`delete`, `sizeof`, C-style / `reinterpret_cast` / `const_cast`, pointers to
  members (`&T::member`), `std::stoi(s, &pos)`;
- designated initialisers (`T{.a = 1}`), `static` local variables, `enum class` declared inside a function, a
  range-`for` with an init-statement, `continue` inside a `switch`;
- assignment through a reference returned by a function (`obj.at(i, j) = v`; only the standard element accessors
  above are assignable), `catch` of a type that cannot be matched (see *Catch clauses*).

Accepted but different, or failing at run time with an `AttributeError` / `TypeError` that names the member:

- Integers are 64-bit and checked: overflow raises `OverflowError`, unsigned types do not wrap, `int` does not
  truncate to 32 bits.
- Types are checked dynamically. The parser converts on stores into named `int` / `double` / `bool` / `char`
  variables and on returns; a `double` struct member or `std::vector<double>` element initialised or assigned from
  an `int` keeps the `int` (`p.w = 3; p.w / 2` is `1`). Write `3.0`.
- Functions and constructors are not overloaded on parameter types. Constructors may differ in argument count;
  a user-defined copy constructor replaces a one-argument constructor (copying is member-wise, done by the
  runtime). Other overloads (a unary and a binary `operator-` in one class, const / non-const pairs that differ in
  behaviour) are not distinguished: the last one wins. User `operator++`/`--`, `operator[]`, conversion operators
  and delegating constructors (`T() : T(0) {}`) are not supported; `<=>` works only as `= default`.
- A base class with constructors of several argument counts cannot be initialised from a derived constructor's
  initialiser list, and a vector member cannot be initialised as `rows(n, value)` there (build it in the body).
- Templates are type-erased: `T x{};` has no value, and a template cannot be specialised.
- A moved-from variable cannot be read at all (`BorrowError`), not even compared with `nullptr`.
- `std::optional<int>` holding a value is the value: `o.reset()` and `o.emplace(x)` work on a named variable only
  as `o.reset()` / `o = x`. `p.use_count()` is not available on class objects.
- Reverse iterators (`rbegin`/`rend`), `std::multimap` / `std::multiset` (parsed as map / set: one entry per key),
  map keys other than numbers, bools and strings (no `std::pair` keys), `std::span::subspan`/`first`/`last` and
  writing through a span, `std::string_view::remove_prefix`/`remove_suffix`, `std::sort` on a string,
  `std::stoi` with a base, `reserve`/`capacity` on strings, nested brace elision for `std::array<std::array>`.
- Structured bindings by reference (`auto& [a, b] = s;`) copy scalar members. A reference to an object with a
  destructor (`Base& b = d;`) can run that destructor twice at scope exit (known bug).
- The order in which a container destroys its elements is forward (libc++ and libstdc++ destroy a vector in
  reverse).
- `throw "text";` throws a string: `catch (const char*)` and `catch (const std::string&)` both match it.

## Memory model

- **Value types** (`std::string`, containers, structs and classes held by value) are copied on
  copy-initialisation, by-value parameters and insertion into containers, as in C++.
- **Reference counting.** Heap objects carry an atomic refcount. Releasing the last reference frees the object at
  once. `std::make_shared<T>`/`std::shared_ptr` share one object; `std::weak_ptr` (`lock`, `expired`) does not keep
  it alive (`weak(x)` is the untyped form).
- **Cycle collector.** A synchronous trial-deletion collector reclaims cycles among containers, instances and
  closures. It runs when an allocation would exceed the heap cap or the `forge::mem` budget, on
  `forge::gc::collect()`, the REPL's `:gc`, and engine reset or destruction. It does not run while tasks are active.
- `std::make_unique<T>`/`std::unique_ptr` cannot be copied (`BorrowError`); `std::move` transfers it.
- `std::move(x)` leaves `x` moved-from: a later read throws `BorrowError`. Scalars are unaffected.
- Dereferencing an empty pointer or optional, or an `end()` iterator, throws `NullPointerError`/`IndexError`.
  `at()`, `substr`, `erase` and the like are bounds-checked.
- **Destructors.** `~T()` runs when the last reference goes away: block locals at the closing brace in reverse
  order (also on `break`, `return` and exceptions), `unique_ptr` on reset or scope exit, `shared_ptr` when the last
  owner drops. Order: most-derived destructor, then bases, then members. A constructor that throws destroys
  nothing. Destructors run at the next statement boundary on the owning thread, so they may print, call functions
  and throw. Prvalue initialisation (`G g("x");`, `G g = make();`) elides the copy as in C++17.
- **Borrows while iterating.** A range-`for` borrows its container. Structural changes to it (`push_back`, `insert`,
  `erase`, `clear`, `resize`, inserting a new key with `operator[]`, ...) throw `BorrowError`. Changing elements in
  place is allowed.
- **References.** Non-const `T&` parameters (functions, methods, lambdas, `std::for_each`,
  `forge::shared::with`) and `for (T& x : c)` write through to the caller's variable, element or member. `const T&`
  and by-value parameters never do.
- **Handles** (files, locks, watchers, processes, pools, devices) wrap a C++ RAII owner. They close when the last
  reference drops, or explicitly with `close()`/`dispose(h)`. Using a closed handle throws `ObjectDisposedError`.
- **Memory budget.** Script heap memory is reserved from the process `forge::mem` budget (tag `script`; see
  [memory.md](memory.md), `FORGE_MEM_LIMIT`). `--heap BYTES`, `options::heap_bytes` and `limit {.mem = ...}` set a
  tighter cap. Over the cap the runtime runs the cycle collector, then throws a catchable `OutOfMemoryError`. Handlers
  get about 4 MiB of grace to run before a second error; the cap is enforced again once the heap is 16 KiB below it.
  Building the error object itself is exempt from the cap. Growth in place (`push_back`, `insert`, `append`, `resize`
  on containers and strings) is charged like a new object.

## Errors

All built-in errors derive from `Error` (fields `message`, `trace`, `line`; `what()` returns the message). User
classes may derive from `Error` or from the `std::` exception names; a user class's own `line` or `trace` member is
left alone.

| group | classes |
|---|---|
| general | `TypeError`, `KeyError`, `NameError`, `AttributeError`, `ArithmeticError` (also `ZeroDivisionError`), `SyntaxError`, `ImportError`, `RecursionError`, `BadOptionalAccess`, `SystemExit` |
| under `LogicError` | `ValueError`, `IndexError`, `LengthError`, `DomainError` |
| under `RuntimeError` | `IOError`, `OSError`, `OverflowError`, `UnderflowError`, `RangeError` |
| runtime safety | `BorrowError`, `NullPointerError`, `FrozenError`, `AccessError`, `ObjectDisposedError`, `RaceError` |
| features | `ShellError`, `LimitExceededError`, `PipelineError`, `PluginError`, `AIError`, `HostError` |
| under `OSError` | `PermissionError` (subclass `SandboxViolationError`), `TimeoutError`, `OutOfMemoryError`, `CancelledError`, `DeadlockError`, `FileNotFoundError`, `FileExistsError`, `NotADirectoryError`, `IsADirectoryError`, `NotSupportedError`, `InterruptedError`, `BlockingIOError`, `ProcessLookupError`, `NoSpaceError`, `TooManyFilesError`, `BusyError`, `ReadOnlyFileSystemError`, `SignalError`, `ConnectionError` (subclasses `ConnectionRefusedError`, `ConnectionResetError`, `BrokenPipeError`), `SecurityError` (subclasses `IntegrityError`, `PolicyViolation`) |

**The `std::` names** are these classes, and the tree follows `<stdexcept>`: `catch (const std::logic_error&)`
handles `std::invalid_argument` and `std::out_of_range`, `catch (const std::runtime_error&)` handles
`std::overflow_error`, `std::system_error` (every `OSError`) and I/O errors.

| C++ name | class | C++ name | class |
|---|---|---|---|
| `std::exception`, `forge::error` | `Error` | `std::runtime_error` | `RuntimeError` |
| `std::logic_error` | `LogicError` | `std::overflow_error` | `OverflowError` |
| `std::invalid_argument` | `ValueError` | `std::underflow_error` | `UnderflowError` |
| `std::out_of_range` | `IndexError` | `std::range_error` | `RangeError` |
| `std::length_error` | `LengthError` | `std::system_error` | `OSError` |
| `std::domain_error` | `DomainError` | `std::filesystem::filesystem_error`, `std::ios_base::failure` | `IOError` |
| `std::bad_alloc` | `OutOfMemoryError` | `std::bad_cast`, `std::bad_any_cast` | `TypeError` |
| `std::bad_optional_access` (`o.value()` on an empty optional) | `BadOptionalAccess` | | |

Differences from C++ that follow from the aliasing: the errors the runtime itself raises are script classes, so
`std::stoi("x")` is a `ValueError` (`std::invalid_argument`) and `v.at(9)` an `IndexError` (`std::out_of_range`) as
in C++, but integer overflow is an `OverflowError` and so a `std::runtime_error`, and `OutOfMemoryError` is an
`OSError` and so also a `std::runtime_error`. Other `std::` exception types (`std::bad_function_call`,
`std::format_error`, `std::regex_error`, ...) are not supported in a catch clause: the parser says so.

### Catch clauses

`try { ... } catch (T x) { ... }` with any number of clauses; the first clause that matches handles the error.
`throw expr;` throws any value; `throw;` in a handler rethrows the caught value with its dynamic type. A
function-try-block (`int f() try { ... } catch (...) { ... }`) is accepted for functions, not for constructors.

| clause | handles |
|---|---|
| `catch (...)` | everything catchable |
| a class: `catch (const E& e)`, `catch (E e)`, `catch (E&)`; qualified (`forge::PermissionError`, `std::out_of_range`) or not (`PermissionError`) | an object of class `E` or of a class derived from it, user-defined or built-in. Nothing else: a class clause is never a catch-all, and `catch (const std::exception&)` does not handle a thrown `int`, string or plain struct |
| `catch (int)`, `catch (double)`, `catch (bool)`, `catch (char)` | a value of exactly that kind (no arithmetic conversions; every integer type is `int`, enum values are `int`) |
| `catch (const std::string&)`, `catch (std::string_view)`, `catch (const char*)` | a thrown string (`throw "text";` and `throw std::string("text");` are the same here) |
| `catch (const std::vector<T>&)`, `catch (const std::map<K, V>&)` | a thrown list / map (element types are not checked) |

By value, by reference and by const reference match the same way; the handler's variable is the thrown object.
A name the parser does not know as a type is taken as a class name and looked up when the error is matched, so a
class that does not exist matches nothing. `catch (auto)`, a template parameter, `std::any`, `std::optional`, smart
pointers and `std::function` are syntax errors that name the type. Local destructors and `defer` blocks run while
the error unwinds, innermost first, before the handler. The tree-walker and the VM share one implementation
(`Interp::run_try`, `Interp::catch_matches`). Tests: `tests/script/catch_types.fs`, `catch_clause_types`.

**Errors no clause can handle.** These end the run (or the block), and pass through every catch clause, including
`catch (...)`, in every function, lambda and task; a script cannot outlive them by catching and carrying on:

| error | when |
|---|---|
| `std::exit(n)` (`SystemExit`) | always; from any thread it ends the whole run with status `n` |
| `TimeoutError`, `LimitExceededError` from the whole-run limits | `--timeout` / `options::timeout_ms`, `--max-steps` / `options::max_steps` |
| `CancelledError` | SIGINT / SIGTERM / `engine::interrupt()`; a cancelled task (`t.cancel()`) inside that task |
| `TimeoutError` / `LimitExceededError` of a `limit {.timeout}` / `limit {.cpu_s}` block | inside that block; outside it the error is an ordinary one and can be caught |

Destructors and `defer` blocks still run while such an error unwinds, for at most 100 ms; after that the error is
raised again inside them. Every other error is catchable, including `OutOfMemoryError` (the heap cap re-arms, so
catching it does not lift the limit) and the `allocs`, `handles` and `procs` limit errors. Test:
`fatal_errors_pass_through_catch`.

- `catch (const OSError& e)` catches every OS-level error. OS error codes (errno, GetLastError, WSA, NTSTATUS,
  Mach) map to these classes in `forge/os/oserror.hpp`. `EPERM` becomes `SandboxViolationError` only when the active
  OS sandbox policy denies that kind of operation; `EACCES` stays `PermissionError`.
- `forge::security` errors (FSS-ERR-1, [security-api.md](security-api.md)) arriving from C++: `integrity_error` ->
  `IntegrityError`, `policy_violation` -> `PolicyViolation` (both `SecurityError`, an `OSError`), `permission_denied`
  -> `PermissionError`, `sandbox_error` -> `SandboxViolationError`. The last two keep their existing classes, which
  are not under `SecurityError`.
- `SignalError` is a fault caught by `fault_guard` or a crashed `ProcessPool` worker.
- SIGINT/SIGTERM (Windows: Ctrl-C, Ctrl-Break, console close) raise `CancelledError` during a top-level run
  (`options::cancel_signals`, default on; the old handlers are restored afterwards). `engine::interrupt()` does the
  same from the host.
- Native C++ exceptions are mapped by `Interp::from_cpp` (`script/src/runtime.cpp`): heap exhaustion and
  `std::bad_alloc` to `OutOfMemoryError`, parse errors to `SyntaxError`, typed OS exceptions to their class,
  `forge_exception` by code (`IOError`, `ValueError`, `NotSupportedError`, `OutOfMemoryError`, else `Error`),
  `std::out_of_range` to `IndexError`, `std::invalid_argument` to `ValueError`, anything else to `Error`. The engine
  catches everything at its API, so no C++ exception leaves `engine::run`/`eval`.

## Threads and tasks

- `std::async(f, args...)`, `std::thread t(f, args...)` and `forge::spawn(f, args...)` run `f` on another thread
  and return a task. `get()`/`join()`/`wait()` wait and rethrow; `done` tells whether it finished; `cancel()`
  requests cancellation. `forge::await(t)` is the same as `t.get()`.
- Arguments cross threads deep-copied, frozen (`forge::freeze(x)`, deep-immutable), moved, or as
  `forge::shared<T>` (locked on every access: `with(fn)`, `load()`, `store(v)`). Handles and tasks held in
  globals or closures are not copied into the new thread.
- `forge::channel<T> ch(cap)` has `send`, `recv`, `close`, and works in range-`for`.
- `forge::parallel(list, fn)` maps `fn(item)` or `fn(item, index)` over a list on the task pool and returns the
  results in order. Each chunk runs in its own interpreter with deep-copied globals.
- **Debug mode** (`--debug`, `options::debug`):
  - race detector: a mutable object belongs to the thread that created or received it; changing it from another
    thread without holding its `forge::shared` lock raises `RaceError`;
  - lock order: nested `forge::shared::with` calls record lock-order edges; a cycle raises `DeadlockError` before
    the threads can deadlock;
  - stall detector: if every script thread is blocked (channel, `get`/`join`, shared lock) for
    `options::deadlock_ms` (default 1000 ms), the waiting thread raises `DeadlockError`.
- The runtime can be built with ThreadSanitizer (`-DFORGE_TSAN=ON`).

## Execution model

- A script is parsed and resolved once. `engine::run` and `run_file` execute it on the **bytecode VM**; the REPL
  and `eval` use the **tree-walker**. A function called `jit_threshold` times (default 2) is compiled to bytecode,
  so hot REPL code also runs on the VM. `--tree` (`options::use_vm = false`) turns the VM off.
- Both paths share one value model, heap and built-in table. Every test program runs on both and must give the same
  output.
- A verifier checks stack depth, jump targets, constant and local indices and operands before a chunk runs.
  Malformed bytecode is rejected. There is no JIT to machine code.

### `.fsc` parse cache

`run_file()` can cache the parsed and resolved program (not bytecode; bytecode is always compiled in memory and
verified). It skips lexing, parsing and resolution on later runs. Off unless `options::fsc_cache_dir`,
`--cache-dir DIR` or `$FORGE_SCRIPT_CACHE` is set.

- File `<dir>/<key>.fsc`, key = BLAKE3(`"forge-fsc "` ‖ VM version ‖ file name ‖ source). A changed source (even
  with the same size and mtime), name or VM version gives a different key. Options are not in the key: the
  cache holds no capability decisions. Static checks run on the cached tree on every run, and capabilities,
  limits and the OS sandbox apply at run time, so a tree cached by a run with `allow_fs` is still refused by
  a run without it.
- Loading checks magic, VM version, the recomputed key, a BLAKE3 checksum of the payload (catches corruption,
  not forgery), every length and count, node kinds, nesting (≤ 4096) and back-references (no cycles), and
  rejects trailing bytes and files over 64 MiB. Any failure means the source is parsed normally and the entry
  is rewritten.
- Signing: with `--cache-sign-seed HEX` / `options::fsc_sign_seed` / `$FORGE_SCRIPT_CACHE_SIGN_SEED` (32-byte
  Ed25519 seed) the files are signed. A signed file loads only if its signer is trusted (`--cache-trust HEX`, or the
  engine's own key).
- Unsigned files load only from a directory owned by the current user and not writable by group or other, and
  only if the file is a regular file owned by the user and not group/other-writable (POSIX). Never on Windows.
  `--cache-require-signed` rejects them everywhere. Unsigned files are not written where they would not load
  (a non-private directory, or Windows): the run reports `miss: ...; not stored: ...`.
- POSIX: the directory is created 0700 and opened once; the entry is read with `O_NOFOLLOW` relative to that
  descriptor (a symlinked entry is rejected) and written 0600 through a unique `O_EXCL` temp file renamed into
  place, so concurrent runs never see a torn file. Windows writes through a temp file and `MoveFileEx`; a
  replace that loses a race with a reader fails and leaves the old, complete entry.
- No size limit or eviction: delete the directory to clear it. A temp file is removed when a write fails; a
  process killed mid-write can leave a `*.fsc.tmp-*` file behind.
- `engine::cache_status()` reports `off`, `hit`, `miss (stored): <reason>` or `miss: <reason>`.
- Tests: `tests/test_script_fsc.cpp` (round trip of every tree field, corrupt/stale/foreign files, trust rules,
  permissions, concurrent stores, every `tests/script/*.fs` uncached, cold and warm). Fuzz: `fuzz/fuzz_fsc.cpp`.

## REPL

`forge script` with no file. Type statements or expressions; a value without `;` is printed. Input continues on
the next line (prompt `. `) while the parser reports it incomplete. Errors are printed and the session keeps its
globals. Ctrl-C cancels the running statement with `CancelledError`. The REPL does not run the static checks.

| command | effect |
|---|---|
| `:type <expr>` | type of an expression |
| `:dis <code>` | bytecode listing |
| `:mem` | heap bytes, objects, collections, steps |
| `:gc` | run the cycle collector |
| `:reset` | clear all globals |
| `:vm on\|off` | compile hot functions or not |
| `:help`, `:quit` (`:q`, `:exit`) | |

There is no line editing, history or tab completion.

## Shell calls

```cpp
std::string dir = "my files";
auto r = forge::sh("ls -la {} | wc -l", dir);       // {} is replaced by dir, quoted for sh
std::println("{} {} {}", r.out, r.code, r.ok);
auto procs = forge::pwsh("Get-Process | Select-Object -First 3 | ConvertTo-Json").json();
auto here = forge::shell("echo hi");                 // the host's default shell
auto g = forge::run(std::vector<std::string>{"git", "status"});   // no shell at all
```

- `forge::<backend>(template, args...)` exists for every shell backend and alias (`sh`, `bash`, `zsh`, `pwsh`,
  `ps`, `cmd`, `fish`, `nu`, ...; see [shells.md](shells.md)). Each `{}` is replaced by the next argument quoted for
  that shell; `{{` and `}}` are literal braces. A missing shell raises `ShellError`.
- `forge::shell(...)` uses `forge::host_shell()`: `$FORGE_SHELL` if set; on Windows `pwsh`, then `powershell`,
  then `cmd`; on Android `mksh` if a binary of that name exists, else `sh`; `sh` elsewhere.
- Result (`ProcResult`): `cmd`, `out`, `err`, `code`, `ok`, `ms`, `timed_out`, plus `lines()`, `json()` (parse
  `out`; `ValueError` if it is not JSON) and `check()` (raise `ShellError` unless `ok`).
- `run(cmd[, opts])` / `forge::proc::run`: `cmd` is a list (argv) or a string split on whitespace with simple
  quotes; no shell. Options: `cwd`, `stdin`, `timeout` (ms), `merge` (stderr into stdout), `env` (map), `check`.
- `forge::proc::start(argv)` returns a process handle: `read([timeout])`, `wait([timeout])`, `kill()`, `running`,
  `pid`. `forge::proc::which(name)` finds an executable or returns nil.
- Shell calls and `run` capture at most 64 MiB per output stream. They inherit the child limits of an enclosing
  `limit` block and are cut off at the script's deadline.

## Resource limits

`forge::limit(opts, fn)` runs `fn` with narrower limits and restores them afterwards, also when `fn` throws. `opts`
is a map or a braced list of `{"key", value}` pairs. Unknown keys and malformed values raise `ValueError`.

| key | limits | over the limit |
|---|---|---|
| `timeout` | wall-clock time (ms, or a duration string) | `TimeoutError` |
| `cpu_s` | CPU time of the running thread (seconds, or a duration string) | `LimitExceededError` |
| `mem` | extra managed-heap bytes (number or size string) | `OutOfMemoryError` |
| `proc_mem`, `proc_cpu` | OS memory (bytes / size string) and CPU seconds (number / duration string) for child processes started inside the block | the OS kills the child |
| `allocs` | managed-heap objects allocated by the running thread inside the block | `OutOfMemoryError` |
| `handles` | OS handles open at once: files, locks, file watchers, `proc::start` processes, `ProcessPool`s and their futures; a handle counts until closed | `TooManyFilesError` |
| `procs` | child processes started inside the block: shell calls, `run`/`proc::run`, `proc::start`, each `ProcessPool` worker | `LimitExceededError` |

**Sizes**: a number of bytes or a string such as `"64K"`, `"512M"`, `"1.5G"`, `"1T"`. K/M/G/T are powers of 1024;
`KB` and `KiB` mean `K`; a trailing `B` is optional. **Durations**: a number (ms for `timeout`, seconds for `cpu_s`
and `proc_cpu`) or a string with a unit: `"250us"`, `"200ms"`, `"5s"`, `"2m"` (or `"2min"`), `"1h"`. A child CPU
limit is rounded up to whole seconds. `allocs`, `handles` and `procs` take integers.

**Block syntax.** `limit {opts} { body }` is sugar for `forge::limit(opts, [&] { body })`; `forge::limit` may be
written instead of `limit`. Options are `{"key", value}` pairs or designators (`.key = value`); the parser rejects
unknown and repeated keys. `limit (expr) { body }` takes the options from any map expression.

```cpp
limit {{"timeout", "5s"}, {"mem", "64M"}} { build(); }                  // statement: value discarded
auto r = limit {.cpu_s = "2s", .allocs = 100000} { return solve(); };  // expression: the block's return value
limit (opts) { work(); }                                                // options from a variable
```

The body sees and changes the enclosing locals. In the statement form `return` is a syntax error. `break` and
`continue` cannot leave a block.

**Nesting and scope.** Nested blocks can only narrow: the inner block gets the earlier deadline, the smaller heap
cap and the smaller child limits. Each `handles`/`procs` block has its own counter, and an acquisition counts against
every enclosing block. Tasks started inside a block (`std::async`, `std::thread`, `forge::parallel`) and modules it
imports share its `handles`/`procs` counters, child limits and deadline; `cpu_s` and `allocs` count only the thread
that entered the block. An `allocs`, `handles` or `procs` limit that fires is switched off until its block exits, so
`catch` code inside the block can still run. A `timeout` or `cpu_s` limit that fires cannot be caught inside its
block (see *Catch clauses*): the error leaves the block, where it can be caught, and clean-up code on the way out
gets 100 ms.
Over `handles` the new resource is closed again before the error; over `procs` nothing is started.

Whole-run limits (`options` / CLI): `max_steps` (`--max-steps`), `timeout_ms` (`--timeout`), `heap_bytes`
(`--heap`), `mem_limit_bytes` (`--mem-limit`), call depth `max_depth` (default 800).

**`--mem-limit SIZE`** (`options::mem_limit_bytes`) is a hard cap on the memory of the run. `SIZE` is a size string
as in `limit {.mem = "2G"}` (`script::parse_size`); `0` and malformed sizes are refused (exit 2). It bounds three
things:

| what | how | over the limit |
|---|---|---|
| the managed heap | the same accounting as `--heap`: bytes charged for strings, containers, objects, closures and call frames, counted from the creation of the engine | cycle collection, then a catchable `OutOfMemoryError` ("script memory limit of N bytes exceeded (--mem-limit)"), with the same 4 MiB of grace for handlers |
| each child process | every shell call, `run` / `proc::run`, `proc::start` child starts with an OS memory cap of `SIZE`, as if inside `limit {.proc_mem = SIZE}`; a `ProcessPool` worker runs its tasks under the same `mem_limit_bytes` | the child's allocations fail or it is killed (`RLIMIT_AS`; Job Object on Windows; resident-size watchdog on macOS and Android) |
| the `forge script` process itself (CLI only) | Linux: `RLIMIT_AS` set to the address space in use at start plus `SIZE` plus headroom (`SIZE`/2, at least 64 MiB). Windows: a Job Object process-memory limit of the committed memory at start plus the same | a native allocation fails: `OutOfMemoryError` ("out of memory") where the interpreter can report it, otherwise the process ends |

How it differs from `--heap BYTES`: `--heap` caps only the managed heap of this interpreter. `--mem-limit` caps
the same heap (with both given, the lower one applies; a `limit {.mem}` block can narrow either, never widen it),
and also memory the heap accounting does not see: child processes, and in the CLI native memory of the process
(captured child output, file contents being read, the parsed program, thread stacks). `limit {.proc_mem}` can
lower the cap of children inside a block but not raise it.

What it does not bound: the heap accounting counts payload bytes, not allocator overhead, so the process uses more
than `SIZE` when the heap cap fires; the headroom on the OS cap covers that, which makes the OS cap a backstop for
native memory at about 1.5 x `SIZE` rather than an exact figure. The OS cap counts address space (Linux) or
committed memory (Windows), not resident memory, so thread stacks and allocator reserves count, and a child capped
at a small `SIZE` may fail to start. macOS, iOS, Android and the BSDs get no OS cap on the `forge script` process (no usable mechanism for a
running process; the heap and child caps still apply). An embedder's process is never capped: the engine applies
only the heap and child parts. Children are capped one by one, not as a sum. `forge rpc-serve` passes
`--script-mem-mb` to its script children as `--mem-limit` too, on top of the OS cap it already set; that OS cap has
no headroom, so there it usually fires first (`OutOfMemoryError: out of memory`).

Tests: `tests/script/limits.fs`, `limit_block.fs`, `limit_counts.fs`, `mem_containers.fs` (heap accounting per
container kind); `mem_limit_whole_run`, `mem_limit_size_strings`, `mem_limit_children` in `tests/test_script.cpp`.
The OS cap on the CLI process has no automated test.

## Capabilities and sandboxing

Four capabilities, all on by default: `fs`, `net`, `proc` (processes and services), `env` (environment and device
access); plus `reflect` (mutating reflection, see *Reflection*). A denied call raises `PermissionError`.

- **Whole run**: `--no-fs`, `--no-net`, `--no-proc` (`options::allow_*`).
- **Per call**: `forge::sandbox(opts, fn)` runs `fn` with capabilities turned off. `opts` is a map such as
  `std::map<std::string, bool>{{"fs", false}, {"reflect", false}}`. A capability can only be narrowed; a nested
  `forge::sandbox` cannot turn one back on. There is no `sandbox {...} {}` block syntax.

```cpp
const std::map<std::string, bool> no_net{{"net", false}};
forge::sandbox(no_net, [] { untrusted_step(); });
```

**Checked at the call, not at the name.** A capability is checked when a built-in runs, against the interpreter
that runs it. These are all denied inside a sandbox: aliases (`auto r = forge::fs::read;`), `std::function`s,
lambdas, methods or map entries made before the sandbox, higher-order calls (`std::transform`, `std::for_each`),
`eval`, `forge::reflect::call`, imports, shell calls, and tasks started inside it (`std::async`, `std::thread`,
`forge::parallel` inherit the capabilities). A `ProcessPool` worker never gets a capability its parent lacks. A
file handle opened before a no-fs block cannot be read or written inside it; a process started before a no-proc
block cannot be killed inside it. `std::getenv` is gated by `env` like `os::env`. Test:
`tests/script/sandbox_escape.fs`.

**OS sandbox.** When `fs`, `net` or `proc` is off for the whole run, the engine also applies an OS sandbox from
`forge/os/sandbox.hpp` to the process (seccomp-BPF on Linux/Android, Capsicum on FreeBSD, pledge on OpenBSD,
`sandbox_init` on macOS, a job object on Windows). Mapping: `--no-net` denies sockets other than AF_UNIX,
`--no-proc` denies exec, `--no-fs` denies file writes (reads are blocked by the interpreter only). It is one-way and
process-wide. `--os-sandbox best-effort` (default) records a failure in `engine::sandbox_status()`; `require` makes
it fatal; `off` leaves interpreter checks only. On Windows a running process cannot drop fs or net access, so the
interpreter checks those. Only the Linux mechanism has an automated test (`tests/test_script_os_sandbox.cpp`). See
[security.md](security.md).

A nested `forge::sandbox` block without `isolate` is enforced by the interpreter only: every OS mechanism above
can only be tightened for the rest of the process's life, so it cannot be applied on entry and lifted when the block
returns. Use `isolate` when the block needs OS enforcement.

**Isolated blocks** (POSIX only). `forge::sandbox` with `{"isolate", true}` runs the block in a forked child that
applies the narrower OS policy. The block's return value (JSON-representable values only), its print output and
any exception (with its type) come back. The child works on a copy of the heap, so assignments inside the block are
not seen afterwards. On Windows and iOS it raises `PermissionError` without running the block. Forking copies only
the calling thread: avoid it while other tasks hold locks the block needs. **Untested.**

**Plugins** run native code, so they are refused unless all of `fs`, `net`, `proc` and `env` are on. Under a
security policy they are loaded through `security::open_plugin`: `plugin_load` must be allowed, and
`integrity.plugins` `"required"` needs a detached signature `<plugin>.sig` (`forge sign libx.so --key k.key`) by one
of `integrity.trusted_keys` (any valid signature when the list is empty). `"optional"` verifies signed plugins and
loads unsigned ones (audited `weaken`); `"off"` does not check. With `integrity.os_code_signing` the plugin must
also be OS code signed (Authenticode on Windows, a valid non-ad-hoc code signature on macOS; on Linux / BSD there is
no such mechanism and the load is audited instead). The signature binds the file name: renaming a signed plugin
breaks it. The bytes that were verified are the bytes loaded (see [security-api.md](security-api.md#integration));
a failure raises `IntegrityError`.

**Security policy (FSS).** `forge script --policy FILE` (or a policy found by `security::load_policy`, see
[security-api.md](security-api.md#policy)) runs the script under the FORGE Security Standard; with no policy and no
`FORGE_SECURITY_PROFILE` (profile `none`) nothing in this paragraph applies. With a policy:

- The capabilities come from the policy (default deny, P1): `net` only when `net.mode` is not `none`, `proc` only
  with `exec`, `env` only with `env`, `fs` unless both `fs.read` and `fs.write` are empty. The `--no-*` flags narrow
  the policy further. Profile `strict` therefore runs with no net, proc or env and the file system confined to its
  roots; a signature does not add capabilities.
- File operations are **path-scoped**: each built-in passes the path it touches to `security::enforce` (reads:
  `fs.read`, `fs.list`, `fs.exists`, `fs.stat`, `fs.walk`, `fs.glob` (its base and the pattern's literal directory),
  `fs.open(p, "r")`, `fs.copy` source, imports, `forge::watch`, locale catalogs; writes: `fs.write`, `fs.append`,
  `fs.open` for writing, `fs.mkdir`, `fs.remove`, `fs.rename`, `fs.move`, `fs.copy` target, `fs.touch`, `fs.chmod`,
  `fs.lock`, `fs.temp_file`, ...). The canonical path (symlinks resolved, `..` folded) must lie under a root. Network
  calls pass `host:port` (`net.http`, `net.port_open`, `net.resolve`, `forge::remote*`, the AI endpoint), process
  calls the program (`run`, `proc.start`, shell blocks, `ProcessPool`), `os.env` / `std::getenv` the variable. A
  denial raises `PermissionError` and is audited (FSS-AUD). The `Caps` flags are still checked first.
- **Signed scripts** (`integrity.scripts`). Sign with `forge sign job.fs --key k.key`, which writes the detached
  signature `job.fs.sig` next to it (Ed25519 over BLAKE3, bound to the file name; [modelfile.md](modelfile.md#detached-signatures)).
  `"required"`: the script file and every module it imports (`import`, `#include`, `forge::import`) must have a
  `.sig` signed by one of `integrity.trusted_keys` (any valid signature when the list is empty); each is checked on
  the exact bytes that are parsed. An unsigned, modified or untrusted file raises `IntegrityError` (the main file
  fails before any code runs, exit 1; an import fails at the import). `-e` code and the REPL cannot carry a signature
  and are refused (exit 2). `"optional"`: signed files are verified (a bad signature still fails), unsigned files run
  and are audited (`weaken`). The `.fsc` parse cache is separate: it caches parses of verified sources.
- `security::lockdown(policy)` is applied before any script code runs (it replaces `--os-sandbox`): resource limits,
  no-new-privs, Landlock write roots, seccomp / pledge / sandbox_init / job object, Windows mitigations.

Embedders get the path scoping too: the runtime checks against `security::active_policy()`, so a host that calls
`security::lockdown` or `set_active_policy` confines the scripts it runs, and `engine::run_file` and imports check
`integrity.scripts`. Tests: `tests/test_script_security.cpp`, `tests/test_signing_detached.cpp`,
`tests/security_cli_test.py`, `tests/signing_cli_test.py`.

**Remote scripts** under `forge rpc-serve --allow-scripts` each run in a child process (`forge script --rpc-child`)
with `--script-timeout` (default 60000 ms), a `--script-mem-mb` memory cap (default 2048) and the `--script-no-*`
flags. With any `--script-no-*` flag the child uses `--os-sandbox require` (`best-effort` on Windows). Under a
security policy the child loads the same policy and locks itself down before running the script. When the policy
has `integrity.scripts`, send the script's signature with it: `forge::remote(node, src, {{"name", "job.fs"},
{"signature", fs.read("job.fs.sig")}})` (`name` must be the signed file name); the child verifies the source it
received against it. See [rpc.md](rpc.md).

## Process pools

`ProcessPool(workers, opts)` runs tasks in worker processes (`forge::os::process_pool`): fork on Linux, macOS,
FreeBSD and Android; spawn of the same executable on Windows. Not available on iOS (`NotSupportedError`). `workers`
omitted or nil means `$FORGE_PROCESSES`, else the hardware thread count.

A task is ForgeScript source that defines `task(arg)`. Each call runs it in a fresh engine in a worker. The argument
and the result must be JSON-representable.

```cpp
auto pp = ProcessPool(4);                                   // capabilities default to off
auto f = pp.submit("auto task(auto x) { return x * x; }", 7);
f.get();                                                    // 49; also f.get(timeout_ms), f.done()
std::vector<int> xs = {1, 2, 3};
pp.map("auto task(auto x) { return x + 1; }", xs);         // [2, 3, 4]
pp.shutdown();                                              // also pp.size(), pp.resize(n)
```

- **Options**: `fs`, `net`, `proc`, `env` (bools, default false; a worker never gets one the caller lacks), and
  `timeout` (ms per task, enforced inside the worker: `TimeoutError`). The pool also applies the matching OS sandbox
  to each worker (no network / no exec / no file writes unless enabled).
- **Time limits in the caller**: `f.get(ms)` raises `TimeoutError` after `ms`. A `get` inside a `limit {.timeout}`
  block or a run with `--timeout` raises `TimeoutError` at that deadline. `shutdown()` and closing the pool never
  wait past that deadline: workers still busy are killed.
- **Errors** keep their type (`ValueError` in the task is `ValueError` in the caller). A task the worker's static
  checks refuse because it is certain to use a capability the worker lacks raises `PermissionError`. A crashed worker raises
  `SignalError` and is replaced. A task's print output (up to 1 MiB) is replayed in the caller when its result is
  collected.
- **Limits**: each worker counts against `procs`; the pool and each future count against `handles`.
- Programs that embed the engine and may use spawn mode must call
  `forge::script::run_process_worker_if_requested(argc, argv)` first in `main()`. The `forge` CLI does.

## Remote execution

`script/src/remote.cpp`. Both calls go to a `forge rpc-serve --allow-scripts` node ([rpc.md](rpc.md)) and need the
`net` capability. Address `host:port` (or `rpc:host:port`, `[v6]:port`). Options: `token`, `key` (secret-key file),
`tls`, `ca`, `fingerprint` (SHA-256 hex), `server_name`, `tls_cert`/`tls_key` (mTLS client certificate),
`timeout_ms`, `name`, `args` (list), `signature` (the script's detached `.sig` text, sent with the source for nodes
whose policy has `integrity.scripts`; `name` must then be the signed file name). Credentials default to the
`FORGE_RPC_*` environment.

```cpp
auto r = forge::remote("box2:7000", "std::println(\"hi\");", {{"token", tok}});   // {ok, output, error_type, error}

int k = 10;
std::map<std::string, int> weights = {{"a", 2}};
auto score = [k, &weights](std::string key, int x) { return weights[key] * x + k; };
int v = forge::remote_call("box2:7000", score, {"a", 5}, {{"token", tok}});        // 20, computed on box2
```

`forge::remote_call(node, fn[, args[, opts]])` ships a lambda or free function:

- **What travels**: the function's source text plus every variable it uses from outside its body (captured locals,
  globals, and other script functions it calls, shipped the same way). Built-ins are not shipped. On the node,
  captures are copies: assignments to them are not seen by the caller.
- **Plain data only**: captured values and arguments must be nil, bool, int, double, string, list or map, nested
  to any depth. Ints and doubles stay distinct (typed encoding `forge::__marshal`/`__unmarshal`). Handles, objects,
  `this`, methods, classes, tasks, channels and natives are refused before connecting with a `TypeError` naming the
  variable. Two different values under one captured name, and namespaced names, are refused too.
- **Sandbox**: the function runs like a `forge::remote` script, with the node's timeout, memory cap and
  capabilities.
- **Result**: the return value comes back typed. An exception keeps its type (message prefixed `remote: `; a
  function the node's static checks refuse for a denied capability raises `PermissionError`); the
  node's print output is replayed first. Authentication failures raise `PermissionError`, connection failures
  `IOError`.

Tests: `remote_scripts` and `remote_call_functions` in `tests/test_script.cpp` (loopback, in-process node).

## AI-assisted commands

`script/src/ai.cpp`. Drafted code never runs without confirmation.

| call | result |
|---|---|
| `forge::ask(prompt[, context])` | reply text |
| `forge::ask_json(prompt[, context])` | reply parsed as JSON into script values (`AIError` if not JSON) |
| `forge::explain(text)` | short explanation of an error or output |
| `forge::suggest_cmd(task[, opts])` | one drafted shell command (string); never run |
| `forge::ai_run(task[, opts])` | `{command, ran, result}`: drafts a command, runs it through `forge::<shell>` only if confirmed |
| `forge::ai_eval(task[, opts])` | `{code, ran, result}`: drafts script code, runs it with `eval` only if confirmed |

Options: `{"context", text}`, `{"shell", name}` (any backend in `forge::shells()`, default `sh`), `{"confirm", fn}`.

**Backend**, first match wins:

1. the host's `options::ai` callback;
2. **in-process model**: `--ask-model M.fpk`, `options::ask_model` or `$FORGE_ASK_MODEL` names a `.fpk`, and the
   host registered a generator. `forge_script` does not link the inference module; a host that does calls
   `forge::script::install_local_ai()` from `<forge/script/local_ai.hpp>`. The `forge` CLI does, so `forge script`
   and rpc-serve script children have it. The model is loaded once per path; each call opens a fresh session,
   decodes greedily (same prompt, same reply), formats the prompt as `system: …\nuser: …\nassistant:`, and stops at
   `--ask-max-tokens` / `options::ask_max_tokens` / `$FORGE_ASK_MAX_TOKENS` (default 256). `ask_json` constrains
   decoding to a JSON value with the grammar sampler; a value cut off by the token cap raises `AIError`. No `net`
   capability is needed. If a model is configured but the host has no generator, the endpoint below is used if set,
   else `AIError`;
3. `$FORGE_AI_ENDPOINT`: an OpenAI-compatible `/v1/chat/completions` server such as `forge-serve`
   (`FORGE_AI_MODEL`, `FORGE_AI_KEY` optional). Needs the `net` capability.

**Confirmation**: `{"confirm", fn}` per call, else `options::confirm`. `forge script` asks `Run it? [y/N]` on the
terminal. With no confirmer, `ran` is false and nothing runs (rpc-serve children never confirm). `ai_run` goes
through the normal shell call, so capabilities and child limits apply.

Tests: `ai_assisted_commands` (stub backend) and `ask_local_model` (a tiny random Llama built in-process, checked
against direct greedy decoding) in `tests/test_script.cpp`.

## Automation

Library calls, not keywords (`script/src/automation.cpp`):

```cpp
auto p = forge::pipeline({{"cache", ".forge-cache"}});          // optional input-fingerprint cache file
p.job("gen", [&]() { return forge::sh("./gen.sh").code; });
p.job("build", {"gen"}, [&]() { /* ... */ }, {{"inputs", std::vector<std::string>{"src/*.cpp"}},
                                              {"outputs", std::vector<std::string>{"out/app"}}, {"retries", 1}});
auto r = p.run({{"parallel", 4}});   // also: targets, keep_going, dry_run, force, raise
// r["ok"], r["order"], r["jobs"][name] = {status: ok|skipped|failed|cancelled|would_run, ms, value, error, error_type}

forge::watch("src/**/*.cpp", [&](auto changes) { /* ... */ return true; },   // false stops
             {{"debounce_ms", 100}, {"count", 1}, {"for", "10m"}, {"ignore", ignores}, {"polling", false}});

forge::every("10m", fn, {{"count", 3}});  forge::at("02:00", fn);  forge::cron("*/5 * * * *", fn);
auto s = forge::scheduler(); s.every("1h", f, {{"times", 2}}); s.cron("@daily", g); s.upcoming(); s.run({{"for", "1d"}});
forge::cron_next(expr[, unix_s]);  forge::duration_ms("1h30m");
```

- **Pipelines**: a DAG with unknown-dependency and cycle checks (`PipelineError`); `p.order()` gives the run order.
  A failed job cancels its dependants, and without `keep_going` everything not yet started. A job is skipped when
  all outputs exist and are newer than its inputs, or its input fingerprint (path, size, mtime) matches the cache,
  unless a dependency re-ran or `force` is set. `parallel > 1` runs each ready wave as tasks (same copy rules as
  `std::async`). `TimeoutError` and `OutOfMemoryError` are never retried or swallowed.
- **Watch**: `tree_watcher` (inotify, FSEvents, ReadDirectoryChangesW, or polling) with debounce; events are
  filtered by the glob relative to the pattern's literal root.
- **Schedules** run on the calling thread, sleeping in slices so timeouts apply. A callback returning `false` stops
  its entry; `s.cancel(id)` removes one. Cron is 5-field POSIX (lists, ranges, steps, `@daily` etc.;
  day-of-month and day-of-week OR together when both are set) in local time. To survive reboots, install the script
  as a service with `service.install`.

## Modules and plugins

```cpp
#include "util.cpp"            // splice another script's names in
import "geom" as g;            // module object: g.area(2, 3)
import tools.strings;          // dotted form = tools/strings
auto m = forge::import("strs");                 // at run time
import "libhello_plugin.so" as hello;           // native plugin (.so/.dylib/.dll), or forge::plugin(path)
hello.add(2, 3.5); hello.VERSION;
forge::module_path(); forge::modules();
```

- A name without an extension tries `name.fs`, `name.cpp`, `name/mod.fs`, `name/mod.cpp`.
- Search order: the importing script's directory, the engine base directory, `$FORGE_SCRIPT_PATH` (`:` or `;`
  separated; `;` only on Windows), `~/.forge/modules`, the current directory. Absolute paths are used as given.
  Modules load once per engine; import cycles raise `ImportError`. Importing needs the `fs` capability.
- Native plugins use the C ABI in `script/include/forge/script/plugin.h`: export
  `forge_script_plugin_init(const forge_plugin_api*)`, register functions with `def(reg, name, min, max, fn, user)`
  and constants with `constant(reg, name, json)`. Arguments and results cross as JSON; a non-zero return raises
  `PluginError`. Example: `examples/script_plugin/hello_plugin.c` (CMake target `forge_script_example_plugin`).
  Loaded libraries stay loaded for the life of the process.

## Reflection

Runtime reflection in the spirit of C++26 reflection (P2996), as ordinary built-ins rather than a `^^` operator. Same
behaviour on both execution paths. Code: `script/src/reflect.cpp`. Example: [examples/script/reflection.cpp](examples/script/reflection.cpp).

The parser records class `[[attributes]]`, and for each member its name, C++ type name, `const`/`static`, access,
attributes and line; for functions the parameter types, return type and qualifiers.

| `forge::reflect::` built-in | result |
|---|---|
| `typeof(v)`, `kind(v)`, `type_info(v\|T)`, `find_type("Name")` | type name, kind, or an info map (name, kind, bases, fields, methods, attributes) |
| `fields(T)`, `statics(T)`, `methods(T)`, `bases(T)`, `is_base_of(B, D)`, `has(T, "m")` | member lists as maps `{name, type, const, static, access, attributes, line}` |
| `attributes(T[, "member"])`, `attributes(fn)` | attribute texts, e.g. `"key(\"x\")"` |
| `get(obj, "f")`, `set(obj, "f", v)` | read or write a field by name |
| `invoke(obj, "m", args...)`, `call("fn", args...)`, `construct("T"\|T, args...)` | call or construct by name |
| `enum_values(E)`, `enum_name(E, v)` | `[{name, value}]` in declaration order; value back to name |
| `symbols([module])` | names a module (or the globals) defines |
| `signature(fn\|T\|"name")` | `{name, kind, returns, params[{name, type, default, ref, out, const}], arity, variadic, const, static, virtual, access, attributes, file, line}` |

`forge::typeof` and `forge::type_info` are also available without `reflect::`.

Reflection never bypasses the language:

- Private and protected members are listed, but `get`/`set`/`invoke` on them raise `AccessError`.
- `set` on a `const` member or a frozen object raises `TypeError`; so does `invoke` of a non-`const` method on a
  frozen object.
- `set` while the object or field is borrowed (e.g. during iteration) raises `BorrowError`.
- `set` checks the value against the member's declared type. Enums are read-only.
- `set`, `invoke`, `call` and `construct` need the `reflect` capability (on by default; off with
  `forge::sandbox(std::map<std::string, bool>{{"reflect", false}}, fn)`, then `PermissionError`). Read-only queries
  always work.

Test: `reflection` in `tests/test_script.cpp`.

## Built-in reference

Modules are reachable as `forge::<module>::<name>` and as `<module>.<name>` (e.g. `forge::fs::read` or `fs.read`).
Arguments marked by a capability raise `PermissionError` when it is off.

| module | functions |
|---|---|
| core | `print`, `write`, `str`, `repr`, `type`, `len`, `int`, `float`, `bool`, `range`, `abs`, `min`, `max`, `sorted`, `assert`, `sleep(ms)`, `eval`, `weak`, `freeze`, `is_frozen`, `channel`, `dispose`, `parallel`, `limit`, `sandbox`, `run`, `ProcessPool` |
| `fs` (fs) | `read`, `write`, `append`, `exists`, `is_dir`, `stat`, `list`, `mkdir`, `remove`, `rename`, `copy`, `move`, `open`, `glob`, `walk`, `match`, `symlink`, `readlink`, `chmod`, `mode`, `touch`, `set_mtime`, `write_atomic`, `temp_dir`, `temp_file`, `make_temp_dir`, `disk_usage`, `lock`, `watch`, `watch_tree` |
| `path` | `join`, `base`, `dir`, `ext`, `abs` |
| `os` | `platform`, `pid`, `cwd`, `chdir`, `env`/`setenv` (env), `cpu_count`, `total_memory`, `home`, `caps`, `processes`, `signal`, `user`, `dir`, `release`, `uptime`, `loadavg`, `memory`, `is_tty`, `term_size`, `devices`, `clipboard` |
| `proc` (proc) | `run`, `start`, `which` |
| `service` | `unit` (render a systemd unit or launchd plist), `install`, `uninstall`, `start`, `stop` (proc), `status` |
| `net` (net) | `resolve`, `interfaces`, `port_open`, `http`, `parse_url` |
| `time` | `now`, `ms`, `ns`, `iso`, `parse`, `utc_offset`, `zone` |
| `json` | `parse`, `dump(v[, pretty])` |
| `hash` | `blake3` |
| `re` | `match`, `find_all`, `replace` |
| `math` | `pi`, `e`, `sqrt`, `floor`, `ceil`, `round`, `pow`, `log` |
| `gc`, `mem` | `gc.collect()`, `gc.stats()`, `mem.stats()` |
| shells | `forge::sh`, `bash`, `pwsh`, `cmd`, ... (one per backend), `forge::shell`, `forge::host_shell`, `forge::shells` |
| `forge::text::`, `forge::locale::` | Unicode text and locale functions; see [text-locale.md](text-locale.md) |
| `forge::mobile::`, `forge::sensors::`, `forge::touch::` | device access (env); see below |

There are no script bindings for models, GPU devices or `forge-serve` (`model.*`, `gpu.*`, `serve.*` are not
implemented). Use the AI commands above for in-process inference.

### Mobile and device built-ins

Gated by the `env` capability. Methods copy the owner per call, so disposing a handle while a task uses it is safe.

```cpp
auto dev = forge::mobile::device();              // or ({{"sim", trace_json}}), ({{"sim_file", p}}), ({{"null", true}})
std::println("{} {}", dev.backend(), dev.info()["form"]);
auto acc = forge::sensors::open("accelerometer", 100, dev);   // forge::sensors::list(dev) lists kinds
forge::channel<int> ch(64);
while (acc.pump(ch, 32, 50) == 0) {}            // never blocks; stops at channel capacity
auto e = ch.recv();                              // {"type":"sensor","kind":..., "t_ns":..., "v":[...]}
auto touch = forge::touch::open(dev);
for (auto g : touch.gestures(16)) std::println("{}", g["kind"]);   // tap, double_tap, long_press, swipe, pinch, rotate
```

- Sensor handle: `poll(timeout)`, `pump(ch, max, timeout)`, `kind()`, `dropped()`, `finished()`, `close()`.
- Touch handle: `poll`, `gestures`, `pump`, `finished`, `close`.
- Device handle: `backend`, `display`, `power`, `connectivity`, `info`, `vibrate`, `notify`, `clipboard_get`/`set`,
  `brightness`, `set_brightness`, `keep_awake`, `torch`, `open_url`, `permission`, `request_permission`, `sim_log`,
  `close`.
- Unsupported features raise `NotSupportedError`. See [mobile.md](mobile.md).

## Write once, emit anywhere

`forge script --emit <target> file.cpp [-o out]` translates a script into a standalone script for that shell. The
output runs without FORGE. With `-o` the file is made executable. `--emit auto` uses `host_shell()`; `--emit android`
is mksh. `--emit ast` dumps the parse tree. `--emit-targets` lists every backend with an emitter.

```
$ forge script --emit bash -o hello.bash examples/scripts/hello_emit.cpp && ./hello.bash
$ forge script --emit pwsh -o hello.ps1  examples/scripts/hello_emit.cpp && pwsh -File hello.ps1
```

**Portable subset** (the same for every target):

- `int`, `bool`, `std::string`, `auto` locals; free functions with `int`/`std::string` parameters, which may
  recurse.
- `+ - * / %` (C truncation), comparisons, `&& || !` (short-circuit), `?:`, `+=`, `++`, string `+`, `.size()`,
  `.empty()`, `std::to_string`, `std::stoi`.
- `if`/`else`, `while`, `for(;;)`, range-`for` over a braced list of literals, `break`, `continue`, `return`,
  `std::exit`.
- `std::println`/`std::print` with plain `{}`, `std::cout << ... << std::endl`, `std::cerr`.
- `os.env(name, default)` / `std::getenv`, `os.setenv`, `os.cwd()`, `os.platform` (linux, windows, macos, freebsd,
  android, ios, other).
- `fs.exists`, `is_dir`, `is_file`, `mkdir`, `remove`, `write`, `append`, `read`, and the `std::filesystem`
  spellings of these. (`fs.is_file` exists only in emitted code; in the interpreter use `fs.stat(p)["is_file"]`.)
- `forge::sh(tmpl, args...)` and `forge::shell(...)` results (`.out`, `.code`, `.ok`), with `{}` quoted for the
  target.

Anything else is an error naming the file and line, for example
`x.cpp:1: emit bash: classes and structs can't be expressed in bash ...`. Not expressible anywhere: classes,
lambdas, containers and maps, exceptions, floating point, templates, tasks, `defer`.

| target | extension | not expressible on this target |
|---|---|---|
| sh, bash, zsh, dash, ash, yash, ksh, mksh/android, oksh, posh, hush, osh | `.sh` | – |
| fish | `.fish` | – |
| pwsh, powershell (output avoids 7-only syntax) | `.ps1` | – (`forge::sh` needs an `sh` on PATH on Windows) |
| xonsh (plain Python) | `.xsh` | – |
| elvish (0.20+) | `.elv` | – |
| nu | `.nu` | functions that read or write script-level variables |
| tcsh, csh | `.csh` | functions other than `main`, `fs.read`, `forge::sh` results |
| cmd | `.cmd` | functions, `.size()`, string ordering, loops inside if/for blocks, break/continue in range-for, `fs.read`, `forge::sh` |
| rc, es | `.rc`, `.es` | string ordering |
| ysh | `.ysh` | string ordering |

**Tests.**

- `emit_golden_all_shells` (`tests/test_script.cpp`) emits `tests/data/emit/functions.cpp`, `flat.cpp` and
  `batch.cpp` for every installed backend with an emitter, runs each output with the real shell, and requires the
  interpreter's stdout and exit code. Missing shells are skipped. ctest adds `FORGE_TEST_SHELL_PATH` (CMake cache
  variable; default `../opt/shells/bin` when it exists) to `PATH`.
- `emit_text_golden` compares the emitted text for ysh, rc and es with `tests/data/emit/expected/`, so emitter changes
  are caught without those shells (`FORGE_UPDATE_EMIT_GOLDEN=1` rewrites the files).
- Which shells are marked tested is in [shells.md](shells.md). Emit targets marked untested in code: posh, real BSD
  csh, Windows PowerShell 5.1, cmd. The Windows CI job runs the emit goldens for cmd, pwsh and powershell 5.1
  (`FORGE_TEST_ONLY=emit`). mksh as Android's `/system/bin/sh` is untested.
- `ctest -L cross` syntax-checks the emitter and backends for FreeBSD, macOS, Windows (MinGW) and musl with zig.
  These tests exist only when CMake finds zig.

**OS targets**: Linux output is tested. macOS and FreeBSD use the same POSIX, fish, zsh, nu, elvish, xonsh and pwsh
code (`os.platform` maps `uname -s`); macOS CI runs the goldens only for the shells its runner has, FreeBSD output has
not been run. Windows uses pwsh/powershell or cmd output. Android output (`--emit android`) is untested. iOS has no
user shell; use the embedded interpreter (`bindings/apple`).

## Static checks

`forge script --check file` (add `--json` for `[{"file","line","col","severity","id","message"}]`) runs the static
checker (`forge/script/check.hpp`, `script/src/check.cpp`) without running anything; exit 2 on errors. Every normal
run also runs it first (not the REPL): a script with any **error** is refused with exit 2 and `StaticCheckError`.
`--strict-check` refuses on warnings too; `--no-check` turns the checks off. Errors are reported only when the
operation is certain to fail whenever it is reached; anything uncertain is a warning. The runtime checks remain the
backstop.

Suppress with `// forge-check: ignore` or `// forge-check: ignore[id,id]` on the line or the line above, or
`// forge-check: ignore-file[id]` for the whole file.

| id | severity | what |
|---|---|---|
| `syntax` | error | parse failure |
| `use-after-move` | error on every path, warning on some | flow-sensitive over function locals and top-level globals (branches, `&&`/`\|\|`/`?:`, loops, try/catch, field paths, references) |
| `unique-copy` | error | copy-initialising from a `std::make_unique` local |
| `iter-mutation` | error for a literal or container-typed local, else warning | size-changing method on the container a range-for is iterating |
| `capability` | error, or warning when the value may differ at run time | fs/net/proc/env built-ins, `run()` and shell calls when `--no-*` or an enclosing literal `forge::sandbox` denies them; also through locals, containers, function returns and parameters holding a built-in, `eval("literal")`, and imports (an import is itself an error with `fs` denied). Warnings for indirect calls: see *Indirect calls* below |
| `limit-option` | error | in a literal `limit {...}` or `forge::limit({...}, fn)`: unknown key, malformed size or duration string, or a string for `allocs`/`handles`/`procs` |
| `pool-task` | error | `submit`/`map` with a literal task source that does not parse, does not define `task`, or whose `task` does not take one argument |
| `pool-json` | error | a lambda or shell result passed as a `ProcessPool` argument |
| `thread-send` | warning | argument to `forge::spawn`/`std::thread`/`std::async` that is not a literal, moved, frozen, const or `forge::shared` |
| `thread-capture` | warning | a thread lambda reading or writing a captured mutable local |
| `channel-send` | warning | the same rule for values sent on a channel |
| `lock-order` | warning | `forge::shared` values locked in a cyclic order by nested `with` (cycles of any length) |
| `unreachable` | warning | statements after return/throw/break/continue |
| `unused-variable` | warning | local initialised from a literal and never read (names starting `_` exempt) |
| `shadow` | warning | local redeclaring a name from an enclosing block of the same function |
| `missing-return` | warning | non-void function that can fall off its end |
| `arg-count` | warning | call of a top-level script function with too few or too many arguments |
| `catch-order` | warning | a catch after `catch (...)` or after a built-in ancestor (e.g. `OSError` before `TimeoutError`, `std::runtime_error` before `OSError`) |
| `infinite-loop` | warning | `while (true)`/`for (;;)` with no break, return, throw or exit |
| `format` | warning | format literal whose placeholder count does not match the arguments |
| `deprecated` | warning | removed std APIs (`std::random_shuffle`, `std::auto_ptr`, `std::bind1st`, ...) and script functions or methods marked `[[deprecated]]` |
| `type` | warning | arithmetic or comparison between a string and a number, both literals or literal-initialised `const` locals |

The checker makes two passes: the first summarises functions and lambdas (returned built-ins, built-ins stored in
fields), the second reports, so a call may come before the definition. Imports resolve like the runtime; extra
directories with `--module-dir DIR` or `check_options::module_dirs` (the engine passes its base directory).

**Indirect calls** (`script/src/capflow.hpp`). The `capability` errors above see a built-in only where the call
names it as `lib.fn(...)` (`fs.read(p)`) or `run(...)`. A second, whole-script pass follows callables and reports
`capability` **warnings**; it runs with `--check` and `--strict-check` (a normal run drops warnings, so it is skipped
there). It is flow-insensitive:

1. Every local, global, field name and function return gets the set of callables it may hold: script functions,
   lambdas, methods, and capability built-ins in any spelling (`forge::fs::read`, `fs::read` after
   `using namespace forge`, `forge::sh`, `std::getenv`, ...). Sets are fed by initialisers, assignments, arguments,
   returns, range-`for`, container literals and inserts, wrappers (`std::function`, `std::bind`, `std::move`,
   `std::make_shared`) and element accessors, to a fixed point. A container or struct is the union of what it
   holds.
2. Each function gets a summary, also to a fixed point over the call graph: the capabilities its body may need
   through the built-ins it calls, the functions it may call and the callables it hands to a built-in that may call
   them (`std::for_each`, `std::async`, `forge::sandbox`, ...). A parameter the function itself calls stays
   symbolic, so `apply(pure_fn)` does not inherit what `apply(reads_files)` needs.
3. Method calls resolve through the classes the receiver may be, a declared base type standing for all its
   subclasses (virtual dispatch); when nothing is known about the receiver, every class with a method of that name
   counts.
4. A callee nothing is known about (a value from `eval` or `forge::reflect`, a member of an imported module, a
   parameter no call site passes anything to) is treated as any function or built-in whose address is taken
   anywhere in the script.

A call is reported when it sits where a capability is denied (by the run, or by an enclosing
`forge::sandbox(<literal options>, [] { ... })` in any literal form: `{{"fs", false}}`,
`std::map<std::string, bool>{{"fs", false}}`) and may reach a built-in that needs it. The message gives one path,
e.g. `'apply' may need the 'fs' capability (denied by the enclosing sandbox block): apply -> its parameter 'f' ->
lambda (line 4) -> forge::fs::read (line 4)`. With a capability denied for the whole run, a script function is not
reported at its call sites: the line that uses the built-in is.

Known gaps: fields are tracked by field name, not per object, and containers per variable, not per element (both
can warn about a capability-free callable stored next to one that needs a capability); a write through an iterator
is not followed back to its container; a function is summarised once for all its call sites except for the
parameters it calls directly; `eval` of a non-literal string, functions of imported modules and methods of handles
opened before the block (`file.read()`) are not followed; sandbox options held in a variable are not read; a lambda
written inside a denied block is checked there even if it only runs later, outside. The `reflect` capability is not
checked statically. Non-literal paths to `forge::import`/`plugin` are not followed. Not built: undefined-variable
checks, type inference beyond literals.

Automated tests: `limit-option` (`limit_block_syntax_and_check` in `tests/test_script.cpp`) and the indirect-call
`capability` warnings (`static_capability_indirect_*` in `tests/test_script_security.cpp`, positive and negative
cases per indirect form; run-time counterpart `tests/script/indirect_calls.fs`). The other checks run on every test
script but are not asserted individually.

## Embedding

`forge::script::engine` (`forge/script/script.hpp`):

- `engine(options)`, `run_file(path)`, `run(src, name)` (VM), `eval(src)` (tree-walker, keeps globals),
  `repl()`, `interrupt()`, `reset()`, `disassemble(src)`, `type_of(expr)`, `stats()`, `sandbox_status()`,
  `cache_status()`. Results carry `ok`, `value`, `error_type`, `error`, `line`, `exit_code`.
- `options` holds the limits, capabilities, static-check, OS-sandbox and cache settings above, plus `argv`, `out`
  (print sink), `ai`, `confirm`, `ask_model`, `ask_max_tokens`, `jit_threshold`, `deadlock_ms`, `cancel_signals`.
- **Host bindings**: `define(name, min, max, fn)` adds a native function (`name` may be qualified, e.g.
  `"ui::find"`). Values cross as JSON text. A script function passed as an argument arrives as `{"$callback": id}`;
  call it later with `call(id, args_json)` and free it with `release(id)`. A host function that throws raises
  `HostError` in the script; a `forge::security` or typed OS exception keeps its class (`IntegrityError`,
  `PermissionError`, `FileNotFoundError`, ...). Test: `host_bindings`.

## Testing

- `tests/test_script.cpp`: every program runs through the VM and the tree-walker and must give the same output;
  per-feature cases, shell and emit goldens (shell tests skip missing shells; POSIX shell cases are not built on
  Windows).
- `tests/test_script_golden.cpp`: every `tests/script/*.fs` on both paths against `<name>.out`
  (`FORGE_GOLDEN_UPDATE=1` rewrites them).
- `tests/test_script_os_sandbox.cpp` (Linux): OS enforcement of the capability flags.
- `tests/test_script_oserror.cpp`: OS error mapping, fault handling, `ProcessPool` and its time limit.
- `fuzz/fuzz_script.cpp`: fuzzes the lexer, parser, resolver, compiler and verifier without executing anything.

## Platform notes and limits

- CI (`.github/workflows/ci.yml`) runs ctest, including the script tests, on Linux, macOS and Windows (MSVC and
  MinGW), plus `test_script` under ThreadSanitizer and AddressSanitizer on Linux. The OS-sandbox test is
  Linux-only. `isolate` blocks are POSIX-only, and the Windows OS sandbox cannot drop fs or net access from a
  running process.
- iOS: no `ProcessPool`, no `isolate`, no emitted shell scripts.
- Not implemented: `finally`, shell-block syntax (`sh { ... }`), `sandbox {...}` block syntax, REPL history and
  completion, script bindings for models, GPU and serving.
