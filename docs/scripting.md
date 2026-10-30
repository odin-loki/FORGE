# FORGE scripting design (`forge script`)

Status: **design, not implemented.** All higher-level features below are approved and in scope.

## Purpose

A small, interpreted, dynamic language built into the `forge` binary. One script runs the same on
Windows, Linux and macOS, because built-ins go through the [OS wrapper](os-wrapper.md). When you need
a real shell, embedded blocks call it and give back structured results.

```
forge script build.fs [args...]   # run a file
forge script -e 'print(os.cpu_count())'
forge script                      # REPL
```

## Language

A small managed language in the style of the .NET Micro Framework: dynamic by default, memory-safe by
construction, with deterministic cleanup. There is **no `unsafe` keyword**, no pointers, no address-of,
no pointer arithmetic and no way to reinterpret memory. Native code is reachable only through built-ins
and vetted plugins.

### Syntax

- `let x = 1` (rebindable), `const y = 2`, `if/elif/else`, `while`, `for x in xs`, `match`,
  `fn name(a, b) { ... }`, closures, `return`, `break`, `continue`.
- String interpolation `"hello ${name}"`, list/map literals `[1, 2]`, `{a: 1}`.
- Pipes: `a |> f(b)` means `f(a, b)`; works for values and for command output.
- `eval(src)` and `load(file)` for runtime code; `import "mod"` for modules.

### Type system

- **Dynamic with optional annotations** (gradual typing): `fn add(a: int, b: int) -> int`. Annotations
  are checked at call/assignment boundaries at run time and used by the compiler to pick specialised
  opcodes; unannotated code stays fully dynamic.
- Built-in types: `nil`, `bool`, `int` (64-bit, overflow throws `OverflowError`), `float`, `string`
  (immutable UTF-8), `bytes`, `list`, `map`, `set`, `fn`, `handle` (opaque OS resource), `task`.
- **Classes**: reference types, single inheritance, `init`/`dispose`, fields declared up front (no
  ad-hoc field creation unless the class is marked `dynamic`).
- **Structs**: value types, copied on assignment, no inheritance; good for small records.
- **Interfaces**: named method sets; `implements` is checked when the class is defined; `x is Disposable`.
- **Generics-lite**: type parameters on annotations only (`list<int>`, `map<string, Job>`,
  `fn first<T>(xs: list<T>) -> T`). Erased at run time, checked at boundaries; no specialisation or
  variance rules. Enough to document intent and catch mistakes without a heavy type checker.
- `nil` safety: `T?` marks nullable annotations; `?.` and `??` operators.

### Execution model (hybrid)

- Scripts and modules are compiled to a **compact, verified stack bytecode** (NETMF uses a compact IL
  interpreter for the same reasons: small footprint, fast startup, no JIT, predictable on small
  devices). Bytecode is cached as `.fsc` next to the source (keyed by content hash and VM version).
- The **REPL and `eval`** use a direct **AST tree-walker** path: no compile step, best error locations,
  instant feedback. Functions defined at the REPL are compiled to bytecode lazily once called N times
  (default 2), so hot REPL code also runs on the VM.
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
  `IndexError`, `KeyError`, `OverflowError`, `OutOfMemoryError`, `TimeoutError`, `CancelledError`,
  `ObjectDisposedError`, `ShellError`, `PermissionError`, `LimitExceededError`.
- C++ exceptions and `std::expected` errors from `forge::os`, `forge::mem` and other modules are mapped
  to script exceptions at the built-in boundary; no C++ exception ever escapes into or past the VM.
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
  memory budget (`FORGE_MEM_LIMIT`, `--mem-limit`, or a `limit {mem: ...}` block). When an allocation
  would exceed the budget the runtime first evicts caches and runs the cycle collector, then throws a
  catchable `OutOfMemoryError` instead of being OOM-killed. A small reserve is kept so handlers and
  `finally` blocks can still run.
- `mem.usage()`, `mem.budget()`, `gc.stats()` expose live bytes, object counts and collector runs.

### Resource limits

Per script, per task, or per `limit {}` block: heap bytes, total allocations, instruction count /
CPU time, wall-clock timeout, recursion depth (default 10 000 frames, checked, never a C++ stack
overflow), open handles, child processes, and output size. Child processes get OS limits (cgroups v2,
Job Objects, setrlimit). Exceeding any limit throws `LimitExceededError` (or `OutOfMemoryError`).

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
- Errors via exceptions/`std::expected` internally; all mapped to script exceptions at the boundary.
- Enforced by clang-tidy (cppcoreguidelines, bugprone), `-Wall -Wextra -Werror`, ASan/UBSan/TSan CI
  jobs and fuzzing of the lexer, parser, compiler and verifier.

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

## Built-ins (from the OS wrapper)

`fs.*` (read, write, glob, walk, copy, remove, watch), `os.*` (env, cwd, platform, caps, sysinfo,
user), `proc.*` (spawn, kill, list), `net.*` (resolve, http_get, tcp), `time.*`, `json.*`, `path.*`,
`str.*`, `re.*` (regex), `reg.*` (Windows).

## FORGE built-ins

- `model.load(path)`, `model.generate(prompt, opts)` (inference via `mind`).
- `gpu.devices()`, grid dispatch helpers.
- `mem.budget()`, `mem.usage()`, `mem.limit(size)` from `forge::mem`.
- `serve.start(opts)` / `serve.stop()`.

## Higher-level features (approved, all in scope)

1. **Parallel tasks** — `spawn fn`, `await`, `parallel(list, fn, n)`; thread pool from `forge::os`.
2. **Resource-limited runs** — `limit {mem: "2G", cpu_s: 60, timeout: "5m"} { ... }` applies to
   child processes (cgroups / Job Objects) and to script work via the `forge::mem` budget, so
   jobs fail cleanly instead of OOM-killing the box.
3. **Job pipelines** — declare steps with dependencies (`job build after fetch { }`), run as a DAG,
   skip up-to-date steps, cache results; a portable make/CI runner.
4. **File-watch triggers** — `watch "src/**/*.cpp" { |changes| ... }` with debounce.
5. **Scheduled tasks** — `every "10m" { }`, `at "02:00" { }`; optionally installed as an OS service.
6. **AI-assisted commands** — `ai "find the 5 largest files here"` asks a local model to draft a
   script, shows it, and runs only on confirmation; `ai.explain(r.err)` for failed commands.
7. **Remote execution** — `remote "host:7000" { ... }` ships the block to `forge rpc-serve`.
   Requires adding auth/TLS first (rpc-serve has none today).
8. **Modules/plugins** — `import` from a search path, native plugins via `os::library`.
9. **Sandboxed runs** — run untrusted scripts with fs/net capabilities off, using `sandbox.hpp`.

## Safety

- Scripts run with the user's privileges; nothing is sandboxed unless asked.
- AI-generated code is never run without confirmation.
- Remote execution stays disabled until rpc-serve has authentication.

## Testing

Golden-output tests for the language (`tests/script/*.fs` + expected output), unit tests per
built-in, shell-block tests that skip if the shell is missing, and a parser fuzz target.

## Phasing

All phases are in scope.

1. **Core runtime**: value model, ARC heap on `forge::mem`, weak refs, exceptions, lexer/parser,
   tree-walker, REPL, `run` + `sh/bash/zsh/ps/cmd` blocks, fs/os/json built-ins, `using`/`defer`.
2. **Bytecode VM**: compiler, verifier, `.fsc` cache, lazy REPL compilation, dual-path golden tests;
   classes, structs, interfaces, optional annotations and generics-lite.
3. **Concurrency and limits**: tasks, async, channels, freeze/move sharing rules, cycle collector,
   resource limits and `OutOfMemoryError`, FORGE built-ins (model, gpu, mem, serve).
   Features 1-2 (parallel tasks, resource-limited runs).
4. **Automation**: features 3-5 and 8 (job pipelines, file-watch triggers, scheduled tasks, modules
   and plugins).
5. **Advanced**: features 6, 7 and 9 (AI-assisted commands, remote execution once rpc-serve has
   auth/TLS, sandboxed runs).
