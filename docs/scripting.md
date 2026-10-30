# FORGE scripting design (`forge script`)

Status: **design, not implemented.** Higher-level features below are proposals for discussion.

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

- Dynamic values: `nil`, bool, int (64-bit), float, string, list, map, function, native handle.
- `let x = 1`, `if/elif/else`, `while`, `for x in list`, `fn name(a, b) { ... }`, closures,
  `return`, `break`, `continue`, `try { } catch e { }`, `throw`.
- String interpolation `"hello ${name}"`, list/map literals `[1, 2]`, `{a: 1}`.
- Pipes: `a |> f(b)` means `f(a, b)`; works for values and for command output.
- `eval(src)` and `load(file)` for runtime code; `import "mod"` for modules.
- Implementation: hand-written lexer, Pratt parser, tree-walking interpreter in C++23
  (`script/` module, library `forge_script`). Reference-counted values; no GC pause.

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

## Proposed higher-level features (for discussion)

Ranked by value vs effort:

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

1. Core language + REPL + `run` + `sh/bash/ps/cmd` blocks + fs/os/json built-ins.
2. Parallel tasks, resource limits, FORGE built-ins.
3. Pipelines, watch triggers, scheduling, modules.
4. AI-assisted commands, remote execution (after rpc-serve auth), sandboxing.
