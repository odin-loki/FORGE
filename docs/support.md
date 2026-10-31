# Support utilities (logger, processes, pool, cache, settings)

Added in PRs #5-#9. Header-only unless noted.

## Logger: `core/include/forge/core/log.hpp` (tests: `tests/test_log.cpp`, `tests/test_log_strip.cpp`)
- Levels trace, debug, info, warn, error, fatal, off; default warn. `FORGE_LOG="info,mind=debug"` sets the
  global and per-module thresholds; `set_level`, `set_module_level`, `configure(spec)` at run time.
- `forge::log::info(...)` (module "forge") or `forge::log::logger L{"mind"}`; `std::format` style.
- Sinks: stderr (default, `set_stderr`), `add_file_sink(path, max_bytes, keep)` with size rotation, syslog,
  any callback (`add_sink`/`remove_sink`). Text or JSON lines (`set_format`, `FORGE_LOG_FORMAT=json`).
  One locked write per line.
- `FORGE_LOG_MIN` / `FORGE_LOG_MIN_LEVEL` (0..6) strip calls at compile time; the `FORGE_LOG_TRACE..FATAL`
  macros skip argument evaluation when disabled.
- Not yet: async sink, Windows event log sink, CMake option for the strip level (TODO.md).

## Processes: `os/include/forge/os/process.hpp`, `core/include/forge/core/memory.hpp` (tests: `tests/test_os_process.cpp`, `tests/test_memory.cpp`)
- `spawn_spec` (argv, env, cwd, limits, output cap); `wait_any(children, timeout_ms)`, `wait_all(...)`.
- `mem::spawn_leased(budget, spec, bytes, tag)` reserves `bytes` from a `forge::mem` budget, uses it as the
  child's memory cap and releases it when the child is reaped.

## Thread pool: `support/include/forge/task/pool.hpp` (test: `tests/test_pool.cpp`)
- `parallel_for(n, grain, f)`; `submit(f, {priority, stop_token})` returns a `std::future`; higher priority
  first, FIFO within a priority; a task cancelled before it starts throws `task::cancelled` from its future.
- `pending()`, `wait_idle()`; the destructor drains the queue.

## Process pool (PENDING: branch `forge-multiprocess`, not on main yet): `os/include/forge/os/process_pool.hpp` (test: `tests/test_process_pool.cpp`)
- `forge::process_pool` (alias of `forge::os::process_pool`) mirrors the thread pool with worker processes:
  `submit(name, payload)` returns `std::future<std::string>`; `wait_all()`, `pending()`, `size()`,
  `set_workers(n)` (grow, or retire workers gracefully after their current task), `shutdown()` (drains the
  queue, sends each worker a quit message, reaps it; also run by the destructor). Worker count defaults to
  `FORGE_PROCESSES`, then `hardware_concurrency()`.
- Tasks are named `std::string(std::string)` functions registered with `FORGE_PROCESS_TASK(name, fn)` or
  `register_process_task`; payloads and results are byte strings you serialise yourself.
- Modes: `fork` (default on Linux, macOS, FreeBSD, Android) and `spawn` (re-launch this binary with
  `--forge-process-worker=...`: fork+exec on POSIX, `CreateProcess` with pipe stdin/stdout on Windows, where
  it is the only mode; `fork` throws `not_supported` there). Spawn mode needs
  `process_pool::run_worker_if_requested(argc, argv)` at the top of `main()`. On iOS the constructor throws
  `os_exception{not_supported}`.
- IPC: one pipe pair per worker, length-prefixed frames (type, id, task name, payload).
- Errors: a task that throws fails its future with `os::os_exception` (kind carried across from the worker's
  exception type). A worker that dies mid-task fails that future with `os_errc::signal` and `signo`
  (POSIX signal, or Windows access violation / illegal instruction / divide by zero as 1/2/3) or `os_errc::io`
  for a non-zero exit; the worker is respawned for the next task.
- `options.sandbox` applies `os::apply_sandbox` in every worker before its first task.
- Verified so far: Linux (all tests in `tests/test_process_pool.cpp` pass); Windows (mingw) and Android build
  cleanly. macOS, FreeBSD, Windows runtime and iOS are untested. Not exposed in ForgeScript yet.

## Cache: `support/include/forge/cache/cache.hpp` (test: `tests/test_cache.cpp`)
- `cache::store(dir, {.max_bytes})`; `put(value)` returns its BLAKE3 content key; `key_of({...})` builds memo keys.
- `get()` verifies integrity and deletes corrupt entries; writes use `os::fs::write_file_atomic`; LRU eviction
  when `max_bytes` is set; thread-safe; `store_stats`.

## Time and files: `os/include/forge/os/time.hpp`, `os/include/forge/os/fs.hpp` (test: `tests/test_util.cpp`)
- `stopwatch`, `deadline` (`after_ms`, `never`, `child(ms)` never extends past the parent).
- `fs::write_file_atomic(path, data, mode)`: temp file, fsync, rename, fsync of the directory.

## Settings registry: `os/include/forge/os/settings.hpp` (test: `tests/test_util.cpp`)
- `settings::define<T>(name, default, help)`, `get()`, `reload()`, `markdown()` (the table),
  `undocumented(prefix)`, `problems()`. Kinds string, int, bool, size (K/M/G/T, base 1024), real.
  `get_raw()` of an undefined name throws.
