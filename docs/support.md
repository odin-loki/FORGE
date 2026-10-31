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
