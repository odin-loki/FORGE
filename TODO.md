# FORGE TODO

## Add a logging module (logger) to FORGE
Grow today's minimal `core/include/forge/core/log.hpp` (stderr only, `FORGE_LOG` threshold) into a proper
logger module, so PRISM can use it instead of its own logger.
- **Levels:** trace, debug, info, warn, error, fatal; global threshold plus per-component/category overrides
  (`FORGE_LOG=warn,flow=debug`), settable at run time.
- **Sinks:** stderr (default), file with rotation, syslog/journald and Windows event log via the OS wrapper,
  a callback sink for embedders (PRISM, bindings), and an optional JSON-lines structured format.
  Thread-safe, with an async/buffered option that never blocks hot paths.
- **Compile-time strip:** a `FORGE_LOG_MIN_LEVEL` macro/CMake option that removes calls below the level at
  compile time (format arguments not evaluated), so trace/debug cost nothing in release builds.
- Keep the `std::format` call style; document it in docs/ and add tests (levels, filtering, sinks, strip).
- **Status:** scheduled in the current workstream after the quantised dot kernels and Qwen 3.5. It must be
  thread-safe, use the FORGE OS wrapper for files/syslog/clock/thread ids, and ship with tests.
