# FORGE TODO

## [x] Add a logging module (logger) to FORGE (done: PR forge-logger)
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
- **Status: done** in `core/log.hpp` + `tests/test_log*.cpp`: levels incl. fatal, per-module overrides
  (`FORGE_LOG`, `configure`, `set_module_level`), stderr/file (size rotation)/syslog/callback sinks, JSON lines,
  one locked write per line, `FORGE_LOG_MIN`/`FORGE_LOG_MIN_LEVEL` strip and `FORGE_LOG_*` macros that skip
  argument evaluation. Clock and thread ids come from the OS wrapper.
- [ ] Still open: async/buffered sink, Windows event log sink, a CMake option for the strip level.
