# Vendored SQLite

The SQLite amalgamation (`sqlite3.c`, `sqlite3.h`, `sqlite3ext.h`), unmodified, from https://www.sqlite.org/.
SQLite is in the public domain (https://www.sqlite.org/copyright.html), so it is compatible with FORGE's dual licence.

- `PIN` records the version, the download URL, the SHA3-256 that sqlite.org publishes for the zip
  (https://www.sqlite.org/download.html), a SHA-256 of the zip, and the SHA-256 and SHA3-256 of each vendored file.
- ctest `sqlite_pin` (`cmake/check_sqlite_pin.cmake`) recomputes the file hashes and fails on any difference, on a
  version mismatch with `sqlite3.h`, or on an unlisted source file in this directory.
- `.gitattributes` marks these files `-text`, so a Windows checkout does not change their line endings.

Update: download the new amalgamation, check the zip's SHA3-256 against the download page, replace the three files and
rewrite `PIN` (all fields), then run `ctest -R sqlite_pin`.

Build options (CMake target `forge_sqlite`, root `CMakeLists.txt`): FTS5, RTREE, JSON (built in), `SQLITE_THREADSAFE=2`
(multi-thread: one connection is used by one thread at a time), `SQLITE_DQS=0`, `SQLITE_DEFAULT_FOREIGN_KEYS=1`,
`SQLITE_OMIT_DEPRECATED`, `SQLITE_DEFAULT_WAL_SYNCHRONOUS=1`, `SQLITE_USE_ALLOCA`, `SQLITE_MAX_ATTACHED=125`
(`forge::db::parallel_bulk_load` attaches one staging database per partition), `SQLITE_OMIT_LOAD_EXTENSION`.
