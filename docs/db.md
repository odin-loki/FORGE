# forge::db: SQLite, shard sets and parallel search

Headers under `support/include/forge/db/`; link `forge::db` (CMake target `forge_db`: header-only code over the static library `forge_sqlite`). `#include <forge/db/shards.hpp>` pulls in the whole layer. Tests: `test_db`, `test_db_parallel`, `test_db_shards`, `test_task_algorithms`, `sqlite_pin`. Example: `docs/examples/db_search.cpp` (ctest `doc_example_db_search`). Benchmark: `perf/bench_db.cpp`. ANN: [ann.md](ann.md).

## Vendored SQLite

`third_party/sqlite/` holds the unmodified amalgamation of SQLite **3.54.1** (`sqlite3.c`, `sqlite3.h`, `sqlite3ext.h`), downloaded from `https://www.sqlite.org/2026/sqlite-amalgamation-3540100.zip`. `PIN` records the version, the URL, the SHA3-256 that sqlite.org publishes for the zip (`92281c86e2957f9b9305890f9ac04204caa7b677cc868dc326d942689df28e30`, compared with the download on 2026-11-18), a SHA-256 of the zip (`0a9c8e34...e6ad1f9`), and the SHA-256 and SHA3-256 of each of the three files. ctest `sqlite_pin` (`cmake/check_sqlite_pin.cmake`) recomputes them. SQLite is public domain; it is listed in `NOTICE` and in the SBOM (`tools/sbom.py`).

`forge_sqlite` (static, C99, warnings silenced on this target only) is built with FTS5, RTREE, JSON (built in), `SQLITE_THREADSAFE=2`, `SQLITE_DQS=0`, `SQLITE_DEFAULT_FOREIGN_KEYS=1`, `SQLITE_OMIT_DEPRECATED`, `SQLITE_DEFAULT_WAL_SYNCHRONOUS=1`, `SQLITE_USE_ALLOCA`, and two more: `SQLITE_MAX_ATTACHED=125` (bulk load) and `SQLITE_OMIT_LOAD_EXTENSION` (no `dlopen`, so it builds the same on every CI platform). `test_db` (`sqlite_build_options`) reads them back through `PRAGMA compile_options`. Only Linux with GCC 13 has built it so far; the other CI platforms (MSVC, MinGW, macOS, iOS, FreeBSD, Android) have not.

## The wrapper (`db/sqlite.hpp`)

```cpp
auto c = db::connection::open("a.db", {.wal = true, .synchronous = "NORMAL"});   // open_options: read_only, immutable, uri, busy_timeout_ms, cache_size_kib, mmap_size, pragmas, sorter_threads...
c.exec("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT, v BLOB)");
{ db::transaction tx(c); auto ins = c.prepare("INSERT INTO t(name, v) VALUES(?, ?)"); ins.run("a", blob); tx.commit(); }
auto st = c.prepare("SELECT id, name FROM t WHERE id > :lo"); st.bind(":lo", 0);
for (auto& r : st.rows()) use(r.get<std::int64_t>(0), r.get<std::string_view>(1));
auto [n] = c.query_row<std::int64_t>("SELECT count(*) FROM t");
```

Typed `bind` and `get<T>` (integers, bool, floating point, strings, `std::span<const std::byte>` blobs, `std::optional<T>` for NULL; `static_text`/`static_blob` bind without a copy), `transaction` (rolls back unless committed; a transaction inside a transaction is a savepoint), `connection::try_open` (returns `forge::result`), `integrity_check()`, `attach/detach`. Every failure throws `db::error`, a `forge::forge_exception` whose `errc` is mapped from the SQLite code (`busy`/IO/cantopen to `io_error`, corrupt to `integrity_error`, not-a-database to `format_error`, nomem to `out_of_memory`, constraint/misuse/range to `invalid_argument`, SQL syntax to `parse_error`) and which carries the primary and extended SQLite codes and the SQL text. One connection is used by one thread at a time (`SQLITE_THREADSAFE=2`).

FTS5 helpers (`db::fts5`): `create`, `rebuild`, `optimize`, `integrity_check`, `words`, `quote_phrase`, `query_from_text` (every word quoted, so FTS5 operators in user text are harmless), `search` (k best by `-bm25()`, ties by rowid).

## Parallel layer (`db/parallel.hpp`, `task/algorithms.hpp`)

`read_pool` (N read-only connections to one file), `rowid_partitions` / `range_sql` (fixed rowid ranges, `rowid BETWEEN :lo AND :hi`), `parallel_map`, `parallel_query`, `parallel_scan`, `parallel_filter`, `parallel_reduce`, `parallel_sort` (per-partition stable sort, then the stable parallel merge), `parallel_topk` and `parallel_bm25_topk` (order: score descending, id ascending), `parallel_bulk_load` (one staging database per partition filled in parallel, then `ATTACH` and `INSERT ... SELECT` in partition order inside one transaction: the target gets the same rows and rowids as a serial loop). Partitions are fixed by the caller, partition results are combined in partition order, and ties never depend on timing, so **every function returns exactly the serial result for any pool size**. `task/algorithms.hpp` holds the pieces: `parallel_sort` (stable, equals `std::stable_sort`), `parallel_merge` (co-ranking, stable), `merge_runs`, `parallel_topk_by`, `parallel_reduce`, `parallel_inclusive_scan` / `parallel_exclusive_scan` (fixed blocks, so floating-point results also do not depend on the thread count; they equal the serial run of the same blocked algorithm, not a plain left fold). `pool == nullptr` runs serially on the caller.

Checked (`test_task_algorithms`, `test_db_parallel`): sort, merge, k-way merge, top-k, reduce and scans, and query, scan, filter, reduce (integer and double), top-k, bm25 top-k, sort and bulk load, each compared with the serial result for 1, 2, 3, 8 and 16 threads on fixed-seed data (30000 rows with gaps in the rowids and many equal scores), and for two different partition counts. Not provided: a parallel index build (SQLite has one writer per database; `open_options::sorter_threads` sets `PRAGMA threads` for SQLite's own sorter).

## Shard sets: the default storage (`db/shards.hpp`)

A database is a directory:

```
lean.db/manifest.json      small text: schema version, per-shard {file, key range, rows, bytes, sha256, blake3}, caller statistics
lean.db/shard-000000.fdb   a SQLite file of at most 10 MiB, rows sorted by key
lean.db/shard-000001.fdb   ...
```

`db::open(path)` gives a `shard_set` (a plain SQLite file gives a one-shard set; `db::open_single` / `connection::open` is the explicit single-file path for small or temporary databases). `db::create` / `shard_writer` writes one.

**Layout is the search structure.** Shards hold consecutive key ranges, so `shard_set::shards_for_key / shards_for_range / shards_for_prefix` binary-search the manifest and a lookup opens only the shard(s) whose `[min, max]` can hold the answer. Everything else (`shard_map`, `shard_query`, `shard_range_query`, `shard_topk`, `shard_bm25_topk`, `ann::shard_ann`) runs over a list of shards on `forge::task` and combines results in shard order or by a total order, so results never depend on the thread count.

**Writing** (`shard_writer`): rows (`std::vector<db::value>`, the key column a string or bytes, stored as a BLOB so keys compare as `memcmp`) arrive presorted (streamed in waves of `wave_bytes`, 256 MiB by default) or in any order (buffered, then stably sorted with `task::parallel_sort`). The first shard of the first wave is built to calibrate file bytes per raw byte; the rest are cut to 90 % of the budget by that ratio and built in parallel; each file is finalised (`page_size` 4096, `journal_mode=DELETE`, `VACUUM`, `user_version`) and **checked on its final size**: over the budget, it is split in halves and rebuilt (a single row larger than the budget is an error). `shard_schema::finalize` runs inside each shard before `VACUUM` (fill the FTS5 table, save an HNSW graph, ...). Builds are byte-reproducible: for the same rows, schema, hook and options the shard files and the manifest are identical bytes for any thread count (`test_db_shards`, `reproducible_budgeted_builds`: SHA-256 of every shard and the manifest text compared across 1, 2, 3, 8 and 16 threads, several waves). The manifest is written with `os::fs::write_file_atomic`; shard files are renamed into place.

**Opening** (`shard_set`): the manifest is parsed and its key ranges checked; each shard is verified (size, BLAKE3 and SHA-256) the first time it is opened (`verify_mode`; `verify_all(pool)` checks all now); a mismatch throws `errc::integrity_error` naming the shard. Shards open read-only with `immutable=1` (no locking, no journal files), `mmap_size` = the shard size and a 1 MiB page cache, so a query on a warm shard is served from the page cache through the mapping. Connections are pooled per shard; `max_open_bytes` (1 GiB by default) bounds the shards that keep open connections, least recently used closed first. Prefetch: `shard_map` calls `posix_fadvise(WILLNEED)` on the shards of a query before running it (POSIX only; no-op elsewhere). `shard_schedule::blocked` hands each worker a contiguous run of shards instead of the next one.

**Hashes.** The manifest keeps SHA-256 per shard (`forge::hash::sha256`, new: what git and external pins use) and BLAKE3 (the library's own hash, the one `forge::cache` keys on; two to three times cheaper per byte to check, and the one the cache layer uses). Both are checked on open by default. The BLAKE3 of the manifest text is `shard_set::id()`.

**Content-addressed storage and cache.** `shard_options::blob_dir` additionally keeps each shard as `<blob_dir>/<sha256>.fdb` (hard link, else copy), and `shard_open_options::blob_dir` falls back to it when `<dir>/<file>` is missing: a shard built or fetched once is reused by every database and version that lists the same hash (`test_db_shards`, `verification_and_content_addressed_store`). Query results can be kept in a `forge::cache::store` (its `max_bytes` gives the LRU budget, its atomic writes and locking apply) with `cached_result(store, set, {query parts}, compute)`, keyed by `set.id()` plus the query, so a rebuilt database never serves a stale answer; `encode_hits/decode_hits` serialise `shard_hit` lists. The cache store cannot hold the shards themselves: its entries carry a 32-byte hash header, so a cached file is not a SQLite file.

**Exact BM25 over shards.** FTS5's `bm25()` takes its statistics from the table it runs on, so a shard alone scores differently from the whole. `shard_bm25_topk(set, pool, "fts", terms, k)` scores every shard with the global statistics through a registered FTS5 auxiliary function `forge_bm25(fts, avgdl, idf_blob, weights...)`, which is FTS5's own formula (k1 = 1.2, b = 0.75, same arithmetic, same order) with the average document length and per-term IDF passed in: `N` (rows) and total tokens come from the manifest (`shard_info::fts_rows/fts_tokens`, written when `shard_schema::fts_table` is set), each term's document count is the sum of per-shard `count(*) ... MATCH` queries. The ranking and the scores are bit-identical to `bm25()` on one table holding all the rows (`test_db_shards`, `bm25_over_shards_equals_unsharded`: 6 queries x OR/AND x k in {1, 10, 100} x 5 thread counts x pruning on/off against the unsharded table, comparing ids and exact doubles; also weights, and a one-shard set). Limits: each term must be one plain word (one phrase per term; `fts5::words` splits text that way), and the `fts` table's rowid must carry the global id. Not done: a precomputed per-term document-frequency table. Per-term counts come from the shards at query time (a stored table keyed by FTS5 tokens would need the tokenizer's normalisation and does not handle multi-token phrases), and per-shard bloom filters are not stored either.

**Early stop.** A term adds less than `(k1 + 1) * idf` to a row, so a shard whose matching terms sum below the current k-th best score cannot contribute. Shards are scored in descending order of that bound and skipped when it is strictly below the k-th score (the shard's per-term hit counts, from the same count queries, decide which terms count). The result never changes (`prune` on and off are compared in the test); `bm25_report` says how many shards were skipped.

**ANN.** One HNSW graph per shard, saved inside the shard; see [ann.md](ann.md#ann-over-shards-annshard_annhpp) for why and for the measured recall.

Not done: NUMA binding and topology-aware worker placement (`os/numa.hpp`, `os/cpu_topology.hpp` are not used; `shard_schedule::blocked` is the locality hook but was measured only on a single-node machine), `os/async_file.hpp` / `io/aio.hpp` prefetch (`posix_fadvise` is used instead), a FORGE VFS over `core/mapped.hpp` (SQLite's own `mmap_size` does the mapping), and FORGE's arena/tlsf allocators (no hot path here allocates enough to matter in the measurements below, and none was profiled).

## Measured

BENCH_PLACEHOLDER
