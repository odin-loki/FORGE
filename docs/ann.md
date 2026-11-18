# forge::ann: nearest-neighbour search templates

Headers under `support/include/forge/ann/` (`#include <forge/ann/ann.hpp>` for all). Header-only; link `forge::ann` (CMake target `forge_ann`, which pulls in `forge::db` and the vendored SQLite). Tests: `test_ann` (and `test_ann_slow`, label `slow`), `test_db_shards` (the per-shard index). Example: `docs/examples/db_search.cpp` (ctest `doc_example_db_search`).

## Contract

```cpp
namespace forge::ann {
struct hit { std::uint64_t id; float dist; };             // ordered by (dist, id); NaN sorts last (hit_before)
template<class M, class V> concept metric = ...;           // m.distance(a, b) -> float, smaller = closer
// optional on M: void distances(const V& q, std::span<const V> xs, std::span<float> out) const;   (batch; flat_index uses it when present)
template<class V> struct codec;                            // static void encode(const V&, std::vector<std::byte>&); static V decode(std::span<const std::byte>)

template<class V, class M> requires metric<M, V> class flat_index {          // exact
  explicit flat_index(M m = {}); void add(std::uint64_t id, V v); std::size_t size() const;
  std::vector<hit> search(const V& q, std::size_t k, forge::task::pool* pool = nullptr) const; };

struct hnsw_params { std::size_t M = 16, ef_construction = 200, ef_search = 160; std::uint64_t seed = 0x5eed; };   // ef_search was 64 until 18 Nov (recall 0.858 on the 20k random 64-d case)
template<class V, class M> requires metric<M, V> class hnsw {
  explicit hnsw(hnsw_params p = {}, M m = {}); void add(std::uint64_t id, V v);
  void build(forge::task::pool* pool = nullptr);                            // parallel, deterministic for a fixed seed
  std::vector<hit> search(const V& q, std::size_t k, std::size_t ef = 0) const;   // ef = 0: max(ef_search, k)
  void save(forge::db::connection&, std::string_view table) const;
  static hnsw load(forge::db::connection&, std::string_view table, M m = {}); };

template<class V, class Exact> std::vector<hit> rerank(std::span<const hit> cand, const V& q,
    std::function<const V&(std::uint64_t)> get, const Exact& exact, std::size_t k);
}
```

Beyond the contract: `hnsw::size()` (added), `built()` (searchable), `params()`, `graph_hash()`, `edges()`; `flat_index::vector_at/id_at`; `reciprocal_rank_fusion(lists, k, c = 60, weights)` in `ann/fusion.hpp`; `ann::shard_ann` and `ann::ann_finalize` in `ann/shard_ann.hpp` (below).

`pool == nullptr` means serial on the calling thread, never the global pool. A search returns the `k` nearest ordered by `(dist, id)`. `flat_index::search` equals the brute-force sort of all `(dist, id)` pairs for any pool size (`test_ann`, `flat_index_equals_brute_force`, on data with many exact distance ties, for 1, 2, 3, 8 and 16 threads, through both the batch and the plain metric path).

## Built-in metrics

| Metric | Vectors | Distance |
|---|---|---|
| `l2`, `l2_squared` | `std::vector<float>`, `std::span<const float>` (any contiguous float range) | Euclidean, resp. squared |
| `cosine` | same | `1 - a.b / (\|a\|\|b\|)`; 1 when a vector is zero |
| `inner_product` | same | `-(a.b)` |
| `hamming` | `std::vector<std::uint64_t>` bit vectors | popcount of the xor (missing words count as zero) |
| `weighted_jaccard` | `sparse_vec` = `std::vector<std::pair<std::uint64_t,float>>` sorted by feature | `1 - sum(min w) / sum(max w)` |
| `minhash_jaccard` | `minhash_sig` = `std::vector<std::uint32_t>` | `1 - fraction of equal slots`; `minhash_sketch(features, k, seed)` and `wminhash_sketch(sparse_vec, k, seed)` (Ioffe's consistent weighted sampling) build the signatures |

The float kernels are scalar C++ with eight independent accumulators in a fixed order (the compiler vectorises them); they do not use `forge::simd`, and there is no GPU path: nothing in this layer uses a device. Distances are bit-identical on every thread and every run. Codecs (`codec<V>`) exist for `std::vector<float|uint64_t|uint32_t>` (so `minhash_sig` too) and `sparse_vec`; they write little-endian by copying memory.

## HNSW

`hnsw` follows Malkov and Yashunin: levels from an exponential distribution, greedy descent through the upper levels, best-first search with `ef` candidates on level 0, the neighbour-selection heuristic (keep a candidate only if it is closer to the new node than to every neighbour already kept; pruned candidates refill the list up to its cap). Lists hold at most `2M` neighbours on level 0 and `M` above; a new node selects up to that cap on each level it joins.

**Parallel build, deterministic.** `add()` only queues. `build(pool)` inserts the queued nodes in batches of `max(1, built / 16)` (at most 128). Phase 1 (parallel, read-only) lets every node of a batch search the graph as it was before the batch, plus brute-force the earlier nodes of its own batch, and choose its neighbours. Phase 2 (serial, insertion order) commits nodes and the entry point. Phase 3 (parallel over target nodes) adds back links in insertion order and re-selects lists that overflowed. Levels come from `splitmix64(seed, insertion index)`; every comparison orders by `(distance, node index)`. So a fixed seed and insertion order give the identical graph for any pool size, including `nullptr` (`test_ann`, `hnsw_deterministic_for_any_thread_count`: the graph hash and 50 search results equal for 0, 1, 2, 3, 8 and 16 threads; an incremental second `build` is deterministic too). Not provided: removal, update.

**Recall (measured, 8 threads, `test_ann`).** Uniform Gaussian vectors in 64 dimensions are close to the hardest case for a graph index.

| Data | n | M / efConstruction | ef | recall@10 | CPU time per query |
|---|---|---|---|---|---|
| random 64-d | 20000 | 16 / 200 | 64 (the old default) | 0.858 | 0.48 to 0.64 ms |
| | | | 96 | 0.926 | |
| | | | 128 | 0.9555 | |
| | | | **160 (the default)** | **0.9695** | 0.85 to 1.3 ms |
| | | | 192 | 0.9785 | |
| | | | 256 | 0.989 | |
| | | 16 / 400 | 64 / 128 / 160 | 0.8575 / 0.957 / 0.970 | (a larger efConstruction does not help) |
| | | 24 / 200 | 64 / 128 | 0.9305 / 0.9835 | about 2x the time of M = 16 |
| | | 32 / 200 | 64 / 128 | 0.966 / 0.996 | about 2x to 2.5x; build 5.6 s against 2.8 s, 2x the edges |
| random 64-d | 5000 | 16 / 200 | 64 / 128 / 256 | 0.9605 / 0.995 / 0.9995 | |
| clustered 256-bit Hamming codes (100 centres, 30 flipped bits) | 20000 | 16 / 200 | 64 | 1.000 | |

**The default is now `ef_search = 160`** (M = 16, efConstruction = 200 unchanged): recall@10 = 0.9695 on the 20000 random 64-d vectors, 0.95 gate met with a margin. Only default values changed; the types, names and signatures are the same, so code that spells its own parameters is not affected, and saved graphs keep the `ef_search` they were built with. The cost: a default search visits about 2 to 2.5 times as many nodes as at ef = 64 (single-thread CPU time per query on this machine, best of 40 runs on a loaded host, 20000 vectors: 0.5 to 0.65 ms at ef = 64, 0.85 to 1.3 ms at ef = 160; the spread between runs of the same code was as large as the difference between the arena and non-arena builds, so read these as roughly 2x). Callers that want the old speed pass `ef = 64` to `search()` or set `ef_search` in the params. `M = 32` reaches 0.966 already at ef = 64 but doubles the memory and the build time, so the default keeps M = 16. The gate in `test_ann` is recall@10 >= 0.95 at the default ef (`ef = 0`) on 20000 random 64-d vectors, in the default test set, and >= 0.95 on the Hamming set; it also records ef = 64 and ef = 256. The 20000-vector build takes 3 s on 8 threads of a loaded machine. FORGE's arena allocator (`mem::arena`) was tried for the candidate heaps of `search_layer_`: a search makes 23 to 26 heap allocations (measured with a counting `operator new`), about 1 us of 500 to 1300 us, and the A/B timings showed no consistent gain, so the allocator is not used (see [db.md](db.md)).

**Persistence.** `save(conn, table)` replaces `table` in one transaction with `(kind TEXT, k INTEGER, data BLOB, PRIMARY KEY(kind, k)) WITHOUT ROWID`: row `('params', 0)` (magic, M, efConstruction, ef_search, seed), `('graph', 0)` (node count, entry point, top level) and one `('node', i)` per node (id, level, the `codec<V>` bytes, the link lists). `load` validates the numbering, the link targets and levels and the entry point and throws `forge_exception(errc::format_error)` on a damaged table. Saving with queued nodes throws. A loaded graph has the same `graph_hash()` and gives the same results.

## Re-ranking and fusion

`rerank(candidates, q, get, exact, k)` recomputes the distance with an exact metric and keeps the `k` closest by `(dist, id)`. `reciprocal_rank_fusion({bm25_ids, ann_ids}, k)` returns `fused{id, score}` with `score = sum weight / (c + rank)` (rank from 1, a duplicate inside a list counts once), ordered by `(score desc, id asc)`.

## ANN over shards (`ann/shard_ann.hpp`)

For a `db::shard_set` ([db.md](db.md)): `sc.finalize = ann::ann_finalize<V, M>("SELECT id, vec FROM t ORDER BY k", params, metric)` builds an `hnsw` over each shard's rows while the shard is written and saves it in the shard's `ann` table; `ann::shard_ann<V, M> ix(set)` loads a shard's graph on first use, searches every selected shard in parallel and merges the per-shard top-k by `(dist, id)`.

Choice: one graph per shard, not one global graph. Each shard stays a self-contained, content-addressed file, rebuilt and re-hashed alone when its rows change; shards build in parallel; a query can restrict itself to the shards a key range selects. The cost is that a query searches every selected shard's graph. Measured: 9000 random 16-d vectors in 25 shards, recall@10 against `flat_index` = 1.000 at ef 64 (`test_db_shards`); at benchmark scale see [db.md](db.md). Results do not depend on the thread count (checked for 1, 2, 3, 8, 16).

## GPU batch search (`ann/gpu_flat.hpp`, `graph/ann_gpu.hpp`)

```cpp
ann::gpu_flat_l2 ix(graph::ann_matmul(reg.get()), dim);      // reg = graph::device_registry("cuda"); nullptr = the CPU lowering
ix.add(id, vec); ...
auto hits = ix.search_batch(queries, k, &pool);               // hits[q] == flat_index<std::vector<float>, ann::l2>::search(queries[q], k)
```

An exact batched search for `l2` over `std::vector<float>` that lets a device do the arithmetic. The device (through the graph registry's own `matmul_t` lowering, the one `graph::bench_matmul` times) computes `Q . X^T` for a chunk of vectors; the host turns that into approximate squared distances (`|q|^2 + |x|^2 - 2 q.x`), keeps every vector whose approximate distance is within `2 * err` of the k-th smallest, and **re-ranks those on the host with the plain `l2` metric**, so the result is `flat_index::search`'s exactly (distances bit-identical, ties by id). `err = 2 * (dim + 6) * 2^-24 * (|q|^2 + max |x|^2)` bounds the rounding error of any f32 summation order, which needs an IEEE f32 matmul (not TF32 or half precision; `error_scale` widens the bound if a backend needs it). If the matmul callable returns false (no device, out of memory) the batch is answered by the CPU path and the result is the same.

Tests: `test_gpu_ann_logic` (everywhere: a CPU matmul, a noisy one within the bound, a failing one; 3000 vectors, d = 37 and 64, vectors on a coarse grid so many distances tie, chunk sizes 500 and 65536, k = 1, 10, 64, with and without a pool) and `test_gpu_ann_device` (label `gpu`; exits 77 without CUDA). Measured on the RTX 3090 of this machine (CUDA 12 through WSL2, 300 queries per batch, query blocks of 128, 32768-vector chunks, the CPU side `flat_index` on an 8-thread pool of a host with load 50):

| n | d | k | device path | CPU `flat_index` (8 threads) | candidates re-ranked per query |
|---|---|---|---|---|---|
| 60000 | 37 | 1 / 10 / 100 | 0.165 / 0.130 / 0.122 s | 0.113 / 0.110 / 0.233 s | 1.0 / 10.0 / 100 |
| 60000 | 64 | 1 / 10 / 100 | 0.143 / 0.133 / 0.120 s | 0.226 / 0.199 / 0.337 s | 1.0 / 10.0 / 100.1 |
| 60000 | 128 | 1 / 10 / 100 | 0.139 / 0.129 / 0.126 s | 0.524 / 0.515 / 0.600 s | 1.0 / 10.0 / 100.1 |
| 20000 | 128 | 10 | 0.053 s | 0.079 s | 10.3 |

All results equal `flat_index`'s. The device time barely moves with k or d: the product data crosses host memory on every call (the registry lowering takes host pointers, uploads `X` each time and reads `Y` back), so the path is transfer-bound. It wins about 1.5x at d = 64, about 4x at d = 128 and loses at d = 37 or on 20000 vectors of d <= 64. It does not keep the vectors resident on the device between batches (that needs the executor's residency protocol, not used here); one run each on a loaded machine, so treat the ratios as indicative.
