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

struct hnsw_params { std::size_t M = 16, ef_construction = 200, ef_search = 64; std::uint64_t seed = 0x5eed; };
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

| Data | n | ef | recall@10 |
|---|---|---|---|
| random 64-d, M = 16, efConstruction = 200 | 20000 | 64 (the default) | 0.858 |
| | | 128 | 0.9555 |
| | | 256 | 0.989 |
| random 64-d | 5000 | 64 / 128 / 256 | 0.9605 / 0.995 / 0.9995 |
| clustered 256-bit Hamming codes (100 centres, 30 flipped bits) | 20000 | 64 | 1.000 |

The gate in `test_ann` is recall@10 >= 0.95 at ef = 256 on random 64-d vectors (a 5000-vector run in the default set, 20000 in `test_ann_slow`) and >= 0.95 at the default ef on the Hamming set. **The brief asked for 0.95 at the default ef (64) on 20000 random 64-d vectors; the default parameters do not reach it (0.858).** Raise `ef_search` (128 gives 0.9555) or build with `M = 32` (0.966 at ef 64 in a one-off run, not part of the tests). The 20000-vector build took 3 s on 8 threads of a loaded machine.

**Persistence.** `save(conn, table)` replaces `table` in one transaction with `(kind TEXT, k INTEGER, data BLOB, PRIMARY KEY(kind, k)) WITHOUT ROWID`: row `('params', 0)` (magic, M, efConstruction, ef_search, seed), `('graph', 0)` (node count, entry point, top level) and one `('node', i)` per node (id, level, the `codec<V>` bytes, the link lists). `load` validates the numbering, the link targets and levels and the entry point and throws `forge_exception(errc::format_error)` on a damaged table. Saving with queued nodes throws. A loaded graph has the same `graph_hash()` and gives the same results.

## Re-ranking and fusion

`rerank(candidates, q, get, exact, k)` recomputes the distance with an exact metric and keeps the `k` closest by `(dist, id)`. `reciprocal_rank_fusion({bm25_ids, ann_ids}, k)` returns `fused{id, score}` with `score = sum weight / (c + rank)` (rank from 1, a duplicate inside a list counts once), ordered by `(score desc, id asc)`.

## ANN over shards (`ann/shard_ann.hpp`)

For a `db::shard_set` ([db.md](db.md)): `sc.finalize = ann::ann_finalize<V, M>("SELECT id, vec FROM t ORDER BY k", params, metric)` builds an `hnsw` over each shard's rows while the shard is written and saves it in the shard's `ann` table; `ann::shard_ann<V, M> ix(set)` loads a shard's graph on first use, searches every selected shard in parallel and merges the per-shard top-k by `(dist, id)`.

Choice: one graph per shard, not one global graph. Each shard stays a self-contained, content-addressed file, rebuilt and re-hashed alone when its rows change; shards build in parallel; a query can restrict itself to the shards a key range selects. The cost is that a query searches every selected shard's graph. Measured: 9000 random 16-d vectors in 25 shards, recall@10 against `flat_index` = 1.000 at ef 64 (`test_db_shards`); at benchmark scale see [db.md](db.md). Results do not depend on the thread count (checked for 1, 2, 3, 8, 16).
