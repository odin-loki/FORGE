# GPU (RTX 3090) against CPU with one shard per pinned core: forge::db search, forge::ann search, FastSMT hint lookup

Measured 19 Nov 2026 on the shared build machine. Every number below comes from the raw outputs in
[`bench_gpu_vs_cpu/`](bench_gpu_vs_cpu/). Each configuration ran 5 times unless a row says otherwise. Cells give the median, with the
min..max over the reps in brackets in the raw files. The 1-minute load average and MemAvailable were sampled before and after every
configuration and are in the raw rows.

## Summary

| workload | CPU, one shard per pinned core | best GPU path | GPU / CPU | where the GPU wins |
|---|---|---|---|---|
| BM25 top-10, point lookups | 13.2 q/s (frequent words), 58.8 q/s (rare term), 2.6 M lookups/s (warm) | none: FORGE has no GPU BM25 or lookup | n/a | n/a |
| exact kNN, top-10, 3.6M x 32-d | flat: 41.5 / 62.5 / 64.1 q/s at batch 1 / 64 / 1024 | `gpu_knn_l2` (new): 486 / 4279 / 6298 q/s | 11.7x / 68x / 98x | every batch size |
| same, the shipped GPU path | (as above) | `gpu_flat_l2` as shipped: 2.3 / 48.2 / 65.1 q/s | 0.06x / 0.77x / 1.02x | nowhere: rows cross PCIe on every call |
| approximate kNN at recall@10 0.99 | HNSW per shard ef 128: 108 / 210 / 250 q/s | `gpu_knn_l2`, exact (recall 1.0) | 4.5x / 20x / 25x | every batch size |
| FastSMT hint kNN (prism::sim), 7007 formulas | 18,982 q/s (HNSW), 2,889 q/s (exact), 64 pinned workers | none: the symbolic metric has no GPU kernel | n/a | n/a |
| FastSMT solving (z3) | 0.043 to 0.067 s median per easy formula on one pinned core | CPU work, not a GPU candidate | n/a | n/a |

Training an IVF index and clustering the shards (Odin's extra question) bought no significant speedup at matched recall on these
vectors, on any path. The bench vectors are i.i.d. N(0, 1), so they have no cluster structure for k-means to find. On a clustered copy
of the same rows (a Gaussian mixture), IVF on the GPU at batch 1024 was 3.8x to 5.9x faster than brute force on the GPU, and 10.9x to
17x faster than the CPU HNSW (recall 0.976 to 0.998). On the CPU with one shard per core it was not faster at matched recall. See the IVF section.

## Setup

- Machine: 64 logical CPUs (32 cores x 2 SMT threads; CPUs 2k and 2k+1 are siblings), 47 GB WSL2, RTX 3090 (24 GB). Driver
  610.88 (CUDA UMD 13.3), g++ 13.3, Release build.
- Shared machine: other jobs held the load at 60 to 85 throughout. The benches added up to 60 busy threads, so the recorded load was
  75 to 125. CPU numbers are therefore lower than on an idle machine. The GPU paths need little host CPU and are less affected.
- Data: perf/bench_db's generator, 3.6M rows (random-word FTS5 bodies, 32-d float vectors drawn i.i.d. from N(0, 1)). The set was built
  at a 44 MiB budget: 60 shards, 2232 MiB, one HNSW graph per shard (M 16, efConstruction 100), on tmpfs (`/dev/shm`). With 60 shards
  and 64 CPUs every shard has its own core.
- "One shard per pinned core": worker `w` is a thread pinned with `pthread_setaffinity_np` to logical CPU `w` (checked with
  `sched_getcpu`). It owns shard `w` for the whole run: its connection, its prepared statements, its HNSW graph and its flat index. A
  query fans out to all 60 workers and the caller merges. CPUs 0 to 59 cover 30 physical cores, so worker pairs share a core.
- "1 thread": the same code with one worker, pinned to CPU 0, owning every shard.
- "fresh" means a new `shard_set` handle (new connections, statements and graphs). On tmpfs nothing is read from a disk, so this is not
  a cold page cache. "warm" is the second pass on the same handles.
- Queries: 1024 random N(0, 1) vectors (seed 4242) for ANN. BM25 queries use perf/bench_db's vocabulary: 2 or 3 frequent words joined
  by OR, and single clustered rare terms. Point lookups use 20000 random keys.
- Latency at batch B is the time of one call. Every query in a batch completes together, so the p50 at batch 1024 is the time of the
  whole batch.

## Is the GPU path real?

- CUDA in WSL works. `nvidia-smi` sees the RTX 3090, and FORGE's runtime dlopens the real `libcuda.so.1`. The tests and benches
  report the device "NVIDIA GeForce RTX 3090" and run the image "ptx sm_80". The CPU mock (`tests/mock_cuda.c`) was not on the library
  path.
- Kernels ran on the device, not on a CPU fallback:
  - The runtime's dispatch counters counted the launches. For example, `gpu_flat_l2` launched `matmul_tiled` 5555 times in one row, and
    `forge_knn_l2_part` launched 6 times for 1024 queries. `report.fell_back` was false on every GPU row.
  - `nvidia-smi`, sampled every 0.5 s, showed up to 100 % utilisation. Peak memory was 1985 MiB, against an idle 1143 MiB (Xwayland).
    That stays inside the 16 GB budget.
- `tests/test_gpu_ann` passes on the device: gpu_flat_device, gpu_knn_logic, gpu_knn_device, ivf_logic, ivf_device and
  gpu_flat_logic. Every device result equals `flat_index` (ids and distances), or equals the CPU IVF scan on 1000 of 1000 queries.
- Not a GPU path:
  - `shard_bm25_topk` and point lookups have no GPU path; they are SQLite/FTS5 on the CPU.
  - prism::sim's hint lookup has no GPU path either: its metric is a weighted Jaccard plus Hamming or sub-term Jaccard over sparse,
    variable-length vectors, and FORGE's GPU searches are dense l2 only.
  - z3 solving is CPU work.

  None of these is labelled GPU anywhere below.

## New code (branch gpu-bench)

The shipped `ann::gpu_flat_l2` does the distance matrix as a device matmul. The database rows (439 MiB here) are uploaded on every call
and the whole distance matrix comes back (up to 1758 MiB per 128-query block), so it cannot beat the CPU. That left the obvious kernel
missing: brute-force top-k with the rows resident on the device. I added it, plus the IVF Odin asked for.

- `ops/gen/gpu_gen.py` has three new kernels. All three use shared memory and barriers only, with no warp shuffles. They compile for
  PTX sm_70/sm_80 and for HIP gfx90a, gfx942 and gfx1100.
  - `forge_knn_l2_part` scans 128-row tiles for 8 queries per block and keeps a private top 16 per thread.
  - `forge_knn_l2_merge` merges a query's split lists.
  - `forge_ivf_l2_part` scans given row ranges.
- `gpu/cuda.hpp` gets `runtime::knn_l2_16` and `runtime::ivf_l2_16`. The rows are uploaded once and kept resident, keyed by pointer;
  `knn_forget` releases them.
- `ann/gpu_knn.hpp` adds `ann::gpu_knn_l2`, which is exact against `flat_index`. The host re-scores the device's 16 candidates with the
  l2 metric, and a per-query f32 error-bound check proves the true top k is among them. Where the check fails, that query runs on the
  CPU. Here it failed on 0 of 1024 queries.
- `ann/ivf.hpp` adds `ann::kmeans` (Lloyd; the same centroids for any thread count) and `ann::ivf_flat` (inverted lists, an exact scan
  of the probed lists). It searches three ways:
  - on the CPU;
  - through `gpu_flat_l2`'s matmul callable over the probed lists;
  - through the device scan kernel.
- `graph/ann_gpu.hpp` adds the glue: `cuda_knn`, `cuda_ivf` and `open_knn_device`.
- Tests: `tests/test_gpu_ann.cpp` gains `gpu_knn_logic`, `gpu_knn_device`, `ivf_logic` and `ivf_device`, each registered in CTest.
- Benches: `perf/bench_gpu_cpu.cpp`, `perf/bench_ivf.cpp`, and `perf/run_bench_gpu_cpu.sh` to drive them.

## Database search (no GPU path)

60 shards, one per pinned core, against one thread. "library" is `db::shard_bm25_topk` as shipped, on an unpinned `task::pool(60)`.
The pinned harness reproduces its results exactly: ids and scores were compared on every query. The harness is faster mostly because
each worker keeps its prepared statements and connection across queries, while the library prepares statements and acquires leases per
query. That explanation was not measured on its own.

| workload | config | handle | queries/s | p50 ms | p99 ms |
|---|---|---|---|---|---|
| BM25 frequent words (2-3 terms, OR), 20 queries | 60 pinned (1 shard/core) | fresh | 12.6 | 67.5 | 186.8 |
| | | warm | 13.2 | 65.2 | 168.1 |
| | 1 thread (5 queries) | fresh | 0.2 | 2704.7 | 7324.7 |
| | | warm | 0.2 | 2356.6 | 7955.3 |
| | library, task::pool(60) | fresh | 2.5 | 389.2 | 541.3 |
| | | warm | 2.5 | 379.5 | 600.3 |
| BM25 clustered rare term, 50 queries | 60 pinned | fresh | 56.3 | 17.8 | 26.6 |
| | | warm | 58.8 | 17.1 | 29.7 |
| | 1 thread | fresh | 25.7 | 37.0 | 75.0 |
| | | warm | 30.5 | 30.3 | 70.9 |
| | library, task::pool(60) | fresh | 41.2 | 23.2 | 32.5 |
| | | warm | 42.3 | 23.1 | 30.7 |
| point lookup by key, 20000 keys (latency = one lookup) | 60 pinned | fresh | 302,651 | 0.006 | 0.046 |
| | | warm | 2,595,540 | 0.005 | 0.017 |
| | 1 thread | fresh | 16,675 | 0.028 | 0.343 |
| | | warm | 31,420 | 0.011 | 0.230 |

Frequent-word BM25 is about 36x faster at p50 on 60 pinned cores than on one thread (2357 ms against 65 ms, warm). A rare term touches
only 2 or 3 shards, so it gains 1.8x.

## ANN search: 3.6M vectors, 32-d, top-10

Recall@10 is measured against the exact answer, which pinned flat and `gpu_knn_l2` both produced: equal ids and distances on all 1024
queries.

| method | batch 1: q/s, p50 ms | batch 64: q/s, p50 ms | batch 1024: q/s, p50 ms | recall@10 |
|---|---|---|---|---|
| CPU flat, 60 pinned workers | 41.5, 23.7 | 62.5, 1027 | 64.1, 15978 | 1.0000 |
| CPU flat, 1 thread | 2.5, 389 | 1.9, 33742 | not run | 1.0000 |
| CPU HNSW per shard ef 64, 60 pinned | 125.9, 7.98 | 336.6, 189 | 381.5, 2681 | 0.9735-0.9747 |
| CPU HNSW per shard ef 128, 60 pinned | 108.3, 8.87 | 210.4, 304 | 250.1, 4092 | 0.9910-0.9938 |
| CPU HNSW per shard ef 128, 1 thread | 2.9, 325 | not run | not run | 0.9910 |
| GPU `gpu_flat_l2` as shipped, all shards in one index | 2.3, 430 | 48.2, 1337 | 65.1, 15720 | 1.0000 |
| GPU `gpu_knn_l2` (new), all shards, resident | 485.6, 2.05 | 4278.7, 15.0 | 6298.3, 162 | 1.0000 |
| GPU `gpu_knn_l2` (new), one index per shard, shards in turn | 40.0, 25.0 | 323.2, 198 | 2281.3, 448 | 1.0000 |

HNSW recall sweep (per-shard graphs, 256 queries):

| ef | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| recall@10 | 0.8191 | 0.9219 | 0.9730 | 0.9918 | 0.9988 | 1.0000 |

Host-to-device transfer, measured on its own:

- `gpu_knn_l2` uploads all 3.6M rows (439 MiB) once per index: 215.5 ms median (194.8..251.6), 2.14 GB/s, pageable memory under WSL.
  The device memory in use after the upload was 1.68 GiB of 24 GiB.
- The device call alone (query upload, two kernels, 16 results back, no host re-ranking) takes 2.03 / 13.2 / 160.4 ms at batch
  1 / 64 / 1024.
- `gpu_flat_l2` as shipped moves 439 MiB of rows up per query block and up to 1758 MiB of distances back. That transfer is the time in
  its row.

GPU over all shards at once fits easily (1.7 GiB). It is 12x / 13x / 2.8x faster than the same kernel searching the 60 shards one after
another.

## IVF and cluster-aligned shards (Odin's extra arm)

`ann::ivf_flat`: k-means on a 5 % sample (every 20th row, 180,000 rows), 20 Lloyd iterations, nlist = sqrt(N) = 1897 and 4 sqrt(N) =
7588, then a sweep of nprobe. Each method was timed at the first nprobe whose recall@10 reached 0.97 (and 0.99). Training is a one-off
cost and is reported apart.

The configurations:

- CPU 1 thread is one global index.
- CPU one shard per pinned core is 60 `ivf_flat`s, one per shard, sharing the centroids. Workers compute the probes for their slice of
  the batch, then scan their shard for every query.
- GPU scan kernel is one global index, rows resident, probes computed on the host pool.
- GPU matmul is `gpu_flat_l2`'s matmul over each list touched by the batch.

### On the bench vectors (i.i.d. N(0, 1), no cluster structure)

| | nlist 1897 | nlist 7588 |
|---|---|---|
| k-means training (one-off) | 9.30 s (8.78..10.06) | 42.43 s (36.36..44.69) |
| assigning 3.6M rows (one-off) | 10.31 s | 27.06 s |
| recall 0.964 / 0.995 / 1.000 at nprobe | 256 / 512 / 1024 (13.8 / 27.5 / 54.9 % of rows scanned) | 0.952 / 0.990 / 0.999 at 512 / 1024 / 2048 (8.2 / 16.3 / 32.2 %) |

| method (recall@10) | batch 1: q/s, p50 ms | batch 64: q/s | batch 1024: q/s |
|---|---|---|---|
| CPU ivf nlist 1897 nprobe 512, 1 thread (0.9945) | 14.6 to 19.6, 50.7 to 66.5 | | |
| CPU ivf nlist 1897 nprobe 512, 60 pinned (0.995) | 62.6, 15.9 | 182.8 | 210.9 |
| CPU ivf nlist 7588 nprobe 1024, 60 pinned (0.990) | 66.6, 15.1 | 217.9 | 227.3 |
| CPU HNSW per shard ef 128, 60 pinned (0.991-0.994), for comparison | 108.3, 8.87 | 210.4 | 250.1 |
| GPU ivf nlist 1897 nprobe 512, scan kernel (0.995) | 1208.9, 0.83 | 3481.6 | 4035.2 |
| GPU ivf nlist 7588 nprobe 1024, scan kernel (0.990) | 708.2, 1.41 | 4123.8 | 5019.9 |
| GPU ivf, `gpu_flat_l2`'s matmul over probed lists (0.990-0.995) | not run | 18.6 to 28.5 | 33.3 to 40.3 |
| GPU `gpu_knn_l2`, exact (1.0), for comparison | 485.6, 2.05 | 4278.7 | 6298.3 |

Each IVF row ran twice, because the 0.97 and 0.99 targets landed on the same nprobe. Both runs are in the raw file, and they agree
within the spread.

- On the CPU, IVF at recall 0.99 is 0.6x HNSW at batch 1, and 0.84x to 1.04x at batch 64 and 1024.
- On the GPU it beats exact brute force only at batch 1: 2.5x for nlist 1897 and 1.5x for nlist 7588. At batch 64 and 1024 it is 0.64x
  to 0.96x of brute force, because reaching 0.99 means scanning 16 to 27 % of the rows anyway.
- Reusing `gpu_flat_l2`'s matmul over the probed lists is 100x to 150x slower than the scan kernel: one product per touched list, with the
  lists crossing the bus.

### On clustered data (same ids and shards; vectors replaced by a 4096-centre Gaussian mixture, centres N(0, 1), noise sigma 0.3)

This is not the bench data. It shows what clustering buys when the data has clusters. Here HNSW reaches recall 1.0000 already at ef 16.

| method (recall@10) | batch 1: q/s, p50 ms | batch 64: q/s | batch 1024: q/s |
|---|---|---|---|
| CPU HNSW per shard ef 16, 60 pinned (0.9999-1.0) | 175.4, 4.19 | 1707.8 | 2184.2 |
| CPU flat, 60 pinned (1.0) | | | 24.2 |
| GPU `gpu_knn_l2`, exact (1.0) | 498.8, 2.00 | 4369.6 | 6296.6 |
| CPU ivf nlist 7588 nprobe 32, 1 thread (0.975) | 402.1, 2.33 | | |
| CPU ivf nlist 7588 nprobe 32, 60 pinned (0.975-0.976) | 94.0, 11.7 | 2032.6 | 3406.0 |
| CPU ivf nlist 7588 nprobe 64, 1 thread (0.9945) | 241.4, 3.92 | | |
| CPU ivf nlist 7588 nprobe 64, 60 pinned (0.994) | 88.5, 11.9 | 1294.9 | 1843.3 |
| CPU ivf nlist 1897 nprobe 32, 60 pinned (0.998) | 87.1, 12.0 | 1342.7 | 1848.9 |
| GPU ivf nlist 7588 nprobe 32, scan kernel (0.975-0.976) | 1070.9, 0.92 | 12257.0 | 37438.0 |
| GPU ivf nlist 7588 nprobe 64, scan kernel (0.994) | 943.1, 1.02 | 10845.2 | 27270.0 |
| GPU ivf nlist 1897 nprobe 16, scan kernel (0.976-0.982) | 672.3, 1.35 | 9463.7 | 37340.0 |
| GPU ivf nlist 1897 nprobe 32, scan kernel (0.998) | 599.3, 1.49 | 8589.4 | 23704.5 |

Training took 9.54 s (nlist 1897) and 38.71 s (nlist 7588); assigning the rows took 9.75 s and 30.07 s.

- On the GPU, at recall 0.994 to 0.998, IVF reached 23,705 to 27,270 q/s at batch 1024. That is 3.8x to 4.3x GPU brute force and 10.9x
  to 12.5x CPU HNSW. At batch 64 it is 2.0x to 2.5x GPU brute force, and at batch 1 it is 1.2x to 1.9x.
- On the CPU with one shard per pinned core, IVF is not faster than HNSW at matched recall: 1843 to 1849 q/s against 2184 at batch 1024.
  At batch 1 the fan-out costs more than the search, so one thread running IVF (241 to 402 q/s) beats the 60-worker fan-out of either
  method (88 to 175 q/s).

### Cluster-aligned shards

The source is a copy of the 3.6M bench rows, re-keyed `(cluster, id)` with the nlist 1897 centroids: 106 shards of up to 10 MiB, 931 MiB
in all, with the centroids stored in `manifest.json` under stats `ivf`. A query reads its nprobe nearest clusters by key range from the
shards that hold them. It is compared with the key-sorted 255-shard set (perf/bench_db's 10 MiB budget) and `ann::shard_ann` over every
shard, on a `task::pool(64)`. Building the cluster copy took 45.1 s (assign, sort, write; one-off). Bytes read is an estimate: rows read
times the shard's average row bytes.

| method | recall@10 | shards opened / query | rows read / query | bytes read / query | fresh: q/s, p50, p99 ms | warm: q/s, p50, p99 ms |
|---|---|---|---|---|---|---|
| cluster shards nprobe 8, 64 pinned | 0.3265 | 8.1 of 106 | 15,456 | 4.0 MiB | 175.0, 5.15, 9.89 | 182.5, 4.79, 9.43 |
| cluster shards nprobe 64, 64 pinned | 0.7585 | 50.6 | 123,818 | 32.0 MiB | 94.1, 9.80, 19.3 | 100.4, 9.41, 15.6 |
| cluster shards nprobe 256, 64 pinned | 0.9585 | 98.8 | 494,890 | 128.0 MiB | 44.6, 21.7, 30.7 | 46.4, 21.1, 29.5 |
| cluster shards nprobe 512, 64 pinned | 0.9945 | 105.5 | 989,398 | 255.8 MiB | 28.8, 33.9, 46.7 | 29.0, 33.8, 57.8 |
| cluster shards nprobe 512, 1 thread | 0.9945 | 105.5 | 989,398 | 255.8 MiB | 1.0, 1004, 1575 | 0.9, 1045, 1648 |
| key-sorted + shard_ann, ef 64 | 0.9975 | 255 of 255 | (graphs) | 2242 MiB of shards opened on first use | 45.6, 19.7, 29.0 | 50.2, 19.7, 26.2 |
| key-sorted + shard_ann, ef 160 | 1.0000 | 255 of 255 | (graphs) | (same) | 25.9, 36.8, 47.3 | 25.6, 38.8, 46.2 |

The first query on a fresh key-sorted handle loads every graph: 313 ms median at ef 64 and 377 ms at ef 160. Its time is included in
the fresh pass.

On these vectors the cluster-aligned layout gives no saving at useful recall. To reach 0.99 a query must open 105 of the 106 shards and
read 256 MiB, and it runs at 29 q/s, against 46 to 50 q/s for `shard_ann` at higher recall. The I/O saving is real only at low recall
(nprobe 8: 8 shards, 4 MiB, recall 0.33). I did not build cluster-aligned shards for the mixture data: the key-sorted comparison set
holds graphs over the original vectors, so the comparison would not have been like for like.

### IVF-PQ

Not measured. I spent the time on the matched-recall runs above.

## FastSMT

Data: the M1-labelled set, read-only, joined with its formula text by `tools/fastsmt/bench/prep_m1_labelled.py` (in OS-Proving). It
holds 7272 formulas with a labelled time for all 6 strategies. The scripts are `tools/fastsmt/bench/` in OS-Proving.

### Strategy-hint kNN (prism::sim, CPU only)

Each formula's signature is computed with `embed_smt2`, which parses it with Z3. Formulas average tens of KB, and Z3 keeps the memory it
used, so the embeddings ran in separate processes of 500 formulas, 16 pinned workers each, to stay under the 8 GB RSS budget. Two
500-formula chunks went over the budget and were retried in chunks of 50 with 8 workers; two of those still went over and were left out
(100 formulas). In all, 7007 of 7272 formulas were embedded; the rest failed to parse or were left out.

The index holds 7007 entries. Every entry is also a query, and it finds itself at distance 0, so this times the lookup; it does not
measure hint quality. k = 15, regression hint. `strategy_hint` as shipped uses HNSW (ef 128) because the index has more than 2000
entries.

| step | config | queries/s | p50 ms | p99 ms |
|---|---|---|---|---|
| `embed_smt2` (signature), one-off per formula | 16 pinned workers | 12.7 formulas/s | 29.5 | 12074 |
| sym_index HNSW build, 7007 entries (one-off) | task::pool(64) | 2.0 s total | | |
| `strategy_hint` as shipped (HNSW) | 1 thread, 2000 queries | 106.2 | 7.60 | 33.1 |
| `strategy_hint` as shipped (HNSW) | 64 pinned workers, queries split | 18,982 | 0.444 | 13.2 |
| exact kNN (`search_exact`) + regression | 1 thread, 300 queries | 26.7 | 34.6 | 80.7 |
| exact kNN + regression | 64 pinned workers, queries split | 2,889 | 11.8 | 54.5 |
| exact kNN over 64 index shards (one per pinned core, fan-out per query) + regression | 64 pinned workers | 172.2 | 4.34 | 9.04 |

The sharded exact search returned the same neighbour distances as the unsharded one on all 300 checked queries. Each row ran 5 times
except the two one-off rows; medians are shown, and the spread is in `bench_gpu_vs_cpu/fastsmt_hint_knn.md`.

There is no GPU row because there is no GPU path for this metric. A lookup takes microseconds to milliseconds, against a z3 run of tens
of milliseconds to 10 s, so this step is not where FastSMT spends its time.

### Solver step (CPU only, for context)

z3 4.8.12 ran the 6 M1 strategies on 20 formulas, 5 reps each, pinned to CPU 63 with `taskset`. The sample is the first 20 formulas whose
labelled times are all under 2 s, so it is biased to easy formulas. It ran twice: run 1 at load 72 to 79, run 2 at load 91 to 112.

| strategy | run 1: median wall s | run 1: per-formula max/min over reps (median, worst) | run 2: median wall s | run 2: max/min (median, worst) | answers (each run) |
|---|---|---|---|---|---|
| default | 0.0564 | 1.23, 1.51 | 0.0455 | 1.30, 2.14 | sat 70, unsat 30 |
| qfbv | 0.0634 | 1.24, 1.42 | 0.0512 | 1.35, 1.79 | sat 70, unsat 30 |
| simplify-bitblast-sat | 0.0626 | 1.22, 1.45 | 0.0582 | 1.30, 1.56 | sat 70, unsat 30 |
| smt | 0.0480 | 1.19, 1.51 | 0.0428 | 1.30, 1.49 | sat 70, unsat 30 |
| solveeqs-bitblast-sat | 0.0666 | 1.19, 1.48 | 0.0534 | 1.26, 2.33 | sat 70, unsat 30 |
| propagate-smt | 0.0524 | 1.22, 1.41 | 0.0522 | 1.19, 1.45 | sat 70, unsat 30 |

The labelling loop also recorded times over all 7272 formulas; the loop measured those, not this bench. For each strategy, between 3736
(smt) and 4651 (simplify-bitblast-sat) of the 7272 formulas hit the 10 s timeout. The mean is 5.3 to 6.5 s per formula per strategy.

There is no model-scoring step to put on a GPU. The hint is a kNN regression; there is no learned model.

## Reproduce

```sh
# WSL, from the gpu-bench worktree
cmake -S . -B ~/pbuild/forge-gpubench -G Ninja -DCMAKE_BUILD_TYPE=Release -DFORGE_BUILD_DB_BENCH=ON
ninja -C ~/pbuild/forge-gpubench bench_db bench_gpu_cpu bench_ivf test_gpu_ann
for t in gpu_flat_device gpu_knn_logic gpu_knn_device ivf_logic ivf_device; do FORGE_TEST_ONLY=$t ~/pbuild/forge-gpubench/test_gpu_ann; done
BUILD=~/pbuild/forge-gpubench DATA=/dev/shm/gpubench OUT=$PWD/bench-out REPS=5 bash perf/run_bench_gpu_cpu.sh
# FastSMT (OS-Proving): M1=/mnt/d/ParanoidBSD/data-fastsmt/m1 OUT=<dir> bash tools/fastsmt/bench/build.sh && bash tools/fastsmt/bench/run_bench.sh
```

`run_bench_gpu_cpu.sh` builds the 60-shard set (about 24 min with 8 threads) and the 255-shard set if they are missing. It then runs:

- `bench_gpu_cpu --dir $DATA/db60/sharded`;
- `bench_ivf --dir $DATA/db60/sharded --keysorted $DATA/db255/sharded --cluster-out $DATA/cluster`;
- `bench_ivf --mixture 4096 --sections ivf`.

`nvidia-smi` is sampled every 0.5 s alongside each run. Allow about 3 h on this machine under its usual load.
