# bench_gpu_cpu: /dev/shm/gpubench/db60/sharded, 60 shards, 64 CPUs available, pinned workers 60 (one shard per core), reps 5, load 75, avail 26.0 GiB, rss 4 MiB

shard files verified against the manifest (sha256), load 78, avail 24.6 GiB

## db: BM25 top-10 and point lookups (no GPU path exists in FORGE for either)

| workload | config | handle | queries/s | p50 ms | p99 ms | reps | machine before / after |
|---|---|---|---|---|---|---|---|
| bm25 frequent words (2-3 terms, OR) (20 queries) | pinned, 60 workers (1 shard/core) | fresh | 12.6 (12.1..13.1) | 67.467 (63.236..72.896) | 186.769 (176.910..206.854) | 5 | load 82, avail 25.7 GiB / load 93, avail 26.2 GiB | results = library
| bm25 frequent words (2-3 terms, OR) (20 queries) | pinned, 60 workers (1 shard/core) | warm | 13.2 (12.7..13.6) | 65.165 (61.612..67.186) | 168.098 (156.808..176.905) | 5 | load 82, avail 25.7 GiB / load 93, avail 26.2 GiB | results = library
| bm25 clustered rare term (50 queries) | pinned, 60 workers (1 shard/core) | fresh | 56.3 (53.3..60.0) | 17.796 (16.402..18.746) | 26.618 (24.273..29.301) | 5 | load 93, avail 26.2 GiB / load 92, avail 26.8 GiB | results = library
| bm25 clustered rare term (50 queries) | pinned, 60 workers (1 shard/core) | warm | 58.8 (50.2..61.8) | 17.071 (15.939..19.737) | 29.682 (23.645..38.876) | 5 | load 93, avail 26.2 GiB / load 92, avail 26.8 GiB | results = library
| point lookup by key (20000 keys, latency = one lookup on its worker) | pinned, 60 workers (1 shard/core) | fresh | 302650.7 (239764.9..496622.3) | 0.006 (0.006..0.009) | 0.046 (0.044..0.070) | 5 | load 92, avail 26.8 GiB / load 92, avail 27.1 GiB | all keys found
| point lookup by key (20000 keys, latency = one lookup on its worker) | pinned, 60 workers (1 shard/core) | warm | 2595539.9 (2399409.8..3936281.1) | 0.005 (0.005..0.005) | 0.017 (0.015..0.021) | 5 | load 92, avail 26.8 GiB / load 92, avail 27.1 GiB | all keys found
| bm25 frequent words (2-3 terms, OR) (5 queries) | 1 thread (pinned) | fresh | 0.2 (0.2..0.3) | 2704.685 (2048.769..3019.582) | 7324.675 (7175.919..9467.668) | 5 | load 92, avail 27.1 GiB / load 74, avail 27.5 GiB | results = library
| bm25 frequent words (2-3 terms, OR) (5 queries) | 1 thread (pinned) | warm | 0.2 (0.2..0.3) | 2356.603 (1894.642..2578.054) | 7955.314 (7114.614..8433.002) | 5 | load 92, avail 27.1 GiB / load 74, avail 27.5 GiB | results = library
| bm25 clustered rare term (50 queries) | 1 thread (pinned) | fresh | 25.7 (22.1..32.7) | 36.988 (29.450..42.170) | 74.952 (52.786..89.097) | 5 | load 74, avail 27.5 GiB / load 74, avail 27.2 GiB | results = library
| bm25 clustered rare term (50 queries) | 1 thread (pinned) | warm | 30.5 (26.9..35.1) | 30.272 (25.164..33.320) | 70.879 (58.437..79.777) | 5 | load 74, avail 27.5 GiB / load 74, avail 27.2 GiB | results = library
| point lookup by key (20000 keys, latency = one lookup on its worker) | 1 thread (pinned) | fresh | 16674.7 (11333.6..20645.6) | 0.028 (0.023..0.048) | 0.343 (0.314..0.436) | 5 | load 74, avail 27.2 GiB / load 75, avail 26.2 GiB | all keys found
| point lookup by key (20000 keys, latency = one lookup on its worker) | 1 thread (pinned) | warm | 31419.5 (19665.3..49697.0) | 0.011 (0.007..0.021) | 0.230 (0.191..0.323) | 5 | load 74, avail 27.2 GiB / load 75, avail 26.2 GiB | all keys found
| bm25 frequent words (2-3 terms, OR) (20 queries) | library shard_bm25_topk, task::pool(60), unpinned | fresh | 2.5 (2.3..2.9) | 389.164 (329.432..416.123) | 541.274 (480.521..622.937) | 5 | load 75, avail 26.2 GiB / load 92, avail 28.5 GiB |
| bm25 frequent words (2-3 terms, OR) (20 queries) | library shard_bm25_topk, task::pool(60), unpinned | warm | 2.5 (2.1..2.5) | 379.534 (371.342..444.086) | 600.300 (522.036..614.851) | 5 | load 75, avail 26.2 GiB / load 92, avail 28.5 GiB |
| bm25 clustered rare term (50 queries) | library shard_bm25_topk, task::pool(60), unpinned | fresh | 41.2 (39.9..42.5) | 23.239 (22.918..24.614) | 32.452 (28.474..32.682) | 5 | load 92, avail 28.5 GiB / load 90, avail 27.5 GiB |
| bm25 clustered rare term (50 queries) | library shard_bm25_topk, task::pool(60), unpinned | warm | 42.3 (39.8..45.1) | 23.099 (21.749..24.640) | 30.728 (28.954..34.188) | 5 | load 92, avail 28.5 GiB / load 90, avail 27.5 GiB |

## ann: 3600000 vectors (dim 32) in 60 shards, 1024 queries, top-10; rss 844 MiB, load 90, avail 26.8 GiB

per-shard flat indexes and HNSW graphs loaded on their workers: 2.8 s, rss 3469 MiB

exact answers (pinned flat, 1024 queries): 16.67 s

| method | batch | queries/s | latency p50 ms | latency p99 ms | reps | machine before / after | recall@10 |
|---|---|---|---|---|---|---|---|
| CPU flat (exact), 60 pinned workers | 1 (200 queries) | 41.5 (41.1..42.1) | 23.739 (23.321..24.016) | 30.515 (29.328..31.547) | 5 | load 105, avail 26.7 GiB / load 112, avail 28.3 GiB | 1.0000 (= exact, ids and distances)
| CPU flat (exact), 60 pinned workers | 64 (1024 queries) | 62.5 (61.4..64.5) | 1026.911 (994.634..1044.447) | 1057.934 (1010.078..1104.340) | 5 | load 112, avail 28.3 GiB / load 122, avail 27.7 GiB | 1.0000 (= exact, ids and distances)
| CPU flat (exact), 60 pinned workers | 1024 (1024 queries) | 64.1 (63.3..66.8) | 15977.669 (15321.946..16176.884) | 15977.669 (15321.946..16176.884) | 5 | load 122, avail 27.7 GiB / load 124, avail 28.1 GiB | 1.0000 (= exact, ids and distances)
| CPU HNSW per shard, ef 16 (recall sweep, 256 queries) | | | | | | | 0.8191 |
| CPU HNSW per shard, ef 32 (recall sweep, 256 queries) | | | | | | | 0.9219 |
| CPU HNSW per shard, ef 64 (recall sweep, 256 queries) | | | | | | | 0.9730 |
| CPU HNSW per shard, ef 128 (recall sweep, 256 queries) | | | | | | | 0.9918 |
| CPU HNSW per shard, ef 256 (recall sweep, 256 queries) | | | | | | | 0.9988 |
| CPU HNSW per shard, ef 512 (recall sweep, 256 queries) | | | | | | | 1.0000 |
| CPU HNSW per shard, ef 64, 60 pinned workers | 1 (200 queries) | 125.9 (125.1..127.6) | 7.975 (7.946..7.987) | 9.659 (9.013..10.801) | 5 | load 120, avail 27.6 GiB / load 120, avail 26.9 GiB | 0.9735
| CPU HNSW per shard, ef 64, 60 pinned workers | 64 (1024 queries) | 336.6 (313.2..341.0) | 189.067 (186.043..205.999) | 213.034 (200.433..218.322) | 5 | load 120, avail 26.9 GiB / load 121, avail 26.8 GiB | 0.9747
| CPU HNSW per shard, ef 64, 60 pinned workers | 1024 (1024 queries) | 381.5 (349.9..395.6) | 2680.531 (2585.007..2921.651) | 2680.531 (2585.007..2921.651) | 5 | load 121, avail 26.8 GiB / load 118, avail 25.7 GiB | 0.9747
| CPU HNSW per shard, ef 128, 60 pinned workers | 1 (200 queries) | 108.3 (107.3..112.4) | 8.867 (8.580..9.011) | 13.966 (12.585..14.186) | 5 | load 118, avail 25.7 GiB / load 114, avail 26.8 GiB | 0.9910
| CPU HNSW per shard, ef 128, 60 pinned workers | 64 (1024 queries) | 210.4 (197.9..221.2) | 303.919 (292.243..321.131) | 322.277 (306.232..369.967) | 5 | load 114, avail 26.8 GiB / load 114, avail 27.4 GiB | 0.9938
| CPU HNSW per shard, ef 128, 60 pinned workers | 1024 (1024 queries) | 250.1 (244.8..281.0) | 4091.853 (3641.568..4178.867) | 4091.853 (3641.568..4178.867) | 5 | load 114, avail 27.3 GiB / load 115, avail 28.0 GiB | 0.9938
| CPU flat (exact), 1 thread | 1 (20 queries) | 2.5 (2.2..2.7) | 389.491 (367.176..443.605) | 621.793 (596.981..691.421) | 5 | load 115, avail 28.0 GiB / load 94, avail 27.4 GiB | 1.0000 (= exact, ids and distances)
| CPU flat (exact), 1 thread | 64 (64 queries) | 1.9 (1.6..2.0) | 33741.950 (31621.608..39226.569) | 33741.950 (31621.608..39226.569) | 5 | load 94, avail 27.4 GiB / load 78, avail 24.8 GiB | 1.0000 (= exact, ids and distances)
| CPU HNSW per shard, ef 128, 1 thread | 1 (200 queries) | 2.9 (2.1..3.2) | 324.776 (296.955..460.943) | 608.902 (542.170..977.460) | 5 | load 78, avail 24.8 GiB / load 80, avail 26.0 GiB | 0.9910
| GPU: cuda (NVIDIA GeForce RTX 3090), image ptx sm_80 | | | | | | | |
| GPU gpu_flat_l2 (as shipped), all 60 shards in one index | 1 (20 queries) | 2.3 (2.2..2.5) | 430.333 (403.172..460.187) | 465.885 (442.343..490.452) | 5 | load 80, avail 25.3 GiB / load 77, avail 25.3 GiB | 1.0000 (= exact, ids and distances)
|   (device matmul_t kernel launches during the row above: 5555; 5555 device calls; fell back to CPU: no; bytes over the bus per 1-query block: rows 439 MiB up, distances 14 MiB down) | | | | | | | |
| GPU gpu_flat_l2 (as shipped), all 60 shards in one index | 64 (256 queries) | 48.2 (47.2..52.2) | 1337.138 (1218.462..1373.157) | 1387.403 (1268.596..1484.297) | 5 | load 77, avail 25.4 GiB / load 81, avail 25.9 GiB | 1.0000 (= exact, ids and distances)
|   (device matmul_t kernel launches during the row above: 1155; 1155 device calls; fell back to CPU: no; bytes over the bus per 64-query block: rows 439 MiB up, distances 879 MiB down) | | | | | | | |
| GPU gpu_flat_l2 (as shipped), all 60 shards in one index | 1024 (1024 queries) | 65.1 (63.8..68.5) | 15719.512 (14941.795..16056.324) | 15719.512 (14941.795..16056.324) | 5 | load 81, avail 25.9 GiB / load 86, avail 27.4 GiB | 1.0000 (= exact, ids and distances)
|   (device matmul_t kernel launches during the row above: 2640; 2640 device calls; fell back to CPU: no; bytes over the bus per 128-query block: rows 439 MiB up, distances 1758 MiB down) | | | | | | | |
| GPU gpu_knn_l2: host-to-device upload of all 3600000 rows (439 MiB), once per index | | 215.5 (194.8..251.6) ms | | 5 | load 85, avail 27.3 GiB / load 85, avail 27.2 GiB | 2.14 GB/s (median) |
|   (device memory in use after upload: 1.68 GiB of 24.00 GiB) | | | | | | | |
| GPU gpu_knn_l2 (new), all 60 shards in one resident index | 1 (200 queries) | 485.6 (477.0..493.4) | 2.046 (2.016..2.068) | 2.228 (2.186..2.343) | 5 | load 85, avail 27.2 GiB / load 85, avail 25.8 GiB | 1.0000 (= exact, ids and distances)
|   (forge_knn_l2_part launches: 1001; queries answered on the CPU by the exactness check: 0; device call failed: no) | | | | | | | |
| GPU gpu_knn_l2 (new), all 60 shards in one resident index | 64 (1024 queries) | 4278.7 (4240.3..4346.9) | 14.976 (14.634..15.063) | 15.196 (15.175..15.447) | 5 | load 85, avail 25.8 GiB / load 85, avail 25.2 GiB | 1.0000 (= exact, ids and distances)
|   (forge_knn_l2_part launches: 81; queries answered on the CPU by the exactness check: 0; device call failed: no) | | | | | | | |
| GPU gpu_knn_l2 (new), all 60 shards in one resident index | 1024 (1024 queries) | 6298.3 (6257.5..6308.7) | 162.129 (161.726..163.027) | 162.129 (161.726..163.027) | 5 | load 85, avail 25.2 GiB / load 85, avail 25.0 GiB | 1.0000 (= exact, ids and distances)
|   (forge_knn_l2_part launches: 6; queries answered on the CPU by the exactness check: 0; device call failed: no) | | | | | | | |
|   (device call only, knn_l2_16, batch 1: 2.032 (1.992..2.288) ms over 20 calls; load 85, avail 25.0 GiB) | | | | | | | |
|   (device call only, knn_l2_16, batch 64: 13.154 (12.789..13.361) ms over 20 calls; load 85, avail 25.0 GiB) | | | | | | | |
|   (device call only, knn_l2_16, batch 1024: 160.368 (159.632..161.181) ms over 20 calls; load 85, avail 25.0 GiB) | | | | | | | |
| GPU gpu_knn_l2 (new), one resident index per shard, shards in turn | 1 (50 queries) | 40.0 (39.7..40.9) | 24.976 (24.300..25.166) | 26.401 (25.712..26.522) | 5 | load 84, avail 24.2 GiB / load 84, avail 24.7 GiB | 1.0000 (= exact, ids and distances)
| GPU gpu_knn_l2 (new), one resident index per shard, shards in turn | 64 (1024 queries) | 323.2 (255.9..424.3) | 198.199 (149.941..262.220) | 211.268 (180.775..274.552) | 5 | load 84, avail 24.7 GiB / load 83, avail 24.0 GiB | 1.0000 (= exact, ids and distances)
| GPU gpu_knn_l2 (new), one resident index per shard, shards in turn | 1024 (1024 queries) | 2281.3 (2240.8..2324.6) | 447.988 (439.761..456.213) | 447.988 (439.761..456.213) | 5 | load 83, avail 24.0 GiB / load 83, avail 23.8 GiB | 1.0000 (= exact, ids and distances)

rss at end 2852 MiB, load 83, avail 24.3 GiB
