# bench_hint_knn: 7272 labelled formulas from /mnt/d/ParanoidBSD/wt/gpubench-work/fastsmt/m1_labelled.tsv, 64 pinned workers, reps 5, load 103, avail 30.1 GiB, rss 445 MiB

No GPU path: prism::sim's metric (weighted Jaccard + Hamming / sub-term Jaccard on sparse variable-length vectors) has no FORGE GPU kernel; ann::gpu_flat_l2 and ann::gpu_knn_l2 are dense l2 only.

| step | config | items/s | p50 ms | p99 ms | reps | machine before / after | notes |
|---|---|---|---|---|---|---|---|
| embed_smt2 (signature) | 16 pinned workers, 31 chunks in separate processes | 12.7 | 29.527 | 12073.925 | 1 | (per chunk in embed.log) | 7007 of 7272 embedded; 563 s of chunk wall time |
| sym_index HNSW build (7007 entries) | task::pool(64) | | 2.0 s total | | 1 | load 103, avail 30.1 GiB | |
| strategy_hint as shipped (HNSW ef 128, k=15) (2000 queries) | 1 thread (pinned) | 106.2 (99.0..122.5) | 7.596 (6.915..8.323) | 33.101 (23.633..35.787) | 5 | load 103, avail 30.1 GiB / load 98, avail 28.5 GiB |
| strategy_hint as shipped (HNSW ef 128, k=15) (7007 queries) | 64 pinned workers, queries split | 18981.7 (17032.7..20491.2) | 0.444 (0.441..0.462) | 13.219 (12.733..13.688) | 5 | load 98, avail 28.5 GiB / load 98, avail 28.3 GiB |
| hint, exact kNN (search_exact, k=15) + regression (300 queries) | 1 thread (pinned) | 26.7 (20.0..31.0) | 34.550 (30.848..47.497) | 80.703 (59.049..123.711) | 5 | load 98, avail 28.3 GiB / load 98, avail 28.0 GiB |
| hint, exact kNN (search_exact, k=15) + regression (7007 queries) | 64 pinned workers, queries split | 2888.8 (2666.4..3143.5) | 11.753 (11.618..12.551) | 54.508 (51.328..63.146) | 5 | load 98, avail 28.0 GiB / load 116, avail 27.8 GiB |
| hint, exact kNN over 64 shards (one per pinned core, fan-out per query) + regression (7007 queries) | 64 pinned workers | 172.2 (158.8..183.4) | 4.338 (4.226..4.613) | 9.035 (8.297..11.367) | 5 | load 116, avail 27.7 GiB / load 99, avail 28.0 GiB | neighbour distances equal to the unsharded exact search on 300 of 300 checked queries

rss at end 519 MiB, load 99, avail 28.0 GiB
