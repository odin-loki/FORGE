# bench_ivf: 3600000 vectors (dim 32, Gaussian mixture of 4096 centres, sigma 0.3) from /dev/shm/gpubench/db60/sharded (60 shards), 1024 queries, pinned workers 60, reps 5, load 77, avail 25.6 GiB, rss 1282 MiB

exact answers: gpu_knn_l2 0.44 s (0 of 1024 queries on the CPU path, device yes)

## IVF (ann::ivf_flat, new): k-means on 180000 sample rows (5%), 20 Lloyd iterations, task::pool(60)

| method | batch | queries/s | p50 ms | p99 ms | reps | machine before / after | recall@10 |
|---|---|---|---|---|---|---|---|
| baselines: per-shard flat + HNSW built on their workers (one-off): 176.9 s | | | | | 1 | load 102, avail 26.0 GiB | |
| CPU HNSW per shard ef 16 (recall sweep, 256 queries) | | | | | | | 1.0000 |
| CPU HNSW per shard ef 32 (recall sweep, 256 queries) | | | | | | | 1.0000 |
| CPU HNSW per shard ef 64 (recall sweep, 256 queries) | | | | | | | 1.0000 |
| CPU HNSW per shard ef 128 (recall sweep, 256 queries) | | | | | | | 1.0000 |
| CPU HNSW per shard ef 256 (recall sweep, 256 queries) | | | | | | | 1.0000 |
| CPU HNSW per shard ef 512 (recall sweep, 256 queries) | | | | | | | 1.0000 |
| CPU HNSW per shard ef 16 (first ef reaching the target), one shard per pinned core (60) | 1 (200 queries) | 175.4 (170.6..178.6) | 4.191 (4.134..4.202) | 8.105 (8.081..8.122) | 5 | load 105, avail 26.6 GiB / load 102, avail 26.9 GiB | 1.0000
| CPU HNSW per shard ef 16 (first ef reaching the target), one shard per pinned core (60) | 64 (1024 queries) | 1707.8 (1633.7..1939.7) | 35.726 (32.013..36.789) | 53.873 (40.379..59.536) | 5 | load 102, avail 26.9 GiB / load 105, avail 26.6 GiB | 0.9999
| CPU HNSW per shard ef 16 (first ef reaching the target), one shard per pinned core (60) | 1024 (1024 queries) | 2184.2 (1882.8..2393.0) | 468.313 (427.536..543.376) | 468.313 (427.536..543.376) | 5 | load 105, avail 26.6 GiB / load 105, avail 26.5 GiB | 0.9999
| CPU flat (exact), one shard per pinned core (60) | 1024 (1024 queries) | 24.2 (23.4..24.9) | 42320.632 (41160.383..43672.025) | 42320.632 (41160.383..43672.025) | 5 | load 105, avail 26.5 GiB / load 115, avail 28.0 GiB | 1.0000
| GPU gpu_knn_l2 (exact, resident) | 1 (200 queries) | 498.8 (384.2..508.5) | 1.999 (1.964..2.071) | 2.191 (2.046..19.074) | 5 | load 115, avail 27.3 GiB / load 112, avail 27.1 GiB | 1.0000
| GPU gpu_knn_l2 (exact, resident) | 64 (1024 queries) | 4369.6 (4341.8..4472.0) | 14.658 (14.305..14.678) | 14.853 (14.462..14.999) | 5 | load 112, avail 27.1 GiB / load 112, avail 27.2 GiB | 1.0000
| GPU gpu_knn_l2 (exact, resident) | 1024 (1024 queries) | 6296.6 (6276.6..6309.4) | 162.227 (161.795..162.565) | 162.227 (161.795..162.565) | 5 | load 112, avail 27.2 GiB / load 112, avail 27.2 GiB | 1.0000
| training nlist 1897 (one-off): k-means 9.54 (9.02..10.57) s | | | | | 5 | load 107, avail 27.4 GiB / load 107, avail 24.7 GiB | |
| index build nlist 1897 (one-off): assign 3600000 rows 9.75 s; list sizes min 8, mean 1898, max 42734 | | | | | 1 | load 101, avail 24.2 GiB | |
| ivf nlist 1897 nprobe 1 (recall sweep) | | | | | | rows scanned per query 5294 (0.15% of N) | 0.5781 |
| ivf nlist 1897 nprobe 2 (recall sweep) | | | | | | rows scanned per query 9929 (0.28% of N) | 0.7369 |
| ivf nlist 1897 nprobe 4 (recall sweep) | | | | | | rows scanned per query 20242 (0.56% of N) | 0.8582 |
| ivf nlist 1897 nprobe 8 (recall sweep) | | | | | | rows scanned per query 40131 (1.11% of N) | 0.9405 |
| ivf nlist 1897 nprobe 16 (recall sweep) | | | | | | rows scanned per query 79121 (2.20% of N) | 0.9823 |
| ivf nlist 1897 nprobe 32 (recall sweep) | | | | | | rows scanned per query 152646 (4.24% of N) | 0.9979 |
| ivf nlist 1897 nprobe 64 (recall sweep) | | | | | | rows scanned per query 287636 (7.99% of N) | 0.9999 |
| ivf nlist 1897 nprobe 128 (recall sweep) | | | | | | rows scanned per query 532201 (14.78% of N) | 1.0000 |
| ivf nlist 1897 nprobe 256 (recall sweep) | | | | | | rows scanned per query 953616 (26.49% of N) | 1.0000 |
| ivf nlist 1897 nprobe 512 (recall sweep) | | | | | | rows scanned per query 1641059 (45.58% of N) | 1.0000 |
| ivf nlist 1897 nprobe 1024 (recall sweep) | | | | | | rows scanned per query 2649080 (73.59% of N) | 1.0000 |
| CPU ivf nlist 1897 nprobe 16, 1 thread | 1 (200 queries) | 291.8 (265.2..361.2) | 3.076 (2.504..3.413) | 6.774 (6.508..9.278) | 5 | load 118, avail 26.4 GiB / load 113, avail 25.8 GiB | 0.9755
| CPU ivf nlist 1897 nprobe 16, one shard per pinned core (60) | 1 (200 queries) | 91.8 (90.4..95.8) | 11.883 (11.869..11.909) | 16.081 (15.936..16.126) | 5 | load 113, avail 25.8 GiB / load 108, avail 24.9 GiB | 0.9755
| CPU ivf nlist 1897 nprobe 16, one shard per pinned core (60) | 64 (1024 queries) | 1625.4 (1300.1..1815.7) | 38.350 (35.071..41.858) | 50.072 (37.224..92.219) | 5 | load 108, avail 24.9 GiB / load 108, avail 24.7 GiB | 0.9823
| CPU ivf nlist 1897 nprobe 16, one shard per pinned core (60) | 1024 (1024 queries) | 2715.4 (2556.8..2762.1) | 376.465 (370.077..399.929) | 376.465 (370.077..399.929) | 5 | load 108, avail 24.7 GiB / load 108, avail 24.8 GiB | 0.9823
| GPU ivf nlist 1897 nprobe 16: upload of the 3600000 rows sorted by list (once) | | 239.3 ms | | | 1 | load 108, avail 24.8 GiB | 1.93 GB/s |
| GPU ivf nlist 1897 nprobe 16, scan kernel (new; rows resident, probes on the host pool) | 1 (200 queries) | 672.3 (616.7..673.7) | 1.349 (1.325..1.409) | 2.651 (2.624..2.848) | 5 | load 108, avail 24.8 GiB / load 108, avail 24.7 GiB | 0.9755
|   (forge_ivf_l2_part launches 1001) | | | | | | | |
| GPU ivf nlist 1897 nprobe 16, scan kernel (new; rows resident, probes on the host pool) | 64 (1024 queries) | 9463.7 (9213.5..9877.1) | 6.676 (6.469..6.978) | 7.504 (7.433..7.887) | 5 | load 108, avail 24.7 GiB / load 104, avail 24.8 GiB | 0.9823
|   (forge_ivf_l2_part launches 81) | | | | | | | |
| GPU ivf nlist 1897 nprobe 16, scan kernel (new; rows resident, probes on the host pool) | 1024 (1024 queries) | 37340.0 (35192.1..38021.5) | 26.854 (26.652..28.644) | 26.854 (26.652..28.644) | 5 | load 104, avail 24.8 GiB / load 104, avail 24.7 GiB | 0.9823
|   (forge_ivf_l2_part launches 6) | | | | | | | |
| GPU ivf nlist 1897 nprobe 16, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 64 (256 queries) | 93.7 (86.8..103.1) | 683.512 (617.906..739.123) | 734.377 (642.605..761.393) | 5 | load 104, avail 24.7 GiB / load 95, avail 24.3 GiB | 0.9781
|   (device matmul products over the calls of the row above, warm-up included: 13410) | | | | | | | |
| GPU ivf nlist 1897 nprobe 16, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 1024 (1024 queries) | 394.3 (358.9..412.0) | 2589.118 (2472.500..2835.348) | 2589.118 (2472.500..2835.348) | 5 | load 95, avail 24.3 GiB / load 90, avail 24.1 GiB | 0.9823
|   (device matmul products over the calls of the row above, warm-up included: 10224) | | | | | | | |
| CPU ivf nlist 1897 nprobe 32, 1 thread | 1 (200 queries) | 164.2 (127.3..167.2) | 5.818 (5.637..7.567) | 12.148 (10.999..17.484) | 5 | load 90, avail 24.1 GiB / load 90, avail 24.3 GiB | 0.9980
| CPU ivf nlist 1897 nprobe 32, one shard per pinned core (60) | 1 (200 queries) | 87.1 (86.1..90.0) | 11.952 (11.882..11.964) | 16.049 (16.019..19.626) | 5 | load 90, avail 24.3 GiB / load 87, avail 26.2 GiB | 0.9980
| CPU ivf nlist 1897 nprobe 32, one shard per pinned core (60) | 64 (1024 queries) | 1342.7 (1187.6..1357.4) | 48.361 (45.176..53.677) | 58.992 (54.290..66.257) | 5 | load 87, avail 26.2 GiB / load 88, avail 26.2 GiB | 0.9979
| CPU ivf nlist 1897 nprobe 32, one shard per pinned core (60) | 1024 (1024 queries) | 1848.9 (1733.7..1935.3) | 553.213 (528.546..590.041) | 553.213 (528.546..590.041) | 5 | load 88, avail 26.2 GiB / load 92, avail 26.2 GiB | 0.9979
| GPU ivf nlist 1897 nprobe 32: upload of the 3600000 rows sorted by list (once) | | 258.8 ms | | | 1 | load 92, avail 26.1 GiB | 1.78 GB/s |
| GPU ivf nlist 1897 nprobe 32, scan kernel (new; rows resident, probes on the host pool) | 1 (200 queries) | 599.3 (401.7..611.8) | 1.488 (1.449..1.674) | 2.584 (2.560..14.325) | 5 | load 92, avail 26.1 GiB / load 92, avail 26.0 GiB | 0.9980
|   (forge_ivf_l2_part launches 1001) | | | | | | | |
| GPU ivf nlist 1897 nprobe 32, scan kernel (new; rows resident, probes on the host pool) | 64 (1024 queries) | 8589.4 (7977.0..9261.4) | 7.523 (6.893..8.006) | 8.594 (7.542..8.701) | 5 | load 92, avail 26.0 GiB / load 92, avail 25.9 GiB | 0.9979
|   (forge_ivf_l2_part launches 81) | | | | | | | |
| GPU ivf nlist 1897 nprobe 32, scan kernel (new; rows resident, probes on the host pool) | 1024 (1024 queries) | 23704.5 (23648.7..24343.9) | 42.767 (41.828..42.939) | 42.767 (41.828..42.939) | 5 | load 92, avail 25.9 GiB / load 92, avail 25.9 GiB | 0.9979
|   (forge_ivf_l2_part launches 6) | | | | | | | |
| GPU ivf nlist 1897 nprobe 32, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 64 (256 queries) | 71.3 (70.3..79.0) | 897.913 (824.150..907.351) | 934.980 (832.933..943.259) | 5 | load 92, avail 25.9 GiB / load 88, avail 24.8 GiB | 0.9980
|   (device matmul products over the calls of the row above, warm-up included: 20345) | | | | | | | |
| GPU ivf nlist 1897 nprobe 32, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 1024 (1024 queries) | 262.6 (260.2..267.2) | 3899.110 (3830.855..3933.871) | 3899.110 (3830.855..3933.871) | 5 | load 88, avail 24.8 GiB / load 82, avail 26.9 GiB | 0.9979
|   (device matmul products over the calls of the row above, warm-up included: 10914) | | | | | | | |
| training nlist 7588 (one-off): k-means 38.71 (33.70..41.70) s | | | | | 5 | load 82, avail 28.5 GiB / load 107, avail 25.4 GiB | |
| index build nlist 7588 (one-off): assign 3600000 rows 30.07 s; list sizes min 1, mean 474, max 10476 | | | | | 1 | load 113, avail 24.8 GiB | |
| ivf nlist 7588 nprobe 1 (recall sweep) | | | | | | rows scanned per query 1469 (0.04% of N) | 0.4115 |
| ivf nlist 7588 nprobe 2 (recall sweep) | | | | | | rows scanned per query 2625 (0.07% of N) | 0.5752 |
| ivf nlist 7588 nprobe 4 (recall sweep) | | | | | | rows scanned per query 4899 (0.14% of N) | 0.7296 |
| ivf nlist 7588 nprobe 8 (recall sweep) | | | | | | rows scanned per query 9239 (0.26% of N) | 0.8479 |
| ivf nlist 7588 nprobe 16 (recall sweep) | | | | | | rows scanned per query 19275 (0.54% of N) | 0.9315 |
| ivf nlist 7588 nprobe 32 (recall sweep) | | | | | | rows scanned per query 38674 (1.07% of N) | 0.9762 |
| ivf nlist 7588 nprobe 64 (recall sweep) | | | | | | rows scanned per query 76748 (2.13% of N) | 0.9940 |
| ivf nlist 7588 nprobe 128 (recall sweep) | | | | | | rows scanned per query 148421 (4.12% of N) | 0.9993 |
| ivf nlist 7588 nprobe 256 (recall sweep) | | | | | | rows scanned per query 281757 (7.83% of N) | 0.9999 |
| ivf nlist 7588 nprobe 512 (recall sweep) | | | | | | rows scanned per query 522510 (14.51% of N) | 1.0000 |
| ivf nlist 7588 nprobe 1024 (recall sweep) | | | | | | rows scanned per query 938620 (26.07% of N) | 1.0000 |
| ivf nlist 7588 nprobe 2048 (recall sweep) | | | | | | rows scanned per query 1622623 (45.07% of N) | 1.0000 |
| CPU ivf nlist 7588 nprobe 32, 1 thread | 1 (200 queries) | 402.1 (344.3..432.4) | 2.326 (2.188..2.765) | 5.172 (4.794..6.007) | 5 | load 120, avail 24.8 GiB / load 115, avail 24.4 GiB | 0.9750
| CPU ivf nlist 7588 nprobe 32, one shard per pinned core (60) | 1 (200 queries) | 94.0 (91.9..94.7) | 11.658 (10.720..11.924) | 16.093 (15.907..16.107) | 5 | load 115, avail 24.4 GiB / load 110, avail 24.6 GiB | 0.9750
| CPU ivf nlist 7588 nprobe 32, one shard per pinned core (60) | 64 (1024 queries) | 2032.6 (1932.7..2185.9) | 30.600 (29.316..33.093) | 37.351 (33.944..40.484) | 5 | load 110, avail 24.6 GiB / load 110, avail 24.8 GiB | 0.9762
| CPU ivf nlist 7588 nprobe 32, one shard per pinned core (60) | 1024 (1024 queries) | 3406.0 (3378.1..3685.1) | 299.915 (277.249..302.300) | 299.915 (277.249..302.300) | 5 | load 110, avail 24.8 GiB / load 112, avail 25.0 GiB | 0.9762
| GPU ivf nlist 7588 nprobe 32: upload of the 3600000 rows sorted by list (once) | | 226.9 ms | | | 1 | load 112, avail 24.9 GiB | 2.03 GB/s |
| GPU ivf nlist 7588 nprobe 32, scan kernel (new; rows resident, probes on the host pool) | 1 (200 queries) | 1070.9 (1055.1..1073.5) | 0.918 (0.912..0.938) | 1.256 (1.244..1.266) | 5 | load 112, avail 24.9 GiB / load 112, avail 24.8 GiB | 0.9750
|   (forge_ivf_l2_part launches 1001) | | | | | | | |
| GPU ivf nlist 7588 nprobe 32, scan kernel (new; rows resident, probes on the host pool) | 64 (1024 queries) | 12257.0 (11820.2..12816.2) | 5.068 (4.759..5.345) | 6.306 (5.900..7.641) | 5 | load 112, avail 24.8 GiB / load 112, avail 24.7 GiB | 0.9762
|   (forge_ivf_l2_part launches 81) | | | | | | | |
| GPU ivf nlist 7588 nprobe 32, scan kernel (new; rows resident, probes on the host pool) | 1024 (1024 queries) | 37438.0 (36530.8..37563.6) | 26.825 (26.576..27.328) | 26.825 (26.576..27.328) | 5 | load 112, avail 24.7 GiB / load 112, avail 24.7 GiB | 0.9762
|   (forge_ivf_l2_part launches 6) | | | | | | | |
| GPU ivf nlist 7588 nprobe 32, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 64 (256 queries) | 78.9 (71.2..96.1) | 812.500 (670.647..893.170) | 862.014 (710.978..1004.591) | 5 | load 112, avail 24.7 GiB / load 98, avail 26.3 GiB | 0.9750
|   (device matmul products over the calls of the row above, warm-up included: 32710) | | | | | | | |
| GPU ivf nlist 7588 nprobe 32, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 1024 (1024 queries) | 327.0 (313.3..333.7) | 3130.635 (3067.608..3266.986) | 3130.635 (3067.608..3266.986) | 5 | load 98, avail 26.3 GiB / load 91, avail 25.6 GiB | 0.9762
|   (device matmul products over the calls of the row above, warm-up included: 36876) | | | | | | | |
| CPU ivf nlist 7588 nprobe 64, 1 thread | 1 (200 queries) | 241.4 (204.3..271.9) | 3.917 (3.457..4.732) | 7.418 (6.786..8.926) | 5 | load 91, avail 25.6 GiB / load 91, avail 25.7 GiB | 0.9945
| CPU ivf nlist 7588 nprobe 64, one shard per pinned core (60) | 1 (200 queries) | 88.5 (84.5..88.9) | 11.942 (11.892..11.978) | 16.150 (16.054..16.186) | 5 | load 91, avail 25.7 GiB / load 88, avail 24.6 GiB | 0.9945
| CPU ivf nlist 7588 nprobe 64, one shard per pinned core (60) | 64 (1024 queries) | 1294.9 (1229.3..1399.0) | 48.535 (45.335..51.449) | 59.489 (50.990..62.567) | 5 | load 88, avail 24.6 GiB / load 92, avail 24.1 GiB | 0.9940
| CPU ivf nlist 7588 nprobe 64, one shard per pinned core (60) | 1024 (1024 queries) | 1843.3 (1773.2..1910.2) | 554.696 (534.695..576.215) | 554.696 (534.695..576.215) | 5 | load 92, avail 24.1 GiB / load 92, avail 24.0 GiB | 0.9940
| GPU ivf nlist 7588 nprobe 64: upload of the 3600000 rows sorted by list (once) | | 256.5 ms | | | 1 | load 92, avail 24.0 GiB | 1.80 GB/s |
| GPU ivf nlist 7588 nprobe 64, scan kernel (new; rows resident, probes on the host pool) | 1 (200 queries) | 943.1 (358.5..997.4) | 1.020 (0.971..2.441) | 1.427 (1.281..6.077) | 5 | load 92, avail 24.0 GiB / load 92, avail 24.0 GiB | 0.9945
|   (forge_ivf_l2_part launches 1001) | | | | | | | |
| GPU ivf nlist 7588 nprobe 64, scan kernel (new; rows resident, probes on the host pool) | 64 (1024 queries) | 10845.2 (10080.7..12037.9) | 5.964 (5.143..6.069) | 6.867 (6.139..10.028) | 5 | load 92, avail 24.0 GiB / load 92, avail 23.9 GiB | 0.9940
|   (forge_ivf_l2_part launches 81) | | | | | | | |
| GPU ivf nlist 7588 nprobe 64, scan kernel (new; rows resident, probes on the host pool) | 1024 (1024 queries) | 27270.0 (26766.8..27673.4) | 37.124 (36.599..37.633) | 37.124 (36.599..37.633) | 5 | load 92, avail 23.9 GiB / load 92, avail 23.9 GiB | 0.9940
|   (forge_ivf_l2_part launches 6) | | | | | | | |
| GPU ivf nlist 7588 nprobe 64, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 64 (256 queries) | 59.5 (57.4..62.2) | 1087.718 (1048.582..1119.356) | 1107.164 (1093.485..1149.113) | 5 | load 92, avail 23.9 GiB / load 84, avail 25.2 GiB | 0.9941
|   (device matmul products over the calls of the row above, warm-up included: 52943) | | | | | | | |
| GPU ivf nlist 7588 nprobe 64, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 1024 (1024 queries) | 254.0 (246.6..262.0) | 4029.800 (3906.696..4150.694) | 4029.800 (3906.696..4150.694) | 5 | load 84, avail 25.2 GiB / load 75, avail 27.9 GiB | 0.9940
|   (device matmul products over the calls of the row above, warm-up included: 40806) | | | | | | | |


rss at end 3842 MiB, load 75, avail 29.0 GiB
