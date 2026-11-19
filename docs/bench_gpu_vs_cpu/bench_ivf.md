# bench_ivf: 3600000 vectors (dim 32, as stored: i.i.d. N(0,1)) from /dev/shm/gpubench/db60/sharded (60 shards), 1024 queries, pinned workers 60, reps 5, load 76, avail 27.3 GiB, rss 1274 MiB

exact answers: gpu_knn_l2 0.42 s (0 of 1024 queries on the CPU path, device yes)

## IVF (ann::ivf_flat, new): k-means on 180000 sample rows (5%), 20 Lloyd iterations, task::pool(60)

| method | batch | queries/s | p50 ms | p99 ms | reps | machine before / after | recall@10 |
|---|---|---|---|---|---|---|---|
| baselines: per-shard flat + HNSW built on their workers (one-off): 169.4 s | | | | | 1 | load 110, avail 28.0 GiB | |
| CPU HNSW per shard ef 16 (recall sweep, 256 queries) | | | | | | | 0.8191 |
| CPU HNSW per shard ef 32 (recall sweep, 256 queries) | | | | | | | 0.9219 |
| CPU HNSW per shard ef 64 (recall sweep, 256 queries) | | | | | | | 0.9730 |
| CPU HNSW per shard ef 128 (recall sweep, 256 queries) | | | | | | | 0.9918 |
| CPU HNSW per shard ef 256 (recall sweep, 256 queries) | | | | | | | 0.9988 |
| CPU HNSW per shard ef 512 (recall sweep, 256 queries) | | | | | | | 1.0000 |
| CPU HNSW per shard ef 64 (first ef reaching the target), one shard per pinned core (60) | 1 (200 queries) | 132.2 (131.3..136.9) | 7.867 (7.808..7.920) | 9.357 (8.750..10.848) | 5 | load 112, avail 27.7 GiB / load 112, avail 28.7 GiB | 0.9735
| CPU HNSW per shard ef 64 (first ef reaching the target), one shard per pinned core (60) | 64 (1024 queries) | 448.6 (438.2..456.7) | 133.384 (132.338..140.757) | 192.967 (165.766..235.271) | 5 | load 112, avail 28.7 GiB / load 111, avail 28.7 GiB | 0.9747
| CPU HNSW per shard ef 64 (first ef reaching the target), one shard per pinned core (60) | 1024 (1024 queries) | 484.2 (463.2..506.3) | 2114.077 (2022.030..2209.873) | 2114.077 (2022.030..2209.873) | 5 | load 111, avail 28.7 GiB / load 108, avail 27.7 GiB | 0.9747
| CPU flat (exact), one shard per pinned core (60) | 1024 (1024 queries) | 34.5 (34.1..36.0) | 29668.955 (28472.262..30049.974) | 29668.955 (28472.262..30049.974) | 5 | load 108, avail 27.7 GiB / load 117, avail 27.4 GiB | 1.0000
| GPU gpu_knn_l2 (exact, resident) | 1 (200 queries) | 500.7 (492.4..512.6) | 1.989 (1.946..2.031) | 2.125 (2.018..2.221) | 5 | load 117, avail 28.2 GiB / load 113, avail 28.9 GiB | 1.0000
| GPU gpu_knn_l2 (exact, resident) | 64 (1024 queries) | 4414.8 (4412.5..4440.8) | 14.440 (14.383..14.469) | 14.672 (14.604..14.832) | 5 | load 113, avail 28.9 GiB / load 113, avail 28.3 GiB | 1.0000
| GPU gpu_knn_l2 (exact, resident) | 1024 (1024 queries) | 6301.9 (6289.0..6336.2) | 162.072 (161.161..162.324) | 162.072 (161.161..162.324) | 5 | load 113, avail 28.4 GiB / load 113, avail 28.8 GiB | 1.0000
| training nlist 1897 (one-off): k-means 9.30 (8.78..10.06) s | | | | | 5 | load 110, avail 28.9 GiB / load 107, avail 24.6 GiB | |
| index build nlist 1897 (one-off): assign 3600000 rows 10.31 s; list sizes min 3, mean 1898, max 2291 | | | | | 1 | load 101, avail 22.8 GiB | |
| ivf nlist 1897 nprobe 1 (recall sweep) | | | | | | rows scanned per query 1934 (0.05% of N) | 0.0854 |
| ivf nlist 1897 nprobe 2 (recall sweep) | | | | | | rows scanned per query 3859 (0.11% of N) | 0.1378 |
| ivf nlist 1897 nprobe 4 (recall sweep) | | | | | | rows scanned per query 7709 (0.21% of N) | 0.2158 |
| ivf nlist 1897 nprobe 8 (recall sweep) | | | | | | rows scanned per query 15444 (0.43% of N) | 0.3257 |
| ivf nlist 1897 nprobe 16 (recall sweep) | | | | | | rows scanned per query 30934 (0.86% of N) | 0.4569 |
| ivf nlist 1897 nprobe 32 (recall sweep) | | | | | | rows scanned per query 61880 (1.72% of N) | 0.6066 |
| ivf nlist 1897 nprobe 64 (recall sweep) | | | | | | rows scanned per query 123808 (3.44% of N) | 0.7581 |
| ivf nlist 1897 nprobe 128 (recall sweep) | | | | | | rows scanned per query 247549 (6.88% of N) | 0.8865 |
| ivf nlist 1897 nprobe 256 (recall sweep) | | | | | | rows scanned per query 494894 (13.75% of N) | 0.9642 |
| ivf nlist 1897 nprobe 512 (recall sweep) | | | | | | rows scanned per query 989345 (27.48% of N) | 0.9952 |
| ivf nlist 1897 nprobe 1024 (recall sweep) | | | | | | rows scanned per query 1976028 (54.89% of N) | 1.0000 |
| CPU ivf nlist 1897 nprobe 512, 1 thread | 1 (200 queries) | 14.6 (14.4..19.2) | 66.516 (51.730..69.056) | 87.699 (65.756..103.707) | 5 | load 116, avail 23.2 GiB / load 89, avail 23.6 GiB | 0.9945
| CPU ivf nlist 1897 nprobe 512, one shard per pinned core (60) | 1 (200 queries) | 62.6 (62.0..63.4) | 15.909 (15.781..16.015) | 20.819 (20.773..23.090) | 5 | load 89, avail 23.6 GiB / load 92, avail 23.2 GiB | 0.9945
| CPU ivf nlist 1897 nprobe 512, one shard per pinned core (60) | 64 (1024 queries) | 182.8 (174.8..183.3) | 352.484 (348.612..366.862) | 373.707 (365.528..390.192) | 5 | load 92, avail 23.2 GiB / load 103, avail 25.2 GiB | 0.9952
| CPU ivf nlist 1897 nprobe 512, one shard per pinned core (60) | 1024 (1024 queries) | 210.9 (202.7..216.1) | 4854.731 (4738.377..5051.846) | 4854.731 (4738.377..5051.846) | 5 | load 103, avail 25.2 GiB / load 112, avail 24.7 GiB | 0.9952
| GPU ivf nlist 1897 nprobe 512: upload of the 3600000 rows sorted by list (once) | | 243.3 ms | | | 1 | load 112, avail 24.6 GiB | 1.89 GB/s |
| GPU ivf nlist 1897 nprobe 512, scan kernel (new; rows resident, probes on the host pool) | 1 (200 queries) | 1208.9 (933.0..1300.3) | 0.826 (0.760..0.871) | 0.971 (0.859..7.370) | 5 | load 112, avail 24.6 GiB / load 112, avail 24.6 GiB | 0.9945
|   (forge_ivf_l2_part launches 1001) | | | | | | | |
| GPU ivf nlist 1897 nprobe 512, scan kernel (new; rows resident, probes on the host pool) | 64 (1024 queries) | 3481.6 (3443.6..3519.9) | 18.317 (18.036..18.466) | 19.996 (19.106..20.685) | 5 | load 112, avail 24.6 GiB / load 114, avail 24.4 GiB | 0.9952
|   (forge_ivf_l2_part launches 81) | | | | | | | |
| GPU ivf nlist 1897 nprobe 512, scan kernel (new; rows resident, probes on the host pool) | 1024 (1024 queries) | 4035.2 (4016.2..4066.1) | 253.229 (250.963..254.624) | 253.229 (250.963..254.624) | 5 | load 114, avail 24.4 GiB / load 114, avail 24.5 GiB | 0.9952
|   (forge_ivf_l2_part launches 6) | | | | | | | |
| GPU ivf nlist 1897 nprobe 512, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 64 (256 queries) | 28.5 (27.8..28.8) | 2233.363 (2195.131..2307.754) | 2379.358 (2298.037..2473.679) | 5 | load 114, avail 24.5 GiB / load 93, avail 23.3 GiB | 0.9945
|   (device matmul products over the calls of the row above, warm-up included: 39363) | | | | | | | |
| GPU ivf nlist 1897 nprobe 512, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 1024 (1024 queries) | 39.2 (35.4..45.7) | 26073.461 (22428.590..28909.645) | 26073.461 (22428.590..28909.645) | 5 | load 93, avail 23.3 GiB / load 82, avail 24.3 GiB | 0.9952
|   (device matmul products over the calls of the row above, warm-up included: 11304) | | | | | | | |
| CPU ivf nlist 1897 nprobe 512, 1 thread | 1 (200 queries) | 19.6 (18.5..21.6) | 50.653 (45.299..54.084) | 65.350 (62.329..75.817) | 5 | load 82, avail 24.3 GiB / load 79, avail 25.2 GiB | 0.9945
| CPU ivf nlist 1897 nprobe 512, one shard per pinned core (60) | 1 (200 queries) | 62.3 (61.8..62.7) | 15.988 (15.939..16.043) | 20.720 (20.120..21.298) | 5 | load 79, avail 25.2 GiB / load 81, avail 25.1 GiB | 0.9945
| CPU ivf nlist 1897 nprobe 512, one shard per pinned core (60) | 64 (1024 queries) | 187.1 (182.9..188.0) | 341.343 (338.214..350.986) | 364.405 (355.047..366.381) | 5 | load 81, avail 25.1 GiB / load 92, avail 24.6 GiB | 0.9952
| CPU ivf nlist 1897 nprobe 512, one shard per pinned core (60) | 1024 (1024 queries) | 181.0 (178.5..193.3) | 5656.464 (5295.283..5736.431) | 5656.464 (5295.283..5736.431) | 5 | load 92, avail 24.6 GiB / load 101, avail 25.2 GiB | 0.9952
| GPU ivf nlist 1897 nprobe 512: upload of the 3600000 rows sorted by list (once) | | 316.8 ms | | | 1 | load 101, avail 25.1 GiB | 1.45 GB/s |
| GPU ivf nlist 1897 nprobe 512, scan kernel (new; rows resident, probes on the host pool) | 1 (200 queries) | 1042.4 (302.9..1066.2) | 0.942 (0.915..3.614) | 1.255 (1.177..7.676) | 5 | load 101, avail 25.1 GiB / load 99, avail 25.3 GiB | 0.9945
|   (forge_ivf_l2_part launches 1001) | | | | | | | |
| GPU ivf nlist 1897 nprobe 512, scan kernel (new; rows resident, probes on the host pool) | 64 (1024 queries) | 3410.6 (3347.4..3494.7) | 18.795 (18.272..18.929) | 19.930 (19.022..21.832) | 5 | load 99, avail 25.3 GiB / load 99, avail 25.3 GiB | 0.9952
|   (forge_ivf_l2_part launches 81) | | | | | | | |
| GPU ivf nlist 1897 nprobe 512, scan kernel (new; rows resident, probes on the host pool) | 1024 (1024 queries) | 4050.6 (4015.5..4061.1) | 252.261 (251.566..254.426) | 252.261 (251.566..254.426) | 5 | load 99, avail 25.3 GiB / load 99, avail 25.7 GiB | 0.9952
|   (forge_ivf_l2_part launches 6) | | | | | | | |
| GPU ivf nlist 1897 nprobe 512, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 64 (256 queries) | 25.4 (24.5..26.7) | 2558.326 (2375.020..2597.493) | 2607.898 (2562.137..2761.932) | 5 | load 99, avail 25.7 GiB / load 88, avail 25.0 GiB | 0.9945
|   (device matmul products over the calls of the row above, warm-up included: 39363) | | | | | | | |
| GPU ivf nlist 1897 nprobe 512, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 1024 (1024 queries) | 40.3 (37.9..41.7) | 25411.364 (24526.898..27009.695) | 25411.364 (24526.898..27009.695) | 5 | load 88, avail 25.0 GiB / load 80, avail 24.9 GiB | 0.9952
|   (device matmul products over the calls of the row above, warm-up included: 11304) | | | | | | | |
| training nlist 7588 (one-off): k-means 42.43 (36.36..44.69) s | | | | | 5 | load 80, avail 25.8 GiB / load 101, avail 26.8 GiB | |
| index build nlist 7588 (one-off): assign 3600000 rows 27.06 s; list sizes min 1, mean 474, max 883 | | | | | 1 | load 105, avail 25.4 GiB | |
| ivf nlist 7588 nprobe 1 (recall sweep) | | | | | | rows scanned per query 559 (0.02% of N) | 0.0627 |
| ivf nlist 7588 nprobe 2 (recall sweep) | | | | | | rows scanned per query 1127 (0.03% of N) | 0.1021 |
| ivf nlist 7588 nprobe 4 (recall sweep) | | | | | | rows scanned per query 2264 (0.06% of N) | 0.1611 |
| ivf nlist 7588 nprobe 8 (recall sweep) | | | | | | rows scanned per query 4544 (0.13% of N) | 0.2365 |
| ivf nlist 7588 nprobe 16 (recall sweep) | | | | | | rows scanned per query 9136 (0.25% of N) | 0.3412 |
| ivf nlist 7588 nprobe 32 (recall sweep) | | | | | | rows scanned per query 18351 (0.51% of N) | 0.4627 |
| ivf nlist 7588 nprobe 64 (recall sweep) | | | | | | rows scanned per query 36753 (1.02% of N) | 0.6033 |
| ivf nlist 7588 nprobe 128 (recall sweep) | | | | | | rows scanned per query 73602 (2.04% of N) | 0.7432 |
| ivf nlist 7588 nprobe 256 (recall sweep) | | | | | | rows scanned per query 147198 (4.09% of N) | 0.8663 |
| ivf nlist 7588 nprobe 512 (recall sweep) | | | | | | rows scanned per query 293926 (8.16% of N) | 0.9523 |
| ivf nlist 7588 nprobe 1024 (recall sweep) | | | | | | rows scanned per query 585117 (16.25% of N) | 0.9904 |
| ivf nlist 7588 nprobe 2048 (recall sweep) | | | | | | rows scanned per query 1159464 (32.21% of N) | 0.9987 |
| CPU ivf nlist 7588 nprobe 1024, 1 thread | 1 (200 queries) | 34.0 (32.3..39.8) | 28.772 (25.486..31.190) | 38.875 (31.074..40.471) | 5 | load 112, avail 26.7 GiB / load 96, avail 23.3 GiB | 0.9905
| CPU ivf nlist 7588 nprobe 1024, one shard per pinned core (60) | 1 (200 queries) | 66.6 (64.5..67.6) | 15.118 (14.631..15.710) | 20.067 (19.718..22.034) | 5 | load 96, avail 23.3 GiB / load 94, avail 23.6 GiB | 0.9905
| CPU ivf nlist 7588 nprobe 1024, one shard per pinned core (60) | 64 (1024 queries) | 217.9 (211.0..222.6) | 292.378 (286.363..304.085) | 317.386 (306.544..343.940) | 5 | load 94, avail 23.6 GiB / load 98, avail 25.8 GiB | 0.9904
| CPU ivf nlist 7588 nprobe 1024, one shard per pinned core (60) | 1024 (1024 queries) | 227.3 (221.7..232.2) | 4504.176 (4408.598..4618.453) | 4504.176 (4408.598..4618.453) | 5 | load 98, avail 25.8 GiB / load 104, avail 24.1 GiB | 0.9904
| GPU ivf nlist 7588 nprobe 1024: upload of the 3600000 rows sorted by list (once) | | 256.8 ms | | | 1 | load 104, avail 24.1 GiB | 1.79 GB/s |
| GPU ivf nlist 7588 nprobe 1024, scan kernel (new; rows resident, probes on the host pool) | 1 (200 queries) | 708.2 (271.1..733.1) | 1.411 (1.361..3.640) | 1.752 (1.516..6.404) | 5 | load 104, avail 24.1 GiB / load 104, avail 23.7 GiB | 0.9905
|   (forge_ivf_l2_part launches 1001) | | | | | | | |
| GPU ivf nlist 7588 nprobe 1024, scan kernel (new; rows resident, probes on the host pool) | 64 (1024 queries) | 4123.8 (3546.8..4173.2) | 15.407 (15.238..18.127) | 16.128 (15.867..20.693) | 5 | load 104, avail 23.7 GiB / load 104, avail 23.4 GiB | 0.9904
|   (forge_ivf_l2_part launches 81) | | | | | | | |
| GPU ivf nlist 7588 nprobe 1024, scan kernel (new; rows resident, probes on the host pool) | 1024 (1024 queries) | 5019.9 (4916.3..5161.2) | 203.282 (197.818..207.633) | 203.282 (197.818..207.633) | 5 | load 104, avail 23.4 GiB / load 103, avail 23.2 GiB | 0.9904
|   (forge_ivf_l2_part launches 6) | | | | | | | |
| GPU ivf nlist 7588 nprobe 1024, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 64 (256 queries) | 18.6 (17.7..19.2) | 3440.221 (3391.868..3621.420) | 3557.959 (3443.330..3666.064) | 5 | load 103, avail 23.2 GiB / load 77, avail 25.6 GiB | 0.9910
|   (device matmul products over the calls of the row above, warm-up included: 145560) | | | | | | | |
| GPU ivf nlist 7588 nprobe 1024, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 1024 (1024 queries) | 33.3 (32.4..37.2) | 30709.961 (27499.398..31607.369) | 30709.961 (27499.398..31607.369) | 5 | load 77, avail 25.6 GiB / load 77, avail 24.8 GiB | 0.9904
|   (device matmul products over the calls of the row above, warm-up included: 44514) | | | | | | | |
| CPU ivf nlist 7588 nprobe 1024, 1 thread | 1 (200 queries) | 30.4 (26.3..32.4) | 32.989 (30.061..37.315) | 40.297 (39.140..51.877) | 5 | load 77, avail 24.8 GiB / load 77, avail 23.6 GiB | 0.9905
| CPU ivf nlist 7588 nprobe 1024, one shard per pinned core (60) | 1 (200 queries) | 58.8 (57.2..68.6) | 14.884 (14.339..15.827) | 24.831 (20.257..51.267) | 5 | load 77, avail 23.6 GiB / load 83, avail 27.6 GiB | 0.9905
| CPU ivf nlist 7588 nprobe 1024, one shard per pinned core (60) | 64 (1024 queries) | 225.2 (205.1..241.9) | 271.303 (257.887..285.899) | 423.307 (315.186..504.506) | 5 | load 83, avail 27.6 GiB / load 95, avail 27.3 GiB | 0.9904
| CPU ivf nlist 7588 nprobe 1024, one shard per pinned core (60) | 1024 (1024 queries) | 247.2 (235.2..263.8) | 4140.544 (3880.131..4352.840) | 4140.544 (3880.131..4352.840) | 5 | load 95, avail 27.3 GiB / load 102, avail 25.3 GiB | 0.9904
| GPU ivf nlist 7588 nprobe 1024: upload of the 3600000 rows sorted by list (once) | | 518.7 ms | | | 1 | load 102, avail 25.3 GiB | 0.89 GB/s |
| GPU ivf nlist 7588 nprobe 1024, scan kernel (new; rows resident, probes on the host pool) | 1 (200 queries) | 618.0 (232.5..641.1) | 1.608 (1.564..3.827) | 1.968 (1.843..8.430) | 5 | load 102, avail 25.3 GiB / load 102, avail 25.0 GiB | 0.9905
|   (forge_ivf_l2_part launches 1001) | | | | | | | |
| GPU ivf nlist 7588 nprobe 1024, scan kernel (new; rows resident, probes on the host pool) | 64 (1024 queries) | 3659.7 (3207.3..3807.3) | 16.829 (16.402..18.969) | 22.713 (20.192..24.305) | 5 | load 102, avail 25.0 GiB / load 100, avail 24.9 GiB | 0.9904
|   (forge_ivf_l2_part launches 81) | | | | | | | |
| GPU ivf nlist 7588 nprobe 1024, scan kernel (new; rows resident, probes on the host pool) | 1024 (1024 queries) | 4882.1 (4733.0..4928.1) | 208.925 (207.103..215.896) | 208.925 (207.103..215.896) | 5 | load 100, avail 24.9 GiB / load 100, avail 24.8 GiB | 0.9904
|   (forge_ivf_l2_part launches 6) | | | | | | | |
| GPU ivf nlist 7588 nprobe 1024, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 64 (256 queries) | 18.2 (16.9..19.1) | 3534.347 (3414.358..3683.527) | 3577.827 (3511.069..4179.529) | 5 | load 100, avail 24.8 GiB / load 79, avail 24.2 GiB | 0.9910
|   (device matmul products over the calls of the row above, warm-up included: 145560) | | | | | | | |
| GPU ivf nlist 7588 nprobe 1024, gpu_flat_l2's matmul_t over the probed lists (one product per touched list) | 1024 (1024 queries) | 32.7 (32.1..34.9) | 31310.838 (29288.093..31807.279) | 31310.838 (29288.093..31807.279) | 5 | load 79, avail 24.2 GiB / load 82, avail 24.6 GiB | 0.9904
|   (device matmul products over the calls of the row above, warm-up included: 44514) | | | | | | | |

## cluster-aligned shards: nlist 1897, rows keyed (cluster, id), 10 MiB budget, centroids in manifest stats "ivf"

cluster-aligned set: 106 shards, 930.8 MiB (avg 8.78 MiB)

| method | handle | queries/s | p50 ms | p99 ms | reps | machine before / after | recall@10; shards opened / query; rows read / query; bytes read / query (est.) |
|---|---|---|---|---|---|---|---|
| cluster shards, nprobe 8, 1 thread (200 queries) | fresh | 63.0 (39.1..66.7) | 15.035 (13.453..22.851) | 37.442 (27.237..54.623) | 5 | load 82, avail 25.5 GiB / load 84, avail 24.9 GiB | 0.3265; 8.1 of 106; 15456; 4.00 MiB
| cluster shards, nprobe 8, 1 thread (200 queries) | warm | 74.0 (68.1..78.0) | 12.333 (11.488..13.688) | 31.774 (26.136..38.119) | 5 | load 82, avail 25.5 GiB / load 84, avail 24.9 GiB | 0.3265; 8.1 of 106; 15456; 4.00 MiB
| cluster shards, nprobe 8, 64 pinned workers (shard s on worker s % W) (200 queries) | fresh | 175.0 (158.0..185.3) | 5.149 (4.763..6.060) | 9.892 (8.951..12.747) | 5 | load 84, avail 24.9 GiB / load 85, avail 24.4 GiB | 0.3265; 8.1 of 106; 15456; 4.00 MiB
| cluster shards, nprobe 8, 64 pinned workers (shard s on worker s % W) (200 queries) | warm | 182.5 (175.7..184.9) | 4.786 (4.673..5.022) | 9.429 (8.952..9.571) | 5 | load 84, avail 24.9 GiB / load 85, avail 24.4 GiB | 0.3265; 8.1 of 106; 15456; 4.00 MiB
| cluster shards, nprobe 64, 1 thread (200 queries) | fresh | 7.8 (7.6..10.1) | 125.688 (96.779..129.785) | 212.577 (154.697..239.794) | 5 | load 85, avail 24.4 GiB / load 78, avail 25.0 GiB | 0.7585; 50.6 of 106; 123818; 32.01 MiB
| cluster shards, nprobe 64, 1 thread (200 queries) | warm | 8.6 (7.8..9.3) | 112.158 (103.018..125.707) | 192.894 (185.084..208.149) | 5 | load 85, avail 24.4 GiB / load 78, avail 25.0 GiB | 0.7585; 50.6 of 106; 123818; 32.01 MiB
| cluster shards, nprobe 64, 64 pinned workers (shard s on worker s % W) (200 queries) | fresh | 94.1 (84.4..97.6) | 9.802 (9.590..11.263) | 19.265 (16.025..20.608) | 5 | load 78, avail 25.0 GiB / load 82, avail 25.6 GiB | 0.7585; 50.6 of 106; 123818; 32.01 MiB
| cluster shards, nprobe 64, 64 pinned workers (shard s on worker s % W) (200 queries) | warm | 100.4 (89.1..102.5) | 9.412 (9.325..10.765) | 15.627 (15.111..18.366) | 5 | load 78, avail 25.0 GiB / load 82, avail 25.6 GiB | 0.7585; 50.6 of 106; 123818; 32.01 MiB
| cluster shards, nprobe 256, 1 thread (200 queries) | fresh | 2.0 (1.9..2.4) | 494.298 (403.434..505.419) | 791.666 (771.157..932.658) | 5 | load 82, avail 25.6 GiB / load 82, avail 19.1 GiB | 0.9585; 98.8 of 106; 494890; 127.96 MiB
| cluster shards, nprobe 256, 1 thread (200 queries) | warm | 2.0 (2.0..2.5) | 477.486 (412.073..501.716) | 725.004 (682.668..771.508) | 5 | load 82, avail 25.6 GiB / load 82, avail 19.1 GiB | 0.9585; 98.8 of 106; 494890; 127.96 MiB
| cluster shards, nprobe 256, 64 pinned workers (shard s on worker s % W) (200 queries) | fresh | 44.6 (43.6..48.2) | 21.740 (20.143..22.296) | 30.668 (27.909..30.884) | 5 | load 82, avail 19.1 GiB / load 93, avail 21.2 GiB | 0.9585; 98.8 of 106; 494890; 127.96 MiB
| cluster shards, nprobe 256, 64 pinned workers (shard s on worker s % W) (200 queries) | warm | 46.4 (43.8..48.2) | 21.058 (20.411..22.115) | 29.521 (26.546..36.070) | 5 | load 82, avail 19.1 GiB / load 93, avail 21.2 GiB | 0.9585; 98.8 of 106; 494890; 127.96 MiB
| cluster shards, nprobe 512, 1 thread (200 queries) | fresh | 1.0 (0.9..1.1) | 1004.087 (929.880..1143.304) | 1574.720 (1467.339..1681.210) | 5 | load 93, avail 21.2 GiB / load 78, avail 18.8 GiB | 0.9945; 105.5 of 106; 989398; 255.82 MiB
| cluster shards, nprobe 512, 1 thread (200 queries) | warm | 0.9 (0.9..1.0) | 1045.139 (968.984..1137.198) | 1647.509 (1336.734..1710.110) | 5 | load 93, avail 21.2 GiB / load 78, avail 18.8 GiB | 0.9945; 105.5 of 106; 989398; 255.82 MiB
| cluster shards, nprobe 512, 64 pinned workers (shard s on worker s % W) (200 queries) | fresh | 28.8 (24.1..29.4) | 33.858 (33.094..35.337) | 46.677 (40.890..129.678) | 5 | load 78, avail 18.8 GiB / load 105, avail 24.0 GiB | 0.9945; 105.5 of 106; 989398; 255.82 MiB
| cluster shards, nprobe 512, 64 pinned workers (shard s on worker s % W) (200 queries) | warm | 29.0 (27.2..29.3) | 33.776 (33.047..34.235) | 57.774 (39.809..95.285) | 5 | load 78, avail 18.8 GiB / load 105, avail 24.0 GiB | 0.9945; 105.5 of 106; 989398; 255.82 MiB
| key-sorted shards + shard_ann (HNSW per shard), ef 64, task::pool(64) (200 queries) | fresh | 45.6 (39.7..55.4) | 19.693 (16.456..23.256) | 28.953 (22.643..31.435) | 5 | load 105, avail 24.0 GiB / load 111, avail 19.7 GiB | 0.9975; 255 of 255 (every graph); graphs loaded on first use: 2242 MiB of shard files opened; first query of a fresh handle 313 (282..465) ms
| key-sorted shards + shard_ann (HNSW per shard), ef 64, task::pool(64) (200 queries) | warm | 50.2 (40.2..59.4) | 19.699 (16.635..24.592) | 26.175 (22.003..33.095) | 5 | load 105, avail 24.0 GiB / load 111, avail 19.7 GiB | 0.9975; 255 of 255 (every graph); graphs loaded on first use: 2242 MiB of shard files opened; first query of a fresh handle 313 (282..465) ms
| key-sorted shards + shard_ann (HNSW per shard), ef 160, task::pool(64) (200 queries) | fresh | 25.9 (23.3..27.6) | 36.815 (33.955..40.579) | 47.332 (40.709..60.388) | 5 | load 111, avail 19.7 GiB / load 110, avail 16.5 GiB | 1.0000; 255 of 255 (every graph); graphs loaded on first use: 2242 MiB of shard files opened; first query of a fresh handle 377 (357..388) ms
| key-sorted shards + shard_ann (HNSW per shard), ef 160, task::pool(64) (200 queries) | warm | 25.6 (23.5..28.1) | 38.830 (35.407..42.118) | 46.155 (42.176..55.992) | 5 | load 111, avail 19.7 GiB / load 110, avail 16.5 GiB | 1.0000; 255 of 255 (every graph); graphs loaded on first use: 2242 MiB of shard files opened; first query of a fresh handle 377 (357..388) ms

rss at end 4570 MiB, load 110, avail 16.5 GiB
