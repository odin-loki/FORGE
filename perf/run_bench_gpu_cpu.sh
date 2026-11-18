#!/bin/bash
# Reproduce docs/bench_gpu_vs_cpu.md (WSL, Linux). Builds the data set if missing, then runs bench_gpu_cpu and bench_ivf with
# nvidia-smi sampled alongside. Every heavy step waits for the shared-machine capacity guard when it exists.
#   BUILD=~/pbuild/forge-gpubench DATA=/dev/shm/gpubench OUT=/mnt/d/ParanoidBSD/wt/gpubench-work/results bash perf/run_bench_gpu_cpu.sh
# Build first: cmake -S . -B $BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release -DFORGE_BUILD_DB_BENCH=ON && ninja -C $BUILD bench_db bench_gpu_cpu bench_ivf
set -u
BUILD=${BUILD:-$HOME/pbuild/forge-gpubench}; DATA=${DATA:-/dev/shm/gpubench}; OUT=${OUT:-$PWD/bench-out}; REPS=${REPS:-5}
GUARD=${GUARD:-/mnt/d/ParanoidBSD/wt/capacity_ok.sh}
wait_capacity() { [ -f "$GUARD" ] || return 0; until bash "$GUARD"; do sleep 60; done; }
mkdir -p "$OUT" "$DATA"
# 3.6M rows (perf/bench_db's generator), 44 MiB budget: 60 shards, one per core on a 64-CPU machine
if [ ! -f "$DATA/db60/sharded/manifest.json" ]; then wait_capacity; "$BUILD/bench_db" --dir "$DATA/db60" --rows 3600000 --budget-mib 44 --no-single --build-only; fi
# the same rows at the default 10 MiB budget (255 shards), key-sorted, for the cluster-aligned comparison
if [ ! -f "$DATA/db255/sharded/manifest.json" ]; then wait_capacity; "$BUILD/bench_db" --dir "$DATA/db255" --rows 3600000 --no-single --build-only; fi
smi() { nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -lms 500 > "$OUT/$1.smi.csv" & echo $!; }
wait_capacity; uptime > "$OUT/bench_gpu_cpu.load"
S=$(smi bench_gpu_cpu); "$BUILD/bench_gpu_cpu" --dir "$DATA/db60/sharded" --reps "$REPS" > "$OUT/bench_gpu_cpu.md" 2> "$OUT/bench_gpu_cpu.err"; echo "bench_gpu_cpu rc=$?"; kill $S
uptime >> "$OUT/bench_gpu_cpu.load"
wait_capacity; uptime > "$OUT/bench_ivf.load"
S=$(smi bench_ivf); "$BUILD/bench_ivf" --dir "$DATA/db60/sharded" --keysorted "$DATA/db255/sharded" --cluster-out "$DATA/cluster" --reps "$REPS" > "$OUT/bench_ivf.md" 2> "$OUT/bench_ivf.err"; echo "bench_ivf rc=$?"; kill $S
uptime >> "$OUT/bench_ivf.load"
# the same IVF arm on clustered data: the stored vectors replaced by a 4096-centre Gaussian mixture (same ids and shards)
wait_capacity; S=$(smi bench_ivf_mixture); "$BUILD/bench_ivf" --dir "$DATA/db60/sharded" --mixture 4096 --sections ivf --reps "$REPS" > "$OUT/bench_ivf_mixture.md" 2> "$OUT/bench_ivf_mixture.err"; echo "bench_ivf mixture rc=$?"; kill $S
echo ALL-BENCH-DONE
