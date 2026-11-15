#!/bin/sh
# Keeps the compression plan (tools/compress_plan.cpp) running until it reports done: restarts it when it dies
# (crash, OOM kill, reboot of a long session, ...), and kills and restarts it when its heartbeat goes stale.
# Jobs resume from their checkpoints, finished jobs are skipped, so a restart loses little work.
#
#   tools/compress_plan_supervise.sh [PHASES]          e.g. nohup tools/compress_plan_supervise.sh & (POSIX sh)
#
# Environment: everything compress_plan.cpp reads (FORGE_BIN, COMPRESS_RUNS, COMPRESS_MEM_LIMIT, ...), and
#   COMPRESS_PLAN              the plan script (default: compress_plan.cpp next to this file)
#   COMPRESS_MAX_RESTARTS      give up after this many restarts (default 1000)
#   COMPRESS_RESTART_DELAY     seconds before a restart (default 60)
#   COMPRESS_HEARTBEAT_STALE   minutes without a heartbeat before the plan is considered hung (default 30; 0 = never)
#
# Exit status: 0 when the plan is done, 75 when another plan instance holds the lock, 1 when it gave up.
set -u
here=$(cd "$(dirname "$0")" && pwd)
FORGE_BIN=${FORGE_BIN:-forge}
PLAN=${COMPRESS_PLAN:-$here/compress_plan.cpp}
RUNS=${COMPRESS_RUNS:-/home/user/runs}
MAX_RESTARTS=${COMPRESS_MAX_RESTARTS:-1000}
DELAY=${COMPRESS_RESTART_DELAY:-60}
STALE=${COMPRESS_HEARTBEAT_STALE:-30}
export FORGE_BIN COMPRESS_RUNS="$RUNS"
mkdir -p "$RUNS" || exit 1
HB="$RUNS/plan.heartbeat"
LOG="$RUNS/supervise.log"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] supervise: $*" | tee -a "$LOG"; }

restarts=0
while :; do
    say "starting the plan (restart $restarts)"
    "$FORGE_BIN" script "$PLAN" "$@" &
    pid=$!
    trap 'kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; exit 130' INT TERM
    # watchdog: the plan rewrites its heartbeat at least every 30 s, also while a job runs (exit noticed within 5 s)
    while kill -0 "$pid" 2>/dev/null; do
        sleep 5
        if [ "$STALE" -gt 0 ] && [ -f "$HB" ] && [ -n "$(find "$HB" -mmin +"$STALE" 2>/dev/null)" ]; then
            say "heartbeat older than $STALE min: killing the plan (pid $pid)"
            kill "$pid" 2>/dev/null; sleep 10; kill -9 "$pid" 2>/dev/null
        fi
    done
    wait "$pid"
    rc=$?
    trap - INT TERM
    if [ "$rc" -eq 0 ] && [ -f "$RUNS/plan.done" ]; then say "the plan is done"; exit 0; fi
    if [ "$rc" -eq 75 ]; then say "another plan instance is running (lock $RUNS/plan.lock)"; exit 75; fi
    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then say "giving up after $MAX_RESTARTS restarts (last exit $rc)"; exit 1; fi
    say "the plan exited with status $rc; restarting in ${DELAY}s"
    sleep "$DELAY"
done
