#!/bin/sh
# Reproducible-build check: two clean Release builds in different directories (of different path length) must
# produce bit-identical forge, forge-serve and libforge_c. Usage: ci/repro_check.sh [cmake args...]
#   REPRO_JOBS=N    parallel compile jobs per build (default: ninja's)
#   REPRO_TMP=DIR   where the two build trees go (default /tmp)
# On a difference the trees are kept, and ./repro-diff/ gets, per differing file, the first differing byte
# offsets (cmp -l) and the two section tables (readelf -SW, when available) to show where they differ.
set -eu
src=$(cd "$(dirname "$0")/.." && pwd)
tmp=${REPRO_TMP:-/tmp}
a=$(mktemp -d "$tmp/forge-repro-a.XXXX"); b=$(mktemp -d "$tmp/forge-repro-bb.XXXX")
for d in "$a" "$b"; do
  cmake -S "$src" -B "$d" -G Ninja -DCMAKE_BUILD_TYPE=Release -DFORGE_BUILD_TESTS=OFF "$@" >/dev/null
  ninja -C "$d" ${REPRO_JOBS:+-j$REPRO_JOBS} forge forge-serve forge_c >/dev/null
done
rc=0
for f in forge forge-serve libforge.so; do
  ha=$(sha256sum "$a/$f" | cut -d' ' -f1); hb=$(sha256sum "$b/$f" | cut -d' ' -f1)
  if [ "$ha" = "$hb" ]; then echo "identical  $f  $ha"; else
    echo "DIFFERENT  $f  $ha  $hb"; rc=1
    mkdir -p repro-diff
    cmp -l "$a/$f" "$b/$f" 2>/dev/null | head -200 > "repro-diff/$f.cmp" || true
    if command -v readelf >/dev/null 2>&1; then readelf -SW "$a/$f" > "repro-diff/$f.a.sections" 2>/dev/null || true; readelf -SW "$b/$f" > "repro-diff/$f.b.sections" 2>/dev/null || true; fi
  fi
done
if [ $rc = 0 ]; then rm -rf "$a" "$b"; else echo "kept $a $b (details in ./repro-diff)"; fi
exit $rc
