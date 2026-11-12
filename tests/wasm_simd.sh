#!/bin/sh
# Build tests/test_simd.cpp and the simd kernel tables for wasm32-wasi (scalar + SIMD128) and run them.
# The rest of FORGE is not ported to WebAssembly; this covers what is: the simd headers, the kernel tables
# and their dispatch (docs/simd.md, "WebAssembly").
#
#   CXX="zig c++ -target wasm32-wasi" tests/wasm_simd.sh [out-dir]      (or a wasi-sdk clang++)
#   WASM_RUN="wasmtime run --dir=."   overrides the runner (default: node's WASI, tests/wasm_run.mjs)
#   WASM_SIMD=0                       builds without -msimd128 (scalar table only)
#
# Needs the generated headers of a native build: GEN=build/dev/generated (default).
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$R/build/wasm}
GEN=${GEN:-$R/build/dev/generated}
CXX=${CXX:-zig c++ -target wasm32-wasi}
[ -d "$GEN" ] || { echo "wasm_simd.sh: $GEN missing (configure a native build first, or set GEN)"; exit 2; }
mkdir -p "$OUT"
INC=""
for d in tests inference/include modelfile/include graph/include ops/include core/include os/include simd/include gpu/include \
         support/include third_party security/include verify/include loadbalancer/include; do INC="$INC -I$R/$d"; done
FLAGS="-std=c++23 -O2 -DNDEBUG -DFORGE_SOURCE_DIR=\"$R\" $INC -I$GEN"
DEFS=""
if [ "${WASM_SIMD:-1}" = 1 ]; then
  $CXX $FLAGS -msimd128 -ffp-contract=off -c "$R/simd/src/kernels_gn.cpp" -o "$OUT/kernels_gn.o"
  DEFS="-DFORGE_SIMD_HAVE_WASM128=1"
  OBJS="$OUT/kernels_gn.o"
fi
for s in simd/src/kernels_scalar simd/src/qi8_dispatch simd/src/seq_dispatch tests/test_simd tests/wasm_no_exceptions; do
  $CXX $FLAGS -c "$R/$s.cpp" -o "$OUT/$(basename $s).o"
  OBJS="$OBJS $OUT/$(basename $s).o"
done
$CXX $FLAGS -ffp-contract=off -c "$R/simd/src/seq_scalar.cpp" -o "$OUT/seq_scalar.o"
$CXX $FLAGS $DEFS -c "$R/simd/src/dispatch.cpp" -o "$OUT/dispatch.o"
$CXX -O2 $OBJS "$OUT/seq_scalar.o" "$OUT/dispatch.o" -o "$OUT/test_simd.wasm"
echo "built $OUT/test_simd.wasm"
if [ -n "$WASM_RUN" ]; then
  (cd "$R" && $WASM_RUN "$OUT/test_simd.wasm")
else
  (cd "$R" && node --no-warnings "$R/tests/wasm_run.mjs" "$OUT/test_simd.wasm")
fi
