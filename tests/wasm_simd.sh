#!/bin/sh
# Build tests/test_simd.cpp, test_simd_tables.cpp, test_simd_tiera.cpp (and test_simd_types.cpp) and the simd
# tables for wasm32-wasi (scalar + SIMD128: kernels, Tier A, seq, int8) and run them.
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
OBJS=""
if [ "${WASM_SIMD:-1}" = 1 ]; then
  $CXX $FLAGS -msimd128 -ffp-contract=off -c "$R/simd/src/kernels_gn.cpp" -o "$OUT/kernels_gn.o"
  # the Tier A, seq and int8 tables of the simd128 backend
  $CXX $FLAGS -msimd128 -ffp-contract=off -c "$R/simd/src/isa_extra.cpp" -o "$OUT/isa_extra.o"
  DEFS="-DFORGE_SIMD_HAVE_WASM128=1"
  OBJS="$OUT/kernels_gn.o $OUT/isa_extra.o"
fi
for s in simd/src/kernels_scalar simd/src/qi8_dispatch simd/src/seq_dispatch simd/src/tiera_scalar simd/src/tiera_dispatch tests/wasm_no_exceptions; do
  $CXX $FLAGS -c "$R/$s.cpp" -o "$OUT/$(basename $s).o"
  OBJS="$OBJS $OUT/$(basename $s).o"
done
$CXX $FLAGS -ffp-contract=off -c "$R/simd/src/seq_scalar.cpp" -o "$OUT/seq_scalar.o"
$CXX $FLAGS $DEFS -c "$R/simd/src/dispatch.cpp" -o "$OUT/dispatch.o"
TESTS="test_simd test_simd_tables test_simd_tiera"
[ -f "$R/tests/test_simd_types.cpp" ] && TESTS="$TESTS test_simd_types"
for t in $TESTS; do
  $CXX $FLAGS -c "$R/tests/$t.cpp" -o "$OUT/$t.o"
  $CXX -O2 $OBJS "$OUT/$t.o" "$OUT/seq_scalar.o" "$OUT/dispatch.o" -o "$OUT/$t.wasm"
  echo "built $OUT/$t.wasm"
done
if [ "${WASM_SIMD:-1}" = 1 ] && [ -f "$R/tests/simd_types_isa.cpp" ]; then   # vec_types.hpp on SIMD128 registers (tests/simd_types_isa.cpp, as CMake's test_simd_types_<table>)
  $CXX $FLAGS -msimd128 -c "$R/tests/simd_types_isa.cpp" -o "$OUT/simd_types_isa.o"
  $CXX $FLAGS -c "$R/tests/simd_types_main.cpp" -o "$OUT/simd_types_main.o"
  $CXX -O2 $OBJS "$OUT/simd_types_isa.o" "$OUT/simd_types_main.o" "$OUT/seq_scalar.o" "$OUT/dispatch.o" -o "$OUT/test_simd_types_simd128.wasm"
  echo "built $OUT/test_simd_types_simd128.wasm"
  TESTS="$TESTS test_simd_types_simd128"
fi
rc=0
# --no-turbo-fast-api-calls: node 22.23's WASI fd_write runs as a V8 fast API call and allocates; once the module's
# memory is large that reports external memory pressure, a GC starts inside the fast call and node segfaults
# walking the stack (in a different test each run; test_simd_tiera's 1M-element sorts made it certain).
for t in $TESTS; do
  if [ -n "$WASM_RUN" ]; then
    (cd "$R" && $WASM_RUN "$OUT/$t.wasm") || rc=1
  else
    (cd "$R" && node --no-warnings --no-turbo-fast-api-calls "$R/tests/wasm_run.mjs" "$OUT/$t.wasm") || rc=1
  fi
done
exit $rc
