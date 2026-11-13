# Cross-build for ppc64le and run ctest under qemu as a POWER9 (exercises the `vsx` table, built for POWER8).
# Compilers default to the GNU cross toolchain; override with -DFORGE_CROSS_CC/-DFORGE_CROSS_CXX
# (e.g. a wrapper running `zig c++ -target powerpc64le-linux-musl`). Build with -DCMAKE_EXE_LINKER_FLAGS=-static.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ppc64le)
set(FORGE_CROSS_CC powerpc64le-linux-gnu-gcc CACHE STRING "")
set(FORGE_CROSS_CXX powerpc64le-linux-gnu-g++ CACHE STRING "")
set(CMAKE_C_COMPILER ${FORGE_CROSS_CC})
set(CMAKE_CXX_COMPILER ${FORGE_CROSS_CXX})
# POWER8 baseline: Ubuntu 24.04's cross g++ defaults to -mcpu=power9, whose xsmaxcdp/xsmincdp qemu 8.2 gets wrong for
# signed zeros (max(+0, -0) gives +0, hardware and C give -0), failing the bit-exact min/max/relu checks.
set(CMAKE_C_FLAGS_INIT "-mcpu=power8")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=power8")
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-ppc64le;-cpu;power9)
