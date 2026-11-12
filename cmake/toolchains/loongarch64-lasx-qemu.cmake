# Cross-build for LoongArch64 and run ctest under qemu (la464: LSX + LASX; exercises the `lasx` and `lsx` tables).
# Compilers default to the GNU cross toolchain; override with -DFORGE_CROSS_CC/-DFORGE_CROSS_CXX
# (e.g. a wrapper running `zig c++ -target loongarch64-linux-musl`). Build with -DCMAKE_EXE_LINKER_FLAGS=-static.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR loongarch64)
set(FORGE_CROSS_CC loongarch64-linux-gnu-gcc CACHE STRING "")
set(FORGE_CROSS_CXX loongarch64-linux-gnu-g++ CACHE STRING "")
set(CMAKE_C_COMPILER ${FORGE_CROSS_CC})
set(CMAKE_CXX_COMPILER ${FORGE_CROSS_CXX})
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-loongarch64;-cpu;la464)
