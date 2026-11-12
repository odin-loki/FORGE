# Cross-build for MIPS64 little-endian and run ctest under qemu as a Loongson-3A4000 (MIPS64r2 with MSA;
# exercises the `msa` table). Compilers default to the GNU cross toolchain; override with
# -DFORGE_CROSS_CC/-DFORGE_CROSS_CXX (e.g. a wrapper running `zig c++ -target mips64el-linux-muslabi64`).
# Build with -DCMAKE_EXE_LINKER_FLAGS=-static.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR mips64el)
set(FORGE_CROSS_CC mips64el-linux-gnuabi64-gcc CACHE STRING "")
set(FORGE_CROSS_CXX mips64el-linux-gnuabi64-g++ CACHE STRING "")
set(CMAKE_C_COMPILER ${FORGE_CROSS_CC})
set(CMAKE_CXX_COMPILER ${FORGE_CROSS_CXX})
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-mips64el;-cpu;Loongson-3A4000)
