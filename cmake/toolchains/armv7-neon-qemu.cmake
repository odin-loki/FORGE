# Cross-build for 32-bit Arm (Armv7-A, hard float) and run ctest under qemu as a Cortex-A7 (NEON + VFPv4;
# exercises the 32-bit `neon` table). Compilers default to the GNU cross toolchain; override with
# -DFORGE_CROSS_CC/-DFORGE_CROSS_CXX (e.g. a wrapper running `zig c++ -target arm-linux-musleabihf`).
# Build with -DCMAKE_EXE_LINKER_FLAGS=-static. The baseline is left at the compiler's default: only the `neon`
# table's translation units are built with -mfpu=neon-vfpv4 / -mcpu=cortex_a7.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7)
set(FORGE_CROSS_CC arm-linux-gnueabihf-gcc CACHE STRING "")
set(FORGE_CROSS_CXX arm-linux-gnueabihf-g++ CACHE STRING "")
set(CMAKE_C_COMPILER ${FORGE_CROSS_CC})
set(CMAKE_CXX_COMPILER ${FORGE_CROSS_CXX})
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-arm;-cpu;cortex-a7)
