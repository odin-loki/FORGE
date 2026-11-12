# Cross-build for AArch64 and run ctest under qemu with 256-bit SVE (exercises the `sve` lane table).
# Compilers default to the GNU cross toolchain; override with -DFORGE_CROSS_CC/-DFORGE_CROSS_CXX
# (e.g. a wrapper running `zig c++ -target aarch64-linux-musl`). Build with -DCMAKE_EXE_LINKER_FLAGS=-static.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(FORGE_CROSS_CC aarch64-linux-gnu-gcc CACHE STRING "")
set(FORGE_CROSS_CXX aarch64-linux-gnu-g++ CACHE STRING "")
set(CMAKE_C_COMPILER ${FORGE_CROSS_CC})
set(CMAKE_CXX_COMPILER ${FORGE_CROSS_CXX})
# The emulated CPU of the ordinary tests; -DFORGE_QEMU_CPU=max,sve512=on reruns them at another vector length
# without rebuilding (the lane_sve_vl<N> / tile_sme_svl<N> tests set their own).
set(FORGE_QEMU_CPU "max,sve256=on" CACHE STRING "qemu -cpu model for tests")
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-aarch64;-cpu;${FORGE_QEMU_CPU})
