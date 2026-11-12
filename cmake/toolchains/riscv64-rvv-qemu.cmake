# Cross-build for RISC-V 64 and run ctest under qemu with V at VLEN=256 (exercises the `rvv` lane table).
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(FORGE_CROSS_CC riscv64-linux-gnu-gcc CACHE STRING "")
set(FORGE_CROSS_CXX riscv64-linux-gnu-g++ CACHE STRING "")
set(CMAKE_C_COMPILER ${FORGE_CROSS_CC})
set(CMAKE_CXX_COMPILER ${FORGE_CROSS_CXX})
# The emulated CPU of the ordinary tests; -DFORGE_QEMU_CPU=rv64,v=true,vlen=512 reruns them at another vector
# length without rebuilding (the lane_rvv_vl<N> tests set their own).
set(FORGE_QEMU_CPU "rv64,v=true,vlen=256" CACHE STRING "qemu -cpu model for tests")
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-riscv64;-cpu;${FORGE_QEMU_CPU})
