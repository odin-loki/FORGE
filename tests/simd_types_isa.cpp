// vec_types.hpp checks for the Tier P backend this file is compiled for (tests/simd_types_check.hpp). CMake builds
// one executable per kernel table from this file (with that table's flags) and tests/simd_types_main.cpp
// (baseline flags), which asks the dispatcher whether the CPU can run the table before calling in here.
// The x86 AVX tables are named by FORGE_TYPES_AVX2 / FORGE_TYPES_AVX512; every other backend is read from the
// compile flags as in simd/src/isa_extra.cpp.
#include <forge/simd/vec.hpp>
#if defined(FORGE_TYPES_AVX512)
#define FORGE_ISA_TG forge::simd::x86_64_v4
#define FORGE_ISA_NAME "avx512"
static_assert(forge::simd::vec<float, FORGE_ISA_TG>::K == 2, "compile with the avx512 table's flags");
#elif defined(FORGE_TYPES_AVX2)
#define FORGE_ISA_TG forge::simd::x86_64_v3
#define FORGE_ISA_NAME "avx2"
static_assert(forge::simd::vec<float, FORGE_ISA_TG>::K == 1, "compile with the avx2 table's flags");
#else
#include "../simd/src/isa_target.inc"
#endif
#include "simd_types_check.hpp"
#if defined(FORGE_GN_ARM32)
#include "simd_arm32_ieee.hpp"   // the Armv7 backend's subnormal lanes against VFP arithmetic
#endif

extern "C" const char* forge_types_table() { return FORGE_ISA_NAME; }
extern "C" int forge_types_bits() { return forge::simd::vec<float, FORGE_ISA_TG>::N * 32; }
extern "C" int forge_types_run() {
    int fails = forge_types::check<FORGE_ISA_TG>(FORGE_ISA_NAME);
#if defined(FORGE_GN_ARM32)
    fails += forge_arm32::ieee_check();
#endif
    return fails;
}
