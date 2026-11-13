// Kernel table for the vec_gn.hpp backend this file is compiled for. CMake builds it once per table with that
// ISA's flags (-msse4.2, -mlsx, -mlasx, -mvsx, -mzvector, -msimd128, -mmsa, -mfpu=neon-vfpv4); the two x86
// tables are told apart by FORGE_SIMD_TABLE_SSE42 because a whole build with -march=native enables SSE4.1 in both.
#include <forge/simd/algo.hpp>
#include <forge/simd/qdot.hpp>
#include <forge/simd/dispatch.hpp>
#if !defined(FORGE_HAVE_GN)
#error "kernels_gn.cpp: the compile flags enable none of the vec_gn.hpp backends"
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(FORGE_SIMD_TABLE_SSE42)
#if !defined(FORGE_GN_SSE41)
#error "the sse4.2 table must be compiled with -msse4.2 (MSVC: FORGE_SIMD_SSE41=1)"
#endif
#define FORGE_GN_TABLE mv_sse42, x86_64_v2, "sse4.2"
using forge_gn_target = forge::simd::x86_64_v2;
#else
#define FORGE_GN_TABLE mv_sse2, x86_64_v1, "sse2"
using forge_gn_target = forge::simd::x86_64_v1;
#endif
#elif defined(__loongarch_asx)
#define FORGE_GN_TABLE mv_lasx, loongarch_lasx, "lasx"
using forge_gn_target = forge::simd::loongarch_lasx;
#elif defined(__loongarch_sx)
#define FORGE_GN_TABLE mv_lsx, loongarch_lsx, "lsx"
using forge_gn_target = forge::simd::loongarch_lsx;
#elif defined(__powerpc64__)
#define FORGE_GN_TABLE mv_vsx, power8, "vsx"
using forge_gn_target = forge::simd::power8;
#elif defined(__s390x__)
#define FORGE_GN_TABLE mv_vxe, s390x_z14, "vxe"
using forge_gn_target = forge::simd::s390x_z14;
#elif defined(__wasm__)
#define FORGE_GN_TABLE mv_simd128, wasm_simd, "simd128"
using forge_gn_target = forge::simd::wasm_simd;
#elif defined(__mips__)
#define FORGE_GN_TABLE mv_msa, mips_msa, "msa"
using forge_gn_target = forge::simd::mips_msa;
#else
#define FORGE_GN_TABLE mv_neon, armv7_neon, "neon"
using forge_gn_target = forge::simd::armv7_neon;
#endif
static_assert(forge::simd::vec<float, forge_gn_target>::K == 6 && forge::simd::vec<float, forge_gn_target>::N == forge::simd::gn::lanes,
              "this TU must build the native vec_gn.hpp kernels");
// FORGE_GN_EXPAND: MSVC's traditional preprocessor passes the expanded FORGE_GN_TABLE on as one argument unless
// the call is rescanned.
#define FORGE_GN_EXPAND(x) x
#define FORGE_GN_EXPORT(...) FORGE_GN_EXPAND(FORGE_SIMD_EXPORT_TABLE(__VA_ARGS__))
FORGE_GN_EXPORT(FORGE_GN_TABLE)
