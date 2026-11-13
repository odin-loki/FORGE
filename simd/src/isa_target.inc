// Which Tier P backend the compile flags of this translation unit select, for the per-ISA table source
// (isa_extra.cpp). Defines
//   FORGE_ISA_NS     the namespace of that ISA's tables (the one kernels_<isa>.cpp / kernels_gn.cpp export into)
//   FORGE_ISA_TG     the target type (caps.hpp)
//   FORGE_ISA_NAME   the table name ("neon", "sve", "rvv", "sse4.2", "lasx", ...)
//   FORGE_ISA_KIND   the Tier P backend number (vec.hpp impl::kind)
//   FORGE_ISA_SOFT_FMA  when the ISA has no fused multiply-add instruction (SSE, WebAssembly)
// The scalable-vector tables are per length (mv_sve256, mv_rvv512, ...); the SSE tables are told apart by
// FORGE_SIMD_TABLE_SSE42 as in kernels_gn.cpp.
#include <forge/simd/vec.hpp>
#define FORGE_ISA_CAT2(a, b) a##b
#define FORGE_ISA_CAT(a, b) FORGE_ISA_CAT2(a, b)
#if defined(FORGE_HAVE_SVE_FIXED)
#define FORGE_ISA_NS FORGE_ISA_CAT(mv_sve, __ARM_FEATURE_SVE_BITS)
#define FORGE_ISA_TG forge::simd::sve_vl<FORGE_HAVE_SVE_FIXED>
#define FORGE_ISA_NAME "sve"
#define FORGE_ISA_KIND 4
#elif defined(FORGE_HAVE_RVV_FIXED)
#define FORGE_ISA_NS FORGE_ISA_CAT(mv_rvv, __riscv_v_fixed_vlen)
#define FORGE_ISA_TG forge::simd::rvv_vl<FORGE_HAVE_RVV_FIXED>
#define FORGE_ISA_NAME "rvv"
#define FORGE_ISA_KIND 5
#elif defined(__aarch64__) && defined(__ARM_NEON)
#define FORGE_ISA_NS mv_neon
#define FORGE_ISA_TG forge::simd::apple_m
#define FORGE_ISA_NAME "neon"
#define FORGE_ISA_KIND 3
#elif defined(FORGE_HAVE_GN)
#define FORGE_ISA_KIND 6
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(FORGE_SIMD_TABLE_SSE42)
#define FORGE_ISA_NS mv_sse42
#define FORGE_ISA_TG forge::simd::x86_64_v2
#define FORGE_ISA_NAME "sse4.2"
#else
#define FORGE_ISA_NS mv_sse2
#define FORGE_ISA_TG forge::simd::x86_64_v1
#define FORGE_ISA_NAME "sse2"
#endif
#if !defined(__FMA__)
#define FORGE_ISA_SOFT_FMA 1   // no fused multiply-add instruction: Tier P's fma is not correctly rounded here
#endif
#elif defined(__loongarch_asx)
#define FORGE_ISA_NS mv_lasx
#define FORGE_ISA_TG forge::simd::loongarch_lasx
#define FORGE_ISA_NAME "lasx"
#elif defined(__loongarch_sx)
#define FORGE_ISA_NS mv_lsx
#define FORGE_ISA_TG forge::simd::loongarch_lsx
#define FORGE_ISA_NAME "lsx"
#elif defined(__powerpc64__)
#define FORGE_ISA_NS mv_vsx
#define FORGE_ISA_TG forge::simd::power8
#define FORGE_ISA_NAME "vsx"
#elif defined(__s390x__)
#define FORGE_ISA_NS mv_vxe
#define FORGE_ISA_TG forge::simd::s390x_z14
#define FORGE_ISA_NAME "vxe"
#elif defined(__wasm__)
#define FORGE_ISA_NS mv_simd128
#define FORGE_ISA_TG forge::simd::wasm_simd
#define FORGE_ISA_NAME "simd128"
#define FORGE_ISA_SOFT_FMA 1
#elif defined(__mips__)
#define FORGE_ISA_NS mv_msa
#define FORGE_ISA_TG forge::simd::mips_msa
#define FORGE_ISA_NAME "msa"
#else
#define FORGE_ISA_NS mv_neon
#define FORGE_ISA_TG forge::simd::armv7_neon
#define FORGE_ISA_NAME "neon"
#endif
#else
#error "isa_target.inc: the compile flags enable no Tier P backend (NEON, fixed-length SVE / RVV or a vec_gn.hpp ISA)"
#endif
static_assert(forge::simd::vec<float, FORGE_ISA_TG>::K == FORGE_ISA_KIND, "this TU must build the native Tier P backend");
