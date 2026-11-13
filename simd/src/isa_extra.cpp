// The tables beside the kernel table for the Tier P backend this file is compiled for: Tier A algorithms
// (tier_a.hpp), the attention / delta-rule kernels (seq.hpp) and, where no hand-written integer matvec exists
// (everything except AArch64 and the x86 AVX tables), the int8 matvec (qdot_i8.hpp). CMake compiles it once per
// table with that table's flags plus -ffp-contract=off, next to kernels_neon.cpp / kernels_sve.cpp /
// kernels_rvv.cpp / kernels_gn.cpp; isa_target.inc reads the target from the flags. Nothing here is forked per
// ISA: the same templates (algo_*.hpp over vec<float, Tg>, seq_impl.hpp over a 16-lane vector built from
// vec<float, Tg>, qi8_gn.inc over GNU vector types) are instantiated for each.
//
// Exports forge::simd::<FORGE_ISA_NS>::extra (isa_tables.hpp), which dispatch.cpp hands to tier_a_by_name(),
// seq_by_name() and qi8_by_name().
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(__riscv_v) || defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86) || \
    defined(__loongarch_sx) || (defined(__powerpc64__) && defined(__VSX__)) || defined(__s390x__) || defined(__wasm_simd128__) || defined(__mips_msa) || \
    (defined(__arm__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA))
#include "isa_target.inc"
#include "isa_tables.hpp"

// ---- Tier A ----
#define FORGE_TIERA_NS FORGE_ISA_CAT(tiera_, FORGE_ISA_NS)
#define FORGE_TIERA_TG FORGE_ISA_TG
#define FORGE_TIERA_NAME FORGE_ISA_NAME
#include "tiera_impl.inc"

// ---- seq ----
#define FORGE_SEQ_NS FORGE_ISA_CAT(seq_, FORGE_ISA_NS)
#define FORGE_SEQ_TG FORGE_ISA_TG
#if defined(FORGE_ISA_SOFT_FMA)
#define FORGE_SEQ_SOFT_FMA 1
#endif
#include <forge/simd/seq_impl.hpp>
FORGE_SEQ_EXPORT_TABLE(FORGE_ISA_NAME)

// ---- int8 matvec: GNU vector types (not MSVC); AArch64 has its own NEON / dotprod / SVE kernels ----
#if defined(__GNUC__) && !defined(__aarch64__)
#define QI8_NS FORGE_ISA_CAT(gn_, FORGE_ISA_NS)
#include "qi8_gn.inc"
namespace forge::simd::FORGE_ISA_NS {
static const qi8_kernels qi8_table = {FORGE_ISA_NAME, &qi8::QI8_NS::quantize, &qi8::QI8_NS::matvec};
static const qi8_kernels* const qi8_ptr = &qi8_table;
}
#else
namespace forge::simd::FORGE_ISA_NS { static const qi8_kernels* const qi8_ptr = nullptr; }
#endif

namespace forge::simd::FORGE_ISA_NS {
extern const isa_tables extra;
const isa_tables extra = {FORGE_ISA_NAME, &FORGE_TIERA_NS::table, &FORGE_SEQ_NS::table, qi8_ptr};
}
#endif
