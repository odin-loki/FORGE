// Tier R: generated wrappers are feature-gated at compile time, on every architecture.
//
// Each section checks (a) that a wrapper is callable for a Target that has its feature and is NOT callable for
// one that lacks it (a requires-expression: the negative case is a compile-time fact, not a SIGILL), and
// (b) that a sample of wrappers computes the right thing, immediates (template parameters) included.
// The optional ISAs (SVE, RVV, LSX, z/Vector, MSA, SIMD128) are only present when this file is compiled for
// them; CMake builds such a variant next to the baseline one. Returns 77 (skipped) when the CPU lacks the ISA.
#include "check.hpp"
#include <forge/simd/raw.hpp>
#include <cstdint>
#define CHECKV(...) CHECK((__VA_ARGS__))   // wrapper calls carry template-argument commas
#if defined(__linux__) && !defined(__x86_64__)
#include <sys/auxv.h>
#endif

using namespace forge::simd;
namespace r = forge::simd::raw::native;

TEST(raw_isa_named) { CHECKV(forge::simd::raw::isa != nullptr); CHECKV(r::intrinsic_count >= 0); }

// ------------------------------------------------------------------------------------------------ x86
#if defined(__x86_64__) || defined(_M_X64)
#include <forge/simd/dispatch.hpp>
template <class T> concept can_avx512_add = requires(__m512 a) { r::avx512f::mm512_add_ps<T>(a, a); };
template <class T> concept can_avx2_add = requires(__m256i a) { r::avx2::mm256_add_epi32<T>(a, a); };
template <class T> concept can_vnni = requires(__m512i a) { r::avx512vnni::mm512_dpbusd_epi32<T>(a, a, a); };
template <class T> concept can_vl_bw = requires(__m256i a, __mmask32 k) { r::avx512bw::mm256_mask_add_epi8<T>(a, k, a, a); };
template <class T> concept can_sse2 = requires(__m128i a) { r::sse2::mm_add_epi32<T>(a, a); };
template <class T> concept can_gfni = requires(__m128i a) { r::gfni::mm_gf2p8mul_epi8<T>(a, a); };
static_assert(can_avx512_add<x86_64_v4> && !can_avx512_add<x86_64_v3>);
static_assert(can_avx2_add<x86_64_v3> && !can_avx2_add<scalar_t>);
static_assert(can_vnni<sapphire> && !can_vnni<x86_64_v4>);
static_assert(can_vl_bw<x86_64_v4> && !can_vl_bw<x86_64_v3>);                       // needs avx512bw AND avx512vl
static_assert(!can_vl_bw<caps<sse2, avx512f, avx512bw>>);
static_assert(can_sse2<x86_64_v2> && !can_sse2<scalar_t>);
static_assert(can_gfni<caps<sse2, gfni>> && !can_gfni<sapphire>);

TEST(raw_generated_count) { CHECKV(r::intrinsic_count > 5000); }
TEST(raw_sse2_executes) {
    if (!detect().sse2) return;
    __m128i c = r::sse2::mm_add_epi32<x86_64_v2>(r::sse2::mm_set1_epi32<x86_64_v2>(40), r::sse2::mm_set1_epi32<x86_64_v2>(2));
    CHECKV(r::sse2::mm_cvtsi128_si32<x86_64_v2>(c) == 42);
    __m128i s = r::sse2::mm_slli_epi32<x86_64_v2>(c, 1);
    CHECKV(r::sse2::mm_cvtsi128_si32<x86_64_v2>(s) == 84);
    CHECKV(r::sse2::mm_extract_epi16<2, x86_64_v2>(r::sse2::mm_set1_epi32<x86_64_v2>(0x00050007)) == 7);   // immediate = template argument
}
TEST(raw_avx2_executes) {
    if (!detect().avx2 || !detect().fma) return;
    __m256i a = _mm256_set1_epi32(20), b = _mm256_set1_epi32(22);
    __m256i c = r::avx2::mm256_add_epi32<x86_64_v3>(a, b);
    alignas(32) int out[8]; _mm256_store_si256((__m256i*)out, c);
    for (int v : out) CHECKV(v == 42);
    __m256 f = r::fma::mm256_fmadd_ps<x86_64_v3>(_mm256_set1_ps(2), _mm256_set1_ps(3), _mm256_set1_ps(1));
    CHECKV(_mm256_cvtss_f32(f) == 7.f);
    __m256i sh = r::avx2::mm256_shuffle_epi32<0x1b, x86_64_v3>(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    _mm256_store_si256((__m256i*)out, sh);
    CHECKV(out[0] == 3 && out[3] == 0 && out[4] == 7 && out[7] == 4);
    CHECKV(r::avx::mm256_movemask_ps<x86_64_v3>(_mm256_set1_ps(-1.f)) == 0xff);
}

// ------------------------------------------------------------------------------------------------ Arm
#elif defined(__aarch64__) || defined(_M_ARM64) || (defined(__arm__) && defined(__ARM_NEON))
template <class T> concept can_neon = requires(float32x4_t a) { r::neon::addq_f32<T>(a, a); };
static_assert(can_neon<apple_m> && can_neon<caps<neon>> && !can_neon<scalar_t>);
#if FORGE_RAW_A64
template <class T> concept can_dot = requires(int32x4_t a, int8x16_t b) { r::dotprod::dotq_s32<T>(a, b, b); };
template <class T> concept can_aes = requires(uint8x16_t a) { r::aes::aeseq_u8<T>(a, a); };
static_assert(can_dot<apple_m> && !can_dot<caps<neon>>);
static_assert(can_aes<caps<neon, arm_aes>> && !can_aes<apple_m>);
#endif
TEST(raw_generated_count) { CHECKV(r::intrinsic_count > 20000); }
TEST(raw_neon_executes) {
    using T = caps<neon>;
    float32x4_t s = r::neon::addq_f32<T>(r::neon::dupq_n_f32<T>(40.f), r::neon::dupq_n_f32<T>(2.f));
    CHECKV(r::neon::getq_lane_f32<3, T>(s) == 42.f);                                   // lane index = template argument
    int32x4_t h = r::neon::shrq_n_s32<2, T>(r::neon::dupq_n_s32<T>(168));
    CHECKV(r::neon::getq_lane_s32<0, T>(h) == 42);
    int32_t buf[4] = {1, 2, 3, 4};
    int32x4_t v = r::neon::ld1q_s32<T>(buf);
    v = r::neon::mulq_s32<T>(v, v); r::neon::st1q_s32<T>(buf, v);
    CHECKV(buf[0] == 1 && buf[3] == 16);
#if FORGE_RAW_A64
    CHECKV(r::neon::addvq_s32<T>(v) == 30);
#endif
}
#if FORGE_RAW_ARM_SVE
template <class T> concept can_sve = requires(svbool_t p, svfloat32_t a) { r::sve::add_f32_x<T>(p, a, a); };
template <class T> concept can_sve2 = requires(svint8_t a) { r::sve2::eor3_s8<T>(a, a, a); };
static_assert(can_sve<neoverse2> && !can_sve<apple_m>);
static_assert(can_sve2<neoverse2> && !can_sve2<caps<neon, sve>>);
static bool cpu_has_sve() {
#if defined(__linux__)
    return getauxval(AT_HWCAP) & (1ul << 22);
#else
    return false;
#endif
}
TEST(raw_sve_executes) {
    if (!cpu_has_sve()) { std::puts("  (no SVE on this CPU: skipped)"); std::exit(forge_test::failures() ? 1 : 77); }
    using T = caps<neon, sve>;
    svbool_t pg = r::sve::ptrue_b32<T>();
    svfloat32_t s = r::sve::add_f32_x<T>(pg, r::sve::dup_n_f32<T>(40.f), r::sve::dup_n_f32<T>(2.f));
    CHECKV(r::sve::lastb_f32<T>(pg, s) == 42.f);
    svint32_t o = r::sve::add_x<T>(pg, r::sve::dup_n_s32<T>(40), r::sve::dup_n_s32<T>(2));      // overloaded ACLE form
    CHECKV(r::sve::addv_s32<T>(pg, o) == int64_t(42) * int64_t(r::sve::cntw<T>()));
    CHECKV(r::sve::cntp_b8<T>(r::sve::ptrue_b8<T>(), r::sve::ptrue_pat_b8<SV_VL4, T>()) == 4);   // enum immediate
    CHECKV(r::sve::addv_s32<T>(pg, r::sve::asrd_n_s32_x<3, T>(pg, r::sve::dup_n_s32<T>(336))) == int64_t(42) * int64_t(r::sve::cntw<T>()));
}
#endif

// ------------------------------------------------------------------------------------------------ RISC-V
#elif defined(__riscv)
TEST(raw_generated_count) { CHECKV(r::intrinsic_count > 10000); }
#if defined(__riscv_vector)
template <class T> concept can_rvv = requires(vint32m1_t a, size_t n) { r::rvv::vadd_vv_i32m1<T>(a, a, n); };
static_assert(can_rvv<rvv1> && !can_rvv<scalar_t>);
#if defined(__riscv_zvbb) && FORGE_RAW_CLANG
template <class T> concept can_zvbb = requires(vuint32m1_t a, size_t n) { r::zvbb::vclz_v_u32m1<T>(a, n); };
static_assert(can_zvbb<caps<rvv, zvbb>> && !can_zvbb<rvv1>);
#endif
TEST(raw_rvv_executes) {
#if defined(__linux__)
    if (!(getauxval(AT_HWCAP) & (1ul << ('V' - 'A')))) { std::puts("  (no V on this CPU: skipped)"); std::exit(forge_test::failures() ? 1 : 77); }
#endif
    using T = rvv1;
    int32_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8}, b[8] = {41, 40, 39, 38, 37, 36, 35, 34}, out[8] = {};
    for (size_t i = 0; i < 8;) {
        size_t vl = r::rvv::vsetvl_e32m1<T>(8 - i);
        vint32m1_t s = r::rvv::vadd_vv_i32m1<T>(r::rvv::vle32_v_i32m1<T>(a + i, vl), r::rvv::vle32_v_i32m1<T>(b + i, vl), vl);
        r::rvv::vse32_v_i32m1<T>(out + i, s, vl); i += vl;
    }
    for (int v : out) CHECKV(v == 42);
    size_t vl = r::rvv::vsetvl_e32m1<T>(1);
    vint32m2_t w = r::rvv::vmv_v_x_i32m2<T>(7, r::rvv::vsetvlmax_e32m2<T>());
    CHECKV(r::rvv::vmv_x_s_i32m1_i32<T>(r::rvv::vget_v_i32m2_i32m1<1, T>(w)) == 7);              // index = template argument
    vfloat32m1_t f = r::rvv::vfmacc_vf_f32m1<T>(r::rvv::vfmv_v_f_f32m1<T>(2.f, vl), 8.f, r::rvv::vfmv_v_f_f32m1<T>(5.f, vl), vl);
    CHECKV(r::rvv::vfmv_f_s_f32m1_f32<T>(f) == 42.f);
}
#endif

// ------------------------------------------------------------------------------------------------ LoongArch
#elif defined(__loongarch__)
TEST(raw_generated_count) { CHECKV(r::intrinsic_count > 1000); }
#if defined(__loongarch_sx)
template <class T> concept can_lsx = requires(__m128i a) { r::lsx::vadd_w<T>(a, a); };
static_assert(can_lsx<caps<lsx>> && !can_lsx<scalar_t>);
#if defined(__loongarch_asx)
template <class T> concept can_lasx = requires(__m256i a) { r::lasx::xvadd_w<T>(a, a); };
static_assert(can_lasx<caps<lsx, lasx>> && !can_lasx<caps<lsx>>);
#endif
TEST(raw_lsx_executes) {
#if defined(__linux__)
    if (!(getauxval(AT_HWCAP) & (1ul << 4))) { std::puts("  (no LSX on this CPU: skipped)"); std::exit(forge_test::failures() ? 1 : 77); }
#endif
    using T = caps<lsx>;
    __m128i s = r::lsx::vadd_w<T>(r::lsx::vreplgr2vr_w<T>(40), r::lsx::vreplgr2vr_w<T>(2));
    CHECKV(r::lsx::vpickve2gr_w<3, T>(s) == 42);                                        // lane index = template argument
    CHECKV(r::lsx::vpickve2gr_w<0, T>(r::lsx::vslli_w<2, T>(s)) == 168);
}
#endif

// ------------------------------------------------------------------------------------------------ PowerPC
#elif defined(__powerpc__) || defined(__powerpc64__)
TEST(raw_generated_count) { CHECKV(r::intrinsic_count > 2000); }
#if defined(__ALTIVEC__)
template <class T> concept can_altivec = requires(__vector signed int a) { r::altivec::v_add<T>(a, a); };
static_assert(can_altivec<caps<altivec>> && !can_altivec<scalar_t>);
#if defined(__VSX__)
template <class T> concept can_vsx = requires(__vector double a) { r::vsx::v_add<T>(a, a); };
static_assert(can_vsx<caps<altivec, vsx>> && !can_vsx<caps<altivec>>);
#endif
TEST(raw_altivec_executes) {
#if defined(__linux__)
    if (!(getauxval(AT_HWCAP) & 0x10000000ul)) { std::puts("  (no AltiVec on this CPU: skipped)"); std::exit(forge_test::failures() ? 1 : 77); }
#endif
    using T = caps<altivec>;
    __vector signed int a = {1, 2, 3, 4}, b = {41, 40, 39, 38};
    __vector signed int s = r::altivec::v_add<T>(a, b);                                 // one overload per vector type
    __vector float f = r::altivec::v_add<T>((__vector float){1, 2, 3, 4}, (__vector float){41, 40, 39, 38});
    for (int i = 0; i < 4; ++i) { CHECKV(s[i] == 42); CHECKV(f[i] == 42.f); }
    __vector signed int m = r::altivec::v_max<T>(a, b);
    CHECKV(m[0] == 41 && m[3] == 38);
#if defined(__VSX__)
    __vector double d = r::vsx::v_mul<caps<altivec, vsx>>((__vector double){6, 7}, (__vector double){7, 6});
    CHECKV(d[0] == 42. && d[1] == 42.);
#endif
}
#endif

// ------------------------------------------------------------------------------------------------ WebAssembly
#elif defined(__wasm__)
TEST(raw_generated_count) { CHECKV(r::intrinsic_count > 250); }
#if defined(__wasm_simd128__)
template <class T> concept can_simd128 = requires(v128_t a) { r::simd128::i32x4_add<T>(a, a); };
template <class T> concept can_relaxed = requires(v128_t a) { r::relaxed_simd::f32x4_relaxed_madd<T>(a, a, a); };
static_assert(can_simd128<caps<wasm_simd128>> && !can_simd128<scalar_t>);
static_assert(can_relaxed<caps<wasm_simd128, wasm_relaxed_simd>> && !can_relaxed<caps<wasm_simd128>>);
TEST(raw_simd128_executes) {
    using T = caps<wasm_simd128>;
    v128_t s = r::simd128::i32x4_add<T>(r::simd128::i32x4_splat<T>(40), r::simd128::i32x4_splat<T>(2));
    CHECKV(r::simd128::i32x4_extract_lane<3, T>(s) == 42);                               // lane index = template argument
    CHECKV(r::simd128::i32x4_extract_lane<0, T>(r::simd128::i32x4_shl<T>(s, 2)) == 168);
}
#endif

// ------------------------------------------------------------------------------------------------ s390x
#elif defined(__s390x__)
TEST(raw_generated_count) { CHECKV(r::intrinsic_count > 1000); }
#if defined(__VEC__)
template <class T> concept can_zvector = requires(__vector signed int a) { r::zvector::v_max<T>(a, a); };
static_assert(can_zvector<caps<zvector>> && !can_zvector<scalar_t>);
#if __ARCH__ >= 12
template <class T> concept can_vxe = requires(__vector float a) { r::zvector_e1::v_max<T>(a, a); };
static_assert(can_vxe<caps<zvector, zvector_e1>> && !can_vxe<caps<zvector>>);
#endif
TEST(raw_zvector_executes) {
#if defined(__linux__)
    if (!(getauxval(AT_HWCAP) & (1ul << 11))) { std::puts("  (no vector facility on this CPU: skipped)"); std::exit(forge_test::failures() ? 1 : 77); }
#endif
    using T = caps<zvector>;
    __vector signed int a = {1, 2, 3, 4}, b = {41, 40, 39, 38};
    __vector signed int m = r::zvector::v_max<T>(a, b);
    CHECKV(m[0] == 41 && m[3] == 38);
    __vector signed int s = r::zvector::v_splats<T>(42);
    for (int i = 0; i < 4; ++i) CHECKV(s[i] == 42);
    CHECKV(r::zvector::v_extract<T>(a, 2) == 3);
}
#endif

// ------------------------------------------------------------------------------------------------ MIPS
#elif defined(__mips__)
TEST(raw_generated_count) { CHECKV(r::intrinsic_count > 400); }
#if defined(__mips_msa)
template <class T> concept can_msa = requires(v4i32 a) { r::msa::addv_w<T>(a, a); };
static_assert(can_msa<caps<msa>> && !can_msa<scalar_t>);
TEST(raw_msa_executes) {
#if defined(__linux__)
    if (!(getauxval(AT_HWCAP) & (1ul << 1))) { std::puts("  (no MSA on this CPU: skipped)"); std::exit(forge_test::failures() ? 1 : 77); }
#endif
    using T = caps<msa>;
    v4i32 s = r::msa::addv_w<T>(r::msa::fill_w<T>(40), r::msa::fill_w<T>(2));
    CHECKV(r::msa::copy_s_w<3, T>(s) == 42);                                            // lane index = template argument
    CHECKV(r::msa::copy_s_w<0, T>(r::msa::slli_w<2, T>(s)) == 168);
}
#endif
#else
TEST(raw_generated_count) { CHECKV(r::intrinsic_count == 0); }
#endif
