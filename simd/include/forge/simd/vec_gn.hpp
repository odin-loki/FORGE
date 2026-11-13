#pragma once
// Tier P backends for the 128/256-bit ISAs that have no dedicated branch in vec.hpp ("generic native", kind 6):
//
//   ISA                      tag            lanes  header            built when
//   x86 SSE2 / SSE4.1        sse2           4      emmintrin.h       x86 with SSE2 (x86-64 always)
//   LoongArch LSX            lsx            4      lsxintrin.h       __loongarch_sx
//   LoongArch LASX           lasx           8      lasxintrin.h      __loongarch_asx
//   Power VSX                vsx            4      altivec.h         __VSX__ (ppc64le baseline)
//   s390x vector (z14, VXE)  s390_vxe       4      vecintrin.h       -mzvector -march=z14 or later
//   WebAssembly SIMD128      wasm_simd128   4      wasm_simd128.h    -msimd128
//   MIPS MSA                 msa            4      msa.h             -mmsa
//   Armv7 NEON + VFPv4       neon           4      arm_neon.h        32-bit Arm, -mfpu=neon-vfpv4
//                                                                        (subnormal lanes go through VFP: see "Armv7" below)
//
// One backend per translation unit, chosen by the compile flags. vec.hpp maps a target onto it when the
// target has the backend's tag and the same lane count. Every function has internal linkage, so two TUs
// built with different flags (SSE2 and SSE4.1 tables, LSX and LASX tables) never share a body.
//
// The SSE backend is intrinsics only (MSVC has no vector extensions). The others use GNU vector types for
// the operations every compiler lowers to one instruction (+ - * /, compares, bit operations, shifts) and
// the ISA's intrinsics for fused multiply-add, square root, rounding and mask extraction. Build those TUs
// with -ffp-contract=off so a*b+c written with operators is never fused behind the algorithm's back.
#include <forge/simd/caps.hpp>
#include <cstring>

#if defined(_MSC_VER) && !defined(__clang__)
#define FORGE_GN_FN static __forceinline
#else
#define FORGE_GN_FN [[maybe_unused]] static inline __attribute__((always_inline))
#endif

// ---------------------------------------------------------------- x86 SSE2 / SSE4.1
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && \
    (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#include <emmintrin.h>
#if defined(__SSE4_1__) || defined(__AVX__) || defined(FORGE_SIMD_SSE41)   // FORGE_SIMD_SSE41: MSVC, which has no /arch:SSE4.2
#include <smmintrin.h>
#define FORGE_GN_SSE41 1
#endif
#if defined(__FMA__)
#include <immintrin.h>
#endif
#define FORGE_HAVE_GN 1
namespace forge::simd::gn {
using tag = sse2;
inline constexpr int lanes = 4;
#if defined(FORGE_GN_SSE41)
inline constexpr const char* isa = "sse4.1";
#else
inline constexpr const char* isa = "sse2";
#endif
using vf = __m128;
using vm = __m128;
FORGE_GN_FN vf add(vf a, vf b) { return _mm_add_ps(a, b); }
FORGE_GN_FN vf sub(vf a, vf b) { return _mm_sub_ps(a, b); }
FORGE_GN_FN vf mul(vf a, vf b) { return _mm_mul_ps(a, b); }
FORGE_GN_FN vf div(vf a, vf b) { return _mm_div_ps(a, b); }
FORGE_GN_FN vf max(vf a, vf b) { return _mm_max_ps(a, b); }
FORGE_GN_FN vf min(vf a, vf b) { return _mm_min_ps(a, b); }
FORGE_GN_FN vf splat(float s) { return _mm_set1_ps(s); }
FORGE_GN_FN vf load(const float* p) { return _mm_loadu_ps(p); }
FORGE_GN_FN void store(float* p, vf v) { _mm_storeu_ps(p, v); }
FORGE_GN_FN vf sqrt(vf a) { return _mm_sqrt_ps(a); }
// SSE has no fused multiply-add. The product of two floats is exact in double, so a*b+c is formed in double
// and rounded once more to float: correctly rounded except for double-rounding ties (error <= 0.5 ulp + 2^-29 ulp).
FORGE_GN_FN vf fma(vf a, vf b, vf c) {
#if defined(__FMA__)
    return _mm_fmadd_ps(a, b, c);
#else
    __m128d lo = _mm_add_pd(_mm_mul_pd(_mm_cvtps_pd(a), _mm_cvtps_pd(b)), _mm_cvtps_pd(c));
    __m128d hi = _mm_add_pd(_mm_mul_pd(_mm_cvtps_pd(_mm_movehl_ps(a, a)), _mm_cvtps_pd(_mm_movehl_ps(b, b))),
                            _mm_cvtps_pd(_mm_movehl_ps(c, c)));
    return _mm_movelh_ps(_mm_cvtpd_ps(lo), _mm_cvtpd_ps(hi));
#endif
}
FORGE_GN_FN vm lt(vf a, vf b) { return _mm_cmplt_ps(a, b); }
FORGE_GN_FN vm isnan(vf a) { return _mm_cmpunord_ps(a, a); }
FORGE_GN_FN vm mor(vm a, vm b) { return _mm_or_ps(a, b); }
FORGE_GN_FN vm mand(vm a, vm b) { return _mm_and_ps(a, b); }
FORGE_GN_FN vf select(vm m, vf a, vf b) {
#if defined(FORGE_GN_SSE41)
    return _mm_blendv_ps(b, a, m);
#else
    return _mm_or_ps(_mm_and_ps(m, a), _mm_andnot_ps(m, b));
#endif
}
FORGE_GN_FN vm mask_first_n(std::size_t n) {
    return _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_set1_epi32(int(n > 4 ? 4 : n)), _mm_setr_epi32(0, 1, 2, 3)));
}
FORGE_GN_FN unsigned mask_bits(vm m) { return unsigned(_mm_movemask_ps(m)); }
FORGE_GN_FN float reduce_add(vf v) {
    __m128 lo = _mm_add_ps(v, _mm_movehl_ps(v, v));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
FORGE_GN_FN vf round_nearest(vf x) {
#if defined(FORGE_GN_SSE41)
    return _mm_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
    // cvtps_epi32 rounds to nearest even (default MXCSR); |x| >= 2^23 and NaN are already integral / kept
    const __m128 sign = _mm_set1_ps(-0.0f);
    __m128 r = _mm_or_ps(_mm_cvtepi32_ps(_mm_cvtps_epi32(x)), _mm_and_ps(sign, x));
    __m128 below_2p23 = _mm_cmplt_ps(_mm_andnot_ps(sign, x), _mm_set1_ps(8388608.f));
    return _mm_or_ps(_mm_and_ps(below_2p23, r), _mm_andnot_ps(below_2p23, x));
#endif
}
FORGE_GN_FN vf ldexp_int(vf x, vf n) {
    return _mm_mul_ps(x, _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(_mm_cvtps_epi32(n), _mm_set1_epi32(127)), 23)));
}
// x = m * 2^e, m in [1, 2): raw mantissa/exponent fields (x positive and normal)
FORGE_GN_FN void mant_exp(vf x, vf& m, vf& e) {
    __m128i xi = _mm_castps_si128(x);
    e = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_srli_epi32(xi, 23), _mm_set1_epi32(127)));
    m = _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(xi, _mm_set1_epi32(0x007fffff)), _mm_set1_epi32(0x3f800000)));
}
} // namespace forge::simd::gn

// ---------------------------------------------------------------- GNU vector-extension backends
#elif defined(__GNUC__) && (defined(__loongarch_sx) || (defined(__powerpc64__) && defined(__VSX__)) || \
      (defined(__s390x__) && defined(__VEC__) && __VEC__ >= 10302) || defined(__wasm_simd128__) || \
      defined(__mips_msa) || (defined(__arm__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)))
#define FORGE_HAVE_GN 1
#if defined(__loongarch_asx)
#include <lasxintrin.h>
#define FORGE_GN_LANES 8
#define FORGE_GN_TAG lasx
#define FORGE_GN_ISA "lasx"
#elif defined(__loongarch_sx)
#include <lsxintrin.h>
#define FORGE_GN_LANES 4
#define FORGE_GN_TAG lsx
#define FORGE_GN_ISA "lsx"
#elif defined(__powerpc64__)
#include <altivec.h>
#undef vector   // altivec.h may define these context keywords as macros; only __vector / __bool are used here
#undef bool
#undef pixel
#define FORGE_GN_LANES 4
#define FORGE_GN_TAG vsx
#define FORGE_GN_ISA "vsx"
#elif defined(__s390x__)
#include <vecintrin.h>
#undef vector
#undef bool
#define FORGE_GN_LANES 4
#define FORGE_GN_TAG s390_vxe
#define FORGE_GN_ISA "vxe"
#elif defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define FORGE_GN_LANES 4
#define FORGE_GN_TAG wasm_simd128
#define FORGE_GN_ISA "simd128"
#elif defined(__mips_msa)
#include <msa.h>
#define FORGE_GN_LANES 4
#define FORGE_GN_TAG msa
#define FORGE_GN_ISA "msa"
#else
#include <arm_neon.h>
#define FORGE_GN_LANES 4
#define FORGE_GN_TAG neon
#define FORGE_GN_ISA "neon"
#define FORGE_GN_ARM32 1
#endif
namespace forge::simd::gn {
using tag = FORGE_GN_TAG;
inline constexpr int lanes = FORGE_GN_LANES;
inline constexpr const char* isa = FORGE_GN_ISA;
static_assert(sizeof(int) == 4 && sizeof(unsigned) == 4);
typedef float vf __attribute__((vector_size(4 * FORGE_GN_LANES)));
typedef int vi __attribute__((vector_size(4 * FORGE_GN_LANES)));
typedef unsigned vu __attribute__((vector_size(4 * FORGE_GN_LANES)));
using vm = vi;   // all-ones / all-zeros per lane

FORGE_GN_FN vm raw_lt(vf a, vf b) { return vm(a < b); }   // the ISA's compare instruction
FORGE_GN_FN vf div(vf a, vf b) { return a / b; }
FORGE_GN_FN vf load(const float* p) { vf v; std::memcpy(&v, p, sizeof v); return v; }
FORGE_GN_FN void store(float* p, vf v) { std::memcpy(p, &v, sizeof v); }
#if defined(FORGE_GN_ARM32)
// ---- Armv7: IEEE 754 results from Advanced SIMD ----
// On AArch32 every Advanced SIMD floating-point instruction runs with the "standard FPSCR value" whatever FPSCR
// holds: flush-to-zero (a subnormal operand is read as zero, a result below FLT_MIN becomes zero) and default
// NaN. VFP instructions on the same registers follow FPSCR, which Linux starts with FZ clear, so they are IEEE.
// add / sub / mul / fma / lt below run the NEON instruction and one cheap test of its operands and result; the
// lanes NEON may have got wrong (a subnormal operand that matters, a result it flushed, a NaN) are recomputed
// one at a time by a VFP instruction, and those lanes then hold what the scalar code computes. Division and
// square root have no NEON instruction (the compiler emits VFP ones), rounding and the conversions never see
// or produce a subnormal that changes the result, and everything else is a bit operation.
//
// A translation unit using this backend should be compiled without auto-vectorisation of its scalar code
// (clang: -fno-vectorize -fno-slp-vectorize; GCC does not vectorise floating point on NEON without
// -funsafe-math-optimizations): a scalar loop turned into NEON instructions flushes as well.
#if defined(__clang__)
#define FORGE_GN_VFP [[maybe_unused]] __attribute__((noinline, target("no-neon")))
#else
#define FORGE_GN_VFP [[maybe_unused]] __attribute__((noinline, target("fpu=vfpv4")))
#endif
#define FORGE_GN_FIX [[maybe_unused]] __attribute__((noinline)) static
namespace vfp {   // one lane, VFP instructions (compiled without NEON, never inlined into NEON code)
FORGE_GN_VFP static float add(float a, float b) { return a + b; }
FORGE_GN_VFP static float sub(float a, float b) { return a - b; }
FORGE_GN_VFP static float mul(float a, float b) { return a * b; }
FORGE_GN_VFP static float fma(float a, float b, float c) { return __builtin_fmaf(a, b, c); }   // vfma.f32
FORGE_GN_VFP static int lt(float a, float b) { return a < b ? -1 : 0; }
} // namespace vfp
namespace a32 {
using u4 = uint32x4_t;
// The bit pattern, hidden from the optimiser: clang rewrites a test such as (bits & 0x7fffffff) != 0 as the
// floating-point compare x != 0, which is a NEON compare here and reads a subnormal as zero.
FORGE_GN_FN u4 bits(vf x) { u4 b = u4(x); __asm__("" : "+w"(b)); return b; }
FORGE_GN_FN u4 normal(vf x) { return vtstq_u32(bits(x), vdupq_n_u32(0x7f800000u)); }    // exponent field not zero: normal, inf, NaN
FORGE_GN_FN u4 nonzero(vf x) { return vtstq_u32(bits(x), vdupq_n_u32(0x7fffffffu)); }
FORGE_GN_FN u4 subnormal(vf x) { return vbicq_u32(nonzero(x), normal(x)); }
FORGE_GN_FN u4 nan(vf x) { return vmvnq_u32(vceqq_f32(float32x4_t(x), float32x4_t(x))); }
// The first test of every operation is a NEON absolute compare, which reads a subnormal as zero by design:
FORGE_GN_FN u4 mag_above(vf x, float t) { return vcagtq_f32(float32x4_t(x), vdupq_n_f32(t)); }   // |x| > t; false for NaN
FORGE_GN_FN u4 mag_normal(vf x) { return vcageq_f32(float32x4_t(x), vdupq_n_f32(1.17549435e-38f)); }   // normal or inf; false for zero, subnormal, NaN
FORGE_GN_FN bool any(u4 m) {
    return vget_lane_u64(vreinterpret_u64_u32(vorr_u32(vget_low_u32(m), vget_high_u32(m))), 0) != 0;
}
FORGE_GN_FN bool all(u4 m) {
    return vget_lane_u64(vreinterpret_u64_u32(vand_u32(vget_low_u32(m), vget_high_u32(m))), 0) == ~0ull;
}
// The second, exact test and the VFP lanes. r is the NEON result. A lane is redone when its result is NaN, an
// operand is subnormal, or the result is zero although the operands do not make it zero (flushed, or an exact
// cancellation, which VFP reproduces).
FORGE_GN_FIX vf fix_add(vf a, vf b, vf r) {
    u4 bad = vorrq_u32(vorrq_u32(nan(r), vorrq_u32(subnormal(a), subnormal(b))), vbicq_u32(vandq_u32(nonzero(a), nonzero(b)), normal(r)));
    if (any(bad)) for (int i = 0; i < 4; ++i) if (bad[i]) r[i] = vfp::add(a[i], b[i]);
    return r;
}
FORGE_GN_FIX vf fix_sub(vf a, vf b, vf r) {
    u4 bad = vorrq_u32(vorrq_u32(nan(r), vorrq_u32(subnormal(a), subnormal(b))), vbicq_u32(vandq_u32(nonzero(a), nonzero(b)), normal(r)));
    if (any(bad)) for (int i = 0; i < 4; ++i) if (bad[i]) r[i] = vfp::sub(a[i], b[i]);
    return r;
}
FORGE_GN_FIX vf fix_mul(vf a, vf b, vf r) {   // a subnormal operand gives a zero or NaN result here, so the result alone tells
    u4 bad = vorrq_u32(nan(r), vbicq_u32(vandq_u32(nonzero(a), nonzero(b)), normal(r)));
    if (any(bad)) for (int i = 0; i < 4; ++i) if (bad[i]) r[i] = vfp::mul(a[i], b[i]);
    return r;
}
FORGE_GN_FIX vf fix_fma(vf a, vf b, vf c, vf r) {
    u4 sub3 = vorrq_u32(subnormal(a), vorrq_u32(subnormal(b), subnormal(c)));
    u4 lost = vbicq_u32(vorrq_u32(vandq_u32(nonzero(a), nonzero(b)), nonzero(c)), normal(r));
    u4 bad = vorrq_u32(nan(r), vorrq_u32(sub3, lost));
    if (any(bad)) for (int i = 0; i < 4; ++i) if (bad[i]) r[i] = vfp::fma(a[i], b[i], c[i]);
    return r;
}
FORGE_GN_FIX vm fix_lt(vf a, vf b, vm r) {
    u4 bad = vorrq_u32(subnormal(a), subnormal(b));
    if (any(bad)) for (int i = 0; i < 4; ++i) if (bad[i]) r[i] = vfp::lt(a[i], b[i]);
    return r;
}
} // namespace a32
// A sum or difference is wrong only when it is tiny: a subnormal operand changes a result above 2^-100 by less
// than a quarter of its last place, and a sum of normal numbers that lands below FLT_MIN is flushed to zero.
FORGE_GN_FN vf add(vf a, vf b) {
    vf r = a + b;
    if (__builtin_expect(!a32::all(a32::mag_above(r, 0x1p-100f)), 0)) r = a32::fix_add(a, b, r);
    return r;
}
FORGE_GN_FN vf sub(vf a, vf b) {
    vf r = a - b;
    if (__builtin_expect(!a32::all(a32::mag_above(r, 0x1p-100f)), 0)) r = a32::fix_sub(a, b, r);
    return r;
}
// A product NEON got wrong is zero (flushed, or a subnormal operand read as zero) or NaN (subnormal * inf).
FORGE_GN_FN vf mul(vf a, vf b) {
    vf r = a * b;
    if (__builtin_expect(!a32::all(a32::mag_above(r, 0.f)), 0)) r = a32::fix_mul(a, b, r);
    return r;
}
// A subnormal operand of a fused multiply-add can change any result (it breaks a tie of the exact product), so
// all three operands are tested as well as the result.
FORGE_GN_FN vf fma(vf a, vf b, vf c) {
    vf r = vf(vfmaq_f32(float32x4_t(c), float32x4_t(a), float32x4_t(b)));
    a32::u4 ok = vandq_u32(vandq_u32(a32::mag_above(r, 0.f), a32::mag_normal(a)), vandq_u32(a32::mag_normal(b), a32::mag_normal(c)));
    if (__builtin_expect(!a32::all(ok), 0)) r = a32::fix_fma(a, b, c, r);
    return r;
}
// A compare is wrong only between a subnormal and a zero or another subnormal.
FORGE_GN_FN vm lt(vf a, vf b) {
    vm r = raw_lt(a, b);
    if (__builtin_expect(!a32::all(vorrq_u32(a32::mag_normal(a), a32::mag_normal(b))), 0)) r = a32::fix_lt(a, b, r);
    return r;
}
FORGE_GN_FN vf splat(float s) { return vf(vdupq_n_f32(s)); }   // a move: `vf{} + s` would be a NEON addition
#else
FORGE_GN_FN vf add(vf a, vf b) { return a + b; }
FORGE_GN_FN vf sub(vf a, vf b) { return a - b; }
FORGE_GN_FN vf mul(vf a, vf b) { return a * b; }
FORGE_GN_FN vf splat(float s) { return vf{} + s; }
FORGE_GN_FN vm lt(vf a, vf b) { return raw_lt(a, b); }
#endif
#if defined(__mips_msa)
FORGE_GN_FN vm isnan(vf a) { return vm(__msa_fcun_w(v4f32(a), v4f32(a))); }   // GCC 14 lowers a != a to fcne, false for NaN
#else
FORGE_GN_FN vm isnan(vf a) { return vm(a != a); }
#endif
FORGE_GN_FN vm mor(vm a, vm b) { return a | b; }
FORGE_GN_FN vm mand(vm a, vm b) { return a & b; }
#if defined(__mips_msa)
// bsel.v directly: with the and/or form below, clang 22 (zig 0.17) produced wrong expm1 / log1p / rounding
// results on MSA under qemu; every other ISA here passes with it
FORGE_GN_FN vf select(vm m, vf a, vf b) { return vf(__msa_bsel_v(v16u8(m), v16u8(b), v16u8(a))); }
#else
FORGE_GN_FN vf select(vm m, vf a, vf b) { return vf((vi(a) & m) | (vi(b) & ~m)); }
#endif
// x86 max/min semantics (the second operand when unordered), as on every other target
FORGE_GN_FN vf max(vf a, vf b) { return select(lt(b, a), a, b); }
FORGE_GN_FN vf min(vf a, vf b) { return select(lt(a, b), a, b); }
FORGE_GN_FN vm mask_first_n(std::size_t n) {
#if FORGE_GN_LANES == 8
    const vi idx = {0, 1, 2, 3, 4, 5, 6, 7};
#else
    const vi idx = {0, 1, 2, 3};
#endif
    return vm(idx < (vi{} + int(n > std::size_t(lanes) ? std::size_t(lanes) : n)));
}
FORGE_GN_FN vf round_magic(vf x) {   // nearest even through 2^23, for ISAs without a rounding instruction
    const vi sign = vi{} + int(0x80000000u);
    vf ax = vf(vi(x) & ~sign);
    vf big = vf{} + 8388608.f;
    vf r = (ax + big) - big;   // plain instructions: a subnormal gives zero here on every ISA
    return select(raw_lt(ax, big), vf(vi(r) | (vi(x) & sign)), x);
}

#if defined(__loongarch_asx)
FORGE_GN_FN vf fma(vf a, vf b, vf c) { return vf(__lasx_xvfmadd_s(__m256(a), __m256(b), __m256(c))); }
FORGE_GN_FN vf sqrt(vf a) { return vf(__lasx_xvfsqrt_s(__m256(a))); }
FORGE_GN_FN vf round_nearest(vf a) { return vf(__lasx_xvfrintrne_s(__m256(a))); }
FORGE_GN_FN unsigned mask_bits(vm m) {
    __m256i k = __lasx_xvmskltz_w(__m256i(m));   // 4 bits per 128-bit half
    return unsigned(__lasx_xvpickve2gr_wu(k, 0)) | (unsigned(__lasx_xvpickve2gr_wu(k, 4)) << 4);
}
#define FORGE_GN_HAVE_BITS 1
#elif defined(__loongarch_sx)
FORGE_GN_FN vf fma(vf a, vf b, vf c) { return vf(__lsx_vfmadd_s(__m128(a), __m128(b), __m128(c))); }
FORGE_GN_FN vf sqrt(vf a) { return vf(__lsx_vfsqrt_s(__m128(a))); }
FORGE_GN_FN vf round_nearest(vf a) { return vf(__lsx_vfrintrne_s(__m128(a))); }
FORGE_GN_FN unsigned mask_bits(vm m) { return unsigned(__lsx_vpickve2gr_wu(__lsx_vmskltz_w(__m128i(m)), 0)); }
#define FORGE_GN_HAVE_BITS 1
#elif defined(__powerpc64__)
typedef __vector float nf;
FORGE_GN_FN vf fma(vf a, vf b, vf c) { return vf(vec_madd(nf(a), nf(b), nf(c))); }
FORGE_GN_FN vf sqrt(vf a) { return vf(vec_sqrt(nf(a))); }
FORGE_GN_FN vf round_nearest(vf a) { return vf(vec_rint(nf(a))); }   // xvrspic: current mode (nearest even)
#elif defined(__s390x__)
typedef __vector float nf;
FORGE_GN_FN vf fma(vf a, vf b, vf c) { return vf(vec_madd(nf(a), nf(b), nf(c))); }
FORGE_GN_FN vf sqrt(vf a) { return vf(vec_sqrt(nf(a))); }
FORGE_GN_FN vf round_nearest(vf a) { return vf(vec_rint(nf(a))); }   // vfisb, current mode (nearest even)
#elif defined(__wasm_simd128__)
// No fused multiply-add in SIMD128 (relaxed-simd's madd may or may not fuse, so it is not used): through f64,
// as in the SSE backend.
FORGE_GN_FN vf fma(vf a, vf b, vf c) {
    v128_t A = v128_t(a), B = v128_t(b), C = v128_t(c);
    v128_t lo = wasm_f64x2_add(wasm_f64x2_mul(wasm_f64x2_promote_low_f32x4(A), wasm_f64x2_promote_low_f32x4(B)),
                               wasm_f64x2_promote_low_f32x4(C));
    v128_t Ah = wasm_i32x4_shuffle(A, A, 2, 3, 2, 3), Bh = wasm_i32x4_shuffle(B, B, 2, 3, 2, 3), Ch = wasm_i32x4_shuffle(C, C, 2, 3, 2, 3);
    v128_t hi = wasm_f64x2_add(wasm_f64x2_mul(wasm_f64x2_promote_low_f32x4(Ah), wasm_f64x2_promote_low_f32x4(Bh)),
                               wasm_f64x2_promote_low_f32x4(Ch));
    return vf(wasm_i32x4_shuffle(wasm_f32x4_demote_f64x2_zero(lo), wasm_f32x4_demote_f64x2_zero(hi), 0, 1, 4, 5));
}
FORGE_GN_FN vf sqrt(vf a) { return vf(wasm_f32x4_sqrt(v128_t(a))); }
FORGE_GN_FN vf round_nearest(vf a) { return vf(wasm_f32x4_nearest(v128_t(a))); }
FORGE_GN_FN unsigned mask_bits(vm m) { return unsigned(wasm_i32x4_bitmask(v128_t(m))); }
#define FORGE_GN_HAVE_BITS 1
#elif defined(__mips_msa)
FORGE_GN_FN vf fma(vf a, vf b, vf c) { return vf(__msa_fmadd_w(v4f32(c), v4f32(a), v4f32(b))); }
FORGE_GN_FN vf sqrt(vf a) { return vf(__msa_fsqrt_w(v4f32(a))); }
FORGE_GN_FN vf round_nearest(vf a) { return vf(__msa_frint_w(v4f32(a))); }   // MSACSR mode (nearest even)
#else   // Armv7 NEON + VFPv4: no vector divide (the compiler emits VFP ones), square root or rounding; fma is above
FORGE_GN_FN vf sqrt(vf a) { vf o; for (int i = 0; i < lanes; ++i) o[i] = __builtin_sqrtf(a[i]); return o; }
FORGE_GN_FN vf round_nearest(vf a) { return round_magic(a); }
#endif

#if !defined(FORGE_GN_HAVE_BITS)
FORGE_GN_FN unsigned mask_bits(vm m) {
    unsigned b = 0;
    for (int i = 0; i < lanes; ++i) b |= (unsigned(m[i]) & 1u) << i;
    return b;
}
#endif
FORGE_GN_FN float reduce_add(vf v) {
#if defined(FORGE_GN_ARM32)   // the same tree with the checked add, then one VFP addition
    vf t = add(v, vf(vcombine_f32(vget_high_f32(float32x4_t(v)), vget_low_f32(float32x4_t(v)))));
    return vfp::add(t[0], t[1]);
#elif FORGE_GN_LANES == 8
    return ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
#else
    return (v[0] + v[2]) + (v[1] + v[3]);
#endif
}
// n is integer valued; clamped first so the float -> int conversion is always in range
FORGE_GN_FN vf ldexp_int(vf x, vf n) {
    const vf hi = vf{} + 1024.f, lo = vf{} - 1024.f;
    n = select(raw_lt(n, hi), n, hi);   // min, max: n is no subnormal
    n = select(raw_lt(lo, n), n, lo);
    vi e = vi(vu(__builtin_convertvector(n, vi) + 127) << 23);
    return mul(x, vf(e));
}
FORGE_GN_FN void mant_exp(vf x, vf& m, vf& e) {
    vu xi = vu(x);
    e = __builtin_convertvector(vi(xi >> 23) - 127, vf);
    m = vf((xi & 0x007fffffu) | 0x3f800000u);
}
} // namespace forge::simd::gn
#endif

#if defined(FORGE_HAVE_GN)
#define FORGE_IFGN(...) __VA_ARGS__
#else
#define FORGE_IFGN(...)
#endif
