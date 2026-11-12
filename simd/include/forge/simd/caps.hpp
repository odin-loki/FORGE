#pragma once
// Capabilities, not hierarchies: a target is a *set* of features; implementations require subsets.
#include <type_traits>

namespace forge::simd {

// ---- feature tags ----
// (FMA3 is tagged fma3, not fma: MSVC rejects a class and the simd::fma function template sharing a name.)
#define FORGE_FEATURES(X) \
    X(sse2) X(sse4_2) X(popcnt) X(avx) X(avx2) X(fma3) X(f16c) X(bmi2) \
    X(avx512f) X(avx512bw) X(avx512dq) X(avx512vl) X(avx512cd) X(avx512vnni) X(avx512bf16) \
    X(avx512fp16) X(avx512ifma) X(avx512vbmi) X(avx512vbmi2) X(avx512bitalg) X(avx512vpopcntdq) \
    X(avxvnni) X(avx10_1) X(amx_tile) X(amx_int8) X(amx_bf16) X(amx_fp16) \
    X(neon) X(dotprod) X(i8mm) X(arm_bf16) X(arm_fp16) X(sve) X(sve2) X(sme) X(sme2) \
    X(rvv) X(zvfh) X(zvbb) X(vsx) X(wasm_simd128) X(lsx) X(lasx) \
    X(s390_vxe) X(msa)

#define FORGE_TAG(n) struct n { static constexpr const char* name = #n; };
FORGE_FEATURES(FORGE_TAG)
#undef FORGE_TAG

// ---- Tier R feature tags ----
// Features only the raw intrinsic wrappers (forge/simd/raw.hpp) gate on; no kernel table is selected by them.
// The names are the "tag" of a feature in ops/intrinsics/<isa>.json. Kept apart from FORGE_FEATURES so the two
// lists can grow independently.
#define FORGE_RAW_FEATURES(X) \
    /* x86 */ X(lzcnt) X(bmi1) X(avx512vp2intersect) X(avxvnniint8) X(avxvnniint16) X(avxifma) X(avxneconvert) \
        X(avx10_2) X(aesni) X(pclmulqdq) X(shani) X(sha512) X(sm3) X(sm4) X(gfni) X(vaes) X(vpclmulqdq) \
        X(amx_complex) X(amx_fp8) X(amx_avx512) \
    /* Arm */ X(arm_aes) X(arm_complex) X(arm_crc) X(arm_f16f32dot) X(arm_f16f32mm) X(arm_f16mm) X(arm_f32mm) \
        X(arm_f64mm) X(arm_f8f16mm) X(arm_f8f32mm) X(arm_faminmax) X(arm_fp16fml) X(arm_fp8) X(arm_fp8dot2) \
        X(arm_fp8dot4) X(arm_fp8fma) X(arm_frintts) X(arm_lut) X(arm_rcpc3) X(arm_rdm) X(arm_sha2) X(arm_sha3) \
        X(arm_sm4) X(sme2p1) X(sme2p2) X(sme2p3) X(sme_b16b16) X(sme_f16f16) X(sme_f64f64) X(sme_f8f16) \
        X(sme_f8f32) X(sme_i16i64) X(sme_lutv2) X(sme_mop4) X(sme_tmop) X(sve2p1) X(sve2p2) X(sve2p3) X(sve_aes) \
        X(sve_aes2) X(sve_b16b16) X(sve_b16mm) X(sve_bfscale) X(sve_bitperm) X(sve_f16f32mm) X(sve_sha3) \
        X(sve_sm4) \
    /* RISC-V */ X(zbb) X(zbc) X(zbkb) X(zbkx) X(zknd) X(zkne) X(zknh) X(zksh) X(zvbc) X(zvfbfmin) X(zvfbfwma) \
        X(zvfhmin) X(zvkb) X(zvkg) X(zvkned) X(zvknha) X(zvknhb) X(zvksed) X(zvksh) X(zvl256b) X(zvl512b) \
    /* PowerPC */ X(altivec) X(power8_vector) X(power9_vector) X(power10_vector) \
    /* WebAssembly */ X(wasm_relaxed_simd) X(wasm_fp16) \
    /* s390x */ X(zvector) X(zvector_e1) X(zvector_e2) X(zvector_e3)
    // (lsx, lasx, vsx, wasm_simd128 and msa are in FORGE_FEATURES above: Tier P and Tier R share those tags.)

#define FORGE_TAG(n) struct n { static constexpr const char* name = #n; };
FORGE_RAW_FEATURES(FORGE_TAG)
#undef FORGE_TAG
// ---- end Tier R feature tags ----





template <class... F> struct caps {
    template <class G> static constexpr bool has = (std::is_same_v<F, G> || ...);
    using is_target = void;
};

template <class Base, class... Extra> struct caps_extend_t;
template <class... B, class... E> struct caps_extend_t<caps<B...>, E...> { using type = caps<B..., E...>; };
template <class Base, class... E> struct caps_extend : caps_extend_t<typename Base::caps_type, E...>::type {
};

template <class T> concept Target = requires { typename T::is_target; };
template <class T, class F> concept Has = Target<T> && T::template has<F>;

// ---- presets ----
struct scalar_t : caps<> { using caps_type = caps<>; static constexpr const char* name = "scalar"; };
struct x86_64_v2 : caps<sse2, sse4_2, popcnt> {
    using caps_type = caps<sse2, sse4_2, popcnt>; static constexpr const char* name = "x86-64-v2"; };
struct x86_64_v3 : caps<sse2, sse4_2, popcnt, avx, avx2, fma3, f16c, bmi2> {
    using caps_type = caps<sse2, sse4_2, popcnt, avx, avx2, fma3, f16c, bmi2>; static constexpr const char* name = "x86-64-v3"; };
struct x86_64_v4 : caps_extend<x86_64_v3, avx512f, avx512bw, avx512dq, avx512vl, avx512cd> {
    using caps_type = caps_extend_t<x86_64_v3::caps_type, avx512f, avx512bw, avx512dq, avx512vl, avx512cd>::type;
    static constexpr const char* name = "x86-64-v4"; };
struct zen4 : caps_extend<x86_64_v4, avx512vnni, avx512bf16, avx512ifma, avx512vbmi, avx512vbmi2, avx512bitalg, avx512vpopcntdq> {
    static constexpr const char* name = "zen4"; };
struct sapphire : caps_extend<x86_64_v4, avx512vnni, avx512bf16, avx512fp16, avx512ifma, avx512vbmi, avx512vbmi2,
                              avx512bitalg, avx512vpopcntdq, avxvnni, amx_tile, amx_int8, amx_bf16> {
    static constexpr const char* name = "sapphire-rapids"; };
struct neoverse2 : caps<neon, dotprod, i8mm, arm_bf16, arm_fp16, sve, sve2> {
    using caps_type = caps<neon, dotprod, i8mm, arm_bf16, arm_fp16, sve, sve2>; static constexpr const char* name = "neoverse-v2"; };
struct apple_m : caps<neon, dotprod, i8mm, arm_bf16, arm_fp16> {
    using caps_type = caps<neon, dotprod, i8mm, arm_bf16, arm_fp16>; static constexpr const char* name = "apple-m"; };
struct rvv1 : caps<rvv, zvfh> { using caps_type = caps<rvv, zvfh>; static constexpr const char* name = "rvv1"; };
// Tier P presets for the vec_gn.hpp backends (one kernel table each; see docs/simd.md).
struct x86_64_v1 : caps<sse2> { using caps_type = caps<sse2>; static constexpr const char* name = "x86-64-v1"; };
struct loongarch_lsx : caps<lsx> { using caps_type = caps<lsx>; static constexpr const char* name = "loongarch-lsx"; };
struct loongarch_lasx : caps<lsx, lasx> { using caps_type = caps<lsx, lasx>; static constexpr const char* name = "loongarch-lasx"; };
// POWER presets: vsx selects the Tier P kernel table; altivec / powerN_vector are the Tier R gates the same -mcpu implies.
struct power7 : caps<altivec, vsx> { using caps_type = caps<altivec, vsx>; static constexpr const char* name = "power7"; };
struct power8 : caps<altivec, vsx, power8_vector> {
    using caps_type = caps<altivec, vsx, power8_vector>; static constexpr const char* name = "power8"; };
struct s390x_z14 : caps<s390_vxe> { using caps_type = caps<s390_vxe>; static constexpr const char* name = "z14"; };
struct wasm_simd : caps<wasm_simd128> { using caps_type = caps<wasm_simd128>; static constexpr const char* name = "wasm-simd128"; };
struct mips_msa : caps<msa> { using caps_type = caps<msa>; static constexpr const char* name = "mips-msa"; };
struct armv7_neon : caps<neon> { using caps_type = caps<neon>; static constexpr const char* name = "armv7-neon"; };
// Scalable-vector targets at a fixed register length (bits); one kernel table is compiled per length.
template <int Bits> struct sve_vl : caps<neon, dotprod, i8mm, arm_bf16, arm_fp16, sve, sve2> {
    using caps_type = caps<neon, dotprod, i8mm, arm_bf16, arm_fp16, sve, sve2>; static constexpr int vl = Bits; static constexpr const char* name = "sve"; };
template <int Bits> struct rvv_vl : caps<rvv, zvfh> {
    using caps_type = caps<rvv, zvfh>; static constexpr int vl = Bits; static constexpr const char* name = "rvv"; };
template <class T> constexpr int vl_bits() { if constexpr (requires { T::vl; }) return T::vl; else return 256; }





// ---- Tier R presets and compile-time detection ----
// (loongarch_lsx / loongarch_lasx / power7 / power8 / wasm_simd / mips_msa are with the Tier P presets above.)
struct power9 : caps_extend<power7, power8_vector, power9_vector> { static constexpr const char* name = "power9"; };
struct power10 : caps_extend<power7, power8_vector, power9_vector, power10_vector> { static constexpr const char* name = "power10"; };
struct wasm_relaxed : caps<wasm_simd128, wasm_relaxed_simd> {
    using caps_type = caps<wasm_simd128, wasm_relaxed_simd>; static constexpr const char* name = "wasm-relaxed-simd"; };
struct z13 : caps<zvector> { using caps_type = caps<zvector>; static constexpr const char* name = "z13"; };
struct z14 : caps<zvector, zvector_e1, s390_vxe> { using caps_type = caps<zvector, zvector_e1, s390_vxe>; static constexpr const char* name = "z14"; };
struct z15 : caps<zvector, zvector_e1, zvector_e2, s390_vxe> {
    using caps_type = caps<zvector, zvector_e1, zvector_e2, s390_vxe>; static constexpr const char* name = "z15"; };

// The feature set this translation unit is compiled for, from the compiler's predefined macros
// (-march / -m flags, /arch). `compiled_t` is a Target: Has<compiled_t, avx2> is true in a TU built with -mavx2.
template <class F> inline constexpr bool compiled_in = false;
#if defined(__SSE2__) || defined(_M_X64)
template <> inline constexpr bool compiled_in<sse2> = true;
#endif
#if defined(__SSE4_2__)
template <> inline constexpr bool compiled_in<sse4_2> = true;
#endif
#if defined(__POPCNT__)
template <> inline constexpr bool compiled_in<popcnt> = true;
#endif
#if defined(__LZCNT__)
template <> inline constexpr bool compiled_in<lzcnt> = true;
#endif
#if defined(__BMI__)
template <> inline constexpr bool compiled_in<bmi1> = true;
#endif
#if defined(__BMI2__)
template <> inline constexpr bool compiled_in<bmi2> = true;
#endif
#if defined(__AVX__)
template <> inline constexpr bool compiled_in<avx> = true;
#endif
#if defined(__AVX2__)
template <> inline constexpr bool compiled_in<avx2> = true;
#endif
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
template <> inline constexpr bool compiled_in<fma3> = true;
#endif
#if defined(__F16C__)
template <> inline constexpr bool compiled_in<f16c> = true;
#endif
#if defined(__AES__)
template <> inline constexpr bool compiled_in<aesni> = true;
#endif
#if defined(__PCLMUL__)
template <> inline constexpr bool compiled_in<pclmulqdq> = true;
#endif
#if defined(__SHA__)
template <> inline constexpr bool compiled_in<shani> = true;
#endif
#if defined(__AVXVNNI__)
template <> inline constexpr bool compiled_in<avxvnni> = true;
#endif
#if defined(__AMX_TILE__)
template <> inline constexpr bool compiled_in<amx_tile> = true;
#endif
#if defined(__AMX_INT8__)
template <> inline constexpr bool compiled_in<amx_int8> = true;
#endif
#if defined(__AMX_BF16__)
template <> inline constexpr bool compiled_in<amx_bf16> = true;
#endif
#if defined(__AMX_FP16__)
template <> inline constexpr bool compiled_in<amx_fp16> = true;
#endif
#if defined(__AVX10_2__)
template <> inline constexpr bool compiled_in<avx10_2> = true;
#endif
#if defined(__AVX512BF16__)
template <> inline constexpr bool compiled_in<avx512bf16> = true;
#endif
#if defined(__AVX512BITALG__)
template <> inline constexpr bool compiled_in<avx512bitalg> = true;
#endif
#if defined(__AVX512BW__)
template <> inline constexpr bool compiled_in<avx512bw> = true;
#endif
#if defined(__AVX512CD__)
template <> inline constexpr bool compiled_in<avx512cd> = true;
#endif
#if defined(__AVX512DQ__)
template <> inline constexpr bool compiled_in<avx512dq> = true;
#endif
#if defined(__AVX512F__)
template <> inline constexpr bool compiled_in<avx512f> = true;
#endif
#if defined(__AVX512FP16__)
template <> inline constexpr bool compiled_in<avx512fp16> = true;
#endif
#if defined(__AVX512IFMA__)
template <> inline constexpr bool compiled_in<avx512ifma> = true;
#endif
#if defined(__AVX512VBMI__)
template <> inline constexpr bool compiled_in<avx512vbmi> = true;
#endif
#if defined(__AVX512VBMI2__)
template <> inline constexpr bool compiled_in<avx512vbmi2> = true;
#endif
#if defined(__AVX512VL__)
template <> inline constexpr bool compiled_in<avx512vl> = true;
#endif
#if defined(__AVX512VNNI__)
template <> inline constexpr bool compiled_in<avx512vnni> = true;
#endif
#if defined(__AVX512VP2INTERSECT__)
template <> inline constexpr bool compiled_in<avx512vp2intersect> = true;
#endif
#if defined(__AVX512VPOPCNTDQ__)
template <> inline constexpr bool compiled_in<avx512vpopcntdq> = true;
#endif
#if defined(__AVXIFMA__)
template <> inline constexpr bool compiled_in<avxifma> = true;
#endif
#if defined(__AVXNECONVERT__)
template <> inline constexpr bool compiled_in<avxneconvert> = true;
#endif
#if defined(__AVXVNNIINT16__)
template <> inline constexpr bool compiled_in<avxvnniint16> = true;
#endif
#if defined(__AVXVNNIINT8__)
template <> inline constexpr bool compiled_in<avxvnniint8> = true;
#endif
#if defined(__GFNI__)
template <> inline constexpr bool compiled_in<gfni> = true;
#endif
#if defined(__SHA512__)
template <> inline constexpr bool compiled_in<sha512> = true;
#endif
#if defined(__SM3__)
template <> inline constexpr bool compiled_in<sm3> = true;
#endif
#if defined(__SM4__)
template <> inline constexpr bool compiled_in<sm4> = true;
#endif
#if defined(__VAES__)
template <> inline constexpr bool compiled_in<vaes> = true;
#endif
#if defined(__VPCLMULQDQ__)
template <> inline constexpr bool compiled_in<vpclmulqdq> = true;
#endif
#if defined(__ARM_NEON) || defined(_M_ARM64)
template <> inline constexpr bool compiled_in<neon> = true;
#endif
#if defined(__ARM_FEATURE_DOTPROD)
template <> inline constexpr bool compiled_in<dotprod> = true;
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
template <> inline constexpr bool compiled_in<i8mm> = true;
#endif
#if defined(__ARM_FEATURE_BF16)
template <> inline constexpr bool compiled_in<arm_bf16> = true;
#endif
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <> inline constexpr bool compiled_in<arm_fp16> = true;
#endif
#if defined(__ARM_FEATURE_SVE)
template <> inline constexpr bool compiled_in<sve> = true;
#endif
#if defined(__ARM_FEATURE_SVE2)
template <> inline constexpr bool compiled_in<sve2> = true;
#endif
#if defined(__ARM_FEATURE_SME)
template <> inline constexpr bool compiled_in<sme> = true;
#endif
#if defined(__ARM_FEATURE_SME2)
template <> inline constexpr bool compiled_in<sme2> = true;
#endif
#if defined(__ARM_FEATURE_AES)
template <> inline constexpr bool compiled_in<arm_aes> = true;
#endif
#if defined(__ARM_FEATURE_SHA2)
template <> inline constexpr bool compiled_in<arm_sha2> = true;
#endif
#if defined(__ARM_FEATURE_SHA3)
template <> inline constexpr bool compiled_in<arm_sha3> = true;
#endif
#if defined(__ARM_FEATURE_SM4)
template <> inline constexpr bool compiled_in<arm_sm4> = true;
#endif
#if defined(__ARM_FEATURE_CRC32)
template <> inline constexpr bool compiled_in<arm_crc> = true;
#endif
#if defined(__ARM_FEATURE_QRDMX)
template <> inline constexpr bool compiled_in<arm_rdm> = true;
#endif
#if defined(__ARM_FEATURE_COMPLEX)
template <> inline constexpr bool compiled_in<arm_complex> = true;
#endif
#if defined(__ARM_FEATURE_FRINT)
template <> inline constexpr bool compiled_in<arm_frintts> = true;
#endif
#if defined(__ARM_FEATURE_SVE2_AES)
template <> inline constexpr bool compiled_in<sve_aes> = true;
#endif
#if defined(__ARM_FEATURE_SVE2_BITPERM)
template <> inline constexpr bool compiled_in<sve_bitperm> = true;
#endif
#if defined(__ARM_FEATURE_SVE2_SHA3)
template <> inline constexpr bool compiled_in<sve_sha3> = true;
#endif
#if defined(__ARM_FEATURE_SVE2_SM4)
template <> inline constexpr bool compiled_in<sve_sm4> = true;
#endif
#if defined(__ARM_FEATURE_SVE2p1)
template <> inline constexpr bool compiled_in<sve2p1> = true;
#endif
#if defined(__ARM_FEATURE_SME2p1)
template <> inline constexpr bool compiled_in<sme2p1> = true;
#endif
#if defined(__ARM_FEATURE_SME_I16I64)
template <> inline constexpr bool compiled_in<sme_i16i64> = true;
#endif
#if defined(__ARM_FEATURE_SME_F64F64)
template <> inline constexpr bool compiled_in<sme_f64f64> = true;
#endif
#if defined(__riscv_vector)
template <> inline constexpr bool compiled_in<rvv> = true;
#endif
#if defined(__riscv_zbb)
template <> inline constexpr bool compiled_in<zbb> = true;
#endif
#if defined(__riscv_zbc)
template <> inline constexpr bool compiled_in<zbc> = true;
#endif
#if defined(__riscv_zbkb)
template <> inline constexpr bool compiled_in<zbkb> = true;
#endif
#if defined(__riscv_zbkx)
template <> inline constexpr bool compiled_in<zbkx> = true;
#endif
#if defined(__riscv_zknd)
template <> inline constexpr bool compiled_in<zknd> = true;
#endif
#if defined(__riscv_zkne)
template <> inline constexpr bool compiled_in<zkne> = true;
#endif
#if defined(__riscv_zknh)
template <> inline constexpr bool compiled_in<zknh> = true;
#endif
#if defined(__riscv_zksh)
template <> inline constexpr bool compiled_in<zksh> = true;
#endif
#if defined(__riscv_zvbb)
template <> inline constexpr bool compiled_in<zvbb> = true;
#endif
#if defined(__riscv_zvbc)
template <> inline constexpr bool compiled_in<zvbc> = true;
#endif
#if defined(__riscv_zvfbfmin)
template <> inline constexpr bool compiled_in<zvfbfmin> = true;
#endif
#if defined(__riscv_zvfbfwma)
template <> inline constexpr bool compiled_in<zvfbfwma> = true;
#endif
#if defined(__riscv_zvfh)
template <> inline constexpr bool compiled_in<zvfh> = true;
#endif
#if defined(__riscv_zvfhmin)
template <> inline constexpr bool compiled_in<zvfhmin> = true;
#endif
#if defined(__riscv_zvkb)
template <> inline constexpr bool compiled_in<zvkb> = true;
#endif
#if defined(__riscv_zvkg)
template <> inline constexpr bool compiled_in<zvkg> = true;
#endif
#if defined(__riscv_zvkned)
template <> inline constexpr bool compiled_in<zvkned> = true;
#endif
#if defined(__riscv_zvknha)
template <> inline constexpr bool compiled_in<zvknha> = true;
#endif
#if defined(__riscv_zvknhb)
template <> inline constexpr bool compiled_in<zvknhb> = true;
#endif
#if defined(__riscv_zvksed)
template <> inline constexpr bool compiled_in<zvksed> = true;
#endif
#if defined(__riscv_zvksh)
template <> inline constexpr bool compiled_in<zvksh> = true;
#endif
#if defined(__riscv_zvl256b)
template <> inline constexpr bool compiled_in<zvl256b> = true;
#endif
#if defined(__riscv_zvl512b)
template <> inline constexpr bool compiled_in<zvl512b> = true;
#endif
#if defined(__loongarch_sx)
template <> inline constexpr bool compiled_in<lsx> = true;
#endif
#if defined(__loongarch_asx)
template <> inline constexpr bool compiled_in<lasx> = true;
#endif
#if defined(__ALTIVEC__)
template <> inline constexpr bool compiled_in<altivec> = true;
#endif
#if defined(__VSX__)
template <> inline constexpr bool compiled_in<vsx> = true;
#endif
#if defined(__POWER8_VECTOR__)
template <> inline constexpr bool compiled_in<power8_vector> = true;
#endif
#if defined(__POWER9_VECTOR__)
template <> inline constexpr bool compiled_in<power9_vector> = true;
#endif
#if defined(__POWER10_VECTOR__)
template <> inline constexpr bool compiled_in<power10_vector> = true;
#endif
#if defined(__wasm_simd128__)
template <> inline constexpr bool compiled_in<wasm_simd128> = true;
#endif
#if defined(__wasm_relaxed_simd__)
template <> inline constexpr bool compiled_in<wasm_relaxed_simd> = true;
#endif
#if defined(__wasm_fp16__)
template <> inline constexpr bool compiled_in<wasm_fp16> = true;
#endif
#if defined(__VX__) && defined(__ARCH__) && __ARCH__ >= 11
template <> inline constexpr bool compiled_in<zvector> = true;
#endif
#if defined(__VX__) && defined(__ARCH__) && __ARCH__ >= 12
template <> inline constexpr bool compiled_in<zvector_e1> = true;
#endif
#if defined(__VX__) && defined(__ARCH__) && __ARCH__ >= 13
template <> inline constexpr bool compiled_in<zvector_e2> = true;
#endif
#if defined(__VX__) && defined(__ARCH__) && __ARCH__ >= 15
template <> inline constexpr bool compiled_in<zvector_e3> = true;
#endif
#if defined(__mips_msa)
template <> inline constexpr bool compiled_in<msa> = true;
#endif
struct compiled_t {
    template <class G> static constexpr bool has = compiled_in<G>;
    using is_target = void;
    static constexpr const char* name = "compiled";
};
// ---- end Tier R presets ----

// Widest native f32 register width (in lanes) implied by a target.
template <Target T> constexpr int native_f32_lanes() {
    if constexpr (Has<T, avx512f>) return 16;
    else if constexpr (Has<T, avx2>) return 8;
    else if constexpr (Has<T, sve>) return vl_bits<T>() / 32;   // SVE / RVV: fixed length per tag (default 256)
    else if constexpr (Has<T, rvv>) return vl_bits<T>() / 32;
    else if constexpr (Has<T, lasx>) return 8;
    else if constexpr (Has<T, neon>) return 4;
    else return 4; // scalar emulation, 4-wide for ILP
}

} // namespace forge::simd
