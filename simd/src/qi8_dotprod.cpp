// Integer matvec kernels, AArch64 NEON with the ARMv8.2 dot-product extension (sdot: Cortex-A55/A75 and later,
// every Apple M core, Neoverse N1 and later). CMake defines FORGE_QI8_HAVE_DOTPROD when the compiler accepts
// the flags (qi8_arm.inc).
#if defined(FORGE_QI8_HAVE_DOTPROD) && (defined(__aarch64__) || defined(_M_ARM64))
#define QI8_NS dotprod
#define QI8_ARM_PART 1
#include "qi8_arm.inc"
namespace forge::simd::qi8::dotprod {
using sc_t = int32x4_t;
static inline sc_t mksc(int s) { return vdupq_n_s32(s); }
static inline int32x4_t dot16(int32x4_t acc, int8x16_t w, int8x16_t a, sc_t sc) {
    return vmlaq_s32(acc, vdotq_s32(vdupq_n_s32(0), w, a), sc);
}
static inline int32x4_t dot32(int32x4_t acc, int8x16_t w0, int8x16_t w1, int8x16_t a0, int8x16_t a1, sc_t sc) {
    return vmlaq_s32(acc, vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, a0), w1, a1), sc);
}
static inline int32x4_t sdot32(int8x16_t w0, int8x16_t w1, int8x16_t a0, int8x16_t a1) {
    return vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, a0), w1, a1);
}
}
#undef QI8_ARM_PART
#define QI8_ARM_PART 2
#include "qi8_arm.inc"
namespace forge::simd::qi8_tables {
extern const qi8_kernels dotprod;
const qi8_kernels dotprod = {"dotprod", &qi8::dotprod::quantize, &qi8::dotprod::matvec};
}
#endif
