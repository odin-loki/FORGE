// Integer matvec kernels, AArch64 baseline NEON (ARMv8.0): smull / smlal into int16, then widening
// multiply-accumulates by the sub-block scale into int32 (qi8_arm.inc).
#if defined(__aarch64__) || defined(_M_ARM64)
#define QI8_NS neon
#define QI8_ARM_PART 1
#include "qi8_arm.inc"
namespace forge::simd::qi8::neon {
using sc_t = int16x8_t;
static inline sc_t mksc(int s) { return vdupq_n_s16(std::int16_t(s)); }
// int16 bounds per lane: 2 products of |w| <= 63, |a| <= 127 = 16002; 4 products of |w| <= 31 = 15748;
// 2 products of |w| <= 128 = 32512. All below 32767.
static inline int32x4_t dot16(int32x4_t acc, int8x16_t w, int8x16_t a, sc_t sc) {
    const int16x8_t p = vmlal_high_s8(vmull_s8(vget_low_s8(w), vget_low_s8(a)), w, a);
    return vmlal_high_s16(vmlal_s16(acc, vget_low_s16(p), vget_low_s16(sc)), p, sc);
}
static inline int32x4_t dot32(int32x4_t acc, int8x16_t w0, int8x16_t w1, int8x16_t a0, int8x16_t a1, sc_t sc) {
    int16x8_t p = vmlal_high_s8(vmull_s8(vget_low_s8(w0), vget_low_s8(a0)), w0, a0);
    p = vmlal_high_s8(vmlal_s8(p, vget_low_s8(w1), vget_low_s8(a1)), w1, a1);
    return vmlal_high_s16(vmlal_s16(acc, vget_low_s16(p), vget_low_s16(sc)), p, sc);
}
static inline int32x4_t sdot32(int8x16_t w0, int8x16_t w1, int8x16_t a0, int8x16_t a1) {
    const int16x8_t p0 = vmlal_high_s8(vmull_s8(vget_low_s8(w0), vget_low_s8(a0)), w0, a0);
    const int16x8_t p1 = vmlal_high_s8(vmull_s8(vget_low_s8(w1), vget_low_s8(a1)), w1, a1);
    return vpadalq_s16(vpaddlq_s16(p0), p1);
}
}
#undef QI8_ARM_PART
#define QI8_ARM_PART 2
#include "qi8_arm.inc"
namespace forge::simd::qi8_tables {
extern const qi8_kernels neon;
const qi8_kernels neon = {"neon", &qi8::neon::quantize, &qi8::neon::matvec};
}
#endif
