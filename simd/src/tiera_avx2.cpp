// tier_a_kernels table: AVX2 + FMA.
#if defined(__x86_64__) || defined(_M_X64)
#define FORGE_TIERA_NS tiera_avx2
#define FORGE_TIERA_TG x86_64_v3
#define FORGE_TIERA_NAME "avx2"
#include "tiera_impl.inc"
static_assert(forge::simd::vec<float, forge::simd::x86_64_v3>::K == 1, "tiera_avx2.cpp must be compiled with AVX2 + FMA");
#endif
