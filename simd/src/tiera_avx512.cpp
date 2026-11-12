// tier_a_kernels table: AVX-512 F/BW/DQ/VL.
#if defined(__x86_64__) || defined(_M_X64)
#define FORGE_TIERA_NS tiera_avx512
#define FORGE_TIERA_TG x86_64_v4
#define FORGE_TIERA_NAME "avx512"
#include "tiera_impl.inc"
static_assert(forge::simd::vec<float, forge::simd::x86_64_v4>::K == 2, "tiera_avx512.cpp must be compiled with AVX-512");
#endif
