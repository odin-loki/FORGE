// SVE kernel table at one fixed vector length. CMake compiles this file once per length
// (-msve-vector-bits=128/256/512/1024); each copy exports forge::lane::mv_sve<B>::table.
#include <forge/lane/algo.hpp>
#include <forge/lane/dispatch.hpp>
#if defined(__aarch64__)
#if !defined(FORGE_HAVE_SVE_FIXED)
#error "kernels_sve.cpp must be compiled with SVE and -msve-vector-bits=<128..1024>"
#endif
#define FORGE_CAT2(a, b) a##b
#define FORGE_CAT(a, b) FORGE_CAT2(a, b)
using forge_sve_tag = forge::lane::sve_vl<FORGE_HAVE_SVE_FIXED>;
static_assert(forge::lane::vec<float, forge_sve_tag>::K == 4 && forge::lane::vec<float, forge_sve_tag>::N == FORGE_HAVE_SVE_FIXED / 32);
FORGE_LANE_EXPORT_TABLE(FORGE_CAT(mv_sve, __ARM_FEATURE_SVE_BITS), forge_sve_tag, "sve")
#endif
