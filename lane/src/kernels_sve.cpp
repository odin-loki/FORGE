#include <forge/lane/algo.hpp>
#include <forge/lane/dispatch.hpp>
#if defined(__aarch64__)
#if !defined(FORGE_HAVE_SVE256)
#error "kernels_sve.cpp must be compiled with SVE and -msve-vector-bits=256"
#endif
static_assert(forge::lane::vec<float, forge::lane::neoverse2>::K == 4 && forge::lane::vec<float, forge::lane::neoverse2>::N == 8);
FORGE_LANE_EXPORT_TABLE(mv_sve, neoverse2, "sve")
namespace forge::lane::mv_sve { int vector_bytes() { long v; __asm__ volatile("rdvl %0, #1" : "=r"(v)); return int(v); } }  // svcntb() folds to 32 under -msve-vector-bits; only called when HWCAP_SVE is set
#endif
