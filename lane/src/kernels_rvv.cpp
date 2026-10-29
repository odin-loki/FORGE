#include <forge/lane/algo.hpp>
#include <forge/lane/dispatch.hpp>
#if defined(__riscv)
#if !defined(FORGE_HAVE_RVV256)
#error "kernels_rvv.cpp must be compiled with V, zvl256b and -mrvv-vector-bits=256"
#endif
static_assert(forge::lane::vec<float, forge::lane::rvv1>::K == 5 && forge::lane::vec<float, forge::lane::rvv1>::N == 8);
FORGE_LANE_EXPORT_TABLE(mv_rvv, rvv1, "rvv")
namespace forge::lane::mv_rvv { int vector_bytes() { return int(__riscv_vsetvlmax_e8m1()); } }  // only called when HWCAP V is set
#endif
