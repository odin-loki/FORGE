#include <forge/lane/algo.hpp>
#include <forge/lane/qdot.hpp>
#include <forge/lane/dispatch.hpp>
static_assert(forge::lane::vec<float, forge::lane::x86_64_v3>::K == 1, "avx2 TU must build native AVX2 kernels");
FORGE_LANE_EXPORT_TABLE(mv_avx2, x86_64_v3, "avx2")
