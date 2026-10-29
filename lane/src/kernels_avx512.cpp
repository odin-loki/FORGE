#include <forge/lane/algo.hpp>
#include <forge/lane/dispatch.hpp>
static_assert(forge::lane::vec<float, forge::lane::x86_64_v4>::K == 2, "avx512 TU must build native AVX-512 kernels");
FORGE_LANE_EXPORT_TABLE(mv_avx512, x86_64_v4, "avx512")
