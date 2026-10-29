#include <forge/lane/algo.hpp>
#include <forge/lane/dispatch.hpp>
#if defined(__aarch64__)
static_assert(forge::lane::vec<float, forge::lane::apple_m>::K == 3, "neon TU must build native NEON kernels");
FORGE_LANE_EXPORT_TABLE(mv_neon, apple_m, "neon")
#endif
