#include <forge/lane/algo.hpp>
#include <forge/lane/dispatch.hpp>
// Each target lives in its own TU, own flags, own namespace — no ODR leakage across ISAs.
FORGE_LANE_EXPORT_TABLE(mv_scalar, scalar_t, "scalar")
