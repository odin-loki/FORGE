// tier_a_kernels dispatch: the same target names and usability rules as the lane kernel tables (dispatch.cpp).
#include <forge/simd/dispatch.hpp>
#include <forge/simd/tier_a.hpp>
#include <cstdlib>
#include <string_view>

namespace forge::simd {
namespace tiera_scalar { extern const tier_a_kernels table; }
#if defined(__x86_64__) || defined(_M_X64)
namespace tiera_avx2 { extern const tier_a_kernels table; }
namespace tiera_avx512 { extern const tier_a_kernels table; }
#endif

const tier_a_kernels* tier_a_by_name(const char* t) {
    std::string_view n = t ? t : "";
    if (n == "scalar") return &tiera_scalar::table;
#if defined(__x86_64__) || defined(_M_X64)
    if (n == "avx2") return by_name("avx2") ? &tiera_avx2::table : nullptr;
    if (n == "avx512") return by_name("avx512") ? &tiera_avx512::table : nullptr;
#endif
    return nullptr;
}

const tier_a_kernels& tier_a_active() {
    static const tier_a_kernels* k = [] {
        for (const char* var : {"FORGE_SIMD_TARGET", "FORGE_LANE_TARGET"})
            if (const char* e = std::getenv(var))
                if (auto* p = tier_a_by_name(e)) return p;
        for (const char* t : {"avx512", "avx2"}) if (auto* p = tier_a_by_name(t)) return p;
        return &tiera_scalar::table;
    }();
    return *k;
}

} // namespace forge::simd
