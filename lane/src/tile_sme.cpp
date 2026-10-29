// Arm SME f32 GEMM with FMOPA into ZA tile 0. Built with SME enabled (-march=armv9-a+sme) on aarch64 when the
// compiler supports the ACLE SME attributes; validated under qemu-aarch64 -cpu max (which implements SME).
#include <forge/lane/tile.hpp>
#include <algorithm>
#include <vector>
#if defined(__aarch64__) && defined(FORGE_HAVE_SME)
#include <arm_sme.h>
#if defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP2_SME
#define HWCAP2_SME (1UL << 23)
#endif
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace forge::lane::tile {
bool sme_available() {
    static const bool ok = [] {
#if defined(__linux__)
        return (getauxval(AT_HWCAP2) & HWCAP2_SME) != 0;
#elif defined(__APPLE__)
        int v = 0; std::size_t l = sizeof v; return sysctlbyname("hw.optional.arm.FEAT_SME", &v, &l, nullptr, 0) == 0 && v;
#else
        return false;
#endif
    }();
    return ok;
}
namespace {
std::size_t svl_bytes() __arm_streaming_compatible { return std::size_t(svcntsb()); }
// One VLxVL block: C[i0.., j0..] = sum_p Acol(p) (x) Bcol(p). At/Bt are k-major panels (column p contiguous, VL floats).
__arm_new("za") void block(const float* At, const float* Bt, float* C, std::size_t ldc, std::size_t mr, std::size_t nr, std::size_t k) __arm_streaming {
    const std::size_t vl = svcntw();
    svbool_t pa = svwhilelt_b32_u64(0, mr), pb = svwhilelt_b32_u64(0, nr);
    svzero_za();
    for (std::size_t p = 0; p < k; ++p) {
        svfloat32_t a = svld1_f32(pa, At + p * vl), b = svld1_f32(pb, Bt + p * vl);
        svmopa_za32_f32_m(0, pa, pb, a, b);
    }
    for (std::size_t r = 0; r < mr; ++r) svst1_hor_za32(0, uint32_t(r), pb, C + r * ldc);
}
void run(const float* At, const float* Bt, float* C, std::size_t ldc, std::size_t mr, std::size_t nr, std::size_t k) __arm_streaming { block(At, Bt, C, ldc, mr, nr, k); }
}
std::size_t sme_vl_bytes() { return sme_available() ? svl_bytes() : 0; }
bool sme_gemm_f32(const float* A, const float* B, float* C, std::size_t m, std::size_t n, std::size_t k) {
    if (!sme_available()) return false;
    const std::size_t vl = svl_bytes() / 4;
    std::vector<float> at(k * vl), bt(k * vl);
    for (std::size_t i0 = 0; i0 < m; i0 += vl) {
        std::size_t mr = std::min(vl, m - i0);
        std::fill(at.begin(), at.end(), 0.f);
        for (std::size_t r = 0; r < mr; ++r) for (std::size_t p = 0; p < k; ++p) at[p * vl + r] = A[(i0 + r) * k + p];
        for (std::size_t j0 = 0; j0 < n; j0 += vl) {
            std::size_t nr = std::min(vl, n - j0);
            std::fill(bt.begin(), bt.end(), 0.f);
            for (std::size_t c = 0; c < nr; ++c) for (std::size_t p = 0; p < k; ++p) bt[p * vl + c] = B[(j0 + c) * k + p];
            run(at.data(), bt.data(), C + i0 * n + j0, n, mr, nr, k);
        }
    }
    return true;
}
} // namespace forge::lane::tile
#else
namespace forge::lane::tile {
bool sme_available() { return false; }
std::size_t sme_vl_bytes() { return 0; }
bool sme_gemm_f32(const float*, const float*, float*, std::size_t, std::size_t, std::size_t) { return false; }
} // namespace forge::lane::tile
#endif
