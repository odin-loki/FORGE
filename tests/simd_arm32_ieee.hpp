#pragma once
// 32-bit Arm only: the operations of the Armv7 vec_gn.hpp backend against VFP arithmetic, lane by lane and bit
// for bit (NaN payloads apart). Advanced SIMD on AArch32 flushes subnormals to zero; the backend recomputes the
// lanes that touches with VFP instructions (docs/simd.md, "32-bit Arm"), and this is the check that it does:
// operands are weighted towards zero, subnormals, FLT_MIN, exact cancellations, infinity and NaN.
// Included by tests/simd_types_isa.cpp, which is compiled with the neon table's flags. The references are
// single scalar operations on volatile values, which the compiler emits as VFP instructions.
#include <forge/simd/vec_gn.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

namespace forge_arm32 {
namespace {
using namespace forge::simd;
#define FORGE_A32_NI [[maybe_unused]] __attribute__((noinline)) static
FORGE_A32_NI gn::vf v_add(gn::vf a, gn::vf b) { return gn::add(a, b); }
FORGE_A32_NI gn::vf v_sub(gn::vf a, gn::vf b) { return gn::sub(a, b); }
FORGE_A32_NI gn::vf v_mul(gn::vf a, gn::vf b) { return gn::mul(a, b); }
FORGE_A32_NI gn::vf v_fma(gn::vf a, gn::vf b, gn::vf c) { return gn::fma(a, b, c); }
FORGE_A32_NI gn::vm v_lt(gn::vf a, gn::vf b) { return gn::lt(a, b); }
FORGE_A32_NI gn::vf v_max(gn::vf a, gn::vf b) { return gn::max(a, b); }
FORGE_A32_NI gn::vf v_ld(gn::vf a, gn::vf b) { return gn::ldexp_int(a, b); }
FORGE_A32_NI gn::vf v_rn(gn::vf a) { return gn::round_nearest(a); }
FORGE_A32_NI gn::vf v_div(gn::vf a, gn::vf b) { return gn::div(a, b); }
FORGE_A32_NI gn::vf v_sqrt(gn::vf a) { return gn::sqrt(a); }
FORGE_A32_NI gn::vf v_splat(float a) { return gn::splat(a); }
FORGE_A32_NI float v_red(gn::vf a) { return gn::reduce_add(a); }
#undef FORGE_A32_NI
inline std::uint32_t bits(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float fb(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
inline bool same(float a, float b) { return bits(a) == bits(b) || (a != a && b != b); }

inline int ieee_check() {
    std::mt19937 rng(1);
    auto gen = [&]() -> float {
        std::uint32_t r = rng();
        switch (rng() % 12) {
        case 0: return fb(r & 0x807fffffu);                               // subnormal or zero
        case 1: return fb((r & 0x807fffffu) | ((rng() % 60) << 23));      // small exponents
        case 2: return rng() % 2 ? 0.f : -0.f;
        case 3: return fb((r & 0x807fffffu) | ((100 + rng() % 60) << 23));
        case 4: return fb((r & 0x80000000u) | 0x00800000u | (rng() % 4));   // just above FLT_MIN
        case 5: return fb((r & 0x80000000u) | (0x00800000u - 1 - rng() % 4));
        case 6: return rng() % 8 ? fb((r & 0x80000000u) | 0x7f800000u) : fb(r | 0x7f800000u);   // inf / NaN
        case 7: return fb((r & 0x80000000u) | ((127u + rng() % 3) << 23) | (rng() % 2 ? 0 : 1u << (rng() % 23)));
        case 8: return float(int(rng() % 400) - 200);
        default: return fb(r);
        }
    };
    long fails = 0, n = 0;
    auto bad = [&](const char* op, float a, float b, float c, float got, float want) {
        if (++fails < 20) std::fprintf(stderr, "  CHECK failed [neon] %s(%a, %a, %a) = %a, expected %a\n", op, double(a), double(b), double(c), double(got), double(want));
    };
    for (int it = 0; it < 300000; ++it) {
        float A[4], B[4], C[4];
        for (int i = 0; i < 4; ++i) { A[i] = gen(); B[i] = gen(); C[i] = gen(); }
        if (it % 3 == 0) for (int i = 0; i < 4; ++i) B[i] = rng() % 2 ? A[i] : -A[i];
        if (it % 5 == 0) for (int i = 0; i < 4; ++i) if (rng() % 2) { volatile float p = A[i] * B[i]; C[i] = -p; }
        gn::vf a = gn::load(A), b = gn::load(B), c = gn::load(C);
        gn::vf r1 = v_add(a, b), r2 = v_sub(a, b), r3 = v_mul(a, b), r4 = v_fma(a, b, c), r6 = v_max(a, b), r9 = v_div(a, b), r10 = v_sqrt(a);
        gn::vm r5 = v_lt(a, b);
        for (int i = 0; i < 4; ++i) {
            volatile float x = A[i], y = B[i], z = C[i];
            float e;
            e = x + y; if (!same(r1[i], e)) bad("add", x, y, 0, r1[i], e);
            e = x - y; if (!same(r2[i], e)) bad("sub", x, y, 0, r2[i], e);
            e = x * y; if (!same(r3[i], e)) bad("mul", x, y, 0, r3[i], e);
            e = std::fmaf(x, y, z); if (!same(r4[i], e)) bad("fma", x, y, z, r4[i], e);
            int l = x < y ? -1 : 0; if (r5[i] != l) bad("lt", x, y, 0, float(r5[i]), float(l));
            e = y < x ? x : y; if (!same(r6[i], e)) bad("max", x, y, 0, r6[i], e);
            e = x / y; if (!same(r9[i], e)) bad("div", x, y, 0, r9[i], e);
            e = std::sqrt(x); if (!same(r10[i], e)) bad("sqrt", x, 0, 0, r10[i], e);
            if (bits(v_splat(x)[i]) != bits(x)) bad("splat", x, 0, 0, v_splat(x)[i], x);
            n += 9;
        }
        { volatile float s0 = A[0], s1 = A[1], s2 = A[2], s3 = A[3]; volatile float t0 = s0 + s2, t1 = s1 + s3; float e = t0 + t1;
          float g = v_red(a); if (!same(g, e)) bad("reduce_add", s0, s1, s2, g, e); ++n; }
        // ldexp: x * 2^n for integer n in range, round to nearest of anything
        float N[4]; for (int i = 0; i < 4; ++i) N[i] = float(int(rng() % 250) - 125);
        gn::vf r7 = v_ld(a, gn::load(N)), r8 = v_rn(a);
        for (int i = 0; i < 4; ++i) {
            volatile float x = A[i];
            float e = std::ldexp(x, int(N[i]));
            if (N[i] > -127 && !same(r7[i], e)) bad("ldexp", x, N[i], 0, r7[i], e);
            e = std::nearbyintf(x); if (!same(r8[i], e)) bad("round", x, 0, 0, r8[i], e);
            n += 2;
        }
    }
    std::printf("  %-10s IEEE lanes against VFP (add sub mul fma lt max div sqrt splat reduce_add ldexp round): %ld checks, %ld failed\n", "neon", n, fails);
    return fails > 1000000 ? 1000000 : int(fails);
}
} // namespace
} // namespace forge_arm32
