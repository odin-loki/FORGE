// Minimal reproductions, without any FORGE code, of the code-generation defects found while running
// tests/test_simd_types.cpp under qemu-user with zig 0.17's clang (docs/simd.md, "Element types"):
//   loongarch64 (baseline, -O2): a 16-byte struct of int8 passed by value comes back with lane 9 wrong
//                                ("shr lane 9: 0 (want 1)")
//   mips64el -mmsa:              __builtin_elementwise_fma on 2 x double computes a + b * c
// vec_types.hpp works around the MSA one with the intrinsic; the LoongArch one only affects the portable-array
// implementation (emulated_t), which no LoongArch build uses.
//   zig c++ -std=c++23 -O2 -static -target loongarch64-linux-musl tools/repro_llvm_vector_codegen.cpp && qemu-loongarch64 -cpu la464 ./a.out
//   zig c++ -std=c++23 -O2 -target mips64el-linux-muslabi64 -mmsa -c tools/repro_llvm_vector_codegen.cpp -o r.o   (then link without -mmsa)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

struct arr { std::int8_t v[16]; };
__attribute__((noinline)) arr band(arr a, arr b) { arr o; for (int i = 0; i < 16; ++i) o.v[i] = std::int8_t(a.v[i] & b.v[i]); return o; }
__attribute__((noinline)) arr shr(arr a, arr b) { arr o; for (int i = 0; i < 16; ++i) o.v[i] = std::int8_t(a.v[i] >> int(b.v[i])); return o; }
__attribute__((noinline)) arr shr_masked(arr a, arr b, arr m) { return shr(a, band(b, m)); }

typedef double vd __attribute__((vector_size(16)));
typedef std::uint64_t vu __attribute__((vector_size(16)));
typedef int vi __attribute__((vector_size(16)));
__attribute__((noinline)) vd vfma(vd a, vd b, vd c) { return __builtin_elementwise_fma(a, b, c); }
__attribute__((noinline)) vd vabs(vd a) { return (vd)((vu)a & 0x7fffffffffffffffull); }
__attribute__((noinline)) vd vsqrtabs(vd a) { return __builtin_elementwise_sqrt(vabs(a)); }
__attribute__((noinline)) vi vlt(vi a, vi b) { return a < b; }

int main() {
    int bad = 0;
    arr a, b, m;
    for (int i = 0; i < 16; ++i) { a.v[i] = 2; b.v[i] = std::int8_t(9 + 16 * (i % 7)); m.v[i] = 7; }
    arr o = shr_masked(a, b, m);
    for (int i = 0; i < 16; ++i) if (o.v[i] != 1) { std::printf("shr lane %d: %d (want 1)\n", i, o.v[i]); ++bad; }
    vd x = {3.0, -5.0}, y = {7.0, 11.0}, z = {1.0, 2.0};
    vd f = vfma(x, y, z);
    for (int i = 0; i < 2; ++i) if (f[i] != std::fma(x[i], y[i], z[i])) { std::printf("fma lane %d: %g (want %g)\n", i, f[i], std::fma(x[i], y[i], z[i])); ++bad; }
    vd s = vsqrtabs(vd{-16.0, 9.0});
    if (s[0] != 4.0 || s[1] != 3.0) { std::printf("sqrt(abs): %g %g (want 4 3)\n", s[0], s[1]); ++bad; }
    vi c = vlt(vi{1, 5, 3, 0}, vi{2, 4, 3, 7});
    int t[4]; std::memcpy(t, &c, 16);
    if (t[0] != -1 || t[1] != 0 || t[2] != 0 || t[3] != -1) { std::printf("compare lanes: %d %d %d %d (want -1 0 0 -1)\n", t[0], t[1], t[2], t[3]); ++bad; }
    std::printf("%d bad\n", bad);
    return bad != 0;
}
