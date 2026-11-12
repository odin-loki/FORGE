#pragma once
// Run-time dispatch for the Tier A algorithms of algo_num.hpp, algo_search.hpp, algo_sort.hpp and
// algo_text.hpp: one table of function pointers per target, built in simd/src/tiera_*.cpp with that target's
// compiler flags. Tables: "scalar" everywhere (compiled with the baseline flags, so on aarch64 the compiler
// still uses NEON); "avx2" and "avx512" on x86_64. There are no SVE or RVV tables for these algorithms.
//
//   const auto& a = forge::simd::tier_a_active();          // FORGE_SIMD_TARGET picks one, like simd::active()
//   std::size_t at = a.find_byte(p, n, '\n');
//
// The templates themselves (forge::simd::algo::find_byte<Tg> ...) can also be instantiated directly, but
// only in a translation unit compiled for that target.
#include <cstddef>
#include <cstdint>

namespace forge::simd {

struct tier_a_kernels {
    const char* target;
    // f32 reductions and compares (cmp: 0 eq, 1 ne, 2 lt, 3 le, 4 gt, 5 ge; see algo::cmp)
    float (*min_f32)(const float*, std::size_t);
    float (*max_f32)(const float*, std::size_t);
    std::size_t (*argmin_f32)(const float*, std::size_t);
    std::size_t (*argmax_f32)(const float*, std::size_t);
    std::size_t (*count_f32)(const float*, std::size_t, int cmp, float t);
    std::size_t (*find_f32)(const float*, std::size_t, int cmp, float t);
    bool (*any_f32)(const float*, std::size_t, int cmp, float t);
    bool (*all_f32)(const float*, std::size_t, int cmp, float t);
    void (*clamp_f32)(const float* x, float* y, std::size_t n, float lo, float hi);
    // prefix sums
    void (*scan_add_f32)(const float* x, float* y, std::size_t n, bool exclusive);
    void (*scan_add_f64)(const double* x, double* y, std::size_t n, bool exclusive);
    void (*scan_add_i32)(const std::int32_t* x, std::int32_t* y, std::size_t n, bool exclusive);
    // histograms
    void (*histogram_u8)(const std::uint8_t* x, std::size_t n, std::uint32_t* hist256);
    std::size_t (*histogram_f32)(const float* x, std::size_t n, float lo, float hi, std::uint32_t* hist, std::size_t bins);
    // transpose
    void (*transpose_f32)(const float* a, float* b, std::size_t rows, std::size_t cols);
    void (*transpose_f64)(const double* a, double* b, std::size_t rows, std::size_t cols);
    void (*transpose_square_f32)(float* a, std::size_t n);
    void (*transpose_square_f64)(double* a, std::size_t n);
    // f64 and integer BLAS-1
    double (*dot_f64)(const double*, const double*, std::size_t);
    double (*sum_f64)(const double*, std::size_t);
    void (*axpy_f64)(double a, const double* x, double* y, std::size_t n);
    std::int64_t (*dot_i32)(const std::int32_t*, const std::int32_t*, std::size_t);
    std::int64_t (*dot_i16)(const std::int16_t*, const std::int16_t*, std::size_t);
    std::int64_t (*dot_i8)(const std::int8_t*, const std::int8_t*, std::size_t);
    void (*axpy_i32)(std::int32_t a, const std::int32_t* x, std::int32_t* y, std::size_t n);
    std::int64_t (*sum_i32)(const std::int32_t*, std::size_t);
    std::int32_t (*min_i32)(const std::int32_t*, std::size_t);
    std::int32_t (*max_i32)(const std::int32_t*, std::size_t);
    std::size_t (*find_i32)(const std::int32_t*, std::size_t, std::int32_t v);
    std::size_t (*count_eq_i32)(const std::int32_t*, std::size_t, std::int32_t v);
    // bitsets (op: 0 and, 1 or, 2 xor, 3 and-not; see algo::bitop)
    std::uint64_t (*popcount)(const std::uint64_t*, std::size_t);
    std::uint64_t (*popcount_and)(const std::uint64_t*, const std::uint64_t*, std::size_t);
    void (*bitset_op)(int op, const std::uint64_t* a, const std::uint64_t* b, std::uint64_t* out, std::size_t n);
    std::size_t (*bitset_find_first)(const std::uint64_t*, std::size_t n, std::size_t from);
    // bytes
    std::size_t (*find_byte)(const std::uint8_t*, std::size_t, std::uint8_t c);
    std::size_t (*count_byte)(const std::uint8_t*, std::size_t, std::uint8_t c);
    std::size_t (*mismatch)(const std::uint8_t*, const std::uint8_t*, std::size_t);
    std::size_t (*find_bytes)(const std::uint8_t* hay, std::size_t n, const std::uint8_t* needle, std::size_t m);
    // sorting
    void (*sort_f32)(float*, std::size_t);
    void (*sort_i32)(std::int32_t*, std::size_t);
    void (*sort_u32)(std::uint32_t*, std::size_t);
    void (*sort_small_f32)(float*, std::size_t);           // network; no NaN
    void (*sort_small_i32)(std::int32_t*, std::size_t);
    void (*partial_sort_f32)(float*, std::size_t n, std::size_t k);
    float (*select_f32)(const float*, std::size_t n, std::size_t k);
    std::size_t (*topk_f32)(const float* x, std::size_t n, std::size_t k, std::uint32_t* idx, float* val);
    // text
    std::size_t (*utf8_error)(const std::uint8_t*, std::size_t);
    void (*hex_encode)(const std::uint8_t* in, std::size_t n, char* out, bool upper);
    bool (*hex_decode)(const char* in, std::size_t n, std::uint8_t* out);
    std::size_t (*base64_encode)(const std::uint8_t* in, std::size_t n, char* out, bool url);
    std::ptrdiff_t (*base64_decode)(const char* in, std::size_t n, std::uint8_t* out, bool url);
};

const tier_a_kernels& tier_a_active();                 // FORGE_SIMD_TARGET / FORGE_LANE_TARGET, else the best usable table
const tier_a_kernels* tier_a_by_name(const char* t);   // nullptr when not built or not usable on this CPU

} // namespace forge::simd
