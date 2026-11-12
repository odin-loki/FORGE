// Builds one tier_a_kernels table. The including file defines FORGE_TIERA_NS (namespace) and FORGE_TIERA_TG
// (target type) and FORGE_TIERA_NAME, and is compiled with that target's flags.
#include <forge/simd/algo_sort.hpp>
#include <forge/simd/algo_text.hpp>
#include <forge/simd/tier_a.hpp>

namespace forge::simd::FORGE_TIERA_NS {
using TG = FORGE_TIERA_TG;
namespace {
std::size_t count_f32(const float* x, std::size_t n, int c, float t) { return algo::count_f32<TG>(x, n, algo::cmp(c), t); }
std::size_t find_f32(const float* x, std::size_t n, int c, float t) { return algo::find_f32<TG>(x, n, algo::cmp(c), t); }
bool any_f32(const float* x, std::size_t n, int c, float t) { return algo::any_f32<TG>(x, n, algo::cmp(c), t); }
bool all_f32(const float* x, std::size_t n, int c, float t) { return algo::all_f32<TG>(x, n, algo::cmp(c), t); }
void bitset_op(int op, const std::uint64_t* a, const std::uint64_t* b, std::uint64_t* o, std::size_t n) { algo::bitset_op<TG>(algo::bitop(op), a, b, o, n); }
std::size_t bitset_find_first(const std::uint64_t* w, std::size_t n, std::size_t from) { return algo::bitset_find_first<TG>(w, n, from); }
}
extern const tier_a_kernels table;
const tier_a_kernels table = {
    FORGE_TIERA_NAME,
    &algo::min_f32<TG>, &algo::max_f32<TG>, &algo::argmin_f32<TG>, &algo::argmax_f32<TG>,
    &count_f32, &find_f32, &any_f32, &all_f32, &algo::clamp_f32<TG>,
    &algo::scan_add_f32<TG>, &algo::scan_add_f64<TG>, &algo::scan_add_i32<TG>,
    &algo::histogram_u8<TG>, &algo::histogram_f32<TG>,
    &algo::transpose<TG, float>, &algo::transpose<TG, double>, &algo::transpose_square<TG, float>, &algo::transpose_square<TG, double>,
    &algo::dot_f64<TG>, &algo::sum_f64<TG>, &algo::axpy_f64<TG>, &algo::dot_i32<TG>, &algo::dot_i16<TG>, &algo::dot_i8<TG>,
    &algo::axpy_i32<TG>, &algo::sum_i32<TG>, &algo::min_i32<TG>, &algo::max_i32<TG>, &algo::find_i32<TG>, &algo::count_eq_i32<TG>,
    &algo::popcount<TG>, &algo::popcount_and<TG>, &bitset_op, &bitset_find_first,
    &algo::find_byte<TG>, &algo::count_byte<TG>, &algo::mismatch<TG>, &algo::find_bytes<TG>,
    &algo::sort_f32<TG>, &algo::sort_i32<TG>, &algo::sort_u32<TG>, &algo::sort_small<TG, float>, &algo::sort_small<TG, std::int32_t>,
    &algo::partial_sort_f32<TG>, &algo::select_f32<TG>, &algo::topk_f32<TG>,
    &algo::utf8_error<TG>, &algo::hex_encode<TG>, &algo::hex_decode<TG>, &algo::base64_encode<TG>, &algo::base64_decode<TG>,
};
} // namespace forge::simd::FORGE_TIERA_NS
