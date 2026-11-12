#pragma once
// Interop with the standard library's data-parallel types and multidimensional views (FORGE_PLAN A.8).
//
//   std::experimental::simd   to_std(v) / from_std<Tg>(s) and the mask forms, when the library ships
//                             <experimental/simd> (__cpp_lib_experimental_parallel_simd: libstdc++ 11 and later).
//                             FORGE_SIMD_HAVE_STD_SIMD is 1 when they exist.
//   std::mdspan               forge::simd::md::* overloads of the Tier A algorithms (tier_a.hpp) over
//                             std::mdspan, when the library has <mdspan> (__cpp_lib_mdspan: libc++ 18 and
//                             later, libstdc++ 15 and later). FORGE_SIMD_HAVE_MDSPAN is 1 when they exist.
//
// C++26 std::simd (<simd>, __cpp_lib_simd) has no conversions here yet: no standard library available to this
// repository ships it, so nothing could be compiled against it.
//
// Conversions go through memory (store + load of one register), which compilers fold away: the two types are
// the same register when the ABI the library deduces for N lanes is the native one.
#include <forge/simd/tier_a.hpp>
#include <forge/simd/vec_types.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <version>
#if __has_include(<experimental/simd>)
#include <experimental/simd>
#endif
#if defined(__cpp_lib_mdspan)
#include <mdspan>
#endif

#if defined(__cpp_lib_experimental_parallel_simd)
#define FORGE_SIMD_HAVE_STD_SIMD 1
namespace forge::simd {

namespace stdx = std::experimental;
// The std::experimental::simd with as many lanes as vec<T, Tg> (the native ABI for that count when the
// translation unit has one, a fixed_size ABI otherwise).
template <class T, Target Tg> using std_simd = stdx::simd<T, stdx::simd_abi::deduce_t<T, std::size_t(vec<T, Tg>::N)>>;
template <class T, Target Tg> using std_simd_mask = typename std_simd<T, Tg>::mask_type;

template <Target Tg> inline std_simd<float, Tg> to_std(vec<float, Tg> v) {
    float t[vec<float, Tg>::N]; store(t, v);
    return std_simd<float, Tg>(t, stdx::element_aligned);
}
template <vt::elem T, Target Tg> inline std_simd<T, Tg> to_std(vec<T, Tg> v) {
    T t[vec<T, Tg>::N]; store(t, v);
    return std_simd<T, Tg>(t, stdx::element_aligned);
}
// from_std<Tg>(s): s must have vec<T, Tg>::N lanes
template <Target Tg, class Abi> requires (stdx::simd<float, Abi>::size() == std::size_t(vec<float, Tg>::N))
inline vec<float, Tg> from_std(const stdx::simd<float, Abi>& s) {
    float t[vec<float, Tg>::N]; s.copy_to(t, stdx::element_aligned);
    return load<Tg>(t);
}
template <Target Tg, vt::elem T, class Abi> requires (stdx::simd<T, Abi>::size() == std::size_t(vec<T, Tg>::N))
inline vec<T, Tg> from_std(const stdx::simd<T, Abi>& s) {
    T t[vec<T, Tg>::N]; s.copy_to(t, stdx::element_aligned);
    return vec<T, Tg>::load(t);
}
// masks (vec_types.hpp element types; mask<float> has no lane accessor)
template <vt::elem T, Target Tg> inline std_simd_mask<T, Tg> to_std(mask<T, Tg> m) {
    std_simd_mask<T, Tg> r;
    for (int i = 0; i < mask<T, Tg>::N; ++i) r[std::size_t(i)] = m[i];
    return r;
}
template <Target Tg, vt::elem T, class Abi> requires (stdx::simd_mask<T, Abi>::size() == std::size_t(mask<T, Tg>::N))
inline mask<T, Tg> from_std(const stdx::simd_mask<T, Abi>& s) {
    typename mask<T, Tg>::lane_t t[mask<T, Tg>::N];
    for (int i = 0; i < mask<T, Tg>::N; ++i) t[i] = s[std::size_t(i)] ? typename mask<T, Tg>::lane_t(-1) : typename mask<T, Tg>::lane_t(0);
    mask<T, Tg> r; vt::from_arr(r.m, t); return r;
}

} // namespace forge::simd
#else
#define FORGE_SIMD_HAVE_STD_SIMD 0
#endif

#if defined(__cpp_lib_mdspan)
#define FORGE_SIMD_HAVE_MDSPAN 1
// Tier A over std::mdspan, through the active table (tier_a_active()). A view is used in place when its
// elements are contiguous in index order (layout_right with every extent used, or rank 1 with stride 1);
// any other layout (layout_stride with gaps, layout_left of rank > 1) is copied to a temporary first, and
// copied back for the functions that write. Functions named like the table entries; rank-N views are treated
// as the sequence of their elements in row-major order unless stated.
namespace forge::simd::md {

namespace detail {
template <class M> inline bool contiguous(const M& m) {
    if constexpr (M::rank() == 0) return true;
    else if constexpr (M::rank() == 1) return m.extent(0) == 0 || m.stride(0) == 1;
    else {
        if (!m.is_exhaustive()) return false;
        // row-major order: the last index is the fastest
        std::size_t want = 1;
        for (std::size_t d = M::rank(); d-- > 0;) { if (std::size_t(m.stride(d)) != want) return false; want *= std::size_t(m.extent(d)); }
        return true;
    }
}
template <class M, class F> inline void for_each_index(const M& m, F&& f) {   // row-major
    const std::size_t n = std::size_t(m.size());
    if constexpr (M::rank() == 1) { for (std::size_t i = 0; i < n; ++i) f(m[typename M::index_type(i)]); }
    else if constexpr (M::rank() == 2) {
        for (std::size_t i = 0; i < std::size_t(m.extent(0)); ++i) for (std::size_t j = 0; j < std::size_t(m.extent(1)); ++j) f(m[typename M::index_type(i), typename M::index_type(j)]);
    } else {
        static_assert(M::rank() == 3, "forge::simd::md: non-contiguous views are supported up to rank 3");
        for (std::size_t i = 0; i < std::size_t(m.extent(0)); ++i) for (std::size_t j = 0; j < std::size_t(m.extent(1)); ++j)
            for (std::size_t k = 0; k < std::size_t(m.extent(2)); ++k) f(m[typename M::index_type(i), typename M::index_type(j), typename M::index_type(k)]);
    }
}
// read access: the view's own storage, or a row-major copy
template <class M> struct in {
    using T = std::remove_const_t<typename M::element_type>;
    std::vector<T> tmp; const T* p; std::size_t n;
    explicit in(const M& m) : n(std::size_t(m.size())) {
        if (contiguous(m)) { p = m.data_handle(); return; }
        tmp.reserve(n); for_each_index(m, [&](const T& x) { tmp.push_back(x); }); p = tmp.data();
    }
};
// write access: as in, plus copy-back on destruction when a temporary was needed
template <class M> struct io {
    using T = typename M::element_type;
    M m; std::vector<T> tmp; T* p; std::size_t n;
    explicit io(const M& v) : m(v), n(std::size_t(v.size())) {
        if (contiguous(m)) { p = m.data_handle(); return; }
        tmp.reserve(n); for_each_index(m, [&](const T& x) { tmp.push_back(x); }); p = tmp.data();
    }
    ~io() { if (!tmp.empty()) { std::size_t i = 0; for_each_index(m, [&](T& x) { x = tmp[i++]; }); } }
    io(const io&) = delete; io& operator=(const io&) = delete;
};
} // namespace detail

template <class M, class T> concept view_of = requires { typename M::element_type; } && std::same_as<std::remove_const_t<typename M::element_type>, T> &&
    std::same_as<M, std::mdspan<typename M::element_type, typename M::extents_type, typename M::layout_type, typename M::accessor_type>> &&
    std::same_as<typename M::accessor_type, std::default_accessor<typename M::element_type>>;
template <class M, class T> concept mutable_view_of = view_of<M, T> && !std::is_const_v<typename M::element_type>;

// ---- f32 reductions and compares (NaN handling as in tier_a.hpp) ----
template <view_of<float> M> float min(const M& x) { detail::in<M> a(x); return tier_a_active().min_f32(a.p, a.n); }
template <view_of<float> M> float max(const M& x) { detail::in<M> a(x); return tier_a_active().max_f32(a.p, a.n); }
template <view_of<float> M> std::size_t argmin(const M& x) { detail::in<M> a(x); return tier_a_active().argmin_f32(a.p, a.n); }   // row-major position
template <view_of<float> M> std::size_t argmax(const M& x) { detail::in<M> a(x); return tier_a_active().argmax_f32(a.p, a.n); }
// cmp: 0 eq, 1 ne, 2 lt, 3 le, 4 gt, 5 ge (algo::cmp)
template <view_of<float> M> std::size_t count(const M& x, int cmp, float t) { detail::in<M> a(x); return tier_a_active().count_f32(a.p, a.n, cmp, t); }
template <view_of<float> M> std::size_t find(const M& x, int cmp, float t) { detail::in<M> a(x); return tier_a_active().find_f32(a.p, a.n, cmp, t); }
template <view_of<float> M> bool any(const M& x, int cmp, float t) { detail::in<M> a(x); return tier_a_active().any_f32(a.p, a.n, cmp, t); }
template <view_of<float> M> bool all(const M& x, int cmp, float t) { detail::in<M> a(x); return tier_a_active().all_f32(a.p, a.n, cmp, t); }
// y = clamp(x, lo, hi), element by element in row-major order (x and y hold the same number of elements)
template <view_of<float> X, mutable_view_of<float> Y> void clamp(const X& x, const Y& y, float lo, float hi) {
    detail::in<X> a(x); detail::io<Y> b(y);
    tier_a_active().clamp_f32(a.p, b.p, a.n < b.n ? a.n : b.n, lo, hi);
}
// ---- prefix sums (rank 1 or row-major) ----
template <view_of<float> X, mutable_view_of<float> Y> void scan_add(const X& x, const Y& y, bool exclusive = false) {
    detail::in<X> a(x); detail::io<Y> b(y); tier_a_active().scan_add_f32(a.p, b.p, a.n < b.n ? a.n : b.n, exclusive);
}
template <view_of<double> X, mutable_view_of<double> Y> void scan_add(const X& x, const Y& y, bool exclusive = false) {
    detail::in<X> a(x); detail::io<Y> b(y); tier_a_active().scan_add_f64(a.p, b.p, a.n < b.n ? a.n : b.n, exclusive);
}
template <view_of<std::int32_t> X, mutable_view_of<std::int32_t> Y> void scan_add(const X& x, const Y& y, bool exclusive = false) {
    detail::in<X> a(x); detail::io<Y> b(y); tier_a_active().scan_add_i32(a.p, b.p, a.n < b.n ? a.n : b.n, exclusive);
}
// ---- f64 / integer BLAS-1 ----
template <view_of<double> M> double sum(const M& x) { detail::in<M> a(x); return tier_a_active().sum_f64(a.p, a.n); }
template <view_of<std::int32_t> M> std::int64_t sum(const M& x) { detail::in<M> a(x); return tier_a_active().sum_i32(a.p, a.n); }
template <view_of<double> X, view_of<double> Y> double dot(const X& x, const Y& y) { detail::in<X> a(x); detail::in<Y> b(y); return tier_a_active().dot_f64(a.p, b.p, a.n < b.n ? a.n : b.n); }
template <view_of<std::int32_t> X, view_of<std::int32_t> Y> std::int64_t dot(const X& x, const Y& y) { detail::in<X> a(x); detail::in<Y> b(y); return tier_a_active().dot_i32(a.p, b.p, a.n < b.n ? a.n : b.n); }
template <view_of<std::int16_t> X, view_of<std::int16_t> Y> std::int64_t dot(const X& x, const Y& y) { detail::in<X> a(x); detail::in<Y> b(y); return tier_a_active().dot_i16(a.p, b.p, a.n < b.n ? a.n : b.n); }
template <view_of<std::int8_t> X, view_of<std::int8_t> Y> std::int64_t dot(const X& x, const Y& y) { detail::in<X> a(x); detail::in<Y> b(y); return tier_a_active().dot_i8(a.p, b.p, a.n < b.n ? a.n : b.n); }
template <view_of<std::int32_t> M> std::int32_t min(const M& x) { detail::in<M> a(x); return tier_a_active().min_i32(a.p, a.n); }
template <view_of<std::int32_t> M> std::int32_t max(const M& x) { detail::in<M> a(x); return tier_a_active().max_i32(a.p, a.n); }
template <view_of<std::int32_t> M> std::size_t find(const M& x, std::int32_t v) { detail::in<M> a(x); return tier_a_active().find_i32(a.p, a.n, v); }
// ---- bytes ----
template <view_of<std::uint8_t> M> std::size_t find_byte(const M& x, std::uint8_t c) { detail::in<M> a(x); return tier_a_active().find_byte(a.p, a.n, c); }
template <view_of<std::uint8_t> M> std::size_t count_byte(const M& x, std::uint8_t c) { detail::in<M> a(x); return tier_a_active().count_byte(a.p, a.n, c); }
template <view_of<std::uint8_t> M> void histogram(const M& x, std::uint32_t* hist256) { detail::in<M> a(x); tier_a_active().histogram_u8(a.p, a.n, hist256); }
// ---- sorting (the view's elements in row-major order; a rank-1 view is the usual case) ----
template <mutable_view_of<float> M> void sort(const M& x) { detail::io<M> a(x); tier_a_active().sort_f32(a.p, a.n); }
template <mutable_view_of<std::int32_t> M> void sort(const M& x) { detail::io<M> a(x); tier_a_active().sort_i32(a.p, a.n); }
template <mutable_view_of<std::uint32_t> M> void sort(const M& x) { detail::io<M> a(x); tier_a_active().sort_u32(a.p, a.n); }
template <mutable_view_of<float> M> void partial_sort(const M& x, std::size_t k) { detail::io<M> a(x); tier_a_active().partial_sort_f32(a.p, a.n, k); }
template <view_of<float> M> float select(const M& x, std::size_t k) { detail::in<M> a(x); return tier_a_active().select_f32(a.p, a.n, k); }
// indices (row-major positions) and values of the k largest; idx and val have room for min(k, size)
template <view_of<float> M> std::size_t topk(const M& x, std::size_t k, std::uint32_t* idx, float* val) { detail::in<M> a(x); return tier_a_active().topk_f32(a.p, a.n, k, idx, val); }
// Rows of a rank-2 view: out[r] = index of the first maximum of row r (the extent(1) when the row is all NaN).
template <view_of<float> M> requires (M::rank() == 2) void argmax_rows(const M& x, std::size_t* out) {
    const std::size_t rows = std::size_t(x.extent(0)), cols = std::size_t(x.extent(1));
    detail::in<M> a(x);
    for (std::size_t r = 0; r < rows; ++r) out[r] = tier_a_active().argmax_f32(a.p + r * cols, cols);
}
// ---- transpose: b[j, i] = a[i, j] for rank-2 views (b has the extents swapped) ----
template <view_of<float> A, mutable_view_of<float> B> requires (A::rank() == 2 && B::rank() == 2) void transpose(const A& a, const B& b) {
    detail::in<A> x(a); detail::io<B> y(b);
    tier_a_active().transpose_f32(x.p, y.p, std::size_t(a.extent(0)), std::size_t(a.extent(1)));
}
template <view_of<double> A, mutable_view_of<double> B> requires (A::rank() == 2 && B::rank() == 2) void transpose(const A& a, const B& b) {
    detail::in<A> x(a); detail::io<B> y(b);
    tier_a_active().transpose_f64(x.p, y.p, std::size_t(a.extent(0)), std::size_t(a.extent(1)));
}

} // namespace forge::simd::md
#else
#define FORGE_SIMD_HAVE_MDSPAN 0
#endif
