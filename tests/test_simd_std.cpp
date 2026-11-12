// forge/simd/interop.hpp: conversions between forge::simd::vec and std::experimental::simd, and the std::mdspan
// overloads of the Tier A algorithms. Each half exists only where the standard library has the type
// (FORGE_SIMD_HAVE_STD_SIMD / FORGE_SIMD_HAVE_MDSPAN); the test prints which halves ran. With libstdc++ 13 that
// is the simd half; test_simd_std_libcxx (the same source built with -stdlib=libc++) runs the mdspan half.
#include "check.hpp"
#include <forge/simd/interop.hpp>
#include <forge/simd/dispatch.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

using namespace forge::simd;

#if defined(FORGE_TEST_OWN_TIERA_DISPATCH)   // the libc++ build links no kernel library: the scalar table is the active one
namespace forge::simd {
namespace tiera_scalar { extern const tier_a_kernels table; }
const tier_a_kernels& tier_a_active() { return tiera_scalar::table; }
}
#endif

TEST(interop_reports_what_the_library_has) {
    std::printf("  std::experimental::simd conversions: %s; std::mdspan overloads: %s\n", FORGE_SIMD_HAVE_STD_SIMD ? "yes" : "no (library has none)",
                FORGE_SIMD_HAVE_MDSPAN ? "yes" : "no (library has no <mdspan>)");
#if defined(_GLIBCXX_RELEASE)
    std::printf("  libstdc++ %d\n", _GLIBCXX_RELEASE);
#elif defined(_LIBCPP_VERSION)
    std::printf("  libc++ %d\n", _LIBCPP_VERSION);
#endif
}

#if FORGE_SIMD_HAVE_STD_SIMD
template <class T, Target Tg> static void roundtrip() {
    using V = vec<T, Tg>;
    constexpr int N = V::N;
    T a[N], b[N], o[N];
    for (int i = 0; i < N; ++i) { a[i] = T(3 * i + 1); b[i] = T(100 - 7 * i); }
    const V va = V::load(a), vb = V::load(b);
    auto sa = to_std(va), sb = to_std(vb);
    static_assert(std::is_same_v<decltype(sa), std_simd<T, Tg>>);
    CHECK(sa.size() == std::size_t(N));
    for (int i = 0; i < N; ++i) CHECK(sa[std::size_t(i)] == a[i]);
    // arithmetic done by the library, brought back
    store(o, from_std<Tg>(sa + sb));
    for (int i = 0; i < N; ++i) CHECK(o[i] == T(a[i] + b[i]));
    // and the other way round: forge arithmetic, compared in the library
    CHECK(stdx::all_of(to_std(va + vb) == sa + sb));
    // masks
    const auto m = lt(va, vb);
    const auto sm = to_std(m);
    for (int i = 0; i < N; ++i) CHECK(sm[std::size_t(i)] == (a[i] < b[i]));
    CHECK(stdx::popcount(sm) == m.count());
    const auto back = from_std<Tg>(sa < sb);
    for (int i = 0; i < N; ++i) CHECK(back[i] == (a[i] < b[i]));
    store(o, select(back, va, vb));
    for (int i = 0; i < N; ++i) CHECK(o[i] == (a[i] < b[i] ? a[i] : b[i]));
}
TEST(std_experimental_simd_roundtrip) {
    roundtrip<std::int8_t, scalar_t>(); roundtrip<std::uint8_t, scalar_t>(); roundtrip<std::int16_t, scalar_t>(); roundtrip<std::uint16_t, scalar_t>();
    roundtrip<std::int32_t, scalar_t>(); roundtrip<std::uint32_t, scalar_t>(); roundtrip<std::int64_t, scalar_t>(); roundtrip<std::uint64_t, scalar_t>();
    roundtrip<double, scalar_t>(); roundtrip<std::int16_t, emulated_t>(); roundtrip<double, emulated_t>();
    // float (vec.hpp)
    constexpr int N = vec<float, scalar_t>::N;
    float a[N], o[N];
    for (int i = 0; i < N; ++i) a[i] = 0.5f * float(i) - 1.f;
    auto s = to_std(load<scalar_t>(a));
    CHECK(s.size() == std::size_t(N));
    store(o, from_std<scalar_t>(s * s));
    for (int i = 0; i < N; ++i) CHECK(o[i] == a[i] * a[i]);
}
#endif

#if FORGE_SIMD_HAVE_MDSPAN
TEST(mdspan_overloads_match_the_tables) {
    const auto& k = tier_a_active();
    std::vector<float> x(4 * 6);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = float((i * 37) % 23) - 11.f;
    x[9] = std::nanf("");
    // contiguous rank-1 and rank-2 views use the storage in place
    std::mdspan<const float, std::dextents<std::size_t, 1>> v1(x.data(), x.size());
    std::mdspan<const float, std::dextents<std::size_t, 2>> v2(x.data(), 4, 6);
    CHECK(md::min(v1) == k.min_f32(x.data(), x.size()) && md::max(v2) == k.max_f32(x.data(), x.size()));
    CHECK(md::argmax(v2) == k.argmax_f32(x.data(), x.size()) && md::argmin(v1) == k.argmin_f32(x.data(), x.size()));
    CHECK(md::count(v2, 2, 0.f) == k.count_f32(x.data(), x.size(), 2, 0.f) && md::find(v1, 4, 10.f) == k.find_f32(x.data(), x.size(), 4, 10.f));
    CHECK(md::any(v1, 0, 11.f) == k.any_f32(x.data(), x.size(), 0, 11.f) && md::all(v1, 3, 100.f) == k.all_f32(x.data(), x.size(), 3, 100.f));
    std::size_t rows[4]; md::argmax_rows(v2, rows);
    for (std::size_t r = 0; r < 4; ++r) CHECK(rows[r] == k.argmax_f32(x.data() + 6 * r, 6));
    // a strided view (every second element) and a layout_left matrix go through a copy
    std::mdspan<const float, std::dextents<std::size_t, 1>, std::layout_stride> every2(x.data(), {std::dextents<std::size_t, 1>(12), std::array<std::size_t, 1>{2}});
    float mn = INFINITY, mx = -INFINITY;
    for (std::size_t i = 0; i < 12; ++i) if (x[2 * i] == x[2 * i]) { mn = std::min(mn, x[2 * i]); mx = std::max(mx, x[2 * i]); }
    CHECK(md::min(every2) == mn && md::max(every2) == mx);
    std::mdspan<const float, std::dextents<std::size_t, 2>, std::layout_left> left(x.data(), 4, 6);
    std::vector<float> rowmajor;
    for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = 0; j < 6; ++j) rowmajor.push_back(left[std::array<std::size_t, 2>{i, j}]);
    CHECK(md::argmax(left) == k.argmax_f32(rowmajor.data(), rowmajor.size()));
    // clamp and scan into a strided output: only the viewed elements change
    std::vector<float> y(24, -99.f);
    std::mdspan<float, std::dextents<std::size_t, 1>, std::layout_stride> yout(y.data(), {std::dextents<std::size_t, 1>(12), std::array<std::size_t, 1>{2}});
    md::clamp(every2, yout, -3.f, 3.f);
    for (std::size_t i = 0; i < 12; ++i) {
        const float in = x[2 * i], want = in < -3.f ? -3.f : in > 3.f ? 3.f : in;
        CHECK((y[2 * i] == want || (want != want && y[2 * i] != y[2 * i])) && y[2 * i + 1] == -99.f);
    }
    // sort a strided view in place
    std::vector<float> s(30); for (std::size_t i = 0; i < s.size(); ++i) s[i] = float((i * 17) % 31);
    std::vector<float> want;
    for (std::size_t i = 0; i < 10; ++i) want.push_back(s[3 * i]);
    std::sort(want.begin(), want.end());
    std::mdspan<float, std::dextents<std::size_t, 1>, std::layout_stride> every3(s.data(), {std::dextents<std::size_t, 1>(10), std::array<std::size_t, 1>{3}});
    md::sort(every3);
    for (std::size_t i = 0; i < 10; ++i) CHECK(s[3 * i] == want[i]);
    CHECK(s[1] == 17.f && s[2] == float((2 * 17) % 31));
    // integers, doubles, bytes
    std::vector<std::int32_t> q(50); std::iota(q.begin(), q.end(), -20);
    std::vector<std::int32_t> ps(50);
    std::mdspan<const std::int32_t, std::dextents<std::size_t, 1>> qv(q.data(), q.size());
    md::scan_add(qv, std::mdspan<std::int32_t, std::dextents<std::size_t, 1>>(ps.data(), ps.size()));
    std::int32_t run = 0; for (std::size_t i = 0; i < 50; ++i) { run += q[i]; CHECK(ps[i] == run); }
    CHECK(md::sum(qv) == run && md::dot(qv, qv) == k.dot_i32(q.data(), q.data(), 50) && md::min(qv) == -20 && md::max(qv) == 29 && md::find(qv, 5) == 25);
    std::vector<double> d(12); for (std::size_t i = 0; i < 12; ++i) d[i] = 0.25 * double(i) - 1.0;
    std::mdspan<const double, std::extents<std::size_t, 3, 4>> dm(d.data());
    std::vector<double> dt(12);
    md::transpose(dm, std::mdspan<double, std::extents<std::size_t, 4, 3>>(dt.data()));
    for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = 0; j < 4; ++j) CHECK(dt[j * 3 + i] == d[i * 4 + j]);
    CHECK(md::sum(dm) == k.sum_f64(d.data(), 12) && md::dot(dm, dm) == k.dot_f64(d.data(), d.data(), 12));
    std::vector<std::uint8_t> bytes(100); for (std::size_t i = 0; i < 100; ++i) bytes[i] = std::uint8_t(i % 7);
    std::mdspan<const std::uint8_t, std::dextents<std::size_t, 1>> bv(bytes.data(), bytes.size());
    CHECK(md::find_byte(bv, 6) == 6 && md::count_byte(bv, 3) == k.count_byte(bytes.data(), 100, 3));
    std::uint32_t idx[5]; float val[5];
    CHECK(md::topk(v2, 5, idx, val) == 5);
    std::uint32_t idx2[5]; float val2[5]; k.topk_f32(x.data(), x.size(), 5, idx2, val2);
    for (int i = 0; i < 5; ++i) CHECK(idx[i] == idx2[i] && val[i] == val2[i]);
}
#endif
