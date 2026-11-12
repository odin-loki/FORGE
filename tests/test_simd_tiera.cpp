// Tier A algorithms (simd/algo_num.hpp, algo_search.hpp, algo_sort.hpp, algo_text.hpp) through every
// tier_a_kernels table usable on this CPU, each against a scalar reference written here, over random inputs
// and edge sizes (0, 1, vector width +- 1, unaligned starts, large). Inputs live in exactly-sized heap blocks
// so an out-of-bounds read or write is visible to ASan.
#include "check.hpp"
#include <forge/simd/tier_a.hpp>
#include <forge/simd/dispatch.hpp>
#include <forge/simd/algo_sort.hpp>
#include <forge/simd/algo_text.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
using namespace forge::simd;

namespace {
std::vector<const tier_a_kernels*> tables() {
    std::vector<const tier_a_kernels*> v;
    v.push_back(tier_a_by_name("scalar"));
    for (const char* t : target_names()) if (auto* k = tier_a_by_name(t)) if (k != v[0]) v.push_back(k);   // every table usable on this CPU
    return v;
}
struct rng {
    std::uint64_t s;
    explicit rng(std::uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    std::uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    std::uint32_t u32() { return std::uint32_t(next() >> 32); }
    std::size_t below(std::size_t n) { return n ? std::size_t(next() % n) : 0; }
    float unit() { return float(u32() >> 8) / float(1 << 24); }
};
const std::size_t kSizes[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000, 4097};
const float kInf = std::numeric_limits<float>::infinity();
const float kNaN = std::numeric_limits<float>::quiet_NaN();

// n elements at an address that is `off` elements past an exactly-sized allocation's start
template <class T> struct block {
    std::unique_ptr<T[]> mem; T* p; std::size_t n;
    block(std::size_t n_, std::size_t off = 0) : mem(new T[n_ + off]), p(mem.get() + off), n(n_) {}
};
// mode 0: smooth random; 1: few distinct values (many ties); 2: specials mixed in (NaN, inf, zeros)
void fill_f32(rng& r, float* x, std::size_t n, int mode) {
    for (std::size_t i = 0; i < n; ++i) {
        if (mode == 1) x[i] = float(int(r.below(5))) - 2.0f;
        else x[i] = (r.unit() - 0.5f) * 200.0f;
        if (mode == 2) switch (r.below(12)) {
            case 0: x[i] = kNaN; break; case 1: x[i] = kInf; break; case 2: x[i] = -kInf; break;
            case 3: x[i] = 0.0f; break; case 4: x[i] = -0.0f; break; case 5: x[i] = -kNaN; break; default: break;
        }
    }
}
bool same_bits(const float* a, const float* b, std::size_t n) { return n == 0 || std::memcmp(a, b, n * sizeof(float)) == 0; }
bool same_float(float a, float b) { return std::memcmp(&a, &b, 4) == 0 || a == b; }
bool cmp_ref(int op, float v, float t) { switch (op) { case 0: return v == t; case 1: return v != t; case 2: return v < t; case 3: return v <= t; case 4: return v > t; default: return v >= t; } }
} // namespace

TEST(tiera_tables_present) {
    auto ts = tables();
    CHECK(!ts.empty()); CHECK(std::string(ts[0]->target) == "scalar");
    std::printf("  tier A tables:"); for (auto* t : ts) std::printf(" %s", t->target); std::printf("  active: %s\n", tier_a_active().target);
    CHECK(tier_a_by_name("nope") == nullptr); CHECK(tier_a_by_name(nullptr) == nullptr);
#if defined(__x86_64__) || defined(_M_X64)
    CHECK((tier_a_by_name("avx2") != nullptr) == (by_name("avx2") != nullptr));
#endif
}

TEST(tiera_f32_minmax_arg_count_find_clamp) {
    rng r(1);
    for (auto* k : tables()) for (std::size_t n : kSizes) for (int mode = 0; mode < 3; ++mode) for (std::size_t off = 0; off < 2; ++off) {
        block<float> b(n, off); float* x = b.p; fill_f32(r, x, n, mode);
        float mn = kInf, mx = -kInf;
        for (std::size_t i = 0; i < n; ++i) if (x[i] == x[i]) { mn = x[i] < mn ? x[i] : mn; mx = x[i] > mx ? x[i] : mx; }
        std::size_t amn = n, amx = n;
        for (std::size_t i = n; i-- > 0;) { if (x[i] == mn) amn = i; if (x[i] == mx) amx = i; }
        CHECK_MSG(k->min_f32(x, n) == mn && k->max_f32(x, n) == mx, "%s n=%zu mode=%d", k->target, n, mode);
        CHECK_MSG(k->argmin_f32(x, n) == amn && k->argmax_f32(x, n) == amx, "%s n=%zu mode=%d arg", k->target, n, mode);
        for (float t : {0.0f, 1.0f, -2.0f, 37.5f, kInf, kNaN, n ? x[r.below(n)] : 3.0f}) for (int op = 0; op < 6; ++op) {
            std::size_t cnt = 0, first = n;
            for (std::size_t i = 0; i < n; ++i) if (cmp_ref(op, x[i], t)) { ++cnt; if (first == n) first = i; }
            bool ok = k->count_f32(x, n, op, t) == cnt && k->find_f32(x, n, op, t) == first && k->any_f32(x, n, op, t) == (cnt > 0) && k->all_f32(x, n, op, t) == (cnt == n);
            CHECK_MSG(ok, "%s n=%zu mode=%d op=%d t=%g", k->target, n, mode, op, double(t));
        }
        block<float> y(n, off), ref(n);
        const float lo = -30.0f, hi = 45.5f;
        for (std::size_t i = 0; i < n; ++i) ref.p[i] = x[i] < lo ? lo : x[i] > hi ? hi : x[i];
        k->clamp_f32(x, y.p, n, lo, hi);
        CHECK_MSG(same_bits(y.p, ref.p, n), "%s clamp n=%zu mode=%d", k->target, n, mode);
        k->clamp_f32(x, x, n, lo, hi);   // in place
        CHECK(same_bits(x, ref.p, n));
    }
    // a match at every position of a longer array, and none
    for (auto* k : tables()) {
        const std::size_t n = 200; block<float> b(n); for (std::size_t i = 0; i < n; ++i) b.p[i] = 1.0f;
        CHECK(k->find_f32(b.p, n, 4, 5.0f) == n);
        for (std::size_t at = 0; at < n; ++at) { b.p[at] = 9.0f; CHECK(k->find_f32(b.p, n, 4, 5.0f) == at); CHECK(k->argmax_f32(b.p, n) == at); b.p[at] = 1.0f; }
    }
}

TEST(tiera_generic_predicates_scalar_target) {
    // find_if / count_if / any_of / all_of take a vec -> mask predicate; instantiated here for the scalar target.
    rng r(2);
    for (std::size_t n : kSizes) {
        block<float> b(n); fill_f32(r, b.p, n, 2);
        auto pred = [](vec<float, scalar_t> v) { return lt(splat<scalar_t>(10.0f), v) & lt(v, splat<scalar_t>(50.0f)); };
        std::size_t cnt = 0, first = n;
        for (std::size_t i = 0; i < n; ++i) if (b.p[i] > 10.0f && b.p[i] < 50.0f) { ++cnt; if (first == n) first = i; }
        CHECK(algo::count_if<scalar_t>(b.p, n, pred) == cnt); CHECK(algo::find_if<scalar_t>(b.p, n, pred) == first);
        CHECK(algo::any_of<scalar_t>(b.p, n, pred) == (cnt > 0)); CHECK(algo::all_of<scalar_t>(b.p, n, pred) == (cnt == n));
    }
}

TEST(tiera_prefix_sums) {
    rng r(3);
    for (auto* k : tables()) for (std::size_t n : kSizes) for (int excl = 0; excl < 2; ++excl) for (std::size_t off = 0; off < 2; ++off) {
        block<float> x(n, off), y(n, off), ref(n); fill_f32(r, x.p, n, 0);
        float acc = 0; for (std::size_t i = 0; i < n; ++i) { if (excl) { ref.p[i] = acc; acc += x.p[i]; } else { acc += x.p[i]; ref.p[i] = acc; } }
        k->scan_add_f32(x.p, y.p, n, excl != 0);
        CHECK_MSG(same_bits(y.p, ref.p, n), "%s scan f32 n=%zu excl=%d", k->target, n, excl);
        block<double> xd(n), yd(n); double accd = 0; bool okd = true;
        for (std::size_t i = 0; i < n; ++i) xd.p[i] = double(x.p[i]) * 1.000001;
        k->scan_add_f64(xd.p, yd.p, n, excl != 0);
        for (std::size_t i = 0; i < n; ++i) { if (excl) { okd &= yd.p[i] == accd; accd += xd.p[i]; } else { accd += xd.p[i]; okd &= yd.p[i] == accd; } }
        CHECK(okd);
        block<std::int32_t> xi(n, off), yi(n, off), ri(n);
        for (std::size_t i = 0; i < n; ++i) xi.p[i] = std::int32_t(r.u32());   // full range: sums wrap
        std::uint32_t a = 0; for (std::size_t i = 0; i < n; ++i) { if (excl) { ri.p[i] = std::int32_t(a); a += std::uint32_t(xi.p[i]); } else { a += std::uint32_t(xi.p[i]); ri.p[i] = std::int32_t(a); } }
        k->scan_add_i32(xi.p, yi.p, n, excl != 0);
        CHECK_MSG(n == 0 || std::memcmp(yi.p, ri.p, n * 4) == 0, "%s scan i32 n=%zu excl=%d", k->target, n, excl);
        k->scan_add_i32(xi.p, xi.p, n, excl != 0);   // in place
        CHECK(n == 0 || std::memcmp(xi.p, ri.p, n * 4) == 0);
    }
}

TEST(tiera_histograms) {
    rng r(4);
    for (auto* k : tables()) for (std::size_t n : kSizes) {
        block<std::uint8_t> b(n, 1); for (std::size_t i = 0; i < n; ++i) b.p[i] = std::uint8_t(r.below(3) ? r.below(256) : 7);
        std::uint32_t h[256], ref[256]; for (int i = 0; i < 256; ++i) h[i] = ref[i] = std::uint32_t(i);   // added to, not cleared
        for (std::size_t i = 0; i < n; ++i) ++ref[b.p[i]];
        k->histogram_u8(b.p, n, h);
        CHECK_MSG(std::memcmp(h, ref, sizeof h) == 0, "%s hist u8 n=%zu", k->target, n);
        block<float> x(n); fill_f32(r, x.p, n, 2);
        for (std::size_t bins : {std::size_t(1), std::size_t(7), std::size_t(64)}) {
            const float lo = -50.0f, hi = 75.0f;
            std::vector<std::uint32_t> hf(bins, 0), rf(bins, 0); std::size_t counted = 0;
            const float scale = float(double(bins) / (double(hi) - double(lo)));
            for (std::size_t i = 0; i < n; ++i) if (x.p[i] >= lo && x.p[i] < hi) { volatile float s = x.p[i] - lo; volatile float q = s * scale; std::size_t bi = std::size_t(std::int32_t(q)); ++rf[std::min(bi, bins - 1)]; ++counted; }
            CHECK(k->histogram_f32(x.p, n, lo, hi, hf.data(), bins) == counted);
            CHECK_MSG(hf == rf, "%s hist f32 n=%zu bins=%zu", k->target, n, bins);
        }
        std::uint32_t z = 0; CHECK(k->histogram_f32(x.p, n, 1.0f, 1.0f, &z, 1) == 0); CHECK(k->histogram_f32(x.p, n, 0.0f, 1.0f, &z, 0) == 0);
    }
}

TEST(tiera_transpose) {
    rng r(5);
    for (auto* k : tables()) {
        for (auto [rows, cols] : {std::pair<std::size_t, std::size_t>{0, 0}, {1, 1}, {1, 9}, {9, 1}, {3, 5}, {32, 32}, {33, 65}, {70, 31}, {100, 129}}) {
            block<float> a(rows * cols), b(rows * cols); for (std::size_t i = 0; i < rows * cols; ++i) a.p[i] = float(r.u32() & 0xffff);
            k->transpose_f32(a.p, b.p, rows, cols);
            bool ok = true; for (std::size_t i = 0; i < rows; ++i) for (std::size_t j = 0; j < cols; ++j) ok &= b.p[j * rows + i] == a.p[i * cols + j];
            CHECK_MSG(ok, "%s transpose %zux%zu", k->target, rows, cols);
            block<double> ad(rows * cols), bd(rows * cols); for (std::size_t i = 0; i < rows * cols; ++i) ad.p[i] = double(i) * 0.5;
            k->transpose_f64(ad.p, bd.p, rows, cols);
            ok = true; for (std::size_t i = 0; i < rows; ++i) for (std::size_t j = 0; j < cols; ++j) ok &= bd.p[j * rows + i] == ad.p[i * cols + j];
            CHECK(ok);
        }
        for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(31), std::size_t(32), std::size_t(33), std::size_t(64), std::size_t(97)}) {
            block<float> a(n * n), o(n * n); for (std::size_t i = 0; i < n * n; ++i) o.p[i] = a.p[i] = float(i);
            k->transpose_square_f32(a.p, n);
            bool ok = true; for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) ok &= a.p[i * n + j] == o.p[j * n + i];
            CHECK_MSG(ok, "%s transpose_square n=%zu", k->target, n);
            block<double> d(n * n); for (std::size_t i = 0; i < n * n; ++i) d.p[i] = double(i);
            k->transpose_square_f64(d.p, n); k->transpose_square_f64(d.p, n);
            ok = true; for (std::size_t i = 0; i < n * n; ++i) ok &= d.p[i] == double(i);
            CHECK(ok);
        }
    }
}

TEST(tiera_f64_and_integer_blas1) {
    rng r(6);
    for (auto* k : tables()) for (std::size_t n : kSizes) {
        block<double> a(n, 1), b(n), y(n), yr(n);
        long double dot = 0, mag = 0, sum = 0, smag = 0;
        for (std::size_t i = 0; i < n; ++i) { a.p[i] = (double(r.unit()) - 0.5) * 1e3; b.p[i] = (double(r.unit()) - 0.5) * 1e-2; y.p[i] = yr.p[i] = double(r.unit());
            dot += (long double)a.p[i] * b.p[i]; mag += std::fabs((long double)a.p[i] * b.p[i]); sum += a.p[i]; smag += std::fabs(a.p[i]); }
        // tolerance: n * eps * sum |a_i b_i| (any summation order, with or without fused multiply-add)
        const double tol = 2.3e-16 * double(n + 1);
        CHECK_MSG(std::fabs(double(k->dot_f64(a.p, b.p, n) - dot)) <= tol * double(mag) + 1e-300, "%s dot_f64 n=%zu", k->target, n);
        CHECK(std::fabs(double(k->sum_f64(a.p, n) - sum)) <= tol * double(smag) + 1e-300);
        k->axpy_f64(1.5, a.p, y.p, n);
        bool ok = true; for (std::size_t i = 0; i < n; ++i) ok &= std::fabs(y.p[i] - (yr.p[i] + 1.5 * a.p[i])) <= 1e-12 * (1 + std::fabs(a.p[i]));
        CHECK(ok);
        block<std::int32_t> ia(n, 1), ib(n), iy(n), iyr(n); block<std::int16_t> sa(n), sb(n); block<std::int8_t> ba(n), bb(n);
        std::uint64_t d32 = 0; std::int64_t d16 = 0, d8 = 0, s32 = 0; std::int32_t mn = std::numeric_limits<std::int32_t>::max(), mx = std::numeric_limits<std::int32_t>::min();
        const std::int32_t needle = 12345; std::size_t first = n, cnt = 0;
        for (std::size_t i = 0; i < n; ++i) {
            ia.p[i] = std::int32_t(r.u32()); ib.p[i] = std::int32_t(r.u32()); iy.p[i] = iyr.p[i] = std::int32_t(r.u32());
            if (r.below(9) == 0) ia.p[i] = needle;
            sa.p[i] = std::int16_t(r.u32()); sb.p[i] = std::int16_t(r.u32()); ba.p[i] = std::int8_t(r.u32()); bb.p[i] = std::int8_t(r.u32());
            d32 += std::uint64_t(std::int64_t(ia.p[i]) * ib.p[i]); d16 += std::int64_t(sa.p[i]) * sb.p[i]; d8 += std::int64_t(ba.p[i]) * bb.p[i]; s32 += ia.p[i];
            mn = std::min(mn, ia.p[i]); mx = std::max(mx, ia.p[i]);
            if (ia.p[i] == needle) { ++cnt; if (first == n) first = i; }
            iyr.p[i] = std::int32_t(std::uint32_t(iyr.p[i]) + 0x9E3779B9u * std::uint32_t(ia.p[i]));
        }
        CHECK_MSG(k->dot_i32(ia.p, ib.p, n) == std::int64_t(d32) && k->dot_i16(sa.p, sb.p, n) == d16 && k->dot_i8(ba.p, bb.p, n) == d8, "%s int dots n=%zu", k->target, n);
        CHECK(k->sum_i32(ia.p, n) == s32); CHECK(k->min_i32(ia.p, n) == mn); CHECK(k->max_i32(ia.p, n) == mx);
        CHECK_MSG(k->find_i32(ia.p, n, needle) == first && k->count_eq_i32(ia.p, n, needle) == cnt, "%s find_i32 n=%zu", k->target, n);
        k->axpy_i32(std::int32_t(0x9E3779B9u), ia.p, iy.p, n);
        CHECK(n == 0 || std::memcmp(iy.p, iyr.p, n * 4) == 0);
    }
    // extreme values: the int16 and int8 dots must not overflow their partial sums
    for (auto* k : tables()) {
        const std::size_t n = 200000; block<std::int16_t> a(n); block<std::int8_t> b(n);
        for (std::size_t i = 0; i < n; ++i) { a.p[i] = -32768; b.p[i] = -128; }
        CHECK(k->dot_i16(a.p, a.p, n) == std::int64_t(n) * 32768 * 32768); CHECK(k->dot_i8(b.p, b.p, n) == std::int64_t(n) * 128 * 128);
    }
}

TEST(tiera_bitsets) {
    rng r(7);
    for (auto* k : tables()) for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(7), std::size_t(8), std::size_t(9), std::size_t(33), std::size_t(1000)}) {
        block<std::uint64_t> a(n), b(n), o(n);
        std::uint64_t pc = 0, pa = 0;
        for (std::size_t i = 0; i < n; ++i) { a.p[i] = r.next() & r.next(); b.p[i] = r.next(); for (int j = 0; j < 64; ++j) { pc += a.p[i] >> j & 1; pa += (a.p[i] & b.p[i]) >> j & 1; } }
        CHECK(k->popcount(a.p, n) == pc); CHECK(k->popcount_and(a.p, b.p, n) == pa);
        for (int op = 0; op < 4; ++op) {
            k->bitset_op(op, a.p, b.p, o.p, n);
            bool ok = true; for (std::size_t i = 0; i < n; ++i) ok &= o.p[i] == (op == 0 ? a.p[i] & b.p[i] : op == 1 ? a.p[i] | b.p[i] : op == 2 ? a.p[i] ^ b.p[i] : a.p[i] & ~b.p[i]);
            CHECK_MSG(ok, "%s bitset_op %d n=%zu", k->target, op, n);
        }
        for (int rep = 0; rep < 50; ++rep) {
            std::size_t from = r.below(n * 64 + 70), ref = n * 64;
            for (std::size_t bit = from; bit < n * 64; ++bit) if (a.p[bit / 64] >> (bit % 64) & 1) { ref = bit; break; }
            CHECK(k->bitset_find_first(a.p, n, from) == ref);
        }
        for (std::size_t i = 0; i < n; ++i) a.p[i] = 0;
        CHECK(k->bitset_find_first(a.p, n, 0) == n * 64);
        if (n) { a.p[n - 1] = std::uint64_t(1) << 63; CHECK(k->bitset_find_first(a.p, n, 0) == n * 64 - 1); CHECK(k->bitset_find_first(a.p, n, n * 64 - 1) == n * 64 - 1); }
    }
}

TEST(tiera_byte_search) {
    rng r(8);
    for (auto* k : tables()) {
        for (std::size_t n : kSizes) for (std::size_t off = 0; off < 3; ++off) {
            block<std::uint8_t> b(n, off), c(n, off);
            for (std::size_t i = 0; i < n; ++i) c.p[i] = b.p[i] = std::uint8_t('a' + r.below(4));
            CHECK(k->find_byte(b.p, n, 'z') == n); CHECK(k->count_byte(b.p, n, 'z') == 0); CHECK(k->mismatch(b.p, c.p, n) == n);
            std::size_t cnt = 0; for (std::size_t i = 0; i < n; ++i) cnt += b.p[i] == 'b';
            CHECK(k->count_byte(b.p, n, 'b') == cnt);
            const void* m = n ? std::memchr(b.p, 'c', n) : nullptr;
            CHECK(k->find_byte(b.p, n, 'c') == (m ? std::size_t(static_cast<const std::uint8_t*>(m) - b.p) : n));
            for (std::size_t at = 0; at < n; at += (n > 300 ? 37 : 1)) {   // the only match / difference at every position
                std::uint8_t keep = b.p[at]; b.p[at] = 0x80;
                bool ok = k->find_byte(b.p, n, 0x80) == at && k->mismatch(b.p, c.p, n) == at && k->count_byte(b.p, n, 0x80) == 1;
                CHECK_MSG(ok, "%s find_byte n=%zu off=%zu at=%zu", k->target, n, off, at);
                b.p[at] = keep;
            }
        }
        // substring search against std::search: random haystacks over a tiny alphabet (many partial matches)
        for (int rep = 0; rep < 3000; ++rep) {
            std::size_t n = r.below(rep % 10 == 0 ? 600 : 90), m = r.below(rep % 3 == 0 ? 70 : 6);
            block<std::uint8_t> h(n, r.below(3)), nd(m, r.below(2));
            for (std::size_t i = 0; i < n; ++i) h.p[i] = std::uint8_t('a' + r.below(2));
            for (std::size_t i = 0; i < m; ++i) nd.p[i] = std::uint8_t('a' + r.below(2));
            if (m && m <= n && r.below(2)) std::memcpy(nd.p, h.p + r.below(n - m + 1), m);   // a needle that does occur
            auto it = std::search(h.p, h.p + n, nd.p, nd.p + m);
            std::size_t ref = m == 0 ? 0 : std::size_t(it - h.p);
            CHECK_MSG(k->find_bytes(h.p, n, nd.p, m) == ref, "%s find_bytes n=%zu m=%zu", k->target, n, m);
        }
        // the needle at every position, including the very end
        for (std::size_t n : {std::size_t(5), std::size_t(64), std::size_t(65), std::size_t(131)}) for (std::size_t m : {std::size_t(2), std::size_t(5)}) for (std::size_t at = 0; at + m <= n; ++at) {
            block<std::uint8_t> h(n); std::memset(h.p, 'x', n); std::memcpy(h.p + at, "ab_cd", m); h.p[at + m - 1] = 'Q';
            std::uint8_t nd[5]; std::memcpy(nd, "ab_cd", 5); nd[m - 1] = 'Q';
            CHECK_MSG(k->find_bytes(h.p, n, nd, m) == at, "%s find_bytes at=%zu n=%zu m=%zu", k->target, at, n, m);
        }
    }
}

TEST(sorting_networks_zero_one_principle) {
    // A comparator network sorts every input if it sorts every 0/1 input: all 2^n of them for n <= 16.
    for (std::size_t n = 0; n <= 16; ++n) {
        bool ok = true;
        for (std::uint32_t bits = 0; bits < (1u << n); ++bits) {
            int a[16]; int ones = 0;
            for (std::size_t i = 0; i < n; ++i) { a[i] = int(bits >> i & 1); ones += a[i]; }
            algo::sort_small<scalar_t, int>(a, n);
            for (std::size_t i = 0; i < n; ++i) ok &= a[i] == (i + std::size_t(ones) >= n ? 1 : 0);
        }
        CHECK_MSG(ok, "sort_small n=%zu", n);
    }
    int a8[8] = {5, 1, 4, 2, 8, 7, 3, 6}; algo::sort_network<8, scalar_t, int>(a8);
    CHECK(std::is_sorted(a8, a8 + 8) && a8[0] == 1 && a8[7] == 8);
    double a5[5] = {0.5, -1, 3, 2, -7}; algo::sort_network<5, scalar_t, double>(a5); CHECK(std::is_sorted(a5, a5 + 5));
    int a12[12] = {9, 3, 11, 0, 7, 5, 1, 10, 2, 8, 6, 4}; algo::sort_network<12, scalar_t, int>(a12);
    bool ok = true; for (int i = 0; i < 12; ++i) ok &= a12[i] == i; CHECK(ok);
    rng r(9);
    for (auto* k : tables()) for (std::size_t n = 0; n <= 40; ++n) for (int rep = 0; rep < 20; ++rep) {
        block<std::int32_t> a(n), b(n); block<float> f(n), g(n);
        for (std::size_t i = 0; i < n; ++i) { b.p[i] = a.p[i] = std::int32_t(r.u32()) >> int(r.below(28)); g.p[i] = f.p[i] = float(a.p[i]) * 0.25f; }
        k->sort_small_i32(a.p, n); std::sort(b.p, b.p + n);
        k->sort_small_f32(f.p, n); std::sort(g.p, g.p + n);
        CHECK_MSG((n == 0 || std::memcmp(a.p, b.p, n * 4) == 0) && same_bits(f.p, g.p, n), "%s sort_small n=%zu", k->target, n);
    }
}

namespace {
std::int32_t total_key(float f) { std::int32_t b; std::memcpy(&b, &f, 4); return b < 0 ? b ^ 0x7fffffff : b; }
bool total_less(float a, float b) { return total_key(a) < total_key(b); }
// pattern 0 random, 1 sorted, 2 reversed, 3 all equal, 4 few distinct, 5 organ pipe, 6 extremes mixed in, 7 sawtooth
void fill_i32(rng& r, std::int32_t* a, std::size_t n, int pattern) {
    for (std::size_t i = 0; i < n; ++i) switch (pattern) {
        case 0: a[i] = std::int32_t(r.u32()); break;
        case 1: a[i] = std::int32_t(i) - 1000; break;
        case 2: a[i] = std::int32_t(n - i); break;
        case 3: a[i] = 42; break;
        case 4: a[i] = std::int32_t(r.below(3)); break;
        case 5: a[i] = std::int32_t(i < n / 2 ? i : n - i); break;
        case 6: a[i] = r.below(4) == 0 ? std::numeric_limits<std::int32_t>::max() : r.below(4) == 0 ? std::numeric_limits<std::int32_t>::min() : std::int32_t(r.u32()); break;
        default: a[i] = std::int32_t(i % 17); break;
    }
}
}

TEST(tiera_sort_matches_std_sort) {
    rng r(10);
    std::vector<std::size_t> sizes(std::begin(kSizes), std::end(kSizes));
    sizes.push_back(65536 + 7); sizes.push_back((std::size_t(1) << 20) + 3);
    for (auto* k : tables()) for (std::size_t n : sizes) for (int pattern = 0; pattern < 8; ++pattern) {
        if (n > 100000 && pattern > 4) continue;
        block<std::int32_t> a(n, 1), b(n); fill_i32(r, a.p, n, pattern);
        if (n) std::memcpy(b.p, a.p, n * 4);
        k->sort_i32(a.p, n); std::sort(b.p, b.p + n);
        CHECK_MSG(n == 0 || std::memcmp(a.p, b.p, n * 4) == 0, "%s sort_i32 n=%zu pattern=%d", k->target, n, pattern);
        block<std::uint32_t> u(n), v(n); for (std::size_t i = 0; i < n; ++i) v.p[i] = u.p[i] = std::uint32_t(b.p[(i * 7919) % (n ? n : 1)]) * 2654435761u >> (pattern == 4 ? 30 : 0);
        k->sort_u32(u.p, n); std::sort(v.p, v.p + n);
        CHECK_MSG(n == 0 || std::memcmp(u.p, v.p, n * 4) == 0, "%s sort_u32 n=%zu pattern=%d", k->target, n, pattern);
        block<float> f(n, 1), g(n);
        if (pattern < 3) fill_f32(r, f.p, n, pattern); else for (std::size_t i = 0; i < n; ++i) f.p[i] = float(a.p[i] % 1000) * 0.5f;
        if (n) std::memcpy(g.p, f.p, n * 4);
        k->sort_f32(f.p, n); std::sort(g.p, g.p + n, total_less);
        CHECK_MSG(same_bits(f.p, g.p, n), "%s sort_f32 n=%zu pattern=%d", k->target, n, pattern);   // bits: zero signs and NaN payloads survive
    }
}

TEST(tiera_partial_sort_select_topk) {
    rng r(11);
    for (auto* k : tables()) for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(9), std::size_t(33), std::size_t(100), std::size_t(1000), std::size_t(50000)}) for (int mode = 0; mode < 3; ++mode) {
        block<float> x(n, 1); fill_f32(r, x.p, n, mode);
        std::vector<float> sorted(x.p, x.p + n); std::sort(sorted.begin(), sorted.end(), total_less);
        for (std::size_t kk : {std::size_t(0), std::size_t(1), std::size_t(5), std::size_t(40), n / 2, n > 0 ? n - 1 : 0, n, n + 3}) {
            block<float> a(n); if (n) std::memcpy(a.p, x.p, n * 4);
            k->partial_sort_f32(a.p, n, kk);
            const std::size_t m = std::min(kk, n);
            bool ok = same_bits(a.p, sorted.data(), m);
            std::vector<float> rest(a.p + m, a.p + n); std::sort(rest.begin(), rest.end(), total_less);   // the tail is a permutation of the remaining values
            ok = ok && same_bits(rest.data(), sorted.data() + m, n - m);
            CHECK_MSG(ok, "%s partial_sort n=%zu k=%zu mode=%d", k->target, n, kk, mode);
            if (kk < n) { float sel = k->select_f32(x.p, n, kk); CHECK(same_bits(&sorted[kk], &sel, 1)); }
            // top-k reference: value descending with NaN lowest, ties by index
            std::vector<std::uint32_t> order(n); std::iota(order.begin(), order.end(), 0u);
            auto key = [&](std::uint32_t i) { float f = x.p[i]; return f != f ? std::numeric_limits<std::int32_t>::min() : total_key(f); };
            std::stable_sort(order.begin(), order.end(), [&](std::uint32_t p, std::uint32_t q) { return key(p) > key(q); });
            std::vector<std::uint32_t> idx(m + 1, 0xdeadbeefu); std::vector<float> val(m + 1, -1.0f);
            std::size_t got = k->topk_f32(x.p, n, kk, idx.data(), val.data());
            ok = got == m && idx[m] == 0xdeadbeefu;
            for (std::size_t j = 0; j < m && ok; ++j) ok = idx[j] == order[j] && same_bits(&val[j], &x.p[order[j]], 1);
            CHECK_MSG(ok, "%s topk n=%zu k=%zu mode=%d", k->target, n, kk, mode);
            if (m) CHECK(k->topk_f32(x.p, n, kk, idx.data(), nullptr) == m);
        }
    }
}

namespace {
// Independent reference: decode a code point and check its range against the sequence length.
std::size_t utf8_error_ref(const std::uint8_t* p, std::size_t n) {
    std::size_t i = 0;
    while (i < n) {
        std::uint32_t c = p[i]; int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (!len || i + std::size_t(len) > n) return i;
        std::uint32_t cp = len == 1 ? c : c & (0x7fu >> len);
        for (int j = 1; j < len; ++j) { if ((p[i + std::size_t(j)] & 0xC0) != 0x80) return i; cp = cp << 6 | (p[i + std::size_t(j)] & 0x3f); }
        static const std::uint32_t min_cp[5] = {0, 0, 0x80, 0x800, 0x10000};
        if (cp < min_cp[len] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return i;
        i += std::size_t(len);
    }
    return n;
}
void append_cp(std::vector<std::uint8_t>& s, std::uint32_t cp) {
    if (cp < 0x80) s.push_back(std::uint8_t(cp));
    else if (cp < 0x800) { s.push_back(std::uint8_t(0xC0 | cp >> 6)); s.push_back(std::uint8_t(0x80 | (cp & 63))); }
    else if (cp < 0x10000) { s.push_back(std::uint8_t(0xE0 | cp >> 12)); s.push_back(std::uint8_t(0x80 | (cp >> 6 & 63))); s.push_back(std::uint8_t(0x80 | (cp & 63))); }
    else { s.push_back(std::uint8_t(0xF0 | cp >> 18)); s.push_back(std::uint8_t(0x80 | (cp >> 12 & 63))); s.push_back(std::uint8_t(0x80 | (cp >> 6 & 63))); s.push_back(std::uint8_t(0x80 | (cp & 63))); }
}
}

TEST(tiera_utf8_validation) {
    rng r(12);
    for (auto* k : tables()) {
        // every 1-, 2- and 3-byte string over a byte set that covers each class boundary, and 4-byte strings with those leads
        const std::uint8_t probe[] = {0x00, 0x7f, 0x80, 0x8f, 0x90, 0x9f, 0xa0, 0xbf, 0xc0, 0xc1, 0xc2, 0xdf, 0xe0, 0xe1, 0xec, 0xed, 0xee, 0xef, 0xf0, 0xf1, 0xf3, 0xf4, 0xf5, 0xff};
        bool ok = true; std::uint8_t s[4];
        for (std::uint8_t a : probe) { s[0] = a; ok &= k->utf8_error(s, 1) == utf8_error_ref(s, 1);
            for (std::uint8_t b : probe) { s[1] = b; ok &= k->utf8_error(s, 2) == utf8_error_ref(s, 2);
                for (std::uint8_t c : probe) { s[2] = c; ok &= k->utf8_error(s, 3) == utf8_error_ref(s, 3);
                    for (std::uint8_t d : probe) { s[3] = d; ok &= k->utf8_error(s, 4) == utf8_error_ref(s, 4); } } } }
        CHECK_MSG(ok, "%s utf8 probes", k->target);
        // boundary code points, valid
        std::vector<std::uint8_t> v;
        for (std::uint32_t cp : {0x0u, 0x7Fu, 0x80u, 0x7FFu, 0x800u, 0xD7FFu, 0xE000u, 0xFFFFu, 0x10000u, 0x10FFFFu}) append_cp(v, cp);
        CHECK(k->utf8_error(v.data(), v.size()) == v.size());
        // random valid text with long ASCII runs, then one corrupted byte: must agree with the reference
        for (int rep = 0; rep < 1500; ++rep) {
            std::vector<std::uint8_t> t; std::size_t want = r.below(rep % 20 == 0 ? 5000 : 300);
            while (t.size() < want) {
                if (r.below(4)) { for (std::size_t run = r.below(90); run--;) t.push_back(std::uint8_t(' ' + r.below(95))); }
                else { std::uint32_t cp = r.below(3) == 0 ? std::uint32_t(r.below(0x110000)) : std::uint32_t(r.below(0x3000)); if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0x263A; append_cp(t, cp); }
            }
            block<std::uint8_t> b(t.size(), r.below(4)); if (!t.empty()) std::memcpy(b.p, t.data(), t.size());
            CHECK_MSG(k->utf8_error(b.p, b.n) == b.n, "%s valid text flagged (len %zu)", k->target, b.n);
            if (b.n) {
                std::size_t at = r.below(b.n); b.p[at] = std::uint8_t(r.below(3) == 0 ? r.below(256) : 0x80 + r.below(0x80));
                CHECK_MSG(k->utf8_error(b.p, b.n) == utf8_error_ref(b.p, b.n), "%s corrupted at %zu of %zu", k->target, at, b.n);
                std::size_t cut = r.below(b.n + 1);   // truncation anywhere
                CHECK(k->utf8_error(b.p, cut) == utf8_error_ref(b.p, cut));
            }
        }
        // one bad byte at every position of an ASCII block (crosses every vector boundary)
        for (std::size_t n : {std::size_t(1), std::size_t(31), std::size_t(32), std::size_t(33), std::size_t(64), std::size_t(65), std::size_t(200)}) {
            block<std::uint8_t> b(n); std::memset(b.p, 'a', n);
            CHECK(k->utf8_error(b.p, n) == n);
            for (std::size_t at = 0; at < n; ++at) { b.p[at] = 0xff; CHECK(k->utf8_error(b.p, n) == at); b.p[at] = 'a'; }
        }
        CHECK(k->utf8_error(nullptr, 0) == 0);
    }
}

TEST(tiera_hex_and_base64) {
    rng r(13);
    static const char* b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (auto* k : tables()) {
        for (std::size_t n : kSizes) for (int rep = 0; rep < 4; ++rep) {
            block<std::uint8_t> in(n, rep & 1), back(n); for (std::size_t i = 0; i < n; ++i) in.p[i] = std::uint8_t(r.u32());
            block<char> hex(2 * n); std::string ref;
            for (bool upper : {false, true}) {
                k->hex_encode(in.p, n, hex.p, upper); ref.clear();
                for (std::size_t i = 0; i < n; ++i) { char t[3]; std::snprintf(t, sizeof t, upper ? "%02X" : "%02x", in.p[i]); ref += t; }
                CHECK_MSG(std::string(hex.p, 2 * n) == ref, "%s hex_encode n=%zu", k->target, n);
                CHECK(k->hex_decode(hex.p, 2 * n, back.p)); CHECK(n == 0 || std::memcmp(back.p, in.p, n) == 0);
            }
            if (n) {
                CHECK(!k->hex_decode(hex.p, 2 * n - 1, back.p));
                for (char bad : {'g', 'G', '/', ':', '@', '`', ' ', char(0xb1), char(0)}) { std::size_t at = r.below(2 * n); char keep = hex.p[at]; hex.p[at] = bad; CHECK_MSG(!k->hex_decode(hex.p, 2 * n, back.p), "%s hex_decode accepted 0x%02x", k->target, unsigned(static_cast<unsigned char>(bad))); hex.p[at] = keep; }
            }
            for (bool url : {false, true}) {
                std::string alphabet = std::string(b64) + (url ? "-_" : "+/");
                std::string want;   // reference encoder: bit by bit
                { std::uint32_t acc = 0; int nb = 0; for (std::size_t i = 0; i < n; ++i) { acc = acc << 8 | in.p[i]; nb += 8; while (nb >= 6) { want += alphabet[(acc >> (nb - 6)) & 63]; nb -= 6; } }
                  if (nb) want += alphabet[(acc << (6 - nb)) & 63];
                  if (!url) while (want.size() % 4) want += '='; }
                CHECK(algo::base64_encoded_size(n, url) == want.size());
                block<char> enc(want.size()); block<std::uint8_t> dec(algo::base64_decoded_max(want.size()));
                CHECK(k->base64_encode(in.p, n, enc.p, url) == want.size());
                CHECK_MSG(std::string(enc.p, want.size()) == want, "%s base64_encode n=%zu url=%d", k->target, n, int(url));
                std::ptrdiff_t got = k->base64_decode(enc.p, want.size(), dec.p, url);
                CHECK_MSG(got == std::ptrdiff_t(n) && (n == 0 || std::memcmp(dec.p, in.p, n) == 0), "%s base64_decode n=%zu url=%d", k->target, n, int(url));
                if (!want.empty()) {
                    for (char bad : {' ', '\n', '*', char(0), char(0xff), url ? '+' : '-', url ? '/' : '_'}) {   // one foreign character anywhere but a pad position
                        std::size_t at = r.below(want.size()); if (enc.p[at] == '=') continue;
                        char keep = enc.p[at]; enc.p[at] = bad;
                        CHECK_MSG(k->base64_decode(enc.p, want.size(), dec.p, url) == -1, "%s base64_decode accepted 0x%02x at %zu/%zu", k->target, unsigned(static_cast<unsigned char>(bad)), at, want.size());
                        enc.p[at] = keep;
                    }
                    CHECK(k->base64_decode(enc.p, want.size() - 1, dec.p, url) == -1 || (url && (want.size() - 1) % 4 != 1));   // a dropped character
                }
            }
        }
        // RFC 4648 section 10 vectors
        const char* plain[] = {"", "f", "fo", "foo", "foob", "fooba", "foobar"};
        const char* coded[] = {"", "Zg==", "Zm8=", "Zm9v", "Zm9vYg==", "Zm9vYmE=", "Zm9vYmFy"};
        for (int i = 0; i < 7; ++i) {
            char out[16] = {}; std::uint8_t dec[16] = {};
            std::size_t m = k->base64_encode(reinterpret_cast<const std::uint8_t*>(plain[i]), std::strlen(plain[i]), out, false);
            CHECK(std::string(out, m) == coded[i]);
            CHECK(k->base64_decode(coded[i], std::strlen(coded[i]), dec, false) == std::ptrdiff_t(std::strlen(plain[i])));
            CHECK(std::memcmp(dec, plain[i], std::strlen(plain[i])) == 0);
        }
        std::uint8_t d[8];
        CHECK(k->base64_decode("Zg", 2, d, false) == -1);       // padding missing
        CHECK(k->base64_decode("Zg==", 4, d, true) == -1);      // padding in the unpadded form
        CHECK(k->base64_decode("Zg", 2, d, true) == 1 && d[0] == 'f');
        CHECK(k->base64_decode("Zh==", 4, d, false) == -1);     // unused bits set
        CHECK(k->base64_decode("Zm9=", 4, d, false) == -1);
        CHECK(k->base64_decode("Z===", 4, d, false) == -1);
        CHECK(k->base64_decode("====", 4, d, false) == -1);
        CHECK(k->base64_decode("Zg=v", 4, d, false) == -1);     // padding in the middle
        CHECK(k->base64_decode("Z", 1, d, true) == -1);
        CHECK(k->base64_decode("-_-_", 4, d, true) == 3 && d[0] == 0xfb && d[1] == 0xff && d[2] == 0xbf);
        CHECK(k->base64_decode("+/+/", 4, d, false) == 3 && d[0] == 0xfb);
    }
}
