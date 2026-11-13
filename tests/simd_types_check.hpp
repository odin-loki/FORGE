#pragma once
// Checks of forge/simd/vec_types.hpp for one target, against scalar references written here with plain integer
// arithmetic. forge_types::check<Tg>(name) is instantiated in a translation unit compiled with that target's
// flags (tests/simd_types_isa.cpp, once per table) and returns the number of failed checks.
//
// Exhaustive (FORGE_PLAN A.10): every pair of inputs (2^16) for every binary int8 / uint8 operation, every input
// (2^16) for every unary int16 / uint16 operation and conversion. 32/64-bit lanes and double: edge values
// crossed with each other plus random values.
#include <forge/simd/vec_types.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace forge_types {
namespace {   // internal linkage: this header is compiled once per ISA with different flags
using namespace forge::simd;

struct ctx {
    const char* target; int fails = 0; long checks = 0;
    void fail(const char* what, const char* type, long long at) {
        if (fails < 20) std::fprintf(stderr, "  CHECK failed [%s] %s<%s> at %lld\n", target, what, type, at);
        ++fails;
    }
    // with the lane's bit patterns: result, expected, inputs
    template <class T> static unsigned long long bits(T x) { unsigned long long b = 0; std::memcpy(&b, &x, sizeof(T)); return b; }
    template <class T> void fail(const char* what, const char* type, long long at, T got, T want, T in0, T in1) {
        if (fails < 20) std::fprintf(stderr, "  CHECK failed [%s] %s<%s> at %lld: got %#llx, expected %#llx (inputs %#llx, %#llx)\n", target, what, type, at,
                                     bits(got), bits(want), bits(in0), bits(in1));
        ++fails;
    }
};
struct rng {
    std::uint64_t s;
    explicit rng(std::uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    std::uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};
template <class T> const char* tname() {
    if constexpr (std::is_same_v<T, double>) return "f64";
    else if constexpr (std::is_signed_v<T>) return sizeof(T) == 1 ? "i8" : sizeof(T) == 2 ? "i16" : sizeof(T) == 4 ? "i32" : "i64";
    else return sizeof(T) == 1 ? "u8" : sizeof(T) == 2 ? "u16" : sizeof(T) == 4 ? "u32" : "u64";
}
template <class T> bool same(T a, T b) { return std::memcmp(&a, &b, sizeof(T)) == 0 || (a != a && b != b); }

// ---- scalar references (independent of vt::) ----
template <class T> using U = std::make_unsigned_t<T>;
template <class T> T r_add(T x, T y) { if constexpr (std::is_floating_point_v<T>) return x + y; else return T(U<T>(U<T>(x) + U<T>(y))); }
template <class T> T r_sub(T x, T y) { if constexpr (std::is_floating_point_v<T>) return x - y; else return T(U<T>(U<T>(x) - U<T>(y))); }
template <class T> T r_mul(T x, T y) {
    if constexpr (std::is_floating_point_v<T>) return x * y;
    else return T(U<T>(std::uint64_t(U<T>(x)) * std::uint64_t(U<T>(y))));
}
template <class T> T r_sat_add(T x, T y) {
    const T mx = std::numeric_limits<T>::max(), mn = std::numeric_limits<T>::min();
    if constexpr (std::is_unsigned_v<T>) return y > T(mx - x) ? mx : T(x + y);
    else { if (y > 0 && x > T(mx - y)) return mx; if (y < 0 && x < T(mn - y)) return mn; return T(x + y); }
}
template <class T> T r_sat_sub(T x, T y) {
    const T mx = std::numeric_limits<T>::max(), mn = std::numeric_limits<T>::min();
    if constexpr (std::is_unsigned_v<T>) return y > x ? T(0) : T(x - y);
    else { if (y < 0 && x > T(mx + y)) return mx; if (y > 0 && x < T(mn + y)) return mn; return T(x - y); }
}
template <class T> T r_abs(T x) {
    if constexpr (std::is_floating_point_v<T>) return std::fabs(x);
    else if constexpr (std::is_unsigned_v<T>) return x;
    else return x < 0 ? T(U<T>(0) - U<T>(x)) : x;
}
template <class T> T r_pop(T x) { int c = 0; for (U<T> u = U<T>(x); u; u = U<T>(u >> 1)) c += int(u & 1); return T(c); }
template <class T> T r_shl(T x, int n) { return T(U<T>(std::uint64_t(U<T>(x)) << n)); }
template <class T> T r_shr(T x, int n) {
    if constexpr (std::is_unsigned_v<T>) return T(x >> n);
    else return x < 0 ? T(~(U<T>(~U<T>(x)) >> n)) : T(U<T>(x) >> n);   // arithmetic, written without >> on negatives
}
template <class T, class D> D r_clamp(T x) {
    const long double lo = static_cast<long double>(std::numeric_limits<D>::min()), hi = static_cast<long double>(std::numeric_limits<D>::max());
    const long double v = static_cast<long double>(x);
    return v < lo ? std::numeric_limits<D>::min() : v > hi ? std::numeric_limits<D>::max() : D(x);
}

// interesting values of T followed by random ones
template <class T> std::vector<T> samples(std::size_t random, std::uint64_t seed) {
    std::vector<T> v;
    if constexpr (std::is_floating_point_v<T>) {
        const T inf = std::numeric_limits<T>::infinity();
        for (T x : {T(0), T(-0.0), T(1), T(-1), T(0.5), T(-2.5), T(3), T(1e-300), T(-1e300), T(4503599627370496.0), T(123456789.125), inf, T(-inf),
                    std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::denorm_min(), std::numeric_limits<T>::max(), std::numeric_limits<T>::min()}) v.push_back(x);
        rng r(seed);
        for (std::size_t i = 0; i < random; ++i) {
            const double m = double(std::int64_t(r.next() >> 11)) / double(1ull << 52) - 1.0;
            v.push_back(T(std::ldexp(m, int(r.next() % 80) - 40)));
        }
    } else {
        const T mx = std::numeric_limits<T>::max(), mn = std::numeric_limits<T>::min();
        for (T x : {T(0), T(1), T(2), T(3), T(7), T(8), T(15), T(16), T(127), T(128), T(255), mx, T(mx - 1), T(mx / 2), T(mx / 2 + 1), mn, T(mn + 1),
                    T(U<T>(0) - U<T>(1)), T(U<T>(0) - U<T>(2)), T(U<T>(0) - U<T>(127)), T(U<T>(0) - U<T>(128)), T(U<T>(0x5555555555555555ull)), T(U<T>(0xAAAAAAAAAAAAAAAAull))}) v.push_back(x);
        rng r(seed);
        for (std::size_t i = 0; i < random; ++i) v.push_back(T(U<T>(r.next() >> (r.next() % 3 == 0 ? 40 : 0))));
    }
    return v;
}
// all ordered pairs of the values in s, padded to a multiple of `pad`
template <class T> void cross(const std::vector<T>& s, std::size_t pad, std::vector<T>& a, std::vector<T>& b) {
    a.clear(); b.clear();
    for (T x : s) for (T y : s) { a.push_back(x); b.push_back(y); }
    while (a.size() % pad) { a.push_back(T(1)); b.push_back(T(1)); }
}

template <class T, Target Tg> struct checker {
    using V = vec<T, Tg>;
    using M = mask<T, Tg>;
    static constexpr int N = V::N;
    ctx& c;
    const char* tn = tname<T>();

    template <class F, class R> void binary(const char* what, const std::vector<T>& a, const std::vector<T>& b, F&& f, R&& ref) {
        T o[N];
        for (std::size_t i = 0; i + N <= a.size(); i += N) {
            store(o, f(V::load(a.data() + i), V::load(b.data() + i)));
            for (int j = 0; j < N; ++j) { ++c.checks; const T w = T(ref(a[i + j], b[i + j])); if (!same(o[j], w)) c.fail(what, tn, (long long)(i + j), o[j], w, a[i + j], b[i + j]); }
        }
    }
    template <class F, class R> void compare(const char* what, const std::vector<T>& a, const std::vector<T>& b, F&& f, R&& ref) {
        for (std::size_t i = 0; i + N <= a.size(); i += N) {
            const M m = f(V::load(a.data() + i), V::load(b.data() + i));
            T o[N]; store(o, select(m, V::splat(T(1)), V::zero()));
            int cnt = 0, first = -1; std::uint64_t bits = 0;
            for (int j = 0; j < N; ++j) {
                const bool want = ref(a[i + j], b[i + j]);
                ++c.checks;
                if ((o[j] == T(1)) != want || m[j] != want) c.fail(what, tn, (long long)(i + j));
                cnt += want; if (want && first < 0) first = j; if (want && j < 64) bits |= std::uint64_t(1) << j;
            }
            if (m.count() != cnt || m.first() != first || m.any() != (cnt != 0) || m.all_set() != (cnt == N) || m.none() != (cnt == 0) || m.bits() != bits) {
                if (c.fails < 20) std::fprintf(stderr, "  mask summary of %s: count %d (%d) first %d (%d) any %d all %d none %d bits %#llx (%#llx)\n", what, m.count(), cnt,
                                               m.first(), first, int(m.any()), int(m.all_set()), int(m.none()), (unsigned long long)m.bits(), (unsigned long long)bits);
                c.fail("mask summary", tn, (long long)i);
            }
        }
    }
    template <class F, class R> void unary(const char* what, const std::vector<T>& a, F&& f, R&& ref) {
        T o[N];
        for (std::size_t i = 0; i + N <= a.size(); i += N) {
            store(o, f(V::load(a.data() + i)));
            for (int j = 0; j < N; ++j) { ++c.checks; const T w = T(ref(a[i + j])); if (!same(o[j], w)) c.fail(what, tn, (long long)(i + j), o[j], w, a[i + j], T(0)); }
        }
    }

    void arithmetic(const std::vector<T>& a, const std::vector<T>& b) {
        binary("add", a, b, [](V x, V y) { return x + y; }, [](T x, T y) { return r_add(x, y); });
        binary("sub", a, b, [](V x, V y) { return x - y; }, [](T x, T y) { return r_sub(x, y); });
        binary("mul", a, b, [](V x, V y) { return x * y; }, [](T x, T y) { return r_mul(x, y); });
        binary("min", a, b, [](V x, V y) { return min(x, y); }, [](T x, T y) { return x < y ? x : y; });
        binary("max", a, b, [](V x, V y) { return max(x, y); }, [](T x, T y) { return x > y ? x : y; });
        compare("eq", a, b, [](V x, V y) { return eq(x, y); }, [](T x, T y) { return x == y; });
        compare("ne", a, b, [](V x, V y) { return ne(x, y); }, [](T x, T y) { return x != y; });
        compare("lt", a, b, [](V x, V y) { return lt(x, y); }, [](T x, T y) { return x < y; });
        compare("le", a, b, [](V x, V y) { return le(x, y); }, [](T x, T y) { return x <= y; });
        compare("gt", a, b, [](V x, V y) { return gt(x, y); }, [](T x, T y) { return x > y; });
        compare("ge", a, b, [](V x, V y) { return ge(x, y); }, [](T x, T y) { return x >= y; });
        compare("lt&!eq|gt", a, b, [](V x, V y) { return (lt(x, y) & !eq(x, y)) | gt(x, y); }, [](T x, T y) { return x < y || x > y; });
        compare("le^ge", a, b, [](V x, V y) { return le(x, y) ^ ge(x, y); }, [](T x, T y) { return (x <= y) != (x >= y); });
        binary("select", a, b, [](V x, V y) { return select(lt(x, y), y, x); }, [](T x, T y) { return x < y ? y : x; });
        unary("abs", a, [](V x) { return abs(x); }, [](T x) { return r_abs(x); });
        unary("neg", a, [](V x) { return -x; }, [](T x) { if constexpr (std::is_floating_point_v<T>) return T(-x); else return r_sub(T(0), x); });
        if constexpr (std::is_integral_v<T>) {
            binary("add_sat", a, b, [](V x, V y) { return add_sat(x, y); }, [](T x, T y) { return r_sat_add(x, y); });
            binary("sub_sat", a, b, [](V x, V y) { return sub_sat(x, y); }, [](T x, T y) { return r_sat_sub(x, y); });
            binary("and", a, b, [](V x, V y) { return x & y; }, [](T x, T y) { return T(x & y); });
            binary("or", a, b, [](V x, V y) { return x | y; }, [](T x, T y) { return T(x | y); });
            binary("xor", a, b, [](V x, V y) { return x ^ y; }, [](T x, T y) { return T(x ^ y); });
            binary("andnot", a, b, [](V x, V y) { return bit_andnot(x, y); }, [](T x, T y) { return T(x & T(~y)); });
            const T cm = T(sizeof(T) * 8 - 1);
            binary("shl lanes", a, b, [cm](V x, V y) { return shl(x, y & V::splat(cm)); }, [cm](T x, T y) { return r_shl(x, int(y & cm)); });
            binary("shr lanes", a, b, [cm](V x, V y) { return shr(x, y & V::splat(cm)); }, [cm](T x, T y) { return r_shr(x, int(y & cm)); });
            unary("not", a, [](V x) { return ~x; }, [](T x) { return T(~x); });
            unary("popcount", a, [](V x) { return popcount(x); }, [](T x) { return r_pop(x); });
            for (int n = 0; n < int(sizeof(T) * 8); ++n) {
                unary("shl n", a, [n](V x) { return x << n; }, [n](T x) { return r_shl(x, n); });
                unary("shr n", a, [n](V x) { return x >> n; }, [n](T x) { return r_shr(x, n); });
            }
        } else {
            binary("div", a, b, [](V x, V y) { return x / y; }, [](T x, T y) { return x / y; });
        }
    }
    // conversions over the values of a (a.size() a multiple of 2N)
    void conversions(const std::vector<T>& a) {
        if constexpr (std::is_integral_v<T>) {
            using O = std::conditional_t<std::is_signed_v<T>, std::make_unsigned_t<T>, std::make_signed_t<T>>;
            for (std::size_t i = 0; i + N <= a.size(); i += N) {
                O o[N]; store(o, convert<O>(V::load(a.data() + i)));
                T back[N]; store(back, bitcast<T>(bitcast<O>(V::load(a.data() + i))));
                for (int j = 0; j < N; ++j) { ++c.checks; if (o[j] != O(a[i + j]) || back[j] != a[i + j]) c.fail("convert / bitcast", tn, (long long)(i + j)); }
            }
            if constexpr (sizeof(T) < 8) {
                using W = vt::wider<T>;
                for (std::size_t i = 0; i + N <= a.size(); i += N) {
                    W lo[N / 2], hi[N / 2];
                    store(lo, widen_lo(V::load(a.data() + i))); store(hi, widen_hi(V::load(a.data() + i)));
                    for (int j = 0; j < N / 2; ++j) { ++c.checks; if (lo[j] != W(a[i + j]) || hi[j] != W(a[i + N / 2 + j])) c.fail("widen", tn, (long long)(i + j)); }
                }
            }
            if constexpr (sizeof(T) > 1) {
                using H = vt::narrower<T>;
                using HU = std::make_unsigned_t<H>;
                for (std::size_t i = 0; i + 2 * N <= a.size(); i += 2 * N) {
                    const V lo = V::load(a.data() + i), hi = V::load(a.data() + i + N);
                    H o[2 * N]; store(o, narrow(lo, hi));
                    for (int j = 0; j < 2 * N; ++j) { ++c.checks; if (o[j] != H(a[i + j])) c.fail("narrow", tn, (long long)(i + j)); }
                    if constexpr (std::is_signed_v<T>) {
                        store(o, narrow_sat<H>(lo, hi));
                        for (int j = 0; j < 2 * N; ++j) { ++c.checks; if (o[j] != r_clamp<T, H>(a[i + j])) c.fail("narrow_sat", tn, (long long)(i + j)); }
                    }
                    HU u[2 * N]; store(u, narrow_sat<HU>(lo, hi));
                    for (int j = 0; j < 2 * N; ++j) { ++c.checks; if (u[j] != r_clamp<T, HU>(a[i + j])) c.fail("narrow_sat unsigned", tn, (long long)(i + j)); }
                }
            }
        }
    }
    void memory_and_reductions(std::uint64_t seed) {
        rng r(seed);
        auto rv = [&] { if constexpr (std::is_floating_point_v<T>) return T(double(std::int64_t(r.next() >> 12)) / 1048576.0 - 2e6); else return T(U<T>(r.next())); };
        // loads / stores at every offset of an exactly-sized block; splat, zero, iota, operator[]
        for (int off = 0; off < 3; ++off) {
            std::unique_ptr<T[]> src(new T[N + off]), dst(new T[N + off]);
            for (int i = 0; i < N + off; ++i) { src[i] = rv(); dst[i] = T(0); }
            const V v = V::load(src.get() + off);
            store(dst.get() + off, v);
            T arr[N]; v.to_array(arr);
            for (int i = 0; i < N; ++i) { ++c.checks; if (!same(dst[off + i], src[off + i]) || !same(v[i], src[off + i]) || !same(arr[i], src[off + i])) c.fail("load/store", tn, i); }
        }
        {
            T o[N]; store(o, V::splat(T(5))); for (int i = 0; i < N; ++i) if (o[i] != T(5)) c.fail("splat", tn, i);
            store(o, V::zero()); for (int i = 0; i < N; ++i) if (o[i] != T(0)) c.fail("zero", tn, i);
            store(o, V::iota(T(3), T(2))); for (int i = 0; i < N; ++i) if (o[i] != T(3 + 2 * i)) c.fail("iota", tn, i);
            c.checks += 3 * N;
        }
        // aligned load / store and the streaming store
        {
            alignas(128) T a[N], b[N], d[N];
            for (int i = 0; i < N; ++i) { a[i] = rv(); b[i] = T(0); d[i] = T(0); }
            store_aligned(b, V::load_aligned(a));
            stream(d, V::load_aligned(a)); stream_fence();
            for (int i = 0; i < N; ++i) { ++c.checks; if (!same(a[i], b[i]) || !same(a[i], d[i])) c.fail("aligned / stream", tn, i); }
        }
        // masks: first_n for every n, from_bits, and masked load / store that must not touch inactive lanes
        for (int n = 0; n <= N + 2; ++n) {
            const M m = M::first_n(std::size_t(n));
            const int want = n > N ? N : n;
            ++c.checks;
            if (m.count() != want || m.any() != (want > 0) || m.all_set() != (want == N) || m.first() != (want ? 0 : -1)) c.fail("first_n", tn, n);
            // exactly `want` readable elements: an out-of-bounds lane access is an ASan / page fault, not a pass
            std::unique_ptr<T[]> src(new T[std::size_t(want) + 1]), dst(new T[std::size_t(want) + 1]);
            for (int i = 0; i <= want; ++i) { src[i] = rv(); dst[i] = T(9); }
            const V v = V::load(src.get(), m);
            T o[N]; store(o, v);
            for (int i = 0; i < N; ++i) { ++c.checks; if (!same(o[i], i < want ? src[i] : T(0))) c.fail("masked load", tn, i); }
            store(dst.get(), V::load(src.get(), m) + V::zero(), m);
            for (int i = 0; i < want; ++i) { ++c.checks; if (!same(dst[i], src[i])) c.fail("masked store", tn, i); }
            ++c.checks; if (dst[want] != T(9)) c.fail("masked store wrote past the mask", tn, n);
        }
        for (int rep = 0; rep < 64; ++rep) {
            const std::uint64_t b = r.next() & (N >= 64 ? ~std::uint64_t(0) : ((std::uint64_t(1) << N) - 1));
            const M m = M::from_bits(b);
            T src[N], dst[N], o[N];
            for (int i = 0; i < N; ++i) { src[i] = rv(); dst[i] = T(7); }
            store(o, V::load(src, m)); store(dst, V::load(src), m);
            int cnt = 0;
            for (int i = 0; i < N; ++i) {
                const bool on = i < 64 && ((b >> i) & 1); cnt += on; ++c.checks;
                if (!same(o[i], on ? src[i] : T(0)) || !same(dst[i], on ? src[i] : T(7)) || m[i] != on) c.fail("from_bits mask load/store", tn, i);
            }
            if (m.count() != cnt || m.bits() != b) c.fail("from_bits summary", tn, rep);
        }
        // gather / scatter (scatter: the highest lane wins on repeated indices)
        {
            constexpr int TB = 1000;
            std::vector<T> table(TB); for (auto& x : table) x = rv();
            for (int rep = 0; rep < 50; ++rep) {
                std::int32_t idx[N]; T o[N], vals[N];
                for (int i = 0; i < N; ++i) { idx[i] = std::int32_t(r.next() % (rep % 5 == 0 ? 3 : TB)); vals[i] = rv(); }
                store(o, V::gather(table.data(), idx));
                for (int i = 0; i < N; ++i) { ++c.checks; if (!same(o[i], table[std::size_t(idx[i])])) c.fail("gather", tn, i); }
                std::vector<T> got = table, want = table;
                scatter(got.data(), idx, V::load(vals));
                for (int i = 0; i < N; ++i) want[std::size_t(idx[i])] = vals[i];
                ++c.checks;
                if (std::memcmp(got.data(), want.data(), TB * sizeof(T)) != 0) c.fail("scatter", tn, rep);
            }
        }
        // reductions
        for (int rep = 0; rep < 200; ++rep) {
            T x[N]; for (auto& e : x) e = rep < 3 ? T(rep) : rv();
            const V v = V::load(x);
            T t[N]; std::memcpy(t, x, sizeof t);
            for (int w = N / 2; w >= 1; w /= 2) for (int i = 0; i < w; ++i) t[i] = r_add(t[i], t[i + w]);
            T mn = x[0], mx = x[0];
            for (int i = 1; i < N; ++i) { mn = x[i] < mn ? x[i] : mn; mx = x[i] > mx ? x[i] : mx; }
            c.checks += 3;
            if (!same(reduce_add(v), t[0])) c.fail("reduce_add", tn, rep);
            if (!same(reduce_min(v), mn) || !same(reduce_max(v), mx)) c.fail("reduce_min/max", tn, rep);
            if constexpr (std::is_integral_v<T>) {
                T a = x[0], o = x[0], e = x[0];
                for (int i = 1; i < N; ++i) { a = T(a & x[i]); o = T(o | x[i]); e = T(e ^ x[i]); }
                ++c.checks;
                if (reduce_and(v) != a || reduce_or(v) != o || reduce_xor(v) != e) c.fail("reduce_and/or/xor", tn, rep);
            }
        }
    }
};

template <class T> std::vector<T> all16() { std::vector<T> v(65536); for (int i = 0; i < 65536; ++i) v[std::size_t(i)] = T(std::uint16_t(i)); return v; }

template <Target Tg> void check_small(ctx& c) {
    // every pair of 8-bit inputs, every binary operation
    for (int sg = 0; sg < 2; ++sg) {
        auto run = [&]<class T>(T) {
            std::vector<T> a(65536), b(65536);
            for (int i = 0; i < 65536; ++i) { a[std::size_t(i)] = T(std::uint8_t(i >> 8)); b[std::size_t(i)] = T(std::uint8_t(i)); }
            checker<T, Tg> k{c};
            k.arithmetic(a, b);
            k.conversions(a);   // a holds every 8-bit value 256 times
            k.memory_and_reductions(11);
        };
        if (sg) run(std::int8_t(0)); else run(std::uint8_t(0));
    }
    // every 16-bit input, every unary operation and conversion; binary operations on edge pairs and random pairs
    for (int sg = 0; sg < 2; ++sg) {
        auto run = [&]<class T>(T) {
            const std::vector<T> a = all16<T>();
            std::vector<T> b(65536);
            rng r(3); for (auto& x : b) x = T(std::uint16_t(r.next()));
            checker<T, Tg> k{c};
            k.arithmetic(a, b);            // unary ops see all 2^16 inputs; binary ops all inputs against random partners
            std::vector<T> ea, eb; cross(samples<T>(40, 5), std::size_t(2 * vec<T, Tg>::N), ea, eb);
            k.arithmetic(ea, eb);
            k.conversions(a);
            k.memory_and_reductions(12);
        };
        if (sg) run(std::int16_t(0)); else run(std::uint16_t(0));
    }
}
template <class T, Target Tg> void check_wide(ctx& c, std::uint64_t seed) {
    std::vector<T> a, b; cross(samples<T>(120, seed), std::size_t(2 * vec<T, Tg>::N), a, b);
    checker<T, Tg> k{c};
    k.arithmetic(a, b);
    k.conversions(a);
    k.memory_and_reductions(seed + 100);
}

template <Target Tg> void check_float_interop(ctx& c) {
    using VF = vec<float, Tg>; constexpr int N = VF::N;
    using VI = vec<std::int32_t, Tg>;
    if (VI::N != N) { c.fail("vec<int32> lanes != vec<float> lanes", "f32", VI::N); return; }
    rng r(21);
    // int32 <-> float, bitcasts
    for (int rep = 0; rep < 4000; ++rep) {
        float f[N], of[N]; std::int32_t i32[N], oi[N];
        for (int i = 0; i < N; ++i) { i32[i] = std::int32_t(r.next() >> (rep % 3 ? 33 : 44)) - (rep % 2 ? 1000 : 0); f[i] = float(double(std::int64_t(r.next() >> 23)) / 4096.0 - 1e6)   /* < 2^29: to_int stays in int32 range */ * (rep % 7 ? 1.f : 1e-3f); }
        store(oi, to_int(forge::simd::load<Tg>(f))); forge::simd::store(of, to_float(VI::load(i32)));
        for (int i = 0; i < N; ++i) { ++c.checks; if (oi[i] != std::int32_t(f[i]) || !same(of[i], float(i32[i]))) c.fail("to_int / to_float", "f32", rep); }
        std::uint32_t ob[N]; store(ob, bitcast<std::uint32_t>(forge::simd::load<Tg>(f)));
        forge::simd::store(of, bitcast_f32(vec<std::uint32_t, Tg>::load(ob)));
        for (int i = 0; i < N; ++i) { ++c.checks; if (ob[i] != std::bit_cast<std::uint32_t>(f[i]) || !same(of[i], f[i])) c.fail("bitcast f32", "f32", rep); }
        // float <-> double
        double lo[N / 2], hi[N / 2];
        store(lo, promote_lo(forge::simd::load<Tg>(f))); store(hi, promote_hi(forge::simd::load<Tg>(f)));
        for (int i = 0; i < N / 2; ++i) { ++c.checks; if (lo[i] != double(f[i]) || hi[i] != double(f[N / 2 + i])) c.fail("promote", "f32", rep); }
        double d[N]; for (int i = 0; i < N; ++i) d[i] = double(f[i]) * 1.0000001 + 1e-9;
        forge::simd::store(of, demote(vec<double, Tg>::load(d), vec<double, Tg>::load(d + N / 2)));
        for (int i = 0; i < N; ++i) { ++c.checks; if (!same(of[i], float(d[i]))) c.fail("demote", "f32", rep); }
    }
    // int64 <-> double, sqrt, fma
    {
        using VD = vec<double, Tg>; constexpr int ND = VD::N;
        for (int rep = 0; rep < 4000; ++rep) {
            double x[ND], y[ND], z[ND], o[ND]; std::int64_t q[ND], oq[ND];
            for (int i = 0; i < ND; ++i) {
                x[i] = double(std::int64_t(r.next() >> 11)) / 65536.0 - 3e10; y[i] = double(std::int64_t(r.next() >> 30)) / 1024.0 - 1e6;
                z[i] = -x[i] * y[i] * (1.0 + 1e-15 * double(r.next() % 7)); q[i] = std::int64_t(r.next()) >> (rep % 4 ? 11 : 0);
            }
            store(oq, convert<std::int64_t>(VD::load(x))); store(o, convert<double>(vec<std::int64_t, Tg>::load(q)));
            for (int i = 0; i < ND; ++i) { ++c.checks; if (oq[i] != std::int64_t(x[i]) || !same(o[i], double(q[i]))) c.fail("int64 <-> double", "f64", rep); }
            store(o, sqrt(abs(VD::load(x))));
            for (int i = 0; i < ND; ++i) { ++c.checks; const double w = std::sqrt(std::fabs(x[i])); if (!same(o[i], w)) c.fail("sqrt", "f64", rep, o[i], w, x[i], 0.0); }
            store(o, fma(VD::load(x), VD::load(y), VD::load(z)));
            for (int i = 0; i < ND; ++i) { ++c.checks; const double w = std::fma(x[i], y[i], z[i]); if (!same(o[i], w)) c.fail("fma", "f64", rep, o[i], w, x[i], y[i]); }
        }
    }
    // half precision: every bit pattern in, edge and random floats out; bfloat16 likewise
    {
        std::vector<std::uint16_t> h(65536 + N); for (std::size_t i = 0; i < h.size(); ++i) h[i] = std::uint16_t(i);
        float o[N];
        for (std::size_t i = 0; i + N <= 65536; i += N) {
            forge::simd::store(o, load_f16<Tg>(h.data() + i));
            for (int j = 0; j < N; ++j) { ++c.checks; const float w = float(forge::f16::from_bits(h[i + j])); if (!(o[j] == w && std::signbit(o[j]) == std::signbit(w)) && !(o[j] != o[j] && w != w)) c.fail("load_f16", "f16", (long long)(i + j)); }
            forge::simd::store(o, load_bf16<Tg>(h.data() + i));
            for (int j = 0; j < N; ++j) { ++c.checks; const float w = float(forge::bf16::from_bits(h[i + j])); if (std::memcmp(&o[j], &w, 4) != 0) c.fail("load_bf16", "bf16", (long long)(i + j)); }
        }
        std::vector<float> f;
        for (float x : {0.f, -0.f, 1.f, -1.f, 65504.f, 65519.99f, 65520.f, 1e10f, -1e10f, 6.1e-5f, 5.96e-8f, 2.98e-8f, 2.9802322e-8f, 1e-10f, 1.00048828125f, 1.000732421875f,
                        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) f.push_back(x);
        for (int i = 0; i < 200000; ++i) { std::uint32_t b = std::uint32_t(r.next()); if (i % 2) b = (b & 0x807fffffu) | ((std::uint32_t(r.next() % 60) + 97u) << 23); f.push_back(std::bit_cast<float>(b)); }
        while (f.size() % std::size_t(N)) f.push_back(1.f);
        std::uint16_t oh[N];
        for (std::size_t i = 0; i + N <= f.size(); i += N) {
            store_f16<Tg>(oh, forge::simd::load<Tg>(f.data() + i));
            for (int j = 0; j < N; ++j) {
                ++c.checks; const float x = f[i + j]; const std::uint16_t w = forge::f16(x).bits;
                if (x != x ? (oh[j] & 0x7c00) != 0x7c00 || (oh[j] & 0x3ff) == 0 : oh[j] != w) c.fail("store_f16", "f16", (long long)(i + j), std::uint32_t(oh[j]), std::uint32_t(w), std::bit_cast<std::uint32_t>(x), 0u);
            }
            store_bf16<Tg>(oh, forge::simd::load<Tg>(f.data() + i));
            for (int j = 0; j < N; ++j) { ++c.checks; if (oh[j] != forge::bf16(f[i + j]).bits) c.fail("store_bf16", "bf16", (long long)(i + j)); }
        }
    }
}

// Runs every check for Tg; returns the number of failed checks.
template <Target Tg> int check(const char* target) {
    ctx c{target};
    check_small<Tg>(c);
    check_wide<std::int32_t, Tg>(c, 31); check_wide<std::uint32_t, Tg>(c, 32);
    check_wide<std::int64_t, Tg>(c, 33); check_wide<std::uint64_t, Tg>(c, 34);
    check_wide<double, Tg>(c, 35);
    check_float_interop<Tg>(c);
    std::printf("  %-10s %2d-byte registers (%s): %ld checks, %d failed\n", target, vec<std::int8_t, Tg>::bytes,
                vec<std::int8_t, Tg>::E ? "portable arrays" : "GNU vectors", c.checks, c.fails);
    return c.fails;
}
} // namespace
} // namespace forge_types
