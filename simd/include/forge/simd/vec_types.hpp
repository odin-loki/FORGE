#pragma once
// Tier P element types beyond float: vec<T, Tg> and mask<T, Tg> for double, int8/16/32/64 and uint8/16/32/64,
// plus f16 / bf16 loads and stores of vec<float, Tg>. Included on demand (vec.hpp stays float only).
//
// Width. A vec<T, Tg> is one native register of the target: reg_bytes / sizeof(T) lanes, the register being
// the one vec<float, Tg> uses in this translation unit (64 bytes for AVX-512, 32 for AVX2 / LASX, 16 for SSE,
// NEON, LSX, VSX, VXE, SIMD128 and MSA, the fixed length for SVE / RVV). As with vec<float>, the native width is
// only used in a translation unit compiled with the matching flags; elsewhere (and for scalar_t) the vector
// is 16 bytes.
//
// Implementation. On GCC and Clang the register is a GNU vector type and the element-wise operations are
// written with vector operators and builtins, which the compiler lowers to the instructions of whatever ISA
// the translation unit is built for: the same source is the SSE, AVX2, AVX-512, NEON, SVE, RVV, LSX / LASX,
// VSX, VXE, SIMD128 and MSA backend. ISA intrinsics are used where there is no generic form: predicated loads
// and stores (AVX-512, AVX2 for 32/64-bit lanes, SVE, RVV), gathers and scatters (AVX2 gathers, AVX-512, SVE,
// RVV; 32/64-bit lanes), non-temporal stores (x86; the compiler builtin elsewhere under Clang) and half
// precision conversion (F16C, AVX-512, AArch64 NEON). Everything else is emulated lane by lane without touching
// memory outside the active lanes. MSVC has no GNU vector types: every target uses the portable array
// implementation there (`emulated_t` selects it on any compiler; the tests run it against the same references).
//
// Semantics (the same on every backend; tests/test_simd_types.cpp checks them against scalar references,
// exhaustively for the 8-bit binary and 16-bit unary operations):
//   + - *            integers wrap modulo 2^w; double is IEEE (no contraction is introduced here)
//   min, max         min(a, b) = a < b ? a : b, max(a, b) = a > b ? a : b (double: the second operand when unordered)
//   abs              integers wrap (abs(INT_MIN) = INT_MIN); unsigned: identity; double: clears the sign bit
//   add_sat, sub_sat clamp to the range of T
//   shl, shr         by one count or per-lane counts, 0 <= count < w (anything else is a precondition
//                    violation); shr is arithmetic for signed T, logical for unsigned T
//   compares         eq ne lt le gt ge -> mask<T, Tg>; double compares are ordered except ne
//   convert<U>       same lane width: the C++ value conversion per lane (double -> int64 truncates; out of
//                    range is a precondition violation)
//   widen_lo/hi      the low / high half of the lanes, value-converted to the type twice as wide
//   narrow           two vectors -> one of the half-width type, truncating; narrow_sat<U> clamps to U's range
//   gather, scatter  N indices of int32 (0 <= index < 2^28); scatter stores in lane order, so the highest lane wins
//   reduce_*         integers: exact; double reduce_add: the fixed tree (v[i] + v[i + N/2]) halved down to one lane
#include <forge/simd/vec.hpp>
#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

#if (defined(__GNUC__) || defined(__clang__)) && !defined(FORGE_VEC_TYPES_NO_GNU)
#define FORGE_VT_GNU 1
#else
#define FORGE_VT_GNU 0
#endif
// GNU vector types are used only where the translation unit is compiled for a vector ISA. Without one the
// compiler would scalarise every vector operation in its type legaliser (a little-used path: zig 0.17's
// clang miscompiles int32 vector compares for baseline mips64 there), so those builds use the portable arrays.
#if FORGE_VT_GNU && !(defined(__SSE2__) || defined(__ARM_NEON) || defined(__riscv_v) || defined(__loongarch_sx) || defined(__ALTIVEC__) || \
                      defined(__VSX__) || defined(__VX__) || defined(__VEC__) || defined(__wasm_simd128__) || defined(__mips_msa))
#undef FORGE_VT_GNU
#define FORGE_VT_GNU 0
#endif
#if defined(__has_builtin)
#define FORGE_VT_HAS(x) __has_builtin(x)
#else
#define FORGE_VT_HAS(x) 0
#endif
// Helpers that do not depend on the target are always inlined: a shared out-of-line copy could come from a
// translation unit built for another ISA.
#if defined(__GNUC__) || defined(__clang__)
#define FORGE_VT_INL inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define FORGE_VT_INL __forceinline
#else
#define FORGE_VT_INL inline
#endif
#if FORGE_VT_GNU
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpsabi"
#endif

namespace forge::simd {

// A target whose vec<T> always uses the portable array implementation (16-byte registers).
struct emulated_t : caps<> { using caps_type = caps<>; static constexpr const char* name = "emulated"; };

namespace vt {
template <class T> concept elem = std::same_as<T, double> || std::same_as<T, std::int8_t> || std::same_as<T, std::uint8_t> ||
    std::same_as<T, std::int16_t> || std::same_as<T, std::uint16_t> || std::same_as<T, std::int32_t> || std::same_as<T, std::uint32_t> ||
    std::same_as<T, std::int64_t> || std::same_as<T, std::uint64_t>;
template <class T> concept ielem = elem<T> && std::integral<T>;
template <int Sz> struct ints;
template <> struct ints<1> { using s = std::int8_t; using u = std::uint8_t; };
template <> struct ints<2> { using s = std::int16_t; using u = std::uint16_t; };
template <> struct ints<4> { using s = std::int32_t; using u = std::uint32_t; };
template <> struct ints<8> { using s = std::int64_t; using u = std::uint64_t; };
template <class T> using mi = typename ints<int(sizeof(T))>::s;   // the signed integer of T's width (mask lanes)
template <class T> using mu = typename ints<int(sizeof(T))>::u;
template <class T> using arith = std::conditional_t<std::floating_point<T>, T, mu<T>>;   // wrapping arithmetic is done unsigned
template <class T> using wider = std::conditional_t<std::floating_point<T>, double, std::conditional_t<std::signed_integral<T>,
    typename ints<(int(sizeof(T)) < 8 ? int(sizeof(T)) * 2 : 8)>::s, typename ints<(int(sizeof(T)) < 8 ? int(sizeof(T)) * 2 : 8)>::u>>;
template <class T> using narrower = std::conditional_t<std::signed_integral<T>, typename ints<(int(sizeof(T)) > 1 ? int(sizeof(T)) / 2 : 1)>::s,
    typename ints<(int(sizeof(T)) > 1 ? int(sizeof(T)) / 2 : 1)>::u>;

template <Target Tg> constexpr bool emulated() { return !FORGE_VT_GNU || std::is_base_of_v<emulated_t, Tg>; }
template <Target Tg> constexpr int reg_bytes() { return emulated<Tg>() || impl::kind<Tg>() == 0 ? 16 : vec<float, Tg>::N * 4; }

template <class T, int N> struct earr { T v[N]; };
#if FORGE_VT_GNU
template <class T, int Bytes> struct gnu { typedef T type __attribute__((vector_size(Bytes))); };
template <class T, int Bytes, bool E> using reg = std::conditional_t<E, earr<T, Bytes / int(sizeof(T))>, typename gnu<T, Bytes>::type>;
#else
template <class T, int Bytes, bool E> using reg = earr<T, Bytes / int(sizeof(T))>;
#endif
// lanes <-> array (both representations store lane i at byte offset i * sizeof(T))
template <class R, class T> FORGE_VT_INL void to_arr(const R& r, T* t) { std::memcpy(t, &r, sizeof(R)); }
template <class R, class T> FORGE_VT_INL void from_arr(R& r, const T* t) { std::memcpy(&r, t, sizeof(R)); }

// scalar forms (the emulated backend, and the definition of each operation)
template <class T> FORGE_VT_INL constexpr T wadd(T x, T y) { if constexpr (std::floating_point<T>) return x + y; else return T(std::uint64_t(mu<T>(x)) + std::uint64_t(mu<T>(y))); }
template <class T> FORGE_VT_INL constexpr T wsub(T x, T y) { if constexpr (std::floating_point<T>) return x - y; else return T(std::uint64_t(mu<T>(x)) - std::uint64_t(mu<T>(y))); }
template <class T> FORGE_VT_INL constexpr T wmul(T x, T y) { if constexpr (std::floating_point<T>) return x * y; else return T(std::uint64_t(mu<T>(x)) * std::uint64_t(mu<T>(y))); }
template <class T> FORGE_VT_INL constexpr T sat_add(T x, T y) {
    if constexpr (std::unsigned_integral<T>) { T s = wadd(x, y); return s < x ? std::numeric_limits<T>::max() : s; }
    else { const T s = wadd(x, y); const bool ovf = (x < 0) == (y < 0) && (s < 0) != (x < 0); return ovf ? (x < 0 ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max()) : s; }
}
template <class T> FORGE_VT_INL constexpr T sat_sub(T x, T y) {
    if constexpr (std::unsigned_integral<T>) return x > y ? T(x - y) : T(0);
    else { const T d = wsub(x, y); const bool ovf = (x < 0) != (y < 0) && (d < 0) != (x < 0); return ovf ? (x < 0 ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max()) : d; }
}
template <class T> FORGE_VT_INL constexpr T sabs(T x) {
    if constexpr (std::floating_point<T>) return std::bit_cast<T>(std::bit_cast<std::uint64_t>(x) & 0x7fffffffffffffffull);
    else if constexpr (std::unsigned_integral<T>) return x;
    else return x < 0 ? wsub(T(0), x) : x;
}
template <class T> FORGE_VT_INL constexpr T shl1(T x, int n) { return T(std::uint64_t(mu<T>(x)) << n); }
template <class T> FORGE_VT_INL constexpr T shr1(T x, int n) { return T(x >> n); }   // C++20: arithmetic for signed
template <class T, class U> FORGE_VT_INL constexpr U clampto(T x) {   // saturating value conversion T -> U (integers)
    using L = std::numeric_limits<U>;
    if constexpr (std::signed_integral<T> && std::signed_integral<U>) return x < T(L::min()) ? L::min() : x > T(L::max()) ? L::max() : U(x);
    else if constexpr (std::signed_integral<T>) return x < 0 ? U(0) : std::uint64_t(x) > std::uint64_t(L::max()) ? L::max() : U(x);
    else return std::uint64_t(x) > std::uint64_t(L::max()) ? L::max() : U(x);
}
} // namespace vt

#define FORGE_VT_T template <vt::elem T, Target Tg>
#define FORGE_VT_TI template <vt::ielem T, Target Tg>
#if FORGE_VT_GNU
#define FORGE_VT_IFG(...) if constexpr (!vec<T, Tg>::E) { __VA_ARGS__ } else
#else
#define FORGE_VT_IFG(...)
#endif
#define FORGE_VT_LANES for (int i = 0; i < vec<T, Tg>::N; ++i)

template <class T, Target Tg> struct mask {
    static_assert(vt::elem<T>, "forge::simd::mask<T, Tg>: T must be float (vec.hpp), double or a fixed-width integer");
    static constexpr bool E = vt::emulated<Tg>();
    static constexpr int bytes = vt::reg_bytes<Tg>();
    static constexpr int N = bytes / int(sizeof(T));
    using lane_t = vt::mi<T>;                    // every lane is all ones (-1) or zero
    using native_t = vt::reg<lane_t, bytes, E>;
    native_t m;
    static mask first_n(std::size_t n) {         // lanes [0, min(n, N)) set
        lane_t t[N];
        for (int i = 0; i < N; ++i) t[i] = lane_t(-lane_t(std::size_t(i) < n));
        mask r; vt::from_arr(r.m, t); return r;
    }
    static mask all() { return first_n(std::size_t(N)); }
    static mask none_set() { return first_n(0); }
    static mask from_bits(std::uint64_t b) {     // lane i from bit i (N <= 64)
        lane_t t[N];
        for (int i = 0; i < N; ++i) t[i] = lane_t(-lane_t(i < 64 && ((b >> i) & 1)));
        mask r; vt::from_arr(r.m, t); return r;
    }
    bool operator[](int i) const { lane_t t[N]; vt::to_arr(m, t); return t[i] != 0; }
    // bit i = lane i; lanes past 63 (8-bit lanes of a 1024-bit register) are not representable: use count / first
    std::uint64_t bits() const {
        lane_t t[N]; vt::to_arr(m, t);
        std::uint64_t b = 0;
        for (int i = 0; i < N && i < 64; ++i) b |= std::uint64_t(t[i] & 1) << i;
        return b;
    }
    int count() const { lane_t t[N]; vt::to_arr(m, t); int c = 0; for (int i = 0; i < N; ++i) c += int(t[i] & 1); return c; }
    bool any() const { lane_t t[N]; vt::to_arr(m, t); lane_t o = 0; for (int i = 0; i < N; ++i) o = lane_t(o | t[i]); return o != 0; }
    bool all_set() const { lane_t t[N]; vt::to_arr(m, t); lane_t a = lane_t(-1); for (int i = 0; i < N; ++i) a = lane_t(a & t[i]); return a != 0; }
    bool none() const { return !any(); }
    int first() const { lane_t t[N]; vt::to_arr(m, t); for (int i = 0; i < N; ++i) if (t[i]) return i; return -1; }
};

template <class T, Target Tg> struct vec {
    static_assert(vt::elem<T>, "forge::simd::vec<T, Tg>: T must be float (vec.hpp), double or a fixed-width integer");
    static constexpr bool E = vt::emulated<Tg>();
    static constexpr int bytes = vt::reg_bytes<Tg>();
    static constexpr int N = bytes / int(sizeof(T));
    using value_type = T;
    using mask_t = mask<T, Tg>;
    using native_t = vt::reg<T, bytes, E>;
    native_t r;
    native_t native() const { return r; }
    static vec from_native(native_t n) { vec v; v.r = n; return v; }
    static constexpr int size() { return N; }
    T operator[](int i) const { T t[N]; vt::to_arr(r, t); return t[i]; }
    void to_array(T* out) const { vt::to_arr(r, out); }
    static vec zero() { vec v; T t[N] = {}; vt::from_arr(v.r, t); return v; }
    static vec splat(T s) { vec v; T t[N]; for (auto& x : t) x = s; vt::from_arr(v.r, t); return v; }
    static vec iota(T start = T(0), T step = T(1)) { vec v; T t[N]; for (int i = 0; i < N; ++i) t[i] = vt::wadd(start, vt::wmul(T(i), step)); vt::from_arr(v.r, t); return v; }
    static vec load(const T* p) { vec v; std::memcpy(&v.r, p, std::size_t(bytes)); return v; }                      // any alignment
    static vec load_aligned(const T* p) {                                                                          // p % bytes == 0
#if FORGE_VT_GNU
        p = static_cast<const T*>(__builtin_assume_aligned(p, std::size_t(bytes)));
#endif
        vec v; std::memcpy(&v.r, p, std::size_t(bytes)); return v;
    }
    static vec load(const T* p, mask_t m);        // lanes outside the mask read as 0 and never touch memory
    static vec gather(const T* base, const std::int32_t* idx);   // lane i = base[idx[i]]
};

// ---------------------------------------------------------------- ISA glue (GNU vector <-> intrinsic types)
#if FORGE_VT_GNU
namespace vt {
#if defined(FORGE_HAVE_SVE_FIXED)
template <class T> struct svfx;
#define FORGE_VT_SV(T, S) template <> struct svfx<T> { typedef S type __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS))); };
FORGE_VT_SV(std::int8_t, svint8_t) FORGE_VT_SV(std::uint8_t, svuint8_t) FORGE_VT_SV(std::int16_t, svint16_t) FORGE_VT_SV(std::uint16_t, svuint16_t)
FORGE_VT_SV(std::int32_t, svint32_t) FORGE_VT_SV(std::uint32_t, svuint32_t) FORGE_VT_SV(std::int64_t, svint64_t) FORGE_VT_SV(std::uint64_t, svuint64_t)
FORGE_VT_SV(double, svfloat64_t) FORGE_VT_SV(float, svfloat32_t)
#undef FORGE_VT_SV
// predicate of the lanes whose mask lane is non-zero
template <class T, Target Tg> inline svbool_t sv_pred(const mask<T, Tg>& m) {
    typename svfx<mi<T>>::type v = m.m;
    return svcmpne(svptrue_b8(), v, mi<T>(0));
}
#endif
#if defined(FORGE_HAVE_RVV_FIXED)
template <class T> struct rvfx;
#define FORGE_VT_RV(T, S) template <> struct rvfx<T> { typedef S type __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen))); };
FORGE_VT_RV(std::int8_t, vint8m1_t) FORGE_VT_RV(std::uint8_t, vuint8m1_t) FORGE_VT_RV(std::int16_t, vint16m1_t) FORGE_VT_RV(std::uint16_t, vuint16m1_t)
FORGE_VT_RV(std::int32_t, vint32m1_t) FORGE_VT_RV(std::uint32_t, vuint32m1_t) FORGE_VT_RV(std::int64_t, vint64m1_t) FORGE_VT_RV(std::uint64_t, vuint64m1_t)
FORGE_VT_RV(double, vfloat64m1_t)
#undef FORGE_VT_RV
#endif
// which intrinsic family applies to vec<T, Tg> in this translation unit
template <class T, Target Tg> constexpr int isa() {   // 0 none, 2 AVX-512, 1 AVX2, 4 SVE, 5 RVV
    if constexpr (emulated<Tg>()) return 0;
    else return impl::kind<Tg>() == 1 || impl::kind<Tg>() == 2 || impl::kind<Tg>() == 4 || impl::kind<Tg>() == 5 ? impl::kind<Tg>() : 0;
}
} // namespace vt
#endif

// ---------------------------------------------------------------- loads, stores
template <class T, Target Tg> inline vec<T, Tg> vec<T, Tg>::load(const T* p, mask<T, Tg> m) {
    vec o;
#if FORGE_VT_GNU
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
    if constexpr (vt::isa<T, Tg>() == 2) {
        const __m512i mm = (__m512i)m.m;
        if constexpr (sizeof(T) == 1) o.r = (native_t)_mm512_maskz_loadu_epi8(_mm512_movepi8_mask(mm), p);
        else if constexpr (sizeof(T) == 2) o.r = (native_t)_mm512_maskz_loadu_epi16(_mm512_movepi16_mask(mm), p);
        else if constexpr (sizeof(T) == 4) o.r = (native_t)_mm512_maskz_loadu_epi32(_mm512_movepi32_mask(mm), p);
        else o.r = (native_t)_mm512_maskz_loadu_epi64(_mm512_movepi64_mask(mm), p);
        return o;
    }
#endif
#if defined(__AVX2__)
    if constexpr (vt::isa<T, Tg>() == 1 && sizeof(T) >= 4) {
        if constexpr (sizeof(T) == 4) o.r = (native_t)_mm256_maskload_epi32(reinterpret_cast<const int*>(p), (__m256i)m.m);
        else o.r = (native_t)_mm256_maskload_epi64(reinterpret_cast<const long long*>(p), (__m256i)m.m);
        return o;
    }
#endif
#if defined(FORGE_HAVE_SVE_FIXED)
    if constexpr (vt::isa<T, Tg>() == 4) { typename vt::svfx<T>::type v = svld1(vt::sv_pred(m), p); o.r = v; return o; }
#endif
#if defined(FORGE_HAVE_RVV_FIXED)
    if constexpr (vt::isa<T, Tg>() == 5) {
        const std::size_t vl = std::size_t(N);
        typename vt::rvfx<vt::mi<T>>::type mv = m.m;
        typename vt::rvfx<T>::type z = vec::zero().r, v;
        const auto pg = __riscv_vmsne(mv, vt::mi<T>(0), vl);
        if constexpr (sizeof(T) == 1) v = __riscv_vle8_mu(pg, z, p, vl);
        else if constexpr (sizeof(T) == 2) v = __riscv_vle16_mu(pg, z, p, vl);
        else if constexpr (sizeof(T) == 4) v = __riscv_vle32_mu(pg, z, p, vl);
        else v = __riscv_vle64_mu(pg, z, p, vl);
        o.r = v; return o;
    }
#endif
#endif
    typename mask<T, Tg>::lane_t mt[N]; vt::to_arr(m.m, mt);
    T t[N] = {};
    for (int i = 0; i < N; ++i) if (mt[i]) t[i] = p[i];
    vt::from_arr(o.r, t);
    return o;
}
FORGE_VT_T inline void store(T* p, vec<T, Tg> v) { std::memcpy(p, &v.r, std::size_t(vec<T, Tg>::bytes)); }
FORGE_VT_T inline void store_aligned(T* p, vec<T, Tg> v) {
#if FORGE_VT_GNU
    p = static_cast<T*>(__builtin_assume_aligned(p, std::size_t(vec<T, Tg>::bytes)));
#endif
    std::memcpy(p, &v.r, std::size_t(vec<T, Tg>::bytes));
}
// Masked store: lanes outside the mask leave memory untouched (and are never written).
FORGE_VT_T inline void store(T* p, vec<T, Tg> v, mask<T, Tg> m) {
    constexpr int N = vec<T, Tg>::N;
#if FORGE_VT_GNU
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
    if constexpr (vt::isa<T, Tg>() == 2) {
        const __m512i mm = (__m512i)m.m, x = (__m512i)v.r;
        if constexpr (sizeof(T) == 1) _mm512_mask_storeu_epi8(p, _mm512_movepi8_mask(mm), x);
        else if constexpr (sizeof(T) == 2) _mm512_mask_storeu_epi16(p, _mm512_movepi16_mask(mm), x);
        else if constexpr (sizeof(T) == 4) _mm512_mask_storeu_epi32(p, _mm512_movepi32_mask(mm), x);
        else _mm512_mask_storeu_epi64(p, _mm512_movepi64_mask(mm), x);
        return;
    }
#endif
#if defined(__AVX2__)
    if constexpr (vt::isa<T, Tg>() == 1 && sizeof(T) >= 4) {
        if constexpr (sizeof(T) == 4) _mm256_maskstore_epi32(reinterpret_cast<int*>(p), (__m256i)m.m, (__m256i)v.r);
        else _mm256_maskstore_epi64(reinterpret_cast<long long*>(p), (__m256i)m.m, (__m256i)v.r);
        return;
    }
#endif
#if defined(FORGE_HAVE_SVE_FIXED)
    if constexpr (vt::isa<T, Tg>() == 4) { typename vt::svfx<T>::type x = v.r; svst1(vt::sv_pred(m), p, x); return; }
#endif
#if defined(FORGE_HAVE_RVV_FIXED)
    if constexpr (vt::isa<T, Tg>() == 5) {
        const std::size_t vl = std::size_t(N);
        typename vt::rvfx<vt::mi<T>>::type mv = m.m;
        typename vt::rvfx<T>::type x = v.r;
        const auto pg = __riscv_vmsne(mv, vt::mi<T>(0), vl);
        if constexpr (sizeof(T) == 1) __riscv_vse8(pg, p, x, vl);
        else if constexpr (sizeof(T) == 2) __riscv_vse16(pg, p, x, vl);
        else if constexpr (sizeof(T) == 4) __riscv_vse32(pg, p, x, vl);
        else __riscv_vse64(pg, p, x, vl);
        return;
    }
#endif
#endif
    typename mask<T, Tg>::lane_t mt[N]; vt::to_arr(m.m, mt);
    T t[N]; vt::to_arr(v.r, t);
    for (int i = 0; i < N; ++i) if (mt[i]) p[i] = t[i];
}
// Non-temporal (streaming) store: p must be aligned to the register size (vec<T, Tg>::bytes). x86: movntdq /
// vmovntdq; other ISAs: the compiler's non-temporal hint under Clang (AArch64 stnp), an ordinary store otherwise.
FORGE_VT_T inline void stream(T* p, vec<T, Tg> v) {
#if FORGE_VT_GNU
    if constexpr (!vec<T, Tg>::E) {
#if defined(__AVX512F__)
        if constexpr (vec<T, Tg>::bytes == 64) { _mm512_stream_si512(reinterpret_cast<__m512i*>(p), (__m512i)v.r); return; }
#endif
#if defined(__AVX__)
        if constexpr (vec<T, Tg>::bytes == 32) { _mm256_stream_si256(reinterpret_cast<__m256i*>(p), (__m256i)v.r); return; }
#endif
#if defined(__SSE2__)
        if constexpr (vec<T, Tg>::bytes == 16) { _mm_stream_si128(reinterpret_cast<__m128i*>(p), (__m128i)v.r); return; }
#endif
#if defined(__clang__) && !defined(__SSE2__)
        __builtin_nontemporal_store(v.r, reinterpret_cast<typename vec<T, Tg>::native_t*>(p));
        return;
#endif
    }
#endif
    store_aligned(p, v);
}
// After a run of stream() calls and before the data is read by another thread (x86: sfence; elsewhere nothing).
FORGE_VT_INL void stream_fence() {
#if defined(__SSE2__) || defined(_M_X64)
    _mm_sfence();
#endif
}

// ---------------------------------------------------------------- gather / scatter
template <class T, Target Tg> inline vec<T, Tg> vec<T, Tg>::gather(const T* base, const std::int32_t* idx) {
    vec o;
#if FORGE_VT_GNU
#if defined(__AVX512F__)
    if constexpr (vt::isa<T, Tg>() == 2 && sizeof(T) >= 4) {
        if constexpr (sizeof(T) == 4) o.r = (native_t)_mm512_i32gather_epi32(_mm512_loadu_si512(idx), base, 4);
        else o.r = (native_t)_mm512_i32gather_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(idx)), base, 8);
        return o;
    }
#endif
#if defined(__AVX2__)
    if constexpr (vt::isa<T, Tg>() == 1 && sizeof(T) >= 4) {
        if constexpr (sizeof(T) == 4) o.r = (native_t)_mm256_i32gather_epi32(reinterpret_cast<const int*>(base), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(idx)), 4);
        else o.r = (native_t)_mm256_i32gather_epi64(reinterpret_cast<const long long*>(base), _mm_loadu_si128(reinterpret_cast<const __m128i*>(idx)), 8);
        return o;
    }
#endif
#if defined(FORGE_HAVE_SVE_FIXED)
    if constexpr (vt::isa<T, Tg>() == 4 && sizeof(T) >= 4) {
        typename vt::svfx<T>::type v;
        if constexpr (sizeof(T) == 4) v = svld1_gather_index(svptrue_b32(), base, svld1_s32(svptrue_b32(), idx));
        else v = svld1_gather_index(svptrue_b64(), base, svld1sw_s64(svptrue_b64(), idx));
        o.r = v; return o;
    }
#endif
#if defined(FORGE_HAVE_RVV_FIXED)
    if constexpr (vt::isa<T, Tg>() == 5) {
        const std::size_t vl = std::size_t(N);
        const auto* ui = reinterpret_cast<const std::uint32_t*>(idx);
        typename vt::rvfx<T>::type v;
        if constexpr (sizeof(T) == 1) v = __riscv_vluxei32(base, __riscv_vle32_v_u32m4(ui, vl), vl);
        else if constexpr (sizeof(T) == 2) v = __riscv_vluxei32(base, __riscv_vsll(__riscv_vle32_v_u32m2(ui, vl), 1, vl), vl);
        else if constexpr (sizeof(T) == 4) v = __riscv_vluxei32(base, __riscv_vsll(__riscv_vle32_v_u32m1(ui, vl), 2, vl), vl);
        else v = __riscv_vluxei32(base, __riscv_vsll(__riscv_vle32_v_u32mf2(ui, vl), 3, vl), vl);
        o.r = v; return o;
    }
#endif
#endif
    T t[N];
    for (int i = 0; i < N; ++i) t[i] = base[idx[i]];
    vt::from_arr(o.r, t);
    return o;
}
// base[idx[i]] = lane i, in lane order (with repeated indices the highest lane wins).
FORGE_VT_T inline void scatter(T* base, const std::int32_t* idx, vec<T, Tg> v) {
    constexpr int N = vec<T, Tg>::N;
#if FORGE_VT_GNU
#if defined(__AVX512F__)
    if constexpr (vt::isa<T, Tg>() == 2 && sizeof(T) >= 4) {
        if constexpr (sizeof(T) == 4) _mm512_i32scatter_epi32(base, _mm512_loadu_si512(idx), (__m512i)v.r, 4);
        else _mm512_i32scatter_epi64(base, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(idx)), (__m512i)v.r, 8);
        return;
    }
#endif
#if defined(FORGE_HAVE_SVE_FIXED)
    if constexpr (vt::isa<T, Tg>() == 4 && sizeof(T) >= 4) {
        typename vt::svfx<T>::type x = v.r;
        if constexpr (sizeof(T) == 4) svst1_scatter_index(svptrue_b32(), base, svld1_s32(svptrue_b32(), idx), x);
        else svst1_scatter_index(svptrue_b64(), base, svld1sw_s64(svptrue_b64(), idx), x);
        return;
    }
#endif
#if defined(FORGE_HAVE_RVV_FIXED)
    if constexpr (vt::isa<T, Tg>() == 5) {   // ordered indexed store: element order is lane order
        const std::size_t vl = std::size_t(N);
        const auto* ui = reinterpret_cast<const std::uint32_t*>(idx);
        typename vt::rvfx<T>::type x = v.r;
        if constexpr (sizeof(T) == 1) __riscv_vsoxei32(base, __riscv_vle32_v_u32m4(ui, vl), x, vl);
        else if constexpr (sizeof(T) == 2) __riscv_vsoxei32(base, __riscv_vsll(__riscv_vle32_v_u32m2(ui, vl), 1, vl), x, vl);
        else if constexpr (sizeof(T) == 4) __riscv_vsoxei32(base, __riscv_vsll(__riscv_vle32_v_u32m1(ui, vl), 2, vl), x, vl);
        else __riscv_vsoxei32(base, __riscv_vsll(__riscv_vle32_v_u32mf2(ui, vl), 3, vl), x, vl);
        return;
    }
#endif
#endif
    T t[N]; vt::to_arr(v.r, t);
    for (int i = 0; i < N; ++i) base[idx[i]] = t[i];
}

// ---------------------------------------------------------------- element-wise operations
// G: the GNU-vector expression over registers x, y (R = register type, AR = the unsigned / floating register the
// arithmetic is done in, MR = the mask register); S: the scalar expression over lanes x, y.
#define FORGE_VT_BIN(CONC, name, G, S)                                                                        \
    template <CONC T, Target Tg> inline vec<T, Tg> name(vec<T, Tg> a, vec<T, Tg> b) {                         \
        vec<T, Tg> o;                                                                                         \
        FORGE_VT_IFG(using R [[maybe_unused]] = typename vec<T, Tg>::native_t;                                \
                     using AR [[maybe_unused]] = vt::reg<vt::arith<T>, vec<T, Tg>::bytes, false>;             \
                     const R x = a.r, y = b.r; o.r = (R)(G);)                                                 \
        FORGE_VT_LANES { const T x = a.r.v[i], y = b.r.v[i]; o.r.v[i] = T(S); }                               \
        return o;                                                                                             \
    }
FORGE_VT_BIN(vt::elem, add, (AR)x + (AR)y, vt::wadd(x, y))
FORGE_VT_BIN(vt::elem, sub, (AR)x - (AR)y, vt::wsub(x, y))
FORGE_VT_BIN(vt::elem, mul, (AR)x * (AR)y, vt::wmul(x, y))
FORGE_VT_BIN(std::floating_point, div, x / y, x / y)
FORGE_VT_BIN(vt::elem, min, x < y ? x : y, x < y ? x : y)
FORGE_VT_BIN(vt::elem, max, x > y ? x : y, x > y ? x : y)
FORGE_VT_BIN(vt::ielem, bit_and, x & y, x & y)
FORGE_VT_BIN(vt::ielem, bit_or, x | y, x | y)
FORGE_VT_BIN(vt::ielem, bit_xor, x ^ y, x ^ y)
FORGE_VT_BIN(vt::ielem, bit_andnot, x & ~y, x & ~y)          // a & ~b
FORGE_VT_BIN(vt::ielem, shl, (AR)x << (AR)y, vt::shl1(x, int(y)))   // per-lane counts
FORGE_VT_BIN(vt::ielem, shr, x >> y, vt::shr1(x, int(y)))
#undef FORGE_VT_BIN

FORGE_VT_T inline vec<T, Tg> operator+(vec<T, Tg> a, vec<T, Tg> b) { return add(a, b); }
FORGE_VT_T inline vec<T, Tg> operator-(vec<T, Tg> a, vec<T, Tg> b) { return sub(a, b); }
FORGE_VT_T inline vec<T, Tg> operator*(vec<T, Tg> a, vec<T, Tg> b) { return mul(a, b); }
template <std::floating_point T, Target Tg> requires vt::elem<T> inline vec<T, Tg> operator/(vec<T, Tg> a, vec<T, Tg> b) { return div(a, b); }
FORGE_VT_T inline vec<T, Tg> operator-(vec<T, Tg> a) { return sub(vec<T, Tg>::splat(T(-0.0)), a); }   // integers: 0 - x; double: -0.0 - x, which flips the sign of zeros too
FORGE_VT_TI inline vec<T, Tg> operator&(vec<T, Tg> a, vec<T, Tg> b) { return bit_and(a, b); }
FORGE_VT_TI inline vec<T, Tg> operator|(vec<T, Tg> a, vec<T, Tg> b) { return bit_or(a, b); }
FORGE_VT_TI inline vec<T, Tg> operator^(vec<T, Tg> a, vec<T, Tg> b) { return bit_xor(a, b); }
FORGE_VT_TI inline vec<T, Tg> operator~(vec<T, Tg> a) { return bit_xor(a, vec<T, Tg>::splat(T(~T(0)))); }
// shifts by one count for every lane, 0 <= n < bits of T
FORGE_VT_TI inline vec<T, Tg> shl(vec<T, Tg> a, int n) {
    vec<T, Tg> o;
    FORGE_VT_IFG(using AR = vt::reg<vt::mu<T>, vec<T, Tg>::bytes, false>; o.r = (typename vec<T, Tg>::native_t)((AR)a.r << n);)
    FORGE_VT_LANES o.r.v[i] = vt::shl1(a.r.v[i], n);
    return o;
}
FORGE_VT_TI inline vec<T, Tg> shr(vec<T, Tg> a, int n) {
    vec<T, Tg> o;
    FORGE_VT_IFG(o.r = a.r >> n;)
    FORGE_VT_LANES o.r.v[i] = vt::shr1(a.r.v[i], n);
    return o;
}
FORGE_VT_TI inline vec<T, Tg> operator<<(vec<T, Tg> a, int n) { return shl(a, n); }
FORGE_VT_TI inline vec<T, Tg> operator>>(vec<T, Tg> a, int n) { return shr(a, n); }

// compares
#define FORGE_VT_CMP(name, OP)                                                                                \
    FORGE_VT_T inline mask<T, Tg> name(vec<T, Tg> a, vec<T, Tg> b) {                                          \
        mask<T, Tg> o;                                                                                        \
        FORGE_VT_IFG(o.m = (typename mask<T, Tg>::native_t)(a.r OP b.r);)                                     \
        FORGE_VT_LANES o.m.v[i] = vt::mi<T>(-vt::mi<T>(a.r.v[i] OP b.r.v[i]));                                \
        return o;                                                                                             \
    }
FORGE_VT_CMP(eq, ==) FORGE_VT_CMP(ne, !=) FORGE_VT_CMP(lt, <) FORGE_VT_CMP(le, <=) FORGE_VT_CMP(gt, >) FORGE_VT_CMP(ge, >=)
#undef FORGE_VT_CMP
#define FORGE_VT_MOP(OP)                                                                                      \
    FORGE_VT_T inline mask<T, Tg> operator OP(mask<T, Tg> a, mask<T, Tg> b) {                                 \
        mask<T, Tg> o;                                                                                        \
        FORGE_VT_IFG(o.m = a.m OP b.m;)                                                                       \
        FORGE_VT_LANES o.m.v[i] = vt::mi<T>(a.m.v[i] OP b.m.v[i]);                                            \
        return o;                                                                                             \
    }
FORGE_VT_MOP(&) FORGE_VT_MOP(|) FORGE_VT_MOP(^)
#undef FORGE_VT_MOP
FORGE_VT_T inline mask<T, Tg> operator!(mask<T, Tg> a) { return a ^ mask<T, Tg>::all(); }
// select(m, a, b): m ? a : b
FORGE_VT_T inline vec<T, Tg> select(mask<T, Tg> m, vec<T, Tg> a, vec<T, Tg> b) {
    vec<T, Tg> o;
    FORGE_VT_IFG(o.r = m.m ? a.r : b.r;)
    FORGE_VT_LANES o.r.v[i] = m.m.v[i] ? a.r.v[i] : b.r.v[i];
    return o;
}

FORGE_VT_T inline vec<T, Tg> abs(vec<T, Tg> a) {
    vec<T, Tg> o;
#if FORGE_VT_GNU
    if constexpr (!vec<T, Tg>::E) {
        using R = typename vec<T, Tg>::native_t;
        if constexpr (std::unsigned_integral<T>) o.r = a.r;
        else if constexpr (std::floating_point<T>) {
#if defined(__mips_msa)   // the and-mask becomes a vector fabs that zig 0.17's clang mis-lowers in front of fsqrt.d: clear bit 63 explicitly
            if constexpr (vec<T, Tg>::bytes == 16) { o.r = (R)__msa_bclri_d((v2u64)a.r, 63); return o; }
#endif
            using UR = vt::reg<std::uint64_t, vec<T, Tg>::bytes, false>;
            o.r = (R)((UR)a.r & std::uint64_t(0x7fffffffffffffffull));
        } else {
            using UR = vt::reg<vt::mu<T>, vec<T, Tg>::bytes, false>;
            const R z = R{};
            o.r = a.r < z ? (R)((UR)z - (UR)a.r) : a.r;
        }
        return o;
    } else
#endif
    FORGE_VT_LANES o.r.v[i] = vt::sabs(a.r.v[i]);
    return o;
}
FORGE_VT_TI inline vec<T, Tg> add_sat(vec<T, Tg> a, vec<T, Tg> b) {
    vec<T, Tg> o;
#if FORGE_VT_GNU
    if constexpr (!vec<T, Tg>::E) {
#if FORGE_VT_HAS(__builtin_elementwise_add_sat)
        o.r = __builtin_elementwise_add_sat(a.r, b.r);
#else
        using R = typename vec<T, Tg>::native_t; using UR = vt::reg<vt::mu<T>, vec<T, Tg>::bytes, false>; using SR = vt::reg<vt::mi<T>, vec<T, Tg>::bytes, false>;
        const UR ux = (UR)a.r, uy = (UR)b.r, us = ux + uy;
        if constexpr (std::unsigned_integral<T>) o.r = (R)(us | (UR)(us < ux));
        else {
            const SR ovf = (SR)((ux ^ us) & (uy ^ us)) < SR{};
            const SR sat = (SR)((UR)((SR)ux >> int(sizeof(T) * 8 - 1)) ^ vt::mu<T>(std::numeric_limits<T>::max()));
            o.r = (R)(ovf ? sat : (SR)us);
        }
#endif
        return o;
    } else
#endif
    FORGE_VT_LANES o.r.v[i] = vt::sat_add(a.r.v[i], b.r.v[i]);
    return o;
}
FORGE_VT_TI inline vec<T, Tg> sub_sat(vec<T, Tg> a, vec<T, Tg> b) {
    vec<T, Tg> o;
#if FORGE_VT_GNU
    if constexpr (!vec<T, Tg>::E) {
#if FORGE_VT_HAS(__builtin_elementwise_sub_sat)
        o.r = __builtin_elementwise_sub_sat(a.r, b.r);
#else
        using R = typename vec<T, Tg>::native_t; using UR = vt::reg<vt::mu<T>, vec<T, Tg>::bytes, false>; using SR = vt::reg<vt::mi<T>, vec<T, Tg>::bytes, false>;
        const UR ux = (UR)a.r, uy = (UR)b.r, ud = ux - uy;
        if constexpr (std::unsigned_integral<T>) o.r = (R)(ud & (UR)(ux >= uy));
        else {
            const SR ovf = (SR)((ux ^ uy) & (ux ^ ud)) < SR{};
            const SR sat = (SR)((UR)((SR)ux >> int(sizeof(T) * 8 - 1)) ^ vt::mu<T>(std::numeric_limits<T>::max()));
            o.r = (R)(ovf ? sat : (SR)ud);
        }
#endif
        return o;
    } else
#endif
    FORGE_VT_LANES o.r.v[i] = vt::sat_sub(a.r.v[i], b.r.v[i]);
    return o;
}
// population count of every lane (result in the same type)
FORGE_VT_TI inline vec<T, Tg> popcount(vec<T, Tg> a) {
    vec<T, Tg> o;
#if FORGE_VT_GNU
    if constexpr (!vec<T, Tg>::E) {
        using R = typename vec<T, Tg>::native_t; using UR = vt::reg<vt::mu<T>, vec<T, Tg>::bytes, false>;
#if FORGE_VT_HAS(__builtin_elementwise_popcount)
        o.r = (R)__builtin_elementwise_popcount((UR)a.r);
#else
        using U = vt::mu<T>;
        const U k = U(~U(0));
        UR v = (UR)a.r;
        v = v - ((v >> 1) & U(k / 3));
        v = (v & U(k / 5)) + ((v >> 2) & U(k / 5));
        v = (v + (v >> 4)) & U(k / 17);
        o.r = (R)((v * U(k / 255)) >> int(sizeof(T) * 8 - 8));
#endif
        return o;
    } else
#endif
    FORGE_VT_LANES o.r.v[i] = T(std::popcount(vt::mu<T>(a.r.v[i])));
    return o;
}
template <Target Tg> inline vec<double, Tg> sqrt(vec<double, Tg> a) {
    vec<double, Tg> o;
#if FORGE_VT_GNU && defined(__mips_msa)   // the intrinsic: see fma below
    if constexpr (!vec<double, Tg>::E && vec<double, Tg>::bytes == 16) { o.r = (typename vec<double, Tg>::native_t)__msa_fsqrt_d((v2f64)a.r); return o; }
#elif FORGE_VT_GNU && FORGE_VT_HAS(__builtin_elementwise_sqrt)
    if constexpr (!vec<double, Tg>::E) { o.r = __builtin_elementwise_sqrt(a.r); return o; }
#endif
    double t[vec<double, Tg>::N]; vt::to_arr(a.r, t);
    for (auto& x : t) x = std::sqrt(x);
    vt::from_arr(o.r, t);
    return o;
}
// a * b + c with one rounding
template <Target Tg> inline vec<double, Tg> fma(vec<double, Tg> a, vec<double, Tg> b, vec<double, Tg> c) {
    vec<double, Tg> o;
#if FORGE_VT_GNU && defined(__mips_msa)
    // zig 0.17's clang lowers the generic vector fma to fmadd.d with the operands rotated (a + b * c): use the intrinsic
    if constexpr (!vec<double, Tg>::E && vec<double, Tg>::bytes == 16) {
        o.r = (typename vec<double, Tg>::native_t)__msa_fmadd_d((v2f64)c.r, (v2f64)a.r, (v2f64)b.r); return o;
    }
#elif FORGE_VT_GNU && FORGE_VT_HAS(__builtin_elementwise_fma)
    if constexpr (!vec<double, Tg>::E) { o.r = __builtin_elementwise_fma(a.r, b.r, c.r); return o; }
#endif
    constexpr int N = vec<double, Tg>::N;
    double x[N], y[N], z[N]; vt::to_arr(a.r, x); vt::to_arr(b.r, y); vt::to_arr(c.r, z);
    for (int i = 0; i < N; ++i) z[i] = std::fma(x[i], y[i], z[i]);
    vt::from_arr(o.r, z);
    return o;
}

// ---------------------------------------------------------------- reductions
namespace vt {
template <class T, int N, class F> FORGE_VT_INL T fold(T* t, F&& f) {   // (t[i], t[i + w]) for w = N/2 .. 1
    for (int w = N / 2; w >= 1; w /= 2) for (int i = 0; i < w; ++i) t[i] = f(t[i], t[i + w]);
    return t[0];
}
} // namespace vt
#define FORGE_VT_RED(CONC, name, S)                                                                           \
    template <CONC T, Target Tg> inline T name(vec<T, Tg> v) {                                                \
        T t[vec<T, Tg>::N]; vt::to_arr(v.r, t);                                                               \
        return vt::fold<T, vec<T, Tg>::N>(t, [](T x, T y) { return T(S); });                                  \
    }
FORGE_VT_RED(vt::elem, reduce_add, vt::wadd(x, y))
FORGE_VT_RED(vt::elem, reduce_min, y < x ? y : x)
FORGE_VT_RED(vt::elem, reduce_max, y > x ? y : x)
FORGE_VT_RED(vt::ielem, reduce_and, x & y)
FORGE_VT_RED(vt::ielem, reduce_or, x | y)
FORGE_VT_RED(vt::ielem, reduce_xor, x ^ y)
#undef FORGE_VT_RED

// ---------------------------------------------------------------- conversions
// Value conversion between types of the same width (int32 <-> uint32, int64 <-> double, ...).
template <vt::elem U, vt::elem T, Target Tg> requires (sizeof(U) == sizeof(T)) inline vec<U, Tg> convert(vec<T, Tg> a) {
    vec<U, Tg> o;
#if FORGE_VT_GNU
    if constexpr (!vec<T, Tg>::E) { o.r = __builtin_convertvector(a.r, typename vec<U, Tg>::native_t); return o; } else
#endif
    FORGE_VT_LANES o.r.v[i] = U(a.r.v[i]);
    return o;
}
// The same bits as another element type (any vt::elem; lanes in memory order).
template <vt::elem U, vt::elem T, Target Tg> inline vec<U, Tg> bitcast(vec<T, Tg> a) { vec<U, Tg> o; std::memcpy(&o.r, &a.r, sizeof(o.r)); return o; }
// vec<float> <-> integer vectors of the same register
template <vt::elem U, Target Tg> requires (sizeof(U) == 4) inline vec<U, Tg> bitcast(vec<float, Tg> a) {
    static_assert(sizeof(a.r) == sizeof(typename vec<U, Tg>::native_t), "vec<float, Tg> and vec<U, Tg> must be the same register here");
    vec<U, Tg> o; std::memcpy(&o.r, &a.r, sizeof(o.r)); return o;
}
template <vt::elem T, Target Tg> requires (sizeof(T) == 4) inline vec<float, Tg> bitcast_f32(vec<T, Tg> a) {
    static_assert(sizeof(a.r) == sizeof(typename vec<float, Tg>::native_t), "vec<float, Tg> and vec<T, Tg> must be the same register here");
    vec<float, Tg> o; std::memcpy(&o.r, &a.r, sizeof(o.r)); return o;
}
// float -> int32 (truncating; out of range is a precondition violation) and int32 -> float (nearest even)
template <Target Tg> inline vec<std::int32_t, Tg> to_int(vec<float, Tg> a) {
    constexpr int N = vec<std::int32_t, Tg>::N;
    static_assert(N == vec<float, Tg>::N);
    vec<std::int32_t, Tg> o;
#if FORGE_VT_GNU
    if constexpr (!vec<std::int32_t, Tg>::E) {
        vt::reg<float, vec<std::int32_t, Tg>::bytes, false> f; std::memcpy(&f, &a.r, sizeof f);
        o.r = __builtin_convertvector(f, typename vec<std::int32_t, Tg>::native_t);
        return o;
    } else
#endif
    {
        float t[N]; std::memcpy(t, &a.r, sizeof t);
        for (int i = 0; i < N; ++i) o.r.v[i] = std::int32_t(t[i]);
    }
    return o;
}
template <Target Tg> inline vec<float, Tg> to_float(vec<std::int32_t, Tg> a) {
    constexpr int N = vec<std::int32_t, Tg>::N;
    static_assert(N == vec<float, Tg>::N);
    vec<float, Tg> o;
#if FORGE_VT_GNU
    if constexpr (!vec<std::int32_t, Tg>::E) {
        const auto f = __builtin_convertvector(a.r, vt::reg<float, vec<std::int32_t, Tg>::bytes, false>);
        std::memcpy(&o.r, &f, sizeof f);
        return o;
    } else
#endif
    {
        float t[N];
        for (int i = 0; i < N; ++i) t[i] = float(a.r.v[i]);
        std::memcpy(&o.r, t, sizeof t);
    }
    return o;
}
// Low / high half of the lanes, value-converted to the type twice as wide (int8 -> int16, ..., int32 -> int64).
namespace vt {
template <class W, class T, Target Tg> inline vec<W, Tg> widen_half(const T* half) {   // N/2 values of T
    constexpr int H = vec<T, Tg>::N / 2;
    vec<W, Tg> o;
#if FORGE_VT_GNU
    if constexpr (!vec<T, Tg>::E) {
        reg<T, vec<T, Tg>::bytes / 2, false> h; std::memcpy(&h, half, sizeof h);
        o.r = __builtin_convertvector(h, typename vec<W, Tg>::native_t);
        return o;
    } else
#endif
    for (int i = 0; i < H; ++i) o.r.v[i] = W(half[i]);
    return o;
}
} // namespace vt
template <vt::ielem T, Target Tg> requires (sizeof(T) < 8) inline vec<vt::wider<T>, Tg> widen_lo(vec<T, Tg> a) {
    T t[vec<T, Tg>::N]; vt::to_arr(a.r, t); return vt::widen_half<vt::wider<T>, T, Tg>(t);
}
template <vt::ielem T, Target Tg> requires (sizeof(T) < 8) inline vec<vt::wider<T>, Tg> widen_hi(vec<T, Tg> a) {
    T t[vec<T, Tg>::N]; vt::to_arr(a.r, t); return vt::widen_half<vt::wider<T>, T, Tg>(t + vec<T, Tg>::N / 2);
}
// float -> double of the low / high half
template <Target Tg> inline vec<double, Tg> promote_lo(vec<float, Tg> a) {
    constexpr int N = vec<float, Tg>::N; static_assert(N == 2 * vec<double, Tg>::N);
    float t[N]; std::memcpy(t, &a.r, sizeof t);
    double d[N / 2]; for (int i = 0; i < N / 2; ++i) d[i] = double(t[i]);
    return vec<double, Tg>::load(d);
}
template <Target Tg> inline vec<double, Tg> promote_hi(vec<float, Tg> a) {
    constexpr int N = vec<float, Tg>::N; static_assert(N == 2 * vec<double, Tg>::N);
    float t[N]; std::memcpy(t, &a.r, sizeof t);
    double d[N / 2]; for (int i = 0; i < N / 2; ++i) d[i] = double(t[N / 2 + i]);
    return vec<double, Tg>::load(d);
}
// two double vectors -> one float vector (nearest even)
template <Target Tg> inline vec<float, Tg> demote(vec<double, Tg> lo, vec<double, Tg> hi) {
    constexpr int H = vec<double, Tg>::N; static_assert(2 * H == vec<float, Tg>::N);
    double d[2 * H]; vt::to_arr(lo.r, d); vt::to_arr(hi.r, d + H);
    float t[2 * H]; for (int i = 0; i < 2 * H; ++i) t[i] = float(d[i]);
    vec<float, Tg> o; std::memcpy(&o.r, t, sizeof t); return o;
}
// Two vectors -> one of the half-width type (lo's lanes first). narrow truncates; narrow_sat<U> clamps to the
// range of U (the half-width type of either signedness for signed T; the unsigned one for unsigned T).
template <vt::ielem T, Target Tg> requires (sizeof(T) > 1) inline vec<vt::narrower<T>, Tg> narrow(vec<T, Tg> lo, vec<T, Tg> hi) {
    using U = vt::narrower<T>;
    constexpr int N = vec<T, Tg>::N;
    vec<U, Tg> o;
#if FORGE_VT_GNU
    if constexpr (!vec<T, Tg>::E) {
        using HR = vt::reg<U, vec<T, Tg>::bytes / 2, false>;
        const HR a = __builtin_convertvector(lo.r, HR), b = __builtin_convertvector(hi.r, HR);
        std::memcpy(&o.r, &a, sizeof a); std::memcpy(reinterpret_cast<char*>(&o.r) + sizeof a, &b, sizeof b);
        return o;
    } else
#endif
    for (int i = 0; i < N; ++i) { o.r.v[i] = U(lo.r.v[i]); o.r.v[N + i] = U(hi.r.v[i]); }
    return o;
}
template <vt::ielem U, vt::ielem T, Target Tg> requires (sizeof(T) == 2 * sizeof(U) && (std::signed_integral<T> || std::unsigned_integral<U>))
inline vec<U, Tg> narrow_sat(vec<T, Tg> lo, vec<T, Tg> hi) {
    constexpr int N = vec<T, Tg>::N;
    vec<U, Tg> o;
#if FORGE_VT_GNU
    if constexpr (!vec<T, Tg>::E) {
        using R = typename vec<T, Tg>::native_t;
        using HR = vt::reg<U, vec<T, Tg>::bytes / 2, false>;
        const T hiv = T(std::numeric_limits<U>::max());
        R x = lo.r, y = hi.r;
        const R top = R{} + hiv;
        x = x > top ? top : x; y = y > top ? top : y;
        if constexpr (std::signed_integral<T>) {
            const R bot = R{} + T(std::numeric_limits<U>::min());
            x = x < bot ? bot : x; y = y < bot ? bot : y;
        }
        const HR a = __builtin_convertvector(x, HR), b = __builtin_convertvector(y, HR);
        std::memcpy(&o.r, &a, sizeof a); std::memcpy(reinterpret_cast<char*>(&o.r) + sizeof a, &b, sizeof b);
        return o;
    } else
#endif
    for (int i = 0; i < N; ++i) { o.r.v[i] = vt::clampto<T, U>(lo.r.v[i]); o.r.v[N + i] = vt::clampto<T, U>(hi.r.v[i]); }
    return o;
}

// ---------------------------------------------------------------- half precision and bfloat16
// N = vec<float, Tg>::N values of IEEE binary16 (as uint16 bit patterns) <-> vec<float, Tg>. Native where the
// translation unit has the conversion (x86 F16C / AVX-512, AArch64); elsewhere forge::f16 per lane. Storing
// rounds to nearest even and overflows to infinity.
template <Target Tg> inline vec<float, Tg> load_f16(const std::uint16_t* p) {
    constexpr int N = vec<float, Tg>::N;
    vec<float, Tg> o;
#if defined(__AVX512F__)
    if constexpr (vec<float, Tg>::K == 2) { o.r = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))); return o; }
#endif
#if defined(__AVX2__) && defined(__F16C__)
    if constexpr (vec<float, Tg>::K == 1) { o.r = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))); return o; }
#endif
#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(_MSC_VER)
    if constexpr (vec<float, Tg>::K == 3) { o.r = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p))); return o; }
#endif
    float t[N];
    for (int i = 0; i < N; ++i) t[i] = float(forge::f16::from_bits(p[i]));
    return forge::simd::load<Tg>(t);
}
namespace vt {
// One float -> binary16 bits. Kept out of line on LoongArch: zig 0.17's clang miscompiles the conversion loop
// when it vectorises it with LASX (odd lanes lose the sign and the infinities).
#if defined(__loongarch__) && (defined(__GNUC__) || defined(__clang__))
template <Target Tg> __attribute__((noinline)) std::uint16_t f16_bits(float x) { return forge::f16(x).bits; }
#else
template <Target Tg> inline std::uint16_t f16_bits(float x) { return forge::f16(x).bits; }
#endif
} // namespace vt
template <Target Tg> inline void store_f16(std::uint16_t* p, vec<float, Tg> v) {
    constexpr int N = vec<float, Tg>::N;
#if defined(__AVX512F__)
    if constexpr (vec<float, Tg>::K == 2) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), _mm512_cvtps_ph(v.r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); return; }
#endif
#if defined(__AVX2__) && defined(__F16C__)
    if constexpr (vec<float, Tg>::K == 1) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_cvtps_ph(v.r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); return; }
#endif
#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(_MSC_VER)
    if constexpr (vec<float, Tg>::K == 3) { vst1_u16(p, vreinterpret_u16_f16(vcvt_f16_f32(v.r))); return; }
#endif
    float t[N]; forge::simd::store(t, v);
    for (int i = 0; i < N; ++i) p[i] = vt::f16_bits<Tg>(t[i]);
}
// bfloat16 is the upper half of a binary32: integer shifts on every ISA (the forge::bf16 rounding: nearest
// even, NaN keeps its upper payload with the quiet bit set).
template <Target Tg> inline vec<float, Tg> load_bf16(const std::uint16_t* p) {
    constexpr int N = vec<float, Tg>::N;
    std::uint32_t t[N];
    for (int i = 0; i < N; ++i) t[i] = std::uint32_t(p[i]) << 16;
    vec<float, Tg> o; std::memcpy(&o.r, t, sizeof t); return o;
}
template <Target Tg> inline void store_bf16(std::uint16_t* p, vec<float, Tg> v) {
    constexpr int N = vec<float, Tg>::N;
    std::uint32_t t[N]; std::memcpy(t, &v.r, sizeof t);
    for (int i = 0; i < N; ++i) {
        std::uint32_t x = t[i];
        const bool nan = (x & 0x7fffffffu) > 0x7f800000u;
        const std::uint32_t r = x + 0x7fffu + ((x >> 16) & 1u);
        p[i] = nan ? std::uint16_t((x >> 16) | 0x40) : std::uint16_t(r >> 16);
    }
}

#undef FORGE_VT_T
#undef FORGE_VT_INL
#undef FORGE_VT_TI
#undef FORGE_VT_IFG
#undef FORGE_VT_LANES

} // namespace forge::simd
#if FORGE_VT_GNU
#pragma GCC diagnostic pop
#endif
