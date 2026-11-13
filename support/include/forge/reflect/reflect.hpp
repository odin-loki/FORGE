#pragma once
// reflect: field-list reflection for serialisation (save games, config, messages, property panels).
//   struct cfg { int w; std::string name; std::vector<float> v; std::optional<inner> in; mode m; };
//   FORGE_REFLECT(cfg, w, name, v, in, m)            up to 64 fields, at namespace scope next to the type
//   FORGE_REFLECT_ENUM(mode, fast, exact)            optional: enumerators by name
//   json::value j = reflect::to_json(c);             cfg c2 = reflect::from_json<cfg>(j);
//   auto bytes = reflect::to_binary(c);              cfg c3 = reflect::from_binary<cfg>(bytes);
//   reflect::schema_hash<cfg>()                      changes when a field or a type in the layout changes
//
// Supported member types: bool, integers, float/double, enums, std::string, std::vector, std::array,
// std::optional, std::variant, std::map / std::unordered_map with std::string keys, and nested reflected
// structs.
//
// JSON: structs are objects; missing keys keep the member's default and unknown keys are ignored. Enums with
// FORGE_REFLECT_ENUM are written by name (a value with no name is written as a number; both are read).
// optional: null or the value. variant: {"index": n, "value": v}. Numbers pass through double, so 64-bit
// integers beyond 2^53 lose precision in JSON (not in the binary form).
//
// Binary ("FRB1"): magic, the writer's 64-bit schema hash, then the value. A struct is a field count followed
// by (32-bit FNV-1a of the field name, byte length, payload) records, so a reader skips fields it does not
// know and leaves members the writer did not have at their defaults; a field whose type changed is an error
// if its payload does not parse exactly. Integers are zig-zag varints, floats little-endian IEEE, strings
// and containers length-prefixed. from_binary bounds-checks everything and raises forge_exception
// (errc::parse_error) on malformed input; binary_info reports whether the writer's schema matched.
// Two fields of one struct whose names share a hash are not supported (the first one wins).
//
// C++26 static reflection: with FORGE_REFLECT_STATIC defined on a compiler that reports
// __cpp_impl_reflection, aggregates need no FORGE_REFLECT (members are enumerated with std::meta). That
// branch has NEVER BEEN COMPILED: no compiler available to this project accepts it yet (clang 23 has no
// <meta>). Treat it as a sketch to be fixed when one does.
#include <forge/json/json.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>
#if defined(__cpp_impl_reflection) && defined(FORGE_REFLECT_STATIC)
#include <meta>
#define FORGE_REFLECT_HAVE_STATIC 1
#endif

// FORGE_RF_k(M, T, a, b, ...) expands to M(T, a), M(T, b), ...; written out so it also works with MSVC's
// traditional preprocessor (no __VA_OPT__).
#define FORGE_RF_EXPAND(x) x
#define FORGE_RF_1(M, T, a) M(T, a)
#define FORGE_RF_2(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_1(M, T, __VA_ARGS__))
#define FORGE_RF_3(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_2(M, T, __VA_ARGS__))
#define FORGE_RF_4(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_3(M, T, __VA_ARGS__))
#define FORGE_RF_5(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_4(M, T, __VA_ARGS__))
#define FORGE_RF_6(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_5(M, T, __VA_ARGS__))
#define FORGE_RF_7(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_6(M, T, __VA_ARGS__))
#define FORGE_RF_8(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_7(M, T, __VA_ARGS__))
#define FORGE_RF_9(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_8(M, T, __VA_ARGS__))
#define FORGE_RF_10(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_9(M, T, __VA_ARGS__))
#define FORGE_RF_11(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_10(M, T, __VA_ARGS__))
#define FORGE_RF_12(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_11(M, T, __VA_ARGS__))
#define FORGE_RF_13(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_12(M, T, __VA_ARGS__))
#define FORGE_RF_14(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_13(M, T, __VA_ARGS__))
#define FORGE_RF_15(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_14(M, T, __VA_ARGS__))
#define FORGE_RF_16(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_15(M, T, __VA_ARGS__))
#define FORGE_RF_17(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_16(M, T, __VA_ARGS__))
#define FORGE_RF_18(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_17(M, T, __VA_ARGS__))
#define FORGE_RF_19(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_18(M, T, __VA_ARGS__))
#define FORGE_RF_20(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_19(M, T, __VA_ARGS__))
#define FORGE_RF_21(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_20(M, T, __VA_ARGS__))
#define FORGE_RF_22(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_21(M, T, __VA_ARGS__))
#define FORGE_RF_23(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_22(M, T, __VA_ARGS__))
#define FORGE_RF_24(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_23(M, T, __VA_ARGS__))
#define FORGE_RF_25(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_24(M, T, __VA_ARGS__))
#define FORGE_RF_26(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_25(M, T, __VA_ARGS__))
#define FORGE_RF_27(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_26(M, T, __VA_ARGS__))
#define FORGE_RF_28(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_27(M, T, __VA_ARGS__))
#define FORGE_RF_29(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_28(M, T, __VA_ARGS__))
#define FORGE_RF_30(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_29(M, T, __VA_ARGS__))
#define FORGE_RF_31(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_30(M, T, __VA_ARGS__))
#define FORGE_RF_32(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_31(M, T, __VA_ARGS__))
#define FORGE_RF_33(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_32(M, T, __VA_ARGS__))
#define FORGE_RF_34(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_33(M, T, __VA_ARGS__))
#define FORGE_RF_35(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_34(M, T, __VA_ARGS__))
#define FORGE_RF_36(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_35(M, T, __VA_ARGS__))
#define FORGE_RF_37(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_36(M, T, __VA_ARGS__))
#define FORGE_RF_38(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_37(M, T, __VA_ARGS__))
#define FORGE_RF_39(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_38(M, T, __VA_ARGS__))
#define FORGE_RF_40(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_39(M, T, __VA_ARGS__))
#define FORGE_RF_41(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_40(M, T, __VA_ARGS__))
#define FORGE_RF_42(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_41(M, T, __VA_ARGS__))
#define FORGE_RF_43(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_42(M, T, __VA_ARGS__))
#define FORGE_RF_44(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_43(M, T, __VA_ARGS__))
#define FORGE_RF_45(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_44(M, T, __VA_ARGS__))
#define FORGE_RF_46(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_45(M, T, __VA_ARGS__))
#define FORGE_RF_47(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_46(M, T, __VA_ARGS__))
#define FORGE_RF_48(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_47(M, T, __VA_ARGS__))
#define FORGE_RF_49(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_48(M, T, __VA_ARGS__))
#define FORGE_RF_50(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_49(M, T, __VA_ARGS__))
#define FORGE_RF_51(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_50(M, T, __VA_ARGS__))
#define FORGE_RF_52(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_51(M, T, __VA_ARGS__))
#define FORGE_RF_53(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_52(M, T, __VA_ARGS__))
#define FORGE_RF_54(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_53(M, T, __VA_ARGS__))
#define FORGE_RF_55(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_54(M, T, __VA_ARGS__))
#define FORGE_RF_56(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_55(M, T, __VA_ARGS__))
#define FORGE_RF_57(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_56(M, T, __VA_ARGS__))
#define FORGE_RF_58(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_57(M, T, __VA_ARGS__))
#define FORGE_RF_59(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_58(M, T, __VA_ARGS__))
#define FORGE_RF_60(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_59(M, T, __VA_ARGS__))
#define FORGE_RF_61(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_60(M, T, __VA_ARGS__))
#define FORGE_RF_62(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_61(M, T, __VA_ARGS__))
#define FORGE_RF_63(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_62(M, T, __VA_ARGS__))
#define FORGE_RF_64(M, T, a, ...) M(T, a), FORGE_RF_EXPAND(FORGE_RF_63(M, T, __VA_ARGS__))
#define FORGE_RF_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, N, ...) N
#define FORGE_RF_COUNT(...) FORGE_RF_EXPAND(FORGE_RF_N(__VA_ARGS__, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
#define FORGE_RF_CAT2(a, b) a##b
#define FORGE_RF_CAT(a, b) FORGE_RF_CAT2(a, b)
#define FORGE_RF_APPLY(M, T, ...) FORGE_RF_EXPAND(FORGE_RF_CAT(FORGE_RF_, FORGE_RF_COUNT(__VA_ARGS__))(M, T, __VA_ARGS__))
#define FORGE_RF_F(T, f) std::pair<const char*, decltype(&T::f)>{#f, &T::f}
#define FORGE_RF_E(T, e) std::pair<const char*, T>{#e, T::e}
#define FORGE_REFLECT(T, ...) \
    inline constexpr auto forge_reflect_fields(const T*) { return std::make_tuple(FORGE_RF_APPLY(FORGE_RF_F, T, __VA_ARGS__)); }
#define FORGE_REFLECT_ENUM(T, ...) \
    inline constexpr auto forge_reflect_enum(const T*) { return std::array{FORGE_RF_APPLY(FORGE_RF_E, T, __VA_ARGS__)}; }

namespace forge::reflect {
template <class T> concept macro_reflected = requires(const T* p) { forge_reflect_fields(p); };
template <class T> concept named_enum = std::is_enum_v<T> && requires(const T* p) { forge_reflect_enum(p); };
template <class T> struct is_vec : std::false_type {};
template <class U, class A> struct is_vec<std::vector<U, A>> : std::true_type {};
template <class T> struct is_arr : std::false_type {};
template <class U, std::size_t N> struct is_arr<std::array<U, N>> : std::true_type {};
template <class T> struct is_opt : std::false_type {};
template <class U> struct is_opt<std::optional<U>> : std::true_type {};
template <class T> struct is_var : std::false_type {};
template <class... U> struct is_var<std::variant<U...>> : std::true_type {};
template <class T> struct is_smap : std::false_type {};
template <class U, class C, class A> struct is_smap<std::map<std::string, U, C, A>> : std::true_type {};
template <class U, class H, class E, class A> struct is_smap<std::unordered_map<std::string, U, H, E, A>> : std::true_type {};
template <class T> inline constexpr bool is_std_shape = is_vec<T>::value || is_arr<T>::value || is_opt<T>::value || is_var<T>::value || is_smap<T>::value || std::is_same_v<T, std::string>;

#if defined(FORGE_REFLECT_HAVE_STATIC)
// ---- never compiled (see the header comment) ----
template <class T> concept reflected = macro_reflected<T> || (std::is_class_v<T> && std::is_aggregate_v<T> && !is_std_shape<T>);
namespace detail {
template <class T> consteval auto static_members() { return std::define_static_array(std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked())); }
}
template <class T, class F> constexpr void for_each_field(T& obj, F&& f) {
    using U = std::remove_const_t<T>;
    if constexpr (macro_reflected<U>) std::apply([&](auto... m) { (f(std::string_view(m.first), obj.*(m.second)), ...); }, forge_reflect_fields(static_cast<const U*>(nullptr)));
    else { template for (constexpr auto m : detail::static_members<U>()) f(std::meta::identifier_of(m), obj.[:m:]); }
}
#else
template <class T> concept reflected = macro_reflected<T>;
// Calls f(name, member) for every field of obj in declaration order.
template <class T, class F> constexpr void for_each_field(T& obj, F&& f) {
    std::apply([&](auto... m) { (f(std::string_view(m.first), obj.*(m.second)), ...); }, forge_reflect_fields(static_cast<const std::remove_const_t<T>*>(nullptr)));
}
#endif
template <class T> std::size_t field_count() { std::size_t n = 0; T tmp{}; for_each_field(tmp, [&](std::string_view, auto&) { ++n; }); return n; }

// Name of an enumerator registered with FORGE_REFLECT_ENUM; nullptr for a value without one.
template <named_enum E> const char* enum_name(E v) { for (auto& [n, e] : forge_reflect_enum(static_cast<const E*>(nullptr))) if (e == v) return n; return nullptr; }
template <named_enum E> std::optional<E> enum_from_name(std::string_view s) { for (auto& [n, e] : forge_reflect_enum(static_cast<const E*>(nullptr))) if (s == n) return e; return std::nullopt; }

// ------------------------------------------------- JSON -------------------------------------------------
namespace detail {
// A JSON number as the integer type I; anything that is not finite or does not fit is a parse error
// (converting an out-of-range double to an integer is undefined behaviour).
template <class I> I int_from(double d) {
    const double lo = std::is_signed_v<I> ? -std::ldexp(1.0, int(sizeof(I) * 8 - 1)) : 0.0, hi = std::ldexp(1.0, int(sizeof(I) * 8 - (std::is_signed_v<I> ? 1 : 0)));
    if (!(d >= lo && d < hi)) raise(errc::parse_error, "reflect: number {} does not fit a {}-bit integer field", d, sizeof(I) * 8);
    return I(d);
}
} // namespace detail
template <class T> json::value to_json(const T& v) {
    if constexpr (std::is_same_v<T, bool>) return json::value(v);
    else if constexpr (std::is_enum_v<T>) {
        if constexpr (named_enum<T>) if (const char* n = enum_name(v)) return json::value(std::string(n));
        return json::value(double(std::underlying_type_t<T>(v)));
    }
    else if constexpr (std::is_arithmetic_v<T>) return json::value(double(v));
    else if constexpr (std::is_same_v<T, std::string>) return json::value(v);
    else if constexpr (std::is_same_v<T, std::vector<bool>>) { json::array a; for (bool e : v) a.push_back(json::value(e)); return json::value(std::move(a)); }   // libc++: elements are __bit_const_reference
    else if constexpr (is_vec<T>::value || is_arr<T>::value) { json::array a; for (const auto& e : v) a.push_back(to_json(e)); return json::value(std::move(a)); }
    else if constexpr (is_opt<T>::value) return v ? to_json(*v) : json::value(nullptr);
    else if constexpr (is_var<T>::value) {
        json::object o; o.emplace_back("index", json::value(double(v.index())));
        std::visit([&](const auto& alt) { o.emplace_back("value", to_json(alt)); }, v);
        return json::value(std::move(o));
    }
    else if constexpr (is_smap<T>::value) { json::object o; for (auto& [k, e] : v) o.emplace_back(k, to_json(e)); return json::value(std::move(o)); }
    else {
        static_assert(reflected<T>, "reflect::to_json: type has no FORGE_REFLECT");
        json::object o;
        for_each_field(v, [&](std::string_view name, const auto& m) { o.emplace_back(std::string(name), to_json(m)); });
        return json::value(std::move(o));
    }
}
template <class T> T from_json(const json::value& j) {
    if constexpr (std::is_same_v<T, bool>) return j.b();
    else if constexpr (std::is_enum_v<T>) {
        if constexpr (named_enum<T>) if (j.is_str()) { if (auto e = enum_from_name<T>(j.str())) return *e; raise(errc::parse_error, "reflect: unknown enumerator '{}'", j.str()); }
        return T(detail::int_from<std::underlying_type_t<T>>(j.num()));
    }
    else if constexpr (std::is_integral_v<T>) return detail::int_from<T>(j.num());
    else if constexpr (std::is_arithmetic_v<T>) return T(j.num());
    else if constexpr (std::is_same_v<T, std::string>) return j.str();
    else if constexpr (is_vec<T>::value) { T out; for (auto& e : j.arr()) out.push_back(from_json<typename T::value_type>(e)); return out; }
    else if constexpr (is_arr<T>::value) {
        T out{}; if (j.arr().size() != out.size()) raise(errc::parse_error, "reflect: array of {} elements where {} are needed", j.arr().size(), out.size());
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = from_json<typename T::value_type>(j.arr()[i]);
        return out;
    }
    else if constexpr (is_opt<T>::value) { if (j.is_null()) return std::nullopt; return T(from_json<typename T::value_type>(j)); }
    else if constexpr (is_var<T>::value) {
        const std::size_t idx = detail::int_from<std::size_t>(j["index"].num());
        if (idx >= std::variant_size_v<T>) raise(errc::parse_error, "reflect: variant index {} out of range", idx);
        T out; const json::value& val = j["value"];
        [&]<std::size_t... I>(std::index_sequence<I...>) { ((idx == I ? void(out = T(std::in_place_index<I>, from_json<std::variant_alternative_t<I, T>>(val))) : void()), ...); }(std::make_index_sequence<std::variant_size_v<T>>{});
        return out;
    }
    else if constexpr (is_smap<T>::value) { T out; for (auto& [k, e] : j.obj()) out[k] = from_json<typename T::mapped_type>(e); return out; }
    else {
        static_assert(reflected<T>, "reflect::from_json: type has no FORGE_REFLECT");
        T out{};
        if (!j.is_obj()) raise(errc::parse_error, "reflect: expected an object");
        for_each_field(out, [&](std::string_view name, auto& m) {   // missing keys keep their defaults (forward-compatible save files)
            if (auto* p = j.find(name)) m = from_json<std::remove_cvref_t<decltype(m)>>(*p);
        });
        return out;
    }
}

// ------------------------------------------------- schema -------------------------------------------------
namespace detail {
constexpr std::uint32_t fnv1a32(std::string_view s) { std::uint32_t h = 2166136261u; for (char c : s) { h ^= std::uint8_t(c); h *= 16777619u; } return h; }
constexpr std::uint64_t fnv1a64(std::string_view s) { std::uint64_t h = 14695981039346656037ull; for (char c : s) { h ^= std::uint8_t(c); h *= 1099511628211ull; } return h; }
template <class T> void schema(std::string& s, int depth) {
    if (depth > 24) { s += "..."; return; }   // self-referential types (struct node { std::vector<node> kids; })
    if constexpr (std::is_same_v<T, bool>) s += "b";
    else if constexpr (std::is_enum_v<T>) {
        s += "e" + std::to_string(sizeof(T));
        if constexpr (named_enum<T>) { s += "("; for (auto& [n, e] : forge_reflect_enum(static_cast<const T*>(nullptr))) { s += n; s += "=" + std::to_string(static_cast<long long>(std::underlying_type_t<T>(e))) + ","; } s += ")"; }
    }
    else if constexpr (std::is_floating_point_v<T>) s += "f" + std::to_string(sizeof(T) * 8);
    else if constexpr (std::is_integral_v<T>) s += (std::is_signed_v<T> ? "i" : "u") + std::to_string(sizeof(T) * 8);
    else if constexpr (std::is_same_v<T, std::string>) s += "s";
    else if constexpr (is_vec<T>::value) { s += "["; schema<typename T::value_type>(s, depth + 1); s += "]"; }
    else if constexpr (is_arr<T>::value) { s += "[" + std::to_string(std::tuple_size_v<T>) + ":"; schema<typename T::value_type>(s, depth + 1); s += "]"; }
    else if constexpr (is_opt<T>::value) { s += "?"; schema<typename T::value_type>(s, depth + 1); }
    else if constexpr (is_var<T>::value) { s += "<"; [&]<std::size_t... I>(std::index_sequence<I...>) { ((schema<std::variant_alternative_t<I, T>>(s, depth + 1), s += "|"), ...); }(std::make_index_sequence<std::variant_size_v<T>>{}); s += ">"; }
    else if constexpr (is_smap<T>::value) { s += "{s:"; schema<typename T::mapped_type>(s, depth + 1); s += "}"; }
    else {
        static_assert(reflected<T>, "reflect: type has no FORGE_REFLECT");
        s += "{"; T tmp{};
        for_each_field(tmp, [&](std::string_view name, auto& m) { s += name; s += ":"; schema<std::remove_cvref_t<decltype(m)>>(s, depth + 1); s += ","; });
        s += "}";
    }
}
} // namespace detail
// A canonical description of T's serialised layout, e.g. "{w:i32,name:s,v:[f32],}", and its 64-bit FNV-1a hash.
template <class T> std::string schema_string() { std::string s; detail::schema<T>(s, 0); return s; }
template <class T> std::uint64_t schema_hash() { return detail::fnv1a64(schema_string<T>()); }

// ------------------------------------------------- binary -------------------------------------------------
namespace detail {
struct bin_writer {
    std::vector<std::uint8_t> o;
    void varint(std::uint64_t v) { while (v >= 0x80) { o.push_back(std::uint8_t(v) | 0x80); v >>= 7; } o.push_back(std::uint8_t(v)); }
    void raw(const void* p, std::size_t n) { const auto* b = static_cast<const std::uint8_t*>(p); o.insert(o.end(), b, b + n); }
};
struct bin_reader {
    const std::uint8_t* p; std::size_t n, pos = 0; int depth = 0;
    std::size_t left() const { return n - pos; }
    std::uint64_t varint() {
        std::uint64_t v = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (pos >= n) raise(errc::parse_error, "reflect: binary data is truncated");
            const std::uint8_t b = p[pos++];
            if (shift == 63 && b > 1) raise(errc::parse_error, "reflect: varint overflows 64 bits");
            v |= std::uint64_t(b & 0x7f) << shift;
            if (!(b & 0x80)) return v;
        }
        raise(errc::parse_error, "reflect: varint is too long");
    }
    const std::uint8_t* take(std::size_t k) { if (k > left()) raise(errc::parse_error, "reflect: binary data is truncated"); const std::uint8_t* q = p + pos; pos += k; return q; }
    std::size_t count() { const std::uint64_t c = varint(); if (c > left()) raise(errc::parse_error, "reflect: element count exceeds the data"); return std::size_t(c); }   // every element takes at least one byte
};
template <class T> void put(bin_writer& w, const T& v) {
    if constexpr (std::is_same_v<T, bool>) w.o.push_back(v ? 1 : 0);
    else if constexpr (std::is_enum_v<T>) put(w, std::underlying_type_t<T>(v));
    else if constexpr (std::is_floating_point_v<T>) {
        static_assert(sizeof(T) == 4 || sizeof(T) == 8, "reflect: only float and double are supported");
        if constexpr (sizeof(T) == 4) { std::uint32_t b; std::memcpy(&b, &v, 4); for (int i = 0; i < 4; ++i) w.o.push_back(std::uint8_t(b >> (8 * i))); }
        else { std::uint64_t b; std::memcpy(&b, &v, 8); for (int i = 0; i < 8; ++i) w.o.push_back(std::uint8_t(b >> (8 * i))); }
    }
    else if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_signed_v<T>) { const std::int64_t s = std::int64_t(v); w.varint((std::uint64_t(s) << 1) ^ std::uint64_t(s >> 63)); }
        else w.varint(std::uint64_t(v));
    }
    else if constexpr (std::is_same_v<T, std::string>) { w.varint(v.size()); w.raw(v.data(), v.size()); }
    else if constexpr (is_vec<T>::value) { w.varint(v.size()); for (const auto& e : v) put(w, static_cast<const typename T::value_type&>(e)); }
    else if constexpr (is_arr<T>::value) { for (const auto& e : v) put(w, e); }
    else if constexpr (is_opt<T>::value) { w.o.push_back(v ? 1 : 0); if (v) put(w, *v); }
    else if constexpr (is_var<T>::value) { w.varint(v.index()); std::visit([&](const auto& alt) { put(w, alt); }, v); }
    else if constexpr (is_smap<T>::value) { w.varint(v.size()); for (auto& [k, e] : v) { put(w, k); put(w, e); } }
    else {
        static_assert(reflected<T>, "reflect::to_binary: type has no FORGE_REFLECT");
        std::size_t nf = 0; for_each_field(v, [&](std::string_view, const auto&) { ++nf; });
        w.varint(nf);
        for_each_field(v, [&](std::string_view name, const auto& m) {
            bin_writer sub; put(sub, m);
            w.varint(fnv1a32(name)); w.varint(sub.o.size()); w.raw(sub.o.data(), sub.o.size());
        });
    }
}
template <class T> void get(bin_reader& r, T& v) {
    struct nest { bin_reader& r; explicit nest(bin_reader& x) : r(x) { if (++r.depth > 200) raise(errc::parse_error, "reflect: binary data is nested too deeply"); } ~nest() { --r.depth; } } guard(r);
    if constexpr (std::is_same_v<T, bool>) { const std::uint8_t b = *r.take(1); if (b > 1) raise(errc::parse_error, "reflect: bool is not 0 or 1"); v = b != 0; }
    else if constexpr (std::is_enum_v<T>) { std::underlying_type_t<T> u{}; get(r, u); v = T(u); }
    else if constexpr (std::is_floating_point_v<T>) {
        const std::uint8_t* q = r.take(sizeof(T));
        if constexpr (sizeof(T) == 4) { std::uint32_t b = 0; for (int i = 0; i < 4; ++i) b |= std::uint32_t(q[i]) << (8 * i); std::memcpy(&v, &b, 4); }
        else { std::uint64_t b = 0; for (int i = 0; i < 8; ++i) b |= std::uint64_t(q[i]) << (8 * i); std::memcpy(&v, &b, 8); }
    }
    else if constexpr (std::is_integral_v<T>) {
        const std::uint64_t u = r.varint();
        if constexpr (std::is_signed_v<T>) { const std::int64_t s = std::int64_t(u >> 1) ^ -std::int64_t(u & 1); if (s < std::int64_t(std::numeric_limits<T>::min()) || s > std::int64_t(std::numeric_limits<T>::max())) raise(errc::parse_error, "reflect: integer does not fit its field"); v = T(s); }
        else { if (u > std::uint64_t(std::numeric_limits<T>::max())) raise(errc::parse_error, "reflect: integer does not fit its field"); v = T(u); }
    }
    else if constexpr (std::is_same_v<T, std::string>) { const std::uint64_t k = r.varint(); if (k > r.left()) raise(errc::parse_error, "reflect: string runs past the data"); const std::uint8_t* q = r.take(std::size_t(k)); v.assign(reinterpret_cast<const char*>(q), std::size_t(k)); }
    else if constexpr (is_vec<T>::value) {
        const std::size_t k = r.count(); v.clear();
        for (std::size_t i = 0; i < k; ++i) { typename T::value_type e{}; get(r, e); v.push_back(std::move(e)); }
    }
    else if constexpr (is_arr<T>::value) { for (auto& e : v) get(r, e); }
    else if constexpr (is_opt<T>::value) { const std::uint8_t b = *r.take(1); if (b > 1) raise(errc::parse_error, "reflect: optional flag is not 0 or 1"); if (b) { typename T::value_type e{}; get(r, e); v = std::move(e); } else v.reset(); }
    else if constexpr (is_var<T>::value) {
        const std::uint64_t idx = r.varint();
        if (idx >= std::variant_size_v<T>) raise(errc::parse_error, "reflect: variant index out of range");
        [&]<std::size_t... I>(std::index_sequence<I...>) { ((idx == I ? void([&] { std::variant_alternative_t<I, T> e{}; get(r, e); v = T(std::in_place_index<I>, std::move(e)); }()) : void()), ...); }(std::make_index_sequence<std::variant_size_v<T>>{});
    }
    else if constexpr (is_smap<T>::value) {
        const std::size_t k = r.count(); v.clear();
        for (std::size_t i = 0; i < k; ++i) { std::string key; get(r, key); typename T::mapped_type e{}; get(r, e); v[std::move(key)] = std::move(e); }
    }
    else {
        static_assert(reflected<T>, "reflect::from_binary: type has no FORGE_REFLECT");
        const std::size_t nf = r.count();
        for (std::size_t i = 0; i < nf; ++i) {
            const std::uint64_t h = r.varint(), len = r.varint();
            if (len > r.left()) raise(errc::parse_error, "reflect: field runs past the data");
            bin_reader sub{r.take(std::size_t(len)), std::size_t(len)}; sub.depth = r.depth;
            bool done = false;
            for_each_field(v, [&](std::string_view name, auto& m) {
                if (done || fnv1a32(name) != h) return;
                done = true; get(sub, m);
                if (sub.left() != 0) raise(errc::parse_error, "reflect: field '{}' has {} unread byte(s): its type changed", name, sub.left());
            });   // a field this build does not know is skipped
        }
    }
}
inline constexpr std::uint8_t bin_magic[4] = {'F', 'R', 'B', '1'};
} // namespace detail

struct binary_info { std::uint64_t writer_schema = 0; bool schema_matches = false; };
template <class T> std::vector<std::uint8_t> to_binary(const T& v) {
    detail::bin_writer w; w.raw(detail::bin_magic, 4);
    const std::uint64_t h = schema_hash<T>(); for (int i = 0; i < 8; ++i) w.o.push_back(std::uint8_t(h >> (8 * i)));
    detail::put(w, v);
    return std::move(w.o);
}
// Throws forge_exception (errc::parse_error) on malformed data, including trailing bytes.
template <class T> T from_binary(std::span<const std::uint8_t> bytes, binary_info* info = nullptr) {
    if (bytes.size() < 12 || std::memcmp(bytes.data(), detail::bin_magic, 4) != 0) raise(errc::parse_error, "reflect: not a FRB1 binary blob");
    std::uint64_t h = 0; for (int i = 0; i < 8; ++i) h |= std::uint64_t(bytes[4 + std::size_t(i)]) << (8 * i);
    if (info) { info->writer_schema = h; info->schema_matches = h == schema_hash<T>(); }
    detail::bin_reader r{bytes.data() + 12, bytes.size() - 12};
    T out{}; detail::get(r, out);
    if (r.left() != 0) raise(errc::parse_error, "reflect: {} trailing byte(s) after the value", r.left());
    return out;
}
} // namespace forge::reflect
