// reflect: 64-field structs, nested structs, containers / optional / variant, named enums, JSON and binary
// round trips, schema hashes, and field add / remove tolerance between two versions of a struct.
#include "check.hpp"
#include <forge/reflect/reflect.hpp>
#include <cmath>
#include <random>
using namespace forge;

namespace rt {
enum class mode : std::uint8_t { fast, exact, off = 9 };
FORGE_REFLECT_ENUM(mode, fast, exact, off)
enum plain_enum : int { pe_a = -3, pe_b = 4 };   // no names registered: numbers
struct vec3 { float x = 0, y = 0, z = 0; bool operator==(const vec3&) const = default; };
FORGE_REFLECT(vec3, x, y, z)
struct inner { std::string label; std::vector<vec3> pts; std::optional<int> maybe; bool operator==(const inner&) const = default; };
FORGE_REFLECT(inner, label, pts, maybe)
struct everything {
    bool flag = false; std::int8_t i8 = 0; std::uint8_t u8 = 0; std::int16_t i16 = 0; std::uint16_t u16 = 0; std::int32_t i32 = 0; std::uint32_t u32 = 0;
    std::int64_t i64 = 0; std::uint64_t u64 = 0; float f = 0; double d = 0; std::string s; mode m = mode::fast; plain_enum pe = pe_a;
    std::vector<int> vi; std::vector<std::string> vs; std::vector<bool> vb; std::array<double, 3> arr{}; std::optional<inner> opt; std::optional<std::string> none;
    std::variant<int, std::string, vec3> var; std::map<std::string, std::vector<int>> dict; std::unordered_map<std::string, mode> udict; inner in; std::vector<inner> ins;
    std::vector<std::optional<std::variant<bool, double>>> deep;
    bool operator==(const everything&) const = default;
};
FORGE_REFLECT(everything, flag, i8, u8, i16, u16, i32, u32, i64, u64, f, d, s, m, pe, vi, vs, vb, arr, opt, none, var, dict, udict, in, ins, deep)
struct wide {   // 64 fields
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0, a8 = 0, a9 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0, a14 = 0, a15 = 0,
        a16 = 0, a17 = 0, a18 = 0, a19 = 0, a20 = 0, a21 = 0, a22 = 0, a23 = 0, a24 = 0, a25 = 0, a26 = 0, a27 = 0, a28 = 0, a29 = 0, a30 = 0, a31 = 0,
        a32 = 0, a33 = 0, a34 = 0, a35 = 0, a36 = 0, a37 = 0, a38 = 0, a39 = 0, a40 = 0, a41 = 0, a42 = 0, a43 = 0, a44 = 0, a45 = 0, a46 = 0, a47 = 0,
        a48 = 0, a49 = 0, a50 = 0, a51 = 0, a52 = 0, a53 = 0, a54 = 0, a55 = 0, a56 = 0, a57 = 0, a58 = 0, a59 = 0, a60 = 0, a61 = 0, a62 = 0;
    std::string last;
    bool operator==(const wide&) const = default;
};
FORGE_REFLECT(wide, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
              a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, a61, a62, last)
// Two versions of one message: v2 dropped `legacy`, added `added` and `extra`, and reordered.
struct msg_v1 { int id = 0; std::string name; double legacy = 0; std::vector<int> data; };
FORGE_REFLECT(msg_v1, id, name, legacy, data)
struct msg_v2 { std::vector<int> data; int id = 0; std::string added = "default"; std::string name; inner extra; };
FORGE_REFLECT(msg_v2, data, id, added, name, extra)
struct msg_retyped { std::string id; std::string name; };   // id changed type
FORGE_REFLECT(msg_retyped, id, name)
struct node { int v = 0; std::vector<node> kids; bool operator==(const node&) const = default; };   // self-referential
FORGE_REFLECT(node, v, kids)

everything sample() {
    everything e; e.flag = true; e.i8 = -128; e.u8 = 255; e.i16 = -32768; e.u16 = 65535; e.i32 = std::numeric_limits<std::int32_t>::min(); e.u32 = 0xffffffffu;
    e.i64 = std::numeric_limits<std::int64_t>::min(); e.u64 = std::numeric_limits<std::uint64_t>::max(); e.f = -1.5f; e.d = 3.141592653589793; e.s = "h\xc3\xa9llo \"q\"\n";
    e.m = rt::mode::off; e.pe = pe_b; e.vi = {1, -2, 300000}; e.vs = {"", "a", "bc"}; e.vb = {true, false, true}; e.arr = {1.5, -2.5, 1e300};
    e.opt = inner{"in", {{1, 2, 3}, {4, 5, 6}}, 7}; e.var = vec3{9, 8, 7}; e.dict = {{"k", {1, 2}}, {"empty", {}}}; e.udict = {{"a", rt::mode::exact}, {"b", rt::mode::fast}};
    e.in = inner{"x", {}, std::nullopt}; e.ins = {inner{"p", {{0, 0, 1}}, -1}, inner{}}; e.deep = {std::nullopt, std::variant<bool, double>(true), std::variant<bool, double>(2.5)};
    return e;
}
} // namespace rt

TEST(reflect_json_round_trip_all_shapes) {
    auto e = rt::sample();
    e.i64 = -(1ll << 52); e.u64 = 1ull << 52;   // JSON numbers are doubles: stay inside 2^53
    auto j = reflect::to_json(e);
    CHECK(reflect::from_json<rt::everything>(j) == e);
    // through text: json::dump writes non-integers with 9 significant digits, so use a double that survives that
    e.d = 2.25; e.i64 = -(1ll << 40); e.u64 = 1ull << 40; e.arr[2] = 1e30; j = reflect::to_json(e);   // (and integers its integer path prints exactly)
    auto back = reflect::from_json<rt::everything>(json::parse(json::dump(j)));
    CHECK(back == e);
    CHECK(j["m"].str() == "off"); CHECK(j["pe"].num() == 4); CHECK(j["none"].is_null()); CHECK(j["var"]["index"].num() == 2 && j["var"]["value"]["x"].num() == 9);
    CHECK(j["opt"]["pts"][1]["z"].num() == 6); CHECK(j["dict"]["k"][1].num() == 2); CHECK(j["arr"].arr().size() == 3); CHECK(j["vb"][1].b() == false);
    // names and numbers are both accepted for a named enum; an unknown name or a bad variant index is an error
    CHECK(reflect::from_json<rt::mode>(json::parse("\"exact\"")) == rt::mode::exact); CHECK(reflect::from_json<rt::mode>(json::parse("9")) == rt::mode::off);
    CHECK(reflect::to_json(rt::mode(77)).num() == 77);
    CHECK_THROWS(reflect::from_json<rt::mode>(json::parse("\"nope\"")));
    CHECK_THROWS((reflect::from_json<std::variant<int, bool>>(json::parse(R"({"index":2,"value":1})"))));
    CHECK_THROWS((reflect::from_json<std::array<int, 3>>(json::parse("[1,2]"))));
    CHECK_THROWS(reflect::from_json<rt::vec3>(json::parse("[1,2,3]")));
    // numbers that do not fit the field are errors, not undefined conversions
    CHECK_THROWS(reflect::from_json<int>(json::parse("1e30"))); CHECK_THROWS(reflect::from_json<std::uint8_t>(json::parse("256"))); CHECK_THROWS(reflect::from_json<unsigned>(json::parse("-1")));
    CHECK_THROWS(reflect::from_json<std::int64_t>(json::parse("9.3e18"))); CHECK(reflect::from_json<std::int8_t>(json::parse("-128")) == -128);
    CHECK_THROWS((reflect::from_json<std::variant<int, bool>>(json::parse(R"({"index":-1,"value":1})")))); CHECK_THROWS(reflect::from_json<rt::mode>(json::parse("1e9")));
    auto partial = reflect::from_json<rt::inner>(json::parse(R"({"label":"only","unknown":[1,2]})"));   // missing keys keep defaults, unknown keys are ignored
    CHECK(partial.label == "only" && partial.pts.empty() && !partial.maybe);
    CHECK(std::string(reflect::enum_name(rt::mode::exact)) == "exact" && reflect::enum_name(rt::mode(3)) == nullptr && reflect::enum_from_name<rt::mode>("fast") == rt::mode::fast && !reflect::enum_from_name<rt::mode>("x"));
}

TEST(reflect_sixty_four_fields) {
    CHECK(reflect::field_count<rt::wide>() == 64); CHECK(reflect::field_count<rt::everything>() == 26);
    rt::wide w; w.a0 = 1; w.a31 = -31; w.a62 = 62; w.last = "end";
    auto j = reflect::to_json(w); CHECK(j.obj().size() == 64 && j.obj().front().first == "a0" && j.obj().back().first == "last" && j["a31"].num() == -31);
    CHECK(reflect::from_json<rt::wide>(json::parse(json::dump(j))) == w);
    CHECK(reflect::from_binary<rt::wide>(reflect::to_binary(w)) == w);
}

TEST(reflect_binary_round_trip_and_size) {
    auto e = rt::sample();   // full 64-bit extremes survive the binary form
    auto bytes = reflect::to_binary(e); reflect::binary_info info;
    auto back = reflect::from_binary<rt::everything>(bytes, &info);
    CHECK(back == e); CHECK(info.schema_matches && info.writer_schema == reflect::schema_hash<rt::everything>());
    CHECK(std::memcmp(bytes.data(), "FRB1", 4) == 0);
    CHECK_MSG(bytes.size() < json::dump(reflect::to_json(e)).size(), "binary %zu bytes vs JSON %zu", bytes.size(), json::dump(reflect::to_json(e)).size());
    for (float f : {0.0f, -0.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::denorm_min()}) { rt::vec3 v{f, f, f}; auto b = reflect::from_binary<rt::vec3>(reflect::to_binary(v)); CHECK(std::memcmp(&b, &v, sizeof v) == 0); }
    rt::vec3 nanv{std::nanf("1"), 0, 0}; CHECK(std::isnan(reflect::from_binary<rt::vec3>(reflect::to_binary(nanv)).x));
    CHECK(reflect::from_binary<int>(reflect::to_binary(-123456)) == -123456);
    CHECK((reflect::from_binary<std::vector<std::string>>(reflect::to_binary(std::vector<std::string>{"a", ""})) == std::vector<std::string>{"a", ""}));
    rt::node tree{1, {{2, {{4, {}}}}, {3, {}}}};
    CHECK(reflect::from_binary<rt::node>(reflect::to_binary(tree)) == tree); CHECK(reflect::from_json<rt::node>(reflect::to_json(tree)) == tree);
    CHECK(reflect::schema_string<rt::node>().find("...") != std::string::npos);   // the self reference is cut off, not followed forever
}

TEST(reflect_schema_hash_and_versions) {
    CHECK(reflect::schema_string<rt::vec3>() == "{x:f32,y:f32,z:f32,}");
    CHECK(reflect::schema_string<rt::inner>() == "{label:s,pts:[{x:f32,y:f32,z:f32,}],maybe:?i32,}");
    CHECK(reflect::schema_string<rt::mode>() == "e1(fast=0,exact=1,off=9,)");
    CHECK(reflect::schema_hash<rt::msg_v1>() != reflect::schema_hash<rt::msg_v2>()); CHECK(reflect::schema_hash<rt::vec3>() == reflect::schema_hash<rt::vec3>());
    CHECK(reflect::schema_hash<std::vector<int>>() != reflect::schema_hash<std::vector<unsigned>>());
    // old writer, new reader: the dropped field is skipped, new fields keep their defaults, order does not matter
    rt::msg_v1 a; a.id = 42; a.name = "n"; a.legacy = 2.5; a.data = {7, 8, 9};
    reflect::binary_info info; auto b = reflect::from_binary<rt::msg_v2>(reflect::to_binary(a), &info);
    CHECK(b.id == 42 && b.name == "n" && b.data == a.data && b.added == "default" && b.extra == rt::inner{}); CHECK(!info.schema_matches && info.writer_schema == reflect::schema_hash<rt::msg_v1>());
    // new writer, old reader
    rt::msg_v2 n; n.id = 5; n.name = "new"; n.added = "x"; n.data = {1}; n.extra.label = "e";
    auto o = reflect::from_binary<rt::msg_v1>(reflect::to_binary(n)); CHECK(o.id == 5 && o.name == "new" && o.data == n.data && o.legacy == 0);
    auto oj = reflect::from_json<rt::msg_v1>(reflect::to_json(n)); CHECK(oj.id == 5 && oj.name == "new" && oj.legacy == 0);   // JSON tolerates the same changes
    // a field that kept its name but changed type is an error, not silent garbage
    CHECK_THROWS(reflect::from_binary<rt::msg_retyped>(reflect::to_binary(a)));
}

TEST(reflect_binary_rejects_malformed_data) {
    auto good = reflect::to_binary(rt::sample());
    auto rejects = [](const std::vector<std::uint8_t>& b) { try { (void)reflect::from_binary<rt::everything>(b); } catch (const forge_exception&) { return true; } return false; };
    for (std::size_t cut = 0; cut < good.size(); ++cut) { std::vector<std::uint8_t> part(good.begin(), good.begin() + std::ptrdiff_t(cut)); CHECK_MSG(rejects(part), "prefix %zu/%zu accepted", cut, good.size()); }
    auto extra = good; extra.push_back(0); CHECK(rejects(extra));
    auto magic = good; magic[0] = 'X'; CHECK(rejects(magic));
    // random corruption: any outcome but a crash or a non-forge exception (run under ASan in the fuzz build)
    std::mt19937 g(1); int accepted = 0;
    for (int i = 0; i < 20000; ++i) {
        auto m = good; const int edits = 1 + int(g() % 3);
        for (int k = 0; k < edits; ++k) m[12 + g() % (m.size() - 12)] = std::uint8_t(g());
        accepted += !rejects(m);
    }
    CHECK(accepted > 0);   // some edits only change values
    // a count far larger than the data must be refused before anything is allocated
    std::vector<std::uint8_t> bomb(good.begin(), good.begin() + 4); for (int i = 0; i < 8; ++i) bomb.push_back(0);
    for (int i = 0; i < 9; ++i) bomb.push_back(0xff); bomb.push_back(0x01);
    try { (void)reflect::from_binary<std::vector<std::vector<int>>>(bomb); CHECK(false); } catch (const forge_exception&) {}
    std::vector<std::uint8_t> deep(good.begin(), good.begin() + 12);   // node nesting 300 deep: refused by the depth limit
    std::vector<std::uint8_t> body = {2, 0, 0, 0, 0};   // placeholder to keep the vector non-empty
    (void)body;
    rt::node chain; rt::node* cur = &chain; for (int i = 0; i < 300; ++i) { cur->kids.emplace_back(); cur = &cur->kids[0]; }
    auto chain_bytes = reflect::to_binary(chain);
    try { (void)reflect::from_binary<rt::node>(chain_bytes); CHECK(false); } catch (const forge_exception&) {}
    CHECK_THROWS(reflect::from_binary<bool>(std::vector<std::uint8_t>{'F', 'R', 'B', '1', 0, 0, 0, 0, 0, 0, 0, 0, 2}));
    CHECK_THROWS(reflect::from_binary<std::int8_t>(reflect::to_binary(1000)));   // out of range for the field
}
