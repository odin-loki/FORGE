#include "fuzz.hpp"
#include <forge/reflect/reflect.hpp>
#include <cstdlib>
// reflect::from_binary on arbitrary bytes. An accepted blob must re-encode to something that decodes to the
// same JSON. Seed: fuzz/corpus/reflect/sample.bin (written by this file's main when built with
// -DFORGE_FUZZ_REFLECT_WRITE_SEED).
namespace fz {
enum class kind : std::uint8_t { a, b, c };
struct leaf { std::string name; std::vector<std::int32_t> v; std::optional<double> w; };
FORGE_REFLECT(leaf, name, v, w)
struct top {
    bool flag = false; std::int64_t big = 0; std::uint16_t small = 0; float f = 0; kind k = kind::a; std::array<std::uint8_t, 3> rgb{};
    std::variant<int, std::string, leaf> var; std::map<std::string, leaf> dict; std::vector<leaf> leaves; std::vector<top> kids;
};
FORGE_REFLECT(top, flag, big, small, f, k, rgb, var, dict, leaves, kids)
}
#if defined(FORGE_FUZZ_REFLECT_WRITE_SEED)
int main(int, char** argv) {
    fz::top t; t.flag = true; t.big = -5; t.small = 7; t.f = 1.5f; t.k = fz::kind::c; t.rgb = {1, 2, 3}; t.var = fz::leaf{"v", {1, -2}, 0.5};
    t.dict["x"] = fz::leaf{"d", {}, std::nullopt}; t.leaves = {fz::leaf{"l", {9}, 2.0}}; t.kids.emplace_back(); t.kids[0].var = std::string("s");
    auto b = forge::reflect::to_binary(t);
    std::ofstream(argv[1], std::ios::binary).write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
}
#else
FORGE_FUZZ({
    auto t = forge::reflect::from_binary<fz::top>({data, size});
    auto again = forge::reflect::from_binary<fz::top>(forge::reflect::to_binary(t));
    if (forge::json::dump(forge::reflect::to_json(again)) != forge::json::dump(forge::reflect::to_json(t))) std::abort();
})
#endif
