#include "fuzz.hpp"
#include <forge/pack/blueprint.hpp>
FORGE_FUZZ({
    auto bp = forge::pack::parse_blueprint({reinterpret_cast<const char*>(data), size});
    forge::pack::build_options o; o.dims = {{"seq", 3}};
    static std::vector<float> zeros(4096, 0.f);
    o.weight = [](const std::string&) -> std::pair<forge::shape_t, const float*> { return {forge::shape_t{8, 8}, zeros.data()}; };
    (void)forge::pack::build_graph(bp, o);
})
