#include "fuzz.hpp"
#include <forge/modelfile/blueprint.hpp>
#include <forge/graph/graph.hpp>
// Parses a blueprint and builds its graph for two sequence lengths, then lowers both graphs against one state
// store, as a session does for its length buckets: a blueprint whose state slots change layout with `seq`, name
// a slot twice or mix appended and recurrent state must be refused there (graph::state_store::check_append).
FORGE_FUZZ({
    auto bp = forge::modelfile::parse_blueprint({reinterpret_cast<const char*>(data), size});
    static std::vector<float> zeros(4096, 0.f);
    forge::graph::state_store st; st.capacity = 8;
    for (forge::i64 seq : {forge::i64(3), forge::i64(1)}) {
        forge::modelfile::build_options o; o.dims = {{"seq", seq}};
        o.weight = [](const std::string&) -> std::pair<forge::shape_t, const float*> { return {forge::shape_t{8, 8}, zeros.data()}; };
        auto g = forge::modelfile::build_graph(bp, o);
        if (g.size() > 512) continue;   // keep one input cheap: lowering is not what a huge graph should spend the budget on
        try { (void)forge::graph::compile(std::move(g), &st); }
        catch (const forge::forge_exception&) {}
        catch (const std::out_of_range&) {}
        catch (const std::invalid_argument&) {}
    }
})
