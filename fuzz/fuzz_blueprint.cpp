#include "fuzz.hpp"
#include <forge/modelfile/blueprint.hpp>
#include <forge/modelfile/blueprint_bin.hpp>
#include <forge/graph/graph.hpp>
// Parses a blueprint and builds its graph for two sequence lengths, then lowers both graphs against one state
// store, as a session does for its length buckets: a blueprint whose state slots change layout with `seq`, name
// a slot twice or mix appended and recurrent state must be refused there (graph::state_store::check_append).
//
// Both forms of a blueprint are untrusted input (modelfile/blueprint_bin.hpp). An input that starts with the binary
// magic goes through the decoder first; what it accepts must encode back to the same bytes. A text input that lexes
// must survive text -> binary -> text with the same canonical text. Either way the result is then parsed and built.
FORGE_FUZZ({
    namespace mf = forge::modelfile;
    std::string_view in(reinterpret_cast<const char*>(data), size);
    std::string text;
    if (mf::is_binary_blueprint(in)) {
        auto toks = mf::decode_blueprint({data, size});                 // raises format_error on anything non-canonical
        text = mf::canonical_blueprint_text(toks);
        auto again = mf::encode_blueprint(text);
        if (again.size() != size || std::memcmp(again.data(), data, size) != 0) throw std::logic_error("binary blueprint: decode -> text -> encode changed the bytes");
    } else {
        text = std::string(in);
        auto canon = mf::canonical_blueprint_text(mf::bp_lex(text));      // parse_error when it does not lex
        auto bin = mf::encode_blueprint(text);
        if (mf::canonical_blueprint_text(mf::decode_blueprint(bin)) != canon) throw std::logic_error("blueprint: text -> binary -> text changed the canonical text");
        if (mf::canonical_blueprint_text(canon) != canon) throw std::logic_error("blueprint: canonical text is not a fixed point");
    }
    auto bp = forge::modelfile::parse_blueprint(text);
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
