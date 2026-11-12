#include "fuzz.hpp"
#include <forge/inference/grammar.hpp>
#include <forge/inference/json_schema.hpp>
#include <stdexcept>
// JSON Schema -> GBNF (inference/json_schema.hpp), reached from forge-serve's "response_format". A schema the
// converter accepts must give a grammar the GBNF parser accepts: anything else is a finding.
FORGE_FUZZ({
    auto schema = forge::json::parse({reinterpret_cast<const char*>(data), size});
    std::string g = forge::inference::json_schema_to_gbnf(schema);
    try { forge::inference::grammar gr(g); (void)gr.start(); }
    catch (const forge::forge_exception& e) {
        // a $ref cycle with nothing between the references ({"$ref": "#/$defs/a"} inside a itself) becomes a
        // left-recursive rule, which the grammar parser refuses with a clear message: a rejection, not a finding
        if (std::string_view(e.what()).find("left-recursive") != std::string_view::npos) return;
        throw std::logic_error(std::string("json_schema emitted a grammar that does not parse: ") + e.what() + "\n" + g);
    }
})
