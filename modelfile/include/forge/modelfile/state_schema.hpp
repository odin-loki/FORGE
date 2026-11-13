#pragma once
// The .fpk "state" section (FORGE_PLAN D.2 "STATE: state schema"): the session state slots a model keeps between
// runs, declared in the file instead of being implied by its blueprint.
//
//   {"version":1,"slots":[
//     {"name":"kv.0.k","kind":"append","shape":[2,8],"dtype":"f32"},                   rows appended per token
//     {"name":"conv.1","kind":"recurrent","op":"causal_conv","shape":[3,32],"dtype":"f32"}   fixed-size state
//   ]}
//
// append     shape is one logical row: the appended tensor without its leading sequence axis ([heads, dim] for a
//            KV cache). A session may store the rows in a KV codec; the schema describes the logical f32 rows.
// recurrent  shape is the whole buffer (graph::state_ops::state_shape); `op` names the state kernel.
//
// The section is optional. Importers write it for models that have state; a file without it (every file written
// before the section existed) gets the same schema derived from its blueprint when one is needed. A session checks
// a saved prefix (session::load_prefix) against the schema before it restores anything, and a model whose declared
// schema differs from what its blueprint builds is refused. The section is covered by the signature like any other.
#include <forge/graph/passes.hpp>
#include <forge/graph/state_ops.hpp>
#include <forge/json/json.hpp>
#include <forge/modelfile/blueprint.hpp>

namespace forge::modelfile {

inline constexpr char state_section[] = "state";

struct state_slot {
    std::string name;
    bool recurrent = false;
    std::string op;          // recurrent slots: the state kernel ("causal_conv", "delta_rule")
    shape_t shape;           // append: one row; recurrent: the buffer
    std::string dtype = "f32";
    bool operator==(const state_slot&) const = default;
};

struct state_schema {
    std::vector<state_slot> slots;   // sorted by name
    const state_slot* find(std::string_view n) const {
        auto it = std::lower_bound(slots.begin(), slots.end(), n, [](const state_slot& s, std::string_view k) { return s.name < k; });
        return it != slots.end() && it->name == n ? &*it : nullptr;
    }
    bool operator==(const state_schema&) const = default;

    std::string dump() const {
        json::array a;
        for (auto& s : slots) {
            json::array sh; for (auto d : s.shape) sh.push_back(double(d));
            json::object o{{"name", s.name}, {"kind", s.recurrent ? "recurrent" : "append"}};
            if (s.recurrent) o.emplace_back("op", s.op);
            o.emplace_back("shape", sh); o.emplace_back("dtype", s.dtype);
            a.push_back(std::move(o));
        }
        return json::dump(json::value(json::object{{"version", 1.0}, {"slots", a}}));
    }
    // Parses a "state" section. The text is untrusted: every count and dimension is bounded, names are unique.
    static state_schema parse(std::string_view text) {
        auto bad = [](const char* why) { raise(errc::format_error, "state section: {}", why); };
        json::value v = json::parse(text);
        if (!v.is_obj()) bad("not a JSON object");
        if (v.num_or("version", 0) != 1) raise(errc::unsupported, "state section: version {} is not supported (this build reads version 1)", v.num_or("version", 0));
        auto* sl = v.find("slots");
        if (!sl || !sl->is_arr()) bad("no slot list");
        if (sl->arr().size() > 65536) bad("too many slots");
        state_schema out;
        for (auto& e : sl->arr()) {
            if (!e.is_obj()) bad("a slot is not an object");
            state_slot s; s.name = e.str_or("name", "");
            if (s.name.empty() || s.name.size() > 256) bad("bad slot name");
            auto kind = e.str_or("kind", "");
            if (kind != "append" && kind != "recurrent") bad("slot kind must be append or recurrent");
            s.recurrent = kind == "recurrent"; s.op = e.str_or("op", "");
            if (s.recurrent != !s.op.empty() || s.op.size() > 64) bad("a recurrent slot names its op; an append slot has none");
            s.dtype = e.str_or("dtype", "");
            if (s.dtype != "f32") bad("slot dtype must be f32");
            auto* sh = e.find("shape");
            if (!sh || !sh->is_arr() || sh->arr().empty() || sh->arr().size() > 8) bad("bad slot shape");
            std::uint64_t n = 1;
            for (auto& d : sh->arr()) {
                if (!d.is_num() || !(d.num() >= 1 && d.num() <= 1e9) || d.num() != std::floor(d.num())) bad("bad slot dimension");
                s.shape.push_back(d.i()); n *= std::uint64_t(d.i());
                if (n > (std::uint64_t(1) << 40)) bad("slot too large");
            }
            out.slots.push_back(std::move(s));
        }
        std::sort(out.slots.begin(), out.slots.end(), [](const state_slot& a, const state_slot& b) { return a.name < b.name; });
        for (std::size_t i = 1; i < out.slots.size(); ++i) if (out.slots[i].name == out.slots[i - 1].name) bad("a slot is declared twice");
        return out;
    }
};

// The schema a blueprint implies: its graph is built once for one token (weights are only asked for their shapes)
// and every state node is listed. `weight_shape` returns the shape of a named tensor. Symbolic input dimensions other
// than the blueprint's params are bound to 1.
inline state_schema derive_state_schema(const blueprint& bp, const std::function<shape_t(const std::string&)>& weight_shape) {
    state_schema out;
    if (bp.states.empty()) return out;
    build_options bo;
    auto bind = [&](const std::vector<bp_io_decl>& ios) { for (auto& d : ios) for (auto& dim : d.dims) if (!dim.empty() && !std::isdigit(static_cast<unsigned char>(dim[0])) && !bp.params.count(dim)) bo.dims[dim] = 1; };
    bind(bp.inputs); bind(bp.outputs);
    for (auto& [en, e] : bp.entries) { bind(e.inputs); bind(e.outputs); }
    bo.dims["seq"] = 1;
    bo.weight = [&](const std::string& n) -> std::pair<shape_t, const float*> { return {weight_shape(n), nullptr}; };
    std::map<std::string, state_slot> by_name;
    std::vector<std::string> graphs{""};   // the main graph, then every entry point: they share the state slots
    for (auto& [en, e] : bp.entries) graphs.push_back(en);
    for (auto& gname : graphs) {
    graph::tensor_graph g;
    try { g = build_graph(bp, bo, gname); graph::verify_graph(g); }
    catch (const std::out_of_range& e) { raise(errc::format_error, "blueprint \"{}\": malformed graph ({})", bp.name, e.what()); }
    for (std::size_t i = 0; i < g.size(); ++i) try {
        const auto& n = g.at(graph::value_id(i));
        state_slot s; s.name = n.name;
        if (n.kind == graph::node_kind::state_append) {
            const auto& xs = g.at(n.inputs.at(0)).shape;
            if (xs.size() < 2) raise(errc::format_error, "blueprint \"{}\": state slot '{}' is appended rows of rank {}", bp.name, n.name, xs.size());
            s.shape.assign(xs.begin() + 1, xs.end());
        } else if (n.kind == graph::node_kind::state_op) {
            std::vector<shape_t> in; for (auto v : n.inputs) in.push_back(g.at(v).shape);
            s.recurrent = true; s.op = n.attrs.str("fn"); s.shape = graph::state_ops::state_shape(s.op, in);
        } else continue;
        if (auto [it, fresh] = by_name.emplace(s.name, s); !fresh && !(it->second == s))
            raise(errc::format_error, "blueprint \"{}\": state slot '{}' is used with two layouts", bp.name, s.name);
    } catch (const std::out_of_range& e) { raise(errc::format_error, "blueprint \"{}\": malformed state node ({})", bp.name, e.what()); }
    }
    for (auto& [k, s] : by_name) out.slots.push_back(std::move(s));
    return out;
}

} // namespace forge::modelfile
