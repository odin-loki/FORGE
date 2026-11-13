#pragma once
// .fpk -> .fpk re-quantisation (`forge requant`): the weights of a model file that are still in a float codec are
// encoded again in another codec or a mixed-precision recipe (quant_recipe.hpp), without going back to the source
// checkpoint. This is where the "calib" section (modelfile/calib.hpp) is consumed: the recipe reads the recorded
// statistics of the full-precision weights. When the input has no such section and its weights are f32, the
// statistics are computed here (and written to the output, so the next requant has them).
//
// What is re-encoded: rank-2 tensors in f32 / f16 / bf16 that are not normalisation weights and not LoRA factors
// ("adapter.*"). Everything else is copied byte for byte: tensors already in a block codec (decoding and quantising
// again would stack two errors), vectors, norms, stacked expert tensors.
// What is carried over: every section except the signature, precompiled plans and the experts index; the tensor
// alignment; adapters; the state schema. Refused: encrypted input without its key, input with an "experts" section
// (its byte offsets would have to be rebuilt: import the source again instead), and out == in.
#include <forge/import/hf.hpp>
#include <forge/modelfile/adapters.hpp>
#include <forge/modelfile/blueprint_bin.hpp>

namespace forge::import {

struct requant_report {
    std::size_t tensors = 0, requantised = 0, kept = 0, calib_moved = 0;   // calib_moved: types the calibration data changed
    std::uint64_t bytes = 0;
    bool calib_from_file = false;   // statistics came from the input's "calib" section (else computed from its f32 weights)
    std::vector<std::string> notes;
};

struct requant_options {
    import_options base;        // codec, emb_codec, recipe, align (0 = keep the input's)
    bool use_calib = true;      // false: the recipe runs without calibration data
};

inline requant_report requantise(const std::string& in, const std::string& out, const requant_options& ro) {
    const import_options& opt = ro.base;
    check_write(out);
    {
        std::error_code ec;
        if (std::filesystem::exists(out, ec) && std::filesystem::equivalent(in, out, ec)) raise(errc::invalid_argument, "requant: the output must be another file than the input");
    }
    modelfile::reader r(in);
    if (r.locked()) raise(errc::invalid_argument, "{}: the model is encrypted; give its key (FORGE_MODEL_KEY / FORGE_MODEL_KEY_FILE)", in);
    if (r.has_section("experts")) raise(errc::unsupported, "{}: files with an experts index cannot be re-quantised in place; import the source again with the codec you want", in);
    if (auto bad = r.verify(); !bad.empty()) raise(errc::format_error, "{}: refusing to re-quantise a corrupt file ({})", in, bad.front());
    if (!is_mix_ftype(opt.codec) && !graph::codecs::find(opt.codec)) raise(errc::unsupported, "unknown codec '{}'", opt.codec);
    auto bp = modelfile::load_blueprint(r);
    requant_report rep;

    modelfile::calib_table calib;
    if (ro.use_calib && r.has_section(modelfile::calib_section)) { calib = modelfile::calib_table::parse(r.section_text(modelfile::calib_section)); rep.calib_from_file = true; }

    auto is_float = [](const std::string& c) { return c == "f32" || c == "f16" || c == "bf16"; };
    auto quantisable = [&](const modelfile::tensor_entry& e) {
        return is_float(e.codec) && e.shape.size() == 2 && e.name.find("norm") == std::string::npos && !e.name.starts_with(modelfile::adapter_prefix);
    };
    quant_recipe rc = opt.recipe;
    rc.ftype = opt.codec; rc.n_layer = int(bp.param_or("n_layers", 0));
    if (rc.n_expert == 0) rc.n_expert = int(bp.param_or("n_experts", 0));
    std::map<int, int> v_index;
    for (auto& [n, e] : r.tensors()) if (recipe_leaf(n) == "wv" && recipe_layer(n) >= 0) v_index[recipe_layer(n)] = 0;
    { int i = 0; for (auto& [l, vi] : v_index) vi = i++; rc.n_attn_v = i; }

    // tensors in file order, so the output keeps the input's layout order
    std::vector<const modelfile::tensor_entry*> order;
    for (auto& [n, e] : r.tensors()) order.push_back(&e);
    std::sort(order.begin(), order.end(), [](auto* a, auto* b) { return a->offset < b->offset; });

    // without recorded statistics, f32 weights can still be measured here
    if (ro.use_calib && !rep.calib_from_file)
        for (auto* e : order) if (quantisable(*e) && e->codec == "f32") calib.tensors[e->name] = modelfile::tensor_stats::of(r.load(e->name).data);
    if (ro.use_calib && !calib.empty()) rc.calib = &calib;

    modelfile::stream_writer w(out);
    w.set_alignment(opt.align ? import_alignment(opt, in) : r.tensor_alignment());
    const bool float_target = is_float(opt.codec);
    for (auto* e : order) {
        ++rep.tensors;
        if (!quantisable(*e)) { w.add_raw(e->name, e->shape, e->codec, r.tensor_bytes(e->name)); ++rep.kept; continue; }
        const i64 cols = e->shape.back();
        std::string c;
        auto pick = [&](const quant_recipe& q) {
            int il = recipe_layer(e->name);
            auto t = recipe_type(q, e->name, cols, false, il >= 0 && v_index.count(il) ? v_index[il] : -1);
            return is_mix_ftype(t) ? mix_base(t) : t;
        };
        if (opt.mixed()) {
            c = pick(rc);
            if (rc.calib) { auto plain = rc; plain.calib = nullptr; if (pick(plain) != c) ++rep.calib_moved; }
        } else if (e->name == "tok_emb") c = !opt.emb_codec.empty() ? opt.emb_codec : float_target ? opt.codec : std::string("q8_0");
        else c = opt.codec;
        if (c == e->codec) { w.add_raw(e->name, e->shape, e->codec, r.tensor_bytes(e->name)); ++rep.kept; continue; }
        auto t = r.load(e->name);
        w.add_tensor(e->name, e->shape, t.data, c); ++rep.requantised;
    }
    for (auto& [name, s] : r.sections()) {
        if (name == "data" || name == "tensors" || name == "signature" || name == "aot" || name == "encryption" || name == "enc_macs" || name == "manifest" || name == modelfile::calib_section || name == modelfile::shards_section) continue;
        w.set_section(name, std::string(r.section_text(name)));
    }
    if (!calib.empty()) w.set_section(modelfile::calib_section, calib.dump());
    auto man = r.manifest();
    if (man.is_obj()) {
        auto& o = man.obj_mut();
        auto set = [&](const char* k, json::value v) { for (auto& [kk, vv] : o) if (kk == k) { vv = std::move(v); return; } o.emplace_back(k, std::move(v)); };
        set("codec", opt.codec); set("requantised", true);
        std::erase_if(o, [](auto& kv) { return kv.first == "sharded"; });   // the output is one file
    }
    w.set_manifest(man);
    if (r.has_section("signature")) rep.notes.push_back("the signature was dropped (the weights changed): sign the output again");
    if (r.has_section("aot")) rep.notes.push_back("precompiled plans were dropped (they are keyed on the tensor index): run forge aot again");
    if (r.has_section("encryption")) rep.notes.push_back("the output is not encrypted: run forge encrypt on it");
    w.finish();
    rep.bytes = std::filesystem::file_size(out);
    return rep;
}

} // namespace forge::import
