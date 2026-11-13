#pragma once
// Import of a "blueprint directory": a model described directly by its blueprint, for architectures that have no
// template (FORGE_PLAN E.1: "adding a new architecture = writing a blueprint").
//
//   model/
//     blueprint.fbp           the blueprint, as text (any session kind, any opset this build runs, entry points)
//     *.safetensors           the weights, stored under the names the blueprint reads: `W.enc.w1` is tensor "enc.w1"
//     manifest.json           optional: extra manifest fields (bos_token_id, max_position_embeddings, ...)
//     tokenizer.json          optional, copied as the tokenizer section
//
// Every tensor the blueprint reads (main graph and entry points) must be present; tensors it does not read are an
// error unless allow_unused is set. Rank-2 weights take the requested codec, everything else stays f32. The state
// section is derived as for every import.
#include <forge/import/hf.hpp>

namespace forge::import {

inline constexpr char blueprint_dir_file[] = "blueprint.fbp";
inline bool is_blueprint_dir(const std::filesystem::path& dir) { std::error_code ec; return std::filesystem::is_regular_file(dir / blueprint_dir_file, ec); }

inline import_report import_blueprint_dir(const std::filesystem::path& dir, const std::string& out, const import_options& opt = {}) {
    check_write(out);
    auto bp_text = read_file(dir / blueprint_dir_file);
    auto bp = modelfile::parse_blueprint(bp_text);
    modelfile::check_opset(bp);
    std::vector<std::unique_ptr<safetensors>> files;
    {
        std::vector<std::filesystem::path> paths;
        for (auto& e : std::filesystem::directory_iterator(dir)) if (e.is_regular_file() && e.path().extension() == ".safetensors") paths.push_back(e.path());
        std::sort(paths.begin(), paths.end());
        for (auto& p : paths) files.push_back(std::make_unique<safetensors>(p.string()));
    }
    auto find = [&](const std::string& n) -> const safetensors* { for (auto& f : files) if (f->has(n)) return f.get(); return nullptr; };
    modelfile::stream_writer w(out);
    w.set_alignment(import_alignment(opt, dir));
    import_report rep{"blueprint", bp.name, 0, 0};
    const bool quantise = opt.codec != "f32" && !is_mix_ftype(opt.codec);
    if (is_mix_ftype(opt.codec)) raise(errc::unsupported, "{}: the mixed-precision presets need a template; use a plain codec", dir.string());
    for (auto& name : modelfile::blueprint_weights(bp)) {
        auto* f = find(name);
        if (!f) raise(errc::not_found, "{}: the blueprint reads tensor {}, which no .safetensors file in the directory has", dir.string(), name);
        auto t = f->load(name);
        w.add_tensor(name, t.shape, t.data, quantise && t.shape.size() == 2 ? opt.codec : std::string("f32")); ++rep.tensors;
    }
    check_template_tensors(dir.string(), bp.name, bp, w.entries(), opt);
    json::object man{{"format", "fpk"}, {"forge_format_version", double(modelfile::format_version)}, {"architecture", "blueprint"}, {"template", bp.name},
                     {"codec", opt.codec}, {"session", bp.session}};
    if (std::filesystem::exists(dir / "manifest.json")) {   // extra fields; the ones above are the importer's
        auto extra = json::parse(read_file(dir / "manifest.json"));
        if (!extra.is_obj()) raise(errc::format_error, "{}: manifest.json is not a JSON object", dir.string());
        for (auto& [k, v] : extra.obj()) { bool own = false; for (auto& [mk, mv] : man) own = own || mk == k; if (!own && k != "tensor_alignment") man.emplace_back(k, v); }
    }
    if (std::filesystem::exists(dir / "tokenizer.json")) w.set_section("tokenizer", read_file(dir / "tokenizer.json"));
    auto lic = hf_license(dir); lic.store(man, w); rep.license = lic.summary();
    w.set_manifest(json::value(man));
    w.set_section("blueprint", bp_text);
    add_state_section(w, bp);
    w.finish();
    rep.bytes = std::filesystem::file_size(out);
    return rep;
}

} // namespace forge::import
