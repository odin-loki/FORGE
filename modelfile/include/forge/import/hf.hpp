#pragma once
// Hugging Face directory (config.json + *.safetensors + tokenizer.json) -> .fpk
#include <forge/import/safetensors.hpp>
#include <forge/import/hf_quant.hpp>
#include <forge/import/quant_recipe.hpp>
#include <forge/modelfile/blueprint.hpp>
#include <forge/modelfile/fpk.hpp>
#include <forge/security/enforce.hpp>
#include <forge/security/path.hpp>
#include <forge/os/fs.hpp>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <set>
#include <sstream>

namespace forge::import {

// FSS CAP-1: under an active forge::security policy every importer input must pass enforce(fs_read) and the output
// enforce(fs_write) (permission_denied, audited). No policy (profile none): no check.
inline void check_read(const std::filesystem::path& p) { security::enforce(security::capability::fs_read, os::fs::u8str(p)); }
inline void check_write(const std::string& p) { security::enforce(security::capability::fs_write, p); }

inline std::string read_file(const std::filesystem::path& p) {
    check_read(p);
    std::ifstream f(p, std::ios::binary);
    if (!f) raise(errc::io_error, "cannot read {}", p.string());
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

#ifndef FORGE_SOURCE_DIR
#define FORGE_SOURCE_DIR "."
#endif
// Directory of the running executable relative to which an installed tree keeps its templates
// (<prefix>/bin/forge -> <prefix>/share/forge/templates). The build overrides it for non-default install dirs.
#ifndef FORGE_TEMPLATES_RELDIR
#define FORGE_TEMPLATES_RELDIR "../share/forge/templates"
#endif
// Blueprint templates (*.fbp): $FORGE_TEMPLATES when set; else <executable dir>/FORGE_TEMPLATES_RELDIR when that
// directory exists (installed tree); else import/templates in the source tree the binary was built from.
inline std::filesystem::path template_dir() {
    if (auto* e = std::getenv("FORGE_TEMPLATES")) return e;
    if (auto exe = os::exe_path(); !exe.empty()) {
        std::error_code ec;
        auto d = (os::fs::u8path(exe).parent_path() / FORGE_TEMPLATES_RELDIR).lexically_normal();
        if (std::filesystem::is_directory(d, ec)) return d;
    }
    return std::filesystem::path(FORGE_SOURCE_DIR) / "import" / "templates";
}
// FSS CAP-3: a blueprint template is looked up by name (`--template`, import_options::template_name), so the name is
// joined onto template_dir() with security::try_safe_join: `..`, absolute paths, drive letters, ':' and reserved
// device names are refused, and so is a name that resolves (through a symlink) to a file outside the directory.
inline std::filesystem::path template_path(const std::string& name) {
    auto dir = template_dir();
    if (name.empty()) raise(errc::invalid_argument, "template: empty name");
    auto j =security::try_safe_join(os::fs::u8str(dir), name + ".fbp");
    if (!j) raise(errc::invalid_argument, "template '{}': not a file inside the template directory {} ({})", name, dir.string(), j.err.what);
    return os::fs::u8path(j.value);
}

// Substitute {{key}} with config values (numbers formatted losslessly). {{key|default}} takes `default` (may be empty)
// when the config has no such key: optional pieces such as {{rope_args|}} and {{opset|1}}.
inline std::string instantiate(std::string tpl, const json::value& cfg) {
    std::size_t p = 0;
    while ((p = tpl.find("{{", p)) != std::string::npos) {
        auto e = tpl.find("}}", p);
        if (e == std::string::npos) raise(errc::parse_error, "template: unterminated '{{{{' placeholder");
        auto key = tpl.substr(p + 2, e - p - 2);
        std::string dflt; bool has_default = false;
        if (auto bar = key.find('|'); bar != std::string::npos) { dflt = key.substr(bar + 1); key.resize(bar); has_default = true; }
        auto* v = cfg.find(key);
        if (!v && !has_default) raise(errc::not_found, "template needs config key '{}'", key);
        std::string s;
        if (!v) s = dflt;
        else if (v->is_num()) { char b[40]; std::snprintf(b, 40, "%.17g", v->num()); s = b; } else s = v->str();
        tpl.replace(p, e - p + 2, s);
        p += s.size();   // substituted text is not scanned again
    }
    return tpl;
}

struct import_options {
    std::string codec = "f32"; std::string template_name;
    std::string emb_codec;   // tok_emb codec; empty = q8_0 when codec is quantised, else f32
    // Quantised sources (GGUF files, AWQ / GPTQ checkpoints): "" or "keep" stores the quantised tensors as they are;
    // a codec name decodes and re-quantises them (GGUF: as that codec; AWQ / GPTQ: as `codec`).
    std::string requant;
    // Mixed precision (codec "q4_k_m"/"q5_k_m", or any override set): per-tensor types from quant_recipe.hpp.
    quant_recipe recipe;
    bool mixed() const { return is_mix_ftype(codec) || !recipe.output_type.empty() || !recipe.token_embedding_type.empty() || !recipe.tensor_types.empty() || recipe.head_bf16; }
    // Every source tensor must be consumed by the template (else the import fails and lists them). allow_unused imports
    // anyway: HF leaves the leftovers out, GGUF keeps them as `extra.<name>`; both list them in import_report::unused.
    bool allow_unused = false;
    // Source tensors that are known not to matter, on top of the built-in buffers (ignored_tensor below). An entry is a
    // whole name, `prefix*` or `*suffix`.
    std::vector<std::string> ignore_tensors;
};

struct import_report {
    std::string arch, template_name; std::size_t tensors = 0, bytes = 0;
    std::vector<std::string> unused;   // "name [shape]" of source tensors left out under allow_unused
    std::string license;               // the licence recorded in the manifest ("" = the source names none)
    std::vector<std::string> notes{}; // what the import changed (transcoded types, re-quantisation warnings); the CLI prints them
};

// Buffers that checkpoints carry but no graph reads (position_ids, masked_bias, inv_freq), plus import_options::ignore_tensors.
inline bool ignored_tensor(const std::string& n, const import_options& o) {
    auto dot = n.rfind('.'); std::string_view leaf = std::string_view(n).substr(dot == std::string::npos ? 0 : dot + 1);
    for (const char* k : {"position_ids", "masked_bias", "inv_freq"}) if (leaf == k) return true;
    for (auto& p : o.ignore_tensors) {
        if (p.empty()) continue;
        if (p.back() == '*' ? n.starts_with(std::string_view(p).substr(0, p.size() - 1)) : p.front() == '*' ? n.ends_with(std::string_view(p).substr(1)) : n == p) return true;
    }
    return false;
}

// Source tensors the template did not consume: an error unless allow_unused (the importers call this before finish()).
inline void check_unused(const std::string& src, const std::string& tname, const std::vector<std::string>& unused, const import_options& opt) {
    if (unused.empty() || opt.allow_unused) return;
    std::string list;
    for (std::size_t i = 0; i < unused.size() && i < 24; ++i) list += (i ? ", " : "") + unused[i];
    if (unused.size() > 24) list += std::format(", ... ({} more)", unused.size() - 24);
    raise(errc::unsupported, "{}: {} source tensor(s) are not used by template '{}': {}. Importing would drop weights the model may need; "
          "pass --allow-unused (import_options::allow_unused) to import anyway, or --ignore-tensor NAME for buffers known to be harmless",
          src, unused.size(), tname, list);
}

// The blueprint and the written tensors must agree: every W.<path> the graph reads was written (else the template wants
// a tensor the source does not have), and every written tensor is read (else the importer and the template disagree,
// e.g. --template llama on a checkpoint with QK-norm weights). `extra.*` (GGUF leftovers under allow_unused) is exempt.
inline void check_template_tensors(const std::string& src, const std::string& tname, const modelfile::blueprint& bp,
                                   const std::vector<modelfile::tensor_entry>& written, const import_options& opt) {
    std::set<std::string> need; for (auto& n : modelfile::blueprint_weights(bp)) need.insert(n);
    std::set<std::string> have; for (auto& t : written) have.insert(t.name);
    std::string missing, dead; std::size_t nm = 0, nd = 0;
    for (auto& n : need) if (!have.count(n) && ++nm <= 24) missing += (missing.empty() ? "" : ", ") + n;
    if (nm) raise(errc::not_found, "{}: template '{}' reads {} tensor(s) the import did not produce: {}{}", src, tname, nm, missing, nm > 24 ? ", ..." : "");
    for (auto& t : written) if (!need.count(t.name) && !t.name.starts_with("extra.") && ++nd <= 24) dead += (dead.empty() ? "" : ", ") + t.name + " " + shape_str(t.shape);
    if (nd && !opt.allow_unused)
        raise(errc::unsupported, "{}: template '{}' never reads {} imported tensor(s): {}{} (wrong --template for this model? --allow-unused imports anyway)",
              src, tname, nd, dead, nd > 24 ? ", ..." : "");
}

// RoPE frequency scaling (config.json rope_scaling / rope_parameters, GGUF <arch>.rope.scaling.*), normalised to the
// rope op's attrs (ops/rope.hpp). Templates take it through {{rope_args|}} (appended to each rope call), {{attn_args|}}
// (YaRN's attention factor: q and k are each scaled by it, so attention's scale is multiplied by its square) and
// {{opset|1}} (2 when any scaling attr is emitted, else 1).
struct rope_spec {
    ops::rope_scaling type = ops::rope_scaling::none;
    double factor = 1, orig_ctx = 0, low_freq = 1, high_freq = 4, beta_fast = 32, beta_slow = 1, attn_factor = 1;
    bool truncate = true;
    std::vector<float> freq_factors;   // per-pair multipliers (GGUF rope_freqs.weight)
    bool active() const { return type != ops::rope_scaling::none || !freq_factors.empty(); }
    static std::string g(double v) { char b[40]; std::snprintf(b, 40, "%.17g", v); return b; }
    std::string rope_args() const {
        std::string s;
        using rs = ops::rope_scaling;
        if (type != rs::none) s += std::format(", scaling: {}, factor: {}", int(type), g(factor));
        if (type == rs::dynamic || type == rs::yarn || type == rs::llama3) s += ", orig_ctx: " + g(orig_ctx);
        if (type == rs::llama3) s += ", low_freq: " + g(low_freq) + ", high_freq: " + g(high_freq);
        if (type == rs::yarn) s += ", beta_fast: " + g(beta_fast) + ", beta_slow: " + g(beta_slow) + ", truncate: " + (truncate ? "1" : "0");
        if (!freq_factors.empty()) {
            s += ", freq_factors: [";
            for (std::size_t i = 0; i < freq_factors.size(); ++i) { std::uint32_t u; std::memcpy(&u, &freq_factors[i], 4); s += (i ? ", " : "") + std::to_string(u); }
            s += "]";
        }
        return s;
    }
    std::string attn_args(double head_dim) const { return attn_factor == 1.0 ? std::string() : ", scale: " + g(attn_factor * attn_factor / std::sqrt(head_dim)); }
    // Checks the spec against the template text and the rope op (bad parameters raise here, not at the first run).
    void validate(const std::string& tname, const std::string& tpl, double head_dim) const {
        if (!active() && attn_factor == 1.0) return;
        if (active() && tpl.find("{{rope_args") == std::string::npos)
            raise(errc::unsupported, "template '{}' does not take RoPE frequency scaling (it has no rope_args placeholder); the model needs it", tname);
        if (attn_factor != 1.0 && tpl.find("{{attn_args") == std::string::npos)
            raise(errc::unsupported, "template '{}' does not take a YaRN attention factor (it has no attn_args placeholder); the model needs it", tname);
        if (!(attn_factor > 0) || !std::isfinite(attn_factor)) raise(errc::invalid_argument, "rope scaling: bad attention factor {}", attn_factor);
        ops::attrs a{{"base", 10000.0}};
        if (type != ops::rope_scaling::none) { a.set("scaling", double(int(type))); a.set("factor", factor); a.set("orig_ctx", orig_ctx);
                                               a.set("low_freq", low_freq); a.set("high_freq", high_freq); a.set("beta_fast", beta_fast); a.set("beta_slow", beta_slow); }
        i64 d = i64(head_dim);
        if (!freq_factors.empty() && i64(freq_factors.size()) != d / 2)
            raise(errc::shape_mismatch, "rope scaling: {} frequency factors for head dimension {} (needs {})", freq_factors.size(), d, d / 2);
        (void)ops::rope_inv_freq(a, d, 1.0);
    }
};

// config.json: `rope_scaling` (transformers 4) or `rope_parameters` (transformers 5), keyed by rope_type (older: type).
// Follows transformers/modeling_rope_utils.py as of transformers 5: the pretraining length of yarn / llama3 is the
// config's top-level original_max_position_embeddings, else the one in the rope dict, else max_position_embeddings; a
// yarn factor that is not given is max_position_embeddings / that length. (transformers 4.46 took max_position_embeddings
// as YaRN's pretraining length; the two agree for configs whose factor is that ratio.) LongRoPE and unknown types are refused.
inline rope_spec rope_spec_from_hf(const json::value& cfg) {
    rope_spec r;
    const json::value* rs = cfg.find("rope_scaling");
    if (!rs || !rs->is_obj()) rs = cfg.find("rope_parameters");
    if (!rs || !rs->is_obj()) return r;
    auto num = [&](const char* k) -> const json::value* { auto* v = rs->find(k); return v && v->is_num() ? v : nullptr; };
    auto need = [&](const char* k) -> double { if (auto* v = num(k)) return v->num(); raise(errc::format_error, "config.json: rope scaling needs a numeric '{}'", k); };
    std::string type = "default";
    if (auto* t = rs->find("rope_type"); t && t->is_str()) type = t->str();
    else if (auto* t2 = rs->find("type"); t2 && t2->is_str()) type = t2->str();
    const json::value* mp = cfg.find("max_position_embeddings");
    auto max_pos = [&]() -> double { if (mp && mp->is_num()) return mp->num(); raise(errc::format_error, "config.json: rope scaling '{}' needs max_position_embeddings", type); };
    if (type == "default") return r;
    auto orig_ctx = [&]() -> double {   // standardize_rope_params
        if (auto* t = cfg.find("original_max_position_embeddings"); t && t->is_num()) return t->num();
        if (auto* o = num("original_max_position_embeddings")) return o->num();
        return max_pos();
    };
    if (type == "linear") { r.type = ops::rope_scaling::linear; r.factor = need("factor"); }
    else if (type == "dynamic") { r.type = ops::rope_scaling::dynamic; r.factor = need("factor"); r.orig_ctx = max_pos(); }
    else if (type == "yarn") {
        r.type = ops::rope_scaling::yarn; r.orig_ctx = orig_ctx();
        if (!(r.orig_ctx > 0)) raise(errc::format_error, "config.json: rope scaling: original_max_position_embeddings must be positive");
        r.factor = num("factor") ? num("factor")->num() : max_pos() / r.orig_ctx;
        auto mscale = [&](double m) { return r.factor <= 1 ? 1.0 : 0.1 * m * std::log(r.factor) + 1.0; };
        auto *af = num("attention_factor"), *ms = num("mscale"), *ma = num("mscale_all_dim");
        r.attn_factor = af ? af->num() : ms && ma && ms->num() != 0 && ma->num() != 0 ? mscale(ms->num()) / mscale(ma->num()) : mscale(1.0);
        if (auto* v = num("beta_fast"); v && v->num() != 0) r.beta_fast = v->num();
        if (auto* v = num("beta_slow"); v && v->num() != 0) r.beta_slow = v->num();
        if (auto* t = rs->find("truncate"); t && t->is_bool()) r.truncate = t->b();
    } else if (type == "llama3") {
        r.type = ops::rope_scaling::llama3; r.factor = need("factor"); r.low_freq = need("low_freq_factor"); r.high_freq = need("high_freq_factor");
        r.orig_ctx = orig_ctx();
    } else if (type == "longrope") raise(errc::unsupported, "rope scaling type 'longrope' (LongRoPE: per-dimension short/long factor tables) is not supported");
    else raise(errc::unsupported, "rope scaling type '{}' is not supported (supported: default, linear, dynamic, yarn, llama3)", type);
    return r;
}

// The model's licence as the source states it. `id` is the short identifier (SPDX-style, e.g. "apache-2.0"), `text`
// the licence text where the source ships one. Stored in the manifest (license, license_name, license_link,
// license_files) and the optional `license` section (docs/format.md).
struct license_info {
    std::string id, name, link, text; std::vector<std::string> files;
    bool empty() const { return id.empty() && name.empty() && link.empty() && text.empty() && files.empty(); }
    void add_file(const std::string& fname, std::string body) {   // several files: each under a "===== NAME =====" line
        if (files.size() == 1) text = "===== " + files[0] + " =====\n" + text;
        if (!files.empty()) text += (text.ends_with("\n") ? "" : "\n") + std::string("===== ") + fname + " =====\n";
        text += body; files.push_back(fname);
    }
    template <class W> void store(json::object& man, W& w) const {   // W: modelfile::writer or stream_writer
        if (!id.empty()) man.emplace_back("license", id);
        if (!name.empty()) man.emplace_back("license_name", name);
        if (!link.empty()) man.emplace_back("license_link", link);
        if (!files.empty()) { json::array a; for (auto& f : files) a.push_back(json::value(f)); man.emplace_back("license_files", a); }
        if (!text.empty()) w.set_section("license", text);
    }
    std::string summary() const { return !id.empty() ? id : !name.empty() ? name : !files.empty() ? files[0] : !link.empty() ? link : !text.empty() ? "(text)" : ""; }
};

// A Hugging Face model directory: the README.md front matter (license, license_name, license_link) and the LICENSE* /
// LICENCE* / COPYING* / NOTICE* files next to the weights (each up to 1 MiB of text).
inline license_info hf_license(const std::filesystem::path& dir) {
    license_info li;
    auto trim = [](std::string s) {
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.pop_back();
        std::size_t b = 0; while (b < s.size() && (s[b] == ' ' || s[b] == '\t')) ++b;
        s.erase(0, b);
        if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) s = s.substr(1, s.size() - 2);
        return s;
    };
    std::vector<std::filesystem::path> files;
    std::filesystem::path readme;
    for (auto& f : std::filesystem::directory_iterator(dir)) {
        if (!f.is_regular_file()) continue;
        std::string up = os::fs::u8str(f.path().filename());
        for (auto& c : up) c = char(std::toupper((unsigned char)c));
        if (up == "README.MD") readme = f.path();
        else if (up.starts_with("LICENSE") || up.starts_with("LICENCE") || up.starts_with("COPYING") || up.starts_with("NOTICE")) files.push_back(f.path());
    }
    if (!readme.empty()) {
        std::istringstream in(read_file(readme)); std::string line; std::string* list = nullptr;
        if (std::getline(in, line) && trim(line) == "---")
            while (std::getline(in, line)) {
                if (trim(line) == "---" || trim(line) == "...") break;
                if (list && trim(line).starts_with("- ")) {   // license:\n- mit
                    auto v = trim(trim(line).substr(2));
                    if (!v.empty()) *list += (list->empty() ? "" : ", ") + v;
                    continue;
                }
                list = nullptr;
                for (auto [key, dst] : {std::pair<const char*, std::string*>{"license:", &li.id}, {"license_name:", &li.name}, {"license_link:", &li.link}})
                    if (line.starts_with(key)) {
                        *dst = trim(line.substr(std::strlen(key)));
                        if (dst->empty()) list = dst;
                        else if (dst->front() == '[' && dst->back() == ']') *dst = trim(dst->substr(1, dst->size() - 2));
                    }
            }
    }
    std::sort(files.begin(), files.end());
    for (auto& f : files) {
        auto fname = os::fs::u8str(f.filename());
        if (std::filesystem::file_size(f) > (1u << 20)) { li.add_file(fname, "(not stored: larger than 1 MiB)\n"); continue; }
        li.add_file(fname, read_file(f));
    }
    return li;
}

// The .fpk "experts" section (format minor 1): one entry per stacked expert tensor [n_experts, rows, k] (moe_ffn
// gate/up/down). Expert e of a tensor occupies bytes [offset + e * expert_bytes, + expert_bytes) of the data section
// (file offset = 64 + offset for stream_writer files); "aligned" says every expert slice starts on a 4 KiB file offset,
// so an expert can be read with one direct (O_DIRECT / unbuffered) read. Groundwork for streaming experts from disk.
struct expert_index {
    json::array ents;
    void add(const modelfile::tensor_entry& e, i64 layer, const char* proj) {
        if (e.shape.size() != 3 || e.shape[0] < 1) raise(errc::shape_mismatch, "expert tensor {} has shape {}", e.name, shape_str(e.shape));
        std::uint64_t ne = std::uint64_t(e.shape[0]), per = e.bytes / ne, foff = 64 + e.offset;
        ents.push_back(json::object{{"layer", double(layer)}, {"proj", std::string(proj)}, {"name", e.name}, {"codec", e.codec},
                                    {"n_experts", double(ne)}, {"offset", double(e.offset)}, {"file_offset", double(foff)},
                                    {"expert_bytes", double(per)}, {"aligned", foff % modelfile::expert_alignment == 0 && per % modelfile::expert_alignment == 0}});
    }
    bool empty() const { return ents.empty(); }
    std::string dump() const {
        return json::dump(json::value(json::object{{"version", 1.0}, {"align", double(modelfile::expert_alignment)}, {"tensors", ents}}));
    }
};

inline import_report import_hf(const std::filesystem::path& dir, const std::string& out, const import_options& opt_in = {}) {
    check_read(dir); check_write(out);
    auto cfg = json::parse(read_file(dir / "config.json"));
    // AWQ / GPTQ checkpoint (hf_quant.hpp). keep (the default): quantised Linears are stored as grouped-codec blocks
    // where they fit, everything else as f16 (these checkpoints are fp16). --codec C: decoded and encoded as C.
    import_options opt = opt_in;
    const auto qscheme = quant_scheme_of(cfg);
    const bool qkeep = qscheme && (opt.requant.empty() || opt.requant == "keep");
    if (qkeep) opt.codec = "f16";
    std::string arch0 = cfg.find("architectures") ? cfg["architectures"][0].str() : "";
    if (cfg.find("text_config") && cfg["text_config"].is_obj()) {   // multimodal wrapper (Qwen3.5): import the text model
        json::object t = cfg["text_config"].obj();
        bool has_arch = false; for (auto& [k, v] : t) has_arch |= k == "architectures";
        if (!has_arch && !arch0.empty()) t.emplace_back("architectures", json::value(json::array{json::value(arch0)}));
        for (auto& [k, v] : cfg.obj()) if (k == "tie_word_embeddings" || k == "bos_token_id" || k == "eos_token_id") {
            bool dup = false; for (auto& [kk, vv] : t) dup |= kk == k; if (!dup) t.emplace_back(k, v);
        }
        cfg = json::value(t);
    }
    std::string arch = cfg.find("architectures") ? cfg["architectures"][0].str() : cfg.str_or("model_type", "unknown");
    std::string tname = opt.template_name;
    if (tname.empty()) {
        if (arch == "Qwen2ForCausalLM" || cfg.str_or("model_type", "") == "qwen2") tname = "qwen2";   // llama + Q/K/V biases
        else if (arch == "LlamaForCausalLM" || arch == "MistralForCausalLM" || cfg.str_or("model_type", "") == "llama")
            tname = "llama";
        else if (arch == "Qwen3ForCausalLM" || cfg.str_or("model_type", "") == "qwen3") tname = "qwen3";
        else if (arch == "Qwen3_5ForConditionalGeneration" || arch == "Qwen3_5ForCausalLM" || cfg.str_or("model_type", "") == "qwen3_5")
            tname = "qwen3_5";
        else if (cfg.str_or("model_type", "") == "qwen3_5_text") tname = "qwen3_5";
        else if (arch == "MixtralForCausalLM" || cfg.str_or("model_type", "") == "mixtral") tname = "mixtral";
        else if (arch == "DeepseekV2ForCausalLM" || arch == "DeepseekV3ForCausalLM" || cfg.str_or("model_type", "") == "deepseek_v2") tname = "deepseek_v2";
        else if (arch == "Qwen3MoeForCausalLM" || cfg.str_or("model_type", "") == "qwen3_moe") tname = "qwen3_moe";
        else raise(errc::unsupported, "no blueprint template for architecture {} (pass --template)", arch);
    }
    // normalise config with defaults the template needs
    json::object c = cfg.obj();
    auto num = [&](const char* k, double d) { return cfg.num_or(k, d); };
    double heads = num("num_attention_heads", 0), hidden = num("hidden_size", 0);
    auto set = [&](const std::string& k, json::value v) {
        for (auto& [kk, vv] : c) if (kk == k) { vv = v; return; }
        c.emplace_back(k, v);
    };
    set("head_dim", num("head_dim", hidden / heads));
    set("num_key_value_heads", num("num_key_value_heads", heads));
    {   // transformers 5 keeps rope_theta inside rope_parameters
        const json::value* rp = cfg.find("rope_parameters");
        set("rope_theta", rp && rp->is_obj() && rp->find("rope_theta") && (*rp)["rope_theta"].is_num() ? (*rp)["rope_theta"].num() : num("rope_theta", 10000.0));
    }
    set("rms_norm_eps", num("rms_norm_eps", 1e-6));
    set("rope_neox", 1.0);  // HF checkpoints use rotate_half (NeoX) RoPE
    // RoPE frequency scaling (linear, dynamic NTK, YaRN, llama3): rope attrs through {{rope_args|}} / {{attn_args|}}.
    auto tpl_text = read_file(template_path(tname));   // FSS CAP-3: the name is joined safely onto template_dir()
    rope_spec rsp = rope_spec_from_hf(cfg);
    if (tname == "qwen3_5" && rsp.active()) raise(errc::unsupported, "qwen3_5: rope scaling is not supported by the template");
    if (tname != "deepseek_v2") rsp.validate(tname, tpl_text, num("head_dim", hidden / heads));   // deepseek_v2 refuses rope_scaling below
    set("rope_args", json::value(rsp.rope_args()));
    set("attn_args", json::value(rsp.attn_args(num("head_dim", hidden / heads))));
    set("opset", rsp.active() ? 2.0 : 1.0);
    if (tname == "deepseek_v2") {
        double nope = num("qk_nope_head_dim", 128), rd = num("qk_rope_head_dim", 64);
        set("qk_head_dim", nope + rd);
        set("attn_scale", 1.0 / std::sqrt(nope + rd));
        if ((cfg.find("rope_scaling") && cfg["rope_scaling"].is_obj()) || rsp.active() || rsp.attn_factor != 1.0)
            raise(errc::unsupported, "deepseek_v2: rope_scaling (YaRN) is not supported yet");
        if (cfg.str_or("topk_method", "greedy") != "greedy" || cfg.str_or("scoring_func", "softmax") != "softmax")
            raise(errc::unsupported, "deepseek_v2: only greedy top-k with softmax scoring is supported");
        bool qlora = cfg.find("q_lora_rank") && cfg["q_lora_rank"].is_num();
        set("mla_q_expr", json::value(std::string(qlora ? "linear(rms_norm(linear(h, W.layer[l].wq_a), W.layer[l].q_norm, eps), W.layer[l].wq_b)"
                                                        : "linear(h, W.layer[l].wq)")));
        set("first_k_dense_replace", num("first_k_dense_replace", 0));
        set("norm_topk_prob", cfg.find("norm_topk_prob") && cfg["norm_topk_prob"].is_bool() ? json::value(cfg["norm_topk_prob"].b() ? 1.0 : 0.0) : json::value(0.0));
        set("routed_scaling_factor", num("routed_scaling_factor", 1.0));
        if (num("n_group", 1) > 1 && cfg.str_or("topk_method", "greedy") != "greedy") raise(errc::unsupported, "deepseek_v2: group-limited routing");
    }
    if (tname == "qwen3_moe") {
        if (num("decoder_sparse_step", 1) != 1) raise(errc::unsupported, "qwen3_moe: decoder_sparse_step != 1 (dense layers in between)");
        if (const auto* ml = cfg.find("mlp_only_layers"); ml && ml->is_arr() && !ml->arr().empty()) raise(errc::unsupported, "qwen3_moe: mlp_only_layers");
        set("norm_topk_prob", cfg.find("norm_topk_prob") && cfg["norm_topk_prob"].is_bool() ? json::value(cfg["norm_topk_prob"].b() ? 1.0 : 0.0) : json::value(0.0));
        set("routed_scaling_factor", 1.0);
    }
    if (tname == "qwen3_5") {
        const json::value* rp = cfg.find("rope_parameters");
        auto rnum = [&](const char* k, double d) { return rp && rp->is_obj() && rp->find(k) ? (*rp)[k].num() : cfg.num_or(k, d); };
        set("rope_theta", rnum("rope_theta", 10000.0));
        double hd = num("head_dim", hidden / heads);
        set("rotary_dim", std::floor(hd * rnum("partial_rotary_factor", 1.0) + 1e-9));
        if (rnum("partial_rotary_factor", 1.0) >= 1.0) raise(errc::unsupported, "qwen3_5: full-width RoPE is not supported by the template");
        double iv = num("full_attention_interval", 4), L0 = num("num_hidden_layers", 0);
        if (iv < 1 || std::fmod(L0, iv) != 0) raise(errc::unsupported, "qwen3_5: {} layers is not a multiple of full_attention_interval {}", L0, iv);
        if (const auto* lt = cfg.find("layer_types"); lt && lt->is_arr()) {
            const auto& a = lt->arr();
            for (std::size_t i = 0; i < a.size(); ++i)
                if (a[i].str() != ((i + 1) % std::size_t(iv) == 0 ? "full_attention" : "linear_attention"))
                    raise(errc::unsupported, "qwen3_5: layer_types[{}] = {} breaks the every-{}th-layer-full pattern", i, a[i].str(), iv);
        }
        if (cfg.find("num_experts") && cfg.num_or("num_experts", 0) > 0) raise(errc::unsupported, "qwen3_5: MoE variants are not supported yet");
        if (cfg.str_or("hidden_act", "silu") != "silu") raise(errc::unsupported, "qwen3_5: hidden_act must be silu");
        set("full_attention_interval", iv);
        set("n_blocks", L0 / iv);
        set("linear_conv_kernel_dim", num("linear_conv_kernel_dim", 4));
    }
    json::value cv(c);
    auto bp_text = instantiate(tpl_text, cv);
    auto bp = modelfile::parse_blueprint(bp_text);  // validate before writing anything
    modelfile::check_opset(bp);

    // gather safetensors shards
    std::vector<std::unique_ptr<safetensors>> shards;
    for (auto& f : std::filesystem::directory_iterator(dir))
        if (f.path().extension() == ".safetensors") shards.push_back(std::make_unique<safetensors>(f.path().string()));
    if (shards.empty()) raise(errc::not_found, "{}: no .safetensors files", dir.string());
    // Vision tower and multi-token-prediction heads are not part of the text model: never loaded.
    auto skipped = [](const std::string& n) {
        for (const char* p : {"visual.", "model.visual.", "mtp.", "model.mtp."}) if (n.starts_with(p)) return true;
        return false;
    };
    quantised_checkpoint qck(qscheme, shards);
    std::set<std::string> consumed;   // source tensors the template took: the rest is reported (check_unused)
    auto find = [&](const std::string& n) -> tensor {
        if (skipped(n)) raise(errc::not_found, "tensor {} belongs to a skipped (vision/mtp) tower", n);
        qck.forget();
        for (auto& s : shards) if (s->has(n)) { consumed.insert(n); return s->load(n); }
        if (qck.has(n)) {   // <base>.qweight / qzeros / scales (GPTQ: g_idx; an all-zero bias is accepted), decoded
            auto base = n.substr(0, n.size() - 7);
            for (const char* sfx : {".qweight", ".qzeros", ".scales", ".g_idx", ".bias"}) consumed.insert(base + sfx);
            return qck.load(n);
        }
        raise(errc::not_found, "{}: template '{}' needs tensor {}, which the checkpoint does not have", dir.string(), tname, n);
    };
    auto has = [&](const std::string& n) { if (skipped(n)) return false; for (auto& s : shards) if (s->has(n)) return true; return qck.has(n); };
    // Qwen3.5 multimodal checkpoints keep the text model under model.language_model.
    std::string mp = has("model.language_model.embed_tokens.weight") ? "model.language_model." : "model.";

    modelfile::stream_writer w(out);   // each tensor is encoded, written and freed before the next is loaded
    import_report rep{arch, tname, 0, 0};
    i64 L = i64(bp.param("n_layers"));
    quant_recipe rc = opt.recipe;
    rc.ftype = opt.codec; rc.n_layer = int(L);
    if (rc.n_expert == 0 && bp.params.count("n_experts")) rc.n_expert = int(bp.param("n_experts"));
    std::vector<int> v_index(std::size_t(L), -1);
    for (i64 l = 0; l < L; ++l)
        if (has(std::format("{}layers.{}.self_attn.v_proj.weight", mp, l))) v_index[std::size_t(l)] = rc.n_attn_v++;
    bool in_deltanet = false;
    expert_index xi;
    auto put = [&](const std::string& n, const tensor& t, bool quantisable, std::size_t align = modelfile::alignment) {
        // an AWQ / GPTQ Linear that fits the grouped codec keeps its codes (the tensor just read by find)
        if (auto nat = qck.take(); nat && qkeep && quantisable && nat->shape == t.shape) { w.add_raw(n, t.shape, nat->codec, nat->blocks, align); ++rep.tensors; ++qck.n_native; return; }
        std::string c = "f32";
        if (quantisable) {
            if (opt.mixed()) {
                int il = recipe_layer(n);
                c = recipe_type(rc, n, t.shape.empty() ? 1 : t.shape.back(), in_deltanet, il >= 0 ? v_index[std::size_t(il)] : -1);
                if (c == "q4_k_m" || c == "q5_k_m") c = mix_base(c);
            } else c = opt.codec;
        }
        w.add_tensor(n, t.shape, t.data, c, align); ++rep.tensors;
    };
    // routed experts of one layer, stacked into [n_experts, rows, k] per projection (moe_ffn's layout), each tensor
    // starting on a 4 KiB file offset and listed in the "experts" section
    auto experts = [&](i64 l, const std::string& Pfmt, const std::string& Q, const char* g, const char* u, const char* d) {
        i64 ne = i64(bp.param("n_experts"));
        const char* src[3] = {g, u, d}; const char* proj[3] = {"gate", "up", "down"};
        for (int j = 0; j < 3; ++j) {
            tensor st;
            std::vector<std::uint8_t> raw; std::string raw_codec; bool all_raw = qkeep;   // AWQ / GPTQ experts: stack the grouped blocks
            for (i64 e = 0; e < ne; ++e) {
                auto t = find(std::vformat(Pfmt, std::make_format_args(e)) + src[j] + ".weight");
                if (auto nat = qck.take(); nat && all_raw && (raw_codec.empty() || raw_codec == nat->codec)) { raw_codec = nat->codec; raw.insert(raw.end(), nat->blocks.begin(), nat->blocks.end()); ++qck.n_native; }
                else all_raw = false;
                if (t.shape.size() != 2) raise(errc::shape_mismatch, "expert weight {} has shape {}", src[j], shape_str(t.shape));
                if (e == 0) { st = tensor(shape_t{ne, t.shape[0], t.shape[1]}); }
                else if (t.shape[0] != st.shape[1] || t.shape[1] != st.shape[2]) raise(errc::shape_mismatch, "experts of layer {} differ in shape", l);
                std::copy(t.data.begin(), t.data.end(), st.data.begin() + e * t.shape[0] * t.shape[1]);
            }
            if (all_raw && !raw.empty()) { w.add_raw(Q + "experts.w_" + proj[j], st.shape, raw_codec, raw, modelfile::expert_alignment); ++rep.tensors; }
            else put(Q + "experts.w_" + proj[j], st, true, modelfile::expert_alignment);
            xi.add(w.last(), l, proj[j]);
        }
    };
    {   // quantised embeddings: the runtime gathers rows straight from q8_0/q4_k blocks
        std::string ec = opt.emb_codec.empty() ? (opt.codec == "f32" || opt.codec == "f16" || opt.codec == "bf16" ? opt.codec : std::string("q8_0")) : opt.emb_codec;
        auto t = find(mp + "embed_tokens.weight");
        if (opt.mixed() && opt.emb_codec.empty()) ec = recipe_type(rc, "tok_emb", t.shape.back());
        w.add_tensor("tok_emb", t.shape, t.data, ec); ++rep.tensors;
    }
    auto mlp = [&](const std::string& P, const std::string& Q, const char* g, const char* u, const char* d) {
        put(Q + "w_gate", find(P + g + ".weight"), true);
        put(Q + "w_up", find(P + u + ".weight"), true);
        put(Q + "w_down", find(P + d + ".weight"), true);
    };
    bool q35 = tname == "qwen3_5";
    auto plus1 = [&](tensor t) { if (q35) for (auto& x : t.data) x += 1.f; return t; };   // Qwen3.5 RMSNorm: x * (1 + w)
    for (i64 l = 0; l < L; ++l) {
        auto P = std::format("{}layers.{}.", mp, l); auto Q = std::format("layer.{}.", l);
        in_deltanet = false;
        put(Q + "attn_norm", plus1(find(P + "input_layernorm.weight")), false);
        put(Q + "ffn_norm", plus1(find(P + "post_attention_layernorm.weight")), false);
        if (q35 && has(P + "linear_attn.in_proj_qkv.weight")) {
            auto A = P + "linear_attn.";
            in_deltanet = true;
            put(Q + "w_qkv", find(A + "in_proj_qkv.weight"), true);
            put(Q + "w_z", find(A + "in_proj_z.weight"), true);
            put(Q + "w_a", find(A + "in_proj_a.weight"), false);
            put(Q + "w_b", find(A + "in_proj_b.weight"), false);
            auto cw = find(A + "conv1d.weight");          // [C, 1, K] -> [C, K]
            if (cw.shape.size() == 3) cw.shape = {cw.shape[0], cw.shape[2]};
            put(Q + "conv_w", cw, false);
            put(Q + "dt_bias", find(A + "dt_bias"), false);
            auto an = find(A + "A_log"); for (auto& x : an.data) x = -std::exp(x);   // g = -exp(A_log) * softplus(..)
            put(Q + "a_neg", an, false);
            put(Q + "ssm_norm", find(A + "norm.weight"), false);   // gated norm: plain weight
            put(Q + "wo", find(A + "out_proj.weight"), true);
            mlp(P + "mlp.", Q, "gate_proj", "up_proj", "down_proj");
            continue;
        }
        put(Q + "wo", find(P + "self_attn.o_proj.weight"), true);
        if (tname == "deepseek_v2") {
            if (has(P + "self_attn.q_a_proj.weight")) {
                put(Q + "wq_a", find(P + "self_attn.q_a_proj.weight"), true);
                put(Q + "q_norm", find(P + "self_attn.q_a_layernorm.weight"), false);
                put(Q + "wq_b", find(P + "self_attn.q_b_proj.weight"), true);
            } else put(Q + "wq", find(P + "self_attn.q_proj.weight"), true);
            put(Q + "wkv_a", find(P + "self_attn.kv_a_proj_with_mqa.weight"), true);
            put(Q + "kv_norm", find(P + "self_attn.kv_a_layernorm.weight"), false);
            {   // split kv_b [h*(nope+v), r] into absorbed per-head W_uk [h, nope, r] and W_uv [h, r, v]
                auto kvb = find(P + "self_attn.kv_b_proj.weight");
                i64 h = i64(bp.param("n_heads")), nope = i64(bp.param("nope")), vd = i64(bp.param("v_dim")), r = i64(bp.param("kv_lora"));
                if (kvb.shape != shape_t{h * (nope + vd), r}) raise(errc::shape_mismatch, "kv_b_proj has shape {}", shape_str(kvb.shape));
                tensor uk(shape_t{h, nope, r}), uv(shape_t{h, r, vd});
                for (i64 hh = 0; hh < h; ++hh) for (i64 j = 0; j < nope + vd; ++j) for (i64 c = 0; c < r; ++c) {
                    float x = kvb.data[std::size_t((hh * (nope + vd) + j) * r + c)];
                    if (j < nope) uk.data[std::size_t((hh * nope + j) * r + c)] = x;
                    else uv.data[std::size_t((hh * r + c) * vd + (j - nope))] = x;
                }
                put(Q + "w_uk", uk, false); put(Q + "w_uv", uv, false);
            }
            if (l < i64(bp.param("first_dense"))) mlp(P + "mlp.", Q, "gate_proj", "up_proj", "down_proj");
            else {
                put(Q + "router", find(P + "mlp.gate.weight"), false);
                mlp(P + "mlp.shared_experts.", Q + "shared.", "gate_proj", "up_proj", "down_proj");
                experts(l, P + "mlp.experts.{}.", Q, "gate_proj", "up_proj", "down_proj");
            }
            continue;
        }
        put(Q + "wq", find(P + "self_attn.q_proj.weight"), true);
        put(Q + "wk", find(P + "self_attn.k_proj.weight"), true);
        put(Q + "wv", find(P + "self_attn.v_proj.weight"), true);
        if (tname == "qwen2") {   // Qwen2 / Qwen2.5: Q, K and V projections carry a bias (o_proj and the MLP do not)
            put(Q + "bq", find(P + "self_attn.q_proj.bias"), false);
            put(Q + "bk", find(P + "self_attn.k_proj.bias"), false);
            put(Q + "bv", find(P + "self_attn.v_proj.bias"), false);
        }
        if (tname == "qwen3" || tname == "qwen3_moe" || q35) {
            put(Q + "q_norm", plus1(find(P + "self_attn.q_norm.weight")), false);
            put(Q + "k_norm", plus1(find(P + "self_attn.k_norm.weight")), false);
        }
        if (tname == "mixtral") {
            put(Q + "router", find(P + "block_sparse_moe.gate.weight"), false);
            experts(l, P + "block_sparse_moe.experts.{}.", Q, "w1", "w3", "w2");
        } else if (tname == "qwen3_moe") {
            put(Q + "router", find(P + "mlp.gate.weight"), false);
            experts(l, P + "mlp.experts.{}.", Q, "gate_proj", "up_proj", "down_proj");
        } else mlp(P + "mlp.", Q, "gate_proj", "up_proj", "down_proj");
    }
    put("out_norm", plus1(find(mp + "norm.weight")), false);
    put("lm_head", has("lm_head.weight") ? find("lm_head.weight") : find(mp + "embed_tokens.weight"), true);

    // Every source tensor is accounted for: consumed, part of a skipped tower, a known buffer, or reported.
    for (auto& s : shards)
        for (auto& [n, e] : s->entries())
            if (!consumed.count(n) && !skipped(n) && !ignored_tensor(n, opt)) rep.unused.push_back(n + " " + shape_str(e.shape));
    check_unused(dir.string(), tname, rep.unused, opt);
    check_template_tensors(dir.string(), tname, bp, w.entries(), opt);

    json::object man{{"format", "fpk"}, {"forge_format_version", double(modelfile::format_version)}, {"architecture", arch},
                     {"template", tname}, {"codec", opt.codec}, {"session", bp.session},
                     {"bos_token_id", cfg.num_or("bos_token_id", -1)}, {"eos_token_id", cfg.num_or("eos_token_id", -1)},
                     {"max_position_embeddings", cfg.num_or("max_position_embeddings", 2048)}};
    // tokenizer_config.json's add_bos_token decides whether prompts start with BOS (absent: the old default, BOS added).
    if (std::filesystem::exists(dir / "tokenizer_config.json")) {
        auto tc = json::parse(read_file(dir / "tokenizer_config.json"));
        if (auto* ab = tc.find("add_bos_token"); ab && ab->is_bool()) man.emplace_back("add_bos", ab->b());
    }
    if (!xi.empty()) { man.emplace_back("forge_format_minor", double(modelfile::format_minor)); w.set_section("experts", xi.dump()); }
    auto lic = hf_license(dir);   // README.md front matter + LICENSE* files: manifest fields and the `license` section
    lic.store(man, w); rep.license = lic.summary();
    if (qscheme) {   // what the checkpoint was, and what became of its quantised Linears
        man.emplace_back("source_quantisation", json::object{{"method", qscheme->method}, {"bits", double(qscheme->bits)}, {"group_size", double(qscheme->group_size)},
                                                             {"linears", double(qck.n_linear)}, {"stored_as_grouped_codec", double(qck.n_native)},
                                                             {"mode", std::string(qkeep ? "keep" : "requantised")}});
        if (qkeep) rep.notes.push_back(std::format("{} checkpoint ({} bits, group {}): {} of {} quantised Linear layers stored as q{}_g32 blocks, the rest decoded to f16",
                                                   qscheme->method, qscheme->bits, qscheme->group_size, qck.n_native, qck.n_linear, qscheme->bits));
        else rep.notes.push_back(std::format("warning: {} quantised Linear layers of this {} checkpoint were decoded and encoded again as {}{}", qck.n_linear, qscheme->method, opt.codec,
                                             opt.codec == "f32" ? "" : " (double quantisation: the error of both formats adds up)"));
    }
    w.set_manifest(json::value(man));
    w.set_section("blueprint", bp_text);
    if (std::filesystem::exists(dir / "tokenizer.json")) w.set_section("tokenizer", read_file(dir / "tokenizer.json"));
    w.finish();
    rep.bytes = std::filesystem::file_size(out);
    return rep;
}

} // namespace forge::import
