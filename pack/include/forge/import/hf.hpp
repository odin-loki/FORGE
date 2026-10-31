#pragma once
// Hugging Face directory (config.json + *.safetensors + tokenizer.json) -> .fpk
#include <forge/import/safetensors.hpp>
#include <forge/pack/blueprint.hpp>
#include <forge/pack/fpk.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace forge::import {

inline std::string read_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) raise(errc::io_error, "cannot read {}", p.string());
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

#ifndef FORGE_SOURCE_DIR
#define FORGE_SOURCE_DIR "."
#endif
inline std::filesystem::path template_dir() {
    if (auto* e = std::getenv("FORGE_TEMPLATES")) return e;
    return std::filesystem::path(FORGE_SOURCE_DIR) / "import" / "templates";
}

// Substitute {{key}} with config values (numbers formatted losslessly).
inline std::string instantiate(std::string tpl, const json::value& cfg) {
    std::size_t p;
    while ((p = tpl.find("{{")) != std::string::npos) {
        auto e = tpl.find("}}", p);
        auto key = tpl.substr(p + 2, e - p - 2);
        auto* v = cfg.find(key);
        if (!v) raise(errc::not_found, "template needs config key '{}'", key);
        std::string s;
        if (v->is_num()) { char b[40]; std::snprintf(b, 40, "%.17g", v->num()); s = b; } else s = v->str();
        tpl.replace(p, e - p + 2, s);
    }
    return tpl;
}

struct import_options { std::string codec = "f32"; std::string template_name; };

struct import_report { std::string arch, template_name; std::size_t tensors = 0, bytes = 0; };

inline import_report import_hf(const std::filesystem::path& dir, const std::string& out, const import_options& opt = {}) {
    auto cfg = json::parse(read_file(dir / "config.json"));
    std::string arch = cfg.find("architectures") ? cfg["architectures"][0].str() : cfg.str_or("model_type", "unknown");
    std::string tname = opt.template_name;
    if (tname.empty()) {
        if (arch == "LlamaForCausalLM" || arch == "MistralForCausalLM" || arch == "Qwen2ForCausalLM" || cfg.str_or("model_type", "") == "llama")
            tname = "llama";
        else if (arch == "Qwen3ForCausalLM" || cfg.str_or("model_type", "") == "qwen3") tname = "qwen3";
        else if (arch == "Qwen3_5ForConditionalGeneration" || arch == "Qwen3_5ForCausalLM" || cfg.str_or("model_type", "") == "qwen3_5")
            raise(errc::unsupported, "qwen3_5 (Gated DeltaNet hybrid attention) has no blueprint template yet");
        else if (arch == "MixtralForCausalLM" || cfg.str_or("model_type", "") == "mixtral") tname = "mixtral";
        else if (arch == "DeepseekV2ForCausalLM" || arch == "DeepseekV3ForCausalLM" || cfg.str_or("model_type", "") == "deepseek_v2") tname = "deepseek_v2";
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
    set("rope_theta", num("rope_theta", 10000.0));
    set("rms_norm_eps", num("rms_norm_eps", 1e-6));
    set("rope_neox", 1.0);  // HF checkpoints use rotate_half (NeoX) RoPE
    if (tname == "deepseek_v2") {
        double nope = num("qk_nope_head_dim", 128), rd = num("qk_rope_head_dim", 64);
        set("qk_head_dim", nope + rd);
        set("attn_scale", 1.0 / std::sqrt(nope + rd));
        if (cfg.find("rope_scaling") && cfg["rope_scaling"].is_obj())
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
    json::value cv(c);
    auto bp_text = instantiate(read_file(template_dir() / (tname + ".fbp")), cv);
    auto bp = pack::parse_blueprint(bp_text);  // validate before writing anything

    // gather safetensors shards
    std::vector<std::unique_ptr<safetensors>> shards;
    for (auto& f : std::filesystem::directory_iterator(dir))
        if (f.path().extension() == ".safetensors") shards.push_back(std::make_unique<safetensors>(f.path().string()));
    if (shards.empty()) raise(errc::not_found, "{}: no .safetensors files", dir.string());
    auto find = [&](const std::string& n) -> tensor {
        for (auto& s : shards) if (s->has(n)) return s->load(n);
        raise(errc::not_found, "missing tensor {}", n);
    };
    auto has = [&](const std::string& n) { for (auto& s : shards) if (s->has(n)) return true; return false; };

    pack::writer w;
    import_report rep{arch, tname, 0, 0};
    auto put = [&](const std::string& n, const tensor& t, bool quantisable) {
        w.add_tensor(n, t.shape, t.data, quantisable ? opt.codec : std::string("f32")); ++rep.tensors;
    };
    i64 L = i64(bp.param("n_layers"));
    auto emb = find("model.embed_tokens.weight");
    put("tok_emb", emb, false);
    auto mlp = [&](const std::string& P, const std::string& Q, const char* g, const char* u, const char* d) {
        put(Q + "w_gate", find(P + g + ".weight"), true);
        put(Q + "w_up", find(P + u + ".weight"), true);
        put(Q + "w_down", find(P + d + ".weight"), true);
    };
    for (i64 l = 0; l < L; ++l) {
        auto P = std::format("model.layers.{}.", l); auto Q = std::format("layer.{}.", l);
        put(Q + "attn_norm", find(P + "input_layernorm.weight"), false);
        put(Q + "ffn_norm", find(P + "post_attention_layernorm.weight"), false);
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
                for (i64 e = 0; e < i64(bp.param("n_experts")); ++e)
                    mlp(std::format("{}mlp.experts.{}.", P, e), std::format("{}expert.{}.", Q, e), "gate_proj", "up_proj", "down_proj");
            }
            continue;
        }
        put(Q + "wq", find(P + "self_attn.q_proj.weight"), true);
        put(Q + "wk", find(P + "self_attn.k_proj.weight"), true);
        put(Q + "wv", find(P + "self_attn.v_proj.weight"), true);
        if (tname == "qwen3") {
            put(Q + "q_norm", find(P + "self_attn.q_norm.weight"), false);
            put(Q + "k_norm", find(P + "self_attn.k_norm.weight"), false);
        }
        if (tname == "mixtral") {
            put(Q + "router", find(P + "block_sparse_moe.gate.weight"), false);
            for (i64 e = 0; e < i64(bp.param("n_experts")); ++e)
                mlp(std::format("{}block_sparse_moe.experts.{}.", P, e), std::format("{}expert.{}.", Q, e), "w1", "w3", "w2");
        } else mlp(P + "mlp.", Q, "gate_proj", "up_proj", "down_proj");
    }
    put("out_norm", find("model.norm.weight"), false);
    put("lm_head", has("lm_head.weight") ? find("lm_head.weight") : emb, true);

    json::object man{{"format", "fpk"}, {"forge_format_version", double(pack::format_version)}, {"architecture", arch},
                     {"template", tname}, {"codec", opt.codec}, {"session", bp.session},
                     {"bos_token_id", cfg.num_or("bos_token_id", -1)}, {"eos_token_id", cfg.num_or("eos_token_id", -1)},
                     {"max_position_embeddings", cfg.num_or("max_position_embeddings", 2048)}};
    w.set_manifest(json::value(man));
    w.set_section("blueprint", bp_text);
    if (std::filesystem::exists(dir / "tokenizer.json")) w.set_section("tokenizer", read_file(dir / "tokenizer.json"));
    w.write(out);
    rep.bytes = std::filesystem::file_size(out);
    return rep;
}

} // namespace forge::import
