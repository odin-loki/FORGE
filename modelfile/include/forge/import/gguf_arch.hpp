#pragma once
// GGUF importers for the architectures of hf_arch.hpp that llama.cpp also defines: gemma, gemma2, gemma3, phi3, gpt2
// and bert (general.architecture). Included at the end of gguf.hpp; import_gguf() calls import_gguf_extra() first and
// continues with its own Llama-style path when this returns nothing. Tensors are copied verbatim (no re-quantisation):
// the llama.cpp converter already stores Gemma's RMSNorm weights as 1 + w and GPT-2's Conv1D weights transposed, which
// is the layout the templates expect.
// Not stored in a GGUF, so taken from llama.cpp's own constants: Gemma-2/3 attention scale (1 / sqrt(head_dim), or
// 1 / sqrt(d_model / heads) for the 27B sizes), Gemma-3's local RoPE base (10000) and sliding-window period (6).
#include <cmath>
#include <optional>

namespace forge::import {

inline std::optional<import_report> import_gguf_extra(const gguf_file& g, const std::string& path, const std::string& out, const import_options& opt, const std::string& arch) {
    const bool gemma = arch == "gemma" || arch == "gemma2" || arch == "gemma3";
    if (!gemma && arch != "phi3" && arch != "gpt2" && arch != "bert") return std::nullopt;
    auto K = [&](const std::string& k) { return g.need(arch + "." + k).num(); };
    auto Kor = [&](const std::string& k, double d) { auto* v = g.get(arch + "." + k); return v && !v->is_arr() ? v->num() : d; };
    auto shape_of = [&](const std::string& n) -> const shape_t& {
        auto* t = g.find(n); if (!t) raise(errc::not_found, "{}: missing tensor {}", path, n);
        return t->shape;
    };
    double hidden = K("embedding_length"), heads = K("attention.head_count"), layers = K("block_count");
    if (hidden <= 0 || heads <= 0 || layers <= 0) raise(errc::format_error, "{}: bad {} dimensions", path, arch);
    double hd = Kor("attention.key_length", hidden / heads);
    json::object cfg{{"hidden_size", hidden}, {"num_hidden_layers", layers}, {"num_attention_heads", heads},
                     {"num_key_value_heads", Kor("attention.head_count_kv", heads)}, {"head_dim", hd},
                     {"intermediate_size", Kor("feed_forward_length", 4 * hidden)}, {"vocab_size", double(shape_of("token_embd.weight").at(0))},
                     {"rms_norm_eps", Kor("attention.layer_norm_rms_epsilon", 1e-6)}, {"layer_norm_eps", Kor("attention.layer_norm_epsilon", 1e-5)},
                     {"rope_theta", Kor("rope.freq_base", arch == "gemma3" ? 1000000.0 : 10000.0)}};
    std::string tname = arch;
    json::object extra;
    if (gemma) {
        cfg.emplace_back("act", std::string("gelu_tanh"));
        if (arch != "gemma") {
            bool big = arch == "gemma2" ? layers == 46 : layers == 62;   // the 27B sizes scale by d_model / heads (llama.cpp)
            double fc = arch == "gemma2" ? Kor("final_logit_softcapping", 30.0) : 0.0;
            cfg.emplace_back("attn_scale", 1.0 / std::sqrt(big ? hidden / heads : hd));
            cfg.emplace_back("attn_softcap", arch == "gemma2" ? Kor("attn_logit_softcapping", 50.0) : 0.0);
            cfg.emplace_back("final_softcap", fc);
            cfg.emplace_back("logits_expr", std::string(fc > 0 ? "tanh(lg / final_softcap) * final_softcap" : "lg"));
            cfg.emplace_back("sliding_window", Kor("attention.sliding_window", arch == "gemma2" ? 4096.0 : 1024.0));
            cfg.emplace_back("sliding_window_pattern", arch == "gemma2" ? 2.0 : Kor("attention.sliding_window_pattern", 6.0));
            cfg.emplace_back("rope_local_base_freq", 10000.0);
        }
    } else if (arch == "phi3") {
        if (g.find("rope_factors_long.weight") || g.find("rope_factors_short.weight")) raise(errc::unsupported, "{}: phi3 LongRoPE factors are not supported", path);
        double rot = Kor("rope.dimension_count", hd);
        if (rot < 2 || rot > hd || std::fmod(rot, 2) != 0) raise(errc::unsupported, "{}: phi3 rotary width {} of head_dim {}", path, rot, hd);
        cfg.emplace_back("rotary_dim", rot); cfg.emplace_back("rope_fn", std::string(rot < hd ? "rope_part" : "rope_full"));
        for (auto& kv : cfg) if (kv.first == "rms_norm_eps") kv.second = Kor("attention.layer_norm_rms_epsilon", 1e-5);
    } else if (arch == "gpt2") {
        cfg.emplace_back("act", std::string("gelu_tanh"));
        extra.emplace_back("add_bos", false);
    } else {   // bert
        double pt = Kor("pooling_type", 2);
        if (pt != 1 && pt != 2) raise(errc::unsupported, "{}: bert pooling_type {} (only mean = 1 and cls = 2 are supported)", path, pt);
        cfg.emplace_back("act", std::string("gelu")); cfg.emplace_back("position_offset", 0.0);   // the converter drops RoBERTa's offset rows
        cfg.emplace_back("pool_expr", std::string(pt == 1 ? "masked_mean(x, mask)" : "reshape(slice(x, 0, 0, 1), [d_model])"));
        for (auto& kv : cfg) if (kv.first == "layer_norm_eps") kv.second = Kor("attention.layer_norm_epsilon", 1e-12);
        extra.emplace_back("pooling", std::string(pt == 1 ? "mean" : "cls"));
    }
    if (!opt.template_name.empty()) tname = opt.template_name;
    auto bp_text = instantiate(read_file(template_path(tname)), json::value(cfg));
    auto bp = modelfile::parse_blueprint(bp_text);

    modelfile::stream_writer w(out);
    import_report rep{arch, tname, 0, 0};
    std::set<std::string> used;
    auto put = [&](const std::string& dst, const std::string& src) {
        auto* t = g.find(src);
        if (!t) raise(errc::not_found, "{}: missing tensor {}", path, src);
        used.insert(src);
        if (!gguf_file::loadable(t->type)) raise(errc::unsupported, "{}: tensor {} has ggml type {}, which this importer cannot copy", path, src, gguf_file::type_name(t->type));
        w.add_raw(dst, t->shape, gguf_file::codec_of(t->type), g.bytes(*t)); ++rep.tensors;
    };
    auto pair = [&](const std::string& dw, const std::string& db, const std::string& src) { put(dw, src + ".weight"); put(db, src + ".bias"); };
    put("tok_emb", "token_embd.weight");
    const i64 L = i64(bp.param("n_layers"));
    for (i64 l = 0; l < L; ++l) {
        auto P = std::format("blk.{}.", l), Q = std::format("layer.{}.", l);
        if (gemma) {
            put(Q + "attn_norm", P + "attn_norm.weight"); put(Q + "ffn_norm", P + "ffn_norm.weight");
            if (arch != "gemma") { put(Q + "attn_post_norm", P + "post_attention_norm.weight"); put(Q + "ffn_post_norm", P + "post_ffw_norm.weight"); }
            put(Q + "wq", P + "attn_q.weight"); put(Q + "wk", P + "attn_k.weight"); put(Q + "wv", P + "attn_v.weight"); put(Q + "wo", P + "attn_output.weight");
            if (arch == "gemma3") { put(Q + "q_norm", P + "attn_q_norm.weight"); put(Q + "k_norm", P + "attn_k_norm.weight"); }
            put(Q + "w_gate", P + "ffn_gate.weight"); put(Q + "w_up", P + "ffn_up.weight"); put(Q + "w_down", P + "ffn_down.weight");
        } else if (arch == "phi3") {
            put(Q + "attn_norm", P + "attn_norm.weight"); put(Q + "ffn_norm", P + "ffn_norm.weight");
            put(Q + "w_qkv", P + "attn_qkv.weight"); put(Q + "wo", P + "attn_output.weight");
            put(Q + "w_gate_up", P + "ffn_up.weight"); put(Q + "w_down", P + "ffn_down.weight");
        } else if (arch == "gpt2") {
            pair(Q + "ln1_w", Q + "ln1_b", P + "attn_norm"); pair(Q + "ln2_w", Q + "ln2_b", P + "ffn_norm");
            pair(Q + "w_qkv", Q + "b_qkv", P + "attn_qkv"); pair(Q + "wo", Q + "bo", P + "attn_output");
            pair(Q + "w_fc", Q + "b_fc", P + "ffn_up"); pair(Q + "w_proj", Q + "b_proj", P + "ffn_down");
        } else {
            pair(Q + "wq", Q + "bq", P + "attn_q"); pair(Q + "wk", Q + "bk", P + "attn_k"); pair(Q + "wv", Q + "bv", P + "attn_v");
            pair(Q + "wo", Q + "bo", P + "attn_output"); pair(Q + "attn_norm_w", Q + "attn_norm_b", P + "attn_output_norm");
            pair(Q + "w_in", Q + "b_in", P + "ffn_up"); pair(Q + "w_out", Q + "b_out", P + "ffn_down");
            pair(Q + "ffn_norm_w", Q + "ffn_norm_b", P + "layer_output_norm");
        }
    }
    if (arch == "gpt2") { put("pos_emb", "position_embd.weight"); pair("out_norm_w", "out_norm_b", "output_norm"); }
    else if (arch == "bert") { put("pos_emb", "position_embd.weight"); put("type_emb", "token_types.weight"); pair("emb_norm_w", "emb_norm_b", "token_embd_norm"); }
    else put("out_norm", "output_norm.weight");
    if (arch != "bert") put("lm_head", g.find("output.weight") ? "output.weight" : "token_embd.weight");
    // Every source tensor is accounted for, as in import_gguf: consumed, a known buffer, or reported (an error unless
    // --allow-unused, which keeps loadable leftovers as extra.<name>).
    for (auto& t : g.tensors())
        if (!used.count(t.name) && !ignored_tensor(t.name, opt))
            rep.unused.push_back(t.name + " " + shape_str(t.shape) + (gguf_file::loadable(t.type) ? "" : " (" + gguf_file::type_name(t.type) + ": type not supported)"));
    check_unused(path, tname, rep.unused, opt);
    for (auto& t : g.tensors())
        if (!used.count(t.name) && !ignored_tensor(t.name, opt)) {
            if (gguf_file::loadable(t.type)) { w.add_raw("extra." + t.name, t.shape, gguf_file::codec_of(t.type), g.bytes(t)); ++rep.tensors; }
            else rep.notes.push_back(std::format("left out unused tensor {}: ggml type {} is not supported", t.name, gguf_file::type_name(t.type)));
        }
    check_template_tensors(path, tname, bp, w.entries(), opt);
    if (!opt.codec.empty() && opt.codec != "keep" && opt.codec != "f32") rep.notes.push_back(std::format("--codec {} is not applied to {} files: tensors keep the type the file uses", opt.codec, arch));

    // tokenizer (tokenizer.ggml.model gpt2 / llama / t5 / bert) and chat template, as import_gguf stores them
    store_gguf_tokenizer(g, w, [&](const std::string& m) { rep.notes.push_back("warning: " + m); });
    auto num_or = [&](const char* k, double d) { auto* v = g.get(k); return v && v->is_num() ? v->num() : d; };
    json::object man{{"format", "fpk"}, {"forge_format_version", double(modelfile::format_version)}, {"architecture", arch},
                     {"source", "gguf"}, {"template", tname}, {"session", bp.session},
                     {"bos_token_id", num_or("tokenizer.ggml.bos_token_id", -1)}, {"eos_token_id", num_or("tokenizer.ggml.eos_token_id", -1)},
                     {"max_position_embeddings", Kor("context_length", 2048)}};
    // BERT: [CLS] ... [SEP] come from the tokenizer's post-processor, not from add_bos / add_eos
    if (auto* ab = g.get("tokenizer.ggml.add_bos_token"); arch != "bert" && ab && ab->is_bool()) man.emplace_back("add_bos", ab->b());
    else for (auto& kv : extra) if (kv.first == "add_bos") man.push_back(kv);
    if (auto* ae = g.get("tokenizer.ggml.add_eos_token"); arch != "bert" && ae && ae->is_bool()) man.emplace_back("add_eos", ae->b());
    for (auto& kv : extra) if (kv.first != "add_bos") man.push_back(kv);
    if (arch != "bert") gguf_eos_ids(g, man);
    {   // licence, as import_gguf records it
        license_info lic;
        auto gs = [&](const char* k) { auto* v = g.get(k); return v && v->is_str() ? v->str() : std::string(); };
        lic.id = gs("general.license"); lic.name = gs("general.license.name"); lic.link = gs("general.license.link");
        lic.store(man, w); rep.license = lic.summary();
    }
    w.set_manifest(json::value(man));
    w.set_section("blueprint", bp_text);
    w.finish();
    rep.bytes = std::filesystem::file_size(out);
    return rep;
}

} // namespace forge::import
