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
    auto bp_text = instantiate(read_file(template_dir() / (tname + ".fbp")), json::value(cfg));
    auto bp = modelfile::parse_blueprint(bp_text);

    modelfile::stream_writer w(out);
    import_report rep{arch, tname, 0, 0};
    std::set<std::string> used;
    auto put = [&](const std::string& dst, const std::string& src) {
        auto* t = g.find(src);
        if (!t) raise(errc::not_found, "{}: missing tensor {}", path, src);
        used.insert(src);
        w.add_raw(dst, t->shape, gguf_file::codec_of(t->type) ? gguf_file::codec_of(t->type) : "?", g.bytes(*t)); ++rep.tensors;
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
    for (auto& t : g.tensors())   // everything else stays addressable, as in import_gguf
        if (!used.count(t.name) && gguf_file::codec_of(t.type)) { w.add_raw("extra." + t.name, t.shape, gguf_file::codec_of(t.type), g.bytes(t)); ++rep.tensors; }

    // byte-level BPE vocabularies (tokenizer.ggml.model "gpt2") become a tokenizer section; SentencePiece / WordPiece
    // vocabularies are not converted, so such files take token ids
    if (auto* tm = g.get("tokenizer.ggml.model"); tm && tm->is_str() && tm->str() == "gpt2" && g.get("tokenizer.ggml.tokens")) {
        const auto& toks = g.need("tokenizer.ggml.tokens").arr();
        json::object vocab; json::array added;
        const json::array* types = g.get("tokenizer.ggml.token_type") ? &g.need("tokenizer.ggml.token_type").arr() : nullptr;
        for (std::size_t i = 0; i < toks.size(); ++i) {
            vocab.emplace_back(toks[i].str(), double(i));
            if (types && i < types->size() && (*types)[i].num() == 3) added.push_back(json::object{{"id", double(i)}, {"content", toks[i].str()}});
        }
        json::array merges; if (auto* m = g.get("tokenizer.ggml.merges")) merges = m->arr();
        w.set_section("tokenizer", json::dump(json::value(json::object{{"added_tokens", added}, {"model", json::object{{"type", "BPE"}, {"vocab", vocab}, {"merges", merges}}}})));
    }
    auto num_or = [&](const char* k, double d) { auto* v = g.get(k); return v && v->is_num() ? v->num() : d; };
    json::object man{{"format", "fpk"}, {"forge_format_version", double(modelfile::format_version)}, {"architecture", arch},
                     {"source", "gguf"}, {"template", tname}, {"session", bp.session},
                     {"bos_token_id", num_or("tokenizer.ggml.bos_token_id", -1)}, {"eos_token_id", num_or("tokenizer.ggml.eos_token_id", -1)},
                     {"max_position_embeddings", Kor("context_length", 2048)}};
    if (auto* ab = g.get("tokenizer.ggml.add_bos_token"); ab && ab->is_bool()) man.emplace_back("add_bos", ab->b());
    else for (auto& kv : extra) if (kv.first == "add_bos") man.push_back(kv);
    for (auto& kv : extra) if (kv.first != "add_bos") man.push_back(kv);
    w.set_manifest(json::value(man));
    w.set_section("blueprint", bp_text);
    w.finish();
    rep.bytes = std::filesystem::file_size(out);
    return rep;
}

} // namespace forge::import
