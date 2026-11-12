#pragma once
// Hugging Face importers for the architectures beyond the Llama-style decoders of hf.hpp. Included at the end of
// hf.hpp; import_hf() calls import_hf_extra() first and falls through to its own decoder path when this returns
// nothing. One function per family maps config.json onto a blueprint template (import/templates/<name>.fbp) and the
// checkpoint's tensors onto the template's weight names:
//   gpt2                     GPT-2 (learned positions, LayerNorm, Conv1D weights stored [in, out], tied head)
//   gemma / gemma2 / gemma3  Gemma text decoders (RMSNorm 1 + w, sqrt(d) embedding scale, GELU-tanh GeGLU; Gemma-2
//                            pre+post norms, soft-capped attention and logits, alternating sliding window; Gemma-3
//                            QK-norm, sliding-window pattern with its own RoPE base)
//   phi3                     Phi-3 / Phi-4-mini (fused qkv projection, fused gate-up MLP, optional partial RoPE)
//   bert                     BERT / RoBERTa / MiniLM / BGE / E5 encoders as embedding models (CLS or mean pooling)
//   vit                      ViT, CLIP and SigLIP vision encoders as image-embedding models
//   whisper_encoder          Whisper audio encoder (the decoder is not imported)
// 2-D projection weights take --codec; everything else (norms, biases, position tables, conv kernels) stays f32.
#include <cmath>
#include <optional>

namespace forge::import {

namespace arch_detail {

struct importer {
    std::filesystem::path dir; std::string out; const import_options& opt;
    json::value cfg; json::object c;                       // source config, and the config the template is filled from
    std::vector<std::unique_ptr<safetensors>> shards;
    std::string pre;                                       // tensor name prefix of this checkpoint ("", "bert.", ...)
    std::unique_ptr<modelfile::stream_writer> w;
    import_report rep; std::string bp_text; modelfile::blueprint bp;
    // Tensor accounting (hf.hpp check_unused): every source tensor is consumed, a known buffer, or part of something
    // this family leaves out on purpose (`outside`: another tower, a task head, a tied copy); anything else fails the
    // import unless --allow-unused. What `outside` left out is counted in a note.
    mutable std::set<std::string> consumed;
    std::function<bool(const std::string&)> outside; std::string outside_what;

    importer(const std::filesystem::path& d, const json::value& cf, const std::string& o, const import_options& op, const std::string& arch)
        : dir(d), out(o), opt(op), cfg(cf), c(cf.obj()) {
        rep.arch = arch;
        for (auto& f : std::filesystem::directory_iterator(dir))
            if (f.path().extension() == ".safetensors") shards.push_back(std::make_unique<safetensors>(f.path().string()));
        if (shards.empty()) raise(errc::not_found, "{}: no .safetensors files", dir.string());
    }
    double num(const char* k, double d) const { auto* v = cfg.find(k); return v && v->is_num() ? v->num() : d; }
    double need(const char* k) const { auto* v = cfg.find(k); if (!v || !v->is_num()) raise(errc::not_found, "config.json: missing number '{}'", k); return v->num(); }
    std::string str(const char* k, const std::string& d) const { auto* v = cfg.find(k); return v && v->is_str() ? v->str() : d; }
    void set(const std::string& k, json::value v) {
        for (auto& [kk, vv] : c) if (kk == k) { vv = v; return; }
        c.emplace_back(k, v);
    }
    bool raw_has(const std::string& n) const { for (auto& s : shards) if (s->has(n)) return true; return false; }
    bool has(const std::string& n) const { return raw_has(pre + n); }
    tensor raw_get(const std::string& n) const {
        for (auto& s : shards) if (s->has(n)) { consumed.insert(n); return s->load(n); }
        raise(errc::not_found, "missing tensor {}", n);
    }
    // config values nested under `key` (text_config, vision_config) become the config proper
    void descend(const char* key) {
        auto* v = cfg.find(key);
        if (!v || !v->is_obj()) return;
        json::value sub = *v;
        for (auto& kv : sub.obj()) set(kv.first, kv.second);
        cfg = std::move(sub);
    }
    tensor get(const std::string& n) const {
        for (auto& s : shards) if (s->has(pre + n)) { consumed.insert(pre + n); return s->load(pre + n); }
        raise(errc::not_found, "missing tensor {}{}", pre, n);
    }
    // the first prefix under which `probe` exists
    void find_prefix(std::initializer_list<const char*> prefixes, const std::string& probe) {
        for (auto* p : prefixes) if (raw_has(std::string(p) + probe)) { pre = p; return; }
        raise(errc::not_found, "{}: no tensor named {} (tried {} prefixes)", dir.string(), probe, prefixes.size());
    }
    // instantiate + validate the template, then open the output
    void begin(const std::string& tname) {
        rep.template_name = opt.template_name.empty() ? tname : opt.template_name;
        bp_text = instantiate(read_file(template_path(rep.template_name)), json::value(c));
        bp = modelfile::parse_blueprint(bp_text);
        w = std::make_unique<modelfile::stream_writer>(out);
        w->set_alignment(import_alignment(opt, dir));
    }
    std::string qcodec() const { return is_mix_ftype(opt.codec) ? mix_base(opt.codec) : opt.codec; }
    modelfile::calib_table calib;   // import_options::calib: statistics of the f32 weights that get quantised
    void put(const std::string& n, const tensor& t, bool quantisable = false) {
        if (opt.calib && quantisable && t.shape.size() == 2) calib.tensors[n] = modelfile::tensor_stats::of(t.data);
        w->add_tensor(n, t.shape, t.data, quantisable && t.shape.size() == 2 ? qcodec() : std::string("f32")); ++rep.tensors;
    }
    void emb(const std::string& n, const tensor& t) {   // token embedding: q8_0 under a quantised codec, as hf.hpp
        std::string ec = !opt.emb_codec.empty() ? opt.emb_codec : (opt.codec == "f32" || opt.codec == "f16" || opt.codec == "bf16" ? opt.codec : std::string("q8_0"));
        if (opt.calib) calib.tensors[n] = modelfile::tensor_stats::of(t.data);
        w->add_tensor(n, t.shape, t.data, ec); ++rep.tensors;
    }
    // a linear layer: weight (quantisable) + bias, under names <dst>w<sfx> / <dst>b<sfx>
    void linear(const std::string& dst_w, const std::string& dst_b, const std::string& src, bool conv1d_layout = false) {
        auto t = get(src + ".weight");
        put(dst_w, conv1d_layout ? transposed(t) : t, true);
        if (!dst_b.empty()) put(dst_b, get(src + ".bias"));
    }
    void norm(const std::string& dst, const std::string& src) { put(dst + "_w", get(src + ".weight")); put(dst + "_b", get(src + ".bias")); }
    static tensor transposed(const tensor& t) {
        if (t.shape.size() != 2) raise(errc::shape_mismatch, "expected a matrix, got {}", shape_str(t.shape));
        tensor o(shape_t{t.shape[1], t.shape[0]});
        for (i64 i = 0; i < t.shape[0]; ++i) for (i64 j = 0; j < t.shape[1]; ++j) o.data[std::size_t(j * t.shape[0] + i)] = t.data[std::size_t(i * t.shape[1] + j)];
        return o;
    }
    static tensor reshaped(tensor t, shape_t s) {
        if (numel(s) != numel(t.shape)) raise(errc::shape_mismatch, "cannot reshape {} to {}", shape_str(t.shape), shape_str(s));
        t.shape = std::move(s); return t;
    }
    static tensor plus1(tensor t) { for (auto& x : t.data) x += 1.f; return t; }
    static tensor rows(const tensor& t, i64 lo, i64 hi) {   // rows [lo, hi) of a matrix
        if (t.shape.size() != 2 || lo < 0 || hi > t.shape[0] || lo > hi) raise(errc::shape_mismatch, "rows {}..{} of {}", lo, hi, shape_str(t.shape));
        tensor o(shape_t{hi - lo, t.shape[1]}); std::copy_n(t.data.begin() + lo * t.shape[1], (hi - lo) * t.shape[1], o.data.begin()); return o;
    }
    import_report finish(json::object extra = {}) {
        json::object man{{"format", "fpk"}, {"forge_format_version", double(modelfile::format_version)}, {"architecture", rep.arch},
                         {"template", rep.template_name}, {"codec", opt.codec}, {"session", bp.session},
                         {"bos_token_id", cfg.num_or("bos_token_id", -1)}, {"eos_token_id", cfg.num_or("eos_token_id", -1)}};
        for (auto& kv : extra) man.push_back(kv);
        bool have_ctx = false; for (auto& [k, v] : man) have_ctx |= k == "max_position_embeddings";
        if (!have_ctx) man.emplace_back("max_position_embeddings", cfg.num_or("max_position_embeddings", 2048));
        // tokenizer (BPE, SentencePiece, Unigram, WordPiece, WordLevel) and chat template, as import_hf stores them
        store_hf_tokenizer(dir, man, *w, rep);
        std::size_t left_out = 0;
        for (auto& s : shards)
            for (auto& [n, e] : s->entries()) {
                if (consumed.count(n) || ignored_tensor(n, opt)) continue;
                if (outside && outside(n)) { ++left_out; continue; }
                rep.unused.push_back(n + " " + shape_str(e.shape));
            }
        if (left_out) rep.notes.push_back(std::format("left out {} tensor(s) the {} template does not use: {}", left_out, rep.template_name, outside_what));
        check_unused(dir.string(), rep.template_name, rep.unused, opt);
        check_template_tensors(dir.string(), rep.template_name, bp, w->entries(), opt);
        auto lic = hf_license(dir); lic.store(man, *w); rep.license = lic.summary();
        w->set_manifest(json::value(man));
        w->set_section("blueprint", bp_text);
        add_state_section(*w, bp);
        if (!calib.empty()) w->set_section(modelfile::calib_section, calib.dump());
        w->finish();
        rep.bytes = std::filesystem::file_size(out);
        return rep;
    }
};

inline std::string act_op(const std::string& a, const char* what) {
    if (a == "gelu") return "gelu";
    if (a == "gelu_new" || a == "gelu_pytorch_tanh" || a == "gelu_tanh") return "gelu_tanh";
    if (a == "quick_gelu") return "quick_gelu";
    if (a == "silu" || a == "swish") return "silu";
    if (a == "relu") return "relu";
    if (a == "mish") return "mish";
    raise(errc::unsupported, "{}: activation '{}' has no op", what, a);
}

// ---- GPT-2 ---------------------------------------------------------------------------------------------------------
inline import_report import_gpt2(importer& I) {
    double d = I.num("n_embd", I.num("hidden_size", 0)), h = I.num("n_head", I.num("num_attention_heads", 0));
    if (d <= 0 || h <= 0 || std::fmod(d, h) != 0) raise(errc::format_error, "gpt2: bad n_embd {} / n_head {}", d, h);
    I.set("hidden_size", d); I.set("num_hidden_layers", I.num("n_layer", I.num("num_hidden_layers", 0)));
    I.set("num_attention_heads", h); I.set("head_dim", d / h); I.set("vocab_size", I.need("vocab_size"));
    I.set("layer_norm_eps", I.num("layer_norm_epsilon", 1e-5));
    I.set("act", act_op(I.str("activation_function", "gelu_new"), "gpt2"));
    I.find_prefix({"transformer.", ""}, "wte.weight");
    I.outside = [](const std::string& n) { return n.ends_with(".attn.bias") || n == "lm_head.weight"; };   // causal-mask buffers; consumed above when present
    I.outside_what = "causal-mask buffers (h.N.attn.bias)";
    I.begin("gpt2");
    auto wte = I.get("wte.weight");
    I.emb("tok_emb", wte); I.put("pos_emb", I.get("wpe.weight"));
    i64 L = i64(I.bp.param("n_layers"));
    for (i64 l = 0; l < L; ++l) {
        auto P = std::format("h.{}.", l), Q = std::format("layer.{}.", l);
        I.norm(Q + "ln1", P + "ln_1"); I.norm(Q + "ln2", P + "ln_2");
        I.linear(Q + "w_qkv", Q + "b_qkv", P + "attn.c_attn", true);   // Conv1D: [in, out] -> [out, in]
        I.linear(Q + "wo", Q + "bo", P + "attn.c_proj", true);
        I.linear(Q + "w_fc", Q + "b_fc", P + "mlp.c_fc", true);
        I.linear(Q + "w_proj", Q + "b_proj", P + "mlp.c_proj", true);
    }
    I.norm("out_norm", "ln_f");
    I.put("lm_head", I.raw_has("lm_head.weight") ? I.raw_get("lm_head.weight") : wte, true);
    return I.finish({{"max_position_embeddings", I.num("n_positions", I.num("n_ctx", 1024))}, {"add_bos", false}});
}

// ---- Gemma / Gemma-2 / Gemma-3 text --------------------------------------------------------------------------------
inline import_report import_gemma(importer& I, int gen) {
    double heads = I.need("num_attention_heads"), hidden = I.need("hidden_size"), hd = I.num("head_dim", hidden / heads);
    I.set("head_dim", hd); I.set("num_key_value_heads", I.num("num_key_value_heads", heads));
    I.set("rms_norm_eps", I.num("rms_norm_eps", 1e-6)); I.set("rope_theta", I.num("rope_theta", 10000.0));
    auto act = act_op(I.str("hidden_activation", I.str("hidden_act", "gelu_pytorch_tanh")), "gemma");
    I.set("act", act);
    if (gen >= 2) {
        double win = I.num("sliding_window", 4096), pat = gen == 2 ? 2 : I.num("sliding_window_pattern", 6);
        if (const auto* lt = I.cfg.find("layer_types"); gen == 3 && lt && lt->is_arr() && !lt->arr().empty()) {
            // newer configs list the layer kinds: recover the period and check the list follows it
            const auto& a = lt->arr(); std::size_t p = 0;
            for (std::size_t i = 0; i < a.size() && !p; ++i) if (a[i].str() == "full_attention") p = i + 1;
            if (!p) { p = a.size() + 1; }   // no global layer at all: every layer slides
            for (std::size_t i = 0; i < a.size(); ++i)
                if ((a[i].str() == "full_attention") != ((i + 1) % p == 0)) raise(errc::unsupported, "gemma3: layer_types[{}] = {} breaks the every-{}th-layer-global pattern", i, a[i].str(), p);
            pat = double(p);
        }
        I.set("sliding_window", win); I.set("sliding_window_pattern", pat);
        I.set("attn_scale", 1.0 / std::sqrt(I.num("query_pre_attn_scalar", hd)));
        I.set("attn_softcap", I.num("attn_logit_softcapping", 0));
        double fc = I.num("final_logit_softcapping", 0);
        I.set("final_softcap", fc);
        I.set("logits_expr", std::string(fc > 0 ? "tanh(lg / final_softcap) * final_softcap" : "lg"));
        if (gen == 3) I.set("rope_local_base_freq", I.num("rope_local_base_freq", 10000.0));
    }
    I.find_prefix({"model.", "model.language_model.", "language_model.model."}, "embed_tokens.weight");
    I.outside = [](const std::string& n) {
        for (const char* p : {"vision_tower.", "model.vision_tower.", "multi_modal_projector.", "model.multi_modal_projector."}) if (n.starts_with(p)) return true;
        return n.ends_with("lm_head.weight");   // tied to the embedding, which is what the head is written from
    };
    I.outside_what = "the tied lm_head copy, and the vision tower / projector of multimodal checkpoints";
    I.begin(gen == 1 ? "gemma" : gen == 2 ? "gemma2" : "gemma3");
    auto te = I.get("embed_tokens.weight");
    I.emb("tok_emb", te);
    i64 L = i64(I.bp.param("n_layers"));
    for (i64 l = 0; l < L; ++l) {
        auto P = std::format("layers.{}.", l), Q = std::format("layer.{}.", l);
        I.put(Q + "attn_norm", importer::plus1(I.get(P + "input_layernorm.weight")));
        if (gen == 1) I.put(Q + "ffn_norm", importer::plus1(I.get(P + "post_attention_layernorm.weight")));
        else {
            I.put(Q + "attn_post_norm", importer::plus1(I.get(P + "post_attention_layernorm.weight")));
            I.put(Q + "ffn_norm", importer::plus1(I.get(P + "pre_feedforward_layernorm.weight")));
            I.put(Q + "ffn_post_norm", importer::plus1(I.get(P + "post_feedforward_layernorm.weight")));
        }
        static const char* const proj[4][2] = {{"wq", "q_proj"}, {"wk", "k_proj"}, {"wv", "v_proj"}, {"wo", "o_proj"}};
        for (auto& pr : proj) I.put(Q + pr[0], I.get(P + "self_attn." + pr[1] + ".weight"), true);
        if (gen == 3) {
            I.put(Q + "q_norm", importer::plus1(I.get(P + "self_attn.q_norm.weight")));
            I.put(Q + "k_norm", importer::plus1(I.get(P + "self_attn.k_norm.weight")));
        }
        I.put(Q + "w_gate", I.get(P + "mlp.gate_proj.weight"), true);
        I.put(Q + "w_up", I.get(P + "mlp.up_proj.weight"), true);
        I.put(Q + "w_down", I.get(P + "mlp.down_proj.weight"), true);
    }
    I.put("out_norm", importer::plus1(I.get("norm.weight")));
    I.put("lm_head", te, true);   // Gemma ties the head to the embedding
    return I.finish();
}

// ---- Phi-3 / Phi-4-mini --------------------------------------------------------------------------------------------
inline import_report import_phi3(importer& I) {
    double heads = I.need("num_attention_heads"), hidden = I.need("hidden_size"), hd = hidden / heads;
    if (const auto* rs = I.cfg.find("rope_scaling"); rs && rs->is_obj()) raise(errc::unsupported, "phi3: rope_scaling (LongRoPE, the 128k-context variants) is not supported");
    if (I.str("hidden_act", "silu") != "silu") raise(errc::unsupported, "phi3: hidden_act must be silu");
    double rot = std::floor(hd * I.num("partial_rotary_factor", 1.0) + 1e-9);
    if (rot < 2 || rot > hd || std::fmod(rot, 2) != 0) raise(errc::unsupported, "phi3: rotary width {} of head_dim {}", rot, hd);
    I.set("head_dim", hd); I.set("num_key_value_heads", I.num("num_key_value_heads", heads));
    I.set("rms_norm_eps", I.num("rms_norm_eps", 1e-5)); I.set("rope_theta", I.num("rope_theta", 10000.0));
    I.set("rotary_dim", rot); I.set("rope_fn", std::string(rot < hd ? "rope_part" : "rope_full"));
    I.find_prefix({"model."}, "embed_tokens.weight");
    I.begin("phi3");
    auto te = I.get("embed_tokens.weight");
    I.emb("tok_emb", te);
    i64 L = i64(I.bp.param("n_layers"));
    for (i64 l = 0; l < L; ++l) {
        auto P = std::format("layers.{}.", l), Q = std::format("layer.{}.", l);
        I.put(Q + "attn_norm", I.get(P + "input_layernorm.weight"));
        I.put(Q + "ffn_norm", I.get(P + "post_attention_layernorm.weight"));
        I.put(Q + "w_qkv", I.get(P + "self_attn.qkv_proj.weight"), true);
        I.put(Q + "wo", I.get(P + "self_attn.o_proj.weight"), true);
        I.put(Q + "w_gate_up", I.get(P + "mlp.gate_up_proj.weight"), true);
        I.put(Q + "w_down", I.get(P + "mlp.down_proj.weight"), true);
    }
    I.put("out_norm", I.get("norm.weight"));
    I.put("lm_head", I.raw_has("lm_head.weight") ? I.raw_get("lm_head.weight") : te, true);
    return I.finish();
}

// ---- BERT family (embedding model) ---------------------------------------------------------------------------------
inline import_report import_bert(importer& I, bool roberta) {
    double heads = I.need("num_attention_heads"), hidden = I.need("hidden_size");
    if (I.str("position_embedding_type", "absolute") != "absolute") raise(errc::unsupported, "bert: only absolute position embeddings are supported");
    I.set("head_dim", hidden / heads); I.set("layer_norm_eps", I.num("layer_norm_eps", 1e-12));
    I.set("act", act_op(I.str("hidden_act", "gelu"), "bert"));
    // RoBERTa / XLM-R number positions from padding_idx + 1
    double off = roberta ? I.num("pad_token_id", 1) + 1 : 0;
    I.set("position_offset", off);
    // sentence-transformers pooling (1_Pooling/config.json) and a Normalize module (modules.json); default: CLS
    bool mean = false, l2 = false;
    if (std::filesystem::exists(I.dir / "1_Pooling" / "config.json")) {
        auto pc = json::parse(read_file(I.dir / "1_Pooling" / "config.json"));
        auto flag = [&](const char* k) { auto* v = pc.find(k); return v && v->is_bool() && v->b(); };
        mean = flag("pooling_mode_mean_tokens");
        if (!mean && !flag("pooling_mode_cls_token")) raise(errc::unsupported, "bert: only CLS and mean pooling are supported");
    }
    if (std::filesystem::exists(I.dir / "modules.json")) l2 = read_file(I.dir / "modules.json").find("Normalize") != std::string::npos;
    if (auto* p = I.cfg.find("forge_pooling"); p && p->is_str()) mean = p->str() == "mean";   // explicit override in config.json
    if (auto* p = I.cfg.find("forge_normalize"); p && p->is_bool()) l2 = p->b();
    std::string pool = mean ? "masked_mean(x, mask)" : "reshape(slice(x, 0, 0, 1), [d_model])";
    I.set("pool_expr", l2 ? "l2_normalize(" + pool + ")" : pool);
    I.find_prefix({"", "bert.", "roberta.", "xlm_roberta.", "model."}, "embeddings.word_embeddings.weight");
    I.outside = [](const std::string& n) {
        if (n.find("pooler.") != std::string::npos) return true;
        for (const char* p : {"cls.", "classifier.", "lm_head.", "qa_outputs."}) if (n.starts_with(p)) return true;
        return false;
    };
    I.outside_what = "the pooler and task heads (cls, classifier, lm_head, qa_outputs)";
    I.begin("bert");
    I.emb("tok_emb", I.get("embeddings.word_embeddings.weight"));
    I.put("pos_emb", I.get("embeddings.position_embeddings.weight"));
    I.put("type_emb", I.get("embeddings.token_type_embeddings.weight"));
    I.norm("emb_norm", "embeddings.LayerNorm");
    i64 L = i64(I.bp.param("n_layers"));
    for (i64 l = 0; l < L; ++l) {
        auto P = std::format("encoder.layer.{}.", l), Q = std::format("layer.{}.", l);
        I.linear(Q + "wq", Q + "bq", P + "attention.self.query"); I.linear(Q + "wk", Q + "bk", P + "attention.self.key");
        I.linear(Q + "wv", Q + "bv", P + "attention.self.value"); I.linear(Q + "wo", Q + "bo", P + "attention.output.dense");
        I.norm(Q + "attn_norm", P + "attention.output.LayerNorm");
        I.linear(Q + "w_in", Q + "b_in", P + "intermediate.dense"); I.linear(Q + "w_out", Q + "b_out", P + "output.dense");
        I.norm(Q + "ffn_norm", P + "output.LayerNorm");
    }
    return I.finish({{"max_position_embeddings", I.num("max_position_embeddings", 512) - off}, {"pooling", std::string(mean ? "mean" : "cls")}, {"normalize", l2}});
}

// ---- ViT / CLIP vision / SigLIP vision (image-embedding model) -----------------------------------------------------
inline import_report import_vision(importer& I, const std::string& kind) {
    // CLIP / SigLIP checkpoints nest the tower's settings under vision_config
    I.descend("vision_config");
    double heads = I.need("num_attention_heads"), hidden = I.need("hidden_size"), img = I.num("image_size", 224), patch = I.num("patch_size", 16);
    if (std::fmod(img, patch) != 0) raise(errc::unsupported, "{}: image_size {} is not a multiple of patch_size {}", kind, img, patch);
    double np = (img / patch) * (img / patch);
    bool cls = kind != "siglip";
    I.set("head_dim", hidden / heads); I.set("image_size", img); I.set("patch_size", patch); I.set("num_channels", I.num("num_channels", 3));
    I.set("num_patches", np); I.set("num_tokens", np + (cls ? 1 : 0)); I.set("layer_norm_eps", I.num("layer_norm_eps", kind == "vit" ? 1e-12 : kind == "clip" ? 1e-5 : 1e-6));
    I.set("act", act_op(I.str("hidden_act", kind == "clip" ? "quick_gelu" : kind == "siglip" ? "gelu_pytorch_tanh" : "gelu"), kind.c_str()));
    I.set("embed_expr", std::string(cls ? "concat(W.cls, p, axis: 0) + W.pos_emb" : "p + W.pos_emb"));
    I.set("pre_expr", std::string(kind == "clip" ? "layer_norm(x, W.pre_norm_w, W.pre_norm_b, eps)" : "x"));
    if (kind == "vit") I.find_prefix({"", "vit."}, "embeddings.patch_embeddings.projection.weight");
    else I.find_prefix({"vision_model.", "vision_tower.vision_model.", ""}, "embeddings.patch_embedding.weight");
    bool proj = kind == "clip" && I.raw_has("visual_projection.weight");
    I.outside = [](const std::string& n) {
        if (n.find("pooler.") != std::string::npos) return true;
        for (const char* p : {"text_model.", "text_projection.", "logit_scale", "logit_bias", "classifier."}) if (n.starts_with(p)) return true;
        return false;
    };
    I.outside_what = "the text tower, logit scale / bias, pooler and classifier head";
    bool map_head = kind == "siglip" && I.has("head.probe");
    std::string out = kind == "siglip" ? (map_head ? "map_pool(layer_norm(x, W.post_norm_w, W.post_norm_b, eps))" : "reduce_mean(layer_norm(x, W.post_norm_w, W.post_norm_b, eps), axis: 0)")
                                       : "layer_norm(reshape(slice(x, 0, 0, 1), [d_model]), W.post_norm_w, W.post_norm_b, eps)";
    if (proj) out = "reshape(linear(reshape(" + out + ", [1, d_model]), W.proj), [-1])";
    I.set("pool_expr", out);
    double od = hidden;
    if (proj) od = double(I.raw_get("visual_projection.weight").shape[0]);
    I.set("out_dim", od);
    I.begin("vit");
    i64 d = i64(hidden), L = i64(I.bp.param("n_layers"));
    if (kind == "vit") {
        I.put("patch_w", I.get("embeddings.patch_embeddings.projection.weight")); I.put("patch_b", I.get("embeddings.patch_embeddings.projection.bias"));
        I.put("cls", importer::reshaped(I.get("embeddings.cls_token"), {1, d}));
        I.put("pos_emb", importer::reshaped(I.get("embeddings.position_embeddings"), {i64(np) + 1, d}));
    } else {
        I.put("patch_w", I.get("embeddings.patch_embedding.weight"));
        I.put("patch_b", I.has("embeddings.patch_embedding.bias") ? I.get("embeddings.patch_embedding.bias") : tensor(shape_t{d}, 0.f));
        if (cls) I.put("cls", importer::reshaped(I.get("embeddings.class_embedding"), {1, d}));
        I.put("pos_emb", I.get("embeddings.position_embedding.weight"));
        if (kind == "clip") I.norm("pre_norm", I.has("pre_layrnorm.weight") ? "pre_layrnorm" : "pre_layernorm");   // (sic) the checkpoint key
    }
    for (i64 l = 0; l < L; ++l) {
        auto Q = std::format("layer.{}.", l);
        if (kind == "vit") {
            auto P = std::format("encoder.layer.{}.", l);
            I.norm(Q + "norm1", P + "layernorm_before"); I.norm(Q + "norm2", P + "layernorm_after");
            I.linear(Q + "wq", Q + "bq", P + "attention.attention.query"); I.linear(Q + "wk", Q + "bk", P + "attention.attention.key");
            I.linear(Q + "wv", Q + "bv", P + "attention.attention.value"); I.linear(Q + "wo", Q + "bo", P + "attention.output.dense");
            I.linear(Q + "w_in", Q + "b_in", P + "intermediate.dense"); I.linear(Q + "w_out", Q + "b_out", P + "output.dense");
        } else {
            auto P = std::format("encoder.layers.{}.", l);
            I.norm(Q + "norm1", P + "layer_norm1"); I.norm(Q + "norm2", P + "layer_norm2");
            I.linear(Q + "wq", Q + "bq", P + "self_attn.q_proj"); I.linear(Q + "wk", Q + "bk", P + "self_attn.k_proj");
            I.linear(Q + "wv", Q + "bv", P + "self_attn.v_proj"); I.linear(Q + "wo", Q + "bo", P + "self_attn.out_proj");
            I.linear(Q + "w_in", Q + "b_in", P + "mlp.fc1"); I.linear(Q + "w_out", Q + "b_out", P + "mlp.fc2");
        }
    }
    I.norm("post_norm", kind == "vit" ? "layernorm" : "post_layernorm");
    if (proj) I.put("proj", I.raw_get("visual_projection.weight"), true);
    if (map_head) {   // SigLIP attention-pooling head: nn.MultiheadAttention stores q/k/v stacked in in_proj
        auto ipw = I.get("head.attention.in_proj_weight"), ipb = I.get("head.attention.in_proj_bias");
        ipb.shape = {3 * d, 1};
        const char* nm[3] = {"q", "k", "v"};
        for (int j = 0; j < 3; ++j) {
            I.put(std::string("head.w") + nm[j], importer::rows(ipw, j * d, (j + 1) * d), true);
            I.put(std::string("head.b") + nm[j], importer::reshaped(importer::rows(ipb, j * d, (j + 1) * d), {d}));
        }
        I.put("head.probe", importer::reshaped(I.get("head.probe"), {1, d}));
        I.linear("head.wo", "head.bo", "head.attention.out_proj"); I.norm("head.norm", "head.layernorm");
        I.linear("head.w_in", "head.b_in", "head.mlp.fc1"); I.linear("head.w_out", "head.b_out", "head.mlp.fc2");
    }
    return I.finish({{"max_position_embeddings", np + 1}, {"image_size", img}, {"patch_size", patch}});
}

// ---- Whisper encoder -----------------------------------------------------------------------------------------------
inline import_report import_whisper_encoder(importer& I) {
    double d = I.need("d_model"), heads = I.need("encoder_attention_heads"), ctx = I.num("max_source_positions", 1500);
    I.set("hidden_size", d); I.set("num_hidden_layers", I.need("encoder_layers")); I.set("num_attention_heads", heads);
    I.set("head_dim", d / heads); I.set("intermediate_size", I.need("encoder_ffn_dim")); I.set("num_mel_bins", I.num("num_mel_bins", 80));
    I.set("max_source_positions", ctx); I.set("num_frames", 2 * ctx);
    I.set("act", act_op(I.str("activation_function", "gelu"), "whisper"));
    I.find_prefix({"model.encoder.", "encoder."}, "conv1.weight");
    I.outside = [](const std::string& n) { for (const char* p : {"model.decoder.", "decoder.", "proj_out."}) if (n.starts_with(p)) return true; return false; };
    I.outside_what = "the decoder (only the encoder is imported)";
    I.begin("whisper_encoder");
    I.put("conv1_w", I.get("conv1.weight")); I.put("conv1_b", I.get("conv1.bias"));
    I.put("conv2_w", I.get("conv2.weight")); I.put("conv2_b", I.get("conv2.bias"));
    I.put("pos_emb", I.get("embed_positions.weight"));
    i64 L = i64(I.bp.param("n_layers"));
    for (i64 l = 0; l < L; ++l) {
        auto P = std::format("layers.{}.", l), Q = std::format("layer.{}.", l);
        I.norm(Q + "norm1", P + "self_attn_layer_norm"); I.norm(Q + "norm2", P + "final_layer_norm");
        I.linear(Q + "wq", Q + "bq", P + "self_attn.q_proj"); I.linear(Q + "wk", "", P + "self_attn.k_proj");   // Whisper's key projection has no bias
        I.linear(Q + "wv", Q + "bv", P + "self_attn.v_proj"); I.linear(Q + "wo", Q + "bo", P + "self_attn.out_proj");
        I.linear(Q + "w_in", Q + "b_in", P + "fc1"); I.linear(Q + "w_out", Q + "b_out", P + "fc2");
    }
    I.norm("post_norm", "layer_norm");
    return I.finish({{"max_position_embeddings", ctx}, {"num_mel_bins", I.num("num_mel_bins", 80)}, {"num_frames", 2 * ctx}});
}

} // namespace arch_detail

// The template an HF config selects among the families above ("" = not one of them). `--template` names win.
inline std::string hf_extra_template(const json::value& cfg, const std::string& forced = {}) {
    static const char* const known[] = {"gpt2", "gemma", "gemma2", "gemma3", "phi3", "bert", "roberta", "vit", "clip_vision", "siglip_vision", "whisper_encoder"};
    if (!forced.empty()) { for (auto* k : known) if (forced == k) return forced; return {}; }
    std::string arch = cfg.find("architectures") && cfg["architectures"].is_arr() && !cfg["architectures"].arr().empty() ? cfg["architectures"][0].str() : "";
    std::string mt = cfg.str_or("model_type", "");
    auto is = [&](std::initializer_list<const char*> l) { for (auto* a : l) if (arch == a) return true; return false; };
    if (is({"GPT2LMHeadModel", "GPT2Model"}) || mt == "gpt2") return "gpt2";
    if (is({"Gemma3ForCausalLM", "Gemma3ForConditionalGeneration", "Gemma3TextModel"}) || mt == "gemma3_text" || mt == "gemma3") return "gemma3";
    if (is({"Gemma2ForCausalLM", "Gemma2Model"}) || mt == "gemma2") return "gemma2";
    if (is({"GemmaForCausalLM", "GemmaModel"}) || mt == "gemma") return "gemma";
    if (is({"Phi3ForCausalLM", "Phi3Model"}) || mt == "phi3") return "phi3";
    if (is({"RobertaModel", "RobertaForMaskedLM", "XLMRobertaModel", "XLMRobertaForMaskedLM"}) || mt == "roberta" || mt == "xlm-roberta") return "roberta";
    if (is({"BertModel", "BertForMaskedLM", "BertForPreTraining", "BertForSequenceClassification"}) || mt == "bert") return "bert";
    if (is({"ViTModel", "ViTForImageClassification", "ViTForMaskedImageModeling"}) || mt == "vit") return "vit";
    if (is({"CLIPVisionModel", "CLIPVisionModelWithProjection", "CLIPModel"}) || mt == "clip" || mt == "clip_vision_model") return "clip_vision";
    if (is({"SiglipVisionModel", "SiglipModel"}) || mt == "siglip" || mt == "siglip_vision_model") return "siglip_vision";
    if (is({"WhisperForConditionalGeneration", "WhisperModel"}) || mt == "whisper") return "whisper_encoder";
    return {};
}

// Returns nothing when the config is not one of the families handled here (the caller then runs its own path).
inline std::optional<import_report> import_hf_extra(const std::filesystem::path& dir, const json::value& cfg, const std::string& out, const import_options& opt) {
    std::string t = hf_extra_template(cfg, opt.template_name);
    if (t.empty()) return std::nullopt;
    std::string arch = cfg.find("architectures") && cfg["architectures"].is_arr() && !cfg["architectures"].arr().empty() ? cfg["architectures"][0].str() : cfg.str_or("model_type", t);
    import_options o2 = opt; o2.template_name.clear();   // the family decides the template file
    arch_detail::importer I(dir, cfg, out, o2, arch);
    if (t == "gpt2") return arch_detail::import_gpt2(I);
    if (t == "gemma") return arch_detail::import_gemma(I, 1);
    if (t == "gemma2") return arch_detail::import_gemma(I, 2);
    if (t == "gemma3") {   // multimodal Gemma-3 checkpoints keep the decoder's settings under text_config
        I.descend("text_config");
        return arch_detail::import_gemma(I, 3);
    }
    if (t == "phi3") return arch_detail::import_phi3(I);
    if (t == "bert") return arch_detail::import_bert(I, false);
    if (t == "roberta") return arch_detail::import_bert(I, true);
    if (t == "vit") return arch_detail::import_vision(I, "vit");
    if (t == "clip_vision") return arch_detail::import_vision(I, "clip");
    if (t == "siglip_vision") return arch_detail::import_vision(I, "siglip");
    return arch_detail::import_whisper_encoder(I);
}

} // namespace forge::import
