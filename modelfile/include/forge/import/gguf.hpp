#pragma once
#include <forge/core/mapped.hpp>
// GGUF (v2/v3) reader and GGUF -> .fpk importer. Quantised blocks are copied verbatim: FORGE's codecs use the GGUF
// block layouts (gguf_file::type_of lists them), so no re-quantisation error is introduced unless --codec asks for it.
#include <forge/import/hf.hpp>
#include <map>
#include <set>

namespace forge::import {

class gguf_file {
public:
    struct tinfo { std::string name; shape_t shape; std::uint32_t type = 0; std::uint64_t offset = 0; };
    explicit gguf_file(const std::string& path) : path_(path) {
        check_read(path);   // FSS CAP-1 (hf.hpp)
        map_ = forge::map_readonly(path, size_);
        try {
            if (!map_) raise(errc::format_error, "{}: empty file", path);
            if (size_ < 24 || std::memcmp(map_, "GGUF", 4)) raise(errc::format_error, "{}: not a GGUF file", path);
            auto ver = rd<std::uint32_t>();
            if (ver < 2 || ver > 3) raise(errc::unsupported, "{}: GGUF version {} unsupported", path, ver);
            auto nt = rd<std::uint64_t>(), nkv = rd<std::uint64_t>();
            for (std::uint64_t i = 0; i < nkv; ++i) { auto k = rstr(); auto t = rd<std::uint32_t>(); kv_.emplace_back(k, rval(t)); }
            for (std::uint64_t i = 0; i < nt; ++i) {
                tinfo ti; ti.name = rstr();
                auto nd = rd<std::uint32_t>(); if (nd > 8) raise(errc::format_error, "{}: tensor {} has {} dims", path_, ti.name, nd);
                std::vector<i64> ne(nd); for (auto& d : ne) d = i64(rd<std::uint64_t>());
                ti.shape.assign(ne.rbegin(), ne.rend());   // ggml ne0 is the innermost dimension
                ti.type = rd<std::uint32_t>(); ti.offset = rd<std::uint64_t>();
                tensors_.push_back(ti);
            }
            double al = get("general.alignment") ? get("general.alignment")->num() : 32;
            if (!(al >= 1 && al <= double(1u << 30))) raise(errc::format_error, "{}: bad general.alignment", path);
            std::size_t align = std::size_t(al);
            data_ = (p_ + align - 1) / align * align;
        } catch (...) { forge::unmap_readonly(map_, size_); map_ = nullptr; throw; }   // no dtor on a throwing ctor
    }
    gguf_file(const gguf_file&) = delete;
    ~gguf_file() { forge::unmap_readonly(map_, size_); }
    const json::value* get(std::string_view k) const { for (auto& [kk, v] : kv_) if (kk == k) return &v; return nullptr; }
    const json::value& need(std::string_view k) const { if (auto* v = get(k)) return *v; raise(errc::not_found, "{}: missing GGUF key {}", path_, k); }
    const std::vector<tinfo>& tensors() const { return tensors_; }
    const tinfo* find(const std::string& n) const { for (auto& t : tensors_) if (t.name == n) return &t; return nullptr; }
    // ---- tensor types ----
    // One row per ggml tensor type FORGE knows. `codec`: the FORGE weight codec with the same block layout (the
    // blocks are stored verbatim). No codec but a block size: a plain numeric or activation type, decoded to f32 at
    // import ("transcoded"). Neither: recognised but not loadable, with the reason.
    struct ggml_type { std::uint32_t id; const char* name; const char* codec; std::size_t block, block_bytes; const char* why; };
    static const ggml_type* type_of(std::uint32_t t) {
        static const char* const grid = "its codebook grid is not part of FORGE (no table is guessed: re-export the model as IQ4_XS, Q3_K or another supported type)";
        static const ggml_type tab[] = {
            {0, "F32", "f32", 1, 4, nullptr}, {1, "F16", "f16", 1, 2, nullptr}, {30, "BF16", "bf16", 1, 2, nullptr},
            {2, "Q4_0", "q4_0", 32, 18, nullptr}, {3, "Q4_1", "q4_1", 32, 20, nullptr}, {6, "Q5_0", "q5_0", 32, 22, nullptr},
            {7, "Q5_1", "q5_1", 32, 24, nullptr}, {8, "Q8_0", "q8_0", 32, 34, nullptr},
            {10, "Q2_K", "q2_k", 256, 84, nullptr}, {11, "Q3_K", "q3_k", 256, 110, nullptr}, {12, "Q4_K", "q4_k", 256, 144, nullptr},
            {13, "Q5_K", "q5_k", 256, 176, nullptr}, {14, "Q6_K", "q6_k", 256, 210, nullptr},
            {20, "IQ4_NL", "iq4_nl", 32, 18, nullptr}, {23, "IQ4_XS", "iq4_xs", 256, 136, nullptr},
            {34, "TQ1_0", "tq1_0", 256, 54, nullptr}, {35, "TQ2_0", "tq2_0", 256, 66, nullptr},
            // transcoded to f32
            {15, "Q8_K", nullptr, 256, 292, nullptr}, {24, "I8", nullptr, 1, 1, nullptr}, {25, "I16", nullptr, 1, 2, nullptr},
            {26, "I32", nullptr, 1, 4, nullptr}, {27, "I64", nullptr, 1, 8, nullptr}, {28, "F64", nullptr, 1, 8, nullptr},
            // recognised, not loadable
            {9, "Q8_1", nullptr, 0, 0, "it is an activation format with no reference dequantiser (and two published block sizes)"},
            {16, "IQ2_XXS", nullptr, 0, 0, grid}, {17, "IQ2_XS", nullptr, 0, 0, grid}, {18, "IQ3_XXS", nullptr, 0, 0, grid},
            {19, "IQ1_S", nullptr, 0, 0, grid}, {21, "IQ3_S", nullptr, 0, 0, grid}, {22, "IQ2_S", nullptr, 0, 0, grid},
            {29, "IQ1_M", nullptr, 0, 0, grid},
            {39, "MXFP4", nullptr, 0, 0, "ggml's MXFP4 block layout differs from FORGE's mxfp4 codec and has no importer yet"},
        };
        for (auto& e : tab) if (e.id == t) return &e;
        return nullptr;
    }
    static constexpr const char* supported_types = "F32 F16 BF16 F64 I8 I16 I32 I64 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K Q8_K IQ4_NL IQ4_XS TQ1_0 TQ2_0";
    // The FORGE codec a type is stored as without transcoding, or nullptr.
    static const char* codec_of(std::uint32_t t) { auto* e = type_of(t); return e ? e->codec : nullptr; }
    static bool loadable(std::uint32_t t) { auto* e = type_of(t); return e && e->block; }
    static std::string type_name(std::uint32_t t) { auto* e = type_of(t); return e ? std::string(e->name) : std::format("type {}", t); }
    // Element count, overflow-checked: no loadable type packs more than 8 elements into a byte, so a count beyond
    // 8 x the file size cannot be backed by data.
    std::uint64_t count(const tinfo& t) const {
        std::uint64_t n = 1;
        for (auto d : t.shape) if (d == 0) return 0;
        for (auto d : t.shape) {
            if (d < 0 || std::uint64_t(d) > std::uint64_t(size_) * 8 / n) raise(errc::format_error, "tensor {}: bad shape or out of bounds", t.name);
            n *= std::uint64_t(d);
        }
        return n;
    }
    // The tensor's bytes in the file, bounds-checked (whole blocks of its ggml type).
    std::span<const std::uint8_t> bytes(const tinfo& t) const {
        auto* ty = type_of(t.type);
        if (!ty) raise(errc::unsupported, "tensor {}: ggml type {} not supported (supported: {})", t.name, t.type, supported_types);
        if (!ty->block) raise(errc::unsupported, "tensor {}: ggml type {} is not supported: {}", t.name, ty->name, ty->why);
        std::uint64_t n = count(t);
        if (n % ty->block) raise(errc::format_error, "tensor {}: {} elements is not a whole number of {} blocks ({} per block)", t.name, n, ty->name, ty->block);
        std::uint64_t nb = n / ty->block * ty->block_bytes;   // n <= 8 * size_ and block_bytes <= 292: cannot overflow 64 bits
        if (data_ > size_ || t.offset > size_ - data_ || nb > size_ - data_ - t.offset) raise(errc::format_error, "tensor {} out of bounds", t.name);
        return {map_ + data_ + std::size_t(t.offset), std::size_t(nb)};
    }
    // A tensor as the importer stores it: the file's blocks under the matching FORGE codec, or, for a transcoded
    // type, an f32 copy (`own`). I32 / I64 / F64 values round to the nearest f32; Q8_K decodes as d * q.
    struct stored_tensor { const char* codec; std::span<const std::uint8_t> bytes; std::vector<std::uint8_t> own; const char* type; bool transcoded; };
    stored_tensor stored(const tinfo& t) const {
        auto b = bytes(t); auto* ty = type_of(t.type);
        if (ty->codec) return {ty->codec, b, {}, ty->name, false};
        std::size_t n = std::size_t(count(t));   // fits: bytes() found that many elements inside the mapping
        stored_tensor r{"f32", {}, std::vector<std::uint8_t>(n * 4), ty->name, true};
        auto put = [&](std::size_t i, float v) { std::memcpy(r.own.data() + 4 * i, &v, 4); };
        auto conv = [&]<class T>(T) { for (std::size_t i = 0; i < n; ++i) { T v; std::memcpy(&v, b.data() + i * sizeof(T), sizeof(T)); put(i, float(v)); } };
        switch (t.type) {
            case 24: conv(std::int8_t{}); break; case 25: conv(std::int16_t{}); break; case 26: conv(std::int32_t{}); break;
            case 27: conv(std::int64_t{}); break; case 28: conv(double{}); break;
            case 15:   // Q8_K: { f32 d; i8 qs[256]; i16 bsums[16]; }  x = d * q
                for (std::size_t blk = 0; blk < n / 256; ++blk) {
                    const std::uint8_t* p = b.data() + blk * 292; float d; std::memcpy(&d, p, 4);
                    for (std::size_t i = 0; i < 256; ++i) put(blk * 256 + i, d * float(std::int8_t(p[4 + i])));
                }
                break;
            default: raise(errc::unsupported, "tensor {}: no transcoder for ggml type {}", t.name, ty->name);
        }
        r.bytes = r.own;
        return r;
    }
private:
    template <class T> T rd() { if (p_ + sizeof(T) > size_) raise(errc::format_error, "{}: truncated", path_); T v; std::memcpy(&v, map_ + p_, sizeof(T)); p_ += sizeof(T); return v; }
    std::string rstr() { auto n = rd<std::uint64_t>(); if (n > size_ - p_) raise(errc::format_error, "{}: truncated", path_); std::string s(reinterpret_cast<const char*>(map_ + p_), n); p_ += n; return s; }
    json::value rval(std::uint32_t t) {
        switch (t) {
        case 0: return double(rd<std::uint8_t>()); case 1: return double(rd<std::int8_t>());
        case 2: return double(rd<std::uint16_t>()); case 3: return double(rd<std::int16_t>());
        case 4: return double(rd<std::uint32_t>()); case 5: return double(rd<std::int32_t>());
        case 6: return double(rd<float>()); case 7: return rd<std::uint8_t>() != 0;
        case 8: return rstr();
        case 9: { auto et = rd<std::uint32_t>(); auto n = rd<std::uint64_t>(); if (n > size_ - p_ || ++adepth_ > 8) raise(errc::format_error, "{}: bad GGUF array", path_); json::array a; a.reserve(n); for (std::uint64_t i = 0; i < n; ++i) a.push_back(rval(et)); --adepth_; return a; }
        case 10: return double(rd<std::uint64_t>()); case 11: return double(rd<std::int64_t>());
        case 12: return rd<double>();
        default: raise(errc::format_error, "{}: bad GGUF value type {}", path_, t);
        }
    }
    std::string path_;
    const std::uint8_t* map_ = nullptr;
    std::size_t size_ = 0, p_ = 4, data_ = 0; int adepth_ = 0;
    std::vector<std::pair<std::string, json::value>> kv_;
    std::vector<tinfo> tensors_;
};

inline import_report import_gguf(const std::string& path, const std::string& out, const import_options& opt = {}) {
    check_write(out);
    gguf_file g(path);
    auto arch = g.need("general.architecture").str();
    if (arch != "llama" && arch != "qwen2" && arch != "qwen3" && arch != "mistral" && arch != "qwen35" && arch != "qwen3moe" && arch != "qwen35moe")
        raise(errc::unsupported, "GGUF architecture '{}' has no template", arch);
    // qwen2: the llama graph + Q/K/V biases (blk.N.attn_{q,k,v}.bias). A qwen2 file that carries no bias tensors is the
    // plain llama graph (NeoX RoPE); a file with only some of them fails on the missing one.
    bool qwen2 = arch == "qwen2" && (g.find("blk.0.attn_q.bias") || g.find("blk.0.attn_k.bias") || g.find("blk.0.attn_v.bias"));
    bool q35 = arch == "qwen35" || arch == "qwen35moe";
    bool moe = arch == "qwen3moe" || arch == "qwen35moe";   // sparse MoE FFNs (moe_ffn over stacked *_exps tensors)
    auto K = [&](const std::string& k) { return g.need(arch + "." + k).num(); };
    // per-layer keys (qwen35 head_count_kv: 0 on linear-attention layers) collapse to their largest entry
    auto Kor = [&](const std::string& k, double d) {
        auto* v = g.get(arch + "." + k);
        if (!v) return d;
        if (!v->is_arr()) return v->num();
        double m = d; bool any = false;
        for (auto& e : v->arr()) if (!any || e.num() > m) { m = e.num(); any = true; }
        return m;
    };
    double heads = K("attention.head_count"), hidden = K("embedding_length");
    const auto& toks = g.need("tokenizer.ggml.tokens").arr();
    json::object cfg{{"hidden_size", hidden}, {"num_hidden_layers", K("block_count")}, {"num_attention_heads", heads},
                     {"num_key_value_heads", Kor("attention.head_count_kv", heads)}, {"head_dim", Kor("attention.key_length", hidden / heads)},
                     {"intermediate_size", moe ? Kor("feed_forward_length", 0) : K("feed_forward_length")}, {"vocab_size", double(toks.size())},
                     {"rms_norm_eps", Kor("attention.layer_norm_rms_epsilon", 1e-6)}, {"rope_theta", Kor("rope.freq_base", 10000.0)},
                     // llama.cpp's converter permutes Q/K for llama so it can use interleaved (GPT-J) RoPE
                     {"rope_neox", arch == "llama" || arch == "mistral" ? 0.0 : 1.0}};
    // Qwen3.5 (llama.cpp / Ollama arch "qwen35"): the hybrid Gated DeltaNet + gated attention decoder of the HF
    // qwen3_5 template. The GGUF converter already applies the (1 + w) RMSNorm offset, A = -exp(A_log) and the conv
    // squeeze; it may also reorder the V heads from grouped-by-K-head to tiled order (ssm.v_head_reordered), which
    // the import undoes below.
    i64 lk_heads = 0, lk_dim = 0, lv_heads = 0, lv_dim = 0;
    bool v_tiled = false;
    if (q35) {
        double L0 = K("block_count"), iv = Kor("full_attention_interval", 4);
        if (iv < 1 || std::fmod(L0, iv) != 0) raise(errc::unsupported, "qwen35: {} layers is not a multiple of full_attention_interval {}", L0, iv);
        if (!moe && Kor("expert_count", 0) > 0) raise(errc::unsupported, "qwen35: expert_count set on a dense architecture (expected qwen35moe)");
        lk_heads = i64(K("ssm.group_count")); lk_dim = i64(K("ssm.state_size")); lv_heads = i64(K("ssm.time_step_rank"));
        if (lk_heads <= 0 || lv_heads <= 0 || lv_heads % lk_heads) raise(errc::format_error, "qwen35: bad ssm head counts {} / {}", lk_heads, lv_heads);
        lv_dim = i64(K("ssm.inner_size")) / lv_heads;
        double hd = Kor("attention.key_length", hidden / heads), rot = Kor("rope.dimension_count", hd);
        if (rot >= hd) raise(errc::unsupported, "qwen35: full-width RoPE is not supported by the template");
        auto* vr = g.get("qwen35.ssm.v_head_reordered");
        v_tiled = vr && (vr->is_bool() ? vr->b() : vr->num() != 0) && lv_heads != lk_heads;
        cfg.emplace_back("rotary_dim", rot); cfg.emplace_back("full_attention_interval", iv); cfg.emplace_back("n_blocks", L0 / iv);
        cfg.emplace_back("linear_num_key_heads", double(lk_heads)); cfg.emplace_back("linear_key_head_dim", double(lk_dim));
        cfg.emplace_back("linear_num_value_heads", double(lv_heads)); cfg.emplace_back("linear_value_head_dim", double(lv_dim));
        cfg.emplace_back("linear_conv_kernel_dim", Kor("ssm.conv_kernel", 4));
    }
    if (moe) {   // llama.cpp's qwen3moe / qwen35moe: softmax gating, weights renormalised over the k (norm_w = true)
        double ne = K("expert_count"), nk = K("expert_used_count"), ws = Kor("expert_weights_scale", 0);
        if (ne < 1 || nk < 1 || nk > ne) raise(errc::format_error, "{}: bad expert counts {} / {}", arch, nk, ne);
        if (Kor("expert_gating_func", 1) != 1) raise(errc::unsupported, "{}: only softmax expert gating is supported", arch);
        cfg.emplace_back("num_experts", ne); cfg.emplace_back("num_experts_per_tok", nk);
        cfg.emplace_back("moe_intermediate_size", K("expert_feed_forward_length"));
        cfg.emplace_back("norm_topk_prob", 1.0); cfg.emplace_back("routed_scaling_factor", ws > 0 ? ws : 1.0);
    }
    std::string tname = !opt.template_name.empty() ? opt.template_name : arch == "qwen3" ? "qwen3" : arch == "qwen3moe" ? "qwen3_moe"
                      : arch == "qwen35moe" ? "qwen3_5_moe" : q35 ? "qwen3_5" : qwen2 ? "qwen2" : "llama";
    auto tpl_text = read_file(template_path(tname));   // FSS CAP-3: the name is joined safely onto template_dir()
    // RoPE frequency scaling: <arch>.rope.scaling.{type, factor, original_context_length, attn_factor, yarn_attn_factor,
    // yarn_log_multiplier, yarn_ext_factor, yarn_beta_fast, yarn_beta_slow}, and the per-pair multipliers llama.cpp's converter bakes into rope_freqs.weight
    // for llama3-style scaling (the GGUF carries the table, not the low/high frequency factors).
    rope_spec rsp;
    const double head_dim = Kor("attention.key_length", hidden / heads);
    {
        auto* tv = g.get(arch + ".rope.scaling.type");
        std::string type = tv && tv->is_str() ? tv->str() : "none";
        auto knum = [&](const char* k, double d) { auto* v = g.get(arch + ".rope.scaling." + k); return v && v->is_num() ? v->num() : d; };
        if (type == "linear" || type == "yarn") {
            rsp.type = type == "yarn" ? ops::rope_scaling::yarn : ops::rope_scaling::linear;
            rsp.factor = knum("factor", 0);
            if (!(rsp.factor > 0)) raise(errc::format_error, "{}: rope scaling type '{}' without {}.rope.scaling.factor", path, type, arch);
            if (type == "yarn") {
                rsp.orig_ctx = knum("original_context_length", Kor("context_length", 0));
                if (!(rsp.orig_ctx > 0)) raise(errc::format_error, "{}: YaRN rope scaling without an original context length", path);
                rsp.beta_fast = knum("yarn_beta_fast", 32.0); rsp.beta_slow = knum("yarn_beta_slow", 1.0);
                // magnitude as llama.cpp: attn_factor * (1 + 0.1 ln f), or with a log multiplier L (0.1 * mscale_all_dim in
                // the converter) the Hugging Face ratio get_mscale(f, 1) / get_mscale(f, mscale_all_dim)
                double lnf = rsp.factor > 1 ? std::log(rsp.factor) : 0.0, lm = knum("yarn_log_multiplier", 0);
                rsp.attn_factor = knum("yarn_attn_factor", knum("attn_factor", 1.0)) * (1.0 + 0.1 * lnf) / (lm != 0 ? 1.0 + lm * lnf : 1.0);
                // yarn_ext_factor mixes the extrapolated band in: 1 (default) is YaRN, 0 is plain interpolation (linear)
                if (double ext = knum("yarn_ext_factor", 1.0); ext == 0) { rsp.type = ops::rope_scaling::linear; rsp.attn_factor = 1.0; }
                else if (ext != 1.0) raise(errc::unsupported, "{}: {}.rope.scaling.yarn_ext_factor {} is not supported (0 or 1)", path, arch, ext);
            }
        } else if (type == "longrope") raise(errc::unsupported, "{}: rope scaling type 'longrope' (LongRoPE) is not supported", path);
        else if (type != "none") raise(errc::unsupported, "{}: rope scaling type '{}' is not supported (supported: none, linear, yarn)", path, type);
        if (auto* ft = g.find("rope_freqs.weight")) {
            if (ft->shape.size() != 1) raise(errc::format_error, "{}: rope_freqs.weight has shape {}", path, shape_str(ft->shape));
            auto st = g.stored(*ft);   // any loadable type (an unsupported one raises with its name)
            rsp.freq_factors.resize(std::size_t(ft->shape[0]));
            graph::codecs::find(st.codec)->decode(st.bytes.data(), rsp.freq_factors.size(), rsp.freq_factors.data());
        }
        if (q35 && (rsp.active() || rsp.attn_factor != 1.0)) raise(errc::unsupported, "{}: {}: rope scaling is not supported by the template", path, arch);
        rsp.validate(tname, tpl_text, head_dim);
        cfg.emplace_back("rope_args", rsp.rope_args()); cfg.emplace_back("attn_args", rsp.attn_args(head_dim));
        cfg.emplace_back("opset", rsp.active() ? 2.0 : 1.0);
    }
    auto bp_text = instantiate(tpl_text, json::value(cfg));
    auto bp = modelfile::parse_blueprint(bp_text);
    modelfile::check_opset(bp);

    // streamed: tensors go straight from the mapped GGUF to disk, so a 20 GB model never sits in memory twice
    modelfile::stream_writer w(out);
    expert_index xi;
    import_report rep{arch, tname, 0, 0};
    std::set<std::string> used;
    if (!rsp.freq_factors.empty()) used.insert("rope_freqs.weight");   // consumed above, into the rope attrs
    // ---- tensor types and codecs (docs/modelfile.md) ----
    // --codec keep (the default): every tensor keeps its stored type. --codec C: the weight matrices are decoded and
    // re-encoded as C (tok_emb as --emb-codec; q8_0 by default when C is quantised), which quantises an already
    // quantised model a second time. Norms, biases, routers and the DeltaNet gate / conv tensors keep their type.
    const std::string rq = opt.requant.empty() || opt.requant == "keep" ? std::string() : opt.requant;
    if (is_mix_ftype(rq)) raise(errc::unsupported, "--codec {} (mixed precision) applies to Hugging Face inputs; a GGUF takes one codec or 'keep'", rq);
    const std::string rq_emb = !opt.emb_codec.empty() ? opt.emb_codec : rq == "f32" || rq == "f16" || rq == "bf16" ? rq : std::string("q8_0");
    if (!rq.empty() && !graph::codecs::find(rq)) raise(errc::unsupported, "unknown codec '{}'", rq);
    if (!rq.empty() && !graph::codecs::find(rq_emb)) raise(errc::unsupported, "unknown codec '{}'", rq_emb);
    std::map<std::string, std::size_t> seen, transcoded, requant_from;
    std::size_t requantised = 0, double_q = 0;
    auto fetch = [&](const gguf_file::tinfo& t) { auto s = g.stored(t); ++seen[s.type]; if (s.transcoded) ++transcoded[s.type]; return s; };
    // the codec `dst` is re-encoded as, or empty to keep what is stored (`have`)
    auto target = [&](const std::string& dst, const shape_t& shape, std::string_view have) -> std::string {
        static const std::set<std::string_view> mats = {"tok_emb", "lm_head", "wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down", "w_qkv", "w_z"};
        if (rq.empty() || shape.size() < 2 || dst.starts_with("extra.") || !mats.count(recipe_leaf(dst))) return {};
        std::string want = dst == "tok_emb" ? rq_emb : rq;
        if (auto* c = graph::codecs::find(want); c->block > 1 && shape.back() % i64(c->block)) {   // rows must be whole blocks
            if (shape.back() % 32) return {};
            want = "q8_0";
        }
        return want == have ? std::string() : want;
    };
    auto requant_note = [&](const gguf_file::stored_tensor& s, const std::string& want) {
        ++requantised; ++requant_from[s.type];
        bool lossy_src = graph::codecs::find(s.codec)->block > 1 || std::string_view(s.type) == "Q8_K";
        if (lossy_src && want != "f32" && want != "f16" && want != "bf16") ++double_q;
    };
    // every tensor is written here: `bytes` are blocks of s.codec (s.bytes, or a permuted copy)
    auto store = [&](const std::string& dst, const shape_t& shape, const gguf_file::stored_tensor& s, std::span<const std::uint8_t> bytes,
                     std::size_t align = modelfile::alignment) {
        ++rep.tensors;
        auto want = target(dst, shape, s.codec);
        if (want.empty()) { w.add_raw(dst, shape, s.codec, bytes, align); return; }
        std::vector<float> x(std::size_t(numel(shape)));
        if (graph::codecs::find(s.codec)->bytes(x.size()) != bytes.size()) raise(errc::format_error, "tensor {}: {} bytes do not match {} {}", dst, bytes.size(), s.codec, shape_str(shape));
        graph::codecs::find(s.codec)->decode(bytes.data(), x.size(), x.data());
        w.add_tensor(dst, shape, x, want, align);
        requant_note(s, want);
    };
    auto put = [&](const std::string& dst, const std::string& src) {
        used.insert(src);
        auto* t = g.find(src);
        if (!t) raise(errc::not_found, "{}: template '{}' needs tensor {}, which the file does not have", path, tname, src);
        auto s = fetch(*t);
        store(dst, t->shape, s, s.bytes);
    };
    auto need = [&](const std::string& src) -> const gguf_file::tinfo& {
        auto* t = g.find(src);
        if (!t) raise(errc::not_found, "{}: template '{}' needs tensor {}, which the file does not have", path, tname, src);
        used.insert(src);
        return *t;
    };
    // tiled -> grouped V-head order: grouped head k*per + v sits at tiled head v*lk_heads + k
    auto vhead = [&](i64 h) { i64 per = lv_heads / lk_heads; return (h % per) * lk_heads + h / per; };
    // copy src with the rows [row0, row0 + lv_heads*rows_per_head) put back in grouped V-head order (raw blocks)
    auto put_vrows = [&](const std::string& dst, const std::string& src, i64 row0, i64 rows_per_head) {
        auto& t = need(src);
        auto st = fetch(t); auto* codec = st.codec; auto b = st.bytes;
        if (!v_tiled) { store(dst, t.shape, st, b); return; }
        i64 rows = t.shape.empty() ? 1 : t.shape[0], cols = t.shape.size() > 1 ? numel(t.shape) / rows : 1;
        if (t.shape.size() == 1) { rows = t.shape[0]; cols = 1; }
        auto* c = graph::codecs::find(codec);
        if (c->block > 1 && cols % i64(c->block)) raise(errc::unsupported, "tensor {}: rows are not whole {} blocks", src, codec);
        std::size_t rb = c->bytes(std::size_t(cols));
        if (rows < row0 + lv_heads * rows_per_head) raise(errc::shape_mismatch, "tensor {} has shape {}", src, shape_str(t.shape));
        std::vector<std::uint8_t> o(b.begin(), b.end());
        for (i64 h = 0; h < lv_heads; ++h)
            std::memcpy(o.data() + std::size_t(row0 + h * rows_per_head) * rb, b.data() + std::size_t(row0 + vhead(h) * rows_per_head) * rb,
                        rb * std::size_t(rows_per_head));
        store(dst, t.shape, st, o);
    };
    // the out projection consumes V heads along its input (column) dimension: blocks straddle heads, so decode,
    // permute and re-encode (q8_0 for quantised sources, which keeps the extra rounding negligible)
    auto put_vcols = [&](const std::string& dst, const std::string& src) {
        auto& t = need(src);
        auto st = fetch(t); auto* codec = st.codec;
        if (!v_tiled) { store(dst, t.shape, st, st.bytes); return; }
        auto* c = graph::codecs::find(codec);
        if (t.shape.size() != 2 || t.shape[1] != lv_heads * lv_dim) raise(errc::shape_mismatch, "tensor {} has shape {}", src, shape_str(t.shape));
        std::size_t n = std::size_t(numel(t.shape));
        std::vector<float> x(n), y(n);
        c->decode(st.bytes.data(), n, x.data());
        for (i64 r = 0; r < t.shape[0]; ++r)
            for (i64 h = 0; h < lv_heads; ++h)
                std::memcpy(&y[std::size_t(r * t.shape[1] + h * lv_dim)], &x[std::size_t(r * t.shape[1] + vhead(h) * lv_dim)], std::size_t(lv_dim) * 4);
        std::string out_codec = c->block > 1 ? "q8_0" : std::string(codec);
        if (auto want = target(dst, t.shape, {}); !want.empty()) { out_codec = want; requant_note(st, want); }
        w.add_tensor(dst, t.shape, y, out_codec); ++rep.tensors;
    };
    // MoE FFN: the stacked *_exps tensors ([n_experts, rows, k]; one expert = a contiguous run of whole rows) are copied
    // verbatim, each starting on a 4 KiB file offset and listed in the "experts" section; plus the router and, for
    // qwen35moe, the sigmoid-gated shared expert.
    auto put_moe = [&](i64 l, const std::string& P, const std::string& Q) {
        if (g.find(P + "ffn_gate_up_exps.weight")) raise(errc::unsupported, "{}: fused ffn_gate_up_exps tensors are not supported", path);
        put(Q + "router", P + "ffn_gate_inp.weight");
        const char* proj[3][2] = {{"gate", "ffn_gate_exps.weight"}, {"up", "ffn_up_exps.weight"}, {"down", "ffn_down_exps.weight"}};
        for (auto& pj : proj) {
            auto& t = need(P + pj[1]);
            if (t.shape.size() != 3) raise(errc::format_error, "{}: {} has shape {}", path, t.name, shape_str(t.shape));
            { auto s = fetch(t); store(Q + "experts.w_" + pj[0], t.shape, s, s.bytes, modelfile::expert_alignment); }
            xi.add(w.last(), l, pj[0]);
        }
        if (g.find(P + "ffn_up_shexp.weight")) {
            put(Q + "shared.w_gate", P + "ffn_gate_shexp.weight"); put(Q + "shared.w_up", P + "ffn_up_shexp.weight");
            put(Q + "shared.w_down", P + "ffn_down_shexp.weight");
            auto& gt = need(P + "ffn_gate_inp_shexp.weight");   // [d] -> a one-row linear [1, d]
            shape_t gs = gt.shape; if (gs.size() == 1) gs = {1, gs[0]};
            { auto s = fetch(gt); store(Q + "shared.gate", gs, s, s.bytes); }
        } else if (q35) raise(errc::not_found, "{}: layer {} has no shared expert (ffn_up_shexp)", path, l);
    };
    put("tok_emb", "token_embd.weight");
    for (i64 l = 0, L = i64(bp.param("n_layers")); l < L; ++l) {
        auto P = std::format("blk.{}.", l), Q = std::format("layer.{}.", l);
        if (q35) {
            put(Q + "attn_norm", P + "attn_norm.weight"); put(Q + "ffn_norm", P + "post_attention_norm.weight");
            if (moe) put_moe(l, P, Q);
            else { put(Q + "w_gate", P + "ffn_gate.weight"); put(Q + "w_up", P + "ffn_up.weight"); put(Q + "w_down", P + "ffn_down.weight"); }
            if (g.find(P + "attn_qkv.weight")) {   // Gated DeltaNet layer
                put_vrows(Q + "w_qkv", P + "attn_qkv.weight", 2 * lk_heads * lk_dim, lv_dim);
                put_vrows(Q + "w_z", P + "attn_gate.weight", 0, lv_dim);
                put_vrows(Q + "w_a", P + "ssm_alpha.weight", 0, 1);
                put_vrows(Q + "w_b", P + "ssm_beta.weight", 0, 1);
                put_vrows(Q + "conv_w", P + "ssm_conv1d.weight", 2 * lk_heads * lk_dim, lv_dim);
                put_vrows(Q + "dt_bias", g.find(P + "ssm_dt.bias") ? P + "ssm_dt.bias" : P + "ssm_dt", 0, 1);
                put_vrows(Q + "a_neg", P + "ssm_a", 0, 1);
                put(Q + "ssm_norm", P + "ssm_norm.weight");
                put_vcols(Q + "wo", P + "ssm_out.weight");
            } else {                                // gated full-attention layer
                put(Q + "wq", P + "attn_q.weight"); put(Q + "q_norm", P + "attn_q_norm.weight");
                put(Q + "wk", P + "attn_k.weight"); put(Q + "k_norm", P + "attn_k_norm.weight");
                put(Q + "wv", P + "attn_v.weight"); put(Q + "wo", P + "attn_output.weight");
            }
            continue;
        }
        put(Q + "attn_norm", P + "attn_norm.weight"); put(Q + "ffn_norm", P + "ffn_norm.weight");
        put(Q + "wq", P + "attn_q.weight"); put(Q + "wk", P + "attn_k.weight"); put(Q + "wv", P + "attn_v.weight");
        if (qwen2) { put(Q + "bq", P + "attn_q.bias"); put(Q + "bk", P + "attn_k.bias"); put(Q + "bv", P + "attn_v.bias"); }
        if (tname == "qwen3" || arch == "qwen3moe") { put(Q + "q_norm", P + "attn_q_norm.weight"); put(Q + "k_norm", P + "attn_k_norm.weight"); }
        put(Q + "wo", P + "attn_output.weight");
        if (moe) { put_moe(l, P, Q); continue; }
        put(Q + "w_gate", P + "ffn_gate.weight");
        put(Q + "w_up", P + "ffn_up.weight"); put(Q + "w_down", P + "ffn_down.weight");
    }
    put("out_norm", "output_norm.weight");
    put("lm_head", g.find("output.weight") ? "output.weight" : "token_embd.weight");

    // vision tower and multi-token-prediction heads are not part of the text model: never copied
    auto skipped = [&](const std::string& n) { return q35 && (n.starts_with("v.") || n.starts_with("mm.") || n.starts_with("mtp.")); };
    // Every source tensor is accounted for: consumed, part of a skipped tower, a known buffer, or reported (with its GGUF
    // type). Under allow_unused the leftovers stay addressable as extra.<name>; a leftover in a type FORGE cannot load
    // is left out and noted.
    for (auto& t : g.tensors())
        if (!used.count(t.name) && !skipped(t.name) && !ignored_tensor(t.name, opt))
            rep.unused.push_back(t.name + " " + shape_str(t.shape) + (gguf_file::loadable(t.type) ? "" : " (" + gguf_file::type_name(t.type) + ": type not supported)"));
    check_unused(path, tname, rep.unused, opt);
    for (auto& t : g.tensors())
        if (!used.count(t.name) && !skipped(t.name) && !ignored_tensor(t.name, opt)) {
            if (gguf_file::loadable(t.type)) { auto s = fetch(t); store("extra." + t.name, t.shape, s, s.bytes); }
            else rep.notes.push_back(std::format("left out unused tensor {}: ggml type {} is not supported", t.name, gguf_file::type_name(t.type)));
        }
    check_template_tensors(path, tname, bp, w.entries(), opt);

    // tokenizer.json synthesised from the GGUF vocabulary (tokenizer_import.hpp): tokenizer.ggml.model gpt2 (byte-level
    // BPE split by the tokenizer.ggml.pre regex), llama (SentencePiece BPE), t5 (Unigram), bert (WordPiece), with the
    // token types as added tokens. The chat template gets its own section.
    {
        std::string warn;
        if (auto tj = gguf_tokenizer_json(g, &warn); !tj.empty()) w.set_section("tokenizer", tj);
        if (!warn.empty()) std::fprintf(stderr, "warning: %s: %s\n", path.c_str(), warn.c_str());
        if (auto* ct = g.get("tokenizer.chat_template"); ct && ct->is_str() && !ct->str().empty()) w.set_section("chat_template", ct->str());
    }
    auto num_or = [&](const char* k, double d) { auto* v = g.get(k); return v ? v->num() : d; };
    json::object man{{"format", "fpk"}, {"forge_format_version", double(modelfile::format_version)}, {"architecture", arch},
                     {"source", "gguf"}, {"template", tname}, {"session", bp.session},
                     {"bos_token_id", num_or("tokenizer.ggml.bos_token_id", -1)}, {"eos_token_id", num_or("tokenizer.ggml.eos_token_id", -1)},
                     {"max_position_embeddings", Kor("context_length", 2048)}};
    // Whether prompts start with BOS: GGUF says so explicitly (Qwen-family files set it to false).
    if (auto* ab = g.get("tokenizer.ggml.add_bos_token"); ab && ab->is_bool()) man.emplace_back("add_bos", ab->b());
    if (auto* ae = g.get("tokenizer.ggml.add_eos_token"); ae && ae->is_bool()) man.emplace_back("add_eos", ae->b());
    if (!xi.empty()) { man.emplace_back("forge_format_minor", double(modelfile::format_minor)); w.set_section("experts", xi.dump()); }
    {   // tensor types: what the GGUF stored, and what the import changed
        json::object ty; for (auto& [k, n] : seen) ty.emplace_back(k, double(n));
        man.emplace_back("gguf_tensor_types", ty);
        if (!transcoded.empty()) {
            json::array a;
            for (auto& [k, n] : transcoded) {
                a.push_back(json::object{{"from", k}, {"to", "f32"}, {"tensors", double(n)}});
                rep.notes.push_back(std::format("transcoded {} {} tensor(s) to f32 (FORGE has no codec with that layout)", n, k));
            }
            man.emplace_back("transcoded", a);
        }
        if (!rq.empty()) {
            json::object fr; for (auto& [k, n] : requant_from) fr.emplace_back(k, double(n));
            man.emplace_back("codec", rq);
            man.emplace_back("requantised", json::object{{"codec", rq}, {"emb_codec", rq_emb}, {"tensors", double(requantised)}, {"from", fr}, {"double_quantised", double(double_q)}});
            rep.notes.push_back(std::format("re-encoded {} weight tensor(s) as {} (token embedding: {})", requantised, rq, rq_emb));
            if (double_q) rep.notes.push_back(std::format("warning: {} tensor(s) were already quantised and have been quantised a second time: the error of both formats adds up. "
                                                          "Import the original (unquantised) weights with --codec {} for best quality", double_q, rq));
        }
    }
    {   // licence: general.license (identifier), general.license.name, general.license.link
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
