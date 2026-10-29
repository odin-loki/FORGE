#pragma once
#include <forge/core/mapped.hpp>
// GGUF (v2/v3) reader and GGUF -> .fpk importer. Quantised blocks (Q8_0, Q4_K) are copied verbatim:
// FORGE's codecs use the GGUF block layouts, so no re-quantisation error is introduced.
#include <forge/import/hf.hpp>
#include <set>

namespace forge::import {

class gguf_file {
public:
    struct tinfo { std::string name; shape_t shape; std::uint32_t type = 0; std::uint64_t offset = 0; };
    explicit gguf_file(const std::string& path) : path_(path) {
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
            std::size_t align = std::size_t(get("general.alignment") ? get("general.alignment")->num() : 32);
            data_ = (p_ + align - 1) / align * align;
        } catch (...) { forge::unmap_readonly(map_, size_); map_ = nullptr; throw; }   // no dtor on a throwing ctor
    }
    gguf_file(const gguf_file&) = delete;
    ~gguf_file() { forge::unmap_readonly(map_, size_); }
    const json::value* get(std::string_view k) const { for (auto& [kk, v] : kv_) if (kk == k) return &v; return nullptr; }
    const json::value& need(std::string_view k) const { if (auto* v = get(k)) return *v; raise(errc::not_found, "{}: missing GGUF key {}", path_, k); }
    const std::vector<tinfo>& tensors() const { return tensors_; }
    const tinfo* find(const std::string& n) const { for (auto& t : tensors_) if (t.name == n) return &t; return nullptr; }
    static const char* codec_of(std::uint32_t t) {
        switch (t) { case 0: return "f32"; case 1: return "f16"; case 8: return "q8_0"; case 12: return "q4_k"; case 30: return "bf16"; default: return nullptr; }
    }
    std::span<const std::uint8_t> bytes(const tinfo& t) const {
        auto* c = codec_of(t.type);
        if (!c) raise(errc::unsupported, "tensor {}: ggml type {} not supported (supported: F32 F16 BF16 Q8_0 Q4_K)", t.name, t.type);
        for (auto d : t.shape) if (d < 0 || d > (i64(1) << 40)) raise(errc::format_error, "tensor {}: bad shape", t.name);
        std::size_t n = flow::codecs::find(c)->bytes(std::size_t(numel(t.shape)));
        if (t.offset > size_ || n > size_ || data_ + t.offset + n > size_) raise(errc::format_error, "tensor {} out of bounds", t.name);
        return {map_ + data_ + t.offset, n};
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
    gguf_file g(path);
    auto arch = g.need("general.architecture").str();
    if (arch != "llama" && arch != "qwen2" && arch != "mistral")
        raise(errc::unsupported, "GGUF architecture '{}' has no template", arch);
    auto K = [&](const std::string& k) { return g.need(arch + "." + k).num(); };
    auto Kor = [&](const std::string& k, double d) { auto* v = g.get(arch + "." + k); return v ? v->num() : d; };
    double heads = K("attention.head_count"), hidden = K("embedding_length");
    const auto& toks = g.need("tokenizer.ggml.tokens").arr();
    json::object cfg{{"hidden_size", hidden}, {"num_hidden_layers", K("block_count")}, {"num_attention_heads", heads},
                     {"num_key_value_heads", Kor("attention.head_count_kv", heads)}, {"head_dim", Kor("attention.key_length", hidden / heads)},
                     {"intermediate_size", K("feed_forward_length")}, {"vocab_size", double(toks.size())},
                     {"rms_norm_eps", Kor("attention.layer_norm_rms_epsilon", 1e-6)}, {"rope_theta", Kor("rope.freq_base", 10000.0)},
                     // llama.cpp's converter permutes Q/K for llama so it can use interleaved (GPT-J) RoPE
                     {"rope_neox", arch == "llama" || arch == "mistral" ? 0.0 : 1.0}};
    std::string tname = opt.template_name.empty() ? "llama" : opt.template_name;
    auto bp_text = instantiate(read_file(template_dir() / (tname + ".fbp")), json::value(cfg));
    auto bp = pack::parse_blueprint(bp_text);

    pack::writer w;
    import_report rep{arch, tname, 0, 0};
    std::set<std::string> used;
    auto put = [&](const std::string& dst, const std::string& src) {
        used.insert(src);
        auto* t = g.find(src);
        if (!t) raise(errc::not_found, "{}: missing tensor {}", path, src);
        w.add_raw(dst, t->shape, gguf_file::codec_of(t->type) ? gguf_file::codec_of(t->type) : "?", g.bytes(*t));
        ++rep.tensors;
    };
    put("tok_emb", "token_embd.weight");
    for (i64 l = 0, L = i64(bp.param("n_layers")); l < L; ++l) {
        auto P = std::format("blk.{}.", l), Q = std::format("layer.{}.", l);
        put(Q + "attn_norm", P + "attn_norm.weight"); put(Q + "ffn_norm", P + "ffn_norm.weight");
        put(Q + "wq", P + "attn_q.weight"); put(Q + "wk", P + "attn_k.weight"); put(Q + "wv", P + "attn_v.weight");
        put(Q + "wo", P + "attn_output.weight"); put(Q + "w_gate", P + "ffn_gate.weight");
        put(Q + "w_up", P + "ffn_up.weight"); put(Q + "w_down", P + "ffn_down.weight");
    }
    put("out_norm", "output_norm.weight");
    put("lm_head", g.find("output.weight") ? "output.weight" : "token_embd.weight");

    for (auto& t : g.tensors())  // keep everything else (rope freqs, extra heads, ...) addressable
        if (!used.count(t.name) && gguf_file::codec_of(t.type)) { w.add_raw("extra." + t.name, t.shape, gguf_file::codec_of(t.type), g.bytes(t)); ++rep.tensors; }

    // tokenizer.json synthesised from the GGUF vocabulary (gpt2-style byte-level BPE)
    auto model = g.get("tokenizer.ggml.model") ? g.need("tokenizer.ggml.model").str() : std::string("gpt2");
    if (model == "gpt2") {
        json::object vocab; json::array added;
        const json::array* types = g.get("tokenizer.ggml.token_type") ? &g.need("tokenizer.ggml.token_type").arr() : nullptr;
        for (std::size_t i = 0; i < toks.size(); ++i) {
            vocab.emplace_back(toks[i].str(), double(i));
            if (types && (*types)[i].num() == 3) added.push_back(json::object{{"id", double(i)}, {"content", toks[i].str()}});
        }
        json::array merges; if (auto* m = g.get("tokenizer.ggml.merges")) merges = m->arr();
        json::object tj{{"added_tokens", added}, {"model", json::object{{"type", "BPE"}, {"vocab", vocab}, {"merges", merges}}}};
        w.set_section("tokenizer", json::dump(json::value(tj)));
    }
    auto num_or = [&](const char* k, double d) { auto* v = g.get(k); return v ? v->num() : d; };
    json::object man{{"format", "fpk"}, {"forge_format_version", double(pack::format_version)}, {"architecture", arch},
                     {"source", "gguf"}, {"template", tname}, {"session", bp.session},
                     {"bos_token_id", num_or("tokenizer.ggml.bos_token_id", -1)}, {"eos_token_id", num_or("tokenizer.ggml.eos_token_id", -1)},
                     {"max_position_embeddings", Kor("context_length", 2048)}};
    w.set_manifest(json::value(man));
    w.set_section("blueprint", bp_text);
    w.write(out);
    rep.bytes = std::filesystem::file_size(out);
    return rep;
}

} // namespace forge::import
