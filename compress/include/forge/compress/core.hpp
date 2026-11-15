// forge::compress core: mutable weights over a FORGE .fpk, a structural model spec, a decoder blueprint generator
// and a FORGE graph runner with activation taps.
//
// Everything that computes runs on FORGE: graphs are built by modelfile::build_graph from blueprints this file
// generates, compiled by graph::compile and executed by graph::executor on FORGE's CPU kernels. Weight nodes point
// straight at the f32 buffers held in WeightStore, so pruning a tensor in place is seen by the next run without
// recompiling (only shape changes need a new Runner).
#pragma once
#include <forge/compress/mapped.hpp>
#include <forge/graph/graph.hpp>
#include <forge/import/hf.hpp>
#include <forge/modelfile/blueprint.hpp>
#include <forge/modelfile/blueprint_bin.hpp>
#include <forge/modelfile/fpk.hpp>
#include <forge/modelfile/tokenizer.hpp>
#include <forge/task/engine.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <format>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace forge::compress {

using forge::i64;
using forge::shape_t;
using forge::tensor;
using forge::json::value;
using jobj = forge::json::object;

// ---------------------------------------------------------------------------------------------------- logging
inline double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
inline double log_t0() { static const double t0 = now_s(); return t0; }
template <class... A> void say(std::format_string<A...> f, A&&... a) {
    const double t0 = log_t0();
    std::fprintf(stderr, "[%8.1fs] %s\n", now_s() - t0, std::format(f, std::forward<A>(a)...).c_str());
    std::fflush(stderr);
}
[[noreturn]] inline void die(const std::string& m) { throw std::runtime_error(m); }

// ---------------------------------------------------------------------------------------------------- weights
// A mutable f32 tensor of a model being compressed. Like forge::tensor (shape, data), but `data` is a mapped_vector
// (mapped.hpp): big tensors live in scratch-file mappings, and a tensor loaded from a .fpk stays in the file (read
// in place for f32, decoded on first access otherwise) until it is first written. It converts to and from
// forge::tensor where a FORGE API needs one (a copy).
struct wtensor {
    shape_t shape;
    fbuf data = weights_buf();
    wtensor() = default;
    explicit wtensor(shape_t s, float fill = 0.f) : shape(std::move(s)) { data.assign(std::size_t(numel(shape)), fill); }
    wtensor(const tensor& t) : shape(t.shape) { data = t.data; }   // NOLINT: implicit, so tensors are stored as before
    static fbuf weights_buf() { fbuf b; b.set_use(scratch::use::weights); return b; }
    operator tensor() const { tensor r; r.shape = shape; r.data = data; return r; }   // NOLINT
    tensor_view view() { return {data.data(), shape}; }
    std::size_t size() const { return data.size(); }
};

// Every tensor of a model, as f32 and owned here (mutable). Names are FORGE names ("layer.3.wq").
struct WeightStore {
    std::map<std::string, wtensor> t;
    wtensor& at(const std::string& n) {
        auto it = t.find(n);
        if (it == t.end()) die("no tensor " + n);
        return it->second;
    }
    const wtensor& at(const std::string& n) const {
        auto it = t.find(n);
        if (it == t.end()) die("no tensor " + n);
        return it->second;
    }
    bool has(const std::string& n) const { return t.count(n) != 0; }
    i64 params() const { i64 s = 0; for (auto& [k, v] : t) s += forge::numel(v.shape); return s; }
    i64 nonzeros() const {
        i64 s = 0;
        for (auto& [k, v] : t) for (float x : v.data) s += x != 0.f;
        return s;
    }
};

// ---------------------------------------------------------------------------------------------------- spec
// The structure of a decoder-only LM, layer by layer. This is what compression edits; the blueprint is regenerated
// from it, so pruned models may have per-layer FFN widths, missing sublayers, linear replacements or dense/MoE mixes.
struct LayerSpec {
    bool attn = true;
    enum class Ffn { dense, moe, linear, none } ffn = Ffn::dense;
    i64 moe_k = 2;           // experts per token (moe)
    bool moe_norm = true;    // renormalise the top-k weights (moe)
    double moe_scale = 1.0;  // routed scaling factor (moe)
    double act_thresh = 0;   // > 0: FFN hidden activations with |a| < act_thresh are zeroed (activation sparsity)
};

struct ModelSpec {
    std::string family;  // "llama" | "qwen3" | "qwen2" (the attention flavour)
    bool qk_norm = false, qkv_bias = false;
    i64 d_model = 0, n_heads = 0, n_kv = 0, head_dim = 0, vocab = 0, kv_slots = 0;
    double eps = 1e-6, rope_base = 10000;
    std::string rope_neox = "1", rope_args, attn_args;
    int opset = 1;
    bool logit_bias = false;   // logits get + W.logit_bias (distribution correction)
    bool tied_head = false;    // the LM head reads W.tok_emb (tied embeddings stored once)
    bool head_mix = false;     // a d x d map before the head (a tied head's norm gain in a rotated basis)
    std::vector<LayerSpec> layers;
    // factored projections: "layer.L.X" computed as linear(linear(h, X_a), X_b) (X_a [r, in], X_b [out, r])
    std::set<std::string> lowrank;
    // query heads per layer (head pruning); absent = n_heads. Must stay a multiple of n_kv (GQA groups).
    std::map<i64, i64> heads;
    // tensor aliases: a weight name the blueprint references -> the stored tensor it reads (shared bases)
    std::map<std::string, std::string> alias;

    i64 heads_of(i64 l) const { auto it = heads.find(l); return it == heads.end() ? n_heads : it->second; }
    bool factored(i64 l, const char* x) const { return lowrank.count(std::format("layer.{}.{}", l, x)) != 0; }
    i64 n_layers() const { return i64(layers.size()); }
};

inline std::string fmt_num(double v) {
    if (v == double(i64(v)) && std::abs(v) < 1e15) return std::to_string(i64(v));
    return std::format("{:.17g}", v);
}

// Reads the spec from a blueprint imported by FORGE's llama / qwen2 / qwen3 / qwen3_moe templates, or written by
// emit_blueprint below (which records the per-layer structure in a "# compress-layers:" comment).
inline ModelSpec spec_from_blueprint(const std::string& text, const forge::modelfile::blueprint& bp,
                                     const std::string& template_name) {
    ModelSpec s;
    std::string act_line;
    s.d_model = i64(bp.param("d_model"));
    s.n_heads = i64(bp.param("n_heads"));
    s.n_kv = i64(bp.param("n_kv"));
    s.head_dim = i64(bp.param("head_dim"));
    s.vocab = i64(bp.param("vocab"));
    s.eps = bp.param("eps");
    s.rope_base = bp.param("rope.base");
    s.kv_slots = i64(bp.param("n_layers"));
    s.opset = bp.opset;
    std::smatch m;
    if (std::regex_search(text, m, std::regex(R"(neox: ([0-9]+)([^)]*)\))"))) { s.rope_neox = m[1]; s.rope_args = m[2]; }
    if (std::regex_search(text, m, std::regex(R"(groups: n_heads / n_kv([^)]*)\))"))) s.attn_args = m[1];
    s.qk_norm = text.find(".q_norm") != std::string::npos;
    s.qkv_bias = text.find(".bq") != std::string::npos;
    s.family = s.qk_norm ? "qwen3" : (s.qkv_bias ? "qwen2" : "llama");
    s.logit_bias = text.find("W.logit_bias") != std::string::npos;
    {
        std::regex lr(R"(W\.layer\[(\d+)\]\.(\w+)_a\))");
        for (auto it = std::sregex_iterator(text.begin(), text.end(), lr); it != std::sregex_iterator(); ++it)
            s.lowrank.insert(std::format("layer.{}.{}", (*it)[1].str(), (*it)[2].str()));
        std::smatch hm;
        std::regex al(R"(W\.layer\[(\d+)\]\.(\w+)_b\) *# *alias=([\w.]+))");
        for (auto it = std::sregex_iterator(text.begin(), text.end(), al); it != std::sregex_iterator(); ++it) {}
        if (std::regex_search(text, hm, std::regex(R"(# compress-alias: ([^\n]*))"))) {
            std::string A = hm[1];
            for (std::size_t p = 0; p < A.size();) { auto q = A.find(' ', p); if (q == std::string::npos) q = A.size(); auto kv = A.substr(p, q - p); auto e = kv.find('='); if (e != std::string::npos) s.alias[kv.substr(0, e)] = kv.substr(e + 1); p = q + 1; }
        }
        if (std::regex_search(text, hm, std::regex(R"(# compress-act: ([^\n]*))"))) {
            act_line = hm[1];
        }
        if (std::regex_search(text, hm, std::regex(R"(# compress-heads: ([^\n]*))"))) {
            std::string H = hm[1]; i64 l = 0;
            for (std::size_t p = 0; p < H.size();) { auto q = H.find(' ', p); if (q == std::string::npos) q = H.size(); if (q > p) { i64 n = std::stoll(H.substr(p, q - p)); if (n != s.n_heads) s.heads[l] = n; ++l; } p = q + 1; }
        }
    }
    s.head_mix = text.find("W.head_mix") != std::string::npos;
    s.tied_head = text.find("W.out_norm, eps), W.tok_emb)") != std::string::npos || text.find("W.head_mix), W.tok_emb)") != std::string::npos;
    if (std::regex_search(text, m, std::regex(R"(# compress-layers: ([^\n]*))"))) {
        std::string L = m[1];
        for (std::size_t p = 0; p < L.size();) {
            auto q = L.find(' ', p);
            if (q == std::string::npos) q = L.size();
            std::string tok = L.substr(p, q - p);
            p = q + 1;
            if (tok.empty()) continue;
            // token: <a|-><d|m|l|-><k>  e.g. "ad", "am2", "-l", "a-"
            LayerSpec ls;
            ls.attn = tok[0] == 'a';
            switch (tok[1]) {
                case 'd': ls.ffn = LayerSpec::Ffn::dense; break;
                case 'm': ls.ffn = LayerSpec::Ffn::moe; ls.moe_k = std::stoll(tok.substr(2)); break;
                case 'l': ls.ffn = LayerSpec::Ffn::linear; break;
                default: ls.ffn = LayerSpec::Ffn::none;
            }
            s.layers.push_back(ls);
        }
        if (std::regex_search(text, m, std::regex(R"(norm: ([0-9.]+), scale: ([0-9.e+-]+))"))) {
            for (auto& l : s.layers) { l.moe_norm = std::stod(m[1]) != 0; l.moe_scale = std::stod(m[2]); }
        }
        if (!act_line.empty()) {
            std::size_t l = 0;
            for (std::size_t p = 0; p < act_line.size() && l < s.layers.size();) { auto q = act_line.find(' ', p); if (q == std::string::npos) q = act_line.size(); if (q > p) s.layers[l++].act_thresh = std::stod(act_line.substr(p, q - p)); p = q + 1; }
        }
        return s;
    }
    const bool moe = template_name == "qwen3_moe" || template_name == "mixtral";
    for (i64 l = 0; l < s.kv_slots; ++l) {
        LayerSpec ls;
        if (moe) {
            ls.ffn = LayerSpec::Ffn::moe;
            ls.moe_k = i64(bp.param("n_active"));
            ls.moe_norm = bp.param_or("norm_topk", 1) != 0;
            ls.moe_scale = bp.param_or("routed_scale", 1);
        }
        s.layers.push_back(ls);
    }
    if (template_name != "llama" && template_name != "qwen2" && template_name != "qwen3" && template_name != "qwen3_moe" && template_name != "compressed")
        die("unsupported template " + template_name + " (llama, qwen2, qwen3, qwen3_moe)");
    return s;
}

// What the generated graph computes.
struct EmitOptions {
    enum class Mode { lm, block } mode = Mode::lm;  // lm: tokens -> logits; block: one layer, xin -> xout
    i64 block = 0;                                  // the layer (block mode)
    bool keep_all_layers_state = true;
};

// Writes a decoder blueprint for the spec, one block per layer (unrolled: each layer may differ).
inline std::string emit_blueprint(const ModelSpec& s, const EmitOptions& eo = {}) {
    std::string o;
    std::string layers;
    for (auto& l : s.layers) {
        layers += l.attn ? 'a' : '-';
        switch (l.ffn) {
            case LayerSpec::Ffn::dense: layers += 'd'; break;
            case LayerSpec::Ffn::moe: layers += std::format("m{}", l.moe_k); break;
            case LayerSpec::Ffn::linear: layers += 'l'; break;
            case LayerSpec::Ffn::none: layers += '-'; break;
        }
        layers += ' ';
    }
    const bool block = eo.mode == EmitOptions::Mode::block;
    o += std::format("# Generated by forge compress from a {} decoder; one block per layer.\n", s.family);
    o += "# compress-layers: " + layers + "\n";
    if (!s.heads.empty()) { o += "# compress-heads:"; for (i64 l = 0; l < s.n_layers(); ++l) o += std::format(" {}", s.heads_of(l)); o += "\n"; }
    if (!s.alias.empty()) { o += "# compress-alias:"; for (auto& [k, v] : s.alias) o += " " + k + "=" + v; o += "\n"; }
    {
        bool any = false; for (auto& L : s.layers) any |= L.act_thresh > 0;
        if (any) { o += "# compress-act:"; for (auto& L : s.layers) o += " " + fmt_num(L.act_thresh); o += "\n"; }
    }
    o += std::format("blueprint \"decoder-lm\" opset {}\n\n", s.opset);
    o += std::format("params {{\n  d_model: {}  n_layers: {}  n_heads: {}  n_kv: {}  head_dim: {}\n  vocab: {}  eps: {}\n  rope: {{ base: {} }}\n}}\n\n",
                     s.d_model, block ? 1 : s.kv_slots, s.n_heads, s.n_kv, s.head_dim, s.vocab, fmt_num(s.eps), fmt_num(s.rope_base));
    o += "state kv[n_layers] : kv_cache(heads: n_kv, dim: head_dim)\n\n";
    if (block) o += "input xin       : f32[seq, d_model]\n";
    else o += "input tokens    : f32[seq]\n";
    o += "input positions : f32[seq]\n\ngraph {\n";
    o += block ? "  x = xin + 0\n" : "  x = embed(W.tok_emb, tokens)\n";
    auto wref = [&](const std::string& name) {   // W.<path> of a tensor, through an alias
        auto it = s.alias.find(name);
        const std::string& n = it == s.alias.end() ? name : it->second;
        auto d = n.find('.', 6);
        if (n.rfind("layer.", 0) == 0 && d != std::string::npos) return std::format("W.layer[{}].{}", n.substr(6, d - 6), n.substr(d + 1));
        return "W." + n;
    };
    auto lin = [&](const std::string& x, i64 l, const char* n) {
        return s.factored(l, n) ? std::format("linear(linear({}, {}), {})", x, wref(std::format("layer.{}.{}_a", l, n)), wref(std::format("layer.{}.{}_b", l, n)))
                                : std::format("linear({}, {})", x, wref(std::format("layer.{}.{}", l, n)));
    };
    auto emit_layer = [&](i64 l, i64 slot) {
        const auto& L = s.layers[std::size_t(l)];
        const std::string w = std::format("W.layer[{}]", l);
        o += std::format("  # layer {}\n", l);
        if (L.attn) {
            o += std::format("  h = rms_norm(x, {}.attn_norm, eps)\n", w);
            auto proj = [&](const char* n, const char* b) {
                const std::string li = lin("h", l, n);
                return s.qkv_bias ? std::format("{} + {}.{}", li, w, b) : li;
            };
            const i64 nh = s.heads_of(l);
            const std::string rope = std::format(", positions, base: rope.base, neox: {}{})", s.rope_neox, s.rope_args);
            if (s.qk_norm) {
                o += std::format("  q = rope(rms_norm(heads({}, {}, head_dim), {}.q_norm, eps){}\n", proj("wq", "bq"), nh, w, rope);
                o += std::format("  k = rope(rms_norm(heads({}, n_kv, head_dim), {}.k_norm, eps){}\n", proj("wk", "bk"), w, rope);
            } else {
                o += std::format("  q = rope(heads({}, {}, head_dim){}\n", proj("wq", "bq"), nh, rope);
                o += std::format("  k = rope(heads({}, n_kv, head_dim){}\n", proj("wk", "bk"), rope);
            }
            o += std::format("  v = heads({}, n_kv, head_dim)\n", proj("wv", "bv"));
            o += std::format("  kk, vv = kv[{}].append(k, v)\n", slot);
            o += std::format("  a = attention(q, kk, vv, causal: true, groups: {}{})\n", nh / s.n_kv, s.attn_args);
            o += std::format("  x = x + {}\n", lin(std::format("merge(a, {} * head_dim)", nh), l, "wo"));
        }
        switch (L.ffn) {
            case LayerSpec::Ffn::dense:
                o += std::format("  h = rms_norm(x, {}.ffn_norm, eps)\n", w);
                if (L.act_thresh > 0) {   // activation sparsity: a * step(|a| - t), step as r / (r + 1e-6) with r = relu(|a| - t)
                    o += std::format("  a = silu({}) * {}\n", lin("h", l, "w_gate"), lin("h", l, "w_up"));
                    o += std::format("  r = relu(abs(a) - {})\n", fmt_num(L.act_thresh));
                    o += std::format("  x = x + {}\n", lin("a * (r / (r + 0.000001))", l, "w_down"));
                } else o += std::format("  x = x + {}\n", lin(std::format("silu({}) * {}", lin("h", l, "w_gate"), lin("h", l, "w_up")), l, "w_down"));
                break;
            case LayerSpec::Ffn::moe:
                o += std::format("  h = rms_norm(x, {}.ffn_norm, eps)\n", w);
                o += std::format("  x = x + moe_ffn(h, linear(h, {0}.router), {0}.experts.w_gate, {0}.experts.w_up, {0}.experts.w_down, k: {1}, norm: {2}, scale: {3})\n",
                                 w, L.moe_k, L.moe_norm ? 1 : 0, fmt_num(L.moe_scale));
                break;
            case LayerSpec::Ffn::linear:   // low-rank linear replacement of a removed FFN (fitted by least squares)
                o += std::format("  h = rms_norm(x, {}.ffn_norm, eps)\n", w);
                o += std::format("  x = x + linear(linear(h, {0}.lin_a), {0}.lin_b)\n", w);
                break;
            case LayerSpec::Ffn::none: break;
        }
    };
    if (block) emit_layer(eo.block, 0);
    else for (i64 l = 0; l < s.n_layers(); ++l) emit_layer(l, l);
    if (block) o += "  xout = x + 0\n}\n\noutput xout : f32[seq, d_model]\nsession one_shot\n";
    else {
        std::string hn = "rms_norm(last(x, seq), W.out_norm, eps)";
        if (s.head_mix) hn = "linear(" + hn + ", W.head_mix)";
        o += std::format("  logits = linear({}, W.{}){}\n}}\n\noutput logits : f32[1, vocab]\nsession generative_sequence\n",
                         hn, s.tied_head ? "tok_emb" : "lm_head", s.logit_bias ? " + W.logit_bias" : "");
    }
    return o;
}

// The tensors one layer uses (for slicing a WeightStore per block).
inline std::vector<std::string> layer_tensors(const ModelSpec& s, i64 l) {
    std::vector<std::string> v;
    const auto& L = s.layers[std::size_t(l)];
    auto p0 = [&](const char* n) { v.push_back(std::format("layer.{}.{}", l, n)); };
    auto p = [&](const char* n) {   // a factored projection is stored as its two factors
        if (s.factored(l, n)) { v.push_back(std::format("layer.{}.{}_a", l, n)); v.push_back(std::format("layer.{}.{}_b", l, n)); }
        else p0(n);
    };
    if (L.attn) {
        p0("attn_norm"); p("wq"); p("wk"); p("wv"); p("wo");
        if (s.qk_norm) { p0("q_norm"); p0("k_norm"); }
        if (s.qkv_bias) { p0("bq"); p0("bk"); p0("bv"); }
    }
    switch (L.ffn) {
        case LayerSpec::Ffn::dense: p0("ffn_norm"); p("w_gate"); p("w_up"); p("w_down"); break;
        case LayerSpec::Ffn::moe: p0("ffn_norm"); p0("router"); p0("experts.w_gate"); p0("experts.w_up"); p0("experts.w_down"); break;
        case LayerSpec::Ffn::linear: p0("ffn_norm"); p0("lin_a"); p0("lin_b"); break;
        case LayerSpec::Ffn::none: break;
    }
    return v;
}

inline void json_set(forge::json::value& o, const std::string& k, forge::json::value v) {
    for (auto& [kk, vv] : o.obj_mut()) if (kk == k) { vv = std::move(v); return; }
    o.obj_mut().push_back({k, std::move(v)});
}

// ---------------------------------------------------------------------------------------------------- model
// A model being compressed: spec + weights + the parts of the source file that pass through (tokenizer, manifest...).
struct Model {
    std::string src_path;
    ModelSpec spec;
    WeightStore w;
    forge::json::value manifest;
    std::map<std::string, std::string> passthrough;  // sections copied verbatim (tokenizer, chat_template, license)
    std::optional<forge::modelfile::tokenizer> tok;
    std::map<std::string, std::string> codec;        // per-tensor storage codec chosen by a quant stage

    // One tensor of a .fpk, lazily (FORGE_COMPRESS_LAZY=0: decoded now): an aligned f32 tensor is read in place in
    // the reader's mapping until it is first written; other codecs are decoded on first access. Encrypted files are
    // decoded now (their plaintext is not kept in the reader).
    static wtensor load_tensor(const std::shared_ptr<forge::modelfile::reader>& rd, const std::string& name) {
        const auto& e = rd->entry(name);
        const auto* cd = forge::graph::codecs::find(e.codec);
        if (!cd) die("tensor " + name + ": unknown codec " + e.codec);
        wtensor t; t.shape = e.shape;
        const std::size_t n = std::size_t(forge::numel(e.shape));
        if (rd->encrypted() || rd->locked()) { auto x = rd->load(name); t.data = x.data; return t; }
        auto b = rd->tensor_bytes(name);
        if (os::getenv("FORGE_COMPRESS_LAZY", "1") == "0") { t.data.resize(n); cd->decode(b.data(), n, t.data.data()); return t; }
        fbuf::lazy_source src;
        src.owner = rd;
        if (std::string_view(e.codec) == "f32" && reinterpret_cast<std::uintptr_t>(b.data()) % alignof(float) == 0) src.borrow = reinterpret_cast<const float*>(b.data());
        const std::uint8_t* p = b.data();
        src.fill = [cd, p, n](float* dst) { cd->decode(p, n, dst); };
        t.data = fbuf::lazy(n, std::move(src));
        return t;
    }

    static Model load(const std::string& path) {
        Model m;
        m.src_path = path;
        auto rd = std::make_shared<forge::modelfile::reader>(path);   // kept alive by the lazy tensors that read it
        const auto& r = *rd;
        m.manifest = r.manifest();
        std::string text = forge::modelfile::blueprint_text(r.section_text("blueprint"));
        auto bp = forge::modelfile::parse_blueprint(text);
        std::string tpl = m.manifest.find("template") && m.manifest["template"].is_str() ? m.manifest["template"].str() : "compressed";
        if (text.find("# compress-layers:") != std::string::npos) tpl = "compressed";
        m.spec = spec_from_blueprint(text, bp, tpl);
        for (auto& [name, e] : r.tensors()) m.w.t.emplace(name, load_tensor(rd, name));
        if (!m.w.has("lm_head") && m.w.has("tok_emb")) m.w.t["lm_head"] = m.w.at("tok_emb");   // tied head stored once
        for (auto& [name, sec] : r.sections()) {
            if (name == "tokenizer" || name == "chat_template" || name == "license" || name == "tokenizer_json")
                m.passthrough[name] = std::string(r.section_text(name));
        }
        if (m.passthrough.count("tokenizer")) m.tok.emplace(m.passthrough["tokenizer"]);
        else if (m.passthrough.count("tokenizer_json")) m.tok.emplace(m.passthrough["tokenizer_json"]);
        return m;
    }

    // Writes a .fpk FORGE can run: the regenerated blueprint, its state section, the source's tokenizer and manifest.
    // `codec_for(name, tensor)` picks each tensor's codec (default f32; norms, routers and 1-D tensors stay f32).
    void save(const std::string& path, const std::function<std::string(const std::string&, const wtensor&)>& codec_for = {}) const {
        ModelSpec sp = spec;
        // identical LM head and embedding (tied models, also after rotation / quantisation): store one tensor
        sp.tied_head = w.has("lm_head") && w.has("tok_emb") && w.at("lm_head").shape == w.at("tok_emb").shape && w.at("lm_head").data == w.at("tok_emb").data
                       && (codec.count("lm_head") ? codec.at("lm_head") : std::string()) == (codec.count("tok_emb") ? codec.at("tok_emb") : std::string());
        std::string text = emit_blueprint(sp);
        auto bp = forge::modelfile::parse_blueprint(text);
        forge::modelfile::stream_writer wr(path);
        wr.set_alignment(4096);
        std::set<std::string> used;
        for (i64 l = 0; l < spec.n_layers(); ++l) for (auto& n : layer_tensors(spec, l)) { auto a = spec.alias.find(n); used.insert(a == spec.alias.end() ? n : a->second); }
        used.insert("tok_emb"); used.insert("out_norm");
        if (!sp.tied_head) used.insert("lm_head");
        if (sp.logit_bias) used.insert("logit_bias");
        if (sp.head_mix) used.insert("head_mix");
        for (auto& n : used) {
            const auto& t = w.at(n);
            std::string c = "f32";
            const bool plain = t.shape.size() >= 2 && n.find("norm") == std::string::npos && n.find("router") == std::string::npos && n != "head_mix";
            if (auto it = codec.find(n); it != codec.end()) c = it->second;      // quantised by a stage: exact in that codec
            else if (codec_for && plain) c = codec_for(n, t);
            wr.add_tensor(n, t.shape, t.data.span(), c);
        }
        wr.set_section("blueprint", text);
        for (auto& [k, v] : passthrough) wr.set_section(k, v);
        forge::import::add_state_section(wr, bp);
        auto man = manifest;
        json_set(man, "template", forge::json::value("compressed"));
        json_set(man, "compressed", forge::json::value(true));
        wr.set_manifest(man);
        wr.finish();
    }

    std::vector<int> encode(const std::string& text) const {
        if (!tok) die("model has no tokenizer");
        return tok->encode(text);
    }
};

// ---------------------------------------------------------------------------------------------------- runner
// A compiled FORGE graph over a Model's live weights, with optional taps: the f32 input of selected linear layers
// (`linear(x, W.<name>)` = matmul_t(x, W)) and the two inputs of moe_ffn ops, exposed as extra graph outputs.
struct Runner {
    struct Tap { std::string weight; std::string out; };
    ModelSpec spec;
    std::string text;
    forge::graph::state_store st;
    std::unique_ptr<forge::graph::plan> plan;
    std::unique_ptr<forge::graph::executor> ex;
    std::map<std::string, std::string> tap_of;   // weight name -> output name ("tap:<group>")
    i64 seq = 0;
    bool block = false;
    std::string out_name;

    // tap_filter(weight name) -> group name ("" = no tap). Weights that share an input share a group.
    Runner(const ModelSpec& s, WeightStore& w, i64 seq_len, const EmitOptions& eo, bool all_rows,
           const std::function<std::string(const std::string&)>& tap_filter = {})
        : spec(s), seq(seq_len), block(eo.mode == EmitOptions::Mode::block) {
        text = emit_blueprint(s, eo);
        auto bp = forge::modelfile::parse_blueprint(text);
        forge::modelfile::build_options bo;
        bo.dims["seq"] = seq_len;
        bo.all_rows = all_rows;
        // Block mode emits layer `eo.block` as slot 0 but keeps its tensor names (W.layer[l]).
        bo.weight = [&w](const std::string& name) -> std::pair<shape_t, const float*> {
            // the blueprint already names the aliased (stored) tensor. Mutable access: a lazily loaded tensor gets
            // its own storage now, so in-place edits made while this graph lives are seen by it
            auto& t = w.at(name);
            return {t.shape, t.data.data()};
        };
        auto g = forge::modelfile::build_graph(bp, bo);
        out_name = bp.outputs.front().name;
        if (tap_filter) {
            std::map<forge::graph::value_id, std::string> done;
            const auto nodes = g.nodes();   // copy: output() appends to the graph
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                const auto& n = nodes[i];
                if (n.kind != forge::graph::node_kind::op) continue;
                const bool weighted = (n.op == forge::ops::op::matmul_t || n.op == forge::ops::op::rms_norm) && n.inputs.size() >= 2;
                if (weighted && g.at(n.inputs[1]).kind == forge::graph::node_kind::weight) {
                    const std::string wn = g.at(n.inputs[1]).name;
                    std::string grp = tap_filter(wn);
                    if (grp.empty()) continue;
                    const std::string on = "tap:" + grp;
                    if (!done.count(n.inputs[0])) { g.output(on, n.inputs[0]); done[n.inputs[0]] = on; }
                    tap_of[wn] = done[n.inputs[0]];
                } else if (n.op == forge::ops::op::moe_ffn) {
                    const std::string wn = g.at(n.inputs[2]).name;   // layer.L.experts.w_gate
                    std::string grp = tap_filter(wn);
                    if (grp.empty()) continue;
                    g.output("tap:" + grp, n.inputs[0]);
                    g.output("tap:" + grp + ".router", n.inputs[1]);
                    tap_of[wn] = "tap:" + grp;
                }
            }
        }
        st.capacity = std::max<i64>(seq_len, 16);
        forge::graph::pass_options po;
        plan = forge::graph::compile(std::move(g), &st, po);
        ex = std::make_unique<forge::graph::executor>(*plan, &st);
    }

    // One sequence from the start (the KV state is cleared first).
    void run_tokens(std::span<const int> ids) {
        if (i64(ids.size()) != seq) die(std::format("runner compiled for {} rows, got {}", seq, ids.size()));
        st.reset();
        std::vector<float> tk(ids.begin(), ids.end());
        std::vector<float> ps(static_cast<std::size_t>(seq));
        for (i64 i = 0; i < seq; ++i) ps[std::size_t(i)] = float(i);
        ex->set_input("tokens", tk);
        ex->set_input("positions", ps);
        ex->run();
    }
    void run_hidden(std::span<const float> x) {
        if (i64(x.size()) != seq * spec.d_model) die("run_hidden: bad size");
        st.reset();
        std::vector<float> ps(static_cast<std::size_t>(seq));
        for (i64 i = 0; i < seq; ++i) ps[std::size_t(i)] = float(i);
        ex->set_input("xin", x);
        ex->set_input("positions", ps);
        ex->run();
    }
    std::span<const float> output() const { auto v = ex->output_view(out_name); return {v.data, v.size()}; }
    std::span<const float> tap(const std::string& out) const { auto v = ex->output_view(out); return {v.data, v.size()}; }
    bool has_tap(const std::string& out) const {
        for (auto& [k, v] : tap_of) if (v == out) return true;
        return false;
    }
};

}  // namespace forge::compress
