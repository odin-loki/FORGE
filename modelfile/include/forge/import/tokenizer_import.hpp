#pragma once
// Tokenizer sources that are not a ready tokenizer.json:
//   - a SentencePiece .model protobuf (tokenizer.model / spiece.model next to a Hugging Face checkpoint),
//   - GGUF metadata (tokenizer.ggml.*),
//   - a WordPiece vocab.txt (BERT checkpoints saved without tokenizer.json),
// all turned into the tokenizer.json text the .fpk "tokenizer" section holds (modelfile/tokenizer.hpp reads it), and
// the chat template that travels with either (tokenizer_config.json / chat_template.jinja / tokenizer.chat_template).
#include <forge/json/json.hpp>
#include <forge/modelfile/tokenizer.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace forge::import {
namespace tokimp {

inline std::string base64(std::string_view s) {
    static const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o; o.reserve((s.size() + 2) / 3 * 4);
    for (std::size_t i = 0; i < s.size(); i += 3) {
        unsigned n = unsigned((unsigned char)s[i]) << 16;
        if (i + 1 < s.size()) n |= unsigned((unsigned char)s[i + 1]) << 8;
        if (i + 2 < s.size()) n |= unsigned((unsigned char)s[i + 2]);
        o += a[n >> 18]; o += a[(n >> 12) & 63]; o += i + 1 < s.size() ? a[(n >> 6) & 63] : '='; o += i + 2 < s.size() ? a[n & 63] : '=';
    }
    return o;
}

// What a SentencePiece model says about itself, from either source.
struct sp_model {
    struct piece { std::string text; double score = 0; int type = modelfile::tokenizer::type_normal; };
    std::vector<piece> pieces;
    bool unigram = false, byte_fallback = false, dummy_prefix = true, remove_extra_ws = false;
    int unk_id = -1;
    std::string charsmap;   // precompiled normalization rules (may be empty)
};

inline json::value replace_obj(const char* kind, const char* pat, const char* with) {
    return json::object{{"type", "Replace"}, {"pattern", json::object{{kind, pat}}}, {"content", with}};
}
inline double finite(double v) { return std::isfinite(v) ? v : -1e9; }

// tokenizer.json for a SentencePiece model: SentencePiece's own normalization (charsmap, whitespace collapsing, dummy
// prefix, "▁" for spaces) as a normalizer chain, no pre-tokenizer -- the model sees the whole text, as in
// SentencePiece -- and control / user-defined pieces as added tokens so that "<s>" or "[INST]" written in a prompt
// becomes its id (SentencePiece itself would spell it out; llama.cpp and transformers parse it).
inline json::value sentencepiece_json(const sp_model& m) {
    using T = modelfile::tokenizer;
    json::array vocab, types, added, norm, dec;
    for (std::size_t i = 0; i < m.pieces.size(); ++i) {
        auto& p = m.pieces[i];
        vocab.push_back(json::array{p.text, finite(p.score)});
        types.push_back(double(p.type));
        if (p.type == T::type_control || p.type == T::type_unknown || p.type == T::type_user_defined)
            added.push_back(json::object{{"id", double(i)}, {"content", p.text}, {"special", p.type != T::type_user_defined}, {"normalized", false},
                                         {"single_word", false}, {"lstrip", false}, {"rstrip", false}});
    }
    if (!m.charsmap.empty()) norm.push_back(json::object{{"type", "Precompiled"}, {"precompiled_charsmap", base64(m.charsmap)}, {"forge_match", "longest"}});
    if (m.remove_extra_ws) { norm.push_back(replace_obj("Regex", "\\A +| +\\z", "")); norm.push_back(replace_obj("Regex", " {2,}", " ")); }
    if (m.dummy_prefix) norm.push_back(json::object{{"type", "Prepend"}, {"prepend", "\xE2\x96\x81"}});
    norm.push_back(replace_obj("String", " ", "\xE2\x96\x81"));
    dec.push_back(replace_obj("String", "\xE2\x96\x81", " "));
    dec.push_back(json::object{{"type", "ByteFallback"}}); dec.push_back(json::object{{"type", "Fuse"}});
    if (m.dummy_prefix) dec.push_back(json::object{{"type", "Strip"}, {"content", " "}, {"start", 1.0}, {"stop", 0.0}});
    return json::object{{"forge_tokenizer_version", double(modelfile::tokenizer_format_version)}, {"added_tokens", added},
                        {"normalizer", json::object{{"type", "Sequence"}, {"normalizers", norm}}}, {"pre_tokenizer", nullptr}, {"post_processor", nullptr},
                        {"decoder", json::object{{"type", "Sequence"}, {"decoders", dec}}},
                        {"model", json::object{{"type", m.unigram ? "Unigram" : "SentencePieceBPE"}, {"unk_id", m.unk_id >= 0 ? json::value(double(m.unk_id)) : json::value(nullptr)},
                                               {"byte_fallback", m.byte_fallback}, {"vocab", vocab}, {"token_types", types}}}};
}

// protobuf wire format, the little of it sentencepiece_model.proto needs
struct pb_field { std::uint32_t num = 0, wt = 0; std::uint64_t v = 0; std::string_view bytes; };
template <class F> void pb_each(std::string_view s, F&& f) {
    std::size_t p = 0;
    auto varint = [&] {
        std::uint64_t v = 0;
        for (int sh = 0; sh < 64; sh += 7) {
            if (p >= s.size()) raise(errc::format_error, "sentencepiece model: truncated varint");
            auto b = std::uint8_t(s[p++]); v |= std::uint64_t(b & 0x7f) << sh;
            if (!(b & 0x80)) return v;
        }
        raise(errc::format_error, "sentencepiece model: varint too long");
    };
    auto need = [&](std::uint64_t n) { if (n > s.size() - p) raise(errc::format_error, "sentencepiece model: truncated field"); };
    while (p < s.size()) {
        pb_field x; auto key = varint(); x.num = std::uint32_t(key >> 3); x.wt = std::uint32_t(key & 7);
        switch (x.wt) {
        case 0: x.v = varint(); break;
        case 1: need(8); std::memcpy(&x.v, s.data() + p, 8); p += 8; break;
        case 5: { need(4); std::uint32_t u; std::memcpy(&u, s.data() + p, 4); x.v = u; p += 4; break; }
        case 2: { auto n = varint(); need(n); x.bytes = s.substr(p, std::size_t(n)); p += std::size_t(n); break; }
        default: raise(errc::format_error, "sentencepiece model: unsupported wire type {}", x.wt);
        }
        f(x);
    }
}

// llama.cpp's pre-tokenizer names (tokenizer.ggml.pre): the regexes each one splits with before byte-level BPE.
// Returns false for a name this table does not have.
// (Characters outside ASCII are written as regex escapes, \x{..}, so the source stays ASCII for every compiler.)
inline bool gguf_pre_regexes(std::string_view pre, std::vector<std::string>& re, bool& ignore_merges) {
    auto is = [&](std::initializer_list<const char*> names) { for (auto n : names) if (pre == n) return true; return false; };
    static const char* const gpt2 = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S))";
    static const char* const contr = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";
    ignore_merges = false; re.clear();
    if (is({"llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2", "dbrx", "smaug-bpe", "chatglm-bpe", "glm4"})) {
        re = {std::string(contr) + R"(|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"};
        ignore_merges = is({"llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2"});
    } else if (is({"qwen2", "deepseek-r1-qwen", "stablelm2", "hunyuan", "megrez", "kormo"})) {
        re = {std::string(contr) + R"(|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"};
    } else if (is({"default", "gpt-2", "gpt2", "phi-2", "jina-es", "jina-de", "jina-v1-en", "jina-v2-es", "jina-v2-de", "jina-v2-code", "roberta-bpe",
                   "mpt", "olmo", "jais", "trillion", "granite-docling", "gigachat", "bailingmoe", "llada-moe", "minerva-7b", "exaone4"})) {
        re = {};   // the ByteLevel pre-tokenizer's own regex
    } else if (is({"starcoder", "refact", "command-r", "smollm", "codeshell", "exaone", "minerva"})) {
        re = {R"(\p{N})", gpt2};
    } else if (pre == "falcon") {
        re = {R"([\p{P}\$\+<=>\^~\|`]+)", gpt2, "[0-9][0-9][0-9]"};
    } else if (pre == "deepseek-llm") {
        re = {R"([\r\n])",
              "\\s?[A-Za-z\\x{00b5}\\x{00c0}-\\x{00d6}\\x{00d8}-\\x{00f6}\\x{00f8}-\\x{01ba}\\x{01bc}-\\x{01bf}\\x{01c4}-\\x{0293}\\x{0295}-\\x{02af}\\x{0370}-\\x{0373}\\x{0376}\\x{0377}\\x{037b}-\\x{037d}\\x{037f}"
              "\\x{0386}\\x{0388}-\\x{038a}\\x{038c}\\x{038e}-\\x{03a1}\\x{03a3}-\\x{03f5}\\x{03f7}-\\x{0481}\\x{048a}-\\x{052f}\\x{0531}-\\x{0556}\\x{10a0}-\\x{10c5}\\x{13a0}-\\x{13f5}\\x{13f8}-\\x{13fd}"
              "\\x{1c90}-\\x{1cba}\\x{1cbd}-\\x{1cbf}\\x{1d00}-\\x{1d2b}\\x{1d6b}-\\x{1d77}\\x{1d79}-\\x{1d9a}\\x{1e00}-\\x{1f15}\\x{1f18}-\\x{1f1d}\\x{1f20}-\\x{1f45}\\x{1f48}-\\x{1f4d}\\x{1f50}-\\x{1f57}"
              "\\x{1f59}\\x{1f5b}\\x{1f5d}\\x{1f5f}-\\x{1f7d}\\x{1f80}-\\x{1fb4}\\x{1fb6}-\\x{1fbc}\\x{1fbe}\\x{1fc2}-\\x{1fc4}\\x{1fc6}-\\x{1fcc}\\x{1fd0}-\\x{1fd3}\\x{1fd6}-\\x{1fdb}\\x{1fe0}-\\x{1fec}"
              "\\x{1ff2}-\\x{1ff4}\\x{1ff6}-\\x{1ffc}\\x{2102}\\x{2107}\\x{210a}-\\x{2113}\\x{2115}\\x{2119}-\\x{211d}\\x{2124}\\x{2126}\\x{2128}\\x{212a}-\\x{212d}\\x{212f}-\\x{2134}\\x{2139}\\x{213c}-\\x{213f}"
              "\\x{2145}-\\x{2149}\\x{214e}\\x{2183}\\x{2184}\\x{2c00}-\\x{2c7b}\\x{2c7e}-\\x{2ce4}\\x{2ceb}-\\x{2cee}\\x{2cf2}\\x{2cf3}\\x{a640}-\\x{a66d}\\x{a680}-\\x{a69b}\\x{a722}-\\x{a76f}\\x{a771}-\\x{a787}"
              "\\x{a78b}-\\x{a78e}\\x{ab70}-\\x{abbf}\\x{fb00}-\\x{fb06}\\x{fb13}-\\x{fb17}\\x{ff21}-\\x{ff3a}\\x{ff41}-\\x{ff5a}"
              "\\x{10400}-\\x{1044f}\\x{104b0}-\\x{104d3}\\x{104d8}-\\x{104fb}\\x{10c80}-\\x{10cb2}\\x{10cc0}-\\x{10cf2}\\x{118a0}-\\x{118df}\\x{1e900}-\\x{1e943}]+",
              "\\s?[!-/:-~\\x{ff01}-\\x{ff0f}\\x{ff1a}-\\x{ff5e}\\x{2018}-\\x{201f}\\x{3000}-\\x{3002}]+", R"(\s+$)", "[\\x{4e00}-\\x{9fa5}\\x{0800}-\\x{4e00}\\x{ac00}-\\x{d7ff}]+", R"(\p{N}+)"};
    } else if (pre == "deepseek-coder") {
        re = {R"([\r\n])", R"(\s?\p{L}+)", R"(\s?\p{P}+)", "[\\x{4e00}-\\x{9fa5}\\x{0800}-\\x{4e00}\\x{ac00}-\\x{d7ff}]+", R"(\p{N})"};
    } else if (is({"deepseek-v3", "hunyuan-dense"})) {
        re = {R"(\p{N}{1,3})", "[\\x{4e00}-\\x{9fa5}\\x{3040}-\\x{309f}\\x{30a0}-\\x{30ff}]+",
              R"([!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"};
    } else if (pre == "tekken") {
        re = {R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)"};
        ignore_merges = true;
    } else if (is({"gpt-4o", "llama4"})) {
        re = {std::string(R"([^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+)") + contr + R"(?|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*)" + contr
              + R"(?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)"};
        ignore_merges = pre == "llama4";
    } else if (is({"viking"})) {
        re = {" ?[^(\\s|.,!?\\x{2026}\\x{3002}\\x{ff0c}\\x{3001}\\x{0964}\\x{06d4}\\x{060c})]+", R"(\p{N})"};
    } else if (is({"poro-chat", "bloom", "gpt3-finnish"})) {
        re = {" ?[^(\\s|.,!?\\x{2026}\\x{3002}\\x{ff0c}\\x{3001}\\x{0964}\\x{06d4}\\x{060c})]+"};
    } else return false;
    return true;
}
}  // namespace tokimp

// tokenizer.json text for a serialized SentencePiece ModelProto (BPE or Unigram). tokenizer_config: the checkpoint's
// tokenizer_config.json, whose "added_tokens_decoder" lists the tokens transformers adds on top of the model's pieces.
inline std::string sentencepiece_tokenizer_json(std::string_view proto, const json::value* tokenizer_config = nullptr) {
    using namespace tokimp;
    sp_model m; int type = 1; bool suffix = false; m.unk_id = 0; m.remove_extra_ws = true;
    pb_each(proto, [&](const pb_field& f) {
        if (f.num == 1 && f.wt == 2) {
            sp_model::piece p;
            pb_each(f.bytes, [&](const pb_field& g) {
                if (g.num == 1 && g.wt == 2) p.text = std::string(g.bytes);
                else if (g.num == 2 && g.wt == 5) { float x; auto u = std::uint32_t(g.v); std::memcpy(&x, &u, 4); p.score = x; }
                else if (g.num == 3 && g.wt == 0) p.type = int(g.v & 0xFF);
            });
            if (m.pieces.size() >= (8u << 20)) raise(errc::format_error, "sentencepiece model: too many pieces");
            m.pieces.push_back(std::move(p));
        } else if (f.num == 2 && f.wt == 2) {
            pb_each(f.bytes, [&](const pb_field& g) {
                if (g.wt != 0) return;
                if (g.num == 3) type = int(g.v); else if (g.num == 24) suffix = g.v != 0; else if (g.num == 35) m.byte_fallback = g.v != 0;
                else if (g.num == 40) m.unk_id = int(std::int32_t(g.v));
            });
        } else if (f.num == 3 && f.wt == 2) {
            pb_each(f.bytes, [&](const pb_field& g) {
                if (g.num == 2 && g.wt == 2) m.charsmap = std::string(g.bytes);
                else if (g.num == 3 && g.wt == 0) m.dummy_prefix = g.v != 0;
                else if (g.num == 4 && g.wt == 0) m.remove_extra_ws = g.v != 0;
                else if (g.num == 5 && g.wt == 0 && !g.v) raise(errc::unsupported, "sentencepiece model: escape_whitespaces = false is not supported");
            });
        }
    });
    if (m.pieces.empty()) raise(errc::format_error, "sentencepiece model: no pieces");
    if (type != 1 && type != 2) raise(errc::unsupported, "sentencepiece model: only unigram and bpe models are supported (model_type {})", type);
    if (suffix) raise(errc::unsupported, "sentencepiece model: treat_whitespace_as_suffix is not supported");
    m.unigram = type == 1;
    if (m.unk_id < 0 || std::size_t(m.unk_id) >= m.pieces.size()) m.unk_id = -1;
    auto tj = sentencepiece_json(m);
    if (auto* dec = tokenizer_config ? tokenizer_config->find("added_tokens_decoder") : nullptr; dec && dec->is_obj()) {
        json::array added = tj["added_tokens"].arr();
        for (auto& [idstr, t] : dec->obj()) {
            char* end = nullptr; double id = std::strtod(idstr.c_str(), &end);
            if (end == idstr.c_str() || *end || !t.find("content") || !t["content"].is_str()) continue;
            bool have = false; for (auto& a : added) have |= a["id"].num() == id;
            if (have) continue;
            auto fl = [&](const char* k, bool d) { auto* v = t.find(k); return v && v->is_bool() ? v->b() : d; };
            added.push_back(json::object{{"id", id}, {"content", t["content"].str()}, {"special", fl("special", false)}, {"normalized", fl("normalized", false)},
                                         {"single_word", fl("single_word", false)}, {"lstrip", fl("lstrip", false)}, {"rstrip", fl("rstrip", false)}});
        }
        for (auto& [k, v] : tj.obj_mut()) if (k == "added_tokens") v = added;
    }
    return json::dump(tj);
}

// tokenizer.json text synthesised from GGUF metadata; "" when the file has no vocabulary or a tokenizer model this
// build does not know (rwkv, ...), which `warning` then says. G needs get(key) -> const json::value* (import::gguf_file).
//   tokenizer.ggml.model  gpt2            byte-level BPE; tokenizer.ggml.pre picks the split regex
//                         llama           SentencePiece BPE over tokenizer.ggml.scores
//                         t5 / unigram    Unigram
//                         bert            WordPiece
//   tokenizer.ggml.token_type: 3 control and 2 unknown become special added tokens, 4 user-defined plain added tokens
//   tokenizer.huggingface.json, when present, is used as it is.
template <class G> std::string gguf_tokenizer_json(const G& g, std::string* warning = nullptr) {
    using namespace tokimp;
    using T = modelfile::tokenizer;
    if (auto* hf = g.get("tokenizer.huggingface.json"); hf && hf->is_str() && !hf->str().empty()) return hf->str();
    auto* tk = g.get("tokenizer.ggml.tokens");
    if (!tk || !tk->is_arr()) return "";
    auto& toks = tk->arr();
    auto arr = [&](const char* k) -> const json::array* { auto* v = g.get(k); return v && v->is_arr() && v->arr().size() == toks.size() ? &v->arr() : nullptr; };
    auto flag = [&](const char* k, bool d) { auto* v = g.get(k); return v && v->is_bool() ? v->b() : d; };
    auto num = [&](const char* k, double d) { auto* v = g.get(k); return v && v->is_num() ? v->num() : d; };
    auto* types = arr("tokenizer.ggml.token_type"); auto* scores = arr("tokenizer.ggml.scores");
    auto type_of = [&](std::size_t i) { return types && (*types)[i].is_num() ? int((*types)[i].num()) : int(T::type_normal); };
    auto model = g.get("tokenizer.ggml.model") && g.get("tokenizer.ggml.model")->is_str() ? g.get("tokenizer.ggml.model")->str() : std::string("gpt2");
    auto added_tokens = [&] {
        json::array added;
        for (std::size_t i = 0; i < toks.size(); ++i)
            if (int t = type_of(i); t == T::type_control || t == T::type_unknown || t == T::type_user_defined)
                added.push_back(json::object{{"id", double(i)}, {"content", toks[i].str()}, {"special", t != T::type_user_defined}, {"normalized", false},
                                             {"single_word", false}, {"lstrip", false}, {"rstrip", false}});
        return added;
    };
    if (model == "llama" || model == "t5" || model == "unigram") {
        sp_model m; m.unigram = model != "llama";
        for (std::size_t i = 0; i < toks.size(); ++i) {
            m.pieces.push_back({toks[i].str(), scores && (*scores)[i].is_num() ? (*scores)[i].num() : 0.0, type_of(i)});
            m.byte_fallback |= m.pieces.back().type == T::type_byte;
        }
        m.unk_id = int(num("tokenizer.ggml.unknown_token_id", model == "llama" ? 0 : -1));
        if (m.unk_id >= int(toks.size())) m.unk_id = -1;
        m.dummy_prefix = flag("tokenizer.ggml.add_space_prefix", true);
        m.remove_extra_ws = flag("tokenizer.ggml.remove_extra_whitespaces", false);
        if (auto* cm = g.get("tokenizer.ggml.precompiled_charsmap")) {
            if (cm->is_str()) m.charsmap = cm->str();
            else if (cm->is_arr()) for (auto& b : cm->arr()) m.charsmap += char(b.i() & 0xFF);
        }
        return json::dump(sentencepiece_json(m));
    }
    if (model == "gpt2") {
        json::object vocab; vocab.reserve(toks.size());
        for (std::size_t i = 0; i < toks.size(); ++i) vocab.emplace_back(toks[i].str(), double(i));
        json::array merges; if (auto* m = g.get("tokenizer.ggml.merges"); m && m->is_arr()) merges = m->arr();
        std::string pre = g.get("tokenizer.ggml.pre") && g.get("tokenizer.ggml.pre")->is_str() ? g.get("tokenizer.ggml.pre")->str() : "default";
        std::vector<std::string> re; bool ignore_merges = false;
        if (!gguf_pre_regexes(pre, re, ignore_merges)) {   // llama.cpp's old behaviour: carry on with the default
            if (warning) *warning = "unknown pre-tokenizer '" + pre + "' (tokenizer.ggml.pre); using the GPT-2 one, token ids may differ from the model's";
            gguf_pre_regexes("default", re, ignore_merges);
        }
        json::value pt = json::object{{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", re.empty()}};
        if (!re.empty()) {
            json::array seq;
            for (auto& r : re) seq.push_back(json::object{{"type", "Split"}, {"pattern", json::object{{"Regex", r}}}, {"behavior", "Isolated"}, {"invert", false}});
            seq.push_back(pt);
            pt = json::object{{"type", "Sequence"}, {"pretokenizers", seq}};
        }
        return json::dump(json::object{{"forge_tokenizer_version", double(modelfile::tokenizer_format_version)}, {"added_tokens", added_tokens()}, {"normalizer", nullptr},
                                       {"pre_tokenizer", pt}, {"post_processor", nullptr}, {"decoder", json::object{{"type", "ByteLevel"}}},
                                       {"model", json::object{{"type", "BPE"}, {"ignore_merges", ignore_merges}, {"vocab", vocab}, {"merges", merges}}}});
    }
    if (model == "bert") {
        // llama.cpp's converter writes word-initial pieces as "▁word" and continuation pieces bare; back to WordPiece.
        json::object vocab; vocab.reserve(toks.size());
        static const std::string meta = "\xE2\x96\x81";
        for (std::size_t i = 0; i < toks.size(); ++i) {
            const auto& t = toks[i].str();
            bool bracket = t.size() > 1 && t.front() == '[' && t.back() == ']';
            vocab.emplace_back(bracket || type_of(i) == T::type_control ? t : t.starts_with(meta) ? t.substr(meta.size()) : "##" + t, double(i));
        }
        // [CLS] / [SEP]: llama.cpp's converter writes them as BOS and the separator; older files have cls_token_id and
        // only EOS, and a file with neither still has the pieces by name
        auto first_id = [&](std::initializer_list<const char*> keys, std::initializer_list<const char*> names) {
            for (auto* k : keys) if (int v = int(num(k, -1)); v >= 0 && std::size_t(v) < toks.size()) return v;
            for (auto* n : names) for (std::size_t i = 0; i < toks.size(); ++i) if (toks[i].is_str() && toks[i].str() == n) return int(i);
            return -1;
        };
        int unk = int(num("tokenizer.ggml.unknown_token_id", -1));
        int cls = first_id({"tokenizer.ggml.bos_token_id", "tokenizer.ggml.cls_token_id"}, {"[CLS]", "<s>"});
        int sep = first_id({"tokenizer.ggml.seperator_token_id", "tokenizer.ggml.separator_token_id", "tokenizer.ggml.eos_token_id"}, {"[SEP]", "</s>"});
        auto name = [&](int id) { return id >= 0 && std::size_t(id) < toks.size() ? toks[std::size_t(id)].str() : std::string(); };
        json::value post = nullptr;
        if (cls >= 0 && sep >= 0 && std::size_t(cls) < toks.size() && std::size_t(sep) < toks.size())
            post = json::object{{"type", "BertProcessing"}, {"sep", json::array{name(sep), double(sep)}}, {"cls", json::array{name(cls), double(cls)}}};
        // llama.cpp's WPM tokenizer: NFD, lower-case, control characters dropped, CJK characters and punctuation on their own
        json::array norm{json::object{{"type", "NFD"}},
                         json::object{{"type", "BertNormalizer"}, {"clean_text", true}, {"handle_chinese_chars", true}, {"strip_accents", false}, {"lowercase", true}}};
        return json::dump(json::object{{"forge_tokenizer_version", double(modelfile::tokenizer_format_version)}, {"added_tokens", added_tokens()},
                                       {"normalizer", json::object{{"type", "Sequence"}, {"normalizers", norm}}}, {"pre_tokenizer", json::object{{"type", "BertPreTokenizer"}}},
                                       {"post_processor", post}, {"decoder", json::object{{"type", "WordPiece"}, {"prefix", "##"}, {"cleanup", true}}},
                                       {"model", json::object{{"type", "WordPiece"}, {"unk_token", unk >= 0 ? name(unk) : std::string("[UNK]")}, {"continuing_subword_prefix", "##"},
                                                              {"max_input_chars_per_word", 100.0}, {"vocab", vocab}}}});
    }
    if (warning) *warning = "tokenizer model '" + model + "' (tokenizer.ggml.model) is not supported; the file gets no tokenizer";
    return "";
}

// tokenizer.json text for a WordPiece vocab.txt (one piece per line, the line number is the id), which is all an older
// BERT checkpoint has. tokenizer_config: its tokenizer_config.json (do_lower_case, strip_accents,
// tokenize_chinese_chars, unk_token / cls_token / sep_token); BertTokenizer's defaults where it says nothing.
inline std::string wordpiece_vocab_tokenizer_json(std::string_view vocab_txt, const json::value* tokenizer_config = nullptr) {
    json::object vocab; std::size_t n = 0;
    for (std::size_t p = 0; p < vocab_txt.size();) {
        auto e = vocab_txt.find('\n', p); if (e == std::string_view::npos) e = vocab_txt.size();
        auto line = vocab_txt.substr(p, e - p); if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        p = e + 1;
        if (n >= (8u << 20)) raise(errc::format_error, "vocab.txt: too many pieces");
        vocab.emplace_back(std::string(line), double(n++));
    }
    if (!n) raise(errc::format_error, "vocab.txt: no pieces");
    auto cfg_flag = [&](const char* k, bool d) { auto* v = tokenizer_config ? tokenizer_config->find(k) : nullptr; return v && v->is_bool() ? v->b() : d; };
    auto cfg_tok = [&](const char* k, const char* d) -> std::string {   // "[CLS]" or {"content": "[CLS]", ...}
        auto* v = tokenizer_config ? tokenizer_config->find(k) : nullptr;
        if (v && v->is_str()) return v->str();
        if (v && v->is_obj()) if (auto* c = v->find("content"); c && c->is_str()) return c->str();
        return d;
    };
    auto id_of = [&](const std::string& t) { for (auto& [k, v] : vocab) if (k == t) return v.i32(); return -1; };
    bool lower = cfg_flag("do_lower_case", true);
    json::value strip = nullptr;   // null: accents go when lower-casing, as BertNormalizer does
    if (auto* v = tokenizer_config ? tokenizer_config->find("strip_accents") : nullptr; v && v->is_bool()) strip = v->b();
    auto unk = cfg_tok("unk_token", "[UNK]"), cls = cfg_tok("cls_token", "[CLS]"), sep = cfg_tok("sep_token", "[SEP]");
    json::array added;
    for (auto& t : {cfg_tok("pad_token", "[PAD]"), unk, cls, sep, cfg_tok("mask_token", "[MASK]")})
        if (int id = id_of(t); id >= 0)
            added.push_back(json::object{{"id", double(id)}, {"content", t}, {"special", true}, {"normalized", false}, {"single_word", false}, {"lstrip", false}, {"rstrip", false}});
    json::value post = nullptr;
    if (int c = id_of(cls), q = id_of(sep); c >= 0 && q >= 0)
        post = json::object{{"type", "BertProcessing"}, {"sep", json::array{sep, double(q)}}, {"cls", json::array{cls, double(c)}}};
    return json::dump(json::object{{"forge_tokenizer_version", double(modelfile::tokenizer_format_version)}, {"added_tokens", added},
                                   {"normalizer", json::object{{"type", "BertNormalizer"}, {"clean_text", true}, {"handle_chinese_chars", cfg_flag("tokenize_chinese_chars", true)},
                                                               {"strip_accents", strip}, {"lowercase", lower}}},
                                   {"pre_tokenizer", json::object{{"type", "BertPreTokenizer"}}}, {"post_processor", post},
                                   {"decoder", json::object{{"type", "WordPiece"}, {"prefix", "##"}, {"cleanup", true}}},
                                   {"model", json::object{{"type", "WordPiece"}, {"unk_token", unk}, {"continuing_subword_prefix", "##"}, {"max_input_chars_per_word", 100.0},
                                                          {"vocab", vocab}}}});
}

// "" when the runtime tokenizer (modelfile/tokenizer.hpp) can load this tokenizer.json text, else why not. Importers
// ask before they store a "tokenizer" section: a file whose tokenizer section does not load cannot be opened at all.
inline std::string tokenizer_json_problem(std::string_view text) {
    try { modelfile::tokenizer t(text); return ""; }
    catch (const std::exception& e) { return e.what(); }
}

// What every GGUF importer (gguf.hpp, gguf_arch.hpp) does with the file's tokenizer: the "tokenizer" section from
// gguf_tokenizer_json, the "chat_template" section from tokenizer.chat_template. W: modelfile::stream_writer.
// `warn(text)` hears about a vocabulary that was left out.
template <class G, class W, class Warn> void store_gguf_tokenizer(const G& g, W& w, Warn&& warn) {
    std::string note;
    if (auto tj = gguf_tokenizer_json(g, &note); !tj.empty()) {
        if (auto why = tokenizer_json_problem(tj); why.empty()) w.set_section("tokenizer", tj);
        else note = "the tokenizer could not be converted (" + why + "); the file gets no tokenizer, pass token ids";
    }
    if (!note.empty()) warn(note);
    if (auto* ct = g.get("tokenizer.chat_template"); ct && ct->is_str() && !ct->str().empty()) w.set_section("chat_template", ct->str());
}

// The chat template a tokenizer_config.json carries: a string, or a list of {name, template} of which "default" (else
// the first) is taken. "" when there is none.
inline std::string chat_template_of(const json::value& tokenizer_config) {
    auto* ct = tokenizer_config.find("chat_template");
    if (!ct) return "";
    if (ct->is_str()) return ct->str();
    if (!ct->is_arr()) return "";
    std::string first;
    for (auto& e : ct->arr()) {
        auto* t = e.find("template");
        if (!t || !t->is_str()) continue;
        if (e.str_or("name", "") == "default") return t->str();
        if (first.empty()) first = t->str();
    }
    return first;
}

} // namespace forge::import
