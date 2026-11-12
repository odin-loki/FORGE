#pragma once
// Tokenizers loaded from a Hugging Face tokenizer.json (the .fpk "tokenizer" section).
//
// The Hugging Face `tokenizers` pipeline is implemented stage by stage:
//   added tokens -> normalizer -> pre-tokenizer -> model -> post-processor, and decoder on the way back.
// Models: BPE (byte-level GPT-2 / Llama-3 / Qwen, or character-level with byte_fallback), WordPiece (BERT), Unigram
// (T5 / Gemma / XLM-R) and, as a FORGE extension, "SentencePieceBPE": SentencePiece's score-driven BPE, which is what a
// Llama-2 / Mistral GGUF vocabulary (tokenizer.ggml.model = "llama") describes.
//
// Section versions (docs/format.md):
//   1  written by FORGE before the pipeline existed: a BPE model with no "normalizer" / "pre_tokenizer" / "decoder"
//      keys. Still read the old way (byte-level BPE behind an ASCII-class approximation of the GPT-2 pre-tokenizer).
//   2  a full tokenizer.json. "forge_tokenizer_version": 2 marks one that FORGE synthesised and that may use the
//      "SentencePieceBPE" model type and "token_types"; a verbatim Hugging Face file has no such key.
//
// Patterns are compiled by forge::os::text::regex (the forge_text library, which forge_modelfile links) in its
// Oniguruma dialect; Unicode classes, normalization forms and grapheme clusters come from the same library.
#include <forge/json/json.hpp>
#include <forge/os/text.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cstring>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace forge::modelfile {

inline constexpr int tokenizer_format_version = 2;

namespace tok {
namespace tx = forge::os::text;

struct sv_hash {
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
};
using vocab_map = std::unordered_map<std::string, int, sv_hash, std::equal_to<>>;

inline bool is_ws(char32_t c) { return tx::is_space(c); }                       // Rust char::is_whitespace
// element i of a JSON array (json::value's own operator[] lets std::out_of_range through)
inline const json::value& at(const json::value& a, std::size_t i) {
    if (!a.is_arr() || i >= a.arr().size()) raise(errc::format_error, "tokenizer: expected an array with at least {} elements", i + 1);
    return a.arr()[i];
}
inline bool is_punc(char32_t c) { return (c < 0x80 && std::ispunct(int(c))) || tx::is_punct(c); }
inline bool starts_with(std::string_view s, std::string_view p) { return !p.empty() && s.substr(0, p.size()) == p; }
inline void replace_all(std::string& s, std::string_view a, std::string_view b) {
    if (a.empty()) return;
    for (std::size_t p = 0; (p = s.find(a, p)) != std::string::npos; p += b.size()) s.replace(p, a.size(), b);
}
template <class F> void each_cp(std::string_view s, F&& f) {   // f(code point, byte begin, byte end)
    for (std::size_t i = 0; i < s.size();) { std::size_t b = i; char32_t c = tx::decode_utf8(s, i); f(c, b, i); }
}
inline std::string base64_decode(std::string_view s) {
    std::string o; unsigned acc = 0; int bits = 0;
    for (unsigned char c : s) {
        int v = c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26 : c >= '0' && c <= '9' ? c - '0' + 52
              : c == '+' || c == '-' ? 62 : c == '/' || c == '_' ? 63 : -1;
        if (v < 0) continue;   // padding, line breaks
        acc = (acc << 6) | unsigned(v); bits += 6;
        if (bits >= 8) { bits -= 8; o += char((acc >> bits) & 0xFF); }
    }
    return o;
}

// GPT-2 bytes_to_unicode: every byte gets a printable code point.
struct byte_map {
    std::array<std::string, 256> enc; std::array<short, 324> dec;
    byte_map() {
        dec.fill(-1); int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            unsigned cp = keep ? unsigned(b) : unsigned(256 + n++);
            tx::append_utf8(enc[std::size_t(b)], char32_t(cp)); dec[cp] = short(b);
        }
    }
    static const byte_map& get() { static const byte_map m; return m; }
    // token text -> bytes; a token holding a character outside the map is kept as it is (Hugging Face ByteLevel decoder)
    void decode(std::string_view t, std::string& o) const {
        std::string b; bool ok = true;
        each_cp(t, [&](char32_t c, std::size_t, std::size_t) { if (c < dec.size() && dec[c] >= 0) b += char(dec[c]); else ok = false; });
        o += ok ? std::string_view(b) : t;
    }
};

// A "String" or "Regex" pattern of tokenizer.json.
struct pattern {
    bool is_regex = false; std::string lit; tx::regex re;
    pattern() = default;
    explicit pattern(const json::value& j) {
        if (j.is_str()) lit = j.str();
        else if (auto* s = j.find("String")) lit = s->str();
        else if (auto* r = j.find("Regex")) compile(r->str());
        else raise(errc::format_error, "tokenizer: a pattern needs \"String\" or \"Regex\"");
    }
    void compile(const std::string& p) {
        auto r = tx::regex::compile(p, "R");
        if (!r) raise(errc::unsupported, "tokenizer: pattern {} is not supported ({})", p, r.err.what);
        re = std::move(r.value); re.set_step_limit(2'000'000); is_regex = true;
    }
    static pattern regex(const std::string& p) { pattern x; x.compile(p); return x; }
    std::vector<std::pair<std::size_t, std::size_t>> spans(std::string_view s) const {
        if (is_regex) {
            auto sp = re.find_spans(s);
            if (re.step_limit_hit()) raise(errc::invalid_argument, "tokenizer: a tokenizer.json pattern backtracks too much on this text");
            return sp;
        }
        std::vector<std::pair<std::size_t, std::size_t>> o;
        if (!lit.empty()) for (std::size_t p = 0; (p = s.find(lit, p)) != std::string_view::npos; p += lit.size()) o.push_back({p, p + lit.size()});
        return o;
    }
    void replace(std::string& s, std::string_view with) const {
        if (!is_regex) { replace_all(s, lit, with); return; }
        auto sp = spans(s);
        if (sp.empty()) return;
        std::string o; std::size_t last = 0;
        for (auto [b, e] : sp) { o.append(s, last, b - last); o += with; last = e; }
        o.append(s, last); s = std::move(o);
    }
};

// One piece of text on its way through the pre-tokenizers. `first`: it starts at offset 0 of the input (Metaspace
// prepend_scheme "first" looks at that).
struct piece { std::string s; bool first = false; };

enum class behavior { removed, isolated, merged_prev, merged_next, contiguous };
inline behavior parse_behavior(const std::string& b) {
    if (b == "Removed") return behavior::removed;
    if (b == "Isolated") return behavior::isolated;
    if (b == "MergedWithPrevious") return behavior::merged_prev;
    if (b == "MergedWithNext") return behavior::merged_next;
    if (b == "Contiguous") return behavior::contiguous;
    raise(errc::unsupported, "tokenizer: split behavior {}", b);
}
// NormalizedString::split: `m` are the match spans (ascending, disjoint); pieces are appended to `out`, empty ones dropped.
inline void split_by(const piece& p, const std::vector<std::pair<std::size_t, std::size_t>>& m, behavior beh, bool invert, std::vector<piece>& out) {
    struct span { std::size_t b, e; bool match; };
    std::vector<span> cover; std::size_t prev = 0; const std::string& s = p.s;
    for (auto [b, e] : m) {
        if (b < prev || e > s.size() || e < b) continue;
        if (prev != b) cover.push_back({prev, b, invert});
        cover.push_back({b, e, !invert}); prev = e;
    }
    if (prev != s.size()) cover.push_back({prev, s.size(), invert});
    std::vector<span> sp;   // match = "remove" from here on
    bool pm = false;
    switch (beh) {
    case behavior::removed: sp = cover; break;
    case behavior::isolated: for (auto c : cover) sp.push_back({c.b, c.e, false}); break;
    case behavior::contiguous:
        for (auto c : cover) { if (c.match == pm && !sp.empty()) sp.back().e = c.e; else sp.push_back({c.b, c.e, false}); pm = c.match; }
        break;
    case behavior::merged_prev:
        for (auto c : cover) { if (c.match && !pm && !sp.empty()) sp.back().e = c.e; else sp.push_back({c.b, c.e, false}); pm = c.match; }
        break;
    case behavior::merged_next:
        for (auto it = cover.rbegin(); it != cover.rend(); ++it) { if (it->match && !pm && !sp.empty()) sp.back().b = it->b; else sp.push_back({it->b, it->e, false}); pm = it->match; }
        std::reverse(sp.begin(), sp.end());
        break;
    }
    for (auto& c : sp) if (!c.match && c.e > c.b) out.push_back({s.substr(c.b, c.e - c.b), p.first && c.b == 0});
}
template <class Pred> std::vector<std::pair<std::size_t, std::size_t>> char_spans(std::string_view s, Pred&& pred) {
    std::vector<std::pair<std::size_t, std::size_t>> m;
    each_cp(s, [&](char32_t c, std::size_t b, std::size_t e) { if (pred(c)) m.push_back({b, e}); });
    return m;
}

// ------------------------------------------------------------------------------------------------ normalizers
struct normalizer {
    enum kind { none, nfc, nfd, nfkc, nfkd, lowercase, strip, strip_accents, replace, prepend, sequence, bert, precompiled, nmt } k = none;
    bool left = true, right = true;                               // Strip
    bool clean = true, chinese = true, lower = true, accents = true;   // BertNormalizer
    pattern pat; std::string content;                             // Replace / Prepend
    std::vector<normalizer> seq;
    std::vector<std::uint32_t> trie; std::string normalized;      // Precompiled (SentencePiece charsmap)
    bool longest = false;   // "forge_match": "longest" -- SentencePiece's own lookup, not the `tokenizers` crate's

    normalizer() = default;
    explicit normalizer(const json::value& j, int depth = 0) {
        if (j.is_null()) return;
        if (depth > 16) raise(errc::format_error, "tokenizer: normalizers nested too deep");
        auto t = j.str_or("type", "");
        auto flag = [&](const char* key, bool d) { auto* v = j.find(key); return v && v->is_bool() ? v->b() : d; };
        if (t == "NFC") k = nfc; else if (t == "NFD") k = nfd; else if (t == "NFKC") k = nfkc; else if (t == "NFKD") k = nfkd;
        else if (t == "Lowercase") k = lowercase;
        else if (t == "StripAccents") k = strip_accents;
        else if (t == "Nmt") k = nmt;
        else if (t == "Strip") { k = strip; left = flag("strip_left", true); right = flag("strip_right", true); }
        else if (t == "Replace") { k = replace; pat = pattern(j["pattern"]); content = j.str_or("content", ""); }
        else if (t == "Prepend") { k = prepend; content = j.str_or("prepend", ""); }
        else if (t == "Sequence") { k = sequence; for (auto& n : j["normalizers"].arr()) seq.emplace_back(n, depth + 1); }
        else if (t == "BertNormalizer") {
            k = bert; clean = flag("clean_text", true); chinese = flag("handle_chinese_chars", true); lower = flag("lowercase", true);
            accents = flag("strip_accents", lower);   // null: follows lowercase
        } else if (t == "Precompiled") {
            k = precompiled; longest = j.str_or("forge_match", "") == "longest";
            auto blob = base64_decode(j.str_or("precompiled_charsmap", ""));
            if (blob.empty()) { k = none; return; }   // an empty charsmap is the identity
            std::uint32_t n = 0;
            if (blob.size() < 4) raise(errc::format_error, "tokenizer: truncated precompiled_charsmap");
            std::memcpy(&n, blob.data(), 4);
            if (n % 4 || n > blob.size() - 4) raise(errc::format_error, "tokenizer: bad precompiled_charsmap trie size");
            trie.resize(n / 4);
            for (std::size_t i = 0; i < trie.size(); ++i) {   // little-endian units
                auto* b = reinterpret_cast<const unsigned char*>(blob.data()) + 4 + 4 * i;
                trie[i] = std::uint32_t(b[0]) | std::uint32_t(b[1]) << 8 | std::uint32_t(b[2]) << 16 | std::uint32_t(b[3]) << 24;
            }
            normalized = blob.substr(4 + n);
        } else raise(errc::unsupported, "tokenizer: normalizer type '{}' is not supported", t);
    }
    bool identity() const { return k == none || (k == sequence && std::all_of(seq.begin(), seq.end(), [](auto& n) { return n.identity(); })); }

    static void lower32(std::u32string& u) {
        std::u32string o; o.reserve(u.size());
        for (auto c : u) { if (c == 0x130) { o += U'i'; o += char32_t(0x307); } else o += c < 0x80 ? char32_t(c >= 'A' && c <= 'Z' ? c + 32 : c) : tx::simple_lower(c); }
        u = std::move(o);
    }
    static bool cjk(char32_t c) {
        return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F)
            || (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B920 && c <= 0x2CEAF) || (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);
    }
    // darts-clone double array: the value of the shortest key that is a prefix of `key` (SentencePiece's charsmap lookup
    // as the `tokenizers` crate does it), or npos.
    std::size_t lookup(std::string_view key, std::size_t* len = nullptr) const {
        std::size_t found = std::string::npos, n = 0;
        if (trie.empty()) return std::string::npos;
        auto offset = [](std::uint32_t u) { return std::size_t(u >> 10) << ((u & (1u << 9)) >> 6); };
        std::size_t pos = 0; std::uint32_t unit = trie[0];
        pos ^= offset(unit);
        for (unsigned char c : key) {
            if (!c) break;
            pos ^= c; ++n;
            if (pos >= trie.size()) break;
            unit = trie[pos];
            if ((unit & ((1u << 31) | 0xFF)) != c) break;
            pos ^= offset(unit);
            if (pos >= trie.size()) break;
            if ((unit >> 8) & 1) {
                found = std::size_t(trie[pos] & ((1u << 31) - 1));
                if (!len) return found;   // the shortest key
                *len = n;                 // keep going: the longest key
            }
        }
        return found;
    }
    bool transform(std::string_view chunk, std::string& o, std::size_t* len = nullptr) const {
        auto at = lookup(chunk, len);
        if (at == std::string::npos || at > normalized.size()) return false;
        auto e = normalized.find('\0', at);
        o.append(normalized, at, (e == std::string::npos ? normalized.size() : e) - at);
        return true;
    }

    void apply(std::string& s) const {
        switch (k) {
        case none: break;
        case nfc: s = tx::normalize(s, tx::norm_form::nfc); break;
        case nfd: s = tx::normalize(s, tx::norm_form::nfd); break;
        case nfkc: s = tx::normalize(s, tx::norm_form::nfkc); break;
        case nfkd: s = tx::normalize(s, tx::norm_form::nfkd); break;
        case lowercase: { auto u = tx::utf8_to_utf32(s); lower32(u); s = tx::encode_utf8(u); break; }
        case strip_accents: { std::u32string o; for (auto c : tx::utf8_to_utf32(s)) if (!tx::is_mark(c)) o += c; s = tx::encode_utf8(o); break; }
        case strip: {
            auto u = tx::utf8_to_utf32(s); std::size_t b = 0, e = u.size();
            if (left) while (b < e && is_ws(u[b])) ++b;
            if (right) while (e > b && is_ws(u[e - 1])) --e;
            s = tx::encode_utf8(std::u32string_view(u).substr(b, e - b)); break; }
        case replace: pat.replace(s, content); break;
        case prepend: if (!s.empty()) s.insert(0, content); break;
        case sequence: for (auto& n : seq) n.apply(s); break;
        case nmt: {
            std::u32string o;
            for (auto c : tx::utf8_to_utf32(s)) {
                if ((c >= 1 && c <= 8) || c == 0xB || (c >= 0xE && c <= 0x1F) || c == 0x7F || c == 0x8F || c == 0x9F) continue;
                bool sp = c == 9 || c == 0xA || c == 0xC || c == 0xD || c == 0x1680 || (c >= 0x200B && c <= 0x200F) || c == 0x2028 || c == 0x2029
                       || c == 0x2581 || c == 0xFEFF || c == 0xFFFD;
                o += sp ? U' ' : c;
            }
            s = tx::encode_utf8(o); break; }
        case bert: {
            auto u = tx::utf8_to_utf32(s);
            if (clean) {
                std::u32string o;
                for (auto c : u) {
                    bool wsp = c == '\t' || c == '\n' || c == '\r';
                    auto g = tx::general_category(c);
                    bool other = g == tx::category::Cc || g == tx::category::Cf || g == tx::category::Cn || g == tx::category::Co;
                    if (c == 0 || c == 0xFFFD || (!wsp && other)) continue;
                    o += wsp || is_ws(c) ? U' ' : c;
                }
                u = std::move(o);
            }
            if (chinese) { std::u32string o; for (auto c : u) { if (cjk(c)) { o += U' '; o += c; o += U' '; } else o += c; } u = std::move(o); }
            if (accents) { std::u32string o; for (auto c : tx::normalize32(u, tx::norm_form::nfd)) if (tx::general_category(c) != tx::category::Mn) o += c; u = std::move(o); }
            if (lower) lower32(u);
            s = tx::encode_utf8(u); break; }
        case precompiled: {
            std::string o;
            if (longest) {   // SentencePiece: at each position the longest rule, else the character as it is
                for (std::size_t p = 0; p < s.size();) {
                    std::size_t len = 0;
                    if (transform(std::string_view(s).substr(p), o, &len) && len) { p += len; continue; }
                    std::size_t b = p; tx::decode_utf8(s, p); o.append(s, b, p - b);
                }
                s = std::move(o); break;
            }
            // `tokenizers`: grapheme by grapheme; a short grapheme is looked up whole, else code point by code point.
            auto br = tx::grapheme_breaks(s);
            for (std::size_t g = 0; g + 1 < br.size(); ++g) {
                std::string_view gr = std::string_view(s).substr(br[g], br[g + 1] - br[g]);
                if (gr.size() < 6 && transform(gr, o)) continue;
                each_cp(gr, [&](char32_t, std::size_t b, std::size_t e) { auto part = gr.substr(b, e - b); if (!transform(part, o)) o += part; });
            }
            s = std::move(o); break; }
        }
    }
};

// ------------------------------------------------------------------------------------------------ pre-tokenizers
struct pre_tokenizer {
    enum kind { none, byte_level, split, whitespace, whitespace_split, punctuation, digits, metaspace, bert, sequence, char_delim, legacy } k = none;
    bool add_prefix_space = true, use_regex = true;        // ByteLevel
    pattern pat; behavior beh = behavior::removed; bool invert = false;
    bool individual = false;                               // Digits
    std::string repl = "\xE2\x96\x81"; int prepend = 0; bool do_split = true;   // Metaspace; prepend: 0 always, 1 first, 2 never
    char32_t delim = ' ';
    std::vector<pre_tokenizer> seq;

    static int prepend_scheme(const json::value& j) {
        if (auto* p = j.find("prepend_scheme"); p && p->is_str()) {
            if (p->str() == "always") return 0;
            if (p->str() == "first") return 1;
            if (p->str() == "never") return 2;
            raise(errc::unsupported, "tokenizer: Metaspace prepend_scheme {}", p->str());
        }
        auto* a = j.find("add_prefix_space");
        return a && a->is_bool() && !a->b() ? 2 : 0;
    }
    static const pattern& gpt2() {
        static const pattern p = pattern::regex(R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)");
        return p;
    }
    pre_tokenizer() = default;
    explicit pre_tokenizer(const json::value& j, int depth = 0) {
        if (j.is_null()) return;
        if (depth > 16) raise(errc::format_error, "tokenizer: pre-tokenizers nested too deep");
        auto t = j.str_or("type", "");
        auto flag = [&](const char* key, bool d) { auto* v = j.find(key); return v && v->is_bool() ? v->b() : d; };
        if (t == "ByteLevel") { k = byte_level; add_prefix_space = flag("add_prefix_space", true); use_regex = flag("use_regex", true); }
        else if (t == "Split") { k = split; pat = pattern(j["pattern"]); beh = parse_behavior(j.str_or("behavior", "Removed")); invert = flag("invert", false); }
        else if (t == "Whitespace") k = whitespace;
        else if (t == "WhitespaceSplit") k = whitespace_split;
        else if (t == "Punctuation") { k = punctuation; beh = parse_behavior(j.str_or("behavior", "Isolated")); }
        else if (t == "Digits") { k = digits; individual = flag("individual_digits", false); }
        else if (t == "Metaspace") {
            k = metaspace; repl = j.str_or("replacement", repl); prepend = prepend_scheme(j); do_split = flag("split", true);
            if (repl.empty()) raise(errc::format_error, "tokenizer: Metaspace needs a replacement character");
        }
        else if (t == "BertPreTokenizer") k = bert;
        else if (t == "CharDelimiterSplit") { k = char_delim; auto d = tx::utf8_to_utf32(j.str_or("delimiter", " ")); delim = d.empty() ? U' ' : d[0]; }
        else if (t == "Sequence") { k = sequence; for (auto& n : j["pretokenizers"].arr()) seq.emplace_back(n, depth + 1); }
        else raise(errc::unsupported, "tokenizer: pre-tokenizer type '{}' is not supported", t);
    }

    // The pre-.fpk-pipeline splitter, kept for version 1 sections: ASCII classes, every byte >= 0x80 a letter.
    static int legacy_cls(unsigned char c) {  // 0 space, 1 letter, 2 digit, 3 other
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return 0;
        if (std::isalpha(c) || c >= 0x80) return 1;
        if (std::isdigit(c)) return 2;
        return 3;
    }
    static std::size_t legacy_next(std::string_view t, std::size_t p) {
        static const char* contr[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
        for (auto c : contr) { std::string_view cv(c); if (t.substr(p, cv.size()) == cv) return p + cv.size(); }
        std::size_t q = p;
        if (t[q] == ' ' && q + 1 < t.size() && legacy_cls((unsigned char)t[q + 1]) != 0) ++q;  // " word"
        else if (legacy_cls((unsigned char)t[q]) == 0) {
            while (q < t.size() && legacy_cls((unsigned char)t[q]) == 0) ++q;
            if (q < t.size() && q - p > 1 && t[q - 1] == ' ') --q;  // \s+(?!\S)
            return q;
        }
        int c = legacy_cls((unsigned char)t[q]);
        while (q < t.size() && legacy_cls((unsigned char)t[q]) == c) ++q;
        return q;
    }
    static void to_byte_chars(std::string& s) {
        auto& bm = byte_map::get(); std::string o; o.reserve(s.size() + s.size() / 4);
        for (unsigned char c : s) o += bm.enc[c];
        s = std::move(o);
    }

    void apply(std::vector<piece>& ps) const {
        if (k == none) return;
        if (k == sequence) { for (auto& p : seq) p.apply(ps); return; }
        if (k == bert) {
            std::vector<piece> a, b;
            for (auto& p : ps) split_by(p, char_spans(p.s, is_ws), behavior::removed, false, a);
            for (auto& p : a) split_by(p, char_spans(p.s, is_punc), behavior::isolated, false, b);
            ps = std::move(b); return;
        }
        std::vector<piece> out;
        for (auto& p : ps) {
            switch (k) {
            case byte_level: {
                if (add_prefix_space && p.s[0] != ' ') p.s.insert(0, " ");
                std::size_t at = out.size();
                if (use_regex) split_by(p, gpt2().spans(p.s), behavior::isolated, false, out); else out.push_back(p);
                for (; at < out.size(); ++at) to_byte_chars(out[at].s);
                break; }
            case legacy:
                for (std::size_t q = 0; q < p.s.size();) { std::size_t e = legacy_next(p.s, q); out.push_back({p.s.substr(q, e - q), false}); to_byte_chars(out.back().s); q = e; }
                break;
            case split: split_by(p, pat.spans(p.s), beh, invert, out); break;
            case whitespace: {   // \w+|[^\w\s]+, everything else dropped
                std::vector<std::pair<std::size_t, std::size_t>> m; int run = 0; std::size_t rb = 0;
                each_cp(p.s, [&](char32_t c, std::size_t b, std::size_t) {
                    int cl = tx::is_word_char(c) ? 1 : is_ws(c) ? 0 : 2;
                    if (cl != run) { if (run) m.push_back({rb, b}); run = cl; rb = b; }
                });
                if (run) m.push_back({rb, p.s.size()});
                split_by(p, m, behavior::removed, true, out); break; }
            case whitespace_split: split_by(p, char_spans(p.s, is_ws), behavior::removed, false, out); break;
            case punctuation: split_by(p, char_spans(p.s, is_punc), beh, false, out); break;
            case digits: split_by(p, char_spans(p.s, [](char32_t c) { return tx::is_number(c); }), individual ? behavior::isolated : behavior::contiguous, false, out); break;
            case char_delim: split_by(p, char_spans(p.s, [this](char32_t c) { return c == delim; }), behavior::removed, false, out); break;
            case metaspace: {
                replace_all(p.s, " ", repl);
                if (!starts_with(p.s, repl) && (prepend == 0 || (prepend == 1 && p.first))) p.s.insert(0, repl);
                if (!do_split) { out.push_back(p); break; }
                std::vector<std::pair<std::size_t, std::size_t>> m;
                for (std::size_t q = 0; (q = p.s.find(repl, q)) != std::string::npos; q += repl.size()) m.push_back({q, q + repl.size()});
                split_by(p, m, behavior::merged_next, false, out); break; }
            default: break;
            }
        }
        ps = std::move(out);
    }
};

// ------------------------------------------------------------------------------------------------ decoders
struct decoder {
    enum kind { none, byte_level, metaspace, wordpiece, bpe, byte_fallback, fuse, strip, replace, sequence } k = none;
    std::string repl = "\xE2\x96\x81"; int prepend = 0;     // Metaspace
    std::string prefix = "##"; bool cleanup = true;         // WordPiece
    std::string suffix = "</w>";                            // BPEDecoder
    std::string content = " "; std::size_t start = 0, stop = 0;   // Strip (one character) / Replace content
    pattern pat;
    std::vector<decoder> seq;

    decoder() = default;
    explicit decoder(const json::value& j, int depth = 0) {
        if (j.is_null()) return;
        if (depth > 16) raise(errc::format_error, "tokenizer: decoders nested too deep");
        auto t = j.str_or("type", "");
        auto flag = [&](const char* key, bool d) { auto* v = j.find(key); return v && v->is_bool() ? v->b() : d; };
        auto count = [&](const char* key) { double v = j.num_or(key, 0); return v > 0 && v < 1e9 ? std::size_t(v) : std::size_t(0); };
        if (t == "ByteLevel") k = byte_level;
        else if (t == "Metaspace") { k = metaspace; repl = j.str_or("replacement", repl); prepend = pre_tokenizer::prepend_scheme(j); }
        else if (t == "WordPiece") { k = wordpiece; prefix = j.str_or("prefix", "##"); cleanup = flag("cleanup", true); }
        else if (t == "BPEDecoder") { k = bpe; suffix = j.str_or("suffix", "</w>"); }
        else if (t == "ByteFallback") k = byte_fallback;
        else if (t == "Fuse") k = fuse;
        else if (t == "Strip") { k = strip; content = j.str_or("content", " "); start = count("start"); stop = count("stop"); }
        else if (t == "Replace") { k = replace; pat = pattern(j["pattern"]); content = j.str_or("content", ""); }
        else if (t == "Sequence") { k = sequence; for (auto& n : j["decoders"].arr()) seq.emplace_back(n, depth + 1); }
        else raise(errc::unsupported, "tokenizer: decoder type '{}' is not supported", t);
    }
    static int byte_token(std::string_view t) {   // "<0x41>" -> 0x41, else -1
        auto hex = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
        if (t.size() != 6 || t.substr(0, 3) != "<0x" || t[5] != '>' || hex(t[3]) < 0 || hex(t[4]) < 0) return -1;
        return hex(t[3]) * 16 + hex(t[4]);
    }
    static void clean_up(std::string& s) {
        static const char* const r[][2] = {{" .", "."}, {" ?", "?"}, {" !", "!"}, {" ,", ","}, {" ' ", "'"}, {" n't", "n't"}, {" 'm", "'m"},
                                           {" do not", " don't"}, {" 's", "'s"}, {" 've", "'ve"}, {" 're", "'re"}};
        for (auto& p : r) replace_all(s, p[0], p[1]);
    }
    void strip_token(std::string& t) const {
        std::size_t b = 0, e = t.size();
        if (content.empty()) return;
        for (std::size_t i = 0; i < start && t.compare(b, content.size(), content) == 0 && b + content.size() <= e; ++i) b += content.size();
        for (std::size_t i = 0; i < stop && e - b >= content.size() && t.compare(e - content.size(), content.size(), content) == 0; ++i) e -= content.size();
        t = t.substr(b, e - b);
    }
    // Decoder::decode_chain over the whole token list. Invalid UTF-8 (a multi-byte character split by the caller) is
    // passed through as bytes, not replaced by U+FFFD, so concatenating streamed pieces gives the same bytes.
    void chain(std::vector<std::string>& toks) const {
        switch (k) {
        case none: break;
        case byte_level: { std::string o; for (auto& t : toks) byte_map::get().decode(t, o); toks.assign(1, std::move(o)); break; }
        case metaspace:
            for (std::size_t i = 0; i < toks.size(); ++i) replace_all(toks[i], repl, i == 0 && prepend != 2 ? "" : " ");
            break;
        case wordpiece:
            for (std::size_t i = 0; i < toks.size(); ++i) {
                if (i) { if (starts_with(toks[i], prefix)) toks[i].erase(0, prefix.size()); else toks[i].insert(0, " "); }
                if (cleanup) clean_up(toks[i]);
            }
            break;
        case bpe: for (std::size_t i = 0; i < toks.size(); ++i) replace_all(toks[i], suffix, i + 1 == toks.size() ? "" : " "); break;
        case byte_fallback: {
            std::vector<std::string> o; std::string bytes;
            for (auto& t : toks) {
                int b = byte_token(t);
                if (b >= 0) { bytes += char(b); continue; }
                if (!bytes.empty()) { o.push_back(std::move(bytes)); bytes.clear(); }
                o.push_back(std::move(t));
            }
            if (!bytes.empty()) o.push_back(std::move(bytes));
            toks = std::move(o); break; }
        case fuse: { std::string o; for (auto& t : toks) o += t; toks.assign(1, std::move(o)); break; }
        case strip: for (auto& t : toks) strip_token(t); break;
        case replace: for (auto& t : toks) pat.replace(t, content); break;
        case sequence: for (auto& d : seq) d.chain(toks); break;
        }
    }
    // The text one token contributes in the middle of a sequence (streaming, grammar masks): the per-token part of
    // the chain. What depends on the position in the sequence is left out: Metaspace keeps the leading space,
    // WordPiece puts a space before every word-initial piece, a Strip that follows a Fuse does nothing.
    void one(std::string& t, bool& fused) const {
        switch (k) {
        case none: case fuse: fused = fused || k == fuse; break;
        case byte_level: { std::string o; byte_map::get().decode(t, o); t = std::move(o); break; }
        case metaspace: replace_all(t, repl, " "); break;
        case wordpiece: if (starts_with(t, prefix)) t.erase(0, prefix.size()); else t.insert(0, " "); break;
        case bpe: replace_all(t, suffix, " "); break;
        case byte_fallback: if (int b = byte_token(t); b >= 0) t.assign(1, char(b)); break;
        case strip: if (!fused) strip_token(t); break;
        case replace: pat.replace(t, content); break;
        case sequence: for (auto& d : seq) d.one(t, fused); break;
        }
    }
};
}  // namespace tok

class tokenizer {
public:
    enum class model_type { bpe, wordpiece, unigram, sentencepiece_bpe, wordlevel };
    // SentencePiece piece types, also GGUF tokenizer.ggml.token_type
    enum : std::uint8_t { type_normal = 1, type_unknown = 2, type_control = 3, type_user_defined = 4, type_unused = 5, type_byte = 6 };

    tokenizer() = default;
    explicit tokenizer(std::string_view tokenizer_json) {
        auto j = json::parse(tokenizer_json);
        if (auto* v = j.find("forge_tokenizer_version"); v && v->is_num() && v->num() > tokenizer_format_version)
            raise(errc::unsupported, "tokenizer section version {} is newer than this build reads ({})", v->num(), tokenizer_format_version);
        const auto& m = j["model"];
        auto type = m.str_or("type", "BPE");
        auto flag = [&](const char* key, bool d) { auto* v = m.find(key); return v && v->is_bool() ? v->b() : d; };
        auto str = [&](const char* key, const char* d) { auto* v = m.find(key); return v && v->is_str() ? v->str() : std::string(d); };
        if (type == "BPE") {
            kind_ = model_type::bpe;
            // Version 1: FORGE wrote {"model": ..., "added_tokens": ...} and nothing else.
            legacy_ = !j.find("normalizer") && !j.find("pre_tokenizer") && !j.find("decoder") && !j.find("forge_tokenizer_version");
            for (auto& [t, id] : m["vocab"].obj()) add(t, id.num());
            cont_prefix_ = str("continuing_subword_prefix", ""); eow_suffix_ = str("end_of_word_suffix", "");
            fuse_unk_ = flag("fuse_unk", false); byte_fallback_ = flag("byte_fallback", false); ignore_merges_ = flag("ignore_merges", false);
            unk_ = token_id(str("unk_token", ""));
            int rank = 0;
            if (auto* mg = m.find("merges"); mg && mg->is_arr()) for (auto& e : mg->arr()) {
                std::string a, b;
                if (e.is_str()) { auto& s = e.str(); auto sp = s.find(' '); if (sp == std::string::npos) continue; a = s.substr(0, sp); b = s.substr(sp + 1); }
                else { a = tok::at(e, 0).str(); b = tok::at(e, 1).str(); }
                int ia = token_id(a), ib = token_id(b);
                int in = token_id(a + (cont_prefix_.empty() || !tok::starts_with(b, cont_prefix_) ? b : b.substr(cont_prefix_.size())));
                if (ia >= 0 && ib >= 0 && in >= 0) merges_.try_emplace(pair_key(ia, ib), rank, in);
                ++rank;
            }
        } else if (type == "WordPiece") {
            kind_ = model_type::wordpiece;
            for (auto& [t, id] : m["vocab"].obj()) add(t, id.num());
            cont_prefix_ = str("continuing_subword_prefix", "##");
            unk_ = token_id(str("unk_token", "[UNK]"));
            max_chars_ = std::size_t(std::clamp(m.num_or("max_input_chars_per_word", 100), 1.0, 1e6));
        } else if (type == "WordLevel") {
            kind_ = model_type::wordlevel;
            for (auto& [t, id] : m["vocab"].obj()) add(t, id.num());
            unk_ = token_id(str("unk_token", "<unk>"));
        } else if (type == "Unigram" || type == "SentencePieceBPE") {
            kind_ = type == "Unigram" ? model_type::unigram : model_type::sentencepiece_bpe;
            auto& v = m["vocab"].arr();
            if (v.size() > max_vocab) raise(errc::format_error, "tokenizer: vocabulary of {} pieces is too large", v.size());
            id_to_tok_.reserve(v.size()); scores_.reserve(v.size());
            for (auto& e : v) { int id = int(id_to_tok_.size()); id_to_tok_.push_back(tok::at(e, 0).str()); scores_.push_back(tok::at(e, 1).num()); tok_to_id_.try_emplace(e[0].str(), id); }
            if (auto* tt = m.find("token_types"); tt && tt->is_arr()) {
                if (tt->arr().size() != v.size()) raise(errc::format_error, "tokenizer: token_types has {} entries for {} pieces", tt->arr().size(), v.size());
                for (auto& t : tt->arr()) types_.push_back(std::uint8_t(std::clamp(t.num(), 0.0, 255.0)));
            }
            unk_ = int(m.num_or("unk_id", -1));
            if (unk_ >= int(id_to_tok_.size())) raise(errc::format_error, "tokenizer: unk_id {} is outside the vocabulary", unk_);
            byte_fallback_ = flag("byte_fallback", false);
            min_score_ = 0; bool any = false;
            for (std::size_t i = 0; i < scores_.size(); ++i) if (!any || scores_[i] < min_score_) { min_score_ = scores_[i]; any = true; }
            if (kind_ == model_type::unigram) {   // sorted piece index: the prefix search of the Viterbi lattice
                for (std::size_t i = 0; i < id_to_tok_.size(); ++i)
                    if (!id_to_tok_[i].empty() && tok_to_id_.find(id_to_tok_[i])->second == int(i) && matchable(int(i))) sorted_.push_back(int(i));
                std::sort(sorted_.begin(), sorted_.end(), [&](int a, int b) { return id_to_tok_[std::size_t(a)] < id_to_tok_[std::size_t(b)]; });
            }
        } else raise(errc::unsupported, "tokenizer: model type '{}' is not supported (BPE, WordPiece, WordLevel, Unigram, SentencePieceBPE)", type);
        for (int b = 0; b < 256; ++b) { char buf[8]; std::snprintf(buf, sizeof buf, "<0x%02X>", b); byte_tok_[std::size_t(b)] = token_id(buf); }

        if (legacy_) { pre_.k = tok::pre_tokenizer::legacy; dec_.k = tok::decoder::byte_level; }
        else {
            if (auto* n = j.find("normalizer")) norm_ = tok::normalizer(*n);
            if (auto* p = j.find("pre_tokenizer")) pre_ = tok::pre_tokenizer(*p);
            if (auto* d = j.find("decoder")) dec_ = tok::decoder(*d);
            if (auto* p = j.find("post_processor")) post_processor(*p, 0);
        }
        if (auto* at = j.find("added_tokens"); at && at->is_arr())
            for (auto& t : at->arr()) {
                added a; a.content = t["content"].str(); a.id = int(t["id"].num());
                auto fl = [&](const char* key, bool d) { auto* v = t.find(key); return v && v->is_bool() ? v->b() : d; };
                a.special = fl("special", legacy_); a.normalized = fl("normalized", !a.special) && !legacy_;
                a.single_word = fl("single_word", false); a.lstrip = fl("lstrip", false); a.rstrip = fl("rstrip", false);
                add(a.content, double(a.id));
                if (a.content.empty()) continue;
                if (a.special) { if (special_ids_.size() <= std::size_t(a.id)) special_ids_.resize(std::size_t(a.id) + 1); special_ids_[std::size_t(a.id)] = true; }
                added_.push_back(std::move(a));
            }
        for (std::size_t i = 0; i < added_.size(); ++i) {
            auto& a = added_[i];
            if (a.normalized && !norm_.identity()) {
                // matched in the normalized text, and (as `tokenizers` does) decoded as its normalized form
                std::string c = a.content; norm_.apply(c);
                if (!c.empty()) id_to_tok_[std::size_t(a.id)] = c;
                norm_match_.add(std::move(c), int(i));
            }
            else raw_match_.add(a.content, int(i));
        }
        if (auto* b = j.find("bos_id")) bos_ = int(b->num());
        if (auto* e = j.find("eos_id")) eos_ = int(e->num());
    }

    model_type type() const { return kind_; }
    std::size_t vocab_size() const { return id_to_tok_.size(); }
    int bos() const { return bos_; }
    int eos() const { return eos_; }
    int unk() const { return unk_; }
    void set_special(int bos, int eos) { bos_ = bos; eos_ = eos; }
    // Whether a prompt starts with BOS / ends with EOS (when the model has one). The session sets these from the
    // manifest's "add_bos" / "add_eos"; once set they take the place of the tokenizer.json post-processor.
    bool add_bos() const { return explicit_ || tmpl_.empty() ? add_bos_ && bos_ >= 0 : !tmpl_.empty() && tmpl_.front() >= 0; }
    bool add_eos() const { return explicit_ || tmpl_.empty() ? add_eos_ && eos_ >= 0 : !tmpl_.empty() && tmpl_.back() >= 0; }
    void set_add_bos(bool on) { add_bos_ = on; explicit_ = true; }
    void set_add_eos(bool on) { add_eos_ = on; explicit_ = true; }
    // True when tokenizer.json has a post-processor that adds special tokens (TemplateProcessing, BertProcessing, ...).
    bool has_post_processor() const { return !tmpl_.empty(); }

    // Text -> ids, without BOS / EOS / post-processor tokens. Added tokens written out in the text ("<|im_start|>")
    // become their ids. Input that is not valid UTF-8 is read with U+FFFD in place of the bad bytes.
    std::vector<int> encode(std::string_view text) const {
        std::vector<int> out;
        std::string fixed;
        if (!legacy_ && !tok::tx::is_valid_utf8(text)) { fixed = tok::tx::sanitize_utf8(text); text = fixed; }
        split_added(text, raw_match_, out, [&](std::string_view part, bool first) {
            if (norm_.identity()) { tokenize(part, first, out); return; }
            std::string n(part); norm_.apply(n);
            split_added(n, norm_match_, out, [&](std::string_view sub, bool sub_first) { tokenize(sub, first && sub_first, out); });
        });
        return out;
    }
    // add_special: also what the model expects around a prompt -- the post-processor template, or BOS / EOS as set.
    std::vector<int> encode(std::string_view text, bool add_special) const {
        auto ids = encode(text);
        return add_special ? with_special(std::move(ids)) : ids;
    }
    std::vector<int> with_special(std::vector<int> ids) const {
        if (explicit_ || tmpl_.empty()) {
            if (add_bos()) ids.insert(ids.begin(), bos_);
            if (add_eos()) ids.push_back(eos_);
            return ids;
        }
        std::vector<int> out; out.reserve(ids.size() + tmpl_.size());
        for (int t : tmpl_) { if (t >= 0) out.push_back(t); else out.insert(out.end(), ids.begin(), ids.end()); }
        return out;
    }
    // Ids together with the segment ("token type") id of each: what BERT-family encoders take as a second input.
    struct encoding { std::vector<int> ids, type_ids; };
    // One text with the special tokens around it (as encode(text, true)) and the post-processor's segment ids.
    encoding encode_typed(std::string_view text) const {
        encoding e; auto ids = encode(text);
        if (explicit_ || tmpl_.empty()) { e.ids = with_special(std::move(ids)); e.type_ids.assign(e.ids.size(), 0); return e; }
        for (std::size_t i = 0; i < tmpl_.size(); ++i) {
            if (tmpl_[i] >= 0) { e.ids.push_back(tmpl_[i]); e.type_ids.push_back(tmpl_type_[i]); }
            else { e.ids.insert(e.ids.end(), ids.begin(), ids.end()); e.type_ids.insert(e.type_ids.end(), ids.size(), tmpl_type_[i]); }
        }
        return e;
    }
    // A pair of texts (question / passage, sentence A / B) through the post-processor's pair template:
    // "[CLS] A [SEP] B [SEP]" with B in segment 1 for BERT. Without a pair template: A then B (segments 0 and 1)
    // inside the BOS / EOS that one text gets.
    encoding encode_pair(std::string_view a, std::string_view b) const {
        encoding e; auto ia = encode(a), ib = encode(b);
        auto put = [&](const std::vector<int>& v, int type) { e.ids.insert(e.ids.end(), v.begin(), v.end()); e.type_ids.insert(e.type_ids.end(), v.size(), type); };
        if (explicit_ || pair_.empty()) {
            if (add_bos()) put({explicit_ || tmpl_.empty() ? bos_ : tmpl_.front()}, 0);
            put(ia, 0); put(ib, 1);
            if (add_eos()) put({explicit_ || tmpl_.empty() ? eos_ : tmpl_.back()}, 1);
            return e;
        }
        for (auto& x : pair_) { if (x.id >= 0) put({x.id}, x.type); else put(x.id == -1 ? ia : ib, x.type); }
        return e;
    }
    // ids -> text through the tokenizer.json decoder. Special tokens are written out unless skip_special.
    std::string decode(const std::vector<int>& ids, bool skip_special = false) const {
        if (legacy_) { std::string s; for (int i : ids) if (!skip_special || !is_special(i)) s += decode_one(i); return s; }
        std::vector<std::string> toks;
        for (int i : ids) if (i >= 0 && std::size_t(i) < id_to_tok_.size() && !(skip_special && is_special(i))) toks.push_back(id_to_tok_[std::size_t(i)]);
        bool had = dec_.k != tok::decoder::none;
        dec_.chain(toks);
        std::string s;
        for (std::size_t i = 0; i < toks.size(); ++i) { if (i && !had) s += ' '; s += toks[i]; }
        return s;
    }
    // The bytes one token adds in the middle of a text (streaming output, grammar masks). See tok::decoder::one for
    // what differs from decode(): a SentencePiece word keeps its leading space, and so on.
    std::string decode_one(int id) const {
        if (id < 0 || std::size_t(id) >= id_to_tok_.size()) return "";
        std::string t = id_to_tok_[std::size_t(id)];
        if (is_special(id)) return t;
        if (legacy_) {
            auto& bm = tok::byte_map::get(); std::string o;
            tok::each_cp(t, [&](char32_t c, std::size_t b, std::size_t e) { if (c < bm.dec.size() && bm.dec[c] >= 0) o += char(bm.dec[c]); else o.append(t, b, e - b); });
            return o;
        }
        bool fused = false; dec_.one(t, fused);
        return t;
    }
    // The first two pipeline stages on their own (no added tokens): what the model is given to tokenize.
    std::string normalize(std::string_view text) const { std::string s(text); norm_.apply(s); return s; }
    std::vector<std::string> pre_tokenize(std::string_view text) const {
        std::vector<tok::piece> ps; std::vector<std::string> o;
        if (auto n = normalize(text); !n.empty()) ps.push_back({std::move(n), true});
        pre_.apply(ps);
        for (auto& p : ps) o.push_back(std::move(p.s));
        return o;
    }
    int token_id(std::string_view t) const { auto it = tok_to_id_.find(t); return it == tok_to_id_.end() ? -1 : it->second; }
    // The vocabulary string of an id ("" when out of range): "Ġthe", "▁the", "##ing", "<|im_end|>".
    const std::string& id_to_token(int id) const { static const std::string none; return id < 0 || std::size_t(id) >= id_to_tok_.size() ? none : id_to_tok_[std::size_t(id)]; }
    bool is_special(int id) const { return id >= 0 && std::size_t(id) < special_ids_.size() && special_ids_[std::size_t(id)]; }

private:
    static constexpr std::size_t max_vocab = 8u << 20;
    struct added { std::string content; int id = 0; bool special = false, normalized = false, single_word = false, lstrip = false, rstrip = false; };
    // Added-token matcher: leftmost match, the longest at that position (version 1: the first listed).
    struct matcher {
        std::vector<std::pair<std::string, int>> pats; std::array<std::vector<int>, 256> by_first;
        void add(std::string p, int idx) { if (p.empty()) return; by_first[(unsigned char)p[0]].push_back(int(pats.size())); pats.push_back({std::move(p), idx}); }
        bool empty() const { return pats.empty(); }
    };

    static std::uint64_t pair_key(int a, int b) { return std::uint64_t(std::uint32_t(a)) << 32 | std::uint32_t(b); }
    void add(const std::string& t, double idv) {
        if (!(idv >= 0) || idv > double(max_vocab)) raise(errc::format_error, "tokenizer: token id {} out of range", idv);
        auto id = std::size_t(idv);
        if (id >= id_to_tok_.size()) id_to_tok_.resize(id + 1);
        id_to_tok_[id] = t; tok_to_id_[t] = int(id);
    }
    bool matchable(int id) const {   // pieces the Unigram lattice / SentencePiece merges may produce
        if (types_.empty()) return true;
        auto t = types_[std::size_t(id)];
        return t == type_normal || t == type_user_defined || t == type_unused || t == 0;
    }

    void post_processor(const json::value& j, int depth) {
        if (j.is_null() || depth > 8) return;
        auto t = j.str_or("type", "");
        if (t == "Sequence") { if (auto* ps = j.find("processors"); ps && ps->is_arr()) for (auto& p : ps->arr()) post_processor(p, depth + 1); }
        else if (t == "TemplateProcessing") {
            auto* sp = j.find("special_tokens");
            // one template ("single" or "pair"): special token ids, and -1 / -2 where sequence A / B goes
            auto items = [&](const json::value& list) {
                std::vector<titem> out;
                for (auto& e : list.arr()) {
                    auto type_of = [](const json::value& o) { auto* v = o.find("type_id"); return v && v->is_num() && v->num() >= 0 && v->num() < 256 ? int(v->num()) : 0; };
                    if (auto* sq = e.find("Sequence")) { auto* id = sq->find("id"); out.push_back({id && id->is_str() && id->str() == "B" ? -2 : -1, type_of(*sq)}); continue; }
                    auto& st = e["SpecialToken"]; auto& name = st["id"].str();
                    const json::value* ids = sp && sp->find(name) ? (*sp)[name].find("ids") : nullptr;
                    if (!ids || !ids->is_arr()) raise(errc::format_error, "tokenizer: TemplateProcessing special token {} has no ids", name);
                    for (auto& id : ids->arr()) out.push_back({checked_id(id.num()), type_of(st)});
                    if (out.size() > 64) raise(errc::format_error, "tokenizer: TemplateProcessing template is too long");
                }
                return out;
            };
            auto single = items(j["single"]);
            auto count = [](const std::vector<titem>& v, int what) { return std::count_if(v.begin(), v.end(), [&](const titem& x) { return x.id == what; }); };
            if (count(single, -1) != 1 || count(single, -2) != 0) raise(errc::format_error, "tokenizer: unsupported TemplateProcessing single-sequence template");
            tmpl_.clear(); tmpl_type_.clear(); pair_.clear();
            for (auto& x : single) { tmpl_.push_back(x.id); tmpl_type_.push_back(std::uint8_t(x.type)); }
            if (auto* pr = j.find("pair"); pr && pr->is_arr()) {
                pair_ = items(*pr);
                if (count(pair_, -1) != 1 || count(pair_, -2) != 1) pair_.clear();
            }
            if (tmpl_.size() == 1 && tmpl_type_[0] == 0) { tmpl_.clear(); tmpl_type_.clear(); }
        } else if (t == "BertProcessing" || t == "RobertaProcessing") {
            int cls = checked_id(tok::at(j["cls"], 1).num()), sep = checked_id(tok::at(j["sep"], 1).num());
            tmpl_ = {cls, -1, sep}; tmpl_type_ = {0, 0, 0};
            if (t == "BertProcessing") pair_ = {{cls, 0}, {-1, 0}, {sep, 0}, {-2, 1}, {sep, 1}};        // [CLS] A [SEP] B [SEP], B in segment 1
            else pair_ = {{cls, 0}, {-1, 0}, {sep, 0}, {sep, 0}, {-2, 0}, {sep, 0}};                    // <s> A </s></s> B </s>
        } else if (t != "ByteLevel") raise(errc::unsupported, "tokenizer: post-processor type '{}' is not supported", t);
    }
    static int checked_id(double v) { if (!(v >= 0) || v > double(max_vocab)) raise(errc::format_error, "tokenizer: special token id {} out of range", v); return int(v); }

    // Splits `s` on the added tokens of `m`: ids go to `out`, the text in between to text(part, starts_at_offset_0).
    template <class F> void split_added(std::string_view s, const matcher& m, std::vector<int>& out, F&& text) const {
        if (m.empty()) { if (!s.empty()) text(s, true); return; }
        std::size_t done = 0;   // everything before this is emitted
        for (std::size_t p = 0; p < s.size();) {
            auto& cand = m.by_first[(unsigned char)s[p]];
            const std::pair<std::string, int>* best = nullptr;
            for (int ci : cand) {
                auto& pt = m.pats[std::size_t(ci)];
                if (s.compare(p, pt.first.size(), pt.first) != 0) continue;
                if (!best || (!legacy_ && pt.first.size() > best->first.size())) best = &pt;
                if (legacy_) break;
            }
            if (!best) { ++p; continue; }
            auto& a = added_[std::size_t(best->second)];
            std::size_t b = p, e = p + best->first.size();
            p = e;
            if (a.single_word && ((b > 0 && word_before(s, b)) || (e < s.size() && word_at(s, e)))) continue;
            if (a.lstrip) while (b > done && tok::is_ws(cp_before(s, b, done))) b = prev_cp(s, b);
            if (a.rstrip) for (std::size_t q = e; q < s.size();) { char32_t c = tok::tx::decode_utf8(s, q); if (!tok::is_ws(c)) break; e = q; }
            if (b > done) text(s.substr(done, b - done), done == 0);
            out.push_back(a.id);
            done = e; if (p < e) p = e;
        }
        if (done < s.size()) text(s.substr(done), done == 0);
    }
    static std::size_t prev_cp(std::string_view s, std::size_t i) { --i; while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) --i; return i; }
    static char32_t cp_before(std::string_view s, std::size_t i, std::size_t floor) { std::size_t b = prev_cp(s, i); if (b < floor) return 0; return tok::tx::decode_utf8(s, b); }
    static bool word_before(std::string_view s, std::size_t i) { return tok::tx::is_word_char(cp_before(s, i, 0)); }
    static bool word_at(std::string_view s, std::size_t i) { return tok::tx::is_word_char(tok::tx::decode_utf8(s, i)); }

    void tokenize(std::string_view text, bool first, std::vector<int>& out) const {
        if (text.empty()) return;
        std::vector<tok::piece> ps; ps.push_back({std::string(text), first});
        pre_.apply(ps);
        for (auto& p : ps) {
            switch (kind_) {
            case model_type::bpe: bpe(p.s, out); break;
            case model_type::wordpiece: wordpiece(p.s, out); break;
            case model_type::unigram: unigram(p.s, out); break;
            case model_type::sentencepiece_bpe: sp_bpe(p.s, out); break;
            case model_type::wordlevel: if (int id = token_id(p.s); id >= 0) out.push_back(id); else if (unk_ >= 0) out.push_back(unk_); break;
            }
        }
    }
    bool bytes_of(std::string_view s, std::vector<int>& out) const {   // byte_fallback: "<0xE2>" "<0x96>" ...
        for (unsigned char c : s) if (byte_tok_[c] < 0) return false;
        for (unsigned char c : s) out.push_back(byte_tok_[c]);
        return true;
    }

    // tokenizers' BPE: one symbol per character, then merges in rank order (lowest rank first, leftmost on a tie).
    void bpe(std::string_view w, std::vector<int>& out) const {
        if (ignore_merges_) if (int id = token_id(w); id >= 0) { out.push_back(id); return; }
        struct sym { int id; int prev, next; bool live; };
        std::vector<sym> s; std::string buf;
        bool pending_unk = false;
        auto push = [&](int id) { int n = int(s.size()); s.push_back({id, n - 1, -1, true}); if (n) s[std::size_t(n - 1)].next = n; };
        tok::each_cp(w, [&](char32_t, std::size_t b, std::size_t e) {
            std::string_view c = w.substr(b, e - b);
            if ((b && !cont_prefix_.empty()) || (e == w.size() && !eow_suffix_.empty())) {
                buf.clear(); if (b) buf += cont_prefix_; buf += c; if (e == w.size()) buf += eow_suffix_; c = buf;
            }
            if (int id = token_id(c); id >= 0) { if (pending_unk) { push(unk_); pending_unk = false; } push(id); return; }
            if (byte_fallback_) {
                std::vector<int> bs;
                if (bytes_of(c, bs)) { for (int x : bs) push(x); return; }
            }
            if (unk_ >= 0) { if (pending_unk && !fuse_unk_) push(unk_); pending_unk = true; }
            // no unk token: the character is dropped (version 1 behaviour, and tokenizers')
        });
        if (pending_unk) push(unk_);
        if (s.size() > 1 && !merges_.empty()) {
            struct cand { int rank, pos, id; bool operator<(const cand& o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; } };
            std::priority_queue<cand> q;
            auto consider = [&](int i) {
                if (i < 0 || s[std::size_t(i)].next < 0) return;
                auto it = merges_.find(pair_key(s[std::size_t(i)].id, s[std::size_t(s[std::size_t(i)].next)].id));
                if (it != merges_.end()) q.push({it->second.first, i, it->second.second});
            };
            for (int i = 0; i + 1 < int(s.size()); ++i) consider(i);
            while (!q.empty()) {
                auto c = q.top(); q.pop();
                auto& a = s[std::size_t(c.pos)];
                if (!a.live || a.next < 0) continue;
                auto& b = s[std::size_t(a.next)];
                auto it = merges_.find(pair_key(a.id, b.id));
                if (it == merges_.end() || it->second.second != c.id) continue;   // stale entry
                a.id = c.id; b.live = false; a.next = b.next;
                if (a.next >= 0) s[std::size_t(a.next)].prev = c.pos;
                consider(a.prev); consider(c.pos);
            }
        }
        for (auto& x : s) if (x.live) out.push_back(x.id);
    }

    void wordpiece(std::string_view w, std::vector<int>& out) const {
        std::size_t at = out.size(), chars = tok::tx::count_code_points(w);
        bool bad = chars > max_chars_;
        std::string buf;
        for (std::size_t start = 0; !bad && start < w.size();) {
            std::size_t end = w.size(); int id = -1;
            while (start < end) {
                std::string_view sub = w.substr(start, end - start);
                if (start) { buf = cont_prefix_; buf += sub; sub = buf; }
                if ((id = token_id(sub)) >= 0) break;
                end = prev_cp(w, end);
            }
            if (id < 0) { bad = true; break; }
            out.push_back(id); start = end;
        }
        if (bad) { out.resize(at); if (unk_ >= 0) out.push_back(unk_); }
    }

    // Unigram: the best-scoring segmentation (Viterbi), an unknown character costing min_score - 10. Runs of unknown
    // characters fuse into one unk, or into byte tokens with byte_fallback.
    void unigram(std::string_view w, std::vector<int>& out) const {
        struct node { int id = -1; double score = 0; std::size_t from = std::string::npos; };
        std::vector<node> best(w.size() + 1);
        double unk_score = min_score_ - 10.0;
        for (std::size_t st = 0; st < w.size();) {
            std::size_t nx = st; tok::tx::decode_utf8(w, nx);
            double here = best[st].score; bool single = false;
            if (st == 0 || best[st].from != std::string::npos) {
                // every piece that is a prefix of w[st..]: narrow the sorted index one byte at a time
                auto lo = sorted_.begin(), hi = sorted_.end();
                for (std::size_t len = 1; st + len <= w.size() && lo != hi; ++len) {
                    unsigned char c = (unsigned char)w[st + len - 1];
                    auto byte_at = [&](int id) { auto& t = id_to_tok_[std::size_t(id)]; return t.size() >= len ? int((unsigned char)t[len - 1]) : -1; };
                    lo = std::partition_point(lo, hi, [&](int id) { return byte_at(id) < int(c); });
                    hi = std::partition_point(lo, hi, [&](int id) { return byte_at(id) <= int(c); });
                    if (lo != hi && id_to_tok_[std::size_t(*lo)].size() == len) {
                        auto& t = best[st + len]; double cand = scores_[std::size_t(*lo)] + here;
                        if (t.from == std::string::npos || cand > t.score) { t.score = cand; t.from = st; t.id = *lo; }
                        if (st + len == nx) single = true;
                    }
                }
                if (!single) {
                    auto& t = best[nx]; double cand = unk_score + here;
                    if (t.from == std::string::npos || cand > t.score) { t.score = cand; t.from = st; t.id = -1; }
                }
            }
            st = nx;
        }
        std::vector<std::pair<std::size_t, int>> path;   // (start, id) back to front
        for (std::size_t e = w.size(); e > 0;) { auto& n = best[e]; if (n.from == std::string::npos) return; path.push_back({n.from, n.id}); e = n.from; }
        std::size_t end = w.size();
        std::vector<std::pair<std::string_view, int>> segs;
        for (auto& [from, id] : path) { segs.push_back({w.substr(from, end - from), id == unk_ ? -1 : id}); end = from; }
        std::reverse(segs.begin(), segs.end());
        for (std::size_t i = 0; i < segs.size(); ++i) {
            if (segs[i].second >= 0) { out.push_back(segs[i].second); continue; }
            std::size_t j = i; while (j + 1 < segs.size() && segs[j + 1].second < 0) ++j;   // fuse the run
            std::string_view run(segs[i].first.data(), std::size_t(segs[j].first.data() + segs[j].first.size() - segs[i].first.data()));
            if (int id = token_id(run); id >= 0 && id != unk_) out.push_back(id);
            else if (!(byte_fallback_ && bytes_of(run, out)) && unk_ >= 0) out.push_back(unk_);
            i = j;
        }
    }

    // SentencePiece BPE: one symbol per character; merge the adjacent pair whose concatenation is the best-scoring
    // piece (leftmost on a tie) until none is a piece. "Unused" pieces merge but are split again on output.
    void sp_bpe(std::string_view w, std::vector<int>& out) const {
        struct sym { std::size_t pos, len; int prev, next; };
        std::vector<sym> s;
        tok::each_cp(w, [&](char32_t, std::size_t b, std::size_t e) { int n = int(s.size()); s.push_back({b, e - b, n - 1, -1}); if (n) s[std::size_t(n - 1)].next = n; });
        struct cand { double score; int left, right; std::size_t size; bool operator<(const cand& o) const { return score != o.score ? score < o.score : left > o.left; } };
        std::priority_queue<cand> q;
        std::unordered_map<std::string_view, std::pair<std::string_view, std::string_view>> rev;   // unused piece -> its two halves
        auto consider = [&](int l, int r) {
            if (l < 0 || r < 0) return;
            std::string_view p = w.substr(s[std::size_t(l)].pos, s[std::size_t(l)].len + s[std::size_t(r)].len);
            int id = token_id(p);
            if (id < 0 || std::size_t(id) >= scores_.size() || !matchable(id)) return;
            q.push({scores_[std::size_t(id)], l, r, p.size()});
            if (!types_.empty() && types_[std::size_t(id)] == type_unused) rev[p] = {w.substr(s[std::size_t(l)].pos, s[std::size_t(l)].len), w.substr(s[std::size_t(r)].pos, s[std::size_t(r)].len)};
        };
        for (int i = 1; i < int(s.size()); ++i) consider(i - 1, i);
        while (!q.empty()) {
            auto c = q.top(); q.pop();
            auto& l = s[std::size_t(c.left)]; auto& r = s[std::size_t(c.right)];
            if (!l.len || !r.len || l.len + r.len != c.size || l.next != c.right) continue;   // stale
            l.len += r.len; r.len = 0; l.next = r.next;
            if (l.next >= 0) s[std::size_t(l.next)].prev = c.left;
            consider(l.prev, c.left); consider(c.left, l.next);
        }
        bool prev_unk = false;
        auto emit = [&](auto&& self, std::string_view p, int depth) -> void {
            int id = token_id(p);
            if (id >= 0 && std::size_t(id) < scores_.size() && matchable(id) && !(!types_.empty() && types_[std::size_t(id)] == type_unused)) { out.push_back(id); prev_unk = false; return; }
            if (auto it = rev.find(p); it != rev.end() && depth < 64) { auto halves = it->second; self(self, halves.first, depth + 1); self(self, halves.second, depth + 1); return; }
            if (byte_fallback_ && bytes_of(p, out)) { prev_unk = false; return; }
            if (unk_ >= 0 && !prev_unk) out.push_back(unk_);
            prev_unk = true;
        };
        for (auto& x : s) if (x.len) emit(emit, w.substr(x.pos, x.len), 0);
    }

    model_type kind_ = model_type::bpe;
    bool legacy_ = false;
    tok::vocab_map tok_to_id_;
    std::vector<std::string> id_to_tok_;
    std::unordered_map<std::uint64_t, std::pair<int, int>> merges_;   // (left id, right id) -> (rank, merged id)
    std::string cont_prefix_, eow_suffix_;
    bool fuse_unk_ = false, byte_fallback_ = false, ignore_merges_ = false;
    std::size_t max_chars_ = 100;
    std::vector<double> scores_; std::vector<std::uint8_t> types_; std::vector<int> sorted_; double min_score_ = 0;
    std::array<int, 256> byte_tok_{};
    tok::normalizer norm_; tok::pre_tokenizer pre_; tok::decoder dec_;
    std::vector<added> added_; matcher raw_match_, norm_match_; std::vector<bool> special_ids_;
    struct titem { int id = 0; int type = 0; };   // id >= 0: a special token; -1 / -2: sequence A / B. type: its segment id
    std::vector<int> tmpl_;   // post-processor for one sequence: special token ids, -1 where the text goes
    std::vector<std::uint8_t> tmpl_type_;   // the segment (token type) id of each tmpl_ entry
    std::vector<titem> pair_;   // post-processor for a pair of sequences; empty when tokenizer.json has none
    int bos_ = -1, eos_ = -1, unk_ = -1; bool add_bos_ = true, add_eos_ = false, explicit_ = false;
};

// The name the class had while it only did byte-level BPE.
using bpe_tokenizer = tokenizer;

} // namespace forge::modelfile
