#pragma once
// The two forms of a blueprint (FORGE_PLAN D.3 "Two forms: binary for shipping, text for authoring. Lossless
// round-trip"), and the canonical text form that makes "lossless" checkable.
//
// Text is what blueprint.hpp parses. A blueprint is a program (params, repeat loops, composites, expressions over
// symbolic dimensions), so the binary form encodes that program, not one unrolled graph: it is the token stream of
// the text with the names interned and the numbers typed.
//
//   "FORGEBPB" | u32 version (1) | u32 opset | str name | str session
//   | varint n_strings | n x { u8 kind; str text }        kind 0 identifier, 1 string literal, 2 identifier that is called
//   | varint n_tokens  | tokens                                  (an op, a composite or a built-in: the "ops by name")
//   token: u8 tag
//     0x20 .. 0xff identifier                string table index tag - 0x20 (the first 224 strings: one byte per use)
//     0x01 identifier, 0x02 string literal   + varint index into the string table (0x01 only for an index of 224 or more)
//     0x03 integer                           + varint value (a number with an integral value below 2^53)
//     0x04 float                             + f64, little-endian (finite, positive; any other number)
//     0x10 .. 0x1e punctuation               { } ( ) [ ] , : = . + - * / ..
//   str = varint length + bytes; varint = LEB128, at most 10 bytes; integers little-endian.
//
// What survives the round trip is the token sequence: every name, number (as a value), string, operator and bracket,
// in order. What does not: comments, spacing and line breaks, and the spelling of a number ("1e-5" and "0.00001" are
// the same token). canonical_blueprint_text() prints a token sequence in one fixed layout, so
//     canonical(text) == text_of(decode(encode(text)))                       for every text that lexes, and
//     encode(text_of(decode(bytes))) == bytes                                for every byte string decode accepts.
// The second holds because decode only accepts the canonical encoding: strings unique, all used, in first-use order;
// kinds as the tokens use them; identifiers, integers and floats in the tag their index or value calls for; header
// fields equal to what the tokens say.
//
// A binary blueprint is loaded by turning it back into text and parsing that (blueprint_text / load_blueprint): there
// is one parser, and the decoder in front of it treats its input as untrusted (every length, index and count is
// checked against the bytes present).
#include <forge/modelfile/blueprint.hpp>
#include <forge/modelfile/fpk.hpp>
#include <cstdint>
#include <span>

namespace forge::modelfile {

inline constexpr char bp_bin_magic[8] = {'F', 'O', 'R', 'G', 'E', 'B', 'P', 'B'};
inline constexpr std::uint32_t bp_bin_version = 1;

inline bool is_binary_blueprint(std::span<const std::uint8_t> b) { return b.size() >= 8 && std::memcmp(b.data(), bp_bin_magic, 8) == 0; }
inline bool is_binary_blueprint(std::string_view s) { return is_binary_blueprint(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(s.data()), s.size())); }

namespace detail {
inline constexpr std::string_view bp_puncts[] = {"{", "}", "(", ")", "[", "]", ",", ":", "=", ".", "+", "-", "*", "/", ".."};
inline int bp_punct_index(std::string_view p) { for (int i = 0; i < 15; ++i) if (bp_puncts[i] == p) return i; return -1; }
inline bool bp_is_ident(std::string_view s) {
    if (s.empty() || !(std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_')) return false;
    for (char c : s) if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
    return true;
}
inline bool bp_integral(double v) { return v >= 0 && v < 9007199254740992.0 && v == std::floor(v); }
// The canonical spelling of a number token: digits for an integral value below 2^53, else the shortest %g form that
// reads back as the same double.
inline std::string bp_number_text(double v) {
    if (bp_integral(v)) return std::to_string(static_cast<std::uint64_t>(v));
    char b[40];
    for (int p = 1; p <= 17; ++p) {
        std::snprintf(b, sizeof b, "%.*g", p, v);
        for (char* c = b; *c; ++c) if (*c == ',') *c = '.';   // a locale with a decimal comma
        char* end = nullptr;
        if (std::strtod(b, &end) == v && end && !*end) break;
    }
    return b;
}
// name, opset and session as the leading tokens and the top-level `session` declaration state them
struct bp_header { std::string name; std::uint32_t opset = 1; std::string session = "one_shot"; };
inline bp_header bp_header_of(const std::vector<bp_token>& t) {
    bp_header h;
    auto id = [&](std::size_t i, std::string_view s) { return i < t.size() && t[i].kind == bp_token::k_ident && t[i].text == s; };
    if (id(0, "blueprint") && t.size() > 1 && t[1].kind == bp_token::k_str) h.name = t[1].text;
    if (id(2, "opset") && t.size() > 3 && t[3].kind == bp_token::k_num && t[3].num >= 0 && t[3].num <= 1e6 && t[3].num == std::floor(t[3].num)) h.opset = std::uint32_t(t[3].num);
    int depth = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (t[i].kind == bp_token::k_punct) { if (t[i].text == "{") ++depth; else if (t[i].text == "}") --depth; }
        else if (depth == 0 && id(i, "session") && i + 1 < t.size() && t[i + 1].kind == bp_token::k_ident) h.session = t[i + 1].text;
    }
    return h;
}
inline bool bp_called(const std::vector<bp_token>& t, std::size_t i) {   // identifier i is the name of a call (not a member call)
    return i + 1 < t.size() && t[i + 1].kind == bp_token::k_punct && t[i + 1].text == "(" && !(i > 0 && t[i - 1].kind == bp_token::k_punct && t[i - 1].text == ".");
}
} // namespace detail

// One fixed layout for a token sequence (without the lexer's end marker, or with it). The result lexes back to the
// same tokens; it carries no comments.
inline std::string canonical_blueprint_text(const std::vector<bp_token>& toks) {
    std::size_t n = toks.size();
    if (n && toks[n - 1].kind == bp_token::k_end) --n;
    auto is_p = [&](std::size_t i, std::string_view s) { return i < n && toks[i].kind == bp_token::k_punct && toks[i].text == s; };
    auto is_id = [&](std::size_t i, std::string_view s) { return i < n && toks[i].kind == bp_token::k_ident && toks[i].text == s; };
    auto text = [&](std::size_t i) -> std::string {
        const auto& t = toks[i];
        return t.kind == bp_token::k_num ? detail::bp_number_text(t.num) : t.kind == bp_token::k_str ? "\"" + t.text + "\"" : t.text;
    };
    // does a statement start at token i (inside braces, outside parentheses)?
    auto stmt_start = [&](std::size_t i) {
        if (toks[i].kind != bp_token::k_ident) return false;
        if (toks[i].text == "repeat" || toks[i].text == "return") return true;
        if (i > 0 && (is_p(i - 1, ",") || is_p(i - 1, "."))) return false;
        // declarations inside an `entry name { ... }` block: one per line, the declared name stays on it
        const bool decl_kw = toks[i].text == "input" || toks[i].text == "output" || toks[i].text == "graph";
        if (decl_kw && i + 1 < n && (toks[i + 1].kind == bp_token::k_ident || is_p(i + 1, "{")) && !is_p(i + 1, ":") && !is_p(i + 1, "=") && !is_p(i + 1, ",") && !is_p(i + 1, "(")) return true;
        if (i > 0 && toks[i - 1].kind == bp_token::k_ident && (toks[i - 1].text == "input" || toks[i - 1].text == "output") && is_p(i + 1, ":") && !(i > 1 && is_p(i - 2, "."))) return false;
        if (is_p(i + 1, ":")) return true;                         // params entry
        std::size_t j = i + 1;                                     // a, b, c = ...
        while (is_p(j, ",") && j + 1 < n && toks[j + 1].kind == bp_token::k_ident) j += 2;
        return is_p(j, "=");
    };
    static const std::string_view top[] = {"params", "state", "input", "output", "graph", "session", "op", "entry"};
    std::string out; int brace = 0, paren = 0; bool line_start = true;
    auto newline = [&] { if (!out.empty() && out.back() != '\n') out += '\n'; line_start = true; };
    for (std::size_t i = 0; i < n; ++i) {
        const auto& t = toks[i]; const bool punct = t.kind == bp_token::k_punct;
        if (punct && t.text == "}") { brace = std::max(0, brace - 1); newline(); }
        else if (i > 0 && paren == 0) {
            bool brk = false;
            if (brace == 0 && t.kind == bp_token::k_ident && std::find(std::begin(top), std::end(top), t.text) != std::end(top)) {
                // a declaration keyword, unless it is the name another declaration is introducing (`input state : ...`)
                brk = !(toks[i - 1].kind == bp_token::k_ident && (std::find(std::begin(top), std::end(top), toks[i - 1].text) != std::end(top) || toks[i - 1].text == "blueprint"));
            } else if (brace > 0) brk = stmt_start(i);
            if (brk) newline();
        }
        if (line_start) out.append(std::size_t(brace) * 2, ' ');
        else {
            // spacing: tight around . and .., before , ) ] : and call / index brackets, after ( [ ; one space otherwise.
            // The exceptions keep two tokens from fusing when the text is lexed again.
            const auto& p = toks[i - 1]; const bool pp = p.kind == bp_token::k_punct;
            bool tight = false;
            if (punct && (t.text == "," || t.text == ")" || t.text == "]" || t.text == ":")) tight = true;
            if (punct && (t.text == "(" || t.text == "[") && (p.kind == bp_token::k_ident || (pp && (p.text == ")" || p.text == "]")))) tight = true;
            if (pp && (p.text == "(" || p.text == "[")) tight = true;
            if (punct && (t.text == "." || t.text == "..") && p.kind != bp_token::k_num && !(pp && (p.text == "." || p.text == ".."))) tight = true;
            if (pp && p.text == "." && t.kind == bp_token::k_ident) tight = true;
            if (pp && p.text == ".." && t.kind != bp_token::k_num && !(punct && (t.text == "." || t.text == ".."))) tight = true;
            if (punct && t.text == ".." && p.kind == bp_token::k_num && detail::bp_integral(p.num)) tight = true;      // 0..n (digits then "..": the lexer stops the number there)
            if (pp && p.text == ".." && t.kind == bp_token::k_num) tight = true;                                       // ..8
            if (pp && p.text == "/" && punct && t.text == "/") tight = false;                                          // "//" would start a comment
            if (!tight) out += ' ';
        }
        out += text(i); line_start = false;
        if (punct) {
            if (t.text == "(" || t.text == "[") ++paren;
            else if (t.text == ")" || t.text == "]") paren = std::max(0, paren - 1);
            else if (t.text == "{") { ++brace; newline(); }
            else if (t.text == "}") newline();
        }
    }
    newline();
    return out;
}
inline std::string canonical_blueprint_text(std::string_view text) { return canonical_blueprint_text(bp_lex(text)); }

// Binary form of a blueprint text (which must lex; it need not parse: templates with unbound parts encode too).
inline std::vector<std::uint8_t> encode_blueprint(std::string_view text) {
    auto toks = bp_lex(text);
    if (!toks.empty() && toks.back().kind == bp_token::k_end) toks.pop_back();
    std::vector<std::uint8_t> o;
    auto u8 = [&](std::uint8_t v) { o.push_back(v); };
    auto varint = [&](std::uint64_t v) { while (v >= 0x80) { o.push_back(std::uint8_t(v) | 0x80); v >>= 7; } o.push_back(std::uint8_t(v)); };
    auto u32 = [&](std::uint32_t v) { for (int i = 0; i < 4; ++i) o.push_back(std::uint8_t(v >> (8 * i))); };
    auto str = [&](std::string_view s) { varint(s.size()); o.insert(o.end(), s.begin(), s.end()); };
    auto h = detail::bp_header_of(toks);
    o.insert(o.end(), bp_bin_magic, bp_bin_magic + 8); u32(bp_bin_version); u32(h.opset); str(h.name); str(h.session);
    // string table in first-use order; identifiers and string literals are separate entries even when equal
    std::vector<std::pair<std::uint8_t, std::string>> tab; std::map<std::pair<bool, std::string>, std::size_t> at; std::vector<std::size_t> ref(toks.size(), 0);
    for (std::size_t i = 0; i < toks.size(); ++i) {
        const auto& t = toks[i];
        if (t.kind != bp_token::k_ident && t.kind != bp_token::k_str) continue;
        const bool lit = t.kind == bp_token::k_str;
        auto [it, fresh] = at.emplace(std::pair{lit, t.text}, tab.size());
        if (fresh) tab.emplace_back(std::uint8_t(lit ? 1 : 0), t.text);
        if (!lit && detail::bp_called(toks, i)) tab[it->second].first = 2;
        ref[i] = it->second;
    }
    varint(tab.size());
    for (auto& [k, s] : tab) { u8(k); str(s); }
    varint(toks.size());
    for (std::size_t i = 0; i < toks.size(); ++i) {
        const auto& t = toks[i];
        switch (t.kind) {
        case bp_token::k_ident: if (ref[i] < 224) u8(std::uint8_t(0x20 + ref[i])); else { u8(0x01); varint(ref[i]); } break;
        case bp_token::k_str: u8(0x02); varint(ref[i]); break;
        case bp_token::k_num:
            if (detail::bp_integral(t.num)) { u8(0x03); varint(static_cast<std::uint64_t>(t.num)); }
            else { u8(0x04); std::uint64_t b; std::memcpy(&b, &t.num, 8); for (int k = 0; k < 8; ++k) o.push_back(std::uint8_t(b >> (8 * k))); }
            break;
        case bp_token::k_punct: { int p = detail::bp_punct_index(t.text); if (p < 0) raise(errc::parse_error, "blueprint: token '{}' has no binary form", t.text); u8(std::uint8_t(0x10 + p)); break; }
        default: break;
        }
    }
    return o;
}

struct binary_blueprint_info { std::uint32_t version = 0, opset = 0; std::string name, session; std::size_t strings = 0, tokens = 0; std::vector<std::string> calls; };

// Tokens of a binary blueprint (untrusted bytes). Raises format_error (unsupported for another version) on anything
// that is not the canonical encoding of a token sequence.
inline std::vector<bp_token> decode_blueprint(std::span<const std::uint8_t> b, binary_blueprint_info* info = nullptr) {
    auto bad = [](const char* why) { raise(errc::format_error, "binary blueprint: {}", why); };
    if (!is_binary_blueprint(b)) bad("bad magic");
    std::size_t p = 8;
    auto need = [&](std::uint64_t k) { if (k > b.size() - p) bad("truncated"); };
    auto u8 = [&] { need(1); return b[p++]; };
    auto u32 = [&] { need(4); std::uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= std::uint32_t(b[p++]) << (8 * i); return v; };
    auto varint = [&] {
        std::uint64_t v = 0;
        for (int sh = 0; sh < 70; sh += 7) {
            auto c = u8();
            if (sh == 63 && c > 1) bad("varint overflows");
            v |= std::uint64_t(c & 0x7f) << sh;
            if (!(c & 0x80)) { if (sh && !c) bad("varint is not minimal"); return v; }
        }
        bad("varint too long"); return v;
    };
    auto str = [&] { auto n = varint(); need(n); std::string s(reinterpret_cast<const char*>(b.data() + p), std::size_t(n)); p += std::size_t(n); return s; };
    binary_blueprint_info h;
    h.version = u32();
    if (h.version != bp_bin_version) raise(errc::unsupported, "binary blueprint: version {} is not supported (this build reads version {})", h.version, bp_bin_version);
    h.opset = u32(); h.name = str(); h.session = str();
    auto ns = varint();
    if (ns > b.size() - p) bad("string table larger than the data");            // every entry takes at least 2 bytes
    std::vector<std::pair<std::uint8_t, std::string>> tab; tab.reserve(std::size_t(ns));
    std::set<std::pair<bool, std::string>> uniq;
    for (std::uint64_t i = 0; i < ns; ++i) {
        auto k = u8(); auto s = str();
        if (k > 2) bad("unknown string kind");
        if (k == 1 ? (s.find('"') != std::string::npos) : !detail::bp_is_ident(s)) bad(k == 1 ? "a string literal contains a quote" : "an identifier is not an identifier");
        if (!uniq.emplace(k == 1, s).second) bad("a string is listed twice");
        tab.emplace_back(k, std::move(s));
    }
    auto nt = varint();
    if (nt > b.size() - p) bad("more tokens than bytes");
    std::vector<bp_token> toks; toks.reserve(std::size_t(nt));
    std::size_t next_new = 0;   // strings must first appear in table order
    for (std::uint64_t i = 0; i < nt; ++i) {
        auto tag = u8(); bp_token t; t.line = 0;
        if (tag == 0x01 || tag == 0x02 || tag >= 0x20) {
            std::uint64_t ix = tag >= 0x20 ? std::uint64_t(tag - 0x20) : varint();
            if (tag == 0x01 && ix < 224) bad("an identifier index below 224 is not in its one-byte form");
            if (tag >= 0x20) tag = 0x01;
            if (ix >= tab.size()) bad("string index out of range");
            if ((tab[std::size_t(ix)].first == 1) != (tag == 0x02)) bad("a token uses a string of the other kind");
            if (ix > next_new) bad("the string table is not in first-use order");
            if (ix == next_new) ++next_new;
            t.kind = tag == 0x01 ? bp_token::k_ident : bp_token::k_str; t.text = tab[std::size_t(ix)].second;
        } else if (tag == 0x03) {
            auto v = varint();
            if (v >= (std::uint64_t(1) << 53)) bad("integer out of range");
            t.kind = bp_token::k_num; t.num = double(v); t.text = detail::bp_number_text(t.num);
        } else if (tag == 0x04) {
            need(8); std::uint64_t bits = 0; for (int k = 0; k < 8; ++k) bits |= std::uint64_t(b[p++]) << (8 * k);
            double v; std::memcpy(&v, &bits, 8);
            if (!std::isfinite(v) || !(v > 0) || detail::bp_integral(v)) bad("a float token holds a value that is not written as a float");
            t.kind = bp_token::k_num; t.num = v; t.text = detail::bp_number_text(v);
        } else if (tag >= 0x10 && tag < 0x10 + 15) { t.kind = bp_token::k_punct; t.text = std::string(detail::bp_puncts[tag - 0x10]); }
        else bad("unknown token tag");
        toks.push_back(std::move(t));
    }
    if (p != b.size()) bad("trailing bytes");
    if (next_new != tab.size()) bad("the string table has unused entries");
    // kinds and header must be what the tokens say
    {
        std::map<std::string, bool> called;
        for (std::size_t i = 0; i < toks.size(); ++i) if (toks[i].kind == bp_token::k_ident) { auto& c = called[toks[i].text]; c = c || detail::bp_called(toks, i); }
        for (auto& [k, s] : tab) if (k != 1 && (k == 2) != called[s]) bad("a string's call flag does not match its use");
        auto hh = detail::bp_header_of(toks);
        if (hh.name != h.name || hh.opset != h.opset || hh.session != h.session) bad("the header does not match the tokens");
    }
    if (info) {
        h.strings = tab.size(); h.tokens = toks.size();
        for (auto& [k, s] : tab) if (k == 2) h.calls.push_back(s);
        *info = std::move(h);
    }
    return toks;
}

// The text of a blueprint section or file: text is returned as it is, the binary form as its canonical text.
inline std::string blueprint_text(std::string_view section) {
    if (!is_binary_blueprint(section)) return std::string(section);
    return canonical_blueprint_text(decode_blueprint({reinterpret_cast<const std::uint8_t*>(section.data()), section.size()}));
}
// Parses a blueprint in either form.
inline blueprint load_blueprint(std::string_view section) { return parse_blueprint(blueprint_text(section)); }
inline blueprint load_blueprint(const reader& r) { return load_blueprint(r.section_text("blueprint")); }

// Rewrites the model file `in` to `out` (may be the same path) with its blueprint section in the binary form, or back
// in text (the canonical text when it was binary; unchanged when it already was text). The blueprint must parse.
// The manifest gets "blueprint_form" and, for the binary form, "forge_format_minor": 2. A signature and precompiled
// plans are dropped (both cover the blueprint bytes). Returns false when the section already had the form asked for.
inline bool convert_blueprint_section(const std::string& in, const std::string& out, bool to_binary) {
    std::vector<std::pair<std::string, std::string>> secs; detail::data_layout dl;
    {
        reader r(in);
        auto cur = r.section_text("blueprint");
        if (is_binary_blueprint(cur) == to_binary) return false;
        std::string text = blueprint_text(cur);
        (void)parse_blueprint(text);
        std::string next;
        if (to_binary) { auto b = encode_blueprint(text); next.assign(reinterpret_cast<const char*>(b.data()), b.size()); }
        else next = std::move(text);
        dl = r.layout();
        for (auto& [name, s] : r.sections()) {
            if (name == "signature" || name == "aot") continue;
            auto b = r.raw_section_bytes(name);
            std::string bytes(reinterpret_cast<const char*>(b.data()), b.size());
            if (name == "blueprint") bytes = next;
            else if (name == "manifest") {
                try {
                    auto m = json::parse(bytes);
                    if (m.is_obj()) {
                        auto& o = m.obj_mut();
                        auto set = [&](const char* k, json::value v) { for (auto& [kk, vv] : o) if (kk == k) { vv = std::move(v); return; } o.emplace_back(k, std::move(v)); };
                        set("blueprint_form", to_binary ? "binary" : "text");
                        if (to_binary && m.num_or("forge_format_minor", 0) < 2) set("forge_format_minor", 2.0);
                        bytes = json::dump(m);
                    }
                } catch (const forge_exception&) {}
            }
            secs.emplace_back(name, std::move(bytes));
        }
    }
    detail::write_sections(out, std::move(secs), dl);
    return true;
}

} // namespace forge::modelfile
