#pragma once
// Chat templates: a renderer for the Jinja2 subset that Hugging Face tokenizer_config.json / GGUF
// tokenizer.chat_template templates are written in (ChatML / Qwen, Llama-2, Llama-3, Mistral, Gemma, Phi-3, DeepSeek,
// tool-calling variants). The template text is the model's own, stored in the .fpk "chat_template" section.
//
// Supported: {{ expr }}, {% if / elif / else %}, {% for x in y if cond %} ... {% else %} (loop.index, index0, first, last,
// length, revindex, previtem, nextitem, cycle), {% set x = .. %}, {% set ns.x = .. %}, {% set x %}..{% endset %},
// {% macro %}, {% break %}, {% continue %}, {% raw %}, {# comments #}, whitespace control ({%- -%} {{- -}}) and the
// trim_blocks / lstrip_blocks settings transformers renders with. Expressions: literals, lists, dicts, . and [] access,
// slices, calls with keyword arguments, + - * / // % ** ~, comparisons, in / not in, and / or / not, a if c else b,
// filters (trim length lower upper title capitalize tojson default join first last list string int float replace safe
// items selectattr rejectattr select reject map unique sort reverse indent abs round min max sum dictsort escape),
// tests (defined undefined none string mapping iterable sequence number integer float boolean true false callable
// eq ne lt le gt ge in odd even divisibleby), string / list / dict methods (strip split startswith endswith items get
// keys values append pop ...), namespace(), range(), dict(), raise_exception(), strftime_now().
// tojson is transformers' (json.dumps(x, ensure_ascii=False, indent=..)), not Jinja's HTML-safe one.
//
// A template is untrusted input (it comes out of a model file): nesting depth, the number of evaluation steps, the
// bytes built and the output size are bounded (render_limits); breaking a bound raises, it never crashes or hangs.
// Every error is a forge_exception.
#include <forge/json/json.hpp>
#include <forge/os/text.hpp>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace forge::inference {

struct render_limits {
    std::size_t max_output = 8u << 20;       // bytes of rendered text
    std::uint64_t max_steps = 20'000'000;    // evaluation steps + bytes / elements built along the way
    int max_depth = 150;                     // nesting of expressions, blocks and macro calls
    std::size_t max_template = 1u << 20;     // bytes of template source
};

namespace jinja {
namespace tx = forge::os::text;

[[noreturn]] inline void fail(const std::string& what) { raise(errc::invalid_argument, "chat template: {}", what); }

struct value;
using list_t = std::vector<value>;
using dict_t = std::vector<std::pair<std::string, value>>;
struct macro_t;
struct value {
    enum kind : std::uint8_t { undef, none, boolean, integer, real, string, list, dict, macro } k = undef;
    bool b = false, ns = false;   // ns: a namespace() object (attribute assignment allowed)
    bool tup = false;             // a list that prints as a Python tuple: (1, 2)
    std::uint16_t depth = 0;      // containers: 1 + the deepest element. Bounded, so no value is too deep to print or destroy.
    std::int64_t i = 0; double d = 0; std::string s;
    std::shared_ptr<list_t> l; std::shared_ptr<dict_t> m; std::shared_ptr<const macro_t> f;
    value() = default;
    static value of_none() { value v; v.k = none; return v; }
    static value of(bool x) { value v; v.k = boolean; v.b = x; return v; }
    static value of(std::int64_t x) { value v; v.k = integer; v.i = x; return v; }
    static value of(double x) { value v; v.k = real; v.d = x; return v; }
    static value of(std::string x) { value v; v.k = string; v.s = std::move(x); return v; }
    static constexpr int max_nesting = 100;
    static std::uint16_t nest(int deepest) { if (deepest >= max_nesting) fail("data nested too deep"); return std::uint16_t(deepest + 1); }
    static value of(list_t x) { value v; v.k = list; int d = 0; for (auto& e : x) d = std::max(d, int(e.depth)); v.depth = nest(d); v.l = std::make_shared<list_t>(std::move(x)); return v; }
    static value of(dict_t x) { value v; v.k = dict; int d = 0; for (auto& e : x) d = std::max(d, int(e.second.depth)); v.depth = nest(d); v.m = std::make_shared<dict_t>(std::move(x)); return v; }
    bool is_num() const { return k == integer || k == real || k == boolean; }
    double num() const { return k == integer ? double(i) : k == real ? d : k == boolean ? double(b) : 0; }
    bool truthy() const {
        switch (k) {
        case undef: case none: return false;
        case boolean: return b;
        case integer: return i != 0;
        case real: return d != 0;
        case string: return !s.empty();
        case list: return !l->empty();
        case dict: return !m->empty();
        default: return true;
        }
    }
    const value* get(std::string_view key) const { if (k == dict) for (auto& [kk, v] : *m) if (kk == key) return &v; return nullptr; }
    const char* type_name() const {
        static const char* const n[] = {"undefined", "none", "bool", "int", "float", "str", "list", "dict", "macro"};
        return n[k];
    }
};

inline value from_json(const json::value& j, int depth = 0) {
    if (depth > value::max_nesting) fail("input nested too deep");
    if (j.is_null()) return value::of_none();
    if (j.is_bool()) return value::of(j.b());
    if (j.is_num()) { double d = j.num(); return d == std::floor(d) && std::fabs(d) < 9e15 ? value::of(std::int64_t(d)) : value::of(d); }
    if (j.is_str()) return value::of(j.str());
    if (j.is_arr()) { list_t l; l.reserve(j.arr().size()); for (auto& x : j.arr()) l.push_back(from_json(x, depth + 1)); return value::of(std::move(l)); }
    dict_t d; for (auto& [k, v] : j.obj()) d.emplace_back(k, from_json(v, depth + 1));
    return value::of(std::move(d));
}

inline std::string float_str(double d) {
    if (std::isnan(d)) return "nan";
    if (std::isinf(d)) return d < 0 ? "-inf" : "inf";
    auto s = std::format("{}", d);
    if (s.find_first_of(".en") == std::string::npos) s += ".0";
    return s;
}
// Python's repr() of a string
inline void py_repr(std::string& o, const std::string& s) {
    char q = s.find('\'') != std::string::npos && s.find('"') == std::string::npos ? '"' : '\'';
    o += q;
    for (unsigned char c : s) {
        if (c == q || c == '\\') { o += '\\'; o += char(c); }
        else if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r"; else if (c == '\t') o += "\\t";
        else if (c < 0x20 || c == 0x7F) o += std::format("\\x{:02x}", c);
        else o += char(c);
    }
    o += q;
}
// str(x) at the top level; repr(x) inside containers
inline void to_text(std::string& o, const value& v, bool repr = false, int depth = 0) {
    if (depth > 64) fail("value nested too deep to print");
    switch (v.k) {
    case value::undef: break;
    case value::none: o += "None"; break;
    case value::boolean: o += v.b ? "True" : "False"; break;
    case value::integer: o += std::to_string(v.i); break;
    case value::real: o += float_str(v.d); break;
    case value::string: if (repr) py_repr(o, v.s); else o += v.s; break;
    case value::list: {
        o += v.tup ? '(' : '['; bool f = true;
        for (auto& x : *v.l) { if (!f) o += ", "; f = false; to_text(o, x, true, depth + 1); }
        if (v.tup && v.l->size() == 1) o += ',';
        o += v.tup ? ')' : ']'; break; }
    case value::dict: {
        if (v.ns) { o += "<Namespace>"; break; }
        o += '{'; bool f = true;
        for (auto& [k, x] : *v.m) { if (!f) o += ", "; f = false; py_repr(o, k); o += ": "; to_text(o, x, true, depth + 1); }
        o += '}'; break; }
    case value::macro: o += "<macro>"; break;
    }
}
inline std::string text_of(const value& v) { std::string o; to_text(o, v); return o; }
// json.dumps(v, ensure_ascii=False, indent=indent)
inline void to_json(std::string& o, const value& v, int indent, int level = 0) {
    if (level > 64) fail("value nested too deep for tojson");
    auto nl = [&](int lv) { if (indent >= 0) { o += '\n'; o.append(std::size_t(indent) * std::size_t(lv), ' '); } };
    auto str = [&](const std::string& s) {
        o += '"';
        for (unsigned char c : s) {
            if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r";
            else if (c == '\t') o += "\\t"; else if (c == '\b') o += "\\b"; else if (c == '\f') o += "\\f";
            else if (c < 0x20) o += std::format("\\u{:04x}", c);
            else o += char(c);
        }
        o += '"';
    };
    switch (v.k) {
    case value::undef: fail("tojson: undefined value");
    case value::none: o += "null"; break;
    case value::boolean: o += v.b ? "true" : "false"; break;
    case value::integer: o += std::to_string(v.i); break;
    case value::real: o += std::isnan(v.d) ? "NaN" : std::isinf(v.d) ? (v.d < 0 ? "-Infinity" : "Infinity") : float_str(v.d); break;
    case value::string: str(v.s); break;
    case value::list:
        if (v.l->empty()) { o += "[]"; break; }
        o += '[';
        for (std::size_t i = 0; i < v.l->size(); ++i) { if (i) o += indent >= 0 ? "," : ", "; nl(level + 1); to_json(o, (*v.l)[i], indent, level + 1); }
        nl(level); o += ']'; break;
    case value::dict:
        if (v.m->empty()) { o += "{}"; break; }
        o += '{';
        for (std::size_t i = 0; i < v.m->size(); ++i) {
            if (i) o += indent >= 0 ? "," : ", ";
            nl(level + 1); str((*v.m)[i].first); o += ": "; to_json(o, (*v.m)[i].second, indent, level + 1);
        }
        nl(level); o += '}'; break;
    case value::macro: fail("tojson: a macro is not serializable");
    }
}

// ------------------------------------------------------------------------------------------------ lexer
struct token {
    enum kind { data, var_begin, var_end, block_begin, block_end, name, integer, real, string, op, eof } k = eof;
    std::string s; std::int64_t i = 0; double d = 0; std::size_t pos = 0;
};

inline std::vector<token> lex(std::string_view src) {
    std::vector<token> out;
    std::size_t p = 0; bool line_start = true;
    auto is_ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };
    auto err = [&](const char* what, std::size_t at) { fail(std::format("{} at offset {}", what, at)); };
    // the text before a tag, with the tag's whitespace control applied
    auto emit_data = [&](std::string_view text, char sign, bool block) {
        std::size_t n = text.size();
        if (sign == '-') { while (n && is_ws(text[n - 1])) --n; }
        else if (sign != '+' && block) {   // lstrip_blocks: indentation in front of a block tag goes
            auto nlp = text.rfind('\n'); std::size_t l = nlp == std::string_view::npos ? 0 : nlp + 1;
            if (l > 0 || line_start) { bool blank = true; for (std::size_t i = l; i < n; ++i) blank &= text[i] == ' ' || text[i] == '\t'; if (blank) n = l; }
        }
        if (n) { token t; t.k = token::data; t.s = std::string(text.substr(0, n)); out.push_back(std::move(t)); }
    };
    // after a closing delimiter: "-" eats all whitespace, a block tag eats one newline (trim_blocks)
    auto after_tag = [&](char sign, bool block) {
        line_start = false;
        if (sign == '-') { while (p < src.size() && is_ws(src[p])) { line_start = src[p] == '\n'; ++p; } }
        else if (sign != '+' && block && p < src.size() && src[p] == '\n') { ++p; line_start = true; }
        else if (sign != '+' && block && p + 1 < src.size() && src[p] == '\r' && src[p + 1] == '\n') { p += 2; line_start = true; }
    };
    while (p < src.size()) {
        std::size_t open = src.find('{', p);
        while (open != std::string_view::npos && !(open + 1 < src.size() && (src[open + 1] == '{' || src[open + 1] == '%' || src[open + 1] == '#'))) open = src.find('{', open + 1);
        if (open == std::string_view::npos) { emit_data(src.substr(p), 0, false); break; }
        char kind = src[open + 1];
        std::size_t q = open + 2; char lsign = 0;
        if (q < src.size() && (src[q] == '-' || (src[q] == '+' && kind != '{'))) lsign = src[q++];
        std::string_view text = src.substr(p, open - p);
        if (!text.empty() && text.back() == '\n' && lsign != '-') {}   // (a text ending in a newline leaves the tag at a line start)
        emit_data(text, lsign, kind != '{');
        if (!text.empty()) line_start = text.back() == '\n';
        if (kind == '#') {
            auto e = src.find("#}", q);
            if (e == std::string_view::npos) err("unterminated comment", open);
            char rsign = e > q && (src[e - 1] == '-' || src[e - 1] == '+') ? src[e - 1] : 0;
            p = e + 2; after_tag(rsign, true);
            continue;
        }
        token b; b.k = kind == '{' ? token::var_begin : token::block_begin; b.pos = open; out.push_back(b);
        std::size_t first = out.size();
        const char close = kind == '{' ? '}' : '%';
        int depth = 0; char rsign = 0; bool closed = false;
        p = q;
        while (p < src.size()) {
            char c = src[p];
            if (is_ws(c)) { ++p; continue; }
            // end of tag (only outside brackets)
            if (depth == 0) {
                std::size_t e = p; char sg = 0;
                if ((c == '-' || c == '+') && e + 1 < src.size()) { sg = c; ++e; }
                if (e + 1 < src.size() && src[e] == close && src[e + 1] == '}' && (sg != '+' || kind == '%')) { rsign = sg; p = e + 2; closed = true; break; }
            }
            token t; t.pos = p;
            if (std::isalpha((unsigned char)c) || c == '_') {
                std::size_t e = p; while (e < src.size() && (std::isalnum((unsigned char)src[e]) || src[e] == '_')) ++e;
                t.k = token::name; t.s = std::string(src.substr(p, e - p)); p = e;
            } else if (std::isdigit((unsigned char)c)) {
                std::size_t e = p; while (e < src.size() && (std::isdigit((unsigned char)src[e]) || src[e] == '_')) ++e;
                bool fl = false;
                if (e + 1 < src.size() && src[e] == '.' && std::isdigit((unsigned char)src[e + 1])) { fl = true; ++e; while (e < src.size() && std::isdigit((unsigned char)src[e])) ++e; }
                if (e < src.size() && (src[e] == 'e' || src[e] == 'E')) {
                    std::size_t x = e + 1; if (x < src.size() && (src[x] == '+' || src[x] == '-')) ++x;
                    if (x < src.size() && std::isdigit((unsigned char)src[x])) { fl = true; e = x; while (e < src.size() && std::isdigit((unsigned char)src[e])) ++e; }
                }
                std::string num; for (char ch : src.substr(p, e - p)) if (ch != '_') num += ch;
                if (fl) { t.k = token::real; t.d = std::strtod(num.c_str(), nullptr); }
                else { if (num.size() > 18) err("integer literal too large", p); t.k = token::integer; t.i = std::strtoll(num.c_str(), nullptr, 10); }
                p = e;
            } else if (c == '"' || c == '\'') {
                std::size_t e = p + 1; std::string s;
                for (;; ++e) {
                    if (e >= src.size()) err("unterminated string", p);
                    char ch = src[e];
                    if (ch == c) break;
                    if (ch != '\\') { s += ch; continue; }
                    if (++e >= src.size()) err("unterminated string", p);
                    switch (src[e]) {
                    case 'n': s += '\n'; break; case 't': s += '\t'; break; case 'r': s += '\r'; break; case 'b': s += '\b'; break; case 'f': s += '\f'; break;
                    case 'v': s += '\v'; break; case 'a': s += '\a'; break; case '0': s += '\0'; break;
                    case '\\': s += '\\'; break; case '\'': s += '\''; break; case '"': s += '"'; break; case '\n': break;
                    case 'x': case 'u': case 'U': {
                        std::size_t n = src[e] == 'x' ? 2 : src[e] == 'u' ? 4 : 8; unsigned cp = 0;
                        if (e + n >= src.size()) err("bad escape in string", e);
                        for (std::size_t k = 1; k <= n; ++k) { char h = src[e + k]; if (!std::isxdigit((unsigned char)h)) err("bad escape in string", e); cp = cp * 16 + unsigned(std::isdigit((unsigned char)h) ? h - '0' : (h | 0x20) - 'a' + 10); }
                        if (cp > 0x10FFFF || (cp >= 0xD800 && cp < 0xE000)) cp = 0xFFFD;
                        tx::append_utf8(s, char32_t(cp)); e += n; break; }
                    default: s += '\\'; s += src[e]; break;   // Python keeps unknown escapes
                    }
                }
                t.k = token::string; t.s = std::move(s); p = e + 1;
            } else {
                static const char* const ops[] = {"**", "//", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "~", "<", ">", "=", ".", ",", ":", "|", "(", ")", "[", "]", "{", "}"};
                const char* hit = nullptr;
                for (auto o : ops) if (src.compare(p, std::strlen(o), o) == 0) { hit = o; break; }
                if (!hit) err("unexpected character", p);
                t.k = token::op; t.s = hit; p += t.s.size();
                if (t.s == "(" || t.s == "[" || t.s == "{") ++depth;
                else if (t.s == ")" || t.s == "]" || t.s == "}") --depth;
                if (depth < 0) err("unbalanced bracket", t.pos);
            }
            out.push_back(std::move(t));
            if (out.size() > 4'000'000) err("template too large", p);
        }
        if (!closed) err(kind == '{' ? "unterminated {{" : "unterminated {%", open);
        token e; e.k = kind == '{' ? token::var_end : token::block_end; e.pos = p; out.push_back(e);
        // {% raw %} ... {% endraw %}: everything in between is text
        if (kind == '%' && out.size() == first + 2 && out[first].k == token::name && out[first].s == "raw") {
            out.resize(first - 1);
            after_tag(rsign, true);
            std::size_t e2 = p, close_at = std::string_view::npos, close_end = 0; char ls = 0, rs = 0;
            while ((e2 = src.find("{%", e2)) != std::string_view::npos) {
                std::size_t x = e2 + 2; char l2 = 0;
                if (x < src.size() && (src[x] == '-' || src[x] == '+')) l2 = src[x++];
                while (x < src.size() && is_ws(src[x])) ++x;
                if (src.compare(x, 6, "endraw") == 0) {
                    x += 6; while (x < src.size() && is_ws(src[x])) ++x;
                    char r2 = 0; if (x < src.size() && (src[x] == '-' || src[x] == '+')) r2 = src[x++];
                    if (src.compare(x, 2, "%}") == 0) { close_at = e2; close_end = x + 2; ls = l2; rs = r2; break; }
                }
                e2 += 2;
            }
            if (close_at == std::string_view::npos) err("unterminated {% raw %}", open);
            emit_data(src.substr(p, close_at - p), ls == '-' ? '-' : '+', false);
            p = close_end; after_tag(rs, true);
            continue;
        }
        after_tag(rsign, kind == '%');
    }
    token e; e.k = token::eof; e.pos = src.size(); out.push_back(e);
    return out;
}

// ------------------------------------------------------------------------------------------------ syntax tree
struct expr;
using expr_p = std::unique_ptr<expr>;
struct expr {
    enum kind { lit, name, attr, index, slice, call, filter, test, unary, binary, logic_and, logic_or, logic_not, cond, list, dict, tuple, concat } k = lit;
    value v; std::string s;                     // literal; name / attribute / filter / test / operator
    std::vector<expr_p> a;                      // operands / elements / call arguments (callee first)
    std::vector<std::pair<std::string, expr_p>> kw;   // keyword arguments; dict items use `a` as key, value pairs
    bool neg = false;                           // "is not"
};
struct node;
using body_t = std::vector<node>;
struct node {
    enum kind { text, output, if_, for_, set, set_block, macro, brk, cont } k = text;
    std::string s;                              // text; set target / macro name
    std::string attr;                           // set ns.attr
    std::vector<std::string> names;             // loop variables / macro parameters
    expr_p e, cond;                             // output / set value / iterable; loop filter
    std::vector<std::pair<expr_p, body_t>> branches;   // if / elif (condition null = else)
    body_t body, alt;                           // loop body / else; macro body; set block
    std::vector<expr_p> defaults;               // macro parameter defaults (null = required)
};
struct macro_t { std::string name; std::vector<std::string> params; std::vector<expr_p> defaults; body_t body; };

class parser {
public:
    parser(std::vector<token> t, int max_depth) : t_(std::move(t)), max_depth_(max_depth) {}
    body_t parse() { std::string end; auto b = body({}, end); if (cur().k != token::eof) fail("unexpected tag"); return b; }
private:
    std::vector<token> t_; std::size_t i_ = 0; int depth_ = 0, max_depth_;
    int nodes_ = 0;   // expression nodes in the current tag: chains (a + b + c ...) deepen the tree without recursing here
    struct deeper { parser& p; explicit deeper(parser& x) : p(x) { if (++p.depth_ > p.max_depth_) fail("template nested too deep"); } ~deeper() { --p.depth_; } };
    const token& cur() const { return t_[i_]; }
    [[noreturn]] void bad(const std::string& what) const { fail(std::format("{} at offset {}", what, cur().pos)); }
    bool is_op(const char* o) const { return cur().k == token::op && cur().s == o; }
    bool is_name(const char* n) const { return cur().k == token::name && cur().s == n; }
    bool eat_op(const char* o) { if (is_op(o)) { ++i_; return true; } return false; }
    bool eat_name(const char* n) { if (is_name(n)) { ++i_; return true; } return false; }
    void need_op(const char* o) { if (!eat_op(o)) bad(std::string("expected '") + o + "'"); }
    std::string ident() { if (cur().k != token::name) bad("expected a name"); return t_[i_++].s; }
    void end_block() { if (cur().k != token::block_end) bad("expected %}"); ++i_; }

    // statements up to one of the `ends` keywords, which is consumed (the caller reads the rest of its tag)
    body_t body(std::initializer_list<const char*> ends, std::string& ended) {
        deeper g(*this);
        body_t out;
        for (;;) {
            auto& tk = cur();
            if (tk.k == token::eof) { if (ends.size()) bad("unexpected end of template (missing end tag)"); return out; }
            if (tk.k == token::data) { node n; n.k = node::text; n.s = tk.s; out.push_back(std::move(n)); ++i_; continue; }
            nodes_ = 0;
            if (tk.k == token::var_begin) {
                ++i_; node n; n.k = node::output; n.e = expression();
                if (cur().k != token::var_end) bad("expected }}");
                ++i_; out.push_back(std::move(n)); continue;
            }
            if (tk.k != token::block_begin) bad("unexpected token");
            ++i_;
            auto kw = ident();
            for (auto e : ends) if (kw == e) { ended = kw; return out; }
            out.push_back(statement(kw));
        }
    }
    node statement(const std::string& kw) {
        node n; std::string ended;
        if (kw == "if") {
            n.k = node::if_;
            auto c = expression(); end_block();
            for (;;) {
                auto b = body({"elif", "else", "endif"}, ended);
                n.branches.emplace_back(std::move(c), std::move(b));
                if (ended == "endif") { end_block(); break; }
                if (ended == "elif") { c = expression(); end_block(); continue; }
                end_block();
                auto e = body({"endif"}, ended); end_block();
                n.branches.emplace_back(nullptr, std::move(e)); break;
            }
        } else if (kw == "for") {
            n.k = node::for_;
            do { n.names.push_back(ident()); } while (eat_op(","));
            if (!eat_name("in")) bad("expected 'in'");
            n.e = or_expr();
            if (eat_name("if")) n.cond = expression();
            eat_name("recursive");
            end_block();
            n.body = body({"else", "endfor"}, ended);
            if (ended == "else") { end_block(); n.alt = body({"endfor"}, ended); }
            end_block();
        } else if (kw == "set") {
            n.s = ident();
            if (eat_op(".")) n.attr = ident();
            else while (eat_op(",")) { if (n.names.empty()) n.names.push_back(n.s); n.names.push_back(ident()); }   // set a, b = 1, 2
            if (eat_op("=")) { n.k = node::set; n.e = tuple_or_expr(); end_block(); }
            else { n.k = node::set_block; end_block(); n.body = body({"endset"}, ended); end_block(); }
        } else if (kw == "macro") {
            n.k = node::macro; n.s = ident(); need_op("(");
            while (!is_op(")")) {
                n.names.push_back(ident());
                n.defaults.push_back(eat_op("=") ? expression() : nullptr);
                if (!eat_op(",")) break;
            }
            need_op(")"); end_block();
            n.body = body({"endmacro"}, ended); end_block();
        } else if (kw == "break") { n.k = node::brk; end_block(); }
        else if (kw == "continue") { n.k = node::cont; end_block(); }
        else if (kw == "generation") {   // transformers' assistant-tracking tag: only its body matters here
            end_block(); n.k = node::if_; n.branches.emplace_back(nullptr, body({"endgeneration"}, ended)); end_block();
        } else bad("unsupported tag '" + kw + "'");
        return n;
    }

    expr_p make(expr::kind k, std::string s = {}) {
        if (++nodes_ > 3000) bad("expression too large");
        auto e = std::make_unique<expr>(); e->k = k; e->s = std::move(s); return e;
    }
    expr_p bin(expr::kind k, std::string op, expr_p l, expr_p r) { auto e = make(k, std::move(op)); e->a.push_back(std::move(l)); e->a.push_back(std::move(r)); return e; }
    expr_p tuple_or_expr() {
        auto e = expression();
        if (!is_op(",")) return e;
        auto t = make(expr::tuple); t->a.push_back(std::move(e));
        while (eat_op(",")) { if (cur().k == token::block_end || cur().k == token::var_end) break; t->a.push_back(expression()); }
        return t;
    }
    expr_p expression() {   // a if c else b
        deeper g(*this);
        auto e = or_expr();
        while (eat_name("if")) {
            auto c = make(expr::cond); auto test = or_expr();
            c->a.push_back(std::move(test)); c->a.push_back(std::move(e));
            if (eat_name("else")) c->a.push_back(expression());
            e = std::move(c);
        }
        return e;
    }
    expr_p or_expr() { auto e = and_expr(); while (eat_name("or")) e = bin(expr::logic_or, "or", std::move(e), and_expr()); return e; }
    expr_p and_expr() { auto e = not_expr(); while (eat_name("and")) e = bin(expr::logic_and, "and", std::move(e), not_expr()); return e; }
    expr_p not_expr() {
        if (eat_name("not")) { deeper g(*this); auto e = make(expr::logic_not); e->a.push_back(not_expr()); return e; }
        return compare();
    }
    expr_p compare() {
        auto e = math1();
        for (;;) {
            std::string op;
            if (cur().k == token::op && (cur().s == "==" || cur().s == "!=" || cur().s == "<" || cur().s == ">" || cur().s == "<=" || cur().s == ">=")) op = t_[i_++].s;
            else if (is_name("in")) { ++i_; op = "in"; }
            else if (is_name("not") && t_[i_ + 1].k == token::name && t_[i_ + 1].s == "in") { i_ += 2; op = "not in"; }
            else return e;
            e = bin(expr::binary, op, std::move(e), math1());
        }
    }
    expr_p math1() {
        auto e = concat();
        while (is_op("+") || is_op("-")) { auto op = t_[i_++].s; e = bin(expr::binary, op, std::move(e), concat()); }
        return e;
    }
    expr_p concat() {
        auto e = math2();
        while (eat_op("~")) e = bin(expr::concat, "~", std::move(e), math2());
        return e;
    }
    expr_p math2() {
        auto e = power();
        while (is_op("*") || is_op("/") || is_op("//") || is_op("%")) { auto op = t_[i_++].s; e = bin(expr::binary, op, std::move(e), power()); }
        return e;
    }
    expr_p power() {
        auto e = unary();
        while (eat_op("**")) e = bin(expr::binary, "**", std::move(e), unary());
        return e;
    }
    expr_p unary(bool with_filter = true) {   // filters bind to the signed operand as a whole: -3|abs is 3
        deeper g(*this);
        expr_p e;
        if (is_op("-") || is_op("+")) { e = make(expr::unary, t_[i_++].s); e->a.push_back(unary(false)); }
        else e = postfix(primary());
        if (with_filter) return postfix_filters(std::move(e));
        return e;
    }
    expr_p primary() {
        deeper g(*this);
        auto& tk = cur();
        if (tk.k == token::integer) { auto e = make(expr::lit); e->v = value::of(tk.i); ++i_; return e; }
        if (tk.k == token::real) { auto e = make(expr::lit); e->v = value::of(tk.d); ++i_; return e; }
        if (tk.k == token::string) {
            auto e = make(expr::lit); std::string s;
            while (cur().k == token::string) s += t_[i_++].s;   // adjacent literals concatenate
            e->v = value::of(std::move(s)); return e;
        }
        if (tk.k == token::name) {
            auto n = t_[i_++].s; auto e = make(expr::lit);
            if (n == "true" || n == "True") e->v = value::of(true);
            else if (n == "false" || n == "False") e->v = value::of(false);
            else if (n == "none" || n == "None") e->v = value::of_none();
            else { e->k = expr::name; e->s = n; }
            return e;
        }
        if (eat_op("(")) {
            if (eat_op(")")) return make(expr::tuple);
            auto e = expression();
            if (is_op(",")) {
                auto t = make(expr::tuple); t->a.push_back(std::move(e));
                while (eat_op(",")) { if (is_op(")")) break; t->a.push_back(expression()); }
                e = std::move(t);
            }
            need_op(")"); return e;
        }
        if (eat_op("[")) {
            auto e = make(expr::list);
            while (!is_op("]")) { e->a.push_back(expression()); if (!eat_op(",")) break; }
            need_op("]"); return e;
        }
        if (eat_op("{")) {
            auto e = make(expr::dict);
            while (!is_op("}")) { e->a.push_back(expression()); need_op(":"); e->a.push_back(expression()); if (!eat_op(",")) break; }
            need_op("}"); return e;
        }
        bad("unexpected token in expression");
    }
    void call_args(expr& e) {
        while (!is_op(")")) {
            if (cur().k == token::name && t_[i_ + 1].k == token::op && t_[i_ + 1].s == "=") { auto n = t_[i_].s; i_ += 2; e.kw.emplace_back(n, expression()); }
            else e.a.push_back(expression());
            if (!eat_op(",")) break;
        }
        need_op(")");
    }
    expr_p postfix(expr_p e) {
        for (;;) {
            if (eat_op(".")) {
                auto a = make(expr::attr);
                if (cur().k == token::integer) { a->k = expr::index; auto ix = make(expr::lit); ix->v = value::of(t_[i_++].i); a->a.push_back(std::move(e)); a->a.push_back(std::move(ix)); }
                else { a->s = ident(); a->a.push_back(std::move(e)); }
                e = std::move(a);
            } else if (eat_op("[")) {
                expr_p lo, hi, step; bool sl = false;
                if (!is_op(":")) lo = expression();
                if (eat_op(":")) {
                    sl = true;
                    if (!is_op("]") && !is_op(":")) hi = expression();
                    if (eat_op(":") && !is_op("]")) step = expression();
                }
                need_op("]");
                auto x = make(sl ? expr::slice : expr::index);
                x->a.push_back(std::move(e));
                if (sl) { x->a.push_back(std::move(lo)); x->a.push_back(std::move(hi)); x->a.push_back(std::move(step)); }
                else x->a.push_back(std::move(lo));
                e = std::move(x);
            } else if (eat_op("(")) {
                auto c = make(expr::call); c->a.push_back(std::move(e)); call_args(*c); e = std::move(c);
            } else return e;
        }
    }
    expr_p postfix_filters(expr_p e) {
        for (;;) {
            if (eat_op("|")) {
                auto f = make(expr::filter, ident());
                f->a.push_back(std::move(e));
                if (eat_op("(")) call_args(*f);
                e = std::move(f);
            } else if (is_name("is")) {
                ++i_;
                auto t = make(expr::test);
                t->neg = eat_name("not");
                t->s = ident();
                t->a.push_back(std::move(e));
                if (eat_op("(")) call_args(*t);
                else if ((cur().k == token::name && !is_name("else") && !is_name("or") && !is_name("and") && !is_name("if") && !is_name("in") && !is_name("is") && !is_name("not"))
                         || cur().k == token::string || cur().k == token::integer || cur().k == token::real || is_op("[") || is_op("{"))
                    t->a.push_back(postfix(primary()));
                e = std::move(t);
            } else if (is_op("(") || is_op("[") || is_op(".")) e = postfix(std::move(e));
            else return e;
        }
    }
};

// ------------------------------------------------------------------------------------------------ evaluation
class interp {
public:
    interp(const render_limits& lim, std::time_t now) : lim_(lim), now_(now) { frames_.emplace_back(); }
    void define(std::string name, value v) { frames_[0].emplace_back(std::move(name), std::move(v)); }
    std::string run(const body_t& b) { std::string out; exec(b, out); return out; }

private:
    using frame = std::vector<std::pair<std::string, value>>;
    enum class flow { next, brk, cont };
    const render_limits& lim_; std::time_t now_;
    std::vector<frame> frames_; std::uint64_t steps_ = 0; int depth_ = 0;
    struct deeper { interp& p; explicit deeper(interp& x) : p(x) { if (++p.depth_ > p.lim_.max_depth * 4) fail("evaluation nested too deep"); } ~deeper() { --p.depth_; } };

    void tick(std::uint64_t n = 1) { if ((steps_ += n) > lim_.max_steps) fail("the template takes too many steps (render_limits::max_steps)"); }
    void emit(std::string& out, std::string_view s) { if (out.size() + s.size() > lim_.max_output) fail("output too large (render_limits::max_output)"); tick(1 + s.size() / 16); out += s; }
    const value* lookup(std::string_view n) const {
        for (auto f = frames_.rbegin(); f != frames_.rend(); ++f) for (auto e = f->rbegin(); e != f->rend(); ++e) if (e->first == n) return &e->second;
        return nullptr;
    }
    void assign(const std::string& n, value v) { for (auto& e : frames_.back()) if (e.first == n) { e.second = std::move(v); return; } frames_.back().emplace_back(n, std::move(v)); }

    flow exec(const body_t& b, std::string& out) {
        deeper g(*this);
        for (auto& n : b) {
            tick();
            switch (n.k) {
            case node::text: emit(out, n.s); break;
            case node::output: emit(out, text_of(eval(*n.e))); break;
            case node::if_:
                for (auto& [c, body] : n.branches)
                    if (!c || eval(*c).truthy()) { if (auto f = exec(body, out); f != flow::next) return f; break; }
                break;
            case node::for_: run_for(n, out); break;
            case node::set: {
                auto v = eval(*n.e);
                if (!n.names.empty()) {
                    if (v.k != value::list || v.l->size() != n.names.size()) fail("cannot unpack the value into " + std::to_string(n.names.size()) + " names");
                    for (std::size_t i = 0; i < n.names.size(); ++i) assign(n.names[i], (*v.l)[i]);
                    break;
                }
                if (n.attr.empty()) { assign(n.s, std::move(v)); break; }
                auto* t = lookup(n.s);
                if (!t || t->k != value::dict || !t->ns) fail("cannot assign to attribute '" + n.attr + "': '" + n.s + "' is not a namespace");
                if (measure(v, t->m.get(), 0) >= value::max_nesting) fail("data nested too deep");
                bool found = false;
                for (auto& [k, x] : *t->m) if (k == n.attr) { x = std::move(v); found = true; break; }
                if (!found) t->m->emplace_back(n.attr, std::move(v));
                break; }
            case node::set_block: { std::string s; exec(n.body, s); assign(n.s, value::of(std::move(s))); break; }
            case node::macro: {
                auto m = std::make_shared<macro_t>();   // the macro shares the tree: bodies are referenced, not copied
                m->name = n.s; m->params = n.names;
                value v; v.k = value::macro; v.f = m; v.i = std::int64_t(reinterpret_cast<std::intptr_t>(&n));
                assign(n.s, std::move(v)); break; }
            case node::brk: return flow::brk;
            case node::cont: return flow::cont;
            }
        }
        return flow::next;
    }
    // Real nesting depth of v; raises when v contains the namespace `target` (the assignment would make a cycle).
    int measure(const value& v, const dict_t* target, int level) {
        if (level > 2 * value::max_nesting) fail("data nested too deep");
        tick();
        int d = 0;
        if (v.k == value::list) for (auto& e : *v.l) d = std::max(d, 1 + measure(e, target, level + 1));
        else if (v.k == value::dict) {
            if (v.m.get() == target) fail("a namespace cannot contain itself");
            for (auto& e : *v.m) d = std::max(d, 1 + measure(e.second, target, level + 1));
        }
        return d;
    }
    list_t iterate(const value& v) {
        list_t items;
        switch (v.k) {
        case value::list: items = *v.l; break;
        case value::dict: for (auto& [k, x] : *v.m) items.push_back(value::of(k)); break;
        case value::string: for (auto c : tx::utf8_to_utf32(v.s)) { std::string s; tx::append_utf8(s, c); items.push_back(value::of(std::move(s))); } break;
        case value::undef: break;
        default: fail(std::string("cannot iterate over a ") + v.type_name());
        }
        tick(items.size());
        return items;
    }
    void bind_targets(const std::vector<std::string>& names, const value& item) {
        if (names.size() == 1) { frames_.back().emplace_back(names[0], item); return; }
        if (item.k != value::list || item.l->size() != names.size()) fail("cannot unpack loop item into " + std::to_string(names.size()) + " names");
        for (std::size_t i = 0; i < names.size(); ++i) frames_.back().emplace_back(names[i], (*item.l)[i]);
    }
    void run_for(const node& n, std::string& out) {
        auto items = iterate(eval(*n.e));
        if (n.cond) {
            list_t keep;
            for (auto& it : items) { frames_.emplace_back(); bind_targets(n.names, it); bool ok = eval(*n.cond).truthy(); frames_.pop_back(); if (ok) keep.push_back(std::move(it)); }
            items = std::move(keep);
        }
        if (items.empty()) { exec(n.alt, out); return; }
        auto len = std::int64_t(items.size());
        for (std::int64_t i = 0; i < len; ++i) {
            tick();
            dict_t loop{{"index", value::of(i + 1)}, {"index0", value::of(i)}, {"first", value::of(i == 0)}, {"last", value::of(i + 1 == len)}, {"length", value::of(len)},
                        {"revindex", value::of(len - i)}, {"revindex0", value::of(len - i - 1)},
                        {"previtem", i ? items[std::size_t(i - 1)] : value()}, {"nextitem", i + 1 < len ? items[std::size_t(i + 1)] : value()}};
            frames_.emplace_back();
            bind_targets(n.names, items[std::size_t(i)]);
            frames_.back().emplace_back("loop", value::of(std::move(loop)));
            struct pop { std::vector<frame>& f; ~pop() { f.pop_back(); } } guard{frames_};
            if (exec(n.body, out) == flow::brk) break;
        }
    }

    static std::int64_t as_int(const value& v, const char* what) {
        if (v.k == value::integer) return v.i;
        if (v.k == value::boolean) return v.b;
        if (v.k == value::real && v.d == std::floor(v.d) && std::fabs(v.d) < 9e15) return std::int64_t(v.d);
        fail(std::string(what) + " must be an integer");
    }
    static bool equal(const value& a, const value& b, int depth = 0) {
        if (depth > 64) return false;
        if (a.is_num() && b.is_num()) return a.num() == b.num();
        if (a.k != b.k) return false;
        switch (a.k) {
        case value::undef: case value::none: return true;
        case value::string: return a.s == b.s;
        case value::list: {
            if (a.l->size() != b.l->size()) return false;
            for (std::size_t i = 0; i < a.l->size(); ++i) if (!equal((*a.l)[i], (*b.l)[i], depth + 1)) return false;
            return true; }
        case value::dict: {
            if (a.m->size() != b.m->size()) return false;
            for (auto& [k, v] : *a.m) { auto* o = b.get(k); if (!o || !equal(v, *o, depth + 1)) return false; }
            return true; }
        default: return a.f == b.f;
        }
    }
    static int order(const value& a, const value& b) {
        if (a.is_num() && b.is_num()) return a.num() < b.num() ? -1 : a.num() > b.num() ? 1 : 0;
        if (a.k == value::string && b.k == value::string) return a.s < b.s ? -1 : a.s > b.s ? 1 : 0;
        fail(std::string("cannot order ") + a.type_name() + " and " + b.type_name());
    }
    bool contains(const value& hay, const value& x) {
        if (hay.k == value::string) { if (x.k != value::string) fail("'in <string>' needs a string on the left"); return hay.s.find(x.s) != std::string::npos; }
        if (hay.k == value::list) { tick(hay.l->size()); for (auto& e : *hay.l) if (equal(e, x)) return true; return false; }
        if (hay.k == value::dict) return x.k == value::string && hay.get(x.s);
        if (hay.k == value::undef) return false;
        fail(std::string("'in' needs a string, list or dict, not ") + hay.type_name());
    }
    static std::int64_t floordiv(std::int64_t a, std::int64_t b) { std::int64_t q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
    value arith(const std::string& op, const value& a, const value& b) {
        if (op == "+") {
            if (a.k == value::string && b.k == value::string) { tick(1 + (a.s.size() + b.s.size()) / 16); return value::of(a.s + b.s); }
            if (a.k == value::list && b.k == value::list) { tick(a.l->size() + b.l->size()); list_t l = *a.l; l.insert(l.end(), b.l->begin(), b.l->end()); return value::of(std::move(l)); }
        }
        if (op == "*" && ((a.k == value::string && b.k == value::integer) || (a.k == value::integer && b.k == value::string))) {
            auto& s = a.k == value::string ? a.s : b.s; auto n = a.k == value::integer ? a.i : b.i;
            if (n <= 0 || s.empty()) return value::of(std::string());
            if (std::uint64_t(n) > lim_.max_output / s.size()) fail("string repetition too large");
            tick(1 + s.size() * std::uint64_t(n) / 16);
            std::string o; o.reserve(s.size() * std::size_t(n)); for (std::int64_t i = 0; i < n; ++i) o += s; return value::of(std::move(o));
        }
        if (!a.is_num() || !b.is_num()) fail("unsupported operand types for " + op + ": " + a.type_name() + " and " + b.type_name());
        bool ints = a.k != value::real && b.k != value::real;
        if (ints && op != "/" && op != "**") {
            std::int64_t x = a.k == value::boolean ? a.b : a.i, y = b.k == value::boolean ? b.b : b.i, r = 0;
            if (op == "+") { if (__builtin_add_overflow_p_compat(x, y, r)) fail("integer overflow"); return value::of(r); }
            if (op == "-") { if (y == INT64_MIN || __builtin_add_overflow_p_compat(x, -y, r)) fail("integer overflow"); return value::of(r); }
            if (op == "*") { if (mul_overflow(x, y, r)) fail("integer overflow"); return value::of(r); }
            if (y == 0) fail("division by zero");
            if (x == INT64_MIN && y == -1) fail("integer overflow");
            if (op == "//") return value::of(floordiv(x, y));
            if (op == "%") return value::of(x - floordiv(x, y) * y);
        }
        double x = a.num(), y = b.num();
        if (op == "+") return value::of(x + y);
        if (op == "-") return value::of(x - y);
        if (op == "*") return value::of(x * y);
        if (op == "**") {
            double r = std::pow(x, y);
            if (ints && y >= 0 && std::fabs(r) < 9e15) return value::of(std::int64_t(r));
            return value::of(r);
        }
        if (y == 0) fail("division by zero");
        if (op == "/") return value::of(x / y);
        if (op == "//") return value::of(std::floor(x / y));
        if (op == "%") return value::of(x - std::floor(x / y) * y);
        fail("unsupported operator " + op);
    }
    static bool __builtin_add_overflow_p_compat(std::int64_t x, std::int64_t y, std::int64_t& r) {
        if ((y > 0 && x > INT64_MAX - y) || (y < 0 && x < INT64_MIN - y)) return true;
        r = x + y; return false;
    }
    static bool mul_overflow(std::int64_t x, std::int64_t y, std::int64_t& r) {
        if (x == 0 || y == 0) { r = 0; return false; }
        double p = double(x) * double(y);
        if (std::fabs(p) > 9.0e18) return true;
        r = x * y; return false;
    }

    static std::int64_t clamp_index(std::int64_t i, std::int64_t n) { if (i < 0) i += n; return i; }
    value subscript(const value& o, const value& ix) {
        if (o.k == value::dict) { if (ix.k == value::string) if (auto* v = o.get(ix.s)) return *v; return {}; }
        if (o.k == value::list) { if (!ix.is_num()) return {}; auto i = clamp_index(as_int(ix, "list index"), std::int64_t(o.l->size())); return i >= 0 && i < std::int64_t(o.l->size()) ? (*o.l)[std::size_t(i)] : value(); }
        if (o.k == value::string) {
            if (!ix.is_num()) return {};
            auto u = tx::utf8_to_utf32(o.s); auto i = clamp_index(as_int(ix, "string index"), std::int64_t(u.size()));
            if (i < 0 || i >= std::int64_t(u.size())) return {};
            std::string s; tx::append_utf8(s, u[std::size_t(i)]); return value::of(std::move(s));
        }
        if (o.k == value::undef) fail("cannot index an undefined value");
        if (o.k == value::none) fail("cannot index None");
        return {};
    }
    value slice(const value& o, const value& lo, const value& hi, const value& st) {
        std::int64_t step = st.k == value::undef || st.k == value::none ? 1 : as_int(st, "slice step");
        if (step == 0) fail("slice step cannot be zero");
        auto pick = [&](std::int64_t n, auto&& take) {
            auto bound = [&](const value& v, std::int64_t dflt) {
                if (v.k == value::undef || v.k == value::none) return dflt;
                std::int64_t i = as_int(v, "slice index"); if (i < 0) i += n;
                return step > 0 ? std::clamp<std::int64_t>(i, 0, n) : std::clamp<std::int64_t>(i, -1, n - 1);
            };
            std::int64_t a = bound(lo, step > 0 ? 0 : n - 1), b = bound(hi, step > 0 ? n : -1);
            for (std::int64_t i = a; step > 0 ? i < b : i > b; i += step) { tick(); take(std::size_t(i)); }
        };
        if (o.k == value::list) { list_t l; pick(std::int64_t(o.l->size()), [&](std::size_t i) { l.push_back((*o.l)[i]); }); return value::of(std::move(l)); }
        if (o.k == value::string) { auto u = tx::utf8_to_utf32(o.s); std::string s; pick(std::int64_t(u.size()), [&](std::size_t i) { tx::append_utf8(s, u[i]); }); return value::of(std::move(s)); }
        fail(std::string("cannot slice a ") + o.type_name());
    }

    struct args_t {
        std::vector<value> pos; std::vector<std::pair<std::string, value>> kw;
        const value* named(const char* n, std::size_t i) const { for (auto& [k, v] : kw) if (k == n) return &v; return i < pos.size() ? &pos[i] : nullptr; }
    };
    args_t arguments(const expr& e, std::size_t first) {
        args_t a;
        for (std::size_t i = first; i < e.a.size(); ++i) a.pos.push_back(eval(*e.a[i]));
        for (auto& [k, x] : e.kw) a.kw.emplace_back(k, eval(*x));
        return a;
    }
    static std::string str_arg(const args_t& a, const char* name, std::size_t i, const char* fn, const char* dflt = nullptr) {
        auto* v = a.named(name, i);
        if (!v) { if (dflt) return dflt; fail(std::string(fn) + "() needs argument '" + name + "'"); }
        if (v->k != value::string) fail(std::string(fn) + "(): '" + name + "' must be a string");
        return v->s;
    }
    static bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
    static std::string strip(const std::string& s, const std::string* chars, bool left, bool right) {
        auto drop = [&](char c) { return chars ? chars->find(c) != std::string::npos : is_space(c); };
        std::size_t b = 0, e = s.size();
        if (left) while (b < e && drop(s[b])) ++b;
        if (right) while (e > b && drop(s[e - 1])) --e;
        return s.substr(b, e - b);
    }
    static std::string title(const std::string& s) {
        std::string o = tx::to_lower(s); bool start = true;
        for (auto& c : o) { bool al = std::isalpha((unsigned char)c) != 0; if (al && start) c = char(std::toupper((unsigned char)c)); start = !al; }
        return o;
    }
    static std::string capitalize(const std::string& s) {
        if (s.empty()) return s;
        std::size_t i = 0; tx::decode_utf8(s, i);
        return tx::to_upper(s.substr(0, i)) + tx::to_lower(s.substr(i));
    }
    std::string replace(const std::string& s, const std::string& a, const std::string& b, std::int64_t count) {
        std::string o; std::size_t p = 0;
        if (a.empty()) {   // Python: b between every character
            auto u = tx::utf8_to_utf32(s);
            if ((u.size() + 1) * b.size() + s.size() > lim_.max_output) fail("replace result too large");
            o += b; for (auto c : u) { tx::append_utf8(o, c); o += b; }
            tick(1 + o.size() / 16); return o;
        }
        for (std::size_t f; count != 0 && (f = s.find(a, p)) != std::string::npos; --count) {
            o.append(s, p, f - p); o += b; p = f + a.size();
            if (o.size() > lim_.max_output) fail("replace result too large");
        }
        o.append(s, p); tick(1 + o.size() / 16); return o;
    }

    value method(const value& self, const std::string& name, const args_t& a) {
        auto arg = [&](std::size_t i) -> const value* { return i < a.pos.size() ? &a.pos[i] : nullptr; };
        if (self.k == value::string) {
            const std::string& s = self.s;
            auto chars = [&]() -> const std::string* { auto* v = arg(0); if (!v || v->k == value::none) return nullptr; if (v->k != value::string) fail(name + "(): argument must be a string"); return &v->s; };
            if (name == "strip") return value::of(strip(s, chars(), true, true));
            if (name == "lstrip") return value::of(strip(s, chars(), true, false));
            if (name == "rstrip") return value::of(strip(s, chars(), false, true));
            if (name == "upper") return value::of(tx::to_upper(s));
            if (name == "lower") return value::of(tx::to_lower(s));
            if (name == "title") return value::of(title(s));
            if (name == "capitalize") return value::of(capitalize(s));
            if (name == "startswith" || name == "endswith") {
                auto* p = arg(0); if (!p) fail(name + "() needs an argument");
                auto one = [&](const value& x) { if (x.k != value::string) fail(name + "(): argument must be a string"); return name == "startswith" ? s.starts_with(x.s) : s.ends_with(x.s); };
                if (p->k == value::list) { for (auto& x : *p->l) if (one(x)) return value::of(true); return value::of(false); }
                return value::of(one(*p));
            }
            if (name == "replace") return value::of(replace(s, str_arg(a, "old", 0, "replace"), str_arg(a, "new", 1, "replace"), arg(2) ? as_int(*arg(2), "count") : -1));
            if (name == "find") { auto f = s.find(str_arg(a, "sub", 0, "find")); return value::of(f == std::string::npos ? std::int64_t(-1) : std::int64_t(tx::count_code_points(std::string_view(s).substr(0, f)))); }
            if (name == "count") { auto sub = str_arg(a, "sub", 0, "count"); std::int64_t n = 0; if (!sub.empty()) for (std::size_t p = 0; (p = s.find(sub, p)) != std::string::npos; p += sub.size()) ++n; return value::of(n); }
            if (name == "split" || name == "rsplit") {
                list_t out; auto* sepv = arg(0); std::int64_t maxsplit = arg(1) ? as_int(*arg(1), "maxsplit") : -1;
                if (name == "rsplit" && maxsplit >= 0) fail("rsplit() with maxsplit is not supported");
                if (!sepv || sepv->k == value::none) {
                    std::size_t i = 0;
                    while (i < s.size()) {
                        while (i < s.size() && is_space(s[i])) ++i;
                        if (i >= s.size()) break;
                        if (maxsplit >= 0 && std::int64_t(out.size()) >= maxsplit) { out.push_back(value::of(strip(s.substr(i), nullptr, false, true))); break; }
                        std::size_t b = i; while (i < s.size() && !is_space(s[i])) ++i;
                        out.push_back(value::of(s.substr(b, i - b)));
                    }
                } else {
                    if (sepv->k != value::string || sepv->s.empty()) fail("split(): bad separator");
                    std::size_t p = 0;
                    for (std::size_t f; (maxsplit < 0 || std::int64_t(out.size()) < maxsplit) && (f = s.find(sepv->s, p)) != std::string::npos; p = f + sepv->s.size()) out.push_back(value::of(s.substr(p, f - p)));
                    out.push_back(value::of(s.substr(p)));
                }
                tick(out.size() + s.size() / 16); return value::of(std::move(out));
            }
            if (name == "splitlines") { list_t out; std::size_t p = 0; for (std::size_t f; (f = s.find('\n', p)) != std::string::npos; p = f + 1) out.push_back(value::of(s.substr(p, f - p))); if (p < s.size()) out.push_back(value::of(s.substr(p))); return value::of(std::move(out)); }
            if (name == "join") { if (!arg(0)) fail("join() needs an argument"); return value::of(join(iterate(*arg(0)), s)); }
            if (name == "isdigit") { bool ok = !s.empty(); for (unsigned char c : s) ok &= std::isdigit(c) != 0; return value::of(ok); }
        } else if (self.k == value::dict) {
            if (name == "items") { list_t l; for (auto& [k, v] : *self.m) l.push_back(pair_of(k, v)); tick(l.size()); return value::of(std::move(l)); }
            if (name == "keys") { list_t l; for (auto& [k, v] : *self.m) l.push_back(value::of(k)); tick(l.size()); return value::of(std::move(l)); }
            if (name == "values") { list_t l; for (auto& [k, v] : *self.m) l.push_back(v); tick(l.size()); return value::of(std::move(l)); }
            if (name == "get") {
                auto* k = arg(0); if (!k) fail("get() needs a key");
                if (k->k == value::string) if (auto* v = self.get(k->s)) return *v;
                return arg(1) ? *arg(1) : value::of_none();
            }
            if (auto* v = self.get(name); v && v->k == value::macro) return call_macro(*v, a);
        } else if (self.k == value::list) {
            // Data is immutable, as in transformers' sandboxed environment: build a new list with + instead.
            if (name == "append" || name == "extend" || name == "insert" || name == "pop" || name == "remove" || name == "clear" || name == "sort" || name == "reverse")
                fail("list." + name + "() is not available: template data is immutable (use + and a namespace)");
            if (name == "index") { if (!arg(0)) fail("index() needs an argument"); for (std::size_t i = 0; i < self.l->size(); ++i) if (equal((*self.l)[i], *arg(0))) return value::of(std::int64_t(i)); fail("index(): value not in list"); }
            if (name == "count") { if (!arg(0)) fail("count() needs an argument"); std::int64_t n = 0; for (auto& x : *self.l) n += equal(x, *arg(0)); return value::of(n); }
        }
        fail(std::string("a ") + self.type_name() + " has no method '" + name + "'");
    }
    static value pair_of(const std::string& k, const value& v) { auto p = value::of(list_t{value::of(k), v}); p.tup = true; return p; }
    std::string join(const list_t& items, const std::string& sep) {
        std::string o;
        for (std::size_t i = 0; i < items.size(); ++i) { if (i) o += sep; to_text(o, items[i]); if (o.size() > lim_.max_output) fail("join result too large"); }
        tick(1 + o.size() / 16); return o;
    }
    value call_macro(const value& mv, const args_t& a) {
        deeper g(*this); deeper g2(*this); deeper g3(*this); deeper g4(*this);   // macro calls recurse: count them four times
        auto& n = *reinterpret_cast<const node*>(std::intptr_t(mv.i));
        frame f;
        for (std::size_t i = 0; i < n.names.size(); ++i) {
            const value* v = a.named(n.names[i].c_str(), i);
            f.emplace_back(n.names[i], v ? *v : n.defaults[i] ? eval(*n.defaults[i]) : value());
        }
        // a macro sees the template's globals and its own arguments
        std::vector<frame> saved; saved.push_back(frames_[0]);
        std::swap(saved, frames_); frames_.push_back(std::move(f));
        struct restore { std::vector<frame>& cur; std::vector<frame>& old; ~restore() { old[0] = std::move(cur[0]); std::swap(cur, old); } } r{frames_, saved};
        std::string out; exec(n.body, out);
        return value::of(std::move(out));
    }
    value call_global(const std::string& name, const args_t& a) {
        auto arg = [&](std::size_t i) -> const value* { return i < a.pos.size() ? &a.pos[i] : nullptr; };
        if (name == "raise_exception") raise(errc::invalid_argument, "{}", arg(0) ? text_of(*arg(0)) : std::string("chat template raised an exception"));
        if (name == "namespace" || name == "dict") {
            dict_t d;
            if (arg(0) && arg(0)->k == value::dict) d = *arg(0)->m;
            for (auto& [k, v] : a.kw) { bool f = false; for (auto& [dk, dv] : d) if (dk == k) { dv = v; f = true; } if (!f) d.emplace_back(k, v); }
            auto v = value::of(std::move(d)); v.ns = name == "namespace"; return v;
        }
        if (name == "range") {
            std::int64_t lo = 0, hi = 0, st = 1;
            if (a.pos.size() == 1) hi = as_int(a.pos[0], "range()"); else if (a.pos.size() >= 2) { lo = as_int(a.pos[0], "range()"); hi = as_int(a.pos[1], "range()"); if (a.pos.size() > 2) st = as_int(a.pos[2], "range()"); }
            else fail("range() needs an argument");
            if (st == 0) fail("range() step cannot be zero");
            list_t l;
            for (std::int64_t i = lo; st > 0 ? i < hi : i > hi; i += st) { if (l.size() >= 1'000'000) fail("range() too large"); l.push_back(value::of(i)); }
            tick(l.size()); return value::of(std::move(l));
        }
        if (name == "strftime_now") {
            auto fmt = str_arg(a, "format", 0, "strftime_now");
            if (fmt.size() > 128) fail("strftime_now(): format too long");
            std::time_t t = now_ ? now_ : std::time(nullptr); std::tm tm{};
#if defined(_WIN32)
            localtime_s(&tm, &t);
#else
            localtime_r(&t, &tm);
#endif
            char buf[512]; auto n = std::strftime(buf, sizeof buf, fmt.c_str(), &tm);
            return value::of(std::string(buf, n));
        }
        fail("'" + name + "' is not defined or not callable");
    }

    value filter(const std::string& name, const value& x, const args_t& a) {
        auto arg = [&](std::size_t i) -> const value* { return i < a.pos.size() ? &a.pos[i] : nullptr; };
        auto as_str = [&]() { return x.k == value::string ? x.s : text_of(x); };
        if (name == "trim" || name == "strip") { auto* c = a.named("chars", 0); return value::of(strip(as_str(), c && c->k == value::string ? &c->s : nullptr, true, true)); }
        if (name == "length" || name == "count") {
            if (x.k == value::string) return value::of(std::int64_t(tx::count_code_points(x.s)));
            if (x.k == value::list) return value::of(std::int64_t(x.l->size()));
            if (x.k == value::dict) return value::of(std::int64_t(x.m->size()));
            fail(std::string("length of a ") + x.type_name());
        }
        if (name == "lower") return value::of(tx::to_lower(as_str()));
        if (name == "upper") return value::of(tx::to_upper(as_str()));
        if (name == "title") {   // Jinja's filter: a word starts after whitespace or one of - ( { [ <
            std::string s = as_str(), o; std::size_t b = 0;
            auto flush = [&](std::size_t e) { if (e > b) o += capitalize(s.substr(b, e - b)); b = e; };
            for (std::size_t i = 0; i < s.size(); ++i) if (is_space(s[i]) || s[i] == '-' || s[i] == '(' || s[i] == '{' || s[i] == '[' || s[i] == '<') { flush(i); o += s[i]; b = i + 1; }
            flush(s.size()); return value::of(std::move(o));
        }
        if (name == "capitalize") return value::of(capitalize(as_str()));
        if (name == "tojson") {
            auto* ind = a.named("indent", 0);
            std::string o; to_json(o, x, ind && ind->k != value::none ? int(std::clamp<std::int64_t>(as_int(*ind, "indent"), 0, 16)) : -1);
            if (o.size() > lim_.max_output) fail("tojson result too large");
            tick(1 + o.size() / 16); return value::of(std::move(o));
        }
        if (name == "default" || name == "d") {
            auto* dv = a.named("default_value", 0); auto* fl = a.named("boolean", 1);
            bool use = x.k == value::undef || (fl && fl->truthy() && !x.truthy());
            return use ? (dv ? *dv : value::of(std::string())) : x;
        }
        if (name == "join") { auto* sep = a.named("d", 0); auto items = iterate(x); if (auto* at = a.named("attribute", 1)) for (auto& it : items) it = subscript(it, *at); return value::of(join(items, sep ? text_of(*sep) : std::string())); }
        if (name == "first") { auto l = iterate(x); return l.empty() ? value() : l.front(); }
        if (name == "last") { auto l = iterate(x); return l.empty() ? value() : l.back(); }
        if (name == "list") return value::of(iterate(x));
        if (name == "string" || name == "safe" || name == "e" || name == "escape") {
            auto s = as_str();
            if (name == "e" || name == "escape") { std::string o; for (char c : s) { switch (c) { case '&': o += "&amp;"; break; case '<': o += "&lt;"; break; case '>': o += "&gt;"; break; case '"': o += "&#34;"; break; case '\'': o += "&#39;"; break; default: o += c; } } s = std::move(o); }
            return value::of(std::move(s));
        }
        if (name == "int") {
            if (x.k == value::integer) return x;
            if (x.k == value::boolean) return value::of(std::int64_t(x.b));
            if (x.k == value::real) return std::fabs(x.d) < 9e18 ? value::of(std::int64_t(x.d)) : (arg(0) ? *arg(0) : value::of(std::int64_t(0)));
            if (x.k == value::string) { char* end = nullptr; auto s = strip(x.s, nullptr, true, true); double d = std::strtod(s.c_str(), &end); if (!s.empty() && !*end && std::fabs(d) < 9e18) return value::of(std::int64_t(d)); }
            return arg(0) ? *arg(0) : value::of(std::int64_t(0));
        }
        if (name == "float") {
            if (x.is_num()) return value::of(x.num());
            if (x.k == value::string) { char* end = nullptr; auto s = strip(x.s, nullptr, true, true); double d = std::strtod(s.c_str(), &end); if (!s.empty() && !*end) return value::of(d); }
            return arg(0) ? *arg(0) : value::of(0.0);
        }
        if (name == "bool") return value::of(x.truthy());
        if (name == "abs") { if (x.k == value::integer) return value::of(x.i < 0 ? -x.i : x.i); if (x.is_num()) return value::of(std::fabs(x.num())); fail("abs of a non-number"); }
        if (name == "round") {
            if (!x.is_num()) fail("round of a non-number");
            auto* p = a.named("precision", 0); double m = std::pow(10.0, double(p ? std::clamp<std::int64_t>(as_int(*p, "precision"), 0, 15) : 0));
            return value::of(std::nearbyint(x.num() * m) / m);
        }
        if (name == "replace") return value::of(replace(as_str(), str_arg(a, "old", 0, "replace"), str_arg(a, "new", 1, "replace"), arg(2) ? as_int(*arg(2), "count") : -1));
        if (name == "items") { if (x.k == value::undef) return value::of(list_t{}); if (x.k != value::dict) fail("items needs a mapping"); return method(x, "items", {}); }
        if (name == "reverse") { if (x.k == value::string) return slice(x, {}, {}, value::of(std::int64_t(-1))); auto l = iterate(x); std::reverse(l.begin(), l.end()); return value::of(std::move(l)); }
        if (name == "sort" || name == "unique" || name == "min" || name == "max" || name == "sum") {
            auto l = iterate(x);
            auto* cs = a.named("case_sensitive", name == "sort" ? 1 : 0);
            bool fold = !(cs && cs->truthy());   // Jinja compares strings without case unless asked
            auto key = [&](const value& v) { return fold && v.k == value::string ? value::of(tx::to_lower(v.s)) : v; };
            if (auto* at = a.named("attribute", name == "sort" ? 2 : 1); at && name != "unique") {
                if (name == "sort") { tick(l.size() * 8); std::stable_sort(l.begin(), l.end(), [&](const value& p, const value& q) { return order(key(subscript(p, *at)), key(subscript(q, *at))) < 0; }); if (auto* r = a.named("reverse", 0); r && r->truthy()) std::reverse(l.begin(), l.end()); return value::of(std::move(l)); }
                for (auto& it : l) it = subscript(it, *at);
            }
            if (name == "sort") { tick(l.size() * 8); std::stable_sort(l.begin(), l.end(), [&](const value& p, const value& q) { return order(key(p), key(q)) < 0; }); if (auto* r = a.named("reverse", 0); r && r->truthy()) std::reverse(l.begin(), l.end()); return value::of(std::move(l)); }
            if (name == "unique") { list_t o; tick(l.size() * l.size() / 8); for (auto& it : l) { bool dup = false; for (auto& e : o) dup |= equal(key(e), key(it)); if (!dup) o.push_back(it); } return value::of(std::move(o)); }
            if (name == "sum") { value acc = a.named("start", 1) ? *a.named("start", 1) : value::of(std::int64_t(0)); for (auto& it : l) acc = arith("+", acc, it); return acc; }
            if (l.empty()) return {};
            auto best = l[0]; for (auto& it : l) if (int c = order(key(it), key(best)); c != 0 && (name == "min") == (c < 0)) best = it;
            return best;
        }
        if (name == "dictsort") { if (x.k != value::dict) fail("dictsort needs a mapping"); auto d = *x.m; tick(d.size() * 8); std::stable_sort(d.begin(), d.end(), [](auto& p, auto& q) { return tx::to_lower(p.first) < tx::to_lower(q.first); }); list_t l; for (auto& [k, v] : d) l.push_back(pair_of(k, v)); return value::of(std::move(l)); }
        if (name == "indent") {
            auto* w = a.named("width", 0); auto* first = a.named("first", 1); auto* blank = a.named("blank", 2);
            std::string pad = w && w->k == value::string ? w->s : std::string(std::size_t(std::clamp<std::int64_t>(w ? as_int(*w, "width") : 4, 0, 256)), ' ');
            auto s = as_str(); std::string o; bool at_start = true, do_first = first && first->truthy(), do_blank = blank && blank->truthy();
            std::size_t line = 0;
            for (std::size_t p = 0; p <= s.size(); ++p) {
                if (p < s.size() && s[p] != '\n') continue;
                std::string_view ln(s.data() + line, p - line);
                if (at_start ? do_first : (!ln.empty() || do_blank)) o += pad;
                o += ln; if (p < s.size()) o += '\n';
                at_start = false; line = p + 1;
                if (o.size() > lim_.max_output) fail("indent result too large");
            }
            tick(1 + o.size() / 16); return value::of(std::move(o));
        }
        if (name == "selectattr" || name == "rejectattr" || name == "select" || name == "reject") {
            bool attr = name.ends_with("attr"), want = name[0] == 's';
            auto l = iterate(x); list_t o; std::size_t at = attr ? 1 : 0;
            if (attr && (!arg(0) || arg(0)->k != value::string)) fail(name + " needs an attribute name");
            for (auto& it : l) {
                value v = attr ? attribute(it, arg(0)->s) : it;
                bool ok = v.truthy();
                if (arg(at)) { if (arg(at)->k != value::string) fail(name + ": test name must be a string"); args_t ta; for (std::size_t i = at + 1; i < a.pos.size(); ++i) ta.pos.push_back(a.pos[i]); ok = test(arg(at)->s, v, ta); }
                if (ok == want) o.push_back(it);
            }
            return value::of(std::move(o));
        }
        if (name == "map") {
            auto l = iterate(x);
            if (auto* at = a.named("attribute", 99)) { auto* dv = a.named("default", 99); for (auto& it : l) { it = at->k == value::string ? attribute(it, at->s) : subscript(it, *at); if (it.k == value::undef && dv) it = *dv; } return value::of(std::move(l)); }
            if (!arg(0) || arg(0)->k != value::string) fail("map needs a filter name or attribute=");
            args_t fa; for (std::size_t i = 1; i < a.pos.size(); ++i) fa.pos.push_back(a.pos[i]);
            for (auto& it : l) it = filter(arg(0)->s, it, fa);
            return value::of(std::move(l));
        }
        fail("unknown filter '" + name + "'");
    }
    bool test(const std::string& name, const value& x, const args_t& a) {
        auto arg = [&]() -> const value& { if (a.pos.empty()) fail("test '" + name + "' needs an argument"); return a.pos[0]; };
        if (name == "defined") return x.k != value::undef;
        if (name == "undefined") return x.k == value::undef;
        if (name == "none") return x.k == value::none;
        if (name == "string") return x.k == value::string;
        if (name == "mapping") return x.k == value::dict;
        if (name == "iterable") return x.k == value::string || x.k == value::list || x.k == value::dict;
        if (name == "sequence") return x.k == value::string || x.k == value::list || x.k == value::dict;
        if (name == "number") return x.k == value::integer || x.k == value::real || x.k == value::boolean;
        if (name == "integer") return x.k == value::integer;
        if (name == "float") return x.k == value::real;
        if (name == "boolean") return x.k == value::boolean;
        if (name == "true") return x.k == value::boolean && x.b;
        if (name == "false") return x.k == value::boolean && !x.b;
        if (name == "callable") return x.k == value::macro;
        if (name == "eq" || name == "equalto" || name == "sameas") return equal(x, arg());
        if (name == "ne") return !equal(x, arg());
        if (name == "lt" || name == "lessthan") return order(x, arg()) < 0;
        if (name == "le") return order(x, arg()) <= 0;
        if (name == "gt" || name == "greaterthan") return order(x, arg()) > 0;
        if (name == "ge") return order(x, arg()) >= 0;
        if (name == "in") return contains(arg(), x);
        if (name == "odd") return (as_int(x, "odd") & 1) != 0;
        if (name == "even") return (as_int(x, "even") & 1) == 0;
        if (name == "divisibleby") { auto d = as_int(arg(), "divisibleby"); if (!d) fail("division by zero"); return as_int(x, "divisibleby") % d == 0; }
        if (name == "lower") return x.k == value::string && tx::to_lower(x.s) == x.s;
        if (name == "upper") return x.k == value::string && tx::to_upper(x.s) == x.s;
        fail("unknown test '" + name + "'");
    }
    value attribute(const value& o, const std::string& name) {
        if (o.k == value::dict) { if (auto* v = o.get(name)) return *v; return {}; }
        if (o.k == value::undef) fail("'" + name + "': the value it is read from is undefined");
        return {};   // a missing attribute is undefined (None.x, "str".x too, as Jinja's getattr falls back)
    }

    value eval(const expr& e) {
        deeper g(*this);
        tick();
        switch (e.k) {
        case expr::lit: return e.v;
        case expr::name: if (auto* v = lookup(e.s)) return *v; return {};
        case expr::attr: return attribute(eval(*e.a[0]), e.s);
        case expr::index: return subscript(eval(*e.a[0]), eval(*e.a[1]));
        case expr::slice: return slice(eval(*e.a[0]), e.a[1] ? eval(*e.a[1]) : value(), e.a[2] ? eval(*e.a[2]) : value(), e.a[3] ? eval(*e.a[3]) : value());
        case expr::list: case expr::tuple: { list_t l; for (auto& x : e.a) l.push_back(eval(*x)); auto v = value::of(std::move(l)); v.tup = e.k == expr::tuple; return v; }
        case expr::dict: {
            dict_t d;
            for (std::size_t i = 0; i + 1 < e.a.size(); i += 2) {
                auto k = eval(*e.a[i]); auto ks = k.k == value::string ? k.s : text_of(k); auto v = eval(*e.a[i + 1]); bool f = false;
                for (auto& [dk, dv] : d) if (dk == ks) { dv = v; f = true; }
                if (!f) d.emplace_back(std::move(ks), std::move(v));
            }
            return value::of(std::move(d)); }
        case expr::call: {
            auto& callee = *e.a[0];
            if (callee.k == expr::attr) {
                auto self = eval(*callee.a[0]);
                auto a = arguments(e, 1);
                if (callee.s == "cycle" && self.k == value::dict && self.get("index0") && self.get("revindex0")) {   // loop.cycle(a, b, ...)
                    if (a.pos.empty()) fail("loop.cycle() needs arguments");
                    return a.pos[std::size_t(self.get("index0")->i) % a.pos.size()];
                }
                return method(self, callee.s, a);
            }
            if (callee.k == expr::name) {
                auto a = arguments(e, 1);
                if (auto* v = lookup(callee.s)) { if (v->k != value::macro) fail("'" + callee.s + "' is not callable"); auto m = *v; return call_macro(m, a); }
                return call_global(callee.s, a);
            }
            fail("this expression is not callable"); }
        case expr::filter: { auto x = eval(*e.a[0]); return filter(e.s, x, arguments(e, 1)); }
        case expr::test: { auto x = eval(*e.a[0]); return value::of(test(e.s, x, arguments(e, 1)) != e.neg); }
        case expr::unary: {
            auto x = eval(*e.a[0]);
            if (!x.is_num()) fail("unary " + e.s + " needs a number");
            if (e.s == "+") return x;
            if (x.k == value::real) return value::of(-x.d);
            auto i = x.k == value::boolean ? std::int64_t(x.b) : x.i;
            if (i == INT64_MIN) fail("integer overflow");
            return value::of(-i); }
        case expr::logic_not: return value::of(!eval(*e.a[0]).truthy());
        case expr::logic_and: { auto l = eval(*e.a[0]); return l.truthy() ? eval(*e.a[1]) : l; }
        case expr::logic_or: { auto l = eval(*e.a[0]); return l.truthy() ? l : eval(*e.a[1]); }
        case expr::cond: return eval(*e.a[0]).truthy() ? eval(*e.a[1]) : e.a.size() > 2 ? eval(*e.a[2]) : value();
        case expr::concat: {
            std::string s; to_text(s, eval(*e.a[0])); to_text(s, eval(*e.a[1]));
            if (s.size() > lim_.max_output) fail("string too large");
            tick(1 + s.size() / 16); return value::of(std::move(s)); }
        case expr::binary: {
            auto l = eval(*e.a[0]); auto r = eval(*e.a[1]);
            if (e.s == "==") return value::of(equal(l, r));
            if (e.s == "!=") return value::of(!equal(l, r));
            if (e.s == "in") return value::of(contains(r, l));
            if (e.s == "not in") return value::of(!contains(r, l));
            if (e.s == "<") return value::of(order(l, r) < 0);
            if (e.s == ">") return value::of(order(l, r) > 0);
            if (e.s == "<=") return value::of(order(l, r) <= 0);
            if (e.s == ">=") return value::of(order(l, r) >= 0);
            if (l.k == value::undef || r.k == value::undef) fail("operator " + e.s + " on an undefined value");
            return arith(e.s, l, r); }
        }
        return {};
    }
};
}  // namespace jinja

// What a template is rendered with, besides the messages.
struct chat_options {
    bool add_generation_prompt = true;        // end the prompt with the header of the assistant's turn
    std::string bos_token, eos_token;         // apply_chat_template fills these from the tokenizer
    json::value tools;                        // OpenAI-style tool list (null: none); templates print it with tojson
    json::value extra;                        // an object of further template variables ({"enable_thinking": false}, ...)
    std::time_t now = 0;                      // strftime_now()'s clock; 0 = the system's
    render_limits limits;
};

// A parsed template; parse once, render any number of times (render is const and thread-safe).
class chat_template {
public:
    explicit chat_template(std::string_view source, const render_limits& lim = {}) {
        if (source.size() > lim.max_template) jinja::fail("template too large");
        jinja::parser p(jinja::lex(source), lim.max_depth);
        body_ = std::make_shared<const jinja::body_t>(p.parse());
    }
    // messages: [{"role": "user", "content": "..."}, ...] (any JSON the template expects: tool_calls, content parts).
    std::string render(const json::value& messages, const chat_options& o = {}) const {
        jinja::interp in(o.limits, o.now);
        in.define("messages", jinja::from_json(messages));
        in.define("add_generation_prompt", jinja::value::of(o.add_generation_prompt));
        in.define("bos_token", jinja::value::of(o.bos_token));
        in.define("eos_token", jinja::value::of(o.eos_token));
        if (!o.tools.is_null()) in.define("tools", jinja::from_json(o.tools));
        if (o.extra.is_obj()) for (auto& [k, v] : o.extra.obj()) in.define(k, jinja::from_json(v));
        return in.run(*body_);
    }
private:
    std::shared_ptr<const jinja::body_t> body_;
};

inline std::string render_chat_template(std::string_view source, const json::value& messages, const chat_options& o = {}) {
    return chat_template(source, o.limits).render(messages, o);
}

// The model's own template (inference::model::chat_template(), from the .fpk) over `messages`, with bos_token /
// eos_token taken from its tokenizer. Raises not_found when the model file has no chat template.
// The result is tokenized without adding BOS: the template writes the special tokens itself.
template <class Model> std::string apply_chat_template(const Model& m, const json::value& messages, chat_options o = {}) {
    if (m.chat_template().empty()) raise(errc::not_found, "the model file has no chat template");
    if (auto* t = m.tokenizer()) {
        if (o.bos_token.empty()) o.bos_token = t->id_to_token(t->bos());
        if (o.eos_token.empty()) o.eos_token = t->id_to_token(t->eos());
    }
    return render_chat_template(m.chat_template(), messages, o);
}

} // namespace forge::inference
