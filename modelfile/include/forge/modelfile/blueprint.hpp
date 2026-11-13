#pragma once
// Blueprint language: a small declarative graph language (FORGE_PLAN Part E.2).
//
//   blueprint "decoder-lm" opset 1
//   params { d_model: 64  n_layers: 2  rope: { base: 10000 } }
//   state kv[n_layers] : kv_cache(heads: n_kv, dim: head_dim)
//   input tokens : i32[seq]
//   graph {
//     x = embed(W.tok_emb, tokens)
//     repeat l in 0..n_layers { h = rms_norm(x, W.layer[l].attn_norm, eps)  ... }
//     logits = linear(x, W.lm_head)
//   }
//   output logits : f32[seq, vocab]
//   session generative_sequence
//
// Expressions: numbers, lists [a, b], params (dotted), arithmetic + - * /, calls with positional and
// keyword args, `W.<path>` weight references, `state[i].append(k, v)`, fixed-size state ops `state[i].conv(x, w)` and
// `state[i].delta(q, k, v, g, beta)`, tuple assignment `a, b = f(..)`.
// Any spec op is callable by name; positional numeric/list args bind to the op's declared attrs in order.
// Built-in composites: linear(x, W) = x @ W^T, embed(W, ids) = gather, heads(x, n) / merge(x) reshapes,
// scalar(v) = a [1] constant. Number helpers folded at build time: sqrt floor ceil min max mod.
// User composites: `op name(a, b, eps = 1e-6) { t = ...  return expr }` at the top level, called like any op
// (`name(x, y)`, `name(x, y, eps: 1e-5)`). They are inlined into the graph where they are called; recursion (direct
// or through other composites) is a parse error. The text is the canonical form: it is stored verbatim in the .fpk
// "blueprint" section and re-parsed on load.
// Session kinds: generative_sequence (token decoder with state), embedding (one pass, the first output is the
// result vector: sentence / image embeddings, encoder states), one_shot (named inputs to named outputs).
#include <forge/graph/ir.hpp>
#include <forge/ops/ops.gen.hpp>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace forge::modelfile {

struct bp_token { enum kind_t { k_ident, k_num, k_str, k_punct, k_end } kind; std::string text; double num = 0; int line = 0; };

inline std::vector<bp_token> bp_lex(std::string_view s) {
    std::vector<bp_token> t; int line = 1; std::size_t p = 0;
    while (p < s.size()) {
        char c = s[p];
        if (c == '\n') { ++line; ++p; continue; }
        if (std::isspace((unsigned char)c)) { ++p; continue; }
        if (c == '#' || (c == '/' && p + 1 < s.size() && s[p + 1] == '/')) { while (p < s.size() && s[p] != '\n') ++p; continue; }
        if (std::isalpha((unsigned char)c) || c == '_') {
            std::size_t q = p; while (q < s.size() && (std::isalnum((unsigned char)s[q]) || s[q] == '_')) ++q;
            t.push_back({bp_token::k_ident, std::string(s.substr(p, q - p)), 0, line}); p = q; continue;
        }
        if (std::isdigit((unsigned char)c) || (c == '.' && p + 1 < s.size() && std::isdigit((unsigned char)s[p + 1]) && !(p > 0 && s[p - 1] == '.'))) {
            std::size_t q = p;
            while (q < s.size() && (std::isdigit((unsigned char)s[q]) || (s[q] == '.' && !(q + 1 < s.size() && s[q + 1] == '.')) ||
                                    ((s[q] == 'e' || s[q] == 'E') && q + 1 < s.size() && (std::isdigit((unsigned char)s[q + 1]) || s[q + 1] == '-' || s[q + 1] == '+')) ||
                                    ((s[q] == '-' || s[q] == '+') && (s[q - 1] == 'e' || s[q - 1] == 'E'))))
                ++q;
            auto txt = std::string(s.substr(p, q - p));
            char* end = nullptr; double v = std::strtod(txt.c_str(), &end);
            if (end != txt.c_str() + txt.size() || !std::isfinite(v)) raise(errc::parse_error, "blueprint:{}: bad number '{}'", line, txt);
            t.push_back({bp_token::k_num, txt, v, line}); p = q; continue;
        }
        if (c == '"') {
            std::size_t q = p + 1; while (q < s.size() && s[q] != '"') ++q;
            if (q >= s.size()) raise(errc::parse_error, "blueprint:{}: unterminated string", line);
            t.push_back({bp_token::k_str, std::string(s.substr(p + 1, q - p - 1)), 0, line}); p = q + 1; continue;
        }
        if (c == '.' && p + 1 < s.size() && s[p + 1] == '.') { t.push_back({bp_token::k_punct, "..", 0, line}); p += 2; continue; }
        if (std::string_view("{}()[],:=.+-*/").find(c) != std::string_view::npos) { t.push_back({bp_token::k_punct, std::string(1, c), 0, line}); ++p; continue; }
        raise(errc::parse_error, "blueprint:{}: unexpected character '{}'", line, c);
    }
    t.push_back({bp_token::k_end, "<eof>", 0, line});
    return t;
}

// Opsets this engine runs (FORGE_PLAN D.3). A blueprint declares the opset it was written for; build_graph refuses one
// outside the range and names it, and an op the engine does not know is reported with the blueprint's opset.
//   1  the first op set
//   2  rope frequency scaling attrs (scaling, factor, orig_ctx, low_freq, high_freq, beta_fast, beta_slow, truncate,
//      freq_factors). An opset-1 engine would ignore them and rotate by the wrong angles.
//   3  the spatial and encoder op families (conv, pooling, resize, group / batch norm, pad, attention_bias, ...),
//      composite-op defaults and keyword arguments, scalar() and the number helpers, and `session embedding`.
//      An older engine would stop at the first of these with an unknown-op or parse error.
//   4  session kinds chosen by the file with their parameters: `session diffusion_loop(steps: 1000, ...)`,
//      `session streaming(window: 400, hop: 160, ...)` (any `session kind(key: value, ...)`), and named entry points
//      `entry name { input ...  output ...  graph { ... } }` next to the main graph. An older engine stops at the
//      parenthesis or at `entry` with a parse error.
// A blueprint has to declare the opset of the newest thing it uses: the builder (and, for what the parser sees, the
// parser) refuses a use below its opset with an error that names the feature and the opset it needs. Otherwise a file
// that says "opset 1" could carry ops an opset-1 engine does not have, and the declaration would mean nothing.
inline constexpr int opset_min = 1, opset_max = 4;

// The opset that introduced a spec op (1 for the first op set). New ops get a line here with the opset that adds them.
inline int op_min_opset(std::string_view name) {
    static const std::set<std::string_view> opset3{
        "quick_gelu", "mish", "hard_swish", "floor", "attention_bias", "alibi_bias", "sinusoid_embed", "l2_normalize", "topk", "topk_indices",
        "index_select", "masked_mean", "conv1d", "conv2d", "conv_transpose2d", "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d", "global_avg_pool",
        "upsample_nearest", "upsample_bilinear", "group_norm", "batch_norm", "pad"};
    return opset3.count(name) ? 3 : 1;
}
// The opset that introduced an attr of an op that is older than the attr (1: the attr came with the op).
inline int attr_min_opset(std::string_view op, std::string_view attr) {
    static const std::set<std::string_view> rope2{"scaling", "factor", "orig_ctx", "low_freq", "high_freq", "beta_fast", "beta_slow", "truncate", "freq_factors"};
    return op == "rope" && rope2.count(attr) ? 2 : 1;
}

struct bp_state_decl { std::string name; i64 count = 1; std::string kind; std::map<std::string, double> args; };
struct bp_io_decl { std::string name, dtype; std::vector<std::string> dims; };

struct blueprint {
    std::string name;
    int opset = 1;
    // Raises when this blueprint declares less than opset `need`, which `feature` (as the message words it) requires.
    void need_opset(int need, int line, const std::string& feature) const {
        if (opset < need)
            raise(errc::parse_error, "blueprint:{}: {} needs opset {}, but blueprint \"{}\" declares opset {} (write `blueprint \"{}\" opset {}`)",
                  line, feature, need, name, opset, name, need);
    }
    std::map<std::string, double> params;          // dotted keys: "rope.base"
    std::map<std::string, std::string> str_params;
    std::vector<bp_state_decl> states;
    std::vector<bp_io_decl> inputs, outputs;
    std::string session = "one_shot";
    // Parameters of the session kind (opset 4): `session diffusion_loop(steps: 1000, schedule: scaled_linear)`.
    // Numbers (and true / false, and names of numeric params) land in session_args, names and strings in session_strs.
    std::map<std::string, double> session_args;
    std::map<std::string, std::string> session_strs;
    double session_arg(const std::string& k, double d) const { auto it = session_args.find(k); return it == session_args.end() ? d : it->second; }
    std::string session_str(const std::string& k, const std::string& d) const { auto it = session_strs.find(k); return it == session_strs.end() ? d : it->second; }
    std::vector<bp_token> toks;
    std::size_t graph_begin = 0, graph_end = 0;     // token range of the graph body
    // Named entry points (opset 4): further graphs of the same model, each with its own inputs and outputs, sharing
    // the params, composites, weights and state slots of the file. The top-level input / graph / output declarations
    // are the main entry. build_graph(bp, options, "name") builds one.
    struct entry { std::vector<bp_io_decl> inputs, outputs; std::size_t graph_begin = 0, graph_end = 0; };
    std::map<std::string, entry> entries;
    const entry& entry_at(const std::string& n) const {
        auto it = entries.find(n);
        if (it == entries.end()) raise(errc::not_found, "blueprint '{}': no entry '{}'", name, n);
        return it->second;
    }
    // Composite ops: `op name(a, b) { t = ...  return expr }`. A spec op of the same name wins when the
    // engine has one, so files keep working on old engines and pick up fused kernels on new ones.
    // Parameters may carry a numeric default (`op f(x, eps = 1e-6)`) and be passed by name (`f(x, eps: 1e-5)`).
    // Composites are inlined when the graph is built; a composite that reaches itself (directly or through other
    // composites) is rejected by parse_blueprint.
    struct composite { std::vector<std::string> args; std::size_t body_begin = 0, body_end = 0; std::map<std::string, double> defaults; };
    std::map<std::string, composite> composites;
    double param(const std::string& k) const {
        auto it = params.find(k);
        if (it == params.end()) raise(errc::not_found, "blueprint '{}': no param '{}'", name, k);
        return it->second;
    }
    double param_or(const std::string& k, double d) const { auto it = params.find(k); return it == params.end() ? d : it->second; }
};

struct build_options {
    std::map<std::string, i64> dims;  // symbolic dims, e.g. {"seq", 7}
    bool all_rows = false;            // `last(x, n)` keeps every row (speculative verification, scoring)
    // resolves a weight name (e.g. "layer.0.attn_norm") to shape + f32 data
    std::function<std::pair<shape_t, const float*>(const std::string&)> weight;
    // optional: resolves a weight to its encoded bytes + codec name instead (null data = use `weight`)
    struct encoded { shape_t shape; const std::uint8_t* data = nullptr; std::string codec; };
    std::function<encoded(const std::string&)> qweight;
};

namespace detail {

struct bp_val {
    enum kind_t { none, num, list, tensor, tuple, wpath, state } k = none;
    double n = 0;
    std::vector<double> l;
    graph::value_id v = graph::no_value;
    std::vector<graph::value_id> tup;
    std::string path;  // weight path or state name
};

class bp_parser {
public:
    bp_parser(const std::vector<bp_token>& t, std::size_t p = 0) : t_(t), p_(p) {}
    const bp_token& cur() const { return t_[p_]; }
    const bp_token& next() { const auto& t = t_[p_]; if (t.kind != bp_token::k_end) ++p_; return t; }
    double number() { if (cur().kind != bp_token::k_num) fail("expected number"); return next().num; }
    bool is(std::string_view s) const { return (cur().kind == bp_token::k_punct || cur().kind == bp_token::k_ident) && cur().text == s; }
    bool accept(std::string_view s) { if (is(s)) { ++p_; return true; } return false; }
    [[noreturn]] void fail(const std::string& m) const { raise(errc::parse_error, "blueprint:{}: {} (near '{}')", cur().line, m, cur().text); }
    void expect(std::string_view s) { if (!accept(s)) fail(std::format("expected '{}'", s)); }
    std::string ident() { if (cur().kind != bp_token::k_ident) fail("expected identifier"); return next().text; }
    std::size_t pos() const { return p_; }
    void seek(std::size_t p) { p_ = p; }
    void skip_block() {  // assumes current token is '{'
        int depth = 0;
        do { if (is("{")) ++depth; else if (is("}")) --depth; else if (cur().kind == bp_token::k_end) fail("unbalanced braces"); ++p_; } while (depth > 0);
    }
private:
    const std::vector<bp_token>& t_;
    std::size_t p_;
};

class bp_builder {
public:
    // `e`: a named entry point to build instead of the main graph
    bp_builder(const blueprint& bp, const build_options& o, graph::tensor_graph& g, const blueprint::entry* e = nullptr)
        : bp_(bp), o_(o), g_(g), ps_(bp.toks, e ? e->graph_begin : bp.graph_begin), inputs_(e ? e->inputs : bp.inputs), outputs_(e ? e->outputs : bp.outputs),
          graph_end_(e ? e->graph_end : bp.graph_end) {}

    void run() {
        for (auto& in : inputs_) {
            shape_t s; for (auto& d : in.dims) s.push_back(dim(d));
            bp_val v; v.k = bp_val::tensor; v.v = g_.input(in.name, s); env_[in.name] = v;
        }
        for (auto& st : bp_.states) { bp_val v; v.k = bp_val::state; v.path = st.name; env_[st.name] = v; }
        while (ps_.pos() < graph_end_) stmt();
        for (auto& out : outputs_) {
            auto it = env_.find(out.name);
            if (it == env_.end() || it->second.k != bp_val::tensor) raise(errc::invalid_argument, "blueprint: output '{}' never assigned", out.name);
            g_.output(out.name, it->second.v);
        }
    }

private:
    i64 dim(const std::string& d) {
        if (auto it = o_.dims.find(d); it != o_.dims.end()) return it->second;
        if (auto it = bp_.params.find(d); it != bp_.params.end()) return i64(it->second);
        if (!d.empty() && std::isdigit((unsigned char)d[0])) {
            i64 v = 0; auto [e, ec] = std::from_chars(d.data(), d.data() + d.size(), v);
            if (ec == std::errc() && e == d.data() + d.size()) return v;
        }
        raise(errc::invalid_argument, "blueprint: unbound dimension '{}'", d);
    }
    void stmt() {
        if (ps_.accept("repeat")) {
            auto var = ps_.ident(); ps_.expect("in");
            auto lo = expect_num(expr()); ps_.expect(".."); auto hi = expect_num(expr());
            if (!ps_.is("{")) ps_.fail("expected '{' after repeat range");
            std::size_t body = ps_.pos();
            ps_.skip_block();
            std::size_t after = ps_.pos();
            auto saved = env_.count(var) ? std::optional<bp_val>(env_[var]) : std::nullopt;
            for (i64 i = i64(lo); i < i64(hi); ++i) {
                bp_val iv; iv.k = bp_val::num; iv.n = double(i); env_[var] = iv;
                ps_.seek(body + 1);
                while (!ps_.is("}")) stmt();
            }
            if (saved) env_[var] = *saved; else env_.erase(var);
            ps_.seek(after);
            return;
        }
        std::vector<std::string> names{ps_.ident()};
        while (ps_.accept(",")) names.push_back(ps_.ident());
        ps_.expect("=");
        auto v = expr();
        if (names.size() == 1) { env_[names[0]] = v; return; }
        if (v.k != bp_val::tuple || v.tup.size() != names.size())
            ps_.fail(std::format("cannot unpack into {} names", names.size()));
        for (std::size_t i = 0; i < names.size(); ++i) { bp_val t; t.k = bp_val::tensor; t.v = v.tup[i]; env_[names[i]] = t; }
    }
    double expect_num(const bp_val& v) { if (v.k != bp_val::num) ps_.fail("expected a number"); return v.n; }
    graph::value_id as_tensor(const bp_val& v) {
        if (v.k == bp_val::tensor) return v.v;
        if (v.k == bp_val::num) return g_.scalar(float(v.n));
        if (v.k == bp_val::wpath) return weight(v.path);
        ps_.fail("expected a tensor");
    }
    graph::value_id weight(const std::string& name) {
        if (auto it = wcache_.find(name); it != wcache_.end()) return it->second;
        if (!o_.weight) raise(errc::invalid_argument, "blueprint: no weight resolver for '{}'", name);
        if (o_.qweight) if (auto e = o_.qweight(name); e.data) return wcache_[name] = g_.qweight(name, e.shape, e.data, e.codec);
        auto [s, p] = o_.weight(name);
        return wcache_[name] = g_.weight(name, s, p);
    }
    bp_val binop(char op, bp_val a, bp_val b) {
        if (a.k == bp_val::num && b.k == bp_val::num) {
            bp_val r; r.k = bp_val::num;
            r.n = op == '+' ? a.n + b.n : op == '-' ? a.n - b.n : op == '*' ? a.n * b.n : a.n / b.n; return r;
        }
        if (a.k == bp_val::list && b.k == bp_val::num && op == '*') { for (auto& x : a.l) x *= b.n; return a; }
        auto x = as_tensor(a), y = as_tensor(b);
        ops::op o = op == '+' ? ops::op::add : op == '-' ? ops::op::sub : op == '*' ? ops::op::mul : ops::op::div;
        bp_val r; r.k = bp_val::tensor; r.v = g_.apply(o, {x, y}); return r;
    }
    bp_val expr() {
        auto a = term();
        while (ps_.is("+") || ps_.is("-")) { char op = ps_.next().text[0]; a = binop(op, a, term()); }
        return a;
    }
    bp_val term() {
        auto a = unary();
        while (ps_.is("*") || ps_.is("/")) { char op = ps_.next().text[0]; a = binop(op, a, unary()); }
        return a;
    }
    bp_val unary() {
        if (ps_.accept("-")) {
            auto v = unary();
            if (v.k == bp_val::num) { v.n = -v.n; return v; }
            bp_val r; r.k = bp_val::tensor; r.v = g_.apply(ops::op::neg, {as_tensor(v)}); return r;
        }
        return postfix(primary());
    }
    bp_val postfix(bp_val v) {
        while (true) {
            if (ps_.is(".") ) {
                ps_.next(); auto m = ps_.ident();
                if (v.k == bp_val::wpath) { v.path += (v.path.empty() ? "" : ".") + m; continue; }
                if (v.k == bp_val::state && m == "append") {
                    ps_.expect("(");
                    std::vector<graph::value_id> xs;
                    if (!ps_.is(")")) { xs.push_back(as_tensor(expr())); while (ps_.accept(",")) xs.push_back(as_tensor(expr())); }
                    ps_.expect(")");
                    static const char* sfx[] = {"k", "v", "s2", "s3"};
                    bp_val r; r.k = xs.size() == 1 ? bp_val::tensor : bp_val::tuple;
                    for (std::size_t i = 0; i < xs.size(); ++i) {
                        auto id = g_.state_append(v.path + (xs.size() == 1 ? "" : std::string(".") + sfx[i]), xs[i]);
                        if (xs.size() == 1) r.v = id; else r.tup.push_back(id);
                    }
                    v = r; continue;
                }
                if (v.k == bp_val::state && (m == "conv" || m == "delta")) {   // fixed-size state ops (state_ops.hpp)
                    ps_.expect("(");
                    std::vector<graph::value_id> xs;
                    if (!ps_.is(")")) { xs.push_back(as_tensor(expr())); while (ps_.accept(",")) xs.push_back(as_tensor(expr())); }
                    ps_.expect(")");
                    if (xs.size() != (m == "conv" ? 2u : 5u)) ps_.fail(m == "conv" ? "conv(x, w) takes 2 tensors" : "delta(q, k, v, g, beta) takes 5 tensors");
                    bp_val r; r.k = bp_val::tensor; r.v = g_.state_op(v.path, m == "conv" ? "causal_conv" : "delta_rule", xs);
                    v = r; continue;
                }
                ps_.fail("unknown member '" + m + "'");
            }
            if (ps_.is("[") && (v.k == bp_val::wpath || v.k == bp_val::state)) {
                ps_.next(); auto i = expect_num(expr()); ps_.expect("]");
                v.path += "." + std::to_string(i64(i)); continue;
            }
            return v;
        }
    }
    bp_val primary() {
        const auto& t = ps_.cur();
        if (t.kind == bp_token::k_num) { ps_.next(); bp_val v; v.k = bp_val::num; v.n = t.num; return v; }
        if (ps_.accept("(")) { auto v = expr(); ps_.expect(")"); return v; }
        if (ps_.accept("[")) {
            bp_val v; v.k = bp_val::list;
            if (!ps_.is("]")) { v.l.push_back(expect_num(expr())); while (ps_.accept(",")) v.l.push_back(expect_num(expr())); }
            ps_.expect("]"); return v;
        }
        auto id = ps_.ident();
        if (id == "true" || id == "false") { bp_val v; v.k = bp_val::num; v.n = id == "true"; return v; }
        if (id == "W") { bp_val v; v.k = bp_val::wpath; return v; }
        if (ps_.is("(")) return call(id);
        if (auto it = env_.find(id); it != env_.end()) return it->second;
        // dotted params: rope.base
        std::string key = id;
        while (ps_.is(".") && !bp_.params.count(key)) {
            auto save = ps_.pos(); ps_.next();
            if (ps_.cur().kind != bp_token::k_ident) { ps_.seek(save); break; }
            key += "." + ps_.ident();
        }
        if (auto it = bp_.params.find(key); it != bp_.params.end()) { bp_val v; v.k = bp_val::num; v.n = it->second; return v; }
        if (auto it = o_.dims.find(key); it != o_.dims.end()) { bp_val v; v.k = bp_val::num; v.n = double(it->second); return v; }
        ps_.fail("undefined name '" + key + "'");
    }
    bp_val call(const std::string& fn) {
        int line = ps_.cur().line;
        ps_.expect("(");
        // positional arguments; at() reports a missing one as a blueprint error (a std::out_of_range escaped before)
        struct arg_list : std::vector<bp_val> {
            std::string fn; int line = 0;
            const bp_val& at(std::size_t i) const { if (i >= size()) raise(errc::parse_error, "blueprint line {}: too few arguments for '{}'", line, fn); return (*this)[i]; }
        } pos;
        pos.fn = fn; pos.line = line;
        std::vector<std::pair<std::string, bp_val>> kw;
        while (!ps_.is(")")) {
            // (composite bodies may sit after the graph block, so the bound is the token list, not graph_end)
            if (ps_.cur().kind == bp_token::k_ident && ps_.pos() + 1 < bp_.toks.size() && bp_.toks[ps_.pos() + 1].kind == bp_token::k_punct && bp_.toks[ps_.pos() + 1].text == ":") {
                auto k = ps_.ident(); ps_.expect(":"); kw.emplace_back(k, expr());
            } else pos.push_back(expr());
            if (!ps_.accept(",")) break;
        }
        ps_.expect(")");
        auto T = [&](graph::value_id id) { bp_val r; r.k = bp_val::tensor; r.v = id; return r; };
        // built-in composites
        if (fn == "linear") return T(g_.apply(ops::op::matmul_t, {as_tensor(pos.at(0)), as_tensor(pos.at(1))}));
        if (fn == "embed") return T(g_.apply(ops::op::gather, {as_tensor(pos.at(0)), as_tensor(pos.at(1))}));
        if (fn == "heads") return T(g_.apply(ops::op::reshape, {as_tensor(pos.at(0))},
                                             {{"shape", std::vector<i64>{-1, i64(expect_num(pos.at(1))), i64(expect_num(pos.at(2)))}}}));
        if (fn == "merge") return T(g_.apply(ops::op::reshape, {as_tensor(pos.at(0))}, {{"shape", std::vector<i64>{-1, i64(expect_num(pos.at(1)))}}}));
        if (fn == "last") {  // last row along axis 0 (needs the symbolic length)
            i64 n = i64(expect_num(pos.at(1)));
            if (o_.all_rows) return pos.at(0);
            return T(g_.apply(ops::op::slice, {as_tensor(pos.at(0))}, {{"axis", 0.0}, {"start", double(n - 1)}, {"end", double(n)}}));
        }
        if (fn == "scalar") {   // a number as a [1] constant tensor (e.g. the zero bias of attention_bias)
            bp_.need_opset(3, line, "scalar()");
            if (pos.size() != 1 || !kw.empty()) ps_.fail("scalar(x) takes one number");
            return T(g_.scalar(float(expect_num(pos[0]))));
        }
        // number helpers, folded while the graph is built (per-layer attrs: `window * min(1, mod(l + 1, pattern))`)
        // (opset 3. Below it a call on numbers means what it meant before the helpers existed: the spec op of that
        // name on a constant where there is one in the blueprint's opset -- sqrt -- and an error otherwise.)
        const bool helper_name = fn == "sqrt" || fn == "floor" || fn == "ceil" || fn == "min" || fn == "max" || fn == "mod";
        const bool all_numbers = kw.empty() && !pos.empty() && std::all_of(pos.begin(), pos.end(), [](const bp_val& v) { return v.k == bp_val::num; });
        if (helper_name && all_numbers && bp_.opset < 3 && !(ops::op_from_name(fn) && op_min_opset(fn) <= bp_.opset))
            bp_.need_opset(3, line, "the number helper " + fn + "()");
        if (all_numbers && bp_.opset >= 3) {
            auto N = [](double x) { bp_val r; r.k = bp_val::num; r.n = x; return r; };
            if (pos.size() == 1 && fn == "sqrt") return N(std::sqrt(pos[0].n));
            if (pos.size() == 1 && fn == "floor") return N(std::floor(pos[0].n));
            if (pos.size() == 1 && fn == "ceil") return N(std::ceil(pos[0].n));
            if (pos.size() == 2 && fn == "min") return N(std::min(pos[0].n, pos[1].n));
            if (pos.size() == 2 && fn == "max") return N(std::max(pos[0].n, pos[1].n));
            if (pos.size() == 2 && fn == "mod") {
                if (pos[1].n == 0) ps_.fail("mod by zero");
                return N(pos[0].n - std::floor(pos[0].n / pos[1].n) * pos[1].n);
            }
        }
        auto o = ops::op_from_name(fn);
        if (!o) {
            auto cit = bp_.composites.find(fn);
            if (cit == bp_.composites.end())
                raise(errc::parse_error, "blueprint:{}: unknown op '{}' (blueprint \"{}\" declares opset {}; this engine supports opsets {}..{})",
                      line, fn, bp_.name, bp_.opset, opset_min, opset_max);
            return expand_composite(fn, cit->second, pos, kw, line);
        }
        bp_.need_opset(op_min_opset(fn), line, "op '" + fn + "'");
        const auto& inf = ops::info(*o);
        std::vector<std::string> anames;
        for (std::size_t s = 0, e; s < inf.attrs.size(); s = e + 1) {
            e = inf.attrs.find(',', s); if (e == std::string_view::npos) e = inf.attrs.size();
            anames.emplace_back(inf.attrs.substr(s, e - s));
        }
        std::vector<graph::value_id> ins; ops::attrs a; std::size_t ai = 0;
        for (auto& p : pos) {
            if (int(ins.size()) < inf.inputs && (p.k == bp_val::tensor || p.k == bp_val::wpath || (p.k == bp_val::num && ai == 0 && anames.empty())))
                ins.push_back(as_tensor(p));
            else {
                if (ai >= anames.size()) ps_.fail(std::format("too many arguments to {}", fn));
                if (p.k == bp_val::list) { std::vector<i64> l; for (auto x : p.l) l.push_back(i64(x)); a.set(anames[ai++], l); }
                else a.set(anames[ai++], expect_num(p));
            }
        }
        for (auto& [k, v] : kw) {
            if (v.k == bp_val::list) { std::vector<i64> l; for (auto x : v.l) l.push_back(i64(x)); a.set(k, l); }
            else a.set(k, expect_num(v));
        }
        if (int(ins.size()) != inf.inputs) ps_.fail(std::format("{} expects {} tensor inputs, got {}", fn, inf.inputs, ins.size()));
        for (auto& [k, v] : a.v) bp_.need_opset(attr_min_opset(fn, k), line, fn == "rope" ? "RoPE frequency scaling (rope attr '" + k + "')" : fn + " attr '" + k + "'");
        return T(g_.apply(*o, ins, a));
    }

    // Inline a composite at the call site. Arguments bind by position, then by name (`f(x, eps: 1e-5)`); a parameter
    // left unbound takes its declared default (`op f(x, eps = 1e-6)`), else the call is an arity error.
    bp_val expand_composite(const std::string& fn, const blueprint::composite& c, const std::vector<bp_val>& pos,
                            const std::vector<std::pair<std::string, bp_val>>& kw, int line) {
        if (pos.size() > c.args.size()) raise(errc::parse_error, "blueprint:{}: {} expects {} arguments, got {}", line, fn, c.args.size(), pos.size());
        if (!kw.empty()) bp_.need_opset(3, line, "a keyword argument to the composite op '" + fn + "' ('" + kw.front().first + ":')");
        std::map<std::string, bp_val> bound;
        for (std::size_t i = 0; i < pos.size(); ++i) bound[c.args[i]] = pos[i];
        for (auto& [k, v] : kw) {
            if (std::find(c.args.begin(), c.args.end(), k) == c.args.end()) raise(errc::parse_error, "blueprint:{}: {} has no parameter '{}'", line, fn, k);
            if (!bound.emplace(k, v).second) raise(errc::parse_error, "blueprint:{}: {}: parameter '{}' given twice", line, fn, k);
        }
        for (auto& an : c.args) {
            if (bound.count(an)) continue;
            auto d = c.defaults.find(an);
            if (d == c.defaults.end()) raise(errc::parse_error, "blueprint:{}: {} expects {} arguments, got {} (missing '{}')", line, fn, c.args.size(), pos.size() + kw.size(), an);
            bp_val v; v.k = bp_val::num; v.n = d->second; bound[an] = v;
        }
        if (++expansions_ > 100000) raise(errc::parse_error, "blueprint:{}: too many composite expansions", line);
        if (++depth_ > 64) raise(errc::parse_error, "blueprint:{}: composite '{}' recursion too deep", line, fn);
        auto saved_env = env_; auto back = ps_.pos();
        for (auto& [k, v] : bound) env_[k] = v;   // args shadow outer names
        ps_.seek(c.body_begin);
        bp_val r; bool got = false;
        while (ps_.pos() < c.body_end) {
            if (ps_.accept("return")) { r = expr(); got = true; break; }
            stmt();
        }
        if (!got) raise(errc::parse_error, "blueprint:{}: composite '{}' has no return", line, fn);
        env_ = std::move(saved_env); ps_.seek(back); --depth_;
        return r;
    }

    int depth_ = 0; std::size_t expansions_ = 0;
    const blueprint& bp_;
    const build_options& o_;
    graph::tensor_graph& g_;
    bp_parser ps_;
    const std::vector<bp_io_decl>& inputs_;
    const std::vector<bp_io_decl>& outputs_;
    std::size_t graph_end_;
    std::map<std::string, bp_val> env_;
    std::map<std::string, graph::value_id> wcache_;
};

inline void parse_params(bp_parser& ps, blueprint& bp, const std::string& prefix) {
    ps.expect("{");
    while (!ps.accept("}")) {
        auto k = prefix + ps.ident(); ps.expect(":");
        if (ps.is("{")) { parse_params(ps, bp, k + "."); continue; }
        bool neg = ps.accept("-");
        auto& t = ps.cur();
        if (t.kind == bp_token::k_num) { bp.params[k] = neg ? -t.num : t.num; ps.next(); }
        else if (t.kind == bp_token::k_str) { bp.str_params[k] = t.text; ps.next(); }
        else if (t.kind == bp_token::k_ident) {
            auto v = ps.ident();
            if (v == "true" || v == "false") { bp.params[k] = v == "true"; }
            else if (ps.is("(")) {  // e.g. yarn(factor: 8, orig_ctx: 8192) -> k = "yarn", k.factor = 8
                bp.str_params[k] = v; ps.next();
                while (!ps.accept(")")) { auto a = ps.ident(); ps.expect(":"); bp.params[k + "." + a] = ps.number(); ps.accept(","); }
            } else bp.str_params[k] = v;
        } else ps.fail("bad param value");
        ps.accept(",");
    }
}

inline bp_io_decl parse_io(bp_parser& ps) {
    bp_io_decl d; d.name = ps.ident(); ps.expect(":"); d.dtype = ps.ident(); ps.expect("[");
    while (!ps.accept("]")) {
        if (ps.cur().kind == bp_token::k_num) d.dims.push_back(std::to_string(i64(ps.next().num)));
        else d.dims.push_back(ps.ident());
        ps.accept(",");
    }
    return d;
}

} // namespace detail

inline blueprint parse_blueprint(std::string_view src) {
    blueprint bp; bp.toks = bp_lex(src);
    detail::bp_parser ps(bp.toks);
    ps.expect("blueprint");
    if (ps.cur().kind != bp_token::k_str) ps.fail("expected blueprint name string");
    bp.name = ps.next().text;
    if (ps.accept("opset")) {
        double v = ps.number();
        if (!(v >= 0 && v <= 1e6) || v != std::floor(v)) raise(errc::parse_error, "blueprint '{}': bad opset {}", bp.name, v);
        bp.opset = int(v);
    }
    bool have_graph = false;
    while (ps.cur().kind != bp_token::k_end) {
        if (ps.accept("params")) detail::parse_params(ps, bp, "");
        else if (ps.accept("state")) {
            bp_state_decl s; s.name = ps.ident();
            std::string count_name;
            if (ps.accept("[")) {
                if (ps.cur().kind == bp_token::k_num) s.count = i64(ps.next().num);
                else { auto n = ps.ident(); s.count = i64(bp.param(n)); }
                ps.expect("]");
            }
            ps.expect(":"); s.kind = ps.ident();
            if (ps.accept("(")) {
                while (!ps.accept(")")) {
                    auto a = ps.ident(); ps.expect(":");
                    if (ps.cur().kind == bp_token::k_num) s.args[a] = ps.next().num;
                    else { auto v = ps.ident(); if (bp.params.count(v)) s.args[a] = bp.params[v]; }
                    ps.accept(",");
                }
            }
            bp.states.push_back(s);
        } else if (ps.accept("input")) bp.inputs.push_back(detail::parse_io(ps));
        else if (ps.accept("output")) bp.outputs.push_back(detail::parse_io(ps));
        else if (ps.accept("session")) {
            int line = ps.cur().line;
            bp.session = ps.ident();
            if (bp.session == "embedding") bp.need_opset(3, line, "`session embedding`");
            if (bp.session == "diffusion_loop" || bp.session == "streaming") bp.need_opset(4, line, "`session " + bp.session + "`");
            if (ps.is("(")) {   // session parameters
                bp.need_opset(4, line, "parameters of a session kind (`session " + bp.session + "(...)`)");
                ps.next();
                while (!ps.accept(")")) {
                    auto k = ps.ident(); ps.expect(":");
                    bool neg = ps.accept("-");
                    const auto& t = ps.cur();
                    if (t.kind == bp_token::k_num) { bp.session_args[k] = neg ? -t.num : t.num; ps.next(); }
                    else if (t.kind == bp_token::k_str && !neg) { bp.session_strs[k] = t.text; ps.next(); }
                    else if (t.kind == bp_token::k_ident && !neg) {
                        auto v = ps.ident();
                        if (v == "true" || v == "false") bp.session_args[k] = v == "true";
                        else if (auto it = bp.params.find(v); it != bp.params.end()) bp.session_args[k] = it->second;   // a numeric param by name
                        else bp.session_strs[k] = v;
                    } else ps.fail("bad session parameter value");
                    ps.accept(",");
                }
            }
        }
        else if (ps.accept("entry")) {
            int line = ps.cur().line;
            auto nm = ps.ident();
            bp.need_opset(4, line, "the named entry point '" + nm + "'");
            blueprint::entry e; bool have = false;
            ps.expect("{");
            while (!ps.accept("}")) {
                if (ps.accept("input")) e.inputs.push_back(detail::parse_io(ps));
                else if (ps.accept("output")) e.outputs.push_back(detail::parse_io(ps));
                else if (ps.accept("graph")) {
                    if (have) ps.fail("entry '" + nm + "' has two graph blocks");
                    if (!ps.is("{")) ps.fail("expected '{'");
                    e.graph_begin = ps.pos() + 1; ps.skip_block(); e.graph_end = ps.pos() - 1; have = true;
                } else ps.fail("expected input/output/graph in entry '" + nm + "'");
            }
            if (!have) ps.fail("entry '" + nm + "' has no graph block");
            if (e.outputs.empty()) ps.fail("entry '" + nm + "' declares no outputs");
            if (nm == "main" || !bp.entries.emplace(nm, std::move(e)).second) ps.fail("entry '" + nm + "' defined twice (main is the top-level graph)");
        }
        else if (ps.accept("op")) {
            auto nm = ps.ident(); blueprint::composite c; ps.expect("(");
            while (!ps.accept(")")) {
                auto an = ps.ident();
                if (std::find(c.args.begin(), c.args.end(), an) != c.args.end()) ps.fail("op '" + nm + "': parameter '" + an + "' repeated");
                c.args.push_back(an);
                if (ps.is("=")) bp.need_opset(3, ps.cur().line, "a default value for parameter '" + an + "' of the composite op '" + nm + "'");
                if (ps.accept("=")) { bool neg = ps.accept("-"); double v = ps.number(); c.defaults[an] = neg ? -v : v; }
                else if (!c.defaults.empty()) ps.fail("op '" + nm + "': parameter '" + an + "' without a default follows one with a default");
                ps.accept(",");
            }
            if (!ps.is("{")) ps.fail("expected '{' after op signature");
            c.body_begin = ps.pos() + 1; ps.skip_block(); c.body_end = ps.pos() - 1;
            if (!bp.composites.emplace(nm, c).second) ps.fail("op '" + nm + "' defined twice");
        }
        else if (ps.accept("graph")) {
            if (!ps.is("{")) ps.fail("expected '{'");
            bp.graph_begin = ps.pos() + 1;
            ps.skip_block();
            bp.graph_end = ps.pos() - 1;
            have_graph = true;
        } else ps.fail("expected params/state/input/graph/output/session/op/entry");
    }
    // Recursion is forbidden: composites are inlined, so a cycle could never terminate. Calls are `name (` tokens in
    // a body that resolve to a composite (a built-in or spec op of the same name wins, exactly as in the builder).
    {
        auto callees = [&](const blueprint::composite& c) {
            std::vector<std::string> out;
            for (std::size_t i = c.body_begin; i + 1 < c.body_end; ++i) {
                const auto& t = bp.toks[i];
                if (t.kind != bp_token::k_ident || bp.toks[i + 1].kind != bp_token::k_punct || bp.toks[i + 1].text != "(") continue;
                if (i > 0 && bp.toks[i - 1].kind == bp_token::k_punct && bp.toks[i - 1].text == ".") continue;   // member call (state.append)
                if (t.text == "linear" || t.text == "embed" || t.text == "heads" || t.text == "merge" || t.text == "last" || t.text == "scalar") continue;
                if (bp.composites.count(t.text) && !ops::op_from_name(t.text)) out.push_back(t.text);
            }
            return out;
        };
        std::map<std::string, int> mark;   // 1 = on the current path, 2 = done
        std::vector<std::string> path;
        std::function<void(const std::string&)> visit = [&](const std::string& n) {
            if (mark[n] == 2) return;
            if (mark[n] == 1) {
                std::string chain; bool on = false;
                for (auto& p : path) { on = on || p == n; if (on) chain += p + " -> "; }
                raise(errc::parse_error, "blueprint '{}': composite op '{}' is recursive ({}{})", bp.name, n, chain, n);
            }
            mark[n] = 1; path.push_back(n);
            for (auto& m : callees(bp.composites.at(n))) visit(m);
            path.pop_back(); mark[n] = 2;
        };
        for (auto& [n, c] : bp.composites) visit(n);
    }
    if (!have_graph) raise(errc::parse_error, "blueprint '{}': no graph block", bp.name);
    if (bp.outputs.empty()) raise(errc::parse_error, "blueprint '{}': no outputs declared", bp.name);
    return bp;
}

// Refuses a blueprint whose opset this engine does not run (unsupported; the message names the opset and the range).
inline void check_opset(const blueprint& bp) {
    if (bp.opset < opset_min || bp.opset > opset_max)
        raise(errc::unsupported, "blueprint \"{}\" declares opset {}; this engine supports opsets {}..{}{}", bp.name, bp.opset, opset_min, opset_max,
              bp.opset > opset_max ? " (it needs a newer FORGE)" : "");
}

inline graph::tensor_graph build_graph(const blueprint& bp, const build_options& o) {
    check_opset(bp);
    graph::tensor_graph g;
    detail::bp_builder b(bp, o, g);
    b.run();
    return g;
}
// The graph of a named entry point ("" or "main": the main graph).
inline graph::tensor_graph build_graph(const blueprint& bp, const build_options& o, const std::string& entry) {
    if (entry.empty() || entry == "main") return build_graph(bp, o);
    check_opset(bp);
    graph::tensor_graph g;
    detail::bp_builder b(bp, o, g, &bp.entry_at(entry));
    b.run();
    return g;
}

// The names of every weight the graph references (W.<path>), in first-use order, without touching any tensor data:
// the graph is built with one-element placeholder shapes. Importers compare this list with the tensors they wrote.
inline std::vector<std::string> blueprint_weights(const blueprint& bp) {
    std::vector<std::string> names;
    build_options o;
    auto bind = [&](const std::vector<bp_io_decl>& ins) { for (auto& in : ins) for (auto& d : in.dims) if (!bp.params.count(d) && !(!d.empty() && std::isdigit((unsigned char)d[0]))) o.dims[d] = 2; };   // symbolic lengths
    bind(bp.inputs);
    for (auto& [en, e] : bp.entries) bind(e.inputs);
    std::set<std::string> seen;
    o.weight = [&](const std::string& n) { if (seen.insert(n).second) names.push_back(n); return std::pair<shape_t, const float*>(shape_t{1}, nullptr); };
    (void)build_graph(bp, o);
    for (auto& [en, e] : bp.entries) (void)build_graph(bp, o, en);   // entry points read weights too
    return names;
}

} // namespace forge::modelfile
