// forge::script AOT, native mode: translate a parsed and resolved program to C++23 (docs/aot.md).
//
// The translation follows the bytecode compiler (vm.cpp) instruction for instruction, and for the statements the
// VM hands to the tree-walker (try, throw, class, range-for by reference, compound index assignment) it follows
// Interp::exec (tree.cpp). Each VM instruction becomes one call of the same Interp operation the VM calls, with
// operands held in C++ locals that live exactly as long as the VM keeps them on its stack: reference counts are
// observable (struct copy elision in Interp::share, shared_ptr::use_count), so temporaries are scoped to match.
// Node kinds the translator does not know are handed to the tree-walker at run time (Interp::exec / eval on the
// node), never guessed at. Output is deterministic: names are numbered in program order and nothing depends on
// the time, the host or the address of anything.
#include <forge/script/aot.hpp>
#include <forge/script/check.hpp>
#include "aot.hpp"
#include "parser.hpp"
#include <forge/os/os.hpp>
#include <forge/os/process.hpp>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <set>

namespace forge::script {
namespace aot {

// ---------------------------------------------------------------- numbering
namespace {
struct Indexer {
    program_index& ix;
    void node(const Node& n) {
        if (!ix.node_ids.emplace(&n, int(ix.nodes.size())).second) return;
        ix.nodes.push_back(&n);
        for (auto& k : n.kids) if (k) node(*k);
        if (n.extra) node(*n.extra);
        if (n.fn) fn(*n.fn);
        if (n.cls) { for (auto& [nm, d] : n.cls->fields) if (d) node(*d); for (auto& m : n.cls->methods) if (m) fn(*m); }
    }
    void fn(const FnDecl& f) {
        if (!ix.fn_ids.emplace(&f, int(ix.fns.size())).second) return;
        ix.fns.push_back(&f);
        for (auto& p : f.params) if (p.dflt) node(*p.dflt);
        if (f.body) node(*f.body);
    }
};
struct Fnv {
    std::uint64_t h = 1469598103934665603ull;
    void bytes(const void* p, std::size_t n) { auto* b = static_cast<const unsigned char*>(p); for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; } }
    void u(std::uint64_t v) { unsigned char b[8]; for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>(v >> (8 * i)); bytes(b, 8); }
    void s(const std::string& v) { u(v.size()); bytes(v.data(), v.size()); }
};
}

int program_index::node_of(const Node& n) const { auto it = node_ids.find(&n); return it == node_ids.end() ? -1 : it->second; }
int program_index::fn_of(const FnDecl& f) const { auto it = fn_ids.find(&f); return it == fn_ids.end() ? -1 : it->second; }

program_index index_program(const FnDecl& top) {
    program_index ix;
    Indexer{ix}.fn(top);
    Fnv h;
    h.u(abi_version); h.u(ix.nodes.size()); h.u(ix.fns.size());
    auto nid = [&](const NodeP& p) { h.u(p ? std::uint64_t(ix.node_of(*p)) + 1 : 0); };
    for (const Node* n : ix.nodes) {
        h.u(std::uint64_t(n->k)); h.u(std::uint64_t(std::int64_t(n->line))); h.s(n->s); h.u(std::uint64_t(n->i));
        std::uint64_t db; std::memcpy(&db, &n->d, 8); h.u(db);
        h.u((n->flag ? 1u : 0u) | (n->ref ? 2u : 0u) | (n->wb ? 4u : 0u));
        h.s(n->var.name); h.u(std::uint64_t(std::int64_t(n->var.depth))); h.u(std::uint64_t(std::int64_t(n->var.slot)));
        h.s(n->type ? n->type->str() : std::string("\x01"));
        h.u(n->kids.size()); for (auto& k : n->kids) nid(k);
        nid(n->extra);
        h.u(n->fn ? std::uint64_t(ix.fn_of(*n->fn)) + 1 : 0);
        if (n->cls) {
            h.s(n->cls->name); h.s(n->cls->base); h.u(n->cls->fields.size());
            for (auto& [nm, d] : n->cls->fields) { h.s(nm); nid(d); }
            h.u(n->cls->methods.size()); for (auto& m : n->cls->methods) h.u(m ? std::uint64_t(ix.fn_of(*m)) + 1 : 0);
        } else h.u(0xC0FFEEull);
        h.u(n->locals.size()); for (int l : n->locals) h.u(std::uint64_t(std::int64_t(l)));
    }
    for (const FnDecl* f : ix.fns) {
        h.s(f->name); h.u(std::uint64_t(std::int64_t(f->line))); h.u(std::uint64_t(std::int64_t(f->nslots)));
        h.u((f->is_method ? 1u : 0u) | (f->has_out ? 2u : 0u) | (f->is_top ? 4u : 0u) | (f->has_main ? 8u : 0u));
        h.u(f->params.size());
        for (auto& p : f->params) { h.s(p.name); h.u(std::uint64_t(std::int64_t(p.slot))); h.u((p.by_ref ? 1u : 0u) | (p.out ? 2u : 0u)); h.u(std::uint64_t(p.conv)); nid(p.dflt); }
        nid(f->body);
    }
    ix.fingerprint = h.h;
    return ix;
}

} // namespace aot

namespace {

using aot::program_index;

const char* nk_name(NK k) {
    static const char* n[] = {"nil", "true", "false", "integer literal", "floating-point literal", "string literal", "string interpolation", "variable", "list",
        "map", "unary operator", "binary operator", "&&", "||", "??", "call", "index", "member access", "optional member access", "lambda", "shell block",
        "is", "spawn", "await", "self", "super", "?:", "std::move", "copy", "block", "declaration", "assignment", "expression statement", "if", "while", "range-for",
        "function", "return", "break", "continue", "throw", "try", "class", "using", "defer", "import"};
    return std::size_t(k) < sizeof n / sizeof n[0] ? n[std::size_t(k)] : "unknown node";
}

// Text that is safe inside a // comment: printable ASCII, no backslash (line splicing), no '?' (trigraphs), bounded.
std::string comment_safe(const std::string& s, std::size_t max = 60) {
    std::string o;
    for (char c : s) { if (o.size() >= max) { o += "..."; break; } o += (c >= 32 && c < 127 && c != '\\' && c != '?') ? c : '_'; }
    return o;
}

struct Options { bool unbox = true; };

// One unit (program) -> C++ function definitions.
class Gen {
public:
    Gen(const program_index& ix, int unit, const std::string& file, Options opt) : ix_(ix), u_(unit), file_(file), opt_(opt) {}

    std::string run() {
        std::string fwd, defs;
        for (std::size_t i = 0; i < ix_.fns.size(); ++i) fwd += "static Value " + fname(int(i)) + "(Interp& in, const Ref<Env>& env);\n";
        for (std::size_t i = 0; i < ix_.fns.size(); ++i) defs += function(int(i));
        std::string tab = "static const aot::native_fn u" + std::to_string(u_) + "_fns[] = {";
        for (std::size_t i = 0; i < ix_.fns.size(); ++i) { tab += (i % 8 == 0 ? "\n    " : " "); tab += fname(int(i)) + ","; }
        tab += "\n};\n";
        return "static const Node* const* n" + std::to_string(u_) + ";   // node table of unit " + std::to_string(u_) + ", set when the runtime binds it\n" + fwd + "\n" + defs + tab;
    }
    int delegated = 0;      // statements / expressions handed to the tree-walker at run time
    int unboxed = 0;        // locals kept as unboxed ints
    std::set<std::string> delegated_kinds;

private:
    // ---- output
    void L(const std::string& s) { o_.append(std::size_t(ind_) * 2, ' '); o_ += s; o_ += '\n'; }
    void open(int line = 0) { if (++depth_ > 180) fail(line, "statement or expression nesting is too deep for native mode"); L("{"); ++ind_; }
    void close() { --ind_; L("}"); --depth_; }
    [[noreturn]] void fail(int line, const std::string& what) const { throw aot_error(line, file_ + ":" + std::to_string(line) + ": aot native: " + what); }
    std::string T(const char* p = "t") { return p + std::to_string(++tmp_); }
    std::string fname(int i) const { return "u" + std::to_string(u_) + "_f" + std::to_string(i); }
    std::string N(const Node& n) const {
        int id = ix_.node_of(n);
        if (id < 0) fail(n.line, "internal: node is not part of the program");
        return "(*n" + std::to_string(u_) + "[" + std::to_string(id) + "])";
    }
    static std::string num(std::int64_t v) {
        if (v == std::numeric_limits<std::int64_t>::min()) return "rt::imin";
        return "I64(" + std::to_string(v) + "LL)";
    }
    static std::string opx(Op op) { return "Op(" + std::to_string(int(op)) + ")"; }
    const Node& kid(const Node& n, std::size_t i) const {
        if (i >= n.kids.size() || !n.kids[i]) fail(n.line, std::string("malformed ") + nk_name(n.k) + " node");
        return *n.kids[i];
    }

    // ---- unboxed int locals: slots of the current function proven to hold only ints (see analyse())
    struct Types { std::set<int> ints; };
    bool is_int_slot(const VarRef& v) const { return !v.global() && v.depth == 0 && cur_ints_.count(v.slot); }
    std::string islot(int slot) const { return "i" + std::to_string(slot); }

    bool pure_int(const Node& n) const {   // an expression that is certainly an int and can be computed unboxed
        switch (n.k) {
        case NK::Int: return true;
        case NK::Var: return is_int_slot(n.var);
        case NK::Copy: return n.kids.size() == 1 && n.kids[0] && pure_int(*n.kids[0]);
        case NK::Unary: return (n.s == "-" || n.s == "~") && n.kids.size() == 1 && n.kids[0] && pure_int(*n.kids[0]);
        case NK::Binary: {
            Op op = Op(n.i);
            if (!(op == Op::Add || op == Op::Sub || op == Op::Mul || op == Op::Div || op == Op::Mod || op == Op::BitAnd || op == Op::BitOr || op == Op::BitXor || op == Op::Shl || op == Op::Shr)) return false;
            return n.kids.size() == 2 && n.kids[0] && n.kids[1] && pure_int(*n.kids[0]) && pure_int(*n.kids[1]);
        }
        default: return false;
        }
    }
    bool pure_bool(const Node& n) const {   // a comparison of unboxed ints (or a logical combination of such)
        switch (n.k) {
        case NK::True: case NK::False: return true;
        case NK::Binary: {
            Op op = Op(n.i);
            if (!(op == Op::Eq || op == Op::Ne || op == Op::Lt || op == Op::Le || op == Op::Gt || op == Op::Ge)) return false;
            return n.kids.size() == 2 && n.kids[0] && n.kids[1] && pure_int(*n.kids[0]) && pure_int(*n.kids[1]);
        }
        case NK::And: case NK::Or: return n.kids.size() == 2 && n.kids[0] && n.kids[1] && pure_bool(*n.kids[0]) && pure_bool(*n.kids[1]);
        case NK::Unary: return n.s == "not" && n.kids.size() == 1 && n.kids[0] && pure_bool(*n.kids[0]);
        default: return false;
        }
    }
    // C++ expression of type I64 for a pure_int node. Operands are sequenced left to right by the nested calls'
    // own statements, so each compound operation is emitted as statements into fresh I64 temporaries.
    std::string iexpr(const Node& n) {
        switch (n.k) {
        case NK::Int: return num(n.i);
        case NK::Var: return islot(n.var.slot);
        case NK::Copy: return iexpr(*n.kids[0]);
        case NK::Unary: {
            std::string a = iexpr(*n.kids[0]), t = T("x");
            L("const I64 " + t + " = " + (n.s == "~" ? "~" + a : "rt::ineg(in, " + a + ")") + ";");
            return t;
        }
        default: break;
        }
        Op op = Op(n.i);
        std::string a = iexpr(*n.kids[0]);
        std::string b = iexpr(*n.kids[1]);
        return int_op(op, a, b, n.line);
    }
    std::string bexpr(const Node& n) {   // C++ expression of type bool for a pure_bool node
        switch (n.k) {
        case NK::True: return "true";
        case NK::False: return "false";
        case NK::Unary: { std::string a = bexpr(*n.kids[0]); return "(!" + a + ")"; }
        case NK::And: case NK::Or: {
            std::string t = T("c");
            L("bool " + t + ";"); open(n.line);
            std::string a = bexpr(*n.kids[0]);
            L(t + " = " + a + ";");
            L(std::string("if (") + (n.k == NK::And ? "" : "!") + t + ")"); open(n.line);
            std::string b = bexpr(*n.kids[1]);
            L(t + " = " + b + ";");
            close(); close();
            return t;
        }
        default: break;
        }
        Op op = Op(n.i);
        std::string a = iexpr(*n.kids[0]), b = iexpr(*n.kids[1]);
        const char* c = op == Op::Eq ? "==" : op == Op::Ne ? "!=" : op == Op::Lt ? "<" : op == Op::Le ? "<=" : op == Op::Gt ? ">" : ">=";
        return "(" + a + " " + c + " " + b + ")";
    }

    // Which slots of `f` can live unboxed. Optimistic fixpoint: start from every slot that is declared (or is a
    // by-value int parameter) in this function, then drop a slot when something may store a non-int in it or
    // reach it other than through this function's own native code.
    std::set<int> analyse(const FnDecl& f) {
        std::set<int> cand;
        if (!opt_.unbox || f.is_top || !f.body) return cand;
        std::set<int> bad;
        std::map<int, std::vector<const Node*>> stores;   // slot -> initialisers / assignments (Let or Assign nodes)
        for (auto& p : f.params) { if (p.conv == 'i' && !p.by_ref && !p.out && !p.type) cand.insert(p.slot); else bad.insert(p.slot); }
        if (f.is_method) bad.insert(0);
        // Walk this function's own statements; anything nested in another function or handed to the tree-walker
        // makes the slots it mentions ineligible.
        std::function<void(const Node&, bool, int)> poison = [&](const Node& n, bool nested, int depth) {
            if (n.k == NK::Var && !n.var.global() && n.var.depth == depth) bad.insert(n.var.slot);
            if ((n.k == NK::Let || n.k == NK::For || n.k == NK::FnStmt || n.k == NK::ClassStmt || n.k == NK::Import || n.k == NK::Block || n.k == NK::Using) && !n.var.global() && n.var.depth == depth && n.var.slot >= 0) bad.insert(n.var.slot);
            if (n.k == NK::For && depth == 0) bad.insert(int(n.i));
            for (int l : n.locals) if (depth == 0) bad.insert(l);
            for (auto& k : n.kids) if (k) poison(*k, nested, depth);
            if (n.extra) poison(*n.extra, nested, depth);
            if (n.fn) { for (auto& p : n.fn->params) if (p.dflt) poison(*p.dflt, true, depth + 1); if (n.fn->body) poison(*n.fn->body, true, depth + 1); }
            if (n.cls) { for (auto& [nm, d] : n.cls->fields) if (d) poison(*d, true, depth); for (auto& m : n.cls->methods) if (m) { for (auto& p : m->params) if (p.dflt) poison(*p.dflt, true, depth + 1); if (m->body) poison(*m->body, true, depth + 1); } }
        };
        for (auto& p : f.params) if (p.dflt) poison(*p.dflt, false, 0);   // defaults are evaluated by the tree-walker
        std::function<void(const Node&, bool)> walk = [&](const Node& n, bool tree) {
            // An assignment used as a value (f(i++), a = b = c, while (--n > 0)) is evaluated by Interp::eval
            // (assign_value), which stores into the frame: its target cannot live in an unboxed native slot. As a
            // statement an Assign is a child of a Block (or a loop's step in `extra`).
            if (n.k != NK::Block)
                for (auto& k : n.kids)
                    if (k && k->k == NK::Assign && !k->kids.empty() && k->kids[0] && k->kids[0]->k == NK::Var && !k->kids[0]->var.global() && k->kids[0]->var.depth == 0)
                        bad.insert(k->kids[0]->var.slot);
            switch (n.k) {
            case NK::Let:
                if (!n.var.global() && n.var.depth == 0) {
                    if (n.ref || n.kids.size() != 1 || !n.kids[0] || (n.type && (n.type->name != "int" || n.type->nullable || !n.type->args.empty())) || tree) bad.insert(n.var.slot);
                    else { cand.insert(n.var.slot); stores[n.var.slot].push_back(&n); }
                }
                for (auto& k : n.kids) if (k) walk(*k, tree);
                return;
            case NK::Assign: {
                const Node* t = n.kids.empty() ? nullptr : n.kids[0].get();
                if (t && t->k == NK::Var && !t->var.global() && t->var.depth == 0) {
                    if (tree) bad.insert(t->var.slot); else stores[t->var.slot].push_back(&n);
                    if (n.kids.size() > 1 && n.kids[1]) walk(*n.kids[1], tree);
                    return;
                }
                bool tr = tree || (t && t->k == NK::Index && Op(n.i) != Op::None);
                for (auto& k : n.kids) if (k) walk(*k, tr);
                return;
            }
            case NK::For:
                if (!n.var.global() && n.var.depth == 0) bad.insert(n.var.slot);
                if (n.flag) bad.insert(int(n.i));
                for (auto& k : n.kids) if (k) walk(*k, tree || n.s == "ref");
                return;
            case NK::Try:
                if (n.extra) { poison(n, false, 0); return; }
                for (std::size_t i = 0; i < n.kids.size(); ++i) if (n.kids[i]) {
                    if (i > 0 && !n.kids[i]->var.global() && n.kids[i]->var.depth == 0) bad.insert(n.kids[i]->var.slot);
                    walk(*n.kids[i], true);
                }
                return;
            case NK::Block:
                for (auto& s : n.kids) if (s && s->k == NK::Defer) { poison(n, false, 0); return; }
                for (auto& k : n.kids) if (k) walk(*k, tree);
                return;
            case NK::FnStmt: if (!n.var.global() && n.var.depth == 0) bad.insert(n.var.slot); poison(n, false, 0); return;
            case NK::Lambda: case NK::ClassStmt: case NK::Import: case NK::Move: case NK::Super: case NK::Using: case NK::Defer:
            case NK::Interp: case NK::Shell: case NK::Is: case NK::Spawn: case NK::Await: case NK::Self: case NK::Coalesce: case NK::OptField:
                poison(n, false, 0); return;
            case NK::Call: if (n.wb) { poison(n, false, 0); return; } break;
            case NK::Throw: for (auto& k : n.kids) if (k) walk(*k, true); return;
            default: break;
            }
            for (auto& k : n.kids) if (k) walk(*k, tree);
            if (n.extra) walk(*n.extra, tree);
        };
        walk(*f.body, false);
        for (int b : bad) cand.erase(b);
        // A slot in a tree-walked region keeps its boxed form (the region's own load/store paths are generic),
        // and a slot whose stores are not all provably ints is dropped; repeat until nothing changes.
        std::set<int> saved = cur_ints_;
        for (bool changed = true; changed;) {
            changed = false;
            cur_ints_ = cand;
            for (auto it = cand.begin(); it != cand.end();) {
                bool ok = true;
                for (const Node* s : stores[*it]) {
                    const Node* rhs = s->k == NK::Let ? s->kids[0].get() : (s->kids.size() > 1 ? s->kids[1].get() : nullptr);
                    if (!rhs) { ok = false; break; }
                    if (s->k == NK::Assign) {
                        Op op = Op(s->i);
                        bool arith = op == Op::None || op == Op::Add || op == Op::Sub || op == Op::Mul || op == Op::Div || op == Op::Mod || op == Op::BitAnd || op == Op::BitOr || op == Op::BitXor || op == Op::Shl || op == Op::Shr;
                        if (!arith || !pure_int(*rhs)) { ok = false; break; }
                    } else if (!pure_int(*rhs) && !(s->type && int_cast(*rhs))) { ok = false; break; }   // untyped: the initialiser alone decides
                }
                if (!ok) { it = cand.erase(it); changed = true; } else ++it;
            }
        }
        cur_ints_ = saved;
        return cand;
    }
    // `int x = <expr>` where the parser wrapped the initialiser in forge::__cast("int", e): the value is an int
    // or the cast raised, so the slot stays an int and the call itself runs boxed.
    static bool int_cast(const Node& n) {
        if (n.k != NK::Call || n.wb || n.kids.size() != 3 || !n.kids[0] || !n.kids[1]) return false;
        return n.kids[0]->k == NK::Var && n.kids[0]->var.global() && n.kids[0]->s == "forge::__cast" && n.kids[1]->k == NK::Str && n.kids[1]->s == "int";
    }

    // ---- expressions: emit statements, return the name of a `Value` holding the result in the current scope
    std::string expr(const Node& n) {
        if (pure_int(n)) {
            std::string t = T();
            if (n.k == NK::Int) { L("Value " + t + " = " + num(n.i) + ";"); return t; }
            L("Value " + t + ";"); open(n.line);
            std::string x = iexpr(n);
            L(t + " = " + x + ";");
            close();
            return t;
        }
        if (pure_bool(n) && n.k != NK::True && n.k != NK::False) {
            std::string t = T();
            L("Value " + t + ";"); open(n.line);
            std::string x = bexpr(n);
            L(t + " = bool(" + x + ");");
            close();
            return t;
        }
        switch (n.k) {
        case NK::Nil: { auto t = T(); L("Value " + t + " = Nil{};"); return t; }
        case NK::True: { auto t = T(); L("Value " + t + " = true;"); return t; }
        case NK::False: { auto t = T(); L("Value " + t + " = false;"); return t; }
        case NK::Int: { auto t = T(); L("Value " + t + " = " + num(n.i) + ";"); return t; }
        case NK::Float: {
            std::uint64_t b; std::memcpy(&b, &n.d, 8);
            char buf[48]; std::snprintf(buf, sizeof buf, "0x%016llxULL", static_cast<unsigned long long>(b));
            auto t = T(); L("Value " + t + " = rt::dbl(" + buf + ");"); return t;
        }
        case NK::Str:
            if (n.flag && n.s.size() == 1) { auto t = T(); L("Value " + t + " = Interp::new_char(" + N(n) + ".s[0]);   // character literal"); return t; }
            { auto t = T(); L("Value " + t + " = in.new_string(" + N(n) + ".s);   // \"" + comment_safe(n.s, 40) + "\""); return t; }
        case NK::Var: {
            auto t = T();
            if (n.var.global() || tree_) L("Value " + t + " = in.load(" + N(n) + ".var, env);   // " + comment_safe(n.var.name));
            else L("Value " + t + " = " + load_expr(n) + ";   // " + comment_safe(n.var.name));
            return t;
        }
        case NK::List: {
            auto t = T(); L("Value " + t + ";"); open(n.line);
            std::vector<std::string> xs; for (std::size_t i = 0; i < n.kids.size(); ++i) xs.push_back(expr(kid(n, i)));
            auto l = T("l"); L("auto " + l + " = make<List>();");
            for (auto& x : xs) L(l + "->push(in.share(" + x + "));");
            L(t + " = obj(" + l + ");");
            close(); return t;
        }
        case NK::Map: {
            auto t = T(); L("Value " + t + ";"); open(n.line);
            std::vector<std::string> xs; for (std::size_t i = 0; i < n.kids.size(); ++i) xs.push_back(expr(kid(n, i)));
            auto m = T("m"); L("auto " + m + " = make<Map>();");
            for (std::size_t i = 0; i + 1 < xs.size(); i += 2) {
                L("if (!key_of(" + xs[i] + ")) in.raise(\"TypeError\", \"map keys must be nil, bool, numbers or strings\");");
                L(m + "->set(" + xs[i] + ", in.share(" + xs[i + 1] + "));");
            }
            L(t + " = obj(" + m + ");");
            close(); return t;
        }
        case NK::Unary: {
            auto t = T(); L("Value " + t + ";"); open(n.line);
            std::string a = expr(kid(n, 0));
            if (n.s == "not") L(t + " = !Interp::truthy(" + a + ");");
            else if (n.s == "bool") L(t + " = Interp::truthy(" + a + ");");
            else L(t + " = in.unary(" + N(n) + ".s, " + a + ");");
            close(); return t;
        }
        case NK::Ternary: {
            auto t = T(); L("Value " + t + ";"); open(n.line);
            std::string c = cond(kid(n, 0), 0);
            L("if (" + c + ")"); open(n.line); { std::string a = expr(kid(n, 1)); L(t + " = std::move(" + a + ");"); } close();
            L("else"); open(n.line); { std::string b = expr(kid(n, 2)); L(t + " = std::move(" + b + ");"); } close();
            close(); return t;
        }
        case NK::Copy: {
            auto t = T(); L("Value " + t + ";"); open(n.line);
            std::string a = expr(kid(n, 0));
            L(t + " = in.copy_value(" + a + ", -1);");
            close(); return t;
        }
        case NK::Binary: {
            auto t = T(); L("Value " + t + ";"); open(n.line);
            std::string a = expr(kid(n, 0)), b = expr(kid(n, 1));
            L("in.line = " + std::to_string(n.line) + "; " + t + " = rt::bin(in, " + opx(Op(n.i)) + ", " + a + ", " + b + "); " + a + " = Nil{};   // " + op_name(Op(n.i)));
            close(); return t;
        }
        case NK::And: case NK::Or: {
            auto t = T(); L("Value " + t + ";"); open(n.line);
            std::string a = expr(kid(n, 0));
            L(std::string("if (") + (n.k == NK::And ? "!" : "") + "Interp::truthy(" + a + ")) " + t + (n.k == NK::And ? " = false;" : " = true;"));
            L("else"); open(n.line);
            L(a + " = Nil{};");
            std::string b = expr(kid(n, 1));
            L(t + " = Interp::truthy(" + b + ");");
            close(); close(); return t;
        }
        case NK::Call: {
            if (n.wb) return delegate_eval(n);   // lvalue arguments bound to T& parameters: Interp::call_writeback
            auto t = T(); L("Value " + t + ";"); open(n.line);
            std::string f = expr(kid(n, 0));
            auto a = T("a"); L("std::vector<Value> " + a + "; " + a + ".reserve(" + std::to_string(n.kids.size() - 1) + ");");
            for (std::size_t i = 1; i < n.kids.size(); ++i) {
                open(n.line);
                std::string x = expr(kid(n, i));
                L(a + ".push_back(std::move(" + x + "));");
                close();
            }
            auto g = T("f"); L("Value " + g + " = std::move(" + f + ");");
            L(t + " = in.call(" + g + ", " + a + ", " + std::to_string(n.line) + ");");
            close(); return t;
        }
        case NK::Index: {
            auto t = T(); L("Value " + t + ";"); open(n.line);
            std::string a = expr(kid(n, 0)), b = expr(kid(n, 1));
            L("in.line = " + std::to_string(n.line) + "; " + t + " = in.get_index(" + a + ", " + b + ");");
            close(); return t;
        }
        case NK::Field: {
            auto t = T(); L("Value " + t + ";"); open(n.line);
            std::string a = expr(kid(n, 0));
            L("in.line = " + std::to_string(n.line) + "; " + t + " = in.get_field(" + a + ", " + N(n) + ".s);   // ." + comment_safe(n.s));
            close(); return t;
        }
        case NK::Lambda: {
            if (!n.fn) fail(n.line, "malformed lambda node");
            auto t = T(); L("Value " + t + " = in.make_closure(" + N(n) + ".fn, env);"); return t;
        }
        default: return delegate_eval(n);   // Move, Super and node kinds the parser no longer produces: Interp::eval
        }
    }
    std::string delegate_eval(const Node& n) {
        ++delegated; delegated_kinds.insert(nk_name(n.k));
        auto t = T();
        L(std::string("Value ") + t + " = (" + (tree_ ? "" : "in.line = " + std::to_string(n.line) + ", ") + "in.eval(" + N(n) + ", env));   // " + nk_name(n.k) + ": tree-walker");
        return t;
    }
    // Evaluate a condition into a C++ bool declared in the current scope; its Value dies before the branch runs.
    // `line` > 0: the statement's Line instruction (line + step) comes first, inside the same scope.
    std::string cond(const Node& n, int line) {
        auto c = T("c");
        if (pure_bool(n)) {
            if (line) L("rt::ls(in, " + std::to_string(line) + ");");
            L("bool " + c + ";"); open(n.line);
            std::string x = bexpr(n);
            L(c + " = " + x + ";");
            close();
            return c;
        }
        L("bool " + c + ";"); open(n.line);
        if (line) L("rt::ls(in, " + std::to_string(line) + ");");
        std::string v = expr(n);
        L(c + " = Interp::truthy(" + v + ");");
        close();
        return c;
    }

    // ---- control-flow context
    struct Ctx { std::string brk, cont, ret; };   // ret: "%" is replaced by the name of the Value to return
    std::vector<Ctx> ctx_;
    void do_break(const Node& n) { if (ctx_.back().brk.empty()) fail(n.line, "break outside a loop"); L(ctx_.back().brk); }
    void do_continue(const Node& n) { if (ctx_.back().cont.empty()) fail(n.line, "continue outside a loop"); L(ctx_.back().cont); }
    void do_return(const std::string& v) { std::string r = ctx_.back().ret; auto p = r.find('%'); r.replace(p, 1, v); L(r); }
    void push_loop(std::string brk, std::string cont) { ctx_.push_back({std::move(brk), std::move(cont), ctx_.back().ret}); }

    // A statement run by the tree-walker (Interp::exec or exec_block on the node), with its Flow mapped back.
    void delegate_exec(const Node& n, bool as_block) {
        ++delegated; delegated_kinds.insert(nk_name(n.k));
        open(n.line);
        L(std::string("const Flow fl = in.") + (as_block ? "exec_block(" : "exec(") + N(n) + ", env);   // " + nk_name(n.k) + ": tree-walker");
        L("if (fl == Flow::Return)"); open(n.line); L("Value r = std::move(in.ret); in.ret = Nil{};"); do_return("r"); close();
        L("if (fl == Flow::Break) " + (ctx_.back().brk.empty() ? std::string("rt::bad_flow(in);") : ctx_.back().brk));
        L("if (fl == Flow::Continue) " + (ctx_.back().cont.empty() ? std::string("rt::bad_flow(in);") : ctx_.back().cont));
        close();
    }
    static bool has_defer(const Node& b) { for (auto& s : b.kids) if (s && s->k == NK::Defer) return true; return false; }

    // ---- stores
    void store_var(const Node& holder, const VarRef& v, const std::string& val, bool define, bool is_const) {
        if (is_int_slot(v)) fail(holder.line, "internal: boxed store to an unboxed local");
        if (v.global() || tree_) {
            if (define) L("in.store(" + N(holder) + ".var, env, std::move(" + val + "), true, " + (is_const ? "true" : "false") + ");   // " + comment_safe(v.name));
            else L("in.store(" + N(holder) + ".var, env, std::move(" + val + "));   // " + comment_safe(v.name));
        } else L("rt::st(in, env, " + std::to_string(v.depth) + ", " + std::to_string(v.slot) + ", std::move(" + val + "));   // " + comment_safe(v.name));
    }

    // ---- statements, VM discipline (vm.cpp Compiler::stmt)
    void block_inline(const Node& b) {
        if (has_defer(b)) { delegate_exec(b, false); return; }
        open(b.line);
        for (auto& s : b.kids) { if (!s) fail(b.line, "malformed block"); stmt(*s); }
        for (auto it = b.locals.rbegin(); it != b.locals.rend(); ++it) {
            if (cur_ints_.count(*it)) continue;   // unboxed: nothing to release
            if (*it < 0 || *it >= cur_nslots_) fail(b.line, "internal: block local outside the frame");
            L("rt::clr(in, env, " + std::to_string(*it) + ");");
        }
        close();
    }
    void stmt(const Node& n) {
        if (tree_) { tstmt(n); return; }
        const std::string ln = std::to_string(n.line);
        switch (n.k) {
        case NK::ExprStmt: open(n.line); L("rt::ls(in, " + ln + ");"); expr(kid(n, 0)); close(); return;
        case NK::Let: {
            open(n.line); L("rt::ls(in, " + ln + ");");
            if (is_int_slot(n.var)) {
                const Node& init = kid(n, 0);
                if (pure_int(init)) { std::string x = iexpr(init); L(islot(n.var.slot) + " = " + x + ";   // " + comment_safe(n.var.name)); }
                else { std::string v = expr(init); L("rt::let_check(in, " + v + ", " + N(n) + "); " + islot(n.var.slot) + " = rt::need_int(in, " + v + ", " + N(n) + ");   // " + comment_safe(n.var.name)); }
                close(); return;
            }
            std::string v;
            if (n.kids.empty()) { v = T(); L("Value " + v + " = Nil{};"); } else v = expr(kid(n, 0));
            if (n.type) L("rt::let_check(in, " + v + ", " + N(n) + ");");
            store_var(n, n.var, v, true, n.flag);
            close(); return;
        }
        case NK::Assign: {
            const Node& t = kid(n, 0); Op op = Op(n.i);
            if (t.k == NK::Var) {
                open(n.line); L("rt::ls(in, " + ln + ");");
                if (is_int_slot(t.var)) {
                    std::string x;   // x op= e  ==  x = x op e on unboxed ints
                    if (op == Op::None) x = iexpr(kid(n, 1));
                    else {
                        std::string a = T("x"); L("const I64 " + a + " = " + islot(t.var.slot) + ";");
                        std::string b = iexpr(kid(n, 1));
                        x = int_op(op, a, b, n.line);
                    }
                    L(islot(t.var.slot) + " = " + x + ";   // " + comment_safe(t.var.name));
                    close(); return;
                }
                std::string a;
                if (op != Op::None) { a = T(); L("Value " + a + " = " + load_expr(t) + ";"); }
                std::string b = expr(kid(n, 1));
                if (op != Op::None) {
                    auto r = T(); L("in.line = " + ln + "; Value " + r + " = rt::bin(in, " + opx(op) + ", " + a + ", " + b + "); " + a + " = Nil{}; " + b + " = Nil{};");
                    store_var(t, t.var, r, false, false);
                } else store_var(t, t.var, b, false, false);
                close(); return;
            }
            if (t.k == NK::Field) {
                open(n.line); L("rt::ls(in, " + ln + ");");
                std::string o = expr(kid(t, 0));
                std::string g;
                if (op != Op::None) { g = T(); L("Value " + g + ";"); open(n.line); auto d = T(); L("Value " + d + " = " + o + ";"); L("in.line = " + ln + "; " + g + " = in.get_field(" + d + ", " + N(t) + ".s);"); close(); }
                std::string v = expr(kid(n, 1));
                if (op != Op::None) { auto r = T(); L("in.line = " + ln + "; Value " + r + " = rt::bin(in, " + opx(op) + ", " + g + ", " + v + "); " + g + " = Nil{}; " + v + " = Nil{};"); v = r; }
                L("in.set_field(" + o + ", " + N(t) + ".s, std::move(" + v + "));   // ." + comment_safe(t.s));
                close(); return;
            }
            if (t.k == NK::Index && op == Op::None) {
                open(n.line); L("rt::ls(in, " + ln + ");");
                std::string o = expr(kid(t, 0)), i = expr(kid(t, 1)), v = expr(kid(n, 1));
                L("in.set_index(" + o + ", " + i + ", std::move(" + v + "));");
                close(); return;
            }
            as_tree(n); return;   // compound index assignment: the VM delegates it
        }
        case NK::If: {
            open(n.line);
            std::string c = cond(kid(n, 0), n.line);
            L("if (" + c + ")"); block_inline(kid(n, 1));
            if (n.kids.size() > 2) { L("else"); block_inline(kid(n, 2)); }
            close(); return;
        }
        case NK::While: {
            std::string lab = T("L");
            L("for (;;)"); open(n.line);
            std::string c = cond(kid(n, 0), n.line);
            L("if (!" + c + ") break;");
            push_loop("break;", n.extra ? "goto " + lab + ";" : "continue;");
            block_inline(kid(n, 1));
            ctx_.pop_back();
            if (n.extra) { L(lab + ": ;"); push_loop("break;", ""); block_inline(*n.extra); ctx_.pop_back(); }
            close(); return;
        }
        case NK::For: {
            if (n.s == "ref") { as_tree(n); return; }   // write-through element references: the VM delegates them
            if (n.var.global() || n.var.depth < 0 || n.var.slot < 0) fail(n.line, "range-for with a non-local loop variable");
            open(n.line); L("rt::ls(in, " + ln + ");");
            auto it = T("it"); L("Ref<Obj> " + it + ";");
            open(n.line); { std::string s = expr(kid(n, 0)); L(it + " = in.iterate(" + s + ");"); } close();
            L("for (;;)"); open(n.line);
            auto a = T(), b = T();
            L("if (!" + it + ") in.raise(\"TypeError\", \"not an iterator\");");
            L("Value " + a + ", " + b + ";");
            L("if (!in.iter_next(*" + it + ", " + a + ", " + (n.flag ? "&" + b : std::string("nullptr")) + ")) break;");
            L("in.step();");
            if (n.flag) L("rt::st(in, env, 0, " + std::to_string(n.i) + ", std::move(" + b + "));");
            L("rt::st(in, env, " + std::to_string(n.var.depth) + ", " + std::to_string(n.var.slot) + ", std::move(" + a + "));   // " + comment_safe(n.var.name));
            push_loop("break;", "continue;");
            block_inline(kid(n, 1));
            ctx_.pop_back();
            close(); close(); return;
        }
        case NK::FnStmt: {
            if (!n.fn) fail(n.line, "malformed function node");
            open(n.line); L("rt::ls(in, " + ln + ");");
            auto f = T(); L("Value " + f + " = in.make_closure(" + N(n) + ".fn, env);");
            store_var(n, n.var, f, true, false);
            close(); return;
        }
        case NK::Return: {
            open(n.line); L("rt::ls(in, " + ln + ");");
            if (n.kids.empty()) { auto v = T(); L("Value " + v + " = Nil{};"); do_return(v); }
            else { std::string v = expr(kid(n, 0)); do_return(v); }
            close(); return;
        }
        case NK::Break: do_break(n); return;
        case NK::Continue: do_continue(n); return;
        case NK::Block: block_inline(n); return;
        case NK::Defer: return;   // only reachable through a delegated block
        default: as_tree(n); return;   // Try, Using, ClassStmt, Import, Throw
        }
    }
    std::string load_expr(const Node& var) {
        if (var.var.global() || tree_) return "in.load(" + N(var) + ".var, env)";
        return "rt::ld(in, env, " + std::to_string(var.var.depth) + ", " + std::to_string(var.var.slot) + ", n" + std::to_string(u_) + ", " + std::to_string(ix_.node_of(var)) + ")";
    }
    std::string int_op(Op op, const std::string& a, const std::string& b, int line) {
        std::string t = T("x"), e;
        switch (op) {
        case Op::Add: e = "rt::iadd(in, " + a + ", " + b + ")"; break;
        case Op::Sub: e = "rt::isub(in, " + a + ", " + b + ")"; break;
        case Op::Mul: e = "rt::imul(in, " + a + ", " + b + ")"; break;
        case Op::Div: e = "rt::idiv(in, " + a + ", " + b + ")"; break;
        case Op::Mod: e = "rt::imod(in, " + a + ", " + b + ")"; break;
        case Op::Shl: e = "rt::ishl(in, " + a + ", " + b + ")"; break;
        case Op::Shr: e = "rt::ishr(in, " + a + ", " + b + ")"; break;
        case Op::BitAnd: e = "(" + a + " & " + b + ")"; break;
        case Op::BitOr: e = "(" + a + " | " + b + ")"; break;
        case Op::BitXor: e = "(" + a + " ^ " + b + ")"; break;
        default: fail(line, "internal: not an integer operator");
        }
        L("in.line = " + std::to_string(line) + "; const I64 " + t + " = " + e + ";");
        return t;
    }
    void as_tree(const Node& n) { bool saved = tree_; tree_ = true; tstmt(n); tree_ = saved; }

    // ---- statements, tree-walker discipline (tree.cpp Interp::exec / exec_block)
    void tblock(const Node& b) {
        if (has_defer(b)) { delegate_exec(b, true); return; }
        open(b.line);
        if (!b.locals.empty()) L("rt::tree_scope sc(env, " + N(b) + ");");
        for (auto& s : b.kids) { if (!s) fail(b.line, "malformed block"); tstmt(*s); }
        close();
    }
    void tstmt(const Node& n) {
        const std::string ln = std::to_string(n.line);
        switch (n.k) {
        case NK::ExprStmt: open(n.line); L("rt::ls(in, " + ln + ");"); expr(kid(n, 0)); close(); return;
        case NK::Let: {
            open(n.line); L("rt::ls(in, " + ln + ");");
            std::string v;
            if (n.kids.empty()) { v = T(); L("Value " + v + " = Nil{};"); } else v = expr(kid(n, 0));
            if (n.type) L("rt::let_check(in, " + v + ", " + N(n) + ");");
            store_var(n, n.var, v, true, n.flag);
            close(); return;
        }
        case NK::Assign: {
            const Node& t = kid(n, 0); Op op = Op(n.i);
            open(n.line); L("rt::ls(in, " + ln + ");");
            if (t.k == NK::Var) {
                std::string v = expr(kid(n, 1));
                if (op != Op::None) { open(n.line); auto c = T(); L("Value " + c + " = in.load(" + N(t) + ".var, env);"); L(v + " = rt::bin(in, " + opx(op) + ", " + c + ", " + v + ");"); close(); }
                store_var(t, t.var, v, false, false);
            } else if (t.k == NK::Field) {
                std::string o = expr(kid(t, 0)), v = expr(kid(n, 1));
                if (op != Op::None) { open(n.line); auto g = T(); L("Value " + g + " = in.get_field(" + o + ", " + N(t) + ".s);"); L(v + " = rt::bin(in, " + opx(op) + ", " + g + ", " + v + ");"); close(); }
                L("in.set_field(" + o + ", " + N(t) + ".s, std::move(" + v + "));");
            } else {
                if (t.kids.size() < 2) fail(n.line, "malformed assignment target");
                std::string o = expr(kid(t, 0)), i = expr(kid(t, 1)), v = expr(kid(n, 1));
                if (op != Op::None) { open(n.line); auto g = T(); L("Value " + g + " = in.get_index(" + o + ", " + i + ");"); L(v + " = rt::bin(in, " + opx(op) + ", " + g + ", " + v + ");"); close(); }
                L("in.set_index(" + o + ", " + i + ", std::move(" + v + "));");
            }
            close(); return;
        }
        case NK::If: {
            open(n.line); L("rt::ls(in, " + ln + ");");
            std::string c = cond(kid(n, 0), 0);
            L("if (" + c + ")"); tblock(kid(n, 1));
            if (n.kids.size() > 2) { L("else"); tblock(kid(n, 2)); }
            close(); return;
        }
        case NK::While: {
            std::string lab = T("L");
            open(n.line); L("rt::ls(in, " + ln + ");");
            L("for (;;)"); open(n.line);
            std::string c = cond(kid(n, 0), 0);
            L("if (!" + c + ") break;");
            L("in.step();");
            push_loop("break;", n.extra ? "goto " + lab + ";" : "continue;");
            tblock(kid(n, 1));
            ctx_.pop_back();
            if (n.extra) { L(lab + ": ;"); push_loop("break;", ""); tblock(*n.extra); ctx_.pop_back(); }
            close(); close(); return;
        }
        case NK::For: {
            const bool ref = n.s == "ref";
            const std::string second = "VarRef{std::string(), 0, " + std::to_string(n.i) + "}";
            std::string lab = T("L");
            open(n.line); L("rt::ls(in, " + ln + ");");
            std::string src = expr(kid(n, 0));
            auto it = T("it"); L("Ref<Obj> " + it + " = in.iterate(" + src + ");");
            auto a = T(), b = T(); L("Value " + a + ", " + b + ";");
            std::string fl, rv;
            if (ref) { fl = T("fl"); rv = T(); L("int " + fl + " = 0; Value " + rv + ";"); }
            L("for (;;)"); open(n.line);
            L("if (!in.iter_next(*" + it + ", " + a + ", " + (n.flag ? "&" + b : std::string("nullptr")) + ")) break;");
            L("in.store(" + N(n) + ".var, env, " + a + ");   // " + comment_safe(n.var.name));
            if (n.flag) L("in.store(" + second + ", env, " + b + ");");
            if (ref) {
                // for (T& x : c): a rebound x is written back to the element after the body, also on break,
                // continue and return, as Interp::exec does.
                auto a0 = T(), b0 = T();
                L("Value " + a0 + " = " + a + ", " + b0 + " = " + b + ";");
                ctx_.push_back({"{ " + fl + " = 1; goto " + lab + "; }", "goto " + lab + ";", "{ " + rv + " = std::move(%); " + fl + " = 2; goto " + lab + "; }"});
                tblock(kid(n, 1));
                ctx_.pop_back();
                L(lab + ": ;");
                open(n.line);
                auto a1 = T(), b1 = T();
                L("Value " + a1 + " = in.load(" + N(n) + ".var, env), " + b1 + " = " + (n.flag ? "in.load(" + second + ", env)" : std::string("Value(Nil{})")) + ";");
                L("if (!rt::same_binding(" + a0 + ", " + a1 + ")" + (n.flag ? " || !rt::same_binding(" + b0 + ", " + b1 + ")" : std::string()) + ") in.iter_writeback(*" + it + ", " + a1 + ", " + (n.flag ? "&" + b1 : std::string("nullptr")) + ");");
                close();
                L("if (" + fl + " == 1) break;");
                L("if (" + fl + " == 2)"); open(n.line); do_return(rv); close();
                L("in.step();");
            } else {
                push_loop("break;", "goto " + lab + ";");
                tblock(kid(n, 1));
                ctx_.pop_back();
                L(lab + ": ;");
                L("in.step();");
            }
            close(); close(); return;
        }
        case NK::FnStmt: {
            if (!n.fn) fail(n.line, "malformed function node");
            open(n.line); L("rt::ls(in, " + ln + ");");
            auto f = T(); L("Value " + f + " = in.make_closure(" + N(n) + ".fn, env);");
            L("in.store(" + N(n) + ".var, env, std::move(" + f + "), true);");
            close(); return;
        }
        case NK::Return: {
            open(n.line); L("rt::ls(in, " + ln + ");");
            if (n.kids.empty()) { auto v = T(); L("Value " + v + " = Nil{};"); do_return(v); }
            else { std::string v = expr(kid(n, 0)); do_return(v); }
            close(); return;
        }
        case NK::Break: open(n.line); L("rt::ls(in, " + ln + ");"); do_break(n); close(); return;
        case NK::Continue: open(n.line); L("rt::ls(in, " + ln + ");"); do_continue(n); close(); return;
        case NK::Throw: {
            open(n.line); L("rt::ls(in, " + ln + ");");
            std::string v = expr(kid(n, 0));
            L("rt::throw_value(in, std::move(" + v + "), " + ln + ");");
            close(); return;
        }
        case NK::Try: {
            if (n.extra) { delegate_exec(n, false); return; }   // finally: Interp::run_try
            open(n.line); L("rt::ls(in, " + ln + ");");
            auto pend = T("pend"), e = T("e"), h = T("h");
            L("std::exception_ptr " + pend + ";");
            L("try"); tblock(kid(n, 0));
            L("catch (...)"); open(n.line);
            L("ScriptError " + e + " = in.from_cpp(std::current_exception());");
            L("const int " + h + " = rt::match_catch(in, " + e + ", " + N(n) + ");");
            for (std::size_t i = 1; i < n.kids.size(); ++i) {
                const Node& c = kid(n, i);
                L(std::string(i > 1 ? "else " : "") + "if (" + h + " == " + std::to_string(i) + ")"); open(c.line);
                L("in.store(" + N(c) + ".var, env, " + e + ".v);   // catch (" + comment_safe(c.s.empty() ? "..." : c.s) + ")");
                L("try"); tblock(kid(c, 0));
                L("catch (...) { " + pend + " = std::current_exception(); }");
                close();
            }
            L(std::string(n.kids.size() > 1 ? "else " : "") + pend + " = std::make_exception_ptr(" + e + ");");
            close();
            L("if (" + pend + ") std::rethrow_exception(" + pend + ");");
            close(); return;
        }
        case NK::ClassStmt: {
            if (!n.cls) fail(n.line, "malformed class node");
            open(n.line); L("rt::ls(in, " + ln + ");");
            L("in.store(" + N(n) + ".var, env, in.define_class(*" + N(n) + ".cls, env), true, true);   // class " + comment_safe(n.cls->name));
            close(); return;
        }
        case NK::Block: open(n.line); L("rt::ls(in, " + ln + ");"); tblock(n); close(); return;
        case NK::Import: case NK::Using: case NK::Defer: delegate_exec(n, false); return;
        default:
            // An expression in statement position: Interp::exec evaluates it and drops the value.
            open(n.line); L("rt::ls(in, " + ln + ");"); expr(n); close(); return;
        }
    }

    std::string function(int idx) {
        const FnDecl& f = *ix_.fns[std::size_t(idx)];
        o_.clear(); ind_ = 1; tmp_ = 0; depth_ = 0; tree_ = false;
        ctx_.clear(); ctx_.push_back({"", "", "return %;"});
        cur_nslots_ = f.nslots;
        cur_ints_.clear();
        std::set<int> ints = analyse(f);
        cur_ints_ = ints;
        unboxed += int(ints.size());
        std::string head = "// " + comment_safe(f.name.empty() ? "<anonymous>" : f.name, 80) + " (line " + std::to_string(f.line) + ")\n";
        head += "static Value " + fname(idx) + "(Interp& in, const Ref<Env>& env) {\n";
        if (!f.body) { return head + "  (void)in; (void)env;\n  return Nil{};\n}\n\n"; }
        for (int s : ints) {
            bool is_param = false;
            for (auto& p : f.params) if (p.slot == s) is_param = true;
            if (is_param) L("I64 " + islot(s) + " = std::get<I64>(env->slots.at(" + std::to_string(s) + "));   // unboxed int parameter");
            else L("I64 " + islot(s) + " = 0;   // unboxed int local");
        }
        block_inline(*f.body);
        L("return Nil{};");
        return head + o_ + "}\n\n";
    }

    const program_index& ix_; int u_; std::string file_; Options opt_;
    std::string o_; int ind_ = 1, tmp_ = 0, depth_ = 0; bool tree_ = false;
    std::set<int> cur_ints_; int cur_nslots_ = 0;
};

} // namespace

// ---------------------------------------------------------------- program assembly (shared with aot_compile.cpp)
namespace aot {


namespace {
namespace fs = std::filesystem;

std::vector<std::string> search_dirs(const std::string& importer_file, const std::string& base_dir, const aot_options& o) {
    std::vector<std::string> out;
    out.push_back(fs::path(importer_file).parent_path().string());
    if (!base_dir.empty()) out.push_back(base_dir);
    for (auto& d : o.module_dirs) out.push_back(d);
    auto mp = os::getenv("FORGE_SCRIPT_PATH");
    std::size_t i = 0;
#if defined(_WIN32)
    const char* seps = ";";
#else
    const char* seps = ":;";
#endif
    while (!mp.empty() && i <= mp.size()) { auto j = mp.find_first_of(seps, i); if (j == std::string::npos) j = mp.size(); if (j > i) out.push_back(mp.substr(i, j - i)); i = j + 1; }
    auto home = os::getenv("HOME"); if (home.empty()) home = os::getenv("USERPROFILE");
    if (!home.empty()) out.push_back((fs::path(home) / ".forge" / "modules").string());
    std::error_code ec; out.push_back(fs::current_path(ec).string());
    return out;
}
bool native_module(const std::string& p) { auto e = fs::path(p).extension().string(); return e == ".so" || e == ".dylib" || e == ".dll"; }

// Resolve a static import the way Interp::import_module does; empty when it is not found (the binary then
// searches at run time, as the interpreter would, and raises ImportError if it is still missing).
std::string resolve(const std::string& path, const std::string& importer_file, const std::string& base_dir, const aot_options& o) {
    fs::path p(path);
    std::vector<fs::path> forms;
    if (p.extension().empty()) for (auto f : {p.string() + ".fs", p.string() + ".cpp", (p / "mod.fs").string(), (p / "mod.cpp").string()}) forms.emplace_back(f);
    else forms.push_back(p);
    std::vector<fs::path> cand;
    if (p.is_absolute()) cand = forms;
    else for (auto& dir : search_dirs(importer_file, base_dir, o)) for (auto& f : forms) cand.push_back(fs::path(dir) / f);
    for (auto& c : cand) { std::error_code ec; if (fs::is_regular_file(c, ec)) { auto w = fs::weakly_canonical(c, ec); return ec ? c.string() : w.string(); } }
    return {};
}

void emit_bytes(std::string& out, const std::string& name, const std::vector<std::uint8_t>& b) {
    out += "static const unsigned char " + name + "[] = {";
    for (std::size_t i = 0; i < b.size(); ++i) { if (i % 24 == 0) out += "\n    "; out += std::to_string(unsigned(b[i])); out += ','; }
    if (b.empty()) out += "0";
    out += "\n};\n";
}
}

translated translate_program(std::string_view src_in, const std::string& file, const aot_options& o, bool want_cxx) {
    translated tr;
    std::string src(src_in);
    if (src.starts_with("#!")) { auto i = src.find('\n'); src = i == std::string::npos ? "" : "#" + src.substr(i); }   // keep line numbers (engine::run_file)
    std::shared_ptr<const FnDecl> top;
    try { top = parse_program(src, file); }
    catch (const syntax_error& e) { throw aot_error(e.line, std::string("SyntaxError: ") + e.what()); }
    std::error_code ec;
    auto abs = fs::weakly_canonical(fs::path(file), ec);
    const std::string main_abs = ec ? file : abs.string();
    const std::string base_dir = fs::path(main_abs).parent_path().string();
    if (o.run.static_check) {
        check_options co{o.run.allow_fs, o.run.allow_net, o.run.allow_proc, o.run.allow_env, o.run.strict_check, {}};
        if (!base_dir.empty()) co.module_dirs.push_back(base_dir);
        for (auto& d : o.module_dirs) co.module_dirs.push_back(d);
        auto ds = check_parsed(*top, src, file, co);
        if (o.run.strict_check ? !ds.empty() : has_errors(ds))
            throw aot_error(ds.front().line, "StaticCheckError: script refused by static checks (forge script --check for details)\n" + to_text(ds));
    }

    payload& pl = tr.pl;
    pl.name = file; pl.source = src;
    auto& b = pl.baked;
    b.allow_fs = o.run.allow_fs; b.allow_net = o.run.allow_net; b.allow_proc = o.run.allow_proc; b.allow_env = o.run.allow_env;
    b.max_steps = o.run.max_steps; b.timeout_ms = o.run.timeout_ms; b.heap_bytes = o.run.heap_bytes; b.max_depth = o.run.max_depth; b.mem_limit_bytes = o.run.mem_limit_bytes;
    b.os_sandbox = std::uint8_t(o.run.os_sandbox); b.static_check = o.run.static_check; b.strict_check = o.run.strict_check; b.debug = o.run.debug;
    b.allow_widen = o.allow_widen;
    if (!o.policy_file.empty()) {
        auto t = os::read_file(o.policy_file);
        if (!t) throw aot_error(0, "cannot read policy " + o.policy_file + ": " + t.err.what);
        b.policy_json = t.value; b.policy_origin = fs::path(o.policy_file).filename().string();
    }

    // Units: the main program, then every statically imported script module, breadth-first in import order.
    struct Unit { std::string key, file; std::shared_ptr<const FnDecl> top; program_index ix; };
    std::vector<Unit> units;
    units.push_back({file, main_abs, top, index_program(*top)});
    std::map<std::string, std::string> key_of_abs;   // canonical path -> unit key
    std::set<std::string> keys{file};
    for (std::size_t ui = 0; ui < units.size(); ++ui) {
        const std::string importer_key = units[ui].key, importer_file = units[ui].file;
        std::vector<const Node*> nodes = units[ui].ix.nodes;   // copy: units grows below
        for (const Node* n : nodes) {
            if (n->k != NK::Import || n->s.empty()) continue;
            bool seen = false;
            for (auto& m : pl.modules) if (m.importer == importer_key && m.path == n->s) seen = true;
            if (seen) continue;
            std::string found = resolve(n->s, importer_file, base_dir, o);
            if (found.empty() || native_module(found)) continue;
            auto it = key_of_abs.find(found);
            if (it != key_of_abs.end()) { pl.modules.push_back({importer_key, n->s, it->second, std::string()}); continue; }
            auto text = os::read_file(found);
            if (!text) continue;
            std::shared_ptr<const FnDecl> mt;
            std::string key = fs::path(found).lexically_relative(base_dir).generic_string();
            if (key.empty() || key == ".") key = fs::path(found).filename().string();
            for (int k = 2; keys.count(key); ++k) key = fs::path(found).filename().string() + "#" + std::to_string(k);
            try { mt = parse_program(text.value, key); }
            catch (const syntax_error&) { continue; }   // the import raises SyntaxError at run time, as in the interpreter
            keys.insert(key); key_of_abs[found] = key;
            pl.modules.push_back({importer_key, n->s, key, text.value});
            units.push_back({key, found, mt, index_program(*mt)});
        }
    }
    // A module imported from several places is stored once: later entries carry an empty source and share the key.
    if (!want_cxx) return tr;

    std::string& out = tr.cxx;
    out += "// Generated by `forge script --aot` (native mode) from " + comment_safe(file, 200) + ". Do not edit.\n";
    out += "// Links against forge_script_rt; see docs/aot.md.\n";
    out += "#include \"forge_aot_rt.hpp\"\n\n";
    out += "namespace {\nusing namespace forge::script;\nnamespace rt = forge::script::aot_rt;\nusing rt::I64;\n\n";
    for (std::size_t ui = 0; ui < units.size(); ++ui) {
        Gen g(units[ui].ix, int(ui), units[ui].key, Options{});
        out += "// ---- unit " + std::to_string(ui) + ": " + comment_safe(units[ui].key, 200) + "\n";
        out += g.run();
        out += "\n";
        tr.delegated += g.delegated; tr.unboxed += g.unboxed;
        tr.delegated_kinds.insert(g.delegated_kinds.begin(), g.delegated_kinds.end());
    }
    emit_bytes(out, "payload", encode_payload(pl));
    out += "\nstatic const aot::unit units[] = {\n";
    for (std::size_t ui = 0; ui < units.size(); ++ui) {
        char fp[40]; std::snprintf(fp, sizeof fp, "0x%016llxULL", static_cast<unsigned long long>(units[ui].ix.fingerprint));
        std::string lit;   // unit names are matched at run time: emit them as escaped byte strings
        for (unsigned char c : units[ui].key) { char e[8]; std::snprintf(e, sizeof e, "\\%03o", unsigned(c)); lit += e; }
        out += "    {\"" + lit + "\", " + fp + ", " + std::to_string(units[ui].ix.nodes.size()) + "u, " + std::to_string(units[ui].ix.fns.size()) + "u, u" + std::to_string(ui) + "_fns, &n" + std::to_string(ui) + "},\n";
    }
    out += "};\n\n} // namespace\n\n";
    out += "int main(int argc, char** argv) {\n    return forge::script::aot::run_native(argc, argv, payload, sizeof payload, units, sizeof units / sizeof units[0], " + std::to_string(abi_version) + "u);\n}\n";
    return tr;
}

} // namespace aot

std::string aot_translate(std::string_view src, const std::string& file, const aot_options& o) {
    return aot::translate_program(src, file, o, true).cxx;
}

} // namespace forge::script
