// forge::script static checker. See check.hpp and docs/scripting.md "Static checks".
#include <forge/script/check.hpp>
#include "ast.hpp"
#include "parser.hpp"
#include <algorithm>
#include <cctype>
#include <functional>
#include <memory>
#include <tuple>
#include <cstdio>
#include <map>
#include <set>

namespace forge::script {
namespace {

using L = diagnostic::level;

// Built-in exception hierarchy (builtins.cpp): child -> parent. Builtin classes are frozen globals.
const std::map<std::string, std::string>& err_parent() {
    static const std::map<std::string, std::string> m = [] {
        std::map<std::string, std::string> p;
        for (auto n : {"IOError", "OSError", "TypeError", "IndexError", "KeyError", "OverflowError", "ObjectDisposedError", "ShellError",
                       "LimitExceededError", "NameError", "AttributeError", "ValueError", "ArithmeticError", "SyntaxError", "ImportError",
                       "RuntimeError", "LogicError", "BorrowError", "NullPointerError", "FrozenError", "RecursionError", "SystemExit",
                       "RaceError", "PipelineError", "PluginError", "AIError", "AccessError"}) p[n] = "Error";
        for (auto n : {"PermissionError", "TimeoutError", "OutOfMemoryError", "CancelledError", "DeadlockError", "FileNotFoundError",
                       "FileExistsError", "NotSupportedError", "InterruptedError", "ConnectionError", "SignalError", "BlockingIOError",
                       "NotADirectoryError", "IsADirectoryError", "ProcessLookupError", "NoSpaceError", "TooManyFilesError", "BusyError",
                       "ReadOnlyFileSystemError"}) p[n] = "OSError";
        for (auto n : {"ConnectionRefusedError", "ConnectionResetError", "BrokenPipeError"}) p[n] = "ConnectionError";
        p["SandboxViolationError"] = "PermissionError";
        p["ZeroDivisionError"] = "ArithmeticError";
        return p;
    }();
    return m;
}
bool err_is_a(std::string c, const std::string& anc) {
    if (anc.empty() || anc == "..." || anc == "Error" ) return err_parent().count(c) || c == "Error";
    for (int g = 0; g < 8 && !c.empty(); ++g) { if (c == anc) return true; auto it = err_parent().find(c); if (it == err_parent().end()) return false; c = it->second; }
    return false;
}

// Capability of a builtin library call (builtins.cpp `need(...)`).
const char* cap_of(const std::string& lib, const std::string& fn) {
    if (lib == "fs") return "fs";
    if (lib == "net") return "net";
    if (lib == "proc" || lib == "service") return "proc";
    if (lib == "os") {
        if (fn == "env" || fn == "setenv" || fn == "unsetenv" || fn == "environ") return "env";
        if (fn == "chdir") return "fs";
        if (fn == "processes" || fn == "kill" || fn == "signal") return "proc";
    }
    return nullptr;
}

struct Caps { bool fs = true, net = true, proc = true, env = true;
    bool get(const std::string& c) const { return c == "fs" ? fs : c == "net" ? net : c == "proc" ? proc : env; } };

bool is_builtin_var(const Node& n, const char* name) { return n.k == NK::Var && n.var.global() && n.s == name; }
const Node* callee(const Node& n) { return n.k == NK::Call && !n.kids.empty() ? n.kids[0].get() : nullptr; }

enum : int { NONE = 0, MAYBE = 1, DEF = 2 };
using State = std::map<int, int>;   // slot -> moved state, for the current function's own locals
State merge(const State& a, const State& b) {
    State r;
    auto get = [](const State& m, int k) { auto it = m.find(k); return it == m.end() ? int(NONE) : it->second; };
    for (auto* m : {&a, &b}) for (auto& [k, v] : *m) { (void)v; int x = get(a, k), y = get(b, k); r[k] = (x == DEF && y == DEF) ? DEF : (x || y) ? MAYBE : NONE; }
    return r;
}

struct Binding { std::string name; int line = 0; bool used = false; bool report = false; bool unique = false; bool list = false; bool cnst = false; };

struct Fn {
    const FnDecl* decl = nullptr;
    std::map<int, Binding*> slot;                      // current binding per slot
    std::vector<std::unique_ptr<Binding>> all;
    std::vector<std::set<std::string>> scopes;         // names per block, for shadowing
};

struct Checker {
    std::string_view src; std::string file; check_options opt;
    std::vector<diagnostic> out; std::set<std::tuple<int, std::string>> seen;
    std::vector<Fn> fns;
    Caps caps;
    std::vector<std::string> iter_stack;               // containers borrowed by enclosing range-for (by name+slot key)
    std::vector<std::pair<std::string, bool>> iter_list; // parallel: known list/map?
    std::vector<std::string> held;                     // forge::shared values whose with() is active
    std::map<std::string, std::set<std::string>> lock_edges; std::map<std::pair<std::string, std::string>, int> edge_line;
    std::map<std::string, const FnDecl*> globals_fn; std::map<std::string, int> global_assigns;
    bool quiet = false;                                // second loop pass: dedupe only
    std::vector<int> loop_depth_breaks;

    int col_of(int line, const std::string& needle) {
        if (line <= 0) return 0;
        std::size_t p = 0; for (int l = 1; l < line && p != std::string_view::npos; ++l) { p = src.find('\n', p); if (p != std::string_view::npos) ++p; }
        if (p == std::string_view::npos) return 0;
        auto e = src.find('\n', p); auto ln = src.substr(p, e == std::string_view::npos ? src.size() - p : e - p);
        if (!needle.empty()) { auto q = ln.find(needle); if (q != std::string_view::npos) return int(q) + 1; }
        auto q = ln.find_first_not_of(" \t"); return q == std::string_view::npos ? 1 : int(q) + 1;
    }
    void diag(L sev, const char* id, int line, std::string msg, const std::string& needle = {}) {
        if (sev == L::warning && !opt.warnings) return;
        if (!seen.emplace(line, std::string(id) + msg).second) return;
        out.push_back({sev, id, std::move(msg), file, line, col_of(line, needle)});
    }

    Fn& cur() { return fns.back(); }
    Binding* bind_of(const VarRef& v) {
        if (v.global() || v.depth >= int(fns.size())) return nullptr;
        auto& f = fns[fns.size() - 1 - v.depth]; auto it = f.slot.find(v.slot); return it == f.slot.end() ? nullptr : it->second;
    }
    static bool internal_name(const std::string& n) { return n.empty() || n[0] == '_' || n.find_first_of("$#.: <>") != std::string::npos || !(std::isalpha((unsigned char)n[0])); }
    void declare(const VarRef& v, const std::string& name, int line, bool report, bool cnst) {
        auto& f = cur();
        if (!internal_name(name) && !f.scopes.empty()) {
            for (std::size_t i = 0; i + 1 < f.scopes.size(); ++i) if (f.scopes[i].count(name)) { diag(L::warning, "shadow", line, "'" + name + "' shadows a variable declared in an enclosing block", name); break; }
            f.scopes.back().insert(name);
        }
        auto b = std::make_unique<Binding>(); b->name = name; b->line = line; b->report = report && !internal_name(name); b->cnst = cnst;
        f.slot[v.slot] = b.get(); f.all.push_back(std::move(b));
    }
    void end_fn() {
        for (auto& b : cur().all) if (b->report && !b->used) diag(L::warning, "unused-variable", b->line, "variable '" + b->name + "' is never read", b->name);
        fns.pop_back();
    }

    // ---------------------------------------------------------------- expressions
    void read_var(const Node& n, State& st) {
        if (auto* b = bind_of(n.var)) b->used = true;
        if (n.var.global() || n.var.depth != 0) return;
        auto it = st.find(n.var.slot); if (it == st.end() || !it->second) return;
        if (it->second == DEF && !quiet) diag(L::error, "use-after-move", n.line, "'" + n.s + "' is used after std::move (raises BorrowError)", n.s);
        else diag(L::warning, "use-after-move", n.line, "'" + n.s + "' may be used after std::move on some path", n.s);
    }
    std::string key(const Node& n) { return n.k == NK::Var ? (n.var.global() ? "g:" + n.s : n.s + "@" + std::to_string(fns.size() - 1 - std::max(0, n.var.depth)) + ":" + std::to_string(n.var.slot)) : std::string(); }

    void lambda(const Node& n, bool thread_ctx) {
        if (!n.fn || !n.fn->body) return;
        if (thread_ctx) thread_captures(*n.fn->body, 1);
        auto saved_iter = iter_stack; auto saved_list = iter_list; iter_stack.clear(); iter_list.clear();
        fn(*n.fn);
        iter_stack = saved_iter; iter_list = saved_list;
    }
    // captured (depth >= 1) mutable locals read or written inside a thread lambda
    void thread_captures(const Node& n, int depth) {
        if (n.k == NK::Lambda) { if (n.fn && n.fn->body) thread_captures(*n.fn->body, depth + 1); return; }
        if (n.k == NK::Var && !n.var.global() && n.var.depth >= depth) {
            int up = n.var.depth - depth;   // hops above the thread lambda's definer
            if (up < int(fns.size())) { auto& f = fns[fns.size() - 1 - up]; auto it = f.slot.find(n.var.slot);
                if (it != f.slot.end() && !it->second->cnst) diag(L::warning, "thread-capture", n.line, "thread body captures mutable local '" + n.s + "'; move, freeze or wrap it in forge::shared", n.s); }
        }
        for (auto& k : n.kids) if (k) thread_captures(*k, depth);
        if (n.extra) thread_captures(*n.extra, depth);
    }
    bool sendable_arg(const Node& a, State& st) {
        switch (a.k) {
        case NK::Nil: case NK::True: case NK::False: case NK::Int: case NK::Float: case NK::Str: case NK::Move: case NK::Lambda: return true;
        case NK::Call: { auto* c = callee(a); return c && (is_builtin_var(*c, "freeze") || is_builtin_var(*c, "forge::shared")); }
        case NK::Var: { auto* b = bind_of(a.var); (void)st; return b && b->cnst; }
        default: return false;
        }
    }
    static bool json_literal(const Node& a) {
        switch (a.k) {
        case NK::Nil: case NK::True: case NK::False: case NK::Int: case NK::Float: case NK::Str: return true;
        case NK::List: case NK::Map: for (auto& k : a.kids) if (!json_literal(*k)) return false; return true;
        default: return false;
        }
    }
    static bool non_json_literal(const Node& a) { return a.k == NK::Lambda || a.k == NK::Shell; }

    void call(const Node& n, State& st) {
        const Node* c = callee(n);
        std::size_t argc = n.kids.size() - 1;
        // ---- capability
        if (c && c->k == NK::Field && !c->kids.empty() && c->kids[0]->k == NK::Var && c->kids[0]->var.global()) {
            const std::string& lib = c->kids[0]->s;
            if (auto* cp = cap_of(lib, c->s); cp && !caps.get(cp))
                diag(L::error, "capability", n.line, lib + "." + c->s + " needs the '" + cp + "' capability, which this run denies (raises PermissionError)", c->s);
        }
        if (c && is_builtin_var(*c, "run") && !caps.proc) diag(L::error, "capability", n.line, "run() needs the 'proc' capability, which this run denies (raises PermissionError)", "run");
        // ---- format strings
        if (c && c->k == NK::Var && c->var.global() && (c->s == "std::format" || c->s == "std::println" || c->s == "std::print") && argc >= 1 && n.kids[1]->k == NK::Str)
            format_check(n.kids[1]->s, argc - 1, n.line);
        // ---- known script functions
        if (c && c->k == NK::Var && c->var.global() && globals_fn.count(c->s) && !global_assigns.count(c->s)) {
            auto* f = globals_fn[c->s]; std::size_t req = 0; for (auto& p : f->params) if (!p.dflt) ++req;
            if (argc < req || argc > f->params.size())
                diag(L::warning, "arg-count", n.line, c->s + "() takes " + (req == f->params.size() ? std::to_string(req) : std::to_string(req) + ".." + std::to_string(f->params.size())) + " argument(s), called with " + std::to_string(argc), c->s);
        }
        // ---- thread send
        bool thread = c && c->k == NK::Var && c->var.global() && (c->s == "forge::spawn" || c->s == "std::async" || c->s == "forge::async");
        if (thread) for (std::size_t i = 1; i < n.kids.size(); ++i) {
            auto& a = *n.kids[i];
            if (a.k == NK::Lambda) continue;
            if (!sendable_arg(a, st)) diag(L::warning, "thread-send", a.line, "value passed to a thread is not moved, frozen, const or forge::shared; it is deep-copied or raises RaceError on shared access", a.k == NK::Var ? a.s : std::string());
        }
        bool chan_send = c && c->k == NK::Field && (c->s == "send" || c->s == "push") && !c->kids.empty() && c->kids[0]->k == NK::Var && argc == 1;
        if (chan_send) { auto* b = bind_of(c->kids[0]->var); (void)b; }
        // ---- range-for structural mutation
        if (c && c->k == NK::Field && !c->kids.empty() && c->kids[0]->k == NK::Var) {
            static const std::set<std::string> mut = {"push_back", "emplace_back", "pop_back", "push_front", "pop_front", "insert", "emplace", "erase", "clear", "resize", "append", "extend", "remove", "pop", "push", "assign", "swap"};
            if (mut.count(c->s)) { auto k = key(*c->kids[0]); for (std::size_t i = 0; i < iter_stack.size(); ++i) if (!k.empty() && iter_stack[i] == k)
                diag(iter_list[i].second ? L::error : L::warning, "iter-mutation", n.line, "'" + c->kids[0]->s + "." + c->s + "' changes the size of the container a range-for is iterating" + (iter_list[i].second ? std::string(" (raises BorrowError)") : std::string()), c->s); }
        }
        // ---- lock order: x.with(fn)
        bool with = c && c->k == NK::Field && c->s == "with" && !c->kids.empty() && c->kids[0]->k == NK::Var && argc == 1;
        // ---- ProcessPool
        if (c && c->k == NK::Field && (c->s == "submit" || c->s == "map") && argc >= 1) pool_task(n);
        // ---- sandbox(opts, fn)
        bool sb = c && is_builtin_var(*c, "sandbox") && argc == 2 && n.kids[1]->k == NK::Map && n.kids[2]->k == NK::Lambda;
        // evaluate
        if (c) expr(*c, st);
        for (std::size_t i = 1; i < n.kids.size(); ++i) {
            auto& a = *n.kids[i];
            if (a.k == NK::Lambda) {
                if (with && i == 1) {
                    auto k = c->kids[0]->s;
                    for (auto& h : held) if (h != k) { lock_edges[h].insert(k); edge_line.emplace(std::pair{h, k}, n.line); }
                    held.push_back(k); lambda(a, false); held.pop_back(); continue;
                }
                if (sb && i == 2) {
                    Caps saved = caps; auto& m = *n.kids[1];
                    for (std::size_t j = 0; j + 1 < m.kids.size(); j += 2) if (m.kids[j]->k == NK::Str && m.kids[j + 1]->k == NK::False) {
                        auto& s = m.kids[j]->s; if (s == "fs") caps.fs = false; else if (s == "net") caps.net = false; else if (s == "proc") caps.proc = false; else if (s == "env") caps.env = false; }
                    lambda(a, false); caps = saved; continue;
                }
                lambda(a, thread); continue;
            }
            expr(a, st);
        }
    }
    void pool_task(const Node& n) {
        auto& s = *n.kids[1];
        if (s.k == NK::Str) {
            std::shared_ptr<const FnDecl> t;
            try { t = parse_program(s.s, "<task>"); } catch (const std::exception& e) { diag(L::error, "pool-task", n.line, std::string("ProcessPool task source does not parse: ") + e.what()); return; }
            const FnDecl* task = nullptr; bool reassigned = false;
            std::function<void(const Node&)> walk = [&](const Node& x) {
                if (x.k == NK::FnStmt && x.fn && x.fn->name == "task") task = x.fn.get();
                if (x.k == NK::Assign && !x.kids.empty() && x.kids[0]->k == NK::Var && x.kids[0]->s == "task") reassigned = true;
                if (x.k == NK::Block) for (auto& k : x.kids) if (k) walk(*k);
            };
            if (t && t->body) walk(*t->body);
            if (!task && !reassigned) diag(L::error, "pool-task", n.line, "ProcessPool task source does not define task(arg)");
            else if (task) { std::size_t req = 0; for (auto& p : task->params) if (!p.dflt) ++req;
                if (req > 1 || task->params.empty()) diag(L::error, "pool-task", n.line, "ProcessPool task(arg) must take exactly one argument (it takes " + std::to_string(task->params.size()) + ")"); }
        }
        if (n.kids.size() > 2) {
            auto& a = *n.kids[2];
            if (non_json_literal(a)) diag(L::error, "pool-json", a.line, "ProcessPool arguments must be JSON-representable; a function or shell value is not");
            else if ((a.k == NK::List || a.k == NK::Map) && !json_literal(a)) {
                bool bad = false; for (auto& k : a.kids) if (non_json_literal(*k)) bad = true;
                if (bad) diag(L::error, "pool-json", a.line, "ProcessPool arguments must be JSON-representable; this literal contains a function");
            }
        }
    }
    void format_check(const std::string& f, std::size_t nargs, int line) {
        std::size_t autos = 0; int maxi = -1; bool bad = false;
        for (std::size_t i = 0; i < f.size(); ++i) {
            if (f[i] == '{') { if (i + 1 < f.size() && f[i + 1] == '{') { ++i; continue; }
                auto e = f.find('}', i); if (e == std::string::npos) { bad = true; break; }
                std::string in = f.substr(i + 1, e - i - 1); auto colon = in.find(':'); std::string id = in.substr(0, colon);
                if (id.empty()) ++autos; else if (std::all_of(id.begin(), id.end(), ::isdigit)) maxi = std::max(maxi, std::stoi(id)); else bad = true;
                i = e;
            } else if (f[i] == '}') { if (i + 1 < f.size() && f[i + 1] == '}') ++i; else bad = true; }
        }
        if (bad) { diag(L::warning, "format", line, "malformed format string"); return; }
        std::size_t need = std::max<std::size_t>(autos, std::size_t(maxi + 1));
        if (need > nargs) diag(L::warning, "format", line, "format string needs " + std::to_string(need) + " argument(s), " + std::to_string(nargs) + " given");
        else if (need < nargs) diag(L::warning, "format", line, "format string uses " + std::to_string(need) + " argument(s), " + std::to_string(nargs) + " given; the rest are ignored");
    }

    void expr(const Node& n, State& st) {
        switch (n.k) {
        case NK::Var: read_var(n, st); return;
        case NK::Move:
            if (!n.kids.empty() && n.kids[0]->k == NK::Var) { auto& v = *n.kids[0]; read_var(v, st); if (!v.var.global() && v.var.depth == 0) st[v.var.slot] = DEF; return; }
            break;
        case NK::Copy:
            if (!n.kids.empty() && n.kids[0]->k == NK::Var) if (auto* b = bind_of(n.kids[0]->var); b && b->unique)
                diag(L::error, "unique-copy", n.line, "'" + n.kids[0]->s + "' is a unique_ptr and cannot be copied (raises BorrowError); use std::move", n.kids[0]->s);
            break;
        case NK::Lambda: lambda(n, false); return;
        case NK::Call: call(n, st); return;
        case NK::Shell: if (!caps.proc) diag(L::error, "capability", n.line, "shell blocks need the 'proc' capability, which this run denies (raises PermissionError)"); break;
        case NK::And: case NK::Or: case NK::Coalesce:
            if (n.kids.size() == 2) { expr(*n.kids[0], st); State b = st; expr(*n.kids[1], b); st = merge(st, b); return; }
            break;
        case NK::Ternary:
            if (n.kids.size() == 3) { expr(*n.kids[0], st); State a = st, b = st; expr(*n.kids[1], a); expr(*n.kids[2], b); st = merge(a, b); return; }
            break;
        default: break;
        }
        for (auto& k : n.kids) if (k) expr(*k, st);
    }

    // ---------------------------------------------------------------- statements. Returns false when control cannot fall through.
    bool block(const Node& n, State& st) {
        cur().scopes.emplace_back();
        bool live = true; bool warned = false;
        for (auto& k : n.kids) {
            if (!k) continue;
            if (!live && !warned && k->k != NK::FnStmt && k->k != NK::ClassStmt) { diag(L::warning, "unreachable", k->line, "unreachable code"); warned = true; }
            if (!live) { State dead = st; stmt(*k, dead); continue; }
            live = stmt(*k, st);
        }
        cur().scopes.pop_back();
        return live;
    }
    static bool has_exit(const Node& n, int loops) {   // a break of this loop, a return or a throw, not inside a nested lambda
        if (n.k == NK::Lambda || n.k == NK::FnStmt) return false;
        if (n.k == NK::Return || n.k == NK::Throw) return true;
        if (n.k == NK::Break && loops == 0) return true;
        if (n.k == NK::Call) { auto* c = callee(n); if (c && c->k == NK::Var && c->var.global() && (c->s == "std::exit" || c->s == "exit" || c->s == "std::abort")) return true; }
        int inner = loops + ((n.k == NK::While || n.k == NK::For) ? 1 : 0);
        for (auto& k : n.kids) if (k && has_exit(*k, n.k == NK::While || n.k == NK::For ? inner : loops)) return true;
        if (n.extra && has_exit(*n.extra, loops)) return true;
        return false;
    }
    bool stmt(const Node& n, State& st) {
        switch (n.k) {
        case NK::Block: return block(n, st);
        case NK::Let: {
            bool uniq = false, list = false, report = false;
            if (!n.kids.empty() && n.kids[0]) {
                auto& i = *n.kids[0];
                expr(i, st);
                if (auto* c = callee(i); c && is_builtin_var(*c, "forge::__own") && n.kids[0]->kids.size() > 1 && n.kids[0]->kids[1]->k == NK::Int && n.kids[0]->kids[1]->i == 2) uniq = true;
                if (i.k == NK::Move && !i.kids.empty() && i.kids[0]->k == NK::Var) if (auto* b = bind_of(i.kids[0]->var); b && b->unique) uniq = true;
                list = i.k == NK::List || i.k == NK::Map;
                report = i.k == NK::Nil || i.k == NK::Int || i.k == NK::Float || i.k == NK::Str || i.k == NK::True || i.k == NK::False || i.k == NK::List || i.k == NK::Map;
            }
            if (n.type && (n.type->name == "list" || n.type->name == "map" || n.type->name == "vector")) list = true;
            if (!n.var.global()) { declare(n.var, n.s, n.line, report, n.flag); if (auto* b = bind_of(VarRef{n.s, 0, n.var.slot})) { b->unique = uniq; b->list = list; } st.erase(n.var.slot); }
            else if (!n.kids.empty()) { /* global */ }
            return true;
        }
        case NK::Assign:
            if (n.kids.size() == 2) {
                expr(*n.kids[1], st);
                auto& t = *n.kids[0];
                if (t.k == NK::Var) { if (n.s != "=") read_var(t, st); if (!t.var.global() && t.var.depth == 0) st.erase(t.var.slot); }
                else expr(t, st);
            }
            return true;
        case NK::ExprStmt: for (auto& k : n.kids) if (k) expr(*k, st); return true;
        case NK::If: {
            expr(*n.kids[0], st);
            State a = st, b = st;
            bool la = n.kids.size() > 1 && n.kids[1] ? stmt(*n.kids[1], a) : true;
            bool lb = n.kids.size() > 2 && n.kids[2] ? stmt(*n.kids[2], b) : true;
            if (la && lb) st = merge(a, b); else if (la) st = a; else if (lb) st = b;
            return la || lb;
        }
        case NK::While: {
            bool infinite = !n.kids.empty() && n.kids[0] && n.kids[0]->k == NK::True;
            if (infinite && !(n.kids.size() > 1 && n.kids[1] && has_exit(*n.kids[1], 0)) && !(n.extra && has_exit(*n.extra, 0)))
                diag(L::warning, "infinite-loop", n.line, "loop has no break, return or throw; it only ends on a timeout, step limit or cancellation");
            loop(n, st, n.kids[0].get(), n.kids.size() > 1 ? n.kids[1].get() : nullptr, nullptr);
            return !infinite || has_exit(n, 0);
        }
        case NK::For: {
            expr(*n.kids[0], st);
            auto& srcn = *n.kids[0];
            std::string k = key(srcn); bool lst = false;
            if (srcn.k == NK::Var) if (auto* b = bind_of(srcn.var)) lst = b->list;
            declare(n.var, n.var.name, n.line, false, false);
            iter_stack.push_back(k); iter_list.push_back({k, lst});
            loop(n, st, nullptr, n.kids.size() > 1 ? n.kids[1].get() : nullptr, nullptr);
            iter_stack.pop_back(); iter_list.pop_back();
            return true;
        }
        case NK::FnStmt: if (n.fn) { fn(*n.fn); } return true;
        case NK::Return: for (auto& k : n.kids) if (k) expr(*k, st); return false;
        case NK::Throw: for (auto& k : n.kids) if (k) expr(*k, st); return false;
        case NK::Break: case NK::Continue: return false;
        case NK::Try: {
            State entry = st; bool live = n.kids[0] ? stmt(*n.kids[0], st) : true;
            State after = live ? st : State{}; bool any = live;
            std::vector<std::string> prior;
            for (std::size_t i = 1; i < n.kids.size(); ++i) {
                auto& c = *n.kids[i]; std::string t = c.s;
                for (auto& p : prior) if (p.empty() || p == "..." || (err_is_a(t, p) && err_parent().count(t)))
                    { diag(L::warning, "catch-order", c.line, "catch (" + (t.empty() ? std::string("...") : t) + ") is unreachable: an earlier catch (" + (p.empty() ? std::string("...") : p) + ") already handles it", t); break; }
                prior.push_back(t);
                State cs = merge(entry, st);
                if (!c.var.name.empty() && !c.var.global()) declare(c.var, c.var.name, c.line, false, false);
                bool lc = stmt(c, cs);
                if (lc) { after = any ? merge(after, cs) : cs; any = true; }
            }
            if (n.extra) { State fs = any ? after : merge(entry, st); stmt(*n.extra, fs); if (any) after = fs; }
            st = any ? after : st;
            return any;
        }
        case NK::Defer: for (auto& k : n.kids) if (k) { State d = st; stmt(*k, d); } return true;
        default:
            for (auto& k : n.kids) if (k) expr(*k, st);
            return true;
        }
    }
    void loop(const Node& n, State& st, const Node* cond, const Node* body, const Node*) {
        State entry = st;
        State s1 = st; if (cond) expr(*cond, s1);
        State before_body = s1;
        if (body) stmt(*body, s1);
        if (n.extra) stmt(*n.extra, s1);
        // second pass: state carried round the back edge, reported as warnings at most
        State s2 = merge(before_body, s1); for (auto& [k, v] : s2) if (v == DEF && !(before_body.count(k) && before_body[k] == DEF)) v = MAYBE;
        bool q = quiet; quiet = true;
        if (cond) expr(*cond, s2);
        if (body) stmt(*body, s2);
        if (n.extra) stmt(*n.extra, s2);
        quiet = q;
        st = merge(entry, merge(s1, s2));
        for (auto& [k, v] : st) if (v == DEF && !(entry.count(k) && entry[k] == DEF)) v = MAYBE;
    }
    static bool always_returns(const Node& n) {
        switch (n.k) {
        case NK::Return: case NK::Throw: return true;
        case NK::Block: for (auto& k : n.kids) if (k && always_returns(*k)) return true; return false;
        case NK::If: return n.kids.size() > 2 && n.kids[1] && n.kids[2] && always_returns(*n.kids[1]) && always_returns(*n.kids[2]);
        case NK::While: return !n.kids.empty() && n.kids[0] && n.kids[0]->k == NK::True && !has_exit_break(*n.kids[1]);
        case NK::Try: { if (!n.kids[0] || !always_returns(*n.kids[0])) return false; for (std::size_t i = 1; i < n.kids.size(); ++i) if (!always_returns(*n.kids[i])) return false; return true; }
        case NK::ExprStmt: if (!n.kids.empty()) if (auto* c = callee(*n.kids[0]); c && c->k == NK::Var && c->var.global() && (c->s == "std::exit" || c->s == "std::abort")) return true; return false;
        default: return false;
        }
    }
    static bool has_exit_break(const Node& n) {
        if (n.k == NK::Lambda || n.k == NK::FnStmt || n.k == NK::While || n.k == NK::For) return false;
        if (n.k == NK::Break) return true;
        for (auto& k : n.kids) if (k && has_exit_break(*k)) return true;
        return false;
    }
    void fn(const FnDecl& f) {
        fns.emplace_back(); cur().decl = &f; cur().scopes.emplace_back();
        for (auto& p : f.params) declare(VarRef{p.name, 0, p.slot}, p.name, f.line, false, p.cnst);
        State st;
        if (f.body) stmt(*f.body, st);
        static const std::set<std::string> no_value = {"", "void", "auto", "any"};
        if (!f.is_top && f.body && !no_value.count(f.ret_name) && f.name != "main" && !always_returns(*f.body))
            diag(L::warning, "missing-return", f.line, "function '" + f.name + "' is declared to return " + f.ret_name + " but can reach its end without returning (returns nil)", f.name);
        end_fn();
    }
    void collect(const Node& n) {
        if (n.k == NK::FnStmt && n.fn && n.var.global() && !n.fn->is_method) { if (globals_fn.count(n.fn->name)) global_assigns[n.fn->name]++; globals_fn[n.fn->name] = n.fn.get(); }
        if (n.k == NK::Assign && !n.kids.empty() && n.kids[0]->k == NK::Var && n.kids[0]->var.global()) global_assigns[n.kids[0]->s]++;
        if (n.k == NK::Let && n.var.global()) global_assigns[n.s]++;
        for (auto& k : n.kids) if (k) collect(*k);
        if (n.extra) collect(*n.extra);
        if (n.fn && n.fn->body && n.k == NK::Lambda) collect(*n.fn->body);
        if (n.k == NK::FnStmt && n.fn && n.fn->body) collect(*n.fn->body);
    }
    void lock_cycles() {
        std::set<std::string> done;
        for (auto& [a, outs] : lock_edges) for (auto& b : outs) if (lock_edges.count(b) && lock_edges[b].count(a) && a < b && !done.count(a + "|" + b)) {
            done.insert(a + "|" + b);
            int l1 = edge_line[{a, b}], l2 = edge_line[{b, a}];
            diag(L::warning, "lock-order", std::max(l1, l2), "forge::shared values '" + a + "' and '" + b + "' are locked in both orders (lines " + std::to_string(l1) + " and " + std::to_string(l2) + "); this can deadlock (DeadlockError under --debug)");
        }
    }
    void suppress() {
        // `// forge-check: ignore` or `// forge-check: ignore[id,id]` on the flagged line or the line above;
        // `// forge-check: ignore-file[id]` anywhere disables those ids for the file.
        std::vector<std::string> lines; std::size_t p = 0;
        while (p <= src.size()) { auto e = src.find('\n', p); lines.emplace_back(src.substr(p, e == std::string_view::npos ? std::string_view::npos : e - p)); if (e == std::string_view::npos) break; p = e + 1; }
        auto ids_of = [](const std::string& l, const char* tag, bool& all) -> std::set<std::string> {
            std::set<std::string> r; auto q = l.find(tag); if (q == std::string::npos) return r;
            q += std::string(tag).size();
            if (q < l.size() && l[q] == '[') { auto e = l.find(']', q); std::string s = l.substr(q + 1, e == std::string::npos ? std::string::npos : e - q - 1); std::size_t b = 0;
                while (b <= s.size()) { auto c = s.find(',', b); auto id = s.substr(b, c == std::string::npos ? std::string::npos : c - b); id.erase(0, id.find_first_not_of(' ')); id.erase(id.find_last_not_of(' ') + 1); if (!id.empty()) r.insert(id); if (c == std::string::npos) break; b = c + 1; } }
            else all = true;
            r.insert("\x01"); return r;
        };
        std::set<std::string> file_ids; bool file_all = false;
        for (auto& l : lines) { bool a = false; auto s = ids_of(l, "forge-check: ignore-file", a); if (!s.empty()) { file_ids.insert(s.begin(), s.end()); file_all |= a; } }
        std::erase_if(out, [&](const diagnostic& d) {
            if (d.id == "syntax") return false;
            if (file_all || file_ids.count(d.id)) return true;
            for (int ln : {d.line, d.line - 1}) if (ln >= 1 && ln <= int(lines.size())) {
                auto& l = lines[ln - 1]; if (l.find("forge-check: ignore-file") != std::string::npos) continue;
                bool all = false; auto s = ids_of(l, "forge-check: ignore", all); if (s.empty()) continue;
                if (all || s.count(d.id)) return true;
            }
            return false;
        });
    }
};

std::string jesc(const std::string& s) {
    std::string r; for (unsigned char c : s) { if (c == '"' || c == '\\') { r += '\\'; r += char(c); } else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); r += b; } else r += char(c); }
    return r;
}

} // namespace

std::vector<diagnostic> check_parsed(const FnDecl& top, std::string_view src, const std::string& file, const check_options& o) {
    Checker c; c.src = src; c.file = file; c.opt = o; c.caps = {o.allow_fs, o.allow_net, o.allow_proc, o.allow_env};
    if (top.body) c.collect(*top.body);
    c.fn(top);
    c.lock_cycles();
    c.suppress();
    std::stable_sort(c.out.begin(), c.out.end(), [](auto& a, auto& b) { return a.line < b.line || (a.line == b.line && a.col < b.col); });
    return std::move(c.out);
}

std::vector<diagnostic> check(std::string_view src, const std::string& file, const check_options& o) {
    std::shared_ptr<const FnDecl> top;
    try { top = parse_program(src, file); }
    catch (const syntax_error& e) { return {diagnostic{L::error, "syntax", e.what(), file, e.line, 0}}; }
    return check_parsed(*top, src, file, o);
}

bool has_errors(const std::vector<diagnostic>& d) { return std::any_of(d.begin(), d.end(), [](auto& x) { return x.severity == L::error; }); }

std::string to_text(const std::vector<diagnostic>& d) {
    std::string r;
    for (auto& x : d) r += x.file + ":" + std::to_string(x.line) + ":" + std::to_string(x.col) + ": " + (x.severity == L::error ? "error" : "warning") + "[" + x.id + "]: " + x.message + "\n";
    return r;
}
std::string to_json(const std::vector<diagnostic>& d) {
    std::string r = "[";
    for (std::size_t i = 0; i < d.size(); ++i) {
        auto& x = d[i]; if (i) r += ",";
        r += "{\"file\":\"" + jesc(x.file) + "\",\"line\":" + std::to_string(x.line) + ",\"col\":" + std::to_string(x.col) + ",\"severity\":\"" + (x.severity == L::error ? "error" : "warning") + "\",\"id\":\"" + jesc(x.id) + "\",\"message\":\"" + jesc(x.message) + "\"}";
    }
    return r + "]";
}

} // namespace forge::script
