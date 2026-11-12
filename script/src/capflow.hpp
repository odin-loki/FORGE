#pragma once
// Capability flow for the static checker (check.cpp; docs/scripting.md "Static checks", `capability`).
//
// The direct checks in check.cpp see a capability built-in only where it is named in the call (`fs.read(p)`). This
// pass follows callables: which functions, lambdas, methods and built-ins a value may be, and which capabilities
// (fs, net, proc, env) calling it may need. It is flow-insensitive and whole-program over one parsed script:
//
//   1. Value sets. Every local, global, field name and function return gets the set of callables it may hold (or
//      contain: a container or struct is its elements). Assignments, initialisers, arguments, returns, range-for,
//      container inserts, wrappers (std::move, std::bind, std::function, make_shared) and element accessors feed
//      the sets until nothing changes. Fields are tracked by name, methods are resolved through the classes a receiver may be
//      (every class that has the method when the receiver's class is not known: virtual dispatch).
//   2. Summaries. Each function gets the capabilities its body may need: built-ins it calls, summaries of the
//      functions it may call, callables it hands to a built-in that may call them. A parameter that the function
//      itself calls is kept symbolic, so `apply(pure_fn)` does not inherit what `apply(other_fn)` needs.
//   3. A callee nothing is known about (a value from eval or reflect, a parameter nobody passes anything to, a
//      member of an imported module) is "unknown": it may be any function or built-in whose address is taken
//      anywhere in the script, so it needs whatever those need (`analysis::escaped`).
//
// The result is one call_info per call node. check.cpp turns it into `capability` warnings where the call sits in
// a context that denies the capability. Everything here is "may": the run-time checks remain the enforcement.
#include "ast.hpp"
#include "shells/shell.hpp"
#include <array>
#include <cctype>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace forge::script::capflow {

enum : unsigned { FS = 1, NET = 2, PROC = 4, ENV = 8 };
inline constexpr int kCaps = 4;
inline const char* cap_name(int i) { static const char* n[] = {"fs", "net", "proc", "env"}; return n[i]; }

// Capability of a built-in library member (builtins.cpp `need(...)`), as a bit; 0 = none.
inline unsigned lib_cap(const std::string& lib, const std::string& fn) {
    if (lib == "fs") return FS;
    if (lib == "net") return NET;
    if (lib == "proc" || lib == "service") return PROC;
    if (lib == "os") {
        if (fn == "env" || fn == "setenv" || fn == "unsetenv" || fn == "environ") return ENV;
        if (fn == "chdir") return FS;
        if (fn == "processes" || fn == "kill" || fn == "signal") return PROC;
    }
    return 0;
}
// A global name without the leading `::` and `forge::` (the runtime resolves forge::fs::read as fs.read).
inline std::string plain(std::string n) {
    if (n.rfind("::", 0) == 0) n.erase(0, 2);
    if (n.rfind("forge::", 0) == 0) n.erase(0, 7);
    return n;
}
// Capability of a built-in named by a global variable, in any spelling: forge::fs::read, fs::read, forge::sh,
// run, std::getenv, ...
inline unsigned builtin_cap(const std::string& name) {
    std::string n = plain(name);
    if (n == "std::getenv" || n == "getenv") return ENV;
    if (n == "run" || n == "shell") return PROC;
    if (n == "remote" || n == "remote_call") return NET;
    if (auto p = n.find("::"); p != std::string::npos) return lib_cap(n.substr(0, p), n.substr(p + 2));
    return shells().find(n) ? unsigned(PROC) : 0u;   // forge::sh, forge::bash, forge::pwsh, ...
}

// Capability bits with one reason per bit ("helper -> forge::fs::read (line 5)").
struct mask {
    unsigned bits = 0;
    std::array<std::string, kCaps> why;
    bool add(unsigned b, const std::string& reason) {
        bool grew = false;
        for (int i = 0; i < kCaps; ++i) if ((b >> i & 1u) && !(bits >> i & 1u)) { bits |= 1u << i; why[std::size_t(i)] = reason.size() > 240 ? reason.substr(0, 237) + "..." : reason; grew = true; }
        return grew;
    }
    bool add(const mask& o, const std::string& prefix = {}) {
        bool grew = false;
        for (int i = 0; i < kCaps; ++i) if (o.bits >> i & 1u) grew |= add(1u << i, prefix + o.why[std::size_t(i)]);
        return grew;
    }
};

// What a value may be. A container, pair, optional, smart pointer or struct is the union of what it holds.
struct vset {
    std::set<const FnDecl*> fns;          // script functions, lambdas and methods
    std::set<const ClassDecl*> objs;      // instances of these script classes
    std::set<const ClassDecl*> classes;   // the classes themselves (calling one constructs it)
    mask caps;                            // built-ins that need a capability (the reason is the built-in's name)
    bool pure = false;                    // a built-in that needs none
    bool unknown = false;                 // something this pass cannot see
    bool merge(const vset& o) {
        bool grew = caps.add(o.caps);
        for (auto* f : o.fns) grew |= fns.insert(f).second;
        for (auto* c : o.objs) grew |= objs.insert(c).second;
        for (auto* c : o.classes) grew |= classes.insert(c).second;
        if (o.pure && !pure) { pure = true; grew = true; }
        if (o.unknown && !unknown) { unknown = true; grew = true; }
        return grew;
    }
    bool nothing() const { return fns.empty() && objs.empty() && classes.empty() && !caps.bits && !pure && !unknown; }
    bool callable() const { return !fns.empty() || !objs.empty() || !classes.empty() || caps.bits || unknown; }
};

struct call_info {
    std::string callee;        // how the callee is written, for messages
    mask direct;               // the callee names a capability built-in
    bool dotted = false;       // ... in the lib.fn spelling, which check.cpp already reports as an error
    mask through;              // capability built-ins the callee may be, through a value
    mask script;               // what the script functions the call may reach need (callbacks passed to them included)
    mask args;                 // what callables handed to a built-in that may call them need
    mask args_builtin;         // ... the capability built-ins among them only
    bool unknown = false;      // the callee is unresolved: it may need analysis::escaped
    unsigned sandbox = 0;      // forge::sandbox(<literal options>, fn): the capabilities the literal turns off
    bool is_sandbox = false;
};

struct analysis {
    std::map<const Node*, call_info> calls;
    mask escaped;              // what an unresolved callee may need: every address-taken function and built-in
};

class analyzer {
public:
    analysis run(const FnDecl& top) {
        if (top.body) prescan(*top.body);
        for (auto& [c, d] : class_list_) { (void)d; link_class(*c); }
        phase_ = 0;                                    // 1. value sets; receivers still empty after the first fixpoint
        for (bool by_name : {false, true}) {           //    are then resolved by method name (see member())
            by_name_ = by_name;
            for (int i = 0; i < 64; ++i) { changed_ = false; walk_fn(top); if (!changed_) break; }
        }
        phase_ = 1;                                    // 2. summaries
        for (int i = 0; i < 64; ++i) {
            changed_ = false; walk_fn(top);
            mask e = taken_caps_;
            for (auto* f : taken_) e.add(full(*f), label(*f) + " -> ");
            changed_ |= out_.escaped.add(e);
            if (!changed_) break;
        }
        phase_ = 2; walk_fn(top);                      // 3. one record per call
        return std::move(out_);
    }

private:
    using slot_key = std::pair<const FnDecl*, int>;
    analysis out_;
    int phase_ = 0; bool changed_ = false, by_name_ = false;
    // names
    std::set<std::string> defined_;                                   // globals this script defines or assigns
    std::map<std::string, const ClassDecl*> classes_;
    std::map<const ClassDecl*, int> class_list_;
    std::map<const ClassDecl*, std::set<const ClassDecl*>> subs_;      // transitive subclasses
    std::vector<std::string> using_ns_;
    // value sets
    std::map<slot_key, vset> locals_;
    std::map<std::string, vset> globals_, fields_;
    std::map<const FnDecl*, vset> rets_;
    std::set<const FnDecl*> taken_; mask taken_caps_;                  // used as a value somewhere
    // summaries
    std::map<const void*, mask> base_;                                // FnDecl / ClassDecl (field initialisers) -> needs
    std::map<const FnDecl*, std::set<std::size_t>> called_params_;     // parameters the function calls itself
    // walk state
    std::vector<const FnDecl*> chain_;
    const ClassDecl* cur_cls_ = nullptr;
    const void* owner_ = nullptr;                                     // whose summary the code being walked feeds
    const FnDecl* owner_fn_ = nullptr;

    // ------------------------------------------------------------------ names and classes
    void prescan(const Node& n) {
        if ((n.k == NK::Let || n.k == NK::FnStmt || n.k == NK::Import || n.k == NK::For || n.k == NK::Using) && n.var.global() && !n.var.name.empty()) defined_.insert(n.var.name);
        if (n.k == NK::Let && n.var.global()) defined_.insert(n.s);
        if (n.k == NK::FnStmt && n.fn && n.var.global()) globals_[n.var.name.empty() ? n.fn->name : n.var.name].fns.insert(n.fn.get());   // callable before its definition
        if (n.k == NK::Assign && !n.kids.empty() && n.kids[0] && n.kids[0]->k == NK::Var && n.kids[0]->var.global()) defined_.insert(n.kids[0]->s);
        if (n.k == NK::ClassStmt && n.cls) {
            defined_.insert(n.s); defined_.insert(n.var.name); defined_.insert(n.cls->name);
            classes_[n.s] = classes_[n.var.name] = classes_[n.cls->name] = n.cls.get();
            class_list_.emplace(n.cls.get(), 0);
            for (auto& m : n.cls->methods) if (m && m->body) prescan(*m->body);
            for (auto& [name, init] : n.cls->fields) { (void)name; if (init) prescan(*init); }
        }
        if (n.k == NK::Call && n.kids.size() == 2 && n.kids[0] && n.kids[0]->k == NK::Var && n.kids[0]->s == "forge::__using_namespace" && n.kids[1] && n.kids[1]->k == NK::Str)
            using_ns_.push_back(n.kids[1]->s);
        for (auto& k : n.kids) if (k) prescan(*k);
        if (n.extra) prescan(*n.extra);
        if (n.fn && n.fn->body && (n.k == NK::Lambda || n.k == NK::FnStmt)) {
            for (auto& p : n.fn->params) if (p.dflt) prescan(*p.dflt);
            prescan(*n.fn->body);
        }
    }
    const ClassDecl* class_named(const std::string& n) const { auto it = classes_.find(n); return it == classes_.end() ? nullptr : it->second; }
    void link_class(const ClassDecl& c) {
        std::vector<std::string> ups{c.base}; ups.insert(ups.end(), c.interfaces.begin(), c.interfaces.end());
        std::set<const ClassDecl*> seen;
        while (!ups.empty()) {
            auto name = ups.back(); ups.pop_back();
            auto* b = class_named(name); if (!b || !seen.insert(b).second) continue;
            subs_[b].insert(&c);
            ups.push_back(b->base); ups.insert(ups.end(), b->interfaces.begin(), b->interfaces.end());
        }
    }
    // The method a call of `name` on an instance of `c` runs: the most derived definition.
    const FnDecl* lookup(const ClassDecl* c, const std::string& name) const {
        for (int guard = 0; c && guard < 64; ++guard) {
            for (auto& m : c->methods) if (m && m->name == name) return m.get();
            c = class_named(c->base);
        }
        return nullptr;
    }
    // Instances of `c` or of any class derived from it (a declared type, `this`).
    void add_instances(vset& s, const ClassDecl* c) const {
        if (!c) return;
        s.objs.insert(c);
        if (auto it = subs_.find(c); it != subs_.end()) s.objs.insert(it->second.begin(), it->second.end());
    }
    // Classes named in a declared type as written ("const Base&", "std::shared_ptr<Base>").
    void add_declared(vset& s, const std::string& type) const {
        for (std::size_t i = 0; i < type.size();) {
            if (!(std::isalnum(static_cast<unsigned char>(type[i])) || type[i] == '_')) { ++i; continue; }
            std::size_t b = i;
            while (i < type.size() && (std::isalnum(static_cast<unsigned char>(type[i])) || type[i] == '_' || type[i] == ':')) ++i;
            add_instances(s, class_named(type.substr(b, i - b)));
        }
    }
    static std::string label(const FnDecl& f) { return f.name.empty() || f.name == "<lambda>" ? "lambda (line " + std::to_string(f.line) + ")" : f.name; }
    static std::string text_of(const Node& c) {
        switch (c.k) {
        case NK::Var: return c.s;
        case NK::Field: case NK::OptField: return c.kids.empty() || !c.kids[0] ? c.s : text_of(*c.kids[0]) + "." + c.s;
        case NK::Index: return c.kids.empty() || !c.kids[0] ? std::string("[]") : text_of(*c.kids[0]) + "[...]";
        case NK::Call: return c.kids.empty() || !c.kids[0] ? std::string("()") : text_of(*c.kids[0]) + "(...)";
        case NK::Lambda: return "lambda";
        case NK::Super: return c.s;
        case NK::Copy: case NK::Move: return c.kids.empty() || !c.kids[0] ? std::string() : text_of(*c.kids[0]);
        default: return "expression";
        }
    }

    // ------------------------------------------------------------------ locations
    vset* local(const VarRef& v) {
        if (v.global() || v.depth >= int(chain_.size()) || v.slot < 0) return nullptr;
        return &locals_[{chain_[chain_.size() - 1 - std::size_t(v.depth)], v.slot}];
    }
    bool is_builtin_global(const Node& n) const { return n.k == NK::Var && n.var.global() && !defined_.count(n.s); }
    // Mark what a value holds as "address taken": it may be what an unresolved callee turns out to be.
    const vset& take(const vset& s) {
        if (phase_ == 0) { for (auto* f : s.fns) changed_ |= taken_.insert(f).second; changed_ |= taken_caps_.add(s.caps); }
        return s;
    }
    void put(vset& into, const vset& s) { if (phase_ == 0) changed_ |= into.merge(take(s)); }
    // Store into what an lvalue (or an iterator / element expression) names. Does not walk `t`.
    void store_into(const Node& t, const vset& s) {
        if (phase_ != 0 || s.nothing()) return;
        switch (t.k) {
        case NK::Var: if (t.var.global()) put(globals_[t.s], s); else if (auto* l = local(t.var)) put(*l, s); break;
        case NK::Field: case NK::OptField: put(fields_[t.s], s); break;
        case NK::Index: case NK::Copy: case NK::Move: if (!t.kids.empty() && t.kids[0]) store_into(*t.kids[0], s); break;
        case NK::Call:
            if (t.kids.empty() || !t.kids[0]) break;
            if ((t.kids[0]->k == NK::Field || t.kids[0]->k == NK::OptField) && !t.kids[0]->kids.empty() && t.kids[0]->kids[0]) store_into(*t.kids[0]->kids[0], s);   // v.at(i), out.begin()
            else if (is_builtin_global(*t.kids[0])) for (std::size_t i = 1; i < t.kids.size(); ++i) if (t.kids[i]) store_into(*t.kids[i], s);                      // *it, std::get<0>(t)
            break;
        default: break;
        }
    }

    // ------------------------------------------------------------------ summaries
    void need(unsigned bit, const std::string& why) { if (phase_ == 1 && owner_) changed_ |= base_[owner_].add(bit, why); }
    void need(const mask& m, const std::string& prefix = {}) { if (phase_ == 1 && owner_) changed_ |= base_[owner_].add(m, prefix); }
    // What calling `f` may need when nothing is known about its arguments.
    mask full(const FnDecl& f, int depth = 0) {
        mask m; if (auto it = base_.find(&f); it != base_.end()) m = it->second;
        if (auto it = called_params_.find(&f); it != called_params_.end() && depth < 8)
            for (auto k : it->second) if (k < f.params.size()) if (auto l = locals_.find({&f, f.params[k].slot}); l != locals_.end()) m.add(of(l->second, depth + 1), "its parameter '" + f.params[k].name + "' -> ");
        return m;
    }
    // What calling a value may need.
    mask of(const vset& s, int depth = 0) {
        mask m; m.add(s.caps);
        for (auto* f : s.fns) m.add(full(*f, depth), label(*f) + " -> ");
        for (auto* c : s.classes) m.add(construct(*c, depth), c->name + " -> ");
        for (auto* c : s.objs) if (auto* op = lookup(c, "operator()")) m.add(full(*op, depth), c->name + "::operator() -> ");
        if (s.unknown) m.add(out_.escaped, "an unresolved callable -> ");
        return m;
    }
    const FnDecl* ctor_of(const ClassDecl& c, std::size_t argc) const {
        const FnDecl* one = nullptr;
        for (auto& m : c.methods) if (m) { if (m->name == "init/" + std::to_string(argc)) return m.get(); if (m->name == "init" || m->name.rfind("init/", 0) == 0) one = one ? one : m.get(); }
        return one;
    }
    mask construct(const ClassDecl& c, int depth = 0) {
        mask m;
        for (const ClassDecl* k = &c; k; k = class_named(k->base)) {
            if (auto it = base_.find(k); it != base_.end()) m.add(it->second);
            for (auto& x : k->methods) if (x && (x->name == "init" || x->name.rfind("init/", 0) == 0)) m.add(full(*x, depth));
            if (k->base.empty()) break;
        }
        return m;
    }

    // ------------------------------------------------------------------ functions and classes
    void walk_fn(const FnDecl& f) {
        auto* so = owner_; auto* sf = owner_fn_;
        chain_.push_back(&f); owner_ = &f; owner_fn_ = &f;
        for (auto& p : f.params) {
            if (phase_ == 0 && !p.tname.empty()) { vset t; add_declared(t, p.tname); if (!t.objs.empty()) changed_ |= locals_[{&f, p.slot}].merge(t); }
            if (p.dflt) { vset d = go(*p.dflt); put(locals_[{&f, p.slot}], d); }
        }
        if (f.body) go(*f.body);
        chain_.pop_back(); owner_ = so; owner_fn_ = sf;
    }
    void walk_class(const ClassDecl& c) {
        auto* sc = cur_cls_; auto* so = owner_; auto* sf = owner_fn_;
        cur_cls_ = &c;
        owner_ = &c; owner_fn_ = nullptr;
        for (auto& [name, init] : c.fields) if (init) { vset s = go(*init); put(fields_[name], s); }
        for (auto& m : c.methods) if (m) walk_fn(*m);
        cur_cls_ = sc; owner_ = so; owner_fn_ = sf;
    }

    // ------------------------------------------------------------------ expressions and statements
    vset var_value(const Node& n) {
        vset r;
        if (n.s == "this" && cur_cls_) { add_instances(r, cur_cls_); return r; }
        if (!n.var.global()) { if (auto* l = local(n.var)) r = *l; return r; }
        if (auto* c = class_named(n.s)) { r.classes.insert(c); return r; }
        if (defined_.count(n.s)) { if (auto it = globals_.find(n.s); it != globals_.end()) r = it->second; return r; }
        unsigned cap = builtin_cap(n.s);
        if (!cap && n.s.find("::") == std::string::npos) for (auto& ns : using_ns_) if ((cap = builtin_cap(ns + "::" + n.s))) break;
        if (cap) r.caps.add(cap, n.s); else r.pure = true;
        return r;
    }
    // Members of a value: fields by name, the methods of the classes it may be, and (for pairs, iterators and
    // other built-in wrappers) whatever the value holds.
    // `holds`: the member may be something the value holds (a field read, or pair.first / pair.second called).
    vset member(const vset& recv, const std::string& name, bool holds) {
        vset r;
        if (auto it = fields_.find(name); it != fields_.end()) r = it->second;
        if (!recv.objs.empty()) { for (auto* c : recv.objs) if (auto* m = lookup(c, name)) r.fns.insert(m); }
        else if (recv.nothing() && by_name_) for (auto& [c, d] : class_list_) { (void)d; for (auto& m : c->methods) if (m && m->name == name) r.fns.insert(m.get()); }   // receiver of unknown class
        if (holds || name == "first" || name == "second") { vset w = recv; w.classes.clear(); r.merge(w); }
        return r;
    }
    vset go(const Node& n) {
        vset r;
        switch (n.k) {
        case NK::Var: return var_value(n);
        case NK::Lambda: if (n.fn) { walk_fn(*n.fn); r.fns.insert(n.fn.get()); } return r;
        case NK::Call: case NK::Spawn: return call(n);
        case NK::Field: case NK::OptField: {
            if (n.kids.empty() || !n.kids[0]) return r;
            if (is_builtin_global(*n.kids[0]) && !class_named(n.kids[0]->s)) {   // lib.member
                const std::string lib = plain(n.kids[0]->s);
                if (unsigned cap = lib_cap(lib, n.s)) r.caps.add(cap, lib + "." + n.s);
                else if (lib == "reflect") r.unknown = true;
                else r.pure = true;
                return r;
            }
            return member(go(*n.kids[0]), n.s, true);
        }
        case NK::Index: if (!n.kids.empty() && n.kids[0]) r = go(*n.kids[0]); for (std::size_t i = 1; i < n.kids.size(); ++i) if (n.kids[i]) go(*n.kids[i]); return r;
        case NK::Copy: case NK::Move: case NK::Await: for (auto& k : n.kids) if (k) r.merge(go(*k)); return r;
        case NK::List: case NK::Map: case NK::And: case NK::Or: case NK::Coalesce: for (auto& k : n.kids) if (k) r.merge(go(*k)); return r;
        case NK::Ternary: for (std::size_t i = 0; i < n.kids.size(); ++i) if (n.kids[i]) { vset s = go(*n.kids[i]); if (i) r.merge(s); } return r;
        case NK::Shell: need(PROC, "a shell block (line " + std::to_string(n.line) + ")"); for (auto& k : n.kids) if (k) go(*k); return r;
        case NK::Let: {
            vset s; if (!n.kids.empty() && n.kids[0]) s = go(*n.kids[0]);
            if (n.var.global()) put(globals_[n.s], s); else if (auto* l = local(n.var)) put(*l, s);
            return r;
        }
        case NK::Assign: {
            if (n.kids.size() != 2 || !n.kids[0] || !n.kids[1]) break;
            vset s = go(*n.kids[1]);
            if (n.kids[0]->k != NK::Var) go(*n.kids[0]);
            store_into(*n.kids[0], take(s));
            return r;
        }
        case NK::For: {
            vset s; if (!n.kids.empty() && n.kids[0]) s = go(*n.kids[0]);
            if (n.var.global()) put(globals_[n.var.name], s); else if (auto* l = local(n.var)) put(*l, s);
            for (std::size_t i = 1; i < n.kids.size(); ++i) if (n.kids[i]) go(*n.kids[i]);
            if (n.extra) go(*n.extra);
            return r;
        }
        case NK::Using: {
            vset s; for (auto& k : n.kids) if (k) s.merge(go(*k));
            if (!n.var.name.empty()) { if (n.var.global()) put(globals_[n.var.name], s); else if (auto* l = local(n.var)) put(*l, s); }
            return r;
        }
        case NK::FnStmt:
            if (!n.fn) return r;
            { vset s; s.fns.insert(n.fn.get());
              if (phase_ == 0) { if (n.var.global()) changed_ |= globals_[n.var.name.empty() ? n.fn->name : n.var.name].merge(s); else if (auto* l = local(n.var)) changed_ |= l->merge(s); } }
            walk_fn(*n.fn);
            return r;
        case NK::ClassStmt: if (n.cls) walk_class(*n.cls); return r;
        case NK::Return:
            for (auto& k : n.kids) if (k) { vset s = go(*k); if (!chain_.empty()) put(rets_[chain_.back()], s); }
            return r;
        case NK::Import:
            need(FS, "import (line " + std::to_string(n.line) + ")");
            if (phase_ == 0 && !n.var.name.empty() && n.var.global()) { vset u; u.unknown = true; changed_ |= globals_[n.var.name].merge(u); }   // module members are not followed
            return r;
        default: break;
        }
        for (auto& k : n.kids) if (k) go(*k);
        if (n.extra) go(*n.extra);
        return r;
    }

    // Built-ins that keep or wrap a callable argument but never call it.
    static bool stores_only(const std::string& name, bool method) {
        static const std::set<std::string> methods = {"push_back", "emplace_back", "push_front", "emplace_front", "insert", "emplace", "push", "append", "assign",
            "set", "send", "put", "store", "reset", "swap", "emplace_hint", "insert_or_assign", "try_emplace", "push_range", "extend", "add", "fill", "resize",
            "count", "contains", "find", "erase", "remove"};
        static const std::set<std::string> globals = {"std::move", "std::forward", "std::ref", "std::cref", "std::bind", "std::bind_front", "std::function",
            "std::make_pair", "std::make_tuple", "std::make_optional", "std::pair", "std::tuple", "std::optional", "std::any", "std::vector", "std::deque", "std::list",
            "std::forward_list", "std::map", "std::unordered_map", "std::multimap", "std::set", "std::unordered_set", "std::multiset", "std::array", "std::swap",
            "std::get", "std::tie", "std::println", "std::print", "std::format", "std::to_string", "print", "write", "str", "repr", "type", "len", "freeze", "is_frozen",
            "weak", "channel", "shared", "dispose", "assert", "std::fill", "std::fill_n", "std::copy", "std::iota", "std::reverse", "std::find", "std::count"};
        if (method) return methods.count(name) > 0;
        auto p = plain(name);
        return p.rfind("__", 0) == 0 || globals.count(name) > 0 || globals.count(p) > 0;
    }
    // Built-in methods whose result may be something the receiver holds (v.at(i), m.find(k), opt.value(), ch.recv()).
    // Others (size, empty, count, ...) return plain data.
    static bool accessor(const std::string& name) {
        static const std::set<std::string> a = {"at", "front", "back", "get", "value", "value_or", "begin", "end", "rbegin", "rend", "cbegin", "cend", "find",
            "lower_bound", "upper_bound", "equal_range", "lock", "top", "pop", "pop_back", "pop_front", "recv", "receive", "try_recv", "take", "data", "next",
            "load", "extract", "items", "values", "keys", "copy", "clone", "with", "exchange", "release"};
        return a.count(name) > 0;
    }
    // Built-ins whose result is not built from their arguments and may be any callable.
    static bool yields_unknown(const std::string& name) {
        auto p = plain(name);
        return p == "eval" || p == "import" || p == "plugin" || p == "__unmarshal" || p.rfind("reflect::", 0) == 0 || p.rfind("reflect.", 0) == 0;
    }
    // forge::sandbox(opts, fn) with literal options: the capabilities the literal turns off.
    static unsigned sandbox_literal(const Node& opts) {
        unsigned off = 0;
        auto pair = [&](const Node* k, const Node* v) {
            if (!k || !v || k->k != NK::Str || v->k != NK::False) return;
            for (int i = 0; i < kCaps; ++i) if (k->s == cap_name(i)) off |= 1u << i;
        };
        auto flat = [&](const Node& l) { for (std::size_t j = 0; j + 1 < l.kids.size(); j += 2) pair(l.kids[j].get(), l.kids[j + 1].get()); };
        if (opts.k == NK::Map) flat(opts);                                               // {.fs = false} / map literal
        else if (opts.k == NK::List) {                                                   // {{"fs", false}, ...}
            for (auto& e : opts.kids) if (e && e->k == NK::List && e->kids.size() == 2) pair(e->kids[0].get(), e->kids[1].get());
        } else if (opts.k == NK::Call && opts.kids.size() == 4 && opts.kids[0] && opts.kids[0]->k == NK::Var && opts.kids[0]->s == "forge::__make_map" && opts.kids[3] && opts.kids[3]->k == NK::List)
            flat(*opts.kids[3]);                                                         // std::map<std::string, bool>{{"fs", false}}
        return off;
    }

    vset call(const Node& n) {
        vset result;
        if (n.kids.empty() || !n.kids[0]) return result;
        const Node& c = *n.kids[0];
        const std::size_t argc = n.kids.size() - 1;
        std::vector<vset> av(argc);
        for (std::size_t i = 0; i < argc; ++i) if (n.kids[i + 1]) av[i] = go(*n.kids[i + 1]);
        call_info info; info.callee = text_of(c);

        // ---- the callee
        vset target, recv; bool builtin = false, method = false; std::string bname;
        if (is_builtin_global(c) && !class_named(c.s)) {
            builtin = true; bname = c.s; target = var_value(c);
            if (target.caps.bits) { info.direct.add(target.caps); target.caps = {}; }
        } else if ((c.k == NK::Field || c.k == NK::OptField) && !c.kids.empty() && c.kids[0] && is_builtin_global(*c.kids[0]) && !class_named(c.kids[0]->s)) {
            builtin = true; bname = plain(c.kids[0]->s) + "." + c.s;
            if (unsigned cap = lib_cap(plain(c.kids[0]->s), c.s)) { info.direct.add(cap, bname); info.dotted = true; }
        } else if (c.k == NK::Super) {
            const ClassDecl* k = c.type ? class_named(c.type->name) : nullptr;
            if (auto* m = lookup(k ? class_named(k->base) : nullptr, c.s)) target.fns.insert(m);
            else builtin = true;
        } else if (c.k == NK::Field || c.k == NK::OptField) {
            if (!c.kids.empty() && c.kids[0]) recv = go(*c.kids[0]);
            target = member(recv, c.s, false);
            if (c.s != "first" && c.s != "second") target.objs.clear();                   // obj.m() is not obj()
            if (recv.unknown) target.unknown = true;                                      // a member of something unseen (a module, an eval result)
            if (!target.callable()) { builtin = true; method = true; bname = c.s; }       // a built-in method (v.push_back, s.size, ...)
        } else {
            target = go(c);
            if (target.pure && !target.callable()) { builtin = true; bname = info.callee; }   // an alias of a capability-free built-in
        }
        const bool own_param = owner_fn_ && c.k == NK::Var && !c.var.global() && c.var.depth == 0 && chain_.back() == owner_fn_ && param_index(*owner_fn_, c.var.slot) >= 0;

        // ---- script functions, methods, constructors
        vset everything = recv; for (auto& a : av) everything.merge(a);
        auto bind = [&](const FnDecl& f) {
            for (std::size_t i = 0; i < f.params.size() && i < argc; ++i) put(locals_[{&f, f.params[i].slot}], av[i]);
            if (auto it = rets_.find(&f); it != rets_.end()) result.merge(it->second);
        };
        std::vector<const FnDecl*> fns(target.fns.begin(), target.fns.end());
        for (auto* k : target.objs) if (auto* op = lookup(k, "operator()")) fns.push_back(op);
        for (auto* f : fns) {
            bind(*f);
            if (phase_ == 2) info.script.add(with_args(*f, n, av, true, nullptr), label(*f) + " -> ");
            if (phase_ == 1 && !own_param) {
                std::set<std::size_t> passed_on;   // own parameters handed to a parameter that `f` calls
                need(with_args(*f, n, av, false, owner_fn_ ? &passed_on : nullptr), label(*f) + " -> ");
                for (auto k : passed_on) changed_ |= called_params_[owner_fn_].insert(k).second;
            }
        }
        for (auto* k : target.classes) {
            result.objs.insert(k);
            if (auto* ctor = ctor_of(*k, argc)) bind(*ctor);
            else {   // aggregate: arguments initialise the fields in declaration order, base classes first
                std::vector<const ClassDecl*> line; for (const ClassDecl* b = k; b && line.size() < 64; b = class_named(b->base)) line.insert(line.begin(), b);
                std::size_t i = 0; for (auto* b : line) for (auto& [name, init] : b->fields) { (void)init; if (i < argc) put(fields_[name], av[i]); ++i; }
            }
            if (phase_ >= 1) { mask m = construct(*k); info.script.add(m, k->name + " -> "); if (phase_ == 1) need(m, k->name + " -> "); }
        }
        // ---- capability built-ins reached through a value
        if (target.caps.bits) { info.through.add(target.caps); if (!own_param) need(target.caps); }
        if (info.direct.bits) for (int i = 0; i < kCaps; ++i) if (info.direct.bits >> i & 1u) need(1u << i, info.direct.why[std::size_t(i)] + " (line " + std::to_string(n.line) + ")");
        // ---- nothing known about the callee
        // (an empty set only means "unresolved" once the sets are final: phase 0 must not feed on its own gaps)
        const bool unresolved = !builtin && (target.unknown || (phase_ >= 1 && !target.callable()));
        if (unresolved) { info.unknown = true; if (target.unknown) result.unknown = true; if (!own_param) need(out_.escaped, "an unresolved call (line " + std::to_string(n.line) + ") -> "); }
        if (own_param && phase_ == 1) changed_ |= called_params_[owner_fn_].insert(std::size_t(param_index(*owner_fn_, c.var.slot))).second;
        // ---- built-ins: the result may hold what went in; a method may store its arguments in the receiver;
        //      anything not known to only store a callable may call it
        if (builtin || unresolved || target.caps.bits) {
            // wrappers (std::move, std::bind, std::function, make_shared, *it, ...) and accessors pass on what they hold
            if (!builtin || (method ? accessor(bname) : stores_only(bname, false))) result.merge(everything);
            if (builtin && yields_unknown(bname)) result.unknown = true;
            if (method && !c.kids.empty() && c.kids[0]) { vset in; for (auto& a : av) in.merge(a); store_into(*c.kids[0], take(in)); }
            if (!(builtin && stores_only(bname, method))) {
                vset returned;
                for (std::size_t i = 0; i < argc; ++i) {
                    if (!av[i].callable()) continue;
                    take(av[i]);
                    for (auto* f : av[i].fns) {
                        for (auto& p : f->params) put(locals_[{f, p.slot}], everything);
                        if (auto it = rets_.find(f); it != rets_.end()) returned.merge(it->second);
                    }
                    if (phase_ == 0) continue;
                    const Node& an = *n.kids[i + 1];
                    const int pi = owner_fn_ && an.k == NK::Var && !an.var.global() && an.var.depth == 0 && chain_.back() == owner_fn_ ? param_index(*owner_fn_, an.var.slot) : -1;
                    if (pi >= 0) { if (phase_ == 1) changed_ |= called_params_[owner_fn_].insert(std::size_t(pi)).second; }
                    else need(of(av[i]), "a callable passed to " + info.callee + " (line " + std::to_string(n.line) + ") -> ");
                    // Not reported at this call: a literal lambda (its body is checked where it stands) and an
                    // iterator (v.begin()): the built-in passes the elements on, it does not call them.
                    if (an.k != NK::Lambda && !is_iterator(an)) { info.args.add(of(av[i])); info.args_builtin.add(av[i].caps); }
                }
                if (!returned.nothing()) { result.merge(returned); for (std::size_t i = 0; i < argc; ++i) if (n.kids[i + 1]) store_into(*n.kids[i + 1], take(returned)); }
            }
        }
        if (builtin && argc == 2 && n.kids[1] && (plain(bname) == "sandbox")) { info.is_sandbox = true; info.sandbox = sandbox_literal(*n.kids[1]); }
        if (phase_ == 2) out_.calls[&n] = std::move(info);
        return result;
    }
    static bool is_iterator(const Node& a) {
        static const std::set<std::string> names = {"begin", "end", "cbegin", "cend", "rbegin", "rend"};
        return a.k == NK::Call && !a.kids.empty() && a.kids[0] && a.kids[0]->k == NK::Field && names.count(a.kids[0]->s) > 0;
    }
    static int param_index(const FnDecl& f, int slot) {
        for (std::size_t i = 0; i < f.params.size(); ++i) if (f.params[i].slot == slot) return int(i);
        return -1;
    }
    // What calling `f` from this call site may need: its own needs plus what the arguments landing in the
    // parameters it calls need. `skip_literals`: leave out literal lambda arguments (reported where they stand).
    // `passed_on`: an argument that is the caller's own parameter stays symbolic; its index is noted there instead.
    mask with_args(const FnDecl& f, const Node& n, const std::vector<vset>& av, bool skip_literals, std::set<std::size_t>* passed_on) {
        mask m; if (auto it = base_.find(&f); it != base_.end()) m = it->second;
        auto it = called_params_.find(&f); if (it == called_params_.end()) return m;
        for (auto k : it->second) {
            if (k >= f.params.size()) continue;
            const std::string via = "its parameter '" + f.params[k].name + "' -> ";
            if (k < av.size() && n.kids[k + 1]) {
                const Node& an = *n.kids[k + 1];
                if (skip_literals && an.k == NK::Lambda) continue;
                const int pi = passed_on && an.k == NK::Var && !an.var.global() && an.var.depth == 0 && chain_.back() == owner_fn_ ? param_index(*owner_fn_, an.var.slot) : -1;
                if (pi >= 0) passed_on->insert(std::size_t(pi)); else m.add(of(av[k]), via);
            }
            else if (auto l = locals_.find({&f, f.params[k].slot}); l != locals_.end()) m.add(of(l->second), via);   // default argument
        }
        return m;
    }
};

inline analysis analyze(const FnDecl& top) { return analyzer().run(top); }

} // namespace forge::script::capflow
