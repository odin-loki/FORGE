#pragma once
// Grammar-constrained sampling with a GBNF-style grammar (the llama.cpp dialect, byte-level):
//
//   root   ::= object
//   object ::= "{" ws ( pair ( "," ws pair )* )? "}"
//   pair   ::= string ":" ws value
//   string ::= "\"" [^"\\]* "\""
//   ws     ::= [ \t\n]*
//
// Supported: rules, "literals" (\n \t \r \\ \" \xHH escapes), [classes] with ranges and ^negation,
// grouping, alternation |, postfix * + ?, comments (#). Classes match single bytes (ASCII ranges; UTF-8
// literals work byte-wise). Left recursion is rejected. The runtime keeps a set of pushdown stacks and
// masks every vocabulary token whose bytes cannot extend any stack (walked through a token trie).
#include <forge/core/error.hpp>
#include <forge/pack/tokenizer.hpp>
#include <cmath>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace forge::mind {

class grammar {
public:
    struct elem { bool is_rule = false; int rule = -1; bool neg = false; std::vector<std::pair<unsigned char, unsigned char>> ranges;
                  bool match(unsigned char c) const { bool in = false; for (auto [a, b] : ranges) if (c >= a && c <= b) { in = true; break; } return in != neg; } };
    using alt = std::vector<elem>;
    struct rule { std::string name; std::vector<alt> alts; };
    struct pos { int rule, alt, idx; auto operator<=>(const pos&) const = default; };
    using stack = std::vector<pos>;   // back() = next element to match

    explicit grammar(std::string_view src, std::string_view root = "root") : s_(src) {
        skip();
        while (i_ < s_.size()) {
            auto name = ident();
            skip(); if (s_.substr(i_, 3) != "::=") fail("expected ::="); i_ += 3;
            int id = rule_id(name);
            if (defined_.count(id)) fail("rule '" + name + "' defined twice");
            defined_.insert(id);
            rules_[std::size_t(id)].alts = alternatives(0);
            skip();
        }
        for (std::size_t r = 0; r < rules_.size(); ++r)
            if (!defined_.count(int(r))) raise(errc::invalid_argument, "grammar: undefined rule '{}'", rules_[r].name);
        auto it = ids_.find(std::string(root));
        if (it == ids_.end()) raise(errc::invalid_argument, "grammar: no '{}' rule", root);
        root_ = it->second;
        check_left_recursion();
    }
    const std::vector<rule>& rules() const { return rules_; }

    // initial stack set
    std::vector<stack> start() const { std::set<stack> out; stack s; push_alts(s, root_, out); return {out.begin(), out.end()}; }
    // advance every stack by one byte
    std::vector<stack> advance(const std::vector<stack>& in, unsigned char c) const {
        std::set<stack> out;
        for (auto& s : in) {
            if (s.empty()) continue;
            auto p = s.back();
            auto& e = rules_[std::size_t(p.rule)].alts[std::size_t(p.alt)][std::size_t(p.idx)];
            if (!e.match(c)) continue;
            stack n = s; n.back().idx++;
            settle(n, out);
        }
        return {out.begin(), out.end()};
    }
    static bool can_end(const std::vector<stack>& st) { for (auto& s : st) if (s.empty()) return true; return false; }

private:
    // pop finished frames and expand rule references until the top is a terminal (or the stack is empty)
    void settle(stack s, std::set<stack>& out) const {
        while (!s.empty()) {
            auto& p = s.back();
            auto& a = rules_[std::size_t(p.rule)].alts[std::size_t(p.alt)];
            if (std::size_t(p.idx) >= a.size()) { s.pop_back(); continue; }
            break;
        }
        if (s.empty()) { out.insert(s); return; }
        auto p = s.back();
        auto& e = rules_[std::size_t(p.rule)].alts[std::size_t(p.alt)][std::size_t(p.idx)];
        if (!e.is_rule) { out.insert(s); return; }
        s.back().idx++;                 // continuation after the referenced rule
        push_alts(s, e.rule, out);
    }
    void push_alts(const stack& base, int r, std::set<stack>& out) const {
        for (std::size_t a = 0; a < rules_[std::size_t(r)].alts.size(); ++a) {
            stack s = base; s.push_back({r, int(a), 0}); settle(std::move(s), out);
        }
    }
    [[noreturn]] void fail(const std::string& m) const {
        std::size_t line = 1; for (std::size_t k = 0; k < i_ && k < s_.size(); ++k) line += s_[k] == '\n';
        raise(errc::invalid_argument, "grammar line {}: {}", line, m);
    }
    void skip(bool newlines = true) {
        while (i_ < s_.size()) {
            char c = s_[i_];
            if (c == '#') { while (i_ < s_.size() && s_[i_] != '\n') ++i_; }
            else if (c == ' ' || c == '\t' || c == '\r' || (newlines && c == '\n')) ++i_;
            else break;
        }
    }
    std::string ident() {
        std::size_t b = i_;
        while (i_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[i_])) || s_[i_] == '-' || s_[i_] == '_')) ++i_;
        if (b == i_) fail("expected rule name");
        return std::string(s_.substr(b, i_ - b));
    }
    int rule_id(const std::string& n) {
        auto it = ids_.find(n); if (it != ids_.end()) return it->second;
        int id = int(rules_.size()); rules_.push_back({n, {}}); ids_[n] = id; return id;
    }
    int synth(std::vector<alt> alts) { int id = int(rules_.size()); rules_.push_back({std::format("_g{}", id), std::move(alts)}); defined_.insert(id); return id; }
    // a rule body ends at a newline that is followed by `name ::=` or at end of input (at depth 0)
    bool at_rule_end() {
        std::size_t save = i_;
        skip(true);
        if (i_ >= s_.size()) return true;
        std::size_t j = i_;
        while (j < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[j])) || s_[j] == '-' || s_[j] == '_')) ++j;
        std::size_t k = j; while (k < s_.size() && (s_[k] == ' ' || s_[k] == '\t')) ++k;
        bool def = j > i_ && s_.substr(k, 3) == "::=";
        i_ = save;
        return def;
    }
    std::vector<alt> alternatives(int depth) {
        std::vector<alt> alts(1);
        for (;;) {
            if (depth == 0 && at_rule_end()) break;
            skip(true);
            if (i_ >= s_.size()) { if (depth) fail("unclosed ("); break; }
            char c = s_[i_];
            if (c == ')') { if (!depth) fail("unexpected )"); break; }
            if (c == '|') { ++i_; alts.emplace_back(); continue; }
            std::vector<elem> seq;   // one primary may expand to several elements (literals)
            if (c == '"') {
                ++i_;
                while (i_ < s_.size() && s_[i_] != '"') { auto b = chr(); seq.push_back({false, -1, false, {{b, b}}}); }
                if (i_ >= s_.size()) fail("unterminated literal");
                ++i_;
            } else if (c == '[') {
                ++i_; elem e;
                if (i_ < s_.size() && s_[i_] == '^') { e.neg = true; ++i_; }
                while (i_ < s_.size() && s_[i_] != ']') {
                    auto a = chr(), b = a;
                    if (i_ + 1 < s_.size() && s_[i_] == '-' && s_[i_ + 1] != ']') { ++i_; b = chr(); }
                    e.ranges.push_back({a, b});
                }
                if (i_ >= s_.size()) fail("unterminated class");
                ++i_; seq.push_back(e);
            } else if (c == '(') {
                ++i_; auto inner = alternatives(depth + 1);
                if (i_ >= s_.size() || s_[i_] != ')') fail("expected )");
                ++i_; elem e; e.is_rule = true; e.rule = synth(std::move(inner)); seq.push_back(e);
            } else if (c == '.') {
                ++i_; seq.push_back({false, -1, true, {{'\n', '\n'}}});
            } else {
                elem e; e.is_rule = true; e.rule = rule_id(ident()); seq.push_back(e);
            }
            // postfix operators apply to the whole primary
            if (i_ < s_.size() && (s_[i_] == '*' || s_[i_] == '+' || s_[i_] == '?')) {
                char op = s_[i_++];
                int body = seq.size() == 1 && seq[0].is_rule ? seq[0].rule : synth({seq});
                elem b; b.is_rule = true; b.rule = body;
                int r = int(rules_.size()); rules_.push_back({std::format("_r{}", r), {}}); defined_.insert(r);
                elem self; self.is_rule = true; self.rule = r;
                if (op == '*') rules_[std::size_t(r)].alts = {{b, self}, {}};
                if (op == '+') rules_[std::size_t(r)].alts = {{b, self}, {b}};
                if (op == '?') rules_[std::size_t(r)].alts = {{b}, {}};
                seq = {self};
            }
            for (auto& e : seq) alts.back().push_back(e);
        }
        return alts;
    }
    unsigned char chr() {
        char c = s_[i_++];
        if (c != '\\') return static_cast<unsigned char>(c);
        if (i_ >= s_.size()) fail("bad escape");
        char e = s_[i_++];
        switch (e) {
        case 'n': return '\n'; case 't': return '\t'; case 'r': return '\r';
        case 'x': { if (i_ + 2 > s_.size()) fail("bad \\x"); auto v = std::stoi(std::string(s_.substr(i_, 2)), nullptr, 16); i_ += 2; return static_cast<unsigned char>(v); }
        default: return static_cast<unsigned char>(e);
        }
    }
    bool nullable(int r, std::set<int>& seen) const {
        if (seen.count(r)) return false; seen.insert(r);
        for (auto& a : rules_[std::size_t(r)].alts) {
            bool all = true;
            for (auto& e : a) if (!e.is_rule || !nullable(e.rule, seen)) { all = false; break; }
            if (all) return true;
        }
        return false;
    }
    void check_left_recursion() const {
        // r is left-recursive if r can reach itself through leading (nullable-prefixed) rule refs
        for (std::size_t r = 0; r < rules_.size(); ++r) {
            std::vector<int> work{int(r)}; std::set<int> seen;
            while (!work.empty()) {
                int x = work.back(); work.pop_back();
                for (auto& a : rules_[std::size_t(x)].alts) for (auto& e : a) {
                    if (!e.is_rule) break;
                    if (e.rule == int(r)) raise(errc::invalid_argument, "grammar: rule '{}' is left-recursive", rules_[r].name);
                    if (!seen.count(e.rule)) { seen.insert(e.rule); work.push_back(e.rule); }
                    std::set<int> s2; if (!nullable(e.rule, s2)) break;
                }
            }
        }
    }
    std::string_view s_;
    std::size_t i_ = 0;
    std::vector<rule> rules_;
    std::map<std::string, int> ids_;
    std::set<int> defined_;
    int root_ = 0;
};

// Per-generation constraint state: masks logits and advances on accepted tokens.
class grammar_constraint {
public:
    grammar_constraint(std::shared_ptr<const grammar> g, const pack::bpe_tokenizer& tok, int vocab, int eos)
        : g_(std::move(g)), eos_(eos) {
        stacks_ = g_->start();
        tokens_.resize(std::size_t(vocab));
        nodes_.emplace_back();
        for (int t = 0; t < vocab; ++t) {
            if (t == eos) continue;
            auto b = tok.decode_one(t);
            tokens_[std::size_t(t)] = b;
            if (b.empty()) continue;
            int n = 0;
            for (unsigned char c : b) {
                auto it = nodes_[std::size_t(n)].next.find(c);
                if (it == nodes_[std::size_t(n)].next.end()) { int m = int(nodes_.size()); nodes_[std::size_t(n)].next[c] = m; nodes_.emplace_back(); n = m; }
                else n = it->second;
            }
            nodes_[std::size_t(n)].tokens.push_back(t);
        }
    }
    bool finished() const { return grammar::can_end(stacks_) && stacks_.size() == 1; }
    bool can_end() const { return grammar::can_end(stacks_); }
    // set logits of tokens that cannot continue the grammar to -inf
    void apply(std::span<float> logits) const {
        std::vector<char> ok(logits.size(), 0);
        walk(0, stacks_, ok);
        if (eos_ >= 0 && std::size_t(eos_) < ok.size() && grammar::can_end(stacks_)) ok[std::size_t(eos_)] = 1;
        for (std::size_t i = 0; i < logits.size(); ++i) if (!ok[i]) logits[i] = -INFINITY;
    }
    // advance on a sampled token; returns false if the token was not allowed
    bool accept(int t) {
        if (t == eos_) return grammar::can_end(stacks_);
        auto st = stacks_;
        for (unsigned char c : tokens_.at(std::size_t(t))) { st = g_->advance(st, c); if (st.empty()) return false; }
        stacks_ = std::move(st);
        return true;
    }
private:
    struct node { std::map<unsigned char, int> next; std::vector<int> tokens; };
    void walk(int n, const std::vector<grammar::stack>& st, std::vector<char>& ok) const {
        for (int t : nodes_[std::size_t(n)].tokens) if (std::size_t(t) < ok.size()) ok[std::size_t(t)] = 1;
        for (auto [c, m] : nodes_[std::size_t(n)].next) {
            auto s2 = g_->advance(st, c);
            if (!s2.empty()) walk(m, s2, ok);
        }
    }
    std::shared_ptr<const grammar> g_;
    int eos_;
    std::vector<grammar::stack> stacks_;
    std::vector<std::string> tokens_;
    std::vector<node> nodes_;
};

} // namespace forge::mind
