#pragma once
// forge::script syntax tree. Immutable after parsing and resolution; shared (shared_ptr<const>) by the
// tree-walker, the bytecode compiler and every task that runs the same code.
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace forge::script {

struct TypeAnn {
    std::string name;                 // "int", "list", "Foo", "any"
    bool nullable = false;
    std::vector<TypeAnn> args;        // generics-lite: erased, documented, outer type checked
    std::string str() const {
        std::string s = name;
        if (!args.empty()) { s += "<"; for (std::size_t i = 0; i < args.size(); ++i) { if (i) s += ", "; s += args[i].str(); } s += ">"; }
        if (nullable) s += "?";
        return s;
    }
};

enum class NK {
    // expressions
    Nil, True, False, Int, Float, Str, Interp, Var, List, Map, Unary, Binary, And, Or, Coalesce,
    Call, Index, Field, OptField, Lambda, Shell, Is, Spawn, Await, Self, Super,
    Ternary,     // kids: cond, a, b
    Move,        // std::move(lvalue): kids[0] is a Var/Field/Index; leaves a moved-from marker behind
    Copy,        // copy-initialisation from an lvalue: value types are copied, unique_ptr raises BorrowError
    // statements
    Block, Let, Assign, ExprStmt, If, While, For, FnStmt, Return, Break, Continue, Throw, Try,
    ClassStmt, Using, Defer, Import,
};

enum class Op { Add, Sub, Mul, Div, Mod, Eq, Ne, Lt, Le, Gt, Ge, In, BitAnd, BitOr, BitXor, Shl, Shr, None };
inline Op op_of(std::string s) {
    if (s.size() > 1 && s.back() == '=' && s != "==" && s != "!=" && s != "<=" && s != ">=") s.pop_back();   // compound assignment
    if (s == "+") return Op::Add; if (s == "-") return Op::Sub; if (s == "*") return Op::Mul;
    if (s == "/") return Op::Div; if (s == "%") return Op::Mod; if (s == "==") return Op::Eq;
    if (s == "!=") return Op::Ne; if (s == "<") return Op::Lt; if (s == "<=") return Op::Le; if (s == ">") return Op::Gt; if (s == ">=") return Op::Ge;
    if (s == "&") return Op::BitAnd; if (s == "|") return Op::BitOr; if (s == "^") return Op::BitXor;
    if (s == "<<") return Op::Shl; if (s == ">>") return Op::Shr;
    return Op::None;
}
inline const char* op_name(Op o) {
    static const char* n[] = {"+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "in", "&", "|", "^", "<<", ">>", "?"};
    return n[int(o)];
}

struct FnDecl;
struct Node;
using NodeP = std::shared_ptr<const Node>;

// A resolved variable: local slot `slot` in the activation `depth` hops up, or a global by name.
struct VarRef { std::string name; int depth = -1, slot = -1; bool global() const { return depth < 0; } };

struct Node {
    NK k; int line = 0;
    std::string s;               // op / name / string literal / shell kind
    std::int64_t i = 0; double d = 0;
    bool flag = false;           // let: const; using: declares var; field: optional
    VarRef var;                  // Var, Let, For, FnStmt, Using, Import, catch var
    std::shared_ptr<TypeAnn> type;
    std::vector<NodeP> kids;
    std::shared_ptr<const FnDecl> fn;   // Lambda, FnStmt
    // Try: kids[0]=body, then catches (each a Block with s = type name, var), finally in `extra`
    // ClassStmt: extra holds the class description
    std::shared_ptr<const struct ClassDecl> cls;
    NodeP extra;                 // While: per-iteration step run before re-testing (for(;;) increment, do-while test)
    std::vector<int> locals;     // Block: slots declared here, released in reverse order at block exit (RAII)
};

struct Param { std::string name; int slot = 0; std::shared_ptr<TypeAnn> type; NodeP dflt; bool by_ref = false; char conv = 0; };   // conv: 'i' int, 'f' double (implicit arithmetic conversion)

struct FnDecl {
    std::string name; int line = 0;
    std::vector<Param> params;
    std::shared_ptr<TypeAnn> ret;
    NodeP body;
    int nslots = 0;
    bool is_method = false;      // slot 0 = self
    bool is_top = false;         // program / module / REPL entry
    bool has_main = false;       // program defines int main(): run it after the top-level statements
    std::string file;
};

struct ClassDecl {
    std::string name, base;
    std::vector<std::string> interfaces;
    bool is_struct = false, is_dynamic = false, is_interface = false;
    std::vector<std::pair<std::string, NodeP>> fields;     // name, default (may be null)
    std::vector<std::shared_ptr<const FnDecl>> methods;
    std::vector<std::string> required;                     // interface method names
    std::vector<std::string> statics;                      // static data members (kept on the class)
    VarRef var;
};

struct Program { std::shared_ptr<const FnDecl> top; };

} // namespace forge::script
