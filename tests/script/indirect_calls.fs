// Run-time side of the static indirect-call checks (tests/test_script_security.cpp, static_capability_*): every
// indirect route to a capability built-in that the checker warns about is denied by the interpreter when it runs
// inside forge::sandbox, and the same routes to capability-free code go through.
#include <functional>
#include <map>
#include <memory>
#include <vector>

std::string path = forge::fs::temp_file("forge-indirect-");
forge::fs::write(path, "data");

void attempt(const std::string& route, std::function<void()> f) {
    try { f(); std::println("ran: {}", route); }
    catch (const PermissionError& e) { std::println("denied: {}", route); }
}

// ---- capability-needing callables
std::string helper(const std::string& p) { return forge::fs::read(p); }
std::string outer(const std::string& p) { return helper(p); }
auto alias = forge::fs::read;
std::function<std::string(std::string)> fn = helper;
auto lam = [] { return forge::fs::read(path); };
std::vector<std::function<std::string()>> list{lam};
std::map<std::string, std::function<std::string()>> table{{"read", lam}};
struct Holder { std::function<std::string()> cb; };
Holder holder{lam};
struct Source { virtual std::string get() { return "memory"; } virtual ~Source() {} };
struct FileSource : Source { std::string get() override { return forge::fs::read(path); } };
std::string pull(Source& s) { return s.get(); }
std::string apply(std::function<std::string()> f) { return f(); }
std::function<std::string()> pick() { return lam; }

// ---- the same shapes over pure code
std::string pure_helper(const std::string& p) { return p + "!"; }
auto pure_lam = [] { return std::string("pure"); };
std::vector<std::function<std::string()>> pure_list{pure_lam};
Holder pure_holder{pure_lam};

const std::map<std::string, bool> no_fs{{"fs", false}};
forge::sandbox(no_fs, [] {
    attempt("function", [] { helper(path); });
    attempt("function calling a function", [] { outer(path); });
    attempt("alias of a built-in", [] { alias(path); });
    attempt("std::function", [] { fn(path); });
    attempt("lambda in a variable", [] { lam(); });
    attempt("lambda in a vector", [] { list[0](); });
    attempt("lambda in a vector, range-for", [] { for (auto& f : list) f(); });
    attempt("lambda in a map", [] { table["read"](); });
    attempt("lambda in a struct member", [] { holder.cb(); });
    attempt("virtual dispatch", [] { FileSource fsrc; pull(fsrc); });
    attempt("virtual dispatch through shared_ptr", [] { std::shared_ptr<Source> s = std::make_shared<FileSource>(); s->get(); });
    attempt("callback argument", [] { apply(lam); });
    attempt("callback returned by a function", [] { pick()(); });
    attempt("callback handed to std::for_each", [] { std::vector<std::string> ps{path}; std::for_each(ps.begin(), ps.end(), helper); });

    attempt("pure function", [] { pure_helper(path); });
    attempt("pure lambda in a vector", [] { pure_list[0](); });
    attempt("pure lambda in a struct member", [] { pure_holder.cb(); });
    attempt("pure virtual dispatch", [] { Source mem; pull(mem); });
    attempt("pure callback argument", [] { apply(pure_lam); });
});

std::println("after: {}", helper(path));
forge::fs::remove(path);
