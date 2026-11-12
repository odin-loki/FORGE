// Exceptions: catch by type, the <stdexcept> hierarchy, user exception classes, rethrow, unwinding.
#include <functional>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

struct AppError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct ParseError : AppError {
    int line;
    ParseError(const std::string& msg, int l) : AppError(msg), line(l) {}
};
struct Custom : std::exception {
    const char* what() const noexcept override { return "custom what"; }
};
struct Plain { int code; };
struct Guard {
    std::string name;
    explicit Guard(std::string n) : name(std::move(n)) {}
    ~Guard() { std::println("  ~Guard {}", name); }
};

std::string classify(const std::function<void()>& f) {
    try { f(); return "none"; }
    catch (const ParseError& e) { return "ParseError:" + std::string(e.what()) + "@" + std::to_string(e.line); }
    catch (const AppError& e) { return "AppError:" + std::string(e.what()); }
    catch (const std::out_of_range& e) { return "out_of_range"; }
    catch (const std::invalid_argument& e) { return "invalid_argument"; }
    catch (const std::length_error& e) { return "length_error"; }
    catch (const std::logic_error& e) { return "logic_error:" + std::string(e.what()); }
    catch (const std::overflow_error& e) { return "overflow_error"; }
    catch (const std::range_error& e) { return "range_error"; }
    catch (const std::runtime_error& e) { return "runtime_error:" + std::string(e.what()); }
    catch (const std::bad_optional_access& e) { return "bad_optional_access"; }
    catch (const std::exception& e) { return "exception:" + std::string(e.what()); }
    catch (int v) { return "int:" + std::to_string(v); }
    catch (double v) { return "double"; }
    catch (const std::string& s) { return "string:" + s; }
    catch (const Plain& p) { return "Plain:" + std::to_string(p.code); }
    catch (...) { return "unknown"; }
}

int checked_div(int a, int b) {
    if (b == 0) throw std::invalid_argument("division by zero");
    return a / b;
}
void level3() { Guard g("level3"); throw std::runtime_error("deep"); }
void level2() { Guard g("level2"); level3(); }
void level1() { Guard g("level1"); level2(); }
int parse_digit(char c) {
    if (c < '0' || c > '9') throw ParseError(std::string("bad digit '") + c + "'", 1);
    return c - '0';
}
int safe_parse(const std::string& s) try {
    int v = 0;
    for (char c : s) v = v * 10 + parse_digit(c);
    return v;
} catch (const ParseError&) {
    return -1;
}

int main() {
    std::println("{}", classify([] {}));
    std::println("{}", classify([] { throw std::runtime_error("rt"); }));
    std::println("{}", classify([] { throw std::logic_error("lg"); }));
    std::println("{}", classify([] { throw std::invalid_argument("ia"); }));
    std::println("{}", classify([] { throw std::out_of_range("oor"); }));
    std::println("{}", classify([] { throw std::length_error("len"); }));
    std::println("{}", classify([] { throw std::domain_error("dom"); }));
    std::println("{}", classify([] { throw std::overflow_error("ovf"); }));
    std::println("{}", classify([] { throw std::range_error("rng"); }));
    std::println("{}", classify([] { throw std::underflow_error("udf"); }));
    std::println("{}", classify([] { throw AppError("app"); }));
    std::println("{}", classify([] { throw ParseError("parse", 42); }));
    std::println("{}", classify([] { throw Custom(); }));
    std::println("{}", classify([] { throw 7; }));
    std::println("{}", classify([] { throw 2.5; }));
    std::println("{}", classify([] { throw std::string("text"); }));
    std::println("{}", classify([] { throw Plain{3}; }));
    std::println("{}", classify([] { throw 'c'; }));
    std::println("{}", classify([] { std::vector<int> v(2); v.at(5) = 1; }));
    std::println("{}", classify([] { std::string s = "abc"; s.at(10); }));
    std::println("{}", classify([] { std::string s = "abc"; s.substr(10); }));
    std::println("{}", classify([] { std::stoi("not a number"); }));
    std::println("{}", classify([] { std::optional<int> o; o.value(); }));
    std::println("{}", classify([] { checked_div(1, 0); }));

    // a thrown string literal is a const char*
    try { throw "literal"; }
    catch (const char* s) { std::println("cstr: {}", s); }

    // what() and catching by base
    try { checked_div(1, 0); }
    catch (const std::exception& e) { std::println("what: {}", e.what()); }
    try { throw ParseError("unexpected token", 9); }
    catch (const std::runtime_error& e) { std::println("as runtime_error: {}", e.what()); }
    try { throw Custom(); }
    catch (const std::exception& e) { std::println("virtual what: {}", e.what()); }

    // clause order: the first match wins
    try { throw ParseError("p", 1); }
    catch (const std::exception&) { std::println("base clause first"); }
    catch (...) { std::println("unreachable"); }

    // unwinding runs destructors innermost first, then the handler
    try { level1(); }
    catch (const std::exception& e) { std::println("caught {}", e.what()); }

    // rethrow keeps the dynamic type; a handler may throw something else
    try {
        try { throw ParseError("inner", 3); }
        catch (const std::exception& e) { std::println("inner sees {}", e.what()); throw; }
    } catch (const ParseError& e) { std::println("outer sees line {}", e.line); }
    try {
        try { throw std::runtime_error("low level"); }
        catch (const std::runtime_error& e) { throw AppError(std::string("wrapped: ") + e.what()); }
    } catch (const AppError& e) { std::println("{}", e.what()); }

    // nested try blocks, an unmatched inner try propagates outward
    try {
        try { throw 42; }
        catch (const std::exception&) { std::println("not for ints"); }
    } catch (int v) { std::println("outer int {}", v); }

    // flow control with try inside loops and functions
    int ok = 0, failed = 0;
    for (const std::string& text : {"12", "x", "305", "4y"}) {
        try {
            int v = 0;
            for (char c : text) v = v * 10 + parse_digit(c);
            ok += v;
        } catch (const ParseError&) { ++failed; continue; }
    }
    std::println("{} {} {} {}", ok, failed, safe_parse("123"), safe_parse("1z3"));
    auto attempt = [](int d) { try { return checked_div(100, d); } catch (const std::invalid_argument&) { return -1; } };
    std::println("{} {}", attempt(5), attempt(0));

    // exception state after the handler: execution continues normally
    std::vector<int> v{1, 2, 3};
    try { v.at(3) = 9; } catch (const std::out_of_range&) { v.push_back(4); }
    std::println("{} {}", v.size(), v.back());

    // noexcept functions and exceptions thrown from constructors
    struct Positive {
        int v;
        explicit Positive(int x) : v(x) { if (x <= 0) throw std::domain_error("not positive"); }
    };
    try { Positive a(1); Positive b(-1); std::println("unreachable {}", a.v + b.v); }
    catch (const std::domain_error& e) { std::println("ctor threw: {}", e.what()); }
    return 0;
}
