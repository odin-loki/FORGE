// Catch-clause grammar and matching. Only `catch (...)` is a catch-all: every other clause names a type and handles
// that type (or a class derived from it) and nothing else. Regression: an unqualified built-in error name
// (`catch (const PermissionError& e)`) used to parse as a catch-all.
#include <stdexcept>
#include <functional>
#include <string>

void probe(const std::string& what, std::function<void()> f) {
    try { f(); std::println("{}: no error", what); }
    catch (const PermissionError& e) { std::println("{}: PermissionError", what); }
    catch (const forge::TimeoutError& e) { std::println("{}: TimeoutError", what); }
    catch (const std::out_of_range& e) { std::println("{}: out_of_range", what); }
    catch (const std::invalid_argument& e) { std::println("{}: invalid_argument", what); }
    catch (const std::logic_error& e) { std::println("{}: logic_error", what); }
    catch (const std::overflow_error& e) { std::println("{}: overflow_error", what); }
    catch (const OSError& e) { std::println("{}: OSError", what); }
    catch (const std::runtime_error& e) { std::println("{}: runtime_error", what); }
    catch (const std::exception& e) { std::println("{}: exception", what); }
    catch (int v) { std::println("{}: int {}", what, v); }
    catch (double v) { std::println("{}: double {}", what, v); }
    catch (bool v) { std::println("{}: bool {}", what, v); }
    catch (const std::string& v) { std::println("{}: string {}", what, v); }
    catch (...) { std::println("{}: ...", what); }
}

struct Plain { int code; };
struct AppError : std::runtime_error { using std::runtime_error::runtime_error; };
struct ParseError : AppError { int line = 0; ParseError(const std::string& m, int l) : AppError(m), line(l) {} };
struct Other : std::exception { const char* what() const noexcept override { return "other"; } };
namespace app { struct Fault : std::logic_error { using std::logic_error::logic_error; }; }
enum class Color { red, green };

// ---- a bare built-in name is a type, not a catch-all
try { throw std::runtime_error("rt"); }
catch (const PermissionError& e) { std::println("BUG: bare PermissionError caught a runtime_error"); }
catch (const std::runtime_error& e) { std::println("bare name does not widen: {}", e.what()); }
try { throw std::runtime_error("rt"); }
catch (PermissionError e) { std::println("BUG: by value"); }
catch (TimeoutError& e) { std::println("BUG: by reference"); }
catch (const OutOfMemoryError&) { std::println("BUG: unnamed"); }
catch (...) { std::println("by value / by reference / unnamed do not widen"); }
try { throw PermissionError("denied"); }
catch (const PermissionError& e) { std::println("bare name matches its own class: {}", e.what()); }
try { throw forge::SandboxViolationError("os"); }
catch (const PermissionError& e) { std::println("bare name matches a derived class"); }
try { throw PermissionError("p"); }
catch (const forge::PermissionError& e) { std::println("qualified name matches"); }
try { throw std::runtime_error("x"); }
catch (const NoSuchErrorClass& e) { std::println("BUG: unknown class name caught something"); }
catch (...) { std::println("an unknown class name matches nothing"); }

// ---- every kind of thrown value finds its own clause
probe("runtime_error", [] { throw std::runtime_error("x"); });
probe("logic_error", [] { throw std::logic_error("x"); });
probe("invalid_argument", [] { throw std::invalid_argument("x"); });
probe("out_of_range", [] { throw std::out_of_range("x"); });
probe("length_error", [] { throw std::length_error("x"); });
probe("overflow_error", [] { throw std::overflow_error("x"); });
probe("range_error", [] { throw std::range_error("x"); });
probe("exception", [] { throw std::exception(); });
probe("vector.at", [] { std::vector<int> v; v.at(3); });
probe("stoi", [] { std::stoi("zz"); });
probe("PermissionError", [] { throw PermissionError("x"); });
probe("TimeoutError", [] { throw TimeoutError("x"); });
probe("FileNotFoundError", [] { throw FileNotFoundError("x"); });
probe("int", [] { throw 42; });
probe("double", [] { throw 2.5; });
probe("bool", [] { throw true; });
probe("std::string", [] { throw std::string("text"); });
probe("string literal", [] { throw "literal"; });
probe("enum", [] { throw Color::green; });
probe("plain struct", [] { throw Plain{7}; });
probe("user runtime_error", [] { throw AppError("app"); });
probe("user derived", [] { throw ParseError("parse", 3); });
probe("user std::exception", [] { throw Other(); });
probe("namespaced user class", [] { throw app::Fault("f"); });
probe("nothing", [] {});

// ---- no arithmetic conversions between clauses
try { throw 1; } catch (double) { std::println("BUG: double caught int"); } catch (bool) { std::println("BUG: bool caught int"); } catch (int) { std::println("int is caught as int only"); }
try { throw 1.0; } catch (int) { std::println("BUG: int caught double"); } catch (double) { std::println("double is caught as double only"); }
try { throw "s"; } catch (const std::exception&) { std::println("BUG: exception caught a string"); } catch (const char* s) { std::println("const char* catches a string literal: {}", s); }
try { throw std::string("s"); } catch (std::string_view s) { std::println("string by value: {}", s); }

// ---- user classes: base catches derived, first matching clause wins, non-exception classes
try { throw ParseError("bad token", 12); }
catch (const ParseError& e) { std::println("derived first: {} at {}", e.what(), e.line); }
catch (const AppError& e) { std::println("BUG: base clause chosen over an earlier derived clause"); }
try { throw ParseError("bad token", 12); }
catch (const AppError& e) { std::println("base before derived wins: {}", e.what()); }
catch (const ParseError& e) { std::println("BUG: later clause"); }
try { throw ParseError("p", 1); } catch (const std::runtime_error& e) { std::println("std base of a user class: {}", e.what()); }
try { throw ParseError("p", 1); } catch (const std::logic_error& e) { std::println("BUG: unrelated std base"); } catch (const std::exception& e) { std::println("std::exception catches user exceptions"); }
try { throw Plain{9}; } catch (const std::exception& e) { std::println("BUG: std::exception caught a plain struct"); } catch (const Plain& p) { std::println("plain struct by its class: {}", p.code); }
try { throw Plain{9}; } catch (const AppError&) { std::println("BUG: unrelated class"); } catch (...) { std::println("unrelated class clause skipped"); }
try { throw app::Fault("ns"); } catch (const app::Fault& e) { std::println("namespaced class: {}", e.what()); }
try { throw Other(); } catch (const Other& e) { std::println("overridden what(): {}", e.what()); }

// ---- rethrow, nesting, function / lambda boundaries
int depth(int n) { if (n == 0) throw std::out_of_range("bottom"); return depth(n - 1) + 1; }
try { depth(20); } catch (const std::out_of_range& e) { std::println("through 20 frames: {}", e.what()); }
try {
    try { throw ParseError("inner", 5); }
    catch (const AppError& e) { std::println("inner handler: {}", e.what()); throw; }
} catch (const ParseError& e) { std::println("rethrown with its dynamic type: line {}", e.line); }
try {
    try { throw std::runtime_error("first"); }
    catch (const std::logic_error&) { std::println("BUG"); }
} catch (const std::runtime_error& e) { std::println("unmatched inner try propagates: {}", e.what()); }
try {
    try { throw std::runtime_error("first"); }
    catch (const std::runtime_error&) { throw std::logic_error("second"); }
} catch (const std::logic_error& e) { std::println("thrown from a handler: {}", e.what()); }
auto thrower = [](int k) { if (k > 1) throw std::invalid_argument("lambda"); return k; };
try { std::vector<int> v{1, 2, 3}; std::for_each(v.begin(), v.end(), thrower); }
catch (const std::invalid_argument& e) { std::println("through std::for_each and a lambda: {}", e.what()); }
int handled = 0;
for (int i = 0; i < 4; ++i) {
    try { if (i % 2) throw i; handled += 10; }
    catch (int v) { handled += v; continue; }
}
std::println("try in a loop with continue: {}", handled);
int f_ret() { try { throw 5; } catch (int v) { return v * 2; } return -1; }
std::println("return from a handler: {}", f_ret());

// ---- destructors run while unwinding, innermost first, before the handler
struct Guard { std::string n; Guard(std::string s) : n(s) {} ~Guard() { std::println("  ~Guard {}", n); } };
void unwind_inner() { Guard g("inner"); throw std::runtime_error("unwinding"); }
void unwind_outer() { Guard g("outer"); unwind_inner(); }
try { unwind_outer(); } catch (const std::exception& e) { std::println("handler after destructors: {}", e.what()); }

// ---- fatal errors pass through every clause: std::exit is not an exception
try { std::exit(7); }
catch (const std::exception& e) { std::println("BUG: std::exit caught as an exception"); }
catch (...) { std::println("BUG: std::exit caught by catch (...)"); }
std::println("BUG: still running after std::exit");
