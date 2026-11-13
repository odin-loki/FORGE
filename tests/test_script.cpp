// forge::script: language, runtime, ARC/cycles, limits, OS bindings. Every program runs through both the
// tree-walker and the bytecode VM and must produce identical output.
#include <forge/script/script.hpp>
#include <forge/script/check.hpp>
#include "check.hpp"
#include <cstdio>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstdint>

using namespace forge;
TEST_PROCESS_WORKER(script::run_process_worker_if_requested);   // ProcessPool workers re-run this program on Windows (mem_limit_children)

namespace {
struct Out { std::string text; script::result r; };
Out run1(const std::string& src, bool vm, script::options o = {}) {
    Out out;
    o.use_vm = vm; o.jit_threshold = 1;
    o.out = [&](std::string_view s) { out.text += s; };
    script::engine e(o);
    out.r = e.run(src, "<test>");
    if (!out.r.ok) out.text += "!" + out.r.error_type;
    return out;
}
// Runs on both paths; returns the shared output, flags any divergence.
std::string run(const std::string& src, script::options o = {}) {
    auto a = run1(src, false, o), b = run1(src, true, o);
    CHECK_MSG(a.text == b.text, "tree vs vm differ:\n--- tree\n%s\n--- vm\n%s\n--- error tree: %s\n--- error vm: %s", a.text.c_str(), b.text.c_str(), a.r.error.c_str(), b.r.error.c_str());
    if (!b.r.ok && b.text.find('!') == std::string::npos) std::fprintf(stderr, "%s\n", b.r.error.c_str());
    return b.text;
}
#define EXPECT(src, want) do { auto got_ = run(src); CHECK_MSG(got_ == want, "got:\n%s\nwant:\n%s", got_.c_str(), std::string(want).c_str()); } while (0)
}

TEST(arithmetic_and_types) {
    EXPECT("std::println(\"{} {} {} {} {}\", 1 + 2 * 3, 7 / 2, 7 % 3, 7.0 / 2, 2.5 * 2);", "7 3 1 3.5 5\n");
    EXPECT("std::println(\"{} {} {}\", -7 / 2, -7 % 2, 1 << 4 | 1);", "-3 -1 17\n");
    EXPECT("std::string a = \"a\"; std::cout << a + \"b\" << ' ' << std::boolalpha << (1 < 2) << std::endl;", "ab true\n");
    EXPECT("long long x = 9223372036854775807; x + 1;", "!OverflowError");
    EXPECT("int z = 0; std::println(\"{}\", 1 / z);", "!ArithmeticError");
    EXPECT("int x = 3.5;", "!SyntaxError");
    EXPECT("int x = static_cast<int>(3.9); double d = x; std::println(\"{} {}\", x, d / 2);", "3 1.5\n");
    EXPECT("std::println(\"{:>5}|{:<4}|{:.3f}|{:x}|{{}}\", 42, \"ab\", 3.14159, 255);", "   42|ab  |3.142|ff|{}\n");
    EXPECT("std::cout << std::fixed << std::setprecision(2) << 3.14159 << \" \" << 2.0 << '\\n';", "3.14 2.00\n");
    EXPECT("std::println(\"{} {} {}\", std::to_string(42), std::stoi(\"-17\"), std::stod(\"2.5\") * 2);", "42 -17 5\n");
    EXPECT("std::stoi(\"x\");", "!ValueError");
}

TEST(variables_scopes_lambdas) {
    EXPECT("int x = 1;\nauto f = [&]() { return x + 1; };\nx = 10;\nstd::println(\"{}\", f());", "11\n");
    EXPECT("int x = 1;\nauto f = [x]() { return x + 1; };\nx = 10;\nstd::println(\"{}\", f());", "2\n");
    EXPECT("auto counter() { int n = 0; return [=]() mutable { n += 1; return n; }; }\nauto c = counter(); c(); c();\nstd::println(\"{}\", c());", "3\n");
    EXPECT("const int k = 1;\nk = 2;", "!SyntaxError");
    EXPECT("std::println(\"{}\", undefined_name);", "!NameError");
    EXPECT("int s = 0;\nfor (int i = 0; i < 5; ++i) { int sq = i * i; s += sq; }\nstd::println(\"{}\", s);", "30\n");
}

TEST(control_flow) {
    EXPECT("int i = 0;\nwhile (true) { ++i; if (i > 4) break; }\nstd::println(\"{}\", i);", "5\n");
    EXPECT("std::vector<int> o;\nfor (int i = 0; i < 6; i++) { if (i % 2 == 0) continue; o.push_back(i); }\nstd::println(\"{}\", o);", "[1, 3, 5]\n");
    EXPECT("int i = 0; do { i += 2; } while (i < 7);\nstd::println(\"{}\", i);", "8\n");
    EXPECT("std::map<std::string, int> m{{\"b\", 2}, {\"a\", 1}};\nfor (const auto& [k, v] : m) std::println(\"{} {}\", k, v);", "a 1\nb 2\n");
    EXPECT("long long f(int n) { return n <= 1 ? 1 : n * f(n - 1); }\nstd::println(\"{}\", f(20));", "2432902008176640000\n");
    EXPECT(R"CPP(
for (int i = 0; i < 4; ++i) {
    switch (i) {
        case 0: std::print("zero "); break;
        case 1:
        case 2: std::print("small "); break;
        default: std::print("big ");
    }
}
std::println("");
)CPP", "zero small small big \n");
    EXPECT("int main() { std::println(\"in main\"); return 0; }", "in main\n");
}

TEST(strings) {
    EXPECT(R"CPP(
std::string s = "hello world";
std::println("{} {} {} {}", s.size(), s.substr(6), s.find("o"), s.find("zz") == std::string::npos);
s += "!"; s.push_back('?');
std::string t = s; t[0] = 'H';
std::println("{} {} {}", s, t, s.starts_with("hell"));
)CPP", "11 world 4 true\nhello world!? Hello world!? true\n");
    EXPECT("std::string s = \"abc\"; std::reverse(s.begin(), s.end()); std::println(\"{} {}\", s, std::string(3, 'x'));", "cba xxx\n");
    EXPECT("std::string s = \"ab\"; s.at(5);", "!IndexError");
}

TEST(collections) {
    EXPECT(R"CPP(
std::vector<int> v{5, 3, 1, 4};
std::sort(v.begin(), v.end());
v.push_back(9);
auto it = std::find(v.begin(), v.end(), 4);
std::println("{} {} {} {}", v, v.size(), it - v.begin(), std::accumulate(v.begin(), v.end(), 0));
std::ranges::sort(v, [](int a, int b) { return a > b; });
std::println("{} {} {}", v, v.front(), v.back());
)CPP", "[1, 3, 4, 5, 9] 5 2 22\n[9, 5, 4, 3, 1] 9 1\n");
    EXPECT("std::vector<int> a{1, 2};\nauto b = a;\nb.push_back(3);\nstd::println(\"{} {}\", a, b);", "[1, 2] [1, 2, 3]\n");
    EXPECT("std::vector<int> v(3, 7);\nstd::println(\"{}\", v);\nv.at(3);", "[7, 7, 7]\n!IndexError");
    EXPECT(R"CPP(
std::unordered_map<std::string, int> m;
m["x"] = 1; m["y"] += 5; m.insert({"z", 3});
std::println("{} {} {} {}", m.size(), m.contains("y"), m.count("q"), m.at("y"));
m.erase("x");
std::println("{}", m.size());
)CPP", "3 true 0 5\n2\n");
    EXPECT("std::map<std::string, int> m; m.at(\"nope\");", "!KeyError");
    EXPECT("struct Job { std::string n; int c; };\nstd::vector<Job> js{{\"a\", 1}, {\"b\", 2}};\nstd::vector<std::vector<int>> g{{1, 2}, {3}};\nstd::println(\"{} {} {}\", js[1].n, js[0].c + js[1].c, g[1]);", "b 3 [3]\n");
    EXPECT("std::set<int> s{3, 1, 3, 2};\nfor (int x : s) std::print(\"{} \", x);\nstd::println(\"{}\", s.size());", "1 2 3 3\n");
    EXPECT("auto p = std::make_pair(1, std::string(\"a\"));\nauto [n, w] = p;\nstd::println(\"{} {} {}\", n, w, p.first);", "1 a 1\n");
    EXPECT("std::optional<int> o;\nstd::println(\"{} {}\", o.has_value(), o.value_or(7));\no = 3;\nstd::println(\"{}\", *o);", "false 7\n3\n");
}

TEST(classes_inheritance_operators) {
    EXPECT(R"CPP(
class Animal {
public:
    explicit Animal(std::string n) : name_(std::move(n)) {}
    virtual ~Animal() = default;
    virtual std::string speak() const { return name_ + " makes a sound"; }
    const std::string& name() const { return name_; }
protected:
    std::string name_;
};
class Dog : public Animal {
public:
    using Animal::Animal;
    std::string speak() const override { return Animal::speak() + " (woof)"; }
};
Dog d("rex");
const Animal& a = d;
std::println("{} | {}", a.speak(), d.name());
)CPP", "rex makes a sound (woof) | rex\n");
    EXPECT("struct P { int x = 0; int y = 0; };\nP a{1, 2};\nP b = a;\nb.x = 9;\nstd::println(\"{} {}\", a.x, b.x);", "1 9\n");
    EXPECT(R"CPP(
struct V {
    int x, y;
    V operator+(const V& o) const { return {x + o.x, y + o.y}; }
    bool operator==(const V&) const = default;
    auto operator<=>(const V& o) const { return x <=> o.x; }
};
V a{1, 2}, b{3, 4};
V c = a + b;
std::println("{} {} {} {}", c.x, c.y, a == V{1, 2}, a < b);
)CPP", "4 6 true true\n");
    EXPECT("struct S { virtual int f() = 0; };\nS s;", "!TypeError");
    EXPECT(R"CPP(
struct Shape { virtual double area() const = 0; virtual ~Shape() = default; };
struct Sq : Shape { double s; explicit Sq(double v) : s(v) {} double area() const override { return s * s; } };
std::vector<std::unique_ptr<Shape>> v;
v.push_back(std::make_unique<Sq>(2.0));
v.emplace_back(std::make_unique<Sq>(3.0));
double t = 0;
for (const auto& p : v) t += p->area();
std::println("{}", t);
)CPP", "13\n");
    EXPECT("class C { public: static int count; C() { ++count; } };\nint C::count = 0;\nC a; C b;\nstd::println(\"{}\", C::count);", "2\n");
}

TEST(exceptions) {
    EXPECT("try { throw std::invalid_argument(\"bad\"); } catch (const std::invalid_argument& e) { std::println(\"caught {}\", e.what()); }", "caught bad\n");
    EXPECT("try { std::vector<int> v; v.at(1); } catch (const std::out_of_range&) { std::println(\"oor\"); }", "oor\n");
    EXPECT(R"CPP(
class MyErr : public std::runtime_error { public: using std::runtime_error::runtime_error; };
try { throw MyErr("m"); } catch (const std::exception& e) { std::println("{}", e.what()); }
)CPP", "m\n");
    EXPECT("try { try { throw std::runtime_error(\"x\"); } catch (...) { std::println(\"inner\"); throw; } } catch (const std::runtime_error& e) { std::println(\"outer {}\", e.what()); }", "inner\nouter x\n");
    EXPECT("throw std::runtime_error(\"plain\");", "!RuntimeError");
}

TEST(ownership_moves_and_borrows) {
    EXPECT("auto p = std::make_unique<int>(5);\nauto q = std::move(p);\nstd::println(\"{}\", *q);\nstd::println(\"{}\", *p);\n// forge-check: ignore-file", "5\n!BorrowError");
    EXPECT("std::vector<int> a{1, 2};\nauto b = std::move(a);\nstd::println(\"{}\", b);\na.size();\n// forge-check: ignore-file", "[1, 2]\n!BorrowError");
    EXPECT("auto s = std::make_shared<std::vector<int>>();\nauto t = s;\nt->push_back(1);\nstd::println(\"{} {}\", s->size(), t->size());", "1 1\n");
    EXPECT("struct N { int v = 0; };\nauto p = std::make_shared<N>();\nstd::weak_ptr<N> w = p;\nstd::println(\"{}\", w.expired());\np = nullptr;\nstd::println(\"{}\", w.expired());", "false\ntrue\n");
    EXPECT("auto p = std::make_unique<int>(1);\nauto q = p;\n// forge-check: ignore-file", "!BorrowError");
}

TEST(namespaces_enums_templates) {
    EXPECT(R"CPP(
namespace geo { constexpr double pi = 3.5; double twice(double x) { return 2 * x; } }
enum class Color { Red, Green, Blue };
template <typename T> T biggest(const std::vector<T>& v) { T b = v[0]; for (const auto& x : v) if (x > b) b = x; return b; }
std::println("{} {} {}", geo::twice(geo::pi), static_cast<int>(Color::Blue), biggest(std::vector<int>{3, 9, 2}));
using namespace geo;
std::println("{}", twice(1));
)CPP", "7 2 9\n2\n");
}

TEST(reflection) {
    EXPECT(R"CPP(
// Runtime reflection in FORGE script: run with `forge run docs/examples/script/reflection.cpp`
struct [[serializable]] Point {
    [[key("x")]] double x = 0;
    double y = 0;
    const int id = 7;
    Point(double a, double b) : x(a), y(b) {}
    double norm2() const { return x * x + y * y; }
    void scale(double k) { x *= k; y *= k; }
};
enum class Color { Red, Green, Blue };
int add(int a, int b = 2) { return a + b; }

int main() {
    auto p = forge::reflect::construct("Point", 3.0, 4.0);
    std::println("{} is a {}", forge::typeof(p), forge::reflect::kind(p));
    for (const auto& f : forge::reflect::fields(Point)) std::println("field {} : {}", f["name"], f["type"]);
    for (const auto& m : forge::reflect::methods(Point)) std::println("method {}", m["name"]);
    std::println("attrs {}", forge::reflect::attributes(Point).size());
    std::println("norm2 {}", forge::reflect::invoke(p, "norm2"));
    forge::reflect::set(p, "x", 6.0);
    std::println("x {}", forge::reflect::get(p, "x"));
    try { forge::reflect::set(p, "id", 1); } catch (const std::exception& e) { std::println("const refused"); }
    for (const auto& e : forge::reflect::enum_values(Color)) std::println("{} = {}", e["name"], e["value"]);
    auto s = forge::reflect::signature(add);
    std::println("{} returns {} arity {}", s["name"], s["returns"], s["arity"]);
    std::println("call {}", forge::reflect::call("add", 40));
    return 0;
}
)CPP", R"OUT(Point is a object
field x : double
field y : double
field id : const int
method norm2
method scale
attrs 1
norm2 25
x 6
const refused
Red = 0
Green = 1
Blue = 2
add returns int arity 1
call 42
)OUT");
    EXPECT(R"CPP(
class Acct { int secret = 1; void hide() {} public: int bal = 5; int get() const { return bal; } void dep(int n) { bal += n; } };
struct B1 { int a = 1; }; struct D1 : B1 { int b = 2; };
auto o = Acct();
try { forge::reflect::get(o, "secret"); } catch (const std::exception& e) { std::println("get private: {}", e.what()); }
try { forge::reflect::invoke(o, "hide"); } catch (const std::exception& e) { std::println("invoke private: {}", e.what()); }
try { forge::reflect::set(o, "bal", "str"); } catch (const std::exception& e) { std::println("bad type: {}", e.what()); }
forge::reflect::invoke(o, "dep", 10); std::println("{}", forge::reflect::get(o, "bal"));
forge::freeze(o);
try { forge::reflect::set(o, "bal", 1); } catch (const std::exception& e) { std::println("frozen: {}", e.what()); }
try { forge::reflect::invoke(o, "dep", 1); } catch (const std::exception& e) { std::println("frozen call: {}", e.what()); }
std::println("{}", forge::reflect::invoke(o, "get"));
std::println("{} {}", forge::reflect::bases(D1).size(), forge::reflect::is_base_of(B1, D1));
std::println("{}", forge::reflect::has(D1, "b"));
std::vector<int> v{1,2};
auto h = D1();
try { for (auto x : v) forge::reflect::set(v, "size", 1); } catch (const std::exception& e) { std::println("vec: {}", e.what()); }
std::println("{}", forge::reflect::symbols().size() > 0);
)CPP", R"OUT(get private: Acct::secret is private
invoke private: Acct::hide() is private
bad type: Acct::bal expects int, got string
15
frozen: object is frozen
frozen call: Acct::dep() is not const and the object is frozen
15
1 true
true
vec: reflect::set expects an object, class or map, got list
true
)OUT");
    EXPECT(R"CPP(
std::map<std::string, int> m{{"a", 1}};
try { for (const auto& [k, x] : m) forge::reflect::set(m, "a", 2); } catch (const std::exception& e) { std::println("borrow: {}", e.what()); }
forge::reflect::set(m, "a", 3); std::println("{}", m["a"]);
struct P { int x = 1; };
try { forge::sandbox(std::map<std::string, bool>{{"reflect", false}}, [] { auto p = P(); forge::reflect::set(p, "x", 2); }); } catch (const std::exception& e) { std::println("sandbox: {}", e.what()); }
forge::sandbox(std::map<std::string, bool>{{"reflect", false}}, [] { auto p = P(); std::println("read ok {}", forge::reflect::get(p, "x")); });
)CPP", R"OUT(borrow: cannot insert into a map while it is being iterated
3
sandbox: reflection (set) is disabled in this sandbox
read ok 1
)OUT");
    EXPECT("struct S { int x = 1; }; auto s = S(); forge::reflect::get(s, \"nope\");", "!AttributeError");
    EXPECT("class A { int s = 1; }; auto a = A(); forge::reflect::get(a, \"s\");", "!AccessError");
    EXPECT("struct P { int x = 1; }; forge::sandbox(std::map<std::string, bool>{{\"reflect\", false}}, [] { auto p = P(); forge::reflect::set(p, \"x\", 2); });", "!PermissionError");
    EXPECT("std::map<std::string, int> m{{\"a\", 1}}; for (auto& [k, v] : m) forge::reflect::set(m, \"a\", 2);", "!BorrowError");
    EXPECT("struct C { const int k = 1; }; auto c = C(); forge::reflect::set(c, \"k\", 2);", "!TypeError");
}

TEST(destructors_raii) {
    EXPECT(R"CPP(
struct G { std::string n; G(std::string s) : n(s) { std::println("+{}", n); } ~G() { std::println("-{}", n); } };
struct Base { virtual ~Base() { std::println("~Base"); } };
struct Derived : Base { G g{"member"}; ~Derived() override { std::println("~Derived"); } };
void f() { G a("f"); throw std::runtime_error("boom"); }
int main() {
    { G a("a"); G b("b"); std::println("body"); }
    for (int i = 0; i < 2; ++i) { G l("l" + std::to_string(i)); }
    try { f(); } catch (const std::exception& e) { std::println("caught {}", e.what()); }
    { auto u = std::make_unique<G>("u"); auto v = std::move(u); std::println("moved"); }
    { G x("x"); G y = x; y.n = "y"; }
    { std::unique_ptr<Base> p = std::make_unique<Derived>(); }
    { auto s = std::make_shared<G>("s"); auto t = s; s = nullptr; std::println("still shared"); }
    G last("last");
    std::println("end");
    return 0;
}
)CPP", "+a\n+b\nbody\n-b\n-a\n+l0\n-l0\n+l1\n-l1\n+f\n-f\ncaught boom\n+u\nmoved\n-u\n+x\n-y\n-x\n+member\n~Derived\n~Base\n-member\n+s\nstill shared\n-s\n+last\nend\n-last\n");
}

TEST(borrow_rules_while_iterating) {
    EXPECT("std::vector<int> v{1, 2};\ntry { for (int x : v) v.push_back(x); } catch (const BorrowError& e) { std::println(\"{}\", e.what()); }\nv.push_back(3); std::println(\"{}\", v.size());\n// forge-check: ignore-file",
           "cannot call push_back() on a vector while it is being iterated\n3\n");
    EXPECT("std::map<std::string, int> m{{\"a\", 1}};\nfor (auto& [k, x] : m) m[k] = x + 1;\ntry { for (auto& [k, x] : m) m[\"new\"] = 1; } catch (const BorrowError&) { std::println(\"blocked\"); }\nstd::println(\"{}\", m[\"a\"]);",
           "blocked\n2\n");
    EXPECT("std::vector<int> v{1, 2, 3};\nfor (int& x : v) { if (x == 2) break; }\nv.erase(v.begin()); std::println(\"{}\", v.size());", "2\n");
}

TEST(references_write_through) {
    EXPECT("void inc(int& x) { ++x; }\nvoid set(std::string& s) { s = \"new\"; }\nstruct P { int v = 0; };\nint a = 1; inc(a); inc(a);\nstd::string s = \"old\"; set(s);\nstd::vector<int> v{1, 2}; inc(v[1]);\nP p; inc(p.v);\nstd::println(\"{} {} {} {}\", a, s, v[1], p.v);",
           "3 new 3 1\n");
    EXPECT("auto twice = [](double& d) { d *= 2; };\ndouble x = 1.5; twice(x);\nstd::vector<int> v{1, 2, 3};\nfor (int& e : v) e *= 10;\nstd::for_each(v.begin(), v.end(), [](int& e) { e += 1; });\nstd::map<std::string, int> m{{\"a\", 1}};\nfor (auto& [k, n] : m) n += 5;\nstd::string w = \"abc\"; for (char& c : w) c = std::toupper(c);\nstd::println(\"{} {} {} {}\", x, v, m[\"a\"], w);",
           "3 [11, 21, 31] 6 ABC\n");
    EXPECT("void by_value(int x) { x = 9; }\nvoid by_cref(const int& x) { }\nint a = 1; by_value(a); by_cref(a); std::println(\"{}\", a);", "1\n");
}

TEST(debug_race_and_deadlock_detection) {
    script::options o; o.debug = true; o.deadlock_ms = 200;
    auto got = run(R"CPP(
int main() {
    forge::shared<std::shared_ptr<std::vector<int>>> s(std::make_shared<std::vector<int>>());
    auto t = std::async([&]() {
        std::shared_ptr<std::vector<int>> leaked;
        s.with([&](auto& p) { leaked = p; });
        leaked->push_back(1);   // outside the lock
        return 0;
    });
    try { t.get(); } catch (const RaceError&) { std::println("race"); }
    auto a = forge::shared<int>(0); auto b = forge::shared<int>(0);
    auto t1 = std::async([&]() { a.with([&](auto& x) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); b.with([](auto& y) { y = y + 1; }); }); });
    auto t2 = std::async([&]() { std::this_thread::sleep_for(std::chrono::milliseconds(10)); b.with([&](auto& y) { a.with([](auto& x) { x = x + 1; }); }); });
    int dl = 0;
    try { t1.get(); } catch (const DeadlockError&) { ++dl; }
    try { t2.get(); } catch (const DeadlockError&) { ++dl; }
    std::println("inversions {}", dl);
    forge::channel<int> ch(0);
    try { ch.recv(); } catch (const DeadlockError&) { std::println("stall"); }
    auto ok = std::async([](forge::channel<int> c) { c.send(7); return 0; }, ch);
    std::println("recv {}", ch.recv());
    ok.get();
    return 0;
}
)CPP", o);
    CHECK_MSG(got == "race\ninversions 1\nstall\nrecv 7\n", "got:\n%s", got.c_str());
    // Without --debug the same leak is not checked (and the program still runs).
    auto plain = run("forge::shared<int> s(1);\nstd::println(\"{}\", s.with([](auto& x) { return x + 1; }));");
    CHECK(plain == "2\n");
}

TEST(arc_weak_and_cycles) {
    EXPECT(R"CPP(
struct Node { std::shared_ptr<Node> other; };
std::weak_ptr<Node> make() {
    auto a = std::make_shared<Node>();
    auto b = std::make_shared<Node>();
    a->other = b; b->other = a;
    return a;
}
auto w = make();
forge::gc::collect();
std::println("{}", w.expired());
)CPP", "true\n");
}

TEST(limits) {
    script::options o; o.max_steps = 5000;
    auto r = run1("while (true) { }", true, o); CHECK(r.r.error_type == "LimitExceededError");
    r = run1("int f() { return f(); }\nf();", true); CHECK(r.r.error_type == "LimitExceededError");
    r = run1("int f() { return f(); }\nf();", false); CHECK(r.r.error_type == "LimitExceededError");
    o = {}; o.timeout_ms = 100;
    r = run1("while (true) { }", true, o); CHECK(r.r.error_type == "TimeoutError");
}

// Whole-run limits, std::exit and cancellation end the run: no catch clause may swallow them, in any form, at any
// depth, in handlers, lambdas or tasks. A limit {.timeout / .cpu_s} block that fires cannot be caught inside the
// block (the body would carry on with the limit switched off) but is an ordinary error outside it.
TEST(fatal_errors_pass_through_catch) {
    const char* forms[] = {
        "catch (...) { std::println(\"swallowed\"); }",
        "catch (const std::exception& e) { std::println(\"swallowed\"); }",
        "catch (const TimeoutError& e) { std::println(\"swallowed\"); }",
        "catch (const forge::TimeoutError& e) { std::println(\"swallowed\"); }",
        "catch (const LimitExceededError& e) { std::println(\"swallowed\"); }",
        "catch (const OSError& e) { std::println(\"swallowed\"); }",
        "catch (const SystemExit& e) { std::println(\"swallowed\"); }",
        "catch (const CancelledError& e) { std::println(\"swallowed\"); }",
        "catch (const std::runtime_error& e) { std::println(\"swallowed\"); } catch (...) { std::println(\"swallowed\"); }",
    };
    for (bool vm : {false, true}) for (auto* f : forms) {
        std::string c = f;
        // scripts that would run for ever if they could catch the error
        std::string spin = "int n = 0;\nwhile (true) { try { while (true) { n++; } } " + c + " }\n";
        std::string nested = "void inner() { try { while (true) {} } " + c + " }\nvoid outer() { while (true) { try { inner(); } " + c + " } }\nouter();\n";
        std::string lambda = "auto f = [] { try { while (true) {} } " + c + " };\nwhile (true) { try { f(); } " + c + " }\n";
        std::string rethrow = "while (true) { try { try { while (true) {} } catch (...) { std::println(\"swallowed\"); throw; } } " + c + " }\n";
        for (auto* src : {&spin, &nested, &lambda, &rethrow}) {
            script::options o; o.timeout_ms = 60;
            auto r = run1(*src, vm, o);
            CHECK_MSG(r.r.error_type == "TimeoutError" && r.text == "!TimeoutError", "timeout, vm=%d, %s -> %s", int(vm), f, r.text.c_str());
            o = {}; o.max_steps = 20000;
            r = run1(*src, vm, o);
            CHECK_MSG(r.r.error_type == "LimitExceededError" && r.text == "!LimitExceededError", "steps, vm=%d, %s -> %s", int(vm), f, r.text.c_str());
        }
        // std::exit is not an exception: handlers do not run, the exit code is the argument
        auto r = run1("void quit() { std::exit(7); }\ntry { try { quit(); } " + c + " } " + c + "\nstd::println(\"still running\");\n", vm);
        CHECK_MSG(r.r.ok && r.r.exit_code == 7 && r.text.empty(), "exit, vm=%d, %s -> %s (%d)", int(vm), f, r.text.c_str(), r.r.exit_code);
        // a limit block's own timeout: not catchable inside the block, catchable outside it
        r = run1("int rounds = 0;\ntry { limit {.timeout = \"40ms\"} { while (true) { try { while (true) {} } " + c + " rounds++; } } }\n"
                 "catch (const TimeoutError& e) { std::println(\"outside: {} rounds\", rounds); }\nstd::println(\"after\");\n", vm);
        CHECK_MSG(r.r.ok && r.text == "outside: 0 rounds\nafter\n", "limit block, vm=%d, %s -> %s", int(vm), f, r.text.c_str());
        r = run1("int rounds = 0;\ntry { limit {.cpu_s = \"40ms\"} { while (true) { try { while (true) {} } " + c + " rounds++; } } }\n"
                 "catch (const LimitExceededError& e) { std::println(\"outside: {} rounds\", rounds); }\nstd::println(\"after\");\n", vm);
        CHECK_MSG(r.r.ok && r.text == "outside: 0 rounds\nafter\n", "cpu block, vm=%d, %s -> %s", int(vm), f, r.text.c_str());
    }
    // nested blocks: the outer block's deadline is fatal through the inner block and its handlers too
    EXPECT("try { limit {.timeout = \"40ms\"} { try { limit {.timeout = \"10s\"} { while (true) {} } } catch (...) { std::println(\"inner swallowed\"); } while (true) {} } }\n"
           "catch (const TimeoutError&) { std::println(\"outer\"); }", "outer\n");
    // an inner block's own deadline is catchable in the outer block, which carries on
    EXPECT("limit {.timeout = \"10s\"} { try { limit {.timeout = \"30ms\"} { while (true) {} } } catch (const TimeoutError&) { std::println(\"inner timed out\"); } std::println(\"outer goes on\"); }",
           "inner timed out\nouter goes on\n");
    // destructors still run while a fatal error unwinds, but cannot keep the run alive
    {
        script::options o; o.timeout_ms = 60;
        std::string src = "struct G { ~G() { std::println(\"~G\"); } };\nvoid f() { G g; while (true) {} }\ntry { f(); } catch (...) { std::println(\"swallowed\"); }\n";
        for (bool vm : {false, true}) { auto r = run1(src, vm, o); CHECK_MSG(r.text == "~G\n!TimeoutError", "vm=%d: %s", int(vm), r.text.c_str()); }
        src = "struct G { ~G() { while (true) {} } };\nvoid f() { G g; while (true) {} }\ntry { f(); } catch (...) { std::println(\"swallowed\"); }\n";
        for (bool vm : {false, true}) { auto r = run1(src, vm, o); CHECK_MSG(r.text == "!TimeoutError", "vm=%d: %s", int(vm), r.text.c_str()); }
    }
    // a task that exits ends the whole run; a task cannot outlive the run's timeout by catching it
    for (bool vm : {false, true}) {
        auto r = run1("auto t = std::async([] { try { std::exit(5); } catch (...) {} return 1; });\ntry { t.get(); } catch (...) { std::println(\"swallowed\"); }\nwhile (true) {}\n", vm);
        CHECK_MSG(r.r.ok && r.r.exit_code == 5 && r.text.empty(), "task exit, vm=%d: %s (%d)", int(vm), r.text.c_str(), r.r.exit_code);
        script::options o; o.timeout_ms = 80;
        r = run1("auto t = std::async([] { while (true) { try { while (true) {} } catch (...) {} } return 1; });\ntry { t.get(); } catch (...) { std::println(\"swallowed\"); }\n", vm, o);
        CHECK_MSG(r.text == "!TimeoutError", "task timeout, vm=%d: %s", int(vm), r.text.c_str());
    }
    // the next run of the same engine starts clean (REPL: a timed-out line does not poison the session)
    {
        std::string out; script::options o; o.timeout_ms = 50; o.out = [&](std::string_view s) { out += s; };
        script::engine e(o);
        auto r = e.run("try { while (true) {} } catch (...) {}", "<t>");
        CHECK(!r.ok && r.error_type == "TimeoutError");
        r = e.run("try { throw std::runtime_error(\"x\"); } catch (const std::exception& e) { std::println(\"caught {}\", e.what()); }", "<t>");
        CHECK_MSG(r.ok && out == "caught x\n", "%s / %s", out.c_str(), r.error.c_str());
    }
    // ordinary limit errors stay catchable (the heap cap re-arms, so catching does not lift it)
    EXPECT("try { limit {.mem = \"64K\"} { std::vector<std::string> xs; while (true) xs.push_back(std::string(64, 'x')); } } catch (const std::bad_alloc& e) { std::println(\"oom\"); }", "oom\n");
}

// Catch clauses, the parts a golden file cannot show: diagnostics for types that cannot be caught, and that a named
// type never parses as a catch-all (tests/script/catch_types.fs has the matching rules).
TEST(catch_clause_types) {
    EXPECT("try { throw std::runtime_error(\"x\"); } catch (const PermissionError& e) { std::println(\"wrong\"); }", "!RuntimeError");
    EXPECT("try { throw std::runtime_error(\"x\"); } catch (PermissionError) { std::println(\"wrong\"); }", "!RuntimeError");
    EXPECT("try { throw 1; } catch (const std::exception& e) { std::println(\"wrong\"); }", "!Error");
    EXPECT("using namespace std;\ntry { throw out_of_range(\"x\"); } catch (const logic_error& e) { println(\"logic {}\", e.what()); }", "logic x\n");
    EXPECT("using namespace forge;\ntry { fs::read(\"/no/such/file/anywhere\"); } catch (const PermissionError& e) { std::println(\"wrong\"); } catch (const std::exception& e) { std::println(\"other\"); }", "other\n");
    for (auto* bad : {"catch (auto e) {}", "catch (std::any& a) {}", "catch (const std::bad_function_call& e) {}", "catch (std::optional<int> o) {}", "catch (std::function<void()> f) {}"}) {
        auto r = run1(std::string("try { throw 1; } ") + bad, true);
        CHECK_MSG(r.r.error_type == "SyntaxError" && r.r.error.find("catch") != std::string::npos, "%s -> %s", bad, r.r.error.c_str());
    }
    { auto r = run1("template <typename T> void f() { try { throw 1; } catch (const T& e) {} }", true); CHECK_MSG(r.r.error_type == "SyntaxError", "%s", r.r.error.c_str()); }
    // a user exception's own `line` member is not overwritten by the throw site
    EXPECT("struct E : std::runtime_error { int line; E(const std::string& m, int l) : std::runtime_error(m), line(l) {} };\ntry { throw E(\"m\", 12); } catch (const E& e) { std::println(\"{} {}\", e.what(), e.line); }", "m 12\n");
    // function-try-block
    EXPECT("int f(int x) try { if (x > 1) throw std::invalid_argument(\"big\"); return x; } catch (const std::invalid_argument& e) { return -1; }\nstd::println(\"{} {}\", f(1), f(5));", "1 -1\n");
}

// References (golden: tests/script/references.fs). `Base& b = d;` used to copy d: writes through b were lost and the
// destructor ran once per reference. The parts a golden cannot show: the diagnostics.
TEST(reference_binding) {
    EXPECT("struct G { std::string n; G(std::string s) : n(s) {} ~G() { std::println(\"~{}\", n); } };\nvoid f() { G g(\"g\"); G& r = g; const G& c = g; { auto& a = g; a.n = \"h\"; } std::println(\"{} {}\", r.n, c.n); }\nf(); std::println(\"end\");", "h h\n~h\nend\n");
    EXPECT("struct P { int x = 0; };\nvoid f() { std::vector<P> v(2); P& e = v[1]; e.x = 4; const P& c = v[1]; v[1].x += 1; std::println(\"{} {}\", v[1].x, c.x); }\nf();", "5 5\n");
    EXPECT("void f() { int n = 1; int& r = n; r = 5; const int& c = n; n += 1; std::println(\"{} {} {}\", n, r, c); }\nf();", "6 6 6\n");
    EXPECT("struct P { int x = 0; };\nvoid f() { auto u = std::make_unique<P>(); const std::unique_ptr<P>& r = u; r->x = 3; std::println(\"{}\", u->x); }\nf();", "3\n");
    // a const reference cannot be assigned through, whatever the original is
    { auto r = run1("void f() { int n = 1; const int& c = n; c = 2; }", true); CHECK_MSG(r.r.error_type == "SyntaxError" && r.r.error.find("const") != std::string::npos, "%s", r.r.error.c_str()); }
    // assigning through a reference to a member or an element would only rebind the name: refused, not silently dropped
    for (auto* bad : {"struct B { int n = 0; std::string s; };\nvoid f() { B b; int& r = b.n; r = 2; }", "struct B { int n = 0; std::string s; };\nvoid f() { B b; std::string& r = b.s; r += \"x\"; }",
                      "void f() { std::vector<int> v(3); int& e = v[0]; ++e; }", "struct B { int n = 0; }; B g;\nvoid f() { B& r = g; r = B{}; }"}) {
        auto r = run1(bad, true);
        CHECK_MSG(r.r.error_type == "SyntaxError" && r.r.error.find("cannot assign through reference") != std::string::npos, "%s -> %s", bad, r.r.error.c_str());
    }
    // delegating to a constructor that does not exist, a base constructor arity that does not exist
    EXPECT("struct R { int a; R() : R(1, 2, 3) {} R(int x) : a(x) {} };\nR r;", "!TypeError");
    EXPECT("struct B { int a; B() : a(0) {} B(int x) : a(x) {} };\nstruct D : B { D() : B(1, 2) {} };\nD d;", "!TypeError");
}

// fork() while another thread is inside the script heap. ProcessPool's fork-mode workers and sandbox({isolate: true})
// build an interpreter in the forked child; when some other thread of the parent held the heap's object-set mutex
// at the moment of the fork, the child waited for it for ever (test_script hung in an ASan run: a pool dispatcher
// forked a worker while the main thread was allocating). The heap now takes its mutexes around fork (pthread_atfork).
#if !defined(_WIN32) && !defined(FORGE_OS_IOS)
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#include <atomic>
#include <chrono>
#include <thread>
TEST(fork_while_another_thread_allocates) {
    std::atomic<bool> stop{false};
    std::thread busy([&] {   // allocates script objects until told to stop
        script::options o; o.timeout_ms = 60000; o.out = [](std::string_view) {};
        while (!stop.load()) { script::engine e(o); (void)e.run("std::vector<std::string> v; for (int i = 0; i < 3000; ++i) v.push_back(std::to_string(i));", "<busy>"); }
    });
    int hung = 0, bad = 0;
    for (int k = 0; k < 150 && !hung; ++k) {
        pid_t pid = ::fork();
        if (pid < 0) break;
        if (pid == 0) {   // what a fork-mode pool worker does first: build an interpreter and run a script
            script::options o; o.out = [](std::string_view) {};
            script::engine e(o);
            ::_exit(e.run("std::vector<int> v(100, 1); int s = 0; for (int x : v) s += x;", "<child>").ok ? 0 : 3);
        }
        int st = 0; auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            pid_t r = ::waitpid(pid, &st, WNOHANG);
            if (r == pid) break;
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)) { ::kill(pid, SIGKILL); ::waitpid(pid, &st, 0); ++hung; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!hung && !(WIFEXITED(st) && WEXITSTATUS(st) == 0)) ++bad;
    }
    stop = true; busy.join();
    CHECK_MSG(hung == 0, "%d forked child(ren) never finished: deadlock on a heap mutex copied in the locked state", hung);
    CHECK_MSG(bad == 0, "%d forked child(ren) failed", bad);
}
#endif

// Holes found by the conformance sweep (tests/script_conformance compares whole programs with the host compiler; these
// are the same points without needing one).
TEST(cpp_conformance_holes) {
    // assignment and ++ / -- are expressions
    EXPECT("int twice(int x) { return 2 * x; }\nint n = 1;\nstd::println(\"{} {}\", twice(n++), n);\nstd::println(\"{} {}\", twice(++n), n);", "2 2\n6 3\n");
    EXPECT("std::vector<int> v{10, 20, 30}; int p = 0;\nstd::print(\"{} \", v[p++]); std::print(\"{} \", v[p++]); v[p]++; ++v[0];\nstd::println(\"{} {} {}\", p, v[0], v[2]);", "10 20 2 11 31\n");
    EXPECT("int c = 0; while (c++ < 3) std::print(\"{} \", c);\nint a, b; a = b = 7; int d = (a += 2) * 2;\nstd::println(\"| {} {} {} {}\", c, a, b, d);", "1 2 3 | 4 9 7 18\n");
    EXPECT("int k = 5; int old = k--; int now = --k; int m = (1, 2, k + 1);\nstd::println(\"{} {} {} {}\", old, now, k, m);", "5 3 3 4\n");
    // element references as assignment targets
    EXPECT("std::vector<int> v{1, 2, 3}; v.at(1) = 20; v.front() += 5; v.back()--;\nauto it = v.begin(); *it = 9; ++*it;\nstd::tuple<int, std::string> t{1, \"a\"}; std::get<0>(t) = 4;\nint x = 0, y = 0; std::tie(x, y) = std::make_pair(7, 8);\nstd::println(\"{} {} {} {} {} {}\", v[0], v[1], v[2], std::get<0>(t), x, y);", "10 20 2 4 7 8\n");
    EXPECT("std::map<std::string, int> m{{\"a\", 1}}; auto it = m.find(\"a\"); it->second *= 5; std::println(\"{}\", m[\"a\"]);", "5\n");
    EXPECT("struct M { int at(int i) const { return i; } };\nM m; m.at(1, 2) = 3;", "!SyntaxError");
    // a variable keeps its declared arithmetic type
    EXPECT("double x = 7; x = 1; int i = 3; i = 7.9; i += 1.5; double d = 1; d += 1;\nstd::println(\"{} {} {}\", x / 2, i, d / 4);", "0.5 8 0.5\n");
    EXPECT("int g() { return 2.7; }\ndouble h() { return 1; }\nbool b = 5; int t = b + b;\nstd::println(\"{} {} {} {}\", g(), h() / 2, b, t);", "2 0.5 true 2\n");
    // char is a character and a small integer
    EXPECT("char c = 'a'; char n = c + 1; c++; int d = '7' - '0';\nstd::println(\"{} {} {} {} {}\", n, c, d, 'A' + 0, static_cast<char>('A' + 25));", "b b 7 65 Z\n");
    EXPECT("int s = 0; for (char ch : std::string(\"1234\")) s += ch - '0';\nstd::string w = \"ab\"; w += 'c'; w.push_back(static_cast<char>(w[0] + 3));\nstd::println(\"{} {} {}\", s, w, w[1] == 'b');", "10 abcd true\n");
    EXPECT("try { throw 'x'; } catch (int) { std::println(\"int\"); } catch (const std::string&) { std::println(\"string\"); } catch (char c) { std::println(\"char {}\", c); }", "char x\n");
    // members that were missing
    EXPECT("#include <deque>\nstd::deque<int> d{2}; d.push_front(1); d.emplace_front(0); d.push_back(3); d.pop_front();\nstd::println(\"{} {} {}\", d.front(), d.back(), d.size());", "1 3 3\n");
    EXPECT("std::string s = \"abcdef\"; s.resize(3); std::string t = s; t.resize(5, '.'); s.assign(2, 'z'); t.insert(0, 2, '>');\nstd::println(\"{} {} {} {}\", s, t, t.find_last_not_of(\".\"), std::string(t, 2, 3));", "zz >>abc.. 4 abc\n");
    EXPECT("std::vector<int> v{5, 3, 8, 3}; std::vector<int> w(v.begin() + 1, v.end());\nv.erase(std::remove(v.begin(), v.end(), 3), v.end()); w.insert(w.begin(), 2, 0); w.assign({1, 2});\nstd::sort(v.begin(), v.end(), std::greater<int>());\nstd::println(\"{} {} {} {}\", v, w, std::lower_bound(w.begin(), w.end(), 2) - w.begin(), std::gcd(12, 18));", "[8, 5] [1, 2] 1 6\n");
    EXPECT("std::set<int> s{5, 1}; auto [it, fresh] = s.insert(3); std::map<int, int> m; m[5] = 1; m[2] = 2;\nstd::println(\"{} {} {} {}\", *it, fresh, m.begin()->first, *s.lower_bound(2));", "3 true 2 3\n");
    EXPECT("std::println(\"[{:{}}] [{:.{}f}]\", 7, 4, 3.14159, 2);\nstd::cout << std::left << std::setw(4) << 42 << '|' << std::right << std::setfill('0') << std::setw(5) << 42 << '|' << std::hex << 255 << '\\n';", "[   7] [3.14]\n42  |00042|ff\n");
    // [=] copies when the lambda is created; a std::function may call itself; std::optional functions may return {}
    EXPECT("int main() { int base = 10; auto f = [=] { return base * 2; }; auto g = [&] { return base * 2; }; base = 20; std::println(\"{} {}\", f(), g()); return 0; }", "20 40\n");
    EXPECT("std::function<int(int)> fact = [&fact](int n) { return n <= 1 ? 1 : n * fact(n - 1); };\nstd::optional<std::string> none() { return {}; }\nstd::println(\"{} {}\", fact(5), none().value_or(\"empty\"));", "120 empty\n");
    EXPECT("struct P { int v; std::shared_ptr<P> next; };\nauto a = std::make_shared<P>(P{1, nullptr}); auto b = std::make_shared<P>(P{2, a}); auto c = b; c.reset();\nstd::println(\"{} {}\", b->next->v, c == nullptr);", "1 true\n");
    EXPECT("struct V { int major = 0, minor = 0; auto operator<=>(const V&) const = default; };\nV a{1, 2}, b{1, 10}; std::pair<int, int> p{1, 2}, q{1, 3};\nstd::println(\"{} {} {} {}\", a < b, a >= b, p < q, std::vector<int>{1, 2} < std::vector<int>{1, 3});", "true false true true\n");
    EXPECT("std::println(\"{} {} {}\", 017, true + true, std::isdigit('7') != 0);", "15 2 true\n");
    // constructs outside the subset say what they are
    for (auto [src, word] : std::vector<std::pair<const char*, const char*>>{
             {"struct C { int a; int b; }; C c{.a = 1, .b = 2};", "designated"},
             {"int f() { static int n = 0; return ++n; }", "static local"},
             {"void f() { enum class E { a, b }; }", "enum class"},
             {"std::vector<int> v{1}; for (int base = 1; int x : v) {}", "range-for with an init"},
             {"int x = 1; auto p = &x;", "address"}}) {
        auto r = run1(src, true);
        CHECK_MSG(r.r.error_type == "SyntaxError" && r.r.error.find(word) != std::string::npos, "%s -> %s", src, r.r.error.c_str());
    }
}

TEST(tasks_channels) {
    EXPECT("int sq(int x) { return x * x; }\nauto t = std::async(sq, 7);\nstd::println(\"{}\", t.get());", "49\n");
    EXPECT("int total = 0;\nstd::thread th([](int n) { std::println(\"thread {}\", n); }, 3);\nth.join();", "thread 3\n");
    EXPECT(R"CPP(
forge::channel<int> ch(4);
auto t = forge::spawn([](forge::channel<int> c) { for (int i = 0; i < 3; ++i) c.send(i); c.close(); }, ch);
std::vector<int> got;
for (int v : ch) got.push_back(v);
t.get();
std::println("{}", got);
)CPP", "[0, 1, 2]\n");
    EXPECT("auto t = std::async([] { throw std::invalid_argument(\"in task\"); });\ntry { t.get(); } catch (const std::invalid_argument& e) { std::println(\"task failed: {}\", e.what()); }", "task failed: in task\n");
}

TEST(mobile_sensors_touch) {
    EXPECT(R"CPP(
auto dev = forge::mobile::device({{"sim", "{\"display\":{\"width\":100,\"height\":200},\"sensors\":[\"accelerometer\"],\"services\":{\"clipboard\":\"c\"},\"events\":[{\"t_ms\":0,\"type\":\"sensor\",\"kind\":\"accelerometer\",\"v\":[1,2,3]},{\"t_ms\":1,\"type\":\"touch\",\"phase\":\"down\",\"id\":0,\"x\":10,\"y\":10},{\"t_ms\":2,\"type\":\"touch\",\"phase\":\"up\",\"id\":0,\"x\":11,\"y\":10}]}"}});
std::println("{}", dev.backend());
std::println("{}", forge::sensors::list(dev).size());
auto acc = forge::sensors::open("accelerometer", 100, dev);
forge::channel<int> ch(16);
int n = 0;
for (int i = 0; i < 50 && n == 0; ++i) n = acc.pump(ch, 8, 20);
auto e = ch.recv();
std::println("{} {}", e["kind"], e["v"]);
auto t = forge::touch::open(dev);
std::vector<int> kinds;
auto gs = t.gestures(200);   // down and up can land in separate polls under load
for (int i = 0; i < 20 && gs.size() == 0; ++i) gs = t.gestures(200);
std::println("{}", gs.size() > 0 ? gs[0]["kind"] : "none");
std::println("{} {}", dev.clipboard_get(), dev.display()["width"]);
acc.close();
try { acc.poll(); } catch (const std::exception& x) { std::println("disposed"); }
)CPP", "sim\n1\naccelerometer [1, 2, 3]\ntap\nc 100\ndisposed\n");
}

#if !defined(_WIN32)
TEST(shell_and_processes) {
    EXPECT("std::string name = \"it's\";\nauto r = forge::sh(\"printf '%s' {}\", name);\nstd::println(\"{} {} {}\", r.out, r.code, r.ok);", "it's 0 true\n");
    EXPECT("auto r = forge::sh(\"echo a; echo b; exit 3\");\nstd::println(\"{} {} {}\", r.lines(), r.code, r.ok);", "[\"a\", \"b\"] 3 false\n");
    EXPECT("std::println(\"{}\", forge::sh(\"echo '{\\\"k\\\": 5}'\").json()[\"k\"]);", "5\n");
}
#endif

TEST(modules) {
    auto dir = std::filesystem::temp_directory_path() / "forge_script_mod";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir / "util.cpp") << "int twice(int x) { return x * 2; }\nconst std::string NAME = \"util\";\n"; }
    { std::ofstream(dir / "main.cpp") << "#include \"util.cpp\"\nint main() { std::println(\"{} {}\", twice(21), NAME); return 4; }\n"; }
    std::string text; script::options o; o.out = [&](std::string_view s) { text += s; };
    script::engine e(o);
    auto r = e.run_file((dir / "main.cpp").string());
    CHECK_MSG(r.ok, "%s", r.error.c_str()); CHECK_MSG(text == "42 util\n", "%s", text.c_str()); CHECK(r.exit_code == 4);
    std::filesystem::remove_all(dir);
}

TEST(modules_and_plugins) {
    auto dir = std::filesystem::temp_directory_path() / "forge_script_pkg";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir / "lib" / "geom");
    { std::ofstream(dir / "lib" / "geom" / "mod.cpp") << "double area(double w, double h) { return w * h; }\nconst int SIDES = 4;\n"; }
    { std::ofstream(dir / "lib" / "strs.fs") << "std::string shout(std::string s) { return s + \"!\"; }\n"; }
    std::string code = R"CPP(
import "geom" as g;
import "strs";
std::println("{} {}", g.area(2, 3), g.SIDES);
auto m = forge::import("strs");
std::println("{}", m.shout("hi"));
std::println("{}", forge::modules().size());
try { forge::import("nope_missing"); } catch (const ImportError& e) { std::println("missing"); }
)CPP";
#ifdef FORGE_TEST_PLUGIN
    // FORGE_TEST_PLUGIN_PATH: where the plugin lives when the test binary runs away from its build tree (emulator)
    const char* plug = std::getenv("FORGE_TEST_PLUGIN_PATH"); if (!plug || !*plug) plug = FORGE_TEST_PLUGIN;
    code += std::string("auto h = forge::plugin(\"") + plug + "\");\n" + R"CPP(
std::println("{} {} {}", h.add(2, 3.5), h.greet("forge"), h.VERSION);
try { h.fail(1, 2); } catch (const PluginError& e) { std::println("{}", e.message); }
try { h.add(1); } catch (const TypeError& e) { std::println("arity"); }
)CPP";
#endif
    std::string text; script::options o; o.out = [&](std::string_view s) { text += s; };
    ::setenv("FORGE_SCRIPT_PATH", (dir / "lib").string().c_str(), 1);
    script::engine e(o);
    auto r = e.eval(code);
    ::unsetenv("FORGE_SCRIPT_PATH");
    CHECK_MSG(r.ok, "%s", r.error.c_str());
    std::string want = "6 4\nhi!\n2\nmissing\n";
#ifdef FORGE_TEST_PLUGIN
    want += "5.5 hello, forge 1.0\nhello_plugin.fail: this always fails\narity\n";
#endif
    CHECK_MSG(text == want, "%s", text.c_str());
    // Sandboxed runs refuse native code.
#ifdef FORGE_TEST_PLUGIN
    script::options so; so.allow_proc = false; so.os_sandbox = script::options::os_sandbox_mode::off;   // keep the test process unsandboxed std::string t2; so.out = [&](std::string_view s) { t2 += s; };
    script::engine se(so);
    auto r2 = se.eval(std::string("forge::plugin(\"") + FORGE_TEST_PLUGIN + "\");");
    CHECK(!r2.ok); CHECK_MSG(r2.error_type == "PermissionError", "%s", r2.error_type.c_str());
#endif
    std::filesystem::remove_all(dir);
}

TEST(repl_eval_keeps_state) {
    std::string text; script::options o; o.out = [&](std::string_view s) { text += s; };
    script::engine e(o);
    CHECK(e.eval("int x = 40;").ok);
    auto r = e.eval("x + 2"); CHECK_MSG(r.ok, "%s", r.error.c_str()); CHECK_MSG(r.value == "42", "%s", r.value.c_str());
    r = e.eval("int f(int a) { return a; }\nf(\"str\")"); CHECK(!r.ok); CHECK(r.error_type == "TypeError");
    CHECK(e.type_of("std::vector<int>{1}") == "list");
    auto d = e.disassemble("int a = 1 + 2;"); CHECK(d.find("BINARY") != std::string::npos || d.find("CONST") != std::string::npos);
    r = e.eval("int y = (1 +"); CHECK(!r.ok); CHECK(r.error_type == "SyntaxError");
}

TEST(errors_carry_lines_and_traces) {
    script::options o; o.out = [](std::string_view) {};
    script::engine e(o);
    auto r = e.run("int inner() { std::vector<int> v; return v.at(0); }\nint outer() { return inner(); }\n\nouter();", "t.cpp");
    CHECK(!r.ok); CHECK(r.error_type == "IndexError");
    CHECK_MSG(r.error.find("inner") != std::string::npos && r.error.find("outer") != std::string::npos, "%s", r.error.c_str());
}
TEST(os_phase2_surface) {
    EXPECT(R"CPP(
auto d = fs.make_temp_dir("fst-");
fs.write_atomic(path.join(d, "a.cpp"), "x");
fs.mkdir(path.join(d, "s"));
fs.write(path.join(d, "s/b.cpp"), "yy");
fs.copy(path.join(d, "s"), path.join(d, "t"));
std::println("{}", fs.glob("**/*.cpp", d));
std::println("{} {}", fs.walk(d).size(), fs.match("**/*.cpp", "x/y.cpp"));
auto l = fs.lock(path.join(d, ".lock"));
std::println("{}", fs.lock(path.join(d, ".lock"), false) == nullptr);
l.unlock();
std::println("{}", fs.lock(path.join(d, ".lock"), false) != nullptr);
std::println("{} {}", time.iso(0), time.parse("2026-10-30T11:00:00+11:00"));
std::println("{} {}", os.user()["name"].size() > 0, os.memory()["total"] > 0);
fs.remove(d);
std::println("{}", fs.exists(d));
)CPP", "[\"a.cpp\", \"s/b.cpp\", \"t/b.cpp\"]\n5 true\ntrue\ntrue\n1970-01-01T00:00:00.000Z 1793318400\ntrue true\nfalse\n");
}
TEST(automation_pipeline) {
    EXPECT(R"CPP(
std::vector<std::string> log;
auto p = forge::pipeline();
p.job("build", {"gen"}, [&]() { log.push_back("build"); return 2; });
p.job("gen", [&]() { log.push_back("gen"); return 1; });
p.job("test", {"build"}, [&]() { log.push_back("test"); throw std::runtime_error("boom"); });
p.job("docs", {"gen"}, [&]() { log.push_back("docs"); return 0; });
p.job("ship", {"test"}, [&]() { log.push_back("ship"); return 0; });
std::println("{}", p.order());
auto r = p.run();
std::println("{} {} {}", r["ok"], log, r["jobs"]["build"]["value"]);
std::println("{} {} {}", r["jobs"]["test"]["status"], r["jobs"]["test"]["error"], r["jobs"]["ship"]["status"]);
auto q = forge::pipeline();
q.job("a", {"b"}, [&]() { return 0; });
q.job("b", {"a"}, [&]() { return 0; });
try { q.run(); } catch (const PipelineError& e) { std::println("{}", e.message); }
int tries = 0;
auto z = forge::pipeline();
z.job("flaky", {}, [&]() { tries += 1; if (tries < 3) throw std::runtime_error("x"); return tries; }, {{"retries", 2}});
std::println("{}", z.run()["jobs"]["flaky"]["value"]);
auto par = forge::pipeline();
par.job("x", [&]() { return 1; });
par.job("y", [&]() { return 2; });
par.job("sum", {"x", "y"}, [&]() { return 3; });
auto pr = par.run({{"parallel", 4}});
std::println("{} {} {}", pr["ok"], pr["jobs"]["y"]["value"], pr["jobs"]["sum"]["status"]);
)CPP", "[\"gen\", \"build\", \"test\", \"docs\", \"ship\"]\nfalse [\"gen\", \"build\", \"docs\", \"test\"] 2\nfailed boom cancelled\ndependency cycle through 'a'\n3\ntrue 2 ok\n");
}

TEST(automation_incremental) {
    EXPECT(R"CPP(
auto d = fs.make_temp_dir("fpl-");
auto src = path.join(d, "in.txt"); auto out = path.join(d, "out.txt");
fs.write(src, "hello");
int runs = 0;
auto p = forge::pipeline({{"cache", path.join(d, ".cache")}});
p.job("copy", {}, [&]() { runs += 1; fs.write(out, fs.read(src)); return 0; }, {{"inputs", std::vector<std::string>{src}}, {"outputs", std::vector<std::string>{out}}});
p.run(); auto r2 = p.run();
std::println("{} {}", runs, r2["jobs"]["copy"]["status"]);
fs.set_mtime(src, time.now() + 5); p.run();
std::println("{} {}", runs, p.run({{"dry_run", true}})["jobs"]["copy"]["status"]);
p.run({{"force", true}});
std::println("{}", runs);
fs.remove(d);
)CPP", "1 skipped\n2 skipped\n3\n");
}

TEST(automation_schedule) {
    EXPECT(R"CPP(
std::println("{} {} {}", forge::duration_ms("1h30m"), forge::duration_ms("250ms"), forge::duration_ms(7));
std::println("{}", forge::every("20ms", [&](int i) { return true; }, {{"count", 3}}));
int n = 0;
std::println("{} {}", forge::every(10, [&]() { n += 1; return n < 2; }), n);
auto s = forge::scheduler();
s.every("15ms", [&]() { n += 10; }, {{"times", 2}});
s.every("10ms", [&]() { n += 100; }, {{"times", 1}});
std::println("{} {} {}", s.upcoming().size(), s.run(), n);
auto t = forge::cron_next("30 4 * * *", 0);
std::println("{}", t != nullptr);
try { forge::cron("61 * * * *", [&]() {}); } catch (const ValueError& e) { std::println("bad cron"); }
try { forge::every("5parsecs", [&]() {}); } catch (const ValueError& e) { std::println("bad dur"); }
)CPP", "5400000 250 7\n3\n2 2\n2 3 122\ntrue\nbad cron\nbad dur\n");
}

TEST(automation_watch) {
    EXPECT(R"CPP(
auto d = fs.make_temp_dir("fwt-");
std::vector<std::string> got;
auto n = forge::watch(path.join(d, "*.txt"), [&](auto changes) {
    for (auto& c : changes) got.push_back(c["kind"] + " " + path.base(c["path"]));
    return got.size() < 1;
}, {{"debounce_ms", 30}, {"for", "5s"}, {"ready", [&]() { fs.write(path.join(d, "a.log"), "x"); fs.write(path.join(d, "a.txt"), "x"); }}});
std::println("{} {}", n, got);
fs.remove(d);
)CPP", "1 [\"created a.txt\"]\n");
}

TEST(os_phase3_surface) {
    EXPECT(R"CPP(
auto d = fs.make_temp_dir("fst-");
auto w = fs.watch_tree(d, {{"debounce_ms", 40}, {"ignore", std::vector<std::string>{"*.tmp"}}});
fs.mkdir(path.join(d, "x"));
fs.write(path.join(d, "x/f.txt"), "1");
fs.write(path.join(d, "skip.tmp"), "1");
std::vector<std::string> seen;
for (int i = 0; i < 10 && seen.size() < 2; ++i)
    for (auto& e : w.poll(500)) seen.push_back(e["kind"] + " " + path.base(e["path"]));
std::println("{}", seen);
auto u = net.parse_url("http://h:81/p?q=1");
std::println("{} {} {}", u["host"], u["port"], u["query"]);
auto un = service.unit({{"name", "forge-t"}, {"argv", std::vector<std::string>{"/bin/echo", "a b"}}});
std::println("{}", un.find("\"a b\"") != std::string::npos || un.find("<string>a b</string>") != std::string::npos);   // systemd/sc quote, launchd plist
auto st = service.status({{"name", "forge-t"}, {"argv", std::vector<std::string>{"/bin/echo"}}, {"dir", d}});
std::println("{}", st == "unknown" ? "not_installed" : st);   // wine's sc.exe can't report 1060
w.close();   // Windows: the watch holds the directory open
fs.remove(d);
)CPP", "[\"created x\", \"created f.txt\"]\nh 81 q=1\ntrue\nnot_installed\n");
}

TEST(ai_assisted_commands) {
#ifdef _WIN32
    if (std::system("sh -c true >NUL 2>&1") != 0) { std::puts("  (skip: ai_run needs sh on PATH)"); return; }
#endif
    auto fake = [](const std::string& sys, const std::string& p) -> std::string {
        if (sys.find("JSON") != std::string::npos) return "{\"n\": 3, \"tags\": [\"a\"]}";
        if (sys.find("write a FORGE script") != std::string::npos) return "```cpp\nstd::println(\"from ai\");\n```";
        if (sys.find("shell command") != std::string::npos) return "```sh\necho {hi}\n```";
        return "reply:" + p;
    };
    script::options o; o.ai = fake;   // no host confirm: drafted code never runs unless the script confirms
    auto got = run(R"CPP(
std::println("{}", forge::ask("ping"));
std::println("{}", forge::suggest_cmd("say hi"));
auto j = forge::ask_json("x"); std::println("{} {}", j["n"], j["tags"]);
auto r = forge::ai_run("say hi");
std::println("{} {}", r["ran"], r["command"]);
std::string seen;
auto r2 = forge::ai_run("say hi", {{"confirm", [&](auto c) { seen = c; return true; }}});
std::println("{} {} [{}]", r2["ran"], seen, r2["result"].out);
auto r3 = forge::ai_eval("print", {{"confirm", [](auto c) { return true; }}});
std::println("{}", r3["ran"]);
auto r4 = forge::ai_run("say hi", {{"confirm", [](auto c) { return false; }}});
std::println("{}", r4["ran"]);
)CPP", o);
    CHECK_MSG(got == "reply:ping\necho {hi}\n3 [\"a\"]\nfalse echo {hi}\ntrue echo {hi} [{hi}\n]\nfrom ai\ntrue\nfalse\n", "got:\n%s", got.c_str());
    script::options none; auto e = run(R"CPP(
try { forge::ask("x"); } catch (const AIError& e) { std::println("AIError"); }
)CPP", none);
    CHECK_MSG(e == "AIError\n", "got:\n%s", e.c_str());
}

// In-process inference for forge::ask / ask_json: a tiny random Llama (tests/tiny_model.hpp) through the inference
// module, registered the way the forge CLI does it (forge/script/local_ai.hpp).
#include <forge/script/local_ai.hpp>
#include "tiny_model.hpp"
TEST(ask_local_model) {
    namespace stdfs = std::filesystem;
    auto path = (stdfs::temp_directory_path() / ("forge_ask_" + std::to_string(::getpid()) + ".fpk")).string();
    forge_test::make_tiny_model(path);
    script::options o; o.ask_model = path; o.ask_max_tokens = 8;
    // without a registered generator (forge_script alone) a configured model is an AIError, not a silent fallback
    script::set_local_ai(nullptr);
    auto miss = run("try { forge::ask(\"hi\"); } catch (const AIError& e) { std::println(\"{}\", e.what()); }", o);
    CHECK_MSG(miss.find("no in-process inference") != std::string::npos, "got: %s", miss.c_str());
    script::install_local_ai();
    CHECK(script::has_local_ai());
    // the generator is greedy decoding of forge-serve's chat layout, straight through the inference module
    {
        forge::inference::session_options so; so.context = 64;
        forge::inference::session s(path, so);
        auto* tok = s.tokenizer();
        auto ids = tok->encode("user: hi\nassistant:"); ids.insert(ids.begin(), tok->bos());
        forge::inference::sampler smp;
        auto out = s.generate(ids, 8, smp);
        if (!out.empty() && out.back() == tok->eos()) out.pop_back();
        CHECK(script::local_ai_generate({path, "", "hi", false, 8}) == tok->decode(out));
    }
    // forge::ask returns exactly what the model generates for its system prompt + the question
    auto want = script::local_ai_generate({path, "You are a concise assistant inside a FORGE script.", "hi", false, 8});
    auto got = run("std::print(\"{}\", forge::ask(\"hi\"));", o);
    CHECK_MSG(got == want, "got [%s] want [%s]", got.c_str(), want.c_str());
    // $FORGE_ASK_MODEL selects the same model when the option is empty
    ::setenv("FORGE_ASK_MODEL", path.c_str(), 1);
    script::options envo; envo.ask_max_tokens = 8;
    CHECK(run("std::print(\"{}\", forge::ask(\"hi\"));", envo) == want);
    ::unsetenv("FORGE_ASK_MODEL");
    // ask_json is grammar-constrained: a random model yields some JSON value, or a clear error when the token cap
    // cuts the value off; never prose
    auto j = run("try { auto v = forge::ask_json(\"give json\"); std::println(\"json {}\", json.dump(v).size() > 0); } catch (const AIError& e) { std::println(\"{}\", e.what()); }", o);
    CHECK_MSG(j == "json true\n" || j.find("cut off at 8 tokens") != std::string::npos, "got: %s", j.c_str());
    script::options big = o; big.ask_max_tokens = 64;
    auto j2 = run("try { auto v = forge::ask_json(\"give json\"); std::println(\"json {}\", json.dump(v).size() > 0); } catch (const AIError& e) { std::println(\"{}\", e.what()); }", big);
    CHECK_MSG(j2 == "json true\n" || j2.find("cut off at 64 tokens") != std::string::npos, "got: %s", j2.c_str());
    std::printf("ask_json on the tiny model: [%s] [%s]\n", j.c_str(), j2.c_str());
    script::set_local_ai(nullptr);
    stdfs::remove(path);
}

#include <forge/graph/rpc.hpp>
TEST(remote_scripts) {
    graph::rpc::server_security sec; sec.tokens = {"tok"}; sec.allow_scripts = true;
    graph::rpc::server srv(nullptr, 0, "127.0.0.1", sec);
    srv.script_handler = [](const std::string& src, const std::string& name, const std::vector<std::string>& argv, std::uint32_t) {
        script::options o; std::string out; o.argv = argv; o.out = [&](std::string_view s) { out += s; };
        script::engine e(o); auto r = e.run(src, name);
        return graph::rpc::server::script_result{r.ok && r.exit_code == 0, r.error_type, r.error, out};
    };
    srv.start();
    std::string addr = "127.0.0.1:" + std::to_string(srv.port());
    auto got = run("auto r = forge::remote(\"" + addr + "\", \"std::println(\\\"hi {}\\\", args.size());\", {{\"token\", \"tok\"}, {\"args\", std::vector<std::string>{\"a\"}}});\n"
                   "std::println(\"{} {}\", r[\"ok\"], r[\"output\"]);\n"
                   "auto b = forge::remote(\"" + addr + "\", \"throw std::runtime_error(\\\"x\\\");\", {{\"token\", \"tok\"}});\n"
                   "std::println(\"{}\", b[\"ok\"]);\n"
                   "try { forge::remote(\"" + addr + "\", \"1;\", {{\"token\", \"nope\"}}); } catch (const PermissionError& e) { std::println(\"denied\"); }\n");
    CHECK_MSG(got == "true hi 1\n\nfalse\ndenied\n", "got:\n%s", got.c_str());
    srv.stop();
}

// forge::remote_call ships a function's source plus its captured plain values; the node runs it like a remote
// script (here: an engine with fs denied, standing in for `rpc-serve --script-no-fs`) and the result comes back typed.
TEST(remote_call_functions) {
    graph::rpc::server_security sec; sec.tokens = {"tok"}; sec.allow_scripts = true;
    graph::rpc::server srv(nullptr, 0, "127.0.0.1", sec);
    std::string first_src;
    srv.script_handler = [&](const std::string& src, const std::string& name, const std::vector<std::string>& argv, std::uint32_t) {
        if (first_src.empty()) first_src = src;
        script::options o; std::string out; o.argv = argv; o.allow_fs = false; o.os_sandbox = script::options::os_sandbox_mode::off;
        o.out = [&](std::string_view s) { out += s; };
        script::engine e(o); auto r = e.run(src, name);
        return graph::rpc::server::script_result{r.ok && r.exit_code == 0, r.error_type, r.error, out};
    };
    srv.start();
    std::string node = "\"127.0.0.1:" + std::to_string(srv.port()) + "\"", tok = "{{\"token\", \"tok\"}}";
    auto got = run(
        "int fact(int n) { return n < 2 ? 1 : n * fact(n - 1); }\n"
        "auto scale = 2.5;\n"
        "std::map<std::string, int> counts = {{\"a\", 1}};\n"
        "auto fib = [](int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); };\n"
        "int main() {\n"
        "    int k = 10; std::vector<int> xs = {1, 2, 3};\n"
        "    auto twice = [](int v) { return v * 2; };\n"
        "    auto f = [k, xs, &twice](int a, double b) { std::println(\"on node {}\", a); int s = 0; for (auto x : xs) s += x;\n"
        "        return std::vector<std::any>{a + k + s, b * scale, fact(5), counts[\"a\"], counts[\"zz\"], twice(a), \"q\\\"\\n\"}; };\n"
        "    auto r = forge::remote_call(" + node + ", f, {1, 2.0}, " + tok + ");\n"
        "    std::println(\"{} {} {}\", r, r[1] == 5.0, r[0] == 17);\n"
        "    std::println(\"{} {}\", forge::remote_call(" + node + ", fact, {6}, " + tok + "), forge::remote_call(" + node + ", fib, {10}, " + tok + "));\n"
        "    std::println(\"{}\", forge::remote_call(" + node + ", [](auto m) { m[\"n\"] = m.size(); return m; }, {std::map<std::string, int>{{\"b\", 2}}}, " + tok + "));\n"
        "    try { forge::remote_call(" + node + ", []() { throw std::invalid_argument(\"boom\"); }, {}, " + tok + "); } catch (const ValueError& e) { std::println(\"ValueError {}\", e.what()); }\n"
        "    try { forge::remote_call(" + node + ", []() { fs.write(\"forge_rc_x\", \"y\"); }, {}, " + tok + "); } catch (const PermissionError& e) { std::println(\"PermissionError\"); }\n"
        "    auto h = fs.open(\"tests/test_script.cpp\");\n"
        "    try { forge::remote_call(" + node + ", [&]() { return h; }, {}, " + tok + "); } catch (const TypeError& e) { std::println(\"{}\", e.what()); }\n"
        "    try { forge::remote_call(" + node + ", [](auto x) { return x; }, {h}, " + tok + "); } catch (const TypeError& e) { std::println(\"{}\", e.what()); }\n"
        "    try { forge::remote_call(" + node + ", [](auto x) { return x; }, {1}, {{\"token\", \"nope\"}}); } catch (const PermissionError& e) { std::println(\"denied\"); }\n"
        "    return 0;\n"
        "}\n");
    CHECK_MSG(got == "on node 1\n[17, 5, 120, 1, 0, 2, \"q\\\"\\n\"] true true\n720 55\n{b: 2, n: 1}\nValueError remote: boom\nPermissionError\n"
                     "forge::remote_call: captured 'h' is a handle; only nil, bool, numbers, strings, lists and maps (and script functions over them) can be sent to a remote node\n"
                     "forge::remote_call: argument 1 is a handle; only nil, bool, numbers, strings, lists and maps (and script functions over them) can be sent to a remote node\n"
                     "denied\n", "got:\n%s", got.c_str());
    // captured k, xs, twice and the globals scale, counts, fact travel with f; built-ins (std::println) do not
    for (const char* want : {"auto k = forge::__unmarshal(", "auto xs = forge::__unmarshal(", "auto scale = forge::__unmarshal(", "auto counts = forge::__unmarshal(",
                             "auto twice = [&](int v)", "int fact(int n)", "auto __forge_remote_fn = [&](int a, double b)"})
        CHECK_MSG(first_src.find(want) != std::string::npos, "missing %s in shipped:\n%s", want, first_src.c_str());
    // the typed encoding keeps ints and doubles apart and survives a round trip
    EXPECT("auto v = std::vector<std::any>{1, 2.0, -0.5, \"x\", true, nullptr, std::vector<int>{3}};\n"
           "auto w = forge::__unmarshal(forge::__marshal(v)); std::println(\"{} {} {}\", w, w == v, w[1] == 2.0);",
           "[1, 2, -0.5, \"x\", true, nil, [3]] true true\n");
    EXPECT("forge::__unmarshal(\"zz\");", "!ValueError");
    srv.stop();
}

// Golden test for every shell backend found on PATH (plus $FORGE_TEST_SHELL_PATH): a hostile literal
// must round-trip unchanged through {} quoting. Backends not installed are reported and skipped.
#include <cstdlib>
TEST(shell_backends_golden) {
    if (const char* extra = std::getenv("FORGE_TEST_SHELL_PATH")) { std::string p = std::string(extra) + ":" + (std::getenv("PATH") ? std::getenv("PATH") : ""); ::setenv("PATH", p.c_str(), 1); }
    const char* printer[][2] = {
        {"sh", "printf '%s\\n' {}"}, {"bash", "printf '%s\\n' {}"}, {"zsh", "printf '%s\\n' {}"}, {"dash", "printf '%s\\n' {}"},
        {"ash", "printf '%s\\n' {}"}, {"ksh", "printf '%s\\n' {}"}, {"mksh", "printf '%s\\n' {}"}, {"oksh", "printf '%s\\n' {}"},
        {"yash", "printf '%s\\n' {}"}, {"posh", "printf '%s\\n' {}"}, {"osh", "printf '%s\\n' {}"}, {"ysh", "write -- {}"}, {"hush", "printf '%s\\n' {}"},
        {"fish", "printf '%s\\n' {}"}, {"tcsh", "printf '%s\\n' {}"}, {"csh", "printf '%s\\n' {}"},
        {"nu", "print {}"}, {"xonsh", "print({})"}, {"elvish", "echo {}"}, {"pwsh", "Write-Output {}"}, {"powershell", "Write-Output {}"},
        {"cmd", "echo {}"}, {"rc", "echo {}"}, {"es", "echo {}"},
    };
    const std::string hostile = "a 'b' \"c\" $HOME `x` \\n; | & * ? ~ # ! %PATH% {}";
    std::string ran, skipped;
    for (auto& [name, tmpl] : printer) {
        auto got = run(std::string("auto av = false; for (auto& s : forge::shells()) if (s[\"name\"] == \"") + name + "\") av = s[\"available\"];\n"
                       "if (!av) { std::print(\"SKIP\"); } else { auto r = forge::" + name + "(R\"T(" + tmpl + ")T\", R\"H(" + hostile + ")H\"); std::print(\"{}\", r.out); }\n");
        if (got == "SKIP") { skipped += std::string(" ") + name; continue; }
        if (std::string(name) == "cmd") continue;   // cmd echo keeps the quotes; covered by the Windows/wine job
        CHECK_MSG(got == hostile + "\n" || got == hostile + "\r\n", "%s: got [%s]", name, got.c_str());
        ran += std::string(" ") + name;
    }
    std::printf("shells tested:%s\nshells skipped (not installed):%s\n", ran.c_str(), skipped.c_str());
}

// A file on PATH that only has a shell's name is not that shell (npm ships an `rc`, a config-file tool): discovery
// runs the backend's identity probe, skips an impostor and looks further along PATH for the real one.
#if !defined(_WIN32)
#include <cstring>
#include <unistd.h>
TEST(shell_identity_probe) {
    namespace stdfs = std::filesystem;
    auto dir = stdfs::temp_directory_path() / ("forge_fakeshell_" + std::to_string(::getpid()));
    stdfs::create_directories(dir);
    const char* probed[] = {"rc", "es", "fish", "elvish", "nu", "ysh", "xonsh"};
    for (const char* n : probed) {
        auto p = dir / n; { std::ofstream(p) << "#!/bin/sh\necho '{ \"_\": [] }'\n"; }
        stdfs::permissions(p, stdfs::perms::owner_all);
    }
    std::string old = std::getenv("PATH") ? std::getenv("PATH") : "";
    ::setenv("PATH", (dir.string() + ":" + old).c_str(), 1);
    auto got = run("for (auto& s : forge::shells()) std::print(\";{}={}|{}\", s[\"name\"], s[\"exe\"], s[\"available\"]);");
    CHECK_MSG(got.find(dir.string()) == std::string::npos, "an impostor was taken for a shell: %s", got.c_str());
    for (const char* n : probed) {
        auto at = got.find(";" + std::string(n) + "="); CHECK_MSG(at != std::string::npos, "%s missing from forge::shells()", n);
        if (at == std::string::npos) continue;
        auto v = got.substr(at + std::strlen(n) + 2, got.find(';', at + 1) - at - std::strlen(n) - 2);
        // not installed at all, or the real one further along PATH, named by its full path
        CHECK_MSG(v == "|false" || (v[0] == '/' && v.ends_with("|true")), "%s resolved to [%s]", n, v.c_str());
    }
    // running it reaches the real shell or says it is not installed; never the impostor's output
    auto ran = run("try { auto r = forge::rc(\"echo a^b\"); std::print(\"{}\", r.out); } catch (const ShellError& e) { std::print(\"{}\", e.what()); }");
    CHECK_MSG(ran == "ab\n" || ran.find("is not installed") != std::string::npos, "forge::rc gave [%s]", ran.c_str());
    // without the right to start processes nothing is probed: the listing is PATH lookup only
    script::options o; std::string out; o.allow_proc = false; o.os_sandbox = script::options::os_sandbox_mode::off;
    o.out = [&](std::string_view s) { out += s; };
    script::engine e(o); auto r = e.run("for (auto& s : forge::shells()) if (s[\"name\"] == \"rc\") std::print(\"{}\", s[\"exe\"]);", "t");
    CHECK_MSG(r.ok && out == "rc", "no-proc listing gave [%s] %s", out.c_str(), r.error.c_str());
    ::setenv("PATH", old.c_str(), 1);
    stdfs::remove_all(dir);
}
#endif

// "Write once, emit anywhere": every golden script is emitted for every backend that has an emitter and is
// installed, run by that shell without FORGE, and must match the interpreter's output and exit code.
#include <array>
#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif
namespace {
std::string slurp_file(const std::string& p) { std::ifstream f(p, std::ios::binary); return {std::istreambuf_iterator<char>(f), {}}; }
std::pair<std::string, int> sh_capture(const std::string& cmd) {
    std::string out; std::array<char, 4096> buf{};
    FILE* p = ::popen(cmd.c_str(), "r");
    if (!p) return {"", -1};
    size_t n; while ((n = std::fread(buf.data(), 1, buf.size(), p)) > 0) out.append(buf.data(), n);
    int st = ::pclose(p);
    return {out, WIFEXITED(st) ? WEXITSTATUS(st) : -1};
}
const char* runner(const std::string& s) {
    if (s == "ash" || s == "busybox") return "busybox sh"; if (s == "nu") return "nu --no-config-file"; if (s == "xonsh") return "xonsh --no-rc";
    if (s == "elvish") return "elvish -norc"; if (s == "pwsh") return "pwsh -NoProfile -NonInteractive -File";
    if (s == "powershell") return "powershell -NoProfile -File"; if (s == "tcsh" || s == "csh") return "tcsh -f";
    if (s == "cmd.exe" || s == "cmd") return "cmd.exe /d /c";
    return nullptr;
}
}
TEST(emit_golden_all_shells) {
    if (const char* extra = std::getenv("FORGE_TEST_SHELL_PATH")) { std::string p = std::string(extra) + ":" + (std::getenv("PATH") ? std::getenv("PATH") : ""); ::setenv("PATH", p.c_str(), 1); }
    namespace stdfs = std::filesystem;
    auto dir = stdfs::temp_directory_path() / ("forge_emit_" + std::to_string(::getpid()));
    stdfs::create_directories(dir);
    auto avail = run("for (auto& s : forge::shells()) if (s[\"available\"] && s[\"emit\"]) std::print(\"\\t{}={}\\t\", s[\"name\"], s[\"exe\"]);");   // tabs: Windows paths have spaces
    std::string tested, refused;
    for (const char* g : {"functions", "flat", "batch"}) {
        std::string src = slurp_file("tests/data/emit/" + std::string(g) + ".cpp");
        CHECK_MSG(!src.empty(), "missing golden %s", g);
        auto old = stdfs::current_path(); stdfs::current_path(dir);
        script::options o; std::string want; o.out = [&](std::string_view s) { want += s; };
        script::engine e(o); auto r = e.run_file(stdfs::absolute(old / "tests/data/emit" / (std::string(g) + ".cpp")).string());
        stdfs::current_path(old);
        if (!r.ok && r.error.find("is not installed") != std::string::npos) {   // e.g. no sh on a bare Windows box
            refused += " " + std::string(g) + "/*(" + r.error.substr(0, r.error.find('\n')) + ")"; continue;
        }
        CHECK_MSG(r.ok, "%s interp: %s", g, r.error.c_str());
        for (auto& name : script::emit_targets()) {
            auto at = avail.find("\t" + name + "=");
            if (at == std::string::npos) continue;
            auto exe = avail.substr(at + name.size() + 2, avail.find('\t', at + 1) - at - name.size() - 2);
            auto full = exe;   // resolved path: the test's own shell (cmd.exe on Windows) may not share forge's PATH lookup
            exe = stdfs::path(exe).stem().string();
            std::string code;
            try { code = script::emit(src, g, name); }
            catch (const script::emit_error&) { refused += " " + std::string(g) + "/" + name; continue; }
            // PowerShell refuses -File without .ps1
            std::string ext = (name == "pwsh" || name == "powershell") ? "ps1" : name;
            auto path = (dir / (std::string(g) + "." + ext)).string();
            { std::ofstream(path, std::ios::binary) << code; }
            const char* rn = runner(exe);
            std::string cmdline = rn ? std::string(rn) : "\"" + full + "\"";   // the resolved path: cmd.exe would pick System32\\bash.exe (WSL) first
            if (exe == "busybox") cmdline = "\"" + full + "\" sh";        // ash via busybox: run the binary forge found
            else if (exe == "ash") cmdline = "\"" + full + "\"";
#if defined(_WIN32)
            auto [got, rc] = sh_capture("cd /d \"" + dir.string() + "\" && " + cmdline + " \"" + path + "\" 2>&1");
#else
            // xonsh 0.14 on Python 3.13 prints an ast DeprecationWarning per operator to stderr the first time it
            // parses a script (PYTHONWARNINGS does not reach its parser), so its stderr is left out of the comparison.
            auto [got, rc] = sh_capture("cd '" + dir.string() + "' && " + cmdline + " '" + path + (exe == "xonsh" ? "' 2>/dev/null" : "' 2>&1"));
#endif
            if (name == "cmd") std::erase(got, '\r');   // batch output is CRLF
            CHECK_MSG(got == want, "%s/%s output:\n%s\nwant:\n%s", g, name.c_str(), got.c_str(), want.c_str());
            CHECK_MSG(rc == r.exit_code, "%s/%s exit %d want %d", g, name.c_str(), rc, r.exit_code);
            tested += " " + std::string(g) + "/" + name + (exe != name ? "(via " + exe + ")" : "");
        }
    }
    stdfs::remove_all(dir);
    std::printf("emit tested:%s\nemit refused (outside the target's subset):%s\n", tested.c_str(), refused.c_str());
}

// Emitted text for the shells that are rarely installed (ysh, rc, es) is compared with checked-in goldens in
// tests/data/emit/expected/, so emitter changes show up even where emit_golden_all_shells skips them.
// FORGE_UPDATE_EMIT_GOLDEN=1 rewrites the files (then run the outputs with the real shells before committing).
TEST(emit_text_golden) {
    bool update = std::getenv("FORGE_UPDATE_EMIT_GOLDEN") != nullptr;
    for (const char* sh : {"ysh", "rc", "es"}) {
        auto targets = script::emit_targets();
        CHECK_MSG(std::find(targets.begin(), targets.end(), sh) != targets.end(), "%s has no emitter", sh);
        for (const char* g : {"functions", "flat", "batch"}) {
            std::string src = slurp_file("tests/data/emit/" + std::string(g) + ".cpp");
            std::string got;
            try { got = script::emit(src, g, sh); } catch (const script::emit_error& e) { CHECK_MSG(false, "%s/%s: %s", g, sh, e.what()); continue; }
            auto path = "tests/data/emit/expected/" + std::string(g) + "." + sh;
            if (update) { std::ofstream(path, std::ios::binary) << got; continue; }
            auto want = slurp_file(path); std::erase(want, '\r');   // CRLF checkout (core.autocrlf on Windows)
            CHECK_MSG(got == want, "%s differs from the emitted text (FORGE_UPDATE_EMIT_GOLDEN=1 to refresh):\n%s", path.c_str(), got.c_str());
        }
    }
}

TEST(emit_errors_and_host_shell) {
    auto refuse = [](const std::string& src, const std::string& t, const char* what) {
        bool threw = false;
        try { script::emit(src, "x.cpp", t); } catch (const script::emit_error& e) { threw = true; CHECK_MSG(std::string(e.what()).find(what) != std::string::npos, "%s: %s", t.c_str(), e.what()); }
        CHECK_MSG(threw, "%s accepted unsupported script", t.c_str());
    };
    refuse("class A { int x; };\nint main() { return 0; }", "bash", "class");
    refuse("int f() { return 1; }\nint main() { return f(); }", "tcsh", "functions");
    refuse("int main() { auto s = fs.read(\"x\"); return 0; }", "tcsh", "fs.read");
    refuse("int main() { std::string a = \"x\"; if (a < \"y\") return 1; return 0; }", "rc", "string ordering");
    refuse("int main() { return 0; }", "nope", "");
    auto bash = script::emit("int main() { std::println(\"hi\"); return 0; }", "x.cpp", "bash");
    CHECK(bash.starts_with("#!"));
    CHECK(script::emit("int main() { return 0; }", "x.cpp", "android").starts_with("#!/system/bin/sh"));
    CHECK(!script::host_shell().empty());
    auto got = run("auto r = forge::shell(\"echo hi\"); std::print(\"{} {} {}\", r.out, r.code, forge::host_shell());");
    CHECK_MSG(got == "hi\n 0 " + script::host_shell() || got == "hi\r\n 0 " + script::host_shell(), "got [%s]", got.c_str());   // cmd ends lines with CRLF
}

TEST(host_bindings) {
    for (bool vm : {false, true}) {
        script::options o;
        o.use_vm = vm;
        std::string text;
        o.out = [&](std::string_view s) { text += s; };
        script::engine e(o);
        std::uint64_t kept = 0;
        e.define("host::add", 2, 2, [](const std::string& a) {
            // args arrive as a JSON array of two numbers
            double x = 0, y = 0;
            std::sscanf(a.c_str(), "[%lf,%lf]", &x, &y);
            return std::to_string(x + y);
        });
        e.define("host::keep", 1, 1, [&](const std::string& a) {
            const auto p = a.find("\"$callback\":");
            CHECK_MSG(p != std::string::npos, "no callback in %s", a.c_str());
            kept = std::stoull(a.substr(p + 12));
            return std::string("null");
        });
        e.define("host::boom", 0, 0, [](const std::string&) -> std::string { throw std::runtime_error("boom"); });
        auto r = e.run(R"(
            std::print("{} ", host::add(2, 3));
            host::keep([](auto v) { std::print("cb {} ", v); });
            bool caught = false;
            try { host::boom(); } catch (const std::exception& x) { caught = true; }
            std::print("{}", caught);
        )", "<host>");
        CHECK_MSG(r.ok, "%s", r.error.c_str());
        CHECK_MSG(text == "5 true", "got [%s]", text.c_str());
        CHECK(kept != 0);
        CHECK(e.callbacks() == 1);
        auto c = e.call(kept, "[7]");
        CHECK_MSG(c.ok, "%s", c.error.c_str());
        CHECK_MSG(text == "5 truecb 7 ", "got [%s]", text.c_str());
        e.release(kept);
        CHECK(e.callbacks() == 0);
        CHECK(!e.call(kept, "[]").ok);
    }
}

TEST(limit_block_timeout_mem_cpu) {
    // timeout (ms): wall clock for the block only; the outer script keeps running afterwards
    EXPECT("try { forge::limit(std::map<std::string, int>{{\"timeout\", 30}}, [] { while (true) {} }); } catch (const std::exception& e) { std::println(\"t\"); } std::println(\"after\");", "t\nafter\n");
    EXPECT("forge::limit(std::map<std::string, int>{{\"timeout\", 30}}, [] { while (true) {} });", "!TimeoutError");
    // cpu_s: CPU time of the running thread
    EXPECT("forge::limit(std::map<std::string, double>{{\"cpu_s\", 0.05}}, [] { while (true) {} });", "!LimitExceededError");
    EXPECT("int r = forge::limit(std::map<std::string, double>{{\"cpu_s\", 5.0}}, [] { return 41 + 1; }); std::println(\"{}\", r);", "42\n");
    // mem (bytes): the heap may grow by at most this much inside the block
    EXPECT("forge::limit(std::map<std::string, int>{{\"mem\", 65536}}, [] { std::vector<std::string> v; for (int i = 0; i < 100000; ++i) v.push_back(std::string(64, 'x')); });", "!OutOfMemoryError");
    EXPECT("forge::limit(std::map<std::string, int>{{\"mem\", 1048576}}, [] { std::vector<int> v; v.push_back(1); std::println(\"ok\"); });", "ok\n");
}

// limit {...} { } syntax errors and the static checker's limit-option diagnostics (golden: tests/script/limit_*.fs)
TEST(limit_block_syntax_and_check) {
    EXPECT("limit {.timeout = 5} { return; }", "!SyntaxError");                          // return would only leave the block
    EXPECT("int f() { limit {.timeout = 5} { return 1; } return 0; }", "!SyntaxError");
    EXPECT("auto r = limit {.timeout = 5000} { return 7; }; std::println(\"{}\", r);", "7\n");   // expression form may return
    EXPECT("for (int i = 0; i < 3; ++i) { limit {.timeout = 5} { break; } }", "!SyntaxError");
    EXPECT("while (true) { limit {.timeout = 5} { continue; } }", "!SyntaxError");
    EXPECT("limit {.timeuot = 5} { }", "!SyntaxError");                                    // unknown key
    EXPECT("limit {.timeout = 5, .timeout = 6} { }", "!SyntaxError");                      // repeated key
    EXPECT("limit {{\"mem\", \"2G\"}} { std::println(\"ok\"); }", "ok\n");
    EXPECT("int limit = 3; std::println(\"{}\", limit);", "3\n");                          // still a plain name
    EXPECT("auto limit(int x) { return x + 1; } std::println(\"{}\", limit(1));", "2\n");
    // literal option values are checked before the script runs
    auto ids = [](const std::string& src) { std::string s; for (auto& d : script::check(src, "<t>")) if (d.id == "limit-option") s += d.message + "\n"; return s; };
    CHECK(ids("limit {.mem = \"2G\", .timeout = \"200ms\", .proc_cpu = \"1.5s\", .allocs = 10} { }").empty());
    CHECK(!ids("limit {.mem = \"2X\"} { }").empty());
    CHECK(!ids("limit {.timeout = \"5 fortnights\"} { }").empty());
    CHECK(!ids("limit {.handles = \"5\"} { }").empty());
    EXPECT("limit {.mem = \"lots\"} { std::println(\"ran\"); }", "!StaticCheckError");
    // with --no-check the runtime raises the same problem as ValueError
    script::options nc; nc.static_check = false;
    auto got = run("limit {.mem = \"lots\"} { std::println(\"ran\"); }", nc);
    CHECK_MSG(got == "!ValueError", "got %s", got.c_str());
}

// Capability escapes with the run's own capabilities off (the golden sandbox_escape.fs narrows them with forge::sandbox).
TEST(capability_escape_engine_level) {
    script::options o; o.allow_fs = false; o.allow_proc = false; o.os_sandbox = script::options::os_sandbox_mode::off;
    auto expect_denied = [&](const std::string& src) {
        auto got = run(src, o);
        CHECK_MSG(got == "!PermissionError" || got == "!SandboxViolationError" || got == "!StaticCheckError", "%s -> %s", src.c_str(), got.c_str());
    };
    expect_denied("auto f = forge::fs::read; std::vector<std::string> v{\"/etc/hostname\"}; for (auto& p : v) f(p);");
    expect_denied("eval(\"forge::fs::read(\\\"/etc/hostname\\\");\");");
    expect_denied("std::string name = \"forge::sh\"; forge::reflect::call(name, \"echo hi\");");
    expect_denied("std::async([] { return forge::fs::read(\"/etc/hostname\"); }).get();");
    expect_denied("limit {.timeout = \"5s\"} { auto s = forge::bash; s(\"echo hi\"); }");
    expect_denied("forge::sandbox(std::map<std::string, bool>{{\"fs\", true}}, [] { auto r = forge::fs::read; r(\"/etc/hostname\"); });");
}

// options::mem_limit_bytes (forge script --mem-limit): a hard cap on the managed heap for the whole run. Same
// accounting and the same catchable OutOfMemoryError as --heap / limit {mem}; a limit block cannot widen it.
// (tests/script/mem_containers.fs covers the accounting per container kind and for strings.)
TEST(mem_limit_whole_run) {
    script::options o; o.mem_limit_bytes = 4u << 20;
    auto grows = [&](const std::string& what, const std::string& body) {
        auto got = run(body, o);
        CHECK_MSG(got == "!OutOfMemoryError", "%s -> %s", what.c_str(), got.c_str());
    };
    grows("vector<string>", "std::vector<std::string> v; while (true) v.push_back(std::string(256, 'x'));");
    grows("vector<int>", "std::vector<int> v; while (true) v.push_back(1);");
    grows("deque", "#include <deque>\nstd::deque<int> d; while (true) d.push_back(1);");
    grows("list", "#include <list>\nstd::list<int> l; while (true) l.push_back(1);");
    grows("map", "std::map<int, int> m; int i = 0; while (true) { m[i] = i; ++i; }");
    grows("unordered_map", "std::unordered_map<std::string, int> m; int i = 0; while (true) { m[std::to_string(i)] = i; ++i; }");
    grows("set", "#include <set>\nstd::set<int> s; int i = 0; while (true) { s.insert(i); ++i; }");
    grows("unordered_set", "#include <unordered_set>\nstd::unordered_set<int> s; int i = 0; while (true) { s.insert(i); ++i; }");
    grows("string +=", "std::string s = \"x\"; while (true) s += s;");
    grows("string append", "std::string s; while (true) s.append(4096, 'x');");
    grows("string push_back", "std::string s; while (true) s.push_back('x');");
    grows("string(n, c)", "int n = 1024; while (true) { std::string s(n, 'x'); n *= 2; }");
    grows("objects", "struct P { std::string a; int b = 0; }; std::vector<P> v; while (true) v.push_back(P{\"p\", 1});");
    // the message names the limit as it was given
    { auto a = run1("std::vector<int> v; while (true) v.push_back(1);", true, o);
      CHECK_MSG(a.r.error.find("limit of 4194304 bytes exceeded (--mem-limit)") != std::string::npos, "%s", a.r.error.c_str()); }
    // catchable, like the other heap limits; once the memory is released the script carries on
    CHECK(run("try { std::vector<std::string> v; while (true) v.push_back(std::string(256, 'x')); } catch (const forge::OutOfMemoryError& e) { std::println(\"caught\"); }\n"
              "std::vector<int> w; for (int i = 0; i < 1000; ++i) w.push_back(i); std::println(\"{}\", w.size());", o) == "caught\n1000\n");
    CHECK(run("try { std::string s = \"x\"; while (true) s += s; } catch (const std::bad_alloc& e) { std::println(\"bad_alloc\"); }", o) == "bad_alloc\n");
    // still over the cap after the catch: raised again (the handler only gets the usual grace)
    CHECK(run("std::vector<std::string> v;\ntry { while (true) v.push_back(std::string(256, 'x')); } catch (const forge::OutOfMemoryError& e) { std::println(\"caught\"); }\n"
              "while (true) v.push_back(std::string(256, 'x'));", o) == "caught\n!OutOfMemoryError");
    // below the limit nothing fires
    CHECK(run("std::vector<std::string> v; for (int i = 0; i < 2000; ++i) v.push_back(std::string(256, 'x'));\n"
              "std::map<int, int> m; for (int i = 0; i < 2000; ++i) m[i] = i; std::println(\"{} {}\", v.size(), m.size());", o) == "2000 2000\n");
    // a limit block can narrow the cap but not widen it
    CHECK(run("limit {.mem = \"1G\"} { std::vector<int> v; while (true) v.push_back(1); }", o) == "!OutOfMemoryError");
    CHECK(run("try { limit {.mem = \"64K\"} { std::vector<int> v; for (int i = 0; i < 100000; ++i) v.push_back(1); } } catch (const forge::OutOfMemoryError& e) { std::println(\"inner\"); }\n"
              "std::vector<int> w; for (int i = 0; i < 50000; ++i) w.push_back(i); std::println(\"{}\", w.size());", o) == "inner\n50000\n");
    // with --heap as well, the lower of the two applies
    { script::options both = o; both.heap_bytes = 256u << 10;
      CHECK(run("std::vector<int> v; for (int i = 0; i < 60000; ++i) v.push_back(1); std::println(\"ok\");", both) == "!OutOfMemoryError");
      CHECK(run("std::vector<int> v; for (int i = 0; i < 60000; ++i) v.push_back(1); std::println(\"ok\");", o) == "ok\n"); }
    // a task shares the cap
    CHECK(run("auto t = std::async([] { std::vector<int> v; while (true) v.push_back(1); return 0; }); t.get();", o) == "!OutOfMemoryError");
    // forge::gc::stats() shows it, and the cap goes away with the engine (the heap is process-wide)
    CHECK(run("std::println(\"{}\", forge::gc::stats()[\"hard_cap\"] > 0);", o) == "true\n");
    EXPECT("std::println(\"{}\", forge::gc::stats()[\"hard_cap\"]);", "0\n");
    EXPECT("std::vector<std::string> v; for (int i = 0; i < 40000; ++i) v.push_back(std::string(256, 'x')); std::println(\"{}\", v.size());", "40000\n");
}

// Size strings for --mem-limit: the syntax of limit {mem: "2G"}.
TEST(mem_limit_size_strings) {
    auto ok = [](const char* s, std::uint64_t want) { std::uint64_t b = 0; std::string why; bool r = script::parse_size(s, b, &why); CHECK_MSG(r && b == want, "%s -> %llu (%s)", s, static_cast<unsigned long long>(b), why.c_str()); };
    ok("4096", 4096); ok("64K", 64ull << 10); ok("64k", 64ull << 10); ok("512M", 512ull << 20); ok("512MB", 512ull << 20); ok("512MiB", 512ull << 20);
    ok("2G", 2ull << 30); ok("1.5G", 3ull << 29); ok("1T", 1ull << 40); ok("100B", 100); ok(" 8 M ", 8ull << 20);
    for (const char* bad : {"", "lots", "12Q", "M", "-1", "1e3", "99999999999T"}) {
        std::uint64_t b = 7; std::string why;
        CHECK_MSG(!script::parse_size(bad, b, &why) && !why.empty(), "'%s' should be refused", bad);
    }
    std::uint64_t b = 0; CHECK(!script::parse_size("x", b));   // err is optional
}

// The cap reaches child processes: a child started by the script runs under an OS memory cap of the same size
// (spawn_spec::limits.memory_bytes, as limit {proc_mem}), and a ProcessPool worker's engine gets the same limit.
TEST(mem_limit_children) {
    script::options o; o.mem_limit_bytes = 512u << 20;
    // ProcessPool: the worker's own interpreter enforces it
    auto got = run("auto pp = ProcessPool(1);\n"
                   "std::println(\"{}\", pp.submit(\"auto task(auto x) { return forge::gc::stats()[\\\"hard_cap\\\"] > 0; }\", 0).get());\n"
                   "pp.shutdown();", o);
    CHECK_MSG(got == "true\n", "%s", got.c_str());
    EXPECT("auto pp = ProcessPool(1);\nstd::println(\"{}\", pp.submit(\"auto task(auto x) { return forge::gc::stats()[\\\"hard_cap\\\"]; }\", 0).get());\npp.shutdown();", "0\n");
#if defined(__linux__) && !defined(__ANDROID__)
    // RLIMIT_AS in the child: `ulimit -v` prints it in KiB
    if (std::filesystem::exists("/bin/sh")) {
        auto lim = run("std::print(\"{}\", forge::sh(\"ulimit -v\").out);", o);
        CHECK_MSG(lim == "524288\n", "ulimit -v in a child: %s", lim.c_str());
        // limit {proc_mem} can narrow it, not widen it
        CHECK(run("limit {.proc_mem = \"256M\"} { std::print(\"{}\", forge::sh(\"ulimit -v\").out); }", o) == "262144\n");
        CHECK(run("limit {.proc_mem = \"4G\"} { std::print(\"{}\", forge::sh(\"ulimit -v\").out); }", o) == "524288\n");
    }
#endif
}
