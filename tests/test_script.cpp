// forge::script: language, runtime, ARC/cycles, limits, OS bindings. Every program runs through both the
// tree-walker and the bytecode VM and must produce identical output.
#include <forge/script/script.hpp>
#include "check.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstdint>

using namespace forge;

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
    EXPECT("auto p = std::make_unique<int>(5);\nauto q = std::move(p);\nstd::println(\"{}\", *q);\nstd::println(\"{}\", *p);", "5\n!BorrowError");
    EXPECT("std::vector<int> a{1, 2};\nauto b = std::move(a);\nstd::println(\"{}\", b);\na.size();", "[1, 2]\n!BorrowError");
    EXPECT("auto s = std::make_shared<std::vector<int>>();\nauto t = s;\nt->push_back(1);\nstd::println(\"{} {}\", s->size(), t->size());", "1 1\n");
    EXPECT("struct N { int v = 0; };\nauto p = std::make_shared<N>();\nstd::weak_ptr<N> w = p;\nstd::println(\"{}\", w.expired());\np = nullptr;\nstd::println(\"{}\", w.expired());", "false\ntrue\n");
    EXPECT("auto p = std::make_unique<int>(1);\nauto q = p;", "!BorrowError");
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
    EXPECT("std::vector<int> v{1, 2};\ntry { for (int x : v) v.push_back(x); } catch (const BorrowError& e) { std::println(\"{}\", e.what()); }\nv.push_back(3); std::println(\"{}\", v.size());",
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
    code += std::string("auto h = forge::plugin(\"") + FORGE_TEST_PLUGIN + "\");\n" + R"CPP(
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
    script::options so; so.allow_proc = false; std::string t2; so.out = [&](std::string_view s) { t2 += s; };
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
std::println("{}", service.unit({{"name", "forge-t"}, {"argv", std::vector<std::string>{"/bin/echo", "a b"}}}).find("\"a b\"") != std::string::npos);
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

#include <forge/flow/rpc.hpp>
TEST(remote_scripts) {
    flow::rpc::server_security sec; sec.tokens = {"tok"}; sec.allow_scripts = true;
    flow::rpc::server srv(nullptr, 0, "127.0.0.1", sec);
    srv.script_handler = [](const std::string& src, const std::string& name, const std::vector<std::string>& argv, std::uint32_t) {
        script::options o; std::string out; o.argv = argv; o.out = [&](std::string_view s) { out += s; };
        script::engine e(o); auto r = e.run(src, name);
        return flow::rpc::server::script_result{r.ok && r.exit_code == 0, r.error_type, r.error, out};
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
    auto avail = run("for (auto& s : forge::shells()) if (s[\"available\"] && s[\"emit\"]) std::print(\" {}={} \", s[\"name\"], s[\"exe\"]);");
    std::string tested, refused;
    for (const char* g : {"functions", "flat", "batch"}) {
        std::string src = slurp_file("tests/data/emit/" + std::string(g) + ".cpp");
        CHECK_MSG(!src.empty(), "missing golden %s", g);
        auto old = stdfs::current_path(); stdfs::current_path(dir);
        script::options o; std::string want; o.out = [&](std::string_view s) { want += s; };
        script::engine e(o); auto r = e.run_file(stdfs::absolute(old / "tests/data/emit" / (std::string(g) + ".cpp")).string());
        stdfs::current_path(old);
        CHECK_MSG(r.ok, "%s interp: %s", g, r.error.c_str());
        for (auto& name : script::emit_targets()) {
            auto at = avail.find(" " + name + "=");
            if (at == std::string::npos) continue;
            auto exe = avail.substr(at + name.size() + 2, avail.find(' ', at + 1) - at - name.size() - 2);
            exe = stdfs::path(exe).filename().string();
            std::string code;
            try { code = script::emit(src, g, name); }
            catch (const script::emit_error&) { refused += " " + std::string(g) + "/" + name; continue; }
            auto path = (dir / (std::string(g) + "." + name)).string();
            { std::ofstream(path, std::ios::binary) << code; }
            const char* rn = runner(exe);
#if defined(_WIN32)
            auto [got, rc] = sh_capture("cd /d \"" + dir.string() + "\" && " + (rn ? std::string(rn) : exe) + " \"" + path + "\" 2>&1");
#else
            auto [got, rc] = sh_capture("cd '" + dir.string() + "' && " + (rn ? std::string(rn) : exe) + " '" + path + "' 2>&1");
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
