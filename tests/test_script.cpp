// forge::script: language, runtime, ARC/cycles, limits, OS bindings. Every program runs through both the
// tree-walker and the bytecode VM and must produce identical output.
#include <forge/script/script.hpp>
#include "check.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

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
