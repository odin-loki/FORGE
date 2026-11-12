// Classes: constructors, destructors, value semantics, operators, static members, inheritance, virtual dispatch.
#include <compare>
#include <memory>
#include <print>
#include <string>
#include <vector>

struct Vec2 {
    double x = 0, y = 0;
    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return Vec2(x - o.x, y - o.y); }
    Vec2 operator*(double k) const { return {x * k, y * k}; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    bool operator==(const Vec2& o) const { return x == o.x && y == o.y; }
    bool operator!=(const Vec2& o) const { return !(*this == o); }
    double dot(const Vec2& o) const { return x * o.x + y * o.y; }
};

class Counter {
public:
    Counter() : value_(0) { ++instances; }
    explicit Counter(int start) : value_(start) { ++instances; }
    ~Counter() { --instances; }
    void inc() { ++value_; }
    Counter& add(int n) { value_ += n; return *this; }
    int value() const { return value_; }
    static int live() { return instances; }
    static int instances;
    static constexpr int kMax = 99;
private:
    int value_;
};
int Counter::instances = 0;

struct Version {
    int major = 0, minor = 0;
    auto operator<=>(const Version&) const = default;
};

struct Money {
    long long cents;
    bool operator<(const Money& o) const { return cents < o.cents; }
    Money next() const { return Money{cents + 1}; }
    int operator()(int scale) const { return static_cast<int>(cents) * scale; }
    int plus(int i) const { return static_cast<int>(cents) + i; }
};

struct Tracer {
    std::string name;
    explicit Tracer(std::string n) : name(std::move(n)) { std::println("ctor {}", name); }
    ~Tracer() { std::println("dtor {}", name); }
};

class Animal {
public:
    explicit Animal(std::string n) : name_(std::move(n)) {}
    virtual ~Animal() = default;
    virtual std::string sound() const = 0;
    virtual int legs() const { return 4; }
    std::string describe() const { return name_ + " says " + sound() + " on " + std::to_string(legs()) + " legs"; }
    const std::string& name() const { return name_; }
protected:
    std::string name_;
};
class Dog : public Animal {
public:
    using Animal::Animal;
    std::string sound() const override { return "woof"; }
};
class Bird : public Animal {
public:
    Bird(std::string n, bool flies) : Animal(std::move(n)), flies_(flies) {}
    std::string sound() const override { return flies_ ? "tweet" : "squawk"; }
    int legs() const override { return 2; }
private:
    bool flies_;
};
class Puppy final : public Dog {
public:
    Puppy() : Dog("pup") {}
    std::string sound() const override { return Dog::sound() + "!"; }
};

struct Base {
    int id = 1;
    explicit Base(int i) : id(i) { std::print("Base({}) ", i); }
    virtual ~Base() { std::print("~Base "); }
    virtual std::string who() const { return "base"; }
    std::string hello() const { return "hello from " + who(); }
};
struct Derived : Base {
    std::string tag = "d";
    Derived() : Base(7), tag("derived") { std::print("Derived() "); }
    ~Derived() override { std::print("~Derived "); }
    std::string who() const override { return tag; }
};

struct Config {
    std::string host = "localhost";
    int port = 80;
    bool tls = false;
    std::vector<std::string> tags;
};

struct Matrix {
    std::vector<std::vector<int>> rows;
    Matrix(int r, int c) { for (int i = 0; i < r; ++i) rows.push_back(std::vector<int>(c, 0)); }
    void set(int r, int c, int v) { rows[r][c] = v; }
    int at(int r, int c) const { return rows[r][c]; }
    int trace() const { int t = 0; for (std::size_t i = 0; i < rows.size(); ++i) t += rows[i][i]; return t; }
};

namespace geo {
struct Point { int x, y; };
int manhattan(const Point& a, const Point& b) { return (a.x > b.x ? a.x - b.x : b.x - a.x) + (a.y > b.y ? a.y - b.y : b.y - a.y); }
namespace detail { const int scale = 3; }
}

int main() {
    // operators and value semantics
    Vec2 a{1, 2}, b(3, 4);
    Vec2 c = a + b * 2 - Vec2{1, 1};
    c += a * -1;
    std::println("{} {} {} {} {}", c.x, c.y, a == b, a != b, a.dot(b));
    Vec2 copy = a;
    copy.x = 100;
    std::println("{} {}", a.x, copy.x);
    std::vector<Vec2> pts{{0, 0}, {1, 1}};
    pts.push_back(a);
    pts[2].y = -1;
    std::println("{} {} {}", pts.size(), a.y, pts[2].y);

    // constructors, static members, method chaining, scope-based lifetime
    {
        Counter c1, c2(10);
        Counter c3(c2.value());
        c1.inc(); c3.add(5).add(5).inc();
        std::println("{} {} {} {}", c1.value(), c2.value(), c3.value(), Counter::live());
    }
    std::println("{} {} {}", Counter::live(), Counter::instances, Counter::kMax);

    // comparisons
    Version v1{1, 2}, v2{1, 10};
    std::println("{} {} {} {} {}", v1 < v2, v1 == v2, v1 != v2, v1 >= v2, v1 <= v1);
    Money m{5};
    Money old = m;
    m = m.next().next();
    std::println("{} {} {} {} {}", old.cents, m.cents, old < m, m(10), m.plus(3));

    // destructors: reverse order at scope exit, temporaries, early exit from loops
    {
        Tracer t1("one");
        Tracer t2("two");
        std::println("in scope");
    }
    for (int i = 0; i < 3; ++i) {
        Tracer t("loop" + std::to_string(i));
        if (i == 1) break;
    }
    {
        auto owned = std::make_unique<Tracer>("heap");
        std::println("owned {}", owned->name);
    }
    auto shared = std::make_shared<Tracer>("shared");
    auto second = shared;
    shared.reset();
    std::println("still alive {}", second->name);
    second.reset();
    std::println("released");

    // inheritance and virtual dispatch
    {
        std::vector<std::unique_ptr<Animal>> zoo;
        zoo.push_back(std::make_unique<Dog>("rex"));
        zoo.push_back(std::make_unique<Bird>("tweety", true));
        zoo.push_back(std::make_unique<Bird>("pingu", false));
        zoo.push_back(std::make_unique<Puppy>());
        for (const auto& an : zoo) std::println("{}", an->describe());
        const Animal& ref = *zoo[1];
        std::println("{} {}", ref.name(), ref.legs());
    }
    {
        Derived d;
        std::println("");
        std::println("{} {} {}", d.id, d.who(), d.hello());
    }
    std::println("");

    // aggregates, default member initialisers, designated initialisers
    Config def;
    Config custom{"example.org", 443, true, {"a", "b"}};
    std::println("{} {} {} {}", def.host, def.port, def.tls, def.tags.size());
    std::println("{} {} {} {}", custom.host, custom.port, custom.tls, custom.tags[1]);

    // members that are containers, const and non-const overloads
    Matrix mat(3, 3);
    for (int i = 0; i < 3; ++i) mat.set(i, i, i + 1);
    const Matrix& cm = mat;
    std::println("{} {}", cm.trace(), cm.at(2, 2));

    // namespaces
    geo::Point p{1, 2}, q{4, 6};
    using namespace geo;
    std::println("{} {}", manhattan(p, q), geo::detail::scale);
    return 0;
}
