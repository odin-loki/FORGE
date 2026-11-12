// Classes, destructors, value semantics, smart pointers, closures: reference counts and copy elision are observable here.
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct Noisy {
    std::string name;
    Noisy(std::string n) : name(n) { std::println("+{}", name); }
    ~Noisy() { std::println("-{}", name); }
};
struct Vec2 {
    int x = 0; int y = 0;
    Vec2 operator+(const Vec2& o) const { Vec2 r; r.x = x + o.x; r.y = y + o.y; return r; }
    bool operator==(const Vec2& o) const { return x == o.x && y == o.y; }
    int dot(const Vec2& o) const { return x * o.x + y * o.y; }
};
class Animal {
public:
    Animal(std::string n) : name_(n) {}
    virtual ~Animal() { std::println("bye {}", name_); }
    virtual std::string speak() const { return name_ + " makes a sound"; }
    std::string name() const { return name_; }
protected:
    std::string name_;
};
class Dog : public Animal {
public:
    Dog(std::string n, int tricks) : Animal(n), tricks_(tricks) {}
    std::string speak() const override { return name_ + " barks (" + std::to_string(tricks_) + " tricks)"; }
    static int legs() { return 4; }
private:
    int tricks_ = 0;
};
Noisy make_noisy(const std::string& n) { Noisy t(n + "'"); return t; }
std::function<int(int)> make_adder(int k) { return [k](int x) { return x + k; }; }
std::function<int()> make_counter() { int n = 0; return [n]() mutable { n += 1; return n; }; }
int apply_twice(const std::function<int(int)>& f, int v) { return f(f(v)); }
int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }
int with_default(int a, int b = 10, int c = 100) { return a + b + c; }

int main() {
    {
        Noisy a("a");
        { Noisy b("b"); Noisy c = make_noisy("c"); std::println("inner"); }
        std::println("outer");
    }
    Vec2 p; p.x = 1; p.y = 2;
    Vec2 q = p; q.x = 10;
    Vec2 s = p + q;
    std::println("{} {} {} {} {}", p.x, q.x, s.x, s.y, p.dot(q));
    std::println("{} {}", p == q, s == p + q);
    std::vector<Vec2> pts; pts.push_back(p); pts.push_back(q); pts[0].x = 99;
    std::println("{} {}", p.x, pts[0].x);
    {
        std::vector<std::shared_ptr<Animal>> zoo;
        zoo.push_back(std::make_shared<Dog>("rex", 3));
        zoo.push_back(std::make_shared<Animal>("generic"));
        for (const auto& a : zoo) std::println("{}", a->speak());
        auto first = zoo[0];
        std::println("{} {}", first->name(), Dog::legs());
        zoo.clear();
        std::println("cleared, still have {}", first->name());
    }
    auto u = std::make_unique<Noisy>("u");
    auto w = std::move(u);
    try { auto copy = w; } catch (const std::exception& e) { std::println("unique: {}", e.what()); }   // forge-check: ignore[unique-copy]
    std::println("holding {}", w->name);
    auto add5 = make_adder(5);
    auto counter = make_counter();
    counter(); counter();
    int base = 7;
    auto by_ref = [&base](int d) { base += d; return base; };
    by_ref(3);
    std::println("{} {} {} {} {}", add5(1), apply_twice(add5, 0), counter(), base, fact(10));
    std::println("{} {} {}", with_default(1), with_default(1, 2), with_default(1, 2, 3));
    std::vector<int> v{5, 3, 8, 1};
    std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; });
    int total = 0;
    std::for_each(v.begin(), v.end(), [&total](int x) { total += x; });
    std::println("{} {}", v, total);
    try { std::shared_ptr<Animal> none; std::println("{}", none->speak()); } catch (const std::exception& e) { std::println("null: {}", e.what()); }
    std::println("done");
    return 0;
}
