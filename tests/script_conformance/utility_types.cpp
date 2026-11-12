// std::pair, std::tuple, std::optional, structured bindings, std::function, smart pointers, enums, aliases.
#include <functional>
#include <memory>
#include <optional>
#include <print>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

std::pair<int, int> divmod(int a, int b) { return {a / b, a % b}; }
std::tuple<std::string, int, double> record() { return {"ada", 36, 1.5}; }
std::optional<int> find_first_even(const std::vector<int>& v) {
    for (int x : v) if (x % 2 == 0) return x;
    return std::nullopt;
}
std::optional<std::string> lookup(int id) {
    if (id == 1) return "one";
    return {};
}

enum Color { red, green = 5, blue };
enum class Level : int { low = 1, mid, high };
using Table = std::vector<std::pair<std::string, int>>;
typedef std::vector<int> Ints;

struct Node {
    int value;
    std::shared_ptr<Node> next;
};

struct Shape {
    virtual ~Shape() = default;
    virtual double area() const = 0;
    virtual std::string name() const { return "shape"; }
};
struct Rect : Shape {
    double w, h;
    Rect(double w_, double h_) : w(w_), h(h_) {}
    double area() const override { return w * h; }
    std::string name() const override { return "rect"; }
};
struct Circle : Shape {
    double r;
    explicit Circle(double r_) : r(r_) {}
    double area() const override { return 3.0 * r * r; }
};

int apply_twice(const std::function<int(int)>& f, int x) { return f(f(x)); }

int main() {
    // pair
    std::pair<int, std::string> p{1, "one"};
    auto p2 = std::make_pair(2.5, 'c');
    p.first += 10; p.second += "!";
    std::println("{} {} {} {}", p.first, p.second, p2.first, p2.second);
    auto [q, r] = divmod(17, 5);
    std::println("{} {}", q, r);
    std::pair<int, int> a{1, 2}, b{1, 3};
    std::println("{} {} {}", a < b, a == b, a != b);
    std::swap(a, b);
    std::println("{} {}", a.second, b.second);

    // tuple
    std::tuple<int, std::string, double> t{7, "seven", 7.5};
    std::println("{} {} {}", std::get<0>(t), std::get<1>(t), std::get<2>(t));
    std::get<0>(t) = 8;
    auto [name, age, score] = record();
    std::println("{} {} {} {}", name, age, score, std::get<0>(t));
    auto t2 = std::make_tuple(1, 2, 3);
    std::println("{}", std::get<0>(t2) + std::get<1>(t2) + std::get<2>(t2));
    int x = 0, y = 0;
    std::tie(x, y) = divmod(23, 7);
    std::println("{} {}", x, y);

    // structured bindings over arrays, structs and by reference
    int arr[3] = {1, 2, 3};
    auto [a0, a1, a2] = arr;
    std::println("{}", a0 + a1 + a2);
    struct Pt { int x; int y; };
    Pt pt{3, 4};
    auto [px, py] = pt;
    std::println("{} {}", px + pt.x, py);
    Table table{{"a", 1}, {"b", 2}};
    for (auto& [key, val] : table) val *= 10;
    for (const auto& [key, val] : table) std::print("{}{} ", key, val);
    std::println("");

    // optional
    std::optional<int> none, some = 5;
    std::println("{} {} {}", none.has_value(), some.has_value(), *some);
    std::println("{} {}", none.value_or(-1), some.value_or(-1));
    if (some) std::println("truthy {}", some.value());
    if (!none) std::println("empty");
    none = 9; some = std::nullopt;
    std::println("{} {}", *none, some.has_value());
    std::println("{} {}", find_first_even({1, 3, 4, 6}).value(), find_first_even({1, 3}).has_value());
    if (auto found = lookup(1)) std::println("{} {}", *found, found->size());
    std::println("{}", lookup(2).value_or("none"));
    std::optional<std::string> os = "text";
    os = "other";
    std::println("{} {}", *os, some == std::nullopt);

    // enums
    Color c = blue;
    Level lv = Level::mid;
    std::println("{} {} {}", static_cast<int>(c), static_cast<int>(lv), static_cast<int>(Level::high));
    std::println("{} {} {}", c == blue, lv != Level::low, red < green);
    Level next = static_cast<Level>(static_cast<int>(lv) + 1);
    std::println("{}", next == Level::high);

    // aliases
    Ints ints{1, 2, 3};
    Table tb;
    tb.emplace_back("k", static_cast<int>(ints.size()));
    std::println("{} {}", tb[0].first, tb[0].second);

    // std::function and callables
    std::function<int(int)> inc = [](int v) { return v + 1; };
    std::function<int(int, int)> op = [](int l, int rr) { return l * rr; };
    std::println("{} {} {}", inc(1), op(6, 7), apply_twice(inc, 5));
    op = [](int l, int rr) { return l - rr; };
    std::println("{}", op(6, 7));
    std::function<int(int)> fact = [&fact](int n) { return n <= 1 ? 1 : n * fact(n - 1); };
    std::println("{}", fact(10));
    std::vector<std::function<int(int)>> pipeline{inc, [](int v) { return v * v; }, inc};
    int acc = 2;
    for (const auto& step : pipeline) acc = step(acc);
    std::println("{}", acc);
    std::function<void()> empty;
    std::println("{}", static_cast<bool>(empty));

    // shared_ptr / unique_ptr / weak_ptr
    auto n1 = std::make_shared<Node>(Node{1, nullptr});
    auto n2 = std::make_shared<Node>(Node{2, n1});
    std::shared_ptr<Node> alias = n1;
    alias->value = 100;
    std::println("{} {}", n1->value, n2->next->value);
    std::println("{} {}", n1 == alias, n2->next == n1);
    alias.reset();
    std::println("{}", alias == nullptr);
    std::weak_ptr<Node> weak = n2;
    std::println("{} {}", weak.expired(), weak.lock()->value);
    n2.reset();
    std::println("{}", weak.expired());
    auto up = std::make_unique<Rect>(2.0, 3.0);
    std::println("{} {}", up->area(), up != nullptr);
    std::unique_ptr<Rect> moved = std::move(up);
    std::println("{}", moved->name());
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Rect>(1.0, 2.0));
    shapes.push_back(std::make_unique<Circle>(1.0));
    for (const auto& s : shapes) std::print("{}:{} ", s->name(), s->area());
    std::println("");
    std::shared_ptr<Shape> base = std::make_shared<Circle>(2.0);
    std::println("{} {}", base->area(), base->name());
    if (auto rect = std::dynamic_pointer_cast<Rect>(base)) std::println("is rect");
    else std::println("not rect");
    std::shared_ptr<Node> head;
    for (int i = 3; i >= 1; --i) head = std::make_shared<Node>(Node{i, head});
    int list_sum = 0;
    for (auto cur = head; cur; cur = cur->next) list_sum += cur->value;
    std::println("{}", list_sum);
    return 0;
}
