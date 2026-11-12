// Functions, parameter passing, default arguments, recursion, templates, auto, constexpr, lambdas and captures.
#include <algorithm>
#include <functional>
#include <print>
#include <string>
#include <vector>

int add(int a, int b = 10, int c = 100) { return a + b + c; }
void bump(int& x) { x += 1; }
void bump_all(std::vector<int>& v) { for (auto& x : v) x += 1; }
int total(const std::vector<int>& v) { int s = 0; for (int x : v) s += x; return s; }
void by_value(std::vector<int> v) { v.push_back(99); }
void fill_name(std::string& out, const std::string& base, int n) { out = base + std::to_string(n); }
void swap_ints(int& a, int& b) { int t = a; a = b; b = t; }
std::string greet(const std::string& who = "world") { return "hello " + who; }

int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
bool is_even(int n);
bool is_odd(int n) { return n == 0 ? false : is_even(n - 1); }
bool is_even(int n) { return n == 0 ? true : is_odd(n - 1); }
long long power(long long base, int exp) { return exp == 0 ? 1 : base * power(base, exp - 1); }
void hanoi(int n, char from, char to, char via, int& moves) {
    if (n == 0) return;
    hanoi(n - 1, from, via, to, moves);
    ++moves;
    hanoi(n - 1, via, to, from, moves);
}

template <typename T> T largest(T a, T b) { return a > b ? a : b; }
template <typename T> T sum_all(const std::vector<T>& v, T init) { T s = init; for (const auto& x : v) s += x; return s; }
template <typename T, typename U> auto mul(T a, U b) { return a * b; }
template <typename F> int call_n(F f, int n) { int s = 0; for (int i = 0; i < n; ++i) s += f(i); return s; }
template <typename T> void append_twice(std::vector<T>& v, const T& x) { v.push_back(x); v.push_back(x); }
template <class Container> std::size_t count_items(const Container& c) { return c.size(); }

auto make_adder(int n) { return [n](int x) { return x + n; }; }
auto square(auto x) { return x * x; }
constexpr int cube(int x) { return x * x * x; }
constexpr int kLimit = cube(3) + 1;
const int kTable[] = {2, 3, 5, 7};
static int helper_calls = 0;
int counted() { return ++helper_calls; }
int g_next_id = 100;
int next_id() { return g_next_id++; }
[[nodiscard]] inline int tagged(int x) noexcept { return x; }
auto trailing(int x) -> int { return x * 3; }
void early(int x) { if (x < 0) return; std::print("{} ", x); }

int main() {
    std::println("{} {} {}", add(1), add(1, 2), add(1, 2, 3));
    std::println("{} {}", greet(), greet("you"));

    int n = 5;
    bump(n); bump(n);
    std::vector<int> v{1, 2, 3};
    bump_all(v); by_value(v);
    std::string name;
    fill_name(name, "id", n);
    int a = 1, b = 2;
    swap_ints(a, b);
    std::println("{} {} {} {} {} {}", n, total(v), v.size(), name, a, b);
    bump(v[0]);
    std::println("{}", v[0]);

    int moves = 0;
    hanoi(5, 'a', 'c', 'b', moves);
    std::println("{} {} {} {} {}", fib(15), is_even(10), is_odd(7), power(2, 30), moves);

    std::println("{} {} {}", largest(3, 9), largest(2.5, 1.5), largest(std::string("pear"), std::string("apple")));
    std::println("{} {}", sum_all(std::vector<int>{1, 2, 3}, 0), sum_all(std::vector<double>{0.5, 0.25}, 0.0));
    std::println("{} {} {}", mul(3, 4), mul(2, 1.5), largest<int>(7, 4));
    std::println("{} {}", call_n([](int i) { return i * i; }, 4), count_items(v));
    std::vector<std::string> names;
    append_twice(names, std::string("x"));
    std::println("{} {}", names.size(), count_items(names));
    std::println("{} {} {}", square(7), square(1.5), make_adder(10)(5));
    std::println("{} {} {} {}", cube(4), kLimit, kTable[2], tagged(trailing(2)));
    counted(); counted();
    std::println("{} {} {} {}", helper_calls, next_id(), next_id(), next_id());
    early(1); early(-1); early(2);
    std::println("");

    // lambdas: captures
    int base = 10;
    auto by_copy = [base](int x) { return x + base; };
    auto by_ref = [&base](int x) { base += x; return base; };
    auto all_copy = [=]() { return base * 2; };
    auto all_ref = [&]() { base *= 2; };
    base = 20;
    std::println("{} {}", by_copy(1), all_copy());
    std::println("{}", by_ref(5));
    all_ref();
    std::println("{}", base);
    int calls = 0;
    auto counter = [calls]() mutable { return ++calls; };
    counter(); counter();
    std::println("{} {}", counter(), calls);
    auto init_capture = [k = base + 1, text = std::string("cap")]() { return text + std::to_string(k); };
    std::println("{}", init_capture());
    auto mixed = [=, &calls](int x) { calls += x; return base + x; };
    std::println("{} {}", mixed(3), calls);

    // lambdas: generic, returning lambdas, immediately invoked, recursion through std::function
    auto generic = [](auto x, auto y) { return x + y; };
    std::println("{} {} {}", generic(1, 2), generic(1.5, 2), generic(std::string("a"), std::string("b")));
    auto compose = [](auto f, auto g) { return [f, g](int x) { return f(g(x)); }; };
    auto inc_then_double = compose([](int x) { return x * 2; }, [](int x) { return x + 1; });
    std::println("{}", inc_then_double(4));
    int iife = [](int x) { return x * x; }(9);
    std::println("{}", iife);
    auto explicit_ret = [](int x) -> double { return x / 2.0; };
    std::println("{}", explicit_ret(5));
    std::function<int(int)> depth = [&depth](int k) { return k == 0 ? 0 : 1 + depth(k - 1); };
    std::println("{}", depth(12));
    auto multi = [](int x) {
        if (x > 0) return std::string("pos");
        if (x < 0) return std::string("neg");
        return std::string("zero");
    };
    std::println("{} {} {}", multi(1), multi(-1), multi(0));

    // lambdas with algorithms and state
    std::vector<int> data{4, 1, 3};
    int threshold = 2;
    auto above = std::count_if(data.begin(), data.end(), [threshold](int x) { return x > threshold; });
    std::sort(data.begin(), data.end(), [](int l, int r) { return l > r; });
    std::vector<std::function<int()>> thunks;
    for (int i = 0; i < 3; ++i) thunks.push_back([i] { return i * 10; });
    int thunk_sum = 0;
    for (auto& t : thunks) thunk_sum += t();
    std::println("{} {} {}", above, data[0], thunk_sum);
    auto acc = [sum = 0](int x) mutable { sum += x; return sum; };
    acc(1); acc(2);
    std::println("{}", acc(3));
    return 0;
}
