// References, write-back, std::move and control flow through the constructs the VM hands to the tree-walker.
#include <iostream>
#include <map>
#include <string>
#include <vector>

void bump(int& x) { ++x; }
void rename(std::string& s, const std::string& suffix) { s = s + suffix; }
void swap_ints(int& a, int& b) { int t = a; a = b; b = t; }
struct Counter { int n = 0; std::vector<int> seen; };

int find_first_even(const std::vector<int>& v) {
    for (auto& x : v) { if (x % 2 == 0) return x; }
    return -1;
}
int scale_until(std::vector<int>& v, int stop) {
    int touched = 0;
    for (auto& x : v) {
        if (x == stop) break;
        if (x < 0) continue;
        x = x * 10;
        ++touched;
    }
    return touched;
}
int sum_with_try(const std::vector<std::string>& items) {
    int total = 0;
    for (const auto& s : items) {
        try {
            if (s == "skip") continue;
            if (s == "stop") break;
            total += std::stoi(s);
        } catch (const std::exception& e) {
            total += 1000;
        }
    }
    return total;
}
int nested_return(int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            try { if (i * j == 6) return i * 10 + j; }
            catch (...) { return -1; }
        }
    }
    return 0;
}
std::string classify(int n) {
    switch (n % 4) {
    case 0: return "zero";
    case 1:
    case 2: { std::string s = "low"; if (n > 10) s += "+"; return s; }
    default: break;
    }
    return "high";
}
int do_while_sum(int n) { int s = 0; int i = 0; do { s += i; ++i; } while (i < n); return s; }
int rethrow_depth(int d) {
    try {
        if (d == 0) throw std::runtime_error("bottom");
        return rethrow_depth(d - 1);
    } catch (const std::runtime_error& e) {
        if (d < 2) throw;
        return d;
    }
}

int main() {
    int a = 1, b = 2;
    bump(a); bump(a); swap_ints(a, b);
    std::println("{} {}", a, b);
    std::string name = "file"; rename(name, ".txt");
    std::println("{}", name);
    Counter c; bump(c.n); bump(c.n);
    std::vector<int> v{1, 3, 4, -2, 7, 8};
    bump(v[0]);
    std::println("{} {} {}", c.n, v[0], find_first_even(v));
    std::println("{} {}", scale_until(v, 7), v);
    for (auto& x : v) x += 1;
    std::println("{}", v);
    std::map<std::string, int> ages{{"ann", 30}, {"bob", 41}};
    for (auto& [k, age] : ages) age += 1;
    for (const auto& [k, age] : ages) std::println("{}={}", k, age);
    std::vector<std::string> words{"x", "yy"};
    for (auto& w : words) w += "!";
    std::println("{}", words);
    std::println("{}", sum_with_try({"1", "2", "skip", "oops", "4", "stop", "100"}));
    std::println("{} {}", nested_return(5), nested_return(2));
    std::println("{} {} {} {} {}", classify(4), classify(1), classify(14), classify(3), do_while_sum(5));
    std::println("{}", rethrow_depth(4));
    std::vector<int> big{1, 2, 3};
    std::vector<int> moved = std::move(big);
    std::println("{}", moved.size());
    try { std::println("{}", big.size()); } catch (const std::exception& e) { std::println("moved: {}", e.what()); }   // forge-check: ignore[use-after-move]
    std::vector<int> grid{0, 0, 0};
    for (int i = 0; i < 3; ++i) { grid[i] += i * i; grid[i] *= 2; }
    std::println("{}", grid);
    try { grid[7] += 1; } catch (const std::exception& e) { std::println("index: {}", e.what()); }
    try { std::vector<int> it{1, 2, 3}; for (auto x : it) { if (x == 2) it.push_back(9); } } catch (const std::exception& e) { std::println("borrow: {}", e.what()); }   // forge-check: ignore[iter-mutation]
    try { throw std::string("plain"); } catch (const std::exception& e) { std::println("BUG: std::exception caught a string"); } catch (const std::string& s) { std::println("string thrown: {}", s); }
    try { throw std::runtime_error("rt"); } catch (const PermissionError& e) { std::println("BUG: a bare class name caught another class"); } catch (const std::runtime_error& e) { std::println("typed catch: {}", e.what()); }
    return 0;
}
