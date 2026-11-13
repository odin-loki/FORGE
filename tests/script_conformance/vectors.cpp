// std::vector, std::array, std::deque, std::list, std::span: construction, element access, modifiers, iterators.
#include <algorithm>
#include <array>
#include <deque>
#include <list>
#include <print>
#include <span>
#include <string>
#include <vector>

template <typename C>
void show(const std::string& name, const C& c) {
    std::print("{}:", name);
    for (const auto& x : c) std::print(" {}", x);
    std::println("");
}

int sum_span(std::span<const int> xs) {
    int s = 0;
    for (int x : xs) s += x;
    return s;
}

int main() {
    // construction
    std::vector<int> a, b(3), c(3, 7), d{1, 2, 3}, e = {4, 5}, f(d), g(d.begin() + 1, d.end());
    show("a", a); show("b", b); show("c", c); show("d", d); show("e", e); show("f", f); show("g", g);
    std::vector<std::string> names{"x", "yy"};
    std::vector<std::vector<int>> grid(2, std::vector<int>(3, 1));
    grid[1][2] = 9;
    std::println("{} {} {} {}", names.size(), grid.size(), grid[0].size(), grid[1][2]);

    // element access
    std::vector<int> v{10, 20, 30, 40, 50};
    std::println("{} {} {} {} {}", v[0], v.at(1), v.front(), v.back(), v.size());
    v[0] = 11; v.at(1) = 21; v.front() += 1; v.back() -= 1;
    show("v", v);
    std::println("{} {}", v.empty(), a.empty());

    // modifiers
    v.push_back(60); v.emplace_back(70); v.pop_back();
    show("push/pop", v);
    v.insert(v.begin(), 1); v.insert(v.begin() + 2, 2); v.insert(v.end(), 3);
    show("insert", v);
    v.insert(v.begin() + 1, 2, 0);
    show("insert n", v);
    v.insert(v.end(), e.begin(), e.end());
    show("insert range", v);
    v.erase(v.begin()); v.erase(v.begin() + 1, v.begin() + 3);
    show("erase", v);
    auto after = v.erase(v.begin() + 1);
    std::println("erase returns {}", *after);
    v.resize(3); show("resize down", v);
    v.resize(5); show("resize up", v);
    v.resize(7, -1); show("resize fill", v);
    v.assign(3, 4); show("assign n", v);
    v.assign({9, 8, 7}); show("assign list", v);
    v.clear(); std::println("clear {} {}", v.size(), v.empty());
    v.reserve(50); std::println("reserve {}", v.size());
    std::vector<int> p{1, 2}, q{3, 4, 5};
    p.swap(q); show("swap p", p); show("swap q", q);
    std::swap(p, q); show("std::swap p", p);

    // copies are independent, comparison is element-wise
    std::vector<int> orig{1, 2, 3};
    std::vector<int> copy = orig;
    copy[0] = 100;
    std::println("{} {} {}", orig[0], copy[0], orig == copy);
    std::println("{} {} {}", orig == std::vector<int>{1, 2, 3}, orig < copy, orig != copy);
    std::vector<std::vector<int>> nested{{1}, {2, 3}};
    auto nested_copy = nested;
    nested_copy[1].push_back(4);
    std::println("{} {}", nested[1].size(), nested_copy[1].size());

    // iterators
    std::vector<int> it_v{5, 6, 7, 8};
    for (auto it = it_v.begin(); it != it_v.end(); ++it) *it += 1;
    for (auto it = it_v.cbegin(); it != it_v.cend(); ++it) std::print("{} ", *it);
    std::println("");
    auto pos = std::find(it_v.begin(), it_v.end(), 8);
    std::println("{} {} {}", pos - it_v.begin(), *(pos - 1), *(it_v.end() - 1));
    std::println("{} {}", it_v.end() - it_v.begin(), std::distance(it_v.begin(), pos));
    auto second = std::next(it_v.begin());
    auto last = std::prev(it_v.end());
    std::println("{} {}", *second, *last);
    for (auto it = it_v.begin(); it != it_v.end();) {
        if (*it % 2 == 0) it = it_v.erase(it);
        else ++it;
    }
    show("erase in loop", it_v);
    std::vector<bool> flags(3, false);
    flags[1] = true;
    std::println("{} {} {}", bool(flags[0]), bool(flags[1]), flags.size());   // bool(): libstdc++ 14 cannot format vector<bool>::reference
    std::vector<std::pair<int, std::string>> pairs;
    pairs.emplace_back(1, "one");
    pairs.push_back({2, "two"});
    for (const auto& [num, word] : pairs) std::print("{}={} ", num, word);
    std::println("");

    // std::array
    std::array<int, 4> arr{4, 3, 2, 1};
    std::println("{} {} {} {}", arr.size(), arr[0], arr.front(), arr.back());
    std::sort(arr.begin(), arr.end());
    show("array", arr);
    arr.fill(6); show("fill", arr);
    std::array<std::string, 2> sa{"p", "q"};
    std::println("{} {}", sa.at(1), sa.empty());

    // std::deque
    std::deque<int> dq{2, 3};
    dq.push_front(1); dq.push_back(4); dq.emplace_front(0); dq.emplace_back(5);
    show("deque", dq);
    std::println("{} {} {} {}", dq.front(), dq.back(), dq[2], dq.at(3));
    dq.pop_front(); dq.pop_back();
    show("deque popped", dq);
    dq.insert(dq.begin() + 1, 99); dq.erase(dq.begin());
    show("deque edited", dq);
    std::println("{} {}", dq.size(), dq.empty());

    // std::list
    std::list<int> li{3, 1, 2};
    li.push_front(0); li.push_back(4);
    show("list", li);
    li.sort(); show("list sorted", li);
    li.reverse(); show("list reversed", li);
    std::erase(li, 2); show("list removed", li);
    li.pop_front(); li.pop_back();
    std::println("{} {} {}", li.front(), li.back(), li.size());

    // std::span
    std::vector<int> sv{1, 2, 3, 4, 5};
    std::println("{}", sum_span(sv));
    std::span<int> view(sv);
    std::println("{} {} {} {}", sv[0], view.size(), view.front(), view.back());
    return 0;
}
