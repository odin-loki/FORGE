// <algorithm> and <numeric>, iterator and std::ranges forms.
#include <algorithm>
#include <functional>
#include <numeric>
#include <print>
#include <string>
#include <vector>

void show(const std::string& name, const std::vector<int>& c) {
    std::print("{}:", name);
    for (int x : c) std::print(" {}", x);
    std::println("");
}

struct Person { std::string name; int age; };

int main() {
    std::vector<int> v{5, 3, 8, 1, 9, 2, 8};

    // sorting
    auto s = v;
    std::sort(s.begin(), s.end()); show("sort", s);
    std::sort(s.begin(), s.end(), std::greater<int>()); show("greater", s);
    std::sort(s.begin(), s.end(), [](int a, int b) { return a % 3 < b % 3 || (a % 3 == b % 3 && a < b); }); show("custom", s);
    std::sort(s.begin() + 1, s.end() - 1); show("sub-range", s);
    std::vector<Person> people{{"cy", 30}, {"al", 25}, {"bo", 30}, {"di", 25}};
    std::stable_sort(people.begin(), people.end(), [](const Person& a, const Person& b) { return a.age < b.age; });
    for (const auto& p : people) std::print("{}{} ", p.name, p.age);
    std::println("");
    std::sort(people.begin(), people.end(), [](const Person& a, const Person& b) { return a.name > b.name; });
    std::println("{} {}", people.front().name, people.back().name);
    std::vector<std::string> words{"pear", "Apple", "fig"};
    std::sort(words.begin(), words.end());
    std::println("{} {} {}", words[0], words[1], words[2]);
    std::println("{} {}", std::is_sorted(s.begin(), s.end()), std::is_sorted(words.begin(), words.end()));

    // searching and counting
    std::println("{} {}", std::find(v.begin(), v.end(), 8) - v.begin(), std::find(v.begin(), v.end(), 7) == v.end());
    auto big = std::find_if(v.begin(), v.end(), [](int x) { return x > 7; });
    auto odd = std::find_if_not(v.begin(), v.end(), [](int x) { return x % 2 == 1; });
    std::println("{} {}", *big, *odd);
    std::println("{} {}", std::count(v.begin(), v.end(), 8), std::count_if(v.begin(), v.end(), [](int x) { return x < 5; }));
    std::println("{} {} {}", std::any_of(v.begin(), v.end(), [](int x) { return x == 9; }),
                 std::all_of(v.begin(), v.end(), [](int x) { return x > 0; }), std::none_of(v.begin(), v.end(), [](int x) { return x > 9; }));
    std::println("{} {}", *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()));
    std::println("{}", std::max_element(v.begin(), v.end()) - v.begin());
    auto oldest = std::max_element(people.begin(), people.end(), [](const Person& a, const Person& b) { return a.age < b.age; });
    std::println("{}", oldest->age);
    std::vector<int> sorted{1, 3, 3, 5, 7};
    std::println("{} {}", std::binary_search(sorted.begin(), sorted.end(), 5), std::binary_search(sorted.begin(), sorted.end(), 4));
    std::println("{} {}", std::lower_bound(sorted.begin(), sorted.end(), 3) - sorted.begin(), std::upper_bound(sorted.begin(), sorted.end(), 3) - sorted.begin());
    std::println("{}", std::equal(sorted.begin(), sorted.end(), sorted.begin()));

    // modifying
    auto m = v;
    std::reverse(m.begin(), m.end()); show("reverse", m);
    std::fill(m.begin(), m.begin() + 2, 0); show("fill", m);
    std::iota(m.begin(), m.end(), 10); show("iota", m);
    std::transform(m.begin(), m.end(), m.begin(), [](int x) { return x * 2; }); show("transform", m);
    std::vector<int> sums(m.size());
    std::transform(m.begin(), m.end(), v.begin(), sums.begin(), [](int a, int b) { return a + b; }); show("transform2", sums);
    std::vector<std::string> strs;
    std::transform(v.begin(), v.end(), std::back_inserter(strs), [](int x) { return std::to_string(x) + "!"; });
    std::println("{} {}", strs.size(), strs.back());
    std::for_each(m.begin(), m.end(), [](int& x) { x -= 20; }); show("for_each", m);
    int seen = 0;
    std::for_each(v.begin(), v.end(), [&seen](int x) { seen += x; });
    std::println("{}", seen);
    std::replace(m.begin(), m.end(), 0, 99); show("replace", m);
    std::vector<int> out;
    std::copy(v.begin(), v.end(), std::back_inserter(out)); show("copy", out);
    std::vector<int> evens;
    std::copy_if(v.begin(), v.end(), std::back_inserter(evens), [](int x) { return x % 2 == 0; }); show("copy_if", evens);
    auto u = v;
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end()); show("unique", u);
    auto rm = v;
    rm.erase(std::remove(rm.begin(), rm.end(), 8), rm.end()); show("remove", rm);
    rm.erase(std::remove_if(rm.begin(), rm.end(), [](int x) { return x > 4; }), rm.end()); show("remove_if", rm);
    std::erase(rm, 1); show("std::erase", rm);
    auto er = v;
    std::erase_if(er, [](int x) { return x % 2 == 1; }); show("erase_if", er);
    auto rot = sorted;
    std::rotate(rot.begin(), rot.begin() + 2, rot.end()); show("rotate", rot);
    std::swap(rot[0], rot[1]); show("swap", rot);
    auto part = v;
    auto mid = std::partition(part.begin(), part.end(), [](int x) { return x < 5; });
    std::println("{}", mid - part.begin());
    std::vector<int> perm{1, 2, 3};
    int perms = 0;
    do { ++perms; } while (std::next_permutation(perm.begin(), perm.end()));
    std::println("{}", perms);

    // min / max / clamp
    std::println("{} {} {} {}", std::min(2, 9), std::max(2, 9), std::min({4, 2, 6}), std::max({4, 2, 6}));
    std::println("{} {} {}", std::clamp(15, 0, 10), std::clamp(-5, 0, 10), std::clamp(5, 0, 10));
    std::println("{} {}", std::min(std::string("b"), std::string("a")), std::max(1.5, 2.5));
    auto [lo, hi] = std::minmax_element(v.begin(), v.end());
    std::println("{} {}", *lo, *hi);

    // numeric
    std::println("{} {}", std::accumulate(v.begin(), v.end(), 0), std::accumulate(v.begin(), v.end(), 100));
    std::println("{}", std::accumulate(v.begin(), v.end(), 1LL, [](long long a, int b) { return a * b; }));
    std::println("{}", std::accumulate(words.begin(), words.end(), std::string("")));
    std::println("{} {}", std::reduce(v.begin(), v.end()), std::accumulate(v.begin(), v.end(), 0.5));
    std::vector<int> ps(v.size());
    std::partial_sum(v.begin(), v.end(), ps.begin()); show("partial_sum", ps);
    std::println("{}", std::inner_product(sorted.begin(), sorted.end(), sorted.begin(), 0));
    std::println("{} {}", std::gcd(84, 36), std::lcm(4, 6));

    // ranges
    auto r = v;
    std::ranges::sort(r); show("ranges::sort", r);
    std::ranges::reverse(r); show("ranges::reverse", r);
    std::ranges::sort(r, std::greater<>()); show("ranges greater", r);
    std::println("{} {}", std::ranges::find(r, 9) - r.begin(), std::ranges::count(r, 8));
    std::println("{} {}", std::ranges::any_of(r, [](int x) { return x > 8; }), std::ranges::all_of(r, [](int x) { return x > 1; }));
    std::println("{} {}", *std::ranges::min_element(r), *std::ranges::max_element(r));
    std::println("{} {}", std::ranges::count_if(r, [](int x) { return x % 2 == 0; }), std::ranges::find_if(r, [](int x) { return x < 3; }) - r.begin());
    std::ranges::fill(r, 4); show("ranges::fill", r);
    std::ranges::for_each(r, [](int& x) { x += 1; }); show("ranges::for_each", r);
    std::println("{} {}", std::ranges::max(v), std::ranges::min(v));
    return 0;
}
