// std::map, std::unordered_map, std::multimap, std::set, std::unordered_set.
#include <algorithm>
#include <map>
#include <print>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

int main() {
    // std::map: ordered by key
    std::map<std::string, int> m{{"pear", 3}, {"apple", 1}};
    m["fig"] = 7;
    m["apple"] += 10;
    m.insert({"kiwi", 4});
    m.insert(std::make_pair("lime", 5));
    m.emplace("date", 2);
    auto [where, added] = m.insert({"fig", 100});
    std::println("{} {} {}", added, where->first, where->second);
    for (const auto& [k, v] : m) std::print("{}={} ", k, v);
    std::println("");
    std::println("{} {} {}", m.size(), m.count("fig"), m.count("nope"));
    std::println("{} {}", m.contains("pear"), m.contains("plum"));
    std::println("{} {}", m.at("pear"), m["new-key"]);
    std::println("{}", m.size());
    if (auto it = m.find("kiwi"); it != m.end()) { it->second *= 2; std::println("{} {}", it->first, it->second); }
    std::println("{}", m.find("zzz") == m.end());
    std::println("{} {}", m.erase("new-key"), m.erase("absent"));
    m.erase(m.find("date"));
    std::println("{}", m.begin()->first);
    for (auto it = m.begin(); it != m.end(); ++it) std::print("{} ", it->first);
    std::println("");
    for (auto& [k, v] : m) v = 0;
    std::println("{} {}", m["apple"], m["pear"]);
    auto lb = m.lower_bound("g"), ub = m.upper_bound("kiwi");
    std::println("{} {}", lb->first, ub->first);
    m.insert_or_assign("apple", 42);
    m.try_emplace("apple", 1);
    m.try_emplace("zebra", 26);
    std::println("{} {}", m["apple"], m["zebra"]);
    std::map<std::string, int> m2 = m;
    m2.clear();
    std::println("{} {} {}", m.empty(), m2.empty(), m2.size());

    // integer keys, value containers, nested maps
    std::map<int, std::vector<std::string>> groups;
    for (std::string w : {"a", "bb", "cc", "d", "eee"}) groups[static_cast<int>(w.size())].push_back(w);
    for (const auto& [len, words] : groups) {
        std::print("{}:", len);
        for (const auto& w : words) std::print(" {}", w);
        std::print("; ");
    }
    std::println("");
    std::map<int, int> desc;
    for (int i : {5, 1, 9, 3}) desc[i] = i * i;
    for (auto it = desc.begin(); it != desc.end(); ++it) std::print("{}:{} ", it->first, it->second);
    std::println("");
    std::map<std::string, std::map<std::string, int>> nested;
    nested["x"]["y"] = 1;
    nested["x"]["z"] += 2;
    std::println("{} {} {}", nested["x"].size(), nested["x"]["z"], nested.count("q"));
    std::map<char, int> freq;
    for (char ch : std::string("mississippi")) ++freq[ch];
    for (const auto& [ch, n] : freq) std::print("{}{} ", ch, n);
    std::println("");

    // std::unordered_map: no order guarantee, so print sorted
    std::unordered_map<std::string, int> um{{"one", 1}, {"two", 2}};
    um["three"] = 3;
    um.emplace("four", 4);
    um.erase("one");
    std::vector<std::string> keys;
    for (const auto& [k, v] : um) keys.push_back(k + "=" + std::to_string(v));
    std::sort(keys.begin(), keys.end());
    for (const auto& k : keys) std::print("{} ", k);
    std::println("");
    std::println("{} {} {} {}", um.size(), um.count("two"), um.contains("one"), um.at("four"));
    std::println("{} {}", um.find("three") != um.end(), um.find("nine") == um.end());
    int total = 0;
    for (auto& kv : um) total += kv.second;
    std::println("{}", total);

    // std::set
    std::set<int> s{5, 1, 3};
    s.insert(4); s.insert(1); s.emplace(2);
    auto [sit, fresh] = s.insert(9);
    std::println("{} {}", *sit, fresh);
    for (int x : s) std::print("{} ", x);
    std::println("");
    std::println("{} {} {} {}", s.size(), s.count(3), s.contains(7), s.find(4) != s.end());
    s.erase(3);
    std::println("{} {}", *s.begin(), s.size());
    std::println("{} {}", *s.lower_bound(3), *s.upper_bound(4));
    std::set<std::string> words{"pear", "apple", "pear"};
    for (const auto& w : words) std::print("{} ", w);
    std::println("{}", words.size());
    std::vector<int> dups{3, 1, 3, 2, 1};
    std::set<int> uniq;
    uniq.insert(dups.begin(), dups.end());
    std::vector<int> back(uniq.begin(), uniq.end());
    std::println("{} {} {}", uniq.size(), back.front(), back.back());
    std::set<int> other{2, 3, 4};
    int common = 0;
    for (int x : uniq) if (other.count(x)) ++common;
    std::println("{} {}", common, uniq == std::set<int>{1, 2, 3});

    // std::unordered_set
    std::unordered_set<std::string> seen;
    int repeats = 0;
    for (std::string w : {"a", "b", "a", "c", "b", "a"}) {
        if (!seen.insert(w).second) ++repeats;
    }
    std::println("{} {} {} {}", seen.size(), repeats, seen.count("a"), seen.contains("z"));
    seen.erase("a");
    std::println("{} {}", seen.size(), seen.empty());
    return 0;
}
