// Heap accounting per container kind: runaway growth inside limit {.mem} must end in a catchable OutOfMemoryError
// (the same accounting backs --heap and --mem-limit), and the same work below the cap must go through.
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

void runaway(const std::string& what, std::function<void()> grow) {
    try {
        limit {.mem = "2M"} { grow(); }
        std::println("NOT LIMITED: {}", what);
    } catch (const OutOfMemoryError& e) { std::println("limited: {}", what); }
}
struct Rec { std::string name; int n = 0; };

runaway("vector<int> push_back", [] { std::vector<int> v; while (true) v.push_back(1); });
runaway("vector<string> push_back", [] { std::vector<std::string> v; while (true) v.push_back(std::string(256, 'x')); });
runaway("vector insert", [] { std::vector<int> v; while (true) v.insert(v.begin(), 1); });
runaway("vector resize", [] { std::vector<int> v; int n = 1024; while (true) { v.resize(n); n *= 2; } });
runaway("vector(n, value)", [] { int n = 1024; while (true) { std::vector<int> v(n, 7); n *= 2; } });
runaway("vector of vectors", [] { std::vector<std::vector<int>> v; while (true) v.push_back(std::vector<int>{1, 2, 3, 4}); });
runaway("vector of structs", [] { std::vector<Rec> v; while (true) v.push_back(Rec{"r", 1}); });
runaway("deque push_front", [] { std::deque<int> d; while (true) d.push_front(1); });
runaway("list push_back", [] { std::list<int> l; while (true) l.push_back(1); });
runaway("map operator[]", [] { std::map<int, int> m; int i = 0; while (true) { m[i] = i; ++i; } });
runaway("map insert", [] { std::map<std::string, int> m; int i = 0; while (true) { m.insert({std::to_string(i), i}); ++i; } });
runaway("unordered_map emplace", [] { std::unordered_map<int, std::string> m; int i = 0; while (true) { m.emplace(i, "v"); ++i; } });
runaway("set insert", [] { std::set<int> s; int i = 0; while (true) s.insert(i++); });
runaway("unordered_set insert", [] { std::unordered_set<std::string> s; int i = 0; while (true) s.insert(std::to_string(i++)); });
runaway("string +=", [] { std::string s = "x"; while (true) s += s; });
runaway("string += char", [] { std::string s; while (true) s += 'x'; });
runaway("string append", [] { std::string s; while (true) s.append("0123456789abcdef"); });
runaway("string append(n, c)", [] { std::string s; while (true) s.append(4096, 'x'); });
runaway("string push_back", [] { std::string s; while (true) s.push_back('x'); });
runaway("string insert", [] { std::string s; while (true) s.insert(0, "0123456789abcdef"); });
runaway("string(n, c)", [] { int n = 1024; while (true) { std::string s(n, 'x'); n *= 2; } });
runaway("string concatenation", [] { std::string s; while (true) s = s + "0123456789abcdef"; });
runaway("string resize", [] { std::string s; int n = 1024; while (true) { s.resize(n); n *= 2; } });
runaway("closures", [] { std::vector<std::function<int()>> v; int i = 0; while (true) { v.push_back([i] { return i; }); ++i; } });

// below the cap nothing fires, and the memory comes back when the block ends
int total = limit {.mem = "2M"} {
    std::vector<std::string> v; std::map<int, int> m; std::string s;
    for (int i = 0; i < 200; ++i) { v.push_back(std::string(64, 'x')); m[i] = i; s.append("ab"); }
    return int(v.size() + m.size() + s.size());
};
std::println("below the cap: {}", total);

// a second error after the first is caught: the limit is still there
try {
    limit {.mem = "1M"} {
        std::vector<std::string> v;
        try { while (true) v.push_back(std::string(1024, 'x')); } catch (const OutOfMemoryError& e) { std::println("first"); }
        while (true) v.push_back(std::string(1024, 'x'));
    }
} catch (const OutOfMemoryError& e) { std::println("second"); }
