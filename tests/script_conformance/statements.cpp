// Statement forms: if / else, loops, switch, break / continue, init-statements.
#include <map>
#include <print>
#include <string>
#include <vector>

int classify(int x) {
    if (x < 0) return -1;
    else if (x == 0) return 0;
    else return 1;
}

std::string day(int d) {
    switch (d) {
    case 0: return "sun";
    case 6: return "sat";
    case 1: case 2: case 3: case 4: case 5: return "weekday";
    default: return "?";
    }
}

enum class Op { add, sub, mul };

int main() {
    std::println("{} {} {}", classify(-5), classify(0), classify(9));

    int i = 0, sum = 0;
    while (i < 5) { sum += i; ++i; }
    std::println("while {}", sum);

    int n = 0;
    do { n += 3; } while (n < 10);
    std::println("do-while {}", n);
    do n = 100; while (false);
    std::println("do-while once {}", n);

    for (int a = 0, b = 10; a < b; ++a, --b) sum += a * b;
    std::println("for two vars {}", sum);

    int k = 0;
    for (;;) { if (++k > 3) break; }
    std::println("for(;;) {}", k);

    for (int x = 0; x < 6; x++) {
        if (x % 2 == 0) continue;
        if (x == 5) break;
        std::print("{} ", x);
    }
    std::println("");

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (c == r) continue;
            if (c > r + 1) break;
            std::print("({},{})", r, c);
        }
    }
    std::println("");

    std::vector<int> v{3, 1, 4, 1, 5};
    for (int x : v) std::print("{} ", x);
    std::println("");
    for (auto& x : v) x *= 2;
    for (const auto& x : v) std::print("{} ", x);
    std::println("");
    for (std::size_t j = 0; j < v.size(); ++j) std::print("{}:{} ", j, v[j]);
    std::println("");
    for (int j = static_cast<int>(v.size()) - 1; j >= 0; --j) std::print("{} ", v[j]);
    std::println("");
    for (auto x : {10, 20, 30}) std::print("{} ", x);
    std::println("");
    for (char ch : std::string("abc")) std::print("[{}]", ch);
    std::println("");

    // switch with fall-through, default in the middle, nested break
    for (int s = 0; s < 6; ++s) {
        int acc = 0;
        switch (s) {
        case 0: acc += 1;
        case 1: acc += 10; break;
        default: acc += 1000; break;
        case 2: acc += 100;
        case 3: { acc += 5; break; }
        }
        std::print("{} ", acc);
    }
    std::println("");
    for (int d = 0; d < 8; ++d) std::print("{} ", day(d));
    std::println("");

    // switch on char and on enum
    for (Op op : {Op::add, Op::sub, Op::mul}) {
        int r = 0;
        switch (op) {
        case Op::add: r = 6 + 3; break;
        case Op::sub: r = 6 - 3; break;
        case Op::mul: r = 6 * 3; break;
        }
        std::print("{} ", r);
    }
    std::println("");
    std::string word = "a-b+c";
    int plus = 0, minus = 0, other = 0;
    for (char ch : word) {
        switch (ch) {
        case '+': ++plus; break;
        case '-': ++minus; break;
        default: ++other;
        }
    }
    std::println("{} {} {}", plus, minus, other);

    // init-statements
    std::map<std::string, int> m{{"a", 1}, {"b", 2}};
    if (auto it = m.find("b"); it != m.end()) std::println("found {}", it->second);
    else std::println("missing");
    if (auto it = m.find("z"); it != m.end()) std::println("found {}", it->second);
    else std::println("missing z");
    if (int q = classify(7); q > 0) std::println("positive {}", q);
    switch (int w = 2 * 2; w) {
    case 4: std::println("switch init 4"); break;
    default: std::println("switch init other");
    }

    // nested blocks and shadowing
    int t = 1;
    {
        int t = 2;
        { int t = 3; std::print("{} ", t); }
        std::print("{} ", t);
    }
    std::println("{}", t);

    // empty statements and single-statement bodies
    int z = 0;
    for (int e = 0; e < 4; ++e) ;
    if (z == 0) ; else z = 9;
    while (z < 3) z++;
    std::println("z {}", z);
    return 0;
}
