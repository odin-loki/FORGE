// Operators: arithmetic, comparison, logical, bitwise, compound assignment, increment / decrement in every
// expression position, ternary, comma, precedence.
#include <print>
#include <string>
#include <vector>

int twice(int x) { return 2 * x; }
int add(int a, int b) { return a + b; }
int post_ret(int& x) { return x++; }
int pre_ret(int& x) { return ++x; }
bool note(const char* what, bool r) { std::print("<{}>", what); return r; }

int main() {
    int a = 17, b = 5;
    std::println("{} {} {} {} {}", a + b, a - b, a * b, a / b, a % b);
    std::println("{} {} {} {}", -a / b, -a % b, a / -b, a % -b);
    std::println("{} {} {} {} {} {}", a == b, a != b, a < b, a <= b, a > b, a >= b);
    std::println("{} {} {}", a > 0 && b > 0, a < 0 || b > 0, !(a > b));
    std::println("{} {} {} {} {} {}", a & b, a | b, a ^ b, ~a, a << 2, a >> 1);
    std::println("{} {}", 1 << 10, 0xff >> 4);
    std::println("{} {} {}", +a, -a, - -a);

    // precedence and associativity
    std::println("{} {} {}", 2 + 3 * 4, (2 + 3) * 4, 2 * 3 % 4);
    std::println("{} {} {}", 10 - 4 - 3, 100 / 10 / 5, 2 + 3 << 1);
    std::println("{} {}", 1 + 2 == 3, (5 & 3) == 1);
    std::println("{} {}", 1 | 2 ^ 3 & 4, true || false && false);
    std::println("{}", a > b ? a - b : b - a);
    std::println("{}", a < b ? 1 : a == 17 ? 2 : 3);

    // compound assignment
    int c = 10;
    c += 5; std::print("{} ", c);
    c -= 3; std::print("{} ", c);
    c *= 4; std::print("{} ", c);
    c /= 5; std::print("{} ", c);
    c %= 7; std::print("{} ", c);
    c <<= 3; std::print("{} ", c);
    c >>= 1; std::print("{} ", c);
    c |= 3; std::print("{} ", c);
    c &= 6; std::print("{} ", c);
    c ^= 5; std::println("{}", c);
    int d = (c += 2) * 2;
    std::println("{} {}", c, d);
    int e, f;
    e = f = 7;
    std::println("{} {}", e, f);
    double x = 7;
    x /= 2; x += 0.25; x *= 2;
    std::println("{}", x);
    std::string s = "ab";
    s += "cd"; s += 'e';
    std::println("{}", s);

    // pre / post increment and decrement as statements and as values
    int i = 5;
    i++; ++i; i--; --i;
    std::println("{}", i);
    int j = i++;
    int k = ++i;
    std::println("{} {} {}", i, j, k);
    j = i--;
    k = --i;
    std::println("{} {} {}", i, j, k);

    // ... as function arguments
    int n = 1;
    std::println("{}", twice(n++));
    std::println("{}", n);
    std::println("{}", twice(++n));
    std::println("{}", twice(n--));
    std::println("{}", twice(--n));
    std::println("{}", add(n++, 10));
    std::println("{}", n);

    // ... as subscripts, in conditions, in return, in initialisers
    std::vector<int> v{10, 20, 30, 40};
    int p = 0;
    std::print("{} ", v[p++]);
    std::println("{}", v[p++]);
    std::print("{} ", p);
    std::println("{}", v[--p]);
    v[p++] = 99;
    v[p]++;
    ++v[p];
    v[0]--;
    std::println("{} {} {} {}", v[0], v[1], v[2], v[3]);
    int cnt = 0;
    while (cnt++ < 3) std::print("{} ", cnt);
    std::println("| {}", cnt);
    cnt = 3;
    while (cnt-- > 0) std::print("{} ", cnt);
    std::println("| {}", cnt);
    cnt = 0;
    while (++cnt < 3) std::print("{} ", cnt);
    std::println("| {}", cnt);
    if (cnt++ == 3) std::println("post in if {}", cnt);
    if (--cnt == 3) std::println("pre in if {}", cnt);
    int r1 = post_ret(cnt), r2 = pre_ret(cnt);
    std::println("{} {} {}", r1, r2, cnt);
    int q = cnt++ + 1, w = 2 * ++cnt;
    std::println("{} {} {}", q, w, cnt);
    int tern = cnt > 0 ? cnt++ : cnt--;
    std::println("{} {}", tern, cnt);
    for (int u = 0, t = 10; u < 3; u++, t--) std::print("{}{} ", u, t);
    std::println("");
    double dd = 1.5;
    dd++; ++dd; dd--;
    std::println("{}", dd);
    std::string::size_type pos = 0;
    std::string text = "xyz";
    std::print("{}", text[pos++]);
    std::print("{}", text[pos++]);
    std::println("{}", text[pos]);

    // members and elements
    struct P { int x = 0; std::vector<int> ys{1, 2}; };
    P pt;
    pt.x++; ++pt.x; pt.x += 3; pt.ys[1] *= 5; pt.ys[0]--;
    std::println("{} {} {}", pt.x, pt.ys[0], pt.ys[1]);
    int got = pt.x++;
    std::println("{} {}", got, pt.x);

    // short circuit: the right operand is not evaluated
    bool r = note("a", false) && note("b", true);
    std::println(" {}", r);
    r = note("a", true) || note("b", true);
    std::println(" {}", r);
    r = note("a", true) && note("b", false) || note("c", true);
    std::println(" {}", r);
    int guard = 0;
    if (guard != 0 && 10 / guard > 1) std::println("unreachable");
    else std::println("guarded");

    // comma operator
    int m1 = (1, 2, 3);
    int m2 = 0;
    m2 = (m2 += 2, m2 * 10);
    std::println("{} {}", m1, m2);

    // bool arithmetic and conversions in conditions
    bool t = true;
    int bi = t + t;
    std::println("{} {} {}", bi, !t, static_cast<int>(!!5));
    if (v.size()) std::println("non-empty is true");
    if (!s.empty() && s[0] == 'a') std::println("char compare");
    int zero = 0;
    std::println("{} {}", zero ? "yes" : "no", a ? "yes" : "no");
    return 0;
}
