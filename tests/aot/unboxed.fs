// Integer locals that native AOT keeps unboxed: every checked operation must still raise what the interpreter raises.
#include <iostream>
#include <vector>
#include <string>

int sum_to(int n) { int s = 0; for (int i = 1; i <= n; ++i) s += i; return s; }
int collatz(int n) { int steps = 0; while (n != 1) { if (n % 2 == 0) n = n / 2; else n = 3 * n + 1; ++steps; } return steps; }
int mix(int a, int b) {
    int x = a * b - (a << 3) + (b >> 1);
    int y = (x ^ a) | (b & 7);
    y -= a; y *= 2; y /= 3; y %= 1000;
    return -y + ~a;
}
int gcd(int a, int b) { while (b != 0) { int t = a % b; a = b; b = t; } return a; }
int overflow_add(int n) { int big = 9223372036854775807; return big + n; }
int overflow_mul(int n) { int big = 4611686018427387904; int r = big * n; return r; }
int div_by(int a, int b) { int q = a / b; return q; }
int mod_by(int a, int b) { int m = a % b; return m; }
int shift_by(int a, int b) { int r = a << b; return r; }
int negate(int a) { int r = -a; return r; }
int from_double(double d) { int n = static_cast<int>(d); n += 1; return n; }
int from_size(const std::vector<int>& v) { int n = v.size(); int total = 0; for (int i = 0; i < n; ++i) total += v[i] * (i + 1); return total; }
bool is_prime(int n) { if (n < 2) return false; for (int d = 2; d * d <= n; ++d) if (n % d == 0) return false; return true; }
int count_primes(int limit) { int c = 0; for (int i = 0; i < limit; ++i) if (is_prime(i)) c += 1; return c; }
int ternary_chain(int a) { int r = a > 10 ? a - 10 : a < 0 ? -a : a * 2; return r; }
int logic(int a, int b) { int n = 0; if (a < b && b < 100 || a == 7) n = 1; if (!(a >= b)) n += 2; return n; }
double mixed(int a, double f) { int k = a * 2; double r = k * f + a / 4; return r; }
std::string describe(int n) { int twice = n * 2; return "n=" + std::to_string(n) + " twice=" + std::to_string(twice); }
int reassigned(int n) { auto k = 0; k = n; k = k + n * n; auto z = k; z -= 1; return z; }
int shadow(int n) { int total = 0; { int n2 = n + 1; total += n2; } { int n2 = n * 3; total += n2; } return total; }

template <class F> void attempt(const std::string& what, F f) {
    try { std::println("{} -> {}", what, f()); }
    catch (const std::exception& e) { std::println("{} raised {}", what, e.what()); }
}

int main() {
    std::println("{} {} {} {}", sum_to(100), collatz(27), mix(12345, 678), gcd(1071, 462));
    std::println("{} {} {}", mix(-5, 3), mix(0, 0), mix(-1000000, -999));
    attempt("add 0", [] { return overflow_add(0); });
    attempt("add 1", [] { return overflow_add(1); });
    attempt("mul 1", [] { return overflow_mul(1); });
    attempt("mul 2", [] { return overflow_mul(2); });
    attempt("mul -2", [] { return overflow_mul(-2); });
    attempt("mul -3", [] { return overflow_mul(-3); });
    attempt("7/2", [] { return div_by(7, 2); });
    attempt("-7/2", [] { return div_by(-7, 2); });
    attempt("7/0", [] { return div_by(7, 0); });
    attempt("-7%3", [] { return mod_by(-7, 3); });
    attempt("7%-3", [] { return mod_by(7, -3); });
    attempt("7%0", [] { return mod_by(7, 0); });
    attempt("5%-1", [] { return mod_by(5, -1); });
    attempt("1<<62", [] { return shift_by(1, 62); });
    attempt("1<<63", [] { return shift_by(1, 63); });
    attempt("1<<64", [] { return shift_by(1, 64); });
    attempt("1<<-1", [] { return shift_by(1, -1); });
    attempt("neg", [] { return negate(42); });
    attempt("neg min", [] { int m = -9223372036854775807; m -= 1; return negate(m); });
    attempt("min/-1", [] { int m = -9223372036854775807; m -= 1; return div_by(m, -1); });
    attempt("from_double", [] { return from_double(41.9); });
    attempt("from_double big", [] { return from_double(1e30); });
    std::vector<int> v{3, 1, 4, 1, 5, 9, 2, 6};
    std::println("{} {} {}", from_size(v), count_primes(200), ternary_chain(15) + ternary_chain(-4) + ternary_chain(3));
    std::println("{} {} {} {}", logic(1, 2), logic(7, 3), logic(200, 100), logic(5, 5));
    std::println("{} {} {}", mixed(7, 0.5), describe(21), reassigned(9));
    std::println("{}", shadow(4));
    return sum_to(10) - 55;
}
