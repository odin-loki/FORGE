// Forms the native code generator must agree with the interpreter on: assignment and ++ / -- used as values (their
// targets cannot live in unboxed native int slots), char arithmetic, stores that convert to the variable's declared
// type, and catch clauses that match only their type.
#include <functional>
#include <string>
#include <vector>

int twice(int x) { return 2 * x; }

int counters() {
    int i = 0, n = 5, total = 0;
    total += twice(i++);            // 0, i = 1
    total += twice(++i);            // 4, i = 2
    while (n-- > 0) total += n;     // 4 + 3 + 2 + 1 + 0
    int a, b;
    a = b = total;                  // 14
    int c = (a += 1) * 2;           // a = 15, c = 30
    std::vector<int> v{1, 2, 3};
    int p = 0;
    v[p++] = 10;
    v[p]++;
    std::println("{} {} {} {} {} {} {} {}", i, n, total, a, b, c, p, v);
    return total;
}

int digits(const std::string& s) {
    int sum = 0;
    for (char ch : s) if (ch >= '0' && ch <= '9') sum += ch - '0';
    char c = 'a';
    c++;
    char up = c - 'a' + 'A';
    std::println("{} {} {} {}", sum, c, up, 'A' + 1);
    return sum;
}

double typed() {
    double x = 7;
    x = 1;
    int i = 3;
    i = 7.9;
    i += 1.5;
    bool flag = i;
    std::println("{} {} {}", x / 2, i, flag);
    return x / 2;
}

struct AppError : std::runtime_error { using std::runtime_error::runtime_error; };

std::string classify(std::function<void()> f) {
    try { f(); return "none"; }
    catch (const PermissionError& e) { return "PermissionError"; }
    catch (const AppError& e) { return std::string("AppError ") + e.what(); }
    catch (const std::logic_error& e) { return "logic_error"; }
    catch (const std::runtime_error& e) { return "runtime_error"; }
    catch (int v) { return "int " + std::to_string(v); }
    catch (char c) { return "char"; }
    catch (const std::string& s) { return "string " + s; }
    catch (...) { return "other"; }
}

int main() {
    counters();
    digits("a1b2c3");
    typed();
    std::println("{}", classify([] {}));
    std::println("{}", classify([] { throw std::runtime_error("rt"); }));
    std::println("{}", classify([] { throw AppError("app"); }));
    std::println("{}", classify([] { throw std::out_of_range("oor"); }));
    std::println("{}", classify([] { throw PermissionError("p"); }));
    std::println("{}", classify([] { throw 7; }));
    std::println("{}", classify([] { throw 'x'; }));
    std::println("{}", classify([] { throw std::string("s"); }));
    std::println("{}", classify([] { throw 2.5; }));
    return 0;
}
