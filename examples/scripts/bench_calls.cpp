// Function calls: naive Fibonacci and Ackermann. `forge script bench_calls.cpp [n]`.
// Used for the numbers in docs/aot.md (a script call is the interpreter's own call path in every mode).
#include <iostream>
#include <string>

int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
int ack(int m, int n) {
    if (m == 0) return n + 1;
    if (n == 0) return ack(m - 1, 1);
    return ack(m - 1, ack(m, n - 1));
}
int main() {
    int n = 18;
    if (args.size() > 0) n = std::stoi(args[0]);
    std::println("fib({})={} ack(2,{})={}", n, fib(n), n, ack(2, n));
    return 0;
}
