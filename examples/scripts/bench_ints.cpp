// Integer loops: a sieve-free prime count and Collatz lengths. `forge script bench_ints.cpp [limit]`.
// Used for the numbers in docs/aot.md (native AOT keeps these locals as unboxed, overflow-checked ints).
#include <iostream>
#include <string>

bool is_prime(int n) {
    if (n < 2) return false;
    for (int d = 2; d * d <= n; ++d) if (n % d == 0) return false;
    return true;
}
int collatz(int n) {
    int steps = 0;
    while (n != 1) { if (n % 2 == 0) n = n / 2; else n = 3 * n + 1; ++steps; }
    return steps;
}
int main() {
    int limit = 3000;
    if (args.size() > 0) limit = std::stoi(args[0]);
    int primes = 0, longest = 0, checksum = 0;
    for (int i = 1; i <= limit; ++i) {
        if (is_prime(i)) primes += 1;
        int c = collatz(i);
        if (c > longest) longest = c;
        checksum = (checksum * 31 + c) % 1000003;
    }
    std::println("primes={} longest={} checksum={}", primes, longest, checksum);
    return 0;
}
