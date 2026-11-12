// Integer and floating-point semantics, conversions, casts, <cmath>, numeric_limits.
#include <cmath>
#include <cstdint>
#include <limits>
#include <print>
#include <string>

int main() {
    // integer division and remainder follow the dividend's sign
    std::println("{} {} {} {}", 7 / 2, -7 / 2, 7 / -2, -7 / -2);
    std::println("{} {} {} {}", 7 % 2, -7 % 2, 7 % -2, -7 % -2);
    std::println("{} {}", 1000000 * 1000000LL, 1LL << 40);

    // literals
    std::println("{} {} {} {}", 0x1F, 0b1010, 017, 1'000'000);
    std::println("{} {} {}", 10u, 10L, 10LL);
    std::println("{} {} {}", 1e3, 2.5e-3, .5);
    std::println("{} {}", 'A' + 0, '\n' + 0);

    // int <-> double
    int i = 7;
    double d = i;
    std::println("{} {} {}", d / 2, i / 2, i / 2.0);
    std::println("{} {} {}", static_cast<int>(2.9), static_cast<int>(-2.9), static_cast<int>(0.5));
    std::println("{} {}", static_cast<double>(i) / 2, static_cast<long long>(1e10));
    std::println("{} {} {}", 1 + 2.5, 3 * 0.5, 10 - 0.25);
    std::println("{} {}", 5 / 2 * 2.0, 5 / 2.0 * 2);
    double mix = 1;
    mix += 1 / 2;
    mix += 1.0 / 2;
    std::println("{}", mix);
    int trunc = static_cast<int>(d * 1.5);
    std::println("{}", trunc);

    // comparisons across types
    std::println("{} {} {}", 1 == 1.0, 2 < 2.5, 3 >= 3.0);

    // bool conversions
    bool b1 = 5, b2 = 0.0, b3 = -1;
    std::println("{} {} {} {}", b1, b2, b3, true + true);
    std::println("{} {}", static_cast<int>(true), static_cast<bool>(42));

    // char arithmetic
    char c = 'a';
    char next = static_cast<char>(c + 1);
    std::println("{} {} {}", next, c - 'a', static_cast<int>(c));
    std::println("{} {}", '7' - '0', static_cast<char>('A' + 25));
    char up = static_cast<char>(c - 'a' + 'A');
    std::println("{} {}", up, c < 'b');
    int digit_sum = 0;
    for (char ch : std::string("12345")) digit_sum += ch - '0';
    std::println("{}", digit_sum);

    // fixed-width and unsigned types hold values in range
    std::int64_t big = 9007199254740993;
    std::uint8_t byte = 200;
    std::size_t sz = 12;
    unsigned int u = 40000;
    std::println("{} {} {} {}", big, static_cast<int>(byte), sz, u);
    std::println("{} {}", sz - 2, sz * 3);

    // floating point
    std::println("{} {} {}", 0.1 + 0.2, 1.0 / 3.0, 2.0 / 3.0);
    std::println("{} {} {}", 1e100, 1e-7, 123456789.0);
    std::println("{} {} {}", 100.0, 1.5, -0.25);
    std::println("{} {}", 3.0 * 2, 1e21);
    std::println("{:.2f} {:.0f} {:.3e} {:g}", 3.14159, 2.5, 12345.678, 0.0001);
    float f = 1.5f;
    std::println("{} {}", f, f * 2);
    double inf = std::numeric_limits<double>::infinity();
    double nan = std::nan("");
    std::println("{} {} {}", inf, -inf, std::isnan(nan));
    std::println("{} {} {}", std::isinf(inf), nan == nan, inf > 1e308);
    std::println("{} {}", 1.0 / inf, std::isfinite(1.0));

    // <cmath>
    std::println("{} {} {}", std::sqrt(16.0), std::pow(2.0, 10), std::pow(2, 0.5));
    std::println("{} {} {} {}", std::floor(2.7), std::ceil(2.1), std::round(2.5), std::trunc(-2.7));
    std::println("{} {} {}", std::floor(-2.5), std::ceil(-2.5), std::round(-2.5));
    std::println("{} {} {}", std::abs(-3), std::abs(-3.5), std::fabs(-2.25));
    std::println("{} {}", std::fmod(7.5, 2.0), std::fmod(-7.5, 2.0));
    std::println("{:.6f} {:.6f} {:.6f}", std::sin(1.0), std::cos(1.0), std::tan(1.0));
    std::println("{:.6f} {:.6f} {:.6f}", std::exp(1.0), std::log(10.0), std::log10(1000.0));
    std::println("{:.6f} {:.6f}", std::atan2(1.0, 1.0), std::log2(8.0));
    std::println("{} {}", std::hypot(3.0, 4.0), std::cbrt(27.0));
    std::println("{} {} {}", std::min(3, 7), std::max(3, 7), std::max(2.5, 1.5));
    std::println("{} {}", std::lround(2.5), std::lround(-2.5));

    // numeric_limits
    std::println("{} {}", std::numeric_limits<int>::max() > 2000000000, std::numeric_limits<int>::min() < 0);
    std::println("{}", std::numeric_limits<long long>::max());
    std::println("{}", std::numeric_limits<long long>::min());
    std::println("{}", std::numeric_limits<double>::max() > 1e300);
    std::println("{}", std::numeric_limits<double>::epsilon() < 1e-10);

    // integer helpers
    long long fact = 1;
    for (int k = 2; k <= 20; ++k) fact *= k;
    std::println("{}", fact);
    int gcd_a = 84, gcd_b = 36;
    while (gcd_b != 0) { int t = gcd_a % gcd_b; gcd_a = gcd_b; gcd_b = t; }
    std::println("{}", gcd_a);
    int bits = 0;
    for (int x = 0b101101; x; x >>= 1) bits += x & 1;
    std::println("{}", bits);
    return 0;
}
