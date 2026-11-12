// std::format / std::print / std::println format specifications, and std::cout with manipulators.
#include <format>
#include <iomanip>
#include <iostream>
#include <print>
#include <string>

int main() {
    // integers
    std::println("[{}] [{:d}] [{:5}] [{:<5}] [{:>5}] [{:^5}]", 42, 42, 42, 42, 42, 42);
    std::println("[{:05}] [{:+}] [{:+}] [{: }] [{:*^7}] [{:#>6}]", 42, 42, -42, 42, 42, 42);
    std::println("[{:x}] [{:X}] [{:#x}] [{:o}] [{:#o}] [{:b}] [{:#b}]", 255, 255, 255, 8, 8, 5, 5);
    std::println("[{:08.3f}] [{:#010x}] [{:c}]", 3.14159, 255, 65);
    std::println("[{}] [{}] [{}]", -0, 1234567890123LL, 0);

    // floating point
    std::println("[{}] [{}] [{}] [{}]", 1.0, 0.5, 100.0, 1.0 / 3);
    std::println("[{:.2f}] [{:.0f}] [{:10.3f}] [{:<10.1f}] [{:+.1f}]", 3.14159, 2.6, 2.71828, 9.99, 1.25);
    std::println("[{:e}] [{:.2e}] [{:E}] [{:g}] [{:.3g}]", 12345.678, 12345.678, 0.00012, 12345.678, 12345.678);
    std::println("[{:f}] [{:.1f}] [{:6.2f}] [{:06.2f}]", 1.5, -0.05, -1.0, -1.0);
    std::println("[{}] [{}] [{}]", 1e10, 1e-5, 123456789.125);

    // strings, chars, bools
    std::println("[{}] [{:10}] [{:>10}] [{:^10}] [{:-^10}] [{:.3}]", "abc", "abc", "abc", "abc", "abc", "abcdef");
    std::println("[{}] [{:3}] [{:>3}] [{}]", 'x', 'x', 'x', std::string("str"));
    std::println("[{}] [{}] [{:>6}] [{:d}]", true, false, true, true);
    std::println("{{}} {{{}}} }}{{", 7);

    // argument indices and dynamic width / precision
    std::println("{0} {1} {0} {1}", "a", "b");
    std::println("{1}-{0}", 1, 2);
    std::println("[{:{}}] [{:.{}f}] [{:>{}.{}f}]", 7, 4, 3.14159, 2, 2.71828, 8, 3);
    std::println("[{0:{1}}]", 5, 3);

    // std::format returns a string
    std::string s = std::format("{}-{:02}-{:02}", 2024, 3, 9);
    std::println("{} {}", s, s.size());
    std::string row = std::format("{:<8}|{:>6.2f}|{:^5}|", "name", 12.5, true);
    std::println("{}", row);
    std::println("{}", std::format("{}", std::format("{:>4}", 1)));

    // print without newline, println with no arguments
    std::print("a");
    std::print("{}{}", 'b', "c");
    std::println("");
    std::println("no placeholders");
    std::println("");

    // iostreams
    std::cout << "int " << 42 << " double " << 2.5 << " char " << 'c' << " bool " << true << "\n";
    std::cout << std::boolalpha << true << ' ' << false << std::noboolalpha << ' ' << true << std::endl;
    std::cout << std::fixed << std::setprecision(3) << 3.14159 << ' ' << 2.0 << '\n';
    std::cout << std::setprecision(1) << 0.25 << ' ' << 100.0 << '\n';
    std::cout << std::setw(6) << 42 << '|' << std::setw(6) << "ab" << '|' << 7 << '\n';
    std::cout << std::left << std::setw(6) << 42 << '|' << std::right << std::setw(6) << 42 << '|' << '\n';
    std::cout << std::setfill('0') << std::setw(5) << 42 << std::setfill(' ') << '\n';
    std::cout << std::hex << 255 << ' ' << std::dec << 255 << '\n';
    std::string who = "world";
    std::cout << "hello, " << who << '!' << std::endl;
    int a = 3, b = 4;
    std::cout << a << " + " << b << " = " << a + b << "\n";
    std::cout << (a < b ? "less" : "more") << ' ' << (a << 1) << '\n';
    std::cerr << "";
    return 0;
}
