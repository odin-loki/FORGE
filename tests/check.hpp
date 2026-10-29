#pragma once
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace forge_test {
struct tcase { const char* name; std::function<void()> fn; };
inline std::vector<tcase>& registry() { static std::vector<tcase> r; return r; }
inline int& failures() { static int f = 0; return f; }
struct reg { reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };
}
#define FT_CAT2(a, b) a##b
#define FT_CAT(a, b) FT_CAT2(a, b)
#define TEST(name) static void name(); static forge_test::reg FT_CAT(r_, name)(#name, name); static void name()
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "  CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); ++forge_test::failures(); } } while (0)
#define CHECK_MSG(c, ...) do { if (!(c)) { std::fprintf(stderr, "  CHECK failed %s:%d: %s :: ", __FILE__, __LINE__, #c); std::fprintf(stderr, __VA_ARGS__); std::fputc('\n', stderr); ++forge_test::failures(); } } while (0)
#define CHECK_THROWS(expr) do { bool t_ = false; try { (void)(expr); } catch (...) { t_ = true; } CHECK_MSG(t_, "expected exception"); } while (0)
int main() {
    for (auto& t : forge_test::registry()) {
        int before = forge_test::failures();
        try { t.fn(); } catch (const std::exception& e) { std::fprintf(stderr, "  exception: %s\n", e.what()); ++forge_test::failures(); }
        std::printf("[%s] %s\n", forge_test::failures() == before ? " ok " : "FAIL", t.name);
    }
    std::printf("%d failure(s)\n", forge_test::failures());
    return forge_test::failures() ? 1 : 0;
}
