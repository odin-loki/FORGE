// task::parallel_for (task/engine.hpp), the engine's data-parallel entry point, on whichever executor
// FORGE_TASK_SCHEDULER selects for this process. ctest runs this binary twice: engine_pool and engine_sched.
// The same contract must hold on both: ranges cover [0, n) exactly once, sizes respect [lo, hi], nested calls
// run on the caller, exceptions come back, and a kernel that computes each item independently gives the bits
// of one serial pass whatever the ranges were.
#include "check.hpp"
#include <forge/task/pool.hpp>
#include <forge/simd/dispatch.hpp>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace forge;

TEST(engine_reports) {
    std::printf("  engine: %s, %u thread(s)\n", task::engine_name(), task::engine_threads());
    const char* e = std::getenv("FORGE_TASK_SCHEDULER");
    CHECK((e && std::string_view(e) == "sched") == (task::engine() == task::engine_kind::sched));
    CHECK(task::engine_threads() >= 1);
    if (task::engine() == task::engine_kind::sched && task::engine_threads() > 1) CHECK(task::engine_scheduler()->size() + 1 == task::engine_threads());
}

TEST(engine_ranges_cover_once_within_bounds) {
    for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(7), std::size_t(64), std::size_t(1000), std::size_t(100003)})
        for (std::size_t lo : {std::size_t(1), std::size_t(3), std::size_t(64), std::size_t(5000)})
            for (std::size_t hi : {std::size_t(-1), lo, 4 * lo}) {
                std::vector<std::atomic<int>> hit(n);
                std::mutex mu; std::vector<std::pair<std::size_t, std::size_t>> ranges;
                task::parallel_for(n, lo, hi, [&](std::size_t b, std::size_t e) {
                    for (std::size_t i = b; i < e; ++i) hit[i].fetch_add(1);
                    std::lock_guard l(mu); ranges.push_back({b, e});
                });
                bool once = true; for (auto& h : hit) once = once && h.load() == 1;
                CHECK_MSG(once, "n=%zu lo=%zu hi=%zu: an item was missed or run twice", n, lo, hi);
                std::size_t short_ranges = 0;
                for (auto [b, e] : ranges) {
                    CHECK_MSG(b < e && e <= n, "empty or out-of-range claim [%zu, %zu)", b, e);
                    CHECK_MSG(e - b <= std::max(hi, lo), "claim of %zu items above hi=%zu", e - b, hi);
                    if (e - b < std::min(lo, n) && e != n) ++short_ranges;
                }
                CHECK_MSG(short_ranges == 0, "n=%zu lo=%zu: a claim below the grain that is not the last one", n, lo);
                if (n == 0) CHECK(ranges.empty());
                if (n && n <= lo) CHECK(ranges.size() == 1);
            }
    // the two-argument form
    std::atomic<std::size_t> sum{0};
    task::parallel_for(std::size_t(10000), std::size_t(16), [&](std::size_t b, std::size_t e) { std::size_t s = 0; for (std::size_t i = b; i < e; ++i) s += i; sum += s; });
    CHECK(sum.load() == std::size_t(10000) * 9999 / 2);
}

TEST(engine_nested_runs_on_the_caller_and_exceptions_propagate) {
    std::atomic<int> inner_ranges{0}, outer{0};
    task::parallel_for(std::size_t(64), std::size_t(1), [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) {
            ++outer;
            const auto me = std::this_thread::get_id();
            task::parallel_for(std::size_t(1000), std::size_t(1), [&](std::size_t ib, std::size_t ie) {
                CHECK(std::this_thread::get_id() == me);
                CHECK(ib == 0 && ie == 1000);
                ++inner_ranges;
            });
        }
    });
    CHECK(outer.load() == 64 && inner_ranges.load() == 64);
    bool thrown = false;
    try {
        task::parallel_for(std::size_t(5000), std::size_t(1), [&](std::size_t b, std::size_t) { if (b >= 100) throw std::runtime_error("boom"); });
    } catch (const std::runtime_error& e) { thrown = std::string_view(e.what()) == "boom"; }
    CHECK(thrown);
    // the engine still works after an exception
    std::atomic<int> c{0};
    task::parallel_for(std::size_t(1000), std::size_t(1), [&](std::size_t b, std::size_t e) { c += int(e - b); });
    CHECK(c.load() == 1000);
}

// Row-wise kernels through the engine against one serial pass over [0, n): identical bits (each row is
// computed independently of the ranges). The reduction across rows is done per row and summed in row order.
TEST(engine_rowwise_kernels_match_a_serial_pass) {
    const auto& k = simd::active();
    const std::size_t rows = 513, cols = 301;
    std::vector<float> x(rows * cols), w(cols), y1(rows * cols), y2(rows * cols), d1(rows), d2(rows);
    std::uint64_t s = 12345;
    auto next = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return float(std::int32_t(s >> 40) - (1 << 23)) / float(1 << 21); };
    for (auto& v : x) v = next();
    for (auto& v : w) v = next();
    auto body = [&](float* y, float* d) {
        return [&k, &x, &w, y, d, cols](std::size_t b, std::size_t e) {
            for (std::size_t r = b; r < e; ++r) { k.softmax_row(x.data() + r * cols, y + r * cols, cols); d[r] = k.dot(y + r * cols, w.data(), cols); }
        };
    };
    body(y1.data(), d1.data())(0, rows);
    for (int rep = 0; rep < 20; ++rep) {
        std::fill(y2.begin(), y2.end(), 0.f);
        task::parallel_for(rows, std::size_t(1 + rep % 5), body(y2.data(), d2.data()));
        CHECK(std::memcmp(y1.data(), y2.data(), y1.size() * 4) == 0 && std::memcmp(d1.data(), d2.data(), d1.size() * 4) == 0);
    }
    float a = 0, b = 0;
    for (std::size_t r = 0; r < rows; ++r) { a += d1[r]; b += d2[r]; }
    CHECK(std::memcmp(&a, &b, 4) == 0);
}
