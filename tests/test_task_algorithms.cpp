// task/algorithms.hpp: every parallel algorithm gives exactly the serial result for 1, 2, 3, 8 and 16 threads.
#include "check.hpp"
#include <forge/task/algorithms.hpp>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
using namespace forge;

static const unsigned kThreads[] = {1, 2, 3, 8, 16};

struct rec { std::uint32_t key; std::uint32_t seq; bool operator==(const rec&) const = default; };

static std::vector<rec> make_recs(std::size_t n, std::uint32_t range, std::uint64_t seed) {
    std::mt19937_64 g(seed); std::vector<rec> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = {std::uint32_t(g() % range), std::uint32_t(i)};   // many equal keys
    return v;
}

TEST(sort_equals_stable_sort) {
    for (std::size_t n : {0u, 1u, 7u, 5000u, 100003u}) {
        auto base = make_recs(n, 97, 11 + n);
        auto cmp = [](const rec& a, const rec& b) { return a.key < b.key; };
        auto want = base; std::stable_sort(want.begin(), want.end(), cmp);
        for (unsigned th : kThreads) {
            task::pool p(th);
            auto got = base; task::parallel_sort(&p, got.begin(), got.end(), cmp);
            CHECK_MSG(got == want, "n=%zu threads=%u", std::size_t(n), th);
        }
        auto got = base; task::parallel_sort(nullptr, got.begin(), got.end(), cmp);
        CHECK(got == want);
    }
}

TEST(merge_equals_std_merge) {
    std::mt19937_64 g(5);
    for (int round = 0; round < 40; ++round) {
        const std::size_t n1 = g() % 40000, n2 = g() % 40000;
        auto a = make_recs(n1, 50, g()), b = make_recs(n2, 50, g());
        auto cmp = [](const rec& x, const rec& y) { return x.key < y.key; };
        std::stable_sort(a.begin(), a.end(), cmp); std::stable_sort(b.begin(), b.end(), cmp);
        std::vector<rec> want(n1 + n2); std::merge(a.begin(), a.end(), b.begin(), b.end(), want.begin(), cmp);
        for (unsigned th : kThreads) {
            task::pool p(th);
            std::vector<rec> got(n1 + n2);
            task::parallel_merge(&p, a.begin(), a.end(), b.begin(), b.end(), got.begin(), cmp, 512);
            CHECK_MSG(got == want, "round=%d threads=%u", round, th);
        }
    }
    // k-way
    std::vector<std::vector<rec>> runs;
    auto cmp = [](const rec& x, const rec& y) { return x.key < y.key; };
    for (int r = 0; r < 7; ++r) { auto v = make_recs(3000 + 500 * r, 40, 100 + r); std::stable_sort(v.begin(), v.end(), cmp); runs.push_back(std::move(v)); }
    std::vector<rec> want; for (auto& r : runs) want.insert(want.end(), r.begin(), r.end());
    std::stable_sort(want.begin(), want.end(), cmp);   // run order = concatenation order, so stable_sort is the reference
    for (unsigned th : kThreads) { task::pool p(th); CHECK(task::merge_runs(&p, runs, cmp, 256) == want); }
}

TEST(topk_equals_stable_sort_prefix) {
    auto base = make_recs(200000, 1000, 3);   // ties everywhere: the stable order decides
    auto cmp = [](const rec& a, const rec& b) { return a.key > b.key; };
    for (std::size_t k : {1u, 10u, 1000u, 300000u}) {
        auto want = base; std::stable_sort(want.begin(), want.end(), cmp); want.resize(std::min<std::size_t>(k, want.size()));
        for (unsigned th : kThreads) {
            task::pool p(th);
            auto got = task::parallel_topk(&p, base.begin(), base.end(), k, cmp, 2048);
            CHECK_MSG(got == want, "k=%zu threads=%u", std::size_t(k), th);
        }
        CHECK(task::parallel_topk(nullptr, base.begin(), base.end(), k, cmp, 2048) == want);
    }
    auto none = task::parallel_topk_by(nullptr, 0, 5, [](std::size_t i) { return int(i); }, std::less<>{});
    CHECK(none.empty());
}

TEST(reduce_and_scan_equal_serial) {
    std::mt19937_64 g(9);
    std::vector<std::int64_t> v(250001); for (auto& x : v) x = std::int64_t(g() % 1000) - 500;
    const auto want_sum = std::accumulate(v.begin(), v.end(), std::int64_t(7));
    std::vector<std::int64_t> want_inc(v.size()); std::inclusive_scan(v.begin(), v.end(), want_inc.begin());
    std::vector<std::int64_t> want_exc(v.size()); std::exclusive_scan(v.begin(), v.end(), want_exc.begin(), std::int64_t(7));
    const auto want_max = *std::max_element(v.begin(), v.end());
    for (unsigned th : kThreads) {
        task::pool p(th);
        CHECK(task::parallel_reduce(&p, v.begin(), v.end(), std::int64_t(7)) == want_sum);
        CHECK(task::parallel_reduce(&p, v.begin(), v.end(), std::int64_t(-1000000), [](auto a, auto b) { return std::max(a, b); }) == want_max);
        std::vector<std::int64_t> inc(v.size()), exc(v.size());
        task::parallel_inclusive_scan(&p, v.begin(), v.end(), inc.begin());
        task::parallel_exclusive_scan(&p, v.begin(), v.end(), exc.begin(), std::int64_t(7));
        CHECK_MSG(inc == want_inc, "threads=%u", th);
        CHECK_MSG(exc == want_exc, "threads=%u", th);
        auto in_place = v; task::parallel_inclusive_scan(&p, in_place.begin(), in_place.end(), in_place.begin(), std::plus<>{}, 1000);
        CHECK(in_place == want_inc);
    }
    // floating point: the blocked result is identical for every thread count (it is not a plain left fold)
    std::vector<double> f(100000); for (auto& x : f) x = double(g() % 100000) * 1e-3 + 1e-9 * double(g() % 7);
    const double serial = task::parallel_reduce(nullptr, f.begin(), f.end(), 0.0);
    for (unsigned th : kThreads) { task::pool p(th); CHECK(task::parallel_reduce(&p, f.begin(), f.end(), 0.0) == serial); }
    CHECK(task::parallel_reduce(nullptr, f.begin(), f.begin(), 5.0) == 5.0);
}
