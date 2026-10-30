#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/core/memory.hpp>
#include <forge/mind/mind.hpp>
#include <cstring>
#include <filesystem>
#include <thread>
#include <unistd.h>

using namespace forge;
static std::vector<int> greedy(mind::session& s, std::vector<int> p, int n) { mind::sampler g; s.reset(); return s.generate(p, n, g); }

TEST(parse_and_format_sizes) {
    CHECK(mem::parse_size("512") == 512u);
    CHECK(mem::parse_size("4k") == 4096u);
    CHECK(mem::parse_size("1.5G") == 1610612736u);
    CHECK(mem::parse_size("2GiB") == 2147483648u);
    CHECK(mem::parse_size("8 MB") == 8u << 20);
    CHECK(mem::parse_size("50%", 1000) == 500u);
    CHECK(!mem::parse_size("50%"));
    CHECK(!mem::parse_size("lots"));
    CHECK(!mem::parse_size("-1G"));
    CHECK(mem::format_size(1536) == "1.5 KiB");
}
TEST(host_limits_detected) {
    auto h = mem::detect_host_limits();
    CHECK(h.physical > 0);
    CHECK(h.effective() > 0 && h.effective() <= h.physical);
    std::printf("  physical=%s available=%s cgroup=%s\n", mem::format_size(h.physical).c_str(), mem::format_size(h.available).c_str(), mem::format_size(h.cgroup).c_str());
}
TEST(budget_reserve_release_and_tags) {
    mem::budget b("t", 1000);
    { auto a = b.reserve(400, "weights"); auto c = b.try_reserve(500, "kv");
      CHECK(a && c); CHECK(b.used() == 900); CHECK(b.stats().by_tag["weights"] == 400);
      CHECK(!b.try_reserve(200)); CHECK(b.stats().denials == 1);
      CHECK_THROWS(b.reserve(200, "x"));
      CHECK(c.resize(550)); CHECK(b.used() == 950); CHECK(!c.resize(700)); CHECK(c.resize(100)); CHECK(b.used() == 500);
      auto moved = std::move(a); CHECK(b.used() == 500); }
    CHECK(b.used() == 0); CHECK(b.stats().peak == 950);
    CHECK_THROWS(b.reserve(5000));
    try { b.reserve(5000); } catch (const forge_exception& e) { CHECK(e.code == errc::out_of_memory); }
}
TEST(budget_evicts_caches_lowest_priority_lru_first) {
    mem::budget b("t", 1000);
    std::uint64_t a = 300, c = 300; std::vector<std::string> order;
    int ia = b.add_cache("a", [&] { return a; }, [&](std::uint64_t) { order.push_back("a"); auto f = a; a = 0; return f; }, 0);
    b.add_cache("c", [&] { return c; }, [&](std::uint64_t) { order.push_back("c"); auto f = c; c = 0; return f; }, 0);
    b.touch_cache(ia);   // a is now most recently used
    auto l = b.reserve(600, "work");
    CHECK(b.cache_bytes() == 600);
    // Reserve treats caches as outside used_, so they only get evicted when asked or under trim().
    CHECK(b.under_pressure());
    auto freed = b.trim(); CHECK(freed == 300); CHECK(order.size() == 1 && order[0] == "c");
    CHECK(b.evict(0) == 300); CHECK(order.size() == 2 && order[1] == "a");
    CHECK(b.stats().evictions == 2 && b.stats().evicted_bytes == 600);
}
TEST(budget_try_reserve_evicts_to_fit) {
    mem::budget b("t", 1000);
    auto held = b.reserve(500, "a");
    mem::lease spare = b.reserve(400, "cache");
    b.add_cache("c", [&] { return spare.bytes(); }, [&](std::uint64_t) { auto n = spare.bytes(); spare.release(); return n; });
    auto l = b.try_reserve(300, "new");   // only fits once the cache gives its lease back
    CHECK(l); CHECK(b.used() == 800);
}
TEST(budget_reserve_waits_for_release) {
    mem::budget b("t", 1000);
    auto a = b.reserve(900);
    std::thread t([&] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); a.release(); });
    auto l = b.reserve(500, "late", std::chrono::milliseconds(5000));
    t.join(); CHECK(l); CHECK(b.stats().waits == 1);
}
TEST(budget_set_usage_and_prometheus) {
    mem::budget b("host", 1 << 20);
    b.set_usage("kv", 4096); CHECK(b.used() == 4096);
    b.set_usage("kv", 1024); CHECK(b.used() == 1024);
    auto p = b.prometheus();
    CHECK(p.find("forge_mem_limit_bytes{pool=\"host\"} 1048576") != std::string::npos);
    CHECK(p.find("forge_mem_used_bytes_by_tag{pool=\"host\",tag=\"kv\"} 1024") != std::string::npos);
    b.set_usage("kv", 0); CHECK(b.used() == 0);
}
TEST(pmr_resource_charges_budget) {
    mem::budget b("t", 64 << 10);
    mem::budget_resource r(b, "vec");
    { std::pmr::vector<char> v(&r); v.resize(16 << 10); CHECK(b.used() >= (16u << 10)); CHECK_THROWS(v.resize(128 << 10)); }
    CHECK(b.used() == 0);
}
TEST(governor_env_and_pressure) {
    ::setenv("FORGE_MEM_DEVICE_LIMIT", "1M", 1);
    auto& g = mem::governor::instance();
    CHECK(g.pool("testdev:0").limit() == (1u << 20));
    CHECK(g.host().limit() > 0);
    std::uint64_t c = 100; g.pool("testdev:0").add_cache("c", [&] { return c; }, [&](std::uint64_t) { auto f = c; c = 0; return f; });
    g.on_pressure(os::pressure_level::critical);
    CHECK(c == 0); CHECK(g.pressure_events() >= 1);
    CHECK(g.prometheus().find("forge_mem_pressure_events_total") != std::string::npos);
    CHECK(g.describe().find("budget   host") != std::string::npos);
}
TEST(spill_roundtrip_and_cap) {
    auto dir = std::filesystem::temp_directory_path() / ("forge_spill_test_" + std::to_string(::getpid()));
    {
        mem::spill_store s(dir);
        std::vector<int> v(1000); for (int i = 0; i < 1000; ++i) v[std::size_t(i)] = i * 3;
        auto k = s.put(v.data(), v.size() * sizeof(int)); CHECK(k); CHECK(s.bytes() == 4000);
        auto back = s.take(k); CHECK(back && back->size() == 4000);
        CHECK(std::memcmp(back->data(), v.data(), 4000) == 0); CHECK(s.bytes() == 0); CHECK(!s.take(k));
    }
    CHECK(!std::filesystem::exists(dir));
}
TEST(scheduler_reports_kv_and_defers_under_budget) {
    auto path = (std::filesystem::temp_directory_path() / ("forge_test_mem_" + std::to_string(::getpid()) + ".fpk")).string();
    forge_test::make_tiny_model(path);
    {   // scope: the model's mapping must be gone before the file is deleted (Windows)
    auto m = mind::model::open(path);
    mem::budget b("kvtest", 0);
    std::vector<std::vector<int>> outs(4), want(4);
    for (int i = 0; i < 4; ++i) { mind::session s(m); want[std::size_t(i)] = greedy(s, {256, i + 1, 2}, 6); }
    {   // unlimited: behaves exactly like before but reports KV usage
        mind::batch_scheduler s(m, {.max_active = 4, .prefix_cache = 2, .budget = &b});
        for (int i = 0; i < 4; ++i) s.submit({.prompt = {256, i + 1, 2}, .max_new = 6, .on_done = [&, i](const std::vector<int>& o, const std::string&) { outs[std::size_t(i)] = o; }});
        s.run_until_idle();
        CHECK(s.stats().peak_active == 4); CHECK(s.stats().deferred == 0);
        CHECK(b.cache_bytes() > 0);
    }
    for (int i = 0; i < 4; ++i) CHECK(outs[std::size_t(i)] == want[std::size_t(i)]);
    CHECK(b.used() == 0); CHECK(b.cache_bytes() == 0);
    b.set_limit(1);   // absurdly tight: requests must run one at a time but still all finish, same output
    std::fill(outs.begin(), outs.end(), std::vector<int>{});
    mind::batch_scheduler s(m, {.max_active = 4, .prefix_cache = 2, .budget = &b});
    for (int i = 0; i < 4; ++i) s.submit({.prompt = {256, i + 1, 2}, .max_new = 6, .on_done = [&, i](const std::vector<int>& o, const std::string&) { outs[std::size_t(i)] = o; }});
    s.run_until_idle();
    CHECK(s.stats().peak_active == 1); CHECK(s.stats().deferred > 0); CHECK(s.stats().completed == 4);
    for (int i = 0; i < 4; ++i) CHECK(outs[std::size_t(i)] == want[std::size_t(i)]);
    }
    std::filesystem::remove(path);
}
