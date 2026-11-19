// forge::db parallel layer: partitioned query / scan / filter / reduce / top-k / bm25 / sort / bulk load equal the serial
// result for 1, 2, 3, 8 and 16 threads (fixed seed).
#include "check.hpp"
#include <forge/db/parallel.hpp>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <random>
#include <string>
using namespace forge;
namespace fs = std::filesystem;

static const unsigned kThreads[] = {1, 2, 3, 8, 16};
static const char* kWords[] = {"bitvec", "add", "comm", "nat", "zero", "lemma", "theorem", "list", "append", "assoc", "mul", "int", "fin", "decide", "simp", "omega", "ring", "norm", "cast", "shift"};

struct fixture {
    fs::path path;
    std::size_t n = 30000;
    explicit fixture(const char* tag) {
        path = fs::temp_directory_path() / (std::string("forge_dbpar_") + tag + "_" + std::to_string(std::uintptr_t(this) & 0xfffff) + ".db");
        std::error_code ec; fs::remove(path, ec);
        auto c = db::connection::open(path.string(), {.wal = true, .synchronous = "OFF"});
        c.exec("CREATE TABLE items(id INTEGER PRIMARY KEY, grp INTEGER, v REAL, name TEXT, body TEXT)");
        db::fts5::create(c, "fts", {"name", "body"}, {.content = "items", .content_rowid = "id"});
        std::mt19937_64 g(2024);
        db::transaction tx(c);
        auto ins = c.prepare("INSERT INTO items(id, grp, v, name, body) VALUES(?,?,?,?,?)");
        std::int64_t id = 100;
        for (std::size_t i = 0; i < n; ++i) {
            id += 1 + std::int64_t(g() % 5);   // gaps: rowids are not dense
            std::string body;
            for (int w = 0, nw = 3 + int(g() % 12); w < nw; ++w) { body += kWords[g() % 20]; body += ' '; }
            ins.run(id, std::int64_t(g() % 17), double(g() % 500) / 8.0, "n" + std::to_string(g() % 977), body);   // many equal scores
        }
        tx.commit();
        db::fts5::rebuild(c, "fts");
        c.exec("PRAGMA wal_checkpoint(TRUNCATE)");
    }
    ~fixture() { std::error_code ec; fs::remove(path, ec); fs::remove(path.string() + "-wal", ec); fs::remove(path.string() + "-shm", ec); }
};

TEST(partitions_cover_the_rowid_range) {
    for (std::int64_t lo : {std::int64_t(1), std::int64_t(-50), std::int64_t(1) << 40})
        for (std::size_t parts : {1u, 2u, 7u, 64u, 1000u}) {
            auto r = db::split_rowid_range(lo, lo + 99, parts);
            CHECK(!r.empty()); CHECK(r.size() == std::min<std::size_t>(parts, 100));
            CHECK(r.front().lo == lo); CHECK(r.back().hi == lo + 99);
            for (std::size_t i = 1; i < r.size(); ++i) CHECK(r[i].lo == r[i - 1].hi + 1);
        }
    CHECK(db::split_rowid_range(5, 4, 8).empty());
    auto one = db::split_rowid_range(7, 7, 8); CHECK(one.size() == 1 && one[0].lo == 7 && one[0].hi == 7);
}

TEST(query_scan_filter_reduce_equal_serial) {
    fixture fx("q");
    db::read_pool rp(fx.path.string(), 16);
    auto parts = db::rowid_partitions(rp, "items", 64);
    CHECK(parts.size() == 64);
    auto direct = db::connection::open(fx.path.string(), {.read_only = true});
    using R = std::pair<std::int64_t, double>;
    std::vector<R> want;
    { auto st = direct.prepare("SELECT id, v FROM items ORDER BY id"); for (auto& r : st.rows()) want.push_back({r.get<std::int64_t>(0), r.get<double>(1)}); }
    CHECK(want.size() == fx.n);
    const auto sql = db::range_sql("items", "id, v");
    auto fn = [](const db::row& r) { return R{r.get<std::int64_t>(0), r.get<double>(1)}; };
    auto want_filter = std::vector<std::int64_t>();
    for (auto& [id, v] : want) if (v > 40.0 && id % 3 == 0) want_filter.push_back(id);
    const std::int64_t want_isum = std::accumulate(want.begin(), want.end(), std::int64_t(0), [](auto a, auto& p) { return a + p.first; });
    const double serial_dsum = db::parallel_reduce<double>(rp, nullptr, sql, parts, 0.0, [](const db::row& r) { return r.get<double>(1); }, std::plus<>{});
    for (unsigned th : kThreads) {
        task::pool tp(th);
        CHECK_MSG((db::parallel_query<R>(rp, &tp, sql, parts, fn) == want), "query threads=%u", th);
        std::vector<std::int64_t> per(parts.size(), 0); std::vector<std::size_t> cnt(parts.size(), 0);
        db::parallel_scan(rp, &tp, sql, parts, [&](std::size_t i, const db::row& r) { per[i] += r.get<std::int64_t>(0); ++cnt[i]; });
        CHECK(std::accumulate(per.begin(), per.end(), std::int64_t(0)) == want_isum);
        CHECK(std::accumulate(cnt.begin(), cnt.end(), std::size_t(0)) == fx.n);
        auto got_filter = db::parallel_filter<std::int64_t>(rp, &tp, sql, parts, [](const db::row& r) { return r.get<double>(1) > 40.0 && r.get<std::int64_t>(0) % 3 == 0; },
                                                            [](const db::row& r) { return r.get<std::int64_t>(0); });
        CHECK_MSG(got_filter == want_filter, "filter threads=%u", th);
        CHECK(db::parallel_reduce<std::int64_t>(rp, &tp, sql, parts, 0, [](const db::row& r) { return r.get<std::int64_t>(0); }, std::plus<>{}) == want_isum);
        CHECK_MSG(db::parallel_reduce<double>(rp, &tp, sql, parts, 0.0, [](const db::row& r) { return r.get<double>(1); }, std::plus<>{}) == serial_dsum, "double reduce threads=%u", th);
    }
    // the extra WHERE of range_sql, and an empty range
    auto ex = db::parallel_query<std::int64_t>(rp, nullptr, db::range_sql("items", "id", "grp = 3"), parts, [](const db::row& r) { return r.get<std::int64_t>(0); });
    CHECK(ex.size() == std::size_t(direct.scalar<std::int64_t>("SELECT count(*) FROM items WHERE grp = 3")));
    CHECK(db::parallel_query<R>(rp, nullptr, sql, {}, fn).empty());
}

TEST(topk_equals_serial_for_any_thread_count_and_partitioning) {
    fixture fx("k");
    db::read_pool rp(fx.path.string(), 16);
    auto direct = db::connection::open(fx.path.string(), {.read_only = true});
    for (std::size_t k : {1u, 10u, 100u, 5000u}) {
        std::vector<db::scored> want;
        { auto st = direct.prepare("SELECT id, v FROM items ORDER BY v DESC, id ASC LIMIT ?1"); st.bind(1, std::int64_t(k));
          for (auto& r : st.rows()) want.push_back({r.get<std::int64_t>(0), r.get<double>(1)}); }
        for (std::size_t np : {7u, 64u}) {
            auto parts = db::rowid_partitions(rp, "items", np);
            for (unsigned th : kThreads) {
                task::pool tp(th);
                auto got = db::parallel_topk(rp, &tp, db::range_sql("items", "id, v").substr(0, db::range_sql("items", "id, v").find(" ORDER BY")), parts, k);
                CHECK_MSG(got == want, "k=%zu parts=%zu threads=%u", std::size_t(k), std::size_t(np), th);
            }
        }
    }
}

TEST(bm25_topk_equals_serial) {
    fixture fx("b");
    db::read_pool rp(fx.path.string(), 16);
    auto direct = db::connection::open(fx.path.string(), {.read_only = true});
    for (const char* text : {"bitvec add", "lemma theorem omega", "ring norm cast shift decide", "nat"}) {
        const auto q = db::fts5::query_from_text(text);
        for (std::size_t k : {5u, 50u}) {
            auto m = db::fts5::search(direct, "fts", q, k);
            std::vector<db::scored> want; for (auto& x : m) want.push_back({x.rowid, x.score});
            CHECK(want.size() == k);
            for (unsigned th : kThreads) {
                task::pool tp(th);
                CHECK_MSG(db::parallel_bm25_topk(rp, &tp, "fts", q, k, 64) == want, "q=%s k=%zu threads=%u", text, std::size_t(k), th);
                CHECK(db::parallel_bm25_topk(rp, &tp, "fts", q, k, 5) == want);   // another partitioning
            }
        }
    }
    CHECK(db::parallel_bm25_topk(rp, nullptr, "fts", "", 5).empty());
}

TEST(sort_equals_stable_sort) {
    fixture fx("s");
    db::read_pool rp(fx.path.string(), 16);
    auto parts = db::rowid_partitions(rp, "items", 32);
    auto direct = db::connection::open(fx.path.string(), {.read_only = true});
    struct rec { std::int64_t id, grp; std::string name; bool operator==(const rec&) const = default; };
    std::vector<rec> want;
    { auto st = direct.prepare("SELECT id, grp, name FROM items ORDER BY id"); for (auto& r : st.rows()) want.push_back({r.get<std::int64_t>(0), r.get<std::int64_t>(1), r.get<std::string>(2)}); }
    auto cmp = [](const rec& a, const rec& b) { return a.grp != b.grp ? a.grp < b.grp : a.name < b.name; };   // ties keep rowid order
    std::stable_sort(want.begin(), want.end(), cmp);
    for (unsigned th : kThreads) {
        task::pool tp(th);
        auto got = db::parallel_sort<rec>(rp, &tp, db::range_sql("items", "id, grp, name"), parts,
                                          [](const db::row& r) { return rec{r.get<std::int64_t>(0), r.get<std::int64_t>(1), r.get<std::string>(2)}; }, cmp);
        CHECK_MSG(got == want, "threads=%u", th);
    }
}

TEST(bulk_load_equals_serial_insert) {
    auto dir = fs::temp_directory_path() / ("forge_dbbulk_" + std::to_string(std::uintptr_t(&kThreads) & 0xfffff));
    fs::create_directories(dir);
    const std::size_t rows = 20001;
    auto fill = [](std::size_t i, db::statement& s) { s.bind(1, std::int64_t(i * 7 % 1000)).bind(2, "row" + std::to_string(i)).bind(3, double(i) / 4).bind(4, i % 5 == 0 ? std::optional<int>() : std::optional<int>(int(i))); };
    auto make = [&](const char* nm) { auto c = db::connection::open((dir / nm).string()); c.exec("CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, a INTEGER, b TEXT, c REAL, d INTEGER)"); return c; };
    auto serial = make("serial.db");
    { db::transaction tx(serial); auto ins = serial.prepare("INSERT INTO t(a, b, c, d) VALUES(?,?,?,?)");
      for (std::size_t i = 0; i < rows; ++i) { ins.reset(); fill(i, ins); ins.step(); } tx.commit(); }
    auto dump = [](db::connection& c) { std::string s; auto st = c.prepare("SELECT id, a, b, c, d FROM t ORDER BY id");
        for (auto& r : st.rows()) { s += std::to_string(r.get<std::int64_t>(0)) + "|" + std::to_string(r.get<std::int64_t>(1)) + "|" + r.get<std::string>(2) + "|" + std::to_string(r.get<double>(3)) + "|" + (r.is_null(4) ? "N" : std::to_string(r.get<std::int64_t>(4))) + "\n"; } return s; };
    const std::string want = dump(serial);
    for (unsigned th : {1u, 3u, 8u})
        for (std::size_t parts : {1u, 3u, 8u}) {
            task::pool tp(th);
            auto par = make(("par" + std::to_string(th) + "_" + std::to_string(parts) + ".db").c_str());
            auto n = db::parallel_bulk_load(&tp, par, {.table = "t", .columns = {"a", "b", "c", "d"}, .rows = rows, .parts = parts}, fill);
            CHECK(n == std::int64_t(rows));
            CHECK_MSG(dump(par) == want, "threads=%u parts=%zu", th, std::size_t(parts));
            CHECK(par.integrity_check().empty());
        }
    // a failing merge (constraint) rolls the target back and leaves no staging files behind
    auto bad = make("bad.db"); bad.exec("CREATE UNIQUE INDEX ub ON t(b)");
    auto dup = [](std::size_t i, db::statement& s) { s.bind(1, 1).bind(2, "same" + std::to_string(i % 10)).bind(3, 0.5).bind(4, 1); };
    { task::pool tp(2); CHECK_THROWS(db::parallel_bulk_load(&tp, bad, {.table = "t", .columns = {"a", "b", "c", "d"}, .rows = 100, .parts = 4}, dup)); }
    CHECK(bad.scalar<std::int64_t>("SELECT count(*) FROM t") == 0);
    std::size_t leftovers = 0; for (auto& e : fs::directory_iterator(dir)) if (e.path().filename().string().rfind("forge_stage_", 0) == 0) ++leftovers;
    CHECK(leftovers == 0);
    std::error_code ec; fs::remove_all(dir, ec);
}
