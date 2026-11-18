// forge::db (db/sqlite.hpp): connection, statements, typed binding and columns, transactions, errors, FTS5.
#include "check.hpp"
#include <forge/db/sqlite.hpp>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
using namespace forge;
namespace fs = std::filesystem;

static fs::path tmp(const char* tag) {
    auto d = fs::temp_directory_path() / (std::string("forge_db_") + tag + "_" + std::to_string(std::uintptr_t(&tag) & 0xffffff));
    std::error_code ec; fs::remove(d, ec); return d;
}

TEST(sqlite_build_options) {
    db::connection c = db::connection::open(":memory:");
    CHECK(std::string(sqlite3_libversion()) == "3.54.1");
    // compile options of the vendored build: FTS5, RTREE, THREADSAFE=2, DQS=0
    bool fts5 = false, rtree = false, ts2 = false, dqs0 = false;
    auto st = c.prepare("PRAGMA compile_options");
    for (auto& r : st.rows()) {
        auto o = r.get<std::string>(0);
        fts5 |= o == "ENABLE_FTS5"; rtree |= o == "ENABLE_RTREE"; ts2 |= o == "THREADSAFE=2"; dqs0 |= o == "DQS=0";
    }
    CHECK(fts5); CHECK(rtree); CHECK(ts2); CHECK(dqs0);
    CHECK(c.scalar<std::int64_t>("PRAGMA foreign_keys") == 1);
    CHECK(c.scalar<std::string>("SELECT json_extract('{\"a\":[1,2,3]}', '$.a[1]')") == "2");   // JSON is built in
    // DQS=0: a double-quoted string that is not an identifier is an error
    CHECK_THROWS(c.exec("SELECT \"nosuchcolumn\""));
}

TEST(typed_bind_and_columns) {
    auto c = db::connection::open("");
    c.exec("CREATE TABLE t(id INTEGER PRIMARY KEY, i INTEGER, d REAL, s TEXT, b BLOB, n)");
    auto ins = c.prepare("INSERT INTO t(i, d, s, b, n) VALUES(?, ?, ?, ?, ?)");
    std::vector<std::byte> blob{std::byte{0}, std::byte{1}, std::byte{255}};
    CHECK(ins.run(std::int64_t(-5), 2.5, std::string("hello"), blob, nullptr) == 1);
    CHECK(ins.run(7u, 0.25f, "lit", std::span<const std::byte>(blob), std::optional<int>(3)) == 1);
    CHECK(ins.run(true, 1.0, std::string_view("sv"), std::vector<std::byte>{}, std::optional<int>()) == 1);
    CHECK(c.last_insert_rowid() == 3);
    auto st = c.prepare("SELECT id, i, d, s, b, n FROM t ORDER BY id");
    std::vector<std::tuple<std::int64_t, std::int64_t, double, std::string, std::size_t, std::optional<std::int64_t>>> got;
    for (auto& r : st.rows()) got.push_back({r.get<std::int64_t>(0), r.get<std::int64_t>(1), r.get<double>(2), r.get<std::string>(3), r.get<std::span<const std::byte>>(4).size(), r.get<std::optional<std::int64_t>>(5)});
    CHECK(got.size() == 3);
    CHECK(std::get<1>(got[0]) == -5 && std::get<2>(got[0]) == 2.5 && std::get<3>(got[0]) == "hello" && std::get<4>(got[0]) == 3 && !std::get<5>(got[0]));
    CHECK(std::get<1>(got[1]) == 7 && std::get<2>(got[1]) == 0.25 && std::get<5>(got[1]) == 3);
    CHECK(std::get<1>(got[2]) == 1 && std::get<4>(got[2]) == 0);
    auto [cnt, mx] = c.query_row<std::int64_t, double>("SELECT count(*), max(d) FROM t");
    CHECK(cnt == 3 && mx == 2.5);
    CHECK(c.scalar<bool>("SELECT i FROM t WHERE id = ?1", 3));
    CHECK(c.scalar_or<std::int64_t>(-1, "SELECT i FROM t WHERE id = ?1", 99) == -1);
    CHECK_THROWS(c.scalar<std::int64_t>("SELECT i FROM t WHERE id = ?1", 99));
    // named parameters and column lookup by name
    auto nm = c.prepare("SELECT s FROM t WHERE id BETWEEN :lo AND :hi ORDER BY id");
    nm.bind(":lo", 1).bind(":hi", 2);
    CHECK(nm.step()); CHECK(nm.current().get<std::string_view>("s") == "hello");
    CHECK_THROWS(nm.bind(":nope", 1));
    // a string with an embedded NUL round-trips
    std::string withnul("a\0b", 3);
    c.execute("INSERT INTO t(s) VALUES(?)", withnul);
    CHECK(c.scalar<std::string>("SELECT s FROM t ORDER BY id DESC LIMIT 1") == withnul);
}

TEST(transactions_and_errors) {
    auto c = db::connection::open("");
    c.exec("CREATE TABLE u(k TEXT PRIMARY KEY, v INTEGER NOT NULL)");
    { db::transaction tx(c); c.execute("INSERT INTO u VALUES('a', 1)"); tx.commit(); }
    { db::transaction tx(c); c.execute("INSERT INTO u VALUES('b', 2)"); }   // rolled back
    CHECK(c.scalar<std::int64_t>("SELECT count(*) FROM u") == 1);
    { db::transaction outer(c);
      c.execute("INSERT INTO u VALUES('c', 3)");
      { db::transaction inner(c); c.execute("INSERT INTO u VALUES('d', 4)"); inner.rollback(); }
      { db::transaction inner(c); c.execute("INSERT INTO u VALUES('e', 5)"); inner.commit(); }
      outer.commit(); }
    CHECK(c.scalar<std::int64_t>("SELECT count(*) FROM u") == 3);
    CHECK(!c.in_transaction());
    try { c.execute("INSERT INTO u VALUES('a', 9)"); CHECK(false); }
    catch (const db::error& e) { CHECK(e.constraint()); CHECK(e.code == errc::invalid_argument); CHECK(e.extended_code == SQLITE_CONSTRAINT_PRIMARYKEY); CHECK(!e.sql.empty()); }
    try { c.prepare("SELEC 1"); CHECK(false); } catch (const forge_exception& e) { CHECK(e.code == errc::parse_error); }
    CHECK_THROWS(c.execute("INSERT INTO u VALUES('z', ?)", nullptr));   // NOT NULL
    auto r = db::connection::try_open("/nonexistent-dir/x.db", {.create = false});
    CHECK(!r); CHECK(r.error().code == errc::io_error);
    CHECK(c.integrity_check().empty());
}

TEST(file_wal_read_only_immutable) {
    auto p = tmp("wal"); auto ps = p.string();
    { auto w = db::connection::open(ps, {.wal = true, .synchronous = "NORMAL"});
      CHECK(w.pragma("journal_mode") == "wal");
      w.exec("CREATE TABLE t(x INTEGER)"); w.execute("INSERT INTO t VALUES(1)");
      auto r = db::connection::open(ps, {.read_only = true});
      CHECK(r.scalar<std::int64_t>("SELECT count(*) FROM t") == 1);
      CHECK_THROWS(r.execute("INSERT INTO t VALUES(2)"));
      w.execute("INSERT INTO t VALUES(3)");
      CHECK(r.scalar<std::int64_t>("SELECT count(*) FROM t") == 2); }
    { auto w = db::connection::open(ps); w.exec("PRAGMA wal_checkpoint(TRUNCATE)"); w.exec("PRAGMA journal_mode=DELETE"); }
    { auto im = db::connection::open(ps, {.immutable = true});
      CHECK(im.scalar<std::int64_t>("SELECT count(*) FROM t") == 2);
      CHECK_THROWS(im.execute("INSERT INTO t VALUES(2)")); }
    CHECK_THROWS(db::connection::open(ps + ".missing", {.read_only = true}));
    std::error_code ec; fs::remove(p, ec); fs::remove(ps + "-wal", ec); fs::remove(ps + "-shm", ec);
    CHECK(db::file_uri("C:\\a b\\c#d?.db", "mode=ro") == "file:/C:/a%20b/c%23d%3F.db?mode=ro");
}

TEST(fts5_helpers) {
    auto c = db::connection::open("");
    c.exec("CREATE TABLE docs(id INTEGER PRIMARY KEY, name TEXT, body TEXT)");
    db::fts5::create(c, "fts", {"name", "body"}, {.content = "docs", .content_rowid = "id"});
    auto ins = c.prepare("INSERT INTO docs(id, name, body) VALUES(?, ?, ?)");
    const char* bodies[] = {"bitvec add comm theorem", "nat add zero lemma", "list append assoc", "bitvec mul comm lemma", "unrelated words here"};
    for (int i = 0; i < 5; ++i) ins.run(i + 1, "d" + std::to_string(i), bodies[i]);
    db::fts5::rebuild(c, "fts");
    db::fts5::integrity_check(c, "fts");
    auto q = db::fts5::query_from_text("BitVec ADD, comm \"x\" NEAR AND");
    CHECK(q == "\"BitVec\" OR \"ADD\" OR \"comm\" OR \"x\" OR \"NEAR\" OR \"AND\"");
    auto m = db::fts5::search(c, "fts", db::fts5::query_from_text("bitvec comm"), 10);
    CHECK(m.size() == 2); CHECK(m[0].score >= m[1].score);
    CHECK(db::fts5::search(c, "fts", "", 5).empty());
    CHECK(db::fts5::search(c, "fts", db::fts5::query_from_text("zzzz"), 5).empty());
    CHECK_THROWS(db::fts5::search(c, "fts", "AND OR (", 5));   // raw, unquoted syntax error
    CHECK(db::fts5::words("a_b-c d9\xc3\xa9").size() == 3);
}
