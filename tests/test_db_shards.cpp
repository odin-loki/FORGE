// forge::db shard sets (db/shards.hpp): byte-reproducible builds, budget, manifest, key/prefix/range lookup, exact BM25 over
// shards (== one unsharded table), per-shard ANN, verification, content-addressed store, cache. Synthetic data only.
#include "check.hpp"
#include <forge/ann/ann.hpp>
#include <forge/db/shards.hpp>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>
using namespace forge;
namespace fs = std::filesystem;

static const unsigned kThreads[] = {1, 2, 3, 8, 16};
using fvec = std::vector<float>;

static fs::path scratch(const char* tag) {
    static int n = 0;
    auto d = fs::temp_directory_path() / (std::string("forge_shards_") + tag + "_" + std::to_string(std::uintptr_t(&n) & 0xfffff) + "_" + std::to_string(n++));
    std::error_code ec; fs::remove_all(d, ec); fs::create_directories(d); return d;
}

struct doc { std::int64_t id; std::string body; fvec vec; };
static std::vector<doc> make_docs(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 g(seed); std::normal_distribution<float> nd;
    std::vector<doc> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i].id = std::int64_t(i + 1) * 3;
        const int nw = 4 + int(g() % 12);
        for (int w = 0; w < nw; ++w) { const std::uint64_t r = g() % 1000; v[i].body += "w" + std::to_string(r * r / 5000) + (w % 5 == 0 ? "x" : "") + " "; }   // skewed word frequencies
        if (i % 997 == 0) v[i].body += "needle ";   // a rare term, so some shards cannot match it
        v[i].vec.resize(16); for (auto& f : v[i].vec) f = nd(g);
    }
    return v;
}
static std::string vec_bytes(const fvec& v) { return std::string(reinterpret_cast<const char*>(v.data()), v.size() * 4); }

static const char* kDdl = "CREATE TABLE t(k BLOB PRIMARY KEY, id INTEGER NOT NULL, body TEXT, vec BLOB) WITHOUT ROWID;"
                          "CREATE VIRTUAL TABLE fts USING fts5(body, content='', tokenize='unicode61');";
static db::shard_schema make_schema(bool with_ann, std::function<void(db::connection&)> extra = {}) {
    db::shard_schema sc; sc.ddl = kDdl; sc.table = "t"; sc.columns = {"k", "id", "body", "vec"}; sc.key_index = 0; sc.fts_table = "fts"; sc.schema_version = 3;
    auto ann = with_ann ? ann::ann_finalize<fvec, ann::l2>("SELECT id, vec FROM t ORDER BY k", {.M = 12, .ef_construction = 100, .ef_search = 64, .seed = 5}, {}) : std::function<void(db::connection&)>{};
    sc.finalize = [ann, extra](db::connection& c) {
        c.exec("INSERT INTO fts(rowid, body) SELECT id, body FROM t ORDER BY k");
        if (ann) ann(c);
        if (extra) extra(c);
    };
    return sc;
}
static std::vector<db::value> row_of(const doc& d) {
    return {db::int_key(d.id), d.id, d.body, std::vector<std::byte>(reinterpret_cast<const std::byte*>(d.vec.data()), reinterpret_cast<const std::byte*>(d.vec.data()) + d.vec.size() * 4)};
}
// the same rows in ONE sqlite file: the unsharded reference
static void build_unsharded(const fs::path& file, const std::vector<doc>& docs) {
    auto c = db::connection::open(file.string());
    c.exec(kDdl);
    db::transaction tx(c);
    auto ins = c.prepare("INSERT INTO t(k, id, body, vec) VALUES(?,?,?,?)");
    for (auto& d : docs) ins.run(db::static_blob{db::static_span_(db::int_key(d.id))}, d.id, d.body, std::span<const std::byte>(reinterpret_cast<const std::byte*>(d.vec.data()), d.vec.size() * 4));
    c.exec("INSERT INTO fts(rowid, body) SELECT id, body FROM t ORDER BY k");
    tx.commit();
}

TEST(sha256_and_keys) {
    CHECK(hash::sha256::of_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(hash::sha256::of_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(hash::sha256::of_hex(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    CHECK(db::int_key(-5) < db::int_key(-1)); CHECK(db::int_key(-1) < db::int_key(0)); CHECK(db::int_key(0) < db::int_key(7)); CHECK(db::int_key(7) < db::int_key(1ll << 40));
    CHECK(db::shard_set::prefix_end("ab") == "ac"); CHECK(db::shard_set::prefix_end(std::string("a\xff", 2)) == "b");
}

TEST(reproducible_budgeted_builds) {
    auto docs = make_docs(6000, 1);
    const std::uint64_t budget = 192 << 10;
    std::string ref_manifest; std::vector<std::string> ref_hashes;
    for (unsigned th : kThreads) {
        auto dir = scratch("repro");
        task::pool tp(th);
        db::shard_writer w(dir, make_schema(true), {.budget = budget, .presorted = true, .wave_bytes = 300 << 10}, &tp);   // several waves
        for (auto& d : docs) w.add(row_of(d));
        w.set_stat("docs", json::value(double(docs.size())));
        auto m = w.finish();
        CHECK(m.shards.size() > 5);
        std::uint64_t rows = 0; for (auto& s : m.shards) { CHECK(s.bytes <= budget); CHECK(fs::file_size(dir / s.file) == s.bytes); rows += s.rows; }
        CHECK(rows == docs.size()); CHECK(m.rows == docs.size());
        std::string text; { std::ifstream f(dir / "manifest.json", std::ios::binary); text.assign(std::istreambuf_iterator<char>(f), {}); }
        std::vector<std::string> hs; for (auto& s : m.shards) hs.push_back(hash::sha256::file_hex((dir / s.file).string()));
        for (std::size_t i = 0; i < hs.size(); ++i) CHECK(hs[i] == m.shards[i].sha256);
        if (ref_manifest.empty()) { ref_manifest = text; ref_hashes = hs; }
        else { CHECK_MSG(text == ref_manifest, "manifest differs at %u threads", th); CHECK_MSG(hs == ref_hashes, "shard bytes differ at %u threads", th); }
        // no temp files left behind
        for (auto& de : fs::directory_iterator(dir)) CHECK(de.path().filename().string().rfind(".tmp", 0) != 0);
        std::error_code ec; fs::remove_all(dir, ec);
    }
    // unsorted input is sorted by the writer and gives the same shards as presorted input (single wave)
    auto shuffled = docs; std::mt19937_64 g(9); std::shuffle(shuffled.begin(), shuffled.end(), g);
    auto d1 = scratch("a"), d2 = scratch("b");
    { db::shard_writer w(d1, make_schema(false), {.budget = budget}, nullptr); for (auto& d : shuffled) w.add(row_of(d)); w.finish(); }
    { db::shard_writer w(d2, make_schema(false), {.budget = budget, .presorted = true, .wave_bytes = std::uint64_t(1) << 40}, nullptr); for (auto& d : docs) w.add(row_of(d)); w.finish(); }
    std::string a, b; { std::ifstream f(d1 / "manifest.json", std::ios::binary); a.assign(std::istreambuf_iterator<char>(f), {}); } { std::ifstream f(d2 / "manifest.json", std::ios::binary); b.assign(std::istreambuf_iterator<char>(f), {}); }
    CHECK(a == b);
    { db::shard_writer bad(scratch("c"), make_schema(false), {.presorted = true});
      bad.add({db::int_key(9), 9, "x", std::vector<std::byte>{}});
      CHECK_THROWS(bad.add({db::int_key(5), 5, "x", std::vector<std::byte>{}})); }
    std::error_code ec; fs::remove_all(d1, ec); fs::remove_all(d2, ec);
}

TEST(over_budget_shards_are_split) {
    auto docs = make_docs(4000, 2);
    // later rows get a large filler table: the first shard's size estimate is too optimistic, so some shards must be split
    auto sc = make_schema(false);
    // the filler must be deterministic (no randomblob): hex(zeroblob) with a counter
    sc.finalize = [inner = sc.finalize](db::connection& c) {
        inner(c);
        c.exec("DROP TABLE IF EXISTS filler"); c.exec("CREATE TABLE filler(x BLOB)");
        if (c.scalar<std::int64_t>("SELECT min(id) FROM t") > 6000) c.exec("INSERT INTO filler SELECT hex(zeroblob(450) || cast(i AS BLOB)) FROM (WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i < 4 * (SELECT count(*) FROM t)) SELECT i FROM n)");
    };
    const std::uint64_t budget = 160 << 10;
    auto dir = scratch("split"); task::pool tp(4);
    db::shard_writer w(dir, sc, {.budget = budget, .presorted = true}, &tp);
    for (auto& d : docs) w.add(row_of(d));
    auto m = w.finish();
    std::uint64_t rows = 0; bool big = false;
    for (auto& s : m.shards) { CHECK(s.bytes <= budget); rows += s.rows; big |= s.rows < 100; }
    CHECK(rows == docs.size());
    std::printf("  split test: %zu shards for %zu rows (budget %llu bytes)\n", m.shards.size(), docs.size(), (unsigned long long)budget);
    for (std::size_t i = 1; i < m.shards.size(); ++i) CHECK(m.shards[i - 1].max_key <= m.shards[i].min_key);
    std::error_code ec; fs::remove_all(dir, ec);
}

struct built_fixture {
    std::vector<doc> docs; fs::path dir, single; std::uint64_t budget = 192 << 10;
    built_fixture() {
        docs = make_docs(9000, 4);
        dir = scratch("fx"); single = scratch("fxs") / "all.db";
        task::pool tp(8);
        db::shard_writer w(dir, make_schema(true), {.budget = budget, .presorted = true}, &tp);
        for (auto& d : docs) w.add(row_of(d));
        w.finish();
        build_unsharded(single, docs);
    }
    ~built_fixture() { std::error_code ec; fs::remove_all(dir, ec); fs::remove_all(single.parent_path(), ec); }
};
static built_fixture& fx() { static built_fixture f; return f; }

TEST(manifest_lookup_range_prefix_equal_unsharded) {
    auto& F = fx();
    auto set = db::open(F.dir);
    CHECK(set.size() > 5); CHECK(!set.is_single_file()); CHECK(set.manifest().schema_version == 3);
    auto ref = db::connection::open(F.single.string(), {.read_only = true});
    task::pool tp(4);
    std::mt19937_64 g(3);
    for (int i = 0; i < 200; ++i) {   // point lookups touch exactly the shard(s) the key range allows
        const std::int64_t id = std::int64_t(g() % 9100) * 3 / 3 + 0;   // ids are multiples of 3: roughly 2/3 of probes miss
        const auto key = db::int_key(id);
        auto idx = set.shards_for_key(key);
        CHECK(idx.size() <= 1);
        auto got = db::shard_query<std::string>(set, &tp, idx, "SELECT body FROM t WHERE k = :k", [&](db::statement& st) { st.bind(":k", db::static_span_(key)); }, [](const db::row& r) { return r.get<std::string>(0); });
        auto want = ref.scalar_or<std::string>("<none>", "SELECT body FROM t WHERE k = ?1", std::span<const std::byte>(reinterpret_cast<const std::byte*>(key.data()), key.size()));
        if (want == "<none>") CHECK(got.empty()); else { CHECK(got.size() == 1); CHECK(got[0] == want); }
    }
    // range scan over keys [lo, hi)
    for (int trial = 0; trial < 20; ++trial) {
        std::int64_t a = std::int64_t(g() % 27000), b = a + std::int64_t(g() % 4000);
        const auto lo = db::int_key(a), hi = db::int_key(b);
        auto got = db::shard_range_query<std::int64_t>(set, &tp, lo, hi, "SELECT id FROM t WHERE k >= :lo AND k < :hi ORDER BY k", [](const db::row& r) { return r.get<std::int64_t>(0); });
        std::vector<std::int64_t> want; auto st = ref.prepare("SELECT id FROM t WHERE id >= ?1 AND id < ?2 ORDER BY id"); st.bind(1, a).bind(2, b);
        for (auto& r : st.rows()) want.push_back(r.get<std::int64_t>(0));
        CHECK_MSG(got == want, "range %lld..%lld", (long long)a, (long long)b);
        CHECK(set.shards_for_range(lo, hi).size() < set.size() || b - a > 20000);
    }
    // prefix lookup: text keys sharing a prefix (the first 7 bytes of int_key are equal for ids that differ in the low byte)
    const std::string prefix = db::int_key(3000).substr(0, 7);
    auto idx = set.shards_for_prefix(prefix);
    auto got = db::shard_query<std::int64_t>(set, &tp, idx, "SELECT id FROM t WHERE k >= :lo AND k < :hi ORDER BY k",
                                             [&](db::statement& st) { st.bind(":lo", db::static_span_(prefix)); const auto e = db::shard_set::prefix_end(prefix); st.bind(":hi", db::static_span_(e)); },
                                             [](const db::row& r) { return r.get<std::int64_t>(0); });
    std::vector<std::int64_t> want; { auto st = ref.prepare("SELECT id FROM t WHERE substr(k, 1, 7) = ?1 ORDER BY k"); st.bind(1, std::span<const std::byte>(reinterpret_cast<const std::byte*>(prefix.data()), prefix.size())); for (auto& r : st.rows()) want.push_back(r.get<std::int64_t>(0)); }
    CHECK(got == want); CHECK(!want.empty()); CHECK(idx.size() <= 2);
    // full scan / reduce over all shards, every thread count
    const auto want_sum = ref.scalar<std::int64_t>("SELECT sum(id) FROM t");
    for (unsigned th : kThreads) {
        task::pool p(th);
        auto sums = db::shard_map<std::int64_t>(set, &p, set.all_shards(), [](db::connection& c, std::size_t) { return c.scalar<std::int64_t>("SELECT coalesce(sum(id), 0) FROM t"); });
        CHECK(std::accumulate(sums.begin(), sums.end(), std::int64_t(0)) == want_sum);
        auto blocked = db::shard_map<std::int64_t>(set, &p, set.all_shards(), [](db::connection& c, std::size_t) { return c.scalar<std::int64_t>("SELECT coalesce(sum(id), 0) FROM t"); }, {.schedule = db::shard_schedule::blocked});
        CHECK(blocked == sums);
        auto top = db::shard_topk(set, &p, set.all_shards(), "SELECT id, length(body) FROM t", [](db::statement&) {}, 25);
        auto st = ref.prepare("SELECT id, length(body) AS l FROM t ORDER BY l DESC, id ASC LIMIT 25"); std::vector<db::scored> w;
        for (auto& r : st.rows()) w.push_back({r.get<std::int64_t>(0), r.get<double>(1)});
        CHECK_MSG(top == w, "topk threads=%u", th);
    }
}

TEST(bm25_over_shards_equals_unsharded) {
    auto& F = fx();
    auto set = db::open(F.dir);
    auto ref = db::connection::open(F.single.string(), {.read_only = true});
    const std::vector<std::vector<std::string>> queries = {{"w0", "w1"}, {"w100", "w50x", "w7"}, {"w199", "w180"}, {"w3", "w4", "w5", "w6", "w7", "w8"}, {"w0x"}, {"nosuchterm", "w2"}, {"needle", "w0"}, {"needle", "w1", "w2"}};
    std::size_t skipped_total = 0;
    for (const auto& terms : queries)
        for (const char* op : {"OR", "AND"}) for (std::size_t k : {1u, 10u, 100u}) {
            std::string expr; for (std::size_t i = 0; i < terms.size(); ++i) { if (i) expr += std::string(" ") + op + " "; expr += db::fts5::quote_phrase(terms[i]); }
            std::vector<db::shard_hit> want;
            { auto st = ref.prepare("SELECT rowid, -bm25(fts) AS s FROM fts WHERE fts MATCH ?1 ORDER BY s DESC, rowid ASC LIMIT ?2"); st.bind(1, expr).bind(2, std::int64_t(k));
              for (auto& r : st.rows()) want.push_back({0, r.get<std::int64_t>(0), r.get<double>(1)}); }
            for (unsigned th : kThreads) for (bool prune : {true, false}) {
                task::pool tp(th); db::bm25_report rep;
                auto got = db::shard_bm25_topk(set, &tp, "fts", terms, k, {.op = op, .prune = prune}, &rep);
                bool same = got.size() == want.size();
                for (std::size_t i = 0; same && i < got.size(); ++i) same = got[i].id == want[i].id && got[i].score == want[i].score;   // exact: ids AND bit-identical scores
                CHECK_MSG(same, "bm25 mismatch op=%s k=%zu threads=%u prune=%d first=%s", op, std::size_t(k), th, int(prune), terms[0].c_str());
                if (prune) skipped_total += rep.skipped_bound;
                CHECK(rep.rows == 9000);
            }
        }
    std::printf("  bm25: shards skipped by the score bound over all queries: %zu\n", skipped_total);
    // weights
    { std::vector<db::shard_hit> want; auto st = ref.prepare("SELECT rowid, -bm25(fts, 2.5) FROM fts WHERE fts MATCH ?1 ORDER BY 2 DESC, rowid ASC LIMIT 20"); st.bind(1, db::fts5::quote_phrase("w0") + " OR " + db::fts5::quote_phrase("w1"));
      for (auto& r : st.rows()) want.push_back({0, r.get<std::int64_t>(0), r.get<double>(1)});
      auto got = db::shard_bm25_topk(set, nullptr, "fts", {"w0", "w1"}, 20, {.weights = {2.5}});
      CHECK(got.size() == want.size()); for (std::size_t i = 0; i < got.size(); ++i) CHECK(got[i].id == want[i].id && got[i].score == want[i].score); }
    // a one-shard set over the single file gives the same answer
    auto one = db::open(F.single);
    CHECK(one.size() == 1 && one.is_single_file());
    auto a = db::shard_bm25_topk(one, nullptr, "fts", {"w0", "w1"}, 10), b = db::shard_bm25_topk(set, nullptr, "fts", {"w0", "w1"}, 10);
    CHECK(a.size() == b.size()); for (std::size_t i = 0; i < a.size(); ++i) CHECK(a[i].id == b[i].id && a[i].score == b[i].score);
    // cached result
    auto cdir = scratch("cache"); cache::store store(cdir, {.max_bytes = 1 << 20});
    int computed = 0;
    auto run = [&] { return db::decode_hits(db::cached_result(store, set, {"bm25", "w0 w1", "10"}, [&] { ++computed; return db::encode_hits(b); })); };
    CHECK(run() == b); CHECK(run() == b); CHECK(computed == 1);
    std::error_code ec; fs::remove_all(cdir, ec);
}

TEST(ann_per_shard_recall_and_thread_independence) {
    auto& F = fx();
    auto set = db::open(F.dir);
    ann::shard_ann<fvec, ann::l2> ix(set);
    ann::flat_index<fvec, ann::l2> flat;
    for (auto& d : F.docs) flat.add(std::uint64_t(d.id), d.vec);
    std::mt19937_64 g(8); std::normal_distribution<float> nd;
    double rec = 0; const int nq = 60;
    task::pool p8(8);
    for (int q = 0; q < nq; ++q) {
        fvec v(16); for (auto& f : v) f = nd(g);
        auto want = flat.search(v, 10, &p8);
        std::set<std::uint64_t> w; for (auto& h : want) w.insert(h.id);
        auto ref = ix.search(v, 10, nullptr);
        for (auto& h : ref) rec += w.count(h.id);
        for (unsigned th : kThreads) { task::pool tp(th); CHECK(ix.search(v, 10, &tp) == ref); }
        CHECK(std::is_sorted(ref.begin(), ref.end(), ann::hit_before));
    }
    rec /= nq * 10.0;
    std::printf("  per-shard hnsw over %zu shards: recall@10 vs flat_index = %.4f (ef=64)\n", set.size(), rec);
    CHECK(rec >= 0.95);
    CHECK(ix.loaded() == set.size());
}

TEST(verification_and_content_addressed_store) {
    auto docs = make_docs(1500, 6);
    auto dir = scratch("ver"), blobs = scratch("blobs");
    { db::shard_writer w(dir, make_schema(false), {.budget = 128 << 10, .blob_dir = blobs}); for (auto& d : docs) w.add(row_of(d)); w.finish(); }
    { auto set = db::open(dir); CHECK(set.size() >= 2); set.verify_all(nullptr); auto l = set.acquire(0); CHECK(l->scalar<std::int64_t>("SELECT count(*) FROM t") > 0); }
    // blob store: shard files are kept as <sha256>.fdb, and an opened set falls back to them
    { auto set = db::open(dir); const auto& s0 = set.info(0); CHECK(fs::exists(blobs / (s0.sha256 + ".fdb"))); }
    auto victim = [&] { auto set = db::open(dir); return dir / set.info(1).file; }();
    const auto saved = fs::path(dir / "saved.fdb"); fs::copy_file(victim, saved);
    fs::remove(victim);
    { auto set = db::open(dir, {.blob_dir = blobs}); auto l = set.acquire(1); CHECK(l->scalar<std::int64_t>("SELECT count(*) FROM t") > 0); }
    // corrupt one byte: the manifest hash no longer matches
    fs::copy_file(saved, victim);
    { std::fstream f(victim, std::ios::in | std::ios::out | std::ios::binary); f.seekp(5000); char c = 0; f.read(&c, 1); f.seekp(5000); c = char(c ^ 0x55); f.write(&c, 1); }
    { auto set = db::open(dir); try { set.verify_all(nullptr); CHECK(false); } catch (const forge_exception& e) { CHECK(e.code == errc::integrity_error); }
      try { auto l = set.acquire(1); CHECK(false); } catch (const forge_exception& e) { CHECK(e.code == errc::integrity_error); }
      auto l0 = set.acquire(0); CHECK(l0->scalar<std::int64_t>("SELECT count(*) FROM t") > 0);   // other shards still open
      auto weak = db::open(dir, {.verify = db::verify_mode::size}); auto l1 = weak.acquire(1); (void)l1; }   // same size: only the hash catches it
    // a truncated shard fails the size check
    fs::resize_file(victim, 1000);
    { auto set = db::open(dir, {.verify = db::verify_mode::size}); CHECK_THROWS(set.acquire(1)); }
    std::error_code ec; fs::remove_all(dir, ec); fs::remove_all(blobs, ec);
}

TEST(lru_budget_limits_open_shards) {
    auto& F = fx();
    auto set = db::open(F.dir, {.max_open_bytes = 3 * F.budget});
    for (std::size_t i = 0; i < set.size(); ++i) { auto l = set.acquire(i); (void)l->scalar<std::int64_t>("SELECT count(*) FROM t"); }
    CHECK(set.open_shards() <= 4);   // 3 shards' bytes, at most one more than the budget while the last lease was held
    auto single = db::open_single(F.single.string(), {.read_only = true});
    CHECK(single.scalar<std::int64_t>("SELECT count(*) FROM t") == 9000);
}
