// perf/bench_db: sharded forge::db database against one SQLite file, on synthetic data (random-word text with a skewed
// vocabulary, 32-d float vectors). Not a test; build with -DFORGE_BUILD_DB_BENCH=ON.
//
//   bench_db --dir D:/scratch/dbbench --rows 1000000 [--budget-mib 10] [--threads 1,4,8] [--no-single] [--ann-rows N]
//
// Measures: build time and size; cold and warm point lookups, prefix scans, BM25 top-10, ANN top-10, and a full
// scan + reduce, for each thread count. "Cold" evicts the files from the page cache with posix_fadvise(DONTNEED) and opens
// fresh handles. Results of the two layouts are compared (the same ids and scores), so a mismatch aborts the run.
#include <forge/ann/ann.hpp>
#include <forge/db/shards.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>
#if defined(__unix__)
#include <fcntl.h>
#include <unistd.h>
#endif
using namespace forge;
namespace fs = std::filesystem;
using fvec = std::vector<float>;
using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

static constexpr std::size_t kDim = 32;
static std::string word(std::uint64_t r) { return "w" + std::to_string(r * r / 40000); }   // skewed: small ids are frequent
static void gen_row(std::uint64_t i, std::string& body, fvec& vec) {
    std::mt19937_64 g(0x9E3779B97F4A7C15ull * (i + 1));
    body.clear(); const int nw = 12 + int(g() % 24);
    for (int w = 0; w < nw; ++w) { body += word(g() % 4000); body += ' '; }
    vec.resize(kDim); std::normal_distribution<float> nd; for (auto& f : vec) f = nd(g);
}
static std::vector<std::byte> bytes_of(const fvec& v) { return {reinterpret_cast<const std::byte*>(v.data()), reinterpret_cast<const std::byte*>(v.data()) + v.size() * 4}; }
static const char* kDdl = "CREATE TABLE t(k BLOB PRIMARY KEY, id INTEGER NOT NULL, body TEXT, vec BLOB) WITHOUT ROWID;"
                          "CREATE VIRTUAL TABLE fts USING fts5(body, content='', tokenize='unicode61');";

static void evict(const fs::path& p) {
#if defined(__unix__)
    auto one = [](const fs::path& f) { int fd = ::open(f.string().c_str(), O_RDONLY); if (fd >= 0) { (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); } };
    if (fs::is_directory(p)) for (auto& e : fs::directory_iterator(p)) one(e.path()); else one(p);
#else
    (void)p;
#endif
}
static std::uint64_t dir_bytes(const fs::path& p) { std::uint64_t n = 0; if (fs::is_directory(p)) { for (auto& e : fs::directory_iterator(p)) n += e.is_regular_file() ? e.file_size() : 0; } else n = fs::file_size(p); return n; }

int main(int argc, char** argv) {
    fs::path dir = "bench_db_out"; std::uint64_t rows = 1000000, budget_mib = 10, ann_rows = 0; std::vector<unsigned> threads = {1, 4, 8}; bool do_single = true, reuse = false; db::verify_mode vm = db::verify_mode::sha256; std::size_t nlook = 20000, npre = 200, nq = 100;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--dir") dir = next(); else if (a == "--rows") rows = std::stoull(next()); else if (a == "--budget-mib") budget_mib = std::stoull(next());
        else if (a == "--ann-rows") ann_rows = std::stoull(next()); else if (a == "--no-single") do_single = false; else if (a == "--reuse") reuse = true;
        else if (a == "--verify") { auto v = next(); vm = v == "none" ? db::verify_mode::none : v == "size" ? db::verify_mode::size : v == "blake3" ? db::verify_mode::blake3 : db::verify_mode::sha256; }
        else if (a == "--lookups") nlook = std::stoull(next()); else if (a == "--prefixes") npre = std::stoull(next()); else if (a == "--queries") nq = std::stoull(next());
        else if (a == "--threads") { threads.clear(); std::stringstream ss(next()); std::string t; while (std::getline(ss, t, ',')) threads.push_back(unsigned(std::stoul(t))); }
    }
    if (!ann_rows) ann_rows = rows;
    fs::create_directories(dir);
    const fs::path sdir = dir / "sharded", single = dir / "single.db";
    std::printf("# bench_db: %llu rows, budget %llu MiB, dim %zu, dir %s\n", (unsigned long long)rows, (unsigned long long)budget_mib, kDim, dir.string().c_str());

    // ---- build ---------------------------------------------------------------------------------------------
    const unsigned build_threads = 8;
    task::pool bp(build_threads);
    auto sc = db::shard_schema{}; sc.ddl = kDdl; sc.table = "t"; sc.columns = {"k", "id", "body", "vec"}; sc.fts_table = "fts";
    sc.finalize = [af = ann::ann_finalize<fvec, ann::l2>("SELECT id, vec FROM t ORDER BY k", {.M = 8, .ef_construction = 64, .ef_search = 64, .seed = 1}, {})](db::connection& c) {
        c.exec("INSERT INTO fts(rowid, body) SELECT id, body FROM t ORDER BY k"); af(c);
    };
    auto t0 = clk::now();
    if (!reuse || !fs::exists(sdir / "manifest.json")) {
        fs::remove_all(sdir);
        db::shard_writer w(sdir, sc, {.budget = budget_mib << 20, .presorted = true, .wave_bytes = 128ull << 20}, &bp);
        std::string body; fvec vec;
        for (std::uint64_t i = 0; i < rows; ++i) { gen_row(i, body, vec); const std::int64_t id = std::int64_t(i + 1); w.add({db::int_key(id), id, body, bytes_of(vec)}); }
        w.finish();
    }
    const double build_sharded = since(t0);
    auto set0 = db::open(sdir);
    if (!reuse || build_sharded > 0.5) std::printf("build sharded: %.1f s (%u threads), %zu shards, %.1f MiB\n", build_sharded, build_threads, set0.size(), double(dir_bytes(sdir)) / (1 << 20));
    double build_single = 0;
    if (do_single && (!reuse || !fs::exists(single))) {
        fs::remove(single);
        t0 = clk::now();
        {
            auto c = db::connection::open(single.string(), {.synchronous = "OFF"}); c.exec(kDdl);
            std::string body; fvec vec;
            { db::transaction tx(c); auto ins = c.prepare("INSERT INTO t(k, id, body, vec) VALUES(?,?,?,?)");
              for (std::uint64_t i = 0; i < rows; ++i) { gen_row(i, body, vec); const std::int64_t id = std::int64_t(i + 1); const auto k = db::int_key(id); ins.run(db::static_blob{db::static_span_(k)}, id, body, bytes_of(vec)); }
              c.exec("INSERT INTO fts(rowid, body) SELECT id, body FROM t ORDER BY k"); tx.commit(); }
            // one global graph, stored in the file (what an unsharded database would keep)
            ann::hnsw<fvec, ann::l2> h({.M = 8, .ef_construction = 64, .ef_search = 64, .seed = 1});
            auto st = c.prepare("SELECT id, vec FROM t ORDER BY k"); for (auto& r : st.rows()) h.add(std::uint64_t(r.get<std::int64_t>(0)), ann::codec<fvec>::decode(r.get<std::span<const std::byte>>(1)));
            h.build(&bp); h.save(c, "ann");
            c.exec("PRAGMA journal_mode=DELETE"); c.exec("VACUUM");
        }
        build_single = since(t0);
        std::printf("build single:  %.1f s (insert serial, global graph on %u threads), 1 file, %.1f MiB\n", build_single, build_threads, double(dir_bytes(single)) / (1 << 20));
    }

    // ---- queries -------------------------------------------------------------------------------------------
    std::mt19937_64 g(77);
    std::vector<std::int64_t> look(nlook); for (auto& x : look) x = std::int64_t(g() % rows) + 1;
    std::vector<std::string> prefixes(npre); for (auto& p : prefixes) p = db::int_key(std::int64_t(g() % rows) + 1).substr(0, 7);   // ~256 consecutive ids
    std::vector<std::vector<std::string>> queries(nq); for (auto& q : queries) { const int n = 2 + int(g() % 2); for (int i = 0; i < n; ++i) q.push_back(word(g() % 4000)); }
    std::vector<fvec> qv(nq); { std::normal_distribution<float> nd; for (auto& v : qv) { v.resize(kDim); for (auto& f : v) f = nd(g); } }
    auto blob = [](const std::string& s) { return db::static_span_(s); };

    // exact nearest neighbours of the ANN queries (brute force over every vector), to score both indexes
    std::vector<std::vector<ann::hit>> exact_nn;
    {
        ann::flat_index<fvec, ann::l2> flat; fvec v; std::string body;
        for (std::uint64_t r = 0; r < rows; ++r) { gen_row(r, body, v); flat.add(r + 1, v); }
        for (auto& q : qv) exact_nn.push_back(flat.search(q, 10, &bp));
    }
    auto recall = [&](const std::vector<std::vector<ann::hit>>& got) { std::size_t ok = 0; for (std::size_t r = 0; r < nq; ++r) for (auto& h : got[r]) for (auto& e : exact_nn[r]) if (e.id == h.id) { ++ok; break; } return double(ok) / double(nq * 10); };
    std::printf("\n| op | layout | cache | threads | time | per op |\n|---|---|---|---|---|---|\n");
    auto row = [&](const char* op, const char* layout, const char* cache, unsigned th, double s, double nops) {
        std::printf("| %s | %s | %s | %u | %.3f s | %.1f us |\n", op, layout, cache, th, s, s / nops * 1e6); std::fflush(stdout);
    };
    std::uint64_t chk_sharded = 0, chk_single = 0;
    for (unsigned th : threads) {
        task::pool tp(th);
        for (int cold = 1; cold >= 0; --cold) {
            const char* cn = cold ? "cold" : "warm";
            // sharded
            if (cold) evict(sdir);
            t0 = clk::now();
            auto set = std::make_unique<db::shard_set>(db::open(sdir, {.verify = vm}));
            const double t_open = since(t0);
            if (cold) row("open (verify lazily)", "sharded", cn, th, t_open, 1);
            t0 = clk::now(); std::vector<std::int64_t> sums(nlook);
            task::parallel_blocks(&tp, nlook, 256, [&](std::size_t, std::size_t b, std::size_t e) { for (std::size_t i = b; i < e; ++i) {
                const auto key = db::int_key(look[i]); auto idx = set->shards_for_key(key);
                auto l = set->acquire(idx.at(0)); auto st = l->prepare("SELECT id FROM t WHERE k = ?1"); st.bind(1, blob(key)); sums[i] = st.step() ? st.get<std::int64_t>(0) : -1; } });
            row("point lookup", "sharded", cn, th, since(t0), nlook);
            for (std::size_t i = 0; i < nlook; ++i) if (sums[i] != look[i]) { std::fprintf(stderr, "lookup mismatch\n"); return 1; }
            t0 = clk::now(); std::uint64_t nrows = 0;
            for (auto& p : prefixes) { auto idx = set->shards_for_prefix(p); auto e = db::shard_set::prefix_end(p);
                auto v = db::shard_query<std::int64_t>(*set, &tp, idx, "SELECT id FROM t WHERE k >= :lo AND k < :hi", [&](db::statement& st) { st.bind(":lo", blob(p)).bind(":hi", blob(e)); }, [](const db::row& r) { return r.get<std::int64_t>(0); }); nrows += v.size(); }
            row("prefix scan", "sharded", cn, th, since(t0), npre); chk_sharded = nrows;
            t0 = clk::now(); std::vector<std::vector<db::shard_hit>> br; std::size_t skipped = 0;
            for (auto& q : queries) { db::bm25_report rep; br.push_back(db::shard_bm25_topk(*set, &tp, "fts", q, 10, {}, &rep)); skipped += rep.skipped_bound; }
            row("bm25 top-10", "sharded", cn, th, since(t0), nq);
            t0 = clk::now(); ann::shard_ann<fvec, ann::l2> sa(*set);
            std::vector<std::vector<ann::hit>> ah; for (auto& v : qv) ah.push_back(sa.search(v, 10, &tp));
            row("ann top-10 (cold: incl. graph load)", "sharded", cn, th, since(t0), nq);
            if (!cold) std::printf("|   (per-shard graphs: ann recall@10 against brute force %.3f)| | | | | |\n", recall(ah));
            t0 = clk::now();
            auto parts = db::shard_map<std::int64_t>(*set, &tp, set->all_shards(), [](db::connection& c, std::size_t) { return c.scalar<std::int64_t>("SELECT coalesce(sum(length(body)), 0) FROM t"); });
            std::int64_t total = 0; for (auto x : parts) total += x;
            row("scan+reduce sum(length(body))", "sharded", cn, th, since(t0), 1);
            std::printf("|   (bm25 shards skipped by bound: %zu of %zu)| | | | | |\n", skipped, queries.size() * set->size());
            if (!do_single) continue;
            // single file
            if (cold) evict(single);
            t0 = clk::now(); db::open_options ro; ro.mmap_size = std::int64_t(fs::file_size(single)); ro.cache_size_kib = 1024;
            db::read_pool rp(single.string(), th, ro);
            if (cold) row("open", "single", cn, th, since(t0), 1);
            t0 = clk::now();
            task::parallel_blocks(&tp, nlook, 256, [&](std::size_t, std::size_t b, std::size_t e) { auto l = rp.acquire(); auto st = l->prepare("SELECT id FROM t WHERE k = ?1");
                for (std::size_t i = b; i < e; ++i) { const auto key = db::int_key(look[i]); st.reset(); st.bind(1, blob(key)); sums[i] = st.step() ? st.get<std::int64_t>(0) : -1; } });
            row("point lookup", "single", cn, th, since(t0), nlook);
            t0 = clk::now(); nrows = 0;
            { auto l = rp.acquire(); auto st = l->prepare("SELECT id FROM t WHERE k >= ?1 AND k < ?2");
              for (auto& p : prefixes) { auto e = db::shard_set::prefix_end(p); st.reset(); st.bind(1, blob(p)).bind(2, blob(e)); while (st.step()) ++nrows; } }
            row("prefix scan", "single", cn, th, since(t0), npre); chk_single = nrows;
            t0 = clk::now(); std::vector<std::vector<db::scored>> sr;
            auto bparts = db::rowid_partitions(rp, "fts", 64);
            for (auto& q : queries) sr.push_back(db::parallel_bm25_topk(rp, &tp, "fts", db::fts5::query_from_text([&] { std::string s; for (auto& t : q) s += t + " "; return s; }()), 10, bparts));
            row("bm25 top-10", "single", cn, th, since(t0), nq);
            for (std::size_t i = 0; i < nq; ++i) { if (sr[i].size() != br[i].size()) { std::fprintf(stderr, "bm25 size mismatch q%zu\n", i); return 1; }
                for (std::size_t j = 0; j < sr[i].size(); ++j) if (sr[i][j].id != br[i][j].id || sr[i][j].score != br[i][j].score) { std::fprintf(stderr, "bm25 mismatch q%zu\n", i); return 1; } }
            t0 = clk::now();
            ann::hnsw<fvec, ann::l2> gh; { auto l = rp.acquire(); gh = ann::hnsw<fvec, ann::l2>::load(*l, "ann"); }
            std::vector<std::vector<ann::hit>> gr; for (auto& v : qv) gr.push_back(gh.search(v, 10));
            row("ann top-10 (cold: incl. graph load)", "single", cn, th, since(t0), nq);
            if (!cold) std::printf("|   (ann recall@10 against brute force: per-shard graphs %.3f, one global graph %.3f)| | | | | |\n", recall(ah), recall(gr));
            t0 = clk::now();
            std::vector<std::pair<std::string, std::string>> kparts;
            for (std::int64_t i = 0; i < 64; ++i) kparts.push_back({db::int_key(std::int64_t(rows) * i / 64 + 1), db::int_key(std::int64_t(rows) * (i + 1) / 64 + 1)});
            auto sp = db::parallel_map<std::int64_t>(rp, &tp, kparts, [](db::connection& c, const std::pair<std::string, std::string>& r) {
                auto st = c.prepare("SELECT coalesce(sum(length(body)), 0) FROM t WHERE k >= ?1 AND k < ?2"); st.bind(1, db::static_span_(r.first)).bind(2, db::static_span_(r.second)); st.step(); return st.get<std::int64_t>(0); });
            std::int64_t total2 = 0; for (auto x : sp) total2 += x;
            row("scan+reduce sum(length(body))", "single", cn, th, since(t0), 1);
            if (total2 != total) { std::fprintf(stderr, "scan mismatch %lld vs %lld\n", (long long)total, (long long)total2); return 1; }
        }
    }
    if (do_single) std::printf("\nprefix-scan rows: sharded %llu, single %llu\n", (unsigned long long)chk_sharded, (unsigned long long)chk_single);
    return 0;
}
