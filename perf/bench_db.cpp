// perf/bench_db: sharded forge::db database against one SQLite file, on synthetic data (random-word text with a skewed vocabulary
// plus rare and clustered terms, 32-d float vectors). Not a test; build with -DFORGE_BUILD_DB_BENCH=ON.
//
//   bench_db --dir D:/scratch/dbbench --rows 3600000 [--budget-mib 10] [--threads 1,4,8,16] [--no-single] [--reuse]
//            [--single-queries 10] [--build-only] [--sections layout,bm25,io,ann]
//
// Sections (all print markdown tables):
//   layout  point lookup, prefix scan, BM25 (frequent words) and a full scan, sharded against one file, cold and warm, per thread count
//   bm25    query classes (frequent, clustered-rare, rare+frequent, AND, unique, phrase): time, shards counted/with hits/scored/skipped,
//           term filters on and off, no pruning, against the unsharded table (ids and exact scores compared)
//   io      prefetch kinds (none, fadvise, aio) and schedules (dynamic, blocked, numa) on a cold pass of BM25 and a full scan
//   ann     recall@10 and time against ef for the per-shard graphs and for one global graph with the same parameters
// "Cold" evicts the files with posix_fadvise(DONTNEED) and opens fresh handles; "warm" is the next pass on the same handles. NOTE: on a
// 9p mount (WSL drvfs) DONTNEED does not drop anything and a fresh mapping re-reads every page from the host, so "cold" there means
// "fresh handle"; see docs/db.md.
#include <forge/ann/ann.hpp>
#include <forge/db/shards.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <set>
#include <sstream>
using namespace forge;
namespace fs = std::filesystem;
using fvec = std::vector<float>;
using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

static constexpr std::size_t kDim = 32;
static std::string word(std::uint64_t r) { return "w" + std::to_string(r * r / 40000); }   // skewed: small ids are frequent
// Row i: 12 to 35 skewed words; a clustered term "c<i/40000>" in every third row of its id range (so it lives in 2 to 3 shards); a unique-ish term
// "u<i/200000>" in rows with i % 20011 == 0 (a handful of rows spread over a few shards).
static void gen_row(std::uint64_t i, std::string& body, fvec& vec) {
    std::mt19937_64 g(0x9E3779B97F4A7C15ull * (i + 1));
    body.clear(); const int nw = 12 + int(g() % 24);
    for (int w = 0; w < nw; ++w) { body += word(g() % 4000); body += ' '; }
    if (i % 3 == 0) body += "c" + std::to_string(i / 40000) + " ";
    if (i % 20011 == 0) body += "u" + std::to_string(i / 200000) + " ";
    vec.resize(kDim); std::normal_distribution<float> nd; for (auto& f : vec) f = nd(g);
}
static std::vector<std::byte> bytes_of(const fvec& v) { return {reinterpret_cast<const std::byte*>(v.data()), reinterpret_cast<const std::byte*>(v.data()) + v.size() * 4}; }
static const char* kDdl = "CREATE TABLE t(k BLOB PRIMARY KEY, id INTEGER NOT NULL, body TEXT, vec BLOB) WITHOUT ROWID;"
                          "CREATE VIRTUAL TABLE fts USING fts5(body, content='', tokenize='unicode61');";

static void evict(const fs::path& p) {
    if (fs::is_directory(p)) { for (auto& e : fs::directory_iterator(p)) if (e.is_regular_file()) db::drop_page_cache(e.path().string()); } else db::drop_page_cache(p.string());
}
static std::uint64_t dir_bytes(const fs::path& p) { std::uint64_t n = 0; if (fs::is_directory(p)) { for (auto& e : fs::directory_iterator(p)) n += e.is_regular_file() ? e.file_size() : 0; } else n = fs::file_size(p); return n; }
static std::string expr_of(const std::vector<std::string>& q, const char* op) { std::string e; for (std::size_t i = 0; i < q.size(); ++i) { if (i) e += std::string(" ") + op + " "; e += db::fts5::quote_phrase(q[i]); } return e; }

int main(int argc, char** argv) {
    fs::path dir = "bench_db_out"; std::uint64_t rows = 1000000, budget_mib = 10; std::vector<unsigned> threads = {1, 4, 8, 16};
    bool do_single = true, reuse = false, build_only = false; db::verify_mode vm = db::verify_mode::sha256;
    std::size_t nlook = 20000, npre = 200, nq = 100, nsq = 10; std::set<std::string> sections = {"layout", "bm25", "io", "ann"};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--dir") dir = next(); else if (a == "--rows") rows = std::stoull(next()); else if (a == "--budget-mib") budget_mib = std::stoull(next());
        else if (a == "--no-single") do_single = false; else if (a == "--reuse") reuse = true; else if (a == "--build-only") build_only = true;
        else if (a == "--verify") { auto v = next(); vm = v == "none" ? db::verify_mode::none : v == "size" ? db::verify_mode::size : v == "blake3" ? db::verify_mode::blake3 : db::verify_mode::sha256; }
        else if (a == "--lookups") nlook = std::stoull(next()); else if (a == "--prefixes") npre = std::stoull(next()); else if (a == "--queries") nq = std::stoull(next());
        else if (a == "--single-queries") nsq = std::stoull(next());
        else if (a == "--sections") { sections.clear(); std::stringstream ss(next()); std::string t; while (std::getline(ss, t, ',')) sections.insert(t); }
        else if (a == "--threads") { threads.clear(); std::stringstream ss(next()); std::string t; while (std::getline(ss, t, ',')) threads.push_back(unsigned(std::stoul(t))); }
    }
    fs::create_directories(dir);
    const fs::path sdir = dir / "sharded", single = dir / "single.db";
    const ann::hnsw_params gp{.M = 16, .ef_construction = 100, .ef_search = 64, .seed = 1};   // the same parameters for the per-shard graphs and the global one
    std::printf("# bench_db: %llu rows, budget %llu MiB, dim %zu, dir %s, hnsw M=%zu efConstruction=%zu\n", (unsigned long long)rows, (unsigned long long)budget_mib, kDim, dir.string().c_str(), gp.M, gp.ef_construction);

    // ---- build ---------------------------------------------------------------------------------------------
    const unsigned build_threads = 8;
    task::pool bp(build_threads);
    auto sc = db::shard_schema{}; sc.ddl = kDdl; sc.table = "t"; sc.columns = {"k", "id", "body", "vec"}; sc.fts_table = "fts";
    sc.finalize = [af = ann::ann_finalize<fvec, ann::l2>("SELECT id, vec FROM t ORDER BY k", gp, {})](db::connection& c) {
        c.exec("INSERT INTO fts(rowid, body) SELECT id, body FROM t ORDER BY k"); af(c);
    };
    auto t0 = clk::now();
    if (!reuse || !fs::exists(sdir / "manifest.json")) {
        fs::remove_all(sdir);
        db::shard_writer w(sdir, sc, {.budget = budget_mib << 20, .presorted = true, .wave_bytes = 128ull << 20}, &bp);
        std::string body; fvec vec;
        for (std::uint64_t i = 0; i < rows; ++i) { gen_row(i, body, vec); const std::int64_t id = std::int64_t(i + 1); w.add({db::int_key(id), id, body, bytes_of(vec)}); }
        w.finish();
        std::printf("build sharded: %.1f s (%u threads), %.1f MiB\n", since(t0), build_threads, double(dir_bytes(sdir)) / (1 << 20));
    }
    { auto set0 = db::open(sdir); std::printf("sharded set: %zu shards, %.1f MiB, term filters %s\n", set0.size(), double(dir_bytes(sdir)) / (1 << 20), set0.has_bloom() ? "yes" : "no"); }
    if (do_single && (!reuse || !fs::exists(single))) {
        fs::remove(single);
        t0 = clk::now();
        {
            auto c = db::connection::open(single.string(), {.synchronous = "OFF"}); c.exec(kDdl);
            std::string body; fvec vec;
            { db::transaction tx(c); auto ins = c.prepare("INSERT INTO t(k, id, body, vec) VALUES(?,?,?,?)");
              for (std::uint64_t i = 0; i < rows; ++i) { gen_row(i, body, vec); const std::int64_t id = std::int64_t(i + 1); const auto k = db::int_key(id); ins.run(db::static_blob{db::static_span_(k)}, id, body, bytes_of(vec)); }
              c.exec("INSERT INTO fts(rowid, body) SELECT id, body FROM t ORDER BY k"); tx.commit(); }
            const double t_rows = since(t0);
            // one global graph, stored in the file (what an unsharded database would keep)
            ann::hnsw<fvec, ann::l2> h(gp);
            auto st = c.prepare("SELECT id, vec FROM t ORDER BY k"); for (auto& r : st.rows()) h.add(std::uint64_t(r.get<std::int64_t>(0)), ann::codec<fvec>::decode(r.get<std::span<const std::byte>>(1)));
            const auto tg = clk::now(); h.build(&bp); const double t_graph = since(tg); h.save(c, "ann");
            c.exec("PRAGMA journal_mode=DELETE"); c.exec("VACUUM");
            std::printf("build single: rows+fts %.1f s (serial), global graph %.1f s (%u threads), total %.1f s, 1 file, %.1f MiB\n", t_rows, t_graph, build_threads, since(t0), double(fs::file_size(single)) / (1 << 20));
        }
    }
    if (build_only) return 0;
    if (!fs::exists(single)) do_single = false;

    // ---- query sets ----------------------------------------------------------------------------------------
    std::mt19937_64 g(77);
    std::vector<std::int64_t> look(nlook); for (auto& x : look) x = std::int64_t(g() % rows) + 1;
    std::vector<std::string> prefixes(npre); for (auto& p : prefixes) p = db::int_key(std::int64_t(g() % rows) + 1).substr(0, 7);   // ~256 consecutive ids
    const std::uint64_t nclusters = std::max<std::uint64_t>(1, rows / 40000), nuniq = std::max<std::uint64_t>(1, rows / 200000);
    struct qclass { const char* name; const char* op; std::vector<std::vector<std::string>> qs; };
    std::vector<qclass> classes;
    { qclass c{"frequent words (2-3 terms, OR)", "OR", {}}; for (std::size_t i = 0; i < nq; ++i) { std::vector<std::string> q; const int n = 2 + int(g() % 2); for (int j = 0; j < n; ++j) q.push_back(word(g() % 4000)); c.qs.push_back(q); } classes.push_back(c); }
    { qclass c{"clustered rare term (1 term)", "OR", {}}; for (std::size_t i = 0; i < nq; ++i) c.qs.push_back({"c" + std::to_string(g() % nclusters)}); classes.push_back(c); }
    { qclass c{"rare + frequent (OR)", "OR", {}}; for (std::size_t i = 0; i < nq; ++i) c.qs.push_back({"c" + std::to_string(g() % nclusters), word(g() % 4000), word(g() % 4000)}); classes.push_back(c); }
    { qclass c{"rare AND frequent", "AND", {}}; for (std::size_t i = 0; i < nq; ++i) c.qs.push_back({"c" + std::to_string(g() % nclusters), word(g() % 400)}); classes.push_back(c); }
    { qclass c{"unique term + frequent (OR)", "OR", {}}; for (std::size_t i = 0; i < nq; ++i) c.qs.push_back({"u" + std::to_string(g() % nuniq), word(g() % 4000)}); classes.push_back(c); }
    { qclass c{"phrase (2 adjacent words) OR frequent", "OR", {}};
      for (std::size_t i = 0; i < nq; ++i) { std::string body; fvec v; gen_row(g() % rows, body, v); std::stringstream ss(body); std::string a, b; ss >> a >> b; c.qs.push_back({a + " " + b, word(g() % 4000)}); } classes.push_back(c); }
    std::vector<fvec> qv(nq); { std::normal_distribution<float> nd; for (auto& v : qv) { v.resize(kDim); for (auto& f : v) f = nd(g); } }
    auto blob = [](const std::string& s) { return db::static_span_(s); };
    std::printf("\n");

    auto layout = [&] {
        std::printf("## layout: sharded vs one file\n\n| op | layout | cache | threads | time | per op |\n|---|---|---|---|---|---|\n");
        auto row = [&](const char* op, const char* layout, const char* cache, unsigned th, double s, double nops) { std::printf("| %s | %s | %s | %u | %.3f s | %.1f us |\n", op, layout, cache, th, s, s / nops * 1e6); std::fflush(stdout); };
        std::uint64_t chk_sharded = 0, chk_single = 0;
        for (unsigned th : threads) {
            task::pool tp(th);
            for (int cold = 1; cold >= 0; --cold) {
                const char* cn = cold ? "cold" : "warm";
                if (cold) evict(sdir);
                t0 = clk::now();
                auto set = std::make_unique<db::shard_set>(db::open(sdir, {.verify = vm}));
                if (cold) row("open", "sharded", cn, th, since(t0), 1);
                t0 = clk::now(); std::vector<std::int64_t> sums(nlook);
                task::parallel_blocks(&tp, nlook, 256, [&](std::size_t, std::size_t b, std::size_t e) { for (std::size_t i = b; i < e; ++i) {
                    const auto key = db::int_key(look[i]); auto idx = set->shards_for_key(key);
                    auto l = set->acquire(idx.at(0)); auto st = l->prepare("SELECT id FROM t WHERE k = ?1"); st.bind(1, blob(key)); sums[i] = st.step() ? st.get<std::int64_t>(0) : -1; } });
                row("point lookup", "sharded", cn, th, since(t0), double(nlook));
                for (std::size_t i = 0; i < nlook; ++i) if (sums[i] != look[i]) { std::fprintf(stderr, "lookup mismatch\n"); std::exit(1); }
                t0 = clk::now(); std::uint64_t nrows = 0;
                for (auto& p : prefixes) { auto idx = set->shards_for_prefix(p); auto e = db::shard_set::prefix_end(p);
                    auto v = db::shard_query<std::int64_t>(*set, &tp, idx, "SELECT id FROM t WHERE k >= :lo AND k < :hi", [&](db::statement& st) { st.bind(":lo", blob(p)).bind(":hi", blob(e)); }, [](const db::row& r) { return r.get<std::int64_t>(0); }); nrows += v.size(); }
                row("prefix scan (~256 keys)", "sharded", cn, th, since(t0), double(npre)); chk_sharded = nrows;
                t0 = clk::now(); std::vector<std::vector<db::shard_hit>> br;
                for (auto& q : classes[0].qs) br.push_back(db::shard_bm25_topk(*set, &tp, "fts", q, 10, {}));
                row("bm25 top-10 (frequent words)", "sharded", cn, th, since(t0), double(classes[0].qs.size()));
                t0 = clk::now();
                auto parts = db::shard_map<std::int64_t>(*set, &tp, set->all_shards(), [](db::connection& c, std::size_t) { return c.scalar<std::int64_t>("SELECT coalesce(sum(length(body)), 0) FROM t"); });
                std::int64_t total = 0; for (auto x : parts) total += x;
                row("scan+reduce sum(length(body))", "sharded", cn, th, since(t0), 1);
                if (!do_single) continue;
                if (cold) evict(single);
                t0 = clk::now(); db::open_options ro; ro.mmap_size = std::int64_t(fs::file_size(single)); ro.cache_size_kib = 1024;
                db::read_pool rp(single.string(), th, ro);
                if (cold) row("open", "single", cn, th, since(t0), 1);
                t0 = clk::now();
                task::parallel_blocks(&tp, nlook, 256, [&](std::size_t, std::size_t b, std::size_t e) { auto l = rp.acquire(); auto st = l->prepare("SELECT id FROM t WHERE k = ?1");
                    for (std::size_t i = b; i < e; ++i) { const auto key = db::int_key(look[i]); st.reset(); st.bind(1, blob(key)); sums[i] = st.step() ? st.get<std::int64_t>(0) : -1; } });
                row("point lookup", "single", cn, th, since(t0), double(nlook));
                t0 = clk::now(); nrows = 0;
                { auto l = rp.acquire(); auto st = l->prepare("SELECT id FROM t WHERE k >= ?1 AND k < ?2");
                  for (auto& p : prefixes) { auto e = db::shard_set::prefix_end(p); st.reset(); st.bind(1, blob(p)).bind(2, blob(e)); while (st.step()) ++nrows; } }
                row("prefix scan (~256 keys)", "single", cn, th, since(t0), double(npre)); chk_single = nrows;
                t0 = clk::now(); std::vector<std::vector<db::scored>> sr;
                auto bparts = db::rowid_partitions(rp, "fts", 64);
                const std::size_t ns = std::min(nsq, classes[0].qs.size());
                for (std::size_t i = 0; i < ns; ++i) { std::string s; for (auto& t : classes[0].qs[i]) s += t + " "; sr.push_back(db::parallel_bm25_topk(rp, &tp, "fts", db::fts5::query_from_text(s), 10, bparts)); }
                row("bm25 top-10 (frequent words)", "single", cn, th, since(t0), double(ns));
                for (std::size_t i = 0; i < ns; ++i) { if (sr[i].size() != br[i].size()) { std::fprintf(stderr, "bm25 size mismatch q%zu\n", i); std::exit(1); }
                    for (std::size_t j = 0; j < sr[i].size(); ++j) if (sr[i][j].id != br[i][j].id || sr[i][j].score != br[i][j].score) { std::fprintf(stderr, "bm25 mismatch q%zu\n", i); std::exit(1); } }
                t0 = clk::now();
                std::vector<std::pair<std::string, std::string>> kparts;
                for (std::int64_t i = 0; i < 64; ++i) kparts.push_back({db::int_key(std::int64_t(rows) * i / 64 + 1), db::int_key(std::int64_t(rows) * (i + 1) / 64 + 1)});
                auto sp = db::parallel_map<std::int64_t>(rp, &tp, kparts, [](db::connection& c, const std::pair<std::string, std::string>& r) {
                    auto st = c.prepare("SELECT coalesce(sum(length(body)), 0) FROM t WHERE k >= ?1 AND k < ?2"); st.bind(1, db::static_span_(r.first)).bind(2, db::static_span_(r.second)); st.step(); return st.get<std::int64_t>(0); });
                std::int64_t total2 = 0; for (auto x : sp) total2 += x;
                row("scan+reduce sum(length(body))", "single", cn, th, since(t0), 1);
                if (total2 != total) { std::fprintf(stderr, "scan mismatch %lld vs %lld\n", (long long)total, (long long)total2); std::exit(1); }
            }
        }
        if (do_single) std::printf("\nprefix-scan rows: sharded %llu, single %llu (equal: %s); BM25 results of both layouts compared (ids and exact scores) on the first %zu queries\n\n", (unsigned long long)chk_sharded, (unsigned long long)chk_single, chk_sharded == chk_single ? "yes" : "NO", std::min(nsq, classes[0].qs.size()));
    };

    auto bm25 = [&] {
        std::printf("## bm25 query classes (8 threads, warm, %zu queries per class, top-10)\n\n", nq);
        std::printf("| class | variant | ms/query | shards counted (opened for df) | shards with a term | shards scored | skipped by bound | equal to unsharded |\n|---|---|---|---|---|---|---|---|\n");
        task::pool tp(8);
        auto set = db::open(sdir, {.verify = vm});
        set.verify_all(&tp);
        db::open_options ro; ro.mmap_size = std::int64_t(do_single ? fs::file_size(single) : 0); ro.cache_size_kib = 1024;
        std::unique_ptr<db::read_pool> rp; if (do_single) rp = std::make_unique<db::read_pool>(single.string(), 8, ro);
        for (auto& qc : classes) {
            struct var { const char* name; db::bm25_options bo; };
            const var vars[] = {{"term filters + bound pruning (default)", {.op = qc.op}}, {"no term filters", {.op = qc.op, .bloom = false}}, {"no filters, no pruning", {.op = qc.op, .prune = false, .bloom = false}}};
            // unsharded reference ids and scores (bm25() of the one table), on up to nsq queries
            std::vector<std::vector<db::shard_hit>> ref;
            if (do_single) for (std::size_t i = 0; i < std::min(nsq, qc.qs.size()); ++i) {
                auto l = rp->acquire(); auto st = l->prepare("SELECT rowid, -bm25(fts) AS s FROM fts WHERE fts MATCH ?1 ORDER BY s DESC, rowid ASC LIMIT 10"); st.bind(1, expr_of(qc.qs[i], qc.op));
                std::vector<db::shard_hit> v; for (auto& r : st.rows()) v.push_back({0, r.get<std::int64_t>(0), r.get<double>(1)}); ref.push_back(std::move(v)); }
            for (auto& v : vars) {
                for (auto& q : qc.qs) (void)db::shard_bm25_topk(set, &tp, "fts", q, 10, v.bo);   // one warm-up pass
                t0 = clk::now(); double counted = 0, with = 0, scored = 0, skipped = 0; bool equal = true; std::size_t qi = 0;
                for (auto& q : qc.qs) {
                    db::bm25_report rep; auto h = db::shard_bm25_topk(set, &tp, "fts", q, 10, v.bo, &rep);
                    counted += double(rep.counted); with += double(rep.with_hits); scored += double(rep.scored); skipped += double(rep.skipped_bound);
                    if (qi < ref.size()) { bool same = h.size() == ref[qi].size(); for (std::size_t j = 0; same && j < h.size(); ++j) same = h[j].id == ref[qi][j].id && h[j].score == ref[qi][j].score; equal = equal && same; }
                    ++qi;
                }
                const double s = since(t0), n = double(qc.qs.size());
                std::printf("| %s | %s | %.2f | %.1f of %zu | %.1f | %.1f | %.1f | %s |\n", qc.name, v.name, s / n * 1e3, counted / n, set.size(), with / n, scored / n, skipped / n, ref.empty() ? "(no single file)" : equal ? "yes" : "NO");
                std::fflush(stdout);
                if (!equal) { std::fprintf(stderr, "bm25 mismatch in class %s variant %s\n", qc.name, v.name); std::exit(1); }
            }
            if (do_single) {   // the unsharded table, partitioned by rowid, 8 threads: time for the same class
                auto bparts = db::rowid_partitions(*rp, "fts", 64); t0 = clk::now(); std::size_t n = std::min(nsq, qc.qs.size());
                for (std::size_t i = 0; i < n; ++i) (void)db::parallel_bm25_topk(*rp, &tp, "fts", expr_of(qc.qs[i], qc.op), 10, bparts);
                std::printf("| %s | one file, rowid partitions (%zu queries) | %.2f | | | | | reference |\n", qc.name, n, since(t0) / double(n) * 1e3);
            }
        }
        std::printf("\n");
    };

    auto io = [&] {
        std::printf("## I/O: prefetch kinds and schedules, cold pass (fresh handles after DONTNEED), then warm\n\n| op | prefetch | schedule | threads | cold | warm |\n|---|---|---|---|---|---|\n");
        struct sched { const char* name; db::shard_schedule s; };
        const sched scheds[] = {{"dynamic", db::shard_schedule::dynamic}, {"blocked", db::shard_schedule::blocked}, {"numa", db::shard_schedule::numa}};
        for (unsigned th : threads) {
            if (th < 4) continue;
            task::pool tp(th);
            for (auto kind : {db::prefetch_kind::none, db::prefetch_kind::fadvise, db::prefetch_kind::aio})
                for (auto& sc2 : scheds) {
                    if (sc2.s != db::shard_schedule::dynamic && kind != db::prefetch_kind::aio) continue;   // schedules compared with one prefetch kind
                    for (int op = 0; op < 2; ++op) {
                        evict(sdir);
                        auto set = db::open(sdir, {.verify = db::verify_mode::none});
                        auto run = [&] {
                            const auto t = clk::now();
                            if (op == 0) { for (std::size_t i = 0; i < 20; ++i) (void)db::shard_bm25_topk(set, &tp, "fts", classes[0].qs[i], 10, {}, nullptr, {.prefetch = kind != db::prefetch_kind::none, .schedule = sc2.s, .kind = kind}); }
                            else { (void)db::shard_map<std::int64_t>(set, &tp, set.all_shards(), [](db::connection& c, std::size_t) { return c.scalar<std::int64_t>("SELECT coalesce(sum(length(body)), 0) FROM t"); },
                                                                      {.prefetch = kind != db::prefetch_kind::none, .schedule = sc2.s, .kind = kind}); }
                            return since(t);
                        };
                        const double cold = run(), warm = run();
                        std::printf("| %s | %s | %s | %u | %.3f s | %.3f s |\n", op == 0 ? "bm25 top-10 x 20 (frequent)" : "full scan", db::prefetch_kind_name(kind), sc2.name, th, cold, warm); std::fflush(stdout);
                    }
                }
        }
        std::printf("\nNUMA nodes on this machine: %zu\n\n", db::system_node_map().count());
    };

    auto annsec = [&] {
        std::printf("## ann: recall@10 against brute force and time, per-shard graphs vs one global graph (same M, efConstruction)\n\n| index | ef | recall@10 | ms/query (8 threads, warm; per-shard: all graphs loaded) |\n|---|---|---|---|\n");
        std::vector<std::vector<ann::hit>> exact_nn;
        { task::pool tp(8); ann::flat_index<fvec, ann::l2> flat; fvec v; std::string body;
          for (std::uint64_t r = 0; r < rows; ++r) { gen_row(r, body, v); flat.add(r + 1, v); }
          for (auto& q : qv) exact_nn.push_back(flat.search(q, 10, &tp)); }
        auto recall = [&](const std::vector<std::vector<ann::hit>>& got) { std::size_t ok = 0; for (std::size_t r = 0; r < nq; ++r) for (auto& h : got[r]) for (auto& e : exact_nn[r]) if (e.id == h.id) { ++ok; break; } return double(ok) / double(nq * 10); };
        task::pool tp(8);
        auto set = db::open(sdir, {.verify = vm}); ann::shard_ann<fvec, ann::l2> sa(set);
        (void)sa.search(qv[0], 10, &tp);   // loads every graph
        for (std::size_t ef : {16u, 32u, 64u, 128u, 256u}) {
            t0 = clk::now(); std::vector<std::vector<ann::hit>> ah; for (auto& v : qv) ah.push_back(sa.search(v, 10, &tp, ef));
            std::printf("| per-shard graphs (%zu) | %zu | %.3f | %.2f |\n", set.size(), ef, recall(ah), since(t0) / double(nq) * 1e3); std::fflush(stdout);
        }
        if (do_single) {
            t0 = clk::now(); ann::hnsw<fvec, ann::l2> gh; { auto c = db::connection::open(single.string(), {.read_only = true}); gh = ann::hnsw<fvec, ann::l2>::load(c, "ann"); }
            std::printf("| one global graph: load | | | %.1f s |\n", since(t0));
            for (std::size_t ef : {16u, 32u, 64u, 128u, 256u, 512u, 1024u}) {
                t0 = clk::now(); std::vector<std::vector<ann::hit>> gr; for (auto& v : qv) gr.push_back(gh.search(v, 10, ef));
                std::printf("| one global graph (1 thread: a graph search is serial) | %zu | %.3f | %.2f |\n", ef, recall(gr), since(t0) / double(nq) * 1e3); std::fflush(stdout);
            }
        }
        std::printf("\n");
    };

    if (sections.count("layout")) layout();
    if (sections.count("bm25")) bm25();
    if (sections.count("io")) io();
    if (sections.count("ann")) annsec();
    return 0;
}
