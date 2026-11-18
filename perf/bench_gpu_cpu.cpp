// perf/bench_gpu_cpu: CPU with one shard per pinned core against the GPU, on a forge::db shard set built by perf/bench_db (3.6M rows,
// 32-d float vectors, FTS5 bodies, one HNSW graph per shard). Not a test; build with -DFORGE_BUILD_DB_BENCH=ON. docs/bench_gpu_vs_cpu.md.
//
//   bench_gpu_cpu --dir /dev/shm/gpubench/db64/sharded [--sections db,ann] [--reps 5] [--cpus 0-63] [--device cuda]
//
// Execution models compared:
//   pinned   one worker thread per shard, worker w pinned (pthread_setaffinity_np) to CPU cpus[w] and owning shard w for the whole run
//            (its connection, prepared statements, HNSW graph and flat index live on that worker); a query fans out to every worker
//            and the caller merges. With more shards than CPUs, worker w owns shards w, w + W, ...
//   1 thread the same code with one worker (pinned to cpus[0]) owning every shard.
//   library  the FORGE function as shipped (db::shard_bm25_topk on a task::pool of the same size; unpinned): the reference whose
//            results the pinned harness must reproduce exactly.
//   GPU      ann::gpu_flat_l2 as shipped (a device matmul through graph::gpu_registry, the rows cross the bus on every call) and
//            ann::gpu_knn_l2 (new: rows resident on the device, top-16 selected there), over all rows at once and per shard.
// db: BM25 top-10 (global IDF; frequent-word OR queries and single clustered rare terms) and point lookups by key. There is no GPU
// path for either in FORGE; none is timed as one.
// ann: recall@10 against the exact answer, throughput and latency at batch sizes 1, 64 and 1024; host-to-device upload timed apart.
// Every configuration runs --reps times; each line reports the median and the min..max over the reps, with the 1-minute load average
// and MemAvailable sampled before and after the configuration (the machine is shared). "cold" = a fresh shard_set handle (new
// connections, statements, graphs): on tmpfs (/dev/shm) nothing is read from a disk, so it is not a cold page cache.
#include <forge/ann/ann.hpp>
#include <forge/ann/gpu_knn.hpp>
#include <forge/db/shards.hpp>
#include <forge/graph/ann_gpu.hpp>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <thread>
using namespace forge;
namespace fs = std::filesystem;
using fvec = std::vector<float>;
using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

// ---- machine state -----------------------------------------------------------------------------------------------
static std::string machine() {
    double l1 = 0; { std::ifstream f("/proc/loadavg"); f >> l1; }
    long mem = 0; { std::ifstream f("/proc/meminfo"); std::string k; long v; std::string u; while (f >> k >> v >> u) if (k == "MemAvailable:") { mem = v; break; } }
    char b[64]; std::snprintf(b, sizeof b, "load %.0f, avail %.1f GiB", l1, double(mem) / 1048576.0); return b;
}
static long rss_mib() { std::ifstream f("/proc/self/status"); std::string k; long v; std::string u; while (f >> k) { if (k == "VmRSS:") { f >> v; return v / 1024; } std::getline(f, u); } return 0; }

// ---- pinned worker group -----------------------------------------------------------------------------------------
class pinned_group {
public:
    explicit pinned_group(std::vector<int> cpus) : cpus_(std::move(cpus)), ok_(cpus_.size(), 0) {
        for (std::size_t i = 0; i < cpus_.size(); ++i) th_.emplace_back([this, i] {
            cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpus_[i], &s);
            ok_[i] = pthread_setaffinity_np(pthread_self(), sizeof s, &s) == 0 && sched_getcpu() == cpus_[i];
            loop(i); });
        run([](std::size_t) {});   // every worker has pinned itself
    }
    ~pinned_group() { stop_ = true; gen_.fetch_add(1); gen_.notify_all(); for (auto& t : th_) t.join(); }
    std::size_t size() const { return cpus_.size(); }
    std::size_t pinned() const { std::size_t n = 0; for (char c : ok_) n += c ? 1 : 0; return n; }
    // f(worker) on every worker at once; returns when all are done
    void run(const std::function<void(std::size_t)>& f) {
        job_ = &f; left_.store(cpus_.size()); gen_.fetch_add(1); gen_.notify_all();
        for (;;) { auto l = left_.load(); if (!l) break; left_.wait(l); }
    }
private:
    void loop(std::size_t i) {
        std::uint64_t seen = 0;
        for (;;) {
            gen_.wait(seen); seen = gen_.load();
            if (stop_) return;
            (*job_)(i);
            if (left_.fetch_sub(1) == 1) left_.notify_all();
        }
    }
    std::vector<int> cpus_; std::vector<char> ok_; std::vector<std::thread> th_;
    const std::function<void(std::size_t)>* job_ = nullptr;
    std::atomic<std::uint64_t> gen_{0}; std::atomic<std::size_t> left_{0}; std::atomic<bool> stop_{false};
};

// ---- stats -------------------------------------------------------------------------------------------------------
static double pct(std::vector<double> v, double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); const double r = p * double(v.size() - 1); const std::size_t i = std::size_t(r); return i + 1 < v.size() ? v[i] + (r - double(i)) * (v[i + 1] - v[i]) : v[i]; }
struct rep_stats { double qps, p50, p99; };                 // queries per second, latency percentiles in ms
static rep_stats stats_of(const std::vector<double>& lat_s, double wall_s, double nqueries) { std::vector<double> ms; for (double x : lat_s) ms.push_back(x * 1e3); return {nqueries / wall_s, pct(ms, 0.5), pct(ms, 0.99)}; }
static std::string med_range(std::vector<double> v, const char* fmt) {
    std::sort(v.begin(), v.end()); char a[48], b[48], c[48];
    std::snprintf(a, sizeof a, fmt, pct(v, 0.5)); std::snprintf(b, sizeof b, fmt, v.front()); std::snprintf(c, sizeof c, fmt, v.back());
    return std::string(a) + " (" + b + ".." + c + ")";
}
static void print_row(const std::string& head, const std::vector<rep_stats>& reps, const std::string& m0, const std::string& m1, const std::string& extra = "") {
    std::vector<double> q, a, b; for (auto& r : reps) { q.push_back(r.qps); a.push_back(r.p50); b.push_back(r.p99); }
    std::printf("| %s | %s | %s | %s | %zu | %s / %s |%s\n", head.c_str(), med_range(q, "%.1f").c_str(), med_range(a, "%.3f").c_str(), med_range(b, "%.3f").c_str(), reps.size(), m0.c_str(), m1.c_str(), extra.c_str());
    std::fflush(stdout);
}

// ---- query generators (perf/bench_db's vocabulary) ---------------------------------------------------------------
static std::string word(std::uint64_t r) { return "w" + std::to_string(r * r / 40000); }

// ---- BM25 over a shard set on a pinned group: db::shard_bm25_topk's algorithm (global statistics, term filters, bound pruning),
// with shard w's work always on worker w % W. Results must equal the library's (checked).
struct bm25_state {
    db::shard_set* s; std::size_t W;
    std::vector<std::optional<db::shard_set::lease>> leases;   // a worker's own connection per shard, kept across queries
    std::vector<std::map<std::string, std::unique_ptr<db::statement>>> stmts;
    bool have_df_checked = false; std::vector<char> have_df;
    explicit bm25_state(db::shard_set& set, std::size_t w) : s(&set), W(w), leases(set.size()), stmts(set.size()), have_df(set.size(), 0) {}
    db::connection& conn(std::size_t sh) { if (!leases[sh]) { leases[sh].emplace(s->acquire(sh)); have_df[sh] = (*leases[sh])->table_exists("forge_df") ? 1 : 0; } return **leases[sh]; }
    db::statement& stmt(std::size_t sh, const std::string& sql) { auto& m = stmts[sh]; auto it = m.find(sql); if (it == m.end()) it = m.emplace(sql, std::make_unique<db::statement>(conn(sh).prepare(sql))).first; return *it->second; }
};
static std::vector<db::shard_hit> pinned_bm25(pinned_group& pg, bm25_state& st, const std::vector<std::string>& terms_in, std::size_t k, const std::string& op) {
    db::shard_set& s = *st.s;
    std::vector<std::string> terms; for (auto& t : terms_in) if (!t.empty() && std::find(terms.begin(), terms.end(), t) == terms.end()) terms.push_back(t);
    const std::size_t ns = s.size(), nt = terms.size();
    std::string expr; for (std::size_t i = 0; i < nt; ++i) { if (i) expr += " " + op + " "; expr += db::fts5::quote_phrase(terms[i]); }
    struct counts { std::int64_t rows = 0, tokens = 0; std::vector<std::int64_t> df; };
    std::vector<counts> cs(ns); std::vector<std::vector<char>> maybe(ns); std::vector<char> open(ns, 0);
    const bool use_bloom = s.has_bloom();
    for (std::size_t i = 0; i < ns; ++i) {
        const auto& si = s.info(i); cs[i].df.assign(nt, 0); maybe[i].assign(nt, 1); bool o = true;
        if (use_bloom && si.fts_rows >= 0) { cs[i].rows = si.fts_rows; cs[i].tokens = si.fts_tokens; o = false; for (std::size_t t = 0; t < nt; ++t) { maybe[i][t] = s.may_contain(i, terms[t]) ? 1 : 0; if (maybe[i][t]) o = true; } }
        open[i] = o ? 1 : 0;
    }
    const std::string ft = db::quote_ident("fts");
    pg.run([&](std::size_t w) {   // phase 1: per-shard document counts, on the owning worker
        for (std::size_t i = w; i < ns; i += pg.size()) {
            if (!open[i]) continue;
            const auto& si = s.info(i); counts& r = cs[i];
            r.df.assign(nt, 0);
            if (si.fts_rows >= 0) { r.rows = si.fts_rows; r.tokens = si.fts_tokens; } else { auto t = db::fts_totals(st.conn(i), "fts"); r.rows = t.first; r.tokens = t.second; }
            if (r.rows == 0) continue;
            st.conn(i);
            for (std::size_t t = 0; t < nt; ++t) {
                if (!maybe[i][t]) continue;
                const auto& wd = terms[t];
                const bool plain = st.have_df[i] && std::all_of(wd.begin(), wd.end(), [](unsigned char ch) { return std::isalnum(ch) && ch < 0x80; });
                if (plain) { std::string lw = wd; for (auto& ch : lw) ch = char(std::tolower(static_cast<unsigned char>(ch)));
                    auto& q = st.stmt(i, "SELECT n FROM forge_df WHERE term = ?1"); q.reset(); q.bind(1, lw); r.df[t] = q.step() ? q.get<std::int64_t>(0) : 0; }
                else { auto& q = st.stmt(i, "SELECT count(*) FROM " + ft + " WHERE " + ft + " MATCH ?1"); q.reset(); q.bind(1, db::fts5::quote_phrase(wd)); q.step(); r.df[t] = q.get<std::int64_t>(0); }
            }
        }
    });
    std::int64_t N = 0, T = 0; std::vector<std::int64_t> df(nt, 0);
    for (auto& c : cs) { N += c.rows; T += c.tokens; for (std::size_t t = 0; t < nt; ++t) df[t] += c.df[t]; }
    if (N == 0) return {};
    const double avgdl = double(T) / double(N);
    std::vector<double> idf(nt);
    for (std::size_t t = 0; t < nt; ++t) { const double x = std::log((double(N - df[t]) + 0.5) / (double(df[t]) + 0.5)); idf[t] = x <= 0.0 ? 1e-6 : x; }
    std::string idf_blob(nt * sizeof(double), '\0'); std::memcpy(idf_blob.data(), idf.data(), idf_blob.size());
    std::vector<double> bound(ns, -1);
    for (std::size_t i = 0; i < ns; ++i) {
        double b = 0; std::size_t present = 0;
        for (std::size_t t = 0; t < nt; ++t) if (cs[i].df[t] > 0) { b += idf[t] * (1.2 + 1.0); ++present; }
        if (present == 0 || (op == "AND" && present < nt)) continue;
        bound[i] = b;
    }
    std::mutex mu; std::vector<db::shard_hit> heap;
    const std::string sql = "SELECT rowid, -forge_bm25(" + ft + ", ?1, ?2) FROM " + ft + " WHERE " + ft + " MATCH ?3";
    pg.run([&](std::size_t w) {   // phase 2: score the owned shards, best bound first, skipping those below the current k-th score
        std::vector<std::size_t> mine; for (std::size_t i = w; i < ns; i += pg.size()) if (bound[i] >= 0) mine.push_back(i);
        std::stable_sort(mine.begin(), mine.end(), [&](std::size_t a, std::size_t b) { return bound[a] > bound[b]; });
        for (std::size_t sh : mine) {
            { std::lock_guard g(mu); if (heap.size() >= k && bound[sh] < heap.front().score) continue; }
            std::vector<db::shard_hit> local;
            auto& q = st.stmt(sh, sql); q.reset();
            q.bind(1, avgdl).bind(2, std::span<const std::byte>(reinterpret_cast<const std::byte*>(idf_blob.data()), idf_blob.size())).bind(3, expr);
            while (q.step()) {
                db::shard_hit h{std::uint32_t(sh), q.get<std::int64_t>(0), q.get<double>(1)};
                if (local.size() < k) { local.push_back(h); std::push_heap(local.begin(), local.end(), db::shard_hit_before); }
                else if (db::shard_hit_before(h, local.front())) { std::pop_heap(local.begin(), local.end(), db::shard_hit_before); local.back() = h; std::push_heap(local.begin(), local.end(), db::shard_hit_before); }
            }
            q.reset();
            std::lock_guard g(mu);
            for (auto& h : local) {
                if (heap.size() < k) { heap.push_back(h); std::push_heap(heap.begin(), heap.end(), db::shard_hit_before); }
                else if (db::shard_hit_before(h, heap.front())) { std::pop_heap(heap.begin(), heap.end(), db::shard_hit_before); heap.back() = h; std::push_heap(heap.begin(), heap.end(), db::shard_hit_before); }
            }
        }
    });
    std::sort(heap.begin(), heap.end(), db::shard_hit_before);
    return heap;
}

// ---- main --------------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    fs::path dir; std::set<std::string> sections = {"db", "ann"}; int reps = 5; std::string cpu_list = "", device = "cuda";
    std::size_t nlook = 20000, nbm_freq = 20, nbm_rare = 50, nbm_1t_freq = 5, ann_nq = 1024, ann_single = 200;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; auto next = [&] { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--dir") dir = next(); else if (a == "--reps") reps = std::stoi(next()); else if (a == "--cpus") cpu_list = next(); else if (a == "--device") device = next();
        else if (a == "--lookups") nlook = std::stoull(next()); else if (a == "--bm25-frequent") nbm_freq = std::stoull(next()); else if (a == "--bm25-rare") nbm_rare = std::stoull(next());
        else if (a == "--bm25-frequent-1t") nbm_1t_freq = std::stoull(next()); else if (a == "--ann-queries") ann_nq = std::stoull(next()); else if (a == "--ann-single") ann_single = std::stoull(next());
        else if (a == "--sections") { sections.clear(); std::stringstream ss(next()); std::string t; while (std::getline(ss, t, ',')) sections.insert(t); }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (dir.empty()) { std::fprintf(stderr, "--dir <shard set> required\n"); return 2; }
    std::vector<int> cpus;
    if (cpu_list.empty()) { cpu_set_t s; sched_getaffinity(0, sizeof s, &s); for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &s)) cpus.push_back(c); }
    else { std::stringstream ss(cpu_list); std::string t; while (std::getline(ss, t, ',')) { auto d = t.find('-'); int a = std::stoi(t.substr(0, d)), b = d == std::string::npos ? a : std::stoi(t.substr(d + 1)); for (int c = a; c <= b; ++c) cpus.push_back(c); } }
    db::shard_open_options oo; oo.verify = db::verify_mode::none; oo.max_open_bytes = 64ull << 30;
    std::size_t nshards = db::open(dir, oo).size();
    const std::size_t W = std::min(nshards, cpus.size());
    std::vector<int> wcpus(cpus.begin(), cpus.begin() + std::ptrdiff_t(W));
    std::printf("# bench_gpu_cpu: %s, %zu shards, %zu CPUs available, pinned workers %zu (%s), reps %d, %s, rss %ld MiB\n\n", dir.string().c_str(), nshards, cpus.size(), W,
                W == nshards ? "one shard per core" : "several shards per core", reps, machine().c_str(), rss_mib());
    { auto s = db::open(dir, oo); s.verify_all(); std::printf("shard files verified against the manifest (sha256), %s\n\n", machine().c_str()); }

    std::mt19937_64 g(77);
    std::uint64_t rows = 0; { auto s = db::open(dir, oo); for (std::size_t i = 0; i < s.size(); ++i) rows += std::uint64_t(s.info(i).rows); }

    if (sections.count("db")) {
        std::vector<std::vector<std::string>> qfreq, qrare;
        for (std::size_t i = 0; i < std::max(nbm_freq, nbm_1t_freq); ++i) { std::vector<std::string> q; const int n = 2 + int(g() % 2); for (int j = 0; j < n; ++j) q.push_back(word(g() % 4000)); qfreq.push_back(q); }
        const std::uint64_t nclusters = std::max<std::uint64_t>(1, rows / 40000);
        for (std::size_t i = 0; i < nbm_rare; ++i) qrare.push_back({"c" + std::to_string(g() % nclusters)});
        std::vector<std::int64_t> look(nlook); for (auto& x : look) x = std::int64_t(g() % rows) + 1;

        // reference answers: the library function, unpinned pool
        std::vector<std::vector<db::shard_hit>> ref_f, ref_r;
        {
            task::pool tp{unsigned(W)}; auto s = db::open(dir, oo);
            for (auto& q : qfreq) ref_f.push_back(db::shard_bm25_topk(s, &tp, "fts", q, 10, {}));
            for (auto& q : qrare) ref_r.push_back(db::shard_bm25_topk(s, &tp, "fts", q, 10, {}));
        }
        std::printf("## db: BM25 top-10 and point lookups (no GPU path exists in FORGE for either)\n\n");
        std::printf("| workload | config | handle | queries/s | p50 ms | p99 ms | reps | machine before / after |\n|---|---|---|---|---|---|---|---|\n");
        auto same_hits = [](const std::vector<db::shard_hit>& a, const std::vector<db::shard_hit>& b) { if (a.size() != b.size()) return false; for (std::size_t i = 0; i < a.size(); ++i) if (a[i].id != b[i].id || a[i].score != b[i].score) return false; return true; };
        struct cfg { std::string name; std::vector<int> cpus; };
        std::vector<cfg> cfgs = {{"pinned, " + std::to_string(W) + " workers (1 shard/core)", wcpus}, {"1 thread (pinned)", {cpus[0]}}};
        for (auto& c : cfgs) {
            pinned_group pg(c.cpus);
            if (pg.pinned() != pg.size()) std::printf("WARNING: only %zu of %zu workers pinned\n", pg.pinned(), pg.size());
            const bool one = c.cpus.size() == 1;
            struct wl { const char* name; std::vector<std::vector<std::string>>* qs; std::vector<std::vector<db::shard_hit>>* ref; std::size_t n; };
            std::vector<wl> wls = {{"bm25 frequent words (2-3 terms, OR)", &qfreq, &ref_f, one ? nbm_1t_freq : nbm_freq}, {"bm25 clustered rare term", &qrare, &ref_r, nbm_rare}};
            for (auto& w : wls) {
                std::vector<rep_stats> cold, warm; const std::string m0 = machine(); bool equal = true;
                for (int r = 0; r < reps; ++r) {
                    auto set = db::open(dir, oo); (void)set.has_bloom();
                    bm25_state st(set, pg.size());
                    for (int pass = 0; pass < 2; ++pass) {
                        std::vector<double> lat; const auto t0 = clk::now();
                        for (std::size_t i = 0; i < w.n; ++i) { const auto t = clk::now(); auto h = pinned_bm25(pg, st, (*w.qs)[i], 10, "OR"); lat.push_back(since(t)); if (!same_hits(h, (*w.ref)[i])) equal = false; }
                        (pass == 0 ? cold : warm).push_back(stats_of(lat, since(t0), double(w.n)));
                    }
                    st.stmts.clear(); st.leases.clear();
                }
                const std::string m1 = machine(), eq = equal ? " results = library" : " RESULTS DIFFER FROM THE LIBRARY";
                print_row(std::string(w.name) + " (" + std::to_string(w.n) + " queries) | " + c.name + " | fresh", cold, m0, m1, eq);
                print_row(std::string(w.name) + " (" + std::to_string(w.n) + " queries) | " + c.name + " | warm", warm, m0, m1, eq);
                if (!equal) return 1;
            }
            {   // point lookups: each key goes to the worker owning its shard; that worker keeps one connection and statement per shard
                std::vector<rep_stats> cold, warm; const std::string m0 = machine(); bool ok = true;
                for (int r = 0; r < reps; ++r) {
                    auto set = db::open(dir, oo);
                    std::vector<std::vector<std::size_t>> mine(pg.size()); std::vector<std::size_t> shard_of(nlook);
                    std::vector<std::string> keys(nlook);
                    for (std::size_t i = 0; i < nlook; ++i) { keys[i] = db::int_key(look[i]); shard_of[i] = set.shards_for_key(keys[i]).at(0); mine[shard_of[i] % pg.size()].push_back(i); }
                    std::vector<std::optional<db::shard_set::lease>> leases(set.size()); std::vector<std::optional<db::statement>> st(set.size());
                    for (int pass = 0; pass < 2; ++pass) {
                        std::vector<double> lat(nlook); std::vector<std::int64_t> got(nlook, -1);
                        const auto t0 = clk::now();
                        pg.run([&](std::size_t w) {
                            for (std::size_t i : mine[w]) {
                                const auto t = clk::now(); const std::size_t sh = shard_of[i];
                                if (!st[sh]) { leases[sh].emplace(set.acquire(sh)); st[sh].emplace((*leases[sh])->prepare("SELECT id FROM t WHERE k = ?1")); }
                                auto& q = *st[sh]; q.reset(); q.bind(1, db::static_span_(keys[i])); got[i] = q.step() ? q.get<std::int64_t>(0) : -1; q.reset();
                                lat[i] = since(t);
                            }
                        });
                        const double wall = since(t0);
                        for (std::size_t i = 0; i < nlook; ++i) if (got[i] != look[i]) ok = false;
                        (pass == 0 ? cold : warm).push_back(stats_of(lat, wall, double(nlook)));
                    }
                    st.clear(); leases.clear();
                }
                const std::string m1 = machine(), eq = ok ? " all keys found" : " LOOKUP MISMATCH";
                print_row("point lookup by key (" + std::to_string(nlook) + " keys, latency = one lookup on its worker) | " + c.name + " | fresh", cold, m0, m1, eq);
                print_row("point lookup by key (" + std::to_string(nlook) + " keys, latency = one lookup on its worker) | " + c.name + " | warm", warm, m0, m1, eq);
                if (!ok) return 1;
            }
        }
        {   // the library as shipped, for reference: shard_bm25_topk on an unpinned task::pool of W threads (dynamic schedule)
            task::pool tp{unsigned(W)};
            for (int wi = 0; wi < 2; ++wi) {
                auto& qs = wi == 0 ? qfreq : qrare; const std::size_t n = wi == 0 ? nbm_freq : nbm_rare;
                std::vector<rep_stats> cold, warm; const std::string m0 = machine();
                for (int r = 0; r < reps; ++r) { auto set = db::open(dir, oo);
                    for (int pass = 0; pass < 2; ++pass) { std::vector<double> lat; const auto t0 = clk::now();
                        for (std::size_t i = 0; i < n; ++i) { const auto t = clk::now(); (void)db::shard_bm25_topk(set, &tp, "fts", qs[i], 10, {}); lat.push_back(since(t)); }
                        (pass == 0 ? cold : warm).push_back(stats_of(lat, since(t0), double(n))); } }
                const std::string m1 = machine(), nm = std::string(wi == 0 ? "bm25 frequent words (2-3 terms, OR)" : "bm25 clustered rare term") + " (" + std::to_string(n) + " queries) | library shard_bm25_topk, task::pool(" + std::to_string(W) + "), unpinned";
                print_row(nm + " | fresh", cold, m0, m1); print_row(nm + " | warm", warm, m0, m1);
            }
        }
        std::printf("\n");
    }

    if (sections.count("ann")) {
        constexpr std::size_t kDim = 32, K = 10;
        // the vectors, per shard, as stored (SELECT id, vec FROM t)
        std::vector<std::vector<std::uint64_t>> sid(nshards); std::vector<fvec> sdata(nshards);
        { auto set = db::open(dir, oo); task::pool tp{unsigned(W)};
          task::parallel_blocks(&tp, nshards, 1, [&](std::size_t i, std::size_t, std::size_t) {
              auto l = set.acquire(i); auto st = l->prepare("SELECT id, vec FROM t ORDER BY k");
              for (auto& r : st.rows()) { sid[i].push_back(std::uint64_t(r.get<std::int64_t>(0))); auto b = r.get<std::span<const std::byte>>(1); const std::size_t o = sdata[i].size(); sdata[i].resize(o + kDim); std::memcpy(sdata[i].data() + o, b.data(), kDim * 4); } }); }
        std::size_t nvec = 0; for (auto& v : sid) nvec += v.size();
        std::vector<fvec> qv(ann_nq); { std::normal_distribution<float> nd; std::mt19937_64 gq(4242); for (auto& v : qv) { v.resize(kDim); for (auto& f : v) f = nd(gq); } }
        std::printf("## ann: %zu vectors (dim %zu) in %zu shards, %zu queries, top-%zu; rss %ld MiB, %s\n\n", nvec, kDim, nshards, ann_nq, K, rss_mib(), machine().c_str());

        pinned_group pg(wcpus);
        // per-shard indexes, built and loaded on the owning worker
        std::vector<std::unique_ptr<ann::flat_index<fvec, ann::l2>>> flat(nshards);
        std::vector<std::unique_ptr<ann::hnsw<fvec, ann::l2>>> graph(nshards);
        { auto set = db::open(dir, oo); const auto t0 = clk::now();
          pg.run([&](std::size_t w) { for (std::size_t i = w; i < nshards; i += pg.size()) {
              flat[i] = std::make_unique<ann::flat_index<fvec, ann::l2>>();
              for (std::size_t j = 0; j < sid[i].size(); ++j) flat[i]->add(sid[i][j], fvec(sdata[i].begin() + std::ptrdiff_t(j * kDim), sdata[i].begin() + std::ptrdiff_t((j + 1) * kDim)));
              auto l = set.acquire(i); graph[i] = std::make_unique<ann::hnsw<fvec, ann::l2>>(ann::hnsw<fvec, ann::l2>::load(*l, "ann")); } });
          std::printf("per-shard flat indexes and HNSW graphs loaded on their workers: %.1f s, rss %ld MiB\n\n", since(t0), rss_mib()); }

        auto merge = [&](std::vector<std::vector<ann::hit>>& parts) { std::vector<ann::hit> all; for (auto& p : parts) all.insert(all.end(), p.begin(), p.end()); std::sort(all.begin(), all.end(), ann::hit_before); if (all.size() > K) all.resize(K); return all; };
        // pinned CPU search of a batch: worker w answers every query of the batch on its shards; the caller merges
        auto cpu_batch = [&](pinned_group& grp, std::span<const fvec> qs, std::size_t ef) {
            std::vector<std::vector<std::vector<ann::hit>>> per(qs.size(), std::vector<std::vector<ann::hit>>(nshards));
            grp.run([&](std::size_t w) { for (std::size_t i = w; i < nshards; i += grp.size()) for (std::size_t q = 0; q < qs.size(); ++q)
                per[q][i] = ef ? graph[i]->search(qs[q], K, ef) : flat[i]->search(qs[q], K, nullptr); });
            std::vector<std::vector<ann::hit>> out(qs.size()); for (std::size_t q = 0; q < qs.size(); ++q) out[q] = merge(per[q]);
            return out;
        };
        // exact answers: pinned flat over every shard (merge of exact per-shard top-10 = exact global top-10)
        std::vector<std::vector<ann::hit>> exact; { const auto t0 = clk::now(); exact = cpu_batch(pg, qv, 0); std::printf("exact answers (pinned flat, %zu queries): %.2f s\n\n", ann_nq, since(t0)); }
        auto recall = [&](const std::vector<std::vector<ann::hit>>& got, std::size_t q0) { std::size_t ok = 0; for (std::size_t r = 0; r < got.size(); ++r) for (auto& h : got[r]) for (auto& e : exact[q0 + r]) if (e.id == h.id) { ok++; break; } return double(ok) / double(got.size() * K); };
        auto same_as_exact = [&](const std::vector<std::vector<ann::hit>>& got, std::size_t q0) { std::size_t bad = 0; for (std::size_t r = 0; r < got.size(); ++r) if (got[r] != exact[q0 + r]) ++bad; return bad; };

        std::printf("| method | batch | queries/s | latency p50 ms | latency p99 ms | reps | machine before / after | recall@10 |\n|---|---|---|---|---|---|---|---|\n");
        // a method runs `batch` queries per call; latency = the time of one call (every query of the batch completes together)
        auto bench = [&](const std::string& name, std::size_t batch, const std::function<std::vector<std::vector<ann::hit>>(std::span<const fvec>)>& fn, std::size_t nq_total, bool want_exact, const std::string& extra = "") {
            nq_total = std::min(nq_total, qv.size()); nq_total -= nq_total % batch; if (!nq_total) nq_total = batch;
            std::vector<rep_stats> rs; const std::string m0 = machine(); double rec = 0; std::size_t bad = 0;
            (void)fn(std::span<const fvec>(qv.data(), batch));   // warm-up call
            for (int r = 0; r < reps; ++r) {
                std::vector<double> lat; std::vector<std::vector<ann::hit>> all; const auto t0 = clk::now();
                for (std::size_t q0 = 0; q0 + batch <= nq_total; q0 += batch) { const auto t = clk::now(); auto h = fn(std::span<const fvec>(qv.data() + q0, batch)); lat.push_back(since(t)); all.insert(all.end(), h.begin(), h.end()); }
                rs.push_back(stats_of(lat, since(t0), double(nq_total)));
                rec = recall(all, 0); bad = same_as_exact(all, 0);
            }
            char rb[96]; std::snprintf(rb, sizeof rb, " %.4f%s", rec, want_exact ? (bad ? " (NOT equal to exact)" : " (= exact, ids and distances)") : "");
            print_row(name + " | " + std::to_string(batch) + " (" + std::to_string(nq_total) + " queries)", rs, m0, machine(), rb + extra);
        };

        // ---- CPU, one shard per pinned core ----
        for (std::size_t b : {std::size_t(1), std::size_t(64), std::size_t(1024)})
            bench("CPU flat (exact), " + std::to_string(W) + " pinned workers", b, [&](std::span<const fvec> qs) { return cpu_batch(pg, qs, 0); }, b == 1 ? ann_single : ann_nq, true);
        std::size_t ef_match = 0;
        for (std::size_t ef : {16u, 32u, 64u, 128u, 256u, 512u}) {
            auto h = cpu_batch(pg, std::span<const fvec>(qv.data(), std::min<std::size_t>(256, qv.size())), ef);
            const double rc = recall(h, 0);
            std::printf("| CPU HNSW per shard, ef %zu (recall sweep, 256 queries) | | | | | | | %.4f |\n", ef, rc);
            if (!ef_match && rc >= 0.99) ef_match = ef;
        }
        if (!ef_match) ef_match = 512;
        for (std::size_t ef : {std::size_t(64), ef_match}) {
            for (std::size_t b : {std::size_t(1), std::size_t(64), std::size_t(1024)})
                bench("CPU HNSW per shard, ef " + std::to_string(ef) + ", " + std::to_string(W) + " pinned workers", b, [&](std::span<const fvec> qs) { return cpu_batch(pg, qs, ef); }, b == 1 ? ann_single : ann_nq, false);
            if (ef == ef_match) break;
        }
        {   // context: one worker for every shard
            pinned_group one({cpus[0]});
            for (std::size_t b : {std::size_t(1), std::size_t(64)})
                bench("CPU flat (exact), 1 thread", b, [&](std::span<const fvec> qs) { return cpu_batch(one, qs, 0); }, b == 1 ? 20 : 64, true);
            bench("CPU HNSW per shard, ef " + std::to_string(ef_match) + ", 1 thread", 1, [&](std::span<const fvec> qs) { return cpu_batch(one, qs, ef_match); }, ann_single, false);
        }

        // ---- GPU ----
        std::string why; auto rt = graph::open_knn_device(device, &why);
        if (!rt) { std::printf("\nno %s device (%s): GPU rows skipped\n", device.c_str(), why.c_str()); return 0; }
        std::printf("| GPU: %s (%s), image %s | | | | | | | |\n", device.c_str(), rt->device_name().c_str(), rt->image_arch().c_str());
        task::pool host{unsigned(W)};   // host-side work of the GPU paths (re-ranking, candidate filtering)
        {   // gpu_flat_l2 as shipped: matmul_t through the CUDA registry, non-resident (rows uploaded on every call)
            auto reg = graph::gpu_registry(rt, false);
            ann::gpu_flat_l2 ix(graph::ann_matmul(reg.get()), kDim);
            for (std::size_t i = 0; i < nshards; ++i) for (std::size_t j = 0; j < sid[i].size(); ++j) ix.add(sid[i][j], std::span<const float>(sdata[i].data() + j * kDim, kDim));
            for (std::size_t b : {std::size_t(1), std::size_t(64), std::size_t(1024)}) {
                const auto d0 = rt->dispatches(); ann::gpu_flat_l2::report rep{}; bool fb = false; std::size_t calls = 0;
                bench("GPU gpu_flat_l2 (as shipped), all " + std::to_string(nshards) + " shards in one index", b,
                      [&](std::span<const fvec> qs) { auto h = ix.search_batch(qs, K, &host, &rep); fb = fb || rep.fell_back; calls += rep.device_calls; return h; }, b == 1 ? 20 : b == 64 ? 256 : ann_nq, true);
                auto d1 = rt->dispatches(); const std::size_t mm = (d1.count("matmul_tiled") ? d1["matmul_tiled"] : 0) - (d0.count("matmul_tiled") ? d0.at("matmul_tiled") : 0);
                std::printf("|   (device matmul_t kernel launches during the row above: %zu; %zu device calls; fell back to CPU: %s; bytes over the bus per %zu-query block: rows %.0f MiB up, distances %.0f MiB down) | | | | | | | |\n",
                            mm, calls, fb ? "YES" : "no", std::min<std::size_t>(b, 128), double(nvec * kDim * 4) / 1048576.0, double(std::min<std::size_t>(b, 128) * nvec * 4) / 1048576.0);
            }
        }
        {   // gpu_knn_l2 (new): rows resident on the device, top-16 selected there, 16 candidates per query back to the host
            ann::gpu_knn_l2 ix(graph::cuda_knn(rt), kDim);
            for (std::size_t i = 0; i < nshards; ++i) for (std::size_t j = 0; j < sid[i].size(); ++j) ix.add(sid[i][j], std::span<const float>(sdata[i].data() + j * kDim, kDim));
            std::vector<double> ups; const std::string m0 = machine();
            for (int r = 0; r < reps; ++r) { rt->knn_forget(ix.data()); ann::gpu_knn_l2::report rep; (void)ix.search_batch(std::span<const fvec>(qv.data(), 1), K, &host, &rep); ups.push_back(rep.upload_s); }
            std::printf("| GPU gpu_knn_l2: host-to-device upload of all %zu rows (%.0f MiB), once per index | | %s ms | | %d | %s / %s | %.2f GB/s (median) |\n", nvec, double(nvec * kDim * 4) / 1048576.0,
                        med_range([&] { std::vector<double> v; for (double u : ups) v.push_back(u * 1e3); return v; }(), "%.1f").c_str(), reps, m0.c_str(), machine().c_str(), double(nvec * kDim * 4) / pct(ups, 0.5) / 1e9);
            const auto mem = rt->memory(); std::printf("|   (device memory in use after upload: %.2f GiB of %.2f GiB) | | | | | | | |\n", double(mem.total - mem.free) / 1073741824.0, double(mem.total) / 1073741824.0);
            for (std::size_t b : {std::size_t(1), std::size_t(64), std::size_t(1024)}) {
                const auto d0 = rt->dispatches(); std::size_t fbq = 0; bool fb = false;
                bench("GPU gpu_knn_l2 (new), all " + std::to_string(nshards) + " shards in one resident index", b,
                      [&](std::span<const fvec> qs) { ann::gpu_knn_l2::report rep; auto h = ix.search_batch(qs, K, &host, &rep); fbq += rep.fallback_queries; fb = fb || rep.fell_back; return h; }, b == 1 ? ann_single : ann_nq, true);
                auto d1 = rt->dispatches();
                std::printf("|   (forge_knn_l2_part launches: %zu; queries answered on the CPU by the exactness check: %zu; device call failed: %s) | | | | | | | |\n",
                            (d1.count("knn_l2_part") ? d1["knn_l2_part"] : 0) - (d0.count("knn_l2_part") ? d0.at("knn_l2_part") : 0), fbq, fb ? "YES" : "no");
            }
            for (std::size_t b : {std::size_t(1), std::size_t(64), std::size_t(1024)}) {   // the device call alone (query upload, 2 kernels, 16 results down), no host re-ranking
                std::vector<float> qm(b * kDim), od(b * 16); std::vector<std::uint32_t> oi(b * 16);
                for (std::size_t i = 0; i < b; ++i) std::copy(qv[i].begin(), qv[i].end(), qm.begin() + std::ptrdiff_t(i * kDim));
                std::vector<double> ms; const std::string m0 = machine();
                for (int r = 0; r < 20; ++r) { const auto t = clk::now(); if (!rt->knn_l2_16(ix.data(), nvec, kDim, qm.data(), b, od.data(), oi.data())) break; ms.push_back(since(t) * 1e3); }
                if (!ms.empty()) std::printf("|   (device call only, knn_l2_16, batch %zu: %s ms over 20 calls; %s) | | | | | | | |\n", b, med_range(ms, "%.3f").c_str(), m0.c_str());
            }
            rt->knn_forget(ix.data());
        }
        {   // gpu_knn_l2 per shard: one resident index per shard, searched one after the other from one host thread, then merged
            std::vector<std::unique_ptr<ann::gpu_knn_l2>> per(nshards);
            for (std::size_t i = 0; i < nshards; ++i) { per[i] = std::make_unique<ann::gpu_knn_l2>(graph::cuda_knn(rt), kDim); for (std::size_t j = 0; j < sid[i].size(); ++j) per[i]->add(sid[i][j], std::span<const float>(sdata[i].data() + j * kDim, kDim)); }
            for (std::size_t b : {std::size_t(1), std::size_t(64), std::size_t(1024)})
                bench("GPU gpu_knn_l2 (new), one resident index per shard, shards in turn", b, [&](std::span<const fvec> qs) {
                    std::vector<std::vector<std::vector<ann::hit>>> parts(nshards);
                    for (std::size_t i = 0; i < nshards; ++i) parts[i] = per[i]->search_batch(qs, K, &host);
                    std::vector<std::vector<ann::hit>> out(qs.size());
                    for (std::size_t q = 0; q < qs.size(); ++q) { std::vector<std::vector<ann::hit>> pq(nshards); for (std::size_t i = 0; i < nshards; ++i) pq[i] = std::move(parts[i][q]); out[q] = merge(pq); }
                    return out; }, b == 1 ? 50 : ann_nq, true);
            for (auto& p : per) rt->knn_forget(p->data());
        }
        std::printf("\nrss at end %ld MiB, %s\n", rss_mib(), machine().c_str());
    }
    return 0;
}
