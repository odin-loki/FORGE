// perf/bench_ivf: does a trained, clustered ANN index beat the per-shard graphs? Not a test; -DFORGE_BUILD_DB_BENCH=ON.
// docs/bench_gpu_vs_cpu.md ("IVF and cluster-aligned shards").
//
//   bench_ivf --dir /dev/shm/gpubench/db64/sharded [--keysorted /dev/shm/gpubench/db255/sharded] [--cluster-out /dev/shm/gpubench/cluster]
//             [--sections ivf,cluster] [--reps 5] [--target 0.97]
//
// ivf      ann::ivf_flat over every vector of the --dir set: k-means on a 5% sample (every 20th row) for nlist = sqrt(N) and 4 sqrt(N);
//          recall@10 against the exact answer over nprobe; at the smallest nprobe reaching --target (and 0.99) throughput and latency at
//          batch 1, 64, 1024 for: CPU 1 thread; CPU one shard per pinned core (each shard an ivf_flat over its rows with the shared
//          centroids; workers compute the probes of their slice of the batch, then scan their shard for every query); GPU scan kernel
//          (rows resident; probes on the host); GPU through gpu_flat_l2's matmul over the probed lists.
// cluster  a copy of the vectors (with perf/bench_db's bodies) re-sharded so that rows are keyed (cluster, id): each <= 10 MiB shard
//          holds one cluster or a few adjacent ones (a large cluster spans shards); the centroids are in manifest.json ("stats": "ivf").
//          A query opens only the shards holding its nprobe nearest clusters and reads those clusters' rows by key range. Against the
//          key-sorted --keysorted set with ann::shard_ann (one HNSW graph per shard, every shard searched). Cold = fresh handles.
#include <forge/ann/ann.hpp>
#include <forge/ann/ivf.hpp>
#include <forge/ann/gpu_knn.hpp>
#include <forge/db/shards.hpp>
#include <forge/graph/ann_gpu.hpp>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
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
static std::string machine() {
    double l1 = 0; { std::ifstream f("/proc/loadavg"); f >> l1; }
    long mem = 0; { std::ifstream f("/proc/meminfo"); std::string k; long v; std::string u; while (f >> k >> v >> u) if (k == "MemAvailable:") { mem = v; break; } }
    char b[64]; std::snprintf(b, sizeof b, "load %.0f, avail %.1f GiB", l1, double(mem) / 1048576.0); return b;
}
static long rss_mib() { std::ifstream f("/proc/self/status"); std::string k; long v; std::string u; while (f >> k) { if (k == "VmRSS:") { f >> v; return v / 1024; } std::getline(f, u); } return 0; }
class pinned_group {
public:
    explicit pinned_group(std::vector<int> cpus) : cpus_(std::move(cpus)) {
        for (std::size_t i = 0; i < cpus_.size(); ++i) th_.emplace_back([this, i] { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpus_[i], &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); loop(i); });
        run([](std::size_t) {});
    }
    ~pinned_group() { stop_ = true; gen_.fetch_add(1); gen_.notify_all(); for (auto& t : th_) t.join(); }
    std::size_t size() const { return cpus_.size(); }
    void run(const std::function<void(std::size_t)>& f) { job_ = &f; left_.store(cpus_.size()); gen_.fetch_add(1); gen_.notify_all(); for (;;) { auto l = left_.load(); if (!l) break; left_.wait(l); } }
private:
    void loop(std::size_t i) { std::uint64_t seen = 0; for (;;) { gen_.wait(seen); seen = gen_.load(); if (stop_) return; (*job_)(i); if (left_.fetch_sub(1) == 1) left_.notify_all(); } }
    std::vector<int> cpus_; std::vector<std::thread> th_; const std::function<void(std::size_t)>* job_ = nullptr;
    std::atomic<std::uint64_t> gen_{0}; std::atomic<std::size_t> left_{0}; std::atomic<bool> stop_{false};
};
static double pct(std::vector<double> v, double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); const double r = p * double(v.size() - 1); const std::size_t i = std::size_t(r); return i + 1 < v.size() ? v[i] + (r - double(i)) * (v[i + 1] - v[i]) : v[i]; }
struct rep_stats { double qps, p50, p99; };
static rep_stats stats_of(const std::vector<double>& lat, double wall, double n) { std::vector<double> ms; for (double x : lat) ms.push_back(x * 1e3); return {n / wall, pct(ms, 0.5), pct(ms, 0.99)}; }
static std::string med_range(std::vector<double> v, const char* fmt) { std::sort(v.begin(), v.end()); char a[48], b[48], c[48]; std::snprintf(a, sizeof a, fmt, pct(v, 0.5)); std::snprintf(b, sizeof b, fmt, v.front()); std::snprintf(c, sizeof c, fmt, v.back()); return std::string(a) + " (" + b + ".." + c + ")"; }
static void print_row(const std::string& head, const std::vector<rep_stats>& reps, const std::string& m0, const std::string& m1, const std::string& extra = "") {
    std::vector<double> q, a, b; for (auto& r : reps) { q.push_back(r.qps); a.push_back(r.p50); b.push_back(r.p99); }
    std::printf("| %s | %s | %s | %s | %zu | %s / %s |%s\n", head.c_str(), med_range(q, "%.1f").c_str(), med_range(a, "%.3f").c_str(), med_range(b, "%.3f").c_str(), reps.size(), m0.c_str(), m1.c_str(), extra.c_str());
    std::fflush(stdout);
}
// perf/bench_db's row generator (bodies for the re-sharded copy; the vectors come from the source set)
static std::string word(std::uint64_t r) { return "w" + std::to_string(r * r / 40000); }
static std::string body_of(std::uint64_t i) {
    std::mt19937_64 g(0x9E3779B97F4A7C15ull * (i + 1)); std::string body; const int nw = 12 + int(g() % 24);
    for (int w = 0; w < nw; ++w) { body += word(g() % 4000); body += ' '; }
    if (i % 3 == 0) body += "c" + std::to_string(i / 40000) + " ";
    if (i % 20011 == 0) body += "u" + std::to_string(i / 200000) + " ";
    return body;
}
static std::string hexf(const std::vector<float>& v) { static const char* H = "0123456789abcdef"; std::string s; s.reserve(v.size() * 8); const auto* p = reinterpret_cast<const unsigned char*>(v.data()); for (std::size_t i = 0; i < v.size() * 4; ++i) { s += H[p[i] >> 4]; s += H[p[i] & 15]; } return s; }
static std::vector<float> unhexf(const std::string& s) { std::vector<float> v(s.size() / 8); auto* p = reinterpret_cast<unsigned char*>(v.data()); auto h = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; }; for (std::size_t i = 0; i < v.size() * 4; ++i) p[i] = (unsigned char)(h(s[2 * i]) * 16 + h(s[2 * i + 1])); return v; }
static std::string ckey(std::uint32_t c, std::int64_t id) { return db::int_key(std::int64_t(c)) + db::int_key(id); }

int main(int argc, char** argv) {
    fs::path dir, keysorted, cluster_out = "/dev/shm/gpubench/cluster"; std::set<std::string> sections = {"ivf", "cluster"}; int reps = 5; double target = 0.97;
    std::size_t nq = 1024, nq_single = 200, nq_cluster = 200, mixture = 0;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; auto next = [&] { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--dir") dir = next(); else if (a == "--keysorted") keysorted = next(); else if (a == "--cluster-out") cluster_out = next(); else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--target") target = std::stod(next()); else if (a == "--mixture") mixture = std::stoull(next()); else if (a == "--cluster-queries") nq_cluster = std::stoull(next());
        else if (a == "--sections") { sections.clear(); std::stringstream ss(next()); std::string t; while (std::getline(ss, t, ',')) sections.insert(t); }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; } }
    std::vector<int> cpus; { cpu_set_t s; sched_getaffinity(0, sizeof s, &s); for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &s)) cpus.push_back(c); }
    constexpr std::size_t D = 32, K = 10;
    db::shard_open_options oo; oo.verify = db::verify_mode::none; oo.max_open_bytes = 64ull << 30;
    const std::size_t nshards = db::open(dir, oo).size(), W = std::min(nshards, cpus.size());
    std::vector<int> wcpus(cpus.begin(), cpus.begin() + std::ptrdiff_t(W));
    task::pool tp{unsigned(W)};
    // vectors per shard as stored
    std::vector<std::vector<std::uint64_t>> sid(nshards); std::vector<fvec> sdata(nshards);
    { auto set = db::open(dir, oo); task::parallel_blocks(&tp, nshards, 1, [&](std::size_t i, std::size_t, std::size_t) {
          auto l = set.acquire(i); auto st = l->prepare("SELECT id, vec FROM t ORDER BY k");
          for (auto& r : st.rows()) { sid[i].push_back(std::uint64_t(r.get<std::int64_t>(0))); auto b = r.get<std::span<const std::byte>>(1); const std::size_t o = sdata[i].size(); sdata[i].resize(o + D); std::memcpy(sdata[i].data() + o, b.data(), D * 4); } }); }
    // --mixture C: replace every vector (same ids, same shards) by a Gaussian mixture point: centre c = id % C (centres N(0, 1)), plus
    // N(0, 0.3^2) noise; queries are fresh points of the same mixture. The bench_db vectors are i.i.d. N(0, 1): no cluster structure.
    std::vector<fvec> centres;
    if (mixture) { std::mt19937_64 gc(99); std::normal_distribution<float> nd; centres.assign(mixture, fvec(D)); for (auto& c : centres) for (auto& f : c) f = nd(gc);
        for (std::size_t i = 0; i < nshards; ++i) for (std::size_t j = 0; j < sid[i].size(); ++j) { std::mt19937_64 g(0xC0FFEEull * (sid[i][j] + 7)); const auto& c = centres[sid[i][j] % mixture];
            for (std::size_t t = 0; t < D; ++t) sdata[i][j * D + t] = c[t] + 0.3f * nd(g); } }
    std::size_t N = 0; for (auto& v : sid) N += v.size();
    std::vector<float> all; all.reserve(N * D); std::vector<std::uint64_t> allid; for (std::size_t i = 0; i < nshards; ++i) { all.insert(all.end(), sdata[i].begin(), sdata[i].end()); allid.insert(allid.end(), sid[i].begin(), sid[i].end()); }
    std::vector<fvec> qv(nq); { std::normal_distribution<float> nd; std::mt19937_64 gq(4242); for (auto& v : qv) { v.resize(D); const fvec* c = mixture ? &centres[gq() % mixture] : nullptr; for (std::size_t t = 0; t < D; ++t) v[t] = c ? (*c)[t] + 0.3f * nd(gq) : nd(gq); } }   // = bench_gpu_cpu's queries without --mixture
    std::printf("# bench_ivf: %zu vectors (dim %zu, %s) from %s (%zu shards), %zu queries, pinned workers %zu, reps %d, %s, rss %ld MiB\n\n", N, D, mixture ? ("Gaussian mixture of " + std::to_string(mixture) + " centres, sigma 0.3").c_str() : "as stored: i.i.d. N(0,1)", dir.string().c_str(), nshards, nq, W, reps, machine().c_str(), rss_mib());
    // exact answers on the GPU (gpu_knn_l2: exact, checked per query; CPU fallback where the check fails)
    std::string why; auto rt = graph::open_knn_device("cuda", &why);
    std::vector<std::vector<ann::hit>> exact;
    { ann::gpu_knn_l2 gx(graph::cuda_knn(rt), D); for (std::size_t r = 0; r < N; ++r) gx.add(allid[r], std::span<const float>(all.data() + r * D, D));
      const auto t0 = clk::now(); ann::gpu_knn_l2::report rep; exact = gx.search_batch(qv, K, &tp, &rep); if (rt) rt->knn_forget(gx.data());
      std::printf("exact answers: gpu_knn_l2 %.2f s (%zu of %zu queries on the CPU path, device %s)\n\n", since(t0), rep.fallback_queries, nq, rt ? "yes" : ("no: " + why).c_str()); }
    auto recall = [&](const std::vector<std::vector<ann::hit>>& got, std::size_t q0) { std::size_t ok = 0; for (std::size_t r = 0; r < got.size(); ++r) for (auto& h : got[r]) for (auto& e : exact[q0 + r]) if (e.id == h.id) { ++ok; break; } return double(ok) / double(got.size() * K); };
    auto bench = [&](const std::string& name, std::size_t batch, const std::function<std::vector<std::vector<ann::hit>>(std::span<const fvec>)>& fn, std::size_t total, const std::string& extra = "") {
        total = std::min(total, qv.size()); total -= total % batch; if (!total) total = batch;
        std::vector<rep_stats> rs; const std::string m0 = machine(); double rec = 0; (void)fn(std::span<const fvec>(qv.data(), batch));
        for (int r = 0; r < reps; ++r) { std::vector<double> lat; std::vector<std::vector<ann::hit>> got; const auto t0 = clk::now();
            for (std::size_t q0 = 0; q0 + batch <= total; q0 += batch) { const auto t = clk::now(); auto h = fn(std::span<const fvec>(qv.data() + q0, batch)); lat.push_back(since(t)); got.insert(got.end(), h.begin(), h.end()); }
            rs.push_back(stats_of(lat, since(t0), double(total))); rec = recall(got, 0); }
        char rb[64]; std::snprintf(rb, sizeof rb, " %.4f", rec);
        print_row(name + " | " + std::to_string(batch) + " (" + std::to_string(total) + " queries)", rs, m0, machine(), rb + extra);
    };
    std::vector<float> sample; for (std::size_t r = 0; r < N; r += 20) sample.insert(sample.end(), all.begin() + std::ptrdiff_t(r * D), all.begin() + std::ptrdiff_t((r + 1) * D));
    const std::size_t S = sample.size() / D, nl1 = std::size_t(std::lround(std::sqrt(double(N))));
    std::vector<float> cents_sqrt;

    if (sections.count("ivf")) {
        std::printf("## IVF (ann::ivf_flat, new): k-means on %zu sample rows (5%%), 20 Lloyd iterations, task::pool(%zu)\n\n", S, W);
        std::printf("| method | batch | queries/s | p50 ms | p99 ms | reps | machine before / after | recall@10 |\n|---|---|---|---|---|---|---|---|\n");
        pinned_group pg(wcpus);
        {   // baselines on the same vectors: exact flat and per-shard HNSW (M 16, efConstruction 100 as perf/bench_db), one shard per pinned core; gpu_knn_l2
            std::vector<std::unique_ptr<ann::flat_index<fvec, ann::l2>>> flat(nshards); std::vector<std::unique_ptr<ann::hnsw<fvec, ann::l2>>> graph(nshards);
            const auto t0 = clk::now();
            pg.run([&](std::size_t w) { for (std::size_t i = w; i < nshards; i += pg.size()) {
                flat[i] = std::make_unique<ann::flat_index<fvec, ann::l2>>(); graph[i] = std::make_unique<ann::hnsw<fvec, ann::l2>>(ann::hnsw_params{.M = 16, .ef_construction = 100, .ef_search = 64, .seed = 1});
                for (std::size_t j = 0; j < sid[i].size(); ++j) { fvec v(sdata[i].begin() + std::ptrdiff_t(j * D), sdata[i].begin() + std::ptrdiff_t((j + 1) * D)); flat[i]->add(sid[i][j], v); graph[i]->add(sid[i][j], v); }
                graph[i]->build(nullptr); } });
            std::printf("| baselines: per-shard flat + HNSW built on their workers (one-off): %.1f s | | | | | 1 | %s | |\n", since(t0), machine().c_str());
            auto fan = [&](std::span<const fvec> qs, std::size_t ef) {
                std::vector<std::vector<std::vector<ann::hit>>> part(qs.size(), std::vector<std::vector<ann::hit>>(nshards));
                pg.run([&](std::size_t w) { for (std::size_t i = w; i < nshards; i += pg.size()) for (std::size_t q = 0; q < qs.size(); ++q) part[q][i] = ef ? graph[i]->search(qs[q], K, ef) : flat[i]->search(qs[q], K, nullptr); });
                std::vector<std::vector<ann::hit>> o(qs.size());
                for (std::size_t q = 0; q < qs.size(); ++q) { std::vector<ann::hit> a; for (auto& p : part[q]) a.insert(a.end(), p.begin(), p.end()); std::sort(a.begin(), a.end(), ann::hit_before); if (a.size() > K) a.resize(K); o[q] = std::move(a); }
                return o; };
            std::size_t ef_t = 0;
            for (std::size_t ef : {16u, 32u, 64u, 128u, 256u, 512u}) { const double rc = recall(fan(std::span<const fvec>(qv.data(), 256), ef), 0);
                std::printf("| CPU HNSW per shard ef %zu (recall sweep, 256 queries) | | | | | | | %.4f |\n", ef, rc); if (!ef_t && rc >= target) ef_t = ef; }
            if (!ef_t) ef_t = 512;
            for (std::size_t b : {std::size_t(1), std::size_t(64), std::size_t(1024)})
                bench("CPU HNSW per shard ef " + std::to_string(ef_t) + " (first ef reaching the target), one shard per pinned core (" + std::to_string(W) + ")", b, [&](std::span<const fvec> qs) { return fan(qs, ef_t); }, b == 1 ? nq_single : nq);
            bench("CPU flat (exact), one shard per pinned core (" + std::to_string(W) + ")", 1024, [&](std::span<const fvec> qs) { return fan(qs, 0); }, nq);
            if (rt) { ann::gpu_knn_l2 gx(graph::cuda_knn(rt), D); for (std::size_t r = 0; r < N; ++r) gx.add(allid[r], std::span<const float>(all.data() + r * D, D));
                for (std::size_t b : {std::size_t(1), std::size_t(64), std::size_t(1024)})
                    bench("GPU gpu_knn_l2 (exact, resident)", b, [&](std::span<const fvec> qs) { return gx.search_batch(qs, K, &tp); }, b == 1 ? nq_single : nq);
                rt->knn_forget(gx.data()); }
        }
        for (std::size_t nlist : {nl1, 4 * nl1}) {
            std::vector<double> tt; std::vector<float> cents; const std::string m0 = machine();
            for (int r = 0; r < reps; ++r) { const auto t0 = clk::now(); cents = ann::kmeans(sample.data(), S, D, nlist, 20, 7, &tp); tt.push_back(since(t0)); }
            std::printf("| training nlist %zu (one-off): k-means %s s | | | | | %d | %s / %s | |\n", nlist, med_range(tt, "%.2f").c_str(), reps, m0.c_str(), machine().c_str());
            if (nlist == nl1) cents_sqrt = cents;
            ann::ivf_flat gx(cents, D); for (std::size_t r = 0; r < N; ++r) gx.add(allid[r], std::span<const float>(all.data() + r * D, D));
            { const auto t0 = clk::now(); gx.finalize(&tp); std::size_t mx = 0, mn = N; for (std::size_t l = 0; l < nlist; ++l) { mx = std::max(mx, gx.list_end(l) - gx.list_begin(l)); mn = std::min(mn, gx.list_end(l) - gx.list_begin(l)); }
              std::printf("| index build nlist %zu (one-off): assign %zu rows %.2f s; list sizes min %zu, mean %.0f, max %zu | | | | | 1 | %s | |\n", nlist, N, since(t0), mn, double(N) / double(nlist), mx, machine().c_str()); }
            // per-shard indexes with the shared centroids, built on their workers
            std::vector<std::unique_ptr<ann::ivf_flat>> per(nshards);
            pg.run([&](std::size_t w) { for (std::size_t i = w; i < nshards; i += pg.size()) { per[i] = std::make_unique<ann::ivf_flat>(cents, D); for (std::size_t j = 0; j < sid[i].size(); ++j) per[i]->add(sid[i][j], std::span<const float>(sdata[i].data() + j * D, D)); per[i]->finalize(); } });
            // recall sweep
            std::size_t np_t = 0, np_99 = 0;
            for (std::size_t np : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 2048u}) {
                if (np > nlist) break;
                std::vector<std::vector<ann::hit>> got(nq); std::vector<std::size_t> scanned(nq, 0);
                task::parallel_blocks(&tp, nq, 1, [&](std::size_t q, std::size_t, std::size_t) { auto p = gx.probes(qv[q], np); got[q] = gx.search_lists(qv[q], K, p, &scanned[q]); });
                double sc = 0; for (auto s : scanned) sc += double(s); const double rc = recall(got, 0);
                std::printf("| ivf nlist %zu nprobe %zu (recall sweep) | | | | | | rows scanned per query %.0f (%.2f%% of N) | %.4f |\n", nlist, np, sc / double(nq), 100.0 * sc / double(nq) / double(N), rc);
                if (!np_t && rc >= target) np_t = np; if (!np_99 && rc >= 0.99) np_99 = np;
            }
            if (!np_t) np_t = std::min<std::size_t>(nlist, 2048); if (!np_99) np_99 = std::min<std::size_t>(nlist, 2048);
            for (std::size_t np : {np_t, np_99}) {
                const std::string tag = "ivf nlist " + std::to_string(nlist) + " nprobe " + std::to_string(np);
                bench("CPU " + tag + ", 1 thread", 1, [&](std::span<const fvec> qs) { std::vector<std::vector<ann::hit>> o; for (auto& q : qs) { auto p = gx.probes(q, np); o.push_back(gx.search_lists(q, K, p)); } return o; }, nq_single);
                for (std::size_t b : {std::size_t(1), std::size_t(64), std::size_t(1024)})
                    bench("CPU " + tag + ", one shard per pinned core (" + std::to_string(W) + ")", b, [&](std::span<const fvec> qs) {
                        std::vector<std::vector<std::uint32_t>> pr(qs.size());
                        pg.run([&](std::size_t w) { for (std::size_t q = w; q < qs.size(); q += pg.size()) pr[q] = gx.probes(qs[q], np); });
                        std::vector<std::vector<std::vector<ann::hit>>> part(qs.size(), std::vector<std::vector<ann::hit>>(nshards));
                        pg.run([&](std::size_t w) { for (std::size_t i = w; i < nshards; i += pg.size()) for (std::size_t q = 0; q < qs.size(); ++q) part[q][i] = per[i]->search_lists(qs[q], K, pr[q]); });
                        std::vector<std::vector<ann::hit>> o(qs.size());
                        for (std::size_t q = 0; q < qs.size(); ++q) { std::vector<ann::hit> a; for (auto& p : part[q]) a.insert(a.end(), p.begin(), p.end()); std::sort(a.begin(), a.end(), ann::hit_before); if (a.size() > K) a.resize(K); o[q] = std::move(a); }
                        return o; }, b == 1 ? nq_single : nq);
                if (rt) {
                    double up = 0; bool fb = false;
                    { rt->knn_forget(gx.data()); (void)gx.search_batch_device(std::span<const fvec>(qv.data(), 1), K, np, graph::cuda_ivf(rt), &tp, &fb, &up); }
                    std::printf("| GPU %s: upload of the %zu rows sorted by list (once) | | %.1f ms | | | 1 | %s | %.2f GB/s |\n", tag.c_str(), N, up * 1e3, machine().c_str(), double(N * D * 4) / up / 1e9);
                    for (std::size_t b : {std::size_t(1), std::size_t(64), std::size_t(1024)}) {
                        const auto d0 = rt->dispatches();
                        bench("GPU " + tag + ", scan kernel (new; rows resident, probes on the host pool)", b, [&](std::span<const fvec> qs) { return gx.search_batch_device(qs, K, np, graph::cuda_ivf(rt), &tp, &fb); }, b == 1 ? nq_single : nq,
                              fb ? " DEVICE FAILED (CPU path)" : "");
                        auto d1 = rt->dispatches(); std::printf("|   (forge_ivf_l2_part launches %zu) | | | | | | | |\n", (d1.count("ivf_l2_part") ? d1["ivf_l2_part"] : 0) - (d0.count("ivf_l2_part") ? d0.at("ivf_l2_part") : 0));
                    }
                    rt->knn_forget(gx.data());
                    auto reg = graph::gpu_registry(rt, false);
                    for (std::size_t b : {std::size_t(64), std::size_t(1024)}) {
                        std::size_t calls = 0;
                        bench("GPU " + tag + ", gpu_flat_l2's matmul_t over the probed lists (one product per touched list)", b, [&](std::span<const fvec> qs) { return gx.search_batch_matmul(qs, K, np, graph::ann_matmul(reg.get()), &tp, &calls); }, b == 64 ? 256 : nq);
                        std::printf("|   (device matmul products over the calls of the row above, warm-up included: %zu) | | | | | | | |\n", calls);
                    }
                }
            }
        }
        std::printf("\n");
    }

    if (sections.count("cluster") && mixture) std::printf("cluster section skipped with --mixture (it re-shards the stored vectors)\n");
    if (sections.count("cluster") && !mixture) {
        if (cents_sqrt.empty()) cents_sqrt = ann::kmeans(sample.data(), S, D, nl1, 20, 7, &tp);
        const std::size_t NL = cents_sqrt.size() / D;
        std::printf("## cluster-aligned shards: nlist %zu, rows keyed (cluster, id), 10 MiB budget, centroids in manifest stats \"ivf\"\n\n", NL);
        const fs::path csd = cluster_out / "sharded";
        if (!fs::exists(csd / "manifest.json")) {
            const auto t0 = clk::now();
            std::vector<std::uint32_t> lst(N); task::parallel_blocks(&tp, N, 4096, [&](std::size_t, std::size_t b, std::size_t e) { for (std::size_t r = b; r < e; ++r) lst[r] = std::uint32_t(ann::detail::ivf_nearest(all.data() + r * D, cents_sqrt.data(), NL, D)); });
            std::vector<std::size_t> order(N); std::iota(order.begin(), order.end(), std::size_t(0));
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return lst[a] != lst[b] ? lst[a] < lst[b] : allid[a] < allid[b]; });
            db::shard_schema sc; sc.ddl = "CREATE TABLE t(k BLOB PRIMARY KEY, id INTEGER NOT NULL, body TEXT, vec BLOB) WITHOUT ROWID;"; sc.table = "t"; sc.columns = {"k", "id", "body", "vec"};
            fs::remove_all(csd);
            db::shard_writer w(csd, sc, {.budget = 10ull << 20, .presorted = true, .wave_bytes = 128ull << 20}, &tp);
            json::object ivf; ivf.push_back({"nlist", json::value(double(NL))}); ivf.push_back({"dim", json::value(double(D))}); ivf.push_back({"key", json::value(std::string("int_key(cluster) + int_key(id)"))});
            ivf.push_back({"centroids_f32_hex", json::value(hexf(cents_sqrt))});
            w.set_stat("ivf", json::value(ivf));
            for (std::size_t r : order) { const std::int64_t id = std::int64_t(allid[r]); const auto k = ckey(lst[r], id);
                std::vector<std::byte> vb(reinterpret_cast<const std::byte*>(all.data() + r * D), reinterpret_cast<const std::byte*>(all.data() + (r + 1) * D));
                w.add({k, id, body_of(std::uint64_t(id - 1)), vb}); }
            w.finish();
            std::printf("built %s: %.1f s (assign + sort + write, one-off)\n", csd.string().c_str(), since(t0));
        }
        // read the centroids back from the manifest
        std::vector<float> mc; { auto s = db::open(csd, oo); const auto* iv = json::value(s.manifest().stats).find("ivf"); if (!iv) { std::fprintf(stderr, "no ivf stats in manifest\n"); return 1; } mc = unhexf(iv->find("centroids_f32_hex")->str()); }
        if (mc != cents_sqrt) std::printf("WARNING: manifest centroids differ from the trained ones (another build?); using the manifest's\n");
        const std::size_t ncs = db::open(csd, oo).size(); std::uint64_t cbytes = 0; { auto s = db::open(csd, oo); for (std::size_t i = 0; i < ncs; ++i) cbytes += s.info(i).bytes; }
        std::printf("cluster-aligned set: %zu shards, %.1f MiB (avg %.2f MiB)\n\n", ncs, double(cbytes) / 1048576.0, double(cbytes) / double(ncs) / 1048576.0);
        ann::ivf_flat probe_ix(mc, D);
        std::printf("| method | handle | queries/s | p50 ms | p99 ms | reps | machine before / after | recall@10; shards opened / query; rows read / query; bytes read / query (est.) |\n|---|---|---|---|---|---|---|---|\n");
        const std::size_t nqc = std::min(nq_cluster, nq);
        for (std::size_t np : {8u, 64u, 256u, 512u}) for (int mode = 0; mode < 2; ++mode) {
            const std::size_t Wc = mode == 0 ? 1 : std::min(cpus.size(), ncs);
            pinned_group pg(std::vector<int>(cpus.begin(), cpus.begin() + std::ptrdiff_t(Wc)));
            std::vector<rep_stats> cold, warm; const std::string m0 = machine(); double rec = 0, sh_q = 0, rows_q = 0, bytes_q = 0;
            for (int r = 0; r < reps; ++r) {
                auto set = db::open(csd, oo);
                std::vector<std::optional<db::shard_set::lease>> leases(ncs); std::vector<std::optional<db::statement>> st(ncs);
                for (int pass = 0; pass < 2; ++pass) {
                    std::vector<double> lat; std::vector<std::vector<ann::hit>> got; double shs = 0, rows = 0, bytes = 0; const auto t0 = clk::now();
                    for (std::size_t q = 0; q < nqc; ++q) {
                        const auto t = clk::now();
                        auto pr = probe_ix.probes(qv[q], np);
                        std::map<std::size_t, std::vector<std::uint32_t>> work;   // shard -> clusters to read there
                        for (auto c : pr) for (auto s : set.shards_for_range(ckey(c, 0), ckey(c, std::numeric_limits<std::int64_t>::max()))) work[s].push_back(c);
                        std::vector<std::pair<std::size_t, std::vector<std::uint32_t>>> wl(work.begin(), work.end());
                        std::vector<std::vector<ann::hit>> part(wl.size()); std::vector<std::size_t> nrows(wl.size(), 0);
                        pg.run([&](std::size_t w) { for (std::size_t j = 0; j < wl.size(); ++j) { const std::size_t s = wl[j].first; if (s % pg.size() != w) continue;
                            if (!st[s]) { leases[s].emplace(set.acquire(s)); st[s].emplace((*leases[s])->prepare("SELECT id, vec FROM t WHERE k >= ?1 AND k < ?2")); }
                            std::vector<ann::hit> heap; const ann::l2 m;
                            for (auto c : wl[j].second) { auto& qq = *st[s]; qq.reset(); const auto lo = ckey(c, 0), hi = ckey(c + 1, 0); qq.bind(1, db::static_span_(lo)).bind(2, db::static_span_(hi));
                                while (qq.step()) { auto b = qq.get<std::span<const std::byte>>(1); const ann::hit h{std::uint64_t(qq.get<std::int64_t>(0)), m.distance(std::span<const float>(qv[q]), std::span<const float>(reinterpret_cast<const float*>(b.data()), D))}; ++nrows[j];
                                    if (heap.size() < K) { heap.push_back(h); std::push_heap(heap.begin(), heap.end(), ann::hit_before); } else if (ann::hit_before(h, heap.front())) { std::pop_heap(heap.begin(), heap.end(), ann::hit_before); heap.back() = h; std::push_heap(heap.begin(), heap.end(), ann::hit_before); } }
                                qq.reset(); }
                            part[j] = std::move(heap); } });
                        std::vector<ann::hit> a; for (auto& p : part) a.insert(a.end(), p.begin(), p.end()); std::sort(a.begin(), a.end(), ann::hit_before); if (a.size() > K) a.resize(K);
                        lat.push_back(since(t)); got.push_back(std::move(a));
                        shs += double(wl.size()); for (std::size_t j = 0; j < wl.size(); ++j) { rows += double(nrows[j]); const auto& si = set.info(wl[j].first); bytes += double(nrows[j]) * double(si.bytes) / double(std::max<std::int64_t>(1, si.rows)); }
                    }
                    (pass == 0 ? cold : warm).push_back(stats_of(lat, since(t0), double(nqc)));
                    rec = recall(got, 0); sh_q = shs / double(nqc); rows_q = rows / double(nqc); bytes_q = bytes / double(nqc);
                }
                st.clear(); leases.clear();
            }
            char ex[200]; std::snprintf(ex, sizeof ex, " %.4f; %.1f of %zu; %.0f; %.2f MiB", rec, sh_q, ncs, rows_q, bytes_q / 1048576.0);
            const std::string nm = "cluster shards, nprobe " + std::to_string(np) + ", " + (mode == 0 ? std::string("1 thread") : std::to_string(Wc) + " pinned workers (shard s on worker s % W)") + " (" + std::to_string(nqc) + " queries)";
            print_row(nm + " | fresh", cold, m0, machine(), ex); print_row(nm + " | warm", warm, m0, machine(), ex);
        }
        if (!keysorted.empty()) {   // key-sorted shards + shard_ann over every shard (library), task::pool of up to 64 threads
            const std::size_t nks = db::open(keysorted, oo).size(); std::uint64_t kb = 0; { auto s = db::open(keysorted, oo); for (std::size_t i = 0; i < nks; ++i) kb += s.info(i).bytes; }
            task::pool kp{unsigned(std::min<std::size_t>(cpus.size(), nks))};
            for (std::size_t ef : {64u, 160u}) {
                std::vector<rep_stats> cold, warm; std::vector<double> first; const std::string m0 = machine(); double rec = 0;
                for (int r = 0; r < reps; ++r) {
                    auto set = db::open(keysorted, oo); ann::shard_ann<fvec, ann::l2> sa(set);
                    for (int pass = 0; pass < 2; ++pass) { std::vector<double> lat; std::vector<std::vector<ann::hit>> got; const auto t0 = clk::now();
                        for (std::size_t q = 0; q < nqc; ++q) { const auto t = clk::now(); got.push_back(sa.search(qv[q], K, &kp, ef)); lat.push_back(since(t)); }
                        if (pass == 0) first.push_back(lat[0] * 1e3);
                        (pass == 0 ? cold : warm).push_back(stats_of(lat, since(t0), double(nqc))); rec = recall(got, 0); }
                }
                char ex[220]; std::snprintf(ex, sizeof ex, " %.4f; %zu of %zu (every graph); graphs loaded on first use: %.0f MiB of shard files opened; first query of a fresh handle %s ms", rec, nks, nks, double(kb) / 1048576.0, med_range(first, "%.0f").c_str());
                const std::string nm = "key-sorted shards + shard_ann (HNSW per shard), ef " + std::to_string(ef) + ", task::pool(" + std::to_string(std::min<std::size_t>(cpus.size(), nks)) + ") (" + std::to_string(nqc) + " queries)";
                print_row(nm + " | fresh", cold, m0, machine(), ex); print_row(nm + " | warm", warm, m0, machine(), ex);
            }
        }
    }
    std::printf("\nrss at end %ld MiB, %s\n", rss_mib(), machine().c_str());
    return 0;
}
