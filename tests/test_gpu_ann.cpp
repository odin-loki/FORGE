// forge::ann::gpu_flat_l2: batched exact search through a matmul (a device's, or a CPU stand-in) equals flat_index::search.
// gpu_flat_logic runs everywhere (a plain CPU matmul, a noisy one within the error bound, a failing one); gpu_flat_device uses
// graph::device_registry("cuda") and exits 77 without a device.
#include "check.hpp"
#include <forge/graph/ann_gpu.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
using namespace forge;
using fvec = std::vector<float>;
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<fvec> random_vectors(std::size_t n, std::size_t d, std::uint64_t seed, bool with_ties) {
    std::mt19937_64 g(seed); std::normal_distribution<float> nd;
    std::vector<fvec> v(n, fvec(d));
    for (auto& x : v) for (auto& f : x) f = with_ties ? float(int(nd(g) * 2.0f)) * 0.5f : nd(g);   // with_ties: a coarse grid, so many exact distance ties
    return v;
}
static bool same(const std::vector<ann::hit>& a, const std::vector<ann::hit>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i].id != b[i].id || a[i].dist != b[i].dist) return false;
    return true;
}
static void cpu_matmul(const float* x, std::size_t rows, std::size_t k, const float* w, std::size_t cols, float* y) {
    for (std::size_t r = 0; r < rows; ++r) for (std::size_t c = 0; c < cols; ++c) { float s = 0; for (std::size_t i = 0; i < k; ++i) s += x[r * k + i] * w[c * k + i]; y[r * cols + c] = s; }
}

TEST(gpu_flat_logic) {
    task::pool pool(8);
    for (bool ties : {false, true}) for (std::size_t d : {37u, 64u}) {
        auto vs = random_vectors(3000 + 100, d, 7 + d, ties);
        ann::flat_index<fvec, ann::l2> exact; std::vector<fvec> qs(vs.begin() + 3000, vs.end());
        std::size_t calls_fail = 0;
        struct mm_case { const char* name; ann::gpu_flat_l2::matmul_fn fn; };
        std::mt19937_64 ng(5); std::uniform_real_distribution<float> ud(-1, 1);
        mm_case cases[] = {
            {"cpu f32", [](const float* x, std::size_t r, std::size_t k, const float* w, std::size_t c, float* y) { cpu_matmul(x, r, k, w, c, y); return true; }},
            {"noisy (1e-6 relative to |q||x|)", [&](const float* x, std::size_t r, std::size_t k, const float* w, std::size_t c, float* y) {
                 cpu_matmul(x, r, k, w, c, y);
                 for (std::size_t i = 0; i < r; ++i) { double qn = 0; for (std::size_t j = 0; j < k; ++j) qn += double(x[i * k + j]) * x[i * k + j];
                     for (std::size_t j = 0; j < c; ++j) { double xn = 0; for (std::size_t t = 0; t < k; ++t) xn += double(w[j * k + t]) * w[j * k + t]; y[i * c + j] += float(1e-6 * std::sqrt(qn * xn) * ud(ng)); } }
                 return true; }},
            {"failing", [&](const float*, std::size_t, std::size_t, const float*, std::size_t, float*) { ++calls_fail; return false; }},
        };
        for (auto& mc : cases) for (std::size_t chunk : {std::size_t(500), std::size_t(1) << 16}) {
            ann::gpu_flat_l2 ix(mc.fn, d, chunk, 40);
            exact = ann::flat_index<fvec, ann::l2>();
            for (std::size_t i = 0; i < 3000; ++i) { ix.add(i * 7 + 1, vs[i]); exact.add(i * 7 + 1, vs[i]); }
            for (std::size_t k : {1u, 10u, 64u}) {
                ann::gpu_flat_l2::report rep;
                auto got = ix.search_batch(qs, k, &pool, &rep);
                bool ok = got.size() == qs.size();
                for (std::size_t q = 0; ok && q < qs.size(); ++q) ok = same(got[q], exact.search(qs[q], k, nullptr));
                CHECK_MSG(ok, "gpu_flat != flat_index: %s ties=%d d=%zu chunk=%zu k=%zu", mc.name, int(ties), d, chunk, k);
                if (std::string(mc.name) == "failing") CHECK(rep.fell_back); else CHECK(!rep.fell_back);
                if (std::string(mc.name) != "failing") CHECK_MSG(rep.candidates < qs.size() * (k + 4000), "candidates %zu", rep.candidates);
                auto serial = ix.search_batch(qs, k, nullptr);   // no pool: same result
                CHECK(serial.size() == got.size()); for (std::size_t q = 0; q < got.size(); ++q) CHECK(same(serial[q], got[q]));
            }
        }
        CHECK(calls_fail > 0);
    }
    { ann::gpu_flat_l2 ix(nullptr, 4); CHECK(ix.search_batch(std::vector<fvec>{{1, 2, 3, 4}}, 3)[0].empty()); CHECK_THROWS(ix.add(1, fvec(3, 0.f))); }
}

TEST(gpu_flat_device) {
    std::string why; auto reg = graph::device_registry("cuda", &why);
    if (!reg) { std::printf("no CUDA device (%s): skipped\n", why.c_str()); std::exit(77); }
    task::pool pool(8);
    for (bool ties : {false, true}) for (std::size_t d : {37u, 64u, 128u}) {
        const std::size_t n = ties ? 20000 : 60000, nq = 300;
        auto vs = random_vectors(n + nq, d, 11 + d, ties);
        ann::gpu_flat_l2 ix(graph::ann_matmul(reg.get()), d, 1u << 15, 128);
        ann::flat_index<fvec, ann::l2> exact;
        for (std::size_t i = 0; i < n; ++i) { ix.add(i + 100, vs[i]); exact.add(i + 100, vs[i]); }
        std::vector<fvec> qs(vs.begin() + std::ptrdiff_t(n), vs.end());
        for (std::size_t k : {1u, 10u, 100u}) {
            ann::gpu_flat_l2::report rep; const double t0 = now_s();
            auto got = ix.search_batch(qs, k, &pool, &rep);
            const double tg = now_s() - t0;
            std::vector<std::vector<ann::hit>> want(nq); const double t1 = now_s();
            for (std::size_t q = 0; q < nq; ++q) want[q] = exact.search(qs[q], k, &pool);
            const double tc = now_s() - t1;
            bool ok = true; for (std::size_t q = 0; ok && q < nq; ++q) ok = same(got[q], want[q]);
            CHECK_MSG(ok, "device result != flat_index: ties=%d d=%zu k=%zu", int(ties), d, k);
            CHECK(!rep.fell_back && rep.device_calls > 0);
            std::printf("  cuda flat: n=%zu d=%zu k=%zu: device path %.3f s, CPU flat_index (8 threads) %.3f s, %zu candidates re-ranked (%.1f per query)\n", n, d, k, tg, tc, rep.candidates, double(rep.candidates) / double(nq));
        }
    }
}

// forge::ann::gpu_knn_l2 with a CPU stand-in for the device's 16-nearest call: exact, noisy within the error bound (the host
// must still return flat_index's answer, through the boundary check or the per-query fallback), failing, and too-small.
static bool cpu_knn16(const float* x, std::size_t n, std::size_t d, const float* q, std::size_t nq, float* od, std::uint32_t* oi, double noise, std::mt19937_64* g) {
    std::uniform_real_distribution<double> ud(-1, 1);
    for (std::size_t i = 0; i < nq; ++i) {
        std::vector<std::pair<float, std::uint32_t>> all(n);
        for (std::size_t r = 0; r < n; ++r) { float s = 0; for (std::size_t j = 0; j < d; ++j) { float df = q[i * d + j] - x[r * d + j]; s = std::fma(df, df, s); }
            if (noise > 0) s = float(double(s) * (1.0 + noise * ud(*g)));
            all[r] = {s, std::uint32_t(r)}; }
        const std::size_t m = std::min<std::size_t>(16, n);
        std::partial_sort(all.begin(), all.begin() + std::ptrdiff_t(m), all.end());
        for (std::size_t c = 0; c < 16; ++c) { od[i * 16 + c] = c < m ? all[c].first : __builtin_inff(); oi[i * 16 + c] = c < m ? all[c].second : 0xFFFFFFFFu; }
    }
    return true;
}
TEST(gpu_knn_logic) {
    task::pool pool(8);
    for (bool ties : {false, true}) for (std::size_t d : {8u, 32u, 64u}) {
        auto vs = random_vectors(3000 + 200, d, 3 + d, ties);
        std::vector<fvec> qs(vs.begin() + 3000, vs.end());
        ann::flat_index<fvec, ann::l2> exact; for (std::size_t i = 0; i < 3000; ++i) exact.add(i * 3 + 5, vs[i]);
        std::mt19937_64 ng(9); std::size_t fails = 0;
        const double g = 2.0 * double(d + 4) * std::ldexp(1.0, -24);
        struct kcase { const char* name; ann::gpu_knn_l2::knn_fn fn; bool failing; };
        kcase cases[] = {
            {"exact", [](const float* x, std::size_t n, std::size_t dd, const float* q, std::size_t nq, float* od, std::uint32_t* oi, double* up) { if (up) *up = 0; return cpu_knn16(x, n, dd, q, nq, od, oi, 0, nullptr); }, false},
            {"noisy (within the bound)", [&](const float* x, std::size_t n, std::size_t dd, const float* q, std::size_t nq, float* od, std::uint32_t* oi, double* up) { if (up) *up = 0; return cpu_knn16(x, n, dd, q, nq, od, oi, 0.9 * g, &ng); }, false},
            {"failing", [&](const float*, std::size_t, std::size_t, const float*, std::size_t, float*, std::uint32_t*, double*) { ++fails; return false; }, true},
        };
        for (auto& kc : cases) {
            ann::gpu_knn_l2 ix(kc.fn, d, 64);
            for (std::size_t i = 0; i < 3000; ++i) ix.add(i * 3 + 5, vs[i]);
            for (std::size_t k : {1u, 10u, 16u, 40u}) {
                ann::gpu_knn_l2::report rep;
                auto got = ix.search_batch(qs, k, &pool, &rep);
                bool ok = got.size() == qs.size();
                for (std::size_t q = 0; ok && q < qs.size(); ++q) ok = same(got[q], exact.search(qs[q], k, nullptr));
                CHECK_MSG(ok, "gpu_knn != flat_index: %s ties=%d d=%zu k=%zu", kc.name, int(ties), d, k);
                if (k > 16) CHECK(rep.device_calls == 0);   // more than the device returns: CPU path
                else if (kc.failing) CHECK(rep.fell_back);
                else { CHECK(!rep.fell_back && rep.device_calls == (qs.size() + 63) / 64); if (!ties && k < 16 && std::string(kc.name) == "exact") CHECK_MSG(rep.fallback_queries * 10 < qs.size(), "fallbacks %zu", rep.fallback_queries); }
                if (k == 16 && !kc.failing) CHECK(rep.fallback_queries == qs.size());   // k = 16 leaves no margin: every query takes the CPU path
                auto serial = ix.search_batch(qs, k, nullptr); for (std::size_t q = 0; q < got.size(); ++q) CHECK(same(serial[q], got[q]));
            }
        }
        CHECK(fails > 0);
    }
    { ann::gpu_knn_l2 ix(nullptr, 4); CHECK(ix.search_batch(std::vector<fvec>{{1, 2, 3, 4}}, 3)[0].empty()); CHECK_THROWS(ix.add(1, fvec(3, 0.f)));
      ix.add(7, fvec{1, 2, 3, 4}); auto h = ix.search_batch(std::vector<fvec>{{1, 2, 3, 4}}, 3); CHECK(h[0].size() == 1 && h[0][0].id == 7 && h[0][0].dist == 0.f); }
    { ann::gpu_knn_l2 ix(nullptr, 2); ix.add(1, fvec{1, 1}); ix.add(2, fvec{__builtin_nanf(""), 0}); for (std::size_t i = 0; i < 40; ++i) ix.add(10 + i, fvec{float(i), 0});
      ann::flat_index<fvec, ann::l2> ex; ex.add(1, fvec{1, 1}); ex.add(2, fvec{__builtin_nanf(""), 0}); for (std::size_t i = 0; i < 40; ++i) ex.add(10 + i, fvec{float(i), 0});
      CHECK(same(ix.search_batch(std::vector<fvec>{{0.5f, 0}}, 5)[0], ex.search(fvec{0.5f, 0}, 5))); }   // NaN data: CPU path, flat_index's order
}

TEST(gpu_knn_device) {
    std::string why; auto rt = graph::open_knn_device("cuda", &why);
    if (!rt) { std::printf("no CUDA device (%s): skipped\n", why.c_str()); std::exit(77); }
    task::pool pool(8);
    for (bool ties : {false, true}) for (std::size_t d : {7u, 32u, 64u}) {
        const std::size_t n = ties ? 30000 : 200000 + 37, nq = 1000;
        auto vs = random_vectors(n + nq, d, 21 + d, ties);
        ann::gpu_knn_l2 ix(graph::cuda_knn(rt), d, 512);
        ann::flat_index<fvec, ann::l2> exact;
        for (std::size_t i = 0; i < n; ++i) { ix.add(i + 1000, vs[i]); exact.add(i + 1000, vs[i]); }
        std::vector<fvec> qs(vs.begin() + std::ptrdiff_t(n), vs.end());
        for (std::size_t k : {1u, 10u, 16u}) {
            const auto before = rt->dispatches();
            ann::gpu_knn_l2::report rep; const double t0 = now_s();
            auto got = ix.search_batch(qs, k, &pool, &rep);
            const double tg = now_s() - t0;
            auto after = rt->dispatches();
            std::vector<std::vector<ann::hit>> want(nq); const double t1 = now_s();
            for (std::size_t q = 0; q < nq; ++q) want[q] = exact.search(qs[q], k, &pool);
            const double tc = now_s() - t1;
            bool ok = true; std::size_t bad = 0; for (std::size_t q = 0; q < nq; ++q) if (!same(got[q], want[q])) { ok = false; ++bad; }
            CHECK_MSG(ok, "device knn != flat_index: ties=%d d=%zu k=%zu (%zu queries differ)", int(ties), d, k, bad);
            CHECK(!rep.fell_back && rep.device_calls == 2);
            const std::size_t b0 = before.count("knn_l2_part") ? before.at("knn_l2_part") : 0;
            CHECK(after["knn_l2_part"] == b0 + 2);   // the kernel ran on the device, once per batch of 512
            if (!ties && k < 16) CHECK_MSG(rep.fallback_queries * 20 < nq, "fallbacks %zu", rep.fallback_queries);
            std::printf("  cuda knn: n=%zu d=%zu k=%zu: device path %.3f s (upload %.3f s), CPU flat_index (8 threads) %.3f s, %zu of %zu queries answered on the CPU\n", n, d, k, tg, rep.upload_s, tc, rep.fallback_queries, nq);
        }
        rt->knn_forget(ix.data());
    }
}

// forge::ann::ivf_flat: k-means determinism, nprobe = nlist is exact, the CPU scan against the matmul and device scans.
static std::vector<fvec> clustered_vectors(std::size_t n, std::size_t d, std::size_t centres, std::uint64_t seed) {
    std::mt19937_64 g(seed); std::normal_distribution<float> nd;
    std::vector<fvec> cs(centres, fvec(d)); for (auto& c : cs) for (auto& f : c) f = 4.0f * nd(g);
    std::vector<fvec> v(n, fvec(d)); for (auto& x : v) { const auto& c = cs[g() % centres]; for (std::size_t j = 0; j < d; ++j) x[j] = c[j] + nd(g); }
    return v;
}
TEST(ivf_logic) {
    task::pool pool(8);
    const std::size_t n = 20000, d = 24, nq = 200, nlist = 64;
    auto vs = clustered_vectors(n + nq, d, 40, 17);
    std::vector<float> flatv; for (std::size_t i = 0; i < n; ++i) flatv.insert(flatv.end(), vs[i].begin(), vs[i].end());
    auto c1 = ann::kmeans(flatv.data(), n, d, nlist, 10, 5, &pool), c2 = ann::kmeans(flatv.data(), n, d, nlist, 10, 5, nullptr);
    CHECK(c1 == c2);   // same centroids for any thread count
    CHECK_THROWS(ann::kmeans(flatv.data(), 10, d, 20, 3, 1));
    ann::ivf_flat ix(c1, d); ann::flat_index<fvec, ann::l2> exact;
    for (std::size_t i = 0; i < n; ++i) { ix.add(i * 2 + 1, vs[i]); exact.add(i * 2 + 1, vs[i]); }
    ix.finalize(&pool);
    std::size_t total = 0; for (std::size_t l = 0; l < nlist; ++l) total += ix.list_end(l) - ix.list_begin(l);
    CHECK(total == n);
    std::vector<fvec> qs(vs.begin() + std::ptrdiff_t(n), vs.end());
    std::size_t hitsum = 0;
    for (auto& q : qs) {
        CHECK(same(ix.search(q, 10, nlist), exact.search(q, 10)));   // every list probed: exact
        auto want = exact.search(q, 10); auto got = ix.search(q, 10, 8);
        for (auto& h : got) for (auto& w : want) if (h.id == w.id) { ++hitsum; break; }
    }
    CHECK_MSG(double(hitsum) / double(nq * 10) > 0.9, "recall@10 at nprobe 8 of 64 on clustered data: %.3f", double(hitsum) / double(nq * 10));
    // the matmul scan (a CPU f32 matmul standing in for the device) and a failing device scan both give the CPU scan's answer
    auto mm = [](const float* x, std::size_t r, std::size_t k, const float* w, std::size_t c, float* y) { cpu_matmul(x, r, k, w, c, y); return true; };
    std::size_t calls = 0; auto gm = ix.search_batch_matmul(qs, 10, 8, mm, &pool, &calls);
    std::size_t eq = 0; for (std::size_t i = 0; i < nq; ++i) eq += same(gm[i], ix.search(qs[i], 10, 8)) ? 1 : 0;
    CHECK_MSG(eq == nq, "matmul scan equal on %zu of %zu", eq, nq); CHECK(calls > 0);
    bool fb = false; auto gf = ix.search_batch_device(qs, 10, 8, [](const float*, std::size_t, std::size_t, const float*, std::size_t, const std::uint32_t*, std::size_t, float*, std::uint32_t*, double*) { return false; }, &pool, &fb);
    CHECK(fb); for (std::size_t i = 0; i < nq; ++i) CHECK(same(gf[i], ix.search(qs[i], 10, 8)));
    // a late add re-buckets
    ix.add(999999, qs[0]); CHECK(ix.search(qs[0], 1, 1)[0].id == 999999);
}
TEST(ivf_device) {
    std::string why; auto rt = graph::open_knn_device("cuda", &why);
    if (!rt) { std::printf("no CUDA device (%s): skipped\n", why.c_str()); std::exit(77); }
    task::pool pool(8);
    const std::size_t n = 300000, d = 32, nq = 1000, nlist = 548;
    auto vs = clustered_vectors(n + nq, d, 300, 23);
    std::vector<float> flatv; for (std::size_t i = 0; i < n / 20; ++i) flatv.insert(flatv.end(), vs[i * 20].begin(), vs[i * 20].end());
    ann::ivf_flat ix(ann::kmeans(flatv.data(), n / 20, d, nlist, 15, 3, &pool), d);
    for (std::size_t i = 0; i < n; ++i) ix.add(i, vs[i]);
    ix.finalize(&pool);
    std::vector<fvec> qs(vs.begin() + std::ptrdiff_t(n), vs.end());
    for (std::size_t np : {1u, 8u, 32u}) {
        const auto d0 = rt->dispatches(); bool fb = true; double up = 0; const double t0 = now_s();
        auto got = ix.search_batch_device(qs, 10, np, graph::cuda_ivf(rt), &pool, &fb, &up);
        const double tg = now_s() - t0; auto d1 = rt->dispatches();
        std::size_t eq = 0; const double t1 = now_s(); for (std::size_t i = 0; i < nq; ++i) eq += same(got[i], ix.search(qs[i], 10, np)) ? 1 : 0; const double tc = now_s() - t1;
        CHECK(!fb); CHECK(d1["ivf_l2_part"] == (d0.count("ivf_l2_part") ? d0.at("ivf_l2_part") : 0) + 1);
        CHECK_MSG(eq * 1000 >= nq * 999, "device ivf equal to the CPU scan on %zu of %zu (nprobe %zu)", eq, nq, np);
        std::printf("  cuda ivf: n=%zu nlist=%zu nprobe=%zu: device %.3f s (upload %.3f s), CPU scan (1 thread) %.3f s, equal on %zu of %zu\n", n, nlist, np, tg, up, tc, eq, nq);
    }
    rt->knn_forget(ix.data());
}
