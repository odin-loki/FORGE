// forge::ann::gpu_flat_l2: batched exact search through a matmul (a device's, or a CPU stand-in) equals flat_index::search.
// gpu_flat_logic runs everywhere (a plain CPU matmul, a noisy one within the error bound, a failing one); gpu_flat_device uses
// graph::device_registry("cuda") and exits 77 without a device.
#include "check.hpp"
#include <forge/graph/ann_gpu.hpp>
#include <chrono>
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
