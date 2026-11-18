// forge::ann: metrics, flat_index == brute force, HNSW recall and determinism, save/load, rerank, rank fusion.
// FORGE_TEST_SLOW=1 (ctest test_ann_slow, label slow) runs the 20k-vector recall cases at full size.
#include "check.hpp"
#include <forge/ann/ann.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
using namespace forge;
using fvec = std::vector<float>;

static bool slow() { const char* e = std::getenv("FORGE_TEST_SLOW"); return e && *e == '1'; }
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<fvec> random_vectors(std::size_t n, std::size_t d, std::uint64_t seed) {
    std::mt19937_64 g(seed); std::normal_distribution<float> nd;
    std::vector<fvec> v(n, fvec(d));
    for (auto& x : v) for (auto& f : x) f = nd(g);
    return v;
}

TEST(metrics) {
    fvec a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, b{2, 2, 3, 4, 5, 6, 7, 8, 9, 12};
    CHECK(ann::l2_squared{}.distance(a, b) == 1 + 4);
    CHECK(std::abs(ann::l2{}.distance(a, b) - std::sqrt(5.0f)) < 1e-6f);
    CHECK(ann::inner_product{}.distance(a, a) == -385.0f);
    CHECK(std::abs(ann::cosine{}.distance(a, a)) < 1e-6f);
    CHECK(ann::cosine{}.distance(a, fvec(10, 0.0f)) == 1.0f);
    std::vector<std::uint64_t> x{0xFFull, 0}, y{0x0Full, 1};
    CHECK(ann::hamming{}.distance(x, y) == 4 + 1);
    ann::sparse_vec s1{{1, 1.0f}, {5, 2.0f}, {9, 1.0f}}, s2{{1, 2.0f}, {7, 1.0f}, {9, 1.0f}};
    CHECK(std::abs(ann::weighted_jaccard{}.distance(s1, s2) - (1.0f - (1.0f + 0 + 1.0f) / (2.0f + 2.0f + 1.0f + 1.0f))) < 1e-6f);   // min: 1,0,1; max: 2,2,1,1
    CHECK(ann::weighted_jaccard{}.distance({}, {}) == 0.0f);
    static_assert(ann::metric<ann::l2, fvec> && ann::metric<ann::cosine, std::span<const float>> && ann::metric<ann::hamming, std::vector<std::uint64_t>>);
    static_assert(ann::batch_metric<ann::l2, fvec> && !ann::batch_metric<ann::cosine, fvec>);
    // minhash estimates Jaccard: two sets sharing 60 of 140 features -> J = 60/140
    std::vector<std::uint64_t> fa, fb;
    for (std::uint64_t i = 0; i < 100; ++i) fa.push_back(i);
    for (std::uint64_t i = 40; i < 140; ++i) fb.push_back(i);
    auto sa = ann::minhash_sketch(fa, 512), sb = ann::minhash_sketch(fb, 512);
    CHECK(sa == ann::minhash_sketch(fa, 512));
    CHECK(std::abs((1.0f - ann::minhash_jaccard{}.distance(sa, sb)) - 60.0f / 140.0f) < 0.06f);
    // weighted minhash estimates weighted Jaccard
    ann::sparse_vec wa, wb;
    for (std::uint64_t i = 0; i < 60; ++i) { wa.push_back({i, 1.0f + float(i % 4)}); wb.push_back({i + 20, 1.0f + float((i + 1) % 4)}); }
    const float wj = 1.0f - ann::weighted_jaccard{}.distance(wa, wb);
    const float est = 1.0f - ann::minhash_jaccard{}.distance(ann::wminhash_sketch(wa, 1024), ann::wminhash_sketch(wb, 1024));
    CHECK_MSG(std::abs(est - wj) < 0.05f, "wj=%f est=%f", wj, est);
    // codecs round-trip
    std::vector<std::byte> buf; ann::codec<fvec>::encode(a, buf); CHECK(ann::codec<fvec>::decode(buf) == a);
    buf.clear(); ann::codec<ann::sparse_vec>::encode(s1, buf); CHECK(ann::codec<ann::sparse_vec>::decode(buf) == s1);
    buf.clear(); ann::codec<ann::minhash_sig>::encode(sa, buf); CHECK(ann::codec<ann::minhash_sig>::decode(buf) == sa);
}

// a metric without a batch `distances`, to compare the two flat_index code paths
struct l2_plain { float distance(const fvec& a, const fvec& b) const { return ann::l2{}.distance(a, b); } };

TEST(flat_index_equals_brute_force) {
    // integer coordinates in a small range: many exact distance ties, so the (dist, id) order matters
    std::mt19937_64 g(3); const std::size_t n = 9000, d = 6;
    std::vector<fvec> vs(n, fvec(d)); for (auto& v : vs) for (auto& f : v) f = float(g() % 4);
    ann::flat_index<fvec, ann::l2> a; ann::flat_index<fvec, l2_plain> b;
    std::vector<std::uint64_t> ids(n);
    for (std::size_t i = 0; i < n; ++i) { ids[i] = (i * 2654435761u) % 100003; a.add(ids[i], vs[i]); b.add(ids[i], vs[i]); }
    for (std::size_t qi = 0; qi < 5; ++qi) {
        fvec q(d); for (auto& f : q) f = float(g() % 4);
        std::vector<ann::hit> all; for (std::size_t i = 0; i < n; ++i) all.push_back({ids[i], ann::l2{}.distance(q, vs[i])});
        std::sort(all.begin(), all.end(), ann::hit_before);
        for (std::size_t k : {1u, 10u, 100u, 20000u}) {
            std::vector<ann::hit> want(all.begin(), all.begin() + std::min(k, n));
            for (unsigned th : {1u, 2u, 3u, 8u, 16u}) {
                task::pool p(th);
                CHECK_MSG(a.search(q, k, &p) == want, "batch path k=%zu threads=%u", std::size_t(k), th);
                CHECK_MSG(b.search(q, k, &p) == want, "plain path k=%zu threads=%u", std::size_t(k), th);
            }
            CHECK(a.search(q, k) == want);
        }
    }
    // hamming and sparse vectors
    ann::flat_index<std::vector<std::uint64_t>, ann::hamming> h;
    std::vector<std::vector<std::uint64_t>> hv(3000, std::vector<std::uint64_t>(4));
    for (std::size_t i = 0; i < hv.size(); ++i) { for (auto& w : hv[i]) w = g(); h.add(i, hv[i]); }
    std::vector<ann::hit> all; for (std::size_t i = 0; i < hv.size(); ++i) all.push_back({i, ann::hamming{}.distance(hv[7], hv[i])});
    std::sort(all.begin(), all.end(), ann::hit_before); all.resize(10);
    task::pool p(8); CHECK(h.search(hv[7], 10, &p) == all); CHECK(all[0].id == 7 && all[0].dist == 0);
    CHECK(a.search(fvec(d), 0).empty());
    { ann::flat_index<fvec, ann::l2> empty; CHECK(empty.search(fvec(d), 5).empty()); }
}

static double recall_at(std::size_t k, const std::vector<std::vector<ann::hit>>& got, const std::vector<std::vector<ann::hit>>& want) {
    double s = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        std::set<std::uint64_t> w; for (std::size_t j = 0; j < std::min(k, want[i].size()); ++j) w.insert(want[i][j].id);
        std::size_t hitc = 0; for (std::size_t j = 0; j < std::min(k, got[i].size()); ++j) hitc += w.count(got[i][j].id);
        s += double(hitc) / double(std::max<std::size_t>(1, w.size()));
    }
    return s / double(got.size());
}

TEST(hnsw_recall_random_64d) {
    const std::size_t n = slow() ? 20000 : 5000, nq = 200;
    auto vs = random_vectors(n + nq, 64, 42);
    ann::hnsw<fvec, ann::l2> ix;           // default parameters: M=16, efConstruction=200, ef_search=64
    ann::flat_index<fvec, ann::l2> exact;
    for (std::size_t i = 0; i < n; ++i) { ix.add(i, vs[i]); exact.add(i, vs[i]); }
    task::pool p(8);
    const double t0 = now_s(); ix.build(&p); const double t1 = now_s();
    std::vector<std::vector<ann::hit>> want;
    for (std::size_t i = 0; i < nq; ++i) want.push_back(exact.search(vs[n + i], 10, &p));
    // Uniform Gaussian vectors in 64 dimensions are about the hardest case for a graph index. Recall is recorded at the default
    // ef (64) and at ef 128 and 256; the gate (>= 0.95) is at ef = 256 (see docs/ann.md for the measured curve).
    double r64 = 0, r256 = 0;
    for (std::size_t ef : {std::size_t(0), std::size_t(128), std::size_t(256)}) {
        std::vector<std::vector<ann::hit>> got;
        for (std::size_t i = 0; i < nq; ++i) got.push_back(ix.search(vs[n + i], 10, ef));
        const double r = recall_at(10, got, want);
        if (ef == 0) r64 = r; if (ef == 256) r256 = r;
        std::printf("  hnsw random 64-d n=%zu: build %.2f s (8 threads), recall@10 = %.4f at ef=%zu, edges=%zu\n", n, t1 - t0, r, ef ? ef : ix.params().ef_search, ix.edges());
        for (auto& h : got) { CHECK(h.size() == 10); CHECK(std::is_sorted(h.begin(), h.end(), ann::hit_before)); }
    }
    CHECK_MSG(r256 >= 0.95, "recall at ef=256: %.4f", r256);
    CHECK_MSG(r64 >= 0.8, "recall at default ef: %.4f", r64);
}

TEST(hnsw_recall_hamming) {
    const std::size_t n = slow() ? 20000 : 8000, nq = 200, words = 4, nc = 100;
    std::mt19937_64 g(11);
    std::vector<std::vector<std::uint64_t>> centers(nc, std::vector<std::uint64_t>(words));
    for (auto& c : centers) for (auto& w : c) w = g();
    auto noisy = [&](const std::vector<std::uint64_t>& c, int flips) { auto v = c; for (int f = 0; f < flips; ++f) v[g() % words] ^= 1ull << (g() % 64); return v; };
    ann::hnsw<std::vector<std::uint64_t>, ann::hamming> ix;
    ann::flat_index<std::vector<std::uint64_t>, ann::hamming> exact;
    for (std::size_t i = 0; i < n; ++i) { auto v = noisy(centers[g() % nc], 30); ix.add(i, v); exact.add(i, std::move(v)); }
    task::pool p(8); ix.build(&p);
    std::vector<std::vector<ann::hit>> got, want;
    for (std::size_t i = 0; i < nq; ++i) { auto q = noisy(centers[g() % nc], 30); got.push_back(ix.search(q, 10)); want.push_back(exact.search(q, 10, &p)); }
    // many equal distances: count a hit when its distance is within the exact 10th distance
    double s = 0;
    for (std::size_t i = 0; i < nq; ++i) { std::size_t ok = 0; for (auto& h : got[i]) ok += h.dist <= want[i].back().dist; s += double(ok) / 10.0; }
    s /= double(nq);
    std::printf("  hnsw hamming n=%zu: recall@10 = %.4f (distance-tie aware), id-recall %.4f\n", n, s, recall_at(10, got, want));
    CHECK_MSG(s >= 0.95, "recall %.4f", s);
}

TEST(hnsw_deterministic_for_any_thread_count) {
    auto vs = random_vectors(3000, 16, 5);
    std::uint64_t ref_hash = 0; std::vector<std::vector<ann::hit>> ref;
    for (unsigned th : {0u, 1u, 2u, 3u, 8u, 16u}) {   // 0: no pool
        ann::hnsw<fvec, ann::l2> ix({.M = 8, .ef_construction = 60, .ef_search = 32, .seed = 99});
        for (std::size_t i = 0; i < vs.size(); ++i) ix.add(i * 3 + 1, vs[i]);
        if (th == 0) ix.build(nullptr); else { task::pool p(th); ix.build(&p); }
        std::vector<std::vector<ann::hit>> res;
        for (std::size_t q = 0; q < 50; ++q) res.push_back(ix.search(vs[q * 7], 10));
        if (th == 0) { ref_hash = ix.graph_hash(); ref = res; }
        else { CHECK_MSG(ix.graph_hash() == ref_hash, "graph differs at %u threads", th); CHECK_MSG(res == ref, "results differ at %u threads", th); }
    }
    // a different seed gives a different graph; incremental build (two add/build rounds) is also deterministic
    ann::hnsw<fvec, ann::l2> other({.M = 8, .ef_construction = 60, .ef_search = 32, .seed = 100});
    for (std::size_t i = 0; i < vs.size(); ++i) other.add(i * 3 + 1, vs[i]);
    other.build(); CHECK(other.graph_hash() != ref_hash);
    auto inc = [&](unsigned th) {
        task::pool p(th); ann::hnsw<fvec, ann::l2> ix({.M = 8, .ef_construction = 60, .seed = 4});
        for (std::size_t i = 0; i < 1000; ++i) ix.add(i, vs[i]);
        ix.build(&p);
        for (std::size_t i = 1000; i < 2000; ++i) ix.add(i, vs[i]);
        ix.build(&p); return ix.graph_hash(); };
    CHECK(inc(1) == inc(8));
}

TEST(hnsw_save_load_roundtrip) {
    auto vs = random_vectors(2000, 24, 8);
    ann::hnsw<fvec, ann::cosine> ix({.M = 12, .ef_construction = 80, .ef_search = 48, .seed = 21});
    for (std::size_t i = 0; i < vs.size(); ++i) ix.add(1000 + i, vs[i]);
    ix.build();
    auto c = db::connection::open("");
    ix.save(c, "ann_test");
    auto back = ann::hnsw<fvec, ann::cosine>::load(c, "ann_test");
    CHECK(back.size() == ix.size() && back.built() == ix.built());
    CHECK(back.graph_hash() == ix.graph_hash());
    CHECK(back.params().M == 12 && back.params().ef_search == 48 && back.params().seed == 21);
    for (std::size_t q = 0; q < 40; ++q) CHECK(back.search(vs[q * 13], 10) == ix.search(vs[q * 13], 10));
    ix.save(c, "ann_test");   // replaces the table
    CHECK(c.scalar<std::int64_t>("SELECT count(*) FROM ann_test WHERE kind = 'node'") == 2000);
    // minhash signatures and sparse vectors persist too
    ann::hnsw<ann::minhash_sig, ann::minhash_jaccard> mh; std::vector<ann::minhash_sig> sigs;
    for (std::uint64_t i = 0; i < 300; ++i) { std::vector<std::uint64_t> f; for (std::uint64_t j = 0; j < 20; ++j) f.push_back((i / 10) * 100 + j); sigs.push_back(ann::minhash_sketch(f, 32)); mh.add(i, sigs.back()); }
    mh.build(); mh.save(c, "ann_mh");
    auto mh2 = ann::hnsw<ann::minhash_sig, ann::minhash_jaccard>::load(c, "ann_mh");
    CHECK(mh2.graph_hash() == mh.graph_hash() && mh2.search(sigs[5], 5) == mh.search(sigs[5], 5));
    // corruption is reported, not trusted
    c.exec("UPDATE ann_test SET data = x'00' WHERE kind = 'graph'");
    CHECK_THROWS((ann::hnsw<fvec, ann::cosine>::load(c, "ann_test")));
    CHECK_THROWS((ann::hnsw<fvec, ann::cosine>::load(c, "no_such_table")));
    // saving with queued nodes is refused
    ann::hnsw<fvec, ann::l2> pending; pending.add(1, fvec{1, 2}); CHECK_THROWS(pending.save(c, "x"));
    CHECK(pending.search(fvec{1, 2}, 1).empty());
}

TEST(rerank_and_fusion) {
    auto vs = random_vectors(500, 8, 77);
    ann::flat_index<fvec, ann::l2> ix; for (std::size_t i = 0; i < vs.size(); ++i) ix.add(i, vs[i]);
    auto cand = ix.search(vs[0], 50);
    auto re = ann::rerank<fvec>(cand, vs[0], [&](std::uint64_t id) -> const fvec& { return vs[id]; }, ann::cosine{}, 5);
    CHECK(re.size() == 5); CHECK(std::is_sorted(re.begin(), re.end(), ann::hit_before));
    CHECK(re[0].id == 0 || re[0].dist < 1e-5f);
    auto f = ann::reciprocal_rank_fusion({{5, 3, 9}, {3, 7, 5, 5}}, 10);
    CHECK(f.size() == 4);
    CHECK(f[0].id == 3);   // rank 2 + rank 1 beats 5 (rank 1 + rank 3)
    CHECK(std::abs(f[0].score - (1.0 / 62 + 1.0 / 61)) < 1e-12 && std::abs(f[1].score - (1.0 / 61 + 1.0 / 63)) < 1e-12);
    CHECK(ann::reciprocal_rank_fusion({{1, 2}, {2, 1}}, 1)[0].id == 1);   // exact tie: smaller id first
    CHECK(ann::reciprocal_rank_fusion({{1, 2}}, 1).size() == 1);
}
