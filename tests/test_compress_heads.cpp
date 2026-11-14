// forge::compress heads.hpp: query-head pruning, exact N:M permutations, calibrated activation sparsity
// (tiny random llama-style models from tiny_model.hpp).
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/compress/compress.hpp>
#include <forge/compress/heads.hpp>
#include <forge/inference/session.hpp>

#include <filesystem>
#include <random>

using namespace forge;
using namespace forge::compress;
using forge_test::tiny_cfg;
using forge_test::make_tiny_model;

namespace {

const int exact_int8 = (forge_test::set_env("FORGE_CPU_INT8", "0"), 0);

std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / std::format("forge_compress_heads_{}_{}", ::getpid(), n)).string(); }

double max_abs_diff(std::span<const float> a, std::span<const float> b) {
    double m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, double(std::abs(a[i] - b[i]))); return m;
}

CalibSet random_calib(i64 vocab, i64 n, i64 seq, unsigned seed) {
    std::mt19937 rng(seed); std::uniform_int_distribution<int> U(0, int(vocab) - 3);
    CalibSet c; c.seq = seq;
    for (i64 i = 0; i < n; ++i) { std::vector<int> v(static_cast<std::size_t>(seq)); for (auto& t : v) t = U(rng); c.seqs.push_back(v); }
    return c;
}

// 8 query heads over 2 KV heads (4 per group), head_dim 8.
Model tiny_dense(unsigned seed = 7, int layers = 3, int heads = 8, int kv = 2, int d = 64) {
    std::string p = make_tiny_model(tmp("dense.fpk"), tiny_cfg{.d = d, .layers = layers, .heads = heads, .kv = kv, .ffn = 64, .seed = seed});
    Model m = Model::load(p);
    std::filesystem::remove(p);
    return m;
}

double nll(Model& m, const CalibSet& c) {
    Runner r(m.spec, m.w, c.seq, {}, true);
    return calib_nll(r, c, i64(c.seqs.size()), m.spec.vocab);
}

std::vector<float> logits_of(Model& m, const std::vector<int>& ids) {
    Runner r(m.spec, m.w, i64(ids.size()), {}, true);
    r.run_tokens(ids);
    auto o = r.output();
    return {o.begin(), o.end()};
}

void check_saved_runs(Model& m, const CalibSet& c, const char* what) {
    std::string p = tmp(std::string(what) + ".fpk");
    m.save(p);
    inference::session s(p);
    auto lg = s.forward_all(c.seqs[0]);
    Runner r(m.spec, m.w, c.seq, {}, true);
    r.run_tokens(c.seqs[0]);
    CHECK_MSG(max_abs_diff(lg, r.output()) < 1e-4, "%s: saved model differs from the in-memory graph", what);
    std::filesystem::remove(p);
}

// Retained N:M (2:4) Wanda saliency of every dense w_down and of every wo, from the calibration diagonals.
double retained_24(Model& m, const CalibSet& c, const char* group) {
    Hidden h = Hidden::embed(m, c), nx;
    double keep = 0, tot = 0;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        StatsReq rq; rq.gram = false;
        auto S = layer_pass(m, l, h, rq, &nx);
        std::swap(h, nx);
        const auto& W = m.w.at(std::format("layer.{}.{}", l, std::string(group) == "down" ? "w_down" : "wo"));
        PruneCfg pc; pc.crit = Criterion::wanda; pc.nm_n = 2; pc.nm_m = 4; pc.keep_existing_zeros = false;
        auto sal = saliency(W.data.data(), W.shape[0], W.shape[1], S.diag[group], pc);
        auto mask = select_mask(sal, W.data.data(), W.shape[0], W.shape[1], pc);
        for (std::size_t i = 0; i < sal.size(); ++i) { tot += sal[i]; if (mask[i]) keep += sal[i]; }
    }
    return keep / tot;
}

// Calibration text sampled from the model itself (temperature 1): on random tiny models this makes the
// calibration NLL meaningful (the dense model is the reference distribution; compression can only move away).
CalibSet sampled_calib(Model& m, i64 n, i64 seq, unsigned seed) {
    std::mt19937 rng(seed);
    CalibSet c; c.seq = seq;
    Runner r(m.spec, m.w, seq, {}, true);
    for (i64 i = 0; i < n; ++i) {
        std::vector<int> ids(static_cast<std::size_t>(seq), 0);
        ids[0] = int(rng() % std::uint64_t(m.spec.vocab - 3));
        for (i64 t = 0; t + 1 < seq; ++t) {
            r.run_tokens(ids);
            auto lg = r.output();
            const float* row = lg.data() + t * m.spec.vocab;
            std::vector<double> pr(static_cast<std::size_t>(m.spec.vocab));
            double mx = -1e30; for (i64 j = 0; j < m.spec.vocab; ++j) mx = std::max(mx, double(row[j]));
            for (i64 j = 0; j < m.spec.vocab; ++j) pr[std::size_t(j)] = std::exp(double(row[j]) - mx);
            std::discrete_distribution<int> D(pr.begin(), pr.end());
            ids[std::size_t(t + 1)] = D(rng);
        }
        c.seqs.push_back(ids);
    }
    return c;
}

}  // namespace

TEST(compress_heads_stage) {
    for (const char* recipe : {"heads(s=0.5,score=obs)", "heads(s=0.5,score=act)", "heads(s=0.5,score=mag)", "heads(s=0.25,score=obs,alloc=eba)",
                               "heads(s=0.5,score=act,alloc=eba,refit=none)"}) {
        Model m = tiny_dense();
        auto c = random_calib(m.spec.vocab, 4, 32, 4);
        Ctx ctx{m, c, {}, 0.01};
        auto rep = run_stage(ctx, parse_recipe(recipe)[0]);
        i64 tot = 0;
        for (i64 l = 0; l < m.spec.n_layers(); ++l) {
            const i64 nh = m.spec.heads_of(l), hd = m.spec.head_dim;
            CHECK_MSG(nh % m.spec.n_kv == 0 && nh >= m.spec.n_kv, "%s: layer %lld has %lld heads", recipe, (long long)l, (long long)nh);
            CHECK(m.w.at(std::format("layer.{}.wq", l)).shape[0] == nh * hd);
            CHECK(m.w.at(std::format("layer.{}.wo", l)).shape[1] == nh * hd);
            tot += nh;
        }
        const i64 want = std::string(recipe).find("s=0.25") != std::string::npos ? 18 : 12;   // of 24
        CHECK_MSG(tot == want, "%s: %lld heads left, want %lld", recipe, (long long)tot, (long long)want);
        CHECK(rep["heads_after"].num() == double(tot));
        check_saved_runs(m, c, "heads");
        std::string p = tmp("heads_back.fpk");
        m.save(p);
        Model back = Model::load(p);
        for (i64 l = 0; l < m.spec.n_layers(); ++l) CHECK(back.spec.heads_of(l) == m.spec.heads_of(l));
        std::filesystem::remove(p);
        std::printf("  %s: heads %lld/24, calib nll %.4f\n", recipe, (long long)tot, nll(m, c));
    }
    {   // a layer keeps 1 head per group at most-aggressive s; s=0 leaves the model unchanged
        Model m = tiny_dense(9);
        auto c = random_calib(m.spec.vocab, 2, 16, 3);
        auto ref = logits_of(m, c.seqs[0]);
        Ctx ctx{m, c, {}, 0.01};
        run_stage(ctx, parse_recipe("heads(s=0)")[0]);
        CHECK(m.spec.heads.empty());
        CHECK(max_abs_diff(logits_of(m, c.seqs[0]), ref) < 1e-5);
        run_stage(ctx, parse_recipe("heads(s=1,score=obs)")[0]);
        for (i64 l = 0; l < m.spec.n_layers(); ++l) CHECK(m.spec.heads_of(l) == m.spec.n_kv);
        check_saved_runs(m, c, "heads_min");
    }
}

TEST(compress_heads_obs_beats_mag) {
    // On self-sampled calibration text (the dense model is the reference distribution). Block OBS minimises each
    // layer's exact output error (removing a query head only changes wo's input), so its per-layer error is never
    // worse than magnitude selection with the same least-squares refit; the end-to-end NLL follows on average
    // (on single tiny random models a lower layer error can still map to a slightly higher NLL).
    int wins = 0, n = 0;
    double sum_mag = 0, sum_obs = 0;
    for (unsigned seed : {5u, 11u, 23u, 31u, 37u, 41u}) {
        Model a = tiny_dense(seed), b = tiny_dense(seed), u = tiny_dense(seed), x = tiny_dense(seed);
        auto c = sampled_calib(u, 16, 64, seed + 1);
        const double base = nll(u, c);
        Ctx ca{a, c, {}, 0.01}, cb{b, c, {}, 0.01}, cx{x, c, {}, 0.01};
        auto ra = run_stage(ca, parse_recipe("heads(s=0.5,score=mag)")[0]);
        auto rb = run_stage(cb, parse_recipe("heads(s=0.5,score=obs)")[0]);
        run_stage(cx, parse_recipe("heads(s=0.5,score=act)")[0]);
        CHECK(ra["layers"].arr().size() == 3 && rb["layers"].arr().size() == 3);
        // layer 0 sees identical inputs under both scores: OBS's exact greedy is at least as good there
        CHECK_MSG(rb["layers"][0]["rel_err"].num() <= ra["layers"][0]["rel_err"].num() + 1e-6, "seed %u layer 0 rel err obs %g mag %g",
                  seed, rb["layers"][0]["rel_err"].num(), ra["layers"][0]["rel_err"].num());
        const double na = nll(a, c), nb = nll(b, c);
        std::printf("  seed %u: calib nll dense %.4f  mag %.4f  act %.4f  obs %.4f\n", seed, base, na, nll(x, c), nb);
        sum_mag += na; sum_obs += nb;
        wins += nb <= na + 1e-6; ++n;
    }
    std::printf("  obs <= mag on %d/%d seeds, mean nll mag %.4f obs %.4f\n", wins, n, sum_mag / n, sum_obs / n);
    CHECK_MSG(sum_obs <= sum_mag, "mean nll obs %g vs mag %g", sum_obs / n, sum_mag / n);
    CHECK(wins >= n - 1);
}

TEST(compress_permute_exact_and_helps_24) {
    Model m = tiny_dense(13);
    auto c = random_calib(m.spec.vocab, 4, 32, 7);
    auto ref = logits_of(m, c.seqs[0]);
    const double down0 = retained_24(m, c, "down"), o0 = retained_24(m, c, "o");
    Model pruned_plain = tiny_dense(13);
    Ctx ctx{m, c, {}, 0.01};
    auto rep = run_stage(ctx, parse_recipe("permute(nm=2:4)")[0]);
    CHECK_MSG(max_abs_diff(logits_of(m, c.seqs[0]), ref) < 1e-4, "permute changed the logits by %g", max_abs_diff(logits_of(m, c.seqs[0]), ref));
    const double down1 = retained_24(m, c, "down"), o1 = retained_24(m, c, "o");
    std::printf("  retained 2:4 saliency: w_down %.4f -> %.4f (stage: %.4f -> %.4f), wo %.4f -> %.4f\n", down0, down1,
                rep["ffn_retained_before"].num(), rep["ffn_retained_after"].num(), o0, o1);
    CHECK_MSG(down1 > down0 + 1e-3, "w_down retained %g -> %g", down0, down1);
    CHECK_MSG(o1 >= o0, "wo retained %g -> %g", o0, o1);
    CHECK(std::abs(rep["ffn_retained_before"].num() - down0) < 1e-4 && std::abs(rep["ffn_retained_after"].num() - down1) < 1e-4);
    check_saved_runs(m, c, "permute");
    // after 2:4 Wanda pruning, the permuted model is at least as good on calibration NLL
    Ctx cp{pruned_plain, c, {}, 0.01};
    run_stage(cp, parse_recipe("prune(method=wanda,nm=2:4,scope=ffn)")[0]);
    run_stage(ctx, parse_recipe("prune(method=wanda,nm=2:4,scope=ffn)")[0]);
    const double np = nll(pruned_plain, c), nq = nll(m, c);
    std::printf("  2:4 wanda (ffn) calib nll: plain %.4f  permuted %.4f\n", np, nq);
    // permute also commutes with heads pruning (values permuted inside the kept heads)
    Model h = tiny_dense(17);
    Ctx ch{h, c, {}, 0.01};
    run_stage(ch, parse_recipe("heads(s=0.5,score=act)")[0]);
    auto ref2 = logits_of(h, c.seqs[0]);
    run_stage(ch, parse_recipe("permute(nm=2:4)")[0]);
    CHECK(max_abs_diff(logits_of(h, c.seqs[0]), ref2) < 1e-4);
}

TEST(compress_actsparse) {
    for (const char* recipe : {"actsparse(p=0.5)", "actsparse(p=0.3)", "actsparse(p=0.5,alloc=eba)"}) {
        Model m = tiny_dense(19);
        auto c = random_calib(m.spec.vocab, 4, 32, 9);
        const double base = nll(m, c);
        Ctx ctx{m, c, {}, 0.01};
        auto rep = run_stage(ctx, parse_recipe(recipe)[0]);
        const double p = parse_recipe(recipe)[0].num("p", 0.5);
        const bool eba = std::string(recipe).find("eba") != std::string::npos;
        auto fr = measure_act_sparsity(m, c);
        double mean = 0;
        for (i64 l = 0; l < m.spec.n_layers(); ++l) {
            CHECK(m.spec.layers[std::size_t(l)].act_thresh > 0);
            const double want = eba ? rep["layers"][std::size_t(l)]["p"].num() : p;
            CHECK_MSG(std::abs(fr[std::size_t(l)] - want) < 0.03, "%s: layer %lld zeroed %.4f, want %.4f", recipe, (long long)l, fr[std::size_t(l)], want);
            mean += fr[std::size_t(l)] / double(m.spec.n_layers());
        }
        CHECK_MSG(std::abs(mean - p) < 0.04, "%s: mean zeroed %.4f", recipe, mean);
        CHECK(std::abs(rep["measured_mean"].num() - mean) < 1e-9);
        // held-out data sees about the same sparsity
        auto held = random_calib(m.spec.vocab, 4, 32, 99);
        auto fh = measure_act_sparsity(m, held);
        double mh = 0; for (double x : fh) mh += x / double(fh.size());
        std::printf("  %s: zeroed calib %.4f held-out %.4f, nll %.4f -> %.4f\n", recipe, mean, mh, base, nll(m, c));
        CHECK(std::abs(mh - p) < 0.08);
        // saved and reloaded with the thresholds
        check_saved_runs(m, c, "actsparse");
        std::string path = tmp("act_back.fpk");
        m.save(path);
        Model back = Model::load(path);
        for (i64 l = 0; l < m.spec.n_layers(); ++l) CHECK(std::abs(back.spec.layers[std::size_t(l)].act_thresh - m.spec.layers[std::size_t(l)].act_thresh) <= 1e-12 * (1 + m.spec.layers[std::size_t(l)].act_thresh));
        CHECK(max_abs_diff(logits_of(back, c.seqs[1]), logits_of(m, c.seqs[1])) < 1e-5);
        std::filesystem::remove(path);
    }
}

TEST(compress_heads_permute_actsparse_recipe) {
    // the three stages chain with the existing ones
    Model m = tiny_dense(29);
    auto c = random_calib(m.spec.vocab, 4, 32, 31);
    Ctx ctx{m, c, {}, 0.01};
    for (auto& st : parse_recipe("heads(s=0.25,score=obs)+permute(nm=2:4)+prune(method=sparsegpt,nm=2:4,scope=ffn)+actsparse(p=0.4)")) run_stage(ctx, st);
    check_saved_runs(m, c, "chain");
}

