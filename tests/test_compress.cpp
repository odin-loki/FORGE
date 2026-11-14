// forge::compress: pruning, refits, allocation, structured / depth / MoE compression and evaluation.
// Unit tests run on tiny random models (dense llama-style from tiny_model.hpp, and a tiny Qwen3-MoE built here).
// With FORGE_COMPRESS_MODELS=<dir of imported .fpk files> the integration tests also run on real checkpoints
// (qwen3-0.6b, tinyllama-sparse24, tinymixtral-qwen3moe .f32.fpk).
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/compress/compress.hpp>
#include <forge/compress/eval.hpp>
#include <forge/inference/session.hpp>

#include <filesystem>
#include <random>

using namespace forge;
using namespace forge::compress;
using forge_test::tiny_cfg;
using forge_test::make_tiny_model;

namespace {

// Exact quantised matmuls (FORGE reads FORGE_CPU_INT8 once per process): compressed weights are compared exactly.
const int exact_int8 = (forge_test::set_env("FORGE_CPU_INT8", "0"), 0);

std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / std::format("forge_compress_{}_{}", ::getpid(), n)).string(); }

double max_abs_diff(std::span<const float> a, std::span<const float> b) {
    double m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, double(std::abs(a[i] - b[i]))); return m;
}

la::Sym random_spd(i64 n, unsigned seed, i64 rows = 0) {
    std::mt19937 rng(seed); std::normal_distribution<float> N(0, 1);
    if (!rows) rows = 3 * n;
    std::vector<float> X(static_cast<std::size_t>(rows * n));
    for (auto& x : X) x = N(rng);
    for (i64 r = 0; r < rows; ++r) for (i64 j = 0; j < n; ++j) X[std::size_t(r * n + j)] *= 1.f + float(j % 5);   // uneven scales
    la::Gram g(n); g.add(X.data(), rows);
    return g.H;
}

std::vector<float> random_w(i64 rows, i64 cols, unsigned seed) {
    std::mt19937 rng(seed); std::normal_distribution<float> N(0, 1);
    std::vector<float> w(static_cast<std::size_t>(rows * cols)); for (auto& x : w) x = N(rng); return w;
}

CalibSet random_calib(i64 vocab, i64 n, i64 seq, unsigned seed) {
    std::mt19937 rng(seed); std::uniform_int_distribution<int> U(0, int(vocab) - 3);
    CalibSet c; c.seq = seq;
    for (i64 i = 0; i < n; ++i) { std::vector<int> v(static_cast<std::size_t>(seq)); for (auto& t : v) t = U(rng); c.seqs.push_back(v); }
    return c;
}

// A tiny random Qwen3-MoE (QK-norm, 4 experts, top-2) written through Model::save.
Model tiny_moe(unsigned seed = 3) {
    std::string base = make_tiny_model(tmp("moe_base.fpk"), tiny_cfg{.d = 32, .layers = 3, .heads = 4, .kv = 2, .ffn = 48, .seed = seed});
    Model m = Model::load(base);
    std::filesystem::remove(base);
    std::mt19937 rng(seed); std::normal_distribution<float> N(0, 1);
    m.spec.qk_norm = true; m.spec.family = "qwen3";
    const i64 d = m.spec.d_model, E = 4, f = 24, hd = m.spec.head_dim;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) {
        auto& L = m.spec.layers[std::size_t(l)];
        for (const char* n : {"w_gate", "w_up", "w_down"}) m.w.t.erase(std::format("layer.{}.{}", l, n));
        L.ffn = LayerSpec::Ffn::moe; L.moe_k = 2; L.moe_norm = true;
        auto put = [&](const std::string& n, shape_t s, float sd, float mean = 0) { tensor t(s); for (auto& x : t.data) x = mean + sd * N(rng); m.w.t[n] = t; };
        const std::string p = std::format("layer.{}.", l);
        put(p + "q_norm", {hd}, .1f, 1); put(p + "k_norm", {hd}, .1f, 1);
        put(p + "router", {E, d}, 1.f / std::sqrt(float(d)));
        put(p + "experts.w_gate", {E, f, d}, 1.f / std::sqrt(float(d)));
        put(p + "experts.w_up", {E, f, d}, 1.f / std::sqrt(float(d)));
        put(p + "experts.w_down", {E, d, f}, 1.f / std::sqrt(float(f)));
    }
    std::string p = tmp("moe.fpk");
    m.save(p);
    Model r = Model::load(p);
    std::filesystem::remove(p);
    return r;
}

Model tiny_dense(unsigned seed = 7, int layers = 3) {
    std::string p = make_tiny_model(tmp("dense.fpk"), tiny_cfg{.d = 32, .layers = layers, .heads = 4, .kv = 2, .ffn = 64, .seed = seed});
    Model m = Model::load(p);
    std::filesystem::remove(p);
    return m;
}

// Mean NLL of the model on a calibration set, through FORGE graphs (the model's live weights).
double nll(Model& m, const CalibSet& c) {
    Runner r(m.spec, m.w, c.seq, {}, true);
    return calib_nll(r, c, i64(c.seqs.size()), m.spec.vocab);
}

// The saved model runs in a FORGE session and matches the in-memory graph.
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

}  // namespace

// ------------------------------------------------------------------------------------------------ linear algebra
TEST(compress_cholesky_and_inverse) {
    for (i64 n : {5, 97, 230}) {
        auto H = random_spd(n, unsigned(n));
        la::Sym L = H;
        CHECK(la::cholesky(L, 32));
        double e = 0;   // L L^T == H
        for (i64 i = 0; i < n; ++i) for (i64 j = 0; j < n; ++j) { double s = 0; for (i64 k = 0; k < n; ++k) s += L(i, k) * L(j, k); e = std::max(e, std::abs(s - H(i, j)) / (1 + std::abs(H(i, j)))); }
        CHECK_MSG(e < 1e-9, "n=%lld chol err %g", (long long)n, e);
        auto Hi = la::inverse_from_cholesky(L);
        double ei = 0;
        for (i64 i = 0; i < n; ++i) for (i64 j = 0; j < n; ++j) { double s = 0; for (i64 k = 0; k < n; ++k) s += H(i, k) * Hi(k, j); ei = std::max(ei, std::abs(s - (i == j))); }
        CHECK_MSG(ei < 1e-7, "n=%lld inverse err %g", (long long)n, ei);
        auto U = la::inverse_upper_cholesky(H, 0.0);
        double eu = 0;
        for (i64 i = 0; i < n; ++i) for (i64 j = 0; j < n; ++j) { double s = 0; for (i64 k = 0; k < n; ++k) s += U(k, i) * U(k, j); eu = std::max(eu, std::abs(s - Hi(i, j)) / (1 + std::abs(Hi(i, j)))); }
        CHECK_MSG(eu < 1e-6, "n=%lld U^T U vs H^-1 err %g", (long long)n, eu);
        // FORGE's serial reference agrees
        linalg::mat<double> A(static_cast<std::size_t>(n), std::size_t(n));
        for (i64 i = 0; i < n; ++i) for (i64 j = 0; j < n; ++j) A(static_cast<std::size_t>(i), std::size_t(j)) = H(i, j);
        auto Lr = linalg::cholesky(A);
        double er = 0; for (i64 i = 0; i < n; ++i) for (i64 j = 0; j <= i; ++j) er = std::max(er, std::abs(Lr(static_cast<std::size_t>(i), std::size_t(j)) - L(i, j)));
        CHECK_MSG(er < 1e-8, "n=%lld vs forge::linalg::cholesky %g", (long long)n, er);
    }
}

TEST(compress_gram_matches_naive) {
    const i64 rows = 70, d = 9;
    auto X = random_w(rows, d, 2);
    la::Gram g(d); g.add(X.data(), 40); g.add(X.data() + 40 * d, rows - 40);
    double e = 0;
    for (i64 i = 0; i < d; ++i) for (i64 j = 0; j < d; ++j) { double s = 0; for (i64 r = 0; r < rows; ++r) s += double(X[std::size_t(r * d + i)]) * X[std::size_t(r * d + j)]; e = std::max(e, std::abs(s - g.H(i, j))); }
    CHECK_MSG(e < 1e-3, "gram err %g", e);
    CHECK(g.count == rows);
}

TEST(compress_ls_refit_is_least_squares) {
    // with all columns kept, the refit returns W itself; with a subset it beats the plain column drop
    const i64 rows = 6, n = 20;
    auto H = random_spd(n, 4);
    auto W = random_w(rows, n, 5);
    std::vector<i64> all(static_cast<std::size_t>(n)); std::iota(all.begin(), all.end(), 0);
    auto Wa = la::ls_refit_columns(W.data(), rows, H, all, 0.0);
    CHECK(max_abs_diff(Wa, W) < 1e-3);
    std::vector<i64> S{0, 2, 3, 7, 11, 12, 19};
    auto Ws = la::ls_refit_columns(W.data(), rows, H, S, 0.0);
    std::vector<float> full(W.size(), 0.f), drop(W.size(), 0.f);
    for (i64 r = 0; r < rows; ++r) for (std::size_t j = 0; j < S.size(); ++j) { full[std::size_t(r * n + S[j])] = Ws[std::size_t(r) * S.size() + j]; drop[std::size_t(r * n + S[j])] = W[std::size_t(r * n + S[j])]; }
    CHECK(rel_error(full.data(), W.data(), rows, n, H) < rel_error(drop.data(), W.data(), rows, n, H));
}

// ------------------------------------------------------------------------------------------------ unstructured
TEST(compress_masks_and_nm) {
    const i64 rows = 8, cols = 32;
    auto W = random_w(rows, cols, 9);
    std::vector<double> dg(static_cast<std::size_t>(cols)); for (i64 j = 0; j < cols; ++j) dg[std::size_t(j)] = 1 + j;
    for (auto crit : {Criterion::magnitude, Criterion::wanda, Criterion::ria}) {
        PruneCfg c; c.crit = crit; c.sparsity = 0.5;
        auto s = saliency(W.data(), rows, cols, dg, c);
        auto m = select_mask(s, W.data(), rows, cols, c);
        for (i64 r = 0; r < rows; ++r) { i64 z = 0; for (i64 j = 0; j < cols; ++j) z += !m[std::size_t(r * cols + j)]; CHECK(z == cols / 2); }
        c.nm_n = 2; c.nm_m = 4;
        m = select_mask(s, W.data(), rows, cols, c);
        for (i64 r = 0; r < rows; ++r) for (i64 g = 0; g < cols; g += 4) { int k = 0; for (int t = 0; t < 4; ++t) k += m[std::size_t(r * cols + g + t)]; CHECK(k == 2); }
    }
}

TEST(compress_sparsegpt_and_mcr_reduce_error) {
    const i64 rows = 16, cols = 64;
    auto H = random_spd(cols, 11);
    auto W0 = random_w(rows, cols, 12);
    std::vector<double> dg(static_cast<std::size_t>(cols)); for (i64 j = 0; j < cols; ++j) dg[std::size_t(j)] = H(j, j);
    for (int nm : {0, 1}) {
        PruneCfg c; c.crit = Criterion::wanda; c.sparsity = 0.5; c.block = 16;
        if (nm) { c.nm_n = 2; c.nm_m = 4; }
        auto Ww = W0;
        apply_mask(Ww.data(), select_mask(saliency(Ww.data(), rows, cols, dg, c), Ww.data(), rows, cols, c));
        auto Ws = W0;
        PruneCfg cs = c; cs.crit = Criterion::sparsegpt;
        sparsegpt(Ws.data(), rows, cols, H, cs);
        const double ew = rel_error(Ww.data(), W0.data(), rows, cols, H), es = rel_error(Ws.data(), W0.data(), rows, cols, H);
        i64 z = 0; for (float x : Ws) z += x == 0.f;
        CHECK_MSG(std::abs(double(z) / double(Ws.size()) - 0.5) < 0.02, "sparsegpt sparsity %g", double(z) / double(Ws.size()));
        if (nm) for (i64 r = 0; r < rows; ++r) for (i64 g = 0; g < cols; g += 4) { int k = 0; for (int t = 0; t < 4; ++t) k += Ws[std::size_t(r * cols + g + t)] != 0.f; CHECK(k <= 2); }
        CHECK_MSG(es < ew, "nm=%d sparsegpt %g should beat wanda %g", nm, es, ew);
        // the masked refit improves both and keeps their masks
        auto Wm = Ww; auto mask = mask_of(Wm.data(), Wm.size());
        auto rr = mcr_refit(Wm.data(), W0.data(), rows, cols, H, mask, 60, 0.0, 1e-8);
        CHECK_MSG(rr.err_after < ew * 0.9, "nm=%d mcr %g vs wanda %g", nm, rr.err_after, ew);
        for (std::size_t i = 0; i < mask.size(); ++i) if (!mask[i]) CHECK(Wm[i] == 0.f);
        // exactness: the refit solves each row's least squares on its support
        auto Wx = Ws; auto ms = mask_of(Wx.data(), Wx.size());
        mcr_refit(Wx.data(), W0.data(), rows, cols, H, ms, 200, 0.0, 1e-10);
        for (i64 r = 0; r < 2; ++r) {
            std::vector<i64> S; for (i64 j = 0; j < cols; ++j) if (ms[std::size_t(r * cols + j)]) S.push_back(j);
            auto ls = la::ls_refit_columns(W0.data() + r * cols, 1, H, S, 0.0);
            double e = 0; for (std::size_t j = 0; j < S.size(); ++j) e = std::max(e, double(std::abs(ls[j] - Wx[std::size_t(r * cols + S[j])])));
            CHECK_MSG(e < 1e-3, "row %lld: mcr vs direct least squares %g", (long long)r, e);
        }
        CHECK(rel_error(Wx.data(), W0.data(), rows, cols, H) <= es + 1e-9);
    }
}

TEST(compress_eba_allocation_spends_budget) {
    std::vector<double> grid = sparsity_grid(0.1, 0.9);
    std::vector<Unit> u(3);
    for (int i = 0; i < 3; ++i) { u[std::size_t(i)].name = std::to_string(i); u[std::size_t(i)].params = 100; u[std::size_t(i)].layer = i; for (double g : grid) u[std::size_t(i)].err.push_back((i + 1) * g * g); }
    auto a = eba_allocate(u, grid, 0.5);
    double tot = 0; for (auto& [k, v] : a) tot += v;
    CHECK(tot / 3 >= 0.5 - 1e-9);
    CHECK(a["0"] >= a["1"] && a["1"] >= a["2"]);   // the cheapest unit is pruned most
}

// ------------------------------------------------------------------------------------------------ model level
TEST(compress_regenerated_blueprint_and_blocks) {
    Model m = tiny_dense();
    auto c = random_calib(m.spec.vocab, 2, 24, 1);
    std::string p = tmp("orig.fpk");
    m.save(p);
    inference::session s(p);
    auto ref = s.forward_all(c.seqs[0]);
    Runner r(m.spec, m.w, 24, {}, true);
    r.run_tokens(c.seqs[0]);
    CHECK(max_abs_diff(ref, r.output()) < 1e-5);
    // block-by-block hidden states reproduce the last layer's input to the LM head
    Hidden h = Hidden::embed(m, c), nx;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) { layer_forward(m, l, h, nx); std::swap(h, nx); }
    Runner t(m.spec, m.w, 24, {}, true, [](const std::string& w) { return w == "out_norm" ? std::string("x") : std::string(); });
    t.run_tokens(c.seqs[0]);
    CHECK(max_abs_diff(t.tap("tap:x"), std::span<const float>(h.at(0), std::size_t(24 * m.spec.d_model))) < 1e-4);
    std::filesystem::remove(p);
}

TEST(compress_prune_stages_on_tiny_model) {
    for (const char* recipe : {"prune(method=wanda,s=0.5)", "prune(method=sparsegpt,s=0.5)", "prune(method=ria,nm=2:4)",
                               "prune(method=wanda,s=0.5,refit=mcr)", "prune(method=wanda,s=0.5,alloc=eba)", "prune(method=wanda,s=0.5,alloc=owl)", "prune(method=sparsegpt,nm=2:4,alloc=eba,frac=1)"}) {
        Model m = tiny_dense();
        auto c = random_calib(m.spec.vocab, 4, 32, 2);
        auto before = count(m);
        Ctx ctx{m, c, {}, 0.01};
        for (auto& st : parse_recipe(recipe)) run_stage(ctx, st);
        auto after = count(m);
        const double sp = 1.0 - double(after.body_nonzero) / double(before.body_nonzero);
        CHECK_MSG(sp > 0.4 && sp < 0.6, "%s: body sparsity %g", recipe, sp);
        check_saved_runs(m, c, "prune");
    }
}

TEST(compress_partial_nm_allocation) {
    Model m = tiny_dense();
    auto c = random_calib(m.spec.vocab, 4, 32, 12);
    Ctx ctx{m, c, {}, 0.01};
    auto rep = run_stage(ctx, parse_recipe("prune(method=sparsegpt,nm=2:4,alloc=eba,frac=0.5)")[0]);
    i64 sparse = 0, dense = 0;
    for (i64 l = 0; l < m.spec.n_layers(); ++l) for (const char* n : {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"}) {
        const auto& t = m.w.at(std::format("layer.{}.{}", l, n));
        (is_2of4(t.data.data(), t.data.size(), t.shape.back()) ? sparse : dense) += forge::numel(t.shape);
    }
    const double f = double(sparse) / double(sparse + dense);
    CHECK_MSG(f >= 0.5 && f < 0.8, "2:4 fraction %g", f);
}

TEST(compress_refit_beats_selection_only) {
    Model a = tiny_dense(21), b = tiny_dense(21), base = tiny_dense(21);
    auto c = random_calib(a.spec.vocab, 6, 32, 3);
    Ctx ca{a, c, {}, 0.01}, cb{b, c, {}, 0.01};
    for (auto& st : parse_recipe("prune(method=wanda,s=0.6)")) run_stage(ca, st);
    for (auto& st : parse_recipe("prune(method=wanda,s=0.6,refit=mcr)")) run_stage(cb, st);
    // closeness to the dense model's predictions on the calibration data
    auto kl = [&](Model& m) {
        Runner r0(base.spec, base.w, 32, {}, true), r1(m.spec, m.w, 32, {}, true);
        double s = 0;
        for (auto& ids : c.seqs) { r0.run_tokens(ids); r1.run_tokens(ids); auto p = r0.output(), q = r1.output(); for (std::size_t i = 0; i < p.size(); ++i) s += double(p[i] - q[i]) * (p[i] - q[i]); }
        return s;
    };
    CHECK_MSG(kl(b) < kl(a), "mcr logits distance %g vs wanda %g", kl(b), kl(a));
}

TEST(compress_width_depth_linear) {
    for (const char* recipe : {"width(s=0.5,score=obs,round=16)", "width(s=0.5,score=act,round=16)", "width(s=0.25,score=mag,round=16,alloc=eba)"}) {
        Model m = tiny_dense();
        auto c = random_calib(m.spec.vocab, 4, 32, 4);
        Ctx ctx{m, c, {}, 0.01};
        for (auto& st : parse_recipe(recipe)) run_stage(ctx, st);
        i64 tot = 0; for (i64 l = 0; l < m.spec.n_layers(); ++l) tot += m.w.at(std::format("layer.{}.w_down", l)).shape[1];
        CHECK_MSG(tot < 64 * m.spec.n_layers(), "%s: ffn widths unchanged", recipe);
        check_saved_runs(m, c, "width");
    }
    {   // OBS beats plain magnitude selection (calibration NLL)
        Model a = tiny_dense(5), b = tiny_dense(5);
        auto c = random_calib(a.spec.vocab, 6, 32, 5);
        Ctx ca{a, c, {}, 0.01}, cb{b, c, {}, 0.01};
        run_stage(ca, parse_recipe("width(s=0.5,score=mag,round=16)")[0]);
        run_stage(cb, parse_recipe("width(s=0.5,score=obs,round=16)")[0]);
        CHECK_MSG(nll(b, c) <= nll(a, c) + 1e-6, "obs %g vs mag %g", nll(b, c), nll(a, c));
    }
    for (const char* recipe : {"depth(n=2,score=bi)", "depth(n=2,score=probe,shortlist=4,probe=2)", "depth(n=1,unit=ffn,score=bi,replace=linear,rank=8)", "depth(n=1,unit=layer,score=probe,probe=2)"}) {
        Model m = tiny_dense(7, 4);
        auto c = random_calib(m.spec.vocab, 4, 32, 6);
        const std::size_t before = m.w.t.size();
        Ctx ctx{m, c, {}, 0.01};
        for (auto& st : parse_recipe(recipe)) run_stage(ctx, st);
        CHECK_MSG(m.w.t.size() != before || std::any_of(m.spec.layers.begin(), m.spec.layers.end(), [](auto& L) { return L.ffn == LayerSpec::Ffn::linear; }), "%s: nothing removed", recipe);
        check_saved_runs(m, c, "depth");
    }
}

TEST(compress_moe_stages) {
    for (const char* recipe : {"experts(drop=1,score=reap)", "experts(drop=2,score=freq)", "merge(drop=1,method=ls)", "merge(drop=1,method=avg)",
                               "moe2dense(width=0.5,round=8)", "moe2dense(width=1.0,round=8)", "prune(method=sparsegpt,s=0.5)"}) {
        Model m = tiny_moe();
        auto c = random_calib(m.spec.vocab, 4, 32, 7);
        Ctx ctx{m, c, {}, 0.01};
        for (auto& st : parse_recipe(recipe)) run_stage(ctx, st);
        check_saved_runs(m, c, "moe");
    }
    {   // a full-width MoE -> dense conversion with the least-squares down projection reproduces the MoE closely
        Model m = tiny_moe(9);
        auto c = random_calib(m.spec.vocab, 8, 32, 8);
        Hidden h = Hidden::embed(m, c);
        StatsReq rq; rq.gram = false; rq.moe_dense_acts = true;
        auto S = layer_pass(m, 0, h, rq, nullptr);
        double err = moe_to_dense(m, 0, S, 96, 1e-6);
        CHECK_MSG(err < 0.35, "full-width dense fit error %g", err);
        CHECK(m.spec.layers[0].ffn == LayerSpec::Ffn::dense);
    }
}

TEST(compress_eval_runs) {
    Model m = tiny_dense();
    std::string dir = tmp("evaldata");
    std::filesystem::create_directories(dir + "/tasks");
    { std::ofstream f(dir + "/wikitext2_test.txt"); for (int i = 0; i < 200; ++i) f << "the quick brown fox jumps over the lazy dog. "; }
    { std::ofstream f(dir + "/tasks/toy.jsonl"); f << R"({"ctxs":["the quick brown"],"conts":[" fox"," cat"],"label":0,"norm":false})" << "\n" << R"({"ctxs":["a b","a c"],"conts":[" d"," d"],"label":1,"norm":false})" << "\n"; }
    std::string p = tmp("eval.fpk");
    m.save(p);
    EvalCfg ec; ec.data_dir = dir; ec.ppl_window = 64; ec.ppl_windows = 4; ec.tasks = {"toy"}; ec.task_limit = 10;
    auto r = evaluate(p, ec);
    CHECK(r.ppl > 1 && std::isfinite(r.ppl));
    CHECK(r.n["toy"] == 2);
    // the evaluator's bucketed forward equals unpadded FORGE logits
    Evaluator ev(p);
    std::vector<int> ids{1, 2, 3, 4, 5};
    auto a = ev.run(ids);
    std::vector<float> av(a.begin(), a.end());
    inference::session s(p);
    auto b = s.forward_all(ids);
    CHECK(max_abs_diff(av, b) < 1e-4);
    std::filesystem::remove_all(dir); std::filesystem::remove(p);
}

// ------------------------------------------------------------------------------------------------ real models
TEST(compress_integration_real_models) {
    const char* dir = std::getenv("FORGE_COMPRESS_MODELS");
    if (!dir || !*dir) { std::printf("  (FORGE_COMPRESS_MODELS not set: real-model checks skipped)\n"); return; }
    for (const char* name : {"qwen3-0.6b", "tinyllama-sparse24", "tinymixtral"}) {
        std::string p = std::string(dir) + "/" + name + ".f32.fpk";
        if (!std::filesystem::exists(p)) { std::printf("  skip %s\n", name); continue; }
        Model m = Model::load(p);
        auto ids = m.encode("The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square.");
        ids.resize(32);
        inference::session ref(p);
        auto lref = ref.forward_all(ids);
        Runner r(m.spec, m.w, 32, {}, true);
        r.run_tokens(ids);
        CHECK_MSG(max_abs_diff(r.output(), lref) < 1e-3, "%s: regenerated blueprint differs", name);
        std::string out = tmp(std::string(name) + "_rt.fpk");
        m.save(out);
        inference::session s2(out);
        CHECK_MSG(max_abs_diff(s2.forward_all(ids), lref) < 1e-3, "%s: saved model differs", name);
        std::filesystem::remove(out);
        std::printf("  %s: regenerated + saved models match the import\n", name);
    }
}

// ------------------------------------------------------------------------------------------------ quantisation
TEST(compress_gptq_beats_rtn_and_is_exact_in_codec) {
    const i64 rows = 16, cols = 128;
    auto H = random_spd(cols, 31);
    auto W0 = random_w(rows, cols, 32);
    for (const char* cn : {"q4_0", "q8_0", "q4_g32"}) {
        const auto& cd = codec_of(cn);
        auto Wr = W0, Wg = W0;
        rtn_quantise(Wr.data(), rows, cols, cd, false);
        gptq_quantise(Wg.data(), rows, cols, H, cd, 0.01, false);
        const double er = rel_error(Wr.data(), W0.data(), rows, cols, H), eg = rel_error(Wg.data(), W0.data(), rows, cols, H);
        CHECK_MSG(eg < er, "%s: gptq %g should beat rtn %g", cn, eg, er);
        // the GPTQ result survives another encode/decode unchanged: it is what the file stores
        auto Wx = Wg; rtn_quantise(Wx.data(), rows, cols, cd, false);
        CHECK_MSG(max_abs_diff(Wx, Wg) < 1e-6, "%s: gptq weights not on the codec grid (%g)", cn, max_abs_diff(Wx, Wg));
    }
    // zeros of a pruned matrix survive quantisation
    auto Ws = W0; for (std::size_t i = 0; i < Ws.size(); i += 2) Ws[i] = 0.f;
    gptq_quantise(Ws.data(), rows, cols, H, codec_of("q4_0"), 0.01, true);
    bool zeros = true; for (std::size_t i = 0; i < Ws.size(); i += 2) zeros &= Ws[i] == 0.f;
    CHECK(zeros);
}

TEST(compress_quant_stage_saves_in_codec) {
    for (const char* recipe : {"quant(method=gptq,codec=q4_0)", "quant(method=rtn,codec=q8_0,head=rtn)", "prune(method=sparsegpt,s=0.5)+quant(method=gptq,codec=q8_0)"}) {
        Model m = tiny_dense();
        auto c = random_calib(m.spec.vocab, 4, 32, 9);
        Ctx ctx{m, c, {}, 0.01};
        for (auto& st : parse_recipe(recipe)) run_stage(ctx, st);
        CHECK(!m.codec.empty());
        std::string p = tmp("quant.fpk");
        m.save(p);
        modelfile::reader r(p);
        CHECK(r.entry("layer.0.wq").codec == m.codec.at("layer.0.wq"));
        // FORGE's quantised kernels on the stored codes reproduce the in-memory (dequantised) model
        inference::session s(p);
        auto lg = s.forward_all(c.seqs[0]);
        Runner rr(m.spec, m.w, c.seq, {}, true);
        rr.run_tokens(c.seqs[0]);
        CHECK_MSG(max_abs_diff(lg, rr.output()) < 1e-3, "%s: stored codes differ from the quantised weights (%g)", recipe, max_abs_diff(lg, rr.output()));
        std::filesystem::remove(p);
    }
}

// FORGE_COMPRESS_BENCH=1: timings of the factorisations behind SparseGPT / OBS / MoE -> dense.
TEST(compress_bench_linalg) {
    if (!std::getenv("FORGE_COMPRESS_BENCH")) return;
    for (i64 n : {2048, 4096}) {
        auto X = random_w(2 * n, n, 1);
        double t0 = now_s(); la::Gram g(n); g.add(X.data(), 2 * n); double t1 = now_s();
        la::Sym L = g.H; CHECK(la::cholesky(L)); double t2 = now_s();
        auto Hi = la::inverse_from_cholesky(L); double t3 = now_s();
        std::vector<float> W = random_w(256, n, 2);
        obs_columns(W.data(), 256, n, g.H, n * 3 / 4, 0.01); double t4 = now_s();
        std::printf("  n=%lld gram %.2fs cholesky %.2fs inverse %.2fs obs(25%%) %.2fs\n", (long long)n, t1 - t0, t2 - t1, t3 - t2, t4 - t3);
    }
}
