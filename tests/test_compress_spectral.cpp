// forge::compress spectral module: ESD diagnostics (Hill alpha, Marchenko-Pastur edge), alloc=alpha, the lowrank
// stage (mp / energy / whiten / mdl) and cross-layer basis sharing, on tiny random models and synthetic matrices.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/compress/compress.hpp>
#include <forge/compress/spectral.hpp>
#include <forge/inference/session.hpp>

#include <filesystem>
#include <random>

using namespace forge;
using namespace forge::compress;
using forge_test::tiny_cfg;
using forge_test::make_tiny_model;

namespace {

const int exact_int8 = (forge_test::set_env("FORGE_CPU_INT8", "0"), 0);

std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / std::format("forge_spectral_{}_{}", ::getpid(), n)).string(); }

double max_abs_diff(std::span<const float> a, std::span<const float> b) {
    double m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, double(std::abs(a[i] - b[i]))); return m;
}
double max_abs(std::span<const float> a) { double m = 0; for (float x : a) m = std::max(m, double(std::abs(x))); return m; }

std::vector<float> randn(i64 n, unsigned seed, float sd = 1.f) {
    std::mt19937 rng(seed); std::normal_distribution<float> N(0, sd);
    std::vector<float> v(static_cast<std::size_t>(n)); for (auto& x : v) x = N(rng); return v;
}

// Rank-k signal (scale s) plus N(0, noise^2) entries: [out, in].
std::vector<float> planted(i64 out, i64 in, i64 k, float s, float noise, unsigned seed) {
    auto B = randn(out * k, seed), A = randn(k * in, seed + 1), E = randn(out * in, seed + 2, noise);
    std::vector<float> W(static_cast<std::size_t>(out * in));
    la::gemm_nt(B.data(), [&] { std::vector<float> At(static_cast<std::size_t>(in * k)); la::transpose(A.data(), At.data(), k, in); return At; }().data(), W.data(), out, in, k);
    for (std::size_t i = 0; i < W.size(); ++i) W[i] = s * W[i] + E[i];
    return W;
}

CalibSet random_calib(i64 vocab, i64 n, i64 seq, unsigned seed) {
    std::mt19937 rng(seed); std::uniform_int_distribution<int> U(0, int(vocab) - 3);
    CalibSet c; c.seq = seq;
    for (i64 i = 0; i < n; ++i) { std::vector<int> v(static_cast<std::size_t>(seq)); for (auto& t : v) t = U(rng); c.seqs.push_back(v); }
    return c;
}

Model tiny_dense(unsigned seed = 7, int layers = 3) {
    std::string p = make_tiny_model(tmp("dense.fpk"), tiny_cfg{.d = 32, .layers = layers, .heads = 4, .kv = 2, .ffn = 64, .seed = seed});
    Model m = Model::load(p);
    std::filesystem::remove(p);
    return m;
}

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

// tr((W - W') H (W - W')^T) for W' = B A.
double h_err(const std::vector<float>& W, const tensor& A, const tensor& B, i64 out, i64 in, const la::Sym& H) {
    const i64 r = A.shape[0];
    std::vector<float> At(static_cast<std::size_t>(in * r)), R(static_cast<std::size_t>(out * in));
    la::transpose(A.data.data(), At.data(), r, in);
    la::gemm_nt(B.data.data(), At.data(), R.data(), out, in, r);
    double e = 0;
    for (i64 i = 0; i < out; ++i) {
        std::vector<double> d(static_cast<std::size_t>(in));
        for (i64 j = 0; j < in; ++j) d[std::size_t(j)] = double(W[std::size_t(i * in + j)]) - R[std::size_t(i * in + j)];
        for (i64 a = 0; a < in; ++a) { double s = 0; for (i64 b = 0; b < in; ++b) s += H(a, b) * d[std::size_t(b)]; e += d[std::size_t(a)] * s; }
    }
    return e;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ diagnostics
TEST(spectral_mp_law_and_planted_rank) {
    // pure noise: the median-based sigma^2 recovers the entry variance, almost nothing above the edge
    for (auto [out, in] : {std::pair<i64, i64>{200, 200}, {80, 240}, {240, 80}}) {
        auto W = randn(out * in, unsigned(out + in), 0.5f);
        auto sp = spectral::spectrum_of(W.data(), out, in);
        CHECK_MSG(std::abs(sp.sigma2 / 0.25 - 1) < 0.08, "%lldx%lld sigma2 %g", (long long)out, (long long)in, sp.sigma2);
        CHECK_MSG(sp.mp_rank <= 2, "noise mp_rank %lld", (long long)sp.mp_rank);
        CHECK(sp.esd.front() < 1.1 * sp.lambda_plus);
        CHECK(sp.alpha > 1.0);
    }
    // rank 5 + noise: the MP signal rank recovers it, energy keeps it
    for (auto [out, in] : {std::pair<i64, i64>{96, 160}, {160, 96}}) {
        auto W = planted(out, in, 5, 0.6f, 1.0f, 11);
        auto sp = spectral::spectrum_of(W.data(), out, in);
        CHECK_MSG(sp.mp_rank == 5, "planted rank 5, mp_rank %lld (edge %g, ev5 %g ev6 %g)", (long long)sp.mp_rank, sp.lambda_plus, sp.esd[4], sp.esd[5]);
        CHECK_MSG(std::abs(sp.sigma2 - 1.0) < 0.2, "sigma2 %g", sp.sigma2);
        CHECK(sp.signal_frac > 0.5 && sp.signal_frac < 1.0);
        auto Wc = planted(out, in, 5, 0.6f, 0.01f, 12);   // nearly exact low rank
        auto ev = spectral::sq_singular_values(Wc.data(), out, in);
        CHECK(spectral::energy_rank(ev, 0.999) == 5);
        CHECK(spectral::energy_rank(ev, 1.0) >= 5);
        // a heavier-tailed spectrum has a smaller alpha than noise
        auto sn = spectral::spectrum_of(randn(out * in, 5).data(), out, in);
        CHECK_MSG(sp.alpha < sn.alpha, "alpha planted %g noise %g", sp.alpha, sn.alpha);
    }
    // MP median: matches a large simulation (gamma = 1/2)
    auto W = randn(300 * 600, 99);
    auto sp = spectral::spectrum_of(W.data(), 300, 600);
    std::vector<double> e = sp.esd; std::nth_element(e.begin(), e.begin() + 150, e.end());
    CHECK_MSG(std::abs(e[150] / spectral::mp_median(0.5) - 1) < 0.03, "median %g vs MP %g", e[150], spectral::mp_median(0.5));
}

TEST(spectral_lowrank_stage_planted) {
    Model m = tiny_dense(21, 2);
    auto c = random_calib(m.spec.vocab, 2, 16, 3);
    auto W = planted(64, 32, 3, 0.15f, 0.02f, 41);   // w_gate [64, 32]: rank 3 + noise
    m.w.at("layer.0.w_gate").data = W;
    Model m2 = m;
    Ctx ctx{m, c, {}, 0.01};
    auto rep = run_stage(ctx, parse_recipe("lowrank(method=mp,scope=ffn)")[0]);
    CHECK(m.spec.factored(0, "w_gate"));
    CHECK(m.w.at("layer.0.w_gate_a").shape == (shape_t{3, 32}) && m.w.at("layer.0.w_gate_b").shape == (shape_t{64, 3}));
    CHECK(!m.spec.factored(0, "wq"));   // scope=ffn
    Ctx ctx2{m2, c, {}, 0.01};
    run_stage(ctx2, parse_recipe("lowrank(method=energy,keep=0.995,scope=ffn,min_save=0.5)")[0]);
    CHECK(m2.spec.factored(0, "w_gate") && m2.w.at("layer.0.w_gate_a").shape[0] == 3);
    CHECK(!m2.spec.factored(0, "w_up"));   // a random matrix: keeping 99.5% saves nothing
    check_saved_runs(m, c, "lr_mp");
}

// ------------------------------------------------------------------------------------------------ exactness
TEST(spectral_lowrank_full_rank_exact) {
    // the exact (dense eigh) path and the fast path (blocked subspace iteration, forced with exact_max=0)
    for (const char* solver : {"", ",exact_max=0,block0=8,block=8,over=4"}) for (const char* method : {"energy", "whiten", "mdl"}) {
        Model m = tiny_dense(5, 3);
        auto c = random_calib(m.spec.vocab, 2, 24, 8);
        auto ref = logits_of(m, c.seqs[0]);
        Ctx ctx{m, c, {}, 0.01};
        auto rep = run_stage(ctx, parse_recipe(std::format("lowrank(method={},rank=100000,min_save=-10{})", method, solver))[0]);
        for (i64 l = 0; l < 3; ++l) for (const char* n : {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"}) CHECK(m.spec.factored(l, n));
        const double e = max_abs_diff(logits_of(m, c.seqs[0]), ref);
        CHECK_MSG(e < 2e-3 * std::max(1.0, max_abs(ref)), "%s%s full rank: logits differ by %g", method, solver, e);
        if (std::string(method) == "whiten" && !*solver) check_saved_runs(m, c, "lr_full");
    }
}

// ------------------------------------------------------------------------------------------------ whitening
TEST(spectral_whiten_beats_plain_svd) {
    for (auto [out, in] : {std::pair<i64, i64>{24, 48}, {48, 24}, {40, 40}}) {
        auto W = randn(out * in, unsigned(7 * out + in));
        // correlated inputs: X = Z C with a random mixing matrix and uneven scales
        const i64 n = 4 * in;
        auto Z = randn(n * in, 3), C = randn(in * in, 4);
        for (i64 i = 0; i < in; ++i) for (i64 j = 0; j < in; ++j) C[std::size_t(i * in + j)] *= (i == j ? 3.f : 0.5f) * (1.f + float(j % 4));
        std::vector<float> Ct(C.size()), X(static_cast<std::size_t>(n * in));
        la::transpose(C.data(), Ct.data(), in, in);
        la::gemm_nt(Z.data(), Ct.data(), X.data(), n, in, in);
        la::Gram g(in); g.add(X.data(), n);
        for (i64 r : {2, 6, 12}) {
            auto Fp = spectral::analyse(W.data(), out, in, nullptr);
            auto Fw = spectral::analyse(W.data(), out, in, &g.H);
            auto [Ap, Bp] = spectral::factors(Fp, r);
            auto [Aw, Bw] = spectral::factors(Fw, r);
            const double ep = h_err(W, Ap, Bp, out, in, g.H), ew = h_err(W, Aw, Bw, out, in, g.H);
            CHECK_MSG(ew <= ep * (1 + 1e-5), "%lldx%lld r=%lld: whitened %g > plain %g", (long long)out, (long long)in, (long long)r, ew, ep);
            CHECK_MSG(ew < 0.95 * ep, "%lldx%lld r=%lld: whitening should help on correlated inputs (%g vs %g)", (long long)out, (long long)in, (long long)r, ew, ep);
            // the spectrum predicts the H-weighted error
            const double tot = Fw.eig.trace;
            CHECK_MSG(std::abs(spectral::rel_residual(Fw.eig, r) * tot - ew) <= 1e-3 * tot, "predicted %g measured %g", spectral::rel_residual(Fw.eig, r) * tot, ew);
        }
    }
}

// ------------------------------------------------------------------------------------------------ mdl
TEST(spectral_mdl_rank) {
    for (auto [out, in] : {std::pair<i64, i64>{64, 128}, {128, 64}}) {
        std::vector<i64> ranks;
        for (unsigned seed : {1u, 2u, 3u}) {
            auto W = planted(out, in, 6, 0.3f, 0.05f, seed * 10);
            auto ev = spectral::sq_singular_values(W.data(), out, in);
            std::vector<double> cost;
            const i64 r = spectral::mdl_rank(ev, out, in, 16, &cost);
            CHECK_MSG(r >= 1 && r < std::min(out, in), "mdl rank %lld", (long long)r);
            CHECK_MSG(r == 6, "mdl should find the planted rank 6, got %lld", (long long)r);
            CHECK(cost[std::size_t(r)] <= cost[1] && cost[std::size_t(r)] <= cost.back());
            ranks.push_back(r);
            // fewer bits per weight make parameters cheaper: the rank does not drop
            CHECK(spectral::mdl_rank(ev, out, in, 8) >= 1);
        }
        CHECK(ranks[0] == ranks[1] && ranks[1] == ranks[2]);
    }
    // on a model: rank in [1, full), the same on a second run of the same model, the model still runs
    Model m = tiny_dense(9, 2), m2 = m;
    auto c = random_calib(m.spec.vocab, 2, 16, 4);
    Ctx ctx{m, c, {}, 0.01}, ctx2{m2, c, {}, 0.01};
    auto r1 = run_stage(ctx, parse_recipe("lowrank(method=mdl,min_save=-10)")[0]);
    auto r2 = run_stage(ctx2, parse_recipe("lowrank(method=mdl,min_save=-10)")[0]);
    const auto& a = r1["matrices"].arr(); const auto& b = r2["matrices"].arr();
    CHECK(a.size() == b.size() && !a.empty());
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double r = a[i]["rank"].num();
        CHECK(r >= 1 && r <= a[i]["dims"].num());
        CHECK(r == b[i]["rank"].num());
        CHECK(a[i]["whitened"].b());
    }
    check_saved_runs(m, c, "lr_mdl");
}

// ------------------------------------------------------------------------------------------------ alpha allocation
TEST(spectral_alpha_allocation) {
    Model m = tiny_dense(13, 4);
    // layer 1 gets a heavy-tailed spectrum (power-law singular values), layer 3 a lighter one
    for (i64 l : {1, 3}) for (const char* n : {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"}) {
        auto& t = m.w.at(std::format("layer.{}.{}", l, n));
        const i64 out = t.shape[0], in = t.shape[1];
        auto F = spectral::analyse(t.data.data(), out, in, nullptr);
        auto [A, B] = spectral::factors(F, std::min(out, in));
        // rescale the k-th singular direction by (k+1)^-p
        const double p = l == 1 ? 1.5 : 0.1;
        for (i64 k = 0; k < A.shape[0]; ++k) for (i64 j = 0; j < in; ++j) A.data[std::size_t(k * in + j)] *= float(std::pow(double(k + 1), -p));
        std::vector<float> At(static_cast<std::size_t>(in * A.shape[0]));
        la::transpose(A.data.data(), At.data(), A.shape[0], in);
        la::gemm_nt(B.data.data(), At.data(), t.data.data(), out, in, A.shape[0]);
    }
    auto c = random_calib(m.spec.vocab, 2, 16, 5);
    Ctx ctx{m, c, {}, 0.01};
    PruneCfg pc; pc.sparsity = 0.5;
    Stage st{"prune", {{"alloc", "alpha"}}};
    auto a = alloc_registry().at("alpha")(ctx, pc, 0.5, st);
    double P = 0, S = 0;
    std::map<i64, double> ls;
    for (i64 l = 0; l < 4; ++l) for (const char* n : {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"}) {
        const std::string nm = std::format("layer.{}.{}", l, n);
        CHECK(a.count(nm));
        const double pr = double(forge::numel(m.w.at(nm).shape));
        P += pr; S += pr * a[nm]; ls[l] = a[nm];
        CHECK(a[nm] >= 0.3 - 1e-9 && a[nm] <= 0.7 + 1e-9);
    }
    CHECK_MSG(std::abs(S / P - 0.5) < 1e-6, "weighted mean sparsity %g", S / P);
    // order: alpha ascending <=> sparsity ascending
    std::map<i64, double> la_;
    for (i64 l = 0; l < 4; ++l) {
        double s = 0; int k = 0;
        for (const char* n : {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"}) { auto& t = m.w.at(std::format("layer.{}.{}", l, n)); s += spectral::spectrum_of(t.data.data(), t.shape[0], t.shape[1]).alpha; ++k; }
        la_[l] = s / k;
    }
    for (i64 i = 0; i < 4; ++i) for (i64 j = 0; j < 4; ++j) if (la_[i] < la_[j] - 1e-9) CHECK_MSG(ls[i] <= ls[j] + 1e-9, "layer %lld alpha %g s %g vs layer %lld alpha %g s %g", (long long)i, la_[i], ls[i], (long long)j, la_[j], ls[j]);
    CHECK_MSG(la_[1] < la_[0] && ls[1] < ls[0], "heavy-tailed layer 1 keeps more weights (alpha %g vs %g)", la_[1], la_[0]);
    // per-projection mode and the full prune stage hit the target
    Stage sp{"prune", {{"alloc", "alpha"}, {"alpha_unit", "proj"}}};
    auto a2 = alloc_registry().at("alpha")(ctx, pc, 0.5, sp);
    double S2 = 0; for (auto& [k, v] : a2) S2 += v * double(forge::numel(m.w.at(k).shape));
    CHECK(std::abs(S2 / P - 0.5) < 1e-6);
    run_stage(ctx, parse_recipe("prune(method=wanda,s=0.5,alloc=alpha)")[0]);
    double z = 0;
    for (i64 l = 0; l < 4; ++l) for (const char* n : {"wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"}) for (float x : m.w.at(std::format("layer.{}.{}", l, n)).data) z += x == 0.f;
    CHECK_MSG(std::abs(z / P - 0.5) < 0.02, "pruned fraction %g", z / P);
    check_saved_runs(m, c, "alpha");
    // MoE: experts are units
    Model mm = tiny_moe();
    Ctx cm{mm, c, {}, 0.01};
    auto am = alloc_registry().at("alpha")(cm, pc, 0.4, Stage{"prune", {{"alloc", "alpha"}, {"alpha_unit", "proj"}}});
    CHECK(am.count("layer.1.experts.w_gate#2") && am.count("layer.0.wq"));
    double Pm = 0, Sm = 0; for (auto& [k, v] : am) { const auto& t = mm.w.at(k.substr(0, k.find('#'))); const double pr = k.find('#') != std::string::npos ? double(t.shape[1] * t.shape[2]) : double(forge::numel(t.shape)); Pm += pr; Sm += pr * v; }
    CHECK(std::abs(Sm / Pm - 0.4) < 1e-6);
}

// ------------------------------------------------------------------------------------------------ basis sharing
TEST(spectral_share_basis) {
    for (const char* solver : {"", ",exact_max=0,block0=8,block=8,over=4"}) {   // full rank: exact (both solvers)
        Model m = tiny_dense(17, 4);
        auto c = random_calib(m.spec.vocab, 2, 16, 6);
        auto ref = logits_of(m, c.seqs[0]);
        Ctx ctx{m, c, {}, 0.01};
        run_stage(ctx, parse_recipe(std::format("share(group=2,rank=1000,min_save=-10{})", solver))[0]);
        CHECK(m.spec.factored(3, "w_down") && m.spec.alias.count("layer.3.w_down_b"));
        const double e = max_abs_diff(logits_of(m, c.seqs[0]), ref);
        CHECK_MSG(e < 2e-3 * std::max(1.0, max_abs(ref)), "full-rank share differs by %g", e);
    }
    for (const char* data : {"0", "1"}) {
        Model m = tiny_dense(17, 5);
        auto c = random_calib(m.spec.vocab, 2, 16, 6);
        const i64 before = spectral::stored_params(m);
        Ctx ctx{m, c, {}, 0.01};
        auto rep = run_stage(ctx, parse_recipe(std::format("share(group=2,rank=8,scope=attn,data={})", data))[0]);
        // groups {0,1}, {2,3}; layer 4 alone stays dense
        i64 expect = 0;
        for (const char* n : {"wq", "wk", "wv", "wo"}) {
            for (i64 l = 0; l < 4; ++l) CHECK(m.spec.factored(l, n));
            CHECK(!m.spec.factored(4, n));
            CHECK(m.spec.alias.at(std::format("layer.1.{}_b", n)) == std::format("layer.0.{}_b", n));
            CHECK(m.spec.alias.at(std::format("layer.3.{}_b", n)) == std::format("layer.2.{}_b", n));
            CHECK(!m.spec.alias.count(std::format("layer.0.{}_b", n)) && !m.spec.alias.count(std::format("layer.1.{}_a", n)));
            const auto& b = m.w.at(std::format("layer.0.{}_b", n));
            const i64 out = b.shape[0], in = m.w.at(std::format("layer.0.{}_a", n)).shape[1];
            CHECK(b.shape[1] == 8);
            // per group: one basis [out, 8] + two per-layer [8, in] coefficients instead of two [out, in]
            expect += 2 * (2 * out * in - (8 * out + 2 * 8 * in));
        }
        CHECK(!m.spec.factored(0, "w_gate"));
        const i64 after = spectral::stored_params(m);
        CHECK_MSG(before - after == expect, "params %lld -> %lld (expected saving %lld)", (long long)before, (long long)after, (long long)expect);
        std::string p = tmp(std::format("share{}.fpk", data));
        m.save(p);
        {
            modelfile::reader rd(p);
            for (const char* n : {"wq", "wk", "wv", "wo"}) {
                CHECK(rd.tensors().count(std::format("layer.0.{}_b", n)) && rd.tensors().count(std::format("layer.2.{}_b", n)));
                CHECK(!rd.tensors().count(std::format("layer.1.{}_b", n)) && !rd.tensors().count(std::format("layer.3.{}_b", n)));   // stored once per group
                CHECK(rd.tensors().count(std::format("layer.1.{}_a", n)) && rd.tensors().count(std::format("layer.3.{}_a", n)));
            }
            Model back = Model::load(p);
            CHECK(back.spec.alias == m.spec.alias && back.spec.lowrank == m.spec.lowrank);
            inference::session s(p);
            Runner r(m.spec, m.w, c.seq, {}, true);
            r.run_tokens(c.seqs[0]);
            CHECK_MSG(max_abs_diff(s.forward_all(c.seqs[0]), r.output()) < 1e-4, "share data=%s: saved model differs", data);
        }
        std::filesystem::remove(p);
        // later stages see the shared model (the aliased layers keep valid store entries)
        run_stage(ctx, parse_recipe("prune(method=wanda,s=0.3)")[0]);
        check_saved_runs(m, c, "share_pruned");
    }
}

TEST(spectral_report_stage) {
    Model m = tiny_dense(3, 2);
    auto c = random_calib(m.spec.vocab, 1, 8, 1);
    Ctx ctx{m, c, {}, 0.01};
    auto r = run_stage(ctx, parse_recipe("spectra")[0]);
    CHECK(r["matrices"].arr().size() == 14);
    for (auto& x : r["matrices"].arr()) CHECK(x["alpha"].num() > 1 && x["lambda_plus"].num() > 0);
}

// ------------------------------------------------------------------------------------------------ fast solver
namespace {
// A decaying spectrum (like trained weights: power-law singular values over a noise floor): A diag(d) B + noise.
std::vector<float> decaying(i64 out, i64 in, double pw, float noise, unsigned seed) {
    const i64 p = std::min(out, in);
    auto A = randn(out * p, seed), B = randn(p * in, seed + 1), E = randn(out * in, seed + 2, noise);
    for (i64 i = 0; i < out; ++i) for (i64 j = 0; j < p; ++j) A[std::size_t(i * p + j)] *= float(std::pow(double(j + 1), -pw) / std::sqrt(double(p)));
    std::vector<float> Bt(static_cast<std::size_t>(in * p)), W(static_cast<std::size_t>(out * in));
    la::transpose(B.data(), Bt.data(), p, in);
    la::gemm_nt(A.data(), Bt.data(), W.data(), out, in, p);
    for (std::size_t i = 0; i < W.size(); ++i) W[i] += E[i];
    return W;
}
}  // namespace

TEST(spectral_fast_matches_exact) {
    spectral::SpecOpts fast; fast.exact_max = 0;
    struct Case { const char* what; i64 out, in; std::vector<float> W; };
    std::vector<Case> cases;
    cases.push_back({"planted", 400, 700, planted(400, 700, 8, 0.08f, 1.0f, 31)});
    cases.push_back({"decaying", 600, 450, decaying(600, 450, 0.8, 0.02f, 32)});
    for (auto& cs : cases) {
        const i64 out = cs.out, in = cs.in, p = std::min(out, in);
        const bool planted_case = std::string(cs.what) == "planted";
        auto G = spectral::small_gram(cs.W.data(), out, in);
        double t0 = now_s();
        auto E = spectral::eig_desc(G.data(), p);   // reference: dense eigh
        const double te = now_s() - t0;
        t0 = now_s();
        spectral::TopEig T(G, p, fast);
        T.grow(160);
        const double tf = now_s() - t0;
        CHECK(T.k() >= 160 && !T.exact());
        double worst = 0;
        for (i64 i = 0; i < 160; ++i) worst = std::max(worst, std::abs(T.w[std::size_t(i)] - E.w[std::size_t(i)]) / E.w[std::size_t(i)]);
        CHECK_MSG(worst < 1e-3, "%s: top-160 eigenvalues rel err %g", cs.what, worst);
        double tr = 0; for (double x : E.w) tr += x;
        CHECK(std::abs(T.trace - tr) <= 1e-4 * tr);
        for (i64 i = 0; i < 8; ++i) {   // leading eigenvectors
            double d = 0; for (i64 j = 0; j < p; ++j) d += double(T.Ut[std::size_t(i * p + j)]) * E.V(std::size_t(j), std::size_t(i));
            CHECK_MSG(std::abs(d) > 0.999, "%s: eigenvector %lld overlap %g", cs.what, (long long)i, d);
        }
        std::printf("  %s %lldx%lld: top-160 rel err %.2e, %d subspace iterations, %.2fs (dense eigh %.2fs)\n", cs.what, (long long)out, (long long)in, worst, T.iters, tf, te);
        // ranks: energy (with and without a cap), MDL, MP signal rank, Hill alpha and sigma^2
        for (double keep : {0.5, 0.9, 0.99}) {
            spectral::TopEig F(G, p, fast);
            const i64 re = spectral::energy_rank(E.w, keep), rf = spectral::energy_rank(F, keep, p);
            if (planted_case) CHECK_MSG(rf == re, "planted keep %g: fast rank %lld exact %lld", keep, (long long)rf, (long long)re);
            else CHECK_MSG(std::abs(rf - re) <= 1, "decaying keep %g: fast rank %lld exact %lld", keep, (long long)rf, (long long)re);
            spectral::TopEig Fc(G, p, fast);
            const i64 rc = spectral::energy_rank(Fc, keep, 20);
            CHECK(re > 20 ? rc == 21 && Fc.k() <= 20 + fast.over + fast.block0 : rc == re);
        }
        {
            spectral::TopEig F(G, p, fast);
            const i64 me = spectral::mdl_rank(E.w, out, in, 16), mf = spectral::mdl_rank(F, out, in, 16, p);
            if (planted_case) CHECK_MSG(me == mf, "%s: mdl fast %lld exact %lld", cs.what, (long long)mf, (long long)me);
            else CHECK_MSG(std::abs(me - mf) <= 1, "%s: mdl fast %lld exact %lld", cs.what, (long long)mf, (long long)me);
            std::printf("  %s: mdl rank %lld/%lld (fast/exact), eigenpairs computed %lld\n", cs.what, (long long)mf, (long long)me, (long long)F.k());
        }
        {
            spectral::TopEig F(G, p, fast);
            auto sf = spectral::analyse_spectrum(F, out, in);
            auto se = spectral::analyse_esd(E.w, out, in);
            CHECK(!sf.exact && se.exact);
            if (planted_case) CHECK_MSG(sf.mp_rank == se.mp_rank && se.mp_rank == 8, "mp rank fast %lld exact %lld", (long long)sf.mp_rank, (long long)se.mp_rank);
            CHECK_MSG(std::abs(sf.sigma2 / se.sigma2 - 1) < 0.05, "%s: sigma2 fast %g exact %g", cs.what, sf.sigma2, se.sigma2);
            CHECK_MSG(std::abs(sf.alpha / se.alpha - 1) < 0.05, "%s: alpha fast %g exact %g", cs.what, sf.alpha, se.alpha);
            CHECK(std::abs(sf.signal_frac - se.signal_frac) < 0.02);
            std::printf("  %s: sigma2 %.4g/%.4g alpha %.4g/%.4g mp_rank %lld/%lld (fast/exact)\n", cs.what, sf.sigma2, se.sigma2, sf.alpha, se.alpha, (long long)sf.mp_rank, (long long)se.mp_rank);
        }
        // factors from the fast path: the same reconstruction error as the exact ones
        {
            auto Fe = spectral::analyse(cs.W.data(), out, in, nullptr);
            auto Ff = spectral::analyse(cs.W.data(), out, in, nullptr, 1e-6, fast);
            la::Sym I(in); for (i64 i = 0; i < in; ++i) I(i, i) = 1;
            for (i64 r : {8, 40}) {
                auto [Ae, Be] = spectral::factors(Fe, r);
                auto [Af, Bf] = spectral::factors(Ff, r);
                const double ee = h_err(cs.W, Ae, Be, out, in, I), ef = h_err(cs.W, Af, Bf, out, in, I);
                CHECK_MSG(std::abs(ef - ee) <= 1e-3 * ee, "%s r=%lld: fast factor error %g exact %g", cs.what, (long long)r, ef, ee);
            }
        }
    }
}
