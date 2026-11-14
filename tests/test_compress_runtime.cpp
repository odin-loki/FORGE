// forge::compress at run time: context mixing in generation (mixer.hpp), Jacobi self-sampling (probes.hpp),
// KV-cache codec evaluation and speculative decoding with a compressed draft (eval.hpp). Tiny random models.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/compress/cli.hpp>
#include <forge/compress/mixer.hpp>
#include <forge/inference/session.hpp>

#include <filesystem>
#include <random>

using namespace forge;
using namespace forge::compress;
using forge_test::tiny_cfg;
using forge_test::make_tiny_model;

namespace {

std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / std::format("forge_crt_{}_{}", ::getpid(), n)).string(); }

std::string tiny_fpk(const std::string& name, unsigned seed = 7, int layers = 2) {
    return make_tiny_model(tmp(name), tiny_cfg{.d = 32, .layers = layers, .heads = 4, .kv = 2, .ffn = 64, .seed = seed});
}

std::vector<float> random_logits(i64 V, unsigned seed, float sd = 3.f) {
    std::mt19937 r(seed); std::normal_distribution<float> N(0, sd);
    std::vector<float> z(static_cast<std::size_t>(V)); for (auto& x : z) x = N(r); return z;
}

// mean NLL at T = 1 of positions [from, seq) under the model (Runner on its live weights)
double seq_nll(Model& m, const CalibSet& c, i64 from) {
    Runner r(m.spec, m.w, c.seq, {}, true);
    const i64 V = m.spec.vocab; double tot = 0; i64 n = 0;
    for (auto& s : c.seqs) {
        r.run_tokens(s);
        auto z = r.output();
        for (i64 t = from; t < c.seq; ++t) { tot -= logprob(z.data() + (t - 1) * V, V, s[std::size_t(t)]); ++n; }
    }
    return tot / double(n);
}

std::string wikitext_dir(const std::string& name) {
    std::string dir = tmp(name);
    std::filesystem::create_directories(dir);
    std::ofstream f(dir + "/wikitext2_test.txt");
    std::mt19937 r(5);
    static const char* words[] = {"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "a", "cat", "sat", "on", "mat"};
    for (int i = 0; i < 3000; ++i) f << words[r() % 14] << (i % 11 == 10 ? ". " : " ");
    return dir;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ context mixing
TEST(mixer_distribution_is_normalised_and_matches_scoring) {
    const i64 V = 300;
    StreamMixer sm; ContextMixer cm;
    std::mt19937 r(1);
    std::vector<int> pat; for (int i = 0; i < 12; ++i) pat.push_back(int(r() % V));
    sm.observe(pat[0]); cm.hist.push_back(pat[0]); cm.uni[pat[0]] = 1;
    double worst = 0, diff = 0;
    for (int step = 1; step < 120; ++step) {
        auto z = random_logits(V, unsigned(step));
        const int y = step % 3 == 0 ? int(r() % V) : pat[std::size_t(step) % pat.size()];
        const auto& P = sm.distribution(z);
        double tot = 0; for (double p : P) { CHECK(p > 0); tot += p; }
        worst = std::max(worst, std::abs(tot - 1));
        // the same models scored the evaluation way give the same probability of the observed token
        double lp[3]; double lse[3]; MixModels::lses(z, lse);
        for (int i = 0; i < 3; ++i) lp[i] = MixModels::scale[i] * z[std::size_t(y)] - lse[i];
        const double a = sm.logprob(y), b = cm.step(lp, y);
        diff = std::max(diff, std::abs(a - b));
        sm.observe(y);
    }
    CHECK_MSG(worst < 1e-9, "mixed distribution sums to 1 +- %g", worst);
    CHECK_MSG(diff < 1e-9, "stream mixer vs ContextMixer log p differ by %g", diff);
    std::printf("  mixer: |sum - 1| <= %.2e, stream vs scoring |dlogp| <= %.2e\n", worst, diff);
}

TEST(mixer_raises_repeated_continuation) {
    std::string p = tiny_fpk("mix.fpk");
    inference::session s(p);
    const std::size_t V = std::size_t(s.vocab());
    std::vector<int> pat{40, 71, 99, 12, 150, 33, 7, 201, 88, 64};
    std::vector<int> prompt; for (int rep = 0; rep < 5; ++rep) prompt.insert(prompt.end(), pat.begin(), pat.end());
    // the network's and the mixer's probability of each next token of the repetition, after the first copy
    StreamMixer mx;
    auto L = s.forward_all(prompt);
    mx.observe(prompt[0]);
    double lnet = 0, lmix = 0; int n = 0;
    for (std::size_t t = 1; t < prompt.size(); ++t) {
        std::span<const float> row(L.data() + (t - 1) * V, V);
        const auto& P = mx.distribution(row);
        if (t >= pat.size() * 2) { lnet += logprob(row.data(), i64(V), prompt[t]); lmix += std::log(P[std::size_t(prompt[t])]); ++n; }
        mx.observe(prompt[t]);
    }
    // the continuation after the prompt: pat[0]
    std::span<const float> last(L.data() + (prompt.size() - 1) * V, V);
    const double pn = std::exp(logprob(last.data(), i64(V), pat[0])), pm = mx.distribution(last)[std::size_t(pat[0])];
    CHECK_MSG(pm > pn && pm > 0.5, "continuation: mixer %g vs network %g", pm, pn);
    CHECK_MSG(lmix > lnet, "repeated text: mixer %g vs network %g nats/token", -lmix / n, -lnet / n);
    std::printf("  repetition: p(continuation) network %.4f -> mixed %.4f; nll %.3f -> %.3f nats/token\n", pn, pm, -lnet / n, -lmix / n);
    // generation with the mixer (forge run --mix 1): greedy continues the repetition
    s.reset();
    StreamMixer g; inference::sampler smp;
    auto out = generate_mixed(s, prompt, 20, smp, g);
    CHECK(out.size() == 20);
    int ok = 0; for (std::size_t i = 0; i < out.size(); ++i) ok += out[i] == pat[i % pat.size()];
    CHECK_MSG(ok >= 18, "mixed greedy continues the pattern: %d/20", ok);
    // plain greedy of the random network does not
    s.reset(); inference::sampler g0;
    auto plain = s.generate(prompt, 20, g0);
    int ok0 = 0; for (std::size_t i = 0; i < plain.size(); ++i) ok0 += plain[i] == pat[i % pat.size()];
    // sampling at T = 1 with the mixer also runs (temperature and top-p apply to the mixed distribution)
    s.reset();
    StreamMixer g2; inference::sampler_config sc; sc.temperature = 1.f; sc.top_p = 0.9f; sc.seed = 3; inference::sampler smp2(sc);
    auto out2 = generate_mixed(s, prompt, 20, smp2, g2);
    CHECK(out2.size() == 20);
    CHECK(g2.m.hist.size() == prompt.size() + 20);
    std::printf("  generate --mix: greedy follows the pattern %d/20 (plain greedy %d/20)\n", ok, ok0);
    std::filesystem::remove(p);
}

// ------------------------------------------------------------------------------------------------ Jacobi sampling
TEST(jacobi_fixed_point_is_autoregressive_sample) {
    std::string p = tiny_fpk("jac.fpk", 11);
    Model m = Model::load(p);
    const i64 seq = 24, n = 3, V = m.spec.vocab;
    const std::uint64_t seed = 5;
    auto prompts = self_prompts(m, n);
    // autoregressive sampling with the same per-position noise keys (and the same Runner shape: causal rows)
    Runner r(m.spec, m.w, seq, {}, true);
    std::vector<std::vector<int>> ar;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        std::vector<int> tok(static_cast<std::size_t>(seq), 0);
        std::copy(prompts[i].begin(), prompts[i].end(), tok.begin());
        std::vector<std::pair<float, int>> cand;
        for (i64 t = i64(prompts[i].size()); t < seq; ++t) {
            r.run_tokens(tok);
            tok[std::size_t(t)] = sample_nucleus(r.output().data() + (t - 1) * V, V, position_key(seed, i, std::uint64_t(t)), 0.95, cand);
        }
        ar.push_back(tok);
    }
    JacobiStats st;
    auto J = jacobi_generate(m, n, seq, seq, seed, &st);
    CHECK(J.seqs == ar);
    std::printf("  jacobi: fixed point after %lld passes over %lld sequences (changes per pass:", (long long)st.passes, (long long)n);
    for (i64 c : st.changed) std::printf(" %lld", (long long)c);
    std::printf(")\n");
    // after k passes the first |prompt| + k positions are exact
    auto J3 = jacobi_generate(m, n, seq, 3, seed);
    for (std::size_t i = 0; i < prompts.size(); ++i) CHECK(std::equal(ar[i].begin(), ar[i].begin() + i64(prompts[i].size()) + 3, J3.seqs[i].begin()));
    // the model-file variant (session forward_all) gives the same fixed point
    auto Jf = jacobi_generate(p, n, seq, seq, seed);
    int same = 0; for (std::size_t i = 0; i < ar.size(); ++i) same += Jf.seqs[i] == ar[i];
    CHECK_MSG(same == int(n), "session variant reproduces %d/%lld sequences", same, (long long)n);
    std::filesystem::remove(p);
}

TEST(jacobi_nll_approaches_decoded_samples) {
    // Jacobi text converges to autoregressive text as passes grow: its NLL under the model goes from the init's to
    // that of decoded samples (on this random network, as on real models, few positions settle ahead of the
    // one-per-pass frontier, so it takes many passes: see probes.hpp)
    std::string p = tiny_fpk("jacnll.fpk", 13);
    Model m = Model::load(p);
    const i64 seq = 48, n = 16;
    const i64 pl = i64(self_prompts(m, 1)[0].size());
    auto dec = self_generate(p, m, n, seq, 2, 8);       // FORGE decoding (other draws)
    const double nll_dec = seq_nll(m, dec, pl + 1);
    std::vector<double> by, ex;
    const std::vector<i64> ks{0, 2, 6, 16, 32, seq};
    for (i64 k : ks) {
        JacobiStats st;
        auto c = jacobi_generate(m, n, seq, k, 1, &st);
        by.push_back(seq_nll(m, c, pl + 1)); ex.push_back(st.exact_fraction());
    }
    std::printf("  jacobi nll (nats/token) [exact fraction] by passes:");
    for (std::size_t i = 0; i < ks.size(); ++i) std::printf(" %lld: %.3f [%.2f]", (long long)ks[i], by[i], ex[i]);
    std::printf("; decoded %.3f\n", nll_dec);
    CHECK_MSG(std::abs(by.back() - nll_dec) < 0.15 * nll_dec, "converged: nll %g vs decoded %g", by.back(), nll_dec);
    CHECK(ex.back() == 1.0);
    for (std::size_t i = 1; i < ks.size(); ++i) CHECK(ex[i] >= ex[i - 1] && ex[i] >= double(std::min(ks[i], seq - pl)) / double(seq - pl) - 1e-12);
    CHECK_MSG(std::abs(by[3] - nll_dec) < std::abs(by[0] - nll_dec), "16 passes approach decoded nll: %g vs init %g", by[3], by[0]);
    CHECK_MSG(std::abs(by[4] - nll_dec) < std::abs(by[1] - nll_dec), "32 passes closer than 2: %g vs %g", by[4], by[1]);
    std::filesystem::remove(p);
}

// ------------------------------------------------------------------------------------------------ KV codecs
TEST(eval_kv_codec_differs_slightly) {
    std::string p = tiny_fpk("kv.fpk", 17, 3);
    std::string dir = wikitext_dir("kvdata");
    EvalCfg ec; ec.data_dir = dir; ec.ppl_window = 128; ec.ppl_windows = 4;
    auto base = evaluate(p, ec);
    std::printf("  kv codec ppl: f32 %.5f", base.ppl);
    for (const char* kc : {"int8", "fp8", "tbq4", "tbq3", "k8v4"}) {
        ec.kv_codec = kc;
        auto r = evaluate(p, ec);
        const double rel = std::abs(r.ppl - base.ppl) / base.ppl;
        CHECK_MSG(std::isfinite(r.ppl) && rel < 0.05, "%s: ppl %g vs %g", kc, r.ppl, base.ppl);
        CHECK_MSG(r.ppl != base.ppl, "%s: identical to f32 (codec not applied?)", kc);
        CHECK(r.kv_codec == kc);
        std::printf(", %s %.5f", kc, r.ppl);
    }
    std::printf("\n");
    cli::opts o; o.kv["kv-codec"] = "int8";
    CHECK(cli::eval_cfg(o).kv_codec == "int8");
    std::filesystem::remove_all(dir); std::filesystem::remove(p);
}

// ------------------------------------------------------------------------------------------------ speculative
TEST(eval_speculative_draft) {
    std::string p = tiny_fpk("spec.fpk", 19, 3);
    std::string dir = wikitext_dir("specdata");
    // a perturbed draft: noise on every weight matrix
    std::string q = tmp("spec_draft.fpk");
    {
        Model m = Model::load(p);
        std::mt19937 r(2); std::normal_distribution<float> N(0, 1);
        for (auto& [name, t] : m.w.t) if (t.shape.size() == 2 && name != "tok_emb") for (auto& x : t.data) x += 0.35f * std::abs(x) * N(r);
        m.save(q);
    }
    EvalCfg ec; ec.data_dir = dir; ec.ppl_windows = -1; ec.spec_tokens = 32; ec.spec_prompts = 4; ec.spec_prompt_len = 48; ec.spec_k = 4;
    ec.draft = p;
    auto same = evaluate(p, ec);
    CHECK(same.spec.has_value());
    CHECK_MSG(same.spec->acceptance > 0.95, "self draft acceptance %g", same.spec->acceptance);
    CHECK(same.spec->lossless());
    ec.draft = q;
    auto pert = evaluate(p, ec);
    CHECK_MSG(pert.spec->acceptance < same.spec->acceptance, "perturbed draft acceptance %g vs %g", pert.spec->acceptance, same.spec->acceptance);
    CHECK_MSG(pert.spec->lossless(), "perturbed draft: %lld/%lld identical", (long long)pert.spec->identical, (long long)pert.spec->prompts);
    CHECK(pert.spec->new_tokens == 4 * 32);
    auto js = forge::json::dump(cli::eval_json(pert));
    CHECK(js.find("\"speculative\"") != std::string::npos && js.find("\"acceptance\"") != std::string::npos);
    std::printf("  speculative: self draft acceptance %.3f (%.2f tok/pass, x%.2f), perturbed draft %.3f (%.2f tok/pass, x%.2f), outputs identical %lld/%lld\n",
                same.spec->acceptance, same.spec->tokens_per_pass, same.spec->speedup, pert.spec->acceptance, pert.spec->tokens_per_pass, pert.spec->speedup,
                (long long)pert.spec->identical, (long long)pert.spec->prompts);
    std::filesystem::remove_all(dir); std::filesystem::remove(p); std::filesystem::remove(q);
}

