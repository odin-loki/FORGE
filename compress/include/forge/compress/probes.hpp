// Data-free probes of the model's global function and corrections of its output distribution.
//
//   self_generate     calibration text written by the model itself (FORGE continuous batching, sampled at T=1,
//                     nucleus 0.95): in-distribution for the model, no dataset needed
//   jacobi_generate   the same distribution of text from full-sequence prefills only (Jacobi fixed-point iteration
//                     of autoregressive sampling): every position is resampled in parallel from the model's
//                     distribution given the current tokens before it, with one fixed random draw per position, so
//                     positions converge from the left and the fixed point is exactly the autoregressive sample.
//                     Prefill rows are much cheaper than decode steps, but on real models few positions settle
//                     ahead of the guaranteed one-per-pass frontier (qwen3-0.6b: ~75 % of the positions change in
//                     every pass, NLL ~8 nats/token after 6 passes vs 1.8 for decoded text), so --calib-data self
//                     keeps decoding by default and --self-mode jacobi is opt-in
//   Reference         the uncompressed model's next-token distributions on probe sequences (log-probabilities
//                     kept as f16), captured before compression
//   kl                mean KL(p_reference || p_model) per token: the measured error of the global function, used as
//                     feedback by the mixed-precision allocator and the distribution corrections
//   calibrate         two corrections fitted to minimise that KL, each kept only if it lowers the KL on held-out
//                     probe sequences:
//                       temperature  logits * t, folded exactly into the output norm gain (t searched by golden
//                                    section on the KL)
//                       logit bias   b_v = log E[p_ref(v)] - log E[p_model(v)] (fixed-point, shrunk towards 0 for
//                                    rarely seen tokens): removes the unigram drift compression leaves behind
#pragma once
#include <forge/compress/calib.hpp>
#include <forge/inference/batch.hpp>

#include <algorithm>
#include <random>

namespace forge::compress {

// Prompts of the self-generated sequences: a BOS (when the model adds one) and a common sentence start.
inline std::vector<std::vector<int>> self_prompts(const value& man, const std::function<std::vector<int>(const std::string&)>& encode, i64 n) {
    const bool bos = man.find("add_bos") && man["add_bos"].is_bool() && man["add_bos"].b();
    const int bos_id = man.find("bos_token_id") && man["bos_token_id"].is_num() ? int(man["bos_token_id"].num()) : -1;
    static const char* starts[] = {"The", "In", "A", "It", "This", "When", "However,", "There", "One", "We", "He", "She", "After", "According to", "Today", "If"};
    std::vector<std::vector<int>> out;
    for (i64 i = 0; i < n; ++i) {
        auto p = encode(starts[i % 16]);
        if (bos && bos_id >= 0) p.insert(p.begin(), bos_id);
        out.push_back(std::move(p));
    }
    return out;
}
inline std::vector<std::vector<int>> self_prompts(const Model& m, i64 n) { return self_prompts(m.manifest, [&m](const std::string& t) { return m.encode(t); }, n); }

inline CalibSet self_generate(const std::string& fpk, const Model& m, i64 n, i64 seq, std::uint64_t seed = 0, int parallel = 16) {
    auto model = inference::model::open(fpk, false);
    inference::batch_options bo; bo.max_active = parallel; bo.session.context = seq + 8;
    inference::batch_scheduler sch(model, bo);
    const auto prompts = self_prompts(m, n);
    CalibSet c; c.seq = seq;
    std::vector<std::vector<int>> outs(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) {
        inference::request r;
        r.prompt = prompts[std::size_t(i)];
        r.max_new = int(seq - i64(r.prompt.size()));
        r.sampling.temperature = 1.f; r.sampling.top_p = 0.95f; r.sampling.seed = seed * 1000003 + std::uint64_t(i);
        r.stop_tokens = {};
        auto prompt = r.prompt;
        r.on_done = [&outs, i, prompt](const std::vector<int>& toks, const std::string&) {
            auto& o = outs[std::size_t(i)]; o = prompt; o.insert(o.end(), toks.begin(), toks.end());
        };
        sch.submit(std::move(r));
    }
    sch.run_until_idle();
    for (auto& o : outs) {
        if (i64(o.size()) < seq) { const auto cp = o; while (i64(o.size()) < seq) o.insert(o.end(), cp.begin(), cp.end()); }   // short (EOS) samples repeat
        o.resize(static_cast<std::size_t>(seq));
        c.seqs.push_back(o);
    }
    return c;
}

// Coupled nucleus sampling at temperature 1 (Gumbel-max): the candidates (tokens within e^-30 of the likeliest) in
// descending probability (ties: lower id) are cut after the token that reaches top_p of the mass, and the token
// drawn is argmax_v z_v + g(key, v), with Gumbel noise g that is a fixed function of (key, v). That is an exact draw
// from the truncated softmax, and a deterministic function of (row, key). Two rows with similar distributions and the
// same key mostly give the same token (the Gumbel-max coupling): what lets Jacobi iterations settle well before the
// left-to-right convergence guarantee reaches a position. (Inverse-CDF draws change token whenever the cumulative
// mass in front of it moves, which on real models resamples almost every position in every pass.)
inline double gumbel_of(std::uint64_t key, std::uint64_t v) {
    std::uint64_t x = key ^ (v + 0x9E3779B97F4A7C15ull) * 0xBF58476D1CE4E5B9ull;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 31;   // splitmix64
    const double u = (double(x >> 11) + 0.5) * 0x1.0p-53;   // (0, 1)
    return -std::log(-std::log(u));
}
inline std::uint64_t position_key(std::uint64_t seed, std::uint64_t seqno, std::uint64_t t) {
    std::uint64_t x = seed * 0x9E3779B97F4A7C15ull ^ (seqno + 1) * 0xC2B2AE3D27D4EB4Full ^ (t + 1) * 0x165667B19E3779F9ull;
    x ^= x >> 33; x *= 0xFF51AFD7ED558CCDull; x ^= x >> 33;
    return x;
}
inline int sample_nucleus(const float* z, i64 V, std::uint64_t key, double top_p, std::vector<std::pair<float, int>>& cand) {
    float mx = -INFINITY; for (i64 v = 0; v < V; ++v) mx = std::max(mx, z[v]);
    double Z = 0; cand.clear();
    for (i64 v = 0; v < V; ++v) {
        const float d = z[v] - mx;
        Z += std::exp(double(d));
        if (d > -30.f) cand.push_back({d, int(v)});
    }
    std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    double cum = 0; std::size_t keep = cand.size();
    for (std::size_t i = 0; i < cand.size(); ++i) { cum += std::exp(double(cand[i].first)); if (cum >= top_p * Z) { keep = i + 1; break; } }
    int best = cand[0].second; double bv = -INFINITY;
    for (std::size_t i = 0; i < keep; ++i) {
        const double x = double(cand[i].first) + gumbel_of(key, std::uint64_t(cand[i].second));
        if (x > bv) { bv = x; best = cand[i].second; }
    }
    return best;
}

struct JacobiStats {
    i64 passes = 0;                 // passes run (summed over sequences)
    std::vector<i64> changed;       // tokens changed by pass p (summed over sequences)
    double seconds = 0;
    i64 tokens = 0;                 // generated positions (summed over sequences)
    i64 exact = 0;                  // of them, positions known to equal the autoregressive sample (the converged prefix)
    double exact_fraction() const { return tokens ? double(exact) / double(tokens) : 0; }
};

// Jacobi self-sampling. `run(ids)` returns the [seq, V] next-token logits of a full-sequence prefill. Sequence i
// starts as prompts[i] followed by `init` tokens ("repeat": the prompt repeated, "random": uniform tokens); each
// pass resamples every position t + 1 >= |prompt| from row t with its own fixed noise key (position_key(seed, i, t + 1);
// coupled nucleus sampling, top_p, T = 1) in parallel. After pass k the first |prompt| + k positions equal the
// autoregressive sample with the same keys; a
// pass that changes nothing has reached that fixed point, so the iteration stops early.
template<class Run>
CalibSet jacobi_core(Run&& run, i64 V, const std::vector<std::vector<int>>& prompts, i64 seq, i64 passes, std::uint64_t seed,
                     double top_p = 0.95, const std::string& init = "repeat", JacobiStats* stats = nullptr) {
    const double t0 = now_s();
    CalibSet c; c.seq = seq;
    JacobiStats st; st.changed.assign(std::size_t(std::max<i64>(passes, 0)), 0);
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        const auto& pr = prompts[i];
        const i64 pl = std::min<i64>(std::max<i64>(i64(pr.size()), 1), seq);
        std::mt19937_64 rng(seed * 1000003 + i);   // (random init only)
        std::vector<int> tok(static_cast<std::size_t>(seq), 0);
        for (i64 t = 0; t < seq; ++t) {
            if (t < i64(pr.size())) tok[std::size_t(t)] = pr[std::size_t(t)];
            else if (init == "random") tok[std::size_t(t)] = int(rng() % std::uint64_t(V));
            else tok[std::size_t(t)] = pr.empty() ? 0 : pr[std::size_t(t) % pr.size()];
        }
        i64 f = pl;   // tok[< f] equals the autoregressive sample
        for (i64 p = 0; p < passes; ++p) {
            std::span<const float> L = run(std::span<const int>(tok));
            std::vector<int> nt(tok);
            la::parallel_for(static_cast<std::size_t>(seq - pl), 4, [&](std::size_t b, std::size_t e) {
                std::vector<std::pair<float, int>> cand;
                for (std::size_t j = b; j < e; ++j) {
                    const i64 t = pl + i64(j);   // position resampled from row t - 1
                    nt[std::size_t(t)] = sample_nucleus(L.data() + (t - 1) * V, V, position_key(seed, i, std::uint64_t(t)), top_p, cand);
                }
            });
            i64 ch = 0; for (i64 t = pl; t < seq; ++t) ch += nt[std::size_t(t)] != tok[std::size_t(t)];
            // nt[f] came from an exact prefix; so did nt[f + j] while tok[f .. f + j) did not change
            if (f < seq) { i64 g = f + 1; while (g < seq && nt[std::size_t(g - 1)] == tok[std::size_t(g - 1)]) ++g; f = g; }
            tok.swap(nt);
            st.changed[std::size_t(p)] += ch; ++st.passes;
            if (ch == 0) { f = seq; break; }   // the fixed point: the autoregressive sample
        }
        st.tokens += seq - pl; st.exact += f - pl;
        c.seqs.push_back(std::move(tok));
    }
    st.seconds = now_s() - t0;
    if (stats) *stats = st;
    return c;
}

// On the model's live weights (FORGE graph Runner, f32).
inline CalibSet jacobi_generate(Model& m, i64 n, i64 seq, i64 passes = 6, std::uint64_t seed = 0, JacobiStats* stats = nullptr,
                                double top_p = 0.95, const std::string& init = "repeat") {
    Runner r(m.spec, m.w, seq, {}, true);
    auto run = [&](std::span<const int> ids) { r.run_tokens(ids); return r.output(); };
    return jacobi_core(run, m.spec.vocab, self_prompts(m, n), seq, passes, seed, top_p, init, stats);
}

// On a model file (forge::inference::session forward_all: the file's codecs and FORGE's kernels).
inline CalibSet jacobi_generate(const std::string& fpk, i64 n, i64 seq, i64 passes = 6, std::uint64_t seed = 0,
                                JacobiStats* stats = nullptr, double top_p = 0.95, const std::string& init = "repeat") {
    inference::session_options o; o.context = seq + 8;
    inference::session s(fpk, o);
    if (!s.tokenizer()) die("jacobi_generate: the model has no tokenizer");
    const auto prompts = self_prompts(s.manifest(), [&s](const std::string& t) { return s.tokenizer()->encode(t); }, n);
    std::vector<float> L;
    auto run = [&](std::span<const int> ids) { s.reset(); L = s.forward_all(ids); return std::span<const float>(L); };
    return jacobi_core(run, s.vocab(), prompts, seq, passes, seed, top_p, init, stats);
}

// Self-generated text by `mode`: "decode" (default: continuous-batching decoding of the model file) or "jacobi"
// (`passes` full-sequence prefill passes on the live weights).
inline CalibSet self_text(const std::string& fpk, Model& m, i64 n, i64 seq, std::uint64_t seed, const std::string& mode = "decode", i64 passes = 6) {
    if (mode == "decode") return self_generate(fpk, m, n, seq, seed);
    if (mode != "jacobi") die("--self-mode must be jacobi or decode, not " + mode);
    JacobiStats st;
    auto c = jacobi_generate(m, n, seq, passes, seed, &st);
    say("  jacobi self-sampling: {} sequences x {} tokens, {} passes, {:.2f} ms/token, {:.1f} % of the positions exact", n, seq, st.passes,
        1e3 * st.seconds / double(std::max<i64>(st.tokens, 1)), 100 * st.exact_fraction());
    return c;
}

inline double row_lse(const float* z, i64 V, double t = 1.0) {
    double mx = -INFINITY; for (i64 j = 0; j < V; ++j) mx = std::max(mx, t * z[j]);
    double s = 0; for (i64 j = 0; j < V; ++j) s += std::exp(t * z[j] - mx);
    return mx + std::log(s);
}

// Reference next-token distributions of the uncompressed model on probe sequences.
struct Reference {
    CalibSet probes;
    i64 vocab = 0;
    std::vector<std::vector<std::uint16_t>> logp;   // per sequence [seq, vocab] f16 log-probabilities
    bool empty() const { return probes.seqs.empty(); }
    float at(std::size_t s, i64 t, i64 v) const { return float(f16::from_bits(logp[s][std::size_t(t * vocab + v)])); }
};

inline Reference capture_reference(Model& m, const CalibSet& probes) {
    Reference R; R.probes = probes; R.vocab = m.spec.vocab;
    Runner r(m.spec, m.w, probes.seq, {}, true);
    for (auto& ids : probes.seqs) {
        r.run_tokens(ids);
        auto z = r.output();
        std::vector<std::uint16_t> lp(z.size());
        la::parallel_for(static_cast<std::size_t>(probes.seq), 4, [&](std::size_t b, std::size_t e) {
            for (std::size_t t = b; t < e; ++t) {
                const float* row = z.data() + i64(t) * R.vocab;
                const double lse = row_lse(row, R.vocab);
                for (i64 v = 0; v < R.vocab; ++v) lp[t * std::size_t(R.vocab) + std::size_t(v)] = f16(float(row[v] - lse)).bits;
            }
        });
        R.logp.push_back(std::move(lp));
    }
    return R;
}

// Model logits on the probes (all sequences, [n][seq*vocab]).
inline std::vector<std::vector<float>> probe_logits(Model& m, const Reference& R) {
    Runner r(m.spec, m.w, R.probes.seq, {}, true);
    std::vector<std::vector<float>> out;
    for (auto& ids : R.probes.seqs) { r.run_tokens(ids); auto z = r.output(); out.emplace_back(z.begin(), z.end()); }
    return out;
}

// mean KL(ref || softmax(t * z + b)) over the probe sequences in [s0, s1)
inline double kl_of(const Reference& R, const std::vector<std::vector<float>>& Z, double t, const std::vector<float>* bias, std::size_t s0, std::size_t s1) {
    const i64 V = R.vocab, T = R.probes.seq;
    double tot = 0; i64 cnt = 0;
    for (std::size_t s = s0; s < s1; ++s) {
        std::vector<double> part(static_cast<std::size_t>(T), 0.0);
        la::parallel_for(static_cast<std::size_t>(T), 4, [&](std::size_t b, std::size_t e) {
            std::vector<float> zz(static_cast<std::size_t>(V));
            for (std::size_t i = b; i < e; ++i) {
                const float* z = Z[s].data() + i64(i) * V;
                for (i64 v = 0; v < V; ++v) zz[std::size_t(v)] = float(t * z[v] + (bias ? (*bias)[std::size_t(v)] : 0.f));
                const double lse = row_lse(zz.data(), V);
                double k = 0;
                for (i64 v = 0; v < V; ++v) { const double lr = R.at(s, i64(i), v); k += std::exp(lr) * (lr - (zz[std::size_t(v)] - lse)); }
                part[i] = k;
            }
        });
        for (double x : part) tot += x;
        cnt += T;
    }
    return tot / double(std::max<i64>(cnt, 1));
}

inline double kl(Model& m, const Reference& R) {
    auto Z = probe_logits(m, R);
    return kl_of(R, Z, 1.0, nullptr, 0, R.probes.seqs.size());
}

// Fits a logit temperature and a logit bias on the first half of the probes, keeps each if it lowers the KL on the
// second half. The temperature is folded into out_norm; the bias becomes the tensor `logit_bias`.
inline value calibrate_distribution(Model& m, const Reference& R, bool use_bias = true) {
    auto Z = probe_logits(m, R);
    const std::size_t n = R.probes.seqs.size(), half = std::max<std::size_t>(1, n / 2);
    const std::size_t h0 = n > 1 ? half : 0;   // held-out range [h0, n)
    const double base_fit = kl_of(R, Z, 1.0, nullptr, 0, half), base_hold = kl_of(R, Z, 1.0, nullptr, h0, n);
    // golden-section search for t in [0.7, 1.4]
    double a = 0.7, b = 1.4;
    const double g = (std::sqrt(5.0) - 1) / 2;
    double c = b - g * (b - a), d = a + g * (b - a);
    double fc = kl_of(R, Z, c, nullptr, 0, half), fd = kl_of(R, Z, d, nullptr, 0, half);
    for (int it = 0; it < 14; ++it) {
        if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = kl_of(R, Z, c, nullptr, 0, half); }
        else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = kl_of(R, Z, d, nullptr, 0, half); }
    }
    double t = (a + b) / 2;
    const double t_hold = kl_of(R, Z, t, nullptr, h0, n);
    const bool keep_t = t_hold < base_hold;
    if (!keep_t) t = 1.0;
    // logit bias: fixed point on mean probabilities, shrunk by the evidence each token has
    const i64 V = R.vocab, T = R.probes.seq;
    std::vector<float> bias(static_cast<std::size_t>(V), 0.f);
    double b_hold = keep_t ? t_hold : base_hold;
    bool keep_b = false;
    if (use_bias) {
        for (int it = 0; it < 3; ++it) {
            std::vector<double> pr(static_cast<std::size_t>(V), 0.0), pm(static_cast<std::size_t>(V), 0.0);
            for (std::size_t s = 0; s < half; ++s) for (i64 i = 0; i < T; ++i) {
                const float* z = Z[s].data() + i * V;
                std::vector<float> zz(static_cast<std::size_t>(V));
                for (i64 v = 0; v < V; ++v) zz[std::size_t(v)] = float(t * z[v] + bias[std::size_t(v)]);
                const double lse = row_lse(zz.data(), V);
                for (i64 v = 0; v < V; ++v) { pr[std::size_t(v)] += std::exp(double(R.at(s, i, v))); pm[std::size_t(v)] += std::exp(zz[std::size_t(v)] - lse); }
            }
            const double N = double(half) * double(T);
            for (i64 v = 0; v < V; ++v) {
                const double r = pr[std::size_t(v)] / N, q = pm[std::size_t(v)] / N;
                const double evidence = (pr[std::size_t(v)] + pm[std::size_t(v)]) / (pr[std::size_t(v)] + pm[std::size_t(v)] + 1.0);   // ~ expected count / (count + 1)
                bias[std::size_t(v)] += float(evidence * (std::log(r + 1e-9) - std::log(q + 1e-9)));
            }
        }
        const double kb = kl_of(R, Z, t, &bias, h0, n);
        keep_b = kb < b_hold;
        if (keep_b) b_hold = kb;
    }
    if (keep_t) for (auto& x : m.w.at("out_norm").data) x = float(x * t);
    if (keep_b) { tensor tb({V}); tb.data = bias; m.w.t["logit_bias"] = tb; m.spec.logit_bias = true; }
    return value(jobj{{"kl_before", value(base_hold)}, {"temperature", value(t)}, {"temperature_kept", value(keep_t)}, {"logit_bias_kept", value(keep_b)},
                      {"kl_after", value(b_hold)}, {"kl_fit_before", value(base_fit)}});
}

}  // namespace forge::compress
