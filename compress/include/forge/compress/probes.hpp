// Data-free probes of the model's global function and corrections of its output distribution.
//
//   self_generate     calibration text written by the model itself (FORGE continuous batching, sampled at T=1,
//                     nucleus 0.95): in-distribution for the model, no dataset needed
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

namespace forge::compress {

inline CalibSet self_generate(const std::string& fpk, const Model& m, i64 n, i64 seq, std::uint64_t seed = 0, int parallel = 16) {
    auto model = inference::model::open(fpk, false);
    inference::batch_options bo; bo.max_active = parallel; bo.session.context = seq + 8;
    inference::batch_scheduler sch(model, bo);
    const bool bos = m.manifest.find("add_bos") && m.manifest["add_bos"].is_bool() && m.manifest["add_bos"].b();
    const int bos_id = m.manifest.find("bos_token_id") && m.manifest["bos_token_id"].is_num() ? int(m.manifest["bos_token_id"].num()) : -1;
    static const char* starts[] = {"The", "In", "A", "It", "This", "When", "However,", "There", "One", "We", "He", "She", "After", "According to", "Today", "If"};
    CalibSet c; c.seq = seq;
    std::vector<std::vector<int>> outs(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) {
        inference::request r;
        r.prompt = m.encode(starts[i % 16]);
        if (bos && bos_id >= 0) r.prompt.insert(r.prompt.begin(), bos_id);
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
