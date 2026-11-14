// Evaluation of a model file with FORGE's inference engine (forge::inference::session, the same path forge run and
// forge-serve use):
//   * perplexity on WikiText-2 (non-overlapping windows, evenly spaced over the test split)
//   * zero-shot multiple choice by log-likelihood (lm-evaluation-harness conventions: acc, or acc_norm by
//     continuation byte length for ARC / HellaSwag / OpenBookQA; partial scoring for WinoGrande)
// Sequences are padded to a few length buckets (causal attention makes the padding invisible to earlier rows and the
// KV cache is rewound past it), so FORGE compiles a handful of plans instead of one per length.
#pragma once
#include <forge/compress/calib.hpp>
#include <forge/compress/mixer.hpp>
#include <forge/inference/session.hpp>
#include <forge/inference/speculative.hpp>

#include <array>
#include <optional>
#include <unordered_map>

namespace forge::compress {

struct EvalCfg {
    std::string data_dir;
    i64 ppl_window = 512;
    i64 ppl_windows = 40;   // 0 = the whole test split
    std::vector<std::string> tasks;
    i64 task_limit = 200;
    bool mix = false;
    std::string kv_codec;    // session_options::kv_codec of the evaluation sessions ("" = f32 / $FORGE_KV_CODEC)
    // speculative decoding with a (compressed) draft model: greedy, prompts = WikiText windows
    std::string draft;
    i64 spec_tokens = 64;    // new tokens per prompt
    i64 spec_prompts = 8;    // prompts (evenly spaced windows of the test split)
    i64 spec_prompt_len = 128;
    int spec_k = 4;          // drafts per pass (0 = FORGE's planner)
};

struct SpecResult {
    i64 prompts = 0, new_tokens = 0, identical = 0;   // identical: prompts whose speculative output equals plain greedy
    double acceptance = 0, tokens_per_pass = 0;
    double plain_tps = 0, spec_tps = 0, speedup = 0;  // decode tokens/s (prefill excluded) and spec / plain
    double plain_seconds = 0, spec_seconds = 0;       // wall time, prefill included
    bool lossless() const { return identical == prompts; }
};

struct EvalResult {
    double ppl = 0, nll = 0;
    double ppl_mix = 0, nll_mix = 0;   // with the online context mixer (eval --mix 1)
    i64 ppl_tokens = 0;
    std::map<std::string, double> acc;
    std::map<std::string, i64> n;
    double mean_acc = 0;
    double seconds = 0;
    std::string kv_codec;
    std::optional<SpecResult> spec;
};

inline const std::vector<i64>& buckets() { static const std::vector<i64> b{8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512}; return b; }
inline i64 bucket_of(i64 n) { for (i64 b : buckets()) if (n <= b) return b; return n; }

// log-softmax of row `r` at token `t`, in f64.
inline double logprob(const float* row, i64 vocab, int t) {
    double m = -INFINITY; for (i64 j = 0; j < vocab; ++j) m = std::max(m, double(row[j]));
    double s = 0; for (i64 j = 0; j < vocab; ++j) s += std::exp(double(row[j]) - m);
    return double(row[t]) - m - std::log(s);
}

// Online context mixing (mixer.hpp): the network's next-token distribution mixed with cheap adaptive models of the
// text seen so far. ContextMixer scores observed tokens (evaluation: only log p_mix(y) is needed, from the network's
// log-probabilities of y); StreamMixer (generation) gives the full mixed distribution with the same models and
// updates. Only past tokens are used: a valid online predictor.
struct ContextMixer : MixModels {
    // lp_net[0..2]: log p of the observed token y under components 0..2; returns log p_mix(y) and learns from y
    double step(const double lp_net[3], int y) {
        const Query q = query();
        select(q);
        double p[K];
        for (int i = 0; i < 3; ++i) p[i] = std::exp(lp_net[i]);
        text_probs(q, y, p[0], p + 3);
        const auto& ws = w[std::size_t(q.set)];
        double pm = 0; for (int i = 0; i < K; ++i) pm += ws[std::size_t(i)] * p[i];
        pm = std::max(pm, 1e-300);
        learn(q, p, pm);
        push(y);
        return std::log(pm);
    }
};

inline double logprob_scaled(const float* row, i64 vocab, int t, double s) {
    double m = -INFINITY; for (i64 j = 0; j < vocab; ++j) m = std::max(m, s * row[j]);
    double z = 0; for (i64 j = 0; j < vocab; ++j) z += std::exp(s * row[j] - m);
    return s * row[t] - m - std::log(z);
}

class Evaluator {
public:
    Evaluator(const std::string& fpk, i64 context = 1024, const std::string& kv_codec = "") : s_(fpk, session_opts(context, kv_codec)), vocab_(s_.vocab()) {
        const auto& man = s_.manifest();
        add_bos_ = man.find("add_bos") && man["add_bos"].is_bool() && man["add_bos"].b();
        bos_ = man.find("bos_token_id") && man["bos_token_id"].is_num() ? int(man["bos_token_id"].num()) : -1;
    }

    std::vector<int> encode(const std::string& text, bool bos) const {
        auto v = s_.tokenizer()->encode(text);
        if (bos && add_bos_ && bos_ >= 0) v.insert(v.begin(), bos_);
        return v;
    }

    // Runs `ids` from the current position (padded to a bucket) and returns the [ids.size(), vocab] logits.
    std::span<const float> run(std::span<const int> ids) {
        const i64 n = i64(ids.size()), pos = s_.position();
        const i64 b = bucket_of(n);
        buf_.assign(ids.begin(), ids.end());
        buf_.resize(static_cast<std::size_t>(b), ids.empty() ? 0 : ids.back());
        out_ = s_.forward_all(buf_);
        s_.rewind(pos + n);
        return {out_.data(), std::size_t(n * vocab_)};
    }

    void reset() { s_.reset(); }
    i64 vocab() const { return vocab_; }

    // perplexity with the online context mixer (see ContextMixer); windows are independent
    // returns {plain nll, mixed nll}
    std::pair<double, double> ppl_mix(const std::vector<int>& toks, i64 window, i64 n_windows) {
        const i64 total = i64(toks.size()) / window;
        std::vector<i64> pick;
        if (n_windows <= 0 || n_windows >= total) for (i64 i = 0; i < total; ++i) pick.push_back(i);
        else for (i64 i = 0; i < n_windows; ++i) pick.push_back(i * total / n_windows);
        double nll = 0, plain = 0; i64 cnt = 0;
        for (i64 w : pick) {
            reset();
            std::span<const int> ids(toks.data() + w * window, std::size_t(window));
            auto lg = run(ids);
            std::vector<std::array<double, 3>> lp(std::size_t(window - 1));
            la::parallel_for(std::size_t(window - 1), 8, [&](std::size_t b, std::size_t e) {
                for (std::size_t t = b; t < e; ++t) {
                    const float* row = lg.data() + i64(t) * vocab_;
                    lp[t] = {logprob(row, vocab_, ids[t + 1]), logprob_scaled(row, vocab_, ids[t + 1], 1.25), logprob_scaled(row, vocab_, ids[t + 1], 0.8)};
                }
            });
            ContextMixer cm;
            cm.hist.push_back(ids[0]); cm.uni[ids[0]] = 1;
            for (i64 t = 0; t + 1 < window; ++t) { nll -= cm.step(lp[std::size_t(t)].data(), ids[std::size_t(t + 1)]); plain -= lp[std::size_t(t)][0]; }
            cnt += window - 1;
        }
        return {plain / double(cnt), nll / double(cnt)};
    }

    std::pair<double, i64> ppl(const std::vector<int>& toks, i64 window, i64 n_windows) {
        const i64 total = i64(toks.size()) / window;
        std::vector<i64> pick;
        if (n_windows <= 0 || n_windows >= total) for (i64 i = 0; i < total; ++i) pick.push_back(i);
        else for (i64 i = 0; i < n_windows; ++i) pick.push_back(i * total / n_windows);
        double nll = 0; i64 cnt = 0;
        for (i64 w : pick) {
            reset();
            std::span<const int> ids(toks.data() + w * window, std::size_t(window));
            auto lg = run(ids);
            std::vector<double> part(static_cast<std::size_t>(window - 1));
            la::parallel_for(static_cast<std::size_t>(window - 1), 8, [&](std::size_t b, std::size_t e) {
                for (std::size_t t = b; t < e; ++t) part[t] = -logprob(lg.data() + i64(t) * vocab_, vocab_, ids[t + 1]);
            });
            for (double x : part) nll += x;
            cnt += window - 1;
        }
        return {nll / double(cnt), cnt};
    }

    // Sum log p(cont | ctx) for one choice; ctx logits are reused when `ctx_cached` holds ctx_ids already run.
    double score(const std::vector<int>& ctx_ids, const std::vector<int>& cont_ids, const std::vector<float>& ctx_last) {
        double lp = logprob(ctx_last.data(), vocab_, cont_ids[0]);
        if (cont_ids.size() > 1) {
            s_.rewind(i64(ctx_ids.size()));
            std::span<const int> rest(cont_ids.data(), cont_ids.size() - 1);
            auto lg = run(rest);
            for (std::size_t t = 1; t < cont_ids.size(); ++t) lp += logprob(lg.data() + i64(t - 1) * vocab_, vocab_, cont_ids[t]);
        }
        return lp;
    }

    double task(const std::string& path, i64 limit, i64* n_out) {
        auto items = read_jsonl(path, std::size_t(limit));
        i64 correct = 0;
        for (auto& it : items) {
            const auto& ctxs = it["ctxs"].arr();
            const auto& conts = it["conts"].arr();
            const bool norm = it.find("norm") && it["norm"].b();
            const i64 label = it["label"].i();
            std::vector<double> sc;
            std::string cached_ctx; std::vector<int> ctx_ids; std::vector<float> last;
            for (std::size_t c = 0; c < conts.size(); ++c) {
                const std::string& ctx = ctxs.size() == 1 ? ctxs[0].str() : ctxs[c].str();
                const std::string& cont = conts[c].str();
                auto cenc = encode(ctx, true);
                auto whole = encode(ctx + cont, true);
                std::vector<int> cont_ids;
                if (whole.size() > cenc.size() && std::equal(cenc.begin(), cenc.end(), whole.begin())) cont_ids.assign(whole.begin() + i64(cenc.size()), whole.end());
                else cont_ids = encode(cont, false);
                // keep at most 512 tokens: drop the oldest context tokens
                const i64 maxlen = 512;
                if (i64(cenc.size() + cont_ids.size()) > maxlen) cenc.erase(cenc.begin(), cenc.begin() + (i64(cenc.size() + cont_ids.size()) - maxlen));
                if (cont_ids.empty()) { sc.push_back(-INFINITY); continue; }
                if (ctx != cached_ctx || ctx_ids != cenc) {
                    reset();
                    auto lg = run(cenc);
                    last.assign(lg.end() - vocab_, lg.end());
                    ctx_ids = cenc; cached_ctx = ctx;
                }
                double lp = score(ctx_ids, cont_ids, last);
                sc.push_back(norm ? lp / double(std::max<std::size_t>(1, cont.size())) : lp);
            }
            const i64 best = i64(std::max_element(sc.begin(), sc.end()) - sc.begin());
            correct += best == label;
        }
        if (n_out) *n_out = i64(items.size());
        return items.empty() ? 0.0 : double(correct) / double(items.size());
    }

private:
    static forge::inference::session_options session_opts(i64 ctx, const std::string& kv) { forge::inference::session_options o; o.context = ctx; o.kv_codec = kv; return o; }
    forge::inference::session s_;
    i64 vocab_;
    bool add_bos_ = false;
    int bos_ = -1;
    std::vector<int> buf_;
    std::vector<float> out_;
};

// Tokenised WikiText-2 test split, cached per tokenizer.
inline std::vector<int> ppl_tokens(Evaluator& ev, const std::string& data_dir, const std::string& tok_key) {
    const std::string cache = data_dir + "/cache/wt2_test_" + tok_key + ".i32";
    if (std::filesystem::exists(cache)) {
        auto raw = read_file(cache);
        std::vector<int> v(raw.size() / 4);
        std::memcpy(v.data(), raw.data(), v.size() * 4);
        return v;
    }
    auto v = ev.encode(read_file(data_dir + "/wikitext2_test.txt"), true);
    std::filesystem::create_directories(data_dir + "/cache");
    std::ofstream f(cache, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
    return v;
}

inline std::string tokenizer_key(const std::string& fpk) {
    forge::modelfile::reader r(fpk);
    std::string t = r.has_section("tokenizer") ? std::string(r.section_text("tokenizer")) : std::string(r.section_text("tokenizer_json"));
    return std::format("{:016x}", std::hash<std::string>{}(t));
}

// Greedy speculative decoding with `draft_fpk` as the draft (FORGE's speculative_generate) against plain greedy
// decoding of the target, on the given prompts: acceptance, tokens/s, and whether the outputs are identical (greedy
// verification is lossless: they should be, up to batched-vs-single-row rounding of near ties).
inline SpecResult speculative_eval(const std::string& target_fpk, const std::string& draft_fpk, const std::vector<std::vector<int>>& prompts,
                                   i64 n_new, int k, const std::string& kv_codec = "") {
    namespace inf = forge::inference;
    i64 longest = 0; for (auto& p : prompts) longest = std::max<i64>(longest, i64(p.size()));
    inf::session_options o; o.context = longest + n_new + std::max(k, 16) + 8; o.kv_codec = kv_codec;
    inf::session target(target_fpk, o), draft(draft_fpk, o);
    SpecResult R;
    double plain_decode = 0, spec_decode = 0;
    std::size_t drafted = 0, accepted = 0, passes = 0, spec_new = 0;
    for (auto& p : prompts) {
        if (p.empty()) continue;
        inf::sampler g1;   // greedy
        inf::generate_stats gs;
        target.reset();
        double t0 = now_s();
        auto plain = target.generate(p, int(n_new), g1, {}, &gs);
        R.plain_seconds += now_s() - t0;
        plain_decode += gs.decode_s;
        inf::sampler g2;
        inf::speculative_stats st;
        t0 = now_s();
        auto spec = inf::speculative_generate(target, draft, p, int(n_new), g2, k, {}, &st);
        R.spec_seconds += now_s() - t0;
        spec_decode += st.seconds - st.prefill_seconds;
        drafted += st.drafted; accepted += st.accepted; passes += st.passes; spec_new += st.new_tokens;
        R.new_tokens += i64(plain.size());
        R.identical += plain == spec;
        ++R.prompts;
    }
    R.acceptance = drafted ? double(accepted) / double(drafted) : 0;
    R.tokens_per_pass = passes ? double(spec_new) / double(passes) : 0;
    R.plain_tps = plain_decode > 0 ? double(R.new_tokens) / plain_decode : 0;
    R.spec_tps = spec_decode > 0 ? double(spec_new) / spec_decode : 0;
    R.speedup = R.plain_tps > 0 ? R.spec_tps / R.plain_tps : 0;
    return R;
}

// `n` prompts of `len` tokens: windows evenly spaced over `toks`
inline std::vector<std::vector<int>> spaced_windows(const std::vector<int>& toks, i64 n, i64 len) {
    std::vector<std::vector<int>> out;
    const i64 total = i64(toks.size()) / std::max<i64>(len, 1), m = std::min(n, total);
    for (i64 i = 0; i < m; ++i) { const i64 w = i * total / m; out.emplace_back(toks.begin() + w * len, toks.begin() + (w + 1) * len); }
    return out;
}

inline EvalResult evaluate(const std::string& fpk, const EvalCfg& c) {
    const double t0 = now_s();
    EvalResult R;
    R.kv_codec = c.kv_codec;
    Evaluator ev(fpk, std::max<i64>(1024, c.ppl_window), c.kv_codec);
    if (!c.kv_codec.empty()) say("  KV cache codec {}", c.kv_codec);
    if (c.ppl_windows >= 0) {
        auto toks = ppl_tokens(ev, c.data_dir, tokenizer_key(fpk));
        if (c.mix) {   // one pass: the plain and the context-mixed perplexity
            auto [pl, mx] = ev.ppl_mix(toks, c.ppl_window, c.ppl_windows);
            const i64 total = i64(toks.size()) / c.ppl_window;
            R.nll = pl; R.ppl = std::exp(pl); R.ppl_tokens = (c.ppl_windows <= 0 || c.ppl_windows >= total ? total : c.ppl_windows) * (c.ppl_window - 1);
            R.nll_mix = mx; R.ppl_mix = std::exp(mx);
        } else {
            auto [nll, n] = ev.ppl(toks, c.ppl_window, c.ppl_windows);
            R.nll = nll; R.ppl = std::exp(nll); R.ppl_tokens = n;
        }
        say("  ppl {:.4f} over {} tokens", R.ppl, R.ppl_tokens);
        if (c.mix) say("  ppl with context mixing {:.4f}", R.ppl_mix);
    }
    double sum = 0;
    for (auto& t : c.tasks) {
        i64 n = 0;
        double a = ev.task(c.data_dir + "/tasks/" + t + ".jsonl", c.task_limit, &n);
        R.acc[t] = a; R.n[t] = n; sum += a;
        say("  {} acc {:.4f} (n={})", t, a, n);
    }
    R.mean_acc = c.tasks.empty() ? 0 : sum / double(c.tasks.size());
    if (!c.draft.empty()) {
        auto toks = ppl_tokens(ev, c.data_dir, tokenizer_key(fpk));
        auto prompts = spaced_windows(toks, c.spec_prompts, c.spec_prompt_len);
        R.spec = speculative_eval(fpk, c.draft, prompts, c.spec_tokens, c.spec_k, c.kv_codec);
        const auto& S = *R.spec;
        say("  speculative ({} drafts / pass, draft {}): acceptance {:.3f}, {:.2f} tokens/pass, {:.1f} tok/s vs {:.1f} plain (x{:.2f}), identical {}/{}",
            c.spec_k, c.draft, S.acceptance, S.tokens_per_pass, S.spec_tps, S.plain_tps, S.speedup, S.identical, S.prompts);
    }
    R.seconds = now_s() - t0;
    return R;
}

}  // namespace forge::compress
