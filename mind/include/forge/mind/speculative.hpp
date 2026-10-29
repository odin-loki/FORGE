#pragma once
// Speculative decoding (Leviathan et al. / Chen et al.): a cheap draft model proposes `k` tokens, the
// target scores all of them in one forward_all pass, and modified rejection sampling keeps the output
// distribution exactly that of the target. With a greedy sampler the output is identical to target greedy.
#include <forge/mind/session.hpp>

namespace forge::mind {

struct speculative_stats { std::size_t rounds = 0, drafted = 0, accepted = 0, new_tokens = 0; double seconds = 0;
                           double acceptance() const { return drafted ? double(accepted) / double(drafted) : 0; } };

inline std::vector<int> speculative_generate(session& target, session& draft, std::span<const int> prompt, int max_new,
                                             sampler& s, int k = 4, const std::function<bool(int)>& on_token = {},
                                             speculative_stats* stats = nullptr) {
    target.check_generative(); draft.check_generative();
    if (prompt.empty()) raise(errc::invalid_argument, "speculative_generate: empty prompt");
    if (target.vocab() != draft.vocab()) raise(errc::invalid_argument, "draft vocab {} != target vocab {}", draft.vocab(), target.vocab());
    k = std::max(k, 1);
    auto t0 = std::chrono::steady_clock::now();
    speculative_stats st;
    target.reset(); draft.reset();
    std::vector<int> hist(prompt.begin(), prompt.end()), out;
    if (prompt.size() > 1) {
        auto pre = prompt.subspan(0, prompt.size() - 1);
        target.forward(pre); draft.forward(pre);
    }
    int cur = prompt.back();                  // fed to neither model yet
    int eos = target.tokenizer() ? target.tokenizer()->eos() : -1;
    std::size_t V = std::size_t(target.vocab());
    bool done = false;
    auto emit = [&](int t) {
        out.push_back(t); hist.push_back(t);
        if ((on_token && !on_token(t)) || t == eos || int(out.size()) >= max_new) done = true;
    };
    while (!done) {
        i64 P = target.position();
        int kk = int(std::min<i64>(k, std::min<i64>(max_new - i64(out.size()), target.context() - P - 1)));
        if (kk <= 0) {   // no room to speculate: plain step
            auto l = target.forward(std::span<const int>(&cur, 1));
            draft.forward(std::span<const int>(&cur, 1));
            int t = s.sample(l, hist); emit(t); cur = t;
            if (target.position() + 1 > target.context()) break;
            continue;
        }
        // 1. draft proposes kk tokens
        std::vector<int> d; std::vector<std::vector<double>> q;
        std::vector<int> dh = hist;
        int feed = cur;
        for (int i = 0; i < kk; ++i) {
            auto l = draft.forward(std::span<const int>(&feed, 1));
            q.push_back(s.probs(l, dh));
            int t = s.draw(q.back());
            d.push_back(t); dh.push_back(t); feed = t;
        }
        // 2. target scores cur + drafts in one pass
        std::vector<int> batch; batch.push_back(cur); batch.insert(batch.end(), d.begin(), d.end());
        auto L = target.forward_all(batch);
        ++st.rounds; st.drafted += std::size_t(kk);
        // 3. accept / reject
        int acc = 0; int next = -1;
        std::vector<int> th = hist;
        for (int i = 0; i < kk && !done; ++i) {
            auto p = s.probs(std::span<const float>(L.data() + std::size_t(i) * V, V), th);
            double pi = p[std::size_t(d[std::size_t(i)])], qi = q[std::size_t(i)][std::size_t(d[std::size_t(i)])];
            if (qi > 0 && (pi >= qi || s.uniform() < pi / qi)) {
                ++acc; th.push_back(d[std::size_t(i)]); emit(d[std::size_t(i)]);
                continue;
            }
            std::vector<double> r(V);
            for (std::size_t j = 0; j < V; ++j) r[j] = std::max(0.0, p[j] - q[std::size_t(i)][j]);
            next = s.draw(r);
            break;
        }
        st.accepted += std::size_t(acc);
        if (done) {   // stopped inside the accepted run: keep state consistent with what was emitted
            target.rewind(P + 1 + acc); draft.rewind(P + std::min(acc + 1, kk));
            break;
        }
        if (next < 0) {   // all accepted: bonus token from the target's last row
            auto p = s.probs(std::span<const float>(L.data() + std::size_t(kk) * V, V), th);
            next = s.draw(p);
            draft.forward(std::span<const int>(&d.back(), 1));   // draft never saw its own last proposal
        } else {
            target.rewind(P + 1 + acc);
            draft.rewind(P + 1 + acc);
        }
        emit(next); cur = next;
    }
    st.new_tokens = out.size();
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (stats) *stats = st;
    return out;
}

} // namespace forge::mind
