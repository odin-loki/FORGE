#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/mind/mind.hpp>
#include <filesystem>

using namespace forge;
using namespace forge::mind;

static std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / ("forge_test_" + n)).string(); }
static const std::string& model_a() { static std::string p = forge_test::make_tiny_model(tmp("a.fpk")); return p; }
static const std::string& model_b() { static std::string p = forge_test::make_tiny_model(tmp("b.fpk"), {.seed = 99}); return p; }
static const std::string& model_q() { static std::string p = forge_test::make_tiny_model(tmp("q.fpk"), {.codec = "q8_0"}); return p; }

static std::vector<int> greedy(session& s, std::vector<int> prompt, int n) { sampler g; s.reset(); return s.generate(prompt, n, g); }
static float maxdiff(const std::vector<float>& a, const std::vector<float>& b) { float m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i])); return m; }

TEST(paged_kv_matches_preallocated) {
    auto m = model::open(model_a());
    session flat(m, {.page_rows = 0}), paged(m, {.page_rows = 4});
    std::vector<int> p = {1, 5, 9, 33, 70, 101, 7};
    CHECK(greedy(flat, p, 20) == greedy(paged, p, 20));
    // paged storage tracks usage: 27 rows held -> 28 rows reserved per slot (7 pages of 4)
    for (auto& [k, sl] : paged.state().slots) CHECK_MSG(sl.capacity == 28, "%s cap %lld", k.c_str(), (long long)sl.capacity);
    for (auto& [k, sl] : flat.state().slots) CHECK(sl.capacity == 512);
    CHECK(paged.state().bytes() * 10 < flat.state().bytes());
}

TEST(forward_all_rows_match_incremental) {
    auto m = model::open(model_a());
    session a(m), b(m);
    std::vector<int> p = {3, 1, 4, 1, 5, 9, 2, 6};
    auto all = a.forward_all(p);
    std::size_t V = std::size_t(a.vocab());
    CHECK(all.size() == p.size() * V);
    for (std::size_t i = 0; i < p.size(); ++i) {
        auto l = b.forward(std::span<const int>(&p[i], 1));
        CHECK_MSG(maxdiff(l, {all.begin() + std::ptrdiff_t(i * V), all.begin() + std::ptrdiff_t((i + 1) * V)}) < 1e-4f, "row %zu", i);
    }
}

TEST(rewind_and_copy_prefix) {
    auto m = model::open(model_a());
    session a(m), b(m);
    std::vector<int> p = {10, 20, 30, 40, 50};
    a.forward(p);
    auto l1 = a.forward(std::vector<int>{60});
    a.rewind(5);
    CHECK(a.position() == 5);
    auto l2 = a.forward(std::vector<int>{60});
    CHECK(maxdiff(l1, l2) == 0.f);
    b.copy_prefix(a, 5);
    auto l3 = b.forward(std::vector<int>{60});
    CHECK(maxdiff(l1, l3) == 0.f);
}

TEST(speculative_greedy_equals_target_greedy) {
    auto tm = model::open(model_a());
    std::vector<int> p = {256, 72, 101, 108, 108, 111};
    session ref(tm); auto want = greedy(ref, p, 40);
    for (auto* dpath : {&model_a(), &model_q(), &model_b()}) {
        for (int k : {1, 3, 6}) {
            session t(tm), d(model::open(*dpath));
            sampler g; speculative_stats st;
            auto got = speculative_generate(t, d, p, 40, g, k, {}, &st);
            CHECK_MSG(got == want, "draft %s k=%d", dpath->c_str(), k);
            if (dpath == &model_a()) CHECK_MSG(st.accepted == st.drafted, "self-draft acceptance %zu/%zu", st.accepted, st.drafted);
            if (k == 3) std::printf("  draft=%s k=%d rounds=%zu acceptance=%.2f\n", std::filesystem::path(*dpath).filename().c_str(), k, st.rounds, st.acceptance());
        }
    }
}

TEST(speculative_sampling_preserves_distribution) {
    // first sampled token after a fixed prompt: empirical distribution with a mismatched draft must match the target
    auto tm = model::open(model_a()), dm = model::open(model_b());
    std::vector<int> p = {256, 65, 66};
    session t(tm), d(dm), ref(tm);
    sampler_config sc{.temperature = 1.0f, .seed = 5};
    sampler probe(sc);
    auto want = probe.probs(ref.forward(p), p);
    std::vector<double> cnt(want.size(), 0);
    int N = 3000;
    sampler s(sc);
    for (int i = 0; i < N; ++i) { auto o = speculative_generate(t, d, p, 2, s, 3); cnt[std::size_t(o[0])] += 1; }
    double tv = 0; for (std::size_t i = 0; i < want.size(); ++i) tv += std::abs(cnt[i] / N - want[i]);
    std::printf("  total variation vs target = %.3f\n", tv / 2);
    CHECK(tv / 2 < 0.08);
}

TEST(grammar_parse_and_match) {
    grammar g(R"(
        root ::= "{" ws pair ("," ws pair)* "}"   # comment
        pair ::= key ":" ws num
        key  ::= "\"" [a-z_]+ "\""
        num  ::= "-"? [0-9]+
        ws   ::= [ \t\n]*
    )");
    auto run = [&](std::string s) { auto st = g.start(); for (unsigned char c : s) { st = g.advance(st, c); if (st.empty()) return 0; } return grammar::can_end(st) ? 2 : 1; };
    CHECK(run("{\"a\":1}") == 2);
    CHECK(run("{ \"ab_c\": -12,\n \"z\":0}") == 2);
    CHECK(run("{\"a\":") == 1);
    CHECK(run("{\"A\":1}") == 0);
    CHECK(run("{\"a\":1}x") == 0);
    CHECK_THROWS(grammar("root ::= root \"a\""));
    CHECK_THROWS(grammar("root ::= missing"));
    CHECK_THROWS(grammar("root ::= \"a"));
}

TEST(grammar_constrained_generation) {
    auto m = model::open(model_a());
    auto g = std::make_shared<grammar>(R"(root ::= "{\"name\": \"" [a-z] [a-z]? [a-z]? [a-z]? [a-z]? "\", \"age\": " [1-9] [0-9]? "}")");
    for (std::uint64_t seed : {1ull, 2ull, 3ull}) {
        session s(m);
        sampler smp({.temperature = 1.2f, .seed = seed});
        grammar_constraint gc(g, *m->tokenizer(), int(m->vocab()), m->tokenizer()->eos());
        smp.filter = [&](std::span<float> l) { gc.apply(l); };
        std::vector<int> out;
        auto r = s.generate(std::vector<int>{256}, 60, smp, [&](int t) { CHECK(gc.accept(t)); out.push_back(t); return !gc.finished(); });
        auto text = m->tokenizer()->decode(out);
        std::printf("  seed %llu: %s\n", (unsigned long long)seed, text.c_str());
        CHECK(gc.finished());
        CHECK(text.starts_with("{\"name\": \"") && text.ends_with("}"));
    }
}

TEST(batch_scheduler_matches_sequential) {
    auto m = model::open(model_a());
    std::vector<std::vector<int>> prompts = {{256, 1, 2, 3}, {256, 9, 8}, {256, 1, 2, 3, 4, 5, 6}, {256, 40}, {256, 1, 2, 3, 99}, {256, 7, 7, 7, 7, 7, 7, 7, 7, 7}};
    std::vector<int> lens = {12, 5, 20, 1, 9, 15};
    std::vector<std::vector<int>> want;
    for (std::size_t i = 0; i < prompts.size(); ++i) { session s(m); want.push_back(greedy(s, prompts[i], lens[i])); }
    batch_scheduler b(m, {.max_active = 3, .prefill_chunk = 3, .prefix_cache = 4, .session = {.page_rows = 8}});
    std::vector<std::vector<int>> got(prompts.size());
    std::vector<std::string> why(prompts.size());
    for (std::size_t i = 0; i < prompts.size(); ++i)
        b.submit({.prompt = prompts[i], .max_new = lens[i], .on_done = [&, i](const std::vector<int>& o, const std::string& w) { got[i] = o; why[i] = w; }});
    b.run_until_idle();
    for (std::size_t i = 0; i < prompts.size(); ++i) CHECK_MSG(got[i] == want[i], "request %zu (%s)", i, why[i].c_str());
    auto& st = b.stats();
    std::printf("  steps=%zu peak_active=%zu prefix_hits=%zu reused=%zu peak_kv=%zuB\n", st.steps, st.peak_active, st.prefix_hits, st.prefix_tokens_reused, st.peak_kv_bytes);
    CHECK(st.completed == prompts.size());
    CHECK(st.peak_active == 3);
    CHECK(st.prefix_hits >= 1);
}

TEST(batch_scheduler_threaded) {
    auto m = model::open(model_a());
    batch_scheduler b(m, {.max_active = 4});
    b.start();
    std::mutex mu; std::condition_variable cv; int done = 0;
    for (int i = 0; i < 8; ++i)
        b.submit({.prompt = {256, i + 1}, .max_new = 6, .on_done = [&](const std::vector<int>&, const std::string&) { std::lock_guard lk(mu); ++done; cv.notify_all(); }});
    std::unique_lock lk(mu);
    CHECK(cv.wait_for(lk, std::chrono::seconds(30), [&] { return done == 8; }));
    lk.unlock();
    b.stop();
}
