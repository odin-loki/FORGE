#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/mind/mind.hpp>
#include <filesystem>
#include <unistd.h>

using namespace forge;
using namespace forge::mind;

static std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / ("forge_test_" + std::to_string(::getpid()) + "_" + n)).string(); }
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

TEST(lookup_speculation_exact) {
    auto tm = model::open(model_a());
    std::vector<int> p = {256, 72, 101, 108, 108, 111, 32, 72, 101, 108, 108, 111, 32, 72, 101};
    session ref(tm); auto want = greedy(ref, p, 40);
    for (int k : {1, 4, 8}) {
        session t(tm); sampler g; speculative_stats st;
        auto got = lookup_generate(t, p, 40, g, k, 3, {}, &st);
        CHECK_MSG(got == want, "lookup k=%d", k);
        CHECK(st.drafted > 0);
        if (k == 4) std::printf("  lookup k=4 rounds=%zu drafted=%zu acceptance=%.2f\n", st.rounds, st.drafted, st.acceptance());
    }
    CHECK((lookup_draft(std::vector<int>{1, 2, 3, 9, 1, 2, 3}, 3, 4) == std::vector<int>{9, 1, 2, 3}));
    CHECK(lookup_draft(std::vector<int>{1, 2, 3}, 2, 4).empty());
    // Sampling: the first token's distribution must equal the target's even though a draft is always proposed.
    std::vector<int> q = {256, 65, 66, 256, 65, 66};
    sampler_config sc{.temperature = 1.0f, .seed = 8};
    session r2(tm); sampler probe(sc); auto pw = probe.probs(r2.forward(q), q);
    std::vector<double> cnt(pw.size(), 0); const int N = 3000; sampler s(sc); session t(tm);
    for (int i = 0; i < N; ++i) { auto o = lookup_generate(t, q, 1, s, 3, 3); cnt[std::size_t(o[0])] += 1; }
    double tv = 0; for (std::size_t i = 0; i < pw.size(); ++i) tv += std::abs(cnt[i] / N - pw[i]);
    CHECK_MSG(tv / 2 < 0.08, "lookup tv %.3f", tv / 2);
}
TEST(presence_frequency_penalties) {
    std::vector<float> l = {1.f, 1.f, 1.f, 0.5f};
    std::vector<int> h = {0, 0, 0, 1};
    sampler sp({.presence_penalty = 0.5f});
    auto pp = sp.probs(l, h);   // greedy: argmax after penalty -> token 2
    CHECK(pp[2] == 1.0);
    sampler sf({.temperature = 1.f, .frequency_penalty = 0.25f});
    auto pf = sf.probs(l, h);
    CHECK(std::fabs(std::log(pf[2] / pf[0]) - 0.75) < 1e-6 && std::fabs(std::log(pf[2] / pf[1]) - 0.25) < 1e-6);
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

TEST(batch_scheduler_priority_and_cancel) {
    auto m = model::open(model_a());
    batch_scheduler b(m, {.max_active = 1, .prefix_cache = 0});
    std::vector<std::string> order;
    auto done = [&](std::string tag) { return [&order, tag](const std::vector<int>&, const std::string& w) { order.push_back(tag + ":" + w); }; };
    b.submit({.prompt = {256, 1}, .max_new = 3, .on_done = done("A")});
    b.submit({.prompt = {256, 2}, .max_new = 3, .on_done = done("B")});
    b.submit({.prompt = {256, 3}, .max_new = 3, .priority = 5, .on_done = done("C")});
    auto d = b.submit({.prompt = {256, 4}, .max_new = 3, .priority = 5, .on_done = done("D")});
    CHECK(b.cancel(d)); CHECK(!b.cancel(d)); CHECK(!b.cancel(999));
    b.run_until_idle();
    CHECK((order == std::vector<std::string>{"D:cancelled", "C:length", "A:length", "B:length"}));
    std::vector<int> partial; std::string why;
    auto e = b.submit({.prompt = {256, 5}, .max_new = 50, .on_done = [&](const std::vector<int>& o, const std::string& w) { partial = o; why = w; }});
    for (int i = 0; i < 4; ++i) b.step();
    CHECK(b.cancel(e)); b.run_until_idle();
    CHECK(why == "cancelled" && partial.size() == 4);
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

#include <forge/mind/json_schema.hpp>
TEST(sampler_typical_and_mirostat) {
    // typical: entropy of {0.5,0.25,0.125,0.125} is 1.75 bits; the 0.25 token (2 bits) is most typical.
    std::vector<float> l = {std::log(0.5f), std::log(0.25f), std::log(0.125f), std::log(0.125f)};
    sampler_config c; c.temperature = 1; c.typical_p = 0.2f; sampler s(c);
    auto p = s.probs(l); CHECK(std::fabs(p[1] - 1.0) < 1e-12);
    c.typical_p = 0.8f; sampler s2(c); p = s2.probs(l); CHECK(p[1] > 0 && p[0] > 0 && p[2] + p[3] > 0);
    // mirostat: observed mean surprise converges near tau on a Zipf distribution.
    std::vector<float> z(1000); for (std::size_t i = 0; i < z.size(); ++i) z[i] = -1.1f * std::log(float(i + 1));
    sampler_config m; m.temperature = 1; m.mirostat_tau = 5; m.mirostat_eta = 0.1f; m.seed = 7; sampler ms(m);
    double ls = 0; for (std::size_t i = 0; i < z.size(); ++i) ls += std::exp(double(z[i])); double surp = 0; int N = 4000;
    for (int i = 0; i < N; ++i) { int t = ms.sample(z); if (i >= 1000) surp += -std::log2(std::exp(double(z[std::size_t(t)])) / ls); }
    surp /= (N - 1000); CHECK_MSG(std::fabs(surp - 5) < 0.6, "mean surprise %g", surp);
}
TEST(json_schema_grammar) {
    auto sch = json::parse(R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},
        "tags":{"type":"array","items":{"enum":["a","b"]},"maxItems":2},"ok":{"type":"boolean"}},"required":["name","age"]})");
    grammar g(json_schema_to_gbnf(sch));
    auto run = [&](std::string s) { auto st = g.start(); for (unsigned char ch : s) { st = g.advance(st, ch); if (st.empty()) return false; } return grammar::can_end(st); };
    CHECK(run(R"({"name":"x\"y","age":-12})"));
    CHECK(run(R"({"name":"","age":0,"tags":["a","b"],"ok":true})"));
    CHECK(run(R"({"name":"q","age":3,"ok":false})"));
    CHECK(run(R"({"name":"q","age":3,"tags":[]})"));
    CHECK(!run(R"({"age":3,"name":"q"})"));            // order is fixed
    CHECK(!run(R"({"name":"q"})"));                    // missing required
    CHECK(!run(R"({"name":"q","age":3,"tags":["a","b","a"]})"));
    CHECK(!run(R"({"name":"q","age":1.5})"));
    CHECK(!run(R"({"name":"q","age":3,"tags":["c"]})"));
    auto sch2 = json::parse(R"({"$defs":{"pt":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":3}},
        "anyOf":[{"$ref":"#/$defs/pt"},{"type":"null"},{"const":{"k":1}}]})");
    grammar g2(json_schema_to_gbnf(sch2));
    auto run2 = [&](std::string s) { auto st = g2.start(); for (unsigned char ch : s) { st = g2.advance(st, ch); if (st.empty()) return false; } return grammar::can_end(st); };
    CHECK(run2("[1,2.5e3]")); CHECK(run2("[1,2,3]")); CHECK(!run2("[1]")); CHECK(!run2("[1,2,3,4]")); CHECK(run2("null")); CHECK(run2(R"({"k":1})"));
    CHECK(run2("[1,2]") && !run2("[01,2]"));
    CHECK_THROWS(json_schema_to_gbnf(json::parse(R"({"$ref":"#/$defs/none"})")));
}

TEST(lora_hot_swap) {
    auto m = model::open(model_a());
    session base(m); auto L0 = base.forward(std::vector<int>{1, 2, 3});
    std::size_t w0 = m->loaded_weights();
    auto vocab = m->vocab(); i64 d = 0; { pack::reader r(model_a()); d = r.load("lm_head").shape[1]; }
    std::mt19937 g(4); std::normal_distribution<float> N(0, 0.05f); const i64 r = 4;
    model::lora l; l.a.shape = {r, d}; l.b.shape = {vocab, r};
    for (i64 i = 0; i < r * d; ++i) l.a.data.push_back(N(g)); for (i64 i = 0; i < vocab * r; ++i) l.b.data.push_back(N(g));
    m->add_adapter("one", {{"lm_head", l}}, float(r)); m->add_adapter("two", {{"lm_head", l}}, float(2 * r));
    { pack::reader rr(model_a()); auto wq = rr.load("layer.0.wq"); model::lora z; z.a.shape = {r, wq.shape[1]}; z.a.data.assign(std::size_t(r * wq.shape[1]), 0.f);
      z.b.shape = {wq.shape[0], r}; z.b.data.assign(std::size_t(wq.shape[0] * r), 1.f); m->add_adapter("zero", {{"layer.0.wq", z}}, 8); }
    CHECK_THROWS(m->add_adapter("one", {{"lm_head", l}}, 1.f));
    model::lora bad = l; bad.b.shape = {vocab + 1, r}; CHECK_THROWS(m->add_adapter("bad", {{"lm_head", bad}}, 1.f));
    session s(m); std::vector<int> ids{1, 2, 3};
    s.set_adapter("one"); auto L1 = s.forward(ids);
    s.set_adapter("two"); auto L2 = s.forward(ids);
    s.set_adapter("zero"); auto Lz = s.forward(ids);
    s.set_adapter(""); auto Lb = s.forward(ids);
    CHECK_THROWS(s.set_adapter("nope"));
    double lin = 0, diff = 0, z = 0, b = 0;
    for (std::size_t i = 0; i < L0.size(); ++i) { lin = std::max(lin, double(std::fabs((L2[i] - L0[i]) - 2 * (L1[i] - L0[i]))));
        diff = std::max(diff, double(std::fabs(L1[i] - L0[i]))); z = std::max(z, double(std::fabs(Lz[i] - L0[i]))); b = std::max(b, double(std::fabs(Lb[i] - L0[i]))); }
    CHECK_MSG(diff > 1e-3, "adapter had no effect"); CHECK_MSG(lin < 1e-4, "delta not linear in alpha: %g", lin);
    CHECK(z == 0); CHECK(b == 0);
    CHECK(m->loaded_weights() == w0);                         // base weights were not reloaded or copied
    // Two sessions with different adapters on one model.
    session s1(m), s2(m); s1.set_adapter("one"); s2.set_adapter("two");
    auto a = s1.forward(ids), c = s2.forward(ids); CHECK(a == L1 && c == L2);
}
TEST(lora_adapter_file) {
    auto m = model::open(model_a());
    auto vocab = m->vocab(); i64 d = 0; { pack::reader r(model_a()); d = r.load("lm_head").shape[1]; }
    std::mt19937 g(9); std::normal_distribution<float> N(0, 0.05f); const i64 r = 2;
    model::lora l; l.a.shape = {r, d}; l.b.shape = {vocab, r};
    for (i64 i = 0; i < r * d; ++i) l.a.data.push_back(N(g)); for (i64 i = 0; i < vocab * r; ++i) l.b.data.push_back(N(g));
    auto path = (std::filesystem::temp_directory_path() / ("forge_lora_" + std::to_string(::getpid()) + ".fpk")).string();
    { pack::writer w; w.set_manifest(json::object{{"format", "fpk"}, {"lora_alpha", 3.0}});
      w.add_tensor("lm_head.lora_a", l.a.shape, l.a.data); w.add_tensor("lm_head.lora_b", l.b.shape, l.b.data); w.write(path); }
    m->add_adapter_file("file", path); m->add_adapter("mem", {{"lm_head", l}}, 3.f);
    std::filesystem::remove(path);
    std::vector<int> ids{4, 5, 6};
    session a(m), b(m); a.set_adapter("file"); b.set_adapter("mem");
    CHECK(a.forward(ids) == b.forward(ids));
}
