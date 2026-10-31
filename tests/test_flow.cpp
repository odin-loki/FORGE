#include <cstring>
#include <map>
#include "check.hpp"
#include <forge/flow/flow.hpp>
#include <random>

using namespace forge;
using namespace forge::flow;

TEST(verify_reports_location) {
    graph g;
    auto a = g.input("a", {3, 4}), b = g.input("b", {5, 4});
    g.output("y", g.matmul(a, b));
    bool threw = false;
    try { verify_graph(g); } catch (const forge_exception& e) { threw = true; CHECK(std::string(e.what()).find("node %2 (matmul)") != std::string::npos); }
    CHECK(threw);
}
TEST(cse_fold_simplify_fuse_dce) {
    graph g;
    auto x = g.input("x", {4, 8});
    auto c = g.apply(spec::op::add, {g.scalar(1.f), g.scalar(2.f)});      // folds to 3
    auto e1 = g.apply(spec::op::exp, {x}), e2 = g.apply(spec::op::exp, {x}); // CSE
    auto s = g.apply(spec::op::silu, {e1});
    auto y = g.mul(s, e2);                                                   // fuse -> silu_mul
    auto y1 = g.mul(y, g.scalar(1.f));                                       // simplify away
    g.apply(spec::op::tanh, {x});                                            // dead
    g.output("y", g.add(y1, c));
    auto o = optimise(g);
    int n_exp = 0, n_silu_mul = 0, n_tanh = 0, n_mul = 0;
    for (auto& n : o.nodes()) if (n.kind == node_kind::op) {
        n_exp += n.op == spec::op::exp; n_silu_mul += n.op == spec::op::silu_mul; n_tanh += n.op == spec::op::tanh; n_mul += n.op == spec::op::mul;
    }
    CHECK(n_exp == 1); CHECK(n_silu_mul == 1); CHECK(n_tanh == 0); CHECK(n_mul == 0);
    // numerics: optimised == unoptimised
    std::mt19937 r(1); auto X = verify::random_tensor({4, 8}, r, -2, 2);
    auto p1 = compile(g), p2 = compile(g, nullptr, {false, false, false, false});
    executor e1x(*p1), e2x(*p2);
    e1x.set_input("x", X.data); e2x.set_input("x", X.data); e1x.run(); e2x.run();
    auto c1 = verify::compare(e1x.output("y"), e2x.output("y"), {spec::tol_kind::rel, 1e-5});
    CHECK_MSG(c1.ok, "%s", c1.message.c_str());
}

// Pass equivalence on random graphs.
TEST(pass_equivalence_random_graphs) {
    std::mt19937 r(77);
    const spec::op un[] = {spec::op::exp, spec::op::tanh, spec::op::sigmoid, spec::op::silu, spec::op::neg, spec::op::square};
    const spec::op bin[] = {spec::op::add, spec::op::mul, spec::op::sub, spec::op::maximum};
    for (int trial = 0; trial < 40; ++trial) {
        graph g; std::vector<value_id> vals = {g.input("x", {3, 17}), g.input("y", {3, 17})};
        for (int i = 0; i < 12; ++i) {
            if (r() % 2) vals.push_back(g.apply(un[r() % 6], {vals[r() % vals.size()]}));
            else vals.push_back(g.apply(bin[r() % 4], {vals[r() % vals.size()], vals[r() % vals.size()]}));
            vals.back() = g.apply(spec::op::tanh, {vals.back()}); // keep values bounded
        }
        g.output("o", vals.back());
        g.output("p", vals[vals.size() / 2]);
        auto X = verify::random_tensor({3, 17}, r, -1, 1), Y = verify::random_tensor({3, 17}, r, -1, 1);
        auto a = compile(g), b = compile(g, nullptr, {false, false, false, false});
        CHECK(plan_is_safe(a->mem));
        executor ea(*a), eb(*b);
        for (auto* e : {&ea, &eb}) { e->set_input("x", X.data); e->set_input("y", Y.data); e->run(); }
        for (auto nm : {"o", "p"}) {
            auto c = verify::compare(ea.output(nm), eb.output(nm), {spec::tol_kind::abs, 1e-5});
            CHECK_MSG(c.ok, "trial %d %s: %s", trial, nm, c.message.c_str());
        }
    }
}

TEST(memory_plan_reuse_and_peak) {
    graph g;
    auto x = g.input("x", {1024});
    value_id v = x;
    for (int i = 0; i < 20; ++i) v = g.apply(i % 2 ? spec::op::exp : spec::op::tanh, {v});
    g.output("y", v);
    auto p = compile(g);
    CHECK(plan_is_safe(p->mem));
    std::printf("  arena %zu naive %zu in-place %d\n", p->mem.arena_bytes, p->mem.naive_bytes, p->mem.in_place);
    CHECK(p->mem.arena_bytes <= 2 * 4096);             // chain reuses buffers
    CHECK(p->mem.naive_bytes >= p->mem.arena_bytes && p->mem.in_place == 20);
    // predicted peak == measured: the executor's arena is exactly the predicted size
    CHECK(p->arena.size() * sizeof(float) >= p->mem.arena_bytes && p->arena.size() * sizeof(float) <= p->mem.arena_bytes + 4);
}

TEST(lowerings_pass_oracle) {
    auto& reg = registry::cpu();
    int n = 0;
    for (std::size_t o = 0; o < spec::op_count; ++o)
        for (auto& l : reg.lowerings(spec::op(o))) {
            std::string msg;
            bool ok = registry::oracle_check(l, &msg);
            CHECK_MSG(ok, "%s: %s", l.name.c_str(), msg.c_str());
            ++n;
        }
    std::printf("  %d lowerings verified\n", n);
}

// Oracle matrix: every spec op through the full compile pipeline vs its reference.
TEST(oracle_matrix_all_ops) {
    std::mt19937 rng(99);
    int ok = 0;
    for (std::size_t o = 0; o < spec::op_count; ++o) {
        auto c = verify::make_case(spec::op(o), rng);
        graph g; std::vector<value_id> ins;
        for (std::size_t i = 0; i < c.in.size(); ++i) ins.push_back(g.input("i" + std::to_string(i), c.in[i].shape));
        g.output("y", g.apply(spec::op(o), ins, c.a));
        auto p = compile(g, nullptr, {false, false, false, false});
        executor e(*p);
        for (std::size_t i = 0; i < c.in.size(); ++i) e.set_input("i" + std::to_string(i), c.in[i].data);
        e.run();
        auto ref = spec::reference(spec::op(o))(c.in, c.a);
        auto r = verify::compare(e.output("y"), ref, spec::info(spec::op(o)).tol);
        CHECK_MSG(r.ok, "%s: %s", std::string(spec::info(spec::op(o)).name).c_str(), r.message.c_str());
        ok += r.ok;
    }
    std::printf("  %d/%zu ops pass\n", ok, spec::op_count);
}

TEST(state_append_symbolic_length) {
    state_store st; st.capacity = 16;
    graph g;
    auto x = g.input("x", {1, 2, 4});
    auto kv = g.state_append("k0", x);
    g.output("y", g.apply(spec::op::attention, {x, kv, kv}, {{"causal", 1.0}}));
    auto p = compile(g, &st);
    executor e(*p, &st);
    std::vector<float> in(8, 0.5f);
    for (int t = 0; t < 5; ++t) { e.set_input("x", in); e.run(); }
    CHECK(st.slots["k0"].len == 5);
    auto y = e.output("y");
    for (float v : y.data) CHECK(std::fabs(v - 0.5f) < 1e-6f);
}

TEST(codecs_exact_and_layout) {
    // values exactly representable in a codec survive the round trip bit-for-bit
    auto rt = [](std::string_view name, std::vector<float> x) {
        auto* c = codecs::find(name);
        std::vector<u8> b(c->bytes(x.size())); std::vector<float> y(x.size());
        c->encode(x.data(), x.size(), b.data()); c->decode(b.data(), x.size(), y.data());
        for (std::size_t i = 0; i < x.size(); ++i) CHECK_MSG(y[i] == x[i], "%s[%zu] %g -> %g", std::string(name).c_str(), i, x[i], y[i]);
    };
    std::vector<float> t(64); for (int i = 0; i < 64; ++i) t[i] = float((i % 3) - 1) * 0.5f;
    rt("tq2", t);                                                    // ternary data at one magnitude is exact
    std::vector<float> m4(32); const float e2m1[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    for (int i = 0; i < 32; ++i) m4[i] = e2m1[i % 8] * (i & 8 ? -1.f : 1.f) * 0.25f;
    rt("mxfp4", m4);                                                  // amax 1.5 -> X = -2, so every value is an exact E2M1 * 2^-2
    std::vector<float> g(32); for (int i = 0; i < 32; ++i) g[i] = float(i % 16) * 0.125f - 1.f;
    rt("q4_g32", g);                                                  // 16 levels, step 0.125, min -1
    CHECK(codecs::find("mxfp8")->bytes(64) == 66 && codecs::find("nvfp4")->bytes(16) == 9 && codecs::find("q4_g32")->bytes(33) == 40);
    for (auto& c : codecs::all()) CHECK(dtype_from_name(c.name) == c.tag);
}

TEST(codecs_roundtrip) {
    std::mt19937 r(4);
    auto x = verify::random_tensor({1000}, r, -3, 3);
    for (auto& c : codecs::all()) {
        std::vector<u8> buf(c.bytes(x.size()));
        std::vector<float> y(x.size());
        c.encode(x.data.data(), x.size(), buf.data());
        c.decode(buf.data(), x.size(), y.data());
        double maxe = 0; for (std::size_t i = 0; i < y.size(); ++i) maxe = std::max(maxe, double(std::fabs(y[i] - x.data[i])));
        static const std::map<std::string_view, double> bounds = {
            {"f32", 0}, {"f16", 2e-3}, {"bf16", 2e-2}, {"fp8_e4m3", 0.13}, {"fp8_e5m2", 0.26}, {"q8_0", 0.013}, {"q4_k", 0.25},
            {"q4_g32", 0.21}, {"q8_g32", 0.013}, {"mxfp8", 0.126}, {"mxfp6", 0.126}, {"mxfp4", 0.51}, {"mxint8", 0.016},
            {"nvfp4", 0.51}, {"tq2", 3.0}, {"q6_k", 0.07}, {"q5_k", 0.13}};
        CHECK_MSG(bounds.count(c.name), "no bound for %s", std::string(c.name).c_str());
        double bound = bounds.count(c.name) ? bounds.at(c.name) : 0;
        std::printf("  %-9s %5zu bytes  max err %.5f\n", std::string(c.name).c_str(), buf.size(), maxe);
        CHECK_MSG(maxe <= bound, "%s", std::string(c.name).c_str());
    }
}


// Native quantised kernels (lane/qdot.hpp) must be bit-identical to the scalar codec decode + lane dot, on every
// lane target this machine can run, including a partial final tile (k not a multiple of 256 for q8_0).
TEST(qdot_matches_decode_exactly) {
    std::mt19937 r(11);
    for (const char* tgt : {"scalar", "avx2", "avx512"}) {
        const auto* ker = lane::by_name(tgt);
        if (!ker) { std::printf("  qdot: target %s not available\n", tgt); continue; }
        for (const char* cn : {"q8_0", "q4_k", "q6_k", "q5_k", "f16", "q8_g32"}) {
            const auto* c = codecs::find(cn);
            for (std::size_t k : {std::size_t(256), std::size_t(1024), std::size_t(768 + (c->block == 32 ? 96 : 0))}) {
                auto w = verify::random_tensor({i64(k)}, r, -2, 2); auto x = verify::random_tensor({i64(k)}, r, -1, 1);
                std::vector<u8> enc(c->bytes(k)); c->encode(w.data.data(), k, enc.data());
                std::vector<float> ref(k), got(k);
                c->decode(enc.data(), k, ref.data());
                int fmt = qweights::fmt_of(c);
                ker->qdecode(fmt, enc.data(), k, got.data(), c->decode);
                CHECK_MSG(std::memcmp(ref.data(), got.data(), k * 4) == 0, "%s/%s k=%zu: qdecode differs", tgt, cn, k);
                float a = ker->dot(x.data.data(), ref.data(), k);
                float b = ker->qdot(fmt, enc.data(), x.data.data(), k, c->decode, c->block, c->block_bytes);
                CHECK_MSG(std::memcmp(&a, &b, 4) == 0, "%s/%s k=%zu: qdot %.9g vs dot %.9g", tgt, cn, k, double(b), double(a));
            }
        }
        std::printf("  qdot: %s exact for q8_0 q4_k q6_k q5_k f16 q8_g32\n", tgt);
    }
}

#include <forge/flow/autodiff.hpp>
// Reverse-mode gradients vs central finite differences (double-precision forward via the same plan in f32,
// so the tolerance is loose but the check catches wrong formulas immediately).
TEST(autodiff_matches_finite_differences) {
    using namespace forge::flow; using spec::op;
    auto build = [](graph& g, std::vector<value_id>& ins) {
        auto x = g.input("x", {3, 4}), W = g.input("W", {5, 4}), V = g.input("V", {5, 2}), bvec = g.input("b", {5}), nw = g.input("nw", {4}), c = g.input("c", {3, 1});
        ins = {x, W, V, bvec, nw, c};
        auto h = g.apply(op::rms_norm, {x, nw}, {{"eps", 1e-5}});
        auto z = g.add(g.apply(op::matmul_t, {h, W}), bvec);                       // [3,5]
        auto t = g.apply(op::tanh, {z}), s = g.apply(op::silu, {z}), sg = g.apply(op::sigmoid, {t});
        auto sm = g.apply(op::softmax, {g.add(t, s)});
        auto m = g.matmul(g.mul(sm, sg), V);                                         // [3,2]
        auto q = g.apply(op::div, {g.apply(op::square, {m}), g.add(g.apply(op::exp, {c}), g.scalar(1.f))});
        auto r = g.apply(op::relu, {g.apply(op::sub, {q, g.scalar(0.01f)})});
        auto lse = g.apply(op::logsumexp, {g.apply(op::transpose, {g.add(r, g.apply(op::log, {g.add(g.apply(op::abs, {m}), g.scalar(1.f))}))}, {{"perm", std::vector<i64>{1, 0}}})}, {{"axis", 1.0}});
        auto loss = g.apply(op::reduce_mean, {g.apply(op::sqrt, {g.add(g.apply(op::reshape, {lse}, {{"shape", std::vector<i64>{2}}}), g.scalar(4.f))})}, {{"axis", 0.0}});
        return loss;
    };
    // std::normal_distribution differs between libstdc++/libc++/MSVC; Box-Muller on raw mt19937 output is portable,
    // so every platform checks the same point (away from the relu/abs kinks where finite differences are meaningless).
    std::mt19937 rng(11);
    auto N = [&] { double u1 = (double(rng()) + 1.0) / 4294967297.0, u2 = double(rng()) / 4294967296.0; return float(std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2)); };
    std::vector<std::vector<float>> data; for (i64 n : {12, 20, 10, 5, 4, 3}) { std::vector<float> v(static_cast<std::size_t>(n)); for (auto& e : v) e = N(); data.push_back(v); }
    const char* names[] = {"x", "W", "V", "b", "nw", "c"};
    graph g; std::vector<value_id> ins; auto loss = build(g, ins);
    auto gr = grad(g, loss, ins);
    g.output("loss", loss); for (std::size_t i = 0; i < ins.size(); ++i) g.output(std::string("d") + names[i], gr[ins[i]]);
    auto p = compile(g); executor ex(*p);
    auto run = [&](const std::vector<std::vector<float>>& d) { for (std::size_t i = 0; i < d.size(); ++i) ex.set_input(names[i], d[i]); ex.run(); return double(ex.output_view("loss").data[0]); };
    run(data);
    std::vector<std::vector<float>> an; for (auto nm : names) { auto v = ex.output_view(std::string("d") + nm); an.emplace_back(v.data, v.data + v.size()); }
    double worst = 0;
    for (std::size_t i = 0; i < data.size(); ++i) for (std::size_t j = 0; j < data[i].size(); ++j) {
        auto d2 = data; float h = 2e-3f; d2[i][j] += h; double fp = run(d2); d2[i][j] -= 2 * h; double fm = run(d2);
        double fd = (fp - fm) / (2 * h), e = std::fabs(fd - an[i][j]) / std::max(1e-2, std::fabs(fd));
        worst = std::max(worst, e); CHECK_MSG(e < 3e-2, "d%s[%zu]: analytic %g fd %g", names[i], j, double(an[i][j]), fd);
    }
    std::printf("  autodiff worst relative error vs finite differences: %.2e\n", worst);
    graph g2; auto a = g2.input("a", {4}); auto am = g2.apply(op::argmax, {a}, {{"axis", 0.0}});
    CHECK_THROWS(grad(g2, am, {a}));
    graph g3; auto u = g3.input("u", {2}), v = g3.input("v", {2}); auto l3 = g3.apply(op::reduce_sum, {g3.apply(op::exp, {u})}, {{"axis", 0.0}});
    auto z3 = grad(g3, l3, {u, v}); CHECK(g3.at(z3[v]).kind == node_kind::constant);
}
