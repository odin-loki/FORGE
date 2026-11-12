// Edge cases of the spatial / encoder op families (ops/defs/spatial.fspec, sequence.fspec): each case runs the op
// through compile() + executor (so the CPU lowering when there is one) and through the scalar reference, and checks
// both against values worked out independently (hand-computed results, adjoint identities, equivalent op chains).
#include <cmath>
#include <cstring>
#include <map>
#include "check.hpp"
#include <forge/graph/graph.hpp>
#include <random>

using namespace forge;
using namespace forge::graph;

namespace {
tensor rnd(shape_t s, std::mt19937& rng, float lo = -1.f, float hi = 1.f) { return verify::random_tensor(std::move(s), rng, lo, hi); }
tensor filled(shape_t s, std::initializer_list<float> v) { tensor t(std::move(s)); std::copy(v.begin(), v.end(), t.data.begin()); return t; }
// the op through the compile pipeline (fold / fuse off so the op itself runs)
tensor run_op(ops::op o, const std::vector<tensor>& in, const ops::attrs& a) {
    tensor_graph g; std::vector<value_id> ins;
    for (std::size_t i = 0; i < in.size(); ++i) ins.push_back(g.input("i" + std::to_string(i), in[i].shape));
    g.output("y", g.apply(o, ins, a));
    auto p = compile(g, nullptr, {false, false, false, false});
    executor e(*p);
    for (std::size_t i = 0; i < in.size(); ++i) e.set_input("i" + std::to_string(i), in[i].data);
    e.run();
    return e.output("y");
}
// reference and compiled result agree within the spec tolerance; returns the reference
tensor both(ops::op o, const std::vector<tensor>& in, const ops::attrs& a, const char* what = "") {
    auto ref = ops::reference(o)(in, a);
    auto got = run_op(o, in, a);
    auto r = verify::compare(got, ref, ops::info(o).tol);
    CHECK_MSG(r.ok, "%s %s: lowering vs reference: %s", std::string(ops::info(o).name).c_str(), what, r.message.c_str());
    return ref;
}
double max_diff(const tensor& a, const tensor& b) {
    if (a.shape != b.shape) return 1e30;
    double m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(double(a.data[i]) - b.data[i]));
    return m;
}
double dot(const tensor& a, const tensor& b) { double s = 0; for (std::size_t i = 0; i < a.size(); ++i) s += double(a.data[i]) * b.data[i]; return s; }
using IL = std::vector<i64>;
}

TEST(conv2d_known_values) {
    // 3x3 all-ones kernel over a 3x4 image of ones, pad 1: each output counts its in-image neighbours
    tensor x({1, 3, 4}, 1.f), w({1, 1, 3, 3}, 1.f), b({1}, 0.5f);
    auto y = both(ops::op::conv2d, {x, w, b}, {{"pad", 1.0}});
    float want[12] = {4, 6, 6, 4, 6, 9, 9, 6, 4, 6, 6, 4};
    CHECK(y.shape == (shape_t{1, 3, 4}));
    for (int i = 0; i < 12; ++i) CHECK(y.data[std::size_t(i)] == want[i] + 0.5f);
    // 1x1 kernel = a per-pixel linear map
    std::mt19937 rng(1);
    auto x2 = rnd({3, 4, 5}, rng), w2 = rnd({2, 3, 1, 1}, rng), b2 = rnd({2}, rng);
    auto y2 = both(ops::op::conv2d, {x2, w2, b2}, {});
    for (i64 o = 0; o < 2; ++o) for (i64 p = 0; p < 20; ++p) {
        double s = b2.data[std::size_t(o)];
        for (i64 c = 0; c < 3; ++c) s += double(w2.data[std::size_t(o * 3 + c)]) * x2.data[std::size_t(c * 20 + p)];
        CHECK(std::fabs(y2.data[std::size_t(o * 20 + p)] - s) < 1e-5);
    }
    // stride == kernel (patch embedding): each output is one patch dotted with the kernel
    auto x3 = rnd({2, 4, 6}, rng), w3 = rnd({3, 2, 2, 3}, rng); tensor b3({1}, 0.f);
    auto y3 = both(ops::op::conv2d, {x3, w3, b3}, {{"stride", IL{2, 3}}});
    CHECK(y3.shape == (shape_t{3, 2, 2}));
    double s = 0;
    for (i64 c = 0; c < 2; ++c) for (i64 i = 0; i < 2; ++i) for (i64 j = 0; j < 3; ++j)
        s += double(w3.data[std::size_t(((1 * 2 + c) * 2 + i) * 3 + j)]) * x3.data[std::size_t((c * 4 + 2 + i) * 6 + 3 + j)];
    CHECK(std::fabs(y3.data[std::size_t((1 * 2 + 1) * 2 + 1)] - s) < 1e-5);
}

TEST(conv2d_stride_pad_dilation_groups_sweep) {
    std::mt19937 rng(2); int n = 0;
    for (i64 groups : {1, 2, 4}) for (i64 st : {1, 2, 3}) for (i64 pd : {0, 1, 2}) for (i64 dl : {1, 2}) for (i64 k : {1, 3}) {
        i64 C = 4, O = groups == 4 ? 4 : 6;   // groups 4: depthwise (one filter per channel); also 8 = channel multiplier 2
        for (i64 mult : {1, 2}) {
            if (groups != 4 && mult == 2) continue;
            auto x = rnd({C, 7, 9}, rng), w = rnd({O * mult, C / groups, k, k + 1}, rng), b = rnd({O * mult}, rng);
            ops::attrs a{{"stride", IL{st, 1}}, {"pad", IL{pd, pd + 1}}, {"dilation", IL{dl, 1}}, {"groups", double(groups)}};
            auto y = both(ops::op::conv2d, {x, w, b}, a, "sweep");
            // independent check of one output element
            i64 oh = y.shape[1], ow = y.shape[2], oc = O * mult - 1, yy = oh / 2, xx = ow - 1, cg = C / groups, og = O * mult / groups;
            double s = b.data[std::size_t(oc)];
            for (i64 c = 0; c < cg; ++c) for (i64 i = 0; i < k; ++i) for (i64 j = 0; j < k + 1; ++j) {
                i64 iy = yy * st - pd + i * dl, ix = xx - (pd + 1) + j;
                if (iy < 0 || iy >= 7 || ix < 0 || ix >= 9) continue;
                s += double(x.data[std::size_t((((oc / og) * cg + c) * 7 + iy) * 9 + ix)]) * w.data[std::size_t(((oc * cg + c) * k + i) * (k + 1) + j)];
            }
            CHECK_MSG(std::fabs(y.data[std::size_t((oc * oh + yy) * ow + xx)] - s) < 1e-4, "conv2d g%lld s%lld p%lld d%lld k%lld", (long long)groups, (long long)st, (long long)pd, (long long)dl, (long long)k);
            ++n;
        }
    }
    std::printf("  %d conv2d configurations\n", n);
    // a scalar bias broadcasts; shape errors are reported
    auto x = rnd({4, 5, 5}, rng), w = rnd({2, 4, 3, 3}, rng);
    auto y0 = both(ops::op::conv2d, {x, w, tensor({1}, 0.25f)}, {});
    auto y1 = both(ops::op::conv2d, {x, w, tensor({2}, 0.25f)}, {});
    CHECK(max_diff(y0, y1) == 0);
    CHECK_THROWS(ops::reference(ops::op::conv2d)({x, w, tensor({3}, 0.f)}, {}));
    CHECK_THROWS(ops::reference(ops::op::conv2d)({x, w, tensor({2}, 0.f)}, {{"groups", 3.0}}));
    CHECK_THROWS(ops::reference(ops::op::conv2d)({x, rnd({2, 4, 7, 7}, rng), tensor({2}, 0.f)}, {}));   // kernel larger than the image
    CHECK_THROWS(ops::reference(ops::op::conv2d)({x, w, tensor({2}, 0.f)}, {{"stride", 0.0}}));
}

TEST(conv1d_matches_conv2d_and_sweep) {
    std::mt19937 rng(3);
    for (i64 groups : {1, 2, 4}) for (i64 st : {1, 2}) for (i64 pd : {0, 1, 3}) for (i64 dl : {1, 2, 3}) {
        auto x = rnd({4, 17}, rng), w = rnd({8, 4 / groups, 3}, rng), b = rnd({8}, rng);
        ops::attrs a{{"stride", double(st)}, {"pad", double(pd)}, {"dilation", double(dl)}, {"groups", double(groups)}};
        auto y = both(ops::op::conv1d, {x, w, b}, a, "sweep");
        tensor x2 = x, w2 = w; x2.shape = {4, 1, 17}; w2.shape = {8, 4 / groups, 1, 3};
        auto y2 = ops::reference(ops::op::conv2d)({x2, w2, b}, {{"stride", IL{1, st}}, {"pad", IL{0, pd}}, {"dilation", IL{1, dl}}, {"groups", double(groups)}});
        y2.shape = y.shape;
        CHECK(max_diff(y, y2) == 0);
    }
    // Whisper's two stem convolutions keep / halve the length
    auto x = rnd({5, 20}, rng), w = rnd({6, 5, 3}, rng), b = rnd({6}, rng);
    CHECK(both(ops::op::conv1d, {x, w, b}, {{"pad", 1.0}}).shape == (shape_t{6, 20}));
    CHECK(both(ops::op::conv1d, {x, w, b}, {{"stride", 2.0}, {"pad", 1.0}}).shape == (shape_t{6, 10}));
}

TEST(conv_transpose2d_is_the_adjoint_of_conv2d) {
    std::mt19937 rng(4); int n = 0;
    for (i64 groups : {1, 2}) for (i64 st : {1, 2, 3}) for (i64 pd : {0, 1}) for (i64 dl : {1, 2}) {
        const i64 C = 4, O = 6, k = 3, H = 8, W = 7;
        auto x = rnd({C, H, W}, rng), w = rnd({O, C / groups, k, k}, rng); tensor zero({1}, 0.f);
        ops::attrs a{{"stride", double(st)}, {"pad", double(pd)}, {"dilation", double(dl)}, {"groups", double(groups)}};
        auto y = ops::reference(ops::op::conv2d)({x, w, zero}, a);
        auto u = rnd(y.shape, rng);
        // output_pad restores the rows / columns the strided convolution dropped
        i64 oph = (H + 2 * pd - dl * (k - 1) - 1) % st, opw = (W + 2 * pd - dl * (k - 1) - 1) % st;
        ops::attrs at = a; at.set("output_pad", IL{oph, opw});
        auto xt = both(ops::op::conv_transpose2d, {u, w, zero}, at, "adjoint");
        CHECK(xt.shape == x.shape);
        double l = dot(y, u), r = dot(x, xt);
        CHECK_MSG(std::fabs(l - r) < 1e-3 * std::max(1.0, std::fabs(l)), "<conv x, u> %g != <x, convT u> %g (g%lld s%lld p%lld d%lld)", l, r, (long long)groups, (long long)st, (long long)pd, (long long)dl);
        ++n;
    }
    std::printf("  %d adjoint pairs\n", n);
    // stride-2 2x2 transposed conv of a single 1 stamps the kernel; the bias is added everywhere
    tensor x({1, 2, 2}, 0.f); x.data[3] = 1.f;
    auto w = filled({1, 1, 2, 2}, {1, 2, 3, 4});
    auto y = both(ops::op::conv_transpose2d, {x, w, tensor({1}, 10.f)}, {{"stride", 2.0}});
    CHECK(y.shape == (shape_t{1, 4, 4}));
    CHECK(y.data[10] == 11 && y.data[11] == 12 && y.data[14] == 13 && y.data[15] == 14 && y.data[0] == 10);
    CHECK_THROWS(ops::reference(ops::op::conv_transpose2d)({x, w, tensor({1}, 0.f)}, {{"stride", 2.0}, {"output_pad", 2.0}}));
}

TEST(pooling_known_values) {
    auto x = filled({1, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto mp = both(ops::op::max_pool2d, {x}, {{"kernel", 2.0}});   // stride defaults to the kernel; the odd row is dropped
    CHECK(mp.shape == (shape_t{1, 1, 2}) && mp.data[0] == 6 && mp.data[1] == 8);
    auto mp2 = both(ops::op::max_pool2d, {x}, {{"kernel", 3.0}, {"stride", 1.0}, {"pad", 1.0}});
    CHECK(mp2.shape == x.shape && mp2.data[0] == 6 && mp2.data[11] == 12 && mp2.data[4] == 10);
    tensor neg({1, 2, 2}, -5.f);
    CHECK(both(ops::op::max_pool2d, {neg}, {{"kernel", 3.0}, {"stride", 1.0}, {"pad", 1.0}}).data[0] == -5);   // padding never wins
    auto ap = both(ops::op::avg_pool2d, {x}, {{"kernel", 3.0}, {"stride", 1.0}, {"pad", 1.0}});
    CHECK(std::fabs(ap.data[0] - (1 + 2 + 5 + 6) / 9.f) < 1e-6);
    auto ap0 = both(ops::op::avg_pool2d, {x}, {{"kernel", 3.0}, {"stride", 1.0}, {"pad", 1.0}, {"count_pad", 0.0}});
    CHECK(std::fabs(ap0.data[0] - (1 + 2 + 5 + 6) / 4.f) < 1e-6 && std::fabs(ap0.data[5] - 6.f) < 1e-6);
    CHECK_THROWS(ops::reference(ops::op::max_pool2d)({x}, {{"kernel", 2.0}, {"pad", 2.0}}));
    std::mt19937 rng(5);
    auto y = rnd({3, 6, 8}, rng);
    auto g = both(ops::op::global_avg_pool, {y}, {});
    auto a1 = both(ops::op::adaptive_avg_pool2d, {y}, {{"out", 1.0}});
    CHECK(g.shape == (shape_t{3}) && a1.shape == (shape_t{3, 1, 1}));
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::fabs(g.data[c] - a1.data[c]) < 1e-6);
    auto a2 = both(ops::op::adaptive_avg_pool2d, {y}, {{"out", IL{3, 2}}});   // divisible sizes = plain average pooling
    auto p2 = both(ops::op::avg_pool2d, {y}, {{"kernel", IL{2, 4}}});
    CHECK(max_diff(a2, p2) < 1e-6);
    auto a3 = both(ops::op::adaptive_avg_pool2d, {filled({1, 1, 5}, {1, 2, 3, 4, 5})}, {{"out", IL{1, 3}}});   // overlapping cells [0,2) [1,4) [3,5)
    CHECK(std::fabs(a3.data[0] - 1.5f) < 1e-6 && std::fabs(a3.data[1] - 3.f) < 1e-6 && std::fabs(a3.data[2] - 4.5f) < 1e-6);
}

TEST(upsample_and_pad) {
    auto x = filled({1, 2, 2}, {1, 2, 3, 4});
    auto n = both(ops::op::upsample_nearest, {x}, {{"scale", IL{1, 2}}});
    CHECK(n.shape == (shape_t{1, 2, 4}) && n.data[0] == 1 && n.data[1] == 1 && n.data[2] == 2 && n.data[7] == 4);
    // half-pixel centres: columns sit at 0, 0.25, 0.75, 1 of the source
    auto b = both(ops::op::upsample_bilinear, {x}, {{"scale", IL{1, 2}}});
    CHECK(std::fabs(b.data[0] - 1.f) < 1e-6 && std::fabs(b.data[1] - 1.25f) < 1e-6 && std::fabs(b.data[2] - 1.75f) < 1e-6 && std::fabs(b.data[3] - 2.f) < 1e-6);
    // align_corners: a linear ramp stays a linear ramp between the same end points
    auto ramp = filled({1, 1, 4}, {0, 1, 2, 3});
    auto ba = both(ops::op::upsample_bilinear, {ramp}, {{"scale", IL{1, 3}}, {"align_corners", 1.0}});
    for (int i = 0; i < 12; ++i) CHECK(std::fabs(ba.data[std::size_t(i)] - float(i) * 3.f / 11.f) < 1e-6);
    std::mt19937 rng(6); auto y = rnd({2, 3, 4}, rng);
    CHECK(max_diff(both(ops::op::upsample_bilinear, {y}, {{"scale", 1.0}}), y) < 1e-7);
    auto v = filled({5}, {1, 2, 3, 4, 5});
    auto pc = both(ops::op::pad, {v}, {{"pads", IL{2, 1}}, {"value", 9.0}});
    auto pr = both(ops::op::pad, {v}, {{"pads", IL{2, 3}}, {"mode", 1.0}});
    auto pe = both(ops::op::pad, {v}, {{"pads", IL{2, 1}}, {"mode", 2.0}});
    CHECK(pc.data == (std::vector<float>{9, 9, 1, 2, 3, 4, 5, 9}));
    CHECK(pr.data == (std::vector<float>{3, 2, 1, 2, 3, 4, 5, 4, 3, 2}));
    CHECK(pe.data == (std::vector<float>{1, 1, 1, 2, 3, 4, 5, 5}));
    auto m = filled({2, 2}, {1, 2, 3, 4});   // pads run from the last axis backwards
    auto pm = both(ops::op::pad, {m}, {{"pads", IL{1, 0, 0, 1}}, {"mode", 1.0}});
    CHECK(pm.shape == (shape_t{3, 3}) && pm.data == (std::vector<float>{2, 1, 2, 4, 3, 4, 2, 1, 2}));
    CHECK_THROWS(ops::reference(ops::op::pad)({v}, {{"pads", IL{5, 0}}, {"mode", 1.0}}));
    CHECK_THROWS(ops::reference(ops::op::pad)({v}, {{"pads", IL{1}}}));
}

TEST(norms) {
    std::mt19937 rng(7);
    auto x = rnd({6, 4, 5}, rng, -3.f, 3.f), w = rnd({6}, rng), b = rnd({6}, rng);
    // groups == channels: every channel normalised on its own (instance norm)
    auto gn = both(ops::op::group_norm, {x, w, b}, {{"groups", 6.0}, {"eps", 1e-5}});
    for (i64 c = 0; c < 6; ++c) {
        double mu = 0, var = 0; for (i64 i = 0; i < 20; ++i) mu += x.data[std::size_t(c * 20 + i)];
        mu /= 20; for (i64 i = 0; i < 20; ++i) var += std::pow(x.data[std::size_t(c * 20 + i)] - mu, 2);
        var /= 20;
        for (i64 i = 0; i < 20; ++i) CHECK(std::fabs(gn.data[std::size_t(c * 20 + i)] - ((x.data[std::size_t(c * 20 + i)] - mu) / std::sqrt(var + 1e-5) * w.data[std::size_t(c)] + b.data[std::size_t(c)])) < 1e-5);
    }
    // one group: layer norm over the whole sample, with per-channel affine
    auto g1 = both(ops::op::group_norm, {x, tensor({6}, 1.f), tensor({6}, 0.f)}, {{"groups", 1.0}});
    tensor flat = x; flat.shape = {1, 120};
    auto ln = both(ops::op::layer_norm, {flat, tensor({120}, 1.f), tensor({120}, 0.f)}, {{"eps", 1e-5}});
    ln.shape = g1.shape; CHECK(max_diff(g1, ln) < 1e-5);
    CHECK_THROWS(ops::reference(ops::op::group_norm)({x, w, b}, {{"groups", 4.0}}));
    // batch norm with mean 0 / var 1 / w 1 / b 0 is the identity (up to eps); otherwise the affine form
    auto id = both(ops::op::batch_norm, {x, tensor({6}, 1.f), tensor({6}, 0.f), tensor({6}, 0.f), tensor({6}, 1.f)}, {{"eps", 0.0}});
    CHECK(max_diff(id, x) < 1e-6);
    auto mu = rnd({6}, rng), var = rnd({6}, rng, 0.5f, 2.f);
    auto bn = both(ops::op::batch_norm, {x, w, b, mu, var}, {{"eps", 1e-3}});
    CHECK(std::fabs(bn.data[25] - ((x.data[25] - mu.data[1]) / std::sqrt(var.data[1] + 1e-3) * w.data[1] + b.data[1])) < 1e-5);
    auto l2 = both(ops::op::l2_normalize, {filled({2, 2}, {3, 4, 0, 0})}, {});
    CHECK(std::fabs(l2.data[0] - 0.6f) < 1e-6 && std::fabs(l2.data[1] - 0.8f) < 1e-6 && l2.data[2] == 0 && l2.data[3] == 0);
}

TEST(attention_bias_masks_softcap_window) {
    std::mt19937 rng(8);
    const i64 s = 5, t = 5, hq = 4, hk = 2, d = 8, dv = 6;
    auto q = rnd({s, hq, d}, rng, -2.f, 2.f), k = rnd({t, hk, d}, rng, -2.f, 2.f), v = rnd({t, hk, dv}, rng);
    tensor zero({1}, 0.f);
    auto att = [&](double causal, double window) { return ops::reference(ops::op::attention)({q, k, v}, {{"causal", causal}, {"window", window}}); };
    // zero bias: the plain attention op, bidirectional and causal, with and without a sliding window
    CHECK(max_diff(both(ops::op::attention_bias, {q, k, v, zero}, {}), att(0, 0)) < 1e-6);
    CHECK(max_diff(both(ops::op::attention_bias, {q, k, v, zero}, {{"causal", 1.0}}), att(1, 0)) < 1e-6);
    CHECK(max_diff(both(ops::op::attention_bias, {q, k, v, zero}, {{"causal", 1.0}, {"window", 2.0}}), att(1, 2)) < 1e-6);
    // padding mask: -inf on the last two keys == attention over the first three keys only (queries at the same positions)
    tensor mask({1, 1, t}, 0.f); mask.data[3] = mask.data[4] = -INFINITY;
    auto masked = both(ops::op::attention_bias, {q, k, v, mask}, {}, "padding mask");
    tensor k3({3, hk, d}), v3({3, hk, dv});
    std::copy_n(k.data.begin(), 3 * hk * d, k3.data.begin()); std::copy_n(v.data.begin(), 3 * hk * dv, v3.data.begin());
    tensor q3({3, hq, d}); std::copy_n(q.data.begin(), 3 * hq * d, q3.data.begin());
    auto small = ops::reference(ops::op::attention)({q3, k3, v3}, {{"causal", 0.0}});
    for (std::size_t i = 0; i < small.size(); ++i) CHECK(std::fabs(masked.data[i] - small.data[i]) < 1e-5);
    // a bidirectional window masks both sides: window 1 leaves only the key at the query's own position
    auto self = both(ops::op::attention_bias, {q, k, v, zero}, {{"window", 1.0}}, "window 1");
    for (i64 i = 0; i < s; ++i) for (i64 h = 0; h < hq; ++h) for (i64 e = 0; e < dv; ++e)
        CHECK(std::fabs(self.data[std::size_t((i * hq + h) * dv + e)] - v.data[std::size_t((i * hk + h / 2) * dv + e)]) < 1e-6);
    // every key masked: zeros, not NaN
    tensor all({1}, -INFINITY);
    for (float x : both(ops::op::attention_bias, {q, k, v, all}, {}, "all masked").data) CHECK(x == 0.f);
    // per-head [hq, s, t] bias against an explicit softmax
    auto bias = rnd({hq, s, t}, rng, -3.f, 3.f);
    auto y = both(ops::op::attention_bias, {q, k, v, bias}, {{"softcap", 2.5}, {"scale", 0.3}}, "full bias + softcap");
    for (i64 i = 0; i < s; ++i) for (i64 h = 0; h < hq; ++h) {
        std::vector<double> p(static_cast<std::size_t>(t)); double sum = 0;
        for (i64 j = 0; j < t; ++j) {
            double dt = 0; for (i64 e = 0; e < d; ++e) dt += double(q.data[std::size_t((i * hq + h) * d + e)]) * k.data[std::size_t((j * hk + h / 2) * d + e)];
            p[std::size_t(j)] = std::exp(2.5 * std::tanh(dt * 0.3 / 2.5) + bias.data[std::size_t((h * s + i) * t + j)]); sum += p[std::size_t(j)];
        }
        for (i64 e = 0; e < dv; ++e) {
            double acc = 0; for (i64 j = 0; j < t; ++j) acc += p[std::size_t(j)] / sum * v.data[std::size_t((j * hk + h / 2) * dv + e)];
            CHECK(std::fabs(y.data[std::size_t((i * hq + h) * dv + e)] - acc) < 1e-5);
        }
    }
    // a huge cap changes nothing
    CHECK(max_diff(both(ops::op::attention_bias, {q, k, v, zero}, {{"softcap", 1e6}}), att(0, 0)) < 1e-4);
    CHECK_THROWS(ops::reference(ops::op::attention_bias)({q, k, v, tensor({3, s, t}, 0.f)}, {}));
}

// attention_bias over a growing KV state (soft-capped causal decoder): token-by-token == one prefill pass
TEST(attention_bias_reads_a_state_slot) {
    std::mt19937 rng(9);
    const i64 S = 6, H = 2, D = 4;
    auto x = rnd({S, H, D}, rng, -2.f, 2.f);
    auto build = [&](i64 n) {
        tensor_graph g;
        auto in = g.input("x", {n, H, D});
        auto kv = g.state_append("kv", in);
        g.output("y", g.apply(ops::op::attention_bias, {in, kv, kv, g.scalar(0.f)}, {{"causal", 1.0}, {"softcap", 1.5}, {"window", 3.0}}));
        return g;
    };
    state_store full; full.capacity = 16;
    auto pf = compile(build(S), &full); executor ef(*pf, &full);
    ef.set_input("x", x.data); ef.run();
    auto want = ef.output("y");
    state_store st; st.capacity = 16;
    auto p = compile(build(1), &st); executor e(*p, &st);
    for (i64 i = 0; i < S; ++i) {
        e.set_input("x", std::vector<float>(x.data.begin() + i * H * D, x.data.begin() + (i + 1) * H * D)); e.run();
        auto y = e.output("y");
        for (std::size_t j = 0; j < y.size(); ++j) CHECK(std::fabs(y.data[j] - want.data[std::size_t(i * H * D) + j]) < 1e-5);
    }
    state_store q8; q8.capacity = 16; q8.codec = kv_codec::int8;   // quantised KV is refused, not misread
    CHECK_THROWS(compile(build(1), &q8));
}

TEST(positions_and_selection) {
    auto sl = ops::ref::alibi_slopes(8);
    for (int i = 0; i < 8; ++i) CHECK(std::fabs(sl[std::size_t(i)] - std::pow(0.5, i + 1)) < 1e-12);
    auto s6 = ops::ref::alibi_slopes(6);   // 4 slopes of the power of two below, then every other slope of 8
    CHECK(s6.size() == 6 && std::fabs(s6[0] - 0.25) < 1e-12 && std::fabs(s6[3] - 1.0 / 256) < 1e-12 && std::fabs(s6[4] - 0.5) < 1e-12 && std::fabs(s6[5] - 0.125) < 1e-12);
    auto ab = both(ops::op::alibi_bias, {filled({2}, {3, 4}), filled({5}, {0, 1, 2, 3, 4})}, {{"heads", 2.0}});
    CHECK(ab.shape == (shape_t{2, 2, 5}) && std::fabs(ab.data[0] + 3 * 0.0625f) < 1e-7 && ab.data[3] == 0 && std::fabs(ab.data[5 + 5 + 0] + 3 * 0.00390625f) < 1e-8);
    // Whisper's sinusoids(length, channels): shift 1
    auto se = both(ops::op::sinusoid_embed, {filled({3}, {0, 1, 2})}, {{"dim", 6.0}, {"shift", 1.0}});
    for (int p = 0; p < 3; ++p) for (int i = 0; i < 3; ++i) {
        double f = std::exp(-std::log(10000.0) / 2 * i);
        CHECK(std::fabs(se.data[std::size_t(p * 6 + i)] - std::sin(p * f)) < 1e-6 && std::fabs(se.data[std::size_t(p * 6 + 3 + i)] - std::cos(p * f)) < 1e-6);
    }
    // diffusion timestep embedding (flip_sin_to_cos, shift 0), odd dim padded with 0
    auto te = both(ops::op::sinusoid_embed, {filled({1}, {37})}, {{"dim", 5.0}, {"flip", 1.0}});
    CHECK(std::fabs(te.data[0] - std::cos(37.0)) < 1e-6 && std::fabs(te.data[2] - std::sin(37.0)) < 1e-6 && std::fabs(te.data[3] - std::sin(0.37)) < 1e-6 && te.data[4] == 0);
    auto x = filled({2, 5}, {1, 5, 3, 5, 2, -1, -7, -1, 0, -3});
    auto tk = both(ops::op::topk, {x}, {{"k", 3.0}});
    auto ti = both(ops::op::topk_indices, {x}, {{"k", 3.0}});
    CHECK(tk.data == (std::vector<float>{5, 5, 3, 0, -1, -1}) && ti.data == (std::vector<float>{1, 3, 2, 3, 0, 2}));   // ties: lower index first
    CHECK_THROWS(ops::reference(ops::op::topk)({x}, {{"k", 6.0}}));
    auto is = both(ops::op::index_select, {x, filled({3}, {4, 0, -2})}, {{"axis", 1.0}});
    CHECK(is.data == (std::vector<float>{2, 1, 5, -3, -1, 0}));
    CHECK_THROWS(ops::reference(ops::op::index_select)({x, filled({1}, {5})}, {{"axis", 1.0}}));
    auto mm = both(ops::op::masked_mean, {filled({3, 2}, {1, 2, 10, 20, 3, 6}), filled({3}, {1, 0, 1})}, {});
    CHECK(mm.data == (std::vector<float>{2, 4}));
    CHECK(both(ops::op::masked_mean, {filled({2, 1}, {1, 2}), filled({2}, {0, 0})}, {}).data[0] == 0);
    auto fl = both(ops::op::floor, {filled({4}, {-1.5f, -0.f, 2.f, 2.99f})}, {});
    CHECK(fl.data[0] == -2 && fl.data[2] == 2 && fl.data[3] == 2);
    auto hs = both(ops::op::hard_swish, {filled({4}, {-4, -3, 1, 5})}, {});
    CHECK(hs.data[0] == 0 && hs.data[1] == 0 && std::fabs(hs.data[2] - 4.f / 6) < 1e-6 && hs.data[3] == 5);
    auto qg = both(ops::op::quick_gelu, {filled({2}, {0, 2})}, {});
    CHECK(qg.data[0] == 0 && std::fabs(qg.data[1] - 2 / (1 + std::exp(-3.404))) < 1e-6);
    auto mi = both(ops::op::mish, {filled({2}, {0, 30})}, {});
    CHECK(mi.data[0] == 0 && std::fabs(mi.data[1] - 30) < 1e-5);
}

// The new ops inside a graph with folding / CSE / memory planning on: a small conv stem + norm + pooled output.
TEST(spatial_ops_compose_through_the_pipeline) {
    std::mt19937 rng(10);
    auto img = rnd({3, 8, 8}, rng), w1 = rnd({4, 3, 3, 3}, rng), b1 = rnd({4}, rng), gw = rnd({4}, rng), gb = rnd({4}, rng);
    tensor_graph g;
    auto x = g.input("img", img.shape);
    auto c = g.apply(ops::op::conv2d, {x, g.constant(w1), g.constant(b1)}, {{"stride", 2.0}, {"pad", 1.0}});
    auto n = g.apply(ops::op::group_norm, {c, g.constant(gw), g.constant(gb)}, {{"groups", 2.0}});
    auto a = g.apply(ops::op::hard_swish, {n});
    auto u = g.apply(ops::op::upsample_bilinear, {a}, {{"scale", 2.0}});
    auto p = g.apply(ops::op::max_pool2d, {u}, {{"kernel", 2.0}});
    g.output("y", g.apply(ops::op::l2_normalize, {g.apply(ops::op::global_avg_pool, {p})}));
    auto pl = compile(g); executor e(*pl);
    e.set_input("img", img.data); e.run();
    auto R = [](ops::op o, std::vector<tensor> in, ops::attrs at = {}) { return ops::reference(o)(in, at); };
    auto want = R(ops::op::l2_normalize, {R(ops::op::global_avg_pool, {R(ops::op::max_pool2d, {R(ops::op::upsample_bilinear, {R(ops::op::hard_swish,
        {R(ops::op::group_norm, {R(ops::op::conv2d, {img, w1, b1}, {{"stride", 2.0}, {"pad", 1.0}}), gw, gb}, {{"groups", 2.0}})})}, {{"scale", 2.0}})}, {{"kernel", 2.0}})})});
    CHECK(max_diff(e.output("y"), want) < 1e-5);
}
