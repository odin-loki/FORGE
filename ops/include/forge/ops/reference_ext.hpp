#pragma once
// Scalar references of the spatial (ops/defs/spatial.fspec) and encoder (ops/defs/sequence.fspec) op families.
// Included at the end of reference.hpp. Clarity over speed; sums are accumulated in double.
#include <forge/ops/op.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace forge::ops::ref {

// ---- elementwise ---------------------------------------------------------------------------------------------------
inline tensor quick_gelu(const tv& in, const attrs&) { return detail::map1(in, [](double x) { return x * detail::sigm(1.702 * x); }); }
inline tensor mish(const tv& in, const attrs&) {
    return detail::map1(in, [](double x) { return x * std::tanh(x > 20 ? x : std::log1p(std::exp(x))); });
}
inline tensor hard_swish(const tv& in, const attrs&) { return detail::map1(in, [](double x) { return x * std::clamp(x + 3.0, 0.0, 6.0) / 6.0; }); }
inline tensor floor(const tv& in, const attrs&) { return detail::map1(in, [](double x) { return std::floor(x); }); }

// ---- convolution ---------------------------------------------------------------------------------------------------
namespace detail {
inline double bias_at(const tensor& b, i64 o) { return b.data[b.size() == 1 ? 0 : std::size_t(o)]; }
}
inline tensor conv2d(const tv& in, const attrs& a) {
    const auto& x = in.at(0); const auto& w = in.at(1); const auto& b = in.at(2);
    tensor o(infer_shape(shape_rule::conv2d, {x.shape, w.shape, b.shape}, a));
    auto st = attr2(a, "stride", 1), pd = attr2(a, "pad", 0), dl = attr2(a, "dilation", 1);
    i64 C = x.shape[0], H = x.shape[1], W = x.shape[2], O = w.shape[0], cg = w.shape[1], kh = w.shape[2], kw = w.shape[3];
    i64 g = C / cg, og = O / g, oh = o.shape[1], ow = o.shape[2];
    for (i64 oc = 0; oc < O; ++oc) {
        i64 c0 = (oc / og) * cg;
        for (i64 y = 0; y < oh; ++y) for (i64 xx = 0; xx < ow; ++xx) {
            double s = detail::bias_at(b, oc);
            for (i64 c = 0; c < cg; ++c) for (i64 i = 0; i < kh; ++i) for (i64 j = 0; j < kw; ++j) {
                i64 iy = y * st[0] - pd[0] + i * dl[0], ix = xx * st[1] - pd[1] + j * dl[1];
                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                s += double(x.data[std::size_t(((c0 + c) * H + iy) * W + ix)]) * w.data[std::size_t(((oc * cg + c) * kh + i) * kw + j)];
            }
            o.data[std::size_t((oc * oh + y) * ow + xx)] = float(s);
        }
    }
    return o;
}
inline tensor conv1d(const tv& in, const attrs& a) {
    const auto& x = in.at(0); const auto& w = in.at(1); const auto& b = in.at(2);
    tensor o(infer_shape(shape_rule::conv1d, {x.shape, w.shape, b.shape}, a));
    i64 st = i64(a.num("stride", 1)), pd = i64(a.num("pad", 0)), dl = i64(a.num("dilation", 1));
    i64 C = x.shape[0], L = x.shape[1], O = w.shape[0], cg = w.shape[1], k = w.shape[2], g = C / cg, og = O / g, ol = o.shape[1];
    for (i64 oc = 0; oc < O; ++oc) {
        i64 c0 = (oc / og) * cg;
        for (i64 p = 0; p < ol; ++p) {
            double s = detail::bias_at(b, oc);
            for (i64 c = 0; c < cg; ++c) for (i64 j = 0; j < k; ++j) {
                i64 ix = p * st - pd + j * dl;
                if (ix < 0 || ix >= L) continue;
                s += double(x.data[std::size_t((c0 + c) * L + ix)]) * w.data[std::size_t((oc * cg + c) * k + j)];
            }
            o.data[std::size_t(oc * ol + p)] = float(s);
        }
    }
    return o;
}
inline tensor conv_transpose2d(const tv& in, const attrs& a) {
    const auto& x = in.at(0); const auto& w = in.at(1); const auto& b = in.at(2);
    tensor o(infer_shape(shape_rule::conv_transpose2d, {x.shape, w.shape, b.shape}, a));
    auto st = attr2(a, "stride", 1), pd = attr2(a, "pad", 0), dl = attr2(a, "dilation", 1);
    i64 C = x.shape[0], H = x.shape[1], W = x.shape[2], og = w.shape[1], kh = w.shape[2], kw = w.shape[3];
    i64 g = i64(a.num("groups", 1)), cg = C / g, O = o.shape[0], oh = o.shape[1], ow = o.shape[2];
    std::vector<double> acc(o.size());
    for (i64 oc = 0; oc < O; ++oc) std::fill_n(acc.begin() + oc * oh * ow, oh * ow, detail::bias_at(b, oc));
    for (i64 c = 0; c < C; ++c) {
        i64 grp = c / cg;
        for (i64 y = 0; y < H; ++y) for (i64 xx = 0; xx < W; ++xx) {
            double xv = x.data[std::size_t((c * H + y) * W + xx)];
            for (i64 ol = 0; ol < og; ++ol) for (i64 i = 0; i < kh; ++i) for (i64 j = 0; j < kw; ++j) {
                i64 oy = y * st[0] - pd[0] + i * dl[0], ox = xx * st[1] - pd[1] + j * dl[1];
                if (oy < 0 || oy >= oh || ox < 0 || ox >= ow) continue;
                acc[std::size_t(((grp * og + ol) * oh + oy) * ow + ox)] += xv * w.data[std::size_t(((c * og + ol) * kh + i) * kw + j)];
            }
        }
    }
    for (std::size_t i = 0; i < acc.size(); ++i) o.data[i] = float(acc[i]);
    return o;
}

// ---- pooling / resize ----------------------------------------------------------------------------------------------
namespace detail {
inline tensor pool2d(const tv& in, const attrs& a, bool is_max) {
    const auto& x = in.at(0);
    tensor o(infer_shape(shape_rule::pool2d, {x.shape}, a));
    auto k = attr2(a, "kernel", 2), st = a.has("stride") ? attr2(a, "stride", 1) : k, pd = attr2(a, "pad", 0);
    bool count_pad = a.num("count_pad", 1) != 0;
    i64 C = x.shape[0], H = x.shape[1], W = x.shape[2], oh = o.shape[1], ow = o.shape[2];
    for (i64 c = 0; c < C; ++c) for (i64 y = 0; y < oh; ++y) for (i64 xx = 0; xx < ow; ++xx) {
        double s = 0, m = -std::numeric_limits<double>::infinity(); i64 cnt = 0;
        for (i64 i = 0; i < k[0]; ++i) for (i64 j = 0; j < k[1]; ++j) {
            i64 iy = y * st[0] - pd[0] + i, ix = xx * st[1] - pd[1] + j;
            if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
            double v = x.data[std::size_t((c * H + iy) * W + ix)];
            s += v; m = std::max(m, v); ++cnt;
        }
        double den = count_pad ? double(k[0] * k[1]) : double(std::max<i64>(cnt, 1));
        o.data[std::size_t((c * oh + y) * ow + xx)] = float(is_max ? m : s / den);
    }
    return o;
}
// source coordinate of output cell `d` for a bilinear resize: lower index, upper index, weight of the upper one
inline void bilinear_src(i64 d, i64 n, i64 on, bool align, i64& i0, i64& i1, double& t) {
    double s = align ? (on > 1 ? double(d) * double(n - 1) / double(on - 1) : 0.0)
                     : std::max(0.0, (double(d) + 0.5) * double(n) / double(on) - 0.5);
    i0 = std::min<i64>(i64(s), n - 1); i1 = std::min<i64>(i0 + 1, n - 1); t = s - double(i0);
}
}
inline tensor max_pool2d(const tv& in, const attrs& a) { return detail::pool2d(in, a, true); }
inline tensor avg_pool2d(const tv& in, const attrs& a) { return detail::pool2d(in, a, false); }
inline tensor adaptive_avg_pool2d(const tv& in, const attrs& a) {
    const auto& x = in.at(0);
    tensor o(infer_shape(shape_rule::adaptive_pool2d, {x.shape}, a));
    i64 C = x.shape[0], H = x.shape[1], W = x.shape[2], oh = o.shape[1], ow = o.shape[2];
    for (i64 c = 0; c < C; ++c) for (i64 y = 0; y < oh; ++y) for (i64 xx = 0; xx < ow; ++xx) {
        i64 y0 = y * H / oh, y1 = ((y + 1) * H + oh - 1) / oh, x0 = xx * W / ow, x1 = ((xx + 1) * W + ow - 1) / ow;
        double s = 0;
        for (i64 i = y0; i < y1; ++i) for (i64 j = x0; j < x1; ++j) s += x.data[std::size_t((c * H + i) * W + j)];
        o.data[std::size_t((c * oh + y) * ow + xx)] = float(s / double(std::max<i64>((y1 - y0) * (x1 - x0), 1)));
    }
    return o;
}
inline tensor global_avg_pool(const tv& in, const attrs& a) {
    const auto& x = in.at(0);
    tensor o(infer_shape(shape_rule::global_pool, {x.shape}, a));
    i64 C = x.shape[0], n = C ? i64(x.size()) / C : 0;
    for (i64 c = 0; c < C; ++c) {
        double s = 0; for (i64 i = 0; i < n; ++i) s += x.data[std::size_t(c * n + i)];
        o.data[std::size_t(c)] = float(n ? s / double(n) : 0.0);
    }
    return o;
}
inline tensor upsample_nearest(const tv& in, const attrs& a) {
    const auto& x = in.at(0);
    tensor o(infer_shape(shape_rule::upsample2d, {x.shape}, a));
    auto sc = attr2(a, "scale", 2);
    i64 C = x.shape[0], H = x.shape[1], W = x.shape[2], oh = o.shape[1], ow = o.shape[2];
    for (i64 c = 0; c < C; ++c) for (i64 y = 0; y < oh; ++y) for (i64 xx = 0; xx < ow; ++xx)
        o.data[std::size_t((c * oh + y) * ow + xx)] = x.data[std::size_t((c * H + y / sc[0]) * W + xx / sc[1])];
    return o;
}
inline tensor upsample_bilinear(const tv& in, const attrs& a) {
    const auto& x = in.at(0);
    tensor o(infer_shape(shape_rule::upsample2d, {x.shape}, a));
    bool align = a.num("align_corners", 0) != 0;
    i64 C = x.shape[0], H = x.shape[1], W = x.shape[2], oh = o.shape[1], ow = o.shape[2];
    for (i64 c = 0; c < C; ++c) for (i64 y = 0; y < oh; ++y) {
        i64 y0, y1; double ty; detail::bilinear_src(y, H, oh, align, y0, y1, ty);
        for (i64 xx = 0; xx < ow; ++xx) {
            i64 x0, x1; double tx; detail::bilinear_src(xx, W, ow, align, x0, x1, tx);
            auto at = [&](i64 i, i64 j) { return double(x.data[std::size_t((c * H + i) * W + j)]); };
            double top = at(y0, x0) * (1 - tx) + at(y0, x1) * tx, bot = at(y1, x0) * (1 - tx) + at(y1, x1) * tx;
            o.data[std::size_t((c * oh + y) * ow + xx)] = float(top * (1 - ty) + bot * ty);
        }
    }
    return o;
}

// ---- normalisation -------------------------------------------------------------------------------------------------
inline tensor group_norm(const tv& in, const attrs& a) {
    const auto& x = in.at(0); const auto& w = in.at(1); const auto& b = in.at(2);
    if (x.shape.size() < 2) raise(errc::shape_mismatch, "group_norm expects [c, ...]");
    i64 C = x.shape[0], n = C ? i64(x.size()) / C : 0, g = i64(a.num("groups", 32)); double eps = a.num("eps", 1e-5);
    if (g < 1 || C % g || i64(w.size()) != C || i64(b.size()) != C) raise(errc::shape_mismatch, "group_norm: {} channels, {} groups, w {} b {}", C, g, w.size(), b.size());
    tensor o(x.shape); i64 cg = C / g, m = cg * n;
    for (i64 gi = 0; gi < g; ++gi) {
        const float* p = x.data.data() + gi * m;
        double mu = 0; for (i64 i = 0; i < m; ++i) mu += p[i];
        mu /= double(std::max<i64>(m, 1));
        double var = 0; for (i64 i = 0; i < m; ++i) var += (p[i] - mu) * (p[i] - mu);
        var /= double(std::max<i64>(m, 1));
        double r = 1.0 / std::sqrt(var + eps);
        for (i64 i = 0; i < m; ++i) { auto c = std::size_t(gi * cg + i / n); o.data[std::size_t(gi * m + i)] = float((p[i] - mu) * r * w.data[c] + b.data[c]); }
    }
    return o;
}
inline tensor batch_norm(const tv& in, const attrs& a) {
    const auto& x = in.at(0); const auto& w = in.at(1); const auto& b = in.at(2); const auto& mu = in.at(3); const auto& var = in.at(4);
    if (x.shape.empty()) raise(errc::shape_mismatch, "batch_norm expects [c, ...]");
    i64 C = x.shape[0], n = C ? i64(x.size()) / C : 0; double eps = a.num("eps", 1e-5);
    for (const tensor* t : {&w, &b, &mu, &var}) if (i64(t->size()) != C) raise(errc::shape_mismatch, "batch_norm: parameter of {} values for {} channels", t->size(), C);
    tensor o(x.shape);
    for (i64 c = 0; c < C; ++c) {
        auto cc = std::size_t(c); double r = double(w.data[cc]) / std::sqrt(double(var.data[cc]) + eps);
        for (i64 i = 0; i < n; ++i) o.data[std::size_t(c * n + i)] = float((double(x.data[std::size_t(c * n + i)]) - mu.data[cc]) * r + b.data[cc]);
    }
    return o;
}
inline tensor l2_normalize(const tv& in, const attrs& a) {
    double eps = a.num("eps", 1e-12);
    return last_axis_rows(in.at(0), [&](const float* x, float* y, i64 n) {
        double s = 0; for (i64 i = 0; i < n; ++i) s += double(x[i]) * x[i];
        double r = 1.0 / std::max(std::sqrt(s), eps);
        for (i64 i = 0; i < n; ++i) y[i] = float(x[i] * r);
    });
}

// ---- movement ------------------------------------------------------------------------------------------------------
inline tensor pad(const tv& in, const attrs& a) {
    const auto& x = in.at(0);
    tensor o(infer_shape(shape_rule::pad, {x.shape}, a));
    auto pp = pad_pairs(a, x.shape.size());
    i64 mode = i64(a.num("mode", 0)); float value = float(a.num("value", 0));
    auto ist = detail::strides(x.shape);
    for (i64 f = 0; f < i64(o.size()); ++f) {
        i64 rem = f, src = 0; bool outside = false;
        for (int d = int(o.shape.size()) - 1; d >= 0; --d) {
            auto dd = std::size_t(d);
            i64 c = rem % o.shape[dd] - pp[dd][0], n = x.shape[dd]; rem /= o.shape[dd];
            if (c < 0 || c >= n) {
                if (mode == 0) { outside = true; continue; }
                c = mode == 1 ? (c < 0 ? -c : 2 * (n - 1) - c) : std::clamp<i64>(c, 0, n - 1);
            }
            src += c * ist[dd];
        }
        o.data[std::size_t(f)] = outside ? value : x.data[std::size_t(src)];
    }
    return o;
}
inline tensor index_select(const tv& in, const attrs& a) {
    const auto& x = in.at(0); const auto& idx = in.at(1);
    tensor o(infer_shape(shape_rule::index_select, {x.shape, idx.shape}, a));
    i64 ax = norm_axis(i64(a.num("axis", 0)), x.shape.size());
    auto v = detail::split_axis(x.shape, ax); i64 m = idx.shape[0];
    for (i64 j = 0; j < m; ++j) {
        i64 r = i64(idx.data[std::size_t(j)]); if (r < 0) r += v.n;
        if (r < 0 || r >= v.n) raise(errc::invalid_argument, "index_select index {} out of range for axis of {}", i64(idx.data[std::size_t(j)]), v.n);
        for (i64 p = 0; p < v.outer; ++p)
            std::copy_n(&x.data[std::size_t((p * v.n + r) * v.inner)], v.inner, &o.data[std::size_t((p * m + j) * v.inner)]);
    }
    return o;
}
namespace detail {
inline tensor topk_impl(const tv& in, const attrs& a, bool indices) {
    const auto& x = in.at(0);
    tensor o(infer_shape(shape_rule::topk, {x.shape}, a));
    i64 n = x.shape.back(), k = o.shape.back(), rows = n ? i64(x.size()) / n : 0;
    std::vector<i64> idx(static_cast<std::size_t>(n));
    for (i64 r = 0; r < rows; ++r) {
        const float* p = x.data.data() + r * n;
        for (i64 i = 0; i < n; ++i) idx[std::size_t(i)] = i;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](i64 l, i64 m) { return p[l] != p[m] ? p[l] > p[m] : l < m; });
        for (i64 j = 0; j < k; ++j) o.data[std::size_t(r * k + j)] = indices ? float(idx[std::size_t(j)]) : p[idx[std::size_t(j)]];
    }
    return o;
}
}
inline tensor topk(const tv& in, const attrs& a) { return detail::topk_impl(in, a, false); }
inline tensor topk_indices(const tv& in, const attrs& a) { return detail::topk_impl(in, a, true); }
inline tensor masked_mean(const tv& in, const attrs& a) {
    const auto& x = in.at(0); const auto& m = in.at(1);
    tensor o(infer_shape(shape_rule::masked_mean, {x.shape, m.shape}, a));
    i64 s = x.shape[0], d = i64(o.size());
    std::vector<double> acc(o.size(), 0.0); i64 cnt = 0;
    for (i64 i = 0; i < s; ++i) {
        if (m.data[std::size_t(i)] == 0) continue;
        ++cnt; for (i64 j = 0; j < d; ++j) acc[std::size_t(j)] += x.data[std::size_t(i * d + j)];
    }
    for (i64 j = 0; j < d; ++j) o.data[std::size_t(j)] = float(cnt ? acc[std::size_t(j)] / double(cnt) : 0.0);
    return o;
}

// ---- positions -----------------------------------------------------------------------------------------------------
// ALiBi head slopes (Press et al.): 2^(-8 (h + 1) / n) for a power-of-two n, else the slopes of the power of two
// below n followed by every other slope of twice that.
inline std::vector<double> alibi_slopes(i64 n) {
    auto pow2 = [](i64 m) { std::vector<double> s; double st = std::pow(2.0, -8.0 / double(m)); for (i64 i = 0; i < m; ++i) s.push_back(std::pow(st, double(i + 1))); return s; };
    if (n < 1) return {};
    i64 p = 1; while (p * 2 <= n) p *= 2;
    auto s = pow2(p);
    if (p != n) { auto e = pow2(2 * p); for (i64 i = 0; i64(s.size()) < n; i += 2) s.push_back(e[std::size_t(i)]); }
    return s;
}
inline tensor alibi_bias(const tv& in, const attrs& a) {
    const auto& qp = in.at(0); const auto& kp = in.at(1);
    tensor o(infer_shape(shape_rule::alibi, {qp.shape, kp.shape}, a));
    i64 H = o.shape[0], s = o.shape[1], t = o.shape[2]; auto sl = alibi_slopes(H);
    for (i64 h = 0; h < H; ++h) for (i64 i = 0; i < s; ++i) for (i64 j = 0; j < t; ++j)
        o.data[std::size_t((h * s + i) * t + j)] = float(-sl[std::size_t(h)] * std::fabs(double(qp.data[std::size_t(i)]) - double(kp.data[std::size_t(j)])));
    return o;
}
inline tensor sinusoid_embed(const tv& in, const attrs& a) {
    const auto& x = in.at(0);
    tensor o(infer_shape(shape_rule::embed_dim, {x.shape}, a));
    i64 dim = o.shape.back(), half = dim / 2; bool flip = a.num("flip", 0) != 0;
    double mp = a.num("max_period", 10000.0), den = std::max(1.0, double(half) - a.num("shift", 0));
    for (i64 r = 0; r < i64(x.size()); ++r) {
        float* y = o.data.data() + r * dim;
        for (i64 i = 0; i < half; ++i) {
            double th = double(x.data[std::size_t(r)]) * std::exp(-std::log(mp) * double(i) / den);
            y[flip ? half + i : i] = float(std::sin(th)); y[flip ? i : half + i] = float(std::cos(th));
        }
        if (dim % 2) y[dim - 1] = 0;
    }
    return o;
}

// ---- attention with an additive bias -------------------------------------------------------------------------------
// q[s,hq,d], k[t,hk,d], v[t,hk,dv], bias broadcast (right-aligned) to [hq, s, t]. Query i sits at position t - s + i.
// causal: keys after the query are masked. window w > 0: keys w or more positions away are masked (behind the query;
// also ahead of it when not causal). softcap c > 0: scores -> c tanh(scores / c) before the bias. A bias of -inf
// masks a key. A query whose keys are all masked yields 0.
inline tensor attention_bias(const tv& in, const attrs& a) {
    const auto& q = in.at(0); const auto& k = in.at(1); const auto& v = in.at(2); const auto& bias = in.at(3);
    infer_shape(shape_rule::attention, {q.shape, k.shape, v.shape, bias.shape}, a);
    i64 s = q.shape[0], hq = q.shape[1], d = q.shape[2], t = k.shape[0], hk = k.shape[1], dv = v.shape[2], grp = hq / hk;
    bool causal = a.num("causal", 0) != 0;
    double sc = a.num("scale", 1.0 / std::sqrt(double(d))), cap = a.num("softcap", 0);
    i64 window = i64(a.num("window", 0));
    const shape_t full{hq, s, t};
    tensor o(shape_t{s, hq, dv});
    std::vector<double> sco(static_cast<std::size_t>(t));
    for (i64 i = 0; i < s; ++i) {
        i64 pos = t - s + i;
        for (i64 h = 0; h < hq; ++h) {
            i64 kh = h / grp; double mx = -INFINITY;
            for (i64 j = 0; j < t; ++j) {
                bool ok = !causal || j <= pos;
                if (window > 0 && (pos - j >= window || j - pos >= window)) ok = false;
                double bv = bias.data[std::size_t(detail::bindex((h * s + i) * t + j, full, bias.shape))];
                if (!ok || (std::isinf(bv) && bv < 0)) { sco[std::size_t(j)] = -INFINITY; continue; }
                double dot = 0;
                for (i64 e = 0; e < d; ++e) dot += double(q.data[std::size_t((i * hq + h) * d + e)]) * k.data[std::size_t((j * hk + kh) * d + e)];
                dot *= sc; if (cap > 0) dot = cap * std::tanh(dot / cap);
                sco[std::size_t(j)] = dot + bv; mx = std::max(mx, sco[std::size_t(j)]);
            }
            double sum = 0; for (auto& x : sco) { x = std::isinf(x) ? 0 : std::exp(x - mx); sum += x; }
            for (i64 e = 0; e < dv; ++e) {
                double acc = 0;
                for (i64 j = 0; j < t; ++j) acc += sco[std::size_t(j)] * v.data[std::size_t((j * hk + kh) * dv + e)];
                o.data[std::size_t((i * hq + h) * dv + e)] = float(sum > 0 ? acc / sum : 0.0);
            }
        }
    }
    return o;
}

} // namespace forge::ops::ref
