#pragma once
// CPU lowerings of the spatial / encoder op families (ops/defs/spatial.fspec, sequence.fspec). Included by kernels.hpp;
// registered by add_ext_cpu() at the end of registry::make_cpu. Ops without an entry here run the scalar reference.
//   conv2d / conv1d : im2col + the blocked gemm (gpu::prims::gemm_nt), one gemm per group; depthwise runs direct loops
//   gelu, quick_gelu, mish, hard_swish, floor : float strips over the task pool
//   layer_norm, l2_normalize : one row per task
//   attention_bias : the plain attention loop (simd dot / saxpy) with the bias, soft-cap and masks of the reference
#include <cmath>
#include <vector>

namespace forge::graph::cpu_kernels {

template <class F> kernel_fn map_float(F f) {
    return [f](std::span<const tensor_view> in, tensor_view out, const ops::attrs&) {
        task::parallel_for(out.size(), 8192, [&](std::size_t b, std::size_t e) { for (std::size_t i = b; i < e; ++i) out.data[i] = f(in[0].data[i]); });
    };
}

struct conv_geom { std::size_t C, H, W, O, cg, kh, kw, oh, ow, sh, sw, ph, pw, dh, dw; };
// x[C, H, W] * w[O, cg, kh, kw] + b -> out[O, oh, ow]
inline void conv_core(const float* x, const float* w, const float* b, std::size_t bn, float* out, const conv_geom& g) {
    const std::size_t groups = g.C / g.cg, og = g.O / groups, np = g.oh * g.ow, k = g.cg * g.kh * g.kw;
    auto bias = [&](std::size_t oc) { return b[bn == 1 ? 0 : oc]; };
    auto src = [&](std::size_t c, std::size_t y, std::size_t xx, std::size_t i, std::size_t j, float& v) {
        std::ptrdiff_t iy = std::ptrdiff_t(y * g.sh + i * g.dh) - std::ptrdiff_t(g.ph), ix = std::ptrdiff_t(xx * g.sw + j * g.dw) - std::ptrdiff_t(g.pw);
        if (iy < 0 || iy >= std::ptrdiff_t(g.H) || ix < 0 || ix >= std::ptrdiff_t(g.W)) return false;
        v = x[(c * g.H + std::size_t(iy)) * g.W + std::size_t(ix)]; return true;
    };
    if (g.cg == 1 && og == 1) {   // depthwise: no gemm to win from
        task::parallel_for(g.O, 1, [&](std::size_t b0, std::size_t e0) {
            for (std::size_t oc = b0; oc < e0; ++oc) for (std::size_t y = 0; y < g.oh; ++y) for (std::size_t xx = 0; xx < g.ow; ++xx) {
                float s = bias(oc), v;
                for (std::size_t i = 0; i < g.kh; ++i) for (std::size_t j = 0; j < g.kw; ++j) if (src(oc, y, xx, i, j, v)) s += v * w[(oc * g.kh + i) * g.kw + j];
                out[(oc * g.oh + y) * g.ow + xx] = s;
            }
        });
        return;
    }
    std::vector<float> cols(np * k), prod(np * og);
    for (std::size_t gi = 0; gi < groups; ++gi) {
        task::parallel_for(np, 64, [&](std::size_t b0, std::size_t e0) {
            for (std::size_t p = b0; p < e0; ++p) {
                std::size_t y = p / g.ow, xx = p % g.ow; float* col = cols.data() + p * k;
                for (std::size_t c = 0; c < g.cg; ++c) for (std::size_t i = 0; i < g.kh; ++i) for (std::size_t j = 0; j < g.kw; ++j) {
                    float v = 0.f; src(gi * g.cg + c, y, xx, i, j, v); *col++ = v;
                }
            }
        });
        gpu::prims::gemm_nt(cols.data(), w + gi * og * k, prod.data(), np, og, k);
        for (std::size_t oc = 0; oc < og; ++oc) {
            float* o = out + (gi * og + oc) * np; float bv = bias(gi * og + oc);
            for (std::size_t p = 0; p < np; ++p) o[p] = prod[p * og + oc] + bv;
        }
    }
}
inline kernel_fn conv2d() {
    return [](std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
        auto st = ops::attr2(a, "stride", 1), pd = ops::attr2(a, "pad", 0), dl = ops::attr2(a, "dilation", 1);
        const auto &x = in[0].shape, &w = in[1].shape;
        auto u = [](i64 v) { return std::size_t(v); };
        conv_geom g{u(x[0]), u(x[1]), u(x[2]), u(w[0]), u(w[1]), u(w[2]), u(w[3]), u(out.shape[1]), u(out.shape[2]),
                    u(st[0]), u(st[1]), u(pd[0]), u(pd[1]), u(dl[0]), u(dl[1])};
        conv_core(in[0].data, in[1].data, in[2].data, in[2].size(), out.data, g);
    };
}
inline kernel_fn conv1d() {
    return [](std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
        const auto &x = in[0].shape, &w = in[1].shape;
        auto u = [](i64 v) { return std::size_t(v); };
        conv_geom g{u(x[0]), 1, u(x[1]), u(w[0]), u(w[1]), 1, u(w[2]), 1, u(out.shape[1]),
                    1, u(i64(a.num("stride", 1))), 0, u(i64(a.num("pad", 0))), 1, u(i64(a.num("dilation", 1)))};
        conv_core(in[0].data, in[1].data, in[2].data, in[2].size(), out.data, g);
    };
}
inline kernel_fn layer_norm() {
    return [](std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
        std::size_t r, n; rows(in[0], r, n);
        double eps = a.num("eps", 1e-5); const float* w = in[1].data; const float* b = in[2].data;
        task::parallel_for(r, 4, [&](std::size_t b0, std::size_t e0) {
            for (std::size_t i = b0; i < e0; ++i) {
                const float* x = in[0].data + i * n; float* y = out.data + i * n;
                double m = 0; for (std::size_t j = 0; j < n; ++j) m += x[j];
                m /= double(n);
                double v = 0; for (std::size_t j = 0; j < n; ++j) { double d = x[j] - m; v += d * d; }
                float inv = float(1.0 / std::sqrt(v / double(n) + eps)), mf = float(m);
                for (std::size_t j = 0; j < n; ++j) y[j] = (x[j] - mf) * inv * w[j] + b[j];
            }
        });
    };
}
inline bool norm_params_1d(sp in, const shape_t&, const ops::attrs&) {
    return in.size() == 3 && !in[0].empty() && in[1].size() == 1 && in[2].size() == 1 && in[1][0] == in[0].back() && in[2][0] == in[0].back();
}
inline kernel_fn l2_normalize() {
    return [](std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
        std::size_t r, n; rows(in[0], r, n);
        double eps = a.num("eps", 1e-12);
        task::parallel_for(r, 4, [&](std::size_t b0, std::size_t e0) {
            for (std::size_t i = b0; i < e0; ++i) {
                const float* x = in[0].data + i * n; float* y = out.data + i * n;
                double s = 0; for (std::size_t j = 0; j < n; ++j) s += double(x[j]) * x[j];
                float inv = float(1.0 / std::max(std::sqrt(s), eps));
                for (std::size_t j = 0; j < n; ++j) y[j] = x[j] * inv;
            }
        });
    };
}
// f32 keys / values only: plan.hpp refuses attention_bias over a quantised KV slot.
inline kernel_fn attention_bias() {
    return [](std::span<const tensor_view> in, tensor_view out, const ops::attrs& a) {
        const auto &q = in[0], &k = in[1], &v = in[2], &bias = in[3];
        std::size_t s = std::size_t(q.shape[0]), hq = std::size_t(q.shape[1]), d = std::size_t(q.shape[2]), dv = std::size_t(v.shape[2]);
        std::size_t t = std::size_t(k.shape[0]), hk = std::size_t(k.shape[1]), grp = hq / hk;
        bool causal = a.num("causal", 0) != 0;
        float sc = float(a.num("scale", 1.0 / std::sqrt(double(d)))), cap = float(a.num("softcap", 0));
        std::size_t window = std::size_t(a.num("window", 0));
        // bias strides for a right-aligned broadcast against [hq, s, t]
        std::size_t bd[3] = {1, 1, 1}; const auto& bs = bias.shape;
        for (std::size_t i = 0; i < bs.size(); ++i) bd[3 - bs.size() + i] = std::size_t(bs[i]);
        const auto& ker = simd::active();
        task::parallel_for(s * hq, 1, [&](std::size_t b0, std::size_t e0) {
            std::vector<float> sco(t), acc(dv);
            for (std::size_t w = b0; w < e0; ++w) {
                std::size_t i = w / hq, h = w % hq, kh = h / grp, pos = t - s + i;
                std::size_t lo = 0, hi = causal ? pos + 1 : t;
                if (window && pos + 1 > window) lo = pos + 1 - window;
                if (window && !causal) hi = std::min(t, pos + window);
                const float* qp = q.data + (i * hq + h) * d;
                const float* brow = bias.data + ((bd[0] == 1 ? 0 : h) * bd[1] + (bd[1] == 1 ? 0 : i)) * bd[2];
                float mx = -INFINITY;
                for (std::size_t j = lo; j < hi; ++j) {
                    float bv = brow[bd[2] == 1 ? 0 : j];
                    if (std::isinf(bv) && bv < 0) { sco[j] = -INFINITY; continue; }
                    float x = ker.dot(qp, k.data + (j * hk + kh) * d, d) * sc;
                    if (cap > 0) x = cap * std::tanh(x / cap);
                    sco[j] = x + bv; mx = std::max(mx, sco[j]);
                }
                float sum = 0;
                for (std::size_t j = lo; j < hi; ++j) { sco[j] = std::isinf(sco[j]) ? 0.f : std::exp(sco[j] - mx); sum += sco[j]; }
                std::fill(acc.begin(), acc.end(), 0.f);
                if (sum > 0) for (std::size_t j = lo; j < hi; ++j) if (sco[j] != 0.f) ker.saxpy(sco[j] / sum, v.data + (j * hk + kh) * dv, acc.data(), dv);
                std::copy(acc.begin(), acc.end(), out.data + (i * hq + h) * dv);
            }
        });
    };
}

inline void add_ext_cpu(registry& r) {
    r.add({"cpu.conv2d.im2col", ops::op::conv2d, 10, always, conv2d()});
    r.add({"cpu.conv1d.im2col", ops::op::conv1d, 10, always, conv1d()});
    r.add({"cpu.gelu", ops::op::gelu, 10, always, map_float([](float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f)); })});
    r.add({"cpu.quick_gelu", ops::op::quick_gelu, 10, always, map_float([](float x) { return x / (1.0f + std::exp(-1.702f * x)); })});
    r.add({"cpu.mish", ops::op::mish, 10, always, map_float([](float x) { return x * std::tanh(x > 20.f ? x : std::log1p(std::exp(x))); })});
    r.add({"cpu.hard_swish", ops::op::hard_swish, 10, always, map_float([](float x) { return x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f; })});
    r.add({"cpu.floor", ops::op::floor, 10, always, map_float([](float x) { return std::floor(x); })});
    r.add({"cpu.layer_norm", ops::op::layer_norm, 10, norm_params_1d, layer_norm()});
    r.add({"cpu.l2_normalize", ops::op::l2_normalize, 10, always, l2_normalize()});
    r.add({"cpu.attention_bias", ops::op::attention_bias, 10, always, attention_bias()});
}

} // namespace forge::graph::cpu_kernels
