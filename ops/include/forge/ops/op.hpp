#pragma once
// Op metadata types (class, shape rule, tolerance, fusion) used by the table generated from ops/defs/*.fspec.
// See ops/README.md.
#include <forge/core/tensor.hpp>
#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace forge::ops {

enum class op_class : u8 { elementwise, reduction, contraction, data_movement, stateful };
enum class shape_rule : u8 { same, broadcast, reduce, matmul, matmul_t, attention, transpose, reshape,
                             gather, concat, slice, expand,
                             // spatial / encoder rules (infer_shape_ext below); append only
                             conv1d, conv2d, conv_transpose2d, pool2d, adaptive_pool2d, global_pool, upsample2d, pad,
                             alibi, embed_dim, topk, index_select, masked_mean };
enum class tol_kind : u8 { exact, ulp, rel, abs };
enum class fusion : u8 { none, elementwise, epilogue };

struct tolerance { tol_kind kind; double value; };

struct op_info {
    std::string_view name;
    op_class cls;
    int inputs;
    shape_rule shape;
    tolerance tol;
    fusion fuse;
    double cost;   // flops per output element (contraction: per MAC)
    std::string_view attrs;
    std::string_view doc;
};

using attr_value = std::variant<double, std::vector<i64>, std::string>;

struct attrs {
    std::map<std::string, attr_value, std::less<>> v;
    attrs() = default;
    attrs(std::initializer_list<std::pair<const std::string, attr_value>> il) : v(il) {}
    double num(std::string_view k, double dflt = 0) const {
        auto it = v.find(k);
        if (it == v.end()) return dflt;
        if (auto* d = std::get_if<double>(&it->second)) return *d;
        if (auto* iv = std::get_if<std::vector<i64>>(&it->second); iv && iv->size() == 1) return double((*iv)[0]);
        raise(errc::invalid_argument, "attr '{}' is not a number", k);
    }
    std::vector<i64> ints(std::string_view k) const {
        auto it = v.find(k);
        if (it == v.end()) return {};
        if (auto* iv = std::get_if<std::vector<i64>>(&it->second)) return *iv;
        if (auto* d = std::get_if<double>(&it->second)) return {i64(*d)};
        raise(errc::invalid_argument, "attr '{}' is not an int list", k);
    }
    std::string str(std::string_view k, std::string dflt = {}) const {
        auto it = v.find(k);
        if (it == v.end()) return dflt;
        if (auto* s = std::get_if<std::string>(&it->second)) return *s;
        raise(errc::invalid_argument, "attr '{}' is not a string", k);
    }
    bool has(std::string_view k) const { return v.find(k) != v.end(); }
    void set(std::string k, attr_value a) { v[std::move(k)] = std::move(a); }
    bool operator==(const attrs&) const = default;
};

using ref_fn = tensor (*)(const std::vector<tensor>&, const attrs&);

inline i64 norm_axis(i64 axis, std::size_t rank) {
    i64 r = i64(rank);
    if (axis < 0) axis += r;
    if (axis < 0 || axis >= r) raise(errc::invalid_argument, "axis {} out of range for rank {}", axis, rank);
    return axis;
}

inline shape_t broadcast_shapes(const std::vector<shape_t>& in) {
    std::size_t rank = 0;
    for (auto& s : in) rank = std::max(rank, s.size());
    shape_t out(rank, 1);
    for (auto& s : in) {
        for (std::size_t i = 0; i < s.size(); ++i) {
            auto& o = out[rank - s.size() + i];
            i64 d = s[i];
            if (o == 1) o = d;
            else if (d != 1 && d != o)
                raise(errc::shape_mismatch, "cannot broadcast {} with {}", shape_str(s), shape_str(out));
        }
    }
    return out;
}

// ---- spatial / encoder shape rules -------------------------------------------------------------------------------
// Spatial ops take one sample without a batch axis: x[c, l] (1-D) or x[c, h, w] (2-D), like sequences are [s, d].
// A 2-D attr (stride, pad, dilation, kernel, scale, out) is one number for both axes or a list [h, w].
inline std::array<i64, 2> attr2(const attrs& a, std::string_view k, i64 dflt) {
    auto v = a.ints(k);
    if (v.empty()) return {dflt, dflt};
    if (v.size() == 1) return {v[0], v[0]};
    if (v.size() == 2) return {v[0], v[1]};
    raise(errc::invalid_argument, "attr '{}' must be a number or a list of 2", k);
}
inline i64 conv_out(i64 n, i64 k, i64 s, i64 p, i64 d) {
    if (s < 1 || d < 1 || p < 0 || k < 1) raise(errc::invalid_argument, "conv/pool: bad stride {}, dilation {}, pad {} or kernel {}", s, d, p, k);
    i64 e = n + 2 * p - d * (k - 1) - 1;
    if (e < 0) raise(errc::shape_mismatch, "conv/pool: kernel {} (dilation {}) exceeds padded input {}", k, d, n + 2 * p);
    return e / s + 1;
}
inline i64 conv_groups(const attrs& a, i64 cin, i64 cout) {
    i64 g = i64(a.num("groups", 1));
    if (g < 1 || cin % g || cout % g) raise(errc::shape_mismatch, "conv: groups {} does not divide channels {} / {}", g, cin, cout);
    return g;
}
inline void conv_bias_check(const shape_t& b, i64 o) {
    if (numel(b) != o && numel(b) != 1) raise(errc::shape_mismatch, "conv: bias {} for {} output channels", shape_str(b), o);
}
// pad attr: [before, after] pairs starting at the LAST axis (torch.nn.functional.pad order)
inline std::vector<std::array<i64, 2>> pad_pairs(const attrs& a, std::size_t rank) {
    auto v = a.ints("pads");
    if (v.size() % 2 || v.size() / 2 > rank) raise(errc::invalid_argument, "pad: pads needs [before, after] pairs for at most {} axes", rank);
    std::vector<std::array<i64, 2>> p(rank, std::array<i64, 2>{0, 0});
    for (std::size_t i = 0; i < v.size() / 2; ++i) {
        if (v[2 * i] < 0 || v[2 * i + 1] < 0) raise(errc::invalid_argument, "pad: negative padding");
        p[rank - 1 - i] = {v[2 * i], v[2 * i + 1]};
    }
    return p;
}
inline shape_t infer_shape_ext(shape_rule r, const std::vector<shape_t>& in, const attrs& a) {
    auto need = [&](std::size_t n) {
        if (in.size() < n) raise(errc::invalid_argument, "expected {} inputs, got {}", n, in.size());
    };
    auto rank = [&](std::size_t i, std::size_t n, const char* what) {
        if (in[i].size() != n) raise(errc::shape_mismatch, "{} must have rank {}, got {}", what, n, shape_str(in[i]));
    };
    switch (r) {
    case shape_rule::conv1d: {   // x[c, l], w[o, c/g, k], b[o] or [1]
        need(3); rank(0, 2, "conv1d input [c, l]"); rank(1, 3, "conv1d weight [o, c/g, k]");
        i64 g = conv_groups(a, in[0][0], in[1][0]);
        if (in[1][1] * g != in[0][0]) raise(errc::shape_mismatch, "conv1d: weight {} does not match {} input channels / {} groups", shape_str(in[1]), in[0][0], g);
        conv_bias_check(in[2], in[1][0]);
        return {in[1][0], conv_out(in[0][1], in[1][2], i64(a.num("stride", 1)), i64(a.num("pad", 0)), i64(a.num("dilation", 1)))};
    }
    case shape_rule::conv2d: {   // x[c, h, w], w[o, c/g, kh, kw], b[o] or [1]
        need(3); rank(0, 3, "conv2d input [c, h, w]"); rank(1, 4, "conv2d weight [o, c/g, kh, kw]");
        i64 g = conv_groups(a, in[0][0], in[1][0]);
        if (in[1][1] * g != in[0][0]) raise(errc::shape_mismatch, "conv2d: weight {} does not match {} input channels / {} groups", shape_str(in[1]), in[0][0], g);
        conv_bias_check(in[2], in[1][0]);
        auto s = attr2(a, "stride", 1), p = attr2(a, "pad", 0), d = attr2(a, "dilation", 1);
        return {in[1][0], conv_out(in[0][1], in[1][2], s[0], p[0], d[0]), conv_out(in[0][2], in[1][3], s[1], p[1], d[1])};
    }
    case shape_rule::conv_transpose2d: {   // x[c, h, w], w[c, o/g, kh, kw], b[o] or [1]
        need(3); rank(0, 3, "conv_transpose2d input [c, h, w]"); rank(1, 4, "conv_transpose2d weight [c, o/g, kh, kw]");
        i64 g = i64(a.num("groups", 1));
        if (g < 1 || in[0][0] % g || in[1][0] != in[0][0])
            raise(errc::shape_mismatch, "conv_transpose2d: weight {} does not match {} input channels / {} groups", shape_str(in[1]), in[0][0], g);
        i64 o = in[1][1] * g; conv_bias_check(in[2], o);
        auto s = attr2(a, "stride", 1), p = attr2(a, "pad", 0), d = attr2(a, "dilation", 1), op = attr2(a, "output_pad", 0);
        shape_t out{o, 0, 0};
        for (std::size_t i = 0; i < 2; ++i) {
            if (s[i] < 1 || d[i] < 1 || p[i] < 0 || op[i] < 0 || op[i] >= std::max(s[i], d[i]))
                raise(errc::invalid_argument, "conv_transpose2d: bad stride, dilation, pad or output_pad");
            out[i + 1] = (in[0][i + 1] - 1) * s[i] - 2 * p[i] + d[i] * (in[1][i + 2] - 1) + op[i] + 1;
            if (out[i + 1] < 1) raise(errc::shape_mismatch, "conv_transpose2d: empty output");
        }
        return out;
    }
    case shape_rule::pool2d: {   // x[c, h, w]; stride defaults to the kernel
        need(1); rank(0, 3, "pool2d input [c, h, w]");
        auto k = attr2(a, "kernel", 2), s = a.has("stride") ? attr2(a, "stride", 1) : k, p = attr2(a, "pad", 0);
        if (2 * p[0] > k[0] || 2 * p[1] > k[1]) raise(errc::invalid_argument, "pool2d: pad must be at most half the kernel");
        return {in[0][0], conv_out(in[0][1], k[0], s[0], p[0], 1), conv_out(in[0][2], k[1], s[1], p[1], 1)};
    }
    case shape_rule::adaptive_pool2d: {
        need(1); rank(0, 3, "adaptive pool input [c, h, w]");
        auto o = attr2(a, "out", 1);
        if (o[0] < 1 || o[1] < 1) raise(errc::invalid_argument, "adaptive pool: bad output size");
        return {in[0][0], o[0], o[1]};
    }
    case shape_rule::global_pool: {   // x[c, ...] -> [c]
        need(1); if (in[0].size() < 2) raise(errc::shape_mismatch, "global pool needs [c, ...]");
        return {in[0][0]};
    }
    case shape_rule::upsample2d: {
        need(1); rank(0, 3, "upsample input [c, h, w]");
        auto s = attr2(a, "scale", 2);
        if (s[0] < 1 || s[1] < 1) raise(errc::invalid_argument, "upsample: bad scale");
        return {in[0][0], in[0][1] * s[0], in[0][2] * s[1]};
    }
    case shape_rule::pad: {
        need(1); shape_t s = in[0]; auto p = pad_pairs(a, s.size());
        i64 mode = i64(a.num("mode", 0));
        if (mode < 0 || mode > 2) raise(errc::invalid_argument, "pad: mode must be 0 (constant), 1 (reflect) or 2 (replicate)");
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (mode == 1 && (p[i][0] >= s[i] || p[i][1] >= s[i])) raise(errc::invalid_argument, "pad: reflect padding must be smaller than the axis");
            if (mode == 2 && s[i] == 0 && (p[i][0] || p[i][1])) raise(errc::invalid_argument, "pad: cannot replicate an empty axis");
            s[i] += p[i][0] + p[i][1];
        }
        return s;
    }
    case shape_rule::alibi: {   // query positions [s], key positions [t] -> [heads, s, t]
        need(2); rank(0, 1, "alibi query positions [s]"); rank(1, 1, "alibi key positions [t]");
        i64 h = i64(a.num("heads", 1)); if (h < 1) raise(errc::invalid_argument, "alibi: heads must be positive");
        return {h, in[0][0], in[1][0]};
    }
    case shape_rule::embed_dim: {   // t[...] -> [..., dim]
        need(1); i64 d = i64(a.num("dim", 0));
        if (d < 2) raise(errc::invalid_argument, "sinusoid: dim must be at least 2");
        shape_t s = in[0]; s.push_back(d); return s;
    }
    case shape_rule::topk: {
        need(1); if (in[0].empty()) raise(errc::shape_mismatch, "topk needs rank >= 1");
        i64 k = i64(a.num("k", 1));
        if (k < 1 || k > in[0].back()) raise(errc::invalid_argument, "topk: k {} out of range for axis of {}", k, in[0].back());
        shape_t s = in[0]; s.back() = k; return s;
    }
    case shape_rule::index_select: {   // x, idx[m] -> axis replaced by m
        need(2); rank(1, 1, "index_select indices [m]");
        shape_t s = in[0]; i64 ax = norm_axis(i64(a.num("axis", 0)), s.size());
        s[std::size_t(ax)] = in[1][0]; return s;
    }
    case shape_rule::masked_mean: {   // x[s, ...], mask[s] -> [...]
        need(2); if (in[0].size() < 2 || numel(in[1]) != in[0][0]) raise(errc::shape_mismatch, "masked_mean: x [s, ...] and mask [s]");
        return shape_t(in[0].begin() + 1, in[0].end());
    }
    default: break;
    }
    raise(errc::unsupported, "unknown shape rule");
}

// Shape inference driven by the spec's shape rule.
inline shape_t infer_shape(shape_rule r, const std::vector<shape_t>& in, const attrs& a) {
    auto need = [&](std::size_t n) {
        if (in.size() < n) raise(errc::invalid_argument, "expected {} inputs, got {}", n, in.size());
    };
    switch (r) {
    case shape_rule::same: need(1); return in[0];
    case shape_rule::broadcast: return broadcast_shapes(in);
    case shape_rule::reduce: {
        need(1); shape_t s = in[0];
        i64 ax = norm_axis(i64(a.num("axis", -1)), s.size());
        s.erase(s.begin() + ax);
        return s;
    }
    case shape_rule::matmul: case shape_rule::matmul_t: {
        need(2); const auto& x = in[0]; const auto& w = in[1];
        if (x.size() < 2 || w.size() < 2) raise(errc::shape_mismatch, "matmul needs rank>=2");
        bool t = r == shape_rule::matmul_t;
        i64 k = x.back();
        i64 wk = t ? w[w.size() - 1] : w[w.size() - 2];
        i64 n = t ? w[w.size() - 2] : w[w.size() - 1];
        if (k != wk) raise(errc::shape_mismatch, "matmul inner dims differ: {} vs {}", shape_str(x), shape_str(w));
        if (w.size() > 2 && (w.size() != x.size() || !std::equal(w.begin(), w.end() - 2, x.begin())))
            raise(errc::shape_mismatch, "batched matmul batch dims differ");
        shape_t s = x; s.back() = n; return s;
    }
    case shape_rule::attention: {
        need(3); const auto& q = in[0]; const auto& k = in[1]; const auto& v = in[2];
        if (q.size() != 3 || k.size() != 3 || v.size() != 3) raise(errc::shape_mismatch, "attention expects [s,h,d]");
        if (k[0] != v[0] || k[1] != v[1]) raise(errc::shape_mismatch, "attention k/v shapes differ");
        if (q[2] != k[2]) raise(errc::shape_mismatch, "attention head dims differ");
        if (q[1] % k[1]) raise(errc::shape_mismatch, "q heads not multiple of kv heads");
        if (in.size() > 3) {   // attention_bias: additive bias, right-aligned broadcast against [hq, s, t]
            const auto& b = in[3]; const i64 full[3] = {q[1], q[0], k[0]};
            if (b.size() > 3) raise(errc::shape_mismatch, "attention bias must broadcast to [heads, s, t]");
            for (std::size_t i = 0; i < b.size(); ++i)
                if (b[b.size() - 1 - i] != 1 && b[b.size() - 1 - i] != full[2 - i])
                    raise(errc::shape_mismatch, "attention bias {} does not broadcast to [{}, {}, {}]", shape_str(b), full[0], full[1], full[2]);
        }
        return shape_t{q[0], q[1], v[2]};  // value head dim may differ (MLA)
    }
    case shape_rule::expand: {  // repeat a size-1 axis n times
        need(1); shape_t s = in[0]; i64 ax = norm_axis(i64(a.num("axis", 0)), s.size());
        if (s[std::size_t(ax)] != 1) raise(errc::shape_mismatch, "expand: axis {} has size {}, need 1", ax, s[std::size_t(ax)]);
        s[std::size_t(ax)] = i64(a.num("n", 1)); return s;
    }
    case shape_rule::transpose: {
        need(1); auto p = a.ints("perm"); shape_t s = in[0];
        if (p.empty()) { std::reverse(s.begin(), s.end()); return s; }
        if (p.size() != s.size()) raise(errc::shape_mismatch, "perm rank mismatch");
        shape_t o(s.size()); for (std::size_t i = 0; i < p.size(); ++i) o[i] = s[std::size_t(norm_axis(p[i], s.size()))];
        return o;
    }
    case shape_rule::reshape: {
        need(1); auto t = a.ints("shape"); i64 known = 1; int neg = -1;
        for (std::size_t i = 0; i < t.size(); ++i) { if (t[i] == -1) neg = int(i); else known *= t[i]; }
        i64 n = numel(in[0]);
        if (neg >= 0) { if (known == 0 || n % known) raise(errc::shape_mismatch, "bad reshape"); t[std::size_t(neg)] = n / known; }
        if (numel(t) != n) raise(errc::shape_mismatch, "reshape {} -> {} changes size", shape_str(in[0]), shape_str(t));
        return t;
    }
    case shape_rule::gather: {
        need(2); if (in[0].size() != 2) raise(errc::shape_mismatch, "gather table must be rank 2");
        shape_t s = in[1]; s.push_back(in[0][1]); return s;
    }
    case shape_rule::concat: {
        need(2); shape_t s = in[0]; i64 ax = norm_axis(i64(a.num("axis", 0)), s.size());
        for (std::size_t j = 1; j < in.size(); ++j) {
            if (in[j].size() != s.size()) raise(errc::shape_mismatch, "concat rank mismatch");
            for (std::size_t i = 0; i < s.size(); ++i)
                if (i64(i) != ax && in[j][i] != s[i]) raise(errc::shape_mismatch, "concat dim mismatch");
            if (j) s[std::size_t(ax)] += in[j][std::size_t(ax)];
        }
        return s;
    }
    case shape_rule::slice: {
        need(1); shape_t s = in[0]; i64 ax = norm_axis(i64(a.num("axis", 0)), s.size());
        i64 st = i64(a.num("start", 0)), en = i64(a.num("end", double(s[std::size_t(ax)])));
        if (st < 0) st += s[std::size_t(ax)];
        if (en < 0) en += s[std::size_t(ax)];
        en = std::min(en, s[std::size_t(ax)]);
        if (st < 0 || st > en) raise(errc::shape_mismatch, "bad slice");
        s[std::size_t(ax)] = en - st; return s;
    }
    case shape_rule::conv1d: case shape_rule::conv2d: case shape_rule::conv_transpose2d: case shape_rule::pool2d:
    case shape_rule::adaptive_pool2d: case shape_rule::global_pool: case shape_rule::upsample2d: case shape_rule::pad:
    case shape_rule::alibi: case shape_rule::embed_dim: case shape_rule::topk: case shape_rule::index_select:
    case shape_rule::masked_mean: return infer_shape_ext(r, in, a);
    }
    raise(errc::unsupported, "unknown shape rule");
}

} // namespace forge::ops
