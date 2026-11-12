#pragma once
// verify -> canonicalise (CSE, constant folding, DCE, algebraic) -> fuse.
#include <forge/graph/ir.hpp>
#include <forge/graph/state_ops.hpp>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

namespace forge::graph {

// Shape/dtype inference from the spec rules; errors name the exact node.
inline void verify_graph(tensor_graph& g, const std::map<std::string, shape_t>& state_shapes = {}) {
    for (std::size_t i = 0; i < g.size(); ++i) {
        auto& n = g.nodes()[i];
        for (auto in : n.inputs)
            if (in >= i) raise(errc::invalid_argument, "node %{} uses %{} before definition", i, in);
        if (n.kind == node_kind::op) {
            const auto& inf = ops::info(n.op);
            if (int(n.inputs.size()) != inf.inputs)
                raise(errc::invalid_argument, "node %{} ({}): expected {} inputs, got {}", i, inf.name, inf.inputs, n.inputs.size());
            std::vector<shape_t> in;
            for (auto v : n.inputs) in.push_back(g.at(v).shape);
            try { n.shape = ops::infer_shape(inf.shape, in, n.attrs); }
            catch (const forge_exception& e) { raise(e.code, "node %{} ({}): {}", i, inf.name, e.what()); }
        } else if (n.kind == node_kind::state_append) {
            shape_t s = g.at(n.inputs[0]).shape;
            auto it = state_shapes.find(n.name);
            i64 past = it == state_shapes.end() ? 0 : it->second.at(0);
            s.at(0) += past;
            n.shape = s;
        } else if (n.kind == node_kind::state_op) {
            std::vector<shape_t> in; for (auto v : n.inputs) in.push_back(g.at(v).shape);
            try { n.shape = state_op_shape(n.attrs.str("fn"), in, n.attrs); }
            catch (const forge_exception& e) { raise(e.code, "node %{} (state_op {}): {}", i, n.attrs.str("fn"), e.what()); }
        }
    }
    for (auto& [nm, v] : g.outputs()) if (v >= g.size()) raise(errc::invalid_argument, "output {} undefined", nm);
}

// Rebuild graph keeping only nodes reachable from outputs (and state appends: side effects).
inline tensor_graph dce(const tensor_graph& g) {
    std::vector<char> live(g.size(), 0);
    std::vector<value_id> stack;
    for (auto& [n, v] : g.outputs()) stack.push_back(v);
    for (std::size_t i = 0; i < g.size(); ++i) if (g.at(value_id(i)).kind == node_kind::state_append || g.at(value_id(i)).kind == node_kind::state_op) stack.push_back(value_id(i));
    while (!stack.empty()) {
        auto v = stack.back(); stack.pop_back();
        if (live[v]) continue;
        live[v] = 1;
        for (auto in : g.at(v).inputs) stack.push_back(in);
    }
    tensor_graph o; std::vector<value_id> remap(g.size(), no_value);
    for (std::size_t i = 0; i < g.size(); ++i) {
        if (!live[i]) continue;
        node n = g.at(value_id(i));
        for (auto& in : n.inputs) in = remap[in];
        remap[i] = value_id(o.nodes().size());
        o.nodes().push_back(std::move(n));
    }
    for (auto& [nm, v] : g.outputs()) o.output(nm, remap[v]);
    return o;
}

// Common-subexpression elimination over ops with identical (op, inputs, attrs).
inline tensor_graph cse(const tensor_graph& g) {
    tensor_graph o; std::vector<value_id> remap(g.size());
    std::map<std::string, value_id> seen;
    for (std::size_t i = 0; i < g.size(); ++i) {
        node n = g.at(value_id(i));
        for (auto& in : n.inputs) in = remap[in];
        if (n.kind == node_kind::op) {
            std::string key = std::to_string(int(n.op));
            for (auto in : n.inputs) key += "," + std::to_string(in);
            for (auto& [k, a] : n.attrs.v) {
                key += ";" + k + "=";
                if (auto* d = std::get_if<double>(&a)) key += std::to_string(*d);
                else if (auto* s = std::get_if<std::string>(&a)) key += *s;
                else for (auto x : std::get<std::vector<i64>>(a)) key += std::to_string(x) + ".";
            }
            if (auto it = seen.find(key); it != seen.end()) { remap[i] = it->second; continue; }
            remap[i] = value_id(o.nodes().size());
            seen[key] = remap[i];
        } else remap[i] = value_id(o.nodes().size());
        o.nodes().push_back(std::move(n));
    }
    for (auto& [nm, v] : g.outputs()) o.output(nm, remap[v]);
    return o;
}

// Constant folding: ops whose inputs are all constants are evaluated with the reference.
inline tensor_graph fold_constants(tensor_graph g) {
    for (auto& n : g.nodes()) {
        if (n.kind != node_kind::op) continue;
        bool all_const = !n.inputs.empty();
        for (auto in : n.inputs) all_const &= g.at(in).kind == node_kind::constant;
        if (!all_const) continue;
        std::vector<tensor> ins;
        for (auto in : n.inputs) ins.push_back(*g.at(in).constant);
        auto t = ops::reference(n.op)(ins, n.attrs);
        n.kind = node_kind::constant; n.shape = t.shape;
        n.constant = std::make_shared<tensor>(std::move(t)); n.inputs.clear();
    }
    return g;
}

// Algebraic simplification: x*1, x+0, reshape(reshape(x)) etc.
inline tensor_graph simplify(tensor_graph g) {
    auto is_scalar_const = [&](value_id v, float val) {
        const auto& n = g.at(v);
        return n.kind == node_kind::constant && n.constant->size() == 1 && n.constant->data[0] == val;
    };
    std::vector<value_id> alias(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) alias[i] = value_id(i);
    for (std::size_t i = 0; i < g.size(); ++i) {
        auto& n = g.nodes()[i];
        for (auto& in : n.inputs) in = alias[in];
        if (n.kind != node_kind::op) continue;
        auto same_shape = [&](value_id v) { return g.at(v).shape == n.shape; };
        if ((n.op == ops::op::mul && is_scalar_const(n.inputs[1], 1.f) && same_shape(n.inputs[0])) ||
            (n.op == ops::op::add && is_scalar_const(n.inputs[1], 0.f) && same_shape(n.inputs[0])))
            alias[i] = n.inputs[0];
        else if (n.op == ops::op::reshape && g.at(n.inputs[0]).kind == node_kind::op && g.at(n.inputs[0]).op == ops::op::reshape)
            n.inputs[0] = g.at(n.inputs[0]).inputs[0];
        else if (n.op == ops::op::scale && n.attrs.num("alpha", 1) == 1.0)
            alias[i] = n.inputs[0];
    }
    for (auto& [nm, v] : g.outputs()) v = alias[v];
    return g;
}

// Fusion: silu(a) * b -> silu_mul(a, b) when silu(a) has no other users.
inline tensor_graph fuse(tensor_graph g) {
    std::vector<int> uses(g.size(), 0);
    for (auto& n : g.nodes()) for (auto in : n.inputs) ++uses[in];
    for (auto& [nm, v] : g.outputs()) ++uses[v];
    for (auto& n : g.nodes()) {
        if (n.kind != node_kind::op || n.op != ops::op::mul) continue;
        for (int side = 0; side < 2; ++side) {
            auto a = n.inputs[std::size_t(side)], b = n.inputs[std::size_t(1 - side)];
            const auto& an = g.at(a);
            if (an.kind == node_kind::op && an.op == ops::op::silu && uses[a] == 1 && an.shape == n.shape && g.at(b).shape == n.shape) {
                n.op = ops::op::silu_mul; n.inputs = {an.inputs[0], b};
                break;
            }
        }
    }
    return g;
}

// batch_norm(conv(x, W, b), w, bb, mean, var) with constant f32 parameters -> conv(x, W', b') with
//     s[o] = w[o] / sqrt(var[o] + eps),   W'[o, ...] = W[o, ...] * s[o],   b'[o] = (b[o] - mean[o]) * s[o] + bb[o]
// (computed in double, rounded once to float). conv2d and conv1d; the convolution's result and its W and b must
// have no other users. The folded graph is not bit-identical to the unfused one: each product x * W' rounds
// differently from (x * W) * s. The difference stays within the convolution's own tolerance (rel 1e-4 of the
// reference); tests/test_graph.cpp (fusion_conv_bn_folds) bounds it at 2e-5 of the largest output.
// FORGE_FUSE_CONV_BN=0 (or pass_options::fold_conv_bn = false) keeps the batch_norm.
inline tensor_graph fold_conv_bn(tensor_graph g) {
    std::vector<int> uses(g.size(), 0);
    for (auto& n : g.nodes()) for (auto in : n.inputs) ++uses[in];
    for (auto& [nm, v] : g.outputs()) ++uses[v];
    auto data_of = [&](value_id v) -> const float* {
        const auto& n = g.at(v);
        if (n.kind == node_kind::constant) return n.constant->data.data();
        if (n.kind == node_kind::weight && n.weight && !n.qweight) return n.weight;
        return nullptr;
    };
    std::vector<value_id> alias(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) alias[i] = value_id(i);
    for (std::size_t i = 0; i < g.size(); ++i) {
        auto& n = g.nodes()[i];
        for (auto& in : n.inputs) in = alias[in];
        if (n.kind != node_kind::op || n.op != ops::op::batch_norm || n.inputs.size() != 5) continue;
        const value_id cv = n.inputs[0];
        auto& c = g.at(cv);
        if (c.kind != node_kind::op || (c.op != ops::op::conv2d && c.op != ops::op::conv1d) || c.inputs.size() != 3 || uses[cv] != 1) continue;
        const value_id wv = c.inputs[1], bv = c.inputs[2];
        if (wv == bv || uses[wv] != 1 || uses[bv] != 1) continue;
        const float* W = data_of(wv); const float* B = data_of(bv);
        const float* gw = data_of(n.inputs[1]); const float* gb = data_of(n.inputs[2]); const float* mu = data_of(n.inputs[3]); const float* var = data_of(n.inputs[4]);
        if (!W || !B || !gw || !gb || !mu || !var) continue;
        const shape_t ws = g.at(wv).shape;
        if (ws.empty() || ws[0] <= 0) continue;
        const std::size_t O = std::size_t(ws[0]), per = std::size_t(numel(ws)) / O, bn = std::size_t(numel(g.at(bv).shape));
        bool sizes = bn == 1 || bn == O;
        for (std::size_t j = 1; j < 5; ++j) sizes = sizes && std::size_t(numel(g.at(n.inputs[j]).shape)) == O;
        if (!sizes) continue;
        const double eps = n.attrs.num("eps", 1e-5);
        tensor w2(ws), b2(shape_t{i64(O)});
        for (std::size_t o = 0; o < O; ++o) {
            const double sc = double(gw[o]) / std::sqrt(double(var[o]) + eps);
            for (std::size_t k = 0; k < per; ++k) w2.data[o * per + k] = float(double(W[o * per + k]) * sc);
            b2.data[o] = float((double(B[bn == 1 ? 0 : o]) - double(mu[o])) * sc + double(gb[o]));
        }
        auto as_const = [&](value_id v, tensor t) {
            auto& m = g.at(v);
            m.kind = node_kind::constant; m.shape = t.shape; m.weight = nullptr; m.qweight = nullptr; m.qcodec.clear(); m.inputs.clear();
            m.constant = std::make_shared<tensor>(std::move(t));
        };
        as_const(wv, std::move(w2)); as_const(bv, std::move(b2));
        alias[i] = cv;   // the batch_norm is now the convolution itself
    }
    for (auto& [nm, v] : g.outputs()) v = alias[v];
    return g;
}

class registry;
// fuse: silu(a) * b -> silu_mul. fold_conv_bn: batch_norm of a conv2d / conv1d with constant parameters folded
// into the convolution's weights (FORGE_FUSE_CONV_BN=0 off). fuse_chain / fuse_qact: the host-plan fusions of
// fuse_cpu.hpp (FORGE_FUSE_CHAIN=0 / FORGE_FUSE_QACT=0 off); chain_min_elems is the smallest tensor a row chain
// is formed for. FORGE_FUSE=0 turns off everything but silu_mul.
struct pass_options {
    bool cse = true, fold = true, simplify = true, fuse = true;
    registry* lowerings = nullptr; /* null = CPU */
    bool fold_conv_bn = true, fuse_chain = true, fuse_qact = true;
    std::size_t chain_min_elems = 4096;
};
namespace fusion {
inline bool env_on(const char* name) { const char* e = std::getenv(name); return !(e && e[0] == '0' && e[1] == 0); }
inline bool on(bool option, const char* name) { return option && env_on("FORGE_FUSE") && env_on(name); }
}

inline tensor_graph optimise(tensor_graph g, const pass_options& o = {}, const std::map<std::string, shape_t>& st = {}) {
    verify_graph(g, st);
    if (o.fold) g = fold_constants(std::move(g));
    if (o.simplify) g = simplify(std::move(g));
    if (o.cse) g = cse(g);
    if (fusion::on(o.fold_conv_bn, "FORGE_FUSE_CONV_BN")) g = fold_conv_bn(std::move(g));
    if (o.fuse) g = fuse(std::move(g));
    g = dce(g);
    verify_graph(g, st);
    return g;
}

} // namespace forge::graph
