#pragma once
// Fusions of host (CPU) plans, applied at the end of lower_plan(). Included by plan.hpp.
//
// Both work on the lowered step list and leave it as it is: every step keeps its lowering, inputs, output and
// buffer, and is only marked. So the device paths never see them (they are applied to plans without device hooks
// and without pipeline stages only: on a device every op lowers and runs exactly as before), a precompiled plan
// (aot.hpp) lists the same steps, natively compiled code runs the steps unfused, and turning a fusion off
// changes nothing but the order in which the same kernels are called.
//
// 1. Row chains (FORGE_FUSE_CHAIN=0 or pass_options::fuse_chain = false turns them off).
//    A run of adjacent steps in which every step computes row r of its output from row r of the previous
//    step's output and of its other tensor inputs, plus row vectors or scalars (a bias, a norm weight):
//    elementwise ops, bias adds, rms_norm, layer_norm. Unfused, each step is one pass over the whole tensor.
//    Fused, the tensor is cut into blocks of rows of about 64 KB and each block goes through all steps while it
//    is in cache: one pass over memory for the chain. This is what the patterns
//        matmul_t -> add(bias) -> gelu / silu / relu          (the epilogue: the matmul itself is not blocked)
//        add -> rms_norm            rms_norm / layer_norm -> mul / scale            exp -> add -> tanh -> ...
//    become. A block is run by calling each step's own lowering on row slices of its views, so the fused
//    chain does the same arithmetic as the unfused steps: bit-identical results (the kernels already have to
//    give a row the same bits whatever range of rows they are called on, because task::parallel_for hands them
//    ranges that depend on timing). An intermediate that has other consumers is still written in full.
//
// 2. Shared int8 activations (FORGE_FUSE_QACT=0 / pass_options::fuse_qact = false).
//    Quantised matmul_t on the integer path (qweights::gemm_nt_i8) first quantises its activation rows to int8.
//    When several such steps multiply the same activation (the q / k / v projections after one norm, the gate
//    and up projections of an FFN), the first of them to run quantises it into a buffer the plan owns and the
//    others reuse it: the norm's output is quantised once per run instead of once per product. Quantisation is
//    a deterministic function of the row, so the products are bit-identical to the unfused ones.
//
// Buffers. The memory plan was made for the unfused order. A chain interleaves its steps, so it is only formed
// when every pair of buffers it touches is either disjoint or the same buffer used row for row (an in-place
// step); a planner that put a later output over part of an earlier input would otherwise be read after being
// overwritten.
// (no includes: plan.hpp includes this file after its own declarations)

namespace forge::graph {

namespace fusion {
enum class row_kind { none, elementwise, row_local };
// Ops whose row r of the output depends on row r of the inputs only (last axis = row).
inline row_kind row_kind_of(ops::op o) {
    using enum ops::op;
    switch (o) {
        case add: case sub: case mul: case div: case maximum: case minimum: case pow: case silu_mul:
        case neg: case abs: case sqrt: case rsqrt: case recip: case square: case exp: case exp2: case expm1: case log: case log2: case log1p:
        case sin: case cos: case tan: case tanh: case sinh: case cosh: case asin: case acos: case atan: case erf: case erfc:
        case sigmoid: case silu: case gelu: case gelu_tanh: case relu: case softplus: case scale:
        case quick_gelu: case mish: case hard_swish: case floor:
            return row_kind::elementwise;
        case rms_norm: case layer_norm:
            return row_kind::row_local;
        default: return row_kind::none;
    }
}
struct buf { const float* p; std::size_t n; bool full; };
inline bool overlap(const buf& a, const buf& b) { return a.p < b.p + b.n && b.p < a.p + a.n; }
} // namespace fusion

inline void run_chain(plan& p, const cpu_chain& c) {
    task::parallel_for(c.units, c.block, c.block, [&](std::size_t b, std::size_t e) {
        std::vector<tensor_view> ins;
        // [rows, row length]: the last dimension stays what the step's lowering was chosen for (a row lowering
        // reads it); a chain without row operands is cut by elements, as rows of one
        const shape_t sub{i64(e - b), i64(c.unit)};
        for (std::size_t t = 0; t < c.count; ++t) {
            const step& s = p.steps[c.first + t];
            ins.clear();
            for (std::size_t j = 0; j < s.in.size(); ++j) {
                tensor_view v = p.views[s.in[j]];
                if ((c.full[t] >> j) & 1u) { v.data += b * c.unit; v.shape = sub; }
                ins.push_back(std::move(v));
            }
            tensor_view out = p.views[s.out];
            out.data += b * c.unit; out.shape = sub;
            s.low->fn(ins, out, *s.attrs);   // a nested parallel_for inside the kernel runs on this thread
        }
    });
}

// matmul_t over quantised rows on the integer path, with the activation's int8 form shared between the steps
// of one qact group.
inline void run_shared_qgemm(plan& p, const step& s) {
    const tensor_view& x = p.views[s.in[0]]; const tensor_view& w = p.views[s.in[1]]; const tensor_view& out = p.views[s.out];
    const std::size_t k = std::size_t(x.shape.back()), m = x.size() / k, n = std::size_t(w.shape[0]);
    const auto* c = qweights::codec_of(w);
    const int fmt = qweights::fmt_of(c);
    qact_slot& q = p.qacts[std::size_t(s.qact)];
    if (q.run != p.run_id || q.rows != m) {
        qweights::quantize_acts(fmt, x.data, m, k, q.act);
        q.run = p.run_id; q.rows = m;
    }
    qweights::gemm_nt_i8_acts(fmt, w.qdata, c->bytes(k), n, reinterpret_cast<const std::uint8_t*>(q.act.data()), m, k, out.data);
}

inline void fuse_cpu(plan& p, registry& reg, const pass_options& po) {
    p.chains.clear(); p.qacts.clear();
    for (auto& s : p.steps) { s.chain = 0; s.qact = -1; }
    if (p.hooks || !reg.stages.empty() || reg.hooks) return;   // host plans only
    auto& G = p.g;
    // ---- shared int8 activations ----
    if (fusion::on(po.fuse_qact, "FORGE_FUSE_QACT") && qweights::int8_enabled()) {
        std::map<std::tuple<value_id, bool, std::size_t>, std::vector<std::size_t>> groups;   // (activation, q8_0 layout, k) -> steps
        for (std::size_t i = 0; i < p.steps.size(); ++i) {
            const step& s = p.steps[i];
            if (s.kind != 0 || s.low != &qweights::gemm_nt_lowering() || s.in.size() != 2) continue;
            const tensor_view& x = p.views[s.in[0]]; const tensor_view& w = p.views[s.in[1]];
            if (!x.data || !w.qdata || x.shape.empty() || G.at(s.in[0]).kind == node_kind::state_append) continue;
            const int fmt = qweights::fmt_of(qweights::codec_of(w));
            const std::size_t k = std::size_t(x.shape.back());
            if (!simd::qi8::supported(fmt, k)) continue;
            groups[{s.in[0], simd::qfmt(fmt) == simd::qfmt::q8_0, k}].push_back(i);
        }
        for (auto& [key, steps] : groups) {
            if (steps.size() < 2) continue;
            qact_slot q; q.q8 = std::get<1>(key);
            p.qacts.push_back(std::move(q));
            for (auto i : steps) p.steps[i].qact = int(p.qacts.size() - 1);
        }
    }
    // ---- row chains ----
    if (!fusion::on(po.fuse_chain, "FORGE_FUSE_CHAIN")) return;
    auto row_step = [&](const step& s) {
        return s.kind == 0 && s.low && !s.low->on_device && s.qact < 0 && G.at(s.out).kind == node_kind::op &&
               fusion::row_kind_of(G.at(s.out).op) != fusion::row_kind::none;
    };
    for (std::size_t i = 0; i < p.steps.size(); ++i) {
        if (!row_step(p.steps[i])) continue;
        const tensor_view& o0 = p.views[p.steps[i].out];
        if (!o0.data || o0.shape.empty()) continue;
        const std::size_t total = o0.size(), n = std::size_t(o0.shape.back());
        if (total == 0 || n == 0 || total < po.chain_min_elems) continue;
        cpu_chain c; c.first = i;
        bool need_rows = false;
        std::vector<fusion::buf> outs, operands;
        auto try_add = [&](std::size_t si) -> bool {
            const step& s = p.steps[si];
            const tensor_view& out = p.views[s.out];
            if (!out.data || out.size() != total || out.shape.empty() || std::size_t(out.shape.back()) != n) return false;
            const auto kind = fusion::row_kind_of(G.at(s.out).op);
            std::uint32_t full = 0; bool rows = kind == fusion::row_kind::row_local, linked = si == c.first;
            std::vector<fusion::buf> ops_;
            if (s.in.empty() || s.in.size() > 30) return false;
            for (std::size_t j = 0; j < s.in.size(); ++j) {
                const tensor_view& v = p.views[s.in[j]];
                if (!v.data || v.shape.empty() || G.at(s.in[j]).kind == node_kind::state_append) return false;
                if (v.size() == total && std::size_t(v.shape.back()) == n) {   // a whole tensor, sliced by rows
                    full |= 1u << j;
                    if (si > c.first && s.in[j] == p.steps[si - 1].out) linked = true;
                    ops_.push_back({v.data, total, true});
                } else if (v.size() == 1) ops_.push_back({v.data, 1, false});
                else if (v.size() == n && std::size_t(v.shape.back()) == n) { rows = true; ops_.push_back({v.data, n, false}); }
                else return false;                                             // some other broadcast: not row-local
            }
            if (!linked) return false;
            if (kind == fusion::row_kind::row_local && !(full & 1u)) return false;   // the normalised operand is input 0
            // buffers: disjoint, or the same buffer row for row
            const fusion::buf ob{out.data, total, true};
            auto ok = [](const fusion::buf& w, const fusion::buf& x) { return !fusion::overlap(w, x) || (x.full && x.p == w.p); };
            for (auto& x : operands) if (!ok(ob, x)) return false;      // this output against earlier operands
            for (auto& x : ops_) if (!ok(ob, x)) return false;          // and its own
            for (auto& w : outs) {
                if (!ok(w, ob)) return false;                           // against earlier outputs
                for (auto& x : ops_) if (!ok(w, x)) return false;       // earlier outputs against these operands
            }
            outs.push_back(ob); operands.insert(operands.end(), ops_.begin(), ops_.end());
            c.full.push_back(full); need_rows = need_rows || rows;
            return true;
        };
        std::size_t j = i;
        while (j < p.steps.size() && row_step(p.steps[j]) && try_add(j)) ++j;
        c.count = j - i;
        if (c.count < 2) continue;
        c.unit = need_rows ? n : 1;
        c.units = total / c.unit;
        c.block = std::max<std::size_t>(1, 16384 / c.unit);
        p.chains.push_back(std::move(c));
        p.steps[i].chain = int(p.chains.size());
        i = j - 1;
    }
}

} // namespace forge::graph
