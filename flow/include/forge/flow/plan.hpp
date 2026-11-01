#pragma once
// partition -> memory-plan -> lower -> flat plan -> replay. Plans are cached per shape bucket.
#include <map>
#include <cstdio>
#include <algorithm>
#include <forge/flow/codecs.hpp>
#include <forge/flow/kernels.hpp>
#include <forge/lane/qdot.hpp>
#include <forge/flow/kv_codec.hpp>
#include <forge/flow/memplan.hpp>
#include <forge/flow/passes.hpp>
#include <forge/grid/grid.hpp>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <set>

namespace forge::flow {

// Per-session state store: slot -> preallocated [capacity, ...] tensor + current length.
struct state_store {
    // Fixed slots (state_op: recurrent state, conv window) keep a constant-size buffer; `len` counts the rows
    // consumed so far, and they cannot be rewound except to 0.
    struct slot { tensor buf; i64 len = 0; i64 capacity = 0; kv_codec codec = kv_codec::f32; bool fixed = false; };
    std::map<std::string, slot> slots;
    i64 capacity = 4096;   // maximum rows per slot (context length)
    // Paged allocation: when > 0, slot storage grows in pages of `page_rows` rows on demand instead of
    // reserving `capacity` rows up front, so memory tracks the tokens actually held.
    i64 page_rows = 0;
    // KV codec for new slots (rank-3 [rows, heads, d] slots only; others stay f32). Set before the first
    // plan is compiled against this store. Quantised slots hold [rows, heads, kv_head_words(d)] floats.
    kv_codec codec = kv_codec::f32;
    // Value-slot codec for mixed modes (k8v4: int8 keys, tbq4 values); f32 here means "same as codec".
    kv_codec codec_v = kv_codec::f32;
    // Host offload: drop the device copy of every slot right after the attention that reads it, so device
    // memory holds at most one layer's KV at a time (the host copy is authoritative between steps).
    bool offload = false;
    // physical row shape for a slot fed by logical rows `xs` ([rows, heads, d])
    kv_codec codec_for(const shape_t& xs, bool value = false) const {
        if (xs.size() != 3) return kv_codec::f32;
        return value && codec_v != kv_codec::f32 ? codec_v : codec;
    }
    static shape_t physical(const shape_t& xs, kv_codec c) {
        if (c == kv_codec::f32) return xs;
        return {xs[0], xs[1], kv_head_words(xs[2], c)};
    }
    i64 reserved_rows() const { i64 r = 0; for (auto& [k, s] : slots) r += s.capacity; return r; }
    std::size_t bytes() const { std::size_t b = 0; for (auto& [k, s] : slots) b += s.buf.data.size() * sizeof(float); return b; }
    // make room for `rows` rows in `sl` (row shape taken from `row_shape`)
    void ensure(slot& sl, const shape_t& row_shape, i64 rows) {
        if (rows > capacity) raise(errc::out_of_memory, "state exceeds context capacity {}", capacity);
        if (rows <= sl.capacity) return;
        i64 want = page_rows > 0 ? std::min(capacity, (rows + page_rows - 1) / page_rows * page_rows) : capacity;
        shape_t cap = row_shape; cap[0] = want;
        tensor nb(cap);
        if (sl.capacity) std::copy_n(sl.buf.data.data(), std::size_t(numel(sl.buf.shape) / std::max<i64>(sl.capacity, 1) * sl.len), nb.data.data());
        sl.buf = std::move(nb); sl.capacity = want;
    }
    std::map<std::string, shape_t> shapes() const {
        std::map<std::string, shape_t> r;
        for (auto& [k, s] : slots) { shape_t sh = s.buf.shape; sh[0] = s.len; r[k] = sh; }
        return r;
    }
    void reset() { for (auto& [k, s] : slots) s.len = 0; }
    // rewind all slots to `len` rows (e.g. prefix reuse / speculative rollback)
    void truncate(i64 len) {
        for (auto& [k, s] : slots) {
            if (s.fixed && len > 0 && len < s.len)
                raise(errc::unsupported, "state slot '{}' is recurrent and cannot be rewound to {} (has {})", k, len, s.len);
            s.len = std::min(s.len, len);
        }
    }
};

struct partition_report {
    struct dev { std::string name; double flops = 0; double bytes = 0; int nodes = 0; };
    std::vector<dev> devices;
};

inline double node_flops(const graph& g, const node& n) {
    if (n.kind != node_kind::op) return 0;
    const auto& inf = spec::info(n.op);
    double out = double(numel(n.shape));
    if (inf.cls == spec::op_class::contraction && !n.inputs.empty()) {
        if (n.op == spec::op::attention) return 4.0 * out * double(g.at(n.inputs[1]).shape[0]);
        return 2.0 * out * double(g.at(n.inputs[0]).shape.back());
    }
    return out * inf.cost;
}

// Capability + cost partitioner. With one device all nodes land there; with several, contiguous
// layer ranges are balanced by flops (pipeline split). Nodes whose op has no lowering on a device
// are never assigned to it.
inline partition_report partition(graph& g, const std::vector<grid::device>& devs) {
    partition_report r;
    for (auto& d : devs) r.devices.push_back({d.caps().name});
    double total = 0; for (auto& n : g.nodes()) total += node_flops(g, n);
    double acc = 0;
    for (auto& n : g.nodes()) {
        double f = node_flops(g, n);
        int d = devs.size() <= 1 ? 0 : std::min(int(devs.size()) - 1, int(acc / std::max(total, 1.0) * double(devs.size())));
        n.device = d; acc += f;
        r.devices[std::size_t(d)].flops += f;
        r.devices[std::size_t(d)].bytes += double(numel(n.shape)) * 4;
        r.devices[std::size_t(d)].nodes++;
    }
    return r;
}

// Quantised weights: matmul_t reads encoded rows in place (decode one row into a per-thread buffer, then
// SIMD dot products against every input row), so a q4_k model costs its file size in memory, not 4 bytes/param.
namespace qweights {
using codecs::codec;
inline const codec* codec_of(const tensor_view& v) { return static_cast<const codec*>(v.qcodec); }
inline int fmt_of(const codec* c) {
    using lane::qfmt;
    return int(c->name == "q8_0" ? qfmt::q8_0 : c->name == "q4_k" ? qfmt::q4_k : c->name == "q6_k" ? qfmt::q6_k
             : c->name == "q5_k" ? qfmt::q5_k : qfmt::generic);
}
// Few input rows (decode): fused tile decode + dot per (row, weight row). Many rows (prefill): decode each weight
// row once (SIMD decoders) and reuse it. Both are bit-identical to dot(x, codec.decode(row)).
inline void gemm_nt(std::span<const tensor_view> in, tensor_view out, const spec::attrs&) {
    const auto& x = in[0]; const auto& w = in[1];
    std::size_t k = std::size_t(x.shape.back()), m = x.size() / k, n = std::size_t(w.shape[0]);
    const auto* c = codec_of(w); std::size_t rb = c->bytes(k);
    // FORGE_QDOT=0: plain scalar codec decode + dot (the pre-SIMD path; benchmarking and bisecting)
    static const bool native = [] { auto* e = std::getenv("FORGE_QDOT"); return !(e && std::string(e) == "0"); }();
    int fmt = fmt_of(c);
    bool fused = native && m <= 2 && c->block <= 256 && 256 % c->block == 0;
    task::parallel_for(n, 8, [&](std::size_t b, std::size_t e) {
        const auto& ker = lane::active();
        if (fused) {
            for (std::size_t j = b; j < e; ++j)
                for (std::size_t i = 0; i < m; ++i)
                    out.data[i * n + j] = ker.qdot(fmt, w.qdata + j * rb, x.data + i * k, k, c->decode, c->block, c->block_bytes);
            return;
        }
        thread_local std::vector<float> row;
        if (row.size() < k) row.resize(k);
        for (std::size_t j = b; j < e; ++j) {
            if (native) ker.qdecode(fmt, w.qdata + j * rb, k, row.data(), c->decode); else c->decode(w.qdata + j * rb, k, row.data());
            for (std::size_t i = 0; i < m; ++i) out.data[i * n + j] = ker.dot(x.data + i * k, row.data(), k);
        }
    });
}
// gather (embedding lookup) straight from encoded rows: a q8_0/q4_k tok_emb never gets an f32 copy.
inline void gather_rows(std::span<const tensor_view> in, tensor_view out, const spec::attrs&) {
    const auto& w = in[0]; const auto* c = codec_of(w);
    std::size_t d = std::size_t(w.shape[1]), rowsn = std::size_t(w.shape[0]), rb = c->bytes(d);
    for (std::size_t i = 0; i < in[1].size(); ++i) {
        auto r = std::size_t(in[1].data[i]);
        if (r >= rowsn) raise(errc::invalid_argument, "gather index {} out of range", r);
        c->decode(w.qdata + r * rb, d, out.data + i * d);
    }
}
inline const lowering& gather_lowering() {
    static const lowering l{"cpu.qgather", spec::op::gather, 100, [](auto, auto&, auto&) { return true; }, gather_rows, 1};
    return l;
}
inline const lowering& gemm_nt_lowering() {
    static const lowering l{"cpu.qgemm_nt", spec::op::matmul_t, 100, [](auto, auto&, auto&) { return true; }, gemm_nt, 1};
    return l;
}
// f32 copies for quantised weights that something other than matmul_t consumes, shared across plans.
inline std::shared_ptr<tensor> decoded(const std::uint8_t* p, const codec* c, const shape_t& s) {
    static std::mutex mu; static std::map<const std::uint8_t*, std::weak_ptr<tensor>> cache;
    std::lock_guard lk(mu);
    if (auto sp = cache[p].lock()) return sp;
    auto t = std::make_shared<tensor>(s);
    c->decode(p, t->size(), t->data.data());
    cache[p] = t;
    return t;
}
} // namespace qweights

struct step {
    const lowering* low;
    std::vector<value_id> in;
    value_id out;
    const spec::attrs* attrs;
    int kind; // 0 op, 1 state_append, 2 state_op
};

class plan {
public:
    graph g;
    memory_plan mem;
    std::vector<step> steps;
    std::deque<spec::attrs> attr_store;
    std::vector<std::shared_ptr<tensor>> decoded_weights;   // f32 fallbacks for quantised weights   // per-step attribute overrides (stable addresses)
    std::vector<tensor_view> views;   // per value
    std::vector<std::pair<value_id, value_id>> state_aliases;  // (view, state node) pairs rebound at run time
    std::vector<float> arena;
    std::map<std::string, value_id> inputs;
    partition_report part;
    std::vector<int> stage_of_node;    // pipeline stage per node (empty = single device)
    device_hooks* hooks = nullptr;     // device residency protocol (null = everything on the host)
    bool pinned = false;
    bool deterministic = true;        // CPU kernels use fixed reduction orders
    std::string describe() const {
        std::string s = std::format("plan: {} nodes, {} steps, arena {} bytes (naive {}), {} in-place, {} aliases\n",
                                    g.size(), steps.size(), mem.arena_bytes, mem.naive_bytes, mem.in_place, mem.aliases);
        std::map<std::string, int> uses;
        for (auto& st : steps) if (st.low) uses[st.low->name]++;
        for (auto& [k, v] : uses) s += std::format("  {:<16} x{}\n", k, v);
        return s;
    }
};

inline std::unique_ptr<plan> compile(graph g, state_store* st = nullptr, const pass_options& po = {}) {
    auto p = std::make_unique<plan>();
    auto sshapes = st ? st->shapes() : std::map<std::string, shape_t>{};
    p->g = optimise(std::move(g), po, sshapes);
    p->part = partition(p->g, grid::all());
    p->mem = plan_memory(p->g);
    if (!plan_is_safe(p->mem)) raise(errc::invalid_argument, "memory planner produced overlapping buffers");
    p->arena.assign(p->mem.arena_bytes / sizeof(float) + 1, 0.f);
    auto& G = p->g;
    p->views.resize(G.size());
    auto& reg = po.lowerings ? *po.lowerings : registry::cpu();
    p->hooks = reg.stages.empty() ? reg.hooks.get() : nullptr;
    // Pipeline split: contiguous node ranges per stage, balanced by spec cost x output size.
    std::vector<registry*> stage_of(G.size(), &reg);
    if (!reg.stages.empty()) {
        std::vector<double> c(G.size(), 0.0); double tot = 0;
        for (std::size_t i = 0; i < G.size(); ++i) { auto& n = G.nodes()[i]; if (n.kind != node_kind::op) continue;
            double e = double(std::max<i64>(numel(n.shape), 1)); for (auto in : n.inputs) e = std::max(e, double(numel(G.at(in).shape)));
            c[i] = std::max(spec::info(n.op).cost, 1.0) * e; tot += c[i]; }
        auto k = reg.stages.size(); std::vector<double> w(reg.stage_weight);
        w.resize(k, 1.0); double ws = 0; for (auto x : w) ws += x;
        double acc = 0, edge = w[0] / ws * tot; std::size_t st = 0;
        for (std::size_t i = 0; i < G.size(); ++i) {
            while (st + 1 < k && acc >= edge) { ++st; edge += w[st] / ws * tot; }
            auto* r = reg.stages[st].get(); stage_of[i] = r ? r : &registry::cpu(); acc += c[i];
        }
        p->stage_of_node.resize(G.size()); for (std::size_t i = 0; i < G.size(); ++i) {
            std::size_t s = 0; for (; s < k; ++s) if ((reg.stages[s] ? reg.stages[s].get() : &registry::cpu()) == stage_of[i]) break;
            p->stage_of_node[i] = int(s); }
    }
    std::vector<bool> q_ok(G.size(), true);   // every consumer is matmul_t (weight operand) or gather (table)
    for (auto& n : G.nodes()) for (std::size_t j = 0; j < n.inputs.size(); ++j)
        if (!(n.kind == node_kind::op && ((n.op == spec::op::matmul_t && j == 1) || (n.op == spec::op::gather && j == 0))))
            q_ok[n.inputs[j]] = false;
    for (auto& [nm, v] : G.outputs()) q_ok[v] = false;
    bool q_direct = !p->hooks && reg.stages.empty();
    // Resident device backends have no quantised kernels, so a weight they consume is decoded to an f32 host copy
    // and mirrored on the device (4 bytes/param on both sides: a 9B q4_k model needed ~36 GB of RAM). Decode only up
    // to FORGE_GPU_WEIGHT_BUDGET bytes of f32 (default 8G; accepts K/M/G suffixes); the remaining matmul_t/gather
    // weights stay encoded in the mapped file and run on the CPU quantised kernels. Backends with a device quantised
    // matmul (registry::device_qgemm) keep those codecs encoded and run them on the device instead.
    bool q_spill = p->hooks && reg.stages.empty();
    std::size_t q_budget = std::size_t(8) << 30, q_used = 0;
    auto dev_q = [&](const std::string& codec) {
        return p->hooks && reg.device_qgemm && std::find(reg.device_qgemm_codecs.begin(), reg.device_qgemm_codecs.end(), codec) != reg.device_qgemm_codecs.end(); };
    if (auto* e = std::getenv("FORGE_GPU_WEIGHT_BUDGET"); e && *e) {
        char* end = nullptr; double v = std::strtod(e, &end);
        double mul = 1; if (end) switch (*end) { case 'k': case 'K': mul = 1024.0; break; case 'm': case 'M': mul = 1024.0 * 1024; break;
                                                  case 'g': case 'G': mul = 1024.0 * 1024 * 1024; break; default: break; }
        q_budget = v > 0 ? std::size_t(v * mul) : 0;
    }
    // state slots read as attention values (input 2): they take the store's value codec in mixed modes (k8v4)
    std::set<value_id> value_slots;
    for (std::size_t i = 0; i < G.size(); ++i) {
        auto& n = G.nodes()[i];
        if (n.kind == node_kind::op && n.op == spec::op::attention && n.inputs.size() == 3 && G.at(n.inputs[2]).kind == node_kind::state_append)
            value_slots.insert(n.inputs[2]);
    }
    for (std::size_t i = 0; i < G.size(); ++i) {
        auto& n = G.nodes()[i];
        auto& s = p->mem.slots[i];
        tensor_view v; v.shape = n.shape;
        if (n.kind == node_kind::weight && n.qweight) {
            auto* c = codecs::find(n.qcodec);
            if (!c) raise(errc::unsupported, "weight {}: unknown codec {}", n.name, n.qcodec);
            bool direct_ok = q_ok[i] && n.shape.size() == 2 && std::size_t(n.shape[1]) % c->block == 0;
            std::size_t f32_bytes = std::size_t(numel(n.shape)) * sizeof(float);
            bool keep_q = direct_ok && (q_direct || dev_q(n.qcodec) || (q_spill && q_used + f32_bytes > q_budget));
            if (direct_ok && q_spill && !keep_q) q_used += f32_bytes;
            if (keep_q) { v.qdata = n.qweight; v.qcodec = c; }
            else { auto t = qweights::decoded(n.qweight, c, n.shape); v.data = t->data.data(); p->decoded_weights.push_back(std::move(t)); }
        }
        else if (n.kind == node_kind::weight) v.data = const_cast<float*>(n.weight);
        else if (n.kind == node_kind::constant) v.data = n.constant->data.data();
        else if (n.kind == node_kind::state_append) {
            if (!st) raise(errc::invalid_argument, "graph uses state slot '{}' but no state store given", n.name);
            auto& sl = st->slots[n.name];
            const auto& xs = G.at(n.inputs[0]).shape;
            if (n.shape[0] > st->capacity) raise(errc::out_of_memory, "state slot '{}' exceeds capacity {}", n.name, st->capacity);
            if (!sl.capacity) sl.codec = st->codec_for(xs, value_slots.count(value_id(i)) != 0);
            auto ps = state_store::physical(xs, sl.codec);
            st->ensure(sl, ps, std::max<i64>(n.shape[0], 1));
            v.shape = state_store::physical(n.shape, sl.codec);
            v.data = sl.buf.data.data();   // rebound on every run (pages may move, plans may be shared)
            p->steps.push_back({nullptr, n.inputs, value_id(i), &n.attrs, 1});
        } else {
            value_id r = value_id(i);
            while (p->mem.slots[r].kind == buffer_slot::alias) r = p->mem.slots[r].root;
            const auto& rs = p->mem.slots[r];
            if (rs.kind == buffer_slot::external) {
                v.data = p->views[r].data;
                if (G.at(r).kind == node_kind::state_append) p->state_aliases.push_back({value_id(i), r});
            }
            else v.data = p->arena.data() + rs.offset / sizeof(float);
        }
        (void)s;
        p->views[i] = v;
        if (n.kind == node_kind::input) p->inputs[n.name] = value_id(i);
        if (n.kind == node_kind::state_op) {
            if (!st) raise(errc::invalid_argument, "graph uses state slot '{}' but no state store given", n.name);
            for (auto in : n.inputs)
                if (G.at(in).kind == node_kind::state_append)
                    raise(errc::unsupported, "state slot '{}' feeds state_op '{}'", G.at(in).name, n.name);
            p->steps.push_back({nullptr, n.inputs, value_id(i), &n.attrs, 2});
        }
        if (n.kind == node_kind::op) {
            // state views grow at run time; only consumers that read the live length are allowed
            for (auto in : n.inputs)
                if (G.at(in).kind == node_kind::state_append && n.op != spec::op::attention)
                    raise(errc::unsupported, "state slot '{}' feeds '{}': only attention may consume a growing state view",
                          G.at(in).name, spec::info(n.op).name);
            std::vector<shape_t> ins; for (auto in : n.inputs) ins.push_back(G.at(in).shape);
            const spec::attrs* at = &n.attrs;
            if (n.op == spec::op::attention && st && n.inputs.size() == 3) {
                auto codec_of = [&](value_id in) {
                    auto& m = G.at(in);
                    return m.kind == node_kind::state_append ? st->slots[m.name].codec : kv_codec::f32;
                };
                auto kc = codec_of(n.inputs[1]), vc = codec_of(n.inputs[2]);
                if (kc != kv_codec::f32 || vc != kv_codec::f32) {
                    auto& a2 = p->attr_store.emplace_back(n.attrs);
                    a2.v["kv_codec_k"] = double(int(kc)); a2.v["kv_codec_v"] = double(int(vc));
                    a2.v["kv_dv"] = double(G.at(n.inputs[2]).shape[2]);
                    at = &a2;
                }
            }
            if (n.op == spec::op::matmul_t && n.inputs.size() == 2 && p->views[n.inputs[1]].qdata)
                p->steps.push_back({dev_q(G.at(n.inputs[1]).qcodec) ? &*reg.device_qgemm : &qweights::gemm_nt_lowering(), n.inputs, value_id(i), at, 0});
            else if (n.op == spec::op::gather && n.inputs.size() == 2 && p->views[n.inputs[0]].qdata)
                p->steps.push_back({&qweights::gather_lowering(), n.inputs, value_id(i), at, 0});
            else p->steps.push_back({&stage_of[i]->select(n.op, ins, n.shape, n.attrs), n.inputs, value_id(i), at, 0});
        }
    }
    return p;
}

class executor {
public:
    explicit executor(plan& p, state_store* st = nullptr) : p_(p), st_(st) {}
    void set_input(const std::string& name, std::span<const float> data) {
        auto it = p_.inputs.find(name);
        if (it == p_.inputs.end()) return; // input eliminated by DCE
        auto& v = p_.views[it->second];
        if (data.size() != v.size()) raise(errc::shape_mismatch, "input {} expects {} elements, got {}", name, v.size(), data.size());
        auto* h = p_.hooks;
        if (h) h->to_host(v.data, v.size());
        std::copy(data.begin(), data.end(), v.data);
        if (h) h->host_wrote(v.data, v.size());
    }
    void run() {
        auto* h = p_.hooks;
        if (!h) return run_steps(nullptr);
        if (!p_.pinned) {   // weights and constants never change: they stay on the device across runs
            for (std::size_t i = 0; i < p_.g.size(); ++i) {
                auto k = p_.g.nodes()[i].kind;
                if ((k == node_kind::weight || k == node_kind::constant) && p_.views[i].data) h->pin(p_.views[i].data, p_.views[i].size());
            }
            p_.pinned = true;
        }
        h->begin_run();
        struct guard { device_hooks* h; ~guard() { h->end_run(); } } g{h};
        run_steps(h);
    }
private:
    void run_steps(device_hooks* h) {
        std::vector<tensor_view> ins;
        for (auto& s : p_.steps) {
            ins.clear();
            for (auto v : s.in) ins.push_back(p_.views[v]);
            auto& out = p_.views[s.out];
            if (s.kind == 2) {   // fixed-size state op: state is host memory, updated in place
                auto& node = p_.g.at(s.out);
                auto fn = s.attrs->str("fn");
                std::vector<shape_t> shs; for (auto& v : ins) shs.push_back(v.shape);
                auto ss = state_ops::state_shape(fn, shs);
                auto& sl = st_->slots[node.name];
                if (sl.len == 0 || sl.buf.shape != ss) {   // fresh: zeros (in place when the shape is unchanged, so a device copy stays keyed)
                    if (sl.buf.shape == ss && !sl.buf.data.empty()) { std::fill(sl.buf.data.begin(), sl.buf.data.end(), 0.f); if (h) h->host_wrote(sl.buf.data.data(), sl.buf.data.size()); }
                    else { if (h && !sl.buf.data.empty()) h->release(sl.buf.data.data(), sl.buf.data.size()); sl.buf = tensor(ss); }
                    sl.len = 0; sl.fixed = true;
                }
                if (h && h->state && h->state(fn, ins, sl.buf.data.data(), sl.buf.data.size(), out, *s.attrs)) { sl.len += ins[0].shape[0]; continue; }
                if (h) { for (auto& v : ins) h->to_host(v.data, v.size()); h->to_host(out.data, out.size()); h->to_host(sl.buf.data.data(), sl.buf.data.size()); }
                if (fn == "causal_conv")
                    state_ops::causal_conv(ins[0].data, ins[1].data, sl.buf.data.data(), out.data, ins[0].shape[0], ins[0].shape[1], ins[1].shape[1]);
                else if (fn == "delta_rule") {
                    i64 dk = ins[0].shape[2];
                    state_ops::delta_rule(ins[0].data, ins[1].data, ins[2].data, ins[3].data, ins[4].data, sl.buf.data.data(), out.data,
                                          ins[0].shape[0], ins[0].shape[1], dk, ins[2].shape[1], ins[2].shape[2],
                                          s.attrs->num("l2norm", 1) != 0, float(s.attrs->num("scale", 1.0 / std::sqrt(double(dk)))));
                } else raise(errc::not_found, "unknown state op '{}'", fn);
                sl.len += ins[0].shape[0];
                if (h) { h->host_wrote(out.data, out.size()); h->host_wrote(sl.buf.data.data(), sl.buf.data.size()); }
                continue;
            }
            if (s.kind == 1) {
                auto& node = p_.g.at(s.out);
                auto& sl = st_->slots[node.name];
                i64 rows = ins[0].shape[0];
                auto ps = state_store::physical(ins[0].shape, sl.codec);
                i64 row = numel(ps) / std::max<i64>(rows, 1);
                if (h && sl.capacity && sl.len + rows > sl.capacity) h->release(sl.buf.data.data(), sl.buf.data.size());  // about to move: sync + drop
                const float* before = sl.buf.data.data();
                st_->ensure(sl, ps, sl.len + rows);
                float* dst = sl.buf.data.data() + sl.len * row;
                std::size_t nphys = std::size_t(rows * row);
                if (sl.codec != kv_codec::f32) {   // quantise on the host (the device copy of the rows is patched)
                    if (h) { h->to_host(ins[0].data, ins[0].size()); h->to_host(dst, nphys); if (sl.buf.data.data() != before) h->host_wrote(sl.buf.data.data(), sl.buf.data.size()); }
                    i64 heads = ins[0].shape[1], d = ins[0].shape[2], hw = kv_head_words(d, sl.codec);
                    for (i64 r = 0; r < rows * heads; ++r) kv_encode_head(ins[0].data + r * d, d, sl.codec, dst + r * hw);
                    if (h) h->host_wrote(dst, nphys);
                } else if (h && h->copy) {
                    if (sl.buf.data.data() != before) h->host_wrote(sl.buf.data.data(), sl.buf.data.size());
                    h->copy(ins[0].data, ins[0].size(), sl.buf.data.data(), sl.buf.data.size(), std::size_t(sl.len * row));
                } else {
                    if (h) { h->to_host(ins[0].data, ins[0].size()); h->to_host(dst, ins[0].size()); }
                    std::copy_n(ins[0].data, ins[0].size(), dst);
                    if (h) h->host_wrote(dst, ins[0].size());
                }
                sl.len += ins[0].shape[0];
                out.data = sl.buf.data.data();
                for (auto [a, r] : p_.state_aliases) if (r == s.out) p_.views[a].data = out.data;
                out.shape[0] = sl.len;   // symbolic length: downstream kernels read it at run time
                continue;
            }
            if (h && p_.g.at(s.out).op == spec::op::reshape && ins.size() == 1 && ins[0].size() == out.size() &&
                (ins[0].data == out.data || ins[0].data + ins[0].size() <= out.data || out.data + out.size() <= ins[0].data)) {
                if (ins[0].data != out.data) h->copy(const_cast<float*>(ins[0].data), out.size(), out.data, out.size(), 0);   // stays on the device (aliased: no-op)
                continue;
            }
            if (h && !s.low->on_device) {
                if (std::getenv("FORGE_PLAN_TRACE")) {   // which ops round-trip through the host on a device run
                    static std::map<std::string, std::size_t> c; static bool reg = false;
                    if (!reg) { reg = true; std::atexit([] { for (auto& [k, n] : c) std::fprintf(stderr, "host-op %s %zu\n", k.c_str(), n); }); }
                    ++c[std::string(spec::info(p_.g.at(s.out).op).name) + " " + std::to_string(out.size())];
                }
                for (auto& v : ins) if (v.data) h->to_host(v.data, v.size());   // encoded weights (qdata) live on the host only
                h->to_host(out.data, out.size());
                s.low->fn(ins, out, *s.attrs);
                h->host_wrote(out.data, out.size());
            } else s.low->fn(ins, out, *s.attrs);
            if (h && st_ && st_->offload && p_.g.at(s.out).op == spec::op::attention)
                for (auto v : s.in) if (p_.g.at(v).kind == node_kind::state_append) {
                    auto& sl = st_->slots[p_.g.at(v).name];
                    h->release(sl.buf.data.data(), sl.buf.data.size());
                }
        }
    }
public:
    tensor output(const std::string& name) const {
        for (auto& [nm, v] : p_.g.outputs()) if (nm == name) {
            auto& vw = p_.views[v]; tensor t(vw.shape); std::copy_n(vw.data, t.size(), t.data.data()); return t;
        }
        raise(errc::not_found, "no output {}", name);
    }
    tensor_view output_view(const std::string& name) const {
        for (auto& [nm, v] : p_.g.outputs()) if (nm == name) return p_.views[v];
        raise(errc::not_found, "no output {}", name);
    }
private:
    plan& p_;
    state_store* st_;
};

// Shape-bucket plan cache: key = caller-provided bucket string (e.g. "prefill:64/past:0").
class plan_cache {
public:
    template <class Build> plan& get(const std::string& key, Build&& build) {
        auto it = cache_.find(key);
        if (it != cache_.end()) { ++hits_; return *it->second; }
        ++misses_;
        return *(cache_[key] = build());
    }
    std::size_t hits() const { return hits_; }
    std::size_t misses() const { return misses_; }
    void clear() { cache_.clear(); }
private:
    std::map<std::string, std::unique_ptr<plan>> cache_;
    std::size_t hits_ = 0, misses_ = 0;
};

} // namespace forge::flow
