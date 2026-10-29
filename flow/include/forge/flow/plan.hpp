#pragma once
// partition -> memory-plan -> lower -> flat plan -> replay. Plans are cached per shape bucket.
#include <forge/flow/kernels.hpp>
#include <forge/flow/memplan.hpp>
#include <forge/flow/passes.hpp>
#include <forge/grid/grid.hpp>
#include <chrono>
#include <cstdlib>

namespace forge::flow {

// Per-session state store: slot -> preallocated [capacity, ...] tensor + current length.
struct state_store {
    struct slot { tensor buf; i64 len = 0; i64 capacity = 0; };
    std::map<std::string, slot> slots;
    i64 capacity = 4096;   // maximum rows per slot (context length)
    // Paged allocation: when > 0, slot storage grows in pages of `page_rows` rows on demand instead of
    // reserving `capacity` rows up front, so memory tracks the tokens actually held.
    i64 page_rows = 0;
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
    void truncate(i64 len) { for (auto& [k, s] : slots) s.len = std::min(s.len, len); }
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

struct step {
    const lowering* low;
    std::vector<value_id> in;
    value_id out;
    const spec::attrs* attrs;
    int kind; // 0 op, 1 state_append
};

class plan {
public:
    graph g;
    memory_plan mem;
    std::vector<step> steps;
    std::vector<tensor_view> views;   // per value
    std::vector<std::pair<value_id, value_id>> state_aliases;  // (view, state node) pairs rebound at run time
    std::vector<float> arena;
    std::map<std::string, value_id> inputs;
    partition_report part;
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
    p->hooks = reg.hooks.get();
    for (std::size_t i = 0; i < G.size(); ++i) {
        auto& n = G.nodes()[i];
        auto& s = p->mem.slots[i];
        tensor_view v; v.shape = n.shape;
        if (n.kind == node_kind::weight) v.data = const_cast<float*>(n.weight);
        else if (n.kind == node_kind::constant) v.data = n.constant->data.data();
        else if (n.kind == node_kind::state_append) {
            if (!st) raise(errc::invalid_argument, "graph uses state slot '{}' but no state store given", n.name);
            auto& sl = st->slots[n.name];
            const auto& xs = G.at(n.inputs[0]).shape;
            if (n.shape[0] > st->capacity) raise(errc::out_of_memory, "state slot '{}' exceeds capacity {}", n.name, st->capacity);
            st->ensure(sl, xs, std::max<i64>(n.shape[0], 1));
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
        if (n.kind == node_kind::op) {
            // state views grow at run time; only consumers that read the live length are allowed
            for (auto in : n.inputs)
                if (G.at(in).kind == node_kind::state_append && n.op != spec::op::attention)
                    raise(errc::unsupported, "state slot '{}' feeds '{}': only attention may consume a growing state view",
                          G.at(in).name, spec::info(n.op).name);
            std::vector<shape_t> ins; for (auto in : n.inputs) ins.push_back(G.at(in).shape);
            p->steps.push_back({&reg.select(n.op, ins, n.shape, n.attrs), n.inputs, value_id(i), &n.attrs, 0});
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
                if (k == node_kind::weight || k == node_kind::constant) h->pin(p_.views[i].data, p_.views[i].size());
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
            if (s.kind == 1) {
                auto& node = p_.g.at(s.out);
                auto& sl = st_->slots[node.name];
                i64 row = numel(ins[0].shape) / std::max<i64>(ins[0].shape[0], 1);
                i64 rows = ins[0].shape[0];
                if (h && sl.capacity && sl.len + rows > sl.capacity) h->release(sl.buf.data.data(), sl.buf.data.size());  // about to move: sync + drop
                const float* before = sl.buf.data.data();
                st_->ensure(sl, ins[0].shape, sl.len + rows);
                float* dst = sl.buf.data.data() + sl.len * row;
                if (h && h->copy) {
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
            if (h && !s.low->on_device) {
                for (auto& v : ins) h->to_host(v.data, v.size());
                h->to_host(out.data, out.size());
                s.low->fn(ins, out, *s.attrs);
                h->host_wrote(out.data, out.size());
            } else s.low->fn(ins, out, *s.attrs);
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
