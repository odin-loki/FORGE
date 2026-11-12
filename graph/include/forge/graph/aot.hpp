#pragma once
// Ahead-of-time plans (docs/aot-models.md): a compiled plan (optimised graph, memory plan, lowering per step) written
// to a versioned blob and brought back without running the compile pipeline.
//
//   build time : compile() as usual for each target (sequence length, all_rows, KV codec), then builder::add + finish
//   load time  : image(blob) checks the layout and BLAKE3; instantiate() rebuilds the plan: weights are re-resolved by
//                name, shapes are re-inferred from the op spec, the memory plan is re-checked against the graph, and
//                lowerings are looked up by name in the registry (and pass the oracle, as in compile()).
//
// The blob is untrusted input. Nothing in it is used as an address: every offset is checked against the arena, every
// buffer size is recomputed from the inferred shapes, and a blob that fails any check is rejected (forge_exception),
// after which callers compile the plan the usual way. What a blob can still decide is which valid graph runs, the
// same thing the blueprint decides; a signature has to cover it for the model to be trusted (modelfile/aot.hpp).
//
// Blob layout (little-endian; str = u32 length + bytes):
//   "FORGEAOT" | u32 format_version | u32 n_entries | u8 model_key[32] | u32 pipeline_version | str forge_version
//   | str simd_target (informational) | n_entries x { u8 key[32]; i64 seq; u8 all_rows; str kv_codec; str device;
//   u64 offset; u64 size } | entries | u8 blake3[32] over everything before it
// Entry:
//   u32 n_nodes x { u8 kind; str op; u32 n_in, u32 in[]; attrs; u32 rank, i64 dim[]; str name; u8 weight_kind;
//                   str codec; constants: u64 n, f32[n] }
//   u32 n_outputs x { str name; u32 value }
//   u64 arena_bytes; u64 alignment; u64 naive_bytes; u32 in_place; u32 aliases; n_nodes x { u8 kind; u64 offset;
//   u64 bytes; u32 root }
//   u32 n_steps x { u8 kind; u32 out; str lowering; u8 fuse }
// attrs: u32 n x { str key; u8 type (0 f64, 1 i64 list, 2 str); value }
#include <forge/graph/plan.hpp>
#include <forge/hash/blake3.hpp>
#include <cstring>

namespace forge::graph::aot {

inline constexpr char magic[9] = "FORGEAOT";
inline constexpr std::uint32_t format_version = 1;
// Bump when optimise(), plan_memory() or the lowering selection in lower_plan() change what they produce for the same
// graph: it is part of every key, so older precompiled plans stop matching and are recompiled.
inline constexpr std::uint32_t pipeline_version = 1;
inline constexpr std::string_view forge_version = "0.1.0";
using digest = hash::blake3::digest;

// What a plan was compiled for. The CPU lowerings dispatch on the SIMD target at run time, so one plan serves every
// CPU; device plans are not serialisable yet (their kernels are created per process by the backend).
struct target {
    i64 seq = 1;                   // new tokens per run (the blueprint's `seq` dim)
    bool all_rows = false;         // logits for every row
    std::string kv_codec = "f32";  // kv_codec_pair_name of the session's state store
    std::string device = "cpu";
    std::string name() const { return std::format("{}:{}:{}:{}", device, all_rows ? "all" : "seq", seq, kv_codec); }
};

namespace detail {
// limits for untrusted blobs
inline constexpr std::size_t max_nodes = std::size_t(1) << 22, max_str = 4096, max_inputs = 256, max_rank = 16, max_attrs = 256,
                             max_entries = 4096;
inline constexpr i64 max_dim = i64(1) << 40, max_elems = i64(1) << 40;

struct wbuf {
    std::vector<std::uint8_t> b;
    void u8(std::uint8_t v) { b.push_back(v); }
    void u32(std::uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(std::uint8_t(v >> (8 * i))); }
    void u64(std::uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back(std::uint8_t(v >> (8 * i))); }
    void i64_(i64 v) { u64(std::uint64_t(v)); }
    void f64(double v) { std::uint64_t u; std::memcpy(&u, &v, 8); u64(u); }
    void f32(float v) { std::uint32_t u; std::memcpy(&u, &v, 4); u32(u); }
    void str(std::string_view s) { u32(std::uint32_t(s.size())); b.insert(b.end(), s.begin(), s.end()); }
    void raw(const void* p, std::size_t n) { auto* c = static_cast<const std::uint8_t*>(p); b.insert(b.end(), c, c + n); }
};
struct rbuf {
    const std::uint8_t* p; std::size_t n, at = 0;
    rbuf(std::span<const std::uint8_t> s) : p(s.data()), n(s.size()) {}
    void need(std::size_t k) const { if (k > n - at) raise(errc::format_error, "aot: truncated plan blob"); }
    std::uint8_t u8() { need(1); return p[at++]; }
    std::uint32_t u32() { need(4); std::uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= std::uint32_t(p[at + std::size_t(i)]) << (8 * i); at += 4; return v; }
    std::uint64_t u64() { need(8); std::uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= std::uint64_t(p[at + std::size_t(i)]) << (8 * i); at += 8; return v; }
    i64 i64_() { return i64(u64()); }
    double f64() { auto u = u64(); double v; std::memcpy(&v, &u, 8); return v; }
    float f32() { auto u = u32(); float v; std::memcpy(&v, &u, 4); return v; }
    std::string str(std::size_t max = max_str) {
        std::size_t k = u32(); if (k > max) raise(errc::format_error, "aot: string of {} bytes in plan blob", k);
        need(k); std::string s(reinterpret_cast<const char*>(p + at), k); at += k; return s;
    }
    // a count that must leave at least `each` bytes per item in the blob (so hostile counts cannot drive allocations)
    std::size_t count(std::size_t max, std::size_t each) {
        std::size_t k = u32(); if (k > max || (each && k > (n - at) / each)) raise(errc::format_error, "aot: count {} out of range in plan blob", k);
        return k;
    }
};
// numel with overflow and range checks
inline i64 checked_numel(const shape_t& s) {
    if (s.size() > max_rank) raise(errc::format_error, "aot: rank {} in plan blob", s.size());
    i64 r = 1;
    for (auto d : s) {
        if (d < 0 || d > max_dim) raise(errc::format_error, "aot: dimension {} in plan blob", d);
        if (d && r > max_elems / d) raise(errc::format_error, "aot: tensor too large in plan blob");
        r *= d;
    }
    return r;
}
inline void put_attrs(wbuf& w, const ops::attrs& a) {
    w.u32(std::uint32_t(a.v.size()));
    for (auto& [k, v] : a.v) {
        w.str(k);
        if (auto* d = std::get_if<double>(&v)) { w.u8(0); w.f64(*d); }
        else if (auto* iv = std::get_if<std::vector<i64>>(&v)) { w.u8(1); w.u32(std::uint32_t(iv->size())); for (auto x : *iv) w.i64_(x); }
        else { w.u8(2); w.str(std::get<std::string>(v)); }
    }
}
inline ops::attrs get_attrs(rbuf& r) {
    ops::attrs a;
    for (std::size_t i = 0, n = r.count(max_attrs, 6); i < n; ++i) {
        auto k = r.str(256);
        switch (r.u8()) {
            case 0: a.set(std::move(k), r.f64()); break;
            case 1: { std::vector<i64> iv(r.count(4096, 8)); for (auto& x : iv) x = r.i64_(); a.set(std::move(k), std::move(iv)); break; }
            case 2: a.set(std::move(k), r.str()); break;
            default: raise(errc::format_error, "aot: bad attribute type in plan blob");
        }
    }
    return a;
}
inline void put_target(hash::blake3& h, const target& t) {
    wbuf w; w.str(t.device); w.str(t.kv_codec); w.i64_(t.seq); w.u8(t.all_rows);
    h.update(w.b.data(), w.b.size());
}
} // namespace detail

// Identifies the model and the compiler: the blueprint text, the tensor index (names, shapes, codecs and content
// hashes of every weight) and the format / pass pipeline / FORGE versions.
inline digest model_key(std::string_view blueprint, std::string_view tensor_index) {
    hash::blake3 h;
    detail::wbuf w; w.str("forge-aot-plan"); w.u32(format_version); w.u32(pipeline_version); w.str(forge_version);
    h.update(w.b.data(), w.b.size());
    auto hb = hash::blake3::of({reinterpret_cast<const std::uint8_t*>(blueprint.data()), blueprint.size()});
    auto ht = hash::blake3::of({reinterpret_cast<const std::uint8_t*>(tensor_index.data()), tensor_index.size()});
    h.update(hb.data(), hb.size()); h.update(ht.data(), ht.size());
    return h.final();
}
inline digest entry_key(const digest& model, const target& t) {
    hash::blake3 h; h.update(model.data(), model.size()); detail::put_target(h, t); return h.final();
}

// One plan as entry bytes. The plan must be a CPU plan (no device hooks, no pipeline stages).
inline std::vector<std::uint8_t> serialise_plan(const plan& p) {
    if (p.hooks || !p.stage_of_node.empty()) raise(errc::unsupported, "aot: only CPU plans can be precompiled (this plan was lowered for a device)");
    detail::wbuf w;
    const auto& G = p.g;
    w.u32(std::uint32_t(G.size()));
    for (std::size_t i = 0; i < G.size(); ++i) {
        const auto& n = G.at(value_id(i));
        w.u8(std::uint8_t(n.kind));
        w.str(n.kind == node_kind::op ? ops::info(n.op).name : std::string_view{});
        w.u32(std::uint32_t(n.inputs.size())); for (auto in : n.inputs) w.u32(in);
        detail::put_attrs(w, n.attrs);
        w.u32(std::uint32_t(n.shape.size())); for (auto d : n.shape) w.i64_(d);
        w.str(n.name);
        w.u8(n.kind != node_kind::weight ? 0 : n.qweight ? 2 : 1);
        w.str(n.qcodec);
        if (n.kind == node_kind::constant) {
            w.u64(n.constant->data.size());
            for (auto f : n.constant->data) w.f32(f);
        }
    }
    w.u32(std::uint32_t(G.outputs().size()));
    for (auto& [nm, v] : G.outputs()) { w.str(nm); w.u32(v); }
    w.u64(p.mem.arena_bytes); w.u64(p.mem.alignment); w.u64(p.mem.naive_bytes); w.u32(std::uint32_t(p.mem.in_place)); w.u32(std::uint32_t(p.mem.aliases));
    for (auto& s : p.mem.slots) { w.u8(std::uint8_t(s.kind)); w.u64(s.offset); w.u64(s.bytes); w.u32(s.root); }
    w.u32(std::uint32_t(p.steps.size()));
    for (auto& s : p.steps) {
        if (s.low && s.low->on_device) raise(errc::unsupported, "aot: step '{}' runs on a device", s.low->name);
        w.u8(std::uint8_t(s.kind)); w.u32(s.out); w.str(s.low ? std::string_view(s.low->name) : std::string_view{}); w.u8(std::uint8_t(s.fuse));
    }
    return std::move(w.b);
}

// Collects plans for one model and writes the blob.
class builder {
public:
    explicit builder(const digest& model) : model_(model) {}
    void add(const target& t, const plan& p) {
        for (auto& e : entries_) if (e.t.name() == t.name()) raise(errc::invalid_argument, "aot: target {} given twice", t.name());
        entries_.push_back({t, serialise_plan(p)});
    }
    std::size_t size() const { return entries_.size(); }
    std::vector<std::uint8_t> finish(std::string_view simd_target = {}) const {
        detail::wbuf w;
        w.raw(magic, 8); w.u32(format_version); w.u32(std::uint32_t(entries_.size())); w.raw(model_.data(), 32);
        w.u32(pipeline_version); w.str(forge_version); w.str(simd_target);
        std::size_t dir = w.b.size(), fixed = 0;
        for (auto& e : entries_) fixed += 32 + 8 + 1 + 4 + e.t.kv_codec.size() + 4 + e.t.device.size() + 16;
        std::uint64_t off = dir + fixed;
        for (auto& e : entries_) {
            auto k = entry_key(model_, e.t);
            w.raw(k.data(), 32); w.i64_(e.t.seq); w.u8(e.t.all_rows); w.str(e.t.kv_codec); w.str(e.t.device); w.u64(off); w.u64(e.bytes.size());
            off += e.bytes.size();
        }
        for (auto& e : entries_) w.raw(e.bytes.data(), e.bytes.size());
        auto h = hash::blake3::of(w.b);
        w.raw(h.data(), 32);
        return std::move(w.b);
    }
private:
    struct ent { target t; std::vector<std::uint8_t> bytes; };
    digest model_;
    std::vector<ent> entries_;
};

// A checked view of a blob: header, directory and whole-blob hash. Entries are parsed on demand by instantiate().
// The bytes are borrowed and must outlive the image.
class image {
public:
    struct entry { target t; digest key{}; std::uint64_t offset = 0, size = 0; };
    explicit image(std::span<const std::uint8_t> blob, bool check_hash = true) : blob_(blob) {
        if (blob.size() < 8 + 4 + 4 + 32 + 4 + 32 || std::memcmp(blob.data(), magic, 8)) raise(errc::format_error, "aot: not a plan blob");
        detail::rbuf r(blob.first(blob.size() - 32)); r.at = 8;
        if (auto v = r.u32(); v != format_version) raise(errc::unsupported, "aot: plan blob format {} not supported", v);
        std::size_t n = r.count(detail::max_entries, 32 + 8 + 1 + 4 + 4 + 16);
        r.need(32); std::memcpy(model_.data(), r.p + r.at, 32); r.at += 32;
        pipeline_ = r.u32(); forge_ = r.str(64); simd_ = r.str(64);
        for (std::size_t i = 0; i < n; ++i) {
            entry e;
            r.need(32); std::memcpy(e.key.data(), r.p + r.at, 32); r.at += 32;
            e.t.seq = r.i64_(); e.t.all_rows = r.u8() != 0; e.t.kv_codec = r.str(64); e.t.device = r.str(256);
            e.offset = r.u64(); e.size = r.u64();
            if (e.offset > r.n || e.size > r.n - e.offset) raise(errc::format_error, "aot: entry {} out of bounds", i);
            entries_.push_back(std::move(e));
        }
        if (check_hash) {
            auto h = hash::blake3::of(blob.first(blob.size() - 32));
            if (std::memcmp(h.data(), blob.data() + blob.size() - 32, 32)) raise(errc::integrity_error, "aot: plan blob hash mismatch");
        }
    }
    const digest& model() const { return model_; }
    std::uint32_t pipeline() const { return pipeline_; }
    const std::string& forge() const { return forge_; }
    const std::string& simd() const { return simd_; }   // the SIMD target of the machine that built it (informational)
    const std::vector<entry>& entries() const { return entries_; }
    // the entry compiled for `t` under model key `model`, or null (other model, other version, target not built)
    const entry* find(const digest& model, const target& t) const {
        if (model != model_) return nullptr;
        auto k = entry_key(model, t);
        for (auto& e : entries_) if (e.key == k) return &e;
        return nullptr;
    }
    std::span<const std::uint8_t> bytes(const entry& e) const { return blob_.subspan(std::size_t(e.offset), std::size_t(e.size)); }
    std::span<const std::uint8_t> blob() const { return blob_; }
private:
    std::span<const std::uint8_t> blob_;
    digest model_{};
    std::uint32_t pipeline_ = 0;
    std::string forge_, simd_;
    std::vector<entry> entries_;
};

// Resolves the weights a plan names (the same callbacks the blueprint builder uses).
struct binder {
    struct encoded { shape_t shape; const std::uint8_t* data = nullptr; std::string codec; };
    std::function<std::pair<shape_t, const float*>(const std::string&)> weight;
    std::function<encoded(const std::string&)> qweight;   // optional; null data = not stored encoded
};

namespace detail {
struct step_image { int kind = 0; value_id out = 0; std::string low; int fuse = 0; };
struct entry_image { tensor_graph g; memory_plan mem; std::vector<step_image> steps; std::vector<shape_t> shapes; };

// Parse one entry into a graph (weights bound through `b`), a memory plan and the step list. Shapes of op nodes are
// kept aside in `shapes`: the caller re-infers them.
inline entry_image parse_entry(std::span<const std::uint8_t> bytes, const binder& b) {
    rbuf r(bytes);
    entry_image im;
    std::size_t N = r.count(max_nodes, 1 + 4 + 4 + 4 + 4 + 4 + 1 + 4);
    auto& nodes = im.g.nodes(); nodes.reserve(N); im.shapes.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        node n;
        auto kind = r.u8();
        if (kind > std::uint8_t(node_kind::state_op) || kind == std::uint8_t(node_kind::output)) raise(errc::format_error, "aot: node %{} has kind {}", i, kind);
        n.kind = node_kind(kind);
        auto opname = r.str(64);
        if (n.kind == node_kind::op) {
            auto o = ops::op_from_name(opname);
            if (!o) raise(errc::format_error, "aot: node %{}: unknown op '{}'", i, opname);
            n.op = *o;
        }
        n.inputs.resize(r.count(max_inputs, 4));
        for (auto& in : n.inputs) { in = r.u32(); if (in >= i) raise(errc::format_error, "aot: node %{} uses %{} before definition", i, in); }
        n.attrs = get_attrs(r);
        n.shape.resize(r.count(max_rank, 8)); for (auto& d : n.shape) d = r.i64_();
        i64 elems = checked_numel(n.shape);
        n.name = r.str();
        auto wk = r.u8(); auto codec = r.str(64);
        switch (n.kind) {
            case node_kind::input: if (!n.inputs.empty()) raise(errc::format_error, "aot: input %{} has inputs", i); break;
            case node_kind::weight: {
                if (!n.inputs.empty() || (wk != 1 && wk != 2)) raise(errc::format_error, "aot: bad weight node %{}", i);
                binder::encoded q; if (b.qweight) q = b.qweight(n.name);
                if (wk == 2) {
                    if (!q.data || q.codec != codec || q.shape != n.shape) raise(errc::invalid_argument, "aot: weight '{}' is not stored as {} {}", n.name, codec, shape_str(n.shape));
                    n.qweight = q.data; n.qcodec = codec;
                } else {
                    if (q.data || !b.weight) raise(errc::invalid_argument, "aot: weight '{}' is stored encoded, the plan expects f32", n.name);
                    auto [s, p] = b.weight(n.name);
                    if (!p || s != n.shape) raise(errc::invalid_argument, "aot: weight '{}' has shape {}, the plan expects {}", n.name, shape_str(s), shape_str(n.shape));
                    n.weight = p;
                }
                break;
            }
            case node_kind::constant: {
                if (!n.inputs.empty()) raise(errc::format_error, "aot: constant %{} has inputs", i);
                std::uint64_t k = r.u64();
                if (k != std::uint64_t(elems) || k > (r.n - r.at) / 4) raise(errc::format_error, "aot: constant %{} does not match its shape", i);
                tensor t(n.shape);
                for (auto& f : t.data) f = r.f32();
                n.constant = std::make_shared<tensor>(std::move(t));
                break;
            }
            case node_kind::state_append: if (n.inputs.size() != 1 || n.name.empty()) raise(errc::format_error, "aot: bad state_append %{}", i); break;
            case node_kind::state_op: if (n.inputs.empty() || n.name.empty()) raise(errc::format_error, "aot: bad state_op %{}", i); break;
            default: break;
        }
        im.shapes.push_back(n.shape);
        nodes.push_back(std::move(n));
    }
    for (std::size_t i = 0, k = r.count(max_nodes, 8); i < k; ++i) {
        auto nm = r.str(); auto v = r.u32();
        if (v >= N) raise(errc::format_error, "aot: output '{}' undefined", nm);
        im.g.output(std::move(nm), v);
    }
    auto& m = im.mem;
    m.arena_bytes = std::size_t(r.u64()); m.alignment = std::size_t(r.u64()); m.naive_bytes = std::size_t(r.u64());
    m.in_place = int(r.u32()); m.aliases = int(r.u32());
    r.need(N * 21);
    m.slots.resize(N);
    for (auto& s : m.slots) {
        auto k = r.u8(); if (k > 2) raise(errc::format_error, "aot: bad buffer kind {}", k);
        s.kind = buffer_slot::kind_t(k); s.offset = std::size_t(r.u64()); s.bytes = std::size_t(r.u64()); s.root = r.u32();
    }
    im.steps.resize(r.count(max_nodes, 1 + 4 + 4 + 1));
    for (auto& s : im.steps) {
        s.kind = r.u8(); s.out = r.u32(); s.low = r.str(128); s.fuse = r.u8();
        if (s.kind > 2 || s.out >= N) raise(errc::format_error, "aot: bad step in plan blob");
    }
    if (r.at != r.n) raise(errc::format_error, "aot: {} trailing bytes in plan entry", r.n - r.at);
    return im;
}

// The memory plan of a blob, checked against the graph it came with: liveness is recomputed here (the stored plan
// only supplies kinds, offsets and alias roots), sizes come from the inferred shapes, every arena buffer lies inside
// the arena, aliases are only the ones plan_memory() makes (reshape views, in-place element-wise ops over a buffer
// that dies there), and no two live buffers overlap.
inline void check_memory(const tensor_graph& g, const std::vector<shape_t>& shapes, memory_plan& m) {
    std::size_t N = g.size();
    if (m.slots.size() != N || m.alignment != 64) raise(errc::format_error, "aot: memory plan does not fit the graph");
    std::vector<int> last(N, -1);
    for (std::size_t i = 0; i < N; ++i) for (auto in : g.at(value_id(i)).inputs) last[in] = int(i);
    for (auto& [nm, v] : g.outputs()) last[v] = int(N);
    auto up = [&](std::size_t x) { return (x + m.alignment - 1) / m.alignment * m.alignment; };
    std::vector<value_id> root_of(N);
    std::vector<int> group_last(N, -1);   // per storage root: last use of any value stored there so far
    std::size_t total = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const auto& n = g.at(value_id(i)); auto& s = m.slots[i];
        s.def = int(i); s.last = std::max(int(i), last[i]);
        if (s.bytes != std::size_t(checked_numel(shapes[i])) * sizeof(float)) raise(errc::format_error, "aot: buffer %{} has {} bytes, its shape needs {}", i, s.bytes, std::size_t(numel(shapes[i])) * sizeof(float));
        bool ext = n.kind == node_kind::weight || n.kind == node_kind::constant || n.kind == node_kind::state_append;
        root_of[i] = value_id(i);
        if (ext != (s.kind == buffer_slot::external)) raise(errc::format_error, "aot: buffer %{} has the wrong kind", i);
        if (s.kind == buffer_slot::arena) {
            if (s.offset % m.alignment || s.offset > m.arena_bytes || up(s.bytes) > m.arena_bytes - s.offset) raise(errc::format_error, "aot: buffer %{} lies outside the arena", i);
            total += up(s.bytes);
        } else if (s.kind == buffer_slot::alias) {
            if (s.root >= i) raise(errc::format_error, "aot: buffer %{} aliases a later value", i);
            value_id r = root_of[s.root];
            if (is_view_op(n)) {
                if (s.root != n.inputs[0] || m.slots[s.root].bytes != s.bytes) raise(errc::format_error, "aot: view %{} does not alias its input", i);
            } else if (is_inplace_candidate(n)) {
                bool mine = false; for (auto in : n.inputs) mine = mine || root_of[in] == r;
                if (!mine || s.root != r || m.slots[r].kind != buffer_slot::arena || m.slots[r].bytes != s.bytes || group_last[r] > int(i))
                    raise(errc::format_error, "aot: buffer %{} reuses storage that is still live", i);
            } else raise(errc::format_error, "aot: buffer %{} may not alias", i);
            root_of[i] = r;
        }
        group_last[root_of[i]] = std::max(group_last[root_of[i]], s.last);
    }
    if (m.arena_bytes > total || m.arena_bytes > std::size_t(max_elems)) raise(errc::format_error, "aot: arena of {} bytes is larger than its buffers", m.arena_bytes);
    for (std::size_t i = 0; i < N; ++i) if (m.slots[i].kind == buffer_slot::alias) m.slots[root_of[i]].last = std::max(m.slots[root_of[i]].last, m.slots[i].last);
    if (!plan_is_safe(m)) raise(errc::format_error, "aot: memory plan has overlapping live buffers");
}
// State slots: one node per slot in a plan, and a slot the session already holds must be used the way this plan uses
// it (same kind, same row layout), since plans of one session share the store and the executor trusts the slot.
inline void check_state(const tensor_graph& g, const state_store* st) {
    std::set<std::string> seen;
    for (std::size_t i = 0; i < g.size(); ++i) {
        const auto& n = g.at(value_id(i));
        if (n.kind != node_kind::state_append && n.kind != node_kind::state_op) continue;
        if (!seen.insert(n.name).second) raise(errc::format_error, "aot: state slot '{}' is used by two nodes", n.name);
        if (!st) raise(errc::invalid_argument, "aot: plan uses state slot '{}' but no state store given", n.name);
        auto it = st->slots.find(n.name);
        if (it == st->slots.end()) continue;
        const auto& sl = it->second;
        if (n.kind == node_kind::state_op) {
            if (!sl.fixed && sl.capacity) raise(errc::format_error, "aot: state slot '{}' holds appended rows, the plan uses it as recurrent state", n.name);
            continue;
        }
        if (sl.fixed) raise(errc::format_error, "aot: state slot '{}' is recurrent state, the plan appends to it", n.name);
        const auto& xs = g.at(n.inputs[0]).shape;
        if (xs.empty() || (sl.codec != kv_codec::f32 && xs.size() != 3)) raise(errc::format_error, "aot: state slot '{}' cannot take rows of shape {}", n.name, shape_str(xs));
        if (!sl.capacity) continue;
        auto ps = state_store::physical(g.at(n.inputs[0]).shape, sl.codec);
        if (ps.size() != sl.buf.shape.size() || ps.empty() || !std::equal(ps.begin() + 1, ps.end(), sl.buf.shape.begin() + 1))
            raise(errc::format_error, "aot: state slot '{}' holds rows of another shape", n.name);
    }
}
} // namespace detail

// Rebuild the plan of entry `e`. `st` is the session's state store, as for compile(). Raises forge_exception when the
// entry is malformed, names weights the model does not have in that form, or does not lower the same way here; the
// caller then compiles the plan instead. `max_arena_bytes` bounds what a blob may make this allocate (fuzzing).
inline std::unique_ptr<plan> instantiate(const image& img, const image::entry& e, const binder& b, state_store* st = nullptr, const pass_options& po = {},
                                         std::size_t max_arena_bytes = ~std::size_t(0)) {
    if (po.lowerings) raise(errc::unsupported, "aot: precompiled plans are CPU plans");
    if (e.t.device != "cpu") raise(errc::unsupported, "aot: plan was compiled for device '{}'", e.t.device);
    try {
        auto im = detail::parse_entry(img.bytes(e), b);
        auto sshapes = st ? st->shapes() : std::map<std::string, shape_t>{};
        verify_graph(im.g, sshapes);   // arity and shapes from the op spec, not from the blob
        for (std::size_t i = 0; i < im.g.size(); ++i) {
            auto& n = im.g.nodes()[i];
            if (n.kind == node_kind::state_append) {   // stored for an empty store: [rows, ...] of its input
                if (im.shapes[i] != im.g.at(n.inputs[0]).shape) raise(errc::format_error, "aot: state_append %{} does not match its input", i);
            } else if (im.shapes[i] != n.shape) raise(errc::format_error, "aot: node %{} has shape {}, the plan says {}", i, shape_str(n.shape), shape_str(im.shapes[i]));
            detail::checked_numel(n.shape);
        }
        detail::check_state(im.g, st);
        // buffer sizes are those of the stored shapes (a state view's storage is the store's, not the arena's)
        detail::check_memory(im.g, im.shapes, im.mem);
        if (im.mem.arena_bytes > max_arena_bytes) raise(errc::out_of_memory, "aot: plan needs an arena of {} bytes", im.mem.arena_bytes);
        auto p = std::make_unique<plan>();
        p->g = std::move(im.g);
        p->part = partition(p->g, gpu::all());
        p->mem = std::move(im.mem);
        std::vector<const detail::step_image*> step_of(p->g.size(), nullptr);
        for (auto& s : im.steps) step_of[s.out] = &s;
        auto& reg = registry::cpu();
        lower_plan(*p, st, po, [&](value_id v, std::span<const shape_t> ins) -> const lowering* {
            auto* s = step_of[v]; const auto& n = p->g.at(v);
            if (!s || s->kind != 0) raise(errc::format_error, "aot: node %{} has no step", v);
            if (s->low == "reference") return nullptr;   // checked below: the registry must also end up there
            for (auto& l : reg.lowerings(n.op))
                if (l.name == s->low) {
                    if ((l.supports && !l.supports(ins, n.shape, n.attrs)) || !reg.ensure_verified(l))
                        raise(errc::unsupported, "aot: lowering '{}' cannot run node %{} here", s->low, v);
                    return &l;
                }
            return nullptr;   // a special lowering (quantised weights): chosen by lower_plan, compared below
        });
        if (p->steps.size() != im.steps.size()) raise(errc::format_error, "aot: plan has {} steps here, {} when it was built", p->steps.size(), im.steps.size());
        for (std::size_t i = 0; i < p->steps.size(); ++i) {
            const auto& a = p->steps[i]; const auto& s = im.steps[i];
            if (a.kind != s.kind || a.out != s.out || a.fuse != s.fuse || (a.low ? a.low->name : std::string()) != s.low)
                raise(errc::format_error, "aot: step {} lowers to '{}' here, '{}' when it was built", i, a.low ? a.low->name : std::string("state"), s.low);
        }
        p->aot = true;
        return p;
    } catch (const forge_exception&) { throw; }
    catch (const std::exception& ex) { raise(errc::format_error, "aot: malformed plan entry ({})", ex.what()); }
}

} // namespace forge::graph::aot
