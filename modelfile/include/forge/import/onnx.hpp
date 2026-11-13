#pragma once
// ONNX -> .fpk importer (FORGE_PLAN Part B "ONNX: partial"). A dependency-free protobuf wire decoder reads
// ModelProto; the graph is translated node-by-node into blueprint text and initializers become weights.
// Supported: static-shape inference graphs (MLPs, classifiers, transformer encoder blocks) built from
//   MatMul Gemm Add Sub Mul Div Pow Max Min Relu Sigmoid Tanh Exp Log Sqrt Neg Abs Erf Gelu Softplus
//   Reciprocal Softmax LayerNormalization Transpose Reshape Flatten Identity Constant ReduceMean ReduceSum
// One symbolic dimension (any dim_param) maps to the blueprint dimension `seq`. Anything else is rejected
// with the op name, so the error says exactly what is missing.
#include <forge/import/hf.hpp>
#include <charconv>
#include <limits>
#include <list>
#include <map>
#include <set>

namespace forge::import {

namespace onnx_pb {

// protobuf wire format: (field << 3 | wire_type) keys; 0 varint, 1 fixed64, 2 length-delimited, 5 fixed32
struct field { std::uint32_t num = 0, wt = 0; std::uint64_t v = 0; std::string_view bytes; };

class cursor {
public:
    explicit cursor(std::string_view s) : s_(s) {}
    bool done() const { return p_ >= s_.size(); }
    std::uint64_t varint() {
        std::uint64_t v = 0;
        for (int sh = 0; sh < 64; sh += 7) {
            if (p_ >= s_.size()) raise(errc::format_error, "onnx: truncated varint");
            auto b = std::uint8_t(s_[p_++]);
            v |= std::uint64_t(b & 0x7f) << sh;
            if (!(b & 0x80)) return v;
        }
        raise(errc::format_error, "onnx: varint too long");
    }
    field next() {
        field f; auto key = varint(); f.num = std::uint32_t(key >> 3); f.wt = std::uint32_t(key & 7);
        switch (f.wt) {
        case 0: f.v = varint(); break;
        case 1: need(8); std::memcpy(&f.v, s_.data() + p_, 8); p_ += 8; break;
        case 5: { need(4); std::uint32_t x; std::memcpy(&x, s_.data() + p_, 4); f.v = x; p_ += 4; break; }
        case 2: { auto n = varint(); need(n); f.bytes = s_.substr(p_, std::size_t(n)); p_ += std::size_t(n); break; }
        default: raise(errc::format_error, "onnx: unsupported wire type {}", f.wt);
        }
        return f;
    }
private:
    void need(std::uint64_t n) { if (n > s_.size() - p_) raise(errc::format_error, "onnx: truncated field"); }
    std::string_view s_; std::size_t p_ = 0;
};

template <class F> void each(std::string_view s, F&& f) { cursor c(s); while (!c.done()) f(c.next()); }

// repeated scalar fields may be packed (wt 2) or not
inline void ints_of(const field& f, std::vector<i64>& out) {
    if (f.wt == 2) { cursor c(f.bytes); while (!c.done()) out.push_back(i64(c.varint())); }
    else out.push_back(i64(f.v));
}
inline void floats_of(const field& f, std::vector<float>& out) {
    auto put = [&](std::uint32_t u) { float x; std::memcpy(&x, &u, 4); out.push_back(x); };
    if (f.wt == 2) {
        if (f.bytes.size() % 4) raise(errc::format_error, "onnx: packed floats not a multiple of 4");
        for (std::size_t i = 0; i < f.bytes.size(); i += 4) { std::uint32_t u; std::memcpy(&u, f.bytes.data() + i, 4); put(u); }
    } else put(std::uint32_t(f.v));
}

struct tensor_t {
    std::string name; shape_t dims; int dtype = 0; std::vector<float> f; std::vector<i64> i;
    bool integer() const { return dtype == 6 || dtype == 7 || dtype == 9 || dtype == 2 || dtype == 3 || dtype == 5 || dtype == 12 || dtype == 13; }
    std::size_t count() const { return integer() ? i.size() : f.size(); }
    double at(std::size_t k) const { return integer() ? double(i[k]) : double(f[k]); }
};
// External tensor data (TensorProto.data_location = EXTERNAL, external_data = {location, offset, length}): the bytes
// of a file next to the model. `location` comes from the model file, so it is joined to the model's directory with
// security::safe_join (no absolute path, no "..", no symlink out of that directory), the file is opened under the
// fs_read capability, and offset and length are checked against the file's size and against the bytes the tensor's
// shape and type need. With an empty `base` (a model that did not come from a file) external data is refused.
struct external_loader {
    std::string base;
    std::string load(const std::string& tensor, const std::string& location, std::uint64_t offset, std::optional<std::uint64_t> length, std::uint64_t expect) const {
        if (base.empty()) raise(errc::unsupported, "onnx: tensor {} uses external data, and there is no model directory to resolve '{}' against", tensor, location.substr(0, 200));
        if (location.empty()) raise(errc::format_error, "onnx: tensor {} has external data without a location", tensor);
        std::string path;
        try { path = security::safe_join(base, location); }
        catch (const std::exception& e) { raise(errc::format_error, "onnx: tensor {}: external data location '{}' is not a path inside the model directory ({})", tensor, location.substr(0, 200), e.what()); }
        security::enforce(security::capability::fs_read, path);
        std::error_code ec; auto fsz = std::filesystem::file_size(os::fs::u8path(path), ec);
        if (ec) raise(errc::io_error, "onnx: tensor {}: cannot read external data file {}", tensor, path);
        std::uint64_t size = std::uint64_t(fsz);
        if (offset > size) raise(errc::format_error, "onnx: tensor {}: external data offset {} is past the end of {} ({} bytes)", tensor, offset, path, size);
        std::uint64_t len = length ? *length : size - offset;
        if (len > size - offset) raise(errc::format_error, "onnx: tensor {}: external data [{}, +{}) does not fit in {} ({} bytes)", tensor, offset, len, path, size);
        if (len != expect) raise(errc::format_error, "onnx: tensor {}: external data holds {} bytes, its shape and type need {}", tensor, len, expect);
        std::string out(std::size_t(len), '\0');
        std::ifstream f(os::fs::u8path(path), std::ios::binary);
        f.seekg(std::streamoff(offset)); f.read(out.data(), std::streamsize(len));
        if (!f || std::uint64_t(f.gcount()) != len) raise(errc::io_error, "onnx: tensor {}: short read from {}", tensor, path);
        return out;
    }
};
struct attr_t { std::string name; float f = 0; i64 i = 0; std::string s; std::vector<float> fl; std::vector<i64> il; std::optional<tensor_t> t; int type = 0; };
struct node_t { std::string op, name, domain; std::vector<std::string> in, out; std::map<std::string, attr_t> attrs; };
struct vinfo_t { std::string name; int elem = 0; std::vector<i64> dims; std::vector<bool> symbolic; std::vector<std::string> names; };   // names: dim_param of symbolic dims
struct model_t { i64 opset = 0; std::string producer, doc; std::vector<std::pair<std::string, std::string>> metadata; std::vector<node_t> nodes; std::vector<tensor_t> inits; std::vector<vinfo_t> inputs, outputs; };

inline tensor_t parse_tensor(std::string_view s, const external_loader* ext = nullptr) {
    tensor_t t; std::string_view raw; std::vector<i64> i32; std::string ext_bytes;
    std::string ext_location; std::uint64_t ext_offset = 0; std::optional<std::uint64_t> ext_length; bool external = false;
    each(s, [&](const field& f) {
        switch (f.num) {
        case 1: ints_of(f, t.dims); break;
        case 2: t.dtype = int(f.v); break;
        case 4: floats_of(f, t.f); break;
        case 5: ints_of(f, i32); break;
        case 7: ints_of(f, t.i); break;
        case 8: t.name = std::string(f.bytes); break;
        case 9: raw = f.bytes; break;
        case 13: {   // external_data: StringStringEntryProto {key = 1, value = 2}
            std::string k, v;
            each(f.bytes, [&](const field& e) { if (e.num == 1) k = std::string(e.bytes); else if (e.num == 2) v = std::string(e.bytes); });
            auto num = [&](const std::string& x) { std::uint64_t n = 0; auto [p, ec] = std::from_chars(x.data(), x.data() + x.size(), n); if (ec != std::errc() || p != x.data() + x.size()) raise(errc::format_error, "onnx: tensor {} has a bad external data {} '{}'", t.name, k, x.substr(0, 40)); return n; };
            if (k == "location") ext_location = v; else if (k == "offset") ext_offset = num(v); else if (k == "length") ext_length = num(v);
            external = true; break;
        }
        case 14: if (f.v == 1) external = true; break;   // data_location = EXTERNAL
        }
    });
    i64 n = 1; for (auto d : t.dims) { if (d < 0 || d > (i64(1) << 40)) raise(errc::format_error, "onnx: tensor {} has bad dims", t.name); n *= d; if (n > (i64(1) << 40)) raise(errc::format_error, "onnx: tensor {} too large", t.name); }
    auto width = [&]() -> std::size_t {
        switch (t.dtype) { case 1: case 6: case 12: return 4; case 11: case 7: case 13: return 8; case 10: case 5: return 2; case 9: case 2: case 3: return 1; default: return 0; }
    };
    if (external) {
        if (!width()) raise(errc::unsupported, "onnx: tensor {} has data type {} (supported: float, double, float16, int8/16/32/64, uint8/32/64, bool)", t.name, t.dtype);
        static const external_loader none;
        ext_bytes = (ext ? ext : &none)->load(t.name, ext_location, ext_offset, ext_length, std::uint64_t(n) * width());
        raw = ext_bytes;
    }
    auto take = [&](std::size_t w, auto conv) {
        if (raw.size() != std::size_t(n) * w) raise(errc::format_error, "onnx: tensor {} raw_data has {} bytes, expected {}", t.name, raw.size(), std::size_t(n) * w);
        for (std::size_t k = 0; k < std::size_t(n); ++k) conv(raw.data() + k * w);
    };
    switch (t.dtype) {
    case 1: if (!raw.empty()) take(4, [&](const char* p) { float x; std::memcpy(&x, p, 4); t.f.push_back(x); }); break;            // FLOAT
    case 11: if (!raw.empty()) take(8, [&](const char* p) { double x; std::memcpy(&x, p, 8); t.f.push_back(float(x)); }); break;  // DOUBLE
    case 7: if (!raw.empty()) take(8, [&](const char* p) { std::int64_t x; std::memcpy(&x, p, 8); t.i.push_back(x); }); break;    // INT64
    case 6: if (!raw.empty()) take(4, [&](const char* p) { std::int32_t x; std::memcpy(&x, p, 4); t.i.push_back(x); }); else t.i = i32; break;
    case 10: if (!raw.empty()) take(2, [&](const char* p) { std::uint16_t h; std::memcpy(&h, p, 2); t.f.push_back(float(f16::from_bits(h))); }); break;
    case 9: if (!raw.empty()) take(1, [&](const char* p) { t.i.push_back(*p != 0); }); else for (auto v : i32) t.i.push_back(v != 0); break;                       // BOOL
    case 2: if (!raw.empty()) take(1, [&](const char* p) { t.i.push_back(std::uint8_t(*p)); }); else t.i = i32; break;                                            // UINT8
    case 3: if (!raw.empty()) take(1, [&](const char* p) { t.i.push_back(std::int8_t(*p)); }); else t.i = i32; break;                                             // INT8
    case 5: if (!raw.empty()) take(2, [&](const char* p) { std::int16_t x; std::memcpy(&x, p, 2); t.i.push_back(x); }); else t.i = i32; break;                    // INT16
    case 12: if (!raw.empty()) take(4, [&](const char* p) { std::uint32_t x; std::memcpy(&x, p, 4); t.i.push_back(i64(x)); }); break;                             // UINT32 (uint64_data is not read)
    case 13: if (!raw.empty()) take(8, [&](const char* p) { std::uint64_t x; std::memcpy(&x, p, 8); if (x > std::uint64_t(1) << 62) raise(errc::format_error, "onnx: tensor {} holds a value out of range", t.name); t.i.push_back(i64(x)); }); break;
    default: raise(errc::unsupported, "onnx: tensor {} has data type {} (supported: float, double, float16, int8/16/32/64, uint8/32/64, bool)", t.name, t.dtype);
    }
    if (t.integer()) { if (i64(t.i.size()) != n) raise(errc::format_error, "onnx: tensor {} has {} values, expected {}", t.name, t.i.size(), n); }
    else if (i64(t.f.size()) != n) raise(errc::format_error, "onnx: tensor {} has {} values, expected {}", t.name, t.f.size(), n);
    return t;
}

inline attr_t parse_attr(std::string_view s, const external_loader* ext = nullptr) {
    attr_t a;
    each(s, [&](const field& f) {
        switch (f.num) {
        case 1: a.name = std::string(f.bytes); break;
        case 2: { std::uint32_t u = std::uint32_t(f.v); std::memcpy(&a.f, &u, 4); break; }
        case 3: a.i = i64(f.v); break;
        case 4: a.s = std::string(f.bytes); break;
        case 5: a.t = parse_tensor(f.bytes, ext); break;
        case 7: floats_of(f, a.fl); break;
        case 8: ints_of(f, a.il); break;
        case 20: a.type = int(f.v); break;
        }
    });
    return a;
}

inline vinfo_t parse_vinfo(std::string_view s) {
    vinfo_t v;
    each(s, [&](const field& f) {
        if (f.num == 1) v.name = std::string(f.bytes);
        else if (f.num == 2) each(f.bytes, [&](const field& tp) {           // TypeProto
            if (tp.num != 1) return;                                          // tensor_type
            each(tp.bytes, [&](const field& tt) {
                if (tt.num == 1) v.elem = int(tt.v);
                else if (tt.num == 2) each(tt.bytes, [&](const field& sh) {  // TensorShapeProto
                    if (sh.num != 1) return;
                    i64 val = -1; bool sym = true; std::string nm;
                    each(sh.bytes, [&](const field& d) { if (d.num == 1) { val = i64(d.v); sym = false; } else if (d.num == 2) nm = std::string(d.bytes); });
                    if (!sym && (val < 0 || val > (i64(1) << 40))) { val = -1; sym = true; }   // a negative dim_value is how some exporters write "unknown"
                    v.dims.push_back(val); v.symbolic.push_back(sym); v.names.push_back(sym ? nm : std::string());
                });
            });
        });
    });
    return v;
}

inline model_t parse_model(std::string_view s, const external_loader* ext = nullptr) {
    model_t m; bool graph = false;
    each(s, [&](const field& f) {
        if (f.num == 2) m.producer = std::string(f.bytes);
        else if (f.num == 6) m.doc = std::string(f.bytes);   // ModelProto.doc_string
        else if (f.num == 14) {   // metadata_props: StringStringEntryProto {key = 1, value = 2}
            std::string k, v;
            each(f.bytes, [&](const field& e) { if (e.num == 1) k = std::string(e.bytes); else if (e.num == 2) v = std::string(e.bytes); });
            m.metadata.emplace_back(std::move(k), std::move(v));
        } else if (f.num == 8) {  // opset_import
            std::string dom; i64 ver = 0;
            each(f.bytes, [&](const field& o) { if (o.num == 1) dom = std::string(o.bytes); else if (o.num == 2) ver = i64(o.v); });
            if (dom.empty() || dom == "ai.onnx") m.opset = ver;
        } else if (f.num == 7) {
            graph = true;
            each(f.bytes, [&](const field& g) {
                switch (g.num) {
                case 1: {
                    node_t n;
                    each(g.bytes, [&](const field& nf) {
                        switch (nf.num) {
                        case 1: n.in.emplace_back(nf.bytes); break;
                        case 2: n.out.emplace_back(nf.bytes); break;
                        case 3: n.name = std::string(nf.bytes); break;
                        case 4: n.op = std::string(nf.bytes); break;
                        case 5: { auto a = parse_attr(nf.bytes, ext); n.attrs[a.name] = std::move(a); break; }
                        case 7: n.domain = std::string(nf.bytes); break;
                        }
                    });
                    m.nodes.push_back(std::move(n)); break;
                }
                case 5: m.inits.push_back(parse_tensor(g.bytes, ext)); break;
                case 11: m.inputs.push_back(parse_vinfo(g.bytes)); break;
                case 12: m.outputs.push_back(parse_vinfo(g.bytes)); break;
                }
            });
        }
    });
    if (!graph) raise(errc::format_error, "onnx: no graph in model");
    return m;
}

} // namespace onnx_pb

// Translates an ONNX model to (blueprint text, weights).
struct onnx_translation {
    std::string blueprint;
    std::vector<std::pair<std::string, onnx_pb::tensor_t>> weights;
    std::set<std::string> exact;   // weights that must stay f32 (index tables, values the graph treats as integers)
    // [{onnx, name, shape, dims, dtype}]: shape has -1 for a symbolic dimension; dims is the same as blueprint
    // expressions ("seq", "dim_batch", "8"); dtype is "float" or "int" (integer and bool inputs are fed as numbers)
    json::array inputs, outputs;
    std::vector<std::string> symbols;   // the symbolic dimensions, in the order they were met ("seq" first)
};

namespace onnx_detail {

// A dimension: an integer times a product of symbolic dimensions (k, k * seq, k * seq * dim_batch).
struct mono {
    i64 k = 1; std::vector<std::string> s;
    mono() = default;
    mono(i64 v) : k(v) {}
    explicit mono(std::string sym) : s{std::move(sym)} {}
    bool is_const() const { return s.empty(); }
    bool operator==(const mono& o) const { return k == o.k && (k == 0 || s == o.s); }
    bool is(i64 v) const { return is_const() && k == v; }
    std::string text() const {
        if (s.empty() || k == 0) return std::to_string(k);
        std::string t = k == 1 ? std::string() : std::to_string(k);
        for (auto& x : s) t += (t.empty() ? "" : " * ") + x;
        return t;
    }
};
inline mono mul(const mono& a, const mono& b) {
    mono r;
    if (a.k != 0 && b.k != 0 && (std::abs(a.k) > (i64(1) << 53) / std::max<i64>(std::abs(b.k), 1))) raise(errc::format_error, "onnx: a dimension overflows");
    r.k = a.k * b.k; r.s = a.s; r.s.insert(r.s.end(), b.s.begin(), b.s.end()); std::sort(r.s.begin(), r.s.end());
    if (r.s.size() > 8) raise(errc::unsupported, "onnx: a dimension is a product of more than 8 symbolic dimensions");
    if (r.k == 0) r.s.clear();
    return r;
}
inline std::optional<mono> div(const mono& a, const mono& b) {   // exact division, else nullopt
    if (b.k == 0 || a.k % b.k) return std::nullopt;
    mono r; r.k = a.k / b.k; r.s = a.s;
    for (auto& x : b.s) { auto it = std::find(r.s.begin(), r.s.end(), x); if (it == r.s.end()) return std::nullopt; r.s.erase(it); }
    if (r.k == 0) r.s.clear();
    return r;
}
using shape = std::vector<mono>;
inline mono product(const shape& s) { mono r(1); for (auto& d : s) r = mul(r, d); return r; }
inline std::string list_text(const shape& s) { std::string t = "["; for (std::size_t i = 0; i < s.size(); ++i) t += (i ? ", " : "") + s[i].text(); return t + "]"; }
inline std::string ints_text(const std::vector<i64>& v) { std::string t = "["; for (std::size_t i = 0; i < v.size(); ++i) t += (i ? ", " : "") + std::to_string(v[i]); return t + "]"; }

// A value of the graph while it is translated.
//   tensor    a run-time value: a blueprint identifier and its shape
//   constant  known at import: an initializer or Constant (`w`), and for small ones (shape arithmetic, axes, scalars)
//             the numbers themselves (`v`), which the translator computes with
//   symbolic  a scalar or vector of dimensions, at least one of them symbolic: what Shape, Gather, Concat, ... of a
//             shape produce on the way to a Reshape or Expand
struct val {
    enum kind_t { none, tensor, constant, symbolic } kind = none;
    std::string expr;
    shape dims;                          // tensor shape; constant shape; symbolic: {} (scalar) or {n}
    bool integer = false;                // the values are integers (or booleans)
    std::vector<double> v; bool has_v = false;   // constant: folded values
    const onnx_pb::tensor_t* w = nullptr;         // constant: the source tensor
    std::vector<mono> sv;                // symbolic: the elements
    std::vector<i64> const_dims() const { std::vector<i64> d; for (auto& x : dims) d.push_back(x.k); return d; }
};

inline constexpr std::size_t fold_limit = 4096;   // constants up to this many elements are computed at import

} // namespace onnx_detail

class onnx_translator {
    using mono = onnx_detail::mono; using shape = onnx_detail::shape; using val = onnx_detail::val;
    using tensor_t = onnx_pb::tensor_t; using node_t = onnx_pb::node_t;
public:
    explicit onnx_translator(const onnx_pb::model_t& m) : m_(m) {}

    onnx_translation run() {
        for (auto& t : m_.inits) inits_[t.name] = &t;
        declare_inputs();
        for (auto& n : m_.nodes) {
            if (!n.domain.empty() && n.domain != "ai.onnx") fail(n, "uses the custom domain '" + n.domain + "'");
            if (n.out.empty()) continue;
            node(n);
        }
        std::string outs, outs_body;
        for (auto& o : m_.outputs) {
            val x = as_tensor(get(o.name), o.name);
            auto id = sanitize(o.name, "out_");
            outs_body += "  " + id + " = " + x.expr + "\n";
            // a declared dimension is one token: a number, a symbolic dimension, or `dyn` for a product of them (the
            // exact expression is in the manifest's onnx_outputs)
            std::string decl = "[";
            for (std::size_t i = 0; i < x.dims.size(); ++i) { const auto& d = x.dims[i]; decl += (i ? ", " : "") + (d.is_const() || (d.k == 1 && d.s.size() == 1) ? d.text() : std::string("dyn")); }
            outs += "output " + id + " : f32" + decl + "]\n";
            tr_.outputs.push_back(io_json(o.name, id, x.dims, x.integer));
        }
        if (m_.outputs.empty()) raise(errc::format_error, "onnx: the graph has no outputs");
        tr_.blueprint = std::format("# Translated from ONNX by forge import\nblueprint \"onnx\" opset {}\n\n{}\ngraph {{\n{}{}}}\n\n{}session one_shot\n", opset_, ins_, body_, outs_body, outs);
        tr_.symbols = symbols_;
        return std::move(tr_);
    }

private:
    // ---------------------------------------------------------------- errors
    [[noreturn]] void fail(const node_t& n, const std::string& why) const {
        raise(errc::unsupported, "onnx: operator '{}' (node {}, opset {}) {}", n.op, n.name.empty() ? "?" : n.name, m_.opset, why);
    }
    // ---------------------------------------------------------------- names
    std::string sanitize(const std::string& base, const char* prefix) {
        std::string s = prefix;
        for (char c : base.substr(0, 60)) s += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
        static const std::set<std::string> reserved{"W", "seq", "repeat", "in", "graph", "input", "output", "state", "params", "session", "blueprint", "opset", "op", "return", "true", "false", "dyn"};
        while (used_.count(s) || reserved.count(s)) s += "_";
        used_.insert(s); return s;
    }
    std::string symbol(const std::string& param) {   // the blueprint dimension for an ONNX dim_param ("" = unnamed)
        if (!param.empty()) if (auto it = sym_of_.find(param); it != sym_of_.end()) return it->second;
        std::string s = symbols_.empty() ? std::string("seq") : sanitize(param.empty() ? std::to_string(symbols_.size()) : param, "dim_");
        used_.insert(s); symbols_.push_back(s);
        if (!param.empty()) sym_of_[param] = s;
        return s;
    }
    json::value io_json(const std::string& onnx, const std::string& id, const shape& d, bool integer) const {
        json::array a, t; for (auto& x : d) { a.push_back(x.is_const() ? double(x.k) : -1.0); t.push_back(x.text()); }
        return json::object{{"onnx", onnx}, {"name", id}, {"shape", a}, {"dims", t}, {"dtype", integer ? "int" : "float"}};
    }
    void declare_inputs() {
        // `seq` is the dimension sessions bind by default: a model's only symbolic dimension, or with several the
        // one whose ONNX name starts with "seq" (sequence, seq_len, ...), else the first one met
        std::set<std::string> params;
        for (auto& in : m_.inputs) if (!inits_.count(in.name)) for (auto& p : in.names) if (!p.empty()) params.insert(p);
        for (auto& p : params) { std::string l = p; for (auto& c : l) c = char(std::tolower(static_cast<unsigned char>(c))); if (l.rfind("seq", 0) == 0 && symbols_.empty()) symbol(p); }
        for (auto& in : m_.inputs) {
            if (inits_.count(in.name)) continue;  // old exporters list initializers as inputs
            // float, double, float16; int8/16/32/64, uint8/32/64 and bool are fed as numbers
            static const std::set<int> floats{0, 1, 10, 11}, ints{2, 3, 5, 6, 7, 9, 12, 13};
            if (!floats.count(in.elem) && !ints.count(in.elem)) raise(errc::unsupported, "onnx: input {} has element type {} (supported: float types, integer types, bool)", in.name, in.elem);
            if (in.dims.size() > 8) raise(errc::unsupported, "onnx: input {} has rank {} (at most 8)", in.name, in.dims.size());
            val x; x.kind = val::tensor; x.integer = ints.count(in.elem) != 0; x.expr = sanitize(in.name, "in_");
            for (std::size_t i = 0; i < in.dims.size(); ++i) x.dims.push_back(in.symbolic[i] ? mono(symbol(i < in.names.size() ? in.names[i] : std::string())) : mono(in.dims[i]));
            ins_ += "input " + x.expr + " : " + (x.integer ? "i32" : "f32") + onnx_detail::list_text(x.dims) + "\n";
            tr_.inputs.push_back(io_json(in.name, x.expr, x.dims, x.integer));
            vals_[in.name] = std::move(x);
        }
    }
    // ---------------------------------------------------------------- values
    static val constant_of(const tensor_t& t) {
        val c; c.kind = val::constant; c.w = &t; c.integer = t.integer();
        for (auto d : t.dims) c.dims.push_back(mono(d));
        if (t.count() <= onnx_detail::fold_limit) { c.has_v = true; c.v.reserve(t.count()); for (std::size_t k = 0; k < t.count(); ++k) c.v.push_back(t.at(k)); }
        return c;
    }
    val folded(std::vector<i64> dims, std::vector<double> v, bool integer) {
        val c; c.kind = val::constant; c.integer = integer; c.has_v = true; c.v = std::move(v);
        for (auto d : dims) c.dims.push_back(mono(d));
        return c;
    }
    const val& get(const std::string& name) {
        if (auto it = vals_.find(name); it != vals_.end()) return it->second;
        if (auto it = inits_.find(name); it != inits_.end()) return vals_[name] = constant_of(*it->second);
        raise(errc::not_found, "onnx: value '{}' is used before it is defined", name);
    }
    // a weight for a constant; `exact` keeps it f32 whatever codec the import asks for
    std::string weight_of(const val& c, const std::string& hint, bool exact = false) {
        if (c.w) if (auto it = weight_id_.find(c.w); it != weight_id_.end()) { if (exact) tr_.exact.insert(it->second); return "W." + it->second; }
        tensor_t t; t.name = hint; t.dtype = 1;
        if (c.w) { t.dims = c.w->dims; if (c.w->integer()) { t.f.reserve(c.w->i.size()); for (auto x : c.w->i) t.f.push_back(float(x)); } else t.f = c.w->f; }
        else { for (auto& d : c.dims) t.dims.push_back(d.k); t.f.reserve(c.v.size()); for (auto x : c.v) t.f.push_back(float(x)); }
        if (t.dims.empty()) t.dims = {1};   // scalars broadcast as [1]
        auto id = "w_" + std::to_string(tr_.weights.size());
        if (c.w) weight_id_[c.w] = id;
        if (exact || c.integer) tr_.exact.insert(id);
        tr_.weights.emplace_back(id, std::move(t));
        return "W." + id;
    }
    // any value as a run-time tensor (constants become weights; a scalar becomes [1])
    val as_tensor(const val& x, const std::string& hint) {
        if (x.kind == val::tensor) return x;
        if (x.kind == val::symbolic) {
            for (auto& e : x.sv) if (!e.is_const()) raise(errc::unsupported, "onnx: value '{}' is computed from a symbolic dimension and then used as data; only shapes may depend on symbolic dimensions", hint);
            std::vector<double> v; for (auto& e : x.sv) v.push_back(double(e.k));
            return as_tensor(folded(x.dims.empty() ? std::vector<i64>{} : std::vector<i64>{i64(v.size())}, v, true), hint);
        }
        if (x.kind != val::constant) raise(errc::not_found, "onnx: value '{}' has no definition", hint);
        val t; t.kind = val::tensor; t.integer = x.integer; t.expr = weight_of(x, hint);
        t.dims = x.dims.empty() ? shape{mono(1)} : x.dims;
        return t;
    }
    std::string emit(const std::string& onnx_name, const std::string& expr) {
        auto id = sanitize(onnx_name, "t_");
        body_ += "  " + id + " = " + expr + "\n";
        if (++lines_ > 200000) raise(errc::unsupported, "onnx: the graph expands to more than 200000 blueprint statements");
        return id;
    }
    void set_tensor(const std::string& onnx_name, const std::string& expr, shape dims, bool integer = false) {
        if (dims.size() > 8) raise(errc::unsupported, "onnx: value '{}' has rank {} (at most 8)", onnx_name, dims.size());
        val t; t.kind = val::tensor; t.expr = emit(onnx_name, expr); t.dims = std::move(dims); t.integer = integer;
        vals_[onnx_name] = std::move(t);
    }
    // op(args...) as blueprint text; records the opset the op needs
    std::string F(const char* op, std::initializer_list<std::string> args) {
        opset_ = std::max(opset_, modelfile::op_min_opset(op));
        std::string s = std::string(op) + "(";
        bool first = true; for (auto& a : args) { if (a.empty()) continue; s += (first ? "" : ", ") + a; first = false; }
        return s + ")";
    }
    static std::string num(double v) { return std::format("{:.9g}", v); }
    std::string reshape(const std::string& e, const shape& to) { return F("reshape", {e, "shape: " + onnx_detail::list_text(to)}); }
    std::string scalar_weight(double v, const std::string& hint) { return weight_of(folded({1}, {v}, false), hint); }

    // ---------------------------------------------------------------- attribute and operand helpers
    static i64 attr_i(const node_t& n, const char* k, i64 d) { auto it = n.attrs.find(k); return it == n.attrs.end() ? d : it->second.i; }
    static double attr_f(const node_t& n, const char* k, double d) { auto it = n.attrs.find(k); return it == n.attrs.end() ? d : double(it->second.f); }
    static std::string attr_s(const node_t& n, const char* k, const char* d) { auto it = n.attrs.find(k); return it == n.attrs.end() ? std::string(d) : it->second.s; }
    static std::vector<i64> attr_ints(const node_t& n, const char* k) { auto it = n.attrs.find(k); return it == n.attrs.end() ? std::vector<i64>{} : it->second.il; }
    bool has_in(const node_t& n, std::size_t i) const { return i < n.in.size() && !n.in[i].empty(); }
    const val& in(const node_t& n, std::size_t i) {
        if (!has_in(n, i)) raise(errc::format_error, "onnx: {} node {} is missing input {}", n.op, n.name, i);
        return get(n.in[i]);
    }
    val T(const node_t& n, std::size_t i) { return as_tensor(in(n, i), n.in[i]); }
    // the integers of a constant operand (axes, pads, shapes, ...)
    std::vector<i64> ints(const node_t& n, std::size_t i, const char* what) {
        const val& c = in(n, i);
        if (c.kind == val::symbolic) { std::vector<i64> r; for (auto& e : c.sv) { if (!e.is_const()) fail(n, std::string("needs a constant ") + what + "; it depends on a symbolic dimension"); r.push_back(e.k); } return r; }
        if (c.kind != val::constant || !c.has_v) fail(n, std::string("needs a constant ") + what + " (input '" + n.in[i] + "' is computed at run time)");
        std::vector<i64> r; for (auto x : c.v) { if (!(std::fabs(x) < 9e18)) fail(n, std::string("has a bad ") + what); r.push_back(i64(x)); }
        return r;
    }
    std::optional<double> scalar(const node_t& n, std::size_t i) {
        if (!has_in(n, i)) return std::nullopt;
        const val& c = in(n, i);
        if (c.kind != val::constant || !c.has_v || c.v.size() != 1) return std::nullopt;
        return c.v[0];
    }
    static i64 norm_axis(const node_t&, i64 ax, i64 rank) { if (ax < 0) ax += rank; if (ax < 0 || ax >= std::max<i64>(rank, 1)) raise(errc::format_error, "onnx: axis out of range"); return ax; }
    std::vector<i64> statics(const node_t& n, const shape& s, const char* what) {
        std::vector<i64> r; for (auto& d : s) { if (!d.is_const()) fail(n, std::string("needs static ") + what + ", got " + onnx_detail::list_text(s)); r.push_back(d.k); }
        return r;
    }
    shape bcast(const node_t& n, const shape& a, const shape& b) {
        shape big = a.size() >= b.size() ? a : b; const shape& small = a.size() >= b.size() ? b : a;
        for (std::size_t i = 0; i < small.size(); ++i) {
            auto& x = big[big.size() - small.size() + i]; const auto& y = small[i];
            if (x == y || y.is(1)) continue;
            if (x.is(1)) { x = y; continue; }
            fail(n, "cannot broadcast " + onnx_detail::list_text(a) + " with " + onnx_detail::list_text(b));
        }
        return big;
    }

    // ---------------------------------------------------------------- folding: constants
    // N-d constant helpers (row-major)
    static std::vector<i64> strides_of(const std::vector<i64>& d) { std::vector<i64> s(d.size(), 1); for (std::size_t i = d.size(); i-- > 1;) s[i - 1] = s[i] * d[i]; return s; }
    static i64 count_of(const std::vector<i64>& d) { i64 n = 1; for (auto x : d) { if (x < 0 || (x && n > i64(onnx_detail::fold_limit) * 16 / x)) return -1; n *= x; } return n; }
    bool fold_constants(const node_t& n) {   // every input is a constant with values; true when the node was computed
        std::vector<const val*> xs;
        for (std::size_t i = 0; i < n.in.size(); ++i) { if (n.in[i].empty()) { xs.push_back(nullptr); continue; } const val& c = get(n.in[i]); if (c.kind != val::constant || !c.has_v) return false; xs.push_back(&c); }
        auto X = [&](std::size_t i) -> const val& { if (i >= xs.size() || !xs[i]) raise(errc::format_error, "onnx: {} node {} is missing input {}", n.op, n.name, i); return *xs[i]; };
        auto out = [&](std::vector<i64> dims, std::vector<double> v, bool integer) { if (v.size() > onnx_detail::fold_limit) return false; vals_[n.out[0]] = folded(std::move(dims), std::move(v), integer); return true; };
        const std::string& op = n.op;
        static const std::set<std::string> bin{"Add", "Sub", "Mul", "Div", "Pow", "Min", "Max", "Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual", "And", "Or", "Mod"};
        if (bin.count(op) && xs.size() == 2) {
            const val& a = X(0); const val& b = X(1);
            auto da = a.const_dims(), db = b.const_dims(); std::size_t r = std::max(da.size(), db.size());
            std::vector<i64> d(r, 1);
            for (std::size_t i = 0; i < r; ++i) {
                i64 x = i + da.size() >= r ? da[i + da.size() - r] : 1, y = i + db.size() >= r ? db[i + db.size() - r] : 1;
                if (x != y && x != 1 && y != 1) return false;
                d[i] = x == 1 ? y : x;
            }
            i64 cnt = count_of(d); if (cnt < 0 || cnt > i64(onnx_detail::fold_limit)) return false;
            auto sa = strides_of(da), sb = strides_of(db), so = strides_of(d);
            std::vector<double> v(static_cast<std::size_t>(cnt));
            const bool both_int = a.integer && b.integer; bool integer = both_int;
            for (i64 k = 0; k < cnt; ++k) {
                i64 ia = 0, ib = 0;
                for (std::size_t i = 0; i < r; ++i) {
                    i64 c = (k / so[i]) % std::max<i64>(d[i], 1);
                    if (i + da.size() >= r) { auto j = i + da.size() - r; ia += (da[j] == 1 ? 0 : c) * sa[j]; }
                    if (i + db.size() >= r) { auto j = i + db.size() - r; ib += (db[j] == 1 ? 0 : c) * sb[j]; }
                }
                double x = a.v[std::size_t(ia)], y = b.v[std::size_t(ib)], z;
                if (op == "Add") z = x + y; else if (op == "Sub") z = x - y; else if (op == "Mul") z = x * y;
                else if (op == "Div") { if (both_int) { if (y == 0) return false; z = std::trunc(x / y); } else z = x / y; }
                else if (op == "Pow") z = std::pow(x, y); else if (op == "Min") z = std::min(x, y); else if (op == "Max") z = std::max(x, y);
                else if (op == "Mod") { if (!both_int || y == 0 || attr_i(n, "fmod", 0)) return false; z = x - std::floor(x / y) * y; }
                else { integer = true; z = op == "Equal" ? x == y : op == "Less" ? x < y : op == "Greater" ? x > y : op == "LessOrEqual" ? x <= y : op == "GreaterOrEqual" ? x >= y : op == "And" ? (x != 0 && y != 0) : (x != 0 || y != 0); }
                v[std::size_t(k)] = z;
            }
            return out(d, std::move(v), integer);
        }
        static const std::set<std::string> un{"Neg", "Abs", "Sqrt", "Floor", "Ceil", "Not", "Identity", "Cast", "Exp", "Log", "Reciprocal"};
        if (un.count(op) && xs.size() >= 1) {
            const val& a = X(0); std::vector<double> v = a.v; bool integer = a.integer;
            if (op == "Cast") {
                i64 to = attr_i(n, "to", 1); static const std::set<i64> its{2, 3, 5, 6, 7, 12, 13};
                if (to == 9) { for (auto& x : v) x = x != 0; integer = true; }
                else if (its.count(to)) { for (auto& x : v) x = std::trunc(x); integer = true; }
                else integer = false;
            } else if (op == "Neg") for (auto& x : v) x = -x;
            else if (op == "Abs") for (auto& x : v) x = std::fabs(x);
            else if (op == "Not") { for (auto& x : v) x = x == 0; integer = true; }
            else if (op == "Floor") for (auto& x : v) x = std::floor(x);
            else if (op == "Ceil") for (auto& x : v) x = std::ceil(x);
            else if (op != "Identity") { integer = false; for (auto& x : v) x = op == "Sqrt" ? std::sqrt(x) : op == "Exp" ? std::exp(x) : op == "Log" ? std::log(x) : 1.0 / x; }
            return out(a.const_dims(), std::move(v), integer);
        }
        if (op == "Shape") { auto d = X(0).const_dims(); std::vector<double> v(d.begin(), d.end()); const i64 nv = i64(v.size()); return out({nv}, std::move(v), true); }
        if (op == "Size") { return out({}, {double(X(0).v.size())}, true); }
        if (op == "Unsqueeze" || op == "Squeeze" || op == "Reshape" || op == "Flatten") {
            const val& a = X(0); auto d = a.const_dims(); std::vector<i64> nd;
            if (op == "Reshape") {
                auto want = X(1).v; i64 known = 1, neg = -1;
                for (std::size_t i = 0; i < want.size(); ++i) { i64 w = i64(want[i]); if (w == 0 && !attr_i(n, "allowzero", 0)) { if (i >= d.size()) return false; w = d[i]; } if (w == -1) { neg = i64(i); nd.push_back(-1); continue; } if (w < 0) return false; known *= w; nd.push_back(w); }
                if (neg >= 0) { if (!known || i64(a.v.size()) % known) return false; nd[std::size_t(neg)] = i64(a.v.size()) / known; }
            } else if (op == "Flatten") {
                i64 ax = attr_i(n, "axis", 1); if (ax < 0) ax += i64(d.size()); if (ax < 0 || ax > i64(d.size())) return false;
                i64 p = 1, q = 1; for (i64 i = 0; i < i64(d.size()); ++i) (i < ax ? p : q) *= d[std::size_t(i)];
                nd = {p, q};
            } else {
                std::vector<i64> axes = n.attrs.count("axes") ? attr_ints(n, "axes") : xs.size() > 1 && xs[1] ? [&] { std::vector<i64> r; for (auto x : X(1).v) r.push_back(i64(x)); return r; }() : std::vector<i64>{};
                if (op == "Unsqueeze") {
                    i64 r = i64(d.size() + axes.size()); std::vector<char> is_new(std::size_t(r), 0);
                    for (auto ax : axes) { if (ax < 0) ax += r; if (ax < 0 || ax >= r || is_new[std::size_t(ax)]) return false; is_new[std::size_t(ax)] = 1; }
                    std::size_t j = 0; for (i64 i = 0; i < r; ++i) nd.push_back(is_new[std::size_t(i)] ? 1 : d[j++]);
                } else {
                    std::vector<char> drop(d.size(), 0);
                    if (axes.empty()) for (std::size_t i = 0; i < d.size(); ++i) drop[i] = d[i] == 1;
                    for (auto ax : axes) { if (ax < 0) ax += i64(d.size()); if (ax < 0 || ax >= i64(d.size()) || d[std::size_t(ax)] != 1) return false; drop[std::size_t(ax)] = 1; }
                    for (std::size_t i = 0; i < d.size(); ++i) if (!drop[i]) nd.push_back(d[i]);
                }
            }
            if (count_of(nd) != i64(a.v.size())) return false;
            return out(nd, a.v, a.integer);
        }
        if (op == "Concat") {
            const val& a0 = X(0); auto d = a0.const_dims(); if (d.empty()) return false;
            i64 ax = attr_i(n, "axis", 0); if (ax < 0) ax += i64(d.size()); if (ax < 0 || ax >= i64(d.size())) return false;
            // concatenate along ax: outer blocks of each input in turn
            i64 outer = 1; for (i64 i = 0; i < ax; ++i) outer *= d[std::size_t(i)];
            std::vector<double> v; bool integer = true; i64 total = 0;
            for (std::size_t j = 0; j < xs.size(); ++j) { auto dj = X(j).const_dims(); if (dj.size() != d.size()) return false; for (std::size_t i = 0; i < d.size(); ++i) if (i64(i) != ax && dj[i] != d[i]) return false; total += dj[std::size_t(ax)]; integer = integer && X(j).integer; }
            for (i64 o = 0; o < outer; ++o) for (std::size_t j = 0; j < xs.size(); ++j) { const val& a = X(j); i64 blk = outer ? i64(a.v.size()) / outer : 0; v.insert(v.end(), a.v.begin() + o * blk, a.v.begin() + (o + 1) * blk); if (v.size() > onnx_detail::fold_limit) return false; }
            d[std::size_t(ax)] = total;
            return out(d, std::move(v), integer);
        }
        if (op == "Gather") {
            const val& a = X(0); const val& ix = X(1); auto d = a.const_dims(); if (d.empty()) return false;
            i64 ax = attr_i(n, "axis", 0); if (ax < 0) ax += i64(d.size()); if (ax < 0 || ax >= i64(d.size())) return false;
            i64 outer = 1, inner = 1; for (i64 i = 0; i < ax; ++i) outer *= d[std::size_t(i)]; for (std::size_t i = std::size_t(ax) + 1; i < d.size(); ++i) inner *= d[i];
            std::vector<double> v;
            for (i64 o = 0; o < outer; ++o) for (auto xi : ix.v) { i64 k = i64(xi); if (k < 0) k += d[std::size_t(ax)]; if (k < 0 || k >= d[std::size_t(ax)]) return false; auto b = a.v.begin() + (o * d[std::size_t(ax)] + k) * inner; v.insert(v.end(), b, b + inner); if (v.size() > onnx_detail::fold_limit) return false; }
            std::vector<i64> nd(d.begin(), d.begin() + ax); auto id = ix.const_dims(); nd.insert(nd.end(), id.begin(), id.end()); nd.insert(nd.end(), d.begin() + ax + 1, d.end());
            return out(nd, std::move(v), a.integer);
        }
        if (op == "Slice") {
            const val& a = X(0); auto d = a.const_dims(); if (d.size() != 1) return false;   // shape arithmetic slices vectors
            std::vector<i64> st, en, axs, steps;
            if (m_.opset < 10) { st = attr_ints(n, "starts"); en = attr_ints(n, "ends"); axs = attr_ints(n, "axes"); }
            else { for (auto x : X(1).v) st.push_back(i64(x)); for (auto x : X(2).v) en.push_back(i64(x)); if (xs.size() > 3 && xs[3]) for (auto x : X(3).v) axs.push_back(i64(x)); if (xs.size() > 4 && xs[4]) for (auto x : X(4).v) steps.push_back(i64(x)); }
            if (st.size() != 1 || en.size() != 1 || (!axs.empty() && axs[0] != 0 && axs[0] != -1) || (!steps.empty() && steps[0] != 1)) return false;
            i64 len = d[0], s0 = st[0] < 0 ? st[0] + len : st[0], e0 = en[0] < 0 ? en[0] + len : en[0];
            s0 = std::clamp<i64>(s0, 0, len); e0 = std::clamp<i64>(e0, s0, len);
            return out({e0 - s0}, std::vector<double>(a.v.begin() + s0, a.v.begin() + e0), a.integer);
        }
        if (op == "Range") {
            double s = X(0).v.at(0), l = X(1).v.at(0), d = X(2).v.at(0); if (d == 0) return false;
            double cnt = std::max(0.0, std::ceil((l - s) / d)); if (!(cnt <= double(onnx_detail::fold_limit))) return false;
            std::vector<double> v; for (i64 k = 0; k < i64(cnt); ++k) v.push_back(s + double(k) * d);
            const i64 nv = i64(v.size());   // before the call: argument order is unspecified (GCC moves v first)
            return out({nv}, std::move(v), X(0).integer && X(1).integer && X(2).integer);
        }
        if (op == "ConstantOfShape") {
            std::vector<i64> d; for (auto x : X(0).v) d.push_back(i64(x));
            i64 cnt = count_of(d); if (cnt < 0 || cnt > i64(onnx_detail::fold_limit)) return false;
            double fill = 0; bool integer = false;
            if (auto it = n.attrs.find("value"); it != n.attrs.end() && it->second.t && it->second.t->count() == 1) { fill = it->second.t->at(0); integer = it->second.t->integer(); }
            return out(d, std::vector<double>(std::size_t(cnt), fill), integer);
        }
        if (op == "Where" && xs.size() == 3) {
            const val& c = X(0); const val& a = X(1); const val& b = X(2);
            if (c.v.size() != a.v.size() || a.v.size() != b.v.size()) { if (c.v.size() == 1) { vals_[n.out[0]] = c.v[0] != 0 ? a : b; return true; } return false; }
            std::vector<double> v(a.v.size()); for (std::size_t i = 0; i < v.size(); ++i) v[i] = c.v[i] != 0 ? a.v[i] : b.v[i];
            return out(a.const_dims(), std::move(v), a.integer && b.integer);
        }
        if (op == "ReduceProd" && xs.size() >= 1 && !n.attrs.count("axes") && (xs.size() < 2 || !xs[1])) {
            double p = 1; for (auto x : X(0).v) p *= x;
            return out(attr_i(n, "keepdims", 1) ? std::vector<i64>(X(0).dims.size(), 1) : std::vector<i64>{}, {p}, X(0).integer);
        }
        return false;
    }

    // ---------------------------------------------------------------- folding: shapes with symbolic dimensions
    val sym(std::vector<mono> elems, bool scalar_) {
        bool all_const = true; for (auto& e : elems) all_const = all_const && e.is_const();
        if (all_const) { std::vector<double> v; for (auto& e : elems) v.push_back(double(e.k)); const i64 nv = i64(v.size()); return folded(scalar_ ? std::vector<i64>{} : std::vector<i64>{nv}, std::move(v), true); }
        val s; s.kind = val::symbolic; s.integer = true; s.sv = std::move(elems);
        if (!scalar_) s.dims = {mono(i64(s.sv.size()))};
        return s;
    }
    // a constant or symbolic value as a list of dimensions (rank 0 or 1)
    std::optional<std::pair<std::vector<mono>, bool>> as_dims(const val& x) {
        if (x.kind == val::symbolic) return std::pair{x.sv, x.dims.empty()};
        if (x.kind == val::constant && x.has_v && x.dims.size() <= 1) { std::vector<mono> r; for (auto v : x.v) { if (v != std::trunc(v) || std::fabs(v) > 9e15) return std::nullopt; r.push_back(mono(i64(v))); } return std::pair{r, x.dims.empty()}; }
        return std::nullopt;
    }
    bool fold_symbolic(const node_t& n) {   // at least one input is symbolic
        const std::string& op = n.op;
        auto D = [&](std::size_t i) { auto d = as_dims(in(n, i)); if (!d) fail(n, "mixes a shape that depends on a symbolic dimension with run-time data (input '" + n.in[i] + "')"); return *d; };
        auto set = [&](std::vector<mono> e, bool scalar_) { vals_[n.out[0]] = sym(std::move(e), scalar_); return true; };
        if (op == "Identity" || op == "Cast") { auto [e, sc] = D(0); return set(e, sc); }
        if (op == "Unsqueeze") { auto [e, sc] = D(0); if (!sc || e.size() != 1) fail(n, "of a shape vector"); return set(e, false); }
        if (op == "Squeeze") { auto [e, sc] = D(0); if (e.size() != 1) fail(n, "of a shape vector with more than one element"); return set(e, true); }
        if (op == "Concat") { std::vector<mono> all; for (std::size_t i = 0; i < n.in.size(); ++i) { auto [e, sc] = D(i); all.insert(all.end(), e.begin(), e.end()); } return set(all, false); }
        if (op == "Gather") {
            auto [e, sc] = D(0); const val& ix = in(n, 1);
            if (ix.kind != val::constant || !ix.has_v || ix.dims.size() > 1) fail(n, "of a shape needs constant indices");
            std::vector<mono> r; for (auto v : ix.v) { i64 k = i64(v); if (k < 0) k += i64(e.size()); if (k < 0 || k >= i64(e.size())) raise(errc::format_error, "onnx: Gather index out of range"); r.push_back(e[std::size_t(k)]); }
            return set(r, ix.dims.empty());
        }
        if (op == "Slice") {
            auto [e, sc] = D(0); std::vector<i64> st, en, steps;
            if (m_.opset < 10) { st = attr_ints(n, "starts"); en = attr_ints(n, "ends"); } else { st = ints(n, 1, "starts"); en = ints(n, 2, "ends"); if (has_in(n, 4)) steps = ints(n, 4, "steps"); }
            if (st.size() != 1 || en.size() != 1 || (!steps.empty() && steps[0] != 1)) fail(n, "of a shape with several axes or a step");
            i64 len = i64(e.size()), s0 = st[0] < 0 ? st[0] + len : st[0], e0 = en[0] < 0 ? en[0] + len : en[0];
            s0 = std::clamp<i64>(s0, 0, len); e0 = std::clamp<i64>(e0, s0, len);
            return set(std::vector<mono>(e.begin() + s0, e.begin() + e0), false);
        }
        if (op == "Mul" || op == "Div" || op == "Add" || op == "Sub") {
            auto [a, sa] = D(0); auto [b, sb] = D(1);
            if (a.size() != b.size() && a.size() != 1 && b.size() != 1) fail(n, "of shapes of different lengths");
            std::size_t cnt = std::max(a.size(), b.size()); std::vector<mono> r;
            for (std::size_t i = 0; i < cnt; ++i) {
                const mono& x = a[a.size() == 1 ? 0 : i]; const mono& y = b[b.size() == 1 ? 0 : i];
                if (op == "Mul") r.push_back(onnx_detail::mul(x, y));
                else if (op == "Div") { auto q = onnx_detail::div(x, y); if (!q) fail(n, "divides the dimension " + x.text() + " by " + y.text() + ", which is not exact"); r.push_back(*q); }
                else if (x.is_const() && y.is_const()) r.push_back(mono(op == "Add" ? x.k + y.k : x.k - y.k));
                else if (y.is(0)) r.push_back(x);
                else if (x.is(0) && op == "Add") r.push_back(y);
                else fail(n, "adds to or subtracts from a symbolic dimension (" + x.text() + ", " + y.text() + "); dimensions can be multiplied and divided, not offset");
            }
            return set(r, sa && sb);
        }
        if (op == "ReduceProd") { auto [e, sc] = D(0); return set({onnx_detail::product(e)}, !attr_i(n, "keepdims", 1)); }
        if (op == "Reshape") { auto [e, sc] = D(0); auto to = ints(n, 1, "shape"); if (to.size() > 1) fail(n, "of a shape vector into a matrix"); return set(e, to.empty()); }
        fail(n, "is applied to a shape that depends on a symbolic dimension; shape arithmetic supports Shape, Size, Gather, Slice, Concat, Unsqueeze, Squeeze, Cast, Mul, Div, ReduceProd");
    }

    // ---------------------------------------------------------------- the nodes
    // Convolution-style ops take [c, ...] in FORGE and [n, c, ...] in ONNX: batch 1 is reshaped away and back.
    struct nchw { std::string expr; i64 c = 0; std::vector<i64> sp; bool lifted = false; };   // lifted: a [c, l] input made [c, 1, l]
    nchw unbatch(const node_t& n, const val& x, bool lift_1d) {
        if (x.dims.size() < 3 || x.dims.size() > 4) fail(n, "is supported for [n, c, l] and [n, c, h, w] inputs, got " + onnx_detail::list_text(x.dims));
        if (!x.dims[0].is(1)) fail(n, "needs batch size 1 (the input is " + onnx_detail::list_text(x.dims) + "); export the model with a fixed batch of 1");
        auto d = statics(n, shape(x.dims.begin() + 1, x.dims.end()), "channel and spatial dimensions");
        nchw r; r.c = d[0]; r.sp.assign(d.begin() + 1, d.end());
        shape to{mono(r.c)};
        if (r.sp.size() == 1 && lift_1d) { r.sp.insert(r.sp.begin(), 1); r.lifted = true; }
        for (auto s : r.sp) to.push_back(mono(s));
        r.expr = reshape(x.expr, to);
        return r;
    }
    void rebatch(const node_t& n, const std::string& expr, i64 c, std::vector<i64> sp, bool lifted) {
        shape out{mono(1), mono(c)};
        for (std::size_t i = lifted ? 1 : 0; i < sp.size(); ++i) out.push_back(mono(sp[i]));
        set_tensor(n.out[0], reshape(expr, out), out);
    }
    // ONNX pads [b0, b1, ..., e0, e1, ...] -> per-axis (begin, end); auto_pad resolved from the static sizes
    std::vector<std::pair<i64, i64>> conv_pads(const node_t& n, const std::vector<i64>& sp, const std::vector<i64>& k, const std::vector<i64>& st, const std::vector<i64>& dl) {
        std::vector<std::pair<i64, i64>> p(sp.size(), {0, 0});
        auto ap = attr_s(n, "auto_pad", "NOTSET");
        if (ap == "SAME_UPPER" || ap == "SAME_LOWER") {
            for (std::size_t i = 0; i < sp.size(); ++i) {
                i64 out = (sp[i] + st[i] - 1) / st[i], total = std::max<i64>((out - 1) * st[i] + dl[i] * (k[i] - 1) + 1 - sp[i], 0);
                p[i] = ap == "SAME_UPPER" ? std::pair{total / 2, total - total / 2} : std::pair{total - total / 2, total / 2};
            }
        } else if (ap == "NOTSET" || ap.empty()) {
            auto raw = attr_ints(n, "pads");
            if (!raw.empty()) {
                std::size_t r = raw.size() / 2;   // a [c, l] op lifted to [c, 1, l] has one pair for the last axis
                if (raw.size() % 2 || r > sp.size()) fail(n, "has a pads attribute of the wrong length");
                for (std::size_t i = 0; i < r; ++i) p[sp.size() - r + i] = {raw[i], raw[r + i]};
            }
        } else if (ap != "VALID") fail(n, "has auto_pad '" + ap + "'");
        for (auto& [b, e] : p) if (b < 0 || e < 0 || b > 4096 || e > 4096) fail(n, "has negative or absurd padding");
        return p;
    }
    static std::vector<i64> spatial_attr(const node_t& n, const char* k, std::size_t dims, i64 dflt) {
        auto v = attr_ints(n, k);
        if (v.empty()) return std::vector<i64>(dims, dflt);
        while (v.size() < dims) v.insert(v.begin(), dflt);   // lifted 1-d
        if (v.size() != dims) raise(errc::format_error, "onnx: attribute {} of {} has the wrong length", k, n.name);
        for (auto x : v) if (x < 1 || x > 65536) raise(errc::format_error, "onnx: attribute {} of {} is out of range", k, n.name);
        return v;
    }
    // explicit padding in front of an op whose own pad attr cannot express it; returns the padded expr and sizes
    std::string pre_pad(const std::string& e, std::vector<i64>& sp, const std::vector<std::pair<i64, i64>>& p, double value) {
        std::vector<i64> pairs;   // from the last axis backwards
        for (std::size_t i = sp.size(); i-- > 0;) { pairs.push_back(p[i].first); pairs.push_back(p[i].second); }
        for (std::size_t i = 0; i < sp.size(); ++i) sp[i] += p[i].first + p[i].second;
        return F("pad", {e, "pads: " + onnx_detail::ints_text(pairs), "mode: 0", "value: " + num(value)});
    }

    void node(const node_t& n) {
        const std::string& op = n.op; const std::string& o = n.out[0];
        // ---- constants and shapes first
        if (op == "Constant") {
            if (auto it = n.attrs.find("value"); it != n.attrs.end() && it->second.t) { owned_.push_back(*it->second.t); owned_.back().name = o; vals_[o] = constant_of(owned_.back()); return; }
            if (auto it = n.attrs.find("value_float"); it != n.attrs.end()) { vals_[o] = folded({}, {double(it->second.f)}, false); return; }
            if (auto it = n.attrs.find("value_int"); it != n.attrs.end()) { vals_[o] = folded({}, {double(it->second.i)}, true); return; }
            if (auto it = n.attrs.find("value_floats"); it != n.attrs.end() && it->second.fl.size() <= onnx_detail::fold_limit) { vals_[o] = folded({i64(it->second.fl.size())}, std::vector<double>(it->second.fl.begin(), it->second.fl.end()), false); return; }
            if (auto it = n.attrs.find("value_ints"); it != n.attrs.end() && it->second.il.size() <= onnx_detail::fold_limit) { vals_[o] = folded({i64(it->second.il.size())}, std::vector<double>(it->second.il.begin(), it->second.il.end()), true); return; }
            fail(n, "has no tensor, float or int value");
        }
        if (op == "Shape") {   // of anything: a tensor's tracked shape, a constant's, a shape vector's own length
            const shape d = in(n, 0).dims;
            i64 r = i64(d.size()), s = attr_i(n, "start", 0), e = n.attrs.count("end") ? attr_i(n, "end", r) : r;
            if (s < 0) s += r;
            if (e < 0) e += r;
            s = std::clamp<i64>(s, 0, r); e = std::clamp<i64>(e, s, r);
            vals_[o] = sym(std::vector<mono>(d.begin() + s, d.begin() + e), false); return;
        }
        if (op == "Size") { const shape d = in(n, 0).dims; vals_[o] = sym({onnx_detail::product(d)}, true); return; }
        bool any_tensor = false, any_sym = false, all_small = true;
        for (auto& nm : n.in) { if (nm.empty()) continue; const val& x = get(nm); any_tensor = any_tensor || x.kind == val::tensor; any_sym = any_sym || x.kind == val::symbolic; all_small = all_small && x.kind == val::constant && x.has_v; }
        static const std::set<std::string> shape_users{"Reshape", "Expand", "ConstantOfShape", "Resize", "Slice", "Pad", "Tile"};   // take a shape as an operand next to data
        // shape arithmetic, unless the op applies a shape to data (operand 0: a tensor or a constant)
        const bool first_sym = has_in(n, 0) && get(n.in[0]).kind == val::symbolic;
        if (any_sym && op != "ConstantOfShape" && (!shape_users.count(op) || first_sym)) { fold_symbolic(n); return; }
        if (!any_tensor && !any_sym && all_small && !n.in.empty() && fold_constants(n)) return;

        // ---- run-time ops
        static const std::map<std::string, const char*> unary{
            {"Relu", "relu"}, {"Sigmoid", "sigmoid"}, {"Tanh", "tanh"}, {"Exp", "exp"}, {"Log", "log"}, {"Sqrt", "sqrt"}, {"Neg", "neg"}, {"Abs", "abs"}, {"Erf", "erf"},
            {"Softplus", "softplus"}, {"Reciprocal", "recip"}, {"Sin", "sin"}, {"Cos", "cos"}, {"Tan", "tan"}, {"Asin", "asin"}, {"Acos", "acos"}, {"Atan", "atan"},
            {"Sinh", "sinh"}, {"Cosh", "cosh"}, {"Floor", "floor"}, {"Mish", "mish"}, {"HardSwish", "hard_swish"}};
        static const std::map<std::string, const char*> binary{{"Add", "add"}, {"Sub", "sub"}, {"Mul", "mul"}, {"Div", "div"}, {"Pow", "pow"}};
        if (unary.count(op)) { auto x = T(n, 0); set_tensor(o, F(unary.at(op), {x.expr}), x.dims, x.integer && (op == "Neg" || op == "Abs" || op == "Relu" || op == "Floor")); return; }
        if (op == "Identity" || op == "Dropout") { vals_[o] = in(n, 0); return; }
        if (op == "Cast") {
            const val& x = in(n, 0); i64 to = attr_i(n, "to", 1);
            static const std::set<i64> fl{1, 10, 11, 16}, its{2, 3, 5, 6, 7, 9, 12, 13};
            if (fl.count(to)) { val y = x; y.integer = false; vals_[o] = y; return; }           // the numbers do not change
            if (!its.count(to)) fail(n, "to data type " + std::to_string(to));
            if (x.integer && to != 9) { vals_[o] = x; return; }
            fail(n, to == 9 ? "to bool of a run-time tensor (there is no comparison op)" : "to an integer type of a run-time float tensor (there is no truncation op); integer inputs and constants cast freely");
        }
        if (op == "Ceil") { auto x = T(n, 0); set_tensor(o, F("neg", {F("floor", {F("neg", {x.expr})})}), x.dims, x.integer); return; }
        if (op == "Gelu") { auto x = T(n, 0); set_tensor(o, F(attr_s(n, "approximate", "none") == "tanh" ? "gelu_tanh" : "gelu", {x.expr}), x.dims); return; }
        if (op == "LeakyRelu") { auto x = T(n, 0); double a = attr_f(n, "alpha", 0.01); if (!(a >= 0 && a <= 1)) fail(n, "with alpha outside [0, 1]"); set_tensor(o, F("maximum", {x.expr, F("scale", {x.expr, "alpha: " + num(a)})}), x.dims); return; }
        if (op == "PRelu") { auto x = T(n, 0); auto s = T(n, 1); auto d = bcast(n, x.dims, s.dims); set_tensor(o, F("sub", {F("relu", {x.expr}), F("mul", {s.expr, F("relu", {F("neg", {x.expr})})})}), d); return; }
        if (op == "Elu" || op == "Selu" || op == "Celu") {
            auto x = T(n, 0); double a = attr_f(n, "alpha", op == "Selu" ? 1.6732632423543772 : 1.0), g = op == "Selu" ? attr_f(n, "gamma", 1.0507009873554805) : 1.0;
            // relu(x) + alpha * (exp(min(x, 0) / c) - 1), c = alpha for Celu
            std::string neg = F("minimum", {x.expr, scalar_weight(0, o + ".zero")});
            if (op == "Celu") neg = F("scale", {neg, "alpha: " + num(1.0 / a)});
            std::string e = F("add", {F("relu", {x.expr}), F("scale", {F("expm1", {neg}), "alpha: " + num(a)})});
            if (g != 1.0) e = F("scale", {e, "alpha: " + num(g)});
            set_tensor(o, e, x.dims); return;
        }
        if (op == "HardSigmoid") { auto x = T(n, 0); double a = attr_f(n, "alpha", 0.2), b = attr_f(n, "beta", 0.5);
            set_tensor(o, F("minimum", {F("maximum", {F("add", {F("scale", {x.expr, "alpha: " + num(a)}), scalar_weight(b, o + ".beta")}), scalar_weight(0, o + ".zero")}), scalar_weight(1, o + ".one")}), x.dims); return; }
        if (op == "Softsign") { auto x = T(n, 0); set_tensor(o, F("div", {x.expr, F("add", {F("abs", {x.expr}), scalar_weight(1, o + ".one")})}), x.dims); return; }
        if (op == "Clip") {
            auto x = T(n, 0); std::optional<double> lo, hi;
            if (m_.opset < 11) { if (n.attrs.count("min")) lo = attr_f(n, "min", 0); if (n.attrs.count("max")) hi = attr_f(n, "max", 0); }
            else { if (has_in(n, 1)) { lo = scalar(n, 1); if (!lo) fail(n, "needs a constant scalar min"); } if (has_in(n, 2)) { hi = scalar(n, 2); if (!hi) fail(n, "needs a constant scalar max"); } }
            std::string e = x.expr;
            if (lo) e = F("maximum", {e, scalar_weight(*lo, o + ".min")});
            if (hi) e = F("minimum", {e, scalar_weight(*hi, o + ".max")});
            if (!lo && !hi) { vals_[o] = x; return; }
            set_tensor(o, e, x.dims, x.integer); return;
        }
        if (binary.count(op)) {
            auto sb = scalar(n, 1);
            const val& a0 = in(n, 0); const val& b0 = in(n, 1);
            const bool integer = a0.integer && b0.integer;
            if (integer && op == "Div") fail(n, "of two integer tensors at run time (integer division)");
            if (sb && op == "Mul") { auto x = T(n, 0); set_tensor(o, F("scale", {x.expr, "alpha: " + num(*sb)}), x.dims, integer); return; }
            if (sb && op == "Div" && *sb != 0) { auto x = T(n, 0); set_tensor(o, F("scale", {x.expr, "alpha: " + num(1.0 / *sb)}), x.dims); return; }
            auto x = T(n, 0); auto y = T(n, 1);
            set_tensor(o, F(binary.at(op), {x.expr, y.expr}), bcast(n, x.dims, y.dims), integer && op != "Pow"); return;
        }
        if (op == "Max" || op == "Min" || op == "Sum" || op == "Mean") {
            if (n.in.empty()) fail(n, "has no inputs");
            auto acc = T(n, 0); const char* f = op == "Max" ? "maximum" : op == "Min" ? "minimum" : "add";
            for (std::size_t i = 1; i < n.in.size(); ++i) { auto y = T(n, i); acc.dims = bcast(n, acc.dims, y.dims); acc.expr = F(f, {acc.expr, y.expr}); }
            if (op == "Mean") acc.expr = F("scale", {acc.expr, "alpha: " + num(1.0 / double(n.in.size()))});
            if (n.in.size() == 1 && op != "Mean") { vals_[o] = acc; return; }
            set_tensor(o, acc.expr, acc.dims); return;
        }
        if (op == "Where") {
            auto c = T(n, 0); auto a = T(n, 1); auto b = T(n, 2);
            set_tensor(o, F("select", {a.expr, b.expr, c.expr}), bcast(n, bcast(n, a.dims, b.dims), c.dims), a.integer && b.integer); return;
        }
        if (op == "MatMul") {
            auto a = T(n, 0); auto b = T(n, 1);
            if (b.dims.size() < 2 || a.dims.size() < 2) fail(n, "needs operands of rank 2 or more");
            if (!(a.dims.back() == b.dims[b.dims.size() - 2])) fail(n, "inner dimensions differ: " + onnx_detail::list_text(a.dims) + " @ " + onnx_detail::list_text(b.dims));
            if (b.dims.size() > 2 && (b.dims.size() != a.dims.size() || !std::equal(b.dims.begin(), b.dims.end() - 2, a.dims.begin()))) fail(n, "broadcasts batch dimensions (" + onnx_detail::list_text(a.dims) + " @ " + onnx_detail::list_text(b.dims) + ")");
            shape os = a.dims; os.back() = b.dims.back();
            set_tensor(o, F("matmul", {a.expr, b.expr}), os); return;
        }
        if (op == "Gemm") {
            if (attr_i(n, "transA", 0)) fail(n, "with transA=1");
            bool tb = attr_i(n, "transB", 0); auto a = T(n, 0); auto b = T(n, 1);
            if (a.dims.size() != 2 || b.dims.size() != 2) fail(n, "needs rank-2 operands");
            std::string e = F(tb ? "matmul_t" : "matmul", {a.expr, b.expr});
            shape os{a.dims[0], tb ? b.dims[0] : b.dims[1]};
            if (double al = attr_f(n, "alpha", 1); al != 1) e = F("scale", {e, "alpha: " + num(al)});
            if (has_in(n, 2)) {
                double be = attr_f(n, "beta", 1); auto c = T(n, 2); std::string ce = c.expr;
                if (be != 1) ce = F("scale", {ce, "alpha: " + num(be)});
                e = F("add", {e, ce});
            }
            set_tensor(o, e, os); return;
        }
        if (op == "Softmax" || op == "LogSoftmax") {
            auto x = T(n, 0); i64 r = i64(x.dims.size());
            i64 ax = norm_axis(n, attr_i(n, "axis", m_.opset >= 13 ? -1 : 1), r);
            if (m_.opset < 13 && ax != r - 1) {   // before opset 13 the axes from `axis` on are flattened together
                bool trailing_ones = true; for (i64 i = ax + 1; i < r; ++i) trailing_ones = trailing_ones && x.dims[std::size_t(i)].is(1);
                if (!trailing_ones) fail(n, "over the flattened axes from " + std::to_string(ax) + " (the pre-13 definition); re-export with opset 13 or later");
            }
            std::string e = x.expr; std::vector<i64> perm;
            if (ax != r - 1) { for (i64 i = 0; i < r; ++i) perm.push_back(i); std::swap(perm[std::size_t(ax)], perm[std::size_t(r - 1)]); e = F("transpose", {e, "perm: " + onnx_detail::ints_text(perm)}); }
            e = F("softmax", {e});
            if (op == "LogSoftmax") e = F("log", {e});
            if (!perm.empty()) e = F("transpose", {e, "perm: " + onnx_detail::ints_text(perm)});
            set_tensor(o, e, x.dims); return;
        }
        if (op == "LayerNormalization") {
            auto x = T(n, 0);
            if (norm_axis(n, attr_i(n, "axis", -1), i64(x.dims.size())) != i64(x.dims.size()) - 1) fail(n, "must normalise the last axis only");
            auto g = T(n, 1); std::string b;
            if (has_in(n, 2)) b = T(n, 2).expr;
            else { auto c = statics(n, {x.dims.back()}, "last dimension"); b = weight_of(folded({c[0]}, std::vector<double>(std::size_t(std::min<i64>(c[0], 1 << 24)), 0.0), false), o + ".zero_bias"); }
            set_tensor(o, F("layer_norm", {x.expr, g.expr, b, "eps: " + num(attr_f(n, "epsilon", 1e-5))}), x.dims); return;
        }
        if (op == "LpNormalization") {
            auto x = T(n, 0);
            if (attr_i(n, "p", 2) != 2 || norm_axis(n, attr_i(n, "axis", -1), i64(x.dims.size())) != i64(x.dims.size()) - 1) fail(n, "is supported for p=2 over the last axis");
            set_tensor(o, F("l2_normalize", {x.expr, "eps: 0"}), x.dims); return;
        }
        if (op == "BatchNormalization") {
            if (attr_i(n, "training_mode", 0)) fail(n, "in training mode");
            auto x = T(n, 0);
            if (x.dims.size() < 2) fail(n, "needs an input of rank 2 or more");
            const val* p[4]; for (int i = 0; i < 4; ++i) { p[i] = &in(n, std::size_t(i + 1)); if (p[i]->kind != val::constant || !p[i]->w) fail(n, "needs constant scale, bias, mean and variance"); }
            std::size_t C = p[0]->w->count();
            for (auto* q : p) if (q->w->count() != C || q->w->integer()) fail(n, "has scale, bias, mean and variance of different sizes");
            if (!x.dims[1].is(i64(C))) fail(n, "has " + std::to_string(C) + " channels of statistics for an input of " + onnx_detail::list_text(x.dims));
            // folded at import: y = x * s + t, s = w / sqrt(var + eps), t = b - mean * s, as [C, 1, ..., 1]
            double eps = attr_f(n, "epsilon", 1e-5);
            tensor_t s, t; s.dtype = t.dtype = 1; s.dims = t.dims = shape_t(x.dims.size() - 1, 1); s.dims[0] = t.dims[0] = i64(C);
            for (std::size_t c = 0; c < C; ++c) { double sc = double(p[0]->w->f[c]) / std::sqrt(double(p[3]->w->f[c]) + eps); s.f.push_back(float(sc)); t.f.push_back(float(double(p[1]->w->f[c]) - double(p[2]->w->f[c]) * sc)); }
            auto sid = "w_" + std::to_string(tr_.weights.size()); tr_.weights.emplace_back(sid, std::move(s));
            auto tid = "w_" + std::to_string(tr_.weights.size()); tr_.weights.emplace_back(tid, std::move(t));
            set_tensor(o, F("add", {F("mul", {x.expr, "W." + sid}), "W." + tid}), x.dims); return;
        }
        if (op == "InstanceNormalization" || op == "GroupNormalization") {
            auto x = T(n, 0);
            if (x.dims.size() < 3 || !x.dims[0].is(1)) fail(n, "needs an [1, c, ...] input, got " + onnx_detail::list_text(x.dims));
            auto d = statics(n, shape(x.dims.begin() + 1, x.dims.end()), "channel and spatial dimensions");
            i64 groups = op == "InstanceNormalization" ? d[0] : attr_i(n, "num_groups", 1);
            if (groups < 1 || d[0] % groups) fail(n, "has a group count that does not divide the channels");
            std::string w, b;
            const val& ws = in(n, 1); const val& bs = in(n, 2);
            if (op == "GroupNormalization" && ws.kind == val::constant && ws.w && bs.kind == val::constant && bs.w && i64(ws.w->count()) == groups && groups != d[0]) {
                // opset 18 gives scale and bias per group: repeat them per channel
                tensor_t wt, bt; wt.dtype = bt.dtype = 1; wt.dims = bt.dims = {d[0]};
                if (bs.w->count() != ws.w->count() || ws.w->integer() || bs.w->integer()) fail(n, "has scale and bias of different sizes");
                for (i64 c = 0; c < d[0]; ++c) { wt.f.push_back(ws.w->f[std::size_t(c / (d[0] / groups))]); bt.f.push_back(bs.w->f[std::size_t(c / (d[0] / groups))]); }
                auto wi = "w_" + std::to_string(tr_.weights.size()); tr_.weights.emplace_back(wi, std::move(wt));
                auto bi = "w_" + std::to_string(tr_.weights.size()); tr_.weights.emplace_back(bi, std::move(bt));
                w = "W." + wi; b = "W." + bi;
            } else { auto wv = T(n, 1); auto bv = T(n, 2); if (!wv.dims.back().is(d[0]) || !bv.dims.back().is(d[0])) fail(n, "needs per-channel scale and bias"); w = wv.expr; b = bv.expr; }
            shape inner; for (auto v : d) inner.push_back(mono(v));
            set_tensor(o, reshape(F("group_norm", {reshape(x.expr, inner), w, b, "groups: " + std::to_string(groups), "eps: " + num(attr_f(n, "epsilon", 1e-5))}), x.dims), x.dims); return;
        }
        if (op == "Conv" || op == "ConvTranspose") {
            auto x = T(n, 0); const val& wv = in(n, 1);
            if (wv.kind != val::constant || !wv.w || wv.w->integer()) fail(n, "needs a constant float weight");
            const bool tr = op == "ConvTranspose";
            auto u = unbatch(n, x, false);
            const auto& wd = wv.w->dims;
            if (wd.size() != u.sp.size() + 2) fail(n, "has a weight of rank " + std::to_string(wd.size()) + " for " + onnx_detail::list_text(x.dims));
            if (tr && u.sp.size() != 2) fail(n, "is supported for 2-d inputs");
            std::vector<i64> k(wd.begin() + 2, wd.end());
            auto st = spatial_attr(n, "strides", u.sp.size(), 1), dl = spatial_attr(n, "dilations", u.sp.size(), 1);
            i64 g = attr_i(n, "group", 1);
            if (g < 1 || u.c % g || (!tr && (wd[1] * g != u.c || wd[0] % g)) || (tr && wd[0] != u.c)) fail(n, "has a weight " + shape_str(wd) + " that does not fit " + std::to_string(u.c) + " channels in " + std::to_string(g) + " groups");
            for (auto kk : k) if (kk < 1) fail(n, "has an empty kernel");
            auto pads = conv_pads(n, u.sp, k, st, dl);
            std::string wexpr = weight_of(wv, n.in[1]);
            const i64 O = tr ? wd[1] * g : wd[0];
            std::string bexpr = has_in(n, 2) ? T(n, 2).expr : scalar_weight(0, o + ".zero_bias");
            std::string e = u.expr; std::vector<i64> sp = u.sp, out;
            bool sym_pad = true; for (auto& [b, e2] : pads) sym_pad = sym_pad && b == e2;
            if (tr) {
                if (!sym_pad || attr_s(n, "auto_pad", "NOTSET") != "NOTSET" || n.attrs.count("output_shape")) fail(n, "with asymmetric pads, auto_pad or output_shape");
                auto op_ = attr_ints(n, "output_padding"); if (op_.empty()) op_ = {0, 0};
                if (op_.size() != 2) fail(n, "has an output_padding of the wrong length");
                for (std::size_t i = 0; i < 2; ++i) { out.push_back((sp[i] - 1) * st[i] - 2 * pads[i].first + dl[i] * (k[i] - 1) + op_[i] + 1); if (out.back() < 1) fail(n, "has an empty output"); }
                e = F("conv_transpose2d", {e, wexpr, bexpr, "stride: " + onnx_detail::ints_text(st), "pad: " + onnx_detail::ints_text({pads[0].first, pads[1].first}),
                                           "output_pad: " + onnx_detail::ints_text(op_), "dilation: " + onnx_detail::ints_text(dl), "groups: " + std::to_string(g)});
            } else {
                if (!sym_pad) { e = pre_pad(e, sp, pads, 0); for (auto& p : pads) p = {0, 0}; }
                for (std::size_t i = 0; i < sp.size(); ++i) { i64 v = (sp[i] + 2 * pads[i].first - dl[i] * (k[i] - 1) - 1) / st[i] + 1; if (sp[i] + 2 * pads[i].first < dl[i] * (k[i] - 1) + 1 || v < 1) fail(n, "has a kernel larger than its padded input"); out.push_back(v); }
                if (sp.size() == 1) e = F("conv1d", {e, wexpr, bexpr, "stride: " + std::to_string(st[0]), "pad: " + std::to_string(pads[0].first), "dilation: " + std::to_string(dl[0]), "groups: " + std::to_string(g)});
                else e = F("conv2d", {e, wexpr, bexpr, "stride: " + onnx_detail::ints_text(st), "pad: " + onnx_detail::ints_text({pads[0].first, pads[1].first}), "dilation: " + onnx_detail::ints_text(dl), "groups: " + std::to_string(g)});
            }
            rebatch(n, e, O, out, false); return;
        }
        if (op == "MaxPool" || op == "AveragePool") {
            if (n.out.size() > 1 && !n.out[1].empty()) fail(n, "with the Indices output");
            auto x = T(n, 0); auto u = unbatch(n, x, true);
            auto k = spatial_attr(n, "kernel_shape", u.sp.size(), 1);
            if (!n.attrs.count("kernel_shape")) fail(n, "has no kernel_shape");
            auto st = spatial_attr(n, "strides", u.sp.size(), 1), dl = spatial_attr(n, "dilations", u.sp.size(), 1);
            for (auto d : dl) if (d != 1) fail(n, "with dilations");
            auto pads = conv_pads(n, u.sp, k, st, dl);
            const bool avg = op == "AveragePool"; const bool cip = attr_i(n, "count_include_pad", 0) != 0;
            std::string e = u.expr; std::vector<i64> sp = u.sp, out;
            bool simple = true; for (std::size_t i = 0; i < pads.size(); ++i) simple = simple && pads[i].first == pads[i].second && 2 * pads[i].first <= k[i];
            if (!simple) {
                if (avg && !cip) fail(n, "with asymmetric padding that is excluded from the average");
                e = pre_pad(e, sp, pads, avg ? 0.0 : -3.4028234663852886e38); for (auto& p : pads) p = {0, 0};
            }
            for (std::size_t i = 0; i < sp.size(); ++i) {
                i64 span = sp[i] + 2 * pads[i].first - k[i]; if (span < 0) fail(n, "has a kernel larger than its padded input");
                i64 fl = span / st[i] + 1, ce = (span + st[i] - 1) / st[i] + 1;
                if (attr_i(n, "ceil_mode", 0) && ce != fl) fail(n, "with ceil_mode=1 where it changes the output size");
                out.push_back(fl);
            }
            std::vector<std::string> a{e, "kernel: " + onnx_detail::ints_text(k), "stride: " + onnx_detail::ints_text(st), "pad: " + onnx_detail::ints_text({pads[0].first, pads[1].first})};
            if (avg) e = F("avg_pool2d", {a[0], a[1], a[2], a[3], std::string("count_pad: ") + (cip || !simple ? "1" : "0")}); else e = F("max_pool2d", {a[0], a[1], a[2], a[3]});
            rebatch(n, e, u.c, out, u.lifted); return;
        }
        if (op == "GlobalAveragePool" || op == "GlobalMaxPool") {
            auto x = T(n, 0); auto u = unbatch(n, x, true);
            std::string e = op == "GlobalAveragePool" ? F("global_avg_pool", {u.expr}) : F("max_pool2d", {u.expr, "kernel: " + onnx_detail::ints_text(u.sp), "stride: " + onnx_detail::ints_text(u.sp), "pad: [0, 0]"});
            shape out{mono(1), mono(u.c)}; for (std::size_t i = u.lifted ? 1 : 0; i < u.sp.size(); ++i) out.push_back(mono(1));
            set_tensor(o, reshape(e, out), out); return;
        }
        if (op == "Resize" || op == "Upsample") {
            auto x = T(n, 0); auto u = unbatch(n, x, false);
            if (u.sp.size() != 2) fail(n, "is supported for [1, c, h, w] inputs");
            std::vector<double> scales; std::vector<i64> sizes;
            if (op == "Upsample") { if (m_.opset < 9) for (auto f : n.attrs.count("scales") ? n.attrs.at("scales").fl : std::vector<float>{}) scales.push_back(f); else { const val& s = in(n, 1); if (s.kind != val::constant || !s.has_v) fail(n, "needs constant scales"); scales = s.v; } }
            else {
                std::size_t si = m_.opset >= 11 ? 2 : 1;
                if (has_in(n, si)) { const val& s = in(n, si); if (s.kind != val::constant || !s.has_v) fail(n, "needs constant scales"); scales = s.v; }
                if (scales.empty() && has_in(n, 3)) sizes = ints(n, 3, "sizes");
            }
            i64 sc[2];
            if (!scales.empty()) {
                if (scales.size() != 4 || scales[0] != 1 || scales[1] != 1) fail(n, "must leave the batch and channel axes alone");
                for (int i = 0; i < 2; ++i) { double s = scales[std::size_t(2 + i)]; if (!(s >= 1 && s <= 64) || s != std::floor(s)) fail(n, "with scale " + num(s) + " (integer upscaling only)"); sc[i] = i64(s); }
            } else if (sizes.size() == 4) {
                if (sizes[0] != 1 || sizes[1] != u.c) fail(n, "must leave the batch and channel axes alone");
                for (int i = 0; i < 2; ++i) { i64 t = sizes[std::size_t(2 + i)]; if (t < u.sp[std::size_t(i)] || t % u.sp[std::size_t(i)] || t / u.sp[std::size_t(i)] > 64) fail(n, "to a size that is not an integer multiple of the input"); sc[i] = t / u.sp[std::size_t(i)]; }
            } else fail(n, "needs constant scales or sizes for all four axes");
            auto mode = attr_s(n, "mode", "nearest"), ctm = attr_s(n, "coordinate_transformation_mode", op == "Upsample" || m_.opset < 11 ? "asymmetric" : "half_pixel"), nm = attr_s(n, "nearest_mode", "round_prefer_floor");
            std::string e;
            if (mode == "nearest") {
                // which source pixel an output pixel takes: floor(i / scale) for these combinations
                bool ok = (ctm == "asymmetric" && (nm == "floor" || op == "Upsample" || m_.opset < 11)) || ((ctm == "half_pixel" || ctm == "pytorch_half_pixel") && (nm == "round_prefer_floor" || nm == "round_prefer_ceil"));
                if (!ok) fail(n, "nearest with coordinate_transformation_mode '" + ctm + "' and nearest_mode '" + nm + "'");
                e = F("upsample_nearest", {u.expr, "scale: " + onnx_detail::ints_text({sc[0], sc[1]})});
            } else if (mode == "linear") {
                int ac = ctm == "align_corners" ? 1 : (ctm == "half_pixel" || ctm == "pytorch_half_pixel") ? 0 : -1;
                if (ac < 0) fail(n, "linear with coordinate_transformation_mode '" + ctm + "'");
                e = F("upsample_bilinear", {u.expr, "scale: " + onnx_detail::ints_text({sc[0], sc[1]}), "align_corners: " + std::to_string(ac)});
            } else fail(n, "with mode '" + mode + "'");
            rebatch(n, e, u.c, {u.sp[0] * sc[0], u.sp[1] * sc[1]}, false); return;
        }
        if (op == "Transpose") {
            auto x = T(n, 0); std::vector<i64> perm = attr_ints(n, "perm"); shape os;
            if (perm.empty()) for (i64 i = i64(x.dims.size()) - 1; i >= 0; --i) perm.push_back(i);
            if (perm.size() != x.dims.size()) fail(n, "has a perm of the wrong length");
            std::vector<char> seen(perm.size(), 0);
            for (auto p : perm) { if (p < 0 || p >= i64(perm.size()) || seen[std::size_t(p)]) fail(n, "has a perm that is not a permutation"); seen[std::size_t(p)] = 1; os.push_back(x.dims[std::size_t(p)]); }
            set_tensor(o, F("transpose", {x.expr, "perm: " + onnx_detail::ints_text(perm)}), os, x.integer); return;
        }
        if (op == "Reshape" || op == "Flatten" || op == "Squeeze" || op == "Unsqueeze") {
            auto x = T(n, 0); shape to; const shape& s = x.dims;
            if (op == "Flatten") {
                i64 ax = attr_i(n, "axis", 1); if (ax < 0) ax += i64(s.size()); if (ax < 0 || ax > i64(s.size())) fail(n, "has an axis out of range");
                to = {onnx_detail::product(shape(s.begin(), s.begin() + ax)), onnx_detail::product(shape(s.begin() + ax, s.end()))};
            } else if (op == "Reshape") {
                auto want = as_dims(in(n, 1));
                if (!want) fail(n, "needs a shape that is a constant or computed from input shapes (input '" + n.in[1] + "' is run-time data)");
                int neg = -1; mono known(1);
                for (std::size_t i = 0; i < want->first.size(); ++i) {
                    mono w = want->first[i];
                    if (w.is(0) && !attr_i(n, "allowzero", 0)) { if (i >= s.size()) fail(n, "copies a dimension the input does not have"); w = s[i]; }
                    if (w.is(-1)) { if (neg >= 0) fail(n, "has two -1 entries"); neg = int(i); to.push_back(mono(1)); continue; }
                    if (w.k < 0) fail(n, "has a negative dimension");
                    known = onnx_detail::mul(known, w); to.push_back(w);
                }
                if (neg >= 0) { auto q = onnx_detail::div(onnx_detail::product(s), known); if (!q) fail(n, "cannot work out the -1 dimension: " + onnx_detail::product(s).text() + " elements into " + known.text()); to[std::size_t(neg)] = *q; }
                else if (!(onnx_detail::product(s) == known)) fail(n, "changes the element count: " + onnx_detail::list_text(s) + " -> " + onnx_detail::list_text(to));
            } else {
                std::vector<i64> axes = n.attrs.count("axes") ? attr_ints(n, "axes") : has_in(n, 1) ? ints(n, 1, "axes") : std::vector<i64>{};
                if (op == "Unsqueeze") {
                    i64 r = i64(s.size() + axes.size()); std::vector<char> is_new(std::size_t(r), 0);
                    for (auto ax : axes) { if (ax < 0) ax += r; if (ax < 0 || ax >= r || is_new[std::size_t(ax)]) fail(n, "has bad axes"); is_new[std::size_t(ax)] = 1; }
                    std::size_t j = 0; for (i64 i = 0; i < r; ++i) to.push_back(is_new[std::size_t(i)] ? mono(1) : s[j++]);
                } else {
                    std::vector<char> drop(s.size(), 0);
                    if (axes.empty()) for (std::size_t i = 0; i < s.size(); ++i) drop[i] = s[i].is(1);
                    for (auto ax : axes) { if (ax < 0) ax += i64(s.size()); if (ax < 0 || ax >= i64(s.size()) || !s[std::size_t(ax)].is(1)) fail(n, "squeezes an axis that is not 1"); drop[std::size_t(ax)] = 1; }
                    for (std::size_t i = 0; i < s.size(); ++i) if (!drop[i]) to.push_back(s[i]);
                }
            }
            if (to.empty()) to = {mono(1)};   // FORGE has no rank-0 tensors: a scalar is [1]
            set_tensor(o, reshape(x.expr, to), to, x.integer); return;
        }
        if (op == "Expand") {
            auto x = T(n, 0); auto want = as_dims(in(n, 1));
            if (!want) fail(n, "needs a constant or shape-derived target shape");
            shape cur = x.dims; shape target = want->first; std::string e = x.expr;
            if (target.size() > 8) fail(n, "to rank " + std::to_string(target.size()));
            if (cur.size() < target.size()) { shape padded(target.size() - cur.size(), mono(1)); padded.insert(padded.end(), cur.begin(), cur.end()); cur = padded; e = reshape(e, cur); }
            for (std::size_t i = 0; i < target.size(); ++i) {
                auto& c = cur[cur.size() - target.size() + i]; const auto& t = target[i];
                if (c == t || t.is(1)) continue;
                if (!c.is(1)) fail(n, "cannot broadcast " + onnx_detail::list_text(x.dims) + " to " + onnx_detail::list_text(target));
                e = F("expand", {e, "axis: " + std::to_string(cur.size() - target.size() + i), "n: " + t.text()}); c = t;
            }
            if (e == x.expr) { vals_[o] = x; return; }
            set_tensor(o, e, cur, x.integer); return;
        }
        if (op == "Concat") {
            if (n.in.empty()) fail(n, "has no inputs");
            auto acc = T(n, 0); i64 r = i64(acc.dims.size()); i64 ax = norm_axis(n, attr_i(n, "axis", 0), r); bool integer = acc.integer;
            for (std::size_t i = 1; i < n.in.size(); ++i) {
                auto y = T(n, i);
                if (i64(y.dims.size()) != r) fail(n, "of tensors of different ranks");
                for (i64 d = 0; d < r; ++d) if (d != ax && !(y.dims[std::size_t(d)] == acc.dims[std::size_t(d)])) fail(n, "of " + onnx_detail::list_text(acc.dims) + " and " + onnx_detail::list_text(y.dims));
                auto& a = acc.dims[std::size_t(ax)]; const auto& b = y.dims[std::size_t(ax)];
                if (!a.is_const() || !b.is_const()) fail(n, "along a symbolic dimension (" + a.text() + " + " + b.text() + " is not a product)");
                a = mono(a.k + b.k); integer = integer && y.integer;
                acc.expr = F("concat", {acc.expr, y.expr, "axis: " + std::to_string(ax)});
            }
            if (n.in.size() == 1) { vals_[o] = acc; return; }
            set_tensor(o, acc.expr, acc.dims, integer); return;
        }
        if (op == "Slice") {
            auto x = T(n, 0); std::vector<i64> st, en, axs, steps;
            if (m_.opset < 10) { st = attr_ints(n, "starts"); en = attr_ints(n, "ends"); axs = attr_ints(n, "axes"); }
            else { st = ints(n, 1, "starts"); en = ints(n, 2, "ends"); if (has_in(n, 3)) axs = ints(n, 3, "axes"); if (has_in(n, 4)) steps = ints(n, 4, "steps"); }
            if (st.size() != en.size() || (!axs.empty() && axs.size() != st.size()) || (!steps.empty() && steps.size() != st.size())) fail(n, "has starts, ends, axes and steps of different lengths");
            std::string e = x.expr; shape os = x.dims;
            for (std::size_t i = 0; i < st.size(); ++i) {
                i64 ax = norm_axis(n, axs.empty() ? i64(i) : axs[i], i64(os.size()));
                if (!steps.empty() && steps[i] != 1) fail(n, "with step " + std::to_string(steps[i]));
                auto& d = os[std::size_t(ax)];
                const bool to_end = en[i] >= (i64(1) << 31);
                if (!d.is_const()) { if (st[i] == 0 && to_end) continue; fail(n, "of part of a symbolic dimension (" + d.text() + ")"); }
                i64 len = d.k, s0 = st[i] < 0 ? st[i] + len : st[i], e0 = en[i] < 0 ? en[i] + len : en[i];
                s0 = std::clamp<i64>(s0, 0, len); e0 = std::clamp<i64>(e0, s0, len);
                if (s0 == 0 && e0 == len) continue;
                e = F("slice", {e, "axis: " + std::to_string(ax), "start: " + std::to_string(s0), "end: " + std::to_string(e0)}); d = mono(e0 - s0);
            }
            if (e == x.expr) { vals_[o] = x; return; }
            set_tensor(o, e, os, x.integer); return;
        }
        if (op == "Split") {
            auto x = T(n, 0); i64 ax = norm_axis(n, attr_i(n, "axis", 0), i64(x.dims.size()));
            auto len = statics(n, {x.dims[std::size_t(ax)]}, "size along the split axis")[0];
            std::vector<i64> parts = n.attrs.count("split") ? attr_ints(n, "split") : has_in(n, 1) ? ints(n, 1, "split sizes") : std::vector<i64>{};
            if (parts.empty()) { i64 k = n.attrs.count("num_outputs") ? attr_i(n, "num_outputs", 1) : i64(n.out.size()); if (k < 1 || k > 4096) fail(n, "into " + std::to_string(k) + " parts"); i64 each_ = (len + k - 1) / k; for (i64 i = 0; i < k; ++i) parts.push_back(std::min(each_, len - i * each_)); }
            if (parts.size() != n.out.size()) fail(n, "has " + std::to_string(parts.size()) + " sizes for " + std::to_string(n.out.size()) + " outputs");
            i64 at = 0;
            for (std::size_t i = 0; i < parts.size(); ++i) {
                if (parts[i] < 0 || at + parts[i] > len) fail(n, "has sizes that do not fit the axis");
                shape os = x.dims; os[std::size_t(ax)] = mono(parts[i]);
                if (!n.out[i].empty()) set_tensor(n.out[i], F("slice", {x.expr, "axis: " + std::to_string(ax), "start: " + std::to_string(at), "end: " + std::to_string(at + parts[i])}), os, x.integer);
                at += parts[i];
            }
            if (at != len) fail(n, "has sizes that do not cover the axis");
            return;
        }
        if (op == "Pad") {
            auto x = T(n, 0); std::vector<i64> pads; double value = 0; i64 r = i64(x.dims.size());
            if (m_.opset < 11) { pads = attr_ints(n, "pads"); value = attr_f(n, "value", 0); }
            else { pads = ints(n, 1, "pads"); if (has_in(n, 2)) { auto v = scalar(n, 2); if (!v) fail(n, "needs a constant value"); value = *v; } }
            std::vector<std::pair<i64, i64>> per(std::size_t(r), {0, 0});
            if (has_in(n, 3)) { auto axes = ints(n, 3, "axes"); if (pads.size() != 2 * axes.size()) fail(n, "has pads and axes of different lengths"); for (std::size_t i = 0; i < axes.size(); ++i) per[std::size_t(norm_axis(n, axes[i], r))] = {pads[i], pads[axes.size() + i]}; }
            else { if (i64(pads.size()) != 2 * r) fail(n, "has " + std::to_string(pads.size()) + " pads for rank " + std::to_string(r)); for (i64 i = 0; i < r; ++i) per[std::size_t(i)] = {pads[std::size_t(i)], pads[std::size_t(r + i)]}; }
            auto mode = attr_s(n, "mode", "constant"); int mi = mode == "constant" ? 0 : mode == "reflect" ? 1 : mode == "edge" ? 2 : -1;
            if (mi < 0) fail(n, "with mode '" + mode + "'");
            shape os = x.dims; std::vector<i64> pairs; bool any = false;
            for (i64 i = r - 1; i >= 0; --i) {
                auto [b, e] = per[std::size_t(i)];
                if (b < 0 || e < 0 || b > (1 << 20) || e > (1 << 20)) fail(n, "with negative or absurd padding");
                if (b || e) { if (!os[std::size_t(i)].is_const()) fail(n, "of a symbolic dimension"); os[std::size_t(i)] = mono(os[std::size_t(i)].k + b + e); any = true; }
                pairs.push_back(b); pairs.push_back(e);
            }
            if (!any) { vals_[o] = x; return; }
            set_tensor(o, F("pad", {x.expr, "pads: " + onnx_detail::ints_text(pairs), "mode: " + std::to_string(mi), "value: " + num(value)}), os, x.integer); return;
        }
        if (op == "Gather") {
            const val& data = in(n, 0); const val& idx = in(n, 1);
            i64 r = i64(data.dims.size()); i64 ax = norm_axis(n, attr_i(n, "axis", 0), r);
            if (r < 1) fail(n, "of a scalar");
            if (idx.kind == val::constant && idx.has_v) {   // fixed positions: index_select
                auto x = T(n, 0);
                std::vector<double> iv = idx.v;
                // a negative index counts from the end (index_select does the same, also along a symbolic dimension)
                if (x.dims[std::size_t(ax)].is_const()) for (auto& v : iv) { if (v < 0) v += double(x.dims[std::size_t(ax)].k); if (v < 0 || v >= double(x.dims[std::size_t(ax)].k)) fail(n, "has an index out of range"); }
                std::string e = F("index_select", {x.expr, weight_of(folded({i64(iv.size())}, iv, true), o + ".index", true), "axis: " + std::to_string(ax)});
                shape mid = x.dims; mid[std::size_t(ax)] = mono(i64(iv.size()));
                shape os(x.dims.begin(), x.dims.begin() + ax); os.insert(os.end(), idx.dims.begin(), idx.dims.end()); os.insert(os.end(), x.dims.begin() + ax + 1, x.dims.end());
                if (os.empty()) os = {mono(1)};
                if (!(os == mid)) e = reshape(e, os);
                set_tensor(o, e, os, x.integer); return;
            }
            // run-time indices (token ids): an embedding lookup along axis 0
            if (ax != 0) fail(n, "with run-time indices along axis " + std::to_string(ax) + " (axis 0, an embedding lookup, is supported)");
            auto table = T(n, 0); auto ids = as_tensor(idx, n.in[1]);
            if (!ids.integer) fail(n, "with indices that are not integers");
            shape tail(table.dims.begin() + 1, table.dims.end());
            std::string te = table.expr;
            if (table.dims.size() != 2) te = reshape(te, {table.dims[0], onnx_detail::product(tail)});
            std::string e = F("gather", {te, ids.expr});
            shape os = ids.dims; os.insert(os.end(), tail.begin(), tail.end());
            shape flat = ids.dims; flat.push_back(onnx_detail::product(tail));
            if (!(os == flat)) e = reshape(e, os);
            set_tensor(o, e, os, table.integer); return;
        }
        if (op == "ReduceMean" || op == "ReduceSum" || op == "ReduceMax" || op == "ReduceMin" || op == "ReduceL2" || op == "ReduceLogSumExp" || op == "ReduceSumSquare" || op == "ReduceL1") {
            auto x = T(n, 0); i64 r = i64(x.dims.size());
            std::vector<i64> axes = n.attrs.count("axes") ? attr_ints(n, "axes") : has_in(n, 1) ? ints(n, 1, "axes") : std::vector<i64>{};
            if (axes.empty()) { if (attr_i(n, "noop_with_empty_axes", 0)) { vals_[o] = x; return; } for (i64 i = 0; i < r; ++i) axes.push_back(i); }
            for (auto& a : axes) a = norm_axis(n, a, r);
            std::sort(axes.begin(), axes.end()); if (std::adjacent_find(axes.begin(), axes.end()) != axes.end()) fail(n, "names an axis twice");
            const char* f = op == "ReduceMean" ? "reduce_mean" : op == "ReduceMax" ? "reduce_max" : op == "ReduceMin" ? "reduce_min" : op == "ReduceLogSumExp" ? "logsumexp" : "reduce_sum";
            std::string e = x.expr;
            if (op == "ReduceL2" || op == "ReduceSumSquare") e = F("square", {e}); else if (op == "ReduceL1") e = F("abs", {e});
            shape kept = x.dims, dropped;
            for (i64 i = 0; i < r; ++i) { bool red = std::binary_search(axes.begin(), axes.end(), i); if (red) kept[std::size_t(i)] = mono(1); else dropped.push_back(x.dims[std::size_t(i)]); }
            if (dropped.empty()) e = F(f, {reshape(e, {mono(1), onnx_detail::product(x.dims)}), "axis: 1"});   // every axis: one reduction over [1, n], which leaves [1] (no rank-0 value)
            else for (std::size_t i = axes.size(); i-- > 0;) e = F(f, {e, "axis: " + std::to_string(axes[i])});   // from the last axis, so earlier indices stay valid
            if (op == "ReduceL2") e = F("sqrt", {e});
            const bool keep = attr_i(n, "keepdims", 1) != 0; shape os = keep ? kept : dropped;
            if (os.empty()) os = {mono(1)};
            if (keep || (dropped.empty() && !(os == shape{mono(1)}))) e = reshape(e, os);
            set_tensor(o, e, os); return;
        }
        if (op == "ArgMax") {
            auto x = T(n, 0); i64 r = i64(x.dims.size()), ax = norm_axis(n, attr_i(n, "axis", 0), r);
            if (attr_i(n, "select_last_index", 0)) fail(n, "with select_last_index=1");
            shape os = x.dims; std::string e;
            if (r == 1) e = F("argmax", {reshape(x.expr, {mono(1), x.dims[0]}), "axis: 1"});   // [1, n] -> [1]: no rank-0 value
            else e = F("argmax", {x.expr, "axis: " + std::to_string(ax)});
            if (attr_i(n, "keepdims", 1)) { os[std::size_t(ax)] = mono(1); if (r != 1) e = reshape(e, os); }
            else { os.erase(os.begin() + ax); if (os.empty()) os = {mono(1)}; }
            set_tensor(o, e, os, true); return;
        }
        if (op == "ConstantOfShape") {
            auto want = as_dims(in(n, 0)); if (!want) fail(n, "needs a constant or shape-derived shape");
            double fill = 0; bool integer = false;
            if (auto it = n.attrs.find("value"); it != n.attrs.end() && it->second.t && it->second.t->count() == 1) { fill = it->second.t->at(0); integer = it->second.t->integer(); }
            // a constant block whose size may depend on symbolic dimensions: a [1, ..., 1] weight expanded along each axis
            shape cur(want->first.size(), mono(1)); if (cur.empty()) cur = {mono(1)};
            std::string e = weight_of(folded(std::vector<i64>(cur.size(), 1), {fill}, integer), o + ".fill");
            for (std::size_t i = 0; i < want->first.size(); ++i) if (!want->first[i].is(1)) { e = F("expand", {e, "axis: " + std::to_string(i), "n: " + want->first[i].text()}); cur[i] = want->first[i]; }
            set_tensor(o, e, cur, integer); return;
        }
        fail(n, "is not supported by the FORGE importer");
    }

    const onnx_pb::model_t& m_;
    onnx_translation tr_;
    std::map<std::string, const tensor_t*> inits_;
    std::map<std::string, val> vals_;
    std::map<const tensor_t*, std::string> weight_id_;
    std::list<tensor_t> owned_;
    std::set<std::string> used_;
    std::map<std::string, std::string> sym_of_;
    std::vector<std::string> symbols_;
    std::string ins_, body_;
    int opset_ = 1; std::size_t lines_ = 0;
};

// malformed graphs (bad perms, out-of-range axes) surface as a format error rather than a library exception
inline onnx_translation translate_onnx(const onnx_pb::model_t& m) {
    try { return onnx_translator(m).run(); }
    catch (const std::out_of_range& e) { raise(errc::format_error, "onnx: malformed graph ({})", e.what()); }
    catch (const std::length_error& e) { raise(errc::format_error, "onnx: malformed graph ({})", e.what()); }
}

inline import_report import_onnx(const std::string& path, const std::string& out, const import_options& opt = {}) {
    check_write(out);
    auto bytes = read_file(path);   // checks fs_read (hf.hpp)
    // external tensor data is resolved against the model file's directory
    onnx_pb::external_loader ext; { auto dir = os::fs::u8path(path).parent_path(); ext.base = os::fs::u8str(dir.empty() ? std::filesystem::path(".") : dir); }
    auto m = onnx_pb::parse_model(bytes, &ext);
    auto tr = translate_onnx(m);
    auto bp = modelfile::parse_blueprint(tr.blueprint);  // validate syntax before writing
    modelfile::writer w;
    w.set_alignment(import_alignment(opt, path));
    import_report rep{"onnx", "onnx", 0, 0};
    for (auto& [k, t] : tr.weights) {
        // matrices take the requested codec; vectors, convolution kernels and index tables stay f32
        auto codec = (opt.codec == "f32" || t.dims.size() != 2 || tr.exact.count(k)) ? std::string("f32") : opt.codec;
        w.add_tensor(k, t.dims, t.f, codec); ++rep.tensors;
    }
    json::object man{{"format", "fpk"}, {"forge_format_version", double(modelfile::format_version)}, {"architecture", "onnx"},
                     {"source", "onnx"}, {"producer", m.producer}, {"onnx_opset", double(m.opset)}, {"session", bp.session},
                     {"onnx_inputs", tr.inputs}, {"onnx_outputs", tr.outputs}};
    {   // licence: metadata_props license / licence (identifier), license_name, license_url / license_link; a
        // doc_string that mentions a licence is kept as the licence text
        license_info lic;
        auto lower = [](std::string s) { for (auto& c : s) c = char(std::tolower((unsigned char)c)); return s; };
        for (auto& [k, v] : m.metadata) {
            auto lk = lower(k);
            if (lk == "license" || lk == "licence") lic.id = v;
            else if (lk == "license_name" || lk == "licence_name") lic.name = v;
            else if (lk == "license_url" || lk == "license_link" || lk == "licence_url" || lk == "licence_link") lic.link = v;
        }
        if (auto ld = lower(m.doc); ld.find("license") != std::string::npos || ld.find("licence") != std::string::npos) lic.text = m.doc;
        lic.store(man, w); rep.license = lic.summary();
    }
    w.set_manifest(json::value(man));
    w.set_section("blueprint", tr.blueprint);
    w.write(out);
    rep.bytes = std::filesystem::file_size(out);
    return rep;
}

} // namespace forge::import
