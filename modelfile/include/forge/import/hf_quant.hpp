#pragma once
// AWQ and GPTQ checkpoints (Hugging Face safetensors with `quantization_config` in config.json).
//
// A quantised Linear `<base>` [out, in] is stored without `<base>.weight`, as
//     <base>.qweight  int32   packed integer codes q, 32 / bits to a word, lowest bits first
//     <base>.qzeros   int32   packed zero points z, one per (group of input columns, output row)
//     <base>.scales   f16/f32 one scale s per (group, output row)
//     <base>.g_idx    int32   GPTQ only: the group of every input column
// and the weight is W[o][i] = s[g(i)][o] * (q[i][o] - z[g(i)][o]). The two exporters pack differently. What is
// implemented here is the layout each one documents, checked against a packer written from the same description
// (verify/golden/hf_quant.py). Neither has been checked against a checkpoint written by AutoAWQ / AutoGPTQ itself.
//
//   AWQ, AutoAWQ WQLinear "gemm" (4 bits only):
//     qweight [in, out / 8], qzeros [in / group, out / 8], scales [in / group, out]; g(i) = i / group.
//     The eight codes of output rows 8c .. 8c+7 share word c, and code e sits at nibble {0, 4, 1, 5, 2, 6, 3, 7}[e]
//     (AutoAWQ fills nibble p from element {0, 2, 4, 6, 1, 3, 5, 7}[p]). qzeros uses the same order.
//   GPTQ, AutoGPTQ QuantLinear (2, 4 or 8 bits):
//     qweight [in * bits / 32, out], qzeros [groups, out * bits / 32], scales [groups, out], g_idx [in].
//     Code (i, o) is bits [bits * (i % P), +bits) of qweight[i / P][o], P = 32 / bits; the zero of (g, o) is bits
//     [bits * (o % P), +bits) of qzeros[g][o / P]. The original checkpoint format ("gptq", the default) stores z - 1,
//     so z = (stored + 1) mod 2^bits; `checkpoint_format: "gptq_v2"` stores z itself.
//
// A Linear whose groups are runs of whole 32-column blocks (group size a multiple of 32, g_idx in order) at 4 or 8
// bits maps onto FORGE's grouped codec q4_g32 / q8_g32 ({f16 scale; f16 min; q[32]}, x = q * scale + min) with
// min = -z * s: the codes are stored as they are, the values agree with the checkpoint's up to the f16 rounding of
// min (and of the scales, when the checkpoint has them in f32). Anything else is decoded to f32.
#include <forge/import/safetensors.hpp>
#include <forge/graph/codecs.hpp>
#include <memory>
#include <optional>

namespace forge::import {

struct quant_scheme {
    std::string method;            // "awq" or "gptq"
    int bits = 4;
    i64 group_size = 128;          // input columns per group; -1: the whole row
    bool zero_minus_one = false;   // GPTQ v1 checkpoints: qzeros holds z - 1
};

// The scheme of a config.json, or nullopt when it is not an AWQ / GPTQ checkpoint. Unsupported AWQ / GPTQ variants raise.
inline std::optional<quant_scheme> quant_scheme_of(const json::value& cfg) {
    const auto* q = cfg.find("quantization_config");
    if (!q || !q->is_obj()) return std::nullopt;
    quant_scheme s;
    s.method = q->str_or("quant_method", "");
    auto flag = [&](const char* k, bool d) { auto* v = q->find(k); return v && v->is_bool() ? v->b() : d; };
    s.bits = int(q->num_or("bits", q->num_or("w_bit", 4)));
    s.group_size = i64(q->num_or("group_size", q->num_or("q_group_size", 128)));
    if (s.method == "awq") {
        auto ver = q->str_or("version", "gemm");
        for (auto& ch : ver) if (ch >= 'A' && ch <= 'Z') ch = char(ch - 'A' + 'a');
        if (ver != "gemm") raise(errc::unsupported, "AWQ checkpoint version '{}' is not supported (only 'gemm')", ver);
        if (!flag("zero_point", true)) raise(errc::unsupported, "AWQ checkpoints without zero_point are not supported");
        if (s.bits != 4) raise(errc::unsupported, "AWQ checkpoints with {} bits are not supported (the gemm packing is defined for 4)", s.bits);
    } else if (s.method == "gptq") {
        if (s.bits != 2 && s.bits != 4 && s.bits != 8) raise(errc::unsupported, "GPTQ checkpoints with {} bits are not supported (2, 4 or 8)", s.bits);
        auto fmt = q->str_or("checkpoint_format", "gptq");
        if (fmt != "gptq" && fmt != "gptq_v2") raise(errc::unsupported, "GPTQ checkpoint_format '{}' is not supported (gptq, gptq_v2)", fmt);
        s.zero_minus_one = fmt == "gptq";
    } else return std::nullopt;   // another scheme (or dense weights with a config block): the tensors are imported as stored
    if (s.group_size == 0 || s.group_size < -1) raise(errc::format_error, "quantization_config: bad group_size {}", s.group_size);
    return s;
}

// One unpacked Linear: codes and per-(group, row) scale / zero, in FORGE's [out][in] order.
struct qlinear {
    i64 out = 0, in = 0, groups = 0; int bits = 4;
    std::vector<u8> q;                  // [out][in]
    std::vector<float> scale, zero;     // [groups][out]
    std::vector<std::int32_t> group;    // [in]
    float value(i64 o, i64 i) const {
        std::size_t g = std::size_t(group[std::size_t(i)]) * std::size_t(out) + std::size_t(o);
        return (float(q[std::size_t(o * in + i)]) - zero[g]) * scale[g];
    }
    tensor decode() const {
        tensor t(shape_t{out, in});
        for (i64 o = 0; o < out; ++o) for (i64 i = 0; i < in; ++i) t.data[std::size_t(o * in + i)] = value(o, i);
        return t;
    }
    // Width of the uniform, in-order groups (every group a run of `w` consecutive columns), or 0.
    i64 group_width() const {
        if (groups < 1 || in % groups) return 0;
        i64 w = in / groups;
        for (i64 i = 0; i < in; ++i) if (group[std::size_t(i)] != i / w) return 0;
        return w;
    }
    // The grouped codec this Linear can be stored in without re-quantising, or nullptr.
    const char* native_codec() const {
        i64 w = group_width();
        if (!w || w % 32 || in % 32) return nullptr;
        return bits == 4 ? "q4_g32" : bits == 8 ? "q8_g32" : nullptr;
    }
    std::vector<u8> native_blocks() const {
        const std::size_t bb = bits == 4 ? 20 : 36, nb = std::size_t(in / 32); const i64 w = group_width();
        std::vector<u8> b(std::size_t(out) * nb * bb, 0);
        for (i64 o = 0; o < out; ++o)
            for (std::size_t k = 0; k < nb; ++k) {
                u8* p = b.data() + (std::size_t(o) * nb + k) * bb;
                std::size_t g = std::size_t(i64(k) * 32 / w) * std::size_t(out) + std::size_t(o);
                graph::codecs::detail::put16(p, scale[g]); graph::codecs::detail::put16(p + 2, -zero[g] * scale[g]);
                const u8* qq = q.data() + std::size_t(o * in) + k * 32;
                for (int e = 0; e < 32; ++e) { if (bits == 8) p[4 + e] = qq[e]; else p[4 + e / 2] |= u8(qq[e] << (4 * (e & 1))); }
            }
        return b;
    }
};

// Reads the quantised Linears of a set of shards. `load("<base>.weight")` gives the decoded weight and keeps the
// Linear so that `take()` can hand out its grouped blocks.
class quantised_checkpoint {
public:
    using shard_list = std::vector<std::unique_ptr<safetensors>>;
    quantised_checkpoint(std::optional<quant_scheme> s, const shard_list& shards) : scheme_(std::move(s)), shards_(shards) {}
    bool active() const { return scheme_.has_value(); }
    const quant_scheme& scheme() const { return *scheme_; }
    bool has(const std::string& weight) const { return active() && weight.ends_with(".weight") && shard(base_of(weight) + ".qweight"); }
    tensor load(const std::string& weight) {
        last_ = std::make_unique<qlinear>(scheme_->method == "awq" ? unpack_awq(base_of(weight)) : unpack_gptq(base_of(weight)));
        return last_->decode();
    }
    void forget() { last_.reset(); }
    struct native { std::string codec; shape_t shape; std::vector<u8> blocks; };
    // The Linear of the last load() as grouped-codec blocks (nullopt: none pending, or it does not fit); clears it.
    std::optional<native> take() {
        auto l = std::move(last_);
        if (!l) return std::nullopt;
        if (auto* c = l->native_codec()) return native{c, shape_t{l->out, l->in}, l->native_blocks()};
        return std::nullopt;
    }
    std::size_t n_native = 0, n_linear = 0;   // Linears stored as grouped blocks (counted by the importer) / read in total
private:
    static std::string base_of(const std::string& weight) { return weight.substr(0, weight.size() - 7); }
    const safetensors* shard(const std::string& n) const { for (auto& s : shards_) if (s->has(n)) return s.get(); return nullptr; }
    const safetensors& need(const std::string& n) const { if (auto* s = shard(n)) return *s; raise(errc::not_found, "missing tensor {}", n); }
    static i64 dim(const shape_t& s, std::size_t i, const std::string& n) { if (s.size() != 2 || s[0] < 1 || s[1] < 1) raise(errc::shape_mismatch, "{} has shape {}", n, shape_str(s)); return s[i]; }
    void no_bias(const std::string& base) const {
        if (auto* s = shard(base + ".bias")) for (float v : s->load(base + ".bias").data) if (v != 0) raise(errc::unsupported, "{}: quantised Linear layers with a bias are not supported", base);
    }
    // scales [groups][out] and unpacked zeros: `zw` is the int32 qzeros matrix [groups][out / per word]
    void scales_zeros(const std::string& base, qlinear& l, const std::vector<std::int32_t>& zw, const shape_t& zs, auto&& zpos) const {
        auto sc = need(base + ".scales").load(base + ".scales");
        if (sc.shape != shape_t{l.groups, l.out}) raise(errc::shape_mismatch, "{}.scales has shape {}, expected {}", base, shape_str(sc.shape), shape_str(shape_t{l.groups, l.out}));
        const i64 P = 32 / l.bits; const std::uint32_t mask = (1u << l.bits) - 1;
        if (l.out % P || zs != shape_t{l.groups, l.out / P}) raise(errc::shape_mismatch, "{}.qzeros has shape {}, expected {}", base, shape_str(zs), shape_str(shape_t{l.groups, l.out / P}));
        l.scale = std::move(sc.data); l.zero.resize(l.scale.size());
        for (i64 g = 0; g < l.groups; ++g)
            for (i64 o = 0; o < l.out; ++o) {
                std::uint32_t z = (std::uint32_t(zw[std::size_t(g * (l.out / P) + o / P)]) >> (l.bits * zpos(o % P))) & mask;
                if (scheme_->zero_minus_one) z = (z + 1) & mask;
                l.zero[std::size_t(g * l.out + o)] = float(z);
            }
    }
    qlinear unpack_awq(const std::string& base) {
        static constexpr int pos[8] = {0, 4, 1, 5, 2, 6, 3, 7};   // nibble of element e within its word
        shape_t qs, zs; qlinear l; l.bits = scheme_->bits;
        auto qw = need(base + ".qweight").load_i32(base + ".qweight", qs);
        auto zw = need(base + ".qzeros").load_i32(base + ".qzeros", zs);
        const i64 P = 8;
        l.in = dim(qs, 0, base + ".qweight"); l.out = dim(qs, 1, base + ".qweight") * P;
        i64 gs = scheme_->group_size < 0 ? l.in : scheme_->group_size;
        if (l.in % gs) raise(errc::shape_mismatch, "{}: {} input columns are not a multiple of group_size {}", base, l.in, gs);
        l.groups = l.in / gs;
        scales_zeros(base, l, zw, zs, [](i64 e) { return pos[e]; });
        l.group.resize(std::size_t(l.in)); for (i64 i = 0; i < l.in; ++i) l.group[std::size_t(i)] = std::int32_t(i / gs);
        l.q.resize(std::size_t(l.in * l.out));
        for (i64 i = 0; i < l.in; ++i)
            for (i64 o = 0; o < l.out; ++o)
                l.q[std::size_t(o * l.in + i)] = u8((std::uint32_t(qw[std::size_t(i * (l.out / P) + o / P)]) >> (4 * pos[o % P])) & 15);
        no_bias(base); ++n_linear;
        return l;
    }
    qlinear unpack_gptq(const std::string& base) {
        shape_t qs, zs, gsh; qlinear l; l.bits = scheme_->bits;
        auto qw = need(base + ".qweight").load_i32(base + ".qweight", qs);
        auto zw = need(base + ".qzeros").load_i32(base + ".qzeros", zs);
        const i64 P = 32 / l.bits; const std::uint32_t mask = (1u << l.bits) - 1;
        l.in = dim(qs, 0, base + ".qweight") * P; l.out = dim(qs, 1, base + ".qweight");
        l.groups = dim(zs, 0, base + ".qzeros");
        if (shard(base + ".g_idx")) {
            l.group = need(base + ".g_idx").load_i32(base + ".g_idx", gsh);
            if (gsh != shape_t{l.in}) raise(errc::shape_mismatch, "{}.g_idx has shape {}, expected [{}]", base, shape_str(gsh), l.in);
            for (auto g : l.group) if (g < 0 || g >= l.groups) raise(errc::format_error, "{}.g_idx: group {} out of range", base, g);
        } else {
            i64 gs = scheme_->group_size < 0 ? l.in : scheme_->group_size;
            if (l.in % gs || l.in / gs != l.groups) raise(errc::shape_mismatch, "{}: no g_idx, and {} columns in groups of {} do not make {} groups", base, l.in, gs, l.groups);
            l.group.resize(std::size_t(l.in)); for (i64 i = 0; i < l.in; ++i) l.group[std::size_t(i)] = std::int32_t(i / gs);
        }
        scales_zeros(base, l, zw, zs, [](i64 e) { return int(e); });
        l.q.resize(std::size_t(l.in * l.out));
        for (i64 i = 0; i < l.in; ++i)
            for (i64 o = 0; o < l.out; ++o)
                l.q[std::size_t(o * l.in + i)] = u8((std::uint32_t(qw[std::size_t((i / P) * l.out + o)]) >> (l.bits * (i % P))) & mask);
        no_bias(base); ++n_linear;
        return l;
    }
    std::optional<quant_scheme> scheme_;
    const shard_list& shards_;
    std::unique_ptr<qlinear> last_;
};

} // namespace forge::import
