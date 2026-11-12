#pragma once
// Native AOT: writes the C++ translation unit that executes compiled plans as straight-line code (docs/aot-models.md).
//
// For every plan the generator emits its tables (value shapes, arena offsets, step inputs, attributes) and one
// function per run of consecutive kernel steps. Inside a function each step is one statement:
//   - a direct call with constant shapes and arena offsets for the element-wise, row and GEMM lowerings
//     (rt::binary<&K::add>(A + 1024, A + 2048, A + 1024, 256);)
//   - a call of the lowering's kernel over views bound once per plan for everything else (attention, rope, gather,
//     quantised matmul, MoE, reference kernels)
// State steps (KV appends, recurrent state) are not compiled: the engine runs them between the functions.
// The unit includes the FORGE headers (aot_rt.hpp), so the kernels are the ones the interpreted plan calls; it exports
// forge_aot_model_v1_get() (forge_aot.h).
#include <forge/graph/aot.hpp>

namespace forge::graph::aot {

struct codegen_options {
    std::string fpk_path;         // recorded in the unit (absolute path of the model file)
    std::string embed_fpk;        // non-empty: embed this file's bytes (.incbin; GCC/Clang only)
    std::string simd_target;      // lowest SIMD target the build flags require
    std::string compiler;         // note recorded in the unit
    float temperature = 0; int top_k = 0; float top_p = 1; int max_new_tokens = 32;   // defaults of the executable
};

class codegen {
public:
    // Plans in blob entry order.
    void add(const target& t, const plan& p) {
        if (p.hooks || !p.stage_of_node.empty()) raise(errc::unsupported, "aot: only CPU plans can be compiled to native code");
        std::string ns = "p" + std::to_string(plans_.size());
        const auto& G = p.g;
        const float* lo = p.arena.data(); const float* hi = lo + p.arena.size();
        std::vector<char> is_state(G.size(), 0);
        for (std::size_t i = 0; i < G.size(); ++i) if (G.at(value_id(i)).kind == node_kind::state_append) is_state[i] = 1;
        for (auto [a, r] : p.state_aliases) is_state[a] = 1;
        auto in_arena = [&](std::size_t i) { return !is_state[i] && p.views[i].data >= lo && p.views[i].data < hi; };
        auto off = [&](std::size_t i) { return std::size_t(p.views[i].data - lo); };
        std::string dims = "0", vals, sin, steps, attrs, segs, segfn, first, count;
        std::size_t ndims = 1, nin = 0;
        for (std::size_t i = 0; i < G.size(); ++i) {
            const auto& v = p.views[i];
            int kind = in_arena(i) ? 0 : is_state[i] ? 2 : 1;
            if (kind == 1 && !v.data && !v.qdata) raise(errc::unsupported, "aot: value %{} has no storage", i);
            std::string codec = "nullptr";
            if (v.qdata) codec = quote(static_cast<const codecs::codec*>(v.qcodec)->name);
            vals += std::format("    {{{}, {}, {}, {}u, {}}},\n", kind, v.shape.size(), ndims, kind == 0 ? off(i) : 0, codec);
            for (auto d : v.shape) { dims += std::format(", {}", d); ++ndims; }
        }
        // a pointer expression for a value in a direct call (empty: not available as a constant expression)
        auto ptr = [&](value_id v) -> std::string {
            if (in_arena(v)) return std::format("A + {}", off(v));
            if (!is_state[v] && p.views[v].data) return std::format("V[{}].data", v);
            return {};
        };
        std::size_t nseg = 0; bool open = false; std::size_t seg_first = 0;
        auto close = [&](std::size_t end) {
            if (!open) return;
            segfn += "}\n"; open = false;
            first += std::format("{}{}", nseg ? ", " : "", seg_first); count += std::format("{}{}", nseg ? ", " : "", end - seg_first);
            segs += std::format("{}seg{}", nseg ? ", " : "", nseg); ++nseg;
        };
        for (std::size_t s = 0; s < p.steps.size(); ++s) {
            const auto& st = p.steps[s];
            if (st.kind != 0) {   // state step: the engine's
                close(s);
                steps += std::format("    {{nullptr, nullptr, {}, 0, {}}},\n", nin, st.out);
                continue;
            }
            if (!st.low || st.low->on_device || st.fuse) raise(errc::unsupported, "aot: step {} cannot be compiled to native code", s);
            const auto& n = G.at(st.out);
            if (!open) { segfn += std::format("void seg{}(rt::instance& I) {{\n    float* const A = I.arena; const tensor_view* const V = I.v.data(); (void)A; (void)V;\n", nseg); open = true; seg_first = s; }
            steps += std::format("    {{{}, {}, {}, {}, {}}},\n", quote(st.low->name), quote(ops::info(n.op).name), nin, st.in.size(), st.out);
            for (auto in : st.in) { sin += std::format("{}{}", nin ? ", " : "", in); ++nin; }
            if (!st.attrs->v.empty()) attrs += std::format("        case {}: {} break;\n", s, attr_code(*st.attrs));
            std::string line = direct(p, st, ptr);
            if (line.empty()) { line = std::format("I.call({});", s); ++generic_; } else ++direct_;
            segfn += std::format("    {:<96} // {} %{} = {} {}\n", line, s, st.out, clean(st.low->name), shape_str(p.views[st.out].shape));
        }
        close(p.steps.size());
        std::string o;
        o += std::format("// ---- plan {}: {} ({} values, {} steps, arena {} floats) ----\nnamespace {} {{\n", plans_.size(), clean(t.name()), G.size(), p.steps.size(), p.arena.size(), ns);
        o += std::format("const std::int64_t dims[] = {{{}}};\n", dims);
        o += std::format("const rt::value_desc values[] = {{\n{}}};\n", vals);
        o += std::format("const std::uint32_t step_in[] = {{{}}};\n", nin ? sin : "0");
        o += std::format("const rt::step_desc steps[] = {{\n{}}};\n", steps.empty() ? "    {nullptr, nullptr, 0, 0, 0},\n" : steps);
        o += std::format("ops::attrs attrs(std::uint32_t s) {{\n    ops::attrs a;\n    switch (s) {{\n{}        default: break;\n    }}\n    return a;\n}}\n", attrs);
        o += std::format("const rt::plan_desc desc = {{{}, {}, {}u, values, dims, steps, step_in, attrs}};\n", G.size(), p.steps.size(), p.arena.size());
        o += segfn;
        if (nseg) {
            o += std::format("const rt::segment_fn segs[] = {{{}}};\nconst std::uint32_t seg_first[] = {{{}}}, seg_steps[] = {{{}}};\n", segs, first, count);
            o += std::format("void* bind(const forge_aot_value* v, float* arena, char* err, size_t cap) {{ return rt::bind_c(desc, reinterpret_cast<const rt::value*>(v), arena, err, cap); }}\n");
            o += std::format("int run(void* inst, uint32_t seg, const forge_aot_value* v, char* err, size_t cap) {{ return rt::run_c(inst, segs, {}, seg, reinterpret_cast<const rt::value*>(v), err, cap); }}\n", nseg);
        }
        o += std::format("}} // namespace {}\n\n", ns);
        plans_.push_back({ns, std::move(o), std::format("    {{{}, {}, {}u, {}, {}, {}, {}, {}, rt::unbind_c}},\n", G.size(), p.steps.size(), p.arena.size(), nseg,
                                                     nseg ? ns + "::seg_first" : "nullptr", nseg ? ns + "::seg_steps" : "nullptr", nseg ? ns + "::bind" : "nullptr", nseg ? ns + "::run" : "nullptr")});
    }
    std::size_t plans() const { return plans_.size(); }
    std::size_t direct_steps() const { return direct_; }     // steps emitted as direct calls with constant shapes
    std::size_t generic_steps() const { return generic_; }   // steps emitted as kernel calls over bound views

    // The translation unit for the plans added so far; `blob` is the plan blob holding the same plans in order.
    std::string finish(std::span<const std::uint8_t> blob, const codegen_options& o) const {
        std::string s;
        s += std::format("// Generated by `forge aot --aot-mode native` (FORGE {}). Do not edit: regenerate from the model.\n", forge_version);
        s += std::format("// Model: {}\n// {} plan(s), {} direct step(s), {} kernel call(s) over bound views.\n", clean(o.fpk_path), plans_.size(), direct_, generic_);
        s += "#define FORGE_AOT_BUILD 1\n#include <forge_aot.h>\n#include <forge/graph/aot_rt.hpp>\n\n";
        if (!o.embed_fpk.empty()) {
            s += "#if !defined(__GNUC__) || defined(_MSC_VER)\n#error \"--embed-weights needs a GCC- or Clang-compatible compiler (.incbin)\"\n#endif\n";
            // the file's bytes in read-only data, 64-byte aligned (f32 tensors are used in place)
            std::string path;
            for (unsigned char c : o.embed_fpk) {
                if (c < 32 || c >= 127) raise(errc::unsupported, "aot: cannot embed a file whose path has non-ASCII characters ({})", clean(o.embed_fpk));
                if (c == '"' || c == '\\') path += '\\';
                path += char(c);
            }
            std::string body = ".balign 64\nforge_aot_fpk_begin:\n.incbin \"" + path + "\"\nforge_aot_fpk_end:\n.byte 0\n";
            s += std::format("#if defined(__APPLE__)\n__asm__({});\n#elif defined(_WIN32)\n__asm__({});\n#else\n__asm__({});\n#endif\n",
                             quote(".section __TEXT,__const\n" + body + ".text\n"), quote(".section .rdata,\"dr\"\n" + body + ".text\n"), quote(".pushsection .rodata\n" + body + ".popsection\n"));
            s += "extern const unsigned char forge_aot_fpk_begin[] __asm__(\"forge_aot_fpk_begin\");\nextern const unsigned char forge_aot_fpk_end[] __asm__(\"forge_aot_fpk_end\");\n\n";
        }
        s += "namespace {\nusing namespace forge;\nusing namespace forge::graph;\nnamespace rt = forge::graph::aot::rt;\nusing K = forge::simd::kernels;\n";
        s += "static_assert(sizeof(forge_aot_value) == sizeof(rt::value) && offsetof(forge_aot_value, qdata) == offsetof(rt::value, qdata) && offsetof(forge_aot_value, rows) == offsetof(rt::value, rows));\n\n";
        for (auto& p : plans_) s += p.code;
        s += "const forge_aot_plan_v1 plan_table[] = {\n";
        for (auto& p : plans_) s += p.entry;
        if (plans_.empty()) s += "    {0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr},\n";
        s += "};\n\n// the plan blob these functions were generated from (docs/format.md, section \"aot\")\nconst unsigned char plan_blob[] = {";
        static const char* hx = "0123456789abcdef";
        for (std::size_t i = 0; i < blob.size(); ++i) {
            if (i % 32 == 0) s += "\n";
            s += "0x"; s += hx[blob[i] >> 4]; s += hx[blob[i] & 15]; s += ',';
        }
        s += "\n};\n\n";
        s += std::format("int check(char* err, size_t cap) {{ return rt::check_c({}, err, cap); }}\n\n", quote(o.simd_target));
        s += "const forge_aot_model_v1 model_desc = {\n";
        s += std::format("    FORGE_AOT_ABI_VERSION, sizeof(forge_aot_model_v1), {}, {}, {},\n", quote(forge_version), quote(o.simd_target), quote(o.compiler));
        if (o.embed_fpk.empty()) s += std::format("    {}, nullptr, 0,\n", quote(o.fpk_path));
        else s += std::format("    {}, forge_aot_fpk_begin, 0,   // size filled in below (not a constant expression)\n", quote(o.fpk_path));
        s += std::format("    plan_blob, sizeof plan_blob, {}, plan_table, check,\n", plans_.size());
        s += std::format("    {}f, {}, {}f, {}\n}};\n", fnum(o.temperature), o.top_k, fnum(o.top_p), o.max_new_tokens);
        s += "} // namespace\n\n";
        s += "extern \"C\" FORGE_AOT_EXPORT const forge_aot_model_v1* forge_aot_model_v1_get(void) {\n";
        if (o.embed_fpk.empty()) s += "    return &model_desc;\n}\n";
        else s += "    static const forge_aot_model_v1 m = [] { auto d = model_desc; d.fpk_size = uint64_t(forge_aot_fpk_end - forge_aot_fpk_begin); return d; }();\n    return &m;\n}\n";
        s += "\n#if defined(FORGE_AOT_EXE)   // the standalone executable: forge run's loop over this model\n#include <forge/inference/aot_main.hpp>\n";
        s += "int main(int argc, char** argv) { return forge::inference::aot_main(argc, argv, forge_aot_model_v1_get()); }\n#endif\n";
        return s;
    }
private:
    // a C++ string literal (octal escapes for anything that is not plain printable ASCII)
    static std::string quote(std::string_view t) {
        std::string s = "\"";
        for (unsigned char c : t) {
            if (c == '"' || c == '\\') { s += '\\'; s += char(c); }
            else if (c >= 32 && c < 127) s += char(c);
            else s += std::format("\\{:03o}", unsigned(c));
        }
        return s + "\"";
    }
    static std::string clean(std::string_view t) { std::string s; for (unsigned char c : t) s += c >= 32 && c < 127 ? char(c) : '?'; return s; }
    static std::string fnum(double d) { auto s = std::format("{:.9g}", d); if (s.find_first_of(".en") == std::string::npos) s += ".0"; return s; }
    static std::string dnum(double d) {
        if (std::isnan(d)) return "std::numeric_limits<double>::quiet_NaN()";
        if (std::isinf(d)) return d > 0 ? "std::numeric_limits<double>::infinity()" : "-std::numeric_limits<double>::infinity()";
        auto s = std::format("{:.17g}", d); if (s.find_first_of(".en") == std::string::npos) s += ".0"; return s;
    }
    static std::string attr_code(const ops::attrs& a) {
        std::string s;
        for (auto& [k, v] : a.v) {
            if (auto* d = std::get_if<double>(&v)) s += std::format("a.set({}, {}); ", quote(k), dnum(*d));
            else if (auto* iv = std::get_if<std::vector<i64>>(&v)) {
                s += std::format("a.set({}, std::vector<i64>{{", quote(k));
                for (std::size_t i = 0; i < iv->size(); ++i) s += std::format("{}i64({}LL)", i ? ", " : "", (*iv)[i]);
                s += "}); ";
            } else s += std::format("a.set({}, std::string({})); ", quote(k), quote(std::get<std::string>(v)));
        }
        return s;
    }
    // The direct statement for a step, or empty (then the step is a kernel call over its bound views). Mirrors the
    // lowerings of kernels.hpp through the forms in aot_rt.hpp.
    template <class Ptr> static std::string direct(const plan& p, const step& st, Ptr&& ptr) {
        const std::string& low = st.low->name;
        const auto& out = p.views[st.out];
        for (auto d : out.shape) if (d <= 0) return {};
        for (auto in : st.in) for (auto d : p.views[in].shape) if (d <= 0) return {};
        std::vector<std::string> x; for (auto in : st.in) x.push_back(ptr(in));
        std::string o = ptr(st.out);
        if (o.empty()) return {};
        for (auto& e : x) if (e.empty()) return {};
        auto size = [&](value_id v) { return p.views[v].size(); };
        static const std::map<std::string, std::string> un = {{"lane.exp", "exp"}, {"lane.log", "log"}, {"lane.tanh", "tanh"}, {"lane.sigmoid", "sigmoid"}, {"lane.silu", "silu"}, {"lane.gelu_tanh", "gelu"}};
        static const std::map<std::string, std::string> bin = {{"lane.add", "add"}, {"lane.sub", "sub"}, {"lane.mul", "mul"}, {"lane.silu_mul", "silu_mul"}};
        if (auto it = un.find(low); it != un.end() && x.size() == 1) return std::format("rt::unary<&K::{}>({}, {}, {});", it->second, x[0], o, out.size());
        if (auto it = bin.find(low); it != bin.end() && x.size() == 2 && size(st.in[0]) == out.size() && size(st.in[1]) == out.size())
            return std::format("rt::binary<&K::{}>({}, {}, {}, {});", it->second, x[0], x[1], o, out.size());
        if (x.empty() || p.views[st.in[0]].shape.empty()) return {};
        std::size_t n = std::size_t(p.views[st.in[0]].shape.back()), r = size(st.in[0]) / n;
        if (low == "lane.softmax" && x.size() == 1) return std::format("rt::softmax({}, {}, {}, {});", x[0], o, r, n);
        if (low == "lane.rms_norm" && x.size() == 2) return std::format("rt::rms_norm({}, {}, {}, {}, {}, float({}));", x[0], x[1], o, r, n, dnum(st.attrs->num("eps", 1e-6)));
        if (low == "lane.gemm_nt" && x.size() == 2 && p.views[st.in[1]].shape.size() == 2) return std::format("rt::gemm_nt({}, {}, {}, {}, {}, {});", x[0], x[1], o, r, p.views[st.in[1]].shape[0], n);
        if (low == "lane.gemm_nn" && x.size() == 2 && p.views[st.in[1]].shape.size() == 2) return std::format("rt::gemm_nn({}, {}, {}, {}, {}, {});", x[0], x[1], o, r, p.views[st.in[1]].shape[1], n);
        if (low == "cpu.reshape" && x.size() == 1) return p.views[st.in[0]].data == out.data ? std::string("/* a view of its input */") : std::format("rt::copy({}, {}, {});", x[0], o, out.size());
        return {};
    }
    struct unit { std::string ns, code, entry; };
    std::vector<unit> plans_;
    std::size_t direct_ = 0, generic_ = 0;
};

} // namespace forge::graph::aot
