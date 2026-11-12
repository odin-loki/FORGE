#pragma once
// Runtime support for natively compiled plans: what the code written by aot_codegen.hpp calls (docs/aot-models.md).
//
// The generated code is a straight-line function per run of kernel steps. A step is either a direct call with the
// shapes and arena offsets as constants (the forms below), or a call of the lowering's kernel over views that were
// built once when the code was bound to a plan. The direct forms repeat the bodies of the CPU lowerings in
// kernels.hpp (same kernels, same chunking, same order), which is what keeps the results bit-identical to the
// interpreted plan; tests/test_aot_native.cpp compares the two. Change them together.
#include <forge/graph/plan.hpp>
#include <cstdio>

namespace forge::graph::aot::rt {

// ---- direct forms of the CPU lowerings (kernels.hpp: cpu_kernels) ----
template <auto member> inline void unary(const float* x, float* o, std::size_t n) {
    auto f = simd::active().*member;
    task::parallel_for(n, 16384, [&](std::size_t b, std::size_t e) { f(x + b, o + b, e - b); });
}
template <auto member> inline void binary(const float* x, const float* y, float* o, std::size_t n) {
    auto f = simd::active().*member;
    task::parallel_for(n, 16384, [&](std::size_t b, std::size_t e) { f(x + b, y + b, o + b, e - b); });
}
inline void softmax(const float* x, float* o, std::size_t r, std::size_t n) {
    task::parallel_for(r, 4, [&](std::size_t b, std::size_t e) { for (std::size_t i = b; i < e; ++i) simd::active().softmax_row(x + i * n, o + i * n, n); });
}
inline void rms_norm(const float* x, const float* w, float* o, std::size_t r, std::size_t n, float eps) {
    task::parallel_for(r, 4, [&](std::size_t b, std::size_t e) { for (std::size_t i = b; i < e; ++i) simd::active().rms_norm_row(x + i * n, w, o + i * n, n, eps); });
}
inline void gemm_nt(const float* x, const float* w, float* o, std::size_t m, std::size_t n, std::size_t k) { gpu::prims::gemm_nt(x, w, o, m, n, k); }
inline void gemm_nn(const float* x, const float* w, float* o, std::size_t m, std::size_t n, std::size_t k) {
    task::parallel_for((m + 3) / 4, 1, [&](std::size_t b, std::size_t e) {
        std::size_t r0 = b * 4, r1 = std::min(m, e * 4);
        simd::active().gemm_nn(x + r0 * k, w, o + r0 * n, r1 - r0, n, k);
    });
}
inline void copy(const float* x, float* o, std::size_t n) { std::copy_n(x, n, o); }

// ---- tables the generator emits ----
struct value { float* data; const std::uint8_t* qdata; std::int64_t rows; };   // layout of forge_aot_value (forge_aot.h)
enum : std::uint8_t { v_arena = 0, v_fixed = 1, v_state = 2 };   // arena slice; weight or constant; state view
struct value_desc { std::uint8_t kind, rank; std::uint32_t dims; std::uint64_t arena_off; const char* codec; };
struct step_desc { const char* low; const char* op; std::uint32_t in_first, n_in, out; };
struct plan_desc {
    std::uint32_t n_values, n_steps; std::uint64_t arena_floats;
    const value_desc* values; const std::int64_t* dims; const step_desc* steps; const std::uint32_t* step_in;
    ops::attrs (*attrs)(std::uint32_t step);
};

// The kernel of a CPU lowering by name, as lower_plan() selects it.
inline kernel_fn kernel(std::string_view low, std::string_view opname) {
    if (low == qweights::gemm_nt_lowering().name) return qweights::gemm_nt;
    if (low == qweights::gather_lowering().name) return qweights::gather_rows;
    if (low == qweights::moe_lowering().name) return qweights::moe_ffn;
    auto o = ops::op_from_name(opname);
    if (!o) raise(errc::not_found, "unknown op '{}'", opname);
    if (low == "reference") return reference_kernel(*o);
    for (auto& l : registry::cpu().lowerings(*o)) if (l.name == low) return l.fn;
    raise(errc::not_found, "this build has no lowering '{}' for {}", low, opname);
}

// The code bound to one plan's buffers.
struct instance {
    const plan_desc* d = nullptr;
    float* arena = nullptr;
    std::vector<tensor_view> v;    // per value
    std::vector<tensor_view> in;   // step inputs, concatenated: step s reads in[steps[s].in_first ...]
    std::vector<kernel_fn> k;      // per step
    std::vector<ops::attrs> a;     // per step
    std::vector<std::uint32_t> dyn;                             // state views: data and row count change per run
    std::vector<std::pair<std::uint32_t, std::uint32_t>> dyn_in;   // (index in `in`, value)
    instance(const plan_desc& pd, const value* vals, float* arena_) : d(&pd), arena(arena_) {
        v.resize(pd.n_values);
        for (std::uint32_t i = 0; i < pd.n_values; ++i) {
            const auto& vd = pd.values[i]; auto& t = v[i];
            t.shape.assign(pd.dims + vd.dims, pd.dims + vd.dims + vd.rank);
            if (vd.kind == v_arena) {
                t.data = arena + vd.arena_off;
                if (vals[i].data != t.data) raise(errc::invalid_argument, "value %{} is not where the code expects it in the arena", i);
            } else {
                t.data = vals[i].data; t.qdata = vals[i].qdata;
                if (vd.codec) {
                    t.qcodec = codecs::find(vd.codec);
                    if (!t.qcodec || !t.qdata) raise(errc::invalid_argument, "value %{} is not an encoded {} weight", i, vd.codec);
                } else if (!t.data && vd.kind == v_fixed) raise(errc::invalid_argument, "value %{} has no data", i);
                if (vd.kind == v_state) dyn.push_back(i);
            }
        }
        k.resize(pd.n_steps); a.resize(pd.n_steps);
        for (std::uint32_t s = 0; s < pd.n_steps; ++s) {
            const auto& sd = pd.steps[s];
            if (!sd.low) continue;   // a state step: the engine runs it
            k[s] = kernel(sd.low, sd.op); a[s] = pd.attrs(s);
            for (std::uint32_t j = 0; j < sd.n_in; ++j) {
                std::uint32_t id = pd.step_in[sd.in_first + j];
                if (in.size() != sd.in_first + j) raise(errc::invalid_argument, "step table out of order");
                in.push_back(v[id]);
                if (pd.values[id].kind == v_state) dyn_in.push_back({sd.in_first + j, id});
            }
        }
    }
    void refresh(const value* vals) {
        for (auto id : dyn) { v[id].data = vals[id].data; if (!v[id].shape.empty()) v[id].shape[0] = vals[id].rows; }
        for (auto [j, id] : dyn_in) { in[j].data = v[id].data; if (!in[j].shape.empty()) in[j].shape[0] = v[id].shape[0]; }
    }
    std::span<const tensor_view> ins(std::uint32_t s) const { return {in.data() + d->steps[s].in_first, d->steps[s].n_in}; }
    void call(std::uint32_t s) { k[s](ins(s), v[d->steps[s].out], a[s]); }
};

// ---- the C entry points (forge_aot_plan_v1) ----
inline void fail(char* err, std::size_t cap, const char* what) { if (err && cap) std::snprintf(err, cap, "%s", what); }
inline void* bind_c(const plan_desc& d, const value* vals, float* arena, char* err, std::size_t cap) {
    try { return new instance(d, vals, arena); }
    catch (const std::exception& e) { fail(err, cap, e.what()); }
    catch (...) { fail(err, cap, "unknown error"); }
    return nullptr;
}
using segment_fn = void (*)(instance&);
inline int run_c(void* inst, const segment_fn* segs, std::uint32_t n_segs, std::uint32_t seg, const value* vals, char* err, std::size_t cap) {
    try {
        if (!inst || seg >= n_segs) { fail(err, cap, "no such compiled segment"); return 1; }
        auto& I = *static_cast<instance*>(inst);
        I.refresh(vals);
        segs[seg](I);
        return 0;
    }
    catch (const std::exception& e) { fail(err, cap, e.what()); }
    catch (...) { fail(err, cap, "unknown error"); }
    return 1;
}
inline void unbind_c(void* inst) { delete static_cast<instance*>(inst); }
// The code needs at least SIMD target `target` (it was compiled for that instruction set).
inline int check_c(const char* target, char* err, std::size_t cap) {
    try {
        if (simd::by_name(target)) return 0;
        std::string m = std::string("compiled for the ") + target + " SIMD target, which this CPU or build does not have (" + simd::active().target + " is active)";
        fail(err, cap, m.c_str());
    } catch (...) { fail(err, cap, "cannot detect CPU features"); }
    return 1;
}

} // namespace forge::graph::aot::rt
