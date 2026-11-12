#pragma once
// Device kernels beyond the per-op ones, for the runtimes that execute the generated OpenCL C (OpenCL, Level Zero):
// quantised matmul_t / gather / FFN gate-up over weights kept encoded on the device, one-pass and split-over-keys
// attention, the Gated DeltaNet state ops, broadcast binaries and the sparse mixture of experts. The kernels are the
// PORT_* section of ops/gen/gpu_gen.py: one work item per output, no workgroup-shared memory, so the same source runs
// on any OpenCL 1.2 device. Entry points and arguments mirror gpu::cuda_hip::runtime (cuda.hpp), which is what
// graph::gpu_registry (graph/vulkan.hpp) probes for with requires-expressions.
//
// CRTP: D is the runtime. It provides (private, with this class as a friend):
//   mu_, in_run_, capturing_, pinned_, res_, dispatches_, stats_, begin_op();
//   struct kcall;                                   one or more kernel launches sharing temporaries
//   k_touch(host, count, mode)                      make a host-keyed buffer resident (no argument is bound)
//   k_buf(kcall&, host, count, mode) -> entry*      bind a host-keyed tensor (mode 0 in, 1 out, 2 in/out)
//   k_dev(kcall&, dev), k_u32, k_f32                bind a raw device buffer / a scalar
//   k_scratch(kcall&, bytes), k_upload(dev, p, n)   temporary device buffer, host -> device copy
//   k_launch(kcall&, kernel, global)                enqueue (and record while capturing); clears the bound arguments
//   k_done(kcall&), k_abort(kcall&)                 write-backs, read-backs outside a run, free the temporaries
//   qweight(bytes*, n) -> dev                       encoded weight uploaded once and cached (evictable, LRU)
//   k_has(kernel)                                   the program has this kernel
// HARDWARE-UNTESTED beyond the CPU mocks (tests/mock_opencl.c, tests/mock_ze.c run this very kernel source as C).
#include <forge/gpu/cuda.hpp>
#include <forge/gpu/determinism.hpp>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace forge::gpu {

template<class D> class portable_kernels {
    D& d() { return static_cast<D&>(*this); }
    static bool off(const char* n) { const char* e = std::getenv(n); return e && std::string(e) == "0"; }
    static bool overlaps(const float* a, std::size_t an, const float* b, std::size_t bn) { return a < b + bn && b < a + an; }
public:
    static constexpr bool has_bcast = true;          // graph::gpu_registry: broadcast binaries run on the device
    static constexpr bool qglu_any_shape = true;     // run_qglu takes every codec and shape (no interleaved layout)
    static constexpr bool exact_activations = true;  // quantised matmuls dot the f32 activations (no q8_1 / int8 rounding)
    static constexpr unsigned attention_splits = 16; // forge_attention_dec: key chunks per (row, head)

    bool has_function(const std::string& n) { std::lock_guard lk(d().mu_); return d().k_has(n); }

    // y[rows, cols] = x[rows, k] . W[cols, k]^T with W's rows encoded (rb bytes each; fmt 1 q8_0, 2 q4_k, 3 q6_k, 4 q5_k).
    bool run_qmatmul(const tensor_view& x, const std::uint8_t* q, std::size_t rb, std::uint32_t cols, std::uint32_t fmt, tensor_view out) {
        if (x.shape.empty() || !q || fmt < 1 || fmt > 4) return false;
        std::uint32_t k = std::uint32_t(x.shape.back()), rows = k ? std::uint32_t(x.size() / k) : 0;
        if (!k || k % 32 || out.size() != std::size_t(rows) * cols) return false;
        std::lock_guard lk(d().mu_); d().begin_op();
        if (!d().k_has("forge_qmatmul_t")) return false;
        ++d().dispatches_["qmatmul_t"];
        void* w = d().qweight(q, rb * cols);
        typename D::kcall c;
        try {
            d().k_touch(x.data, x.size(), 0); d().k_touch(out.data, out.size(), 1);
            d().k_buf(c, x.data, x.size(), 0); d().k_dev(c, w); d().k_buf(c, out.data, out.size(), 1);
            for (std::uint32_t v : {rows, cols, k, fmt, std::uint32_t(rb)}) d().k_u32(c, v);
            d().k_launch(c, "forge_qmatmul_t", std::size_t(rows) * cols);
            d().k_done(c);
        } catch (...) { d().k_abort(c); throw; }
        return true;
    }
    // FFN gate/up/SiLU: y[rows, n] = silu(x . Wg^T) * (x . Wu^T) in one launch (forge_qglu), both weights [n, k] of one
    // codec. Bit-identical to run_qmatmul twice + silu_mul on this backend (same dot order, same expression).
    bool run_qglu(const tensor_view& x, const std::uint8_t* qg, const std::uint8_t* qu, std::size_t rb, std::uint32_t n, std::uint32_t fmt, tensor_view out) {
        if (x.shape.empty() || !qg || !qu || fmt < 1 || fmt > 4) return false;
        std::uint32_t k = std::uint32_t(x.shape.back()), rows = k ? std::uint32_t(x.size() / k) : 0;
        if (!k || k % 32 || out.size() != std::size_t(rows) * n) return false;
        std::lock_guard lk(d().mu_); d().begin_op();
        if (!d().k_has("forge_qglu")) return false;
        ++d().dispatches_["qglu"];
        void* wg = d().qweight(qg, rb * n); void* wu = d().qweight(qu, rb * n);
        wg = d().qweight(qg, rb * n);   // again: the second upload may have evicted the first under memory pressure
        typename D::kcall c;
        try {
            d().k_touch(x.data, x.size(), 0); d().k_touch(out.data, out.size(), 1);
            d().k_buf(c, x.data, x.size(), 0); d().k_dev(c, wg); d().k_dev(c, wu); d().k_buf(c, out.data, out.size(), 1);
            for (std::uint32_t v : {rows, n, k, fmt, std::uint32_t(rb)}) d().k_u32(c, v);
            d().k_launch(c, "forge_qglu", std::size_t(rows) * n);
            d().k_done(c);
        } catch (...) { d().k_abort(c); throw; }
        return true;
    }
    // Embedding gather over an encoded table [nrows, dim] (forge_qgather); out-of-range indices give zero rows.
    bool run_qgather(const tensor_view& idx, const std::uint8_t* q, std::size_t rb, std::uint32_t nrows, std::uint32_t dim, std::uint32_t fmt, tensor_view out) {
        if (!q || fmt < 1 || fmt > 4 || !dim || dim % 32 || out.size() != idx.size() * dim) return false;
        std::lock_guard lk(d().mu_); d().begin_op();
        if (!d().k_has("forge_qgather")) return false;
        ++d().dispatches_["qgather"];
        void* w = d().qweight(q, rb * nrows);
        std::uint32_t n = std::uint32_t(idx.size());
        typename D::kcall c;
        try {
            d().k_touch(idx.data, idx.size(), 0); d().k_touch(out.data, out.size(), 1);
            d().k_buf(c, idx.data, idx.size(), 0); d().k_dev(c, w); d().k_buf(c, out.data, out.size(), 1);
            for (std::uint32_t v : {n, dim, fmt, std::uint32_t(rb), nrows}) d().k_u32(c, v);
            d().k_launch(c, "forge_qgather", std::size_t(n) * dim);
            d().k_done(c);
        } catch (...) { d().k_abort(c); throw; }
        return true;
    }
    // Broadcast add/sub/mul/div (op 0..3, cuda_hip::bcast_op): y[i] = a[i % ma] op b[i % mb].
    bool run_bcast(int op, const std::vector<tensor_view>& in, tensor_view out) {
        std::uint32_t n = std::uint32_t(out.size()), ma = std::uint32_t(in[0].size()), mb = std::uint32_t(in[1].size());
        if (!ma || !mb || n % ma || n % mb) return false;
        std::lock_guard lk(d().mu_); d().begin_op();
        if (!d().k_has("forge_bcast")) return false;
        ++d().dispatches_["bcast"];
        typename D::kcall c;
        try {
            d().k_touch(in[0].data, ma, 0); d().k_touch(in[1].data, mb, 0); d().k_touch(out.data, n, 1);
            d().k_buf(c, in[0].data, ma, 0); d().k_buf(c, in[1].data, mb, 0); d().k_buf(c, out.data, n, 1);
            for (std::uint32_t v : {n, ma, mb, std::uint32_t(op)}) d().k_u32(c, v);
            d().k_launch(c, "forge_bcast", n);
            d().k_done(c);
        } catch (...) { d().k_abort(c); throw; }
        return true;
    }
    // Gated DeltaNet state ops (forge_delta_rule / forge_causal_conv) with the state slot S resident and pinned.
    // Only inside a run; false = run it on the host.
    bool run_state(const std::string& fn, const std::vector<tensor_view>& in, float* S, std::size_t sn, tensor_view out, const ops::attrs& a) {
        std::lock_guard lk(d().mu_); d().begin_op();
        if (!d().in_run_ || !d().k_has("forge_" + fn)) return false;
        std::uint32_t u[6] = {}; std::size_t grid = 0; float sc = 0;
        if (fn == "delta_rule" && in.size() == 5) {
            u[0] = std::uint32_t(in[0].shape[0]); u[1] = std::uint32_t(in[0].shape[1]); u[2] = std::uint32_t(in[0].shape[2]);
            u[3] = std::uint32_t(in[2].shape[1]); u[4] = std::uint32_t(in[2].shape[2]); u[5] = a.num("l2norm", 1) != 0;
            if (!u[1] || u[3] % u[1] || !u[2]) return false;
            sc = float(a.num("scale", 1.0 / std::sqrt(double(u[2])))); grid = std::size_t(u[3]) * u[4];
        } else if (fn == "causal_conv" && in.size() == 2) {
            u[0] = std::uint32_t(in[0].shape[0]); u[1] = std::uint32_t(in[0].shape[1]); u[2] = std::uint32_t(in[1].shape[1]);
            if (u[2] < 1u || u[2] > 9u) return false;
            grid = u[1];
        } else return false;
        // the kernels are not in-place safe: an output overlapping an input or the state goes to the host path
        for (auto& v : in) if (overlaps(out.data, out.size(), v.data, v.size())) return false;
        if (overlaps(out.data, out.size(), S, sn)) return false;
        bool known = false; for (auto& [p, n] : d().pinned_) if (p == S && n == sn) known = true;
        if (!known) d().pinned_.push_back({S, sn});
        ++d().dispatches_[fn];
        typename D::kcall c;
        try {
            for (auto& v : in) d().k_touch(v.data, v.size(), 0);
            d().k_touch(S, sn, 0); d().k_touch(out.data, out.size(), 1);
            for (auto& v : in) d().k_buf(c, v.data, v.size(), 0);
            if (auto* e = d().k_buf(c, S, sn, 2)) e->pinned = true;
            d().k_buf(c, out.data, out.size(), 1);
            if (fn == "delta_rule") { for (int i = 0; i < 6; ++i) d().k_u32(c, u[i]); d().k_f32(c, sc); }
            else for (int i = 0; i < 3; ++i) d().k_u32(c, u[i]);
            d().k_launch(c, "forge_" + fn, grid);
            d().k_done(c);
        } catch (...) { d().k_abort(c); throw; }
        return true;
    }
    // Sparse mixture of experts (graph/plan.hpp moe_ffn semantics): h [t, d]; route holds 3 floats per (token, slot)
    // pair (expert id, token row, weight), kk slots per token, computed by the host router; gate/up are stacked
    // [e, f, d] and down [e, d, f], uploaded once and kept on the device. Only the routed experts' rows are read.
    bool run_moe(const tensor_view& h, const std::vector<float>& route, std::uint32_t kk, const moe_weight& wg, const moe_weight& wu, const moe_weight& wd,
                 std::uint32_t f, tensor_view out) {
        if (h.shape.size() != 2 || !kk || !f) return false;
        std::uint32_t t = std::uint32_t(h.shape[0]), dim = std::uint32_t(h.shape[1]), np = t * kk;
        if (route.size() != std::size_t(np) * 3 || out.size() != h.size() || !wg.data || !wu.data || !wd.data) return false;
        for (auto* w : {&wg, &wu}) if (w->fmt > 4 || (w->fmt && dim % 32)) return false;
        if (wd.fmt > 4 || (wd.fmt && f % 32)) return false;
        std::lock_guard lk(d().mu_); d().begin_op();
        if (!d().k_has("forge_moe_hidden") || !d().k_has("forge_moe_down")) return false;
        ++d().dispatches_["moe_ffn"];
        if (!np) return true;
        typename D::kcall c;
        try {
            void* g = nullptr; void* up = nullptr; void* dn = nullptr;
            for (int pass = 0; pass < 2; ++pass) {   // twice: an upload may evict a weight an earlier call returned
                g = d().qweight(wg.data, wg.bytes); up = d().qweight(wu.data, wu.bytes); dn = d().qweight(wd.data, wd.bytes); }
            d().k_touch(h.data, h.size(), 0); d().k_touch(out.data, out.size(), 1);
            void* rt = d().k_scratch(c, route.size() * 4); d().k_upload(rt, route.data(), route.size() * 4);
            void* hid = d().k_scratch(c, std::size_t(np) * f * 4);
            d().k_buf(c, h.data, h.size(), 0); d().k_dev(c, rt); d().k_dev(c, g); d().k_dev(c, up); d().k_dev(c, hid);
            for (std::uint32_t v : {np, f, dim, wg.fmt, wg.rb, wu.fmt, wu.rb}) d().k_u32(c, v);
            d().k_launch(c, "forge_moe_hidden", std::size_t(np) * f);
            d().k_dev(c, hid); d().k_dev(c, rt); d().k_dev(c, dn); d().k_buf(c, out.data, out.size(), 1);
            for (std::uint32_t v : {t, kk, dim, f, wd.fmt, wd.rb}) d().k_u32(c, v);
            d().k_launch(c, "forge_moe_down", std::size_t(t) * dim);
            d().k_done(c);
        } catch (...) { d().k_abort(c); throw; }
        return true;
    }

    // A spec op whose kernel takes (inputs..., output, scalars...) exactly as cuda_hip::kernel_args lays them out: the
    // encoder / vision kernels (layer_norm, group_norm, conv1d, conv2d, pooling, attention_bias; gpu_gen.py PORT_NN).
    // OpenCL, Level Zero and Metal reach the same kernels through their generic run_op; this is the entry for runtimes
    // whose run_op is not generic (Vulkan). false = no such kernel on this device; the caller checks the shapes
    // (cuda_hip::supports).
    bool run_nn(const std::string& op, const std::vector<tensor_view>& in, const ops::attrs& a, tensor_view out) {
        auto L = cuda_hip::kernel_args(op, in, a, out);
        std::lock_guard lk(d().mu_); d().begin_op();
        if (!d().k_has("forge_" + op)) return false;
        ++d().dispatches_[op];
        typename D::kcall c;
        try {
            for (auto& v : in) d().k_touch(v.data, v.size(), 0);
            d().k_touch(out.data, out.size(), 1);
            for (auto& v : in) d().k_buf(c, v.data, v.size(), 0);
            d().k_buf(c, out.data, out.size(), 1);
            for (auto& sc : L.scalars) { if (sc.is_float) d().k_f32(c, sc.f); else d().k_u32(c, sc.u); }
            d().k_launch(c, "forge_" + op, L.threads);
            d().k_done(c);
        } catch (...) { d().k_abort(c); throw; }
        return true;
    }

protected:
    // Attention kernel selection for run_op (plain f32 K/V): 1 = ran, -1 = use forge_attention.
    //   rows x heads <= 256 (decode): forge_attention_dec + forge_attention_comb over attention_splits key chunks;
    //     off with FORGE_ATTN_DEC=0 and in determinism mode.
    //   otherwise (prefill): forge_attention_fa, one pass with an online softmax; off with FORGE_GPU_NO_FA.
    int attention_port(const std::vector<tensor_view>& in, const ops::attrs& a, tensor_view out) {
        if (a.num("kv_codec_k", 0) != 0 || a.num("kv_codec_v", 0) != 0) return -1;
        auto L = cuda_hip::kernel_args("attention", in, a, out);
        std::uint32_t s = L.scalars[0].u, hq = L.scalars[1].u, dv = L.scalars[5].u;
        const bool dec = s * hq <= 256u && !off("FORGE_ATTN_DEC") && !deterministic();
        if (!dec && std::getenv("FORGE_GPU_NO_FA")) return -1;
        std::lock_guard lk(d().mu_); d().begin_op();
        if (dec ? !(d().k_has("forge_attention_dec") && d().k_has("forge_attention_comb")) : !d().k_has("forge_attention_fa")) return -1;
        ++d().dispatches_["attention"]; ++d().dispatches_[dec ? "attention_dec" : "attention_fa"];
        typename D::kcall c;
        try {
            for (auto& v : in) d().k_touch(v.data, v.size(), 0);
            d().k_touch(out.data, out.size(), 1);
            for (auto& v : in) d().k_buf(c, v.data, v.size(), 0);
            auto scalars = [&] { for (auto& sc : L.scalars) { if (sc.is_float) d().k_f32(c, sc.f); else d().k_u32(c, sc.u); } };
            if (dec) {
                const std::uint32_t ns = attention_splits; const std::size_t nrow = std::size_t(s) * hq;
                void* part = d().k_scratch(c, nrow * ns * (dv + 2) * 4);
                d().k_dev(c, part); scalars(); d().k_u32(c, ns);
                d().k_launch(c, "forge_attention_dec", nrow * ns);
                d().k_dev(c, part); d().k_buf(c, out.data, out.size(), 1);
                for (std::uint32_t v : {std::uint32_t(nrow), ns, dv}) d().k_u32(c, v);
                d().k_launch(c, "forge_attention_comb", nrow * dv);
            } else {
                d().k_buf(c, out.data, out.size(), 1); scalars();
                d().k_launch(c, "forge_attention_fa", L.threads);
            }
            d().k_done(c);
        } catch (...) { d().k_abort(c); throw; }
        return 1;
    }
};

} // namespace forge::gpu
