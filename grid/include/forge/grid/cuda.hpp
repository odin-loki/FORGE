#pragma once
// CUDA / HIP host runtime. The driver library (libcuda / libamdhip64) is dlopened at run time, so FORGE has
// no link or build dependency on either SDK. Kernels come from the same generated single source as the
// compile checks (spec/gen/gpu_gen.py -> kernels.cu), built at FORGE build time into PTX (CUDA, JIT-compiled
// by the driver for the installed GPU) and an AMDGPU code object (HIP), and embedded below.
// Without a driver or a device, open() fails with a reason and callers fall back to the CPU.
//
// Features (HARDWARE-UNTESTED beyond the CPU mock driver in tests/mock_cuda.c):
//   * every generated kernel (elementwise, softmax, rms_norm, matmul_t, rope, gather, slice, attention with
//     int8/fp8 KV codecs);
//   * one stream per runtime, async copies, events for timing;
//   * residency protocol (same contract as the Vulkan backend): between begin_run()/end_run() buffers stay
//     on the device keyed by host address, pinned ranges (weights) survive across runs;
//   * stream capture into a graph and replay (`capture_begin/capture_end/graph_launch`) where the driver
//     exports the graph API;
//   * multiple devices (`open(api, ordinal)`, `device_count`) and peer copies between runtimes.
#include <cstdlib>
#include <forge/grid/gpu_images.gen.hpp>
#include <forge/core/tensor.hpp>
#include <forge/spec/ops.gen.hpp>
#include <forge/core/dl.hpp>
#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace forge::grid::gpu {

enum class api { cuda, hip };

// Process-wide driver call counters for the benchmark harness (forge bench, tools/bench_llm.sh): how many
// kernels are issued (directly or recorded into a captured graph), graphs replayed, copies and syncs. Relaxed
// atomics, so the cost is one uncontended increment per call.
struct call_counters {
    std::atomic<std::uint64_t> launches{0}, captured{0}, graph_launches{0}, dtoh{0}, htod{0}, dtod{0}, syncs{0}, instantiates{0}, node_updates{0};
    struct snapshot { std::uint64_t launches = 0, captured = 0, graph_launches = 0, dtoh = 0, htod = 0, dtod = 0, syncs = 0, instantiates = 0, node_updates = 0; };
    snapshot read() const {
        auto r = [](const std::atomic<std::uint64_t>& a) { return a.load(std::memory_order_relaxed); };
        return {r(launches), r(captured), r(graph_launches), r(dtoh), r(htod), r(dtod), r(syncs), r(instantiates), r(node_updates)};
    }
};
inline call_counters& counters() { static call_counters c; return c; }
inline void count(std::atomic<std::uint64_t>& a) { a.fetch_add(1, std::memory_order_relaxed); }
inline const char* api_name(api a) { return a == api::cuda ? "cuda" : "hip"; }

// Kernels in kernels.cu and how their arguments are laid out.
enum class sig { unary, binary, scale, rows, rms, matmul, rope, gather, slice, attention };
inline bool kernel_sig(const std::string& op, sig& s) {
    static const std::map<std::string, sig> m = [] {
        std::map<std::string, sig> r;
        for (auto n : {"neg", "abs", "sqrt", "rsqrt", "recip", "square", "exp", "exp2", "log", "log2", "tanh", "sigmoid",
                       "silu", "relu", "softplus", "gelu_tanh"}) r[n] = sig::unary;
        for (auto n : {"add", "sub", "mul", "div", "maximum", "minimum", "silu_mul"}) r[n] = sig::binary;
        r["scale"] = sig::scale; r["softmax"] = sig::rows; r["rms_norm"] = sig::rms; r["matmul_t"] = sig::matmul;
        r["rope"] = sig::rope; r["gather"] = sig::gather; r["slice"] = sig::slice; r["attention"] = sig::attention;
        return r; }();
    auto it = m.find(op); if (it == m.end()) return false; s = it->second; return true;
}

inline std::vector<std::pair<std::string, std::string_view>> module_images(api a) {
    std::vector<std::pair<std::string, std::string_view>> r;
    if (a == api::cuda) { if (images::ptx_size) r.push_back({"ptx", {images::ptx, images::ptx_size}}); }
    else for (std::size_t i = 0; i < images::hsaco_count; ++i)
        r.push_back({images::hsaco[i].arch, {reinterpret_cast<const char*>(images::hsaco[i].data), images::hsaco[i].size}});
    return r;
}

// Shape rules for the kernels above (mirrors run_op); false = no CUDA/HIP lowering, use the CPU.
inline bool supports(const std::string& op, const std::vector<shape_t>& in, const spec::attrs& a) {
    sig s; if (!kernel_sig(op, s) || in.empty()) return false;
    auto& x = in[0];
    switch (s) {
    case sig::unary: case sig::scale: return in.size() == 1;
    case sig::binary: return in.size() == 2 && (in[1] == x || numel(in[1]) == numel(x));   // [n] op [1, n]: same elements
    case sig::rows: return !x.empty() && (a.num("axis", -1) == -1 || a.num("axis", -1) == double(x.size()) - 1);
    case sig::rms: return in.size() == 2 && !x.empty() && in[1].size() == 1 && in[1][0] == x.back();
    case sig::matmul: return in.size() == 2 && x.size() >= 2 && in[1].size() == 2 && in[1][1] == x.back();
    case sig::rope: return in.size() == 2 && x.size() == 3 && x[2] % 2 == 0 && numel(in[1]) == x[0];
    case sig::gather: return in.size() == 2 && x.size() == 2;
    case sig::slice: return in.size() == 1 && !x.empty() && a.num("step", 1) == 1;
    case sig::attention: return in.size() == 3 && x.size() == 3 && in[1].size() == 3 && in[2].size() == 3 && in[1][1] > 0 && x[1] % in[1][1] == 0;
    }
    return false;
}

// Broadcast binaries the CUDA/HIP runtime runs with forge_bcast: the smaller input's shape (leading 1s dropped)
// is a suffix of the larger one's. Other GPU backends keep these on the CPU.
inline int bcast_op(const std::string& op, const std::vector<shape_t>& in) {
    int o = op == "add" ? 0 : op == "sub" ? 1 : op == "mul" ? 2 : op == "div" ? 3 : -1;
    if (o < 0 || in.size() != 2 || numel(in[0]) == numel(in[1])) return -1;
    auto strip = [](shape_t s) { while (s.size() > 1 && s[0] == 1) s.erase(s.begin()); return s; };
    shape_t a = strip(in[0]), b = strip(in[1]);
    const shape_t& big = numel(in[0]) > numel(in[1]) ? a : b; const shape_t& sm = numel(in[0]) > numel(in[1]) ? b : a;
    if (sm.size() > big.size() || numel(sm) == 0) return -1;
    for (std::size_t i = 0; i < sm.size(); ++i) if (sm[sm.size() - 1 - i] != big[big.size() - 1 - i]) return -1;
    return o;
}

// Scalar arguments and launch size for a kernel, shared by the OpenCL and Level Zero runtimes (cuda.hpp's own
// run_op packs the same values). Buffers come first in kernel order (inputs, the rope table after pos, output).
struct scalar { bool is_float; std::uint32_t u; float f; };
struct launch_args { sig s{}; std::uint32_t threads = 0; std::vector<scalar> scalars; std::vector<float> rope_inv; };
inline launch_args kernel_args(const std::string& op, const std::vector<tensor_view>& in, const spec::attrs& a, const tensor_view& out) {
    launch_args L; kernel_sig(op, L.s);
    auto& x = in[0];
    std::uint32_t n = std::uint32_t(out.size()); L.threads = n;
    auto U = [&](std::uint32_t v) { L.scalars.push_back({false, v, 0.f}); };
    auto F = [&](float v) { L.scalars.push_back({true, 0, v}); };
    switch (L.s) {
    case sig::unary: case sig::binary: U(n); break;
    case sig::scale: U(n); F(float(a.num("alpha", 1.0))); break;
    case sig::rows: case sig::rms: {
        std::uint32_t cols = std::uint32_t(x.shape.back()), rows = cols ? std::uint32_t(x.size()) / cols : 0;
        L.threads = rows; U(rows); U(cols); if (L.s == sig::rms) F(float(a.num("eps", 1e-6))); break; }
    case sig::matmul: {
        std::uint32_t k = std::uint32_t(x.shape.back()), rows = std::uint32_t(x.size()) / k, cols = std::uint32_t(in[1].shape[0]);
        L.threads = rows * cols; U(rows); U(cols); U(k); break; }
    case sig::rope: {
        std::uint32_t rows = std::uint32_t(x.shape[0]), cols = std::uint32_t(x.shape[1]), k = std::uint32_t(x.shape[2]);
        double base = a.num("base", 10000.0); L.rope_inv.resize(k / 2);
        for (std::uint32_t j = 0; j < k / 2; ++j) L.rope_inv[j] = float(std::pow(base, -2.0 * double(j) / double(k)));
        L.threads = rows * cols * (k / 2); U(rows); U(cols); U(k); U(a.num("neox", 1) != 0); break; }
    case sig::gather: U(n); U(std::uint32_t(x.shape[0])); U(std::uint32_t(x.shape[1])); break;
    case sig::slice: {
        auto ax = std::size_t(spec::norm_axis(i64(a.num("axis", 0)), x.shape.size()));
        i64 inner = 1; for (std::size_t d = ax + 1; d < x.shape.size(); ++d) inner *= x.shape[d];
        i64 st = i64(a.num("start", 0)); if (st < 0) st += x.shape[ax];
        U(n); U(std::uint32_t(x.shape[ax])); U(std::uint32_t(inner)); U(std::uint32_t(st)); U(std::uint32_t(out.shape[ax])); break; }
    case sig::attention: {
        auto kc = std::uint32_t(a.num("kv_codec_k", 0)), vc = std::uint32_t(a.num("kv_codec_v", 0));
        if (std::uint32_t dd = std::uint32_t(x.shape[2]), dvv = std::uint32_t(a.num("kv_dv", 0)); (kc >= 3u && dd > 256u) || (vc >= 3u && dvv > 256u)) raise(errc::unsupported, "TurboQuant KV on the GPU supports head dims up to 256");
        if (kc >= 5u || vc >= 5u) raise(errc::unsupported, "TurboQuant QJL KV types (tbq4p, tbq3p) run on the CPU only");
        std::uint32_t s = std::uint32_t(x.shape[0]), hq = std::uint32_t(x.shape[1]), d = std::uint32_t(x.shape[2]);
        U(s); U(hq); U(d); U(std::uint32_t(in[1].shape[0])); U(std::uint32_t(in[1].shape[1]));
        U(vc ? std::uint32_t(a.num("kv_dv", 0)) : std::uint32_t(in[2].shape[2]));
        U((a.num("causal", 1) != 0 ? 1u : 0u) | (kc << 1) | (vc << 4) | (std::uint32_t(a.num("window", 0)) << 7));
        F(float(a.num("scale", 1.0 / std::sqrt(double(d))))); L.threads = s * hq; break; }
    }
    return L;
}

class runtime {
public:
    static int device_count(api a) {
        runtime probe(a);
        if (!probe.load_lib().empty()) return 0;
        if (probe.f_.init(0) != 0) return 0;
        int n = 0; return probe.f_.count(&n) == 0 ? n : 0;
    }
    static std::unique_ptr<runtime> open(api a, std::string* why = nullptr, int ordinal = 0) {
        std::unique_ptr<runtime> r(new runtime(a));
        r->ordinal_ = ordinal;
        std::string e = r->init();
        if (!e.empty()) { if (why) *why = std::string(api_name(a)) + ": " + e; return nullptr; }
        return r;
    }
    ~runtime() {
        try { std::lock_guard lk(mu_); for (auto& [k, e] : res_) f_.free_(e.d); res_.clear(); for (auto& [k, d] : qres_) if (d) f_.free_(d); qres_.clear();
            for (auto& [k, d] : pool_) f_.free_(d); pool_.clear(); for (auto& [k, d] : rope_) f_.free_(d); rope_.clear(); if (q81_) f_.free_(q81_); } catch (...) {}
        for (auto& c : segc_) drop_seg(c);
        for (auto& [e, g] : graphs_) { if (f_.gexec_destroy) f_.gexec_destroy(e); if (f_.graph_destroy) f_.graph_destroy(g); }
        if (stream_ && f_.stream_destroy) f_.stream_destroy(stream_);
        if (mod_ && f_.unload) f_.unload(mod_);
        if (ctx_ && f_.ctx_destroy) { f_.ctx_destroy(ctx_); ctx_epoch().fetch_add(1, std::memory_order_acq_rel); }
        if (lib_) dlclose(lib_);
    }
    api which() const { return api_; }
    const char* backend() const { return api_name(api_); }
    int ordinal() const { return ordinal_; }
    const std::string& device_name() const { return name_; }
    const std::string& image_arch() const { return arch_; }
    bool has_graphs() const { return f_.capture_begin && f_.capture_end && f_.graph_inst && f_.graph_launch; }
    bool has_streams() const { return stream_ != nullptr; }
    std::map<std::string, std::size_t> dispatches() { std::lock_guard lk(mu_); return dispatches_; }
    struct mem_info { std::size_t free = 0, total = 0; };
    mem_info memory() { mem_info m; if (f_.mem_info) { bind(); f_.mem_info(&m.free, &m.total); } return m; }
    struct residency_stats { std::size_t uploads = 0, downloads = 0, resident = 0, resident_bytes = 0, peak_resident_bytes = 0, qweight_bytes = 0; };
    residency_stats residency() { std::lock_guard lk(mu_); auto s = stats_; s.resident = res_.size(); return s; }

    // ---- residency protocol (see flow::device_hooks) ----
    void begin_run() { std::lock_guard lk(mu_); q81_drop(); in_run_ = true; seg_ = 0; }
    void end_run() {
        std::lock_guard lk(mu_); bind();
        for (auto it = res_.begin(); it != res_.end();) {
            auto& e = it->second;
            if (!e.host_valid) download(e);
            if (e.pinned) { ++it; continue; }
            drop(e); it = res_.erase(it);
        }
        sync(); in_run_ = false;
    }
    void end_run_discard() {   // like end_run, but non-pinned intermediates go back to the pool without a download
        std::lock_guard lk(mu_); bind();
        for (auto it = res_.begin(); it != res_.end();) {
            auto& e = it->second;
            if (e.kept) { ++it; continue; }   // stays resident, host copy fetched on demand
            if (e.pinned) { if (!e.host_valid) download(e); ++it; continue; }
            drop(e); it = res_.erase(it);
        }
        sync(); in_run_ = false;
    }
    void keep(const float* p, std::size_t n) { std::lock_guard lk(mu_); each_overlap(p, n, [&](entry& e) { e.kept = true; }); }
    // a plan arena held as one device buffer across runs (flow::device_hooks::arena): intermediates are sub-ranges
    // of it, so they keep their device addresses run after run
    void arena(float* h, std::size_t n) {
        std::lock_guard lk(mu_); bind();
        auto key = reinterpret_cast<std::uintptr_t>(h);
        if (auto it = res_.find(key); it != res_.end() && it->second.count == n && it->second.arena) return;
        auto b = acquire(h, n, false); b.e->kept = true; b.e->arena = true;
    }
    void forget(const float* p, std::size_t n) {   // drop device copies without fetching them
        std::lock_guard lk(mu_); bind(); flush();
        std::vector<std::uintptr_t> ks;
        each_overlap(p, n, [&](entry& e) { ks.push_back(reinterpret_cast<std::uintptr_t>(e.host)); });
        for (auto k : ks) { drop(res_[k]); res_.erase(k); }
    }
    bool resident() { std::lock_guard lk(mu_); return in_run_; }
    void pin(const float* p, std::size_t n) { std::lock_guard lk(mu_); pinned_.push_back({p, n}); }
    void to_host(const float* p, std::size_t n) {
        std::lock_guard lk(mu_); bind();
        bool any = false;
        each_overlap(p, n, [&](entry& e) {
            if (e.host_valid) return;
            if (!e.arena) { download(e); any = true; return; }
            auto el = reinterpret_cast<std::uintptr_t>(e.host), lo = std::max(el, reinterpret_cast<std::uintptr_t>(p)),
                 hi = std::min(el + e.count * 4, reinterpret_cast<std::uintptr_t>(p) + n * 4);   // just the asked range
            if (hi > lo) { ck(dtoh_s(reinterpret_cast<void*>(lo), e.d + (lo - el), hi - lo, stream_), "memcpy DtoH"); ++stats_.downloads; any = true; }
        });
        if (any) sync();
    }
    void host_wrote(const float* p, std::size_t n) {
        std::lock_guard lk(mu_); bind();
        auto lo = reinterpret_cast<std::uintptr_t>(p), hi = lo + n * 4;
        each_overlap(p, n, [&](entry& e) {
            auto el = reinterpret_cast<std::uintptr_t>(e.host), eh = el + e.count * 4;
            auto a = std::max(lo, el), b = std::min(hi, eh);
            ck(htod_s(e.d + (a - el), reinterpret_cast<const void*>(a), b - a, stream_), "memcpy HtoD");
        });
        sync();
    }
    void release(const float* p, std::size_t n) {
        std::lock_guard lk(mu_); bind();
        std::vector<std::uintptr_t> ks;
        each_overlap(p, n, [&](entry& e) { if (!e.host_valid) download(e); ks.push_back(reinterpret_cast<std::uintptr_t>(e.host)); });
        if (!ks.empty()) sync();
        for (auto k : ks) { drop(res_[k]); res_.erase(k); }
    }
    static constexpr bool has_bcast = true;
    bool run_bcast(int op, const std::vector<tensor_view>& in, tensor_view out) {
        std::uint32_t n = std::uint32_t(out.size()), ma = std::uint32_t(in[0].size()), mb = std::uint32_t(in[1].size()), o = std::uint32_t(op);
        if (!ma || !mb || n % ma || n % mb) return false;
        std::lock_guard lk(mu_); bind();
        if (!has_fn_locked("forge_bcast")) return false;
        if (!in_run_) {   // outside a run the views are host memory
            for (std::uint32_t i = 0; i < n; ++i) { float x = in[0].data[i % ma], z = in[1].data[i % mb];
                out.data[i] = o == 0 ? x + z : o == 1 ? x - z : o == 2 ? x * z : x / z; }
            return true;
        }
        ++dispatches_["bcast"];
        auto ad = acquire(in[0].data, in[0].size(), false).ptr;
        auto bd = acquire(in[1].data, in[1].size(), false).ptr;
        ad = acquire(in[0].data, in[0].size(), false).ptr;
        auto y = acquire(out.data, out.size(), true); y.e->host_valid = false; std::uint64_t yd = y.ptr;
        void* args[] = {&ad, &bd, &yd, &n, &ma, &mb, &o};
        writes(yd, std::size_t(n) * 4);
        if (n) ck(launch_s(function("forge_bcast"), (n + 255) / 256, 1, 1, 256, 1, 1, 0, stream_, args, nullptr), "launch");
        return true;
    }
    // dst_base[off + r * ds + c] = src[r * w + c] on the device (concat)
    bool copy2d_resident(float* src, std::size_t sn, float* dst_base, std::size_t dst_count, std::size_t rows, std::size_t w, std::size_t ds, std::size_t off) {
        std::lock_guard lk(mu_); bind();
        if (!has_fn_locked("forge_copy2d")) return false;
        if (!rows || !w) return true;
        ++dispatches_["copy2d"];
        auto d = acquire(dst_base, dst_count, false);
        auto s = acquire(src, sn, false);
        d = acquire(dst_base, dst_count, false);
        d.e->host_valid = false;
        std::uint64_t sp = s.ptr, dp = d.ptr + off * 4; std::uint32_t r = std::uint32_t(rows), ww = std::uint32_t(w), dss = std::uint32_t(ds);
        void* args[] = {&sp, &dp, &r, &ww, &dss};
        writes(d.ptr, dst_count * 4);
        ck(launch_s(function("forge_copy2d"), unsigned((rows * w + 255) / 256), 1, 1, 256, 1, 1, 0, stream_, args, nullptr), "launch");
        return true;
    }
    void copy_resident(float* src, std::size_t n, float* dst_base, std::size_t dst_count, std::size_t off) {
        std::lock_guard lk(mu_); bind();
        if (!n) return;
        auto d = acquire(dst_base, dst_count, false);
        auto s = acquire(src, n, false);
        d = acquire(dst_base, dst_count, false);
        d.e->host_valid = false;
        ck(dtod_s(d.ptr + off * 4, s.ptr, n * 4), "memcpy DtoD");
    }

    // ---- graphs: capture the stream's work between capture_begin/capture_end and replay it ----
    void capture_begin() { std::lock_guard lk(mu_); need_graphs(); bind(); flush(); user_cap_ = true; ck(f_.capture_begin(stream_, 2 /* RELAXED: run_op may allocate an output buffer while capturing */), "stream begin capture"); }
    std::size_t capture_end() { user_cap_ = false;
        std::lock_guard lk(mu_); need_graphs(); bind();
        void* g = nullptr; ck(f_.capture_end(stream_, &g), "stream end capture");
        void* e = nullptr; ck(f_.graph_inst(&e, g, 0), "graph instantiate");
        graphs_.push_back({e, g}); return graphs_.size() - 1;
    }
    void graph_launch(std::size_t id) { std::lock_guard lk(mu_); need_graphs(); bind(); gpu::count(counters().graph_launches); ck(f_.graph_launch(graphs_.at(id).first, stream_), "graph launch"); sync(); }

    // ---- multi-device: copy device memory to another runtime (peer copy where available, else via host) ----
    void copy_to_peer(runtime& dst, std::uint64_t dst_ptr, std::uint64_t src_ptr, std::size_t bytes) {
        if (f_.peer && dst.api_ == api_) {
            if (api_ == api::cuda) ck(f_.peer(dst_ptr, reinterpret_cast<std::uintptr_t>(dst.ctx_), src_ptr, reinterpret_cast<std::uintptr_t>(ctx_), bytes), "memcpy peer");
            else ck(f_.peer(dst_ptr, std::uintptr_t(dst.ordinal_), src_ptr, std::uintptr_t(ordinal_), bytes), "memcpy peer");
            return;
        }
        std::vector<char> tmp(bytes);
        { std::lock_guard lk(mu_); bind(); ck(dtoh_s(tmp.data(), src_ptr, bytes, stream_), "memcpy DtoH"); sync(); }
        std::lock_guard lk(dst.mu_); dst.bind(); dst.ck(dst.htod_s(dst_ptr, tmp.data(), bytes, dst.stream_), "memcpy HtoD"); dst.sync();
    }
    std::uint64_t device_alloc(std::size_t bytes) { std::lock_guard lk(mu_); bind(); std::uint64_t p = 0; ck(f_.malloc_(&p, std::max<std::size_t>(4, bytes)), "malloc"); return p; }
    void device_free(std::uint64_t p) { std::lock_guard lk(mu_); bind(); f_.free_(p); }
    void upload(std::uint64_t d, const void* h, std::size_t bytes) { std::lock_guard lk(mu_); bind(); ck(htod_s(d, h, bytes, stream_), "memcpy HtoD"); sync(); }
    void download(void* h, std::uint64_t d, std::size_t bytes) { std::lock_guard lk(mu_); bind(); ck(dtoh_s(h, d, bytes, stream_), "memcpy DtoH"); sync(); }

    // Quantised matmul_t (forge_qmatmul_t): y[rows, cols] = x[rows, k] . W[cols, k]^T with W in its encoded rows
    // (fmt 1 q8_0, 2 q4_k, 3 q6_k, 4 q5_k; rb bytes per row). The encoded weight is uploaded once and stays on the
    // device for the runtime's lifetime (keyed by host address and size); x/out follow the residency protocol.
    bool has_function(const std::string& n) {
        std::lock_guard lk(mu_); bind();
        if (fn_.count(n)) return true;
        void* f = nullptr; if (f_.get_fn(&f, mod_, n.c_str()) != 0 || !f) return false;
        fn_[n] = f; return true;
    }
    // rows at or above which run_qmatmul uses the tiled MMQ kernel ($FORGE_MMQ_MIN_ROWS, default 16; 0 disables)
    static std::uint32_t mmq_min_rows() {
        static const std::uint32_t v = [] { const char* e = std::getenv("FORGE_MMQ_MIN_ROWS"); if (!e) return 16u; auto n = std::uint32_t(std::strtoul(e, nullptr, 10)); return n ? n : 0xffffffffu; }();
        return v;
    }
    bool has_fn_locked(const std::string& n) {   // caller holds mu_
        if (fn_.count(n)) return true;
        void* f = nullptr; if (f_.get_fn(&f, mod_, n.c_str()) != 0 || !f) return false;
        fn_[n] = f; return true;
    }
    // Embedding gather over an encoded table (forge_qgather); the table is uploaded once, like run_qmatmul's weights.
    bool run_qgather(const tensor_view& idx, const std::uint8_t* q, std::size_t rb, std::uint32_t nrows, std::uint32_t d, std::uint32_t fmt, tensor_view out) {
        if (!q || fmt < 1 || fmt > 4 || !d || d % 32 || out.size() != idx.size() * d) return false;
        std::lock_guard lk(mu_); bind();
        if (!has_fn_locked("forge_qgather")) return false;
        ++dispatches_["qgather"];
        std::size_t qbytes = rb * nrows;
        auto& qd = qres_[{reinterpret_cast<std::uintptr_t>(q), qbytes}];
        if (!qd) {
            std::uint64_t dp = 0; ck(f_.malloc_(&dp, std::max<std::size_t>(4, qbytes)), "malloc");
            qd = dp; ck(htod_s(qd, q, qbytes, stream_), "memcpy HtoD"); ++stats_.uploads; stats_.qweight_bytes += qbytes;
        }
        std::vector<std::uint64_t> temps;
        auto free_temps = [&] { for (auto p : temps) f_.free_(p); };
        std::uint32_t n = std::uint32_t(idx.size());
        try {
            std::uint64_t xd = 0, yd = 0;
            if (in_run_) {
                xd = acquire(idx.data, idx.size(), false).ptr;
                auto b = acquire(out.data, out.size(), true); b.e->host_valid = false; yd = b.ptr;
            } else {
                ck(f_.malloc_(&xd, std::max<std::size_t>(4, idx.size() * 4)), "malloc"); temps.push_back(xd);
                ck(htod_s(xd, idx.data, idx.size() * 4, stream_), "memcpy HtoD");
                ck(f_.malloc_(&yd, std::max<std::size_t>(4, out.size() * 4)), "malloc"); temps.push_back(yd);
            }
            std::uint64_t wd = qd; std::uint32_t rb32 = std::uint32_t(rb);
            void* args[] = {&xd, &wd, &yd, &n, &d, &fmt, &rb32, &nrows};
            if (n) ck(launch_s(function("forge_qgather"), n, 1, 1, 256, 1, 1, 0, stream_, args, nullptr), "launch");
            if (!in_run_) { ck(dtoh_s(out.data, yd, out.size() * 4, stream_), "memcpy DtoH"); sync(); }
        } catch (...) { if (!in_run_) sync(); free_temps(); throw; }
        if (!temps.empty()) { sync(); free_temps(); }
        return true;
    }
    static bool use_mmvq() { static const bool on = [] { auto* e = std::getenv("FORGE_MMVQ"); return !(e && std::string(e) == "0"); }(); return on; }
    std::uint64_t q81_ = 0; std::size_t q81_cap_ = 0;   // q8_1 activation scratch (stream-ordered reuse)
    // The scratch holds the q8_1 form of device buffer q81_src_ ([q81_n_] blocks). Consecutive decode matmuls that
    // read the same activation (qkv/z/a/b, gate/up) reuse it instead of re-quantising; any other kernel, copy or free
    // may change that buffer, so launch_s/htod_s/dtod/free_s drop it (q81_keep_ marks run_qmatmul's own launches).
    std::uint64_t q81_src_ = 0; std::uint32_t q81_n_ = 0; bool q81_keep_ = false;
    void q81_drop() { q81_src_ = 0; }
    // Device ranges the next launch writes (set by callers that know them). launch_s keeps the q8_1 scratch when
    // none of them overlaps its source, so e.g. the a/b/z matmuls of a DeltaNet layer reuse the quantised input of
    // in_proj_qkv across the conv/slice kernels in between; a launch with no declared writes drops it.
    std::uint64_t wr_[3][2] = {}; unsigned nwr_ = 0;
    void writes(std::uint64_t p, std::size_t bytes) { if (nwr_ < 3) { wr_[nwr_][0] = p; wr_[nwr_][1] = p + bytes; } ++nwr_; }
    void q81_reserve(std::size_t need) {
        if (q81_cap_ >= need) return;
        q81_drop(); if (q81_) free_s(q81_); q81_ = 0;
        ck(f_.malloc_(&q81_, std::max<std::size_t>(need, 1 << 20)), "malloc"); q81_cap_ = std::max<std::size_t>(need, 1 << 20);
    }
    // Decode rms_norm / silu_mul also emit the q8_1 form of their output for the next matmul ($FORGE_FUSE_Q81=0 off).
    static bool fuse_q81() { static const bool on = [] { auto* e = std::getenv("FORGE_FUSE_Q81"); return !(e && std::string(e) == "0"); }(); return on; }
    bool run_qmatmul(const tensor_view& x, const std::uint8_t* q, std::size_t rb, std::uint32_t cols, std::uint32_t fmt, tensor_view out) {
        if (x.shape.empty() || !q || fmt < 1 || fmt > 4) return false;
        std::uint32_t k = std::uint32_t(x.shape.back()), rows = k ? std::uint32_t(x.size() / k) : 0;
        if (!k || k % 32 || out.size() != std::size_t(rows) * cols) return false;
        std::lock_guard lk(mu_); bind();
        ++dispatches_["qmatmul_t"];
        void* fn = function("forge_qmatmul_t");
        std::size_t qbytes = rb * cols;
        auto& qd = qres_[{reinterpret_cast<std::uintptr_t>(q), qbytes}];
        if (!qd) {
            std::uint64_t d = 0; ck(f_.malloc_(&d, std::max<std::size_t>(4, qbytes)), "malloc");
            qd = d; ck(htod_s(qd, q, qbytes, stream_), "memcpy HtoD"); ++stats_.uploads; stats_.qweight_bytes += qbytes;
        }
        std::vector<std::uint64_t> temps;
        auto free_temps = [&] { for (auto p : temps) f_.free_(p); };
        try {
            std::uint64_t xd = 0, yd = 0;
            if (in_run_) {
                xd = acquire(x.data, x.size(), false).ptr;
                auto b = acquire(out.data, out.size(), true); b.e->host_valid = false; yd = b.ptr;
            } else {
                ck(f_.malloc_(&xd, std::max<std::size_t>(4, x.size() * 4)), "malloc"); temps.push_back(xd);
                ck(htod_s(xd, x.data, x.size() * 4, stream_), "memcpy HtoD");
                ck(f_.malloc_(&yd, std::max<std::size_t>(4, out.size() * 4)), "malloc"); temps.push_back(yd);
            }
            std::uint64_t wd = qd; std::uint32_t rb32 = std::uint32_t(rb);
            void* args[] = {&xd, &wd, &yd, &rows, &cols, &k, &fmt, &rb32};
            if (rows >= mmq_min_rows() && has_fn_locked("forge_qmatmul_mmq")) {   // batched prefill: 64x64 tiles
                ++dispatches_["qmatmul_mmq"];
                unsigned grid = ((cols + 63) / 64) * ((rows + 63) / 64);
                if (grid) ck(launch_s(function("forge_qmatmul_mmq"), grid, 1, 1, 256, 1, 1, 0, stream_, args, nullptr), "launch");
            } else if (fmt <= 3 && use_mmvq() && has_fn_locked("forge_qmatmul_q81") && has_fn_locked("forge_quant_q81")) {
                ++dispatches_["qmatmul_q81"];   // decode: q8_1 activations + dp4a (llama.cpp mmvq)
                std::uint32_t nb = rows * (k / 32); std::size_t need = std::size_t(nb) * 40;
                q81_reserve(need);
                std::uint64_t qx = q81_;
                q81_keep_ = true;
                if (q81_src_ != xd || q81_n_ != nb) {
                    void* qa[] = {&xd, &qx, &nb};
                    ck(launch_s(function("forge_quant_q81"), (nb + 7) / 8, 1, 1, 256, 1, 1, 0, stream_, qa, nullptr), "launch");
                    q81_src_ = xd; q81_n_ = nb;
                } else ++dispatches_["quant_q81_reused"];
                void* a2[] = {&qx, &wd, &yd, &rows, &cols, &k, &fmt, &rb32};
                unsigned grid = ((cols + 3) / 4) * ((rows + 3) / 4);   // a warp per column, 4 columns per block
                if (grid) ck(launch_s(function("forge_qmatmul_q81"), grid, 1, 1, 128, 1, 1, 0, stream_, a2, nullptr), "launch");
                q81_keep_ = false;
                if (yd < xd + std::uint64_t(nb) * 128 && xd < yd + std::uint64_t(out.size()) * 4) q81_drop();   // y overwrote the source
            } else {
                unsigned grid = cols * ((rows + 3) / 4);
                if (grid) ck(launch_s(fn, grid, 1, 1, 128, 1, 1, 0, stream_, args, nullptr), "launch");
            }
            if (!in_run_) { ck(dtoh_s(out.data, yd, out.size() * 4, stream_), "memcpy DtoH"); sync(); }
        } catch (...) { q81_keep_ = false; q81_drop(); if (!in_run_) sync(); free_temps(); throw; }
        if (!temps.empty()) { sync(); free_temps(); }
        return true;
    }

    // Gated DeltaNet state ops (flow/state_ops.hpp) on the device. The state slot S is pinned, so it stays resident
    // across runs (tokens) like the weights; inputs/output follow the residency protocol. Only inside a run.
    bool run_state(const std::string& fn, const std::vector<tensor_view>& in, float* S, std::size_t sn, tensor_view out, const spec::attrs& a) {
        std::lock_guard lk(mu_); bind();
        if (!in_run_ || !has_fn_locked("forge_" + fn)) return false;
        std::uint32_t u[6] = {}; std::uint32_t grid = 0; float sc = 0;
        if (fn == "delta_rule" && in.size() == 5) {
            u[0] = std::uint32_t(in[0].shape[0]); u[1] = std::uint32_t(in[0].shape[1]); u[2] = std::uint32_t(in[0].shape[2]);
            u[3] = std::uint32_t(in[2].shape[1]); u[4] = std::uint32_t(in[2].shape[2]); u[5] = a.num("l2norm", 1) != 0;
            if (u[2] > 256u || !u[1] || u[3] % u[1]) return false;
            sc = float(a.num("scale", 1.0 / std::sqrt(double(u[2])))); grid = u[3];
        } else if (fn == "causal_conv" && in.size() == 2) {
            u[0] = std::uint32_t(in[0].shape[0]); u[1] = std::uint32_t(in[0].shape[1]); u[2] = std::uint32_t(in[1].shape[1]);
            if (u[2] < 1u || u[2] > 9u) return false;
            grid = (u[1] + 127u) / 128u;
        } else return false;
        bool known = false; for (auto& [p, n] : pinned_) if (p == S && n == sn) known = true;
        if (!known) pinned_.push_back({S, sn});
        // the kernels are not in-place safe: an output overlapping an input or the state goes to the host path
        auto overlaps = [](const float* a, std::size_t an, const float* b, std::size_t bn) { return a < b + bn && b < a + an; };
        for (auto& v : in) if (overlaps(out.data, out.size(), v.data, v.size())) return false;
        if (overlaps(out.data, out.size(), S, sn)) return false;
        // acquire twice: a later acquire may replace (and free) a resident buffer an earlier one returned when the
        // ranges overlap, so the pointers are only final once a full pass hits the cache
        std::vector<std::uint64_t> ptrs;
        for (int pass = 0; pass < 2; ++pass) {
            ptrs.clear();
            for (auto& v : in) ptrs.push_back(acquire(v.data, v.size(), false).ptr);
            auto sb = acquire(S, sn, false); sb.e->pinned = true; ptrs.push_back(sb.ptr);
            ptrs.push_back(acquire(out.data, out.size(), pass == 0).ptr);
        }
        for (auto& [k, e] : res_) { auto kp = reinterpret_cast<const float*>(k); if (overlaps(kp, e.count, S, sn) || overlaps(kp, e.count, out.data, out.size())) e.host_valid = false; }
        std::vector<void*> args; for (auto& p : ptrs) args.push_back(&p);
        if (fn == "delta_rule") { for (int i = 0; i < 6; ++i) args.push_back(&u[i]); args.push_back(&sc); }
        else for (int i = 0; i < 3; ++i) args.push_back(&u[i]);
        ++dispatches_[fn];
        writes(ptrs.back(), out.size() * 4); writes(ptrs[ptrs.size() - 2], sn * 4);
        if (grid) ck(launch_s(function("forge_" + fn), grid, 1, 1, 128, 1, 1, 0, stream_, args.data(), nullptr), "launch");
        return true;
    }

    // Run one op over host views; false if there is no kernel for this op/shape. Inside a run (begin_run)
    // buffers stay resident; outside, every buffer is uploaded and the output downloaded.
    bool run_op(const std::string& op, const std::vector<tensor_view>& in, const spec::attrs& a, tensor_view out) {
        std::vector<shape_t> ss; for (auto& v : in) ss.push_back(v.shape);
        if (!supports(op, ss, a)) return false;
        sig s; kernel_sig(op, s);
        auto& x = in[0];
        std::uint32_t n = std::uint32_t(out.size()), rows = 0, cols = 0, k = 0, threads = n, u4 = 0, u5 = 0, u6 = 0, u7 = 0;
        float f0 = 0;
        std::vector<float> extra;   // rope: inverse frequencies (computed in double on the host, like the CPU path)
        switch (s) {
        case sig::unary: case sig::binary: break;
        case sig::scale: f0 = float(a.num("alpha", 1.0)); break;
        case sig::rows: case sig::rms:
            cols = std::uint32_t(x.shape.back()); rows = cols ? std::uint32_t(x.size()) / cols : 0; threads = rows; f0 = float(a.num("eps", 1e-6)); break;
        case sig::matmul:
            k = std::uint32_t(x.shape.back()); rows = std::uint32_t(x.size()) / k; cols = std::uint32_t(in[1].shape[0]); threads = rows * cols; break;
        case sig::rope: {
            rows = std::uint32_t(x.shape[0]); cols = std::uint32_t(x.shape[1]); k = std::uint32_t(x.shape[2]); u4 = a.num("neox", 1) != 0;
            double base = a.num("base", 10000.0); extra.resize(k / 2);
            for (std::uint32_t j = 0; j < k / 2; ++j) extra[j] = float(std::pow(base, -2.0 * double(j) / double(k)));
            threads = rows * cols * (k / 2); break; }
        case sig::gather: rows = std::uint32_t(x.shape[0]); cols = std::uint32_t(x.shape[1]); break;
        case sig::slice: {
            auto ax = std::size_t(spec::norm_axis(i64(a.num("axis", 0)), x.shape.size()));
            i64 inner = 1; for (std::size_t d = ax + 1; d < x.shape.size(); ++d) inner *= x.shape[d];
            i64 st = i64(a.num("start", 0)); if (st < 0) st += x.shape[ax];
            rows = std::uint32_t(x.shape[ax]); cols = std::uint32_t(inner); k = std::uint32_t(st); u4 = std::uint32_t(out.shape[ax]); break; }
        case sig::attention: {
            auto kc = std::uint32_t(a.num("kv_codec_k", 0)), vc = std::uint32_t(a.num("kv_codec_v", 0));
        if (std::uint32_t dd = std::uint32_t(x.shape[2]), dvv = std::uint32_t(a.num("kv_dv", 0)); (kc >= 3u && dd > 256u) || (vc >= 3u && dvv > 256u)) raise(errc::unsupported, "TurboQuant KV on the GPU supports head dims up to 256");
        if (kc >= 5u || vc >= 5u) raise(errc::unsupported, "TurboQuant QJL KV types (tbq4p, tbq3p) run on the CPU only");
            rows = std::uint32_t(x.shape[0]); cols = std::uint32_t(x.shape[1]); k = std::uint32_t(x.shape[2]);
            u4 = std::uint32_t(in[1].shape[0]); u5 = std::uint32_t(in[1].shape[1]);
            u6 = vc ? std::uint32_t(a.num("kv_dv", 0)) : std::uint32_t(in[2].shape[2]);
            u7 = (a.num("causal", 1) != 0 ? 1u : 0u) | (kc << 1) | (vc << 4) | (std::uint32_t(a.num("window", 0)) << 7);
            f0 = float(a.num("scale", 1.0 / std::sqrt(double(k)))); threads = rows * cols; break; }
        }
        std::lock_guard lk(mu_); bind();
        ++dispatches_[op];
        void* fn = function("forge_" + op);
        std::vector<std::uint64_t> ptrs, temps;
        auto free_temps = [&] { for (auto p : temps) f_.free_(p); };
        try {
            auto buf = [&](const float* h, std::size_t cnt, bool is_out) {
                if (in_run_) { auto b = acquire(const_cast<float*>(h), cnt, is_out); if (is_out) b.e->host_valid = false; return b.ptr; }
                std::uint64_t d = 0; ck(f_.malloc_(&d, std::max<std::size_t>(4, cnt * 4)), "malloc"); temps.push_back(d);
                if (!is_out) ck(htod_s(d, h, cnt * 4, stream_), "memcpy HtoD");
                return d;
            };
            for (std::size_t i = 0; i < in.size(); ++i) {
                ptrs.push_back(buf(in[i].data, in[i].size(), false));
                if (s == sig::rope && i == 1) {   // the inverse-frequency table goes after pos
                    auto& d = rope_[extra];   // the table depends only on (base, dim): upload once
                    if (!d) { ck(f_.malloc_(&d, std::max<std::size_t>(4, extra.size() * 4)), "malloc");
                              ck(htod_s(d, extra.data(), extra.size() * 4, stream_), "memcpy HtoD"); }
                    ptrs.push_back(d);
                }
            }
            ptrs.push_back(buf(out.data, out.size(), true));
            std::vector<void*> args; for (auto& p : ptrs) args.push_back(&p);
            switch (s) {
            case sig::unary: case sig::binary: args.push_back(&n); break;
            case sig::scale: args.push_back(&n); args.push_back(&f0); break;
            case sig::rows: args.push_back(&rows); args.push_back(&cols); break;
            case sig::rms: args.push_back(&rows); args.push_back(&cols); args.push_back(&f0); break;
            case sig::matmul: args.push_back(&rows); args.push_back(&cols); args.push_back(&k); break;
            case sig::rope: args.push_back(&rows); args.push_back(&cols); args.push_back(&k); args.push_back(&u4); break;
            case sig::gather: args.push_back(&n); args.push_back(&rows); args.push_back(&cols); break;
            case sig::slice: args.push_back(&n); args.push_back(&rows); args.push_back(&cols); args.push_back(&k); args.push_back(&u4); break;
            case sig::attention:
                for (auto* p : {&rows, &cols, &k, &u4, &u5, &u6, &u7}) args.push_back(p);
                args.push_back(&f0); break;
            }
            unsigned block = 64, grid = (threads + block - 1) / block;
            if (s == sig::attention && !std::getenv("FORGE_GPU_NO_FA") && has_fn_locked("forge_attention_fa")) {   // block per (row, head)
                fn = function("forge_attention_fa"); block = 128; grid = rows * cols; }
            if (s == sig::rms && has_fn_locked("forge_rms_norm_b")) { fn = function("forge_rms_norm_b"); block = 256; grid = rows; }
            std::uint32_t fnb = 0; std::uint64_t qx = 0;
            if (in_run_ && fuse_q81() && use_mmvq() && rows < mmq_min_rows()) {
                if (s == sig::rms && cols % 32 == 0 && has_fn_locked("forge_rms_norm_q81")) {
                    fn = function("forge_rms_norm_q81"); fnb = rows * (cols / 32); }
                else if (op == "silu_mul" && n % 32 == 0 && n <= (1u << 18) && has_fn_locked("forge_silu_mul_q81")) {
                    fn = function("forge_silu_mul_q81"); fnb = n / 32; block = 256; grid = (fnb + 7) / 8; }
                if (fnb) { q81_reserve(std::size_t(fnb) * 40); qx = q81_; args.push_back(&qx); ++dispatches_["fused_q81"]; }
            }
            writes(ptrs.back(), out.size() * 4);
            if (grid) ck(launch_s(fn, grid, 1, 1, block, 1, 1, 0, stream_, args.data(), nullptr), "launch");
            if (fnb) { q81_src_ = ptrs.back(); q81_n_ = fnb; }
            if (!in_run_) { ck(dtoh_s(out.data, ptrs.back(), out.size() * 4, stream_), "memcpy DtoH"); sync(); }
        } catch (...) { if (!in_run_) sync(); free_temps(); throw; }
        if (!temps.empty()) { sync(); free_temps(); }
        return true;
    }

private:
    explicit runtime(api a) : api_(a) {}
    // Driver entry points, normalised: CUDA and HIP module APIs have the same shapes. Optional ones may be null.
    struct fns {
        int (*init)(unsigned) = nullptr;
        int (*count)(int*) = nullptr;
        int (*name)(char*, int, int) = nullptr;
        int (*device_get)(int*, int) = nullptr;               // CUDA
        int (*ctx_create)(void**, unsigned, int) = nullptr;   // CUDA
        int (*ctx_destroy)(void*) = nullptr;
        int (*ctx_set)(void*) = nullptr;                      // CUDA
        int (*set_device)(int) = nullptr;                      // HIP
        int (*load)(void**, const void*) = nullptr;
        int (*unload)(void*) = nullptr;
        int (*get_fn)(void**, void*, const char*) = nullptr;
        int (*malloc_)(std::uint64_t*, std::size_t) = nullptr;
        int (*free_)(std::uint64_t) = nullptr;
        int (*htod)(std::uint64_t, const void*, std::size_t) = nullptr;
        int (*dtoh)(void*, std::uint64_t, std::size_t) = nullptr;
        int (*htod_a)(std::uint64_t, const void*, std::size_t, void*) = nullptr;
        int (*dtoh_a)(void*, std::uint64_t, std::size_t, void*) = nullptr;
        int (*dtod_a)(std::uint64_t, std::uint64_t, std::size_t, void*) = nullptr;
        int (*launch)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**) = nullptr;
        int (*sync_)() = nullptr;
        int (*stream_create)(void**, unsigned) = nullptr;
        int (*stream_destroy)(void*) = nullptr;
        int (*stream_sync)(void*) = nullptr;
        int (*capture_begin)(void*, int) = nullptr;
        int (*capture_end)(void*, void**) = nullptr;
        int (*graph_inst)(void**, void*, unsigned long long) = nullptr;
        int (*graph_launch)(void*, void*) = nullptr;
        int (*gexec_destroy)(void*) = nullptr;
        int (*graph_destroy)(void*) = nullptr;
        int (*exec_update)(void*, void*, void*) = nullptr;   // cuGraphExecUpdate_v2 (CUDA only)
        int (*param_info)(void*, std::size_t, std::size_t*, std::size_t*) = nullptr;   // cuFuncGetParamInfo (CUDA 12.4+)
        int (*kn_set)(void*, void*, const void*) = nullptr;                             // cuGraphExecKernelNodeSetParams
        int (*g_roots)(void*, void**, std::size_t*) = nullptr;                          // cuGraphGetRootNodes
        int (*g_deps)(void*, void**, std::size_t*) = nullptr;                           // cuGraphNodeGetDependentNodes
        int (*g_type)(void*, int*) = nullptr;                                           // cuGraphNodeGetType
        int (*peer)(std::uint64_t, std::uintptr_t, std::uint64_t, std::uintptr_t, std::size_t) = nullptr;
        int (*mem_info)(std::size_t*, std::size_t*) = nullptr;
        // wrappers that fall back to the synchronous copies when the async ones are missing
        int htod_async(std::uint64_t d, const void* h, std::size_t n, void* s) const { return htod_a ? htod_a(d, h, n, s) : htod(d, h, n); }
        int dtoh_async(void* h, std::uint64_t d, std::size_t n, void* s) const { return dtoh_a ? dtoh_a(h, d, n, s) : dtoh(h, d, n); }
        int dtod_async(std::uint64_t d, std::uint64_t s, std::size_t n, void* st) const {
            if (dtod_a) return dtod_a(d, s, n, st);
            std::vector<char> t(n); int r = dtoh(t.data(), s, n); return r ? r : htod(d, t.data(), n);
        }
    };
    struct entry { float* host; std::size_t count; std::uint64_t d; bool host_valid = true, pinned = false, kept = false, arena = false; };
    struct bound { entry* e; std::uint64_t ptr; };

    template<class F> bool sym(F& f, const char* n) { f = reinterpret_cast<F>(dlsym(lib_, n)); return f != nullptr; }
    template<class F> void opt(F& f, const char* n) { f = reinterpret_cast<F>(dlsym(lib_, n)); }
    void ck(int r, const char* what) {
        if (r != 0) throw std::runtime_error(std::string(api_name(api_)) + ": " + what + " failed (" + std::to_string(r) + ")");
    }
    // ---- decode graphs (FORGE_GRAPHS=1, CUDA): inside a run, kernel launches are stream-captured; a host copy, a
    // free or a sync ends the segment, which is replayed through an exec graph cached per segment index and refreshed
    // with cuGraphExecUpdate (same topology every token, only pointers/scalars change), as llama.cpp's ggml-cuda does.
    bool gmode() const {
        static const bool on = [] { auto* e = std::getenv("FORGE_GRAPHS"); return e && std::string(e) == "1"; }();
        return on && api_ == api::cuda && stream_ && has_graphs();
    }
    bool capturing_ = false, user_cap_ = false; std::size_t seg_ = 0;
    // A segment's kernels (function, launch shape, argument bytes). Once a segment has been captured and instantiated,
    // later tokens replay it without capturing: launches are only recorded, and if the sequence of functions and
    // shapes matches, flush() patches the kernel nodes whose arguments changed (cuGraphExecKernelNodeSetParams) and
    // relaunches the exec graph -- no capture, no cuGraphExecUpdate. A mismatch falls back to capturing the recorded
    // launches. Needs cuFuncGetParamInfo (CUDA 12.4+); without it every segment is captured and exec-updated.
    struct krec { void* fn; unsigned d[7]; std::uint32_t off, len; };
    struct segc { void* exec = nullptr; void* graph = nullptr; std::vector<void*> nodes; std::vector<krec> recs; std::vector<unsigned char> bytes; bool replay = false; };
    std::vector<segc> segc_;
    std::vector<krec> prec_; std::vector<unsigned char> pbytes_;
    bool recording_ = false, deferring_ = false;
    std::unordered_map<void*, std::vector<std::uint32_t>> psize_;   // per function: parameter sizes
    bool replay_ok() const {
        static const bool on = [] { auto* e = std::getenv("FORGE_GRAPH_REPLAY"); return !(e && std::string(e) == "0"); }();
        return on && f_.param_info && f_.kn_set && f_.g_roots && f_.g_deps && f_.g_type;
    }
    const std::vector<std::uint32_t>* params_of(void* fn) {
        auto it = psize_.find(fn); if (it != psize_.end()) return &it->second;
        std::vector<std::uint32_t> v;
        for (std::size_t i = 0; i < 64; ++i) { std::size_t o = 0, z = 0; if (f_.param_info(fn, i, &o, &z) != 0) break; v.push_back(std::uint32_t(z)); }
        return &(psize_[fn] = std::move(v));
    }
    void record(void* fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned sh, void** a) {
        krec r{fn, {gx, gy, gz, bx, by, bz, sh}, std::uint32_t(pbytes_.size()), 0};
        if (replay_ok()) for (auto z : *params_of(fn)) {
            std::size_t at = (pbytes_.size() + 7) & ~std::size_t(7); pbytes_.resize(at + z);
            std::memcpy(pbytes_.data() + at, *a++, z);
        }
        r.len = std::uint32_t(pbytes_.size() - r.off); prec_.push_back(r);
    }
    // the kernelParams array of record r, pointing into bytes
    void arg_ptrs(const krec& r, unsigned char* bytes, std::vector<void*>& out) {
        out.clear(); std::size_t at = r.off;
        for (auto z : *params_of(r.fn)) { at = (at + 7) & ~std::size_t(7); out.push_back(bytes + at); at += z; }
    }
    void reissue_captured() {   // begin capturing and launch every recorded kernel of the segment
        ck(f_.capture_begin(stream_, 2), "stream begin capture"); capturing_ = true; deferring_ = false;
        std::vector<void*> ap;
        for (auto& r : prec_) { arg_ptrs(r, pbytes_.data(), ap); ck(f_.launch(r.fn, r.d[0], r.d[1], r.d[2], r.d[3], r.d[4], r.d[5], r.d[6], stream_, ap.data(), nullptr), "launch"); }
    }
    int launch_s(void* fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned sh, void* st, void** a, void** ex) {
        if (!q81_keep_ && q81_src_) {
            bool hit = nwr_ == 0 || nwr_ > 3; const std::uint64_t lo = q81_src_, hi = lo + std::uint64_t(q81_n_) * 128;
            for (unsigned i = 0; i < nwr_ && i < 3; ++i) if (wr_[i][0] < hi && lo < wr_[i][1]) hit = true;
            if (hit) q81_drop();
        }
        nwr_ = 0;
        if (!(in_run_ && !user_cap_ && gmode() && st == stream_ && !ex)) {
            count(user_cap_ ? counters().captured : counters().launches);
            return f_.launch(fn, gx, gy, gz, bx, by, bz, sh, st, a, ex);
        }
        if (!recording_) {
            recording_ = true; prec_.clear(); pbytes_.clear();
            if (seg_ >= segc_.size()) segc_.resize(seg_ + 1);
            auto& c = segc_[seg_];
            deferring_ = replay_ok() && c.replay && c.exec;
            if (!deferring_) { ck(f_.capture_begin(stream_, 2), "stream begin capture"); capturing_ = true; }
        }
        record(fn, gx, gy, gz, bx, by, bz, sh, a);
        count(counters().captured);
        int r = 0;
        if (deferring_) {
            auto& c = segc_[seg_]; std::size_t i = prec_.size() - 1;
            if (i >= c.recs.size() || c.recs[i].fn != fn || std::memcmp(c.recs[i].d, prec_[i].d, sizeof prec_[i].d) || c.recs[i].len != prec_[i].len)
                reissue_captured();   // the segment changed shape: capture it after all
        } else r = f_.launch(fn, gx, gy, gz, bx, by, bz, sh, st, a, ex);
        // Split long segments so the GPU runs one while the host records the next ($FORGE_GRAPH_SEG kernels, 0 = off).
        if (r == 0 && ++seg_kernels_ >= graph_seg()) flush();
        return r;
    }
    std::size_t seg_kernels_ = 0;
    static std::size_t graph_seg() {
        static const std::size_t v = [] { auto* e = std::getenv("FORGE_GRAPH_SEG"); std::size_t n = e ? std::strtoul(e, nullptr, 10) : 64; return n ? n : ~std::size_t(0); }();
        return v;
    }
    int htod_s(std::uint64_t d, const void* h, std::size_t n, void* st) { flush(); q81_drop(); gpu::count(counters().htod); return f_.htod_async(d, h, n, st); }
    int dtoh_s(void* h, std::uint64_t d, std::size_t n, void* st) { flush(); gpu::count(counters().dtoh); return f_.dtoh_async(h, d, n, st); }
    int free_s(std::uint64_t p) { flush(); q81_drop(); return f_.free_(p); }
    // device-to-device copy on the stream; inside a graph-mode run it is a kernel (forge_copy_words), so segments hold
    // kernel nodes only and replay can patch them
    int dtod_s(std::uint64_t d, std::uint64_t s, std::size_t n) {
        gpu::count(counters().dtod); q81_drop();
        if (in_run_ && gmode() && n % 4 == 0 && n / 4 < (1ull << 32) && has_fn_locked("forge_copy_words")) {
            std::uint32_t w = std::uint32_t(n / 4); void* args[] = {&s, &d, &w}; writes(d, n);
            return w ? launch_s(function("forge_copy_words"), (w + 255) / 256, 1, 1, 256, 1, 1, 0, stream_, args, nullptr) : 0;
        }
        if (deferring_) reissue_captured();   // a memcpy node: this segment is captured (and not replayed)
        return f_.dtod_async(d, s, n, stream_);
    }
    void drop_seg(segc& c) {
        if (c.exec && f_.gexec_destroy) f_.gexec_destroy(c.exec);
        if (c.graph && f_.graph_destroy) f_.graph_destroy(c.graph);
        c = segc{};
    }
    // the kernel nodes of a captured single-stream graph, in launch order (a chain from its one root)
    bool chain(void* g, std::vector<void*>& out) {
        out.clear(); std::size_t n = 1; void* node = nullptr;
        if (f_.g_roots(g, &node, &n) != 0 || n != 1) return false;
        while (node) {
            int t = -1; if (f_.g_type(node, &t) != 0 || t != 0 /* CU_GRAPH_NODE_TYPE_KERNEL */) return false;
            out.push_back(node);
            void* next = nullptr; std::size_t m = 0;
            if (f_.g_deps(node, nullptr, &m) != 0 || m > 1) return false;
            if (m == 1 && f_.g_deps(node, &next, &m) != 0) return false;
            node = m ? next : nullptr;
        }
        return true;
    }
    void flush() {
        if (!recording_) return;
        recording_ = false; seg_kernels_ = 0;
        auto& c = segc_[seg_++];
        if (deferring_) {
            deferring_ = false;
            bool ok = prec_.size() == c.recs.size();
            std::vector<void*> ap;
            for (std::size_t i = 0; ok && i < prec_.size(); ++i) {
                auto& r = prec_[i];
                if (!std::memcmp(pbytes_.data() + r.off, c.bytes.data() + c.recs[i].off, r.len)) continue;
                arg_ptrs(r, pbytes_.data(), ap);
                struct { void* fn; unsigned d[7]; void** kp; void** ex; } kp{r.fn, {r.d[0], r.d[1], r.d[2], r.d[3], r.d[4], r.d[5], r.d[6]}, ap.data(), nullptr};
                if (f_.kn_set(c.exec, c.nodes[i], &kp) != 0) ok = false; else gpu::count(counters().node_updates);
            }
            if (ok) {
                c.recs.swap(prec_); c.bytes.swap(pbytes_);
                gpu::count(counters().graph_launches); ck(f_.graph_launch(c.exec, stream_), "graph launch");
                return;
            }
            reissue_captured();   // fewer kernels than last time, or a patch failed (exec state is suspect: rebuild)
            drop_seg(c);
        }
        capturing_ = false;
        void* g = nullptr; ck(f_.capture_end(stream_, &g), "stream end capture");
        void* done = nullptr;   // destroyed after the launch (a driver may share storage between graph and exec)
        if (replay_ok()) {   // keep the graph: replay patches its nodes
            drop_seg(c);
            ck(f_.graph_inst(&c.exec, g, 0), "graph instantiate"); gpu::count(counters().instantiates);
            c.graph = g; c.replay = chain(g, c.nodes) && c.nodes.size() == prec_.size();
            c.recs.swap(prec_); c.bytes.swap(pbytes_);
        } else {
            alignas(8) char info[64] = {};
            if (c.exec && !(f_.exec_update && f_.exec_update(c.exec, g, info) == 0)) { f_.gexec_destroy(c.exec); c.exec = nullptr; }
            if (!c.exec) { ck(f_.graph_inst(&c.exec, g, 0), "graph instantiate"); gpu::count(counters().instantiates); }
            done = g;
        }
        gpu::count(counters().graph_launches); ck(f_.graph_launch(c.exec, stream_), "graph launch");
        if (done && f_.graph_destroy) f_.graph_destroy(done);
    }
    void need_graphs() { if (!has_graphs()) throw std::runtime_error(std::string(api_name(api_)) + ": driver has no graph API"); }
    // The context made current on this thread by the last bind() (any runtime): cuCtxSetCurrent is skipped when it is
    // already ours, instead of once per op.
    // already ours, instead of once per op. Destroying any context bumps the epoch, which invalidates every cache.
    struct bound_t { void* c = nullptr; unsigned long e = 0; };
    static bound_t& bound_ctx() { thread_local bound_t b; return b; }
    static std::atomic<unsigned long>& ctx_epoch() { static std::atomic<unsigned long> e{1}; return e; }
    void bind() {
        if (api_ == api::cuda) {
            auto& b = bound_ctx(); const unsigned long e = ctx_epoch().load(std::memory_order_acquire);
            if (f_.ctx_set && ctx_ && (b.c != ctx_ || b.e != e)) { f_.ctx_set(ctx_); b.c = ctx_; b.e = e; }
        }
        else if (f_.set_device) f_.set_device(ordinal_);
    }
    void sync() { flush(); gpu::count(counters().syncs); if (stream_ && f_.stream_sync) ck(f_.stream_sync(stream_), "stream synchronize"); else ck(f_.sync_(), "synchronize"); }
    void* function(const std::string& n) {
        auto it = fn_.find(n); if (it != fn_.end()) return it->second;
        void* f = nullptr; ck(f_.get_fn(&f, mod_, n.c_str()), ("get function " + n).c_str());
        return fn_[n] = f;
    }
    template<class F> void each_overlap(const float* p, std::size_t n, F&& f) {
        auto lo = reinterpret_cast<std::uintptr_t>(p), hi = lo + std::max<std::size_t>(n, 1) * 4;
        for (auto& [k, e] : res_) { auto el = k, eh = k + e.count * 4; if (el < hi && lo < eh) f(e); }
    }
    void download(entry& e) { ck(dtoh_s(e.host, e.d, e.count * 4, stream_), "memcpy DtoH"); e.host_valid = true; ++stats_.downloads; }
    // freed device buffers are pooled by size and reused: decode runs the same shapes every token
    void drop(entry& e) {
        stats_.resident_bytes -= e.count * 4;
        if (pool_bytes_ + e.count * 4 > pool_cap()) { free_s(e.d); return; }
        pool_.emplace(std::max<std::size_t>(4, e.count * 4), e.d); pool_bytes_ += std::max<std::size_t>(4, e.count * 4);
    }
    void dev_alloc(std::uint64_t* d, std::size_t bytes) {
        bytes = std::max<std::size_t>(4, bytes);
        if (auto it = pool_.find(bytes); it != pool_.end()) { *d = it->second; pool_.erase(it); pool_bytes_ -= bytes; return; }
        if (f_.malloc_(d, bytes) != 0) {   // out of memory: give the pool back and retry
            for (auto& [k, p] : pool_) free_s(p); pool_.clear(); pool_bytes_ = 0;
            ck(f_.malloc_(d, bytes), "malloc");
        }
    }
    static std::size_t pool_cap() {
        static const std::size_t v = [] { const char* e = std::getenv("FORGE_GPU_POOL_MB"); return (e ? std::strtoull(e, nullptr, 10) : 4096ull) << 20; }();
        return v;
    }
    bound acquire(float* h, std::size_t count, bool is_out) {
        auto key = reinterpret_cast<std::uintptr_t>(h);
        if (auto it = res_.find(key); it != res_.end() && it->second.count == count) return {&it->second, it->second.d};
        auto it = res_.upper_bound(key);
        if (it != res_.begin()) {   // a sub-range of a resident buffer (e.g. the live prefix of a KV slot)
            auto& c = std::prev(it)->second; auto cl = std::prev(it)->first;
            if (cl + c.count * 4 >= key + count * 4 && count) return {&c, c.d + (key - cl)};
        }
        // Resident entries overlapping [h, h+count) (decode re-views the same host arenas with new extents every
        // token): the overlapping part moves device-to-device into the new buffer and any part sticking out stays
        // resident as its own entry, so nothing goes through the host and the stream never syncs. Pinned/kept
        // entries that stick out are fetched to the host as before.
        std::vector<std::uintptr_t> olds, fetched; bool dl = false;
        const auto lo = key, hi = key + std::max<std::size_t>(count, 1) * 4;
        each_overlap(h, count, [&](entry& o) {
            auto ol = reinterpret_cast<std::uintptr_t>(o.host), oh = ol + o.count * 4;
            if ((o.pinned || o.kept) && !(ol >= lo && oh <= hi)) { if (!o.host_valid) { download(o); dl = true; } fetched.push_back(ol); }
            else olds.push_back(ol); });
        if (dl) sync();
        for (auto k : fetched) { drop(res_[k]); res_.erase(k); }
        entry e{h, count, 0};
        dev_alloc(&e.d, count * 4);
        for (auto& [p, n] : pinned_) if (p == h && n == count) e.pinned = true;
        std::sort(olds.begin(), olds.end());
        if (!is_out) {   // upload only the gaps the resident pieces do not cover
            std::uintptr_t at = lo;
            auto up = [&](std::uintptr_t x, std::uintptr_t y) { if (y > x) { ck(htod_s(e.d + (x - lo), reinterpret_cast<const void*>(x), y - x, stream_), "memcpy HtoD"); ++stats_.uploads; } };
            for (auto k : olds) { up(at, std::max(k, lo)); at = std::max(at, std::min(hi, k + res_[k].count * 4)); }
            up(at, lo + count * 4);
        }
        std::vector<entry> rest;
        for (auto k : olds) {
            entry o = res_[k]; res_.erase(k);
            auto ol = k, oh = k + o.count * 4, a = std::max(ol, lo), z = std::min(oh, hi);
            if (!is_out && z > a) { ck(dtod_s(e.d + (a - lo), o.d + (a - ol), z - a), "memcpy DtoD"); }
            if (!o.host_valid) e.host_valid = false;
            if (o.kept) e.kept = true;
            if (!o.host_valid) for (auto [x, y] : {std::pair{ol, std::min(oh, lo)}, std::pair{std::max(ol, hi), oh}}) if (y > x) {
                entry r{reinterpret_cast<float*>(x), (y - x) / 4, 0}; r.host_valid = false;
                dev_alloc(&r.d, y - x); ck(dtod_s(r.d, o.d + (x - ol), y - x), "memcpy DtoD");
                stats_.resident_bytes += y - x; rest.push_back(r);
            }
            drop(o);
        }
        for (auto& r : rest) res_.emplace(reinterpret_cast<std::uintptr_t>(r.host), r);
        stats_.resident_bytes += count * 4; stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, stats_.resident_bytes);
        auto& r = res_.emplace(key, e).first->second; return {&r, r.d};
    }
    std::string load_lib() {
        const char* libs_cuda[] = {"libcuda.so.1", "libcuda.so", "nvcuda.dll", nullptr};
        const char* libs_hip[] = {"libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so", "amdhip64.dll", nullptr};
        for (auto p = api_ == api::cuda ? libs_cuda : libs_hip; *p; ++p) if ((lib_ = dlopen(*p, RTLD_NOW | RTLD_LOCAL))) break;
        if (!lib_) return api_ == api::cuda ? "libcuda not found (no NVIDIA driver)" : "libamdhip64 not found (no ROCm runtime)";
        bool ok;
        if (api_ == api::cuda) {
            ok = sym(f_.init, "cuInit") && sym(f_.count, "cuDeviceGetCount") && sym(f_.name, "cuDeviceGetName") &&
                 sym(f_.ctx_create, "cuCtxCreate_v2") && sym(f_.ctx_destroy, "cuCtxDestroy_v2") && sym(f_.load, "cuModuleLoadData") &&
                 sym(f_.unload, "cuModuleUnload") && sym(f_.get_fn, "cuModuleGetFunction") && sym(f_.malloc_, "cuMemAlloc_v2") &&
                 sym(f_.free_, "cuMemFree_v2") && sym(f_.htod, "cuMemcpyHtoD_v2") && sym(f_.dtoh, "cuMemcpyDtoH_v2") &&
                 sym(f_.launch, "cuLaunchKernel") && sym(f_.sync_, "cuCtxSynchronize");
            opt(f_.device_get, "cuDeviceGet"); opt(f_.ctx_set, "cuCtxSetCurrent");
            opt(f_.htod_a, "cuMemcpyHtoDAsync_v2"); opt(f_.dtoh_a, "cuMemcpyDtoHAsync_v2"); opt(f_.dtod_a, "cuMemcpyDtoDAsync_v2");
            opt(f_.stream_create, "cuStreamCreate"); opt(f_.stream_destroy, "cuStreamDestroy_v2"); opt(f_.stream_sync, "cuStreamSynchronize");
            opt(f_.capture_begin, "cuStreamBeginCapture_v2"); opt(f_.capture_end, "cuStreamEndCapture");
            opt(f_.graph_inst, "cuGraphInstantiateWithFlags"); opt(f_.exec_update, "cuGraphExecUpdate_v2");
            opt(f_.param_info, "cuFuncGetParamInfo"); opt(f_.kn_set, "cuGraphExecKernelNodeSetParams");
            opt(f_.g_roots, "cuGraphGetRootNodes"); opt(f_.g_deps, "cuGraphNodeGetDependentNodes"); opt(f_.g_type, "cuGraphNodeGetType"); opt(f_.graph_launch, "cuGraphLaunch");
            opt(f_.gexec_destroy, "cuGraphExecDestroy"); opt(f_.graph_destroy, "cuGraphDestroy");
            opt(f_.peer, "cuMemcpyPeer"); opt(f_.mem_info, "cuMemGetInfo_v2");
        } else {
            ok = sym(f_.init, "hipInit") && sym(f_.count, "hipGetDeviceCount") && sym(f_.name, "hipDeviceGetName") &&
                 sym(f_.set_device, "hipSetDevice") && sym(f_.load, "hipModuleLoadData") && sym(f_.unload, "hipModuleUnload") &&
                 sym(f_.get_fn, "hipModuleGetFunction") && sym(f_.malloc_, "hipMalloc") && sym(f_.free_, "hipFree") &&
                 sym(f_.htod, "hipMemcpyHtoD") && sym(f_.dtoh, "hipMemcpyDtoH") && sym(f_.launch, "hipModuleLaunchKernel") &&
                 sym(f_.sync_, "hipDeviceSynchronize");
            opt(f_.htod_a, "hipMemcpyHtoDAsync"); opt(f_.dtoh_a, "hipMemcpyDtoHAsync"); opt(f_.dtod_a, "hipMemcpyDtoDAsync");
            opt(f_.stream_create, "hipStreamCreateWithFlags"); opt(f_.stream_destroy, "hipStreamDestroy"); opt(f_.stream_sync, "hipStreamSynchronize");
            opt(f_.capture_begin, "hipStreamBeginCapture"); opt(f_.capture_end, "hipStreamEndCapture");
            opt(f_.graph_inst, "hipGraphInstantiateWithFlags"); opt(f_.graph_launch, "hipGraphLaunch");
            opt(f_.gexec_destroy, "hipGraphExecDestroy"); opt(f_.graph_destroy, "hipGraphDestroy");
            opt(f_.peer, "hipMemcpyPeer"); opt(f_.mem_info, "hipMemGetInfo");
        }
        return ok ? std::string{} : "driver library is missing entry points";
    }
    std::string init() {
        auto imgs = module_images(api_);
        if (imgs.empty()) return "this build has no embedded kernel image (no clang with the GPU target at build time)";
        if (auto e = load_lib(); !e.empty()) return e;
        if (f_.init(0) != 0) return "driver init failed";
        int n = 0; if (f_.count(&n) != 0 || n <= 0) return "no devices";
        if (ordinal_ < 0 || ordinal_ >= n) return "device ordinal " + std::to_string(ordinal_) + " out of range (" + std::to_string(n) + " devices)";
        int dev = ordinal_;
        if (api_ == api::cuda && f_.device_get && f_.device_get(&dev, ordinal_) != 0) return "cuDeviceGet failed";
        char buf[256] = {}; f_.name(buf, sizeof buf, dev); name_ = buf;
        if (api_ == api::cuda) { if (f_.ctx_create(&ctx_, 0, dev) != 0) return "context creation failed"; }
        else if (f_.set_device(ordinal_) != 0) return "hipSetDevice failed";
        if (f_.stream_create && f_.stream_create(&stream_, 1 /* non-blocking */) != 0) stream_ = nullptr;
        for (auto& [arch, img] : imgs) {
            std::string image(img);
            if (f_.load(&mod_, image.data()) == 0) { arch_ = arch; return {}; }
            mod_ = nullptr;
        }
        return "module load failed (no embedded PTX/code object is accepted by this device)";
    }
    api api_;
    int ordinal_ = 0;
    void* lib_ = nullptr; void* ctx_ = nullptr; void* mod_ = nullptr; void* stream_ = nullptr;
    fns f_;
    std::string name_, arch_;
    std::mutex mu_;
    bool in_run_ = false;
    std::map<std::uintptr_t, entry> res_;
    std::map<std::pair<std::uintptr_t, std::size_t>, std::uint64_t> qres_;
    std::multimap<std::size_t, std::uint64_t> pool_; std::size_t pool_bytes_ = 0;
    std::map<std::vector<float>, std::uint64_t> rope_;   // encoded weights (run_qmatmul)
    std::vector<std::pair<const float*, std::size_t>> pinned_;
    std::vector<std::pair<void*, void*>> graphs_;
    std::map<std::string, void*> fn_;
    std::map<std::string, std::size_t> dispatches_;
    residency_stats stats_;
};

} // namespace forge::grid::gpu
