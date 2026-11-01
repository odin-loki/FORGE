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
            for (auto& [k, d] : pool_) f_.free_(d); pool_.clear(); for (auto& [k, d] : rope_) f_.free_(d); rope_.clear(); } catch (...) {}
        for (auto& [e, g] : graphs_) { if (f_.gexec_destroy) f_.gexec_destroy(e); if (f_.graph_destroy) f_.graph_destroy(g); }
        if (stream_ && f_.stream_destroy) f_.stream_destroy(stream_);
        if (mod_ && f_.unload) f_.unload(mod_);
        if (ctx_ && f_.ctx_destroy) f_.ctx_destroy(ctx_);
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
    void begin_run() { std::lock_guard lk(mu_); in_run_ = true; }
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
    bool resident() { std::lock_guard lk(mu_); return in_run_; }
    void pin(const float* p, std::size_t n) { std::lock_guard lk(mu_); pinned_.push_back({p, n}); }
    void to_host(const float* p, std::size_t n) {
        std::lock_guard lk(mu_); bind();
        bool any = false;
        each_overlap(p, n, [&](entry& e) { if (!e.host_valid) { download(e); any = true; } });
        if (any) sync();
    }
    void host_wrote(const float* p, std::size_t n) {
        std::lock_guard lk(mu_); bind();
        auto lo = reinterpret_cast<std::uintptr_t>(p), hi = lo + n * 4;
        each_overlap(p, n, [&](entry& e) {
            auto el = reinterpret_cast<std::uintptr_t>(e.host), eh = el + e.count * 4;
            auto a = std::max(lo, el), b = std::min(hi, eh);
            ck(f_.htod_async(e.d + (a - el), reinterpret_cast<const void*>(a), b - a, stream_), "memcpy HtoD");
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
    void copy_resident(float* src, std::size_t n, float* dst_base, std::size_t dst_count, std::size_t off) {
        std::lock_guard lk(mu_); bind();
        if (!n) return;
        auto d = acquire(dst_base, dst_count, false);
        auto s = acquire(src, n, false);
        d = acquire(dst_base, dst_count, false);
        d.e->host_valid = false;
        ck(f_.dtod_async(d.ptr + off * 4, s.ptr, n * 4, stream_), "memcpy DtoD");
    }

    // ---- graphs: capture the stream's work between capture_begin/capture_end and replay it ----
    void capture_begin() { std::lock_guard lk(mu_); need_graphs(); bind(); ck(f_.capture_begin(stream_, 2 /* RELAXED: run_op may allocate an output buffer while capturing */), "stream begin capture"); }
    std::size_t capture_end() {
        std::lock_guard lk(mu_); need_graphs(); bind();
        void* g = nullptr; ck(f_.capture_end(stream_, &g), "stream end capture");
        void* e = nullptr; ck(f_.graph_inst(&e, g, 0), "graph instantiate");
        graphs_.push_back({e, g}); return graphs_.size() - 1;
    }
    void graph_launch(std::size_t id) { std::lock_guard lk(mu_); need_graphs(); bind(); ck(f_.graph_launch(graphs_.at(id).first, stream_), "graph launch"); sync(); }

    // ---- multi-device: copy device memory to another runtime (peer copy where available, else via host) ----
    void copy_to_peer(runtime& dst, std::uint64_t dst_ptr, std::uint64_t src_ptr, std::size_t bytes) {
        if (f_.peer && dst.api_ == api_) {
            if (api_ == api::cuda) ck(f_.peer(dst_ptr, reinterpret_cast<std::uintptr_t>(dst.ctx_), src_ptr, reinterpret_cast<std::uintptr_t>(ctx_), bytes), "memcpy peer");
            else ck(f_.peer(dst_ptr, std::uintptr_t(dst.ordinal_), src_ptr, std::uintptr_t(ordinal_), bytes), "memcpy peer");
            return;
        }
        std::vector<char> tmp(bytes);
        { std::lock_guard lk(mu_); bind(); ck(f_.dtoh_async(tmp.data(), src_ptr, bytes, stream_), "memcpy DtoH"); sync(); }
        std::lock_guard lk(dst.mu_); dst.bind(); dst.ck(dst.f_.htod_async(dst_ptr, tmp.data(), bytes, dst.stream_), "memcpy HtoD"); dst.sync();
    }
    std::uint64_t device_alloc(std::size_t bytes) { std::lock_guard lk(mu_); bind(); std::uint64_t p = 0; ck(f_.malloc_(&p, std::max<std::size_t>(4, bytes)), "malloc"); return p; }
    void device_free(std::uint64_t p) { std::lock_guard lk(mu_); bind(); f_.free_(p); }
    void upload(std::uint64_t d, const void* h, std::size_t bytes) { std::lock_guard lk(mu_); bind(); ck(f_.htod_async(d, h, bytes, stream_), "memcpy HtoD"); sync(); }
    void download(void* h, std::uint64_t d, std::size_t bytes) { std::lock_guard lk(mu_); bind(); ck(f_.dtoh_async(h, d, bytes, stream_), "memcpy DtoH"); sync(); }

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
            qd = d; ck(f_.htod_async(qd, q, qbytes, stream_), "memcpy HtoD"); ++stats_.uploads; stats_.qweight_bytes += qbytes;
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
                ck(f_.htod_async(xd, x.data, x.size() * 4, stream_), "memcpy HtoD");
                ck(f_.malloc_(&yd, std::max<std::size_t>(4, out.size() * 4)), "malloc"); temps.push_back(yd);
            }
            std::uint64_t wd = qd; std::uint32_t rb32 = std::uint32_t(rb);
            void* args[] = {&xd, &wd, &yd, &rows, &cols, &k, &fmt, &rb32};
            if (rows >= mmq_min_rows() && has_fn_locked("forge_qmatmul_mmq")) {   // batched prefill: 64x64 tiles
                ++dispatches_["qmatmul_mmq"];
                unsigned grid = ((cols + 63) / 64) * ((rows + 63) / 64);
                if (grid) ck(f_.launch(function("forge_qmatmul_mmq"), grid, 1, 1, 256, 1, 1, 0, stream_, args, nullptr), "launch");
            } else {
                unsigned grid = cols * ((rows + 3) / 4);
                if (grid) ck(f_.launch(fn, grid, 1, 1, 128, 1, 1, 0, stream_, args, nullptr), "launch");
            }
            if (!in_run_) { ck(f_.dtoh_async(out.data, yd, out.size() * 4, stream_), "memcpy DtoH"); sync(); }
        } catch (...) { if (!in_run_) sync(); free_temps(); throw; }
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
        if (grid) ck(f_.launch(function("forge_" + fn), grid, 1, 1, 128, 1, 1, 0, stream_, args.data(), nullptr), "launch");
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
                if (!is_out) ck(f_.htod_async(d, h, cnt * 4, stream_), "memcpy HtoD");
                return d;
            };
            for (std::size_t i = 0; i < in.size(); ++i) {
                ptrs.push_back(buf(in[i].data, in[i].size(), false));
                if (s == sig::rope && i == 1) {   // the inverse-frequency table goes after pos
                    auto& d = rope_[extra];   // the table depends only on (base, dim): upload once
                    if (!d) { ck(f_.malloc_(&d, std::max<std::size_t>(4, extra.size() * 4)), "malloc");
                              ck(f_.htod_async(d, extra.data(), extra.size() * 4, stream_), "memcpy HtoD"); }
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
            if (grid) ck(f_.launch(fn, grid, 1, 1, block, 1, 1, 0, stream_, args.data(), nullptr), "launch");
            if (!in_run_) { ck(f_.dtoh_async(out.data, ptrs.back(), out.size() * 4, stream_), "memcpy DtoH"); sync(); }
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
    struct entry { float* host; std::size_t count; std::uint64_t d; bool host_valid = true, pinned = false, kept = false; };
    struct bound { entry* e; std::uint64_t ptr; };

    template<class F> bool sym(F& f, const char* n) { f = reinterpret_cast<F>(dlsym(lib_, n)); return f != nullptr; }
    template<class F> void opt(F& f, const char* n) { f = reinterpret_cast<F>(dlsym(lib_, n)); }
    void ck(int r, const char* what) {
        if (r != 0) throw std::runtime_error(std::string(api_name(api_)) + ": " + what + " failed (" + std::to_string(r) + ")");
    }
    void need_graphs() { if (!has_graphs()) throw std::runtime_error(std::string(api_name(api_)) + ": driver has no graph API"); }
    void bind() { if (api_ == api::cuda) { if (f_.ctx_set && ctx_) f_.ctx_set(ctx_); } else if (f_.set_device) f_.set_device(ordinal_); }
    void sync() { if (stream_ && f_.stream_sync) ck(f_.stream_sync(stream_), "stream synchronize"); else ck(f_.sync_(), "synchronize"); }
    void* function(const std::string& n) {
        auto it = fn_.find(n); if (it != fn_.end()) return it->second;
        void* f = nullptr; ck(f_.get_fn(&f, mod_, n.c_str()), ("get function " + n).c_str());
        return fn_[n] = f;
    }
    template<class F> void each_overlap(const float* p, std::size_t n, F&& f) {
        auto lo = reinterpret_cast<std::uintptr_t>(p), hi = lo + std::max<std::size_t>(n, 1) * 4;
        for (auto& [k, e] : res_) { auto el = k, eh = k + e.count * 4; if (el < hi && lo < eh) f(e); }
    }
    void download(entry& e) { ck(f_.dtoh_async(e.host, e.d, e.count * 4, stream_), "memcpy DtoH"); e.host_valid = true; ++stats_.downloads; }
    // freed device buffers are pooled by size and reused: decode runs the same shapes every token
    void drop(entry& e) {
        stats_.resident_bytes -= e.count * 4;
        if (pool_bytes_ + e.count * 4 > pool_cap()) { f_.free_(e.d); return; }
        pool_.emplace(std::max<std::size_t>(4, e.count * 4), e.d); pool_bytes_ += std::max<std::size_t>(4, e.count * 4);
    }
    void dev_alloc(std::uint64_t* d, std::size_t bytes) {
        bytes = std::max<std::size_t>(4, bytes);
        if (auto it = pool_.find(bytes); it != pool_.end()) { *d = it->second; pool_.erase(it); pool_bytes_ -= bytes; return; }
        if (f_.malloc_(d, bytes) != 0) {   // out of memory: give the pool back and retry
            for (auto& [k, p] : pool_) f_.free_(p); pool_.clear(); pool_bytes_ = 0;
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
        std::vector<std::uintptr_t> ks; bool dl = false;
        each_overlap(h, count, [&](entry& e) { if (!e.host_valid) { download(e); dl = true; } ks.push_back(reinterpret_cast<std::uintptr_t>(e.host)); });
        if (dl || !ks.empty()) sync();
        for (auto k : ks) { drop(res_[k]); res_.erase(k); }
        entry e{h, count, 0};
        dev_alloc(&e.d, count * 4);
        for (auto& [p, n] : pinned_) if (p == h && n == count) e.pinned = true;
        if (!is_out) { ck(f_.htod_async(e.d, h, count * 4, stream_), "memcpy HtoD"); ++stats_.uploads; }
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
            opt(f_.graph_inst, "cuGraphInstantiateWithFlags"); opt(f_.graph_launch, "cuGraphLaunch");
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
