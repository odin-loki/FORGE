#pragma once
// Metal runtime (macOS / iOS / iPadOS / visionOS). The implementation is Objective-C++ (gpu/src/metal.mm), built
// only on Apple platforms (FORGE_GPU_METAL); elsewhere open() reports "not an Apple platform".
// Kernels: the generated kernels.metal (same bodies as kernels.cu, plus the portable kernels of kernels.cl through
// gpu_gen.py's msl_port), compiled at open() with newLibraryWithSource.
// Apple GPUs have unified memory: buffers are MTLStorageModeShared, so the residency hooks keep nothing on the device
// and each op wraps the host views. The portable kernel families (quantised matmul_t / gather / FFN gate-up,
// attention, DeltaNet state ops, broadcast, mixture of experts) come from gpu::portable_kernels<>; encoded weights
// are the one thing kept in device buffers between ops.
// HARDWARE-UNTESTED: written without a Mac; on Linux only the MSL syntax check (gpu_compile_metal_syntax) and a
// syntax check of metal.mm against stub headers (gpu_compile_metal_host_syntax) run. The portable-kernel entry points
// have NEVER BEEN EXECUTED.
#include <forge/gpu/select.hpp>
#include <forge/gpu/cuda.hpp>
#include <forge/gpu/portable_kernels.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace forge::gpu::metal {

class runtime : public portable_kernels<runtime> {
    friend class portable_kernels<runtime>;
public:
    static int device_count();
    static std::unique_ptr<runtime> open(std::string* why = nullptr, int ordinal = 0);
    ~runtime();
    const char* backend() const { return "metal"; }
    int ordinal() const { return ordinal_; }
    const std::string& device_name() const { return name_; }
    gpu::device_class device_class() const { return gpu::device_class::integrated; }   // unified memory (gpu/select.hpp)
    const std::string& image_arch() const { static const std::string s = "msl"; return s; }
    bool has_graphs() const { return false; }
    bool has_streams() const { return true; }
    using residency_stats = cuda_hip::runtime::residency_stats;
    residency_stats residency() { return {}; }
    // unified memory: nothing to keep in sync. begin_run / end_run only mark the run (the state ops run inside one).
    void begin_run() { std::lock_guard lk(mu_); in_run_ = true; }
    void end_run() { std::lock_guard lk(mu_); in_run_ = false; }
    void pin(const float*, std::size_t) {} void to_host(const float*, std::size_t) {}
    void host_wrote(const float*, std::size_t) {} void release(const float*, std::size_t) {}
    void copy_resident(float* src, std::size_t n, float* dst_base, std::size_t, std::size_t off) { std::copy(src, src + n, dst_base + off); }
    bool run_op(const std::string& op, const std::vector<tensor_view>& in, const ops::attrs& a, tensor_view out);
    std::map<std::string, std::size_t> dispatches() { std::lock_guard lk(mu_); return dispatches_; }
private:
    runtime() = default;
    struct impl; impl* p_ = nullptr;
    int ordinal_ = 0;
    std::string name_;
    std::map<std::string, std::size_t> dispatches_;
    // ---- kernel-call builder for portable_kernels<> (gpu/portable_kernels.hpp); mu_ held ----
    // No residency: a call only records its arguments; k_launch wraps each host tensor in a shared buffer (copied in
    // unless it is output-only), runs the kernel to completion and copies outputs and in/out tensors back.
    struct entry { bool pinned = false; };
    struct kcall {
        struct arg { int kind = 0; const float* h = nullptr; std::size_t cnt = 0; int mode = 0; void* dev = nullptr; std::uint32_t u = 0; float f = 0; };   // kind 0 host tensor, 1 device buffer, 2 u32, 3 f32
        std::vector<arg> args;
        std::vector<void*> temps;   // k_scratch buffers, released in k_done / k_abort
    };
    void begin_op() { op_mark_ = clock_; }
    void k_touch(const float*, std::size_t, int) {}
    entry* k_buf(kcall& c, const float* h, std::size_t cnt, int mode) { kcall::arg a; a.h = h; a.cnt = cnt; a.mode = mode; c.args.push_back(a); return nullptr; }
    void k_dev(kcall& c, void* m) { kcall::arg a; a.kind = 1; a.dev = m; c.args.push_back(a); }
    void k_u32(kcall& c, std::uint32_t v) { kcall::arg a; a.kind = 2; a.u = v; c.args.push_back(a); }
    void k_f32(kcall& c, float v) { kcall::arg a; a.kind = 3; a.f = v; c.args.push_back(a); }
    void* k_scratch(kcall& c, std::size_t bytes);
    void k_upload(void* m, const void* h, std::size_t bytes);
    void k_launch(kcall& c, const std::string& name, std::size_t global);
    void k_done(kcall& c);
    void k_abort(kcall& c) { k_done(c); }
    bool k_has(const std::string& n);
    void* qweight(const std::uint8_t* q, std::size_t bytes);   // encoded weight in a device buffer, cached by host address and size
    std::mutex mu_;
    bool in_run_ = false, capturing_ = false;
    std::vector<std::pair<const float*, std::size_t>> pinned_;
    std::uint64_t clock_ = 0, op_mark_ = 0;   // encoded weights touched after begin_op() belong to the op in progress
};

#ifndef FORGE_GPU_METAL
inline int runtime::device_count() { return 0; }
inline std::unique_ptr<runtime> runtime::open(std::string* why, int) { if (why) *why = "metal: not an Apple platform build"; return nullptr; }
inline runtime::~runtime() {}
inline bool runtime::run_op(const std::string&, const std::vector<tensor_view>&, const ops::attrs&, tensor_view) { return false; }
inline void* runtime::k_scratch(kcall&, std::size_t) { return nullptr; }
inline void runtime::k_upload(void*, const void*, std::size_t) {}
inline void runtime::k_launch(kcall& c, const std::string&, std::size_t) { c.args.clear(); }
inline void runtime::k_done(kcall& c) { c.temps.clear(); }
inline bool runtime::k_has(const std::string&) { return false; }
inline void* runtime::qweight(const std::uint8_t*, std::size_t) { return nullptr; }
#endif

} // namespace forge::gpu::metal
