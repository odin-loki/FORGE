#pragma once
// CUDA / HIP host runtime. The driver library (libcuda / libamdhip64) is dlopened at run time, so FORGE has
// no link or build dependency on either SDK. Kernels come from the same generated single source as the
// compile checks (spec/gen/gpu_gen.py -> kernels.cu), built at FORGE build time into PTX (CUDA, JIT-compiled
// by the driver for the installed GPU) and an AMDGPU code object (HIP), and embedded below.
// Without a driver or a device, open() fails with a reason and callers fall back to the CPU.
#include <forge/grid/gpu_images.gen.hpp>
#include <forge/core/tensor.hpp>
#include <forge/spec/ops.gen.hpp>
#include <dlfcn.h>
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
enum class sig { unary, binary, scale, rows, rms, matmul };
inline bool kernel_sig(const std::string& op, sig& s) {
    static const std::map<std::string, sig> m = [] {
        std::map<std::string, sig> r;
        for (auto n : {"neg", "abs", "sqrt", "rsqrt", "recip", "square", "exp", "exp2", "log", "log2", "tanh", "sigmoid",
                       "silu", "relu", "softplus", "gelu_tanh"}) r[n] = sig::unary;
        for (auto n : {"add", "sub", "mul", "div", "maximum", "minimum", "silu_mul"}) r[n] = sig::binary;
        r["scale"] = sig::scale; r["softmax"] = sig::rows; r["rms_norm"] = sig::rms; r["matmul_t"] = sig::matmul;
        return r; }();
    auto it = m.find(op); if (it == m.end()) return false; s = it->second; return true;
}

// Embedded modules for an API: one PTX (JIT-compiled by the driver for any newer GPU) or one AMDGPU code
// object per gfx target. Empty if this build could not produce them.
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
    case sig::binary: return in.size() == 2 && in[1] == x;
    case sig::rows: return !x.empty() && (a.num("axis", -1) == -1 || a.num("axis", -1) == double(x.size()) - 1);
    case sig::rms: return in.size() == 2 && !x.empty() && in[1].size() == 1 && in[1][0] == x.back();
    case sig::matmul: return in.size() == 2 && x.size() >= 2 && in[1].size() == 2 && in[1][1] == x.back();
    }
    return false;
}

class runtime {
public:
    static std::unique_ptr<runtime> open(api a, std::string* why = nullptr) {
        std::unique_ptr<runtime> r(new runtime(a));
        std::string e = r->init();
        if (!e.empty()) { if (why) *why = std::string(api_name(a)) + ": " + e; return nullptr; }
        return r;
    }
    ~runtime() {
        if (mod_ && f_.unload) f_.unload(mod_);
        if (ctx_ && f_.ctx_destroy) f_.ctx_destroy(ctx_);
        if (lib_) dlclose(lib_);
    }
    api which() const { return api_; }
    const std::string& device_name() const { return name_; }
    const std::string& image_arch() const { return arch_; }
    std::map<std::string, std::size_t> dispatches() { std::lock_guard lk(mu_); return dispatches_; }

    // Run one op on the device over host views; false if there is no kernel for this op/shape.
    bool run_op(const std::string& op, const std::vector<tensor_view>& in, const spec::attrs& a, tensor_view out) {
        std::vector<shape_t> ss; for (auto& v : in) ss.push_back(v.shape);
        if (!supports(op, ss, a)) return false;
        sig s; kernel_sig(op, s);
        auto& x = in[0];
        std::uint32_t n = std::uint32_t(out.size()), rows = 0, cols = 0, k = 0, threads = n;
        switch (s) {
        case sig::unary: case sig::scale: case sig::binary: break;
        case sig::rows:
            cols = std::uint32_t(x.shape.back()); rows = cols ? std::uint32_t(x.size()) / cols : 0; threads = rows; break;
        case sig::rms:
            cols = std::uint32_t(x.shape.back()); rows = cols ? std::uint32_t(x.size()) / cols : 0; threads = rows; break;
        case sig::matmul:
            k = std::uint32_t(x.shape.back()); rows = std::uint32_t(x.size()) / k; cols = std::uint32_t(in[1].shape[0]); threads = rows * cols; break;
        }
        std::lock_guard lk(mu_);
        ++dispatches_[op];
        void* fn = function("forge_" + op);
        std::vector<std::uint64_t> d;
        auto free_all = [&] { for (auto p : d) f_.free_(p); };
        try {
            for (auto& v : in) { d.push_back(alloc(v.size())); ck(f_.htod(d.back(), v.data, v.size() * 4), "memcpy HtoD"); }
            d.push_back(alloc(out.size()));
            std::vector<void*> args;
            for (auto& p : d) args.push_back(&p);
            float alpha = float(a.num("alpha", 1.0)), eps = float(a.num("eps", 1e-6));
            switch (s) {
            case sig::unary: case sig::binary: args.push_back(&n); break;
            case sig::scale: args.push_back(&n); args.push_back(&alpha); break;
            case sig::rows: args.push_back(&rows); args.push_back(&cols); break;
            case sig::rms: args.push_back(&rows); args.push_back(&cols); args.push_back(&eps); break;
            case sig::matmul: args.push_back(&rows); args.push_back(&cols); args.push_back(&k); break;
            }
            unsigned block = 64, grid = (threads + block - 1) / block;
            if (grid) ck(f_.launch(fn, grid, 1, 1, block, 1, 1, 0, nullptr, args.data(), nullptr), "launch");
            ck(f_.sync(), "synchronize");
            ck(f_.dtoh(out.data, d.back(), out.size() * 4), "memcpy DtoH");
        } catch (...) { free_all(); throw; }
        free_all();
        return true;
    }

private:
    explicit runtime(api a) : api_(a) {}
    // Driver entry points, normalised: CUDA and HIP module APIs have the same shapes.
    struct fns {
        int (*init)(unsigned) = nullptr;
        int (*count)(int*) = nullptr;
        int (*name)(char*, int, int) = nullptr;             // (buf, len, device)
        int (*ctx_create)(void**, unsigned, int) = nullptr; // CUDA only
        int (*ctx_destroy)(void*) = nullptr;
        int (*set_device)(int) = nullptr;                    // HIP only
        int (*load)(void**, const void*) = nullptr;
        int (*unload)(void*) = nullptr;
        int (*get_fn)(void**, void*, const char*) = nullptr;
        int (*malloc_)(std::uint64_t*, std::size_t) = nullptr;
        int (*free_)(std::uint64_t) = nullptr;
        int (*htod)(std::uint64_t, const void*, std::size_t) = nullptr;
        int (*dtoh)(void*, std::uint64_t, std::size_t) = nullptr;
        int (*launch)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**) = nullptr;
        int (*sync)() = nullptr;
    };
    template<class F> bool sym(F& f, const char* n) { f = reinterpret_cast<F>(dlsym(lib_, n)); return f != nullptr; }
    void ck(int r, const char* what) {
        if (r != 0) throw std::runtime_error(std::string(api_name(api_)) + ": " + what + " failed (" + std::to_string(r) + ")");
    }
    std::uint64_t alloc(std::size_t count) { std::uint64_t p = 0; ck(f_.malloc_(&p, std::max<std::size_t>(4, count * 4)), "malloc"); return p; }
    void* function(const std::string& n) {
        auto it = fn_.find(n); if (it != fn_.end()) return it->second;
        void* f = nullptr; ck(f_.get_fn(&f, mod_, n.c_str()), ("get function " + n).c_str());
        return fn_[n] = f;
    }
    std::string init() {
        auto imgs = module_images(api_);
        if (imgs.empty()) return "this build has no embedded kernel image (no clang with the GPU target at build time)";
        const char* libs_cuda[] = {"libcuda.so.1", "libcuda.so", nullptr};
        const char* libs_hip[] = {"libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so", nullptr};
        for (auto p = api_ == api::cuda ? libs_cuda : libs_hip; *p; ++p) if ((lib_ = dlopen(*p, RTLD_NOW | RTLD_LOCAL))) break;
        if (!lib_) return api_ == api::cuda ? "libcuda not found (no NVIDIA driver)" : "libamdhip64 not found (no ROCm runtime)";
        bool ok;
        if (api_ == api::cuda) {
            ok = sym(f_.init, "cuInit") && sym(f_.count, "cuDeviceGetCount") && sym(f_.name, "cuDeviceGetName") &&
                 sym(f_.ctx_create, "cuCtxCreate_v2") && sym(f_.ctx_destroy, "cuCtxDestroy_v2") && sym(f_.load, "cuModuleLoadData") &&
                 sym(f_.unload, "cuModuleUnload") && sym(f_.get_fn, "cuModuleGetFunction") && sym(f_.malloc_, "cuMemAlloc_v2") &&
                 sym(f_.free_, "cuMemFree_v2") && sym(f_.htod, "cuMemcpyHtoD_v2") && sym(f_.dtoh, "cuMemcpyDtoH_v2") &&
                 sym(f_.launch, "cuLaunchKernel") && sym(f_.sync, "cuCtxSynchronize");
        } else {
            ok = sym(f_.init, "hipInit") && sym(f_.count, "hipGetDeviceCount") && sym(f_.name, "hipDeviceGetName") &&
                 sym(f_.set_device, "hipSetDevice") && sym(f_.load, "hipModuleLoadData") && sym(f_.unload, "hipModuleUnload") &&
                 sym(f_.get_fn, "hipModuleGetFunction") && sym(f_.malloc_, "hipMalloc") && sym(f_.free_, "hipFree") &&
                 sym(f_.htod, "hipMemcpyHtoD") && sym(f_.dtoh, "hipMemcpyDtoH") && sym(f_.launch, "hipModuleLaunchKernel") &&
                 sym(f_.sync, "hipDeviceSynchronize");
        }
        if (!ok) return "driver library is missing entry points";
        if (f_.init(0) != 0) return "driver init failed";
        int n = 0; if (f_.count(&n) != 0 || n <= 0) return "no devices";
        char buf[256] = {}; f_.name(buf, sizeof buf, 0); name_ = buf;
        if (api_ == api::cuda) { if (f_.ctx_create(&ctx_, 0, 0) != 0) return "context creation failed"; }
        else if (f_.set_device(0) != 0) return "hipSetDevice failed";
        for (auto& [arch, img] : imgs) {   // the driver rejects code objects for other gfx targets
            std::string image(img);          // PTX must be NUL-terminated
            if (f_.load(&mod_, image.data()) == 0) { arch_ = arch; return {}; }
            mod_ = nullptr;
        }
        return "module load failed (no embedded PTX/code object is accepted by this device)";
    }
    api api_;
    void* lib_ = nullptr; void* ctx_ = nullptr; void* mod_ = nullptr;
    fns f_;
    std::string name_, arch_;
    std::mutex mu_;
    std::map<std::string, void*> fn_;
    std::map<std::string, std::size_t> dispatches_;
};

} // namespace forge::grid::gpu
