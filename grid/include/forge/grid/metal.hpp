#pragma once
// Metal runtime (macOS / iOS / iPadOS / visionOS). The implementation is Objective-C++ (grid/src/metal.mm), built
// only on Apple platforms (FORGE_GRID_METAL); elsewhere open() reports "not an Apple platform".
// Kernels: the generated kernels.metal (same bodies as kernels.cu), compiled at open() with newLibraryWithSource.
// Apple GPUs have unified memory: buffers are MTLStorageModeShared, so the residency hooks are no-ops and each
// op wraps the host views (page-aligned ones without a copy).
// HARDWARE-UNTESTED: written without a Mac; on Linux only the MSL syntax check (gpu_compile_metal_syntax) runs.
#include <forge/grid/cuda.hpp>
#include <map>
#include <memory>
#include <string>

namespace forge::grid::metal {

class runtime {
public:
    static int device_count();
    static std::unique_ptr<runtime> open(std::string* why = nullptr, int ordinal = 0);
    ~runtime();
    const char* backend() const { return "metal"; }
    int ordinal() const { return ordinal_; }
    const std::string& device_name() const { return name_; }
    const std::string& image_arch() const { static const std::string s = "msl"; return s; }
    bool has_graphs() const { return false; }
    bool has_streams() const { return true; }
    using residency_stats = gpu::runtime::residency_stats;
    residency_stats residency() { return {}; }
    // unified memory: nothing to keep in sync
    void begin_run() {} void end_run() {} void pin(const float*, std::size_t) {} void to_host(const float*, std::size_t) {}
    void host_wrote(const float*, std::size_t) {} void release(const float*, std::size_t) {}
    void copy_resident(float* src, std::size_t n, float* dst_base, std::size_t, std::size_t off) { std::copy(src, src + n, dst_base + off); }
    bool run_op(const std::string& op, const std::vector<tensor_view>& in, const spec::attrs& a, tensor_view out);
    std::map<std::string, std::size_t> dispatches() { return dispatches_; }
private:
    runtime() = default;
    struct impl; impl* p_ = nullptr;
    int ordinal_ = 0;
    std::string name_;
    std::map<std::string, std::size_t> dispatches_;
};

#ifndef FORGE_GRID_METAL
inline int runtime::device_count() { return 0; }
inline std::unique_ptr<runtime> runtime::open(std::string* why, int) { if (why) *why = "metal: not an Apple platform build"; return nullptr; }
inline runtime::~runtime() {}
inline bool runtime::run_op(const std::string&, const std::vector<tensor_view>&, const spec::attrs&, tensor_view) { return false; }
#endif

} // namespace forge::grid::metal
