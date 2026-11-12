#pragma once
// Level Zero host runtime (Intel GPUs: Arc, Data Center GPU Max/Flex, integrated Xe; also the backend SYCL/
// oneAPI programs run on). libze_loader is dlopened; types come from the vendored MIT ze_api.h. Kernels are
// the generated OpenCL C compiled offline to SPIR-V (clang -> LLVM IR -> llvm-spirv) at build time; builds
// without llvm-spirv have no image and open() reports that (use FORGE_DEVICE=opencl on the same GPU instead).
//
// HARDWARE-UNTESTED beyond the CPU mock (tests/mock_ze.c). Same contract as gpu/cuda.hpp: run_op, the
// residency protocol, recorded graphs (replayed by re-appending to the immediate command list), multi-device
// by ordinal (all devices of all drivers), host-staged copies between runtimes.
#include <forge/gpu/cuda.hpp>
#include <forge/gpu/portable_kernels.hpp>
#include <forge/gpu/kernel_cache.hpp>
#include <set>
#include <level-zero/ze_api.h>
#include <forge/core/dl.hpp>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace forge::gpu::ze {

using cuda_hip::sig;
class runtime : public portable_kernels<runtime> {
    friend class portable_kernels<runtime>;
public:
    static int device_count() { runtime p; if (!p.load_lib().empty() || p.f_.init(0) != 0) return 0; return int(p.all_devices().size()); }
    static std::unique_ptr<runtime> open(std::string* why = nullptr, int ordinal = 0) {
        std::unique_ptr<runtime> r(new runtime);
        r->ordinal_ = ordinal;
        std::string e = r->init();
        if (!e.empty()) { if (why) *why = "level_zero: " + e; return nullptr; }
        return r;
    }
    ~runtime() {
        dm_.detach();
        try { std::lock_guard lk(mu_); for (auto& [k, e] : res_) f_.free_(ctx_, e.d); res_.clear(); } catch (...) {}
        for (auto& [k, q] : qres_) f_.free_(ctx_, q.mem);
        for (auto p : keep_) f_.free_(ctx_, p);
        for (auto& [n, k] : kern_) f_.kernel_destroy(k);
        if (mod_) f_.module_destroy(mod_);
        if (list_) f_.list_destroy(list_);
        if (ctx_) f_.ctx_destroy(ctx_);
        if (lib_) dlclose(lib_);
    }
    const char* backend() const { return "level_zero"; }
    int ordinal() const { return ordinal_; }
    const std::string& device_name() const { return name_; }
    // For automatic device selection (gpu/select.hpp): ze_device_properties_t type and the INTEGRATED flag.
    gpu::device_class device_class() const { return software_name(name_) || cpu_type_ ? gpu::device_class::software : integrated_ ? gpu::device_class::integrated : gpu::device_class::discrete; }
    const std::string& image_arch() const { static const std::string s = "spirv"; return s; }
    bool has_graphs() const { return true; }
    bool has_streams() const { return true; }
    // Native module binary cache (gpu/kernel_cache.hpp; $FORGE_GPU_KERNEL_CACHE=0 off): true when open() created the
    // module from the cached native binary (zeModuleGetNativeBinary) instead of compiling the SPIR-V.
    bool program_from_cache() const { return from_cache_; }
    const std::string& kernel_cache_path() const { return cache_path_; }
    std::map<std::string, std::size_t> dispatches() { std::lock_guard lk(mu_); return dispatches_; }
    struct mem_info { std::size_t free = 0, total = 0; };
    mem_info memory() { mem_info m; m.total = total_; m.free = total_ > stats_.resident_bytes ? total_ - stats_.resident_bytes : 0; return m; }
    using residency_stats = cuda_hip::runtime::residency_stats;
    residency_stats residency() { std::lock_guard lk(mu_); auto s = stats_; s.resident = res_.size(); return s; }
    mem::budget* memory_pool() { return dm_.pool(); }   // "level_zero:N" (gpu/device_memory.hpp)
    std::uint64_t charged_bytes() { std::lock_guard lk(mu_); return dm_.charged(); }

    void begin_run() { std::lock_guard lk(mu_); in_run_ = true; }
    void end_run() {
        std::lock_guard lk(mu_);
        for (auto it = res_.begin(); it != res_.end();) {
            auto& e = it->second; if (!e.host_valid) download(e);
            if (e.pinned) { ++it; continue; }
            sync(); drop(e); it = res_.erase(it);
        }
        sync(); in_run_ = false;
    }
    bool resident() { std::lock_guard lk(mu_); return in_run_; }
    void pin(const float* p, std::size_t n) { std::lock_guard lk(mu_); pinned_.push_back({p, n}); }
    void to_host(const float* p, std::size_t n) { std::lock_guard lk(mu_); each_overlap(p, n, [&](entry& e) { if (!e.host_valid) download(e); }); sync(); }
    void host_wrote(const float* p, std::size_t n) {
        std::lock_guard lk(mu_);
        auto lo = reinterpret_cast<std::uintptr_t>(p), hi = lo + n * 4;
        each_overlap(p, n, [&](entry& e) {
            auto el = reinterpret_cast<std::uintptr_t>(e.host), eh = el + e.count * 4;
            auto a = std::max(lo, el), b = std::min(hi, eh);
            copy(static_cast<char*>(e.d) + (a - el), reinterpret_cast<const void*>(a), b - a);
        });
        sync();
    }
    void release(const float* p, std::size_t n) {
        std::lock_guard lk(mu_);
        std::vector<std::uintptr_t> ks;
        each_overlap(p, n, [&](entry& e) { if (!e.host_valid) download(e); ks.push_back(reinterpret_cast<std::uintptr_t>(e.host)); });
        sync(); for (auto k : ks) { drop(res_[k]); res_.erase(k); }
        // the range is going away (or being replaced): its pins go too, or a later buffer at the same address and size
        // would inherit one and outlive its run as a stale resident entry
        std::erase_if(pinned_, [&](const auto& pr) { return pr.first < p + std::max<std::size_t>(n, 1) && p < pr.first + std::max<std::size_t>(pr.second, 1); });
    }
    // [p, p+n) is going away for good (a plan's constants, a state slot): drop every overlapping device copy without
    // fetching it and the pin records inside the range. Without this a later host buffer at the same address would
    // inherit the pin or be taken for a sub-range of the stale entry, and that entry's next sync-back would write
    // its old size over the new buffer (graph::plan::~plan calls it through device_hooks::forget).
    void forget(const float* p, std::size_t n) {
        std::lock_guard lk(mu_); sync();
        std::vector<std::uintptr_t> ks;
        each_overlap(p, n, [&](entry& e) { ks.push_back(reinterpret_cast<std::uintptr_t>(e.host)); });
        for (auto k : ks) { drop(res_[k]); res_.erase(k); }
        std::erase_if(pinned_, [&](const auto& pr) { return pr.first < p + std::max<std::size_t>(n, 1) && p < pr.first + std::max<std::size_t>(pr.second, 1); });
    }
    void copy_resident(float* src, std::size_t n, float* dst_base, std::size_t dst_count, std::size_t off) {
        std::lock_guard lk(mu_); if (!n) return;
        begin_op();
        acquire(dst_base, dst_count, false);
        void* s = acquire(src, n, false).ptr;
        auto d = acquire(dst_base, dst_count, false);
        d.e->host_valid = false;
        copy(static_cast<char*>(d.ptr) + off * 4, s, n * 4);
    }

    void capture_begin() { std::lock_guard lk(mu_); capturing_ = true; cur_.clear(); }
    std::size_t capture_end() { std::lock_guard lk(mu_); capturing_ = false; graphs_.push_back(std::move(cur_)); cur_.clear(); return graphs_.size() - 1; }
    void graph_launch(std::size_t id) { std::lock_guard lk(mu_); for (auto& l : graphs_.at(id)) enqueue(l); sync(); }

    std::uint64_t device_alloc(std::size_t bytes) { std::lock_guard lk(mu_); begin_op(); return reinterpret_cast<std::uintptr_t>(alloc(bytes, "raw")); }
    void device_free(std::uint64_t p) { std::lock_guard lk(mu_); dfree(reinterpret_cast<void*>(p)); }
    void upload(std::uint64_t d, const void* h, std::size_t bytes) { std::lock_guard lk(mu_); copy(reinterpret_cast<void*>(d), h, bytes); sync(); }
    void download(void* h, std::uint64_t d, std::size_t bytes) { std::lock_guard lk(mu_); copy(h, reinterpret_cast<void*>(d), bytes); sync(); }
    void copy_to_peer(runtime& dst, std::uint64_t dst_ptr, std::uint64_t src_ptr, std::size_t bytes) {
        std::vector<char> t(bytes); download(t.data(), src_ptr, bytes); dst.upload(dst_ptr, t.data(), bytes);
    }

    bool run_op(const std::string& op, const std::vector<tensor_view>& in, const ops::attrs& a, tensor_view out) {
        std::vector<shape_t> ss; for (auto& v : in) ss.push_back(v.shape);
        if (!cuda_hip::supports(op, ss, a)) return false;
        if (op == "attention") if (int r = attention_port(in, a, out); r >= 0) return true;   // one-pass / split-K kernels
        auto L = cuda_hip::kernel_args(op, in, a, out);
        std::lock_guard lk(mu_); begin_op();
        ++dispatches_[op];
        // Settle the residency map before binding anything: acquiring a range that partly overlaps a resident entry
        // drops that entry (after syncing it back), and an input bound earlier in this same launch may live inside
        // it (an activation inside a stale larger entry of the reused arena, then an output that overlaps the same
        // entry). Its handle would then be freed while the kernel still uses it. After this pass every range has
        // its entry, so the binds below only hit the cache.
        if (in_run_) {
            for (auto& v : in) acquire(const_cast<float*>(v.data), v.size(), false);
            acquire(out.data, out.size(), true);
            for (auto& v : in) acquire(const_cast<float*>(v.data), v.size(), false);
        }
        launch l; l.kernel = kernel("forge_" + op); l.global = L.threads;
        std::vector<void*> temps;
        auto free_temps = [&] { for (auto p : temps) dfree(p); };
        try {
            auto buf = [&](const float* h, std::size_t cnt, bool is_out) {
                void* d;
                if (in_run_) { auto b = acquire(const_cast<float*>(h), cnt, is_out); if (is_out) b.e->host_valid = false; d = b.ptr; }
                else { d = alloc(cnt * 4, "scratch"); temps.push_back(d); if (!is_out) copy(d, h, cnt * 4); }
                l.args.push_back({d, 0, false}); return d;
            };
            for (std::size_t i = 0; i < in.size(); ++i) {
                buf(in[i].data, in[i].size(), false);
                if (L.s == sig::rope && i == 1) {
                    void* d = alloc(L.rope_inv.size() * 4, "scratch"); (capturing_ ? keep_ : temps).push_back(d);
                    copy(d, L.rope_inv.data(), L.rope_inv.size() * 4); l.args.push_back({d, 0, false});
                }
            }
            void* od = buf(out.data, out.size(), true);
            for (auto& sc : L.scalars) { arg g{nullptr, sc.u, true}; if (sc.is_float) std::memcpy(&g.v, &sc.f, 4); l.args.push_back(g); }
            if (L.threads) enqueue(l);
            if (capturing_) cur_.push_back(l);
            if (!in_run_) { copy(out.data, od, out.size() * 4); sync(); }
        } catch (...) { sync(); free_temps(); throw; }
        if (!temps.empty()) { sync(); free_temps(); }
        return true;
    }

private:
    runtime() = default;
    struct arg { void* ptr; std::uint32_t v; bool scalar; };
    struct launch { ze_kernel_handle_t kernel = nullptr; std::uint32_t global = 0; std::vector<arg> args; };
    struct entry { float* host; std::size_t count; void* d; bool host_valid = true, pinned = false; std::uint64_t last = 0; };
    struct bound { entry* e; void* ptr; };
    // ---- kernel-call builder for portable_kernels<> (gpu/portable_kernels.hpp); mu_ held ----
    struct kcall {
        launch l; std::vector<void*> temps;
        struct rd_t { void* d; float* h; std::size_t bytes; }; std::vector<rd_t> rd;   // outputs to read back (outside a run)
    };
    struct qentry { void* mem = nullptr; std::size_t bytes = 0; std::uint64_t last = 0; };
    void k_touch(const float* h, std::size_t cnt, int mode) { if (in_run_) acquire(const_cast<float*>(h), cnt, mode == 1); }
    entry* k_buf(kcall& c, const float* h, std::size_t cnt, int mode) {   // mode 0 in, 1 out, 2 in/out
        void* dv; entry* ent = nullptr;
        if (in_run_) { auto b = acquire(const_cast<float*>(h), cnt, mode == 1); if (mode) b.e->host_valid = false; dv = b.ptr; ent = b.e; }
        else {
            dv = alloc(cnt * 4, "scratch"); c.temps.push_back(dv);
            if (mode != 1) copy(dv, h, cnt * 4);
            if (mode) c.rd.push_back({dv, const_cast<float*>(h), cnt * 4});
        }
        c.l.args.push_back({dv, 0, false}); return ent;
    }
    void k_dev(kcall& c, void* p) { c.l.args.push_back({p, 0, false}); }
    void k_u32(kcall& c, std::uint32_t v) { c.l.args.push_back({nullptr, v, true}); }
    void k_f32(kcall& c, float v) { arg g{nullptr, 0, true}; std::memcpy(&g.v, &v, 4); c.l.args.push_back(g); }
    void* k_scratch(kcall& c, std::size_t bytes) { void* p = alloc(bytes, "scratch"); c.temps.push_back(p); return p; }
    void k_upload(void* p, const void* h, std::size_t bytes) { copy(p, h, bytes); sync(); }
    void k_launch(kcall& c, const std::string& name, std::size_t global) {
        c.l.kernel = kernel(name); c.l.global = std::uint32_t(global);
        if (global) enqueue(c.l);
        if (capturing_) cur_.push_back(c.l);
        c.l.args.clear();
    }
    void k_done(kcall& c) {
        for (auto& r : c.rd) copy(r.h, r.d, r.bytes);
        if (c.temps.empty() && c.rd.empty()) return;
        sync();
        if (capturing_) keep_.insert(keep_.end(), c.temps.begin(), c.temps.end());   // a recorded launch still refers to them
        else for (auto p : c.temps) dfree(p);
        c.temps.clear();
    }
    void k_abort(kcall& c) { try { sync(); } catch (...) {} for (auto p : c.temps) dfree(p); c.temps.clear(); }
    bool k_has(const std::string& n) {
        if (kern_.count(n)) return true;
        if (missing_.count(n)) return false;
        ze_kernel_desc_t kd{}; kd.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC; kd.pKernelName = n.c_str();
        ze_kernel_handle_t k = nullptr;
        if (f_.kernel_create(mod_, &kd, &k) != ZE_RESULT_SUCCESS || !k) { missing_.insert(n); return false; }
        ck(f_.group_size(k, 64, 1, 1), "zeKernelSetGroupSize");
        kern_[n] = k; return true;
    }
    // An encoded weight (or any read-only blob) uploaded once and kept for the runtime's lifetime, keyed by host address
    // and size; evictable least recently used first (the host copy is the source of truth, so nothing is synced back).
    void* qweight(const std::uint8_t* q, std::size_t bytes) {
        auto key = std::make_pair(reinterpret_cast<std::uintptr_t>(q), bytes);
        if (auto it = qres_.find(key); it != qres_.end()) { it->second.last = ++clock_; return it->second.mem; }
        void* p = alloc(bytes, "weights");
        try { copy(p, q, bytes); sync(); } catch (...) { dfree(p); throw; }
        qres_[key] = {p, bytes, ++clock_}; ++stats_.uploads; stats_.qweight_bytes += bytes;
        return p;
    }
    // Encoded weights not bound by the op in progress, least recently used first (re-uploaded from the host on next use).
    std::uint64_t evict_weights(std::uint64_t want) {
        std::vector<std::pair<std::uint64_t, std::pair<std::uintptr_t, std::size_t>>> vs;
        for (auto& [k, q] : qres_) if (q.last <= op_mark_) vs.push_back({q.last, k});
        if (vs.empty()) return 0;
        std::sort(vs.begin(), vs.end()); sync();
        std::uint64_t got = 0;
        for (auto& [last, k] : vs) {
            if (want && got >= want) break;
            auto& q = qres_[k]; got += q.bytes; stats_.qweight_bytes -= q.bytes; dfree(q.mem); qres_.erase(k); ++stats_.evictions;
        }
        stats_.evicted_bytes += got; return got;
    }
    struct fns {
        decltype(&::zeInit) init = nullptr;
        decltype(&::zeDriverGet) drivers = nullptr;
        decltype(&::zeDeviceGet) devices = nullptr;
        decltype(&::zeDeviceGetProperties) props = nullptr;
        decltype(&::zeDeviceGetMemoryProperties) mem_props = nullptr;
        decltype(&::zeContextCreate) ctx_create = nullptr;
        decltype(&::zeContextDestroy) ctx_destroy = nullptr;
        decltype(&::zeCommandListCreateImmediate) list_create = nullptr;
        decltype(&::zeCommandListDestroy) list_destroy = nullptr;
        decltype(&::zeCommandListHostSynchronize) list_sync = nullptr;
        decltype(&::zeCommandListAppendMemoryCopy) memcpy_ = nullptr;
        decltype(&::zeCommandListAppendBarrier) barrier = nullptr;
        decltype(&::zeCommandListAppendLaunchKernel) launch = nullptr;
        decltype(&::zeMemAllocDevice) alloc = nullptr;
        decltype(&::zeMemFree) free_ = nullptr;
        decltype(&::zeModuleCreate) module_create = nullptr;
        decltype(&::zeModuleDestroy) module_destroy = nullptr;
        decltype(&::zeModuleBuildLogGetString) log_get = nullptr;
        decltype(&::zeModuleBuildLogDestroy) log_destroy = nullptr;
        decltype(&::zeKernelCreate) kernel_create = nullptr;
        decltype(&::zeKernelDestroy) kernel_destroy = nullptr;
        decltype(&::zeKernelSetGroupSize) group_size = nullptr;
        decltype(&::zeKernelSetArgumentValue) set_arg = nullptr;
        // optional (native binary cache): absent entry points only turn the cache off
        decltype(&::zeModuleGetNativeBinary) native_bin = nullptr;
        decltype(&::zeDriverGetProperties) drv_props = nullptr;
    };
    template<class F> bool sym(F& f, const char* n) { f = reinterpret_cast<F>(dlsym(lib_, n)); return f != nullptr; }
    void ck(ze_result_t r, const char* what) { if (r != ZE_RESULT_SUCCESS) throw std::runtime_error(std::string("level_zero: ") + what + " failed (" + std::to_string(int(r)) + ")"); }
    void sync() { ck(f_.list_sync(list_, UINT64_MAX), "zeCommandListHostSynchronize"); }
    void copy(void* dst, const void* src, std::size_t n) {
        ck(f_.memcpy_(list_, dst, src, n, nullptr, 0, nullptr), "zeCommandListAppendMemoryCopy");
        ck(f_.barrier(list_, nullptr, 0, nullptr), "zeCommandListAppendBarrier");
    }
    // ---- device memory accounting (gpu/device_memory.hpp): pool "level_zero:N" ----
    device_memory dm_;
    std::uint64_t clock_ = 0, op_mark_ = 0;   // buffers touched after begin_op() belong to the op in progress
    void begin_op() { op_mark_ = clock_; }
    void* alloc(std::size_t bytes, const char* tag) {
        bytes = std::max<std::size_t>(4, bytes);
        return dm_.allocate(bytes, tag, [&]() -> std::optional<void*> {
            ze_device_mem_alloc_desc_t d{}; d.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
            void* p = nullptr; auto r = f_.alloc(ctx_, &d, bytes, 64, dev_, &p);
            if (r == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY || r == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY) return std::nullopt;
            ck(r, "zeMemAllocDevice"); return p; });
    }
    void dfree(void* p) { dm_.release(p); f_.free_(ctx_, p); }
    // device_memory's evict callback (mu_ held): resident buffers least recently used first, synced to the host
    // first; never one the op in progress has bound, and nothing while a graph is being recorded.
    std::uint64_t evict_device(std::uint64_t want, bool external) {
        if (capturing_) return 0;
        std::uint64_t got = evict_tensors(want, external);
        if (!want || got < want) got += evict_weights(want ? want - got : 0);
        return got;
    }
    std::uint64_t evict_tensors(std::uint64_t want, bool external) {
        if (external) op_mark_ = clock_;
        std::vector<std::pair<std::uint64_t, std::uintptr_t>> vs;
        for (auto& [k, e] : res_) if (e.last <= op_mark_) vs.push_back({e.last, k});
        std::sort(vs.begin(), vs.end());
        std::size_t n = 0; std::uint64_t got = 0;
        while (n < vs.size() && (!want || got < want)) got += std::max<std::size_t>(4, res_[vs[n++].second].count * 4);
        if (!n) return 0;
        for (std::size_t i = 0; i < n; ++i) { auto& e = res_[vs[i].second]; if (!e.host_valid) download(e); }
        sync();
        for (std::size_t i = 0; i < n; ++i) { drop(res_[vs[i].second]); res_.erase(vs[i].second); }
        stats_.evictions += n; stats_.evicted_bytes += got;
        return got;
    }
    void enqueue(const launch& l) {
        for (std::uint32_t i = 0; i < l.args.size(); ++i) {
            auto& g = l.args[i];
            if (g.scalar) ck(f_.set_arg(l.kernel, i, 4, &g.v), "zeKernelSetArgumentValue");
            else ck(f_.set_arg(l.kernel, i, sizeof(void*), &g.ptr), "zeKernelSetArgumentValue");
        }
        ze_group_count_t gc{(l.global + 63) / 64, 1, 1};
        ck(f_.launch(list_, l.kernel, &gc, nullptr, 0, nullptr), "zeCommandListAppendLaunchKernel");
        ck(f_.barrier(list_, nullptr, 0, nullptr), "zeCommandListAppendBarrier");
    }
    ze_kernel_handle_t kernel(const std::string& n) {
        auto it = kern_.find(n); if (it != kern_.end()) return it->second;
        ze_kernel_desc_t d{}; d.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC; d.pKernelName = n.c_str();
        ze_kernel_handle_t k = nullptr; ck(f_.kernel_create(mod_, &d, &k), ("zeKernelCreate " + n).c_str());
        ck(f_.group_size(k, 64, 1, 1), "zeKernelSetGroupSize");
        return kern_[n] = k;
    }
    template<class F> void each_overlap(const float* p, std::size_t n, F&& f) {
        auto lo = reinterpret_cast<std::uintptr_t>(p), hi = lo + std::max<std::size_t>(n, 1) * 4;
        for (auto& [k, e] : res_) { auto el = k, eh = k + e.count * 4; if (el < hi && lo < eh) f(e); }
    }
    void download(entry& e) { copy(e.host, e.d, e.count * 4); e.host_valid = true; ++stats_.downloads; }
    void drop(entry& e) { dfree(e.d); stats_.resident_bytes -= e.count * 4; }
    bound acquire(float* h, std::size_t count, bool is_out) {
        auto key = reinterpret_cast<std::uintptr_t>(h);
        if (auto it = res_.find(key); it != res_.end() && it->second.count == count) { it->second.last = ++clock_; return {&it->second, it->second.d}; }
        auto it = res_.upper_bound(key);
        if (it != res_.begin()) {
            auto& c = std::prev(it)->second; auto cl = std::prev(it)->first;
            if (cl + c.count * 4 >= key + count * 4 && count) { c.last = ++clock_; return {&c, static_cast<char*>(c.d) + (key - cl)}; }
        }
        std::vector<std::uintptr_t> ks;
        each_overlap(h, count, [&](entry& e) { if (!e.host_valid) download(e); ks.push_back(reinterpret_cast<std::uintptr_t>(e.host)); });
        if (!ks.empty()) sync();
        for (auto k : ks) { drop(res_[k]); res_.erase(k); }
        entry e{h, count, alloc(count * 4, "resident")}; e.last = ++clock_;
        for (auto& [p, n] : pinned_) if (p == h && n == count) e.pinned = true;
        if (!is_out) { copy(e.d, h, count * 4); ++stats_.uploads; }
        stats_.resident_bytes += count * 4; stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, stats_.resident_bytes);
        auto& r = res_.emplace(key, e).first->second; return {&r, r.d};
    }
    std::string load_lib() {
        for (auto p : {"libze_loader.so.1", "libze_loader.so", "ze_loader.dll"}) if ((lib_ = dlopen(p, RTLD_NOW | RTLD_LOCAL))) break;
        if (!lib_) return "libze_loader not found (no Level Zero runtime)";
        bool ok = sym(f_.init, "zeInit") && sym(f_.drivers, "zeDriverGet") && sym(f_.devices, "zeDeviceGet") &&
                  sym(f_.props, "zeDeviceGetProperties") && sym(f_.mem_props, "zeDeviceGetMemoryProperties") &&
                  sym(f_.ctx_create, "zeContextCreate") && sym(f_.ctx_destroy, "zeContextDestroy") &&
                  sym(f_.list_create, "zeCommandListCreateImmediate") && sym(f_.list_destroy, "zeCommandListDestroy") &&
                  sym(f_.list_sync, "zeCommandListHostSynchronize") && sym(f_.memcpy_, "zeCommandListAppendMemoryCopy") &&
                  sym(f_.barrier, "zeCommandListAppendBarrier") && sym(f_.launch, "zeCommandListAppendLaunchKernel") &&
                  sym(f_.alloc, "zeMemAllocDevice") && sym(f_.free_, "zeMemFree") && sym(f_.module_create, "zeModuleCreate") &&
                  sym(f_.module_destroy, "zeModuleDestroy") && sym(f_.log_get, "zeModuleBuildLogGetString") &&
                  sym(f_.log_destroy, "zeModuleBuildLogDestroy") && sym(f_.kernel_create, "zeKernelCreate") &&
                  sym(f_.kernel_destroy, "zeKernelDestroy") && sym(f_.group_size, "zeKernelSetGroupSize") &&
                  sym(f_.set_arg, "zeKernelSetArgumentValue");
        sym(f_.native_bin, "zeModuleGetNativeBinary"); sym(f_.drv_props, "zeDriverGetProperties");
        return ok ? std::string{} : "libze_loader is missing entry points";
    }
    std::vector<std::pair<ze_driver_handle_t, ze_device_handle_t>> all_devices() {
        std::uint32_t nd = 0; if (f_.drivers(&nd, nullptr) != ZE_RESULT_SUCCESS || !nd) return {};
        std::vector<ze_driver_handle_t> ds(nd); f_.drivers(&nd, ds.data());
        std::vector<std::pair<ze_driver_handle_t, ze_device_handle_t>> r;
        for (auto d : ds) {
            std::uint32_t n = 0; if (f_.devices(d, &n, nullptr) != ZE_RESULT_SUCCESS || !n) continue;
            std::vector<ze_device_handle_t> v(n); f_.devices(d, &n, v.data());
            for (auto x : v) r.push_back({d, x});
        }
        return r;
    }
    std::string init() {
        // FORGE_ZE_TEST_IMAGE=opencl-c hands the OpenCL C source to zeModuleCreate instead (only the CPU mock accepts it).
        const char* tst = std::getenv("FORGE_ZE_TEST_IMAGE");
        bool src = tst && std::string(tst) == "opencl-c";
        const void* img = src ? static_cast<const void*>(cuda_hip::images::opencl_c) : cuda_hip::images::spirv_l0;
        std::size_t img_size = src ? cuda_hip::images::opencl_c_size : cuda_hip::images::spirv_l0_size;
        if (!img_size) return "this build has no SPIR-V kernel image (llvm-spirv was not found at build time)";
        if (auto e = load_lib(); !e.empty()) return e;
        if (f_.init(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS) return "zeInit failed";
        auto ds = all_devices();
        if (ds.empty()) return "no devices";
        if (ordinal_ < 0 || ordinal_ >= int(ds.size())) return "device ordinal " + std::to_string(ordinal_) + " out of range (" + std::to_string(ds.size()) + " devices)";
        auto [drv, dev] = ds[std::size_t(ordinal_)]; dev_ = dev;
        ze_device_properties_t p{}; p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        if (f_.props(dev_, &p) == ZE_RESULT_SUCCESS) { name_ = p.name; integrated_ = (p.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0; cpu_type_ = p.type == ZE_DEVICE_TYPE_CPU; }
        std::uint32_t nm = 1; ze_device_memory_properties_t mp{}; mp.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
        if (f_.mem_props(dev_, &nm, &mp) == ZE_RESULT_SUCCESS && nm) total_ = mp.totalSize;
        ze_context_desc_t cd{}; cd.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
        if (f_.ctx_create(drv, &cd, &ctx_) != ZE_RESULT_SUCCESS) return "zeContextCreate failed";
        ze_command_queue_desc_t qd{}; qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC; qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        if (f_.list_create(ctx_, dev_, &qd, &list_) != ZE_RESULT_SUCCESS) return "zeCommandListCreateImmediate failed";
        ze_module_desc_t md{}; md.stype = ZE_STRUCTURE_TYPE_MODULE_DESC; md.format = ZE_MODULE_FORMAT_IL_SPIRV;
        md.inputSize = img_size; md.pInputModule = static_cast<const std::uint8_t*>(img); md.pBuildFlags = "";
        // Native binary cache: keyed by device name, vendor/device id, device UUID, driver version, the image hash and
        // the build flags. A binary the driver refuses (or that lacks a kernel) falls back to compiling the image.
        std::string key;
        {
            char hx[64]; std::string uu; for (auto b : p.uuid.id) { std::snprintf(hx, sizeof hx, "%02x", unsigned(b)); uu += hx; }
            ze_driver_properties_t dp{}; dp.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
            std::uint32_t dver = f_.drv_props && f_.drv_props(drv, &dp) == ZE_RESULT_SUCCESS ? dp.driverVersion : 0;
            std::snprintf(hx, sizeof hx, "%08x:%08x\n%u\n%016llx", unsigned(p.vendorId), unsigned(p.deviceId), unsigned(dver),
                          static_cast<unsigned long long>(kernel_cache::fnv({static_cast<const char*>(img), img_size})));
            key = "level_zero\n" + name_ + "\n" + uu + "\n" + hx + "\n" + (src ? "opencl-c" : "spirv") + "\n" + md.pBuildFlags;
        }
        const bool cache = f_.native_bin && kernel_cache::enabled();
        if (cache) {
            cache_path_ = kernel_cache::path("ze-module", key);
            if (auto bin = kernel_cache::load("ze-module", key); !bin.empty()) {
                ze_module_desc_t nd = md; nd.format = ZE_MODULE_FORMAT_NATIVE; nd.inputSize = bin.size(); nd.pInputModule = bin.data();
                ze_module_handle_t m = nullptr;
                if (f_.module_create(ctx_, dev_, &nd, &m, nullptr) == ZE_RESULT_SUCCESS && m) {
                    ze_kernel_desc_t kd{}; kd.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC; kd.pKernelName = "forge_scale";   // the binary must hold our kernels
                    ze_kernel_handle_t k = nullptr;
                    if (f_.kernel_create(m, &kd, &k) == ZE_RESULT_SUCCESS && k) { f_.kernel_destroy(k); mod_ = m; from_cache_ = true; }
                    else f_.module_destroy(m);
                }
            }
        }
        ze_module_build_log_handle_t log = nullptr;
        if (from_cache_) {}
        else if (f_.module_create(ctx_, dev_, &md, &mod_, &log) != ZE_RESULT_SUCCESS) {
            std::string s; std::size_t n = 0;
            if (log && f_.log_get(log, &n, nullptr) == ZE_RESULT_SUCCESS && n) { s.resize(n); f_.log_get(log, &n, s.data()); }
            if (log) f_.log_destroy(log);
            mod_ = nullptr; return "zeModuleCreate failed: " + s;
        }
        if (log) f_.log_destroy(log);
        if (cache && !from_cache_) {
            std::size_t n = 0;
            if (f_.native_bin(mod_, &n, nullptr) == ZE_RESULT_SUCCESS && n) {
                std::vector<std::uint8_t> b(n);
                if (f_.native_bin(mod_, &n, b.data()) == ZE_RESULT_SUCCESS) kernel_cache::store("ze-module", key, b.data(), n);
            }
        }
        dm_.attach("level_zero:" + std::to_string(ordinal_), mu_, [this](std::uint64_t w, bool ext) { return evict_device(w, ext); });
        return {};
    }
    int ordinal_ = 0;
    void* lib_ = nullptr;
    ze_device_handle_t dev_ = nullptr; ze_context_handle_t ctx_ = nullptr; ze_command_list_handle_t list_ = nullptr; ze_module_handle_t mod_ = nullptr;
    fns f_;
    std::string name_, cache_path_;
    bool from_cache_ = false, integrated_ = false, cpu_type_ = false;
    std::size_t total_ = 0;
    std::mutex mu_;
    bool in_run_ = false, capturing_ = false;
    std::map<std::uintptr_t, entry> res_;
    std::vector<std::pair<const float*, std::size_t>> pinned_;
    std::vector<launch> cur_;
    std::vector<std::vector<launch>> graphs_;
    std::vector<void*> keep_;
    std::map<std::string, ze_kernel_handle_t> kern_;
    std::set<std::string> missing_;
    std::map<std::pair<std::uintptr_t, std::size_t>, qentry> qres_;
    std::map<std::string, std::size_t> dispatches_;
    residency_stats stats_;
};

} // namespace forge::gpu::ze
