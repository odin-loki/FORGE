// Metal runtime implementation (see gpu/metal.hpp). Compiled with -fobjc-arc on Apple platforms only.
// HARDWARE-UNTESTED. The kernel-call builder for gpu::portable_kernels<> (k_* / qweight below) has never been
// compiled by an Apple toolchain or executed: on Linux it is only parsed against tests/shims/metal_stub
// (ctest gpu_compile_metal_host_syntax).
#define FORGE_GPU_METAL 1
#include <forge/gpu/metal.hpp>
#include <forge/core/memory.hpp>
#import <Metal/Metal.h>
#include <mutex>
#include <unistd.h>

namespace forge::gpu::metal {

struct runtime::impl {
    id<MTLDevice> dev; id<MTLCommandQueue> q; id<MTLLibrary> lib;
    std::map<std::string, id<MTLComputePipelineState>> pso;
    // portable kernels: scratch buffers of the call in progress by handle, encoded weights by host address and size
    struct held_t { id<MTLBuffer> buf; mem::lease charge; };
    std::map<void*, held_t> held;
    struct qent { id<MTLBuffer> buf; std::size_t bytes = 0; std::uint64_t last = 0; mem::lease charge; };
    std::map<std::pair<std::uintptr_t, std::size_t>, qent> qres;
    std::map<std::string, bool> has;
    id<MTLComputePipelineState> pipeline(const std::string& fn) {
        auto& p = pso[fn];
        if (!p) {
            id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]];
            NSError* err = nil; p = [dev newComputePipelineStateWithFunction:f error:&err];
            if (!p) throw std::runtime_error("metal: pipeline for " + fn + " failed");
        }
        return p;
    }
};

static NSArray<id<MTLDevice>>* devices() {
#if TARGET_OS_OSX
    return MTLCopyAllDevices();
#else
    id<MTLDevice> d = MTLCreateSystemDefaultDevice(); return d ? @[d] : @[];
#endif
}

int runtime::device_count() { @autoreleasepool { return int(devices().count); } }

std::unique_ptr<runtime> runtime::open(std::string* why, int ordinal) {
    @autoreleasepool {
        auto ds = devices();
        auto fail = [&](std::string e) { if (why) *why = "metal: " + e; return std::unique_ptr<runtime>(); };
        if (!ds.count) return fail("no Metal device");
        if (ordinal < 0 || ordinal >= int(ds.count)) return fail("device ordinal " + std::to_string(ordinal) + " out of range");
        std::unique_ptr<runtime> r(new runtime); r->ordinal_ = ordinal; r->p_ = new impl;
        r->p_->dev = ds[NSUInteger(ordinal)]; r->name_ = r->p_->dev.name.UTF8String;
        r->p_->q = [r->p_->dev newCommandQueue];
        NSString* src = [[NSString alloc] initWithBytes:cuda_hip::images::metal_src length:cuda_hip::images::metal_src_size encoding:NSUTF8StringEncoding];
        MTLCompileOptions* o = [MTLCompileOptions new]; o.fastMathEnabled = NO;
        NSError* err = nil;
        r->p_->lib = [r->p_->dev newLibraryWithSource:src options:o error:&err];
        if (!r->p_->lib) return fail(std::string("MSL compile failed: ") + (err ? err.localizedDescription.UTF8String : "?"));
        return r;
    }
}

runtime::~runtime() { delete p_; }

bool runtime::run_op(const std::string& op, const std::vector<tensor_view>& in, const ops::attrs& a, tensor_view out) {
    std::vector<shape_t> ss; for (auto& v : in) ss.push_back(v.shape);
    if (!cuda_hip::supports(op, ss, a)) return false;
    if (op == "attention") if (int r = attention_port(in, a, out); r >= 0) return true;   // one-pass / split-over-keys kernels (plain f32 K/V)
    auto L = cuda_hip::kernel_args(op, in, a, out);
    std::lock_guard lk(mu_);
    // Device memory accounting (docs/memory.md): the op's buffers are charged to the pool "metal:N" while it runs.
    // Unified memory and no residency, so there is nothing to evict: an op that does not fit fails.
    std::size_t total = std::max<std::size_t>(4, out.size() * 4) + (L.s == cuda_hip::sig::rope ? std::max<std::size_t>(4, L.rope_inv.size() * 4) : 0);
    for (auto& v : in) total += std::max<std::size_t>(4, v.size() * 4);
    auto& pool = mem::governor::instance().pool("metal:" + std::to_string(ordinal_));
    mem::lease charge = pool.try_reserve(total, "scratch");
    if (!charge) raise(errc::out_of_memory, "{}: cannot allocate {} for {} ({} of {} in use; raise FORGE_MEM_DEVICE_LIMIT)",
                       pool.name(), mem::format_size(total), op, mem::format_size(pool.used()), mem::format_size(pool.limit()));
    @autoreleasepool {
        ++dispatches_[op];
        id<MTLComputePipelineState> pso = p_->pipeline("forge_" + op);
        id<MTLCommandBuffer> cb = [p_->q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        NSUInteger idx = 0;
        auto bind = [&](const void* h, std::size_t bytes) {
            id<MTLBuffer> b = [p_->dev newBufferWithBytes:h length:std::max<std::size_t>(4, bytes) options:MTLResourceStorageModeShared];
            [enc setBuffer:b offset:0 atIndex:idx++]; return b;
        };
        for (std::size_t i = 0; i < in.size(); ++i) {
            bind(in[i].data, in[i].size() * 4);
            if (L.s == cuda_hip::sig::rope && i == 1) bind(L.rope_inv.data(), L.rope_inv.size() * 4);
        }
        id<MTLBuffer> ob = [p_->dev newBufferWithLength:std::max<std::size_t>(4, out.size() * 4) options:MTLResourceStorageModeShared];
        [enc setBuffer:ob offset:0 atIndex:idx++];
        for (auto& s : L.scalars) { if (s.is_float) [enc setBytes:&s.f length:4 atIndex:idx++]; else [enc setBytes:&s.u length:4 atIndex:idx++]; }
        NSUInteger tg = std::min<NSUInteger>(64, pso.maxTotalThreadsPerThreadgroup);
        if (L.threads) [enc dispatchThreads:MTLSizeMake(L.threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) throw std::runtime_error("metal: command buffer failed");
        std::memcpy(out.data, ob.contents, out.size() * 4);
    }
    return true;
}

// ---- kernel-call builder for gpu::portable_kernels<> (mu_ held by the caller) ----
bool runtime::k_has(const std::string& n) {
    if (auto it = p_->has.find(n); it != p_->has.end()) return it->second;
    @autoreleasepool {
        id<MTLFunction> f = [p_->lib newFunctionWithName:[NSString stringWithUTF8String:n.c_str()]];
        const bool ok = f != nil;
        p_->has[n] = ok;
        return ok;
    }
}

// An encoded weight (or the f32 expert weights of the mixture of experts) copied once into a shared buffer and kept,
// keyed by host address and size and charged to the pool "metal:N" as "weights". When the pool is full, the weights
// the op in progress has not bound are dropped (the host copy is the source of truth) and the reservation is retried.
void* runtime::qweight(const std::uint8_t* q, std::size_t bytes) {
    auto key = std::make_pair(reinterpret_cast<std::uintptr_t>(q), bytes);
    if (auto it = p_->qres.find(key); it != p_->qres.end()) { it->second.last = ++clock_; return (__bridge void*)it->second.buf; }
    auto& pool = mem::governor::instance().pool("metal:" + std::to_string(ordinal_));
    const std::size_t len = std::max<std::size_t>(4, bytes);
    mem::lease charge;
    try { charge = pool.try_reserve(len, "weights"); } catch (...) {}
    if (!charge) {
        for (auto it = p_->qres.begin(); it != p_->qres.end();) { if (it->second.last <= op_mark_) it = p_->qres.erase(it); else ++it; }
        charge = pool.try_reserve(len, "weights");
        if (!charge) raise(errc::out_of_memory, "{}: cannot allocate {} for an encoded weight ({} of {} in use; raise FORGE_MEM_DEVICE_LIMIT)",
                           pool.name(), mem::format_size(len), mem::format_size(pool.used()), mem::format_size(pool.limit()));
    }
    @autoreleasepool {
        id<MTLBuffer> b = [p_->dev newBufferWithLength:len options:MTLResourceStorageModeShared];
        if (!b) raise(errc::out_of_memory, "metal: no buffer of {} for an encoded weight", mem::format_size(len));
        std::memcpy(b.contents, q, bytes);
        impl::qent e; e.buf = b; e.bytes = bytes; e.last = ++clock_; e.charge = std::move(charge);
        auto it = p_->qres.emplace(key, std::move(e)).first;
        return (__bridge void*)it->second.buf;
    }
}

void* runtime::k_scratch(kcall& c, std::size_t bytes) {
    const std::size_t len = std::max<std::size_t>(4, bytes);
    auto& pool = mem::governor::instance().pool("metal:" + std::to_string(ordinal_));
    mem::lease charge = pool.try_reserve(len, "scratch");
    if (!charge) raise(errc::out_of_memory, "{}: cannot allocate {} of scratch ({} of {} in use; raise FORGE_MEM_DEVICE_LIMIT)",
                       pool.name(), mem::format_size(len), mem::format_size(pool.used()), mem::format_size(pool.limit()));
    @autoreleasepool {
        id<MTLBuffer> b = [p_->dev newBufferWithLength:len options:MTLResourceStorageModeShared];
        if (!b) raise(errc::out_of_memory, "metal: no scratch buffer of {}", mem::format_size(len));
        void* h = (__bridge void*)b;
        impl::held_t e; e.buf = b; e.charge = std::move(charge);
        p_->held.insert_or_assign(h, std::move(e));
        c.temps.push_back(h);
        return h;
    }
}

void runtime::k_upload(void* m, const void* h, std::size_t bytes) {
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)m;
    std::memcpy(b.contents, h, bytes);
}

// Runs the kernel to completion: host tensors are wrapped in shared buffers for this launch (inputs and in/out
// tensors copied in), then outputs and in/out tensors are copied back. Arguments are bound in call order, buffers and
// scalars alike, which is the [[buffer(i)]] order of the generated signature.
void runtime::k_launch(kcall& c, const std::string& name, std::size_t global) {
    if (global) {
        std::size_t total = 4;
        for (auto& a : c.args) if (a.kind == 0) total += std::max<std::size_t>(4, a.cnt * 4);
        auto& pool = mem::governor::instance().pool("metal:" + std::to_string(ordinal_));
        mem::lease charge = pool.try_reserve(total, "scratch");
        if (!charge) raise(errc::out_of_memory, "{}: cannot allocate {} for {} ({} of {} in use; raise FORGE_MEM_DEVICE_LIMIT)",
                           pool.name(), mem::format_size(total), name, mem::format_size(pool.used()), mem::format_size(pool.limit()));
        @autoreleasepool {
            id<MTLComputePipelineState> pso = p_->pipeline(name);
            id<MTLCommandBuffer> cb = [p_->q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            NSUInteger idx = 0;
            struct back_t { id<MTLBuffer> buf; float* h; std::size_t bytes; };
            std::vector<back_t> backs;
            for (auto& a : c.args) {
                if (a.kind == 0) {
                    const std::size_t bytes = a.cnt * 4, len = std::max<std::size_t>(4, bytes);
                    id<MTLBuffer> b = [p_->dev newBufferWithLength:len options:MTLResourceStorageModeShared];
                    if (!b) raise(errc::out_of_memory, "metal: no buffer of {} for {}", mem::format_size(len), name);
                    if (a.mode != 1 && bytes) std::memcpy(b.contents, a.h, bytes);
                    [enc setBuffer:b offset:0 atIndex:idx++];
                    if (a.mode != 0) { back_t k; k.buf = b; k.h = const_cast<float*>(a.h); k.bytes = bytes; backs.push_back(k); }
                } else if (a.kind == 1) {
                    id<MTLBuffer> b = (__bridge id<MTLBuffer>)a.dev;
                    [enc setBuffer:b offset:0 atIndex:idx++];
                } else if (a.kind == 2) {
                    [enc setBytes:&a.u length:4 atIndex:idx++];
                } else {
                    [enc setBytes:&a.f length:4 atIndex:idx++];
                }
            }
            NSUInteger tg = std::min<NSUInteger>(64, pso.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(global, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) throw std::runtime_error("metal: command buffer failed (" + name + ")");
            for (auto& k : backs) if (k.bytes) std::memcpy(k.h, k.buf.contents, k.bytes);
        }
    }
    c.args.clear();
}

void runtime::k_done(kcall& c) {
    for (auto h : c.temps) p_->held.erase(h);
    c.temps.clear();
}

} // namespace forge::gpu::metal
