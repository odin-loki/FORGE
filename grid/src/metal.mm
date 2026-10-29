// Metal runtime implementation (see grid/metal.hpp). Compiled with -fobjc-arc on Apple platforms only.
// HARDWARE-UNTESTED.
#define FORGE_GRID_METAL 1
#include <forge/grid/metal.hpp>
#import <Metal/Metal.h>
#include <mutex>
#include <unistd.h>

namespace forge::grid::metal {

struct runtime::impl {
    id<MTLDevice> dev; id<MTLCommandQueue> q; id<MTLLibrary> lib;
    std::map<std::string, id<MTLComputePipelineState>> pso;
    std::mutex mu;
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
        NSString* src = [[NSString alloc] initWithBytes:gpu::images::metal_src length:gpu::images::metal_src_size encoding:NSUTF8StringEncoding];
        MTLCompileOptions* o = [MTLCompileOptions new]; o.fastMathEnabled = NO;
        NSError* err = nil;
        r->p_->lib = [r->p_->dev newLibraryWithSource:src options:o error:&err];
        if (!r->p_->lib) return fail(std::string("MSL compile failed: ") + (err ? err.localizedDescription.UTF8String : "?"));
        return r;
    }
}

runtime::~runtime() { delete p_; }

bool runtime::run_op(const std::string& op, const std::vector<tensor_view>& in, const spec::attrs& a, tensor_view out) {
    std::vector<shape_t> ss; for (auto& v : in) ss.push_back(v.shape);
    if (!gpu::supports(op, ss, a)) return false;
    auto L = gpu::kernel_args(op, in, a, out);
    std::lock_guard lk(p_->mu);
    @autoreleasepool {
        ++dispatches_[op];
        auto& pso = p_->pso["forge_" + op];
        if (!pso) {
            id<MTLFunction> f = [p_->lib newFunctionWithName:[NSString stringWithUTF8String:("forge_" + op).c_str()]];
            NSError* err = nil; pso = [p_->dev newComputePipelineStateWithFunction:f error:&err];
            if (!pso) throw std::runtime_error("metal: pipeline for " + op + " failed");
        }
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
            if (L.s == gpu::sig::rope && i == 1) bind(L.rope_inv.data(), L.rope_inv.size() * 4);
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

} // namespace forge::grid::metal
