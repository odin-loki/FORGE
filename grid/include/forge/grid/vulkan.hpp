#pragma once
// grid Vulkan backend: loads libvulkan at run time (no link dependency), runs the compute kernels that
// spec/gen/gpu_gen.py generates from spec/ops (embedded as SPIR-V). Tested on Mesa lavapipe in CI.
// Available when <vulkan/vulkan.h> and the generated SPIR-V header are both present at build time.
#if __has_include(<vulkan/vulkan.h>) && __has_include(<forge/grid/spirv.gen.hpp>) && (defined(__linux__) || defined(__APPLE__))
#define FORGE_GRID_VULKAN 1
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <forge/grid/spirv.gen.hpp>
#include <forge/core/tensor.hpp>
#include <dlfcn.h>
#include <cstring>
#include <map>
#include <mutex>
#include <forge/spec/ops.gen.hpp>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace forge::grid::vk {

struct push { std::uint32_t n = 0, rows = 0, cols = 0, k = 0; float alpha = 0; std::uint32_t a = 0, b = 0, c = 0, d = 0; float beta = 0; };
struct hbuf { float* data; std::size_t count; };

class runtime {
public:
    // Returns nullptr (with reason) when no loader / no compute-capable device is present.
    static std::unique_ptr<runtime> open(std::string* why = nullptr) {
        auto r = std::unique_ptr<runtime>(new runtime);
        std::string e = r->init();
        if (!e.empty()) { if (why) *why = e; return nullptr; }
        return r;
    }
    ~runtime() {
        if (dev_) {
            vkDeviceWaitIdle(dev_);
            for (auto& [n, p] : pipes_) { vkDestroyPipeline(dev_, p.pipe, nullptr); vkDestroyPipelineLayout(dev_, p.layout, nullptr);
                                          vkDestroyDescriptorSetLayout(dev_, p.dsl, nullptr); vkDestroyShaderModule(dev_, p.mod, nullptr); }
            if (pool_) vkDestroyCommandPool(dev_, pool_, nullptr);
            vkDestroyDevice(dev_, nullptr);
        }
        if (inst_) vkDestroyInstance(inst_, nullptr);
        if (lib_) dlclose(lib_);
    }
    const std::string& device_name() const { return name_; }
    std::map<std::string, std::size_t> dispatches() { std::lock_guard lk(mu_); return dispatches_; }

    static bool has_kernel(const std::string& n) { for (auto& b : spv::table) if (n == b.name) return true; return false; }

    // Run kernel `name` over the given host buffers (bound 0..N-1 in order; the last is written back).
    void run(const std::string& name, const std::vector<hbuf>& bufs, const push& pc, std::uint32_t invocations) {
        std::lock_guard lk(mu_);
        ++dispatches_[name];
        auto& p = pipeline(name, std::uint32_t(bufs.size()));
        struct dbuf { VkBuffer b; VkDeviceMemory m; VkDeviceSize sz; };
        std::vector<dbuf> d;
        for (auto& h : bufs) {
            VkDeviceSize sz = std::max<VkDeviceSize>(4, h.count * 4);
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = sz; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            VkBuffer b; ck(vkCreateBuffer(dev_, &bi, nullptr, &b), "vkCreateBuffer");
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev_, b, &mr);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = memtype(mr.memoryTypeBits);
            VkDeviceMemory m; ck(vkAllocateMemory(dev_, &ai, nullptr, &m), "vkAllocateMemory");
            ck(vkBindBufferMemory(dev_, b, m, 0), "vkBindBufferMemory");
            void* ptr; ck(vkMapMemory(dev_, m, 0, sz, 0, &ptr), "vkMapMemory");
            std::memcpy(ptr, h.data, h.count * 4); vkUnmapMemory(dev_, m);
            d.push_back({b, m, sz});
        }
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, std::uint32_t(bufs.size())};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pi.maxSets = 1; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
        VkDescriptorPool dp; ck(vkCreateDescriptorPool(dev_, &pi, nullptr, &dp), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo sa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; sa.descriptorPool = dp; sa.descriptorSetCount = 1; sa.pSetLayouts = &p.dsl;
        VkDescriptorSet ds; ck(vkAllocateDescriptorSets(dev_, &sa, &ds), "vkAllocateDescriptorSets");
        std::vector<VkDescriptorBufferInfo> dbi; std::vector<VkWriteDescriptorSet> w;
        dbi.reserve(d.size());
        for (std::size_t i = 0; i < d.size(); ++i) {
            dbi.push_back({d[i].b, 0, VK_WHOLE_SIZE});
            VkWriteDescriptorSet x{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; x.dstSet = ds; x.dstBinding = std::uint32_t(i); x.descriptorCount = 1;
            x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; x.pBufferInfo = &dbi.back(); w.push_back(x);
        }
        vkUpdateDescriptorSets(dev_, std::uint32_t(w.size()), w.data(), 0, nullptr);
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ca.commandPool = pool_; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
        VkCommandBuffer cb; ck(vkAllocateCommandBuffers(dev_, &ca, &cb), "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo bb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        ck(vkBeginCommandBuffer(cb, &bb), "vkBeginCommandBuffer");
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &pc);
        vkCmdDispatch(cb, (invocations + 63) / 64, 1, 1);
        ck(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f; ck(vkCreateFence(dev_, &fi, nullptr, &f), "vkCreateFence");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        ck(vkQueueSubmit(q_, 1, &si, f), "vkQueueSubmit");
        ck(vkWaitForFences(dev_, 1, &f, VK_TRUE, ~0ull), "vkWaitForFences");
        auto& out = d.back(); void* ptr; ck(vkMapMemory(dev_, out.m, 0, out.sz, 0, &ptr), "vkMapMemory");
        std::memcpy(bufs.back().data, ptr, bufs.back().count * 4); vkUnmapMemory(dev_, out.m);
        vkDestroyFence(dev_, f, nullptr); vkFreeCommandBuffers(dev_, pool_, 1, &cb); vkDestroyDescriptorPool(dev_, dp, nullptr);
        for (auto& x : d) { vkDestroyBuffer(dev_, x.b, nullptr); vkFreeMemory(dev_, x.m, nullptr); }
    }

private:
    runtime() = default;
    struct pipe { VkShaderModule mod{}; VkDescriptorSetLayout dsl{}; VkPipelineLayout layout{}; VkPipeline pipe{}; };
    static void ck(VkResult r, const char* what) { if (r != VK_SUCCESS) throw std::runtime_error(std::string("vulkan: ") + what + " failed (" + std::to_string(int(r)) + ")"); }

#define FORGE_VK_FNS(X) X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr)
#define FORGE_VK_DEV_FNS(X) X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkDeviceWaitIdle) X(vkCreateCommandPool) X(vkDestroyCommandPool) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) \
    X(vkMapMemory) X(vkUnmapMemory) X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateComputePipelines) X(vkDestroyPipeline) \
    X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkAllocateCommandBuffers) \
    X(vkFreeCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) \
    X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCreateFence) X(vkDestroyFence) X(vkQueueSubmit) X(vkWaitForFences)
#define FORGE_VK_DECL(f) PFN_##f f = nullptr;
    FORGE_VK_FNS(FORGE_VK_DECL)
    FORGE_VK_DEV_FNS(FORGE_VK_DECL)
#undef FORGE_VK_DECL

    std::string init() {
        for (const char* so : {"libvulkan.so.1", "libvulkan.so", "libvulkan.1.dylib", "libMoltenVK.dylib"})
            if ((lib_ = dlopen(so, RTLD_NOW | RTLD_LOCAL))) break;
        if (!lib_) return "vulkan loader not found";
        auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib_, "vkGetInstanceProcAddr"));
        if (!gipa) return "vkGetInstanceProcAddr missing";
        auto create = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.pApplicationName = "forge"; app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
        if (!create || create(&ici, nullptr, &inst_) != VK_SUCCESS) { inst_ = nullptr; return "vkCreateInstance failed"; }
#define FORGE_VK_LOAD(f) f = reinterpret_cast<PFN_##f>(gipa(inst_, #f)); if (!f) return "missing " #f;
        FORGE_VK_FNS(FORGE_VK_LOAD)
#undef FORGE_VK_LOAD
        std::uint32_t n = 0; vkEnumeratePhysicalDevices(inst_, &n, nullptr);
        if (!n) return "no vulkan physical devices";
        std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst_, &n, pds.data());
        int best = -1; std::uint32_t fam = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            std::uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &qn, nullptr);
            std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(pds[i], &qn, qf.data());
            for (std::uint32_t j = 0; j < qn; ++j) if (qf[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(pds[i], &pr);
                bool discrete = pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
                if (best < 0 || discrete) { best = int(i); fam = j; name_ = pr.deviceName; }
                break;
            }
        }
        if (best < 0) return "no compute queue";
        phys_ = pds[std::size_t(best)];
        float prio = 1.f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = fam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
        if (vkCreateDevice(phys_, &dci, nullptr, &dev_) != VK_SUCCESS) { dev_ = nullptr; return "vkCreateDevice failed"; }
#define FORGE_VK_LOAD(f) f = reinterpret_cast<PFN_##f>(vkGetDeviceProcAddr(dev_, #f)); if (!f) f = reinterpret_cast<PFN_##f>(gipa(inst_, #f)); if (!f) return "missing " #f;
        FORGE_VK_DEV_FNS(FORGE_VK_LOAD)
#undef FORGE_VK_LOAD
        vkGetDeviceQueue(dev_, fam, 0, &q_);
        VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cp.queueFamilyIndex = fam; cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(dev_, &cp, nullptr, &pool_) != VK_SUCCESS) return "vkCreateCommandPool failed";
        return {};
    }
    std::uint32_t memtype(std::uint32_t bits) {
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
        auto want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        throw std::runtime_error("vulkan: no host-visible coherent memory");
    }
    pipe& pipeline(const std::string& name, std::uint32_t nbuf) {
        auto it = pipes_.find(name); if (it != pipes_.end()) return it->second;
        const spv::blob* b = nullptr; for (auto& x : spv::table) if (name == x.name) b = &x;
        if (!b) throw std::runtime_error("vulkan: no kernel '" + name + "'");
        pipe p;
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; mi.codeSize = b->count * 4; mi.pCode = b->words;
        ck(vkCreateShaderModule(dev_, &mi, nullptr, &p.mod), "vkCreateShaderModule");
        std::vector<VkDescriptorSetLayoutBinding> lb(nbuf);
        for (std::uint32_t i = 0; i < nbuf; ++i) lb[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; li.bindingCount = nbuf; li.pBindings = lb.data();
        ck(vkCreateDescriptorSetLayout(dev_, &li, nullptr, &p.dsl), "vkCreateDescriptorSetLayout");
        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pl.setLayoutCount = 1; pl.pSetLayouts = &p.dsl; pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pr;
        ck(vkCreatePipelineLayout(dev_, &pl, nullptr, &p.layout), "vkCreatePipelineLayout");
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, p.mod, "main", nullptr};
        ci.layout = p.layout;
        ck(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipe), "vkCreateComputePipelines");
        return pipes_.emplace(name, p).first->second;
    }

    void* lib_ = nullptr;
    VkInstance inst_ = nullptr; VkPhysicalDevice phys_ = nullptr; VkDevice dev_ = nullptr; VkQueue q_ = nullptr; VkCommandPool pool_{};
    std::string name_;
    std::mutex mu_;
    std::map<std::string, std::size_t> dispatches_;
    std::map<std::string, pipe> pipes_;
};

// Run one spec op on the device over host views (out.shape must already be the op's output shape).
// Elementwise ops need same-shape inputs; softmax/rms_norm act on the last axis; matmul_t is a[..., m, k] @ b[n, k]^T.
// Returns false if this op/shape/attrs combination has no Vulkan lowering (the caller falls back to the CPU).
inline bool supports(const std::string& op, const std::vector<shape_t>& in, const spec::attrs& a) {
    if (!runtime::has_kernel(op) || in.empty()) return false;
    auto& x = in[0];
    if (op == "softmax") return !x.empty() && (a.num("axis", -1) == -1 || a.num("axis", -1) == double(x.size()) - 1);
    if (op == "rms_norm") return in.size() == 2 && !x.empty() && in[1].size() == 1 && in[1][0] == x.back();
    if (op == "matmul_t") return in.size() == 2 && x.size() >= 2 && in[1].size() == 2 && in[1][1] == x.back();
    if (op == "rope") return in.size() == 2 && x.size() == 3 && x[2] % 2 == 0 && numel(in[1]) == x[0];
    if (op == "gather") return in.size() == 2 && x.size() == 2;
    if (op == "slice") return in.size() == 1 && !x.empty() && a.num("step", 1) == 1;
    if (op == "attention") return in.size() == 3 && x.size() == 3 && in[1].size() == 3 && in[2].size() == 3 && in[1][1] > 0 && x[1] % in[1][1] == 0;
    for (auto& t : in) if (t != x) return false;
    return true;
}
inline bool run_op(runtime& rt, const std::string& op, const std::vector<tensor_view>& in, const spec::attrs& a, tensor_view out) {
    std::vector<shape_t> ss; for (auto& v : in) ss.push_back(v.shape);
    if (!supports(op, ss, a)) return false;
    auto& x = in[0];
    push pc; pc.n = std::uint32_t(out.size());
    std::uint32_t inv = pc.n;
    if (op == "scale") pc.alpha = float(a.num("alpha", 1.0));
    else if (op == "softmax" || op == "rms_norm") {
        pc.cols = std::uint32_t(x.shape.back()); pc.rows = pc.cols ? std::uint32_t(x.size()) / pc.cols : 0; inv = pc.rows;
        if (op == "rms_norm") pc.alpha = float(a.num("eps", 1e-6));
    } else if (op == "matmul_t") {
        pc.k = std::uint32_t(x.shape.back()); pc.rows = std::uint32_t(x.size()) / pc.k; pc.cols = std::uint32_t(in[1].shape[0]); inv = pc.rows * pc.cols;
    } else if (op == "rope") {
        pc.rows = std::uint32_t(x.shape[0]); pc.cols = std::uint32_t(x.shape[1]); pc.k = std::uint32_t(x.shape[2]);
        pc.alpha = float(a.num("base", 10000.0)); pc.a = a.num("neox", 1) != 0; inv = pc.rows * pc.cols * (pc.k / 2);
    } else if (op == "slice") {
        auto ax = std::size_t(spec::norm_axis(i64(a.num("axis", 0)), x.shape.size()));
        i64 inner = 1; for (std::size_t d = ax + 1; d < x.shape.size(); ++d) inner *= x.shape[d];
        i64 st = i64(a.num("start", 0)); if (st < 0) st += x.shape[ax];
        pc.cols = std::uint32_t(x.shape[ax]); pc.k = std::uint32_t(inner); pc.a = std::uint32_t(st); pc.b = std::uint32_t(out.shape[ax]);
    } else if (op == "gather") {
        pc.rows = std::uint32_t(x.shape[0]); pc.cols = std::uint32_t(x.shape[1]);
    } else if (op == "attention") {
        pc.rows = std::uint32_t(x.shape[0]); pc.cols = std::uint32_t(x.shape[1]); pc.k = std::uint32_t(x.shape[2]);
        pc.a = std::uint32_t(in[1].shape[0]); pc.b = std::uint32_t(in[1].shape[1]); pc.c = std::uint32_t(in[2].shape[2]);
        pc.d = (a.num("causal", 1) != 0 ? 1u : 0u) | (std::uint32_t(a.num("window", 0)) << 1);
        pc.alpha = float(a.num("scale", 1.0 / std::sqrt(double(pc.k)))); inv = pc.rows * pc.cols;
    }
    std::vector<hbuf> b; for (auto& v : in) b.push_back({v.data, v.size()}); b.push_back({out.data, out.size()});
    rt.run(op, b, pc, inv);
    return true;
}
inline bool run_op(runtime& rt, const std::string& op, const std::vector<forge::tensor>& in, const spec::attrs& a, forge::tensor& out) {
    auto o = spec::op_from_name(op);
    if (!o) return false;
    std::vector<shape_t> ss; std::vector<tensor_view> vs;
    for (auto& t : in) { ss.push_back(t.shape); vs.push_back({const_cast<float*>(t.data.data()), t.shape}); }
    if (!supports(op, ss, a)) return false;
    out = forge::tensor(spec::infer_shape(spec::info(*o).shape, ss, a));
    return run_op(rt, op, vs, a, out.view());
}

} // namespace forge::grid::vk
#endif
