#pragma once
#include <functional>
#include <optional>
#include <memory>
// Lowerings: implementations of spec ops for a device. Only lowerings that pass the oracle are eligible.
#include <forge/core/log.hpp>
#include <forge/flow/ir.hpp>
#include <forge/flow/kv_codec.hpp>
#include <forge/grid/prims.hpp>
#include <forge/lane/dispatch.hpp>
#include <forge/verify/oracle.hpp>
#include <memory>
#include <mutex>
#include <span>

namespace forge::flow {

using kernel_fn = std::function<void(std::span<const tensor_view>, tensor_view, const spec::attrs&)>;
using supports_fn = std::function<bool(std::span<const shape_t>, const shape_t&, const spec::attrs&)>;

struct lowering {
    std::string name;
    spec::op op;
    int priority;          // higher = more specific
    supports_fn supports;
    kernel_fn fn;
    int verified = -1;     // -1 unknown, 0 failed oracle, 1 passed
    bool on_device = false;
    double tol = 0;         // >0 replaces the spec tolerance with a relative bound (e.g. Vulkan sin/cos: spec allows 2^-11 error)
    // on_device: reads/writes device-resident buffers during an executor run (see device_hooks)
};

// A device backend's residency protocol, driven by the executor: buffers may live on the device for the
// duration of run(); the executor asks for host copies before CPU work and reports host writes after it.
struct device_hooks {
    std::function<void()> begin_run, end_run;
    // optional: end a run without copying device intermediates back (the executor has already fetched outputs/state)
    std::function<void()> end_run_discard;
    std::function<void(const float*, std::size_t)> pin, to_host, host_wrote, release;
    std::function<void(float* src, std::size_t n, float* dst_base, std::size_t dst_count, std::size_t off)> copy;  // device-side
    // fixed-size state op (state_ops.hpp: delta_rule, causal_conv) run on the device with the slot `S` kept resident;
    // false when the backend has no kernel for it (the executor then runs the host version).
    std::function<bool(const std::string& fn, std::span<const tensor_view> in, float* S, std::size_t sn, tensor_view out, const spec::attrs& a)> state;
};

inline kernel_fn reference_kernel(spec::op o) {
    return [o](std::span<const tensor_view> in, tensor_view out, const spec::attrs& a) {
        std::vector<tensor> ts;
        for (auto& v : in) { tensor t(v.shape); std::copy_n(v.data, t.size(), t.data.data()); ts.push_back(std::move(t)); }
        auto r = spec::reference(o)(ts, a);
        std::copy(r.data.begin(), r.data.end(), out.data);
    };
}

class registry {
public:
    // A fresh registry with the CPU lowerings (backends add theirs on top, at higher priority).
    static std::unique_ptr<registry> make() { std::unique_ptr<registry> p(new registry()); make_cpu(*p); return p; }
    static registry& cpu() { static registry* r = [] { auto* p = new registry(); make_cpu(*p); return p; }(); return *r; }
    void add(lowering l) { by_op_[std::size_t(l.op)].push_back(std::move(l)); }
    // Most specific verified lowering that supports the shapes; reference otherwise.
    const lowering& select(spec::op o, std::span<const shape_t> in, const shape_t& out, const spec::attrs& a) {
        auto& v = by_op_[std::size_t(o)];
        const lowering* best = nullptr;
        for (auto& l : v) {
            if (l.supports && !l.supports(in, out, a)) continue;
            if (!ensure_verified(l)) continue;
            if (!best || l.priority > best->priority) best = &l;
        }
        return best ? *best : ref_[std::size_t(o)];
    }
    std::vector<lowering>& lowerings(spec::op o) { return by_op_[std::size_t(o)]; }
    std::shared_ptr<device_hooks> hooks;  // set by device backends that keep data resident
    // Pipeline split ("pipe:a,b,..."): when non-empty, compile() cuts the graph into contiguous stages of about
    // equal cost (weights in stage_weight) and selects each node's lowering from its stage's registry
    // (null entry = CPU). Stage registries must be non-resident; activations cross stages through host memory.
    std::vector<std::unique_ptr<registry>> stages;
    // Resident device backends: matmul_t over an encoded weight (tensor_view::qdata) on the device, for these codecs.
    std::optional<lowering> device_qgemm;
    std::vector<std::string> device_qgemm_codecs;
    std::vector<double> stage_weight;
    bool ensure_verified(lowering& l) {
        std::lock_guard g(m_);
        if (l.verified >= 0) return l.verified == 1;
        l.verified = oracle_check(l) ? 1 : 0;
        if (!l.verified) log::warn("lowering {} failed the oracle and is disabled", l.name);
        return l.verified == 1;
    }
    static bool oracle_check(const lowering& l, std::string* msg = nullptr) {
        std::mt19937 rng(1234);
        for (int trial = 0; trial < 3; ++trial) {
            auto c = verify::make_case(l.op, rng, trial == 0 ? 37 : (trial == 1 ? 64 : 5));
            auto ref = spec::reference(l.op)(c.in, c.a);
            std::vector<shape_t> ss; for (auto& t : c.in) ss.push_back(t.shape);
            if (l.supports && !l.supports(ss, ref.shape, c.a)) continue;
            std::vector<tensor_view> vs; for (auto& t : c.in) vs.push_back(t.view());
            tensor out(ref.shape, NAN);
            l.fn(vs, out.view(), c.a);
            auto r = verify::compare(out, ref, l.tol > 0 ? spec::tolerance{spec::tol_kind::rel, l.tol} : spec::info(l.op).tol);
            if (!r.ok) { if (msg) *msg = r.message; return false; }
        }
        return true;
    }
private:
    registry() { for (std::size_t i = 0; i < spec::op_count; ++i) ref_[i] = {"reference", spec::op(i), 0, nullptr, reference_kernel(spec::op(i)), 1}; }
    static void make_cpu(registry& r);
    std::array<std::vector<lowering>, spec::op_count> by_op_;
    std::array<lowering, spec::op_count> ref_;
    std::mutex m_;
};

namespace cpu_kernels {
using sp = std::span<const shape_t>;
inline bool same_shapes(sp in, const shape_t& out, const spec::attrs&) {
    for (auto& s : in) if (s != out) return false;
    return true;
}
inline bool always(sp, const shape_t&, const spec::attrs&) { return true; }

template <auto member> kernel_fn unary() {
    return [](std::span<const tensor_view> in, tensor_view out, const spec::attrs&) {
        auto f = lane::active().*member;
        std::size_t n = out.size();
        task::parallel_for(n, 16384, [&](std::size_t b, std::size_t e) { f(in[0].data + b, out.data + b, e - b); });
    };
}
template <auto member> kernel_fn binary() {
    return [](std::span<const tensor_view> in, tensor_view out, const spec::attrs&) {
        auto f = lane::active().*member;
        std::size_t n = out.size();
        task::parallel_for(n, 16384, [&](std::size_t b, std::size_t e) { f(in[0].data + b, in[1].data + b, out.data + b, e - b); });
    };
}
inline void rows(const tensor_view& x, std::size_t& r, std::size_t& n) { n = std::size_t(x.shape.back()); r = x.size() / n; }

inline kernel_fn softmax() {
    return [](std::span<const tensor_view> in, tensor_view out, const spec::attrs&) {
        std::size_t r, n; rows(in[0], r, n);
        task::parallel_for(r, 4, [&](std::size_t b, std::size_t e) {
            for (std::size_t i = b; i < e; ++i) lane::active().softmax_row(in[0].data + i * n, out.data + i * n, n);
        });
    };
}
inline kernel_fn rms_norm() {
    return [](std::span<const tensor_view> in, tensor_view out, const spec::attrs& a) {
        std::size_t r, n; rows(in[0], r, n);
        float eps = float(a.num("eps", 1e-6));
        task::parallel_for(r, 4, [&](std::size_t b, std::size_t e) {
            for (std::size_t i = b; i < e; ++i) lane::active().rms_norm_row(in[0].data + i * n, in[1].data, out.data + i * n, n, eps);
        });
    };
}
// matmul_t with a 2-D [n,k] weight: the HF linear layer layout.
inline kernel_fn matmul_t() {
    return [](std::span<const tensor_view> in, tensor_view out, const spec::attrs&) {
        std::size_t k = std::size_t(in[0].shape.back()), m = in[0].size() / k, n = std::size_t(in[1].shape[0]);
        grid::prims::gemm_nt(in[0].data, in[1].data, out.data, m, n, k);
    };
}
inline bool weight_2d(sp in, const shape_t&, const spec::attrs&) { return in.size() == 2 && in[1].size() == 2; }
inline kernel_fn matmul() {
    return [](std::span<const tensor_view> in, tensor_view out, const spec::attrs&) {
        std::size_t k = std::size_t(in[0].shape.back()), m = in[0].size() / k, n = std::size_t(in[1].shape[1]);
        task::parallel_for((m + 3) / 4, 1, [&](std::size_t b, std::size_t e) {
            std::size_t r0 = b * 4, r1 = std::min(m, e * 4);
            lane::active().gemm_nn(in[0].data + r0 * k, in[1].data, out.data + r0 * n, r1 - r0, n, k);
        });
    };
}
inline kernel_fn attention() {
    return [](std::span<const tensor_view> in, tensor_view out, const spec::attrs& a) {
        const auto &q = in[0], &k = in[1], &v = in[2];
        std::size_t s = std::size_t(q.shape[0]), hq = std::size_t(q.shape[1]), d = std::size_t(q.shape[2]), dv = std::size_t(v.shape[2]);
        std::size_t t = std::size_t(k.shape[0]), hk = std::size_t(k.shape[1]), grp = hq / hk;
        bool causal = a.num("causal", 1) != 0;
        float sc = float(a.num("scale", 1.0 / std::sqrt(double(d))));
        std::size_t window = std::size_t(a.num("window", 0));
        const auto& ker = lane::active();
        auto kc = kv_codec(int(a.num("kv_codec_k", 0))), vc = kv_codec(int(a.num("kv_codec_v", 0)));
        if (vc != kv_codec::f32) dv = std::size_t(a.num("kv_dv", 0));
        std::size_t kw = std::size_t(kv_head_words(i64(d), kc)), vw = std::size_t(kv_head_words(i64(dv), vc));
        if (kc != kv_codec::f32 || vc != kv_codec::f32) {   // quantised KV: decode each (row, head) on the fly
            task::parallel_for(s * hq, 1, [&](std::size_t b, std::size_t e) {
                std::vector<float> sco(t), acc(dv), kt(d), vt(dv);
                for (std::size_t w = b; w < e; ++w) {
                    std::size_t i = w / hq, h = w % hq, kh = h / grp, pos = t - s + i;
                    std::size_t lo = 0, hi = causal ? pos + 1 : t;
                    if (window && pos + 1 > window) lo = pos + 1 - window;
                    const float* qp = q.data + (i * hq + h) * d;
                    float mx = -INFINITY;
                    for (std::size_t j = lo; j < hi; ++j) {
                        const float* kp = k.data + (j * hk + kh) * kw;
                        if (kc != kv_codec::f32) { kv_decode_head(kp, i64(d), kc, kt.data()); kp = kt.data(); }
                        sco[j] = ker.dot(qp, kp, d) * sc; mx = std::max(mx, sco[j]);
                    }
                    float sum = 0;
                    for (std::size_t j = lo; j < hi; ++j) { sco[j] = std::exp(sco[j] - mx); sum += sco[j]; }
                    std::fill(acc.begin(), acc.end(), 0.f);
                    for (std::size_t j = lo; j < hi; ++j) {
                        const float* vp = v.data + (j * hk + kh) * vw;
                        if (vc != kv_codec::f32) { kv_decode_head(vp, i64(dv), vc, vt.data()); vp = vt.data(); }
                        ker.saxpy(sco[j] / sum, vp, acc.data(), dv);
                    }
                    std::copy(acc.begin(), acc.end(), out.data + (i * hq + h) * dv);
                }
            });
            return;
        }
        task::parallel_for(s * hq, 1, [&](std::size_t b, std::size_t e) {
            std::vector<float> sco(t), acc(dv);
            for (std::size_t w = b; w < e; ++w) {
                std::size_t i = w / hq, h = w % hq, kh = h / grp, pos = t - s + i;
                std::size_t lo = 0, hi = causal ? pos + 1 : t;
                if (window && pos + 1 > window) lo = pos + 1 - window;
                const float* qp = q.data + (i * hq + h) * d;
                float mx = -INFINITY;
                for (std::size_t j = lo; j < hi; ++j) { sco[j] = ker.dot(qp, k.data + (j * hk + kh) * d, d) * sc; mx = std::max(mx, sco[j]); }
                float sum = 0;
                for (std::size_t j = lo; j < hi; ++j) { sco[j] = std::exp(sco[j] - mx); sum += sco[j]; }
                std::fill(acc.begin(), acc.end(), 0.f);
                for (std::size_t j = lo; j < hi; ++j) ker.saxpy(sco[j] / sum, v.data + (j * hk + kh) * dv, acc.data(), dv);
                std::copy(acc.begin(), acc.end(), out.data + (i * hq + h) * dv);
            }
        });
    };
}
inline kernel_fn rope() {
    return [](std::span<const tensor_view> in, tensor_view out, const spec::attrs& a) {
        const auto& x = in[0]; const auto& pos = in[1];
        double base = a.num("base", 10000.0); bool neox = a.num("neox", 1) != 0;
        std::size_t s = std::size_t(x.shape[0]), h = std::size_t(x.shape[1]), d = std::size_t(x.shape[2]), half = d / 2;
        std::vector<double> inv(half);
        for (std::size_t j = 0; j < half; ++j) inv[j] = std::pow(base, -2.0 * double(j) / double(d));
        task::parallel_for(s, 1, [&](std::size_t b, std::size_t e) {
            std::vector<float> c(half), sn(half);
            for (std::size_t i = b; i < e; ++i) {
                for (std::size_t j = 0; j < half; ++j) { double th = double(pos.data[i]) * inv[j]; c[j] = float(std::cos(th)); sn[j] = float(std::sin(th)); }
                for (std::size_t hh = 0; hh < h; ++hh) {
                    const float* src = x.data + (i * h + hh) * d; float* dst = out.data + (i * h + hh) * d;
                    for (std::size_t j = 0; j < half; ++j) {
                        std::size_t i0 = neox ? j : 2 * j, i1 = neox ? j + half : 2 * j + 1;
                        float x0 = src[i0], x1 = src[i1];
                        dst[i0] = x0 * c[j] - x1 * sn[j]; dst[i1] = x0 * sn[j] + x1 * c[j];
                    }
                }
            }
        });
    };
}
inline kernel_fn gather() {
    return [](std::span<const tensor_view> in, tensor_view out, const spec::attrs&) {
        std::size_t d = std::size_t(in[0].shape[1]), rowsn = std::size_t(in[0].shape[0]);
        for (std::size_t i = 0; i < in[1].size(); ++i) {
            auto r = std::size_t(in[1].data[i]);
            if (r >= rowsn) raise(errc::invalid_argument, "gather index {} out of range", r);
            std::copy_n(in[0].data + r * d, d, out.data + i * d);
        }
    };
}
inline kernel_fn reshape() {
    return [](std::span<const tensor_view> in, tensor_view out, const spec::attrs&) {
        if (in[0].data != out.data) std::copy_n(in[0].data, out.size(), out.data);
    };
}
} // namespace cpu_kernels

inline void registry::make_cpu(registry& r) {
    using namespace cpu_kernels;
    using K = lane::kernels;
    r.add({"lane.exp", spec::op::exp, 10, always, unary<&K::exp>()});
    r.add({"lane.log", spec::op::log, 10, always, unary<&K::log>()});
    r.add({"lane.tanh", spec::op::tanh, 10, always, unary<&K::tanh>()});
    r.add({"lane.sigmoid", spec::op::sigmoid, 10, always, unary<&K::sigmoid>()});
    r.add({"lane.silu", spec::op::silu, 10, always, unary<&K::silu>()});
    r.add({"lane.gelu_tanh", spec::op::gelu_tanh, 10, always, unary<&K::gelu>()});
    r.add({"lane.add", spec::op::add, 10, same_shapes, binary<&K::add>()});
    r.add({"lane.sub", spec::op::sub, 10, same_shapes, binary<&K::sub>()});
    r.add({"lane.mul", spec::op::mul, 10, same_shapes, binary<&K::mul>()});
    r.add({"lane.silu_mul", spec::op::silu_mul, 10, same_shapes, binary<&K::silu_mul>()});
    r.add({"lane.softmax", spec::op::softmax, 10, always, softmax()});
    r.add({"lane.rms_norm", spec::op::rms_norm, 10, always, rms_norm()});
    r.add({"lane.gemm_nt", spec::op::matmul_t, 10, weight_2d, matmul_t()});
    r.add({"lane.gemm_nn", spec::op::matmul, 10, weight_2d, matmul()});
    r.add({"lane.attention", spec::op::attention, 10, always, attention()});
    r.add({"cpu.rope", spec::op::rope, 10, always, rope()});
    r.add({"cpu.gather", spec::op::gather, 10, always, gather()});
    r.add({"cpu.reshape", spec::op::reshape, 10, always, reshape()});
}

} // namespace forge::flow
