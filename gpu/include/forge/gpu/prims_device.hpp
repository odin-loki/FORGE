#pragma once
// gpu::prims on a device runtime. prims.hpp holds the CPU implementations; this header runs the same primitives
// through the generated device kernels, for any runtime with the run_op contract of gpu/cuda.hpp (CUDA, HIP, OpenCL,
// Level Zero, Metal, Vulkan). Each function returns false when the device has no kernel for the primitive or the
// shape, and then nothing was computed: call the CPU primitive of the same name instead. There is no device FFT.
//
//   primitive          device kernels
//   gemv, gemm_batched forge_matmul_t; the epilogue as forge_bcast (bias) and the activation's unary kernel
//   conv1d, conv2d     forge_conv1d / forge_conv2d
//   softmax            forge_softmax
//   rms_norm           forge_rms_norm
//   layer_norm         forge_layer_norm
//   flash_attention    forge_attention_fa (one pass, online softmax) or the split-over-keys / tiled / tensor-core
//                      kernels the runtime picks for the shape; forge_attention where those are absent
//
// The data are host arrays, as for the CPU primitives. Outside a run every call uploads its inputs and reads its
// output back; inside begin_run() / end_run() they stay on the device between calls.
#include <forge/gpu/prims.hpp>
#include <forge/gpu/cuda.hpp>
#include <forge/core/tensor.hpp>
#include <forge/ops/ops.gen.hpp>
#include <string>
#include <vector>

namespace forge::gpu::prims::device {

namespace detail {
inline tensor_view view(const float* p, shape_t s) { return {const_cast<float*>(p), std::move(s)}; }
inline const char* act_kernel(act a) {
    switch (a) {
    case act::relu: return "relu"; case act::silu: return "silu"; case act::gelu_tanh: return "gelu_tanh"; case act::tanh: return "tanh";
    case act::sigmoid: return "sigmoid"; case act::none: break;
    }
    return nullptr;
}
// y[rows, n] = act(y + bias) in place on the device
template <class RT> bool run_epilogue(RT& rt, float* y, std::size_t rows, std::size_t n, const epilogue& e) {
    const shape_t ys{i64(rows), i64(n)};
    if (e.bias) {
        if (rows == 1) { if (!rt.run_op("add", {view(y, {i64(n)}), view(e.bias, {i64(n)})}, {}, view(y, {i64(n)}))) return false; }
        else if constexpr (requires { rt.run_bcast(0, std::vector<tensor_view>{}, tensor_view{}); }) {
            if (!rt.run_bcast(0, {view(y, ys), view(e.bias, {i64(n)})}, view(y, ys))) return false;
        } else return false;
    }
    if (const char* k = act_kernel(e.fn)) return rt.run_op(k, {view(y, ys)}, {}, view(y, ys));
    return true;
}
}

// y[n] = act(W[n, k] . x[k] + bias)
template <class RT> bool gemv(RT& rt, const float* W, const float* x, float* y, std::size_t n, std::size_t k, const epilogue& e = {}) {
    if (!rt.run_op("matmul_t", {detail::view(x, {1, i64(k)}), detail::view(W, {i64(n), i64(k)})}, {}, detail::view(y, {1, i64(n)}))) return false;
    return detail::run_epilogue(rt, y, 1, n, e);
}
// C[b, m, n] = act(A[b, m, k] . B^T + bias); B is [n, k] (shared_b) or [batch, n, k]
template <class RT> bool gemm_batched(RT& rt, const float* A, const float* B, float* C, std::size_t batch, std::size_t m, std::size_t n, std::size_t k,
                                      bool shared_b = true, const epilogue& e = {}) {
    if (shared_b) {
        if (!rt.run_op("matmul_t", {detail::view(A, {i64(batch * m), i64(k)}), detail::view(B, {i64(n), i64(k)})}, {}, detail::view(C, {i64(batch * m), i64(n)}))) return false;
    } else for (std::size_t b = 0; b < batch; ++b)
        if (!rt.run_op("matmul_t", {detail::view(A + b * m * k, {i64(m), i64(k)}), detail::view(B + b * n * k, {i64(n), i64(k)})}, {}, detail::view(C + b * m * n, {i64(m), i64(n)}))) return false;
    return detail::run_epilogue(rt, C, batch * m, n, e);
}
// x[C, H, W] * w[O, cg, kh, kw] + b -> y[O, oh, ow] (prims::conv2d); nb = 1: one shared bias, else b is [O]
template <class RT> bool conv2d(RT& rt, const float* x, const float* w, const float* b, float* y, std::size_t C, std::size_t H, std::size_t W, std::size_t O,
                                std::size_t cg, std::size_t kh, std::size_t kw, const conv_geom& g = {}, std::size_t nb = 0) {
    if (!b || !cg || C % cg) return false;
    const std::size_t oh = conv_out(H, kh, g.stride[0], g.pad[0], g.dilation[0]), ow = conv_out(W, kw, g.stride[1], g.pad[1], g.dilation[1]);
    ops::attrs a{{"stride", std::vector<i64>{i64(g.stride[0]), i64(g.stride[1])}}, {"pad", std::vector<i64>{i64(g.pad[0]), i64(g.pad[1])}},
                 {"dilation", std::vector<i64>{i64(g.dilation[0]), i64(g.dilation[1])}}, {"groups", double(C / cg)}};
    return rt.run_op("conv2d", {detail::view(x, {i64(C), i64(H), i64(W)}), detail::view(w, {i64(O), i64(cg), i64(kh), i64(kw)}), detail::view(b, {i64(nb == 1 ? 1 : O)})}, a,
                     detail::view(y, {i64(O), i64(oh), i64(ow)}));
}
template <class RT> bool conv1d(RT& rt, const float* x, const float* w, const float* b, float* y, std::size_t C, std::size_t L, std::size_t O, std::size_t cg,
                                std::size_t k, std::size_t stride = 1, std::size_t pad = 0, std::size_t dilation = 1, std::size_t nb = 0) {
    if (!b || !cg || C % cg) return false;
    ops::attrs a{{"stride", double(stride)}, {"pad", double(pad)}, {"dilation", double(dilation)}, {"groups", double(C / cg)}};
    return rt.run_op("conv1d", {detail::view(x, {i64(C), i64(L)}), detail::view(w, {i64(O), i64(cg), i64(k)}), detail::view(b, {i64(nb == 1 ? 1 : O)})}, a,
                     detail::view(y, {i64(O), i64(conv_out(L, k, stride, pad, dilation))}));
}
template <class RT> bool softmax(RT& rt, const float* x, float* y, std::size_t rows, std::size_t cols) {
    const shape_t s{i64(rows), i64(cols)};
    return rt.run_op("softmax", {detail::view(x, s)}, {}, detail::view(y, s));
}
template <class RT> bool rms_norm(RT& rt, const float* x, const float* w, float* y, std::size_t rows, std::size_t cols, float eps = 1e-6f) {
    const shape_t s{i64(rows), i64(cols)};
    return rt.run_op("rms_norm", {detail::view(x, s), detail::view(w, {i64(cols)})}, {{"eps", double(eps)}}, detail::view(y, s));
}
template <class RT> bool layer_norm(RT& rt, const float* x, const float* w, const float* b, float* y, std::size_t rows, std::size_t cols, float eps = 1e-5f) {
    const shape_t s{i64(rows), i64(cols)};
    return rt.run_op("layer_norm", {detail::view(x, s), detail::view(w, {i64(cols)}), detail::view(b, {i64(cols)})}, {{"eps", double(eps)}}, detail::view(y, s));
}
// q[s, hq, d], k[t, hk, d], v[t, hk, dv] -> y[s, hq, dv] (prims::flash_attention); the tile size is the kernel's own
template <class RT> bool flash_attention(RT& rt, const float* q, const float* k, const float* v, float* y, std::size_t s, std::size_t hq, std::size_t d, std::size_t t,
                                         std::size_t hk, std::size_t dv, const attention_opts& o = {}) {
    ops::attrs a{{"causal", o.causal ? 1.0 : 0.0}, {"window", double(o.window)}};
    if (o.scale != 0) a.set("scale", double(o.scale));
    return rt.run_op("attention", {detail::view(q, {i64(s), i64(hq), i64(d)}), detail::view(k, {i64(t), i64(hk), i64(d)}), detail::view(v, {i64(t), i64(hk), i64(dv)})}, a,
                     detail::view(y, {i64(s), i64(hq), i64(dv)}));
}

} // namespace forge::gpu::prims::device
