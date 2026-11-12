#pragma once
// gpu::prims — portable primitives (CPU backend implementations; GPU backends lower the same specs).
// The device versions of the linear-algebra, convolution, normalisation and attention primitives are in
// gpu/prims_device.hpp.
#include <forge/gpu/gpu.hpp>
#include <forge/rand/philox.hpp>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numeric>
#include <limits>
#include <stdexcept>

namespace forge::gpu::prims {

template <class T, class Op = std::plus<T>> T reduce(span<const T> x, T init = T{}, Op op = {}) {
    std::size_t n = x.size(), parts = std::max<std::size_t>(1, std::min<std::size_t>(64, n / 4096));
    std::vector<T> partial(parts, init);
    task::parallel_for(parts, 1, [&](std::size_t b, std::size_t e) {
        for (std::size_t p = b; p < e; ++p) {
            std::size_t lo = n * p / parts, hi = n * (p + 1) / parts; T acc = init;
            for (std::size_t i = lo; i < hi; ++i) acc = op(acc, x[i]);
            partial[p] = acc;
        }
    });
    T r = init; for (auto& v : partial) r = op(r, v); return r;
}

// Two-pass blocked scan (inclusive or exclusive).
template <class T> void scan(span<const T> x, span<T> y, bool inclusive = true) {
    std::size_t n = x.size(), parts = std::max<std::size_t>(1, std::min<std::size_t>(64, n / 4096));
    std::vector<T> sums(parts + 1, T{});
    task::parallel_for(parts, 1, [&](std::size_t b, std::size_t e) {
        for (std::size_t p = b; p < e; ++p) {
            std::size_t lo = n * p / parts, hi = n * (p + 1) / parts; T s{};
            for (std::size_t i = lo; i < hi; ++i) s += x[i];
            sums[p + 1] = s;
        }
    });
    for (std::size_t p = 1; p <= parts; ++p) sums[p] += sums[p - 1];
    task::parallel_for(parts, 1, [&](std::size_t b, std::size_t e) {
        for (std::size_t p = b; p < e; ++p) {
            std::size_t lo = n * p / parts, hi = n * (p + 1) / parts; T s = sums[p];
            for (std::size_t i = lo; i < hi; ++i) { if (inclusive) { s += x[i]; y[i] = s; } else { y[i] = s; s += x[i]; } }
        }
    });
}

// Stream compaction: keep x[i] where pred(x[i]); returns count.
template <class T, class P> std::size_t compact(span<const T> x, span<T> out, P pred) {
    std::vector<std::uint32_t> flags(x.size()), pos(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) flags[i] = pred(x[i]) ? 1u : 0u;
    scan<std::uint32_t>(flags, pos, false);
    std::size_t c = x.empty() ? 0 : pos.back() + flags.back();
    for (std::size_t i = 0; i < x.size(); ++i) if (flags[i]) out[pos[i]] = x[i];
    return c;
}

// LSD radix sort on 32-bit keys (optionally carrying values).
inline void radix_sort(span<std::uint32_t> keys, span<std::uint32_t> vals = {}) {
    std::vector<std::uint32_t> k2(keys.size()), v2(vals.size());
    span<std::uint32_t> ka = keys, kb = k2, va = vals, vb = v2;
    for (int sh = 0; sh < 32; sh += 8) {
        std::array<std::size_t, 257> cnt{};
        for (auto k : ka) ++cnt[((k >> sh) & 0xff) + 1];
        for (int i = 0; i < 256; ++i) cnt[std::size_t(i) + 1] += cnt[std::size_t(i)];
        for (std::size_t i = 0; i < ka.size(); ++i) {
            auto d = cnt[(ka[i] >> sh) & 0xff]++;
            kb[d] = ka[i]; if (!va.empty()) vb[d] = va[i];
        }
        std::swap(ka, kb); std::swap(va, vb);
    }
    // 4 passes: result is back in the original buffers.
}

inline void histogram(span<const std::uint32_t> x, span<std::uint32_t> bins) {
    for (auto v : x) if (v < bins.size()) ++bins[v];
}

// Segmented inclusive sum: seg_start[i] != 0 begins a new segment at i.
template <class T> void segmented_scan(span<const T> x, span<const std::uint8_t> seg_start, span<T> out) {
    T acc{}; for (std::size_t i = 0; i < x.size(); ++i) { acc = (i == 0 || seg_start[i]) ? x[i] : acc + x[i]; out[i] = acc; }
}
// Per-segment sums given CSR-style offsets (offsets.size() == segments + 1).
template <class T> void segmented_reduce(span<const T> x, span<const std::uint32_t> offsets, span<T> out) {
    for (std::size_t s = 0; s + 1 < offsets.size(); ++s) { T a{}; for (auto i = offsets[s]; i < offsets[s + 1]; ++i) a += x[i]; out[s] = a; }
}
// Bitonic sorting network (ascending); the data-independent compare/exchange sequence that GPU kernels use.
// Any length: pads to a power of two with +inf internally.
inline void bitonic_sort(span<float> x) {
    std::size_t n = 1; while (n < x.size()) n <<= 1;
    std::vector<float> a(n, std::numeric_limits<float>::infinity()); std::copy(x.begin(), x.end(), a.begin());
    for (std::size_t k = 2; k <= n; k <<= 1)
        for (std::size_t j = k >> 1; j > 0; j >>= 1)
            for (std::size_t i = 0; i < n; ++i) { std::size_t l = i ^ j; if (l > i) { bool up = (i & k) == 0; if ((a[i] > a[l]) == up) std::swap(a[i], a[l]); } }
    std::copy(a.begin(), a.begin() + std::ptrdiff_t(x.size()), x.begin());
}
// Open-addressing hash map, u64 keys -> u32 values, linear probing, power-of-two capacity, grows at 50% load.
class hash_map {
public:
    explicit hash_map(std::size_t cap = 16) { std::size_t c = 16; while (c < cap * 2) c <<= 1; k_.assign(c, empty); v_.assign(c, 0); }
    void insert(std::uint64_t key, std::uint32_t val) {
        if (key == empty) throw std::invalid_argument("hash_map: reserved key");
        if ((n_ + 1) * 2 > k_.size()) grow();
        std::size_t i = slot(key); if (k_[i] == empty) { k_[i] = key; ++n_; } v_[i] = val;
    }
    const std::uint32_t* find(std::uint64_t key) const { std::size_t i = slot(key); return k_[i] == key ? &v_[i] : nullptr; }
    std::size_t size() const { return n_; }
    static constexpr std::uint64_t empty = ~std::uint64_t(0);
private:
    static std::uint64_t mix(std::uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; return x ^ (x >> 33); }
    std::size_t slot(std::uint64_t key) const { std::size_t m = k_.size() - 1, i = std::size_t(mix(key)) & m; while (k_[i] != empty && k_[i] != key) i = (i + 1) & m; return i; }
    void grow() { auto k = std::move(k_); auto v = std::move(v_); k_.assign(k.size() * 2, empty); v_.assign(k.size() * 2, 0); n_ = 0; for (std::size_t i = 0; i < k.size(); ++i) if (k[i] != empty) insert(k[i], v[i]); }
    std::vector<std::uint64_t> k_; std::vector<std::uint32_t> v_; std::size_t n_ = 0;
};

// Top-k (values descending) returning indices.
inline std::vector<std::uint32_t> top_k(span<const float> x, std::size_t k) {
    std::vector<std::uint32_t> idx(x.size());
    std::iota(idx.begin(), idx.end(), 0u);
    k = std::min(k, x.size());
    std::partial_sort(idx.begin(), idx.begin() + std::ptrdiff_t(k), idx.end(),
                      [&](auto a, auto b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
    idx.resize(k);
    return idx;
}

// GEMM: C[m,n] = A[m,k] * B[n,k]^T, rows split across the pool, micro-kernel from lane dispatch.
inline void gemm_nt(const float* A, const float* B, float* C, std::size_t m, std::size_t n, std::size_t k) {
    const auto& ker = simd::active();
    if (m == 1) {   // GEMV: split over output columns
        task::parallel_for(n, 64, [&](std::size_t b, std::size_t e) { ker.gemm_nt(A, B + b * k, C + b, 1, e - b, k);
            // gemm_nt writes C[j] with stride n=e-b for m=1 -> contiguous; fine.
        });
        return;
    }
    task::parallel_for((m + 3) / 4, 1, [&](std::size_t b, std::size_t e) {
        std::size_t r0 = b * 4, r1 = std::min(m, e * 4);
        // compute rows [r0, r1) against all n columns, in column blocks to keep B in cache
        constexpr std::size_t NB = 64;
        std::vector<float> tmp((r1 - r0) * NB);
        for (std::size_t j = 0; j < n; j += NB) {
            std::size_t nb = std::min(NB, n - j);
            ker.gemm_nt(A + r0 * k, B + j * k, tmp.data(), r1 - r0, nb, k);
            for (std::size_t r = 0; r < r1 - r0; ++r) std::copy_n(&tmp[r * nb], nb, C + (r0 + r) * n + j);
        }
    });
}

// Philox RNG fill: identical sequence on every backend for (seed, offset).
inline void philox_uniform(span<float> out, std::uint64_t seed, std::uint64_t offset = 0) {
    rand::philox4x32 g{{std::uint32_t(seed), std::uint32_t(seed >> 32)}};
    for (std::size_t i = 0; i < out.size(); i += 4) {
        auto b = g(offset + i / 4);
        for (std::size_t j = 0; j < 4 && i + j < out.size(); ++j) out[i + j] = rand::philox4x32::to_unit(b[j]);
    }
}

// ---- linear algebra, convolution, FFT, normalisation and attention (FORGE_PLAN B.6) --------------------------------
// CPU implementations on the simd kernel table; rows / tiles are split across the task pool. The same primitives on
// a device runtime are in gpu/prims_device.hpp (through the generated kernels, where one exists).

// What a matrix product applies to each output row before it is written: y = act(y + bias).
enum class act { none, relu, silu, gelu_tanh, tanh, sigmoid };   // gelu_tanh: the tanh form, as the simd table's gelu
struct epilogue {
    const float* bias = nullptr;   // [n], added to every output row; null = none
    act fn = act::none;
};
inline void apply_epilogue(float* y, std::size_t n, const epilogue& e) {
    const auto& ker = simd::active();
    if (e.bias) ker.add(y, e.bias, y, n);
    switch (e.fn) {
    case act::none: break;
    case act::relu: for (std::size_t i = 0; i < n; ++i) y[i] = y[i] > 0.f ? y[i] : 0.f; break;
    case act::silu: ker.silu(y, y, n); break;
    case act::gelu_tanh: ker.gelu(y, y, n); break;
    case act::tanh: ker.tanh(y, y, n); break;
    case act::sigmoid: ker.sigmoid(y, y, n); break;
    }
}

// GEMV: y[n] = act(W[n, k] . x[k] + bias).
inline void gemv(const float* W, const float* x, float* y, std::size_t n, std::size_t k, const epilogue& e = {}) {
    const auto& ker = simd::active();
    task::parallel_for(n, 64, [&](std::size_t b, std::size_t en) { for (std::size_t j = b; j < en; ++j) y[j] = ker.dot(W + j * k, x, k); });
    apply_epilogue(y, n, e);
}

// Batched GEMM with an epilogue: C[b, m, n] = act(A[b, m, k] . B^T + bias) for b < batch. B is [n, k], shared by every
// batch (shared_b) or [batch, n, k]. The epilogue runs on each output row while it is still in cache.
inline void gemm_batched(const float* A, const float* B, float* C, std::size_t batch, std::size_t m, std::size_t n, std::size_t k,
                         bool shared_b = true, const epilogue& e = {}) {
    if (shared_b) gemm_nt(A, B, C, batch * m, n, k);
    else for (std::size_t b = 0; b < batch; ++b) gemm_nt(A + b * m * k, B + b * n * k, C + b * m * n, m, n, k);
    if (e.bias || e.fn != act::none)
        task::parallel_for(batch * m, 8, [&](std::size_t r0, std::size_t r1) { for (std::size_t r = r0; r < r1; ++r) apply_epilogue(C + r * n, n, e); });
}

// Convolution as im2col + gemm_nt, one product per group: x[C, H, W] * w[O, C / groups, kh, kw] + b ([O], or [1] with
// nb = 1) -> y[O, oh, ow]; cross-correlation with zero padding, as ops::ref::conv2d. conv1d is the kh = 1 case.
struct conv_geom { std::size_t stride[2] = {1, 1}, pad[2] = {0, 0}, dilation[2] = {1, 1}; };
inline std::size_t conv_out(std::size_t in, std::size_t k, std::size_t stride, std::size_t pad, std::size_t dilation) {
    const std::size_t span = dilation * (k - 1) + 1;
    return in + 2 * pad < span ? 0 : (in + 2 * pad - span) / stride + 1;
}
inline void conv2d(const float* x, const float* w, const float* b, float* y, std::size_t C, std::size_t H, std::size_t W, std::size_t O, std::size_t cg,
                   std::size_t kh, std::size_t kw, const conv_geom& g = {}, std::size_t nb = 0) {
    if (!cg || C % cg || O % (C / cg)) throw std::invalid_argument("prims::conv2d: channels do not match the weight's groups");
    const std::size_t groups = C / cg, og = O / groups, oh = conv_out(H, kh, g.stride[0], g.pad[0], g.dilation[0]), ow = conv_out(W, kw, g.stride[1], g.pad[1], g.dilation[1]);
    const std::size_t np = oh * ow, kk = cg * kh * kw;
    if (!np) return;
    std::vector<float> col(np * kk), out(np * og);
    for (std::size_t gi = 0; gi < groups; ++gi) {
        task::parallel_for(np, 16, [&](std::size_t p0, std::size_t p1) {
            for (std::size_t p = p0; p < p1; ++p) {
                const std::size_t oy = p / ow, ox = p % ow; float* c = col.data() + p * kk;
                for (std::size_t ch = 0; ch < cg; ++ch) for (std::size_t i = 0; i < kh; ++i) for (std::size_t j = 0; j < kw; ++j) {
                    const std::ptrdiff_t iy = std::ptrdiff_t(oy * g.stride[0] + i * g.dilation[0]) - std::ptrdiff_t(g.pad[0]);
                    const std::ptrdiff_t ix = std::ptrdiff_t(ox * g.stride[1] + j * g.dilation[1]) - std::ptrdiff_t(g.pad[1]);
                    *c++ = iy < 0 || iy >= std::ptrdiff_t(H) || ix < 0 || ix >= std::ptrdiff_t(W) ? 0.f : x[((gi * cg + ch) * H + std::size_t(iy)) * W + std::size_t(ix)];
                }
            }
        });
        gemm_nt(col.data(), w + gi * og * kk, out.data(), np, og, kk);   // [np, og]
        for (std::size_t o = 0; o < og; ++o) {
            const std::size_t oc = gi * og + o; const float bias = b ? b[nb == 1 ? 0 : oc] : 0.f;
            for (std::size_t p = 0; p < np; ++p) y[oc * np + p] = out[p * og + o] + bias;
        }
    }
}
inline void conv1d(const float* x, const float* w, const float* b, float* y, std::size_t C, std::size_t L, std::size_t O, std::size_t cg, std::size_t k,
                   std::size_t stride = 1, std::size_t pad = 0, std::size_t dilation = 1, std::size_t nb = 0) {
    conv_geom g; g.stride[1] = stride; g.pad[1] = pad; g.dilation[1] = dilation;
    conv2d(x, w, b, y, C, 1, L, O, cg, 1, k, g, nb);
}

// FFT of n complex values in place (inverse: scaled by 1 / n). Powers of two: iterative radix 2; any other length:
// Bluestein's chirp-z over a power-of-two FFT, so every n costs O(n log n). Twiddles are computed in double.
namespace detail {
inline void fft_pow2(std::complex<float>* x, std::size_t n, bool inverse) {
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit; if (i < j) std::swap(x[i], x[j]);
    }
    const double pi = 3.14159265358979323846;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len / 2; std::vector<std::complex<float>> tw(half);
        for (std::size_t i = 0; i < half; ++i) { const double a = (inverse ? 2.0 : -2.0) * pi * double(i) / double(len); tw[i] = {float(std::cos(a)), float(std::sin(a))}; }
        task::parallel_for(n / len, std::max<std::size_t>(1, 4096 / len), [&](std::size_t b0, std::size_t b1) {
            for (std::size_t b = b0; b < b1; ++b) { auto* p = x + b * len;
                for (std::size_t i = 0; i < half; ++i) { const auto u = p[i], v = p[i + half] * tw[i]; p[i] = u + v; p[i + half] = u - v; } }
        });
    }
}
}
inline void fft(span<std::complex<float>> x, bool inverse = false) {
    const std::size_t n = x.size();
    if (n < 2) return;
    if ((n & (n - 1)) == 0) detail::fft_pow2(x.data(), n, inverse);
    else {
        std::size_t m = 1; while (m < 2 * n - 1) m <<= 1;
        const double pi = 3.14159265358979323846;
        std::vector<std::complex<float>> chirp(n), a(m), b(m);
        for (std::size_t i = 0; i < n; ++i) {   // exp(-+ i pi i^2 / n), with i^2 reduced mod 2n so the angle stays accurate
            const double ang = (inverse ? 1.0 : -1.0) * pi * double((i * i) % (2 * n)) / double(n);
            chirp[i] = {float(std::cos(ang)), float(std::sin(ang))};
        }
        for (std::size_t i = 0; i < n; ++i) a[i] = x[i] * chirp[i];
        b[0] = std::conj(chirp[0]);
        for (std::size_t i = 1; i < n; ++i) b[i] = b[m - i] = std::conj(chirp[i]);
        detail::fft_pow2(a.data(), m, false); detail::fft_pow2(b.data(), m, false);
        for (std::size_t i = 0; i < m; ++i) a[i] *= b[i];
        detail::fft_pow2(a.data(), m, true);
        const float sc = 1.f / float(m);
        for (std::size_t i = 0; i < n; ++i) x[i] = a[i] * sc * chirp[i];
    }
    if (inverse) { const float sc = 1.f / float(n); for (auto& v : x) v *= sc; }
}

// Row-wise softmax, RMS norm (y = x / sqrt(mean(x^2) + eps) * w) and layer norm ((x - mean) / sqrt(var + eps) * w + b)
// over [rows, cols]; in place is fine.
inline void softmax(const float* x, float* y, std::size_t rows, std::size_t cols) {
    const auto& ker = simd::active();
    task::parallel_for(rows, 4, [&](std::size_t r0, std::size_t r1) { for (std::size_t r = r0; r < r1; ++r) ker.softmax_row(x + r * cols, y + r * cols, cols); });
}
inline void rms_norm(const float* x, const float* w, float* y, std::size_t rows, std::size_t cols, float eps = 1e-6f) {
    const auto& ker = simd::active();
    task::parallel_for(rows, 4, [&](std::size_t r0, std::size_t r1) { for (std::size_t r = r0; r < r1; ++r) ker.rms_norm_row(x + r * cols, w, y + r * cols, cols, eps); });
}
inline void layer_norm(const float* x, const float* w, const float* b, float* y, std::size_t rows, std::size_t cols, float eps = 1e-5f) {
    task::parallel_for(rows, 4, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* xr = x + r * cols; float* yr = y + r * cols;
            double m = 0; for (std::size_t j = 0; j < cols; ++j) m += xr[j];
            m /= double(cols);
            double v = 0; for (std::size_t j = 0; j < cols; ++j) { const double d = xr[j] - m; v += d * d; }
            const double inv = 1.0 / std::sqrt(v / double(cols) + double(eps));
            for (std::size_t j = 0; j < cols; ++j) yr[j] = float((xr[j] - m) * inv * w[j] + b[j]);
        }
    });
}

// Flash-style attention: q[s, hq, d], k[t, hk, d], v[t, hk, dv] -> y[s, hq, dv], GQA with hq a multiple of hk; query i
// sits at position t - s + i. Keys are taken in tiles with an online softmax (running maximum and weight sum, the
// accumulator rescaled when the maximum moves), so no [s, t] score matrix is ever stored. causal masks later keys,
// window w > 0 masks keys w or more positions behind the query. A row with no visible key gives zeros.
struct attention_opts { bool causal = true; std::size_t window = 0; float scale = 0; /* 0 = 1 / sqrt(d) */ std::size_t tile = 64; };
inline void flash_attention(const float* q, const float* k, const float* v, float* y, std::size_t s, std::size_t hq, std::size_t d, std::size_t t, std::size_t hk,
                            std::size_t dv, const attention_opts& o = {}) {
    if (!hk || hq % hk) throw std::invalid_argument("prims::flash_attention: query heads must be a multiple of key heads");
    const auto& ker = simd::active();
    const float scale = o.scale != 0 ? o.scale : 1.f / std::sqrt(float(d));
    const std::size_t tile = std::max<std::size_t>(1, o.tile);
    task::parallel_for(s * hq, 1, [&](std::size_t i0, std::size_t i1) {
        std::vector<float> sc(tile);
        for (std::size_t id = i0; id < i1; ++id) {
            const std::size_t i = id / hq, h = id % hq, kh = h / (hq / hk);
            const std::ptrdiff_t pos = std::ptrdiff_t(t) - std::ptrdiff_t(s) + std::ptrdiff_t(i);
            std::size_t jend = t; if (o.causal) jend = pos + 1 < 0 ? 0 : std::min<std::size_t>(t, std::size_t(pos + 1));
            std::size_t jbeg = 0; if (o.window && pos - std::ptrdiff_t(o.window) + 1 > 0) jbeg = std::size_t(pos - std::ptrdiff_t(o.window) + 1);
            const float* qp = q + id * d; float* yp = y + id * dv;
            std::fill(yp, yp + dv, 0.f);
            float m = -std::numeric_limits<float>::infinity(), sum = 0;
            for (std::size_t j0 = jbeg; j0 < jend; j0 += tile) {
                const std::size_t nt = std::min(tile, jend - j0); float tm = -std::numeric_limits<float>::infinity();
                for (std::size_t j = 0; j < nt; ++j) { sc[j] = ker.dot(qp, k + ((j0 + j) * hk + kh) * d, d) * scale; tm = std::max(tm, sc[j]); }
                if (tm > m) {   // the running maximum moved: rescale what was accumulated under the old one
                    const float r = std::isinf(m) ? 0.f : std::exp(m - tm);
                    sum *= r; for (std::size_t e = 0; e < dv; ++e) yp[e] *= r;
                    m = tm;
                }
                for (std::size_t j = 0; j < nt; ++j) { const float w = std::exp(sc[j] - m); sum += w; ker.saxpy(w, v + ((j0 + j) * hk + kh) * dv, yp, dv); }
            }
            const float inv = sum > 0 ? 1.f / sum : 0.f;
            for (std::size_t e = 0; e < dv; ++e) yp[e] *= inv;
        }
    });
}

} // namespace forge::gpu::prims
