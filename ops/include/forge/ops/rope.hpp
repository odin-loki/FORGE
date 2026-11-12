#pragma once
// RoPE inverse frequencies, shared by the scalar reference, the CPU kernel and every GPU host (the device kernels take
// the table as a buffer, or run on the CPU when they cannot). One function so that every lowering rotates by the same
// angles. With no scaling attrs the table is base^(-2j/d), computed exactly as before, so old files give the same bits.
//
// attrs of the `rope` op read here (all optional):
//   base          theta (default 10000)
//   scaling       0 none, 1 linear, 2 dynamic NTK, 3 YaRN, 4 llama3 (Hugging Face modeling_rope_utils.py names)
//   factor        the scaling factor (linear, dynamic, yarn, llama3)
//   orig_ctx      original_max_position_embeddings (yarn, llama3); max_position_embeddings (dynamic)
//   low_freq, high_freq         llama3 low_freq_factor / high_freq_factor (defaults 1, 4)
//   beta_fast, beta_slow, truncate   yarn (defaults 32, 1, 1)
//   freq_factors  per-pair multipliers applied last, d/2 IEEE-754 binary32 bit patterns as an int list (GGUF
//                 rope_freqs.weight: attrs carry numbers, int lists and strings, so the floats travel as their bits)
// YaRN's attention factor is not a rope attr: it scales q.k by its square, which templates put in attention's `scale`.
#include <forge/ops/op.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace forge::ops {

enum class rope_scaling : int { none = 0, linear = 1, dynamic = 2, yarn = 3, llama3 = 4 };

inline rope_scaling rope_scaling_of(const attrs& a) {
    double s = a.num("scaling", 0);
    if (!(s >= 0 && s <= 4) || s != std::floor(s)) raise(errc::invalid_argument, "rope: unknown scaling {} (0 none, 1 linear, 2 dynamic, 3 yarn, 4 llama3)", s);
    return rope_scaling(int(s));
}
// Dynamic NTK depends on the positions of each call: lowerings that build the table without them use the CPU.
inline bool rope_dynamic(const attrs& a) { return a.has("scaling") && rope_scaling_of(a) == rope_scaling::dynamic; }
// Any attr that changes the table away from base^(-2j/d).
inline bool rope_scaled(const attrs& a) { return (a.has("scaling") && rope_scaling_of(a) != rope_scaling::none) || a.has("freq_factors"); }

// The d/2 inverse frequencies for head dimension d. seq_len (1 + the largest position of the call) is read by dynamic
// NTK only.
inline std::vector<double> rope_inv_freq(const attrs& a, i64 d, double seq_len = 0) {
    const double base = a.num("base", 10000.0);
    const std::size_t half = std::size_t(d > 0 ? d / 2 : 0);
    std::vector<double> inv(half);
    const rope_scaling sc = a.has("scaling") ? rope_scaling_of(a) : rope_scaling::none;
    const double factor = a.num("factor", 1.0), orig = a.num("orig_ctx", 0);
    if (sc != rope_scaling::none && !(factor > 0 && std::isfinite(factor))) raise(errc::invalid_argument, "rope: scaling factor {} must be positive", factor);
    if ((sc == rope_scaling::dynamic || sc == rope_scaling::yarn || sc == rope_scaling::llama3) && !(orig > 0))
        raise(errc::invalid_argument, "rope: scaling {} needs orig_ctx > 0", int(sc));
    double b = base;
    if (sc == rope_scaling::dynamic && seq_len > orig && d > 2)   // HF _compute_dynamic_ntk_parameters
        b = base * std::pow(factor * seq_len / orig - (factor - 1.0), double(d) / double(d - 2));
    for (std::size_t j = 0; j < half; ++j) inv[j] = std::pow(b, -2.0 * double(j) / double(d));
    const double pi = 3.14159265358979323846;
    switch (sc) {
    case rope_scaling::none: case rope_scaling::dynamic: break;
    case rope_scaling::linear: for (auto& x : inv) x /= factor; break;
    case rope_scaling::yarn: {   // HF _compute_yarn_parameters (llama.cpp ggml_rope_yarn_corr_dims + rope_yarn_ramp)
        const double bf = a.num("beta_fast", 32.0), bs = a.num("beta_slow", 1.0);
        auto corr = [&](double rot) { return double(d) * std::log(orig / (rot * 2.0 * pi)) / (2.0 * std::log(base)); };
        double lo = corr(bf), hi = corr(bs);
        if (a.num("truncate", 1) != 0) { lo = std::floor(lo); hi = std::ceil(hi); }
        lo = std::max(lo, 0.0); hi = std::min(hi, double(d - 1));
        if (lo == hi) hi += 0.001;
        for (std::size_t j = 0; j < half; ++j) {
            double ramp = std::min(1.0, std::max(0.0, (double(j) - lo) / (hi - lo))), extra = 1.0 - ramp;   // 1 = keep (extrapolate)
            inv[j] = inv[j] / factor * (1.0 - extra) + inv[j] * extra;
        }
        break; }
    case rope_scaling::llama3: {   // HF _compute_llama3_parameters
        const double lf = a.num("low_freq", 1.0), hf = a.num("high_freq", 4.0);
        if (!(lf > 0) || !(hf > 0) || lf == hf) raise(errc::invalid_argument, "rope: llama3 scaling needs low_freq != high_freq, both positive");
        const double low_wl = orig / lf, high_wl = orig / hf;
        for (auto& x : inv) {
            double wl = 2.0 * pi / x;
            if (wl > low_wl) x /= factor;
            else if (!(wl < high_wl)) {
                double smooth = (orig / wl - lf) / (hf - lf);
                x = (1.0 - smooth) * x / factor + smooth * x;
            }
        }
        break; }
    }
    if (a.has("freq_factors")) {
        auto bits = a.ints("freq_factors");
        if (bits.size() != half) raise(errc::shape_mismatch, "rope: freq_factors has {} entries for head dimension {} (needs {})", bits.size(), d, half);
        for (std::size_t j = 0; j < half; ++j) {
            std::uint32_t u = std::uint32_t(bits[j]); float f; std::memcpy(&f, &u, 4);
            if (!std::isfinite(f)) raise(errc::invalid_argument, "rope: freq_factors[{}] is not finite", j);
            inv[j] *= double(f);
        }
    }
    return inv;
}

// seq_len for the dynamic variant: 1 + the largest position (HF: torch.max(position_ids) + 1).
inline double rope_seq_len(const float* pos, std::size_t n) {
    double m = -1; for (std::size_t i = 0; i < n; ++i) m = std::max(m, double(pos[i]));
    return m + 1.0;
}

} // namespace forge::ops
