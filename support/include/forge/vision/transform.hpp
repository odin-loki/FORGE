#pragma once
// Pre-processing on HWC float images (vision::image): u8 <-> float, HWC <-> CHW, normalise / standardise,
// resize (nearest, bilinear, bicubic, area; optional antialiasing), crop, pad, letterbox, and colour
// conversions (RGB/BGR/grey, planar YUV 4:2:0 and NV12 to and from RGB). Plain loops; nothing is vectorised by hand.
#include <forge/vision/codec.hpp>
#include <forge/vision/vision.hpp>
#include <stdexcept>

namespace forge::vision {

// ---- sample formats and layouts ----
// u8 HWC -> float HWC, value * scale (1/255 by default).
inline image to_float(const std::uint8_t* px, std::size_t h, std::size_t w, std::size_t c, float scale = 1.0f / 255.0f) {
    image im{h, w, c, std::vector<float>(h * w * c)};
    for (std::size_t i = 0; i < im.px.size(); ++i) im.px[i] = float(px[i]) * scale;
    return im;
}
// A decoded raster as floats in [0, 1] (8- and 16-bit).
inline image to_float(const raster& r) {
    image im{r.h, r.w, r.c, std::vector<float>(r.samples())};
    if (r.depth == 8) for (std::size_t i = 0; i < im.px.size(); ++i) im.px[i] = float(r.u8[i]) * (1.0f / 255.0f);
    else for (std::size_t i = 0; i < im.px.size(); ++i) im.px[i] = float(r.u16[i]) * (1.0f / 65535.0f);
    return im;
}
// float -> u8 with value * scale, rounded to nearest and clamped to [0, 255]; NaN becomes 0.
inline std::vector<std::uint8_t> to_u8(const image& im, float scale = 255.0f) {
    std::vector<std::uint8_t> o(im.px.size());
    for (std::size_t i = 0; i < o.size(); ++i) { const float v = im.px[i] * scale + 0.5f; o[i] = v >= 255.5f ? 255 : v > 0.0f ? std::uint8_t(v) : 0; }
    return o;
}
inline raster to_raster(const image& im, float scale = 255.0f) { raster r; r.w = im.w; r.h = im.h; r.c = im.c; r.depth = 8; r.u8 = to_u8(im, scale); return r; }
// Interleaved HWC -> planar CHW (the usual model input layout) and back.
inline std::vector<float> hwc_to_chw(const image& im) {
    std::vector<float> o(im.px.size()); const std::size_t n = im.h * im.w;
    for (std::size_t i = 0; i < n; ++i) for (std::size_t k = 0; k < im.c; ++k) o[k * n + i] = im.px[i * im.c + k];
    return o;
}
inline image chw_to_hwc(const float* chw, std::size_t c, std::size_t h, std::size_t w) {
    image im{h, w, c, std::vector<float>(h * w * c)}; const std::size_t n = h * w;
    for (std::size_t i = 0; i < n; ++i) for (std::size_t k = 0; k < c; ++k) im.px[i * c + k] = chw[k * n + i];
    return im;
}

// ---- normalisation ----
// x = (x - mean[k]) / std[k] per channel. mean and std need one entry per channel (or one entry for all).
inline void normalize(image& im, const std::vector<float>& mean, const std::vector<float>& stddev) {
    if ((mean.size() != im.c && mean.size() != 1) || (stddev.size() != im.c && stddev.size() != 1)) throw std::invalid_argument("normalize: mean/std size does not match the channel count");
    for (std::size_t k = 0; k < im.c; ++k) {
        const float m = mean[mean.size() == 1 ? 0 : k], inv = 1.0f / stddev[stddev.size() == 1 ? 0 : k];
        for (std::size_t i = k; i < im.px.size(); i += im.c) im.px[i] = (im.px[i] - m) * inv;
    }
}
inline const std::vector<float>& imagenet_mean() { static const std::vector<float> v{0.485f, 0.456f, 0.406f}; return v; }
inline const std::vector<float>& imagenet_std() { static const std::vector<float> v{0.229f, 0.224f, 0.225f}; return v; }
// Per-channel zero mean and unit (population) variance over the image; a constant channel becomes all zero.
// Returns the means and standard deviations that were removed.
inline std::pair<std::vector<float>, std::vector<float>> standardize(image& im) {
    std::vector<float> mean(im.c), sd(im.c); const std::size_t n = im.h * im.w;
    for (std::size_t k = 0; k < im.c && n; ++k) {
        double s = 0, ss = 0;
        for (std::size_t i = k; i < im.px.size(); i += im.c) s += im.px[i];
        const double m = s / double(n);
        for (std::size_t i = k; i < im.px.size(); i += im.c) { const double d = im.px[i] - m; ss += d * d; }
        const double dv = std::sqrt(ss / double(n));
        mean[k] = float(m); sd[k] = float(dv);
        for (std::size_t i = k; i < im.px.size(); i += im.c) im.px[i] = dv > 0 ? float((im.px[i] - m) / dv) : 0.0f;
    }
    return {mean, sd};
}
// u8 HWC -> normalised float CHW in one pass: (px * scale - mean[k]) / std[k].
inline std::vector<float> preprocess_chw(const std::uint8_t* px, std::size_t h, std::size_t w, std::size_t c, const std::vector<float>& mean, const std::vector<float>& stddev, float scale = 1.0f / 255.0f) {
    if (mean.size() != c || stddev.size() != c) throw std::invalid_argument("preprocess_chw: mean/std size does not match the channel count");
    std::vector<float> o(h * w * c); const std::size_t n = h * w;
    for (std::size_t k = 0; k < c; ++k) { const float m = mean[k], inv = 1.0f / stddev[k]; for (std::size_t i = 0; i < n; ++i) o[k * n + i] = (float(px[i * c + k]) * scale - m) * inv; }
    return o;
}

// ---- resize ----
enum class interp { nearest, bilinear, bicubic, area };
namespace detail {
struct taps { std::vector<std::size_t> start; std::vector<std::size_t> count; std::vector<float> weight; std::size_t width = 0; };
inline double kernel(interp m, double x) {
    if (m == interp::area) return x > -0.5 && x <= 0.5 ? 1 : 0;   // half-open, so a sample on a boundary belongs to one output only
    x = std::fabs(x);
    if (m == interp::bilinear) return x < 1 ? 1 - x : 0;
    const double a = -0.5;   // Keys cubic, a = -0.5 (Catmull-Rom)
    return x < 1 ? ((a + 2) * x - (a + 3)) * x * x + 1 : x < 2 ? (((x - 5) * x + 8) * x - 4) * a : 0;
}
// Weights of the input samples contributing to each of `out` outputs along one axis of length `in`.
// Sample centres are at half-integers; taps outside the image are dropped and the rest renormalised.
inline taps make_taps(std::size_t in, std::size_t out, interp m, bool antialias) {
    taps t; const double scale = double(in) / double(out);
    const double stretch = (antialias || m == interp::area) && scale > 1 ? scale : 1.0;
    const double radius = (m == interp::bicubic ? 2.0 : m == interp::bilinear ? 1.0 : 0.5) * stretch;
    t.width = std::size_t(std::ceil(radius * 2)) + 2;
    t.start.resize(out); t.count.resize(out); t.weight.assign(out * t.width, 0.0f);
    for (std::size_t o = 0; o < out; ++o) {
        const double centre = (double(o) + 0.5) * scale;
        std::ptrdiff_t lo = std::ptrdiff_t(std::floor(centre - radius)), hi = std::ptrdiff_t(std::ceil(centre + radius));
        lo = std::max<std::ptrdiff_t>(lo, 0); hi = std::min<std::ptrdiff_t>(hi, std::ptrdiff_t(in));
        double sum = 0; std::size_t n = 0; std::vector<double> ws;
        for (std::ptrdiff_t i = lo; i < hi && n < t.width; ++i, ++n) { const double wv = kernel(m, (double(i) + 0.5 - centre) / stretch); ws.push_back(wv); sum += wv; }
        if (n == 0 || sum == 0) { lo = std::clamp<std::ptrdiff_t>(std::ptrdiff_t(centre), 0, std::ptrdiff_t(in) - 1); ws.assign(1, 1.0); sum = 1; n = 1; }   // degenerate: nearest sample
        t.start[o] = std::size_t(lo); t.count[o] = n;
        for (std::size_t k = 0; k < n; ++k) t.weight[o * t.width + k] = float(ws[k] / sum);
    }
    return t;
}
} // namespace detail
// Resize to H x W.
//   nearest   the input sample under each output centre: src[floor((o + 0.5) * in / out)]
//   bilinear  triangle filter; bicubic: Keys cubic with a = -0.5; area: box filter (the exact mean of the
//             covered input samples when the reduction factor is an integer)
// antialias widens the bilinear / bicubic filter by the reduction factor when shrinking (as PIL does; area
// always does). With antialias off they read a fixed 2 or 4 taps, which aliases when shrinking a lot.
// Bicubic can overshoot the input range. An empty source or target throws.
inline image resize(const image& s, std::size_t H, std::size_t W, interp m = interp::bilinear, bool antialias = true) {
    if (s.h == 0 || s.w == 0 || H == 0 || W == 0) throw std::invalid_argument("resize: empty image or target");
    image d{H, W, s.c, std::vector<float>(H * W * s.c)};
    if (m == interp::nearest) {
        for (std::size_t y = 0; y < H; ++y) { const std::size_t sy = std::min(s.h - 1, std::size_t((double(y) + 0.5) * double(s.h) / double(H)));
            for (std::size_t x = 0; x < W; ++x) { const std::size_t sx = std::min(s.w - 1, std::size_t((double(x) + 0.5) * double(s.w) / double(W)));
                for (std::size_t k = 0; k < s.c; ++k) d.px[(y * W + x) * s.c + k] = s.px[(sy * s.w + sx) * s.c + k]; } }
        return d;
    }
    const auto tx = detail::make_taps(s.w, W, m, antialias), ty = detail::make_taps(s.h, H, m, antialias);
    std::vector<float> tmp(s.h * W * s.c);   // horizontal pass
    for (std::size_t y = 0; y < s.h; ++y) for (std::size_t x = 0; x < W; ++x) for (std::size_t k = 0; k < s.c; ++k) {
        float acc = 0; const float* wt = tx.weight.data() + x * tx.width;
        for (std::size_t q = 0; q < tx.count[x]; ++q) acc += wt[q] * s.px[(y * s.w + tx.start[x] + q) * s.c + k];
        tmp[(y * W + x) * s.c + k] = acc;
    }
    for (std::size_t y = 0; y < H; ++y) { const float* wt = ty.weight.data() + y * ty.width;
        for (std::size_t i = 0; i < W * s.c; ++i) { float acc = 0; for (std::size_t q = 0; q < ty.count[y]; ++q) acc += wt[q] * tmp[(ty.start[y] + q) * W * s.c + i]; d.px[y * W * s.c + i] = acc; } }
    return d;
}

// ---- crop / pad / letterbox ----
inline image crop(const image& s, std::size_t y0, std::size_t x0, std::size_t h, std::size_t w) {
    if (y0 > s.h || x0 > s.w || h > s.h - y0 || w > s.w - x0) throw std::out_of_range("crop: rectangle is outside the image");
    image d{h, w, s.c, std::vector<float>(h * w * s.c)};
    for (std::size_t y = 0; y < h; ++y) std::copy_n(s.px.data() + ((y0 + y) * s.w + x0) * s.c, w * s.c, d.px.data() + y * w * s.c);
    return d;
}
enum class pad_mode { constant, edge, reflect };   // reflect: mirror without repeating the border sample
inline image pad(const image& s, std::size_t top, std::size_t bottom, std::size_t left, std::size_t right, pad_mode mode = pad_mode::constant, float value = 0.0f) {
    if (mode != pad_mode::constant && (s.h == 0 || s.w == 0)) throw std::invalid_argument("pad: edge/reflect need a non-empty image");
    image d{s.h + top + bottom, s.w + left + right, s.c, std::vector<float>((s.h + top + bottom) * (s.w + left + right) * s.c, value)};
    auto src = [&](std::ptrdiff_t i, std::size_t n) -> std::size_t {
        const std::ptrdiff_t N = std::ptrdiff_t(n);
        if (mode == pad_mode::edge || N == 1) return std::size_t(std::clamp<std::ptrdiff_t>(i, 0, N - 1));
        const std::ptrdiff_t period = 2 * (N - 1); i %= period; if (i < 0) i += period;
        return std::size_t(i < N ? i : period - i);
    };
    for (std::size_t y = 0; y < d.h; ++y) for (std::size_t x = 0; x < d.w; ++x) {
        const std::ptrdiff_t sy = std::ptrdiff_t(y) - std::ptrdiff_t(top), sx = std::ptrdiff_t(x) - std::ptrdiff_t(left);
        const bool inside = sy >= 0 && sx >= 0 && sy < std::ptrdiff_t(s.h) && sx < std::ptrdiff_t(s.w);
        if (!inside && mode == pad_mode::constant) continue;
        const std::size_t yy = inside ? std::size_t(sy) : src(sy, s.h), xx = inside ? std::size_t(sx) : src(sx, s.w);
        for (std::size_t k = 0; k < s.c; ++k) d.px[(y * d.w + x) * s.c + k] = s.px[(yy * s.w + xx) * s.c + k];
    }
    return d;
}
// Fit the image inside H x W keeping its aspect ratio, centred, with `fill` around it (the detector input
// convention). scale and the offsets map boxes back: x_src = (x_dst - pad_x) / scale.
struct letterboxed { image img; float scale = 1; std::size_t pad_x = 0, pad_y = 0, inner_w = 0, inner_h = 0; };
inline letterboxed letterbox(const image& s, std::size_t H, std::size_t W, float fill = 114.0f / 255.0f, interp m = interp::bilinear) {
    if (s.h == 0 || s.w == 0 || H == 0 || W == 0) throw std::invalid_argument("letterbox: empty image or target");
    letterboxed o; o.scale = std::min(float(W) / float(s.w), float(H) / float(s.h));
    o.inner_w = std::clamp<std::size_t>(std::size_t(std::lround(double(s.w) * o.scale)), 1, W); o.inner_h = std::clamp<std::size_t>(std::size_t(std::lround(double(s.h) * o.scale)), 1, H);
    o.pad_x = (W - o.inner_w) / 2; o.pad_y = (H - o.inner_h) / 2;
    o.img = pad(resize(s, o.inner_h, o.inner_w, m), o.pad_y, H - o.inner_h - o.pad_y, o.pad_x, W - o.inner_w - o.pad_x, pad_mode::constant, fill);
    return o;
}

// ---- colour ----
// Swap channels 0 and 2 in place (RGB <-> BGR, RGBA <-> BGRA).
inline void swap_rb(image& im) { if (im.c < 3) throw std::invalid_argument("swap_rb: needs at least 3 channels"); for (std::size_t i = 0; i + 2 < im.px.size(); i += im.c) std::swap(im.px[i], im.px[i + 2]); }
// Luma, BT.601 weights (0.299, 0.587, 0.114); an alpha channel is dropped.
inline image rgb_to_gray(const image& s) {
    if (s.c < 3) throw std::invalid_argument("rgb_to_gray: needs at least 3 channels");
    image d{s.h, s.w, 1, std::vector<float>(s.h * s.w)};
    for (std::size_t i = 0; i < d.px.size(); ++i) d.px[i] = 0.299f * s.px[i * s.c] + 0.587f * s.px[i * s.c + 1] + 0.114f * s.px[i * s.c + 2];
    return d;
}
inline image gray_to_rgb(const image& s) {
    if (s.c != 1) throw std::invalid_argument("gray_to_rgb: needs 1 channel");
    image d{s.h, s.w, 3, std::vector<float>(s.h * s.w * 3)};
    for (std::size_t i = 0; i < s.px.size(); ++i) d.px[3 * i] = d.px[3 * i + 1] = d.px[3 * i + 2] = s.px[i];
    return d;
}
// 8-bit video YCbCr. matrix: BT.601 or BT.709; range: limited (Y 16..235, C 16..240) or full (0..255).
struct yuv_spec { bool bt709 = false, full_range = false; };
namespace detail {
struct yuv_coef { float kr, kb, yoff, yscale, cscale; };
inline yuv_coef coef(yuv_spec s) { return {s.bt709 ? 0.2126f : 0.299f, s.bt709 ? 0.0722f : 0.114f, s.full_range ? 0.0f : 16.0f, s.full_range ? 1.0f : 219.0f / 255.0f, s.full_range ? 1.0f : 224.0f / 255.0f}; }
inline std::uint8_t clamp8(float v) { v += 0.5f; return v >= 255.5f ? 255 : v > 0.0f ? std::uint8_t(v) : 0; }
inline void yuv_px(yuv_coef k, float y, float u, float v, std::uint8_t* rgb) {
    const float Y = (y - k.yoff) / k.yscale, U = (u - 128.0f) / k.cscale, V = (v - 128.0f) / k.cscale, kg = 1.0f - k.kr - k.kb;
    const float r = Y + 2 * (1 - k.kr) * V, b = Y + 2 * (1 - k.kb) * U, g = (Y - k.kr * r - k.kb * b) / kg;
    rgb[0] = clamp8(r); rgb[1] = clamp8(g); rgb[2] = clamp8(b);
}
inline void rgb_px(yuv_coef k, const std::uint8_t* rgb, float& y, float& u, float& v) {
    const float r = rgb[0], g = rgb[1], b = rgb[2], Y = k.kr * r + (1 - k.kr - k.kb) * g + k.kb * b;
    y = Y * k.yscale + k.yoff; u = (b - Y) / (2 * (1 - k.kb)) * k.cscale + 128.0f; v = (r - Y) / (2 * (1 - k.kr)) * k.cscale + 128.0f;
}
} // namespace detail
// Planar 4:2:0 (I420) -> interleaved RGB. Chroma planes are ((w + 1) / 2) x ((h + 1) / 2); each chroma sample
// is replicated over its 2 x 2 block (no chroma interpolation). Strides are in bytes per row.
inline void yuv420_to_rgb(const std::uint8_t* y, std::size_t y_stride, const std::uint8_t* u, const std::uint8_t* v, std::size_t c_stride, std::size_t w, std::size_t h, std::uint8_t* rgb, yuv_spec spec = {}) {
    const auto k = detail::coef(spec);
    for (std::size_t j = 0; j < h; ++j) for (std::size_t i = 0; i < w; ++i) detail::yuv_px(k, y[j * y_stride + i], u[(j / 2) * c_stride + i / 2], v[(j / 2) * c_stride + i / 2], rgb + (j * w + i) * 3);
}
// NV12: a Y plane and one interleaved UV plane of ((w + 1) / 2) pairs per row.
inline void nv12_to_rgb(const std::uint8_t* y, std::size_t y_stride, const std::uint8_t* uv, std::size_t uv_stride, std::size_t w, std::size_t h, std::uint8_t* rgb, yuv_spec spec = {}) {
    const auto k = detail::coef(spec);
    for (std::size_t j = 0; j < h; ++j) for (std::size_t i = 0; i < w; ++i) detail::yuv_px(k, y[j * y_stride + i], uv[(j / 2) * uv_stride + (i / 2) * 2], uv[(j / 2) * uv_stride + (i / 2) * 2 + 1], rgb + (j * w + i) * 3);
}
// Interleaved RGB -> tightly packed planes (strides w and (w + 1) / 2); chroma is the mean over each 2 x 2
// block (smaller at odd right / bottom edges).
inline void rgb_to_yuv420(const std::uint8_t* rgb, std::size_t w, std::size_t h, std::uint8_t* y, std::uint8_t* u, std::uint8_t* v, yuv_spec spec = {}) {
    const auto k = detail::coef(spec); const std::size_t cw = (w + 1) / 2, ch = (h + 1) / 2;
    std::vector<float> us(cw * ch, 0.0f), vs(cw * ch, 0.0f); std::vector<std::uint8_t> cnt(cw * ch, 0);
    for (std::size_t j = 0; j < h; ++j) for (std::size_t i = 0; i < w; ++i) {
        float Y, U, V; detail::rgb_px(k, rgb + (j * w + i) * 3, Y, U, V);
        y[j * w + i] = detail::clamp8(Y); const std::size_t q = (j / 2) * cw + i / 2; us[q] += U; vs[q] += V; ++cnt[q];
    }
    for (std::size_t q = 0; q < cw * ch; ++q) { u[q] = detail::clamp8(us[q] / float(cnt[q])); v[q] = detail::clamp8(vs[q] / float(cnt[q])); }
}
inline void rgb_to_nv12(const std::uint8_t* rgb, std::size_t w, std::size_t h, std::uint8_t* y, std::uint8_t* uv, yuv_spec spec = {}) {
    const std::size_t n = ((w + 1) / 2) * ((h + 1) / 2); std::vector<std::uint8_t> u(n), v(n);
    rgb_to_yuv420(rgb, w, h, y, u.data(), v.data(), spec);
    for (std::size_t q = 0; q < n; ++q) { uv[2 * q] = u[q]; uv[2 * q + 1] = v[q]; }
}

} // namespace forge::vision
