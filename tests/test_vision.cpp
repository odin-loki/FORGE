// vision: image codecs (codec.hpp) against the vectors of tools/gen_vision_vectors.py, round trips, malformed
// input, and the pre-processing helpers (transform.hpp).
#include "check.hpp"
#include <forge/vision/transform.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
using namespace forge;
using vision::image; using vision::interp; using vision::raster;

namespace {
std::vector<std::uint8_t> slurp(const std::filesystem::path& p) { std::ifstream f(p, std::ios::binary); return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()}; }
bool same(const raster& a, const raster& b) { return a.w == b.w && a.h == b.h && a.c == b.c && a.depth == b.depth && a.u8 == b.u8 && a.u16 == b.u16; }
raster expected(const std::vector<std::uint8_t>& raw) {
    raster r; r.w = vision::codec::le32(raw.data()); r.h = vision::codec::le32(raw.data() + 4); r.c = vision::codec::le32(raw.data() + 8); r.depth = vision::codec::le32(raw.data() + 12);
    if (r.depth == 8) r.u8.assign(raw.begin() + 16, raw.end());
    else for (std::size_t i = 16; i + 1 < raw.size(); i += 2) r.u16.push_back(std::uint16_t(raw[i] | raw[i + 1] << 8));
    return r;
}
raster random_raster(std::size_t w, std::size_t h, std::size_t c, unsigned depth, int mode, unsigned seed) {   // mode 0 noise, 1 smooth, 2 flat
    std::mt19937 g(seed); raster r; r.w = w; r.h = h; r.c = c; r.depth = depth; const unsigned maxv = depth == 8 ? 255 : 65535;
    for (std::size_t y = 0; y < h; ++y) for (std::size_t x = 0; x < w; ++x) for (std::size_t k = 0; k < c; ++k) {
        unsigned v = mode == 0 ? unsigned(g()) & maxv : mode == 1 ? unsigned(x * 3 + y * 5 + k * 40) % (maxv + 1) : 200;
        if (depth == 8) r.u8.push_back(std::uint8_t(v)); else r.u16.push_back(std::uint16_t(v));
    }
    return r;
}
template <class F> bool rejects(F&& f) { try { f(); } catch (const forge_exception&) { return true; } return false; }
const std::filesystem::path kDir = "tests/data/vision";
}

TEST(codec_checksums_and_zlib_round_trip) {
    const char* s = "123456789";
    CHECK(vision::crc32(reinterpret_cast<const std::uint8_t*>(s), 9) == 0xCBF43926u);
    CHECK(vision::crc32(reinterpret_cast<const std::uint8_t*>(s) + 4, 5, vision::crc32(reinterpret_cast<const std::uint8_t*>(s), 4)) == 0xCBF43926u);   // running
    CHECK(vision::adler32(reinterpret_cast<const std::uint8_t*>("Wikipedia"), 9) == 0x11E60398u);
    std::mt19937 g(1);
    for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(259), std::size_t(70000), std::size_t(300000)})
        for (int mode = 0; mode < 4; ++mode) {
            std::vector<std::uint8_t> d(n);
            for (std::size_t i = 0; i < n; ++i) d[i] = mode == 0 ? std::uint8_t(g()) : mode == 1 ? std::uint8_t("abcabcabd"[i % 9]) : mode == 2 ? 0 : std::uint8_t((i / 300) * 7 + (g() % 3 == 0));
            auto z = vision::zlib_compress(d.data(), d.size());
            auto back = vision::zlib_decompress(z.data(), z.size(), d.size());
            CHECK_MSG(back == d, "zlib round trip n=%zu mode=%d", n, mode);
            if (mode == 0) CHECK(z.size() <= n + 5 * (n / 65535 + 1) + 6);                 // noise falls back to stored blocks
            if (mode == 2 && n > 1000) CHECK(z.size() < n / 50);                           // zeros compress
            if (n) CHECK(rejects([&] { vision::zlib_decompress(z.data(), z.size(), d.size() - 1); }));   // output cap
            if (n > 10) { auto bad = z; bad[bad.size() - 1] ^= 1; CHECK(rejects([&] { vision::zlib_decompress(bad.data(), bad.size(), d.size()); })); }   // Adler-32
        }
    // streams made by zlib itself (stored, fixed and dynamic blocks)
    std::string text; for (int i = 0; i < 20; ++i) text += "the quick brown fox jumps over the lazy dog. ";
    for (int i = 0; i < 256; ++i) text += char(i);
    int seen = 0;
    for (const char* f : {"text_l0.z", "text_l1.z", "text_l6.z", "text_l9.z", "fixed.z", "empty.z"}) {
        auto z = slurp(std::filesystem::path("fuzz/corpus/inflate") / f); if (z.empty()) continue;
        ++seen; auto out = vision::zlib_decompress(z.data(), z.size(), 1 << 20);
        CHECK_MSG(std::string(out.begin(), out.end()) == (std::string(f) == "empty.z" ? std::string() : text), "%s", f);
    }
    CHECK(seen == 6);
}

TEST(codec_decodes_every_reference_vector) {
    int n = 0, png = 0, bmp = 0, pnm = 0;
    for (auto& e : std::filesystem::directory_iterator(kDir)) {
        if (e.path().extension() == ".raw") continue;
        auto data = slurp(e.path()); auto want = expected(slurp(e.path().string() + ".raw"));
        try {
            auto got = vision::decode_image(data);
            CHECK_MSG(same(got, want), "%s: decoded pixels differ (%zux%zu c=%zu depth=%u)", e.path().filename().string().c_str(), got.w, got.h, got.c, got.depth);
        } catch (const std::exception& ex) { CHECK_MSG(false, "%s: %s", e.path().filename().string().c_str(), ex.what()); }
        ++n; const auto name = e.path().filename().string(); png += name.rfind("png_", 0) == 0; bmp += name.rfind("bmp_", 0) == 0; pnm += name.rfind("pnm_", 0) == 0;
    }
    CHECK_MSG(n >= 99 && png >= 80 && bmp >= 11 && pnm >= 6, "found %d vectors (%d png, %d bmp, %d pnm) under tests/data/vision", n, png, bmp, pnm);
}

TEST(codec_encode_decode_round_trips) {
    unsigned seed = 1;
    for (auto [w, h] : {std::pair<std::size_t, std::size_t>{1, 1}, {1, 9}, {9, 1}, {17, 13}, {300, 211}})
        for (std::size_t c = 1; c <= 4; ++c) for (unsigned depth : {8u, 16u}) for (int mode = 0; mode < 3; ++mode) {
            if (w == 300 && mode == 0 && depth == 16) continue;
            auto r = random_raster(w, h, c, depth, mode, seed++);
            auto p = vision::encode_png(r);
            CHECK_MSG(same(vision::decode_png(p), r), "png round trip %zux%zu c=%zu depth=%u mode=%d", w, h, c, depth, mode);
            if (mode == 2 && w == 300) CHECK(p.size() < r.samples() * depth / 8 / 4);   // a flat image must actually compress
            if (c == 1 || c == 3) CHECK(same(vision::decode_pnm(vision::encode_pnm(r)), r));
            if (depth == 8 && c != 2) {
                auto back = vision::decode_bmp(vision::encode_bmp(r));
                if (c == 1) { bool ok = back.c == 3 && back.w == w && back.h == h; for (std::size_t i = 0; ok && i < r.u8.size(); ++i) ok = back.u8[3 * i] == r.u8[i] && back.u8[3 * i + 1] == r.u8[i] && back.u8[3 * i + 2] == r.u8[i]; CHECK(ok); }
                else CHECK_MSG(same(back, r), "bmp round trip %zux%zu c=%zu", w, h, c);
            }
        }
    raster bad; bad.w = 2; bad.h = 2; bad.c = 3; bad.u8.resize(5);
    CHECK_THROWS(vision::encode_png(bad)); CHECK_THROWS(vision::encode_bmp(bad)); CHECK_THROWS(vision::encode_pnm(bad));
    bad.c = 2; bad.u8.resize(8); CHECK_THROWS(vision::encode_bmp(bad)); CHECK_THROWS(vision::encode_pnm(bad));
}

TEST(codec_rejects_malformed_input_without_crashing) {
    int checked = 0;
    for (auto& e : std::filesystem::directory_iterator(kDir)) {
        if (e.path().extension() == ".raw") continue;
        auto data = slurp(e.path()); if (data.size() > 900) continue;
        const std::string name = e.path().filename().string(); const bool text = name.find("_p2") != std::string::npos || name.find("_p3") != std::string::npos;
        ++checked;
        // every proper prefix: PNG and BMP must be rejected; a cut text PNM may still parse, but must not crash
        for (std::size_t cut = 0; cut < data.size(); ++cut) {
            std::vector<std::uint8_t> part(data.begin(), data.begin() + std::ptrdiff_t(cut));   // exactly sized: ASan sees any over-read
            bool threw = rejects([&] { (void)vision::decode_image(part); });
            if (!text) CHECK_MSG(threw, "%s: prefix of %zu/%zu bytes was accepted", name.c_str(), cut, data.size());
        }
        // one byte inverted anywhere in a PNG breaks the signature, a length, a type or a CRC
        if (name.rfind("png_", 0) == 0) for (std::size_t i = 0; i < data.size(); ++i) {
            auto mut = data; mut[i] ^= 0xff;
            CHECK_MSG(rejects([&] { (void)vision::decode_png(mut); }), "%s: byte %zu inverted but accepted", name.c_str(), i);
        }
        // BMP / PNM: any single-byte change either decodes or is rejected cleanly
        else for (std::size_t i = 0; i < data.size(); ++i) for (std::uint8_t v : {std::uint8_t(0), std::uint8_t(0xff), std::uint8_t(data[i] + 1)}) {
            auto mut = data; mut[i] = v; try { (void)vision::decode_image(mut, {.max_pixels = 1 << 16}); } catch (const forge_exception&) {}
        }
    }
    CHECK(checked > 60);
    // declared size far beyond the data, and beyond the limit
    auto big = vision::encode_png(random_raster(64, 64, 3, 8, 2, 7));
    CHECK(rejects([&] { (void)vision::decode_png(big, {.max_pixels = 64 * 64 - 1}); }));
    CHECK(!rejects([&] { (void)vision::decode_png(big, {.max_pixels = 64 * 64}); }));
    auto huge = big; huge[16] = 0x7f; huge[20] = 0x7f;   // width and height 0x7f000040: IHDR CRC now wrong
    CHECK(rejects([&] { (void)vision::decode_png(huge); }));
    std::uint32_t crc = vision::crc32(huge.data() + 12, 17); huge[29] = std::uint8_t(crc >> 24); huge[30] = std::uint8_t(crc >> 16); huge[31] = std::uint8_t(crc >> 8); huge[32] = std::uint8_t(crc);
    CHECK(rejects([&] { (void)vision::decode_png(huge); }));                                  // CRC fixed: refused by the pixel limit
    CHECK(rejects([&] { (void)vision::decode_png(huge, {.max_pixels = std::size_t(-1)}); })); // limit lifted: refused because the data is too short (no giant allocation)
    const std::uint8_t jpg[4] = {0xFF, 0xD8, 0xFF, 0xE0};
    try { (void)vision::decode_image(jpg); CHECK(false); } catch (const forge_exception& ex) { CHECK(ex.code == errc::unsupported); }
    CHECK(rejects([] { (void)vision::decode_image({}); })); CHECK(rejects([] { const std::uint8_t x[3] = {'P', '4', ' '}; (void)vision::decode_image(x); }));
    const std::string bomb = "P6 100000 100000 255\n"; CHECK(rejects([&] { (void)vision::decode_pnm({reinterpret_cast<const std::uint8_t*>(bomb.data()), bomb.size()}); }));
}

TEST(transform_formats_layouts_normalisation) {
    const std::uint8_t px[12] = {0, 255, 128, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    auto im = vision::to_float(px, 2, 2, 3);
    CHECK(im.h == 2 && im.w == 2 && im.c == 3 && im.px[1] == 1.0f && std::fabs(im.px[2] - 128.0f / 255.0f) < 1e-7f);
    auto u = vision::to_u8(im); CHECK(std::equal(u.begin(), u.end(), px));
    image edge{1, 1, 5, {-0.2f, 1.7f, 0.5f / 255.0f, 0.49f / 255.0f, std::nanf("")}};
    CHECK((vision::to_u8(edge) == std::vector<std::uint8_t>{0, 255, 1, 0, 0}));
    auto chw = vision::hwc_to_chw(im);
    CHECK(chw[0] == im.px[0] && chw[1] == im.px[3] && chw[4] == im.px[1] && chw[8 + 3] == im.px[11]);
    auto back = vision::chw_to_hwc(chw.data(), 3, 2, 2); CHECK(back.px == im.px);
    auto nrm = im; vision::normalize(nrm, {0.5f, 0.25f, 0.0f}, {0.5f, 0.25f, 2.0f});
    CHECK(std::fabs(nrm.px[0] + 1.0f) < 1e-6f && std::fabs(nrm.px[1] - 3.0f) < 1e-6f && std::fabs(nrm.px[2] - 128.0f / 255.0f / 2.0f) < 1e-6f);
    auto one = im; vision::normalize(one, {0.5f}, {0.5f}); CHECK(std::fabs(one.px[1] - 1.0f) < 1e-6f);
    CHECK_THROWS(vision::normalize(nrm, {0.f, 0.f}, {1.f, 1.f}));
    auto pre = vision::preprocess_chw(px, 2, 2, 3, vision::imagenet_mean(), vision::imagenet_std());
    auto ref = im; vision::normalize(ref, vision::imagenet_mean(), vision::imagenet_std()); auto refc = vision::hwc_to_chw(ref);
    double e = 0; for (std::size_t i = 0; i < pre.size(); ++i) e = std::max(e, double(std::fabs(pre[i] - refc[i]))); CHECK(e < 1e-6);
    std::mt19937 g(3); std::normal_distribution<float> d(5.0f, 3.0f);
    image s{16, 16, 2, std::vector<float>(512)}; for (std::size_t i = 0; i < 512; ++i) s.px[i] = i % 2 ? 7.0f : d(g);   // channel 1 constant
    auto [mean, sd] = vision::standardize(s);
    double m = 0, v = 0; for (std::size_t i = 0; i < 512; i += 2) m += s.px[i]; m /= 256; for (std::size_t i = 0; i < 512; i += 2) v += (s.px[i] - m) * (s.px[i] - m); v /= 256;
    CHECK(std::fabs(m) < 1e-5 && std::fabs(v - 1) < 1e-4); CHECK(std::fabs(mean[0] - 5.0f) < 1.0f && std::fabs(sd[0] - 3.0f) < 1.0f && mean[1] == 7.0f && sd[1] == 0.0f && s.px[1] == 0.0f);
    auto r16 = random_raster(3, 2, 1, 16, 2, 1); r16.u16[0] = 65535; auto f16 = vision::to_float(r16); CHECK(f16.px[0] == 1.0f && std::fabs(f16.px[1] - 200.0f / 65535.0f) < 1e-7f);
    CHECK(same(vision::to_raster(vision::to_float(random_raster(5, 4, 3, 8, 0, 9))), random_raster(5, 4, 3, 8, 0, 9)));
}

TEST(transform_resize) {
    image s{5, 6, 1, {}}; for (std::size_t y = 0; y < 5; ++y) for (std::size_t x = 0; x < 6; ++x) s.px.push_back(float((y * 7 + x * 3) % 11));
    // Pillow 12.3 Image.resize on this 5 x 6 image to 4 x 3 (it antialiases when shrinking; BICUBIC is a = -0.5, BOX is area)
    struct pil { interp m; std::vector<float> v; };
    const pil ref[] = {
        {interp::bilinear, {3.50000f, 5.81250f, 3.60714f, 6.55952f, 4.92708f, 4.44048f, 5.55952f, 5.07292f, 3.44048f, 6.39286f, 4.18750f, 6.50000f}},
        {interp::bicubic, {3.05498f, 6.27920f, 3.29765f, 6.97267f, 4.74815f, 4.57613f, 5.42387f, 5.25185f, 3.02733f, 6.70235f, 3.72080f, 6.94502f}},
        {interp::area, {1.50000f, 7.50000f, 2.50000f, 6.50000f, 4.25000f, 4.75000f, 6.00000f, 6.50000f, 1.50000f, 7.50000f, 2.50000f, 8.50000f}},
    };
    for (auto& r : ref) {
        auto d = vision::resize(s, 4, 3, r.m); double e = 0; for (std::size_t i = 0; i < 12; ++i) e = std::max(e, double(std::fabs(d.px[i] - r.v[i])));
        CHECK_MSG(e < 2e-4, "resize mode %d differs from Pillow by %g", int(r.m), e);
    }
    std::mt19937 g(5); image rnd{9, 7, 3, std::vector<float>(9 * 7 * 3)}; for (auto& v : rnd.px) v = float(g() % 1000) / 1000.0f;
    for (interp m : {interp::nearest, interp::bilinear, interp::bicubic, interp::area}) for (bool aa : {false, true}) {
        auto same_size = vision::resize(rnd, 9, 7, m, aa); double e = 0; for (std::size_t i = 0; i < rnd.px.size(); ++i) e = std::max(e, double(std::fabs(same_size.px[i] - rnd.px[i])));
        CHECK_MSG(e < 1e-6, "identity resize mode %d aa %d err %g", int(m), int(aa), e);
        image flat{4, 5, 2, std::vector<float>(40, 0.37f)};   // weights sum to one: a constant image stays constant at any size
        for (auto [H, W] : {std::pair<std::size_t, std::size_t>{1, 1}, {3, 2}, {11, 17}, {4, 50}}) { auto d = vision::resize(flat, H, W, m, aa); double fe = 0; for (float v : d.px) fe = std::max(fe, double(std::fabs(v - 0.37f))); CHECK(d.h == H && d.w == W && d.c == 2 && fe < 1e-6); }
    }
    auto up = vision::resize(s, 10, 12, interp::nearest);   // exact 2x: every source sample replicated 2 x 2
    bool ok = true; for (std::size_t y = 0; y < 10; ++y) for (std::size_t x = 0; x < 12; ++x) ok &= up.at(y, x, 0) == s.at(y / 2, x / 2, 0); CHECK(ok);
    image e4{4, 4, 1, {}}; for (int i = 0; i < 16; ++i) e4.px.push_back(float(i));
    auto half = vision::resize(e4, 2, 2, interp::area);     // integer factor: the exact block means
    CHECK(half.px[0] == 2.5f && half.px[1] == 4.5f && half.px[2] == 10.5f && half.px[3] == 12.5f);
    auto old = vision::resize_bilinear(rnd, 14, 5), now = vision::resize(rnd, 14, 5, interp::bilinear, false);   // no antialias == the older helper
    double e = 0; for (std::size_t i = 0; i < old.px.size(); ++i) e = std::max(e, double(std::fabs(old.px[i] - now.px[i]))); CHECK_MSG(e < 1e-5, "vs resize_bilinear %g", e);
    image ramp{1, 8, 1, {0, 1, 2, 3, 4, 5, 6, 7}};          // upscaling a linear ramp stays linear away from the borders
    for (interp m : {interp::bilinear, interp::bicubic}) { auto d = vision::resize(ramp, 1, 16, m); double re = 0; for (std::size_t x = 4; x < 12; ++x) re = std::max(re, std::fabs(double(d.px[x]) - ((double(x) + 0.5) / 2 - 0.5))); CHECK_MSG(re < 1e-5, "ramp mode %d err %g", int(m), re); }
    image cb{8, 8, 1, {}}; for (int i = 0; i < 64; ++i) cb.px.push_back(float((i / 8 + i % 8) & 1));   // checkerboard / 2 with antialias: grey
    for (interp m : {interp::bilinear, interp::area}) {   // (bilinear: interior only; at the border the filter is cut off and renormalised)
        auto d = vision::resize(cb, 4, 4, m, true); double ce = 0;
        for (std::size_t y = 0; y < 4; ++y) for (std::size_t x = 0; x < 4; ++x) if (m == interp::area || (y > 0 && y < 3 && x > 0 && x < 3)) ce = std::max(ce, double(std::fabs(d.at(y, x, 0) - 0.5f)));
        CHECK_MSG(ce < 1e-6, "checkerboard mode %d err %g", int(m), ce);
    }
    CHECK_THROWS(vision::resize(s, 0, 3)); CHECK_THROWS(vision::resize(image{}, 2, 2));
}

TEST(transform_crop_pad_letterbox_colour) {
    image s{3, 4, 1, {0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23}};
    auto c = vision::crop(s, 1, 1, 2, 2); CHECK((c.px == std::vector<float>{11, 12, 21, 22}));
    CHECK(vision::crop(s, 0, 0, 3, 4).px == s.px); CHECK(vision::crop(s, 3, 4, 0, 0).px.empty());
    CHECK_THROWS(vision::crop(s, 2, 0, 2, 1)); CHECK_THROWS(vision::crop(s, 0, 5, 1, 1)); CHECK_THROWS(vision::crop(s, 0, 0, std::size_t(-1), 1));
    auto pc = vision::pad(s, 1, 0, 2, 1, vision::pad_mode::constant, -1.0f);
    CHECK(pc.h == 4 && pc.w == 7 && pc.at(0, 0, 0) == -1 && pc.at(1, 2, 0) == 0 && pc.at(3, 5, 0) == 23 && pc.at(3, 6, 0) == -1 && pc.at(1, 1, 0) == -1);
    auto pe = vision::pad(s, 1, 1, 2, 2, vision::pad_mode::edge);
    CHECK(pe.at(0, 0, 0) == 0 && pe.at(0, 7, 0) == 3 && pe.at(4, 0, 0) == 20 && pe.at(4, 7, 0) == 23 && pe.at(2, 1, 0) == 10);
    auto pr = vision::pad(s, 2, 2, 3, 3, vision::pad_mode::reflect);   // rows 2,1,0,1,2,1,0 and columns 3,2,1,0,1,2,3,2,1,0
    CHECK(pr.at(0, 0, 0) == 23 && pr.at(0, 3, 0) == 20 && pr.at(2, 0, 0) == 3 && pr.at(6, 9, 0) == 0 && pr.at(5, 7, 0) == 12);
    image wide{10, 40, 3, std::vector<float>(1200, 1.0f)};
    auto lb = vision::letterbox(wide, 32, 32, 0.25f);
    CHECK(lb.img.h == 32 && lb.img.w == 32 && lb.inner_w == 32 && lb.inner_h == 8 && lb.pad_x == 0 && lb.pad_y == 12 && std::fabs(lb.scale - 0.8f) < 1e-6f);
    CHECK(lb.img.at(0, 0, 0) == 0.25f && lb.img.at(11, 31, 2) == 0.25f && std::fabs(lb.img.at(12, 0, 0) - 1.0f) < 1e-6f && std::fabs(lb.img.at(19, 31, 1) - 1.0f) < 1e-6f && lb.img.at(20, 5, 0) == 0.25f);
    auto tall = vision::letterbox(image{40, 10, 1, std::vector<float>(400, 0.5f)}, 20, 30);
    CHECK(tall.inner_h == 20 && tall.inner_w == 5 && tall.pad_x == 12 && tall.pad_y == 0);
    image rgb{1, 2, 3, {0.1f, 0.2f, 0.3f, 1.0f, 0.0f, 0.0f}}; auto bgr = rgb; vision::swap_rb(bgr);
    CHECK((bgr.px == std::vector<float>{0.3f, 0.2f, 0.1f, 0.0f, 0.0f, 1.0f}));
    auto gray = vision::rgb_to_gray(rgb); CHECK(std::fabs(gray.px[0] - (0.0299f + 0.1174f + 0.0342f)) < 1e-6f && std::fabs(gray.px[1] - 0.299f) < 1e-7f);
    auto g3 = vision::gray_to_rgb(gray); CHECK(g3.c == 3 && g3.px[3] == gray.px[1] && g3.px[5] == gray.px[1]);
    CHECK_THROWS(vision::swap_rb(gray)); CHECK_THROWS(vision::rgb_to_gray(gray)); CHECK_THROWS(vision::gray_to_rgb(rgb));
}

TEST(transform_yuv420_and_nv12) {
    // Reference points: limited-range BT.601 red is (81, 90, 240), BT.709 red is (63, 102, 240); white is Y 235
    // (limited) or 255 (full) with neutral chroma.
    const std::uint8_t red[12] = {255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0}, white[12] = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    std::uint8_t y[4], u[1], v[1], uv[2], back[12];
    vision::rgb_to_yuv420(red, 2, 2, y, u, v); CHECK(y[0] == 81 && y[3] == 81 && u[0] == 90 && v[0] == 240);
    vision::rgb_to_yuv420(red, 2, 2, y, u, v, {.bt709 = true}); CHECK(y[0] == 63 && u[0] == 102 && v[0] == 240);
    vision::rgb_to_yuv420(white, 2, 2, y, u, v); CHECK(y[0] == 235 && u[0] == 128 && v[0] == 128);
    vision::rgb_to_yuv420(white, 2, 2, y, u, v, {.full_range = true}); CHECK(y[0] == 255 && u[0] == 128 && v[0] == 128);
    vision::rgb_to_nv12(red, 2, 2, y, uv); CHECK(uv[0] == 90 && uv[1] == 240);
    vision::nv12_to_rgb(y, 2, uv, 2, 2, 2, back); CHECK(back[0] >= 253 && back[1] <= 2 && back[2] <= 2);
    // round trip of images that are constant over each 2 x 2 block (so chroma subsampling loses nothing), odd size
    std::mt19937 g(4);
    for (auto spec : {vision::yuv_spec{}, vision::yuv_spec{.bt709 = true}, vision::yuv_spec{.full_range = true}, vision::yuv_spec{.bt709 = true, .full_range = true}}) {
        const std::size_t w = 7, h = 5, cw = 4, ch = 3; std::vector<std::uint8_t> rgb(w * h * 3), blk(cw * ch * 3); for (auto& b : blk) b = std::uint8_t(g());
        for (std::size_t j = 0; j < h; ++j) for (std::size_t i = 0; i < w; ++i) for (int k = 0; k < 3; ++k) rgb[(j * w + i) * 3 + std::size_t(k)] = blk[((j / 2) * cw + i / 2) * 3 + std::size_t(k)];
        std::vector<std::uint8_t> Y(w * h), U(cw * ch), V(cw * ch), UV(cw * ch * 2), out(w * h * 3), out2(w * h * 3);
        vision::rgb_to_yuv420(rgb.data(), w, h, Y.data(), U.data(), V.data(), spec);
        vision::yuv420_to_rgb(Y.data(), w, U.data(), V.data(), cw, w, h, out.data(), spec);
        int worst = 0; for (std::size_t i = 0; i < rgb.size(); ++i) worst = std::max(worst, std::abs(int(out[i]) - int(rgb[i])));
        CHECK_MSG(worst <= (spec.full_range ? 2 : 3), "yuv round trip bt709=%d full=%d worst %d", int(spec.bt709), int(spec.full_range), worst);   // 8-bit quantisation of Y, U, V
        vision::rgb_to_nv12(rgb.data(), w, h, Y.data(), UV.data(), spec); vision::nv12_to_rgb(Y.data(), w, UV.data(), cw * 2, w, h, out2.data(), spec);
        CHECK(out2 == out);
        std::vector<std::uint8_t> grey(w * h * 3, 77); vision::rgb_to_yuv420(grey.data(), w, h, Y.data(), U.data(), V.data(), spec);
        bool neutral = true; for (auto c : U) neutral &= c == 128; for (auto c : V) neutral &= c == 128; CHECK(neutral);
    }
}
