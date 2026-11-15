// 2:4 sparse matmul (graph/s24_gemm.hpp) and lossless entropy-coded model storage (modelfile/pack.hpp).
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/graph/graph.hpp>
#include <forge/graph/plan.hpp>
#include <forge/inference/inference.hpp>
#include <forge/modelfile/pack.hpp>
#include <forge/verify/oracle.hpp>

using namespace forge;
using namespace forge::graph;

namespace {
std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / ("forge_tps_" + n)).string(); }
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
bool bench_on() { const char* e = std::getenv("FORGE_BENCH"); return e && *e == '1'; }

// reference: decode each weight row, dense dot in double
void ref_gemm(const codecs::codec* c, const std::vector<u8>& enc, std::size_t n, const float* x, std::size_t m, std::size_t k,
              std::vector<double>& y, std::vector<double>& l1) {
    std::vector<float> row(k); std::size_t rb = c->bytes(k);
    y.assign(m * n, 0); l1.assign(m * n, 0);
    for (std::size_t j = 0; j < n; ++j) {
        c->decode(enc.data() + j * rb, k, row.data());
        for (std::size_t i = 0; i < m; ++i) {
            double s = 0, a = 0;
            for (std::size_t t = 0; t < k; ++t) { s += double(x[i * k + t]) * row[t]; a += std::fabs(double(x[i * k + t]) * row[t]); }
            y[i * n + j] = s; l1[i * n + j] = a;
        }
    }
}
}

// Every kernel tier this CPU runs, every s24 codec, partial last blocks, 1..70 activation rows: the sparse product
// equals decode + dense dot up to float summation order (|err| <= 2^-20 * sum |x w|), and the plan's qgemm lowering
// on an s24 weight passes matmul_t's oracle tolerance against the f32 reference op on the decoded weights.
TEST(s24_gemm_matches_decode) {
    std::mt19937 r(3);
    for (s24::isa t : {s24::isa::scalar, s24::isa::avx2, s24::isa::avx512}) {
        if (int(t) > int(s24::best_isa())) { std::printf("  s24: tier %d not available here\n", int(t)); continue; }
        for (const char* cn : {"s24_f16", "s24_q8", "s24_q4"}) {
            const auto* c = codecs::find(cn); const int bits = s24::bits_of(c);
            CHECK(bits != 0);
            for (std::size_t k : {std::size_t(32), std::size_t(256), std::size_t(1000), std::size_t(2048 + 96)}) {
                for (std::size_t m : {std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(5), std::size_t(7), std::size_t(8), std::size_t(13), std::size_t(70)}) {
                    const std::size_t n = 19, rb = c->bytes(k);
                    auto W = verify::random_tensor({i64(n), i64(k)}, r, -0.2f, 0.2f);
                    auto X = verify::random_tensor({i64(m), i64(k)}, r, -1, 1);
                    std::vector<u8> enc(n * rb);
                    for (std::size_t j = 0; j < n; ++j) c->encode(W.data.data() + j * k, k, enc.data() + j * rb);
                    std::vector<double> ref, l1; ref_gemm(c, enc, n, X.data.data(), m, k, ref, l1);
                    std::vector<float> y(m * n, -7.f);
                    s24::gemm_nt(bits, enc.data(), rb, n, X.data.data(), m, k, y.data(), t);
                    double worst = 0;
                    for (std::size_t i = 0; i < m * n; ++i) worst = std::max(worst, std::fabs(y[i] - ref[i]) / (l1[i] + 1e-30));
                    CHECK_MSG(worst <= std::ldexp(1.0, -20), "tier %d %s k=%zu m=%zu: rel err %.3g", int(t), cn, k, m, worst);
                    // batch invariance: each activation row alone gives the same bits as in the batch
                    for (std::size_t i = 0; i < m; i += std::max<std::size_t>(1, m / 3)) {
                        std::vector<float> y1(n);
                        s24::gemm_nt(bits, enc.data(), rb, n, X.data.data() + i * k, 1, k, y1.data(), t);
                        CHECK_MSG(std::memcmp(y1.data(), y.data() + i * n, n * 4) == 0, "tier %d %s k=%zu m=%zu: row %zu differs from its m=1 result", int(t), cn, k, m, i);
                    }
                }
            }
        }
        std::printf("  s24: tier %d matches decode + dense dot for s24_f16 s24_q8 s24_q4\n", int(t));
    }
    // through the plan's lowering, against the oracle tolerance of matmul_t
    for (const char* cn : {"s24_f16", "s24_q8", "s24_q4"}) {
        const auto* c = codecs::find(cn); const std::size_t n = 64, k = 512, m = 5, rb = c->bytes(k);
        auto W = verify::random_tensor({i64(n), i64(k)}, r, -0.2f, 0.2f);
        auto X = verify::random_tensor({i64(m), i64(k)}, r, -1, 1);
        std::vector<u8> enc(n * rb);
        for (std::size_t j = 0; j < n; ++j) c->encode(W.data.data() + j * k, k, enc.data() + j * rb);
        tensor Wd({i64(n), i64(k)}); c->decode(enc.data(), n * k, Wd.data.data());
        tensor ref = ops::ref::matmul_t({X, Wd}, {});
        tensor got({i64(m), i64(n)});
        tensor_view in[2] = {X.view(), tensor_view{nullptr, {i64(n), i64(k)}, enc.data(), c}};
        qweights::gemm_nt(in, got.view(), {});
        auto cr = verify::compare(got, ref, ops::info(ops::op::matmul_t).tol);
        CHECK_MSG(cr.ok, "%s via qgemm: %s (max rel %.3g)", cn, cr.message.c_str(), cr.max_rel);
    }
}

// Timing (FORGE_BENCH=1): matvec and small batches, s24 kernels vs the generic decode + dot path and the dense
// f32 / q8_0 paths of qweights::gemm_nt, all threads.
TEST(s24_gemm_bench) {
    if (!bench_on()) { std::printf("  (FORGE_BENCH=1 for timings)\n"); return; }
    std::mt19937 r(9);
    struct shp { std::size_t n, k; };
    for (shp s : {shp{2048, 2048}, shp{5632, 2048}, shp{2048, 5632}}) {
        auto W = verify::random_tensor({i64(s.n), i64(s.k)}, r, -0.1f, 0.1f);
        for (std::size_t m : {std::size_t(1), std::size_t(8), std::size_t(32)}) {
            auto X = verify::random_tensor({i64(m), i64(s.k)}, r, -1, 1);
            tensor y({i64(m), i64(s.n)});
            auto timeit = [&](auto&& f) {
                f(); double best = 1e9;
                for (int rep = 0; rep < 5; ++rep) { double t0 = now_s(); int it = 0; do { f(); ++it; } while (now_s() - t0 < 0.15); best = std::min(best, (now_s() - t0) / it); }
                return best * 1e3;
            };
            std::string line = std::format("  n={} k={} m={}:", s.n, s.k, m);
            {   // dense f32 (the matmul_t lowering of an f32 weight: one dot per output)
                auto& ker = simd::active();
                double ms = timeit([&] { task::parallel_for(s.n, 8, [&](std::size_t b, std::size_t e) {
                    for (std::size_t j = b; j < e; ++j) for (std::size_t i = 0; i < m; ++i) y.data[i * s.n + j] = ker.dot(X.data.data() + i * s.k, W.data.data() + j * s.k, s.k); }); });
                line += std::format(" f32 {:.3f}ms", ms);
            }
            for (const char* cn : {"q8_0", "s24_f16", "s24_q8", "s24_q4"}) {
                const auto* c = codecs::find(cn); std::size_t rb = c->bytes(s.k);
                std::vector<u8> enc(s.n * rb);
                for (std::size_t j = 0; j < s.n; ++j) c->encode(W.data.data() + j * s.k, s.k, enc.data() + j * rb);
                tensor_view in[2] = {X.view(), tensor_view{nullptr, {i64(s.n), i64(s.k)}, enc.data(), c}};
                double ms = timeit([&] { qweights::gemm_nt(in, y.view(), {}); });
                line += std::format(" {} {:.3f}ms", cn, ms);
                if (s24::bits_of(c)) {
                    double g = timeit([&] {   // the generic path these codecs took before: decode a row to f32, dense dot
                        const auto& ker = simd::active();
                        task::parallel_for(s.n, 8, [&](std::size_t b, std::size_t e) {
                            thread_local std::vector<float> row; if (row.size() < s.k) row.resize(s.k);
                            for (std::size_t j = b; j < e; ++j) { c->decode(enc.data() + j * rb, s.k, row.data());
                                for (std::size_t i = 0; i < m; ++i) y.data[i * s.n + j] = ker.dot(X.data.data() + i * s.k, row.data(), s.k); } }); });
                    line += std::format(" (generic {:.3f})", g);
                }
            }
            std::printf("%s\n", line.c_str());
        }
    }
}

// ---- packed model files (modelfile/pack.hpp) ----
namespace {
std::string file_hash(const std::string& p) {
    std::size_t n = 0; const std::uint8_t* m = forge::map_readonly(p, n);
    auto h = hash::blake3::hex(hash::blake3::of({m, n})); forge::unmap_readonly(m, n); return h;
}
std::vector<std::uint8_t> read_all(const std::string& p) { std::ifstream f(p, std::ios::binary); return {std::istreambuf_iterator<char>(f), {}}; }
void write_all(const std::string& p, const std::vector<std::uint8_t>& b) { std::ofstream f(p, std::ios::binary); f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size())); }
// encode + decode one segment; true when the bytes come back
bool seg_roundtrip(const modelfile::pack::layout& L, std::size_t row, const std::vector<std::uint8_t>& src) {
    std::vector<std::uint8_t> enc, back(src.size(), 0xAA);
    if (!modelfile::pack::encode_segment(L, row, src.data(), src.size(), enc)) return true;   // stored: nothing to decode
    modelfile::pack::decode_segment(L, row, enc.data(), enc.size(), back.data(), back.size());
    return back == src;
}
}

// Segments: random bytes, constant runs, skewed bytes, single periods, weights in every layout.
TEST(pack_segments_roundtrip) {
    std::mt19937 r(17);
    namespace pk = modelfile::pack;
    auto rnd = [&](std::size_t n) { std::vector<std::uint8_t> v(n); for (auto& b : v) b = std::uint8_t(r()); return v; };
    for (const char* cn : {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "s24_f16", "s24_q8", "s24_q4", "q4_k", "q6_k", "fp8_e4m3", "(none)"}) {
        const auto L = pk::layout_for(cn);
        CHECK(L.period >= 1 && L.ncls >= 1 && L.ncls <= pk::max_classes);
        for (std::size_t periods : {std::size_t(1), std::size_t(3), std::size_t(1000), std::size_t(40000)}) {
            const std::size_t n = periods * L.period;
            CHECK_MSG(seg_roundtrip(L, 0, rnd(n)), "%s random n=%zu", cn, n);
            CHECK_MSG(seg_roundtrip(L, 0, std::vector<std::uint8_t>(n, 0x3C)), "%s constant n=%zu", cn, n);
            std::vector<std::uint8_t> skew(n); for (auto& b : skew) b = std::uint8_t(r() % 7 == 0 ? r() : r() % 3);
            CHECK_MSG(seg_roundtrip(L, L.period * 4, skew), "%s skewed n=%zu", cn, n);
            std::vector<std::uint8_t> ctx(n); for (std::size_t i = 0; i < n; ++i) ctx[i] = std::uint8_t(i ? (ctx[i - 1] * 5 + r() % 4) : 1);   // order-1 structure
            CHECK_MSG(seg_roundtrip(L, 0, ctx), "%s order-1 n=%zu", cn, n);
        }
        if (const auto* c = graph::codecs::find(cn)) {   // weight-like data in the codec, rows of 256
            auto W = verify::random_tensor({64, 256}, r, -1, 1);
            for (std::size_t i = 0; i < W.data.size(); i += 3) W.data[i] *= 0.01f;
            std::vector<std::uint8_t> enc(c->bytes(W.size())); c->encode(W.data.data(), W.size(), enc.data());
            if (enc.size() % L.period == 0) CHECK_MSG(seg_roundtrip(L, c->bytes(256), enc), "%s weights", cn);
        }
    }
}

TEST(pack_file_roundtrip_and_corruption) {
    namespace pk = modelfile::pack;
    std::mt19937 r(23);
    const auto src = tmp("codecs.fpk"), packed = tmp("codecs.fpkz"), back = tmp("codecs.back.fpk");
    {
        modelfile::writer w;
        for (const char* cn : {"f32", "f16", "bf16", "q8_0", "q4_0", "s24_f16", "s24_q8", "s24_q4", "q4_k"}) {
            auto W = verify::random_tensor({96, 256}, r, -0.5f, 0.5f);
            w.add_tensor(std::string("w.") + cn, W.shape, W.data, cn);
        }
        w.add_tensor("w.const", {256}, std::vector<float>(256, 0.25f));
        w.set_manifest(json::value(json::object{{"format", "fpk"}, {"note", std::string(3000, 'x')}}));
        w.write(src);
    }
    auto st = pk::pack_file(src, packed);
    CHECK(pk::is_packed(packed) && !pk::is_packed(src));
    CHECK_MSG(st.out < st.in, "packed %llu of %llu bytes", (unsigned long long)st.out, (unsigned long long)st.in);
    CHECK(st.by_codec.count("q8_0") && st.by_codec.count("(other)"));
    auto us = pk::unpack_file(packed, back);
    CHECK(file_hash(back) == file_hash(src) && us.blake3 == file_hash(src) && read_all(back) == read_all(src));
    {   // the unpacked file is the model: it opens and every tensor verifies
        modelfile::reader rd(back); CHECK(rd.verify().empty());
    }
    // a damaged packed file never yields anything but the original: unpacking fails and leaves no file, or (a byte
    // the decoder never reads, such as the low end of a coder's final flush) gives back the original exactly
    auto bytes = read_all(packed);
    int failed = 0;
    for (std::size_t at = 50; at < bytes.size(); at += bytes.size() / 23 + 1) {
        auto bad = bytes; bad[at] ^= 0x41;
        const auto badp = tmp("bad.fpkz"), out = tmp("bad.out.fpk");
        write_all(badp, bad); std::filesystem::remove(out);
        try { pk::unpack_file(badp, out); CHECK_MSG(read_all(out) == read_all(src), "byte %zu flipped: unpacked a different file", at); }
        catch (const std::exception&) { ++failed; CHECK(!std::filesystem::exists(out) && !std::filesystem::exists(out + ".partial")); }
    }
    CHECK(failed >= 20);
    {   // truncated, and not a packed file at all
        auto cut = bytes; cut.resize(cut.size() / 3); write_all(tmp("cut.fpkz"), cut);
        CHECK_THROWS(pk::unpack_file(tmp("cut.fpkz"), tmp("cut.out")));
        CHECK_THROWS(pk::unpack_file(src, tmp("notpacked.out")));
    }
    // any file packs losslessly (one generic stream when it is not an .fpk), empty and tiny ones included
    for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(5), std::size_t(70000)}) {
        std::vector<std::uint8_t> b(n); for (std::size_t i = 0; i < n; ++i) b[i] = std::uint8_t(i % 13 == 0 ? r() : i % 5);
        const auto f = tmp("blob.bin"); write_all(f, b);
        pk::pack_file(f, tmp("blob.fpkz")); pk::unpack_file(tmp("blob.fpkz"), tmp("blob.back"));
        CHECK_MSG(read_all(tmp("blob.back")) == b, "blob n=%zu", n);
    }
}

// A tiny model packed, unpacked and run: the unpacked file is the same file and gives the same logits. The same
// model in s24_q8 (the 2:4 sparse matmul path) agrees with a copy of it whose weights are the decoded f32 values.
TEST(pack_tiny_model_runs) {
    namespace pk = modelfile::pack;
    const std::vector<int> prompt{256, 72, 101, 108, 108, 111, 44, 32};
    for (const char* codec : {"f32", "q8_0", "s24_q8"}) {
        const auto m = forge_test::make_tiny_model(tmp(std::string("tiny_") + codec + ".fpk"), {.d = 64, .heads = 4, .kv = 2, .ffn = 128, .codec = codec});
        const auto z = m + "z", u = m + ".unpacked.fpk";
        auto st = pk::pack_file(m, z); pk::unpack_file(z, u);
        CHECK(file_hash(u) == file_hash(m));
        inference::session a(m), b(u);
        auto la = a.forward(prompt), lb = b.forward(prompt);
        CHECK(!la.empty() && la == lb);
        std::printf("  tiny %s: %llu -> %llu bytes (%.2fx)\n", codec, (unsigned long long)st.in, (unsigned long long)st.out, double(st.in) / double(st.out));
        if (std::string(codec) == "s24_q8") {
            const auto d = m + ".dense.fpk";
            modelfile::reader rd(m); modelfile::writer w;
            std::vector<const modelfile::tensor_entry*> order; for (auto& [n, e] : rd.tensors()) order.push_back(&e);
            std::sort(order.begin(), order.end(), [](auto* x, auto* y) { return x->offset < y->offset; });
            for (auto* e : order) { auto t = rd.load(e->name); w.add_tensor(e->name, e->shape, t.data); }
            for (auto& [n, sc] : rd.sections()) if (n != "data" && n != "tensors") w.set_section(n, std::string(rd.section_text(n)));
            w.write(d);
            inference::session dn(d);
            auto ld = dn.forward(prompt);
            float md = 0, mx = 0; for (std::size_t i = 0; i < ld.size() && i < la.size(); ++i) { md = std::max(md, std::fabs(ld[i] - la[i])); mx = std::max(mx, std::fabs(ld[i])); }
            CHECK_MSG(ld.size() == la.size() && md <= 1e-4f * std::max(1.f, mx), "s24_q8 vs dense logits differ by %g (max %g)", double(md), double(mx));
        }
    }
}
