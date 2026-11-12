// Bit identity of the per-ISA tables beside the kernel table: the attention / delta-rule kernels (seq.hpp)
// and the int8 matvec (qdot_i8.hpp) of every table usable on this CPU against the scalar table, and that every
// usable kernel table has its Tier A and seq companions. Needs only the simd library, so it cross-builds and
// runs under qemu-user / node for the architectures without hardware here (ctest tables_* with label emu).
#include "check.hpp"
#include <forge/simd/dispatch.hpp>
#include <forge/simd/qdot_i8.hpp>
#include <forge/simd/seq.hpp>
#include <forge/simd/tier_a.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
using namespace forge::simd;

namespace {
struct rng {
    std::uint64_t s;
    explicit rng(std::uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    std::uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    std::uint32_t u32() { return std::uint32_t(next() >> 32); }
    std::size_t below(std::size_t n) { return n ? std::size_t(next() % n) : 0; }
    float unit() { return float(u32() >> 8) / float(1 << 24); }
    float sym(float a) { return (unit() - 0.5f) * 2.0f * a; }
    // roughly normal: sum of four uniforms
    float gauss() { return (unit() + unit() + unit() + unit() - 2.0f) * 1.7320508f; }
};
bool same(const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * 4) == 0); }
// one int8 KV row: word 0 the f32 scale, then d bytes (padded to a whole number of floats)
std::size_t i8_words(std::size_t d) { return 1 + (d + 3) / 4; }
void fill_i8_row(rng& r, float* row, std::size_t d) {
    const float scale = 0.002f + r.unit() * 0.02f;
    std::memcpy(row, &scale, 4);
    auto* b = reinterpret_cast<std::int8_t*>(row + 1);
    for (std::size_t e = 0; e < 4 * (i8_words(d) - 1); ++e) b[e] = e < d ? std::int8_t(int(r.below(255)) - 127) : std::int8_t(0);
}
} // namespace

TEST(every_kernel_table_has_its_companions) {
    std::string kt, at, st, qt;
    for (const char* n : target_names()) {
        const bool k = by_name(n) != nullptr;
        if (k) { kt += ' '; kt += n; }
        if (tier_a_by_name(n)) { at += ' '; at += n; }
        if (seq_by_name(n)) { st += ' '; st += n; }
#if !defined(__arm__)   // the opt-in Armv7 NEON table has no companions
        CHECK_MSG(k == (tier_a_by_name(n) != nullptr), "tier A table for %s", n);
        CHECK_MSG(k == (seq_by_name(n) != nullptr), "seq table for %s", n);
#endif
        if (auto* t = tier_a_by_name(n)) CHECK(std::string_view(t->target) == by_name(n)->target);
        if (auto* t = seq_by_name(n)) CHECK(std::string_view(t->target) == by_name(n)->target);
    }
    for (const char* n : qi8_target_names()) if (qi8_by_name(n)) { qt += ' '; qt += n; }
    std::printf("  kernel tables:%s\n  tier A tables:%s (active %s)\n  seq tables:%s (active %s)\n  qi8 tables:%s (active %s)\n",
                kt.c_str(), at.c_str(), tier_a_active().target, st.c_str(), seq_active().target, qt.c_str(), qi8_active().target);
    CHECK(tier_a_by_name("scalar") && seq_by_name("scalar") && qi8_by_name("scalar"));
    CHECK(qi8_by_name(qi8_active().target) == &qi8_active());
    // outside x86 (own AVX kernels) the best kernel table's companions are the active ones
    const std::string_view best = active().target;
    if (best != "scalar" && !std::getenv("FORGE_SIMD_TARGET") && !std::getenv("FORGE_LANE_TARGET")) {
        CHECK_MSG(best == tier_a_active().target, "tier A active %s, kernels %s", tier_a_active().target, active().target);
        CHECK_MSG(best == seq_active().target, "seq active %s, kernels %s", seq_active().target, active().target);
    }
#if !defined(__x86_64__) && !defined(_M_X64) && !defined(__aarch64__) && !defined(_M_ARM64) && !defined(__arm__) && defined(__GNUC__)
    if (best != "scalar" && !std::getenv("FORGE_QI8_TARGET")) CHECK_MSG(best == qi8_active().target, "qi8 active %s, kernels %s", qi8_active().target, active().target);
#endif
}

// seq tables: every entry of every table against the scalar table, bit for bit.
TEST(seq_tables_match_scalar_bit_for_bit) {
    const seq_kernels* ref = seq_by_name("scalar");
    std::vector<const seq_kernels*> ts;
    for (const char* n : target_names()) if (auto* t = seq_by_name(n)) if (t != ref) ts.push_back(t);
    rng r(7);
    // exp: the grid the kernels use (differences of scores, the cut at -80, the clamp ends) and random values
    for (auto* k : ts) {
        for (int i = -9000; i <= 9000; ++i) { float x = float(i) * 0.01f; CHECK_MSG(ref->exp1(x) == k->exp1(x), "%s exp1(%g)", k->target, double(x)); }
        for (float x : {-0.f, 0.f, -87.f, -87.5f, 88.f, 89.f, -1e-30f, 1e-30f, -80.f}) { float a = ref->exp1(x), b = k->exp1(x); CHECK(std::memcmp(&a, &b, 4) == 0); }
    }
    // {t positions, d, dv, ns, key int8, value int8}
    struct cfg { std::size_t t, d, dv, ns; int k8, v8; };
    for (cfg c : {cfg{600, 64, 64, 4, 0, 0}, cfg{600, 80, 48, 3, 1, 1}, cfg{530, 33, 33, 1, 1, 0}, cfg{700, 600, 40, 2, 1, 1},
                  cfg{300, 128, 17, 9, 0, 1}, cfg{520, 16, 5, 5, 1, 1}}) {
        const std::size_t kw = c.k8 ? i8_words(c.d) : c.d, vw = c.v8 ? i8_words(c.dv) : c.dv;
        std::vector<float> kb(c.t * kw), vb(c.t * vw), q(c.ns * c.d);
        for (std::size_t j = 0; j < c.t; ++j) {
            if (c.k8) fill_i8_row(r, kb.data() + j * kw, c.d); else for (std::size_t e = 0; e < c.d; ++e) kb[j * kw + e] = r.gauss();
            if (c.v8) fill_i8_row(r, vb.data() + j * vw, c.dv); else for (std::size_t e = 0; e < c.dv; ++e) vb[j * vw + e] = r.gauss();
        }
        for (auto& x : q) x = r.gauss();
        if (c.ns > 1) for (std::size_t e = 0; e < c.d; ++e) q[e] = 0.f;   // an all-zero query: uniform weights
        const float sc = 1.f / std::sqrt(float(c.d));
        std::vector<const float*> qp; for (std::size_t s = 0; s < c.ns; ++s) qp.push_back(q.data() + s * c.d);
        const seq_kv K{kb.data(), kw, c.k8}, V{vb.data(), vw, c.v8};
        // chunks: a full one, the tail, a short offset one with per-slot ranges, and one where a slot is empty
        struct span { std::size_t c0, c1; int mode; };
        for (span sp : {span{256, 512, 0}, span{512, c.t, 0}, span{256, 256 + 37, 1}, span{0, 256, 2}, span{0, 256, 0}}) {
            if (sp.c1 > c.t || sp.c0 >= sp.c1) continue;
            std::vector<std::size_t> lo(c.ns), hi(c.ns);
            for (std::size_t s = 0; s < c.ns; ++s) {
                lo[s] = sp.mode == 1 ? sp.c0 + 3 * s : sp.mode == 2 ? (s % 2 ? sp.c0 + 100 + s : 0) : 0;
                hi[s] = sp.mode == 1 ? sp.c0 + 21 + 5 * s : sp.mode == 2 ? (s == c.ns - 1 ? lo[s] : sp.c0 + 200 - 7 * s) : c.t;
            }
            for (int q16 = 0; q16 < 2; ++q16) {
                auto run = [&](const seq_kernels* k, std::vector<float>& m, std::vector<float>& l, std::vector<float>& a) {
                    m.assign(c.ns, 0.f); l.assign(c.ns, 0.f); a.assign(c.ns * c.dv, 0.f);
                    std::vector<float> scr(k->attn_scratch(c.ns, c.d, c.dv));
                    (q16 ? k->attn_chunk_q16 : k->attn_chunk)(K, V, c.d, c.dv, sc, sp.c0, sp.c1, c.ns, qp.data(), lo.data(), hi.data(), m.data(), l.data(), a.data(), scr.data());
                };
                std::vector<float> m1, l1, a1, m2, l2, a2;
                run(ref, m1, l1, a1);
                for (auto* k : ts) {
                    run(k, m2, l2, a2);
                    CHECK_MSG(same(m1, m2) && same(l1, l2) && same(a1, a2), "%s attn_chunk%s d=%zu dv=%zu ns=%zu k8=%d v8=%d chunk [%zu,%zu) mode %d",
                              k->target, q16 ? "_q16" : "", c.d, c.dv, c.ns, c.k8, c.v8, sp.c0, sp.c1, sp.mode);
                }
                // merge two partials and finish (slot 0)
                if (sp.mode == 0 && m1[0] != -std::numeric_limits<float>::infinity()) {
                    for (auto* k : ts) {
                        float M1 = m1[0] - 0.75f, L1 = l1[0] * 1.5f, M2 = M1, L2 = L1;
                        std::vector<float> A1(a1.begin(), a1.begin() + std::ptrdiff_t(c.dv)), A2 = A1, o1(c.dv), o2(c.dv);
                        for (auto& x : A1) x *= 0.5f;
                        A2 = A1;
                        ref->attn_merge(&M1, &L1, A1.data(), m1[0], l1[0], a1.data(), c.dv);
                        k->attn_merge(&M2, &L2, A2.data(), m1[0], l1[0], a1.data(), c.dv);
                        CHECK_MSG(std::memcmp(&M1, &M2, 4) == 0 && std::memcmp(&L1, &L2, 4) == 0 && same(A1, A2), "%s attn_merge dv=%zu", k->target, c.dv);
                        ref->attn_finish(A1.data(), L1, o1.data(), c.dv); k->attn_finish(A2.data(), L2, o2.data(), c.dv);
                        CHECK_MSG(same(o1, o2), "%s attn_finish dv=%zu", k->target, c.dv);
                    }
                }
            }
        }
    }
    // l2norm_scale and the gated delta rule
    for (std::size_t n : {std::size_t(1), std::size_t(7), std::size_t(16), std::size_t(33), std::size_t(64), std::size_t(100)}) {
        std::vector<float> x(n), y1(n), y2(n);
        for (auto& v : x) v = r.gauss();
        for (int l2 = 0; l2 < 2; ++l2) for (auto* k : ts) {
            ref->l2norm_scale(x.data(), y1.data(), n, 0.37f, l2); k->l2norm_scale(x.data(), y2.data(), n, 0.37f, l2);
            CHECK_MSG(same(y1, y2), "%s l2norm_scale n=%zu l2=%d", k->target, n, l2);
        }
    }
    for (std::size_t nj : {std::size_t(1), std::size_t(5), std::size_t(16), std::size_t(17), std::size_t(31), std::size_t(32)}) {
        const std::size_t s = 6, dk = 24, dv = 40, j0 = 3;
        std::vector<float> q(s * dk), kk(s * dk), v(s * dv), g(s), beta(s), S0(dk * dv);
        for (auto& x : q) x = r.gauss() * 0.2f;
        for (auto& x : kk) x = r.gauss() * 0.2f;
        for (auto& x : v) x = r.gauss();
        for (auto& x : g) x = -r.unit();
        for (auto& x : beta) x = r.unit();
        for (auto& x : S0) x = r.gauss() * 0.1f;
        std::vector<float> S1 = S0, o1(s * dv, 0.f);
        ref->delta_cols(q.data(), kk.data(), dk, v.data(), dv, g.data(), beta.data(), 1, S1.data(), o1.data(), dv, s, dk, dv, j0, nj);
        for (auto* k : ts) {
            std::vector<float> S2 = S0, o2(s * dv, 0.f);
            k->delta_cols(q.data(), kk.data(), dk, v.data(), dv, g.data(), beta.data(), 1, S2.data(), o2.data(), dv, s, dk, dv, j0, nj);
            CHECK_MSG(same(S1, S2) && same(o1, o2), "%s delta_cols nj=%zu", k->target, nj);
        }
    }
    std::printf("  seq: %zu table(s) identical to scalar\n", ts.size());
}

// int8 matvec tables: quantisation and products of every table against the scalar reference, on raw blocks
// (random bytes and the ends of every integer range) with finite f16 scales.
static constexpr std::uint64_t QI8_REF_HASH = 0xc763b823f7512501ull;   // the same constant as test_graph: fixed raw blocks, every platform
TEST(qi8_tables_match_scalar_bit_for_bit) {
    const qi8_kernels* ref = qi8_by_name("scalar");
    std::vector<const qi8_kernels*> ts;
    for (const char* n : qi8_target_names()) if (auto* t = qi8_by_name(n)) if (t != ref) ts.push_back(t);
    rng r(5);
    std::uint64_t ref_hash = 1469598103934665603ull;
    const struct { const char* name; int fmt; } fmts[] = {{"q8_0", 1}, {"q4_k", 2}, {"q5_k", 4}, {"q6_k", 3}};
    for (auto [cn, fmt] : fmts) {
        const std::size_t k = 512, rows = 9, nv = 13, bb = qi8::block_bytes(fmt), rb = (fmt == 1 ? k / 32 : k / 256) * bb, ab = qi8::act_bytes(fmt, k);
        CHECK(qi8::supported(fmt, k));
        std::vector<float> X(nv * k);
        for (std::size_t v = 0; v < nv; ++v)
            for (std::size_t i = 0; i < k; ++i)
                X[v * k + i] = v == 0 ? 1.f : v == 1 ? -1.f : v == 2 ? (i & 1 ? 1.f : -1.f) : v == 3 ? (i % 3 ? -1.f : 1.f)
                             : v == 4 ? (i == 7 ? std::numeric_limits<float>::quiet_NaN() : i & 2 ? 0.5f : -1.f)
                             : v == 5 ? r.sym(1.f) * (i % 7 ? 1.f : 1e-3f)                 // small values next to large ones
                             : v == 6 ? (i < 256 ? 0.f : r.sym(3.f))                       // an all-zero block (scale 0)
                             : v == 7 ? float(int(r.below(5)) - 2) * 0.5f                  // exact ties (x.5 after scaling)
                             : v == 8 ? (i == 300 ? std::numeric_limits<float>::infinity() : r.sym(1.f))
                             : r.sym(1.f);
        std::vector<qi8::act_k> abuf_ref((nv * ab) / sizeof(qi8::act_k) + 1), abuf((nv * ab) / sizeof(qi8::act_k) + 1);
        auto* aref = reinterpret_cast<std::uint8_t*>(abuf_ref.data()); auto* aq = reinterpret_cast<std::uint8_t*>(abuf.data());
        for (std::size_t v = 0; v < nv; ++v) ref->quantize(fmt, X.data() + v * k, k, aref + v * ab);
        for (auto* ker : ts) {
            std::memset(aq, 0, nv * ab);
            for (std::size_t v = 0; v < nv; ++v) ker->quantize(fmt, X.data() + v * k, k, aq + v * ab);
            // compare the fields (struct padding is not written)
            bool eq = true;
            for (std::size_t v = 0; v < nv && eq; ++v) {
                if (fmt == 1) for (std::size_t b = 0; b < k / 32; ++b) { const auto* x = reinterpret_cast<const qi8::act_32*>(aref + v * ab) + b; const auto* y = reinterpret_cast<const qi8::act_32*>(aq + v * ab) + b; eq = eq && std::memcmp(&x->d, &y->d, 4) == 0 && std::memcmp(x->q, y->q, 32) == 0; }
                else for (std::size_t b = 0; b < k / 256; ++b) { const auto* x = reinterpret_cast<const qi8::act_k*>(aref + v * ab) + b; const auto* y = reinterpret_cast<const qi8::act_k*>(aq + v * ab) + b; eq = eq && std::memcmp(&x->d, &y->d, 4) == 0 && std::memcmp(x->q, y->q, 256) == 0 && std::memcmp(x->bsum16, y->bsum16, 32) == 0 && std::memcmp(x->bsum32, y->bsum32, 16) == 0; }
                CHECK_MSG(eq, "%s/%s: quantised activations of vector %zu differ from the scalar reference", ker->target, cn, v);
            }
        }
        for (int pat = 0; pat < 6; ++pat) {
            std::vector<std::uint8_t> enc(rows * rb);
            for (auto& b : enc) b = pat == 0 ? std::uint8_t(0xFF) : pat == 1 ? std::uint8_t(0x80) : pat == 2 ? std::uint8_t(0x7F) : std::uint8_t(r.u32());
            const std::uint16_t fin = 0x2000;   // a small finite f16
            for (std::size_t blk = 0; blk < rows * rb / bb; ++blk) {
                std::uint8_t* p = enc.data() + blk * bb;
                const std::uint16_t f = pat < 3 ? fin : std::uint16_t(0x2000 + (r.u32() & 0x1fff) + (r.u32() & 0x8000));   // finite, either sign
                if (fmt == 3) std::memcpy(p + 208, &f, 2);
                else { std::memcpy(p, &f, 2); if (fmt != 1) std::memcpy(p + 2, &f, 2); }
            }
            std::vector<float> yref(nv * rows), y(nv * rows), y1(rows);
            ref->matvec(fmt, enc.data(), rb, rows, aref, ab, k, nv, yref.data(), rows);
            if (pat < 3)   // fixed inputs (vectors 0..4): the same bits on every architecture and standard library
                for (std::size_t i = 0; i < 5 * rows * 4; ++i) ref_hash = (ref_hash ^ reinterpret_cast<const std::uint8_t*>(yref.data())[i]) * 1099511628211ull;
            for (auto* ker : ts) {
                for (std::size_t n : {std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(5), std::size_t(8), std::size_t(9), nv}) {
                    std::fill(y.begin(), y.end(), -1.f);
                    ker->matvec(fmt, enc.data(), rb, rows, aref, ab, k, n, y.data(), rows);
                    CHECK_MSG(std::memcmp(y.data(), yref.data(), n * rows * 4) == 0, "%s/%s pattern %d nv=%zu: differs from the scalar reference", ker->target, cn, pat, n);
                }
                for (std::size_t v = 0; v < nv; v += 4) {   // one vector at a time, in odd row slices
                    for (std::size_t r0 = 0; r0 < rows; r0 += 4) ker->matvec(fmt, enc.data() + r0 * rb, rb, std::min<std::size_t>(4, rows - r0), aref + v * ab, ab, k, 1, y1.data() + r0, rows);
                    CHECK_MSG(std::memcmp(y1.data(), yref.data() + v * rows, rows * 4) == 0, "%s/%s pattern %d: vector %zu alone differs from the batch", ker->target, cn, pat, v);
                }
            }
        }
    }
    std::printf("  qi8: %zu table(s) identical to scalar; reference hash %016llx\n", ts.size(), static_cast<unsigned long long>(ref_hash));
    // The block formats are little-endian byte layouts read with memcpy (qdot_i8.hpp, as the codecs), so the
    // fixed-block hash is only comparable between little-endian hosts; a big-endian host (s390x) still checks
    // its tables against its own scalar reference above.
    const std::uint16_t one = 1; std::uint8_t lsb; std::memcpy(&lsb, &one, 1);
    if (lsb == 1) CHECK_MSG(ref_hash == QI8_REF_HASH, "the scalar integer reference gives other bits here than on x86-64");
    else std::printf("  big-endian host: reference hash not compared\n");
}
