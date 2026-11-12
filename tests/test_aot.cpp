// Ahead-of-time plans (docs/aot-models.md): precompiled plans give the same bits as plans compiled at run time, and a
// blob that does not match, is damaged or is hostile is rejected and the plan is compiled instead.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/inference/inference.hpp>
#include <forge/inference/aot_build.hpp>
#include <chrono>
#include <filesystem>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace forge;
using namespace forge::inference;
namespace fs = std::filesystem;

static std::string tmp(const std::string& n) { return (fs::temp_directory_path() / ("forge_aot_" + std::to_string(::getpid()) + "_" + n)).string(); }
static bool same_bits(std::span<const float> a, std::span<const float> b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0; }

// `src` with the plan blob as its "aot" section
static std::string with_plans(const std::string& src, std::span<const std::uint8_t> blob, const std::string& name) {
    auto out = tmp(name);
    modelfile::aot::write_section(src, out, blob);
    return out;
}
static aot_plans plans_of(const std::string& path, aot_build_options o = {}) {
    model m(path);
    return aot_compile_plans(m, std::move(o));
}
static std::shared_ptr<model> jit_model(const std::string& path) { auto m = model::open(path); m->set_aot(false); return m; }

// Runs the same token stream through a precompiled and a run-time compiled session and compares every logit bitwise.
// `chunks` are the lengths fed in turn (prefill, then decode steps ...).
static void expect_identical(const std::string& aot_path, const std::string& ref_path, const std::vector<int>& chunks, session_options so = {},
                             std::size_t min_used = 1, bool all_rows = false) {
    auto ma = model::open(aot_path); auto mr = jit_model(ref_path);
    session a(ma, so), r(mr, so);
    int tok = 3;
    for (int n : chunks) {
        std::vector<int> ids; for (int i = 0; i < n; ++i) ids.push_back((tok = (tok * 37 + 11) % 250));
        auto la = all_rows ? a.forward_all(ids) : a.forward(ids), lr = all_rows ? r.forward_all(ids) : r.forward(ids);
        CHECK_MSG(same_bits(la, lr), "logits differ for a chunk of %d at position %lld", n, (long long)r.position());
    }
    CHECK_MSG(ma->aot().stats().used >= min_used, "precompiled plans used: %zu, rejected %zu (%s)", ma->aot().stats().used, ma->aot().stats().rejected, ma->aot().stats().last_reject.c_str());
    CHECK(mr->aot().stats().used == 0);
    // greedy generation: the same tokens
    sampler g1, g2; a.reset(); r.reset();
    std::vector<int> p = {1, 5, 9, 33};
    CHECK(a.generate(p, 12, g1) == r.generate(p, 12, g2));
}

static const std::string& model_f32() { static std::string p = forge_test::make_tiny_model(tmp("f32.fpk")); return p; }

TEST(aot_blob_round_trip_is_bit_identical) {
    auto pl = plans_of(model_f32(), {.shapes = {1, 4, 7}, .all_rows = true});
    CHECK(pl.targets.size() == 6);
    graph::aot::image img(pl.blob);
    CHECK(img.entries().size() == 6);
    CHECK(img.forge() == graph::aot::forge_version);
    auto f = with_plans(model_f32(), pl.blob, "rt.fpk");
    modelfile::reader r(f);
    CHECK(r.has_section("aot") && r.verify().empty());
    expect_identical(f, model_f32(), {4, 1, 1, 7, 1, 4}, {}, 3);
    expect_identical(f, model_f32(), {7, 4, 1}, {}, 3, true);
    // a session over the precompiled file compiled nothing for the shapes it has
    auto m = model::open(f); session s(m);
    std::vector<int> ids = {1, 2, 3, 4};
    s.forward(ids); s.forward(std::span<const int>(ids.data(), 1));
    CHECK(m->aot().stats().used == 2 && m->aot().stats().rejected == 0);
    CHECK(m->plan_for(1, false, const_cast<graph::state_store*>(&s.state())).aot);
}

TEST(aot_unlisted_shape_compiles_at_run_time) {
    auto pl = plans_of(model_f32(), {.shapes = {1}});
    auto f = with_plans(model_f32(), pl.blob, "one.fpk");
    expect_identical(f, model_f32(), {5, 1, 1, 3}, {}, 1);   // 5 and 3 are not precompiled
    auto m = model::open(f); session s(m);
    std::vector<int> ids = {1, 2, 3};
    s.forward(ids);
    CHECK(m->aot().stats().used == 0 && m->aot().stats().rejected == 0);   // not a rejection: just not there
    CHECK(!m->plan_for(3, false, const_cast<graph::state_store*>(&s.state())).aot);
}

TEST(aot_quantised_weights) {
    // q3_k (block 256) does not divide the 32-wide rows: those weights take the decoded-f32 path of lower_plan
    for (const char* codec : {"q8_0", "q4_k", "f16", "q4_0", "q5_1", "iq4_nl", "q3_k", "tq2_0"}) {
        auto src = forge_test::make_tiny_model(tmp(std::string(codec) + ".fpk"), {.codec = codec});
        auto pl = plans_of(src, {.shapes = {1, 6}});
        expect_identical(with_plans(src, pl.blob, std::string(codec) + "_aot.fpk"), src, {6, 1, 1, 6, 1}, {}, 2);
    }
}

TEST(aot_kv_codecs) {
    for (const char* kv : {"int8", "fp8", "tbq4", "k8v4"}) {
        auto pl = plans_of(model_f32(), {.shapes = {1, 5}, .kv_codecs = {"f32", kv}});
        CHECK(pl.targets.size() == 4);
        auto f = with_plans(model_f32(), pl.blob, std::string("kv_") + kv + ".fpk");
        expect_identical(f, model_f32(), {5, 1, 1, 5}, {.kv_codec = kv}, 2);
        expect_identical(f, model_f32(), {5, 1, 1}, {}, 2);
    }
    // a codec that was not precompiled: compiled at run time
    auto pl = plans_of(model_f32(), {.shapes = {1, 5}});
    expect_identical(with_plans(model_f32(), pl.blob, "kv_none.fpk"), model_f32(), {5, 1}, {.kv_codec = "int8"}, 0);
}

TEST(aot_recurrent_state_model) {   // Qwen3.5-style hybrid: state_op steps (DeltaNet, conv window) next to KV appends
    auto src = forge_test::make_tiny_qwen35_model(tmp("q35.fpk"));
    auto pl = plans_of(src, {.shapes = {1, 5}});
    expect_identical(with_plans(src, pl.blob, "q35_aot.fpk"), src, {5, 1, 1, 5, 1}, {}, 2);
}

TEST(aot_one_shot_model) {   // session kind one_shot (what an imported ONNX graph is): session::infer over a precompiled plan
    const char* bp = R"(
blueprint "mlp" opset 1
params { d: 8  scale: { a: 2 } }
input x : f32[seq, d]
graph {
  h = linear(x, W.w) * scale.a + 1
  repeat i in 0..3 { h = tanh(h) }
  y = softmax(h)
}
output y : f32[seq, d]
session one_shot)";
    auto src = tmp("oneshot.fpk");
    {
        modelfile::writer w;
        std::vector<float> d(64); for (std::size_t i = 0; i < d.size(); ++i) d[i] = float(int(i * 7 % 13) - 6) * 0.1f;
        w.add_tensor("w", {8, 8}, d);
        w.set_manifest(json::value(json::object{{"format", "fpk"}, {"session", "one_shot"}}));
        w.set_section("blueprint", bp);
        w.write(src);
    }
    auto pl = plans_of(src, {.shapes = {3}});
    CHECK(pl.targets.size() == 1);
    auto f = with_plans(src, pl.blob, "oneshot_aot.fpk");
    auto ma = model::open(f); auto mr = jit_model(src);
    session a(ma), r(mr);
    std::vector<float> x(24); for (std::size_t i = 0; i < x.size(); ++i) x[i] = float(i) * 0.05f - 0.5f;
    auto ya = a.infer({{"x", x}}, 3), yr = r.infer({{"x", x}}, 3);
    CHECK(same_bits(ya.at("y"), yr.at("y")) && ya.at("y").size() == 24);
    CHECK(ma->aot().stats().used == 1);
    auto yb = a.infer({{"x", std::vector<float>(16, 0.25f)}}, 2);   // a length that was not precompiled
    CHECK(same_bits(yb.at("y"), r.infer({{"x", std::vector<float>(16, 0.25f)}}, 2).at("y")));
}

TEST(aot_paged_and_preallocated_state) {
    auto pl = plans_of(model_f32(), {.shapes = {1, 3}});
    auto f = with_plans(model_f32(), pl.blob, "paged.fpk");
    expect_identical(f, model_f32(), {3, 1, 1, 1, 3, 1, 1, 3}, {.page_rows = 2}, 2);
    expect_identical(f, model_f32(), {3, 1, 3}, {.page_rows = 0}, 2);
}

TEST(aot_adapter_sessions_compile_at_run_time) {
    auto pl = plans_of(model_f32(), {.shapes = {1, 4}});
    auto f = with_plans(model_f32(), pl.blob, "lora.fpk");
    auto mk = [&](const std::string& path) {
        auto m = model::open(path);
        model::lora d; d.a = tensor({2, 32}, 0.01f); d.b = tensor({32, 2}, 0.02f);
        m->add_adapter("ad", {{"layer.0.wq", d}}, 4.f);
        return m;
    };
    auto ma = mk(f), mr = mk(model_f32()); mr->set_aot(false);
    session a(ma), r(mr); a.set_adapter("ad"); r.set_adapter("ad");
    std::vector<int> ids = {4, 8, 15, 16};
    CHECK(same_bits(a.forward(ids), r.forward(ids)));
    CHECK(ma->aot().stats().used == 0);   // the precompiled plans are for the base weights
    a.set_adapter(""); r.set_adapter("");
    CHECK(same_bits(a.forward(ids), r.forward(ids)));
    CHECK(ma->aot().stats().used == 1);
}

TEST(aot_other_model_falls_back) {
    auto other = forge_test::make_tiny_model(tmp("other.fpk"), {.seed = 99});
    auto pl = plans_of(model_f32(), {.shapes = {1, 4}});
    auto f = with_plans(other, pl.blob, "mismatch.fpk");   // same architecture, other weights: another key
    expect_identical(f, other, {4, 1, 1}, {}, 0);
    auto m = model::open(f); session s(m);
    std::vector<int> ids = {1, 2, 3, 4};
    s.forward(ids);
    CHECK(m->aot().stats().used == 0 && m->aot().stats().rejected >= 1);
    // FORGE_AOT=0 ignores a matching blob too
    auto g = with_plans(model_f32(), pl.blob, "off.fpk");
    forge_test::set_env("FORGE_AOT", "0");
    { auto m2 = model::open(g); session s2(m2); s2.forward(ids); CHECK(!m2->aot().available() && m2->aot().stats().used == 0); }
    forge_test::set_env("FORGE_AOT", nullptr);
}

TEST(aot_corrupt_blob_is_rejected) {
    auto pl = plans_of(model_f32(), {.shapes = {1, 4}});
    for (std::size_t at : {std::size_t(9), std::size_t(20), pl.blob.size() / 2, pl.blob.size() - 40, pl.blob.size() - 1}) {
        auto bad = pl.blob; bad[at] ^= 0x40;
        CHECK_THROWS(graph::aot::image(bad));
        auto f = with_plans(model_f32(), bad, "corrupt.fpk");   // the section table hash is right: only the blob's own hash catches it
        auto m = model::open(f);
        CHECK(!m->aot().available() && m->aot().stats().rejected == 1);
        expect_identical(f, model_f32(), {4, 1}, {}, 0);
    }
    for (std::size_t n : {std::size_t(0), std::size_t(8), std::size_t(40), std::size_t(100), pl.blob.size() - 1}) {
        std::vector<std::uint8_t> cut(pl.blob.begin(), pl.blob.begin() + std::ptrdiff_t(n));
        CHECK_THROWS(graph::aot::image(cut));
    }
    // bytes changed in the file after it was written: the table hash no longer matches, verified opens refuse the file
    auto f = with_plans(model_f32(), pl.blob, "flip.fpk");
    std::uint64_t off = 0;
    { modelfile::reader r(f); off = r.sections().at("aot").offset; }
    { std::fstream io(f, std::ios::in | std::ios::out | std::ios::binary); io.seekp(std::streamoff(off + pl.blob.size() / 2)); char c = 0x55; io.write(&c, 1); }
    CHECK_THROWS(model(f, true));
    expect_identical(f, model_f32(), {4, 1}, {}, 0);
}

// Blobs with a valid hash and key whose contents are wrong: each must be rejected by the checks in instantiate(), not
// trusted. The plan of a real model is altered in memory, serialised, and put back.
TEST(aot_hostile_plans_are_rejected) {
    model m(model_f32()); m.set_aot(false);
    graph::state_store st; st.capacity = 512;
    auto& p = m.plan_for(4, false, &st);
    auto key = modelfile::aot::model_key(m.file());
    graph::aot::target t; t.seq = 4;
    int n_case = 0;
    auto expect_rejected = [&](const char* what) {
        graph::aot::builder b(key); b.add(t, p);
        auto f = with_plans(model_f32(), b.finish(), "hostile.fpk");
        auto mm = model::open(f); session s(mm);
        std::vector<int> ids = {1, 2, 3, 4};
        auto mr = jit_model(model_f32()); session r(mr);
        CHECK_MSG(same_bits(s.forward(ids), r.forward(ids)), "%s: wrong logits", what);
        CHECK_MSG(mm->aot().stats().used == 0 && mm->aot().stats().rejected == 1, "%s: not rejected (%s)", what, mm->aot().stats().last_reject.c_str());
        ++n_case;
    };
    std::size_t arena_i = 0, alias_i = 0, op_i = 0;
    for (std::size_t i = 0; i < p.mem.slots.size(); ++i) {
        if (p.mem.slots[i].kind == graph::buffer_slot::arena && p.g.at(graph::value_id(i)).kind == graph::node_kind::op) { arena_i = i; op_i = i; }
        if (p.mem.slots[i].kind == graph::buffer_slot::alias) alias_i = i;
    }
    CHECK(arena_i && alias_i);
    {   // sanity: unaltered, it is accepted
        graph::aot::builder b(key); b.add(t, p);
        auto mm = model::open(with_plans(model_f32(), b.finish(), "sane.fpk")); session s(mm);
        std::vector<int> ids = {1, 2, 3, 4}; s.forward(ids);
        CHECK(mm->aot().stats().used == 1);
    }
    { auto& s = p.mem.slots[arena_i]; auto o = s.offset; s.offset = p.mem.arena_bytes + 64 * 1024; expect_rejected("buffer beyond the arena"); s.offset = o; }
    { auto& s = p.mem.slots[arena_i]; auto o = s.offset; s.offset = std::size_t(-64); expect_rejected("offset wraps"); s.offset = o; }
    { auto& s = p.mem.slots[arena_i]; auto o = s.offset; s.offset += 4; expect_rejected("misaligned buffer"); s.offset = o; }
    { auto& s = p.mem.slots[arena_i]; auto o = s.bytes; s.bytes = 4; expect_rejected("buffer smaller than its shape"); s.bytes = o; }
    { auto o = p.mem.arena_bytes; p.mem.arena_bytes = 64; expect_rejected("arena smaller than its buffers"); p.mem.arena_bytes = o; }
    { auto o = p.mem.arena_bytes; p.mem.arena_bytes = std::size_t(1) << 46; expect_rejected("huge arena"); p.mem.arena_bytes = o; }
    {   // two live buffers at one offset
        std::size_t other = 0;
        for (std::size_t i = 0; i < p.mem.slots.size(); ++i) { auto& s = p.mem.slots[i]; auto& a = p.mem.slots[arena_i];
            if (i != arena_i && s.kind == graph::buffer_slot::arena && s.bytes && s.def <= a.last && a.def <= s.last && s.offset != a.offset) other = i; }
        CHECK(other);
        auto& s = p.mem.slots[arena_i]; auto o = s.offset; s.offset = p.mem.slots[other].offset; expect_rejected("overlapping live buffers"); s.offset = o;
    }
    { auto& s = p.mem.slots[alias_i]; auto o = s.root; s.root = graph::value_id(alias_i + 1 < p.mem.slots.size() ? alias_i + 1 : alias_i); expect_rejected("alias of a later value"); s.root = o; }
    { auto& s = p.mem.slots[alias_i]; auto o = s.root; s.root = 0; expect_rejected("alias of an unrelated value"); s.root = o; }
    {   // a weight claimed as arena storage, an op output claimed as external
        std::size_t w = 0; for (std::size_t i = 0; i < p.g.size(); ++i) if (p.g.at(graph::value_id(i)).kind == graph::node_kind::weight) w = i;
        auto& s = p.mem.slots[w]; auto o = s.kind; s.kind = graph::buffer_slot::arena; expect_rejected("weight in the arena"); s.kind = o;
        auto& s2 = p.mem.slots[op_i]; auto o2 = s2.kind; s2.kind = graph::buffer_slot::external; expect_rejected("op output outside the arena"); s2.kind = o2;
    }
    { auto& n = p.g.nodes()[op_i]; auto o = n.shape; n.shape.back() *= 2; expect_rejected("shape that the op does not produce"); n.shape = o; }
    { auto& n = p.g.nodes()[op_i]; auto o = n.shape; n.shape = {i64(1) << 62, i64(1) << 62}; expect_rejected("overflowing shape"); n.shape = o; }
    { auto& n = p.g.nodes()[op_i]; auto o = n.inputs; n.inputs.push_back(0); expect_rejected("wrong arity"); n.inputs = o; }
    { auto& n = p.g.nodes()[op_i]; auto o = n.inputs; n.inputs[0] = graph::value_id(p.g.size() + 7); expect_rejected("input that does not exist"); n.inputs = o; }
    {   // a weight under another name, and a weight with a shape the file does not have
        std::size_t w = 0; for (std::size_t i = 0; i < p.g.size(); ++i) if (p.g.at(graph::value_id(i)).kind == graph::node_kind::weight && p.g.at(graph::value_id(i)).shape.size() == 2) w = i;
        auto& n = p.g.nodes()[w];
        auto o = n.name; n.name = "no.such.weight"; expect_rejected("unknown weight"); n.name = o;
        auto os = n.shape; n.shape[0] *= 8; expect_rejected("weight larger than stored"); n.shape = os;
    }
    {   // a lowering of another op, and a name no registry has
        std::size_t si = 0; for (std::size_t i = 0; i < p.steps.size(); ++i) if (p.steps[i].kind == 0 && p.steps[i].low && p.steps[i].low->name == "lane.rms_norm") si = i;
        CHECK(si);
        auto* o = p.steps[si].low;
        graph::lowering fake = *o; fake.name = "lane.softmax"; p.steps[si].low = &fake; expect_rejected("lowering of another op");
        fake.name = "evil.kernel"; expect_rejected("unknown lowering");
        p.steps[si].low = o;
        auto os = p.steps[si].out; p.steps[si].out = graph::value_id(op_i == os ? 0 : op_i); expect_rejected("step for another node"); p.steps[si].out = os;
        auto st2 = p.steps; p.steps.pop_back(); expect_rejected("missing step"); p.steps = st2;
    }
    {   // state: a second node appending to a slot, and a state_append claimed as an ordinary op output
        std::size_t a = 0, b2 = 0;
        for (std::size_t i = 0; i < p.g.size(); ++i) if (p.g.at(graph::value_id(i)).kind == graph::node_kind::state_append) { if (!a) a = i; else b2 = i; }
        CHECK(a && b2);
        auto& n = p.g.nodes()[b2]; auto o = n.name; n.name = p.g.at(graph::value_id(a)).name; expect_rejected("slot appended twice"); n.name = o;
    }
    CHECK(n_case >= 20);
    std::printf("  %d hostile plans rejected\n", n_case);
}

// A session already holds a slot with another row layout: a plan that would append rows of a different shape to it
// must not be accepted (the executor trusts the slot's layout).
TEST(aot_state_slot_layout_is_checked) {
    auto wide = forge_test::make_tiny_model(tmp("wide.fpk"), {.d = 64, .heads = 4, .kv = 4});
    model mw(wide); mw.set_aot(false);
    graph::state_store sw; sw.capacity = 512;
    auto& pw = mw.plan_for(2, false, &sw);
    // the wide model's graph over a store that holds the narrow model's rows
    model mn(model_f32()); mn.set_aot(false);
    graph::state_store sn; sn.capacity = 512;
    mn.plan_for(2, false, &sn);   // creates the narrow slots
    graph::aot::builder b(modelfile::aot::model_key(mw.file()));
    graph::aot::target t; t.seq = 2; b.add(t, pw);
    auto blob = b.finish();
    graph::aot::image img(blob);
    graph::aot::binder bind;
    bind.weight = [&](const std::string& name) { auto& e = mw.file().entry(name); return std::pair<shape_t, const float*>{e.shape, reinterpret_cast<const float*>(mw.file().section_bytes("data").data() + e.offset)}; };
    {   // fresh store: accepted
        graph::state_store fresh; fresh.capacity = 512;
        auto p = graph::aot::instantiate(img, img.entries()[0], bind, &fresh);
        CHECK(p && p->aot);
    }
    CHECK_THROWS(graph::aot::instantiate(img, img.entries()[0], bind, &sn));   // slots exist with [rows, 2, 8]; this plan appends [rows, 4, 16]
    CHECK_THROWS(graph::aot::instantiate(img, img.entries()[0], bind, nullptr));   // state but no store
}

TEST(aot_signed_files) {
    auto kp = modelfile::generate_keypair();
    auto pub = modelfile::detail::to_hex(kp.pub);
    auto pl = plans_of(model_f32(), {.shapes = {1, 4}});
    std::vector<int> ids = {1, 2, 3, 4};
    // 1. plans added, then signed: the signature covers the section; strict loading accepts it and uses the plans
    auto with = with_plans(model_f32(), pl.blob, "s1.fpk");
    auto signed1 = tmp("s1_signed.fpk");
    modelfile::sign_file(with, signed1, kp);
    { modelfile::reader r(signed1); CHECK(modelfile::aot::covered_by_signature(r)); }
    forge_test::set_env("FORGE_TRUSTED_KEYS", pub.c_str());
    { auto m = model::open(signed1); session s(m); s.forward(ids); CHECK(m->aot().stats().used == 1); }
    forge_test::set_env("FORGE_TRUSTED_KEYS", nullptr);
    expect_identical(signed1, model_f32(), {4, 1, 1}, {}, 2);
    // 2. forge aot on a signed file drops the signature (it could not stay valid)
    auto signed0 = tmp("s0_signed.fpk");
    modelfile::sign_file(model_f32(), signed0, kp);
    auto after = tmp("s2.fpk");
    CHECK(modelfile::aot::write_section(signed0, after, pl.blob));
    { modelfile::reader r(after); CHECK(!r.has_section("signature") && r.has_section("aot")); }
    forge_test::set_env("FORGE_REQUIRE_SIGNED", "1");
    CHECK_THROWS(model(after));
    forge_test::set_env("FORGE_REQUIRE_SIGNED", nullptr);
    // 3. a section slipped into a signed file with the old signature kept: strict loading refuses the file; plain
    //    loading runs the model but does not use the plans
    auto forged = tmp("s3.fpk");
    {
        modelfile::reader r(signed0);
        std::vector<std::pair<std::string, std::string>> secs;
        for (auto& [name, s] : r.sections()) { auto b = r.raw_section_bytes(name); secs.emplace_back(name, std::string(reinterpret_cast<const char*>(b.data()), b.size())); }
        secs.emplace_back("aot", std::string(reinterpret_cast<const char*>(pl.blob.data()), pl.blob.size()));
        modelfile::detail::write_sections(forged, std::move(secs));
    }
    { modelfile::reader r(forged); std::string why; CHECK(!modelfile::aot::covered_by_signature(r, &why)); CHECK(why.find("does not cover") != std::string::npos); }
    forge_test::set_env("FORGE_TRUSTED_KEYS", pub.c_str());
    CHECK_THROWS(model(forged));
    forge_test::set_env("FORGE_TRUSTED_KEYS", nullptr);
    { auto m = model::open(forged); session s(m); s.forward(ids); CHECK(m->aot().stats().used == 0 && m->aot().stats().rejected == 1); }
    expect_identical(forged, model_f32(), {4, 1}, {}, 0);
    // 4. the same, with the section replaced after signing (signature and table hash both stale for it)
    auto swapped = tmp("s4.fpk");
    {
        auto pl2 = plans_of(model_f32(), {.shapes = {1}});
        modelfile::reader r(signed1);
        std::vector<std::pair<std::string, std::string>> secs;
        for (auto& [name, s] : r.sections()) { auto b = name == "aot" ? std::span<const std::uint8_t>(pl2.blob) : r.raw_section_bytes(name); secs.emplace_back(name, std::string(reinterpret_cast<const char*>(b.data()), b.size())); }
        modelfile::detail::write_sections(swapped, std::move(secs));
    }
    { auto m = model::open(swapped); session s(m); s.forward(std::span<const int>(ids.data(), 1)); CHECK(m->aot().stats().used == 0 && m->aot().stats().rejected == 1); }
}

TEST(aot_sidecar) {
    auto src = forge_test::make_tiny_model(tmp("side.fpk"), {.seed = 5});
    auto pl = plans_of(src, {.shapes = {1, 4}});
    modelfile::aot::write_sidecar(src + ".fplan", pl.blob);
    expect_identical(src, src, {4, 1, 1}, {}, 2);
    { auto m = model::open(src); CHECK(m->aot().available() && m->aot().stats().origin.find("sidecar") == 0); }
    // nothing signs a sidecar: a signed model, or strict loading, ignores it
    auto kp = modelfile::generate_keypair();
    auto sg = tmp("side_signed.fpk");
    modelfile::sign_file(src, sg, kp);
    modelfile::aot::write_sidecar(sg + ".fplan", pl.blob);
    { auto m = model::open(sg); CHECK(!m->aot().available() && m->aot().stats().rejected == 1); }
    forge_test::set_env("FORGE_TRUSTED_KEYS", modelfile::detail::to_hex(kp.pub).c_str());
    { auto m = model::open(sg); session s(m); std::vector<int> ids = {1, 2, 3, 4}; s.forward(ids); CHECK(m->aot().stats().used == 0); }
    forge_test::set_env("FORGE_TRUSTED_KEYS", nullptr);
    // a damaged sidecar is ignored
    { std::ofstream f(src + ".fplan", std::ios::binary | std::ios::trunc); f << "FORGEAOT garbage"; }
    expect_identical(src, src, {4, 1}, {}, 0);
}

TEST(aot_write_section_keeps_the_file_intact) {
    auto pl = plans_of(model_f32(), {.shapes = {1}});
    auto f = with_plans(model_f32(), pl.blob, "keep.fpk");
    modelfile::reader a(model_f32()), b(f);
    CHECK(b.verify().empty());
    for (auto& [name, s] : a.sections()) {
        CHECK(b.has_section(name));
        CHECK(b.sections().at(name).offset == s.offset && b.sections().at(name).size == s.size);   // tensors do not move
        auto x = a.raw_section_bytes(name), y = b.raw_section_bytes(name);
        CHECK(x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size()) == 0);
    }
    // replacing and removing
    auto pl2 = plans_of(model_f32(), {.shapes = {1, 2}});
    modelfile::aot::write_section(f, f, pl2.blob);
    { modelfile::reader r(f); CHECK(r.verify().empty() && r.raw_section_bytes("aot").size() == pl2.blob.size()); }
    modelfile::aot::write_section(f, f, {});
    { modelfile::reader r(f); CHECK(r.verify().empty() && !r.has_section("aot")); CHECK(r.file_size() == a.file_size()); }
}

#if defined(FORGE_TEST_BINARY_DIR)
// The golden models (written by the golden_* tests into the build directory), when they are there: Mixtral (sparse
// MoE), DeepSeek-V2 (MLA + MoE, with and without the shared-expert variant), the imported q8_0 Llama, and the MoE
// models re-imported with quantised weights (expert weights stay encoded).
TEST(aot_golden_moe_mla_and_quantised_models) {
    const std::string dir = FORGE_TEST_BINARY_DIR;
    int n = 0;
    for (const char* name : {"golden_moe/mixtral.fpk", "golden_moe/deepseek_v2.fpk", "golden_moe/deepseek_v2_lite.fpk", "golden/tiny_q8_0.fpk", "golden/tiny_f32.fpk"}) {
        auto src = dir + "/" + name;
        if (!fs::exists(src)) continue;
        auto pl = plans_of(src, {.shapes = {1, 6}, .all_rows = true});
        auto f = with_plans(src, pl.blob, "golden" + std::to_string(n) + ".fpk");
        expect_identical(f, src, {6, 1, 1, 6, 1}, {}, 2);
        expect_identical(f, src, {6, 1}, {}, 2, true);
        ++n;
    }
    for (const char* hf : {"golden_moe/mixtral", "golden_moe/deepseek_v2"}) for (const char* codec : {"q8_0", "q4_k"}) {
        if (!fs::exists(dir + "/" + hf + "/config.json")) continue;
        auto src = tmp(std::string("moe_") + codec + std::to_string(n) + ".fpk");
        import::import_options io; io.codec = codec;
        import::import_hf(dir + "/" + hf, src, io);
        auto pl = plans_of(src, {.shapes = {1, 6}});
        expect_identical(with_plans(src, pl.blob, "goldenq" + std::to_string(n) + ".fpk"), src, {6, 1, 1, 6, 1}, {.kv_codec = ""}, 2);
        ++n;
    }
    if (!n) std::printf("  (no golden models in %s: run the golden_tiny_llama and golden_moe_mla tests first)\n", dir.c_str());
    else std::printf("  %d golden model(s)\n", n);
}
#endif

// A valid blob that fits the fuzz harness (fuzz/fuzz_aot.cpp: every weight is an 8x8 f32 matrix). It is the seed of
// the fuzz corpus: FORGE_AOT_WRITE_SEED=<path> writes it; the checked-in copy must still load, or it has to be
// regenerated after a format change.
TEST(aot_fuzz_seed_blob) {
    using namespace forge::graph;
    static std::vector<float> w(64, 0.125f);
    tensor_graph g;
    auto x = g.input("x", {2, 8});
    auto wq = g.weight("w.q", {8, 8}, w.data()), wk = g.weight("w.k", {8, 8}, w.data());
    auto h = g.apply(ops::op::matmul_t, {x, wq});
    auto q3 = g.apply(ops::op::reshape, {h}, {{"shape", std::vector<i64>{2, 2, 4}}});
    auto k3 = g.apply(ops::op::reshape, {g.apply(ops::op::matmul_t, {x, wk})}, {{"shape", std::vector<i64>{2, 2, 4}}});
    auto ks = g.state_append("kv.k", k3), vs = g.state_append("kv.v", k3);
    auto at = g.apply(ops::op::attention, {q3, ks, vs}, {{"causal", 1.0}});
    auto flat = g.apply(ops::op::reshape, {at}, {{"shape", std::vector<i64>{2, 8}}});
    auto y = g.add(g.mul(g.apply(ops::op::silu, {flat}), h), x);
    g.output("y", g.apply(ops::op::softmax, {y}));
    state_store st; st.capacity = 16; st.page_rows = 1;
    auto p = compile(std::move(g), &st);
    aot::builder b(aot::model_key("seed", "seed"));
    aot::target t; t.seq = 2; b.add(t, *p);
    auto blob = b.finish("scalar");
    aot::binder bind; bind.weight = [](const std::string&) { return std::pair<shape_t, const float*>{shape_t{8, 8}, w.data()}; };
    auto load = [&](std::span<const std::uint8_t> bytes) {
        aot::image img(bytes);
        state_store s2; s2.capacity = 16; s2.page_rows = 1;
        auto q = aot::instantiate(img, img.entries().at(0), bind, &s2, {}, 64u << 10);
        std::vector<float> in(16); for (std::size_t i = 0; i < in.size(); ++i) in[i] = float(i) * 0.25f - 1.f;
        executor ea(*q, &s2), er(*p, &st);
        ea.set_input("x", in); er.set_input("x", in);
        ea.run(); er.run();
        return same_bits(ea.output("y").data, er.output("y").data);
    };
    CHECK(load(blob));
    if (const char* out = std::getenv("FORGE_AOT_WRITE_SEED"); out && *out) modelfile::aot::write_sidecar(out, blob);
    if (std::ifstream f("fuzz/corpus/aot/seed.fplan", std::ios::binary); f) {
        std::vector<std::uint8_t> seed((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        CHECK_MSG(load(seed), "fuzz/corpus/aot/seed.fplan no longer loads: regenerate it (FORGE_AOT_WRITE_SEED=fuzz/corpus/aot/seed.fplan test_aot)");
    }
}

// Start-up (open the model, run a 16-token prompt, decode one token) and decode throughput, compiled at run time vs
// precompiled. Printed for docs/aot-models.md; the only assertion is that the outputs agree.
TEST(aot_startup_and_throughput) {
    auto src = forge_test::make_tiny_model(tmp("deep.fpk"), {.d = 64, .layers = 12, .heads = 4, .kv = 2, .ffn = 256});
    auto pl = plans_of(src, {.shapes = {1, 16}});
    auto f = with_plans(src, pl.blob, "deep_aot.fpk");
    std::vector<int> prompt; for (int i = 0; i < 16; ++i) prompt.push_back(1 + i * 7);
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    { session warm(model_f32()); warm.forward(prompt); }   // oracle checks of the lowerings, thread pool: once per process
    std::vector<float> ref;
    double best[2] = {1e9, 1e9}, tps[2] = {0, 0};
    for (int rep = 0; rep < 5; ++rep) for (int aot = 0; aot < 2; ++aot) {
        auto t0 = clk::now();
        auto m = model::open(aot ? f : src); if (!aot) m->set_aot(false);
        session s(m);
        s.forward(prompt);
        int one = 9; auto l = s.forward(std::span<const int>(&one, 1));
        best[aot] = std::min(best[aot], ms(clk::now() - t0));
        if (ref.empty()) ref = l; else CHECK(same_bits(ref, l));
        auto t1 = clk::now();
        for (int i = 0; i < 64; ++i) { int tk = 3 + i; s.forward(std::span<const int>(&tk, 1)); }
        tps[aot] = std::max(tps[aot], 64.0 / std::chrono::duration<double>(clk::now() - t1).count());
        CHECK(m->aot().stats().used == (aot ? 2u : 0u));
    }
    // the part precompiled plans replace: getting the two plans (prefill 16, decode 1), without running them
    double plan_ms[2] = {1e9, 1e9}, open_ms = 1e9;
    for (int rep = 0; rep < 9; ++rep) for (int aot = 0; aot < 2; ++aot) {
        auto t0 = clk::now();
        auto m = model::open(aot ? f : src); if (!aot) m->set_aot(false);
        graph::state_store st; st.capacity = 512; st.page_rows = 256;
        auto t1 = clk::now();
        m->plan_for(16, false, &st); m->plan_for(1, false, &st);
        plan_ms[aot] = std::min(plan_ms[aot], ms(clk::now() - t1));
        open_ms = std::min(open_ms, ms(t1 - t0));
    }
    std::printf("  12-layer d=64 tiny model, %zu-byte plan blob\n", pl.blob.size());
    std::printf("  opening the model (map, manifest, blueprint, tokenizer): %.2f ms\n", open_ms);
    std::printf("  two plans (prefill 16, decode 1): compiled at run time %.2f ms, from precompiled plans %.2f ms\n", plan_ms[0], plan_ms[1]);
    std::printf("  start-up to the first decoded token: compiled at run time %.2f ms, precompiled plans %.2f ms\n", best[0], best[1]);
    std::printf("  decode: compiled at run time %.0f tok/s, precompiled plans %.0f tok/s (the same plan either way)\n", tps[0], tps[1]);
}
