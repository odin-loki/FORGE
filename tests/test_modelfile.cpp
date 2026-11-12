#include "check.hpp"
#include <forge/inference/inference.hpp>
#include <forge/import/gguf.hpp>
#include <cstdio>
#include <filesystem>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#include <fstream>
#include <random>
#include "tiny_model.hpp"
#include <cstring>
static std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / ("forge_test_pack_" + std::to_string(::getpid()) + "_" + n)).string(); }

using namespace forge;

TEST(blake3_matches_reference_implementation) {
    struct kat { std::size_t n; const char* hex; };
    static const kat k[] = {
        {0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"},
        {1, "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"},
        {64, "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"},
        {1023, "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"},
        {1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"},
        {1025, "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444"},
        {2048, "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a"},
        {3073, "7124b49501012f81cc7f11ca069ec9226cecb8a2c850cfe644e327d22d3e1cd3"},
        {100000, "d93c23eedaf165a7e0be908ba86f1a7a520d568d2d13cde787c8580c5c72cc54"},
    };
    for (auto& e : k) {
        std::vector<std::uint8_t> b(e.n); for (std::size_t i = 0; i < e.n; ++i) b[i] = std::uint8_t(i % 251);
        CHECK_MSG(hash::blake3::hex(hash::blake3::of(b)) == e.hex, "n=%zu", e.n);
        hash::blake3 h; std::size_t p = 0;  // streaming in odd pieces
        while (p < b.size()) { std::size_t t = std::min<std::size_t>(77, b.size() - p); h.update(b.data() + p, t); p += t; }
        CHECK(hash::blake3::hex(h.final()) == e.hex);
    }
}

TEST(json_roundtrip_and_errors) {
    auto v = json::parse(R"({"a": [1, 2.5, -3e2, true, null], "s": "x\"\u00e9\ud83d\ude00", "o": {}})");
    CHECK(v["a"][2].num() == -300); CHECK(v["s"].str() == "x\"\xc3\xa9\xf0\x9f\x98\x80");
    CHECK(json::parse(json::dump(v))["a"][1].num() == 2.5);
    bool threw = false; try { json::parse("{\"a\": }"); } catch (const forge_exception&) { threw = true; } CHECK(threw);
}

TEST(fpk_write_read_verify_and_corruption) {
    auto path = std::filesystem::temp_directory_path() / ("forge_test_" + std::to_string(::getpid()) + ".fpk");
    std::vector<float> a(1000); for (std::size_t i = 0; i < a.size(); ++i) a[i] = float(i) * 0.01f - 3.f;
    {
        modelfile::writer w;
        w.set_manifest(json::object{{"k", "v"}});
        w.set_section("blueprint", "blueprint \"x\" graph { y = x } output y : f32[1]");
        w.add_tensor("a", {10, 100}, a, "f32");
        w.add_tensor("b", {1000}, a, "q8_0");
        w.add_tensor("c", {7}, std::span<const float>(a.data(), 7), "q4_k");  // falls back to f16
        w.write(path.string());
    }
    {
        modelfile::reader r(path.string());
        CHECK(r.manifest()["k"].str() == "v");
        CHECK(r.verify().empty());
        auto t = r.load("a"); CHECK(t.shape == shape_t({10, 100}) && t.data == a);
        auto q = r.load("b"); double e = 0; for (std::size_t i = 0; i < a.size(); ++i) e = std::max(e, double(std::fabs(q.data[i] - a[i])));
        CHECK(e < 0.03);
        CHECK(r.entry("c").codec == "f16");
        for (auto& [n, s] : r.sections()) CHECK(s.offset % modelfile::alignment == 0);
    }
    {   // flip one byte inside tensor data -> verify must report it
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        modelfile::reader r(path.string());
        auto off = r.sections().at("data").offset + r.entry("a").offset + 17;
        f.seekp(std::streamoff(off)); char c = 0x5a; f.write(&c, 1);
    }
    {
        modelfile::reader r(path.string());
        auto bad = r.verify();
        CHECK(bad.size() == 2);  // the data section and tensor "a"
    }   // unmap before deleting: Windows refuses to delete a mapped file
    std::filesystem::remove(path);
}

static const char* kTok = R"({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "Ġ": 3, "ab": 4, "Ġab": 5, "!": 6},
  "merges": ["a b", "Ġ ab"]}, "added_tokens": [{"id": 7, "content": "<s>"}]})";

TEST(bpe_tokenizer_merges_and_roundtrip) {
    modelfile::bpe_tokenizer t(kTok);
    auto ids = t.encode("<s>ab abc!");
    CHECK((ids == std::vector<int>{7, 4, 5, 2, 6}));
    CHECK(t.decode(ids) == "<s>ab abc!");
}

TEST(blueprint_parse_build_and_errors) {
    const char* src = R"(
blueprint "mlp" opset 1
params { d: 4  scale: { a: 2 } }
input x : f32[n, d]
graph {
  h = x * scale.a + 1
  repeat i in 0..3 { h = tanh(h) }
  y = softmax(h)
}
output y : f32[n, d]
session one_shot)";
    auto bp = modelfile::parse_blueprint(src);
    CHECK(bp.name == "mlp" && bp.param("d") == 4 && bp.param("scale.a") == 2 && bp.session == "one_shot");
    modelfile::build_options bo; bo.dims["n"] = 3;
    auto g = modelfile::build_graph(bp, bo);
    auto p = graph::compile(g);
    graph::executor e(*p);
    std::vector<float> x(12, 0.5f); e.set_input("x", x); e.run();
    auto y = e.output("y");
    CHECK(y.shape == shape_t({3, 4}) && std::fabs(y.data[0] - 0.25f) < 1e-6f);
    // errors carry a line number
    bool threw = false;
    try { modelfile::build_graph(modelfile::parse_blueprint("blueprint \"e\"\ninput x : f32[2]\ngraph {\n y = nope(x)\n}\noutput y : f32[2]"), {}); }
    catch (const forge_exception& ex) { threw = std::string(ex.what()).find("blueprint:4") != std::string::npos; }
    CHECK(threw);
}

TEST(sampler_chain) {
    std::vector<float> l = {0.f, 5.f, 4.9f, -1.f};
    inference::sampler g; CHECK(g.sample(l) == 1);
    inference::sampler rp({.repetition_penalty = 2.f}); std::vector<int> h = {1}; CHECK(rp.sample(l, h) == 2);
    inference::sampler k1({.temperature = 1.f, .top_k = 1, .seed = 3}); for (int i = 0; i < 20; ++i) CHECK(k1.sample(l) == 1);
    inference::sampler t({.temperature = 1.f, .top_p = 0.9f, .seed = 7});
    int c[4] = {}; for (int i = 0; i < 2000; ++i) ++c[t.sample(l)];
    CHECK(c[0] == 0 && c[3] == 0 && c[1] > c[2] && c[2] > 700);
    inference::sampler mp({.temperature = 1.f, .min_p = 0.95f, .seed = 1}); for (int i = 0; i < 20; ++i) CHECK(mp.sample(l) == 1);
}

TEST(blueprint_composite_ops) {
    auto src = R"(blueprint "c"
op cube(x) { return x * x * x }
op my_gelu(x) {
  inner = 0.7978845608 * (x + 0.044715 * cube(x))
  return 0.5 * x * (1 + tanh(inner))
}
op silu(x) { return x * 0 }
op twice(a, b) { s = my_gelu(a) + b  return s * 2 }
input x : f32[8]
graph { y = my_gelu(x)  z = silu(x)  w = twice(x, x) }
output y : f32[8]
output z : f32[8]
output w : f32[8]
)";
    auto bp = modelfile::parse_blueprint(src); CHECK(bp.composites.size() == 4);
    auto p = graph::compile(modelfile::build_graph(bp, {})); graph::executor ex(*p);
    std::vector<float> x = {-3, -1, -0.5f, 0, 0.25f, 1, 2, 4}; ex.set_input("x", x); ex.run();
    auto y = ex.output_view("y"), z = ex.output_view("z"), w = ex.output_view("w");
    for (std::size_t i = 0; i < 8; ++i) {
        double gt = 0.5 * x[i] * (1 + std::tanh(0.7978845608 * (x[i] + 0.044715 * x[i] * x[i] * x[i])));
        CHECK(std::fabs(y.data[i] - gt) < 1e-5);
        CHECK(std::fabs(z.data[i] - x[i] / (1 + std::exp(-x[i]))) < 1e-5);   // spec op wins over the composite
        CHECK(std::fabs(w.data[i] - 2 * (gt + x[i])) < 1e-4);
    }
    CHECK_THROWS(modelfile::build_graph(modelfile::parse_blueprint("blueprint \"e\"\nop f(a, b) { return a }\ninput x : f32[2]\ngraph { y = f(x) }\noutput y : f32[2]"), {}));
    CHECK_THROWS(modelfile::build_graph(modelfile::parse_blueprint("blueprint \"e\"\nop f(a) { return f(a) }\ninput x : f32[2]\ngraph { y = f(x) }\noutput y : f32[2]"), {}));
    CHECK_THROWS(modelfile::parse_blueprint("blueprint \"e\"\nop f(a) { return a }\nop f(a) { return a }\ninput x : f32[2]\ngraph { y = f(x) }\noutput y : f32[2]"));
    CHECK_THROWS(modelfile::build_graph(modelfile::parse_blueprint("blueprint \"e\"\nop f(a) { b = a }\ninput x : f32[2]\ngraph { y = f(x) }\noutput y : f32[2]"), {}));
}

// Tensor alignment: 64 (default of the writers), 4096, 65536; recorded in the manifest; kept by every tool that
// rewrites the container (sign, encrypt, aot).
TEST(fpk_tensor_alignment) {
    std::mt19937 rng(3); std::normal_distribution<float> N(0, 1);
    std::vector<float> a(1000), b(64), c(7);
    for (auto* v : {&a, &b, &c}) for (auto& x : *v) x = N(rng);
    auto fill = [&](auto& w) { w.add_tensor("a", {10, 100}, a); w.add_tensor("b", {2, 32}, b, "q8_0"); w.add_tensor("c", {7}, c, "f16");
                               w.set_manifest(json::value(json::object{{"format", "fpk"}})); w.set_section("blueprint", "blueprint \"t\"\ninput x : f32[1]\ngraph { y = relu(x) }\noutput y : f32[1]\n"); };
    auto check = [&](const std::string& f, std::size_t al, const char* what) {
        modelfile::reader r(f);
        bool ok = r.tensor_alignment() == al && r.misaligned_tensors().empty() && r.verify().empty() && r.load("a").data == a;
        for (auto& [n, e] : r.tensors()) ok = ok && (r.sections().at("data").offset + e.offset) % al == 0;
        for (auto& [n, sc] : r.sections()) ok = ok && sc.offset % 64 == 0;
        if (!ok) std::fprintf(stderr, "  alignment %zu not kept by %s\n", al, what);
        return ok;
    };
    auto key = modelfile::generate_model_key(); auto kp = modelfile::generate_keypair();
    std::size_t size64 = 0;
    for (std::size_t al : {std::size_t(64), std::size_t(4096), std::size_t(65536)}) {
        auto f = tmp("al.fpk"), g = tmp("al_s.fpk"), h = tmp("al_e.fpk"), s2 = tmp("al_st.fpk");
        { modelfile::writer w; w.set_alignment(al); CHECK(w.tensor_alignment() == al); fill(w); w.write(f); }
        CHECK(check(f, al, "writer"));
        { modelfile::reader r(f); CHECK(r.manifest().num_or("tensor_alignment", 0) == double(al)); }
        if (al == 64) size64 = std::size_t(std::filesystem::file_size(f));
        else CHECK(std::filesystem::file_size(f) > size64 + al);   // the padding is real: why small files stay at 64
        modelfile::sign_file(f, g, kp); CHECK(check(g, al, "sign_file"));
        { std::vector<std::uint8_t> blob(100, 7); modelfile::aot::write_section(g, g, blob); CHECK(check(g, al, "aot::write_section")); }   // forge aot keeps offsets
        modelfile::encrypt_file(g, h, key);
        { modelfile::reader r(h, key); CHECK(r.tensor_alignment() == al && r.misaligned_tensors().empty() && r.verify().empty() && r.load("a").data == a); }
        modelfile::sign_file(h, h, kp);
        { modelfile::reader r(h, key); CHECK(r.misaligned_tensors().empty() && modelfile::verify_signature(r).digest_ok); }
        { modelfile::stream_writer w(s2); w.set_alignment(al); fill(w); w.finish(); }
        CHECK(check(s2, al, "stream_writer"));
        modelfile::sign_file(s2, s2, kp); CHECK(check(s2, al, "sign_file after stream_writer"));   // data moves, its phase is kept
        modelfile::encrypt_file(s2, h, key);
        { modelfile::reader r(h, key); CHECK(r.misaligned_tensors().empty() && r.verify().empty()); }
    }
    {   // the option is set before the first tensor, and only to a valid value
        modelfile::writer w; CHECK_THROWS(w.set_alignment(100)); CHECK_THROWS(w.set_alignment(32)); CHECK_THROWS(w.set_alignment(std::size_t(1) << 21));
        w.add_tensor("a", {10, 100}, a); CHECK_THROWS(w.set_alignment(4096));
    }
    {   // a file without the manifest key is a 64-byte-aligned file, as before
        auto f = tmp("al_old.fpk");
        { modelfile::writer w; fill(w); w.write(f); }
        std::vector<std::pair<std::string, std::string>> secs;
        { modelfile::reader r(f); for (auto& [n, sc] : r.sections()) { auto bts = r.raw_section_bytes(n); secs.emplace_back(n, n == "manifest" ? std::string("{\"format\":\"fpk\"}") : std::string(reinterpret_cast<const char*>(bts.data()), bts.size())); } }
        modelfile::detail::write_sections(f, secs);
        modelfile::reader r(f); CHECK(r.tensor_alignment() == 64 && r.misaligned_tensors().empty() && r.verify().empty() && r.layout().align == 64 && r.layout().phase == 0);
    }
    CHECK(modelfile::default_tensor_alignment(0) == 64 && modelfile::default_tensor_alignment((64u << 20) - 1) == 64 && modelfile::default_tensor_alignment(64u << 20) == 4096);
    CHECK(import::import_alignment({}, "no/such/file") == 64);
    { import::import_options o; o.align = 65536; CHECK(import::import_alignment(o, "x") == 65536); o.align = 128; CHECK_THROWS(import::import_alignment(o, "x")); }
}

// Chunked encryption (the default of encrypt_file): tensors are decrypted one at a time, on use, and every way of
// rearranging the sealed bytes is refused.
TEST(fpk_chunked_encryption) {
    using secs_t = std::map<std::string, std::string>;
    auto src = forge_test::make_tiny_model(tmp("ce_src.fpk")), src2 = forge_test::make_tiny_model(tmp("ce_src2.fpk"), {.seed = 11});
    auto key = modelfile::generate_model_key();
    auto set = [](json::object& o, std::string_view k, json::value v) { for (auto& [kk, vv] : o) if (kk == k) { vv = std::move(v); return; } o.emplace_back(std::string(k), std::move(v)); };
    auto get = [](json::object& o, std::string_view k) -> json::value& { for (auto& [kk, vv] : o) if (kk == k) return vv; raise(errc::not_found, "no key {}", k); };
    auto read_secs = [](const std::string& f) { secs_t m; modelfile::reader r(f); for (auto& [n, s] : r.sections()) { auto b = r.raw_section_bytes(n); m[n].assign(reinterpret_cast<const char*>(b.data()), b.size()); } return m; };
    auto write_secs = [](const std::string& f, const secs_t& m) { modelfile::detail::write_sections(f, {m.begin(), m.end()}); };
    for (std::size_t chunk : {std::size_t(4096), std::size_t(1000), std::size_t(1) << 20}) {
        auto dst = tmp("ce.fpk");
        modelfile::encrypt_file(src, dst, key, {.chunk_bytes = chunk});
        modelfile::reader plain(src), locked(dst), open(dst, key);
        auto hdr = json::parse(locked.section_text("encryption"));
        CHECK(hdr.str_or("alg", "") == "xchacha20-poly1305-chunked" && std::size_t(hdr.num_or("chunk_bytes", 0)) == chunk);
        CHECK(std::size_t(hdr.num_or("chunks", 0)) == (plain.data_size() + chunk - 1) / chunk && locked.sections().at("enc_macs").size == 16 * std::size_t(hdr.num_or("chunks", 0)));
        CHECK(locked.locked() && locked.verify().empty()); CHECK_THROWS(locked.load("lm_head")); CHECK_THROWS(locked.tensor_bytes("lm_head"));
        CHECK(locked.manifest().str_or("format", "") == "fpk");
        CHECK(open.encrypted() && open.chunked_encryption() && open.plaintext_resident() == 0);   // nothing is decrypted at open
        // the tensor index and every offset are what the plain file has: only the bytes of `data` differ
        CHECK(open.section_text("tensors") == plain.section_text("tensors"));
        { auto raw = open.raw_section_bytes("data"), pd = plain.section_bytes("data"); CHECK(raw.size() == pd.size() && std::memcmp(raw.data(), pd.data(), pd.size()) != 0); }
        auto one = open.tensor_bytes("layer.1.wq"), ref = plain.tensor_bytes("layer.1.wq");
        CHECK(one.size() == ref.size() && std::memcmp(one.data(), ref.data(), ref.size()) == 0);
        CHECK(open.plaintext_resident() == ref.size() && open.plaintext_resident() < plain.data_size() / 4);   // one tensor, not the file
        CHECK(reinterpret_cast<std::uintptr_t>(one.data()) % 64 == 0);
        CHECK(open.tensor_bytes("layer.1.wq").data() == one.data());                                          // cached, not decrypted again
        for (auto& [n, e] : plain.tensors()) CHECK(open.load(n).data == plain.load(n).data);                  // load() decodes from a wiped scratch copy
        CHECK(open.plaintext_resident() == ref.size());
        CHECK(open.verify().empty());
        open.release_plaintext(); CHECK(open.plaintext_resident() == 0);
        const_cast<modelfile::reader&>(open).lock_plaintext(true);
        auto lk = open.tensor_bytes("tok_emb"), lref = plain.tensor_bytes("tok_emb");
        CHECK(lk.size() == lref.size() && std::memcmp(lk.data(), lref.data(), lref.size()) == 0);
        std::printf("  chunk %zu: %zu chunks, locked plaintext: %s\n", chunk, std::size_t(hdr.num_or("chunks", 0)), open.plaintext_locked() ? "yes" : "refused by the OS (limit)");
        auto bad = key; bad[0] ^= 1; CHECK_THROWS(modelfile::reader(dst, bad));
        CHECK_THROWS(modelfile::encrypt_file(dst, tmp("ce2.fpk"), key));
        // section_bytes("data") still works (whole-section fallback for old callers)
        { modelfile::reader o2(dst, key); auto d = o2.section_bytes("data"), pd = plain.section_bytes("data"); CHECK(d.size() == pd.size() && std::memcmp(d.data(), pd.data(), pd.size()) == 0); }
    }
    // ---- tampering. Each case rewrites the container so that the section table and its hashes are consistent
    // (what an attacker without the key can do); the AEAD has to catch it. `rejected` = the reader refuses to open,
    // or verify() reports it and reading the tensors fails.
    const std::size_t chunk = 4096;
    auto a = tmp("ce_a.fpk"), b = tmp("ce_b.fpk"), c = tmp("ce_c.fpk"), t = tmp("ce_t.fpk");
    modelfile::encrypt_file(src, a, key, {.chunk_bytes = chunk});
    modelfile::encrypt_file(src, b, key, {.chunk_bytes = chunk});    // same model, same key, another nonce
    modelfile::encrypt_file(src2, c, key, {.chunk_bytes = chunk});   // another model of the same layout, same key
    const secs_t A = read_secs(a), B = read_secs(b), C = read_secs(c);
    const std::size_t n_chunks = A.at("enc_macs").size() / 16;
    CHECK(n_chunks > 8 && A.at("data") != B.at("data"));
    auto rejected = [&](const secs_t& m, const char* what) {
        write_secs(t, m);
        try {
            modelfile::reader r(t, key);
            bool read_fails = false;
            try { for (auto& [n, e] : r.tensors()) (void)r.load(n); } catch (const forge_exception& e) { read_fails = e.code == errc::integrity_error; }
            bool tb_fails = false;
            try { for (auto& [n, e] : r.tensors()) (void)r.tensor_bytes(n); } catch (const forge_exception& e) { tb_fails = e.code == errc::integrity_error; }
            bool ok = read_fails && tb_fails && !r.verify().empty();
            if (!ok) std::fprintf(stderr, "  tamper case '%s' was not rejected\n", what);
            return ok;
        } catch (const forge_exception&) { return true; }
    };
    { write_secs(t, A); modelfile::reader r(t, key); CHECK(r.verify().empty()); }   // the rewrite itself is harmless
    { auto m = A; m["data"][chunk * 3 + 17] ^= 0x01; CHECK(rejected(m, "bit flip in a ciphertext chunk")); }
    { auto m = A; m["data"].back() ^= 0x80; CHECK(rejected(m, "bit flip in the last (short) chunk")); }
    { auto m = A; m["enc_macs"][16 * 2 + 5] ^= 0x10; CHECK(rejected(m, "bit flip in a tag")); }
    {   // reorder: chunks 1 and 2 swapped together with their tags
        auto m = A; auto& d = m["data"]; auto& mc = m["enc_macs"];
        std::swap_ranges(d.begin() + chunk, d.begin() + 2 * chunk, d.begin() + 2 * chunk);
        std::swap_ranges(mc.begin() + 16, mc.begin() + 32, mc.begin() + 32);
        CHECK(rejected(m, "two chunks reordered"));
    }
    {   // truncate: the last chunk and its tag dropped, header untouched / header adjusted / header and index adjusted
        auto m = A; auto last = (n_chunks - 1) * chunk;
        m["data"].resize(last); m["enc_macs"].resize(16 * (n_chunks - 1));
        CHECK(rejected(m, "truncated, header untouched"));
        auto ho = json::parse(A.at("encryption")).obj();
        set(ho, "chunks", double(n_chunks - 1)); set(ho, "data_bytes", double(last));
        m["encryption"] = json::dump(json::value(ho));
        CHECK(rejected(m, "truncated, header adjusted"));
        json::array keep;   // drop the tensors that reached into the removed chunk, so every bounds check passes
        for (auto& e : json::parse(A.at("tensors")).arr()) if (e["offset"].num() + e["bytes"].num() <= double(last)) keep.push_back(e);
        CHECK(!keep.empty() && keep.size() < json::parse(A.at("tensors")).arr().size());
        m["tensors"] = json::dump(json::value(keep));
        CHECK(rejected(m, "truncated, header and index adjusted"));
    }
    {   // splice: a chunk (with its tag) of another file sealed with the same key
        auto m = A; m["data"].replace(chunk * 4, chunk, B.at("data"), chunk * 4, chunk); m["enc_macs"].replace(16 * 4, 16, B.at("enc_macs"), 16 * 4, 16);
        CHECK(rejected(m, "chunk spliced from the same model sealed again"));
        m = A; m["data"].replace(chunk * 4, chunk, C.at("data"), chunk * 4, chunk); m["enc_macs"].replace(16 * 4, 16, C.at("enc_macs"), 16 * 4, 16);
        CHECK(rejected(m, "chunk spliced from another model"));
        m = A; m["data"] = B.at("data"); m["enc_macs"] = B.at("enc_macs");
        CHECK(rejected(m, "all ciphertext of another sealing under this header"));
        m = A; m["data"] = C.at("data"); m["enc_macs"] = C.at("enc_macs"); m["encryption"] = C.at("encryption");
        CHECK(rejected(m, "another model's sealed data under this index"));   // the index (tensor hashes) is bound into every chunk
    }
    {   // header edits
        auto ho = json::parse(A.at("encryption")).obj(); auto m = A;
        set(ho, "chunk_bytes", double(chunk * 2)); m["encryption"] = json::dump(json::value(ho));
        CHECK(rejected(m, "chunk size altered"));
        ho = json::parse(A.at("encryption")).obj(); auto nn = get(ho, "nonce").str(); nn[47] = nn[47] == '0' ? '1' : '0'; set(ho, "nonce", nn);
        m = A; m["encryption"] = json::dump(json::value(ho));
        CHECK(rejected(m, "nonce altered"));
    }
    {   // index edit: two tensors of the same shape exchange their offsets
        auto idx = json::parse(A.at("tensors")).arr(); json::object* p0 = nullptr; json::object* p1 = nullptr;
        for (auto& e : idx) { if (e["name"].str() == "layer.0.wq") p0 = &e.obj_mut(); if (e["name"].str() == "layer.1.wq") p1 = &e.obj_mut(); }
        CHECK(p0 && p1);
        if (p0 && p1) { auto o0 = get(*p0, "offset"), o1 = get(*p1, "offset"); set(*p0, "offset", o1); set(*p1, "offset", o0); }
        auto m = A; m["tensors"] = json::dump(json::value(idx));
        CHECK(rejected(m, "index offsets exchanged"));
    }
    // a session on the sealed file: same logits as the plain file, and only tensor-sized plaintext is held
    forge_test::set_env("FORGE_MODEL_KEY", modelfile::detail::hex(key.data(), 32).c_str());
    {
        inference::session s(a), p(src);
        CHECK(s.forward(std::vector<int>{1, 2, 3}) == p.forward(std::vector<int>{1, 2, 3}));
        CHECK(s.file().chunked_encryption() && s.file().plaintext_resident() > 0 && s.file().plaintext_resident() <= s.file().data_size());
    }
    forge_test::set_env("FORGE_MODEL_KEY", nullptr);
}

TEST(fpk_encryption_hook) {
    auto src = forge_test::make_tiny_model(tmp("enc_src.fpk")), dst = tmp("enc.fpk");
    // the single-AEAD scheme (what files sealed before the chunked scheme carry): still written on request, still read
    auto key = modelfile::generate_model_key(); modelfile::encrypt_file(src, dst, key, {.single_aead = true});
    {   // readers map the file; Windows cannot replace a mapped file, so they close before dst is rewritten
    modelfile::reader plain(src), locked(dst);
    CHECK(json::parse(locked.section_text("encryption")).str_or("alg", "") == "xchacha20-poly1305" && !locked.has_section("enc_macs"));
    CHECK(locked.locked()); CHECK(locked.verify().empty()); CHECK_THROWS(locked.load("lm_head"));
    CHECK(locked.manifest().str_or("format", "") == "fpk");                  // metadata stays readable
    modelfile::reader open(dst, key); CHECK(open.encrypted() && !open.chunked_encryption()); CHECK(open.verify().empty());
    for (auto& [n, e] : plain.tensors()) CHECK(open.load(n).data == plain.load(n).data);
    { auto raw = open.raw_section_bytes("data"), pd = plain.section_bytes("data"); CHECK(raw.size() == pd.size() && std::memcmp(raw.data(), pd.data(), pd.size()) != 0); }
    auto bad = key; bad[0] ^= 1; CHECK_THROWS(modelfile::reader(dst, bad));
    CHECK_THROWS(modelfile::encrypt_file(dst, tmp("enc2.fpk"), key));
    // tamper with one ciphertext byte: decryption must fail
    { std::fstream f(dst, std::ios::in | std::ios::out | std::ios::binary); auto off = std::streamoff(locked.sections().at("data").offset + 5);
      f.seekg(off); char c; f.get(c); f.seekp(off); f.put(char(c ^ 0x40)); }
    CHECK_THROWS(modelfile::reader(dst, key));
    }
    // sign + verify an encrypted file without the key, then load it with the key from the environment
    modelfile::encrypt_file(src, dst, key);
    auto kp = modelfile::generate_keypair(); modelfile::sign_file(dst, dst, kp);
    { modelfile::reader r(dst); auto rep = modelfile::verify_signature(r); CHECK(rep.digest_ok && !rep.valid_keys.empty()); }
    forge_test::set_env("FORGE_MODEL_KEY", modelfile::detail::hex(key.data(), 32).c_str());
    { inference::session s(dst); auto l = s.forward(std::vector<int>{1, 2}); inference::session p(src); CHECK(l == p.forward(std::vector<int>{1, 2})); }
    forge_test::set_env("FORGE_MODEL_KEY", nullptr);
}

// Composite ops: default and keyword parameters, definitions after the graph block, nesting, the number helpers,
// and the rejection of recursion (directly, through other composites, and in composites nothing calls).
TEST(blueprint_composite_defaults_keywords_recursion) {
    auto src = R"(blueprint "c" opset 3
params { k: 3 }
input x : f32[4]
graph {
  a = scaled(x)
  b = scaled(x, by: 4)
  c = scaled(x, 5, bias: 1)
  d = outer(x) * scalar(2)
  e = x * mod(7, k) + min(2, max(0, floor(sqrt(17))))
}
op scaled(v, by = 2, bias = -0.5) { return v * by + bias }
op outer(v) { return scaled(v, bias: 0) + inner(v) }
op inner(v) { return scaled(v, by: k) }
output a : f32[4]
output b : f32[4]
output c : f32[4]
output d : f32[4]
output e : f32[4]
)";
    auto bp = modelfile::parse_blueprint(src);
    CHECK(bp.composites.size() == 3 && bp.composites.at("scaled").defaults.size() == 2);
    auto p = graph::compile(modelfile::build_graph(bp, {})); graph::executor ex(*p);
    std::vector<float> x = {-2, 0, 0.5f, 3}; ex.set_input("x", x); ex.run();
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(std::fabs(ex.output_view("a").data[i] - (2 * x[i] - 0.5f)) < 1e-6);
        CHECK(std::fabs(ex.output_view("b").data[i] - (4 * x[i] - 0.5f)) < 1e-6);
        CHECK(std::fabs(ex.output_view("c").data[i] - (5 * x[i] + 1)) < 1e-6);
        CHECK(std::fabs(ex.output_view("d").data[i] - (10 * x[i] - 1)) < 1e-5);   // (2x + 0 + 3x - 0.5) * 2
        CHECK(std::fabs(ex.output_view("e").data[i] - (x[i] + 2)) < 1e-6);        // mod(7, 3) = 1; min(2, floor(sqrt 17)) = 2
    }
    auto bad = [](const char* ops, const char* call, const char* needle) {
        std::string s = std::string("blueprint \"e\" opset 3\n") + ops + "\ninput x : f32[2]\ngraph { y = " + call + " }\noutput y : f32[2]";
        try { (void)modelfile::build_graph(modelfile::parse_blueprint(s), {}); }
        catch (const forge_exception& e) { return std::string(e.what()).find(needle) != std::string::npos; }
        return false;
    };
    CHECK(bad("op f(a, by = 2) { return a * by }", "f(x, nope: 1)", "no parameter 'nope'"));
    CHECK(bad("op f(a, by = 2) { return a * by }", "f(x, 3, by: 4)", "given twice"));
    CHECK(bad("op f(a, b) { return a + b }", "f(x)", "missing 'b'"));
    CHECK(bad("op f(a, b) { return a + b }", "f(x, x, x)", "expects 2 arguments"));
    CHECK(bad("op f(a = 1, b) { return a + b }", "f(x, x)", "without a default"));
    CHECK(bad("op f(a, a) { return a }", "f(x, x)", "repeated"));
    CHECK(bad("op f(a) { return f(a) }", "f(x)", "is recursive (f -> f)"));
    CHECK(bad("op f(a) { return g(a) }\nop g(a) { return h(a) + 1 }\nop h(a) { return f(a) }", "x + 1", "f -> g -> h -> f"));   // never called: still rejected
    // a composite named like a spec op is shadowed by the op, so its "self call" is the op, not recursion
    CHECK(modelfile::parse_blueprint("blueprint \"e\"\nop tanh(a) { return tanh(a) * 2 }\ninput x : f32[2]\ngraph { y = tanh(x) }\noutput y : f32[2]").composites.size() == 1);
}

// Every shipped template parses once its placeholders are filled (catches a template edit that breaks the grammar).
TEST(templates_parse_with_placeholder_values) {
    int n = 0;
    for (auto& f : std::filesystem::directory_iterator(import::template_dir())) {
        if (f.path().extension() != ".fbp") continue;
        auto text = import::read_file(f.path());
        for (std::size_t p; (p = text.find("{{")) != std::string::npos;) {
            auto e = text.find("}}", p); auto key = text.substr(p + 2, e - p - 2);
            std::string v = "4";
            if (auto bar = key.find('|'); bar != std::string::npos) v = key.substr(bar + 1);   // {{key|default}}
            else if (key == "act") v = "gelu";
            else if (key == "rope_fn") v = "rope_part";
            else if (key == "logits_expr") v = "lg";
            else if (key == "pool_expr" && f.path().stem() == "bert") v = "masked_mean(x, mask)";
            else if (key == "pool_expr") v = "map_pool(x)";
            else if (key == "embed_expr") v = "p + W.pos_emb";
            else if (key == "pre_expr") v = "x";
            else if (key == "mla_q_expr") v = "linear(h, W.layer[l].wq)";
            text.replace(p, e - p + 2, v);
        }
        try { (void)modelfile::parse_blueprint(text); ++n; }
        catch (const std::exception& ex) { CHECK_MSG(false, "%s: %s", f.path().filename().string().c_str(), ex.what()); }
    }
    CHECK(n >= 16);
}

// The fuzz seed for composite ops builds a graph exactly as fuzz/fuzz_blueprint.cpp drives it.
TEST(fuzz_seed_composites_builds) {
    auto text = import::read_file(std::filesystem::path(FORGE_SOURCE_DIR) / "fuzz" / "corpus" / "blueprint" / "composites.fbp");
    auto bp = modelfile::parse_blueprint(text);
    CHECK(bp.composites.size() == 3 && bp.session == "generative_sequence");
    modelfile::build_options o; o.dims = {{"seq", 3}};
    static std::vector<float> zeros(4096, 0.f);
    o.weight = [](const std::string&) -> std::pair<shape_t, const float*> { return {shape_t{8, 8}, zeros.data()}; };
    auto g = modelfile::build_graph(bp, o);
    int biased = 0; for (auto& n : g.nodes()) biased += n.kind == graph::node_kind::op && n.op == ops::op::attention_bias;
    CHECK(biased == 2 && g.outputs().size() == 1);
}

// ---- importers: every tensor accounted for, opset range, licence capture ------------------------------------------
namespace {
namespace fs = std::filesystem;
struct hf_opts {
    std::vector<std::pair<std::string, shape_t>> extra;   // more tensors in model.safetensors
    std::vector<std::string> drop;                         // tensors left out
    std::string config_extra;                              // more config.json members (", \"k\": v")
    std::string arch = "LlamaForCausalLM", readme, license_text;
    bool biases = false;
};
// A random one-layer Llama-shaped Hugging Face directory (config.json + model.safetensors), written in-process.
std::string make_hf_dir(const std::string& name, const hf_opts& o = {}) {
    auto dir = tmp(name); fs::remove_all(dir); fs::create_directories(dir);
    const i64 H = 8, F = 16, V = 16;
    std::vector<std::pair<std::string, shape_t>> ts = {
        {"model.embed_tokens.weight", {V, H}}, {"model.norm.weight", {H}}, {"lm_head.weight", {V, H}},
        {"model.layers.0.input_layernorm.weight", {H}}, {"model.layers.0.post_attention_layernorm.weight", {H}},
        {"model.layers.0.self_attn.q_proj.weight", {H, H}}, {"model.layers.0.self_attn.k_proj.weight", {H, H}},
        {"model.layers.0.self_attn.v_proj.weight", {H, H}}, {"model.layers.0.self_attn.o_proj.weight", {H, H}},
        {"model.layers.0.mlp.gate_proj.weight", {F, H}}, {"model.layers.0.mlp.up_proj.weight", {F, H}}, {"model.layers.0.mlp.down_proj.weight", {H, F}}};
    if (o.biases) for (const char* x : {"q", "k", "v"}) ts.push_back({std::format("model.layers.0.self_attn.{}_proj.bias", x), {H}});
    for (auto& e : o.extra) ts.push_back(e);
    std::string hdr = "{", data; std::mt19937 rng(11); std::uniform_real_distribution<float> U(-0.5f, 0.5f);
    bool first = true;
    for (auto& [n, sh] : ts) {
        if (std::find(o.drop.begin(), o.drop.end(), n) != o.drop.end()) continue;
        std::string dims; for (auto d : sh) dims += (dims.empty() ? "" : ",") + std::to_string(d);
        std::size_t cnt = std::size_t(numel(sh)), b = data.size();
        for (std::size_t i = 0; i < cnt; ++i) { float f = U(rng); data.append(reinterpret_cast<const char*>(&f), 4); }
        hdr += std::format("{}\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}", first ? "" : ",", n, dims, b, data.size());
        first = false;
    }
    hdr += "}";
    { std::ofstream f(fs::path(dir) / "model.safetensors", std::ios::binary); std::uint64_t hl = hdr.size(); f.write(reinterpret_cast<const char*>(&hl), 8); f << hdr << data; }
    { std::ofstream f(fs::path(dir) / "config.json");
      f << std::format(R"({{"architectures": ["{}"], "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
 "intermediate_size": 16, "vocab_size": 16, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 64, "bos_token_id": 1, "eos_token_id": 2{}}})", o.arch, o.config_extra); }
    if (!o.readme.empty()) { std::ofstream f(fs::path(dir) / "README.md", std::ios::binary); f << o.readme; }
    if (!o.license_text.empty()) { std::ofstream f(fs::path(dir) / "LICENSE", std::ios::binary); f << o.license_text; }
    return dir;
}
// what() of the forge_exception `f` throws ("" when it does not throw)
template <class F> std::string error_of(F&& f) { try { f(); } catch (const forge_exception& e) { return e.what(); } return ""; }
bool has(const std::string& hay, std::string_view needle) { return hay.find(needle) != std::string::npos; }
}

TEST(import_hf_accounts_for_every_tensor) {
    auto out = tmp("acct.fpk");
    {   // a clean checkpoint imports with nothing left over, and the blueprint reads exactly the tensors written
        auto rep = import::import_hf(make_hf_dir("hf_ok"), out);
        CHECK(rep.unused.empty() && rep.template_name == "llama" && rep.tensors == 12);
        modelfile::reader r(out); auto bp = modelfile::parse_blueprint(r.section_text("blueprint"));
        CHECK(bp.opset == 1);
        auto ws = modelfile::blueprint_weights(bp);
        CHECK(ws.size() == r.tensors().size());
        for (auto& n : ws) CHECK_MSG(r.tensors().count(n) == 1, "blueprint reads %s", n.c_str());
        inference::session s(out); CHECK(s.forward(std::vector<int>{1, 3, 5}).size() == 16);
    }
    hf_opts stray; stray.extra = {{"model.layers.0.self_attn.q_proj.bias", {8}}, {"model.layers.0.adapter.weight", {4, 8}}};
    {   // leftovers are an error that lists each with its shape and names the way out
        auto dir = make_hf_dir("hf_stray", stray);
        auto msg = error_of([&] { import::import_hf(dir, out); });
        CHECK_MSG(has(msg, "2 source tensor(s)") && has(msg, "model.layers.0.self_attn.q_proj.bias [8]") && has(msg, "model.layers.0.adapter.weight [4,8]") && has(msg, "--allow-unused"),
                  "message: %s", msg.c_str());
        CHECK(!fs::exists(out + ".part"));                       // nothing half-written is left behind
        import::import_options allow; allow.allow_unused = true;  // the escape hatch imports and reports them
        auto rep = import::import_hf(dir, out, allow);
        CHECK(rep.unused.size() == 2 && rep.tensors == 12);
        import::import_options ign; ign.ignore_tensors = {"model.layers.0.self_attn.q_proj.bias", "*.adapter.weight"};
        CHECK(import::import_hf(dir, out, ign).unused.empty());
        ign.ignore_tensors = {"model.layers.0.self_attn.*"};      // prefix form covers one of the two
        CHECK(has(error_of([&] { import::import_hf(dir, out, ign); }), "1 source tensor(s)"));
    }
    {   // buffers no graph reads are ignored without an option
        hf_opts o; o.extra = {{"model.embed_positions.position_ids", {1, 64}}, {"model.layers.0.self_attn.masked_bias", {1}}, {"model.layers.0.self_attn.rotary_emb.inv_freq", {2}}};
        CHECK(import::import_hf(make_hf_dir("hf_buffers", o), out).unused.empty());
    }
    {   // a tensor the template needs but the checkpoint lacks: an error naming it
        hf_opts o; o.drop = {"model.layers.0.self_attn.v_proj.weight"};
        auto msg = error_of([&] { import::import_hf(make_hf_dir("hf_missing", o), out); });
        CHECK_MSG(has(msg, "self_attn.v_proj.weight") && has(msg, "llama"), "message: %s", msg.c_str());
    }
    {   // Qwen2: biases are imported (qwen2 template), and the bias-free llama template refuses the checkpoint
        hf_opts o; o.arch = "Qwen2ForCausalLM"; o.biases = true;
        auto dir = make_hf_dir("hf_qwen2", o);
        auto rep = import::import_hf(dir, out);
        CHECK(rep.template_name == "qwen2" && rep.tensors == 15 && rep.unused.empty());
        { modelfile::reader r(out); CHECK(r.tensors().count("layer.0.bq") && r.tensors().count("layer.0.bk") && r.tensors().count("layer.0.bv"));
          CHECK(r.entry("layer.0.bq").shape == shape_t({8})); }
        import::import_options as_llama; as_llama.template_name = "llama";
        CHECK(has(error_of([&] { import::import_hf(dir, out, as_llama); }), "self_attn.q_proj.bias"));
        o.biases = false;   // and a Qwen2 checkpoint without its biases does not import as if nothing were missing
        CHECK(has(error_of([&] { import::import_hf(make_hf_dir("hf_qwen2_nobias", o), out); }), "self_attn.q_proj.bias"));
    }
    {   // a template that reads tensors the importer does not produce (qwen3's QK-norm on a Llama checkpoint)
        import::import_options o; o.template_name = "qwen3";
        CHECK(has(error_of([&] { import::import_hf(make_hf_dir("hf_wrong_tpl"), out, o); }), "q_norm"));
    }
}

TEST(import_hf_rope_scaling_config) {
    auto out = tmp("ropecfg.fpk");
    auto bp_of = [&](const std::string& extra) {
        hf_opts o; o.config_extra = extra;
        import::import_hf(make_hf_dir("hf_rope", o), out);
        modelfile::reader r(out); return std::string(r.section_text("blueprint"));
    };
    auto plain = bp_of("");
    CHECK(has(plain, "opset 1") && !has(plain, "scaling") && has(plain, "neox: 1)") && has(plain, "groups: n_heads / n_kv)"));
    CHECK(bp_of(R"(, "rope_scaling": null)") == plain);
    CHECK(bp_of(R"(, "rope_scaling": {"rope_type": "default"})") == plain);
    auto lin = bp_of(R"(, "rope_scaling": {"type": "linear", "factor": 2.0})");
    CHECK(has(lin, "opset 2") && has(lin, "neox: 1, scaling: 1, factor: 2)"));
    auto l3 = bp_of(R"(, "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 32})");
    CHECK(has(l3, "scaling: 4, factor: 8, orig_ctx: 32, low_freq: 1, high_freq: 4)"));
    auto dyn = bp_of(R"(, "rope_scaling": {"rope_type": "dynamic", "factor": 2.0})");
    CHECK(has(dyn, "scaling: 2, factor: 2, orig_ctx: 64)"));
    // YaRN: a factor that is not given is max_position_embeddings / original_max_position_embeddings (64 / 16); the
    // attention factor 0.1 ln(4) + 1 scales q and k, i.e. attention's scale by its square
    CHECK(has(bp_of(R"(, "rope_scaling": {"rope_type": "yarn", "factor": 9.0, "original_max_position_embeddings": 16})"), "scaling: 3, factor: 9, orig_ctx: 16,"));
    CHECK(has(bp_of(R"(, "rope_scaling": {"rope_type": "yarn", "factor": 2.0})"), "scaling: 3, factor: 2, orig_ctx: 64,"));
    CHECK(has(bp_of(R"(, "original_max_position_embeddings": 32, "rope_scaling": {"rope_type": "yarn", "factor": 2.0})"), "scaling: 3, factor: 2, orig_ctx: 32,"));
    auto yarn = bp_of(R"(, "rope_scaling": {"rope_type": "yarn", "original_max_position_embeddings": 16})");
    CHECK(has(yarn, "scaling: 3, factor: 4, orig_ctx: 16, beta_fast: 32, beta_slow: 1, truncate: 1)") && has(yarn, "groups: n_heads / n_kv, scale: "));
    { auto bp = modelfile::parse_blueprint(yarn); CHECK(bp.opset == 2);
      double af = 0.1 * std::log(4.0) + 1.0, want = af * af / std::sqrt(4.0), got = std::stod(yarn.substr(yarn.find(", scale: ") + 9));
      CHECK(std::fabs(got - want) < 1e-12);
      inference::session s(out); CHECK(s.forward(std::vector<int>{1, 3, 5}).size() == 16); }
    // transformers 5 keeps the type and theta in rope_parameters
    auto rp = bp_of(R"(, "rope_parameters": {"rope_type": "linear", "factor": 3.0, "rope_theta": 500.0})");
    CHECK(has(rp, "scaling: 1, factor: 3)") && has(rp, "base: 500"));
    // refused, with the reason
    for (auto [cfg, want] : {std::pair<const char*, const char*>{R"(, "rope_scaling": {"rope_type": "longrope", "short_factor": [1.0], "long_factor": [2.0]})", "longrope"},
                             {R"(, "rope_scaling": {"rope_type": "spiral", "factor": 2.0})", "spiral"},
                             {R"(, "rope_scaling": {"rope_type": "linear"})", "factor"},
                             {R"(, "rope_scaling": {"rope_type": "linear", "factor": -1.0})", "factor"}}) {
        hf_opts o; o.config_extra = cfg;
        auto msg = error_of([&] { import::import_hf(make_hf_dir("hf_rope_bad", o), out); });
        CHECK_MSG(has(msg, want), "config %s -> message: %s", cfg, msg.c_str());
    }
}

TEST(blueprint_opset_range) {
    auto src = [](const std::string& head, const std::string& fn = "relu") {
        return "blueprint \"t\" " + head + "\ninput x : f32[2]\ngraph {\n y = " + fn + "(x)\n}\noutput y : f32[2]";
    };
    CHECK(modelfile::opset_min == 1 && modelfile::opset_max >= 2);
    for (int v = modelfile::opset_min; v <= modelfile::opset_max; ++v) (void)modelfile::build_graph(modelfile::parse_blueprint(src("opset " + std::to_string(v))), {});
    (void)modelfile::build_graph(modelfile::parse_blueprint(src("")), {});   // no opset clause = opset 1
    // a newer opset parses (inspect can still show it) but does not build; the message names it and the range
    auto newer = modelfile::parse_blueprint(src("opset 99")); CHECK(newer.opset == 99);
    auto msg = error_of([&] { (void)modelfile::build_graph(newer, {}); });
    CHECK_MSG(has(msg, "opset 99") && has(msg, std::format("{}..{}", modelfile::opset_min, modelfile::opset_max)) && has(msg, "newer FORGE"), "message: %s", msg.c_str());
    try { modelfile::check_opset(newer); CHECK(false); } catch (const forge_exception& e) { CHECK(e.code == errc::unsupported); }
    CHECK(has(error_of([&] { (void)modelfile::build_graph(modelfile::parse_blueprint(src("opset 0")), {}); }), "opset 0"));
    CHECK_THROWS(modelfile::parse_blueprint(src("opset 1.5")));
    // an op this engine does not have: the message names the op and the blueprint's opset
    msg = error_of([&] { (void)modelfile::build_graph(modelfile::parse_blueprint(src("opset 2", "warp_drive")), {}); });
    CHECK_MSG(has(msg, "blueprint:4") && has(msg, "'warp_drive'") && has(msg, "opset 2"), "message: %s", msg.c_str());
    // a model file whose blueprint is too new does not load: the session names the file and the opset
    auto ok = forge_test::make_tiny_model(tmp("opset_ok.fpk")), bad = tmp("opset_bad.fpk");
    {
        modelfile::reader r(ok); modelfile::writer w;
        for (auto& [n, e] : r.tensors()) { auto t = r.load(n); w.add_tensor(n, t.shape, t.data, "f32"); }
        w.set_manifest(r.manifest());
        std::string bp(r.section_text("blueprint")); auto at = bp.find("opset 1"); CHECK(at != std::string::npos);
        bp.replace(at, 7, "opset 7");
        w.set_section("blueprint", bp); w.set_section("tokenizer", std::string(r.section_text("tokenizer")));
        w.write(bad);
    }
    { inference::session s(ok); CHECK(!s.forward(std::vector<int>{1, 2}).empty()); }
    msg = error_of([&] { inference::session s(bad); });
    CHECK_MSG(has(msg, "opset_bad.fpk") && has(msg, "opset 7"), "message: %s", msg.c_str());
}

TEST(licence_is_captured_and_kept) {
    auto out = tmp("lic.fpk"), sgn = tmp("lic_signed.fpk"), enc = tmp("lic_enc.fpk");
    hf_opts o;
    o.readme = "---\nlanguage:\n- en\nlicense: apache-2.0\nlicense_name: \"Apache License 2.0\"\nlicense_link: https://www.apache.org/licenses/LICENSE-2.0\ntags:\n- test\n---\n# A model\nlicense: not-this-one\n";
    o.license_text = "Apache License\nVersion 2.0, January 2004\n(test text)\n";
    auto rep = import::import_hf(make_hf_dir("hf_lic", o), out);
    CHECK(rep.license == "apache-2.0");
    auto check = [&](const std::string& path, const modelfile::model_key* key = nullptr) {
        modelfile::reader r(path, key ? std::optional<modelfile::model_key>(*key) : std::nullopt);
        const auto& m = r.manifest();
        CHECK(m.str_or("license", "") == "apache-2.0" && m.str_or("license_name", "") == "Apache License 2.0");
        CHECK(m.str_or("license_link", "") == "https://www.apache.org/licenses/LICENSE-2.0");
        CHECK(m.find("license_files") && m["license_files"].arr().size() == 1 && m["license_files"][0].str() == "LICENSE");
        CHECK(r.has_section("license") && r.section_text("license") == o.license_text);
        CHECK(r.verify().empty());
    };
    check(out);
    // signing and encrypting copy the licence with everything else, and the signature covers it
    auto kp = modelfile::generate_keypair(); modelfile::sign_file(out, sgn, kp);
    check(sgn);
    { modelfile::reader r(sgn); auto v = modelfile::verify_signature(r); CHECK(v.digest_ok && !v.valid_keys.empty()); }
    auto key = modelfile::generate_model_key(); modelfile::encrypt_file(sgn, enc, key);
    check(enc, &key);
    { modelfile::reader locked(enc); CHECK(locked.locked() && locked.section_text("license") == o.license_text); }   // readable without the key
    modelfile::sign_file(enc, enc, kp);
    check(enc, &key);
    // a list-valued licence, two licence files, no front matter, nothing at all
    {
        hf_opts l; l.readme = "---\nlicense:\n  - mit\n  - cc-by-4.0\n---\n";
        auto dir = make_hf_dir("hf_lic2", l);
        { std::ofstream f(fs::path(dir) / "LICENSE.txt"); f << "MIT text\n"; }
        { std::ofstream f(fs::path(dir) / "NOTICE"); f << "notice text\n"; }
        CHECK(import::import_hf(dir, out).license == "mit, cc-by-4.0");
        modelfile::reader r(out);
        CHECK(r.manifest()["license_files"].arr().size() == 2);
        CHECK(r.section_text("license") == "===== LICENSE.txt =====\nMIT text\n===== NOTICE =====\nnotice text\n");
    }
    {
        hf_opts l; l.readme = "# no front matter\nlicense: mit\n";
        CHECK(import::import_hf(make_hf_dir("hf_lic3", l), out).license.empty());
        modelfile::reader r(out);
        CHECK(!r.manifest().find("license") && !r.has_section("license"));
        inference::session s(out); CHECK(s.forward(std::vector<int>{1, 3}).size() == 16);   // optional: absent is fine
    }
}

// ---- GGUF tensor types: sizes are checked before anything is read ----
// A one-tensor GGUF v3 file: `type` / `dims` (ggml order) in the tensor table, then `data` at the first 32-byte boundary.
static std::string tiny_gguf(const std::string& name, std::uint32_t type, const std::vector<std::uint64_t>& dims, const std::vector<std::uint8_t>& data,
                             std::uint64_t offset = 0) {
    std::string b = "GGUF";
    auto put = [&](auto v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); };
    put(std::uint32_t(3)); put(std::uint64_t(1)); put(std::uint64_t(0));
    put(std::uint64_t(1)); b += "t";
    put(std::uint32_t(dims.size())); for (auto d : dims) put(d);
    put(type); put(offset);
    while (b.size() % 32) b.push_back('\0');
    b.append(reinterpret_cast<const char*>(data.data()), data.size());
    auto path = tmp(name);
    std::ofstream(path, std::ios::binary).write(b.data(), std::streamsize(b.size()));
    return path;
}
static errc gguf_error(const std::string& path, std::string* msg = nullptr) {
    try { import::gguf_file g(path); (void)g.stored(g.tensors().at(0)); }
    catch (const forge_exception& e) { if (msg) *msg = e.what(); return e.code; }
    return errc(-1);
}

TEST(gguf_tensor_types_sizes_and_transcoding) {
    using import::gguf_file;
    const std::uint64_t big = std::uint64_t(1) << 40;
    std::vector<std::uint8_t> blk(18, 0);
    blk[0] = 0x00; blk[1] = 0x38;                       // f16 0.5
    for (int i = 0; i < 16; ++i) blk[std::size_t(2 + i)] = std::uint8_t(i | ((15 - i) << 4));
    {   // Q4_0, one block: verbatim under q4_0, x = d * (q - 8)
        gguf_file g(tiny_gguf("q4_0.gguf", 2, {32}, blk));
        auto s = g.stored(g.tensors()[0]);
        CHECK(std::string(s.codec) == "q4_0" && !s.transcoded && s.bytes.size() == 18 && std::string(s.type) == "Q4_0");
        float x[32]; graph::codecs::find("q4_0")->decode(s.bytes.data(), 32, x);
        for (int i = 0; i < 16; ++i) CHECK(x[i] == 0.5f * float(i - 8) && x[i + 16] == 0.5f * float(7 - i));
    }
    std::string msg;
    CHECK(gguf_error(tiny_gguf("q4_0_33.gguf", 2, {33}, blk), &msg) == errc::format_error && msg.find("whole number") != std::string::npos);   // not whole blocks
    CHECK(gguf_error(tiny_gguf("q4_0_trunc.gguf", 2, {64}, blk)) == errc::format_error);                      // two blocks declared, one present
    CHECK(gguf_error(tiny_gguf("q4_0_off.gguf", 2, {32}, blk, 1)) == errc::format_error);                     // offset pushes it past the end
    CHECK(gguf_error(tiny_gguf("q4_0_off2.gguf", 2, {32}, blk, ~std::uint64_t(0))) == errc::format_error);    // offset near 2^64
    CHECK(gguf_error(tiny_gguf("ovf.gguf", 2, {big, big, big}, blk)) == errc::format_error);                  // element count overflows 64 bits
    CHECK(gguf_error(tiny_gguf("ovf2.gguf", 12, {std::uint64_t(1) << 63, 2}, blk)) == errc::format_error);
    CHECK(gguf_error(tiny_gguf("ovf3.gguf", 0, {big, big >> 8, 1 << 20}, blk)) == errc::format_error);
    CHECK(gguf_error(tiny_gguf("neg.gguf", 28, {~std::uint64_t(0)}, blk)) == errc::format_error);
    { gguf_file g(tiny_gguf("empty.gguf", 11, {0, big}, {})); CHECK(g.stored(g.tensors()[0]).bytes.empty()); }   // an empty tensor is fine
    // types that are recognised but not loadable say why; unknown ids list what is supported
    CHECK(gguf_error(tiny_gguf("iq2.gguf", 16, {256}, std::vector<std::uint8_t>(66)), &msg) == errc::unsupported && msg.find("IQ2_XXS") != std::string::npos && msg.find("codebook") != std::string::npos);
    CHECK(gguf_error(tiny_gguf("q8_1.gguf", 9, {32}, std::vector<std::uint8_t>(40)), &msg) == errc::unsupported && msg.find("Q8_1") != std::string::npos);
    CHECK(gguf_error(tiny_gguf("t99.gguf", 99, {32}, blk), &msg) == errc::unsupported && msg.find("IQ4_XS") != std::string::npos);
    for (std::uint32_t t : {16u, 17u, 18u, 19u, 21u, 22u, 29u, 9u, 39u}) CHECK(!gguf_file::loadable(t) && gguf_file::type_of(t) && gguf_file::type_of(t)->why);
    // every native type's block size agrees with its codec
    for (std::uint32_t t = 0; t < 64; ++t)
        if (auto* ty = gguf_file::type_of(t); ty && ty->codec) {
            auto* c = graph::codecs::find(ty->codec);
            CHECK_MSG(c && c->block == ty->block && c->block_bytes == ty->block_bytes, "%s", ty->name);
        }
    {   // plain numeric types become f32
        std::int16_t v16[3] = {-32768, 7, 1234}; std::vector<std::uint8_t> d(6); std::memcpy(d.data(), v16, 6);
        gguf_file g(tiny_gguf("i16.gguf", 25, {3}, d)); auto s = g.stored(g.tensors()[0]);
        float x[3]; std::memcpy(x, s.bytes.data(), 12);
        CHECK(s.transcoded && std::string(s.codec) == "f32" && s.bytes.size() == 12 && x[0] == -32768.f && x[1] == 7.f && x[2] == 1234.f);
        double v64[2] = {0.1, -3e10}; d.resize(16); std::memcpy(d.data(), v64, 16);
        gguf_file g2(tiny_gguf("f64.gguf", 28, {2}, d)); auto s2 = g2.stored(g2.tensors()[0]);
        std::memcpy(x, s2.bytes.data(), 8);
        CHECK(x[0] == 0.1f && x[1] == -3e10f);
        CHECK(gguf_error(tiny_gguf("i64_trunc.gguf", 27, {3}, d)) == errc::format_error);
    }
    {   // Q8_K: { f32 d; i8 qs[256]; i16 bsums[16]; }, x = d * q
        std::vector<std::uint8_t> d(292, 0); float dd = 0.25f; std::memcpy(d.data(), &dd, 4);
        for (int i = 0; i < 256; ++i) d[std::size_t(4 + i)] = std::uint8_t(std::int8_t(i - 128));
        gguf_file g(tiny_gguf("q8_k.gguf", 15, {256}, d)); auto s = g.stored(g.tensors()[0]);
        CHECK(s.transcoded && s.bytes.size() == 1024);
        float x[256]; std::memcpy(x, s.bytes.data(), 1024);
        for (int i = 0; i < 256; ++i) CHECK(x[i] == 0.25f * float(i - 128));
    }
}

// AWQ / GPTQ unpacking (hf_quant.hpp) on hand-built words: one code at a known place.
TEST(hf_quant_unpacking_positions) {
    auto dir = std::filesystem::path(tmp("hfq")); std::filesystem::create_directories(dir);
    auto write = [&](const std::string& file, const std::string& header, const std::vector<std::uint8_t>& data) {
        std::string b; std::uint64_t n = header.size(); b.append(reinterpret_cast<const char*>(&n), 8); b += header;
        b.append(reinterpret_cast<const char*>(data.data()), data.size());
        std::ofstream(dir / file, std::ios::binary).write(b.data(), std::streamsize(b.size()));
        return (dir / file).string();
    };
    auto bytes = [](auto... parts) { std::vector<std::uint8_t> d; (d.insert(d.end(), reinterpret_cast<const std::uint8_t*>(parts.data()), reinterpret_cast<const std::uint8_t*>(parts.data() + parts.size())), ...); return d; };
    {   // AWQ gemm, in = 32, out = 8, one group: code 5 for (input 3, output 2) sits at nibble 1 of word [3][0]
        std::vector<std::int32_t> qw(32, 0), qz(1, 0); std::vector<float> sc(8, 0.5f);
        qw[3] = 5 << 4;            // nibble 1 -> output row 2
        qz[0] = 3 << 28;           // nibble 7 -> output row 7 has zero 3
        auto p = write("awq.safetensors", R"({"l.qweight":{"dtype":"I32","shape":[32,1],"data_offsets":[0,128]},"l.qzeros":{"dtype":"I32","shape":[1,1],"data_offsets":[128,132]},"l.scales":{"dtype":"F32","shape":[1,8],"data_offsets":[132,164]}})",
                       bytes(qw, qz, sc));
        import::quantised_checkpoint::shard_list sh; sh.push_back(std::make_unique<import::safetensors>(p));
        import::quantised_checkpoint q(import::quant_scheme_of(json::parse(R"({"quantization_config":{"quant_method":"awq","bits":4,"group_size":32,"version":"gemm","zero_point":true}})")), sh);
        CHECK(q.has("l.weight") && !q.has("m.weight"));
        auto w = q.load("l.weight");
        CHECK(w.shape == shape_t({8, 32}));
        for (i64 o = 0; o < 8; ++o) for (i64 i = 0; i < 32; ++i) {
            float want = (float(o == 2 && i == 3 ? 5 : 0) - float(o == 7 ? 3 : 0)) * 0.5f;
            CHECK_MSG(w.data[std::size_t(o * 32 + i)] == want, "awq [%d][%d] = %g, want %g", int(o), int(i), double(w.data[std::size_t(o * 32 + i)]), double(want));
        }
        auto nat = q.take();
        CHECK(nat && nat->codec == "q4_g32" && nat->blocks.size() == 8 * 20);
        if (nat) { std::vector<float> x(256); graph::codecs::find("q4_g32")->decode(nat->blocks.data(), 256, x.data()); CHECK(x == w.data); }
        CHECK(!q.take());
    }
    {   // GPTQ, in = 32, out = 8, 4 bits: code 9 for (input 10, output 5) is bits 8..11 of word [1][5]; zeros store z - 1
        std::vector<std::int32_t> qw(4 * 8, 0), qz(1, 0x77777777), gi(32, 0); std::vector<float> sc(8, 0.25f);
        qw[1 * 8 + 5] = 9 << 8;
        qz[0] = 0x77777777 ^ (0x7 << 12) ^ (0x1 << 12);   // output row 3 stores 1 -> zero 2; the others store 7 -> zero 8
        auto p = write("gptq.safetensors", R"({"l.qweight":{"dtype":"I32","shape":[4,8],"data_offsets":[0,128]},"l.qzeros":{"dtype":"I32","shape":[1,1],"data_offsets":[128,132]},"l.g_idx":{"dtype":"I32","shape":[32],"data_offsets":[132,260]},"l.scales":{"dtype":"F32","shape":[1,8],"data_offsets":[260,292]}})",
                       bytes(qw, qz, gi, sc));
        import::quantised_checkpoint::shard_list sh; sh.push_back(std::make_unique<import::safetensors>(p));
        auto cfg = [](const char* extra) { return json::parse(std::string(R"({"quantization_config":{"quant_method":"gptq","bits":4,"group_size":32,"sym":true)") + extra + "}}"); };
        import::quantised_checkpoint q(import::quant_scheme_of(cfg("")), sh);
        auto w = q.load("l.weight");
        for (i64 o = 0; o < 8; ++o) for (i64 i = 0; i < 32; ++i) {
            float want = (float(o == 5 && i == 10 ? 9 : 0) - float(o == 3 ? 2 : 8)) * 0.25f;
            CHECK_MSG(w.data[std::size_t(o * 32 + i)] == want, "gptq [%d][%d] = %g, want %g", int(o), int(i), double(w.data[std::size_t(o * 32 + i)]), double(want));
        }
        import::quantised_checkpoint q2(import::quant_scheme_of(cfg(R"(,"checkpoint_format":"gptq_v2")")), sh);   // v2 stores the zero itself
        CHECK(q2.load("l.weight").data[0] == -7 * 0.25f);
        CHECK_THROWS(import::quant_scheme_of(cfg(R"(,"checkpoint_format":"other")")));
        CHECK_THROWS(import::quant_scheme_of(json::parse(R"({"quantization_config":{"quant_method":"gptq","bits":3}})")));
        CHECK_THROWS(import::quant_scheme_of(json::parse(R"({"quantization_config":{"quant_method":"awq","version":"gemv"}})")));
        CHECK(!import::quant_scheme_of(json::parse(R"({"quantization_config":{"quant_method":"bitsandbytes"}})")));
        CHECK(!import::quant_scheme_of(json::parse(R"({"hidden_size":8})")));
    }
    std::filesystem::remove_all(dir);
}

// Codecs are names in the tensor index. A reader that does not know a name (a file written by a newer FORGE) still
// opens and lists the file, and fails with `unsupported` when that tensor is loaded.
TEST(fpk_unknown_codec_fails_cleanly) {
    auto path = tmp("codec.fpk");
    { modelfile::writer w; w.set_manifest(json::value(json::object{{"name", "c"}})); std::vector<float> a(64, 0.5f); w.add_tensor("w", {2, 32}, a, "q4_0"); w.write(path); }
    { modelfile::reader r(path); CHECK(r.entry("w").codec == "q4_0"); CHECK(r.load("w").data[0] == 0.5f); }
    std::string b; { std::ifstream f(path, std::ios::binary); b.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()); }
    auto p = b.find("\"q4_0\""); CHECK(p != std::string::npos);
    if (p == std::string::npos) return;
    b.replace(p, 6, "\"zz_9\"");
    std::ofstream(path, std::ios::binary).write(b.data(), std::streamsize(b.size()));
    modelfile::reader r(path);
    CHECK(r.entry("w").codec == "zz_9");
    bool clean = false;
    try { (void)r.load("w"); } catch (const forge_exception& e) { clean = e.code == errc::unsupported && std::string(e.what()).find("zz_9") != std::string::npos; }
    CHECK(clean);
    std::filesystem::remove(path);
}

// A blueprint is untrusted input. This one makes the row shape of a state slot depend on `seq`, so the plan for
// each sequence-length bucket would lay the same slot out differently: [rows, 1, 4] for seq 1, [rows, 1, 16] for
// seq 4. The plans of a session share one state store; the second one must be refused, not left to append
// 16-float rows to a buffer sized for 4-float rows (tests/test_graph.cpp has the executor-level cases). The
// blueprint is also a fuzz seed; ctest runs with the source tree as the working directory.
TEST(blueprint_state_rows_depending_on_seq_are_refused) {
    std::ifstream f("fuzz/corpus/blueprint/regress_state_rows_seq.fbp");
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (src.empty()) { std::printf("  (fuzz/corpus/blueprint/regress_state_rows_seq.fbp not found from this directory: skipped)\n"); return; }
    auto bp = modelfile::parse_blueprint(src);
    auto graph_for = [&](i64 seq) { modelfile::build_options bo; bo.dims["seq"] = seq; return modelfile::build_graph(bp, bo); };
    for (i64 page : {i64(0), i64(2)}) {
        graph::state_store st; st.capacity = 8; st.page_rows = page;
        auto p1 = graph::compile(graph_for(1), &st);
        CHECK(st.slots.count("kv.0.k") == 1);
        CHECK_THROWS(graph::compile(graph_for(4), &st));
        graph::executor e(*p1, &st);
        std::vector<float> x(8, 0.5f);
        for (int t = 0; t < 3; ++t) { e.set_input("x", x); e.run(); }
        CHECK(st.slots["kv.0.k"].len == 6);
        auto again = graph::compile(graph_for(1), &st);   // the same bucket again shares the slot
        CHECK(again != nullptr);
    }
}

// A blueprint must declare the opset of the newest thing it uses: an op, an attr, a language feature or a session kind
// below its opset is refused with the feature and the opset it needs.
TEST(blueprint_opset_enforced_per_feature) {
    // (opset, declarations before the graph, graph body, what follows) -> ok, or the two things the message must name
    struct probe { const char* what; int needs; std::string pre, body, post; const char* names; };
    const std::string out_y = "output y : f32[2]";
    std::vector<probe> probes = {
        {"spec op of opset 3 (mish)", 3, "", "y = mish(x)", out_y, "op 'mish'"},
        {"spec op of opset 3 (l2_normalize)", 3, "", "y = l2_normalize(x)", out_y, "op 'l2_normalize'"},
        {"scalar()", 3, "", "y = x + scalar(1)", out_y, "scalar()"},
        {"number helper min", 3, "", "y = x * min(2, 3)", out_y, "number helper min()"},
        {"number helper mod", 3, "", "y = x * mod(7, 4)", out_y, "number helper mod()"},
        {"number helper floor", 3, "", "y = x * floor(2.5)", out_y, "number helper floor()"},
        {"composite default", 3, "op f(a, k = 2) { return a * k }\n", "y = f(x)", out_y, "default value for parameter 'k' of the composite op 'f'"},
        {"composite keyword argument", 3, "op f(a, k) { return a * k }\n", "y = f(x, k: 2)", out_y, "keyword argument to the composite op 'f'"},
        {"session embedding", 3, "", "y = relu(x)", out_y + "\nsession embedding", "`session embedding`"},
        {"rope scaling attr", 2, "input p : i32[2]\n", "q = reshape(x, shape: [2, 1, 1])\n y = reshape(rope(reshape(concat(q, q, axis: 2), shape: [2, 1, 2]), p, base: 10000, neox: 1, scaling: 1, factor: 2), shape: [4])",
         "output y : f32[4]", "RoPE frequency scaling (rope attr '"},
    };
    for (auto& p : probes) {
        auto src = [&](int opset) { return std::format("blueprint \"t\" opset {}\ninput x : f32[2]\n{}graph {{\n {}\n}}\n{}", opset, p.pre, p.body, p.post); };
        auto build = [&](int opset) { (void)modelfile::build_graph(modelfile::parse_blueprint(src(opset)), {}); };
        for (int v = p.needs; v <= modelfile::opset_max; ++v) {
            auto msg = error_of([&] { build(v); });
            CHECK_MSG(msg.empty(), "%s at opset %d: %s", p.what, v, msg.c_str());
        }
        for (int v = modelfile::opset_min; v < p.needs; ++v) {
            auto msg = error_of([&] { build(v); });
            CHECK_MSG(has(msg, p.names) && has(msg, std::format("needs opset {}", p.needs)) && has(msg, std::format("declares opset {}", v)) && has(msg, "blueprint:"),
                      "%s at opset %d: message '%s'", p.what, v, msg.c_str());
            try { build(v); CHECK(false); } catch (const forge_exception& e) { CHECK(e.code == errc::parse_error); }
        }
    }
    // what opset 1 always had still builds at opset 1: composites with positional arguments, keyword attrs of an op,
    // sqrt of a number as the spec op it was before the helpers, plain rope
    auto ok = error_of([&] {
        (void)modelfile::build_graph(modelfile::parse_blueprint(
            "blueprint \"t\"\ninput x : f32[2]\nop f(a, k) { return a * k }\ngraph {\n y = f(softmax(x, axis: 0), 2) + sqrt(4)\n}\noutput y : f32[2]\nsession one_shot"), {});
    });
    CHECK_MSG(ok.empty(), "%s", ok.c_str());
    // the tables: every op has an opset in range; attrs that came later than their op are listed
    for (auto name : {"relu", "matmul_t", "rope", "attention"}) CHECK(modelfile::op_min_opset(name) == 1);
    for (auto name : {"conv2d", "attention_bias", "masked_mean", "pad"}) CHECK(modelfile::op_min_opset(name) == 3 && ops::op_from_name(name));
    CHECK(modelfile::attr_min_opset("rope", "freq_factors") == 2 && modelfile::attr_min_opset("rope", "base") == 1 && modelfile::attr_min_opset("attention", "scale") == 1);
}

// Every shipped template, instantiated the way the importers do, declares an opset that its body builds under: the
// text is put through the parser and builder with zero weights. (A template that used an opset-3 op under
// "opset 1" would fail here, not at a user's import.)
TEST(shipped_templates_declare_their_opset) {
    namespace fs = std::filesystem;
    std::size_t seen = 0;
    for (auto& f : fs::directory_iterator(fs::path(FORGE_SOURCE_DIR) / "import" / "templates")) {
        if (f.path().extension() != ".fbp") continue;
        ++seen;
        auto text = import::read_file(f.path());
        auto head = text.substr(text.find("\nblueprint ") + 1); head = head.substr(0, head.find('\n'));
        // the declared opset: a number, or the {{opset|1}} placeholder the importer fills with 1 or 2
        bool placeholder = has(head, "{{opset|1}}");
        int declared = placeholder ? 1 : std::atoi(head.substr(head.find("opset ") + 6).c_str());
        CHECK_MSG(has(head, "opset ") && declared >= modelfile::opset_min && declared <= modelfile::opset_max, "%s: '%s'", f.path().filename().string().c_str(), head.c_str());
        // features that need a later opset, looked for in the template text
        int needs = 1;
        if (has(text, "\nsession embedding") || has(text, "scalar(")) needs = 3;
        for (std::size_t p = 0; (p = text.find("\nop ", p)) != std::string::npos; ++p) {   // a composite with a default value
            auto sig = text.substr(p, text.find('{', p) - p);
            if (has(sig, "=")) needs = 3;
        }
        for (std::size_t p = 0; (p = text.find('(', p)) != std::string::npos; ++p) {
            std::size_t b = p; while (b > 0 && (std::isalnum(static_cast<unsigned char>(text[b - 1])) || text[b - 1] == '_')) --b;
            auto name = text.substr(b, p - b);
            if (!name.empty() && ops::op_from_name(name)) needs = std::max(needs, modelfile::op_min_opset(name));
        }
        CHECK_MSG(declared >= needs, "%s declares opset %d but uses opset-%d features", f.path().filename().string().c_str(), declared, needs);
        // templates that can carry RoPE scaling take the opset from the importer
        if (has(text, "{{rope_args")) CHECK_MSG(placeholder, "%s takes rope_args but has a fixed opset", f.path().filename().string().c_str());
    }
    CHECK(seen >= 16);
}
