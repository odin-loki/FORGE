#include "check.hpp"
#include <forge/inference/inference.hpp>
#include <forge/import/gguf.hpp>
#include <cstdio>
#include <filesystem>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#include <fstream>
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

TEST(fpk_encryption_hook) {
    auto src = forge_test::make_tiny_model(tmp("enc_src.fpk")), dst = tmp("enc.fpk");
    auto key = modelfile::generate_model_key(); modelfile::encrypt_file(src, dst, key);
    {   // readers map the file; Windows cannot replace a mapped file, so they close before dst is rewritten
    modelfile::reader plain(src), locked(dst);
    CHECK(locked.locked()); CHECK(locked.verify().empty()); CHECK_THROWS(locked.load("lm_head"));
    CHECK(locked.manifest().str_or("format", "") == "fpk");                  // metadata stays readable
    modelfile::reader open(dst, key); CHECK(open.encrypted()); CHECK(open.verify().empty());
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
    auto src = R"(blueprint "c"
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
        std::string s = std::string("blueprint \"e\"\n") + ops + "\ninput x : f32[2]\ngraph { y = " + call + " }\noutput y : f32[2]";
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
            if (key == "act") v = "gelu";
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
    CHECK(n >= 15);
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
