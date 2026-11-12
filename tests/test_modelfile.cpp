#include "check.hpp"
#include <forge/inference/inference.hpp>
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
