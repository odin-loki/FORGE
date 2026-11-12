// Tokenizer pipeline against goldens made with the reference implementations (tests/data/tokenizer/make_fixtures.py):
// Hugging Face `tokenizers` for tokenizer.json files, `sentencepiece` for .model files, and both again for the GGUF
// conversions. No Python is needed to run this test; the goldens are checked in.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/import/gguf.hpp>
#include <forge/import/tokenizer_import.hpp>
#include <forge/inference/session.hpp>
#include <forge/modelfile/tokenizer.hpp>
#include <filesystem>
#include <random>
#include <set>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace forge;
using modelfile::tokenizer;
namespace fs = std::filesystem;

static fs::path data_dir() { return fs::path(FORGE_SOURCE_DIR) / "tests" / "data" / "tokenizer"; }
static std::string slurp(const fs::path& p) { return import::read_file(p); }
static std::string unhex(const std::string& h) {
    auto d = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
    std::string o; for (std::size_t i = 0; i + 1 < h.size(); i += 2) o += char(d(h[i]) * 16 + d(h[i + 1]));
    return o;
}
static std::vector<int> ints(const json::value& v) { std::vector<int> o; for (auto& x : v.arr()) o.push_back(int(x.num())); return o; }
static std::string show(const std::vector<int>& v) { std::string s; for (int x : v) s += std::to_string(x) + " "; return s; }
static const json::value& goldens() { static const json::value g = json::parse(slurp(data_dir() / "goldens.json")); return g; }
static std::vector<std::string> corpus() { std::vector<std::string> c; for (auto& h : goldens()["corpus"].arr()) c.push_back(unhex(h.str())); return c; }
static fs::path tmp(const char* name) { return fs::temp_directory_path() / (std::string("forge_tok_") + std::to_string(::getpid()) + "_" + name); }

// the tokenizer.json of a fixture: its file, with the "patch" keys put in
static std::string fixture_json(const json::value& fx) {
    if (!fx.find("patch")) return slurp(data_dir() / fx["file"].str());   // as it is: json::dump would round the Unigram scores
    auto j = json::parse(slurp(data_dir() / fx["file"].str()));
    if (auto* patch = fx.find("patch"))
        for (auto& [k, v] : patch->obj()) {
            json::value* slot = nullptr;
            for (auto& [jk, jv] : j.obj_mut()) if (jk == k) slot = &jv;
            if (!slot) { j.obj_mut().emplace_back(k, v); continue; }
            if (k != "model") { *slot = v; continue; }
            for (auto& [mk, mv] : v.obj()) {
                bool found = false;
                for (auto& [sk, sv] : slot->obj_mut()) if (sk == mk) { sv = mv; found = true; }
                if (!found) slot->obj_mut().emplace_back(mk, mv);
            }
        }
    return json::dump(j);
}
static bool line_wanted(const json::value& fx, std::size_t i) {
    auto* l = fx.find("lines");
    if (!l) return true;
    for (auto& x : l->arr()) if (std::size_t(x.num()) == i) return true;
    return false;
}

// every tokenizer.json fixture: normalizer, pre-tokenizer, ids with and without the post-processor, both decodes
TEST(hf_tokenizers_goldens) {
    auto text = corpus(); int fixtures = 0, compared = 0;
    for (auto& [name, fx] : goldens()["fixtures"].obj()) {
        if (fx["oracle"].str() != "tokenizers") continue;
        ++fixtures;
        tokenizer t(fixture_json(fx));
        auto& cases = fx["cases"].arr();
        CHECK(cases.size() == text.size());
        for (std::size_t i = 0; i < cases.size(); ++i) {
            auto& c = cases[i];
            auto valid = os::text::sanitize_utf8(text[i]);
            if (auto* n = c.find("norm")) CHECK_MSG(t.normalize(valid) == n->str(), "%s line %zu: normalize", name.c_str(), i);
            if (auto* p = c.find("pre")) {
                std::vector<std::string> want; for (auto& x : p->arr()) want.push_back(x.str());
                CHECK_MSG(t.pre_tokenize(valid) == want, "%s line %zu: pre_tokenize", name.c_str(), i);
            }
            auto ids = t.encode(text[i]);
            CHECK_MSG(ids == ints(c["ids"]), "%s line %zu: ids %s", name.c_str(), i, show(ids).c_str());
            if (auto* s = c.find("ids_special")) CHECK_MSG(t.encode(text[i], true) == ints(*s), "%s line %zu: ids with special tokens", name.c_str(), i);
            if (auto* d = c.find("dec")) CHECK_MSG(t.decode(ids) == d->str(), "%s line %zu: decode '%s'", name.c_str(), i, t.decode(ids).c_str());
            if (auto* d = c.find("dec_skip")) CHECK_MSG(t.decode(t.encode(text[i], true), true) == d->str(), "%s line %zu: decode, special tokens skipped", name.c_str(), i);
            ++compared;
        }
    }
    CHECK(fixtures >= 60 && compared >= 60 * 30);
}

static void set_from(tokenizer& t, const json::value& fx) { t.set_special(int(fx.num_or("bos", -1)), int(fx.num_or("eos", -1))); }

// SentencePiece .model files, read by FORGE's own protobuf reader, against SentencePieceProcessor
TEST(sentencepiece_models_goldens) {
    auto text = corpus(); int fixtures = 0;
    for (auto& [name, fx] : goldens()["fixtures"].obj()) {
        if (fx["oracle"].str() != "sentencepiece") continue;
        ++fixtures;
        auto tj = import::sentencepiece_tokenizer_json(slurp(data_dir() / fx["file"].str()));
        CHECK(json::parse(tj)["forge_tokenizer_version"].num() == modelfile::tokenizer_format_version);
        tokenizer t(tj); set_from(t, fx);
        CHECK(t.type() == (name.find("unigram") != std::string::npos ? tokenizer::model_type::unigram : tokenizer::model_type::sentencepiece_bpe));
        CHECK(t.unk() == int(fx["unk"].num()));
        auto& cases = fx["cases"].arr();
        for (std::size_t i = 0; i < cases.size(); ++i) {
            if (!line_wanted(fx, i)) continue;
            auto ids = t.encode(text[i]);
            CHECK_MSG(ids == ints(cases[i]["ids"]), "%s line %zu: ids %s", name.c_str(), i, show(ids).c_str());
            if (cases[i]["dec"].is_str()) CHECK_MSG(t.decode(ids) == cases[i]["dec"].str(), "%s line %zu: decode '%s'", name.c_str(), i, t.decode(ids).c_str());
        }
        // control pieces written out in a prompt are parsed (as llama.cpp and transformers do)
        if (t.bos() >= 0) { auto ids = t.encode(t.id_to_token(t.bos()) + "Hello"); CHECK(!ids.empty() && ids[0] == t.bos() && t.is_special(ids[0])); }
    }
    CHECK(fixtures == 4);
}

// GGUF metadata -> tokenizer.json: each tokenizer.ggml.model, the pre-tokenizer names, token types, BOS / EOS flags
TEST(gguf_tokenizers_goldens) {
    auto text = corpus(); int fixtures = 0;
    auto& all = goldens()["fixtures"];
    for (auto& [name, fx] : all.obj()) {
        auto* same = fx.find("same_as");
        if (!same) continue;
        ++fixtures;
        import::gguf_file g((data_dir() / fx["file"].str()).string());
        std::string warn;
        auto tj = import::gguf_tokenizer_json(g, &warn);
        CHECK_MSG(warn.empty() && !tj.empty(), "%s: %s", name.c_str(), warn.c_str());
        tokenizer t(tj);
        auto num = [&](const char* k) { auto* v = g.get(k); return v ? int(v->num()) : -1; };
        t.set_special(num("tokenizer.ggml.bos_token_id"), num("tokenizer.ggml.eos_token_id"));
        if (auto* ab = g.get("tokenizer.ggml.add_bos_token"); ab && name != "gguf_bert") t.set_add_bos(ab->b());
        auto& ref = all[same->str()]["cases"].arr();
        std::vector<std::string> keys; for (auto& k : fx["check"].arr()) keys.push_back(k.str());
        auto has = [&](const char* k) { return std::find(keys.begin(), keys.end(), k) != keys.end(); };
        for (std::size_t i = 0; i < ref.size(); ++i) {
            if (!line_wanted(fx, i)) continue;
            auto ids = t.encode(text[i]);
            if (has("ids")) CHECK_MSG(ids == ints(ref[i]["ids"]), "%s line %zu: ids %s", name.c_str(), i, show(ids).c_str());
            if (has("ids_special")) CHECK_MSG(t.encode(text[i], true) == ints(ref[i]["ids_special"]), "%s line %zu: ids with special tokens", name.c_str(), i);
            if (has("dec")) CHECK_MSG(t.decode(ids) == ref[i]["dec"].str(), "%s line %zu: decode", name.c_str(), i);
        }
    }
    CHECK(fixtures == 9);
    {   // token types: control -> special, user-defined -> added but not special
        import::gguf_file g((data_dir() / "gguf_gpt2_qwen2.gguf").string());
        tokenizer t(import::gguf_tokenizer_json(g));
        CHECK(t.is_special(t.token_id("<|im_start|>")) && !t.is_special(t.token_id("<tool_call>")));
        CHECK((t.encode("<tool_call>x") [0] == t.token_id("<tool_call>")));
        CHECK(t.decode({t.token_id("<|im_end|>"), t.token_id("<tool_call>")}, true) == "<tool_call>");
    }
    {   // an unknown pre-tokenizer name falls back to GPT-2's, and says so
        struct fake {
            json::value toks = json::array{"a", "b"}, model = std::string("gpt2"), pre = std::string("no-such-pre");
            const json::value* get(std::string_view k) const { return k == "tokenizer.ggml.tokens" ? &toks : k == "tokenizer.ggml.model" ? &model : k == "tokenizer.ggml.pre" ? &pre : nullptr; }
        } g;
        std::string warn;
        CHECK(!import::gguf_tokenizer_json(g, &warn).empty() && warn.find("no-such-pre") != std::string::npos);
        g.model = std::string("rwkv"); warn.clear();
        CHECK(import::gguf_tokenizer_json(g, &warn).empty() && warn.find("rwkv") != std::string::npos);
        for (auto pre : {"llama-bpe", "qwen2", "deepseek-llm", "deepseek-coder", "deepseek-v3", "gpt-2", "default", "tekken", "gpt-4o", "falcon", "starcoder", "viking", "bloom"}) {
            g.model = std::string("gpt2"); g.pre = std::string(pre); warn.clear();
            tokenizer t(import::gguf_tokenizer_json(g, &warn));   // every listed regex compiles
            CHECK_MSG(warn.empty(), "%s", pre);
            (void)t.encode("Hello, w\xC3\xB6rld 12345\n\n \xE4\xBD\xA0\xE5\xA5\xBD!");
        }
    }
}

// A version 1 section (model + added_tokens, nothing else) tokenises as the tokenizer did before the pipeline existed.
// One deliberate difference: an added token that follows a space ("a <s>") is now found; the old splitter ate its "<".
TEST(legacy_section_compatible) {
    auto text = corpus();
    auto& fx = goldens()["fixtures"]["legacy_v1"];
    tokenizer t(slurp(data_dir() / "legacy_v1.json"));
    int same = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        bool special_after_space = text[i].find(" <|") != std::string::npos || text[i].find(" <s>") != std::string::npos || text[i].find("><|") != std::string::npos;
        auto ids = t.encode(text[i]);
        if (special_after_space) { CHECK(t.decode(ids) == text[i]); continue; }
        CHECK_MSG(ids == ints(fx["cases"].arr()[i]["ids"]), "legacy line %zu: ids %s", i, show(ids).c_str());
        CHECK_MSG(t.decode(ids) == unhex(fx["cases"].arr()[i]["dec"].str()), "legacy line %zu: decode", i);
        ++same;
    }
    CHECK(same >= 30);
    // raw bytes survive, as before (no U+FFFD), and BOS handling is the caller's flag
    CHECK(t.decode(t.encode("a\xFF\xFE" "b")) == "a\xFF\xFE" "b");
    t.set_special(0, -1); CHECK(t.add_bos()); CHECK(t.encode("a", true).front() == 0);
    CHECK_THROWS(tokenizer(R"({"forge_tokenizer_version": 99, "model": {"type": "BPE", "vocab": {}, "merges": []}})"));
}

// decode(encode(x)) == x for the tokenizers that lose nothing, and streaming (decode_one per token) gives the same bytes
TEST(roundtrip_properties) {
    auto text = corpus();
    std::mt19937 rng(11);
    const char* alphabet[] = {"a", "b", "Z", " ", " ", "\n", "\t", "1", "23", ".", "'s", "\xC3\xA9", "\xE4\xBD\xA0", "\xF0\x9F\x98\x80", "e\xCC\x81", "\xE2\x96\x81", "<", ">", "|",
                              "\xD7\xA9", "\xD9\x85", "\x01", "The ", "ing", "<|endoftext|>", "<s>", "\xE2\x80\x8D", "  ", "\r\n"};
    for (int i = 0; i < 300; ++i) { std::string s; for (int n = int(rng() % 24); n > 0; --n) s += alphabet[rng() % std::size(alphabet)]; text.push_back(s); }
    auto& all = goldens()["fixtures"];
    for (auto name : {"hf_bpe_gpt2", "hf_bpe_llama3", "hf_bpe_qwen2", "legacy_v1"}) {
        tokenizer t(fixture_json(all[name]));
        for (auto& s : text) {
            auto want = std::string(name) == "legacy_v1" ? s : os::text::sanitize_utf8(s);
            if (std::string(name) == "hf_bpe_qwen2") want = os::text::normalize(want, os::text::norm_form::nfc);
            auto ids = t.encode(s);
            CHECK_MSG(t.decode(ids) == want, "%s: round trip of '%s'", name, s.c_str());
            std::string stream; for (int id : ids) stream += t.decode_one(id);
            CHECK_MSG(stream == want, "%s: streamed pieces of '%s'", name, s.c_str());
            for (int id : ids) CHECK(id >= 0 && std::size_t(id) < t.vocab_size());
        }
    }
    for (auto name : {"spm_bpe_llama", "spm_unigram_bytes"}) {   // byte fallback: nothing is lost either
        tokenizer t(import::sentencepiece_tokenizer_json(slurp(data_dir() / all[name]["file"].str())));
        for (auto& s : text) {
            auto want = os::text::sanitize_utf8(s);
            auto ids = t.encode(s);
            // "▁" is SentencePiece's space, and a control piece in the text ("<s>") restarts the dummy prefix: both decode to other text
            bool exact = s.find("\xE2\x96\x81") == std::string::npos;
            for (int id : ids) exact &= !t.is_special(id) && id != t.token_id("<start_of_turn>") && id != t.token_id("<end_of_turn>");
            if (!exact) { (void)t.decode(ids); continue; }
            CHECK_MSG(t.decode(ids) == want, "%s: round trip of '%s'", name, s.c_str());
            std::string stream; for (int id : ids) stream += t.decode_one(id);   // streamed text keeps the dummy prefix's space
            bool dummy = std::string(name) == "spm_bpe_llama" && !ids.empty() && !t.is_special(ids[0]);
            CHECK_MSG(stream == (dummy ? " " : "") + want, "%s: streamed pieces of '%s'", name, s.c_str());
        }
    }
}

// every model type through a .fpk "tokenizer" section
TEST(fpk_roundtrip_every_type) {
    auto text = corpus();
    auto& all = goldens()["fixtures"];
    std::vector<std::pair<std::string, std::string>> docs;
    for (auto name : {"hf_bpe_gpt2", "hf_bpe_llama2", "hf_wordpiece_bert", "hf_unigram_t5", "hf_wordlevel", "legacy_v1"}) docs.push_back({name, fixture_json(all[name])});
    docs.push_back({"spm_bpe_llama", import::sentencepiece_tokenizer_json(slurp(data_dir() / "spm_bpe_llama.model"))});
    docs.push_back({"spm_unigram_charsmap", import::sentencepiece_tokenizer_json(slurp(data_dir() / "spm_unigram_charsmap.model"))});
    std::set<tokenizer::model_type> seen;
    for (auto& [name, doc] : docs) {
        auto path = tmp("rt.fpk");
        {
            modelfile::writer w;
            w.set_manifest(json::value(json::object{{"format", "fpk"}, {"name", name}}));
            w.set_section("tokenizer", doc);
            w.add_tensor("x", {2}, std::vector<float>{1, 2}, "f32");
            w.write(path.string());
        }
        {
            modelfile::reader r(path.string());
            CHECK(r.verify().empty());
            tokenizer a(doc), b(r.section_text("tokenizer"));
            seen.insert(b.type());
            CHECK(a.vocab_size() == b.vocab_size() && a.type() == b.type());
            for (auto& s : text) CHECK_MSG(a.encode(s, true) == b.encode(s, true) && a.decode(a.encode(s)) == b.decode(b.encode(s)), "%s", name.c_str());
        }
        fs::remove(path);
    }
    CHECK(seen.size() == 5);
}

// import_gguf end to end: a one-layer Llama whose GGUF has a SentencePiece ("llama") vocabulary and a chat template
TEST(gguf_import_llama_spm_end_to_end) {
    auto out = tmp("tiny_llama.fpk");
    auto rep = import::import_gguf((data_dir() / "gguf_tiny_llama.gguf").string(), out.string());
    CHECK(rep.arch == "llama");
    {
        auto m = inference::model::open(out.string());
        auto* t = m->tokenizer();
        CHECK(t && t->type() == tokenizer::model_type::sentencepiece_bpe);
        if (t) {
            CHECK(t->bos() == 1 && t->eos() == 2 && t->add_bos() && !t->add_eos());
            auto text = corpus();
            auto& ref = goldens()["fixtures"]["spm_bpe_llama"];
            for (std::size_t i = 0; i < text.size(); ++i) {
                if (!line_wanted(ref, i)) continue;
                auto want = ints(ref["cases"].arr()[i]["ids"]);
                CHECK_MSG(t->encode(text[i]) == want, "line %zu", i);
                want.insert(want.begin(), 1);
                CHECK(t->encode(text[i], true) == want);
            }
        }
        CHECK(m->file().has_section("chat_template") && m->file().section_text("chat_template").find("[INST]") != std::string::npos);
        CHECK(m->manifest()["add_bos"].b() && !m->manifest()["add_eos"].b());
        inference::session s(m);   // and the imported model runs
        auto ids = t->encode("Hello world", true);
        auto logits = s.forward(ids);
        CHECK(logits.size() == std::size_t(m->vocab()));
    }
    fs::remove(out);
}

// import_hf: a checkpoint directory with tokenizer.model (no tokenizer.json) and a chat template in tokenizer_config.json
TEST(hf_import_sentencepiece_and_chat_template) {
    auto dir = tmp("hf_dir"); fs::create_directories(dir);
    const int V = 560, d = 8;
    {   // model.safetensors: u64 header length, JSON header, raw f32 data
        std::vector<std::pair<std::string, std::vector<i64>>> ts = {{"model.embed_tokens.weight", {V, d}}, {"model.norm.weight", {d}}, {"lm_head.weight", {V, d}}};
        for (auto [n, s] : std::initializer_list<std::pair<const char*, std::vector<i64>>>{{"input_layernorm", {d}}, {"post_attention_layernorm", {d}}, {"self_attn.q_proj", {d, d}},
                 {"self_attn.k_proj", {4, d}}, {"self_attn.v_proj", {4, d}}, {"self_attn.o_proj", {d, d}}, {"mlp.gate_proj", {16, d}}, {"mlp.up_proj", {16, d}}, {"mlp.down_proj", {d, 16}}})
            ts.push_back({std::string("model.layers.0.") + n + ".weight", s});
        std::string header = "{"; std::vector<float> data; std::mt19937 rng(3); std::normal_distribution<float> N(0, 0.3f);
        for (auto& [n, s] : ts) {
            std::size_t count = 1; std::string shape; for (auto x : s) { count *= std::size_t(x); shape += (shape.empty() ? "" : ",") + std::to_string(x); }
            header += std::format("{}\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}", header.size() > 1 ? "," : "", n, shape, data.size() * 4, (data.size() + count) * 4);
            for (std::size_t i = 0; i < count; ++i) data.push_back(s.size() == 1 ? 1.f : N(rng));
        }
        header += "}";
        std::ofstream f(dir / "model.safetensors", std::ios::binary);
        std::uint64_t n = header.size(); f.write(reinterpret_cast<const char*>(&n), 8); f << header; f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size() * 4));
    }
    std::ofstream(dir / "config.json") << R"({"architectures":["LlamaForCausalLM"],"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
        "intermediate_size":16,"vocab_size":560,"rms_norm_eps":1e-5,"max_position_embeddings":64,"bos_token_id":1,"eos_token_id":2})";
    std::ofstream(dir / "tokenizer_config.json") << R"({"add_bos_token":true,"add_eos_token":false,"bos_token":"<s>",
        "added_tokens_decoder":{"1":{"content":"<s>","special":true},"560":{"content":"[INST]","special":false,"normalized":false},"561":{"content":"[/INST]","special":false,"normalized":false}},
        "chat_template":[{"name":"tool_use","template":"T"},{"name":"default","template":"{{ bos_token }}{% for m in messages %}[INST] {{ m.content }} [/INST]{% endfor %}"}]})";
    fs::copy_file(data_dir() / "spm_bpe_llama.model", dir / "tokenizer.model");
    auto out = tmp("hf_spm.fpk");
    import::import_hf(dir, out.string());
    {
        auto m = inference::model::open(out.string());
        auto* t = m->tokenizer();
        CHECK(t && t->type() == tokenizer::model_type::sentencepiece_bpe && t->add_bos() && !t->add_eos());
        tokenizer direct(import::sentencepiece_tokenizer_json(slurp(data_dir() / "spm_bpe_llama.model")));
        if (t) {
            CHECK(t->encode("Hello, world!") == direct.encode("Hello, world!"));
            auto ids = t->encode("[INST] hi [/INST]");
            CHECK(ids.front() == 560 && ids.back() == 561 && !t->is_special(560));   // added_tokens_decoder entries beyond the model's pieces
            CHECK(t->encode("hi", true).front() == 1);
        }
        CHECK(m->chat_template().find("[INST] {{ m.content }}") != std::string::npos);   // the "default" entry of the list
    }
    fs::remove(dir / "tokenizer_config.json");
    std::ofstream(dir / "chat_template.jinja") << "JINJA {{ messages[0].content }}";
    import::import_hf(dir, out.string());
    { auto m = inference::model::open(out.string()); CHECK(m->chat_template() == "JINJA {{ messages[0].content }}"); }
    fs::remove_all(dir); fs::remove(out);
}

// bad input is an error, never a crash
TEST(tokenizer_rejects_bad_json) {
    CHECK_THROWS(tokenizer("{}"));
    CHECK_THROWS(tokenizer(R"({"model":{"type":"Nope","vocab":{}}})"));
    CHECK_THROWS(tokenizer(R"({"model":{"type":"BPE","vocab":{"a":-1},"merges":[]}})"));
    CHECK_THROWS(tokenizer(R"({"model":{"type":"BPE","vocab":{"a":1e12},"merges":[]}})"));
    CHECK_THROWS(tokenizer(R"({"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"pre_tokenizer":{"type":"Split","pattern":{"Regex":"(a"},"behavior":"Removed"}})"));
    CHECK_THROWS(tokenizer(R"({"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"pre_tokenizer":{"type":"NoSuchThing"}})"));
    CHECK_THROWS(tokenizer(R"({"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"normalizer":{"type":"Precompiled","precompiled_charsmap":"AAAA"}})"));
    CHECK_THROWS(tokenizer(R"({"model":{"type":"Unigram","vocab":[["a",0]],"unk_id":7}})"));
    CHECK_THROWS(tokenizer(R"({"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"decoder":null,"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"x","type_id":0}}],"special_tokens":{}}})"));
    for (auto bad : {R"({"model":{"type":"BPE","vocab":{"a":0},"merges":[["a"]]}})", R"({"model":{"type":"Unigram","vocab":[["a"]]}})",
                     R"({"model":{"type":"WordLevel","vocab":{"a":0}},"decoder":null,"post_processor":{"type":"BertProcessing","sep":["a"],"cls":["a",0]}})"}) {
        try { tokenizer t(bad); CHECK_MSG(false, "accepted %s", bad); } catch (const forge_exception&) {} catch (...) { CHECK_MSG(false, "not a forge_exception: %s", bad); }
    }
    CHECK_THROWS(import::sentencepiece_tokenizer_json("\x0a\xff\xff\xff"));
    CHECK_THROWS(import::sentencepiece_tokenizer_json(""));
    tokenizer t(R"({"model":{"type":"BPE","vocab":{"a":0},"merges":["a a","b"]},"decoder":null,"added_tokens":[{"id":3,"content":""}]})");
    CHECK(t.encode("aa\xFFzz").size() == 2 && t.decode({0, 99, -1}) == "a" && t.decode_one(99).empty() && t.token_id("zz") == -1);
}

// segment ("token type") ids and sentence pairs through the post-processor: what an embedding session feeds BERT
TEST(token_types_and_pairs) {
    auto& all = goldens()["fixtures"];
    const std::string a = "The quick brown fox", b = "jumps over the lazy dog.";
    auto cat = [](std::initializer_list<std::vector<int>> parts) { std::vector<int> o; for (auto& p : parts) o.insert(o.end(), p.begin(), p.end()); return o; };
    {   // BertProcessing: [CLS] A [SEP] B [SEP], the second sequence and its [SEP] in segment 1
        tokenizer t(fixture_json(all["hf_wordpiece_bert"]));
        int cls = t.token_id("[CLS]"), sep = t.token_id("[SEP]");
        auto ia = t.encode(a), ib = t.encode(b);
        auto one = t.encode_typed(a);
        CHECK(one.ids == t.encode(a, true) && one.ids.front() == cls && one.ids.back() == sep);
        CHECK(one.type_ids == std::vector<int>(one.ids.size(), 0));
        auto two = t.encode_pair(a, b);
        CHECK(two.ids == cat({{cls}, ia, {sep}, ib, {sep}}));
        CHECK(two.type_ids == cat({std::vector<int>(ia.size() + 2, 0), std::vector<int>(ib.size() + 1, 1)}));
    }
    {   // TemplateProcessing "[CLS] $A [SEP] $B:1 [SEP]:1"
        tokenizer t(fixture_json(all["hf_wordpiece_cased"]));
        int cls = t.token_id("[CLS]"), sep = t.token_id("[SEP]");
        auto ia = t.encode(a), ib = t.encode(b);
        auto two = t.encode_pair(a, b);
        CHECK(two.ids == cat({{cls}, ia, {sep}, ib, {sep}}));
        CHECK(two.type_ids == cat({std::vector<int>(ia.size() + 2, 0), std::vector<int>(ib.size() + 1, 1)}));
        CHECK(t.encode_typed(a).ids == t.encode(a, true));
    }
    {   // RobertaProcessing: <s> A </s></s> B </s>, one segment
        tokenizer t(fixture_json(all["hf_bpe_roberta"]));
        int s = t.token_id("<s>"), e = t.token_id("</s>");
        auto ia = t.encode(a), ib = t.encode(b);
        auto two = t.encode_pair(a, b);
        CHECK(two.ids == cat({{s}, ia, {e, e}, ib, {e}}));
        CHECK(two.type_ids == std::vector<int>(two.ids.size(), 0));
    }
    {   // no post-processor: A then B inside the BOS / EOS the session set
        tokenizer t(fixture_json(all["hf_bpe_gpt2"]));
        auto ia = t.encode(a), ib = t.encode(b);
        t.set_special(0, 0); t.set_add_bos(true); t.set_add_eos(true);
        auto two = t.encode_pair(a, b);
        CHECK(two.ids == cat({{0}, ia, ib, {0}}));
        CHECK(two.type_ids == cat({std::vector<int>(ia.size() + 1, 0), std::vector<int>(ib.size() + 1, 1)}));
        CHECK(t.encode_typed(a).ids == cat({{0}, ia, {0}}));
    }
}

// a bare WordPiece vocab.txt (BERT checkpoints without tokenizer.json) gives the ids of the tokenizer.json it came from
TEST(wordpiece_vocab_txt) {
    auto& fx = goldens()["fixtures"]["hf_wordpiece_bert"];
    auto j = json::parse(fixture_json(fx));
    std::vector<std::string> pieces(j["model"]["vocab"].obj().size());
    for (auto& [t, id] : j["model"]["vocab"].obj()) pieces.at(std::size_t(id.num())) = t;
    std::string txt; for (auto& p : pieces) txt += p + "\r\n";   // CRLF line ends, as a Windows clone may have them
    auto cfg = json::parse(R"({"do_lower_case":true,"cls_token":"[CLS]","sep_token":{"content":"[SEP]"},"unk_token":"[UNK]"})");
    auto doc = import::wordpiece_vocab_tokenizer_json(txt, &cfg);
    CHECK(import::tokenizer_json_problem(doc).empty());
    tokenizer t(doc);
    CHECK(t.type() == tokenizer::model_type::wordpiece && t.vocab_size() == pieces.size() && t.has_post_processor());
    auto text = corpus(); auto& cases = fx["cases"].arr();
    for (std::size_t i = 0; i < cases.size(); ++i) {
        auto valid = os::text::sanitize_utf8(text[i]);
        CHECK_MSG(t.encode(valid) == ints(cases[i]["ids"]), "line %zu", i);
        CHECK_MSG(t.encode(valid, true) == ints(cases[i]["ids_special"]), "line %zu (special)", i);
    }
    // a cased vocabulary: do_lower_case false keeps case
    auto cased = json::parse(R"({"do_lower_case":false})");
    tokenizer c(import::wordpiece_vocab_tokenizer_json("[UNK]\n[CLS]\n[SEP]\nHello\nhello\n", &cased));
    CHECK(c.encode("Hello hello HELLO", true) == (std::vector<int>{1, 3, 4, 0, 2}));
    CHECK_THROWS(import::wordpiece_vocab_tokenizer_json(""));
    CHECK(!import::tokenizer_json_problem(R"({"model":{"type":"NoSuchModel"}})").empty());
}

// Files written while the runtime only had byte-level BPE keep other tokenizers in a "tokenizer_json" section: such a
// file still opens, and now tokenizes; a section this build cannot load leaves the model without a tokenizer.
TEST(legacy_tokenizer_json_section) {
    auto base = forge_test::make_tiny_model(tmp("legacy_base.fpk").string());
    auto rewrite = [&](const fs::path& out, const std::string& doc) {
        modelfile::reader r(base); modelfile::writer w;
        for (auto& [n, e] : r.tensors()) { auto t = r.load(n); w.add_tensor(n, t.shape, t.data, "f32"); }
        w.set_manifest(r.manifest());
        w.set_section("blueprint", std::string(r.section_text("blueprint"))); w.set_section("tokenizer_json", doc);
        w.write(out.string());
    };
    auto wp = tmp("legacy_wp.fpk"), bad = tmp("legacy_bad.fpk");
    auto doc = fixture_json(goldens()["fixtures"]["hf_wordpiece_bert"]);
    rewrite(wp, doc); rewrite(bad, R"({"model":{"type":"NoSuchModel"}})");
    {
        auto m = inference::model::open(wp.string());
        CHECK(m->tokenizer() && m->tokenizer()->type() == tokenizer::model_type::wordpiece);
        if (m->tokenizer()) CHECK(m->tokenizer()->encode("Hello, world!") == tokenizer(doc).encode("Hello, world!"));
    }
    { auto m = inference::model::open(bad.string()); CHECK(!m->tokenizer()); }
    fs::remove(base); fs::remove(wp); fs::remove(bad);
}
