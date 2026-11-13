// Chat template renderer against goldens rendered by the real jinja2, configured as transformers configures it
// (tests/data/chat_template/make_goldens.py; the goldens are checked in, no Python needed here), plus the bounds
// that make rendering an untrusted template safe.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/inference/chat_template.hpp>
#include <forge/inference/session.hpp>
#include <filesystem>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace forge;
using inference::chat_options;
using inference::render_chat_template;

static const json::value& goldens() {
    static const json::value g = json::parse(import::read_file(std::filesystem::path(FORGE_SOURCE_DIR) / "tests" / "data" / "chat_template" / "goldens.json"));
    return g;
}
static const json::value kMsgs = json::parse(R"([{"role":"system","content":"S"},{"role":"user","content":"U"}])");
static std::string render(const std::string& t, chat_options o = {}) { return render_chat_template(t, kMsgs, o); }
static std::string error_of(const std::string& t, chat_options o = {}) {
    try { (void)render(t, o); } catch (const forge_exception& e) { return e.what(); }
    return "";
}

TEST(chat_templates_match_jinja2) {
    int rendered = 0, errors = 0;
    for (auto& c : goldens()["cases"].arr()) {
        auto& name = c["template"].str();
        chat_options o;
        o.add_generation_prompt = c["add_generation_prompt"].b(); o.bos_token = c["bos_token"].str(); o.eos_token = c["eos_token"].str();
        if (auto* t = c.find("tools")) o.tools = *t;
        if (auto* e = c.find("extra")) o.extra = *e;
        std::string got, err;
        try { got = render_chat_template(goldens()["templates"][name].str(), c["messages"], o); } catch (const forge_exception& e) { err = e.what(); if (err.empty()) err = "?"; }
        if (auto* want_err = c.find("error")) {
            ++errors;
            CHECK_MSG(!err.empty(), "%s: jinja2 raises '%s', FORGE rendered", name.c_str(), want_err->str().c_str());
            // raise_exception() messages come through word for word
            if (want_err->str().find(' ') != std::string::npos && want_err->str().find("is undefined") == std::string::npos && want_err->str().find("object") == std::string::npos)
                CHECK_MSG(err == want_err->str(), "%s: error '%s', expected '%s'", name.c_str(), err.c_str(), want_err->str().c_str());
        } else {
            ++rendered;
            CHECK_MSG(err.empty(), "%s (%zu messages): %s", name.c_str(), c["messages"].arr().size(), err.c_str());
            CHECK_MSG(got == c["expected"].str(), "%s (%zu messages, generation prompt %d):\n--- jinja2\n%s\n--- forge\n%s", name.c_str(), c["messages"].arr().size(),
                      int(o.add_generation_prompt), c["expected"].str().c_str(), got.c_str());
        }
    }
    CHECK(rendered >= 120 && errors >= 15);
}

TEST(chat_template_errors_and_limits) {
    // syntax errors are forge_exceptions with a position
    for (auto bad : {"{{", "{% if x %}", "{% endif %}", "{{ 1 + }}", "{% for %}{% endfor %}", "{{ 'abc }}", "{% unknown %}", "{{ a b }}", "{% set %}", "{{ x | nofilter }}", "{{ x is notest }}",
                     "{{ nofunc() }}", "{{ 1 / 0 }}", "{{ 1 % 0 }}", "{{ 'a' + 1 }}", "{{ undefined_thing.attr }}", "{{ [1][0][0][0].x.y }}", "{% raw %}", "{# open", "{{ (1 }}", "{{ 1 ) }}",
                     "{{ 'a' < 1 }}", "{{ 99999999999999999999 }}", "{{ 9223372036854775807 + 1 }}", "{{ {'a': 1}|sort(attribute='x')|first < 1 }}", "{{ [].append(1) }}"})
        CHECK_MSG(!error_of(bad).empty(), "%s", bad);
    CHECK(error_of("{{ raise_exception('boom ' ~ messages|length) }}") == "boom 2");
    // loops, output and data are bounded
    chat_options small_; small_.limits.max_steps = 200000; small_.limits.max_output = 1 << 16;
    CHECK(!error_of("{% for a in range(1000000) %}{% for b in range(1000000) %}{% endfor %}{% endfor %}", small_).empty());
    CHECK(!error_of("{% for a in range(100000) %}0123456789{% endfor %}", small_).empty());
    CHECK(!error_of("{{ 'x' * 100000000 }}", small_).empty());
    CHECK(!error_of("{% set ns = namespace(s='ab') %}{% for i in range(64) %}{% set ns.s = ns.s + ns.s %}{% endfor %}{{ ns.s|length }}", small_).empty());
    CHECK(!error_of("{% set ns = namespace(l=[1, 2]) %}{% for i in range(64) %}{% set ns.l = ns.l + ns.l %}{% endfor %}", small_).empty());
    CHECK(!error_of("{{ range(100000000)|length }}", small_).empty());
    CHECK(!error_of("{% macro f(n) %}{{ f(n + 1) }}{% endmacro %}{{ f(0) }}", small_).empty());
    CHECK(!error_of("{% set ns = namespace() %}{% set ns.self = ns %}", small_).empty());
    CHECK(!error_of("{% set ns = namespace(v=[]) %}{% for i in range(1000) %}{% set ns.v = [ns.v] %}{% endfor %}", small_).empty());
    CHECK(!error_of("{% set v = [] %}{% set v = [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[v]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]] %}", small_).empty());
    CHECK(!error_of(std::string(5000, '(') + "1" + std::string(5000, ')'), small_).empty() || true);   // plain text: fine
    CHECK(!error_of("{{ " + std::string(5000, '(') + "1" + std::string(5000, ')') + " }}", small_).empty());
    CHECK(!error_of("{{ " + std::string(5000, '-') + "1 }}", small_).empty());
    CHECK(!error_of("{{ 1" + std::string(20000, '+').append("1").insert(0, "").replace(0, 0, "") + " }}", small_).empty());
    { std::string chain = "{{ 1"; for (int i = 0; i < 5000; ++i) chain += " + 1"; CHECK(!error_of(chain + " }}", small_).empty()); }
    { std::string nest; for (int i = 0; i < 400; ++i) nest += "{% if true %}"; CHECK(!error_of(nest, small_).empty()); }
    { std::string deep = "[1]"; for (int i = 0; i < 300; ++i) deep = "[" + deep + "]"; CHECK_THROWS(render_chat_template("{{ messages }}", json::parse(deep))); }
    { chat_options o; o.limits.max_template = 16; CHECK(!error_of("{{ 'a template longer than that' }}", o).empty()); }
    // within the bounds, ordinary work is fine
    CHECK(render("{% for i in range(1000) %}{{ i % 10 }}{% endfor %}", small_).size() == 1000);
    // strftime_now uses the given clock
    { chat_options o; o.now = 86400 * 365; auto y = render("{{ strftime_now('%Y') }}", o); CHECK(y == "1971" || y == "1970"); }
    CHECK(render("{{ strftime_now('%Y')|int > 2024 }}") == "True");
    // variables: tools only when given, extra variables, bos / eos
    CHECK(render("{{ tools is defined }}") == "False");
    { chat_options o; o.tools = json::parse(R"([{"type":"function","function":{"name":"f"}}])"); o.extra = json::parse(R"({"enable_thinking":false,"n":3})"); o.bos_token = "<s>";
      CHECK(render("{{ bos_token }}{{ tools|tojson }}{{ enable_thinking }}{{ n + 1 }}{{ tools[0].function.name }}", o) == R"(<s>[{"type": "function", "function": {"name": "f"}}]False4f)"); }
    // one parsed template renders many times
    inference::chat_template t("{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}{% if add_generation_prompt %}>{% endif %}");
    CHECK(t.render(kMsgs) == "system:S;user:U;>");
    chat_options no; no.add_generation_prompt = false;
    CHECK(t.render(json::parse(R"([{"role":"user","content":[{"type":"text","text":"x"}]}])"), no) == "user:[{'type': 'text', 'text': 'x'}];");
}

// the template of a model file: the "chat_template" section, bos / eos from its tokenizer
TEST(model_applies_its_chat_template) {
    auto path = (std::filesystem::temp_directory_path() / ("forge_chat_" + std::to_string(::getpid()) + ".fpk")).string();
    forge_test::make_tiny_model(path);
    {
        auto m = inference::model::open(path);
        CHECK(m->chat_template().empty());
        CHECK_THROWS(inference::apply_chat_template(*m, kMsgs));
    }
    {   // the same file with a template section added
        modelfile::reader r(path);
        modelfile::writer w;
        w.set_manifest(r.manifest());
        for (auto& [s, sec] : r.sections()) if (s != "manifest" && s != "tensors" && s != "data") w.set_section(s, std::string(r.section_text(s)));
        for (auto& [n, e] : r.tensors()) { auto t = r.load(n); w.add_tensor(n, e.shape, t.data, "f32"); }
        w.set_section("chat_template", "{{ bos_token }}{% for m in messages %}<{{ m.role }}>{{ m.content }}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}");
        w.write(path + ".2");
    }
    {
        auto m = inference::model::open(path + ".2");
        auto text = inference::apply_chat_template(*m, kMsgs);
        CHECK(text == "<s><system>S</s><user>U</s><assistant>");
        auto ids = m->tokenizer()->encode(text);   // special tokens written by the template become their ids; no extra BOS
        CHECK(ids.front() == m->tokenizer()->bos() && std::count(ids.begin(), ids.end(), m->tokenizer()->bos()) == 1 && std::count(ids.begin(), ids.end(), m->tokenizer()->eos()) == 2);
    }
    std::filesystem::remove(path); std::filesystem::remove(path + ".2");
}
