// Chat helpers (inference/chat_tools.hpp) and the scheduler features forge-serve builds on them: end-of-turn tokens,
// tool-call formats (grammar and parser), per-request LoRA adapters with an adapter-keyed prefix cache, request
// failures that stay with their request, log-probabilities and logit bias.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/inference/inference.hpp>
#include <filesystem>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace forge;
using namespace forge::inference;

static std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / ("forge_chat_tools_" + std::to_string(::getpid()) + "_" + n)).string(); }
static const std::string& model_a() { static std::string p = forge_test::make_tiny_model(tmp("a.fpk")); return p; }

// what end_of_turn_tokens() needs of a model
struct fake_model {
    json::value man; modelfile::tokenizer tok; std::string tmpl; i64 v = 16;
    const json::value& manifest() const { return man; }
    const modelfile::tokenizer* tokenizer() const { return &tok; }
    const std::string& chat_template() const { return tmpl; }
    i64 vocab() const { return v; }
};
static const char* kTok = R"({"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2},"merges":[]},"added_tokens":[
  {"id":3,"content":"<s>","special":true},{"id":4,"content":"</s>","special":true},{"id":5,"content":"<|im_start|>","special":true},
  {"id":6,"content":"<|im_end|>","special":true},{"id":7,"content":"<|eot_id|>","special":true},{"id":8,"content":"<|endoftext|>","special":true}]})";

TEST(end_of_turn_from_manifest_and_template) {
    fake_model m{json::parse(R"({"eos_token_id":4})"), modelfile::tokenizer(kTok), "", 16};
    m.tok.set_special(3, 4);
    CHECK((end_of_turn_tokens(m) == std::vector<int>{4}));
    m.tmpl = "{% for x in messages %}<|im_start|>{{ x.role }}\n{{ x.content }}<|im_end|>\n{% endfor %}";
    CHECK((end_of_turn_tokens(m) == std::vector<int>{4, 6}));          // the marker the template writes
    CHECK((end_of_turn_tokens(m, false) == std::vector<int>{4}));      // raw completions: not template-derived
    m.man = json::parse(R"({"eos_token_id":4,"eos_token_ids":[4,7,7,99,-1,"x",8]})");   // out of range and junk entries are dropped
    CHECK((end_of_turn_tokens(m) == std::vector<int>{4, 7, 8, 6}));
    CHECK((end_of_turn_tokens(m, false) == std::vector<int>{4, 7, 8}));
}

TEST(tool_format_detection) {
    CHECK(detect_tool_format("...<tool_call>\n{{ x }}</tool_call>...") == tool_format::hermes);
    CHECK(detect_tool_format("[AVAILABLE_TOOLS]..[TOOL_CALLS]") == tool_format::mistral);
    CHECK(detect_tool_format("<|start_header_id|>ipython<|python_tag|>") == tool_format::llama3);
    CHECK(detect_tool_format("<|start_header_id|>{{ tools }}<|eot_id|>") == tool_format::llama3);
    CHECK(detect_tool_format("<s>[INST] {{ m }} [/INST]") == tool_format::generic);
    CHECK(detect_tool_format("") == tool_format::generic);
}

static std::vector<tool_call> parse_all(tool_format f, std::string_view text, std::string* content = nullptr) {
    tool_call_parser p(f, {"get_weather", "get_time"});
    p.update(text, true);
    if (content) *content = std::string(text.substr(0, p.content_end()));
    return p.finished();
}
// feeding the text one byte at a time must end in the same result, and never show call text as content
static void check_incremental(tool_format f, const std::string& text) {
    tool_call_parser whole(f, {"get_weather", "get_time"}); whole.update(text, true);
    tool_call_parser inc(f, {"get_weather", "get_time"});
    std::size_t shown = 0;
    for (std::size_t n = 1; n <= text.size(); ++n) {
        inc.update(std::string_view(text).substr(0, n), n == text.size());
        CHECK_MSG(inc.content_end() <= whole.content_end() || whole.finished().empty(), "content ran into the call at %zu of '%s'", n, text.c_str());
        if (!whole.finished().empty()) shown = std::max(shown, inc.content_end());
    }
    auto a = whole.finished(), b = inc.finished();
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) CHECK(a[i].name == b[i].name && a[i].arguments == b[i].arguments);
    CHECK(inc.content_end() == whole.content_end());
    (void)shown;
}

TEST(tool_call_parser_formats) {
    std::string content;
    auto c = parse_all(tool_format::hermes, "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"n\": [1, {\"k\": \"}\"}]}}\n</tool_call>", &content);
    CHECK(c.size() == 1 && c[0].name == "get_weather" && content == "Let me check.\n");
    CHECK(c[0].arguments == "{\"city\": \"Paris\", \"n\": [1, {\"k\": \"}\"}]}");
    c = parse_all(tool_format::hermes, "<tool_call>{\"name\":\"get_time\",\"arguments\":{}}</tool_call>\n<tool_call>{\"arguments\":{\"city\":\"x\"},\"name\":\"get_weather\"}</tool_call>");
    CHECK(c.size() == 2 && c[0].name == "get_time" && c[0].arguments == "{}" && c[1].name == "get_weather" && c[1].arguments == "{\"city\":\"x\"}");
    c = parse_all(tool_format::mistral, "[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}, {\"name\": \"get_time\", \"arguments\": {}}]", &content);
    CHECK(c.size() == 2 && c[0].arguments == "{\"city\": \"Rome\"}" && c[1].name == "get_time" && content.empty());
    c = parse_all(tool_format::llama3, "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Oslo\"}}");
    CHECK(c.size() == 1 && c[0].arguments == "{\"city\": \"Oslo\"}");
    c = parse_all(tool_format::llama3, "  {\"name\": \"get_time\", \"parameters\": {}}");      // Llama 3.1 / 3.2: bare JSON
    CHECK(c.size() == 1 && c[0].name == "get_time");
    c = parse_all(tool_format::generic, "{\"name\": \"get_time\"}");                             // no arguments: {}
    CHECK(c.size() == 1 && c[0].arguments == "{}");
    c = parse_all(tool_format::generic, "{\"name\": \"get_time\", \"arguments\": \"{\\\"tz\\\": 1}\"}");   // arguments as a JSON string
    CHECK(c.size() == 1 && c[0].arguments == "{\"tz\": 1}");
    // not calls: an undeclared function as bare JSON, JSON after text, a cut-off call, a call without a name
    c = parse_all(tool_format::generic, "{\"name\": \"rm_rf\", \"arguments\": {}}", &content);
    CHECK(c.empty() && content == "{\"name\": \"rm_rf\", \"arguments\": {}}");
    std::string after_text = "The answer is {\"name\": \"get_time\"}";
    CHECK(parse_all(tool_format::generic, after_text, &content).empty() && content == after_text);
    CHECK(parse_all(tool_format::hermes, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par").empty());
    CHECK(parse_all(tool_format::hermes, "<tool_call>{\"arguments\": {}}</tool_call>").empty());
    CHECK(parse_all(tool_format::hermes, "<tool_call>not json</tool_call>").empty());
    std::string plain = "plain words with [brackets] and {braces}";
    CHECK(parse_all(tool_format::mistral, plain, &content).empty() && content == plain);
    // hostile shapes: deep nesting (no recursion in the scanner), a marker flood, escapes at the end
    std::string deep = "<tool_call>{\"name\":\"get_time\",\"arguments\":" + std::string(200000, '[') + std::string(200000, ']') + "}</tool_call>";
    c = parse_all(tool_format::hermes, deep); CHECK(c.size() == 1 && c[0].arguments.size() == 400000);
    std::string flood; for (int i = 0; i < 500; ++i) flood += "<tool_call>{\"name\":\"get_time\"}</tool_call>";
    CHECK(parse_all(tool_format::hermes, flood).size() == 64);   // bounded
    CHECK(parse_all(tool_format::hermes, "<tool_call>{\"name\":\"get_time\\").empty());
    for (auto f : {tool_format::hermes, tool_format::mistral, tool_format::llama3, tool_format::generic}) {
        check_incremental(f, "Sure. <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>");
        check_incremental(f, "[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}]");
        check_incremental(f, "{\"name\": \"get_time\", \"parameters\": {\"a\": [1, 2, \"<tool_call>\"]}}");
        check_incremental(f, "{\"other\": 1}");
        check_incremental(f, "just text < [ { \" with pieces of markers <tool_cal and [TOOL_CALL");
    }
}

TEST(tool_call_parser_streams_arguments) {
    tool_call_parser p(tool_format::hermes, {"get_weather"});
    std::string text = "ok <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    bool named_before_done = false; std::size_t partial_args = 0;
    for (std::size_t n = 1; n <= text.size(); ++n) {
        p.update(std::string_view(text).substr(0, n), false);
        if (!p.calls().empty() && p.calls()[0].named && !p.calls()[0].done) { named_before_done = true; partial_args = std::max(partial_args, p.calls()[0].arguments.size()); }
        CHECK(p.content_end() <= 3);
    }
    CHECK(named_before_done && partial_args > 5);   // the name and a growing arguments text are there while the call is still open
    CHECK(p.calls().size() == 1 && p.calls()[0].done && p.calls()[0].arguments == "{\"city\": \"Paris\"}");
}

static const char* kTools = R"([
  {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"enum":["Paris","Rome"]},"days":{"type":"integer"}},"required":["city"]}}},
  {"type":"function","function":{"name":"get_time"}}])";

// every byte string the grammar accepts must come back out of the parser as calls
static bool accepts(const grammar& g, std::string_view s) {
    auto st = g.start();
    for (unsigned char c : s) { st = g.advance(st, c); if (st.empty()) return false; }
    return grammar::can_end(st);
}
TEST(tool_call_grammar_matches_parser) {
    auto tools = json::parse(kTools);
    struct sample { tool_format f; const char* text; bool ok; };
    for (auto& s : {sample{tool_format::hermes, "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>", true},
                    sample{tool_format::hermes, "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\",\"days\":3}}\n</tool_call>", true},
                    sample{tool_format::hermes, "<tool_call>\n{\"name\":\"get_time\",\"arguments\":{}}\n</tool_call>", true},
                    sample{tool_format::hermes, "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}\n</tool_call>", false},   // not in the enum
                    sample{tool_format::hermes, "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{}}\n</tool_call>", false},                  // city is required
                    sample{tool_format::hermes, "<tool_call>\n{\"name\":\"nope\",\"arguments\":{}}\n</tool_call>", false},
                    sample{tool_format::hermes, "hello", false},
                    sample{tool_format::mistral, "[TOOL_CALLS][{\"name\":\"get_time\",\"arguments\":{\"z\":[1,true]}}]", true},
                    sample{tool_format::llama3, "{\"name\":\"get_weather\",\"parameters\":{\"city\":\"Rome\"}}", true},
                    sample{tool_format::generic, "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Rome\"}}", true},
                    sample{tool_format::generic, "{\"name\":\"get_weather\",\"parameters\":{\"city\":\"Rome\"}}", false}}) {
        grammar g(tool_call_grammar(s.f, tools));
        CHECK_MSG(accepts(g, s.text) == s.ok, "%s: %s", tool_format_name(s.f), s.text);
        if (s.ok) { auto c = parse_all(s.f, s.text); CHECK_MSG(c.size() == 1, "%s", s.text); }
    }
    // a named function: only that one; parallel: several calls
    grammar only(tool_call_grammar(tool_format::hermes, tools, "get_time"));
    CHECK(accepts(only, "<tool_call>\n{\"name\":\"get_time\",\"arguments\":{}}\n</tool_call>"));
    CHECK(!accepts(only, "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>"));
    const char* two = "<tool_call>\n{\"name\":\"get_time\",\"arguments\":{}}\n</tool_call>\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Rome\"}}\n</tool_call>";
    CHECK(!accepts(grammar(tool_call_grammar(tool_format::hermes, tools)), two));
    CHECK(accepts(grammar(tool_call_grammar(tool_format::hermes, tools, "", true)), two) && parse_all(tool_format::hermes, two).size() == 2);
    CHECK(accepts(grammar(tool_call_grammar(tool_format::mistral, tools, "", true)), "[TOOL_CALLS][{\"name\":\"get_time\",\"arguments\":{}},{\"name\":\"get_time\",\"arguments\":{}}]"));
    CHECK_THROWS(tool_call_grammar(tool_format::hermes, tools, "missing"));
    CHECK_THROWS(tool_call_grammar(tool_format::hermes, json::parse("[]")));
    CHECK_THROWS(tool_decls(json::parse(R"([{"type":"function","function":{}}])")));
    CHECK_THROWS(tool_decls(json::parse(R"([{"type":"function","function":{"name":"a"}},{"type":"function","function":{"name":"a"}}])")));
    CHECK_THROWS(tool_decls(json::parse(R"({"not":"a list"})")));
    CHECK(tool_decls(json::parse(R"([{"type":"retrieval"}])")).empty());
}

TEST(tool_messages_arguments_become_objects) {
    auto in = json::parse(R"([{"role":"user","content":"hi"},
      {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\": 1}"}},
                                                        {"id":"c2","type":"function","function":{"name":"g","arguments":"not json"}}]},
      {"role":"tool","tool_call_id":"c1","content":"ok"}])");
    auto out = normalize_tool_messages(in);
    auto& calls = out.arr()[1]["tool_calls"].arr();
    CHECK(calls[0]["function"]["arguments"].is_obj() && calls[0]["function"]["arguments"]["a"].num() == 1 && calls[0]["id"].str() == "c1");
    CHECK(calls[1]["function"]["arguments"].is_str());   // left alone
    CHECK(json::dump(out.arr()[0]) == json::dump(in.arr()[0]) && json::dump(out.arr()[2]) == json::dump(in.arr()[2]));
}

TEST(json_schema_rules_compose) {
    // two schemas in one grammar: prefixed rule names do not collide
    std::string rules;
    auto a = json_schema_rules(json::parse(R"({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]})"), "a_", rules);
    auto b = json_schema_rules(json::parse(R"({"type":"object","properties":{"y":{"type":"boolean"}},"required":["y"]})"), "b_", rules);
    grammar g("root ::= " + a + " \"|\" " + b + "\n" + rules + json_schema_base_rules());
    CHECK(accepts(g, "{\"x\":12}|{\"y\":true}") && !accepts(g, "{\"y\":true}|{\"x\":12}"));
    CHECK(json_schema_to_gbnf(json::parse("{}")).rfind("root ::= value\n", 0) == 0);
}

// ---- scheduler
static model::lora rand_lora(i64 out, i64 in, unsigned seed, i64 r = 2) {
    std::mt19937 g(seed); std::normal_distribution<float> N(0, 1);
    model::lora l; l.a.shape = {r, in}; l.b.shape = {out, r};
    l.a.data.resize(std::size_t(r * in)); l.b.data.resize(std::size_t(out * r));
    for (auto& x : l.a.data) x = N(g);
    for (auto& x : l.b.data) x = N(g);
    return l;
}

TEST(batch_scheduler_adapters_per_request) {
    auto m = model::open(model_a());
    // "attn" changes an attention projection, so the KV rows of every position after the first differ from the base
    m->add_adapter("attn", {{"layer.0.wq", rand_lora(32, 32, 1)}}, 8.f);
    m->add_adapter("head", {{"lm_head", rand_lora(260, 32, 2)}}, 8.f);
    std::vector<int> prompt = {256, 1, 2, 3, 4, 5, 6, 7, 8};
    auto alone = [&](const std::string& ad) { inference::session s(m); s.set_adapter(ad); sampler g; return s.generate(prompt, 12, g); };
    auto wb = alone(""), wa = alone("attn"), wh = alone("head");
    CHECK(wb != wa && wb != wh);
    batch_scheduler b(m, {.max_active = 2, .prefix_cache = 8});
    std::map<std::string, std::vector<int>> got; std::map<std::string, std::string> why;
    auto run = [&](const std::string& tag, const std::string& ad) {
        b.submit({.prompt = prompt, .max_new = 12, .adapter = ad, .on_done = [&, tag](const std::vector<int>& o, const std::string& w) { got[tag] = o; why[tag] = w; }});
    };
    // base first: its finished sequence sits in the prefix cache when the adapter requests arrive with the same prompt
    run("b1", ""); b.run_until_idle();
    auto hits0 = b.stats().prefix_hits;
    run("a1", "attn"); b.run_until_idle();
    CHECK_MSG(b.stats().prefix_hits == hits0, "an adapter request took KV rows computed with the base weights");
    CHECK(got["a1"] == wa);
    run("a2", "attn"); b.run_until_idle();
    CHECK(b.stats().prefix_hits == hits0 + 1 && got["a2"] == wa);   // same adapter: shared
    run("b2", ""); run("h1", "head"); run("a3", "attn"); run("b3", ""); b.run_until_idle();   // mixed in one batch
    CHECK(got["b1"] == wb && got["b2"] == wb && got["b3"] == wb && got["h1"] == wh && got["a3"] == wa);
    // an adapter the model does not have fails that request only
    run("bad", "nope"); run("b4", ""); b.run_until_idle();
    CHECK_MSG(why["bad"].rfind("error: ", 0) == 0 && why["bad"].find("nope") != std::string::npos, "%s", why["bad"].c_str());
    CHECK(got["b4"] == wb && why["b4"] == "length" && b.stats().failed == 1);
    CHECK((m->adapter_names() == std::vector<std::string>{"attn", "head"}) && m->has_adapter("attn") && !m->has_adapter("nope"));
}

TEST(batch_scheduler_adapter_prefix_spill) {
    auto m = model::open(model_a());
    m->add_adapter("attn", {{"layer.0.wq", rand_lora(32, 32, 1)}}, 8.f);
    std::vector<int> prompt = {256, 9, 8, 7, 6, 5, 4};
    inference::session ref(m); ref.set_adapter("attn"); sampler g; auto want = ref.generate(prompt, 10, g);
    mem::spill_store store;
    batch_scheduler b(m, {.max_active = 1, .prefix_cache = 1, .spill = &store});
    std::vector<int> got;
    auto run = [&](std::vector<int> p, const std::string& ad) { b.submit({.prompt = std::move(p), .max_new = 10, .adapter = ad, .on_done = [&](const std::vector<int>& o, const std::string&) { got = o; }}); b.run_until_idle(); };
    run(prompt, "attn");                    // cached
    run({256, 100, 101, 102}, "");          // evicts the adapter prefix to the spill store
    CHECK(b.spilled() == 1);
    run(prompt, "");                        // base request, same tokens: must not load the adapter's rows
    CHECK(b.stats().prefix_reloads == 0);
    run(prompt, "attn");                    // reloaded for the adapter it belongs to
    CHECK(b.stats().prefix_reloads == 1 && got == want);
}

TEST(batch_scheduler_request_failures_are_isolated) {
    auto m = model::open(model_a());
    batch_scheduler b(m, {.max_active = 4, .session = {.context = 16}});
    std::map<int, std::string> why; std::map<int, std::size_t> len;
    auto done = [&](int i) { return [&, i](const std::vector<int>& o, const std::string& w) { why[i] = w; len[i] = o.size(); }; };
    b.submit({.prompt = {256, 1, 2}, .max_new = 5, .on_done = done(0)});
    b.submit({.prompt = std::vector<int>(40, 5), .max_new = 5, .on_done = done(1)});     // longer than the context
    b.submit({.prompt = {}, .max_new = 5, .on_done = done(2)});                          // empty
    sampler_config no_eos; no_eos.logit_bias = {{257, -1e9f}};
    b.submit({.prompt = {256, 4}, .max_new = 100, .sampling = no_eos, .on_done = done(3)});   // runs into the end of the context
    b.submit({.prompt = {256, 1, 2}, .max_new = 5, .on_done = done(4)});
    b.run_until_idle();
    CHECK(why[0] == "length" && len[0] == 5 && why[4] == "length" && len[4] == 5);
    CHECK_MSG(why[1].rfind("error: ", 0) == 0 && why[1].find("context") != std::string::npos, "%s", why[1].c_str());
    CHECK_MSG(why[2].rfind("error: ", 0) == 0, "%s", why[2].c_str());
    CHECK_MSG(why[3] == "length" && len[3] == 15, "%s after %zu tokens", why[3].c_str(), len[3]);
    CHECK(b.stats().failed == 2 && b.stats().completed == 5 && b.idle());
}

TEST(batch_scheduler_logprobs_bias_and_stop_tokens) {
    auto m = model::open(model_a());
    batch_scheduler b(m, {.max_active = 2});
    std::vector<int> prompt = {256, 10, 20, 30};
    std::vector<token_logprob> lps; std::vector<int> out;
    b.submit({.prompt = prompt, .max_new = 6, .top_logprobs = 4, .on_logprobs = [&](const token_logprob& l) { lps.push_back(l); },
              .on_done = [&](const std::vector<int>& o, const std::string&) { out = o; }});
    b.run_until_idle();
    CHECK(lps.size() == 6 && out.size() == 6);
    inference::session s(m); auto logits = s.forward(prompt);
    double sum = 0; for (float x : logits) sum += std::exp(double(x));
    for (std::size_t i = 0; i < lps.size(); ++i) {
        CHECK(lps[i].token == out[i] && lps[i].top.size() == 4 && lps[i].top[0].first == out[i]);   // greedy: the sampled token is the likeliest
        CHECK(std::abs(lps[i].top[0].second - lps[i].logprob) < 1e-6f && lps[i].logprob < 0);
        for (std::size_t k = 1; k < 4; ++k) CHECK(lps[i].top[k].second <= lps[i].top[k - 1].second);
    }
    CHECK(std::abs(double(lps[0].logprob) - (double(logits[std::size_t(out[0])]) - std::log(sum))) < 1e-4);
    // logit bias: ban the greedy token, then force another one; the reported log-probabilities stay the model's own
    std::vector<int> banned, forced; std::vector<token_logprob> flp;
    sampler_config ban; ban.logit_bias = {{out[0], -1e9f}, {100000, 5.f}, {-3, 5.f}};   // ids outside the vocabulary are ignored
    sampler_config force; force.logit_bias = {{42, 1e9f}};
    b.submit({.prompt = prompt, .max_new = 3, .sampling = ban, .on_done = [&](const std::vector<int>& o, const std::string&) { banned = o; }});
    b.submit({.prompt = prompt, .max_new = 3, .sampling = force, .top_logprobs = 0, .on_logprobs = [&](const token_logprob& l) { flp.push_back(l); },
              .on_done = [&](const std::vector<int>& o, const std::string&) { forced = o; }});
    b.run_until_idle();
    CHECK(banned.size() == 3 && banned[0] != out[0] && banned[0] == lps[0].top[1].first);
    CHECK((forced == std::vector<int>{42, 42, 42}) && flp.size() == 3 && flp[0].top.empty());
    CHECK(std::abs(double(flp[0].logprob) - (double(logits[42]) - std::log(sum))) < 1e-4);
    // stop tokens end the request like EOS ("stop"), and a grammar lets them end the output where it can end
    std::vector<int> stopped; std::string why;
    b.submit({.prompt = prompt, .max_new = 6, .stop_tokens = {out[2]}, .on_done = [&](const std::vector<int>& o, const std::string& w) { stopped = o; why = w; }});
    b.run_until_idle();
    CHECK(why == "stop" && stopped.size() == 3 && stopped.back() == out[2]);
    sampler_config push; push.logit_bias = {{200, 50.f}};   // token 200 is not text the grammar accepts: only allowed as an end token
    auto g = std::make_shared<grammar>("root ::= \"ab\" [a-z]*");
    b.submit({.prompt = prompt, .max_new = 20, .sampling = push, .grammar = g, .stop_tokens = {200}, .on_done = [&](const std::vector<int>& o, const std::string& w) { stopped = o; why = w; }});
    b.run_until_idle();
    auto* tok = m->tokenizer();
    CHECK(why == "stop" && stopped.size() == 3 && stopped.back() == 200 && tok->decode_one(stopped[0]) + tok->decode_one(stopped[1]) == "ab");
}

TEST(embed_batch_requires_embedding_model) {
    inference::session s(model::open(model_a()));
    std::vector<std::vector<int>> ids = {{1, 2}, {3}};
    CHECK_THROWS(s.embed_batch(ids));   // a generative model
}

TEST(lora_adapter_safetensors_file) {
    // model::add_adapter_file also reads <weight>.lora_a / .lora_b from a .safetensors (what forge-serve --lora takes)
    auto m = model::open(model_a());
    auto l = rand_lora(260, 32, 5);
    auto path = tmp("lora.safetensors");
    {
        json::object hdr; std::string data;
        auto add = [&](const std::string& n, const shape_t& sh, const std::vector<float>& v) {
            json::array dims; for (auto d : sh) dims.push_back(json::value(double(d)));
            hdr.emplace_back(n, json::value(json::object{{"dtype", "F32"}, {"shape", std::move(dims)},
                {"data_offsets", json::value(json::array{json::value(double(data.size())), json::value(double(data.size() + v.size() * 4))})}}));
            data.append(reinterpret_cast<const char*>(v.data()), v.size() * 4);
        };
        add("lm_head.lora_a", l.a.shape, l.a.data); add("lm_head.lora_b", l.b.shape, l.b.data); add("lora_alpha", {1}, {6.f});
        std::string h = json::dump(json::value(std::move(hdr)));
        std::uint64_t n = h.size();
        std::FILE* f = std::fopen(path.c_str(), "wb");
        std::fwrite(&n, 8, 1, f); std::fwrite(h.data(), 1, h.size(), f); std::fwrite(data.data(), 1, data.size(), f); std::fclose(f);
    }
    m->add_adapter_file("file", path); m->add_adapter("mem", {{"lm_head", l}}, 6.f);
    std::vector<int> ids = {256, 3, 4, 5};
    inference::session a(m), b(m); a.set_adapter("file"); b.set_adapter("mem");
    auto la = a.forward(ids), lb = b.forward(ids);
    float d = 0; for (std::size_t i = 0; i < la.size(); ++i) d = std::max(d, std::abs(la[i] - lb[i]));
    CHECK(d == 0.f);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(model_a(), ec);
}
