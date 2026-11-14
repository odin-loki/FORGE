// Remote AI client (inference/remote.hpp): configuration from the environment, request bodies and headers, and
// reply parsing for each provider, all without a network (a fake transport). serve_loopback talks to a real
// forge-serve when FORGE_TEST_SERVE_URL is set (tests/remote_ai_serve_test.py starts one on the tiny golden model).
#include "check.hpp"
#include <forge/inference/remote.hpp>
#include <string>

using namespace forge;
namespace rm = forge::inference::remote;
using forge_test::set_env;

static void clear_ai_env() {
    for (const char* k : {"FORGE_AI_PROVIDER", "FORGE_AI_URL", "FORGE_AI_ENDPOINT", "FORGE_AI_MODEL", "FORGE_AI_KEY_ENV", "FORGE_AI_KEY",
                          "FORGE_AI_CMD", "FORGE_AI_GRAMMAR", "FORGE_AI_TIMEOUT_MS", "FORGE_AI_FALLBACKS"}) set_env(k, nullptr);
}
static bool has(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }
static rm::request chat(std::string sys, std::string user) {
    rm::request r; r.messages = {{"system", std::move(sys)}, {"user", std::move(user)}}; r.max_tokens = 32; return r;
}

TEST(config_from_env_providers) {
    clear_ai_env();
    std::string why = "x";
    CHECK(!rm::config_from_env(&why) && why.empty());   // nothing configured
    // the older FORGE_AI_ENDPOINT / FORGE_AI_KEY still work: an OpenAI-compatible server
    set_env("FORGE_AI_ENDPOINT", "http://127.0.0.1:9/"); set_env("FORGE_AI_KEY", "k-legacy");
    auto c = rm::config_from_env(&why);
    CHECK(c && c->kind == rm::provider::openai && c->key_env == "FORGE_AI_KEY" && c->max_tokens_field == "max_tokens");
    CHECK(c && rm::endpoint(*c) == "http://127.0.0.1:9/v1/chat/completions");
    clear_ai_env();
    set_env("FORGE_AI_PROVIDER", "ollama");
    c = rm::config_from_env(&why);
    CHECK(c && rm::endpoint(*c) == "http://127.0.0.1:11434/v1/chat/completions" && c->key_env.empty());
    set_env("FORGE_AI_PROVIDER", "Claude"); set_env("FORGE_AI_KEY_ENV", "MY_CLAUDE_KEY");
    c = rm::config_from_env(&why);
    CHECK(c && c->kind == rm::provider::anthropic && c->model == "claude-opus-5-5" && c->key_env == "MY_CLAUDE_KEY");
    CHECK(c && rm::endpoint(*c) == "https://api.anthropic.com/v1/messages");
    set_env("FORGE_AI_PROVIDER", "openai"); set_env("FORGE_AI_KEY_ENV", nullptr);
    c = rm::config_from_env(&why);
    CHECK(c && c->key_env == "OPENAI_API_KEY" && c->max_tokens_field == "max_completion_tokens");
    set_env("FORGE_AI_PROVIDER", "forge-serve"); set_env("FORGE_AI_GRAMMAR", "0"); set_env("FORGE_AI_TIMEOUT_MS", "5000");
    c = rm::config_from_env(&why);
    CHECK(c && !c->send_grammar && c->timeout_ms == 5000);
    set_env("FORGE_AI_PROVIDER", "command");
    CHECK(!rm::config_from_env(&why) && has(why, "FORGE_AI_CMD"));
    set_env("FORGE_AI_PROVIDER", "nonsense");
    CHECK(!rm::config_from_env(&why) && has(why, "not a provider"));
    clear_ai_env();
    set_env("FORGE_AI_CMD", "cat");   // a command alone selects the command provider
    c = rm::config_from_env(&why);
    CHECK(c && c->kind == rm::provider::command && c->command == "cat");
    clear_ai_env();
}

TEST(openai_request_body) {
    auto c = *rm::provider_defaults("llamacpp");
    auto r = chat("be brief", "hi"); r.temperature = 0.0; r.grammar = "root ::= \"a\"";
    auto b = json::parse(rm::request_body(c, r));
    CHECK(b["model"].str() == "default" && b["max_tokens"].num() == 32 && b["temperature"].num() == 0 && !b["stream"].b());
    CHECK(b["messages"].arr().size() == 2 && b["messages"][0]["role"].str() == "system" && b["messages"][1]["content"].str() == "hi");
    CHECK(b["grammar"].str() == "root ::= \"a\"" && !b.find("response_format"));
    // without send_grammar the grammar stays home; a schema becomes response_format json_schema
    c.send_grammar = false; r.json_schema = json::parse(R"({"type":"object"})");
    b = json::parse(rm::request_body(c, r));
    CHECK(!b.find("grammar") && b["response_format"]["type"].str() == "json_schema" && b["response_format"]["json_schema"]["schema"]["type"].str() == "object");
    r.json_schema = json::value(); r.json = true;
    b = json::parse(rm::request_body(c, r));
    CHECK(b["response_format"]["type"].str() == "json_object");
    auto o = *rm::provider_defaults("openai"); o.model = "gpt-x";
    b = json::parse(rm::request_body(o, chat("s", "u")));
    CHECK(b["max_completion_tokens"].num() == 32 && !b.find("max_tokens") && !b.find("temperature"));
    auto h = rm::http_request(o, chat("s", "u"), "sk-test");
    CHECK(h.method == "POST" && h.url == "https://api.openai.com/v1/chat/completions" && h.headers["Authorization"] == "Bearer sk-test");
    CHECK(!has(h.body, "sk-test") && !has(h.url, "sk-test"));
}

TEST(anthropic_request_body) {
    auto c = *rm::provider_defaults("anthropic");
    rm::request r; r.messages = {{"system", "rule one"}, {"system", "rule two"}, {"user", "q"}, {"assistant", "a"}, {"user", "q2"}};
    r.max_tokens = 100; r.temperature = 0.5; r.grammar = "root ::= \"x\""; r.json_schema = json::parse(R"({"type":"object","properties":{"ok":{"type":"boolean"}}})");
    auto b = json::parse(rm::request_body(c, r));
    CHECK(b["model"].str() == "claude-opus-5-5" && b["max_tokens"].num() == 100);
    CHECK(b["system"].str() == "rule one\n\nrule two");
    CHECK(b["messages"].arr().size() == 3 && b["messages"][1]["role"].str() == "assistant" && b["messages"][2]["content"].str() == "q2");
    // the 2026-10-21 API rejects sampling parameters; grammars are not an Anthropic feature
    CHECK(!b.find("temperature") && !b.find("top_p") && !b.find("top_k") && !b.find("grammar"));
    CHECK(b["output_config"]["format"]["type"].str() == "json_schema" && b["output_config"]["format"]["schema"]["properties"].find("ok"));
    CHECK(b["fallbacks"].str() == "default");
    auto h = rm::http_request(c, r, "sk-ant-test");
    CHECK(h.headers["x-api-key"] == "sk-ant-test" && h.headers["anthropic-version"] == "2026-10-21" && !h.headers.count("Authorization"));
    CHECK(h.headers["anthropic-beta"] == "server-side-fallback-2026-07-01");
    c.anthropic_fallbacks = false;
    CHECK(!json::parse(rm::request_body(c, r)).find("fallbacks") && !rm::http_request(c, r, "").headers.count("anthropic-beta"));
}

TEST(openai_reply_parsing) {
    auto c = *rm::provider_defaults("vllm");
    auto r = rm::parse_reply(c, 200, R"({"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"length"}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}})");
    CHECK(r.ok() && r.text == "hello" && r.finish_reason == "length" && r.tokens.prompt_tokens == 7 && r.tokens.completion_tokens == 3);
    r = rm::parse_reply(c, 200, R"({"choices":[{"message":{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}}]})");
    CHECK(r.ok() && r.text == "ab");
    r = rm::parse_reply(c, 200, R"({"choices":[{"text":"plain","finish_reason":"stop"}]})");
    CHECK(r.ok() && r.text == "plain");
    r = rm::parse_reply(c, 200, R"({"choices":[{"message":{"content":null,"refusal":"I can't help with that."}}]})");
    CHECK(!r.ok() && has(r.error, "declined"));
    r = rm::parse_reply(c, 400, R"({"error":{"message":"bad model","type":"invalid_request_error"}})");
    CHECK(!r.ok() && r.http_status == 400 && has(r.error, "bad model"));
    r = rm::parse_reply(c, 502, "<html>gateway</html>");
    CHECK(!r.ok() && has(r.error, "502"));
    r = rm::parse_reply(c, 200, R"({"choices":[]})");
    CHECK(!r.ok() && has(r.error, "no choices"));
}

TEST(anthropic_reply_parsing) {
    auto c = *rm::provider_defaults("anthropic");
    auto r = rm::parse_reply(c, 200, R"({"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5",
        "content":[{"type":"thinking","thinking":"","signature":"s"},{"type":"text","text":"Hello"},{"type":"tool_use","id":"t","name":"x","input":{}},{"type":"text","text":", world"}],
        "stop_reason":"end_turn","stop_details":null,"usage":{"input_tokens":12,"output_tokens":5}})");
    CHECK(r.ok() && r.text == "Hello, world" && r.finish_reason == "stop" && r.tokens.prompt_tokens == 12 && r.tokens.completion_tokens == 5);
    r = rm::parse_reply(c, 200, R"({"content":[{"type":"text","text":"cut"}],"stop_reason":"max_tokens","usage":{"input_tokens":1,"output_tokens":9}})");
    CHECK(r.ok() && r.finish_reason == "length");
    r = rm::parse_reply(c, 200, R"({"content":[{"type":"text","text":"partial"}],"stop_reason":"refusal","stop_details":{"type":"refusal","category":"cyber","explanation":"..."}})");
    CHECK(!r.ok() && r.text.empty() && has(r.error, "refusal") && has(r.error, "cyber"));
    r = rm::parse_reply(c, 401, R"({"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}})");
    CHECK(!r.ok() && has(r.error, "401") && has(r.error, "invalid x-api-key"));
    r = rm::parse_reply(c, 529, R"({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}})");
    CHECK(!r.ok() && has(r.error, "Overloaded"));
}

TEST(complete_keeps_the_key_secret) {
    clear_ai_env();
    auto c = *rm::provider_defaults("anthropic"); c.key_env = "FORGE_TEST_REMOTE_KEY";
    // the key variable unset: an error naming the variable, no request
    bool sent = false;
    auto r = rm::complete(c, chat("s", "u"), [&](const os::net::http_request&) { sent = true; return os::result<os::net::http_response>{}; });
    CHECK(!r.ok() && !sent && has(r.error, "FORGE_TEST_REMOTE_KEY"));
    const std::string key = "sk-ant-secret-1234567890";
    set_env("FORGE_TEST_REMOTE_KEY", key.c_str());
    os::net::http_request seen;
    r = rm::complete(c, chat("s", "u"), [&](const os::net::http_request& h) {
        seen = h; os::result<os::net::http_response> out;
        out.value.status = 401; out.value.body = R"({"type":"error","error":{"type":"authentication_error","message":"bad key )" + key + R"("}})";
        return out;
    });
    CHECK(seen.headers["x-api-key"] == key && !has(seen.body, key) && !has(seen.url, key));
    CHECK(!r.ok() && !has(r.error, key) && has(r.error, "***"));   // a server that echoes the key does not leak it
    // a transport failure carries no key either
    r = rm::complete(c, chat("s", "u"), [&](const os::net::http_request&) { os::result<os::net::http_response> out; out.err = {111, "connect: refused " + key}; return out; });
    CHECK(!r.ok() && !has(r.error, key) && has(r.error, "refused"));
    // a local server with no key variable: no credential header at all
    auto l = *rm::provider_defaults("lmstudio");
    r = rm::complete(l, chat("s", "u"), [&](const os::net::http_request& h) {
        seen = h; os::result<os::net::http_response> out; out.value.status = 200;
        out.value.body = R"({"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]})"; return out;
    });
    CHECK(r.ok() && r.text == "ok" && !seen.headers.count("Authorization") && seen.url == "http://127.0.0.1:1234/v1/chat/completions");
    set_env("FORGE_TEST_REMOTE_KEY", nullptr);
}

TEST(command_provider) {
    rm::request r = chat("sys", "hello"); r.json = true;
    CHECK(rm::command_prompt(r) == "system: sys\nuser: hello\nassistant:");
    rm::request one; one.messages = {{"user", "just this"}};
    CHECK(rm::command_prompt(one) == "just this");
    rm::config c = *rm::provider_defaults("command");
#if !defined(_WIN32)
    c.command = "printf '%s|' \"$FORGE_AI_MAX_TOKENS\" \"$FORGE_AI_JSON\"; cat; echo";
    auto out = rm::complete(c, r);
    CHECK_MSG(out.ok() && out.text == "32|1|system: sys\nuser: hello\nassistant:", "got [%s] err [%s]", out.text.c_str(), out.error.c_str());
    c.command = "echo nope >&2; exit 3";
    out = rm::complete(c, r);
    CHECK(!out.ok() && has(out.error, "exit code 3") && has(out.error, "nope"));
    c.command = "sleep 5"; c.timeout_ms = 200;
    out = rm::complete(c, r);
    CHECK(!out.ok() && has(out.error, "timed out"));
#endif
}

// A real exchange with forge-serve (tests/remote_ai_serve_test.py): plain chat, usage, and a GBNF grammar that the
// tiny random model can only satisfy if the server received it.
TEST(serve_loopback) {
    const char* url = std::getenv("FORGE_TEST_SERVE_URL");
    if (!url || !*url) { std::printf("  (skipped: FORGE_TEST_SERVE_URL not set)\n"); return; }
    auto c = *rm::provider_defaults("forge-serve"); c.url = url; c.timeout_ms = 60000;
    auto r = rm::complete(c, chat("be brief", "hi"));
    CHECK_MSG(r.ok() && r.tokens.prompt_tokens > 0 && r.tokens.completion_tokens > 0 && !r.finish_reason.empty(), "err [%s]", r.error.c_str());
    auto g = chat("answer", "yes or no?"); g.grammar = "root ::= \"yes\" | \"no\"";
    r = rm::complete(c, g);
    CHECK_MSG(r.ok() && (r.text == "yes" || r.text == "no"), "got [%s] err [%s]", r.text.c_str(), r.error.c_str());
    auto j = chat("answer", "json"); j.json_schema = json::parse(R"({"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"]})"); j.max_tokens = 64;
    r = rm::complete(c, j);
    CHECK_MSG(r.ok(), "err [%s]", r.error.c_str());
    if (r.ok() && r.finish_reason == "stop") { auto v = json::parse(r.text); CHECK(v.find("a") && v["a"].is_bool()); }
    // the connection refused on a closed port is an error, not an exception
    auto bad = c; bad.url = "http://127.0.0.1:1/v1"; bad.timeout_ms = 2000;
    r = rm::complete(bad, chat("s", "u"));
    CHECK(!r.ok());
}
