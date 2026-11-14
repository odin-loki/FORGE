#pragma once
// Remote AI client: a chat completion on an AI server (or any command), with the same shape as a local .fpk chat
// (messages in; text, token usage and a finish reason out), so a program on FORGE can use either. Guide:
// docs/inference.md, "Remote models". Header-only; it needs only forge::json and forge::os (net, process), not the
// rest of inference, so hosts that do not link a model (forge_script) can include it.
//
// Providers
//   openai     POST {url}/chat/completions: OpenAI, vLLM, llama.cpp server, Ollama (/v1), LM Studio, OpenRouter,
//              Together, ... and FORGE's own forge-serve. Aliases set a default URL (and key variable):
//              forge-serve, ollama, vllm, llamacpp, lmstudio, openrouter.
//   anthropic  POST {url}/v1/messages, the Claude Messages API (anthropic-version 2026-10-21: no temperature /
//              top_p / top_k; only "text" content blocks are read; stop_reason "refusal" is an error).
//   command    any program: the conversation on stdin, the reply on stdout (run by /bin/sh -c, or cmd.exe /c).
//
// Secrets: the API key is read from the environment variable NAMED by config::key_env (FORGE_AI_KEY_ENV) when the
// request is sent. It goes only into the HTTP header (never into argv, a URL, a log line or an error message: a
// server reply quoted in an error has the key replaced by "***"). The HTTPS client verifies the certificate chain
// and the host name, and does not forward credentials across origins on a redirect (os/net.hpp).
//
// Model output from any provider is unverified text: callers that act on it check it themselves.
#include <forge/json/json.hpp>
#include <forge/os/net.hpp>
#include <forge/os/process.hpp>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace forge::inference::remote {

enum class provider { openai, anthropic, command };

inline const char* provider_name(provider p) { return p == provider::openai ? "openai" : p == provider::anthropic ? "anthropic" : "command"; }

struct config {
    provider kind = provider::openai;
    std::string name = "openai";      // as configured (an alias such as "ollama" keeps its name here)
    std::string url;                  // base URL: openai ".../v1" (or the full .../chat/completions), anthropic "https://api.anthropic.com"
    std::string model;                // "" = "default" (openai) / claude-opus-5-5 (anthropic)
    std::string key_env;              // NAME of the environment variable holding the API key; "" = no key
    std::string command;              // provider::command: the command line
    bool send_grammar = false;        // openai: send a request's GBNF grammar ("grammar": llama.cpp server, forge-serve)
    std::string max_tokens_field = "max_tokens";   // openai: "max_completion_tokens" for api.openai.com
    int timeout_ms = 180000;          // the whole exchange (connect + generation)
    std::string ca_file;              // https: trust these CA certificates instead of the system store
    // anthropic: the version header, and server-side refusal fallbacks (fallbacks "default", beta
    // server-side-fallback-2026-07-01): a request a model declines is re-run on a fallback model in the same call.
    std::string anthropic_version = "2026-10-21";
    bool anthropic_fallbacks = true;
};

struct message { std::string role, content; };   // role: system | user | assistant

struct request {
    std::vector<message> messages;
    int max_tokens = 256;
    std::optional<double> temperature;   // not sent to anthropic (the 2026-10-21 API rejects sampling parameters)
    std::string grammar;                 // GBNF; sent only when config::send_grammar (openai) or as FORGE_AI_GRAMMAR (command)
    json::value json_schema;             // non-null: the reply must be a JSON value matching this schema
    bool json = false;                   // the reply must be a JSON value (json_object); implied by json_schema
    int timeout_ms = 0;                  // 0 = config::timeout_ms
};

struct usage { std::size_t prompt_tokens = 0, completion_tokens = 0; };

struct reply {
    std::string text;
    usage tokens;
    std::string finish_reason;   // stop | length | ... ("" when the server gave none)
    std::string error;           // "" = success
    int http_status = 0;
    bool ok() const { return error.empty(); }
};

namespace detail_remote {
inline std::string env(const char* k) { const char* v = std::getenv(k); return v ? v : ""; }
inline std::string env(const std::string& k) { return k.empty() ? std::string() : env(k.c_str()); }
inline bool truthy(const std::string& v) { return v == "1" || v == "true" || v == "yes" || v == "on"; }
inline std::string lower(std::string s) { for (auto& c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a'); return s; }
inline std::string trim_slash(std::string s) { while (!s.empty() && s.back() == '/') s.pop_back(); return s; }
// Never show the key, even when a server quotes the request back in an error.
inline std::string scrub(std::string s, const std::string& key) {
    if (key.size() < 4) return s;
    for (std::size_t p; (p = s.find(key)) != std::string::npos;) s.replace(p, key.size(), "***");
    return s;
}
inline std::string snippet(const std::string& s, std::size_t n = 300) { return s.size() <= n ? s : s.substr(0, n) + "..."; }
inline const json::value* path(const json::value& j, std::initializer_list<std::string_view> ks) {
    const json::value* p = &j;
    for (auto k : ks) { p = p->find(k); if (!p) return nullptr; }
    return p;
}
inline std::size_t count(const json::value& j, std::string_view k) { auto* p = j.find(k); return p && p->is_num() && p->num() > 0 ? std::size_t(p->num()) : 0; }
}   // namespace detail_remote

// A provider by name with its defaults. nullopt (and *why) for an unknown name.
inline std::optional<config> provider_defaults(const std::string& name_in, std::string* why = nullptr) {
    config c; std::string n = detail_remote::lower(name_in); c.name = n;
    if (n.empty() || n == "openai") { c.name = "openai"; c.url = "https://api.openai.com/v1"; c.key_env = "OPENAI_API_KEY"; c.max_tokens_field = "max_completion_tokens"; }
    else if (n == "forge-serve" || n == "forge") { c.name = "forge-serve"; c.url = "http://127.0.0.1:8080/v1"; c.send_grammar = true; }
    else if (n == "ollama") c.url = "http://127.0.0.1:11434/v1";
    else if (n == "vllm") c.url = "http://127.0.0.1:8000/v1";
    else if (n == "llamacpp" || n == "llama.cpp" || n == "llama-server") { c.name = "llamacpp"; c.url = "http://127.0.0.1:8080/v1"; c.send_grammar = true; }
    else if (n == "lmstudio") c.url = "http://127.0.0.1:1234/v1";
    else if (n == "openrouter") { c.url = "https://openrouter.ai/api/v1"; c.key_env = "OPENROUTER_API_KEY"; }
    else if (n == "anthropic" || n == "claude") { c.name = "anthropic"; c.kind = provider::anthropic; c.url = "https://api.anthropic.com"; c.key_env = "ANTHROPIC_API_KEY"; c.model = "claude-opus-5-5"; }
    else if (n == "command" || n == "cmd") { c.name = "command"; c.kind = provider::command; }
    else {
        if (why) *why = "FORGE_AI_PROVIDER=" + name_in + " is not a provider (openai, forge-serve, ollama, vllm, llamacpp, lmstudio, openrouter, anthropic, command)";
        return std::nullopt;
    }
    return c;
}

// From the environment:
//   FORGE_AI_PROVIDER   a provider name (above); default openai-compatible when only a URL is set
//   FORGE_AI_URL        base URL (FORGE_AI_ENDPOINT is the older name of the same setting)
//   FORGE_AI_MODEL      model name
//   FORGE_AI_KEY_ENV    NAME of the variable holding the key ("" or "none": send no key); FORGE_AI_KEY (the key
//                       itself) is still honoured for an endpoint without FORGE_AI_KEY_ENV
//   FORGE_AI_CMD        provider command: the command line
//   FORGE_AI_GRAMMAR    1/0: send GBNF grammars (openai-compatible servers that accept "grammar")
//   FORGE_AI_TIMEOUT_MS whole-request timeout (default 180000)
//   FORGE_AI_FALLBACKS  0: no server-side refusal fallbacks (anthropic)
// nullopt with *why = "" when nothing is configured; nullopt with a reason when the settings are wrong.
inline std::optional<config> config_from_env(std::string* why = nullptr) {
    using detail_remote::env;
    if (why) why->clear();
    std::string prov = env("FORGE_AI_PROVIDER"), url = env("FORGE_AI_URL");
    if (url.empty()) url = env("FORGE_AI_ENDPOINT");
    if (prov.empty() && url.empty() && env("FORGE_AI_CMD").empty()) return std::nullopt;
    if (prov.empty()) prov = url.empty() ? "command" : "openai";
    auto c = provider_defaults(prov, why);
    if (!c) return std::nullopt;
    if (!url.empty()) {
        // An explicit URL for the "openai" kind is any compatible server: its key variable is not OpenAI's unless named.
        if (c->name == "openai" && detail_remote::lower(url).find("api.openai.com") == std::string::npos) { c->key_env.clear(); c->max_tokens_field = "max_tokens"; }
        c->url = url;
    }
    if (auto m = env("FORGE_AI_MODEL"); !m.empty()) c->model = m;
    if (const char* k = std::getenv("FORGE_AI_KEY_ENV")) { std::string v = k; c->key_env = v == "none" ? "" : v; }
    else if (std::getenv("FORGE_AI_KEY") && c->kind != provider::command) c->key_env = "FORGE_AI_KEY";
    if (auto g = env("FORGE_AI_GRAMMAR"); !g.empty()) c->send_grammar = detail_remote::truthy(g);
    if (auto t = env("FORGE_AI_TIMEOUT_MS"); !t.empty()) { int v = std::atoi(t.c_str()); if (v > 0) c->timeout_ms = v; }
    if (auto f = env("FORGE_AI_FALLBACKS"); !f.empty()) c->anthropic_fallbacks = detail_remote::truthy(f);
    if (c->kind == provider::command) {
        c->command = env("FORGE_AI_CMD");
        if (c->command.empty()) { if (why) *why = "FORGE_AI_PROVIDER=command needs FORGE_AI_CMD (the command line)"; return std::nullopt; }
    }
    return c;
}

// The URL a request is sent to.
inline std::string endpoint(const config& c) {
    std::string u = detail_remote::trim_slash(c.url);
    if (c.kind == provider::anthropic) {
        if (u.ends_with("/v1/messages")) return u;
        return u + (u.ends_with("/v1") ? "/messages" : "/v1/messages");
    }
    if (u.ends_with("/chat/completions")) return u;
    return u + (u.ends_with("/v1") ? "/chat/completions" : "/v1/chat/completions");
}

// The JSON body of one request (openai / anthropic).
inline std::string request_body(const config& c, const request& r) {
    using json::array; using json::object; using json::value;
    const bool want_json = r.json || !r.json_schema.is_null();
    if (c.kind == provider::anthropic) {
        std::string system; array msgs;
        for (auto& m : r.messages) {
            if (m.role == "system") { if (!system.empty()) system += "\n\n"; system += m.content; continue; }
            msgs.push_back(value(object{{"role", m.role == "assistant" ? "assistant" : "user"}, {"content", m.content}}));
        }
        object b{{"model", c.model.empty() ? std::string("claude-opus-5-5") : c.model}, {"max_tokens", r.max_tokens > 0 ? r.max_tokens : 256}};
        if (!system.empty()) b.emplace_back("system", system);
        b.emplace_back("messages", value(std::move(msgs)));
        if (!r.json_schema.is_null()) b.emplace_back("output_config", value(object{{"format", value(object{{"type", "json_schema"}, {"schema", r.json_schema}})}}));
        if (c.anthropic_fallbacks) b.emplace_back("fallbacks", "default");
        return json::dump(value(std::move(b)));
    }
    array msgs;
    for (auto& m : r.messages) msgs.push_back(value(object{{"role", m.role}, {"content", m.content}}));
    object b{{"model", c.model.empty() ? std::string("default") : c.model}, {"messages", value(std::move(msgs))},
             {c.max_tokens_field.empty() ? std::string("max_tokens") : c.max_tokens_field, r.max_tokens > 0 ? r.max_tokens : 256}, {"stream", false}};
    if (r.temperature) b.emplace_back("temperature", *r.temperature);
    if (c.send_grammar && !r.grammar.empty()) b.emplace_back("grammar", r.grammar);   // the grammar is the stricter constraint: it wins
    else if (!r.json_schema.is_null()) b.emplace_back("response_format", value(object{{"type", "json_schema"}, {"json_schema", value(object{{"name", "reply"}, {"schema", r.json_schema}})}}));
    else if (want_json) b.emplace_back("response_format", value(object{{"type", "json_object"}}));
    return json::dump(value(std::move(b)));
}

// The HTTP request (body and headers), with `key` as the credential ("" = none).
inline os::net::http_request http_request(const config& c, const request& r, const std::string& key) {
    os::net::http_request h;
    h.method = "POST"; h.url = endpoint(c); h.body = request_body(c, r);
    h.timeout_ms = r.timeout_ms > 0 ? r.timeout_ms : c.timeout_ms; h.deadline_ms = h.timeout_ms;
    h.ca_file = c.ca_file;
    h.headers["Content-Type"] = "application/json";
    h.headers["Accept"] = "application/json";
    if (c.kind == provider::anthropic) {
        h.headers["anthropic-version"] = c.anthropic_version;
        if (c.anthropic_fallbacks) h.headers["anthropic-beta"] = "server-side-fallback-2026-07-01";
        if (!key.empty()) h.headers["x-api-key"] = key;
    } else if (!key.empty()) h.headers["Authorization"] = "Bearer " + key;
    return h;
}

// A provider's answer -> reply. A non-2xx status, a body that is not JSON, an API error object or a refusal set
// reply::error. `key` is scrubbed from anything quoted.
inline reply parse_reply(const config& c, int status, const std::string& body, const std::string& key = {}) {
    reply out; out.http_status = status;
    auto fail = [&](std::string e) { out.error = detail_remote::scrub(std::move(e), key); out.text.clear(); return out; };
    json::value j;
    try { j = json::parse(body); }
    catch (const std::exception&) {
        if (status < 200 || status >= 300) return fail(std::string(provider_name(c.kind)) + ": HTTP " + std::to_string(status) + ": " + detail_remote::snippet(body));
        return fail(std::string(provider_name(c.kind)) + ": the reply is not JSON: " + detail_remote::snippet(body));
    }
    if (status < 200 || status >= 300 || j.find("error")) {
        std::string msg;
        if (auto* e = j.find("error")) msg = e->is_str() ? e->str() : e->str_or("message", json::dump(*e));
        return fail(std::string(provider_name(c.kind)) + ": HTTP " + std::to_string(status) + ": " + detail_remote::snippet(msg.empty() ? body : msg));
    }
    try {
        if (c.kind == provider::anthropic) {
            std::string stop = j.str_or("stop_reason", "");
            if (stop == "refusal") {
                std::string cat; if (auto* d = j.find("stop_details")) cat = d->str_or("category", "");
                return fail("anthropic: the model declined the request (stop_reason refusal" + (cat.empty() ? std::string() : ", category " + cat) + ")");
            }
            if (auto* ct = j.find("content"); ct && ct->is_arr())
                for (auto& b : ct->arr()) if (b.str_or("type", "") == "text") out.text += b.str_or("text", "");
            if (auto* u = j.find("usage")) { out.tokens.prompt_tokens = detail_remote::count(*u, "input_tokens"); out.tokens.completion_tokens = detail_remote::count(*u, "output_tokens"); }
            out.finish_reason = stop == "end_turn" || stop == "stop_sequence" ? "stop" : stop == "max_tokens" ? "length" : stop;
            return out;
        }
        auto* ch = j.find("choices");
        if (!ch || !ch->is_arr() || ch->arr().empty()) return fail("openai: no choices in the reply");
        auto& c0 = ch->arr()[0];
        if (auto* m = c0.find("message")) {
            if (auto* rf = m->find("refusal"); rf && rf->is_str() && !rf->str().empty()) return fail("openai: the model declined the request: " + detail_remote::snippet(rf->str()));
            if (auto* t = m->find("content")) {
                if (t->is_str()) out.text = t->str();
                else if (t->is_arr()) for (auto& p : t->arr()) if (p.str_or("type", "text") == "text") out.text += p.str_or("text", "");
            }
        } else out.text = c0.str_or("text", "");
        out.finish_reason = c0.str_or("finish_reason", "");
        if (auto* u = j.find("usage")) { out.tokens.prompt_tokens = detail_remote::count(*u, "prompt_tokens"); out.tokens.completion_tokens = detail_remote::count(*u, "completion_tokens"); }
        return out;
    } catch (const std::exception& e) { return fail(std::string(provider_name(c.kind)) + ": bad reply: " + e.what()); }
}

// The conversation as the command provider's stdin: one "role: content" block per message, then "assistant:"
// (the format forge-serve and forge::ask use for a model without a chat template). A single user message with no
// system prompt is passed as its bare text.
inline std::string command_prompt(const request& r) {
    if (r.messages.size() == 1 && r.messages[0].role == "user") return r.messages[0].content;
    std::string s;
    for (auto& m : r.messages) s += m.role + ": " + m.content + "\n";
    return s + "assistant:";
}

// The spawn of the command provider: the conversation on stdin; FORGE_AI_MAX_TOKENS, FORGE_AI_JSON (1 when a JSON
// reply is wanted), FORGE_AI_JSON_SCHEMA and FORGE_AI_GRAMMAR in its environment.
inline os::spawn_spec command_spec(const config& c, const request& r) {
    os::spawn_spec sp;
#if defined(FORGE_OS_WINDOWS)
    sp.argv = {"cmd.exe", "/d", "/c", c.command};
#else
    sp.argv = {"/bin/sh", "-c", c.command};
#endif
    sp.stdin_data = command_prompt(r);
    sp.timeout_ms = std::uint64_t(r.timeout_ms > 0 ? r.timeout_ms : c.timeout_ms);
    sp.max_output = 16u << 20;
    sp.env_set["FORGE_AI_MAX_TOKENS"] = std::to_string(r.max_tokens > 0 ? r.max_tokens : 256);
    if (r.json || !r.json_schema.is_null()) sp.env_set["FORGE_AI_JSON"] = "1";
    if (!r.json_schema.is_null()) sp.env_set["FORGE_AI_JSON_SCHEMA"] = json::dump(r.json_schema);
    if (!r.grammar.empty()) sp.env_set["FORGE_AI_GRAMMAR"] = r.grammar;
    return sp;
}

// The HTTP exchange; tests replace it.
using transport = std::function<os::result<os::net::http_response>(const os::net::http_request&)>;

// One chat completion. Never throws for a provider failure: reply::error says what went wrong.
inline reply complete(const config& c, const request& r, const transport& send = {}) {
    reply out;
    if (r.messages.empty()) { out.error = "remote: a request needs at least one message"; return out; }
    if (c.kind == provider::command) {
        if (c.command.empty()) { out.error = "command: no command line (FORGE_AI_CMD)"; return out; }
        auto p = os::spawn_wait(command_spec(c, r));
        if (!p) { out.error = "command: " + p.err.what; return out; }
        if (p->timed_out) { out.error = "command: timed out after " + std::to_string(r.timeout_ms > 0 ? r.timeout_ms : c.timeout_ms) + " ms"; return out; }
        if (p->exit_code != 0) { out.error = "command: exit code " + std::to_string(p->exit_code) + ": " + detail_remote::snippet(p->err); return out; }
        out.text = std::move(p->out);
        while (!out.text.empty() && (out.text.back() == '\n' || out.text.back() == '\r')) out.text.pop_back();
        out.finish_reason = p->truncated ? "length" : "stop";
        return out;
    }
    std::string key;
    if (!c.key_env.empty()) {
        key = detail_remote::env(c.key_env);
        if (key.empty() && (c.kind == provider::anthropic || c.name == "openai" || c.name == "openrouter"))
            { out.error = std::string(c.name) + ": the key variable " + c.key_env + " is not set (FORGE_AI_KEY_ENV names it)"; return out; }
    }
    auto h = http_request(c, r, key);
    auto resp = send ? send(h) : os::net::http(h);
    if (!resp) { out.error = detail_remote::scrub(std::string(provider_name(c.kind)) + ": " + h.url + ": " + resp.err.what, key); return out; }
    return parse_reply(c, resp->status, resp->body, key);
}

// The common case: one system prompt and one user prompt.
inline reply ask(const config& c, const std::string& system, const std::string& prompt, int max_tokens = 256, bool json = false) {
    request r; if (!system.empty()) r.messages.push_back({"system", system});
    r.messages.push_back({"user", prompt}); r.max_tokens = max_tokens; r.json = json; r.temperature = 0.0;
    return complete(c, r);
}

}   // namespace forge::inference::remote
