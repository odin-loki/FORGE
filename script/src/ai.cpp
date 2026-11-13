// forge::script AI-assisted commands (docs/scripting.md, "AI-assisted commands").
//   forge::ask(prompt[, context])          -> model reply (string)
//   forge::suggest_cmd(task[, opts])       -> a single drafted shell command (string); never runs it
//   forge::explain(text)                   -> plain-English explanation of an error / output
//   forge::ask_json(prompt)                -> reply parsed as JSON into script values
//   forge::ai_run(task[, opts])            -> drafts a command, shows it to confirm(), runs it only on yes
//   forge::ai_eval(task[, opts])           -> same, for drafted *script* code (runs via eval)
// Backend: options::ai if the host set one; else an in-process model when options::ask_model / FORGE_ASK_MODEL names a
// .fpk and the host registered set_local_ai() (the forge CLI does; forge_script itself does not link inference);
// else a remote AI (inference/remote.hpp; docs/inference.md "Remote models"): FORGE_AI_PROVIDER / FORGE_AI_URL (formerly
// FORGE_AI_ENDPOINT) for forge-serve, any OpenAI-compatible server or the Anthropic API, or FORGE_AI_CMD. Confirmation: opts {"confirm", fn} if given, else options::confirm;
// with neither, drafted code never runs ({ran:false}). Endpoint calls need the net capability; running needs proc.
#include "interp.hpp"
#include "shells/shell.hpp"
#include <forge/script/script.hpp>
#include <forge/json/json.hpp>
#include <forge/os/net.hpp>
#include <forge/inference/remote.hpp>
#include <cstdlib>
#include <mutex>

namespace forge::script {
namespace {
using Args = std::vector<Value>;
using Fn = Native::fn_t;
Value strv(std::string s) { return obj(make<String>(std::move(s))); }
Value nat(const std::string& name, int lo, int hi, Fn f) { auto n = make<Native>(name, std::move(f), lo, hi); n->frozen = true; return obj(n); }
const Value* kv(const Args& a, std::size_t i, const std::string& k) {
    if (i >= a.size()) return nullptr; const Value* r = nullptr;
    if (auto* m = as<Map>(a[i])) r = m->find_str(k);
    else if (auto* l = as<List>(a[i])) for (auto& e : l->items) if (auto* p = as<List>(e); p && p->items.size() == 2) if (auto* s = as<String>(p->items[0]); s && s->s == k) r = &p->items[1];
    return r && !is_nil(*r) ? r : nullptr;
}
Value from_json(const json::value& j) {
    switch (j.v.index()) {
    case 0: return Nil{};
    case 1: return j.b();
    case 2: { double d = j.num(); if (d == double(std::int64_t(d)) && d > -9e15 && d < 9e15) return std::int64_t(d); return d; }
    case 3: return strv(j.str());
    case 4: { auto l = make<List>(); for (auto& x : j.arr()) l->push(from_json(x)); return obj(l); }
    default: { auto m = make<Map>(); for (auto& [k, x] : j.obj()) m->set_str(k, from_json(x)); return obj(m); }
    }
}
const char* kShellSys = "You write one shell command for the user's task. Reply with only the command, no prose, no code fences.";
const char* kScriptSys = "You write a FORGE script (C++23-style syntax, safe subset: no raw pointers/new/delete) for the user's task. Reply with only the code, no prose, no code fences.";

std::mutex local_mu;
local_ai_fn& local_slot() { static local_ai_fn f; return f; }
local_ai_fn local_get() { std::lock_guard lk(local_mu); return local_slot(); }

std::string complete(Interp& in, const std::string& system, const std::string& prompt, bool want_json = false) {
    if (in.sh->ai) {
        try { return in.sh->ai(system, prompt); }
        catch (const ScriptError&) { throw; }
        catch (const std::exception& e) { in.raise("AIError", std::string("ai backend: ") + e.what()); }
    }
    std::string fpk = in.sh->ask_model;
    if (fpk.empty()) if (const char* m = std::getenv("FORGE_ASK_MODEL")) fpk = m;
    if (!fpk.empty()) {
        if (auto gen = local_get()) {
            local_ai_request rq{fpk, system, prompt, want_json, in.sh->ask_max_tokens};
            if (rq.max_tokens <= 0) if (const char* m = std::getenv("FORGE_ASK_MAX_TOKENS")) rq.max_tokens = std::atoi(m);
            if (rq.max_tokens <= 0) rq.max_tokens = 256;
            try { return gen(rq); }
            catch (const ScriptError&) { throw; }
            catch (const std::exception& e) { in.raise("AIError", "local model " + fpk + ": " + e.what()); }
        }
        std::string why_cfg;
        if (!inference::remote::config_from_env(&why_cfg) && why_cfg.empty())
            in.raise("AIError", "FORGE_ASK_MODEL is set, but this host has no in-process inference (run under `forge script`) and no remote AI is configured (FORGE_AI_URL / FORGE_AI_PROVIDER)");
    }
    // Remote AI (inference/remote.hpp): FORGE_AI_PROVIDER / FORGE_AI_URL (or FORGE_AI_ENDPOINT) / FORGE_AI_CMD.
    std::string why;
    auto cfg = inference::remote::config_from_env(&why);
    if (!cfg) {
        if (!why.empty()) in.raise("AIError", why);
        in.raise("AIError", "no AI backend: set FORGE_ASK_MODEL (a local .fpk), FORGE_AI_URL (e.g. http://127.0.0.1:8080 from forge-serve), FORGE_AI_PROVIDER (openai, anthropic, ollama, command, ...) or options::ai");
    }
    if (cfg->kind == inference::remote::provider::command) {
        if (!in.caps.proc) in.raise("PermissionError", "the AI command provider needs process access, which is disabled in this sandbox");
        enforce_scope(in, SecScope::exec, "/bin/sh");
        in.charge_process("ai command");
    } else {
        if (!in.caps.net) in.raise("PermissionError", "AI endpoint access needs network access, which is disabled in this sandbox");
        if (auto u = os::net::parse_url(inference::remote::endpoint(*cfg))) enforce_scope(in, SecScope::net_connect, (u->host.find(':') != std::string::npos ? "[" + u->host + "]" : u->host) + ":" + std::to_string(u->port ? u->port : u->scheme == "https" ? 443 : 80));
    }
    inference::remote::request rq;
    rq.messages = {{"system", system}, {"user", prompt}};
    rq.temperature = 0.0; rq.json = want_json;
    rq.max_tokens = in.sh->ask_max_tokens;
    if (rq.max_tokens <= 0) if (const char* m = std::getenv("FORGE_ASK_MAX_TOKENS")) rq.max_tokens = std::atoi(m);
    if (rq.max_tokens <= 0) rq.max_tokens = 256;
    auto r = inference::remote::complete(*cfg, rq);
    if (!r.ok()) in.raise("AIError", "ai " + cfg->name + ": " + r.error);
    return r.text;
}
// Strip ``` fences / surrounding blank lines a model may add anyway.
std::string clean_code(std::string s) {
    auto b = s.find("```");
    if (b != std::string::npos) { auto nl = s.find('\n', b); auto e = s.find("```", nl == std::string::npos ? b + 3 : nl); if (nl != std::string::npos) s = s.substr(nl + 1, (e == std::string::npos ? s.size() : e) - nl - 1); }
    auto ws = [](char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; };
    while (!s.empty() && ws(s.back())) s.pop_back();
    std::size_t i = 0; while (i < s.size() && ws(s[i])) ++i;
    return s.substr(i);
}
std::string with_ctx(Interp& in, const Args& a, std::string p) {
    if (auto* c = kv(a, 1, "context")) p += "\n\nContext:\n" + in.str(*c);
    else if (a.size() > 1 && as<String>(a[1])) p += "\n\nContext:\n" + in.str(a[1]);
    return p;
}
bool confirmed(Interp& in, const Args& a, const std::string& code) {
    if (auto* f = kv(a, 1, "confirm")) { Args x{strv(code)}; return Interp::truthy(in.call(*f, x)); }
    if (in.sh->confirm) return in.sh->confirm(code);
    return false;
}
std::string brace_escape(const std::string& s) { std::string o; for (char c : s) { o += c; if (c == '{' || c == '}') o += c; } return o; }

void install(Shared& sh) {
    auto& B = sh.builtins;
    B["forge::ask"] = nat("forge::ask", 1, 2, [](Interp& in, Args& a) -> Value {
        return strv(complete(in, "You are a concise assistant inside a FORGE script.", with_ctx(in, a, in.str(a[0]))));
    });
    B["forge::explain"] = nat("forge::explain", 1, 1, [](Interp& in, Args& a) -> Value {
        return strv(complete(in, "Explain briefly, in plain English, what this error or output means and how to fix it.", in.str(a[0])));
    });
    B["forge::suggest_cmd"] = nat("forge::suggest_cmd", 1, 2, [](Interp& in, Args& a) -> Value {
        std::string shell = "sh"; if (auto* v = kv(a, 1, "shell")) shell = in.str(*v);
        return strv(clean_code(complete(in, std::string(kShellSys) + " Target shell: " + shell + ".", with_ctx(in, a, in.str(a[0])))));
    });
    B["forge::ask_json"] = nat("forge::ask_json", 1, 2, [](Interp& in, Args& a) -> Value {
        auto txt = clean_code(complete(in, "Reply with only valid JSON, no prose.", with_ctx(in, a, in.str(a[0])), true));
        try { return from_json(json::parse(txt)); } catch (const std::exception& e) { in.raise("AIError", std::string("model reply is not valid JSON: ") + e.what()); }
    });
    B["forge::ai_run"] = nat("forge::ai_run", 1, 2, [](Interp& in, Args& a) -> Value {
        std::string shell = "sh"; if (auto* v = kv(a, 1, "shell")) shell = in.str(*v);
        if (!shells().find(shell)) in.raise("ValueError", "forge::ai_run: unknown shell '" + shell + "' (see forge::shells())");
        auto cmd = clean_code(complete(in, std::string(kShellSys) + " Target shell: " + shell + ".", with_ctx(in, a, in.str(a[0]))));
        auto m = make<Map>(); m->set_str("command", strv(cmd)); m->set_str("ran", false);
        if (cmd.empty() || !confirmed(in, a, cmd)) return obj(m);
        auto fn = in.sh->builtins.find("forge::" + shell);
        if (fn == in.sh->builtins.end()) in.raise("NameError", "forge::" + shell + " is not available");
        Args x{strv(brace_escape(cmd))};
        m->set_str("result", in.call(fn->second, x)); m->set_str("ran", true);
        return obj(m);
    });
    B["forge::ai_eval"] = nat("forge::ai_eval", 1, 2, [](Interp& in, Args& a) -> Value {
        auto code = clean_code(complete(in, kScriptSys, with_ctx(in, a, in.str(a[0]))));
        auto m = make<Map>(); m->set_str("code", strv(code)); m->set_str("ran", false);
        if (code.empty() || !confirmed(in, a, code)) return obj(m);
        Args x{strv(code)};
        m->set_str("result", in.call(in.sh->builtins.at("eval"), x)); m->set_str("ran", true);
        return obj(m);
    });
}
} // namespace
void install_ai(Shared& sh) { install(sh); }
void set_local_ai(local_ai_fn fn) { std::lock_guard lk(local_mu); local_slot() = std::move(fn); }
bool has_local_ai() { std::lock_guard lk(local_mu); return bool(local_slot()); }
} // namespace forge::script
