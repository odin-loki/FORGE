// forge-serve — OpenAI-compatible HTTP server over the continuous-batching scheduler.
//
//   forge-serve <model.fpk|hf_dir|model.gguf> [--host 127.0.0.1] [--port 8080] [--max-active 8] [--ctx N] ...   (flags: docs/serve.md)
//
//   GET  /health                 -> {"status":"ok", ...} (503 "draining" during shutdown); never needs an API key
//   GET  /v1/models[/<id>]       -> the model, and one entry per LoRA adapter ("<model>:<adapter>")
//   POST /v1/completions         {"prompt": str | [ids], "max_tokens", "temperature", "top_p", "top_k", "min_p", "n",
//                                 "repetition_penalty", "seed", "stop": str | [str | id], "grammar": GBNF, "stream": bool,
//                                 "typical_p", "mirostat_tau", "mirostat_eta", "presence_penalty", "frequency_penalty",
//                                 "logit_bias", "logprobs", "priority", "adapter", "response_format": {json_object | json_schema}}
//   POST /v1/chat/completions    {"messages": [{"role","content"}], "tools", "tool_choice", ...same fields}
//   POST /v1/embeddings          {"input": str | [str] | [ids] | [[ids]], "model", "encoding_format": "float" | "base64"}
//                                 (models whose blueprint declares `session embedding`; 400 on a generative model)
//   GET  /metrics                -> scheduler counters
#include <forge/import/gguf.hpp>
#include <forge/inference/json_schema.hpp>
#include <forge/inference/inference.hpp>
#include <forge/inference/chat_tools.hpp>
#include <forge/core/net.hpp>
#include <forge/graph/rpc_secure.hpp>
#include <forge/loadbalancer/proxy.hpp>
#include <forge/os/cpu_topology.hpp>
#include <forge/os/file_watch.hpp>
#include <forge/os/memory_pressure.hpp>
#include <forge/os/numa.hpp>
#include <forge/os/service.hpp>
#include <forge/security/security.hpp>
#include "serve_http.hpp"
#include <signal.h>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <random>

using namespace forge;

namespace {

namespace rpc = forge::graph::rpc;
using serve::http_request;
using clk = std::chrono::steady_clock;

// ---- settings (docs/serve.md "Command line")
struct serve_options {
    std::string cors = "*";                 // Access-Control-Allow-Origin value; "" = no CORS headers
    serve::http_limits limits;              // --max-header-bytes / --max-body-bytes
    int request_timeout_s = 60;             // reading one request, first byte to last
    int keep_alive_s = 30;                  // an idle kept-alive connection
    int send_timeout_s = 30;                // a client that stops reading
    int shutdown_s = 30;                    // drain window after SIGINT / SIGTERM
    std::size_t max_connections = 256;
    std::size_t max_queue = 0;
    int max_n = 8;                          // "n" of one request
    double rate_rpm = 0;                    // --rate-limit: requests per minute per key (0 = none)
    i64 context = 0;
};

// An error answered in OpenAI's shape: {"error": {"message", "type", "param", "code"}}.
struct http_error : std::exception {
    int status; std::string message, param, code, extra_headers;
    http_error(int s, std::string m, std::string p = {}, std::string c = {}, std::string h = {})
        : status(s), message(std::move(m)), param(std::move(p)), code(std::move(c)), extra_headers(std::move(h)) {}
    const char* what() const noexcept override { return message.c_str(); }
};
template <class... A> [[noreturn]] void bad_request(std::string_view param, std::format_string<A...> f, A&&... a) {
    throw http_error(400, std::format(f, std::forward<A>(a)...), std::string(param));
}
const char* status_text(int code) {
    switch (code) {
    case 200: return "OK"; case 204: return "No Content"; case 400: return "Bad Request"; case 401: return "Unauthorized";
    case 403: return "Forbidden"; case 404: return "Not Found"; case 405: return "Method Not Allowed"; case 408: return "Request Timeout";
    case 413: return "Content Too Large"; case 417: return "Expectation Failed"; case 429: return "Too Many Requests";
    case 431: return "Request Header Fields Too Large"; case 501: return "Not Implemented"; case 503: return "Service Unavailable";
    case 505: return "HTTP Version Not Supported"; case 507: return "Insufficient Storage"; default: return "Internal Server Error";
    }
}
const char* error_type(int code) {
    return code == 401 ? "authentication_error" : code == 403 ? "permission_error" : code == 404 ? "not_found_error" : code == 429 ? "rate_limit_error"
         : code == 503 ? "overloaded_error" : code >= 500 ? "server_error" : "invalid_request_error";
}
std::string error_json(int status, const std::string& m, const std::string& param = {}, const std::string& code = {});

// JSON strings must be valid UTF-8: byte-level tokens can split or produce invalid sequences.
std::size_t utf8_len(unsigned char c) { return c < 0x80 ? 1 : c >= 0xC2 && c <= 0xDF ? 2 : c >= 0xE0 && c <= 0xEF ? 3 : c >= 0xF0 && c <= 0xF4 ? 4 : 0; }
// valid range of the second byte for a lead byte (RFC 3629: no overlongs, surrogates or > U+10FFFF)
bool second_ok(unsigned char lead, unsigned char b) {
    if (lead == 0xE0) return b >= 0xA0 && b <= 0xBF;
    if (lead == 0xED) return b >= 0x80 && b <= 0x9F;
    if (lead == 0xF0) return b >= 0x90 && b <= 0xBF;
    if (lead == 0xF4) return b >= 0x80 && b <= 0x8F;
    return (b & 0xC0) == 0x80;
}
std::string sanitize(std::string_view s) {
    std::string o;
    for (std::size_t i = 0; i < s.size();) {
        auto n = utf8_len(static_cast<unsigned char>(s[i]));
        bool ok = n > 0 && i + n <= s.size();
        for (std::size_t k = 1; ok && k < n; ++k)
            ok = k == 1 ? second_ok(static_cast<unsigned char>(s[i]), static_cast<unsigned char>(s[i + 1])) : (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80;
        if (ok) { o.append(s.substr(i, n)); i += n; } else { o += "\xEF\xBF\xBD"; ++i; }
    }
    return o;
}
// length of the prefix of `s` that does not end inside an incomplete (but so far valid) sequence
std::size_t complete_prefix(std::string_view s) {
    for (std::size_t back = 1; back <= 3 && back <= s.size(); ++back) {
        auto c = static_cast<unsigned char>(s[s.size() - back]);
        if ((c & 0xC0) == 0x80) continue;
        auto n = utf8_len(c);
        return n > back ? s.size() - back : s.size();
    }
    return s.size();
}
std::string error_json(int status, const std::string& m, const std::string& param, const std::string& code) {
    return json::dump(json::value(json::object{{"error", json::value(json::object{{"message", sanitize(m)}, {"type", error_type(status)},
        {"param", param.empty() ? json::value(nullptr) : json::value(param)}, {"code", code.empty() ? json::value(nullptr) : json::value(code)}})}}));
}

// ---- API keys: "Authorization: Bearer <key>". Keys are compared as BLAKE2b digests in constant time, every key
// each time, so neither a key's length nor which key came close shows in the timing.
struct api_key {
    std::array<std::uint8_t, 32> digest{}; std::string name;
    double rpm = 0;              // requests per minute (0 = the server's --rate-limit)
    std::uint64_t quota = 0;     // tokens (prompt + completion) this process may spend on the key; 0 = no quota
    double bucket = 0; clk::time_point last{}; bool primed = false; std::uint64_t used = 0;
};
std::array<std::uint8_t, 32> key_digest(std::string_view k) {
    std::array<std::uint8_t, 32> d{};
    crypto_blake2b(d.data(), d.size(), reinterpret_cast<const std::uint8_t*>(k.data()), k.size());
    return d;
}
// A key file: one key per line, "#" comments: [rpm=N] [quota=N] <key> [name]
void parse_key_file(const std::string& path, std::vector<api_key>& out) {
    security::require_private_file("--api-key-file", path, security::file_kind::secret);
    auto r = forge::os::read_file(path);
    if (!r) raise(errc::io_error, "--api-key-file {}: {}", path, r.err.what);
    rpc::sec::for_each_config_line(r.value, [&](std::size_t lineno, const std::vector<std::string_view>& f) {
        api_key k; std::size_t i = 0;
        auto num = [&](std::string_view v) { std::uint64_t n = 0; auto [e, ec] = std::from_chars(v.data(), v.data() + v.size(), n); if (v.empty() || ec != std::errc() || e != v.data() + v.size()) raise(errc::parse_error, "{}:{}: bad number '{}'", path, lineno, v); return n; };
        for (; i < f.size(); ++i) {
            if (f[i].starts_with("rpm=")) k.rpm = double(num(f[i].substr(4)));
            else if (f[i].starts_with("quota=")) k.quota = num(f[i].substr(6));
            else break;
        }
        if (i >= f.size()) raise(errc::parse_error, "{}:{}: options without a key", path, lineno);
        if (f[i].size() < 8) raise(errc::parse_error, "{}:{}: an API key needs at least 8 characters", path, lineno);
        k.digest = key_digest(f[i]);
        k.name = i + 1 < f.size() ? std::string(f[i + 1]) : std::format("key{}", out.size() + 1);
        out.push_back(std::move(k));
    });
    if (!r.value.empty()) crypto_wipe(r.value.data(), r.value.size());
}

// One connection: the stream (plain or TLS) and what the response framing needs to know about the request.
struct conn {
    rpc::channel ch;
    int send_timeout_s = 30;
    bool keep = false;       // keep the connection after this response
    bool http11 = true;
    bool chunked = false;    // a streamed response is being sent with chunked transfer coding
    bool send(std::string_view s) {
        if (ch.tls()) return ch.send_all(s.data(), s.size());   // SSL_write fails once the socket's send timeout passes
        auto until = clk::now() + std::chrono::seconds(send_timeout_s);
        while (!s.empty()) {
            long n = ch.sock().send(s.data(), s.size());
            if (n > 0) { s.remove_prefix(std::size_t(n)); continue; }
            if (n < 0 && forge::os::would_block() && clk::now() < until) { ch.sock().wait_writable(1000); continue; }
            return false;
        }
        return true;
    }
};

// per-request channel between the scheduler thread and the connection thread; `idx` is the choice ("n" > 1)
struct channel {
    struct event { int idx = 0; int tok = -1; bool done = false; std::string reason; std::shared_ptr<inference::token_logprob> lp; };
    std::mutex mu; std::condition_variable cv;
    std::deque<event> q;
    std::unique_ptr<std::atomic<bool>[]> cancelled;
};

struct server_stats {
    std::atomic<std::uint64_t> connections{0}, rejected{0}, refused_connections{0}, auth_failures{0}, rate_limited{0}, prompt_tokens{0}, completion_tokens{0},
        embedding_inputs{0}, request_errors{0}, tool_calls{0};
    std::atomic<std::int64_t> active{0};
    std::mutex mu; std::map<int, std::uint64_t> by_status;
    void count(int status) { std::lock_guard lk(mu); ++by_status[status]; }
};

class server {
public:
    server(std::shared_ptr<inference::model> m, inference::batch_options bo, std::string name, serve_options so, std::vector<api_key> keys)
        : m_(std::move(m)), sched_(m_, bo), name_(std::move(name)), opt_(std::move(so)), keys_(std::move(keys)), created_(double(std::time(nullptr))) {
        if (!embedding_model()) {
            ends_chat_ = inference::end_of_turn_tokens(*m_, true);
            ends_text_ = inference::end_of_turn_tokens(*m_, false);
            tool_format_ = inference::detect_tool_format(m_->chat_template());
        }
        if (opt_.rate_rpm > 0 && keys_.empty()) anon_.name = "anonymous";
        sched_.start();
    }
    ~server() { sched_.stop(); }
    void evict_kv(std::size_t keep) { sched_.request_kv_eviction(keep); }
    server_stats& stats() { return st_; }
    void set_draining() { draining_ = true; }
    bool auth_enabled() const { return !keys_.empty(); }

    // One connection, any number of requests (keep-alive, pipelining). Nothing thrown leaves this function.
    void handle(conn& c) noexcept {
        try { serve_connection(c); }
        catch (const std::exception& e) { log::warn("forge-serve: connection ended by an error: {}", e.what()); }
        catch (...) {}
        c.ch.close();
    }
private:
    void serve_connection(conn& c) {
        // Deadlines over a 1 s receive tick: an idle connection is closed after keep_alive_s (at once while draining),
        // a request that started must arrive within request_timeout_s.
        bool started = false, timed_out = false; clk::time_point t0 = clk::now();
        auto recv = [&](char* b, std::size_t n) -> long {
            for (;;) {
                long r = c.ch.recv_some(b, n);
                if (r > 0) { if (!started) { started = true; t0 = clk::now(); } return r; }
                if (r != -2) return r;
                auto waited = std::chrono::duration_cast<std::chrono::seconds>(clk::now() - t0).count();
                if (!started && (draining_ || waited >= opt_.keep_alive_s)) return 0;
                if (started && waited >= opt_.request_timeout_s) { timed_out = true; return -1; }
            }
        };
        serve::request_reader rd(recv, opt_.limits);
        for (;;) {
            started = rd.buffered(); timed_out = false; t0 = clk::now();
            http_request r;
            auto st = rd.read(r, [&] { return c.send("HTTP/1.1 100 Continue\r\n\r\n"); });
            if (st == serve::read_status::closed) return;
            c.http11 = r.version != "HTTP/1.0"; c.keep = false;
            if (st != serve::read_status::ok) {
                if (st == serve::read_status::truncated && !timed_out) return;   // the peer went away mid-request
                int code = serve::status_code(st);
                reply(c, code, error_json(code, st == serve::read_status::truncated ? std::format("the request did not arrive within {} s", opt_.request_timeout_s)
                    : st == serve::read_status::header_too_large ? std::format("request headers exceed {} bytes", opt_.limits.max_header)
                    : st == serve::read_status::body_too_large ? std::format("request body exceeds {} bytes", opt_.limits.max_body)
                    : st == serve::read_status::not_implemented ? std::string("only the chunked transfer coding is supported")
                    : st == serve::read_status::bad_version ? std::string("HTTP/1.0 and HTTP/1.1 only")
                    : st == serve::read_status::expectation_failed ? std::string("only \"Expect: 100-continue\" is supported")
                    : std::string("malformed HTTP request")));
                return;   // the stream position is unknown after a refused request
            }
            c.keep = r.keep_alive && !draining_;
            dispatch(c, r);
            if (!c.keep) return;
        }
    }
    void dispatch(conn& c, const http_request& r) {
        try { route(c, r); return; }
        catch (const http_error& e) { ++st_.request_errors; reply(c, e.status, error_json(e.status, e.message, e.param, e.code), "application/json", e.extra_headers); }
        catch (const forge_exception& e) {
            ++st_.request_errors;
            // what a request can get wrong is 400; running out of memory 503; anything else is the server's fault
            int code = e.code == errc::out_of_memory ? 503 : e.code == errc::invalid_argument || e.code == errc::parse_error || e.code == errc::unsupported ||
                       e.code == errc::not_found || e.code == errc::format_error ? 400 : 500;
            reply(c, code, error_json(code, e.what()), "application/json", code == 503 ? "Retry-After: 1\r\n" : "");
        }
        catch (const std::bad_alloc&) { ++st_.request_errors; reply(c, 503, error_json(503, "out of memory"), "application/json", "Retry-After: 1\r\n"); }
        catch (const std::exception& e) { ++st_.request_errors; reply(c, 500, error_json(500, e.what())); }
        if (c.chunked) c.keep = false;   // an error inside a streamed response: the framing cannot be finished cleanly
    }
    std::string cors_headers() const {
        if (opt_.cors.empty()) return {};
        return "Access-Control-Allow-Origin: " + opt_.cors + "\r\n" + (opt_.cors == "*" ? "" : "Vary: Origin\r\n");
    }
    bool reply(conn& c, int code, const std::string& body, const char* type = "application/json", const std::string& extra = {}) {
        st_.count(code);
        if (c.chunked) {   // headers already went out: the error becomes the stream's last event
            c.keep = false;
            return c.send(chunk("data: " + body + "\n\ndata: [DONE]\n\n") + "0\r\n\r\n");
        }
        if (code >= 400 && code != 404 && code != 405 && code != 401 && code != 429 && code != 400) c.keep = false;
        return c.send(std::format("HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\n{}{}\r\n", code, status_text(code), type, body.size(),
                                  c.keep ? "keep-alive" : "close", extra, cors_headers()) + body);
    }
    static std::string chunk(std::string_view data) { return std::format("{:x}\r\n", data.size()) + std::string(data) + "\r\n"; }
    // Server-Sent Events: chunked on HTTP/1.1 (the connection stays usable), until-close on HTTP/1.0.
    bool begin_stream(conn& c) {
        st_.count(200);
        if (!c.http11) c.keep = false;
        c.chunked = c.http11;
        return c.send(std::format("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: {}\r\n{}{}\r\n",
                                  c.keep ? "keep-alive" : "close", c.chunked ? "Transfer-Encoding: chunked\r\n" : "", cors_headers()));
    }
    bool stream_event(conn& c, const std::string& data) { std::string ev = "data: " + data + "\n\n"; return c.send(c.chunked ? chunk(ev) : ev); }
    bool end_stream(conn& c) { bool ok = stream_event(c, "[DONE]") && (!c.chunked || c.send("0\r\n\r\n")); c.chunked = false; return ok; }

    // ---- authentication, rate limits, quotas
    api_key* authenticate(const http_request& r) {
        if (keys_.empty()) return opt_.rate_rpm > 0 ? &anon_ : nullptr;
        std::string_view presented;
        if (auto it = r.headers.find("authorization"); it != r.headers.end()) {
            std::string_view v = it->second;
            if (v.size() > 7 && serve::detail_http::ieq(v.substr(0, 7), "bearer ")) presented = serve::detail_http::trim(v.substr(7));
        }
        auto d = key_digest(presented);
        api_key* hit = nullptr;
        for (auto& k : keys_) if (rpc::sec::ct_equal_bytes(k.digest.data(), d.data(), d.size())) hit = &k;   // no early exit
        if (!hit || presented.empty()) {
            ++st_.auth_failures;
            throw http_error(401, presented.empty() ? "missing API key: send \"Authorization: Bearer <key>\"" : "incorrect API key", {}, "invalid_api_key", "WWW-Authenticate: Bearer\r\n");
        }
        return hit;
    }
    void admit_key(api_key* k) {
        if (!k) return;
        std::lock_guard lk(key_mu_);
        if (k->quota && k->used >= k->quota) { ++st_.rate_limited; throw http_error(429, std::format("the token quota of key '{}' ({} tokens) is used up", k->name, k->quota), {}, "insufficient_quota"); }
        double rpm = k->rpm > 0 ? k->rpm : opt_.rate_rpm;
        if (rpm <= 0) return;
        auto now = clk::now();
        if (!k->primed) { k->primed = true; k->bucket = rpm; k->last = now; }
        k->bucket = std::min(rpm, k->bucket + std::chrono::duration<double>(now - k->last).count() * rpm / 60.0); k->last = now;
        if (k->bucket < 1.0) {
            ++st_.rate_limited;
            int wait = std::max(1, int(std::ceil((1.0 - k->bucket) * 60.0 / rpm)));
            throw http_error(429, std::format("rate limit of {} requests per minute reached for key '{}'", rpm, k->name), {}, "rate_limit_exceeded", std::format("Retry-After: {}\r\n", wait));
        }
        k->bucket -= 1.0;
    }
    void charge(api_key* k, std::uint64_t tokens) { if (k) { std::lock_guard lk(key_mu_); k->used += tokens; } }

    // ---- routing
    void route(conn& c, const http_request& r) {
        auto path = r.path.substr(0, r.path.find('?'));
        if (r.method == "OPTIONS") {   // CORS preflight: no credentials needed
            st_.count(204);
            c.send(std::format("HTTP/1.1 204 No Content\r\n{}Access-Control-Allow-Headers: Authorization, Content-Type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                               "Access-Control-Max-Age: 600\r\nContent-Length: 0\r\nConnection: {}\r\n\r\n", cors_headers(), c.keep ? "keep-alive" : "close"));
            return;
        }
        auto need = [&](const char* method) {
            if (r.method != method) throw http_error(405, std::format("{} needs {}", path, method), {}, {}, std::format("Allow: {}, OPTIONS\r\n", method));
        };
        if (path == "/health" || path == "/healthz" || path == "/v1/health") {   // liveness: open, so a balancer needs no key
            need("GET");
            const bool drain = draining_.load();
            reply(c, drain ? 503 : 200, json::dump(json::value(json::object{{"status", drain ? "draining" : "ok"}, {"model", name_},
                {"kind", embedding_model() ? "embedding" : "generative"}, {"queue_waiting", double(sched_.waiting())}, {"queue_active", double(sched_.active_count())}})));
            return;
        }
        api_key* key = authenticate(r);
        if (path == "/metrics") { need("GET"); reply(c, 200, metrics(), "text/plain; version=0.0.4"); return; }
        if (path == "/v1/models") { need("GET"); reply(c, 200, json::dump(json::value(json::object{{"object", "list"}, {"data", model_list()}}))); return; }
        if (path.rfind("/v1/models/", 0) == 0) {
            need("GET");
            auto id = path.substr(11);
            for (auto& e : model_list()) if (e.str_or("id", "") == id) { reply(c, 200, json::dump(e)); return; }
            throw http_error(404, std::format("the model '{}' does not exist", id), "model", "model_not_found");
        }
        const bool chat = path == "/v1/chat/completions";
        if (path != "/v1/completions" && !chat && path != "/v1/embeddings") throw http_error(404, "no route " + path);
        need("POST");
        if (draining_) throw http_error(503, "the server is shutting down", {}, {}, "Retry-After: 1\r\n");
        admit_key(key);
        json::value j;
        try { j = json::parse(r.body); } catch (const std::exception& e) { bad_request({}, "the request body is not JSON: {}", e.what()); }
        if (!j.is_obj()) bad_request({}, "the request body must be a JSON object");
        if (path == "/v1/embeddings") { embeddings(c, j, key); return; }
        if (embedding_model())
            raise(errc::unsupported, "{} is an embedding model (blueprint session 'embedding'): it has no completions; use POST /v1/embeddings", name_);
        // Backpressure: a full queue answers 503 + Retry-After instead of piling up work it cannot hold.
        if (opt_.max_queue && sched_.waiting() >= opt_.max_queue) { ++st_.rejected; throw http_error(503, "server busy: request queue is full, retry later", {}, {}, "Retry-After: 1\r\n"); }
        completions(c, j, chat, key);
    }
    json::array model_list() const {
        json::array a;
        auto entry = [&](const std::string& id, const std::string& adapter) {
            json::object o{{"id", id}, {"object", "model"}, {"created", created_}, {"owned_by", "forge"}, {"kind", embedding_model() ? "embedding" : "generative"},
                           {"context_length", double(context())}, {"vocab_size", double(m_->vocab())}, {"architecture", m_->manifest().str_or("architecture", "")},
                           {"device", m_->device()}};
            if (!embedding_model()) { o.emplace_back("chat_template", !m_->chat_template().empty()); o.emplace_back("tool_format", inference::tool_format_name(tool_format_)); }
            if (!adapter.empty()) { o.emplace_back("parent", name_); o.emplace_back("adapter", adapter); }
            a.push_back(json::value(std::move(o)));
        };
        entry(name_, "");
        for (auto& ad : m_->adapter_names()) entry(name_ + ":" + ad, ad);
        return a;
    }
    i64 context() const { return opt_.context ? std::min(opt_.context, m_->max_context()) : m_->max_context(); }
    std::string metrics() {
        auto& s = sched_.stats();
        std::string o;
        auto put = [&](const char* name, const char* type, const char* help, double v) { o += std::format("# HELP {} {}\n# TYPE {} {}\n{} {}\n", name, help, name, type, name, v); };
        put("forge_steps", "counter", "Scheduler iterations", double(s.steps));
        put("forge_completed", "counter", "Requests the scheduler finished", double(s.completed));
        put("forge_failed_total", "counter", "Requests the scheduler ended with an error", double(s.failed));
        put("forge_prefix_hits", "counter", "Requests that reused cached prefix KV rows", double(s.prefix_hits));
        put("forge_prefix_tokens_reused", "counter", "Prompt tokens served from the prefix cache", double(s.prefix_tokens_reused));
        put("forge_peak_active", "gauge", "Most sequences decoded together", double(s.peak_active));
        put("forge_peak_kv_bytes", "gauge", "Peak KV bytes of the active sequences", double(s.peak_kv_bytes));
        put("forge_queue_waiting", "gauge", "Requests waiting for a slot", double(sched_.waiting()));
        put("forge_queue_active", "gauge", "Requests being decoded", double(sched_.active_count()));
        put("forge_admission_deferred_total", "counter", "Admissions deferred by the memory budget", double(s.deferred));
        put("forge_rejected_total", "counter", "Requests refused because the queue was full", double(st_.rejected.load()));
        put("forge_kv_evictions_total", "counter", "Prefix cache entries evicted", double(sched_.evictions()));
        put("forge_kv_spills_total", "counter", "Evicted prefixes written to the spill store", double(s.prefix_spills));
        put("forge_kv_reloads_total", "counter", "Spilled prefixes read back", double(s.prefix_reloads));
        put("forge_kv_reload_failures_total", "counter", "Spilled prefixes that could not be read back", double(s.prefix_reload_failures));
        put("forge_kv_spilled", "gauge", "Prefixes held in the spill store", double(sched_.spilled()));
        put("forge_http_connections_total", "counter", "Connections accepted", double(st_.connections.load()));
        put("forge_http_connections_active", "gauge", "Connections open", double(st_.active.load()));
        put("forge_http_connections_refused_total", "counter", "Connections refused at the connection limit", double(st_.refused_connections.load()));
        put("forge_http_request_errors_total", "counter", "Requests answered with an error", double(st_.request_errors.load()));
        put("forge_auth_failures_total", "counter", "Requests with a missing or wrong API key", double(st_.auth_failures.load()));
        put("forge_rate_limited_total", "counter", "Requests refused by a rate limit or quota", double(st_.rate_limited.load()));
        put("forge_prompt_tokens_total", "counter", "Prompt tokens of answered requests", double(st_.prompt_tokens.load()));
        put("forge_completion_tokens_total", "counter", "Tokens generated for answered requests", double(st_.completion_tokens.load()));
        put("forge_embedding_inputs_total", "counter", "Inputs embedded", double(st_.embedding_inputs.load()));
        put("forge_tool_calls_total", "counter", "Tool calls returned", double(st_.tool_calls.load()));
        put("forge_draining", "gauge", "1 while the server drains before exit", draining_ ? 1 : 0);
        o += "# HELP forge_http_responses_total Responses by status code\n# TYPE forge_http_responses_total counter\n";
        { std::lock_guard lk(st_.mu); for (auto& [code, n] : st_.by_status) o += std::format("forge_http_responses_total{{code=\"{}\"}} {}\n", code, n); }
        return o + mem::governor::instance().prometheus();
    }

    // ---- request fields
    static double number(const json::value& j, std::string_view k, double def, double lo, double hi) {
        auto* v = j.find(k);
        if (!v || v->is_null()) return def;
        if (!v->is_num() || !(v->num() >= lo) || !(v->num() <= hi)) bad_request(k, "\"{}\" must be a number in [{}, {}]", k, lo, hi);
        return v->num();
    }
    static double integer(const json::value& j, std::string_view k, double def, double lo, double hi) {
        double d = number(j, k, def, lo, hi);
        if (d != std::floor(d)) bad_request(k, "\"{}\" must be an integer", k);
        return d;
    }
    static bool boolean(const json::value& j, std::string_view k, bool def) {
        auto* v = j.find(k);
        if (!v || v->is_null()) return def;
        if (!v->is_bool()) bad_request(k, "\"{}\" must be true or false", k);
        return v->b();
    }
    int token_id(const json::value& x, std::string_view field) const {
        if (!x.is_num() || x.num() != std::floor(x.num()) || x.num() < 0 || x.num() >= double(m_->vocab() > 0 ? m_->vocab() : (i64(1) << 31) - 1))
            bad_request(field, "\"{}\": token ids must be integers in [0, {})", field, m_->vocab());
        return x.i32();
    }
    // "model": "<anything>" serves the base weights (clients send their own names); "<name>:<adapter>" or a field
    // "adapter" (alias "lora") selects a LoRA adapter loaded with --lora.
    std::string adapter_of(const json::value& j) const {
        std::string ad;
        for (auto k : {"adapter", "lora"}) if (auto* a = j.find(k); a && !a->is_null()) { if (!a->is_str()) bad_request(k, "\"{}\" must be a string", k); ad = a->str(); }
        if (ad.empty()) if (auto* mo = j.find("model"); mo && mo->is_str()) {
            auto& id = mo->str();
            if (id.size() > name_.size() + 1 && id.compare(0, name_.size(), name_) == 0 && id[name_.size()] == ':') ad = id.substr(name_.size() + 1);
            else if (m_->has_adapter(id)) ad = id;
        }
        if (!ad.empty() && !m_->has_adapter(ad)) throw http_error(404, std::format("the adapter '{}' is not loaded (start forge-serve with --lora {}=<file>)", ad, ad), "model", "model_not_found");
        return ad;
    }

    struct tool_setup {
        bool on = false;                       // parse the output for tool calls
        inference::tool_format fmt = inference::tool_format::generic;
        std::vector<std::string> names;
        json::value tools;                     // what the chat template is given (null with tool_choice "none")
        std::string grammar;                   // tool_choice required / named
    };
    tool_setup tools_of(const json::value& j) const {
        tool_setup t;
        auto* tools = j.find("tools");
        auto* tc = j.find("tool_choice");
        if (!tools || tools->is_null() || (tools->is_arr() && tools->arr().empty())) {
            if (tc && !tc->is_null() && !(tc->is_str() && (tc->str() == "none" || tc->str() == "auto"))) bad_request("tool_choice", "\"tool_choice\" needs \"tools\"");
            return t;
        }
        auto decls = inference::tool_decls(*tools);
        std::string choice = "auto", named;
        if (tc && tc->is_str()) choice = tc->str();
        else if (tc && tc->is_obj()) {
            auto* f = tc->find("function");
            if (tc->str_or("type", "") != "function" || !f || !f->is_obj() || f->str_or("name", "").empty())
                bad_request("tool_choice", "\"tool_choice\" as an object must be {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}");
            choice = "named"; named = f->str_or("name", "");
        } else if (tc && !tc->is_null()) bad_request("tool_choice", "\"tool_choice\" must be \"none\", \"auto\", \"required\" or a function object");
        if (choice != "none" && choice != "auto" && choice != "required" && choice != "named") bad_request("tool_choice", "\"tool_choice\" must be \"none\", \"auto\", \"required\" or a function object");
        if (choice == "none" || decls.empty()) return t;
        t.on = true; t.fmt = tool_format_; t.tools = *tools;
        for (auto& d : decls) t.names.push_back(d.name);
        if (choice == "required" || choice == "named") {
            if (choice == "named" && std::find(t.names.begin(), t.names.end(), named) == t.names.end()) bad_request("tool_choice", "\"tool_choice\" names the function \"{}\", which is not in \"tools\"", named);
            try { t.grammar = inference::tool_call_grammar(t.fmt, *tools, named, boolean(j, "parallel_tool_calls", false) && choice == "required"); }
            catch (const forge_exception& e) { bad_request("tools", "{}", e.what()); }
        }
        return t;
    }

    // per-choice state on the connection thread
    struct lp_item { int token; float logprob; std::vector<std::pair<int, float>> top; std::size_t offset; };
    struct choice {
        std::string text; std::vector<int> ids; std::string finish = "length", error;
        bool done = false, hit_stop = false;
        std::size_t sent = 0, scanned = 0, ids_sent = 0;   // bytes of text streamed / searched for stop strings; ids streamed
        std::vector<lp_item> lps; std::size_t lps_sent = 0;
        std::unique_ptr<inference::tool_call_parser> tp;
        std::vector<std::string> call_ids; std::vector<std::size_t> args_sent; std::vector<char> head_sent;
    };
    std::string new_call_id() {
        static const char* d = "0123456789abcdefghijklmnopqrstuvwxyz";
        std::lock_guard lk(id_mu_);
        std::string s = "call_";
        for (int i = 0; i < 24; ++i) s += d[id_rng_() % 36];
        return s;
    }
    std::string piece(int t) const { auto* tok = m_->tokenizer(); return tok ? tok->decode_one(t) : std::format("{} ", t); }
    json::value logprobs_json(const choice& ch, std::size_t from, std::size_t to, bool chat) const {
        auto bytes = [](const std::string& s) { json::array a; for (unsigned char b : s) a.push_back(json::value(int(b))); return json::value(std::move(a)); };
        if (chat) {
            json::array content;
            for (std::size_t i = from; i < to; ++i) {
                auto& l = ch.lps[i]; auto p = piece(l.token);
                json::array top;
                for (auto& [t, lp] : l.top) { auto tp = piece(t); top.push_back(json::value(json::object{{"token", sanitize(tp)}, {"logprob", double(lp)}, {"bytes", bytes(tp)}})); }
                content.push_back(json::value(json::object{{"token", sanitize(p)}, {"logprob", double(l.logprob)}, {"bytes", bytes(p)}, {"top_logprobs", std::move(top)}}));
            }
            return json::value(json::object{{"content", std::move(content)}});
        }
        // the legacy format keys alternatives by token text: a token that is not valid UTF-8 on its own (part of a
        // character) is written as "bytes:\\xNN..." as in OpenAI's API, so distinct tokens keep distinct keys
        auto label = [](const std::string& p) {
            if (sanitize(p) == p) return p;
            std::string o = "bytes:";
            for (unsigned char b : p) o += std::format("\\x{:02x}", b);
            return o;
        };
        json::array toks, lps, tops, offs;
        for (std::size_t i = from; i < to; ++i) {
            auto& l = ch.lps[i];
            toks.push_back(json::value(label(piece(l.token)))); lps.push_back(json::value(double(l.logprob))); offs.push_back(json::value(double(l.offset)));
            json::object top;
            for (auto& [t, lp] : l.top) top.emplace_back(label(piece(t)), double(lp));
            tops.push_back(json::value(std::move(top)));
        }
        return json::value(json::object{{"tokens", std::move(toks)}, {"token_logprobs", std::move(lps)}, {"top_logprobs", std::move(tops)}, {"text_offset", std::move(offs)}});
    }

    // POST /v1/completions and /v1/chat/completions
    void completions(conn& c, const json::value& j, bool chat, api_key* key) {
        auto* tok = m_->tokenizer();
        std::vector<int> prompt;
        tool_setup tools;
        if (chat) {
            if (!tok) raise(errc::unsupported, "chat needs a tokenizer");
            auto* msgs = j.find("messages");
            if (!msgs || !msgs->is_arr() || msgs->arr().empty()) bad_request("messages", "\"messages\" must be a non-empty array of {{\"role\", \"content\"}} objects");
            for (auto& msg : msgs->arr()) if (!msg.is_obj()) bad_request("messages", "\"messages\": every entry must be an object");
            tools = tools_of(j);
            if (!m_->chat_template().empty()) {   // the model's own chat template; it writes BOS and the role markers itself
                inference::chat_options co;
                if (tools.on) co.tools = tools.tools;
                co.add_generation_prompt = boolean(j, "add_generation_prompt", true);
                if (auto* k = j.find("chat_template_kwargs"); k && k->is_obj()) co.extra = *k;
                prompt = tok->encode(inference::apply_chat_template(*m_, inference::normalize_tool_messages(*msgs), co));
            } else {                              // a file without one: "<role>: <content>" lines
                std::string text;
                for (auto& msg : msgs->arr()) text += msg.str_or("role", "user") + ": " + msg.str_or("content", "") + "\n";
                text += "assistant:";
                prompt = tok->encode(text);
                if (tok->add_bos()) prompt.insert(prompt.begin(), tok->bos());
            }
        } else {
            auto* p = j.find("prompt");
            if (!p) bad_request("prompt", "\"prompt\" is required: a string or a list of token ids");
            if (p->is_arr()) for (auto& x : p->arr()) prompt.push_back(token_id(x, "prompt"));
            else if (p->is_str()) {
                if (!tok) raise(errc::unsupported, "text prompts need a tokenizer; send token ids");
                prompt = tok->encode(p->str());
                if (tok->add_bos() && !j.find("add_bos")) prompt.insert(prompt.begin(), tok->bos());
                else if (boolean(j, "add_bos", false) && tok->bos() >= 0) prompt.insert(prompt.begin(), tok->bos());
            } else bad_request("prompt", "\"prompt\" must be a string or a list of token ids");
        }
        if (prompt.empty()) bad_request(chat ? "messages" : "prompt", "empty prompt");
        if (i64(prompt.size()) >= context())
            throw http_error(400, std::format("the prompt is {} tokens; with at least one new token it does not fit the context of {} tokens", prompt.size(), context()),
                             chat ? "messages" : "prompt", "context_length_exceeded");
        inference::request q;
        q.max_new = int(integer(j, "max_tokens", 16, 1, 1 << 20));
        q.max_new = int(integer(j, "max_completion_tokens", q.max_new, 1, 1 << 20));   // the newer name wins when both are sent
        q.sampling.temperature = float(number(j, "temperature", 0, 0, 100));
        q.sampling.top_p = float(number(j, "top_p", 1, 0, 1));
        q.sampling.top_k = int(integer(j, "top_k", 0, 0, 1 << 30));
        q.sampling.min_p = float(number(j, "min_p", 0, 0, 1));
        q.sampling.typical_p = float(number(j, "typical_p", 1, 0, 1));
        q.sampling.mirostat_tau = float(number(j, "mirostat_tau", 0, 0, 1000));
        q.sampling.mirostat_eta = float(number(j, "mirostat_eta", 0.1, 0, 1000));
        q.sampling.presence_penalty = float(number(j, "presence_penalty", 0, -1000, 1000));
        q.sampling.frequency_penalty = float(number(j, "frequency_penalty", 0, -1000, 1000));
        q.sampling.repetition_penalty = float(number(j, "repetition_penalty", 1, 0.001, 1000));
        const std::uint64_t seed = std::uint64_t(integer(j, "seed", 0, 0, 9007199254740992.0));
        q.priority = int(integer(j, "priority", 0, -1000, 1000));
        q.adapter = adapter_of(j);
        const int n = int(integer(j, "n", 1, 1, opt_.max_n));
        if (auto* lb = j.find("logit_bias"); lb && !lb->is_null()) {
            if (!lb->is_obj() || lb->obj().size() > 1024) bad_request("logit_bias", "\"logit_bias\" must be an object of at most 1024 {{\"<token id>\": bias}} entries");
            for (auto& [k, v] : lb->obj()) {
                int id = -1; auto [e, ec] = std::from_chars(k.data(), k.data() + k.size(), id);
                if (k.empty() || ec != std::errc() || e != k.data() + k.size() || id < 0 || id >= m_->vocab() || !v.is_num() || !(v.num() >= -100) || !(v.num() <= 100))
                    bad_request("logit_bias", "\"logit_bias\": keys are token ids in [0, {}), values numbers in [-100, 100]", m_->vocab());
                // +-100 are absolute in OpenAI's API (ban / force); the logits here are not bounded, so they map to +-1e9
                q.sampling.logit_bias.emplace_back(id, v.num() <= -100 ? -1e9f : v.num() >= 100 ? 1e9f : float(v.num()));
            }
        }
        // stop: strings cut the text, token ids end the sequence like EOS
        std::vector<std::string> stops;
        std::vector<int> ends = chat ? ends_chat_ : ends_text_;
        auto add_end = [&](int id) { if (std::find(ends.begin(), ends.end(), id) == ends.end()) ends.push_back(id); };
        if (auto* s = j.find("stop"); s && !s->is_null()) {
            auto one = [&](const json::value& x) {
                if (x.is_str()) { if (x.str().empty() || x.str().size() > 1024) bad_request("stop", "\"stop\": strings must have 1 to 1024 bytes"); stops.push_back(x.str()); }
                else add_end(token_id(x, "stop"));
            };
            if (s->is_arr()) { if (s->arr().size() > 64) bad_request("stop", "\"stop\": at most 64 entries"); for (auto& x : s->arr()) one(x); }
            else one(*s);
        }
        if (auto* s = j.find("stop_token_ids"); s && !s->is_null()) {
            if (!s->is_arr() || s->arr().size() > 64) bad_request("stop_token_ids", "\"stop_token_ids\" must be a list of at most 64 token ids");
            for (auto& x : s->arr()) add_end(token_id(x, "stop_token_ids"));
        }
        q.stop_tokens = ends;
        // log-probabilities: chat {"logprobs": true, "top_logprobs": 0..20}; completions {"logprobs": 0..20}
        int top_lp = -1;
        if (auto* l = j.find("logprobs"); l && !l->is_null()) {
            if (l->is_bool()) { if (l->b()) top_lp = int(integer(j, "top_logprobs", 0, 0, 20)); }
            else top_lp = int(integer(j, "logprobs", 0, 0, 20));
        } else if (j.find("top_logprobs") && !j["top_logprobs"].is_null()) bad_request("top_logprobs", "\"top_logprobs\" needs \"logprobs\": true");
        q.top_logprobs = top_lp;
        // output constraints: a GBNF grammar, a response_format, or a forced tool call (one of them)
        if (auto* g = j.find("grammar"); g && !g->is_null()) {
            if (!g->is_str()) bad_request("grammar", "\"grammar\" must be GBNF text");
            q.grammar = std::make_shared<inference::grammar>(g->str());
        }
        if (auto* rf = j.find("response_format"); rf && !rf->is_null()) {
            if (!rf->is_obj()) bad_request("response_format", "\"response_format\" must be an object");
            auto type = rf->str_or("type", "");
            if (type == "json_schema") {
                auto* js = rf->find("json_schema");
                if (!js || !js->is_obj() || !js->find("schema")) bad_request("response_format", "\"response_format\" of type json_schema needs \"json_schema\": {{\"schema\": ...}}");
                q.grammar = std::make_shared<inference::grammar>(inference::json_schema_to_gbnf((*js)["schema"]));
            } else if (type == "json_object") {
                static const json::value any_object = json::parse(R"({"type":"object"})");
                q.grammar = std::make_shared<inference::grammar>(inference::json_schema_to_gbnf(any_object));
            } else if (type != "text") bad_request("response_format", "\"response_format\".type must be \"text\", \"json_object\" or \"json_schema\"");
        }
        if (!tools.grammar.empty()) {
            if (q.grammar) bad_request("tool_choice", "a forced tool call (tool_choice \"required\" or a function) cannot be combined with \"grammar\" or a JSON \"response_format\"");
            q.grammar = std::make_shared<inference::grammar>(tools.grammar);
        }
        if (q.grammar && !tok) raise(errc::unsupported, "grammar constraints need a tokenizer");
        const bool stream = boolean(j, "stream", false);
        bool include_usage = false;
        if (auto* so = j.find("stream_options"); so && !so->is_null()) {
            if (!so->is_obj()) bad_request("stream_options", "\"stream_options\" must be an object");
            include_usage = boolean(*so, "include_usage", false);
        }

        auto ch = std::make_shared<channel>();
        ch->cancelled.reset(new std::atomic<bool>[std::size_t(n)]);
        for (int i = 0; i < n; ++i) ch->cancelled[std::size_t(i)] = false;
        q.prompt = prompt;
        for (int i = 0; i < n; ++i) {
            inference::request qi = q;
            qi.sampling.seed = seed + std::uint64_t(i);
            auto pending = std::make_shared<std::shared_ptr<inference::token_logprob>>();
            if (top_lp >= 0) qi.on_logprobs = [pending](const inference::token_logprob& l) { *pending = std::make_shared<inference::token_logprob>(l); };
            qi.on_token = [ch, i, pending](int t) {
                std::lock_guard lk(ch->mu);
                ch->q.push_back({i, t, false, {}, std::move(*pending)}); pending->reset();
                ch->cv.notify_all();
                return !ch->cancelled[std::size_t(i)].load();
            };
            qi.on_done = [ch, i](const std::vector<int>&, const std::string& why) { std::lock_guard lk(ch->mu); ch->q.push_back({i, -1, true, why, nullptr}); ch->cv.notify_all(); };
            sched_.submit(std::move(qi));
        }

        const auto id = std::format("{}-{}", chat ? "chatcmpl" : "cmpl", ++counter_);
        const char* object = chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";
        const std::string model_id = q.adapter.empty() ? name_ : name_ + ":" + q.adapter;
        const double created = double(std::time(nullptr));
        std::vector<choice> cs(static_cast<std::size_t>(n));
        if (tools.on) for (auto& x : cs) x.tp = std::make_unique<inference::tool_call_parser>(tools.fmt, tools.names);
        std::size_t max_stop = 0; for (auto& s : stops) max_stop = std::max(max_stop, s.size());
        bool client_gone = false;
        auto event = [&](json::value choice_obj, bool with_usage_null = true) {
            json::object ev{{"id", id}, {"object", object}, {"created", created}, {"model", model_id}, {"choices", json::value(json::array{std::move(choice_obj)})}};
            if (include_usage && with_usage_null) ev.emplace_back("usage", nullptr);
            return json::dump(json::value(std::move(ev)));
        };
        auto send_event = [&](const std::string& data) {
            if (client_gone) return;
            if (!stream_event(c, data)) { client_gone = true; for (int i = 0; i < n; ++i) ch->cancelled[std::size_t(i)] = true; }
        };
        // streams what became visible of choice i: content up to `upto`, then tool-call deltas
        auto flush = [&](int i, bool final) {
            auto& x = cs[std::size_t(i)];
            std::size_t upto = x.text.size();
            if (x.tp) { x.tp->update(x.text, final); upto = std::min(upto, x.tp->content_end()); }
            if (!final && !x.hit_stop) {
                // text that could be the beginning of a stop string stays back until it is decided
                if (upto == x.text.size()) {
                    std::size_t hold = 0;
                    for (auto& s : stops)
                        for (std::size_t k = std::min(s.size() - 1, upto); k > hold; --k)
                            if (std::string_view(x.text).substr(upto - k, k) == std::string_view(s).substr(0, k)) { hold = k; break; }
                    upto -= hold;
                }
                upto = std::min(upto, complete_prefix(std::string_view(x.text).substr(0, upto)));
            }
            auto lp_field = [&](json::object& o) {
                if (top_lp < 0 || x.lps_sent >= x.lps.size()) return;
                o.emplace_back("logprobs", logprobs_json(x, x.lps_sent, x.lps.size(), chat)); x.lps_sent = x.lps.size();
            };
            if (upto > x.sent) {
                std::string p = sanitize(std::string_view(x.text).substr(x.sent, upto - x.sent));
                x.sent = upto;
                json::object o = chat ? json::object{{"index", i}, {"delta", json::value(json::object{{"content", p}})}, {"finish_reason", nullptr}}
                                      : json::object{{"index", i}, {"text", p}, {"token", x.ids.empty() ? -1 : x.ids.back()}, {"finish_reason", nullptr}};
                if (!chat) {   // "token": the last id of this event; "tokens": every id since the previous event
                    json::array idv; for (; x.ids_sent < x.ids.size(); ++x.ids_sent) idv.push_back(json::value(x.ids[x.ids_sent]));
                    o.emplace_back("tokens", std::move(idv));
                }
                lp_field(o);
                send_event(event(json::value(std::move(o))));
            }
            if (!x.tp) return;
            auto& calls = x.tp->calls();
            for (std::size_t k = 0; k < calls.size(); ++k) {
                if (k >= x.head_sent.size()) { x.head_sent.push_back(0); x.args_sent.push_back(0); x.call_ids.push_back(new_call_id()); }
                auto& call = calls[k];
                if (!call.named) continue;   // arguments before the name: wait for the name
                auto delta = [&](json::object fn, bool head) {
                    json::object tc{{"index", double(k)}};
                    if (head) { tc.emplace_back("id", x.call_ids[k]); tc.emplace_back("type", "function"); }
                    tc.emplace_back("function", json::value(std::move(fn)));
                    send_event(event(json::value(json::object{{"index", i}, {"delta", json::value(json::object{{"tool_calls", json::value(json::array{json::value(std::move(tc))})}})}, {"finish_reason", nullptr}})));
                };
                if (!x.head_sent[k]) { x.head_sent[k] = 1; delta(json::object{{"name", sanitize(call.name)}, {"arguments", ""}}, true); }
                std::string_view args = call.arguments;
                std::size_t to = call.done ? args.size() : complete_prefix(args);
                if (to > x.args_sent[k]) { delta(json::object{{"arguments", sanitize(args.substr(x.args_sent[k], to - x.args_sent[k]))}}, false); x.args_sent[k] = to; }
                else if (call.done && args.empty() && x.args_sent[k] == 0) { delta(json::object{{"arguments", "{}"}}, false); x.args_sent[k] = 2; }
            }
        };
        if (stream) {
            if (!begin_stream(c)) { client_gone = true; for (int i = 0; i < n; ++i) ch->cancelled[std::size_t(i)] = true; }
            if (chat) for (int i = 0; i < n; ++i) send_event(event(json::value(json::object{{"index", i}, {"delta", json::value(json::object{{"role", "assistant"}, {"content", ""}})}, {"finish_reason", nullptr}})));
        }
        for (int open = n; open > 0;) {
            std::deque<channel::event> batch;
            { std::unique_lock lk(ch->mu); ch->cv.wait(lk, [&] { return !ch->q.empty(); }); batch.swap(ch->q); }
            for (auto& e : batch) {
                auto& x = cs[std::size_t(e.idx)];
                if (e.done) {
                    --open; x.done = true;
                    if (x.hit_stop) x.finish = "stop";
                    else if (e.reason.rfind("error: ", 0) == 0) { x.finish = "error"; x.error = e.reason.substr(7); }
                    else x.finish = e.reason == "stop" || e.reason == "grammar" ? "stop" : e.reason;
                    continue;
                }
                if (x.hit_stop || std::find(ends.begin(), ends.end(), e.tok) != ends.end()) continue;   // EOS / end of turn: not part of the text
                std::size_t at = x.text.size();
                x.text += piece(e.tok); x.ids.push_back(e.tok);
                if (e.lp) x.lps.push_back({e.lp->token, e.lp->logprob, std::move(e.lp->top), at});
                if (max_stop) {
                    std::size_t from = x.scanned > max_stop ? x.scanned - max_stop : 0, cut = std::string::npos;
                    for (auto& s : stops) if (auto f = x.text.find(s, from); f != std::string::npos) cut = std::min(cut, f);
                    x.scanned = x.text.size();
                    if (cut != std::string::npos) { x.text.resize(cut); x.hit_stop = true; ch->cancelled[std::size_t(e.idx)] = true; }
                }
                if (stream) flush(e.idx, false);
            }
        }
        // every choice has ended: usage, tool calls, the response
        std::size_t completion = 0; for (auto& x : cs) completion += x.ids.size();
        st_.prompt_tokens += prompt.size(); st_.completion_tokens += completion;
        charge(key, prompt.size() + completion);
        json::object usage{{"prompt_tokens", double(prompt.size())}, {"completion_tokens", double(completion)}, {"total_tokens", double(prompt.size() + completion)}};
        std::string failure;
        for (auto& x : cs) if (x.finish == "error" && failure.empty()) failure = x.error;
        if (!failure.empty()) ++st_.request_errors;
        if (stream) {
            if (!failure.empty()) {   // the status line is gone: the error is an event, then the stream ends
                send_event(error_json(500, failure));
                if (!client_gone) end_stream(c);
                return;
            }
            for (int i = 0; i < n; ++i) {
                auto& x = cs[std::size_t(i)];
                flush(i, true);
                std::vector<inference::tool_call> fin; if (x.tp) fin = x.tp->finished();
                if (x.tp && fin.empty() && x.sent < x.text.size()) {   // text held back for a tool call that never completed
                    json::object o{{"index", i}, {"delta", json::value(json::object{{"content", sanitize(std::string_view(x.text).substr(x.sent))}})}, {"finish_reason", nullptr}};
                    x.sent = x.text.size(); send_event(event(json::value(std::move(o))));
                }
                if (!fin.empty()) x.finish = "tool_calls";
                st_.tool_calls += fin.size();
                json::object o = chat ? json::object{{"index", i}, {"delta", json::value(json::object{})}, {"finish_reason", x.finish}}
                                      : json::object{{"index", i}, {"text", ""}, {"finish_reason", x.finish}};
                if (top_lp >= 0 && x.lps_sent < x.lps.size()) { o.emplace_back("logprobs", logprobs_json(x, x.lps_sent, x.lps.size(), chat)); x.lps_sent = x.lps.size(); }
                json::object ev{{"id", id}, {"object", object}, {"created", created}, {"model", model_id}, {"choices", json::value(json::array{json::value(std::move(o))})}};
                // the last choice's closing event carries the usage (as before stream_options existed); with
                // include_usage a separate event with no choices follows, as in OpenAI's API
                if (i + 1 == n) ev.emplace_back("usage", json::value(usage));
                else if (include_usage) ev.emplace_back("usage", nullptr);
                send_event(json::dump(json::value(std::move(ev))));
            }
            if (include_usage)
                send_event(json::dump(json::value(json::object{{"id", id}, {"object", object}, {"created", created}, {"model", model_id}, {"choices", json::value(json::array{})}, {"usage", json::value(usage)}})));
            if (!client_gone) end_stream(c); else c.keep = false;
            return;
        }
        if (!failure.empty()) throw http_error(failure.find("out of memory") != std::string::npos ? 503 : 500, failure);
        json::array choices;
        for (int i = 0; i < n; ++i) {
            auto& x = cs[std::size_t(i)];
            json::object o{{"index", i}};
            if (chat) {
                json::object msg{{"role", "assistant"}};
                std::vector<inference::tool_call> fin;
                if (x.tp) { x.tp->update(x.text, true); fin = x.tp->finished(); }
                if (!fin.empty()) {
                    std::string content = x.text.substr(0, x.tp->content_end());
                    while (!content.empty() && (content.back() == ' ' || content.back() == '\n' || content.back() == '\r' || content.back() == '\t')) content.pop_back();
                    msg.emplace_back("content", content.empty() ? json::value(nullptr) : json::value(sanitize(content)));
                    json::array tcs;
                    for (auto& tc : fin)
                        tcs.push_back(json::value(json::object{{"id", new_call_id()}, {"type", "function"}, {"function", json::value(json::object{{"name", sanitize(tc.name)}, {"arguments", sanitize(tc.arguments)}})}}));
                    msg.emplace_back("tool_calls", std::move(tcs));
                    x.finish = "tool_calls";
                    st_.tool_calls += fin.size();
                } else msg.emplace_back("content", sanitize(x.text));
                o.emplace_back("message", json::value(std::move(msg)));
            } else {
                json::array idv; for (int t : x.ids) idv.push_back(json::value(t));
                o.emplace_back("text", sanitize(x.text)); o.emplace_back("tokens", json::value(std::move(idv)));
            }
            if (top_lp >= 0) o.emplace_back("logprobs", logprobs_json(x, 0, x.lps.size(), chat));
            o.emplace_back("finish_reason", x.finish);
            choices.push_back(json::value(std::move(o)));
        }
        reply(c, 200, json::dump(json::value(json::object{{"id", id}, {"object", object}, {"created", created}, {"model", model_id}, {"choices", std::move(choices)}, {"usage", json::value(usage)}})));
    }
    bool embedding_model() const { return m_->blueprint().session == "embedding"; }

    // POST /v1/embeddings (OpenAI-compatible). "input" is one text, a list of texts, one list of token ids or a list
    // of such lists; texts go through the model's tokenizer with its special tokens and token types ([CLS] .. [SEP]),
    // ids are taken as they are. One vector per input, in order; "encoding_format": "base64" returns each vector as
    // base64 of its little-endian f32 bytes. usage counts the tokens fed to the model.
    // All inputs of a request are tokenized and checked first, then embedded in one session::embed_batch call: a loop
    // with shared setup (plans and executors per sequence length are built once), not a batch axis in the graph.
    void embeddings(conn& c, const json::value& j, api_key* key) {
        if (!embedding_model())
            raise(errc::unsupported, "{} is not an embedding model (blueprint session '{}'): /v1/embeddings needs a model whose blueprint declares `session embedding`",
                  name_, m_->blueprint().session);
        auto* in = j.find("input");
        if (!in) raise(errc::invalid_argument, "\"input\" is required: a string, a list of strings, or token ids");
        auto fmt = j.str_or("encoding_format", "float");
        if (fmt != "float" && fmt != "base64") raise(errc::invalid_argument, "encoding_format must be \"float\" or \"base64\", got \"{}\"", fmt);
        // one entry per vector to compute: a text, or ids
        struct item { const std::string* text = nullptr; std::vector<int> ids; };
        std::vector<item> items;
        auto ids_of = [&](const json::value& a) {
            std::vector<int> ids;
            for (auto& x : a.arr()) {
                if (!x.is_num() || x.num() < 0 || x.num() >= double(m_->vocab() > 0 ? m_->vocab() : (i64(1) << 31) - 1) || x.num() != std::floor(x.num()))
                    raise(errc::invalid_argument, "\"input\": token ids must be integers in [0, {})", m_->vocab());
                ids.push_back(x.i32());
            }
            if (ids.empty()) raise(errc::invalid_argument, "\"input\": an empty token list");
            return ids;
        };
        if (in->is_str()) items.push_back({&in->str(), {}});
        else if (in->is_arr() && !in->arr().empty()) {
            auto& a = in->arr();
            if (a[0].is_num()) items.push_back({nullptr, ids_of(*in)});
            else for (auto& e : a) {
                if (e.is_str() && a[0].is_str()) items.push_back({&e.str(), {}});
                else if (e.is_arr() && a[0].is_arr()) items.push_back({nullptr, ids_of(e)});
                else raise(errc::invalid_argument, "\"input\" must be a string, a list of strings, a list of token ids or a list of token id lists");
            }
        } else raise(errc::invalid_argument, "\"input\" must be a non-empty string, list of strings, list of token ids or list of token id lists");
        if (items.size() > 2048) raise(errc::invalid_argument, "\"input\": at most 2048 entries per request, got {}", items.size());
        for (auto& it : items) if (it.text && it.text->empty()) raise(errc::invalid_argument, "\"input\": an empty string");
        // tokenize and check every input before anything runs
        std::vector<std::vector<int>> ids(items.size()), types(items.size());
        std::size_t tokens = 0;
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (items[i].text) {   // the tokenizer's special tokens and token types
                if (!m_->tokenizer()) raise(errc::invalid_argument, "{} has no tokenizer: \"input\" must be token ids", name_);
                auto enc = m_->tokenizer()->encode_typed(*items[i].text);
                ids[i] = std::move(enc.ids); types[i] = std::move(enc.type_ids);
            } else ids[i] = std::move(items[i].ids);
            if (i64(ids[i].size()) > m_->max_context())   // a request error (400), not a capacity one
                raise(errc::invalid_argument, "\"input\"[{}] is {} tokens, more than the model's {} positions (inputs are not truncated)", i, ids[i].size(), m_->max_context());
            tokens += ids[i].size();
        }
        std::vector<std::vector<float>> vecs;
        {
            std::lock_guard lk(embed_mu_);   // one embedding session, one request at a time
            if (!embed_) embed_ = std::make_unique<inference::session>(m_);
            if (auto* d = j.find("dimensions"); d && d->is_num()) {
                auto shape = embed_->embedding_shape(1);
                if (shape.size() != 1 || d->num() != double(shape[0]))
                    raise(errc::unsupported, "\"dimensions\": {} is not available, the model's embeddings have {} values", d->num(), shape.empty() ? i64(0) : shape.back());
            }
            vecs = embed_->embed_batch(ids, types);
        }
        json::array data;
        for (std::size_t i = 0; i < vecs.size(); ++i) {
            auto& v = vecs[i];
            json::object e{{"object", "embedding"}, {"index", double(i)}};
            if (fmt == "base64") {
                std::string raw(v.size() * 4, '\0');
                for (std::size_t k = 0; k < v.size(); ++k) {   // little-endian f32, whatever the host order
                    std::uint32_t u; std::memcpy(&u, &v[k], 4);
                    for (int b = 0; b < 4; ++b) raw[k * 4 + std::size_t(b)] = char((u >> (8 * b)) & 0xFF);
                }
                e.emplace_back("embedding", import::tokimp::base64(raw));
            } else {
                json::array vals; vals.reserve(v.size());
                for (float x : v) vals.push_back(double(x));
                e.emplace_back("embedding", std::move(vals));
            }
            data.push_back(json::value(std::move(e)));
        }
        st_.embedding_inputs += items.size(); st_.prompt_tokens += tokens;
        charge(key, tokens);
        reply(c, 200, json::dump(json::value(json::object{{"object", "list"}, {"data", std::move(data)}, {"model", name_},
            {"usage", json::value(json::object{{"prompt_tokens", double(tokens)}, {"total_tokens", double(tokens)}})}})));
    }

    std::shared_ptr<inference::model> m_;
    std::mutex embed_mu_; std::unique_ptr<inference::session> embed_;
    inference::batch_scheduler sched_;
    std::string name_;
    serve_options opt_;
    std::vector<api_key> keys_; api_key anon_; std::mutex key_mu_;
    std::vector<int> ends_chat_, ends_text_;
    inference::tool_format tool_format_ = inference::tool_format::generic;
    std::atomic<std::uint64_t> counter_{0};
    std::atomic<bool> draining_{false};
    server_stats st_;
    double created_ = 0;
    std::mutex id_mu_; std::mt19937_64 id_rng_{std::random_device{}()};
};

bool has_flag(int argc, char** argv, const std::string& k) {
    for (int i = 1; i < argc; ++i) if (argv[i] == "--" + k) return true;
    return false;
}
// every value of a repeatable flag
std::vector<std::string> args_of(int argc, char** argv, const std::string& k) {
    std::vector<std::string> v;
    for (int i = 1; i + 1 < argc; ++i) if (argv[i] == "--" + k) v.push_back(argv[i + 1]);
    return v;
}
void set_recv_timeout(forge_sock_t s, unsigned ms) {
#if defined(_WIN32)
    DWORD v = ms;
#else
    timeval v{time_t(ms / 1000), suseconds_t((ms % 1000) * 1000)};
#endif
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&v), sizeof v);
}
void set_send_timeout(forge_sock_t s, int seconds) {
#if defined(_WIN32)
    DWORD v = DWORD(seconds) * 1000;
#else
    timeval v{time_t(seconds), 0};
#endif
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&v), sizeof v);
}
std::string arg(int argc, char** argv, const std::string& k, const std::string& d) {
    for (int i = 1; i + 1 < argc; ++i) if (argv[i] == "--" + k) return argv[i + 1];
    return d;
}

// FSS (docs/serve.md "Security policy"): --policy F, else the load_policy() search. An active policy becomes the
// process's policy for enforce() (model loads check fs.read and integrity.models through it); profile none: nothing.
security::policy activate_policy(int argc, char** argv) {
    auto p = security::load_policy(arg(argc, argv, "policy", ""));
    if (p.active()) security::set_active_policy(p);
    return p;
}
std::string host_port(const std::string& h, int port) { return (h.find(':') != std::string::npos ? "[" + h + "]" : h) + ":" + std::to_string(port); }
// SBX-1: after the socket is bound and the model loaded, before the first request.
void server_lockdown(const security::policy& p, const char* who, const std::string& listen) {
    if (!p.active()) return;
    auto r = security::lockdown(p);
    for (auto& c : r.controls)
        if (c.verdict != "ok") std::fprintf(stderr, "%s: security %s %s: %s%s\n", who, c.id.c_str(), c.name.c_str(), c.state.c_str(), c.detail.empty() ? "" : (" (" + c.detail + ")").c_str());
    security::audit("sandbox", security::decision::allow, "SBX-1", listen, std::string(who) + ": locked down (profile " + security::name(p.prof) + ", " + p.source + ") before serving");
}

// forge-serve proxy --backend host:port [--backend ...] [--mode l4|l7] [--strategy p2c] [--host] [--port]
//   [--health-ms 2000] [--retries 3] [--route /prefix=host:port,host:port] [--policy F]
int run_proxy(int argc, char** argv) {
    using namespace forge::loadbalancer;
    config c; auto st = parse_strategy(arg(argc, argv, "strategy", "p2c"));
    if (!st) { std::fprintf(stderr, "forge-serve proxy: unknown --strategy\n"); return 2; }
    c.strat = *st;
    auto p = std::make_shared<pool>(c);
    proxy_config pc;
    std::vector<std::string> targets;   // every backend, for the policy's net_connect check
    for (int i = 2; i + 1 < argc; ++i) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--backend") { p->add({v, v}); targets.push_back(v); }
        else if (k == "--route") {
            auto eq = v.find('='); if (eq == std::string::npos) continue;
            auto rp = std::make_shared<pool>(c); std::string list = v.substr(eq + 1);
            for (std::size_t a = 0; a <= list.size();) { auto b = list.find(',', a); if (b == std::string::npos) b = list.size(); if (b > a) { rp->add({list.substr(a, b - a), list.substr(a, b - a)}); targets.push_back(list.substr(a, b - a)); } a = b + 1; }
            rp->set_health_check(tcp_probe());
            pc.routes.push_back({v.substr(0, eq), rp});
        }
    }
    if (!p->size() && pc.routes.empty()) { std::fprintf(stderr, "forge-serve proxy: need at least one --backend host:port\n"); return 2; }
    pc.mode = arg(argc, argv, "mode", "l7") == "l4" ? proxy_config::l4 : proxy_config::l7;
    pc.listen_host = arg(argc, argv, "host", "127.0.0.1"); pc.listen_port = std::stoi(arg(argc, argv, "port", "8080"));
    pc.retries = unsigned(std::stoi(arg(argc, argv, "retries", "3")));
    p->set_health_check(tcp_probe());
    security::policy pol;
    try {
        pol = activate_policy(argc, argv);
        if (pol.active()) {
            security::enforce(pol, security::capability::net_listen, host_port(pc.listen_host, pc.listen_port));
            for (auto& t : targets) security::enforce(pol, security::capability::net_connect, t);
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "forge-serve proxy: %s\n", e.what()); return 1; }
    proxy px(p, pc);
    if (auto s = px.start(); !s) { std::fprintf(stderr, "forge-serve proxy: %s\n", s.err.what.c_str()); return 1; }
    try { server_lockdown(pol, "forge-serve proxy", host_port(pc.listen_host, px.port())); }
    catch (const std::exception& e) { px.stop(); std::fprintf(stderr, "forge-serve proxy: %s\n", e.what()); return 1; }
    std::printf("forge-serve proxy (%s, %s) on %s:%d, %zu backends\n", pc.mode == proxy_config::l4 ? "l4" : "l7", strategy_name(c.strat), pc.listen_host.c_str(), px.port(), p->size());
    std::fflush(stdout);
    static std::atomic<bool> quit{false};
    forge::os::on_signal(forge::os::sig::interrupt, [](int) { quit = true; });
    forge::os::on_signal(forge::os::sig::terminate, [](int) { quit = true; });
    health_checker hc(*p, std::uint64_t(std::stoll(arg(argc, argv, "health-ms", "2000"))) * 1000000ull);
    std::vector<std::unique_ptr<health_checker>> rhc; for (auto& r : pc.routes) rhc.push_back(std::make_unique<health_checker>(*r.target, 2'000'000'000ull));
    while (!quit) forge::os::sleep_ns(100'000'000);
    px.stop();
    std::printf("forge-serve proxy: served %llu, errors %llu\n", (unsigned long long)px.served(), (unsigned long long)px.errors());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "proxy") return run_proxy(argc, argv);
    if (argc < 2 || argv[1][0] == '-') {
        std::puts("usage: forge-serve <model.fpk|hf_dir|model.gguf> [--host 127.0.0.1] [--port 8080] [--max-active 8] [--ctx N] [--codec f32] [--allow-unused]\n"
                  "         [--watch] [--no-pin] [--numa-node N] [--mem-limit 6G|75%|off] [--max-queue N] [--kv-spill] [--policy F]\n"
                  "         [--api-key K]... [--api-key-file F] [--rate-limit RPM] [--tls-cert F --tls-key F [--tls-client-ca F]] [--cors ORIGIN|off]\n"
                  "         [--lora NAME=FILE]... [--max-n 8] [--max-connections 256] [--max-header-bytes N] [--max-body-bytes N]\n"
                  "         [--request-timeout 60] [--keep-alive-timeout 30] [--shutdown-timeout 30]\n"
                  "       forge-serve proxy --backend host:port [...] [--mode l4|l7] [--strategy p2c|rr|wrr|lc|ewma|hash|kv] [--port 8080]");
        return 2;
    }
#if !defined(_WIN32)
    signal(SIGPIPE, SIG_IGN);
#endif
    forge::net_init();
    try {
        // FSS: the policy is active before anything is loaded (model fs.read / integrity checks), the listen address
        // must be allowed, and lockdown() runs once the socket is bound and the model loaded (SBX-1).
        const auto pol = activate_policy(argc, argv);
        const std::string listen_host = arg(argc, argv, "host", "127.0.0.1");
        if (pol.active()) {
            security::enforce(pol, security::capability::net_listen, host_port(listen_host, std::stoi(arg(argc, argv, "port", "8080"))));
            if (has_flag(argc, argv, "watch") && !security::try_enforce(pol, security::capability::exec, forge::os::exe_path()))
                throw security::policy_violation("--watch reloads by re-executing forge-serve, but the security policy (" + pol.source + ") denies exec", "SBX-1");
        }
        std::string path = argv[1];
        namespace fs = std::filesystem;
        if (fs::is_directory(path) || path.ends_with(".gguf")) {   // import into a temp .fpk
            auto out = (fs::temp_directory_path() / (fs::path(path).filename().string() + ".serve.fpk")).string();
            import::import_options o; o.codec = arg(argc, argv, "codec", "f32"); o.allow_unused = has_flag(argc, argv, "allow-unused");
            o.requant = arg(argc, argv, "codec", "keep");   // GGUF / AWQ / GPTQ inputs keep their stored quantisation unless a codec is given
            if (o.codec == "keep") o.codec = "f32";
            if (path.ends_with(".gguf")) import::import_gguf(path, out, o); else import::import_hf(path, out, o);
            path = out;
        }
        // --numa-node N: bind to a node before loading so weights/KV are first-touched node-local.
        if (auto nn = arg(argc, argv, "numa-node", ""); !nn.empty()) {
            auto r = forge::os::bind_thread_to_node(unsigned(std::stoul(nn)));
            std::printf("forge-serve: numa node %s%s\n", nn.c_str(), r ? "" : (" (bind failed: " + r.err.what + ")").c_str());
        }
        // Memory budget (docs/memory.md): --mem-limit overrides FORGE_MEM_LIMIT; 0/off disables.
        auto& host = mem::host_budget();
        if (auto ml = arg(argc, argv, "mem-limit", ""); !ml.empty()) {
            if (ml == "0" || ml == "off") host.set_limit(0);
            else if (auto v = mem::parse_size(ml, mem::detect_host_limits().effective())) host.set_limit(*v);
            else raise(errc::invalid_argument, "bad --mem-limit '{}' (use e.g. 6G, 512M or 75%)", ml);
        }
        mem::governor::instance().watch_pressure();
        auto m = inference::model::open(path);
        if (m->blueprint().session != "generative_sequence" && m->blueprint().session != "embedding")
            raise(errc::unsupported, "model session '{}' is neither generative nor an embedding model", m->blueprint().session);
        inference::batch_options bo;
        bo.max_active = std::stoi(arg(argc, argv, "max-active", "8"));
        bo.session.context = std::stoll(arg(argc, argv, "ctx", "0"));
        if (host.limit()) bo.budget = &host;
        // --kv-spill: prefix KV evicted from memory goes to disk (FORGE_MEM_SPILL_DIR, capped by FORGE_MEM_SPILL_MAX)
        // and is read back when a prompt hits it, instead of being recomputed (docs/memory.md)
        std::unique_ptr<mem::spill_store> spill;
        if (has_flag(argc, argv, "kv-spill")) { spill = std::make_unique<mem::spill_store>(); bo.spill = spill.get(); }
        namespace fos = forge::os;
        // P-core pinning: on hybrid CPUs keep the main thread (and the scheduler/request threads it spawns, which
        // inherit the mask) on performance cores. --no-pin disables.
        if (!has_flag(argc, argv, "no-pin")) if (auto t = fos::cpu_topology_query(); t.hybrid) {
            auto r = fos::pin_thread_to_kind(fos::core_kind::performance);
            std::printf("forge-serve: hybrid CPU, pinned to %zu P-cores%s\n", t.of_kind(fos::core_kind::performance).size(), r ? "" : " (failed)");
        }
        // LoRA adapters: --lora name=file (repeatable); a request selects one with "model": "<model>:<name>" or "adapter"
        for (auto& spec : args_of(argc, argv, "lora")) {
            auto eq = spec.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 >= spec.size()) raise(errc::invalid_argument, "--lora takes name=file, got '{}'", spec);
            m->add_adapter_file(spec.substr(0, eq), spec.substr(eq + 1));
        }
        serve_options so;
        so.max_queue = std::size_t(std::stoull(arg(argc, argv, "max-queue", "0")));
        so.context = bo.session.context;
        so.cors = arg(argc, argv, "cors", "*"); if (so.cors == "off" || so.cors == "none") so.cors.clear();
        if (so.cors.find_first_of("\r\n") != std::string::npos) raise(errc::invalid_argument, "bad --cors value");
        so.limits.max_header = std::size_t(std::stoull(arg(argc, argv, "max-header-bytes", "1048576")));
        so.limits.max_body = std::size_t(std::stoull(arg(argc, argv, "max-body-bytes", "67108864")));
        so.request_timeout_s = std::max(1, std::stoi(arg(argc, argv, "request-timeout", "60")));
        so.keep_alive_s = std::max(0, std::stoi(arg(argc, argv, "keep-alive-timeout", "30")));
        so.shutdown_s = std::max(0, std::stoi(arg(argc, argv, "shutdown-timeout", "30")));
        so.max_connections = std::size_t(std::max(1, std::stoi(arg(argc, argv, "max-connections", "256"))));
        so.max_n = std::clamp(std::stoi(arg(argc, argv, "max-n", "8")), 1, 64);
        so.rate_rpm = std::stod(arg(argc, argv, "rate-limit", "0"));
        // API keys: --api-key K (repeatable; visible in the process list, prefer the file) and --api-key-file F
        std::vector<api_key> keys;
        for (auto& k : args_of(argc, argv, "api-key")) {
            if (k.size() < 8) raise(errc::invalid_argument, "--api-key: a key needs at least 8 characters");
            api_key e; e.digest = key_digest(k); e.name = std::format("key{}", keys.size() + 1); keys.push_back(std::move(e));
        }
        for (auto& f : args_of(argc, argv, "api-key-file")) parse_key_file(f, keys);
        if (has_flag(argc, argv, "api-key-file") && keys.empty()) raise(errc::invalid_argument, "--api-key-file: no keys in the file");
        // TLS: --tls-cert / --tls-key, and --tls-client-ca for mutual TLS (the RPC layer's OpenSSL code, graph/rpc_secure.hpp)
        rpc::tls_server_config tls;
        tls.cert_file = arg(argc, argv, "tls-cert", ""); tls.key_file = arg(argc, argv, "tls-key", ""); tls.client_ca_file = arg(argc, argv, "tls-client-ca", "");
        const bool use_tls = !tls.cert_file.empty() || !tls.key_file.empty() || !tls.client_ca_file.empty();
        if (use_tls && (tls.cert_file.empty() || tls.key_file.empty())) raise(errc::invalid_argument, "TLS needs both --tls-cert and --tls-key");
#ifdef FORGE_HAVE_TLS
        std::shared_ptr<SSL_CTX> tls_ctx;
        if (use_tls) { security::require_private_file("--tls-key", tls.key_file, security::file_kind::secret); tls_ctx = rpc::make_server_tls(tls); }
#else
        if (use_tls) raise(errc::unsupported, "this forge-serve was built without TLS (FORGE_TLS=OFF or no OpenSSL): --tls-cert / --tls-key / --tls-client-ca are not available");
#endif
        const int shutdown_s = so.shutdown_s, send_timeout_s = so.send_timeout_s; const std::size_t max_conn = so.max_connections;
        server srv(m, bo, fs::path(argv[1]).filename().string(), so, std::move(keys));
        // Memory pressure -> drop cached prefix KV sessions (moderate: keep 1, critical: keep none).
        fos::memory_pressure::instance().subscribe([&srv](fos::pressure_level l) {
            if (l == fos::pressure_level::normal) return;
            srv.evict_kv(l == fos::pressure_level::critical ? 0 : 1);
            fos::service::log(fos::log_level::warning, std::string("memory pressure ") + fos::pressure_name(l) + ": evicting prefix KV cache", "forge-serve");
        });
        auto ls = fos::socket::listen(listen_host, std::stoi(arg(argc, argv, "port", "8080")), 128);
        if (!ls) raise(errc::io_error, "{}", ls.err.what);
        fos::socket& lsock = ls.value;
        const int port = lsock.local_port();
        const forge_sock_t s = lsock.native();
        server_lockdown(pol, "forge-serve", host_port(listen_host, port));
        if (listen_host != "127.0.0.1" && listen_host != "::1" && listen_host != "localhost" && !srv.auth_enabled())
            std::fprintf(stderr, "forge-serve: warning: listening on %s without --api-key / --api-key-file: anyone who can reach the port can use the model\n", listen_host.c_str());
        std::printf("forge-serve: %s on %s://%s (max_active %d, memory budget %s)\n", path.c_str(), use_tls ? "https" : "http", host_port(listen_host, port).c_str(), bo.max_active,
                    host.limit() ? mem::format_size(host.limit()).c_str() : "off");
        std::fflush(stdout);
        fos::service::ready("serving " + path);
        // Hot reload: --watch re-execs forge-serve when the model file changes (systemd sees RELOADING=1 then READY=1).
        std::thread watcher;
        if (has_flag(argc, argv, "watch")) watcher = std::thread([&, s] {
            fos::file_watcher w; if (auto st = w.add(argv[1]); !st) { std::fprintf(stderr, "forge-serve: --watch: %s\n", st.err.what.c_str()); return; }
            for (;;) for (auto& e : w.poll(1000)) if (e.kind != fos::watch_kind::removed) {
                fos::sleep_ns(500'000'000);   // let the writer finish
                std::printf("forge-serve: %s changed, reloading\n", e.path.c_str()); std::fflush(stdout);
                fos::service::reloading();
#if !defined(_WIN32)
                forge_sock_close(s);
                std::vector<char*> av(argv, argv + argc); av.push_back(nullptr);
                ::execv(fos::exe_path().c_str(), av.data());
                std::fprintf(stderr, "forge-serve: re-exec failed: %s\n", std::strerror(errno));
#else
                std::fprintf(stderr, "forge-serve: hot reload re-exec is not supported on Windows; restart the service\n");
#endif
            }
        });
        if (watcher.joinable()) watcher.detach();
        // SIGINT / SIGTERM: stop accepting, let requests in flight finish (up to --shutdown-timeout), then exit 0.
        static std::atomic<bool> quit{false};
        fos::on_signal(fos::sig::interrupt, [](int) { quit = true; });
        fos::on_signal(fos::sig::terminate, [](int) { quit = true; });
        auto& st = srv.stats();
        while (!quit) {
            if (!lsock.wait_readable(200)) continue;
            auto acc = lsock.accept();
            if (!acc) continue;
            ++st.connections;
            if (std::size_t(st.active.load()) >= max_conn) {   // bounded threads: over the limit a connection is turned away at once
                ++st.refused_connections;
                if (!use_tls) { static const char busy[] = "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"; acc.value.send(busy, sizeof busy - 1);
                                // take what the client already sent (briefly), so the close is not a reset that eats the reply
                                acc.value.shutdown_write(); set_recv_timeout(acc.value.native(), 50); char sink[2048]; (void)acc.value.recv(sink, sizeof sink); }
                continue;
            }
            ++st.active;
            auto cn = std::make_shared<conn>();
            cn->ch = rpc::channel(std::move(acc.value)); cn->send_timeout_s = send_timeout_s;
            try {
                std::thread([&srv, &st, cn, send_timeout_s
#ifdef FORGE_HAVE_TLS
                             , tls_ctx
#endif
                            ] {
                    struct release { server_stats& s; ~release() { --s.active; } } rel{st};
                    auto& sock = cn->ch.sock();
                    set_send_timeout(sock.native(), send_timeout_s);
#ifdef FORGE_HAVE_TLS
                    if (tls_ctx) {   // handshake under a 10 s receive timeout; a failed one (no certificate, plain HTTP on the port) just closes
                        set_recv_timeout(sock.native(), 10000);
                        std::string why;
                        if (!cn->ch.accept_tls(tls_ctx, &why)) { cn->ch.close(); return; }
                    }
#endif
                    set_recv_timeout(sock.native(), 1000);   // the tick of the idle / request deadlines
                    srv.handle(*cn);
                }).detach();
            } catch (const std::exception& e) { --st.active; std::fprintf(stderr, "forge-serve: cannot start a connection thread: %s\n", e.what()); }
        }
        // drain
        srv.set_draining();
        lsock.close();
        std::printf("forge-serve: shutting down, %lld connection(s) open\n", (long long)st.active.load()); std::fflush(stdout);
        for (auto until = clk::now() + std::chrono::seconds(shutdown_s); st.active.load() > 0 && clk::now() < until;) fos::sleep_ns(20'000'000);
        if (st.active.load() > 0) std::fprintf(stderr, "forge-serve: %lld connection(s) still open after %d s, exiting\n", (long long)st.active.load(), shutdown_s);
        std::fflush(nullptr);
        std::_Exit(st.active.load() > 0 ? 1 : 0);   // connection threads are detached and the scheduler may be mid-step: no static destructors
    } catch (const std::exception& e) {
        std::fprintf(stderr, "forge-serve: error: %s\n", e.what());
        return 1;
    }
}
