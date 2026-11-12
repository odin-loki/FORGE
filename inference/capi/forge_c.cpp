#include <cstddef>
#include <forge.h>
#include <forge/inference/inference.hpp>
#include <forge/inference/aot_build.hpp>
#include <forge/modelfile/aot.hpp>
#include <cstring>

// The C ABI (docs/capi.md). Nothing thrown crosses the boundary: every entry point catches, stores the message in
// the thread-local error and in the handle it was called on, and returns NULL / -1.
namespace {
namespace inf = forge::inference;
thread_local std::string g_err;

// a size-prefixed struct's field, or `def` when the caller's struct ends before it (or is NULL)
#define FORGE_FIELD(p, T, f, def) ((p) && (p)->size >= offsetof(T, f) + sizeof((p)->f) ? (p)->f : (def))

int64_t put_text(const std::string& s, char* out, int64_t cap) {
    if (out && cap > 0) { auto k = std::min<std::size_t>(s.size(), std::size_t(cap - 1)); std::memcpy(out, s.data(), k); out[k] = 0; }
    return int64_t(s.size());
}
inf::sampler_config sampler_of(const forge_sampler_params* p) {
    inf::sampler_config c;
    if (!p) return c;
    c.temperature = FORGE_FIELD(p, forge_sampler_params, temperature, 0.f);
    c.top_k = FORGE_FIELD(p, forge_sampler_params, top_k, 0);
    if (float v = FORGE_FIELD(p, forge_sampler_params, top_p, 0.f); v > 0) c.top_p = v;
    c.min_p = FORGE_FIELD(p, forge_sampler_params, min_p, 0.f);
    if (float v = FORGE_FIELD(p, forge_sampler_params, typical_p, 0.f); v > 0) c.typical_p = v;
    c.mirostat_tau = FORGE_FIELD(p, forge_sampler_params, mirostat_tau, 0.f);
    if (float v = FORGE_FIELD(p, forge_sampler_params, mirostat_eta, 0.f); v > 0) c.mirostat_eta = v;
    if (float v = FORGE_FIELD(p, forge_sampler_params, repetition_penalty, 0.f); v > 0) c.repetition_penalty = v;
    c.presence_penalty = FORGE_FIELD(p, forge_sampler_params, presence_penalty, 0.f);
    c.frequency_penalty = FORGE_FIELD(p, forge_sampler_params, frequency_penalty, 0.f);
    if (int v = FORGE_FIELD(p, forge_sampler_params, repetition_window, 0); v > 0) c.repetition_window = v;
    c.seed = FORGE_FIELD(p, forge_sampler_params, seed, std::uint64_t(0));
    const forge_logit_bias* lb = FORGE_FIELD(p, forge_sampler_params, logit_bias, static_cast<const forge_logit_bias*>(nullptr));
    int64_t nlb = FORGE_FIELD(p, forge_sampler_params, n_logit_bias, int64_t(0));
    if (nlb < 0 || (nlb > 0 && !lb)) forge::raise(forge::errc::invalid_argument, "sampler: logit_bias is NULL or its count is negative");
    for (int64_t i = 0; i < nlb; ++i) c.logit_bias.emplace_back(lb[i].token, lb[i].bias);
    return c;
}
// `grammar` (GBNF) or `json_schema` (converted), not both; null when neither is given
std::shared_ptr<const inf::grammar> grammar_of(const char* gbnf, const char* schema) {
    if (gbnf && schema) forge::raise(forge::errc::invalid_argument, "give a grammar or a json_schema, not both");
    if (gbnf) return std::make_shared<inf::grammar>(gbnf);
    if (schema) return std::make_shared<inf::grammar>(inf::json_schema_to_gbnf(forge::json::parse(schema)));
    return nullptr;
}
inf::session_options session_options_of(int64_t context, const char* kv_codec, int kv_offload) {
    inf::session_options o; o.context = context;
    if (kv_codec) o.kv_codec = kv_codec;
    if (kv_offload > 0) o.kv_offload = 1;
    return o;
}
}  // namespace

struct forge_weights {
    std::shared_ptr<inf::model> m;
    mutable std::string err, text;
};
struct forge_model {
    inf::session s;
    mutable std::string err;
    explicit forge_model(const char* p, inf::session_options o) : s(p, o) {}
    forge_model(std::shared_ptr<inf::model> m, inf::session_options o) : s(std::move(m), o) {}
};
struct forge_batch {
    std::shared_ptr<inf::model> m;
    inf::batch_scheduler sched;
    mutable std::string err;
    forge_batch(std::shared_ptr<inf::model> mm, inf::batch_options o) : m(std::move(mm)), sched(m, o) {}
};

namespace {
template <class H> void note(const H* h) { if (h) h->err = g_err; }
}
// FORGE_TRY: no handle to store the message in. FORGE_TRY_H: also store it in handle `h` (may be NULL).
#define FORGE_TRY(fail) catch (const std::exception& e) { g_err = e.what(); return fail; } catch (...) { g_err = "unknown error"; return fail; }
#define FORGE_TRY_H(h, fail) catch (const std::exception& e) { g_err = e.what(); note(h); return fail; } catch (...) { g_err = "unknown error"; note(h); return fail; }
#define FORGE_NEED(h, fail) do { if (!(h)) { g_err = "NULL handle"; return fail; } } while (0)

extern "C" {
uint32_t forge_abi_version(void) { return FORGE_ABI_VERSION; }
const char* forge_version(void) { return "0.1.0"; }
const char* forge_last_error(void) { return g_err.c_str(); }

forge_model* forge_open(const char* path, int64_t ctx, int verify) {
    try { return new forge_model(path, {.context = ctx, .verify = verify != 0}); } FORGE_TRY(nullptr)
}
void forge_close(forge_model* m) { delete m; }
void forge_reset(forge_model* m) { if (m) m->s.reset(); }
int64_t forge_vocab_size(const forge_model* m) {
    FORGE_NEED(m, -1);
    try { return int64_t(m->s.blueprint().param("vocab")); } FORGE_TRY_H(m, -1)
}
int64_t forge_tokenize(const forge_model* m, const char* text, int add_bos, int32_t* out, int64_t cap) {
    FORGE_NEED(m, -1);
    try {
        auto* t = m->s.tokenizer();
        if (!t) forge::raise(forge::errc::not_found, "model has no tokenizer");
        auto ids = t->encode(text, add_bos < 0);   // -1: the model's default (BOS / EOS flags or the tokenizer's post-processor)
        if (add_bos > 0 && t->bos() >= 0) ids.insert(ids.begin(), t->bos());
        for (int64_t i = 0; i < int64_t(ids.size()) && i < cap; ++i) out[i] = ids[std::size_t(i)];
        return int64_t(ids.size());
    } FORGE_TRY_H(m, -1)
}
int64_t forge_detokenize(const forge_model* m, const int32_t* ids, int64_t n, char* out, int64_t cap) {
    FORGE_NEED(m, -1);
    try {
        auto* t = m->s.tokenizer();
        if (!t) forge::raise(forge::errc::not_found, "model has no tokenizer");
        return put_text(t->decode(std::vector<int>(ids, ids + n)), out, cap);
    } FORGE_TRY_H(m, -1)
}
int64_t forge_chat_template_apply(const forge_model* m, const char* messages_json, int add_generation_prompt, char* out, int64_t cap) {
    return forge_chat_template_apply_ex(m, messages_json, nullptr, nullptr, add_generation_prompt, out, cap);
}
int forge_forward(forge_model* m, const int32_t* ids, int64_t n, float* logits, int64_t cap) {
    FORGE_NEED(m, -1);
    try {
        auto l = m->s.forward(std::span<const int>(ids, std::size_t(n)));
        std::memcpy(logits, l.data(), std::min<std::size_t>(l.size(), std::size_t(cap)) * 4);
        return 0;
    } FORGE_TRY_H(m, -1)
}
int64_t forge_generate(forge_model* m, const int32_t* prompt, int64_t np, int32_t max_new, const forge_sampling* s,
                       int32_t* out, int64_t cap, forge_token_cb cb, void* user) {
    return forge_generate_ex(m, prompt, np, max_new, s, nullptr, out, cap, cb, user);
}
int64_t forge_position(const forge_model* m) { return m ? m->s.position() : -1; }
int forge_rewind(forge_model* m, int64_t p) { FORGE_NEED(m, -1); try { m->s.rewind(p); return 0; } FORGE_TRY_H(m, -1) }
int forge_forward_all(forge_model* m, const int32_t* ids, int64_t n, float* logits, int64_t cap) {
    FORGE_NEED(m, -1);
    try {
        auto l = m->s.forward_all(std::span<const int>(ids, std::size_t(n)));
        std::memcpy(logits, l.data(), std::min<std::size_t>(l.size(), std::size_t(cap)) * 4);
        return 0;
    } FORGE_TRY_H(m, -1)
}
int64_t forge_generate_ex(forge_model* m, const int32_t* prompt, int64_t np, int32_t max_new, const forge_sampling* s,
                          const forge_generate_options* o, int32_t* out, int64_t cap, forge_token_cb cb, void* user) {
    return forge_generate_ex2(m, prompt, np, max_new, s, o, nullptr, out, cap, cb, user);
}
int64_t forge_generate_ex2(forge_model* m, const int32_t* prompt, int64_t np, int32_t max_new, const forge_sampling* s,
                           const forge_generate_options* o, const forge_sampling_ext* ext, int32_t* out, int64_t cap,
                           forge_token_cb cb, void* user) {
    FORGE_NEED(m, -1);
    try {
        inf::sampler_config c;
        if (s) { c.temperature = s->temperature; c.top_k = s->top_k; c.top_p = s->top_p > 0 ? s->top_p : 1.f; c.min_p = s->min_p;
                 c.repetition_penalty = s->repetition_penalty > 0 ? s->repetition_penalty : 1.f; c.seed = s->seed; }
        if (ext) {
            auto has = [&](std::size_t end) { return ext->size >= end; };
            if (has(offsetof(forge_sampling_ext, typical_p) + sizeof(float)) && ext->typical_p > 0) c.typical_p = ext->typical_p;
            if (has(offsetof(forge_sampling_ext, mirostat_tau) + sizeof(float))) c.mirostat_tau = ext->mirostat_tau;
            if (has(offsetof(forge_sampling_ext, mirostat_eta) + sizeof(float)) && ext->mirostat_eta > 0) c.mirostat_eta = ext->mirostat_eta;
        }
        inf::sampler smp(c);
        auto* tok = m->s.tokenizer();
        std::unique_ptr<inf::grammar_constraint> gc;
        if (o && o->grammar) {
            if (!tok) forge::raise(forge::errc::unsupported, "grammar needs a tokenizer");
            gc = std::make_unique<inf::grammar_constraint>(std::make_shared<inf::grammar>(o->grammar), *tok, int(m->s.vocab()), tok->eos());
            smp.filter = [&](std::span<float> l) { gc->apply(l); }; smp.allow = [&](int t) { return gc->allows(t); };
        }
        std::vector<int> r;
        auto on = [&](int t) {
            if (gc && !gc->accept(t)) return false;
            r.push_back(t);
            if (cb) { std::string txt = tok ? tok->decode_one(t) : std::string(); if (cb(t, txt.c_str(), user) != 0) return false; }
            return !(gc && gc->finished());
        };
        std::span<const int> pr(prompt, std::size_t(np));
        if (o && o->draft) inf::speculative_generate(m->s, o->draft->s, pr, max_new, smp, o->spec_k > 0 ? o->spec_k : 4, on);
        else m->s.generate(pr, max_new, smp, on);
        for (int64_t i = 0; i < int64_t(r.size()) && i < cap; ++i) out[i] = r[std::size_t(i)];
        return int64_t(r.size());
    } FORGE_TRY_H(m, -1)
}

// ------------------------------------------------------------------------------------------------ ABI 4
forge_weights* forge_weights_open(const char* path, const forge_open_params* p) {
    try {
        if (!path) forge::raise(forge::errc::invalid_argument, "forge_weights_open: NULL path");
        if (int d = FORGE_FIELD(p, forge_open_params, deterministic, 0); d == 1 || d == 2) forge::gpu::set_deterministic(d == 1);
        auto w = std::make_unique<forge_weights>();
        w->m = inf::model::open(path, FORGE_FIELD(p, forge_open_params, verify_hashes, 0) != 0);
        if (FORGE_FIELD(p, forge_open_params, no_aot, 0) != 0) w->m->set_aot(false);
        if (const char* dev = FORGE_FIELD(p, forge_open_params, device, static_cast<const char*>(nullptr))) w->m->set_device(dev);
        return w.release();
    } FORGE_TRY(nullptr)
}
void forge_weights_close(forge_weights* w) { delete w; }
forge_weights* forge_model_weights(forge_model* m) {
    FORGE_NEED(m, nullptr);
    try { auto w = std::make_unique<forge_weights>(); w->m = m->s.shared_model(); return w.release(); } FORGE_TRY_H(m, nullptr)
}
const char* forge_weights_error(const forge_weights* w) { return w ? w->err.c_str() : ""; }
const char* forge_weights_device(forge_weights* w) {
    FORGE_NEED(w, nullptr);
    try { w->text = w->m->device(); return w->text.c_str(); } FORGE_TRY_H(w, nullptr)
}
const char* forge_weights_set_device(forge_weights* w, const char* device) {
    FORGE_NEED(w, nullptr);
    try {
        if (!device) forge::raise(forge::errc::invalid_argument, "forge_weights_set_device: NULL device");
        w->text = w->m->set_device(device); return w->text.c_str();
    } FORGE_TRY_H(w, nullptr)
}
int forge_set_deterministic(int mode) {
    try {
        if (mode < 0) forge::gpu::reset_deterministic(); else forge::gpu::set_deterministic(mode != 0);
        return forge::gpu::deterministic() ? 1 : 0;
    } FORGE_TRY(-1)
}

forge_model* forge_session_open(forge_weights* w, const forge_session_params* p) {
    FORGE_NEED(w, nullptr);
    try {
        return new forge_model(w->m, session_options_of(FORGE_FIELD(p, forge_session_params, context, int64_t(0)),
                                                         FORGE_FIELD(p, forge_session_params, kv_codec, static_cast<const char*>(nullptr)),
                                                         FORGE_FIELD(p, forge_session_params, kv_offload, 0)));
    } FORGE_TRY_H(w, nullptr)
}
const char* forge_model_error(const forge_model* m) { return m ? m->err.c_str() : ""; }
int64_t forge_context_size(const forge_model* m) { return m ? m->s.context() : -1; }
const char* forge_session_kind(const forge_model* m) {
    FORGE_NEED(m, nullptr);
    try { return m->s.blueprint().session.c_str(); } FORGE_TRY_H(m, nullptr)   // lives as long as the weights
}
int64_t forge_manifest_json(const forge_model* m, char* out, int64_t cap) {
    FORGE_NEED(m, -1);
    try { return put_text(forge::json::dump(m->s.manifest()), out, cap); } FORGE_TRY_H(m, -1)
}

int32_t forge_token_bos(const forge_model* m) { return m && m->s.tokenizer() ? m->s.tokenizer()->bos() : -1; }
int32_t forge_token_eos(const forge_model* m) { return m && m->s.tokenizer() ? m->s.tokenizer()->eos() : -1; }
int64_t forge_token_text(const forge_model* m, int32_t id, char* out, int64_t cap) {
    FORGE_NEED(m, -1);
    try {
        auto* t = m->s.tokenizer();
        if (!t) forge::raise(forge::errc::not_found, "model has no tokenizer");
        if (id < 0 || std::size_t(id) >= t->vocab_size()) forge::raise(forge::errc::invalid_argument, "token id {} is outside the vocabulary of {}", id, t->vocab_size());
        return put_text(t->decode_one(id), out, cap);
    } FORGE_TRY_H(m, -1)
}
int32_t forge_token_id(const forge_model* m, const char* token) {
    if (!m || !token || !m->s.tokenizer()) return -1;
    try { return m->s.tokenizer()->token_id(token); } FORGE_TRY_H(m, -1)
}
int forge_token_is_special(const forge_model* m, int32_t id) { return m && m->s.tokenizer() && m->s.tokenizer()->is_special(id) ? 1 : 0; }
int64_t forge_stop_tokens(const forge_model* m, int chat, int32_t* out, int64_t cap) {
    FORGE_NEED(m, -1);
    try {
        auto ids = inf::end_of_turn_tokens(*m->s.shared_model(), chat != 0);
        for (int64_t i = 0; i < int64_t(ids.size()) && i < cap && out; ++i) out[i] = ids[std::size_t(i)];
        return int64_t(ids.size());
    } FORGE_TRY_H(m, -1)
}
int64_t forge_chat_template(const forge_model* m, char* out, int64_t cap) {
    FORGE_NEED(m, -1);
    try { return put_text(m->s.shared_model()->chat_template(), out, cap); } FORGE_TRY_H(m, -1)
}
int64_t forge_chat_template_apply_ex(const forge_model* m, const char* messages_json, const char* tools_json, const char* extra_json,
                                     int add_generation_prompt, char* out, int64_t cap) {
    FORGE_NEED(m, -1);
    try {
        if (!messages_json) forge::raise(forge::errc::invalid_argument, "chat template: NULL messages");
        inf::chat_options o; o.add_generation_prompt = add_generation_prompt != 0;
        if (tools_json) o.tools = forge::json::parse(tools_json);
        if (extra_json) { o.extra = forge::json::parse(extra_json); if (!o.extra.is_obj()) forge::raise(forge::errc::invalid_argument, "chat template: extra variables must be a JSON object"); }
        return put_text(inf::apply_chat_template(*m->s.shared_model(), inf::normalize_tool_messages(forge::json::parse(messages_json)), o), out, cap);
    } FORGE_TRY_H(m, -1)
}

int64_t forge_generate_v4(forge_model* m, const int32_t* prompt, int64_t np, int32_t max_new, const forge_generate_params* p,
                          int32_t* out, int64_t cap, int32_t* stop_reason) {
    FORGE_NEED(m, -1);
    try {
        if (np <= 0 || !prompt) forge::raise(forge::errc::invalid_argument, "generate: empty prompt");
        using P = forge_generate_params;
        inf::sampler smp(sampler_of(FORGE_FIELD(p, P, sampler, static_cast<const forge_sampler_params*>(nullptr))));
        auto* tok = m->s.tokenizer();
        const int eos = tok ? tok->eos() : -1;
        const bool ignore_eos = FORGE_FIELD(p, P, ignore_eos, 0) != 0;
        std::vector<int> stops;
        if (const int32_t* st = FORGE_FIELD(p, P, stop_tokens, static_cast<const int32_t*>(nullptr))) {
            int64_t n = FORGE_FIELD(p, P, n_stop_tokens, int64_t(0));
            if (n < 0) forge::raise(forge::errc::invalid_argument, "generate: negative stop token count");
            stops.assign(st, st + n);
        }
        std::unique_ptr<inf::grammar_constraint> gc;
        if (auto g = grammar_of(FORGE_FIELD(p, P, grammar, static_cast<const char*>(nullptr)), FORGE_FIELD(p, P, json_schema, static_cast<const char*>(nullptr)))) {
            if (!tok) forge::raise(forge::errc::unsupported, "grammar needs a tokenizer");
            gc = std::make_unique<inf::grammar_constraint>(std::move(g), *tok, int(m->s.vocab()), ignore_eos ? -1 : eos, stops);
            smp.filter = [&](std::span<float> l) { gc->apply(l); }; smp.allow = [&](int t) { return gc->allows(t); };
        }
        forge_token_cb cb = FORGE_FIELD(p, P, on_token, static_cast<forge_token_cb>(nullptr));
        void* user = FORGE_FIELD(p, P, user, static_cast<void*>(nullptr));
        std::vector<int> r; int why = FORGE_STOP_LENGTH;
        auto on = [&](int t) {
            if ((t == eos && !ignore_eos) || std::find(stops.begin(), stops.end(), t) != stops.end()) { why = FORGE_STOP_EOS; return false; }
            if (gc && !gc->accept(t)) { why = FORGE_STOP_GRAMMAR; return false; }
            r.push_back(t);
            if (cb) { std::string txt = tok ? tok->decode_one(t) : std::string(); if (cb(t, txt.c_str(), user) != 0) { why = FORGE_STOP_CALLBACK; return false; } }
            if (gc && gc->finished()) { why = FORGE_STOP_GRAMMAR; return false; }
            return true;
        };
        std::span<const int> pr(prompt, std::size_t(np));
        forge_model* draft = FORGE_FIELD(p, P, draft, static_cast<forge_model*>(nullptr));
        if (FORGE_FIELD(p, P, reset, 0) != 0) m->s.reset();
        if (draft) {
            if (ignore_eos) forge::raise(forge::errc::unsupported, "generate: ignore_eos is not available with a draft model");
            int k = FORGE_FIELD(p, P, spec_k, 0);
            inf::speculative_generate(m->s, draft->s, pr, max_new, smp, k > 0 ? k : 4, on);
        } else m->s.generate(pr, max_new, smp, on, nullptr, false);
        for (int64_t i = 0; i < int64_t(r.size()) && i < cap && out; ++i) out[i] = r[std::size_t(i)];
        if (stop_reason) *stop_reason = why;
        return int64_t(r.size());
    } FORGE_TRY_H(m, -1)
}
int64_t forge_json_schema_to_grammar(const char* schema_json, char* out, int64_t cap) {
    try {
        if (!schema_json) forge::raise(forge::errc::invalid_argument, "NULL schema");
        return put_text(inf::json_schema_to_gbnf(forge::json::parse(schema_json)), out, cap);
    } FORGE_TRY(-1)
}
int forge_grammar_check(const char* grammar) {
    try {
        if (!grammar) forge::raise(forge::errc::invalid_argument, "NULL grammar");
        inf::grammar g(grammar); (void)g; return 0;
    } FORGE_TRY(-1)
}

int forge_adapter_load(forge_weights* w, const char* name, const char* path) {
    FORGE_NEED(w, -1);
    try {
        if (!name || !path) forge::raise(forge::errc::invalid_argument, "adapter: NULL name or path");
        w->m->add_adapter_file(name, path); return 0;
    } FORGE_TRY_H(w, -1)
}
int forge_adapter_exists(forge_weights* w, const char* name) { return w && name && *name && w->m->has_adapter(name) ? 1 : 0; }
int forge_adapter_set(forge_model* m, const char* name) {
    FORGE_NEED(m, -1);
    try { m->s.set_adapter(name ? name : ""); return 0; } FORGE_TRY_H(m, -1)
}
const char* forge_adapter_get(const forge_model* m) { return m ? m->s.adapter().c_str() : ""; }

int64_t forge_state_save(forge_model* m, int64_t n_tokens, void* out, int64_t cap) {
    FORGE_NEED(m, -1);
    try {
        if (n_tokens > m->s.position()) forge::raise(forge::errc::invalid_argument, "state: {} tokens asked, the session holds {}", n_tokens, m->s.position());
        auto b = m->s.save_prefix(n_tokens);
        if (out && cap > 0) std::memcpy(out, b.data(), std::min<std::size_t>(b.size(), std::size_t(cap)));
        return int64_t(b.size());
    } FORGE_TRY_H(m, -1)
}
int forge_state_load(forge_model* m, const void* data, int64_t n) {
    FORGE_NEED(m, -1);
    try {
        if (!data || n < 0) forge::raise(forge::errc::invalid_argument, "state: NULL data");
        try { m->s.load_prefix(std::span<const std::byte>(static_cast<const std::byte*>(data), std::size_t(n))); }
        catch (...) { m->s.reset(); throw; }   // a blob that fails half way must not leave a half-loaded state
        return 0;
    } FORGE_TRY_H(m, -1)
}
int forge_state_copy(forge_model* dst, forge_model* src, int64_t n_tokens) {
    FORGE_NEED(dst, -1); FORGE_NEED(src, -1);
    try {
        if (dst == src) forge::raise(forge::errc::invalid_argument, "state: source and destination are the same session");
        if (dst->s.adapter() != src->s.adapter()) forge::raise(forge::errc::invalid_argument, "state: the sessions use different adapters ('{}' and '{}')", dst->s.adapter(), src->s.adapter());
        if (n_tokens < 0 || n_tokens > src->s.position()) forge::raise(forge::errc::invalid_argument, "state: {} tokens asked, the source holds {}", n_tokens, src->s.position());
        if (n_tokens > dst->s.context()) forge::raise(forge::errc::out_of_memory, "state: {} tokens exceed the destination's context {}", n_tokens, dst->s.context());
        dst->s.copy_prefix(src->s, n_tokens); return 0;
    } FORGE_TRY_H(dst, -1)
}
int64_t forge_state_bytes(const forge_model* m) { return m ? int64_t(m->s.state().bytes()) : -1; }

int64_t forge_embed_dim(const forge_model* m) {
    FORGE_NEED(m, -1);
    try { return int64_t(forge::numel(m->s.embedding_shape(1))); } FORGE_TRY_H(m, -1)
}
int64_t forge_embed_tokens(forge_model* m, const int32_t* ids, int64_t n, const int32_t* type_ids, float* out, int64_t cap) {
    FORGE_NEED(m, -1);
    try {
        if (!ids || n <= 0) forge::raise(forge::errc::invalid_argument, "embed: no tokens");
        std::span<const int> types; if (type_ids) types = std::span<const int>(type_ids, std::size_t(n));
        auto v = m->s.embed(std::span<const int>(ids, std::size_t(n)), types);
        if (out && cap > 0) std::memcpy(out, v.data(), std::min<std::size_t>(v.size(), std::size_t(cap)) * sizeof(float));
        return int64_t(v.size());
    } FORGE_TRY_H(m, -1)
}
int64_t forge_embed_text(forge_model* m, const char* text, const char* pair, float* out, int64_t cap, int64_t* n_tokens) {
    FORGE_NEED(m, -1);
    try {
        if (!text) forge::raise(forge::errc::invalid_argument, "embed: NULL text");
        std::size_t nt = 0; std::string second; if (pair) second = pair;
        auto v = m->s.embed_text(text, pair ? &second : nullptr, &nt);
        if (n_tokens) *n_tokens = int64_t(nt);
        if (out && cap > 0) std::memcpy(out, v.data(), std::min<std::size_t>(v.size(), std::size_t(cap)) * sizeof(float));
        return int64_t(v.size());
    } FORGE_TRY_H(m, -1)
}
int64_t forge_embed_texts(forge_model* m, const char* const* texts, int64_t n, float* out, int64_t cap) {
    FORGE_NEED(m, -1);
    try {
        m->s.check_embedding();
        if (!texts || n <= 0) forge::raise(forge::errc::invalid_argument, "embed: no texts");
        auto* tk = m->s.tokenizer();
        if (!tk) forge::raise(forge::errc::not_found, "embed: the model has no tokenizer; use forge_embed_tokens");
        std::vector<std::vector<int>> ids(static_cast<std::size_t>(n)), types(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            if (!texts[i]) forge::raise(forge::errc::invalid_argument, "embed: texts[{}] is NULL", i);
            auto e = tk->encode_typed(texts[i]);
            ids[std::size_t(i)] = std::move(e.ids); types[std::size_t(i)] = std::move(e.type_ids);
        }
        const int64_t dim = int64_t(forge::numel(m->s.embedding_shape(1)));
        auto vecs = m->s.embed_batch(ids, types);
        for (int64_t i = 0; i < n; ++i) {
            auto& v = vecs[std::size_t(i)];
            if (int64_t(v.size()) != dim) forge::raise(forge::errc::unsupported, "embed: the model's output depends on the sequence length ({} values for texts[{}]); use forge_embed_text per text", v.size(), i);
            if (out && (i + 1) * dim <= cap) std::memcpy(out + i * dim, v.data(), std::size_t(dim) * sizeof(float));
        }
        return dim;
    } FORGE_TRY_H(m, -1)
}

forge_batch* forge_batch_open(forge_weights* w, const forge_batch_params* p) {
    FORGE_NEED(w, nullptr);
    try {
        using P = forge_batch_params;
        inf::batch_options o;
        if (int v = FORGE_FIELD(p, P, max_active, 0); v > 0) o.max_active = v;
        if (int64_t v = FORGE_FIELD(p, P, prefill_chunk, int64_t(0)); v > 0) o.prefill_chunk = v;
        if (int v = FORGE_FIELD(p, P, prefix_cache, 0); v != 0) o.prefix_cache = v < 0 ? 0 : std::size_t(v);
        o.session = session_options_of(FORGE_FIELD(p, P, context, int64_t(0)), FORGE_FIELD(p, P, kv_codec, static_cast<const char*>(nullptr)), 0);
        if (w->m->blueprint().session != "generative_sequence") forge::raise(forge::errc::unsupported, "batch: session kind '{}' cannot generate", w->m->blueprint().session);
        return new forge_batch(w->m, o);
    } FORGE_TRY_H(w, nullptr)
}
void forge_batch_close(forge_batch* b) {
    if (!b) return;
    try { b->sched.cancel_all(); b->sched.run_until_idle(); } catch (...) {}
    delete b;
}
const char* forge_batch_error(const forge_batch* b) { return b ? b->err.c_str() : ""; }
uint64_t forge_batch_submit(forge_batch* b, const forge_batch_request* r) {
    FORGE_NEED(b, 0);
    try {
        using R = forge_batch_request;
        if (!r) forge::raise(forge::errc::invalid_argument, "batch: NULL request");
        const int32_t* prompt = FORGE_FIELD(r, R, prompt, static_cast<const int32_t*>(nullptr));
        const int64_t np = FORGE_FIELD(r, R, n_prompt, int64_t(0));
        if (!prompt || np <= 0) forge::raise(forge::errc::invalid_argument, "batch: empty prompt");
        inf::request q;
        q.prompt.assign(prompt, prompt + np);
        if (int v = FORGE_FIELD(r, R, max_new, 0); v > 0) q.max_new = v;
        q.sampling = sampler_of(FORGE_FIELD(r, R, sampler, static_cast<const forge_sampler_params*>(nullptr)));
        q.grammar = grammar_of(FORGE_FIELD(r, R, grammar, static_cast<const char*>(nullptr)), FORGE_FIELD(r, R, json_schema, static_cast<const char*>(nullptr)));
        if (const int32_t* st = FORGE_FIELD(r, R, stop_tokens, static_cast<const int32_t*>(nullptr))) {
            int64_t n = FORGE_FIELD(r, R, n_stop_tokens, int64_t(0));
            if (n < 0) forge::raise(forge::errc::invalid_argument, "batch: negative stop token count");
            q.stop_tokens.assign(st, st + n);
        }
        if (const char* ad = FORGE_FIELD(r, R, adapter, static_cast<const char*>(nullptr))) {
            if (!b->m->has_adapter(ad)) forge::raise(forge::errc::not_found, "adapter '{}' not loaded", ad);
            q.adapter = ad;
        }
        q.priority = FORGE_FIELD(r, R, priority, 0);
        forge_token_cb on_token = FORGE_FIELD(r, R, on_token, static_cast<forge_token_cb>(nullptr));
        forge_done_cb on_done = FORGE_FIELD(r, R, on_done, static_cast<forge_done_cb>(nullptr));
        void* user = FORGE_FIELD(r, R, user, static_cast<void*>(nullptr));
        const std::uint64_t id = b->sched.allocate_id();
        if (on_token) q.on_token = [on_token, user, tok = b->m->tokenizer()](int t) {
            std::string txt = tok ? tok->decode_one(t) : std::string();
            return on_token(t, txt.c_str(), user) == 0;
        };
        if (on_done) q.on_done = [on_done, user, id](const std::vector<int>& toks, const std::string& why) {
            on_done(id, toks.data(), int64_t(toks.size()), why.c_str(), user);   // int is int32_t on every supported target
        };
        return b->sched.submit(std::move(q), id);
    } FORGE_TRY_H(b, 0)
}
int forge_batch_cancel(forge_batch* b, uint64_t id) {
    FORGE_NEED(b, 0);
    try { return b->sched.cancel(id) ? 1 : 0; } FORGE_TRY_H(b, 0)
}
int forge_batch_step(forge_batch* b) {
    FORGE_NEED(b, -1);
    try { return b->sched.step() ? 1 : 0; } FORGE_TRY_H(b, -1)
}
int forge_batch_run(forge_batch* b) {
    FORGE_NEED(b, -1);
    try { b->sched.run_until_idle(); return 0; } FORGE_TRY_H(b, -1)
}
int forge_batch_stats_get(forge_batch* b, forge_batch_stats* out) {
    FORGE_NEED(b, -1);
    try {
        if (!out || out->size < sizeof(uint32_t) * 2) forge::raise(forge::errc::invalid_argument, "batch: stats needs out->size set");
        auto& s = b->sched.stats();
        const std::uint64_t vals[] = {s.steps, s.completed, s.failed, s.prefix_hits, s.prefix_tokens_reused, s.peak_active, s.peak_kv_bytes,
                                      b->sched.waiting(), b->sched.active_count()};
        // the fields are consecutive uint64_t after the 8-byte header: fill as many as the caller's struct has
        auto* dst = reinterpret_cast<unsigned char*>(out) + offsetof(forge_batch_stats, steps);
        std::size_t room = (std::min<std::size_t>(out->size, sizeof(forge_batch_stats)) - offsetof(forge_batch_stats, steps)) / sizeof(std::uint64_t);
        std::memcpy(dst, vals, std::min(room, std::size(vals)) * sizeof(std::uint64_t));
        return 0;
    } FORGE_TRY_H(b, -1)
}
void forge_batch_evict_prefix_cache(forge_batch* b, int64_t keep) { if (b) b->sched.request_kv_eviction(keep < 0 ? 0 : std::size_t(keep)); }

int64_t forge_aot_precompile(const char* fpk_path, const char* out_path, const int32_t* seq_lengths, int64_t n) {
    try {
        if (!fpk_path) forge::raise(forge::errc::invalid_argument, "aot: NULL model path");
        if (n < 0 || (n > 0 && !seq_lengths)) forge::raise(forge::errc::invalid_argument, "aot: bad sequence length list");
        inf::model m(fpk_path);
        if (m.file().has_section("signature"))
            forge::raise(forge::errc::unsupported, "aot: {} is signed; a plan sidecar is not covered by its signature and would be ignored (use `forge aot --key`)", fpk_path);
        inf::aot_build_options bo;
        for (int64_t i = 0; i < n; ++i) bo.shapes.push_back(seq_lengths[i]);
        auto pl = inf::aot_compile_plans(m, bo);
        forge::modelfile::aot::write_sidecar(out_path ? std::string(out_path) : std::string(fpk_path) + forge::modelfile::aot::sidecar_ext, pl.blob);
        return int64_t(pl.targets.size());
    } FORGE_TRY(-1)
}
}
