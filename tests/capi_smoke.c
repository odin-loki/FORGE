/* C11 consumer of forge.h: proves the ABI is usable from plain C, ABI 1 through 4.
   Usage: capi_smoke model.fpk [embedding_model.fpk [work_dir]]
   With an embedding model the forge_embed_* calls are run too; with a work dir, forge_adapter_load (a .safetensors
   written here) and forge_aot_precompile. */
#include <forge.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("  CHECK failed %s:%d: %s (last error: %s)\n", __FILE__, __LINE__, #c, forge_last_error()); ++fails; } } while (0)

static int on_tok(int32_t t, const char* txt, void* u) { (void)t; (void)txt; ++*(int*)u; return 0; }
static int stop_after_3(int32_t t, const char* txt, void* u) { (void)t; (void)txt; return ++*(int*)u >= 3; }

typedef struct { uint64_t id; int32_t toks[64]; int64_t n; char reason[96]; int tokens_seen; int done; } req_state;
static int batch_tok(int32_t t, const char* txt, void* u) { (void)t; (void)txt; ++((req_state*)u)->tokens_seen; return 0; }
static void batch_done(uint64_t id, const int32_t* toks, int64_t n, const char* reason, void* u) {
    req_state* r = (req_state*)u;
    r->id = id; r->n = n; r->done++;
    for (int64_t i = 0; i < n && i < 64; ++i) r->toks[i] = toks[i];
    snprintf(r->reason, sizeof r->reason, "%s", reason);
}
static int same(const int32_t* a, const int32_t* b, int64_t n) { return memcmp(a, b, (size_t)n * sizeof(int32_t)) == 0; }

/* a minimal .safetensors with one LoRA pair for `weight` ([out, in]) and lora_alpha */
static int write_lora(const char* path, const char* weight, int out_dim, int in_dim) {
    enum { R = 2 };
    char hdr[512];
    size_t na = (size_t)R * (size_t)in_dim * 4, nb = (size_t)out_dim * R * 4;
    int hl = snprintf(hdr, sizeof hdr,
        "{\"%s.lora_a\":{\"dtype\":\"F32\",\"shape\":[%d,%d],\"data_offsets\":[0,%zu]},"
        "\"%s.lora_b\":{\"dtype\":\"F32\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]},"
        "\"lora_alpha\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[%zu,%zu]}}",
        weight, R, in_dim, na, weight, out_dim, R, na, na + nb, na + nb, na + nb + 4);
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    uint64_t n = (uint64_t)hl;
    unsigned char le[8];
    for (int i = 0; i < 8; ++i) le[i] = (unsigned char)(n >> (8 * i));
    fwrite(le, 1, 8, f); fwrite(hdr, 1, (size_t)hl, f);
    unsigned s = 12345u;
    for (size_t i = 0; i < (na + nb) / 4; ++i) { s = s * 1664525u + 1013904223u; float v = ((float)(s >> 8) / 8388608.0f) - 1.0f; fwrite(&v, 4, 1, f); }
    float alpha = 8.0f; fwrite(&alpha, 4, 1, f);
    return fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    if (forge_abi_version() != FORGE_ABI_VERSION || FORGE_ABI_VERSION < 4) return 3;
    if (forge_open("/nonexistent.fpk", 0, 0) != NULL || forge_last_error()[0] == 0) { puts("error path broken"); return 4; }

    /* ---- ABI 1-3, unchanged: one handle owns model and session ---- */
    forge_model* m = forge_open(argv[1], 64, 1);
    if (!m) { printf("open failed: %s\n", forge_last_error()); return 1; }
    int32_t ids[64]; int64_t n = forge_tokenize(m, "the theory", 1, ids, 64);
    char buf[256]; forge_detokenize(m, ids + 1, n - 1, buf, sizeof buf);
    forge_sampling s = {0.f, 0, 1.f, 0.f, 1.f, 0};
    int32_t ref[8]; int calls = 0;
    int64_t g = forge_generate(m, ids, n, 8, &s, ref, 8, on_tok, &calls);
    printf("tokens %lld, roundtrip \"%s\", generated %lld (callbacks %d), vocab %lld\n", (long long)n, buf, (long long)g, calls, (long long)forge_vocab_size(m));
    CHECK(g == 8 && calls == 8 && n > 1 && strcmp(buf, "the theory") == 0);
    const int64_t vocab = forge_vocab_size(m);

    /* ---- shared weights: two sessions on one set of weights ---- */
    CHECK(forge_weights_open("/nonexistent.fpk", NULL) == NULL && strstr(forge_last_error(), "nonexistent") != NULL);
    forge_open_params op; memset(&op, 0, sizeof op); op.size = sizeof op; op.verify_hashes = 1; op.device = "cpu"; op.deterministic = 1;
    forge_weights* w = forge_weights_open(argv[1], &op);
    if (!w) { printf("weights open failed: %s\n", forge_last_error()); return 1; }
    CHECK(strcmp(forge_weights_device(w), "cpu") == 0);
    CHECK(forge_set_deterministic(1) == 1 && forge_set_deterministic(0) == 0 && forge_set_deterministic(-1) >= 0);
    const char* dev = forge_weights_set_device(w, "no-such-device");   /* unavailable: falls back to the CPU */
    CHECK(dev && strcmp(dev, "cpu") == 0);
    forge_session_params sp; memset(&sp, 0, sizeof sp); sp.size = sizeof sp; sp.context = 64; sp.kv_codec = "f32";
    forge_model* a = forge_session_open(w, &sp);
    forge_model* b = forge_session_open(w, NULL);
    CHECK(a && b && forge_context_size(a) == 64 && forge_context_size(b) > 64);
    CHECK(strcmp(forge_session_kind(a), "generative_sequence") == 0 && forge_vocab_size(a) == vocab);
    forge_weights* w2 = forge_model_weights(a);   /* another handle on the same weights */
    forge_weights_close(w2);
    char man[2048]; CHECK(forge_manifest_json(a, man, sizeof man) > 0 && strstr(man, "\"format\":\"fpk\"") != NULL);
    CHECK(forge_manifest_json(a, NULL, 0) == (int64_t)strlen(man));

    /* tokenizer details */
    CHECK(forge_token_bos(a) == ids[0] && forge_token_eos(a) >= 0 && forge_token_eos(a) != forge_token_bos(a));
    CHECK(forge_token_is_special(a, forge_token_eos(a)) == 1 && forge_token_is_special(a, 'a') == 0);
    char piece[32]; CHECK(forge_token_text(a, 'a', piece, sizeof piece) == 1 && strcmp(piece, "a") == 0);
    CHECK(forge_token_text(a, forge_token_eos(a), piece, sizeof piece) == 4 && forge_token_id(a, piece) == forge_token_eos(a));
    CHECK(forge_token_id(a, "no such token") == -1 && forge_token_text(a, 1 << 30, piece, sizeof piece) == -1);
    CHECK(strstr(forge_model_error(a), "vocabulary") != NULL && forge_model_error(b)[0] == 0);   /* the error stays with its handle */
    int32_t stops[8]; int64_t ns = forge_stop_tokens(a, 1, stops, 8);
    CHECK(ns == 1 && stops[0] == forge_token_eos(a));
    CHECK(forge_chat_template(a, buf, sizeof buf) == 0 && buf[0] == 0);   /* the golden model has no chat template */
    CHECK(forge_chat_template_apply_ex(a, "[{\"role\":\"user\",\"content\":\"hi\"}]", NULL, NULL, 1, buf, sizeof buf) == -1);

    /* ---- generation with the full parameter set ---- */
    forge_sampler_params greedy; memset(&greedy, 0, sizeof greedy); greedy.size = sizeof greedy;
    forge_generate_params gp; memset(&gp, 0, sizeof gp); gp.size = sizeof gp; gp.sampler = &greedy; gp.reset = 1; gp.ignore_eos = 1;
    int32_t out[32], out2[32]; int32_t why = -1;
    CHECK(forge_generate_v4(a, ids, n, 8, &gp, out, 32, &why) == 8 && same(out, ref, 8) && why == FORGE_STOP_LENGTH);
    CHECK(forge_generate_v4(b, ids, n, 8, NULL, out2, 32, NULL) >= 1);   /* NULL params: greedy, continues from position 0 */
    /* stop tokens end the output and are not written; the callback can stop it; position is where generation stood */
    gp.stop_tokens = &ref[3]; gp.n_stop_tokens = 1;
    int64_t k = forge_generate_v4(a, ids, n, 8, &gp, out, 32, &why);
    int first = 0; while (first < 8 && ref[first] != ref[3]) ++first;
    CHECK(k == first && why == FORGE_STOP_EOS && same(out, ref, k));
    gp.stop_tokens = NULL; gp.n_stop_tokens = 0; calls = 0; gp.on_token = stop_after_3; gp.user = &calls;
    CHECK(forge_generate_v4(a, ids, n, 8, &gp, out, 32, &why) == 3 && why == FORGE_STOP_CALLBACK && forge_position(a) == n + 2);
    gp.on_token = NULL;
    /* samplers: seeds repeat, penalties and logit bias act */
    forge_sampler_params sm = greedy; sm.temperature = 0.9f; sm.top_k = 40; sm.top_p = 0.95f; sm.seed = 7; sm.repetition_penalty = 1.1f; sm.repetition_window = 32;
    gp.sampler = &sm;
    int64_t k1 = forge_generate_v4(a, ids, n, 16, &gp, out, 32, NULL), k2 = forge_generate_v4(a, ids, n, 16, &gp, out2, 32, NULL);
    CHECK(k1 == 16 && k2 == 16 && same(out, out2, 16));
    sm.seed = 8; CHECK(forge_generate_v4(a, ids, n, 16, &gp, out2, 32, NULL) == 16 && !same(out, out2, 16));
    forge_sampler_params pen = greedy; pen.presence_penalty = 50.f; pen.frequency_penalty = 5.f; gp.sampler = &pen;
    CHECK(forge_generate_v4(a, ids, n, 12, &gp, out, 32, NULL) == 12);
    for (int i = 0; i < 12; ++i) for (int j = i + 1; j < 12; ++j) CHECK(out[i] != out[j]);   /* huge penalties forbid repeats */
    forge_logit_bias bias[2] = {{ref[0], -1e9f}, {'Z', 0.f}};
    forge_sampler_params lb = greedy; lb.logit_bias = bias; lb.n_logit_bias = 2; gp.sampler = &lb;
    CHECK(forge_generate_v4(a, ids, n, 4, &gp, out, 32, NULL) == 4 && out[0] != ref[0]);
    bias[1].bias = 1e9f;
    CHECK(forge_generate_v4(a, ids, n, 3, &gp, out, 32, NULL) == 3 && out[0] == 'Z' && out[1] == 'Z' && out[2] == 'Z');
    /* grammar and JSON Schema */
    gp.sampler = &sm; gp.grammar = "root ::= \"yes\" | \"no\"";
    k = forge_generate_v4(a, ids, n, 8, &gp, out, 32, &why);
    CHECK(k > 0 && why == FORGE_STOP_GRAMMAR && forge_detokenize(a, out, k, buf, sizeof buf) > 0 && (!strcmp(buf, "yes") || !strcmp(buf, "no")));
    gp.grammar = NULL; gp.json_schema = "{\"enum\":[\"red\",\"green\",7]}";
    k = forge_generate_v4(a, ids, n, 12, &gp, out, 32, &why);
    CHECK(k > 0 && forge_detokenize(a, out, k, buf, sizeof buf) > 0 && (!strcmp(buf, "\"red\"") || !strcmp(buf, "\"green\"") || !strcmp(buf, "7")));
    gp.grammar = "root ::= \"x\"";
    CHECK(forge_generate_v4(a, ids, n, 4, &gp, out, 32, NULL) == -1 && strstr(forge_model_error(a), "not both") != NULL);
    gp.grammar = NULL; gp.json_schema = NULL;
    char gbnf[2048]; int64_t gl = forge_json_schema_to_grammar("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}}}", gbnf, sizeof gbnf);
    CHECK(gl > 0 && gl < (int64_t)sizeof gbnf && strncmp(gbnf, "root ::= ", 9) == 0 && forge_grammar_check(gbnf) == 0);
    CHECK(forge_json_schema_to_grammar("{not json", gbnf, sizeof gbnf) == -1 && forge_grammar_check("root ::= \"open") == -1);
    /* a draft session on the same weights: speculative decoding returns the greedy tokens */
    gp.sampler = &greedy; gp.draft = b; gp.spec_k = 3; gp.ignore_eos = 0;
    k = forge_generate_v4(a, ids, n, 8, &gp, out, 32, NULL);
    CHECK(k >= 1 && k <= 8 && same(out, ref, k));
    gp.draft = NULL; gp.ignore_eos = 1;

    /* ---- KV state: save / load / copy ---- */
    forge_reset(a); forge_reset(b);
    float* la = malloc((size_t)vocab * sizeof(float)); float* lb2 = malloc((size_t)vocab * sizeof(float));
    CHECK(forge_forward(a, ids, n, la, vocab) == 0 && forge_position(a) == n && forge_state_bytes(a) > 0);
    int64_t need = forge_state_save(a, -1, NULL, 0);
    void* blob = malloc((size_t)need);
    CHECK(need > 16 && forge_state_save(a, -1, blob, need) == need);
    CHECK(forge_state_load(b, blob, need) == 0 && forge_position(b) == n);
    int32_t next = ref[0];
    CHECK(forge_forward(a, &next, 1, la, vocab) == 0 && forge_forward(b, &next, 1, lb2, vocab) == 0 && memcmp(la, lb2, (size_t)vocab * sizeof(float)) == 0);
    CHECK(forge_state_load(b, blob, need / 2) == -1 && forge_position(b) == 0);   /* truncated blob: refused, state cleared */
    CHECK(forge_state_load(b, "not a blob at all", 17) == -1);
    CHECK(forge_state_save(a, 1000, NULL, 0) == -1);
    CHECK(forge_state_copy(b, a, n) == 0 && forge_position(b) == n);              /* the first n tokens of a */
    CHECK(forge_forward(b, &next, 1, lb2, vocab) == 0 && memcmp(la, lb2, (size_t)vocab * sizeof(float)) == 0);
    CHECK(forge_state_copy(b, a, 1000) == -1 && forge_state_copy(a, a, 1) == -1 && forge_state_copy(b, m, 1) == -1);   /* m: other weights */
    free(blob);

    /* ---- LoRA adapters ---- */
    CHECK(forge_adapter_set(a, "missing") == -1 && forge_adapter_exists(w, "missing") == 0 && forge_adapter_get(a)[0] == 0);
    if (argc >= 4) {
        char lpath[1024]; snprintf(lpath, sizeof lpath, "%s/capi_lora.safetensors", argv[3]);
        CHECK(write_lora(lpath, "lm_head", (int)vocab, 64) == 0);
        CHECK(forge_adapter_load(w, "tuned", lpath) == 0 && forge_adapter_exists(w, "tuned") == 1);
        CHECK(forge_adapter_load(w, "tuned", lpath) == -1 && strstr(forge_weights_error(w), "tuned") != NULL);   /* names are immutable */
        CHECK(forge_adapter_load(w, "broken", "/nonexistent.safetensors") == -1);
        forge_reset(a); CHECK(forge_forward(a, ids, n, la, vocab) == 0);
        CHECK(forge_adapter_set(a, "tuned") == 0 && strcmp(forge_adapter_get(a), "tuned") == 0 && forge_position(a) == 0);
        CHECK(forge_forward(a, ids, n, lb2, vocab) == 0 && memcmp(la, lb2, (size_t)vocab * sizeof(float)) != 0);
        forge_reset(b); CHECK(forge_state_copy(b, a, n) == -1);                    /* different adapters */
        CHECK(forge_adapter_set(a, NULL) == 0 && forge_forward(a, ids, n, lb2, vocab) == 0 && memcmp(la, lb2, (size_t)vocab * sizeof(float)) == 0);
    }

    /* ---- batching: several requests on the shared weights ---- */
    forge_batch_params bp; memset(&bp, 0, sizeof bp); bp.size = sizeof bp; bp.max_active = 2;
    forge_batch* bt = forge_batch_open(w, &bp);
    CHECK(bt != NULL);
    req_state rs[5]; memset(rs, 0, sizeof rs);
    forge_batch_request rq; memset(&rq, 0, sizeof rq); rq.size = sizeof rq;
    rq.prompt = ids; rq.n_prompt = n; rq.max_new = 8; rq.on_token = batch_tok; rq.on_done = batch_done;
    forge_logit_bias no_eos = {forge_token_eos(a), -1e9f};
    forge_sampler_params bs = greedy; bs.logit_bias = &no_eos; bs.n_logit_bias = 1; rq.sampler = &bs;
    uint64_t id[5];
    rq.user = &rs[0]; id[0] = forge_batch_submit(bt, &rq);
    rq.user = &rs[1]; rq.grammar = "root ::= \"yes\" | \"no\""; id[1] = forge_batch_submit(bt, &rq); rq.grammar = NULL;
    rq.user = &rs[2]; rq.stop_tokens = &ref[3]; rq.n_stop_tokens = 1; id[2] = forge_batch_submit(bt, &rq); rq.stop_tokens = NULL; rq.n_stop_tokens = 0;
    rq.user = &rs[3]; rq.max_new = 40; rq.priority = -1; id[3] = forge_batch_submit(bt, &rq); rq.priority = 0; rq.max_new = 8;
    rq.user = &rs[4]; rq.adapter = argc >= 4 ? "tuned" : NULL; id[4] = forge_batch_submit(bt, &rq); rq.adapter = NULL;
    CHECK(id[0] > 0 && id[1] > id[0] && id[4] > id[3]);
    rq.adapter = "missing"; CHECK(forge_batch_submit(bt, &rq) == 0 && strstr(forge_batch_error(bt), "missing") != NULL); rq.adapter = NULL;
    rq.grammar = "root ::= \"open"; CHECK(forge_batch_submit(bt, &rq) == 0); rq.grammar = NULL;
    CHECK(forge_batch_cancel(bt, id[3]) == 1 && rs[3].done == 1 && strcmp(rs[3].reason, "cancelled") == 0 && forge_batch_cancel(bt, id[3]) == 0);
    CHECK(forge_batch_step(bt) == 1);
    CHECK(forge_batch_run(bt) == 0 && forge_batch_step(bt) == 0);
    CHECK(rs[0].done == 1 && rs[0].id == id[0] && rs[0].n == 8 && same(rs[0].toks, ref, 8) && strcmp(rs[0].reason, "length") == 0 && rs[0].tokens_seen == 8);
    CHECK(rs[1].done == 1 && strcmp(rs[1].reason, "stop") == 0 && forge_detokenize(a, rs[1].toks, rs[1].n, buf, sizeof buf) > 0 && (!strcmp(buf, "yes") || !strcmp(buf, "no")));
    CHECK(rs[2].done == 1 && strcmp(rs[2].reason, "stop") == 0 && rs[2].n == first + 1 && rs[2].toks[first] == ref[3]);
    CHECK(rs[4].done == 1 && rs[4].n == 8 && (argc >= 4 ? !same(rs[4].toks, ref, 8) : same(rs[4].toks, ref, 8)));
    forge_batch_stats st; memset(&st, 0, sizeof st); st.size = sizeof st;
    CHECK(forge_batch_stats_get(bt, &st) == 0 && st.completed == 5 && st.failed == 0 && st.waiting == 0 && st.active == 0 && st.peak_active == 2 && st.steps > 8);
    CHECK(st.prefix_hits >= 1);   /* the prompts are identical: later requests reuse the first one's KV rows */
    forge_batch_evict_prefix_cache(bt, 0);
    /* closing with work queued: the request is told it was cancelled */
    memset(&rs[0], 0, sizeof rs[0]); rq.user = &rs[0]; rq.max_new = 30;
    CHECK(forge_batch_submit(bt, &rq) > 0 && forge_batch_step(bt) == 1);
    forge_batch_close(bt);
    CHECK(rs[0].done == 1 && strcmp(rs[0].reason, "cancelled") == 0 && rs[0].n >= 1);

    /* weights may be closed before their sessions */
    forge_weights_close(w);
    forge_reset(a); CHECK(forge_generate_v4(a, ids, n, 8, NULL, out, 32, NULL) >= 1 && same(out, ref, 1));
    CHECK(forge_embed_dim(a) == -1 && forge_embed_text(a, "x", NULL, la, vocab, NULL) == -1 && strstr(forge_model_error(a), "embedding") != NULL);
    forge_close(a); forge_close(b); forge_close(m);
    free(la); free(lb2);

    /* ---- embeddings ---- */
    if (argc >= 3) {
        forge_weights* ew = forge_weights_open(argv[2], NULL);
        forge_model* e = ew ? forge_session_open(ew, NULL) : NULL;
        if (!e) { printf("embedding model open failed: %s\n", forge_last_error()); return 1; }
        CHECK(strcmp(forge_session_kind(e), "embedding") == 0);
        int64_t dim = forge_embed_dim(e);
        CHECK(dim > 0);
        float* v1 = malloc((size_t)dim * 3 * sizeof(float)); float* v2 = malloc((size_t)dim * 3 * sizeof(float));
        int64_t nt = 0;
        CHECK(forge_embed_text(e, "Hello, world!", NULL, v1, dim, &nt) == dim && nt >= 3);
        int32_t tids[64]; int64_t tn = forge_tokenize(e, "Hello, world!", -1, tids, 64);   /* -1: with [CLS] / [SEP] */
        CHECK(tn == nt && forge_embed_tokens(e, tids, tn, NULL, v2, dim) == dim && memcmp(v1, v2, (size_t)dim * sizeof(float)) == 0);
        double norm = 0; for (int64_t i = 0; i < dim; ++i) norm += (double)v1[i] * v1[i];
        CHECK(norm > 0);
        const char* texts[3] = {"The quick brown fox jumps over the lazy dog.", "Hello, world!", "it's John's"};
        CHECK(forge_embed_texts(e, texts, 3, v2, dim * 3) == dim && memcmp(v1, v2 + dim, (size_t)dim * sizeof(float)) == 0);
        CHECK(forge_embed_text(e, texts[2], NULL, v1, dim, NULL) == dim && memcmp(v1, v2 + 2 * dim, (size_t)dim * sizeof(float)) == 0);
        CHECK(forge_embed_text(e, "a", "b", v1, dim, &nt) == dim && nt >= 5);   /* a pair: [CLS] a [SEP] b [SEP] */
        const char* bad[2] = {"ok", NULL};
        CHECK(forge_embed_texts(e, bad, 2, v2, dim * 2) == -1 && forge_embed_tokens(e, tids, 0, NULL, v2, dim) == -1);
        CHECK(forge_generate_v4(e, tids, tn, 4, NULL, out, 32, NULL) == -1 && forge_batch_open(ew, NULL) == NULL);
        free(v1); free(v2);
        forge_close(e); forge_weights_close(ew);
    }

    /* ---- ahead-of-time plans: a sidecar, picked up by the next open ---- */
    if (argc >= 4) {
        char copy[1024], side[1100]; snprintf(copy, sizeof copy, "%s/capi_aot.fpk", argv[3]); snprintf(side, sizeof side, "%s.fplan", copy);
        FILE* in = fopen(argv[1], "rb"); FILE* o = fopen(copy, "wb");
        if (in && o) { char io[65536]; size_t r; while ((r = fread(io, 1, sizeof io, in)) > 0) fwrite(io, 1, r, o); }
        if (in) fclose(in);
        if (o) fclose(o);
        int32_t lens[3] = {1, 3, 4};
        CHECK(forge_aot_precompile(copy, NULL, lens, 3) == 3);
        FILE* sf = fopen(side, "rb"); CHECK(sf != NULL); if (sf) fclose(sf);
        forge_model* pm = forge_open(copy, 64, 0);
        CHECK(pm != NULL && forge_generate(pm, ids, n, 8, &s, out, 8, NULL, NULL) == 8 && same(out, ref, 8));   /* same tokens with precompiled plans */
        forge_close(pm);
        CHECK(forge_aot_precompile("/nonexistent.fpk", NULL, NULL, 0) == -1);
        remove(side); remove(copy);
    }
    /* NULL handles are errors, not crashes */
    CHECK(forge_vocab_size(NULL) == -1 && forge_session_open(NULL, NULL) == NULL && forge_batch_submit(NULL, NULL) == 0 && forge_model_error(NULL)[0] == 0);
    forge_close(NULL); forge_weights_close(NULL); forge_batch_close(NULL);
    printf("capi_smoke: %s (ABI %u)\n", fails ? "FAILED" : "ok", forge_abi_version());
    return fails ? 1 : 0;
}
