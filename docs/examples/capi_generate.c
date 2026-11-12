/* docs/capi.md example: the C ABI end to end. Usage: doc_example_capi_generate model.fpk */
#include <forge.h>
#include <stdio.h>
#include <string.h>

static int on_token(int32_t token, const char* text, void* user) {
    (void)token; (void)text; ++*(int*)user;
    return 0;                                   /* non-zero stops generation */
}
static void on_done(uint64_t id, const int32_t* tokens, int64_t n, const char* reason, void* user) {
    (void)tokens;
    printf("  request %llu: %lld tokens, %s\n", (unsigned long long)id, (long long)n, reason);
    ++*(int*)user;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    if (forge_abi_version() < 4) return 1;

    /* ABI 1: one handle that owns the weights and one sequence */
    forge_model* m = forge_open(argv[1], 256, 0);
    if (!m) { fprintf(stderr, "open failed: %s\n", forge_last_error()); return 1; }
    int32_t prompt[64], out[16];
    int64_t n = forge_tokenize(m, "Hello", 1, prompt, 64);
    forge_sampling s = {0.f, 0, 1.f, 0.f, 1.f, 42};   /* temperature, top_k, top_p, min_p, rep_pen, seed */
    int seen = 0;
    int64_t got = forge_generate(m, prompt, n, 16, &s, out, 16, on_token, &seen);
    char text[256];
    forge_detokenize(m, out, got, text, sizeof text);
    printf("forge %s abi %u: %lld tokens (%d callbacks), position %lld\n", forge_version(), forge_abi_version(),
           (long long)got, seen, (long long)forge_position(m));
    forge_close(m);
    if (!(got > 0 && seen == got)) return 1;

    /* ABI 4: weights opened once, any number of sessions on them */
    forge_open_params op = {0};
    op.size = sizeof op; op.device = "cpu";             /* or "auto"; NULL follows $FORGE_DEVICE */
    forge_weights* w = forge_weights_open(argv[1], &op);
    if (!w) { fprintf(stderr, "open failed: %s\n", forge_last_error()); return 1; }
    forge_model* a = forge_session_open(w, NULL);
    forge_model* b = forge_session_open(w, NULL);
    if (!a || !b) { fprintf(stderr, "session: %s\n", forge_weights_error(w)); return 1; }

    /* the full sampler chain, a JSON Schema constraint and a stop reason */
    forge_sampler_params sp = {0};
    sp.size = sizeof sp; sp.temperature = 0.8f; sp.top_k = 40; sp.top_p = 0.95f; sp.seed = 7;
    sp.repetition_penalty = 1.1f; sp.presence_penalty = 0.2f; sp.frequency_penalty = 0.1f;
    forge_generate_params gp = {0};
    gp.size = sizeof gp; gp.sampler = &sp; gp.reset = 1;
    gp.json_schema = "{\"enum\":[\"yes\",\"no\"]}";
    int32_t why = 0;
    got = forge_generate_v4(a, prompt, n, 16, &gp, out, 16, &why);
    if (got < 0) { fprintf(stderr, "generate: %s\n", forge_model_error(a)); return 1; }
    forge_detokenize(a, out, got, text, sizeof text);
    printf("constrained: %s (stop reason %d)\n", text, why);
    if (strcmp(text, "\"yes\"") != 0 && strcmp(text, "\"no\"") != 0) return 1;

    /* share a computed prefix with the second session instead of running the prompt again */
    if (forge_state_copy(b, a, n) != 0) { fprintf(stderr, "copy: %s\n", forge_model_error(b)); return 1; }
    printf("session b starts at position %lld\n", (long long)forge_position(b));

    /* continuous batching over the same weights */
    forge_batch* batch = forge_batch_open(w, NULL);
    forge_batch_request rq = {0};
    rq.size = sizeof rq; rq.prompt = prompt; rq.n_prompt = n; rq.max_new = 8; rq.on_done = on_done;
    int finished = 0; rq.user = &finished;
    for (int i = 0; i < 3; ++i) if (!forge_batch_submit(batch, &rq)) { fprintf(stderr, "submit: %s\n", forge_batch_error(batch)); return 1; }
    forge_batch_run(batch);
    forge_batch_close(batch);

    forge_close(a); forge_close(b);
    forge_weights_close(w);
    return finished == 3 ? 0 : 1;
}
