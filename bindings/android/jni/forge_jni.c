/* JNI bridge: dev.forge.Forge* (Kotlin, bindings/android/forge) -> the stable C ABI in forge.h (ABI 4).
   Built by CMake with -DFORGE_JNI=ON (Android NDK or any host with jni.h); the `jni_syntax` ctest compiles it.

   Handles cross the boundary as jlong. Text crosses as UTF-8 byte arrays, never as JNI "modified UTF-8" strings:
   GetStringUTFChars / NewStringUTF encode characters outside the BMP as surrogate pairs and reject plain 4-byte
   UTF-8, and a token's text may be part of a character. The Kotlin side encodes and decodes.

   All natives live in dev.forge.ForgeNative (a Kotlin object with @JvmStatic externals). */
#include <jni.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "forge.h"

#define NAT(name) Java_dev_forge_ForgeNative_##name
#define M(h) ((forge_model*)(intptr_t)(h))
#define W(h) ((forge_weights*)(intptr_t)(h))
#define B(h) ((forge_batch*)(intptr_t)(h))

static void throw_msg(JNIEnv* env, const char* what, const char* e) {
    if ((*env)->ExceptionCheck(env)) return;   /* a Java exception (from a listener) is already on its way */
    jclass c = (*env)->FindClass(env, "java/lang/IllegalStateException");
    if (!c) return;
    char buf[768]; snprintf(buf, sizeof buf, "%s: %s", what, e && *e ? e : "unknown error");
    /* the message may hold bytes that are not modified UTF-8 (a path, a token): keep ASCII, replace the rest */
    for (char* p = buf; *p; ++p) if ((unsigned char)*p >= 0x80 || (unsigned char)*p < 0x20) *p = '?';
    (*env)->ThrowNew(env, c, buf);
}
static void throw_forge(JNIEnv* env, const char* what) { throw_msg(env, what, forge_last_error()); }

/* a NUL-terminated copy of a byte array (NULL for a null array); free() it */
static char* dup_bytes(JNIEnv* env, jbyteArray a) {
    if (!a) return NULL;
    jsize n = (*env)->GetArrayLength(env, a);
    char* s = malloc((size_t)n + 1);
    if (!s) return NULL;
    (*env)->GetByteArrayRegion(env, a, 0, n, (jbyte*)s);
    s[n] = 0;
    return s;
}
static jbyteArray new_bytes(JNIEnv* env, const char* p, int64_t n) {
    jbyteArray r = (*env)->NewByteArray(env, (jsize)n);
    if (r && n > 0) (*env)->SetByteArrayRegion(env, r, 0, (jsize)n, (const jbyte*)p);
    return r;
}
static jintArray new_ints(JNIEnv* env, const int32_t* p, int64_t n) {
    jintArray r = (*env)->NewIntArray(env, (jsize)n);
    if (r && n > 0) (*env)->SetIntArrayRegion(env, r, 0, (jsize)n, (const jint*)p);
    return r;
}
static jfloatArray new_floats(JNIEnv* env, const float* p, int64_t n) {
    jfloatArray r = (*env)->NewFloatArray(env, (jsize)n);
    if (r && n > 0) (*env)->SetFloatArrayRegion(env, r, 0, (jsize)n, p);
    return r;
}
/* a malloc'd copy of an int array (NULL for null or empty); *n gets the length */
static int32_t* dup_ints(JNIEnv* env, jintArray a, int64_t* n) {
    *n = a ? (*env)->GetArrayLength(env, a) : 0;
    if (*n == 0) return NULL;
    int32_t* p = malloc((size_t)*n * sizeof(int32_t));
    if (p) (*env)->GetIntArrayRegion(env, a, 0, (jsize)*n, (jint*)p); else *n = 0;
    return p;
}

/* ---- library ---- */
JNIEXPORT jstring JNICALL NAT(version)(JNIEnv* env, jclass c) { (void)c; return (*env)->NewStringUTF(env, forge_version()); }
JNIEXPORT jint JNICALL NAT(abi)(JNIEnv* env, jclass c) { (void)env; (void)c; return (jint)forge_abi_version(); }
JNIEXPORT jint JNICALL NAT(setDeterministic)(JNIEnv* env, jclass c, jint mode) { (void)env; (void)c; return (jint)forge_set_deterministic((int)mode); }
JNIEXPORT jbyteArray JNICALL NAT(jsonSchemaToGrammar)(JNIEnv* env, jclass c, jbyteArray schema) {
    (void)c; char* s = dup_bytes(env, schema);
    int64_t need = forge_json_schema_to_grammar(s, NULL, 0);
    if (need < 0) { free(s); throw_forge(env, "forge_json_schema_to_grammar"); return NULL; }
    char* out = malloc((size_t)need + 1);
    if (out) forge_json_schema_to_grammar(s, out, need + 1);
    free(s);
    jbyteArray r = out ? new_bytes(env, out, need) : NULL; free(out); return r;
}
JNIEXPORT void JNICALL NAT(grammarCheck)(JNIEnv* env, jclass c, jbyteArray grammar) {
    (void)c; char* g = dup_bytes(env, grammar);
    if (forge_grammar_check(g) != 0) throw_forge(env, "grammar");
    free(g);
}
JNIEXPORT jlong JNICALL NAT(aotPrecompile)(JNIEnv* env, jclass c, jbyteArray path, jbyteArray out, jintArray lengths) {
    (void)c; char* p = dup_bytes(env, path); char* o = dup_bytes(env, out);
    int64_t n = 0; int32_t* l = dup_ints(env, lengths, &n);
    int64_t k = forge_aot_precompile(p, o, l, n);
    free(p); free(o); free(l);
    if (k < 0) throw_forge(env, "forge_aot_precompile");
    return (jlong)k;
}

/* ---- weights ---- */
JNIEXPORT jlong JNICALL NAT(weightsOpen)(JNIEnv* env, jclass c, jbyteArray path, jboolean verify, jbyteArray device, jint deterministic, jboolean no_aot) {
    (void)c; char* p = dup_bytes(env, path); char* d = dup_bytes(env, device);
    forge_open_params op; memset(&op, 0, sizeof op);
    op.size = sizeof op; op.verify_hashes = verify ? 1 : 0; op.device = d; op.deterministic = (int32_t)deterministic; op.no_aot = no_aot ? 1 : 0;
    forge_weights* w = forge_weights_open(p, &op);
    free(p); free(d);
    if (!w) throw_forge(env, "forge_weights_open");
    return (jlong)(intptr_t)w;
}
JNIEXPORT void JNICALL NAT(weightsClose)(JNIEnv* env, jclass c, jlong w) { (void)env; (void)c; if (w) forge_weights_close(W(w)); }
JNIEXPORT jlong JNICALL NAT(modelWeights)(JNIEnv* env, jclass c, jlong h) {
    (void)c; forge_weights* w = forge_model_weights(M(h));
    if (!w) throw_forge(env, "forge_model_weights");
    return (jlong)(intptr_t)w;
}
JNIEXPORT jstring JNICALL NAT(weightsDevice)(JNIEnv* env, jclass c, jlong w) {
    (void)c; const char* d = forge_weights_device(W(w));
    if (!d) { throw_forge(env, "forge_weights_device"); return NULL; }
    return (*env)->NewStringUTF(env, d);   /* device names are ASCII */
}
JNIEXPORT jstring JNICALL NAT(weightsSetDevice)(JNIEnv* env, jclass c, jlong w, jbyteArray device) {
    (void)c; char* d = dup_bytes(env, device);
    const char* r = forge_weights_set_device(W(w), d);
    free(d);
    if (!r) { throw_msg(env, "forge_weights_set_device", forge_weights_error(W(w))); return NULL; }
    return (*env)->NewStringUTF(env, r);
}
JNIEXPORT void JNICALL NAT(adapterLoad)(JNIEnv* env, jclass c, jlong w, jbyteArray name, jbyteArray path) {
    (void)c; char* n = dup_bytes(env, name); char* p = dup_bytes(env, path);
    if (forge_adapter_load(W(w), n, p) != 0) throw_msg(env, "forge_adapter_load", forge_weights_error(W(w)));
    free(n); free(p);
}
JNIEXPORT jboolean JNICALL NAT(adapterExists)(JNIEnv* env, jclass c, jlong w, jbyteArray name) {
    (void)c; char* n = dup_bytes(env, name);
    int r = forge_adapter_exists(W(w), n);
    free(n); return r ? JNI_TRUE : JNI_FALSE;
}

/* ---- sessions ---- */
JNIEXPORT jlong JNICALL NAT(open)(JNIEnv* env, jclass c, jbyteArray path, jlong ctx, jboolean verify) {
    (void)c; char* p = dup_bytes(env, path);
    forge_model* m = forge_open(p, (int64_t)ctx, verify ? 1 : 0);
    free(p);
    if (!m) throw_forge(env, "forge_open");
    return (jlong)(intptr_t)m;
}
JNIEXPORT jlong JNICALL NAT(sessionOpen)(JNIEnv* env, jclass c, jlong w, jlong ctx, jbyteArray kv_codec, jboolean kv_offload) {
    (void)c; char* k = dup_bytes(env, kv_codec);
    forge_session_params sp; memset(&sp, 0, sizeof sp);
    sp.size = sizeof sp; sp.context = (int64_t)ctx; sp.kv_codec = k; sp.kv_offload = kv_offload ? 1 : 0;
    forge_model* m = forge_session_open(W(w), &sp);
    free(k);
    if (!m) throw_forge(env, "forge_session_open");
    return (jlong)(intptr_t)m;
}
JNIEXPORT void JNICALL NAT(close)(JNIEnv* env, jclass c, jlong h) { (void)env; (void)c; if (h) forge_close(M(h)); }
JNIEXPORT void JNICALL NAT(reset)(JNIEnv* env, jclass c, jlong h) { (void)env; (void)c; forge_reset(M(h)); }
JNIEXPORT jlong JNICALL NAT(vocab)(JNIEnv* env, jclass c, jlong h) { (void)env; (void)c; return (jlong)forge_vocab_size(M(h)); }
JNIEXPORT jlong JNICALL NAT(position)(JNIEnv* env, jclass c, jlong h) { (void)env; (void)c; return (jlong)forge_position(M(h)); }
JNIEXPORT jlong JNICALL NAT(contextSize)(JNIEnv* env, jclass c, jlong h) { (void)env; (void)c; return (jlong)forge_context_size(M(h)); }
JNIEXPORT void JNICALL NAT(rewind)(JNIEnv* env, jclass c, jlong h, jlong pos) {
    (void)c; if (forge_rewind(M(h), (int64_t)pos) != 0) throw_msg(env, "forge_rewind", forge_model_error(M(h))); }
JNIEXPORT jstring JNICALL NAT(kind)(JNIEnv* env, jclass c, jlong h) {
    (void)c; const char* k = forge_session_kind(M(h));
    if (!k) { throw_forge(env, "forge_session_kind"); return NULL; }
    return (*env)->NewStringUTF(env, k);   /* an identifier: ASCII */
}

/* text getters: size the buffer, read, return UTF-8 bytes */
typedef int64_t (*text_fn)(const forge_model*, char*, int64_t);
static jbyteArray model_text(JNIEnv* env, jlong h, text_fn f, const char* what) {
    int64_t need = f(M(h), NULL, 0);
    if (need < 0) { throw_msg(env, what, forge_model_error(M(h))); return NULL; }
    char* out = malloc((size_t)need + 1);
    if (!out) return NULL;
    f(M(h), out, need + 1);
    jbyteArray r = new_bytes(env, out, need); free(out); return r;
}
JNIEXPORT jbyteArray JNICALL NAT(manifest)(JNIEnv* env, jclass c, jlong h) { (void)c; return model_text(env, h, forge_manifest_json, "forge_manifest_json"); }
JNIEXPORT jbyteArray JNICALL NAT(chatTemplate)(JNIEnv* env, jclass c, jlong h) { (void)c; return model_text(env, h, forge_chat_template, "forge_chat_template"); }
JNIEXPORT jbyteArray JNICALL NAT(applyChatTemplate)(JNIEnv* env, jclass c, jlong h, jbyteArray messages, jbyteArray tools, jbyteArray extra, jboolean add_gen) {
    (void)c; char* m = dup_bytes(env, messages); char* t = dup_bytes(env, tools); char* x = dup_bytes(env, extra);
    jbyteArray r = NULL;
    int64_t need = forge_chat_template_apply_ex(M(h), m, t, x, add_gen ? 1 : 0, NULL, 0);
    if (need < 0) throw_msg(env, "forge_chat_template_apply", forge_model_error(M(h)));
    else {
        char* out = malloc((size_t)need + 1);
        if (out) { forge_chat_template_apply_ex(M(h), m, t, x, add_gen ? 1 : 0, out, need + 1); r = new_bytes(env, out, need); free(out); }
    }
    free(m); free(t); free(x);
    return r;
}

/* ---- tokenizer ---- */
JNIEXPORT jintArray JNICALL NAT(tokenize)(JNIEnv* env, jclass c, jlong h, jbyteArray text, jint add_bos) {
    (void)c; char* t = dup_bytes(env, text);
    int64_t n = forge_tokenize(M(h), t, (int)add_bos, NULL, 0);
    if (n < 0) { free(t); throw_msg(env, "forge_tokenize", forge_model_error(M(h))); return NULL; }
    int32_t* ids = malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
    if (ids) forge_tokenize(M(h), t, (int)add_bos, ids, n);
    free(t);
    jintArray r = ids ? new_ints(env, ids, n) : NULL; free(ids); return r;
}
JNIEXPORT jbyteArray JNICALL NAT(detokenize)(JNIEnv* env, jclass c, jlong h, jintArray ids) {
    (void)c; int64_t n = 0; int32_t* p = dup_ints(env, ids, &n);
    int64_t need = forge_detokenize(M(h), p, n, NULL, 0);
    if (need < 0) { free(p); throw_msg(env, "forge_detokenize", forge_model_error(M(h))); return NULL; }
    char* out = malloc((size_t)need + 1);
    if (out) forge_detokenize(M(h), p, n, out, need + 1);
    free(p);
    jbyteArray r = out ? new_bytes(env, out, need) : NULL; free(out); return r;
}
JNIEXPORT jint JNICALL NAT(tokenBos)(JNIEnv* env, jclass c, jlong h) { (void)env; (void)c; return (jint)forge_token_bos(M(h)); }
JNIEXPORT jint JNICALL NAT(tokenEos)(JNIEnv* env, jclass c, jlong h) { (void)env; (void)c; return (jint)forge_token_eos(M(h)); }
JNIEXPORT jboolean JNICALL NAT(tokenIsSpecial)(JNIEnv* env, jclass c, jlong h, jint id) { (void)env; (void)c; return forge_token_is_special(M(h), (int32_t)id) ? JNI_TRUE : JNI_FALSE; }
JNIEXPORT jint JNICALL NAT(tokenId)(JNIEnv* env, jclass c, jlong h, jbyteArray token) {
    (void)c; char* t = dup_bytes(env, token);
    int32_t id = forge_token_id(M(h), t);
    free(t); return (jint)id;
}
JNIEXPORT jbyteArray JNICALL NAT(tokenBytes)(JNIEnv* env, jclass c, jlong h, jint id) {
    (void)c; char small[64];
    int64_t need = forge_token_text(M(h), (int32_t)id, small, sizeof small);
    if (need < 0) { throw_msg(env, "forge_token_text", forge_model_error(M(h))); return NULL; }
    if (need < (int64_t)sizeof small) return new_bytes(env, small, need);
    char* out = malloc((size_t)need + 1);
    if (!out) return NULL;
    forge_token_text(M(h), (int32_t)id, out, need + 1);
    jbyteArray r = new_bytes(env, out, need); free(out); return r;
}
JNIEXPORT jintArray JNICALL NAT(stopTokens)(JNIEnv* env, jclass c, jlong h, jboolean chat) {
    (void)c; int32_t ids[64];
    int64_t n = forge_stop_tokens(M(h), chat ? 1 : 0, ids, 64);
    if (n < 0) { throw_msg(env, "forge_stop_tokens", forge_model_error(M(h))); return NULL; }
    return new_ints(env, ids, n > 64 ? 64 : n);
}

/* ---- forward passes ---- */
static jfloatArray forward_impl(JNIEnv* env, jlong h, jintArray ids, int all) {
    int64_t v = forge_vocab_size(M(h)), n = 0;
    int32_t* p = dup_ints(env, ids, &n);
    if (v <= 0 || n <= 0) { free(p); throw_msg(env, "forge_forward", n <= 0 ? "no tokens" : forge_model_error(M(h))); return NULL; }
    int64_t count = all ? v * n : v;
    float* lg = malloc((size_t)count * sizeof(float));
    int rc = !lg ? -1 : all ? forge_forward_all(M(h), p, n, lg, count) : forge_forward(M(h), p, n, lg, count);
    free(p);
    if (rc != 0) { free(lg); throw_msg(env, "forge_forward", lg ? forge_model_error(M(h)) : "out of memory"); return NULL; }
    jfloatArray r = new_floats(env, lg, count); free(lg); return r;
}
JNIEXPORT jfloatArray JNICALL NAT(forward)(JNIEnv* env, jclass c, jlong h, jintArray ids) { (void)c; return forward_impl(env, h, ids, 0); }
JNIEXPORT jfloatArray JNICALL NAT(forwardAll)(JNIEnv* env, jclass c, jlong h, jintArray ids) { (void)c; return forward_impl(env, h, ids, 1); }

/* ---- sampler: float[11] {temperature, top_p, min_p, typical_p, mirostat_tau, mirostat_eta, repetition_penalty,
   presence_penalty, frequency_penalty, top_k, repetition_window} + seed + parallel bias arrays ---- */
typedef struct { forge_sampler_params s; forge_logit_bias* bias; } sampler_box;
static int sampler_fill(JNIEnv* env, sampler_box* b, jfloatArray params, jlong seed, jintArray bias_tokens, jfloatArray bias_values) {
    memset(b, 0, sizeof *b); b->s.size = sizeof b->s;
    jfloat f[11]; memset(f, 0, sizeof f);
    if (params) { jsize n = (*env)->GetArrayLength(env, params); (*env)->GetFloatArrayRegion(env, params, 0, n < 11 ? n : 11, f); }
    b->s.temperature = f[0]; b->s.top_p = f[1]; b->s.min_p = f[2]; b->s.typical_p = f[3]; b->s.mirostat_tau = f[4]; b->s.mirostat_eta = f[5];
    b->s.repetition_penalty = f[6]; b->s.presence_penalty = f[7]; b->s.frequency_penalty = f[8]; b->s.top_k = (int32_t)f[9]; b->s.repetition_window = (int32_t)f[10];
    b->s.seed = (uint64_t)seed;
    jsize nb = bias_tokens && bias_values ? (*env)->GetArrayLength(env, bias_tokens) : 0;
    if (nb > 0 && (*env)->GetArrayLength(env, bias_values) == nb) {
        b->bias = malloc((size_t)nb * sizeof(forge_logit_bias));
        if (!b->bias) return -1;
        jint* t = (*env)->GetIntArrayElements(env, bias_tokens, NULL); jfloat* v = (*env)->GetFloatArrayElements(env, bias_values, NULL);
        if (t && v) for (jsize i = 0; i < nb; ++i) { b->bias[i].token = (int32_t)t[i]; b->bias[i].bias = v[i]; }
        if (t) (*env)->ReleaseIntArrayElements(env, bias_tokens, t, JNI_ABORT);
        if (v) (*env)->ReleaseFloatArrayElements(env, bias_values, v, JNI_ABORT);
        b->s.logit_bias = b->bias; b->s.n_logit_bias = nb;
    }
    return 0;
}

/* ---- token callbacks: ForgeNative.TokenSink.onTokenBytes(int, byte[]) -> boolean (true = stop) ---- */
typedef struct { JNIEnv* env; jobject sink; jmethodID m; } cbctx;
static int on_token(int32_t tok, const char* text, void* u) {
    cbctx* c = u;
    if (!c->sink) return 0;
    jbyteArray s = new_bytes(c->env, text ? text : "", text ? (int64_t)strlen(text) : 0);
    jboolean stop = (*c->env)->CallBooleanMethod(c->env, c->sink, c->m, (jint)tok, s);
    if (s) (*c->env)->DeleteLocalRef(c->env, s);
    return (*c->env)->ExceptionCheck(c->env) || stop;
}
static jmethodID sink_method(JNIEnv* env, jobject sink) {
    jclass k = (*env)->GetObjectClass(env, sink);
    return (*env)->GetMethodID(env, k, "onTokenBytes", "(I[B)Z");
}

/* ABI 1 generation (kept: ForgeModel.generateLegacy): EOS is part of the result */
JNIEXPORT jintArray JNICALL NAT(generate)(JNIEnv* env, jclass c, jlong h, jintArray prompt, jint max_new,
        jfloat temp, jint top_k, jfloat top_p, jfloat min_p, jfloat rep, jlong seed, jobject sink) {
    (void)c; forge_sampling s = {temp, top_k, top_p, min_p, rep, (uint64_t)seed};
    int64_t n = 0; int32_t* p = dup_ints(env, prompt, &n);
    int32_t cap = max_new > 0 ? max_new : 1;
    int32_t* out = malloc((size_t)cap * sizeof(int32_t));
    cbctx cc = {env, sink, NULL};
    if (sink) cc.m = sink_method(env, sink);
    int64_t k = !out || (sink && !cc.m) ? -1 : forge_generate(M(h), p, n, max_new, &s, out, cap, sink ? on_token : NULL, &cc);
    free(p);
    if ((*env)->ExceptionCheck(env)) { free(out); return NULL; }   /* a listener threw: no JNI calls until it is delivered */
    if (k < 0) { free(out); throw_msg(env, "forge_generate", forge_model_error(M(h))); return NULL; }
    jintArray r = new_ints(env, out, k < cap ? k : cap); free(out); return r;
}

/* ABI 4 generation. stop_reason_out: int[1] or null. */
JNIEXPORT jintArray JNICALL NAT(generateV4)(JNIEnv* env, jclass c, jlong h, jintArray prompt, jint max_new,
        jfloatArray sampler, jlong seed, jintArray bias_tokens, jfloatArray bias_values, jbyteArray grammar, jbyteArray json_schema,
        jlong draft, jint spec_k, jintArray stop_tokens, jboolean ignore_eos, jboolean reset, jobject sink, jintArray stop_reason_out) {
    (void)c;
    sampler_box sb;
    if (sampler_fill(env, &sb, sampler, seed, bias_tokens, bias_values) != 0) { throw_msg(env, "forge_generate", "out of memory"); return NULL; }
    int64_t n = 0, ns = 0; int32_t* p = dup_ints(env, prompt, &n); int32_t* st = dup_ints(env, stop_tokens, &ns);
    char* g = dup_bytes(env, grammar); char* js = dup_bytes(env, json_schema);
    int32_t cap = max_new > 0 ? max_new : 1;
    int32_t* out = malloc((size_t)cap * sizeof(int32_t));
    cbctx cc = {env, sink, NULL};
    if (sink) cc.m = sink_method(env, sink);
    forge_generate_params gp; memset(&gp, 0, sizeof gp);
    gp.size = sizeof gp; gp.spec_k = (int32_t)spec_k; gp.sampler = &sb.s; gp.grammar = g; gp.json_schema = js; gp.draft = M(draft);
    gp.stop_tokens = st; gp.n_stop_tokens = ns; gp.ignore_eos = ignore_eos ? 1 : 0; gp.reset = reset ? 1 : 0;
    gp.on_token = sink ? on_token : NULL; gp.user = &cc;
    int32_t why = 0;
    int64_t k = !out || (sink && !cc.m) ? -1 : forge_generate_v4(M(h), p, n, max_new, &gp, out, cap, &why);
    free(p); free(st); free(g); free(js); free(sb.bias);
    if ((*env)->ExceptionCheck(env)) { free(out); return NULL; }   /* a listener threw: no JNI calls until it is delivered */
    if (k < 0) { free(out); throw_msg(env, "forge_generate", forge_model_error(M(h))); return NULL; }
    if (stop_reason_out && (*env)->GetArrayLength(env, stop_reason_out) > 0) { jint w = (jint)why; (*env)->SetIntArrayRegion(env, stop_reason_out, 0, 1, &w); }
    jintArray r = new_ints(env, out, k < cap ? k : cap); free(out); return r;
}

/* ---- adapters, KV state ---- */
JNIEXPORT void JNICALL NAT(adapterSet)(JNIEnv* env, jclass c, jlong h, jbyteArray name) {
    (void)c; char* n = dup_bytes(env, name);
    if (forge_adapter_set(M(h), n) != 0) throw_msg(env, "forge_adapter_set", forge_model_error(M(h)));
    free(n);
}
JNIEXPORT jbyteArray JNICALL NAT(adapterGet)(JNIEnv* env, jclass c, jlong h) {
    (void)c; const char* a = forge_adapter_get(M(h));
    return new_bytes(env, a ? a : "", a ? (int64_t)strlen(a) : 0);
}
JNIEXPORT jbyteArray JNICALL NAT(stateSave)(JNIEnv* env, jclass c, jlong h, jlong n_tokens) {
    (void)c; int64_t need = forge_state_save(M(h), (int64_t)n_tokens, NULL, 0);
    if (need < 0 || need > 0x7fffffff) { throw_msg(env, "forge_state_save", need < 0 ? forge_model_error(M(h)) : "the state is larger than a Java array (2 GiB)"); return NULL; }
    void* buf = malloc((size_t)(need > 0 ? need : 1));
    if (!buf) { throw_msg(env, "forge_state_save", "out of memory"); return NULL; }
    forge_state_save(M(h), (int64_t)n_tokens, buf, need);
    jbyteArray r = new_bytes(env, buf, need); free(buf); return r;
}
JNIEXPORT void JNICALL NAT(stateLoad)(JNIEnv* env, jclass c, jlong h, jbyteArray blob) {
    (void)c; if (!blob) { throw_msg(env, "forge_state_load", "null state"); return; }
    jsize n = (*env)->GetArrayLength(env, blob); jbyte* p = (*env)->GetByteArrayElements(env, blob, NULL);
    int rc = p ? forge_state_load(M(h), p, n) : -1;
    if (p) (*env)->ReleaseByteArrayElements(env, blob, p, JNI_ABORT);
    if (rc != 0) throw_msg(env, "forge_state_load", forge_model_error(M(h)));
}
JNIEXPORT void JNICALL NAT(stateCopy)(JNIEnv* env, jclass c, jlong dst, jlong src, jlong n_tokens) {
    (void)c; if (forge_state_copy(M(dst), M(src), (int64_t)n_tokens) != 0) throw_msg(env, "forge_state_copy", forge_model_error(M(dst))); }
JNIEXPORT jlong JNICALL NAT(stateBytes)(JNIEnv* env, jclass c, jlong h) { (void)env; (void)c; return (jlong)forge_state_bytes(M(h)); }

/* ---- embeddings ---- */
JNIEXPORT jlong JNICALL NAT(embedDim)(JNIEnv* env, jclass c, jlong h) {
    (void)c; int64_t d = forge_embed_dim(M(h));
    if (d < 0) throw_msg(env, "forge_embed_dim", forge_model_error(M(h)));
    return (jlong)d;
}
JNIEXPORT jfloatArray JNICALL NAT(embedTokens)(JNIEnv* env, jclass c, jlong h, jintArray ids, jintArray type_ids) {
    (void)c; int64_t n = 0, nt = 0, dim = forge_embed_dim(M(h));
    if (dim < 0) { throw_msg(env, "forge_embed_tokens", forge_model_error(M(h))); return NULL; }
    int32_t* p = dup_ints(env, ids, &n); int32_t* t = dup_ints(env, type_ids, &nt);
    float* out = malloc((size_t)(dim > 0 ? dim : 1) * sizeof(float));
    int64_t k = !out || (t && nt != n) ? -1 : forge_embed_tokens(M(h), p, n, t, out, dim);
    free(p); free(t);
    if (k != dim) { free(out); throw_msg(env, "forge_embed_tokens", t && nt != n ? "token types must have one entry per token" : forge_model_error(M(h))); return NULL; }
    jfloatArray r = new_floats(env, out, dim); free(out); return r;
}
JNIEXPORT jfloatArray JNICALL NAT(embedText)(JNIEnv* env, jclass c, jlong h, jbyteArray text, jbyteArray pair) {
    (void)c; int64_t dim = forge_embed_dim(M(h));
    if (dim < 0) { throw_msg(env, "forge_embed_text", forge_model_error(M(h))); return NULL; }
    char* t = dup_bytes(env, text); char* p = dup_bytes(env, pair);
    float* out = malloc((size_t)(dim > 0 ? dim : 1) * sizeof(float));
    int64_t k = out ? forge_embed_text(M(h), t, p, out, dim, NULL) : -1;
    free(t); free(p);
    if (k != dim) { free(out); throw_msg(env, "forge_embed_text", forge_model_error(M(h))); return NULL; }
    jfloatArray r = new_floats(env, out, dim); free(out); return r;
}
/* texts: byte[][]; returns texts.length * dim floats, row i for texts[i] */
JNIEXPORT jfloatArray JNICALL NAT(embedTexts)(JNIEnv* env, jclass c, jlong h, jobjectArray texts) {
    (void)c; int64_t dim = forge_embed_dim(M(h));
    jsize n = texts ? (*env)->GetArrayLength(env, texts) : 0;
    if (dim < 0 || n <= 0) { throw_msg(env, "forge_embed_texts", dim < 0 ? forge_model_error(M(h)) : "no texts"); return NULL; }
    char** t = calloc((size_t)n, sizeof(char*));
    float* out = malloc((size_t)n * (size_t)(dim > 0 ? dim : 1) * sizeof(float));
    int ok = t && out;
    for (jsize i = 0; ok && i < n; ++i) {
        jbyteArray a = (jbyteArray)(*env)->GetObjectArrayElement(env, texts, i);
        t[i] = dup_bytes(env, a);
        if (a) (*env)->DeleteLocalRef(env, a);
    }
    int64_t k = ok ? forge_embed_texts(M(h), (const char* const*)t, n, out, (int64_t)n * dim) : -1;
    if (t) for (jsize i = 0; i < n; ++i) free(t[i]);
    free(t);
    if (k != dim) { free(out); throw_msg(env, "forge_embed_texts", ok ? forge_model_error(M(h)) : "out of memory"); return NULL; }
    jfloatArray r = new_floats(env, out, (int64_t)n * dim); free(out); return r;
}

/* ---- batching ----
   A request's Java callbacks are global references held in a reqctx until its done callback has run (the C ABI
   calls on_done exactly once per accepted request, also for cancelled ones and at forge_batch_close). Callbacks run
   on the thread that calls batchStep / batchRun / batchCancel / batchClose, which is a Java thread, so its JNIEnv
   is fetched from the VM at call time. No JNI call may be made while a Java exception is pending, so once a
   listener has thrown, the rest of that step / run winds down: the requests still running are cancelled and their
   done callbacks are not delivered; the exception surfaces when the native call returns. */
typedef struct { JavaVM* vm; jobject sink; jobject done; jmethodID on_token, on_done; } reqctx;
static JNIEnv* env_of(JavaVM* vm) {
    JNIEnv* env = NULL;
    return (*vm)->GetEnv(vm, (void**)&env, JNI_VERSION_1_6) == JNI_OK ? env : NULL;
}
static int batch_token(int32_t tok, const char* text, void* u) {
    reqctx* r = u; JNIEnv* env = env_of(r->vm);
    if (!env || !r->sink) return 0;
    if ((*env)->ExceptionCheck(env)) return 1;   /* an earlier callback threw: wind the request down */
    jbyteArray s = new_bytes(env, text ? text : "", text ? (int64_t)strlen(text) : 0);
    jboolean stop = (*env)->CallBooleanMethod(env, r->sink, r->on_token, (jint)tok, s);
    if (s) (*env)->DeleteLocalRef(env, s);
    return (*env)->ExceptionCheck(env) || stop;
}
static void batch_done(uint64_t id, const int32_t* toks, int64_t n, const char* reason, void* u) {
    reqctx* r = u; JNIEnv* env = env_of(r->vm);
    if (env) {
        if (r->done && !(*env)->ExceptionCheck(env)) {
            jintArray t = new_ints(env, toks, n);
            jbyteArray why = new_bytes(env, reason ? reason : "", reason ? (int64_t)strlen(reason) : 0);
            if (t && why) (*env)->CallVoidMethod(env, r->done, r->on_done, (jlong)id, t, why);
            if (t) (*env)->DeleteLocalRef(env, t);
            if (why) (*env)->DeleteLocalRef(env, why);
        }
        if (r->sink) (*env)->DeleteGlobalRef(env, r->sink);
        if (r->done) (*env)->DeleteGlobalRef(env, r->done);
    }
    free(r);
}
JNIEXPORT jlong JNICALL NAT(batchOpen)(JNIEnv* env, jclass c, jlong w, jint max_active, jlong ctx, jlong prefill_chunk, jint prefix_cache, jbyteArray kv_codec) {
    (void)c; char* k = dup_bytes(env, kv_codec);
    forge_batch_params bp; memset(&bp, 0, sizeof bp);
    bp.size = sizeof bp; bp.max_active = (int32_t)max_active; bp.context = (int64_t)ctx; bp.prefill_chunk = (int64_t)prefill_chunk;
    bp.prefix_cache = (int32_t)prefix_cache; bp.kv_codec = k;
    forge_batch* b = forge_batch_open(W(w), &bp);
    free(k);
    if (!b) throw_forge(env, "forge_batch_open");
    return (jlong)(intptr_t)b;
}
JNIEXPORT void JNICALL NAT(batchClose)(JNIEnv* env, jclass c, jlong b) { (void)env; (void)c; if (b) forge_batch_close(B(b)); }
/* sink: ForgeNative.TokenSink or null; done: ForgeNative.DoneSink (onDoneBytes(long, int[], byte[])) or null */
JNIEXPORT jlong JNICALL NAT(batchSubmit)(JNIEnv* env, jclass c, jlong b, jintArray prompt, jint max_new, jfloatArray sampler, jlong seed,
        jintArray bias_tokens, jfloatArray bias_values, jbyteArray grammar, jbyteArray json_schema, jintArray stop_tokens, jbyteArray adapter,
        jint priority, jobject sink, jobject done) {
    (void)c;
    sampler_box sb;
    reqctx* r = calloc(1, sizeof *r);
    if (!r || sampler_fill(env, &sb, sampler, seed, bias_tokens, bias_values) != 0) { free(r); throw_msg(env, "forge_batch_submit", "out of memory"); return 0; }
    (*env)->GetJavaVM(env, &r->vm);
    if (sink) { r->on_token = sink_method(env, sink); r->sink = (*env)->NewGlobalRef(env, sink); }
    if (done) { r->on_done = (*env)->GetMethodID(env, (*env)->GetObjectClass(env, done), "onDoneBytes", "(J[I[B)V"); r->done = (*env)->NewGlobalRef(env, done); }
    uint64_t id = 0;
    if (!(*env)->ExceptionCheck(env) && (!sink || (r->sink && r->on_token)) && (!done || (r->done && r->on_done))) {
        int64_t n = 0, ns = 0; int32_t* p = dup_ints(env, prompt, &n); int32_t* st = dup_ints(env, stop_tokens, &ns);
        char* g = dup_bytes(env, grammar); char* js = dup_bytes(env, json_schema); char* ad = dup_bytes(env, adapter);
        forge_batch_request rq; memset(&rq, 0, sizeof rq);
        rq.size = sizeof rq; rq.max_new = (int32_t)max_new; rq.prompt = p; rq.n_prompt = n; rq.sampler = &sb.s; rq.grammar = g; rq.json_schema = js;
        rq.stop_tokens = st; rq.n_stop_tokens = ns; rq.adapter = ad; rq.priority = (int32_t)priority;
        rq.on_token = sink ? batch_token : NULL; rq.on_done = batch_done; rq.user = r;
        id = forge_batch_submit(B(b), &rq);   /* copies the prompt, grammar, stop tokens and sampler */
        free(p); free(st); free(g); free(js); free(ad);
    }
    free(sb.bias);
    if (!id) {   /* not accepted: batch_done will never run for it */
        if (r->sink) (*env)->DeleteGlobalRef(env, r->sink);
        if (r->done) (*env)->DeleteGlobalRef(env, r->done);
        free(r);
        throw_msg(env, "forge_batch_submit", forge_batch_error(B(b)));
    }
    return (jlong)id;
}
JNIEXPORT jboolean JNICALL NAT(batchCancel)(JNIEnv* env, jclass c, jlong b, jlong id) { (void)env; (void)c; return forge_batch_cancel(B(b), (uint64_t)id) == 1 ? JNI_TRUE : JNI_FALSE; }
JNIEXPORT jboolean JNICALL NAT(batchStep)(JNIEnv* env, jclass c, jlong b) {
    (void)c; int r = forge_batch_step(B(b));
    if (r < 0) throw_msg(env, "forge_batch_step", forge_batch_error(B(b)));
    return r == 1 ? JNI_TRUE : JNI_FALSE;
}
JNIEXPORT void JNICALL NAT(batchRun)(JNIEnv* env, jclass c, jlong b) {
    (void)c; if (forge_batch_run(B(b)) != 0) throw_msg(env, "forge_batch_run", forge_batch_error(B(b))); }
/* long[9]: steps, completed, failed, prefix_hits, prefix_tokens_reused, peak_active, peak_kv_bytes, waiting, active */
JNIEXPORT jlongArray JNICALL NAT(batchStats)(JNIEnv* env, jclass c, jlong b) {
    (void)c; forge_batch_stats s; memset(&s, 0, sizeof s); s.size = sizeof s;
    if (forge_batch_stats_get(B(b), &s) != 0) { throw_msg(env, "forge_batch_stats_get", forge_batch_error(B(b))); return NULL; }
    jlong v[9] = {(jlong)s.steps, (jlong)s.completed, (jlong)s.failed, (jlong)s.prefix_hits, (jlong)s.prefix_tokens_reused, (jlong)s.peak_active,
                  (jlong)s.peak_kv_bytes, (jlong)s.waiting, (jlong)s.active};
    jlongArray r = (*env)->NewLongArray(env, 9);
    if (r) (*env)->SetLongArrayRegion(env, r, 0, 9, v);
    return r;
}
JNIEXPORT void JNICALL NAT(batchEvictPrefixCache)(JNIEnv* env, jclass c, jlong b, jlong keep) { (void)env; (void)c; forge_batch_evict_prefix_cache(B(b), (int64_t)keep); }
