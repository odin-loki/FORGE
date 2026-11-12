/* forge.h — stable C ABI for embedding FORGE (FUSE, Cypha, anything). ABI-versioned; all handles opaque. */
#ifndef FORGE_H
#define FORGE_H
#include <stddef.h>
#include <stdint.h>
/* FORGE_API marks the exported functions. Building libforge defines FORGE_C_BUILD; link a static copy with
   FORGE_C_STATIC. On Windows the functions are dllexport/dllimport (MSVC exports nothing by default). */
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(FORGE_C_BUILD)
#    define FORGE_API __declspec(dllexport)
#  elif defined(FORGE_C_STATIC)
#    define FORGE_API
#  else
#    define FORGE_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define FORGE_API __attribute__((visibility("default")))
#else
#  define FORGE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define FORGE_ABI_VERSION 4   /* 2 = v1 + position/rewind/forward_all/generate_ex; 3 = + generate_ex2 / forge_sampling_ext; 4 = + shared weights,
                                 full sampler chain, adapters, KV state, embeddings, batching, tokenizer details (all additive) */

typedef struct forge_model forge_model;

typedef struct forge_sampling {
    float temperature;        /* 0 = greedy */
    int32_t top_k;            /* 0 = off */
    float top_p;              /* 1 = off */
    float min_p;              /* 0 = off */
    float repetition_penalty; /* 1 = off */
    uint64_t seed;
} forge_sampling;

/* Returns non-zero to stop generation. */
typedef int (*forge_token_cb)(int32_t token, const char* text, void* user);

FORGE_API uint32_t forge_abi_version(void);
FORGE_API const char* forge_version(void);
FORGE_API const char* forge_last_error(void);          /* thread-local; valid until the next call on this thread */

FORGE_API forge_model* forge_open(const char* fpk_path, int64_t context, int verify_hashes);
FORGE_API void forge_close(forge_model* m);
FORGE_API void forge_reset(forge_model* m);             /* clear KV state */
FORGE_API int64_t forge_vocab_size(const forge_model* m);

/* Tokenizer. Return token count, or -1 on error; writes at most cap tokens. */
FORGE_API int64_t forge_tokenize(const forge_model* m, const char* text, int add_bos, int32_t* out, int64_t cap);
/* Writes NUL-terminated text, returns bytes needed (excluding NUL) or -1. */
FORGE_API int64_t forge_detokenize(const forge_model* m, const int32_t* ids, int64_t n, char* out, int64_t cap);
/* Renders the model's own chat template over messages_json, a JSON array [{"role": "user", "content": "..."}, ...].
   Writes NUL-terminated text, returns bytes needed (excluding NUL), or -1 (no template in the file, bad JSON, or the
   template rejects the conversation). Tokenize the result with add_bos = 0: the template writes BOS itself. */
FORGE_API int64_t forge_chat_template_apply(const forge_model* m, const char* messages_json, int add_generation_prompt, char* out, int64_t cap);

/* Runs tokens continuing from the current position; writes last-position logits (vocab floats). */
FORGE_API int forge_forward(forge_model* m, const int32_t* ids, int64_t n, float* logits, int64_t cap);

/* Prefill + decode. Returns number of generated tokens written to out (or -1). */
FORGE_API int64_t forge_generate(forge_model* m, const int32_t* prompt, int64_t n_prompt, int32_t max_new,
                       const forge_sampling* s, int32_t* out, int64_t cap, forge_token_cb cb, void* user);

/* ---- ABI 2 ---- */
FORGE_API int64_t forge_position(const forge_model* m);
FORGE_API int forge_rewind(forge_model* m, int64_t position);  /* roll KV state back to `position` tokens */
/* Logits for every position of ids: writes n * vocab floats. */
FORGE_API int forge_forward_all(forge_model* m, const int32_t* ids, int64_t n, float* logits, int64_t cap);

typedef struct forge_generate_options {
    const char* grammar;   /* GBNF text constraining the output, or NULL */
    forge_model* draft;    /* speculative-decoding draft model (same vocab), or NULL */
    int32_t spec_k;        /* draft tokens per round (default 4) */
} forge_generate_options;
FORGE_API int64_t forge_generate_ex(forge_model* m, const int32_t* prompt, int64_t n_prompt, int32_t max_new,
                          const forge_sampling* s, const forge_generate_options* o,
                          int32_t* out, int64_t cap, forge_token_cb cb, void* user);

/* ABI 3: extra samplers. Size-prefixed so fields can be appended later: set size = sizeof(forge_sampling_ext). */
typedef struct forge_sampling_ext {
    uint32_t size;
    float typical_p;          /* locally typical sampling; 1 (or 0) = off */
    float mirostat_tau;       /* Mirostat v2 target surprise (bits); 0 = off */
    float mirostat_eta;       /* Mirostat learning rate; 0 = default 0.1 */
} forge_sampling_ext;
FORGE_API int64_t forge_generate_ex2(forge_model* m, const int32_t* prompt, int64_t n_prompt, int32_t max_new,
                           const forge_sampling* s, const forge_generate_options* o, const forge_sampling_ext* ext,
                           int32_t* out, int64_t cap, forge_token_cb cb, void* user);

/* ============================================================================================================
   ABI 4 (additive: every symbol and struct above is unchanged).

   Handles
     forge_weights   one opened model file: mapped weights, tokenizer, chat template, compiled plans, LoRA adapters,
                     device. Reference-counted inside the library: sessions keep it alive, so it may be closed
                     while sessions made from it are still in use.
     forge_model     one sequence (KV state, position, selected adapter). forge_open() makes one with private
                     weights; forge_session_open() makes one on shared weights. Every function above that takes a
                     forge_model* works on both kinds.
     forge_batch     a continuous-batching scheduler over shared weights.

   Thread safety, per function, is given as one of:
     [any]      callable from any thread at any time.
     [handle]   not safe to call concurrently with another call on the SAME handle; calls on different handles may
                run in parallel (the engine serialises graph runs on shared weights internally).
     [setup]    changes shared weights: must not run while any session or batch on those weights is inside a call.
   Every function of ABI 1-3 that takes a forge_model* is [handle]; forge_abi_version, forge_version and
   forge_last_error are [any].

   Errors: as before a failing call returns NULL / -1 and forge_last_error() (thread-local) has the message. The
   message is also stored in the handle the call was made on: forge_model_error / forge_weights_error /
   forge_batch_error return it until the next failing call on that handle (they are [handle]).

   Structs that start with `uint32_t size` are size-prefixed: set size = sizeof(the struct); fields past `size` are
   taken as zero, so a caller built against an older header keeps working. A NULL struct pointer means all defaults.
   Buffers: functions that write text or arrays return the size needed (text: bytes without the NUL); they write at
   most `cap` and may be called with cap 0 (and a NULL buffer) to size the buffer.
   ============================================================================================================ */

typedef struct forge_weights forge_weights;
typedef struct forge_batch forge_batch;

/* ---- weights, devices, determinism ---- */
typedef struct forge_open_params {
    uint32_t size;
    int32_t verify_hashes;     /* 1 = check the file's BLAKE3 hashes on open */
    const char* device;        /* "cpu", "auto", "vulkan", "cuda", "metal", ... ; NULL = $FORGE_DEVICE, else cpu */
    int32_t deterministic;     /* 0 = follow $FORGE_DETERMINISTIC, 1 = on, 2 = off. Process-wide: see forge_set_deterministic */
    int32_t no_aot;            /* 1 = compile every plan here even when the file carries precompiled ones */
} forge_open_params;
FORGE_API forge_weights* forge_weights_open(const char* fpk_path, const forge_open_params* p);   /* [any] */
FORGE_API void forge_weights_close(forge_weights* w);                                            /* [handle]; sessions stay valid */
/* A new handle on the weights a session uses (close it with forge_weights_close). */
FORGE_API forge_weights* forge_model_weights(forge_model* m);                                    /* [handle] */
FORGE_API const char* forge_weights_error(const forge_weights* w);                               /* [handle] */
/* The device in use ("cpu" when the requested one is unavailable). The pointer is valid until the next call on w. */
FORGE_API const char* forge_weights_device(forge_weights* w);                                    /* [handle] */
/* Select the device ("auto" scores the devices that open for this model). Returns the device actually used, or
   NULL. Clears the compiled plans. */
FORGE_API const char* forge_weights_set_device(forge_weights* w, const char* device);            /* [setup] */
/* Process-wide determinism mode (same device + same build -> bit-identical results): 1 on, 0 off, -1 follow
   $FORGE_DETERMINISTIC again. Returns the mode now in effect (0 or 1). */
FORGE_API int forge_set_deterministic(int mode);                                                 /* [any] */

/* ---- sessions on shared weights ---- */
typedef struct forge_session_params {
    uint32_t size;
    int32_t kv_offload;        /* 1 = keep KV on the host (drop device copies after each attention); 0 = $FORGE_KV_OFFLOAD */
    int64_t context;           /* tokens of KV state; 0 = the model's maximum */
    const char* kv_codec;      /* "f32", "int8", "fp8", "tbq4", ... ; NULL = $FORGE_KV_CACHE_TYPE, else f32 */
} forge_session_params;
FORGE_API forge_model* forge_session_open(forge_weights* w, const forge_session_params* p);      /* [any]; close with forge_close */
FORGE_API const char* forge_model_error(const forge_model* m);                                   /* [handle] */
FORGE_API int64_t forge_context_size(const forge_model* m);                                      /* [handle] */
/* "generative_sequence", "embedding", "one_shot", ... (the blueprint's session kind). Static lifetime per model. */
FORGE_API const char* forge_session_kind(const forge_model* m);                                  /* [handle] */
/* The manifest of the model file as JSON text. */
FORGE_API int64_t forge_manifest_json(const forge_model* m, char* out, int64_t cap);             /* [handle] */

/* ---- tokenizer details ---- */
FORGE_API int32_t forge_token_bos(const forge_model* m);                                         /* [handle]; -1 = none */
FORGE_API int32_t forge_token_eos(const forge_model* m);                                         /* [handle]; -1 = none */
/* The bytes token `id` adds to a text (what forge_token_cb gets); may be part of a UTF-8 sequence. */
FORGE_API int64_t forge_token_text(const forge_model* m, int32_t id, char* out, int64_t cap);    /* [handle] */
/* The id of a vocabulary entry given exactly as stored ("<|im_end|>", "\xC4\xA0the"), or -1. */
FORGE_API int32_t forge_token_id(const forge_model* m, const char* token);                       /* [handle] */
FORGE_API int forge_token_is_special(const forge_model* m, int32_t id);                          /* [handle]; 1 / 0 */
/* Token ids that end an assistant turn: EOS, the manifest's eos_token_ids, and (chat != 0) the end-of-turn markers
   of the chat template (<|im_end|>, <|eot_id|>, ...). Returns the count; writes at most cap ids. */
FORGE_API int64_t forge_stop_tokens(const forge_model* m, int chat, int32_t* out, int64_t cap);  /* [handle] */
/* The model's chat template source (Jinja); returns 0 and writes "" when the file has none. */
FORGE_API int64_t forge_chat_template(const forge_model* m, char* out, int64_t cap);             /* [handle] */
/* forge_chat_template_apply with tools (an OpenAI "tools" JSON array, or NULL) and extra template variables (a JSON
   object such as {"enable_thinking": false}, or NULL). */
FORGE_API int64_t forge_chat_template_apply_ex(const forge_model* m, const char* messages_json, const char* tools_json,
                                               const char* extra_json, int add_generation_prompt, char* out, int64_t cap);   /* [handle] */

/* ---- sampling and generation ---- */
typedef struct forge_logit_bias { int32_t token; float bias; } forge_logit_bias;
/* The whole sampler chain: bias -> repetition / presence / frequency penalties -> grammar -> temperature -> top-k
   -> top-p -> min-p -> typical -> draw (or Mirostat v2 in place of the truncation steps). Zero in a field means its
   documented default, so a zeroed struct with `size` set is greedy decoding. */
typedef struct forge_sampler_params {
    uint32_t size;
    float temperature;         /* 0 = greedy */
    int32_t top_k;             /* 0 = off */
    float top_p;               /* 0 or 1 = off */
    float min_p;               /* 0 = off */
    float typical_p;           /* 0 or 1 = off */
    float mirostat_tau;        /* 0 = off */
    float mirostat_eta;        /* 0 = 0.1 */
    float repetition_penalty;  /* 0 or 1 = off */
    float presence_penalty;    /* 0 = off: subtracted once from tokens seen in the window */
    float frequency_penalty;   /* 0 = off: subtracted per occurrence in the window */
    int32_t repetition_window; /* tokens of history the three penalties look at; 0 = 64 */
    uint64_t seed;
    const forge_logit_bias* logit_bias;   /* added to raw logits first; NULL = none */
    int64_t n_logit_bias;
} forge_sampler_params;

typedef struct forge_generate_params {
    uint32_t size;
    int32_t spec_k;                        /* draft tokens per round with `draft` (0 = 4) */
    const forge_sampler_params* sampler;   /* NULL = greedy */
    const char* grammar;                   /* GBNF text, or NULL */
    const char* json_schema;               /* a JSON Schema (text): output is compact JSON matching it. Not with `grammar` */
    forge_model* draft;                    /* speculative-decoding draft session (same vocabulary), or NULL */
    const int32_t* stop_tokens;            /* generation ends when one is sampled; like EOS it is not written to `out` */
    int64_t n_stop_tokens;
    int32_t ignore_eos;                    /* 1 = EOS does not end generation (it is then written like any token) */
    int32_t reset;                         /* 1 = clear the KV state first (a fresh sequence) */
    forge_token_cb on_token;               /* called for every token written to `out`; non-zero return stops */
    void* user;
} forge_generate_params;
/* Prefill + decode from the current KV position. Returns the number of tokens generated (writes at most cap), or
   -1. *stop_reason, when not NULL, is set to one of the FORGE_STOP_* values. Unlike forge_generate (ABI 1-3), the
   token that ends generation (EOS, a stop token) is neither written to `out` nor passed to on_token. */
#define FORGE_STOP_LENGTH 0     /* max_new tokens, or the context is full */
#define FORGE_STOP_EOS 1        /* EOS or a stop token */
#define FORGE_STOP_CALLBACK 2   /* on_token returned non-zero */
#define FORGE_STOP_GRAMMAR 3    /* the grammar is complete (or rejected the sampled token) */
FORGE_API int64_t forge_generate_v4(forge_model* m, const int32_t* prompt, int64_t n_prompt, int32_t max_new,
                                    const forge_generate_params* p, int32_t* out, int64_t cap, int32_t* stop_reason);   /* [handle] (m and p->draft) */
/* JSON Schema -> GBNF text (the grammar forge_generate_params.json_schema uses), for callers that cache it. */
FORGE_API int64_t forge_json_schema_to_grammar(const char* schema_json, char* out, int64_t cap);   /* [any] */
/* 0 when `grammar` parses as GBNF, else -1 with the parse error in forge_last_error(). */
FORGE_API int forge_grammar_check(const char* grammar);                                            /* [any] */

/* ---- LoRA adapters ---- */
/* Load an adapter under `name` from an .fpk or .safetensors holding <weight>.lora_a / <weight>.lora_b tensors.
   Names are immutable once loaded. 0 or -1. */
FORGE_API int forge_adapter_load(forge_weights* w, const char* name, const char* path);            /* [setup] */
FORGE_API int forge_adapter_exists(forge_weights* w, const char* name);                            /* [handle]; 1 / 0 */
/* Select an adapter for this session (NULL or "" = the base weights). Clears the session's KV state when the
   adapter changes. 0, or -1 for a name that is not loaded. */
FORGE_API int forge_adapter_set(forge_model* m, const char* name);                                 /* [handle] */
/* The session's adapter ("" = base). Valid until the next forge_adapter_set on m. */
FORGE_API const char* forge_adapter_get(const forge_model* m);                                     /* [handle] */

/* ---- KV state: save, load, share ---- */
/* Serialise the KV state of the first n_tokens tokens (-1 = all) into `out`. Returns the bytes needed, or -1 (a
   model with recurrent state can only save its full length). The blob is tied to the model file and KV codec. */
FORGE_API int64_t forge_state_save(forge_model* m, int64_t n_tokens, void* out, int64_t cap);      /* [handle] */
/* Replace the session's state with a saved one; the session then continues exactly where the saved one stood.
   The session must use the adapter the state was computed with. 0 or -1 (bad blob, other codec, too long). */
FORGE_API int forge_state_load(forge_model* m, const void* data, int64_t n);                       /* [handle] */
/* Copy the first n_tokens tokens of src's state into dst (prefix sharing). Both sessions must come from the same
   forge_weights, use the same KV codec and the same adapter. 0 or -1. */
FORGE_API int forge_state_copy(forge_model* dst, forge_model* src, int64_t n_tokens);              /* [handle] (both) */
FORGE_API int64_t forge_state_bytes(const forge_model* m);                                         /* [handle]; KV bytes held */

/* ---- embeddings (models whose forge_session_kind is "embedding") ---- */
/* Values per embedding (the model's output size for one sequence), or -1. */
FORGE_API int64_t forge_embed_dim(const forge_model* m);                                           /* [handle] */
/* One vector from token ids as they are (no special tokens added). type_ids may be NULL. Returns the number of
   floats (writes at most cap), or -1. */
FORGE_API int64_t forge_embed_tokens(forge_model* m, const int32_t* ids, int64_t n, const int32_t* type_ids, float* out, int64_t cap);   /* [handle] */
/* One vector from text through the model's tokenizer ([CLS] ... [SEP] and token types from its post-processor).
   pair may be NULL (a second sequence: query / passage). *n_tokens, when not NULL, gets the tokens fed. */
FORGE_API int64_t forge_embed_text(forge_model* m, const char* text, const char* pair, float* out, int64_t cap, int64_t* n_tokens);       /* [handle] */
/* n texts in one call: writes n * forge_embed_dim floats, row i for texts[i]. A loop with shared setup (not one
   batched pass): every text is tokenized and checked before any is run. Returns floats per row, or -1. */
FORGE_API int64_t forge_embed_texts(forge_model* m, const char* const* texts, int64_t n, float* out, int64_t cap);                        /* [handle] */

/* ---- continuous batching ---- */
typedef struct forge_batch_params {
    uint32_t size;
    int32_t max_active;        /* sequences decoded together; 0 = 8 */
    int64_t context;           /* KV context per sequence; 0 = the model's maximum */
    int64_t prefill_chunk;     /* prompt tokens per step per sequence; 0 = 256 */
    int32_t prefix_cache;      /* finished sequences kept for prefix reuse; 0 = 4, -1 = none */
    int32_t reserved;
    const char* kv_codec;      /* as forge_session_params */
} forge_batch_params;
/* Called once per request when it ends, on the thread that runs forge_batch_step / forge_batch_run (or, for a
   waiting request that is cancelled, inside forge_batch_cancel). reason: "stop", "length", "grammar", "cancelled"
   or "error: <message>". The token that ended the sequence (EOS, a stop token) is the last of `tokens`.
   tokens / reason are valid during the call only. */
typedef void (*forge_done_cb)(uint64_t request_id, const int32_t* tokens, int64_t n, const char* reason, void* user);
typedef struct forge_batch_request {
    uint32_t size;
    int32_t max_new;                       /* 0 = 64 */
    const int32_t* prompt;                 /* copied by forge_batch_submit */
    int64_t n_prompt;
    const forge_sampler_params* sampler;   /* NULL = greedy */
    const char* grammar;                   /* GBNF, or NULL */
    const char* json_schema;               /* or NULL; not with `grammar` */
    const int32_t* stop_tokens;            /* in addition to EOS */
    int64_t n_stop_tokens;
    const char* adapter;                   /* LoRA adapter name, NULL = base */
    int32_t priority;                      /* higher is admitted first */
    int32_t reserved;
    forge_token_cb on_token;               /* non-zero return cancels the request */
    forge_done_cb on_done;
    void* user;
} forge_batch_request;
typedef struct forge_batch_stats {
    uint32_t size; uint32_t reserved;
    uint64_t steps, completed, failed, prefix_hits, prefix_tokens_reused, peak_active, peak_kv_bytes, waiting, active;
} forge_batch_stats;
FORGE_API forge_batch* forge_batch_open(forge_weights* w, const forge_batch_params* p);            /* [any] */
/* Requests still waiting or running get on_done("cancelled") first. */
FORGE_API void forge_batch_close(forge_batch* b);                                                  /* [handle] */
FORGE_API const char* forge_batch_error(const forge_batch* b);                                     /* [handle] */
/* Queue a request; returns its id (> 0), or 0 on error. Nothing runs until forge_batch_step / forge_batch_run.
   May be called from other threads and from on_token; not from on_done. */
FORGE_API uint64_t forge_batch_submit(forge_batch* b, const forge_batch_request* r);               /* [any] */
/* Cancel a waiting or running request: 1 = cancelled (its on_done gets "cancelled"), 0 = unknown or finished. */
FORGE_API int forge_batch_cancel(forge_batch* b, uint64_t request_id);                             /* [any] */
/* One scheduling step: admit waiting requests, advance every active sequence by one token. Callbacks run on the
   calling thread. 1 = work was done, 0 = idle, -1 = error. */
FORGE_API int forge_batch_step(forge_batch* b);                                                    /* [handle] */
/* Step until nothing is waiting or running. 0 or -1. */
FORGE_API int forge_batch_run(forge_batch* b);                                                     /* [handle] */
FORGE_API int forge_batch_stats_get(forge_batch* b, forge_batch_stats* out);                       /* [handle]; set out->size first */
/* Prefix cache control: drop cached prefixes down to `keep` entries (applied at the next step). */
FORGE_API void forge_batch_evict_prefix_cache(forge_batch* b, int64_t keep);                       /* [any] */

/* ---- ahead-of-time plans (`forge aot` in plan mode) ---- */
/* Compile the model's CPU plans for the given sequence lengths (NULL / 0 = the defaults) and write them to
   out_path, or to "<fpk_path>.fplan" (the sidecar a later open picks up) when out_path is NULL. Returns the number
   of plans written, or -1. Not for signed models: a sidecar is not covered by the signature and is ignored. */
FORGE_API int64_t forge_aot_precompile(const char* fpk_path, const char* out_path, const int32_t* seq_lengths, int64_t n);   /* [any] */


#ifdef __cplusplus
}
#endif
#endif
