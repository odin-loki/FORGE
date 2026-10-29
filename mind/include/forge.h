/* forge.h — stable C ABI for embedding FORGE (FUSE, Cypha, anything). ABI-versioned; all handles opaque. */
#ifndef FORGE_H
#define FORGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define FORGE_ABI_VERSION 1

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

uint32_t forge_abi_version(void);
const char* forge_version(void);
const char* forge_last_error(void);          /* thread-local; valid until the next call on this thread */

forge_model* forge_open(const char* fpk_path, int64_t context, int verify_hashes);
void forge_close(forge_model* m);
void forge_reset(forge_model* m);             /* clear KV state */
int64_t forge_vocab_size(const forge_model* m);

/* Tokenizer. Return token count, or -1 on error; writes at most cap tokens. */
int64_t forge_tokenize(const forge_model* m, const char* text, int add_bos, int32_t* out, int64_t cap);
/* Writes NUL-terminated text, returns bytes needed (excluding NUL) or -1. */
int64_t forge_detokenize(const forge_model* m, const int32_t* ids, int64_t n, char* out, int64_t cap);

/* Runs tokens continuing from the current position; writes last-position logits (vocab floats). */
int forge_forward(forge_model* m, const int32_t* ids, int64_t n, float* logits, int64_t cap);

/* Prefill + decode. Returns number of generated tokens written to out (or -1). */
int64_t forge_generate(forge_model* m, const int32_t* prompt, int64_t n_prompt, int32_t max_new,
                       const forge_sampling* s, int32_t* out, int64_t cap, forge_token_cb cb, void* user);

#ifdef __cplusplus
}
#endif
#endif
