/* forge_aot.h — ABI of a natively compiled model (`forge aot --aot-mode native`, docs/aot-models.md).
 *
 * A compiled model library exports one function, forge_aot_model_v1_get(), returning a description of the model and,
 * for every precompiled plan, the entry points of its straight-line code. The engine (libforge, the forge CLI, or
 * the program the code is linked into) keeps owning the plan's buffers and the session state: it binds the code to
 * its buffers once per plan and then calls it for each run of consecutive kernel steps. State steps (KV appends,
 * recurrent state) stay in the engine, so sessions, samplers, grammars, rewind and prefix sharing work unchanged.
 *
 * Loading such a library executes its code: treat it like any other plugin. */
#ifndef FORGE_AOT_H
#define FORGE_AOT_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FORGE_AOT_ABI_VERSION 1

/* One graph value as the engine holds it. Indexed by value id (the node index of the plan's graph). */
typedef struct forge_aot_value {
    float* data;                 /* f32 buffer: arena slice, f32 weight, constant or state slot; NULL for an encoded weight */
    const unsigned char* qdata;  /* encoded weight bytes, else NULL */
    int64_t rows;                /* size of dimension 0 now (state views grow from run to run) */
} forge_aot_value;

typedef struct forge_aot_plan_v1 {
    uint32_t n_values;               /* graph nodes */
    uint32_t n_steps;                /* plan steps */
    uint64_t arena_floats;           /* arena size the code was generated for */
    uint32_t n_segments;             /* compiled step ranges */
    const uint32_t* segment_first;   /* first step of each segment */
    const uint32_t* segment_steps;   /* number of steps it covers */
    /* Bind the code to a plan's buffers: values[n_values] and the arena base. Returns an instance, or NULL with a
       message in err. The buffers of arena values, weights and constants must not move while the instance lives. */
    void* (*bind)(const forge_aot_value* values, float* arena, char* err, size_t err_cap);
    /* Run one segment. values is read again for the entries that are state views. Returns 0, or non-zero with a
       message in err. */
    int (*run)(void* instance, uint32_t segment, const forge_aot_value* values, char* err, size_t err_cap);
    void (*unbind)(void* instance);
} forge_aot_plan_v1;

typedef struct forge_aot_model_v1 {
    uint32_t abi_version;            /* FORGE_AOT_ABI_VERSION */
    uint32_t struct_size;            /* sizeof(forge_aot_model_v1): fields may be appended */
    const char* forge_version;
    const char* simd_target;         /* lowest SIMD target the code needs ("scalar", "avx2", "avx512", "neon", ...) */
    const char* compiler;            /* compiler and flags it was built with */
    const char* fpk_path;            /* the model file it was compiled from (absolute); weights are read from it */
    const unsigned char* fpk_data;   /* the .fpk itself when built with --embed-weights, else NULL */
    uint64_t fpk_size;
    const unsigned char* plans;      /* plan blob (the bytes of the .fpk "aot" section, docs/format.md) */
    uint64_t plans_size;
    uint32_t n_plans;                /* = entries of the plan blob */
    const forge_aot_plan_v1* plan;   /* plan[i]: code for blob entry i */
    /* 0 when this process can run the code (CPU features); else non-zero with a message in err. */
    int (*check)(char* err, size_t err_cap);
    /* generation defaults recorded at build time (used by the standalone executable) */
    float temperature; int32_t top_k; float top_p; int32_t max_new_tokens;
} forge_aot_model_v1;

#if defined(_WIN32) || defined(__CYGWIN__)
#  define FORGE_AOT_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define FORGE_AOT_EXPORT __attribute__((visibility("default")))
#else
#  define FORGE_AOT_EXPORT
#endif
#define FORGE_AOT_SYMBOL "forge_aot_model_v1_get"
#if defined(FORGE_AOT_BUILD)   /* the generated code defines it */
FORGE_AOT_EXPORT const forge_aot_model_v1* forge_aot_model_v1_get(void);
#endif

#ifdef __cplusplus
}
#endif
#endif
