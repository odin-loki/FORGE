# C ABI (`forge.h`)

## Concepts
`mind/include/forge.h` is a stable C interface to the inference engine. The Python, Kotlin and Swift bindings all build on it, and so can any other language.
- All handles are opaque.
- The ABI is versioned (`FORGE_ABI_VERSION 3`), and each new version only adds to the previous one.
- Errors are reported through `forge_last_error()`. The message is thread-local and stays valid until the next call on the same thread.

The CMake target `forge_c` builds `libforge.so` (soname `libforge.so.1`) from `mind/capi/forge_c.cpp`.

## API reference

| Function | ABI | Purpose |
|----------|-----|---------|
| `forge_abi_version()`, `forge_version()`, `forge_last_error()` | 1 | Version and error queries |
| `forge_open(path, context, verify_hashes)` / `forge_close(m)` | 1 | Open or close a `.fpk`. A `context` of 0 means use the model's maximum. Returns NULL on error |
| `forge_reset(m)` | 1 | Clear the KV state |
| `forge_vocab_size(m)` | 1 | Vocabulary size |
| `forge_tokenize(m, text, add_bos, out, cap)` | 1 | Returns the token count, or -1 |
| `forge_detokenize(m, ids, n, out, cap)` | 1 | Writes NUL-terminated text and returns the bytes needed |
| `forge_forward(m, ids, n, logits, cap)` | 1 | Runs from the current position and writes the last row of logits |
| `forge_generate(m, prompt, n, max_new, &sampling, out, cap, cb, user)` | 1 | Prefill and decode. Returns the number of tokens generated, or -1 |
| `forge_position(m)`, `forge_rewind(m, pos)` | 2 | Read the KV position, or roll it back |
| `forge_forward_all(m, ids, n, logits, cap)` | 2 | Writes `n * vocab` logits |
| `forge_generate_ex(..., &forge_generate_options, ...)` | 2 | Adds a GBNF `grammar`, a speculative `draft` model and `spec_k` (default 4) |
| `forge_generate_ex2(..., &forge_sampling_ext, ...)` | 3 | Adds `typical_p`, `mirostat_tau` and `mirostat_eta`. Set `size = sizeof(forge_sampling_ext)` |

- `forge_sampling` has the fields `{temperature, top_k, top_p, min_p, repetition_penalty, seed}`. The values 0, 0, 1, 0 and 1 turn the corresponding sampler off.
- The callback signature is `int cb(int32_t token, const char* text, void* user)`. Return a non-zero value to stop generation.

## Example
`docs/examples/capi_generate.c` runs as the ctest `doc_example_capi_generate`. `tests/capi_smoke.c` exercises more of the API.

```c
forge_model* m = forge_open(path, 256, 0);
int64_t n = forge_tokenize(m, "Hello", 1, prompt, 64);
forge_sampling s = {0.f, 0, 1.f, 0.f, 1.f, 42};
int64_t got = forge_generate(m, prompt, n, 16, &s, out, 16, on_token, &user);
forge_close(m);
```

After generation, `forge_position` equals the prompt length plus the generated tokens minus 1. This is because the last sampled token has not been fed through the model yet. In the example, 6 + 16 - 1 gives 21.

## Platform notes and limits
- Cross builds (any `CMAKE_CROSSCOMPILING` configuration) do not build `capi_smoke` or this example.
- With `-DFORGE_JNI=ON` (the default in the `android-arm64` preset), the build also produces `libforge_jni`. This is a JNI shim over this ABI, used by the Kotlin binding: see [bindings.md](bindings.md).
