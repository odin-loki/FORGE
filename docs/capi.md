# C ABI (`forge.h`)

## Concepts
`inference/include/forge.h` is a stable C interface to the inference engine. The Python, Kotlin and Swift bindings ([bindings.md](bindings.md)) are built on it. The full declarations are in [reference.md](reference.md).

- Handles are opaque:
  - `forge_weights*` (ABI 4): one opened model file: mapped weights, tokenizer, chat template, compiled plans, LoRA adapters and the device. Reference-counted inside the library, so it may be closed while sessions made from it are still in use.
  - `forge_model*`: one sequence (KV state, position, selected adapter). `forge_open` makes one that owns private weights (ABI 1); `forge_session_open` makes one on shared weights, so several sequences use one set of weights. Every function that takes a `forge_model*` works on both kinds.
  - `forge_batch*` (ABI 4): a continuous-batching scheduler over shared weights.
- The ABI is versioned (`FORGE_ABI_VERSION 4`). Each version only adds functions; existing symbols and struct layouts do not change. Check `forge_abi_version()` at run time.
- Errors: functions return `NULL` or `-1` (`0` for `forge_batch_submit`), and `forge_last_error()` gives the message. It is thread-local and valid until the next call on that thread. Since ABI 4 the message is also stored in the handle the failing call was made on: `forge_model_error`, `forge_weights_error`, `forge_batch_error` return it until the next failing call on that handle. A `NULL` handle is an error, not a crash, in every function.
- Structs added in ABI 3 and 4 start with `uint32_t size`: set it to `sizeof(struct)`. Fields past `size` read as zero, and zero always means the documented default, so `{0}` plus `size` is a valid "all defaults" value and a caller built against an older header keeps working. A `NULL` struct pointer also means all defaults.
- Buffers: functions that write text or arrays return the size needed (text: bytes without the NUL), write at most `cap`, and may be called with `cap` 0 to size the buffer.
- `FORGE_API` exports the functions: `__declspec(dllexport/dllimport)` on Windows (define `FORGE_C_STATIC` to link a static copy), default visibility elsewhere.

CMake target `forge_c` builds the shared library from `inference/capi/forge_c.cpp`: `libforge.so` (soname `libforge.so.1`), `libforge.dylib`, or `forge.dll` on Windows. Settings not covered by a parameter come from the environment (`FORGE_KV_CACHE_TYPE`, `FORGE_MEM_LIMIT`, ...; see [configuration.md](configuration.md)).

### Thread safety
Each function has one of three rules (the header repeats them per function):

| Rule | Meaning |
|---|---|
| any | Callable from any thread at any time |
| handle | Not concurrently with another call on the **same** handle. Calls on different handles may run in parallel; graph runs on shared weights are serialised inside the engine, so sessions on one `forge_weights` do not compute in parallel |
| setup | Changes shared weights: must not run while any session or batch on those weights is inside a call |

## API reference

### ABI 1 to 3 (unchanged)

| Function | ABI | Threads | Purpose |
|---|---|---|---|
| `forge_abi_version()`, `forge_version()`, `forge_last_error()` | 1 | any | Version (`forge_version` returns `"0.1.0"`) and last error |
| `forge_open(path, context, verify_hashes)` | 1 | any | Open a `.fpk` with private weights. `context` 0 = the model's maximum. Returns `NULL` on error |
| `forge_close(m)` | 1 | handle | Free the session (and its private weights) |
| `forge_reset(m)` | 1 | handle | Clear the KV state (position 0) |
| `forge_vocab_size(m)` | 1 | handle | Vocabulary size |
| `forge_tokenize(m, text, add_bos, out, cap)` | 1 | handle | Returns the full token count (writes at most `cap`), or -1. `add_bos`: 1 prepend BOS, 0 not, -1 the model's default (the manifest's `add_bos`, or the tokenizer's post-processor: `[CLS]` ... `[SEP]` for BERT) |
| `forge_detokenize(m, ids, n, out, cap)` | 1 | handle | Writes NUL-terminated text; returns the bytes needed without the NUL, or -1 |
| `forge_chat_template_apply(m, messages_json, add_generation_prompt, out, cap)` | 3 | handle | Renders the model's chat template over a JSON message array; -1 when the file has none |
| `forge_forward(m, ids, n, logits, cap)` | 1 | handle | Runs `ids` from the current position; writes up to `cap` floats of the last row's logits. 0 or -1 |
| `forge_generate(m, prompt, n, max_new, &sampling, out, cap, cb, user)` | 1 | handle | Prefill and decode. Returns the number of tokens generated (may exceed `cap`), or -1 |
| `forge_position(m)` | 2 | handle | Tokens in the KV state |
| `forge_rewind(m, pos)` | 2 | handle | Roll the KV state back to `pos` tokens. 0 or -1 |
| `forge_forward_all(m, ids, n, logits, cap)` | 2 | handle | Writes `n × vocab` logits (up to `cap`) |
| `forge_generate_ex(..., const forge_generate_options*, ...)` | 2 | handle | Adds a GBNF `grammar`, a speculative `draft` model and `spec_k` (≤ 0 → 4) |
| `forge_generate_ex2(..., const forge_generate_options*, const forge_sampling_ext*, ...)` | 3 | handle | Adds `typical_p`, `mirostat_tau`, `mirostat_eta` |

| Struct | Fields |
|---|---|
| `forge_sampling` | `temperature` (0 = greedy), `top_k` (0 = off), `top_p` (1 = off; ≤ 0 treated as 1), `min_p` (0 = off), `repetition_penalty` (1 = off; ≤ 0 treated as 1), `seed` |
| `forge_generate_options` | `grammar` (GBNF text or `NULL`), `draft` (`forge_model*` with the same vocab, or `NULL`), `spec_k` |
| `forge_sampling_ext` | `size`, `typical_p` (0 or 1 = off), `mirostat_tau` (0 = off), `mirostat_eta` (0 = 0.1) |

Semantics of `forge_generate*` (ABI 1 to 3):
- It **continues from the current KV position**: the prompt is appended to what the handle already holds. Call `forge_reset` first for a fresh sequence. With a `draft` model, both handles are reset first.
- Generation stops at `max_new` tokens, at EOS (EOS **is included** in the output), when the context is full, when the callback returns non-zero, or when the grammar is complete or rejects a token.
- The callback is `int cb(int32_t token, const char* text, void* user)`. `text` is that token's bytes, which may be part of a UTF-8 sequence.
- After generation, `forge_position` is the prompt length plus the generated count minus 1: the last sampled token has not been fed through the model yet.

### ABI 4: weights, devices, sessions

| Function | Threads | Purpose |
|---|---|---|
| `forge_weights_open(path, const forge_open_params*)` | any | Open a model file. `forge_open_params`: `verify_hashes`, `device` (`"cpu"`, `"auto"`, `"vulkan"`, `"cuda"`, ...; `NULL` = `$FORGE_DEVICE`, else cpu), `deterministic` (0 = follow `$FORGE_DETERMINISTIC`, 1 = on, 2 = off), `no_aot` (1 = ignore precompiled plans) |
| `forge_weights_close(w)` | handle | Drop this handle; sessions and batches made from it stay valid |
| `forge_model_weights(m)` | handle | A new `forge_weights*` on the weights a session uses (close it too) |
| `forge_weights_device(w)`, `forge_weights_set_device(w, device)` | handle / setup | The device in use; select one (`"auto"` scores the devices that open; an unavailable device falls back to `"cpu"`, which is what is returned). Selecting clears the compiled plans |
| `forge_set_deterministic(mode)` | any | Process-wide determinism mode ([gpu.md](gpu.md)): 1 on, 0 off, -1 follow the environment again. Returns the mode in effect |
| `forge_session_open(w, const forge_session_params*)` | any | A new sequence on shared weights. `forge_session_params`: `context` (0 = the model's maximum), `kv_codec` (`NULL` = `$FORGE_KV_CACHE_TYPE`, else f32), `kv_offload`. Close with `forge_close` |
| `forge_model_error(m)`, `forge_weights_error(w)`, `forge_batch_error(b)` | handle | The last error of that handle (`""` when none) |
| `forge_context_size(m)`, `forge_session_kind(m)`, `forge_manifest_json(m, out, cap)` | handle | Context in tokens; the blueprint's session kind (`"generative_sequence"`, `"embedding"`, ...); the manifest as JSON text |

### ABI 4: tokenizer details and chat

| Function | Threads | Purpose |
|---|---|---|
| `forge_token_bos(m)`, `forge_token_eos(m)` | handle | Ids, or -1 |
| `forge_token_text(m, id, out, cap)` | handle | The bytes a token adds to a text (what the token callback gets) |
| `forge_token_id(m, token)` | handle | Id of a vocabulary entry given as stored (`"<\|im_end\|>"`), or -1 |
| `forge_token_is_special(m, id)` | handle | 1 / 0 |
| `forge_stop_tokens(m, chat, out, cap)` | handle | Ids that end an assistant turn: EOS, the manifest's `eos_token_ids`, and with `chat != 0` the end-of-turn markers of the chat template ([inference.md](inference.md#chat-tools-chat_toolshpp)). Pass them as `stop_tokens` |
| `forge_chat_template(m, out, cap)` | handle | The template source; 0 and `""` when the file has none |
| `forge_chat_template_apply_ex(m, messages_json, tools_json, extra_json, add_generation_prompt, out, cap)` | handle | As `forge_chat_template_apply`, with an OpenAI `tools` array and an object of extra template variables (either may be `NULL`). Tokenize the result with `add_bos` 0 |

### ABI 4: sampling and generation

`forge_sampler_params` is the whole sampler chain: logit bias → repetition / presence / frequency penalties → grammar → temperature → top-k → top-p → min-p → typical → draw (Mirostat v2 replaces the truncation steps).

| Field | Zero means | Notes |
|---|---|---|
| `temperature` | greedy | |
| `top_k`, `top_p`, `min_p`, `typical_p` | off | `top_p` / `typical_p` 1 is also off |
| `mirostat_tau`, `mirostat_eta` | off, 0.1 | |
| `repetition_penalty` | off (1) | |
| `presence_penalty`, `frequency_penalty` | off | Subtracted once / per occurrence for tokens in the window |
| `repetition_window` | 64 | Tokens of history the three penalties look at |
| `seed` | 0 | The same seed, prompt and settings give the same tokens |
| `logit_bias`, `n_logit_bias` | none | `forge_logit_bias {token, bias}` added to the raw logits; ids outside the vocabulary are ignored |

| Function | Threads | Purpose |
|---|---|---|
| `forge_generate_v4(m, prompt, n, max_new, const forge_generate_params*, out, cap, &stop_reason)` | handle (m and the draft) | Prefill and decode from the current position. Returns the tokens generated, or -1. `stop_reason` (may be `NULL`): `FORGE_STOP_LENGTH`, `FORGE_STOP_EOS` (EOS or a stop token), `FORGE_STOP_CALLBACK`, `FORGE_STOP_GRAMMAR` |
| `forge_json_schema_to_grammar(schema_json, out, cap)` | any | JSON Schema → GBNF text (the subset of [inference.md](inference.md)) |
| `forge_grammar_check(grammar)` | any | 0 when the text parses as GBNF, else -1 with the parse error |

`forge_generate_params`: `sampler` (`NULL` = greedy), `grammar` (GBNF) or `json_schema` (not both), `draft` + `spec_k` (speculative decoding; both sessions are reset), `stop_tokens` + `n_stop_tokens`, `ignore_eos`, `reset` (1 = clear the KV state first), `on_token` + `user`.

Differences from `forge_generate` (ABI 1 to 3): the token that ends generation (EOS, a stop token) is **not** written to `out` and not passed to `on_token`; with a grammar, stop tokens are allowed exactly where the grammar can end. `ignore_eos` is not available with `draft`.

### ABI 4: LoRA adapters

| Function | Threads | Purpose |
|---|---|---|
| `forge_adapter_load(w, name, path)` | setup | Load an adapter from an `.fpk` or `.safetensors` with `<weight>.lora_a` / `<weight>.lora_b` tensors ([inference.md](inference.md)). A name cannot be loaded twice. 0 or -1 |
| `forge_adapter_exists(w, name)` | handle | 1 / 0 |
| `forge_adapter_set(m, name)` | handle | Select it for this session (`NULL` or `""` = base weights). Clears the session's KV state when the adapter changes. -1 for an unknown name |
| `forge_adapter_get(m)` | handle | The session's adapter, `""` = base |

There is no unload: an adapter stays with its weights until they are freed (plans of live sessions point at its merged matrices). "Unset" is `forge_adapter_set(m, NULL)`.

### ABI 4: KV state and prefix reuse

| Function | Threads | Purpose |
|---|---|---|
| `forge_state_save(m, n_tokens, out, cap)` | handle | Serialise the KV state of the first `n_tokens` tokens (-1 = all). Returns the bytes needed, or -1. The blob belongs to that model file and KV codec |
| `forge_state_load(m, data, n)` | handle | Replace the session's state with a saved one; it then continues with identical logits. A blob that is not accepted (truncated, other codec, longer than the context) returns -1 and leaves the session empty. The session must use the adapter the state was computed with (not checked) |
| `forge_state_copy(dst, src, n_tokens)` | handle (both) | Copy the first `n_tokens` tokens of `src` into `dst`: prefix sharing between sessions on the same `forge_weights`, same KV codec and adapter |
| `forge_state_bytes(m)` | handle | KV bytes the session holds |
| `forge_rewind`, `forge_reset` | handle | Cut the state back (ABI 1 / 2) |

Models with recurrent state (Qwen3.5) can only save, copy or rewind at their full length ([inference.md](inference.md)). In a `forge_batch` the prefix cache is automatic; `forge_batch_params.prefix_cache` sizes it and `forge_batch_evict_prefix_cache` empties it.

### ABI 4: embeddings
For models whose `forge_session_kind` is `"embedding"` (the `bert` template family).

| Function | Threads | Purpose |
|---|---|---|
| `forge_embed_dim(m)` | handle | Values per embedding, or -1 for a model that is not an embedding model |
| `forge_embed_tokens(m, ids, n, type_ids, out, cap)` | handle | One vector from token ids as given (`type_ids` may be `NULL`). Returns the float count |
| `forge_embed_text(m, text, pair, out, cap, &n_tokens)` | handle | One vector from text through the tokenizer (special tokens and token types from its post-processor); `pair` is an optional second sequence |
| `forge_embed_texts(m, texts, n, out, cap)` | handle | `n` texts: writes `n × dim` floats, row `i` for `texts[i]`; returns `dim`. A loop with shared setup, not one batched pass ([inference.md](inference.md) `embed_batch`); every text is checked before any runs |

Vision and audio encoders take no token input and have no C ABI call.

### ABI 4: batching

| Function | Threads | Purpose |
|---|---|---|
| `forge_batch_open(w, const forge_batch_params*)` | any | A scheduler ([inference.md](inference.md#continuous-batching-batchhpp)). `forge_batch_params`: `max_active` (0 = 8), `context`, `prefill_chunk` (0 = 256), `prefix_cache` (0 = 4 entries, -1 = none), `kv_codec` |
| `forge_batch_submit(b, const forge_batch_request*)` | any | Queue a request; returns its id (> 0) or 0. The prompt, grammar and stop tokens are copied. May be called from other threads and from `on_token`, not from `on_done` |
| `forge_batch_cancel(b, id)` | any | 1 = cancelled (its `on_done` gets `"cancelled"`: at once for a waiting request, at the next step for a running one), 0 = unknown or finished |
| `forge_batch_step(b)` | handle | One step: admit waiting requests, advance every active sequence by one token. 1 = work done, 0 = idle, -1 = error |
| `forge_batch_run(b)` | handle | Step until idle |
| `forge_batch_stats_get(b, &stats)` | handle | `steps`, `completed`, `failed`, `prefix_hits`, `prefix_tokens_reused`, `peak_active`, `peak_kv_bytes`, `waiting`, `active` (set `stats.size` first) |
| `forge_batch_evict_prefix_cache(b, keep)` | any | Drop cached prefixes down to `keep` entries at the next step |
| `forge_batch_close(b)` | handle | Requests still waiting or running get `on_done(..., "cancelled")` first |

`forge_batch_request`: `prompt` + `n_prompt`, `max_new` (0 = 64), `sampler`, `grammar` or `json_schema`, `stop_tokens`, `adapter` (a loaded LoRA adapter; requests with different adapters share a batch, the prefix cache is keyed by adapter), `priority`, `on_token`, `on_done`, `user`. Callbacks run on the thread that calls `forge_batch_step` / `forge_batch_run`. `on_done(id, tokens, n, reason, user)`: `reason` is `"stop"`, `"length"`, `"grammar"`, `"cancelled"` or `"error: <message>"`; here the token that ended the sequence (EOS, a stop token) is the last of `tokens`. A request that fails does not affect the others. There is no background worker in the C ABI: the caller drives the steps.

### ABI 4: ahead-of-time plans

| Function | Threads | Purpose |
|---|---|---|
| `forge_aot_precompile(fpk_path, out_path, seq_lengths, n)` | any | `forge aot` in plan mode with a sidecar ([aot-models.md](aot-models.md)): compiles the CPU plans for the given sequence lengths (`NULL` / 0 = the defaults) and writes them to `out_path`, or to `<fpk_path>.fplan`, which a later open picks up. Returns the number of plans. Signed models are refused (a sidecar is not covered by the signature). Native-code AOT, embedding the plans in the `.fpk`, and other KV codecs are `forge aot` only |

## Example
`docs/examples/capi_generate.c` (ctest `doc_example_capi_generate`, run on `golden/tiny_f32.fpk`). `tests/capi_smoke.c` (ctest `capi_smoke`) exercises every function of ABI 1 to 4, with the tiny BERT golden for the embedding calls.

```c
if (forge_abi_version() < 4) return 1;
forge_open_params op = {0}; op.size = sizeof op; op.device = "auto";
forge_weights* w = forge_weights_open(path, &op);
if (!w) { fprintf(stderr, "%s\n", forge_last_error()); return 1; }
forge_model* a = forge_session_open(w, NULL);        /* two sequences, one set of weights */
forge_model* b = forge_session_open(w, NULL);

int32_t prompt[64], out[64], stops[16];
int64_t n = forge_tokenize(a, "Hello", -1, prompt, 64);
forge_sampler_params sp = {0}; sp.size = sizeof sp;
sp.temperature = 0.8f; sp.top_k = 40; sp.seed = 7; sp.presence_penalty = 0.2f;
forge_generate_params gp = {0}; gp.size = sizeof gp;
gp.sampler = &sp; gp.reset = 1; gp.on_token = on_token; gp.user = &user;
gp.stop_tokens = stops; gp.n_stop_tokens = forge_stop_tokens(a, 1, stops, 16);
int32_t why;
int64_t got = forge_generate_v4(a, prompt, n, 64, &gp, out, 64, &why);
if (got < 0) fprintf(stderr, "%s\n", forge_model_error(a));

gp.json_schema = "{\"enum\":[\"yes\",\"no\"]}";      /* constrained output */
got = forge_generate_v4(b, prompt, n, 8, &gp, out, 64, NULL);

forge_close(a); forge_close(b); forge_weights_close(w);
```

The ABI 1 form still works unchanged:

```c
forge_model* m = forge_open(path, 256, 0);
forge_sampling s = {0.f, 0, 1.f, 0.f, 1.f, 42};   /* temperature, top_k, top_p, min_p, rep_pen, seed */
int64_t got = forge_generate(m, prompt, n, 16, &s, out, 16, on_token, &user);
forge_close(m);
```

## Platform notes and limits
- Built and run with clang 23 on Linux x86_64 only (`capi_smoke`, `doc_example_capi_generate`, `python_bindings`). The header is plain C99; other compilers see ABI 4 for the first time in CI.
- Cross builds (`CMAKE_CROSSCOMPILING`) do not build `capi_smoke` or this example; the ctests also need `FORGE_BUILD_APPS` (they use the golden models).
- C consumers link with the C++ driver (`LINKER_LANGUAGE CXX`) because the library needs the C++ runtime.
- Not in the C ABI: log-probabilities, tool-call grammars and parsing (forge-serve has them, [serve.md](serve.md)), speculative lookup decoding, diffusion and streaming sessions, vision / audio embedding inputs, unloading an adapter, a background batch worker.
- Device selection was exercised with `"cpu"` and an unavailable device name (fallback) only; `"auto"` and GPU devices through the C ABI were not run here.
- With `-DFORGE_JNI=ON` (set in the `android-arm64` preset) the build also produces `libforge_jni`, the JNI shim used by the Kotlin binding.
- `tools/build_xcframework.sh` packages `forge_c` for iOS, the iOS simulator and macOS (not run here; see [bindings.md](bindings.md)).
