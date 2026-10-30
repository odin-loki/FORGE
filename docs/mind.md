# mind: models, sessions and generation

Umbrella header: `<forge/mind/mind.hpp>`. Link `forge_mind`, `forge_verify` and `forge_lane_mv`.

## Concepts
- A **`model`** is an opened `.fpk`: memory-mapped, read-only, shareable between sessions and threads. It owns the device choice (`FORGE_DEVICE` / `set_device`), the plan cache and any LoRA adapters.
- A **`session`** owns the KV cache (paged) and position for one sequence. The blueprint's session kind decides how it is used: `generative_sequence` (language models: `generate`) or one-shot graphs (`infer`). Separate headers provide `diffusion_loop`, `streaming` and `custom` (plugin) session kinds.
- A **`sampler`** turns logits into tokens with a fixed chain: penalties → temperature → top-k → top-p → min-p → typical-p → draw, or Mirostat v2 in place of the truncation steps. Temperature 0 is greedy.

## API
### `model` (`session.hpp`)
- `model::open(path, verify = false)` → `std::shared_ptr<model>`; `verify` checks BLAKE3 hashes on open.
- `set_device("vulkan")` etc. returns the device actually used (falls back to `cpu` with a warning). Device strings: see [configuration.md](configuration.md).
- `blueprint()`, `manifest()`, `tokenizer()` (`nullptr` if none), `vocab()`, `max_context()`.
- LoRA: `add_adapter(name, {weight → {a, b}}, alpha)` or `add_adapter_file(name, path)` (an `.fpk` with `*.lora_a` / `*.lora_b` tensors); sessions pick one with `session::set_adapter(name)`. Base weights stay shared.

### `session`
- `session(path_or_model, session_options)`. `session_options`:

  | Field | Default | Meaning |
  |-------|---------|---------|
  | `context` | 0 | KV capacity; 0 = manifest `max_position_embeddings` (2048 if absent) |
  | `verify` | false | Check hashes when opening by path |
  | `page_rows` | 256 | KV page size in rows; 0 = reserve the whole context up front |
  | `kv_codec` | `""` | `f32`, `int8`, `fp8`; empty = `$FORGE_KV_CODEC`, else f32 |
  | `kv_offload` | -1 | 1 = keep KV on the host; -1 = `$FORGE_KV_OFFLOAD` |

- `generate(prompt_ids, max_new, sampler&, on_token = {}, generate_stats* = nullptr)` → new token ids. `on_token(int)` returning `false` stops. Stops at EOS.
- `forward(ids)` (last-row logits), `forward_all(ids)` (all rows), `position()`, `rewind(pos)`, `reset()`, `copy_prefix(other, n)`.
- `infer({name → data}, seq = 1)` for one-shot graphs (e.g. ONNX imports) → `{output → data}`.
- `tokenizer()` → `const pack::bpe_tokenizer*` with `encode`, `decode`, `bos`, `eos`.

### `sampler_config` (`sampler.hpp`)

| Field | Default (= off) |
|-------|-----------------|
| `temperature` | 0 (greedy) |
| `top_k`, `top_p`, `min_p`, `typical_p` | 0, 1, 0, 1 |
| `mirostat_tau`, `mirostat_eta` | 0, 0.1 |
| `repetition_penalty`, `presence_penalty`, `frequency_penalty` | 1, 0, 0 |
| `repetition_window` | 64 |
| `seed` | 0 |

### Generation helpers
- `speculative_generate(target, draft, prompt, max_new, …)` (`speculative.hpp`): draft-model speculation with rejection sampling, so the output distribution matches the target (identical to target output when greedy). `lookup_generate(target, prompt, max_new, sampler, k, …)`: prompt-lookup (n-gram) speculation with no draft model.
- `grammar.hpp`: GBNF grammars (llama.cpp dialect, byte level) to constrain sampling. `json_schema_to_gbnf(schema)` (`json_schema.hpp`) converts a JSON Schema subset (types, properties/required, items, enum, const, anyOf/oneOf, `$ref`, …) to GBNF.
- `batch_scheduler(model, batch_options)` (`batch.hpp`): continuous batching. `batch_options{max_active = 8, prefill_chunk = 256, prefix_cache = 4, session}`. `submit(request)` → id; `cancel(id)`; `step()`, `run_until_idle()`, or `start()`/`stop()` a background loop; `request_kv_eviction(keep)` trims the prefix cache. `request` holds `prompt`, `max_new`, `sampling`, optional `grammar`, `stop_tokens`, `priority` (higher first), `on_token`, `on_done(tokens, finish_reason)`. Finished sequences seed a prefix cache so prompts sharing a prefix reuse KV rows.
- `diffusion.hpp`: `diffusion_loop` (DDIM / DDPM / flow-Euler schedulers, epsilon / v / x0 prediction, classifier-free guidance, latent cache, Philox noise). The denoiser is any callable.
- `streaming.hpp`: SPSC ring buffer, window/hop framing and a latency budget that drops the oldest frames when the consumer falls behind.
- `plugin.hpp`: `custom` session kind for registering a non-neural engine behind the same entry points.

## Example
`docs/examples/mind_generate.cpp` (ctest `doc_example_mind_generate`, run on `golden/tiny_f32.fpk`):

```cpp
using namespace forge::mind;
session_options opt; opt.context = 256;
session s(argv[1], opt);
std::vector<int> prompt = s.tokenizer()->encode("Hello");
sampler_config sc; sc.temperature = 0.f;   // greedy
sampler smp(sc);
std::vector<int> out = s.generate(prompt, 16, smp);
std::printf("%s\n", s.tokenizer()->decode(out).c_str());
```

`examples/integrations/` shows the other session kinds (`camera_detector.cpp` for diffusion, `neuralvfx_stream.cpp` for streaming, `cypha_classifier.cpp` for plugins, `seatbelt_one_shot.cpp` for one-shot inference, `fuse_runtime.cpp` for the C ABI).

## Notes and limits
- The golden test model has random weights, so its text is meaningless; tests check numerics against a NumPy reference (`verify/golden/*.py`), not text quality.
- `model` serialises graph execution with a mutex; use `batch_scheduler` to serve many requests from one model.
