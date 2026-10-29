# mind: models, sessions and generation

## Concepts
- A `model` is an opened `.fpk` file. It is shared and read-only.
- A `session` owns the KV cache and the current position for one conversation.
- A `sampler` turns logits into token ids using a fixed chain: penalties, then temperature, then top-k, top-p, min-p, typical, and Mirostat v2.

## API (`mind/include/forge/mind/session.hpp`, `sampler.hpp`)
- `model::open(path, verify=false)` returns a `std::shared_ptr<model>`. With `verify` set, the BLAKE3 hashes are checked on open.
- `session(path_or_model, session_options)`. The `session_options` fields and their defaults:
  - `context = 0`: 0 means use the model's maximum.
  - `verify = false`.
  - `page_rows = 256`.
  - `kv_codec = ""`: accepts `f32`, `int8` or `fp8`. When empty, `$FORGE_KV_CODEC` is used.
  - `kv_offload = -1`: -1 means use `$FORGE_KV_OFFLOAD`.
- `session::generate(prompt, max_new, sampler&, on_token = {}, generate_stats* = nullptr)` returns the new token ids. Return `false` from `on_token` to stop early.
- `session::tokenizer()` returns a `const pack::bpe_tokenizer*`, or `nullptr` if the model has none. It provides `encode`, `decode` and `eos`.
- `sampler_config` fields and their defaults:
  - `temperature = 0`: 0 means greedy.
  - `top_k = 0`, `top_p = 1`, `min_p = 0`, `typical_p = 1`.
  - `mirostat_tau = 0`, `mirostat_eta = 0.1`.
  - `repetition_penalty = 1`, `presence_penalty = 0`, `frequency_penalty = 0`, `repetition_window = 64`.
  - `seed = 0`.

## Example
`docs/examples/mind_generate.cpp` runs as the ctest `doc_example_mind_generate` against the generated `golden/tiny_f32.fpk`.

```cpp
forge::mind::session_options opt; opt.context = 256;
forge::mind::session s(path, opt);
auto prompt = s.tokenizer()->encode("Hello");
forge::mind::sampler smp;                     // default config = greedy
auto out = s.generate(prompt, 16, smp);
```

## Platform notes and limits
- Link with `forge_mind`, `forge_verify` and `forge_lane_mv`.
- The golden test model has random weights, so its output text is meaningless. The test only checks that generation runs.
