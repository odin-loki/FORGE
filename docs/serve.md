# serve: the forge-serve HTTP server

## Concepts
`forge-serve` (`apps/forge_serve.cpp`, HTTP reader in `apps/serve_http.hpp`) serves one model over HTTP or HTTPS with an OpenAI-style API. Generation requests go through one continuous-batching scheduler (`inference::batch_scheduler`, see [inference.md](inference.md)), which reuses KV rows for prompts that share a prefix with a recent request (per LoRA adapter).

- The model argument may be an `.fpk`, a Hugging Face directory or a `.gguf`. The last two are imported into `<tmp>/<name>.serve.fpk` at startup, using `--codec` (default `f32`; a GGUF or an AWQ / GPTQ checkpoint keeps its stored quantisation unless `--codec` is given).
- The model is generative (`session generative_sequence`: completions and chat) or an embedding model (`session embedding`: `/v1/embeddings`). Other session kinds are refused.
- One thread per connection, at most `--max-connections`. Connections are kept alive (HTTP/1.1 default, HTTP/1.0 on request) and requests on one connection are answered in order, so pipelining is safe. Streams are sent with chunked transfer coding and leave the connection usable.
- Chat requests end at the model's end-of-turn tokens, can call tools, and can pick a LoRA adapter per request; see [Chat models](#chat-models-end-of-turn-and-tool-calls) and [LoRA adapters](#lora-adapters).
- Optional API keys, rate limits and TLS: [Authentication and TLS](#authentication-and-tls). Without them it is an open plain-HTTP server, bound to 127.0.0.1 by default.
- `forge-serve proxy ...` runs the same binary as a load-balancing proxy: see [loadbalancer.md](loadbalancer.md).

## API reference

### Command line

```
forge-serve <model.fpk|hf_dir|model.gguf> [--host 127.0.0.1] [--port 8080] [--max-active 8] [--ctx 0]
            [--codec f32] [--allow-unused] [--watch] [--no-pin] [--numa-node N] [--mem-limit 6G|75%|off] [--max-queue 0] [--kv-spill]
            [--policy FILE]
            [--api-key KEY]... [--api-key-file FILE] [--rate-limit RPM]
            [--tls-cert FILE --tls-key FILE [--tls-client-ca FILE]] [--cors ORIGIN|off]
            [--lora NAME=FILE]... [--max-n 8] [--max-connections 256]
            [--max-header-bytes 1048576] [--max-body-bytes 67108864]
            [--request-timeout 60] [--keep-alive-timeout 30] [--shutdown-timeout 30]
forge-serve proxy --backend HOST:PORT [...] [--policy FILE]   # see loadbalancer.md
```

| Flag | Default | Effect |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8080` | Listen address: an IPv4 or IPv6 address or a host name (`0.0.0.0` / `::` for all interfaces). `--port 0` picks a free port (printed on startup). A non-loopback `--host` without API keys prints a warning |
| `--max-active` | 8 | Sequences decoded together (`batch_options::max_active`) |
| `--ctx` | 0 | KV context per request; 0 = the model's maximum |
| `--codec` | `f32` | Weight codec for the import of a HF directory or `.gguf`; ignored for `.fpk` |
| `--allow-unused` | off | Import a HF directory or `.gguf` although it has tensors the template does not use (without it the import fails and lists them; [modelfile.md](modelfile.md#what-an-import-checks-and-records)) |
| `--mem-limit` | `FORGE_MEM_LIMIT`, else 85% of RAM or the cgroup limit | Host memory budget (`6G`, `512M`, `75%`; `0`/`off` disables). Under it the scheduler reports live KV bytes, evicts cached prefixes, and keeps new requests queued while the budget is over its soft watermark, so a tight limit runs requests one at a time instead of failing. See [memory.md](memory.md) |
| `--max-queue N` | 0 (no cap) | Once N requests are waiting, new ones get `503` with `Retry-After: 1` |
| `--kv-spill` | off | Cached prefixes evicted from memory are written to the spill directory (`FORGE_MEM_SPILL_DIR`, capped by `FORGE_MEM_SPILL_MAX`) and read back when a new prompt shares them, instead of being recomputed. Up to 64 prefixes. The files are encrypted under a key held only in memory unless `FORGE_MEM_SPILL_ENCRYPT=0` ([memory.md](memory.md)) |
| `--watch` | off | Watches the model path. On change: waits 0.5 s, sends systemd `RELOADING=1`, closes the listener and re-execs itself with the same arguments. POSIX only; on Windows it logs that a restart is needed |
| `--no-pin` | pinning on | On hybrid CPUs forge-serve pins itself (and the threads it starts) to performance cores. This turns that off |
| `--numa-node N` | none | Binds to NUMA node N before loading, so weights and KV are first-touched there |
| `--policy FILE` | `load_policy()` search | FORGE Security Standard policy; see Security policy below |
| `--api-key KEY` | none | An accepted API key (repeatable, at least 8 characters). Visible in the process list: prefer the file |
| `--api-key-file FILE` | none | Keys, one per line: `[rpm=N] [quota=N] <key> [name]`; `#` starts a comment. Under a strict security policy the file must be private (owner-only) |
| `--rate-limit RPM` | 0 (off) | Generation and embedding requests per minute per key (a key's own `rpm=` wins). Without keys: one bucket for all clients |
| `--tls-cert`, `--tls-key` | none | PEM certificate chain and key: serve HTTPS (TLS 1.2 or newer) instead of HTTP. Needs a build with OpenSSL |
| `--tls-client-ca FILE` | none | Mutual TLS: every client must present a certificate that chains to this CA |
| `--cors ORIGIN` | `*` | Value of `Access-Control-Allow-Origin` on every response (`Vary: Origin` is added unless it is `*`). `off` sends no CORS headers |
| `--lora NAME=FILE` | none | Load a LoRA adapter (repeatable): an `.fpk` or `.safetensors` with `<weight>.lora_a` / `<weight>.lora_b` tensors; see [LoRA adapters](#lora-adapters) |
| `--max-n N` | 8 | Largest `n` (choices) one request may ask for (1 to 64) |
| `--max-connections N` | 256 | Open connections (= threads). One more is answered `503` and closed |
| `--max-header-bytes`, `--max-body-bytes` | 1 MiB, 64 MiB | Limits of one request: `431` / `413` beyond them |
| `--request-timeout S` | 60 | A request must arrive completely within S seconds of its first byte, else `408` |
| `--keep-alive-timeout S` | 30 | An idle kept-alive connection is closed after S seconds |
| `--shutdown-timeout S` | 30 | On SIGINT / SIGTERM: time for requests in flight to finish; see [Shutdown](#shutdown) |

Fixed behaviour (no flag):
- The prefix cache keeps 4 finished sequences; prompts are prefilled in chunks of 256 tokens.
- OS memory-pressure events evict the prefix cache (moderate: keep 1, critical: keep 0) and log a warning to the system log.
- After the socket listens, forge-serve calls `service::ready()` (systemd `READY=1`; a no-op without `NOTIFY_SOCKET`).
- `FORGE_HUGEPAGES=1` advises huge pages on weight mappings ≥ 2 MiB (see [configuration.md](configuration.md)).
- `FORGE_DEVICE` picks the device, as for every model.

### Security policy
With a policy (`--policy FILE`, `$FORGE_SECURITY_POLICY`, `./forge-security.json`, the config dir or `$FORGE_SECURITY_PROFILE`; see [security-api.md](security-api.md)); with none of them nothing below happens:

- The policy is active before the model is loaded: the model path must be readable under `fs.read`, and `integrity.models` decides (`required`: a valid signature, by one of `integrity.trusted_keys` if set, else exit 1 with an integrity error; `optional`: signed files are verified, unsigned ones audited).
- `--host`/`--port` must pass `enforce(net_listen)` (loopback, or any address with `net.mode` `any`). `--watch` needs `exec` (it re-executes forge-serve); a policy that denies exec stops forge-serve at startup.
- After the socket is bound and the model loaded, forge-serve calls `security::lockdown(policy)` (SBX-1) before it accepts the first request, prints controls that are not `ok` to stderr, and audits the lockdown. Linux: seccomp is synchronised to every thread (TSYNC); Landlock write rules bind the main thread and the request threads started after it, not the scheduler thread started before (it is covered by `enforce()` only). `--kv-spill` needs its spill directory under `fs.write`.
- `forge-serve proxy --policy` checks the listen address and every `--backend` / `--route` target (`net_connect`; `net.mode` `none` refuses them all), then locks down after the listener starts.

Test: `tests/security_cli_test.py` (strict policy with a signed model serves on loopback; an unsigned model and a non-loopback `--host` are refused).

### Endpoints

| Method and path | Response |
|---|---|
| `GET /health` (also `/healthz`, `/v1/health`) | `{"status":"ok","model","kind":"generative"\|"embedding","queue_waiting","queue_active"}`; `503` with `"status":"draining"` during shutdown. Never needs an API key |
| `GET /v1/models` | `{"object":"list","data":[{"id","object":"model","created","owned_by":"forge","kind","context_length","vocab_size","architecture","device","chat_template","tool_format"}]}`: the model file's name, then one entry `"<model>:<adapter>"` per LoRA adapter (with `parent` and `adapter`) |
| `GET /v1/models/<id>` | One entry of that list; `404` with code `model_not_found` otherwise |
| `POST /v1/completions` | `{"id","object":"text_completion","created","model","choices":[{"index","text","tokens","logprobs"?,"finish_reason"}],"usage"}` |
| `POST /v1/chat/completions` | `{"id","object":"chat.completion",...,"choices":[{"index","message":{"role":"assistant","content","tool_calls"?},"logprobs"?,"finish_reason"}],"usage"}` |
| `POST /v1/embeddings` | `{"object":"list","data":[{"object":"embedding","index","embedding"}],"model","usage":{"prompt_tokens","total_tokens"}}`; see [Embeddings](#embeddings) |
| `GET /metrics` | Prometheus text, see below |
| `OPTIONS *` | `204` with the CORS headers (`Access-Control-Allow-Headers: Authorization, Content-Type`). Needs no API key (browser preflights carry none) |

A known path with the wrong method gets `405` with an `Allow` header; other paths get `404`.

### Embeddings

`POST /v1/embeddings` serves a model whose blueprint declares `session embedding` and takes token input (the `bert` template: BERT, RoBERTa, XLM-R and the MiniLM / BGE / E5 / GTE checkpoints built on them). On such a model the completion endpoints answer `400`; on a generative model `/v1/embeddings` answers `400` naming the model's session kind.

| Field | Default | Notes |
|---|---|---|
| `input` | required | A string, a list of strings (at most 2048), a list of token ids, or a list of token id lists. Text goes through the model's tokenizer, whose post-processor adds the special tokens (`[CLS]` ... `[SEP]`) and token types; ids are fed as given. An empty string, an id outside the vocabulary or an input longer than the model's positions is a `400` (nothing is truncated) |
| `model` | | Accepted and ignored; the response names the served file |
| `encoding_format` | `float` | `float`: a JSON array per vector. `base64`: base64 of the vector's little-endian f32 bytes |
| `dimensions` | | Only the model's own size is accepted; anything else is a `400` |

One vector per input, in input order (`index`). The vector is what `forge embed` prints: pooled, and L2-normalised when the checkpoint says so. `usage.prompt_tokens` counts the tokens fed to the model, special tokens included. **Batching:** all inputs of a request are tokenized and checked first (one that does not fit fails the whole request with a `400` naming it, before anything is computed), then embedded in one `session::embed_batch` call. That is a loop with shared setup, not a batch axis in the graph: inputs run shortest first, and the plan and executor for each sequence length are built once and reused. Requests run one after another on one embedding session; the scheduler, `--max-active`, `--max-queue` and the prefix cache do not apply to them. Vision and audio encoders (`vit`, `whisper_encoder`) take no token input and are not served here. Test: `tests/serve_embeddings_test.py` (ctest `serve_embeddings`) against the tiny BERT of `verify/golden/tiny_bert.py`.

```sh
curl -s localhost:8080/v1/embeddings -d '{"input":["hello world","second text"]}'
```

### Request fields

| Field | Default | Notes |
|---|---|---|
| `prompt` (completions) | required | A string (BOS is prepended if the tokenizer has one) or a list of token ids (sent as is) |
| `add_bos` (completions) | absent | With a string prompt: `false` skips the BOS |
| `messages` (chat) | required | `[{role, content}]` (plus whatever the template reads: `tool_calls`, `tool_call_id`, …). Rendered with the model's own chat template when the file has one (the `chat_template` section, from `tokenizer_config.json` or GGUF `tokenizer.chat_template`); the template writes BOS and the role markers, and special tokens in its output become their ids. A file without a template gets the old prompt: `"<role>: <content>\n"` per message plus `"assistant:"`, with BOS prepended |
| `tools`, `tool_choice`, `parallel_tool_calls` (chat) | none, `auto`, false | Tool calling; see [Chat models](#chat-models-end-of-turn-and-tool-calls) |
| `add_generation_prompt`, `chat_template_kwargs` (chat) | true, none | Passed to the template: whether the prompt ends with the assistant header, and an object of extra template variables (`{"enable_thinking": false}`) |
| `model` | | Any name serves the base weights (clients send their own). `"<model>:<adapter>"`, or the name of a loaded adapter, selects that LoRA adapter; `<model>:` with an unknown adapter is a `404` |
| `adapter` (alias `lora`) | none | LoRA adapter by name; wins over `model` |
| `max_tokens`, `max_completion_tokens` | 16 | The newer name wins when both are sent. 1 to 1048576 |
| `n` | 1 | Choices per request (at most `--max-n`). Choice `i` samples with seed `seed + i`; `usage.completion_tokens` sums them, `prompt_tokens` counts the prompt once |
| `temperature` | 0 | Greedy by default |
| `top_k`, `top_p`, `min_p`, `typical_p` | 0, 1, 0, 1 | |
| `repetition_penalty`, `presence_penalty`, `frequency_penalty` | 1, 0, 0 | |
| `mirostat_tau`, `mirostat_eta` | 0, 0.1 | |
| `seed` | 0 | The same seed and request give the same tokens |
| `logit_bias` | none | `{"<token id>": bias}`, at most 1024 entries, bias in [-100, 100] added to the token's logit before sampling. -100 and 100 are absolute (ban / force) |
| `logprobs`, `top_logprobs` | off | Chat: `"logprobs": true` and `"top_logprobs": 0..20` → `choices[i].logprobs.content[]` of `{token, logprob, bytes, top_logprobs[]}`. Completions: `"logprobs": 0..20` → `{tokens, token_logprobs, top_logprobs, text_offset}` (a token that is not valid UTF-8 alone is named `bytes:\xNN...`). Values are the model's own log-softmax, before the sampler chain, bias and grammar |
| `stop` | none | A string, a token id, or a list of up to 64 of either. A string cuts the output before its first match (matched on the model's bytes); a token id ends the sequence like EOS |
| `stop_token_ids` | none | List of token ids, same effect as ids in `stop` |
| `priority` | 0 | -1000 to 1000. Higher is admitted to the scheduler first; FIFO within a priority |
| `grammar` | none | GBNF text (see [inference.md](inference.md)) |
| `response_format` | none | `{"type":"json_object"}`: any JSON object. `{"type":"json_schema","json_schema":{"schema":{...}}}`: that schema. Both become a grammar (overriding `grammar`); output is compact JSON. `{"type":"text"}`: no constraint |
| `stream` | false | Server-Sent Events |
| `stream_options` | none | `{"include_usage": true}`: every chunk carries `"usage": null` and one extra event with `"choices": []` and the usage precedes `[DONE]` |

Field values are checked: a wrong type or a value out of range is a `400` whose `param` names the field. A prompt that does not leave room for one new token in the context is refused with code `context_length_exceeded`.

`finish_reason` is `stop` (EOS, an end-of-turn or stop token, a stop string, grammar complete or a token the grammar rejected), `length`, `tool_calls`, or `cancelled`. `usage` has `prompt_tokens`, `completion_tokens`, `total_tokens`. Text is sanitised to valid UTF-8 (invalid bytes become U+FFFD). EOS, end-of-turn and stop tokens are not included in `text` or `tokens` and are not counted.

Streaming sends `data: {...}` events: completion chunks carry `choices[i].text`, `token` (the last token id of the event) and `tokens` (every id since the previous event); chat starts each choice with `delta: {"role":"assistant","content":""}`, then `delta.content` and `delta.tool_calls`. A token that ends inside a UTF-8 sequence, and text that could be the beginning of a stop string, is held back until it is decided. Each choice ends with an event carrying its `finish_reason`; the last of them also carries `usage`. Then `data: [DONE]`. A streaming client that disconnects cancels its request.

### Errors
Errors have OpenAI's shape: `{"error":{"message","type","param","code"}}` (`param` and `code` are `null` when there is nothing to name).

| Status | `type` | When |
|---|---|---|
| `400` | `invalid_request_error` | Malformed HTTP or JSON, a bad field (`param`), a prompt longer than the context (`code` `context_length_exceeded`), an embedding request to a generative model and the reverse |
| `401` | `authentication_error` | API keys are configured and the request has none or a wrong one (`code` `invalid_api_key`, `WWW-Authenticate: Bearer`) |
| `404` | `not_found_error` | Unknown path; unknown model id or adapter (`code` `model_not_found`) |
| `405` | `invalid_request_error` | Wrong method (`Allow` header) |
| `408`, `413`, `417`, `431`, `501`, `505` | `invalid_request_error` / `server_error` | Request not received in time; body or headers over the limit; an `Expect` other than `100-continue`; a transfer coding other than `chunked`; not HTTP/1.x. The connection is closed |
| `429` | `rate_limit_error` | Rate limit (`code` `rate_limit_exceeded`, `Retry-After`) or token quota (`code` `insufficient_quota`) |
| `500` | `server_error` | The engine failed on this request. Other requests are not affected |
| `503` | `overloaded_error` | Queue full (`--max-queue`), out of memory, shutting down, or the connection limit (`Retry-After: 1`) |

A request that fails inside the scheduler fails alone: the others keep running. In a stream, where the status line is already sent, the error is one more event (`data: {"error":{...}}`) followed by `data: [DONE]`.

### HTTP behaviour
- **Keep-alive and pipelining.** Responses say `Connection: keep-alive` or `close`. Requests on a connection are read and answered strictly in order; bytes of a following request that arrived early are kept for it.
- **Request bodies**: `Content-Length` or `Transfer-Encoding: chunked` (chunk extensions and trailers are read and dropped). A request with both, with two different `Content-Length`s, with blanks in a header name or with a folded header line is refused (`400`): these are the request-smuggling shapes.
- **`Expect: 100-continue`** is answered with `HTTP/1.1 100 Continue` before the body is read.
- **Limits**: header bytes (`--max-header-bytes`, which also bounds chunk-size lines and trailers), body bytes (`--max-body-bytes`), time to receive a request (`--request-timeout`), idle time (`--keep-alive-timeout`), a client that stops reading a response (30 s), connections (`--max-connections`), `n` (`--max-n`), 2048 embedding inputs, 128 tools, 64 stop entries, 1024 `logit_bias` entries.
- **CORS**: `--cors`. The default `*` lets any web page call the server from a browser; with API keys the page still needs a key.

### Shutdown
SIGINT or SIGTERM stops the listener at once. Requests in flight, streams included, run to their end; new requests on kept connections get `503`, idle connections are closed, `/health` answers `503 draining`. When no connection is left, or after `--shutdown-timeout` seconds, the process exits (status 0 if everything drained, 1 otherwise).

### Chat models: end of turn and tool calls
**End of turn.** A chat request stops when the model emits any of: the EOS token; an id in the manifest's `eos_token_ids` (recorded at import from Hugging Face `generation_config.json` `eos_token_id`, or a `config.json` list, and from GGUF `tokenizer.ggml.eos_token_id` / `eot_token_id` / `eom_token_id`); or one of the known end-of-turn markers that is a token of the model **and** appears in its chat template: `<|im_end|>`, `<|eot_id|>`, `<|eom_id|>`, `<end_of_turn>`, `<|end|>`, `<|endoftext|>`, `<|end_of_text|>`, `</s>`, `<|eot|>`, `<|END_OF_TURN_TOKEN|>`, `<｜end▁of▁sentence｜>`, `<|return|>`. `/v1/completions` uses the first two only. The token is not part of the reply. Files imported before `eos_token_ids` existed still get the template-derived markers. `forge run --chat` stops the same way.

**Tool calls.** With `tools` (OpenAI's list of `{"type":"function","function":{"name","description","parameters"}}`, at most 128) the template is given the tools and the reply is searched for calls in the syntax the model was trained on, which is read off the chat template (`tool_format` in `/v1/models`):

| `tool_format` | Detected by | Call syntax |
|---|---|---|
| `hermes` | `<tool_call>` in the template (Hermes, Qwen, ChatML tool models) | `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`, repeated for several calls |
| `mistral` | `[TOOL_CALLS]` | `[TOOL_CALLS][{"name": ..., "arguments": {...}}, ...]` |
| `llama3` | `<\|python_tag\|>`, `<\|eom_id\|>`, or `<\|start_header_id\|>` with tools | `<\|python_tag\|>{"name": ..., "parameters": {...}}`, or that JSON object as the whole reply |
| `generic` | anything else, and files without a template | `{"name": ..., "arguments": {...}}` as the whole reply |

| `tool_choice` | Effect |
|---|---|
| `"auto"` (default with `tools`) | Output is not constrained. Calls found in it are returned; a bare JSON object (`llama3`, `generic`) only counts when it names a declared function |
| `"none"` | The template is not given the tools and the reply is not parsed |
| `"required"` | A grammar built from the tools' JSON Schemas forces one call (compact JSON) in the model's syntax; with `"parallel_tool_calls": true` one or more (`hermes`, `mistral`) |
| `{"type":"function","function":{"name":"f"}}` | The same grammar for that function only |

A reply with calls has `finish_reason` `tool_calls` and `message.tool_calls: [{"id":"call_...","type":"function","function":{"name","arguments"}}]`, where `arguments` is the JSON text the model wrote (`"{}"` when it gave none); `message.content` is the text before the first call, or `null`. A call whose JSON never completes (cut off by `max_tokens`) is returned as content. In a stream each call arrives as a first `delta.tool_calls` entry with `index`, `id`, `type` and `function.name` (and empty `arguments`), then entries with `function.arguments` fragments as the model writes them; text that might be the start of a call is held back until that is decided. Forced calls cannot be combined with `grammar` or a JSON `response_format` (`400`). Incoming assistant messages may carry `tool_calls` whose `arguments` are a JSON string (as clients send them); they are parsed to objects before the template sees them.

Limits: the `parameters` schemas go through the JSON Schema subset of [inference.md](inference.md) (no `pattern`, `format`, numeric ranges, `additionalProperties`); Mistral's newer `[TOOL_CALLS]name[ARGS]{...}` form and Llama's `<function=name>` form are not parsed; arguments are not validated against the schema in `auto` mode. Checked on the tiny random golden models (forced calls, and calls written by a grammar in `auto` mode), not on real tool-calling checkpoints.

### LoRA adapters
`--lora NAME=FILE` (repeatable) loads adapters at startup: an `.fpk` or a `.safetensors` holding `<weight>.lora_a` (`[r, in]`) and `<weight>.lora_b` (`[out, r]`) tensors under FORGE's weight names (`forge inspect model.fpk --tensors`), with `lora_alpha` in the `.fpk` manifest or as a one-element tensor (default 1). A request selects one with `"model": "<model>:<NAME>"` or `"adapter": "<NAME>"`; requests with different adapters, and with none, run in the same batch. Each adapter has its own merged copies of the targeted matrices and its own plans; the prefix cache (and its spill files) is keyed by adapter, so KV rows are only shared between requests that name the same one. There is no converter from PEFT directories (their tensor names are the source framework's), and adapters are not loaded or unloaded while serving.

### Authentication and TLS
- **API keys.** With `--api-key` or `--api-key-file`, every request except `GET /health` and `OPTIONS` must send `Authorization: Bearer <key>`; otherwise `401`. Keys are compared as BLAKE2b digests in constant time against every configured key. A key-file line may set `rpm=N` (requests per minute; `--rate-limit` is the default for keys without it) and `quota=N` (total prompt + completion tokens the key may spend while this process runs). Only generation and embedding requests are counted. Counters are in memory: a restart resets them.
- **TLS.** `--tls-cert` and `--tls-key` switch the listener to HTTPS, using the OpenSSL code of the RPC layer (`graph/rpc_secure.hpp`; TLS 1.2 minimum). `--tls-client-ca` adds mutual TLS. A failed handshake closes the connection without an HTTP reply. A build without OpenSSL (`-DFORGE_TLS=OFF`, or OpenSSL not found) serves plain HTTP only and refuses these flags at startup with a message saying so. There is no certificate reload (restart, or `--watch`), no OCSP stapling and no per-certificate authorisation.

Tests: `tests/serve_auth_test.py` (ctest `serve_auth`).


### Metrics
Every series has `# HELP` and `# TYPE` lines. Scheduler: `forge_steps`, `forge_completed`, `forge_failed_total`, `forge_prefix_hits`, `forge_prefix_tokens_reused`, `forge_peak_active`, `forge_peak_kv_bytes`, `forge_queue_waiting`, `forge_queue_active`, `forge_admission_deferred_total`, `forge_rejected_total`, `forge_kv_evictions_total`, `forge_kv_spills_total`, `forge_kv_reloads_total`, `forge_kv_reload_failures_total`, `forge_kv_spilled`. HTTP: `forge_http_connections_total`, `forge_http_connections_active`, `forge_http_connections_refused_total`, `forge_http_request_errors_total`, `forge_http_responses_total{code="..."}`, `forge_auth_failures_total`, `forge_rate_limited_total`, `forge_draining`. Work: `forge_prompt_tokens_total`, `forge_completion_tokens_total`, `forge_embedding_inputs_total`, `forge_tool_calls_total`. Then the memory governor's `forge_mem_*` series ([memory.md](memory.md)). With API keys `/metrics` needs one.

## Examples

```sh
forge-serve golden/tiny_f32.fpk --port 18093 &
curl -s localhost:18093/health
curl -s localhost:18093/v1/completions -d '{"prompt":"Hello","max_tokens":4}'
curl -s localhost:18093/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":3}'
curl -s localhost:18093/v1/completions -d '{"prompt":"x","max_tokens":8,"response_format":{"type":"json_schema","json_schema":{"schema":{"enum":["yes","no"]}}}}'
curl -sN localhost:18093/v1/completions -d '{"prompt":"Hello","max_tokens":8,"stream":true}'
curl -s localhost:18093/metrics

# chat with a forced tool call, an API key and TLS
forge-serve model.fpk --api-key-file keys.txt --tls-cert cert.pem --tls-key key.pem --lora fr=fr.safetensors &
curl -s --cacert cert.pem -H 'Authorization: Bearer sk-...' https://localhost:8080/v1/chat/completions -d '{
  "model": "model.fpk:fr", "messages": [{"role":"user","content":"Weather in Paris?"}], "max_tokens": 128,
  "tools": [{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice": "required"}'
```

The `serve_http` ctest (`tests/serve_test.py`) checks concurrent completions against `forge run`, streaming, chat, grammar, JSON Schema, stop strings, and a tiny `--mem-limit` with `--max-queue`. `--kv-spill` itself is covered at the scheduler level in `tests/test_memory.cpp`, not over HTTP. The `serve_api` ctest (`tests/serve_api_test.py`) covers the request fields above, end-of-turn stops, tool calling in the `generic` and `hermes` formats (plain and streamed), keep-alive, pipelining, chunked bodies, `Expect: 100-continue`, the size and time limits, CORS, `/health`, `/v1/models`, `/metrics`, the error shapes, `--lora` and SIGTERM draining, on the tiny golden model and a ChatML variant of it built by the test. `fuzz_serve_http` drives the request reader (keep-alive, chunked bodies) from mutated byte streams; `tests/test_chat_tools.cpp` covers the tool-call grammar and parser for all four formats.

### systemd unit

    [Service]
    Type=notify
    ExecStart=/usr/local/bin/forge-serve /srv/model.fpk --port 8080 --watch
    Environment=FORGE_HUGEPAGES=1

## Platform notes and limits
- Without `--api-key*` there is **no authentication**, and without `--tls-*` no encryption; the default bind is 127.0.0.1. API keys over plain HTTP are readable on the wire.
- One model per process. HTTP/1.0 and HTTP/1.1 only (no HTTP/2), no response compression, no `HEAD`.
- Tool calling, end-of-turn detection and LoRA serving have been run on the tiny random golden models only (see the sections above for what each test does); no real chat or tool-calling checkpoint was served here.
- `logprobs` report the model's distribution, not the distribution after sampling settings. `logit_bias` of exactly ±100 is absolute; other values are added to unbounded logits, so their strength depends on the model.
- Rate limits and quotas are per process and in memory. They are not shared between forge-serve instances behind `forge-serve proxy`.
- IPv6 listen addresses (`--host ::1`) are accepted by the code but were not run: the test host has no IPv6.
- The graceful shutdown was tested on Linux (SIGTERM). On Windows the console control handler path of `os::on_signal` is used and is untested here.
- `--ask-model` is a `forge script` flag, not a forge-serve flag (see [scripting.md](scripting.md)).
- Readiness notification was tested with an abstract `NOTIFY_SOCKET` in `test_os_ext`, not under a real systemd. Running under the Windows SCM or launchd is untested.
