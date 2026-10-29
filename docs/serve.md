# serve: the forge-serve HTTP server

## Concepts
`forge-serve` serves one model over HTTP with an OpenAI-compatible API. Requests share a continuous-batching scheduler (`mind/batch.hpp`), which also reuses KV-cache entries for prompts with a shared prefix.

The same binary also runs as a load-balancing proxy in front of several servers (`forge-serve proxy`, see [balance.md](balance.md)).

If you pass a Hugging Face directory or a `.gguf` file, it is imported into a temporary `.fpk` first. `--codec` picks the weight codec for that import.

## API reference

### Command line
Defaults are taken from `apps/forge_serve.cpp`:

```
forge-serve <model.fpk|hf_dir|model.gguf> [--host 127.0.0.1] [--port 8080] [--max-active 8] [--ctx 0] [--codec f32]
            [--watch] [--no-pin] [--numa-node N]
forge-serve proxy ...                      # see balance.md
```

`--ctx 0` means use the model's maximum context.

### Endpoints

| Method and path | Body or response |
|-----------------|------------------|
| `GET /health` | Returns `{"status":"ok"}` |
| `GET /v1/models` | Returns `{"object":"list","data":[{"id":<file name>,...}]}` |
| `POST /v1/completions` | Takes `prompt` (a string or a list of token ids) and the sampling fields below. The response holds `choices[0].text`, `choices[0].tokens` and `usage` |
| `POST /v1/chat/completions` | Takes `messages: [{role, content}]` and the same sampling fields. The response holds `choices[0].message` |
| `GET /metrics` | Plain-text scheduler counters, for example `forge_steps`, `forge_completed`, `forge_prefix_hits` and `forge_peak_active` |

The sampling fields are:
- `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `typical_p`
- `repetition_penalty`, `presence_penalty`, `frequency_penalty`
- `mirostat_tau`, `mirostat_eta`, `seed`
- `stop` (a list of strings), `grammar` (GBNF), `response_format` (JSON schema) and `stream`

### Flags and environment variables

| Flag / env | Effect |
|---|---|
| `--watch` | Hot reload: a `file_watcher` on the model; on change forge-serve sends `RELOADING=1`, closes the listener and re-execs itself with the same arguments. POSIX only; on Windows it logs that a service restart is needed. |
| `--no-pin` | On hybrid CPUs forge-serve pins itself to P-cores by default (threads it spawns inherit the mask; QoS class on Apple). This turns that off. |
| `--numa-node N` | Bind to NUMA node N before loading, so weights and KV are first-touched on that node. |
| `FORGE_HUGEPAGES=1` | `madvise(MADV_HUGEPAGE)` on weight mappings ≥2 MiB (advisory; file-backed THP needs kernel support). |
| memory pressure | Always on: moderate pressure trims the prefix KV cache to 1 entry, critical to 0 (`batch_scheduler::request_kv_eviction`), and logs to the system log. |
| service mode | After the socket is listening forge-serve calls `service::ready()` (systemd `READY=1`; no-op without `NOTIFY_SOCKET`). Use `Type=notify` in the unit. |

## Examples
These commands were run against `golden/tiny_f32.fpk` on port 18093:

```sh
forge-serve golden/tiny_f32.fpk --port 18093 &
curl -s localhost:18093/health                                     # {"status":"ok"}
curl -s localhost:18093/v1/completions -d '{"prompt":"Hello","max_tokens":4}'
curl -s localhost:18093/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":3}'
curl -s localhost:18093/metrics
```

The `serve_http` ctest (`tests/serve_test.py`) exercises the same endpoints.

### systemd unit

    [Service]
    Type=notify
    ExecStart=/usr/local/bin/forge-serve /srv/model.fpk --port 8080 --watch
    Environment=FORGE_HUGEPAGES=1

## Platform notes and limits
- **`--watch`:** re-executes the server on POSIX only. On Windows it logs that the service needs a restart instead.
- **`--watch` and `--numa-node 0`:** checked live on Linux.
- **Service mode:** `Type=notify` readiness was tested with an abstract `NOTIFY_SOCKET` inside `test_os_ext`, not under a real systemd.
- **Untested:** running under the Windows SCM and launchd.
