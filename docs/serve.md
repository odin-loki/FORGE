# `forge-serve`

    forge-serve <model.fpk|hf_dir|model.gguf> [--host 127.0.0.1] [--port 8080] [--max-active 8] [--ctx N] [--codec f32]
                [--watch] [--no-pin] [--numa-node N]
    forge-serve proxy ...            # see docs/balance.md

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions` (OpenAI-style bodies;
sampling fields `max_tokens temperature top_p top_k min_p typical_p repetition_penalty presence_penalty frequency_penalty
seed stop grammar (GBNF) response_format (json_schema) mirostat_tau mirostat_eta stream`), `GET /metrics`.

    curl -s localhost:8080/v1/completions -d '{"prompt":"Hello","max_tokens":16}'

HF directories and GGUF files are imported into a temporary `.fpk` first (`--codec` picks the weight codec).

| Flag / env | Effect |
|---|---|
| `--watch` | Hot reload: a `file_watcher` on the model; on change forge-serve sends `RELOADING=1`, closes the listener and re-execs itself with the same arguments. POSIX only; on Windows it logs that a service restart is needed. |
| `--no-pin` | On hybrid CPUs forge-serve pins itself to P-cores by default (threads it spawns inherit the mask; QoS class on Apple). This turns that off. |
| `--numa-node N` | Bind to NUMA node N before loading, so weights and KV are first-touched on that node. |
| `FORGE_HUGEPAGES=1` | `madvise(MADV_HUGEPAGE)` on weight mappings ≥2 MiB (advisory; file-backed THP needs kernel support). |
| memory pressure | Always on: moderate pressure trims the prefix KV cache to 1 entry, critical to 0 (`batch_scheduler::request_kv_eviction`), and logs to the system log. |
| service mode | After the socket is listening forge-serve calls `service::ready()` (systemd `READY=1`; no-op without `NOTIFY_SOCKET`). Use `Type=notify` in the unit. |

Example systemd unit:

    [Service]
    Type=notify
    ExecStart=/usr/local/bin/forge-serve /srv/model.fpk --port 8080 --watch
    Environment=FORGE_HUGEPAGES=1
