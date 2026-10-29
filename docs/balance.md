# balance: load balancer and proxy

## Concepts
A `pool` holds a set of backends and picks one per request using a **strategy**. Every pick returns a `lease`. You report the outcome on the lease (success, failure or latency), and the pool uses those reports for health tracking:
- per-backend circuit breakers (closed, open, half-open)
- outlier ejection
- EWMA latency
- in-flight counts

On top of the pool there are:
- active health checks (`health_checker`)
- retries on a *different* backend, with exponential backoff and jitter
- per-backend concurrency limits, with a bounded, deadline-aware wait queue (backpressure)

The pool is header-only and thread-safe. `proxy.hpp` builds the L4 (TCP) and L7 (HTTP) proxies used by `forge-serve proxy` on top of it.

## API reference (`balance/include/forge/balance/balance.hpp`, `proxy.hpp`)

### Strategies

| `strategy` | CLI name | Behaviour |
|------------|----------|-----------|
| `round_robin` | `rr` | Takes each backend in turn |
| `weighted` | `wrr` | Smooth weighted round robin |
| `least_connections` | `lc` | Fewest in-flight requests |
| `least_latency` | `ewma` | Lowest EWMA latency. Stale estimates decay toward the pool mean |
| `consistent_hash` | `hash`, `ring` | Virtual nodes with bounded load |
| `power_of_two` | `p2c` (default) | Picks two backends at random and uses the less loaded one |
| `prefix_affinity` | `kv`, `prefix` | Sends a shared prompt prefix to the backend that holds its KV cache, and spills to others when that backend is overloaded |

`parse_strategy(name)` returns `std::optional<strategy>`.

### Types
- `backend_spec` fields: `id`, `address`, `weight` (default 1.0), `max_inflight` (default 0, meaning unlimited) and `user`.
- `config` fields and their defaults:
  - Strategy: `strat = power_of_two`.
  - Consistent hash: `vnodes = 160`, `hash_load_factor = 1.25`.
  - EWMA latency: `ewma_alpha = 0.3`, `ewma_decay_ns = 10 s`.
  - Circuit breaker: `failure_threshold = 5`, `failure_rate_threshold = 0.5` over `window = 20` with `min_requests = 10`, `open_ns = 5 s`, `half_open_probes = 1`.
  - Health checks: `healthy_after = 2`, `unhealthy_after = 3`.
  - Prefix affinity: `prefix_block = 64`, `prefix_table = 65536`, `prefix_spill = 1.5`.
  - Backpressure: `queue_limit = 1024`.
- `pool(config)` methods:
  - `add(backend_spec)`.
  - `acquire(pick_options = {})` returns a `lease`. `pick_options` has `key`, `prefix`, `exclude` and `wait_ns`.
  - `call(pick_options, fn(const backend_spec&) -> bool, retry_policy)`.
- `lease` methods: `backend()`, `success()`, `success_with_latency(ns)` and `failure`. A lease that is dropped without a report counts as a failure.
- `retry_policy` fields and their defaults: `attempts = 3`, `base_backoff_ns = 2 ms`, `max_backoff_ns = 200 ms`, `wait_ns = 0`.

### Proxy CLI
Defaults are taken from `apps/forge_serve.cpp`:

```
forge-serve proxy --backend HOST:PORT [--backend ...] [--mode l7|l4 (default l7)] [--strategy p2c]
    [--route /prefix=host:port,host:port] [--host 127.0.0.1] [--port 8080] [--retries 3] [--health-ms 2000]
```

- **L7** parses each HTTP request and uses the `X-Session-Id` header as the affinity key, or the client address if the header is absent. `--route` sends matching path prefixes to their own pool.
- **L4** forwards raw TCP.

## Example
`docs/examples/balance_pool.cpp` runs as the ctest `doc_example_balance_pool`:

```cpp
pool p(config{});                              // P2C
p.add({"gpu0", "10.0.0.5:7000", 1.0});
lease l = p.acquire();
l.success_with_latency(1500);
bool ok = p.call({}, [&](const backend_spec& b) { return send_to(b); }, {3, 1000, 10000});
```

## Platform notes and limits
`tests/test_balance.cpp` passes on Linux, under wine and on the Android emulator. For FreeBSD and macOS it is cross-linked only.
