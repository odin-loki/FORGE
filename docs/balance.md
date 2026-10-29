# `forge::balance` — load balancer and proxy

`balance/include/forge/balance/balance.hpp`: `pool` of backends with `strategy` = `round_robin`, `weighted` (smooth WRR),
`least_connections`, `least_latency` (EWMA), `consistent_hash` (vnodes + bounded load), `power_of_two` (P2C) and
`prefix_affinity` (KV-cache affinity with load spill). Health checks (`health_checker`), a per-backend circuit breaker
(closed/open/half-open), retries (`pool::call` with `retry_policy`) and a backpressure queue.

`proxy.hpp`: L4 (TCP splice) or L7 (HTTP) proxy used by `forge-serve proxy`:

    forge-serve proxy --backend 127.0.0.1:8081 --backend 127.0.0.1:8082 [--mode l4|l7] \
        [--strategy p2c|rr|wrr|lc|ewma|hash|kv] [--route /v1/embeddings=host:port,host:port] \
        [--host 127.0.0.1] [--port 8080] [--retries 3] [--health-ms 2000]

Strategy names on the command line: `rr`, `wrr`, `lc`, `ewma`, `hash` (alias `ring`), `p2c` (default), `kv` (alias `prefix`).
L7 mode parses each HTTP request and picks the backend from the affinity key (header `X-Session-Id` by default, else
the client address); `--route PREFIX=host:port,...` sends matching paths to their own pool. L4 mode forwards raw TCP.
Failed backends are retried on another one up to `--retries` times; `--health-ms` sets the TCP health-probe period.

Tests: `tests/test_balance.cpp` (Linux, wine, Android emulator).
