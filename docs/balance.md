# `forge::balance` — load balancer and proxy

`balance/include/forge/balance/balance.hpp`: `pool` of backends with `strategy` = `round_robin`, `weighted` (smooth WRR),
`least_connections`, `least_latency` (EWMA), `consistent_hash` (vnodes + bounded load), `power_of_two` (P2C) and
`prefix_affinity` (KV-cache affinity with load spill). Health checks (`health_checker`), a per-backend circuit breaker
(closed/open/half-open), retries (`pool::call` with `retry_policy`) and a backpressure queue.

`proxy.hpp`: L4 (TCP splice) or L7 (HTTP) proxy used by `forge-serve proxy`:

    forge-serve proxy --backend 127.0.0.1:8081 --backend 127.0.0.1:8082 [--mode l4|l7] \
        [--strategy p2c|rr|wrr|lc|ewma|hash|kv] [--route /v1/embeddings=host:port,host:port] \
        [--host 127.0.0.1] [--port 8080] [--retries 3] [--health-ms 2000]

Tests: `tests/test_balance.cpp`.
