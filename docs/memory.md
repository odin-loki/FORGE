# memory: budgets, eviction and backpressure

FORGE's memory manager stops work from being OOM-killed and lets it run on small machines and in containers. It keeps one view of how much memory the process may use, who is using it, and what can be dropped. When memory runs short, FORGE reacts in this order:

1. Drop caches.
2. Hold back new work.
3. Wait for running work to free memory.
4. Fail the one request that cannot fit with a clear error.

The kernel's OOM killer should never be what stops the process.

The code is `core/include/forge/core/memory.hpp` (namespace `forge::mem`), and the tests are in `tests/test_memory.cpp` and `tests/serve_test.py`.

## Goals
- **Never exceed a known limit silently.** Every large consumer is charged against a budget. If something does not fit, the caller gets `errc::out_of_memory` (a `forge_exception`), or HTTP 503 from `forge-serve`, instead of a `SIGKILL`.
- **Detect the real limit.** The limit is the smallest of physical RAM, a cgroup v1 or v2 limit (Docker, Kubernetes, systemd `MemoryMax=`) and `RLIMIT_AS`. Using physical RAM alone is the usual reason containers get OOM-killed.
- **Prefer slowing down to dying.** When memory is tight, run fewer requests at a time, shrink caches, and queue or reject new requests.
- **Zero cost when unused.** Everything is header-only. The budget is opt-in per consumer, and with no limit every check is a single compare.
- **Observable.** `/metrics` and `forge info` show the budget, how much is in use (in total and per tag), the peak, and counts of evictions, denials and waits.

## Budget model
A **budget** is a named pool with a byte limit:
- `host` is the process heap and anonymous memory.
- Device pools such as `cuda:0` and `vulkan:0` are created on demand with `governor::pool(name)`. The GPU backends charge them for every device allocation (see [GPU memory](#gpu-memory)).

Each budget has a **hard limit** and a **soft watermark**. The soft watermark is `FORGE_MEM_SOFT × limit`, 0.9 by default.

| Concept | API | Meaning |
|---------|-----|---------|
| Lease | `budget::reserve(n, tag, timeout)` / `try_reserve(n, tag)` | RAII hold on `n` bytes. It can be moved and resized with `lease::resize`, and is released by its destructor |
| External usage | `budget::set_usage(tag, n)` | For things that size themselves, such as the KV cache of running requests. These are counted but never blocked |
| Cache | `budget::add_cache(name, size_fn, evict_fn, priority)` | Something that can be dropped. It is evicted with the lowest priority first, then least recently used first (`touch_cache` updates its position) |
| Pressure | `budget::under_pressure(extra)` | True when `used + cached + extra` is over the soft watermark. Admission control uses this |

How the `host` limit is chosen, highest priority first:
1. `--mem-limit` on `forge-serve` (or `set_limit` in code).
2. `FORGE_MEM_LIMIT`, which accepts `6G`, `512M`, `1.5GiB`, a plain byte count, or `75%` of the detected limit. `0`, `off` or `none` turns the limit off.
3. Otherwise, `FORGE_MEM_FRACTION` (default 0.85) of the detected limit.

The detected limit is the smallest of:
- physical RAM;
- the tightest `memory.max` (cgroup v2) or `memory.limit_in_bytes` (cgroup v1) found walking up from `/proc/self/cgroup`;
- `RLIMIT_AS`.

Device pools take their limit from `FORGE_MEM_DEVICE_LIMIT`. With no limit set, they only keep accounts.

### Reserving memory
`try_reserve` works like this:
- If the lease fits, it is granted.
- If it does not fit, the budget asks caches to free the shortfall. If the lease fits after that, it is granted.
- Otherwise the call returns an empty lease and adds one to `denials`.

`reserve` does the same, then:
- If `timeout` is more than zero, it waits on a condition variable until other leases are released, evicting again each time it wakes.
- If the request is larger than the whole budget, or the timeout runs out, it throws `errc::out_of_memory`. The message names the pool, the tag, the size requested, current use, and the setting to change.

## Allocator design
FORGE already has allocators, and the memory manager works alongside them rather than replacing them:
- **`mem::arena`** (`support/include/forge/mem/arena.hpp`): bump-pointer blocks with scope marks, for short-lived scratch memory. Alignment is any power of two.
- **`mem::tlsf`** (`support/include/forge/mem/tlsf.hpp`): an O(1) general-purpose allocator over a fixed region, for device heaps and suballocation.
- **`graph::memplan`**: works out tensor lifetimes at plan time and places all intermediates in one slab, so a graph's peak memory is known before it runs.
- **Paged KV** (`session_options.page_rows`): the KV cache grows a page at a time instead of reserving the full context up front.

The new piece is **`mem::budget_resource`**, a `std::pmr::memory_resource` that charges a budget for every allocation and throws `errc::out_of_memory` past the limit. Any `std::pmr` container, or an arena's upstream, can be accounted this way without changes to its code.

Alignment is passed straight through to the upstream resource, which by default is `new_delete_resource`, so over-aligned requests work.

## Pressure handling
| Mechanism | Status | Notes |
|-----------|--------|-------|
| OS pressure signals | **done** | `governor::watch_pressure()` subscribes to `os::memory_pressure` (Linux PSI, Android `onTrimMemory`, the Apple dispatch source, the Windows resource notification, or a 1 s poll). At *moderate* it trims every pool to its soft watermark. At *critical* it drops every cache. `FORGE_MEM_PRESSURE=0` turns this off |
| Model weights | **done (existing)** | `.fpk` tensors are memory-mapped (`core/mapped.hpp`). The kernel pages them in on demand and can drop clean pages under pressure, so weights are **not** charged to the budget. `FORGE_HUGEPAGES=1` is still available for large mappings |
| Prefix KV cache eviction | **done** | The batch scheduler registers its prefix cache with the budget (priority 10, LRU: a hit moves an entry to the front). The pressure response and admission control both evict from it. With a spill store, evicted prefixes go to disk instead of being dropped (next row) |
| KV of running requests | **done** | The live KV size is reported each step with `set_usage("kv", …)` |
| Admission control | **done** | Before starting a queued request, the scheduler projects its cost from the largest KV size seen per request so far. If that would cross the soft watermark, it first drops prefix-cache entries, then leaves the request queued. At least one request always runs, so the queue keeps draining. Output is identical, only less parallel. Counted in `deferred` |
| Backpressure in `forge-serve` | **done** | `--max-queue N` rejects new requests with **503 + `Retry-After: 1`** when N are already waiting. An `out_of_memory` error or `std::bad_alloc` while handling a request returns 503 instead of crashing the server |
| Spill to disk | **done** | `mem::spill_store` writes evicted blobs to numbered files in `FORGE_MEM_SPILL_DIR` (default `<tmp>/forge-spill-<pid>`), capped at `FORGE_MEM_SPILL_MAX`, and deletes them on destruction. With `batch_options::spill` set (`forge-serve --kv-spill`), a prefix the scheduler evicts from memory is saved there (`session::save_prefix`, up to `spill_entries` prefixes, default 64) instead of dropped. A prompt that shares a spilled prefix reads it back (`session::load_prefix`), moves it into the in-memory cache and copies its rows like any other hit, so the output is identical to recomputing it. If the store is full the prefix is dropped as before |
| GPU memory | **done** | Every device allocation of the CUDA, HIP, OpenCL, Level Zero, Vulkan and Metal backends is charged to the device pool, and resident tensors are evicted least recently used first before an allocation would exceed the pool's limit or fail (see [GPU memory](#gpu-memory)) |

## GPU memory
The GPU backends (`gpu/device_memory.hpp`) charge the pool `<backend>:<ordinal>` (`cuda:0`, `hip:0`, `opencl:0`, `level_zero:0`, `vulkan:0`, `metal:0`) for each device allocation, with tags `resident` (tensors kept on the device by the residency protocol, and freed buffers the CUDA/HIP runtime pools for reuse), `weights` (encoded weights for the device quantised kernels), `scratch` (per-op temporaries and staging buffers) and `raw` (`device_alloc`). Freeing an allocation releases its charge. `FORGE_MEM_DEVICE_LIMIT` (or `budget::set_limit` on the pool) sets the limit. With no limit the pool only keeps accounts, which `/metrics` shows like any other pool.

Each runtime registers its resident buffers with its pool as a cache. Before an allocation would cross the limit, and when the driver itself reports out of memory, the runtime makes room in this order:
1. It frees its pooled free buffers (CUDA/HIP).
2. It evicts resident tensors and encoded weights, least recently used first. A tensor whose newest data is only on the device is copied back to the host first. The next op that needs it uploads it again, which is what `FORGE_GPU_NO_RESIDENCY` does for every buffer, applied on demand to just enough of them.

Buffers bound by the op in progress are never evicted. If nothing more can be evicted the allocation fails with `errc::out_of_memory`, naming the pool, the size and the setting to change. The runtime stays usable: a later op with room runs normally. `budget::evict` and the pressure response reach the same callback from other threads. A runtime that is busy is skipped rather than waited for, so two runtimes sharing a pool cannot deadlock. Metal has unified memory and no residency: each op charges its buffers while it runs, and nothing can be evicted.

Eviction moves device addresses, so the CUDA runtime turns off whole-run replay until runs are confirmed again. Nothing is evicted while a stream capture (CUDA/HIP) or a recorded graph (OpenCL, Level Zero) is being built. `residency()` counts `evictions` and `evicted_bytes` per runtime, and `memory_pool()` / `charged_bytes()` return the pool and the bytes the runtime holds in it.

## Observability
- `forge info` prints physical RAM, available RAM, any cgroup limit and current cgroup usage, any `RLIMIT_AS`, the `host` budget and its soft watermark, and where the limit came from.
- `forge-serve` startup prints the memory budget, or `off`.
- `GET /metrics` adds, for each pool:
  - `forge_mem_limit_bytes`, `forge_mem_used_bytes`, `forge_mem_peak_bytes` and `forge_mem_cache_bytes`;
  - the counters `forge_mem_reservations_total`, `forge_mem_denials_total`, `forge_mem_waits_total`, `forge_mem_evictions_total` and `forge_mem_evicted_bytes_total`;
  - `forge_mem_used_bytes_by_tag{pool,tag}`.

  It also adds `forge_mem_pressure_events_total`, plus the scheduler gauges and counters `forge_queue_waiting`, `forge_queue_active`, `forge_admission_deferred_total`, `forge_rejected_total`, `forge_kv_evictions_total`, `forge_kv_spills_total`, `forge_kv_reloads_total` and `forge_kv_spilled` (prefixes currently on disk).

## Configuration summary
| Setting | Default | Effect |
|---------|---------|--------|
| `FORGE_MEM_LIMIT` | unset | Host budget: a size, a percentage of the detected limit, or `off` |
| `FORGE_MEM_FRACTION` | `0.85` | Share of the detected limit used when `FORGE_MEM_LIMIT` is unset |
| `FORGE_MEM_SOFT` | `0.9` | Soft watermark as a share of the limit |
| `FORGE_MEM_DEVICE_LIMIT` | unset | Limit for every device pool. Resident GPU tensors are evicted to stay under it |
| `FORGE_MEM_PRESSURE` | on | Set to `0` to ignore OS pressure signals |
| `FORGE_MEM_SPILL_DIR` | `<tmp>/forge-spill-<pid>` | Spill directory |
| `FORGE_MEM_SPILL_MAX` | unlimited | Maximum bytes on disk in the spill store |
| `forge-serve --mem-limit` | env | Same values as `FORGE_MEM_LIMIT`, and takes priority over it |
| `forge-serve --max-queue N` | `0` (no cap) | Reject with 503 once N requests are waiting |
| `forge-serve --kv-spill` | off | Spill evicted prefix KV to the spill directory and reload it on a hit |

Example: to run in a 4 GiB container, the defaults are enough because the cgroup limit is detected. To set the limit and queue cap explicitly:

    forge-serve model.fpk --mem-limit 3G --max-queue 32 --max-active 4

## Platform notes
- **Linux and Android:** cgroup v1 and v2 limits are detected. PSI triggers deliver pressure events. Available memory comes from `MemAvailable`.
- **macOS and iOS:** there is no cgroup. The limit is physical RAM or `RLIMIT_AS`. `os::available_memory()` is a rough estimate (a quarter of RAM), so rely on the dispatch pressure source and a set `FORGE_MEM_LIMIT`. On iOS, `os_proc_available_memory` would be more accurate (see the phased plan).
- **Windows:** the limit is physical RAM, so set `FORGE_MEM_LIMIT` when running under a Job Object limit. Pressure comes from `CreateMemoryResourceNotification`.
- **FreeBSD:** the limit is physical RAM or `RLIMIT_AS`, and pressure is detected by polling.
- Cache `size_fn` callbacks run while the budget's lock is held, so they must not call back into the budget. `evict_fn` callbacks run without the lock and may.

## Phased plan
1. **Done:** budgets, leases, caches, the governor, cgroup and rlimit detection, a pmr resource, the spill store, pressure hookup, KV accounting, admission control and prefix-cache eviction in the scheduler, plus `forge-serve --mem-limit`, `--max-queue`, 503 handling, `/metrics` and `forge info`.
2. **Done:** the GPU backends charge their device pools for every allocation and evict device-resident tensors, least recently used first (`FORGE_GPU_NO_RESIDENCY` behaviour on demand), before a device allocation would exceed the budget or fail. Tests: `test_gpu_memory_{cuda,hip,opencl,ze}_mock`.
3. **Done:** the scheduler spills cached prefix KV to the `spill_store` instead of dropping it, and reloads it when a prefix hits (`forge-serve --kv-spill`). Tests: `kv_prefix_blob_roundtrip_identical_logits` and `scheduler_spills_and_reloads_prefix_kv` in `tests/test_memory.cpp`.
4. **Later:** a per-request KV reservation sized from `(prompt + max_tokens) × kv bytes per token`, so a request that can never fit is rejected at submission. Budget-aware `forge run` and C ABI entry points. `os_proc_available_memory` on iOS and Job Object limits on Windows.
