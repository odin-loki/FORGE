# Configuration: environment variables

FORGE has no config file. Programs take command-line flags ([cli.md](cli.md), [serve.md](serve.md)) and read the environment variables below. Each was found by searching the sources for `getenv`.

## Runtime

| Variable | Read by | Effect |
|----------|---------|--------|
| `FORGE_DEVICE` | `mind::model` (so `forge run`, `forge-serve`, C ABI, bindings) | Execution device: `cpu` (default), `vulkan`, `cuda[:N]`, `hip[:N]`, `opencl[:N]`, `ze[:N]` / `level_zero[:N]`, `metal[:N]`, `rpc:host:port`, or a pipeline `pipe:<dev>[*weight],<dev>...` (e.g. `pipe:vulkan*3,cpu`). An unavailable device logs a warning and falls back to CPU |
| `FORGE_LANE_TARGET` | lane | Force a kernel table: `scalar`, `avx2`, `avx512` (x86), `neon`, `sve` (aarch64), `rvv` (riscv64). Ignored if not usable on this CPU |
| `FORGE_THREADS` | task pool | Worker thread count (default: hardware concurrency) |
| `FORGE_LOG` | logging | Threshold by first letter: `trace`, `debug`, `info`, `warn` (default), `error` |
| `FORGE_HUGEPAGES=1` | memory-mapped files | Request huge pages (`madvise`) for mappings of 2 MiB or more |
| `FORGE_KV_CODEC` | sessions | KV cache codec when `session_options.kv_codec` is empty: `f32` (default), `int8`, `fp8` |
| `FORGE_KV_OFFLOAD=1` | sessions | Keep KV on the host and drop device copies after each attention |
| `FORGE_GPU_NO_RESIDENCY` | Vulkan lowerings | If set, tensors are not kept resident on the device between ops |
| `FORGE_SVE_KEEP_VL=1` | lane (aarch64) | Do not change the SVE vector length at start-up |

## Models and security

| Variable | Effect |
|----------|--------|
| `FORGE_MODEL_KEY` / `FORGE_MODEL_KEY_FILE` | Key (hex) or key file for encrypted `.fpk` files |
| `FORGE_REQUIRE_SIGNED=1` | Refuse to load unsigned models |
| `FORGE_TRUSTED_KEYS=<hex>[,<hex>]` | Only accept signatures from these Ed25519 public keys |
| `FORGE_CACHE` | Where `forge run` caches imported models (default `~/.cache/forge`) |
| `FORGE_TEMPLATES` | Directory of blueprint templates for importers (default: `import/templates` in the source tree the binary was built from) |

## Caches and GPU debugging

| Variable | Effect |
|----------|--------|
| `FORGE_CACHE_DIR` | Vulkan pipeline cache directory (else `$XDG_CACHE_HOME/forge`, else `~/.cache/forge`) |
| `FORGE_VK_PIPELINE_CACHE=0` | Disable the Vulkan pipeline cache |
| `FORGE_VK_DEBUG` | Log why an op was not offloaded to Vulkan |
| `FORGE_VK_SSBO_ALIGN=N` | Emulate a stricter storage-buffer offset alignment (testing) |
| `FORGE_OPENCL_TYPE` | OpenCL devices to enumerate: `gpu`, `cpu` or `all` (default; GPUs listed first) |

## OS layer

| Variable | Effect |
|----------|--------|
| `FORGE_AIO=threads` | Force the thread-pool backend for async file I/O instead of io_uring |
| `FORGE_KEYSTORE=file` | Force the file keystore |
| `FORGE_KEYSTORE_DIR` | File keystore directory |
| `NOTIFY_SOCKET`, `WATCHDOG_USEC` | Standard systemd variables, honoured by `os::service` |

## Test-only
`FORGE_OS_EXT_CHILD`, `FORGE_OS_T`, `FORGE_TEST_H264`, `FORGE_ZE_TEST_IMAGE`, `FORGE_OPENCL_TEST_ALIGN`, `MOCK_DEVICES`, `MOCK_ALIGN` are used by the test suites and mock drivers. **Do not export `FORGE_OS_EXT_CHILD` in a shell**: it switches `test_os_ext` into a child-process mode. `FORGE_TEST_TIMEOUT` is read by `tests/android_emulator.sh`; `FORGE_TEST_FPK` by the Android and Swift binding tests; `FORGE_LIB` by the Python binding.
