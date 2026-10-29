# FORGE documentation

Read the guides in this order. Each guide covers the same four things in turn: **concepts**, **API reference**, **examples** and **platform notes and limits**.

| # | Guide | What it covers |
|---|-------|----------------|
| 1 | [README](../README.md) | What FORGE is, quick start, repository layout |
| 2 | [lane.md](lane.md) | CPU feature detection and SIMD kernel dispatch |
| 3 | [grid.md](grid.md) | Devices, graphs and parallel primitives |
| 3b | [mind.md](mind.md) | Models, sessions, samplers and generation |
| 4 | [os.md](os.md) | `forge::os` portability layer (16 extension features) |
| 5 | [balance.md](balance.md) | Load-balancing strategies |
| 6 | [serve.md](serve.md) | `forge-serve` HTTP server and its flags |
| 7 | [bindings.md](bindings.md) | Python, Kotlin and Swift bindings, plus the C ABI (`mind/include/forge.h`) |
| 8 | [reference.md](reference.md) | Generated API reference (`tools/gen_docs.py`) |
| 9 | [STATUS.md](STATUS.md) | What has been tested, on which platform |
| 10 | [RELEASING.md](RELEASING.md) | Release procedure |

The flow, pack, C ABI, file format, building and testing guides are **not written yet**. Until they are, use [reference.md](reference.md) and the header comments.

## Compiled examples

The code in `docs/examples/` is built and run by ctest as `doc_example_*`, so it always matches the current API:

```sh
ninja -C build/dev doc_example_lane_dispatch doc_example_mind_generate
(cd build/dev && ctest -R doc_example)
```
