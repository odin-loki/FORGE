# FORGE documentation

Every guide follows the same shape: **concepts**, **API or command reference**, **examples**, and **platform notes and limits**. Anything that has not been run is labelled as such.

## Getting started
1. [README](../README.md): what FORGE is, quick start, layout
2. [building.md](building.md): presets, CMake options, cross builds, wine, Android emulator, qemu
3. [testing.md](testing.md): test kinds, labels, how to run them
4. [cli.md](cli.md): every `forge` command and flag
5. [configuration.md](configuration.md): all environment variables in one table

## Library layers, bottom up
6. [os.md](os.md): `forge::os`, the portability layer and its 16 extension modules
7. [lane.md](lane.md): CPU feature detection and SIMD kernel dispatch
8. [grid.md](grid.md): devices, graphs, parallel primitives, GPU backends
9. [flow.md](flow.md): graph IR, passes, plans, codecs, autodiff, RPC devices
10. [pack.md](pack.md): the `.fpk` container, signing, encryption, importing
11. [format.md](format.md): `.fpk` byte layout
12. [mind.md](mind.md): models, sessions, samplers, generation

## Serving and embedding
13. [serve.md](serve.md): `forge-serve` HTTP server
14. [balance.md](balance.md): load-balancing pool and `forge-serve proxy`
15. [capi.md](capi.md): the stable C ABI (`forge.h`)
16. [bindings.md](bindings.md): Python, Kotlin and Swift

## Reference and project
17. [reference.md](reference.md): op table, GPU kernel list and C ABI, **generated** by `tools/gen_docs.py` (do not edit by hand)
18. [../spec/README.md](../spec/README.md): op specs and code generators
19. [STATUS.md](STATUS.md): what has been tested where
20. [RELEASING.md](RELEASING.md): release procedure
21. [../CONTRIBUTING.md](../CONTRIBUTING.md): how to change code and docs
22. [../FORGE_PLAN.md](../FORGE_PLAN.md): the original design plan (a roadmap, not a description of current behaviour)

## Compiled examples
Each file in `docs/examples/` is built and run by ctest as `doc_example_<name>`, so the snippets in these guides cannot silently drift from the API:

```sh
(cd build/dev && ctest -R doc_example)
```

| Example | Guide |
|---------|-------|
| `lane_dispatch.cpp` | lane.md |
| `grid_graph.cpp` | grid.md |
| `flow_graph.cpp` | flow.md |
| `pack_roundtrip.cpp` | pack.md |
| `mind_generate.cpp` | mind.md |
| `os_features.cpp` | os.md |
| `balance_pool.cpp` | balance.md |
| `capi_generate.c` | capi.md |
