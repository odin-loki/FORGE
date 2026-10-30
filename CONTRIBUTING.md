# Contributing

1. Build and test (see [docs/building.md](docs/building.md) and [docs/testing.md](docs/testing.md)):
   ```sh
   cmake --preset dev && cmake --build build/dev
   ctest --test-dir build/dev -LE "slow|emu" --output-on-failure
   ```
2. Ops are declared in `spec/ops/*.fspec` and generated code is not edited by hand; see [spec/README.md](spec/README.md).
3. `docs/reference.md` is generated: after changing public headers or the spec run `python3 tools/gen_docs.py` (the `docs_up_to_date` test fails otherwise).
4. Each guide in `docs/` has examples in `docs/examples/` that build and run as `doc_example_*` tests. Keep them compiling when you change an API.
5. Code is C++23. Keep changes covered by a test under `tests/` (hardware-dependent tests should exit 77 to skip).
6. Licensing: FORGE is dual-licensed (see [LICENSE](LICENSE) and [COMMERCIAL-LICENCE.md](COMMERCIAL-LICENCE.md)). By contributing you agree your contribution can be distributed under both.
