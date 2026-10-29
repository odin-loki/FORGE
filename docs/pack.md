# pack: the .fpk model container

## Concepts
A `.fpk` file holds everything needed to run a model:
- a JSON manifest
- a blueprint (the computation graph as text)
- a tokenizer
- a tensor index and the tensor data

Every section and every tensor carries a BLAKE3 hash. Files can also be signed with Ed25519 and have their tensor data encrypted with XChaCha20-Poly1305. The byte layout is described in [format.md](format.md).

## API (`pack/include/forge/pack/fpk.hpp`, `sign.hpp`, `tokenizer.hpp`, `blueprint.hpp`)
- `pack::writer` methods:
  - `set_manifest(json)` and `set_section(name, bytes)`.
  - `add_tensor(name, shape, span<const float>, codec="f32")` quantises the data on write. A codec that cannot represent the shape falls back to another codec (for example q4_k becomes f16).
  - `add_raw(name, shape, codec, bytes)`.
  - `write(path)`.
- `pack::reader(path, key = nullopt)` methods:
  - Metadata: `manifest()`, `sections()`, `tensors()`, `entry(name)` (gives `.codec`, `.shape`, `.offset`) and `has_section`.
  - Data: `section_text`, `load(name)` (returns a tensor decoded to f32) and `file_size`.
  - `verify()` returns the names that fail their hash check, so an empty result means the file is intact.
  - Encryption: `encrypted()` and `locked()`.
- `pack::alignment` is 64.
- `pack::bpe_tokenizer` provides `encode(text)`, `decode(ids)` and `eos()`.
- `sign.hpp` signs a digest built from the section table using the `forge-fpk-v1` scheme.

## Examples
`docs/examples/pack_roundtrip.cpp` runs as the ctest `doc_example_pack_roundtrip`:

```cpp
pack::writer w;
w.set_manifest(json::object{{"name", "demo"}});
w.add_tensor("w", {16, 16}, a, "f32");
w.write(path);
pack::reader r(path);
bool ok = r.verify().empty();
```

To sign, verify and encrypt from the CLI (these exact commands were run against `golden/tiny_f32.fpk`):

```sh
forge keygen -o dk                                # writes dk.key (secret) and dk.pub
forge sign model.fpk --key dk.key -o signed.fpk
forge verify signed.fpk --pubkey dk.pub           # prints VERIFIED; exits 1 on failure
forge encrypt signed.fpk -o sealed.fpk --new-key k.mkey
FORGE_MODEL_KEY_FILE=k.mkey forge inspect sealed.fpk
```

For strict loading, set `FORGE_REQUIRE_SIGNED=1` and/or `FORGE_TRUSTED_KEYS=<hex>[,<hex>]`. You can give the key as `FORGE_MODEL_KEY=<hex>` instead of `FORGE_MODEL_KEY_FILE`.

## Importing
`forge import <hf_dir|model.gguf> -o out.fpk [--codec C] [--template llama]` converts a model to `.fpk`.

The supported codecs are: f32, f16, bf16, fp8_e4m3, fp8_e5m2, q8_0, q4_k, q4_g32, q8_g32, mxfp8, mxfp6, mxfp4, mxint8, nvfp4 and tq2.

The `onnx_import` test covers ONNX import.

## Platform notes and limits
- Readers memory-map the file. Set `FORGE_HUGEPAGES=1` to request huge pages for mappings of 2 MiB or more.
- Section metadata stays readable in encrypted files. This is why inspecting and verifying section hashes work without the key.
