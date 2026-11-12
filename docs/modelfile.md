# modelfile: the .fpk model container and importers

## Concepts
A `.fpk` file holds everything needed to run a model:
- a JSON **manifest**,
- a **blueprint** (the computation graph as text, `modelfile/blueprint.hpp`),
- a **tokenizer** (Hugging Face style `tokenizer.json`),
- a **tensor index** and the **tensor data**, each tensor in one of the weight codecs of `graph/codecs.hpp`.

Every section and every tensor has a BLAKE3 hash. Files can be **signed** (Ed25519) and their tensor data **encrypted** (XChaCha20-Poly1305), both with the vendored Monocypher. Readers memory-map the file. The byte layout is in [format.md](format.md).

Importers turn Hugging Face checkpoints, GGUF files and ONNX graphs into `.fpk`. Language models get their blueprint from a template in `import/templates/` (`llama`, `qwen2`, `qwen3`, `qwen3_5`, `qwen3_moe`, `qwen3_5_moe`, `mixtral`, `deepseek_v2`). MoE templates use the sparse `moe_ffn` op over stacked expert tensors (`layer.N.experts.w_gate|w_up|w_down`, `[n_experts, rows, k]`), indexed in the `experts` section ([format.md](format.md)); HF Mixtral, DeepSeek-V2 and Qwen3-MoE experts are stacked at import.

## API
### `modelfile::writer` (`fpk.hpp`)

| Member | Notes |
|--------|-------|
| `set_manifest(json)`, `set_section(name, bytes)` | |
| `add_tensor(name, shape, span<const float>, codec = "f32")` | Encodes on write with any codec in `graph::codecs::all()`. If a block codec's block size does not divide the element count, the tensor is stored as `f16` |
| `add_raw(name, shape, codec, bytes)` | Stores already-encoded blocks (e.g. GGUF Q4_K) without re-quantising; the byte count must match the codec |
| `write(path)` | Writes to a temporary file and renames it over `path` |

`modelfile::stream_writer(path)` has the same `set_*`/`add_*` calls but writes each tensor to disk as it is added, so peak memory is one tensor. Call `finish()` at the end.

### `modelfile::reader(path, key = nullopt)` (`fpk.hpp`)

| Member | Notes |
|--------|-------|
| `manifest()`, `sections()`, `tensors()`, `entry(name)`, `has_section(name)`, `section_text(name)`, `section_bytes(name)`, `file_size()` | Metadata. `entry` gives `.codec`, `.shape`, `.offset`, `.bytes`, `.blake3` |
| `load(name)` | The tensor decoded to f32 |
| `verify()` | Names of sections and tensors whose hash fails (empty = intact). Tensor hashes are skipped while `locked()` |
| `encrypted()`, `locked()` | `locked()`: encrypted and no key from the constructor, `FORGE_MODEL_KEY` or `FORGE_MODEL_KEY_FILE` |

`modelfile::alignment` is 64; `modelfile::format_version` is 1.

### Others

| Header | Contents |
|--------|----------|
| `tokenizer.hpp` | `bpe_tokenizer`: `encode(text)`, `decode(ids)`, `decode_one(id)`, `token_id(str)`, `vocab_size()`, `bos()`, `eos()`, `add_bos()` (whether a prompt starts with BOS) |
| `sign.hpp` | `generate_keypair()`, `load_secret_key(path)`, `load_public_key(path)`, `sign_file(in, out, key)`, `verify_signature(reader, trusted_keys)` → `signature_report` (`signed_`, `digest_ok`, `trusted`, `valid_keys`, `message`). With no trusted keys, any valid signature counts (integrity only, not provenance) |
| `bundle.hpp` | Ed25519 over a directory: `sign_bundle(root, key)` writes `FORGE-SIGNATURE.json` (path, size, BLAKE3 per file); `verify_bundle(root, trusted)` re-hashes and reports modified, missing and extra files. Also detached `sign_bytes` / `verify_bytes` |
| `blueprint.hpp` | `parse_blueprint(text)`: the blueprint language (params, state slots, `repeat` loops, any spec op by name, composites such as `linear`, `embed`, `heads`, user-defined `op name(args) { ... }`, fixed-size state ops `state[i].conv(...)` / `state[i].delta(...)`). `build_graph(bp, options)` builds the graph and calls `check_opset(bp)`; `blueprint_weights(bp)` lists the `W.<path>` names the graph reads; `opset_min` / `opset_max` |
| `entropy.hpp` | OS randomness for keys and nonces |

## Importers (`modelfile/include/forge/import/`)

| Source | Function, header | Architectures | Weights |
|--------|------------------|---------------|---------|
| Hugging Face directory (`config.json`, `*.safetensors`, `tokenizer.json`) | `import_hf`, `hf.hpp` (+ `safetensors.hpp`) | `llama` template: Llama, Mistral. `qwen2`: Qwen2 / Qwen2.5 (the llama graph plus biases on the Q, K and V projections). `qwen3`: Qwen3 (QK-norm). `qwen3_5`: Qwen3.5 dense text models (Gated DeltaNet + gated full attention, partial RoPE). `mixtral`: Mixtral MoE. `deepseek_v2`: DeepSeek-V2/V3 (absorbed MLA) | Encoded with `--codec`; mixed precision with `q4_k_m` / `q5_k_m` |
| GGUF v2/v3 | `import_gguf`, `gguf.hpp` | `llama`, `mistral` → `llama`; `qwen2` → `qwen2` (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`); `qwen3`; `qwen35` → `qwen3_5`; `qwen3moe` → `qwen3_moe`; `qwen35moe` → `qwen3_5_moe` (untested) | Copied verbatim: F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K. `--codec` is not applied |
| ONNX | `import_onnx`, `onnx.hpp` | Static-shape inference graphs (MLPs, classifiers, encoder blocks) | Initializers become weights |

### What an import checks and records
- **Every source tensor is accounted for** (Hugging Face and GGUF). A tensor the template does not consume fails the import with a list of names and shapes, because dropping weights silently gives a model that runs and answers wrongly. `--allow-unused` (`import_options::allow_unused`) imports anyway: Hugging Face leftovers are left out, GGUF leftovers are kept as `extra.<name>`, and both are listed in `import_report::unused`. `--ignore-tensor` (`import_options::ignore_tensors`) names leftovers that are known to be harmless; `position_ids`, `masked_bias` and `inv_freq` buffers and the vision / multi-token-prediction towers of Qwen3.5 checkpoints are always skipped.
- **Every tensor the template reads exists.** A missing source tensor fails with its name. After the last tensor the importer compares `blueprint_weights(bp)` with what it wrote, in both directions, so a `--template` that does not fit the checkpoint is refused.
- **RoPE frequency scaling** is read from `config.json` (`rope_scaling`, or `rope_parameters` in transformers 5; types `linear`, `dynamic`, `yarn`, `llama3`) and from GGUF (`<arch>.rope.scaling.type` `linear` / `yarn` with `.factor`, `.original_context_length`, `.attn_factor`, `.yarn_attn_factor`, `.yarn_log_multiplier`, `.yarn_ext_factor`, `.yarn_beta_fast`, `.yarn_beta_slow`; the llama3 factors that llama.cpp's converter stores as the `rope_freqs.weight` tensor). It becomes attrs of the `rope` op and, for YaRN's attention factor, `attention`'s `scale`. LongRoPE and unknown types are refused. Templates `llama`, `qwen2`, `qwen3`, `qwen3_moe` and `mixtral` take scaling; `deepseek_v2`, `qwen3_5` and `qwen3_5_moe` refuse a model that asks for it.
- **The licence** is recorded: Hugging Face `README.md` front matter (`license`, `license_name`, `license_link`) and the text of `LICENSE*` / `LICENCE*` / `COPYING*` / `NOTICE*` files (up to 1 MiB each); GGUF `general.license`, `general.license.name`, `general.license.link`; ONNX `metadata_props` (`license`, `license_name`, `license_url`) and a `doc_string` that mentions a licence. It goes into the manifest and the `license` section ([format.md](format.md)), `forge inspect` shows it, and signing and encrypting keep it. A source that states no licence gives a file without these fields; the importer does not guess.

### Opsets
A blueprint declares `opset N` (default 1). This engine runs `modelfile::opset_min` to `opset_max` (1 to 2). `build_graph` and `inference::model` refuse any other opset with an error that names it and the supported range; an op the engine does not know is reported with its name, line and the blueprint's opset. Opset 2 adds the `rope` attrs `scaling`, `factor`, `orig_ctx`, `low_freq`, `high_freq`, `beta_fast`, `beta_slow`, `truncate` and `freq_factors` (`ops/include/forge/ops/rope.hpp`). The importers write opset 2 only for models that scale their RoPE frequencies; every other import stays opset 1 with the same graph as before. Builds older than this check ignore the opset and unknown attrs, so an opset-2 file gives wrong output there.

Template placeholders are `{{key}}`, or `{{key|default}}` for optional pieces (`{{rope_args|}}`, `{{attn_args|}}`, `{{opset|1}}`).

Importer limits that raise an error:
- HF `deepseek_v2`: no YaRN rope scaling, greedy top-k routing with softmax scoring only.
- HF `qwen3_5` and GGUF `qwen35`: no MoE variants, no full-width RoPE, `hidden_act` must be `silu`, and full-attention layers must repeat every `full_attention_interval` layers.
- GGUF `qwen35`: V heads stored in tiled order (`ssm.v_head_reordered`) are put back in grouped order. Row-wise tensors are permuted as raw blocks. The DeltaNet out projection is decoded, permuted and re-encoded, as q8_0 when the source was quantised.
- ONNX ops: MatMul Gemm Add Sub Mul Div Pow Max Min Relu Sigmoid Tanh Exp Log Sqrt Neg Abs Erf Gelu Softplus Reciprocal Softmax LayerNormalization Transpose Reshape Flatten Identity Constant ReduceMean ReduceSum. One symbolic dimension maps to `seq`. Any other op is rejected by name.

Templates are read from `FORGE_TEMPLATES` if set, else from `import/templates/` in the source tree.

### Mixed precision (HF only, `quant_recipe.hpp`)
`--codec q4_k_m` or `q5_k_m` chooses a type per tensor. `--recipe llama` is a port of llama.cpp's `llama_tensor_get_type()` rules. `--recipe forge` (default) applies the same rules, then puts Gated DeltaNet in/out projections and `attn_output` at q6_k and the head and token embedding at q8_0 (bf16 with `--bf16-head`). Norms, conv weights, A/dt and the a/b gates stay f32. Overrides: `--output-tensor-type`, `--token-embedding-type`, `--tensor-type REGEX=TYPE[,...]` (first match wins), `--pure`.

### CLI
```sh
forge import <hf_dir | model.gguf | model.onnx> -o out.fpk [--codec C] [--template NAME] [--emb-codec C] [--allow-unused] [--ignore-tensor NAME[,NAME...]]
```
Codecs: `f32 f16 bf16 fp8_e4m3 fp8_e5m2 q8_0 q4_k q5_k q6_k q4_g32 q8_g32 mxfp8 mxfp6 mxfp4 mxint8 nvfp4 tq2`. With a quantised codec, the token embedding defaults to q8_0 (`--emb-codec` overrides). `forge run` also accepts an HF directory or `.gguf` directly and caches the import in `~/.cache/forge` (`--allow-unused` applies to that import too).

The importers are tested on small random models built by the test scripts (`verify/golden/tiny_llama.py`, `tiny_moe.py`, `tiny_qwen2.py` (HF and GGUF), `tiny_rope.py` (RoPE scaling variants, HF and GGUF), `tiny_qwen3.py` (HF and GGUF), `tiny_qwen3_5.py`, `quant_inplace.py`, `tests/onnx_test.py`), checked against float64 NumPy or transformers references. They have not been checked against full-size published checkpoints in the test suite. The RoPE scaling references are NumPy transcriptions of `transformers/modeling_rope_utils.py`; they have not been run against transformers or llama.cpp themselves.

## Examples
`docs/examples/modelfile_roundtrip.cpp` (ctest `doc_example_modelfile_roundtrip`):

```cpp
modelfile::writer w;
w.set_manifest(json::object{{"name", "demo"}});
w.add_tensor("w", {16, 16}, a, "f32");
w.add_tensor("wq", {256}, a, "q8_0");      // quantised on write
w.write(path);
modelfile::reader r(path);
bool ok = r.verify().empty();              // empty = every hash matches
auto t = r.load("w");                      // decoded to f32
```

Signing and encryption from the CLI:

```sh
forge keygen -o dk                              # dk.key (secret seed), dk.pub
forge sign model.fpk --key dk.key -o signed.fpk
forge verify signed.fpk --pubkey dk.pub         # prints "signed.fpk: VERIFIED"; exit 1 on failure
forge encrypt signed.fpk -o sealed.fpk --new-key k.mkey
FORGE_MODEL_KEY_FILE=k.mkey forge inspect sealed.fpk --verify
```

Strict loading in inference sessions: `FORGE_REQUIRE_SIGNED=1` refuses unsigned files; `FORGE_TRUSTED_KEYS=<hex>[,<hex>]` also requires a signature from one of those keys.

### Detached signatures

Files that are not `.fpk` (native plugins, scripts, anything else) are signed with a detached signature next to them, `<file>.sig` (FSS INT-2, `security/include/forge/security/integrity.hpp`). The same keys as for models (`forge keygen`):

```sh
forge sign libmyplugin.so --key dk.key          # -> libmyplugin.so.sig (the plugin is not modified)
forge sign job.fs --key dk.key                  # -> job.fs.sig
forge verify job.fs --pubkey dk.pub             # job.fs: VERIFIED; exit 1 on failure
```

`<file>.sig` is one JSON object:

```json
{"format":"forge-detached-sig","version":1,"alg":"ed25519","hash":"blake3-256",
 "file":"job.fs","size":31,"blake3":"<64 hex>",
 "signatures":[{"key_id":"<8 hex>","public_key":"<64 hex>","sig":"<128 hex>"}]}
```

Each signature is Ed25519 (RFC 8032, Monocypher) over the 32-byte digest

```
BLAKE3( "forge-detached-sig-v1" || 0x00 || u64le(len(name)) || name || u64le(size) || BLAKE3(file bytes) )
```

- The base name is signed, so a signed file cannot be swapped for another file signed by the same key (putting `b.so` and `b.so.sig` in place of `a.so` fails: "signature is for 'b.so', not 'a.so'"). The cost: a renamed file must be signed again (or signed with `--name`). Names compare exactly on Linux / BSD and ASCII-case-insensitively on Windows and macOS.
- Verification recomputes the BLAKE3 digest and size from the bytes being loaded and the name from the path being loaded; the `file`, `size` and `blake3` fields are informative.
- Several signers: signing again with another key appends to `signatures` (only while the name and content match); signing again with the same key replaces its entry.
- `key_id` is the first 4 bytes of BLAKE3(public key), as for `.fpk`.
- A `.sig` larger than 64 KiB is rejected.

Under a security policy, `integrity.plugins` and `integrity.scripts` decide whether these signatures are required ([security-api.md](security-api.md#integration)).

## Platform notes and limits
- `FORGE_HUGEPAGES=1` asks for transparent huge pages on mappings of 2 MiB or more.
- In encrypted files only `data` is sealed. The manifest, blueprint, tokenizer, tensor index and section table stay readable, which is why `inspect`, section-hash and signature checks work without the key.
- Signing is cross-checked against pyca/cryptography (`modelfile_signing` ctest).
